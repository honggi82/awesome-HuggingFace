# arXiv:2412.17726v2[cs.CV]28Mar2025

## VidTwin: Video VAE with Decoupled Structure and Dynamics

Yuchi Wang1,2*, Junliang Guo2†, Xinyi Xie2,3, Tianyu He2, Xu Sun1†, Jiang Bian2 1Peking University 2Microsoft Research Asia 3CUHK (SZ) wangyuchi@stu.pku.edu.cn, 120040057@link.cuhk.edu.cn, xusun@pku.edu.cn {junliangguo, tianyuhe, jiang.bian}@microsoft.com

https://aka.ms/vidtwin

Abstract

Recent advancements in video autoencoders (Video AEs) have significantly improved the quality and efficiency of video generation. In this paper, we propose a novel and compact video autoencoder, VidTwin, that decouples video into two distinct latent spaces: Structure latent vectors, which capture overall content and global movement, and Dynamics latent vectors, which represent fine-grained details and rapid movements. Specifically, our approach leverages an Encoder-Decoder backbone, augmented with two submodules for extracting these latent spaces, respectively. The first submodule employs a Q-Former to extract low-frequency motion trends, followed by downsampling blocks to remove redundant content details. The second averages the latent vectors along the spatial dimension to capture rapid motion. Extensive experiments show that VidTwin achieves a high compression rate of 0.20% with high reconstruction quality (PSNR of 28.14 on the MCL-JCV dataset), and performs efficiently and effectively in downstream generative tasks. Moreover, our model demonstrates explainability and scalability, paving the way for future research in video latent representation and generation.

### 1. Introduction

The latent diffusion model has recently revolutionized the popular text-to-image field, with representative models such as the Stable Diffusion series [9, 31, 32, 34]. In this paradigm, the image autoencoder plays a critical role by encoding the image into a compact latent space, thereby alleviating modeling complexity and improving training efficiency of the diffusion model. Recently, there has been growing interest in adapting this paradigm for video latent representation and downstream video generation tasks [3, 4,

* Work done during an internship at Microsoft Research Asia. † Junliang Guo and Xu Sun are corresponding authors.

###### Orig. Recon. S. Recon. D. Recon.

[Figure 1]

[Figure 2]

Figure 1. An example illustrating the Structure and Dynamics latents. We select two frames, t1 and t2, and show the original and reconstructed video frames, labeled Orig. and Recon., respectively. S. Recon. and D. Recon. refer to the reconstructed frames decoded using only the corresponding Structure or Dynamics latents. The Structure latent captures the main semantic content and overall motion trends, while the Dynamics latent encodes local details and rapid movements.

13, 17, 29, 58]. However, due to the extra temporal consistency of videos compared to static images, simultaneously modeling visual content and temporal dependencies into a latent space presents a challenging problem [58].

Upon reviewing previous works that explore the conversion of video into latent representations using autoencoders [17, 20, 29, 56, 58], we identify two main design philosophies. First, classical approaches, represent each frame (or a group of frames) as latent vectors or tokens of uniform size [46, 55, 56, 61]. This method is straightforward but overlooks the redundancy between frames. Video inherently exhibits continuity, indicating that adjacent frames typically differ only slightly in details, suggesting significant potential for further compression. The second emerging approach addresses this problem by dividing the representation into two types, i.e., a single or a few content frame(s) along with several motion latent vectors [20, 51, 58]. However, these decoupling methods over-

simplify the dynamic nature of video content, leading to unsatisfactory generation results, such as blurred frames [58].

In this paper, we propose a novel approach that encodes videos into two distinct latent spaces: Structure Latent, which represents the global content and movement, and Dynamics Latent, which captures fine-grained details and rapid motions. For instance, in the video of tightening a screw shown in Fig. 1, the main semantic content, such as the table and screw, corresponds to Structure Latent, while fine-grained details, such as color, texture, and rapid local movements—like the screw’s downward motion and rotation—are captured by Dynamics Latent. These components are combined to form the reconstructed video. To achieve this, we introduce the VidTwin model, designed to effectively learn these interdependent latent representations. Our approach addresses the shortcomings of previous methods that often neglect dynamic content, enabling a video autoencoder capable of achieving high compression without compromising reconstruction quality.

Specifically, we utilize the Spatial-Temporal Transformer [2] as the backbone of our video autoencoder and introduce two submodules to extract Structure Latent and Dynamics Latent, respectively. For the former, leveraging the powerful information extraction capabilities of the QFormer [25] architecture, we apply it solely to the temporal dimension to extract the low-frequency changing motion trends independently of spatial location. We then further downsample the latent in the spatial dimension to remove redundant details while retaining the most important object information. For the latter, since rapid motion information can be represented in low dimensions, we first downsample the latent vectors obtained from the encoder spatially, and then we average these vectors along both the height and width dimensions to further reduce their dimensionality. Additionally, we design a mechanism to adapt the obtained latents for diffusion models by patchifying the two latent vectors and concatenating them as the training target for the diffusion model.

Through experiments, we demonstrate that our model offers several advantages: (1) High compression rate: The decoupling design and more compact latent representation of VidTwin yield a superior compression rate, achieving around a 500× compression factor while maintaining high reconstruction quality. This significantly alleviates memory and computational burdens for downstream models, which are often challenged by the high dimensionality of video data. (2) Effectiveness for downstream tasks: Video autoencoders are commonly used within generative models, which require a smooth latent space. We validate this with the UCF-101 [38] dataset, where our model performs comparably to some well-established models, demonstrating its adaptability to generative tasks. (3) Explainability and Scalability: As shown in Fig. 1, we carefully design the

latent space to ensure meaningful and explainable representations, and preliminary experiments also suggest that the model exhibits scalability, both of which provide opportunities for further research and improvements.

In conclusion, the main contributions of our work are summarized as follows:

- (1) Building on the philosophy of decoupling video rep-

resentation into structure and dynamics, we propose a novel video codec model, VidTwin, which demonstrates effective decoupling with a compact design.

- (2) Our VidTwin achieves a high compression rate and

strong reconstruction ability, and has been verified for its applicability and efficiency in generative models.

- (3) We highlight the importance of video latent represen-

tation in current research trends and hope our VidTwin can inspire or facilitate further related research.

### 2. Related Works

##### 2.1. Visual Autoencoder

With the rapid advancement of visual generation, there has been increasing attention on visual latent representation techniques. Given that the dominant methods for generation are now diffusion models and autoregressive approaches, two primary types of corresponding representations have emerged: (1) Continuous latent vectors: Stable Diffusion [32] was one of the pioneering works to utilize a Variational Autoencoder (VAE) [22] for image encoding, with a diffusion model then modeling this latent space. This approach has since inspired numerous subsequent works [5, 6, 9, 26, 27, 31, 62]. For video data, several studies have incorporated 3D convolutions [1, 4, 7, 17, 36, 39, 53, 61] or spatio-temporal attention mechanisms [16, 19, 29, 52] into the backbone, resulting in a latent space specifically designed for video data, which facilitates more effective video generation. (2) Discrete tokens: In a separate line of work, influenced by the success of language modeling in the NLP community, several models have explored discrete representations of visual information. VQ-VAE [41] introduced a codebook into the VAE [22] training procedure to discretize the representation, while VQ-GAN [8] incorporated adversarial training to improve the quality of generated images. Later models further refined their architectures, such as replacing CNNs with Transformers [54] or improving quantization methods [28, 56]. For video data, some approaches treat frames as independent images for tokenization [11, 45, 60], while others incorporate 3D architectures to capture spatio-temporal features [10, 43, 55, 58]. Among these, MAGVIT-v2 [56] has emerged as a prominent video tokenizer, proposing a look-up-free quantizer and has been widely adopted in recent models.

𝑛𝑞

Conv. Dec. Module

|𝒒|
|---|

𝑤𝑆 ℎ𝑆

𝒛𝑺′ 𝒛𝑺

###### Q-Former

|ℎ|
|---|

|ℎ|
|---|

𝒖𝑺

𝑛𝑞

|𝑓|
|---|

|𝑛𝑞|
|---|

𝑤

𝑤

| | |
|---|---|
| | |

Dec. Module

###### Average on height

|𝓔|
|---|

|𝓓|
|---|

𝒛𝑫(𝒉)

𝒖𝑫(𝒉)

|ℎ|
|---|

|𝒛|
|---|

𝑓

ℎ

𝑤𝐷

Conv.

|𝑓|
|---|

|𝑓|
|---|

|𝑤|
|---|

𝑤

|𝒛𝑫|
|---|

| | |
|---|---|
| | |

𝒛𝑫′ 𝑓

ℎ𝐷

Dec. Module

𝑤𝐷

ℎ𝐷

𝒛𝑫(𝒘)

𝒖𝑫(𝒘)

𝓕𝑺 𝓗𝑺 𝓕𝑫 𝓗𝑫 Average on width

|ℎ|
|---|

𝑓

|𝑓|
|---|

|𝑤|
|---|

- Figure 2. Details of our model. After obtaining the latent z from the Encoder, the process branches into two flows. The Structure Latent extraction module, FS, which consists of a Q-Former and convolutional networks, extracts the Structure Latent component zS. The Dynamics Latent extraction module, FD, comprising convolutional networks and an averaging operator, extracts the Dynamics Latent component zD. Finally, using the decoding module, we align all latents to the same dimension and combine them before passing them into the Decoder.

##### 2.2. Video Compression and Decoupling

Video compression is a critical challenge in computer vision, and the philosophy of decoupling has been employed in traditional video codecs for many years. For instance, MPEG-4 [23] uses I-frames to represent key frames and macroblock motion to capture movement. Building on this concept, Video-LaViT [20] recently designed a pipeline that transforms key frames and motion vectors into tokens, integrating them with large language models. Other representative methods for motion representation include MotionI2V [35], which uses pixel trajectories to capture motion, and [24], which employs optical flow for frame interpolation. Some approaches focus on specific video types, such as GAIA series [12, 47, 57] focuses on talking-face videos and uses self-cross reenactment to disentangle identity and motion, or iVideoGPT [51], which explores embodied videos. CMD [58] utilizes a weighted average of all frames to represent content, while motion is learned by a neural network. However, we identify several limitations in these methods, such as incompatibility with generative models, reliance on complex architectures, or unsatisfactory results due to excessive prior knowledge in some models. In contrast, we revisit the decoupling mechanism and propose a novel approach. Experiments demonstrate that our method has great promise, and we hope it will inspire further innovation in the community.

### 3. Methodology

In this section, we introduce the VidTwin model. In Sec. 3.1, we provide an overview of the architecture of VidTwin. Subsequently, Sec. 3.2 describes the process of converting a video into Structure Latent and Dynamics Latent, while Sec. 3.3 delineates the process of reconstructing the video from these two latents. In Sec. 3.4, we outline the training and inference pipelines, and lastly, in Sec. 3.5, we discuss a design for adapting our proposed latents for use with diffusion models.

##### 3.1. Overall Architecture

A classical autoencoder consists of an encoder E and a decoder D. Given a video x ∈ RC×F×H×W, where C, F, H, W represent the channel, number of frames, height, and width, respectively, the encoder produces a latent vector z = E(x) ∈ Rc×f×h×w, where c, f, h, w are corresponding dimensions with x but with lower dimensions. The decoder attempts to reconstruct the input as xˆ = D(z) = D(E(x)). The encoder and decoder are jointly trained to minimize the reconstruction loss Lrec = ∥xˆ − x∥.

In our VidTwin model, we propose decoupling a video into Structure Latent and Dynamics Latent components. As illustrated in Fig. 2, after obtaining the latent vector z, we introduce two processing functions, FS and FD, which generate the desired latent representations zS and zD. These procedures are described in detail in Sec. 3.2.1 and Sec. 3.2.2. For decoding, we employ two functions,

HS and HD, to align these latents to the same dimensional space before combining them and passing them to the decoder. The overall procedure is summarized as follows:

zS,zD = FS E(x) ,FD E(x) x ˆ = D [HS(zS);HD(zD)]

##### 3.2. Encode Video into Latents

We will use Structure function and Dynamics function to extract Structure Latent and Dynamics Latent, respectively.

###### 3.2.1. Structure Latent Extraction

To extract the temporal low-frequency representation from the encoder’s output latent z ∈ Rc×f×h×w, we employ the Q-Former, a classical interface proposed in BLIP-2 [25] that serves as a bridge between different modalities. We choose this module due to its elegant architecture and proven ability to extract semantic information from visual input. It is a Transformer [42] architecture with learned queries as input. In each block, the latent serves as a condition to perform cross-attention, and the last hidden states are taken as the output. In our scenario, as shown in Fig. 2, we define the query q as nq tokens (nq ≤ f) with dimension dq as input. Then, for the latent z, we turn it into a sequence by merging the spatial dimensions into the batch dimension, resulting in dimension (hw,f,c). We then use an MLP to convert the channel dimension c into dq, and perform standard QFormer operations along the temporal dimension. This process dynamically selects nq representative features from the f frames. The final output zS′ is obtained as:

zS′ = Qformer(z,q) ∈ R(hw)×n

q×dq

Notably, when we combine the height and width dimensions into the batch dimension, it compels the Q-Former to learn the general temporal motion trends independently of location, which aligns with our expectations.

We now have the intermediate latent zS′ , but it still faces two challenges: (1) Spatial compression has not been performed, resulting in a high product of h and w, and (2) the dimensionality of the Q-Former’s hidden state, dq, remains high. To address these, we reshape zS′ into shape (nq,dq,h,w) and apply several convolutional layers to downsample the spatial dimensions while using a bottleneck to reduce the channel dimension dq to a smaller size dS. These operations reduce the dimensionality of the final Structure Latent while preserving main content information by eliminating detailed spatial information. Finally, we obtain the final Structure Latent zS ∈ Rn

q×dS×hS×wS.

###### 3.2.2. Dynamics Latent Extraction

For dynamic local details, we consider that rapid motion information should be low-dimensional and distributed across each frame. Therefore, instead of manipulating the temporal dimension, we primarily focus on the spatial dimensions.

A natural approach to reduce the dimensions is to use a spatial Q-Former to extract the most relevant spatial locations, similar to the method used for the Structure Latent. However, this approach disrupts spatial consistency, leading to performance degradation in our experiments.

Instead, we design an alternative approach. As shown in Fig. 2, we first downsample the latent z along the spatial dimensions using convolutional layers, obtaining an intermediate result zD′ with dimensions (f,c′D,hD,wD). Inspired by [58], we then average zD′ along the height and width dimensions to eliminate these spatial dimensions. The resulting vectors are concatenated and passed through a head G to reduce the channel dimension to dD:

zD = G ([avgh(zD′ );avgw(zD′ )]) ∈ Rf×d

D×(wD+hD)

This results in the Dynamics Latent zD. Notably, this approach reduces the latent dimension from O(wD · hD) to O(wD + hD), effectively extracting compact dynamic details while preserving spatial integrity.

##### 3.3. Decode Latents to Video

With the expected latents Structure Latent and Dynamics Latent obtained, we need to find a way to combine them before inputting them into the decoder. For Structure Latent zS with shape (nq,dS,hS,wS), we apply upsampling layers to recover the spatial size and MLPs to adjust the channel dimension dS and query token number nq, yielding uS ∈ Rc×f×h×w.

For Dynamics Latent zD with shape (f,dD,wD + hD), we process the latents for height and width separately. Specifically, for latents zD(h) ∈ Rf×d

D×wD and zD(w) ∈ Rf×d

D×hD, we use MLPs T to recover the corresponding spatial and channel dimensions, followed by repeating along the missing spatial dimension:

u(Dh) = Repw(T (zD(h))) ∈ Rc×f×h×w u(Dw) = Reph(T (zD(w))) ∈ Rc×f×h×w

Subsequently, we perform an element-wise addition of these latents and pass them to the decoder to obtain the final output video:

xˆ = D(uS + u(Dh) + u(Dw)) ∈ RC×F×H×W

##### 3.4. Training and Inference

We train all modules, including the Encoder E, Decoder D, latent extraction modules FS, FD, and decoding heads HS, HD, jointly to recover the input. Following the standard loss definition for image autoencoders proposed in VQ-GAN [8], we employ the basic reconstruction loss Lrec along with feature-level perceptual loss Lp and adversarial losses LGAN. Considering that VidTwin is likely to be integrated into a generative model, we expect the latent space

- Table 1. Quantitative comparison with baseline methods. The bold values indicate the best results, while the underlined values represent the second-best. Sem., Tempo., and Deta. refer to semantic preservation, temporal consistency, and detail retention, respectively. Our model outperforms the baselines across multiple metrics, demonstrating its superior reconstruction ability.

Method Compress. Rate ↓ PSNR ↑ LPIPS ↓ SSIM ↑ FVD ↓ Sem. ↑ Tempo. ↑ Deta. ↑

iVideoGPT [51] 1.50% 19.353 0.4677 0.5752 1693.10 4.28 4.33 3.59 MAGVIT-v2 [56] 0.65% 24.351 0.3347 0.6877 653.88 4.43 4.46 3.97 CMD [58] 6.85% 27.332 0.2732 0.7746 468.47 4.51 4.35 4.22 EMU-3 [46] 0.53% 25.359 0.2543 0.7260 353.71 4.69 4.57 4.60 CV-VAE [61] 0.53% 28.063 0.2436 0.7546 401.92 4.70 4.51 4.67

###### VidTwin (Ours) 0.20% 28.137 0.2414 0.8044 388.86 4.71 4.62 4.73

to be sufficiently smooth. Thus, we adopt a VAE paradigm, wherein instead of directly inputting the latents into the decoder, we introduce randomness around the latents, namely v(z) = µ(z) + σ(z) · ϵ, where ϵ ∼ N(0,I), and µ(z) and σ(z) are learnable modules predicting the mean and standard deviation. To regularize this distribution, we use the KL divergence loss with the standard Gaussian distribution LKL = KL(N(µz,σz)||N(0,I)). More detailed explanations of the VAE model can be found in Appendix E.2. The final loss is defined as:

L = Lrec + λpLp + λGANLGAN + λKLLKL

During sampling, we use the mean of the latent, i.e., µ(z). If the required latents are predicted from a generative model, we simply follow the decoding method mentioned in Sec. 3.3 to generate the final video.

##### 3.5. Conditional Video Generation with VidTwin

Typically, VidTwin is expected to connect with a generative model. Here, we present a basic design to adapt Structure Latent and Dynamics Latent for use in a diffusion model and welcome other designs from the community.

Given a video, we first apply the trained VidTwin to obtain the Structure Latent latent zS and the Dynamics Latent latent zD. The dimension of zS is (dS,nq,hS,wS), which resembles “video-like” data. For zD, we combine the latents along the height and width dimensions, introducing a pseudo-dimension in the second dimension to yield (dD,1,f,hD+wD), effectively treating it as a single-frame video. We then apply a 3D patchification method to convert both latents into two sequences of tokens, each with dimension dDiff. Since these token embeddings originate from different latents, we align them to a similar scale through normalization and then concatenate them along the length dimension to form the training target.

With the ground-truth latent training target y0 and any relevant conditions c (such as text or video class), we perform the standard diffusion training procedure [15]. This involves sampling noise, adding it to the latent to get the noisy

version yt, and then attempting to remove the noise using a learnable model Di. We utilize the current popular mechanism to predict x0 directly, defined as LDiff = ∥Di(yt,c)− y0∥. During sampling, we follow the DDIM [37] method to predict yˆ0. We also employ classifier-free guidance [14] to further enhance the model’s conditioning capabilities. More details about the diffusion model can be found in Appendix D.3. Finally, we input the predicted latents into the decoder of VidTwin to generate the final output video.

### 4. Experiments

We conduct experiments to validate the proposed VidTwin model, from aspects including the compression rate, reconstruction ability, as well as the effectiveness and efficiency on downstream tasks.

##### 4.1. Setup

###### 4.1.1. Datasets

For training, we utilize a self-collected large-scale textvideo dataset, containing 10 million video-text pairs. Considering the broad variety of content and motion speed in this dataset, we believe training on this dataset is a good choice to fulfill our design philosophy. For evaluation, we use the MCL-JCV dataset [44], which is a classical dataset for evaluating video compression quality. Moreover, to verify the adaptability of the latent emitted by our model to generative models, we evaluate the class-conditioned video generation ability on the UCF-101 [38] dataset, which provides 101 different classes of motion videos.

###### 4.1.2. Implementation Details

We train our model on 8 fps, 16-frame, 224 × 224 video clips and evaluate on 25 fps, 16-frame, 224 × 224 video clips. The backbone of our model is a Spatial-Temporal Transformer [2] with a hidden dimension of 768 and a patch size of 16. Both the encoder and decoder consist of 16 layers, each with 12 attention heads, resulting in a total of about 300M parameters. For the latent representation, we configure one setting with hS = hD = 7, with dimen-

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Original

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

iVideoGPT

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

MAGVIT-v2

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

EMU-3

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

CMD

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

VidTwin (Ours)

- Figure 3. Qualitative comparison with baseline methods. Two examples are presented: a gradually rotating photo and a fast-motion boxing scene. VidTwin demonstrates the ability to reconstruct fine details and accurately capture rapid motion.

sions 4 and 8, respectively. This results in two latent representations of sizes 7 × 7 × 16 × 4 and 16 × 14 × 8 for a 16 × 3 × 224 × 224 video clip. Unless otherwise specified, all results in this paper use this configuration. The model is trained on 4 × A100 GPUs. We use the Adam optimizer with a learning rate of 1.6e-4. Additional hyperparameters and model settings are provided in Appendix D.

- 4.2. Baselines and Evaluation Metrics

and detail retention (Deta.). Each evaluator is presented with 20 samples and asked to rate each on a scale from 1 to 5. The final evaluation score is computed as the Mean Opinion Score (MOS), representing the average rating across evaluators. Moreover, we also introduce the Compression Rate (Compress. Rate) metrics, which we define as the ratio between the dimension of the latent space (or token embeddings) used in the downstream generative model, and the input video’s dimension.

We select several state-of-the-art baselines, including models that represent videos as latents with uniform size, such as MAGVIT-v2 [56], CV-VAE [61] and the visual tokenizer of EMU-3 [46], as well as models that decouple content and motion, like CMD [58] and iVideoGPT [51]. Additional analysis of other concurrent baselines is provided in Appendix A.4. We compare these baselines with our model using standard reconstruction metrics, including PSNR [18], SSIM [49], LPIPS [59], and FVD [40]. Additionally, we conducted a human evaluation by inviting 15 professional evaluators to assess the results based on three criteria: semantic preservation (Sem.), temporal consistency (Tempo.),

##### 4.3. Reconstruction Quality

As shown in Tab. 1, our model achieves state-of-the-art performance across most objective and subjective metrics, demonstrating strong capabilities in video reconstruction. The vision tokenizer of EMU-3 achieves the best FVD score and good reconstruction ability, likely due to the large dataset it was trained on (i.e., InternVid [48]). While most models perform well in semantic preservation and temporal consistency, they vary significantly in detail retention, where our model outperforms the others. Additionally, our model utilizes a highly compact latent space, approximately

[Figure 51]

[Figure 52]

Video A Offers S. Latent

Video B Offers D. Latent

Video C (Generated)

Figure 4. An illustration of a cross-replacement example, where Video C is generated using the Structure Latent from Video A and the Dynamics Latent from Video B.

2.5 to 30 times smaller than those of the baselines. It is encouraging to see that our model achieves comparable or superior reconstruction quality with such a low-dimensional latent space, highlighting the efficiency and effectiveness of VidTwin. We also train our architecture at different parameter scales, and larger models perform even better. This scalability is likely due to our Transformer-based architecture; further details can be found in Appendix B.2.

Furthermore, we conduct case studies, with qualitative results shown in Fig. 3 (zoom in to observe finer details). For the left case, we observe that our model effectively captures local details and the gradual rotation of the object. In comparison, baselines such as CMD show blurred edges and incomplete rotation. In the right case, featuring the fast motion of a man boxing, all baselines struggle to accurately capture the rapid movements, resulting in ghosting artifacts. In contrast, VidTwin produces significantly clearer results, demonstrating the effectiveness of our decoupling strategy for capturing both low-frequency changing objects and rapid local motion. More cases are presented in Appendix A.1.

##### 4.4. Further Analysis

As highlighted in Sec. 1, our VidTwin not only demonstrates strong reconstruction capabilities but also excels in explainability, efficiency, and adaptability with generative models. In this section, we provide evidence to support these claims.

###### 4.4.1. Explorations on the Roles of Latents

In VidTwin, we design two distinct latents: Structure Latent for the main object and overall movement trend, and Dynamics Latent, which captures local details and rapid motions. We present two experiments that provide insight into

Figure 5. We present the FLOPs and training memory costs of the unified generative model, as applied to our model and the baselines.

their respective roles.

First, as discussed in Sec. 3.3, we perform element-wise addition of the latents before inputting them into the decoder. This setup enables us to explore the outputs generated when each latent is passed through the decoder individually, i.e., generating results from D(uS) and D(uD). An example provided in Fig. 1 of the Sec. 1 illustrates the distinct differences between the two latents using a scenario involving the screwing process. As observed, the Structure Latent captures the main semantic content, such as the table and screw, while the Dynamics Latent captures fine-grained details, including color and rapid local movements of the screw. Notably, in frame t2, where the screw drops, the video generated by the Structure Latent shows only a slight change, whereas the one generated by the Dynamics Latent captures this immediate movement. This demonstrates the distinction between low-frequency and high-frequency movement trends.

Second, we conduct a cross-reenactment experiment in which we combine the Structure Latent from one video, A, with the Dynamics Latent from another video, B, to observe the generated output from the decoder, i.e., generating D(uAS,uBD). As shown in Fig. 4, the generated video inherits the main object (house) and overall structure from Video A, which provides Structure Latent, while the local color comes from Video B, which provides Dynamics Latent. Notably, we observe that the movement in the generated video inherits the rapid rotation from Video B, while adjusting the gradually downward camera view according to the scene in Video A. This further validates our motivation to decouple video content into overall structure and detailed dynamics. We provide additional examples for both settings in Appendix A.

One additional note is that, as suggested by the name VidTwin, the Structure Latent and Dynamics Latent latents work together to generate the final video. These separate

- Table 2. The generative ability of our model and the baselines, as tested on UCF-101.

Models TATS [10] MAGVIT-v2 [56] Video-LaViT [20] Ours FVD ↓ 332 58 275 193

analyses are intended to offer a glimpse into the roles of each latent, but it is important to note that isolating them inevitably introduces information loss. In future work, we plan to explore additional methods for better understanding the intrinsic information stored in these separate latents.

- 4.4.2. Computation Resource Analysis for Generative Models

Through our decoupling design, we reduce redundancy, resulting in compact latents with a high compression rate. A key advantage of having lower-dimensional latents is the reduced computational resource requirements for downstream tasks. To demonstrate this, we compare the FLOPs and memory consumption of generative models based on representative baselines. For a fair comparison, instead of using the original generative models from their respective papers, which vary significantly, we construct a pseudo uniform DiT [30] architecture with a uniform patch size, focusing solely on resource consumption rather than generative ability. The results are shown in Fig. 5. As observed, the downstream diffusion model that fits our latent space, which has a higher compression rate, requires significantly fewer FLOPs and less training memory (4 to 8 times and 2 to 3 times smaller than the baselines, respectively). This reduction in resource consumption leads to improved deployment efficiency. Furthermore, given the smaller dimension of our latent space, it is possible to use a smaller diffusion model to fit the distribution, further reducing resource requirements. Additional details about the pseudo DiT model used can be found in Appendix C.2.

- 4.4.3. Generative Quality of Diffusion Models

As shown in Sec. 3.5, we design a basic method to adapt our latent representations to the generation framework of a DiT-based diffusion model. We evaluate the proposed method on the UCF-101 dataset [38] for class-conditional video generation, with the results reported in Tab. 2. Our model achieves performance comparable to several existing methods. It is important to note that the main focus of this paper is not on generation, and we have implemented only a simple baseline model to evaluate the adaptability of our approach to the diffusion framework. Despite this, the results are promising and demonstrate that the latent space in VidTwin is well-suited for downstream generative tasks. We believe that with a more refined design, a larger dataset, and the incorporation of additional techniques during training, a generation model based on our latent space

Table 3. Ablation studies on the proposed techniques. Methods PSNR↑ SSIM↑ VidTwin 26.116 0.731

- (a) w/o Disentanglement 23.512 0.654

- (b) w/o D. Latent Avg. 24.835 0.693

- (c) w/o S. Latent Qformer 25.386 0.702

- (d) w/o S. Latent Move Spa. 23.169 0.630

will achieve even better performance.

##### 4.5. Ablation Studies

We conduct an ablation study to assess the impact of our proposed designs by removing each one. The experiments are evaluated using the same number of training steps, and the results are presented in Tab. 3. The findings can be summarized as follows: (a) When we omit the disentangling paradigm and use a single latent with a similar compression rate, performance drops significantly, demonstrating that our decoupling approach not only produces meaningful latent representations but also enhances performance at the same compression rate. (b) As discussed in Sec. 3.2.2, replacing the averaging method with a Spatial Q-Former to further compress the spatial dimensions of Dynamics Latent results in poorer performance, likely due to the disruption of spatial arrangement. (c) We propose using a Q-Former to extract Structure Latent. When we replace it with simple convolution layers and an MLP to decrease the temporal dimension, performance degrades, highlighting the superior semantic extraction capability of the Q-Former. (d) As mentioned in Sec. 3.2.1, moving the spatial dimensions into the batch dimension to obtain location-independent latents is crucial. Without this, and by placing them into the hidden states dimension instead, we observe a noticeable performance loss.

### 5. Conclusion

In this paper, we present VidTwin, a novel model for video latent representation. VidTwin incorporates carefully designed submodules within an Encoder-Decoder framework to effectively separate Structure and Dynamics latent spaces. Through extensive experiments, we demonstrate that VidTwin achieves high compression rates, has a simple architecture, and performs well in downstream generative tasks. Additionally, inspired by [50], the Structure Latent space in our model appears well-suited for visual understanding tasks, which we plan to explore in future work. Finally, our approach provides explainability and scalability, making it valuable for future research. We hope that our work will inspire new decoupling techniques in the video community and contribute to advancements in both video generation and broader multimodal applications.

### Acknowledgments

This work was conducted during an internship at Microsoft Research Asia and was supported by the company. Additionally, it was partially supported by the National Natural Science Foundation of China under Grant No. 92470205.

### References

- [1] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation, 2024. 2
- [2] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding?,

2021. 2, 5, 14

- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. 1
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models, 2023. 1, 2
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

2023. 2

- [6] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation, 2024. 2
- [7] Liuhan Chen, Zongjian Li, Bin Lin, Bin Zhu, Qian Wang, Shenghai Yuan, Xing Zhou, Xinhua Cheng, and Li Yuan. Od-vae: An omni-dimensional video compressor for improving latent video diffusion model, 2024. 2, 12
- [8] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis, 2021. 2, 4, 14
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 1, 2
- [10] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer, 2022. 2, 8
- [11] Agrim Gupta, Stephen Tian, Yunzhi Zhang, Jiajun Wu, Roberto Mart´ın-Mart´ın, and Li Fei-Fei. Maskvit: Masked visual pre-training for video prediction, 2022. 2

- [12] Tianyu He, Junliang Guo, Runyi Yu, Yuchi Wang, Jialiang Zhu, Kaikai An, Leyi Li, Xu Tan, Chunyu Wang, Han Hu, HsiangTao Wu, Sheng Zhao, and Jiang Bian. Gaia: Zeroshot talking avatar generation, 2024. 3
- [13] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation, 2023. 1
- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 5, 16
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 5, 17
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models, 2022. 2
- [17] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022. 1, 2
- [18] Alain Hor´e and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th International Conference on Pattern Recognition, pages 2366–2369, 2010. 6
- [19] Junpeng Jiang, Gangyi Hong, Lijun Zhou, Enhui Ma, Hengtong Hu, Xia Zhou, Jie Xiang, Fan Liu, Kaicheng Yu, Haiyang Sun, Kun Zhan, Peng Jia, and Miao Zhang. Dive: Dit-based video generation with enhanced control, 2024. 2
- [20] Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization, 2024. 1, 3, 8
- [21] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. 14, 16
- [22] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. 2, 17
- [23] Didier Le Gall. Mpeg: A video compression standard for multimedia applications. Communications of the ACM, 34

(4):46–58, 1991. 3

- [24] Jaihyun Lew, Jooyoung Choi, Chaehun Shin, Dahuin Jung, and Sungroh Yoon. Disentangled motion modeling for video frame interpolation, 2024. 3
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. 2, 4, 15
- [26] Zongjian Li, Bin Lin, Yang Ye, Liuhan Chen, Xinhua Cheng, Shenghai Yuan, and Li Yuan. Wf-vae: Enhancing video vae by wavelet-driven energy flow for latent video diffusion model, 2024. 2, 12
- [27] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, Tanghui Jia, Junwu Zhang, Zhenyu Tang, Yatian Pang, Bin She, Cen Yan, Zhiheng Hu, Xiaoyi Dong, Lin Chen, Zhang Pan, Xing Zhou, Shaoling Dong, Yonghong Tian, and Li Yuan. Open-sora plan: Open-source large video generation model, 2024. 2, 12

- [28] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple, 2023. 2
- [29] OpenAI. Video generation models as world simulators.

2024. 1, 2

- [30] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 8, 14, 15
- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 1, 2
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation,

2015. 17

- [34] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023. 1
- [35] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling, 2024. 3
- [36] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022. 2
- [37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 5, 16, 17
- [38] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild, 2012. 2, 5, 8, 15
- [39] Anni Tang, Tianyu He, Junliang Guo, Xinle Cheng, Li Song, and Jiang Bian. Vidtok: A versatile and open-source video tokenizer. arXiv preprint arXiv:2412.13061, 2024. 2
- [40] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges, 2019. 6
- [41] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning,

2018. 2

- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. 4, 17
- [43] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description, 2022. 2
- [44] Haiqiang Wang, Weihao Gan, Sudeng Hu, Joe Yuchieh Lin, Lina Jin, Longguang Song, Ping Wang, Ioannis Katsavounidis, Anne Aaron, and C-C Jay Kuo. Mcl-jcv: a jnd-based

- h. 264/avc video quality assessment dataset. In 2016 IEEE international conference on image processing (ICIP), pages 1509–1513. IEEE, 2016. 5
- [45] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers,

2022. 2

- [46] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024. 1, 5, 6, 13
- [47] Yuchi Wang, Junliang Guo, Jianhong Bai, Runyi Yu, Tianyu He, Xu Tan, Xu Sun, and Jiang Bian. Instructavatar: Textguided emotion and motion control for avatar generation,

2024. 3

- [48] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation, 2024. 6
- [49] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [50] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024. 8
- [51] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. ivideogpt: Interactive videogpts are scalable world models, 2024. 1, 3, 5, 6, 13, 14
- [52] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7623–7633, 2023. 2
- [53] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2024. 2, 12
- [54] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan, 2022. 2
- [55] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. Magvit: Masked generative video transformer, 2023. 1, 2

- [56] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 1, 2, 5, 6, 8, 13, 14
- [57] Runyi Yu, Tianyu He, Ailing Zhang, Yuchi Wang, Junliang Guo, Xu Tan, Chang Liu, Jie Chen, and Jiang Bian. Make your actor talk: Generalizable and high-fidelity lip sync with motion and appearance disentanglement, 2024. 3
- [58] Sihyun Yu, Weili Nie, De-An Huang, Boyi Li, Jinwoo Shin, and Anima Anandkumar. Efficient video diffusion models via content-frame motion-latent decomposition, 2024. 1, 2, 3, 4, 5, 6, 13
- [59] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 6
- [60] Wentao Zhang, Junliang Guo, Tianyu He, Li Zhao, Linli Xu, and Jiang Bian. Video in-context learning: Autoregressive transformers are zero-shot video imitators. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [61] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cv-vae: A compatible video vae for latent generative video models,

2024. 1, 2, 5, 6, 13

- [62] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 2, 12

### A. Additional Experimental Results

To enhance the visual experience, we strongly encourage viewing the videos on the website.

##### A.1. Additional Reconstruction Examples

- Fig. 8 presents additional reconstruction examples. By zooming in, one can observe that our VidTwin effectively captures intricate details, such as raindrops in the first and second cases. Moreover, by decoupling structural and dynamic motion features, our model excels at preserving rapid motion dynamics. For example, in the third case, VidTwin accurately reproduces the light trails of a fast-moving car, where other baselines fail to do so.

A.2. Additional Decoupling Examples

In Sec. 4.4.1, we demonstrated the ability to separately recover the Structure Latent and Dynamics Latent components. Additional examples are shown in Figure 7. Videos generated using Structure Latent predominantly capture primary structures and main objects, while those generated with Dynamics Latent focus on colors and rapid movements.

- A notable example is observed in the bottom-right case,

where fireworks visible in the first frame disappear in the second. However, the Structure Latent-generated video retains the fireworks from the first frame, demonstrating that Structure Latent effectively encodes low-frequency, gradually evolving information.

We would like to emphasize that our primary objective is not to completely decouple structure and dynamics, as this is a challenging problem even for humans. Instead, we observe potential in reducing temporal redundancy in video representation. Based on this observation, we designed an algorithm that strives to decouple video content into these two spaces. Therefore, the cross-reenactment experiment was only designed to intuitively demonstrate the roles of the two latents rather than being specifically optimized for cross-reenactment videos.

A.3. Additional Cross-Reenactment Examples

Fig. 9 provides further examples of the cross-reenactment experiments described in Sec. 4.4.1. In these examples, the generated videos inherit the basic structure from Video A while incorporating local details and motions from Video

- B. Notably, motion patterns such as horizontal movements and wave-like motions, as seen in the two bottom cases, are effectively transferred.

##### A.4. Comparison with Concurrent Baselines

Recent works have explored the field of video autoencoders [7, 26, 27, 53, 62]. We observe that most of these baselines still fall into the category of methods that represent frames as latent vectors of uniform size, as dis-

cussed in Sec. 1. A comparison between our model and these baselines is presented in Tab. 5. Notably, our model achieves performance comparable to state-of-the-art methods. CogVideoX [53] demonstrates impressive results, likely due to its large-scale training data. Additionally, even our highest compression rate model achieves a lower compression rate than other models (typically 0.6% with 8,8,4).

### B. Additional Analysis for VidTwin

##### B.1. Definition of Compression Rate and the Tradeoff with Reconstruction Quality.

Differs from the typical representation that uses the downsampling factors for height, width, and number of frames for compression rate, we define it as the ratio between the dimension of the latent used in the downstream model and the input video’s dimension. For example, the typical downsampling factor (8,8,4) with channels 4 corresponds to a compression rate of 0.65% in our definition. Additionally, we present the trade-off between compression rate and reconstruction quality in Tab. 6. As shown, lower compression rates generally result in better reconstruction quality.

##### B.2. Initial Scalability Exploration

In Sec. 4.3, we described training our architecture at varying parameter scales and observed consistent performance improvements with larger models. Tab. 4 summarizes the configurations of each model, evaluated at the same training step. The results demonstrate a steady enhancement in reconstruction quality with increasing model size. In future work, we plan to explore additional model scales and investigate potential scaling laws, including exponential trends and other patterns.

##### B.3. Performance of VidTwin with Increased Frames and Higher FPS

We selected the same subjective evaluation subset as in Sec. 4.2 and sampled videos with 32 frames and 40 FPS. A new user study was conducted, and the results are presented in Tab. 7. The findings indicate that VidTwin maintains strong performance with an increased number of frames and a higher frame rate.

##### B.4. Failure modes of VidTwin

We provide a failure case in Fig. 6, depicting a basketball scene with fast player movements and camera motion. While the background remains well-preserved, the fast-moving individuals appear blurred. In terms of components, the S. Latent captures the background but becomes blurred for the players, which is expected as it encodes slowly changing semantic information. The D. Latent captures the fast-changing players but struggles to accurately integrate them into the reconstructed video due to their extremely rapid movement. We plan to address this issue by

Table 4. Settings and performance of VidTwin at different scales.

Models Depth Num. Heads Dim. Hidden Num. Params. PSNR SSIM VidTwin small 12 8 512 126M 24.83 0.683 VidTwin base 16 12 768 335M 26.13 0.732 VidTwin large 16 12 1536 1.3B 27.16 0.751

Table 5. Comparison with other concurrent works.

[Figure 53]

Models (Comp. Rate) PSNR↑ LPIPS↓

CV-VAE (0.53%) 28.06 0.24 OD-VAE (0.53%) 29.18 0.19

[Figure 54]

Open-Sora (0.53%) 29.89 0.15 CogVideoX (0.53%) 31.92 0.09

VidTwin (0.20%) 28.14 0.24 VidTwin (0.48%) 30.04 0.15

###### Orig. Recon. S. Latent D. Latent

- Table 6. The reconstruction quality of different compression rates.

Compression Rate PSNR↑ LPIPS↓

0.11% 24.41 0.35 0.16% 27.03 0.28 0.20% 28.14 0.24 0.48% 30.04 0.15

- Table 7. Subjective evaluation of VidTwin with increased frames and higher FPS.

Figure 6. Failure modes of VidTwin.

MAGVIT-v2 [56]: MAGVIT-v2 employs 3D causal CNN layers to downsample videos into latents, with a temporal downsampling factor of 4 and spatial downsampling factor of 8. The latent dimension is set to 5, as reported in the paper, resulting in a compression rate of:

5 3 × 4 × 8 × 8 ≈ 0.65%.

EMU-3 [46]: EMU-3 is a generative model proposed by BAAI1. For our evaluation, we primarily utilize its video tokenizer, which is based on SBER-MoVQGAN2. This tokenizer incorporates two temporal residual layers with 3D convolutional kernels in both the encoder and decoder modules, enhancing video tokenization. Similar to MAGVITv2, it achieves a 4× temporal compression and 8×8 spatial compression. The compression rate, with a latent size of 4, is calculated in the same manner.

Model Sem. ↑ Tempo. ↑ Deta. ↑

VidTwin 4.71 4.62 4.73 w/ 32 frames 4.70 4.53 4.69 w/ 40 fps 4.73 4.64 4.71

pretraining on low-fps videos and fine-tuning on high-fps videos in future work.

### C. Additional Information on Experimental Settings

##### C.1. Baselines and Compression Rates

This section provides details on the baselines used in our evaluation and discusses their compression rates, as outlined in Sec. 4.2. Notably, MAGVIT-v2 [56], iVideoGPT [51], and CMD [58] do not offer official code or pretrained checkpoints. Therefore, we reimplement these methods based on the descriptions provided in their respective papers.

CV-VAE [61] CV-VAE is a video VAE of latent video models, designed to have a latent space compatible with that of a given image VAE, such as the image VAE in Stable Diffusion (SD). In terms of compression rate, it matches that of EMU-3, achieving 4× temporal compression and 8×8 spatial compression.

CMD [58]: CMD decouples video representations into content frames and motion latents. For a video of size

- 1https://www.baai.ac.cn/
- 2https://github.com/ai-forever/MoVQGAN

[Figure 55]

[Figure 56]

|𝒕𝟏| | |
|---|---|---|
| | | |

[Figure 57]

[Figure 58]

|𝒕𝟐|
|---|

[Figure 59]

[Figure 60]

|𝒕𝟏| | |
|---|---|---|
| | | |

[Figure 61]

[Figure 62]

|𝒕𝟐|
|---|

Orig. Recon. S. Recon. D. Recon.

Orig. Recon. S. Recon. D. Recon.

Figure 7. Additional examples of decoupling Structure Latent and Dynamics Latent.

of downstream generative models. To validate this, in Sec. 4.4.2, we compare the performance of a generative model applied to the latent spaces produced by VidTwin and the baselines.

(c,f,h,w), the content frame has dimensions (c,h,w), and the motion latent is (d,h + w,f), where d is the dimension of the motion vector. Based on the settings described in the paper, the compression rate is:

For a fair comparison, we utilize the same DiT [30] architecture in all experiments. The configuration includes 6 layers, 8 attention heads, a hidden dimension size of 512, and a feed-forward network (FFN) dimension of 2048, resulting in a total of 12,610,560 parameters. Additionally, a unified patch size of 2 is used for all dimensions.

- 2 × 224 × 32

- 3 × 224 × 224 ≈ 6.9%.

d(h + w) chw

1 16

1 f

+

=

+

The primary bottleneck lies in the content frame, and we hypothesize that longer video clips could reduce the compression rate (though at the potential cost of performance).

We calculate the FLOPs using a single sample (batch size = 1). For memory consumption, we employ the Adam [21] optimizer and record the maximum GPU memory usage during training.

iVideoGPT [51]: iVideoGPT employs a conditional VQGAN [8] with dual encoders and decoders. The context frames 1 : T0 are encoded using N0 tokens, while subsequent frames are encoded with fewer tokens (n), conditioned on the context tokens to capture the essential dynamics. The compression rate is given by:

### D. Implementation Details

##### D.1. Model Details

N0d + n(T − T0)d C × T × H × W

,

As described in Sec. 3.1, our VidTwin adopts an EncoderDecoder architecture. Specifically, we utilize a SpatialTemporal Transformer [2] backbone. In each block, spatial attention is first applied to the height and width dimensions, followed by temporal attention along the temporal dimension. Temporal attention uses causal masking, ensuring that earlier frames do not attend to later ones, similar to the configuration in MAGVIT-v2 [56]. We evaluate three different scales (outlined in Tab. 4) by adjusting the depth, hidden state dimensions, and other parameters. For spatial dimen-

and, based on the information in the paper, we calculate it as:

2 × 162 × 64 + 14 × 42 × 64 3 × 16 × 2562 ≈ 1.5%.

##### C.2. Pseudo DiT for Resource Consumption Evaluation

Our VidTwin model offers a highly compressed latent space, significantly reducing the resource requirements

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Original

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

iVideoGPT

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

MAGVIT-v2

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

EMU-3

[Figure 88]

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

CMD

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

VidTwin (Ours)

Figure 8. Additional reconstruction cases comparing our VidTwin model with baselines. Zoom in to observe finer details.

sions, a patch size of 16 is used for both height and width, while for the temporal dimension, the patch size is set to 1.

The Q-Former [25], employed for extracting Structure Latent components, consists of 6 layers with a hidden dimension of 64 and 8 attention heads. For downsampling, we primarily use convolutional layers with a stride of 2, while upsampling is performed using Upsample layers with a factor of 2. By varying the number of convolutional layers, latents of different sizes can be generated.

Recommended latent size settings are as Tab. 8. From our experiments, we see that these configurations exhibit minimal performance differences, allowing users to select a

setting based on specific requirements.

##### D.2. Data and Training Details

The key hyperparameters for training data and optimization are summarized as Tab. 9.

##### D.3. Diffusion Model Details

In Sec. 4.4.3, we describe the design of a diffusion model tailored to the latent space of our VidTwin model. This model adopts the DiT [30] architecture with 18 layers and a hidden state size of 1152. Conditioning is introduced via cross-attention, and for the UCF-101 dataset [38], we use a 256-dimensional vector to encode the class information.

[Figure 105]

[Figure 106]

|𝒕𝟏|
|---|

[Figure 107]

[Figure 108]

|𝒕𝟐|
|---|

[Figure 109]

[Figure 110]

|𝒕𝟏|
|---|

[Figure 111]

[Figure 112]

|𝒕𝟐|
|---|

Video A Offers S. Latent

Video B Offers D. Latent

Video C (Generated)

Video A Offers S. Latent

Video B Offers D. Latent

Video C (Generated)

Figure 9. Additional examples of cross-reenactment.

Table 8. Recommended settings for latent sizes.

Setting Structure Latent Dynamics Latent

- 1 hS = wS = 7,nq = 16,dS = 4 hD = wD = 7,dD = 8

- 2 hS = wS = 7,nq = 16,dS = 4 hD = wD = 4,dD = 16

- 3 hS = wS = 7,nq = 12,dS = 4 hD = wD = 7,dD = 8

The diffusion process consists of 1000 steps, with DDIM [37] used as the sampling strategy and 50 steps for inference. Classifier-free guidance [14] is applied, where conditioning is randomly dropped in 20% of the samples during training. The classifier-free guidance weight is set to

- 5 during sampling.

For training, we use the Adam optimizer [21] with β1 = 0.9,β2 = 0.999. The learning rate is managed with a Lambda scheduler and includes 10,000 warmup steps. Training is conducted on 8 × 40G A100 GPUs, with an input configuration of 16 video frames at a resolution of 224.

### E. Basics for Diffusion Models and VAE

##### E.1. Basics for Diffusion Models

Diffusion models are a class of emerging generative models designed to approximate data distributions. The training process consists of two phases: the forward diffusion process and the backward denoising process. Given a data point sampled from the real data distribution, x0 ∼ q(x)3,

3We follow the notation and derivation process of https:// lilianweng.github.io/posts/2021-07-11-diffusionmodels.

Table 9. Training Configuration

Parameter Value Input Video Resolution 224 Input Video Frames 16 Input Video FPS 8 Optimizer Adam; β1 = 0.9,β2 = 0.99 Learning Rate 1.6 × 10−4 Warmup Steps 5000 Learning Rate Scheduler Cosine Annealing Lp 0.05 Weight Decay 0.0001

LGAN 0.05 LKL 0.001 Training Batch Size 6 Training Device 4 × 80G A100 GPUs

the forward diffusion process gradually adds Gaussian noise to the sample, generating a sequence of noisy samples x1,...,xT. The noise scales are controlled by a variance schedule βt ∈ (0,1), and the density can be expressed as:

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI).

Using the reparameterization trick [15], this process al-

lows for sampling at any arbitrary time step in closed form: q(xt|x0) = N(xt;√αtx0,√1 − αtI),

where αt = 1 − βt and αt = ti=1 αi. From this, it is evident that as T → ∞, xT converges to an isotropic Gaussian distribution, aligning with the initial condition used during inference.

However, obtaining a closed form for the reverse process q(xt−1|xt) is challenging. When βt is sufficiently small, the posterior also approximates a Gaussian distribution. In this case, a model pθ(xt−1|xt) can be trained to approximate these conditional probabilities:

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)),

where µθ(xt,t) and Σθ(xt,t) are parameterized by a denoising network fθ, such as a U-Net [33] or a Transformer [42]. By deriving the variational lower bound to optimize the negative log-likelihood of x0, Ho et al. [15] introduces a simplified DDPM learning objective:

Lsimple =

T

Eq ∥ϵt(xt,x0) − ϵθ(xt,t)∥2 ,

t=1

where ϵt represents the noise added to the original data x0. In our work, we adopt a simpler architecture that directly predicts x0, with the loss function defined as:

L = ∥x0 − fθ(xt,t)∥.

During inference, the reverse process begins by sampling noise from a Gaussian distribution, p(xT) = N(xT;0,I), and iteratively denoising it using pθ(xt−1|xt) until x0 is obtained. DDIM [37] refines this process by ensuring its marginal distribution matches that of DDPM. Consequently, during generation, only a subset of diffusion steps {τ1,...,τS} is sampled, significantly reducing inference latency.

##### E.2. Basics for VAE

Variational Autoencoders (VAEs) [22] are a class of generative models that combine probabilistic reasoning with neural networks to learn the underlying distribution of highdimensional data. A VAE consists of two components: an encoder and a decoder. The encoder maps input data x to a latent variable z characterized by a probabilistic distribution q(z|x), typically parameterized as a Gaussian. The decoder reconstructs the input by sampling from the latent space and generating data through p(x|z).

To ensure that the latent space conforms to a structured prior distribution, typically a standard Gaussian p(z) = N(0,I), VAEs optimize the Evidence Lower Bound (ELBO):

L = Eq(z|x)[log p(x|z)] − DKL(q(z|x)∥p(z)),

where the first term represents the reconstruction loss, ensuring that the generated data resembles the input, and the second term is the Kullback-Leibler divergence, which regularizes the latent space.

A key point of VAEs is the reparameterization trick, which facilitates gradient-based optimization by expressing the latent variable z as:

z = µ + σ · ϵ, ϵ ∼ N(0,I),

where µ and σ are outputs of the encoder network.

VAEs have found applications in areas such as image synthesis, data compression, and representation learning due to their ability to generate diverse, high-quality samples while maintaining interpretability of the latent space. In our work, we employ a VAE as the backbone model and introduce two submodules to decouple the video latent representation effectively.

