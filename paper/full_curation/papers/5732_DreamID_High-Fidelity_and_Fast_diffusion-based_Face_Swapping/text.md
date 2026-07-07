## DreamID: High-Fidelity and Fast diffusion-based Face Swapping via Triplet ID Group Learning

Fulong Ye*, Miao Hua*, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao†, Qian He†, Xinglong Wu Intelligent Creation Team, ByteDance https://superhero-7.github.io/DreamID/

# arXiv:2504.14509v3[cs.CV]25Apr2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Makeup Preservation Large Angles Stylization Complex Lighting Condition

Figure 1. DreamID can generate high fidelity face swapping results at 512 × 512 resolution. In each group, we present the swapped face on the right, which is created by replacing the source face (top-left) with the target face (bottom-left). Our model is capable of generating high-similarity face swaps and performs exceptionally well in a variety of challenging scenarios, including makeup preservation, large angles, stylization, and complex lighting conditions.

#### Abstract

the Triplet ID Group explicit supervision. Finally, to further extend our method, we explicitly modify the Triplet ID Group data during training to fine-tune and preserve specific attributes, such as glasses and face shape. Extensive experiments demonstrate that DreamID outperforms state-of-the-art methods in terms of identity similarity, pose and expression preservation, and image fidelity. Overall, DreamID achieves high-quality face swapping results at 512×512 resolution in just 0.6 seconds and performs exceptionally well in challenging scenarios such as complex lighting, large angles, and occlusions.

In this paper, we introduce DreamID, a diffusion-based face swapping model that achieves high levels of ID similarity, attribute preservation, image fidelity, and fast inference speed. Unlike the typical face swapping training process, which often relies on implicit supervision and struggles to achieve satisfactory results. DreamID establishes explicit supervision for face swapping by constructing Triplet ID Group data, significantly enhancing identity similarity and attribute preservation. The iterative nature of diffusion models poses challenges for utilizing efficient imagespace loss functions, as performing time-consuming multistep sampling to obtain the generated image during training is impractical. To address this issue, we leverage the accelerated diffusion model SD Turbo, reducing the inference steps to a single iteration, enabling efficient pixel-level end-to-end training with explicit Triplet ID Group supervision. Additionally, we propose an improved diffusion-based model architecture comprising SwapNet, FaceNet, and ID Adapter. This robust architecture fully unlocks the power of

#### 1. Introduction

Face swapping is a highly challenging task that aims to transfer identity-related information from a source image to a target image while preserving the attribute information of the target image, such as background, lighting, expression, head pose, and makeup.

Early face swapping research primarily focused on GAN-based methods [2, 3, 15, 22], which encountered two primary challenges. First, the training process tends to be unstable and requires extensive hyperparameter search.

*Equal contribution. †Corresponding Author.

Second, the generated images often suffer from low fidelity and various artifacts, particularly in scenarios with large angles and the edges of the face shape. Addressing these challenges within the GAN framework is particularly difficult. Recently, diffusion models [26, 27, 29? ? ] have achieved remarkable success in image generation, demonstrating significant advantages in image fidelity and diversity. Based on this, recent studies [1, 10, 17, 38] introduce the diffusion model to the face swapping task. While these approaches have significantly improved the quality of image generation, they still fail to achieve satisfactory face swapping results. The primary challenge in face swapping lies in the absence of real ground truth, that is, it is difficult to find a “real” swapped image for a given {source image Xs,target image Xt} pair. As illustrated in Figure 2(a), previous works primarily inject ID and attribute information into the diffusion model through implicit supervision via an ID loss with the source image(when Xs ̸= Xt) and a reconstruction loss with the target image(when Xs = Xt). Due to the lack of explicit supervision, they struggle to achieve high ID similarity and retain fine-grained attribute details such as lighting and makeup.

To overcome these limitations, this paper proposes an accurate and explicit supervised training framework for the face swapping task by constructing Triplet ID Group data to enhance both ID similarity and attribute retention capabilities of face swapping models. Specifically, as shown in Figure 2(b), we prepare two images sharing the same ID (A1, A2), and a third image has a different ID (B). We use a GAN-based proxy face swapping model to swap the A2’s face on B and get a pseudo target image B˜. The Triplet ID Group is (source A1, pseudo target B˜, ground truth A2). Here, A2 has the same ID information as A1 and the same attribute information as B˜. This makes A2 an ideal target for face swapping. Specifically, when A1 serves as the source and B˜ serves as the target, the theoretical face swapping Ground Truth is A2. After constructing the Triplet ID Group data, it is necessary to find an appropriate loss function for end-to-end training. The iterative nature of diffusion models poses challenges for utilizing various practical image-space loss functions, such as ID loss and reconstruction loss. Specifically, incorporating these losses requires accumulating gradients across multiple denoising steps during training, which is computationally expensive. To address this issue, we leverage the recent accelerated diffusion model SD Turbo [30], which reduces the inference steps to just one. This allows us to employ image space loss functions efficiently and significantly improve inference speed.

Additionally, we propose an improved diffusion model architecture for face swapping. Our model architechture is composed of three components:1) the base Unet which we refer to as SwapNet, is responsible for the main process of face swapping. 2) the face Unet feature encoder, named

FaceNet, which extracts pixel-level ID information of the user image. 3) the ID Adapter that extracts the semanticlevel ID information of the user image. Finally, to further extend our method, we explicitly modify the Triplet ID Group data during training to fine-tune and preserve specific attributes, such as glasses and face shape.

###### Faceswap Proxy

###### (Init from FaceDancer)

[Figure 13]

[Figure 14]

[Figure 15]

Proxy Model

: target face : source face B

(Change ID & Keep Attributes)

[Figure 16]

Attribute Extractor

2

[Figure 17]

[Figure 18]

Attribute Extractor

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Extractor

Identity

[Figure 25]

if = if ≠

Identity

Typical Face Extractor Swapping Model

[Figure 26]

[Figure 27]

1

[Figure 28]

[Figure 29]

(a) Implicit supervison (b)Explicit supervison

Figure 2. (a) The typical face-swapping training process, which often relies on implicit supervision. (b) Unlike previous work, DreamID constructs Triplet ID Group data for explicit supervision.

We conduct extensive testing on the FFHQ [19] dataset, and the results indicate that our method outperforms previous state-of-the-art (SOTA) methods in terms of ID similarity, pose, expression preservation, and image quality from quantitative metrics. Moreover, qualitative results show that our method performs exceptionally well in challenging face swapping scenarios such as occlusion, complex lighting and large angles. Overall, due to stronger supervision and a more robust model structure, our model significantly surpasses previous methods and can generate 512×512 fidelity face swapping results with high similarity and attribute preservation, while requiring only one step in inference that takes 0.6 seconds. In summary, our contributions are as follows:

- • We propose a novel precise and explicit supervision face swapping training framework by constructing Triplet ID Group data. In this framework, we employ accelerated diffusion models to reduce the number of inference steps to 1. This enables the effective utilization of image-space loss functions for end-to-end training with the Triplet ID Group. As a result, we significantly improve ID similarity and attribute preservation, while also substantially enhancing inference speed.
- • We introduce an improved diffusion-based model architecture comprising three modules: SwapNet, FaceNet, and ID Adapter. This robust architecture fully unlocks the power of the Triplet ID Group explicit supervision.
- • Extensive experimental results demonstrate that our DreamID significantly outperforms previous methods in terms of ID similarity, attribute preservation, and image fidelity.

#### 2. Related Work

##### 2.1. GAN based Face Swapping Model

Due to the powerful generative capabilities of GANs [8], they are widely used for face swapping. DeepFakes [25] was specifically trained to swap faces between paired identities but is limited to those identities. Research following the disentanglement paradigm for subject-agnostic face swapping includes FaceShifter [21], which adaptively integrates identity and attribute embeddings attentively. HifiFace [32] utilizes 3D shape-aware identity features to enhance the quality of swapped faces. SimSwap [2] introduced Weak Feature Matching Loss to improve attribute preservation, while SimSwap++ [3] enhanced model efficiency. FaceDancer [28] introduced an adaptive feature fusion attention (AFFA) module to fuse attribute features adaptively. CSCS [15] and ReliableSwap [37] developed a reverse pseudo-input generation approach for additional training data. Recently, StyleGAN-based models [19, 20] have been adopted for high-resolution face swapping, with MegaFS [39] pioneering the use of StyleGAN2 [20] as a decoder. InfoSwap [7] proposed an identity contrastive loss that better disentangles the StyleGAN latent space. RAFSwap [34] introduced a Region-Aware Face Swapping network for identity-consistent, high-resolution swapping in a local-global manner. StyleSwap [35] designed a novel Swapping-Guided ID Inversion strategy to improve identity similarity. Despite these advances, such methods are prone to artifacts, particularly under large pose variations and occlusions [18]. These challenges highlight the need for a closer examination of alternative approaches.

##### 2.2. Diffusion based Face Swapping Model

Compared to GANs, the training of diffusion models is more stable and can generate higher quality images. Recently, some works have started to explore the use of diffusion models for face swapping. DiffFace [17] is the first to use a diffusion model for face swapping. It trains an ID-Conditional DDPM [11], samples with facial guidance, and uses a target-preserving blending strategy. However, as they use ID features only to train the model and the swapping is entirely done at the inference stage, this approach significantly burdens the inference process, leading to low throughput even at lower resolutions. DiffSwap [38] reformulates face swapping as a conditional inpainting task guided by identity features and facial landmarks. To introduce identity constraints during training, they propose a midpoint estimation method that can generate swapped faces in only 2 steps. However, the method struggles to achieve effective ID transferability due to the blurry inference results. FaceAdapter [10] uniformly models the face swapping and face reenactment tasks. However, it does not perform well on a single task. ReFace [1] frames the face-

swapping problem as a self-supervised, train-time inpainting task. Because the face area of the target image is masked off during the inference process, it becomes difficult to preserve the attributes. In our method, we construct Triplet ID Group data to establish precise and explicit supervision, thereby boosting ID similarity and attribute preservation.

#### 3. Preliminaries

Stable Diffusion Turbo In this paper, we employ Stable Diffusion Turbo(SD Turbo) [30] as our base model.SD Turbo is a distilled version of Stable Diffusion [27], which allows sampling large-scale foundational image diffusion models in 1 to 4 steps at high image quality. SD Turbo is a latent diffusion model that operates in the latent space of an autoencoder D(E(·)), where E and D represent the encoder and decoder, respectively. For a given image x0 with its corresponding latent feature z0 = E(x0), the diffusion forward process is defined as:

zt = √αtz0 + (1 − αtϵ, (1)

where αt = ts=1(1 − βs), ϵ ∼ N(0,1), and βs is the pre-defined variance schedule at timestep s.

#### 4. Method

This chapter will provide a detailed introduction to our diffusion-based face swapping method, called DreamID. In Section 4.1, we present our precise and explicit supervision training framework. We construct Triplet ID Group data to perform pixel-to-pixel level training, significantly improving ID similarity and attribute preservation. Section 4.2 specifically details our improved diffusion-based face swapping model architecture. In section 4.3, we discuss further extensions for our DreamID. An overview of our method is illustrated in Figure 3.

##### 4.1. Triplet ID Group Learning

The face swapping task can be formally defined as follows:

yres = F(xsrc,xtar) {I|yres} = {Iid|xsrc} ∪ {Inid|xtar}

where xsrc is the source image, xtar is the target image, yres is the face swapping result image, and F denotes the face swapping model function. The objective of face swapping is to seamlessly integrate the identity information Iid from the source image with the non-identity attribute information Inid of the target image (e.g., background, expression, head pose, lighting, etc.). The primary challenge in face swapping is the lack of real ground truth. Unlike typical datadriven tasks that use pair data for explicit end-to-end training, finding a “real” swapped image for a given {xsrc,xtar} pair is difficult. As depicted in Figure 2, previous methods

[Figure 30]

[Figure 31]

FaceNet

[Figure 32]

[Figure 33]

###### Faceswap Proxy

... ...

(Init from GAN)

[Figure 34]

Source

퐼

 (ID 1)

Source

ID

cinematic full head portrait

[Figure 35]

[Figure 36]

[Figure 37]

Landmark Enc

id

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

3DMM

... ...

[Figure 43]

[Figure 44]

exp pose

[Figure 45]

SwapNet Generated Image

Target

Pseudo Target

Source

[Figure 46]

ID 2

ID 1

ID

Cross Attention: Self Attention: Concat:

ID Adapter

ReconstructionLoss

Noisy

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

ID Encoder

IDLoss

Add:

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Train: Frozen:

Source

(Init from SD Turbo) ×1

Pseudo Target

[Figure 56]

ID 1

 (퐼   )

[Figure 57]

FaceNet

​Semantic-level ID Features

[Figure 58]

[Figure 59]

Linear

###### ID h Adapter

Cross Attention

LN

h

Pixel-level ID Features

w

[Figure 60]

SwapNet

Self Attention

[Figure 61]

2w

Cross Attention

Generated Image

ID

(a) Triplet ID Group Training

(b) DreamID Model Architecture

Figure 3. Overview of DreamID. (a)Triplet ID Group Training. We establish explicit supervision for face swapping by constructing Triplet ID Group data. The construction process utilizes two images with the same ID(A1, A2) and one image with a different IDB, along with a FaceSwap Proxy model, to generate a Pseudo Target IDB˜. Additionally, we initialize our DreamID with SD Turbo, reducing the inference steps to a single step. This allows for convenient computation of image-space losses, such as ID Loss and reconstruction Loss. (b) DreamID Model architecture. Our model architecture is composed of three components:1) The base Unet, which we refer to as SwapNet, is responsible for the main process of face swapping. 2) the face Unet feature encoder, named FaceNet, which extracts pixel-level ID information of the user image. 3) the ID Adapter that extracts the semantic-level ID information of the user image. The core feature fusion computation process is illustrated at the bottom.

rely on implicit supervision via an ID loss with the source image and a reconstruction loss with the target image. However, this implicit supervision is biased and can lead to improper convergence. To address this, we introduce a novel Triplet ID Group Learning strategy to establish precise and explicit supervision.

Triplet ID Group Construction As illustrated in Figure 3(a), given two images of the same identity IDA

and IDA

1

, and another image with a different identity IDB. We use IDB and IDA

2

as the source image and target image inputs for a GAN face swapping proxy model. Then we generate a Pseudo Target IDB˜ that alters the ID information of IDA

2

while preserving the attribute information. Thus, we can obtain Triplet ID Group as follows:

2

) (2) Consequently, IDA

###### (IDA

###### ,IDB˜,IDA

1

2

shares the same attribute information as IDB˜ and the same identity information as IDA

2

.

1

This means, we have found the paired data for face swapping: when IDA

is employed as the source and IDB˜ is used as the target, then the theoretical Ground Truth is IDA

1

. Thus, we can use the Triplet ID Group data to perform end-

2

to-end training. Notably, the Pseudo Target IDB˜ generated by GAN face swapping proxy model is not our learning tar-

. This ensures that the upper bound of the supervisory signal is very high.

get; instead, our learning target is the real image IDA

2

Triplet ID Group Training Objectives After constructing the Triplet ID Group data, it is necessary to find an appropriate loss function. However, applying effective image-space loss function, such as ID loss and Reconstruction Loss in diffusion models is non-trivial, due to the iterative denoising nature of diffusion models, incorporating these loss functions requires accumulating gradients across multiple denoising steps during training, which is computationally expensive. To takle this issue, we leverage recent fast sampling methods SD Turbo [30] to reduce the number of inference steps to one. Consequently, we can efficiently employ image space loss functions and significantly improve inference speed. Ultimately, we use three loss functions as our training objectives, i.e. the original Diffusion Loss, the ID Loss and the Reconstruction Loss. Next we will introduce the three loss function respectively.

Diffusion Loss Let x = IDA

2 ∈ R3×H×W in the image

space, E as the encoder module in the SD Turbo, C as condition encoder(details in Section 4.2). The diffusion loss [27] function LDM is defined as follows:

0,ϵ,c,t∥ϵ − ϵθ(zt,c,t)∥2. (3) where, z0 = E(x) ∈ R3×H

LDM = Ez

′×W′ in the latent space, ϵ ∼ N(0,I). And zt get from Equation 1, θ is the parameter of UNet to predict the noisy ϵθ (zt,c,t) condition on zt, source IDA

and target IDB˜ condition embedding c = C(IDA

1

,IDB˜) and t. Here we use one step property of SD Turbo for diffusion loss calculating, which means t = 999.

1

ID and Reconstruction Loss Futhermore, we use one step property of SD Turbo to conduct ID and Reconstruct Loss. As shown in Figure 3, IDA

and IDB˜ are used as the source and target input of DreamID respectively, and generated image IDA˜ is obtained through one-step inference. Then we calculate ID loss between IDA˜ and IDA

1

: Lid = 1 − cos eID

1

present ID Embedding obtained from an off the shelf ID encoder [12]. Then we calculate L2 loss between IDA˜ and IDA

, where eID

,eID

,eID

A˜

A˜

A1

A1

: Lrec = ∥IDA

2

2 − IDA˜∥22 . Finally, we get the total loss: L = λidLid + λDMLDM + λrecLrec (4)

where, λid,λDM,λrec adjust the weights between each loss.

##### 4.2. Model Architecture

After establishing the Triplet ID Group Learning framework, the remaining task is to construct a diffusion model structure that is strong enough to fully leverage this robust supervisory signal. In this section, we will discuss our improved diffusion-based model architecture in detail. Our model architecture is composed of three components:1) the base Unet which we refer as SwapNet, is responsible for the main process of face swapping. 2) the face Unet feature encoder, named FaceNet, which extracts pixel-level ID information of the user image. 3) the ID Adapter that extracts the semantic-level ID information of the user image. As shown in Figure 3(b), we will provide detailed explanation of each component as follows.

SwapNet For the base UNet module, we initialize it from SD-Turbo, thereby enabling single-step inference and significantly enhancing the inference speed. As of the input for our base UNet, we concatenate two components as follows: 1) The Facial landmarks, i.e. Ilandmark. We employ a 3D face reconstruction Model [33] to separately extract the identity, expression, and pose coefficients of the source and target image. Subsequently, we recombine the identity coefficients of the source image with the expression and pose coefficients of the target image to reconstruct a new 3D face model, and project it to obtain the corresponding facial

landmarks. During training, we directly use the pseudo target’s landmarks and feed them into a Pose Guider to extract features. This Pose Guider uses simple convolutional layers to align the landmark image with the same resolution

- as the noise latent. Finally, we add this feature to the noise

together. 2) The latent of target image, i.e., E(IDB˜). Directly inputting the latent of the target image into the base

UNet allows the model to directly capture its attribute information. Combined with the Triplet ID Group data, this enables the model to learn attribute preservation in an endto-end manner. Finally, we expand the convolutional layer of UNet to 8 channels initialized with zero weights.

FaceNet We employ a UNet encoder, namely the FaceNet encoder, which inherits weights from the SwapNet module, to encode source images. Given the latent of a source image E(IDA

1

),we pass it through the UNet encoder to obtain a latent intermediate feature, which is then concatenated with the latent intermediate feature from the SwapNet. As shown

- at the bottom of Figure 3(b), we compute the self-attention on the concatenated features and then pass only the firsthalf dimensions from SwapNet(similar to [14]). FaceNet inputs the original user image directly, enabling strong feature extraction directly from the image space. We refer to the extracted features as pixel-level ID features. ID Adapter While FaceNet is capable of extracting strong pixel-level ID features, its strength can sometimes lead to issues such as “copy-paste” artifacts. To balance this, we introduce the ID Adapter, which complements FaceNet by extracting semantic-level ID features. We first use a face ID encoder model to extract face embeddings, then map these embeddings to the same dimensions as the key-value (KV) matrix inputs of the SwapNet using a Liner layer and LayerNorm(LN). We then augment the SwapNet KV matrix with an extra copy of these mapped face embeddings (similar to [36]). Finally, the cross-attention from the ID Adapter and the cross-attention from SwapNet are add together.

During training, we freeze the ID encoder and the 3DMM model[33], while training all other components, including the SwapNet, FaceNet, and ID Adapter. The text prompt “cinematic full head portrait” is used for all training samples. After being encoded by OpenCLIP-VIT/H[16], the text prompt is injected into SwapNet and FaceNet through cross-attention.

##### 4.3. Extention

In our approach, we can control specific features by explicitly modifying the Triplet ID Group data, which enables various real-world applications. For illustration, we provide two examples involving face glasses and shape transfer. As shown in Figure 4, if we want to keep the glasses from the user image, we first filter out the data with the same glasses for IDA

, and then use the post-processing model [23] to remove the glasses of the pesudo target. This en-

and IDA

1

2

##### 5.2. Main Results

forces the model to preserve the glasses attribute from the user image. Similarly, for face shape transfer, we can employ a face shape post-processing model [31] to alter the Pseudo Target’s face shape, thereby promoting the transfer of face shape from the user image.

Quantitative Evaluation In Table 2, we compare with SoTA methods quantitatively on FFHQ test set, including Inswapper [13], SimSwap [2], CSCS [15], DiffFace [17], DiffSwap [38], FaceAdapter [10], REFace [1]. DreamID outperforms previous methods across all metrics. It achieves an FID score of 4.69, indicating that our model generates higher-fidelity images. In terms of ID similarity, our model achieves a score of 0.71, demonstrating that DreamID can effectively transfer ID information. Furthermore, DreamID achieves a Pose score of 2.20 and an Expression score of 0.789, indicating its excellent ability to preserve attributes.

ID ID 1 ID ID 2

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|Model<br><br>|FID↓<br><br>ID Similarity↑<br><br>ID Retrieval (Top-1/Top-5)↑<br><br>Pose↓ Expression↓|
|---|---|
|Inswapper[13] SimSwap[2] CSCS[15] FaceDancer[28] DiffFace[17] DiffSwap[38] FaceAdapter[10] REFace[1]|8.03 0.65 99.20%/99.90% 2.74 1.51 19.77 0.55 95.24%/97.09% 3.21 1.742 10.17 0.68 99.10%/99.80% 3.81 1.493<br><br>4.91 0.48 92.70%/97.20% 2.32 0.854<br><br>8.66 0.51 93.40%/97.60% 3.78 1.280<br><br>8.65 0.32 66.50%/46.17% 2.84 1.084<br><br>9.39 0.52 93.50%/97.60% 4.15 1.188<br><br><br>5.58 0.57 96.50%/99.20% 3.77 1.040<br><br><br>|
|DreamID|4.69 0.71 99.9%/100% 2.20 0.789|

Figure 4. Data construction for specific feature control.

#### 5. Experiment 5.1. Setup

Datasets and Metrics Our training data is sourced from VGGFace2-HQ [3] and Arc2Face [24], with both datasets containing multiple images per ID. We filtered these images based on clarity and similarity, ultimately selecting approximately 500,000 high-quality samples. We followed prior work [15], obtaining 1,000 source and 1,000 target images from FFHQ [19], and generating 1,000 swapped images. We evaluated fidelity by calculating the FID between the face-swapped images and the real images from the FFHQ dataset. For pose and expression evaluation, we utilized HopeNet [6] and Deep3DFaceRecon [5], respectively, and compared the target and swapped images using L2 distance. We used ArcFace [4] to extract ID embeddings and evaluated similarity by calculating the cosine distance between the ID embeddings of the face-swapped images and the source images. Additionally, we performed face retrieval by searching for the most similar faces among all the source images using cosine similarity as the measurement, and calculated the Top-1 and Top-5 accuracy. Both training and testing were conducted at a resolution of 512×512.

Implementation Details We implemented our model using PyTorch, and all experiments were conducted on 8 NVIDIA A100 GPUs (80GB). Starting from SD Turbo, we trained the model with a learning rate of 1e-5 and a batch size of 8. The training process encompassed 70,000 steps, taking approximately three days. The weights for ID loss, diffusion loss, and reconstruction loss were set to 1, 1, and 10, respectively. For ID embedding and ID loss computation, we used Glint36k [12] as our ID encoder.

- Table 1. Quantitative compare with SOTAs on the FFHQ. DreamID outperforms previous methods in all metrics, demonstrating its superiority in fidelity, ID similarity and attribute preservation.

Method DiffFace[17]DiffSwap[38] Face Adapter[10] REFace[1]DreamID Single Inference Speed 25.8s 7.82s 3.42s 3.75 0.6s

- Table 2. Comparison of inference time of diffusion based models.

Model FID↓ ID Similarity↑ Pose↓ Expression↓

Full Model w/o IDA 5.59 0.70 2.66 0.867 Full Model w/o FaceNet 3.69 0.63 2.09 0.740 Full Model w/o ID Loss 3.13 0.32 1.63 0.619 Full Model 4.69 0.71 2.20 0.789

- Table 3. Ablation study about model architecture and training strategy.

Qualitative Evaluation As shown in Figure 5, we performed qualitative comparisons with SoTAs on the FFHQ and Web data. DreamID demonstrates significant advantages in terms of similarity, natural blending, occlusion handling, and attribute preservation such as expression, lighting, and makeup. As shown in row 1, nearly all other models introduce artifacts in the bangs area of the hair and cause significant interference with expressions. In contrast, our results blend very naturally without introducing any noise, and the expressions are well-preserved. For occlusion handling, as seen in row 2, almost all other models fail to preserve occlusions effectively, whereas our model nearly

Source Target Inswapper SimSwap CSCS FaceDancer DiffFace DiffSwap FaceAdapter REFace DreamID

[Figure 70]

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

ResultsonFFHQResultsonWebData

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

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

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

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

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

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

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Figure 5. Qualitative comparison of state-of-the-art methods on the FFHQ dataset. DreamID demonstrates significant advantages in terms of similarity, natural blending, occlusion handling, and attribute preservation such as expression, lighting, and makeup.

model without FaceNet module. Removing ID Adapter will slightly reduce ID similarity, and both Pose and Expression will also decrease. Removing FaceNet will significantly reduce ID similarity but increase Pose and Expression. By visualizing the observations in the Figure 6, the FaceNet is more capable of extracting pixel-level ID information, but it is more likely to have “copy paste” pattern. In contrast, ID Adapter is relatively weaker in feature extraction but excels at extracting semantic-level ID information. By combining the two, we can leverage the strong feature extraction capabilities of FaceNet while using ID Adapter to encourage the extraction of more essential ID features, thereby preventing the extraction of non-ID information such as expressions.

perfectly maintains the occluded parts. Our method performs exceptionally well in preserving various fine-grained properties of the target image. For example, gaze in row 3, lighting in rows 4 and 8, and makeup in row 7. Even with large side profiles, our model still generates excellent results, which is a significant challenge for previous faceswapping models, where almost all other methods fail to produce reasonable outcomes. Overall, our model significantly outperforms others in terms of similarity, which is particularly noticeable in the fifth row.

Inference Speed We test the inference speed on a NVIDIA A100. Our method only takes 0.6s for a single inference, which is much faster than other diffusion based methods.

Ablations on Training Strategies As shown in Figure 6, removing the Reconstruction Loss (Rec Loss) results in unreasonable outputs. This is related to the inherent strength of diffusion models, which have a tendency to easily perform copy-pasting. Therefore, it is crucial to balance ID loss and Rec Loss against each other. If ID loss is removed, the similarity significantly decreases, as shown in Table 3, indicating the importance of ID loss. In summary, both

##### 5.3. Ablation

We conducted ablation studies on the model architecture and training strategies.

Ablations on Model Architecture As shown in the Table 3, “Full Model” represents the model with both FaceNet and ID Aadapter module. “w/o IDA” represents the model without ID Adapter module, and “w/o FaceNet” represents the

Source Target Full Model w/o ReferenceNet w/o IPAdapter w/o ID Loss w/o Rec Loss

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Figure 6. Visualization of ablation studies of model architecture and training strategy.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Figure 7. Results on out of domain data. DreamID effectively preserves the texture features of the template image, enabling it to generate high-quality results even in styles outside the domain of real human.

scores (2.20/0.789) are better than those of FaceDancer (2.32/0.854). This indicates that the Triplet ID Group training, with real images serving as ground truth, enables the trained model to break through the upper bound of attribute preservation of the proxy model.

losses are complementary and indispensable.

Model FID↓ ID Similarity↑ Pose↓ Expression↓

DreamID(w Inswapper) 5.89 0.72 2.90 0.975 DreamID(w FaceDancer) 4.69 0.71 2.20 0.789

Table 4. Ablation on proxy model selection.

Source Target Origin DreamID After Finetuned

Ablations on Proxy Model In our Triplet ID Group training framework, we discovered an intriguing property: ID similarity largely depends on the ID loss between the generated image IDA˜ and the source image IDA

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

, while attribute preservation largely depends on the reconstruction loss between the IDA˜ and IDA

1

. This means that the better the Pseudo Target retains the attributes of IDA

2

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

, the better the attribute preservation capability of the trained model after learning through the reconstruction loss. In other words, we need a proxy model that excels in attribute preservation, even if its ID transfer capability is not as critical. With this in mind, we compared the effects of using two different models as the proxy model. FaceDancer has good attribute retention but low ID similarity, while Inswapper is the opposite. As shown in Table 4, the FID, Pose, and Expression of the model trained with FaceDancer are significantly better than those trained with Inswapper, while only slightly reducing the ID similarity by about 1 percentage point(statistically insignificant). Therefore, we ultimately chose FaceDancer as our proxy model. Another interesting observation is that DreamID’s pose/expression

2

Figure 8. Illustrate results of Feature-Specific Control Finetune.

Feature-Specific Control Finetune As shown in Figure 8, after fine-tuning the model as mentioned in Section 4.3, both face shape and glasses can be transferred.

Out of Domain Results One property of DreamID is its ability to effectively preserve the texture features of the target image, which enables it to generate high-quality results in styles outside the real human domain, such as sketches, oil paintings, watercolors, etc., as shown in Figure 7.

#### 6. Conclusion

In this paper, we introduced DreamID, a diffusion-based face swapping method that achieves high identity similarity, attribute preservation, and fast inference through explicit supervision via Triplet ID Group data and an improved diffusion model architecture. Extensive experiments demonstrate its superior performance over existing methods. DreamID represents a significant advancement in face swapping, sets a new paradigm that is remarkably simple and effective, offering high-quality results with rapid inference speed.

#### References

- [1] Sanoojan Baliah, Qinliang Lin, Shengcai Liao, Xiaodan Liang, and Muhammad Haris Khan. Realistic and efficient face swapping: A unified approach with diffusion models. arXiv preprint arXiv:2409.07269, 2024. 2, 3, 6, 1
- [2] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In MM ’20: The 28th ACM International Conference on Multimedia, 2020. 1, 3, 6
- [3] Xuanhong Chen, Bingbing Ni, Yutian Liu, Naiyuan Liu, Zhilin Zeng, and Hang Wang. Simswap++: Towards faster and high-quality identity swapping. IEEE Trans. Pattern Anal. Mach. Intell., 46(1):576–592, 2024. 1, 3, 6
- [4] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 6
- [5] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019. 6
- [6] Bardia Doosti, Shujon Naha, Majid Mirbagheri, and David J Crandall. Hope-net: A graph-based model for hand-object pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6608– 6617, 2020. 6
- [7] Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, and Ran He. Information bottleneck disentanglement for identity swapping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3404–3413, 2021. 3, 1
- [8] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2014. 3
- [9] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in neural information processing systems, 37:36777–36804, 2024. 1

- [10] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangning Zhang, Chengjie Wang, and Yong Liu. Face adapter for pre-trained diffusion models with fine-grained id and attribute control. arXiv preprint arXiv:2405.12970, 2024. 2, 3, 6, 1
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851. Curran Associates, Inc., 2020. 3
- [12] https://github.com/deepinsight/insightface/tree/master/model zoo. 5, 6

- [13] https://github.com/haofanwang/inswapper. 6, 1
- [14] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 5
- [15] Ziyao Huang, Fan Tang, Yong Zhang, Juan Cao, Chengyu Li, Sheng Tang, Jintao Li, and Tong-Yee Lee. Identitypreserving face swapping via dual surrogate generative models. ACM Transactions on Graphics, 43(5):1–19, 2024. 1, 3, 6
- [16] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 5
- [17] S. Cho J. Seo J. Nam K. Lee S. Kim K. Lee K. Kim, Y. Kim. Diffface: Diffusion-based face swapping with facial guidance. 2022. 2, 3, 6, 1
- [18] Amina Kammoun, Rim Slama, Hedi Tabia, Tarek Ouni, and Mohmed Abid. Generative adversarial networks for face generation: A survey. ACM Computing Surveys, 55:1 – 37,

2022. 3

- [19] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 4401–4410. Computer Vision Foundation / IEEE,

2019. 2, 3, 6

- [20] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 8107–

8116. Computer Vision Foundation / IEEE, 2020. 3

- [21] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Advancing high fidelity identity swapping for forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5074– 5083, 2020. 3
- [22] Zhian Liu, Maomao Li, Yong Zhang, Cairong Wang, Qi Zhang, Jue Wang, and Yongwei Nie. Fine-grained face swapping via regional gan inversion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8578–8587, 2023. 1

- [23] Junfeng Lyu, Zhibo Wang, and Feng Xu. Portrait eyeglasses and shadow removal by leveraging 3d synthetic data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3429–3439, 2022. 5
- [24] Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos Zafeiriou. Arc2face: A foundation model for id-consistent human faces. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 6
- [25] Ivan Perov, Daiheng Gao, Nikolay Chervoniy, Kunlin Liu, Sugasa Marangonda, Chris Um´e, Mr. Dpfks, Carl Shift Facenheim, Luis RP, Jian Jiang, Sheng Zhang, Pingyu Wu, Bo Zhou, and Weiming Zhang. Deepfacelab: A simple, flexible and extensible face swapping framework. CoRR, abs/2005.05535, 2020. 3
- [26] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3, 5
- [28] Felix Rosberg, Eren Erdal Aksoy, Fernando AlonsoFernandez, and Cristofer Englund. Facedancer: Pose-and occlusion-aware high fidelity face swapping. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3454–3463, 2023. 3, 6, 1
- [29] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [30] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2025. 2, 3, 4, 1

- [31] Scott Schaefer, Travis McPhail, and Joe Warren. Image deformation using moving least squares. In ACM SIGGRAPH 2006 Papers, pages 533–540. 2006. 6
- [32] Yuhan Wang, Xu Chen, Junwei Zhu, Wenqing Chu, Ying Tai, Chengjie Wang, Jilin Li, Yongjian Wu, Feiyue Huang, and Rongrong Ji. Hififace: 3d shape and semantic prior guided high fidelity face swapping. In IJCAI, pages 1136–1142. ijcai.org, 2021. 3
- [33] Zidu Wang, Xiangyu Zhu, Tianshuo Zhang, Baiqin Wang, and Zhen Lei. 3d face reconstruction with the geometric guidance of facial part segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1672–1682, 2024. 5
- [34] Chao Xu, Jiangning Zhang, Miao Hua, Qian He, Zili Yi, and Yong Liu. Region-aware face swapping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7632–7641, 2022. 3
- [35] Zhiliang Xu, Hang Zhou, Zhibin Hong, Ziwei Liu, Jiaming Liu, Zhizhi Guo, Junyu Han, Jingtuo Liu, Errui Ding, and

- Jingdong Wang. Styleswap: Style-based generator empowers robust face swapping. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 3
- [36] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 5

- [37] Ge Yuan, Maomao Li, Yong Zhang, and Huicheng Zheng. Reliableswap: Boosting general face swapping via reliable supervision. arXiv preprint arXiv:2306.05356, 2023. 3, 1
- [38] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie Zhou, and Jiwen Lu. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. CVPR,

2023. 2, 3, 6, 1

- [39] Yuhao Zhu, Qi Li, Jian Wang, Chengzhong Xu, and Zhenan Sun. One shot face swapping on megapixels. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pages 4834–4844, 2021. 3

## DreamID: High-Fidelity and Fast diffusion-based Face Swapping via Triplet ID Group Learning

### Supplementary Material

#### A. Ablations about diffusion steps

We trained two DreamIDs with different inference steps one and four, which are properties inherited from the SD Turbo[30]. As shown in Table 5, we tested the qualitative indicators of both the 1 and 4 step models. We found that the performance difference between them was not significant, but the inference time of the 1-step model was substantially shorter. Therefore, we ultimately chose to use the 1-step model.

ID Similarity↑

ID Retrieval (Top-1/Top-5)↑

Inference Time↓

Model FID↓

Pose↓ Expression↓

DreamID-4step 4.69 0.71 99.9%/100% 2.20 0.789 1.3s DreamID-1step 5.08 0.71 99.9%/100% 2.31 0.790 0.6s

Table 5. Comparison of one step and four step model.

shown on Figure 18, DreamID-Stylization perform well on stylized target images, such as 3D and cartoons. As shown on Figure 19, DreamID can even perform quite well on stylized user images and stylized target images.

#### B. More Results on Image

We show more image results on Figure 9. Our model performs very well in stylized scenes, such as oil paintings, watercolors, sketches, etc. It also handles occlusion effectively, as demonstrated by the flowers in the third row. Additionally, the model excels in preserving light and shadow details. These are all things that previous models could not do, whether it is GAN-based [2, 3, 7, 13, 22, 28, 37] or diffusion-based [1, 10, 17, 38] model.

#### C. DreamID Family

Our Triplet ID Group confers high flexibility on the face swapping task. By flexibly constructing these triplets, we can train models with different characteristics. Based on this, we have trained several models, including DreamIDHigh Similarity, DreamID-High Attribute Preservation, and DreamID-Stylization. Figure 9 show the characteristic of DreamID. DreamID-High Similarity is capable of generating extremely high similarity results, overcoming the problem that traditional face-swapping models are unable to achieve face transformation. DreamID-High Attribute Preservation can effectively preserve fine-grained attribute information, such as lighting/cosmetics, and performs well in handling large angles and occlusions. We show more results of DreamID-High Similarity on Figure 11-17. Previous face swapping work has primarily focused on real human images, with few models supporting stylization such as 3D or cartoon images. We construct stylized triplet ID data by utilizing Pulid[9] to train DreamID-Stylization. As

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

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

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

###### Figure 9. More results of DreamID-High Attribute Preservation.

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

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Figure 10. The characteristic of DreamID. DreamID-High Similarity is capable of generating extremely high similarity results, overcoming the problem that traditional face-swapping models are unable to achieve face transformation. DreamID-High Attribute Preservation can effectively preserve fine-grained attribute information, such as lighting/cosmetics, and performs well in handling large angles and occlusions.

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

###### Figure 11. More results of DreamID-High Similarity.

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

###### Figure 12. More results of DreamID-High Similarity.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

###### Figure 13. More results of DreamID-High Similarity.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

###### Figure 14. More results of DreamID-High Similarity.

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

###### Figure 15. More results of DreamID-High Similarity.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

###### Figure 16. More results of DreamID-High Similarity.

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

- Figure 17. More results of DreamID-High Similarity.

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

Figure 18. Results of DreamID-Style. DreamID can even perform quite well on stylized user images and stylized target images.

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Figure 19. Results of DreamID-Style. DreamID can even perform well on stylized target images, such as 3D and cartoons. This was something that past models were unable to achieve.

