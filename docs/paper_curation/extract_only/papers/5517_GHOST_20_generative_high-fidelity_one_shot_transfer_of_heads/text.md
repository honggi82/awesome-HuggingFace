# GHOST 2.0: Generative High-fidelity One Shot Transfer of Heads

## Alexander Groshev∗1 Anastasiia Iashchenko∗1 Pavel Paramonov∗1 Denis Dimitrov∗∗1,2 Andrey Kuznetsov∗∗1,2 1Sber AI 2AIRI

arXiv:2502.18417v4[cs.CV]9Jun2025

Source Target Result

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

Figure 1. Results of GHOST 2.0 model on the task of head swap. Head from source image is animated in correspondence with the target motion and blended into target background.

## Abstract

While the task of face swapping has recently gained attention in the research community, a related problem of head swapping remains largely unexplored. In addition to skin color transfer, head swap poses extra challenges, such as the need to preserve structural information of the whole head during synthesis and inpaint gaps between swapped head and background. In this pa-

per, we address these concerns with GHOST 2.0, which consists of two problem-specific modules. First, we introduce enhanced Aligner model for head reenactment, which preserves identity information at multiple scales and is robust to extreme pose variations. Secondly, we use a Blender module that seamlessly integrates the reenacted head into the target background by transferring skin color and inpainting mismatched regions. Both modules outperform the baselines on the corresponding tasks, allowing to achieve state-of-theart results in head swapping. We also tackle complex cases, such as large difference in hair styles of source and target.

## 1. Introduction

The use of virtual humans has long passed purely entertainment scope, finding applications in movie and advertisement composition, virtual try-on, deepfake detection, and portrait editing. Head swap, the task of replacing head in the target image with head from the source image, plays an integral role for these use-cases. It requires reenacting source head with the driving one, and seamlessly integrating the result with the target‘s background. Active research is conducted on generation and animation of head avatars, mainly using warping, generative adversarial networks (GANs) and diffusion models (Siarohin et al., 2019; Drobyshev et al., 2022; Zhang et al.; Kirschstein et al., 2023). However, the problem of blending the generated head with the surrounding environment remains largely unaddressed.

Recently, a noticeable progress in a related task of face swapping has been made (Chen et al., 2020; Wang et al., 2021; Groshev et al., 2022; Zhu et al., 2021; Zhao et al., 2023). The task requires source identity preservation and reenactment only within facial region. Since information about head shape and hair is omitted, face swap is a less complex problem than swapping of the whole head for sev-

∗ indicates equal contribution ∗∗ Corresponding authors: Andrey Kuznetsov

<kuznetsov@airi.net>, Denis Dimitrov <dimitrov@airi.net>

eral reasons. First, face recognition models (Cao et al., 2018; Deng et al., 2019; Alansari et al., 2023), usually used to control identity transfer, are trained on narrow face crops. Thus, they cannot be used to embed identity information about the whole head, requiring consideration of other models for this purpose. Secondly, head generation also involves generation of hair, which is a complex task due to its highfrequency texture and diverse styles. Finally, variations in face shapes are usually smaller than variations in head forms. This implies potentially larger regions for inpainting as a result of head replacement. And, similarly to face swap, the skin color of target should be carefully transferred to the generated head for a realistic result.

Currently there are only few papers that address the task of head swap. The first steps were taken by DeepFaceLab (Perov et al., 2020), which requires finetuning for each source identity. Next work, StylePoseGAN (Sarkar et al., 2021), tends to modify background and skin color of the target. A seminal work that partially tackles the aforementioned issues is HeSer (Shu et al., 2022). It splits the generation process into head reenactment stage and referencebased blending stage. This work was followed by diffusionbased approaches (Baliah et al., 2025; Han et al., 2023). However, latest methods still face issues concerning quality of head generation and color transfer.

In this paper we present GHOST 2.0, a one-shot high-fidelity framework for head swap. It consists of two modules: Aligner for head reenactment, and Blender for natural inpainting of the result into the target background. Aligner includes a set of encoders to obtain information on source identity and target motion, which is then used to condition StyleGAN-based generator to obtained aligned head. The choice of architecture ensures good in-the-wild performance, supporting high-quality generation even in extreme poses. Blender creates references for head color transfer and background inpainting, and uses them to stitch the generated head with the background via blending UNet (Ronneberger et al., 2015). For color transfer, the composition of reference is based on correlation between respective head parts of generated and driving head. Background reference is determined by masking the potential gaps arising due to differences in head shape. These references are used to condition UNet which outputs final result.

Our contributions are the following:

• We introduce a new model for head reenactment that surpasses competitors on a range of metrics. In contrast to previous works that focus on face region only, it generates full human head, accounting both for lowfrequency and high-frequency details and preserving identity at different scales. Moreover, when trained on the same dataset as other methods, it gives superior results on generation in extreme poses.

- • We increase quality of final generations by improving robustness of blending module to corner-case scenarios. Specifically, these include the situation when hair styles of source and target are extremely different, which previously led to inconsistent hair transfer. Additionally, we consider the case when target head lacks color references for the source one, resulting in poor color transfer. Finally, we enhance background inpainting to achieve more seamless blending.
- • We trained a new segmentation model specifically for the head swap task. Unlike existing segmentation models, we have annotated the data in such a way as to separate beard and facial hair into a separate class, which is necessary for correct color transfer.

## 2. Related Work

Face swap There are a large number of face-swapping methods. Conceptually, they can be divided into several different groups. Methods from the first group (Chen et al., 2020; Wang et al., 2021; Groshev et al., 2022) extract the identity vector and some other features of the source face and use a generative model to blend them with the attributes of the target. Often, such models rely on the ArcFace (Deng et al., 2019) model and a 3D shape-aware identity extractor, which allows encoding the 3D geometry of the face. There are also approaches (Zhu et al., 2021) based on StyleGAN2 (Karras et al., 2020). They propose inverting the source and driving images into the latent space and feeding them into the StyleGAN2 generator to perform the swap. This approach allows for higher-resolution results but is sensitive to input data and does not perform well on strong rotations or small details of images. With the development of diffusion models, approaches to face replacement using this method have emerged (Zhao et al., 2023; Chen et al., 2024). Diffusion models allow for high-quality results, but they are typically slow and require significant computational power and sufficient VRAM.

Head swap The task of head swap is covered by a limited number of works. DeepFaceLab (Perov et al., 2020) is the first approach enabling this capability. However, it requires a large amount of source data for training, and poorly performs color transfer and fusion of generated head with the background. StylePoseGAN (Sarkar et al., 2021) performs head swap by conditioning StyleGAN (Karras et al., 2020) on pose and combined texture map, with body parts taken from target and head — from source. Still, it tends to modify the background and skin color of the target. HeSer (Shu et al., 2022) tackles these issues by designing a separate module for each task. First, it uses head reenactment model based on (Burkov et al., 2020) to align source head with target in pose and expression. In the second stage a refer-

ence is created for skin color transfer based on correlation between pixels from the same head parts. Together with background inpainting prior, it is used to condition blending UNet (Ronneberger et al., 2015) which fuses the head with background. While this method outperforms the competitors, it suffers from identity leakage of target and is unable to color head parts present in source image but absent in the driving one. While previous methods are based on GANs (Goodfellow et al., 2014), there have been attempts (Baliah et al., 2025; Han et al., 2023; Wang et al., 2022a) to use diffusion models instead (Ho et al., 2020). However, these approaches face issues of pose controllability, preservation of target skin color and overall realism of generated head.

Head reenactment Head reenactment methods can be generally categorized as either warping-based (Siarohin et al., 2019; Zakharov et al., 2020b; Drobyshev et al., 2022; Zhang et al.; Wang et al., 2022c) or reconstruction-based (Zielonka et al., 2022; Li et al., 2023; Qian et al., 2024; Chu et al., 2024; Deng et al., 2024). Warping-based approaches utilize motion and facial expression descriptors of the target to deform source image. These descriptors can be based on keypoints (Siarohin et al., 2019; Zakharov et al., 2020b), blendshapes from parametric models (Ren et al., 2021; Yin

- et al., 2022) or latent representations. The latter usually achieves better expressiveness, however, it requires careful disentanglemet of motion from the appearance of the target. This can be achieved via the use of special losses (Drobyshev et al., 2022; 2024; Pang et al., 2023) or additional regularization embedded into the architecture (Pang
- et al., 2023). However, warping-based approaches generally perform well only if the difference between source and target poses is small. Reconstruction-based methods (Zielonka et al., 2022; Li et al., 2023; Qian et al., 2024; Chu et al., 2024; Deng et al., 2024) construct latent model of source head, and therefore are robust to larger pose deviations. These methods often utilize implicit representations such as TriPlanes (Ma et al., 2023; Ye et al., 2024) and NeRF (Zielonka et al., 2022; Zheng et al., 2022; Bai et al., 2023), or explicit ones, such as voxels (Xu et al., 2023), point clouds (Zheng et al., 2023) and meshes (Khakhulin et al., 2022; Grassal et al., 2022), with particularly photorealistic results achieved recently with Gaussian splatting (Qian et al., 2024; Giebenhain et al., 2024). However, reconstruction with these approaches requires an additional step of per-frame estimation of camera parameters, which increases runtime. Also, due to high computational cost of rendering, the resolution of output images does not exceed 256 × 256 and upsampling to higher resolutions is performed by an additional network.

## 3. Approach

Our pipeline consists of two modules, Aligner and Blender. Aligner is used to perform cross-reenactment by transfer-

ring target motion to source head. Several encoders embed relevant information from input images at different scales, which is then fused in decoder network. Positionally aligning both heads allows to perform further blending. It is based on identifying regions that require inpainting, and construction of color references for them. Color reference for head is obtained via correlation learning, while for background we use LaMa inpainting network (Suvorov et al., 2021). They are supplied to UNet network, which performs final blending of reenacted head into the target background.

[Figure 13]

Figure 2. Aligner architecture. Two appearance encoders Epor and Eid take images of source head IS and face Crop(IS) and produce embeddings fpor and fid. Motion encoder Emotion is used to obtain respective embedding fmtn from the target image IT. The embeddings fpor, fid and fmtn are concatenated and used to condition generator G via AdaIN (Huang & Belongie, 2017) layers. The generator takes a learnable tensor T as input and outputs reenacted head IA and binary mask Mreenact.

### 3.1. Aligner

As we target model usage for in-the-wild scenario, we have chosen the reconstruction approach to face reenactment to increase robustness to large pose variations. We decided to use a simple architecture for a faster and more lightweight solution. Thus it is based on 2D instead of 3D volumetric representations to remove the need for camera estimation, rendering and additional upsampling to higher resolution. However, in principle any reenactment model can be used at this stage, provided it reconstructs the whole head.

Aligner architecture Aligner module, based on (Shu et al., 2022), is illustrated in Fig. 2. It consists of a set of encoders to embed relevant information from source and target images, which is then passed to condition generator. Two encoders, Eid and Epor, are used to encode identity of the source at multiple scales. Local information is encoded by Eid, a pretrained state-of-the-art face recognition network (Deng et al., 2019). It takes central face crops Crop(IS) as

input and outputs face embedding fid ∈ R512. Global information, including hair and head shape, is extracted by Epor, which takes full source image and outputs fpor ∈ R512. Such combination of appearance encoders allows to obtain more fine-grained head reconstruction, while paying special attention to more discriminative facial features.

To embed driving head pose and facial expression, we use motion encoder Emotion. Given full augmented target image IT, it produces motion embedding fmtn ∈ R256 . We address disentanglement of motion from appearance by stretching IT independently along horizontal and vertical axes before supplying it to Emotion. In this way, we change identity of the person, while preserving head pose and expression.

The obtained descriptors fid, fpor and fmtn are concatenated and used to condition StyleGAN-like generator G via AdaIN (Huang & Belongie, 2017). It takes learnable tensor T ∈ R512×4×4 as input and passes it through a series of convolutional blocks to obtain final image IA, injecting the appearance and motion information at each step. To stabilize training, the generator outputs binary head mask Mreenact along with reenacted head.

#### Source Target Iexp Ipose Ifull

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 3. Ablation results on representations learned by pose Epose and expression Eexp encoders

Refined motion encoder The original Aligner architecture of HeSer (Shu et al., 2022) included separate pose Epose and expression Eexp encoders. However, we encountered the problem of target identity leakage with this design, resulting in the mixing of target and source identitties in the output image IA . To obtain more insights into the problem, we conducted ablation study to learn which information is embedded by each encoder (fig. 3). We used embedding of canonical pose as output of Epose, and allowed Eexp to obtain expression embedding from target image, resulting in generation Ipose. In this case, head is still generated in pose of target, not in canonical one. It is overall quite similar to full result Ifull, except for difference in skin color which is now closer to the source one. On the other hand, by using canonical expression and image-based pose representation we obtain final generation Iexp with almost canonical expression and pose. This ablation indicates that Eexp learns both expression and pose embeddings, while Epose mainly transfers appearance of the target.

Next we decided to regularize each encoder to split the

motion information into meaningful separate embeddings. For this we tested different approaches by 1) decreasing size of encoders, 2) using disentanglement and cycle losses as in (Drobyshev et al., 2022; Pang et al., 2023), 3) using pretrained encoders from (Feng et al., 2021; Drobyshev et al., 2022). However, these experiments resulted in inferior performance compared to the baseline of single motion encoder Emotion, which embeds both pose and expression into single vector fmtn ∈ R256. Relevant ablation is shown in section results.

Training losses To train Aligner, we use hinge adversarial loss Ladv, feature-matching loss LFM (Salimans et al., 2016), L1 reconstruction loss LL1, VGG-based perceptual loss LVpercGG (Simonyan & Zisserman, 2014) and dice loss Ldice (Sudre et al., 2017). To improve source identity preservation, we introduce cosine LIDcos and perceptual LIDperc losses comparing embeddings and feature maps of IA and IS obtained with our pretrained encoder Eid. We also noticed that with the new combined motion encoder Emotion expressiveness decreases, and the closure of mouth and eyelids does not follow the target. We introduce additional emotion loss Lemo (Danecek et al., 2022) and keypoint closure loss that compares distance between lips and eyelids for generated and driving heads:

|kilower − kiupper| (1)

#### Lkpt =

kilower∈Klower

where Klower is the set of keypoints on lower eyelid or lip, and kilower and kiupper are symmetric keypoints on lower and upper eyelid or lip, respectively. Since reenactment quality is also influenced by such subtle factors as gaze direction, we also include perceptual gaze loss as in (Drobyshev et al., 2022) starting from 1000 epochs. Additional details on losses are given in supplementary material.

3.2. Blender 3.2.1. PRELIMINARY

We base our Blender on the corresponding module from (Shu et al., 2022). It involves data preprocessing, color reference creation, and blending steps.

[Figure 19]

Figure 5. Masks obtained in data preprocessing stage

[Figure 20]

- Figure 4. Blender architecture. Reenacted IA and target IT images are preprocessed to obtain grayscale animated head IAGH, target background ITBG and masks MAH and MAI defining head and background inpainting regions. Also, IA and IT are passed to Reference Creation module to obtain head color reference ITHR→A for color transfer. The background reference ITIR→A is obtained via External Inpainting module (Suvorov et al., 2021). These inputs are passed to Blending UNet to achieve final result IB.

Data preprocessing Data preprocessing stage prepares inputs for color transfer and background inpainting. In particular, the first problem is viewed as a problem of recoloring gray-scale reenacted head with colors of the target head. For this, segmentation masks MA of the head regions and a binary head mask MAH are obtained from reenacted image IA. Similarly, target segmentation MT and head MTH are obtained from target image IT. Then gray-scale image of the head to be colored is obtained as IAGH = Gray(IA ∗ MAH)

Additionally, we need to define regions that need inpainting due to differences in head shape between source and target. For this, union MU of reenacted MAH and target MTH head masks is dilated to an enlarged head mask MˆAH. Then the area that requires inpainting is denoted as MAI = MˆAH −

- MAH. The region to serve as color reference for background inpainting is then defined as MTI = MˆTH − MTH, where MˆTH is dilated target head mask MTH. Finally, background without head can be obtained as ITBG = IT ∗ (1 − MˆAH). Additionally, we introduce augmentation of inpainting mask
- MAI to enhance inpainting of large mismatched regions.

[Figure 21]

Figure 6. Examples of inpainting mask MAI augmentations

Color reference creation The next step is to provide color references for background inpainting and head color transfer. Creation of these references is based on learning the correlation between corresponding semantic regions of input IA and target IT images. During training of corresponding Reference Creation (RC) module, the same image serves as both input IA and target IT. To prevent the network from merely copying pixels from the same position, random color augmentation C′ is applied to IA and random horizontal flip F is applied to IT. Correlation learning takes place in the latent space based on the representation obtained with Feature Pyramid Network (FPN) (Lin et al., 2017) for IA and IT:

fA = FPN(C′(IA)) (2) fT = FPN(F(IT)) (3)

Next, correlation is calculated between features fA and fT for each spatial location. It is used to weight pixels during resampling in the following step. For each semantic region r ∈ {face, ears, eyes, brows, nose, lips, mouth, teeth, beard, hair,

glasses, hat, headphones, earrings} a correlation matrix Γr ∈ RN

A×NTr is computed, where NAr and NTr are numbers of pixels in region r in input IA and target IT images. Thus, each element Γr(u,v) is calculated as:

r

f¯Ar (u)Tf¯Tr(v) ∥f¯Ar (u)∥∥f¯Tr(v)∥

, u ∈ MAr , v ∈ MTr (4)

Γr(u,v) =

where f¯Ar (u) and f¯Tr(v) are channelwise centralized features fA and fT at locations u and v. Γr(u,v) is then normalized

via softmax and used to determine contribution of each pixel v in corresponding region of IT to the color of pixel u in the head reference image ITr→A(u):

More details on corresponding hyperparameters are given in supplementary.

#### IA Iext IA IB IB

ITr→A(u) =

softmaxv(Γr(u,v)/τ)·IT(v), u ∈ MAr

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

u∈MTr

(5) where τ is temperature coefficient. Overall, computation of head color reference ITHR→A is based on the following inputs:

- Figure 7. Improved blending of hair. IB and IB are final results with and without background extrapolation

Improved blending of hair We also implement additional step on Aligner output to refine blending of hair in the resulting image IB. We utilize soft portrait masks Msoft to segment hair area. They provide better segmentation results than hard ones due to the uncertainty of edge estimation of the hair region. However, due to such choice of masking blending UNet recognizes soft mask areas to belong to the head region and does not extrapolate background to them, resulting in a visible border. To remedy this problem, we create additional image Iext, where the background is extrapolated over the head mask MAH, as shown in fig. 7. Then we obtain refined animated portrait IA by blending extrapolated background Iext with Aligner output based on matting mask Msoft: IA = Msoft∗IA+(1−Msoft)∗Iext.

[Figure 27]

- Figure 8. Post-processing of blended image. Given target IT and reenacted IA images, we obtain masks of reenacted head MAhead and target hair MThair. We substract the first mask from the second one to obtain area for inpanting Minpainting, which is filled by Kandinsky model (Shakhmatov et al., 2023)

#### ITHR→A = RC(fA,MA,fT,MT,IT ∗ MTH) (6)

However, during inference a label mismatch can occur if an area is present in the reenacted image IA but not in the target one IT. In this case, such face region is not colored by the blending UNet, since no color reference for it is given. To remedy this problem, we propose to assign colors from other semantically relevant areas of IT. In case no such area is found (for instance hat is present in IA but not in IT) or is too small to provide robust reference, we copy the colors from the original image IA.

To create color inpainting reference ITIR→A, we use LaMa model (Suvorov et al., 2021), conditioned on inpainting

mask MTI. We found this solution to provide superior quality compared to creation with RC module.

Blending UNet Blending of reenacted head into target background is performed by Blending UNet B submodule.

- As input it takes concatenated head ITHR→A and background inpainting ITIR→A references, head mask MAH, background ITBG, inpainting mask MAI and gray-scale head IAGH and outputs final image IB:

#### IB = B(ITHR→A,ITIR→A,MAH,ITBG,MAI ,IAGH) (7)

The UNet is trained with standard adversarial Ladv, reconstruction LL1 and perceptual losses LVpercGG. Additionally, to learn a meaningful correlation matrix Γr, cycle consistency loss is used:

Lc = λc∥IT→A→T − IT∥1 (8) where ITk→A→T(u) = v∈Mk

softmaxv(Γk(u,v)/τ) ·

A

IT→A(v), u ∈ MTk. Additional regularization is performed by calculation of cycle loss with another target image IT′ having the same identity as IA:

Head transfer on real data When transferring to real data, we must take into account that our model works within cropped images. In contrast to the face swap problem, where the person’s face is always inside the generated region, in the head swap problem the hair can extend beyond the boundaries of the considered area. In this case, we need to remove excess hair from the image. For this purpose, we propose a

#### Lc′ = λc∥IT′→A→T′ − IT∥1 (9)

To make head color reference more similar to source image, regularization loss is used:

#### Lreg = λreg∥MAH· (grayscale(IA) − grayscale(ITHR→A) ∥1

(10)

post-processing blending algorithm using Kandinsky diffusion model (Shakhmatov et al., 2023). We obtain the final image by inpaiting the region masked by Minpainting for UNet output IB, as shown in fig. 8. It should be noted that this step is optional and is applied when person in the target image has hair that significantly differs from hair of source.

Image BiSeNet Ours

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

Figure 9. Qualitative comparison of segmentation models

### 3.3. Segmentation

For our model to work, it is necessary to have a high-quality segmentation model—it will be used in the Blender module, as well as during preprocessing stages to select a person’s head. There are two main requirements for the model: it must be able to segment hairstyles and facial hair as separate classes, and it must be additive, meaning it should segment in such a way that we can combine the segments to obtain a complete head. Additionally, each region should be homogeneous in color—for example, the ’beard’ class should not overlap with the ’skin’ class, as these regions will be used later for color transfer. To train the model, a dataset of 20,000 FullHD images was collected and annotated (Kapitanov et al., 2023), and a segmentation model was trained based on it. We settled on Segformer-B5 (Xie

et al., 2021) as the model for face parsing and segmentation. As a result, the model can segment the following classes: background, person, skin, left brow, right brow, left eye, right eye, mouth, teeth, lips, left ear, right ear, nose, neck, beard, hair, hat, headphone, glasses, earring. Fig. 9 shows a visual comparison of our model and BiSeNet (Yu et al., 2018).

## 4. Experiments

### 4.1. Experiment Setup

Dataset We use VoxCeleb2 (Chung et al., 2018) dataset to train and evaluate our model at 512 × 512 and 256 × 256 resolutions. Since some videos originally have low quality, we filter data using image quality assessment methods (Su et al., 2020; Wang et al., 2023), leaving approximately 70% of the original dataset. We additionally preprocess the data by cropping face and head regions and calculating keypoints with Feng et al. (2021). Our final train and test sets include respectively 135500 and 5500 videos. The split is made so as to avoid intersection between train and test identities. During training, we sample source and target from the same video, while during inference they can feature different identities.

Evaluation metrics To evaluate our Aligner and Blender against the baselines, we use LPIPS (Zhang et al., 2018), SSIM (Wang et al., 2004) and MS-SSIM (Wang et al., 2003) to assess perceptual quality, and PSNR to measure reconstruction error. For Aligner, to compare source identity preservation, we compute cosine distance (CSIM) between embeddings of IA and IS from face recognition model (Deng et al., 2019). On cross-reenactment, we also utilize Frechet Inception Distance (FID) (Heusel et al., 2017). Additionally, for Blender we also measure reconstruction for background inpainting PSNRinpainting and head color transfer PSNRhead.

Additionally, in cross-reenactment scenario we conduct a user study to qualitatively compare preservation of source identity (UAPP), transfer of target motion (UMTN) and overall quality (UQLT).

### 4.2. Aligner evaluation

Baselines Our Aligner is compared against the opensource baselines at 512 × 512 and 256 × 256 resolutions. We note that the majority of competing models are trained on narrow face crops and hence do not reconstruct whole head and hair.

Few 2D reenactment models are available at 512 × 512 resolution. We compare against StyleHEAT (Yin et al., 2022), based on StyleGAN (Karras, 2019) inversion. Additionally, we train baseline from HeSer (Shu et al., 2022) at 512×512

Table 1. Quantitative results on head reenactment at 512 × 512 and 256 × 256 resolution Method

Self-reenactment (256 × 256) Cross-reenactment (256 × 256)

CSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ CSIM ↑ FID ↓

|X2Face (Wiles et al., 2018) FOMM (Siarohin et al., 2019) PIRender (Ren et al., 2021) Bi-layer (Zakharov et al., 2020a) DaGAN (Hong et al., 2022) DPE (Pang et al., 2023) LIA (Wang et al., 2022b) TPSMM (Zhao & Zhang, 2022) Ours<br><br>|0.789 0.201 19.66 0.715 0.847 0.128 21.83 0.783 0.827 0.173 19.46 0.704<br><br>0.788 0.198 20.33 0.772<br><br>0.789 0.208 19.19 0.712 0.845 0.161 20.92 0.768<br><br><br>0.842 0.128 22.11 0.786<br><br>0.843 0.124 21.86 0.804 0.747 0.116 22.30 0.815<br><br><br>|0.636 41.95 0.629 28.12 0.665 19.95 0.654 32.70 0.580 44.00 0.613 50.61 0.659 24.19 0.647 22.06 0.616 36.89|
|---|---|---|
|Method<br><br>|Self-reenactment (512 × 512) CSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑|Cross-reenactment (512 × 512)<br><br>CSIM ↑ FID ↓|

HeSer (Shu et al., 2022) 0.780 0.139 22.99 0.856 0.612 35.33 StyleHEAT (Yin et al., 2022) 0.789 0.574 8.791 0.582 0.673 75.856 Ours 0.754 0.142 22.04 0.848 0.628 29.57

resolution for full head synthesis.

- At 256 × 256 resolution, we compare against the following warping-based approaches: First Order Motion Model (FOMM) (Siarohin et al., 2019), PIRenderer (Ren et al.,

- 2021), Depth-Aware Generative Adversarial Network (DaGAN) (Hong et al., 2022), Disentanglement of Pose and Expression (DPE) model (Pang et al., 2023), Latent Image Animator (LIA) (Wang et al., 2022b) and Thin-Plate Spline Motion Model (TPSMM) (Zhao & Zhang, 2022). We also include methods based on latent face reconstruction with target motion: X2Face (Wiles et al., 2018) and Fast Bilayer Neural Synthesis (Zakharov et al., 2020b). All these methods generate only narrow face crops and not the whole head.

Results The results for self- and cross-reenactment scenarios at 512×512 and 256×256 resolutions are presented in table 1. At 512 × 512 resolution, HeSer (Shu et al.,

- 2022) outperforms other methods by almost all metrics at self-reenactment. This is largely attributed to the leakage of target identity into final generation, which supplies additional information on the desired result. However, it is inferior to GHOST 2.0 on cross-reenactment. As can be seen from fig. 10, GHOST 2.0 is significantly better at preserving source identity and skin color, while the HeSer produces a mixed identity of source and driver. Also, compared to StyleHEAT, our model is more robust to generation in extreme poses, although it may be slightly inferior in terms of identity preservation in frontal views.

At 256 × 256 resolution, we outperform other methods by LPIPS, PSNR and SSIM in self-reenactment. This is in part explained by robustness of our method to generation in difficult poses. Warping-based methods perform well only in case of small displacements, resulting in severe face distortion and artifacts otherwise. However, they usually

- Table 2. Side-by-side comparison at 512 × 512 resolution

|Method<br><br>|UAPP ↑ UMTN ↑ UQLT ↑|
|---|---|
|HeSer (Shu et al., 2022) StyleHEAT (Yin et al., 2022) Ours|0.04 0.15 0.04 0.13 0.05 0.03 0.83 0.80 0.93<br><br>|

- Table 3. Side-by-side comparison at 256 × 256 resolution

|Method<br><br>|UAPP ↑ UMTN ↑ UQLT ↑|
|---|---|
|Bi-layer (Zakharov et al., 2020a) DaGAN (Hong et al., 2022) X2Face (Wiles et al., 2018)<br><br>|0.81 0.83 0.91 0.11 0.13 0.07 0.08 0.04 0.02|
|LIA (Wang et al., 2022b) TPSMM (Zhao & Zhang, 2022) PIRender (Ren et al., 2021)<br><br>|0.52 0.42 0.51 0.20 0.35 0.25 0.28 0.23 0.24|
|DPE (Pang et al., 2023) FOMM (Siarohin et al., 2019) Ours<br><br>|0.27 0.13 0.12 0.10 0.12 0.07 0.63 0.75 0.81|
|Bi-layer (Zakharov et al., 2020a) LIA (Wang et al., 2022b) Ours<br><br>|0.13 0.10 0.07 0.41 0.29 0.29 0.46 0.61 0.64|

excel in identity preservation, as evidenced by CSIM both on self- and cross-reenactment. Please see supplement for visual comparison of the models.

Side-by-side We also conducted side-by-side comparison on cross-reenactment scenario between our Aligner and the competitors both at 256×256 and 512×512 resolutions. We asked the users to choose the model which performs best in terms of the following criteria: source identity preservation (UAPP), target movement transfer (UMTN) and overall generation quality (UQLT). We present the percentage of examples where each model is chosen in tables 2 and 3.

Source Target GHOST 2.0 HeSer StyleHEAT

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

Figure 10. Cross-reenactment results at 512 × 512 resolution

Since at 256 × 256 resolution we have a large number of models to compare against, we split them into three triplets first, compare within them, and then compare winners of these triplets. On average, each image from the test dataset is shown to 41 users.

Our method outperforms the competitors by a large margin in terms of motion preservation and generation quality at both resolutions. It is also significantly better than most of the methods in terms of identity preservation. Several methods, such as DPE (Pang et al., 2023) or LIA (Wang

- et al., 2022b), fail at generation with large head rotations and produce blank outputs. Our method shows robustness to various head poses and expressions.

Source Target GHOST 2.0

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

Ablation We conducted ablation study to justify our design of combined motion encoder Emtn. We compared different strategies to disentangle pose and facial expression descriptors through special losses and architectural changes. The results are shown in table 5. In addition to FID and CSIM, we also use Average Keypoint Distance (AKD) to measure motion transfer.

In HeSer, two separate MobileNetV2 (Sandler et al., 2018) encoders are used to capture pose and expression. We also

Figure 11. GHOST 2.0 results on the task of head swap. Our method achieves natural blending of the reenacted head into target background, corresponding in skin color, lighting and contrast to the rest of the image.

conducted experiment with decreasing their number of channels in half to force them to learn only relevant information. In other experiments, we replaced them with corresponding pretrained encoders from DECA (Feng et al., 2021), and

Table 4. Quantitative results on head blending Method

Plain data LPIPS ↓ SSIM ↑ MSSIM ↑ PSNR ↑ PSNRinpainting ↑ PSNRhead ↑

|Baseline REFace InstantID + SDXL inpainting External inpainting|0.034 0.972 0.987 33.86 35.13 42.13 0.24 0.89 0.78 23.45 36.72 29.29 0.10 0.94 0.91 29.49 38.54 30.38<br><br>0.018 0.984 0.992 36.65 37.98 45.83|
|---|---|
|Method|Mask augmentations and source color jitter LPIPS ↓ SSIM ↑ MSSIM ↑ PSNR ↑ PSNRinpainting ↑ PSNRhead ↑<br><br>|

Baseline 0.093 0.926 0.945 27.76 28.15 41.25 External inpainting 0.060 0.947 0.951 28.21 28.45 44.13

also tried to disentangle representations with cyclic loss from DPE (Pang et al., 2023). Our design with combined encoder Emtn achieves the best quality by all metrics. Further details on other ablations and calculation of AKD are given in supplementary.

Table 5. Results on cross-reenactment for different disentanglement strategies between target motion and identity

|Method|FID ↓ CSIM ↑ AKD ↓<br><br>|
|---|---|
|HeSer (Shu et al., 2022) Smaller encoders DECA encoders (Feng et al., 2021) DPE loss (Pang et al., 2023) Combined Emtn|28.63 0.596 0.0095 28.38 0.622 0.0108 35.41 0.627 0.0155<br><br>31.70 0.603 0.0103 26.83 0.622 0.0098<br><br>|

### 4.3. Blender evaluation

To evaluate our Blender and justify external inpainting usage in it, we trained baseline HeSer Blender on the same data. Additionally, we compare with diffusion-based head swap approaches, such as REFace (Baliah et al., 2025) and InstantID (Wang et al., 2024). For the latter solution, we first use InstantID (Wang et al., 2024) to generate source head with target pose directed by facial keypoints. Then the resulting head is blended into target background using SDXL (Podell

- et al., 2023) inpainting model. It should be noted that both solutions frequently fail to preserve identity and target skin color. As can be seen from table 4, our version with external inpainting outperforms baseline and competitors on almost every metric. However, for background inpainting, SDXL shows slightly better results than LaMa, so further investigation into better inpainting models may constitute an area for future research. Furthermore, for external inpainting and baseline, we present results on training with additional data augmentations. Our solution outperforms baseline in this scenario as well and also doesn’t fail in hard cases when source and target differ significantly (fig. 11).

In addition, we present inference results of our model on real-life and outdoor photos in supplementary 17. Although

our model successfully copes with most cases, sometimes Kandinsky model hallucinates during post-processing and may produce additional attributes such as collars when inpainting excess hair. Also, hair for reenacted source head is generated only inside the crop region. In case target person has shorter hair, it is likely that in the resulting image source hair would look unnaturally cropped from below.

## 5. Conclusion

We have presented a two-stage method for realistic head swapping for in-the-wild images. We improve Aligner architecture by merging pose and expression encoders into single motion encoder, which remedies the problem of driver identity leakage. Our head reenactment model outperforms other methods by both qualitative and quantitative metrics and is more robust to large pose variations. We also introduce additional refinements during blending to improve quality of head transfer and inpainting, which allows to obtain superior results compared to baseline solution.

The limitations of our model are the following. In several cases, our method does not reproduce fine details of source appearance. This can be tackled by using stronger appearance encoders in Aligner. Concerning Blender, some face parts may not be evenly colored if the area of corresponding color reference is small. Additionally, blending stage could be improved by mitigating the generation of additional clothing attributes (e.g. collars) and better hair processing. Complex lighting conditions can also produce inferior results during blending.

## 6. Impact Statement

This paper enhances approach to head swapping via better head reenactment and inpainting modules. While such models find applications in commercial scenarios, they are also known to be used for fraudulent activities. However, we suppose that results of this work can be used to fight such misuse by aiding research on more robust deepfake detection systems.

## 7. Acknowledgments

We thank Nikolay Gerasimenko, Anna Averchenkova and ABT data labelling team for help with side-by-side comparison. We also thank Viacheslav Vasilev for suggestions and comments regarding text. Last but not least, we thank Alexander Kapitanov and his team from SberDevices for training of segmentation model.

## References

Alansari, M., Hay, O. A., Javed, S., Shoufan, A., Zweiri, Y., and Werghi, N. Ghostfacenets: Lightweight face recognition model from cheap operations. IEEE Access, 11:35429–35446, 2023. doi: 10.1109/ACCESS.2023. 3266068.

Bai, Z., Tan, F., Huang, Z., Sarkar, K., Tang, D., Qiu, D., Meka, A., Du, R., Dou, M., Orts-Escolano, S., et al. Learning personalized high quality volumetric head avatars from monocular rgb videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16890–16900, 2023.

Baliah, S., Lin, Q., Liao, S., Liang, X., and Khan, M. H. Realistic and efficient face swapping: A unified approach with diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1062–1071. IEEE, 2025.

Burkov, E., Pasechnik, I., Grigorev, A., and Lempitsky, V. Neural head reenactment with latent pose descriptors. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Cao, Q., Shen, L., Xie, W., Parkhi, O. M., and Zisserman, A. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018), pp. 67–74. IEEE, 2018.

Chen, R., Chen, X., Ni, B., and Ge, Y. Simswap: An efficient framework for high fidelity face swapping. In MM ’20: The 28th ACM International Conference on Multimedia, 2020.

Chen, X., He, K., Zhu, J., Ge, Y., Li, W., and Wang, C. Hifivfs: High fidelity video face swapping, 2024.

Chu, X., Li, Y., Zeng, A., Yang, T., Lin, L., Liu, Y., and Harada, T. GPAvatar: Generalizable and precise head avatar from image(s). In The Twelfth International Conference on Learning Representations, 2024.

Chung, J. S., Nagrani, A., and Zisserman, A. Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622, 2018.

Danecek, R., Black, M. J., and Bolkart, T. EMOCA: Emotion driven monocular face capture and animation. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20311–20322, 2022.

Deng, J., Guo, J., Xue, N., and Zafeiriou, S. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4690–4699, 2019.

Deng, Y., Wang, D., Ren, X., Chen, X., and Wang, B. Portrait4d: Learning one-shot 4d head avatar synthesis using synthetic data. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Drobyshev, N., Chelishev, J., Khakhulin, T., Ivakhnenko, A., Lempitsky, V., and Zakharov, E. Megaportraits: One-shot megapixel neural head avatars. 2022.

Drobyshev, N., Casademunt, A. B., Vougioukas, K., Landgraf, Z., Petridis, S., and Pantic, M. Emoportraits: Emotion-enhanced multimodal one-shot head avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8498–8507, 2024.

Duta, I. C., Liu, L., Zhu, F., and Shao, L. Improved residual networks for image and video recognition. In 2020 25th International Conference on Pattern Recognition (ICPR), pp. 9415–9422. IEEE, 2021.

Feng, Y., Feng, H., Black, M. J., and Bolkart, T. Learning an animatable detailed 3D face model from in-the-wild images. volume 40, 2021.

Giebenhain, S., Kirschstein, T., R¨unz, M., Agapito, L., and Nießner, M. Npga: Neural parametric gaussian avatars. In SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers ’24), December 3-6, Tokyo, Japan, 2024. ISBN 979-8-4007-1131-2/24/12. doi: 10.1145/3680528. 3687689.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Grassal, P.-W., Prinzler, M., Leistner, T., Rother, C., Nießner, M., and Thies, J. Neural head avatars from monocular rgb videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18653–18664, 2022.

Groshev, A., Maltseva, A., Chesakov, D., Kuznetsov, A., and Dimitrov, D. Ghost—a new face swap approach for image and video domains. IEEE Access, 10:83452– 83462, 2022. doi: 10.1109/ACCESS.2022.3196668.

Han, Y., Zhang, J., Zhu, J., Li, X., Ge, Y., Li, W., Wang, C., Liu, Y., Liu, X., and Tai, Y. A generalist facex via learning unified facial representation, 2023.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hong, F.-T., Zhang, L., Shen, L., and Xu, D. Depth-aware generative adversarial network for talking head video generation. 2022.

Huang, X. and Belongie, S. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.

Kapitanov, A., Kvanchiani, K., and Sofia, K. Easyportrait

- face parsing and portrait segmentation dataset. arXiv preprint arXiv:2304.13509, 2023.

Karras, T. A style-based generator architecture for generative adversarial networks. arXiv preprint arXiv:1812.04948, 2019.

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., and Aila, T. Analyzing and improving the image quality of StyleGAN. In Proc. CVPR, 2020.

Khakhulin, T., Sklyarova, V., Lempitsky, V., and Zakharov, E. Realistic one-shot mesh-based head avatars. In European Conference of Computer vision (ECCV), 2022.

Kingma, D. P. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Kirschstein, T., Giebenhain, S., and Nießner, M. Diffusionavatars: Deferred diffusion for high-fidelity 3d head avatars. arXiv preprint arXiv:2311.18635, 2023.

Li, X., De Mello, S., Liu, S., Nagano, K., Iqbal, U., and Kautz, J. Generalizable one-shot 3d neural head avatar. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 47239–47250. Curran Associates, Inc., 2023.

Lin, T.-Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2117–2125, 2017.

Ma, Z., Zhu, X., Qi, G.-J., Lei, Z., and Zhang, L. Otavatar: One-shot talking face avatar with controllable tri-plane rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16901– 16910, 2023.

Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

Pang, Y., Zhang, Y., Quan, W., Fan, Y., Cun, X., Shan, Y., and Yan, D.-M. Dpe: Disentanglement of pose and expression for general video portrait editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 427–436, June 2023.

Perov, I., Gao, D., Chervoniy, N., Liu, K., Marangonda, S., Um´e, C., Facenheim, C. S., RP, L., Jiang, J., Zhang, S., et al. Deepfacelab: Integrated, flexible and extensible face-swapping framework. arXiv preprint arXiv:2005.05535, 2020.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., and Nießner, M. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20299–20309, 2024.

Ren, Y., Li, G., Chen, Y., Li, T. H., and Liu, S. Pirenderer: Controllable portrait image generation via semantic neural rendering. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 13759–13768, 2021.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4510–4520, 2018.

Sarkar, K., Golyanik, V., Liu, L., and Theobalt, C. Style and pose control for image synthesis of humans from a single monocular view. arXiv preprint arXiv:2102.11263, 2021.

Shakhmatov, A., Razzhigaev, A., Nikolich, A., Arkhipkin, V., Pavlov, I., Kuznetsov, A., and Dimitrov, D. kandinsky 2.2, 2023.

Shu, C., Wu, H., Zhou, H., Liu, J., Hong, Z., Ding, C., Han, J., Liu, J., Ding, E., and Wang, J. Few-shot head swapping in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10789–10798, 2022.

Siarohin, A., Lathuili`ere, S., Tulyakov, S., Ricci, E., and Sebe, N. First order motion model for image animation. In Conference on Neural Information Processing Systems (NeurIPS), December 2019.

Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., and Zhang, Y. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., and Jorge Cardoso, M. Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: Third International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017, Held in Conjunction with MICCAI 2017, Qu´ebec City, QC, Canada, September 14, Proceedings 3, pp. 240–248. Springer, 2017.

Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., and Lempitsky, V. Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161, 2021.

Wang, J., Chan, K. C., and Loy, C. C. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 2555–2563, 2023.

Wang, Q., Liu, L., Hua, M., Zhu, P., Zuo, W., Hu, Q., Lu, H., and Cao, B. Hs-diffusion: Semantic-mixing diffusion for head swapping. arXiv preprint arXiv:2212.06458, 2022a.

Wang, Q., Bai, X., Wang, H., Qin, Z., and Chen, A. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

Wang, Y., Chen, X., Zhu, J., Chu, W., Tai, Y., Wang, C., Li, J., Wu, Y., Huang, F., and Ji, R. Hififace: 3d shape and semantic prior guided high fidelity face swapping. CoRR, abs/2106.09965, 2021.

- Wang, Y., Yang, D., Br´emond, F., and Dantcheva, A. Latent image animator: Learning to animate images via latent space navigation. ArXiv, abs/2203.09043, 2022b.

- Wang, Y., Yang, D., Bremond, F., and Dantcheva, A. Latent image animator: Learning to animate images via latent space navigation. In International Conference on Learning Representations, 2022c.
- Wang, Z., Simoncelli, E., and Bovik, A. Multiscale structural similarity for image quality assessment. In The Thrity-Seventh Asilomar Conference on Signals, Systems and Computers, 2003, volume 2, pp. 1398–1402 Vol.2,

2003. doi: 10.1109/ACSSC.2003.1292216.

- Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13:600–612, 2004.

Wiles, O., Koepke, A., and Zisserman, A. X2face: A network for controlling face generation using images, audio, and pose codes. In Proceedings of the European conference on computer vision (ECCV), pp. 670–686, 2018.

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. Segformer: Simple and efficient design for semantic segmentation with transformers. In Neural Information Processing Systems (NeurIPS), 2021.

Xie, S., Girshick, R., Doll´ar, P., Tu, Z., and He, K. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492–1500, 2017.

Xu, Y., Wang, L., Zhao, X., Zhang, H., and Liu, Y. Avatarmav: Fast 3d head avatar reconstruction using motionaware neural voxels. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1–10, 2023.

Ye, Z., Zhong, T., Ren, Y., Yang, J., Li, W., Huang, J., Jiang, Z., He, J., Huang, R., Liu, J., Zhang, C., Yin, X., Ma, Z., and Zhao, Z. Real3d-portrait: One-shot realistic 3d talking portrait synthesis. 2024.

Yin, F., Zhang, Y., Cun, X., Cao, M., Fan, Y., Wang, X., Bai, Q., Wu, B., Wang, J., and Yang, Y. Styleheat: One-shot high-resolution editable talking face generation via pretrained stylegan. In European conference on computer vision, pp. 85–101. Springer, 2022.

Yu, C., Wang, J., Peng, C., Gao, C., Yu, G., and Sang, N. Bisenet: Bilateral segmentation network for real-time

semantic segmentation. In Proceedings of the European conference on computer vision (ECCV), pp. 325–341, 2018.

Zakharov, E., Ivakhnenko, A., Shysheya, A., and Lempitsky, V. Fast bi-layer neural synthesis of one-shot realistic head avatars. In European Conference of Computer vision (ECCV), August 2020a.

Zakharov, E., Ivakhnenko, A., Shysheya, A., and Lempitsky, V. Fast bi-layer neural synthesis of one-shot realistic head avatars. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pp. 524–540. Springer, 2020b.

Zhang, B., Qi, C., Zhang, P., Zhang, B., Wu, H., Chen, D., Chen, Q., Wang, Y., and Wen, F. Metaportrait: Identitypreserving talking head generation with fast personalized adaptation.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Zhao, J. and Zhang, H. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3657–3666, 2022.

Zhao, W., Rao, Y., Shi, W., Liu, Z., Zhou, J., and Lu, J. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. CVPR, 2023.

Zheng, Y., Abrevaya, V. F., B¨uhler, M. C., Chen, X., Black, M. J., and Hilliges, O. I M Avatar: Implicit morphable head avatars from videos. In Computer Vision and Pattern Recognition (CVPR), 2022.

Zheng, Y., Yifan, W., Wetzstein, G., Black, M. J., and Hilliges, O. Pointavatar: Deformable point-based head avatars from videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21057–21067, 2023.

Zhu, Y., Li, Q., Wang, J., Xu, C., and Sun, Z. One shot face swapping on megapixels. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 4834–4844, June 2021.

Zielonka, W., Bolkart, T., and Thies, J. Instant volumetric head avatars. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4574–4584, 2022.

## A. Details on losses

For training Aligner, we use the following combination of losses:

Laligner = λadvLadv + λFMLFM + λL1LL1 +λVpercGGLVpercGG + λIDpercLIDperc + λIDcosLIDcos + λdiceLdice+ λemoLemo + λkptLkpt + λgazeLgaze

(11)

We set the following weights for the losses: λadv = 0.1, λFM = 10, λL1 = 30, λVpercGG = 0.01, λIDperc = 2 × 10−3, λIDcos = 0.01, λdice = 1, λemo = 1, λkpt = 30 and λgaze = 0.5 For training Blender, we use the following losses:

Lblender = λadvLadv + λL1LL1 + λVpercGGLVpercGG +λcLc + λcLc′ + λregLreg

(12)

The weighs are set as λadv = 1, λL1 = 1, λVpercGG = 0.01, λc = 1 and λreg = 1.

## B. Training details

We trained Aligner for 1 000 000 iterations with batch size of 20 on 512x512 resolution, and for 800 000 iterations with batch size of 32 on 256x256 resolution. On 8 NVIDIA A 100 GPUs, it takes 27 and 9 days respectively. We use Adam optimizer (Kingma, 2014) with generator learning rate 1×10−4 and discriminator learning rate 4×10−4, with gradient clipping threshold of 10. We also apply spectral normalization (Miyato et al., 2018) to stabilize training.

We trained Blender for 50000 iterations with batch size of 20 on 512x512 resolution. On 4 NVIDIA A 100 GPUs it takes 2 days. We use Adam optimizer and same optimizer options as in Aligner.

## C. Detailed architecture

Aligner We use ResNeXt-50 (Xie et al., 2017) as our portrait encoder Epor, IResNet-50 (Duta et al., 2021) pretrained with Arcface (Deng et al., 2019) loss as identity encoder Eid and MobileNetV2 (Sandler et al., 2018) as motion encoder Emotion. The dimensions of embeddings produced by these encoders are 512, 512 and 256, respectively.

These embeddings are concatenated and processed with 2layer MLP with ReLU activation and spectral normalization. The intermediate dimension is maintained the same as the input one. The resulting vector is supplied to AdaIn (Huang & Belongie, 2017) layers to condition the generator, which is borrowed from (Burkov et al., 2020). We add one additional upsampling residual block to the original generator to increase output resolution from 256 to 512. Discriminator is also borrowed from (Burkov et al., 2020) in its default version.

Table 6. Ablation of addition of keypoint Lkpt and emotion Lemo losses

Cross-reenactment

Experiment

CSIM ↑ FID ↓ AKD ↓

|with losses w/o losses<br><br>|0.621 26.83 0.0098 0.607 28.60 0.0107|
|---|---|
|Experiment|Self-reenactment CSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ AKD ↓<br><br>|

with losses 0.745 0.150 21.87 0.845 0.0074 w/o losses 0.724 0.154 21.97 0.841 0.0079

Table 7. Ablation on the start epoch for gaze loss Lgaze

Cross-reenactment

Experiment

CSIM ↑ FID ↓ AKD ↓

|Epoch 1000 Epoch 10<br><br>|0.621 26.83 0.0098 0.538 37.22 0.0091|
|---|---|
|Experiment<br><br>|Self-reenactment CSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ AKD ↓|

Epoch 1000 0.745 0.150 21.87 0.845 0.0074 Epoch 10 0.699 0.166 21.47 0.837 0.0082

## D. Further ablations on Aligner

We calculate Average Keypoint Distance (AKD) using keypoints from DECA (Feng et al., 2021) model. Given a triplet of source IS, target IT and generated IA images, we compute absolute distance between pair of normalized keypoints, which are based on shape blensdshapes from IS and pose and expression blendshapes from IT and IA. In this way, we assess motion transfer, while keeping source appearance invariant.

We also show the effect of adding keypoint Lkpt and emotion Lemo losses when training motion encoder Emtn. As can be seen in table 6, they improve image quality, identity preservation and pose transfer in cross-reenactment scenario, and generally lead to better disentanglement between target motion and identity. On self-reenactment, metrics also generally improve with addition of these losses.

Finally, we also justify our choice to include gaze loss Lgaze only at the end of training after 1000 epochs in table 7.

We compare it to the experiment when we include Lgaze only after 10 training epochs, when the model is capable to generate eyes with enough details. Early addition of this loss results in a significant deterioration of source identity preservation and in a noticeable fall in general quality of images.

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

###### Figure 12. Cross-reenactment results on 512 × 512

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

###### Figure 13. Self-reenactment results on 512 × 512

Source Target GHOST 2.0 DPE DaGAN LIA TPSMM

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

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

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Source Target GHOST 2.0 X2face FOMM Bi-layer PIRender

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

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

Figure 14. Cross-reenactment results on 256 × 256

[Figure 207]

###### Figure 15. Head swap results with the same identity and color augmentation applied to the source

[Figure 208]

###### Figure 16. Head swap results with different identities

[Figure 209]

###### Figure 17. Head swap results on real-life and outdoor photos

