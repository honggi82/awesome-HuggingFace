|Noname manuscript No. (will be inserted by the editor)|
|---|

## MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation

### Jiahao Xie · Wei Li · Xiangtai Li · Ziwei Liu · Yew Soon Ong · Chen Change Loy

# arXiv:2309.13042v2[cs.CV]3Oct2024

Received: date / Accepted: date

Abstract We present MosaicFusion, a simple yet effective diffusion-based data augmentation approach for large vocabulary instance segmentation. Our method is training-free and does not rely on any label supervision. Two key designs enable us to employ an off-the-shelf text-to-image diffusion model as a useful dataset generator for object instances and mask annotations. First, we divide an image canvas into several regions and perform a single round of diffusion process to generate multiple instances simultaneously, conditioning on different text prompts. Second, we obtain corresponding instance masks by aggregating cross-attention maps associated with object prompts across layers and diffusion time steps, followed by simple thresholding and edge-aware refinement processing. Without bells and whistles, our MosaicFusion can produce a significant amount of synthetic labeled data for both rare and novel categories. Experimental results on the challenging LVIS long-tailed and openvocabulary benchmarks demonstrate that MosaicFusion can

Jiahao Xie S-Lab, Nanyang Technological University, Singapore E-mail: jiahao003@ntu.edu.sg

Wei Li S-Lab, Nanyang Technological University, Singapore E-mail: wei.l@ntu.edu.sg

Xiangtai Li S-Lab, Nanyang Technological University, Singapore E-mail: xiangtai.li@ntu.edu.sg

Ziwei Liu S-Lab, Nanyang Technological University, Singapore E-mail: ziwei.liu@ntu.edu.sg

Yew Soon Ong Nanyang Technological University, Singapore E-mail: asysong@ntu.edu.sg

Chen Change Loy S-Lab, Nanyang Technological University, Singapore E-mail: ccloy@ntu.edu.sg

significantly improve the performance of existing instance segmentation models, especially for rare and novel categories. Code: https://github.com/Jiahao000/ MosaicFusion.

Keywords Text-to-image diffusion models · Long tail · Open vocabulary · Instance segmentation

### 1 Introduction

Instance segmentation is a fundamental yet challenging task—identifying and segmenting each object in an imagethat empowers remarkable applications in autonomous driving, robotics and medical imaging (Lin et al, 2014; Cordts et al, 2016; Gupta et al, 2019; Waqas Zamir et al, 2019; Kuznetsova et al, 2020). However, manually labeling a large-scale instance segmentation dataset is extremely laborious and expensive as annotators need to provide a mask with precise boundaries and assign a unique label for each object instance. The cost of such a dense labeling process increases dramatically for a very large vocabulary, due to highly diverse and complicated visual scenes in different applications. As a result, it is prohibitive to scale up the vocabulary size of instance segmentation datasets. As illustrated in Fig. 1, such a predicament of data scarcity becomes even worse under a natural data distribution that contains low-sample rare categories and out-of-distributed novel categories, both of which lead to poor performance of stateof-the-art instance segmentation models in long-tailed and open-vocabulary scenarios.

In the literature, researchers have sought ways to reduce the heavy reliance on labeled training data for instance segmentation. One popular method, Copy-Paste (Ghiasi et al, 2021), enhances datasets by placing objects onto backgrounds. However, while effective, this straightforward

|[Figure 1]<br><br>frequent / common<br><br>rare novel<br><br>MosaicFusion|
|---|

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

[Figure 13]

ResNet-50

Swin-B ResNet-50x64

ResNet-50

39.0

15.2

20.6

34.9

Baseline 18.0

9.6

35.9

31.7

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

w/ MosaicFusion

AP

AP

AP AP

generated instances with masks

- Fig. 1: Long-tailed and open-vocabulary instance segmentation on LVIS (Gupta et al, 2019) using our MosaicFusion data augmentation approach, which can generate meaningful synthetic labeled data for both rare and novel categories with-

out further training and label supervision. We evaluate the model with the standard mask AP (i.e., APr and APnovel). MosaicFusion provides strong gains on all considered baseline methods (e.g., Mask R-CNN (He et al, 2017) with ResNet-50, Box-Supervised CenterNet2 (Zhou et al, 2022b) with Swin-B, F-VLM (Kuo et al, 2023) with ResNet-50 and ResNet-50x64)

technique may not offer the extensive diversity and highquality masks needed for optimal model training. Recently, a promising line of research (Zhang et al, 2021c; Baranchuk

- et al, 2022; Li et al, 2022b) builds upon deep generative models to synthesize the massive number of images with pixel-level labels. However, existing approaches rely on training auxiliary segmentation architectures on top of GANs (Zhang et al, 2021c; Li et al, 2022b) or diffusion models (Baranchuk et al, 2022), limiting their applications only to a pre-defined set of classes.

In this study, we aim to address the challenge of limited labeled training data for large vocabulary instance segmentation, especially in long-tailed and open-set scenarios. Driven by the groundbreaking advancements in large-scale text-to-image (T2I) diffusion models—like Google’s Imagen (Saharia et al, 2022), OpenAI’s DALL-E 2 (Ramesh et al, 2022), and Stability AI’s Stable Diffusion (Rombach et al, 2022)—which excel in generating photorealistic images from free-form text prompts, we delve into producing vast amounts of synthetic data using T2I models to enhance instance segmentation training. Yet, this promising avenue comes with two primary hurdles: 1) How can we produce high-quality scene images featuring multiple objects? 2) How can we derive the corresponding per-pixel instance labels, namely, instance masks, from these diffusion models without additional model training or label supervision?

To this end, we propose MosaicFusion, a diffusion-based data augmentation pipeline to generate images and masks simultaneously for large vocabulary instance segmentation. Our method consists of two components: image generation and mask generation. For image generation, we first divide an image canvas into several regions. We then run the diffusion process on each region simultaneously, using the

shared noise prediction model, while conditioning on a different text prompt. In this way, we can control the diffusion model to generate multiple objects at specific locations within a single image. For mask generation, we first aggregate the cross-attention maps of the text token corresponding to a certain object across different layers and time steps in the diffusion process. We then threshold the aggregated attention maps and use standard edge-aware refinement algorithms, such as Bilateral Solver (BS) (Barron and Poole, 2016), to further refine the mask boundaries. The generated images and corresponding instance masks are finally used as a synthetic labeled dataset to train off-the-shelf models for instance segmentation.

Overall, our main contributions are summarized as follows:

- 1) We propose MosaicFusion, an automatic diffusion-

based data augmentation pipeline to expand the existing instance segmentation dataset. Our method can generate images and masks simultaneously without relying on additional off-the-shelf object detection and segmentation models to label the data further.

- 2) Our method allows us to generate customized objects

at specific locations in a single image. We study both singleobject and multi-object image generation scenarios and reveal that generating images with multiple objects is more beneficial than generating those with a single object.

- 3) Extensive experiments on two challenging bench-

marks, i.e., long-tailed and open-vocabulary instance segmentation on LVIS (Gupta et al, 2019), demonstrate that our method can significantly improve the performance of existing object detectors and instance segmentors, especially for rare and unseen categories. Figure 1 shows the non-trivial performance improvement achieved with MosaicFusion.

#### Table 1: Comparison with other diffusion-based data augmentation works in terms of key properties. Our MosaicFusion is the only method with all these desired properties

Properties Ge et al (2022) Zhao et al (2023) Li et al (2023) Ours

Training-free ✓ ✓ ✗ ✓ Directly generate multiple objects ✗ ✗ ✗ ✓ Agnostic to detection architectures ✓ ✓ ✗ ✓ Without extra detectors or segmentors ✗ ✗ ✗ ✓

### 2 Related Work

Text-to-image (T2I) diffusion models. Recent advances in large-scale generative models, such as Imagen (Saharia et al, 2022), DALL-E 2 (Ramesh et al, 2022), and Stable Diffusion (Rombach et al, 2022), have brought significant progress in AI-powered image creation by training on internet-scale text-image datasets. These models can be conditioned on free-form text prompts to generate photorealistic images. This enables improved controllability in personalized image generation (Gal et al, 2023), content editing (Hertz et al, 2023), zero-shot translation (Parmar et al, 2023), and concept customization (Kumari et al, 2022). Such great flexibility and scalability also bring the potential of transforming a T2I model as an effective training data generator. In particular, our work is more related to a strand of recent research (Hertz et al, 2023; Chefer et al, 2023; Xie

- et al, 2023; Phung et al, 2023) that uses cross-attention maps in T2I models for image synthesis and editing. Despite the great success, it remains unclear to what extent the visionlanguage correspondence residing in cross-attention maps can benefit visual perception tasks like instance segmentation. As opposed to their works, we make the first attempt to leverage such a free correspondence in diffusion models to synthesize images and masks for large vocabulary instance segmentation tasks. Data augmentation for instance segmentation. Instance segmentation models are data-hungry and label-expensive. Therefore, many works aim at improving the performance from the data augmentation perspective. Several earlier works adopt synthesis methods via rendering graphics (Su et al, 2015; Hinterstoisser et al, 2018) or copying from computer games (Richter et al, 2016). Due to the huge domain gap between synthetic and real data, another line of works (Dwibedi et al, 2017; Dvornik et al, 2018; Fang et al, 2019; Xie et al, 2021; Ghiasi et al, 2021) use real image sets (Gupta et al, 2019). For instance, Copy-Paste (Ghiasi et al, 2021) shows that pasting objects onto the background using their masks can work well. However, these methods are not scalable for large vocabulary settings since the augmented instances are still confined to existing ones in the training data, i.e., they cannot create new instances for rare or novel categories with substantially more diversity. In contrast, our goal is to generate multiple diverse rare or novel

objects on the same image with their masks. Our method is orthogonal to prior works using real data for augmentation, as verified in Sect. 4.2. Concurrently, several works (Ge et al, 2022; Zhao et al, 2023; Li et al, 2023) also use diffusion models for instance segmentation augmentation. The comparisons with these works are summarized in Table 1. Our MosaicFusion is the only method that is training-free, able to directly generate multiple objects and corresponding masks without relying on off-the-shelf detectors or segmentors, and compatible with various detection architectures.

Long-tailed instance segmentation. This task aims to handle class imbalance problems in instance segmentation. Most approaches adopt data re-sampling (Gupta et al, 2019; Liu et al, 2020; Wu et al, 2020), loss re-weighting (Ren et al, 2020; Tan et al, 2020a, 2021; Zhang et al, 2021b; Wang et al,

- 2021b) and decoupled training (Li et al, 2020; Wang et al, 2020). In particular, several studies (Liu et al, 2020; Zhan et al, 2020; Xie et al, 2022) adopt image-level re-sampling. However, these approaches result in bias of instance cooccurrence. To deal with this issue, several works (Hu et al, 2020; Wu et al, 2020; Zang et al, 2021) perform more finegrained re-sampling at the instance or feature level. For loss re-weighting, most approaches (Ren et al, 2020; Tan et al, 2020a; Wang et al, 2021a) rebalance the ratio of positive and negative samples during training. Meanwhile, decoupled training methods (Li et al, 2020; Wang et al, 2020) introduce different calibration frameworks to improve classification results. In contrast, MosaicFusion focuses on data augmentation and improving different detectors by generating new rare class examples. Open-vocabulary detection and segmentation. This task aims to detect and segment novel categories in a large concept space with the help of pre-trained vision-language models (VLMs) (Zhang et al, 2023; Wu et al, 2023). OVRCNN (Zareian et al, 2021) first puts forth the concept of open-vocabulary object detection. It is pre-trained on image-caption data to recognize novel objects and then fine-tuned for zero-shot detection. With the development of VLMs (Radford et al, 2021a; Jia et al, 2021), ViLD (Gu et al,
- 2022) is the first work to distill the rich representations of pre-trained CLIP (Radford et al, 2021a) into the detector. Subsequently, many studies (Li et al, 2022a; Zhong et al, 2022; Zang et al, 2022; Ghiasi et al, 2022; Du et al, 2022; Gao et al, 2022; Minderer et al, 2022; Chen et al, 2022;

Rasheed et al, 2022; Kuo et al, 2023; Xu et al, 2023; Zang et al, 2024) propose different ways to adapt VLM knowledge into open-vocabulary detection and segmentation. For example, DetPro (Du et al, 2022) introduces a fine-grained automatic prompt learning scheme, and F-VLM (Kuo et al, 2023) adopts frozen VLMs to output novel categories from cropped CLIP features directly. Another related work, Detic (Zhou et al, 2022b), improves the performance on novel categories with the extra large-scale image classification dataset (i.e., ImageNet-21K (Deng et al, 2009)) by supervising the max-size proposal with all image labels. However, it needs more training data and the vocabulary size is limited by the classification dataset. As verified in Sect. 4.3, MosaicFusion is orthogonal to the CLIP knowledge in the open-vocabulary setting, which boosts the state-of-the-art FVLM (Kuo et al, 2023) by a significant margin.

### 3 MosaicFusion

Our MosaicFusion is a training-free diffusion-based dataset augmentation pipeline that can produce image and mask pairs with multiple objects simultaneously using the off-theshelf text-to-image diffusion models. The overall pipeline of our approach is illustrated in Fig. 2. It contains two components: image generation and mask generation. In the following subsections, we first introduce some preliminaries w.r.t. latent diffusion models and cross-attention layers in Sect. 3.1. We then detail the image and mask generation process of MosaicFusion in Sect. 3.2 and Sect. 3.3, respectively.

- 3.1 Preliminary

Stable Diffusion. We build our data generation framework upon the state-of-the-art text-to-image latent diffusion model, i.e., Stable Diffusion (SD) (Rombach et al, 2022). SD runs the diffusion process in a compressed latent space rather than the pixel space for efficiency. It consists of three components: i) a variational autoencoder (VAE) (Kingma and Welling, 2014) that encodes and decodes the latent vectors for images; ii) a time-conditional U-Net (Ronneberger et al, 2015) that denoises the latent vectors at varying time steps; iii) a text encoder like CLIP (Radford et al, 2021b) that encodes the text inputs into textural embeddings. The pre-trained VAE encodes images as latent vectors for diffusion training. During inference, the generation process starts from a random Gaussian noise zT ∼ N(0,1) in the latent space, and is iteratively denoised conditioned on the text prompt p that is encoded via the text encoder. Specifically, at each denoising step t = 1,...,T, given zt and p, U-Net predicts the noise estimation term ϵ and subtracts it from zt to obtain zt−1. The final denoised latent z0 is then passed to the VAE decoder D to obtain the generated image I = D(z0).

Cross-attention layers. The vision-language interaction occurs during the denoising process in the U-Net backbone, where the embeddings of visual and textual features are fused using cross-attention layers that produce spatial attention maps for each textual token. Specifically, at each time step t, the spatial feature map of the noisy image zt is linearly projected into queries and reshaped as Qt ∈ Rn×hw×d. Similarly, the text prompt p is first encoded into textual embeddings via a text encoder, and then linearly projected into keys K ∈ Rn×l×d and values V ∈ Rn×l×d, respectively. The attention maps are the product between queries and keys:

At = Softmax

QtKT √

d

,At ∈ Rn×hw×l (1)

where d is the latent projection dimension, n is the number of attention heads, h and w are the height and width of the spatial feature map of the noisy image, and l is the number of text tokens in the text prompt. Intuitively, At[:,i,j] defines the probability assigned to the token j for the pixel i of the spatial feature map. Therefore, they can be used as a potential source to generate segmentation masks for specific text tokens.

3.2 Image Generation

Mosaic canvas. To generate multiple objects at specific locations in a single image, we first need to define an image canvas to customize a region of interest for each object. Formally, given a H × W image canvas, we divide it into a set of regions R = {Ri = (Xi,Yi,Wi,Hi),i = 1,...,N} in a Mosaic style. Here, without loss of generality, we take N = 4 as an example. Specifically, we first randomly jitter the center of the image canvas within a certain ratio range of W and H to obtain a Mosaic center (x,y),x ∈ [σW,(1 − σ)W],y ∈ [σH,(1 − σ)H], where σ ∈ (0,0.5] is a pre-defined ratio. We then use the Mosaic center (x,y) as the intersections of four candidate regions, i.e., R1 = (0,0,x,y),R2 = (x,0,W − x,y),R3 = (0,y,x,H − y),R4 = (x,y,W − x,H − y). To make different object regions more smoothly transitioned, we further allow a certain overlap for each neighborhood region. Let δx, δy denote the number of overlapped pixels in the horizontal and vertical direction of the canvas, respectively, we can thus obtain the final region coordinates: R1 = (0,0,x+

- δx/2,y + δy/2),R2 = (x − δx/2,0,W − x + δx/2,y +
- δy/2),R3 = (0,y − δy/2,x + δx/2,H − y + δy/2),R4 = (x − δx/2,y − δy/2,W − x + δx/2,H − y + δy/2). Prompt template. Given a Mosaic image canvas R, we then consider what kind of object, i.e., class of interest, to generate for each region. This is achieved by defining a customized text prompt for each region. Specifically, we use a

easel

| | |
|---|---|
| | |

[Figure 18]

reshape

| | | | | |
|---|---|---|---|---|

[Figure 19]

[Figure 20]

[Figure 21]

| | |
|---|---|
| | |

map

extract

threshold refine

filter & expand

Mosaic canvas

[Figure 22]

seaplane

[Figure 23]

reshape

[Figure 24]

[Figure 25]

| | | | | |
|---|---|---|---|---|

[Figure 26]

filter & expand

refine

threshold

extract

aggregate

text prompts

Diffusion

parrot

a photo of a single easel seaplane parrot Christmas tree ...

reshape

[Figure 27]

...

| | | | | |
|---|---|---|---|---|

[Figure 28]

[Figure 29]

[Figure 30]

threshold

refine

filter & expand

extract

###### Model

tree

reshape

[Figure 31]

[Figure 32]

| | | | | |
|---|---|---|---|---|

[Figure 33]

[Figure 34]

[Figure 35]

extract

threshold

refine

filter & expand

|image generation mask generation|
|---|

synthetic image cross-attention maps instance masks

- Fig. 2: Overview of our MosaicFusion pipeline. The left part shows the image generation process, while the right part shows the mask generation process. Given a user-defined Mosaic image canvas and a set of text prompts, we first map the image canvas from the pixel space into the latent space. We then run the diffusion process on each latent region parallelly with the shared noise prediction model, starting from the same initialization noise while conditioning on different text prompts, to generate the synthetic image with multiple objects specified in each region. Simultaneously, we aggregate the region-wise cross-attention maps for each subject token by upscaling them to the original region size in the pixel space and averaging them across all attention heads, layers, and time steps. After that, we binarize the aggregated attention maps, refine the boundaries, filter out the low-quality masks, and expand them to the size of the whole image canvas to obtain the final instance masks

generic text prompt template like “a photo of a single1 c, dc”, where c is the category name, dc is the category definition2. We randomly select N category names c associated with dc from a pre-defined set of interest categories to obtain N text prompts {pi,...,pN}. We then assign prompt pi to region Ri for the subsequent diffusion process, described next.

Diffusion process. Since latent diffusion models like SD run the diffusion process in the latent space, we need to adapt the image canvas from the pixel space into the latent space. Due to the fully convolutional nature of U-Net (and VAE), the coordinates in the latent space can be simply mapped to the pixel space by multiplying an upscaling factor U, whose value is equal to the upscaling factor of UNet in SD, i.e., U = 8. Thus, given the region coordinates Ri = (Xi,Yi,Wi,Hi) in the pixel space, we can obtain the corresponding region coordinates ri = (xi,yi,wi,hi) =

- 1 We use the word “single” to encourage the diffusion model to gen-

erate a single object at each specific location.

- 2 We empirically find that appending a category definition after the

category name reduces the semantic ambiguity of generated images due to the polysemy of some category names. For LVIS (Gupta et al,

- 2019), the category definitions are readily available in annotations, where the meanings are mostly derived from WordNet (Miller, 1995). See Table 2d for ablations on the effect of different prompt templates.

(Xi/U,Yi/U,Wi/U,Hi/U) in the latent space. To avoid fractional indices, we ensure (Xi,Yi,Wi,Hi) in the pixel space are divisible by U. After mapping the image canvas into the latent canvas, we then run N diffusion processes on different latent regions ri simultaneously with a shared noise-prediction network, starting from the same initialization noise while conditioning on different text prompts pi. Since all diffusion processes share the same network, the memory cost is essentially the same as that of the largest region affected by a single diffusion process. As a result, we can obtain a generated image with multiple objects specified in each region.

3.3 Mask Generation

Attention aggregation. As introduced in Sect. 3.1, the cross-attention maps play an important role in passing the information from text tokens to spatial image features. They indicate the influence of each token on each pixel and potentially serve as a good mask source. Formally, given the noised region latent zit and the corresponding text prompt pi,i = 1,...,N, we can obtain the cross-attention maps Ati through the U-Net network during the diffusion process

formulated in Sect. 3.2. Since U-Net contains a series of downsampling and upsampling blocks, we can obtain crossattention maps with varying scales from different layers. To make full use of the spatial and temporal information, we first bicubically resize all attention maps to the original region size (Hi,Wi) in the pixel space, and then average them across all attention heads, layers and time steps3, producing the final aggregated attention maps Aˆi ∈ RH

i×Wi×l. We extract the specific attention map Aˆsi ∈ RH

i×Wi along the last channel dimension of Aˆi for the subject token s that contains the interest category name c. We then nor-

malize Aˆsi within [0,1] and threshold it to a binary region mask Mˆi ∈ {0,1}H

i×Wi. In practice, we use Otsu’s method (Otsu, 1979) to automatically determine the binary threshold.

Edge refinement. The binarized attention maps provide relatively coarse masks of objects, as shown in Fig. 2. To further refine the mask boundaries, we adopt the standard edge refinement post-processing techniques such as Bilateral Solver (BS) (Barron and Poole, 2016) on top of the obtained coarse masks Mˆi to generate fine-grained masks M˜i ∈ {0,1}H

i×Wi.

Mask filtering. To further remove the low-quality masks, we filter the refined region masks M˜i based on some predefined criteria. Specifically, we first apply the connected component analysis (Di Stefano and Bulgarelli, 1999) on the refined masks to group the pixels into several disconnected regions. We then filter out masks with areas less than 5% or over 95% of the whole region since these masks are highly likely segmented incorrectly. After that, we only keep region masks that have one connected component per mask since we want to generate one object at each specific location. To obtain the final instance mask Mi ∈ {0,1}H×W, we expand the region mask M˜i ∈ {0,1}H

i×Wi to the size of the whole image canvas by padding 0 values for blank regions.

### 4 Experiments

- 4.1 Implementation Details 4.1.1 Datasets

We conduct our experiments of object detection and instance segmentation on the challenging LVIS v1.0 dataset (Gupta et al, 2019). LVIS is a large vocabulary instance segmentation dataset that contains 1203 categories with a long-tailed distribution of instances in each category. It has 100k images in the training set and 19.8k images in the validation set. Based on how many images each category appears in the training set, the categories are divided into three groups:

3 We show more details in the experiment section (see Table 2g, Table 2h, and Fig. 3) that averaging cross-attention maps across all layers and time steps is necessary to achieve the best performance.

rare (1-10 images), common (11-100 images), and frequent (>100 images). The number of categories in each group is: rare (337), common (461), and frequent (405). In the openvocabulary detection setting, the frequent and common categories are treated as base categories for training and the rare categories serve as novel categories for testing. The annotations of rare categories are not used during training.

- 4.1.2 Evaluation Metrics

We use the standard average precision (AP) as the evaluation metric, which is averaged at different IoU thresholds (from 0.5 to 0.95) across categories. We report the bounding-box AP and mask AP on all categories (denoted as APbox, APmask) as well as on the rare categories (denoted as APboxr , APmaskr ). We report the average of five independent runs following the best practice of LVIS challenge (Gupta et al, 2019).

- 4.1.3 MosaicFusion

We adopt the open-sourced Stable Diffusion v1.4 model with LMS (Karras et al, 2022) scheduler. We use a guidance scale factor of 7.5 and run 50 inference steps per image for all experiments. We keep the average region size as 384×5124 to generate each object. Thus, the final image and mask size depends on the number of objects we want to generate per image, e.g., 384 × 512 for one object, 384 × 1024 (or 768 × 512) for two objects, and 768 × 1024 for four objects.

- 4.1.4 Baseline Settings

We consider two challenging settings to verify the effectiveness of our approach, i.e., long-tailed instance segmentation and open-vocabulary object detection. Since MosaicFusion is agnostic to the underlying detector, we consider two popular baselines in long-tailed instance segmentation: Mask RCNN (He et al, 2017) and CenterNet2 (Zhou et al, 2021), and use state-of-the-art F-VLM (Kuo et al, 2023) in openvocabulary object detection. Our implementation is based on the MMDetection (Chen et al, 2019) toolbox. We detail each baseline below.

Mask R-CNN baseline. We follow the same setup in Gupta et al (2019). Specifically, we adopt ResNet-50 (He et al, 2016) with FPN (Lin et al, 2017) backbone using the standard 1× training schedule (Chen et al, 2019; Wu et al, 2019)

4 The image resolution used during training in Stable Diffusion is 512×512. We notice that the generated results will get worse if one deviates from this training resolution too much. Thus, we simply choose the aspect ratio of the average LVIS image and keep the longer dimension to 512.

(90k iterations with a batch size of 16)5. We use SGD optimizer with a momentum of 0.9 and a weight decay of 0.0001. The initial learning rate is 0.02, dropped by 10× at 60k and 80k iterations. Data augmentation includes horizontal flip and random resize short side [640,800], long side < 1333. We use repeat factor sampling with an oversample threshold of 10−3.

CenterNet2 baseline. We follow the same setup in Zhou

- et al (2022b). Specifically, we use Swin-B (Liu et al, 2021) with FPN backbone and a longer 4× training schedule (180k iterations with a batch size of 32, a.k.a., Box-Supervised). We use AdamW (Loshchilov and Hutter, 2019) optimizer. The initial learning rate is 0.0001, decayed with a cosine decay schedule (Loshchilov and Hutter, 2017). Data augmentation includes the EfficientDet (Tan et al, 2020c) style large-scale jittering (Ghiasi et al, 2021) with a training image size of 896 × 896. We use repeat factor sampling with an oversample threshold of 10−3 without bells and whistles. Compared with Mask R-CNN, Box-Supervised CenterNet2 is a stronger baseline that can better verify whether MosaicFusion is scaleable with larger backbones and longer training schedules. F-VLM baseline. We follow the same setup in Kuo
- et al (2023). Specifically, we use the pre-trained ResNet50/ResNet-50x64 CLIP (Radford et al, 2021b) model as the frozen backbone and only train the Mask R-CNN (He et al,

2017) with FPN detector head for 46.1k iterations with a batch size of 256. We use SGD optimizer with a momentum of 0.9 and a weight decay of 0.0001. The initial learning rate is 0.36, dropped by 10× at intervals [0.8,0.9,0.95]. Data augmentation includes large-scale jittering with a training image size of 1024 × 1024. The base/novel VLM score weight is 0.35/0.65, respectively. We set the background weight as 0.9, and the VLM temperature as 0.01. We use CLIP prompt templates and take the average text embeddings of each category. The state-of-the-art performance of F-VLM in open-vocabulary object detection makes it a strong baseline to verify the effectiveness of our MosaicFusion in the open-vocabulary setting.

- 4.2 Main Properties

We start by ablating our MosaicFusion using Mask R-CNN R50-FPN as the default architecture on the standard LVIS long-tailed instance segmentation benchmark. Several intriguing properties are observed.

Single object vs. multiple objects. We first study the effect of the number of generated objects per image. As shown in Table 2a, simply generating one (N = 1) object per image has already outperformed the baseline by a significant

5 We are aware that different works may use different notations for a 1× training schedule. In this work, we always refer 1× schedule to a total of 16 × 90k images.

margin (+4.4% APr, +0.7% AP). This indicates that synthetic images generated from diffusion models are indeed useful in improving the performance of instance segmentation tasks, especially for rare categories. Increasing the number of objects per image tends to further improve the performance, with the best performance achieved by setting N = 4 (+5.6% APr, +1.4% AP) as more objects in a single image provide more instances and increase the task difficulty. It is worth noting that a random variant, i.e., randomly selecting the number of objects per image among N = 1,2,4 rather than setting a fixed number achieves the sub-optimal performance compared with N = 4. This demonstrates that the number of objects matters more than the diversity of the spatial layout of the image canvas. See Fig. 4 for the visualization of some corresponding example images with the different number of generated objects.

Mosaic canvas design. We then study the different design choices of the Mosaic image canvas in Table 2b and Table 2c.

- Table 2b ablates the effect of the jittering ratio of the

Mosaic center. A larger ratio creates sub-regions with more diverse shapes. MosaicFusion works the best with a moderate jittering ratio (i.e., σ = 0.375). Jittering the center with a larger ratio tends to distort the image generation quality, especially for those of smaller regions.

- Table 2c further ablates the effect of the number of

overlapped pixels for each neighborhood region. Generating each object with a moderate region overlap (i.e., δx = 64, δy = 48) performs better than strictly separating each region for each object (i.e., δx = 0, δy = 0) since a certain overlap can allow a more smooth transition between different object regions, having a harmonization effect on the whole image. However, creating sub-regions with a larger overlap tends to degrade the performance as different objects will be highly entangled with each other.

Text prompt design. We compare different text prompt templates in Table 2d. Simply using the category name c as the text prompt has already surpassed the baseline by a clear margin (+3.9% APr, +1.1% AP). Decorating c with the prefix “a photo of a single” slightly improves the performance upon c (+0.5% APr, +0.2% AP). Appending the category definition dc after c leads to further performance gains (+1.2% APr, +0.1% AP). We believe that there exist better text prompts and leave more in-depth study on prompt design for future work.

Generated category types. Table 2e studies the types of interest categories to generate. The results demonstrate that generating images with objects from all categories (i.e., rare, common, and frequent, denoted as {r, c, f}) leads to better performance than generating those with objects from rare categories only (denoted as {r}). The improvements are consistent regardless of the metrics on rare categories or all categories (+1.0% APr, +0.7% AP).

- Table 2: Ablations for MosaicFusion on LVIS long-tailed instance segmentation. We use Mask R-CNN R50-FPN (1× schedule). The baseline is trained on the original training set, while others are trained on the training+synthetic set. We report mask AP on the validation set. Unless otherwise specified, the default settings are: (a) the number of generated objects per image is 4, (b) the center jitter ratio is 0.375, (c) the number of overlapped pixels is 64 in the horizontal direction and 48

in the vertical direction, respectively, (d) the text prompt template is “a photo of a single c, dc”, (e) the generated category types are all categories (i.e., rare, common, and frequent), (f) the number of generated images per category is 25, (g-h) the cross-attention maps are aggregated across all attention layers and time steps, and (i) the diffusion model is Stable Diffusion v1.4. The default entry is marked in gray

(a) Number of objects. Increasing the number of objects per image performs better

N APr AP

- 9.6 21.7

- 1 14.0 22.4
- 2 13.3 22.5 4 15.2 23.1

random 14.5 22.9

(d) Text prompt. Appending a category definition is more accurate

Prompt template APr AP

- 9.6 21.7 c 13.5 22.8

a photo of a single c 14.0 23.0 a photo of a single c, dc 15.2 23.1

(g) Attention layer. All layers contribute positively to the results

Layer resolution APr AP

- 9.6 21.7

≤ ×1/32 12.6 22.7 ≤ ×1/16 14.3 22.9

≤ ×1/8 15.2 23.1

(b) Center jitter ratio. Moderately jittering the canvas center works the best

σ APr AP

- 9.6 21.7

0.25 13.8 22.9 0.375 15.2 23.1

0.5 14.2 23.0

(e) Category type. Generating all categories leads to more gains

Category set APr AP

- 9.6 21.7

{r} 14.2 22.4 {r, c, f} 15.2 23.1

(h) Time step. All time steps are necessary for the best performance

Time step APr AP

- 9.6 21.7

≤ 10 14.3 22.8 ≤ 25 14.6 22.9 ≤ 50 15.2 23.1

(c) Overlapped pixels. Allowing a certain region overlap in the canvas is effective

(δx, δy) APr AP

- 9.6 21.7

(0, 0) 14.2 23.0 (64, 48) 15.2 23.1

(128, 96) 13.2 22.9

(f) Number of images. Generating 25 images per category is enough

# Images APr AP

- 9.6 21.7

10 13.1 22.5 25 15.2 23.1 50 14.7 23.0

(i) Diffusion model. MosaicFusion benefits from more advanced diffusion models

Stable Diffusion APr AP

- 9.6 21.7

- v1.2 12.7 22.6
- v1.3 14.4 22.9
- v1.4 15.2 23.1

- v1.5 15.5 23.2

Generated image numbers. Table 2f studies the effect of the number of generated images per interest category6. Only generating 10 images per category has already improved upon the baseline by a significant margin (+3.5% APr, +0.8% AP). A better performance is achieved by setting the number of generated images per category as 25 (+5.6% APr, +1.4% AP). Generating more images does not lead to further performance gains. We hypothesize that the diversity of generated images from existing diffusion models is a main challenge to scale up. We expect further performance improvement to be achieved with the emergence of more expressive diffusion models in the future.

6 Note that the final number of synthetic images per category is not fixed as we need to filter some images during the mask filtering process introduced in Sect. 3.3. We observe that ∼50% generated images and masks will be discarded accordingly.

Attention layers and time steps. We study whether aggregating cross-attention maps across all attention layers and time steps is necessary. Table 2g and Table 2h ablate the influence of layer resolutions and time steps, respectively. The results indicate that all layers and time steps contribute positively to the performance, proving that we should make full use of the spatial and temporal information during the diffusion process to produce the highest-quality attention maps for instance segmentation. See Fig. 3 for the visualization of some example cross-attention maps across different layers and time steps.

Diffusion models. In Table 2i, we study whether our MosaicFusion benefits from the improvement of diffusion models. We use Stable Diffusion v1.x (e.g., v1.2, v1.3, v1.4, and v1.5) as examples for ablations. A better diffusion model generally leads to better instance segmentation performance.

### Table 3: Comparison with Mosaic (Bochkovskiy et al,

- 2020) data augmentation. We use Mask R-CNN R50-FPN (1× schedule) as the baseline and report mask AP. Synthesizing multi-object images is more effective than simply combining single-object images

Table 4: Comparison with Copy-Paste (Ghiasi et al, 2021) data augmentation. We use Mask R-CNN R50-FPN (1× schedule) as the baseline and report mask AP. MosaicFusion is orthogonal to existing data augmentation methods like Copy-Paste

Method APr AP

Mask R-CNN (He et al, 2017) 9.6 21.7 w/ MosaicFusion (N = 1) 14.0 22.4 w/ MosaicFusion (N = 1) + Mosaic aug. 14.0 22.4 w/ MosaicFusion (N = 4) 15.2 23.1

Method APr AP

Mask R-CNN (He et al, 2017) 9.6 21.7 w/ Copy-Paste 10.6 22.1 w/ MosaicFusion 15.2 23.1 w/ MosaicFusion + Copy-Paste 15.5 23.3

Table 5: Comparison between SD and SDXL (Podell et al, 2023). We use Mask R-CNN R50-FPN (1× schedule) as the baseline and report mask AP. MosaicFusion benefits from a higher-resolution diffusion model

Method Region size APr AP

Mask R-CNN (He et al, 2017) - 9.6 21.7 w/ MosaicFusion (SD v1.5) 384 × 512 15.5 23.2 w/ MosaicFusion (SDXL base v1.0) 768 × 1024 16.2 23.4

Table 6: Effect of extra segmentation tools. We use Mask R-CNN R50-FPN (1× schedule) as the baseline and report mask AP. Using SAM (Kirillov et al, 2023) as an additional mask refiner yields further gains

Method APr AP

Mask R-CNN (He et al, 2017) 9.6 21.7 w/ MosaicFusion 15.2 23.1 w/ MosaicFusion + SAM 16.0 23.3

This makes sense as a more advanced diffusion model generates higher-quality images.

Comparison with other data augmentation methods. We compare our MosaicFusion with other existing data augmentation methods for instance segmentation.

We first compare MosaicFusion with Mosaic data augmentation proposed in the popular object detection framework YOLO-v4 (Bochkovskiy et al, 2020). It combines four training images into one with certain ratios after each image has been independently augmented. To ensure a fair comparison, we simply replace the multi-object images generated by MosaicFusion with the four combined generated single-object images using the Mosaic data augmentation while keeping other hyper-parameters unchanged. As shown in Table 3, simply adopting Mosaic augmentation to produce multi-object images does not bring further gains over the single-object version of our MosaicFusion (N = 1). In contrast, our multi-object version of MosaicFusion (N = 4) leads to further performance improvement compared with MosaicFusion (N = 1). This proves that using diffusion models to synthesize multi-object images is more effective than simply combining single-object images with the existing data augmentation counterpart.

We then show that our method is orthogonal to existing data augmentation methods. Here, we use the popular Copy-Paste (Ghiasi et al, 2021) as an example, which produces more instances by randomly copying the existing objects from one image and pasting them into another. The results are shown in Table 4. Either Copy-Paste or MosaicFu-

sion improves over the baseline, while MosaicFusion yields more performance gains. This makes sense as Copy-Paste only copies and pastes existing instances in the training data, whereas MosaicFusion creates new instances with substantially more diversity than existing ones. The diversity of instances matters more for rare categories. A slightly higher performance can be further achieved by combining them together, indicating that MosaicFusion is compatible with other data augmentation methods.

Effect of higher-resolution diffusion models. In Table 2i, we have studied the effect of diffusion models and observed that a better diffusion model generally leads to better instance segmentation performance. Here, we go beyond SD v1 and equip MosaicFusion with a more powerful highresolution diffusion model, e.g., SDXL (Podell et al, 2023). Specifically, we use SDXL base v1.0 to generate higherquality images with higher resolutions. We set the average region size as 768×1024 to generate each object while keeping all other settings in Table 2 unchanged. The results based on Mask R-CNN R50-FPN are shown in Table 5. Using a higher-resolution diffusion model brings more gains.

Effect of employing extra segmentation tools. So far, we have shown that MosaicFusion is the first training-free method that can generate multi-object images and masks without relying on extra detection and segmentation models. Here, we discuss whether incorporating additional segmentation tools further improves the performance. We employ an interactive segmentation model, i.e., SAM (Kirillov et al, 2023), to further refine the generated instance masks due to

- Table 7: LVIS long-tailed instance segmentation benchmark. We report box AP and mask AP on the validation set. We build our method upon two representative baselines: (i) Mask R-CNN baseline (1× schedule) in Gupta et al (2019) using the ResNet-50 w/ FPN backbone, and (ii) Box-Supervised CenterNet2 baseline (4× schedule) in Zhou et al (2022b) using the Swin-B backbone. MosaicFusion improves over the baselines regardless of the architectures with increased gains for the rare categories

Method Backbone APboxr APmaskr APbox APmask

Mask R-CNN (He et al, 2017) ResNet-50 9.1 9.6 22.5 21.7 w/ MosaicFusion ResNet-50 14.8 15.2 24.0 23.1 vs. baseline +5.7 +5.6 +1.5 +1.4

MosaicOS (Zhang et al, 2021a) ResNeXt-101 - 21.7 - 28.3 CenterNet2 (Zhou et al, 2021) ResNeXt-101 - 24.6 - 34.9 AsyncSLL (Han et al, 2020) ResNeSt-269 - 27.8 - 36.0 SeesawLoss (Wang et al, 2021a) ResNeSt-200 - 26.4 - 37.3 Copy-paste (Ghiasi et al, 2021) EfficientNet-B7 - 32.1 41.6 38.1 Tan et al. (Tan et al, 2020b) ResNeSt-269 - 28.5 - 38.8

Box-Supervised (Zhou et al, 2022b) Swin-B 39.9 35.9 45.4 40.7 w/ MosaicFusion Swin-B 43.3 39.0 46.1 41.2 vs. baseline +3.4 +3.1 +0.7 +0.5

its strong zero-shot generalization ability. Specifically, after obtaining the generated annotations from MosaicFusion, we prompt SAM with MosaicFusion-generated bounding boxes to produce the final instance masks. As shown in Table 6, using SAM-refined masks leads to further gains. Thus, if the training and inference costs of extra segmentors are not a major concern, we recommend using them to further improve the performance. Nevertheless, we keep not using extra models in the whole paper to solely verify the effectiveness of using diffusion models as data augmenters.

- 4.3 Comparison with Previous Methods

We perform system-level comparisons with previous methods on the long-tailed instance segmentation benchmark as well as the open-vocabulary object detection benchmark on LVIS.

Long-tailed instance segmentation benchmark. Table 7 presents our results on the long-tailed instance segmentation benchmark. MosaicFusion yields significant performance gains over the commonly-used Mask R-CNN baseline, with the most considerable improvement coming from APr (+5.7% APboxr , +5.6% APmaskr ). We observe consistent performance improvements when building upon a stronger Box-Supervised CenterNet2 baseline, obtaining over 3% higher APr (+3.4% APboxr , +3.1% APmaskr ). This demonstrates that MosaicFusion is agnostic to detector architectures and is an effective plug-and-play data augmentation method for existing instance segmentation models.

Open-vocabulary object detection benchmark. We further demonstrate that MosaicFusion is also a promising data

augmenter in the challenging open-vocabulary object detection benchmark. We build our method upon the state-ofthe-art open-vocabulary detector F-VLM (Kuo et al, 2023), which uses the pre-trained CLIP model as the frozen backbone and Mask R-CNN with FPN as the detector head. The results are shown in Table 8. MosaicFusion still consistently outperforms F-VLM across different backbones especially for APnovel (+2.6% APnovel for ResNet-50, +3.2% APnovel for ResNet-50x64), even though the baseline has incorporated the open-vocabulary knowledge from the CLIP model. This indicates that text-driven diffusion models and CLIP models are complementary to each other. We can achieve better performance by leveraging the benefits from both worlds.

4.4 Further Discussion

We further discuss how to evaluate the synthesized mask quality directly and compare MosaicFusion with other diffusion-based data augmentation methods quantitatively. Moreover, we examine the scalability of MosaicFusion when the model performance is already saturated and the vocabulary size is extremely large. Finally, we perform a case study to show that our general data augmentation pipeline can be potentially extended to broader multimodal understanding tasks.

Mask quality evaluation. In Sect. 4.2 and Sect. 4.3, we use the standard evaluation metrics (i.e., box AP and mask AP) in instance segmentation tasks for evaluation since our goal is to improve the instance segmentation performance, which is a common evaluation protocol used in previous works. Explicitly evaluating synthetic images and masks is chal-

- Table 8: LVIS open-vocabulary instance segmentation benchmark. We report mask AP on the validation set. We build our method upon state-of-the-art F-VLM (Kuo et al, 2023). Unless otherwise specified, all methods use the CLIP (Radford et al, 2021b) pre-training and fixed prompt templates. ∗: our reproduced results. ⋆: prompt optimization (Zhou et al, 2022a) and SoCo pre-training (Wei et al, 2021). †: joint training with ImageNet-21K (Deng et al, 2009). ‡: training with CC-

- 3M (Sharma et al, 2018)

Method Backbone Pre-trained CLIP APnovel AP

ViLD (Gu et al, 2022) ResNet-50 ViT-B/32 16.1 22.5 ViLD-Ens. (Gu et al, 2022) ResNet-50 ViT-B/32 16.6 25.5 DetPro⋆ (Du et al, 2022) ResNet-50 ViT-B/32 19.8 25.9 Detic-ViLD† (Zhou et al, 2022b) ResNet-50 ViT-B/32 17.8 26.8 RegionCLIP‡ (Zhong et al, 2022) ResNet-50 ResNet-50 17.1 28.2

F-VLM (Kuo et al, 2023) ResNet-50 ResNet-50 18.6 24.2 F-VLM∗ ResNet-50 ResNet-50 18.0 23.6 w/ MosaicFusion ResNet-50 ResNet-50 20.6 24.0

- vs. baseline +2.6 +0.4

ViLD (Gu et al, 2022) ResNet-152 ViT-B/32 18.7 23.6 ViLD-Ens. (Gu et al, 2022) ResNet-152 ViT-B/32 18.7 26.0 ViLD-Ens. (Gu et al, 2022) EfficientNet-B7 ViT-L/14 21.7 29.6 ViLD-Ens. (Gu et al, 2022) EfficientNet-B7 EfficientNet-B7 26.3 29.3 DetPro-Cascade⋆ (Du et al, 2022) ResNet-50 ViT-B/32 20.0 27.0 Detic-CN2† (Zhou et al, 2022b) ResNet-50 ViT-B/32 24.6 32.4 MEDet‡ (Chen et al, 2022) ResNet-50 ViT-B/32 22.4 34.4 Centric-OVD† (Rasheed et al, 2022) ResNet-50 ViT-B/32 25.2 32.9 RegionCLIP‡ (Zhong et al, 2022) ResNet-50x4 ResNet-50x4 22.0 32.3 OWL-ViT (Minderer et al, 2022) ViT-L/14 ViT-L/14 25.6 34.7

F-VLM (Kuo et al, 2023) ResNet-50x64 ResNet-50x64 32.8 34.9 F-VLM∗ ResNet-50x64 ResNet-50x64 31.7 34.9 w/ MosaicFusion ResNet-50x64 ResNet-50x64 34.9 35.3

- vs. baseline +3.2 +0.4

lenging as no ground truth is available. One na¨ıve way is to manually label all synthetic images with corresponding instance masks. However, it is extremely time-consuming and expensive. Here, we provide a more cost-effective metric to directly evaluate the synthesized mask quality. Specifically, we use SAM (Kirillov et al, 2023) as a data annotator due to its strong zero-shot generalization ability. We first prompt SAM with MosaicFusion-generated bounding boxes to produce instance masks. We then use these SAM-generated masks as ground truth and compute mean Intersection-overUnion (mIoU) between MosaicFusion-generated masks and ground truth masks. For reference, the mIoU result of MosaicFusion using the default setting in Table 2 is 77.2%. We hope our proposed metric can inspire future works on directly evaluating synthetic images and masks for rare and novel categories.

Comparison with other diffusion-based data augmentation methods. Several concurrent works (Ge et al, 2022; Zhao et al, 2023; Li et al, 2023) also use diffusion models for instance segmentation augmentation. We have discussed the differences with these works in Sect. 2 and Table 1. Here, we additionally provide a quantitative comparison with X-

Paste (Zhao et al, 2023)7. To ensure a fair comparison, we use the same baseline as in Zhao et al (2023), i.e., CenterNet2 (Zhou et al, 2021) with ResNet-50 backbone for a 4× training schedule (90k iterations with a batch size of 64). Considering that X-Paste uses additional retrieved images by the CLIP model to further boost its performance, we remove these retrieved images and only keep the generated ones. Table 9 reports the results on LVIS. Although X-Paste uses multiple extra foreground segmentation models to label the generated data further, our MosaicFusion can still achieve competitive performance with X-Paste. It should be well noted that X-Paste here uses 100k generated images while our MosaicFusion only uses 4k generated ones to achieve such results. As shown in Table 9, MosaicFusion is at least 4.3× more efficient than X-Paste. Thus, MosaicFusion is much more cost-effective than X-Paste. This demonstrates the superiority of directly synthesizing multi-object training images over pasting multiple synthesized instances onto the background to compose training images.

7 We choose X-Paste for comparison due to its open-sourced implementation. Note that Li et al (2023) uses a different setting by training and testing models on its own synthetic datasets. Thus, an apple-toapple quantitative comparison with Li et al (2023) is infeasible.

- Table 9: Comparison with X-Paste (Zhao et al, 2023) on LVIS. We use CenterNet2 with ResNet-50 (4× schedule) as the baseline, and report box AP and mask AP. We reimplement both the baseline and X-Paste using the official code of X-Paste8. All entries use the same network and training settings for a fair comparison. Note that X-Paste uses 100k generated images while MosaicFusion only uses 4k generated images. The dataset synthesis cost (GPU hours) is measured on a single A100 GPU. MosaicFusion is at least 4.3× more efficient than X-Paste. †: X-Paste requires four extra segmentation models to further label the generated data, whose costs are not included here

Method GPU hours APboxr APmaskr APbox APmask

baseline - 21.0 19.0 33.9 30.2 w/ X-Paste 100† 25.5 23.0 34.6 30.7 w/ MosaicFusion 23 25.8 23.4 34.7 30.9

- Table 10: Comparison with the state-of-the-art CoDETR (Zong et al, 2023) detector on LVIS. We report box AP on the validation set

Method Backbone APr AP

Co-DETR (Zong et al, 2023) Swin-L 61.4 64.5 w/ MosaicFusion Swin-L 64.4 65.1 vs. baseline +3.0 +0.6

Scalability in performance-saturated models. In Sect. 4.3, we have chosen several representative baselines to verify the effectiveness of our MosaicFusion for large vocabulary instance segmentation. Here, we further scale up MosaicFusion from the model axis by building upon a recent state-ofthe-art Co-DETR (Zong et al, 2023) detector to examine its effectiveness in performance-saturated models. Specifically, we choose the Objects365 (Shao et al, 2019) pre-trained CoDETR with Swin-L (Liu et al, 2021) backbone and fine-tune the detector on LVIS following the same setup in Zong et al (2023). We fine-tune for 16 epochs with a batch size of 16. We use AdamW (Loshchilov and Hutter, 2019) optimizer with a weight decay of 0.0001. The initial learning rate is 0.0001, dropped by 10× at the 8-th epoch. Data augmentation includes large-scale jittering with copy-paste (Ghiasi et al, 2021), with a training image size of 1536 × 1536. The results are shown in Table 10. After pre-training on Objects365 and reaching a plateau on LVIS (61.4% APr), MosaicFusion still yields +3.0% APr gains over Co-DETR,

8 We find that the baseline and X-Paste cannot be fully reproduced using the official code (see issues here). Therefore, our reproduced results are relatively lower than the original reported performance. Nevertheless, all experiments are done under the same settings for a fair comparison.

- Table 11: Open-vocabulary object detection on V3Det. We report box AP on the validation set. We build our method upon F-VLM (Kuo et al, 2023)

Method Backbone APnovel

Detic (Zhou et al, 2022b) ResNet-50 6.7 RegionCLIP (Zhong et al, 2022) ResNet-50 3.1

F-VLM (Kuo et al, 2023) ResNet-50 3.9 w/ MosaicFusion ResNet-50 7.4 vs. baseline +3.5

F-VLM (Kuo et al, 2023) ResNet-50x64 7.0 w/ MosaicFusion ResNet-50x64 14.5 vs. baseline +7.5

- Table 12: Referring expression segmentation benchmark. We report cIoU on the validation sets of RefCOCO (RefC), RefCOCO+ (RefC+), and RefCOCOg (RefCg). We build our method upon F-LMM (Wu et al, 2024), LISA (Lai et al, 2024), and GLaMM (Rasheed et al, 2024). For FLMM, we use DeepseekVL-1.3B (Lu et al, 2024) as the frozen LMM

Method RefC RefC+ RefCg

F-LMM-1.3B (Wu et al, 2024) 75.0 62.8 68.2 w/ MosaicFusion 77.1 65.3 69.2 vs. baseline +2.1 +2.5 +1.0

LISA-7B (Lai et al, 2024) 74.9 65.1 67.9 w/ MosaicFusion 76.9 67.9 69.7 vs. baseline +2.0 +2.8 +1.8

GLaMM-7B (Rasheed et al, 2024) 79.5 72.6 74.2 w/ MosaicFusion 80.6 73.6 75.0 vs. baseline +1.1 +1.0 +0.8

demonstrating its effectiveness in scenarios where the model performance is already saturated.

Scalability in extremely large-vocabulary size. We further scale up MosaicFusion from the data axis to a recent more challenging V3Det dataset (Wang et al, 2023) to examine its effectiveness in extremely large-vocabulary size. Specifically, V3Det is a vast vocabulary visual detection dataset containing 13204 categories, which is 10 times larger than the existing large vocabulary object detection dataset, such as 1203 categories in LVIS. It has 183k images in the training set, 29k images in the validation set, and 29k images in the testing set. For the open-vocabulary detection setting, the 13204 categories are split into 6709 base categories for training and 6495 novel categories for testing. We follow the default setup in Wang et al (2023) to train an openvocabulary detector on V3Det. Specifically, we use SGD optimizer with a momentum of 0.9, a weight decay of 0.0001,

and a learning rate of 0.02. We train all models using the standard 2× schedule with a batch size of 32. Table 11 reports the results. MosaicFusion significantly outperforms FVLM baselines across different backbones (+3.5% APnovel for ResNet-50, +7.5% APnovel for ResNet-50x64). It is worth noting that the performance improvements on V3Det are more pronounced than LVIS, demonstrating its promising scalability in more challenging data landscapes.

Extension to multimodal understanding tasks. Given the recent promise of large multimodal models (LMMs), we go beyond standard instance segmentation tasks and show that our MosaicFusion can also be potentially extended to multimodal understanding tasks. Specifically, we take referring expression segmentation (RES) as an example, which involves segmenting objects based on free-form human language descriptions. RES benchmarks (Kazemzadeh et al, 2014; Mao et al, 2016) include three commonlyused datasets, i.e., RefCOCO, RefCOCO+, and RefCOCOg. Compared with standard instance segmentation datasets, they have natural language expressions referring to specific objects in images. To generate the data with the corresponding format, we modify our pipeline by adding an additional referring expression generation process. Specifically, since the location of each object is specified in each region during synthesis and known as priori, we use the following generic templates to generate referring expressions at the same time: i) “c”, ii) “top/bottom left/right c”, iii) “c on top/bottom left/right”, and iv) “the c to the left/right of the c′ and above/below the c′′”, where c is the category name, c′ and c′′ are category names in the same image with adjacent regions to c. Take the synthetic image in Fig. 2 as an example, we can generate the following referring expressions for the “easel” category, respectively: i) “easel”, ii) “top left easel”, iii) “easel on top left”, and iv) “the easel to the left of the seaplane and above the parrot”. We generate four referring expressions for each object.

After generating both the visual and textual data, we choose several state-of-the-art LMM baselines to verify the effectiveness of our MosaicFusion. Specifically, we build our method upon F-LMM (Wu et al, 2024), LISA (Lai et al, 2024), and GLaMM (Rasheed et al, 2024). We follow the default setup in each baseline and fine-tune with our data. For F-LMM, we choose DeepseekVL-1.3B (Lu et al, 2024) as the frozen LMM and fine-tune a mask head for 8 epochs using a batch size of 8, with gradient clipping at a max norm of 1.0. We use AdamW optimizer with betas as (0.9, 0.999), a weight decay of 0.01, and a learning rate of 0.0001. For LISA, we fine-tune for 10 epochs using a batch size of 16, and a gradient accumulation step of 10. We use AdamW optimizer with betas as (0.9, 0.95), a weight decay of 0, and a learning rate of 0.0003. For GLaMM, we fine-tune for 5 epochs using a batch size of 16, and a gradient accumula-

tion step of 10. We use AdamW optimizer with betas as (0.9, 0.95), a weight decay of 0, and a learning rate of 0.0003.

As shown in Table 12, MosaicFusion consistently outperforms each LMM baseline, demonstrating its effectiveness in multimodal understanding tasks. It should be well noted that we only use several simple templates here to generate referring expressions. We expect the performance can be further improved by leveraging large language models to rewrite and generate more diverse and detailed text descriptions, which is, however, out of the scope of this paper. We leave more explorations on multimodal understanding tasks for future work.

4.5 Qualitative Results

Apart from quantitative results, we finally provide qualitative results of the cross-attention maps during the diffusion process as well as the synthesized images and masks by our MosaicFusion. Besides, we present and analyze some failure cases during image and mask synthesis.

Cross-attention maps. As illustrated in Sect. 3, crossattention maps in the diffusion process play a key role in producing our instance segmentation masks. We have studied the effects of attention layers and time steps on the final instance segmentation performance in Sect. 4.2 (see Table 2g and Table 2h). In Fig. 3, we visualize some example crossattention maps with respect to each interest word across different time steps and attention layers in the diffusion process, using the same synthetic image example as in Fig. 2. We can observe that the structure of the object is determined in the early steps of the diffusion process while more fine-grained object details emerge in the later steps. A similar phenomenon is observed across different layers. Specifically, low-resolution attention layers produce a coarse object shape while the high-resolution counterparts produce a fine-grained object edge or silhouette. In conclusion, each cross-attention map reflects the object composition to some extent. To make full use of the spatial and temporal information, we aggregate the cross-attention maps by averaging them across different time steps and layers, which produces the attention maps with the highest quality as visualized in Fig. 3. These aggregated attention maps can then serve as a good mask source to generate our final instance masks.

Synthesized images and masks. Our MosaicFusion can generate multiple objects with corresponding masks in a single image. We have studied the effect of the number of generated objects per image in Sect. 4.2 (see Table 2a). Here, we visualize some example results of our synthesized instance segmentation dataset by MosaicFusion. In Fig. 4, we show examples of generating N = 1,2,4 objects per image, respectively. Given only interest category names, MosaicFusion can directly generate high-quality multi-object images and masks by conditioning on a specific text prompt for each

t = 50 t = 1 average

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

⁄⁄⁄×18×116×132average

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

“easel”

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

|[Figure 57]|
|---|

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

[Figure 70]

⁄⁄⁄×18×116×132average

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

“seaplane”

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

|[Figure 86]|
|---|

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

synthetic image

⁄⁄⁄×18×116×132⁄⁄⁄average×18×116×132average

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

a photo of a single

easel seaplane parrot Christmas tree

“parrot”“tree”

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

text prompts

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

|[Figure 142]|
|---|

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

cross-attention maps

- Fig. 3: Visualization of cross-attention maps with respect to each interest subject word across different time steps and layers in the diffusion process. The time steps range from the first step t = 50 to the last step t = 1 in equal intervals (from left to right), while the layer resolutions range from ×1/32 to ×1/8 of the original image size (from top to bottom). In each entry, the last column shows the averaged cross-attention maps across different time steps, while the last row shows the averaged cross-attention maps across different layers. The highest-quality attention maps are produced by averaging them across both time steps and layers (bottom right framed in red)

[Figure 149]

𝑁=4𝑁=2𝑁=1

[Figure 150]

[Figure 151]

[Figure 152]

### Fig. 4: Visualization of our synthesized instance segmentation dataset by MosaicFusion. We show examples of generating N = 1,2,4 objects per image, using the settings in Table 2a

[Figure 153]

- Fig. 5: Visualization of failure cases during synthesis. MosaicFusion fails in some objects that are i) entangled (e.g., “spice rack” with spices, “chessboard” with chess), ii) small (e.g., “legume”, “tinsel”), and iii) abstract (e.g., “hardback book”)

region simultaneously without any further training and extra detectors or segmentors. The synthesized instance segmentation dataset can be used to train various downstream detection and segmentation models to improve their performances, especially for rare and novel categories, as verified in Sect. 4.3.

Failure cases. In Fig. 5, we present some failure examples during image and mask synthesis. MosaicFusion faces challenges in dealing with some i) entangled (e.g., “spice rack” with spices, “chessboard” with chess), ii) small (e.g., “legume”, “tinsel”), and iii) abstract (e.g., “hardback book”) objects. We attribute these phenomena to the inherent limitations of cross-attention in diffusion models. To mitigate these failure cases, one solution is to intervene in the generative process and refine the cross-attention maps to improve the controllability of diffusion models (Hertz et al, 2023; Chefer et al, 2023; Xie et al, 2023; Phung et al, 2023). Furthermore, instead of using one general prompt template, we can also leverage large language models such as GPT-

- 4 (OpenAI, 2023) and LLaMA (Touvron et al, 2023a,b) to automatically generate more diverse and detailed text prompts to reduce the abstractness and ambiguity of some categories.
- 5 Conclusion

More text-to-image diffusion models can be studied in future research. 2) We consider some representative object detection and instance segmentation models to examine the effectiveness of our method for large vocabulary instance segmentation due to resource constraints. Essentially, MosaicFusion is orthogonal to downstream detection and segmentation models, and it can be built upon all baseline methods in Table 7 and Table 8 to improve their performances. More baselines can be explored for further studies. 3) Despite the intriguing properties of MosaicFusion, the synthetic images still have an unavoidable domain gap with the real images due to the limited expressiveness of off-the-shelf diffusion models. As the first work to generate multiple objects and masks in a single image, we leave explorations on generating more complex scene-level images with diffusion models for future work. 4) We mainly focus on synthesizing data in general scenes. It is challenging to directly apply MosaicFusion to a more specific field such as industrial or medical sectors since existing diffusion models cannot synthesize high-quality data for such domain-specific scenarios. Future attention could be devoted to generating synthetic data tailored for more specific domains.

Data availability statements. All data supporting the findings of this study are available online. The LVIS dataset can be downloaded from https://www.lvisdataset.org/dataset.

The V3Det dataset can be downloaded from https: //v3det.openxlab.org.cn/. The RefCOCO, RefCOCO+, and RefCOCOg datasets can be downloaded from https://github.com/lichengunc/refer. The Stable Diffusion models used to generate data are available at https://github.com/CompVis/ stable-diffusion and https://github.com/ Stability-AI/generative-models.

Acknowledgements This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). The project is also supported by NTU NAP and Singapore MOE AcRF Tier 2 (MOE-T2EP20120-0001, MOET2EP20221-0012).

Instance segmentation is a fundamental task in computer vision. In this work, we have introduced a diffusion-based data augmentation method, namely MosaicFusion, for large vocabulary instance segmentation. Our method is trainingfree, equipped with multi-object image and mask generation capability without relying on extra models, and compatible with various downstream detection architectures. We hope our explorations can pave the way to unleash the potential of generative models for discriminative tasks.

Limitations. Our study has several limitations: 1) We choose a representative text-to-image diffusion model, i.e., Stable Diffusion, due to its open-sourced implementation.

### References

Baranchuk D, Rubachev I, Voynov A, Khrulkov V, Babenko A (2022) Label-efficient semantic segmentation with diffusion models. In: ICLR

Barron JT, Poole B (2016) The fast bilateral solver. In: ECCV Bochkovskiy A, Wang CY, Liao HYM (2020) Yolov4: Optimal speed

and accuracy of object detection. arXiv preprint arXiv:200410934

Chefer H, Alaluf Y, Vinker Y, Wolf L, Cohen-Or D (2023) Attend-andexcite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42(4):1–10 Chen K, Wang J, Pang J, Cao Y, Xiong Y, Li X, Sun S, Feng W, Liu Z, Xu J, Zhang Z, Cheng D, Zhu C, Cheng T, Zhao Q, Li B, Lu X, Zhu R, Wu Y, Dai J, Wang J, Shi J, Ouyang W, Loy CC, Lin D

(2019) MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:190607155

Chen P, Sheng K, Zhang M, Shen Y, Li K, Shen C (2022) Open vocabulary object detection with proposal mining and prediction equalization. arXiv preprint arXiv:220611134

Cordts M, Omran M, Ramos S, Rehfeld T, Enzweiler M, Benenson R, Franke U, Roth S, Schiele B (2016) The cityscapes dataset for semantic urban scene understanding. In: CVPR

Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L (2009) Imagenet: A large-scale hierarchical image database. In: CVPR Di Stefano L, Bulgarelli A (1999) A simple and efficient connected components labeling algorithm. In: ICIAP

Du Y, Wei F, Zhang Z, Shi M, Gao Y, Li G (2022) Learning to prompt for open-vocabulary object detection with vision-language model. In: CVPR

Dvornik N, Mairal J, Schmid C (2018) Modeling visual context is key to augmenting object detection datasets. In: ECCV Dwibedi D, Misra I, Hebert M (2017) Cut, paste and learn: Surprisingly easy synthesis for instance detection. In: ICCV

Fang HS, Sun J, Wang R, Gou M, Li YL, Lu C (2019) Instaboost: Boosting instance segmentation via probability map guided copypasting. In: ICCV

Gal R, Arar M, Atzmon Y, Bermano AH, Chechik G, Cohen-Or D

(2023) Designing an encoder for fast personalization of text-toimage models. arXiv preprint arXiv:230212228

Gao M, Xing C, Niebles JC, Li J, Xu R, Liu W, Xiong C (2022) Open vocabulary object detection with pseudo bounding-box labels. In: ECCV

Ge Y, Xu J, Zhao BN, Itti L, Vineet V (2022) Dall-e for detection: Language-driven context image synthesis for object detection. arXiv preprint arXiv:220609592

Ghiasi G, Cui Y, Srinivas A, Qian R, Lin TY, Cubuk ED, Le QV, Zoph B (2021) Simple copy-paste is a strong data augmentation method for instance segmentation. In: CVPR

Ghiasi G, Gu X, Cui Y, Lin TY (2022) Scaling open-vocabulary image segmentation with image-level labels. In: ECCV Gu X, Lin TY, Kuo W, Cui Y (2022) Open-vocabulary object detection via vision and language knowledge distillation. In: ICLR Gupta A, Dollar P, Girshick R (2019) Lvis: A dataset for large vocabulary instance segmentation. In: CVPR

Han J, Niu M, Du Z, Wei L, Xie L, Zhang X, Tian Q (2020) Joint coco and lvis workshop at eccv 2020: Lvis challenge track technical report: Asynchronous semi-supervised learning for large vocabulary instance segmentation. In: ECCVW

He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image

recognition. In: CVPR He K, Gkioxari G, Doll´ar P, Girshick R (2017) Mask r-cnn. In: ICCV Hertz A, Mokady R, Tenenbaum J, Aberman K, Pritch Y, Cohen-Or D

(2023) Prompt-to-prompt image editing with cross attention control. In: ICLR

Hinterstoisser S, Lepetit V, Wohlhart P, Konolige K (2018) On pretrained image features and synthetic images for deep learning. In: ECCV Workshops

Hu X, Jiang Y, Tang K, Chen J, Miao C, Zhang H (2020) Learning to segment the tail. In: CVPR

Jia C, Yang Y, Xia Y, Chen YT, Parekh Z, Pham H, Le Q, Sung YH, Li Z, Duerig T (2021) Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML

Karras T, Aittala M, Aila T, Laine S (2022) Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:220600364

Kazemzadeh S, Ordonez V, Matten M, Berg T (2014) Referitgame: Referring to objects in photographs of natural scenes. In: EMNLP Kingma DP, Welling M (2014) Auto-encoding variational bayes. In: ICLR

Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, Xiao T, Whitehead S, Berg AC, Lo WY, et al (2023) Segment anything. In: ICCV

Kumari N, Zhang B, Zhang R, Shechtman E, Zhu JY (2022) Multiconcept customization of text-to-image diffusion. arXiv preprint arXiv:221204488

Kuo W, Cui Y, Gu X, Piergiovanni A, Angelova A (2023) F-vlm: Openvocabulary object detection upon frozen vision and language models. In: ICLR

Kuznetsova A, Rom H, Alldrin N, Uijlings J, Krasin I, Pont-Tuset J, Kamali S, Popov S, Malloci M, Kolesnikov A, et al (2020) The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV

Lai X, Tian Z, Chen Y, Li Y, Yuan Y, Liu S, Jia J (2024) Lisa: Reasoning segmentation via large language model. In: CVPR Li B, Weinberger KQ, Belongie S, Koltun V, Ranftl R (2022a) Language-driven semantic segmentation. ICLR

Li D, Ling H, Kim SW, Kreis K, Fidler S, Torralba A (2022b) Bigdatasetgan: Synthesizing imagenet with pixel-wise annotations. In: CVPR

- Li Y, Wang T, Kang B, Tang S, Wang C, Li J, Feng J (2020) Overcoming classifier imbalance for long-tail object detection with balanced group softmax. In: CVPR
- Li Z, Zhou Q, Zhang X, Zhang Y, Wang Y, Xie W (2023) Guiding text-to-image diffusion model towards grounded generation. arXiv preprint arXiv:230105221

Lin TY, Maire M, Belongie S, Hays J, Perona P, Ramanan D, Doll´ar P, Zitnick CL (2014) Microsoft coco: Common objects in context. In: ECCV

Lin TY, Doll´ar P, Girshick R, He K, Hariharan B, Belongie S (2017) Feature pyramid networks for object detection. In: CVPR

Liu J, Sun Y, Han C, Dou Z, Li W (2020) Deep representation learning on long-tailed data: A learnable embedding augmentation perspective. In: CVPR

Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, Lin S, Guo B (2021) Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV

Loshchilov I, Hutter F (2017) Sgdr: Stochastic gradient descent with warm restarts. In: ICLR Loshchilov I, Hutter F (2019) Decoupled weight decay regularization. In: ICLR

Lu H, Liu W, Zhang B, Wang B, Dong K, Liu B, Sun J, Ren T, Li Z, Sun Y, et al (2024) Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:240305525

Mao J, Huang J, Toshev A, Camburu O, Yuille AL, Murphy K (2016) Generation and comprehension of unambiguous object descriptions. In: CVPR

Miller GA (1995) Wordnet: a lexical database for english. Commun ACM

Minderer M, Gritsenko A, Stone A, Neumann M, Weissenborn D, Dosovitskiy A, Mahendran A, Arnab A, Dehghani M, Shen Z, et al (2022) Simple open-vocabulary object detection with vision transformers. In: ECCV

OpenAI (2023) Gpt-4 technical report. arXiv preprint arXiv:230308774 Otsu N (1979) A threshold selection method from gray-level histograms. TSMC Parmar G, Singh KK, Zhang R, Li Y, Lu J, Zhu JY (2023) Zero-shot image-to-image translation. arXiv preprint arXiv:230203027 Phung Q, Ge S, Huang JB (2023) Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:230605427

Podell D, English Z, Lacey K, Blattmann A, Dockhorn T, M¨uller J, Penna J, Rombach R (2023) Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:230701952

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, et al (2021a) Learning transferable visual models from natural language supervision. In: ICML

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, et al (2021b) Learning transferable visual models from natural language supervision. In: ICML

Ramesh A, Dhariwal P, Nichol A, Chu C, Chen M (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:220406125

Rasheed H, Maaz M, Khattak MU, Khan S, Khan FS (2022) Bridging the gap between object and image-level representations for openvocabulary detection. In: NeurIPS

Rasheed H, Maaz M, Shaji S, Shaker A, Khan S, Cholakkal H, Anwer RM, Xing E, Yang MH, Khan FS (2024) Glamm: Pixel grounding large multimodal model. In: CVPR

Ren J, Yu C, Ma X, Zhao H, Yi S, et al (2020) Balanced meta-softmax for long-tailed visual recognition. In: NeurIPS Richter SR, Vineet V, Roth S, Koltun V (2016) Playing for data: Ground truth from computer games. In: ECCV Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) Highresolution image synthesis with latent diffusion models. In: CVPR Ronneberger O, Fischer P, Brox T (2015) U-net: Convolutional networks for biomedical image segmentation. In: MICCAI

Saharia C, Chan W, Saxena S, Li L, Whang J, Denton E, Ghasemipour SKS, Ayan BK, Mahdavi SS, Lopes RG, et al (2022) Photorealistic text-to-image diffusion models with deep language understanding. In: NeurIPS

Shao S, Li Z, Zhang T, Peng C, Yu G, Zhang X, Li J, Sun J (2019) Objects365: A large-scale, high-quality dataset for object detection. In: ICCV

Sharma P, Ding N, Goodman S, Soricut R (2018) Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: ACL

Su H, Qi CR, Li Y, Guibas LJ (2015) Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views. In: ICCV

Tan J, Wang C, Li B, Li Q, Ouyang W, Yin C, Yan J (2020a) Equalization loss for long-tailed object recognition. In: CVPR

Tan J, Zhang G, Deng H, Wang C, Lu L, Li Q, Dai J (2020b) 1st place solution of lvis challenge 2020: A good box is not a guarantee of a good mask. arXiv preprint arXiv:200901559

Tan J, Lu X, Zhang G, Yin C, Li Q (2021) Equalization loss v2: A new gradient balance approach for long-tailed object detection. In: CVPR

Tan M, Pang R, Le QV (2020c) Efficientdet: Scalable and efficient object detection. In: CVPR

Touvron H, Lavril T, Izacard G, Martinet X, Lachaux MA, Lacroix T, Rozi`ere B, Goyal N, Hambro E, Azhar F, et al (2023a) Llama: Open and efficient foundation language models. arXiv preprint arXiv:230213971

Touvron H, Martin L, Stone K, Albert P, Almahairi A, Babaei Y, Bashlykov N, Batra S, Bhargava P, Bhosale S, et al (2023b) Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:230709288

Wang J, Zhang W, Zang Y, Cao Y, Pang J, Gong T, Chen K, Liu Z, Loy CC, Lin D (2021a) Seesaw loss for long-tailed instance segmentation. In: CVPR

Wang J, Zhang P, Chu T, Cao Y, Zhou Y, Wu T, Wang B, He C, Lin D (2023) V3det: Vast vocabulary visual detection dataset. In: ICCV Wang T, Li Y, Kang B, Li J, Liew J, Tang S, Hoi S, Feng J (2020) The devil is in classification: A simple framework for long-tail instance segmentation. In: ECCV

Wang T, Zhu Y, Zhao C, Zeng W, Wang J, Tang M (2021b) Adaptive class suppression loss for long-tail object detection. In: CVPR

Waqas Zamir S, Arora A, Gupta A, Khan S, Sun G, Shahbaz Khan F, Zhu F, Shao L, Xia GS, Bai X (2019) isaid: A large-scale dataset for

instance segmentation in aerial images. In: CVPRW Wei F, Gao Y, Wu Z, Hu H, Lin S (2021) Aligning pretraining for detection via object-level contrastive learning. In: NeurIPS

Wu J, Song L, Wang T, Zhang Q, Yuan J (2020) Forest r-cnn: Largevocabulary long-tailed object detection and instance segmentation. In: ACM-MM

Wu J, Li X, Xu S, Yuan H, Ding H, Yang Y, Li X, Zhang J, Tong Y, Jiang X, Ghanem B, Tao D (2023) Towards open vocabulary learning: A survey. arXiv preprint arXiv:230615880

Wu S, Jin S, Zhang W, Xu L, Liu W, Li W, Loy CC (2024) Flmm: Grounding frozen large multimodal models. arXiv preprint arXiv:240605821

Wu Y, Kirillov A, Massa F, Lo WY, Girshick R (2019) Detectron2

- Xie J, Zhan X, Liu Z, Ong YS, Loy CC (2021) Unsupervised objectlevel representation learning from scene images. In: NeurIPS
- Xie J, Zhan X, Liu Z, Ong YS, Loy CC (2022) Delving into inter-image invariance for unsupervised visual representations. IJCV

Xie J, Li Y, Huang Y, Liu H, Zhang W, Zheng Y, Shou MZ (2023) Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In: ICCV

Xu S, Li X, Wu S, Zhang W, Li Y, Cheng G, Tong Y, Chen K, Loy CC

(2023) Dst-det: Simple dynamic self-training for open-vocabulary object detection. arXiv preprint arXiv:231001393

Zang Y, Huang C, Loy CC (2021) Fasa: Feature augmentation and sampling adaptation for long-tailed instance segmentation. In: ICCV Zang Y, Li W, Zhou K, Huang C, Loy CC (2022) Open-vocabulary detr

with conditional matching. ECCV Zang Y, Li W, Han J, Zhou K, Loy CC (2024) Contextual object detection with multimodal large language models. IJCV Zareian A, Rosa KD, Hu DH, Chang SF (2021) Open-vocabulary object detection using captions. CVPR

- Zhan X, Xie J, Liu Z, Ong YS, Loy CC (2020) Online deep clustering for unsupervised representation learning. In: CVPR

Zhang C, Pan TY, Li Y, Hu H, Xuan D, Changpinyo S, Gong B, Chao WL (2021a) Mosaicos: a simple and effective use of object-centric images for long-tailed object detection. In: ICCV

Zhang J, Huang J, Jin S, Lu S (2023) Vision-language models for vision tasks: A survey. arXiv preprint arXiv:230400685 Zhang S, Li Z, Yan S, He X, Sun J (2021b) Distribution alignment: A unified framework for long-tail visual recognition. In: CVPR

Zhang Y, Ling H, Gao J, Yin K, Lafleche JF, Barriuso A, Torralba A, Fidler S (2021c) Datasetgan: Efficient labeled data factory with minimal human effort. In: CVPR

- Zhao H, Sheng D, Bao J, Chen D, Chen D, Wen F, Yuan L, Liu C,

- Zhou W, Chu Q, Zhang W, Yu N (2023) X-paste: Revisiting scalable copy-paste for instance segmentation using clip and stablediffusion. In: ICML

Zhong Y, Yang J, Zhang P, Li C, Codella N, Li LH, Zhou L, Dai X, Yuan L, Li Y, et al (2022) Regionclip: Region-based languageimage pretraining. In: CVPR

Zhou K, Yang J, Loy CC, Liu Z (2022a) Conditional prompt learning for vision-language models. In: CVPR

- Zhou X, Koltun V, Kr¨ahenb¨uhl P (2021) Probabilistic two-stage detection. arXiv preprint arXiv:210307461

Zhou X, Girdhar R, Joulin A, Kr¨ahenb¨uhl P, Misra I (2022b) Detecting twenty-thousand classes using image-level supervision. In: ECCV Zong Z, Song G, Liu Y (2023) Detrs with collaborative hybrid assignments training. In: ICCV

