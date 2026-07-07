## Diffuse to Choose: Enriching Image Conditioned Inpainting in Latent Diffusion Models for Virtual Try-All

# arXiv:2401.13795v1[cs.CV]24Jan2024

Mehmet Saygin Seyfioglu* Karim Bouyarmane†

Suren Kumar Amir Tavanaei Ismail B. Tutar Amazon https://diffuse2choose.github.io

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 1. Diffuse to Choose (DTC) allows users to virtually place any e-commerce item in any setting, ensuring detailed, semantically coherent blending with realistic lighting and shadows.

#### Abstract

As online shopping is growing, the ability for buyers to virtually visualize products in their settings—a phenomenon we define as “Virtual Try-All”—has become crucial. Recent diffusion models inherently contain a world model, rendering them suitable for this task within an inpainting context. However, traditional image-conditioned diffusion models often fail to capture the fine-grained details of products. In contrast, personalization-driven models such

*University of Washington, work done during an internship at Amazon. †Correspondence at Amazon: bouykari@amazon.com

as DreamPaint are good at preserving the item’s details but they are not optimized for real-time applications. We present ”Diffuse to Choose,” a novel diffusion-based imageconditioned inpainting model that efficiently balances fast inference with the retention of high-fidelity details in a given reference item while ensuring accurate semantic manipulations in the given scene content. Our approach is based on incorporating fine-grained features from the reference image directly into the latent feature maps of the main diffusion model, alongside with a perceptual loss to further preserve the reference item’s details. We conduct extensive testing on both in-house and publicly available datasets, and show that Diffuse to Choose is superior to existing zero-

shot diffusion inpainting methods as well as few-shot diffusion personalization algorithms like DreamPaint.

#### 1. Introduction

The ever-growing demand for online shopping underscores the need for a more immersive shopping experience, allowing shoppers to virtually ‘try’ any product from any category (clothes, shoes, furniture, decoration, etc.) within their personal environments. The concept of a Virtual Try-All (Vit-All) model hinges on its functionality as an advanced semantic image composition tool. In practice, this involves taking an image from a user, selecting a region within that image, and using a reference product image from an online catalog to semantically insert the product into the selected area while preserving its details. For such a model to be effective, it must fulfill three primary conditions: 1) operate in any ’in-the-wild’ user image (not only on staged studios or professional human model images with predefined poses), and reference image, 2) integrate the reference product harmoniously with the surrounding context while maintaining the product’s identity (not replacing the product with a generic image of a product from similar category), and 3) perform fast inference to facilitate real-time usage across billions of products and millions of users.

Existing solutions tend to be specialized. Instead of aiming for a general purpose Vit-All approach, models are often developed for specific tasks and domains (model for clothing, model for furniture, model for eyeglasses, etc.). For example, early GAN-based works focused primarily on virtual try-on of clothing on human models in limited contexts or controlled environment (such as only certain clothing segments and no in-the-wild user images, or product images) [3, 5, 12, 17, 18, 38]. Other approaches to the problem utilize somewhat expensive 3D AR/VR technologies for items like furniture in rooms [1, 29], which is hard to scale to every single item on catalogs of billions of products that often lack 3D models. Consequently, a unified model offering a comprehensive Vit-All experience — one that enables consumers to digitally interact with any product from any category in any setting — is currently not available.

The emergence of diffusion models has marked a significant breakthrough in the generative capabilities of complex image modeling [25, 27, 33]. Unlike GANs, Diffusion models inherently grasp the nuances of the 3D world, exhibiting a degree of geometry and physics awareness, as demonstrated in inpainting tasks by [41], establishing their usefulness for Vit-All applications. A DreamBooth-based [26] technique, called DreamPaint [28], showed that Stable Diffusion [25] can be few-shot fine-tuned for the Vit-All use case. It can infer how to warp clothes to a person’s body, or how to place a certain furniture on a certain spot in a semantically correct manner without being explicitly trained

to do so. While DreamPaint meets the first two criteria to be an effective Vit-All model, it requires few-shot fine-tuning for each product separately, compromising its suitability for real-time applications thus failing to meet the third criterion.

A recently introduced image-referenced inpainting model, Paint By Example (PBE) [40], operates in a zeroshot setting and can handle in-the-wild images, meeting criteria one and three. However it encounters a limitation due to its reliance on an information bottleneck in its conditioning process, utilizing only the [CLS] token of the reference image. This constraint leads to an over-generalization of the reference image, degrading the model’s ability to maintain the fine-grained details essential for the Vit-All context, thus PBE fails to meet criterion two. Additionally, operating within a latent space, PBE struggles to retain fine-grained details of each item, underscoring the necessity for incorporating some form of pixel-level guidance.

In this work, we introduce “Diffuse to Choose” (DTC), a novel diffusion inpainting approach designed for the VitAll application. DTC, a latent diffusion model, effectively incorporates fine-grained cues from the reference image into the main U-Net decoder using a secondary U-Net encoder. Inspired by ControlNet [42], we integrate a pixellevel “hint” into the masked region of an empty image, which is then processed through a shallow convolutional network, ensuring dimensional alignment with the masked image processed by the Variational Autoencoder (VAE). DTC harmoniously blends the source and reference images, maintaining the integrity and details of the reference image. To further enhance alignment of basic features such as color, we employ perceptual loss using a pre-trained VGG model [7]. The complete architecture is illustrated in Fig. 2, with examples showcased in Fig. 1 and Fig. 5.

DTC effectively fulfills all three criteria for the Vit-All use case: 1) It efficiently handles in-the-wild images and references, 2) It adeptly preserves the fine-grained details of products while ensuring their seamless integration into the scene, and 3) It facilitates rapid zero-shot inference. We trained DTC on an in-house training dataset with sampled

- 1.2M source-reference pairs and a smaller public dataset, VITON-HD-NoFace [5]1. Our quantitative evaluations and human studies demonstrate that DTC surpasses all PBE variants — for which we implemented several enhancements to facilitate a fair comparison against DTC — and matches the performance of non-real-time, few-shot personalization models like DreamPaint [28], within the Vit-All context.
- 2. Related Work

Virtual Try-On. The primary goal of virtual try-on approaches is to create an image of a person wearing a tar-

1The VITON-HD public dataset was modified to remove and crop all model faces from the images in the dataset.

Source

Masked

Hint

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Mask

+

Resize and insert

[Figure 23]

VAE encoder

[Figure 24]

Adapter Image

[Figure 25]

[Figure 26]

MLP

+ encoder

Gaussian Noise Image Condition

[Figure 27]

Reference

U-Net encoder

U-Net encoder

Hint Signal

U-Net decoder

FiLM

Diffuse-To-Choose (DTC) training pipeline for Virtual Try-All (Vit-All)

Generated

[Figure 28]

Frozen modules

[Figure 29]

[Figure 30]

VAE decoder

Trainable modules

- Figure 2. Pipeline of Diffuse to Choose. The process initiates with masking the source image, followed by inserting the reference image within the masked area. This pixel-level ‘hint’ is then adapted by a shallow CNN to align with the VAE output dimensions of the source image. Subsequently, a U-Net Encoder processes the adapted hint. Then, at each U-Net scale, a FiLM module affinely aligns the skipconnected features from the main U-Net Encoder and pixel-level features from the hint U-Net Encoder. Finally, these aligned feature maps, in conjunction with the main image conditioning, facilitate the inpainting of the masked region. Red indicates trainable modules and blue indicates frozen modules.

get garment, ensuring that the garment’s fine-grained details are preserved and that it blends seamlessly with its surrounding context. Since this is a highly domain-specific and constrained problem (models and garments are often not presented in in-the-wild examples), existing models generally employ warp and paste (blend) techniques, leveraging extra inputs such as pose and human parse maps [3, 5, 7, 12, 17, 18, 37–39]. VITON [8] uses a twostep synthesis and refinement process, initially generating a coarse image with the desired clothing and then refining it to enhance the details. VITON-HD [5] aims for higherresolution images and employs alignment-aware segment normalization to correct misaligned parts. TryOnGAN [18] uses pose conditioning but relies on a purely latent model, which often lacks fine-grained details when representing garments. Given that GAN-based approaches are tailored exclusively for virtual try-on, which is a more narrowly defined problem compared to ours, and they do not possess

the broad mode coverage inherent to diffusion models, we have chosen to concentrate solely on diffusion models in our work. One of the latest diffusion-based virtual try-on models, Tryondiffusion [43], employs a dual U-Net approach on a pixel-level diffusion model. While it offers impressive performance for virtual try-on, it struggles with in-the-wild examples, supports only upper garments, and is not suitable for real-time use. Thus, a latent diffusion approach is necessary to ensure real-time inference in practical use cases. Regarding furniture, most existing works are AR-based 3D approaches from large corporations, which do not provide much detail about their models. In the 2D domain, there is DreamPaint [28], a DreamBooth-based [26] inpainting approach that few-shot fine-tunes the U-Net of Stable Diffusion, but it is not suitable for real-time applications.

Diffusion Based Image Editing. Creating realistic composites by superimposing an object from one image onto another is a common practice in photo editing and is closely

and its harmonious blend with the target image. A naive approach would be using the conventional PBE method, shown in Fig. 3. However, due to the information bottleneck caused by PBE’s reliance on only the [CLS] token for image conditioning, it tends to lose significant details of the reference image, resulting in unsatisfactory performance.

Source Masked Source Reference

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

VAE encoder VAE encoder

To rectify the shortcomings of the PBE in preserving the reference image’s details, we introduce the “Diffuse to Choose” (DTC) method. DTC leverages an auxiliary UNet alongside the primary U-Net within a latent diffusion model, specifically Stable Diffusion v1.5 [25]. The purpose of the auxiliary U-Net is to protect the details of the reference image that might be lost due to both the latent nature of the Stable Diffusion model and the limitations of image conditioning. To this end, we directly infuse fine-grained details of the reference image into the main U-Net’s decoder via affine transformations, ensuring preservation of the reference product’s details in the generated image. Our pipeline is shown in Fig. 2.

[Figure 36]

+

[Figure 37]

Gaussian Noise

Image encoder

MLP

U-Net encoder

Image Condition

Paint By Example pipeline applied on the Vit-All training dataset

U-Net decoder

Generated

[Figure 38]

[Figure 39]

VAE decoder

##### 3.1. Diffusion Inpainting Models

- Figure 3. Pipeline of Paint by Example [40] for Vit-All case. Red are trainable and blue are frozen. Orange pathways indicate skip connections between the encoder and the decoder.

For the Vit-All inpainting task, our objective is as follows: Given a user provided source image, xs ∈ RH×W×3, and a user-defined mask, m, (is a binary matrix of dimension 0,1HxW), with zeros indicating editable regions, and a reference image xr, showcasing the desired product, the objective is to seamlessly incorporate the product image xr within the mask-defined region of xs, ensuring the preservation of xr’s details. Diffusion models provide unparalleled success in image generation and specific tasks such as inpainting [24, 25, 27, 31]. These models follow a Markovian process, gradually introducing noise, denoted as ϵ from N(0,1), to xs over various timesteps t to transform it into an isotropic Gaussian distribution zt. The process is then reversed by iteratively predicting and subtracting the added noise to convert zt back to xs, conditioned by c. In the context of inpainting, this can be mathematically expressed as:

aligned with the Vit-All task. Image Editing, particularly inpainting, has been extensively explored in diffusion models. Initially, there were text-based image editing models [2, 9, 13, 14, 25]. However, it is evident that text alone cannot capture the fine-grained details necessary for accurately describing a product, necessitating the use of image conditioning. DCCF [38] introduced pyramid filters for image composition, followed by Paint by Example [40], which conditions the diffusion model using CLIP embeddings of the reference image. However, relying solely on the [CLS] token often leads to an overgeneralization of the reference image, making it unsuitable for the Vit-All task. Similarly, ObjectStitch [34] combines image and text embeddings to guide the model but faces challenges in conveying finegrained details. In response to these challenges, we introduce Diffuse to Choose, a novel latent diffusion inpainting model. Our model effectively leverages fine-grained details from the reference image, ensuring both the preservation of the product’s fine-grained details and its seamless integration into the chosen location, while working in a zero-shot setting with any in-the-wild image.

t,ϵ,t ∥ϵθ ((m ⊙ xs),zt,t,c) − ϵ∥22 (1)

L = Ez

Here, xs is the source image, m the user-defined mask, zt the noise-added version of xs, and c denotes the embeddings of xr. PBE uses the [CLS] token of CLIP [23] for c, a deliberate information bottleneck, because it relies on self-referencing, and additional patches often leads to copypaste artifacts. However, for the Vit-All task, it is practical to compile a dataset with distinct source and reference images of the same object, thereby eliminating this bottleneck. Consequently, we introduced a series of enhancements to PBE to explore the upper limits of basic imageconditioned inpainting models in the Vit-All context and establish a stronger baseline. Our modifications included

#### 3. Method

We formulate Vit-All as an image-conditioned inpainting task, wherein a single product image is integrated into a user-specified region within a user-specified image, ensuring the preservation of the product’s fine-grained details

while keeping the inpaint image in a latent form, provides complementary signals that yield superior results.

Source

Masked Source

###### Hint

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Mask

Combining Hint Signal with Main U-Net. The Stable Diffusion U-Net Encoder generates feature maps of varying resolutions at each level, consisting of 13 blocks, including the middle layer. The direct addition of the Hint Encoder’s outputs to the skip connections of the Main U-Net Encoder at every level tends to result in a pronounced spatial influence from the Hint signal, which is often not spatially aligned with the source image, thus negatively affecting the performance. In addition to direct addition, we explore two strategies for integrating the Hint Signal into the main UNet: Using Feature-wise Linear Modulation (FiLM) [22], and Cross Attention, computed in a manner akin to [43]. Among these three approaches—direct addition, FiLM, and Cross Attention—FiLM emerges as the most effective. We argue this is due to the image conditioning already capturing the majority of low-level details from the reference image, with mostly fine-grained details being absent. FiLM specifically enhances those feature channels that are essential for preserving the fine-grained details of the reference image.

Adapter

[Figure 44]

VAE encoder +

U-Net encoder

- Figure 4. Hint signal is stitched into a blank image within the masked region, then summed up with latent masked input before fed into Auxiliary U-Net.

using all CLIP patches instead of just the [CLS] token, employing a larger image encoder DINOV2 [21], and adding a refinement loss similar to [7] alongside the diffusion loss given in Eq. 1. Each of these alterations incrementally improved the performance of the PBE approach.

Hinting Strategies, Refinement Loss and Image Encoder. Our objective is to convey pixel-level, fine-grained details from the reference image into the main U-Net, and there are several methods to achieve this. One approach is to focus on high-frequency details by employing techniques like Canny Edges or Holistically-Nested Edge Detection (HED) features. Alternatively, we can directly use the reference image itself. In our experiments, we tested Canny edge extraction using the implementation in the OpenCV library [4], with minimum and maximum threshold values of 30 and 140, and for slightly softer edges, we used the HED model [36]. Despite these strategies yielding comparable results, directly using the reference image proved to be the most effective as it conveys the entire spectrum of details from the reference image, rather than focusing solely on high-frequency details. Thus, instead of pre-filtering to only convey the high-frequency details, it is a better approach to let the FiLM layer decide the most important channels, thus capturing the essential nuances of the reference image.

##### 3.2. Design of Diffuse to Choose

Creating the Hint Signal. Drawing inspiration from ControlNet [42], we propose the incorporation of a secondary U-Net encoder, which serves as a trainable replica of the main U-Net encoder. In the ControlNet architecture however, the secondary U-Net is integrated directly into the main U-Net decoder, providing spatial conditioning. In contrast, DTC demonstrates that the secondary UNet, rather than providing a spatial layout, can serve to guide the main U-Net by exerting a potent pixel-wise influence from the reference image during the decoding process. To generate the hint signal, we start by creating an image of zeros, identical in size to the source image, xs ∈ RH×W×3. Subsequently, we resize the reference image and insert it within the designated mask coordinates within the image of zeros. The same mask is then applied to xs, and this masked source image undergoes processing by the VAE encoder to yield a latent representation, sized 64×64×4. The Hint image is subsequently processed by the Adapter module —a shallow convolutional network comprising four layers— to match with the dimensions of the masked latent inpaint image. Finally, the Hint image and the masked source are added elementwise to produce the final representation of the hint image, which is then processed by the replicated U-Net encoder. This process is not shown in Fig. 2 to keep it concise, but is illustrated in Fig. 4 for clarity. Through a series of ablation studies, we demonstrate that maintaining a distinct representation for the hint image at the pixel level,

For our image encoder, we use DINOV2 [21], which outputs 256×1536 dimensional embeddings to represent a reference image, which is then reduced to 256×768 by a trainable MLP layer. Finally, we utilized a perceptual loss using a pre-trained VGGNet [30], which is computed by comparing the feature maps from the first five layers of VGGNet for both the source and the generated images, thereby implicitly ensuring the alignment of basic features like color.

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

###### Figure 5. DTC can handle variety of e-commerce products and can generate images using in-the-wild images & references.

#### 4. Experiments

##### 4.1. Dataset and Implementation Details

Data. We compiled an in-house training dataset composed of product images. Fortunately, e-commerce products often have multiple images available, so, during training, we do not need to adhere to the self-reference setting employed by PBE, where the reference image is derived from xs, leading to potential overfitting. However not all products yield useful xs,xr pairs, as many product images feature only the product against a white background. While these images are apt for use as xr, they are unsuitable for xs since we require images of products in contextual settings (with a natural background). To address this, we employed an inhouse model to identify products that have at least one xs image depicting the product in a natural setting, interacting with other elements in the scene, and one image with the product itself xr, where we collect images against a white background if it exists. Finally, we use GroundingDINO [19] and SAM [15] alongside with the product type of the xs to create the inpainting mask within xs. From the resulting dataset, we sampled a sample training dataset of 1.2M samples, evenly split between wearables and furniture. To ensure accessibility and reproducibility, we also train and test our model on a public dataset modified to remove model faces, VITONHD-NoFace [5], which provides xr against a white background, masks, and xs where individuals (with removed faces) are wearing xr.

Implementation details. We use a latent diffusion model, Stable Diffusion [25] V1.5 as our backbone in our experiments. Our image resolution is 512 × 512 and we train our model with DDPM [11] using a constant learning rate of 1e − 5 in both PBE and DTC implementations. We use simple augmentations like rotation and flip but avoid strong augmentations given in [40], as we don’t rely on self-referencing. We also use classifier free guidance [10] in similar fashion to [42]. During inference, we use DDIM [32] with guidance scale of 5, and for the hint input we stitch the reference image into the largest rectangular bounding box within the arbitrary-shape binary mask. We use 8 NVIDIA A100 40G GPUs to train our model for 40 epochs.

##### 4.2. Paint by Example Ablations

To ascertain the optimal performance of the naive imageconditioned inpainting approach [40] and create the strongest possible baseline, we implemented a series of modifications to the architecture illustrated in Fig. 3. Originally, PBE utilized self-reference conditioning, which involved cropping the xr from within xs. However, in the Vit-All context, we circumvent this limitation by having separate images for xr. As a result, instead of using only the [CLS] token from CLIP [23] in the Image Encoder, we incorporated all CLIP patches alongside the [CLS] token.

Subsequently, we increased the capacity of the image encoder by adopting DINOV2 [21], a larger and purely imagebased model. Furthermore, akin to [7], we integrated a perceptual loss with the diffusion loss. This involved using an ImageNet pre-trained VGGNet [30] to foster alignment of basic features, such as color and certain textures, between the generated and the source images. A qualitative example of this implementation is presented in Fig. 6.

Without Perceptual With Perceptual Loss

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Figure 6. Effect of Perceptual Loss.

##### 4.3. Diffuse-to-Choose Ablations

Hint Pathway Ablations. It is possible to directly insert the reference image xr into the masked source image m ⊙ xs and process it with a VAE, circumventing the use of the adapter network to align the hint image’s resolution with the latent masked image prior to addition. However, this approach produce suboptimal results, yielding a Clip Score of 86.97 and an FID of 6.26 on our Vit-All dataset, in contrast to the non-latent Hint insertion’s Clip Score of 88.14 and FID of 5.72. We hypothesize that maintaining the hint signal at the pixel level could introduce additional information that is overlooked in its VAE encoded latent counterpart. This indicates that the VAE might be excluding certain features during the encoding process.

Ablations on Hinting Strategies. We explored alternatives to directly inserting the reference image into the masked source image. These alternatives included using Canny Edges and HED features, both of which are designed to convey the high-frequency details that are absent in image-only conditioning. However, we observed a slight underperformance with both HED and Canny edges compared to the direct use of the reference image. This was evidenced by the CLIP scores, which were 87.85 for Canny and 86.98 for HED, compared to 88.14 for direct usage on our Vit-All dataset. Similarly, the FID scores were 6.11 for Canny and 6.57 for HED, against 5.72 for direct insertion.

Ablations of Techniques for Integrating Hint Signal into the Main U-Net. There are multiple ways to merge the signals from the hint U-Net and the main U-Net before incorporating the combined signal into the main U-Net decoder. We explored three approaches: direct addition, the use of an affine transformation layers FiLM [22], and the

- Table 1. Quantitative comparison between DTC variants and PBEbest, which denotes a PBE variant using DINOv2 and perceptual loss. CA denotes Cross-Attention.

Method CLIP Score (↑) FID (↓)

PBEbest 85.43 6.65 Oursaddition 86.94 6.19 OursCA 88.01 5.68 OursFiLM 88.14 5.72

integration of more computationally expensive Cross Attention layers [43]. Results shown on Tab. 1 revealed that both FiLM and Cross Attention layers outperform direct addition. Also, Cross Attention and FiLM yield comparable results, and FiLM is cheaper to compute, therefore we chose to use FiLM in our final model.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Figure 7. Failure cases with generating fine-grained text.

- 4.4. Evaluation and Comparisons

Comparison Against Paint by Example Variants. We implemented a series of enhancements to PBE and trained each variant on VITONHD-NoFace dataset. The results are presented in Table 2. As anticipated, using all CLIP patches surpasses the performance of using only the [CLS] token, which is limited to encoding a generalized version of xr. Furthermore, augmenting the size of the image encoder by using DINOv2 notably enhances performance. Notably, the addition of perceptual loss provides a marginal improvement in scenarios where the model initially struggled with basic features, such as color. While PBE, particularly with DINOv2 and perceptual loss, is adept at handling basic items with minimal details, it often falls short in the inpainting of detailed items. In contrast, DTC exhibits superior performance, especially in preserving the fine-grained details of items. Figure 9 illustrates the outcomes achieved with certain enhancements.

Comparisons Against Few-Shot Personalization Meth-

- Table 2. Quantitative comparison of PBE variants on VITONHDNoFace [5].

Method CLIP Score (↑) FID (↓)

PBE CLIPcls[40] 82.43 9.54 + PBE CLIPall 84.01 8.93 + PBE DINOv2 87.48 6.18 + PBE perceptual 87.79 5.93 Ours 90.14 5.39

Table 3. The average results of the small-scale human evaluation study. Similarity evaluates the resemblance of the inpainted region to the reference image, while Semantic Blending assesses the accuracy of the reference image’s integration within its context.

Method Similarity (↓) Semantic Blending (↓)

PBEbest 3.7 3.13 DreamPaint [28] 2.83 2.53 Ours 2.9 2.5

ods. While personalization methods such as DreamBooth [26] do not support inpainting, the recently introduced DreamPaint approach [28] enables similar few-shot finetuning of the U-Net in a masked setting, allowing for the generation of specified concepts at user-defined locations. However, DreamPaint requires few-shot fine-tuning with multiple product images, taking about 40 minutes per product to be trained. We manually selected 30 samples to compare DTC with DreamPaint and PBE. Visual comparisons are presented in Fig. 8. Furthermore, we conducted a subjective human survey, the results of which are tabulated in Table 3. A total of 20 participants scored each image on a scale from 1 to 5, with 1 being the best, based on both the inpainted region’s similarity to the reference image and its contextual blending. The results show that DTC, despite being a zero-shot model, performs on par with DreamPaint, which requires few-shot fine-tuning with multiple xr.

#### 5. Conclusion and Limitations

Limitations. DTC has limitations. Despite our efforts to inject fine-grained details, the model may still overlook finegrained details, particularly in text engravings, a challenge inherent to Stable Diffusion (see Fig. 7). Additionally, the model might alter human poses since it doesn’t consider pose, leading to discrepancies with pose-agnostic masking, especially for full-body coverage (see Fig. 10 in the Appendix). Introducing pose conditioning could mitigate this, but we prioritized a general-purpose model over taskspecific auxiliary inputs for broader applicability.

Conclusion. In this paper, we introduced ”Diffuse to Choose,” a novel image-conditioned diffusion inpainting model designed for the Virtual Try-All, aiming to integrate e-commerce items into user images while preserving item details. Our main contributions include employing a secondary U-Net to infuse fine-grained signals from the reference image into the primary U-Net decoder using basic affine transformation layers within a latent diffusion model. Moreover, we refined the PBE model for peak performance achievable with straightforward image-conditioned inpainting models. We compared DTC with upgraded PBE variants and a few-shot personalization methods using datasets like VITONHD-NoFace and a larger in-house dataset and show that DTC outperforms existing diffusion based in-

Source & Reference PBE Best DreamPaint Diffuse to Choose (Ours)

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

Figure 8. Qualitative comparison against PBE and DreamPaint.

source & ref PBE PBE All CLIP Patches PBE DINOV2 PBE DINOV2+perceptual Ours

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Figure 9. PBE variants’ performance vs DTC

painting approaches in the Virtual Try-All setting.

#### References

- [1] Mathieu Aubry, Daniel Maturana, Alexei A Efros, Bryan C Russell, and Josef Sivic. Seeing 3d chairs: exemplar partbased 2d-3d alignment using a large dataset of cad models. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3762–3769, 2014. 2
- [2] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 4
- [3] Shuai Bai, Huiling Zhou, Zhikang Li, Chang Zhou, and Hongxia Yang. Single stage virtual try-on via deformable attention flows. In European Conference on Computer Vision, pages 409–425. Springer, 2022. 2, 3

- [4] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000. 5
- [5] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14131–14140, 2021. 2, 3, 7, 8
- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2
- [7] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. arXiv preprint arXiv:2308.06101, 2023. 2, 3, 5, 7
- [8] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In

- Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7543–7552, 2018. 3
- [9] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 4
- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 7
- [12] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2, 3
- [13] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 4
- [14] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2426– 2435, 2022. 4
- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 7, 1
- [16] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2
- [17] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022. 2, 3
- [18] Kathleen M Lewis, Srivatsan Varadharajan, and Ira Kemelmacher-Shlizerman. Tryongan: Body-aware try-on via layered interpolation. ACM Transactions on Graphics (TOG), 40(4):1–10, 2021. 2, 3
- [19] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 7
- [20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 1
- [21] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5, 7, 1
- [22] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a

- general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, 2018. 5, 7
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 7, 1
- [24] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 4
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 7, 1
- [26] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 3, 8
- [27] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2, 4
- [28] Mehmet Saygin Seyfioglu, Karim Bouyarmane, Suren Kumar, Amir Tavanaei, and Ismail B Tutar. Dreampaint: Fewshot inpainting of e-commerce items for virtual try-on without 3d modeling. arXiv preprint arXiv:2305.01257, 2023. 2, 3, 8
- [29] Ka Chun Shum, Hong-Wing Pang, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Conditional 360degree image synthesis for immersive indoor scene decoration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4478–4488, 2023. 2
- [30] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 5, 7
- [31] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 4
- [32] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 7
- [33] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [34] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Object-

- stitch: Object compositing with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18310–18319, 2023. 4
- [35] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2
- [36] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015. 5
- [37] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gpvton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23550–23559, 2023. 3
- [38] Ben Xue, Shenghui Ran, Quan Chen, Rongfei Jia, Binqiang Zhao, and Xing Tang. Dccf: Deep comprehensible color filter learning framework for high-resolution image harmonization. In European Conference on Computer Vision, pages 300–316. Springer, 2022. 2, 4, 3
- [39] Keyu Yan, Tingwei Gao, Hui Zhang, and Chengjun Xie. Linking garment with person via semantically associated landmarks for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17194–17204, 2023. 3
- [40] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391,

2023. 2, 4, 7, 8, 3

- [41] Guanqi Zhan, Chuanxia Zheng, Weidi Xie, and Andrew Zisserman. What does stable diffusion know about the 3d scene? arXiv preprint arXiv:2310.06836, 2023. 2
- [42] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 5, 7
- [43] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4606–4615,

2023. 3, 5, 8

## Diffuse to Choose: Enriching Image Conditioned Inpainting in Latent Diffusion Models for Virtual Try-All

### Supplementary Material

##### 5.1. Masking Strategy During Training and Inference

During training, with equal probability, we alternate between a fine-grained mask (where we only mask the item specifically) and bounding box shaped masks (covering the largest bounding box spanned by the fine-grained mask). For each case, we stitch the reference image within the largest rectangular shape inside the mask. This approach is straightforward in the case of rectangular masks. However, for fine-grained masks, we calculate the largest rectangular area within the binary mask. Initially, we construct a histogram for each row of the matrix, with each entry in the histogram representing the cumulative height of masked areas in the column up to that row. We then calculate the maximum area rectangle that can be formed in each histogram, updating the coordinates of the largest rectangle as we iterate through the rows. This process ultimately yields the top-left and bottom-right coordinates of the largest rectangle fitting inside the mask. An example is shown in Fig. 11. During inference, we stitch the hint image within the largest rectangular region of the mask.

Garment-Only Mask Pose Agnostic Mask

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Figure 10. Pose agnostic masking case.

##### 5.2. Implementation Details and Inference Performance

In all our experiments, we used Stable Diffusion v1-5 [25]. For our image encoder, we employed DINOV2 [21], which outputs 1536-dimensional vectors for every patch of the reference image, of shape 224 × 224 × 3. Thus, it yields 256 × 1536-dimensional outputs. Additionally, we appended the CLS token to obtain 257 × 1536 image conditioning vectors. Subsequently, these vectors were processed through a single layer of a fully connected network, which was trained from scratch, to reduce them to 257 × 768 dimensions. We trained our model using AdamW [20] with a constant learning rate of 1e−5 and used horizontal flip and

rotation as augmentations. To calculate the CLIP score, we used ViT-B/32 [23]. Finally, the model is efficient in inference, taking ≈ 6 seconds to run a single pass on an A100 (40GB) GPU with 40 DDIM steps.

##### 5.3. The effect of masking

Since our approach relies on collages, the mask serves as a strong prior for the DTC model. As illustrated in Fig. 12, the use of masking enables users during inference to manipulate clothing styles. Consequently, users can guide the model to generate a t-shirt in a tucked-in style, or with sleeves rolled up, among other variations.

##### 5.4. Iterative Inpainting

DTC enables a range of enjoyable applications. For instance, users can begin with an empty room and iteratively decorate it, designing as shown in Fig. 13. The same principle applies to clothing; users can generate multiple items of clothing in combination with one another to experiment with different outfit combinations, as shown in Fig. 14.

##### 5.5. Visualization of Hint Signal

As mentioned, in addition to direct stitching, we also utilized Canny Edges and HED edges on our hint pathway, as demonstrated in Fig. 16. For Canny Edges, we used sobel filters for each color channel independently and then combined the results to obtain RGB edge information, which we believed could more effectively convey the details of ecommerce items.

##### 5.6. More on Limitations

For certain items, such as shoes, the model frequently fails to generate satisfactory results. We argue that this issue stems from SAM’s [15] inability to generate appropriate masks specifically for shoes or, more broadly, for items presented in pairs. SAM often masks only one shoe of a pair, leading the model to learn shortcut features from the unmasked shoe during training, rather than acquiring useful, generalizable features. And as mentioned, since we use a latent diffusion model as our backbone, no matter how much extra information we guide it with, we are subject to the capacity of VAE decoder, which often fails to generate very fine grained concepts like detailed engravings etc.

##### 5.7. Comparison Against Other Methods.

Most state-of-the-art GAN-based methods are tailored for single-domain applications, such as virtual try-ons in con-

Reference Image

[Figure 100]

Resize & Stitch

Mask

||[Figure 101]|
|---|
|
|---|

|[Figure 102]| |
|---|---|
| | |

|[Figure 103]| |
|---|---|
| | |

Insert bbox into an image of zeros

Find the largest bbox

- Figure 11. We find the largest rectangular bounding box inside an fine-grained binary mask. Then the same coordinates are used to stitch the reference image into an image of zeros to create the initial hint signal.

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

- Figure 12. DTC allows users to manipulate different styles of the same clothing by adjusting the mask (given in the row above for each image). The first two columns display variations of the same t-shirt, showcasing it both tucked out and tucked in. The third and fourth columns illustrate the same shirt with normal sleeves and with sleeves rolled up.

trolled environments with sanitized backgrounds, and often necessitate additional inputs like pose or depth maps. Also, it is already established that diffusion-based approaches are superior to GANs in performance, possessing more comprehensive world models [41]. Consequently, diffusion-based models are more apt for the Vit-All use case.

The scope of our comparison models is intentionally limited. Personalization models such as ELITE [35], Custom Diffusion [16], DreamBooth [26], and Textual Inversion [6] lack inpainting capabilities, as they aim to directly generate entire views. DreamPaint [28] is the only exception with inpainting support. Among the models that facilitate inpaint-

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

Figure 13. Diffuse to Choose allows users to iteratively decorate an empty room from scratch.

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

Figure 14. Diffuse to Choose allows users to iteratively try out combination of clothes.

ing, including PBE [40] and DreamPaint, we attempted to employ DCCF [38]. However, its tendency to create copypaste artifacts made it unsuitable for the Vit-All task, where semantically blending the item with its environment is as crucial as preserving its detailed features.

##### 5.8. More Examples

We provide more qualitative examples to showcase DTC’s capabilities. Please see Fig. 17, 18, 15 for more examples.

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

###### Figure 15. Some clothing try-on results. Note that DTC can handle in the wild reference and source images.

Reference Image Canny Edges HED

[Figure 156]

[Figure 157]

[Figure 158]

Figure 16. Different reference image representations that we use on hint pathway.

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

###### Figure 17. Additional examples showcasing different products. Note that DTC can infer how the product should look like, given a zero-shot example.

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

[Figure 184]

[Figure 185]

[Figure 186]

###### Figure 18. Additional examples showcasing different products.

