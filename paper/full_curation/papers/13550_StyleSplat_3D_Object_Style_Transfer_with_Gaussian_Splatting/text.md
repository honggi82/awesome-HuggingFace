# arXiv:2407.09473v1[cs.CV]12Jul2024

## StyleSplat: 3D Object Style Transfer with Gaussian Splatting

##### Sahil Jain⋆, Avik Kuthiala⋆, Prabhdeep Singh Sethi, and Prakanshul Saxena

Carnegie Mellon University

Abstract. Recent advancements in radiance fields have opened new avenues for creating high-quality 3D assets and scenes. Style transfer can enhance these 3D assets with diverse artistic styles, transforming creative expression. However, existing techniques are often slow or unable to localize style transfer to specific objects. We introduce StyleSplat, a lightweight method for stylizing 3D objects in scenes represented by 3D Gaussians from reference style images. Our approach first learns a photorealistic representation of the scene using 3D Gaussian splatting while jointly segmenting individual 3D objects. We then use a nearestneighbor feature matching loss to finetune the Gaussians of the selected objects, aligning their spherical harmonic coefficients with the style image to ensure consistency and visual appeal. StyleSplat allows for quick, customizable style transfer and localized stylization of multiple objects within a scene, each with a different style. We demonstrate its effectiveness across various 3D scenes and styles, showcasing enhanced control and customization in 3D creation.

Keywords: 3D Gaussian Splatting · Style Transfer

### 1 Introduction

Breakthroughs in radiance field generation have revolutionized the capture and representation of 3D scenes, allowing for unprecedented levels of detail and realism. The ability to seamlessly transfer artistic styles to objects extracted from these radiance fields offers a transformative approach for industries such as gaming, virtual reality, and digital art. This technology not only enhances creative expression but also significantly reduces the time and effort required to produce visually stunning 3D assets, pushing the boundaries of what is possible in digital content creation.

The emergence of 3D Gaussian splatting (3DGS) [14] has introduced a powerful method for representing radiance fields, offering the advantage of fast training and rendering while preserving high-quality details. Prior to this advancement, neural radiance field (NeRF) based methods [2, 25, 26] have been extensively utilized for reconstructing detailed scenes, and many techniques [19,21,35] aim at transferring artistic styles to individual objects within NeRFs. However, the slow training and rendering speeds of NeRFs pose significant challenges for their practical use.

⋆ equal contribution

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

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Fig. 1: We introduce StyleSplat, an approach for lightweight, customizable, and localized stylization of 3D objects from reference style images. Our approach first learns a photorealistic representation of the scene with 3D Gaussian splatting while jointly segmenting the scene into individual 3D objects using 2D masks. We then employ a nearest-neighbor feature matching loss to finetune and stylize the user-specified objects using the provided style images.

Approaches have also been introduced in the 3DGS paradigm [23, 29] that enable the transfer of style in real time. However, they apply the style globally to entire scenes without providing mechanisms for localized application to individual objects. Furthermore, although text-based approaches [31] can facilitate object-specific style transfer through text instructions using an imageconditioned diffusion model, the reliance on textual descriptions introduces ambiguities, particularly in accurately conveying specific colors, styles, or textures within 3D scenes.

In this paper, we introduce StyleSplat, a lightweight method for stylizing 3D objects in scenes represented using 3D Gaussians from arbitrary style images. Our approach allows for the stylization of multiple objects within a scene, each with a different style image. StyleSplat consists of three steps: 2D mask generation and tracking, 3D Gaussian training & segmentation and finally, 3D style transfer. In the first step, we utilize off-the-shelf image segmentation and tracking models to obtain consistent 2D masks across the complete scene to generate temporally coherent mask identifiers. Then, these masks are used as supervision for segmenting the 3D Gaussians into distinct objects while jointly optimizing their geometry and color, allowing for accurate object selection. Finally, we use a nearest-neighbor feature matching loss to finetune the selected Gaussians by aligning their spherical harmonic coefficients with the provided style image to achieve consistency and visual appeal. This method provides accurate and focused stylization, resulting in more customized and high-quality outcomes.

Our method produces visually pleasing results across diverse scenes and datasets, highlighting its versatility with different artistic styles.

### 2 Related Work

Style transfer on 3D scenes has attracted attention in recent years, with notable contributions emerging for both NeRF and 3DGS representations. Recently, several techniques have also been introduced for object editing within the 3DGS framework. Here, we review these prior efforts to provide context for our proposed approach.

#### 2.1 Style Transfer

Image style transfer, a long-standing challenge in computer vision, involves optimizing a content image and a reference style image to create a new image. The resulting image maintains the content of the original while adopting the style of the reference. Early works [7,8] utilize VGG-Net [30] to extract multi-level features and iteratively optimize a Gram matrix loss and a content loss to render a new image. Since then, numerous works have been proposed to improve various aspects of style transfer, including faster stylization [10, 13, 32] and improving semantic consistency and texture preservation [9,17,22]. Several approaches have been proposed to constrain the style transfer to specific objects in the image. [4] simultaneously segments and stylizes individual objects in an image and uses a Markov random field for anti-aliasing near the object boundaries for seamless blending. [18] performs localized style transfer in real-time by generating object masks using a fast semantic segmentation deep neural network. [27] utilizes Segment Anything (SAM) [16] to generate object masks and contain the style transfer to specific objects and uses a novel loss function to ensure smooth boundaries between the stylized and non-stylized parts of the image.

#### 2.2 Style Transfer in Radiance Fields

Several works have been introduced to extend stylization to 3D scenes. ARF [37] presents a method for transferring artistic styles to 3D scenes represented by neural radiance fields. Their approach effectively incorporates artistic features into rendered images while maintaining multi-view consistency by using a nearest neighbor feature matching (NNFM) loss for 3D stylization instead of the Gram matrix loss. However, ARF applies style changes to the entire scene, modifying all objects within the view, which might not be desirable in applications where style transfer is intended for specific objects. Building upon ARF, several approaches were introduced to localize style transfer to user-specified objects. ARF-Plus [21] introduces perceptual controls for scale, spatial area, and perceived depth and uses semantic segmentation masks to ensure that stylization is applied only to selected areas. S2RF [19] leverages Plenoxel’s [28] grid based representation for the 3D scene and a masked NNFM (mNNFM) loss to constrain the style

transfer only to desired areas. CoARF [35] extends S2RF by adding a semantic component to style transfer using LSeg [20] features in addition to VGG features. However, these methods suffer from slow rendering speeds due to their reliance on ray marching. In the 3DGS paradigm, StyleGaussian [23] and GSS [29] introduce approaches for scene stylization. Both methods seek to generate novel views of a scene using unseen style images at test time, after being trained on a large dataset of style images. Similar to ARF [37], these approaches are limited to stylizing the entire scene.

#### 2.3 Object Editing in 3D Gaussian Splatting

A common approach for localized editing in Gaussian splatting involves appending additional features to each Gaussian to encode semantic information. These features are optimized by rendering feature maps similar to RGB rasterization and using 2D feature maps or segmentation masks from foundation models as guidance. GaussianEditor [5] adds a scalar feature to each Gaussian to identify whether the Gaussian is in the editing region of interest (ROI). It then uses 2D grounded segmentation masks to optimize this feature. Gaussian Grouping [33] appends a feature vector to each Gaussian and uses masks from SAM [16] and DEVA [6] as guidance. Similarly, Feature 3DGS [38] distills LSeg [20] and SAM features into each Gaussian for promptable editing. All these approaches demonstrate text-guided editing as a downstream task. However, textual descriptions in style transfer can be ambiguous, making it difficult to accurately specify particular colors, styles, or textures within 3D scenes. Several diffusion-based methods have also been proposed. Instruct-GS2GS [31] utilizes a 2D diffusion model to modify the appearance of 3D objects with text instructions but it fails to constrain the changes to the specified object. TIP-Editor [39] personalizes a diffusion model using LoRA and accepts both an image and a text prompt with a 3D bounding box for local editing. Fine-tuning typically requires 5 to 25 minutes with this approach. By contrast, our method is lightweight and achieves results in less than a minute.

#### 2.4 Concurrent Work

Several interesting works have emerged that focus on localized image-conditioned editing of 3D Gaussians. One such work is StylizedGS [36], which emphasizes scene stylization while allowing for spatial control through the use of 2D masks, allowing different styles for different regions. However, since the approach uses only 2D masks, it faces significant limitations. Since each Gaussian has a volume, alpha blending may inadvertently include neighboring Gaussians in the computation graph during rasterization, causing unintended style transfer to other parts of the scene. This results in the inability to precisely stylize a single object while leaving the rest of the scene unchanged. Another notable work is ICE-G [11]. Unlike our method, ICE-G copies the style image to the ROI for a single 2D view. It then employs SAM and DINO [3] to propagate the style to multiple

views, effectively updating the data set with the desired style. Fine-tuning for style transfer is subsequently performed on this updated dataset.

### 3 Method

Given a reference style image, a set of posed images, and objects specified by the user, we aim to achieve fast novel view synthesis such that the objects corresponding to the user input are stylized according to the reference style image. Our approach involves three steps: 2D mask generation & object tracking, 3D Gaussian training & segmentation, and 3D style transfer. Fig. 2 provides a brief overview of our method.

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

- Fig. 2: Our approach for StyleSplat. We first use an off-the-shelf segmentation and tracking model [6] to generate view-consistent 2D object masks. Then, we use the multi-view images to learn the geometry and color of 3D Gaussians while simultaneously learning a per Gaussian feature vector. These feature vectors are decoded into object labels using a linear classifier to collect the Gaussians corresponding to the userspecified objects. The SH coefficients of these selected Gaussians are finetuned to align with the style image using NNFM loss.

#### 3.1 Preliminary: 3D Gaussian Splatting

3D Gaussians [14] is an explicit 3D scene representation. Each 3D Gaussian is characterized by a covariance matrix Σ and a center point µ , which is referred to as the mean value of the Gaussian as:

- 1

- 2xTΣ−1x (1)

G(x) = e−

For differentiable optimization, the covariance matrix Σ can be decomposed into a scaling matrix S and a rotation matrix:

Σ = RSSTRT (2)

When rendering novel views, differential splatting is employed for the 3D Gaussians within the camera planes. As introduced by [40], using a viewing transform matrix W and the Jacobian matrix J of the affine approximation of the projective transformation, the covariance matrix Σ′ in camera coordinates can be computed as:

Σ′ = JWΣWTJT (3)

In summary, each 3D Gaussian is characterized by the following attributes: position X ∈ R3, color defined by spherical harmonic (SH) coefficients C ∈ Rk (where k represents number of SH coefficents), opacity α ∈ R, rotation factor r ∈ R4, and scaling factor s ∈ R3. Specifically, for each pixel, the color and opacity of all the Gaussians are computed using the Gaussian’s representation Eq. 1. The blending of N ordered points that overlap the pixel is given by the formula:

- i−1
- j=1

(1 − αi) (4)

C = Σciαi

Here, ci,αi represents the density and color of this point computed by a 3D Gaussian G with covariance Σ multiplied by an optimizable per-point opacity and SH color coefficients.

#### 3.2 2D Mask Generation & Object Tracking

Before segmenting the 3D scene, we need to acquire accurate 2D segmentation masks on the entire sequence. These masks need to be temporally coherent to ensure that the different class indices correspond to the same object across frames. We treat the captured images as a video sequence and make use of a DEcoupled Video segmentation Approach (DEVA) [6] with a strong zero-shot segmentation model (SAM) [16] to get temporally coherent masks.

##### 3.3 3D Gaussian Training & Segmentation We follow the approach of Gaussian Grouping [33] to jointly train and segment

- 3D Gaussians. More specifically, each 3D Gaussian is given a learnable compact feature vector of length 16. These encodings are optimized similar to the spherical harmonic coefficients in the 3DGS pipeline. For a given view, the feature vector for a single pixel is evaluated as follows:

- i−1
- j=i

eiαi′

(1 − αj′ )

Eid =

i∈N

where e′i is the feature vector for the ith Gaussian, and αi′ is the influence of the ith Gaussian on the current pixel, evaluated similar to [34]. The rendered

Eid for each pixel is then passed through a classifier to provide a class label. A cross-entropy loss is used between the predicted class labels vs the class labels obtained in the first stage. Additionally, a spatial consistency loss is added which ensures that the feature vectors for the top-k nearest 3D Gaussians are similar. Once this stage is completed, all Gaussians in the scene corresponding to the same object have similar feature vectors.

#### 3.4 3D Style Transfer

Once we have obtained a scene representation with segmented 3D Gaussians, we use the learned feature vectors to perform style transfer on user-specified objects. To get the mask IDs of specific objects in the scene, the user can specify a bounding box around the object or utilize Grounding DINO [24] to extract the IDs using a text prompt. We select the Gaussians corresponding to the objects of interest by passing the feature vectors through the trained classifier and filtering out Gaussians with activations less than a threshold. We also perform statistical outlier removal, eliminating Gaussians whose positions deviate significantly from their neighbors compared to the average for the scene. We then freeze all the properties of the selected Gaussians and enable gradients only for their SH coefficients. For each training view, we apply the nearest neighbor feature matching (NNFM) loss between the VGG features of the rendered image and the reference style image. The NNFM loss minimizes the cosine distance between the VGG feature of each pixel in the render with its nearest neighbor in the style image and is given by:

1 N i

(Fr(i) · Fs(j))

LNNFM =

min

j

where Fr are the VGG features for the render, Fs are the features for the style image, and N is the number of pixels in the rendered image. Since only the SH coefficients of the user-specified object are trainable, the style transfer is contained to a single object.

### 4 Results

To assess our method, we perform qualitative evaluations on multiple real-world scenes. We visually demonstrate the effectiveness of our method by successfully applying different styles to various objects in a diverse selection of scenes. This section highlights how our 3D segmentation approach for localized stylization prevents leakages (Sec. 4.2), its performance in both single-object (Sec. 4.3) and multi-object settings (Sec. 4.4), and scale control (Sec. 4.5). Finally, we qualitatively compare our approach with S2RF (Sec. 4.6).

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(a) Ground Truth (b) 2D Segmentation (c) 3D Segmentation

- Fig. 3: Effect of 3D segmentation on localized style transfer. The first column shows the initial 3D object. The second column demonstrates the limitations of using a masked loss similar to previous radiance field-based approaches [19,21,35]. 2D masks can be inconsistent across views and introduce errors, leading to artifacts in different parts of the scene due to incorrect Gaussians being modified. The third column illustrates the benefits of training with a collection of noisy masks to learn a view-consistent feature vector per Gaussian, effectively correcting these errors and avoiding leakage.

#### 4.1 Implementation Details

We evaluate our approach on a range of real-world scenes from diverse datasets, including NeRF [25], MipNerf360 [2], LERF [15], and InstantNGP [26]. These datasets provide various challenging environments to test the effectiveness of our method. Additionally, we use style images from the WikiArt dataset [1], which offers a wide variety of artistic styles, demonstrating the versatility of our style transfer technique.

For all scenes, we begin by running the 3D Gaussian and segmentation pipeline for an initial 30,000 iterations. Following this, we freeze the parameters of all the Gaussians, restricting further optimization to only the spherical harmonic (SH) coefficients of the Gaussians that correspond to the selected object. The style transfer optimization is then performed using 25% of the training images, running for 500 to 1,000 iterations depending on the complexity of the scene. This targeted optimization process is highly efficient, taking less than a minute to complete on a single NVIDIA A100 GPU.

#### 4.2 Object selection

Previous radiance field based approaches [19,21,35] use a 2D masked loss in the image space to localize the style transfer. However, 2D masks can be inconsistent across views and contain errors, leading to incorrect Gaussians being stylized (as demonstrated in Fig. 3). Although this is not a problem in neural scene

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

(a) Ground Truth (b) DEVA Masks (c) Features (d) Object (e) Stylized

- Fig. 4: 3D segmentation results. Figure (a) shows the ground truth image, (b) displays the masks extracted using SAM and DEVA, (c) visualizes the learned feature vectors of all objects in the scene, (d) presents the extracted object, and (e) illustrates the final stylized result.

representations, this manifests as artifacts in the final stylized scene for 3DGS. Using a 3D segmentation approach leads to robustness against 2D mask errors.

The results of our object selection approach are illustrated in Fig. 4. The masks provided by DEVA are shown in Fig. 4b and Fig. 4c visualizes the learnt per-Gaussian feature vectors for the bear and pinecone scenes from the InstantNGP and NeRF datasets respectively. The feature vectors are visualized as the first three principal component analysis (PCA) components of the original 16dimensional vectors. We can observe that the approach provides an effective way to select 3D objects in the scene, confining the style transfer to the selected object.

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

Ground Truth Venetian Canal South Ledges Lizard Story

- Fig. 5: Shows single object style transfer on the bear and pinecone scenes with style images of different artistic styles and composition. Our approach localizes style transfer to the selected objects, without affecting the background.

#### 4.3 Single object style transfer

In this case, we focus on stylizing a single object from the scene. Fig. 5 demonstrates that our method effectively confines the style transfer to the selected object, stylizing it according to the provided artwork. In this figure, we use three style images: Venetian Canal, South Ledges, and Lizard Story. We showcase our results on two object-centric scenes - bear, and pinecone. The results demonstrate precise style transfer. For example, in the bear scene, the geometry and texture (curvature and shadows) are faithfully preserved. The adaptive nature of our style transfer, facilitated by the NNFM loss function, is evident in the pinecone scene, where dark regions adopt a blue hue reminiscent of the Venetian Canal, while bright areas take on an orange tone.

#### 4.4 Multi-object style transfer

Our method extends to stylizing multiple objects within a scene, where we select two distinct objects and apply different style images to each. Fig. 6 illustrates the results for these scenes, showcasing both single-object and multi-object applications of our style transfer method across scenes featuring numerous objects.

The first half of the image shows the counter scene from the MipNerf [2] dataset. We stylize the ‘mitten’ and the ‘flower pot’ in this scene. From three different viewpoints, it is evident that our stylization approach is highly controllable, allowing us to accurately select and stylize objects while preserving their geometry.

[Figure 94]

[Figure 95]

[Figure 96]

Starry Night Venetian Canal Starry + Venetian

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

Ground Truth Mitten Stylized Flower Pot Stylized Both Stylized

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

Ground Truth Apple Stylized Tabletop Stylized Both Stylized

- Fig. 6: Multi-object style transfer. The style images above are applied to selected objects highlighted in the two scenes (counter and figurines). In columns 2 & 3 we show stylizing of individual objects and in column 4 we see both the stylized objects together.

##### In the second half of the figure, we stylize the ‘green apple’ and the entire ‘tabletop’ in the figurines scene from the LERF [15] dataset. Our approach is

[Figure 121]

[Figure 122]

Ground Truth Style Image

[Figure 123]

[Figure 124]

[Figure 125]

Layers: [1, 3] Layers: [6, 8] Layers: [11, 13, 15]

[Figure 126]

[Figure 127]

[Figure 128]

Scale: 0.25 Scale: 0.5 Scale: 0.75

[Figure 129]

[Figure 130]

[Figure 131]

LR: 0.025 LR: 0.05 LR: 0.075

- Fig. 7: Visualisation of scale and intensity control. The scale of the style pattern can be controlled by changing the size of the style image or changing the layers for feature extraction. Intensity can be controlled by varying the learning rate.

effective at stylizing the ‘tabletop’ even though it is not in the foreground and is occluded by various objects. The stylization occurs seamlessly, making the stylized ‘tabletop’ appear as if the style is an inherent property of the surface.

#### 4.5 Scale Control

Previous works [12,21] have shown that the scale of the style patterns can be controlled by two parameters - the receptive field of the VGG features and the size of the style image. The receptive field can be controlled by varying the layers of the VGG network which are used to extract features for the NNFM loss. The size of the style image can be controlled by a downscaling factor (scale = 1 means that the style image is used in its original resolution).

In Fig. 7, we present experimental results exploring these scale parameters and their impact on stylization. We can observe that as the scale of the style image is decreased, the repetition of the pattern increases. A similar effect can be observed with the network layer selection. The features from the deeper layers

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

Fig. 8: Qualitative comparison of our method with S2RF on the flower scene.

have larger receptive fields, giving an effect similar to downscaling the style image. It is interesting to note that if we use layers [1,3] which are somewhat early in the network, the features extracted from those layers for style transfer are not able to accurately learn patterns in the style image.

Additionally, we show that the intensity of style transfer can be controlled by the learning rate, with a larger rate giving stronger style transfer.

#### 4.6 Qualitative Comparison

We evaluate our approach in comparison to S2RF [19], which also focuses on localized stylization of specific objects rather than entire 3D scenes. However, S2RF introduces discoloration in non-target areas, likely due to its use of a gridbased representation which shares parameters across local regions. This parameter sharing can inadvertently affect neighboring regions, which is undesirable for this task. In contrast, our method utilizes segmented 3D Gaussians to precisely isolate style transfer to the target object, leaving other Gaussians unaffected. This distinction is evident in the first two rows in Fig. 8, which illustrates the differences between the two approaches when transferring style on two specific flowers. Our method also performs better in preserving content of the ground truth images. In the last two rows, we observe that in the stylized image generated by S2RF, the shape of the individual flowers is difficult to discern, whereas our approach achieves a more realistic style transfer result.

Additionally, S2RF achieves an average rendering speed of 15 FPS whereas our approach achieves 100+ FPS due to its reliance on the 3DGS representation.

### 5 Limitations

While our method provides efficient and controllable stylization of 3D Gaussian splats, it has certain limitations. Firstly, geometric artifacts arising from the initial 3DGS reconstruction process can occasionally affect the quality of the final stylized scenes. Additionally, the use of the Segment Anything Model for view-specific segmentation can sometimes struggle with generating detailed masks from particular angles, leading to the unintended merging of object parts. Consequently, edits intended for specific areas might inadvertently impact adjacent regions. Although these issues are rare in most scenarios, it can occur in complex scenes or areas with poorly defined boundaries.

### 6 Conclusion

In this work, we introduce StyleSplat, a lightweight technique to stylize 3D objects using a reference style image in the 3D Gaussian Splatting paradigm. StyleSplat leverages off-the-shelf image segmentation and tracking models to obtain consistent 2D masks. These masks are then used as supervision for segmenting 3D Gaussians into distinct objects while jointly optimizing their geometry and color. Finally, a nearest-neighbor feature matching loss is used to finetune the selected Gaussians by aligning their spherical harmonic coefficients with the provided style image. After the initial training and segmentation, StyleSplat takes less than a minute to perform stylization, allowing fast experimentation. We showcase the effectiveness of our method on a variety of scenes and styles, highlighting its suitability for artistic endeavors.

### References

- 1. WikiArt.org - Visual Art Encyclopedia — wikiart.org. https://www.wikiart. org/, [Accessed 30-06-2024]
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. CoRR abs/2111.12077

(2021), https://arxiv.org/abs/2111.12077

- 3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers (2021), https:// arxiv.org/abs/2104.14294
- 4. Castillo, C., De, S., Han, X., Singh, B., Yadav, A.K., Goldstein, T.: Son of zorn’s lemma: Targeted style transfer using instance-aware semantic segmentation (2017), https://arxiv.org/abs/1701.02357
- 5. Chen, Y., Chen, Z., Zhang, C., Wang, F., Yang, X., Wang, Y., Cai, Z., Yang, L., Liu, H., Lin, G.: Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21476–21485 (June 2024)

- 6. Cheng, H.K., Oh, S.W., Price, B., Schwing, A., Lee, J.Y.: Tracking anything with decoupled video segmentation (2023)
- 7. Gatys, L.A., Ecker, A.S., Bethge, M.: A neural algorithm of artistic style (2015), https://arxiv.org/abs/1508.06576
- 8. Gatys, L.A., Ecker, A.S., Bethge, M.: Image style transfer using convolutional neural networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 2414–2423 (2016), https://api.semanticscholar.org/CorpusID: 206593710
- 9. Gu, S., Chen, C., Liao, J., Yuan, L.: Arbitrary style transfer with deep feature reshuffle (2018), https://arxiv.org/abs/1805.04103
- 10. Huang, X., Belongie, S.: Arbitrary style transfer in real-time with adaptive instance normalization (2017), https://arxiv.org/abs/1703.06868
- 11. Jaganathan, V., Huang, H.H., Irshad, M.Z., Jampani, V., Raj, A., Kira, Z.: Ice-g: Image conditional editing of 3d gaussian splats (2024)
- 12. Jing, Y., Liu, Y., Yang, Y., Feng, Z., Yu, Y., Tao, D., Song, M.: Stroke controllable fast style transfer with adaptive receptive fields (2018), https://arxiv.org/abs/ 1802.07101
- 13. Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and super-resolution (2016), https://arxiv.org/abs/1603.08155
- 14. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG) 42, 1 – 14 (2023), https://api.semanticscholar.org/CorpusID:259267917
- 15. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: Lerf: Language embedded radiance fields (2023), https://arxiv.org/abs/2303.09553
- 16. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything

(2023)

- 17. Kolkin, N., Salavon, J., Shakhnarovich, G.: Style transfer by relaxed optimal transport and self-similarity (2019), https://arxiv.org/abs/1904.12785
- 18. Kurzman, L., Vazquez, D., Laradji, I.: Class-based styling: Real-time localized style transfer with semantic segmentation (2019), https://arxiv.org/abs/1908.11525
- 19. Lahiri, D., Panse, N., Kumar, M.: S2rf: Semantically stylized radiance fields (2023)
- 20. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven semantic segmentation (2022), https://arxiv.org/abs/2201.03546
- 21. Li, W., Wu, T., Zhong, F., Oztireli, C.: Arf-plus: Controlling perceptual factors in artistic radiance fields for 3d scene stylization (2023), https://arxiv.org/abs/ 2308.12452
- 22. Liao, J., Yao, Y., Yuan, L., Hua, G., Kang, S.B.: Visual attribute transfer through deep image analogy (2017), https://arxiv.org/abs/1705.01088
- 23. Liu, K., Zhan, F., Xu, M., Theobalt, C., Shao, L., Lu, S.: Stylegaussian: Instant 3d style transfer with gaussian splatting (2024)
- 24. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)
- 25. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis (2020), https://arxiv.org/abs/2003.08934
- 26. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4), 102:1–102:15 (Jul 2022). https://doi.org/10.1145/3528223.3530127, https://doi.org/10. 1145/3528223.3530127

- 27. Psychogios, K., Helen, L., Melissari, F., Bourou, S., Anastasakis, Z., Zahariadis, T.: Samstyler: Enhancing visual creativity with neural style transfer and segment anything model (sam). IEEE Access PP, 1–1 (01 2023). https://doi.org/10. 1109/ACCESS.2023.3315235
- 28. Sara Fridovich-Keil and Alex Yu, Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: CVPR (2022)
- 29. Saroha, A., Gladkova, M., Curreli, C., Yenamandra, T., Cremers, D.: Gaussian splatting in style. arXiv preprint arXiv:2403.08498 (2024)
- 30. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition (2015), https://arxiv.org/abs/1409.1556
- 31. Vachha, C., Haque, A.: Instruct-gs2gs: Editing 3d gaussian splats with instructions

(2024), https://instruct-gs2gs.github.io/

- 32. Wu, X., Hu, Z., Sheng, L., Xu, D.: Styleformer: Real-time arbitrary style transfer via parametric style composition. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 14598–14607 (2021). https://doi.org/10.1109/ ICCV48922.2021.01435
- 33. Ye, M., Danelljan, M., Yu, F., Ke, L.: Gaussian grouping: Segment and edit anything in 3d scenes (2023)
- 34. Yifan, W., Serena, F., Wu, S., Öztireli, C., Sorkine-Hornung, O.: Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics 38(6), 1–14 (Nov 2019). https://doi.org/10.1145/3355089.3356513, http://dx.doi.org/10.1145/3355089.3356513
- 35. Zhang, D., Fernandez-Labrador, C., Schroers, C.: Coarf: Controllable 3d artistic style transfer for radiance fields (2024), https://arxiv.org/abs/2404.14967
- 36. Zhang, D., Chen, Z., Yuan, Y.J., Zhang, F.L., He, Z., Shan, S., Gao, L.: Stylizedgs: Controllable stylization for 3d gaussian splatting. arXiv preprint arXiv:2404.05220

(2024)

- 37. Zhang, K., Kolkin, N., Bi, S., Luan, F., Xu, Z., Shechtman, E., Snavely, N.: Arf: Artistic radiance fields (2022)
- 38. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21676–21685 (June 2024)
- 39. Zhuang, J., Kang, D., Cao, Y.P., Li, G., Lin, L., Shan, Y.: Tip-editor: An accurate 3d editor following both text-prompts and image-prompts (2024), https://arxiv. org/abs/2401.14828
- 40. Zwicker, M., Pfister, H., Baar, J., Gross, M.: Surface splatting. Proceedings of the ACM SIGGRAPH Conference on Computer Graphics 2001 (08 2001). https: //doi.org/10.1145/383259.383300

