# arXiv:2307.02421v2[cs.CV]20Nov2023

## DRAGONDIFFUSION: ENABLING DRAG-STYLE MANIPULATION ON DIFFUSION MODELS

#### Chong Mou1 Xintao Wang2 Jiechong Song1 Ying Shan2 Jian Zhang†1

1School of Electronic and Computer Engineering, Shenzhen Graduate School, Peking University 2ARC Lab, Tencent PCG

https://mc-e.github.io/project/DragonDiffusion/

ABSTRACT

Despite the ability of existing large-scale text-to-image (T2I) diffusion models to generate high-quality images from detailed textual descriptions, they often lack the ability to precisely edit the generated or real images. In this paper, we propose a novel image editing method, DragonDiffusion, enabling Drag-style manipulation on Diffusion models. Specifically, we treat image editing as the change of feature correspondence in a pre-trained diffusion model. By leveraging feature correspondence, we develop energy functions that align with the editing target, transforming image editing operations into gradient guidance. Based on this guidance approach, we also construct multi-scale guidance that considers both semantic and geometric alignment. Furthermore, we incorporate a visual cross-attention strategy based on a memory bank design to ensure consistency between the edited result and original image. Benefiting from these efficient designs, all content editing and consistency operations come from the feature correspondence without extra model fine-tuning or additional modules. Extensive experiments demonstrate that our method has promising performance on various image editing tasks, including editing within a single image (e.g., object moving, resizing, and content dragging) and across images (e.g., appearance replacing and object pasting).

1 INTRODUCTION

Thanks to the large-scale training data and huge computing power, generative models have developed rapidly, especially large-scale text-to-image (T2I) diffusion models Saharia et al. (2022); Rombach et al. (2022); Nichol et al. (2022); Ramesh et al. (2022), which aims to generate images conditioned on a given text/prompt. However, this generative capability is usually diverse, and it is challenging to design suitable prompts to generate images consistent with what the user has in mind, let alone fine-grained image editing based on the text condition.

In the community of image editing, previous methods are usually designed based on GANs Abdal et al. (2019; 2020); Alaluf et al. (2022) due to the compact and editable latent space, e.g., the W space in StyleGAN Karras et al. (2019b). Recently, DragGAN Pan et al. (2023) proposes a pointto-point dragging scheme, which can achieve refined content dragging. However, it is limited by the capacity and generalization of GANs. Compared to GANs, diffusion model Ho et al. (2020) has higher stability and superior generation quality. Due to the lack of a concise and editable latent space, numerous diffusion-based image editing methods Hertz et al. (2022); Feng et al. (2022); Balaji et al. (2022) are built based on T2I diffusion models via correspondence between text and image features. Recently, self-guidance Epstein et al. (2023) proposes a differentiable approach that employs crossattention maps between text and image to locate and calculate the size of objects within images. Then, gradient guidance is utilized to edit these properties. However, the correspondence between text and image features is weak, heavily relying on the design of prompts. Moreover, in complex or multi-object scenarios, text struggles to build accurate correspondence with a specific object. In this paper, we aim to investigate whether the diffusion model can achieve drag-style image editing, which is a fine-grained and generalized editing ability not limited to point dragging.

In the large-scale T2I diffusion model, besides the correspondence between text features and intermediate image features, there is also a strong correspondence across image features. This characteristic is studied in DIFT Tang et al. (2023), which demonstrates that this correspondence is high-level,

enabling point-to-point correspondence of relevant image content. Therefore, we are intrigued by the possibility of utilizing this strong correspondence across image features to achieve image editing. In this paper, we regard image editing as the change of feature correspondence and convert it into gradient guidance via energy functions Dhariwal & Nichol (2021) in score-based diffusion Song

- et al. (2020b). Additionally, the content consistency between editing results and original images is also ensured by feature correspondence in a visual cross-attention design. Here, we notice that there is a concurrent work, DragDiffusion Shi et al. (2023), studying this issue. It uses LORA Ryu (2023) to maintain consistency with the original image and optimizes the latent in a specific diffusion step to perform point dragging. Unlike DragDiffusion, our image editing is achieved by energy functions and a visual cross-attention design, without the need for extra model fine-tuning or new blocks. In addition, we can complete various drag-style image editing tasks beyond the point dragging. In summary, the contributions of this paper are as follows:

- • We achieve drag-style image editing via gradient guidance produced by image feature correspondence in the pre-trained diffusion model. In this design, we also study the roles of the features in different layers and develop multi-scale guidance that considers both semantic and geometric correspondence.
- • We design a memory bank, further utilizing the image feature correspondence to maintain the consistency between editing results and original images. In conjunction with gradient guidance, our method allows a direct transfer of T2I generation ability in diffusion models to image editing tasks without the need for extra model fine-tuning or new blocks.
- • Extensive experiments demonstrate that our method has promising performance in various image editing tasks, including editing within a single image (e.g., object moving, resizing, and content dragging) or across images (e.g., appearance replacing and object pasting).

- 2 RELATED WORK

- 2.1 DIFFUSION MODELS

Recently, diffusion models Ho et al. (2020) have achieved great success in the community of image synthesis. It is designed based on thermodynamics Sohl-Dickstein et al. (2015); Song & Ermon (2019), including a diffusion process and a reverse process. In the diffusion process, a natural image x0 is converted to a Gaussian distribution xT by adding random Gaussian noise with T iterations. The reverse process is to recover x0 from xT by several denoising steps. Therefore, the diffusion model is to train a denoiser, conditioned on the current noisy image xt and time step t:

Ex

0,t,ϵt∼N(0,1) ||ϵt − ϵθ(xt,t)||22 , (1) where ϵθ is the function of the denoiser. Recently, some text-conditioned diffusion models (e.g., GLID Nichol et al. (2022) and StableDiffusion(SD) Rombach et al. (2022)) are proposed, which mostly inject text condition into the denoiser through a cross-attention strategy. From the continuous perspective Song et al. (2020b), diffusion models can be viewed as a score function (i.e., ϵθ(xt,t) ≈ ∇xt

log q(xt)) that samples from the corresponding distribution Song & Ermon (2020) according to Langevin dynamics Sohl-Dickstein et al. (2015); Song & Ermon (2019).

- 2.2 ENERGY FUNCTION IN DIFFUSION MODEL

From the continuous perspective of score-based diffusion, the external condition y can be combined by a conditional score function, i.e., ∇xt

log q(xt|y), to sample from a more enriched distribution. The conditional score function can be further decomposed as:

q(xt|y)q(xt) q(y) ∝ ∇xt log q(xt) + ∇xt log q(y|xt), (2)

∇xt log q(xt|y) = log

where the first term is the unconditional denoiser, and the second term refers to the conditional gradient produced by an energy function E(xt;t,y) = q(xt|y). E can be selected based on the generation target, such as a classifier Dhariwal & Nichol (2021) to specify the category of generation results. Energy function has been used in various controllable generation tasks, e.g., sketch-guided generation Voynov et al. (2022), mask-guided generation Singh et al. (2023), universal guidance Yu et al. (2023); Bansal et al. (2023), and image editing Epstein et al. (2023). These methods, inspire us to transform editing operations into conditional gradients, achieving fine-grained image editing.

- 2.3 IMAGE EDITING

In image editing, numerous previous methods Abdal et al. (2019; 2020); Alaluf et al. (2022) invert images into the latent space of StyleGAN Karras et al. (2019b) and then edit the image by manipulating latent vectors. Motivated by the success of diffusion model Ho et al. (2020), various diffusionbased image editing methods Avrahami et al. (2022); Hertz et al. (2022); Kawar et al. (2023); Meng

- et al. (2021); Brooks et al. (2023) are proposed. Most of them use text as the editing control. For example, Kawar et al. (2023); Valevski et al. (2023); Kwon & Ye (2022) perform model fine-tuning on a single image and then generate the editing result by target text. Prompt2Prompt Hertz et al.

- (2022) achieves specific object editing by exchanging text-image attention maps. SDEdit Meng et al.

(2021) performs image editing by adding noise to the original image and then denoising under new text conditions. InstructPix2Pix Brooks et al. (2023) retrain the diffusion model with text as the editing instruction. Recently, Self-guidance Epstein et al. (2023) transforms image editing operations into gradients through the correspondence between text and image features. However, text-guided image editing is coarse. Recently, DragGAN Pan et al. (2023) proposes a point-to-point dragging scheme, which can achieve fine-grained dragging editing. Nevertheless, its editing quality and generalization are limited by GANs. How to utilize the high-quality and diverse generation ability of diffusion models for fine-grained image editing is still an open challenge.

- 3 METHOD

- 3.1 PRELIMINARY: HOW TO CONSTRUCT ENERGY FUNCTION IN DIFFUSION

Modeling an energy function E(xt;t,y) to produce the conditional gradient ∇xt

log q(y|xt) in Eq. 2, remains an open question. E measures the distance between xt and the condition y. Some methods Dhariwal & Nichol (2021); Voynov et al. (2022); Zhao et al. (2022) train a time-dependent distance measuring function, e.g., a classifier Dhariwal & Nichol (2021) to predict the probability that xt belongs to category y. However, the training cost and annotation difficulty are intractable in our image editing task. Some tuning-free methods Yu et al. (2023); Bansal et al. (2023) propose using the clean image x0|t predicted at each time step t to replace xt for distance measuring, i.e., E(xt;t,y) ≈ D(x0|t;t,y). Nevertheless, there is a bias between x0|t and x0, and there is hardly a suitable D for distance measuring in image editing tasks. Hence, the primary issue is whether we can circumvent the training requirement and construct an energy function to measure the distance between xt and the editing target. Recent work Tang et al. (2023) has shown that the feature correspondence in the diffusion UNet-denoiser ϵθ is high-level, enabling point-to-point correspondence measuring. Inspired by this characteristic, we propose reusing ϵθ as a tuning-free energy function to transform image editing operations into the change of feature correspondence.

- 3.2 OVERVIEW

The editing objective of our DragonDiffusion involves two issues: changing the content to be edited and preserving unedited content. For example, if a user wants to move the cup in an image, the generated result only needs to change the position of the cup, while the appearance of the cup and other unedited content should not change. An overview of our method is presented in Fig. 1, which is built on the pre-trained SD Rombach et al. (2022) to support image editing with and without reference images. First, we use DDIM inversion Song et al. (2020a) to transform the original image into zT. If the reference image zref0 exists, it will also be involved in the inversion. In this process, we store some intermediate features and latent at each time step to build a memory bank, which is used to provide guidance for subsequent image editing. In generation, we transform the information stored in the memory bank into content editing and consistency guidance through two paths, i.e., visual cross-attention and gradient guidance. Both of these paths are built based on feature correspondence in the pre-trained SD. Therefore, our image editing pipeline is efficiently built without extra model fine-tuning or new blocks.

- 3.3 DDIM INVERSION WITH MEMORY BANK

In our image editing process, the starting point zT, produced by DDIM inversion Song et al. (2020a), can provide a good generation prior to maintain consistency with the original image. However,

gud

gud

ref

ref

gud ref

gud gud

gud

Appearance Editing

1

Reference Image

T

1

T

ref

[Figure 1]

[Figure 2]

### ...

f

rf

gud

gud

ref

ref

t t

T T

1

[ tgud, tref]

T

[ Tgud, Tref] tgud

1

T

gud

###### ...

Feature Correspondence

Feature Correspondence

T

T, T

t, t

Memory Bank

Guidance

Guidance

Gradient

Gradient

Moving Editing

Original Image

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

Encoder

[Figure 13]

[Figure 14]

Decoder

DDIM Inversion

...

Selfattention

U-Net Denoiser gen

gen T

gen

ref

0

0

0

t ×( −1)

[Figure 15]

Figure 1: Overview of our DragonDiffusion, which consists of two parts: (1)DDIM inversion Song et al. (2020a) to build a memory bank; (2)inference with guidance from the memory bank. Our method is built on the pre-trained SD Rombach et al. (2022) without extra training or modules.

relying solely on the final step of this approximate inversion can hardly provide accurate generation guidance. Therefore, we fully utilize the information in DDIM inversion by building a memory bank

to store the latent zgudt at each inversion step t, as well as corresponding keys Kgudt and values Vtgud in the self-attention module of the decoder within the UNet denoiser. Note that in some cross-image editing tasks (e.g., appearance replacing, object pasting), reference images are required. In these tasks, the memory bank needs to be doubled to store the information of the reference images. Here,

we utilize zreft , Kreft , and Vtref to represent them. The information stored in the memory bank will provide more accurate guidance for the subsequent image editing process.

0

[Figure 16]

 ’0

- 3.4 GRADIENT-GUIDANCE-BASED EDITING DESIGN

[Figure 17]

[Figure 18]

Inspired by classifier guidance Dhariwal & Nichol (2021), we build energy functions to transform image editing operations into gradient guidance in diffusion sampling. An intuitive illustration is presented in Fig. 2, showing a continuous sampling space of the score-based diffusion Song et al. (2020b). The sampling starting point zT, obtained from DDIM inversion, will approximately return to the original point only according to the gradient/score predicted by the denoiser. After incorporating the gradient guidance generated by the energy function that matches the editing target, the additional guidance gradient will change the path to reach a sampling result that meets the editing target.

Guidance

Continuous Sampling Space

T

: Original Gradient : Corrected Gradient

Figure 2: Illustration of continuous sampling space in score-based diffusion. Bright colors indicate areas where target data is densely distributed. The orange and green paths respectively refer to the diffusion paths without and with external gradient guidance.

- 3.4.1 ENERGY FUNCTION VIA FEATURE CORRESPONDENCE

In our DragonDiffusion, energy functions are designed to provide gradient guidance for image editing, mainly including content editing and consistency terms. Specifically, at the t-th time step, we reuse the UNet denoiser ϵθ to extract intermediate features Fgent from the latent zgent at the current time step. The same operation is used to extract guided features Fgudt from zgudt in memory bank. Following DIFT Tang et al. (2023), Fgent and Fgudt come from intermediate features in the UNet decoder. The image editing operation is represented by two binary masks (i.e., mgud and mgen) to locate the original content position and target dragging position, respectively. Therefore, the energy function is built by constraining the correspondence between these two regions in Fgudt and Fgent . Here, we utilize cosine distance cos(·) ∈ [−1,1] to measure the similarity and normalize it to [0,1]:

Slocal(Fgent ,mgen,Fgudt ,mgud) = 0.5 · cos Fgent [mgen], sg(Fgudt [mgud]) + 0.5, (3)

where sg(·) is the gradient clipping operation. Eq. 3 is mainly used for dense constraints on the spatial location of content. In addition, a global appearance similarity is defined as:

Fgudt [mgud] mgud

Fgent [mgen] mgen

Sglobal(Fgent ,mgen,Fgudt ,mgud) = 0.5·cos

, sg(

) +0.5, (4)

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Original Image Guided by the layer-1

Guided by the layer-2

Guided by the layer-2&3

Guided by the layer-3

Guided by the layer-4

Figure 3: Illustration of using features from different layers as guidance to restore the original image. zT is randomly initialized. The generation is solely guided by content consistency guidance in Eq. 6.

which utilizes the mean of the features in a region as a global appearance representation. When we want to have fine control over the spatial position of an object or a rough global control over its appearance, we only need to constrain the similarity in Eq. 3 and Eq. 4 to be as large as possible. Therefore, the energy function to produce editing guidance is defined as:

1 α + β · S(Fgent ,mgen,Fgudt ,mgud)

, S ∈ {Slocal, Sglobal}, (5)

Eedit =

where α and β are two hyper-parameters, which are set as 1 and 4, respectively. In addition to editing, we hope the unedited content remains consistent with the original image. We use a mask mshare to locate areas without editing. The similarity between the editing result and the original image

in mshare can also be calculated by the cosine similarity as Slocal(Fgent ,mshare,Fgudt ,mshare). Therefore, the energy function to produce content consistency guidance is defined as:

1 α + β · Slocal(Fgent ,mshare,Fgudt ,mshare)

. (6)

Econtent =

In addition to Eedit and Econtent, an optional guidance term Eopt may need to be added in some tasks to achieve the editing goal. Finally, the base energy function is defined as:

E = we · Eedit + wc · Econtent + wo · Eopt, (7)

where we, wc, and wo are hyper-parameters to balance these guidance terms. They vary slightly in different editing tasks but are fixed within the same task. Finally, regarding [mgen,mshare] as condition, the conditional score function in Eq. 2 can be written as:

log q(y|zgent ), y = [mgen,mshare]. (8) The conditional gradient ∇z

log q(zgent |y) ∝ ∇z

log q(zgent ) + ∇z

∇z

gen t

gen t

gen t

log q(y|zgent ) can be computed by ∇z

E, which will also multiplies by a learning rate η. In experiments, we find that the gradient guidance in later diffusion generation steps hinders the generation of textures. Therefore, we only add gradient guidance in the first n steps of diffusion generation. Experientially, we set n = 30 in 50 sampling steps.

gen t

gen t

- 3.4.2 MULTI-SCALE FEATURE CORRESPONDANCE

The decoder of the UNet denoiser contains four blocks of different scales. DIFT Tang et al. (2023) finds that the second layer contains more semantic information, while the third layer contains more geometric information. We also studied the role of features from different layers in image editing tasks, as shown in Fig. 3. In the experiment, we set zT as random Gaussian noise and set mgen, mgud as zeros matrixes. mshare is set as a ones matrix. In this way, generation relies solely on content consistency guidance (i.e., Eq. 6) to restore image content. We can find that the guidance from the first layer is too high-level to reconstruct the original image accurately. The guidance from the fourth layer has weak feature correspondence, resulting in significant differences between the reconstructed and original images. The features from the second and third layers are more suitable to produce guidance signals, and each has its own specialty. Concretely, the features in the second layer contain more semantic information and can reconstruct images that are semantically similar to the original image but with some differences in content details. The features in the third layer tend to express low-level characteristics, but they cannot provide effective supervision for high-level texture, resulting in blurry results. In our design, we combine these two levels (i.e., high and low) of guidance by proposing a multi-scale supervision approach. Specifically, we compute gradient guidance on the second and third layers. The reconstructed results in Fig. 3 also demonstrate that this combination can balance the generation of low-level and high-level visual characteristics.

##### 3.4.3 IMPLEMENTATION DETAILS FOR EACH APPLICATION

Object moving. In the task of object moving, mgen and mgud locate the same object in different spatial positions. mshare is the complement (Cu) of the union (∪) of mgen and mgud, i.e., mshare = Cu(mgen∪mgud). However, solely using the content editing and consistency guidance in Eq. 5 and Eq. 6 can lead to some issues, as shown in the second image of Fig. 4. Concretely, although the bread is moved according to the editing signal, some of the bread content is still preserved in its original position in the generated result. This is because the energy function does not constrain the area where the moved object was initially located, causing inpainting to easily restore the original object. To rectify this issue, we use the optional energy term (i.e., Eopt in Eq. 7) to constrain the inpainting content to be dissimilar to the moved object and similar to a predefined reference region. Here, we use mref to locate the reference region and define mipt = {p|p ∈ mgud and p ∈/ mgen} to locate the inpainting region. Finally, Eopt in this task is defined as:

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

 ℎ   

Original Image Moving w/o Moving w

Figure 4: Visualization of the effectiveness of inpainting guidance (Eopt) in the object moving task, presenting that Eopt can guide the inpainting of the area where the object is initially located.

wi α + β · Sglobal(Fgent , mipt, Fgudt , mref)

+ Slocal(Fgent ,mipt,Fgudt ,mipt), (9)

Eopt =

where wi is a weight parameter, set as 2.5 in our implementation. The third image in Fig. 4 shows that this design can effectively achieve the editing goal without impeachable artifact.

Object resizing. The score function in this task is the same as the object moving, except that a scale factor γ > 0 is added during feature extraction. Specifically, we use interpolation to transform

mgud and Fgudt to the target size, and then extract Fgudt [mgud] as the feature of the resized object. To locate the target object with the same size in Fgent , we resize mgen with the same scale factor γ. Then we extract a new mgen of the original size from the center of the resized mgen. Note that if γ < 1, we use 0 to pad the vacant area.

[Figure 31]

Appearance replacing. This task aims to replace the appearance between objects of the same category across images. Therefore, the capacity of the memory bank needs to be doubled to store extra information from the image containing the reference appearance, i.e., zreft , Kreft , and Vtref. mgen and mgud respectively locate the editing object in the original image and the reference object in the reference image. mshare is set as the complement of mgen, i.e., Cu(mgen). To constrain appearance, we choose Sglobal(Fgent ,mgen,Freft ,mgud)

[Figure 32]

[Figure 33]

[Figure 34]

Reference Image Original Image

[Figure 35]

[Figure 36]

- in Eq. 5. In this task, there is no need for Eopt.

Object pasting. Object pasting aims to paste an object from an image onto any position in another image. Although it can be completed by simple copy-paste, it often results in inconsistencies between the paste area and other areas due to differences in light and perspective, as shown in Fig. 5. As can be seen, the result obtained by copy-paste exists discontinuities, while the result gener-

Copy-paste DragonDiffusion

Figure 5: Visual comparison between our DragonDiffusion and direct copypaste in cross-image object pasting.

ated by our DragonDiffusion can achieve a more harmonized integration of the scene and the pasted object. In implementation, similar to the appearance replacing, the memory bank needs to store information of the reference image, which contains the target object. mgen and mgud respectively mark the position of the object in the edited image and reference image. mshare is set as Cu(mgen). Point dragging. In this task, we want to drag image content via several points, like DragGAN Pan et al. (2023). In this case, mgen and mgud locate neighboring areas centered around the destination and starting points. Here, we extract a 3 × 3 rectangular patch centered around each point as the neighboring area. Unlike the previous tasks, mshare here is manually defined.

ObjectMoving&ResizingAppearanceReplacing

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Moving ObjectPasting

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

ContentDragging

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

Continuous

Continuous

Dragging

- Figure 6: Visualization of our method on different image editing applications, including object moving, object resizing, object appearance replacing, object pasting, and content dragging.

- 3.5 VISUAL CROSS-ATTENTION

As mentioned previously, two strategies are used to ensure the consistency between the editing result and the original image: (1) DDIM inversion to initialize zT; (2) content consistency guidance

- in Eq. 6. However, it is still challenging to maintain high consistency. Inspired by the consistency preserving in some video and image editing works Wu et al. (2022); Cao et al. (2023); Wang et al.

- (2023), we design a visual cross-attention guidance. Instead of generating guidance information through an independent inference branch, we reuse the intermediate features of the inversion process stored in the memory bank. Specifically, similar to the injection of text conditions in SD Rombach

- et al. (2022), we replace the key and value in the self-attention module of the UNet decoder with the corresponding key and value collected by the memory bank in DDIM inversion. Note that in the appearance replacing and object pasting tasks, the memory bank stores two sets of keys and

values from the original image (Kgudt ,Vtgud) and the reference image (Kreft ,Vtref). In this case, we concatenate the two sets of keys and values in the length dimension. The visual cross-attention at each time step is defined as:

Qt = Qgent ; Kt = Kgudt or (Kgudt ⃝c Kreft ); Vt = Vtgud or (Vtgud⃝c Vtref) Att(Qt, Kt, Vt) = softmax(QtK

T √ t

d )Vt,

where ⃝c refers to the concatenation operation.

- 4 EXPERIMENTS

(10)

In experiments, we use StableDiffusion-V1.5 Rombach et al. (2022) as the base model. The inference adopts DDIM sampling with 50 steps, and we set the classifier-free guidance scale as 5.

- 4.1 APPLICATIONS

In this paper, our proposed DragonDiffusion can perform various image editing tasks without specific training. These applications include object moving, resizing, appearance replacing, object pasting, and content dragging. In Fig. 6, we present our editing performance on each application. The object moving and resizing in the first block show that our method can naturally move and resize objects in images with good content consistency. The moved objects can blend well with the surrounding content. The second block shows that our method can paste an object from an image into another image and slightly adjust the appearance to blend in with new scenarios. In the third block, we present the performance of object appearance replacing. It shows that our method can replace the appearance with that of a same-category object from a reference image while preserving the original outline. The fourth block shows that our method can drag the content within the image

Table 1: Quantitative evaluation on face manipulation with 68 and 17 points. The accuracy is calculated by Euclidean distance between edited points and target points. The initial distance (i.e., 57.19 and 36.36) is the upper bound, without editing. FID Seitzer (2020) is utilized to quantize the editing quality of different methods. The time complexity is computed on the ‘1 point’ dragging.

| |Preparing complexity↓<br><br>Inference complexity↓<br><br>Unaligned face<br><br>17 Points↓ From 57.19<br><br>68 Points↓ From 36.36<br><br>FID↓ 17/68 points|
|---|---|
|UserControllableLT DragGAN DragDiffusion DragonDiffusion(ours)<br><br>|1.2s 0.05s 32.32 24.15 51.20/50.32 52.40s 6.71s 15.96 10.60 39.27/39.50 48.25s 19.71s 22.95 17.32 38.06/36.55<br><br>3.62s 15.93s 18.51 13.94 35.75/34.58<br><br>|

Original Face Reference Face UserControllableLT DragGAN DragDiffusion Ours

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

68Points17Points

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

- Figure 7: Qualitative comparison between our DragonDiffusion and other methods in face manipulation. The current and target points are labeled with red and blue. The white line indicates distance.

with several points. The dragging results are reasonable with the editing direction, while the result remains consistent with the original image. The last row in Fig. 6 presents the stable performance in continuous editing. More results are presented in the appendix.

- 4.2 COMPARISONS

In this part, we compare our method with the recent UserControllableLT Endo (2022), DragGAN Pan et al. (2023), and DragDiffusion Shi et al. (2023) in the keypoint-based face manipulation.

Time complexity. We divide the time complexity of different methods into two parts, i.e., the preparing and inference stages. The preparing stage involves Diffusion/GAN inversion and model fine-tuning. The inference stage generates the editing result. The time complexity for each method is tested on one point dragging, with the image resolution being 512 × 512. The experiment is conducted on an NVIDIA A100 GPU with Float32 precision. Tab. 1 presents that our method is relatively efficient in the preparing stage, requiring only 3.62s to prepare zT and build a memory bank. The inference complexity is also acceptable for diffusion generation.

Qualitative and quantitative evaluation. Following DragGAN Pan et al. (2023), the comparison is conducted on the face keypoint manipulation with 17 and 68 points. The test set is randomly formed by 800 aligned faces from CelebA-HQ Karras et al. (2017) training set. Note that we do not set fixed regions for all methods, due to the difficulty in manually providing a mask for each face. In addition to accuracy, we also compute the FID Seitzer (2020) between face editing results and CelebA-HQ training set to represent the editing quality. The quantitative and qualitative comparison is presented in Tab. 1 and Fig. 7, respectively. We can find that although DragGAN can produce more accurate editing results, it has limitations in content consistency and robustness in areas outside faces (e.g., the headwear is distorted). The limitations of GAN models also result in DragGAN and UserControllableLT requiring face alignment before editing. In comparison, our method has promising editing accuracy, and the generation prior from SD enables better robustness and generalization for different content. In this task, our method also has better performance than DragDiffusion. Moreover, the visual cross-attention design makes our method achieve attractive content consistency without extra model fine-tuning or modules. More results are shown in the appendix.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Moving w/o Content Consistency guidance

Original Image Moving w/o Inversion Prior

Moving w/o Visual Cross-attention

Full Implementation

- Figure 9: The ablation study of different components in our DragonDiffusion. The experiment is conducted on the task of object moving.

Further discussion on the generalization of DragGAN and our method. Although DragGAN demonstrates powerful drag editing capabilities, its performance is significantly reduced in complex scenarios due to the limited capability of the GANs, as shown in Fig. 8. Specifically, we use the StyleGAN trained on the human bodies Fu et al. (2022) and FFHQ Karras et al. (2019a) to perform body and face editing by DragGAN. As can be seen, the editing quality of DragGAN is sensitive to whether the image is aligned. The alignment operation will filter out the background of the body or change the face pose, which usually does not meet our editing requirements.

In comparison, our DragonDiffusion inherits good generalization of the pre-trained SD and can handle complex and unaligned scenarios effectively. The resolution of images processed by our method is also arbitrary, unlike the fixed size in GANs.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- 4.3 ABLATION STUDY

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

In this part, we demonstrate the effectiveness of some components in our DragonDiffusion, as shown

- in Fig. 9. We conduct the experiment on the object moving task. Specifically, (1) we verify the importance of the inversion prior by randomly initializing zT instead of obtaining from DDIM inversion. As can be seen, the random zT leads to a significant difference between the editing result and the original image. (2) We remove the content consistency

Original Image DragGAN w/o alignment

DragGAN w alignment

Ours w/o alignment

Figure 8: Editing comparison between our DragonDiffusion and DragGAN Pan et al. (2023) on the unaligned body and face.

guidance (i.e., Econtent) in Eq. 7, which causes local distortion in the editing result, e.g., the finger is twisted. (3) We remove the visual cross-attention. It can be seen that visual cross-attention plays an important role in maintaining the consistency between the edited object and the original object. The last image shows the satisfactory editing performance of our method with full implementation. Therefore, these components work together on both edited and unedited content, forming the finegrained image editing model DragonDiffusion, which does not require extra training or modules.

- 5 CONCLUSION

Despite the ability of existing large-scale text-to-image (T2I) diffusion models to generate highquality images from detailed textual descriptions, they often lack the ability to precisely edit the generated or real images. In this paper, we aim to develop a drag-style and general image editing scheme based on the strong correspondence of intermediate image features in the pre-trained diffusion model. To this end, we model image editing as the change of feature correspondence and design energy functions to transform the editing operations into gradient guidance. Based on the gradient guidance strategy, we also propose multi-scale guidance to consider both semantic and geometric alignment. Moreover, a visual cross-attention is added based on a memory bank design, which can enhance the consistency between the original image and the editing result. Due to the reuse of intermediate information from the inversion process, this content consistency strategy almost has no additional cost. Extensive experiments demonstrate that our proposed DragonDiffusion can perform various image editing tasks, including object moving, resizing, appearance replacing, object pasting, and content dragging. At the same time, the complexity of our DragonDiffusion is acceptable, and it does not require extra model fine-tuning or additional modules.

REFERENCES

Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4432–4441, 2019.

Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan++: How to edit the embedded images? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8296–8305, 2020.

Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In Proceedings of the IEEE/CVF conference on computer Vision and pattern recognition, pp. 18511–18521, 2022.

Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18208–18218, 2022.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 843–852, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Yuki Endo. User-controllable latent transformer for stylegan image layout editing. In Computer Graphics Forum, volume 41, pp. 395–406. Wiley Online Library, 2022.

Dave Epstein, Allan Jabri, Ben Poole, Alexei A Efros, and Aleksander Holynski. Diffusion selfguidance for controllable image generation. arXiv preprint arXiv:2306.00986, 2023.

Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032, 2022.

Jianglin Fu, Shikai Li, Yuming Jiang, Kwan-Yee Lin, Chen Qian, Chen Change Loy, Wayne Wu, and Ziwei Liu. Stylegan-human: A data-centric odyssey of human generation. In European Conference on Computer Vision, pp. 1–19. Springer, 2022.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern

- recognition, pp. 4401–4410, 2019a.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern

- recognition, pp. 4401–4410, 2019b.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6007–6017, 2023.

Gihyun Kwon and Jong Chul Ye. Diffusion-based image translation using disentangled style and content representation. arXiv preprint arXiv:2209.15264, 2022.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6038–6047, 2023.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pp. 16784–16804. PMLR, 2022.

Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. arXiv preprint arXiv:2305.10973, 2023.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.

Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/ pytorch-fid, August 2020. Version 0.3.0.

Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023.

Jaskirat Singh, Stephen Gould, and Liang Zheng. High-fidelity guided image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5997–6006, 2023.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. Advances in neural information processing systems, 33:12438–12448, 2020.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881, 2023.

Dani Valevski, Matan Kalman, Eyal Molad, Eyal Segalis, Yossi Matias, and Yaniv Leviathan. Unitune: Text-driven image editing by fine tuning a diffusion model on a single image. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.

Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. arXiv preprint arXiv:2211.13752, 2022.

Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zeroshot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.

Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. arXiv preprint arXiv:2303.09833, 2023.

Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022.

A APPENDIX

- A.1 ALGORITHM PIPELINE OF DRAGONDIFFUSION

To facilitate the understanding of our DragonDiffusion, we present the entire algorithm pipeline in Algorithm 1. Note that the text condition c has a minimal impact on the final result in our method, and only a brief description of the image is needed.

Algorithm 1: Proposed DragonDiffusion Require: text condition c; UNet denoiser ϵθ; pre-defined parameter α¯t; image to be edited x0; the mask

of mgen, mgud, and mshare; the learning rate η; the number of gradient-guidance steps n. Initialization:

- (1) Compute latent z0 of the image to be edited: z0 = Encoder(x0)
- (2) Select an editing task Ts ∈ [‘resizing&moving’, ‘dragging’, ‘pasting’, ‘replacing’];
- (3) Compute latent zref0 of the reference image xref0 : if Ts ∈ [‘resizing&moving’, ‘dragging’] then zref0 = ∅ else zref0 = Encoder(xref0 )
- (4) Compute the inversion prior zgenT and build the memory bank:

zgenT , Bank = DDIMInversion(z0, zref0 ) for t = T, ..., 1 do

if Ts ∈ [‘resizing&moving’, ‘dragging’] then Kgudt , Vtgud, zgudt = Bank[t]; Kreft , Vtref, zreft = ∅; extract Fgent and Fgudt from zgent and zgudt by ϵθ; Kt, Vt = Kgudt , Vtgud;

else

Kgudt , Vtgud, zgudt , Kreft , Vtref, zreft = Bank[t]; extract Fgent , Fgudt and Freft from zgent , zgudt and zreft by ϵθ; Kt, Vt = Kgudt ⃝c Kreft , Vtgud⃝c Vtref;

end ϵˆt = ϵθ(zgent , Kt, Vt, t, c); if T − t < n then

E = we · Eedit + wc · Econtent + wo · Eopt; ϵˆt = ϵˆt + η · ∇ztE;

zt−1 = √α¯t−1(z

t−

√√1α¯−tα¯tϵˆt + √1 − α¯t−1ϵˆt); end

x0 = Decoder(z0); Output: x0

- A.2 EFFICIENCY OF THE MEMORY BANK DESIGN

gud

gud

×  0

×  0

Null-text

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Memory UNet Bank

[Figure 109]

[Figure 110]

UNet Denoiser

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Denoiser

Null-text Inversion

Guidance

DDIM Inversion

Guidance

gen

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

###### Guidance

[Figure 125]

[Figure 126]

gen

0 T

[Figure 127]

[Figure 128]

DDIM Inversion

UNet Denoiser

0 T

UNet Denoiser

gen

gen

× 

UNet Denoiser

gen

× 

gen

0

0 T

× 

[Figure 129]

0

[Figure 130]

0

[Figure 131]

Null-text

###### (a) DDIM Inversion (b) DDIM Inversion w Null-text Optimization (c) DDIM Inversion w Memory Bank

- Figure 10: Different strategies for generating inversion prior (i.e., zT) and guidance information

(i.e., Kgudt ,Vtgud). (a) DDIM inversion + separate branch; (b) null-text inversion Mokady et al.

(2023) + separate branch; (c) our memory bank design.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

DraggingObjectMoving&ResizingAppearanceReplacingObjectPasting

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

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Figure 11: More results of DragonDiffusion on different applications.

In this paper, we designed a memory bank to store intermediate information during the inversion process, which is used to provide guidance for image editing. To verify its effectiveness, we compared it with methods having the same function, as shown

Times: 3.62s Times: 180s Times: 3.62s

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

- in Fig. 10. Specifically, (a) guidance information is

generated by a separate generation branch from zT; (b) null-text optimization is added based on method (a); (c) using our designed memory bank strategy. The editing quality of different methods is presented

Original Image DDIM Inversion DDIM Inversion w Memory Bank

Null-text Inversion

Figure 12: The editing quality of different guidance strategies.

in Fig. 12. It can be seen that extracting guidance information from zT using a separate branch can lead to deviations. This is due to the approximation bias in DDIM inversion. Although incorpo-

rating null-text optimization can yield more accurate results, it comes with higher time complexity. Our method tactfully utilizes a memory bank to store intermediate information during the inversion process, achieving accurate results while maintaining a time complexity of only 3.62 seconds.

- A.3 MORE RESULTS OF DRAGONDIFFUSION ON DIFFERENT APPLICATIONS

In this part, we present more visual results of our DragonDiffusion in different applications, as shown in Fig. 11. The first and second rows show the visualization of our object moving performance. It can be seen that our method has attractive object moving performance and good content consistency even in complex scenarios. The continuous moving editing presents attractive editing stability. The third row demonstrates that our method can perform natural point-drag editing of image content in different scenarios with several points. The fourth and fifth rows show the performance of our method in cross-image object pasting tasks. It can be seen that our method can fine-tune an object in one image and then naturally paste it onto another image. The last two rows demonstrate the performance of our method in object appearance replacing. It can be seen that our DragonDiffusion not only has good editing quality on small objects (e.g., ice-cream) but also performs well in replacing the appearance of large objects (e.g., cakes). Therefore, without any training and additional modules, our DragonDiffusion performs well in various image editing tasks.

- A.4 MORE QUALITATIVE COMPARISONS BETWEEN OUR DRAGONDIFFUSION AND OTHER METHODS ON CONTENT DRAGGING

In this part, we demonstrate more qualitative comparisons between our DragonDiffusion and other methods on more categories. Fig. 13 shows the comparison of drag editing on dogs. Fig. 14 shows the comparison of drag editing on horses. Fig. 15 shows the comparison of drag editing on cars. Fig. 16 shows the comparison of drag editing on churches and elephants. Fig. 17 shows the comparison of drag editing on face manipulation. In these comparisons, DragGAN Pan et al. (2023) requires switching between different models for different categories. Our method and DragDiffusion Shi et al. (2023) benefit from the powerful generalization capabilities of SD Rombach et al. (2022), enabling a single model to address image editing across different categories. These visualization results show that our method can produce better consistency with original images. At the same time, our method well balances the editing accuracy and generation quality.

- A.5 USER STUDY

To further compare with DragGAN Pan et al. (2023) and DragDiffusion Shi et al. (2023), we design a user study, which includes three evaluation aspects: generation quality, editing accuracy, and content consistency. The test samples involve various categories including dog, horse, car, elephant, church, and face. We allow 20 volunteers to choose the best-performing method in each of the 16 groups of images and then compile the votes in Fig. 18. As can be seen, our method has better subjective performance in these three aspects.

- A.6 DEMO VIDEO A demo video is attached to the supplementary materials.

Original Image DragGAN DragDiffusion Ours

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

- Figure 13: More qualitative comparison between our DragonDiffusion and other methods on the dog dragging. It can be seen that DragGAN Pan et al. (2023) is limited in generation quality and content consistency due to the capabilities of GAN models. DragDiffusion Shi et al. (2023) experiences an accuracy decline when dealing with larger editing drags, such as changing the posture of the dog’s body. In comparison, our method has promising performance in these aspects.

[Figure 211]

[Figure 212]

[Figure 213]

Original Image DragGAN DragDiffusion Ours

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

- Figure 14: More qualitative comparison between our DragonDiffusion and other methods on the horse dragging.

Original Image DragGAN DragDiffusion Ours

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

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

- Figure 15: More qualitative comparison between our DragonDiffusion and other methods on the car dragging.

Original Image DragGAN DragDiffusion Ours

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

- Figure 16: More qualitative comparison between our DragonDiffusion and other methods on the church and elephant dragging.

Original Face UserControllableLT DragGAN DragDiffusion Ours

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

- Figure 17: More qualitative comparison between our DragonDiffusion and other methods on the face dragging.

50, 16%

122, 38%

148, 46%

Generation Quality

108, 34%

84, 26%

128, 40%

Editing Accuracy

58, 18%

104, 33%

158, 49%

Consistency with Original Image

[Figure 286]

- Figure 18: User study of DragGAN Pan et al. (2023), DragDiffusion Shi et al. (2023), and our DragonDiffusion. The experiment is conducted on various categories including dog, horse, car, elephant, church and face.

