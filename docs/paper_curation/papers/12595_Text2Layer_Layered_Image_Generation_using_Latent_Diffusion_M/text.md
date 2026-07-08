## Text2Layer: Layered Image Generation using Latent Diffusion Model

Xinyang Zhang, Wentian Zhao, Xin Lu, and Jeff Chien Adobe Inc. {xinyangz, wezhao, xinl, jchien}@adobe.com

# arXiv:2307.09781v1[cs.CV]19Jul2023

||[Figure 1]|
|---|
<br><br>|[Figure 2]|
|---|
<br><br>|[Figure 3]|
|---|
<br><br>|[Figure 4]|
|---|
<br><br>Grand Canyon National Park|
|---|

||[Figure 5]|
|---|
<br><br>|[Figure 6]|
|---|
<br><br>|[Figure 7]|
|---|
<br><br>|[Figure 8]|
|---|
<br><br>outdoor waterproof rugs ...|
|---|

Fig. 1. Examples of two-layer images. Prompts are displayed on the top of images. Each example includes foreground (fg), background (bg), and mask component to compose a two-layer image. From left to right of each example: fg, bg, mask, and composed image.

### Abstract

Layer compositing is one of the most popular image editing workflows among both amateurs and professionals. Motivated by the success of diffusion models, we explore layer compositing from a layered image generation perspective. Instead of generating an image, we propose to generate background, foreground, layer mask, and the composed image simultaneously. To achieve layered image generation, we train an autoencoder that is able to reconstruct layered images and train diffusion models on the latent representation. One benefit of the proposed problem is to enable better compositing workflows in addition to the high-quality image output. Another benefit is producing higher-quality layer masks compared to masks produced by a separate step of image segmentation. Experimental results show that the proposed method is able to generate high-quality layered images and initiates a benchmark for future work.

### I..Introduction

Segmenting an image into layers is essential for image editing applications, such as background replacement and background blur. For many years, the development, therefore, has been focused on automatic approaches for various of image segmentation and matting problems [29], [57], [46], [15]. Recent development rethinks image editing problems from a generative perspective [34], [24], [4], [47] and aims to predict the expected editing region implicitly. For example, Text2live [3] aims to generate desired edits by generating a new image layer guided by text to realize non-disruptive editing. T2ONet [24] proposes to directly generate image editing results guided by text through gen-

erative models. These approaches are limited by the quality of the implicitly predicted editing region and limited by the generated image quality.

Recent efforts in large-scale text2image diffusion models, such as DALL·E 2 [43], Imagen [48], and Stable Diffusion [45] significantly improve the image generation quality and produce high-resolution photo-realistic images. Diffusion-based automatic image editing approach is emerging [2], [36], [47], [27], [25], [17], [4], [11], [63]. For example, Prompt-to-Prompt [17] modulates the crossattention maps in the inference steps to manipulate images. Soon InsturctPix2pix [4] utilized Prompt-to-Prompt to build a supervised dataset for image editing with natural language instruction and then finetuned a Stable Diffusion model with the dataset. However, these approaches still suffer from one of the following drawbacks. First, the user cannot constrain the region of editing, and editing methods often modify more pixels than it desires. Second, they do not resolve the issue that details are hard to describe in plain language and require trial and error to find the ideal editing instruction.

In this paper, we explore layered image generation guided by text with an intent of addressing the aforementioned challenges. As this is the first trial to generate image layers through diffusion models, we formulate the problem as generating a two-layer image, which includes foreground F, a background B, a layer mask m1, and the composed image. The layer mask controls a layer’s level of transparency, and the composed image is formed

1We borrowed the terminology of the layer mask from Photoshop, which is a nondestructive way to hide parts of an image or layer without erasing them

[Figure 9]

[Figure 10]

- Fig. 2. Layer mask examples. The scale, location, and number of objects vary largely.

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

- Fig. 3. Failure cases of salient object segmentation (left) and inpainting (middle and right). The red shaded regions indicate the object to be removed.

based on alpha blending. Examples are shown in Figure 1. With a layered image, a user is able to scale or move the foreground and replace foreground or background in a straightforward manner in image editing applications. Because of the layer mask, users’ edits can be explicitly controlled to apply to either the foreground or background.

The two biggest challenges are layered-image generation using diffusion models and training data synthesis. A trivial solution to generate layered images is running text2image multiple times, however, that doesn’t guarantee the compatibility of the generated images in each run. Another solution is generating an image via a text2image system and then running salient object segmentation on the generated image. However, failure masks often occur due to the diverse scale and location of objects and the number of objects in the image.

Motivated by the latent diffusion models, we train a layered-image autoencoder to learn their latent representation and then train diffusion models on the latent representation to generate layered images. Concretely, the layered-image autoencoder learns to encode and decode two-layer images with a multi-task learning loss that includes reconstruction loss and loss terms inspired by image composition. We found that including the composed image as a layer in the autoencoder training further improves the generated image quality. Such layered output facilitates layer compositing and even improves the image synthesis quality. Meanwhile, the generated layer mask quality also benefits from the generative network in terms of adapting to different object scales as well as having multiple objects as a foreground as shown in Figure 2.

To synthesize training data for layered image generation, we run salient object segmentation on training images used for text2image development. We chose to run salient object segmentation based on the observation that text prompts usually indicate the salient object clearly. As both the salient object segmentation and inpainting often

fail (as shown in Figure 3), we developed an automatic data synthesis system that can distinguish high-quality salient object mask and high-quality impainting results. That system produces the proposed LAION-L2I dataset, consisting of about 57.02M high-quality layered images among 600M LAION-Aesthetic dataset2 [49].

As a first trial to generate layered images, we developed an evaluation metric to measure the output quality. Results showed that the proposed method generally produces layered images with better image quality in terms of FID, higher mask accuracy, and greater text relevance compared to several Stable Diffusion-inspired baseline models. In summary, our contributions are three folds.

- 1) We developed a text2layer method to generate a layered image guided by text, including a foreground, a background, a mask and the composed image.
- 2) We introduced a mechanism to synthesize highquality layered images for diffusion model training and produces 57.02M high-quality layered images for future research.
- 3) We set a benchmark for layered-image generation. Experimental results showed that the proposed method generated higher-quality composed images with a higher text-image relevance score together with a better mask accuracy compared to baselines.

### II..Related Work

As layered image generation has not been systematically investigated, we discuss studies in text2image, textbased editing and image segmentation that are related to our work.

Text-based image generation Text-based image generation is the task of synthesizing images according to natural language descriptions. As a conditional generative problem, many previous works ([58], [61]) approach this problem by training GANs [13]) on image caption datasets. Another line of line work ([44], [7]) tackles the generation in an auto-regressive way with Transformers [54], usually utilizing VQ-VAE [53] codebooks to produce image tokens. Recently, diffusion-based approaches significantly boost the generated image quality in terms of semantic coherence and fine spatial details, such as Imagen [48], DALL·E 2 [43], GLIDE [38], and Parti [59]. Behind the scene, they are developed upon recent advances in denoising diffusion probabilistic models (DDPM, [20], [37]). DDPM [20] and its variants ([50]) learn the reversed process of a Markovian process that iterative adds noise to training images to produce white noise. Latent diffusion [45] trains the diffusion model on the latent space of a pretrained autoencoder instead of pixel space. For text2image generation, it incorporate classifier-free

2https://laion.ai/blog/laion-aesthetics/

guidance [21] to further improving sample quality and textimage relevance. We are motivated by the latent diffusion model and learn layered-image latent representations for layered image generation.

Text-guided image editing A line of papers ([41], [12]) takes a pretrained CLIP [42] image-text encoder to guide image manipulation. They usually measure the editing effect with cosine similarity between CLIP embedding of edited images and target texts. Style-CLIP [41] first inverts the input image into StyleGAN2’s latent space and optimizes the latent vector with CLIP to find the target image. However, Style-CLIP and other GAN-based image editing methods (e.g.,[12]) need to solve challenging GAN inversion [64], [1] problem first and do not support localized editing. Other works [3], [28] use CLIP without a pretrained generator by performing test-time optimization.

Inspired by the recent blooming of diffusion models, several works ([2], [36], [47], [27], [25], [17], [4], [11], [63]) propose to utilize a diffusion model as a generator for image editing. SDEdit [36] first adds noise to the input and then denoises the resulting images conditioned on a target. Blended diffusion [2] performs localized editing with a region mask and a CLIP-guided inference algorithm for DDPM. To generate personalized images with text prompts, Textual Inversion [11] and DreamBooth [47] finetunes the text embedding of pretrained diffusion models to generate images of personalized objects in novel scenarios. Prompt-to-Prompt [17] modifies the cross-attention maps in the inference steps to manipulate images. SINE [63] enables personalized image editing with a single image with model-based guidance and patch-based finetuning. InsturctPix2pix [4] creates a supervised model for image editing. Imagic [25] combines latent space interpolation and model finetuning to edit images with a text prompt. Direct Inversion [9] claims that simply denoising the latent vector of the input image with text instruction produces satisfying editing. Most of these approaches cannot constrain the region of editing, and we aim to generate explicit salient masks together with the image to facilitate editing and layer compositing workflows.

Image matting Image matting studies extracting the foreground alpha matte α from an image I, which is useful for image composition and editing. Formally, image matting algorithms solve an alpha matte α, such that for the i-th pixel of I: Ii ≈ αiFi + (1 − αi)Bi, where F and B are unknown foreground and background. We take the same formulation to compose the foreground and background in the layered image generation.

Image matting have been extensively studied [29], [57], [51], [31]. Traditional approaches [29], [16] for matting rely on affinity or optimization. DIM introduced Composition-1k dataset, and took deep convolutional networks for image matting. Since then, many new deep learn-

ing methods for image matting have been proposed [51], [32], [26]. Recently, MatteFormer [39] attacked matting with a modified SWIM Transformers [35]. Motivated by the success of matting methods, we leverage composition loss [57] and Laplacian loss [23] to train our autoencoder.

Salient object segmentation/detection (SOD) aims at highlighting visually salient object regions in images [56]. Similar to image matting, deep architectures [65], [30], [22], [5] beat the performance of traditional methods [6], [8]. DSS [22] proposed CNN with short-cut connections between the deeper and shallower features for SOD. To deal incomplete saliency, GCPANet [5] built a progressive context-aware feature aggregation module for effective feature fusion in SOD. ICON [65] is a deep architecture that improves the integrity of the predicted salient regions. Higher integrity of composition masks is vital to build our dataset since composition with an incomplete mask will tear apart in the foreground and background and produce many artifacts. Therefore, we chose ICON for estimating masks.

### III..Synthesizing High-Quality Layered Images

We formally define the two-layer image and introduce the proposed approach for high-quality layered image synthesis in this section. With the proposed approach, we generated a 57.02M high-quality layered-image dataset, named LAION-L2I, which includes 57M for training and 20K for testing. We refer to the test set as LL2I-E in the rest of this paper.

#### A..Definition of Two-Layer Image

Intuitively, a two-layer image represents an image with a foreground and a background, as well as a mask m to composite the two layers. Formally, a two-layer image is a triplet I = (F,B,m) where F, B, and m are the introduced foreground, background and mask, respectively. Throughout the paper, we assume they are of the same spatial dimension H × W. Hence, F,B ∈ R3×H×W and m ∈ RH×W. Figure 1 displays samples from LAION-L2I dataset.

#### B..Overview of LAION-L2I Dataset Construction

As shown in recent text2image efforts, large-scale datasets such as LAION-2B and LAION-5B [49] are essential for training diffusion models [45]. To generate pairs of two-layer images I and text prompts, we propose an automatic data annotation method to create a two-layer image out of each image in the LAION dataset, so that they are of a large scale and paired with text prompts. In particular, to cope with limited computational resources, as a first trial, we chose to generate layered images using the LAION-Aesthetics, (short for LAION-A) subset, which

contains around 600 million images with high aesthetic quality and text relevance3.

To generate a two-layer image out of each image in LAION-A, we apply a salient object segmentation method to extract the foreground parts from the original images and then fill the missing regions of the backgrounds using stateof-art image inpainting techniques. We chose salient object segmentation instead of segmenting objects of specific semantic categories for two purposes. First, it sets the promise of being able to generalize to a large-scale dataset. Second, the text prompt associated with each image is generally applicable to a two-layer image that is composed of a salient object, background, and a mask. Meanwhile,

- as salient object detection and inpainting methods do not always produce high-quality output, we trained two classifiers to filter out the samples that are with bad salient masks or bad inpainting results. C..Extracting foreground and background parts

We curated a two-layer image dataset based on LAIONA on the resolution of 512×512. As we defined a two-layer image I as a triplet of (F,B,m), we take a straightforward way of estimating the triplet from an existing image I such that

I ≈ mF + (1 − m)B (1)

Eq (1) is under-constrained with 6 degrees of freedom for each pixel. We estimate them as follows.

Estimation of F and m. In this work, we predict salient mask m from I using state-of-the-art salient object detection method ICON [65]. For the foreground image F, we set a threshold (0.1 for our case) for m and directly copy pixel values Ii of I at spatial locations which have mask value mi greater than the threshold.

Estimation of B. We utilize image inpainting technique to acquire the background B for an image I. Concretely, we first apply a dilation operation on m to obtain an augmented m˜ , which helps alleviate the errors in the estimation of m. Next we use the state-of-art diffusionbased inpainting4 to fill the masked region and produce the inpainted image B. We attempted to use the prompts “background” to enhance the background generation. However, we did not observe quality improvement in the output. So we chose to feed empty text prompt into the inpainting model.

In Section III-D, we will introduce how we further filter the data to obtain a higher quality dataset in detail. To differentiate from the filtered dataset, we name the originally built dataset LAION-L2I (U), where U means unfiltered. Please refer to supplementary material for more

- 3https://laion.ai/blog/laion-aesthetics/
- 4https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

[Figure 17]

[Figure 18]

- (a) Predicted “bad” masks

[Figure 19]

[Figure 20]

- (b) Predicted “good” masks

[Figure 21]

[Figure 22]

- (c) Predicted “bad” inpaintings

[Figure 23]

[Figure 24]

- (d) Predicted “good” inpaintings

Fig. 4. Predicted good and bad salient masks and inpaintings

statistics of the LAION-L2I. Some examples from LAIONL2I are visualized in Figure 1 (more in the supplement).

#### D..Quality filtering

We noticed that the mask prediction model for α and inpainting model for B do not always produce desired results. Specifically, as shown in Figure 14a, some of the masks are incomplete or contain too many background elements. As shown in Figure 14c, the inpainting model may introduce visual artifacts. In addition, the inpainting model tends to retrieve the objects in the missing regions. However, we expect it to only patch the background instead of the foreground. Otherwise, the diffusion model tend to generate contents that are not relevant to the text descriptions.

To solve the aforementioned issues, we train two classifiers to remove the samples that are with bad saliency map or inpainting quality. We first manually annotated 5000 training samples and 1000 test samples respectively. Specifically, we randomly sampled the generated two-layer images from LAION-L2I (U). For salient mask quality labeling, a sample is labeled as a bad sample if one of the three conditions occur:

- • The object masked out is largely incomplete.
- • The mask contains regions from the backgrounds.
- • The selection of the salient object is too small

For inpainting quality labeling, a sample is labeled as a bad sample if there is an object or artifacts in the inpainted region or the inpainted region is not cohesive with the

remaining image. Typical failure cases are shown in the Figure 3.

To obtain the two quality filters as mentioned above, we train two different classifiers with the same neural network architecture and training schedules. For mask quality classifier, the inputs are the original images I and their corresponding masks m. The inputs to the inpainting quality classifier are the inpainted images. The output of both classifiers are the quality labels. For implementation, we replace the last layer of EfficientNet b0 [52] with a fully connected layer with one single neuron which indicates the probability of a sample is good or not. Then we only fine-tune the last layer while freeze all other layers. After the classifiers are well-trained after 100 epochs, we feed all the samples from LAION-L2I (U) into the two classifiers and discard the samples with bad masks or poor inpainting results, which results in a filtered dataset LAION-L2I which contains 57.02M samples. The filtered results are visualized in Figure 14b and Figure 14d. More statistics and analysis of LAION-L2I and data filtering are available in the supplement.

### IV..Modeling A..Text2Layer Formulation

Our main task is to design a model which can generate a two-layer image I given a text prompt. Formally, we train a conditional generative model to model the probability distribution pθ(·;·) such that

I = (F,B,m) ∼ pθ(z;T) (2)

where T is the text prompt and θ is the parameter of the distribution p.

Equipped with LAION-L2I dataset, we are ready to build a conditional diffusion model to synthesize a twolayer image I given text T. Our model architecture is based on Stable Diffusion [45], which has already demonstrated high-quality results in text2image generation.

We need to adapt Stable Diffusion to the two-layer image generation task. Recall that Stable Diffusion employs a pre-trained autoencoder to project the original images to a latent space in a significantly lower dimension before feeding them to a noise prediction module which takes the form of a UNet [46] variant. In our scenario, we found that in addition to reducing computational cost, the autoencoder also plays an essential role in capturing the inherent structure of two-layer image I. Therefore, we designed a novel architecture for autoencoder, termed Composition-Aware Two-Layer Autoencoder (CaT2I-AE), tailored for two-layer image generation which can be easily generalized to multiple layers. In Section V-E, we demonstrate that such a simple design is sufficient to produce high-quality two-layer images.

#### B..CaT2I-AE Architecture

In the original SD-AE (the autoencoder used in Stable Diffusion), the decoder reconstructs an input image I with the encoder’s output latent vector z. We detail the changes made in CaT2I-AE, which compresses two-layer images into latent space and reconstructs them from the latent vectors. For the encoder, we stack F, B, and m into a single tensor X. We keep other parts of the encoder intact. While a complicated encoder is possible, we find this approach already achieved good generation results.

To obtain a better decoded result, we differentiate the generation process of image components and mask component for the last layer by applying multiple prediction heads. Specifically, we remove the last layer of SD-AE decoder and attach three prediction heads to predict F, B and m, respectively. Each prediction head is a tiny two layer convolutional network of conv-bn-relu-conv. Another modification we made compared to SD-AE is that we added an auxiliary output branch to predict the original image I. Our motivation is the supervision of the original image I introduced extra information in addition to F and

- B. In Section V-E, we show that this extra supervision slightly improves the generation quality.
- C..Training Objective

Our training objective for CaT2I-AE extends the original autoencoder training objective in [45] with terms for masks. Given the autoencoder’s reconstructed (F,ˆ B,ˆ I,ˆ mˆ ), the full multi-task loss L is defined as

L = Limg(F,ˆ B,ˆ Iˆ;F,B,I) + λLmask( ˆm,m) (3)

where Limg denotes the loss for image components (B,F,I), Lmask denotes the loss for mask channel, and λ (we take 1 in all the experiments) controls the tradeoff of reconstruction quality between image components and mask component. As for the image components Limg, we directly use a combination of LPIPS [62], ℓ1 norm and adversarial loss as in SD-AE. For the mask m, motivated by the works [60] on image matting, we consider a combination of ℓ1 loss, Laplacian loss [57], and composition loss [23] (details in the supplement).

Lmask = ℓ1(m,mˆ ) + 2ℓcomp(m) + 3ℓlap(m,mˆ ) (4)

After CaT2I-AE is well trained, we utilize the same cross-modal attentive UNet architecture used in work [45] to train a DDPM on the latent space of CaT2I-AE. In particular, let fθ be the UNet, and the loss function LDDPM is defined as

LDDPM = Ez,ϵ,t ∥ϵ − fθ(zt,t)∥22 (5)

where t is a random time step between 1 and tmax, and ϵ is an independent Gaussian noise, z is the latent vector produced by CaT2I-AE, and zt is a noisy version of z.

### V..Experiments

In this section, we extensively compare the proposed text2layer method on the two-layer image generation task to several baselines on the LAION-L2I dataset introduced in Section III. The evaluation focuses on the image quality, mask quality and text-image relevance.

#### A..Baseline Methods

We name our full model CaT2I-AE-SD, and compare it with the following baseline methods on our LAION-L2I dataset.

- • SD-v1-4 and SD-LAION-L2I: SD refers to the Stable Diffusion model for text-to-image synthesis. We compare the quality of CaT2I-AE-SD’s composed images I = mF + (1 − m)B with images generated from Stable Diffusion v1-45. Additionally, because the LAION-L2I dataset is significantly smaller than the LAION-A dataset, we trained a SD model6 for text2image using the LAION-L2I dataset for fair comparison. We denote these two SD models as SDv1-4 and SD-LAION-L2I.
- • SD-AE-UNet and SD-AE-UNet (ft): for these two baselines, we pass each component of I (a.e.,F,B,m) into the SD-AE, and train a UNet-archtecture diffusion model which takes the stacked latent vectors as inputs. Formally, let g(·) be the SD-AE, we feed z defined as below into the UNet.

zF,zB,zm = g(F),g(B),g(m) z = concatch(zF,zB,zm) (6)

where concatch(···) concatenates input tensors along the feature channel. For SD-AE-UNet, we train the UNet model from scratch, while for SD-AE-UNet (ft), we finetune the UNet from the SD-v1-4. By evaluating CaT2I-AE-SD against these two baselines, we show that it is essential to have a well-designed autoencoder for two-layer image generation task.

- • CaT2I-AE-SD (no-sup): this is a variant of CaT2I-AESD without the supervision branch for the original image I. We show the effectiveness of introducing this additional supervision branch in Section V-E.

Because training high-resolution diffusion models (i.e., 512×512) requires excessive computational resources, we primarily evaluate all methods on the 256×256 resolution. In particular, all the methods except SD-v1-4 target at a resolution of 256. We also trained a CaT2I-AE-SD on the resolution of 512, named CaT2I-AE-SD (512) to demonstrate the generation quality of various resolutions.

5https://huggingface.co/CompVis/stable-diffusion-v1-4 6We re-used the SD-AE and trained the same UNet architecture from

scratch for the diffusion model as SD does.

#### B..Metrics

To quantitatively analyze the capability of CaT2I-AESD in synthesizing two-layer images in an extensive way, we take the following metrics.

Fr´echet inception distance (FID) FID is a metric used to evaluate the quality of images generated by a generative model [19]. It compares the distribution of generated images with the distribution of real images.

We follow the same FID metric as done in [45], [48] and take FID to measure the fidelity of composed images. We leverage the Clean-FID proposed in [40]7. As the FID score significantly depends on the real image distribution, we report FID scores on two test sets. One is LL2I-E (See Section III) which contains 20K test samples. The other is a subset from unfiltered LAION-A, which also has 20K samples. The two sets are mutually exclusive. We abbreviate the latter one LA in the following experiments. We introduce LA for quantifying our model’s ability to generate diverse images beyond the distribution of LAIONL2I.

CLIP score Previous work [18] discovered that the CLIP [42], a vision-language model pre-trained on 400M noisy image and caption pairs, could be used for automatic evaluation of image captioning without the need for references. In particular, the cosine similarity between the features of the image encoder and language encoder could serve as a surrogate metric for image-text relevance. We follow the same CLIP score metric as it was used in [10], [14] and refer to the cosine similarity as CLIP score. In particular, we take the CLIP score to validate that images composed by our CaT2I-AE-SD faithfully reflect text prompts. We calculate the CLIP score on the LL2I-E subset with Vit-L/14.

Intersection-Over-Union (IOU). IOU is widely adopted to assess the performance of semantic segmentation. It computes the overlapping between a pair of prediction mask and ground-truth mask as a fraction.

In addition to the quality of composited images, we expect masks m to capture meaningful regions in the composited images. To evaluate the generated masks, we compute the IOU of models’ predicted mask mˆ with two sets of “reference” masks because no ground-truth masks are available for the generated masks given certain text prompts. We use ICON’s [65] predictions on the composed images as the first reference set, since that is how we estimate masks in dataset construction. For the second set, we manually annotate the masks of a test set which consists of 1500 generated composited images for each model.

7https://github.com/GaParmar/clean-fid

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) CaT2I-AE-SD (b) CaT2I-AE-SD (no-sup)

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

(c) SD-AE-UNet (d) SD-AE-UNet (ft)

Fig. 5. Composited samples from CaT2I-AE-SD and baseline models for 256×256 resolution. The prompts are (1) Dogs at the entrance of Arco Iris Boutique and (2) Haunted Mansion Holiday

- at Disneyland Park. Each 2 × 2 block displays the composited images and masks m of the corresponding models. Find more in the supplement.

#### C..Implementation Details

All the experiments are running on four 8×A100 GPU machines. For SD-AE-UNet and SD-LAION-L2I, we train the UNet model for 75k steps with DDPM training. For SD-AE-UNet (ft), we expand the number of input channels and output channels for the UNet model and initialize the weight from SD-v1-4, we then finetune the UNet model for 25k steps. CaT2I-AE-SD and CaT2I-AE-SD (no-sup) require training the autoencoder from scratch, and we train both of them for 250k steps. We follow the same training scheme on UNet for both models as the SD-AE-UNet. To train the 512 resolution CaT2I-AE-SD (512), we resume from CaT2I-AE-SD weight and train for 75k steps with 512 images. During inference, we sample two-layer images with DDIM [50] scheme and classifier-free guidance [21]. In particular, the number of DDIM iterations is 50 and the guidance scale is 7.5.

#### D..Qualitative Results

Figure 5 displays samples from CaT2I-AE-SD and baseline models for the 256×256 results. As shown in the figure, the samples from CaT2I-AE-SD clearly have better details with sharper and more accurate masks compared to other methods. On composited images, CaT2I-AE-SD’s results have fewer artifacts and distortion. On masks, CaT2I-AE-SD’s results are more accurate, while results

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Fig. 6. F, B, m components and composed images for two-layer images sampled from CaT2I-AE-SD (512). Prompts are omitted. From top to down: F, b, and m. For more samples cf. the supplement.

generated by SD-AE-UNet and SD-AE-UNet (f) are noisy.

Figure 6 delves into components (F,B,m) of images generated from CaT2I-AE-SD (512). Take SD samples in the bottom row into account, we find CaT2I-AE-SD (512) generates samples with comparable quality as SD-v1-4. In contrast to the static images generated by SD-v1-4, the F,B,m components coming with CaT2I-AE-SD simplify downstream image editing tasks.

#### E..Quantitative Results

The rest of this section presents quantitative comparisons on image quality, mask quality, and text-image relevance.

Image quality Here we demonstrate that CaT2I-AE-SD achieves superior composition quality compared to baseline methods. We present in Table I (in the 4th and 5th column) FID scores of models on both test sets. Our first observation is that CaT2I-AE-SD outperforms baseline models in generating 256 × 256 two-layer images by big margins, which are around 3.5 for LL2I set and 5 for LA set. We attribute it to the distributional misalignment between the (F,B,m) components of a two-layer image and a natural image used in SD. Such distributional misalignment could cause larger errors in reconstructing F, B, and m from the latent space of SD’s autoencoder. In contrast, CaT2I-AE’s decoder decompresses F, B, and m with separate prediction branches, making better usage of the decoder’s capacity.

Meanwhile, CaT2I-AE-SD achieves a lower FID com-

|Dataset<br><br>|Resolution|Method<br><br>|FID (LL2I-E, ↓)|FID (LA, ↓)<br><br>|CLIP Score (↑)|
|---|---|---|---|---|---|
|LAION-L2I|256<br><br>|CaT2I-AE-SD CaT2I-AE-SD (no-sup) SD-AE-UNet SD-AE-UNet (ft) SD-LAION-L2I<br><br>|10.51 13.84 18.53 20.15 11.80|14.83 19.54 21.96 24.84 15.06<br><br>|0.251 0.241 0.219 0.234 0.250<br><br>|
|LAION-L2I LAION-2B + LAION-A<br><br>|512<br><br>|CaT2I-AE-SD (512) SD-v1-4 [45]|7.54 9.13<br><br>|13.01 9.13<br><br>|0.261 0.279|

- TABLE I. FID and CLIP scores for our CaT2I-AE-SD and various baselines. FID is evaluated on LL2I-E and LA testset, and CLIP score is tested on LL2I-E. The arrows indicate preferred directions for the metrics.

|Model<br><br>|IOU (ICON)|IOU (human)<br><br>|
|---|---|---|
|CaT2I-AE-SD CaT2I-AE-SD (no-sup) SD-AE-UNet SD-AE-UNet (ft)|0.885 0.904 0.779 0.670<br><br>|0.799 0.851 0.766 0.681|

- TABLE II. IOU between models predicted masks and the two reference sets – ICON’s prediction (as ICON, [65]) and human annotated foreground masks (as human).

|Dataset|FID|CLIP Score<br><br>|IOU (human)|
|---|---|---|---|
|LAION-L2I LAION-L2I (U)|10.51 13.82<br><br>|0.251 0.234<br><br>|0.799 0.736|

- TABLE III. Metrics for CaT2I-AE-SD trained on LAION-L2I and a unfiltered variant of LAION-L2I with same size, LAIONL2I (U)

their superior IOU on the human-annotated test set shows CaT2I-AE-SD’s masks are better at capturing foreground semantics compared to other baselines. CaT2I-AE-SD shows a slightly worse number than CaT2I-AE-SD (nosup) indicating that the usage of the composed image in the autoencoder training drives better image quality but might hurt mask quality. That observation introduces future work to better balance each layer’s generation.

Image-text relevance As depicted in Table I (in the last column), the proposed CaT2I-AE-SD outperforms all baseline models in the 256 resolution. CaT2I-AE-SD (512) achieves a better text-image relevance compared to CaT2I-AE-SD, which indicates a promising performance improvement when training on a higher resolution. Similarly, we list the SD-v1-4’s results for reference, and the proposed CaT2I-AE-SD’s result is on par with SD-v1-4’s results even if the SD-v1-4 was trained on a much larger dataset. The results suggest that the combination of our efforts on the dataset and model architecture ensures the generated two-layer images follow the instruction of text prompts.

pared to SD-LAION-L2I on both test sets, which indicates that the composed image quality benefits from the layeredimage generation training compared to the text2image setting. It is possible that the mask in the layered-image brought additional information that benefits the composed image quality. CaT2I-AE-SD (512) achieves better results than CaT2I-AE-SD, which sets the promise that training on a higher resolution could further improve the image quality significantly. We also notice a considerable improvement from CaT2I-AE-SD (no-sup) to CaT2I-AE-SD. This validates our hypothesis that the supervision of the original image I can enhance generative quality.

#### F..Effectiveness of Data Filtering

Large-scale high-quality datasets are indispensable to the success of deep neural networks in computer vision. Besides the quality metrics shown above, here we demonstrate the effect of our data filtering strategies proposed in Section III-D. Illustrated in Table III, the large improvement of FID, CLIP score, and IOU for CaT2I-AESD on the LAION-L2I and its unfiltered variant LAIONL2I (U) demonstrates the effectiveness of the proposed dataset synthesis approach discussed in Section III.

Lastly, we showed the results of SD-v1-4 [45], which was trained on a much larger dataset, for reference. The FID of SD-v1-4 is not directly comparable with other approaches in the table, but CaT2I-AE-SD still achieves lower FID compared to the SD on LL2I set for the resolution of 512, demonstrating the success of our whole two-layer image generation pipeline. Though in the fifth column, we find CaT2I-AE-SD has worse FID compared to the SD-v1-4 on a more broad LA set, we believe this is reasonable considering our current LAION-L2I is less than 1/10 as the size of LAION-Aesthetics and less than 1/30 as the size of LAION-2B. For future work, we will create a larger version of LAION-L2I to eliminate the gap. Mask quality We examine the quality of the masks produced by two-layer image generative models. The IOUs highlighted in Table II show that compared to other baselines, CaT2I-AE-SD and CaT2I-AE-SD (no-sup) produce masks that are more accurate according to the reference ICON [65] salient detection model. In addition,

### VI..Conclusions and Discussions

In this paper, we proposed a layered-image generation problem and scoped it to a two-layer image generation setting. We created a 57.02M high-quality layered-image dataset and used it to build a layered-image generation system using latent diffusion models. We designed the evaluation metric for the layered-image generation and demonstrated a benchmark in terms of image quality, image-text relevance, and mask quality.

Additionally, the proposed method is not limited to two layers, which can be applied to any fixed number of layers.

Meanwhile, a conditional model can be developed as a natural extension of this work, which potentially could generate a layer given existing layers and that can further enable layered image generation of an arbitrary number of layers. We leave it as a future work.

### References

- [1] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 3

- [2] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3

- [3] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 1, 3

- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022. 1, 3

- [5] Zuyao Chen, Qianqian Xu, Runmin Cong, and Qingming Huang. Global context-aware progressive aggregation network for salient object detection. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), 2020. 3

- [6] Ming-Ming Cheng, Niloy J Mitra, Xiaolei Huang, Philip HS Torr, and Shi-Min Hu. Global contrast based salient region detection. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2014. 3

- [7] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-to-image generation via transformers. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2021. 2

- [8] Wolfgang Einh¨auser and Peter K¨onig. Does luminance-contrast contribute to a saliency map for overt visual attention? European Journal of Neuroscience, 2003. 3

- [9] Adham Elarabawy, Harish Kamath, and Samuel Denton. Direct inversion: Optimization-free text-driven real image editing with diffusion models. arXiv preprint arXiv:2211.07825, 2022. 3

- [10] Kevin Frans, Lisa Soros, and Olaf Witkowski. CLIPDraw: Exploring text-to-drawing synthesis through language-image encoders. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2022. 6

- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In Proceedings of International Conference on Learning Representations (ICLR), 2023. 1, 3

- [12] Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip-guided domain adaptation of image generators. arXiv preprint arXiv:2108.00946, 2021. 3

- [13] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2014. 2

- [14] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. ArXiv e-prints, 2021. 6

- [15] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of IEEE International Conference on Computer Vision (ICCV), 2017. 1

- [16] Kaiming He, Christoph Rhemann, Carsten Rother, Xiaoou Tang, and Jian Sun. A global sampling method for alpha matting. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2011. 3

- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with

- cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 3
- [18] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021. 6

- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS),

2017. 6

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 3, 7

- [22] Qibin Hou, Ming-Ming Cheng, Xiaowei Hu, Ali Borji, Zhuowen Tu, and Philip HS Torr. Deeply supervised salient object detection with short connections. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3

- [23] Qiqi Hou and Feng Liu. Context-aware image matting for simultaneous foreground and alpha estimation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 3, 5, 16

- [24] Wentao Jiang, Ning Xu, Jiayun Wang, Chen Gao, Jing Shi, Zhe Lin, and Si Liu. Language-guided global image editing via crossmodal cyclic mechanism. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1

- [25] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Textbased real image editing with diffusion models. arXiv preprint arXiv:2210.09276, 2022. 1, 3

- [26] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson W.H. Lau. Modnet: Real-time trimap-free portrait matting via objective decomposition. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), 2022. 3

- [27] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3

- [28] Gihyun Kwon and Jong Chul Ye. Clipstyler: Image style transfer with a single text condition. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2022. 3

- [29] Anat Levin, Dani Lischinski, and Yair Weiss. A closed-form solution to natural image matting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008. 1, 3

- [30] Guanbin Li and Yizhou Yu. Visual saliency based on multiscale deep features. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 3

- [31] Jizhizi Li, Jing Zhang, Stephen J. Maybank, and Dacheng Tao. Bridging composite and real: Towards end-to-end deep image matting. International Journal of Computer Vision, 2022. 3

- [32] Yaoyi Li and Hongtao Lu. Natural image matting via guided contextual attention. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), 2020. 3, 16

- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Proceedings of the European Conference on Computer Vision (ECCV), 2014. 11, 13, 14

- [34] Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, and Sanja Fidler. Editgan: High-precision semantic image editing. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2021. 1

- [35] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of IEEE International Conference on Computer Vision (ICCV), 2021. 3

- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis

- and editing with stochastic differential equations. In Proceedings of International Conference on Learning Representations (ICLR), 2022. 1, 3
- [37] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proceedings of IEEE Conference on Machine Learning (ICML), 2021. 2

- [38] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In Proceedings of IEEE Conference on Machine Learning (ICML), 2022. 2

- [39] GyuTae Park, SungJoon Son, JaeYoung Yoo, SeHo Kim, and Nojun Kwak. Matteformer: Transformer-based image matting via priortokens. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3

- [40] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 6

- [41] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 3

- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of IEEE Conference on Machine Learning (ICML), 2021. 3, 6, 16

- [43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1, 2

- [44] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Proceedings of IEEE Conference on Machine Learning (ICML), 2021. 2

- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 2, 3, 5, 6, 8, 16

- [46] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015, 2015. 1, 5

- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation. arXiv preprint arxiv:2208.12242, 2022. 1, 3

- [48] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1, 2, 6

- [49] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Neural Information Processing Systems: Datasets and Benchmarks Track,

2022. 2, 3, 11

- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proceedings of International Conference on Learning Representations (ICLR), 2021. 2, 7

- [51] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Semantic image matting. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 3

- [52] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of IEEE Conference on Machine Learning (ICML), 2019. 5

- [53] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In I. Guyon, U. Von Luxburg, S.

- Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2017. 2
- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2017. 2

- [55] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022. 14

- [56] Wenguan Wang, Qiuxia Lai, Huazhu Fu, Jianbing Shen, Haibin Ling, and Ruigang Yang. Salient object detection in the deep learning era: An in-depth survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. 3

- [57] Ning Xu, Brian Price, Scott Cohen, and Thomas Huang. Deep image matting. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1, 3, 5, 16

- [58] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2

- [59] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 2

- [60] Qihang Yu, Jianming Zhang, He Zhang, Yilin Wang, Zhe Lin, Ning Xu, Yutong Bai, and Alan Yuille. Mask guided matting via progressive refinement network. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2021. 5, 16

- [61] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-to-image generation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2

- [62] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 5

- [63] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. arXiv preprint arXiv:2212.04489, 2022. 1, 3

- [64] Jun-Yan Zhu, Philipp Kr¨ahenb¨uhl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. 3

- [65] Mingchen Zhuge, Deng-Ping Fan, Nian Liu, Dingwen Zhang, Dong Xu, and Ling Shao. Salient object detection via integrity learning. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2022. 3, 4, 6, 8, 14

## Appendix I: More Results

In Section 5.4, Figures 5 and 6 illustrate the powerful generative ability of CaT2I-AE-SD. We visualize more samples in this section.

White Goat Beside Fence

scituate light scituate lighthouse from the parking lot

Men Nike Air Max 90 Ul Tra Essential Running Shoes AAA 331

Chuck Klosterman Sees the Future, Blurrily

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

composed

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

mask

- Fig. 7. More 256 samples for CaT2I-AE-SD on LAION-L2I test set. Prompts are displayed at the top.

Emily Ham and Queenie Zeng at STEM Fellowship Indicium Competition

Indian Art Amrita Sher-Gil Self Portrait III -

Canvas Prints

click here to view larger image of Airedale Terrier

Lista Swivel Stool

composed

mask

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

- Fig. 8. Some failed generations from CaT2I-AE-SD. Prompts are displayed at the top.

#### A..More 256 samples

We show more 256 × 256 two-layer images of our CaT2I-AE-SD and baseline methods on LAION-L2I test set in Figures 7 and 9.

#### B..CaT2I-AE-SD (512) samples

Figure 10 provides samples of CaT2I-AE-SD (512) with prompts from test set of LAION-L2I. In addition, Figure 11 displays samples of CaT2I-AE-SD (512) with prompts from more commonly used MSCOCO dataset [33].

#### C..Failure cases

We present some failure cases of CaT2I-AE-SD in Figure 8. We noticed that CaT2I-AE-SD sometimes predicts objects with the wrong geometry (faces in the first column) or structures (the dog in the third column). Besides, some masks (the second and fourth columns) are inaccurate.

White Goat Beside Fence

scituate light scituate lighthouse from the parking lot

Men Nike Air Max 90 Ul Tra Essential Running Shoes AAA 331

Chuck Klosterman Sees the Future, Blurrily

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

composed

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

mask

(a) CaT2I-AE-SD (no-sup)

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

composed

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

mask

(b) SD-AE-UNet

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

composed

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

mask

(c) SD-AE-UNet (ft)

Fig. 9. More 256 samples for baseline methods on LAION-L2I test set. Prompts are displayed at the top.

## Appendix II: Dataset

#### D..Data filtering

In Section 3.4, we discussed the filtering of low-quality masks and inpaintings to ensure the final quality of our LAION-L2I dataset. More examples of before and after data filtering are visualized in Figure 14.

#### E..More examples

Figures 12 and 13 depict ten more two-layer images from our 57.02M LAION-L2I dataset.

#### F..Analysis and statistics

As a complement to Section 3, we present several quantitative analyses of LAION-L2I and LAIONAesthetics (short for LAION-A8 [49]) here. The analysis focuses on the complexity of images, masks, and text prompts. Plus,we also compare the text-image relevance of LAION-L2I and LAION-A.

8https://laion.ai/blog/laion-aesthetics/

Basil in the plate. Flat lay, top view

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

Wonderful and beautiful underwater world with corals and ...

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

Mark Hamill on the set of Star Wars Episode IV: A New Hope, ...

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

Smooth leather boots Black Goots

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

Juniors Tasha Kalra and Sophie Tom stand in front of Rowell Hall

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

foreground background mask image

Fig. 10. More CaT2I-AE-SD (512) synthesized two-layer images. The prompts above images are from LAION-L2I test set.

A couple of kittens standing around metal bowls on a tray.

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

A red motorcycle is parked outside of a cafe.

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

A woman that is standing in the water next to a surfboard.

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

A close up photo of a brown bear

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

A kite flying in the air above the ocean

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

foreground background mask image

- Fig. 11. More CaT2I-AE-SD (512) synthesized two-layer images. The prompts above images are from MSCOCO 2014 [33] validation set.

Rear View Of The Dog Jumping For The Ball

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

iceberg_near_triton_island_green_bay_central

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

Pinterest the world’s catalog of ideas for Prefabricated garage ...

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

Important Points a Checklist for Funeral Planning Must ...

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

1970 Plymouth AAR Cuda For Sale ...

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

foreground background mask image

- Fig. 12. Samples of two-layer images from our LAION-L2I dataset. Prompts are on the top of images.

Since the magnitude of both datasets are large, we sample 100K examples randomly from each dataset for following analyses.

Objects. We apply Yolov7 [55], a high-performance object detector trained on MSCOCO [33] dataset, on both datasets and obtain object categories statistics (in Figure 15) and the number of objects per image statistics (in Figure 16). Though MSCOCO has limited object categories and fewer diverse images compared to LAION5B. We argue that the above analysis with Yolov7 partially reflects that images retained in LAION-L2I have similar complexity to those of LAION-A in terms of “objects.”

We observe that both distributions of number of objects per category and number of objects per image on our LAION-L2I dataset and MSCOCO are similar. For Figure 15, only a few categories have a noticeable difference in terms of the number of objects. For Figure 16, LAIONL2I has slightly more images with two objects compared with LAION-A which has more images of zero MSCOCO objects.

Mask. In constructing LAION-L2I, we leverage

Golf Tee Bobbin Holders

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

northern-cardinal-male-kim-smith

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

Waterproof Outdoor Solar Wall Light with 56 LED

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

forest backlighted by golden sunlight before sunset ...

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

How professional Degree holders are taking the ...

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

foreground background mask image

Fig. 13. More samples of two-layer images from our LAION-L2I dataset. Prompts are on the top of images.

|percentile (p%)|length (LAION-A)<br><br>|length (LAION-L2I)|
|---|---|---|
|1 10 25 50 75 90 99<br><br>|1 3 5 7 10 15 36|1 3 5 7 10 15 36<br><br>|

TABLE IV. Percentiles of the number of tokens in text prompts of LAION-A and dataset. Stopwords, symbols, and punctuation are removed for counting.

ICON [65] to estimate masks m from images in LAION-A. Again, we compare statistical differences between LAIONL2I and LAION-A. We first study the area ratio of salient regions in images under various binary thresholds. As illustrated in Figure 17, the distributions of the area ratio of masks are similar for LAION-L2I and LAION-A across all the thresholds. Our second study concerns the number of connected components for masks, which indicates scattering patterns of objects and the noise level of masks. Still, as shown in Figure 18, the distributions of the two datasets are almost the same.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

- (a) Predicted “bad” masks

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

- (b) Predicted “good” masks

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- (c) Predicted “bad” inpaintings

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

- (d) Predicted “good” inpaintings

Fig. 14. Predicted good and bad salient masks and inpaintings

100k

5

2

10k

5

count

2

1000

5

2

100

5

2

0 10 20 30 40 50 60 70

category

(a) LAION-A

100k

5

2

10k

5

count

2

1000

5

2

100

5

2

0 10 20 30 40 50 60 70

category

(b) LAION-L2I Fig. 15. Number of objects per MSCOCO 80 category in LAIONA (a) and LAION-L2I (b).

35k

30k

30k

25k

25k

20k

count

count

20k

15k

15k

10k

10k

5k

5k

0

0

0 2 4 6 8 10

0 2 4 6 8 10

# objects per image

# objects per image

(b) LAION-L2I Fig. 16. Number of objects per image in LAION-A (a) and LAION-L2I (b).

(a) LAION-A

|dataset<br><br>|CLIP score|
|---|---|
|LAION-A LAION-L2I|0.278 ± 0.037 0.273 ± 0.034<br><br>|

TABLE V. CLIP scores for pairs of image and prompts from LAION-A and LAION-L2I.

Prompt. For the text prompts, we compare the lengths and the word frequency on LAION-L2I and LAION-A. Table IV displays caption lengths at different percentiles. It is clear that both datasets have prompts with similar lengths. Furthermore, to gain a word-level understanding, we calculate the Pearson correlation coefficient for frequencies of the top 5K words between these two datasets, the value is 0.930, suggesting LAION-L2I roughly represents a uniformly distributed subset of LAION-A. Finally,

threshold=0.3 threshold=0.5 threshold=0.7 threshold=0.9

10k

5

2

1000

5

count

2

100

5

2

10

5

- 0 0.5 1

- 1
- 2

0 0.5 1 0 0.5 1 0 0.5 1

area ratio area ratio area ratio area ratio

##### (a) LAION-A

threshold=0.3 threshold=0.5 threshold=0.7 threshold=0.9

10k

5

2

1000

5

count

2

100

5

2

10

5

- 0 0.5 1

- 1
- 2

0 0.5 1 0 0.5 1 0 0.5 1

area ratio area ratio area ratio area ratio

(b) LAION-L2I

- Fig. 17. Histograms of area of masks (normalized to [0, 1]) predicted by ICON. We take four different thresholds for binarizing masks.

10 20 30 40

- 1
- 2

5

10

2

5

100

2

5

1000

2

5

10k

2

5

10 20 30 40

# connected components # connected components

count

threshold=0.5 threshold=0.9

(a) LAION-A

10 20 30 40

- 1
- 2

5

10

2

5

100

2

5

1000

2

5

10k

2

5

10 20 30 40

# connected components # connected components

count

threshold=0.5 threshold=0.9

(b) LAION-L2I

- Fig. 18. Histograms of number of 8-way connected components in masks predicted by ICON. We take two different thresholds for binarizing masks. We filter out connected components whose areas are smaller than 100.

we quantify the image-text relevance with CLIP score [42] as we have done in Section 4. We notice from Table V that the two datasets have similar CLIP scores.

## Appendix III: Implementation Details

We detail the loss terms used in Section 4.3 for training our CaT2I-AE.

Image loss. For the supervision of image components (F,B,I) , we use the same loss terms as used in Latent Diffusion [45]. The loss terms are separately applied to

each of F,B,I with equal weights.

Inspired by literature of image matting [57], [60], [32], [23]. We enforce composition loss [57] and Laplacian loss [23] and for the CaT2I-AE decoder’s predicted mask mˆ and the ground-truth mask m.

Composition loss. Let C be the composed image of (F,B,m), in other word, C = mF +(1−m)B. Then the composition loss [57] is defined as

ℓcomp =

i

(Ci − Cˆi)2 + ϵ2 (7)

where i run through all the pixels, Cˆ = mFˆ + (1 − mBˆ ) is the composed image with decoder’s predicted mask mˆ , and ϵ set to 10−6 for numerical stability.

Laplacian loss. Laplacian loss [23] can be used to measure the difference between the predicted mask mˆ with the ground-truth mask m. Specifically,

ℓlap =

3

2j−1∥ϕj( ˆm) − ϕj(m)∥1 (8)

j=1

where ϕj(m) denotes the j-th level of Laplacian pyramid of the mask.

