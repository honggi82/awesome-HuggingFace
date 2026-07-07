# arXiv:2309.05793v1[cs.CV]11Sep2023

## PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models

### Li Chen1*, Mengyi Zhao1, 2*, Yiheng Liu1*, Mingxu Ding1, Yangyang Song1, Shizun Wang1, 3, Xu Wang1, Hao Yang1, Jing Liu1, Kang Du1, Min Zheng1

1ByteDance, Beijing, China. 2Beihang University, Beijing, China. 3National University of Singapore. {chenli.phd, liuyiheng.yolo, dingmingxu.leo, songyangyang.2021, wangxu.ailab, jing.liu, dukang.daniel, zhengmin.666}@bytedance.com, zhaomengyi@buaa.edu.cn, shizun.wang@u.nus.edu, yanghao.alexis@foxmail.com

##### Abstract

Personalized text-to-image generation has emerged as a powerful and sought-after tool, empowering users to create customized images based on their specific concepts and prompts. However, existing approaches to personalization encounter multiple challenges, including long tuning times, large storage requirements, the necessity for multiple input images per identity, and limitations in preserving identity and editability. To address these obstacles, we present PhotoVerse, an innovative methodology that incorporates a dual-branch conditioning mechanism in both text and image domains, providing effective control over the image generation process. Furthermore, we introduce facial identity loss as a novel component to enhance the preservation of identity during training. Remarkably, our proposed PhotoVerse eliminates the need for test time tuning and relies solely on a single facial photo of the target identity, significantly reducing the resource cost associated with image generation. After a single training phase, our approach enables generating high-quality images within only a few seconds. Moreover, our method can produce diverse images that encompass various scenes and styles. The extensive evaluation demonstrates the superior performance of our approach, which achieves the dual objectives of preserving identity and facilitating editability. Project page: https://photoverse2d.github.io/

### Introduction

The remarkable advancements in text-to-image models, e.g., Imagen (Saharia et al. 2022), DALL-E2 (Ramesh et al. 2022), and Stable Diffusion (Rombach et al. 2022), have garnered significant attention for their ability to generate photorealistic images based on natural language prompts. Despite the impressive capability of these models to generate diverse and sophisticated images by leveraging largescale text-image datasets, they encounter difficulties when it comes to synthesizing desired novel concepts. Consider the scenario where users aim to incorporate themselves, family members, or friends into a new scene, achieving the desired level of fidelity solely through text descriptions becomes challenging. This challenge stems from the fact that these

*These authors contributed equally.

[Figure 1]

Figure 1: Our proposed PhotoVerse exhibits a wide range of diverse results. By providing a single reference image featuring the target concept alongside various prompts, PhotoVerse facilitates the generation of high-quality images that align seamlessly with the given prompts. Notably, our method achieves this outcome within a matter of seconds, eliminating the need for test-time tuning.

novel concepts were absent from the dataset used for training the models, making it hard to generate accurate representations based solely on textual information. In the pursuit of enabling personalized text-to-image synthesis, several methods e.g., Dreambooth (Ruiz et al. 2023a), Textual Inversion (Gal et al. 2022), DreamArtist (Dong, Wei, and Lin 2022), and CustomDiffusion (Kumari et al. 2023) primarily focused on identity preservation and propose the inverse transformation of reference images into the pseudo word through per-subject optimization. Text-to-image models undergo joint fine-tuning to enhance fidelity. The resulting optimized pseudo word can then be leveraged in new prompts to generate scenes incorporating the specified concepts. However, this optimization-based paradigm often requires expensive computational resources and large storage requirements, taking minutes to hours when executed on high-end GPUs. Such lengthy processing times and tuning processes render it impractical for user-friendly usage, where short time generation is desirable. Also, these approaches had limitations in their language editing capabilities due to potential overfitting on a tiny identity-specific image dataset. Recent encoder-based methods e.g., E4T (Gal et al. 2023a), InstantBooth (Shi et al. 2023), SuTI (Chen

et al. 2023), Profusion (Zhou et al. 2023) aimed to address these challenges by optimizing pseudo token embeddings, introducing sampling steps, or incorporating new modules for concept injection. However, challenges persist, such as the need for multiple input images and test-time tuning (Sohn et al. 2023) to maintain identity preservation and editability.

To address the aforementioned issues, we present a novel methodology that leverages a dual-branch conditioning mechanism, combining improved identity textual embeddings and spatial concept cues through dual-modality adapters in both the text and image domains. Furthermore, we introduce a facial identity loss component during training to enhance identity preservation. Our method stands out by effectively balancing high identity similarity and robust editing capabilities. Crucially, our approach achieves a remarkable reduction in generation time for personalized textto-image models, completing the process in just a few seconds and utilizing only a single concept photo. This significant advancement in efficiency streamlines the generation workflow and enhances the user experience.

In summary, our contributions can be categorized into three key aspects:

- • We propose a novel architecture for user-friendly text-toimage personalization. Our approach eliminates the need for test-time tuning and requires only a single image of the target subject. This results in a rapid and effortless generation, typically completed in ∼ 5 seconds.
- • We introduce a dual-branch concept injection paradigm that effectively extracts identity information from both textual embeddings and visual representations. By leveraging this approach, we enhance the preservation of identity during training. Additionally, we incorporate a face identity loss component to further facilitate identity preservation throughout the training process.
- • We demonstrate the exceptional quality of our method in maintaining identity while capturing rich details and essential attributes, such as facial features, expressions, hair color and hairstyle. Our approach not only ensures identity preservation but also preserves editability. It empowers diverse stylization, image editing, and new scene generation in accordance with the provided prompts.

### Related Work

Text-to-Image Synthesis The text-to-image synthesis relies on deep generative models such as Generative Adversarial Networks (GANs) (Xia et al. 2021; Kang et al. 2023), autoregressive models (Ramesh et al. 2021), and diffusion models (Ho, Jain, and Abbeel 2020; Rombach et al. 2022). Initially, prior works primarily focused on generating images under specific domain and text conditions, limiting their applicability. However, with the advent of large-scale image-text datasets and powerful language encoders, textto-image synthesis has achieved remarkable advancements. DALL-E (Ramesh et al. 2021) was the first approach utilizing an autoregressive model to generate diverse and intricate images from arbitrary natural language descriptions. This methodology served as the foundation for subsequent

methods like Make-A-Scene (Gafni et al. 2022), CogView (Ding et al. 2021), Parti (Yu et al. 2022), Muse (Chang et al. 2023), and others.

However, the pioneering work of GLIDE (Nichol

- et al. 2021) introduced diffusion models, surpassing the autoregressive-based DALL-E in producing more photorealistic and high-resolution images. Consequently, diffusion models have gradually emerged as the predominant approach for text-to-image synthesis. Subsequent advancements, such as DALL-E2 (Ramesh et al. 2022), Imagen (Saharia et al. 2022), and LDM (Rombach et al. 2022), have further improved diffusion models in terms of photorealism, language comprehension, and generation diversity. Notably, the release of Stable Diffusion as an open-source model has propelled its widespread adoption, making it the most extensively utilized text-to-image model. This has also led to the creation of numerous fine-tuned models by the research community. Given this context, we employ Stable Diffusion in our experiments.

Image Inversion In the domain of generative models, the ability to invert an image into a latent code that accurately reconstructs the original image holds significant importance. This capability facilitates various downstream applications that rely on manipulating the latent code to enable tasks such as image editing, translation, customization, and overall control over image generation. In the literature on GAN inversion, there are primarily two approaches: optimizationbased inversion, involves directly optimizing the latent code to minimize image reconstruction error. While this method achieves high fidelity, its drawback lies in the requirement of a hundred times iterations, rendering it unsuitable for realtime applications. Encoder-based inversion, trains a neural network to predict the latent code. Once trained, the encoder performs a single forward pass to obtain a generalized result, offering improved efficiency.

The inversion of diffusion models also draws inspiration from the aforementioned methods. However, due to the iterative nature of diffusion model-based image generation, inverting an image into the noise space associated with the image (e.g., DDIM (Song, Meng, and Ermon 2020), DDIB (Su

- et al. 2022), ILVR (Choi et al. 2021), CycleDiffusion (Wu and De la Torre 2022)) results in a latent space that is not as decoupled and smooth as the latent space of GANs. Consequently, identity preservation suffers. Alternatively, some works explore performing inversion in a different latent space, such as textual embedding space (Gal et al. 2022). This space exhibits strong semantic information and better preserves the object’s characteristics. In our research, we employ an encoder-based approach to achieve instant inversion in the text embedding space. And the method is further extended to conditioning on visual features, which can quickly capture the image in multi-modality, and realize fast generation.

Personalization By leveraging the generative capabilities of pre-trained text-to-image models, Personalization offers users the ability to synthesize specific unseen concepts within new scenes using reference images. Typically, the unseen concept is transformed into a pseudo word

(e.g., S∗) within the textual embedding space (Ruiz et al. 2023a; Gal et al. 2022). Optimization-based methods, such as DreamArtist (Dong, Wei, and Lin 2022), directly optimize the pseudo word to establish a mapping between userprovided images and textual inversion. Other approaches, e.g., Dreambooth (Ruiz et al. 2023a) and CustomDiffusion (Kumari et al. 2023) employ fine-tuning of text-to-image models to enhance fidelity. However, these strategies require minutes to hours for concept-specific optimization. In contrast, encoder-based methods such as E4T (Gal et al. 2023a), InstantBooth (Shi et al. 2023), Profusion (Zhou et al. 2023) train an encoder to predict the pseudo word, enabling the generation of personalized images within a few fine-tuning steps.

Nonetheless, tuning the entire model entails substantial storage costs and often results in overfitting on specific concepts. Moreover, many approaches (Gal et al. 2022; Ruiz et al. 2023a; Kumari et al. 2023; Shi et al. 2023; Chen et al. 2023) rely on multiple reference images, which is not always the case users could provide. To address these limitations, our work mitigates these imperfections through the utilization of the parameter-efficient fine-tuning technique and the design of an encoder that can perform the personalization task with only one reference image, enhancing the efficiency and effectiveness of the synthesis process.

### Methodology

The objective of personalized text-to-image synthesis is to train models to learn specific concepts through reference images and subsequently generate new images based on text prompts. In our paper, we aim to achieve instantaneous and optimization-free personalization using only a single reference image. To accomplish this, we propose a novel approach for image customization by integrating a dual-branch conditioning mechanism in both the textual and visual domains. This involves designing adapters to project the reference image into a pseudo word and image feature that accurately represents the concept. These concepts are then injected into the text-to-image model to enhance the fidelity of the generated personalized appearance.

To enable this process, we incorporate the original textto-image model with concept conditions and train it within a concept scope, supplemented by an additional face identity loss. A high-level overview of our proposed method is depicted in Figure 2.

#### Preliminary

We utilize Stable Diffusion (Rombach et al. 2022) as our base text-to-image model, which has demonstrated stability and reliability in the field. This model is trained on a large-scale dataset of image-text pairs, enabling it to generate high-quality and diverse images that align with the given text prompts. Stable Diffusion, based on the Latent Diffusion Model (LDM) architecture, comprises two crucial components.

Firstly, an autoencoder (E,D) is trained on a large-scale image dataset to compress images. The encoder E maps an image x from the pixel space to a low-dimensional latent

[Figure 2]

Figure 2: Overview of our proposed PhotoVerse.

space z = E(x). The decoder D is responsible for reconstructing the latent representation z back into an image with high fidelity, aiming to achieve D(E(x)) ≈ x.

Secondly, a denoising network E, utilizing the UNet (Ronneberger, Fischer, and Brox 2015) architecture as the backbone, is trained to perform the diffusion process in the latent space. This approach offers computational and memory advantages compared to operating directly in the pixel space. Additionally, the diffusion model incorporates a text condition y for text-to-image synthesis. It employs a CLIP text encoder cθ to project the condition y into an intermediate representation cθ(y). This representation is then employed in the intermediate layers of the UNet through a cross-attention mechanism:

QKT √

d′ )V, (1)

Attn(Q,K,V ) = Softmax(

where Q = WQ ·φ(zt),K = WK ·cθ(y),V = WV ·cθ(y), φ(zt) is the hidden states inside Unet, zt is the latent noised to time t, d′ corresponds to the scale factor utilized for attention mechanisms. The training objective of the latent diffusion model is to predict the noise that is added to the latent of the image, which is formulated as:

Ldiffusion = Ez∼E(x),y,ϵ∼N(0,I),t ∥ϵ − ϵθ (zt,t,cθ(y))∥22 ,

(2) here ϵ represents the unscaled noise sample, and E denotes the denoising network. During inference, a random Gaussian noise zT is sampled, and through a series of T iterative

denoising steps, it is transformed into z0′ . Subsequently, the decoder D is employed to generate the final image, given by

x′ = D(z0′ ).

#### Dual-branch Concept Extraction

Prior to extracting facial features, it is essential to preprocess the input images. In the preprocessing stage, Firstly, a face detection algorithm was applied to identify and localize human faces within the input image x. This step ensured that the subsequent analysis focused solely on facial regions of interest. Additionally, to provide a slightly larger region for feature extraction, the bounding boxes around the detected faces were expanded by a scaling factor i.e., 1.3. This expansion ensured that important facial details were captured within the region of interest. Subsequently, the expanded regions are cropped as a new face image xm for subsequent feature extraction. Furthermore, to meet the specific image generation requirements of the diffusion model, the expanded facial regions were resized to the desired shape. Moreover, to remove non-facial elements and focus solely on the facial attributes, a mask was applied to the resized facial regions. This mask effectively masked out irrelevant areas such as background and accessories, ensuring that subsequent identity feature extraction conduct efficiently. For the denoising image during training of the Diffusion model, we also employ the cropped face image with a scaling factor of 3 as xt.

Textual Condition. To personalize a specific concept that cannot be adequately described by existing words, we adopt the approach proposed in (Gal et al. 2022; Wei et al. 2023) to embed the reference image into the textual word embedding space. In this process, we summarize the concept using pseudo-words denoted as S∗. Initially, we utilize the CLIP image encoder, the same encoder employed in Stable Diffusion, to extract the features of the reference image. Following the technique described in (Wei et al. 2023), we enhance the representational capability of the image tokens as well as the editability of the model by selecting features after m layers from the CLIP image encoder, which capture spatial and semantic information at various hierarchical levels of abstraction and concreteness. Subsequently, a multiadapter architecture is employed to translate the image features from each layer into multi-word embeddings, resulting in S∗ = S1,...,Sm. Since CLIP effectively aligns textual embeddings and image features, each text adapter consists of only two MLP layers with non-linear activations, making it lightweight and enabling fast representation translation. This design choice leverages the existing alignment provided by CLIP, ensuring efficient and accurate transformation of image features into textual embeddings.

Visual Condition. Despite the advantages of condition in the textual embedding space, there are certain limitations to consider. For instance, the performance can be influenced by the encoder ability of the following text encoder, and the semantic information in the text space tends to be abstract which leads to higher requirements for token representation capabilities. Consequently, we propose the incorporation of the condition from the image space as an auxiliary aid, which is more specific for the model to understand new

[Figure 3]

Figure 3: Concept injection mechanism in the crossattention module of UNet.

concepts and contributes to the effective learning of pseudotokens in the text space. To accomplish this, we utilize the feature obtained from the CLIP image encoder. These features are then mapped using an image adapter, which follows the same structural design as the text adapter. The resulting feature capture essential visual cues related to identity, enabling a more accurate representation of the desired attributes during image generation.

#### Dual-branch Concept Injection

The original diffusion model might lack the necessary features required for constructing novel concepts. Thus, we propose injecting the new concept into the text-to-image synthesis process by fine-tuning the diffusion model to recover personal appearance concepts with a high level of detail. Rather than fine-tuning the entire UNet, which can be computationally expensive and potentially reduce model editability due to overfitting, we only add conditions and fine-tune the weights in the cross-attention module. Previous studies, such as E4T (Gal et al. 2023a) and Custom Diffusion (Kumari et al. 2023) have also highlighted that the most expressive model parameters reside in attention layers.

As shown in Figure 3, for the textual conditioning branch, we incorporate the Parameter-Efficient Fine-Tuning (PEFT) method i.e., LoRA (Hu et al. 2021), a more efficient tuning approach. LoRA was originally designed to adapt largescale pre-trained natural language models to new downstream tasks or domains while preserving performance. LoRA works by freezing the pre-trained model weights and introducing new trainable rank decomposition matrices into each layer of the model architecture. Consequently, we have KT = WkTp + α∆WkTp and V T = WvTp + α∆WvTp, where p represents the text feature extracted from the text

encoder, ∆WkT and ∆WvT denote the low-rank decomposition ∆W = BA, with A and B containing trainable parameters. Regarding the visual conditioning branch, KS = WkSf and V S = WvSf, where f corresponds to the image representation obtained after passing through the image adapter.

Then we present a random fusion strategy for the multimodality branches:

O = γAttn(Q,KT,V T) + σAttn(Q,KS,V S), (3)

where γ and σ denote two scale factors that regulate the influence of control. A random seed is sampled from the uniform distribution U = (0,1), the fused representation can be obtained in the following manner:

 

γAttn(Q,KT,V T), if seed < r1;γ = 2 σAttn(Q,KS,V S), if seed > r2;σ = 2 γAttn(Q,KT,V T)+ σAttn(Q,KS,V S), otherwise ;γ = σ = 1

O =



(4) where r1 and r2 is the threshold of random seed.

In addition, we introduce regularization for both persuade token embedding after text adapter pf and reference image values V S. Specifically,

LTreg = Mean∥pf∥1, (5) and

LSreg = Mean∥V S∥1. (6)

The whole pipeline can be trained as usual LDM (Rombach et al. 2022) does, except for additional facial identity preserving loss Lface:

Lface = C(f(x) − f(x′)). (7)

Here, f(·) represents a domain-specific feature extractor, x denotes the reference image, and x′ corresponds to the denoised image with the prompt “a photo of S∗”. To achieve the goal of measuring identity in human face scenarios, we employ the Arcface face recognition approach (Deng et al. 2019). The function C(·) computes the cosine similarity of features extracted from the face region in the images. Subsequently, the total training loss is formulated as:

Ltotal = Ldiffusion +λfaceLface +λrtLTreg +λrvLSreg, (8)

where λface is the scale factor of Lface, while λrt and λrv are hyperparameters that control the sparsity level for the specific feature.

Overall, the lightweight adapters and UNet are jointly trained on public datasets within a concept scope, such as (Karras, Laine, and Aila 2019). In this way, PhotoVerse can learn how to recognize novel concepts. At inference time, fine-tuning the model is obviated, thereby enabling fast personalization based on user-provided textual prompts.

### Experiments

#### Experimental settings

Datasets. We fine-tune our model on three public datasets Fairface (K¨arkk¨ainen and Joo 2019), CelebA-HQ (Karras

et al. 2017) and FFHQ (Karras, Laine, and Aila 2019). Evaluation is conducted on a self-collected dataset, which contains 326 images with balanced racial and gender samples for 4 colors: White, Black, Brown, and Yellow from all the race groups.

For the evaluation phase, we generate five images for each photo with the default inversion prompt “a photo of S∗”. This allows us to robustly measure the performances and generalization capabilities of our method.

Implementation Details. For the text-to-image diffusion model, we utilize Stable Diffusion (Rombach et al. 2022) as the base model and fine-tune adapters, loRA as well as visual attention weights. We pre-train the model with learning rate of 1e − 4 and batch size of 64 on V100 GPU. Our model was trained for 60,000 steps. During training, α = 1, λface = 0.01, λrt = 0.01, λrv = 0.001, m = 5, r1 = 31, r2 = 32. For evaluation, we set m = 1, α = 1 during sampling. In addition, one can adjust parameters m, σ, γ, and α with more flexibility to achieve desired generation objectives, such as preserving more identity for photo editing or incorporating stronger style effects for the Ghibli anime style. By customizing these parameters, one can set the level of ID information retention and the degree of stylization according to specific preferences and application requirements. Besides, one can also integrate the image-toimage trick during sampling to reduce randomness, which means replacing initial noise zT with zt and sampling from time step t, here zt is derived from the noisy image of x0.

Evaluation Metrics. In this paper, we aim to assess the ability of generated results to preserve identity by employing facial ID similarity as an evaluation metric. Specifically, we utilize a facial recognition model i.e., Arcface (Deng et al. 2019) to compute the cosine similarity between facial features. The ID similarity serves as a quantitative measure to determine the extent to which the generated outputs retain the original identity information.

#### Quantitative Results

As indicated in Table 1, we conducted an evaluation to assess the facial identity similarity in generated facial images. The evaluation involved measuring the ID similarity across diverse ethnicities by employing cosine similarity on facial features. The obtained results substantiate the efficacy of our approach, which achieves a notable level of similarity, with an average value of 0.696. Remarkably, certain ethnicities, including brown and white, exhibit similarity scores surpassing 0.7, with white ethnicity demonstrating the highest similarity. We posit that this discrepancy could be attributed to the inherent bias of the pre-trained large model. Regardless, it is noteworthy that our method demonstrates consistent and robust performance across all ethnicities, with a marginal deviation of approximately 0.03 observed among different ethnic groups.

#### Qualitative Results

Within Figure 4, we present a comprehensive comparison of our proposed PhotoVerse method alongside state-of-theart (SOTA) personalization techniques, namely DreamBooth

[Figure 4]

- Figure 4: Comparison with State-of-the-Art methods. Our proposed PhotoVerse shows superior performance in preserving identity attributes and generating high-quality images. Notice that DreamBooth, Textual Inversion, E4T and ProFusion require an additional stage of fine-tuning on the provided reference image prior to generation. In contrast, our PhotoVerse is tuning-free and boasts rapid generation speed, offering a distinct advantage in terms of efficiency and user convenience.

(Ruiz et al. 2023a), Textual Inversion (Gal et al. 2022), E4T (Gal et al. 2023b), and ProFusion (Zhou et al. 2023), focusing on qualitative results. The results of corresponding methods are directly taken from (Zhou et al. 2023). Notably, all aforementioned methods require test time tuning. For instance, DreamBooth and Textual Inversion necessitate 3-5 photos per subject, which incurs considerable time and storage requirements for personalization. DreamBooth, in particular, demands approximately 5 minutes, while Textual Inversion entails 5000 steps for fine-tuning. E4T and ProFusion allow customization with just 1 reference image, but they still require additional time for fine-tuning, approximately 30 seconds (Zhou et al. 2023). In contrast, our proposed approach is test time tuning-free, enabling the synthesis of 5 images within a mere 25 seconds. This remarkable efficiency makes our method exceedingly user-friendly, significantly enhancing the user experience.

Furthermore, concerning the preservation of identity attributes in the reference image, our PhotoVerse (shown in the last column) exhibits exceptional proficiency in capturing facial identity information. Our approach successfully retains crucial subject features, including facial features, expressions, hair color, and hairstyle. For instance, when compared to alternative techniques, our proposed method outperforms in restoring intricate hair details while effectively preserving facial features, as evident in the first row. Additionally, as observed in the second row, our approach excels at fully restoring facial features while maintaining consistency with the desired “Manga” style specified in the prompt. In contrast, the Profusion-generated photo exhibits blurry mouth details, while E4T fails to exhibit a sufficiently pronounced “Manga” style. Shifting to the third row, our results successfully capture characteristic expressions present in the input images, such as frowning.

[Figure 5]

- Figure 5: Results of our PhotoVerse with various prompts for stylization and new scene generation.

Regarding the quality of generated images, our method surpasses other works by producing noticeably sharper images with enhanced detail. Moreover, our approach exhibits a high degree of aesthetic appeal, featuring natural lighting, natural colors, and the absence of artifacts.

Figure 5 presents additional results showcasing the performance of our approach across different ethnicities in image editing, image stylization, and novel scene generation, further substantiating the three aforementioned aspects of superiority.

#### Ablation studies

Effect of Visual conditioning branch According to Table 1, it is evident that the image branch has a substantial impact

|Method|Black Brown White Yellow All<br><br>|
|---|---|
|w/o visual conditioning branch w/o LSreg w/o Lface w/o LTreg|0.561 0.563 0.584 0.556 0.556 0.566 0.573 0.589 0.550 0.569 0.632 0.658 0.663 0.622 0.643 0.650 0.668 0.678 0.657 0.663<br><br>|
|PhotoVerse<br><br>|0.685 0.702 0.715 0.682 0.696|

Table 1: Ablation study results of 4 components on ID similarity of 4 races.

on the preservation of ID information. Removing the image branch leads to a loss of 0.14, indicating its crucial role in the model’s ability to maintain identity consistency. This effect can be attributed to the provision of specific, detailed, and spatial conditions by the visual branch during image generation. Moreover, the visual branch contributes positively to optimizing the embedding of textual tokens.

Effect of Regularizations The experimental results illustrate the importance of regularizations for visual values and textual facial embeddings during concept injection. It can promote the sparsity of representations, thereby retaining key values and mitigating overfitting issues, as well as enhancing the generalization capability of our model.

Effect of Face ID loss We also evaluated the impact of face ID loss on the preservation of identity information. The experimental results demonstrate that it also plays an important role in maintaining identity, resulting in an improvement of 0.05 in performance.

### Conclusions

In this paper, we introduced an innovative methodology that incorporates a dual-branch conditioning mechanism in both the text and image domains. This approach provides effective control over the image generation process. Additionally, we introduced facial identity loss as a novel component to enhance identity preservation during training. Remarkably, our proposed PhotoVerse eliminates the need for test-time tuning and relies solely on a single facial photo of the target identity. This significantly reduces the resource costs associated with image generation. Following a single training phase, our approach enables the generation of high-quality images within just a few seconds. Moreover, our method excels in producing diverse images encompassing various scenes and styles. Our approach also supports incorporating other methods such as ControlNet (Zhang and Agrawala 2023), specifically leveraging its control branch for preserving the overall high-level structure, further enhancing the pose control of personalized text-to-image generation. Through extensive evaluation, we have demonstrated the superior performance of our approach in achieving the dual objectives of preserving identity and facilitating editability. The results highlight the potential and effectiveness of PhotoVerse as a promising solution for personalized textto-image generation, addressing the limitations of existing methods and paving the way for enhanced user experiences in this domain.

### References

Chang, H.; Zhang, H.; Barber, J.; Maschinot, A.; Lezama, J.; Jiang, L.; Yang, M.-H.; Murphy, K.; Freeman, W. T.; Rubinstein, M.; et al. 2023. Muse: Text-to-image generation via masked generative transformers. arXiv preprint

- arXiv:2301.00704.

Chen, W.; Hu, H.; Li, Y.; Rui, N.; Jia, X.; Chang, M.W.; and Cohen, W. W. 2023. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint

- arXiv:2304.00186.

Choi, J.; Kim, S.; Jeong, Y.; Gwon, Y.; and Yoon, S. 2021. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938.

Deng, J.; Guo, J.; Xue, N.; and Zafeiriou, S. 2019. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4690–4699.

Ding, M.; Yang, Z.; Hong, W.; Zheng, W.; Zhou, C.; Yin, D.; Lin, J.; Zou, X.; Shao, Z.; Yang, H.; et al. 2021. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34: 19822–19835.

Dong, Z.; Wei, P.; and Lin, L. 2022. Dreamartist: Towards controllable one-shot text-to-image generation via contrastive prompt-tuning. arXiv preprint arXiv:2211.11337.

Gafni, O.; Polyak, A.; Ashual, O.; Sheynin, S.; Parikh, D.; and Taigman, Y. 2022. Make-a-scene: Scene-based text-toimage generation with human priors. In European Conference on Computer Vision, 89–106. Springer.

Gal, R.; Alaluf, Y.; Atzmon, Y.; Patashnik, O.; Bermano, A. H.; Chechik, G.; and Cohen-Or, D. 2022. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618.

Gal, R.; Arar, M.; Atzmon, Y.; Bermano, A. H.; Chechik,

- G.; and Cohen-Or, D. 2023a. Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models. ACM Transactions on Graphics (TOG), 42(4): 1–13. Gal, R.; Arar, M.; Atzmon, Y.; Bermano, A. H.; Chechik,
- G.; and Cohen-Or, D. 2023b. Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models. ACM Transactions on Graphics (TOG), 42(4): 1–13.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Kang, M.; Zhu, J.-Y.; Zhang, R.; Park, J.; Shechtman, E.; Paris, S.; and Park, T. 2023. Scaling up gans for text-toimage synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10124– 10134.

K¨arkk¨ainen, K.; and Joo, J. 2019. Fairface: Face attribute dataset for balanced race, gender, and age. arXiv preprint arXiv:1908.04913.

Karras, T.; Aila, T.; Laine, S.; and Lehtinen, J. 2017. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

Karras, T.; Laine, S.; and Aila, T. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4401–4410.

Kumari, N.; Zhang, B.; Zhang, R.; Shechtman, E.; and Zhu, J.-Y. 2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1931–1941.

Ma, J.; Liang, J.; Chen, C.; and Lu, H. 2023. SubjectDiffusion: Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning. arXiv preprint arXiv:2307.11410.

Nichol, A.; Dhariwal, P.; Ramesh, A.; Shyam, P.; Mishkin, P.; McGrew, B.; Sutskever, I.; and Chen, M. 2021. Glide: Towards photorealistic image generation and editing with textguided diffusion models. arXiv preprint arXiv:2112.10741.

Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot text-toimage generation. In International Conference on Machine Learning, 8821–8831. PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, 234–241. Springer.

Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023a. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22500–22510.

Ruiz, N.; Li, Y.; Jampani, V.; Wei, W.; Hou, T.; Pritch, Y.; Wadhwa, N.; Rubinstein, M.; and Aberman, K. 2023b. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494.

Shi, J.; Xiong, W.; Lin, Z.; and Jung, H. J. 2023. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411.

Sohn, K.; Ruiz, N.; Lee, K.; Chin, D. C.; Blok, I.; Chang, H.; Barber, J.; Jiang, L.; Entis, G.; Li, Y.; et al. 2023. StyleDrop: Text-to-Image Generation in Any Style. arXiv preprint arXiv:2306.00983.

Song, J.; Meng, C.; and Ermon, S. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.

Su, X.; Song, J.; Meng, C.; and Ermon, S. 2022. Dual diffusion implicit bridges for image-to-image translation. arXiv preprint arXiv:2203.08382.

Wei, Y.; Zhang, Y.; Ji, Z.; Bai, J.; Zhang, L.; and Zuo,

- W. 2023. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848.

Wu, C. H.; and De la Torre, F. 2022. Unifying Diffusion Models’ Latent Space, with Applications to CycleDiffusion and Guidance. arXiv preprint arXiv:2210.05559.

Xia, W.; Yang, Y.; Xue, J.-H.; and Wu, B. 2021. Tedigan: Text-guided diverse face image generation and manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2256–2265.

Yu, J.; Xu, Y.; Koh, J. Y.; Luong, T.; Baid, G.; Wang, Z.; Vasudevan, V.; Ku, A.; Yang, Y.; Ayan, B. K.; et al. 2022. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3): 5.

Yuan, G.; Cun, X.; Zhang, Y.; Li, M.; Qi, C.; Wang,

- X.; Shan, Y.; and Zheng, H. 2023. Inserting Anybody in Diffusion Models via Celeb Basis. arXiv preprint arXiv:2306.00926.

Zhang, L.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. arXiv preprint

- arXiv:2302.05543.

Zhou, Y.; Zhang, R.; Sun, T.; and Xu, J. 2023. Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach. arXiv preprint

- arXiv:2305.13579.

## Supplementary Material

### A More Qualitative Results

In Figure 6, 9, 10, 11, 12 and 13, we present supplemental comparison results involving state-of-the-art methods E4T (Gal et al. 2023a), InsertingAnybody (Yuan et al. 2023), Profusion (Zhou et al. 2023), HyperDreamBooth (Ruiz et al. 2023b) and Subject-Diffusion (Ma et al. 2023), respectively. All the results are taken from corresponding papers. These results show the exceptional performance of our PhotoVerse in terms of fidelity, editability, and image quality. Notably, our approach stands apart from methods E4T (Gal et al. 2023a), InsertingAnybody (Yuan et al. 2023), Profusion (Zhou et al. 2023) and HyperDreamBooth (Ruiz et al. 2023b) by eliminating the need for test-time tuning, while maintaining the ability to generate a single image in ∼ 5 seconds. We also provide more results of our PhotoVerse in Figure 14, 15 and 16.

[Figure 6]

Figure 6: Comparison results with E4T.

### B Experiment Details

Adapter Design In our proposed approach, the adapter module is designed to be both straightforward and computationally efficient. It comprises two blocks, with each block consisting of two MLP layers that effectively project the input feature into a 1024-dimensional space. Following this projection, layer normalization and a leaky-ReLU activation layer are applied to further enhance the adapter’s performance.

User Study Follow ProFusion (Zhou et al. 2023), we also evaluated E4T (Gal et al. 2023a), ProFusion (Zhou et al. 2023), and our method using human assessment. The results for each method were obtained from the corresponding papers. We designed two separate questionnaires for E4T, ProFusion, and our PhotoVerse for comparison. Each questionnaire consisted of 25 questions. A total of 15 participants took part in the assessment, ensuring a diverse range of perspectives.

[Figure 7]

- Figure 7: Results of user study on E4T and our PhotoVerse.

[Figure 8]

- Figure 8: Results of user study on Profusion and our PhotoVerse.

### C User Study

As shown in Figure 7 and 8, we conduct a human evaluation of two recent state-of-the-art methods E4T, Profusion and our PhotoVerse. For each of the two methods, we requested the participants to provide their preferences among two images generated using different methods, along with the original image and textual instructions. The participants were tasked with indicating their preferred choice according to the subject fidelity and text fidelity. According to Figure 7 and 8, our method outperformed the other two recent methods, which are 66% VS 34% and 78% VS 22% respectively.

[Figure 9]

###### Figure 9: Comparison results with InsertingAnybody.

[Figure 10]

###### Figure 10: Comparison results with ProFusion.

[Figure 11]

###### Figure 11: Comparison results with HyperDreamBooth.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

