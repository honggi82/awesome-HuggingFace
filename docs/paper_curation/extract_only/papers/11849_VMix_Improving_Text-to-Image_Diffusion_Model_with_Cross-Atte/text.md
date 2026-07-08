## VMix: Improving Text-to-Image Diffusion Model with Cross-Attention Mixing Control

Shaojin Wu1, Fei Ding1*, Mengqi Huang1,2, Wei Liu1, Qian He1 1ByteDance Inc, 2University of Science and Technology of China

{wushaojin, dingfei.212, liuwei.jikun, heqian}@bytedance.com {huangmq}@mail.ustc.edu.cn

# arXiv:2412.20800v1[cs.CV]30Dec2024

### Abstract

SDXL SDXL-DPO VMix(Ours)

[Figure 1]

[Figure 2]

[Figure 3]

While diffusion models show extraordinary talents in textto-image generation, they may still fail to generate highly aesthetic images. More specifically, there is still a gap between the generated images and the real-world aesthetic images in finer-grained dimensions including color, lighting, composition, etc. In this paper, we propose CrossAttention Value Mixing Control (VMix) Adapter, a plugand-play aesthetics adapter, to upgrade the quality of generated images while maintaining generality across visual concepts by (1) disentangling the input text prompt into the content description and aesthetic description by the initialization of aesthetic embedding, and (2) integrating aesthetic conditions into the denoising process through valuemixed cross-attention, with the network connected by zeroinitialized linear layers. Our key insight is to enhance the aesthetic presentation of existing diffusion models by designing a superior condition control method, all while preserving the image-text alignment. Through our meticulous design, VMix is flexible enough to be applied to community models for better visual performance without retraining. To validate the effectiveness of our method, we conducted extensive experiments, showing that VMix outperforms other state-of-the-art methods and is compatible with other community modules (e.g., LoRA, ControlNet, and IPAdapter) for image generation. The project page is https://vmix-diffusion.github.io/VMix/.

yellow hair white dress red flower

yellow hair white dress red flower

yellow hair white dress red flower

text fidelity

skin detail coherent color natural light

skin detail coherent color natural light

skin detail coherent color natural light

visual aesthetics

“A yellow-haired girl in a white dress holds a red flower in her hand, overhead shot”

Figure 1. Comparison of text fidelity and visual aesthetics between SDXL [15], DPO [27], and our VMix. DPO can generate attributes that SDXL fails to produce, but it fails to align with human visual fine-grained preferences. Our method achieves better text fidelity and visual aesthetics simultaneously.

realism and textual alignment, the current results still exhibit significant gaps from human expectations in various aspects, such as generated images with unnatural lighting, distorted human bodies, or supersaturated colors. These misalignment with human expectation is fatal for the realworld applications of AI-generated content such as film production, since human beings are highly sensitive to these “devils in the details”. Therefore, the key challenge has become how to accurately align the generated images with human preference across various aspects.

Existing works have made considerable efforts to improve image quality to meet human preferences, which can be primarily categorized into two streams. The first stream focuses on fine-tuning the pre-trained textto-image models either based on an exceptionally highquality sub-dataset[3], or based on reward models via reinforcement learning[9, 11, 30] and direct preference optimization(DPO)[27]. The latter stream[7, 24] instead focuses on investigating the generation behavior of pretrained diffusion models itself to improve its generation sta-

### 1. Introduction

The past few years have witnessed the flourish in the field of text-to-image generation, especially the advent of largescale pretrained text-to-image diffusion models[5, 14, 15, 19, 20], allowing human to create visually compelling images using text prompts conveniently. Although these large models can produce overall high-quality images in visual

*Corresponding author.

bility. For example, FreeU[24] proposed re-weighting the contributions sourced from skip connections and backbones in the denoising model to strengthen the denoising ability while simultaneously enhancing details. In summary, existing methods align the generation results with human preference by improving the overall generation quality in terms of visual realism or textual consistency.

However, in this study, we argue that existing methods fail to align fine-grained human preference for visually generated content. Images favored by human beings should excel across various fine-grained aesthetic dimensions simultaneously, such as natural light, coherent color, and reasonable composition. On the one hand, these fine-grained aesthetic demands can not be simply addressed by augmenting detailed textual descriptions for pretrained diffusion models to understand. The reason is that their text encoders (e.g., CLIP[16] or T5[17]) are primarily trained for capturing high-level semantics and lacking the accurate awareness for these ineffable visual aesthetics. On the other hand, the optimization direction for overall image generation quality is neither equivalent to nor consistent with the direction for these fine-grained aesthetic dimensions. For instance, while overall better-generated results may exhibit greater textual alignment, they might suffer from poorer visual composition, as depicted in Fig. 1.

To address this challenge, we introduce VMix, a novel plug-and-play adapter designed to systematically bridge the aesthetic quality gap between generated images and realworld counterparts across various aesthetic dimensions. We finetune the adapter on a hand-selected subset of exceptionally high-quality images derived from a large corpus. Inspired by universal photography standards, which encompass aspects like color, lighting, composition, and focus [3], we label these images across various aesthetic dimensions. During training, we freeze the base model and employ the LoRA[8] method to ensure practical applicability. We further design two specialized modules to incorporate these aesthetic labels as additional conditions into the U-Net [21] architecture. The first, termed the aesthetic embedding initialization module, pre-processes the aesthetic textual data, initializing it into embeddings that align with the corresponding images. This step is essential only at the commencement of training. Once training begins, we map the aesthetic labels of various images to embeddings by referencing the initial results. To better integrate this embedding into the U-Net, we introduce the second module, the crossattention mixing control module, which aims to minimize adverse effects on image-text alignment without directly altering the attention maps. Our extensive experiments demonstrate that VMix can be seamlessly integrated with various base models, significantly enhancing their aesthetic performance. Moreover, VMix exhibits excellent compatibility with community modules (i.e., ControlNet[36], IP-

Adapter[32], and LoRA[8]), thereby providing the community with greater creative capabilities.

In summary, our main contributions are:

- • We analyze and explore the differences in generated images across fine-grained aesthetic dimensions, proposing the disentanglement of these attributes in the text prompt to provide a clear direction for model optimization.
- • We introduce VMix, which disentangles the input text prompt into content description and aesthetic description, offering improved guidance to the model via a novel condition control method called value-mixed cross-attention.
- • The proposed VMix approach is universally effective for existing diffusion models, serving as a plug-and-play aesthetic adapter that is highly compatible with community modules.

### 2. Related Work

###### 2.1. Text-to-Image Models

The method of generating images from given textual descriptions has been extensively explored. GAN-based works have demonstrated impressive capabilities in producing realistic images [26, 31, 35]. Generative transformer methods usually train large-scale autoregressive transformers in a discrete token space to model the generation process [18, 33, 34]. More recently, diffusion models have been applied in text-to-image tasks, achieving state-of-theart results in generating high-fidelity images [4, 15, 20]. Given a text prompt, the model typically converts the text into a latent vector with the help of a pretrained language model [16], and then generates images from pure Gaussian noise through a iterative denoising process. Although these models have demonstrated remarkable capabilities in image generation, they struggle to ensure that the generated images perform well across multiple aesthetic dimensions.

###### 2.2. Improving Text-to-Image Models

Despite the significant breakthroughs in text-to-image diffusion models, numerous challenges persist when it comes to simultaneously ensuring text fidelity and visual aesthetics. Researchers are approaching these challenges from various angles to find solutions. Emu [3] highlights the importance of high-quality data, demonstrating that models fine-tuned on such data can achieve further improvements in their generated results. FreeU [24] enhances image generation quality by adjusting the connection weights at different levels of the U-Net, without requiring additional training. DPO [27] optimizes the denoising process by generating images that are more aligned with human preferences compared to less favored images. Unlike these approaches, we propose a new conditional control method that aligns with human aesthetics across fine-grained dimensions while retaining the original model’s semantic understanding capabilities.

###### (a) Initialization Stage (b) Training Stage (c) Inference Stage

Projection Layer

[Figure 4]

Linear

Color Light

Color Light Composition

Color Light

Projection Layer

LN Zero linear

… 𝒇𝒕 𝒇𝒂

𝒇 𝒂

Composition …

Composition

…

Aesthetic labels

Aesthetic input 𝒚𝒂𝒆𝒔

Aesthetic input 𝒚𝒂𝒆𝒔

[Figure 5]

[Figure 6]

×𝜆 ×𝜆

Q 𝐾 𝑉

Q 𝐾 𝑉

Text Model

Self-Attention

Self-Attention

[Figure 7]

[Figure 8]

[Figure 9]

Diffusion Model

Copy

Copy

[cls] token

Q 𝐾 𝑉

Q 𝐾 𝑉

…

…

𝒙𝒕

𝒙𝒕

VMix Cross-Attention

VMix Cross-Attention

Diffusion Model

𝒇𝒂𝒆𝒔

AesEmb

“Photograph of a silhouetted sailboat with two masts against a vibrant sunset over a calm ocean”

“Digital painting of a vibrant, surreal tree against a

Text Model

Text Model

𝒇𝒄

bright, dreamy sky.”

Superior Inferior

Content input 𝒚

Content input 𝒚

- Figure 2. Illustration of of VMix. (a)In the initialization stage, pre-defined aesthetic labels are transformed into [CLS] tokens through CLIP, thereby obtaining AesEmb, which only need to be processed once at the beginning of training. (b)In the training stage, a project layer first maps the input aesthetic description yaes into an embedding fa of the same token dimension as the content text embedding ft. The text embedding ft is then integrated into the denoising network through value-mixed cross-attention. (c)In the inference stage, VMix extract all positive aesthetic embedding from AesEmb to form the aesthetic input, along with the content input, is fed into the model for the denoising process.

- 2.3. Controlling Text-to-Image Models

Text-to-image models can generate results that match specific tasks or personalized content or styles by incorporating additional controlling conditions [13, 28]. Diverse conditional control approaches typically vary in the specific conditional features they introduce or the distinct points at which these conditions are injected into the process. ControlNet [36] integrates additional features into the decoder of the U-Net architecture to learn task-specific input conditions, such as pose, edges, sketch, and depth etc. IPAdapter [32] propose a unique decoupled cross-attention design for controllable image generation. Differing from these approaches, our method does not rely on reference images; instead, it disentangles the input text prompt into content description and aesthetic description and introduces distinctive improvements to the cross-attention layers.

- 3. Methodology

will serve as the input condition for LDM. During training, given a latent noise zt at time step t and condition cθ(y), the denoising network ϵθ(·) aims to predict noise ϵ. This learning process is facilitated by minimizing the following loss function:

L = Ez∼E(I),y,ϵ∼N(0,1),t ∥ϵ − ϵθ (zt,t,cθ(y))∥22 , (1)

The denoising network ϵθ(·) is commonly implemented by U-Net [21]. When condition cθ(y) extracted from the text model is integrated into the denoising network ϵθ(·), the cross-attention layer is needed to achieve cross-modal interaction. The process can be described as follows:

c ·cθ(y), (2)

###### Q = WQ ·x,Kc = WK

c ·cθ(y),Vc = WV

QKcT √

###### Vc, (3)

Attention(Q,Kc,Vc) = softmax

d

where x is the spatial feature extracted from latent noise z, WQ,WK,WV are learnable projection layers, and d correlates with the number of channels in x.

- 3.1. Preliminary

Since the method proposed in this paper is based on the Stable Diffusion [20] (SD), we first provide a brief overview of it. SD is a latent diffusion model (LDM) capable of transforming Gaussian noise into high-fidelity images through an iterative denoising process. LDM operates diffusion processes in a latent space, requiring an autoencoder that includes an encoder and decoder. We denote the encoder as E(·), which encodes an image I into the latent space z = E(I); similarly, the decoder is denoted as D(·), used for decoding from the latent space back to the image. Given an input text prompt y, the text encoder cθ(·) of a pre-trained CLIP [16] converts it into a text embedding cθ(y), which

###### 3.2. The Disentanglement Text Prompts

This paper aims to further enhance generation quality by integrating aesthetic knowledge across different dimensions. For most fine-tuning approaches [6, 22], the condition is solely derived from the text embedding decoded by the text model from the input text prompt, which encompasses highlevel semantic information about the crucial objects and corresponding attributes of an image. In this case, even if there are some aesthetic words in the input text prompt, after several transformer layers, the information can easily drown

in the process of self-attention with other words, resulting in a minimal cross-modal interaction contribution in U-Net and unsatisfactory performance. On the other hand, the excessive inclusion of aesthetic words, which makes the input prompt overly long, could lead to the inability to generate certain subjects within the prompt.

As illustrated in Fig. 2, to solve this problem, we initially disentangle the input text prompt of text-to-image synthesis into content and aesthetic input, with aesthetic input yaes being the fine-grained aesthetic labels we introduce, and content input y about the depiction of the main subject and associated attributes in the image. Our starting point comes from the belief that the model can disentangle style (i.e., aesthetics in this case) and content, which is well-documented in [29].

To enhance the integration of fine-grained aesthetic conditions with the denoising network, we will first introduce the initialization stage of the aesthetic embedding (AesEmb). This phase results in a preprocessed AesEmb that will be consistently utilized throughout both training and inference stages. As shown in Fig. 2(a), we denote a set of opposing aesthetic labels, where ya denotes a specific aesthetic label (e.g. vibrant color, natural lighting, proportional composition, etc.), and yˆa indicates not having that label. Notably, we use [identifier] to signify yˆa, which is a rare token acting as a unique identifier associated with the aesthetic label (e.g. [V], [S]) [22]. In this context, we employ a rare token to represent yˆa to prevent the semantic prior of the text model from leaking into the negative aesthetic labels. This pair of opposing aesthetic labels yp = {ya,yˆa} is then processed by a frozen CLIP model, yielding a pair of [CLS] tokens, denoted as tp = {tcls,tˆcls}.

In practice, more than one set of opposing aesthetic labels is required. Accordingly, we define N sets of aesthetic labels as Y = yp1,yp2,...,ypN , where ypi = {yai,yˆai} represents the ith pair of aesthetic labels. Then we get N sets of [CLS] tokens as T = t1p,t2p,...,tNp , where tip = {tia,tˆia} is the ith pair of [CLS] tokens generated by the CLIP model. We further concatenate T along the token dimension to obtain our final AesEmb:

faes = concat t1p,t2p,...,tNp ∈ R2N×d, (4)

where faes is the AesEmb and d is the feature dimension. It should be emphasized that the initialization of AesEmb requires only a single execution at the start of training and can be cached locally, making the increase in computational cost practically negligible throughout the entire training process.

###### 3.3. Cross-Attention Mixing Control

In the previous section, we disentangle the input text prompt into aesthetic input yaes and content input y, and introduce the initialization method for AesEmb. In this section, we

will further present an effective and nuanced scheme of condition control that leverages fine-grained aesthetic information to enhance the generative quality of the text-to-image model.

Efficient AesEmb Projection Layer. As is depicted in Fig. 2(b), we employ the Stable Diffusion [20] (SD) as our text-to-image model, both yaes and y serving as conditions for the model. Similar to the original SD, the y passes through the text model cθ(·) of CLIP model [16] and is decoded to obtain the text embedding fc, which can be represented by the following equation:

fc = cθ(y) ∈ RC×d, (5)

where C is the token length and d is the feature dimension. Due to the inequity of aesthetic labels in the training dataset, where each image is assigned a variable number of aesthetic labels. We consider two approaches to map aesthetic labels into textual features with the same shape as fc. The first method involves directly processing aesthetic labels through the CLIP model to obtain textual feature c(yaes), as indicated by Eq. (5). Although this approach is straightforward, it introduces certain issues. Firstly, encoding both yaes and y with the text model incurs additional computational costs. More importantly, although we treat different aesthetic dimensions as independent, the attention layers of the text model may potentially compromise this independence.

Given this, we adopt a more efficient method for condition injection. Initially, based on the aesthetic labels included in the input yaes, we index the corresponding [CLS] token from AesEmb faes ∈ R2N×d. For the ith aesthetic label, we retrieve tia if the image has this attribute, otherwise, we obtain tˆia. Thus, we can acquire a feature ft ∈ RN×d reconstituted from faes. Afterward, we use F(·) to represent the combination of a linear layer and a Layer Normalization [1], which serve to upscale the token dimension of ft from N to C. To facilitate a gentler condition injection, we also employ a zero linear layer, defined as Z(·), which is a linear layer with both weight and bias initialized to zeros. The entire projection layer is thus computed as follows:

fa = Z(F(ft)), (6)

where fa is the final textual feature projected from aesthetic labels. At the start of training, the weights and biases of the zero-initialized linear layer, which serves as the connecting layer, are set to zero. This initialization ensures that fine-tuning the model does not introduce harmful noise, thereby preserving the capabilities of the original pre-trained model [36].

Value-Mixed Cross-Attention. Directly adding aesthetic textual features to the content textual features may compromise rich semantic features, leading to decreased image-

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A banana on the left of an apple, 3dcg

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

A cat riding a motorcycle.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Two people facing the viewer, Japanese anime, semi-thick painting

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Elf, girl, blue eyes, beautiful, holy, by Hayao Miyazaki

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

An lady wearing a classical cheongsam, closeup, standing on the streets of old Shanghai, with short slightly curly hair, wearing jade earrings…

Prompt

VMix(Ours) SD FreeU DPO SFT SFT &TI

- Figure 3. Qualitative comparison with various state-of-the-art methods. All results are based on Stable Diffusion [20]. Our VMix method outperforms others, significantly enhancing the quality of image generation across various fine-grained aesthetic dimensions.

text alignment. Since the attention map within the crossattention layers dictates the probability distribution over the text tokens for each image patch, determining the principal tokens in the image patch [2], we aim to preserve this ability inherent in the pre-trained model. This approach ensures a stable enhancement of aesthetic performance while retaining image-text alignment.

To this end, we introduce our value-mixed crossattention module after each self-attention module in the diffusion U-Net model. We employ a dual-branch crossattention module, with one branch for feeding in content features fc and another for aesthetic features fa. The queries for both branches of the cross-attention module are sourced from the spatial feature x of SD, with the keys originating from the content feature fc. However, the sources for the values are distinct; we enable the model to learn a new value for the aesthetic features independently, thus reducing disruption to the original attention map as aesthetic features are fed into the model. The output of crossattention corresponding to the content description branch shares the same formula as Eq. (3), and can be expressed as

Attention(Q,Kc,Vc). The output of new cross-attention associated with the aesthetic description branch can be formulated as:

a · fa, (7)

###### Q = WQ · x,Kc = WK

c · fc,Va = WV

Attention(Q,Kc,Va) = softmax

QKcT √

d

###### Va, (8)

The cross-attention modules of these two branches share the same attention map QKcT. Therefore, we only need to add one parameter WV

for each cross-attention layer. Subsequently, we add the outputs from the content and aesthetic cross-attention layers to obtain xˆ, so the complete process of cross-attention mixing control can be represented as follows:

a

xˆ = Attention(Q,Kc,Vc) + λAttention(Q,Kc,Va),

(9) where λ is a hyperparameter and set to 1 during the training phase, xˆ is the new spatial feature and will be fed into the subsequent blocks of SD.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

A galaxycolored figurine is floating over the sea at sunset, photorealisti c

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

A panda standing on a surfboard in the ocean

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

A kangaroo wearing an orange hoodie and blue sunglasses stands on the grass in front of the Sydney Opera House, holding a sign that says Welcome Friends.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

close up headshot, steampunk, middle-aged man, slick hair big grin in front of gigantic clocktower, pencil sketch

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

two babies, one very happy the other very sad, in the middle a cute rabbit, background a garden with flowers, trees Fujifilm Superia 40

Prompt VMix(Ours) SDXL FreeU DPO SFT SFT &TI

- Figure 4. Qualitative comparison with various state-of-the-art methods. All the results of the methods are based on the SDXL [15]. Our VMix method outperforms others, significantly enhancing the quality of image generation.

###### 3.4. Training and Inference

Full-parameter training of models incurs high costs, and while this approach may achieve a higher upper limit of performance, it is inconsistent with our goal of plug-in versatility due to its high degree of customization. For this purpose, during the training phase, we freeze the parameters of the base model, training only the AesEmb Projection Layer and the newly added value in the Value-Mixed Cross-Attention. Additionally, we have incorporated LoRA [8] into some of the model’s linear layers and convolutional layers, thereby making the model training process more stable and enhancing the applicability. Upon completion, this segment of the network can be directly extracted to form a plug-and-play module, which can be used to enhancing the aesthetic potential of existing models.

During inference, in addition to the user’s prompt y, we also require the aesthetic input yaes. Unlike the training phase, where the yaes in the training data contains a varying number of positive aesthetic labels (such as ”superior light,” ”inferior color”). During inference, we default to us-

ing all positive aesthetic labels as is shown in Fig. 2(c). This approach aims to enhance the model’s generation quality across all aesthetic dimensions. Although we utilized Lora during the training phase, it is not necessary during inference. We will address this aspect in the experimental section with an ablation study.

### 4. Experiments

###### 4.1. Experiments Setting

Implementation Details. We employ AdamW [12] optimizer to train our models. The learning rates are set to 1e−4 and 1e − 5 for SD1.5 and SDXL, respectively. The batch size is set to 256, and the total number of training steps is 50,000 in the experiment. During the inference phase, we employ the DDIM sampler [25] for the sampling process, configuring it with a total of 25 timesteps and a classifierfree guidance scale set to 7.5, without the use of negative prompts.

Datasets. As previously discussed, to align the model with high-quality images across various aesthetic dimensions,

we finetune our model using a curated dataset of manually selected images. In the dataset construction phase, we prioritize image quality over quantity. Similar to [3], we initially extracted 200k images from large, publicly available English datasets such as LAION [23], employing a combination of automatic and human filtering processes. The automatic filtering included aesthetic scoring, OCR scoring, and CLIP scoring. Human filtering was conducted by individuals with a keen aesthetic sense, adhering to universal photography standards to select the finest images. Furthermore, in addition to the content description texts, we annotate these images with categorical labels across different aesthetic dimensions (such as color, lighting, composition, focus) to serve as additional conditions during our training process.

Evaluation Metrics. We assess our the performance using the MJHQ-30K dataset [10] which contains a large number of high-quality, aesthetically pleasing synthetic data. To enhance our evaluation, we created an additional benchmark, LAION-HQ10K, from the LAION [23] collection, including only high-aesthetic and high-resolution real-word images. This set quantifies the gap between our model’s generative capabilities and real-world imagery with exceptional aesthetics. For objective evaluation, we use Fr´echet Inception Distance (FID), CLIP Scores, and AES Scores1 to measure the overall quality, fidelity to the original prompts, and aesthetic excellence of the generated images.

###### 4.2. Qualitative Analyses

Compare to other methods. To validate the effectiveness of VMix, we compared our model to pre-trained model and systematically conducted further comparisons with state-ofthe-art methods such as FreeU [24], DPO [27], Textual Inversion(TI) [6], and Supervised Fine-Tuning(SFT). We further apply the well-trained VMix model to personalized models, thereby demonstrating the universality of our approach. It should be noted that, to validate the influence of the training set on the generation results, we proceed to utilize SFT and TI for training. In this configuration, the UNet model will be unfrozen, allowing all parameters to be updated. As depicted in Fig. 3 and Fig. 4, our VMix significantly outperforms other methods in visual appeal, showcasing remarkable aesthetic performance without compromising the image-text alignment capability. In our comparative analysis with Supervised Fine-Tuning (SFT), it became evident that the model struggles with datasets of exceptionally high quality. This difficulty stems from the presence of complex and abstract samples within the dataset that may surpass the model’s current capabilities, potentially leading to a decline in performance. VMix mitigates this challenge by incorporating fine-grained aesthetic supervision signals, which streamlines the learning process for the model and,

1https://github.com/christophschuhmann/improved-aesthetic-predictor

Method FID ↓ CLIP Score ↑ Aes Score ↑

SD [20] 28.08 30.24 5.35 FreeU [24] 27.09 31.00 5.36 DPO [8] 22.64 30.89 5.54 Textual Inversion [6] 24.72 28.92 5.58 SFT 24.35 30.15 5.43 VMix(Ours) 21.49 30.50 5.79

- Table 1. Quantitative results on MJHQ-30K benchmark [10]. ↑ stands for higher the better, ↓ stands for lower the better.

Method FID ↓ CLIP Score ↑ Aes Score ↑

SD [20] 25.67 32.28 5.43 FreeU [24] 28.69 32.15 5.43 DPO [8] 23.37 32.41 5.44 Textual Inversion [6] 26.62 30.97 5.53 SFT 26.27 32.27 5.40 VMix(Ours) 23.92 32.71 5.68

- Table 2. Quantitative results on LAION-HQ10K benchmark.

Method FID ↓ CLIP Score ↑ Aes Score ↑

Baseline(SD) [20] 28.08 30.24 5.35 w/o lora 21.53 30.49 5.75 w/o vmix 25.64 30.16 5.52 Ours 21.49 30.50 5.79

- Table 3. Ablation Study of lora and value-mixed cross-attention. Experiments were conducted on MJHQ-30K benchmark [10].

consequently, enhances the overall performance.

Compare to personalized models. Acting as a versatile plug-and-play adapter, VMix can be directly applied to the personalized model from Civitai2. With the integration of VMix, a notable improvement in the realism and aesthetic appeal of the generated results is expected. See Fig. 5 for qualitative results.

User study. We further conducted a user study to assess the applicability of VMix as a plug-in. For subjective assessment, 20 evaluators including both aesthetic professionals and non-professionals scored 300 distinct prompts, each yielding 4 generated images. For each case, evaluators need to select the one with the best text fidelity and visual aesthetics from the generation results of the two models. As shown in Fig. 5, the results indicate that both pre-trained and open-source models are more favored by users after the application of our VMix method.

- 4.3. Quantitative Evaluations

As shown in Tab. 1 and Tab. 2, we have secured the highest Aes Score on both the MJHQ30K and LAION-HQ10K benchmarks, which strongly demonstrates the significance of VMix in enhancing aesthetics. Our performance in CLIP Score and FID metrics is also commendable, indicating that

2https://civitai.com

###### Personalized Models Vmix(Ours) Personalized Models Vmix(Ours)

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

AnimeArtXL: Japanese anime, acrylic painting, A side view of an ballerina wearing a sparkling dress, tiptoeing across the stage during a summer dance performance

RealisticVision: Apocalyptic wasteland style with a leader dressed in futuristic clothing with silver hair and a white barren desert in the background

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

LeosamsHelloworldXL: A close-up of a woman's face, partially obscured by a translucent veil adorned with intricate lacework

Dreamshaper: As the sun sets, under the cliff, a few elk drink by the river. The river water reflects the sun and sparkles.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DreamshaperXL: a dragon

Dreamlike: An ancient beauty from the Tang Dynasty

- Figure 5. Qualitative results. We compare images generated by VMix-integrated personalized models with those from standard personalized models. On the left are images produced by the personalized model with VMix integration, while on the right are images from the standard personalized model without modifications.

[Figure 82]

[Figure 83]

- Figure 6. User study. We report the user preference between using VMix and not using VMix.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

𝜆 = 1.4 𝜆 = 1.8

𝑆𝐷(𝜆 = 0) 𝜆 = 1

A cat is walking on the windows.

(a)

| | | | | |
|---|---|---|---|---|
| |VMix<br><br>DPO<br><br>FreeU| | | |
| |SD1.5<br><br>SFT<br><br>SFT&TI| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

31.0 31.5 32.0 32.5 CLIP Score

5.4

5.5

5.6

5.7

AesScore

(b)

- Figure 7. Ablation Study for λ of VMix. (a)Visual performance changes of λ. (b)Performance metrics for VMix, evaluated across a range of λ values from 1 to 2 from right to left.

the incorporation of aesthetic embeddings has not detracted from the model’s inherent capabilities. Our findings are consistent with the observations made in Figure 5, where VMix significantly enhances the aesthetic dimensions of images, such as lighting and color. Additionally, imperfections on body parts, such as unrealistic limbs or missing extremities, can be further corrected. Moreover, details in close-up images, like skin texture, are also improved,

Baseline VMix

###### Single-dimension

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Color Light

None aesthetic labels

[Figure 92]

[Figure 93]

[Figure 94]

“A girl leaning against a window with a breeze blowing, summer portrait, half-length medium view”

Full aesthetic labels

Emotion Texture

Figure 8. Ablation Study for AesEmb of VMix. Left: The effects of using all aesthetic labels versus not using them. Right: The effects of using single-dimensional aesthetic labels.

thereby enhancing the overall aesthetic presentation of the images.

###### 4.4. Ablation Study

Effect of λ. As illustrated in Fig. 7, in Value-Mixed CrossAttention, λ is adjustable during inference. When λ increases, the Aes Score is observed to gradually increase, while there is a slight decline in the CLIP Score. However, our method still maintains a significant advantage over other approaches.

Effect of AesEmb. As illustrated in Fig. 8, we conduct an ablation study on the role of AesEmb. When using only single-dimensional aesthetic label, it can be observed that the image quality improves in specific dimensions. When employing full positive aesthetic labels, the visual performance of the images is superior to the baseline overall. This indicates that the incorporation of AesEmb can enhance the visual appearance of images across various aesthetic dimensions. Throughout this process, we did not utilize LoRA.

Effect of LoRA and VMix cross-attention. In Tab. 3, we examine the impact of lora and value-mixed cross-attention utilized in our method. We discover that both of them can enhance the performance metrics of the baseline. Without value-mixed cross-attention, there is a significant drop in the model’s performance, with the Aes Score decreasing from 5.79 to 5.52 and the CLIP Score from 30.50 to 30.16. This indicates that value-mixed cross-attention plays a significant role in text fidelity and image aesthetics. By combining both, we achieved the best performance.

### 5. Conclusion

In this paper, we present VMix, which uses the disentangled aesthetic description as an additional condition and employs a cross-attention control method to enhance the performance of model across various aesthetic dimensions. We discover one of the most crucial factors for aligning the model with human expectations is training with decoupled, fine-grained aesthetic labels

using a suitable conditional control method. Inspired by these, we proposed an effective conditional control method that significantly improves the generative quality of the model. Extensive experiments validate that VMix surpasses other state-of-the-art methods in terms of text fidelity and visual aesthetics. As a plug-and-play plugin, VMix can seamlessly integrate with open-source models, enhancing aesthetic performance and thereby further promoting the development of the community.

### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 4

- [2] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 5
- [3] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 1, 2, 7
- [4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2
- [5] Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie-vilg 2.0: Improving text-toimage diffusion model with knowledge-enhanced mixtureof-denoising-experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10135–10145, 2023. 1
- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3, 7
- [7] Feihong He, Gang Li, Mengyuan Zhang, Leilei Yan, Lingyu Si, and Fanzhang Li. Freestyle: Free lunch for textguided style transfer using diffusion models. arXiv preprint arXiv:2401.15636, 2024. 1
- [8] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 6, 7
- [9] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023. 1
- [10] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights

- towards enhancing aesthetic quality in text-to-image generation, 2024. 7
- [11] Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19401–19411, 2024. 1
- [12] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [13] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 3
- [14] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1
- [15] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 2, 6
- [16] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 2, 3, 4
- [17] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2
- [18] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 2
- [19] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1

- [20] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3, 4, 5, 7
- [21] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015. 2, 3
- [22] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3, 4

- [23] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 7
- [24] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. arXiv preprint arXiv:2309.11497, 2023. 1, 2, 7
- [25] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6
- [26] Ming Tao, Hao Tang, Fei Wu, Xiao-Yuan Jing, Bing-Kun Bao, and Changsheng Xu. Df-gan: A simple and effective baseline for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16515–16525, 2022. 2
- [27] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 1, 2, 7
- [28] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 3
- [29] Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. Uncovering the disentanglement capability in textto-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1900–1910, 2023. 4
- [30] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 1
- [31] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1316– 1324, 2018. 2
- [32] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 1

- [33] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 2
- [34] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multimodal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023. 2

- [35] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-toimage generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 833–842, 2021. 2
- [36] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3, 4, 1

### 6. Supplementary

###### 6.1. More Qualitative Comparison

VMix decouples aesthetic knowledge from content knowledge and introduces a novel conditional control method. To further verify its effectiveness, we provide additional experimental results here.

Training Stability. In Sec. 3, we introduced Value-Mixed Cross-Attention(vmix cross-attention), which learns a single value for the projected aesthetic embedding. This approach might seem counterintuitive, particularly since in the aesthetic branch, the Q (Query) and V (Value) originate from different sources. In practice, AesEmb is initialized from the [CLS] tokens of the text model, ensuring semantic alignment with the original text embedding. Furthermore, the projected aesthetic embedding must pass through a zero-initialized linear layer before it can finally enter the vmix cross-attention. This process further ensures a gentle injection of aesthetic knowledge, minimizing disruption to the original model. As shown in Fig. 9, the entire training process is relatively stable, with gradual improvements in lighting, color, and other visual aspects.

Effect of VMix Cross-Attention. In vmix cross-attention, we designed the aesthetic branch and the content branch to share the same attention map, denoted as QKcT, to prevent the injection of aesthetic knowledge from significantly impairing the model’s text fidelity capabilities. As demonstrated in Fig. 10, VMix produces an attention score map that closely resembles that of the baseline. After the denoising process, we can obtain a generated image that maintains a layout roughly equivalent to the baseline but with enhanced quality. This indicates that our vmix cross-attention allows the model to concentrate more on refining overall details, thereby directly boosting the model’s performance across various fine-grained aesthetic dimensions, including lighting, color, and more.

Comparison with LoRA. Although in Sec. 4, we compared our method with SFT and textual inversion on the same dataset, the use of LoRA in our training process might obscure the enhancement of the final results. To clarify this, we trained a model with only LoRA on the same dataset. As shown in Fig. 11, our method significantly improves upon the SD [20] more than the approach that uses only LoRA for training.

###### 6.2. More Visualization

We apply VMix with ControlNet[36] and IP-Adapter [32]. As shown in Fig. 12, VMix can be compatible with these standard methods and generates images with better visual aesthetics. As shown in the Fig. 13, we have provided additional comparative results with SDXL [15] as well as its variants. When VMix is incorporated, the generated results show significant improvements across various aesthetic di-

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

1 steps 2000 steps 4000 steps 6000 steps 8000 steps

###### Figure 9. Visualization results of different training steps. Prompts:

(1) A teddy bear walking in the snowstorm. (2) A bridge is depicted in the water.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

BaselineVMix(Ours)

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

A dog wears a hat a cat rides a bike

###### Figure 10. Visualization results of attention maps. VMix is capable of maintaining attention maps that closely resembles that of the baseline(SD [20]) while further enhancing the quality of the generated images.

Baseline VMix(Ours) LoRA-only

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

###### Figure 11. Qualitative comparison. Prompts: (1) Kitten in the forest with flowers with sunlight on them, Cinematic lighting, Unreal Engine 5. (2) Close-up of a young girl wearing a flower crown in the garden, portrait. (3) A green vase with several red roses in it.

mensions, offering enhanced visual performance.

[Figure 126]

[Figure 127]

[Figure 128]

BaselineVMix(Ours)Condition

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Pose Depth Reference+Pose

- Figure 12. Qualitative results about VMix with ControlNet[36] and IP-Adapter [32]. Prompt: a young woman with long, wavy brown hair. she is wearing a sleeveless floral dress with a pattern of various flowers and leaves. the woman is holding a white, fluffy cat close to her face, seemingly in a moment of affection and joy. her eyes are closed, suggesting she is savoring the moment.

###### 6.3. Limitations

Despite the superior aesthetic generation effects achieved by VMix, it still has several limitations: (1) Currently, the aesthetic labels form a closed set, and the included aesthetic dimensions may not cover all necessary aspects. Although we have confirmed the effectiveness of our current method, VMix’s performance is inevitably impacted. We intend to further optimize this aspect in our future work. (2) Images generated by VMix may exhibit a bias towards certain specific objects. For instance, when we attempt to generate concrete objects found in real life, such as cups or mobile phones, and include all aesthetic labels, including emotional ones, during the inference phase, the resulting images might unexpectedly depict humans. This is because, in the training set, emotional labels are typically associated only with people or animals. Consequently, these labels may become bound to specific entities during the training phase, potentially affecting the outcomes of the inference process.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

##### SDXL

photography portrait, a beautiful girl with red hair, holding a sword, high detail

blonde beauty, pale skin, slim, sexy, light brown eyes

#### LeosamsHelloworldXL

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

A girl with brown hair wearing a brown sweater, short, broken hair, brown merlod costume, holding a glass, glass tube soda, black straw, empty street background, half

Film portrait of a girl with a latte perm, in a street scene, Fuji camera, bust

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

#### DreamshaperXL

A female model with smoky eyes and a leather jacket with her hands in her pockets

White hair, mechanical girl, dark, cyberpunk style

- Figure 13. Qualitative comparison between results with VMix(on the right) and without VMix(on the left), shows that VMix significantly enhances the quality of image generation.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

#### AWPortraitXL

A boy solo, white t-shirt, realistic clothing, simple white background, shut up, lifelike, fluffy hair, front light illuminates clear skin texture, textured skin, clear pore details, healthy glowing skin

A boy, sitting in a tree, taking a nap, the shade of the tree, the afternoon sun shining through the leaves on the boy's face, bust

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

#### RealVisXL

A girl with cold white skin, leaning against a window with a breeze blowing, summer portrait, half-length medium view

A young scientist is in his laboratory, with experimental reagents in his hands, studying intently, bust

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

#### AlbedobaseXL

Japanese anime, celluloid, the girl is magical, she has wings like an angel, there is a riot of energies around her. She brews a new reality. new energies are born around it, which are given by forces and resources. Around her fantasy nature. flowers and animals

A wildlife photographer wearing a vest and jacket, lying prone in the grass, waiting to capture a rare animal

- Figure 14. Qualitative comparison between results with VMix(on the right) and without VMix(on the left), shows that VMix significantly enhances the quality of image generation.

