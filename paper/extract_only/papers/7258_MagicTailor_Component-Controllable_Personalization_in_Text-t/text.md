## MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models

Donghao Zhou1∗ , Jiancheng Huang2∗ , Jinbin Bai3 , Jiaze Wang1 , Hao Chen1 , Guangyong Chen4 , Xiaowei Hu5† , Pheng-Ann Heng1

1CUHK 2SIAT, CAS 3NUS 4Zhejiang Lab 5Shanghai AI Lab dhzhou@link.cuhk.edu.hk, huxiaowei@pjlab.org.cn

https://correr-zhou.github.io/MagicTailor

# arXiv:2410.13370v3[cs.CV]21May2025

- (a)
- (b)

(c)

### Abstract

###### Reference Generated Images

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

<dog><ear>

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]|
|---|

Text-to-image diffusion models can generate highquality images but lack fine-grained control of visual concepts, limiting their creativity. Thus, we introduce component-controllable personalization, a new task that enables users to customize and reconfigure individual components within concepts. This task faces two challenges: semantic pollution, where undesired elements disrupt the target concept, and semantic imbalance, which causes disproportionate learning of the target concept and component. To address these, we design MagicTailor, a framework that uses Dynamic Masked Degradation to adaptively perturb unwanted visual semantics and Dual-Stream Balancing for more balanced learning of desired visual semantics. The experimental results show that MagicTailor achieves superior performance in this task and enables more personalized and creative image generation.

<person><hair><person>

|[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

“<person>”

Personaliza on

“<dog> with <ear>,  on the bench”

“<dog> with <ear>, in the snow”

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|
|---|

<person><beard>

|[Figure 21]|
|---|

|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]|
|---|

“<person> with <hair>”

Component-Controllable Personaliza on (Ours)

“<person> with <beard>, with clouds in the background”

“<person> with <beard>, Ukiyo-e painting”

Figure 1: (a) Personalization: T2I models learn from reference images and then generate predefined visual concepts. (b) Componentcontrollable personalization: T2I models learn from additional visual references and then enable the integration of specific components into given concepts, further unleashing creativity. (c) Generated images by MagicTailor: MagicTailor can effectively achieve component-controllable personalization. Note that red and blue circles indicate the target concept and component, respectively.

In this paper, we introduce component-controllable personalization, a new task that enables the reconfiguration of specific components within personalized concepts using additional visual references (Fig. 1(b)). In this approach, a T2I model is fine-tuned with reference images and corresponding category labels, allowing it to learn and generate the desired concept along with the given component. This capability empowers users to refine and customize concepts with precise control, fostering creativity and innovation across various domains, from artworks to inventions.

### 1 Introduction

Text-to-image (T2I) diffusion models [Rombach et al., 2022; Ramesh et al., 2022; Chen et al., 2023] have shown impressive capabilities in generating high-quality images from textual descriptions. While these models can generate images that align well with provided prompts, they struggle when certain visual concepts are hard to express in natural language. To address this, methods like [Gal et al., 2022; Ruiz et al., 2023] enable T2I models to learn specific concepts from a few reference images, allowing for more accurate integration of those concepts into the generated images. This process, as shown in Fig. 1(a), is referred as personalization.

One challenge of this task is semantic pollution (Fig. 2(a)), where unwanted visual elements inadvertently appear in generated images, “polluting” the personalized concept. This happens because the T2I model often mixes visual semantics from different regions during training. Masking out unwanted elements in reference images doesn’t solve the problem, as it disrupts the visual context and causes unintended compositions. Another challenge is semantic imbalance (Fig. 2(b)), where the model overemphasizes certain aspects, leading to unfaithful personalization. This occurs due to the semantic disparity between the concept and component, necessitating a more balanced learning approach to manage concept-level (e.g., person) and component-level (e.g., hair) semantics.

However, existing personalization methods are limited to replicating predefined concepts and lack flexible and finegrained control of these concepts. Such a limitation hinders their practical use in real-world applications, restricting their potential for creative expression. Inspired by the observation that concepts often comprise multiple components, a key problem in personalization lies in how to effectively control and manipulate these individual components.

* Equal contribution. † Corresponding author.

(GANs) [Reed et al., 2016; Xu et al., 2018], and transformers [Ding et al., 2021; Yu et al., 2022; Bai et al., 2024] also showed the potential in conditional generation. The advent of diffusion models has ushered in a new era in T2I generation [Li et al., 2024; Saharia et al., 2022; Ramesh et al.,

- (a) Challenge #1: Seman c Pollu on

- (b) Challenge #2: Seman c Imbalance

###### (C) Id. Fidelity Performance

###### Reference Generated Images (“<person> with <eye>”)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| |Best<br><br>| | | | |
| | | | | | |
| | | | | | |

|[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

<person><eye>

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]|
|---|

- 2022; Chen et al., 2023; Xue et al., 2024]. Leveraging these models, a range of related applications has rapidly emerged, including image editing [Li et al., 2024; Mou et al., 2024; Huang et al., 2024b; Feng et al., 2024; Huang et al., 2024a], image completion and translation [Xie et al., 2023b,a; Lin et

- al., 2024], and controllable image generation [Zhang et al.,

2023; Wang et al., 2024; Zheng et al., 2023]. Despite advancements in T2I diffusion models, generating images that accurately reflect specific, user-defined concepts remains a challenge. This study explores component-controllable personalization, which allows precise adjustment of specific concepts’ components using visual references.

Personalization. Personalization seeks to adapt T2I models to generate given concepts using reference images. Initial approaches [Gal et al., 2022; Ruiz et al., 2023] addressed this task by either optimizing text embeddings or fine-tuning the entire T2I model. Additionally, low-rank adaptation (LoRA) [Hu et al., 2021] has been widely adopted in this field [Ryu, 2022], providing an efficient solution. The scope of personalization has expanded to encompass multiple concepts [Kumari et al., 2023; Avrahami et al., 2023; Gu et al., 2024; Ng et al., 2025]. Besides, several studies [Li et al., 2023; Wei et al., 2023; Zhang et al., 2024b; Song et al., 2024] have explored tuning-free approaches for personalization, but these necessitate additional training on large-scale image datasets [Zhang et al., 2024a]. In contrast, MagicTailor is a tuningbase method that requires only a few images and leverages test-time optimization to enable stable performance. Notably, several works [Huang et al., 2024c; Safaee et al., 2024; Ng et

- al., 2025] have also explored how to learn and customize finegrained elements. However, these methods can only combine elements or process one element at the same level. By comparison, MagicTailor is a versatile framework able to handle both component-level and concept-level elements.

(i) W/o DM-Deg (ii) Mask-Out Strategy (iii) W/ DM-Deg (Ours)

###### Reference Generated Images (“<tower> with <roof>”)

###### Reference Generated Images (“<person> with <hair>”)

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]|
|---|

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

<person>

<roof><tower>

|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]|
|---|

|[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|
|---|

<hair>

(i) W/o DS-Bal (ii) W/ DS-Bal (Ours)

(i) W/o DS-Bal (ii) W/ DS-Bal (Ours)

- Figure 2: Major challenges in component-controllable personalization. (a) Semantic pollution: (i) Undesired elements may interfere with the personalized concept. (ii) A simple mask-out strategy causes unintended results, while (iii) DM-Deg effectively suppresses unwanted semantics. (b) Semantic imbalance: (i) Simultaneously learning the concept and component can distort either one. (ii) DSBal ensures balanced learning, improving personalization. (c) Identity fidelity performance: Calculating DreamSim [Fu et al., 2023] scores on our collected dataset, we show that DM-Deg and DS-Bal can address these challenges for faithful generation.

To address these challenges, we propose MagicTailor, a novel framework that enables component-controllable personalization for T2I models (Fig. 1(c)). We first use a textguided image segmenter to generate segmentation masks for both the concept and component and then design Dynamic Masked Degradation (DM-Deg) to transform reference images into randomly degraded versions, perturbing undesired visual semantics. This method helps suppress the model’s sensitivity to irrelevant details while preserving the overall visual context, effectively mitigating semantic pollution. Next, we initiate a warm-up phase for the T2I model, training it on the degraded images using a masked diffusion loss to focus on the desired semantics and a cross-attention loss to strengthen the correlation between these semantics and pseudo-words. To address semantic imbalance, we develop Dual-Stream Balancing (DS-Bal), a dual-stream learning paradigm that balances the learning of visual semantics. In this phase, the online denoising U-Net performs sample-wise min-max optimization, while the momentum denoising U-Net applies selective preservation regularization. This ensures more faithful personalization of the target concept and component, resulting in outputs that better align with the intended objective.

### 3 Methodology

Let I = {({Ink}Kk=1,cn)}Nn=1 denote a concept-component pair with N samples of concepts and components, where each

sample contains K reference images {Ink}Kk=1 with a category label cn. In this work, we focus on a practical setting involving one concept and one component. Specifically, we set N = 2 and define the first sample as a concept (e.g., dog) while the second one as a component (e.g., ear). In addition, these samples are associated with the pseudo-words P = {pn}Nn=1 serving as their text identifiers. The goal of component-controllable personalization is to fine-tune a textto-image (T2I) model to accurately learn both the concept and the component from I. Using text prompts with P, the finetuned model should generate images that integrate the personalized concept with the specified component.

In the experiments, we validate the superiority of MagicTailor through various qualitative and quantitative comparisons, demonstrating its state-of-the-art (SOTA) performance in component-controllable personalization. Moreover, detailed ablation studies and analysis further confirm the effectiveness of MagicTailor. In addition, we also show its potential for enabling a wide range of creative applications.

### 2 Related Works

Text-to-Image Generation. T2I generation has made remarkable advancements in recent years, enabling the synthesis of vivid and diverse imagery based on textual descriptions. Early methods employed Generative Adversarial Networks

This section begins by providing an overview of the MagicTailor pipeline in Sec. 3.1 and then delves into its two core techniques in Sec. 3.2 and Sec. 3.3.

Dynamic Intensity

“A photo of <Person>”  “A photo of <eyebrow>” 

Frozen

Loss Computa on

[Figure 53]

❄

Reference Images

[Figure 54]

🔥

Text Encoder

| | | |
|---|---|---|
| | | |
| | | |
| | | |

[Figure 55]

✘ LossNeglect

🔥 Trained

[Figure 56]

[Figure 57]

❄

|[Figure 58]|
|---|

|[Figure 59]|
|---|

[Figure 60]

🔥

Text Prompts

###### DS-Bal

Preserving Labels

[Figure 61]

(Sec. 3.3)

|[Figure 62]<br><br>[Figure 63]|
|---|

|[Figure 64]<br><br>[Figure 65]|
|---|

|[Figure 66]|
|---|

Momentum Denoising U-Net

|[Figure 67]<br><br>[Figure 68]|
|---|

“Person” “eyebrow”

[Figure 69]

❄

LoRA

[Figure 70]

❄

Encoder

Image

###### DM-Deg

Text-Guided Image Segmenter

(Sec.3.2) ✘

EMA

[Figure 71]

|[Figure 72]<br><br>[Figure 73]|
|---|

|[Figure 74]|
|---|

[Figure 75]

❄

|[Figure 76]<br><br>[Figure 77]|
|---|

|[Figure 78]<br><br>[Figure 79]|
|---|

Noise

Online Denoising U-Net

Select

|[Figure 80]|
|---|

|[Figure 81]|
|---|

LoRA

Noisy Latent Images

[Figure 82]

🔥

Degraded Images

Extract

Predic ons

✘

[Figure 83]

|[Figure 84]<br><br>[Figure 85]|
|---|

|[Figure 86]<br><br>[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

Downsample

Segmenta on Masks

[Figure 92]

Target Labels

Downsampled Segmenta on Masks Cross-A en on Maps

- Figure 3: Pipeline overview of MagicTailor. This method fine-tunes a T2I diffusion model using reference images to learn both the target concept and component, enabling the generation of images that seamlessly integrate the component into the concept. Two key techniques, Dynamic Masked Degradation (DM-Deg, see Sec. 3.2) and Dual-Stream Balancing (DS-Bal, see Sec. 3.3), address semantic pollution and semantic imbalance, respectively. For clarity, only one image per concept/component is shown, and the warm-up stage is omitted.

#### 3.1 Overall Pipeline

when Aθ(pn,znk(t)) is the cross-attention maps between the pseudo-word pn and the noisy latent image znk(t) and Mnk′′ is downsampled from Mnk to match the shape of Aθ(pn,znk(t)).

The overall pipeline of MagicTailor is illustrated in Fig. 3. The process begins with identifying the desired concept or component within each reference image Ink, employing an off-the-shelf text-guided image segmenter to generate a segmentation mask Mnk based on Ink and its associated category label cn. Conditioned on Mnk, we design Dynamic Masked Degradation (DM-Deg) to perturb undesired visual semantics within Ink, addressing semantic pollution. At each training step, DM-Deg transforms Ink into a randomly degraded image Iˆnk, with the degradation intensity being dynamically regulated. Subsequently, these degraded images, along with structured text prompts, are used to fine-tune a T2I diffusion model to facilitate concept and component learning. The model is formally expressed as {ϵθ,τθ,E,D}, where ϵθ represents the denoising U-Net, τθ is the text encoder, and E and D denote the image encoder and decoder, respectively. To promote the learning of the desired visual semantics, we employ the masked diffusion loss, which is defined as:

Using Ldiff and Lattn, we first warm up the T2I model by jointly learning all samples, aiming to preliminarily inject the knowledge of visual semantics. The loss of the warm-up stage is defined as:

Lwarm-up = Ldiff + λattnLattn , (3)

where λattn = 0.01 is the loss weight for Lattn. For efficient fine-tuning, we only train the denoising U-Net ϵθ in a lowrank adaptation (LoRA) [Hu et al., 2021] manner and the text embedding of the pseudo-words P, keeping the others frozen. Thereafter, we employ Dual-Stream Balancing (DSBal) to address semantic imbalance. In this paradigm, the online denoising U-Net ϵθ conducts sample-wise min-max optimization for the hardest-to-learn sample, and meanwhile the momentum denoising U-Net ϵ˜θ applies selective preserving regularization for the other samples.

Ldiff = En,k,ϵ,t ϵn⊙Mnk′ −ϵθ(znk(t),t,en)⊙Mnk′ 22 , (1)

#### 3.2 Dynamic Masked Degradation

Semantic pollution is a significant challenge for componentcontrollable personalization. As shown in Fig. 2(a.i), the target concept (i.e., person) can be distorted by the owner of the target component (i.e., eye), resulting in a hybrid person. Masking regions outside the target concept and component can damage the overall context, leading to overfitting and odd compositions (Fig. 2(a.ii)). To address this, undesired visual semantics in reference images must be handled appropriately. We propose Dynamic Masked Degradation (DM-Deg), which dynamically perturbs undesired semantics to suppress their

where ϵn ∼ N(0,1) is the unscaled noise, znk(t) is the noisy latent image of Iˆnk with a random time step t, en is the text embedding of the corresponding text prompt, and Mnk′ is downsampled from Mnk to match the shape of ϵ and znk. Additionally, we incorporate the cross-attention loss to strengthen the correlation between desired visual semantics and their corresponding pseudo-words, formulated as:

Lattn = En,k,t Aθ(pn,znk(t)) − Mnk′′ 22 , (2)

###### Reference Generated Images (“<tower> with <roof>”)

###### Reference Generated Images (“<person> with <beard>”)

| |Best| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

<person><beard>

|[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]|
|---|

<tower><roof>

- (a)
- (b)

|[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]|
|---|

Step = 200 Step = 300 Step = 400

Step = 500

|[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

(a) Fixed Intensity (b) Dynamic Intensity (Ours)

(c) IQA Performance

###### Figure 4: Motivation of dynamic intensity. (a) Fixed intensity

(αd = 0.5 here) could cause noisy generated images. (b) Our dynamic intensity can mitigate noise memorization. (c) We report IQA results of Q-Align [Wu et al., 2023] on our dataset, showing that our dynamic intensity helps to enhance the quality of generated images.

Figure 5: Learning process visualization. (a) The vanilla learning paradigm tends to overemphasize the easier one. (b) DS-Bal effectively balances the learning of the concept and component.

influence on the T2I model while preserving the overall visual context (Fig. 2(a.iii)&(c)).

hair), but in some cases, components may have more complex semantics (e.g., simple tower vs. intricate roof). This imbalance complicates joint learning, leading to overemphasis on either the concept or the component, and resulting in incoherent generation (Fig. 5(a)). To address this, we design Dual-Stream Balancing (DS-Bal), a dual-stream learning paradigm integrated with online and momentum denoising U-Nets (Fig. 3) for balanced semantic learning, aiming to improve personalization fidelity (Fig. 5(b) & Fig. 2(c)).

Degradation Imposition. In each training step, DM-Deg imposes degradation in the out-of-mask region for each reference image. We use Gaussian noise for degradation due to its simplicity. For a reference image Ink, we randomly sample a Gaussian noise matrix Gnk ∼ N(0,1) with the same shape as Ink, where the pixel values of Ink range from −1 to 1. The degradation is then applied as follows:

Iˆnk = αdGnk ⊙ (1 − Mnk) + Ink, (4) where ⊙ denotes element-wise multiplication, and αd ∈ [0,1] is a dynamic weight controlling the degradation intensity. While previous works [Xiao et al., 2023; Li et al., 2023] have used noise to fully cover the background or enhance data diversity, DM-Deg aims to produce a degraded image Iˆnk that retains the original visual context. By introducing Iˆnk, we can suppress the T2I model from perceiving undesired visual semantics in out-of-mask regions, as these semantics are perturbed by random noise at each training step.

Sample-Wise Min-Max Optimization. From a loss perspective, the visual semantics of the concept and component are learned by optimizing the masked diffusion loss Ldiff across all the samples. However, this indiscriminate optimization fails to allocate sufficient learning effort to a more challenging sample, leading to an imbalanced learning process. To address this, DS-Bal uses the online denoising UNet to focus on learning the hardest-to-learn sample at each training step. Inheriting the weights of the original denoising U-Net, which is warmed up through joint learning, the online denoising U-Net ϵθ optimizes only the sample with the highest masked diffusion loss as:

Dynamic Intensity. Unfortunately, the T2I model may gradually memorize the introduced noise while learning meaningful visual semantics, leading to noise appearing in generated images (Fig. 4(a)). This behavior is consistent with previous observations on deep networks [Arpit et al., 2017]. To address this, we propose a descending scheme that dynamically regulates the intensity of the imposed noise during training. This scheme follows an exponential curve, maintaining a relatively high intensity in the early stages and decreasing it sharply in later stages. Let d denote the current training step and D denote the total training step. The curve of dynamic intensity is defined as:

Ek,ϵ,t ϵn ⊙ Mnk′ − ϵθ(znk(t),t,en) ⊙ Mnk′ 22 , (6)

Ldiff-max = max

n

where minimizing Ldiff-max can be considered as a form of min-max optimization [Razaviyayn et al., 2020]. The learn-

ing objective of ϵθ may switch across different training steps and is not consistently dominated by the concept or component. Such an optimization scheme can effectively modulate the learning dynamics of multiple samples and avoid the overemphasis on any particular one.

d D

)γ) , (5)

αd = αinit(1 − (

where αinit is the initial value of αd and γ controls the descent rate. We empirically set αinit = 0.5 and γ = 32, tuned within the powers of 2. This dynamic intensity scheme effectively prevents semantic pollution and significantly mitigates the memorization of introduced noise, leading to improved generation performance (Fig. 4(b)&(c)).

Selective Preserving Regularization. At a training step, the sample neglected in Ldiff-max may suffer from knowledge forgetting. This is because the optimization of Ldiff-max, which aims to enhance the knowledge of a specific sample, could inadvertently overshadow the knowledge of the others. In light of this, DS-Bal meanwhile exploits the momentum denoising U-Net ϵ˜θ to preserve the learned visual semantics of the other sample in each training step. Specifically, we first select the sample that is excluded in Ldiff-max, which is expressed as S = {n|n = 1,...,N} − {nmax}, where nmax is the index of the target sample in Ldiff-max and S is the selected

- 3.3 Dual-Stream Balancing Another key challenge is semantic imbalance, which arises from the disparity in visual semantics between the target concept and its component. Specifically, concepts generally possess richer visual semantics than components (e.g., person vs.

- Table 1: Quantitative comparisons on automatic metrics. MagicTailor can achieve SOTA performance on all four automatic metrics. The best results are marked in bold.

Methods CLIP-T ↑ CLIP-I ↑ DINO ↑ DreamSim ↓

Textual Inversion [Gal et al., 2022] 0.236 0.742 0.620 0.558 DreamBooth [Ruiz et al., 2023] 0.266 0.841 0.798 0.323 Custom Diffusion [Kumari et al., 2023] 0.251 0.797 0.750 0.407 Break-A-Scene [Avrahami et al., 2023] 0.259 0.840 0.780 0.338 CLiC [Safaee et al., 2024] 0.263 0.764 0.663 0.499 MagicTailor (Ours) 0.270 0.854 0.813 0.279

- Table 2: Quantitative comparisons on the user study. MagicTailor also outperforms other methods in all aspects of human evaluation.

Reference

Textual Inversion

DreamBooth

Custom Diﬀusion

Break-A-Scene

CLiC

MagicTailor (Ours)

|[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

<person><hair>

|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]|
|---|

“<person> with <hair>, from 3D rendering”

|[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

<person><beard>

|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]|
|---|

“<person> with <beard>, with flowers in the background”

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]|
|---|

<person><lighthouse><roof><bottle><lid><dog><ear><eye>

|[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]|
|---|

“<person> with <eye>, in a close view”

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]|
|---|

|[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]|
|---|

Methods Text Align. ↑ Id. Fidelity ↑ Gen. Quality ↑ Textual Inversion [Gal et al., 2022] 5.8% 2.5% 5.2% DreamBooth [Ruiz et al., 2023] 15.3% 14.7% 12.5% Custom Diffusion [Kumari et al., 2023] 7.1% 7.7% 9.8% Break-A-Scene [Avrahami et al., 2023] 10.8% 12.1% 22.8% CLiC [Safaee et al., 2024] 4.5% 5.1% 6.2% MagicTailor (Ours) 56.5% 57.9% 43.4%

“<lighthouse> with <roof>, in Pixel Art style”

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]|
|---|

|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]|
|---|

“<bottle> with <lid>, in the jungle”

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|
|---|

|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]|
|---|

concept-component pair requires only about five minutes of training on an A100 GPU. For evaluation, we design 20 text prompts covering a wide range of scenarios and generate 14,720 images for each method. To ensure fairness, all random seeds are fixed during both training and inference. More details of the experimental setup are included in Appendix A.

“<dog> with <ear>, near the Eiffel Tower”

- Figure 6: Qualitative comparisons. We present images generated by MagicTailor and other methods across various domains. MagicTailor achieves better text alignment, identity fidelity, and generation quality. Due to space limitations, please zoom in for a better view. More results are provided in Appendix D.

Compared Methods. We compare our MagicTailor with several personalization methods, including Textual Inversion (TI) [Gal et al., 2022], DreamBooth (DB) [Ruiz et al., 2023], Custom Diffusion (CD) [Kumari et al., 2023], Break-AScene (BAS) [Avrahami et al., 2023], and CLiC [Safaee et al., 2024]. These methods were selected for their representativeness of personalization frameworks or relevance to learning fine-grained elements. For a fair comparison, we adapt them to our task with minimal modifications, specifically by incorporating the masked diffusion loss (Eq. 1). Apart from method-specific configurations, all methods are implemented using the same setup to ensure consistency.

index set. Then, we use ϵ˜θ to apply regularization for S, with the masked preserving loss as:

Lpres = En∈S,k,t ϵ ˜θ(znk(t),t,en) ⊙ Mnk′ − ϵθ(znk(t),t,en) ⊙ Mnk′ 22 , (7)

where ϵ˜θ is updated from ϵθ using EMA [Tarvainen and Valpola, 2017] with the smoothing coefficient β = 0.99,

thereby sustaining the prior accumulated knowledge of ϵθ in each training step. By encouraging the consistency between

the output of ϵθ and ϵ˜θ in Lpres, we can facilitate the knowledge maintenance of the other samples while learning a spe-

cific sample in Ldiff-max. Overall, DS-Bal can be considered a mechanism to adaptively assign target labels ϵn or preserving

#### 4.2 Qualitative Comparisons

labels ϵ˜θ(znk(t),t,en) to different samples, enabling dynamic loss supervision (Fig. 3). Using a loss weight λpres = 0.2, the total loss of the DS-Bal stage is formulated as:

The qualitative results are shown in in Fig. 6. As observed, TI, CD, and CLiC primarily suffer from semantic pollution, where undesired visual semantics significantly distort the personalized concept. Besides, DB and BAS also struggle in this challenging task, with an overemphasis on either the concept or the component due to semantic imbalance, sometimes even causing the target component to be completely absent. An interesting finding is that imbalanced learning can exacerbate semantic pollution, leading to the color and texture of the target concept or component being mistakenly transferred to unintended parts of the generated images. In contrast, MagicTailor effectively generates text-aligned images that accurately represent both the target concept and component. To further demonstrate the performance of MagicTailor, we provide additional comparisons in Appendix B.

LDS-Bal = Ldiff-max + λpresLpres + λattnLattn . (8)

### 4 Experimental Results

#### 4.1 Experimental Setup

Dataset, Implementation, and Evaluation. For a systematic investigation, we collect a dataset from diverse domains, including characters, animation, buildings, objects, and animals. We use Stable Diffusion (SD) 2.1 [Rombach et al., 2022] as the pretrained T2I model. For the warm-up and DSBal stages, we set the training steps to 200 and 300, with learning rates of 1 × 10−4 and 1 × 10−5, respectively. Each

Table 3: Effectiveness of key techniques. Our DM-Deg and DSBal effectively contribute to a superior performance trade-off.

Ours (CLIP-T) 2nd-Best (CLIP-T) Ours (DreamSim) 2nd-Best (DreamSim)

0.325

0.274

0.315

0.272

DM-Deg DS-Bal CLIP-T ↑ CLIP-I ↑ DINO ↑ DreamSim ↓

DreamSim

0.305

CLIP-T

0.270

0.295

0.275 0.837 0.798 0.317 ✓ 0.276 0.848 0.809 0.294 ✓ 0.270 0.845 0.802 0.304 ✓ ✓ 0.270 0.854 0.813 0.279

0.268

0.285

0.266

0.275

0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

pres

pres

0.325

0.274

###### SD 1.5 SDXL

###### SD 2.1

0.315

0.272

DreamSim

0.305

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

CLIP-T

0.270

0.295

0.268

0.285

0.266

0.275

###### Reference

1e-3 5e-3 1e-2 5e-2

1e-3 5e-3 1e-2 5e-2

|[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]|
|---|

<person><beard>

attn

attn

- Figure 8: Robustness on loss weights. We report CLIP-T [Gal et al., 2022] for text alignment, and DreamSim [Fu et al., 2023] for identity fidelity as it is most similar to human judgments. Secondbest results in Table 1 are also presented to highlight our robustness.

“<tower> with <roof>, in the jungle”

|[Figure 193]|
|---|

<tower><roof>

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

2 Reference Images

|[Figure 198]|
|---|

<tower><roof>

|[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]|
|---|

|[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]|
|---|

3 Reference Images

<tower><roof>

1 Reference Image

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

- Figure 9: Performance on different numbers of reference images. We present qualitative results to show that MagicTailor can still achieve satisfactory performance when provided only 1 or 2 reference image(s) per concept and component.c

“<person> with <beard>, wearing a raincoat, looking up, in the rain”

|[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

“<person> with <hair>, wearing a jacket, in a mountain landscape”

- Figure 7: Compatibility with different backbones. We equip MagicTailor with SD 1.5 [Rombach et al., 2022], SD 2.1 [Rombach et al., 2022], and SDXL [Podell et al., 2023]. The results show that MagicTailor can be generalized to multiple backbones, and a better backbone could provide better generation quality.

#### 4.3 Quantitative Comparisons

showing its reliability. On top of that, we introduce DM-Deg and DS-Bal, where the superior performance trade-off indicates their significance. Qualitative results can refer to Fig. 2.

Automatic Metrics. We utilize four automatic metrics in the aspects of text alignment (CLIP-T [Gal et al., 2022]) and identity fidelity (CLIP-I [Radford et al., 2021], DINO [Oquab et al., 2023], DreamSim [Fu et al., 2023]). To precisely measure identity fidelity, we segment out the concept and component in each reference and evaluation image, and then eliminate the target component from the segmented concept. As we can see in Tab. 1, component-controllable personalization remains a tough task even for SOTA methods of personalization. By comparison, MagicTailor achieves the best results in both identity fidelity and text alignment. It should be credited to the effective framework tailored to this special task.

Compatibility with Different Backbones. MagicTailor can also collaborate with other T2I diffusion models as it is a model-independent approach. In Fig. 7, we employ MagicTailor in other backbones like SD 1.5 [Rombach et al., 2022] and SDXL [Podell et al., 2023], showcasing MagicTailor can also achieve remarkable results. Notably, we directly use the original hyperparameter values without further selections, showing the generalizability of MagicTailor.

Robustness on Loss Weights. In Fig. 8, we analyze the sensitivity of loss weights in Eq. 8 (i.e., λpres and λattn), since loss weights are often critical for model training. As we can see, when λpres and λattn vary within a reasonable range, our MagicTailor can consistently attain SOTA performance, revealing its robustness on these hyperparameters.

User Study. We further evaluate the methods with a user study. Specifically, a detailed questionnaire is designed to display 20 groups of evaluation images with the corresponding text prompt and reference images. Users are asked to select the best result in each group for three aspects, including text alignment, identity fidelity, and generation quality. Finally, we collect a total of 3,180 valid answers and report the selected rates in Tab. 2. It can be observed that MagicTailor can also achieve superior performance in human preferences, further verifying its effectiveness.

Performance on Different Numbers of Reference Images. In Fig. 9, we reduce the number of reference images to analyze the performance variation. With fewer reference images, MagicTailor can still show satisfactory results. While more reference images could lead to better generalization ability, one reference image per concept/component is enough to obtain a decent result with our MagicTailor.

#### 4.4 Ablation Studies and Analysis

We conduct comprehensive ablation studies and analysis for MagicTailor to verify its capability. More ablation studies and analysis are included in Appendix C.

Generalizability to Complex Prompts. In comparisons, we have used well-categorized text prompts for systemic evaluation. Here we further evaluate MagicTailor’s performance on other complex text prompts involving more complicated contexts. As shown in Fig. 11, MagicTailor effectively generates text-aligned images when performing fidelity personalization, showing its ability to handle diverse user needs.

Effectiveness of Key Techniques. In Tab. 3, we investigate two key techniques by starting from a baseline framework described in Sec. 3.1. Even without DM-Deg and DS-Bal, such a baseline framework can still have competitive performance,

- (a) Decoupled Genera on
- (b) Controlling Mul ple Components

###### (c) Enhancing Other Genera ve Tools

###### Reference

###### Generated Images

Reference Edge Condi on Generated Image Reference Style Image Generated Image

|[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]|
|---|

|[Figure 228]|
|---|

[Figure 229]

<person><eye>

<person><hair>

<tower><roof>

|[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]|
|---|

[Figure 233]

[Figure 234]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

|[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]|
|---|

|[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]|
|---|

“<person> with <eye>,  watercolor painting”

“<person> dresses like batman”

“<eye> in the cover  of a wizard book”

###### MagicTailor + CSGO

MagicTailor + ControlNet

Reference Generated Image 3D Mesh

|[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]|
|---|

|[Figure 244]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 245]

[Figure 246]

###### Reference Reference

###### Generated Image Generated Image

<person><beard>

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]|
|---|
|[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]|
|[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|

|[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]|
|---|
|[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]|
|[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]|

<hair><eye><person>

<hair><person>

|[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]|
|---|

<eye>

“<person> with <hair> and <eye>,  made of clay”

“<person> with <hair> and <eye>,  with clouds in the background”

MagicTailor + InstantMesh

- Figure 10: Further applications of MagicTailor. (a) Decoupled generation: MagicTailor can also separately generate the target concept and component, enriching prospective combinations. (b) Controlling multiple components: MagicTailor shows the potential to handle more than one component, highlighting its effectiveness. (c) Enhancing other generative tools: MagicTailor can seamlessly integrate with various generative tools, adding the capability to control components within their generation pipelines.

Reference Generated Images

“<dog> with <ear>,  joyful, running,  on a sunny day”

<dog><ear>

|[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]|
|---|

|[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]|
|---|

|[Figure 276]|
|---|

“<dog> with <ear>,  calm, lying down, On a wooden floor”

“<dog> with <ear>,  holding a red ball in the  mouth, on the bench”

|[Figure 277]|
|---|

|[Figure 278]|
|---|

<person><beard>

|[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]|
|---|

|[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]|
|---|

“<person> with <beard>,  focused, playing guitar, in an outdoor field”

“<person> with <beard>,  grabbing a cup, smiling, in a coffee shop”

“<person> with <beard>,  looking serious, wearing glasses, in a library”

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

- Figure 11: Generalizability for complex prompts. We present qualitative results generated with complex text prompts. In addition to those well-categorized text prompts, our MagicTailor can also follow more complex ones to generate text-aligned images.

###### Reference Generated Images

###### Reference Generated Images

|[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]|
|---|

|[Figure 291]|[Figure 292]|
|---|---|
| |[Figure 293]|

|[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]|
|---|

|[Figure 300]|[Figure 301]|
|---|---|
| |[Figure 302]|

<person><ear>

<person><hair>

|[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]|
|---|

|[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]|
|---|

“<person> with <ear>”

“<person> with <hair>”

Figure 12: Generalizability for difficult pairs. We show the results of two hard cases involving large geometric discrepancy and cross-domain interactions, showing that MagicTailor can effectively handle such challenging scenarios.

cause such a setting is enough to cover extensive scenarios, and can be further extended to reconfigure multiple components with an iterative procedure. However, as shown in Fig. 10(b), our MagicTailor also exhibits the potential to control two components simultaneously. Handling more components remains a prospective direction of exploring better control over diverse elements for a single concept.

Enhancing Other Generative Tools. We demonstrate how MagicTailor enhances other generative tools like ControlNet [Zhang et al., 2023], CSGO [Xing et al., 2024], and InstantMesh [Xu et al., 2024] in Fig. 10(c). MagicTailor can integrates seamlessly, furnishing them with an additional ability to control the concept’s component in their pipelines. For instance, working with MagicTailor, InstantMesh can conveniently achieve fine-grained 3D mesh design, exhibiting the practicability of MagicTailor in more creative applications.

Generalizability to Difficult Pairs. We further evaluate MagicTailor’s performance on challenging pairs, focusing on two cases: 1) large geometric discrepancy, such as “<person>” in an upper body portrait and “<hair>” in a profile photo, and 2) cross-domain interactions, such as “<person>” and “<ear>” of dogs. As shown in Fig. 12, even facing these hard cases, MagicTailor can still effectively personalize target concepts and components with high fidelity.

#### 4.5 Further Applications

### 5 Conclusion

Decoupled Generation. After learning from a conceptcomponent pair, MagicTailor can also enable decoupled generation. As shown in Fig. 10(a), MagicTailor can generate the target concept and component separately in various and even cross-domain contexts. This should be credited to its remarkable ability to capture different-level visual semantics. Such an ability extends the flexibility of the possible combination between the concept and component.

We introduce component-controllable personalization, enabling precise customization of individual components within concepts. The proposed MagicTailor uses Dynamic Masked Degradation (DM-Deg) to suppress unwanted semantics and Dual-Stream Balancing (DS-Bal) to ensure balanced learning. Experiments show that MagicTailor sets a new standard in this task, with promising creative applications. In the future, we would like to extend our approach to broader image and video generation, enabling finer control over multi-level visual semantics for creative generation capabilities.

Controlling Multiple Components. In this paper, we focus on personalizing one concept and one component, be-

### Acknowledgments

We would like to thank Pengzhi Li, Tian Ye, Jinyu Lin, and Jialin Gao for their valuable discussion and suggestions. This study was supported by the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government via the Hong Kong Centre for Logistics Robotics.

### References

Devansh Arpit, Stanislaw Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. In Int. Conf. Mach. Learn., pages 233–242, 2017.

Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia, pages 1–12, 2023.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In Adv. Neural Inf. Process. Syst., pages 19822–19835, 2021.

Hugging Face. Diffusers: State-of-the-art diffusion models for image and audio generation in pytorch and flax., 2022.

Aosong Feng, Weikang Qiu, Jinbin Bai, Kaicheng Zhou, Zhen Dong, Xiao Zhang, Rex Ying, and Leandros Tassiulas. An item is worth a prompt: Versatile image editing with disentangled control. arXiv preprint arXiv:2403.04880, 2024.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In Int. Conf. Learn. Represent., 2022.

Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. In Adv. Neural Inf. Process. Syst., 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Adv. Neural Inf. Process. Syst., pages 6840–6851, 2020.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jiancheng Huang, Yi Huang, Jianzhuang Liu, Donghao Zhou, Yifan Liu, and Shifeng Chen. Dual-schedule inversion: Training-and tuning-free inversion for real image editing. arXiv preprint arXiv:2412.11152, 2024.

Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525, 2024.

Zehuan Huang, Hongxing Fan, Lipeng Wang, and Lu Sheng. From parts to whole: A unified reference framework for controllable human image generation. arXiv preprint arXiv:2404.15267, 2024.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Jimyeong Kim, Jungwon Park, and Wonjong Rhee. Selectively informative description can reduce undesired embedding entanglements in text-to-image personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8312–8322, 2024.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 1931–1941, 2023.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. arXiv preprint arXiv:2312.04461, 2023.

Pengzhi Li, Qiang Nie, Ying Chen, Xi Jiang, Kai Wu, Yuhuan Lin, Yong Liu, Jinlong Peng, Chengjie Wang, and Feng Zheng. Tuning-free image customization with image and text guidance. arXiv preprint arXiv:2403.12658, 2024.

Jingyu Lin, Guiqin Zhao, Jing Xu, Guoli Wang, Zejin Wang, Antitza Dantcheva, Lan Du, and Cunjian Chen. Difftv: Identity-preserved thermal-to-visible face translation via feature alignment and dual-stage conditions. In ACM Int. Conf. Multimedia, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. arXiv preprint arXiv:2402.02583, 2024.

Kam Woh Ng, Xiatian Zhu, Yi-Zhe Song, and Tao Xiang. Partcraft: Crafting creative objects by parts. In European Conference on Computer Vision, pages 420–437. Springer, 2025.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Int. Conf. Mach. Learn., pages 8748–8763,

- 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

- 2022.

Meisam Razaviyayn, Tianjian Huang, Songtao Lu, Maher Nouiehed, Maziar Sanjabi, and Mingyi Hong. Nonconvex min-max optimization: Applications, challenges, and recent theoretical advances. IEEE Signal Process. Mag., 37(5):55–66, 2020.

Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In Int. Conf. Mach. Learn., pages 1060–1069, 2016.

Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 22500–22510, 2023.

Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2022.

Mehdi Safaee, Aryan Mikaeili, Or Patashnik, Daniel CohenOr, and Ali Mahdavi-Amiri. Clic: Concept learning in context. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 6924–6933, 2024.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Adv. Neural Inf. Process. Syst., 35:36479–36494, 2022.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Kunpeng Song, Yizhe Zhu, Bingchen Liu, Qing Yan, Ahmed Elgammal, and Xiao Yang. Moma: Multimodal llm adapter for fast personalized image generation. In European Conference on Computer Vision, pages 117–132. Springer, 2024.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Adv. Neural Inf. Process. Syst., 2017.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Zeyu Wang, Jingyu Lin, Yifei Qian, Yi Huang, Shicen Tian, Bosong Chai, Juncan Deng, Lan Du, Cunjian Chen, Yufei Guo, et al. Diffx: Guide your layout to cross-modal generative modeling. arXiv preprint arXiv:2407.15488, 2024.

Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Int. Conf. Comput. Vis., pages 15943–15953, 2023.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.

Guangxuan Xiao, Tianwei Yin, William T. Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.

Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 22428–22437, 2023.

Shaoan Xie, Yang Zhao, Zhisheng Xiao, Kelvin CK Chan, Yandong Li, Yanwu Xu, Kun Zhang, and Tingbo Hou. Dreaminpainter: Text-guided subject-driven image inpainting with diffusion models. arXiv preprint arXiv:2312.03771, 2023.

Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024.

Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 1316–1324, 2018.

Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.

Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. In Adv. Neural Inf. Process. Syst., 2024.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., pages 3836–3847, 2023.

Xulu Zhang, Xiao-Yong Wei, Wengyu Zhang, Jinlin Wu, Zhaoxiang Zhang, Zhen Lei, and Qing Li. A survey on personalized content synthesis with diffusion models. arXiv preprint arXiv:2405.05538, 2024.

Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024.

Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In IEEE Conf. Comput. Vis. Pattern Recognit., pages 22490–22499, 2023.

### A More Details of Experimental Setup

#### A.1 Dataset

As there is no existing dataset specifically for componentcontrollable personalization, we curate a dataset from the internet to conduct experiments. Particularly, unlike previous works [Ruiz et al., 2023; Kumari et al., 2023] that focus on very few categories of concepts, the dataset contains concepts and components from various domains, such as characters, animation, buildings, objects, and animals. Overall, the dataset consists of 23 concept-component pairs totally with 138 reference images, where each concept/component contains 3 reference images and a corresponding category label. It is worth noting that the scale of this dataset is aligned with the scale of those datasets used in the compared methods [Gal

- et al., 2022; Ruiz et al., 2023; Kumari et al., 2023; Avrahami
- et al., 2023; Safaee et al., 2024].

#### A.2 Implementation

We utilize SD 2.1 [Rombach et al., 2022] as the pretrained T2I diffusion model. As commonly done, the resolution of reference images is set to 512 × 512. Besides, the LoRA rank and alpha are set to 32. To simplify concept learning, we exclude the region of the target component from the segmentation masks of the target concept, e.g., remove the hair from the person in a “<person> + <hair>” pair. For the warm-up and DS-Bal stage, we set the learning rate to 1e-4 and 1e-5 and the training steps to 200 and 300. Moreover, the learning rate is further scaled by the batch size, which is set to completely contain a concept-component pair. For the cross-attention loss, we follow [Avrahami et al., 2023] to average the corresponding cross-attention maps at resolution 16 × 16 and normalized them to [0, 1]. The model is trained with an AdamW [Loshchilov and Hutter, 2017] optimizer and a DDPM [Ho et al., 2020] sampler. As done in [Avrahami et al., 2023], the tensor precision is set to float16 to accelerate training. For a fair comparison, all random seeds are fixed at 0, and all compared methods use the same implementation above except for method-specific configurations.

#### A.3 Evaluation

To generate images for evaluation, we carefully design 20 text prompts covering extensive situations, which are listed in Tab. 4. These text prompts can be divided into four aspects, including recontextualization, restylization, interaction, and property modification, where each aspect is composed of 5 text prompts. In recontextualization, we change the contexts to different locations and periods. In restylization, we transfer concepts into various artistic styles. In interaction, we explore the spatial interaction with other concepts. In property modification, we modify the properties of concepts in rendering, views, and materials. Such a group of diverse text prompts allows us to systemically evaluate the generalization capability of a method. We generate 32 images per text prompt for each pair, using a DDIM [Song et al., 2020] sampler with 50 steps and a classifier-free guidance scale of 7.5. To ensure fairness, we fix the random seed within the range of [0, 31] across all methods. This process results in 14,720 images for each method to be evaluated, ensuring a thorough comparison.

#### A.4 Automatic Metrics

We utilize four automatic metrics in the aspects of text alignment (CLIP-T [Gal et al., 2022]) and identity fidelity (CLIP-I [Radford et al., 2021], DINO [Oquab et al., 2023], DreamSim [Fu et al., 2023]). To precisely measure identity fidelity, we improve the traditional measurement approach for personalization. This is because a reference image of the target concept/component could contain an undesired component/concept that is not expected to appear in evaluation images. Specifically, we use Grounded-SAM [Ren et al., 2024] to segment out the concept and component in each reference and evaluation image. Then, we further eliminate the target component from the segmented concept as we have done during training. Such a process is similar to the one adopted in [Avrahami et al., 2023]. As a result, using the segmented version of evaluation images and reference images, we can accurately calculate the metrics of identity fidelity.

#### A.5 User Study

We further evaluate the methods with a user study. Specifically, we design a questionnaire to display 20 groups of evaluation images generated by our method and other methods. Besides, each group also contains the corresponding text prompt and the reference images of the concept and component, where we adopt the same text prompts that are used to calculate CLIP-T. The results of our method and all the compared methods are presented on the same page. Clear rules are established for users to evaluate in three aspects, including text alignment, identity fidelity, and generation quality. Users are requested to select the best result in each group by answering the corresponding questions of these three aspects. We hide all the method names and randomize the order of methods to ensure fairness. Finally, 3,180 valid answers are collected for a sufficient evaluation of human preferences.

#### A.6 Compared Methods

In our experiments, we compare MagicTailor with SOTA methods in the domain of personalization, including Textual Inversion (TI) [Gal et al., 2022], DreamBooth-LoRA (DB) [Ruiz et al., 2023], Custom Diffusion (CD) [Kumari et al., 2023], Break-A-Scene (BAS) [Avrahami et al., 2023], and CLiC [Safaee et al., 2024]. We select these methods because TI, DB, and CD are three representatives of personalization frameworks and BAS and CLiC are highly relevant to learning fine-grained elements from reference images. For TI, DB, and CD, we use the third-party implementation in Diffusers [Face, 2022]. For BAS, we use the official implementation. For CLiC, we reproduce it following the resource paper as the official code is not released. Unless otherwise specified, method-specific configurations are set up by following their resource papers or Diffusers. We empirically adjust the learning rate of CD and CLiC to 1e-4 and 5e-5 respectively, because they perform very poorly with the original learning rates. For a fair and meaningful comparison, these methods should be adapted to our task setting with minimal modification. Therefore, for those methods adopting a vanilla diffusion loss, we integrate the masked diffusion loss into them while using the same segmentation masks from MagicTailor.

- Table 4: Text prompts used to generate evaluation images. These text prompts can be divided into four aspects: recontextualization, restylization, interaction, and property modification, covering extensive situations to systemically evaluate the method’s generalizability. Note that “<placeholder>” will be replaced by the combination of pseudo-words (e.g., “<tower> with <roof>”) when generating evaluation images, and will be replaced by the combination of category labels (e.g., “tower with roof”) when calculating the metric of text alignment.

|Recontextualization|Restylization<br><br>|
|---|---|
|“<placeholder>, on the beach” “<placeholder>, in the jungle” “<placeholder>, in the snow” “<placeholder>, at night” “<placeholder>, in autumn”|“<placeholder>, watercolor painting”<br><br>“<placeholder>, Ukiyo-e painting” “<placeholder>, in Pixel Art style” “<placeholder>, in Von Gogh style”<br><br>“<placeholder>, in a comic book”|

|Interaction<br><br>|Property Modification|
|---|---|
|“<placeholder>, with clouds in the background” “<placeholder>, with flowers in the background” “<placeholder>, near the Eiffel Tower” “<placeholder>, on top of water” “<placeholder>, in front of the Mount Fuji”|“<placeholder>, from 3D rendering”<br><br>“<placeholder>, in a far view”<br><br>“<placeholder>, in a close view”<br><br>“<placeholder>, made of clay”<br><br>“<placeholder>, made of plastic”|

### B Additional Comparisons

#### B.1 Detailed Text-Guided Generation

One might wonder if component-controllable personalization can be accomplished by providing detailed textual descriptions to the T2I model. To investigate this, we separately feed the reference images of the concept and component into GPT4o [Hurst et al., 2024] to obtain detailed textual descriptions for them. The text prompt we used is “Please detailedly describe the <concept/component> of the upload images in a parapraph”, where “<concept/component>” is replaced with the category label of the concept or component. Then, we ask GPT-4o to merge these textual descriptions using natural language, and input them into the Stable Diffusion 2.1 [Rombach et al., 2022] to generate the corresponding images. Some examples for a qualitative comparison are shown in Fig. 13. As we can see, such an approach cannot achieve satisfactory results, because it is hard to guarantee that visual semantics can be completely expressed by using the combination of text tokens. In contrast, our MagicTailor is able to accurately learn the desired visual semantics of the concept and component from reference images, and thus lead to consistent and excellent generation in this tough task.

#### B.2 Commercial Models

It is also worth exploring whether existing commercial models, which can understand and somehow generate both text and images by themselves or other integrated tools, are capable of handling component-controllable personalization. We choose two widely recognized commercial models, GPT4o [Hurst et al., 2024] and Gemini 1.5 Flash [Team et al., 2023], for a qualitative comparison. First, we separately feed the reference images of the concept and component into them, along with the text prompt of “The uploaded images contain a special instance of the <concept/component>, please mark it as #<concept/component>”, where “<concept/component>” is replaced with the category label of the concept or component. Then, we instruct them to perform image generation, using the text prompt of “Please generate images containing #<concept> with #<component>”, where

“<concept>” and “<component>” are replaced with the category label of the concept and component, respectively. As shown in Fig. 14, these models struggle to reproduce the given concept, let alone reconfigure the concept’s component. Whereas, our MagicTailor achieves superior results in component-controllable personalization, using a dedicated framework designed for this task. It demonstrates that, even though large commercial models are able to tackle multiple general tasks, there is also plenty of room for the community to explore specialized tasks for real-world applications.

### C More Ablation Studies and Analysis

#### C.1 Dynamic Intensity Matters

- In Tab. 5, we explore DM-Deg by comparing it with 1) maskout strategy; 2) fixed intensity; 3) linear intensity (α goes from 1 to 0, or from 0 to 1); and 4) dynamic intensity with different γ. First, the terrible performance of the mask-out strategy verifies that it is not a valid solution for semantic pollution. Notably, the descent linear intensity shows better identity fidelity than its ascent counterpart, which aligns with and validates our observations on noise memorization. Moreover, the dynamic intensity generally shows better results, and it can achieve better overall performance with a proper γ.

C.2 Momentum Denoising U-Net as a Good Regularizer

- In Tab. 6, we study DS-Bal by modifying the U-Net for regularization as 1) fixed U-Net with β = 0 (i.e., the one just after warm-up); 2) fixed U-Net with β = 1 (i.e., the one from the last step); and 3) momentum U-Net with other β. The results show that employing the U-Net with a high momentum rate can yield better regularization to tackle semantic imbalance, thus leading to excellent performance.

#### C.3 Necessity of Warm-Up Training

In MagicTailor, we start with a warm-up phase for the T2I model to preliminarily inject the knowledge for the subsequent phase of DS-Bal. Here we investigate the necessity of

##### Reference Detailed Textual Descrip ons Detailed-Text-Guided Genera on MagicTailor (Ours)

|[Figure 309]|[Figure 310]|
|---|---|
| |[Figure 311]|

|[Figure 312]|[Figure 313]|
|---|---|
| |[Figure 314]|

|[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]|
|---|

“The individual has a distinct  appearance, marked by fair skin, a wellgroomed beard, and intense blue eyes …  His hair is short and styled in a colorful  gradient, transitioning from blue … to  purple … The fringe is textured with  soft waves, and the closely cropped  sides emphasize the vibrant color …”

<person><hair>

|[Figure 318]<br><br>[Figure 319]<br><br>[Figure 320]|
|---|

|[Figure 321]|[Figure 322]|
|---|---|
| |[Figure 323]|

|[Figure 324]|[Figure 325]|
|---|---|
| |[Figure 326]|

|[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]|
|---|

“The tower combines medieval and East  Asian architectural styles. Its base  features pale stone with Romanesque  arches … The East Asian-inspired roof  has multiple terracotta-tiled tiers,  with upward-curving eaves … This  blend of styles creates a visually  striking and harmonious structure …”

<tower><roof>

|[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]|
|---|

- Figure 13: Comparing with detailed-text-guided generation. We use GPT-4o to generate and merge detailed textual descriptions for the target concept and component, which are fed into Stable Diffusion 2.1 to conduct text-to-image generation. This paradigm cannot perform well and produce inconsistent images, while MagicTailor can achieve faithful and consistent generation.

- Table 5: Ablation of DM-Deg. We compare DM-Deg with its variants and the mask-out strategy. Our DM-Deg attains superior overall performance on text alignment and identity fidelity.

- Table 6: Ablation of DS-Bal. We compare DS-Bal with potential variants, showing its excellence.

U-Net Variants CLIP-T ↑ CLIP-I ↑ DINO ↑ DreamSim ↓

Fixed (β = 0) 0.268 0.850 0.803 0.293 Fixed (β = 1) 0.270 0.851 0.808 0.286 Momentum (β = 0.5) 0.268 0.850 0.805 0.290 Momentum (β = 0.9) 0.269 0.850 0.808 0.288 Momentum (Ours) 0.270 0.854 0.813 0.279

- Table 7: Ablations of warm-up. We compare MagicTailor with the variant that removes warm-up. The results exhibit the significance of the warm-up stage for the framework of MagicTailor.

Intensity Variants CLIP-T ↑ CLIP-I ↑ DINO ↑ DreamSim ↓ Mask-Out Startegy 0.270 0.818 0.760 0.375 Fixed (α = 0.4) 0.270 0.849 0.800 0.297 Fixed (α = 0.6) 0.271 0.845 0.794 0.310 Fixed (α = 0.8) 0.271 0.846 0.796 0.305 Linear (Ascent) 0.270 0.846 0.797 0.307 Linear (Descent) 0.261 0.851 0.802 0.300 Dynamic (γ = 8) 0.266 0.850 0.806 0.289 Dynamic (γ = 16) 0.268 0.854 0.813 0.282 Dynamic (γ = 64) 0.271 0.852 0.812 0.283 Dynamic (Ours) 0.270 0.854 0.813 0.279

Warm-Up Variants CLIP-T ↑ CLIP-I ↑ DINO ↑ DreamSim ↓

w/o Warm-Up 0.272 0.844 0.793 0.320 w/ Warm-Up (Ours) 0.270 0.854 0.813 0.279

such a warm-up phase for generation performance. In Tab. 7, when removing the warm-up phase, even though MagicTailor could obtain slight improvement in text alignment, it severely suffers from the huge drop in identity fidelity. This is because such a scheme makes it difficult to construct a decent momentum denoising U-Net for DS-Bal. Whereas integrated with a warm-up phase, MagicTailor can achieve superior overall performance due to the knowledge reserved from warm-up.

approach cannot address semantic pollution well, where unwanted visual semantics still affect the personalized concept. In contrast, DM-Deg effectively prevents semantic pollution by dynamically perturbing those undesired visual semantics, verifying its remarkable significance in this task.

#### C.5 Robustness on Linking Words

#### C.4 Effectiveness of DM-Deg over Using Informative Training Prompts

Generally, we use “with” to link the pseudo-words of the concept and component in a text prompt, e.g., “<person> with <beard>, in Von Gogh style”. Here we evaluate the robustness of our method on different linking words. We choose several words, which are commonly used to indicate ownership or association, to construct text prompts and then feed them into the same fine-tuned T2I model. As shown in Fig. 16, the generation performance of our MagicTailor remains robust regardless of the linking word used, exhibiting its flexibility to textual descriptions.

One might be curious about whether it is not necessary to employ the proposed DM-Deg, but perhaps to use informative text prompts during training to provide textual prior knowledge for learning the target concept and component. To investigate this, we use Selectively Informative Description (SID) [Kim et al., 2024] with GPT-4o [Hurst et al., 2024] to construct text prompts for the target concept and component, and then use them for training. As shown in Fig. 15, such an

##### Reference GPT-4o Gemini

##### MagicTailor (Ours)

|[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]|
|---|

|[Figure 336]|[Figure 337]|
|---|---|
| |[Figure 338]|

|[Figure 339]|[Figure 340]|
|---|---|
| |[Figure 341]|

|NOT APPLICABLE|NOT APPLICABLE|
|---|---|
| |NOT APPLICABLE|

<person><hair>

|[Figure 342]<br><br>[Figure 343]<br><br>[Figure 344]|
|---|

|[Figure 345]<br><br>[Figure 346]<br><br>[Figure 347]|
|---|

|[Figure 348]|[Figure 349]|
|---|---|
| |[Figure 350]|

|[Figure 351]|[Figure 352]|
|---|---|
| |[Figure 353]|

|[Figure 354]|[Figure 355]|
|---|---|
| |[Figure 356]|

<dog><ear>

|[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]|
|---|

- Figure 14: Comparing with commercial models. We input the reference images of the target concept and component to GPT-4o and Gemini, along with structured text prompts, for conducting image generation. Even though capable of handling multiple general tasks, these models still fall short in this task. In contrast, our MagicTailor performs well using a dedicated framework.

|[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]|
|---|

|[Figure 363]<br><br>[Figure 364]<br><br>[Figure 365]|
|---|

Reference

<person><eye>

Baseline DM-Deg (Ours)

|[Figure 366]|
|---|

|[Figure 367]|
|---|

<person><hair>

|[Figure 368]<br><br>[Figure 369]<br><br>[Figure 370]|
|---|

|[Figure 371]<br><br>[Figure 372]<br><br>[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

“<person> with <eye>”

SID

“<person> with <hair>”

- Figure 15: Ablation of DM-Deg via replacement with SID. We compare our DM-Deg with SID [Kim et al., 2024] that aims to produce informative prompts for training. Besides, we also present baseline (i.e., removing DM-Deg from MagicTailor) results for reference. This comparison indicates the effectiveness of our DM-Deg in addressing semantic pollution.

Reference … and … … including … … containing … … with …

|[Figure 378]<br><br>[Figure 379]<br><br>[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

<hair><person>

|[Figure 385]<br><br>[Figure 386]<br><br>[Figure 387]|
|---|

“<person> … <hair>, laughing”

|[Figure 388]<br><br>[Figure 389]<br><br>[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

<bottle><lid>

|[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]|
|---|

“<bottle> … <lid>, on a dinner table”

Figure 16: Ablation of linking words. We present qualitative results generated with different linking words in text prompts, demonstrating the robustness of MagicTailor.

this tough task. In summary, the proposed MagicTailor effectively addresses both semantic pollution and semantic imbalance through its innovative techniques, DM-Deg and DS-Bal, respectively. These advancements demonstrate its superiority in handling complex visual semantics and achieving remarkable performance in this challenging task.

### D More Qualitative Results

In Fig. 17 & Fig. 18, we provide more evaluation images for a substantial qualitative comparison. It can be clearly observed that semantic pollution remains an intractable problem for these compared methods. This is due to the leak of an effective mechanism to alleviate the T2I model’s perception for these semantics. To address this, our MagicTailor utilizes DM-Deg to dynamically perturb undesired visual semantics during the learning phase, and thus achieve better performance. On the other hand, the compared methods are also severely influenced by semantic imbalance, resulting in overemphasis or even overfitting on the concept or component. This is because the inherent imbalance of visual semantics complicates the learning process. In response to this issue, our MagicTailor applies DS-Bal to balance the learning of visual semantics, effectively showcasing its prowess in

|[Figure 398]<br><br>[Figure 399]<br><br>[Figure 400]|
|---|

|[Figure 401]|
|---|

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

<person><hair>

|[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]|
|---|

“<person> with <hair>, near the Eiffel Tower”

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

|[Figure 414]|
|---|

|[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]|
|---|

|[Figure 418]|
|---|

<dog><ear>

|[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]|
|---|

“<dog> with <ear>, in autumn”

|[Figure 422]|
|---|

|[Figure 423]|
|---|

|[Figure 424]|
|---|

|[Figure 425]|
|---|

|[Figure 426]|
|---|

|[Figure 427]|
|---|

|[Figure 428]<br><br>[Figure 429]<br><br>[Figure 430]|
|---|

<person><eye>

|[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]|
|---|

“<person> with <eye>, watercolor painting”

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

|[Figure 439]|
|---|

|[Figure 440]<br><br>[Figure 441]<br><br>[Figure 442]|
|---|

<person><beard>

|[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]|
|---|

“<person> with <beard>, in a comic book”

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

|[Figure 450]|
|---|

|[Figure 451]|
|---|

|[Figure 452]|
|---|

|[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]|
|---|

<person><hair>

|[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]|
|---|

“<bottle> with <lid>, on top of water”

|[Figure 459]|
|---|

|[Figure 460]|
|---|

|[Figure 461]|
|---|

|[Figure 462]|
|---|

|[Figure 463]|
|---|

|[Figure 464]|
|---|

|[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]|
|---|

<person><hair>

|[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]|
|---|

“<person> with <hair>, in front of the Mount Fuji”

|[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]|
|---|

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

|[Figure 477]|
|---|

|[Figure 478]|
|---|

|[Figure 479]|
|---|

<person><hair>

|[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]|
|---|

“<person> with <hair>, Ukiyo-e painting”

|[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]|
|---|

|[Figure 486]|
|---|

|[Figure 487]|
|---|

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

|[Figure 491]|
|---|

<person><hair>

|[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]|
|---|

“<person> with <hair>, at night”

|[Figure 495]|
|---|

|[Figure 496]|
|---|

|[Figure 497]|
|---|

|[Figure 498]|
|---|

|[Figure 499]|
|---|

|[Figure 500]|
|---|

|[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]|
|---|

<bottle><lid>

|[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]|
|---|

“<bottle> with <lid>, from 3D rendering”

|[Figure 507]|
|---|

|[Figure 508]|
|---|

|[Figure 509]|
|---|

|[Figure 510]|
|---|

|[Figure 511]|
|---|

|[Figure 512]|
|---|

|[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]|
|---|

<person><eye>

|[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]|
|---|

“<person> with <eye>, in Von Gogh style”

|[Figure 519]|
|---|

|[Figure 520]|
|---|

|[Figure 521]|
|---|

|[Figure 522]|
|---|

|[Figure 523]|
|---|

|[Figure 524]|
|---|

|[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]|
|---|

<tower><roof>

|[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]|
|---|

“<tower> with <roof>, Ukiyo-e painting”

|[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]|
|---|

|[Figure 534]|
|---|

|[Figure 535]|
|---|

|[Figure 536]|
|---|

|[Figure 537]|
|---|

|[Figure 538]|
|---|

|[Figure 539]|
|---|

<person><eye>

|[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]|
|---|

“<person> with <eye>, in the snow”

|[Figure 543]|
|---|

|[Figure 544]|
|---|

|[Figure 545]|
|---|

|[Figure 546]|
|---|

|[Figure 547]|
|---|

|[Figure 548]|
|---|

|[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]|
|---|

<person><eye>

|[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]|
|---|

“<person> with <eye>, with clouds in the background”

|[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]|
|---|

|[Figure 558]|
|---|

|[Figure 559]|
|---|

|[Figure 560]|
|---|

|[Figure 561]|
|---|

|[Figure 562]|
|---|

|[Figure 563]|
|---|

<person><hair>

|[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]|
|---|

“<person> with <hair>, in the front of the Mount Fuji”

