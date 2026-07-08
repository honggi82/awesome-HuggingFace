# arXiv:2403.15383v2[cs.CV]15May2024

## ThemeStation: Generating Theme-Aware 3D Assets from Few Exemplars

Zhenwei Wang∗

zhenwwang2-c@my.cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

Tengfei Wang†

tfwang@connect.ust.hk Shanghai AI Lab Shanghai, China

Gerhard Hancke

gp.hancke@cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

Ziwei Liu

ziwei.liu@ntu.edu.sg Nanyang Technological University Singapore

Rynson W.H. Lau†

rynson.lau@cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

[Figure 1]

Figure 1: ThemeStation can generate a gallery of 3D assets (right) from just one or a few exemplars (left). The synthesized models share consistent themes with the reference models, showing the immense potential of our approach for theme-aware 3D-to-3D generation and expanding the scale of existing 3D models. Code and video are at https://3dthemestation.github.io/.

∗Work done when interning at Shanghai Artificial Intelligence Laboratory. †Co-corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the

author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

### ABSTRACT

Real-world applications often require a large gallery of 3D assets that share a consistent theme. While remarkable advances have been made in general 3D content creation from text or image, synthesizing customized 3D assets following the shared theme of input 3D exemplars remains an open and challenging problem. In this work, we present ThemeStation, a novel approach for theme-aware 3D-to-3D generation. ThemeStation synthesizes customized 3D assets based on given few exemplars with two goals: 1) unity for generating 3D assets that thematically align with the given exemplars and 2) diversity for generating 3D assets with a high degree of variations. To this end, we design a two-stage framework that draws a concept image first, followed by a reference-informed 3D modeling stage. We propose a novel dual score distillation (DSD) loss to jointly leverage priors from both the input exemplars and the synthesized concept image. Extensive experiments and a user study confirm that ThemeStation surpasses prior works in producing diverse theme-aware 3D models with impressive quality. ThemeStation also enables various applications such as controllable 3D-to-3D generation.

### CCS CONCEPTS

• Computing methodologies → Computer vision.

### KEYWORDS

3D Generation, Exemplar-based

ACM Reference Format:

Zhenwei Wang, Tengfei Wang, Gerhard Hancke, Ziwei Liu, and Rynson W.H. Lau. 2024. ThemeStation: Generating Theme-Aware 3D Assets from Few Exemplars. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24 (SIGGRAPH Conference Papers ’24), July 27-August 1, 2024, Denver, CO, USA. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3641519.3657471

### 1 INTRODUCTION

In applications such as virtual reality or video games, we often need to create a large number of 3D models that are thematically consistent with each other while being different. For example, we may need to create an entire 3D gallery of buildings to form an ancient town or monsters to form an ecosystem in a virtual world. While it is easy for a highly trained craftsman to create one or a few coherent 3D models, it can be challenging and time-consuming to create a large 3D gallery. We consider if we can automate this laborintensive process, and whether a generative system can produce many unique 3D models that are different from each other while sharing a consistent style.

Recently, diffusion models [Ho et al. 2020] have revolutionized the 3D content creation task by significantly lowering the amount of manual work. This allows even beginners to create 3D assets from text prompts (i.e., text-to-3D) or reference images (i.e., image-to-3D) with minimal effort. Early works [Poole et al. 2023] focus on using well-trained image diffusion models to generate 3D assets from a text prompt with score distillation sampling (SDS). Subsequent

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0525-0/24/07...$15.00 https://doi.org/10.1145/3641519.3657471

works [Melas-Kyriazi et al. 2023; Tang et al. 2023b] extend this approach to enable 3D creation from a single image. While these methods have shown impressive performances, they still suffer from the 3D ambiguity and inconsistency problem due to the limited 3D information from the input modality.

To address these limitations, in this work, we propose to leverage 3D exemplars as input to guide the 3D generation process. Given one or a few exemplar 3D models as input (Fig. 1 left), we present ThemeStation, a novel approach for the theme-aware 3Dto-3D generation task, which aims to generate a diverse range of unique 3D models that are theme-consistent (i.e., semantically and stylistically the same) with the input exemplars while being different from each other. Compared to text prompts and images, 3D exemplars offer a richer source of information with respect to both geometry and appearance, reducing ambiguity in 3D modeling. This, in turn, makes it possible to create higher-quality 3D models. ThemeStation enables the automatic synthesis of, for example, a group of buildings/characters with a shared theme (Fig. 1 right). It aims to satisfy two goals in the 3D generation process: unity and diversity. For unity, we expect the generated models to align with the theme of the given exemplars. For diversity, we aim for the generated models to exhibit a high degree of variations.

However, we note that simply training a generative model on a few 3D exemplars [Wu et al. 2023; Wu and Zheng 2022] leads to only limited variation, primarily restricted to resizing the input models (to different scales and aspect ratios) or repeating them randomly (Fig. 6), without introducing significant modifications to the appearance of the generated models. To address this problem, we design a two-stage generative scheme to mimic the manual 3D modeling workflow of first drawing a concept art and then using a progressive 3D modeling process [Bob 2022; CGHero 2022]. In the first stage, we fine-tune a well-trained image diffusion model [Rombach et al. 2022] on rendered images of the given 3D exemplars to produce diverse concept images. Unlike previous fine-tuning techniques [Gal et al. 2022a; Ruiz et al. 2023] that are subject-driven, our goal is to personalize the pre-trained diffusion model with a specific theme to synthesize images with novel subjects. In the second stage, we convert the synthesized concept images into 3D models. Our setting differs from image-to-3D tasks in that (1) we only regard the concept images as intermediate outputs to provide rough guidance on the overall structure and appearance of the generated 3D models and (2) we take the input 3D exemplars as auxiliary guidance to provide additional geometry and multi-view appearance information. To leverage both the synthesized concept images and input 3D exemplars (also referred to as the reference models in this paper), we propose reference-informed dual score distillation (DSD) to guide the 3D modeling process using two diffusion models: one (Concept Prior) for enforcing content fidelity in concept image reconstruction, similar to [Raj et al. 2023], and the other (Reference Prior) for reconstructing multi-view consistent fine details from the exemplars. Instead of naively combining the two losses, which may lead to severe loss conflict, we apply the two priors based on the noise levels (denoising timesteps). While the concept prior is applied to high noise levels for guiding the global layout, the reference prior is applied to low noise levels for guiding low-level variations.

To evaluate our approach, we have collected a benchmark that contains stylized 3D models with varying complexity. As shown in Fig. 1, ThemeStation can produce a creative gallery of 3D assets conforming to the theme of the input exemplars. Extensive experiments and a user study show that ThemeStation can generate compelling and diverse 3D models with finer details, even with just a single input exemplar. ThemeStation also enables various applications, such as controllable 3D-to-3D generation, showing immense potential for generating creative 3D content and expanding the scale of existing 3D models. Our main contributions can be summarized as:

- • We propose ThemeStation, a two-stage framework for themeaware 3D-to-3D generation, which aims at generating novel 3D assets with unity and diversity given just one or a few 3D exemplars.
- • We make a first attempt to tackle the challenging problem of extending diffusion priors for 3D-to-3D content generation.
- • We introduce dual score distillation (DSD) to enable the joint usage of two conflicted diffusion priors for 3D-to-3D generation by applying the reference prior and concept prior at different noise levels.

- 2 RELATED WORK

- 2.1 3D Generative Models

Remarkable advancements have been made to generative adversarial networks (GANs) and diffusion models for image synthesis [Brock et al. 2018; Karras et al. 2019; Rombach et al. 2022; Saharia et al. 2022]. Many researchers have explored how to apply these methods to generate 3D geometries using different representations, such as point clouds [Nichol et al. 2023; Zhou et al. 2021], meshes [Nash et al. 2020; Pavllo et al. 2021] and neural fields [Chan

- et al. 2022; Erkoç et al. 2023; Niemeyer and Geiger 2021]. Recent works can further generate 3D textured shapes [Chen et al. 2023b; Gupta et al. 2023; Hong et al. 2024; Jun and Nichol 2023; Tang et al. 2024; Wang et al. 2023b]. These methods require a large 3D dataset for training, which limits their performance on in-the-wild generation.

2.2 Diffusion Priors for 3D Generation

Dreamfusion [Poole et al.2023] proposed to distill the score of image distribution from a pre-trained text-to-image (T2I) diffusion model and show promising results in text-to-3D generation. Subsequent works enhance the score distillation scheme [Poole et al. 2023] and achieve higher generative quality for text-to-3D generation [Chen

- et al. 2023a; Lin et al. 2023; Metzer et al. 2023]. Some recent works also apply the diffusion priors to image-to-3D generation [Chen
- et al. 2024; Melas-Kyriazi et al. 2023; Sun et al. 2023; Tang et al. 2023a,b]. To enhance multi-view consistency of the generated 3D content, some researchers seek to fine-tune the pre-trained image diffusion models with multi-view datasets for consistent multi-view image generation [Liu et al. 2023b,a; Long et al. 2023; Yichun et al. 2023]. Although diffusion priors have shown great potential for 3D content generation from text or image inputs, their applicability to 3D customization based on 3D exemplars is still an open and challenging problem.

- 2.3 Exemplar-Based Generation

The exemplar-based 2D image generation task has been widely explored [Avrahami et al. 2023; Gal et al. 2022b; Ruiz et al. 2023]. Recently, DreamBooth3D [Raj et al. 2023] fine-tunes pre-trained diffusion models with only a few images to achieve subject-driven text-to-3D generation but still suffers inconsistency due to the lack of 3D information from the input images. Another line of work takes 3D exemplars as input to generate 3D variations. For example, assembly-based methods [Chaudhuri et al. 2011; Kim et al. 2013; Schor et al.2019;Xu et al.2012;Zheng et al.2013] focus on retrieving compatible parts from a collection of 3D examples and organizing them into a target shape. Some methods extend the idea of 2D SinGAN [Shaham et al. 2019] to train a 3D generative model [Wu et al. 2023; Wu and Zheng 2022] with a single 3D exemplar. Some methods [Li et al. 2023] lift classic 2D patch-based frameworks to 3D generation without the need for offline training. While these methods support 3D variations of sizes and aspect ratios, they do not understand and preserve the semantics of the 3D exemplars. As a result, their results are primarily restricted to resizing, repeating, or reorganizing the input exemplars in some way (Fig. 6), which is different from our setting that aims to produce theme-consistent 3D variations.

3 APPROACH

Our framework is designed to follow the real-world workflow of 3D modeling by introducing a concept art design step before the 3D modeling process. As illustrated in Fig. 2, we first customize a pre-trained text-to-image (T2I) diffusion model to produce a series of concept images that share a consistent theme as the input exemplars, mimicking the concept art designing process in practice (Sec. 3.1). We then utilize an optimization-based method to lift each concept image to a final 3D model, following the practical modeling workflow of pushing a base primitive into a well-crafted 3D model (Sec. 3.2). To this end, we present novel dual score distillation (DSD) that leverages the priors of both the concept images and the exemplars in the optimization process (Sec. 3.3).

- 3.1 Theme-Driven Concept Image Generation

Concept image design is a visual tool to convey the idea and preview the final 3D model. It is usually the first step in the 3D modeling workflow and serves as a bridge between the designer and the modeler [Bob 2022; CGHero 2022]. Following this practice, in this stage, our goal is to generate a variety of concept images {𝒙𝑐} of a specific theme based on the input exemplars {𝒎𝑟 }, as shown in Fig. 2 top. While there are some existing works on subject-driven image generation [Gal et al. 2022a; Ruiz et al. 2023], which fine-tune a pre-trained T2I diffusion model [Rombach et al. 2022] to generate novel contexts for a specific (exactly the same) subject, they are not aligned with our theme-driven setting. Our goal is to generate a diverse set of subjects that exhibit thematic consistency but display content variations relative to the exemplars. Thus, instead of stimulating the subject retention capability of the pre-trained diffusion model through overfitting the inputs, we seek to preserve its imaginative capability while preserving the theme of the input exemplars.

[Figure 2]

Figure 2: Overview of ThemeStation. Given just one or a few reference models, our approach can generate theme-consistent 3D models in two stages. In the first stage, we fine-tune a pre-trained text-to-image (T2I) diffusion model to form a customized theme-driven diffusion model that produces various concept images. In the second stage, we conduct reference-informed 3D asset modeling by progressively optimizing a rough initial model (omitted in this figure for brevity), which is obtained using an off-the-shelf image-to-3D method given the concept image, into a final 3D asset. We use a novel dual score distillation (DSD) loss for optimization, which applies concept prior and reference prior at different noise levels (denoising timesteps).

We observe that the diffusion model, fine-tuned with fewer iterations on the rendered images {𝒙𝑟 } of the input exemplars {𝒎𝑟 }, is already able to learn the theme of the exemplars. Hence, it is able to generate novel subjects that are thematically in line with the input exemplars. To further disentangle the theme (semantics and style) and the content (subject) of the exemplars, we explicitly indicate the learning of the theme using a shared text prompt across all exemplars, e.g., “a 3D model of an owl, in the style of [V]”, during the fine-tuning process.

### 3.2 Reference-Informed 3D Asset Modeling

Given one synthesized concept image 𝒙𝑐 and the input exemplars {𝒎𝑟 }, we conduct reference-informed 3D asset modeling in the second stage. Similar to the workflow of practical 3D modeling that starts with a base primitive, we begin with a rough initial

- 3D model 𝒎𝑖𝑛𝑖𝑡, generated using off-the-shelf image-to-3D techniques [Liu et al. 2023c,a; Long et al. 2023] given the concept image

𝒙𝑐, to accelerate our 3D asset modeling process. As the synthesized concept image, along with the initial 3D model, may have inconsistent spatial structures and unsatisfactory artifacts, we do not enforce our final generated model to be strictly aligned with the concept image. We then take the concept image and the initial model as intermediate outputs and meticulously develop the initial model into the final generated 3D model 𝒎𝑜. Different from previous optimization-based methods that perform score distillation sampling using a single diffusion model [Poole et al. 2023; Wang et al. 2023a], we propose a dual score distillation (DSD) loss to leverage two diffusion priors as guidance simultaneously. Here, one diffusion model, denoted as 𝜙𝑐, functions as the basic concept (concept prior), providing diffusion priors from the concept image 𝒙𝑐 to ensure concept reconstruction, while the other, denoted as

[Figure 3]

#### Figure 3: Comparison of the key ideas between image style transfer (top) and our dual score distillation (bottom). Images are from Gatys et al. [2016] (top) and Dibia [2022] (bottom).

𝜙𝑟, operates as an advisory reference (reference prior), generating diffusion priors pertinent to the input reference models {𝒎𝑟 } to assist with restoring subtle features and alleviating multi-view inconsistency. We further present a clear design of our DSD loss in Sec. 3.3.

### 3.3 Dual Score Distillation

In this subsection, we elaborate on the critical component of our approach, dual score distillation (DSD) for theme-aware 3D-to-3D generation. DSD combines the best of both priors, concept prior and reference prior, to guide the generation process. Both priors are derived through fine-tuning a pre-trained T2I diffusion model. Next, we discuss the preliminaries and show the steps of learning the two priors and the design of DSD loss.

- 3.3.1 Preliminaries. DreamFusion achieves text-to-3D generation by optimizing a 3D representation with parameter 𝜃 so that the randomly rendered images 𝒙 = 𝑔(𝜃) under different camera poses look like 2D samples of a pre-trained T2I diffusion model for a given text prompt 𝑦. Here, 𝑔 is a NeRF-like rendering engine. The T2I dif-

fusion model 𝜙 works by predicting the sampled noise 𝜖𝜙 (𝒙𝑡;𝑦,𝑡) of a rendered view 𝒙𝑡 at noise level 𝑡 for a given text prompt 𝑦. To move all rendered images to a higher density region under the text-conditioned diffusion prior, score distillation sampling (SDS) estimates the gradient for updating 𝜃 as:

∇𝜃LSDS(𝜙,𝑥) = E𝑡,𝜖 𝜔(𝑡) 𝜖𝜙 (𝒙𝑡;𝑦,𝑡) − 𝜖

𝜕𝒙 𝜕𝜃

, (1) where 𝜔(𝑡) is a weighting function.

Following SDS, variational score distillation (VSD) [Wang et al. 2023a] further improves generation diversity and quality, which regards the text-conditioned 3D representation as a random variable rather than a single data point in SDS. The gradient is computed as:

∇𝜃LVSD = E𝑡,𝜖 𝜔(𝑡) 𝜖𝜙 (𝒙𝑡;𝑦,𝑡) − 𝜖lora (𝒙𝑡;𝑦,𝑡,𝑐)

𝜕𝒙 𝜕𝜃

, (2)

where 𝑐 is the camera parameter, and 𝜖lora computes the score of noisy rendered images by a low-rank adaption (LoRA) [Hu et al. 2021] of the pre-trained T2I diffusion model. Despite the promising quality, both VSD and SDS mainly work on distilling the unitary prior from a single diffusion model and may collapse when encountering mixed priors from conflicted diffusion models.

- 3.3.2 Learning of concept prior. To learn concept prior, we leverage not only the concept image itself but also the 3D consistent informa-

tion in its initial 3D model 𝒎𝑖𝑛𝑖𝑡. We observe that the initial model suffers blurry texture and over-smoothed geometry, which is insufficient to provide a high-quality concept prior. Thus, we augment the initial rendered views {𝒙𝑖𝑛𝑖𝑡} of 𝒎𝑖𝑛𝑖𝑡 into augmented views {𝒙𝑖𝑛𝑖𝑡ˆ }, i.e., {𝒙𝑖𝑛𝑖𝑡ˆ } = 𝑎({𝒙𝑖𝑛𝑖𝑡}), where 𝑎(·) is the image-to-image translation operation, similar to [Raj et al. 2023]. These augmented views serve as pseudo-multi-view images of the conceptual subject, providing additional 3D information for further 3D modeling. Finally, the diffusion model 𝜙𝑐 with concept prior is derived by fine-tuning a T2I diffusion model given {𝑥𝑐, {𝒙𝑖𝑛𝑖𝑡ˆ },𝑦}, where 𝑦 is the text prompt with a special identifier, e.g., “a 3D model of [V] owl”.

- 3.3.3 Learning of reference prior. To learn reference prior, we leverage both the color images {𝒙𝑟 } and the normal maps {𝒏𝑟 } rendered from the reference models {𝒎𝑟 } under random viewpoints. While the rendered color images mainly provide 3D consistent priors on textures, the rendered normal maps focus on encoding detailed geometric information. The joint usage of these two kinds of renderings helps to build up a more comprehensive reference prior for introducing 3D consistent details during optimization. To disentangle the learning of image prior and normal prior, we also incorporate

different text prompts,𝑦𝑥 and𝑦𝑛, for color images, e.g., “a 3D model of an owl, in the style of [V]”, and normal maps, e.g., “a 3D model of an owl, in the style of [V], normal map”, respectively. Finally, the diffusion model 𝜙𝑟 with reference prior is derived by fine-tuning a pre-trained T2I diffusion model given {{𝒙𝑟 },𝑦𝑥, {𝒏𝑟 },𝑦𝑛}. Although we convert the 3D reference models into 2D space, their 3D

information has still been implicitly reserved across the consistent multi-view rendered color images and normal maps. Besides, as the pre-trained T2I diffusion models have been shown to possess rich 2D and 3D priors about the visual world [Liu et al. 2023c], we can also inherit these priors to enhance our modeling quality by projecting the 3D inputs into 2D space.

3.3.4 How does dual score distillation work? A straightforward aggregation of these two priors is performing the vanilla score distillation sampling twice indiscriminately for both diffusion models 𝜙𝑐 and 𝜙𝑟 and summing up the losses. However, this naive stack of two priors leads to loss conflicts during optimization and generates distorted results ((b) of Fig. 7). To resolve this, we introduce a dual score distillation (DSD) loss, which applies the two diffusion priors at different noise levels (denoising timesteps) during the reverse diffusion process.

This method is based on our observation that there is a coarse-tofine timestep-based dynamic during the reverse diffusion process. High noise levels, i.e., the early denoising timesteps 𝑡ℎ, control the global layout and rough color distribution of the image being denoised. As the reverse diffusion gradually goes into low noise levels, i.e., the late denoising timesteps 𝑡𝑙, high-frequency details are generated. This intriguing timestep-based dynamic process of T2I diffusion models is incredibly in line with the functionalities of our concept prior and reference priors. Inspired by image style transfer [Gatys et al. 2016] that leverages different layers of a pretrained neural network to control different levels of image content,

- as shown in Fig. 3, we apply the concept prior𝜙𝑐 at high noise levels 𝑡ℎ to enforce the concept fidelity by adjusting the layout and color

holistically, and apply the reference prior 𝜙𝑟 at low noise levels 𝑡𝑙 to recover the finer elements in detail.

Based on Eq. 2, the gradient for updating the 3D representation 𝜃 of the model being optimized given the concept prior is:

∇𝜃Lconcept(𝜙𝑐,𝑡ℎ) = E𝑡ℎ,𝜖 𝜔 𝜖𝜙𝑐 𝒙𝑡ℎ;𝑦,𝑡ℎ − 𝜖lora

𝜕𝒙 𝜕𝜃

, (3)

where 𝜔 is a weighting function, 𝜖𝜙𝑐 𝒙𝑡ℎ;𝑦,𝑡ℎ is the sampled noise of the rendered color image 𝒙𝑡ℎ at high noise level 𝑡ℎ conditioned on prompt 𝑦, and 𝜖𝑙𝑜𝑟𝑎 is the score of noisy rendered images parameterized by a LoRA of a pre-trained diffusion model. For reference prior, we apply it on both rendered color images and normal maps to jointly recover the detailed texture and geometry with the learned image prior and normal prior from the reference models. The gradient given the reference prior is:

∇𝜃Lref (𝜙𝑟,𝑡𝑙) = E𝑡𝑙,𝜖 𝜔 𝜖𝜙𝑟 𝒙𝑡𝑙 ;𝑦𝑥,𝑡𝑙 − 𝜖lora

𝜕𝒙 𝜕𝜃

+ E𝑡𝑙,𝜖 𝜔 𝜖𝜙𝑟 𝒏𝑡𝑙 ;𝑦𝑛,𝑡𝑙 − 𝜖lora

𝜕𝒙 𝜕𝜃

,

(4)

where 𝒙𝑡𝑙 and 𝒏𝑡𝑙 are the rendered color image and normal map

- at low noise level 𝑡𝑙, and 𝑦𝑥 and 𝑦𝑛 are their corresponding text prompts. Finally, the gradient of our DSD loss is:

∇𝜃LDSD = 𝛼∇𝜃Lconcept(𝜙𝑐,𝑡ℎ) + 𝛽∇𝜃Lref (𝜙𝑟,𝑡𝑙), (5) where 𝛼 and 𝛽 are weights to balance the strength of two guidance.

#### Table 1: Quantitative comparison with image-to-3D methods.

Wonder3D OpenLRM SyncD. Shap-E Magic123 Ours CLIP ↑ 0.777 0.840 0.803 0.761 0.868 0.890 Contextual ↓ 3.206 4.137 4.189 3.399 3.345 3.168

#### Table 2: Quantitative comparison with 3D variation methods.

Sin3DM Sin3DGen Ours Visual Diversity ↑ 0.180 0.201 0.315 Geometry Diversity ↑ 0.344 0.634 0.465 Visual Quality ↑ 5.221 5.127 5.848 Geometry Quality ↑ 5.638 5.607 5.616

### 4 EXPERIMENTS

We show the generated 3D models based on a few 3D exemplars in Fig. 10. We can see that our approach can generate various novel

- 3D assets that share consistent themes with the input exemplars. These generated 3D assets exhibit finer texture and elaborate geometry, ready for real-world usage (Fig. 1). Our approach can even work with only one exemplar, as shown in Fig. 11. For the rest of this section, we first conduct experiments and a user study to compare our results with those generated by the state-of-the-art methods. We also conduct experiments to analyze the effectiveness of several important design choices of our approach. We show implementation details in supplementary materials.

### 4.1 Comparisons with State-of-the-Art Methods

- 4.1.1 Benchmark. We have collected a dataset of 66 reference models covering a broad range of themes. These 3D models comprise three main categories, including 15 dioramas, 25 individual objects, and 26 characters, such as small islands, buildings, and characters, as shown in Fig. 10-11. Models in this dataset are exported from the built-in 3D library of Microsoft 3D Viewer or downloaded from Sketchfab1. The text prompts for each 3D model are automatically generated by feeding the model’s subject name, i.e., file name in most cases, into the pre-defined patterns presented in Sec. 3.
- 4.1.2 Methods Compared. To the best of our knowledge, ours is the first work focusing on theme-aware 3D-to-3D generation with diffusion priors. As no existing methods can simultaneously take both images and 3D models as inputs, we compare our method with seven baseline methods from two aspects. On the one hand, we compare with five image-to-3D methods, including multi-viewbased, i.e., Wonder3D [Long et al. 2023], SyncDreamer (SyncD.) [Liu et al. 2023a], feed-forward, i.e., LRM [Hong et al. 2023], Shape-E [Jun and Nichol 2023], and optimization-based, i.e., Magic123 [Qian et al. 2023], to evaluate our second stage that lifts a concept image to a 3D model. Due to the unavailable code of LRM, we use its open-source reproduction OpenLRM [He and Wang 2023]. On the other hand, we also compare with two 3D variation methods: Sin3DM [Wu et al. 2023] and Sin3DGen [Li et al. 2023], to evaluate the overall 3D-to-3D performance of our method. 1https://sketchfab.com/

[Figure 4]

Figure 4: Results of the user study. We compare our method with seven baseline methods using 2AFC pairwise comparisons. All preferences are statistically significant (𝑝 < 0.05, chi-squared test).

- 4.1.3 Quantitative Results. For image-to-3D, as our approach is not targeted to strictly reconstruct the input view, we focus on evaluating the semantic coherence between the input view and randomly rendered views of generated models. Thus, we adopt two metrics: 1) CLIP score [Radford et al. 2021] to measure the global semantic similarity, and 2) Contextual distance [Mechrez et al. 2018] to estimate the semantic distance at the pixel level. Both metrics are commonly used in image-to-3D [Sun et al. 2023; Tang et al. 2023b]. For 3D-to-3D, we use the pairwise IoU distance (1IoU) among generated models and the average LPIPS score across different views to measure the Visual Diversity and Geometry Diversity, respectively. To measure the Visual Quality and Geometry Quality, we use the LAION2 aesthetics predictor to predict the visual and geometry aesthetics scores given the multi-view rendered images (visual) and normal maps (geometry). The quantitative results in Tab. 1 and Tab. 2 show that our approach surpasses the baselines in generative diversity, quality and multi-view semantic coherency. Sin3DGen generates variations at the patch level, achieving higher geometry diversity. Sin3DM generates variations via a diffusion model trained with only one exemplar, achieving higher geometry quality. However, both methods tend to overfit the input and generate meaninglessly repeated or reorganized contents with lower visual diversity and quality (Fig. 6). In contrast, ours generates theme-consistent novel 3D assets with diverse and plausible variations in terms of both geometry and texture.
- 4.1.4 User Study. The metrics used above mainly measure the input-output similarity and pixel/voxel-level diversity, which are not able to present the overall performance of different methods. We thus conduct a user study to estimate real-world user preferences. We invite 30 users publicly to complete a questionnaire for pairwise comparisons. We explain the detailed settings of this user study in supplementary materials. We can see from Fig. 4 that our approach significantly outperforms existing methods in both image-to-3D and 3D-to-3D tasks in terms of human preferences.
- 4.1.5 Qualitative Results. For image-to-3D comparison (Fig. 5), we can see that Shap-E, SyncDreamer, and OpenLRM suffer from lower quality with incomplete shape, blurry appearance, and multi-view inconsistency. Results of Wonder3D and Magic123 can generate 3D consistent models with higher quality. However, Wonder3D still

2https://laion.ai/blog/laion-aesthetics/

[Figure 5]

#### Figure 5: Qualitative comparisons with five image-to-3D methods to evaluate our second stage that lifts a concept image to a 3D model. We show the frontal view as primary for the first line and show the back view as primary for the last two lines.

[Figure 6]

#### Figure 6: Qualitative comparisons with two 3D variation methods to evaluate the overall generative diversity and quality of our method. For each case, we show three generated 3D models.

generates vague texture and incomplete shape, e.g., the severed tail of the triceratops, and Magic123 has problems with oversaturation and oversmooth. All baseline methods lack delicate details, especially in novel views, e.g., epidermal folds in the last line. In contrast, ours generates multi-view consistent 3D models with more details in geometry and texture. For 3D-to-3D comparison (Fig. 6), we can see that the baseline methods tend to randomly resize, repeat, or reorganize the input, which may produce weird results, e.g., multi-head character and stump above treetop. Due to their theme-unaware 3D representation learned from just a few exemplars, it is hard for them to preserve or even understand the semantics of the input 3D exemplars. Instead, our approach combines priors from input 3D exemplars and pre-trained T2I diffusion models, yielding diverse semantically meaningful 3D variations that exhibit significant modifications on content while thematically aligning with the input exemplars.

### 4.2 Ablation Study

- 4.2.1 Settings. To evaluate the effectiveness of our key design choices, we conduct ablation studies on five settings: (a) Baseline, which only uses concept prior across all noise levels, (b) + Ref. prior naive, which naively applies concept prior and reference prior

#### Table 3: Quantitative results of the ablation study.

Baseline +Ref. +Ref. Reverse Ref. naive DSD DSD dominated

CLIP ↑ 0.877 0.876 0.890 0.863 0.874 Contextual ↓ 3.182 3.177 3.168 3.186 3.179 Visual Quality ↑ 5.639 5.726 5.848 5.578 5.701 Geometry Quality ↑ 4.789 5.336 5.616 4.926 5.296

across all noise levels, (c) + Ref. prior DSD (full model), which applies concept prior at high noise levels and reference prior at low noise levels, (d) Reverse DSD, which reverses the choice of noise levels by applying concept prior at low noise levels and reference prior at high noise levels, and (e) Ref. dominated, which applies concept prior at high noise levels and reference prior across all noise levels. We measure the semantic coherence, visual quality and geometry quality as in image-to-3D and 3D-to-3D comparisons for the ablation study. As shown in Tab. 3, our full model (+Ref. DSD) surpasses all baselines in terms of the four metrics mentioned above. We also show the qualitative comparison results in Fig. 7. Next, we further explore the scope and generality of the DSD loss.

[Figure 7]

- Figure 7: Ablation study on two types of effects: (1) reference prior and DSD loss (Sec. 4.2.2) and (2) the choice of noise levels for the DSD loss (Sec. 4.2.3). (a) without the reference prior; (b) a naive combination of concept prior and reference prior; (c) using the proposed dual score distillation (DSD); (d) reversing the choice of noise levels for DSD; and (e) extending the reference prior to all noise levels for DSD. We show the back view for each case.

- 4.2.2 EffectofthereferencepriorandDSD loss. As showninFig.7(a,c) and Tab. 3, it is evident that the introduction of the reference prior and DSD significantly enhances the model quality in terms of semantic coherence, texture and geometry. From Fig. 7(b), we can see that the naive combination of the reference prior and concept prior results in severe loss conflict and produces bumpy surface and blurry texture, which further demonstrate the effectiveness of our DSD for alleviating loss conflicts.
- 4.2.3 Effect of the choices of noise levels for the DSD loss. By comparing (c) with (d) in Fig. 7, we can see a significant performance degradation after reversing the noise levels, which justifies our claim that the timestep-based dynamic process of T2I diffusion models is consistent with the functionalities of our concept prior and reference prior (Sec. 3.3). Besides, by comparing (c) with (e) in Fig. 7, we can see that extending the noise levels for the reference prior has no positive effect but leads to a worse result, indicating that the design of separating two priors at different noise levels can help reduce loss conflict.

### 5 APPLICATION

As shown in Fig. 9, ThemeStation supports the application of controllable 3D-to-3D generation, which allows users to control the concept image generation process via text prompt manipulation and obtain specific 3D variations. This sample application demonstrates the immense potential of ThemeStation to be seamlessly combined with emerging controllable image generation techniques [Brooks et al. 2023; Hertz et al. 2022; Wang et al. 2022] for more interesting 3D-to-3D applications.

### 6 CONCLUSION

In this work, we proposed ThemeStation, a novel approach for the theme-aware 3D-to-3D generation task. Given just one or a few 3D exemplars, we aim to generate a gallery of unique theme-consistent

- 3D models. ThemeStation achieves this goal following a two-stage generative scheme that first draws a concept image as rough guidance and then converts it into a 3D model. Our 3D modeling process involves two priors, one from the input 3D exemplars (reference

[Figure 8]

Figure 8: Failure cases. (a) Our approach may fail to fix huge concept errors when the concept image contains significant artifacts or mistakes, e.g., the tail grows in front of the body. (b) Our approach may fail to generate perfect 3D models of regular shapes, such as a “Minecraft” building with cubic regularization, due to the lack of explicit geometry constraints.

prior) and the other from the concept image (concept prior) generated in the first stage. A dual score distillation (DSD) loss function is proposed to disentangle these two priors and alleviate loss conflict. We have conducted a user study and extensive experiments to validate the effectiveness of our approach.

While ThemeStation produces high-quality 3D assets given just one or a few 3D exemplars and opens up a new venue for themeaware 3D-to-3D generation, it still has several limitations for further improvement. First, similar to prior optimization-based 3D generation methods, it still takes hours for our current pipeline to optimize the initial model into a final 3D asset. We believe advanced diffusion models and neural rendering techniques in the future can help alleviate this problem. Besides, like two sides of a coin, as a two-stage pipeline, although ThemeStation can be easily adapted to emerging image-to-3D methods for obtaining a better initial model, it may also suffer from a bad initialization sometimes, e.g., 3D artifacts and floaters. Training a feed-forward theme-aware 3D-to-3D generation model is a potential solution, which can be an interesting future work. Failure cases are shown in Fig. 8.

### ACKNOWLEDGMENTS

This work is partially supported by the National Key R&D Program of China (2022ZD0160201) and Shanghai Artificial Intelligence Laboratory. This work is also in part supported by a GRF grant from the Research Grants Council of Hong Kong (Ref. No.: 11205620).

### REFERENCES

Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski.

2023. Break-A-Scene: Extracting Multiple Concepts from a Single Image. arXiv preprint arXiv:2305.16311 (2023).

Bob. 2022. 3D Modeling 101: Comprehensive Beginners Guide. Retrieved Jan 03, 2024 from https://wow-how.com/articles/3d-modeling-101-comprehensive-beginnersguide

Andrew Brock, Jeff Donahue, and Karen Simonyan. 2018. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096 (2018).

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18392–18402.

CGHero. 2022. The Stages of Creating a 3D Model. Retrieved Jan 02, 2024 from https://cghero.com/articles/stages-of-creating-3d-model

Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al.

2022. Efficient geometry-aware 3D generative adversarial networks. In CVPR. Siddhartha Chaudhuri, Evangelos Kalogerakis, Leonidas Guibas, and Vladlen Koltun.

2011. Probabilistic reasoning for assembly-based 3D modeling. In ACM SIGGRAPH 2011 papers. 1–10.

Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. 2023b. Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction. arXiv preprint arXiv:2304.06714 (2023).

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023a. Fantasia3D: Disentangling geometry and appearance for high-quality text-to-3D content creation. arXiv preprint arXiv:2303.13873 (2023).

Yongwei Chen, Tengfei Wang, Tong Wu, Xingang Pan, Kui Jia, and Ziwei Liu. 2024. ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance.

Victor Dibia. 2022. Latent Diffusion Models: Components and Denoising Steps. Retrieved Jan 04, 2024 from https://victordibia.com/blog/stable-diffusion-denoising/

Ziya Erkoç, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. 2023. HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion. arXiv:2303.17015 [cs.CV]

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. 2022a. An Image is Worth One Word: Personalizing Text-toImage Generation using Textual Inversion. https://doi.org/10.48550/ARXIV.2208. 01618

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022b. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Leon A Gatys, Alexander S Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2414–2423.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Oğuz. 2023. 3DGen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371

(2023). Zexin He and Tengfei Wang. 2023. OpenLRM: Open-Source Large Reconstruction Models. https://github.com/3DTopia/OpenLRM.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control.

(2022). Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Tengfei Wang, Liang Pan, Dahua Lin, and Ziwei Liu. 2024. 3DTopia: Large Text-to-3D Generation Model with Hybrid Diffusion Priors. arXiv preprint arXiv:2403.02234 (2024).

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3D. arXiv preprint arXiv:2311.04400 (2023).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Heewoo Jun and Alex Nichol. 2023. Shap-e: Generating conditional 3D implicit functions. arXiv preprint arXiv:2305.02463 (2023).

Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4401–4410.

Vladimir G Kim, Wilmot Li, Niloy J Mitra, Siddhartha Chaudhuri, Stephen DiVerdi, and Thomas Funkhouser. 2013. Learning part-based templates from large collections of 3D shapes. ACM Transactions on Graphics (TOG) 32, 4 (2013), 1–12.

Weiyu Li, Xuelin Chen, Jue Wang, and Baoquan Chen. 2023. Patch-based 3D Natural Scene Generation from a Single Example. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16762–16772.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3D: HighResolution Text-to-3D Content Creation. In Conference on Computer Vision and Pattern Recognition (CVPR).

Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. 2023b. One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion. arXiv preprint arXiv:2311.07885 (2023).

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023c. Zero-1-to-3: Zero-shot one image to 3D object. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9298–9309.

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023a. SyncDreamer: Generating Multiview-consistent Images from a Single-view Image. arXiv preprint arXiv:2309.03453 (2023).

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2023. Wonder3D: Single image to 3D using cross-domain diffusion. arXiv preprint arXiv:2310.15008 (2023).

Roey Mechrez, Itamar Talmi, and Lihi Zelnik-Manor. 2018. The contextual loss for image transformation with non-aligned data. In Proceedings of the European conference on computer vision (ECCV). 768–783.

Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. 2023. RealFusion: 360 Reconstruction of Any Object from a Single Image. In Conference on Computer Vision and Pattern Recognition (CVPR).

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. 2023. Latent-NeRF for shape-guided generation of 3D shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12663– 12673.

Charlie Nash, Yaroslav Ganin, SM Ali Eslami, and Peter Battaglia. 2020. Polygen: An autoregressive generative model of 3D meshes. In International conference on machine learning. PMLR, 7220–7229.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen.

2023. Point-E: A System for Generating 3D Point Clouds from Complex Prompts. https://arxiv.org/abs/2212.08751 (2023).

Michael Niemeyer and Andreas Geiger. 2021. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11453–11464.

Dario Pavllo, Jonas Kohler, Thomas Hofmann, and Aurelien Lucchi. 2021. Learning generative models of textured 3D meshes from real-world images. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13879–13889.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Textto-3D using 2D Diffusion. In International Conference on Learning Representations (ICLR).

Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. 2023. Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors. https://arxiv.org/abs/2306.17843 (2023).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. 2023. Dreambooth3D: Subject-driven text-to-3D generation. arXiv preprint arXiv:2303.13508 (2023).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Leonid I Rudin, Stanley Osher, and Emad Fatemi. 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena 60, 1-4 (1992), 259–268.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35 (2022), 36479– 36494.

Nadav Schor, Oren Katzir, Hao Zhang, and Daniel Cohen-Or. 2019. Componet: Learning to generate the unseen by part synthesis and composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 8759–8768.

Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. 2019. SinGAN: Learning a generative model from a single natural image. In Proceedings of the IEEE/CVF international conference on computer vision. 4570–4580.

Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. 2021. Deep marching tetrahedra: a hybrid representation for high-resolution 3D shape synthesis. Advances in Neural Information Processing Systems 34 (2021), 6087–6101.

Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. 2023. DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior. https://arxiv.org/abs/2310.16818 (2023).

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu.

2024. LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation. arXiv preprint arXiv:2402.05054 (2024).

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023a. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. arXiv:2309.16653 [cs.CV]

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. 2023b. Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior. In International Conference on Computer Vision ICCV.

Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, and Baining Guo. 2023b. RODIN: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023).

Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, and Fang Wen. 2022. Pretraining is All You Need for Image-to-Image Translation. In arXiv.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023a. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. https://arxiv.org/abs/2305.16213 (2023).

Rundi Wu, Ruoshi Liu, Carl Vondrick, and Changxi Zheng. 2023. Sin3DM: Learning a Diffusion Model from a Single 3D Textured Shape. arXiv preprint arXiv:2305.15399

(2023). Rundi Wu and Changxi Zheng. 2022. Learning to generate 3D shapes from a single example. arXiv preprint arXiv:2208.02946 (2022).

Kai Xu, Hao Zhang, Daniel Cohen-Or, and Baoquan Chen. 2012. Fit and diverse: Set evolution for inspiring 3D shape galleries. ACM Transactions on Graphics (TOG) 31, 4 (2012), 1–10.

Shi Yichun, Wang Peng, Ye Jianglong, Mai Long, Li Kejie, and Yang Xiao. 2023. MVDream: Multi-view Diffusion for 3D Generation. https://arxiv.org/abs/2308.16512

(2023).

Youyi Zheng, Daniel Cohen-Or, and Niloy J Mitra. 2013. Smart variations: Functional substructures for part compatibility. In Computer Graphics Forum, Vol. 32. Wiley Online Library, 195–204.

Linqi Zhou, Yilun Du, and Jiajun Wu. 2021. 3D shape generation and completion through point-voxel diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5826–5835.

[Figure 9]

#### Figure 9: Application results of controllable 3D-to-3D generation. ThemeStation allows users to specify a desired 3D variation via text prompt manipulation.

[Figure 10]

#### Figure 10: Visual results of ThemeStation, which generates 3D models from a few 3D exemplars. For each case, we show the reference models on the left and six generated models on the right. For each generated model, we show a primary view (top) with its normal map (bottom right) and a secondary view (bottom left).

[Figure 11]

#### Figure 11: Visual results of ThemeStation, which generates 3D models from only one 3D exemplar. For each case, we show the reference model (left) and three generated models on the right. For each generated model, we show a primary view (top) with its normal map (bottom right) and a secondary view (bottom left).

SUPPLEMENTARY MATERIAL

- A IMPLEMENTATION DETAILS

In the first stage, we render 20 images for each reference model with a fixed elevation, i.e., 0 or 20, and randomized azimuth. We fine-tune the pre-trained Stable Diffusion [Rombach et al. 2022] model for 200 iterations (a single exemplar) or 400 iterations (a few exemplars) with a batch size of 8. We set the learning rate as 2×10−6, the image size as 512×512, and the CFG weight at inference as 7.5. We also take the camera pose of the rendered images as an additional condition during the model fine-tuning step to ensure the generated concept images have a correct viewpoint for accurate image-to-3D initialization.

In the second stage, we employ an off-the-shelf image-to-3D method [Long et al. 2023] to lift the synthesized concept image into an initial 3D model, represented as a neural implicit signed distance field (SDF). We use the concept image and 20 augmented views of the initial model for concept prior learning and use 30 normal maps, and 30 color images of the input 3D exemplars for reference prior learning. During optimization, we convert the SDF into DMTet [Shen et al. 2021] at a 192 grid and 512 resolution to directly optimize the textured mesh at each optimization iteration. We render both the normal map and the color image, under randomized viewpoints, as guidance to compute the DSD loss (Eq. 5). We use dynamic diffusion timestep that samples larger timestep from range [0.5, 0.75] when applying the concept prior and samples smaller timestep from range [0.1, 0.25] for the reference prior. We set 𝛼 as 0.2 and 𝛽 as 1.0. The total optimization step is 5000. We also adopt the total variation loss [Rudin et al. 1992] and contextual loss [Mechrez et al. 2018] to enhance the texture quality. Specially, the contextual loss is applied between the rendered color image and the 20 augmented views of the initial model. The whole 3D-to-3D generation process takes around 2 hours using a single NVIDIA A100 GPU.

- B USER STUDY SETTINGS

We randomly select 20 models from our dataset and generate 3 variations for each model. We invite a total of 30 users, recruited publicly, to complete a questionnaire consisting of 30 pairwise comparisons (15 for image-to-3D and 15 for 3D-to-3D) in person, totaling 900 answers. For image-to-3D, we show two generated 3D models (one by our method and one by the baseline method) beside a concept image and ask the users to answer the question: “Which of the two models do you prefer (e.g., higher quality and more details) on the premise of aligning with the input view?" For 3D-to-3D, we show two sets of generated 3D variations beside a reference model and ask the question: “Which of the two sets do you prefer (e.g., higher quality and more diversity) on the premise of sharing consistent themes with the reference?"

- C EVALUATION OF THEME-DRIVEN DIFFUSION MODEL

To evaluate the influence of different fine-tuning iterations for the theme-driven diffusion model that generates concept images in the

#### Table 4: Quantitative evaluation of theme-driven diffusion model.

Iteration100 Iteration200 Iteration300 Iteration400

LPIPS-diversity ↑ 0.627 0.617 0.403 0.347 LAION-aesthetic-score ↑ 6.262 6.355 6.367 5.941

first stage, we conduct ablation studies on four settings, i.e., finetuning the theme-driven diffusion model given one 3D exemplar for 100, 200, 300 and 400 iterations. We use LPIPS-diversity (LPIPS differences across generated images) and LAION-aesthetic-score to estimate the diversity and quality of generated concept images under different settings. The quantitative results are shown in Tab. 4. As can be seen, diversity significantly drops when iteration is 300, and quality drops when iteration is 400, both caused by overfitting. We thus set the fine-tuning iteration to 200 for a single exemplar (Sec. A).

### D POTENTIAL ETHICS ISSUES

As a generative model, ThemeStation may pose ethical issues if used to create baleful and fake content, which requires more vigilance and care. We can adopt the commonly used safety checker in existing text-to-image diffusion models to filter out maliciously generated concept images in our first stage to alleviate the potential ethics issues.

