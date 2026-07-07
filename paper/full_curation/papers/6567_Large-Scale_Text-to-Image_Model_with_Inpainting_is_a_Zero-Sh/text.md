## Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator

Chaehun Shin1 Jooyoung Choi1 Heeseung Kim1 Sungroh Yoon1,2,∗ 1Data Science and AI Laboratory, ECE, Seoul National University 2AIIS, ASRI, INMC, ISRC, and Interdisciplinary Program in AI, Seoul National University

{chaehuny, jy choi, gmltmd789, sryoon}@snu.ac.kr https://diptychprompting.github.io

arXiv:2411.15466v2[cs.CV]4Jun2025

|[Figure 1]| |
|---|---|
| | |

|[Figure 2]|[Figure 3]|
|---|---|

|[Figure 4]|
|---|

|[Figure 5]|[Figure 6]|
|---|---|

|[Figure 7]|[Figure 8]|
|---|---|

|[Figure 9]| |
|---|---|
|[Figure 10]| |

|[Figure 11]|
|---|
|[Figure 12]|

| |[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]| |
|---|---|---|
| | | |

[Figure 17]

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

|[Figure 21]|
|---|

###### (a) Diptych Generation (b) Inpainting-based Diptych Prompting

Figure 1. Given a single reference image, our Diptych Prompting performs zero-shot subject-driven text-to-image generation through diptych inpainting. Building on the (a) diptych generation capability of FLUX [23], we extend it to diptych inpainting with a separate module, resulting in (b) versatility across various tasks including subject-driven text-to-image generation, stylized image generation, and subject-driven image editing.

#### Abstract

complete diptych with the reference image in the left panel, and performs text-conditioned inpainting on the right panel. We further prevent unwanted content leakage by removing the background in the reference image and improve finegrained details in the generated subject by enhancing attention weights between the panels during inpainting. Experimental results confirm that our approach significantly outperforms zero-shot image prompting methods, resulting in images that are visually preferred by users. Additionally, our method supports not only subject-driven generation but also stylized image generation and subject-driven image editing, demonstrating versatility across diverse image generation applications.

Subject-driven text-to-image generation aims to produce images of a new subject within a desired context by accurately capturing both the visual characteristics of the subject and the semantic content of a text prompt. Traditional methods rely on time- and resource-intensive fine-tuning for subject alignment, while recent zero-shot approaches leverage on-the-fly image prompting, often sacrificing subject alignment. In this paper, we introduce Diptych Prompting, a novel zero-shot approach that reinterprets as an inpainting task with precise subject alignment by leveraging the emergent property of diptych generation in large-scale text-to-image models. Diptych Prompting arranges an in-

∗ Correspondence to: Sungroh Yoon (sryoon@snu.ac.kr)

#### 1. Introduction

With recent advancements in generative models, text-toimage (TTI) models [3, 4, 7, 9, 10, 35, 38, 42] have significantly improved, enabling the generation of photorealistic images based on text prompts. Beyond generating images from text, these models support various text-based image tasks, including text-guided editing [12, 13, 18, 30, 53], text-guided style transfer [14, 40, 43], and subject-driven text-to-image generation [11, 22, 24, 28, 32, 33, 41, 48, 50, 51, 55]. Specifically, subject-driven text-to-image generation aims to synthesize images of a specific subject in various contexts based on a text prompt and a reference image, while achieving both subject and text alignment.

Early research on subject-driven text-to-image generation [11, 22, 41, 48] enables the model to synthesize a new subject through fine-tuning on a small set of images containing the target subject. While they achieve strong subject alignment via optimization, they are time- and resourceintensive, requiring hundreds of iterative steps of optimization for each new subject. As an alternative, zero-shot approaches [24, 28, 32, 33, 50, 51, 55] have emerged that do not require additional fine-tuning and instead utilize image prompting through a specialized image encoder. These methods extract the image feature from a reference image and integrate it into the TTI model alongside the text feature. While they achieve on-the-fly subject-driven text-toimage generation with a single forward pass of the encoder, these encoder-based image prompting frameworks suffer from unsatisfactory subject alignment, particularly in capturing granular details.

Recently, as models in NLP fields have been scaled up and demonstrated remarkable capabilities [1, 5], large-scale TTI models [9, 23] have similarly emerged. Notably, the recently released model, FLUX [23], has demonstrated exceptional text comprehension and the ability to effectively translate this understanding into images, even for highly complex and lengthy texts. Among the various capabilities of FLUX, we focus on its ability to generate highquality diptychs−two-paneled art pieces in which each panel contains an interrelated image. As shown in Fig. 1 (a), FLUX’s advanced text understanding and high-resolution image generation enable it to generate side-by-side images of the same object, each reflecting a different context as specified in the prompt for each panel.

Motivated by FLUX’s ability to generate diptychs, we propose “Diptych Prompting”, a novel inpainting-based framework for zero-shot, subject-driven text-to-image generation. In our approach, we reinterpret the task as a diptych inpainting process: the left panel contains a reference image of the subject as a visual cue, and the right panel is generated through inpainting based on a text prompt describing the diptych with the desired context. Using text-conditioned diptych inpainting, Diptych Prompting aligns the generated

image in the right panel with both the reference subject and the text prompt. We enhance this process by removing the background from the reference image to prevent content leakage and focus solely on the subject, and by enhancing attention weights between panels to ensure fine-grained details preservation. These two components enable Diptych Prompting to achieve more consistent, high-quality subjectdriven text-to-image generation.

Through various experiments, Diptych Prompting demonstrates superior performance over existing encoderbased image prompting methods, more effectively capturing both subject and text and producing results preferred by human evaluators. Additionally, our method is not limited to subjects; it can also be applied to styles, enabling stylized image generation [14, 40, 43] when a personal style image is provided as a reference. Furthermore, we showcase the extensibility of our approach to subject-driven image editing [54], allowing modification of specific regions in the target image with the reference subject. By arranging the target image in the right panel of diptych and masking only the region for editing in Diptych Prompting, we successfully integrate the reference subject into the target image.

Our contributions can be summarized as follows:

- • We propose a novel inpainting-based zero-shot subjectdriven text-to-image generation approach without further training, offering a new perspective by highlighting the inherent diptych generation capabilities of FLUX.
- • We propose two techniques to prevent content leakage and reliably capture details in the target subject: isolating the subject from its background and enhancing attention weights between panels.
- • We validate our method’s versatility and robustness, extending its effectiveness even to style-driven generation and subject-driven image editing through comprehensive qualitative and quantitative results.

#### 2. Related Works

##### 2.1. Diffusion-based Text-to-Image Models

Diffusion models [16, 20, 44, 45] have led to significant advancements in TTI models, including GLIDE [31], LDM [38], DALL-E 2 [37], Imagen [42], and eDiff-I [3]. Among these, the Stable Diffusion (SD) series [9, 35, 38] has gained particular attention for its open-source nature and competitive performance to previous research. Starting with the v1 model, which utilizes a U-Net [39] architecture with cross-attention for text, it evolved through v2 and then to SD-XL [35], with improvements in dataset scale, model architecture, resolution, and generation quality.

Recently, generative model research [34] has achieved notable performance improvement by incorporating transformer [47] architectures into diffusion models instead of U-Net. Driven by this advancement, emerging studies now

integrate transformer architecture into TTI models, most notably SD-3 [9] and FLUX [23]. Both models employ the MultiModal-Diffusion Transformer (MM-DiT) architecture, an advanced design for TTI models that conducts joint attention on concatenated text and image embeddings,

Q = [Qt;Qi],K = [Kt;Ki],V = [Vt;Vi], (1) A(Q,K,V ) = W(Q,K)V = softmax

QKT √

V, (2)

d

where [;] is the concatenation, Q, K, and V represent the key components of attention−query, key, and value, respectively; W is the attention weight, and A is the output of the attention. FLUX, in particular, is the largest-scale TTI model among open-source models and exhibits advanced performance in both text comprehension and image generation quality, surpassing previous open-source models.

##### 2.2. Text-Conditioned Inpainting

Image inpainting aims to fill the missing regions of an incomplete image I using a binary mask M that specifies the areas to be reconstructed. Recent advancements in TTI models have led to the development of text-conditioned inpainting [54], which completes the missing regions to align not only with the visible region but also with a text prompt,

Iˆ = Fθ(I,M,T), (3)

where T is the text describing the desired context, and Fθ is the generation process of the text-conditioned models. Various methods [45, 54] have been proposed to implement a plausible Fθ from the pre-trained TTI models.

While an early approach [45] employs pre-trained diffusion models without any further training, more recent works fine-tune the pre-trained TTI model or train additional modules [57] specifically for inpainting tasks. Through additional training for inpainting, these models achieve the two main objectives of text-conditioned inpainting: alignment with the visible regions in I and alignment with the text prompt. Among various inpainting modules, ControlNet [57] equips FLUX with inpainting capability, providing inpainting-specific conditioning for enhanced control. By leveraging this module, we interpret inpainting as a framework for subject-driven text-to-image generation.

##### 2.3. Subject-Driven Image Generation

There has been extensive research on subject-driven text-toimage generation [11, 22, 24, 28, 32, 33, 41, 48, 50, 51, 55], where the generated images not only render the various contexts described by the text prompt but also include the specific subject according to reference images. Subject-driven text-to-image generation is generally categorized into two groups based on whether they require additional training for each new subject.

|[Figure 22]|
|---|

|[Figure 23]|
|---|

SD-v2 SD-XL

|[Figure 24]|
|---|

|[Figure 25]|
|---|

SD-3 FLUX

Figure 2. Diptych Generation Comparisons. We generate the diptych images with various TTI models from the following diptych text: “A diptych with two side-by-side images of same cat. On the left, a photo of a cat in front of Eiffel Tower. On the right, replicate this cat exactly but as a photo of a cat in the jungle”.

The first category [11, 22, 41, 48] involves fine-tuning on a small set of subject images (e.g., 3-5 images) to learn the visual subject and how to generate it. While these methods achieve strong subject alignment through optimization on the subject, the fine-tuning requires retraining for each new subject, making them time- and resource-intensive. Moreover, optimizing on a small set of images may lead to overfitting on the new subject and catastrophic forgetting of prior knowledge which should be carefully prevented.

The second group [24, 28, 32, 33, 50, 51, 55] addresses these limitations through image prompting, which utilizes a specialized image encoder [36] to incorporate a reference image alongside the text prompt to guide the generated output. Such methods enable zero-shot manner; yet, it often lacks target subject alignment. Other notable approaches [17, 52, 56] fine-tune the TTI model for joint-set image generation [2, 46, 49], and extend this to subjectdriven generation or editing through inpainting. However, they still face training constraints, such as costs for dataset construction and training. Leveraging the inherent capability of a recent TTI model with an inpainting module, we propose a novel zero-shot inpainting-based approach without additional training.

#### 3. Method

##### 3.1. Diptych Generation of FLUX

A ‘Diptych’ is an art term referring to a two-paneled artwork in which two panels are displayed side by side, each containing interrelated content. Previous work, HQEdit [19], proposed a pipeline for creating an image editing dataset in the form of diptychs using DALL-E 3 [4]. The strong text-image alignment of the large-scale TTI model, DALL-E 3, plays a critical role in creating coherent editing pairs in the diptych.

The recently released large-scale open-source TTI

𝐼

𝐾 𝐾 𝐾

𝐺 (𝐼 )

[Figure 26]

[Figure 27]

𝑀

[Figure 28]

…

[Figure 29]

|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|[Figure 36]|[Figure 37]|
|---|---|---|---|---|---|---|---|

𝐼

[Figure 38]

𝐼 𝐼

ControlNet Inpainting

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | |×|?|?| | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

𝑄

…

| | | |[Figure 39]|
|---|---|---|---|
| | | | |

|[Figure 40]|
|---|
|[Figure 41]|
|[Figure 42]|
|[Figure 43]|
|[Figure 44]|
|[Figure 45]|
|[Figure 46]|
|[Figure 47]|

𝑄

[Figure 48]

###### SAM

[Figure 49]

[Figure 50]

𝑸𝑲𝑻 𝒅

FLUX

𝑾 = 𝒔𝒐𝒇𝒕𝒎𝒂𝒙

=

Grounding DINO

𝑄

𝑇 𝐼

[Figure 51]

A diptych with two side-by-side images of same {subject name}. On the left, a photo of {subject name}. On the right, replicate this {subject name} but as {target text prompt}.

subject name

Attention Map in FLUX for Diptych Inpainting

𝐼

(a) Diptych Prompting Framework (b) Reference Attention Enhancement

- Figure 3. (a) Overall Diptych Prompting Framework. Given the incomplete diptych Idiptych, text prompt Tdiptych describing the diptych, and the binary mask Mdiptych specifying the right panel as the inpainting target, FLUX with ControlNet module performs text-conditioned inpainting on the right panel while referencing the subject in the left panel. (b) Reference Attention Enhancement. To capture the granular details of the subject in left panel, we enhance the reference attention, an attention weight between the query of the right panel and the key of the left panel.

prompt describing the desired context, Diptych Prompting begins with the triplets for inpainting-based prompting: an incomplete diptych image Idiptych, a binary mask Mdiptych specifying the missing region, and a diptych text Tdiptych.

model, FLUX [23], demonstrates strong text comprehension and image generation capabilities, even surpassing DALL-E 3 [4]. Notably, its capabilities also extend to diptych generation: when we generate an image with diptych text Tdiptych, “A diptych with two side-by-side images of the same {object}. On the left, {description of left image}. On the right, replicate this {object} but as {description of right image}”, FLUX synthesizes a diptych image where the subjects in each panel are interrelated and each description of panel is accurately represented, as shown in Fig. 2.

For the incomplete diptych image Idiptych, we concatenate two images along the width dimension with the left panel containing the reference subject image and the right panel consisting of a blank image of the same size to be inpainted. We observe that simple diptych inpainting often results in excessive interrelation with the reference image by mirroring even subject-unrelated contents, such as background, pose, and location (Fig. 4). To prevent this, we remove the background of the reference image through the background removal process Gseg using Grounding DINO [27] and Segment Anything Model (SAM) [21]. In this process, Grounding DINO uses the subject name to acquire a bounding box of target subject through grounded object detection, and SAM performs subject segmentation with this detection box and removes the background, preparing it as the left panel,

Generating high-quality diptych images requires the robust text-image alignment capability of large-scale TTI model, in which smaller models fall short. Compared to previous models such as SD-v2 [38], SD-XL [35], and SD3 [9], only FLUX [23] successfully synthesizes accurate diptych images that not only effectively interrelate subjects across panels but also render the correct contexts for each panel described in the diptych text. Therefore, we choose FLUX as the base model for our proposed methodology due to its superior ability to generate accurate and contextually aligned diptych images.

Idiptych = [Gseg(Iref); ∅ ]. (4)

##### 3.2. Diptych Prompting Framework

Additionally, our binary mask Mdiptych designates the location of the reference image in the left panel with zeros to provide visual cues, while marking the right panel with ones to indicate the missing areas to be filled,

For zero-shot subject-driven text-to-image generation, most approaches rely on a specialized image encoder for image prompting that extracts image feature from a reference image and integrates it into the TTI model. Instead, to inject detailed subject characteristics into the generated image in a zero-shot manner, we propose a novel prompting approach that reinterprets zero-shot method from the perspective of inpainting, as illustrated in Fig. 3 (a).

Mdiptych = [0h×w;1h×w], (5)

where 0h×w and 1h×w have the same size with each corresponding panel.

For the diptych text Tdiptych describing the diptych configuration with desired context, we utilize the prompt tem-

Given the reference subject image and the target text

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

w/o 𝑮𝒔𝒆𝒈

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

w/ 𝑮𝒔𝒆𝒈

- Figure 4. Background Removal Effects. Simple diptych inpainting exhibits content leakage from the reference image, including background, pose, and location. We mitigate this unwanted leakage through background removal by Gseg.

plate used in Sec. 3.1. From the target text prompt, we use the subject name of reference subject for object, resulting in the following final diptych text: “A diptych with two side-by-side images of same {subject name}. On the left, a photo of {subject name}. On the right, replicate this {subject name} exactly but as {target text prompt}”.

Using these triplets, our Diptych Prompting performs the text-conditioned inpainting,

Iˆdiptych = [Gseg(Iref);Iˆgen] = Fθ(Idiptych,Mdiptych,Tdiptych),

(6) where Iˆgen represents the desired subject-driven image.

##### 3.3. Reference Attention Enhancement

Diptych Prompting reconstructs the right panel of the diptych by referencing the subject in the left panel. However, FLUX [23] with inpainting module often struggles to fully capture the fine details of the subject.

Recent studies [8, 13, 14] have shown that the image generation process in U-Net-based TTI models can be controlled by manipulating key components of the attention−query, key, value, and attention weight−yet similar techniques remain largely unexplored in transformerbased architectures. Given that FLUX, built on the MMDiT architecture, incorporates more attention blocks than previous U-Net-based models, it offers greater potential for such control. In Diptych Prompting, we note that FLUX synthesizes both the reference and generated image simultaneously in a diptych format through its attention blocks and computes the attention between the left and right panels. This leads us to enhance reference attention−the influence of the left panel on the right−to better capture granular details of the reference subject.

In the attention blocks of FLUX, the image feature part can be divided into two regions in diptych inpainting, corresponding to the left and right panel,

###### Q = [Qt;Qli;Qri],K = [Kt;Kli;Kri], (7)

where (·)t is the feature for text, (·)li is for the left panel, and (·)ri is for the right panel.

From this, we acquire the attention weight W(Q,K) as shown in Eq. (2), where W(Q,K) ∈ R(l

t+lli+lri)×(lt+lli+lri), lt is the text sequence length, and l·i is sequence length of each panel in attention blocks as described in Fig. 3 (b). We enhance the reference attention, the attention weight between the query of right panel (Qri) and the key of left panel (Kli) by rescaling the submatrix W(Qri,Kli) with λ > 1.

#### 4. Experiments

##### 4.1. Experimental Settings

Implementation Details Our method is implemented based on the large-scale TTI model, FLUX-dev1, with the additional ControlNet-Inpainting module2. We perform diptych inpainting on a canvas with an aspect ratio of 1:2, sized at 768×1536, where the left half (768×768) serves as the reference. During inference, the ControlNet conditioning scale is set to 0.95 and the reference attention rescaling parameter λ is set to 1.3 for diptych inpainting performed over 30 steps and a guidance scale of 3.5 [15, 29].

Evaluations We measure zero-shot subject-driven text-toimage generation performance on DreamBench [41] that contains 30 subjects, each with 25 evaluation prompts. Following previous work [41], we generate 4 images per subject and prompt, resulting in a total of 3000 images. These images are evaluated using the DINO [6] and CLIP [36]based metrics which quantify the two objectives of subjectdriven text-to-image generation: subject alignment and text alignment. Subject alignment is measured by the average pairwise cosine similarity of features between generated images and real images using the DINO and CLIP image encoders (DINO, CLIP-I). Text alignment is measured by the pairwise cosine similarity between the CLIP image embeddings of the generated images and the CLIP text embeddings of the target texts (CLIP-T).

##### 4.2. Baseline Comparisons

We compare our method to previous zero-shot subjectdriven text-to-image methods with encoder-based image prompting, including ELITE [51], BLIP-Diffusion [24], Kosmos-G [32], Subject-Diffusion [28], IP-Adapter [55], MS-Diffusion [50], and λ-Eclipse [33]. The details of these models are provided in appendix.

Qualitative Results Our qualitative results are presented in Fig. 5, where the reference images are at the leftmost column and generation results are at the right. Despite using an inpainting approach without any specialized training for subject-driven text-to-image generation, Diptych Prompting generates high-quality samples and accurate renderings

- 1FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
- 2FLUX.1-dev-Controlnet-Inpainting-Beta:

https://huggingface.co/alimama-creative/FLUX.1-dev-ControlnetInpainting-Beta

𝝀-Eclipse IP-Adapter Ours

Reference

BLIP-Diffusion

###### MS-Diffusion

(SD-XL)

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

“a puppy with a blue house in the background”

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

“a grey cat in a police outfit”

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

“a backpack on top of pink fabric”

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

“a wolf plushie on top of dirt road”

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

“a number ‘3’ clock on the beach”

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

“a monster toy in the jungle”

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

“a toy in front of Eiffel Tower"

Figure 5. Qualitative Comparisons. Please zoom in for a more detailed view and better comparison.

Subject Align (%) Text Align (%)

Method win tie lose win tie lose ELITE [51] 77.9 4.3 17.8 75.2 8.6 16.2 BLIP-Diff [24] 73.8 8.6 17.6 77.8 4.3 17.9 λ-Eclipse [33] 80.4 4.2 15.4 74.0 3.3 22.7 MS-Diff [50] 59.3 15.6 25.1 58.9 9.1 32.0 IP-A (SD-XL) [55] 76.2 9.7 14.1 76.2 9.7 14.1 IP-A (FLUX) [55] 69.8 12.0 18.2 65.2 20.6 14.2

Table 1. Human Preference Study. We report results of pairwise comparisons between Diptych Prompting and publicly available baselines in two aspects: subject alignment and text alignment. ‘IP-A’ denotes the abbreviation for IP-Adapter.

of text prompt across diverse subjects and situations, significantly outperforming results compared to previous approaches. Our method also demonstrates impressive performance in capturing the granular details of reference subjects, even with challenging examples containing characteristic fine details, such as a ‘monster toy’ or ‘backpack’.

Human Preference Study We confirm the outstanding performance of our method in terms of human perception through a human preference study. We conduct a paired comparisons of our method with each baseline from two perspectives: subject alignment and text alignment. Using Amazon Mechanical Turk, we collected 450 responses from 150 participants for each baseline and each perspective. As shown in Tab. 1, Diptych Prompting outperforms all baselines by a large margin (p < 0.01 in the Wilcoxon signedrank test), which is consistent with the qualitative results. Detailed information and full instructions about our human preference study are included in appendix.

Quantitative Results For quantitative aspects, the comparison results are in Tab. 2. Diptych Prompting demonstrates comparable or superior performances in both subject alignment and text alignment, as measured by DINO and CLIPT scores. We also note that all baseline methods perform image prompting using the CLIP image encoder, resulting in high CLIP-I scores. In contrast, our inpainting-based zero-shot approach leverages the inherent generation capability of a large-scale TTI model without specialized image encoder, which presents a slight disadvantage in terms of CLIP-I. However, the results from other metrics, qualitative comparisons, and human evaluation studies across both aspects confirm the effective performance and robustness of our method.

##### 4.3. Ablation Studies

To analyze the factors contributing to the performance, we conduct in-depth ablation studies for Diptych Prompting.

Model Selection We validate our method across various base models and inpainting methods including the zero-shot approach [45]. As shown in Tab. 3, we demonstrate that utilizing a high-capacity base model and enhancing the in-

Method Model DINO CLIP-I CLIP-T

ELITE [51] SD-v1.4 0.621 0.771 0.293 BLIP-Diff [24] SD-v1.5 0.594 0.779 0.300 Kosmos-G [32] SD-v1.5 0.694 0.847 0.287 Subject-Diff [28] - 0.711 0.787 0.303 λ-Eclipse [33] Kan-v2.2 0.613 0.783 0.307 MS-Diff [50] SD-XL 0.671 0.792 0.321 IP-Adapter [55] † SD-XL 0.613 0.810 0.292 IP-Adapter [55]‡ FLUX 0.561 0.725 0.351

Diptych Prompting FLUX 0.689 0.758 0.344

- Table 2. Quantitative Comparisons. We compare our method to encoder-based image prompting methods in three metrics. † denotes the obtained value from [33], and ‡ indicates our reevaluation with publicly available weights.

Model Inpainting Scale DINO CLIP-I CLIP-T

SD-3

Zero-shot - 0.475 0.670 0.330 ControlNet 0.95 0.576 0.699 0.326

FLUX

Zero-shot - 0.555 0.720 0.336

ControlNet

0.5 0.628 0.737 0.351 0.8 0.670 0.750 0.349

0.95 0.689 0.758 0.344

- Table 3. Model Selection. We present an ablation results of various base models, inpainting method, and the ControlNet conditioning scale for Diptych Prompting.

Gseg λ DINO CLIP-I CLIP-T ✗ 1.3 0.759 0.783 0.333 ✓ 1.0 0.647 0.745 0.343 ✓ 1.3 0.689 0.758 0.344 ✓ 1.5 0.670 0.750 0.342

- Table 4. Gseg and λ Ablation. We report the ablation results of background removal and reference attention enhancement.

painting method leads to improved zero-shot subject-driven text-to-image generation. From these results, we employ the combination of a robust base model, an effective inpainting method, and an appropriate inpainting-conditioning scale for Diptych Prompting. Integrating advanced base models or inpainting methods is expected to improve the performance and expand our method to more tasks in the future.

Gseg and λ Ablation We conduct additional ablation experiments to verify the effectiveness of background removal in preventing the content leakage and reference attention enhancement in the fine-grained details preservation in Diptych Prompting, as shown in Tab. 4.

When background removal is not applied, we observe copy-and-paste-like results (Fig. 4). These results cause subject alignment metrics to increase significantly due to the mirroring of the reference image, yet at the expense of text alignment, resulting in higher DINO, CLIP-I scores and lower CLIP-T score.

We also assess the impact of varying the rescaling factor

Reference

RB-Mod

IP-Adapter (FLUX)

Ours

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

… in watercolor painting

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

… in flat cartoon illustration style

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

… in kid crayon drawing style

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

… in sticker style

Figure 6. Qualitative Comparisons of Stylized Image Generation. Using a style image as a reference, Diptych Prompting generates stylized images.

λ on reference attention enhancement. Rescaling attention weights between the right panel query and the left panel key helps to capture fine details, thereby improving subject alignment metrics. However, using too high values introduces excessive inductive bias, causing abnormal attention weights that negatively impact performance. Qualitative transitions with respect to λ can be verified in appendix.

##### 4.4. Applications

With the strong capabilities demonstrated by Diptych Prompting, we also explore how it can be applied to tasks beyond subject-driven text-to-image generation.

Stylized Image Generation We extend our method beyond subject images and perform stylized image generation using various style images as references. Using style images and prompts in StyleDrop [43], we employ Diptych Prompting as same, but replace the subject name with the term ‘style’ in diptych text and without attention enhancement (λ = 1) for referencing only the stylistic elements except the content. Diptych Prompting successfully generates the stylistic image reflecting the style of the reference as shown in Fig. 6, and the quantitative comparisons are available in appendix. Subject-Driven Image Editing We further adapt our approach to support inpainting-based subject-driven image editing that modifies the target image with the specific sub-

|[Figure 118]<br><br>[Figure 119]| |
|---|---|
| | |

|[Figure 120]|
|---|

|[Figure 121]|[Figure 122]|
|---|---|
| | |

|[Figure 123]|[Figure 124]|
|---|---|
| | |

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|[Figure 128]|
|---|---|
| | |

|[Figure 129]<br><br>[Figure 130]| |
|---|---|
| | |

|[Figure 131]<br><br>[Figure 132]| |
|---|---|
| | |

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]<br><br>[Figure 136]| |
|---|---|
| | |

|[Figure 137]<br><br>[Figure 138]| |
|---|---|
| | |

|[Figure 139]|[Figure 140]|
|---|---|
| | |

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|[Figure 144]|
|---|---|
| | |

|[Figure 145]|[Figure 146]|
|---|---|
| | |

|[Figure 147]<br><br>[Figure 148]| |
|---|---|
| | |

|[Figure 149]|
|---|

Figure 7. Subject-Driven Image Editing. Diptych Prompting extends to subject-driven image editing by placing the target image on the right panel and masking only the area to be edited.

ject. In this setup, we utilize Diptych Prompting with reference subject image, yet assign the right panel as the editing target image and apply the mask only to the region to be edited. Editing results are shown in Fig. 7. Owing to the capability of Diptych Prompting, edited images effectively preserve unmasked areas while seamlessly integrating the desired subject within the target region.

#### 5. Conclusion

In this paper, we proposed Diptych Prompting, an inpainting-based approach for zero-shot subject-driven textto-image generation. Diptych Prompting performed textconditioned diptych inpainting: the left panel is a reference image containing the subject, and the right panel is inpainted based on a text prompt that describes the diptych containing the desired context. By removing the background and enhancing reference attention, we eliminated unnecessary content leakage and improved subject alignment. This innovative approach enjoyed the inherent properties of large-scale TTI models, achieving superior results over previous methods, particularly in accurately capturing target subjects and representing complex contexts. We also demonstrated the versatility of our method in stylized image generation and subject-driven image editing. Building on these contributions, we anticipate that Diptych Prompting will inspire new directions in image generation and across a wide range of generative tasks, including video and 3D.

Acknowledgement This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) [No. 2022R1A3B1077720], Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University), NO. RS-2022-II220959], the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University in 2024, Samsung Electronics (IO221213-04119-01), and a grant from the Yang Young Foundation.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2

- [2] Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in textto-image diffusion models. In ACM SIGGRAPH 2024 conference papers, pages 1–12, 2024. 3
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 2, 3, 4
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020. 2
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5
- [7] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In International Conference on Machine Learning, pages 4055–4075. PMLR, 2023. 2

- [8] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 5
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 4, 12
- [10] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022. 2
- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 2, 3
- [12] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. In European Conference on Computer Vision, 2024. 2
- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. In International Conference on Learning Representations, 2023. 2, 5
- [14] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4775–4785,

2024. 2, 5

- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [17] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers, 2024. 3
- [18] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469– 12478, 2024. 2
- [19] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024. 3
- [20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 2
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-

- head, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 4
- [22] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of textto-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2, 3
- [23] Black Forest Labs. Flux.1-dev. https : / / huggingface . co / black - forest - labs / FLUX.1-dev, 2024. 1, 2, 3, 4, 5, 12
- [24] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. 2024. 2, 3, 5, 7, 12
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 12

- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 15
- [27] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, 2024. 4
- [28] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3, 5, 7, 12
- [29] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 5
- [30] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. 2023 ieee. In CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6038–6047, 2022. 2
- [31] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, 2022. 2
- [32] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 5, 7, 12
- [33] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept personalized text-to-image diffusion models by leveraging CLIP latent space. arXiv preprint arXiv:2402.05195, 2024. 2, 3, 5, 7, 12

- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

- 2023. 2

[35] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations,

- 2024. 2, 4, 12

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 5, 12
- [37] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 12
- [39] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [40] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 2, 14
- [41] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3, 5, 12, 13
- [42] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [43] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, et al. Styledrop: Text-to-image synthesis of any style. Advances in Neural Information Processing Systems, 36, 2024. 2, 8, 14
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 2
- [45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

- generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 2, 3, 7
- [46] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. In ACM Transactions on Graphics (TOG), 2024. 3
- [47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 2
- [48] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 2, 3
- [49] Jiahao Wang, Caixia Yan, Haonan Lin, Weizhan Zhang, Mengmeng Wang, Tieliang Gong, Guang Dai, and Hao Sun. Oneactor: Consistent subject generation via clusterconditioned guidance. In Advances in Neural Information Processing Systems, 2024. 3
- [50] X Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 2, 3, 5, 7, 12
- [51] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 2, 3, 5, 7, 12
- [52] xiaozhijason. Flux context window editing v3.3f (fill model) fix anything in any context. https://civitai.com/ models / 933018 ? modelVersionId = 1044405,

2024. 3

- [53] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural language. arXiv preprint arXiv:2312.04965, 2023. 2
- [54] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391,

2023. 2, 3, 15

- [55] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 5, 7, 12, 14

- [56] Yu Zeng, Vishal M Patel, Haochen Wang, Xun Huang, TingChun Wang, Ming-Yu Liu, and Yogesh Balaji. Jedi: Jointimage diffusion models for finetuning-free personalized textto-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6786–6795, 2024. 3
- [57] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3, 15

## Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator

### Supplementary Material

#### A. Baselines

We provide the details of the encoder-based image prompting baselines that we compared in human preference study,

- as well as in qualitative and quantitative evaluations. All of them utilize a specialized image encoder which extracts image feature from the reference image and injects it into the TTI model. While these models train the specialized image encoder to enable image prompting for zero-shot subjectdriven text-to-image generation, they compromise subject alignment, especially in the granular details of the subject. For qualitative results and the human preference study, we compare our method only to the baselines with available open-source weights.

- • ELITE3 [51] encodes the visual concepts into textual embeddings, leveraging global and local mapping networks to represent primary and auxiliary features separately, ensuring high fidelity and editability in subject-driven textto-image generation.
- • BLIP-Diffusion4 [24] pre-trains a multimodal encoder following BLIP-2 [25] which produces the text-aligned visual representation of the target subject, and learns the subject representation to enable the TTI model to perform efficient subject-driven text-to-image generation.
- • Kosmos-G [32] aligns the output space of Multimodal Large Language Models (MLLMs) with the CLIP [36] space by anchoring the text modality, and bridges the MLLM with a frozen TTI model using AlignerNet and instruction tuning. As there are no available weights for this baseline, we cannot conduct the human preference study and can only compare using automatic quantitative metrics based on the values reported in their paper.
- • Subject-Diffusion [28] utilizes an image encoder trained on their own large-scale subject-driven dataset to incorporate both coarse and fine-grained reference information into the pre-trained TTI model, enabling high-fidelity subject-driven text-to-image generation without test-time fine-tuning. Subject-Diffusion also has no available opensource weights, so we only conduct the quantitative comparisons with their reported values in the paper.
- • λ-Eclipse5 [33] employs a CLIP-based latent space and image-text interleaved pre-training and contrastive loss to project text and image embeddings into a unified space, preserving subject-specific visual features and reflecting

3ELITE: https://github.com/csyxwei/ELITE 4BLIP-Diff: https://github.com/salesforce/LAVIS/tree/main/projects/blip-

diffusion 5λ-Eclipse: https://github.com/eclipse-t2i/lambda-eclipse-inference

the target text prompt.

- • MS-Diffusion6 [50] introduces a layout-guided framework for multi-subject zero-shot subject-driven text-toimage generation by employing a grounding resampler for detailed feature integration and a multi-subject crossattention mechanism to ensure spatial control and mitigate subject conflicts.
- • IP-Adapter7 8 [55] trains an effective lightweight adapter to enable image prompting for pre-trained TTI models, using a decoupled cross-attention mechanism with separate cross-attention layers for text and image prompts. At the time the IP-Adapter paper was released, SD-v1.5 [38] was used; however, more recent versions, including SDXL [35], SD-3 [9], and FLUX [23], have since been made available. For quantitative comparisons, we referenced the results for the SD-XL version from another study [33], while we conducted our own evaluations for the FLUX version to ensure a fair comparison. In all experiments using IP-Adapter, regardless of the base model version, the conditioning scale is set to 0.6.

#### B. Subject-Driven Text-to-Image Generation

##### B.1. Evaluation Setting

We conduct the main comparisons with baselines on 30 subjects in DreamBench [41]. These consist of 21 objects and 9 live subjects, with 25 evaluation prompts for the objects or live subjects. Diptych Prompting uses the subject name to refer to the target subject and utilizes evaluation prompts that include the subject name for the target description in diptych text. In all zero-shot baselines and our method, we enhance the subject names by adding descriptive modifiers to more accurately refer to the target subjects in the text prompt. The subject names for each subject are summarized as follows in the form of (directory name, subject name):

- • backpack, backpack
- • backpack dog, backpack

- • bear plushie, bear plushie

- • berry bowl, ‘Bon appetit’ bowl

- • can, ‘Transatlantic IPA’ can
- • candle, jar candle
- • cat, tabby cat
- • cat2, grey cat
- • clock, number ‘3’ clock

6MS-Diff: https://github.com/MS-Diffusion/MS-Diffusion

- 7IP-Adapter (SD-XL): https://huggingface.co/h94/IP-Adapter
- 8IP-Adapter (FLUX): https://huggingface.co/XLabs-AI/flux-ip-adapter

0.75

0.80

0.40

0.38

0.70

0.75

0.36

###### CLIP-T

###### CLIP-I

###### DINO

0.65

0.70

0.34

0.60

0.65

0.32

0.55

0.60

0.30

4 8 16 32 Ours

4 8 16 32 Ours

4 8 16 32 Ours

Rank

Rank

Rank

Figure S1. DreamBooth Comparisons. Quantitative comparisons to DreamBooth-LoRA with various rank values.

- • colorful sneaker, colorful sneaker

- • dog1, fluffy dog
- • dog2, fluffy dog
- • dog3, curly-haired dog
- • dog5, long-haired dog
- • dog6, puppy
- • dog7, dog
- • dog8, dog
- • duck toy, duck toy

- • fancy boot, fringed cream boot

- • grey sloth plushie, grey sloth plushie

- • monster toy, monster toy

- • pink sunglasses, sunglasses

- • poop emoji, toy

- • rc car, toy

- • red cartoon, cartoon character

- • robot toy, robot toy

- • shiny sneaker, sneaker

- • teapot, clay teapot
- • vase, tall vase
- • wolf plushie, wolf plushie

ment and text alignment in a zero-shot manner by leveraging FLUX’s capabilities. Notably, this is accomplished without any specialized training for subject-driven text-toimage generation. We also note that the fine details in the target subject are well reflected in the generated results, even for challenging subjects that previous zero-shot methods struggled with (e.g., robot toy, ‘Bon appetit’ bowl).

#### C. Human Preference Study

Following the previous work [41], we perform the human preference study by pairwise comparison in two separate questionnaires for each aspect: subject alignment and text alignment. In both questionnaires, users are presented with a reference image, a target text, and two images generated by each method. They are then asked to select which image better satisfies the desired objective according to the following instructions.

For subject alignment:

- • Inspect the reference subject and then inspect the generated subjects.
- • Select which of the two generated items reproduces the identity (item type and details) of the reference item
- • The subject might be wearing accessories (e.g., hats, outfits). These should not affect your answer. Do not take them into account.
- • If you’re not sure, select Cannot Determine / Both Equally.
- • Which Machine-Generated Image best matches the subject of the reference image?

##### B.2. Comparison with Fine-Tuning-Based Method

To provide a more comprehensive comparison, we also compare with DreamBooth [41], a representative finetuning-based method. For efficient training, we attach a LoRA adapter to the pre-trained FLUX and perform finetuning by training only the LoRA adapter while freezing the FLUX. We train for 300 steps using the Adam optimizer with a learning rate of 1 × 10−4. Additionally, to compare different fine-tuning model capacities, we adjusted the rank of the LoRA adapter and conducted comparative experiments using the same metrics (DINO, CLIP-I, CLIPT). The results are presented in Fig. S1, where our Diptych Prompting demonstrates superior performance across various model capacities.

For text alignment:

- • Inspect the target text and then inspect the generated items.
- • Select which of the two generated items is best described by the target text.
- • If you’re again not sure, select Cannot Determine / Both Equally.
- • Which Machine-Generated Image is best described by the reference text?

##### B.3. Additional Results

We include additional samples of Diptych Prompting in Fig. S2 and Fig. S3 for diverse objects and contexts. As demonstrated in the results, our methodology achieves highquality image generation and satisfies both subject align-

Model Arch Param DINO CLIP-I CLIP-T

SD-v2 U-Net 1.2B 0.504 0.744 0.260 SD-XL U-Net 3.5B 0.941 0.954 0.288 SD-3 MM-DiT 7.7B 0.705 0.821 0.340 FLUX MM-DiT 16.9B 0.720 0.828 0.352

Table S1. Diptych Generation Comparisons. Quantitative comparisons of the diptych generation capabilities of various TTI models based on the total number of parameters, including the autoencoder, main network, and text encoder.

#### D. Diptych Generation

Our framework relies on the emerging property of the largescale TTI model, FLUX, particularly its strong understanding of diptych property and the ability to represent diptych accurately. We verify this by synthesizing a total of 2100 diptychs, using 20 objects, each with a pair of two random prompts for each panel among 15 prompts, and comparing the diptych generation performance with those of other previous TTI models. The prompt for diptych generation follows the setup mentioned in Sec. 3.1 of the main paper. We assessed the quality of each diptych by evaluating the interrelation and text alignment of each panel. This is measured through splitting the generated image in half and measuring DINO and CLIP-I scores between each panel, as well as the CLIP-T score between each panel and its description. The results are shown in Tab. S1, in which the diptych generation performance and total number of parameters including the autoencoder, main network, and text encoders are reported. These results exhibit the superior diptych generation capability of FLUX, where smaller models are insufficient. This allows us to extend to inpainting and propose a zero-shot subject-driven text-to-image generation method via diptych inpainting-based interpretation.

#### E. Background Removal Ablation

We provide additional samples for the ablation study conducted with and without the background removal process Gseg in Fig. S4. Consistent with the findings in the main paper, including the background leads to content leakage, where irrelevant elements such as background, pose, and location are mirrored in the generated results. This hinders the accurate reflection of the desired context described by the text and reduces diversity in pose and location. In contrast, removing the background and retaining only the subject information in the reference image on the left panel allows the generated outputs to better align with the desired context while exhibiting greater diversity in pose and location.”

Method DINO CLIP-I CLIP-T

RB-Mod [40] 0.295 0.598 0.372 IP-Adapter [55] 0.337 0.602 0.371 Diptych Prompting 0.357 0.623 0.349

Table S2. Stylized Image Generation Comparisons. Quantitative comparisons of stylized image generation with previous zeroshot methods.

#### F. Reference Attention Enhancement Ablation

We further present the actual sample quality variations according to the reference attention rescaling factor λ values to support the quantitative ablations in the main paper. These variations are visualized in Fig. S5. As seen in the qualitative results, the absence of reference attention enhancement (λ = 1.0) can lead to a loss of fine details of the subject, resulting in subtle discrepancies such as the left eye of the backpack dog, the patch on its right eye, the fur color on the dog’s face, or the texture of the bear plushie’s fur. As the λ value increases, these missed details are better preserved, leading to improved subject alignment performance. However, excessive enhancement can negatively impact the quality of the generated images, causing the subject to appear slightly blurred or exhibit minor color shifts.

#### G. Stylized Image Generation

For stylized image generation, Diptych Prompting places the style image in the left panel and inpaints the right panel using the text prompt “A diptych with two side-byside images of same style. On the left, {original image description}. On the right, replicate this style exactly but as {target image description}” without attention enhancement (λ = 1.0) for referencing only the stylistic elements except the content. Additional samples are provided in Fig. S6. Beyond the qualitative results, we also include quantitative comparisons using the same metrics (DINO, CLIPT, CLIP-I) applied to a total of 2000 generated images in Tab. S2. These images include 4 samples per prompt and per style image, across 25 prompts and 20 style images collected from previous work [43]. As shown in the result, our method demonstrates comparable results to existing zeroshot style transfer methods specialized in stylized image generation, further proving the versatility of our approach.

#### H. Subject-Driven Image Editing

Diptych Prompting is extended to the subject-driven image editing by placing the reference subject image on the left panel and the editing target image on the right panel in the incomplete diptych. By masking only the desired area in the right panel and applying diptych inpainting, the reference subject from the left panel is generated in the masked

region on the right panel, resulting in the subject-driven image editing. Following the previous work [54], we conduct the subject-driven image editing with selected images from a subset of the MSCOCO [26] validation dataset, in which each image contains a bounding box and the bounding box is smaller than half of image size. We applied masking to the inside of the bounding box, enabling the generation of the reference subject within the specified region. More samples of various subjects and editing target images are available in Fig. S7.

#### I. Limitations

Currently, FLUX is the only model with sufficient capability to effectively generate diptychs. However, as more advanced text-to-image (TTI) models become available, we anticipate that our method will be applicable to a wider range of models in the future. In line with advancements in other encoder-based zero-shot approaches, there is a need to explore multi-subject-driven text-to-image generation. We leave this exploration for future work. Furthermore, diptych generation requires the generated image to have an aspect ratio of 2 : 1. Due to the limitation in the generatable resolution of FLUX, we were unable to produce the diptych image

- at a size of 2048 × 1024 pixels and confirmed results up to 1536 × 768 pixels, resulting in subject-driven image (right panel) being 768 × 768 pixels in size. We expect that this issue can be easily addressed by utilizing super-resolution models such as ControlNet [57] or advanced TTI models for high-resolution image generation in the future.

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

“backpack”

“… on top of a white rug.”

“… with a tree and autumn leaves in the background.”

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

“fluffy dog”

“… with a mountain in the background.”

“… wearing a santa hat.”

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

“toy”

“… on top of green grass with sunflowers around it.”

“…on top of the sidewalk in a crowded street.”

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

“robot toy”

“… floating on top of water.” “… on top of a mirror.”

- Figure S2. Subject-Driven Text-to-Image Generation. More samples of subject-driven text-to-image generation using Diptych Prompting.

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

“’Bon appetit’ bowl” “… in the snow.” “a red …”

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

“duck toy”

“… on a cobblestone street.” “… on top of a purple rug in a forest.”

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

“tabby cat”

“… wearing a rainbow scarf.” “… wearing pink glasses.”

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

“… wearing a black top hat and a monocle.” “long-haired dog” “…in a purple wizard outfit.”

- Figure S3. Subject-Driven Text-to-Image Generation. More samples of subject-driven text-to-image generation using Diptych Prompting..

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

w/ 𝑮𝒔𝒆𝒈

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

w/o 𝑮𝒔𝒆𝒈

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

w/ 𝑮𝒔𝒆𝒈

| |
|---|

[Figure 192]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

w/o 𝑮𝒔𝒆𝒈

Figure S4. Gseg Ablation. Qualitative comparisons with and without the background removal process.

# 1.0 𝝀 1.5

|[Figure 198]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

| |
|---|

|[Figure 205]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

|[Figure 212]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Figure S5. λ Ablation. Qualitative transitions according to the varying λ values. we control the λ from 1.0 (without reference attention enhancement) to 1.5. For a detailed view, please zoom in.

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

“A woman “a butterfly ...” “an Opera house in Sydney …” in 3d rendering style”

“a piano …”

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

###### “Flowers in watercolor painting style”

“a Golden Gate bridge ...” “a man riding a snowboard …” “a boat …”

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

“an f1 race car...” “a bench …” “a robot …”

###### “A mushroom in glowing style”

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

“woman working on a laptop in flat cartoon illustration style”

“a fluffy baby sloth with a knitted hat trying to figure out a laptop, close up ...”

“a cow…” “a panda eating bamboo …”

Figure S6. Stylized Image Generation. More samples of stylized image generation using Diptych Prompting.

|[Figure 235]|
|---|

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

|[Figure 242]|
|---|

|[Figure 243]|
|---|

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]<br><br>[Figure 253]| |
|---|---|
| | |

|[Figure 254]<br><br>[Figure 255]| |
|---|---|
| | |

|[Figure 256]|[Figure 257]|
|---|---|
| | |

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|[Figure 261]|
|---|---|
| | |

|[Figure 262]|[Figure 263]|
|---|---|
| | |

|[Figure 264]<br><br>[Figure 265]| |
|---|---|
| | |

|[Figure 266]|
|---|

###### Figure S7. Subject-Driven Image Editing. More samples of subject-driven image editing using Diptych Prompting.

