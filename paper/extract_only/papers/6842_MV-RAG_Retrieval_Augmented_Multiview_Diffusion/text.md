## MV-RAG: Retrieval Augmented Multiview Diffusion

#### Yosef Dayani Omer Benishu Sagie Benaim

##### Hebrew University of Jerusalem

# arXiv:2508.16577v1[cs.CV]22Aug2025

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

“Bolognese dog”

[Figure 15]

“Bolognese dog”

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|[Figure 24]| |
|---|---|---|
| | | |
|[Figure 25]| |[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

“Labubu doll”

“Labubu doll”

MV-RAG	(Ours) Text-to-Multiview	(MVDream)

Figure 1: We introduce a retrieval-augmented diffusion framework for text-to-multiview generation. Given a text prompt, our method retrieves real-world images and adaptively leverages them together with the text, enabling faithful generation of out-of-distribution and newly emerging objects.

### Abstract

Text-to-3D generation approaches have advanced significantly by leveraging pretrained 2D diffusion priors, producing high-quality and 3D-consistent outputs. However, they often fail to produce out-of-domain (OOD) or rare concepts, yielding inconsistent or inaccurate results. To this end, we propose MV-RAG, a novel text-to-3D pipeline that first retrieves relevant 2D images from a large in-the-wild 2D database and then conditions a multiview diffusion model on these images to synthesize consistent and accurate multiview outputs. Training such a retrievalconditioned model is achieved via a novel hybrid strategy bridging structured multiview data and diverse 2D image collections. This involves training on multiview data using augmented conditioning views that simulate retrieval variance for view-specific reconstruction, alongside training on sets of retrieved real-world 2D images using a distinctive held-out view prediction objective: the model predicts the held-out view from the other views to infer 3D consistency from 2D data. To facilitate a rigorous OOD evaluation, we introduce a new collection of challenging OOD prompts. Experiments against state-of-the-art text-to-3D, image-to-3D, and personalization baselines show that our approach significantly improves 3D consistency, photorealism, and text adherence for OOD/rare concepts, while maintaining competitive performance on standard benchmarks. Project page: https://yosefdayani.github.io/MV-RAG/

Retrieved	Images Output Views

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Preprint. Under review.

[Figure 37]

[Figure 38]

### 1 Introduction

The automated generation of 3D content from textual descriptions is a task of great practical importance for various downstream applications, including game modeling [11, 22], computer animation [36, 21], and virtual reality [46].

Current state-of-the-art approaches predominantly leverage powerful, pre-trained 2D text-to-image diffusion models [55, 13] as strong visual and semantic priors, either by optimization [37, 26, 25] or by using them to train generative models that synthesize consistent multiview images [29, 53, 32]. Indeed, such approaches result in high-quality outputs for a diverse set of objects and scenes.

However, they often falter when confronted with text prompts describing out-of-domain (OOD) or rare entities; in these settings, they often yield geometrically inconsistent results (e.g., poorly rendered unseen regions) or fail to adhere to the text, hallucinating details or substituting rare concepts with common ones.

A dominant text-to-3D approach leverages 2D priors via optimization, such as Score Distillation Sampling (SDS) [37, 26, 25]. These methods typically optimize a 3D representation, such as a NeRF [34], by distilling knowledge from pre-trained 2D text-to-image models. While producing high-fidelity results, SDS-based methods often struggle with geometric consistency and often inherit

- the 2D prior’s failures on OOD/rare prompts, yielding flawed 3D assets.

To mitigate such issues, recent efforts [48, 5] have explored retrieval augmentation within this optimization paradigm by retrieving existing 3D assets to provide explicit geometric priors during the generation. This enhances consistency for concepts in the 3D retrieval database but is limited by its scarce scale and diversity.

Another avenue involves 3D personalization techniques like DreamBooth3D [40], which first adapts a pretrained 2D text-to-image model to a specific subject using a few (3-6) user-provided images, and then applies SDS-based optimization. These methods excel at capturing subject-specific details but primarily operate via inference-time fine-tuning for each individual instance. Crucially, such lifting approaches still suffer from the inherent 3D geometry inconsistency and the optimization time of SDS-based approaches.

An alternative strategy to direct 3D optimization involves feed-forward approaches, particularly multi-view diffusion models [53, 29, 32]. These models aim to synthesize a consistent set of multi-view images from a given input (text or image), which can subsequently be employed for 3D reconstruction. Many such models are fine-tuned from general-purpose 2D diffusion backbones using large-scale 3D datasets such as Objaverse [8], thereby enhancing their 3D awareness and enabling more stable and geometrically consistent generation compared to earlier 2D-lifting approaches. Despite these advances, such models exhibit pronounced limitations when confronted with outof-domain (OOD) or infrequent visual concepts. These challenges can be attributed to two main factors: (1) foundational 2D diffusion models, often used as the initialization point, may offer incomplete or biased representations of atypical concepts, and (2) the 3D datasets employed for finetuning, while extensive, often lack sufficient coverage, diversity, or geometric fidelity for uncommon entities. Consequently, these models tend to produce outputs with reduced photorealism, diminished multi-view consistency, or poor handling of unobserved regions when prompted with rare or novel concepts.

To this end, we develop a multiview diffusion model, MV-RAG, that, guided by an input text prompt, can also effectively condition its output on a set of relevant 2D images retrieved based on that text, ultimately producing a consistent set of multiview images that reflect the prompt’s content. Our training strategy uniquely combines two sources of supervision: high-quality multiview images derived from structured 3D datasets, and diverse images sourced from large-scale 2D text-image collections. When training with multiview 3D data, we note that real-world images retrieved for a given text might look very different from the target multiview ground truth. To simulate retrieval variance, we augment ground truth multiview images (geometrically and semantically) creating diverse, "retrieval-like" conditioning views. Our model then learns to reconstruct the original scene, where the generation of each target view is guided explicitly by attending to visual tokens encoded from retrieved images. This allows for fine-grained conditioning based on these simulated retrieved inputs. To incorporate supervision from 2D data, we utilize K+1 images that are semantically similar to the input text from a given 2D text-image dataset. We then present K of these images as conditional

View 4

View 1

Prompt: "Bluetick Coonhound dog"

Camera Poses

3D Self-Attention

View 4

View 1

In-the-wild

- 2D Database

[Figure 44]

[Figure 45]

[Figure 46]

Image Encoder

Resampler

[Figure 47]

View 4

View 1

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Self Attention Decoupled Cross-Attention

Images features

- Figure 2: Overview of our pipeline. Given a text prompt, we retrieve k relevant images from an in-the-wild 2D image corpus. Local features are extracted from each image, projected through a Resampler and integrated into retrieval-attention modules to guide the multi-view generation process.

inputs, and our model is trained to generate the held-out image. This novel objective pushes our model to infer 3D relationships and consistent appearances directly from sets of unstructured, real-world

- 2D views. Together, our hybrid training approach enables our model to learn robust geometric consistency from the multiview data, while simultaneously benefiting from the vast visual diversity and real-world appearance knowledge embedded in 2D image priors.

To adaptively balance the influence of the base model’s prior and the retrieved image signals, we introduce a fusion mechanism that dynamically adjusts their relative contribution based on how out-of-distribution the input prompt is.

We evaluate MV-RAG on OOD/rare concepts, where standard benchmarks fall short. To this end, as current datasets focus on in-domain objects, we curated a new collection of 196 challenging prompts, corresponding to OOD/rare concepts, with corresponding retrieved images. Using this collection, we experimentally validate our approach against diverse state-of-the-art baselines. These include text-to-3D generation methods, image-to-3D techniques using our best retrieved images as input, and relevant 3D personalization approaches fine-tuned on our retrieved images. Our model demonstrates significant improvements in 3D consistency, photorealism, and text adherence for these challenging OOD/rare prompts, markedly outperforming existing methods, while maintaining competitive performance on standard in-domain benchmarks. Extensive ablation studies further validate the impact of our specific design choices. An illustration of our pipeline is provided in Fig. 1.

Contributions. Our key contributions are: (1) Assessing the OOD problem in multiview generation and presenting a RAG pipeline to address this; (2) Introducing a hybrid training scheme to bridge the gap between decoupled 2D retrievals and 3D objects; (3) Developing the prior-guided attention mechanism to dynamically fuse the base model’s prior with the external signals from retrievals; (4) Introducing OOD-Eval, a new benchmark to facilitate further evaluation in this area.

- 2 Related Work
- 3D Generation Using 2D Diffusion Models Generating 3D content by leveraging strong priors from 2D diffusion models [55, 13] is a dominant paradigm. One major approach optimizes 3D representations, such as Neural Radiance Fields (NeRFs) [34] and more recently 3D Gaussian Splatting (3DGS) [19], via Score Distillation Sampling (SDS) [37, 26, 25, 58, 64], directly distilling knowledge from 2D priors. However, SDS often struggles with geometric consistency and fidelity due to weak 3D awareness in the priors [15, 53]. Feed-forward multi-view diffusion models [53, 29, 32, 16], often fine-tuning 2D diffusion priors with 3D dataset supervision, enhance geometric stability over 2D-lifting techniques by directly generating multiple consistent views [53]. However, these models struggle with out-of-domain (OOD) or rare concepts due to limitations in 2D priors and insufficient 3D training data coverage, leading to reduced photorealism or inconsistent geometry. Related image-to-multiview approaches [29, 52, 60, 31, 32], while effective with single clear inputs, are typically ill-suited for leveraging multiple, varied, unposed retrieved images. Our work builds on these advancements, targeting OOD generation by training a multiview diffusion model to effectively incorporate multiple retrieved 2D images.

###### (a) 3D mode (b) 2D mode

GT Renderings

Augmented Renderings

Cameras Retrieved Images

Held-out view

| | |[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]| | |
|---|---|---|---|---|

[Figure 51]

|[Figure 52]|
|---|

|[Figure 53]|[Figure 54]| | |[Figure 55]| |[Figure 56]|
|---|---|---|---|---|---|---|

[Figure 57]

[Figure 58]

In-the-wild 2D Database

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

[Figure 59]

[Figure 60]

Output

Output

|[Figure 61]|
|---|

[Figure 62]

[Figure 63]

[Figure 64]

MV-RAG

MV-RAG

[Figure 65]

"Siamese cat" "Cartoon Fox"

- Figure 3: Overview of our training scheme. Our model adopts a hybrid training strategy that alternates between two modes. 3D mode (left): A 3D object is rendered to produce ground-truth multi-view images. Additional views are generated and subjected to heavy augmentations to serve as retrievals. These augmented views, along with the target camera parameters and the associated prompt, are provided as input to the model. 2D mode (right): We retrieve K + 1 images from a 2D training corpus, where K images are used as retrievals and one held-out image serves as the target view. In this mode, the model performs 2D self-attention rather than 3D attention, and no target camera parameters are provided.

Retrieval Augmented Generation Retrieval Augmented Generation (RAG) improves generative models by incorporating external information, aiding the handling of rare entities or specialized knowledge without full retraining, a successful paradigm in NLP [23, 4]. Notably, [56] show that RAG is preferable to fine-tuning, especially for out-of-distribution (OOD) or less popular concepts, reinforcing the utility of retrieval-based methods in such settings. In 2D image synthesis, RAG methods similarly use retrieved images or text pairs to enhance fidelity for uncommon concepts or guide generation [6, 51, 3, 49]. Recently, text-to-3D generation methods like RetDream [48] and Sculpt3D [5] retrieve existing 3D assets for geometric priors to improve optimization consistency. However, this is limited by the scarcity and diversity of 3D databases, especially for OOD or rare concepts. Inspired by findings in NLP, our work performs RAG in multiview generation via MV-RAG, leveraging abundant 2D image datasets to condition a multiview diffusion model, offering a more scalable way to ground OOD concept generation in real-world visual data.

Personalization. Personalization techniques adapt generative models to specific subjects using few examples. Originating in 2D text-to-image generation via methods such as embedding optimization or model fine-tuning [10, 45], these concepts extended to 3D. DreamBooth3D [40], for instance, adapts SDS optimization with a personalized 2D prior fine-tuned on few-shot examples. Similarly, multiview diffusion models like MVDream [53] can be fine-tuned with DreamBooth-like principles for subject-specific generation with enhanced multiview consistency. These 3D personalization methods typically rely on inference-time optimization or fine-tuning per subject using a few examples. While designed for specific subjects, such approaches could be applied using retrieved image sets. Our work differs by integrating retrieved 2D image sets representing general concepts directly into our multiview diffusion model’s main training phase, instead of inference-time adaptation with limited examples. This aims to improve the model’s intrinsic ability to generate a broader range of concepts, especially OOD ones, without per-subject, inference-time adaptation.

### 3 Method

Given an input text prompt p, we first retrieve relevant 2D images corresponding to p from a corpus of 2D text-image pairs. Then, we use these images, along with p, to guide the generation process of a multi-view diffusion model. An overview of our approach is provided in Fig. 2. We begin by describing the training process, which involves a data preprocessing stage to prepare distinct conditioning and target data for 2D or 3D training modes, followed by the training of our multiview diffusion model.

#### 3.1 Training Data Preprocessing

Our model training leverages geometric grounding from 3D datasets and diversity from 2D datasets. For both, we prepare 2D conditioning images related to a text p, simulating inference-time retrieval, alongside target supervision data. Fig. 3 illustrates our separate 2D and 3D training modes.

- 2D Data Mode Supervision. For this mode, we utilize a large-scale 2D text-image dataset, focusing on in-the-wild images that are neither posed nor aligned. For a text prompt pi, we consider

K+1 relevant images, from which we designate K as the conditioning retrieved views Iret = {Ii}Ki=1, and the remaining image, Itarget, serves as the target image on which the diffusion loss is computed. This process yields training samples of the form D2D = {p,Iret,Itarget}. No ground truth camera poses are assumed for these images.

- 3D Data Mode Supervision. For 3D data, we assume a dataset comprising text prompts and corresponding 3D object models. For each 3D object, we render a set of N ground truth target

views Itarget = {Ii}Ni=1 at N camera poses C. We follow MVDream and use 4 orthogonal camera poses. To simulate the diverse nature of retrieved conditioning images, we render the object from K

additional random poses followed by a sequence of random augmentations. This yields conditioning views Iret = {Ii}Ki=1. We apply a combination of geometric and semantic augmentations designed to mimic in-the-wild variability and enhance generalization. This yields training samples of the form D3D = {p,C,Iret,Itarget}. See Sec.C for additional details.

#### 3.2 Retrieved and Augmented Image Encoding

A core component of our approach is the effective encoding of the K conditioning images into sequences of conditioning tokens. To achieve this, we employ a pre-trained image encoder, E, specifically the Vision Transformer (ViT) version of CLIP [39], to extract rich features from each conditioning image Ii. We extract the sequence of patch-level (local) features from the penultimate layer of the transformer, Fi = E(Ii), for a richer, spatially descriptive representation over a global embedding. E is frozen and not fine-tuned.

To efficiently incorporate the rich spatial information from each retrieved image, we apply a learnable Resampler module ΘR inspired by the Perceiver Resampler architecture [17] and its use in IP-Adapter variants [63]. This module distills the salient visual information from Fi into a compact set Nt of tokens Ti = ΘR(Fi), using a small set of learnable queries attending to Fi. Following prior work, we set Nt = 16. This enables effective conditioning of the diffusion model while reducing computational overhead. The resulting token sequences are then used for conditioning the cross-attention layers, as described next.

#### 3.3 Retrieval-Conditioned Multiview Diffusion

The encoded tokens are then fed into a multiview diffusion model, based on MVDream [53], which extends a 2D text-to-image U-Net architecture for multiview generation. Following MVDream, we incorporate camera pose embeddings for geometric guidance and modify the U-Net’s self-attention layers. These layers are inflated to operate jointly over features from all generated views, forming a 3D-aware self-attention mechanism that promotes multiview consistency.

While MVDream relies solely on text-based cross-attention, we replace this mechanism with a decoupled cross-attention module that incorporates encoded tokens from both the text prompt and the retrieved images. Specifically, the tokens from the conditioning images are processed by a dedicated, trainable cross-attention branch, yielding retrieval-guided features denoted as fret.

For this cross-attention branch, we follow the design of IP-Adapter [63], where the U-Net query features Qi (generated via a shared query projection θQ) attend separately to keys and values from the retrieved tokens Ti and the text embedding. The retrieved tokens are processed through learnable projections θK

to produce fret, while the text embedding is processed through frozen projections θK

and θV

ret

ret

, inherited from a pretrained MVDream model, yielding ftxt. We note that the shared query projection θQ is also frozen.

and θV

txt

txt

This results in a decoupled cross-attention mechanism. During training, we integrate the text and retrieval features as f = λftxt + fret, where λ is a hyperparameter. Empirically, we find that small values of λ ease the adaptation of the newly introduced retrieval branch during training.

#### 3.4 2D and 3D Training Modes

Our full architecture then comprises the encoder, resampler, and U-net-based architecture (comprising of self-attention and decoupled cross-attention layers). This architecture is trained jointly using the two data modes described below:

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

“Banana passionfruit, passiﬂora supersect”

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

| |[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]| |[Figure 94]<br><br>[Figure 95]|
|---|---|---|---|

“Onagadori chicken”

[Figure 96]

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

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

“Bearded Collie dog”

Base Model Output

- Figure 4: Illustration of Prior-Guided Attention. The base model’s activations are leveraged proportionally to its prior knowledge of the object, controlled by the adaptive parameter α. For an object where the base model has limited prior knowledge, a lower adaptive α reduces reliance on the base model. For an object well-represented by the base model, insufficient weighting (e.g., fixed α = 0.3) degrades results.

3D Data Mode: Multiview Reconstruction. When training with samples derived from 3D assets, the model reconstructs a set of predicted views given their camera poses C. The U-Net’s self-attention layers operate across all N view latents, enforcing cross-view consistency. Each target view is conditioned on its camera pose, and the text prompt p provides global guidance via its features ftxt. The visual tokens aggregated from all K augmented conditioning images Iret are used to compute the retrieval attention features fret, jointly guiding the generation of all N target views. A multiview reconstruction loss, LMV (θ,p,C,Iret,Ipred), is applied across all target views.

- 2D Data Mode: Held-out View Prediction. When training with samples from 2D datasets, the objective is to predict the single held-out image Itarget based on the text prompt p and tokens from the K conditioning retrieved images Iret. In this scenario, as only a single target view is generated, the U-Net’s self-attention layers inherently function as standard 2D self-attention, operating within that single view’s features. The text prompt p and the tokens from the retrieved images provide

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

- (a)

- (b)

- (c)

|ret|
|---|

[Figure 118]

conditioning via ftxt and f

r respectively. Crucially, consistent with MVDream’s training on 2D data, no explicit camera pose information is provided.

[Figure 119]

[Figure 120]

[Figure 121]

#### 3.5 Inference Process

Retrieved Images

At inference time, given an input text prompt p, we first retrieve the top K relevant 2D images Iret from our diverse 2D database using the BM25-based text similarity approach. To improve relevance, we compute prompt-caption similarity and discard images below a threshold, yielding K′ ≤ K images.

[Figure 122]

[Figure 123]

|the|
|---|

|If no|
|---|

threshold, we disable retrieval-attention and fall back to the base model. These K images are then encoded into visual tokens as detailed in Section 3.2. Our trained multiview diffusion model then generates N consistent views conditioned on the text prompt p and the set of retrieved tokens, utilizing specified camera poses for the target views.

I images pass

Base Model Output Ours

Prior-guided attention. We introduce an adaptive fusion coefficient α that dynamically balances the influence of the model’s prior knowledge and the retrieved external signals, based on how out-of-distribution (OOD) a prompt is. Diffusion models learn to approximate the score function ∇x log p(x|y), the gradient of the log data density which by definition points toward higherprobability regions of the data distribution [54, 14]. Thus, if a concept is in-domain, the base model’s score will guide denoising toward an accurate reconstruction, whereas for OOD concepts the reconstruction will deviate.

To estimate α during inference, without ground-truth multiview, we first perform a short forward pass using only the base model’s text-based attention ftxt (with the retrieval module disabled) for 10 DDIM steps, generating an initial candidate output. We then measure its similarity to the retrieved

Table 1: Quantitative evaluation on OOD/rare concepts. The models’ performance is assessed on four orthogonal views. Multiview consistency is evaluated by reconstructing the 3D object from the multiview images. See Sec. 4 for further details.

4-Views Re-rendered Method CLIP ↑ DINOv2 ↑ IR ↑ FID ↓ IS ↑ CLIP ↑ DINOv2 ↑ IR ↑ FID ↓ IS ↑ Text-to-3D MVDream [53] 66.47 33.12 58.01 76.71 10.62 70.83 28.66 58.98 96.29 11.39 MV-Adapter [16] (TX) 66.48 28.53 58.42 84.28 9.55 71.33 24.30 56.14 106.66 11.23 SPAD [18] 65.23 19.39 48.54 167.49 9.18 64.46 12.29 43.80 176.66 8.90 TRELLIS [62] (TX) 67.96 21.11 51.01 160.93 6.90 67.16 16.87 51.82 154.43 8.09 Image-to-3D

ImageDream-P [60] 69.20 45.01 65.64 68.40 12.11 70.44 32.77 60.17 103.24 12.84 ImageDream-L [60] 67.55 39.48 63.93 84.69 9.45 70.16 29.60 58.66 120.37 10.42 MV-Adapter [16](IM) 69.74 49.14 71.05 72.71 12.88 71.53 35.25 60.36 107.95 12.64 Era3D [24] 69.13 42.41 64.42 92.68 15.26 71.00 35.65 60.81 93.97 14.45 TRELLIS [62] (IM) 70.31 35.24 59.32 167.61 11.38 67.86 24.43 52.23 146.82 10.72

- 3D Personalization MVDreamBooth [53] 66.14 36.22 55.09 82.73 11.55 68.38 27.91 54.33 107.07 11.92 MV-RAG (Ours) 71.77 50.19 67.41 54.79 13.20 74.28 39.61 66.59 80.54 12.33

images using DINOv2 similarity [35], which serves as a proxy for the base model’s confidence in capturing the concept: high similarity indicates in-domain content, so α favors ftxt; low similarity suggests OOD, shifting weight toward the retrieval-based attention fret. The two sources are fused adaptively as:

f = α · ftxt + (λ′ − α) · fret, for a hyperparameter λ′, replacing the f calculation used in training. The full model is then run with the retrieval module enabled to generate the final outputs. Fig. 4 illustrates its effect.

- 4 Experiments

We evaluate our approach against relevant state-of-the-art baselines, as detailed below, both quantitatively and qualitatively, considering both OOD/rare and in-domain concepts.

Benchmarks As current benchmarks lack OOD/rare concept coverage, we curated 196 examples from Wikipedia Commons [61] (not used in training).

Each example consists of a text prompt and multiple 2D retrieved images of the same concept. Importantly, texts were chosen to be far from any text (or concepts) seen during training. We call this evaluation set “OOD-Eval". See Sec.E for additional details and examples. We also consider in-distribution objects, demonstrating that our success in OOD concepts is not compensated by worse in-domain results. To this end we consider a curated set of 50 in-domain objects from Objaverse-XL [7]. For retrieval, we consider 2D images from the LAION-400M dataset [47]. For each text, we retrieve four 2D images. Unlike for OOD-Eval, we also have corresponding ground truth multiview images. We call this evaluation set “IND-Eval".

Baselines We compare MV-RAG against three categories of state-of-the-art methods. (1) Textto-multiview generation: MVDream [53] (closest in architecture to ours), MV-Adapter [16] (textconditioned), SPAD [18], and TRELLIS [62] (text-conditioned). (2) Image-to-multiview generation applied to the retrieved 2D views: ImageDream [60], MV-Adapter [16] (image-conditioned), Era3D [24], and TRELLIS [62] (image-conditioned). (3) 3D personalization models: we adopt MVDream’s optimization-based personalization approach [53, 45], applied to all k retrieved views. Further details are provided in Sec.C.3.

#### 4.1 Quantitative Evaluation

Metrics We assess generation quality with Inception Score (IS) [2] and FID [12] on the output poses. To assess alignment to the input text, a natural choice would be to consider the CLIP [39] similarity between the input text and the output multiview images produced by the model. However, we found that CLIP [39] (specifically in image-text similarity) is unable to score rare/OOD concepts well, often assigning a low score for such text-image pairs. See Figure. 8 and C.4 for further discussion and

Retrieved Images No 2D Mode No 3D Mode 2D + 3D No Aumentations 2D + 3D Modes (Ours)

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

[Figure 142]

[Figure 143]

- Figure 5: Hybrid training ablations. For the text prompts "Ratonero Bodeguero Andaluz dog" (top) and "Markhor goat" (bottom), we consider the output of our model when trained without our 2D/3D schemes in comparison to our full approach (shown on the RHS).

illustration. This is also demonstrated in [66]. As such, we evaluate image-image similarity between the generated views and held-out ground truth retrieved examples from our evaluation benchmark. Specifically, we compute the average similarity using CLIP [39] and DINOv2 [35], both widely used for image representation. Additionally, we employ an Instance Retrieval (IR) model [50] specifically trained to embed images of the same object instance close together in feature space, making it a more suitable choice for assessing entity-level visual alignment.

To evaluate 3D consistency, we adopt the procedure of [60], measuring how well a reconstruction model aligns with our generated views. Specifically, we sample four views and train a reconstruction model [57] on these “training views.” The model then predicts novel views (re-rendered in Fig. 1), whose fidelity and alignment with the training views are assessed using the metrics described above. Inconsistent multiview generations are expected to degrade reconstruction quality, leading to lower fidelity and alignment scores. Further details are provided in Sec.D.

Table 2: User study. A user study depicting (Q1-Realism), (Q2-Alignment) and (Q3-3D Consistency) reporting MOS (1low, 5-high).

Q1 ↑ Q2 ↑ Q3 ↑

MVDream [53] 1.96 1.85 3.24 ImageDream-P [60] 2.25 2.6 3.03 MV-RAG (Ours) 4.12 4.44 4.44

User study. To complement the quantitative metrics, we conduct a user study on prompts from the OOD-Eval dataset. For each prompt, we present four generated views from our model and the baselines, and ask participants to evaluate: (Q1) Realism — How realistic are the generated views? (Q2) Alignment — How well do the views match the input text? (Q3) 3D Consistency — How consistent are the views across different perspectives? For each question, outputs from all methods are shown in random order, and participants rank them on a scale of 1–5. Additional details are provided in Sec. D.

#### Table 3: Quantitative evaluation on in-domain concepts.

Evaluation on OOD/rare concepts As shown in Tab.1, MV-RAG achieves strong performance across both evaluation modes. In the

Model PSNR↑ SSIM↑ LPIPS↓ CLIP↑ SigLIP↑ Text-to-3D

- 4-views setting, it outperforms all baselines on CLIP, DINO, and FID, while ranking second on IR (behind MV-Adapter (IM)) and IS (behind Era3D). In the more challenging rerendered setting which also reflects 3D consistency, MVRAG leads on CLIP, DINO, IR, and FID, with Era3D attaining a higher IS. Notably, MVDream and ImageDream, which share similar architectures but lack retrieval, consistently underperform across metrics. The user study results in Tab.2 further corroborate MV-RAG’s advantage, showing clear gains in realism, text alignment, and 3D consistency.

MVDream 16.95 0.717 0.363 64.25 34.81 MV-Adapter (TX) 15.37 0.632 0.459 59.62 30.18 SPAD 8.34 0.619 0.468 61.32 29.12 TRELLIS (TX) 16.53 0.743 0.327 60.67 30.98

###### Image-to-3D

ImageDream-P 15.50 0.728 0.400 60.89 31.67 ImageDream-L 15.64 0.732 0.393 61.72 32.52 MV-Adapter (IM) 15.24 0.646 0.448 61.46 32.00 Era3D 12.44 0.722 0.378 58.79 29.94 TRELLIS (IM) 16.02 0.741 0.378 55.39 26.72

###### 3D Personalization

MVDreamBooth 16.31 0.716 0.381 61.68 32.14 MV-RAG (Ours) 16.63 0.730 0.362 64.48 35.34

Evaluation on in-domain concepts We evaluate MV-RAG against all methods on the INDEval benchmark, which contains objects from Objaverse [7] dataset which is used for training in

all baselines. Reconstruction quality is measured using PSNR, SSIM, and LPIPS with respect to the ground-truth views in IND-Eval, while text-image alignment is assessed via CLIP [39] and SigLIP [65] similarity between the generated outputs and the input prompt. As shown in Tab. 3, MV-RAG achieves results that are on par with, or slightly surpass, those of the baselines.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

MVDreamMV-Adapter(TX)Era3DSPADMV-Adapter(IM)ImageDream

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Images Image-to-3DText-to-3D

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

TRELLIS(IM)TRELLIS(TX)MVDreambooth

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

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

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

###### 3Dpersonalisation

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

###### MV-RAG(Ours)

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

| |[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]|[Figure 232]<br><br>[Figure 233]| |
|---|---|---|---|

| |[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]| |
|---|---|---|

|[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]| |
|---|---|

|[Figure 243]|
|---|

[Figure 244]

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

Reference

“Peugeot 202” "Braque du Bourbonnais dog"

"Rhingia Campestris hoverfly"

"IBM 5100 Portable Computer"

- Figure 7: Qualitative evaluation. Text-to-3D models fail to generate unseen (OOD) concepts, while image-to-3D models fail to reconstruct their correct 3D structure from a single view. Existing personalization methods cannot effectively leverage the diversity of retrieved images.

#### 4.2 Qualitative Evaluation

In Fig.7, we compare MV-RAG qualitatively against all baselines. Additional comparisons are provided in the appendix section (Figs.14, 15). Text-to-3D models, which rely solely on text prompts, struggle with out-ofdistribution (OOD) object structures due to the lack of visual priors for rare instances, often failing to capture key attributes or correct geometry. Single-reference image-to-

51

13.2

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

13.0

49

DINOv2Score

ISScore

47

12.7

45

12.5

1 2 3 4 5 Number of Retrieved Images

1 2 3 4 5 Number of Retrieved Images

##### 3D methods are limited by their single viewpoint, constraining pose diversity and missing occluded regions. Variations in lighting, texture, and visibility further degrade consistency and realism, restricting plausible reconstructions to views near the input perspective. Although MVDreamBooth leverages multiple reference images, it fails to adequately handle variation across them (e.g., differing car colors) and struggles to construct accurate 3D structures, highlighting the challenge of integrating diverse visual cues effectively.

Figure 6: Effect of the number of retrieved images on alignment and fidelity.

MV-RAG addresses these by leveraging multiple unposed images from a large 2D corpus, providing complementary viewpoints that enrich generation with diverse, relevant visual cues. Crucially, our framework is designed to isolate view-invariant attributes such as object identity and disentangle nuisance factors like illumination, occlusion, and background clutter. As can be seen in Fig. 7, this naturally results in more diverse and detailed multi-view outputs.

Diversity and Utility. MV-RAG enables diverse outputs, by using a different seed for a given input text and set of retrieved views. This is shown in Fig. 9 (a). In Fig. 9(b) we also demonstrate the MV-RAG utilizes different parts from retrieved views to generate target views

#### 4.3 Ablation Study

Hybrid training In Fig. 5, we present a qualitative ablation of our 2D mode, 3D mode, and augmentations. Without the 2D mode, the model struggles to separate the object from its in-the-wild background, leading to artifacts (e.g., a floating leash on a dog, a goat merged with a rock). Without

- the 3D mode, it fails to consistently distribute visual features across views, producing inaccurate shapes (e.g., tail or horn) and background inconsistencies. Removing augmentations still allows handling in-the-wild settings through the 2D mode but reduces robustness to the high variance in retrieved images, often yielding incorrect 3D structures.

Table 4: Comparison of retrieval approaches for out-of-domain retrieval.

Number of Retrieved Images Fig. 6, considers the effect of number of retrieved images on alignment/fidelity. As seen, using 4 views provides best overall performance.

#### Method Precision@5 ↑

Retrieval Method In Table 4, we ablate the retrieval method used to obtain visual exemplars for conditioning. We evaluate retrieval performance using the OOD-Eval text prompts, where the goal is to retrieve the correct imagetext pair from a combined collection of OOD-Eval and MS-COCO [28]. Specifically, we compare four methods: CLIP [39] with text-to-image similarity (TX-IM) and textto-text similarity (TX-TX), SigLIP [65] with TX-IM similarity, and BM25 [43], a non-semantic bag-of-words model.

CLIP (TX-IM) 0.5366 CLIP (TX-TX) 0.7306 SigLIP (TX-IM) 0.7889 BM25 (TX-TX) 0.8522

We observe that semantic retrieval models like CLIP and SigLIP struggle with rare concepts due to limited exposure and weak conceptual grounding. As shown in Figure 8, CLIP often assigns high similarity scores to incorrect generations or generic category matches, failing to distinguish nuanced, rare object names from broad class labels (see Table 5). This issue directly carries over to retrieval: when queried with OOD descriptions, semantic models frequently return irrelevant results due to their inability to treat specific rare terms as meaningful. In contrast, BM25 relies purely on lexical overlap, matching keywords without relying on learned semantic priors. This makes it more robust in OOD scenarios where the semantics are underrepresented or misaligned in vision-language models.

### 5 Conclusion

We present a novel retrieval-augmented multiview diffusion model for text-to-3D generation. By retrieving relevant 2D images from a large-scale database and conditioning a multiview diffusion model on these images, our approach produces consistent and accurate multiview outputs—especially for out-of-domain (OOD) or rare concepts where prior methods often struggle. Central to our approach is a hybrid training scheme that effectively integrates structured multiview data with diverse 2D image collections, leveraging augmented conditioning views and a unique held-out view prediction objective. To thoroughly assess performance on challenging scenarios, we introduce a new benchmark of OOD prompts. Experimental results demonstrate that our method significantly improves 3D consistency, photorealism, and text alignment for OOD and rare concepts, while maintaining strong performance on standard benchmarks.

### References

- [1] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.
- [2] S. Barratt and R. Sharma. A note on the inception score. arXiv preprint arXiv:1801.01973, 2018.
- [3] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and B. Ommer. Retrievalaugmented diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [4] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J.-B. Lespiau, B. Damoc, A. Doucet, M. Bärtschi, A. Méray, J. Roffi, A. Glaese, J. W. Noland, A. Cassirer, A. Clark, L. Guy, D. Budden, T. Hennigan, S. Osindero, L. Rimell, M. Tsimpoukelli, K. Simonyan, L. Sifre, S. Dieleman, and N. De Freitas. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning (ICML), pages 2206–2240. PMLR, 2022.
- [5] C. Chen, X. Yang, F. Yang, C. Feng, Z. Fu, C.-S. Foo, G. Lin, and F. Liu. Sculpt3d: Multi-view consistent text-to-3d generation with sparse 3d prior. arXiv preprint arXiv:2403.09140, 2024.
- [6] W. Chen, H. Hu, C. Saharia, and W. W. Cohen. Re-Imagen: Retrieval-Augmented Text-to-Image Generator. arXiv preprint arXiv:2209.14491, 2022.
- [7] M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati, A. Fan, C. Laforte, V. Voleti, S. Y. Gadre, E. VanderBilt, A. Kembhavi, C. Vondrick, G. Gkioxari, K. Ehsani, L. Schmidt, and A. Farhadi. Objaverse-xl: A universe of 10m+ 3d objects, 2023.
- [8] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.
- [9] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou. The faiss library, 2025.
- [10] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations (ICLR), 2023.
- [11] J. Gregory. Game Engine Architecture. A K Peters/CRC Press, third edition, 2018.
- [12] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018.
- [13] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 6840–6851, 2020.
- [14] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models, 2020.
- [15] S. Hong, W. Jang, I. H. Kim, I. Kim, and S. Kim. Variational score distillation for text-to-3d generation. arXiv preprint arXiv:2312.09334, 2023.
- [16] Z. Huang, Y.-C. Guo, H. Wang, R. Yi, L. Ma, Y.-P. Cao, and L. Sheng. MV-Adapter: Multi-View Consistent Image Generation Made Easy. arXiv preprint arXiv:2412.03632, 2024.
- [17] A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and J. Carreira. Perceiver: General perception with iterative attention. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139, pages 4651–4664. PMLR, 2021.
- [18] Y. Kant, Z. Wu, M. Vasilkovsky, G. Qian, J. Ren, R. A. Guler, B. Ghanem, S. Tulyakov, I. Gilitschenski, and A. Siarohin. Spad : Spatially aware multiview diffusers, 2024.
- [19] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis. 3d gaussian splatting for real-time radiance field rendering. In ACM SIGGRAPH 2023 Conference Proceedings, 2023.
- [20] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick. Segment anything. arXiv:2304.02643, 2023.
- [21] J. Lasseter. Principles of traditional animation applied to 3d computer animation. ACM SIGGRAPH Computer Graphics, 21(4):35–44, 1987.

- [22] J. P. Lewis, M. Jacobson, A. Witkin, and M. Cohen. Real-time rendering. ACM SIGGRAPH Course Notes, 2002.
- [23] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020.
- [24] P. Li, Y. Liu, X. Long, F. Zhang, C. Lin, M. Li, X. Qi, S. Zhang, W. Luo, P. Tan, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. arXiv preprint arXiv:2405.11616, 2024.
- [25] Y. Liang, X. Sun, Z. Lai, Z. Zhang, J. Wang, and J. Hu. LucidDreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20795–20805, 2024.
- [26] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin. Magic3D: High-resolution text-to-3d content creation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 300–309, 2023.
- [27] J. Lin, X. Ma, S.-C. Lin, J.-H. Yang, R. Pradeep, and R. Nogueira. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021), pages 2356–2362, 2021.
- [28] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.
- [29] R. Liu, R. Wu, B. V. Hoorick, P. Tokmakov, S. Zakharov, and C. Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In International Conference on Computer Vision (ICCV), pages 9298–9309, 2023.
- [30] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [31] Y. Liu, C. Lin, Z. Zeng, X. Long, L. Liu, T. Komura, and W. Wang. Syncdreamer: Generating multiviewconsistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023.
- [32] X. Long, Y.-C. Guo, C. Lin, Y. Liu, Z. Dou, L. Liu, Y. Ma, S.-H. Zhang, M. Habermann, C. Theobalt, and W. Wang. Wonder3D: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21685–21696, 2024.
- [33] I. Loshchilov and F. Hutter. Decoupled weight decay regularization, 2019.
- [34] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis, 2020.
- [35] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.
- [36] R. Parent. Computer Animation: Algorithms and Techniques. Morgan Kaufmann, third edition, 2012.
- [37] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. DreamFusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations (ICLR), 2023.
- [38] L. Qiu, G. Chen, X. Gu, Q. Zuo, M. Xu, Y. Wu, W. Yuan, Z. Dong, L. Bo, and X. Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9914–9925, 2024.
- [39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139, pages 8748–8763. PMLR, 2021.
- [40] A. Raj, S. Kaza, B. Poole, M. Niemeyer, A. Van Den Oord, S. Fidler, and A. Holynski. Dreambooth3d: Subject-driven text-to-3d generation. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 4342–4352, 2023.

- [41] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang. Grounded sam: Assembling open-world models for diverse visual tasks, 2024.
- [42] T. Ridnik, E. Ben-Baruch, A. Noy, and L. Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972, 2021.
- [43] S. E. Robertson, S. Walker, S. Jones, M. Hancock-Beaulieu, and M. Gatford. Okapi at trec-3. In Text Retrieval Conference, 1994.
- [44] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models, 2022.
- [45] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22500–22510, 2023.
- [46] M. J. Schuemie, P. van der Straaten, M. Krijn, and C. van der Mast. Research issues for virtual reality. CyberPsychology & Behavior, 4(2):239–245, 2001.
- [47] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [48] J. Seo, S. Hong, W. Jang, I. H. Kim, M. Kwak, D. Lee, and S. Kim. Retrieval-augmented score distillation for text-to-3d generation. In International Conference on Machine Learning (ICML), 2024.
- [49] R. Shalev-Arkushin, R. Gal, A. H. Bermano, and O. Fried. ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation. arXiv preprint arXiv:2502.09411, 2025.
- [50] S. Shao and Q. Cui. 1st place solution in google universal images embedding. arXiv preprint arXiv:2210.08473, 2022.
- [51] S. Sheynin, O. Ashual, A. Polyak, U. Singer, O. Gafni, E. Nachmani, and Y. Taigman. kNN-Diffusion: Image Generation via Large-Scale Retrieval. arXiv preprint arXiv:2204.02849, 2022.
- [52] R. Shi, H. Chen, Z. Zhang, M. Liu, C. Xu, X. Wei, L. Chen, C. Zeng, and H. Su. Zero123++: A single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.
- [53] Y. Shi, P. Wang, X. Cun, R. Shao, and C. Chen. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
- [54] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution, 2020.
- [55] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.
- [56] H. Soudani, E. Kanoulas, and F. Hasibi. Fine tuning vs. retrieval augmented generation for less popular knowledge. In Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 12–22, 2024.
- [57] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024.
- [58] J. Tang, D. Huang, J. Zhang, Y. Liu, L. Wang, G. Wang, H. Liu, H. Wang, and X. Chen. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In International Conference on Learning Representations (ICLR), 2024.
- [59] A. Trotman, A. Puurula, and B. Burgess. Improvements to bm25 and language models examined. Proceedings of the 19th Australasian Document Computing Symposium, 2014.
- [60] P. Wang and Y. Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023.
- [61] Wikimedia Commons. Wikimedia commons. https://commons.wikimedia.org/wiki/Main_Page,

2025. Accessed: 2025-05-13.

- [62] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang. Structured 3d latents for scalable and versatile 3d generation, 2025.
- [63] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv preprint arXiv:2308.06721, 2023.
- [64] T. Yi, J. Zhang, Z. Huang, Y. Liu, G. Chen, J. Zhang, S. Chen, J. Jia, Y. Chen, and G. Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [65] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [66] S. Zhu, Y. Zhang, X. Tian, and X. Sun. Prompt reverse learning: Enhancing visual language models for rare image recognition.

- A Appendix
- B Additional Qualitative Evaluation

We present additional qualitative evaluations, corresponding to Fig. 4 of the main paper in Fig. 14 and Fig. 15. Additional examples for output of our method are provided in Fig. 12 and Fig. 13.

- C Training, Inference and Implementation Details

#### C.1 Data Preparation

For 3D mode training, we utilize multi-view synthetic renderings from the public Objaverse dataset [38, 8] along with the associated camera parameters. We randomly sampled a subset of 90K objects. For each object, we select four orthogonal views with elevation angles in the range [−5◦,30◦] for supervision. Additionally, we sample 2-3 random views to simulate retrieval, as detailed below. Objaverse contains a wide variety of objects, including both high-fidelity, photorealistic assets and low-textured, abstract ones. To improve the model’s robustness to real-world, non-synthetic data, we apply an aesthetic-based filtering criterion. This criterion incorporates color diversity, texture complexity, and multi-view consistency, which then results in about 65K objects. Following the preprocessing protocol of MVDream, we resize all rendered images to 256 × 256 pixels and replace empty backgrounds with a random gray color. Camera poses are normalized onto a unit sphere by removing translational components.

To simulate retrieval images, we apply a series of augmentations to the additional rendered views. These include perspective distortion, random rotations, resized cropping, and color jitter. To further enhance realism, we employ an image-variation model [63, 44] to generate semantically and visually diverse variants of the same object. In total, we obtain four simulated retrieval images per object. For 2D mode training, we use the ImageNet21K dataset[42], which comprises over 21K semantic classes with multiple images per class. To improve the visual coherence within classes, we use a large language model (GPT-4o) to filter and retain only visually unified categories (e.g., carpet shark, toilet bowl) and exclude abstract or overly broad classes (e.g., human, cycling). This results in a curated subset of 516 visually consistent categories. For each selected class, we sample one target image for supervision and four additional images from the same class to serve as retrieved images.

#### C.2 Training

We fine-tune our model using the AdamW optimizer[33] with a learning rate of 5 × 10−6 and a batch size of 24 for approximately 11,000 steps. Training is performed in an alternating scheme between 2D and 3D modes, allocating an equal number of steps to each mode. As in MVDream, we append ", 3d asset" to the text prompt during 3D mode to help the model distinguish between the two training regimes. The model is initialized from the Stable Diffusion 2.1-based MVDream checkpoint, which remains frozen throughout training. The adapter modules are initialized from the ImageDream checkpoint. We fine-tune both the retrieval-attention modules and the Resampler. For the image encoder, we use OpenCLIP ViT-H/14, which is kept frozen during training. The training was done on a single NVIDIA A100 GPU, with a total training time of approximately 3 hours.

#### C.3 Baselines

For all baselines we use the official implementations and publicly available pretrained checkpoints provided by the respective authors, with the exception of MVDreamBooth, for which training code is not released. For each baseline, we generate 4 views using fixed orthogonal camera angles and elevations, employing the DDIM sampler with 50 steps and a classifier-free guidance (CFG) scale of 5. For image-to-3D baselines, we preprocess the retrieved reference images by segmenting out the background using Grounded-SAM [20, 30, 41]. Among the reference images, we select the one that yields the highest multi-view consistency based on DINO score for evaluation. To ensure a fair comparison in image-image similarity metrics, we compare semantic features against the segmented ground-truth object views.

MV-RAG	(Ours) Text-to-Multiview	(MVDream)

“Bolognese dog”

“Bolognese dog”

“Labubu doll”

“Labubu doll”

[Figure 272]

[Figure 273]

Retrieved	Images Output Views

Retrived Images Zoom in Multiview Images

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Back View

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Side View

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Front View

(a) (b)

- Figure 9: (a). Utility. Our approach learns to utilize all relevant information in retrieved views. On the LHS, we show retrieved views. The middle columns are zoom ins, for aspects used in generation, and the RHS shows back view (top), side view (middle), and front view (bottom). (b). Diversity. Unlike image-prompted methods, our method enables diverse outputs, by using different seeds for a given input text and a fixed set of retrieved images. For the prompt "Peugeot 202", the LHS shows retrieved views, and the RHS shows a single view output (using the same pose) for 4 different seeds.

Austrian Pinscher dog

For the MVDreambooth baseline, we follow the method described in [53], training a separate MVDreamBooth model for each instance in the OOD-Eval set. Each model is optimized for 600 steps. To preserve class identity, we apply a class-preservation loss using ImageNet class names (e.g., dog, car) when available, and default to the prompt when no corresponding class is derived.

[Figure 291]

[Figure 292]

[Figure 293]

62.34 78.96 57.39

Kai Ken dog

[Figure 294]

[Figure 295]

[Figure 296]

#### C.4 Metrics

53.13 58.36

53.15

Stout Scarab minivan

- Figure 8 highlights three representative failure cases of using CLIP text-image similarity as an evaluation metric for out-of-distribution (OOD) objects. In each case, MVDream receives a higher CLIP score than both our model and even the ground truth image, despite generating outputs that are visually or semantically incorrect. We hypothesize that this stems from CLIP’s limited prior knowledge of rare concepts and the fact that models like MVDream are optimized to align with CLIP-based features, potentially leading to overfitting to incorrect semantic associations. Further, as shown in Table 5, we find that for rare concepts, CLIP assigns nearly identical similarity scores to both detailed object names and their coarse class labels. This suggests that CLIP does not treat the additional semantic information in rare object names as meaningful, highlighting a lack of conceptual grounding for these OOD categories. In contrast, for in-domain objects, CLIP shows much stronger separation between specific and generic labels, reinforcing its limitations in recognizing and evaluating uncommon or unseen concepts.

[Figure 297]

[Figure 298]

[Figure 299]

62.78 70.87 59.27

Retrieved Image MVDream Ours

Figure 8: Limitations of CLIP textimage similarity for evaluating OOD objects. Each row shows an example from our OOD-Eval benchmark: the ground truth (GT) image, the output from MVDream, and the output from our model. Below each image is its CLIP similarity score. MVDream receives a higher score than both our model and the GT image, despite producing less faithful generations.

These observations underscore the limitations of using CLIP for OOD evaluation and motivate our decision to adopt image-image similarity metrics instead, which more reliably reflect visual fidelity. To this end, we employ CLIP [39], DINOv2 [35], and an Instance Retrieval (IR) model [50] fine-tuned from CLIP to better align visual object instances.

IND-Eval OOD-Eval

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

|[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]| | |[Figure 308]<br><br>[Figure 309]| |
|---|---|---|---|---|

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

|[Figure 324]<br><br>[Figure 325]<br><br>[Figure 326]|[Figure 327]| | |[Figure 328]<br><br>[Figure 329]|
|---|---|---|---|---|

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

| |[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>[Figure 342]|[Figure 343]<br><br>[Figure 344]| | |
|---|---|---|---|---|

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

|[Figure 354]<br><br>[Figure 355]<br><br>[Figure 356]| | | |[Figure 357]<br><br>[Figure 358]<br><br>[Figure 359]|
|---|---|---|---|---|

- Figure 10: Evaluation Benchmarks Overview. OOD-Eval: Our out-of-distribution benchmark includes 2D images of both rare and well-known objects, featuring a diverse set of categories such as animals, vehicles, insects, foods, and everyday items. IND-Eval: The in-domain benchmark focuses on common, everyday objects that are representative of standard training distributions.

#### C.5 Re-rendering

To more thoroughly assess the 3D consistency and fidelity of the baselines on the OOD-Eval benchmark, we employ LGM [57]. Specifically, we reconstruct a 3D Gaussian representation using four output views generated by each model. From this reconstruction, we render 18 additional novel views sampled along a circular trajectory. These re-rendered views are then evaluated against the retrieved images using the image-image similarity metrics described earlier. We utilize the publicly available LGM implementation with its default configuration settings.

#### C.6 Retrieval Process

We evaluate multiple retrieval strategies based on both image-text and text-text similarity. For embedding-based retrieval with CLIP [39]and SigLIP [65], we build an index using the FAISS library [9], which supports efficient approximate nearest-neighbor search in high-dimensional spaces. We additionally employ Pyserini [59, 43, 27] for text-based retrieval using a Bag-ofWords approach powered by the BM25 ranking function. This approach is a highly optimized and scalable toolkit designed for large-scale retrieval tasks, capable of indexing millions of documents while providing fast query responses. Its retrieval time is typically sub-linear with respect to the size of the corpus due to inverted index structures, enabling near real-time search performance with minimal computational overhead, as demonstrated in large-scale search engine systems.

Table 5: CLIP similarity between retrieved images and concept labels. We compute similarity to the GT retrieved images with the object name against the class label (e.g., "dog", "car") and report average, max, and their absolute difference. OOD examples show minimal semantic separation.

Domain Text Avg Max

Bucovina Shepherd Dog 63.59 67.16 Dog 63.41 67.45

OOD

- Abs. Diff 0.18 0.29 BMW 319 automobile car 66.33 71.11

Car 65.14 71.56

- Abs. Diff 1.19 0.45

Airedale Terrier dog 89.54 96.58 Dog 64.23 69.99 Abs. Diff 25.31 26.59

In-Domain

American Hairless Terrier dog 83.08 92.79 Dog 64.59 70.11 Abs. Diff 18.49 22.68

### D User Study

We provide additional details about the user study referenced in Tab.2. The study involved 8 different objects, each evaluated using 3 methods: MV-RAG, MVDream, and ImageDream. For each object, participants were first shown a brief

text description along with two sample images of the object to establish context. They were then shown sets of four images corresponding to different views-generated by each method. The internal order of the methods was randomized per object to mitigate ordering bias. Participants were asked to rate the following three questions on a scale from 1 to 5: (Q1) “How well do the 4 images match object?“, (Q2) “How realistic do the 4 images look overall?“, and (Q3) “How well do the 4 images appear to be consistent with each other, as if they show different views of the same 3D object?“. The study was conducted using Google Forms, and participants viewed the images on a computer screen. The user population consisted of 30 randomly selected individuals across diverse ages, ethnicities, and genders.

### E Evaluation Dataset Construction

Construction of OOD-Eval. We construct an evaluation benchmark, OOD-Eval, consisting of 196 objects. To ensure diversity and out-of-distribution coverage, we use a large language model (GPT-4o) to curate object names representing rare or unique concepts, as well as familiar objects that are absent from the training data. These include examples such as extinct or rare animal species, uncommon vehicles, and other atypical items. A visual preview of the benchmark

|is𝛼provided=0.06|
|---|

in Fig. 10.

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 371]

[Figure 372]

[Figure 373]

For generating image captions, we leverage a vision-language model, specifically Qwen-VL [1] which provides high-quality textual descriptions of the images. These captions are used in the retrieval process (see Sec 3.5).

“Banana passionfruit, passiflora supersect”

|𝛼 = 0.59|
|---|

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

Construction of IND-Eval. We constructed an in-domain evaluation set by selecting 50 wellknown or everyday objects from the widely used Objaverse-XL dataset. For each object, we retrieve 4 reference images from the large-scale LAION-400M dataset [47] using BM25-based text retrieval (see Sec.3 in main paper). The retrieved images often exhibit significant visual or

|modalityAdaptive𝛼variation(ours)|
|---|

|𝛼 = 0.3|
|---|

|𝛼 = 1.0|
|---|

“Bearded Collie dog”

m (e.g., artistic renderings or paintings of the object), as illustrated in Fig. 10

Base Model Output

### F Limitations

While effective, our method has several limitations. It relies heavily on both the quality of the retrieved image corpus and the capabilities of the underlying generative model, MVDream. When the base model lacks prior knowledge of the object and retrieval fails to provide informative or diverse references, the generated multiviews can be inaccurate or implausible.

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

- (a)
- (b)
- (c)

| |
|---|

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

As shown in Fig. 11, errors may arise when the retrieved images are visually biased-e.g., all showing similar white flowers, leading to reduced diversity and visual artifacts. Furthermore, our training objective promotes texture variation, which can make it difficult to reproduce fine-grained or specific patterns, such as the hoverfly’s dorsal markings.

| |
|---|

Retrieved Images

[Figure 396]

[Figure 397]

| |
|---|

| |
|---|

Our model also employs an adaptive mechanism that balances attention between the base model and the retrieval adapters, based on a similarity score between the generated initial views and retrieved images. When the base model demonstrates high similarity to the target object but exhibits 3D structural errors (such as a floating dog tail), these artifacts may be inherited. This limitation could be addressed by incorporating a more sophisticated and 3D aware scoring function.

Base Model Output

Ours

Figure 11: Limitations. (a) Visually biased retrieved images (e.g., repetitive white flowers) introduce artifacts in the generated multiviews. (b) The model struggles to reproduce fine-grained textures, such as the hoverfly’s dorsal pattern. (c) When the base model (MVDream) is assigned a high attention weight (α), 3D structural inaccuracies from the base model (e.g., a floating tail) are inherited.

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Lastly, our method introduces a retrieval phase prior to generation. Although this adds computational cost

19

Retrieved Images

Base model Output

relative to standard text-to-image pipelines, the overhead is minimal. As shown in Section C.6

### G Broader Impact

Our model advances multiview image generation by conditioning on text prompts and retrieved image corpora, enabling applications in graphics, virtual reality, and content creation. Positively, this technology can facilitate more immersive and accurate 3D visualizations and assist artists or designers in generating diverse object views with limited data. However, improved generative capabilities also pose risks, such as enabling the creation of highly realistic synthetic images that could be misused for disinformation or deepfakes. Additionally, biases present in the retrieval corpus or base models may propagate or amplify undesirable stereotypes or inaccuracies in generated outputs. To mitigate such risks, future deployment should include careful curation of training and retrieval datasets, transparency about generated content, and potentially gating access to the most advanced models.

[Figure 408]

[Figure 409]

|[Figure 410]|[Figure 411]<br><br>[Figure 412]<br><br>[Figure 413]|
|---|---|

| |[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]|
|---|---|

AJS 500 cc 1954 E95 Porcupine racer motorcycle American Bully dog

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

|[Figure 423]|[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]|
|---|---|

|[Figure 427]<br><br>[Figure 428]<br><br>[Figure 429]|[Figure 430]|
|---|---|

American Eskimo Dog Appenzeller Sennenhund dog

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

|[Figure 436]|[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]|
|---|---|

|[Figure 440]|[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]|
|---|---|

Arctophila Superbiens bee Australian Silky Terrier dog

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

| |[Figure 452]<br><br>[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]| |
|---|---|---|

|[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]| |
|---|---|

Chempedak fruit Bentley 4½ Litre supercharged, 1929

[Figure 460]

[Figure 461]

|[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]| |[Figure 465]|
|---|---|---|

|[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]| |
|---|---|

Bergamasco Shepherd dog Bloodhound dog

| |[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]|
|---|---|

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

| |[Figure 479]|[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]|
|---|---|---|

Bracco Italiano dog Briard dog

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

| |[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]|
|---|---|

|[Figure 492]|[Figure 493]<br><br>[Figure 494]<br><br>[Figure 495]|
|---|---|

Cadillac 341 automobile car Chinese Crested Dog

[Figure 496]

[Figure 497]

|[Figure 498]|[Figure 499]<br><br>[Figure 500]<br><br>[Figure 501]|
|---|---|

|[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]|[Figure 505]| |
|---|---|---|

Finnish Spitz dog Glen of Imaal Terrier dog

[Figure 506]

[Figure 507]

| |[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]|
|---|---|

|[Figure 512]<br><br>[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]| |
|---|---|

Goggomobil Dart microcar Havanese dog

#### Figure 12: Additional Results.

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

|[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]| | |
|---|---|---|

| |[Figure 525]|[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]|
|---|---|---|

Hokkaido inu dog Leica M3 camera

[Figure 529]

| |[Figure 530]<br><br>[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]| |
|---|---|---|

[Figure 534]

|[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]| | |
|---|---|---|

Lhasa Apso dog Male Crimson Sunbird

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

|[Figure 547]<br><br>[Figure 548]<br><br>[Figure 549]|[Figure 550]|
|---|---|

|[Figure 551]|[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]|
|---|---|

Russkiy Toy dog

Mediterranean jellyfish

[Figure 555]

[Figure 556]

| | |[Figure 557]<br><br>[Figure 558]<br><br>[Figure 559]<br><br>[Figure 560]|
|---|---|---|

|[Figure 561]<br><br>[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]| | |
|---|---|---|

Maserati 250F racing automobile

Mitu mitu, Alagoas curassow

[Figure 565]

[Figure 566]

|[Figure 567]<br><br>[Figure 568]<br><br>[Figure 569]| |[Figure 570]|
|---|---|---|

| | |[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]<br><br>[Figure 574]|
|---|---|---|

Neapolitan Mastiff dog

Chow Chow dog

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

|[Figure 580]| |[Figure 581]<br><br>[Figure 582]<br><br>[Figure 583]|
|---|---|---|

| |[Figure 584]<br><br>[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]| |
|---|---|---|

Silkie chicken Sugar apple fruit

[Figure 588]

|[Figure 589]<br><br>[Figure 590]<br><br>[Figure 591]<br><br>[Figure 592]| | |
|---|---|---|

[Figure 593]

| |[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]<br><br>[Figure 597]| |
|---|---|---|

Breguet 941 transport aircraft Clathrus ruber fungus

[Figure 598]

[Figure 599]

|[Figure 600]<br><br>[Figure 601]<br><br>[Figure 602]| |[Figure 603]|
|---|---|---|

|[Figure 604]<br><br>[Figure 605]<br><br>[Figure 606]|[Figure 607]|
|---|---|

Cockapoo dog Coton de Tulear dog

|[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]|[Figure 611]|
|---|---|

[Figure 612]

[Figure 613]

| |[Figure 614]<br><br>[Figure 615]<br><br>[Figure 616]<br><br>[Figure 617]| |
|---|---|---|

Dogue de Bordeaux English Cocker Spaniel dog

#### Figure 13: Additional Results.

[Figure 618]

[Figure 619]

MVDreamMV-Adapter(TX)Era3DSPADMV-Adapter(IM)ImageDreamOurs

[Figure 620]

[Figure 621]

Images Text-to-3DImage-to-3DRAG-to-3D

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

TRELLIS(TX)TRELLIS(IM)

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

Personalization

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

MVDreambooth

[Figure 650]

[Figure 651]

| |[Figure 652]<br><br>[Figure 653]<br><br>[Figure 654]|
|---|---|

|[Figure 655]<br><br>[Figure 656]|
|---|

|[Figure 657]<br><br>[Figure 658]|
|---|

|[Figure 659]<br><br>[Figure 660]|
|---|

|[Figure 661]<br><br>[Figure 662]|
|---|

|[Figure 663]<br><br>[Figure 664]|
|---|

|[Figure 665]<br><br>[Figure 666]|
|---|

[Figure 667]

Reference

"Philippine eagle, Pithecophaga jefferyi" “London Duck Tours car”

Figure 14: Additional qualitative evaluation. Additional examples to those shown in Fig. 7.

[Figure 668]

[Figure 669]

MVDreamMV-Adapter(TX)SPADImageDreamMV-Adapter(IM)Era3DOurs

[Figure 670]

[Figure 671]

Images Text-to-3DImage-to-3DRAG-to-3D

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

TRELLIS(IM)TRELLIS(TX)

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

Personalization

MVDreambooth

[Figure 701]

[Figure 702]

| |[Figure 703]<br><br>[Figure 704]<br><br>[Figure 705]|[Figure 706]<br><br>[Figure 707]<br><br>[Figure 708]<br><br>[Figure 709]<br><br>[Figure 710]| |
|---|---|---|---|

| |[Figure 711]<br><br>[Figure 712]<br><br>[Figure 713]<br><br>[Figure 714]<br><br>[Figure 715]<br><br>[Figure 716]| |[Figure 717]<br><br>[Figure 718]|
|---|---|---|---|

Reference

"Le Percheron tractor"

"kākāpō, Strigops habroptilus"

Figure 15: Additional qualitative evaluation. Additional examples to those shown in Fig. 7.

