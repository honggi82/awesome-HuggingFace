# arXiv:2507.13984v2[cs.CV]15Mar2026

## CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models

Quang-Binh Nguyen1 Minh Luu2 Quang Nguyen1 Anh Tran1 Khoi Nguyen1 1Qualcomm AI Research† 2MovianAI

{binhnq, quanghon, anhtra, khoi}@qti.qualcomm.com, v.minhlnh@vinai.io

### Abstract

Disentangling content and style from a single image, known as content-style decomposition (CSD), enables recontextualization of extracted content and stylization of extracted styles, offering greater creative flexibility in visual synthesis. While recent personalization methods have explored the decomposition of explicit content style, they remain tailored for diffusion models. Meanwhile, Visual Autoregressive Modeling (VAR) has emerged as a promising alternative with a next-scale prediction paradigm, achieving performance comparable to that of diffusion models. In this paper, we explore VAR as a generative framework for CSD, leveraging its scale-wise generation process for improved disentanglement. To this end, we propose CSD-VAR, a novel method that introduces three key innovations: (1) a scaleaware alternating optimization strategy that aligns content and style representation with their respective scales to enhance separation, (2) an SVD-based rectification method to mitigate content leakage into style representations, and (3) an Augmented Key-Value (K-V) memory enhancing content identity preservation. To benchmark this task, we introduce CSD-100, a dataset specifically designed for content-style decomposition, featuring diverse subjects rendered in various artistic styles. Experiments demonstrate that CSD-VAR outperforms prior approaches, achieving superior content preservation and stylization fidelity.

### 1. Introduction

The challenge of disentangling content and style, or content-style decomposition (CSD), from a single image can be framed as a dual personalization problem: one that extracts the subject’s structure and details (content) while separately capturing the artistic technique (style). Successfully addressing this problem enables two key applications: recontextualization, where a subject is adapted to different visual environments, and stylization, where an extracted

† Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.

[Figure 1]

[Figure 2]

Input

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

...above a river ...above jungle

A lighthouse in ...A cabybara in ...

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

...in anime style ...stand on a rock

A deer in ... A car in ...

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

...made of glass ...look out

A dragon in ... A bunny in ...

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

...in swimming pool

...in jungle

A lighthouse in ... A car in ...

Figure 1. Given a single input image, our framework separates content and style, enabling flexible recontextualization and stylization to generate new images across diverse contexts.

style is applied to new subjects. For instance, consider an artist working from a single reference – a balloon illustration rendered in a unique artistic style, illustrated in Fig. 1 (first row). Separating its content and style would allow the artist to depict the same balloon in a realistic setting or apply the illustration’s stylistic elements to new subjects, such as landscapes or portraits. This separation fosters greater creative flexibility, enabling novel compositions and crossdomain visual transformations.

In text-to-image personalization, textual inversion [9] optimizes text embeddings to capture specific concepts within the text encoder space while keeping model parameters frozen. Many subsequent methods [7, 15, 19, 20, 31, 50, 56, 58] have extended textual inversion to better preserve object identity and enhance performance across various tasks beyond encoding only a subject’s appearance. However, these approaches do not explicitly separate content and style. More recent methods, such as B-LoRA

[8] and UnZipLoRA [26], address this limitation by decomposing an image into distinct content and style representations. Nevertheless, these techniques are specifically designed for diffusion-based models, and no prior work has explored their application to other generative models, such as Autoregressive (AR) models. Recently, AR models [3, 14, 24, 29, 43–45, 47, 51, 53, 55] have emerged as a compelling alternative to diffusion models, offering comparable generative capabilities while maintaining high efficiency. A notable variant, Visual Autoregressive Modeling (VAR) [14, 29, 44, 47, 51], introduces a next-scale prediction paradigm, where generation starts with a 1×1 token map and progressively expands into a sequence of multiscale token maps (e.g., 2 × 2, ...), increasing in resolution. This motivates our investigation into how content and style representations can be effectively learned within VAR.

However, directly applying textual inversion to VAR by optimizing separate style and content embeddings results in suboptimal representations due to the strong entanglement between these attributes. The inherent coupling of content and style makes simple text prompt guidance insufficient for effective decomposition. First, we empirically observe that early scales primarily encode style, while later scales capture content information. Building on this insight, we propose a scale-aware alternating optimization strategy, where content and style embeddings are optimized at their corresponding scales in an alternating manner to ensure better disentanglement. Second, to further strengthen content-style separation, we introduce a singular value decomposition (SVD)-based rectification method, which explicitly enforces independence between the content and style spaces, reducing mutual interference and enhancing the quality of learned representations. Third, relying solely on textual embeddings proves inadequate for capturing complex concepts and styles, particularly in intricate cases. To address this limitation, we introduce augmented Key-Value (K-V) memory, which serves as auxiliary storage for content and style attributes that textual inversion alone fails to capture. These augmented K-V matrices not only improve content-style disentanglement but also enhance identity preservation, ensuring more faithful and expressive representations.

To our knowledge, no publicly available dataset with quantifiable benchmarks exists for content-style decomposition (CSD). While existing datasets focus on either style transfer or content preservation, they do not fully meet the requirements for evaluating CSD, prompting us to introduce CSD-100, a dataset of 100 images designed specifically for this task. On CSD-100, our CSD-VAR outperforms stateof-the-art by a large margin in all common metrics. In summary, our main contribution includes:

• We are the first to explore VAR-based personalization for content-style decomposition.

- • We analyze scale-dependent representations in VAR and propose a scale-aligned optimization strategy to improve content-style disentanglement.
- • We introduce an SVD-based constraint to enforce orthogonality between content and style representations.
- • We propose augmented K-V memory to enhance disentanglement and better preserve subject identity.
- • We propose a new dataset, CSD-100, specifically designed for the CSD with various styles and subjects.

### 2. Related Work

Text-to-image generative models. Early text-to-image generation methods built on GANs [12], conditioning on textual inputs to map embeddings into image space, but suffered from training instability and mode collapse. The introduction of diffusion models (DMs) [35, 36, 39] improved image quality and diversity by leveraging large-scale image-text datasets and iterative denoising. However, DMs are computationally expensive due to slow sequential inference. To mitigate this, recent works [28, 30, 40] distill multi-step teacher models into efficient few-step student networks, with some methods [5, 27, 32, 57] achieving onestep text-to-image generation.

Autoregressive (AR) models offer an alternative to diffusion-based approaches. Earlier AR models generated images sequentially via next-token prediction [3, 43, 45, 53], but their step-by-step nature made high-resolution generation slow. Recently, next-scale prediction AR has emerged as a more efficient paradigm, generating images progressively from low to high resolution [14, 29, 44, 47,

- 51], refining details at each scale while attending to previous ones. This approach has been successfully applied to text-to-image generation: STAR [29] and Switti [51] use cross-attention for text conditioning, HART [44] improves VAR [47] with a diffusion model for continuous error residuals, and Infinity [14] enables fine-grained generation with bitwise token prediction. While our method is compatible with any next-scale AR model, we use Switti [51] and Infinity [14] as our backbones due to their available training and inference code.

Image personalization. Personalization methods adapt a pretrained model to integrate a specific concept from a given set of images, typically by optimizing specialized text embeddings [9] or fine-tuning model weights [38]. While fine-tuning improves reconstruction fidelity, it requires substantial memory and often leads to overfitting. To mitigate these issues, Parameter-Efficient Fine-Tuning (PEFT) techniques [13, 17, 18, 22] enhance efficiency while reducing memory overhead. Textual inversion methods have shown strong generalization across various tasks [10, 19, 20, 50,

- 52, 58], extending beyond appearance capture due to the flexibility of text embedding manipulation. PEZ [52] op-

timizes hard text prompts to capture diverse targets, while Reversion [19] introduces contrastive learning to steer embeddings into relational space. TokenVerse [10] further explores modulation space in DiT-based models, demonstrating that fine-tuned offset text embeddings can encode various concepts, including objects, accessories, materials, and lighting. However, these methods are designed for diffusion-based models, and directly applying them to VAR models does not yield effective results. In this work, we focus on textual inversion for content-style decomposition within the VAR framework, introducing techniques that explicitly leverage VAR’s multi-scale generation process for improved disentanglement.

Content-Style decomposition. The task of concept extraction has been actively researched for years [1, 20, 25, 50, 59], with recent advances extending beyond simple subject representation. More recently, UnzipLoRA [26] introduced content-style decomposition, emphasizing the simultaneous optimization of content and style representations. This ensures that extracted content remains adaptable across diverse contexts while preserving fidelity, and the style representation generalizes across subjects without content leakage. As a fine-tuning approach, UnzipLoRA achieves this by jointly learning two LoRA modules for content and style, employing prompt separation and block-wise sensitivity analysis. B-LoRA [8] fine-tunes only two sets of sensitive layers for implicit content-style decomposition. UVAP [54] enables fine-grained personalization by learning local subject attributes but is unsuitable for global characteristics like style and relies on LLM-guided data augmentation, limiting scalability. In textual inversion, Inspiration Tree [50] learns hierarchical subconcepts, though disentanglement remains unpredictable. Lego [31] extends textual inversion beyond single-concept appearance, while ConceptExpress [15] enables unsupervised concept extraction but is restricted to spatially localized attribute masks, making it ineffective for capturing global style.

While previous methods focus primarily on fine-tuning diffusion models, our approach explores content-style decomposition within an autoregressive framework, leveraging the multi-scale generation process of VAR models.

### 3. Preliminaries

Visual Autoregressive Modeling (VAR). Unlike conventional next-token prediction autoregressive methods, VAR [47] introduces a novel paradigm in visual autoregressive modeling, shifting the focus from next-token prediction to next-scale prediction. Instead of generating tokens sequentially, VAR predicts an entire token map at each scale based on previously predicted scales. The process starts with a 1 × 1 token map, denoted as s1, and progressively generates a sequence of multi-scale token maps

(s1,s2,...,sK), increasing in resolution up to K × K.

VAR θ, visualized in Fig. 2 (excluding the yellow box ), models the following joint distribution to generate the token maps of different scales sequentially:

p(s1,s2,...,sK) =

K

pθ(sk|s1,s2,...,sk−1;c), (1)

k=1

k×wk is the token map at scale k, with size hk × wk, conditioned on the previously generated scales (s1,s2,...,sk−1). Each token in sk is an index from the VQ-VAE codebook with V vocabularies, trained through multi-scale quantization and shared across scales. In practice, to maintain a similar number of input and output tokens at the current scale, as required by the autoregressive transformer, the output token at previous scale is interpolated to yield fk∗, before being fed back into VAR as input for the next scale k, or:

where c is the condition such as object class, sk ∈ [V ]h

K

pθ(sk|f1,f2,...,fk;c), (2)

p(s1,s2,...,sK) =

k=1

where fk = Interpolate(sk−1,k). (3)

To generate the final output image (Fig. 2, Right), we interpolate all predicted token maps across scales sk to the final scale K, sum them together, and then pass the result through a decoder D, as follows:

K

Interpolate(sk,K); I = D(sK). (4)

sK =

k=1

Teacher forcing training (Fig. 2, Left) is a widely used technique to accelerate the training of AR models by enabling parallel training across different scales. Instead of sequentially generating each scale and using its output as a condition for the next, this approach feeds ground-truth scales directly as conditions. This allows the model to learn more efficiently without waiting for the completion of previous scale predictions. In particular,

p(s1,s2,...,sK) =

K

pθ(sk|f1∗,f2∗,...,fk∗;c), (5)

k=1

where {fk∗}Kk=1 are the interpolated ground-truth scales, obtained by encoding the training image I∗ with the image

encoder E, e.g., VQ-VAE [6], or:

{s∗k}Kk=1 = E(I∗); fk∗ = Interpolate(s∗k−1,k). (6)

The training loss is the cross-entropy loss between predicted and ground-truth tokens at all scales.

K

K

−s∗k log pθ(sk|f1∗,f2∗,...,fk∗;c). (7)

L =

Lk =

k=1

k=1

Inference with Content

Optimization Phase

[Figure 23]

|Our Improvement over the baseline|
|---|

...

###### Predicted ... Tokens

...

Interpolate

Interpolate

...

[Figure 24]

"A photo of a < > object in < > style"

... ...

[Figure 25]

...

Auto-Regressive Transformer

Output image

|Text encoder<br><br>[Figure 26]| |
|---|---|
| | |

...

Cross-attention

[Figure 27]

[Figure 28]

Auto-Regressive Transformer

"A photo of a < > with auturmn leaves"

Text embedding

Augmented K-V for content

Augmented K-V for style

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

Inference with Style

... ...

[Figure 33]

Pooling

...

...

... ...

Rectify style embedding

Interpolate

Interpolate

...

Interpolate

Output image

Auto-Regressive Transformer

[Figure 34]

...

...

|VQ-VAE<br><br>[Figure 35]| |
|---|---|
| | |

...

...

... ...

"A photo of a bunny in < > style"

...

Ground-truth Tokens

Input image

- Figure 2. Overview of our CSD-VAR. During optimization (left), a content-style prompt y, "A photo of a <yc> object in <ys> style", is encoded into text embeddings e. The rectified style embedding es reduces content leakage, while ground-truth scalewise tokens from VQ-VAE are interpolated for next-scale prediction. Augmented K-V memories are prepended at specific scales before feeding into the autoregressive transformer. The model is trained with scale-wise cross-entropy losses, alternating optimization of content and style embeddings. At inference (right), style or content K-V memories are prepended based on the prompt before predicting tokens.

1 2 3 4 5 6 7 8 9 10

Removing Scale

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Score

DINO

CSD-S

- Figure 3. Analysis of style-related scores across different scales.

### 4. Our Approach

Problem statement: Given an input image I∗ containing a subject yc in style ys, our objective is to disentangle its style and content into two distinct representations, enabling the generation of separate images: Ic, which accurately preserves the content of I∗, and Is, which effectively captures its style. To this end, we explore the use of Visual Autoregressive Models (VAR) and leverage the pretrained text-toimage model, i.e., Switti [51] and Infinity [14], for this task.

Our baseline: A feasible approach is Textual Inversion [9], which optimizes the textual embeddings for content, yc, and style, ys, before the text encoder, without modifying Switti and Infinity’s pretrained weights – thus preserving its generative capabilities. In particular, the prompt used for textual inversion is: "A photo of a <yc> object in <ys> style", as shown in Fig. 2. However, this simple baseline produces subpar results, as demonstrated in Tab. 2 and Fig. 8 . In the following section, we introduce three key improvements to enhance the effectiveness of Textual Inversion for text-to-image VAR.

Text-to-image VAR extends the VAR framework for textto-image synthesis. Specifically, we adopt Switti [51] and Infinity [14] as representative backbones for developing our approach due to their publicly available training and inference code. Given a text prompt y = {yi}Ni=1 where N is the number of words in the text prompt, we first encode it using both OpenCLIP ViT-bigG [4] and CLIP ViT-L [34]. The resulting embeddings e = {ei}Ni=1 are then pooled to form the start token f1 = N1 Ni=1 ei to predict the first scale s1. In each transformer block—comprising self-attention, crossattention, and a feed-forward network (FFN)—the text embedding c is incorporated into the cross-attention layers via the standard attention mechanism [49].

#### 4.1. Scale-aware Alternating Optimization Strategy

Examining the detail captured at each scale in VAR reveals correlations with content and style information. Inspired by this, we analyze how different scales contribute to style attributes (e.g., color, texture) and content structure (e.g.,

1) Construct content-projection matrix

that refines the original style embedding es = CLIP(ys) by removing content-related information during training.

"A Golden Retriever" "A German Shepherd" "A French Bulldog"

"Describe variations of concept: dog"

First, we construct a content-related subspace by using a large language model (LLM), such as Llama [48] or ChatGPT [46], to generate variations or subconcepts of the target concept. For instance, the concept “dog” may have subconcepts like “Golden Retriever,” “German Shepherd,” and “Bulldog,” which we embed using a CLIP text encoder.

LLM

###### CLIP-T

... "A Alaskan Malamute"

...in style dog...

...in style

bedroom...

dog... bedroom...

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

optimized

"dog" space

[Figure 40]

dragon...

robot...

dragon...

robot...

Next, we extract the most relevant directions within the content subspace by applying singular value decomposition (SVD) to the matrix M ∈ RQ×d, where the columns are the text embeddings of the Q sub-concepts. The SVD decomposition is: M = UΣV T, where Σ is a diagonal matrix of singular values, and U,V contain singular vectors representing dominant directions in the content embedding space. We then select the top r singular vectors (corresponding to the largest singular values) to construct the projection matrix:

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Input image

2) Rectifying style embedding

Figure 4. Style embedding rectification and examples

object shape, category, and fine-grained details).

Using a set of 35 images with variations in content and style, we extract token maps across all scales, remove a specific scale’s tokens, and reconstruct the image. We then compute DINO [2] and CSD-S [42] scores to measure style similarity between the original image I∗ and the reconstructed image I, as these metrics are widely adopted for accessing style similarity [16, 37]. As shown in Fig. 3, removing smaller-scale token maps (k = {1,2,3}) and the final scale (k = 10) significantly affects style. Based on these findings and the assumed disentanglement of content and style, we categorize scales into two groups: the stylerelated group (Sstyle = {1,2,3,10}) and the content-related group (Scontent = {4,...,9}). The losses for optimizing the style embedding ys and content embedding yc are:

Pproj = Vr⊤Vr, (10)

where Vr ∈ Rr×d is the truncated version of V up to rank r and Pproj remains fixed during training for a given concept.

Finally, we remove content-related information from the style embedding es by projecting it onto Pproj and subtracting the projected content as follows:

e′s = es − e⊤s Pproj. (11)

This ensures that e′s remains orthogonal to content-related variations, effectively preventing unintended subject leakage in the generated images. As shown in Fig. 4, rectifying e′s significantly reduces content leakage.

###### Lk′ (8)

Lstyle =

Lk + α

#### 4.3. Augmented Key-Value (K-V) Memories

k′∈Scontent

k∈Sstyle

With the two improvements above, content and style embeddings can be effectively separated. However, for complex content or style concepts, textual embeddings alone may be insufficient, leading to under-captured representations. To address, we introduce augmented K-V memory to enhance the information captured by textual embeddings.

###### Lk, (9)

Lcontent =

k∈Scontent

where each Lk represents the cross-entropy loss at scale k, as defined in Eq. (7), and α ∈ (0,1) controls the influence of larger-scale token maps on style, ensuring that certain style attributes not fully captured by lower scales are retained. To further improve content-style disentanglement, we introduce an alternating optimization strategy, where content and style embeddings are optimized in separate iterations. This prevents gradient mixing, ensuring a clearer separation between the two representations.

Specifically, we augment several blocks in the autoregressive transformer with O pairs of K-V memories, inserted before the self-attention layers. These K-V memories, denoted as K˜ and V˜, are added just before the K,V matrices at the first scale for style (k = 1) and the fourth scale for content (k = 4), respectively. Formally:

#### 4.2. SVD-based Style Embedding Rectification

K ˜ K

V ˜ V

Attn(Q,K,V ;K,˜ V˜) = Attn Q,

. (12)

,

While the style textual embedding ys is designed to capture texture details, its ability to attend to content scales (with a small α in Eq. (8)) allows residual content information in smaller-scale token maps, leading to content leakage into the style embedding. This contamination results in an imperfect style representation, as shown in Fig. 4. To mitigate this, we propose an SVD-based rectification method

### 5. Our CSD-100 Dataset

To our knowledge, no publicly available dataset provides quantifiable benchmarks for content-style decomposition (CSD). While existing datasets [21, 33, 38, 41] contain elements related to style transfer or content preservation, they

Content Style

[Figure 45]

3D Minimalism

[Figure 46]

Food Musical Instrument Toys

Digital

Art movement

Children drawing

Hardware tools

Transportation

Decorative pattern

Art illustration

Art tools

Glow

Misc

Texture/Material/Print

Animal

Painting Sketch/Drawing

(a) Distribution of content and style categories in CSD-100.

|[Figure 47]|
|---|

(b) Samples from our CSD-100 dataset.

Figure 5. Statistics and samples of the CSD-100 dataset.

do not fully meet the requirements for CSD evaluation. StyleDrop [41] offers diverse style images but is relatively small and lacks sufficient instances where style is applied to salient objects. Similarly, personalization datasets such as DreamBooth [38], and CustomConcept101 [21] focus on content realism rather than explicit content-style representation. Whereas StyleAligned [16] and RB-Modulation [37] provide only benchmark prompts, limiting their usability for standardized evaluation.

To address this gap, we introduce CSD-100, a dataset of approximately 100 images designed to capture diverse content and styles for CSD evaluation. As shown in Fig. 5, our CSD-100 spans a wide distribution of content and style, supporting robust evaluation. This section outlines the pipeline for generating and curating CSD-100.

Data Collection and Curation Process. We begin with the RB-Modulation [37] prompt collection, containing approximately 400 content and 100 style concepts. To refine content categories, we remove overlapping terms and those related to landscapes, houses, humans, swings, and wheels, as they complicate content-style distinction. This filtering results in 180 content concepts while retain-

ing all 100 style concepts. Next, we generate contentstyle image pairs using Flux 1.0 [23], a text-to-image model, with the prompt "A photo of <content> in <style>.", producing around 18,000 images. To refine this, we develop a custom application for manual review, selecting the 10 most representative images per style while ensuring consistency by avoiding multiple objects in a single image. This reduces the dataset to 1,000 images.

For further filtering, we use ChatGPT [46] with a structured prompt to distill the dataset down to 100 high-quality images that best capture diverse content-style combinations. Illustrative examples are shown in Fig. 5. These manual steps require significant human effort, and in the future, we plan to scale the dataset with more images, content concepts, and style categories.

Evaluation Protocol. We curate 50 inference prompts (25 for content, 25 for style) and generate 10 images per prompt for each CSD-100 concept. This results in an extensive evaluation set of 100×50×10 = 50,000 images, ensuring robust assessment. Details of inference prompts and dataset analysis are provided in the supplementary material.

### 6. Experiments

#### 6.1. Experimental Setup

Datasets: To evaluate our method, we use the proposed CSD-100 dataset described in Sec. 5. Additionally, we construct a smaller validation set of 35 images for our ablation study. These images are curated from StyleDrop [41], BLoRA [8], UnZipLoRA [26], and DreamBooth dataset [38], following the same evaluation protocol as CSD-100.

Metrics: We use the following metrics: CSD-C [42] and CLIP-I [34] for content alignment, CSD-S [42] and DINO [2] for style alignment, and CLIP-T [34] for text alignment. For all metrics, higher values indicate better performance.

Implementation details: We build our approach on Switti [51] and Infinty [14] and optimize it using the Adam optimizer with a learning rate of 10−3 for 200 steps and a batch size of 1. The style loss coefficient is empirically set to α = 0.1 across all experiments. K-V memories are randomly initialized using the Xavier uniform initialization scheme [11]. Training is conducted on a single A100 GPU. For generation variations in Sec. 4.2, we generate 200 subconcepts per concept.

#### 6.2. Comparison with Prior Approaches

Qualitative comparison. To assess the effectiveness of CSD, we compare our method against DreamBooth [38], B-LoRA [8], and Inspiration Tree [50]. For DreamBooth, we train two separate models for content and style using distinct prompts: "A photo of a yc" for content and

Ours (Inﬁnity) Ours (Switti) Ours (Inﬁnity)

DreamBooth B-LoRA Inspiration Tree Ours (Switti)

DreamBooth B-LoRA Inspiration Tree

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

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

< >

< >

in swimming pool

made ofgold

|[Figure 58]|
|---|

|[Figure 59]|
|---|

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

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

A cat Input in

A bunny Input in

< >

< >

Ours (Switti) Ours (Inﬁnity)

DreamBooth B-LoRA Inspiration Tree

DreamBooth B-LoRA Inspiration Tree

Ours (Switti) Ours (Inﬁnity)

|[Figure 70]|
|---|

|[Figure 71]|
|---|

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

|[Figure 78]|
|---|

|[Figure 79]|
|---|

< >

< >

|[Figure 80]|
|---|

in jungle

on beach

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]<br><br>Text|
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

|[Figure 90]|
|---|

|[Figure 91]|
|---|

A car Input in

A dragon Input in

< >

< >

B-LoRA

Inspiration Tree

Ours (Switti) Ours (Inﬁnity)

DreamBooth B-LoRA Inspiration Tree

Ours (Switti) Ours (Inﬁnity)

DreamBooth

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
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

[Figure 102]

< >

< >

in snow

|[Figure 103]|
|---|

on a table

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

A robot in

A castle in

Input

Input

< >

< >

Figure 6. Qualitative comparison with prior approaches on our CSD-100 dataset.

Methods Content Align Style Align Text Align CSD-C CLIP-I CSD-S DINO CLIP-T

DreamBooth-C [38] 0.594 0.721 – – 0.271 DreamBooth-S [38] – – 0.537 0.519 0.289 B-LoRA [8] 0.523 0.592 0.476 0.346 0.278 Inspiration Tree [50] 0.497 0.575 0.404 0.353 0.257

CSD-VAR (Switti) 0.603 0.754 0.564 0.521 0.332 CSD-VAR (Infinity) 0.660 0.795 0.552 0.536 0.319

###### Table 1. Quantitative comparison on our CSD-100 dataset.

100

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

###### UserPreference(%)

80

60

40

20

0

Image Quality

Prompt Adherence

Content Alignment

Style Alignment

Overall Quality

Our method

B-LoRA

Inspiration Tree

DreamBooth

| |
|---|

| |
|---|

| |
|---|

Figure 7. User preference study on key criteria for contentstyle decomposition with 100 participants.

"A subject in ys style" for style. While UnzipLoRA [26] is closely related to our work, it lacks a publicly available code, preventing direct comparison.

As shown in Fig. 6, our method excels at preserving content characteristics while realistically adapting them to new environments. In contrast, existing methods struggle with overfitting or content misalignment. For example, DreamBooth strongly overfits to style, often failing to follow the

Content Align Style Align Text Align SA SVD KV CSD-C CLIP-I CSD-S DINO CLIP-T ✓ ✓ ✓ 0.603 0.751 0.564 0.517 0.330 ✓ ✓ 0.581 0.702 0.559 0.509 0.315 ✓ ✓ 0.601 0.725 0.503 0.422 0.289

✓ ✓ 0.501 0.612 0.547 0.508 0.270

0.482 0.527 0.431 0.320 0.302 Table 2. Effectiveness of each component.

target prompt. B-LoRA and Inspiration Tree, while capable of transferring content, produce suboptimal results when recontextualizing content into different environments. In terms of style alignment, other methods exhibit content leakage, where style representations unintentionally retain content details. Notably, in the fox and bear examples (left, first and second row), other methods fail to fully separate content and style, causing unwanted subject appearances in stylized outputs. Our approach mitigates this issue, achieving faithful stylization without content artifacts.

Quantitative comparison. Tab. 1 presents a quantitative comparison of our method with existing approaches in the CSD-100 dataset. Our approach achieves the highest scores in both content alignment and style alignment across backbones, demonstrating superior content identity preservation and faithful stylization. While DreamBooth-C attains high content alignment and DreamBooth-S achieves strong style alignment, both exhibit lower text alignment scores, indicating overfitting to the input image. In contrast, our method maintains the highest text alignment score, reflecting better

Baseline W/o SA W/o SVD W/o KV Our (full)

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

A photo of < > looking out

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

Input

A photo of bunny in < >

Figure 8. Illustrative effectiveness of each component.

Content Align Style Align Text Align Blocks #Params CSD-C CLIP-I CSD-S DINO CLIP-T

First 7K 0.603 0.751 0.564 0.517 0.330 Half 115K 0.611 0.732 0.582 0.550 0.300 All 230K 0.621 0.733 0.580 0.541 0.292

Table 3. Study on # blocks to apply augmented K-V memory.

generalization in following textual descriptions.

User Study. In our user study, participants were shown an input image alongside the outputs of two competing methods. Each output group included two images for content recontextualization and two for style transfer. Participants evaluated and selected the superior output group based on five criteria: Image Quality, Prompt Adherence, Content Alignment, Style Alignment, and Overall Quality. The study gathered 7,500 responses from 100 participants.

As shown in Fig. 7, our method significantly outperforms others in content and style alignment. Additionally, it achieves the highest preference in prompt adherence, demonstrating its effectiveness in preserving both textual guidance and visual fidelity after content-style decomposition. Further details are provided in the Supp.

#### 6.3. Ablation Study

Effectiveness of each component is presented in Tab. 2 and Fig. 8. Removing the scale-aware strategy severely impacts the model’s ability to separate content from style, leading to the lowest content-alignment and text-alignment scores. Without rectification, style alignment degrades due to content leakage in the generated images. Additionally, eliminating augmented K-V memories weakens the model’s ability to capture both content and style, resulting in lower scores across multiple metrics.

Study on the number of blocks to apply augmented K-V memories. We evaluate three settings: applying K-V only to the first block, to the first half of all blocks (15 blocks), and across all 30 transformer blocks. As shown in Tab. 3, increasing the number of blocks improves content and style alignment but slightly reduces text alignment, likely due to overfitting, while providing only marginal gains relative to the added computational cost. Given this trade-off, we

Content Align Style Align Text Align Setting CSD-C CLIP-I CSD-S DINO CLIP-T

Top 3 0.581 0.726 0.561 0.524 0.289 Top 5 0.598 0.724 0.563 0.528 0.317 Top 10 0.603 0.751 0.564 0.517 0.330 Top 20 0.605 0.723 0.561 0.512 0.302

Table 4. Study on # ranks in SVD-based rectification.

Content Align Style Align Text Align # Tokens CSD-C CLIP-I CSD-S DINO CLIP-T

- 1 0.581 0.701 0.544 0.470 0.316
- 2 0.593 0.713 0.552 0.473 0.316 4 0.603 0.751 0.564 0.517 0.330 8 0.601 0.715 0.550 0.516 0.314

16 0.572 0.683 0.582 0.534 0.301 Table 5. Study on # tokens used for learning content and style.

choose to apply K-V to a single block, as it offers the best balance between efficiency and performance.

Study on SVD-based style rectification. We empirically evaluate different r in Tab. 4 when selecting the top r components from a total of 200, and found r = 10 is optimal.

Study on # tokens for learning content and style. As shown in Tab. 5, increasing # tokens does not consistently improve identity preservation. While a moderate increase (4 tokens) enhances content alignment, using too many tokens (16) introduces artifacts and reduces alignment. Based on these findings, we select 4 tokens as the optimal choice.

### 7. Conclusion

We have introduced CSD-VAR, a novel method for decomposing a single image into separate content and style representations using a Visual Autoregressive model, along with CSD-100 as a benchmark dataset for this task. By analyzing scale-level details, we have proposed a scale-aware alternating optimization strategy to enhance content-style disentanglement. Additionally, we have introduced SVDbased style embedding to mitigate content leakage and augmented K-V memories to improve subject identity preservation. Our experiments show that CSD-VAR outperforms existing methods, establishing it as an effective approach for controllable text-to-image generation and creative exploration from a single image.

Discussion. While our method achieves significant improvements, it struggles with images containing subjects with intricate details, highlighting the need for better disentanglement and fine-grained representation learning. In future work, we aim to refine our approach to better capture such details and expand CSD-100 beyond an evaluation benchmark, exploring its potential as a training dataset for learning-based content-style decomposition methods.

### References

- [1] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, New York, NY, USA, 2023. Association for Computing Machinery. 3
- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9630–9640, 2021. 5, 6
- [3] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024. 2
- [4] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2818–2829, 2023. 4
- [5] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher. In European Conference on Computer Vision, pages 176–192. Springer, 2024. 2
- [6] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021. 3
- [7] Zhengcong Fei, Mingyuan Fan, and Junshi Huang. Gradientfree textual inversion. In Proceedings of the 31st ACM International Conference on Multimedia, page 1364–1373, New York, NY, USA, 2023. Association for Computing Machinery. 1
- [8] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part X, page 181–198, Berlin, Heidelberg, 2024. SpringerVerlag. 2, 3, 6, 7, 12
- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 1, 2, 4
- [10] Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. Tokenverse: Versatile multi-concept personalization in token modulation space. arXiv preprint arXiv:2501.12224, 2025. 2, 3
- [11] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Ar-

- tificial Intelligence and Statistics, pages 249–256, Chia Laguna Resort, Sardinia, Italy, 2010. PMLR. 6
- [12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [13] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36:15890–15902, 2023. 2
- [14] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024. 2, 4, 6
- [15] Shaozhe Hao, Kai Han, Zhengyao Lv, Shihao Zhao, and Kwan-Yee K. Wong. Conceptexpress: Harnessing diffusion models for single-image unsupervised concept extraction. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LIX, page 215–233, Berlin, Heidelberg, 2024. SpringerVerlag. 1, 3
- [16] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4775– 4785, 2024. 5, 6
- [17] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019. 2
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2
- [19] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin C.K. Chan, and Ziwei Liu. Reversion: Diffusion-based relation inversion from images. In SIGGRAPH Asia 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 1, 2, 3
- [20] Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, and Philip Teare. An image is worth multiple words: Discovering object level concepts using multi-concept prompt learning. In Forty-first International Conference on Machine Learning, 2024. 1, 2, 3
- [21] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1931–1941, 2023. 5, 6
- [22] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 2

- [23] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 6
- [24] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2
- [25] Yangyang Li, Daqing Liu, Wu Liu, Allen He, Xinchen Liu, Yongdong Zhang, and Guoqing Jin. Omniprism: Learning disentangled visual concept for image generation. arXiv preprint arXiv:2412.12242, 2024. 3
- [26] Chang Liu, Viraj Shah, Aiyu Cui, and Svetlana Lazebnik. Unziplora: Separating content and style from a single image. arXiv preprint arXiv:2412.04465, 2024. 2, 3, 6, 7, 12
- [27] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusionbased text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 2
- [28] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2
- [29] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 2
- [30] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 2
- [31] Saman Motamed, Danda Pani Paudel, and Luc Van Gool. Lego: Learning to disentangle and invert personalized concepts beyond object appearance in text-to-image diffusion models. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XV, page 116–133, Berlin, Heidelberg,

2024. Springer-Verlag. 1, 3

- [32] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7807–7816,

2024. 2

- [33] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations, 2025. 5, 12
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4, 6
- [35] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 2

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [37] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and WenSheng Chu. RB-modulation: Training-free stylization using reference-based modulation. In The Thirteenth International Conference on Learning Representations, 2025. 5, 6
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 5, 6, 7
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [40] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. 2
- [41] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina Blok, and Daniel Castro Chin. Styledrop: Text-to-image synthesis of any style. In Advances in Neural Information Processing Systems, pages 66860–66889. Curran Associates, Inc., 2023. 5, 6, 12
- [42] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Investigating style similarity in diffusion models. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXVI, page 143–160, Berlin, Heidelberg, 2024. Springer-Verlag. 5, 6
- [43] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2
- [44] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. HART: Efficient visual generation with hybrid autoregressive transformer. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [45] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [46] OpenAI Team. Gpt-4 technical report, 2024. 5, 6
- [47] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2025. 2, 3
- [48] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste

- Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 5
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4
- [50] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. ACM Trans. Graph., 42(6), 2023. 1, 2, 3, 6, 7
- [51] Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, and Dmitry Baranchuk. Switti: Designing scale-wise transformers for text-to-image synthesis. arXiv preprint arXiv:2412.01819, 2024. 2, 4, 6
- [52] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. In Advances in Neural Information Processing Systems, pages 51008–51025. Curran Associates, Inc., 2023. 2
- [53] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 2
- [54] You Wu, Kean Liu, Xiaoyue Mi, Fan Tang, Juan Cao, and Jintao Li. U-vap: User-specified visual appearance personalization via decoupled self augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9482–9491, 2024. 3
- [55] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [56] Zhi Xu, Shaozhe Hao, and Kai Han. Cusconcept: Customized visual concept decomposition with diffusion models. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pages 3678–3687, 2025. 1
- [57] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in Neural Information Processing Systems, 37: 47455–47487, 2025. 2
- [58] Xulu Zhang, Xiao-Yong Wei, Jinlin Wu, Tianyi Zhang, Zhaoxiang Zhang, Zhen Lei, and Qing Li. Compositional inversion for stable diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7350– 7358, 2024. 1, 2
- [59] Yanbing Zhang, Mengping Yang, Qin Zhou, and Zhe Wang. Attention calibration for disentangled text-to-image personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4764–4774, 2024. 3

###### Prompt for content Prompt for style

|a {} a {} in the jungle a {} in the snow landscape a {} on the beach a {} on a table a {} floating on a river a {} in downtown a {} with a mountain in the background a {} with a wheat field in the background a {} with autumn leaves a {} with the Eiffel Tower in the background a {} in the desert a {} in space a {} in the sky a {} in the swimming pool a {} on the boat a {} look out a {} is lying down a {} in sticker style a {} in pixel art style a {} in anime style a {} in watercolor style a {} made of gold a {} made of silver a {} made of glass, crystal|a dog in {} style a cat in {} style a dinosaur in {} style a rabbit in {} style a capybara in {} style a dragon in {} style a wolf in {} style a fox in {} style a castle in {} style a robot in {} style a deer in {} style a bedroom in {} style a truck in {} style a chair in {} style a car in {} style a flower in {} style a piano in {} style a crown in {} style a bus in {} style a person texting in {} style a laptop in {} style a lighthouse in {} style a coffee cup in {} style a spaceship in {} style a train in {} style|
|---|---|

Figure 9. List of 50 prompts used in our experiments for evaluating content-style decomposition.

### A. Appendix

This appendix includes our supplementary materials as follow:

- • Validation Data in Appendix A.1
- • CSD-100 Dataset Analysis in Appendix A.2
- • Additional Qualitative Results in Appendix A.3
- • More ablation experimental settings in Appendix A.4
- • User Study in Fig. 10

#### A.1. Validation Data

To assess the effectiveness of our scale-wise detail analysis and ablation studies, we construct a validation dataset comprising 35 diverse concepts. These concepts are primarily curated from StyleDrop [41], B-LoRA [8], UnZipLoRA [26], and DreamBench++ [33]. This dataset allows us to systematically validate our assumptions regarding hierarchical scale representations and evaluate different experimental configurations. Fig. 12 provides an overview of the validation dataset, while Fig. 9 presents the set of 50 prompts used across all experiments.

#### A.2. CSD-100

The CSD-100 dataset in Fig. 11 is designed to provide a comprehensive benchmark for content-style decomposition (CSD) models.Fig. 13 illustrates the content and style distributions in the CSD-100 dataset. The dataset comprises 63 distinct objects (content) and 53 unique styles, providing a diverse set of content-style pairs for evaluating decomposition models. Fig. 13 (Left) depicts the frequency distribution of content categories, where most content types appear only once or twice, ensuring a broad variety of objects. Fig. 13 (Right) shows the distribution of styles, with certain styles occurring more frequently, particularly those with well-defined characteristics (e.g., line drawing, geometric shapes, abstract digital art). By balancing content

Input image

[Figure 126]

A <c> in A <c> floating on a river watercolor style A dragon in <s> style A castle in <s> style

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Decomposed Content

Decomposed Style Model A

A <c> in A <c> floating on a river watercolor style A dragon in <s> style A castle in <s> style

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Decomposed Content

Decomposed Style Model B

Figure 10. Our user study interface

Content Align Style Align Text Align # Augmented K, V CSD-C CLIP-I CSD-S DINO CLIP-T

- 1 0.603 0.751 0.564 0.517 0.330
- 2 0.608 0.724 0.560 0.521 0.320 5 0.607 0.732 0.560 0.517 0.318

Table 6. Ablation study on the number of augmented K-V memories used in self-attention.

and style diversity while maintaining a representative distribution, CSD-100 serves as a well-curated benchmark for assessing the effectiveness of content-style decomposition models.

#### A.3. Additional Qualitative Results

We present additional qualitative results of our method, as shown in Fig. 14

#### A.4. More ablation experimental settings

Study on the Number of Augmented K-V Memories. The results in Tab. 6 show that increasing the number of K-V memories does not consistently improve performance across all metrics. We hypothesize that adding more K-V pairs complicates model distribution alignment, leading to diminishing returns. As a result, we adopt a single K-V memory as the optimal setting.

#### A.5. User Study

We conducted a user study with 100 participants to compare our model against an alternative method. Each participant answered 15 questions, selecting their preferred output based on five multiple-choice criteria. For each question, users evaluated two examples per output group, focusing on content and style (see Fig. 10). To ensure fairness, model outputs were anonymized and their order randomized.

[Figure 135]

###### Figure 11. Visualization of the full CSD-100 dataset, showcasing its diverse content and style pairings

StyleDrop, B-LoRA UnZipLoRA

|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|[Figure 140]|
|---|---|---|---|---|
|[Figure 141]|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|
|[Figure 146]|[Figure 147]|[Figure 148]|[Figure 149]|[Figure 150]|
|[Figure 151]|[Figure 152]|[Figure 153]|[Figure 154]|[Figure 155]|

|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|
|---|---|---|---|
|[Figure 160]|[Figure 161]|[Figure 162]|[Figure 163]|
|[Figure 164]|[Figure 165]|[Figure 166]|[Figure 167]|
|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|

DreamBench++ External sources

Figure 12. Overview of the validation dataset, consisting of 35 curated concepts sourced from existing personalization benchmarks.

[Figure 172]

Figure 13. Distribution of content and style in the CSD-100 dataset

[Figure 173]

[Figure 174]

##### Input

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

|[Figure 181]|
|---|

A castle in ... A deer in ... A girl in ...

...made of glass ...in anime style ...on the boat

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

A boy wears crown in ...

A castle in ... A capybara in ...

...in wheat field ...made of crystal ...in bathroom

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

A flower in ... A cat in ... A chair in ...

...in watercolor ...as sticker ...floating on water

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

A dragon in ... A castle in ... A robot in ...

...in space ...is swimming ...as sticker

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

...made of gold ...with autumn leaves

A flower in ... A robot in ... A dog in ...

...on the beach

Figure 14. Additional qualitative results from our CSD-VAR

