## Paint by Inpaint: Learning to Add Image Objects by Removing Them First

Navve Wasserman1*, Noam Rotstein2*, Roy Ganz2, Ron Kimmel2

1 Weizmann Institute of Science 2 Technion - Israel Institute of Technology

*Indicates equal contribution.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

# arXiv:2404.18212v3[cs.CV]20Mar2025

Add a white buttoned shirt

Add steamed milk

Add goggles

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Add a flowing river

Add a big flat TV

Add pink flowers

Figure 1. Visual Results of the Proposed Models.

### Abstract

as each image offers infinite editing possibilities, each with countless potential outcomes. A particularly intricate editing task is seamlessly adding objects to images, which requires not only realistic visuals but also a nuanced understanding of the global image context, including parameters such as location, scale, and style. While many solutions require the user to provide a mask for the target object [32, 52, 64, 67], recent advancements have capitalized on the success of text-conditioned diffusion models to enable a mask-free approach [4, 73]. Such solutions offer a more convenient and realistic setting; yet, they still encounter challenges, as demonstrated in Figure 3.

Image editing has advanced significantly with the introduction of text-conditioned diffusion models. Despite this progress, seamlessly adding objects to images based on textual instructions without requiring user-provided input masks remains a challenge. We address this by leveraging the insight that removing objects (Inpaint) is significantly simpler than its inverse process of adding them (Paint), attributed to inpainting models that benefit from segmentation mask guidance. Capitalizing on this realization, by implementing an automated and extensive pipeline, we curate a filtered large-scale image dataset containing pairs of images and their corresponding object-removed versions. Using these pairs, we train a diffusion model to inverse the inpainting process, effectively adding objects into images. Unlike other editing datasets, ours features natural target images instead of synthetic ones while ensuring source-target consistency by construction. Additionally, we utilize a large Vision-Language Model to provide detailed descriptions of the removed objects and a Large Language Model to convert these descriptions into diverse, naturallanguage instructions. Our quantitative and qualitative results show that the trained model surpasses existing models in both object addition and general editing tasks. Visit our project page for the released dataset and trained models.

The leading method for such editing, InstructPix2Pix (IP2P) [4], synthesizes a dataset containing triplets of source and target images alongside an editing instruction as guidance. Under this guidance, a model is trained to transform source images into target ones. While showing some success, the model’s effectiveness is bounded by the quality of the synthesized training data. We address this limitation by introducing an alternative automatic method for creating a large-scale, high-quality dataset for image object addition. Our approach is grounded in the observation that adding objects (paint) is essentially the inverse of removing them (inpaint). Namely, by using pairs of images—ones containing objects and others with objects removed—an object addition dataset can be established. In practice, we create the dataset by leveraging abundant images and object masks available in segmentation datasets [17, 29, 33] alongside a high-end inpainting model [52]. The outputs are then used in a reverse manner, with the original images as editing targets and the inpainted ones as sources. This reversed ap-

### 1. Introduction

Image editing plays a central role in the computer vision and graphics communities, with diverse applications spanning various domains. The task is inherently challenging

proach is essential because directly adding objects with an inpainting model requires object segmentations not present in the images. Our approach offers two key advantages over IP2P: (i) While IP2P relies on synthetic source and target images, our targets are real natural images, with source images also being natural outside the typically small edited regions. (ii) Despite employing techniques such as prompt-to-prompt [18] and Directional CLIP-based filtering [12] to address source-target consistency issues, IP2P often fails to achieve this. Our approach inherently maintains consistency by construction.

Mask-based inpainting models have recently shown great success in filling image masks naturally and coherently [52]. However, since these models were not trained specifically for object removal, their use for this purpose is not guaranteed to be artifact-free, potentially leaving remnants of the original object, unintentionally creating new objects, or causing other distortions. Given that the outputs of inpainting serve as training data, these artifacts could potentially impair the performance of the resulting models. To counteract these issues, we propose a comprehensive pipeline of varied filtering and refinement techniques. Additionally, we complement the source and target image pairs with natural language editing instructions by harnessing advancements in multimodal learning [2, 10, 15, 16, 31, 35, 54]. By employing a Large Vision-Language Model (VLM) [66], we generate elaborated captions for the target objects. Next, we utilize a Large Language Model (LLM) [24] to cast these descriptions to natural language instructions for object addition. To further enhance our dataset, we incorporate human-annotated object reference datasets [26, 39] and convert them into adding instructions. Overall, we combine these sources to form an instruction-based object addition dataset, named PIPE (Paint by Inpaint Editing). Unprecedented in size, our dataset features approximately 1 million image pairs, spans over 1400 different classes, and includes thousands of unique attributes.

Utilizing PIPE, we train a diffusion model to follow object addition instructions, setting a new standard for adding realistic image objects, as demonstrated in Figure 1, and as validated across extensive experiments on multiple benchmarks. Besides quantitative results, we conduct a human evaluation survey comparing our model to top-performing models, showcasing its improved capabilities. Furthermore, we demonstrate that PIPE can extend beyond mere object addition; by integrating it with additional editing datasets, we show it significantly improves overall editing results.

###### Our contributions include:

- • Introduction of the Paint by Inpaint framework for image editing.
- • Construction of PIPE, a large-scale, high-quality, maskfree, textual instruction-guided object addition image

dataset.

• Demonstration of a diffusion-based model trained with PIPE, achieving state-of-the-art object addition to images and enhanced general editing performance.

### 2. Related Efforts

###### 2.1. Image Editing

Image editing has long been explored in computer graphics and vision [44, 48]. The field has seen substantial advances with the emergence of diffusion-based image synthesis models [21, 60], especially with their text-conditioned variants [42, 51, 52, 58]. The application of such models can be broadly categorized into two distinct approaches – maskbased and mask-free.

Mask-Based Editing. Such approaches formulate image editing as an inpainting task, using a mask to outline the target edit region. Early diffusion-based techniques utilized pretrained models for inpainting [1, 40, 60, 70], while more recent approaches fine-tune the models specifically for this task [42, 52, 57]. Inpainting models benefit from the possibility of training on large-scale image datasets, as they can be trained with any image paired with a random mask. Various attempts have been made to advance this methodology in different directions [32, 64, 67], but despite this progress, relying on a user-provided mask makes this setting less preferable in real-world applications.

Mask-Free Editing. This paradigm allows image editing using text and natural language as an intuitive interactive tool without the need for additional masks. Kawar et al. [25] optimize a model to align its output with a target embedding text. Bar Tal et al. [3] introduces a model that merges an edit layer with the original image. IP2P turns mask-free image editing into a supervised task by generating an instruction-based dataset using an LLM and Prompt-to-Prompt [18], which adjusts cross-attention layers in diffusion models to align attention maps between source and target prompts. Rotstein et al. [55] take a different approach by leveraging video generators. These maskfree techniques are distinguished by their ability to perform global edits such as style transfer. However, they exhibit limitations in local edits, specifically in maintaining consistency outside the desired edit region. IP2P seeks to address this by utilizing Directional CLIP loss [12] for dataset filtering. Nevertheless, it mitigates the limitation, but only to some extent. In contrast, our dataset ensures consistency by strictly limiting changes to the intended edit regions only.

Instructions-Based Editing. A few studies have introduced textual instructions for intuitive, mask-free image editing without complex prompts [11, 74]. IP2P facilitates this by leveraging GPT-3 [5] to create editing instructions from input image captions. Following the advancements in instruction-following capabilities of LLMs [45, 75], Zhang et al. devise a reward function reflecting user preferences on

|[Figure 13]|
|---|

|[Figure 14]| |
|---|---|
| | |
||colored skateboarding dog”<br><br>“Add a dark|
|---|
| |

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

###### Inpaint

###### Paint

|[Figure 18]|
|---|

“The dog has a dark coat and is depicted in an action-packed pose on a skateboard, capturing its dynamic movement and spirit.”

|colored skateboarding dog”<br><br>“Add a dark|
|---|

LLM

VLM

- Figure 2. Paint by Inpaint Framework. Illustration of our two-phase approach: (1) Building PIPE dataset (blue), which involves: (i) Removing the object utilizing a frozen inpainting model and the object mask. (ii) Generating addition instructions, demonstrated through the VLM-LLM-based procedure, where a VLM extracts visual object details and an LLM formulates them into instructions. (2) Training an editing model (orange), PIPE is employed to train a model to reverse the inpainting process, thereby adding objects to images.

objects. Similar to our work, SmartBrush [67] aims to add objects to images. However, unlike our methodology, it requires an input mask from the user. Instruction-based methods like IP2P and MagicBrush highlight their capability to insert image objects, allocating a considerable portion of their dataset for this purpose, for example, 39% of the MagicBrush dataset is dedicated to this task. Following the initial release of our paper, several works have emerged with related ideas [6, 61].

edited images [73]. Our approach takes a different course; it enriches the class-based instructions constructed from the segmentation datasets by employing a VLM [65] to comprehensively describe the target object, and an LLM [24] to transform the VLM outputs into coherent editing instructions. Our dataset is further enhanced by integrating object reference datasets [26, 39].

- 2.2. Image Editing Datasets

Early editing approaches [68, 71] used datasets with specific classes without direct correspondence between source and target images [33, 43, 62]. Building datasets of natural images and their natural edited versions in the maskfree setting is infeasible, as it requires two identical images differing solely in the edited region. Thus, previous works propose synthetic alternatives, with the previously discussed IP2P’s dataset being one of the most prominent ones. MagicBrush [72] recently introduced a partially synthetic dataset, which was manually created using DALLE2 [51]. While offering more accuracy and consistency, its manual annotation and monitoring limit its scalability. InstInpaint [69] leverages segmentation and inpainting models to develop a dataset focused on object removal, designed to eliminate the segmentation step. We introduce a high-quality image editing dataset that exceeds the scale of any currently available ones. Furthermore, our approach, uniquely leverages real images as the edit targets, distinguishing it from prior datasets consisting of synthetic data.

- 2.3. Object Focused Editing Processing specific objects through diffusion models has gained significant attention in recent research. For instance, various methodologies have been developed to generate images of particular subjects [8, 13, 56]. Within the editing domain, Wang et al.[64] concentrate on mask-based object editing, training their model for inpainting within existing object boundaries, while Patashnik et al.[47] introduce a technique for producing diverse variations of such

### 3. PIPE Dataset

As outlined in Section 2, leading mask-free, instructionfollowing image editing models are trained on datasets that are either small-scale or synthetic and inconsistent. To enhance the efficacy of these models, we propose a systematic method to create a dataset that addresses these limitations. The devised dataset, dubbed PIPE (Paint by InPaint Edit), comprises approximately 1 million image pairs accompanied by diverse object addition instructions. Our methodology, illustrated in blue in Figure 2, unfolds in a twostage procedure. First, drawing on the insight that object removal is more straightforward than object addition, we create pairs of source and target images—without and with objects. Subsequently, we generate a natural language object addition instruction for each pair using various techniques. In the following section, we describe the proposed pipeline in detail.

###### 3.1. Generating Source-Target Image Pairs

In the initial stage of creating PIPE, we leverage extensive image segmentation datasets. Specifically, we utilize COCO [33] and Open Images [28], enriched with segmentation mask annotations from LVIS [17]. Unifying these datasets results in 889,230 unique images with over 1,400 object classes. We use this diverse corpus for object removal using a Stable Diffusion (SD) [52] based inpainting model1. This configuration is the underlying reason why

1https://hf.co/runwayml/stable-diffusion-inpainting

Original CLIP-VQGAN Hive IP2P Ours

Original CLIP-VQGAN Hive IP2P Ours

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Add a black and white cat sitting on the ground

Add a tall, slender, blue vase with red floral patterns

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

Add a tie

Put some red flowers in the vase

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Add an egg to the photo

Add a princess

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

Add a wine bottle

Add cake to the table

- Figure 3. Visual Comparison. Comparison of our model with leading editing models across benchmarks, demonstrating superior fidelity to instructions and precise object addition in terms of style, scale, and position, while maintaining higher consistency with original images.

constructing PIPE via removal is more straightforward than via addition. However, since the inpainting model was not trained specifically for object removal, it can yield suboptimal outcomes, e.g., leaving original object traces or generating new objects. To address this, we implement a pipeline of pre-removal and post-removal steps.

Pre-Removal. Object segmentation masks filtering step, retaining only candidates suitable for the subsequent objectadding. First, we exclude masks according to their size (too large or too small) and location (near image borders). Next, we use CLIP to calculate the semantic similarity between segmented objects and their class names, using low values to filter out abnormal object views (e.g., blurred objects) and non-informative partial views (e.g., occluded objects). In Figure 4a, we provide an example of a car being filtered due to its small size and blur, while a person without these characteristics is not (see Fig. S14 for more examples). To ensure the mask fully covers the object, we apply morphological dilation, a crucial step since any unmasked object parts can lead the inpainting model to regenerate it [49].

Object Removal. Given the dilated masks, we remove the objects using the SD inpainting model. Unlike conventional inpainting objectives, which aim at general image completion, our focus centers on object removal. To this end, we guide the model with positive and negative prompts designed to replace objects with non-objects (e.g., background). The positive prompt is set to “a photo of a background, a photo of an empty place”, while the negative prompt is defined

- as “an object, a <class>”, where <class> denotes the object class name. During the inpainting process,

we utilize 10 diffusion steps and generate 3 distinct outputs per input.

Post-Removal. The last part of our removal pipeline involves employing a multi-step process aimed at filtering and refining the inpainting outputs:

• Removal Verification: For each source image and its three inpainted outputs, we introduce two mechanisms to assess removal effectiveness. First, we measure the semantic diversity of the three inpainted candidates’ regions by calculating the standard deviation of their CLIP embeddings, a metric we refer to as the CLIP consensus. Intuitively, high diversity (no consensus) suggests failed object removal, leaving varied non-background object elements, as shown in the upper row of Figure 4b. Conversely, lower variability (consensus) points to a consistent removal, increasing the likelihood of an appropriate background, as demonstrated in the bottom row of the figure. Next, we calculate the CLIP similarity between the inpainted region of each candidate and the class name of the removed object (e.g., <bread>). This procedure, referred to as multimodal CLIP filtering, is illustrated in Figure 4c. Introducing CLIP consensus and multimodal CLIP filtering mechanisms enhances the robustness of the object removal process. If multiple candidates pass all filtering stages, the one with the lowest multimodal CLIP score is selected. Prior to choosing the CLIP Consensus and Multimodal CLIP filters thresholds, we manually annotated 500 inpainted images, classifying them as successful or failed removals. We tested the filters across varying thresholds and plotted the percentage of success-

[Figure 59]

[Figure 60]

|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

|Car|
|---|

[Figure 71]

[Figure 72]

|Bread|
|---|

[Figure 73]

[Figure 74]

|[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|
|---|

|Person|
|---|

(a) Pre-Removal Abnormal View

(b) Post-Removal CLIP Consensus

(c) Post-Removal Multimodal CLIP

- Figure 4. Dataset Filtering Stages. In constructing PIPE, several filtering stages address inpainting drawbacks. Initially, a pre-removal filter targets abnormal object views due to blur and low quality. Subsequently, a post-removal inconsistency filter identifies a lack of CLIP consensus among three inpainting outputs, indicating substantial variance and potential object regeneration. Finally, a post-removal multimodal CLIP filtering ensures low semantic similarity with the original object name.

ful inpainted images against the percentage of filtered images. As shown in Fig. S21 and Fig. S22, as the filters become more aggressive (lower thresholds), the proportion of successful inpainted images increases for both strategies. This implies that both filtering approaches effectively achieve their aim of filtering out unsuccessful inpainting outputs. We selected thresholds where the slope of successful inpainting begins to plateau, minimizing the loss of images while maximizing quality.

using a two-stage process, as illustrated in Figure 2. In the first stage, we mask out non-object regions and insert the devised image into a VLM, namely CogVLM2 [66], prompting it to generate a detailed object caption that includes visual object details and fine-grained attributes. In the second stage, the caption is reformatted into an instruction using the in-context learning (ICL) capabilities of the LLM. Specifically, we utilize Mistral-7B [24] with 5 ICL examples of the required outputs, prompting it to generate instructions of varying lengths and complexity. This two-stage process, designed to mitigate hallucinations frequently encountered with VLMs [36], has been empirically validated as effective and is inspired by research demonstrating that breaking down tasks into specific model roles enhances LLMs performance [63]. Further details of this procedure are provided in the supplementary materials.

- • Consistency Enforcement: We aim to produce image targets that are consistent with the source ones. By conducting α-blending between the source and inpainted image using the object mask, we limit differences to the mask area while ensuring a smooth, natural transition between regions (see example in Fig. S15).

- • Importance Filtering: In the final removal pipeline step, we filter out instances where the removed object has marginal semantic importance, as such edits are unlikely to be user-requested. We use a CLIP image encoder to assess the similarity between source and target images—not limited to the object region—filtering cases exceeding a manually set threshold.

Manual Reference-based Instructions. To enrich our dataset with additional nuanced, compositional object details, we utilize three object reference datasets: RefCOCO, RefCOCO+ [26], and RefCOCOg [39]. We transform the references into instructions using the template: “add a <object reference>”, where “<object reference>” is replaced with the dataset’s object description.

###### 3.2. Generating Object Addition Instructions

The PIPE dataset is designed to include triplets of source and target images, along with corresponding editing instructions in natural language. However, the process outlined in Section 3.1 only produces pairs of images and the raw class name of the object of interest. To address this gap, we introduce three different strategies for enhancing our dataset with instructions:

Incorporating these diverse approaches produces 1,879,919 different realistic object addition instructions, encompassing both concise and detailed editing scenarios. Examples from PIPE using these diverse approaches are presented in Figure 5 and the appendix. In Table 1, PIPE is compared with other image editing datasets. It sets a new benchmark in image and editing instruction count by a significant margin. Notably, it is the only dataset offering real target images and class diversity.

Class name-based instructions. We augment raw object classes into object addition instructions using the format “add a <class>”, leading to simple and concise instructions.

### 4. Model Training

VLM-LLM based instructions. We propose an automatic procedure designed to produce more varied and comprehensive instructions than those based on class names. Leveraging recent VLM and LLM advances, we craft instructions

We detail the methodology used to train an image editing model using the proposed dataset, as illustrated in orange in

2https://hf.co/THUDM/cogvlm-chat-hf

Source Target Source Target

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Add a light-colored plastic frisbee

Add a bus

Source Target Source Target

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Add a bird closest to camera

Add a black round hat with a flat top

- Figure 5. PIPE dataset Examples. Samples from PIPE using different instruction generation techniques: class name-based (left), VLM-LLM based (center), and reference-based (right).

Figure 2. We leverage the SD 1.5 model [52] for both its architecture and initial weights. This text-conditioned diffusion model incorporates a pre-trained variational autoencoder and a U-Net [53], which is responsible for the diffusion denoising within the latent space of the former. We denote the model parameters as θ, the noisy latent variable

- at timestep t as zt, and the corresponding score estimate as eθ. Similar to SD, our editing process is conditioned on a textual instruction encoding cT through cross-attention which integrates text encodings with visual representations. We employ classifier-free guidance (CFG) [20] to enhance alignment between the output image and the instruction en-

coding cT. Contrary to SD, which generates a completely new image, our method involves editing an existing one. Thus, similarly to IP2P, we condition the diffusion process not only on cT but also on the input image, denoted as cI. Liu et al. [38] demonstrated that a diffusion model can be conditioned on multiple targets, adapting CFG accordingly. Using CFG necessitates modeling both conditional and unconditional scores. To facilitate this, during training we set cT = ∅ with probability p = 0.05 (no text conditioning), cI = ∅ with p = 0.05 (no image conditioning), and cI = ∅,cT = ∅ with p = 0.05 (no conditioning). During inference, using CFG, we compute the score considering both the instruction and the source image. Further implementation details and hyperparameters are provided in the appendix.

### 5. Experiments

Image editing can yield countless different valid outcomes, making its evaluation a significant challenge. To address this, we perform a diverse array of experiments. Given that PIPE is primarily designed for object addition, we initially focus our experiments on this task before extending

Dataset Real Real General # # Source Target Classes Images Edits Images Images

Oxford-Flower ✓ ✓ ✗ 8,189 8,189 CUB-Bird ✓ ✓ ✗ 11,788 11,788 EditBench ✓✗ – ✓ 240 960 InstructPix2Pix ✗ ✗ ✓ 313,010 313,010 MagicBrush ✓ ✓✗ ✓ 10,388 10,388

PIPE ✓✗ ✓ ✓ 889,230 1,879,919

Table 1. Datasets Comparison. Review of PIPE with other editing datasets. ✓ signifies fulfillment, ✗ indicates non-fulfillment, and ✓✗ denotes partial fulfillment, where images are real outside inpainted areas. ”–” means no such images available. ”General Classes” indicates dataset class diversity.

its application to general editing (in Section 6). We quantitatively and qualitatively compare our model with topperforming methods, complemented by an in-depth detailed human evaluation survey. Additionally, in the appendix, we include an ablation study of the VLM-LLM pipeline.

- 5.1. Experimental Settings We consider three benchmarks to evaluate our model’s capabilities in object addition – (i) PIPE test set: 750 images from the COCO validation split, generated using the pipeline outlined in Section 3. (ii) OPA [37]: An object placement assessment dataset that includes source and target images, along with objects to be added. (iii) MagicBrush [72]: A partially synthetic image editing benchmark comprising training and testing sets. To evaluate object addition, we automatically filter the dataset for this task (details in the appendix), resulting in a 144 edits subset.
- 5.2. Quantitative Evaluation We compare our model with leading image editing models, including Hive [73], IP2P [4], VQGAN-CLIP [9], SDEdit [40], Null-Text-Inversion [41], Pix2PixZero [46] and Edit-Freindly DDPM [22]. For evaluating objects additions, we use the standardized metrics from MagicBrush [72]. These metrics compare edited outcomes to

ground-truth targets using both model-free (L1 and L2 distances) and model-based (CLIP [50] and DINO [7] embedding cosine distances) measures. Model-free metrics penalize global changes affecting non-object regions, while model-based approaches evaluate overall semantic similarity. When the edited target caption is available, we use CLIP-T [56] to measure its alignment with the edited image. To complement our evaluation, we adopt the recently proposed Conditional Maximum Mean Discrepancy (CMMD) metric [23]. Like the popular Fr´echet Inception Distance (FID) [19], this metric measures the distributional distance between groups of images. However, unlike FID, CMMD uses CLIP embeddings and works effectively with a reduced number of samples, enabling us to measure distribu-

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑ CMMD↓

VQGAN-CLIP .211 .078 .670 .507 .484 .862 SDEdit .168 .057 .765 .572 .325 .539 Null-Text-Inv .072 .017 .877 .817 .299 .303 Pix2PixZero .086 .024 .846 .750 .294 .322 EF-DDPM .110 .030 .844 .716 .328 .342 Hive .095 .026 .846 .782 .297 .353 IP2P .100 .031 .860 .766 .289 .363 Ours .072 .025 .900 .852 .302 .301

Fine-tune on MagicBrush

IP2P .077 .028 .902 .867 .306 .352 Ours .067 .023 .910 .897 .308 .298

Table 2. Results on MagicBrush Top: Our model and various baselines tested on the MagicBrush test set subset. Bottom: Our model and IP2P fine-tuned on MagicBrush.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CMMD↑

Hive .088 .021 .849 .754 .232 IP2P .098 .027 .861 .753 .142 Ours .057 .014 .945 .903 .060

Table 3. Results on PIPE Test Set.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CMMD↑

Hive .126 .041 .802 .670 .481 IP2P .109 .035 .806 .647 .467 Ours .084 .027 .848 .735 .360

Table 4. Results on OPA.

tion distances for small datasets like MagicBrush. To further demonstrate the superiority of our model, we adopt a measure utilized by [4]. This measure, using changing image guidance scales (sI), plots a graph of two metrics of the edited outcome, both independent of a ground-truth target image: (i) CLIP similarity with the input image. (ii) Directional CLIP similarity [14], which evaluates changes between source-target image embeddings and source-target text caption embeddings. This plot presents a trade-off between preserving the original content and achieving the desired edits.

PIPE Test Results. We evaluate our model against instruction-following models, Hive and IP2P, using the PIPE held-out test set and report the results in Table 3. Our model significantly surpasses the baselines in L1 and L2 metrics, confirming its high consistency, and exhibits a higher level of semantic resemblance to the target ground truth image, as reflected in the CLIP-I and DINO scores.

OPA Results. In Table 4, we evaluate our model on the OPA dataset. Our approach achieves the highest performance across all evaluated metrics.

.950

SDedit

CLIPImageSimilarity

Null_Inv

.900

P2P_zero EF DDPM

.850

hive IP2P Ours

.800

.750

.700

.000 .050 .100 .150 .200 .250

CLIP Text-Image Direction Similarity

Figure 6. Consistency-Instruction Trade-off on MagicBrush.

MagicBrush Results. We evaluate our model on the MagicBrush test subset, which includes source and target prompts in addition to instructions. This allows us to compare our performance not only with instruction-following models like Hive and IP2P but also with prompt-based models like VQGAN-CLIP and SDEdit. As presented in Table 2, our model achieves the best results in most target image similarity metrics (L1, CLIP-I, DINO and CMMD).

The target prompts also allow us to compare the CLIPT metric. While our model surpasses most methods in this metric, VQGAN-CLIP significantly outperforms it. This result is expected as the latter maximizes an equivalent objective during the editing process. Although some methods outperform ours in CLIP-T, they fall behind in other metrics. To highlight our model’s superior balance between consistency with the original image and following the instruction, we present comparisons in Fig. 6. As shown, our method outperforms all others in this tradeoff. Following [72], we also fine-tuned our model on the object-addition training subset of MagicBrush and compared it against the similarly fine-tuned IP2P, with our model exceeding IP2P in all metrics. Evaluations across the benchmarks show our model consistently outperforms competitors, affirming not only its high-quality outputs but also its robustness and adaptability across varied domains.

###### 5.3. Qualitative Examples

Fig. 3 qualitatively compares our model with other topperforming models across several datasets. The results illustrate how the proposed model, in contrast to competing approaches, seamlessly adds synthesized objects into images naturally and coherently, while maintaining consistency with the original images before editing. Furthermore, the examples, along with those in Figure 1, demonstrate our model’s ability to generalize beyond its training classes, successfully integrating items such as a ”princess” and ”buttoned shirt”. More examples are shown in the appendix.

Multiple Object Addition In Sec. C of the appendix, we demonstrate how our method can be extended to add multiple objects.

Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑

IP2P .112 .037 .842 .745 .291 IP2P FT .082 .032 .896 .845 .301 Ours+IP2P FT .074 .026 .906 .866 .303

- Table 5. General Editing Results on MagicBrush Test Set. Model performance evaluation on the full general editing MagicBrush test set. The model, trained on the combined PIPE and IP2P dataset and fine-tuned on the MagicBrush training set, surpasses the previously top-performing fine-tuned IP2P, demonstrating the potential of PIPE for enhancing general editing performance.

Methods

Edit faithfulness Quality

Overall [%] Per-image Overall [%] Per-image

IP2P 26.4 28 28.5 31 Ours 73.6 72 71.5 69

- Table 6. Human Evaluation. Comparison of our model with IP2P on edit faithfulness and quality. “Overall” represents the total vote percentage. “Per-image” quantifies the number of images where a model’s outputs were preferred.

- 5.4. Qualitative Evaluation To complement the quantitative analysis, we conduct a human evaluation survey, comparing our model to IP2P. To this end, we randomly sample 100 images from the Conceptual Captions dataset [59] and request human annotators to provide reasonable addition instructions. Next, we perform the edits using both models and request a different set of human evaluators to review their success. We adopt the queries from [72] and ask evaluators to assess two aspects: alignment faithfulness between results and edit requests, and the output’s general quality and consistency. Overall, we collected 1,833 individual responses from 57 different human evaluators, all participants from a pool of random internet users. To minimize biases and ensure an impartial evaluation, they completed the survey unaware of the research goals. We quantify edit faithfulness and output quality using two metrics: (i) overall global preference measured in percentage and (ii) aggregated per-image preference in absolute numbers (summed to 100). The results in Table 6 showcase a substantial preference by human observers for our model’s outputs in both following instructions and image quality. On average, the global preference metric indicates that our model is preferred approximately 72.6% of the time. Additional survey details are provided in the supplementary materials. An additional human evaluation against hive is presented in Tab. S12.
- 6. Leveraging PIPE for General Editing

We explore the application of our dataset in the broader context of image editing, extending its use beyond merely object addition. We combine the IP2P general editing dataset

CLIPImageSimilarity

.910

.900

.890

IP2P FT

Ours+IP2P FT

.880

.090 .100 .110 .120

CLIP Text-Image Direction Similarity

Figure 7. General Editing Consistency-Instruction Trade-off. Trade-off between consistency to input image (Y-axis) and edit adherence (X-axis), with text guidance fixed at 7 and varying image guidance in the range [1, 2.5].

with PIPE and use it to train an editing diffusion model, following the procedure outlined in Section 4. For evaluation, we utilized the entire MagicBrush test set, comparing our model against the IP2P model, both with and without MagicBrush fine-tuning. Diverging from the object addition concentrated approach, the model is fine-tuned using the full MagicBrush training set.

To ensure fairness and reproducibility, all models were run with the same seed. Evaluations were conducted using the script provided by [72], and the official models were employed with their recommended inference parameters. As illustrated in Table 5, our model sets new state-of-the-art scores for the general editing task, surpassing the current leading models. As presented in Figure 7, our fine-tuned model surpasses the current leading IP2P fine-tuned model, demonstrating higher image consistency for the same directional similarity values. The results collectively affirm that the PIPE dataset can be combined with any editing dataset and improve overall performance. In the appendix, we provide a qualitative visual comparison, showcasing the enhanced capabilities of the new model, not limited to object addition, as well as similar plots for the object addition subset used in Section 5.

### 7. Discussion

In this work, we introduce the Paint by Inpaint framework, which identifies and leverages the fact that adding objects to images is fundamentally the inverse process of removing them. Building on this insight, by harnessing the wealth of available segmentation datasets and utilizing a high-performance mask-based inpainting model, we present PIPE, an object addition dataset. Unlike other mask-free, instruction-following editing datasets, PIPE is both largescale and features consistent and natural editing target images. We demonstrate that training a diffusion model on the dataset leads to state-of-the-art performance in instructionbased image editing, proving the value of the PIPE dataset in achieving consistent and realistic image edits.

### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 2
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 2
- [3] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 1, 6, 7, 5
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 2
- [6] Alper Canberk, Maksym Bondarenko, Ege Ozguroglu, Ruoshi Liu, and Carl Vondrick. Erasedraw: Learning to insert objects by erasing them from images. In European Conference on Computer Vision, pages 144–160. Springer, 2024. 3
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6
- [8] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024. 3
- [9] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance, 2022. 6, 5
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2

- [11] Alaaeldin El-Nouby, Shikhar Sharma, Hannes Schulz, Devon Hjelm, Layla El Asri, Samira Ebrahimi Kahou, Yoshua Bengio, and Graham W Taylor. Tell, draw, and repeat: Generating and modifying images based on continual linguistic instruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10304–10312, 2019. 2
- [12] Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip-guided do-

- main adaptation of image generators. arXiv preprint arXiv:2108.00946, 2021. 2
- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [14] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022. 7
- [15] Roy Ganz, Oren Nuriel, Aviad Aberdam, Yair Kittenplon, Shai Mazor, and Ron Litman. Towards models that can see and read, 2023. 2
- [16] Roy Ganz, Yair Kittenplon, Aviad Aberdam, Elad Ben Avraham, Oren Nuriel, Shai Mazor, and Ron Litman. Question aware vision transformer for multimodal reasoning, 2024. 2
- [17] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019. 1, 3
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [22] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469– 12478, 2024. 6, 5
- [23] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9307– 9315, 2024. 6
- [24] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 2, 3, 5, 8
- [25] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 2

- [26] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 2, 3, 5
- [27] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 7
- [28] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020. 3
- [29] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 1
- [30] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 7
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 2
- [32] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 1, 2
- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 1, 3
- [34] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024. 7
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 2
- [36] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024. 5
- [37] Liu Liu, Zhenchen Liu, Bo Zhang, Jiangtong Li, Li Niu, Qingyang Liu, and Liqing Zhang. Opa: object placement assessment dataset. arXiv preprint arXiv:2107.01889, 2021. 6
- [38] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with

- composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 6
- [39] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 2, 3, 5
- [40] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2, 6, 5
- [41] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 6, 5
- [42] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [43] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 722–729. IEEE, 2008. 3
- [44] Byong Mok Oh, Max Chen, Julie Dorsey, and Fr´edo Durand. Image-based modeling and photo editing. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 433–442, 2001. 2
- [45] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 2
- [46] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 6, 5
- [47] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. arXiv preprint arXiv:2303.11306, 2023. 3
- [48] Patrick P´erez, Michel Gangnet, and Andrew Blake. Poisson image editing. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 577–582. 2023. 2
- [49] Markus Pobitzer, Filip Janicki, Mattia Rigotti, and Cristiano Malossi. Outline-guided object inpainting with diffusion models. arXiv preprint arXiv:2402.16421, 2024. 4
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6
- [51] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image gener-

ation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2, 3

- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3, 6, 8
- [53] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation,

2015. 6

- [54] Noam Rotstein, David Bensaid, Shaked Brody, Roy Ganz, and Ron Kimmel. Fusecap: Leveraging large language models to fuse visual data into enriched image captions. arXiv preprint arXiv:2305.17718, 2023. 2
- [55] Noam Rotstein, Gal Yona, Daniel Silver, Roy Velich, David Bensa¨ıd, and Ron Kimmel. Pathways on the image manifold: Image editing via video generation. arXiv preprint arXiv:2411.16819, 2024. 2
- [56] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3, 6
- [57] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1– 10, 2022. 2
- [58] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [59] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 8
- [60] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [61] Gemma Canet Tarr´es, Zhe Lin, Zhifei Zhang, Jianming Zhang, Yizhi Song, Dan Ruta, Andrew Gilbert, John Collomosse, and Soo Ye Kim. Thinking outside the bbox: Unconstrained generative object compositing. arXiv preprint arXiv:2409.04559, 2024. 3
- [62] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 3
- [63] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1– 26, 2024. 5

- [64] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi PontTuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18359– 18369, 2023. 1, 2, 3
- [65] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3, 8
- [66] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2024. 2, 5
- [67] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023. 1, 2, 3
- [68] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1316– 1324, 2018. 3
- [69] Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar. Inst-inpaint: Instructing to remove objects with diffusion models. arXiv preprint arXiv:2304.03246, 2023. 3
- [70] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 2
- [71] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 5907– 5915, 2017. 3
- [72] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024. 3, 6, 7, 8
- [73] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, ChiaChih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618, 2023. 1, 3, 6, 5
- [74] Tianhao Zhang, Hung-Yu Tseng, Lu Jiang, Weilong Yang, Honglak Lee, and Irfan Essa. Text as neural operator: Image manipulation by text instruction. In Proceedings of the 29th ACM International Conference on Multimedia, pages 1893– 1902, 2021. 2
- [75] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and

###### Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 2

## Paint by Inpaint: Learning to Add Image Objects by Removing Them First Supplementary Material

### A. Additional Model Outputs

ject additions but also demonstrate its effectiveness in enhancing outcomes for other complex tasks, such as object replacement.

In continuation of the demonstrations seen in Figure 1, we further show a variety of object additions performed by our model in Figure S8. The editing results showcase the model’s ability to not only add a diverse assortment of objects and object types but also to integrate them seamlessly into images, ensuring the images remain natural and appealing. Additionally, in Figure S9, we provide an example of our model’s capability to generate diverse results for the same edit using different seeds.

We further analyze this model by testing its performance not on the entire MagicBrush dataset as in Section 6, but on the ’addition only’ subset (discussed in Section F.1) and its complementary ’not addition’ subset. The experiments are performed under the same configuration as Section 6. Results for the addition subset and the complementary subset are presented in Table S7. In both subsets, our model outperforms the other models, indicating that although our dataset focuses on adding instructions, the inclusion of a large amount of high-quality editing data enhances performance for general editing tasks as well.

### B. General Editing

As detailed in Section 6, the model, trained on the combined IP2P and PIPE dataset, achieves new state-of-the-art scores for the general editing task. In Figure S11, we present a visual comparison that contrasts our model’s performance with that of a model trained without the PIPE dataset. The results not only underscore our model’s superiority in ob-

### C. Multiple Object Addition

A straightforward extension of our model allows for adding multiple objects by applying it recurrently, each time with a different addition prompt. However, this approach poses

Add books Add noise canceling headphones Add happy golden retriever dogs

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Add pancakes Add a flying eagle Add pink flowers

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Add a king’s cape Add a flowing river Add green grapes

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- Figure S8. Additional Object Addition Results of the Proposed Model. The first two rows showcase outcomes from the model trained only with the PIPE dataset. The last row presents results from the same model after fine-tuning on the MagicBrush training set, as detailed in Section 5.2.

###### Addition Subset Non-Addition Subset

|Methods L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑<br><br>|L1↓ L2↓ CLIP-I↑ DINO↑ CLIP-T↑|
|---|---|
|IP2P .100 .031 .860 .700 .289 IP2P FT .077 .028 .902 .867 .306 Ours + IP2P FT .069 .024 .913 .889 .308|.114 .038 .839 .742 .290 .083 .032 .895 .841 .300 .075 .027 .905 .862 .303|

- Table S7. Global Editing Performance on Addition and Non-Addition MagicBrush Subsets. Evaluation of our global editing model performance on both the add and complementary non-add instruction subsets of MagicBrush. The model, trained on the combined PIPE and IP2P datasets and fine-tuned on the MagicBrush training set, surpasses IP2P and the fine-tuned IP2P models in both subsets.

source

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

|[Figure 111]|
|---|

Add a plain orange beanie

Add an orange wool scarf

Add a dark brown pair of pants

Add a red logo

Add black long sleeves

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Add a wooden coffee table

Add a potted plant

Add a Persian carpet

Add a long creamcolored couch

Add a large flat TV

[Figure 118]

[Figure 119]

Figure S10. Multiple objects. Example of multiple object addition using latent editing, where successive objects are added without intermediate decoding and encoding.

always accurately handled (see the right images in the figure). These challenges stem from the dataset construction, as our method minimizes alterations outside the near-object region. Future work could explore inpainting both the object and distant regions influenced by it. We hope our work inspires future research to address these limitations.

Add a tattoo

- Figure S9. Editing Diversity. We generated three distinct edited images from the same source image, demonstrating the diversity of our model’s outputs.

### E. PIPE Dataset

a challenge because each object addition requires a decode–encode cycle through the Stable Diffusion variational autoencoder (VAE), where each pass degrades image quality (see Fig. S15). To mitigate this, we perform all edits in latent space—encoding only before the first addition and decoding only after the final addition. Fig. S10 illustrates this process, demonstrating the successful addition of objects without intermediate decoding.

###### E.1. Creating Source-Target Image Pairs

We offer additional details on the post-removal steps described in Section 3.1. The post-removal process involves assessing the CLIP similarity between the class name of the removed object and the inpainted area. This assessment helps evaluate the quality of the object removal, ensuring no objects from the same class remain. To measure CLIP similarity for the inpainted area only, we counter the challenge of CLIP’s unfamiliarity with masked images by reducing the background’s influence on the analysis. We do this by adjusting the background to match the image’s average color and integrating the masked area with this unified background color. A dilated mask smoothed with a Gaussian blur is employed to soften the edges, facilitating a more seamless and natural-looking blend.

### D. Limitations

Despite the impressive results produced by our model, several limitations remain. First, while our data curation pipeline improves robustness during the removal phase, it is not entirely error-free. Additionally, the model struggles with significant changes occurring far from the object but are affected by it. For instance, it handles nearby effects, like TV shadows (see Fig. S12), but struggles with larger shadows or distant reflections, as seen in the center images of Fig. S12. Similarly, object-object interactions are not

To complement the CLIP score similarity, we introduce an additional measure that quantifies the shift in similarity before and after removal. Removals with a high preremoval similarity score, followed by a comparatively lower

###### IP2P FT

###### IP2P FT

###### PIPE + IP2P FT

###### Original PIPE + IP2P FT

###### Original

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Make the drinks blue.

Let the toilet bowl have a lid.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Let’s add a drawing of a girl to the wall.

What if he was with a backpack?

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Make the bust a fire truck.

Add drawing to the refrigerator

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Have there be a model posing next to the sheep

Replace the dove with an owl.

- Figure S11. Visual Comparison on General Editing Tasks. The contribution of the PIPE dataset when combined with the IP2P dataset for general editing tasks, as evaluated on the full MagicBrush test set. The comparison is between a model trained on these merged datasets and a model trained solely on the IP2P dataset, with both models fine-tuned on the MagicBrush training set. The results demonstrate that, although the PIPE dataset focuses solely on object addition instructions, it enhances performance across a variety of editing tasks.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

| |
|---|

[Figure 148]

| |
|---|

Add a yellow umbrella

[Figure 149]

| |
|---|

[Figure 150]

[Figure 151]

Add a big flat TV Add a wide long dress

[Figure 152]

[Figure 153]

| |
|---|

[Figure 154]

[Figure 155]

| |
|---|

[Figure 156]

[Figure 157]

Add a cap

- Figure S12. Limitations. Left: Successful shadow generation near the object. Center: Failures in generating shadows or reflections when distant from the object. Right: Failure in changing hand posture and maintaining the original one.

[Figure 158]

##### Source Target Source Target Source Target Source Target

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Add a a yellow, slightly wrinkled lemon at the top right

Add a lady in the left side with a black jacket

Add a right horse

Add a medium-sized, brown dog with a calm expression at the center right

[Figure 167]

[Figure 168]

#### Source Target Source Target Source Target Source Target

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Add car at the top Add a person wearing a

Add a person

Add a slender silver fish with a yellow tail at the left

helmet, a gray jacket, blue jeans and holding a mobile phone

[Figure 178]

###### Figure S13. Additional PIPE Datasets Examples.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

| |
|---|

[Figure 186]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure S14. Pre-Removal Filtered Examples. Left: Objects with noninformative view and low CLIP Object similarity. Right: Extremely small and large objects, unsuitable for our dataset.

Figure S15. Consistency Enforcement Examples. From left to right: original image, inpainted dog image, inpainted image after alpha blending.

yet significant post-removal score are not filtered, even though they exceed the threshold. This method allows for the efficient exclusion of removals, even when other objects of the same class are in close spatial proximity.

the <class name>. Use few words which best describe the <class- name>”. This process yields an in-depth description centered on the object, highlighting key attributes such as shape, color, and texture. Subsequently, this description is provided to the LLM along with human-crafted prompts for In-Context Learning (ICL), to generate succinct and clear instructions. The implementation of the ICL mechanism is detailed in Table S9.

Figure S21 and Figure S22 present the figures discussed in Section 3.1 related to filtering thresholds and their justification. Table S8 reports the number of images before and after each filtering stage.

- Table S8. Statistics on the dataset before and after each Filtering step.

Furthermore, we enrich the instructions by including a coarse language-based description of the object’s location within the image, derived from the given mask. To accomplish this, we split the image into a nine-section grid and assign each section a descriptive label (e.g., top-right). This spatial description is then randomly appended to the instruction with a 25% probability during the training process.

Initial Pre-Removal Consensus MM CLIP Importance 4,646K 1,494K 1,101K 986K 888K

###### E.2. VLM-LLM Based Instructions

Using a VLM and an LLM, we convert the class names of objects from the segmentation dataset into detailed natural language instructions (Section 3.2). Initially, for each image, we present the masked image (featuring only the object) to CogVLM with the prompt: “Accurately describe the main characteristics of

###### E.3. Integrating Instruction Types

As detailed in Section 3.2, we construct our instructions using three approaches: (i) class name-based (ii) VLM-LLM based, and (iii) manual reference-based. These three cate-

- Table S9. In-Context Learning Prompt. (Top) We provide the model with five examples of captions and their corresponding humanannotated responses. (Bottom) We introduce it with a new caption and request it to provide an instruction.

[USER]: Convert the following sentence into a short image addition instruction: ¡caption 0¿. Use straightforward language and describe only the ¡class name 0¿. Ignore surroundings and background and avoid pictorial description. [ASSISTANT]: ¡example response 0¿

. [USER]: Convert the following sentence into a short image addition instruction: ¡caption 4¿. Use straightforward language and describe only the ¡class name 4¿. Ignore surroundings and background and avoid pictorial description. [ASSISTANT]: ¡example response 4¿

[USER]: Convert the following sentence into a short image addition instruction: ¡new caption¿. Use straightforward language and describe only the ¡new class name¿. Ignore surroundings and background and avoid pictorial description. [ASSISTANT]:

gories are then integrated to assemble the final dataset. The dataset includes 887,773 instances each from Class namebased and VLM-LLM-based methods, with an additional 104,373 from Manual reference-based instructions.

###### E.4. Additional Examples

In Figure S13, we provide further instances of the PIPE dataset that complement those in Figure 5.

### F. Implementation Details

As noted in Section 4, the training of our editing model is initialized with the SD v1.5 model. Conditions are set with cT = ∅, cI = ∅, and both cT = cI = ∅ occurring with a 5% probability each. The input resolution during training is adjusted to 256, applying random cropping for variation. Each GPU manages a batch size of 128. The model undergoes training for 60 epochs, utilizing the ADAM optimizer. It employs a learning rate of 5 · 10−5, without a warm-up phase. Gradient accumulation is set to occur over four steps preceding each update, and the maximum gradient norm is clipped at 1. Utilizing eight NVIDIA A100 GPUs, the total effective batch size, considering the per-GPU batch size, the number of GPUs, and gradient accumulation steps, reaches 4096 (128 · 8 · 4).

For the fine-tuning phase on the MagicBrush training set (Section 5.2), we adjust the learning rate to 10−6 and set the batch size to 8 per GPU, omitting gradient accumulation, and train for 250 epochs.

- F.1. MagicBrush Subset To initially focus our analysis on the specific task of object addition, we applied an automated filtering process to

the MagicBrush dataset. This process aims to isolate image pairs and associated instructions that exclusively pertained to object addition. To ensure an unbiased methodology, we applied an automatic filtering rule across the entire dataset. The filtering criterion applied retained instructions explicitly containing the verbs ”add” or ”put,” indicating object addition. Concurrently, instructions with ”remove” were excluded to avoid object replacement scenarios, and those with the conjunction ”and” were omitted to prevent cases involving multiple instructions.

###### F.2. Evaluation

In our comparative analysis in Section 5.2, we assess our model against leading instruction-following image editing models. To ensure a fair and consistent evaluation across all models, we employed a fixed seed (0) for all comparisons.

Our primary analysis focuses on two instruction-guided models, IP2P [4] and Hive [73]. For IP2P, we utilized the Hugging Face diffusers model and pipeline3, adhering to the default inference parameters. Similarly, for Hive, we employed the official implementation provided by the authors4, with the documented default parameters.

Our comparison extends to models that utilize global descriptions: VQGAN-CLIP [9] Null-Text-Inversion [41], Pix2PixZero [46], Edit-Freindly DDPM [22] and SDEdit [40]. These models were chosen for evaluation within the MagicBrush dataset, as global descriptions are not available in both the OPA and our PIPE dataset. For

- 3https://hf.co/docs/diffusers/training/instructpix2pix
- 4https://github.com/salesforce/HIVE

Original Hive IP2P Ours

Original Hive IP2P Ours

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Add a white tethered cow

Add a zebra

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Add a green and white passenger train

Add a parking meter

- Figure S16. Visual Comparison of the Proposed Model on PIPE Test Set. The visual evaluation highlights the effectiveness of our method against other leading models on the PIPE test set. Our model excels in adhering closely to specified instructions and accurately generating objects in terms such as style, scale, and location.

Put a skateboard on the wall.

Add a car driving out of the garage

Add a mascot

Original CLIP-VQGAN Hive IP2P Ours

Put a cup of coffee on the background.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Original CLIP-VQGAN Hive IP2P Ours

- Figure S17. Visual Comparison of the Proposed Model on MagicBrush Test Subset. Our method versus leading models within the MagicBrush object addition test subset. It illustrates our model’s superior generalization across varied instructions and datasets, outperforming the other approaches.

.000 .050 .100 .150

CLIP Text-Image Direction Similarity

.700

.800

.900

.950

CLIPImageSimilarity

Pix2Pix

Ours

- Figure S18. Model Consistency-Instruction Trade-off: Tradeoff between consistency with the input image (Y-axis) and edit adherence (X-axis) for IP2P and our model on the MagicBrush test subset. Text guidance is fixed at 7, and image guidance ranges from 1 to 2.5.

CLIPImageSimilarity

.910

.900

.890

Pix2Pix + FT

.880

Ours FT

.100 .110 .120 .130 .140

CLIP Text-Image Direction Similarity

Figure S19. Finetuned-Model Consistency-Instruction Tradeoff: Trade-off between consistency with the input image (Y-axis) and edit adherence (X-axis) for IP2P and our model, both fine-tuned on the MagicBrush training set and tested on its test subset. Text guidance is fixed at 7, and image guidance ranges from 1 to 2.5.

Model VQGAN-CLIP SDEdit NTI P2P-Z EFD Hive IP2P Ours VQAScore 0.7675 0.6114 0.6008 0.5356 0.6792 0.5822 0.5408 0.7045

Table S10. VQAScore Metric. We use VQAScore [34] as a VQA-based alignment metric to further evaluate our method.

.975

.950

CLIPImageSimilarity

.925

.900

.875

.850

IP2P Ours Ours_FT

.825

.800

FlowEdit_FLUX

.040 .060 .080 .100 .120 .140

CLIP Text-Image Direction Similarity

Figure S20. Comparison to FLUX-Based Method. We compared our model to a FLUX-based approach, specifically the recently released FlowEdit [27]. As shown in the plot, our model significantly outperforms FlowEdit.

VQGAN-CLIP5, Null-Text-Inversion6 and Edit-Freindly DDPM7, we used the official code base with the default hyperparameters. For SDEdit8 and Pix2PixZero9, we used the image-to-image pipeline of the Diffusers library with the default parameters.

We also evaluated our fine-tuned model against the MagicBrush fine-tuned model, as documented in [72]. Although this model does not serve as a measure of generalizability, it provides a valuable benchmark within the specific context of the MagicBrush dataset. For this comparison, we employed the model checkpoint and parameters as recommended on the official GitHub repository of the MagicBrush project10. In Figure S16 and Figure S17, we provide additional qualitative examples on the tested datasets to complement the ones in Figure 3. We further assess the model’s performance on the MagicBrush subset using the same CLIP Image similarity versus Directional CLIP similarity measure, as explained in Section 6. We plot this measure to compare the IP2P model with our model in Figure S18 and the MagicBrush fine-tuned models in Figure S19. As shown in both comparisons, our models present

- 5https://github.com/nerdyrodent/VQGAN-CLIP
- 6https://github.com/google/prompt-to-prompt/blob/main/

null_text_w_ptp.ipynb

- 7https://github.com/inbarhub/DDPM_inversion
- 8https://hf.co/docs/diffusers/en/api/pipelines/stable_

diffusion/img2img

- 9https://hf.co/docs/diffusers/main/en/api/pipelines/

pix2pix_zero

- 10https://github.com/OSU-NLP-Group/MagicBrush

0.860

SuccessfulInpainting

| | | | | |0.|1150.110|
|---|---|---|---|---|---|---|
| | | |0|.1350.13|0 0.120| |
| | | | | | | |
| | | |0.140 0.145| | | |
| | |0.1|50| | | |
| |0|0.160 0.170<br><br>.185| | | | |
| |0.20|0| | | | |
| | | | | | | |

0.850

Percentage

0.840

0.830

0.820

0.810

0.800

0.790

0.000 0.100 0.200 0.300 0.400 0.500

Filtered Images Percentage

- Figure S21. Concensus Filtering Success for varying Thresholds

.000 .040 .080 .120 .160

Filtered Images Percentage

.796

.798

.800

.802

.804

.806

.808

.810

SuccessfulInpainting

Percentage

0.260

0.265

0.270 0.275

0.280

0.285 0.290

0.295

- Figure S22. Multimodal CLIP Filtering Success for varying Thresholds

a better trade-off between consistency with the input image and adherence to the edit instruction, achieving higher consistency with the instruction for the same similarity to the input image.

To further evaluate our method with more advanced metrics, we extend Table 2 and report the VQAScore [34], as shown in Table S10. Under this metric, our approach maintains its favorable performance, except for VQGAN-CLIP, which, as discussed in Section 5.2, tends to deviate substantially from the original image. Furthermore, we extend Figure 6 by adding a comparison between our model and a FLUX-based [30] approach, the recently released FlowEdit [27], demonstrating superior performance.

### G. Instructions Ablation

We examine the impact of employing our VLM-LLM pipeline, detailed in Section 3.2, for generating natural language instructions. The outcomes of the pipeline, termed ”long instructions”, are compared with brief, class namebased instructions (e.g., “Add a cat”), referred to as ”short instructions”. In Table S11, we assess a model

Short Original Instructions

Long Instructions

Long Instructions

Short Instructions

###### Original

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Add red Mercedes-Benz bus with a large front windshield and an extended rear section

Let's add a black bear to the stream.

- Figure S23. Instructions Ablation Examples. Qualitative comparison of model performance when trained on ’short’ template-based instructions versus ’long’ instructions generated through our VLM-LLM pipeline. Models trained on the latter exhibit superior performance in interpreting complex instructions and closely aligning object additions with editing requests.

Train Instructions Type L1 ↓ L2 ↓ CLIP-I ↑ DINO↑ CLIP-T ↑

Short Instructions 0.083 0.028 0.900 0.856 0.300 Long Instructions 0.072 0.025 0.900 0.852 0.302

Table S11. Instructions Ablation Analysis. A quantitative comparative analysis of model performance, comparing training on ’short’ class-based instructions to ’long’ instructions generated using the VLM and LLM pipeline. This analysis was performed on MagicBrush subset. The results demonstrate that training with VLM-LLM-based instructions significantly enhances performance, thereby confirming its effectiveness.

trained on the PIPE image pairs, comparing its performance when trained with either long or short inputs. The models are evaluated on MagicBrush subset. As expected, training with long instructions leads to improved performance on MagicBrush. This demonstrates that training with comprehensive instructions generated by our VLM-LLM mechanism benefits at inference time. In addition to quantitative results, we provide qualitative results of both models in Figure S23. As illustrated, the model trained with long instructions shows superior performance in interpreting complex instructions that include detailed descriptions and location references, such as ”Let’s add a black bear to the stream”.

### H. Human Evaluation

While quantitative metrics are important for evaluating image editing performance, they do not fully capture human satisfaction with the edited outcomes. To this end, we conduct a human evaluation survey, as explained in Section 5.4, comparing our model with IP2P and hive (Tab. S12). Following [72], we pose two questions: one regarding the execution of the requested edit and another concerning the overall quality of the resulting images. Figure S24 illustrates examples from our human survey along with the questions posed. Overall, our method leads to better results for human perception. Interestingly, as expected due to how PIPE was constructed, our model maintains a higher level of consistency with the original images in both its success and failure cases. For example, in the third row of Fig-

Edit faithfulness Quality

Methods

Overall Per Overall Per[%] image [%] image

Hive 25.9 21 24.8 22 Ours 74.1 79 75.2 78

Table S12. Human Evaluation against Hive.

ure S24, while IP2P generates a more reliable paraglide, it fails to preserve the original background.

### I. Social Impact and Ethical Consideration

Using PIPE or the model trained with it significantly enhances the ability to add objects to images based on textual instructions. This offers considerable benefits, enabling users to seamlessly and quickly incorporate objects into images, thereby eliminating the need for specialized skills or expensive tools. The field of image editing, specifically the addition of objects, presents potential risks. It could be exploited by malicious individuals to create deceptive or harmful imagery, thus facilitating misinformation or adverse effects. Users are, therefore, encouraged to use our findings responsibly and ethically, ensuring that their applications are secure and constructive. Furthermore, PIPE, was developed using a VLM [65] and an LLM [24], with the model training starting from a SD checkpoint [52]. Given that the models were trained on potentially biased or explicit, unfiltered data, the resulting dataset may reflect these original biases.

Guidelines: Compare the edit instruction with the actual changes made in the edited images. Select one edit that most accurately and consistently implements the edit instruction.

[Figure 229]

Reference A B

[Figure 230]

Add a black and orange pencil

[Figure 231]

[Figure 232]

Add green sunglasses

[Figure 233]

[Figure 234]

Add a paraglide

Guidelines: Select one edited image that exhibits the best image quality. (Some aspects you may consider, such as the preservation of visual fidelity from the original image seamless blending of edited elements with the original image, and the overall natural appearance of the modifications, etc.)

[Figure 235]

Reference A B

[Figure 236]

Add a yellowgreen parrot

[Figure 237]

[Figure 238]

Add an orange frisbee

[Figure 239]

[Figure 240]

Add a bottle of wine

- Figure S24. Human Evaluation Examples. Examples of the qualitative survey against IP2P alongside the response distribution (our method in red and the baseline in blue). The examples include both successful and failed cases of our model. The first three top examples correspond with a question focused on the edit completion, and the three bottom ones on the resulting image quality.

