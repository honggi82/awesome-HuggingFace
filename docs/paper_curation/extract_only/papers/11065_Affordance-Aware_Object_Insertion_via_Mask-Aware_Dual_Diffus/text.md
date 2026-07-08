## Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion

#### Jixuan He1,2,*,† Wanhua Li1,* Ye Liu1,3 Junsik Kim1 Donglai Wei4 Hanspeter Pfister1 1Harvard University 2 Cornell Tech 3The Hong Kong Polytechnic University 4 Boston College

|[Figure 1]|
|---|

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]|
|---|

[Figure 7]

|[Figure 8]|
|---|

# arXiv:2412.14462v2[cs.CV]20Apr2025

[Figure 9]

[Figure 10]

mask affordance insertion affordance insertion

point affordance insertion affordance insertion

|[Figure 11]|
|---|

|[Figure 12]|
|---|

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

bounding box affordance insertion affordance insertion null affordance insertion affordance insertion

Figure 1. Given a foreground-background object-scene pair, our model can perform affordance-aware object insertion conditioning on different position prompts, including points, bounding boxes, masks, and even null prompts.

### Abstract

RGB image and the insertion mask. By explicitly modeling the insertion mask in the diffusion process, MADD effectively facilitates the notion of affordance. Experimental results show that our method outperforms the state-ofthe-art methods and exhibits strong generalization performance on in-the-wild images. Please refer to our code on https://github.com/KaKituken/affordance-aware-any.

As a common image editing operation, image composition involves integrating foreground objects into background scenes. In this paper, we expand the application of the concept of Affordance from human-centered image composition tasks to a more general object-scene composition framework, addressing the complex interplay between foreground objects and background scenes. Following the principle of Affordance, we define the affordance-aware object insertion task, which aims to seamlessly insert any object into any scene with various position prompts. To address the limited data issue and incorporate this task, we constructed the SAM-FB dataset, which contains over 3 million examples across more than 3,000 object categories. Furthermore, we introduce a strong baseline: a Mask-Aware Dual Diffusion (MADD) model, which utilizes a dual-stream architecture to simultaneously denoise the

### 1. Introduction

In the image composition scenario, common sense guides our perception of the authenticity of synthesized images: a person cannot levitate, a water bottle needs a surface for support, and a boat should be floating in the water rather than on the ground. Deviation from such common sense often leads to semantic inconsistencies on synthesized images. From a composition perspective, the background’s semantic richness plays a pivotal role in defining the placement and characteristics of foreground objects. To better

*Equal Contribution. †Work done while at Harvard University.

describe the influence of background semantics on the foreground, we borrow the concept of affordance into objectscene composition tasks. Previously, Kulal et al. [24] explored the concept of human affordance for image synthesis. Their work, focused on inserting humans into masked scenes, extends beyond mere color or view adjustments. To generalize the setting to arbitrary object-scene synthesis, we term the new task Affordance-Aware Object Insertion. This task challenges models to identify suitable locations and make necessary adjustments to foreground objects, ensuring the generated images adhere to physical laws.

This work aims to build a foundation model for affordance-aware object insertion, which can put any object into any scene as shown in Figure 1. There are three primary challenges involved. First, the model must accurately recognize the appropriate affordance relationship between a background and the foreground object to be inserted. Adjustments to the inserted foreground object is crucial for achieving the intended semantic consistency. The second challenge lies in the model’s ability to generalize across a diverse range of foreground objects. Previous generative image editing methods, such as textual inversion [12], DreamBooth [43], and DreamEdit [27], are subject-specific thus hard to generalize. Our goal is to train a model that can generalize to any object. Third, our model should support a variety of input prompts for users to specify insertion locations, ranging from sparse formats like points and bounding boxes to dense masks. Moreover, in the absence of explicit prompts, the model can autonomously determine appropriate insertion locations by analyzing the semantic content of both the background and foreground.

We address these challenges through three components: task, data, and model. Extending the concept of affordance beyond Kulal et al.’s initial scope [24], the task of affordance-aware object insertion aims to place an arbitrary object into any scene, accommodating different positional prompts, even in the absence of explicit positional cues. It has significant implications for applications such as automated dataset synthesis. To support this task, a large-scale dataset is necessary. Existing image composition datasets, such as DreamEditBench [27], are limited in terms of the diversity of foreground object categories and the number of training samples. To overcome these limitations, we curate a new dataset called SAM-FB which is derived from SA1B [23] for affordance learning. SAM-FB contains a variety of foreground object categories and over 3 million samples. With SAM-FB, we further introduce a strong baseline: a Mask-Aware Dual Diffusion (MADD) model to utilize the large-scale data, which is a diffusion-based framework that facilitates the seamless integration of diverse objects into any scene. During the denoising procedure, object position is progressively refined while the target RGB image is synthesized simultaneously, ensuring accurate alignment

between objects and positions to achieve affordance-aware insertion. Furthermore, we present a unified representation for sparse and dense prompts, enabling our model to effectively respond to various types of position inputs.

The contributions of this work are as follows:

- • We introduce affordance-aware object insertion, extending object-scene composition with affordance guidance to enable realistic insertions across diverse prompts.
- • We present SAM-FB, a large-scale dataset with over 3 million samples spanning diverse object categories to support affordance-aware insertion.
- • We propose a strong baseline model with a dual-stream architecture that denoises object appearance and the insertion mask, facilitating affordance learning.

### 2. Related Work

Affordance. J.J. Gibson [3] first introduces the concept of affordance, then a series of papers [4, 11, 15, 29] dug into this concept and brought it into the image synthesis. Initially grounded in psychology, the concept emphasizes that an object’s appearance should correspond with its utilitarian aspects as perceived by humans. Further exploration within the field of image synthesis involved adjusting the orientation and gestures of generated human figures to align with their background. Therefore, prior work primarily focuses on the interaction between humans and objects [13, 51, 54] or the human-scene relationship [7, 10, 48]. Kulal et al. [24] made progress by introducing a model trained on person movement video data for placing people within scenes and adjusting its pose according to the surroundings. However, their model’s scope was limited to human figures. Despite these advancements, the concept of ”affordance” in image composition, which encompasses the positioning, viewing angle, and color harmony of objects within scenes, has remained relatively unexplored. Our work offers a generalized and versatile solution for object-scene composition.

Image Editing. Image editing aims to modify existing images using generative models. Generally, image editing can be divided into semantic editing (e.g. adding or removing objects, changing the background), style editing (e.g. altering color or texture), and structural editing (e.g. changing object size or viewpoint). By utilizing generative models like GANs [14] or Diffusion [8], users can edit image content by providing instructions and multi-modal prompts. For instance, InstructPix2Pix [5] and MoEController [26] allow semantic and style editing through text-based instructions provided by users, while Imagen Editor [47] enables more precise control of edit locations by accepting userprovided masks for the areas to be edited. Additionally, other image editing models, such as TF-ICON [35] and ImageBrush [50], accept reference images from users, con-

Model Task Dataset Sample Category Affordance Consistency

[27] Customized Image Composition DreamEditBench 440 22 High (Real Paired) [34] Customized Image Composition MureCom 640 32 High (Real Paired)

[49], [44] Image-guidance Object Insertion Open-Images-v4 1.9M 600 Low (Masked) [30] Image-guidance Object Insertion COCO2014 165K 80 Low (Masked) [24] Human Affordance Insertion Private 2.4M 1 High (Video)

MADD (ours) Object Affordance Insertion SAM-FB 3.2M 3,439 High (Inpainted)

- Table 1. Dataset choice and comparison. We inspect various potential datasets for affordance insertion. SAM-FB contains significantly more samples and object categories. DreamEdit, DreamCom, PBE, ObjectStitch, GLIGEN, Human Affordance.

straining the appearance of objects during editing.

Image Composition. Image composition is a sub-task within image editing, primarily focused on controllably inserting foreground objects into a background based on reference images. Recently, the utilization of generative models empowers high-quality and controllable image composition. Diffusion models have been successfully applied in various domains, including image composition [33, 44], video generation [18], and data augmentation [46]. Stable Diffusion (SD) [42] introduced methods for blending feature and semantic information from images and text, enabling object generation based on prompt instructions. Inpainting [2] is a common approach to achieve image composition. SD-based inpainting models [28, 36, 52] allow users to repaint a specific region with a mask, but it’s hard to insert a desired object into that region. Blended Diffusion [1] enables finer control by leveraging text descriptions of the foreground, but it’s crucial for user to describe the foreground precisely. PBE [49], ObjectStitch [44], and GLI-GEN [30] utilize reference images to incorporate rich visual information, achieving better perspective and color harmony. However, these models require users to provide precise positional cues, such as bounding boxes or masks, to indicate the insertion location. Our proposed method offers a viable solution for handling vague positional information, such as point prompts or even blank prompts.

### 3. Dataset

Instead of manually designing affordance, we use a datadriven approach to extract affordance relationships between the object and background from the large-scale dataset.

##### 3.1. Motivation & Design Decisions

To support affordance-aware object insertion across various objects and scenes, the dataset must provide rich contextual information and extensive coverage and preserve natural affordance cues between objects and their surroundings. Therefore, we designed the dataset with the following key characteristics: 1) Well-aligned input-output pairs. Rather than using techniques like DreamBooth [43] to memorize each object individually, we adopted supervised

fine-tuning to directly learn from large-scale object distributions. Consequently, well-aligned input-output pairs, represented as (f,b,p,x) tuples, are essential. Here, f, b, p, and x denote the foreground image, background image, position prompt, and ground truth image, respectively. 2) Sufficient training samples. Training a model capable of generalizing to diverse scenarios requires a substantial number of training samples. 3) A wide variety of foreground objects. The dataset must encompass a diverse range of foreground objects to ensure that the trained model can generalize across different object categories. However, existing datasets either lack sufficient object diversity or introduce artifacts that disrupt these affordance relationships.

The (f,b,p,x)-formatted data can be derived from multiple datasets relevant to this task, as shown in Table 1. One option is to use datasets designed for Customized Image Composition. Datasets such as DreamEditBench [27] and MureCom [34] are manually captured by placing the same foreground object in different backgrounds under controlled camera settings. However, the cost of constructing such high-quality paired datasets is substantial, resulting in relatively small dataset sizes. Referring to the ImageGuided Object Insertion task, another approach is to leverage large-scale object detection datasets such as COCO [31] and Open-Images [25] to simulate foreground-background pairs. This is achieved by using bounding boxes as position prompts and extracting foreground objects from the original image using masks, treating the remaining portion as the background. However, these datasets are still insufficient to guarantee robust generalization, particularly due to their limited object category diversity. Moreover, masking out the foreground leaves an opaque region in the background, which disrupts affordance relationships. For HumanAffordance [24], they use a vast number of human-centered video to capture the affordance relationship, but the category is human only and the dataset is private. To address these limitations and obtain a sufficient number of training samples while ensuring diversity in foreground objects, we turned to the SA-1B dataset [23], which consists of 11 million highquality images collected from various real-world scenes and 1 billion well-annotated class-agnostic masks. Based on

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

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

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

| | |
|---|---|
|[Figure 43]| |

[Figure 44]

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

Figure 2. Dataset construction pipeline for SAM-FB. The pipeline automatically converts any input image into a tetrad output through four stages, ensuring high-quality foreground and background retention via a rigorous data quality control process.

Filter condition Threshold Reserved Percentage None (Initial) – 100% Relative Size [0.1, 0.75] 7.10% Aspect Ratio ≤ 3 6.88% Components Num. ≤ 4 6.71% Color Std. ≥ 45 1.69% ResNet50 Score ≥ 0.7 0.25%

- Table 2. Reserved percentages for foreground quality control filters. We apply a combination of rule-based and learning-based conditions to ensure high-quality foreground objects are retained through a rigorous filtering process.

SA-1B, we constructed the SAM-FB benchmark dataset for the affordance-aware object insertion task.

- 3.2. Dataset Construction Pipeline

Figure 2 illustrates the pipeline used to construct the aforementioned (f,b,p,x) formatted data. The pipeline automatically transforms any input image into a tetrad output. To enhance affordance representation and improve the overall quality of the SAM-FB dataset, the dataset construction process consists of four key stages.

Foreground Mask Generation. For each unannotated input image, we first apply the Segment Anything Model (SAM) to generate object masks as candidates. Since SAM produces masks at different levels of granularity with varying confidence scores, we perform Non-Maximum Suppression (NMS) with a high threshold (0.6) to remove redundant masks and filter out sub-object-level masks. The remaining masks are used to extract candidate foreground objects.

Foreground Filtering. Although NMS removes some lowquality masks, SAM segmentation can still produce unintended elements due to challenges in controlling granularity and semantic consistency. The resulting masks often suffer from three main issues: Incompleteness: Some objects are partially occluded, resulting in fragmented masks that only

capture small portions of the original object (e.g., half of a tree); Background Masks: Some masks mistakenly segment background elements (e.g., sky, grassland); Undersized / Oversized Objects: Very small objects often have blurry details, reducing dataset quality. To enhance foreground quality, we employ a hybrid quality control (QC) approach combining rule-based and learning-based filtering.

For rule-based filtering, we remove low-quality masks with four constraints: relative size (too small/large objects), aspect ratio (overly thin objects), component count (disjointed objects), and color standard deviation (pure-color backgrounds). While rule-based filtering effectively removes outliers, ensuring object completeness is challenging using predefined rules alone. To address this, we manually annotate 2,000 foreground images with binary labels (good vs. bad) and fine-tune a ResNet-50 [16] classifier for binary classification. The trained classifier assigns a quality score to each foreground object, further refining the dataset. Table 2 demonstrates the specific threshold for each filter operation of our data quality control stage. With these filter operations, only 0.25% of the masks are left, which ensures the quality of the obtained dataset.

Background Inpainting. High-quality foregrounds are then cropped from the original images. Instead of covering the removed regions with opaque masks, which act as strong artificial cues for object position and size, we employ LAMA [45] for background inpainting. LAMA is a simple yet effective method for object removal and hole-filling. Following GroundingSAM [32, 41], we expand the object mask boundary slightly to prevent residual foreground content from remaining in the background. By eliminating such visual cues, our dataset encourages models to infer contextaware object placement based on scene affordances.

Background Inpainting. Despite careful inpainting, some regions inevitably contain inpainting artifacts. To assess the

quality of inpainted backgrounds, we compute the Structural Similarity Index (SSIM) between the inpainted background and the original image. We discard any inpainted background with an SSIM score below a certain threshold to maintain high visual consistency. We empirically set the SSIM threshold to 0.8 based on preliminary experiments assessing inpainting realism. If a background fails the SSIM filter, we discard the entire foreground-background pair, even if the foreground quality is high. This ensures that only high-quality samples are included in the dataset.

*()*, '()*

⋯ *(,'( ⋯

*&,'& *',''

|ℛ|
|---|

!

expertexpert

expert

!!

c

!!"#

Resize

Unet

[Figure 60]

VAE Enc.

(shared)

%!

%!"#

c

expert

#

ℰ

| |[Figure 61]|
|---|---|
| | |

%

MLP DINO

To evaluate quality improvements, we compute the Inception Score (IS) for foregrounds and Fr´echet Inception Distance (FID) for backgrounds, comparing values before and after quality control (QC). After QC, foreground IS improves from 9.55 to 16.68, indicating significantly enhanced foreground clarity, while background FID decreases from 9.23 to 8.46, reflecting better distribution consistency, similar to inpainting tasks. Since FID requires a reference distribution, it is less meaningful for foregrounds, whereas IS is not a reliable metric for complex backgrounds. To justify the diversity of our proposed dataset, we use RAM [53] to recognize the categories of foregrounds we keep. We have identified 3439 different categories, which is a vast improvement over previous datasets. There are no manual annotations required in the pipeline, and users can easily scale up the dataset further with any new source images easily.

c Concatenation

Figure 3. Mask-aware Dual Diffusion Model (MADD). The RGB image feature z and object mask m are jointly denoised, conditioning on the embeddings of the foreground object f, background object b, and the prompt p. (green: reverse process t → t−1)

[Figure 62]

!$| #,%#

both zt

and mt

, which can be trained with:

z

m

'& = '&

L = E ∥ϵz − ˆϵθ(ztz; f, b, p))∥2 +E ∥ϵm − ˆϵθ(mtm; f, b, p)∥2 .

The denoiser model consists of the UNet model and the three encoders for the foreground object f, the background scene b, and the position prompt p, respectively.

%$| #,!#

*̂ = *&

##### 4.2. Architecture Details

UNet Model. We utilize a single UNet with two expert input-output branches to denoise them simultaneously. The two tasks share the entire UNet except for the conv_in, first Down Block, last Up Block, and the conv_out. These independent blocks serve as the Expert input-output branch. Skip connection is also performed between the corresponding Expert input and output branches.

### 4. Method

##### 4.1. Mask-Aware Dual Diffusion Model (MADD)

Preliminaries. Given a foreground object f, a background scene b, and a position prompt p, the goal is to synthesize an image x that integrates the object into the scene, while adhering to affordances and aligning with the position prompt.

Foreground Encoder. In MADD, foreground images serve as a guidance condition and are incorporated into the model via cross-attention. The foreground embedding is extracted using a ViT-like visual encoder. While SD employs the CLIP [40] encoder for text-aligned semantics, we follow Efficient-3DiM [20] and use DINOv2 [37] to preserve finegrained object details, making it more suitable for our task. Background Encoder. Following SD-based image editing methods [5, 44], we employ a pre-trained VAE encoder to convert the background image into a 4-channel latent representation. Since the background is pixel-aligned with the output, we concatenate its latent map with zt before feeding them into the U-Net. The pre-trained VAE decoder then reconstructs the final RGB image from the latent output.

To develop an effective diffusion model, we first extract image features z = E(x) using a pre-trained encoder. Then we predict the object insertion mask m capturing object’s location and size, which can be directly derived from the difference between x and b in the training data. Our MADD model G builds upon the latent structural diffusion framework [33], simultaneously denoising zˆ and mˆ (Figure 3):

zˆ,mˆ = G(f,b,p), (1)

where the estimated image is reconstructed by xˆ = D(zˆ) with the corresponding image decoder. Forward Process. The noised feature maps are

Position Prompt Encoder. We design a unified representation to process different position prompts. Sparse and dense prompts are converted into a 1-channel position map, guiding foreground object placement. Point prompts are transformed into Gaussian heatmaps, bounding boxes are filled with ones inside and zeros outside, and masks are binary

, (2)

###### zt

= αt

z + σt

###### ϵt

###### , mt

= αt

m + σt

###### ϵt

z

m

z

z

z

m

m

m

where ϵz and ϵm ∼ N(0,I) are three independently sampled Gaussian noise, and the time step tz and tm are sampled from a uniform distribution U[0,1].

Reverse Process. The denoiser ˆϵθ simultaneously denoises

FID↓ CLIP Score↑ MSE↓ mask bbox mask bbox mask bbox [42] 15.41 15.47 0.7079 0.8058 860 883

Method

[49] 33.68 24.59 – 0.7664 2373 1615

[30] - 13.82 – 0.7896 – 817 [24] 14.49 14.42 0.8014 0.8637 857 845

Ours 13.53 13.60 0.8727 0.8658 760 775

- Table 3. Method comparisons on the SAM-FB test set. Stable Diffusion, PBE, GLIGEN, Human Affordance.

Prompt Mask Bbox Point Null Avg. FID 13.53 13.60 13.66 13.96 13.69 MSE 760 775 772 860 792 CLIP Score 0.8727 0.8658 0.8567 0.8034 0.8497

- Table 4. Comparison of position prompts on the SAM-FB test set.

maps. For null prompts, an all-one image implies all positions are possible. The position map is resized to match the latent map’s spatial dimensions and concatenated with it before being fed into the diffusion U-Net.

4.3. Implementation Details

We initialize our model with a pre-trained Stable Diffusion Inpainting model and apply strong data augmentation to enhance affordance learning. To optimize fine-tuning, we use a coarse-to-fine strategy: training at 128 × 128 resolution (batch size 1024) for 35K steps, then fine-tuning at 256 × 256 (batch size 256) for 15K steps.

5. Experiments

- 5.1. Results on the SAM-FB Test Set

Evaluation Metrics. Following the previous work [6, 19, 22, 35], we employ the FID score to measure the quality of images obtained by generative models, MSE for pixel similarity and the CLIP score to evaluate the semantic similarity between the edited region and the reference foreground.

Quantitative Results. We compare our method with Stable Diffusion [42], PBE [49], GLI-GEN [30], and Human Affordance [24] on the SAM-FB test set to show the effectiveness of our method. For fairness, we fine-tuned GLIGEN [28] on SAM-FB training data for 2k steps with captions generated by ViT-GPT2 [9, 39] and use Box + Text + Image Inpainting mode to adapt it. Since Human Affordance [24] does not release training data or model weights, we re-trained their model on SAM-FB. The results in Table 3 show that our method attains the best FID score and the highest CLIP score, illustrating our model can be a simple yet strong baseline for affordance-aware object insertion task. Table 4 shows that the mask prompt achieves the best results as it provides more accurate position information.

Qualitative Results. We left 3786 images as test split. Fig-

|Prompt<br><br>|Non-Null Null|
|---|---|
|IoU (↑)|0.5405 0.4696|

Table 5. Mask evaluation

Model Components Method FID (↓) CLIP (↑)

Baseline 18.32 0.79 + Dual diffusion 14.43 0.84 + Expert branch 13.69 0.85

Higher Resolution

256 × 256 13.69 0.85 512 × 512 10.66 0.82

Table 6. Ablation study results on SAM-FB test set.

ure 4 presents the visualization results on the SAM-FB test set. In each group, the leftmost image depicts the background marked with a position prompt. Our MADD predicts the RGB image and mask of the inserted object, which are shown in the last two images of each group. These results demonstrate that MADD not only inserts objects with high quality but also accurately predicts object masks.

Mask Evaluation. Table 5 shows the IoU of the mask between the predicted mask and the ground truth on the SAMFB test set. The relatively high IoU shows that our method learns effective affordance information.

Computational complexity. We access the computational complexity for the newly added modules including Dual Diffusion and Expert Branch. Using Human Affordance model with DINOv2 as baseline, it has 863.02M parameters with 28.3B FLOPs. Our Expert branches shared most of the parameters with the original baseline model and end up with 873.75M parameters with 31.1B FLOPs.

##### 5.2. Ablation Studies

Model Components. Table 6 presents the results of our ablation study. For model design, we replace the image encoder in the Human Affordance model with DINOv2 as the baseline. Next, we diffuse RGB image and object mask simultaneously, but sharing the entire UNet. Finally, we use two Expert branches for mask and RGB streams. The results show that all of them improved performance. Metrics are averaged across different position prompts. Please refer to the supplementary for results on each prompt type.

Higher Resolution. Even though we trained our model at a resolution of 256x256, it is also capable of handling higher-resolution inputs, such as 512x512. To prove that, we then fine-tuned our model on SAM-FB at a resolution of 512x512 for only 200 steps and compared it with the original checkpoint on the test split. Figure 6 shows the results of the original checkpoint and the fine-tuned version at higher resolution using the same input. We highlighted the details of the inserted object. It is clearly demonstrated that the

|[Figure 63]|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

|[Figure 69]|
|---|

[Figure 70]

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

|[Figure 76]|
|---|

[Figure 77]

[Figure 78]

|[Figure 79]|
|---|

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

|[Figure 84]|
|---|

[Figure 85]

[Figure 86]

|[Figure 87]|
|---|

[Figure 88]

[Figure 89]

|[Figure 90]|
|---|

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Figure 2. SAM-FB samples

- Figure 4. Qualitative results of MADD on the SAM-FB test set. Each row corresponds to one type of prompt, i.e., point, bounding box, mask, and null, respectively. Our MADD simultaneously predicts the RGB image and the object mask.

[Figure 95]

[Figure 96]

[Figure 97]

Figure 2. Example of a short caption, which should be centered.

Location Adjustment View Adjustment Size Adjustment

Background Scene Foreground Object Predicted Insertion

[Figure 98]

|[Figure 99]|
|---|

|[Figure 100]|
|---|

[Figure 101]

|[Figure 102]|
|---|

[Figure 103]

[Figure 104]

|[Figure 105]|
|---|

[Figure 106]

Automatic Localization

- Figure 5. We test ambiguous prompts (points and blank) on in-the-wild images. With point prompts, our model adjusts foreground properties for affordance-aware insertion, while it autonomously finds suitable positions when no prompt is given.

model generates sharper edges, clearer reflections, and improved texture details after fine-tuning at higher resolution. The CLIP score drops slightly at 512×512, likely due to the increased complexity of higher-resolution foregrounds, making fine-grained detail synthesis more challenging.

in-the-wild images and compared the results with Stable Diffusion XL [38], GLI-GEN [30], PBE [49] and ObjectStitch [44]. Instead of merely relying on metrics like FID and CLIP Score, we also conducted user study to achieve more comprehensive comparison. We asked 10 users to rank 10 groups of composited image generated from from different model according to the following criteria: 1) Foreground and Background Integration; 2) Foreground Clarity and Detail; 3) Foreground Appearance Consistency with Reference; 4) Lighting and shade on Foreground; 5) Color Consistency. Figure 8 shows the distribution of rank for different models, where rank 1 and rank 5 represent the best and worst quality respectively. Our model achieves 50% of Rank-1 place and 1.60% of the Rank-5 place, which outperforms other methods. Please refer to supplementary for details of each criteria. Our model consistently achieved the highest proportion of Rank-1 placements across all evaluation criteria, as indicated by the largest segments in each pie chart. Especially for keeping the consistency of foreground appearance with reference image. This dominance in Rank-

##### 5.3. Results on In-the-wild Images

We evaluate our model on in-the-wild web-crawled images. Ambiguous Prompts. Training with position augmentations enables the model to capture affordance relationships between objects and scenes. MADD refines object position, size, and view for coherence with the background.

In Figure 5, MADD adjusts a person’s position to stand on the ground rather than floating, aligns a car’s orientation with the lane, and scales coffee beans to match the scene. The model can also place objects without explicit prompts and generate diverse, reasonable insertions based on user input and learned affordance, as shown in Figure 7.

Human Evaluation. To test the generalization ability of

- our MADD model, we performed affordance insertion on

Composition Input 256x256 Resolution 512x512 Resolution

Composition Input 256x256 Resolution 512x512 Resolution

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

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

|[Figure 117]|
|---|

[Figure 118]

|[Figure 119]|
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

|[Figure 125]|
|---|

[Figure 126]

|[Figure 127]|
|---|

[Figure 128]

[Figure 129]

[Figure 130]

|[Figure 131]|
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

can work on images of higher resolution, generating sharper edges, clearer reflections, improved texture details.

|Figure 6. MADD|
|---|

[Figure 137]

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

Figure 9. More in-the-wild affordance-insertion examples. The model can generate an affordance-feasible solution to insert the foreground objects according to the background scene.

- Figure 7. MADD can give different feasible solutions for ambigu-

ous prompts such as point and blank.

[Figure 150]

- Figure 8. Rank distribution for different methods. Our method has the most proportion of rank 1 and least proportion of rank 5.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

masks, MADD treats them as guidance rather than strict constraints, refining placements to ensure coherence. Please refer to the supplementary for additional examples that further demonstrate MADD’s ability to generate suitable and diverse insertions in real-world scenarios.

### 6. Conclusion

In this paper, we extend the concept of affordance beyond human-centered tasks by introducing affordanceaware object insertion, enabling seamless object placement into scenes while adhering to affordance principles. To support this, we construct SAM-FB, a dataset of 3.16 million foreground-background pairs spanning 3,000+ object categories. We propose MADD, a dual-stream diffusion model that jointly denoises foreground appearance and object masks, leveraging affordance relationships for context-aware insertion. MADD accommodates various position prompts and can infer placements autonomously. Our model generalizes well across diverse objects and outperforms prior diffusion-based composition methods, achieving coherent and affordance-aware insertions.

1 distribution across multiple criteria highlights the model’s superior performance compared to others in all aspects.

More Results. Figure 9 presents additional in-the-wild examples of affordance-aware object insertion with various position prompts. For ambiguous prompts, such as points or null inputs, MADD infers a reasonable position and adjusts the foreground accordingly. For bounding boxes and

### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 3
- [2] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000. 3
- [3] Marc H Bornstein. The ecological approach to visual perception, 1980. 2
- [4] Tim Brooks and Alexei A Efros. Hallucinating posecompatible scenes. In European Conference on Computer Vision, pages 510–528. Springer, 2022. 2
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2, 5
- [6] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022. 6
- [7] Vincent Delaitre, David F Fouhey, Ivan Laptev, Josef Sivic, Abhinav Gupta, and Alexei A Efros. Scene semantics from long-term observation of people. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 284–298. Springer, 2012. 2
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6
- [10] David F Fouhey, Vincent Delaitre, Abhinav Gupta, Alexei A Efros, Ivan Laptev, and Josef Sivic. People watching: Human actions as a cue for single view geometry. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 732–745. Springer, 2012. 2
- [11] David F Fouhey, Xiaolong Wang, and Abhinav Gupta. In defense of the direct perception of affordances. arXiv preprint arXiv:1505.01085, 2015. 2
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2
- [13] Georgia Gkioxari, Ross Girshick, Piotr Doll´ar, and Kaiming He. Detecting and recognizing human-object interactions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8359–8367, 2018. 2

- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [15] Abhinav Gupta, Scott Satkin, Alexei A Efros, and Martial Hebert. From 3d scene geometry to human workspace. In CVPR 2011, pages 1961–1968. IEEE, 2011. 2
- [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 3
- [19] Hyeonho Jeong, Gihyun Kwon, and Jong Chul Ye. Zero-shot generation of coherent storybook from plain text story using diffusion models. arXiv preprint arXiv:2302.03900, 2023. 6
- [20] Yifan Jiang, Hao Tang, Jen-Hao Rick Chang, Liangchen Song, Zhangyang Wang, and Liangliang Cao. Efficient3dim: Learning a generalizable single-image novel-view synthesizer in one day. arXiv preprint arXiv:2310.03015,

2023. 5

- [21] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. Advances in neural information processing systems, 33:12104–12114, 2020. 1
- [22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 6
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 2, 3
- [24] Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A Efros, and Krishna Kumar Singh. Putting people in their place: Affordance-aware human insertion into scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17089–17099, 2023. 2, 3, 6, 1
- [25] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 3
- [26] Sijia Li, Chen Chen, and Haonan Lu. Moecontroller: Instruction-based arbitrary image manipulation with mixture-of-expert controllers. arXiv preprint arXiv:2309.04372, 2023. 2
- [27] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. Dreamedit: Subject-driven image editing. arXiv preprint arXiv:2306.12624, 2023. 2, 3

- [28] Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, and Jiaya Jia. Mat: Mask-aware transformer for large hole image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10758– 10768, 2022. 3
- [29] Xueting Li, Sifei Liu, Kihwan Kim, Xiaolong Wang, MingHsuan Yang, and Jan Kautz. Putting humans in a scene: Learning affordance in 3d indoor environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12368–12376, 2019. 2
- [30] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 3, 6, 7, 4
- [31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 3
- [32] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 4
- [33] Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu, Ziwei Liu, and Sergey Tulyakov. Hyperhuman: Hyper-realistic human generation with latent structural diffusion. arXiv preprint arXiv:2310.08579, 2023. 3, 5
- [34] Lingxiao Lu, Bo Zhang, and Li Niu. Dreamcom: Finetuning text-guided inpainting model for image composition. arXiv preprint arXiv:2309.15508, 2023. 3
- [35] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free cross-domain image composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2294–2305, 2023. 2, 6
- [36] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11461–11471, 2022. 3
- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5
- [38] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7
- [39] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 6

- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5
- [41] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159,

2024. 4

- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 6, 4
- [43] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 3
- [44] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Object compositing with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18310–18319, 2023. 3, 5, 7
- [45] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161,

2021. 4

- [46] Brandon Trabucco, Kyle Doherty, Max Gurinas, and Ruslan Salakhutdinov. Effective data augmentation with diffusion models. arXiv preprint arXiv:2302.07944, 2023. 3
- [47] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi PontTuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18359– 18369, 2023. 2
- [48] Xiaolong Wang, Rohit Girdhar, and Abhinav Gupta. Binge watching: Scaling affordance learning from sitcoms. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2596–2605, 2017. 2
- [49] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391,

2023. 3, 6, 7, 4

- [50] Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike, et al. Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation. Advances in Neural Information Processing Systems, 36, 2024. 2

- [51] Bangpeng Yao and Li Fei-Fei. Modeling mutual context of object and human pose in human-object interaction activities. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 17–24. IEEE,

2010. 2

- [52] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5505–5514,

2018. 3

- [53] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514,

2023. 5

- [54] Yuke Zhu, Alireza Fathi, and Li Fei-Fei. Reasoning about object affordances in a knowledge base representation. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part II 13, pages 408–424. Springer, 2014. 2

## Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion Supplementary Material

### A. More Implementation Details

##### A.1. SAM-FB Dataset

Figure 10a shows the object masks before and after the data quality control. We see that with our designed data quality control, the foreground object masks have better quality. Figure 10b illustrates the word cloud of our SAMFB dataset, we observe that our SAM-FB dataset consists of diverse object categories.

##### A.2. Traning Details

We constructed our dual-stream UNet based on the Stable Diffusion Inpainting v1.5 model, incorporating several modifications. Specifically, we replicated both the first down-sampling block (including conv_in and the first Down Block) and the last up-sampling block (including the last Up Block and conv_out) to accommodate dualstream inputs and outputs. These independent blocks serve as the expertise input-output branch. Skip connection is also performed between the corresponding expertise input and output branches. To bypass the need to start from scratch, we initialized our model using the pre-trained Stable Diffusion Inpainting v1.5 checkpoint for the unchanged blocks. Fine-tuning is resource-intensive. To optimize our computing resources, we employed a gradual scaling-up approach to train our model. Initially, the model was trained at a resolution of 128×128, with a batch size of 1024 and a learning rate of 1.25 × 10−4 for 35K steps on 2 A-100 GPUs. This phase included 5,000 warm-up steps, followed by a constant schedule. We then fine-tuned the model at a higher resolution of 256 × 256, reducing the batch size to 256 and adjusting the learning rate to 5 × 10−5. For the diffusion process, the time step is 1000 with a linear noise scheduler. We use 2048 samples of SAM-FB for testing and the rest for training. For each foreground object, all different position prompts p can be generated with the mask s. We apply classifier-free guidance [17], dropping all conditions with 0.1 probability. To ensure a fair comparison with other methods, we re-implemented and re-trained the closest method, Human Affordance, since the authors did not release either the model weights or the full dataset they used. We adopted the diffusion model architecture provided by the authors and trained it using our SAM-FB dataset. Additionally, we replaced the mask input in the original model with our position map to ensure compatibility with the SAM-FB dataset.

##### A.3. Data Augmentation

To prevent the model from learning a copy-and-paste process, we introduce different augmentations for the background images, foreground images, and position prompts.

Background Augmentation Given that the images in the SA-1B dataset are not square, we rescale and crop them focusing on the objects. Our first step is to resize each source image so that its shorter edge measures 256 pixels. Following this, we randomly center a 256 × 256 bounding box around each valid object mask, randomly chosen to provide varied backgrounds for the same object. This technique introduces slight background differences for identical objects. Additionally, we allow the bounding boxes to partially crop the objects. This strategy is specifically designed to enable the model to learn from scenarios where objects are partially obscured by the edges of the image.

Foreground Augmentation To prevent the model from simply copying and pasting foreground objects and to increase the diversity of foreground objects, data augmentation is necessary. We used geometric and color augmentations based on Kulal’s work [24] and StyleGAN-ADA [21]. Geometric augmentations included isotropic scaling, rotation, anisotropic scaling, and cutout, each with a probability of 0.4, 0.4, 0.2, and 0.2 respectively. Color augmentations included brightness, contrast, saturation, image-space filtering, and additive noise, each with a probability of 0.2.

Position Prompt Augmentation Our method requires the model to perceptively adjust the position of the inserted object in response to an ambiguous position prompt; therefore, it is also necessary to augment the position prompts. For points, we perform random jittering to deviate them from their original positions. For bounding boxes, we randomly enlarge each box. We adopt mask enlarging and feathering for mask prompts.

### B. Evaluation B.1. Evaluation metrics

Our method aims to establish a reasonable relationship between foreground objects and background scenes while maintaining the appearance of the object similar to a reference image and generating a high-quality synthetic image. To assess these two capabilities, we employ two metrics. Firstly, we use the FID score, which is widely used to measure the harmony of images obtained by generative models, to evaluate the quality of synthetic images. We use a pretrained Inception model to extract features and calculate the FID score between generated images and ground truth im-

[Figure 157]

[Figure 158]

(a) Examples for foreground quality control (b) Word cloud of foreground categories

- Figure 10. 10a shows the candidate foreground samples in the pipeline. The upper row shows four low-quality samples. The lower row shows the samples after data quality control. 10b shows the word cloud of foreground categories in the SAM-FB dataset.

FID↓ CLIP Score Method

mask bbox point null Avg. mask bbox point null Avg.

Baseline 16.05 17.31 18.73 21.18 18.32 0.8209 0.8254 0.7768 0.7509 0.7935 +Dual Diffusion 13.91 14.12 14.58 14.92 14.38 0.8582 0.8578 0.8435 0.8032 0.8407 +Expertise branch 13.53 13.60 13.66 13.96 13.69 0.8727 0.8658 0.8567 0.8034 0.8497

Table 7. Experimental results on SAM-FB test set. The difference between the four kinds of prompts indicates that the performance will be better with a more precise position prompt.

ages on the SAM-FB test set. Secondly, we evaluate the appearance of the inserted object and the reference foreground image using the CLIP score. We use a CLIP image encoder to extract features. The CLIP score is calculated as the cosine similarity between the two features.

##### B.2. Human Evaluation

When comparing the results generated by our methods and baseline methods, we asked user to evaluate the quality of the generated image following these five criteria:

- • Foreground and Background Integration: How naturally does the inserted foreground blend with the background? Does it look out of place or integrate seamlessly?
- • Foreground Clarity and Detail: Assess the clarity, detail, and resolution of the inserted foreground.
- • Foreground Appearance Consistency with Reference: Check if the inserted foreground’s appearance (shape, texture) matches the foreground in the reference image.
- • Lighting and shade on Foreground: Evaluate whether highlights and shades on the inserted foreground are realistic and consistent with the background lighting.
- • Color Consistency: Assess the overall color harmony. Do the inserted foreground and background tones, hues, and saturation levels align?

We conducted our user survey by asking 10 users to rank outputs from different model in 10 affordance insertion settings, therefore gaining 100 data points.

- B.3. Effect of Inpaintings

Figure 13 shows results if the inserted location is away from masked region. It illustrates that the effects of inpainting artifacts are limited.

- C. More Results

- C.1. Details Maintaining

Using DINOv2 features, the model preserves the appearance more effectively than other image editing models, especially for objects with detailed textures. Figure 11 compares our model with other image editing models. Our model demonstrates the ability to retain the details of an object’s appearance, even texture on the object.

- C.2. SAM-FB Test Split

Figure 14 shows a visualization result with different baselines on SAM-FB test split. Our model generated more authentic results compared to other baseline models.

- C.3. Detailed Ablations

We show the detailed ablation results on different types of position prompts and the result is presented in Table 7. With all the designs, the model achieves the lowest FID score and the highest CLIP score on average.

- C.4. Training Datasets

To illustrate SAM-FB helps the model on affordance insertion task, we retrained our model on 50k COCO2014 and

Background Scene

Detailed Object Stable Diffusion GLI-GEN PBE Ours

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

|[Figure 163]|
|---|

[Figure 164]

| |
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]<br><br>|[Figure 168]|
|---|
|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

- Figure 11. Example of objects with details. Our model could keep the appearance better even with some details compared with SD [42], GLI-GEN [30] and PBE [49]. The first row demonstrates the ability to keep some image texture, and the second row illustrates the ability to keep text texture.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Foreground and Background Integration Color Consistency Foreground Consistency with Reference

Lighting on Foreground Foreground Clarity and Details

44.0% 44.0%

68.0%

48.0% 46.0%

28.0%

24.0%

30.0%

18.0%

20.0%

10.0%

34.0%

14.0%

32.0%

14.0%

- Figure 12. Rank-1 distribution for each criterion. Each pie chart represents the proportion of times each model achieved Rank-1 for a specific evaluation criterion. Our method dominates every metric.

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

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

Imprecise Input Anydoor Ours Imprecise Input Anydoor Ours

[Figure 194]

[Figure 195]

[Figure 196]

In-the-wild Image After Inpainting Composition Result

[Figure 197]

[Figure 198]

|[Figure 199]|
|---|

[Figure 200]

[Figure 201]

[Figure 202]

|[Figure 203]|
|---|

In-the-wild Image After Inpainting Composition Result

- Figure 13. Affordance insertion away from inpainted region Training Datasets FID (↓) CLIP (↑)

Composition Input GLIGEN

###### Ours

[Figure 204]

[Figure 205]

[Figure 206]

|[Figure 207]|
|---|

Figure 2. Example of uncommon objects

SD Inpainting ObjectStith PBE

[Figure 208]

[Figure 209]

[Figure 210]

Figure 14. Samples on SAM-FB test split. Our model inserted the bag with authentic appearance.

formed affordance insertion task providing a bounding box on both common objects like apple and uncommon objects like cruta. Our methods generated the images with highest quality, authentic to the reference foreground with proper lighting. In the last row, we show more affordance insertion results when providing ambiguous prompts. The model will automatically find the proper affordance relationship and adjust the location and view of the foreground object. It’s notable that in the first case, when inserting the cruise, it is actually the view from the back of the reference image.

COCO 31.17 0.75 SAM-FB 28.45 0.77

Table 8. Ablation study results on SAM-FB test set.

SAM-FB subset, each for 10k steps, then evaluated on PascalVOC as in-the-wild benchmark. Table 8 shows SAM-FB lead to lower FID and higher CLIP score.

##### C.5. In-the-wild Generalization

##### C.6. Comparison with Anydoor

We compare our methods with other baseline models on in-the-wild images, and Figure 15 shows the visualization results for comparison. In the first five rows, we per-

Table 9 reports quantitative results on SAM-FB and Rank from Human Evaluation (RfHE) on in-the-wild im-

Background Scene Uncommon Object Stable Diffusion XL GLI-GEN PBE Ours

ObjectStitch

|[Figure 211]|
|---|

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

|[Figure 218]|
|---|

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

|[Figure 225]|
|---|

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

`

|[Figure 232]|
|---|

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

|[Figure 245]|
|---|

|[Figure 246]|
|---|

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

[Figure 254]

[Figure 255]

point affordance insertion affordance insertion null affordance insertion affordance insertion

Figure 15. Example of in-the-wild insertion results with details. Our model could keep the appearance better and adjust the foreground’s properties better compared with SD [42], GLI-GEN [30] and PBE [49] on common objects. In the last row, the model generated reasonable insertion when provided ambiguous prompts.

|Method|Anydoor Ours<br><br>|
|---|---|
|CLIP Score (↑) DINO Score (↑) RfHE(↓)|0.8209 0.8658 0.667 0.693 2.18 2.21<br><br>|

facts around objects.

- C.7. Video Demo Please refer to the video demo in the attachment.
- C.8. Failure Cases

Table 9. Anydoor comparison

Figure 17 shows some failure cases when using the model to perform affordance insertion. Generally, when prompted null position, it requires the model to search for a

ages from web (Unsplash). Users prefer MADD for more reasonable insertion (Figure 16). Anydoor introduces arti-

Imprecise Input Anydoor Ours

[Figure 256]

[Figure 257]

[Figure 258]

|[Figure 259]|
|---|

[Figure 260]

[Figure 261]

[Figure 262]

|[Figure 263]|
|---|

In-the-wildFigure 16.ImageAffordanceAfterinsertionInpaintingcompare withCompositionAnydoorResult

In-the-wild Image After Inpainting Composition Result

Position Background Foreground Object Predicted

||
|---|

[Figure 265]

[Figure 266]

|[Figure 267]<br><br>Scene|
|---|

[Figure 270]

[Figure 271]

[Figure 272]

|[Figure 273]|
|---|

[Figure 274]

|[Figure 275]|
|---|

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Figure 17. Some failure cases when using our model to perform affordance insertion.

possible position to insert the object. However, if there are already similar objects in the scene e.g., traffic light in the first row, it is easy to mislead the model and end up with inserting nothing. When the background is too complex and the foreground object is too small, such as a vegetable in a supermarket shown in the second row, it is also difficult for the model to insert the object correctly.

