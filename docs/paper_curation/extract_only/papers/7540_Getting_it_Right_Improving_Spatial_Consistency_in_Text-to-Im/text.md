# arXiv:2404.01197v2[cs.CV]6Aug2024

## Getting it Right: Improving Spatial Consistency in Text-to-Image Models

Agneet Chatterjee ⋆1 , Gabriela Ben Melech Stan ⋆2 , Estelle Aflalo2 , Sayak Paul3 , Dhruba Ghosh4 , Tejas Gokhale5 , Ludwig Schmidt4,

Hannaneh Hajishirzi4 , Vasudev Lal2 , Chitta Baral1 , and Yezhou Yang1

1 Arizona State University

- 2 Intel Labs
- 3 Hugging Face

4 University of Washington 5 University of Maryland, Baltimore County

Abstract. One of the key shortcomings in current text-to-image (T2I) models is their inability to consistently generate images which faithfully follow the spatial relationships specified in the text prompt. In this paper, we offer a comprehensive investigation of this limitation, while also developing datasets and methods that support algorithmic solutions to improve spatial reasoning in T2I models. We find that spatial relationships are under-represented in the image descriptions found in current vision-language datasets. To alleviate this data bottleneck, we create SPRIGHT, the first spatially focused, large-scale dataset, by re-captioning 6 million images from 4 widely used vision datasets and through a 3-fold evaluation and analysis pipeline, show that SPRIGHT improves the proportion of spatial relationships in existing datasets. We show the efficacy of SPRIGHT data by showing that using only ∼0.25% of SPRIGHT results in a 22% improvement in generating spatially accurate images while also improving FID and CMMD scores. We also find that training on images containing a larger number of objects leads to substantial improvements in spatial consistency, including state-of-theart results on T2I-CompBench with a spatial score of 0.2133, by finetuning on <500 images. Through a set of controlled experiments and ablations, we document additional findings that could support future work that seeks to understand factors that affect spatial consistency in text-to-image models. Project page : https://spright-t2i.github.io/

Keywords: Text to Image Generation · Spatial Relationships

### 1 Introduction

The development of text-to-image (T2I) diffusion models such as Stable Diffusion [50] and DALL-E 3 [40] has led to the growth of image synthesis frameworks that

⋆ Equal contribution. Correspondence to agneet@asu.edu

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

An older man standing on top of a snow covered slope.

A bathroom that has recently been tiled and grouted

A ferris wheel water buildings boats and lights

Two cats are eating out of different bowls.

In the image, a man is standing in front of a snowy mountain range, taking a picture of the mountains with his cell phone. The mountains are in the background, and they are quite large, towering over the man and the surrounding landscape. The man is relatively small in comparison to the mountains, emphasizing the vastness of the mountain range.

The Ferris wheel is located near a bridge and a large building, with the Ferris wheel being the tallest object in the scene. The Ferris wheel is also situated next to a river, with a boat visible on the water. The city skyline can be seen in the background, with the Ferris wheel towering over the other buildings."

The image shows a bathroom with a toilet and a bucket on the floor. The toilet is located on the left side of the bathroom, and the bucket is on the right side, closer to the center. The bathroom is small, and the toilet is relatively large compared to the bucket, which is smaller in size.

In the image, two cats are standing in front of a bookshelf, with one cat being larger and occupying a larger portion of the frame, while the other cat is smaller and positioned to the right of the larger cat. The bookshelf is filled with various books, and a TV is located above the bookshelf. The cats are standing on the floor, and they are eating from their bowls, which are placed on the floor as well.

A close up view of some food on a plate.

The image features a plate with a piece of chicken and a piece of broccoli. The chicken is on the left side of the plate, and the broccoli is

[Figure 6]

[Figure 7]

[Figure 8]

on the right side. The chicken is larger than the broccoli, and the broccoli is positioned above the chicken.

[Figure 9]

[Figure 10]

A black cat taking a drink out of a bright blue cup.

A single glass of white win on a table with a vase

A stuffed teddy bear is sitting on the corner of a suitcase

The stop light is green at an intersection.

A black cat is eating out of a blue bowl, which is held by a person's hand. The cat is positioned behind the bowl, and the person's hand is on the left side of the image.

A small teddy bear is sitting on top of a suitcase, which is placed next to another suitcase. The teddy bear is positioned between the two suitcases, with one suitcase being larger and the other smaller.

The man is speaking at a meeting in a conference room.

A glass of wine is sitting on a table next to a vase, which is taller than the glass. The glass is in front of a TV, which is located behind the glass

The traffic light is positioned on the left side of the pole, and the bus sign is located on the right side of the pole.

A man is sitting at a table with a cup of coffee in front of him. The microphone is placed in front of him, and there is a chair behind him.

- Fig. 1: We find that existing vision-language datasets do not capture spatial relationships well. To alleviate this shortcoming, we synthetically re-caption ∼6M images with a spatial focus, and create the SPRIGHT (SPatially RIGHT) dataset. Shown above are samples from the COCO Validation Set, where text in red denotes ground-truth captions and text in green are corresponding captions from SPRIGHT.

are able to generate high resolution photo-realistic images. These models have been adopted widely in downstream applications such as video generation [55], image editing [21], robotics [16], and more. Multiple variations of T2I models have also been developed, which vary according to their text encoder [6], priors [48], and inference efficiency [37]. However, a common bottleneck that affects all of these methods is their inability to generate spatially consistent images: that is, given a natural language prompt that describes a spatial relationship, these models are unable to generate images that faithfully adhere to it.

In this paper, we present a holistic approach towards investigating and mitigating this shortcoming through diverse lenses. We develop datasets, efficient training techniques, and explore multiple ablations and analyses to understand the behaviour of T2I models towards prompts that contain spatial relationships.

Our first finding reveals that existing vision-language (VL) datasets lack sufficient representation of spatial relationships. Although frequently used in the English lexicon, we find that spatial words are scarcely found within imagetext pairs of the existing datasets. To alleviate this shortcoming, we create the “SPRIGHT” (SPatially RIGHT) dataset, the first spatially-focused large scale dataset. Specifically, we synthetically re-caption ∼6 million images sourced from

4 widely used datasets, with a spatial focus (Section 3). As shown in Figure

- 1, SPRIGHT captions describe the fine-grained relational and spatial characteristics of an image, whereas human-written ground truth captions fail to do so. Through a 3-fold comprehensive evaluation and analysis of the generated captions, we benchmark the quality of the generated captions and find that SPRIGHT largely improves over existing datasets in its ability to capture spatial relationships. Next, leveraging only ∼0.25% of our dataset, we achieve a 22% improvement on the T2I-CompBench [23] Spatial Score, and a 31.04% and 29.72% improvement in the FID [22] and CMMD scores [24], respectively.

Our second finding reveals that significant performance improvements in spatial consistency of a T2I model can be achieved by fine-tuning on images that contain a large number of objects. We achieve state-of-the-art performance, and improve image fidelity, by fine-tuning on <500 image-caption pairs from SPRIGHT; training only on images that have a large number of objects. As investigated in VISOR [20], models often fail to generate the mentioned objects in a spatial prompt; we posit that by optimizing the model over images which have a large number of objects (and consequently, spatial relationships), we teach it to generate a large number of objects, which positively impacts its spatial consistency. In addition to improving spatial consistency, our model achieves large gains in performance across all aspects of T2I generation; generating correct number of distinct objects, attribute binding and accurate generation in response to complex prompts.

We further demonstrate the impact of SPRIGHT by benchmarking the tradeoffs achieved with long and short spatial captions, as well as spatially focused and general captions. We take the first steps towards discovering layer-wise activation patterns associated with spatial relationships, by examining the representation space of CLIP [46] as a text encoder.

Our contributions and key findings are summarized below:

- – We create SPRIGHT, the first spatially focused, large scale vision-language dataset by re-captioning ∼6 million images from 4 widely used existing datasets.To demonstrate the efficacy of SPRIGHT, we fine-tune baseline Stable Diffusion models on a small subset of our data and achieve performance gains across multiple spatial reasoning benchmarks while improving the corresponding FID and CMMD scores.
- – We achieve state-of-the-art performance on spatial relationships by developing an efficient training methodology; specifically, we optimize over a small number (<500) of images which consists of a large number of objects, and achieve a 41% improvement over our baseline model.
- – Through multiple ablations and analyses, we present our findings related to spatial relationships: the impact of long captions, the trade-off between spatial and general captions, layer-wise activations of the CLIP text encoder, effect of training with negations and improvements over attention maps.

### 2 Related Work

Text-to-image generative models. Since the initial release of Stable Diffusion [50] and DALL-E [49], different classes of T2I models have been developed, all optimized to generate highly realistic images corresponding to complex natural language prompts. Models such as PixArt-Alpha [6], Imagen [51], and ParaDiffusion [56] move away from the CLIP text encoder, and explore traditional language models such as T5 [47] and LLaMA [54] to process text prompts. unCLIP [48] based models have led to multiple methods [30,43] that leverage a CLIP-based prior as part of their diffusion pipeline.

Spatial relationships in T2I models. Benchmarking the failures of T2I models on spatial relationships has been well explored by VISOR [20], T2ICompBench [23], GenEval [17], and DALL-E Eval [9]. Both training-based and test-time adaptations have been developed to specifically improve upon these benchmarks. Control-GPT [62] finetunes a ControlNet [61] model by generating TikZ code representations with GPT-4 and optimizing over grounding tokens to generate images. SpaText [1], GLIGEN [31], and ReCo [58] are training-based methods that introduce additional conditioning in their fine-tuning process to achieve better spatial control for image generation. LLM-Grounded Diffusion [32] is a test-time multi-step method that improves over layout generated LLMs in an iterative manner. Layout Guidance [7] restricts objects to their annotated bounding box locations through refinement of attention maps during inference. LayoutGPT [15] creates an LLM guided initial layout in the form of CSS, and then uses layout-to-image models to create indoor scenes. REVISION [4] leverages 3D rendering engines to generate synthetic images which act as additional guidance during image synthesis for accurate depiction of spatial relationships in the generated image.

Synthetic captions for T2I models. The efficacy of using descriptive and detailed captions has recently been explored by DALL-E 3 [40], PixArt-Alpha [6] and RECAP [53]. DALL-E 3 builds an image captioning module by jointly optimizing over a CLIP and language modeling objective. RECAP fine-tunes an image captioning model (PALI [8]) and reports the advantages of fine-tuning the Stable Diffusion family of models on long, synthetic captions. PixArt-Alpha also re-captions images from the LAION [52] and Segment Anything [26] datasets; however their key focus is to develop descriptive image captions. On the contrary, our goal is to develop captions that explicitly capture the spatial relationships seen in the image.

### 3 The SPRIGHT Dataset

We find that current vision-language (VL) datasets do not contain “enough” relational and spatial relationships. Despite being frequently used in the English

vocabulary 6, words like “left/right”, “above/behind” are scarce in existing VL datasets. This holds for both annotator-provided captions, e.g., COCO [33], and web-scraped alt-text captions, e.g., LAION [52]. We posit that the absence of such phrases is one of the fundamental reasons for the lack of spatial consistency in current text-to-image models. Furthermore, language guidance is now being used to perform mid-level [57,60] and low-level [27,64] computer vision tasks. This motivates us to create the SPRIGHT (SPatially RIGHT) dataset, which explicitly encodes fine-grained relational and spatial information found in images.

#### 3.1 Creating the SPRIGHT Dataset

We re-caption approximately six million images from four existing vision-language datasets, i.e. datasets containing images and their corresponding natural language descriptions:

- – CC-12M [3] : We re-caption a total of 2.3 million images from the CC-12M dataset, filtering out images of resolution less than 768 × 768.
- – Segment Anything (SA) [26] : We select Segment Anything as most images in it encapsulates a large number of objects; i.e. larger number of spatial relationships can be captured from a given image. We re-caption 3.5 million images as part of our re-captioning process. Since SA does not have groundtruth captions, we generate its general captions using the CoCa [59] model.
- – COCO [33] : We re-caption images (∼ 40,000) from the validation set.
- – LAION-Aesthetics7 : We used 50,000 images from LAION-Aesthetics. 8

We use LLaVA-1.5-13B [34] with the following prompt to produce synthetic spatial captions to create the SPRIGHT dataset:

Using 2 sentences, describe the spatial relationships seen in the image. You can use words like left/right, above/below, front/behind, far/near/adjacent, inside/outside. Also describe relative sizes of objects seen in the image.

#### 3.2 Impact of SPRIGHT

Table 1 shows that SPRIGHT enhances the presence of spatial phrases across all relationship types on all the datasets. For 11 relationships, while the groundtruth captions of COCO and LAION only capture 21.05% and 6.03% of relationships, SPRIGHT captures 304.79% and 284.7%, respectively, i.e. each recaptioned COCO image in SPRIGHT has ∼3 spatial phrases. This shows that captions in VL datasets largely lack the presence of spatial relationships, and that SPRIGHT is able to improve upon this shortcoming by almost always capturing

- 6 https://www.oxfordlearnersdictionaries.com/us/wordlists/oxford3000-5000
- 7 https://laion.ai/blog/laion-aesthetics/
- 8 The entire LAION-5B dataset has been recalled for safety review: https://laion. ai/notes/laion-maintenance/. We will release our re-captioning outputs for these images based on the conclusions of this safety review.

- Table 1: Compared to ground truth annotations, SPRIGHT consistently improves the presence of relational and spatial relationships captured in its captions, across diverse images from different datasets.

% of Spatial Phrases

Dataset

left right above below front behind next close far small large COCO 0.16 0.47 0.61 0.15 3.39 1.09 6.17 1.39 0.19 3.28 4.15

+ SPRIGHT 26.80 23.48 21.25 5.93 41.68 21.13 36.98 15.85 1.34 48.55 61.80 CC-12M 0.61 1.45 0.40 0.19 1.40 0.43 0.54 0.94 1.07 1.44 1.44

+ SPRIGHT 24.53 22.36 20.42 6.48 41.23 14.37 22.59 12.9 1.10 43.49 66.74 LAION 0.27 0.75 0.16 0.05 0.83 0.11 0.24 0.67 0.91 1.03 1.01

+ SPRIGHT 24.36 21.7 14.27 4.07 42.92 16.38 26.93 13.05 1.16 49.59 70.27 Segment Anything 0.02 0.07 0.27 0.06 5.79 0.19 3.24 7.51 0.05 0.85 10.58 + SPRIGHT 18.48 15.09 23.75 6.56 43.5 13.58 33.02 11.9 1.25 52.19 80.22

COCO Original Captions

|[Figure 11]|
|---|
|[Figure 12]|

|[Figure 13]|
|---|

SPRIGHT Captions

- Fig. 2: Compared to ground truth COCO captions,(Left) Word cloud representations showing that SPRIGHT captions significantly amplify the presence of spatial relationships. (Right)SPRIGHT captions also capture a higher number of object occurances.

spatial relationships in every sentence. Our captions offer several improvements beyond the spatial aspects: (i) As depicted in Table 2 we improve the overall linguistic quality compared to the original captions, and (ii) we identify more objects and amplify their occurrences as illustrated in Figure 2; where we plot the top 10 objects present in the original COCO Captions and find that we significantly upsample their corresponding presence in SPRIGHT.

- 3.3 Dataset Validation We perform 3 levels of evaluation to validate the SPRIGHT captions:

1. FAITHScore. Following [25], we leverage a large language model to deconstruct generated captions into atomic (simple) claims that can be individually

- Table 2: In addition to improving the presence of spatial relationships, SPRIGHT enhances linguistic diversity of captions in comparison to their original versions.

Average / caption Nouns Adjectives Verbs Tokens

Dataset

COCO → COCO+SPRIGHT 3.00 → 14.31 0.83 → 3.82 0.04 → 0.15 11.28 → 68.22 CC-12M → CC-12M+SPRIGHT 3.35 → 13.99 1.36 → 4.36 0.26 → 0.16 22.93 → 67.41 LAION → LAION+SPRIGHT 1.78 → 14.32 0.70 → 4.53 0.11 → 0.14 12.49 → 69.74 SA → SA+SPRIGHT 3.10 → 13.42 0.79 → 4.65 0.01 → 0.12 09.88 → 63.90

and independently verified in a Visual Question Answering (VQA) format. We randomly sample 40,000 image-generated caption pairs from our dataset, and prompt GPT-3.5-Turbo to identify descriptive phrases (as opposed to subjective analysis that cannot be verified from the image) and decompose the descriptions into atomic statements. These atomic statements are then passed to LLaVA-

- 1.5-13B for verification, and correctness is aggregated over 5 categories: entity, relation, colors, counting, and other attributes. We also measure correctness on spatial-related atomic statements, i.e., those containing one of the keywords left/right, above/below, near/far, large/small and background/foreground. The captions are on average 88.9% correct, with spatially-focused relations, being 83.6% correct; with the detailed breakdown presented in the Supplementary Materials. Since there is some uncertainty about bias induced by using LLaVA to evaluate LLaVA-generated captions, we also verify the caption quality in other ways, as described next.
- 2. GPT-4 (V). Inspired by recent methods [40,65], we perform a small-scale study on a split of 444 images from LAION and SA (from Section 4.2) to evaluate our captions with GPT-4(V) Turbo [41]. We prompt GPT-4(V) to rate each caption between a score of 1 to 10, especially focusing on the correctness of the spatial relationships captured. Captions of images from LAION and SA had a {mean, median} rating of {7.49,8} and {7.36,8}, respectively. We present the prompt used in the Supplementary Materials.
- 3. Human Annotation. We also annotate a total of 3,000 images through a crowd-sourced human study, where each participant annotates a maximum of 30 image-text pairs. As evidenced by the average number of tokens in Table 1, most captions in SPRIGHT have >1 sentences. Therefore, for fine-grained evaluation, we randomly select 1 sentence, from a caption in SPRIGHT, and evaluate its correctness for a given image. Across 149 responses, we find the metrics to be: correct=1840 and incorrect=928, yielding an accuracy of 66.57%.
- 4 Improving Spatial Consistency

In this section, we leverage SPRIGHT in an effective and efficient manner, and describe methodologies that significantly advance spatial reasoning in T2I mod-

- Table 3: Quantitative metrics across multiple spatial reasoning and image fidelity metrics, demonstrating the effectiveness of high quality spatially-focused captions in SPRIGHT. Green indicates results of the model fine-tuned on SPRIGHT. For FID, we use cfg = 3.0 and 7.0 for the baseline and the fine-tuned model, respectively.

Method OA (%) (↑)

VISOR (%) (↑) T2I-CompBench (↑) Spatial Score

ZS-FID (↓) CMMD (↓) uncond cond 1 2 3 4

SD 2.1 47.83 30.25 63.24 64.42 35.74 16.13 4.70 0.1507 21.646 0.703 + SPRIGHT 53.59 36.00 67.16 66.09 44.02 24.15 9.13 0.1840 14.925 0.494

- Table 4: Across all reported methods, we achieve state-of-the-art performance on the T2I-CompBench Spatial Score. This is achieved by fine-tuning SD 2.1 on 444 imagecaption pairs from the SPRIGHT dataset; where each image has >18 objects.

|# of Objects per Image # of Training Images T2I-CompBench Spatial Score (↑)|<6 <11 11 >11 > 18 444 1346 1346 1346 444 0.1309 0.1468 0.1667 0.1613 0.2133<br><br>|
|---|---|

els. We use Stable Diffusion v2.1 9 as the base model and our training and validation set consists of 13,500 and 1,500 images respectively, randomly sampled in a 50:50 split between LAION-Aesthetics and Segment Anything. Each image is paired with a typical caption and a spatial caption (from SPRIGHT). During fine-tuning, for each image, we randomly choose one of the given caption types in a 50:50 ratio. We fine-tune the U-Net and the CLIP text encoder as part of our training, both with a learning rate 5 × 10−6 optimized by AdamW [36] and a global batch size of 128. While we train the U-Net for 15,000 steps, the CLIP text encoder remains frozen during the first 10,000 steps. We develop our code-base on top of the Diffusers library [44].

#### 4.1 Improving upon Baseline Methods

We present results on the spatial relationship benchmarks (VISOR [20], T2ICompBench [23]) and image fidelity metrics in Table 3. To account for the inconsistencies associated with FID [10,42], we also report results on CMMD [24]. Across all metrics, our method significantly improves upon the base model by fine-tuning on <15k images. We conclude that the dense, spatially focused captions in SPRIGHT provide effective spatial guidance to T2I models, and alleviate the need to scale up fine-tuning on a large number of images. As shown in Figure 3, the model captures complex spatial relationships (top right), relative sizes (large) and patterns (swirling).

#### 4.2 Efficient Training Methodology

We devise an additional efficient training methodology, which achieves state-ofthe-art performance on the spatial aspect of the T2I-CompBench Benchmark. We

9 https://huggingface.co/stabilityai/stable-diffusion-2-1

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

A vast, sandy desert stretches across the scene, with a giant cactus towering on the left, about twice the height of a nearby small, abandoned wooden wagon on the right. A vibrant sunset colors the sky in the background.

Within a mystical realm, a castle perches atop a steep cliff. To the left of the castle, a winding staircase leads down to a hidden beach, while to the right, a dense forest stretches as far as the eye can see. In front of the castle, a drawbridge spans a deep chasm

A vibrant coral reef occupies the bottom half of the image, with a large sea turtle swimming above it towards the right. In the distant background, a small school of ﬁsh forms a swirling pattern, with the sunlight ﬁltering through the water from the top left corner, illuminating the scene.

A cozy cabin nestled in the woods, with a stream ﬂowing in front and a ﬁre burning in the ﬁreplace inside.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

In the foreground, a large, intricately detailed grandfather clock towers over a small, antique wooden chair to its right. The background

A large, full moon dominates the top right corner of the image, casting a soft glow on a small, abandoned house below, situated in the center of a barren ﬁeld with a twisted, gnarled tree in the foreground

On the left side of a tranquil meadow, a towering oak tree casts its shade over a small pond, while a family of deer grazes peacefully nearby.

A person standing on a hill, with a rainbow stretching across the sky behind them and a valley spreading out below.

reveals a dimly lit library, shelves stocked with books of varying sizes, some towering higher than others, creating a maze of knowledge.

- Fig. 3: Generated images from our model, as described in Section 4.1, on prompts which contain multiple objects and complex spatial relationships. We curate these prompts from ChatGPT.

hypothesize that (a) images that capture a large number of objects inherently also contain multiple spatial relationships; and (b) training on these kinds of images will optimize the model to consistently generate a large number of objects, given a prompt containing spatial relationships; a well-documented failure mode of current T2I models [20].

For our dataset of <15k images the median # of objects/image = 11. We partition our dataset into multiple subsets based on the maximum number of objects present in an image. This partitioning is automated using the openworld image tagging model Recognize Anything [63]. We create five subsets, train corresponding models on a single subset and benchmark them in Table 4. We keep the same hyper-parameters as before, only initiating training of the CLIP Text Encoder from the beginning. With an increase in the # of objects / image, iterative improvement in spatial fidelity is observed, with the best score for the subset containing greater than 18 objects.

Our major finding is that, with 444 training images and spatial captions from SPRIGHT, we achieve a 41% improvement over the baseline SD 2.1 and attain state-of-the-art performance across all reported models on the T2I-CompBench spatial score. In Table 5, compared to SD 2.1, we significantly improve all aspects of the VISOR score, while also enhancing the ZS-FID and CMMD scores

- Table 5: Comparing baseline SD 2.1 with our state-of-the-art model, across multiple spatial reasoning and image fidelity metrics, as described in Section 4.2. Green indicates results from our model. For FID, we use cfg = 3.0 and 7.5 for the baseline model and our model, respectively

Method OA (%) (↑)

VISOR (%) (↑) T2I-CompBench (↑) Spatial Score

ZS-FID (↓) CMMD (↓) uncond cond 1 2 3 4

SD 2.1 47.83 30.25 63.24 64.42 35.74 16.13 4.70 0.1507 21.646 0.703

+ SPRIGHT (<500 images)

60.68 43.23 71.24 71.78 51.88 33.09 16.15 0.2133 16.149 0.512

- Table 6: Results on the VISOR Benchmark. Our model outperforms existing methods, on all aspects related to spatial relationships, consistently generating spatially accurate images as shown by the high VISOR [1-4] values.

VISOR (%) uncond cond 1 2 3 4

Method OA (%)

GLIDE [39] 3.36 1.98 59.06 6.72 1.02 0.17 0.03 GLIDE + CDM [35] 10.17 6.43 63.21 20.07 4.69 0.83 0.11 CogView2 [12] 18.47 12.17 65.89 33.47 11.43 3.22 0.57 DALLE-mini [11] 27.10 16.17 59.67 38.31 17.50 6.89 1.96 DALLE-2 [48] 63.93 37.89 59.27 73.59 47.23 23.26 7.49 Structured Diffusion [14] 28.65 17.87 62.36 44.70 18.73 6.57 1.46 Attend-and-Excite [5] 42.07 25.75 61.21 49.29 19.33 4.56 0.08

Ours (<500 images) 60.68 43.23 71.24 71.78 51.88 33.09 16.15

on COCO-30K images by 25.39% and 27.16%, respectively. Our key findings on VISOR (Table 6) include: (a) a 26.86% increase in the Object Accuracy (OA) score, indicating substantial gains in generating objects mentioned in the input prompt, and (b) a VISOR4 score of 16.15%, demonstrating our model’s consistent generation of spatially accurate images.

We also compare our model’s performance on the GenEval [17] benchmark (Table 7), and find that in addition to improving spatial relationship (see Position), our model shows improvement in generating 1 and 2 objects, along with the correct number of objects. Throughout our experiments, our training approach not only preserves but also enhances the non-spatial aspects associated with a text-to-image model. Additional results and illustrations from VISOR and T2I-CompBench are provided in the Supplementary Materials.

### 5 Ablation Studies and Analyses

To fully ascertain the impact of spatially-focused captions in SPRIGHT, we experiment with multiple nuances of our dataset and the corresponding T2I pipeline. Unless stated otherwise, the experimental setup identical to Section 4.

- Table 7: Results on the GenEval Benchmark. In addition to spatial relationships, we also improve model performance in generating the correct number of objects.

Method Overall

Single object

Two objects

Counting Colors Position Attribute binding

CLIP retrieval [2] 0.35 0.89 0.22 0.37 0.62 0.03 0.00 minDALL-E [29] 0.23 0.73 0.11 0.12 0.37 0.02 0.01

- SD 1.5 0.43 0.97 0.38 0.35 0.76 0.04 0.06
- SD 2.1 0.50 0.98 0.51 0.44 0.85 0.07 0.17 SDXL [45] 0.55 0.98 0.74 0.39 0.85 0.15 0.23 PixArt-Alpha [6] 0.48 0.98 0.50 0.44 0.80 0.08 0.07 Ours (<500 images) 0.51 0.99 0.59 0.49 0.85 0.11 0.15

- Table 8: Comparing (a) the effect the percentage of spatial captions and (b) the effect of long and short spatial captions.

T2I-CompBench Spatial Score (↑)

T2I-CompBench Spatial Score

Model, Setup

% of spatial captions

(↑)

Long Captions Short Captions

25 0.154 50 0.178 75 0.161

- SD 1.5, w/o CLIP FT 0.0910 0.0708

- SD 2.1, w/o CLIP FT 0.1605 0.1420 SD 2.1, w/ CLIP FT 0.1777 0.1230

100 0.140

(a) T2I-CompBench Spatial Scores for models trained on varying ratios of spatial captions. Fine-tuning on a ratio of 50% and 75% of spatial captions yields optimal results.

(b) T2I-CompBench Spatial Scores for models trained on long and short spatial captions. Across multiple setups, we find that longer spatial captions lead to better improvements in spatial consistency.

#### 5.1 Optimal Ratio of Spatial Captions

To understand the impact of spatially focused captions in comparison to groundtruth captions, we fine-tune different models by varying the % of spatial captions. The results suggest that the model trained on 50% spatial captions achieves the best spatial scores on T2I-CompBench (Table 8 (a)). The models trained on only 25% of spatial captions suffer largely from incorrect spatial relationships whereas the model trained only on spatial captions fails to generate the mentioned objects in the input prompt. Figure 4 shows illustrative examples.

#### 5.2 Impact of Long and Short Spatial Captions

We also compare the effect of fine-tuning with shorter and longer variants of spatial captions. We create the shorter variants by randomly sampling 1 sentence from the longer caption, and fine-tune multiple models, with different setups. Across, all setups, (Table 8 (b)) longer captions perform better than their shorter counterparts. In fact, CLIP fine-tuning hurts performance while using shorter captions, but has a positive impact on longer captions. This potentially happens because fine-tuning CLIP enables T2I models to generalize better to longer captions, which are out-of-distribution at the onset of training as they are initially pre-trained on short(er) captions from datasets such as LAION.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

|25|
|---|

25

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

|50|
|---|

50

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|75|
|---|

75

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|100|
|---|

100

A bee on the left of a refridgerator

A rabbit near a bicycle

A cup on the right of a bee A clock on the top of a sheep A bowl on the bottom of a frog

- Fig. 4: Illustrative comparisons between models trained on varying ratio of spatial experiments. Models trained on 50% and 75% spatial captions are optimal.

#### 5.3 Investigating the CLIP Text Encoder

The CLIP Text Encoder enables semantic understanding of the input text prompts in the Stable Diffusion model. As we fine-tune CLIP on the spatial captions, we investigate the various nuances associated with it:

Centered Kernel Alignment (CKA) [28,38] compares layer-wise representations learned by two neural networks. Figure 5 illustrates different representations learned by baseline CLIP, compared against the one trained on SPRIGHT. We compare layer activations across 50 simple and complex prompts and aggregate representations from all the layers. Our findings reveal that the MLP and output attention projection layers play a larger role in enhancing spatial comprehension, as opposed to layers such as the layer norm. This distinction is larger with complex prompts, showing that the longer prompts from SPRIGHT indeed lead to more diverse embeddings being learned within the CLIP space.

Improving Semantic Understanding : To evaluate semantic understanding of the fine-tuned CLIP, we perform the following experiment: given a prompt containing a spatial phrase and 2 objects, we modify the prompt by switching

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

ShortPromptsLongPrompts

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

mlp_fc_1 attn_out_proj q_projection layer_norm_1

- Fig. 5: Comparison of layer-wise representations between Baseline CLIP (X-axis) and fine-tuned CLIP on SPRIGHT (Y-axis). Spatial captions show distinct representations in output attention projections and MLP layers, while layer norm layers are more similar. The representation gap widens with long, complex prompts, suggesting spatial prompts in SPRIGHT create diverse embeddings.

- Table 9: CLIP fine-tuned on SPRIGHT is able to differentiate the spatial nuances present in a textual prompt. While Baseline CLIP shows a high similarity for spatially different prompts, SPRIGHT enables better fine-grained understanding.

“above” “below” “to the left of” “to the right of” “in front of” “behind”

Baseline CLIP 0.9225 0.9259 0.9229 0.9223 0.9231 0.9289 CLIP + SPRIGHT 0.8674 0.8673 0.8658 0.8528 0.8417 0.8713

the objects (e.g. “an airplane above an apple” → “an apple above an airplane”). Although these sentences have the same words, the placement of the two nouns relative to the preposition “above” completely changes the meaning of the sentence. To evaluate if models can discern this spatial distinction, we compute the cosine similarity between the pooled layer outputs of the original and modified prompts, for ∼ 37k sentences. Table 9 shows that CLIP finetuned on SPRIGHT is able to differentiate between the prompts better (i.e. lower cosine similarity) than the baseline.

#### 5.4 Improvement over Attention Maps

Inspired by methods like Attend-and-Excite [5], we visualize attention relevancy maps for both simple and complex spatial prompts. Our model better generates the expected objects and achieves improved spatial localization compared to the baseline. For instance, the baseline models fails to generate objects like the bed and house, which our model successfully generates. The relevancy map indicates that high attention patches for missing words are spread across the image. Additionally, our model correctly attends to spatial words in the image, unlike the baseline. For example, in our model (Figure 6, bottom row), below

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

desk below bed house right road forest

a house on the rightside of a long road that traverses through the forest

a desk belowa bed

- Fig. 6: Visualising the cross-attention relevancy maps for baseline (top row) and finetuned model (bottom row) on SPRIGHT. Images in red are from baseline model while images in green are from our model.

attends to patches below the bed, and right attends to patches on the road’s right, while Stable Diffusion 2.1 does not. We achieve these improvements across the intermediate attention maps and the final generated images.

#### 5.5 Training with Negation

Dealing with negation remains a challenge for multimodal models as reported by previous findings on Visual Question Answering and Reasoning [13,18,19]. Thus, in this section, we investigate the ability of T2I models to reason over spatial relationships and negations, simultaneously. Specifically, we study the impact of training a model with “A man is not to the left of a dog” as a substitute to “A man is to the right of a dog”. To create such captions, we post-process our generated captions and randomly replace spatial occurrences with their negation counter-parts, and ensure that the semantic meaning of the sentence remains unchanged. Training on such a model, we find slight improvements in the spatial score, both while evaluating on prompts containing only negation (0.069 > 0.066) and those that contain a mix of negation and simple statements (0.1427 > 0.1376). There is however, a significant drop in performance, when evaluating on prompts that only contain negation; thus highlighting

- a major scope of improvement in this regard.

### 6 Conclusion

In this work, we present findings and techniques that enable improvement of spatial relationships in text-to-image models. We develop a large-scale dataset, SPRIGHT that captures fine-grained spatial relationships across a diverse set of images. Leveraging SPRIGHT, we develop efficient training techniques and achieve state-of-the art performance in generating spatially accurate images. We thoroughly explore various aspects concerning spatial relationships and evaluate the range of diversity introduced by the SPRIGHT dataset. We leave further scaling studies related to spatial consistency as future work. We believe our findings

and results facilitate a comprehensive understanding of the interplay between spatial relationships and T2I models, and contribute to the future development of robust vision-language models.

### Acknowledgements

We thank Lucain Pouget for helping us in uploading the dataset to the Hugging Face Hub and the Hugging Face team for providing computing resources to host our demo. The authors acknowledge resources and support from the Research Computing facilities at Arizona State University. AC, CB, YY were supported by NSF Robust Intelligence program grants #1750082 and #2132724. TG was supported by Microsoft’s Accelerating Foundation Model Research (AFMR) program and UMBC’s Strategic Award for Research Transitions (START). The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers.

### References

- 1. Avrahami, O., Hayes, T., Gafni, O., Gupta, S., Taigman, Y., Parikh, D., Lischinski, D., Fried, O., Yin, X.: Spatext: Spatio-textual representation for controllable image generation. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE (Jun 2023). https://doi.org/10.1109/cvpr52729. 2023.01762, http://dx.doi.org/10.1109/CVPR52729.2023.01762
- 2. Beaumont, R.: Clip retrieval: Easily compute clip embeddings and build a clip retrieval system with them. https://github.com/rom1504/clip-retrieval (2022)
- 3. Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3558–3568 (2021)
- 4. Chatterjee, A., Luo, Y., Gokhale, T., Yang, Y., Baral, C.: Revision: Rendering tools enable spatial fidelity in vision-language models (2024), https://arxiv.org/abs/ 2408.02231
- 5. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42(4), 1–10 (2023)
- 6. Chen, J., Jincheng, Y., Chongjian, G., Yao, L., Xie, E., Wang, Z., Kwok, J., Luo, P., Lu, H., Li, Z.: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In: The Twelfth International Conference on Learning Representations (2023)
- 7. Chen, M., Laina, I., Vedaldi, A.: Training-free layout control with cross-attention guidance. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5343–5353 (2024)
- 8. Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., et al.: Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794 (2022)
- 9. Cho, J., Zala, A., Bansal, M.: Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3043–3054 (2023)

- 10. Chong, M.J., Forsyth, D.: Effectively unbiased fid and inception score and where to find them. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6070–6079 (2020)
- 11. Dayma, B., Patil, S., Cuenca, P., Saifullah, K., Abraham, T., Le Khac, P., Melas, L., Ghosh, R.: Dall·e mini (7 2021). https://doi.org/10.5281/zenodo.5146400, https://github.com/borisdayma/dalle-mini
- 12. Ding, M., Zheng, W., Hong, W., Tang, J.: Cogview2: Faster and better text-toimage generation via hierarchical transformers. Advances in Neural Information Processing Systems 35, 16890–16902 (2022)
- 13. Dobreva, R., Keller, F.: Investigating negation in pre-trained vision-and-language models. In: Bastings, J., Belinkov, Y., Dupoux, E., Giulianelli, M., Hupkes, D., Pinter, Y., Sajjad, H. (eds.) Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP. pp. 350–362. Association for Computational Linguistics, Punta Cana, Dominican Republic (Nov 2021). https: //doi.org/10.18653/v1/2021.blackboxnlp-1.27, https://aclanthology.org/ 2021.blackboxnlp-1.27
- 14. Feng, W., He, X., Fu, T.J., Jampani, V., Akula, A., Narayana, P., Basu, S., Wang, X.E., Wang, W.Y.: Training-free structured diffusion guidance for compositional text-to-image synthesis (2023)
- 15. Feng, W., Zhu, W., Fu, T.j., Jampani, V., Akula, A., He, X., Basu, S., Wang, X.E., Wang, W.Y.: Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems 36 (2024)
- 16. Gao, J., Hu, K., Xu, G., Xu, H.: Can pre-trained text-to-image models generate visual goals for reinforcement learning? Advances in Neural Information Processing Systems 36 (2024)
- 17. Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. In: Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2023), https: //openreview.net/forum?id=Wbr51vK331
- 18. Gokhale, T., Banerjee, P., Baral, C., Yang, Y.: Vqa-lol: Visual question answering under the lens of logic. In: European conference on computer vision. pp. 379–396. Springer (2020)
- 19. Gokhale, T., Chaudhary, A., Banerjee, P., Baral, C., Yang, Y.: Semantically distributed robust optimization for vision-and-language inference. In: Muresan, S., Nakov, P., Villavicencio, A. (eds.) Findings of the Association for Computational Linguistics: ACL 2022. pp. 1493–1513. Association for Computational Linguistics, Dublin, Ireland (May 2022). https://doi.org/10.18653/v1/2022.findingsacl.118, https://aclanthology.org/2022.findings-acl.118
- 20. Gokhale, T., Palangi, H., Nushi, B., Vineet, V., Horvitz, E., Kamar, E., Baral, C., Yang, Y.: Benchmarking spatial relationships in text-to-image generation. arXiv preprint arXiv:2212.10015 (2022)
- 21. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-or, D.: Prompt-to-prompt image editing with cross-attention control. In: The Eleventh International Conference on Learning Representations (2023), https://openreview. net/forum?id=_CDixzkzeyb
- 22. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)
- 23. Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems 36, 78723–78747 (2023)

- 24. Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., Kumar, S.: Rethinking fid: Towards a better evaluation metric for image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9307–9315 (2024)
- 25. Jing, L., Li, R., Chen, Y., Jia, M., Du, X.: Faithscore: Evaluating hallucinations in large vision-language models. arXiv preprint arXiv:2311.01477 (2023)
- 26. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026

(2023)

- 27. Kondapaneni, N., Marks, M., Knott, M., Guimaraes, R., Perona, P.: Text-image alignment for diffusion-based perception. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13883–13893 (2024)
- 28. Kornblith, S., Norouzi, M., Lee, H., Hinton, G.: Similarity of neural network representations revisited. In: International conference on machine learning. pp. 3519–

3529. PMLR (2019)

- 29. kuprel: min-dalle (2022), https://github.com/kuprel/min-dalle
- 30. Lee, D., Kim, J., Choi, J., Kim, J., Byeon, M., Baek, W., Kim, S.: Karlo-v1.0.alpha on coyo-100m and cc15m. https://github.com/kakaobrain/karlo (2022)
- 31. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22511–22521 (2023)
- 32. Lian, L., Li, B., Yala, A., Darrell, T.: Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. ArXiv preprint abs/2305.13655 (2023), https://arxiv.org/abs/2305.13655
- 33. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
- 34. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26296–26306 (2024)
- 35. Liu, N., Li, S., Du, Y., Torralba, A., Tenenbaum, J.B.: Compositional visual generation with composable diffusion models. In: European Conference on Computer Vision. pp. 423–439. Springer (2022)
- 36. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019), https://openreview.net/forum? id=Bkg6RiCqY7
- 37. Luo, S., Tan, Y., Huang, L., Li, J., Zhao, H.: Latent consistency models: Synthesizing high-resolution images with few-step inference (2023)
- 38. Nguyen, T., Raghu, M., Kornblith, S.: Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth

(2021)

- 39. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models (2022)
- 40. OpenAI: Dalle-3 (2023), https://openai.com/dall-e-3
- 41. OpenAI: Gpt-4(v) (2023), https://cdn.openai.com/papers/GPTV_System_Card. pdf

- 42. Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in gan evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11410–11420 (2022)
- 43. Patel, M., Kim, C., Cheng, S., Baral, C., Yang, Y.: Eclipse: A resource-efficient textto-image prior for image generations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9069–9078 (2024)
- 44. von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., Wolf, T.: Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers

(2022)

- 45. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: SDXL: Improving latent diffusion models for high-resolution image synthesis. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum?id=di52zR8xgf
- 46. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (2021), http://proceedings.mlr.press/v139/ radford21a.html
- 47. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21(140), 1–67 (2020)
- 48. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)
- 49. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event. Proceedings of Machine Learning Research, vol. 139, pp. 8821–8831. PMLR (2021), http://proceedings.mlr.press/v139/ ramesh21a.html
- 50. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 51. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in neural information processing systems 35, 36479–36494 (2022)
- 52. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open largescale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35, 25278–25294 (2022)
- 53. Segalis, E., Valevski, D., Lumen, D., Matias, Y., Leviathan, Y.: A picture is worth a thousand words: Principled recaptioning improves image generation. arXiv preprint arXiv:2310.16656 (2023)
- 54. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C.C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B.,

- Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P.S., Lachaux, M.A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E.M., Subramanian, R., Tan, X.E., Tang, B., Taylor, R., Williams, A., Kuan, J.X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., Scialom, T.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)
- 55. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for textto-video generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7623–7633 (2023)
- 56. Wu, W., Li, Z., He, Y., Shou, M.Z., Shen, C., Cheng, L., Li, Y., Gao, T., Zhang, D., Wang, Z.: Paragraph-to-image generation with information-enriched diffusion model. arXiv preprint arXiv:2311.14284 (2023)
- 57. Xu, M., Zhang, Z., Wei, F., Hu, H., Bai, X.: Side adapter network for openvocabulary semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2945–2954 (June 2023)
- 58. Yang, Z., Wang, J., Gan, Z., Li, L., Lin, K., Wu, C., Duan, N., Liu, Z., Liu, C., Zeng, M., et al.: Reco: Region-controlled text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14246–14255 (2023)
- 59. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research (2022), https://openreview.net/forum?id=Ee277P3AYC
- 60. Yun, S., Park, S.H., Seo, P.H., Shin, J.: Ifseg: Image-free semantic segmentation via vision-language model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2967–2977 (June 2023)
- 61. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023)
- 62. Zhang, T., Zhang, Y., Vineet, V., Joshi, N., Wang, X.: Controllable text-to-image generation with gpt-4. arXiv preprint arXiv:2305.18583 (2023)
- 63. Zhang, Y., Huang, X., Ma, J., Li, Z., Luo, Z., Xie, Y., Qin, Y., Luo, T., Li, Y., Liu, S., et al.: Recognize anything: A strong image tagging model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1724–1732 (2024)
- 64. Zhao, W., Rao, Y., Liu, Z., Liu, B., Zhou, J., Lu, J.: Unleashing text-to-image diffusion models for visual perception. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5729–5739 (2023)
- 65. Zhong, M., Shen, Y., Wang, S., Lu, Y., Jiao, Y., Ouyang, S., Yu, D., Han, J., Chen, W.: Multi-lora composition for image generation. arXiv preprint arXiv:2402.16843

(2024)

## Getting it Right: Improving Spatial Consistency in Text-to-Image Models : Supplementary Material

Agneet Chatterjee ⋆1 , Gabriela Ben Melech Stan ⋆2 , Estelle Aflalo2 , Sayak Paul3 , Dhruba Ghosh4 , Tejas Gokhale5 , Ludwig Schmidt4,

Hannaneh Hajishirzi4 , Vasudev Lal2 , Chitta Baral1 , and Yezhou Yang1

1 Arizona State University

- 2 Intel Labs
- 3 Hugging Face

4 University of Washington 5 University of Maryland, Baltimore County

In this supplementary material, we present additional quantitave and qualitative results from our dataset and method. We discuss fine-grained FaithScore evaluations of the SPRIGHT captions, along with ways to improve the caption quality and its impact on models that support longer token limits. We present the GPT-4 (V) prompt used for evaluation and discuss the limitations of our current work. Lastly, we cover the contributions of each author in this work.

### 1 Results on T2I-CompBench

As shown in Table 1, we achieve state of the art performance on the spatial score in the widely accepted T2I-CompBench benchmark. The significance of training on images containing a large number of objects is emphasized by the enhanced performance of our models across various dimensions in T2I-CompBench. Specifically, we enhance attribute binding parameters such as color and texture, alongside maintaining competitive performance in non-spatial aspects.

### 2 FaithScore Evaluations

Table 2 presents the detailed breakdown of the FaithScore evaluations conducted on the SPRIGHT captions, with the spatially-focused relationships being 83.6% correct, on average.

### 3 CLIP Token Limit

The longer SPRIGHT captions better utilize the CLIP 77-token limit; ground truth and SPRIGHT captions have an average of 14.95 and 81.43 tokens, respectively. Furthermore, T2I models with longer context lengths and multiple

⋆ Equal contribution. Correspondence to agneet@asu.edu

- Table 1: Results on the T2ICompBench Benchmark. a) We achieve state of the art spatial score, across all methods, by efficient fine-tuning on only 444 images.

b) Despite not explicitly optimizing for them, we find substantial improvement and competitive performance on attribute binding and non-spatial aspects.

Method

Attribute Binding Object Relationship Color Shape Texture Spatial Non-Spatial

SD 1.4 0.3765 0.3576 0.4156 0.1246 0.3079 SD 2 0.5065 0.4221 0.4922 0.1342 0.3096 Composable v2 0.4063 0.3299 0.3645 0.0800 0.2980 Structured v2 0.4990 0.4218 0.4900 0.1386 0.3111 Attn-Exct v2 0.6400 0.4517 0.5963 0.1455 0.3109 GORS 0.6603 0.4785 0.6287 0.1815 0.3193 DALLE-2 0.5750 0.5464 0.6374 0.1283 0.3043 SDXL 0.6369 0.5408 0.5637 0.2032 0.3110 PixArt-Alpha 0.6886 0.5582 0.7044 0.2082 0.3179 Kandisnky v2.2 0.5768 0.4999 0.5760 0.1912 0.3132 DALL-E 3 0.8110 0.6750 0.8070 - -

Ours (<500 images) 0.6251 0.4648 0.5920 0.2133 0.3132

- Table 2: FAITHScore caption evaluation of our SPRIGHT dataset. On a sample of 40,000 captions, SPRIGHT obtains an 88.9% accuracy, comparable with the reported 86% and 94% on LLaVA-1k and MSCOCO-Captions, respectively. On the subset of atomic claims about spatial relations, SPRIGHT is correct 83.6% of the time.

Category # Examples Accuracy (%) Overall

— 88.9 FAITHScore

Entities 149,393 91.4 Relations 167,786 85.8 Colors 10,386 83.1 Counting 59,118 94.5 Other 29,661 89.0

Spatial 45,663 83.6

text encoders such as PixArt-Sigma and SD3 can take full advantage of our captions and training technique: we fine-tune PixArt-Sigma (token limit = 300) on SPRIGHT and obtain a spatial score of 0.2501.

### 4 Improvements in Captioning

While our work is to explore the impact of spatially focused captions, we find that improvements in caption quality can be achieved through stronger models like LLaVA-1.6-34B, GPT-4(V) or GPT-4o. To validate this, we conduct a human

study (n=3) on 100 CC-12M images, comparing re-captioning performance of LLaVA-1.5-13B and LLaVA-1.6-34B, and find an improvement from 63% to 78%.

### 5 System Prompt for GPT-4 Evaluation

You are part of a team of bots that evaluates images and their captions. Your job is to come up with a rating between 1 to 10 to evaluate the provided caption for the provided image. Consider the correctness of spatial relationships captured in the provided image. Return the response formatted as a dictionary with two keys: ‘rating’, denoting the numeric rating, and ‘explanation’, denoting a brief justification for the rating.

The captions you are judging are designed to stress-test image captioning programs, and may include:

- 1. Spatial phrases like above, below, left, right, front, behind, background, foreground (focus most on the correctness of these words).
- 2. Relative sizes between objects such as small & large, big & tiny (focus on the correctness of these words).
- 3. Scrambled or misspelled words (the image generator should produce an image associated with the probable meaning). Make a decision as to whether or not the caption is correct, given the image. A few rules:

- 1. It is ok if the caption does not explicitly mention each object in the image; as long as the caption is correct in its entirety, it is fine.
- 2. It is also ok if some captions don’t have spatial relationships; judge them based on their correctness. A caption not containing spatial relationships should not be penalized.
- 3. You will think out loud about your eventual conclusion. Don’t include your reasoning in the final output.
- 4. Return the response formatted as a Python-formatted dictionary having two keys: ‘rating’, denoting the numeric rating, and ‘explanation’, denoting a brief justification for the rating.

### 6 Comparing COCO-30K and Generated Images

In Figure 1, we compare images from COCO, baseline Stable Diffusion and our model. We find that the generated images from our model adhere to the input prompts better, are more photo-realistic in comparison to the baseline.

A garbage bag in a white lighted bathroom

[Figure 58]

A bathroom has a red circular rug by the toilet

[Figure 59]

A black and white cat sits near a window looking outside

A girl with pigtails holding a Two happy dogs have their teddy bear wearing a shirt

head out the car window

SD2.1+SD2.1COCO-30K

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

SPRIGHT

- Fig. 1: Illustrative examples comparing ground-truth images from COCO and generated images from Baseline SD 2.1 and our model. The images generated by our model exhibit greater fidelity to the input prompts, while also achieving a higher level of photorealism.

### 7 Additional Examples from SPRIGHT

Figure 2 and 3 demonstrate a few correct and incorrect examples present in SPRIGHT. While most relationships are accurately described in the captions, on some instances the model struggles to capture the precise spatial nuance.

### 8 Additional Illustrations

Figure 4 shows images generated by our model based on prompts from T2ICompBench, whereas Figure 5 demonstrates that for a given prompt, our model consistently produces spatially accurate images. Figure 6 presents example images generated from the VISOR benchmark.

### 9 Limitations

Since SPRIGHT is a derived dataset, it inherits the limitations of the original datasets. We refer the readers to the respective papers that introduced the original datasets for more details. As shown in our analysis, the generated synthetic

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

The living room has a large couch and a coffee table in the center. The couch is positioned on the left side of the room while the coffee table is located in the middle. Above the couch there is a light ﬁxture hanging from the ceiling. In the background there is a large mirror reﬂecting the room making it appear larger and more spacious. The mirror is positioned on the right side of the room and it is larger than the couch and coffee table.

The image features a large group of people gathered around a river, with some of them standing on boats. The boats are relatively small compared to the number of people present, and they are situated in the water near the shore. The people are spread out along the riverbank, with some standing closer to the water's edge and others further back. The scene captures a lively atmosphere with people interacting and enjoying their time near the water.",

A man is sitting in a boat on the water, with a bridge in the background. The boat is relatively small compared to the bridge, which is quite large and spans

A man is sitting on the beach, facing the ocean. He is positioned in front of the water, with the ocean stretching out behind him

across the water. The man is positioned in the foreground, while the bridge is in the background, creating a sense of depth in the image

[Figure 77]

[Figure 78]

[Figure 79]

In the image a man and a woman are standing in a grassy area. The woman is holding a book and the man is holding a heart. The woman is standing to the left of the man and they are both relatively close to each other. The book is relatively small compared to the woman while the heart held by the man is even smaller. The grassy area they are standing in is quite large covering most of the image

The image shows a highway with a large sign on the left side of the road and a bridge in the middle of the road. The highway is surrounded by trees and hills giving the impression of a rural or mountainous area. The sign is

The image shows a house with a large tree in front of it and a small tree on the right side of the house. The house is located on a street with a sidewalk and there is a mailbox on the sidewalk. The scene also includes a large tree in the background and a small tree near the house

relatively large compared to the bridge which is a smaller structure in the middle of the road. The highway itself is a long straight road that stretches into the distance

- Fig. 2: Illustrative examples from the SPRIGHT dataset, where the captions are correct in its entirety; both in capturing the spatial relationships and overall description of the image. The images are taken from CC-12M and Segment Anything.

captions are not a 100% accurate and could be improved. The improvements can be achieved through better prompting techniques, larger models or by developing methods that better capture low-level image-text grounding. However, the purpose of our work is not to develop the perfect dataset, it is to show the impact of creating such a dataset and its downstream impact in improving vision-language tasks. Since our models are a fine-tuned version of Stable Diffusion, they may also inherit their limitations in terms of biases, inability to generate text in images, errors in generating correct shadow patterns. We present our image fidelity metrics reporting FID on COCO-30K. COCO-30K is not the best dataset to compare against our images, since the average image resolutions in COCO is lesser than those generated by our model which are of dimension 768. Similarly, FID largely varies on image dimensions and has poor sample complexity; hence we also report numbers on the CMMD metric.

### 10 Author Contributions

AC defined the scope of the project, performed the initial hypothesis experiments and conducted the evaluations. GBMS led all the experimental work and customized the training code. EA generated the dataset, performed the dataset

[Figure 80]

[Figure 81]

[Figure 82]

The bride and groom are standing in the center of the image, with the bride on the left and the groom on the right. They are kissing in front of a crowd of people, who are sitting on chairs arranged in rows. The chairs are positioned in front of the couple, with some chairs being closer to the foreground and others further back. The people sitting on the chairs are of various sizes, with some being taller and others shorter, indicating a diverse group of guests

The image shows a large living room with a television on a stand and a couch in the foreground. The television is positioned above the couch and the stand is located in the middle of the room. The couch is situated in front of a large window which allows natural light to ﬁll the room. The room is ﬁlled with furniture including a dining table and several potted plants are placed throughout the space. The living room is adjacent to a patio which can be accessed through a sliding glass door. The room is welllit and

[Figure 83]

The image features a collection of various animals, including a dog, a cat, a panda, and a unicorn. The dog is located on the left side of the image, while the cat is situated above the dog. The panda is positioned in the

spacious creating a comfortable and inviting atmosphere

middle of the image, and the unicorn is on the right side. The unicorn is the largest of the animals, with the panda being the second largest, followed by the cat and the dog being the smallest.

[Figure 84]

[Figure 85]

The image shows a car driving down a road next to a stone wall. The car is positioned on the left side of the road and the wall is on the right side. The car is relatively small compared to the wall which is quite large and extends along the entire length of the road.

A green car with a hood open is parked next to a red car. The green car is much smaller than the red car and is positioned behind it

In the image a large airplane is ﬂying above a smaller airplane both under a cloudy sky. The large airplane is ﬂying above the smaller

one and they are both positioned under the cloudy sky.

- Fig. 3: Illustrative examples from the SPRIGHT dataset, where the captions are not completely correct. The images are taken from CC-12M and Segment Anything.

and relevancy map analyses. SP took part in the initial experiments, suggested the idea of re-captioning and performed few of the evaluations and analyses. DG suggested the idea of training with object thresholds and conducted the FAITHScore and GenEval evaluations. TG initiated the discussions on spatial failures of T2I models and provided consultation on experiments. VL, CB, and YZ coadvised the project, initiated and facilitated discussions, and helped shape the the goal of the project. AC and SP wrote the manuscript in consultation with TG, LW, HH, VL, CB, and YZ. All authors discussed the result and provided feedback for the manuscript.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

A dog on side of a computer

A rabbit next to a train A cat next to a suitcase A butterﬂy next to a train A mouse on side of a bag A boy on side of a bee

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A candle on the left of a mouse

A bird on the left of a microwave

A chair on the right of a giraffe

A bee on the left of a clock

A bird on the right of a man A bag on the right of a dog

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

An airplane on the top of a horse

A bicycle on the top of a turtle

A mouse on the top of a bowl

A boy on the bottom of a bee

A dog on the bottom of a desk

A phone on the bottom of a ﬁsh

##### Fig. 4: Illustrative examples from our model, as described in Section 4.1, on evaluation prompts from the T2I-CompBench benchmark.

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

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

A sheep on the left of a lamp

A rabbit on the top of a candle

A pig on the bottom of a train

A turtle next to an airplane

A vase on the right of a cat

##### Fig. 5: Generated images from our model, as described in Section 4.2, on evaluation prompts from T2I-CompBench. We find that for a given text prompt, our model consistently generates spatially accurate images.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

A TV above a toilet A surfboard above a bike A horse above a pizza An airplane above a bench

A cell phone below a cow

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

A bicycle below a trafﬁc light

A refrigerator to the left of a toilet

A ﬁre hydrant to the left of a dining table

A chair below a tie A cup below a bus

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

An elephant to the leftof an apple

A bear to the leftof an umbrella

A carrot to the rightof a cat

A hair drier to the rightof a wine glass

A bus to the right of a donut

##### Fig. 6: Generated images from our model, as described in Section 4.2, on evaluation prompts from the VISOR benchmark.

