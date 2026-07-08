## COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation

Xueqing Deng Qihang Yu Ali Athar Chenglin Yang Linjie Yang Xiaojie Jin Xiaohui Shen Liang-Chieh Chen

ByteDance Seed Project Page

# arXiv:2502.02589v1[cs.CV]4Feb2025

xueqingdeng@bytedance.com

###### #mask <15 15 < #mask <25 #mask >25

[Figure 1]

[Figure 2]

[Figure 3]

###### #instance masks = 5 #semantic masks = 2 #instance masks = 13 #semantic masks = 5 #instance masks = 54 #semantic masks = 9

The image shows a lively family gathering around <34:along dining table>, with various labeled individuals, dining items, and furniture. In terms of people, the table is surrounded by multiple family members of varying ages, all seated and enjoying the meal together. Key individuals include <50:a lady with blonde short curly hair wearing tank>, <52: a lady with black medium hair in blue shirt>, <53: a women with black medium hair but wearing in tank top>, and <55: a boy in red t shirt > sitting right next to the lady in the left side of the image, as well as on the right there are <32: a women in white t shirt with black dots>, <41: a boy in white t shirt>. < 62,63,48, 47, 13, 56, 53, 54, 55, 52, 50, 33, 42, 43, 41, 32, 45, 44, 47, 46:Additional people> are seated around or standing near the table, filling the room with a sense of warmth and gathering. The dining items are plentiful and include numerous <14, 26, 25, 40: wine glass>, <15, 17, 36, 16, 20, 39, 38: cup>, and <12, 60:other bowl> and <30, 10:spoon> objects scattered across the table. Utensils, such as <5: fork> and <9: knife>, are visible among the plates of food, which include main dishes and side dishes. In the background, the <0: light colored wall> and <1: wooden cabinet> display family photos, framed art, and <4: pink flowers> in <21:a glass vase>, adding a personal touch to the room decor. The combination of food, drink, and a lively gathering suggests a celebratory or holiday meal.

The image depicts a cozy living room with various items of furniture and decor. The <0:wooden floor> represents the wooden flooring that spans the room, partially covered by <1: a black and white patterned rug> in the center. The <2:light yellow walls> surrounding the other elements in the room. <3:A coffee table> sits on the rug where there is <16: a book> on top of it, in front of the seating area. Against the back wall, there is <4: wooden cabinet>, which holds the <10: tv with screen off> , <15: a vase> and some decorative items. The <5: window> is located on the back wall, partially covered by <6: white window blind>, which lets light into the room. Above the cabinet, <7:a shelf> holds various decor pieces, including <14:a vase>.

The image shows a meal plate with various labeled regions. In the center and right side of the plate, the region represents <0:a serving of French fries>. The upper left side shows <1: a slice of onion>. Positioned on the lower left side, it is <2: a small blue bowl> containing a red sauce, likely ketchup, while <3:another bowl> near the top center hold additional dipping sauce. At the bottom right of the image, <4: a silver knife> is partially visible under some of the food items. The main item on the plate, <5:a sandwich with a sesame seed bun> occupies the center and there is <6: another sandwich> on the right. Finally, all these items are put on the <7: dining table surface>. Together, the image captures the different elements of the meal setup, showing a typical plate with a sandwich, fries, and dipping sauce.

- <8: A mirror> is mounted on the left wall along with multiple frames, while

- <9:a light fixture> is visible in the upper right corner. In the foreground, <12: a light-upholstered couch> where <11:a black cat> lies, adds a cozy touch. On the right side, there is <13:a blue-upholstered chair> near the coffee table, and <17:a reddish chair> positioned near the back wall.

###### Empowered Tasks by COCONut-PanCap Dataset

###### Detailed Captioning

###### Panoptic Segmentation Grounded Caption

###### Text-to-Image Generation

Downstream Task

VQA:

USER: Generate an image with the following description. This image shows a cozy, well-lit kitchen with white cabinetry, giving it a bright, clean look. The dark countertops contrast with the white

[Figure 4]

USER: Could you please describe the image in detail?

USER: Please provide a detailed description of the image and segment each part. ASSISTANT:

[Figure 5]

[Figure 6]

USER: How many kinds of vegetables ? ASSISTANT: There are broccoli, carrot ….

ASSISTANT: Three teddy bears, each a different shade of brown, are positioned together on a blue bedsheet. On the left, a golden brown teddy ….

[Figure 7]

cabinets. … ASSISTANT:

[Figure 8]

Referring Segmentation:

[Figure 9]

slightly with the others, also embracing its adjacent bears. The soft blue bedsheet in the background provides a cozy base for this heartwarming scene.

USER: Can you segment the cat which is on the toilet? ASSISTANT: It is <SEG>.

Figure 1. COCONut-PanCap Dataset. Top: The proposed COCONut-PanCap dataset features detailed captions grounded with dense panoptic segmentation masks. Bottom: COCONut-PanCap supports various fine-grained understanding and generation tasks, including detailed captioning, panoptic segmentation grounded caption, and text-to-image generation. The dataset also facilitates several downstream tasks, such as visual question-answering (VQA) and referring segmentation.

### Abstract

masks, ensuring consistency and improving the detail of generated captions. Through human-edited, densely annotated descriptions, COCONut-PanCap supports improved training of vision-language models (VLMs) for image understanding and generative models for text-to-image tasks. Experimental results demonstrate that COCONut-PanCap significantly boosts performance across understanding and generation tasks, offering complementary benefits to largescale datasets. This dataset sets a new benchmark for evaluating models on joint panoptic segmentation and grounded

This paper introduces the COCONut-PanCap dataset, created to enhance panoptic segmentation and grounded image captioning. Building upon the COCO dataset with advanced COCONut panoptic masks, this dataset aims to overcome limitations in existing image-text datasets that often lack detailed, scene-comprehensive descriptions. The COCONut-PanCap dataset incorporates fine-grained, region-level captions grounded in panoptic segmentation

captioning tasks, addressing the need for high-quality, detailed image-text annotations in multi-modal learning.

### 1. Introduction

Recent advancements in multi-modal foundation models have been largely driven by the availability of large-scale paired text-image datasets. These datasets, often collected via web crawling with basic filtering techniques [14, 52, 53], contain low-quality, web-sourced captions that lack depth and accuracy. In contrast, human-annotated caption datasets, such as COCO-caption [6], offer higher-quality descriptions but are limited in scale and tend to be concise, with an average caption length of 10 words. To overcome the limitations of short captions, the research community has leveraged vision-language models (VLMs) [5, 31, 32, 38, 60] to generate detailed synthetic captions. While these machine-generated captions improve visual understanding [5, 32] and generation tasks [31], they remain inferior to high-quality, human-verified annotations [44].

Addressing this challenge requires balancing scalability and annotation quality, as generating detailed and accurate image descriptions at scale remains labor-intensive [15, 44]. In this paper, we introduce an efficient annotation approach that combines dense mask annotations with commercial VLMs [5] to produce high-quality image captions. Our goal is to minimize human effort while generating rich, structured descriptions.

To achieve this, we base our work on the COCO-caption dataset [6] due to its widespread use and diverse image content. We revisit the COCO-caption dataset to provide more detailed and comprehensive caption annotations. Our approach involves creating holistic captions synthesized from region-based dense captions that describe distinct areas within each image. Specifically, we build on recent COCONut panoptic segmentation annotations [9] to generate a new set of detailed captions by: (a) annotating each segmentation region with a VLM-generated draft, carefully refined through human corrections, and (b) summarizing these region captions into a comprehensive image caption while preserving the grounding correspondence between image masks and object references. This enables a novel task that integrates panoptic segmentation with grounded captioning. Our structured annotation process ensures that the captions are both complete, covering the majority of objects in each image, and grounded, with precise segmentation masks.

The final dataset, named COCONut-PanCap, is designed for a wide range of vision-language applications, combining Panoptic segmentation and grounded Captioning. It comprises 118K image-text pairs for training, with an average caption length of 203 words, as well as an additional 25K image-text pairs, with an average caption length of 233 words for validation. We demonstrate that

COCONut-PanCap significantly boosts the performance of both VLM and text-to-image generation models at the instruction tuning and fine-tuning stages, outperforming recent detailed caption datasets [44]. This highlights the potential of our grounding-based captions for both visionlanguage understanding and image generation tasks. Our contributions are summarized as follows:

- • We propose a caption annotation pipeline leveraging panoptic segmentation to create a high-quality, detailed caption dataset comprising 143K annotated images. The resulting annotations are comprehensive, accurate, and include grounding masks, making this dataset substantially larger than recent detailed caption datasets.
- • Our COCONut-PanCap dataset facilitates a new challenging task combining Panoptic segmentation and Grounded Captioning (PGC). We establish evaluation metrics and settings for this PGC task and benchmark several recent methods to assess performance on this novel challenge.
- • We validate the utility of our proposed dataset across various fine-grained Image-to-Text (I2T) and Text-to-Image (T2I) tasks, including detailed caption generation, PGC, visual question answering (VQA), referring segmentation, and text-conditioned image generation. Experimental results show that our dataset significantly enhances model performance across all these tasks.

### 2. Related Work

Detailed Captions from VLMs. Researchers are increasingly interested in creating large-scale datasets with detailed captions generated from advanced vision-language models. DenseFusion1M [32] utilizes a pretrained perceptual model to prompt VLMs, facilitating more detailed image descriptions.

Recap-DataComp1B [31] first fine-tunes the Llama-38B powered LLaVA-1.5 model [36], then applies it to recaption approximately 1.3 billion images from the DataComp1B dataset [14], generating a rich repository of detailed image descriptions. On a similar front, the PixelProse dataset [59] offers general-purpose image captions designed to serve various applications, from visual question answering (VQA) to pre-training tasks. Unlike datasets targeting single applications, PixelProse captions are dense, versatile image descriptions that can be adapted to other formats, such as VQA and instructional data, with the help of large language models (LLMs). Although these detailed caption datasets are large-scale, they are directly generated by VLMs without human verification, falling behind humanannotated captions on quality. Our proposed COCONutPanCap dataset leverages extensive human effort to ensure high-quality annotations.

Human-annotated Detailed Captions. Several efforts have been made toward this goal, utilizing fully human-

|Dataset Name<br><br>|Image Source|Sample<br><br>|Annotated by|Avg. Words<br><br>|Masks|
|---|---|---|---|---|---|
|BLIP-LCS DenseFusion1M [32] LLaVA-Recap118K [38] LLaVA-Details-23K [37] ShareGPT4V [5] ShareGPT4V-PT [5]<br><br>|LAION [53], CC [4], SBU [45]<br><br>LAION [53] COCO [35] COCO [35]<br><br>LAION [53], CC [4], SBU [45], COCO [35] etc. LAION [53], CC [4], SBU [45], COCO [35] etc.|558K 1,059K 118K 23K 100K 1,246K<br><br>|BLIP [30] Vision Specialist Models LLaVA-NEXT [38] GPT4 GPT4-Vision Share-Captioner [5]<br><br>|54 191 186 105 162 144|✗ ✗ ✗ ✗ ✗ ✗<br><br>|
|PixelLM-MUSE [51] Osprey [69] GLaMM-GCG [50]<br><br>|LVIS [17] COCO [35]<br><br>RefCOCOg [40],PSG [65],Flick30K [47]|246K 724K 214K<br><br>|GPT4-Vision GPT4-Vision<br><br>Vision Specialist Models<br><br>|128<br><br>|3.7‡ 3.6|
|COCO-caption [6] DCI [61] DOCCI [44] IIW [15] COCONut-PanCap (ours)<br><br>|COCO [35] SA-1B [24]<br><br>DOCCI [44] WebLI [15] COCO [35]<br><br>|118K 8K 9.6K 8.5K 118K|Human Human Human Human Human<br><br>|11 144 136 217 203<br><br>|✗ ✗ ✗ ✗ 13.2|

- Table 1. Dataset (training set) Comparison. Our proposed COCONut-PanCap dataset stands out for its detailed (2nd highest in Average Words), high-quality (human interactive annotated) captions and high-density segmentation masks (1st in Average Masks). ‡ denotes the mask number for referring segmentation which only counts the targets in QA format. Note that “Samples” means the number of collected annotations, where there may exist one image with multiple different annotation, i.e., in region-level datasets like Osprey.

|Dataset Name<br><br>|Samples|Avg. Words<br><br>|Caption T2I Grd. Seg.|
|---|---|---|---|
|COCO-30K [6] DOCCI-test [44] IIW-test [15] GenEval [16] T2I-CompBench val [20] GLaMM-GCG val-test [50] COCONut-PanCap val (ours)|30,000<br><br>5,000 445 553<br><br>2400 2,000 25,000<br><br>|11 136 217<br><br>8<br>9<br><br><br>128 233|✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗<br><br>✗ ✓ ✗<br><br>✗ ✓ ✗<br><br><br>✓ ✗ ✓ ✓ ✓ ✓<br><br>|

- Table 2. Dataset (evaluation set) Comparison. Our COCONutPanCap validation set provides detailed captions and supports multiple multi-modal tasks, including image captioning, text-toimage generation (T2I), and grounded segmentation (Grd. Seg.).

ing work have made significant strides in creating datasets with region-level captions linked to entity segmentation masks [69] or bounding boxes [70]. However, few datasets associate grounded segmentation directly with captions. GLaMM [50] proposes a Grounding-anything Dataset (GranD) using an automated annotation pipeline that encompasses 7.5M unique concepts grounded in a total of 810M regions available with segmentation masks.

Later, MGLMM [72] further explore the multigranularity GLaMM model to generate a multi-granularity dataset. Our proposed COCONut-PanCap dataset follows a similar approach of grounding captions to dense masks but offers significantly denser masks per caption, as shown in Tab. 1, with an average of 13.2 masks per image compared to 3.6 in GLaMM. Note that we focus on grounded segmentation for detailed captions, rather than descriptions of all levels of segmentation masks (objects or parts) as provided in the GranD dataset [50], which is outside the scope of our study.

annotated data or human-in-the-loop approaches. One example is DOCCI [44] which is a small, high-detailed image caption dataset that is entirely human-annotated, containing only 15K samples but providing diverse details, such as key objects, their attributes, spatial relationships, and text rendering. Two small-scale detailed caption datasets, ImageInWords [15] and DCI [61], use a combination of automatic annotation models with human involvement, both with fewer than 10K samples. Pixmo-Cap [8] introduces a large-scale dataset of detailed image captions from speechbased descriptions, offering richer visual annotations than text-based methods.

### 3. COCONut-PanCap Dataset

We construct a novel dataset based on COCO images to provide detailed captions at both image and mask levels, using COCONut panoptic masks as a foundation for comprehensive region descriptions. Specifically, we leverage panoptic masks from COCONut-S [9] to annotate detailed region captions, incorporating both ‘thing’ and ‘stuff’ masks to cover a wide range of semantic regions.

Our proposed COCONut-PanCap dataset yields smaller scale compare to Pixmo-Cap but we have different focuses where Pixmo-Cap focuses on pretraining the VLMs while we focus on the instruction tuning and finetuning stages of VLMs and image generation models. Our work also shares a similar annotation pipeline with a recent video captioning dataset Shot2Story [18] where both VLM draft and human corrections are used to create complete and accurate annotations.

##### 3.1. Dataset Description

Comprehensively understanding diverse visual elements in complex scenes can benefit multiple tasks including perception, understanding, and generation. In this section, we describe the annotation pipeline for our dataset leveraging the human annotated panoptic masks. We first show the sta-

Grounded Captions with Segmentation Masks. Exist-

[Figure 10]

###### round-1 round-2

[Figure 11]

[Figure 12]

0: road – The ground surface in the foreground, labeled as the road.1: wallother-merged – A section of the wall on the left side of the image, with a mural or design painted on it, including part of a blue octopus.2: window-other – A window located on the upper right side of the image, which has been labeled separately. ………… t, also wearing a hat and purple clothing, possibly engaged in conversation.11: chair – The chair that the first person is sitting on.12: bicycle – A bicycle is located near the center of the image, leaning against the wall.13: handbag – A handbag or bag is near the first person or placed beside the chair.14: chair – Another chair, possibly for the second person sitting in the image.

[Figure 13]

[Figure 14]

[Figure 15]

summarize by VLM, veriﬁca on check by human

###### VLM

human edit

overlaid image

input image human annotated panoptic segmentation mask

set-of-marks prompt instruc on

detailed cap on dra 

- Figure 2. Annotation Pipeline. Given an input image, human-annotated panoptic segmentation masks are overlaid using set-of-marks [66] visualization techniques to prompt the vision-language model (VLM). After generating an initial draft, human effort is investigated for editing and verification. Finally, the annotated metadata will be formatted to construct the datasets for various tasks at instruction tuning or finetuning stage.

[Figure 16]

: Could you please describe the details of the image for each region? : From your image, here is the analysis:

[Figure 17]

[Figure 18]

- 0: Sky - Clear blue sky with some scattered clouds.
- 1: Road - Asphalt road, smooth and unobstructed, ideal for vehicles.
- 2: Building-other-merged - Building structure with large glass windows and beige walls, likely a commercial or public building.
- 3: Tree-merged - Lush green trees with dense foliage, possibly ornamental in a landscaped area.
- 4: Pavement-merged - Grey concrete pavement, typically used for pedestrian pathways.
- 5: Stop sign - Red octagonal traffic stop sign with white text, indicating a vehicle stop required.
- 6: Car - A car in motion, likely in the process of slowing down as it approaches the stop sign.
- 7: Truck - A white truck, stationary, parked in a designated area, possibly for loading or unloading goods.

With the given region-level description, can you merge them into one paragraph in the format of grounded caption?

[Figure 19]

In this scene, the clear blue sky with some scattered clouds serves as a pleasant backdrop. The smooth asphalt road provides an unobstructed pathway, suitable for vehicles. Nearby, there is a building structure with glass windows and beige walls. Lush green trees with dense foliage add a touch of nature. The grey concrete pavement offers a pedestrianfriendly path alongside the road. A upside down red octagonal traffic stop sign with white text signals vehicles to halt. Approaching the stop sign is a car, likely slowing down. In the background, a stationary white truck is parked in a designated area.

is upside down

is approaching

commercial

x

loading or unloading goods

x

|: add : remove x found hallucination and remove<br><br>human edit legend:|
|---|

[Figure 20]

round-1

round-2

- Figure 3. Designed Prompt Template. By giving the concatenated set-of-marks images, the right side (round-1) shows the initial response and the corresponding human edits. Once finalized by humans, these edits will be merged into a single detailed caption grounded with panoptic segmentation masks, as shown in the left side (round-2).

tistical analysis of our final dataset in Tab. 1. On average, our captions contain 203 words spanning 11 sentences. We follow the same split setting in COCO2017 [35] dataset, which includes 118K training images. To provide a comprehensive evaluation set, we adopt the same 25K images from COCONut-val split (which contains COCO2017-val and another 20K Objects365 [55] validation images).

objects to reference in the caption. These masks include both ‘thing’ and ‘stuff’ classes, representing single objects and semantic regions, respectively. We adopt the panoptic segmentation masks from the COCONut-S [9] dataset. The masks are overlaid on the images, labeled with class names c1,c2,...,cn ∈ C, where C is the set of COCO’s 133 panoptic classes. We then construct a prompt with both the edited image and the original image and a textual question for GPT-4V, as illustrated in Fig. 3. The resulting region captions from GPT-4V are reviewed and corrected by human raters for accuracy and consistency.

##### 3.2. Dataset Construction

We argue that high-quality descriptions should provide sufficient details of key objects and their attributes, as well as information about secondary objects and background elements. To achieve this, as shown in Fig. 2, we use humanannotated panoptic segmentation masks to decide the set of

[Figure 21]

- Figure 4. Frequency of Extracted Nouns from the COCONutPanCap Dataset. The top 10 most frequent nouns are: people, table, room, street, dining, man, person, cars, chairs, and field.

##### 3.3. Dataset Analysis

Concepts Beyond COCO’s 133 Classes. To clarify the goal of our annotation task, we focus on key visual features such as objects, attributes, spatial relationships, and counting. As shown in Fig. 4, we utilize the panoptic segmentation mask from COCONut-S, which includes 133 classes in the word vocabulary. Our proposed dataset, however, incorporates additional concepts beyond these 133 classes, such as ‘vegetable’ and ‘parking’. This demonstrates that our human annotators delivers accurate and diverse descriptions when using the provided label names as a reference.

User Study for Caption Quality. We randomly sample 1,000 images from our COCONut-PanCap training set and asked a human evaluator to perform a single-choice selection task. The question is: ‘Please select the best description for the image, considering the correctness of object names, attributes, counting, spatial relationships, and action.’ The compared captions are generated using GPT4V [1], Qwen2-VL [64], and InternVL-2 [7], resulting in a single-choice four-option question. Fig. 5 illustrates the results, showing that our GPT-assisted human-annotated captions receives the highest ratings. More details can be found in the supplementary.

### 4. PGC Baseline: PanCaper

In this section, we introduce our baseline method for joint panoptic segmentation and grounded captioning (PGC), namely PanCaper. We start with an overview of the pixel grounding task and then present our proposed approach, which incorporates a panoptic segmentation module specifically designed for grounding objects in captions.

Revisiting the Pixel Grounding Task. Our baseline model builds upon LISA [28], a model that combines the language generation capabilities of VLMs with the ability to produce segmentation mask. LISA consists of three main components: a VLM, a vision backbone V , and a mask decoder D. With a given text prompt, the VLM (typically LLaVA [36, 37]) generates an output containing a ⟨SEG⟩ token. For instance, with the input prompt, ‘Could you segment the food with high Vitamin C?’ LISA generates the response ‘It is ⟨SEG⟩.’ This process extracts the last-layer

[Figure 22]

Figure 5. Caption Quality via User Study. The study involved human evaluators assessing a random sample of 1,000 captions, with a strong preference shown for captions from our dataset.

embedding of the LLM from LLaVA. Then a language-toprompt (L-P) projection layer (g) transforms the last-layer embeddings corresponding to ⟨SEG⟩ tokens (lseg) into the decoder’s feature space. Meanwhile, the vision backbone extracts dense visual features from the input image. Finally, both the dense features and the CLIP image embedding from LLaVA are fed into the mask decoder to produce the final segmentation mask.

Prompt Instruction for Grounded Captioning. We propose a baseline method for the PGC task by modifying LISA to enable grounded captioning with segmentation masks. Since LISA was originally designed for generating segmentation with a single output mask, two main adjustments are necessary: (1) the use of multiple ⟨SEG⟩ tokens, and (2) extracting noun phrases from the caption for grounding. To facilitate grounded segmentation, we modify the prompt to the VLM as ‘Please provide a detailed description of the image and segment each part.’ This prompt triggers the model to generate caption responses with corresponding ⟨SEGi⟩ tokens, where i ∈ [1,N] and N is the total number of predicted segmentations.

Given a predicted caption for the image, aligning each ⟨SEGi⟩ token requires pairing it with a noun phrase, ‘⟨p⟩phrasei⟨/p⟩,’ where phrasei is the relevant part in the caption to be grounded. With these prompt tokens defined, the model uses the vision backbone V and mask decoder D to facilitate fine-grained, pixel-level grounding, with D producing segmentation masks M.

Enable Panoptic Grounding. To achieve panoptic segmentation from captions, we first classify ⟨SEG⟩ tokens into two types: ⟨SEGt⟩ for ‘thing’ classes and ⟨SEGs⟩ for ‘stuff’ classes. These tokens are then processed by our segmentation modules to produce panoptic segmentation masks. We initialize the vision backbone V with a pretrained kMaX-DeepLab encoder [67] and fine-tune the decoder D using our COCONut-PanCap dataset. Since kMaX-DeepLab operates as a closed-set segmenter, we

align text embeddings of the associated noun phrases with COCO’s 133 panoptic classes. To accomplish this alignment, we use BERT [26] to generate the text embeddings and to calculate cosine similarity, selecting the bestmatching category. Panoptic grounding provides mapping between detailed captions and image regions, which improves interpretability of VLM predictions.

Training Objectives. Our training objective aims to minimize the following losses:

L = λtextLtext + λmaskLmask, (1)

where Ltext is the auto-regressive cross-entropy loss for text generation, and Lmask is the mask loss [63], encouraging the model to produce high-quality segmentation results. λtext and λmask are the respective loss weights. We use the same loss weights as LISA [28].

Evaluation Metrics for Caption Quality. We conduct the analysis with multiple metrics to evaluate the quality and completeness of the generated captions. We introduce a benchmarking suite for the PGC task, with a validation set of 25K images. For the caption quality, we report the caption metrics including CIDEr [62], METEOR [2], ROUGEL [34], BLEU@4 [46] and CAPTURE [10]. For grounded panoptic segmentation, we report PQ scores [23].

### 5. Experimental Results

We assess the effectiveness of human-annotated caption data by performing three primary tasks utilizing our dataset in the fine-tuning/instruction tuning stage: detailed captioning, panoptic grounded captioning (PGC), and text-toimage generation. Additionally, we demonstrate the transferability of the knowledge learned from our dataset through two downstream tasks: VQA and referring segmentation.

Detailed Captioning. We conduct instruction tuning with LLaVA-NeXT framework [38] for this task. We replace the caption data (23k) from the original LLaVA instructiontuning set with detailed captions from our dataset, keeping the same amount of instruction data size. We follow the same training setup used for LLaVA-NeXT with Llama3-8B [11]. Treating it as a QA task, we use the prompt, ‘Could you please describe the image in detail?’ and collect the corresponding response as the caption for the image. We evaluate caption quality using CIDEr [62], METEOR [2], BLEU@4 [46], ROUGE-L [34] and CAPTURE [10] metrics. We also extend the model by adding the mask-pooled features from the panoptic segmentation masks as additional signals to the LLaVA model and name it LLaVA-NeXT-pool. During training, we use the ground truth mask to extract the features while during inference we use the mask proposals from the pretrained kMaXDeepLab [67]. Besides, we also experiment with synthetic captions directly generated using InternVL-2 [7], Qwen2-

VL [64] and GPT-4V [1]. We follow the same data preparation settings as our dataset to build these instruction datasets for these 23K images with different sources of synthetic detailed captions, namely LLaVA 665K-InternVL2-Cap , LLaVA 665K-Qwen2VL-Cap, and LLaVA 665K-GPT4VCap. These datasets are used to produce models LLaVANeXT-I, LLaVA-NeXT-Q, and LLaVA-NeXT-G respectively. More details can be found in the supplementary. The results are presented in Tab. 3. LLaVA-NeXT models show improved performance when fine-tuned on the custom instruction-tuning dataset. Among these, LLaVA-NeXTpool achieves the highest scores in all metrics, with CAPTURE of 61.4, CIDEr of 13.1, BLEU@4 of 5.3, and METEOR of 17.1, significantly higher than the original model variant LLaVA-NeXT, indicating the benefit of added region features for additional visual cues. Models trained on synthetic captions (LLaVA-NeXT-I, LLaVA-NeXT-Q, and LLaVA-NeXT-G) generally show lower scores, showing advantage of our human-annotated caption.

PGC: Stronger Detail Reasoning Performance. We implement our proposed PanCaper based on LISA which uses pre-trained LLaVA-NeXT with a LLM of Llama3-8B, with LoRA [19] adopted. The vision encoder uses a fixed CLIPViT-L/14-336 model, modified with linearly interpolated position embeddings to process 448 resolution images. The trainable components of our model include the mask decoder of kMaX-DeepLab, and the tunable parts in LLaVA same as in LISA. To enhance model performance in visual understanding, we initialize our PanCaper using pretrained LLaVA-NeXT models from the detailed captioning task. We also experiment with a model variant that uses mask pooled features similar to LLaVA-NeXT-pool, and name it PanCaper-Pro.

For comparison, we select 3 related methods LISA, PixelLM [51] and GLaMM [50] for evaluation. It is noteworthy that LISA is not able to perform multi-mask prediction. We therefore adapt LISA [28] for the multi-mask generation with grounded segmentation, namely LISA+. The implementation details can be found in the supplementary. Tab. 4 shows the quantitative results. Our proposed PanCaperPro achieves the highest scores across all captioning metrics (CIDEr: 12.5, CAPTURE: 64.3, BLEU@4: 6.4, METEOR: 17.9), outperforming all other models. Both PanCaper models show significant improvements over other models in all captioning metrics, highlighting the effectiveness of the COCONut-PanCap dataset for detailed caption generation. On grounding segmentation, PanCaper-Pro again leads, with a PQ score of 0.61, PQthing of 0.58, and PQstuff of 0.68, reflecting its robustness on both “thing” and “stuff” classes. Notably, enabling mask pooling in our proposed PanCaper-Pro further enhances segmentation metrics. The baseline models (LISA+ and GLaMM with GranD) achieve much lower PQ scores, due to incomplete segmentation an-

|Training recipe<br><br>|Method<br><br>|Pretrain Dataset<br><br>|Instruction-tuning dataset<br><br>|Mask pooled|CAPTURE<br><br>|CIDEr BLEU@4 METEOR ROUGE-L|
|---|---|---|---|---|---|---|
|finetune<br><br>|LLaVA-NeXT* LLaVA-NeXT LLaVA-NeXT-pool|LAION-CC-SBU LAION-CC-SBU LAION-CC-SBU<br><br>|LLaVA 665K LLaVA 665K-COCONut-PanCap LLaVA 665K-COCONut-PanCap<br><br>|✗ ✗ ✓<br><br>|55.4 58.7 61.4<br><br>|10.8 4.2 13.2 23.1<br>11.2 4.8 16.2 24.6 13.1 5.3 17.1 26.8<br>|
| |LLaVA-NeXT-I LLaVA-NeXT-Q LLaVA-NeXT-G|LAION-CC-SBU LAION-CC-SBU LAION-CC-SBU<br><br>|LLaVA 665K-InternVL2-Cap LLaVA 665K-Qwen2VL-Cap LLaVA 665K-GPT4V-Cap<br><br>|✗ ✗ ✗<br><br>|53.9 55.4 56.2|9.4 4.4 11.5 21.4<br><br>8.9 4.6 12.9 22.5<br><br>9.6 4.7 13.3 22.8<br>|

- Table 3. Caption Benchmark Results Evaluated on Our COCONut-PanCap Val Set. Note that the amount of data in the instruction dataset remains the same; only the sources of the detailed captions vary, with a total of 23K images that have detailed captions.

Caption Grounding segmentation Method Pretrain dataset Instruction dataset Mask pooled CAPTURE CIDEr BLEU@4 METEOR PQ PQthing PQstuff

LISA+ * LAION-CC-SBU GranDf ✗ 46.2 6.6 3.8 9.8 0.43 0.41 0.45 LISA+ LAION-CC-SBU COCONut-PanCap (ours) ✗ 57.9 8.1 4.9 13.8 0.50 0.49 0.44 GLaMM GCG * LAION-CC-SBU+GranD GranDf ✗ 43.2 6.5 3.6 10.6 0.27 0.35 0.21 GLaMM GCG LAION-CC-SBU+GranD COCONut-PanCap (ours) ✗ 56.8 7.8 5.2 14.3 0.55 0.54 0.46 PanCaper (ours) LAION-CC-SBU COCONut-PanCap (ours) ✗ 62.6 12.0 5.8 15.4 0.56 0.55 0.66 PanCaper-Pro (ours) LAION-CC-SBU COCONut-PanCap (ours) ✓ 64.3 12.5 6.4 17.9 0.61 0.58 0.68

- Table 4. Joint Panoptic Segmentation and Grounded Captioning (PGC) on COCONut-PanCap Val Set. * denotes reproduced results.

|Training dataset<br><br>|Evaluation dataset<br><br>|FID↓ FDdinov2 ↓ CLIPScore↑|
|---|---|---|
|SD3 PT dataset [12] COCO-caption [6] DOCCI [44] COCONut-PanCap (ours)<br><br>|DOCCI test set [44]<br><br>|30.2 345 74.9 27.6 321 76.8 22.1 300 77.8 21.4 290 77.9|
|SD3 PT dataset [12] COCO-caption [6] DOCCI [44] COCONut-PanCap (ours)<br><br>|COCONut-PanCap val set (ours)<br><br>|31.8 300 73.8 28.0 294 74.0 24.3 267 75.1 23.1 260 77.3|

- Table 5. Benchmark Results on Text Conditioned Image Generation. Stable-Diffusion-3 (SD3) medium is finetuned with COCO-Caption (short), DOCCI and our COCONut-Panoptic and evaluated on DOCCI test set [44] and our COCONut-PanCap val set. ‘SD3 PT dataset’ denotes the pretraining dataset of SD3, and thus the rows correspond to zero-shot evaluation of SD3.

| |w/o FT COCO-caption [6] DOCCI [44] COCONut-PanCap<br><br>|
|---|---|
|color attribution colors position counting single object two objects|0.37 0.34 0.38 0.40 0.73 0.70 0.74 0.75 0.33 0.30 0.36 0.36 0.65 0.64 0.65 0.70 0.96 0.94 0.95 0.96 0.80 0.78 0.81 0.89<br><br>|
|overall score<br><br>|0.64 0.62 0.65 0.68|

- Table 6. Effects of Fine-tuning the SD3-medium (T2I model) with Different Datasets on GenEval [16]. w/o FT denotes the model is not finetuned with any datasets (i.e., zero-shot testing).

we explore the short COCO-caption and detailed captions from our dataset. For DOCCI images, we directly use the captions from their dataset. Tab. 5 shows the quantitative results. Traning on COCONut-PanCap achieves the best performance across all metrics when evaluated on DOCCItest, with the lowest FID (21.4), lowest FDdinov2 (290), and the highest CLIPScore (77.9), indicating superior generation quality and high image-text relevance. When evaluated on COCONut-PanCap-val set, training on COCONutPanCap again shows the best results with the lowest FID (23.1), FDdinov2 (267), and a high CLIPScore of 77.3.

- Tab. 6 shows the results on GenEval benchmark [16].

Finetuning SD3-medium with COCONut-PanCap consistently scores the highest in most categories, particularly those requiring image details like color attribution, object positioning, and handling multiple objects. Our proposed dataset enables more accurate image generation that requires understanding of relationships, multiple objects and counting, tasks that other datasets struggle with.

VQA. To evaluate the effectiveness of the proposed COCONut-PanCap dataset, we utilize these captions during the instruction-tuning stage and follow the setup of LLaVANeXT [38] across various visual question answering (VQA) and multi-modality understanding benchmarks. We evaluate on MM-Vet [68], SEED-IMG [29], MMBench-en [39], MME [13], POPE [33], and TextVQA [58], covering a broad range of evaluation dimensions. We experiment with different amount of our COCONut-PanCap caption data injected into the instruction tuning stage by replacing the original COCO captioning data with our dataset. As shown in

- Tab. 7, the baseline model LLaVA-NeXT (using its original recaptioned COCO) achieves relatively lower performance across all metrics, with scores such as 43.5 on MMVet, 70.1 on Seed-IMG, and 68.9 on TextVQA. Building

notations in the GranD dataset.

Text-to-Image Generation. We adopt the Stable Diffusion 3 (SD3) medium model1 for text to image generation with LoRA finetuning. We adopt the default training settings but only with different text-image datasets for training. We evaluate with two types of training images from COCO [35] and DOCCI [44] datasets. In details, for the COCO images,

1https://huggingface.co/docs/diffusers/stable diffusion/stable diffusion 3

|Method<br><br>|LLM<br><br>|Instruction-tuning Dataset|MM-Vet Seed-IMG MMBench-en TextVQA POPE MME|
|---|---|---|---|
|LLaVA-NeXT * LLaVA-NeXT-20K LLaVA-NeXT-50K LLaVA-NeXT-Full|Llama3-8B Llama3-8B Llama3-8B Llama3-8B<br><br>|orginal LLaVA 665K [38] LLaVA 665K-COCONut-PanCap-20K LLaVA 665K-COCONut-PanCap-50K LLaVA 665K-COCONut-PanCap-118K<br><br>|43.5 70.1 71.4 68.9 85.4 1523<br>44.1 72.5 73.6 69.8 86.1 1552<br><br><br>44.6 73.1 74.2 70.0 87.1 1600<br>45.5 74.3 75.1 70.7 87.9 1612<br>|
|LLaVA-1.5 LLaVA-1.5<br><br>|Vicuna-7B Vicuna-7B|LLaVA 665K-ShareGPT4V-100K<br><br>LLaVA 665K-COCONut-PanCap-20K|37.8 67.4 70.5 64.6 84.7 1519<br><br>38.5 67.7 70.9 64.5 84.9 1521<br>|

- Table 7. Benchmark Results and Ablation Study on VQA. By adding extra detailed caption data for instruction tuning, the models show increased improvement. * denotes reproduced results. Using only 20K human labeled data can still achieve comparable performance to 100K synthetic data.

|Method|refCOCO refCOCO+ refCOCOg val testA testB val testA testB val test<br><br>| | |
|---|---|---|---|
|GLaMM* [50] PixelLM [51] LISA-7B [28] PanCaper+ PanCaper+ + COCONut-PanCap<br><br>|77.5 79.2 74.9<br><br>73.0 76.5 68.2<br>74.1 76.5 71.1 74.5 76.7 69.9 76.2 77.1 72.3<br>|71.3 74.7 61.5 66.3 71.7 58.3 62.4 67.4 56.5<br><br>69.9 73.4 59.5<br><br>70.5 73.9 60.1<br><br><br>|71.3 71.9 69.3 70.5 66.4 68.5 69.8 70.6<br>72.1 71.6<br>|

- Table 8. Benchmark Results on Referring Segmentation. * denotes reproduced results. It is noted that GLaMM uses extra data from the GranD dataset for pretraining. + denotes our PanCaper model is adapted for referring segmentation task.

on LLaVA-NeXT baseline, we progressively incorporated varying amounts of COCONut-PanCap data (20K, 50K, and 118K (full), as indicated by postfixes in the baseline names) during instruction-tuning. Consistent improvements are observed across all evaluated benchmarks as more of our data is integrated.

Referring Segmentation. In this task, the model processes an image and a textual referring expression to output a segmentation mask corresponding to the expression. The prompt used is, ‘Please segment the ⟨referring text⟩ in the image.’ The target model response is ‘Sure, it is ⟨SEG⟩.’, where the ⟨SEG⟩ token is decoded to obtain the mask. We follow the setup in LISA [28], using multiple segmentation datasets to jointly train the models. Tab. 8 shows the quantitative results. Our model achieves superior performance, particularly when additionally trained with the COCONutPanCap dataset (last row), outperforming all models except GLaMM [50]. This improvement underscores our model’s efficacy in handling complex referring expressions, likely due to the additional data that enhances model generalization and accuracy. It is worth noting that GLaMM performs competitively with our method, though the comparison is uneven given their additional use of the SA-1B dataset [25]. Synthetic vs. Human Annotated Data. Generating synthetic data for captioning has been popular for recent tasks in either training vision encoders [48] or text-to-image generation [31]. We investigate the effect of varying the mix ratio of synthetic captions generated by GPT-4V and our human-annotated data for fine-tuning (where 0 indicates fully synthetic data), using the COCONut-PanCap dataset for training and the COCONut-PanCap-val set for evaluation. We adopt LLaVA-NeXT for the captioning task and

SD3-medium for the image generation task. As shown in Fig. 6, adding 25% human-annotated data yields significant performance improvements in both captioning and generation, with a reduced FID of 26 from 31 (lower is better) and an increased CAPTURE score of 53.6 from 47.5 (higher is better). Consistent improvements are observed as more human-annotated data is incorporated.

[Figure 23]

Figure 6. Varying Synthetic and Human-Annotation Ratios. CAPTURE is used to evaluate the performance of LLaVA-NeXT on detailed captioning, while FID assesses the performance of SD3-medium on text-conditioned image generation.

### 6. Conclusion and Discussion

In this work, we proposed a novel dataset designed to support detailed captioning and grounded segmentation tasks built on COCO images. We demonstrated that our dataset can enhance model performance during instruction tuning and fine-tuning stages across various multi-modal understanding and generation tasks, such as captioning, VQA, grounded segmentation, and text-to-image generation. We

hope that COCONut-PanCap, with its detailed captions grounded with dense panoptic masks, will foster future advancements in multi-modal learning research.

Limitations. High-quality human-labeled data offers significant benefits for instruction tuning in multi-modal tasks, but scaling such datasets is challenging. To address this, we introduce COCONut-PanCap as a starting point for large-scale human-annotated data exploration. Recognizing the relatively smaller dataset size compared to other large dataset, future work may involve using this dataset to train seed models to generate more high-quality synthetic data.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 5, 6

- [2] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL Workshop, 2005. 6
- [3] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In CVPR, 2018. 13
- [4] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In CVPR,

2021. 3

- [5] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2, 3
- [6] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 2, 3, 7
- [7] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 5, 6
- [8] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024. 3
- [9] Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, and Liang-Chieh Chen. Coconut: Modernizing coco segmentation. In CVPR, 2024. 2, 3, 4
- [10] Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption. arXiv preprint arXiv:2405.19092,

2024. 6

- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil

- Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 6
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 7
- [13] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 7
- [14] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. NeurIPS, 2024. 2
- [15] Roopal Garg, Andrea Burns, Burcu Karagol Ayan, Yonatan Bitton, Ceslee Montgomery, Yasumasa Onoe, Andrew Bunner, Ranjay Krishna, Jason Baldridge, and Radu Soricut. Imageinwords: Unlocking hyper-detailed image descriptions. arXiv preprint arXiv:2405.02793, 2024. 2, 3
- [16] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. In NeurIPS, 2023. 3, 7
- [17] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In CVPR,

2019. 3

- [18] Mingfei Han, Linjie Yang, Xiaojun Chang, and Heng Wang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. arXiv preprint arXiv:2311.17043, 2023. 3
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 6, 12
- [20] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv: 2307.06350, 2023. 3
- [21] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 13
- [22] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 13
- [23] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Doll´ar. Panoptic segmentation. In CVPR,

2019. 6

- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 8

- [26] Mikhail V Koroteev. Bert: a review of applications in natural language processing and understanding. arXiv preprint arXiv:2103.11943, 2021. 6
- [27] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 2017. 13
- [28] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, 2024. 5, 6, 8, 12, 13
- [29] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 7
- [30] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 3

- [31] Xianhang Li, Haoqin Tu, Mude Hui, Zeyu Wang, Bingchen Zhao, Junfei Xiao, Sucheng Ren, Jieru Mei, Qing Liu, Huangjie Zheng, Yuyin Zhou, and Cihang Xie. What if we recaption billions of web images with llama-3? arXiv preprint arXiv:2406.08478, 2024. 2, 8
- [32] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. arXiv preprint arXiv:2407.08303, 2024. 2, 3
- [33] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 7
- [34] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In ACL Workshop, 2004. 6
- [35] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3, 4, 7
- [36] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv:2310.03744, 2023. 2, 5, 13
- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3, 5, 12, 13
- [38] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 3, 6, 7, 8, 13
- [39] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023. 7
- [40] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 3, 13

- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016. 13
- [42] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019. 13
- [43] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, 2019. 13
- [44] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, Su Wang, and Jason Baldridge. DOCCI: Descriptions of Connected and Contrasting Images. In ECCV, 2024. 2, 3, 7
- [45] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In NeurIPS, 2011. 3
- [46] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 6
- [47] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV,

2015. 3

- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 8
- [49] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. In CVPR, 2023. 13
- [50] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. In CVPR,

2024. 3, 6, 8, 13

- [51] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In CVPR, 2024. 3, 6, 8
- [52] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [53] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 2, 3
- [54] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A

- benchmark for visual question answering using world knowledge. In ECCV, 2022. 13
- [55] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, 2019. 4
- [56] ShareGPT. ShareGPT. https://sharegpt.com/. 13
- [57] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In ECCV, 2020. 13
- [58] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 7
- [59] Vasu Singla, Kaiyu Yue, Sukriti Paul, Reza Shirkavand, Mayuka Jayawardhana, Alireza Ganjdanesh, Heng Huang, Abhinav Bhatele, Gowthami Somepalli, and Tom Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024. 2
- [60] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [61] Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana Romero-Soriano. A picture is worth more than 77 text tokens: Evaluating clipstyle models on dense captions. In CVPR, 2024. 3
- [62] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 6
- [63] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Max-deeplab: End-to-end panoptic segmentation with mask transformers. In CVPR, 2021. 6
- [64] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 5, 6
- [65] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In ECCV, 2022. 3
- [66] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 4
- [67] Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. k-means Mask Transformer. In ECCV, 2022. 5, 6, 12, 13
- [68] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024. 7
- [69] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel

- understanding with visual instruction tuning. In CVPR, 2024. 3
- [70] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023. 3
- [71] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 13
- [72] Li Zhou, Xu Yuan, Zenghui Sun, Zikun Zhou, and Jingsong Lan. Instruction-guided multi-granularity segmentation and captioning with large multimodal model. arXiv preprint arXiv:2409.13407, 2024. 3

###### patch-level

###### The appendix is organized as follows.

[Figure 24]

[Figure 25]

[Figure 26]

- • In Sec. A, we show implementation details for Detailed Captioning (Sec. A.1), Panoptic segmentation and Grounded (Sec. A.2), and VQA (Sec. A.3).
- • In Sec. B, we show more visualization examples of our proposed COCONut-PanCap dataset (Sec. B.1), and analysis of the tier cases in our dataset annotation user study (Sec. B.2).

[Figure 27]

[Figure 28]

split encode flatten

LLM

[Figure 29]

resize

encode flatten

###### image-level

### A. Experimental Details

(a) LLaVA-NeXt-AnyRes

###### mask-level

In this section, we provide more experimental details for detailed captioning (Sec. A.1), PGC (Sec. A.2), and VQA (Sec. A.3).

[Figure 30]

[Figure 31]

[Figure 32]

…

load

encode

concatenate

…

[Figure 33]

mask

mask-pooling

[Figure 34]

##### A.1. Detailed Captioning

LLM

[Figure 35]

Detailed Captioning Instruction Dataset Construction. The key step in conducting the experiment is constructing the dataset. The original LLaVA-665K dataset consists of LLaVA-158K combined with other VQA datasets. Within LLaVA-158K, a subset of detailed captions corresponds to 23K COCO images. To create our-LLaVA-665K (referred to as LLaVA 665K-COCONut-PanCap in the table), we replace the detailed caption annotations for these 23K COCO images with our annotations. Importantly, the total amount of training data remains unchanged (only the captions for these 23K images are updated), ensuring a fair comparison of the impact of data quality on model performance.

resize

encode flatten

image-level

(b) our LLaVA-NeXt-pool

Figure 7. Comparison of LLaVA-NeXt and our proposed LLaVA-NeXt-pool.

enables more complete region-level feature extraction and is potential in understanding the details better.

##### A.2. PGC

Synthetic Annotation for Detailed Caption. To build the synthetic dataset with state-of-the-art VLM, we use three models, including open-sourced InterVL-2, Qwen2-VL and close-sourced GPT-4V to generate the detailed captions for COCO 118K train set images. We use the same text prompts that is used in LLaVA [37] for prompting the model to create the detailed captions.

We provide more implementation details for the proposed task: Panoptic segmentation and Grounded Captioning (PGC).

PanCaper Implementation Details. We introduce the PanCaper architecture details in this section. Following the architecture in LISA [28], there are three components including the vision backbone, mask decoder and multimodal LLM. Fig. 8 shows the architecture details for PanCaper. We made modification on the vision backbone, and mask decoder part in terms of model architecture. To preserve the learned knowledge of the pre-trained multimodal LLM (i.e., LLaVA-NeXT in our experiments), we leverage LoRA [19] to perform efficient fine-tuning, and completely freeze the vision backbone. The mask decoder is fully finetuned. Additionally, the LLM token embeddings (embed tokens), the LLM head (lm head), and the projection layer are also trainable. The weights of the text generation loss λtext and the mask loss λmask are set to 1.0 and 1.0, respectively. For the PQ-style mask loss, we follow the same settings in kMaX-DeepLab [67], where it consists of mask-level cross entropy loss, dice loss and pixel loss.

LLaVA-NeXT-pool implementation details. Fig. 7 shows the comparison of the original LLaVA-NeXT and our proposed LLaVA-NeXT-pool. As shown in Fig. 7a, in order to preserve the details for the high-resolution images and representations, the original design employs a grid configuration which can also balance the performance efficiency with operational costs. Then both the patch-level and imagelevel features are later concatenated and sent to the LLM. Directly splitting the image into patches could cause prolems, for example, in the figure, the upper part of the dog’s head is partitioned into different patches which may result in incomplete feature extraction for single object. To overcome this drawback, we propose LLaVA-NeXT-pool to extract the dense feature and preserve the object details by utilizing the panoptic segmentation masks in our COCONutPanCap dataset. Fig. 7b shows the details. Compared to the original design, LLaVA-NeXT-pool could effectively extract the features for the dog in our example. Our design

Adapting Baseline Methods for PGC Task. We adopt the same text prompt template to enable the model to perform PGC tasks. For LISA+, we follow the same design

###### image

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

vision backbone

panop c mask decoder

[Figure 42]

❄

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

🔥

[Figure 47]

[Figure 48]

###### predic on

[Figure 49]

[Figure 50]

[Figure 51]

| | |
|---|---|
| | |

[Figure 52]

[Figure 53]

🔥

multi-modal ❄ LLM

[Figure 54]

[Figure 55]

|Please provide a detailed description of the image and segment each part.|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

LoRA

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

[Figure 62]

[Figure 63]

[Figure 64]

🔥: trainable ❄: fronzen

[Figure 65]

The images depicts <p:a running dog[SEG_t]> …. on <p:the grass [SEG_s]>

text prompt

Figure 8. Architecture of PanCaper. We utilize a pretrained vision encoder from kMaX-DeepLab [67] as our vision backbone, which effectively extracts dense features essential for panoptic segmentation.

We focus on the instruction tuning stage by adopting the pretrained weights from the stage-1 across the trainings for all the model variants mentioned in Tab. 7 in the paper. The dataset we used is exactly the same as in LLaVA 665K [36] which includes the earlier version of instruction data proposed in LLaVA 158K [37], ShareGPT [56], VQAv2 [41], GQA [21], openknowledge VQA (OKVQA [42], A-OKVQA [54]), OCR (OCRVQA [43], TextCaps [57]), region-level VQA datasets (Visual Genome [27], RefCOCO [22]). Among these data, LLaVA 158K comprises 77K complex reasoning, 58K conversation and 23K detailed captions. To build the dataset variants shown in Tab. 7, we simply remove the subset of detailed caption 23k, and subsequently add 20K, 50K and 118K COCONut-PanCap dataset to build LLaVA 665K-COCONut-PanCap-20K, LLaVA 665K-COCONutPanCap-50K and LLaVA 665K-COCONut-PanCap-118K. By these steps, we add more detailed caption data to construct the instruction tuning dataset. This results in the total amount of training data of 662K for LLaVA 665K-COCONut-PanCap-20K, 692K for LLaVA 665KCOCONut-PanCap-50K and 760K for LLaVA 665KCOCONut-PanCap-118K. And thus the size of LLaVA 665K-COCONut-PanCap-20K is slightly smaller than the original LLaVA 665K dataset, but the model trained on it yields better performance. For the evaluation settings, we follow the exact settings in LLaVA-NeXT [38] using lmms eval2.

in GLaMM [50] to design the multi entity mask output by utilizing the the GranDf dataset. As the intruction dataset of GranDf is constructed similarly grounding the phrase in the image-level caption, it will output multiple ⟨SEG⟩ tokens. The reasoning results of the number of ⟨SEG⟩ tokens decide the number of output entity mask which are often binary masks. As a result, the model can generate a detailed caption along with interleaved segmentation masks, employing the format “⟨p⟩A man⟨/p⟩⟨SEG⟩ ... next to ⟨p⟩a tree⟨/p⟩⟨SEG⟩”. And thus the format of instruction dataset is significat in task design. Therefore, we formulate our dataset as “⟨p⟩A man⟨/p⟩⟨SEGt⟩ ... next to ⟨p⟩a tree⟨/p⟩⟨SEGs⟩”, where ⟨SEGt⟩ represents the seg token for instance masks of thing and ⟨SEGs⟩ represents for semantic masks of stuff respectively in panoptic setting. Similarly, utilizing the PanCap dataset and special token design, GLaMM [50] is able to generate the entity masks with the tag of ‘thing’ and ‘stuff’.

Training Data Formulation. We adopt the same training data from LISA [28] which comprises mainly three parts, all of which are derived from widely-used public datasets. These include 1) Semantic Segmentation datasets including ADE20K [71], COCO-Stuff [3], and LVIS-PACO [49] part datasets with the generated QA data, 2) Vanilla Referring Segmentation Datasets: refCOCO, refCOCO+, refCLEF [22] and refCOCOg [40] datasets, 3) ReasonSeg dataset [28], and 4) Visual Question Answering Dataset: LLaVA-v1.5-mix665k [36]. To enable the multi-mask generation for grounded caption, there are two options for instruction datasets, GranDf and our COCONut-PanCap where GranDf consists of entity masks while COCONutPanCap consists of panoptic masks.

### B. More Qualitative Results

In this section, we present additional qualitative results of COCONut-PanCap annotations (Sec. B.1) and a detailed analysis of tier cases from the user study (Sec. B.2).

##### A.3. VQA

We provide more implementation details for the VQA experiments. We follow the same setting in LLaVANeXT to create the experimental results for VQA tasks.

2https://github.com/EvolvingLMMs-Lab/lmms-eval

##### B.1. Data Examples

We show more visualization of our proposed COCONutPanCap dataset in Fig. 9 and Fig. 10.

##### B.2. PanCaper and GPT-4V Tier Showcases

In the user study involving 1,000 samples, captions generated by GPT-4V were preferred in 87 cases. Among these, actually, 46 were tier cases where human raters considered both GPT-4V and COCONut-PanCap captions equally good. Fig. 11, Fig. 12 and Fig. 13 illustrate qualitative examples, highlighting the reasons for the tier classification and instances where GPT-4V was chosen.

|[Figure 66]<br><br>The image depicts a natural outdoor with trees and giraffes. <0:The sky is blue>, forming the backdrop of the scene. Below it, there are <1:dense trees, filled with branches and lush green leaves>. Within this environment, two giraffes are prominently featured. The image mainly focuses on <3:a standing giraffe with a long neck and unique patterns> , actively eating leaves from the tree. In<br><br>[Figure 67]<br><br>contrast, there is <4: a second giraffe with similar distinctive patterns>, which is far away from the previous giraffe is resting comfortably on the grass. Both giraffes are surrounded by green trees and <2:a grassy area, predominantly covered with green grass interspersed with patches of exposed brown soil>.<br><br>|
|---|

|The image depicts a dynamic outdoor scene where people are riding horses. In the foreground, two horses take center stage. <9:A black horse with a white mane and tail behind its neck, adorned with a brown bridle, a predominantly dark blue saddle with yellow patterns, and blue leg wraps>, is raising its front hoof. Beside it, <5:a white horse with a black mane and tail, wearing a black bridle, a similar dark blue saddle with yellow patterns, and white leg wraps>, is also raising its front hoof. Both horses are being controlled by <6,7: two man who are dressed in blue and white tops, white pants, and black boots>, actively taming the horses. They are riding horses on <2:a vivid green grassland> that provides the base for the<br><br>[Figure 68]<br><br>action. Adding structure, there is <3: a fence made of wooden posts and railings> in the background. There are some people in the background that are obscured by the horses. For example, there is <8:a person wearing black pants> partially obscured by the white horse and <10:another person in a red top and white pants> who is watching the activity; and <11:a person in a red top and black pants>, partially hidden by the black horse. Together, the elements create a cohesive portrayal of a lively horse-taming event set against a serene natural background. The weather is nice, as <0:the sky is white and cloudless>, forming the backdrop. Below it, there features <1:a dense cluster of trees with brown trunks and green leaves>, framing the scene.<br><br>|
|---|

|[Figure 69]<br><br>The scene includes several individuals actively engaging in skateboarding. There are <6,10: two boys> wearing in green top and black pants>, actively playing <9:11skateboard> in the air. <5: Another guy who is also dressing in green top and black pants> is playing but on the ground. Next to them, there are <7: a half-naked man> observing the skateboarding<br><br>[Figure 70]<br><br>performance, while <8: another guy in a white top and black hat>, also watching the activities. Skateboards are prominently featured in the center area, which includes a black skateboard deck used for tricks. Lastly, the background shows the <0:sky, predominantly blue with scattered clouds>. <1:A light brown building> is obviously seen in the background. The skaters are using the <2:sidewalk, notable for its graffiti and colorful markings>. Around the scene, there is <3:lush green foliage>, adding natural scenery to the skate park.<br><br>|
|---|

|[Figure 71]<br><br>[Figure 72]<br><br>This image showcases a well-organized desk setup. On <1:the wooden desk with a shelf>, there is <4:a DELL computer> occupies the central space, displaying content on its screen. Besides, there is a turtle toy on top of it. Surrounding the computer, multiple items are neatly arranged. To its left, <5:a blue water bottle> stands prominently, next to <6:a book> lying on the desk. Below the computer, <12,13,16,19:additional books> are placed.<br><br>On the upper shelves, various objects add character to the space like <14,15: books> and a drink can. At the top of the shelves, <8:a fluffy blue teddy bear > is positioned on the left, and <11:another teddy bear> is positioned on the right, adding a playful touch. There are various small items as well, like <7:a glass bottle>, <17:books> and a photo frame. In the background, <0:the wall is painted blue>, serving as the backdrop for the scene.<br><br>|
|---|

|[Figure 73]<br><br>[Figure 74]<br><br>The image features a cozy and well-decorated living room. At the center of the room, <4:a wooden coffee table equipped with glasses> holds various items, including <10:a remote control>, <13: a knife> on the plates, and <16:a square small book>. On the left, The seating arrangement includes <14:a patterned couch with colorful cushions and blanket> and <15:another<br><br>neutral-toned couch with vibrant throw pillows> , providing balance to the layout. The rug with colorful patters brings more warm atmosphere to the sitting area. Behind the couch, <20:A chair in the back> complements the seating options. Adding warmth to the room, <8:a black cat> rests comfortably on the couch. Behind the sitting area, there is <5:a 4-layer wall-mounted wooden shelf> with additional decorative items, including <11,12: vases> and other decorative items, enhancing the cozy and inviting atmosphere. Closed to the shelf, there are several <9,19,22:potted plants with green leaves > are placed throughout the room, adding a touch of greenery. <2:The wall painted in warm tones>, create a cozy atmosphere and are adorned with framed artwork and decorations. <0: The floor is neutral-toned>, supporting the entire setup. The <3:ceiling painted white>, contrasts subtly with the walls and reflects the natural light entering the room through <6:the large windows>.<br><br>|
|---|

|[Figure 75]<br><br>[Figure 76]<br><br>The image portrays a lively street scene outside a café. <0:The road> serves as the foreground, where <5:a motorcycle> is prominently parked, its shiny finishes and detailed designs drawing attention. Behind it, <1:the café building>, labeled as “Seaport Cafe” features large windows, a decorative sign, and patriotic bunting. A glowing neon “Corona Light” sign adds to the vibrant atmosphere. The café‘s exterior includes <2:a wooden wall> and <4:a small fence> which separates the outdoor seating area from the street. The seating area is equipped with <12,14:chairs> and <6:a blue umbrella> that provides shade for the patrons. <7,11: Several people are interacting in and around the café>, some standing while others are seated, enjoying their time. <10:One individual who is carrying <9:a black<br><br>backpack> > is joining while there is <8:a guy in blue shirt> is trying to shake hands with her, adding to the dynamic social scene.<br><br>|
|---|

|COCONut-PanCap: Positioned prominently within this image is <1: a zebra, easily distinguishable by its iconic black-and-white stripes and a striped mane along its neck>. Its belly is white, adding contrast to its overall pattern. The zebra is depicted in a natural feeding posture, with its head lowered as it grazes on <0:the grassland which is lush green>, forming the base of the scene.<br><br>|
|---|

[Figure 77]

|GPT-4V: This image features a zebra grazing peacefully on a vibrant green field. The zebra's iconic black-and-white striped pattern stands out sharply against the lush grass, creating a striking visual contrast. Its head is lowered as it nibbles on the grass, showcasing its natural feeding behavior. The mane, with alternating dark and light stripes, adds a touch of elegance to its overall appearance. The bright sunlight enhances the vivid colors and intricate details, highlighting the beauty of the zebra in its natural habitat.|
|---|

tier reason: Compared to COCONut-PanCap, GPT-4V shows more description for the behavior but add hallucination for sunlight.

|COCONut-PanCap: The image mainly features <0:the trees with very lush foliage>. In the front, there are <1,2: two giraffes with distinctive brown spots, yellow patterns, and long necks>. These giraffes are observed to be fighting by entwining their necks with each other. Behind them, there is <3: another giraffe, also with distinctive brown spots and yellow patterns>, which is standing still and partially obscured by the front giraffes.<br><br>|
|---|

[Figure 78]

| |GPT-4V: This image captures two giraffes engaging in a display of interaction, possibly sparring or playfully entwining their necks, a behavior commonly known as "necking." Set against a backdrop of lush green trees, the giraffes' distinctive spotted patterns stand out vividly in the sunlight filtering through the foliage. Their tall, elegant|
|---|---|
|forms and graceful movements highlight the majestic nature of these animals, while the surrounding dense vegetation provides a natural and serene environment. The moment beautifully showcases the unique behavior and striking appearance of giraffes in their habitat.| |

tier reason: Compared to COCONut-PanCap, GPT-4V shows more description for the trees but add hallucination for the sunlight.

|COCONut-PanCap: On <0:the gray asphalt sidewalk>, two motorcycles are parked side by side. <2:One motorcycle is predominantly black with white parts and a brown seat>, positioned to the left of <1:a green bush>, where a registration plate displaying the number “SV-6260” is visible to be mounted on the front fender, and a large cylindrical front headlamp is mounted on the front. <3:The other motorcycle is predominantly silver with a black seat>. The mechanical components include the single-cylinder engine and a straightforward exhaust system.<br><br>|
|---|

[Figure 79]

|GPT-4V: This beautifully restored vintage motorcycle, likely from the early 20th century, showcases a classic design with a cream-colored fuel tank, spoked wheels, and a robust black frame. Key features include a large front headlamp, a registration plate reading "SV-6260," a brown leather saddle, and a small rear storage compartment. The visible singlecylinder engine and minimalistic exhaust system reflect the simple yet durable engineering of the era, with its pristine condition highlighting careful preservation.|
|---|

tier reason: Compared to COCONut-PanCap, GPT-4V shows the story background, but missing details for region-3.

|COCONut-PanCap: At the center of the scene, <5: a cheerful woman is wearing a vibrant swimsuit adorned with purple, blue, yellow, and white patterns>. She is holding <4:a pink umbrella>, which adds a pop of color and shade to the setting. The woman is smiling happily, radiating a joyful and carefree vibe, tying together the natural and human elements seamlessly. There are <1:lush green trees> and <0:blue sky> in the background, adding a natural element to the composition. Below it, there is <2: a serene blue river>, Together, the composition emphasizes the interaction between the lively figure and the tranquil outdoor environment.<br><br>|
|---|

[Figure 80]

|GPT-4V: This vibrant image showcases a cheerful woman holding a bright pink parasol on a sunny day. She is dressed in a colorful floral swimsuit that exudes a retro style, complementing the joyful and carefree atmosphere. Her radiant smile and relaxed pose convey a sense of summer fun, with a backdrop of a clear blue sky, lush green trees, and a calm lake. The parasol adds a playful and vintage charm, enhancing the overall nostalgic feel of the scene.|
|---|

tier reason: : Compared to COCONut-PanCap, GPT-4V recognize the water as lake, which is hard to be distinguished from this image.

|COCONut-PanCap: The image features an owl-shaped candlestick, and a clock are placed on <0:the floor>. The candlestick, designed in the form of an owl, adds a decorative element beside the clock. Prominently, in the center there is <2:a circular clock with hands and Roman numerals>, elegantly designed and placed on an intricate stand featuring ornate patterns and curves. Behind these items, there is <1: a plain and white wall>, providing a neutral background that highlights the objects in the foreground. Together, the objects and their placement create a cohesive and aesthetically pleasing composition.<br><br>|
|---|

[Figure 81]

|GPT-4V: This cozy and artistic composition features a decorative vintage clock paired with a charming owl-shaped candleholder. The clock, with intricate scrollwork and Roman numerals on its face, exudes an antique elegance, while the soft glow of the candle creates a warm and inviting ambiance. The owl candleholder, with its textured surface and whimsical design, adds a playful yet rustic touch to the scene. The subtle lighting and shadows enhance the nostalgic and tranquil atmosphere, making it a perfect display for a quiet, reflective setting.|
|---|

tier reason: Compared to COCONut-PanCap, GPT-4V summarize the style of the image as vintage.

|COCONut-PanCap: The image captures an airplane in flight. The center area highlights <1: an airplane, predominantly white in color, with dark blue English lettering “AIRFRANCE” on its body and distinctive dark blue and red patterns on its tail>. The airplane is depicted flying gracefully through the sky. The background is <0: the sky, characterized by a mix of blue open spaces and scattered clouds>.<br><br>|
|---|

[Figure 82]

|GPT-4V: This image captures an Air France Airbus A380 in mid-flight against a backdrop of cloudy skies. The aircraft, known for its massive size and double-deck design, features the iconic Air France livery with its name prominently displayed on the fuselage and the red, white, and blue stripes on the tail, symbolizing the French flag. The four powerful engines are clearly visible, emphasizing the plane’s engineering prowess. The angle of the shot highlights the sleek design and scale of the A380, showcasing its role as a leader in modern aviation.|
|---|

tier reason: : Compared to COCONut-PanCap, GPT-4V summarize bit more details but with hallucination of model ‘A380’.

|COCONut-PanCap: The image portrays a serene winter landscape. The image mainly highlights <0:a blue sky, creating a clear and bright backdrop>. Below it, there is <1:a pristine snow-covered slope, predominantly white and marked with ski trails> that curve across the terrain. Adjacent to this is <2:a majestic snow-covered mountain>, blending seamlessly into the snowy environment. At the center of the scene, <3:a person who is skiing down the slope, dressed in an orange jacket and black pants> is captured. The skier is equipped with skis attached to their feet, gliding smoothly over the snow. Additionally, <5:a backpack> is carried by the skier, completing their gear.<br><br>|
|---|

[Figure 83]

GPT-4V: This image captures a serene winter landscape featuring a skier ascending a pristine snow-covered slope. The untouched white snow is marked only by the skier's tracks and a winding trail that leads into the distance,

emphasizing the tranquility and solitude of the scene. Majestic mountains with rugged, rocky peaks dominate the background, their shadows adding depth and contrast to the bright sunlight illuminating the snow. The skier, dressed in warm, vibrant gear and carrying poles, navigates the slope with determination, highlighting the spirit of exploration and adventure. The expansive sky above completes the breathtaking alpine setting.

tier reason: Compared to COCONut-PanCap, GPT-4V shows better vocabulary, but the information is almost the same.

[Figure 84]

|COCONut-PanCap: The image highlights <1:a white dog>, standing on the <0:lush green grass>, exuding a lively demeanor. The dog is holding <2:a purple frisbee> in its mouth. The frisbee features black text “SKIMMER” printed on its surface, adding contrast to its vibrant purple color. Together, the scene captures a moment of playful interaction, grounded by the mask numbers for clarity and detail.|
|---|

| |GPT-4V: This image features an adorable white dog standing on a lush green lawn, holding a bright purple|
|---|---|
|frisbee in its mouth. The dog's fluffy coat is well-groomed, and its expressive eyes gaze directly at the viewer, exuding charm and playfulness. The frisbee, labeled with "Skimmer Competition," is slightly chewed, indicating the dog’s enthusiasm for play and activity. The vibrant colors of the frisbee and the green grass create a cheerful and lively atmosphere, perfectly capturing a moment of joy and companionship between a pet and its outdoor playtime.| |

tier reason: Compared to COCONut-PanCap, GPT-4V shows more OCR results of “competition”, but the information is minor.

|COCONut-PanCap: The image captures a dynamic tennis match. In the foreground, the image highlights <3:a man actively playing tennis>, dressed in a black top, blue pants, black socks, and black shoes. He is holding <2:a tennis racket>, characterized by its red and white frame and netting with printed letters. The background is <0:a predominantly blue wall>, decorated with red and black patterns along with white partial text “rates Airline”. <1:The playing field is a mix of green and blue hues>.<br><br>|
|---|

[Figure 85]

|GPT-4V: This image captures a tennis player intensely focused during a match, poised and ready to return a serve on a hard court. Dressed in a dark athletic shirt, blue shorts, and black sneakers, the player holds their racket firmly, leaning slightly forward in a balanced stance that conveys readiness and determination. The backdrop prominently features the USTA logo and the Emirates Airline sponsorship, indicating a professional tennis tournament setting. The court's clean lines highlight the competitive atmosphere|
|---|

tier reason: Compared to COCONut-PanCap, GPT-4V shows extra wording to describe the atmosphere which is not necessary.

