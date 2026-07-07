## BLIP-3: A Family of Open Large Multimodal Models

# arXiv:2408.08872v4[cs.CV]16Sep2025

Le Xue1◦ Manli Shu1◦ Anas Awadalla1,3∗ Jun Wang1∗ An Yan1∗ Senthil Purushwalkam1∗ Honglu Zhou1∗ Viraj Prabhu1∗ Yutong Dai1∗ Michael S Ryoo1∗ Shrikant Kendre1∗ Jieyu Zhang1,3∗ Shaoyen Tseng2∗ Gustavo A Lujan-Moreno2∗ Matthew L Olson2∗ Musashi Hinck2∗ David Cobbley2∗ Vasudev Lal2∗ Can Qin1 Shu Zhang1 Chia-Chih Chen1 Ning Yu1 Juntao Tan1 Tulika Manoj Awalgaonkar1 Shelby Heinecke1† Huan Wang1† Yejin Choi3† Ludwig Schmidt3†

Zeyuan Chen1† Silvio Savarese1† Juan Carlos Niebles1† Caiming Xiong1† Ran Xu1† 1Salesforce AI Research 2Intel Labs 3University of Washington {lxue, ssavarese, jniebles, cxiong, ran.xu}@salesforce.com ◦First Authors; ∗Core Authors; †Senior Authors Project Page

### Abstract

This paper introduces BLIP-3, an open framework for developing Large Multimodal Models (LMMs). The framework comprises meticulously curated datasets, a training recipe, model architectures, and a resulting suite of LMMs. We release 4B and 14B models, including both the pre-trained base model and the instruction fine-tuned ones. Our models undergo rigorous evaluation across a range of tasks, including both single and multi-image benchmarks. Our models demonstrate competitive performance among open-source LMMs with similar model sizes. Our resulting LMMs demonstrate competitive performance among open-source LMMs with similar model sizes, with the ability to comprehend interleaved image-text inputs. Our training code, models, and all datasets used in this work, including the three largescale datasets we create and the preprocessed ones, will be open-sourced to better support the research community.

### 1. Introduction

Large Multimodal Models (LMMs) have attracted significant attention with their potential applications and emergent capabilities. Recent advancements in both proprietary models [25, 63, 64, 82] and open-source LMMs [2, 15, 26, 29, 38, 41, 42, 46, 47, 49, 55, 60, 66, 71] highlight the rapid progress and growing interest in this field. However, despite these advancements, a significant gap remains between open-source and proprietary models, particularly in terms of accessibility to good base models, training recipes, and curated datasets. While there are many strong open-weight models [1, 2, 26], not many of them disclose their training details or release their full training data recipe (including training code). This lack of transparency limits the broader research community’s

ability to replicate, analyze, and improve LMMs. Additionally, many of these models do not contribute new datasets, further restricting progress. These constraints hinder transparency and innovation, preventing open-source communities from fully leveraging and advancing LMM research.

Recent works have demonstrated that large-scale and high-quality data are essential for training robust LMMs [6, 7, 38, 47, 49, 60]. BLIP-2 [42] was one of the pioneering efforts in exploring LMMs, which leveraged synthetic data to achieve impressive results at the time (Figure 1 (a)). However, the data used in BLIP-2 lacks the scale, quality, and diversity required to reach competitive performance compared to more modern LMMs nowadays. In addition, BLIP-2 employs an intricate Q-Former [42] architecture to bridge the vision and language modalities, coupled with a suite of complex training objectives (ITM, ITC, and ITG losses), both of which pose obstacles for larger-scale training. Moreover, BLIP-2 supports only single-image input, whereas interleaved multimodal data formats are the most natural form of multimodal data [37].

In response to these challenges, we introduce BLIP-3 (Figure 1 (b)), a new framework designed to scale up LMM training by utilizing an ensemble of multimodal interleaved datasets, curated caption datasets, and other publicly available datasets [9, 11, 22, 36, 65]. In BLIP-3, as illustrated in Figure 2, we streamline the model architecture by replacing the Q-Former [42] with a more scalable vision token sampler [4] and simplifying the training objectives to focus solely on the auto-regressive loss of text tokens in a multimodal context. Our primary focus is on dataset curation and scaling up the training data. Recently, several high-quality, large-scale datasets have been introduced, such as MINT-1T[6], a trillion-token-scale interleaved dataset, and BLIP3-KALE[7], a knowledge-augmented dataset with high-quality dense captions. In this paper, we introduce

Unified training objective (next-token prediction loss)

Simple and small-scale COCO-style image caption data

Multiple training objectives (ITM, ITC, and ITG losses)

Large-scale comprehensive free-form multimodal interleaved data

Leptis Magna, in modern day Libya, once Africa's premier Roman city. It …

Around a kilometre or two up the road are the circus and the amphitheatre in the…

[Figure 1] A Family of Open Large Multimodal Models_images/imageFile1.png>)

[Figure 2] A Family of Open Large Multimodal Models_images/imageFile2.png>)

[Figure 3] A Family of Open Large Multimodal Models_images/imageFile3.png>)

BLIP-3

sunglasses BLIP-2

A cat wearing

Complex Q-Former

Scalable Vision Token Sampler

(a) BLIP-2 Framework (b) BLIP-3 Framework

Stage-2 Multimodal Pretraining

Interleaved MultiImage Finetuning

Stage-1 Multimodal Pretraining

Single Image Finetuning

Pretrained LLM

BLIP-3 LMMs

100B Multimodal Tokens

30B Multimodal Tokens

6B Multimodal Tokens

4B Multimodal Tokens

(c) BLIP-3 Training Stages

###### Figure 1. We introduce BLIP-3, a framework (b) for developing Large Multimodal Models (LMMs). Our framework improves upon

- BLIP-2 (a) [42] by (1) increasing the richness, scale, and diversity of training data, (2) replacing the Q-Former layers with a more scalable vision token sampler, and (3) simplifying the training process via the unification of the training objectives to a single loss at every training stage. (c) illustrates the stage-by-stage training process, from initializing with a pre-trained LLM to the final LMMs.

three additional specialized datasets: BLIP3-OCR-200M, a large-scale dataset with dense OCR annotations for general base-resolution OCR understanding; BLIP3-GROUNDING50M, a large-scale visual grounding dataset; and BLIP3OCR-HD-30M, a high-resolution dataset with high-quality OCR annotations, designed for high-resolution pre-training.

Loss Loss

…

Vision Tokens Text Tokens Vision Tokens Text Tokens

[Figure 4] A Family of Open Large Multimodal Models_images/imageFile4.png>)

Pre-trained LLM

…

Vision Tokens Text Tokens Vision Tokens Text Tokens

In addition to these datasets, we are committed to open-sourcing the series of models developed in this work, including both the pre-trained base models and instruction-tuned models. Along with the model release, we also provide our training code. By making these resources publicly available, we aim to make LMM research and development more accessible to the community, and we encourage researchers and practitioners to use our models and datasets to understand and further explore the potential and emergent capabilities of LMMs.

[Figure 5] A Family of Open Large Multimodal Models_images/imageFile5.png>)

Token Sampler

Token Sampler

[Figure 6] A Family of Open Large Multimodal Models_images/imageFile6.png>)

##### …

Text Tokenizer Text Tokenizer

[Figure 7] A Family of Open Large Multimodal Models_images/imageFile7.png>)

[Figure 8] A Family of Open Large Multimodal Models_images/imageFile8.png>)

Vision Transformer

Vision Transformer

[Figure 9] A Family of Open Large Multimodal Models_images/imageFile9.png>)

Around a kilometre or two up the road are the circus and the amphitheatre in the second part of the Leptis Magna complex. The amphitheatre was built to seat up to 16,000 spectators who would come to be entertained…

Leptis Magna, in modern day Libya, once Africa's premier Roman city. It is one of the greatest archeological sites in the whole Mediterranean. If Leptis Magna were in Tunisia or Morocco or Egypt…

[Figure 10] A Family of Open Large Multimodal Models_images/imageFile10.png>)

…

Figure 2. Overview of the BLIP-3 architecture. Free-form interleaved images and texts from the ensembled interleaved and caption datasets are input into the framework, with each modality undergoing a separate tokenization process to be fed into the pre-trained LLM in natural order. A standard auto-regressive loss is then applied to the text tokens. The Vision Transformer is kept frozen during training, while all other parameters, including the vision token sampler and the pre-trained LLM, are trained.

### 2. Related Work

Recent advancements in Large Multimodal Models (LMMs) have explored different architectural designs. We roughly divide them into two main categories: the cross-attention style [4, 5] and the self-attention style. The cross-attention approach, exemplified by models like Flamingo [4, 5] and Llama 3.1 [55], integrates vision and language modalities by injecting visual information into layers inside LLMs using a cross-modal cross-attention module. The self-attention approach [8, 10, 14, 15, 17, 23, 24, 30, 42, 48, 49, 67, 70, 77, 78], on the other hand, offers a more streamlined solution. This approach connects pre-trained language models to visual inputs using lightweight connectors. Both visual and textual inputs are tokenized and concatenated in the same input sequence to LLM, and the LLM learns to align both modalities through self-attention. The self-attention design has

become prevalent recently, which has been shown to be effective on a wide range of LMMs [1, 2, 8, 50, 76] . For BLIP-3, we also adopt the self-attention design due to its simplicity.

Training methodologies for LMMs typically follow one of the two strategies. The first one uses a light pre-training procedure and heavily relies on visual instruction tuning, as seen in the LLaVA series [48, 49]. Extensive research has been conducted on creating effective instruction-tuning data for a variety of tasks [12, 24, 40, 43, 73, 86]. The second

strategy involves extensive pre-training on large-scale, diverse datasets, followed by visual instruction fine-tuning. This approach, used by models such as MM1 series [60, 82] and Idefics2 [38], infuses a broad knowledge into the model during the pre-training stage, and then fine-tune it for better instruction following ability and better alignment. While MM1 series [60, 82] provides extensive ablations and studies on the recipes aimed at improving LMMs, it releases limited resources for practitioners to reproduce the model (their models and datasets are not open-sourced).

In this work, we introduce BLIP-3, an open LMM family encompassing a series of models, data recipes, training code, and three new large-scale foundational multimodal datasets. Unlike previous works, BLIP-3 is designed to enable and advance future research in this area by providing comprehensive resources for the community.

### 3. Model Architecture

Architecture Overview. As illustrated in Figure 2, the

- BLIP-3 framework adopts an architecture consisting of a ViT [19, 81], a vision token sampler (perceiver resampler [4]) to downsample the image embeddings, and a pre-trained LLM (Phi-3 series [1]). The input to the model can be free-form multimodal interleaved texts and vision tokens from the diverse multimodal data sources we ensemble.

Any-Resolution Vision Token Sampling. As proved effective in recent LMMs [16, 50, 83], we adopt a dynamic highresolution (i.e., “any-resolution”) image encoding strategy at both the Stage-2 pre-training and the fine-tuning stages. We enable higher-resolution image understanding with image patch-wise encoding. The patch-wise encoding preserves the resolution of the original image as much as possible by splitting a single image into multiple patches and encoding them separately. Following the previous convention, we concatenate the encoded image patches with a downsized original image that provides global information.

In the VL connector, we use a perceiver resampler to downsample the vision tokens. With any-resolution image encoding, we perform the downsampling for each image patch (including the downsized original image) independently. The downsampler vision tokens are then concatenated together and sent to the LLM. With the downsampling in our VL connector, we can reduce the sequence length of vision tokens by a factor of five or more depending on the number of query tokens in the perceiver resampler. We provide ablation studies on different token sampling strategies in Section 7.3.

- 4. Training

Stage-1 Base Resolution Pre-training. The first stage of pre-training focuses on base resolution image-text align-

ment using a diverse dataset mixture comprising open-source datasets and two newly created datasets from BLIP-3 – BLIP3-OCR-200M and BLIP3-GROUNDING-50M, all of which will be made publicly available. In this stage, the base model is trained on approximately 100 billion multimodal tokens from the ensembled dataset. To ensure computational efficiency, the pre-training resolution is set to 384×384 pixels, aligning with the default resolution of SigLIP [81].

Stage-2 High Resolution Pre-training. Building upon the pre-trained checkpoint from Stage 1, the second stage incorporates additional high-quality, high-resolution datasets, including one newly created dataset from BLIP-3 – BLIP3OCR-HD-30M, to further enhance the model’s capability. To maintain consistency with the supervised fine-tuning (SFT) stage, we adopt the same any-resolution vision token sampling strategy introduced in section 3. We set the any-resolution grid to support up to 12 image patches with a maximum side length of 1536 pixels on the longer side. Further details and ablation studies are provided in section 7.2.

Single-Image Supervised Fine-tuning. Then, we fine-tune the pre-trained model on a collection of publicly available instruction-following datasets [38, 48, 71]. During finetuning, we employ the same any-resolution vision token sampling strategy as in Stage 2, enabling the model to process high-resolution images effectively, including text-rich document-style data. The following sections provide additional technical details on the fine-tuning procedure.

Interleaved Multi-Image Supervised Fine-tuning. We conduct a second-stage fine-tuning on the instruction finetuned model on a mixture of multi-image and single-image instruction-following samples. The goal of this second-stage fine-tuning is to enhance the model’s ability to comprehend interleaved image-text input, which is helpful for multimodal in-context learning, multi-image question answering, and many more practical use cases. For the multi-image finetuning, we also adopt the any-resolution vision token sampling strategy, the same as in the previous SFT stage.

### 5. Data

#### 5.1. Pre-training Data Recipe

Our Stage-1 pre-training recipe is detailed in Figure 3. In BLIP-3, we pre-train on a diverse set of multimodal datasets using the specified sampling ratios.

For Stage-2 pre-training, we use the Stage-1 data recipe as a base while incorporating specialized high-resolution, text-rich datasets. Specifically, we replace BLIP3-OCR200M with BLIP3-OCR-HD-30M and introduce the IDL [9] dataset, each making up 10% of the total training data. To preserve the overall structure of the dataset mixture, we adjust the sampling ratios of the remaining caption datasets by reducing BLIP3-KALE from 25% to 10%, Datacomp-1B

from 10% to 2.5%, and BLIP3-GROUNDING-50M from 5% to 2.5%, while keeping the rest unchanged.

Interleaved Dataset Mixture. We combine MINT-1T (including its HTML, PDF, and ArXiv subsets) with OBELICS (HTML only) to create a more diverse and comprehensive dataset mixture that covers a broader range of domains.

- 1. MINT-1T [6]. A trillion-token-scale multimodal interleaved dataset, containing data sources from HTML, PDF, and ArXiv. As evidenced by MM1 [60] and Idefics2 [38], such multimodal interleaved datasets are essential for scaling up large multimodal model training and enabling fundamental capabilities like multimodal in-context learning. In BLIP-3 we mix its HTML, PDF, and ArXiv subsets in a 7:5:1 ratio.
- 2. OBELICS [38]. A multimodal interleaved dataset constructed solely from HTML documents. It differs slightly in domain coverage from MINT-1T due to the specific preprocessing steps adopted.

Caption Dataset Mixture. We integrate a diverse range of caption datasets, with the following details.

- 1. BLIP3-KALE [7]. An open-source large-scale curated high-quality caption dataset, which we use as a base for general detailed caption datasets.
- 2. BLIP3-OCR-200M. A curated large-scale OCR dataset with 200 million images from Datacomp- 1B[22]. For each image, we use the off-the-shelf OCR engine [28] for OCR annotation. Text segments in a caption like "... text ..." are modified to include OCR information as "... text ( ocr_info ) ...", where ocr_info contains normalized bounding box coordinates for the extracted text, specifying its exact position within the image in the format

"<bbox>x1,y1,x2,y2</bbox>". We have multiple granularities of OCR information, including with and without bounding box data. In BLIP-3 pre-training, we only utilize textual information without bounding box data.

- 3. BLIP3-GROUNDING-50M. A curated large-scale grounding dataset to enhance the ability to ground semantic concepts in visual features, which is crucial for tasks like understanding referring expressions [79] (e.g., "the object to the left of the dog"). We curate a dataset of 50 million images from Datacomp-1B [22]. For each image, we identify objects and their location information using one of the state-of-the-art open-world image tagging [85] and object detection models [51]. Objects mentioned in a caption like "... object ..." are modified to include grounding information as " ... object ( grounding_info ) ...", where grounding_info contains bounding box information in one of three formats, each capturing a different granularity of localization: (1) <bbox>x1,y1,x2,y2</bbox>, (2) "starts at (x1,y1) and extends up to (x2,y2)", or (3) "top-left corner of the image".

- 4. BLIP3-OCR-HD-30M. A high-resolution OCR dataset curated for Stage-2 pre-training. Different from BLIP3OCR-200M, which focuses on more general OCR data with fewer resolution constraints, this dataset is filtered to only contain images of which both the height and width are larger than 512 pixels. On the selected high-resolution images, we run PaddleOCR with a high-resolution setup to obtain more accurate OCR annotations. Our empirical study shows that running the OCR annotation process with a high-res setting can help get a more complete and accurate set of OCR annotations that matches its resolution, which is more useful for the Stage-2 pre-training. This process yields a total of 30 million high-resolution samples with detailed OCR annotations that match the image quality.
- 5.Other Datasets Mixture. We also include other publicly available datasets such as uncurated Datacomp-1B [22] image-text pairs, CC12M [11], CC3M [11], VG [36], SBU [65], and IDL [9].

#### 5.2. Supervised Fine-tuning Data Recipe

The datasets used in the fine-tuning stage are from public datasets of different domains [12, 31, 33–35, 38, 48, 56, 58, 59, 61, 68, 69, 71, 75, 84]. In addition to the multimodal image-text data, we also mix in pure text instructionfollowing data [45, 62] during visual instruction tuning. Ultimately, we collect a mixture of 3 million publicly available instruction-tuning samples, on which we fine-tune our model for one epoch.

The multi-image instruction tuning stage starts with a model fine-tuned on single-image data. We use a mixture of public multi-image / interleaved image-text instruction data [32, 54]. To prevent the model from deteriorating on single-image capabilities, we reuse a subset of single-image datasets used in the previous fine-tuning stage and mix them into the multi-image training data.

### 6. Experiments

We evaluate our models (4B and 14B) on a comprehensive suite of multimodal benchmarks, assessing the model’s ability from multiple perspectives. Our evaluation covers general VQA benchmarks [3, 13, 20, 39, 52], domain knowledge [57, 80], OCR ability [53, 69], and hallucination [27, 44]. For models fine-tuned on interleaved multiimage datasets, we also evaluate their performance on common multi-image benchmarks [21, 32, 72, 74].

Single-Image Evaluation. In Table 1, we compare models of comparable sizes, including both closed-source [63, 82] and open-source models [1, 2, 47, 55, 71, 76]. We report individual benchmark scores as well as the overall average score across all benchmarks, following the standard practice.

1https : / / huggingface . co / spaces / opencompass / open_vlm_leaderboard

|“NY”, “BURGER”, “PORK&GRILL”, “www.bigstock.com 206857405”<br><br>[Figure 11] A Family of Open Large Multimodal Models_images/imageFile11.png>)|
|---|

|Please note the balance of euro states trade: Italy in surplus, Germany deficit. Another world indeed. Consider that Italian public debt was not different than today, but it was not a matter of concern, the shock did reflect on the exchange rate, no one was thinking about selling government bonds under par…Please check,<br><br>the following graph: red is gdp% shift in private debt and blue public debt from 1999 to 2007, so much for another of the myths of this crisis, the one that says that “the fault is of the public debt"…<br><br>[Figure 12] A Family of Open Large Multimodal Models_images/imageFile12.png>)<br><br>[Figure 13] A Family of Open Large Multimodal Models_images/imageFile13.png>)|
|---|

OBELICS MINT-1T BLIP3-KALE BLIP3-GROUNDING BLIP3-OCR Datacomp-1B CC12m CC3m SBU VG

[Figure 14] A Family of Open Large Multimodal Models_images/imageFile14.png>)

[Figure 15] A Family of Open Large Multimodal Models_images/imageFile15.png>)

[Figure 16] A Family of Open Large Multimodal Models_images/imageFile16.png>)

[Figure 17] A Family of Open Large Multimodal Models_images/imageFile17.png>)

[Figure 18] A Family of Open Large Multimodal Models_images/imageFile18.png>)

[Figure 19] A Family of Open Large Multimodal Models_images/imageFile19.png>)

[Figure 20] A Family of Open Large Multimodal Models_images/imageFile20.png>)

[Figure 21] A Family of Open Large Multimodal Models_images/imageFile21.png>)

[Figure 22] A Family of Open Large Multimodal Models_images/imageFile22.png>)

[Figure 23] A Family of Open Large Multimodal Models_images/imageFile23.png>)

SBU 1.25% VG 1.25%

CC3m 1.25%

CC12m 1.25%

[Figure 24] A Family of Open Large Multimodal Models_images/imageFile24.png>)

[Figure 25] A Family of Open Large Multimodal Models_images/imageFile25.png>)

[Figure 26] A Family of Open Large Multimodal Models_images/imageFile26.png>)

[Figure 27] A Family of Open Large Multimodal Models_images/imageFile27.png>)

[Figure 28] A Family of Open Large Multimodal Models_images/imageFile28.png>)

[Figure 29] A Family of Open Large Multimodal Models_images/imageFile29.png>)

Datacomp-1B 10%

|[Figure 30] A Family of Open Large Multimodal Models_images/imageFile30.png>)<br><br>The image contains garment (at the bottom right corner), t shirt polo shirt (in the center), neckband collar (at the top left corner), neckband (to the right of the center), neckband collar (above the center)|
|---|

OBELICS 17.5%

[Figure 31] A Family of Open Large Multimodal Models_images/imageFile31.png>)

[Figure 32] A Family of Open Large Multimodal Models_images/imageFile32.png>)

BLIP3-OCR 5%

[Figure 33] A Family of Open Large Multimodal Models_images/imageFile33.png>)

BLIP3-GROUNDING 5%

[Figure 34] A Family of Open Large Multimodal Models_images/imageFile34.png>)

[Figure 35] A Family of Open Large Multimodal Models_images/imageFile35.png>)

[Figure 36] A Family of Open Large Multimodal Models_images/imageFile36.png>)

MINT-1T 32.5%

BLIP3-KALE 25%

|[Figure 37] A Family of Open Large Multimodal Models_images/imageFile37.png>)<br><br>[Figure 38] A Family of Open Large Multimodal Models_images/imageFile38.png>)<br><br>We aim at calculating the path integral over the variations of the string around the bounce configuration, which involves in particular…<br><br>[Figure 39] A Family of Open Large Multimodal Models_images/imageFile39.png>)<br><br>In terms of the introduced variables the action ( a0 ) can be written in the quadratic approximation in the deviations from the bounce …<br><br>The condition ( appl_cond ) for applicability of the effective action ( a0 ) requires that ….|
|---|

|[Figure 40] A Family of Open Large Multimodal Models_images/imageFile40.png>)<br><br>"A shabby chic country kitchen design is showcased in this image, featuring a decorative canister set with a vibrant rooster illustration. The container, which may be made of ceramic or metal, boasts a rustic, weathered appearance with a metal handle and a lid adorned with a curved metal loop. The rooster is depicted in rich colors, including red, blue, and yellow, against a background of faded, handwritten-style text and designs. The container rests on a wooden surface, and the image bears a watermark from Farmhouse Temptations at its bottom right corner."|
|---|

###### Figure 3. Overview of BLIP-3 Stage-1 pre-training datasets base recipe.

MMMU(val)

MME(norm)

SEED-IMG

OCRBench

MMB(dev)

MathVista

HalBench

TextVQA

MMStar

Average

MMVet

RWQA

POPE

Accessibility

Model

openweight

opendata

opencode

data contribution

GPT-4V [63] × × × × 72.0 80.8 49.7 63.3 61.4 49.0 53.8 48.2 78.0 67.8 81.8 39.3 62.1 3B-5B Models

MM1.5-3B [82] × × × × 72.4 - - 64.2 56.9 41.0 37.1 44.4 76.5 65.7 88.1 - Phi-3-vision [1] ✓ × × × 71.0 73.6 47.9 55.3 58.8 43.2 46.1 45.1 73.3 63.7 83.5 39.0 58.4 Phi-3.5-vision [1] ✓ × × × 69.6 81.9 47.3 65.6 53.5 43.2 43.0 43.9 72.0 59.9 86.1 39.6 58.8 MiniCPM-V-2.0 [76] ✓ × × × 67.1 69.1 39.1 64.6 55.8 41.0 38.2 39.8 73.2 59.6 86.3 36.1 55.8 VILA-1.5-3B [47] ✓ ✓ ✓ × 67.9 62.4 40.3 58.5 53.3 38.5 34.1 30.6 58.1 43.7 86.9 31.2 50.5 BLIP-3-4B (ours) ✓ ✓ ✓ ✓ 71.9 74.7 45.3 62.5 61.0 41.2 42.3 44.5 76.5 66.0 87.9 36.5 59.2

10B-15B Models

Pixtral-12B [2] ✓ × × × 71.5 77.9 54.5 68.6 65.4 58.5 51.1 56.3 75.7 72.2 84.2 47.0 64.9 Llama-3.2-11B-V-inst. [55] ✓ × × × 72.7 68.0 49.8 65.0 63.3 57.6 48.0 47.7 54.1 75.3 88.1 40.3 60.8 VILA-1.5-13B [47] ✓ ✓ ✓ × 68.0 74.4 44.2 61.4 53.3 45.0 41.1 42.3 61.1 45.7 85.0 39.3 55.1 Cambrian-13B [71] ✓ ✓ ✓ ✓ 73.2 73.2 47.1 67.0 58.6 48.9 41.6 47.7 72.8 61.0 86.8 39.4 59.8 BLIP-3-14B (ours) ✓ ✓ ✓ ✓ 73.4 77.5 48.7 68.1 64.4 44.5 47.0 45.4 79.0 71.1 86.0 38.7 62.0

- Table 1. Evaluation on single-image benchmarks. For benchmarks where a open-source model doesn’t report its official score, we use the score reported in a third-party leaderboard1or run it ourselves using the evaluation codebase [18] for a fair comparison. We also include the GPT-4V (gpt-4-1106-preview) performance (provided by the evaluation codebase) as a reference in the first row.

Multi-Image Evaluation. In Table 2, we compare BLIP- 3 single-image (SI) model with BLIP-3 interleaved multi-

|Our BLIP3-OCR-200M caption:<br><br>Level 1: "MACK SHOP"<br>Level 2: "MACK SHOP" located above the center<br>Level 3: "MACK SHOP" located at 2/17 of the image from top to bottom and 1/2 of the image from left to right<br>Level 4: "MACK SHOP" approximately starts at [0.2, 0.0] and extends up to [0.8, 0.2]<br>Level 5: "MACK SHOP" starts at [0.192, 0.037] and extends up to [0.817, 0.203]<br>Level 6: "<ocr>MACK SHOP</ocr><bbox>[0.192, 0.037][0.817, 0.203]</bbox>"<br>|
|---|

[Figure 41] A Family of Open Large Multimodal Models_images/imageFile41.png>)

[Figure 42] A Family of Open Large Multimodal Models_images/imageFile42.png>)

[Figure 43] A Family of Open Large Multimodal Models_images/imageFile43.png>)

|Our BLIP3-GROUNDING-50M Coarse Grounding Caption: This image showcases a football player (in the center) in action, wearing a red jersey (in the center) with the logo number (in the center) '11' and the logo of the 'Falcons'. The player is holding an NFL football (at the bottom left corner) in one hand (at the bottom left corner) and is gesturing with the other. The helmet (above the center) has a face mask obscuring the player's face, and the background is blurred, emphasizing the player in the foreground.<br><br>Our BLIP3-GROUNDING-50M Fine-grained Grounding Caption: This image showcases a <object>football player</object><bbox>[0.015, 0.005][0.845, 0.993]</bbox> in action, wearing a <object>red jersey</object><bbox>[0.205, 0.207][0.638, 0.994]</bbox> with the logo …… The <object>helmet</object><bbox>[0.287, 0.002][0.523, 0.449]</bbox> has a face mask obscuring the player's face, and the background is blurred, emphasizing the player in the foreground.|
|---|

|Our BLIP3-OCR-HD-30M caption: Level 1: "Juice Factory", "<Invoices", "New Invoice", "Help", "Save", "invoicely", "SENT", "Estimate", "Upgrade", "Juice", "We appreciate your business. Thank you for shopping at our store.", "Dashboard", "Invoice No.", "Language", "Currency", "Invoices", "INV-694", "English (US)", "Euro - EUR", "Estimates", "From", "Date", "Juice Factory", "2016-04-23", "Track", "200 Juicy Road, Atlanta", "Georgia, United States", "Invoice Due", "Connections", "Immediately", "To", "Files", "Example, Inc.", "Purchase Order Number", "80 Mortimer Street", "PO-123", "Trash", "London W1W 7FE", "United Kingdom", "Name", "Quantity", "Rate", "Amount", "Iron Lady Juice", "12", "3.50", "EUR", "42.00", "New Line", "Sub Total", "42.00", "Shipping Cost", "8.00", "8.00", "Total (EUR)", "50.00", "Total Due", "EUR", "50.00", "Email: example@company.com", "Phone:<br><br>+1-541-754-3010"|
|---|

- Figure 4. Samples from BLIP3-OCR-200M. We extract six levels of OCR granularity, with and without bounding boxes. OCR-related captions in BLIP-3 are preprocessed to remove filler phrases like ’the text’ within the dataloader, which we find improving OCR benchmarks’ performance. Samples from BLIP3-OCR-HD-30M. Structured similarly to BLIP3-OCR-200M, this dataset contains high-resolution images annotated with high-resolution OCR tools for enhanced high resolution OCR understanding. Only Level 1 is shown for simplicity. Samples from BLIP3-GROUNDING-50M. A large-scale dataset of images and corresponding captions containing localization information about objects. Furthermore, we release the associated object bounding box data to facilitate the creation of captions with custom templates.

Model BLINK QBench MuirBench Mantis-Eval GPT-4V [63] 51.1 73.4 - 62.7 3B-5B Models

VILA-1.5-3B [47] 39.8 51.7 29.2 41.9 BLIP-3-4B-SI 42.9 51.9 35.0 49.3 BLIP-3-4B-MI 47.1 69.6 38.0 56.2

10B - 15B Models

VILA-1.5-13B [47] 48.1 61.2 37.7 49.3 BLIP-3-14B-SI 47.1 55.4 49.1 55.3 BLIP-3-14B-MI 53.9 73.4 56.2 61.3

- Table 2. Evaluation on multi-image benchmarks. VILA-1.5 models are evaluated using the same evaluation code as our models.

image (MI) model on multi-image benchmarks. Although the former is fine-tuned from a pre-trained model that can

comprehend interleaved image-text data, it performs poorly on multi-image benchmarks, possibly because the singleimage SFT causes degradation in such ability. With interleaved multi-image SFT, we see significant improvements.

### 7. Ablation Studies

#### 7.1. Ablation on Stage-1 Pre-training

Few-shot Pre-training Evaluation. To analyze the impact of pre-training configurations without the impact of finetuning data, we conduct ablation studies on pre-trained models with few-shot evaluation. Following [60], we randomly sample a few-shot subset from the training set and evaluate model performance. We report CIDEr scores for captioning tasks and accuracy for VQA to provide a clear comparison of few-shot performance across different pre-training

configurations.

Scaling Pre-training Data. For Stage-1 pre-training, we perform an ablation study to explore the relation between the amount of pre-training data and the pre-train few-shot evaluation metrics, by varying the data scale from 2B multimodal tokens to 100B multimodal tokens. The data recipe we used here is a mixture of image caption datasets and multimodal interleaved data as detailed in 3. As shown in Figure 5, we find that scaling up the number of multimodal tokens from 2B to 60B leads to substantial gain for image-text (COCOCaps) and OCR (Text-Caps) tasks, and further increasing the data size to 100B has moderate additional benefit in terms of few-shot evaluation metrics.

110

90

105

Cider

80

Cider

Text Caps 4-shot Text Caps 8-shot

COCO Caps 4-shot COCO Caps 8-shot

100

70

95

4 10 20 40 60 100 Number of training tokens(B)

4 10 20 40 60 100 Number of training tokens(B)

(a) COCO-Caps

(b) Text-Caps

- Figure 5. Few-shot performance given different scales of pretraining data in Stage-1 pre-training.

Visual Backbones. We also explore if different visual backbones have an impact on the performance of visionlanguage tasks. We compare two types of visual encoders, DFN [19] and SigLIP [81]. As shown in 3, we find SigLIP provides better visual representations that boost performance on OCR tasks, and we adopt SigLIP in the final model architecture as the ViT backbone. To ensure computational efficiency and a fair comparison, all models in this ablation study are pre-trained with 10B tokens.

Visual Backbone Text-VQA OK-VQA COCO-Caps Text-Caps DFN 41.1 / 41.9 48.4 / 49.5 107.2 / 109.4 78.2 / 79.9 SigLIP 49.1 / 50.5 48.4 / 48.9 108.7 / 110.2 84.7 / 88.6

- Table 3. Few-shot (4-shot / 8-shot) performances for different visual backbones at Stage-1 pre-training.

#### 7.2. Ablation on Stage-2 Pre-training

In this section, we investigate the impact of Stage-2 pretraining. Model performance is evaluated by fine-tuning on a smaller subset (∼ 1M samples) of our instruction tuning dataset for better computational efficiency. For evaluation, we focus on two representative LMM capabilities: general VQA and OCR, each of which is an averaged score over multiple relevant benchmarks.

Impact of Adding Stage-2 Pre-training and with Different Resolutions. As shown in Table 4, we use our

100B token Stage-1 pre-trained model as the baseline. We then add Stage-2 pre-training on top of it; we use half the data samples for a full Stage-2 pre-training run to maintain computational efficiency. Furthermore, we examine how varying Stage-2 pre-training resolutions (i.e., different numbers of image patches) affect performance. To ensure a fair comparison, all models in Table 4 are fine-tuned with a maximum of 13 image patches and follow the same fine-tuning protocol. From the results, we observe that incorporating Stage-2 pre-training leads to a significant improvement in OCR, while general VQA performance remains stable.

Stage-2 Stage-2 Maximum Resolutions OCR GeneralVQA

× N/A 64.6 63.5 ✓ (768x768), (384x1152), (1152x384) 65.3 63.2 ✓ (1152x1536), (1536x1152) 67.2 63.3

- Table 4. Stage-2 pre-training and its resolution impact. We ablate two resolution grid settings for Stage-2 pre-training: one with up to 4 image patches, at most arranged as (2×2, 1×3, or 3×1), and another with up to 12 image patches, at most arranged as (3×4 or 4×3), with each patch sized at 384×384 pixels.

7.3. Ablation on Instruction Tuning

The ablation study in this subsection focuses on several model design choices. Evaluations are conducted on singleimage fine-tuned checkpoints. Similar to section 7.2, models in this ablation are fine-tuned on a smaller SFT subset for better computational efficiency, so the results are not directly comparable to our main results.

Perceiver Resampler vs. MLP. We experiment with two designs of vision token sampler with similar “vision compression ratios”. Both models are trained from Stage-1 pre-training. For perceiver resample, we use 128 query tokens per image patch to achieve a compression ratio of rc = 729/128 = 5.7 . With MLP projector, to get a similar compression ratio, we append a 2x2 max pooling layer to it to have rc = 4. In Table 5, we compare the two models on benchmarks focusing on various domains and report the average score on each domain. Perceiver resampler, with a slightly higher compression ratio, has a lower score on OCR, but performs better in other domains, and has a better overall score. We therefore choose to use perceiver resampler for BLIP-3, because the higher compression ratio is crucial for efficiency when handling interleaved multi-image input.

Vision Token Sampler GeneralVQA OCR Sci.& Math Hallucination

Perceiver resampler 63.1 70.4 43.4 62.2 MLP + 2x2 pooling 61.7 72.2 41.8 62.7

- Table 5. Ablation study on vision token sampler. We experiment with two sampling designs with similar vision compression ratio.

[Figure 44] A Family of Open Large Multimodal Models_images/imageFile44.png>)

[Figure 45] A Family of Open Large Multimodal Models_images/imageFile45.png>)

[Figure 46] A Family of Open Large Multimodal Models_images/imageFile46.png>)

[Figure 47] A Family of Open Large Multimodal Models_images/imageFile47.png>)

[Figure 48] A Family of Open Large Multimodal Models_images/imageFile48.png>)

* Zoom in for illustration.

User: Which one of the following can be found in the first image? A. B.

User: Where does the object in image 1 appear in image 2?

[Figure 49] A Family of Open Large Multimodal Models_images/imageFile49.png>)

[Figure 50] A Family of Open Large Multimodal Models_images/imageFile50.png>)

User: How much of BLIP-3's pre-training data is from MINT-1T?

BLIP-3: The object in image 1, a screwdriver, appears in image 2 on the left side, near the top left corner.

BLIP-3: B

BLIP-3: MINT-1T contributes 32.5% of BLIP-3's pre-training data.

- Figure 6. Example model outputs of BLIP-3-4B-MI (interleaved multi-image model). The model is capable of understanding interleaved image-text input and user queries about multiple images while maintaining the performance on single-image QAs.

Any-Resolution Vision Token Sampling. Our anyresolution strategy differs from previous work [50] in that every group of image embeddings (of the same image patch) is downsampled with a perceiver resampler to ensure efficiency. In this section, we ablate the effectiveness of our any-resolution strategy by comparing it with other designs. Models in this study are fine-tuned from the Stage-1 pretrained base resolution models.

|resolution| | |
|---|---|---|
|anyres-fixed-sampling anyres-fixed-sampling anyres-patch-sampling|(ntok=128) (ntok=256)| |
|Char<br><br>OCR<br><br>(a)<br><br>SFT<br><br>token samp|tQA OCRB<br><br>Benchmarks<br><br>ablation ling|ench|

DocVQA

30

35

40

45

50

55

60

65

70

Scores

base

Perception OCR Sci. & Math Hallucination

Evaluation Domains

45

50

55

60

65

70

75

AverageScores

BLIP-3

BLIP-3 (instruction-aware)

(b)

- Figure 7. studies. (a). Comparison of different vision strategies on OCR benchmarks. (b). Comparison between our model and its “instruction-aware” alternative. For each evaluation domain in Figure (b), we report the average score on multiple relevant benchmarks.

downsampled vision tokens. The fixed sampling strategy, although it shows improvements over the base resolution baseline, is not as good as the patch-wise sampling. We suspect two reasons for this: (a) With fixed sampling, the vision token compression ratio can be too high to retain text-rich visual information. (b) The perceiver resampler may not work well with a concatenation of different image embeddings.

Instruction-Aware Vision Token Sampling. InstructBLIP [15] proposes an instruction-aware Q-Former [42] for vision token sampling and proves its effectiveness on certain benchmarks. With the perceiver resampler, we experiment with a similar strategy by appending text instruction tokens to the query tokens of the perceiver resampler. This enables the instruction (text tokens) to interact with both query tokens and image embeddings via cross-attention.

From the comparison in Figure 7b, we do not observe a significant difference between our model and its instructionaware version on common task domains. Because of the little difference we observe in this ablation study, we keep the original perceiver resampler architecture in our model for simplicity. We leave the further exploration of the “instructionaware” VL connector to future works.

The “fixed-resolution” baseline resizes all images to the default input size of the vision encoder while keeping the original aspect ratios. Another downsampling strategy with the perceiver resampler is “fixed sampling” ( anyres-fixed-sampling), for which we concatenate the image embeddings from all image patches and then input them as a single sequence to the perceiver resampler to obtain the fixed number of vision tokens for the whole image.

### 8. Conclusion

We introduce BLIP-3, a comprehensive framework for training a series of open-source large multimodal models on a curated mixture of large-scale datasets. BLIP-3 LMMs achieve competitive results on a range of multimodal benchmarks. By open-sourcing BLIP-3 (4B and 14B), our curated datasets, and our training code, we hope to empower the research community with better accessible multimodal foundation models and datasets, allowing practitioners to explore further and advance the potential and emergent abilities of LMMs.

Our evaluation of this design focuses on text-rich OCR benchmarks. From Figure 7a, we see significant improvements with the any-res image encoding strategy even with

### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Qin Cai, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Yen-Chun Chen, Yi-Ling Chen, Parul Chopra, Xiyang Dai, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Victor Fragoso, Dan Iter, Mei Gao, Min Gao, Jianfeng Gao, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Ce Liu, Mengchen Liu, Weishung Liu, Eric Lin, Zeqi Lin, Chong Luo, Piyush Madan, Matt Mazzola, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Xin Wang, Lijuan Wang, Chunyu Wang, Yu Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Haiping Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Sonali Yadav, Fan Yang, Jianwei Yang, Ziyi Yang, Yifan Yang, Donghan Yu, Lu Yuan, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone, 2024. arXiv:2404.14219 [cs]. 1, 2, 3, 4, 5
- [2] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073,

2024. 1, 2, 4, 5

- [3] ai. Grok-1.5 vision preview, 2024. 4
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 1, 2, 3
- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Yitzhak Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. CoRR, abs/2308.01390, 2023. 2
- [6] Anas Awadalla, Le Xue, Oscar Lo, Manli Shu, Hannah Lee,

- Etash Kumar Guha, Matt Jordan, Sheng Shen, Mohamed Awadalla, Silvio Savarese, et al. Mint-1t: Scaling opensource multimodal data by 10x: A multimodal dataset with one trillion tokens. arXiv preprint arXiv:2406.11271, 2024. 1, 4
- [7] Anas Awadalla, Le Xue, Manli Shu, An Yan, Jun Wang, Senthil Purushwalkam, Sheng Shen, Hannah Lee, Oscar Lo, Jae Sung Park, et al. Blip3-kale: Knowledge augmented largescale dense captions. arXiv preprint arXiv:2411.07461, 2024. 1, 4
- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 2
- [9] Ali Furkan Biten, Ruben Tito, Lluis Gomez, Ernest Valveny, and Dimosthenis Karatzas. Ocr-idl: Ocr annotations for industry document library dataset. In European Conference on Computer Vision, pages 241–252. Springer, 2022. 1, 3, 4
- [10] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827, 2024. 2
- [11] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021. 1, 4
- [12] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2, 4
- [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? arXiv preprint arXiv:2403.20330, 2024. 4
- [14] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023. 2
- [15] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS,

2023. 1, 2, 8

- [16] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD, 2024. 3
- [17] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan

- Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 2
- [18] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 5
- [19] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 3, 7
- [20] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR, abs/2306.13394, 2023. 4
- [21] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: multimodal large language models can see but not perceive. CoRR, abs/2404.12390, 2024. 4
- [22] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36, 2024. 1, 4
- [23] Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024. 2
- [24] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790,

2023. 2

- [25] Google. Gemini: A family of highly capable multimodal models, 2023. 1
- [26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [27] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024. 4
- [28] https://github.com/PFCCLab/PPOCRLabel. Awesome multilingual ocr toolkits based on paddlepaddle. 4
- [29] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small lan-

- guage models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. 1
- [30] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36, 2024. 2
- [31] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6693–6702, 2019. 4
- [32] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. MANTIS: Interleaved MultiImage Instruction Tuning, 2024. arXiv:2405.01483 [cs]. 4
- [33] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1988–1997,

2017. 4

- [34] Kushal Kafle, Scott Cohen, Brian Price, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In CVPR, 2018.
- [35] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision – ECCV 2016. Springer International Publishing, 2016. 4
- [36] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 1, 4
- [37] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024. 1
- [38] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?,

2024. 1, 3, 4

- [39] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 4
- [40] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023. 2
- [41] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. BLIP: bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022. 1
- [42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training

- with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 1, 2, 8
- [43] Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. A large-scale dataset towards multi-modal multilingual instruction tuning. arXiv preprint arXiv:2306.04387, 3, 2023. 2
- [44] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In The 2023 Conference on Empirical Methods in Natural Language Processing. 4
- [45] Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https: //huggingface.co/Open-Orca/OpenOrca, 2023. 4
- [46] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language

- models, 2023. 1

[47] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language

- models, 2024. 1, 4, 5, 6

- [48] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 2, 3, 4
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1, 2
- [50] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 3, 8
- [51] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 4
- [52] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 4
- [53] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models, 2024. 4
- [54] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and Jiaqi Wang. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms, 2024. 4
- [55] AI @ Meta Llama Team. The llama 3 herd of models, 2024. 1, 2, 4, 5
- [56] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, pages 2507–2521. Curran Associates, Inc., 2022. 4

- [57] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR),

2024. 4

- [58] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 4
- [59] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 2199–2208, 2021. 4
- [60] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Hongyu Hè, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training, 2024. arXiv:2403.09611 [cs]. 1, 3, 4, 6
- [61] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 947– 952, 2019. 4
- [62] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024. 4
- [63] OpenAI. Gpt-4v(ision) system card, 2023. 1, 4, 5, 6
- [64] OpenAI. Hello gpt-4o, 2024. 1
- [65] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011. 1, 4
- [66] Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning. arXiv preprint arXiv:2311.18799, 2023. 1
- [67] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [68] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VIII, page 146–162, Berlin, Heidelberg, 2022. SpringerVerlag. 4

- [69] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019. 4
- [70] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are incontext learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398– 14409, 2024. 2
- [71] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 1, 3, 4, 5
- [72] Fei Wang, Xingyu Fu, James Y. Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, Tianyi Lorena Yan, Wenjie Jacky Mo, Hsiang-Hui Liu, Pan Lu, Chunyuan Li, Chaowei Xiao, Kai-Wei Chang, Dan Roth, Sheng Zhang, Hoifung Poon, and Muhao Chen. Muirbench: A comprehensive benchmark for robust multiimage understanding. CoRR, abs/2406.09411, 2024. 4
- [73] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023. 2
- [74] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, and Weisi Lin. Q-bench: A benchmark for general-purpose foundation models on low-level vision. In ICLR. OpenReview.net, 2024. 4
- [75] An Yan, Zhengyuan Yang, Junda Wu, Wanrong Zhu, Jianwei Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Julian McAuley, Jianfeng Gao, et al. List items one by one: A new data source and learning paradigm for multimodal llms. arXiv preprint arXiv:2404.16375, 2024. 4
- [76] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 2, 4, 5
- [77] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 2
- [78] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplugowl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13040–13051, 2024. 2
- [79] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expres-

- sions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 4
- [80] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. CoRR, abs/2311.16502,

2023. 4

- [81] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 3, 7
- [82] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024. 1, 3, 4, 5
- [83] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, and Yinfei Yang. Ferretv2: An Improved Baseline for Referring and Grounding with Large Language Models, 2024. 3
- [84] Jieyu Zhang, Le Xue, Linxin Song, Jun Wang, Weikai Huang, Manli Shu, An Yan, Zixian Ma, Juan Carlos Niebles, Caiming Xiong, et al. Provision: Programmatically scaling visioncentric instruction data for multimodal language models. arXiv preprint arXiv:2412.07012, 2024. 4
- [85] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1724– 1732, 2024. 4
- [86] Bo Zhao, Boya Wu, Muyang He, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087, 2023. 2

