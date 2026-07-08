# arXiv:2502.12513v3[cs.CV]5Aug2025

[Figure 1]

## RealSyn: An Effective and Scalable Multimodal Interleaved Document Transformation Paradigm

Tiancheng Gu♥*, Kaicheng Yang♠*, Chaoyi Zhang♥, Yin Xie♠, Xiang An♠, Ziyong Feng♠, Dongnan Liu♥, Weidong Cai♥†, Jiankang Deng❉† ♥The University of Sydney ♠DeepGlint ❉Imperial College London tigu8498@uni.sydney.edu.au, kaichengyang@deepglint.com

Project Page Code

Abstract

###### Example of interleaved image-text documents:

[Figure 2]

After pre-training on extensive image-text pairs, Contrastive Language-Image Pre-training (CLIP) demonstrates promising performance on a wide variety of benchmarks. However, a substantial volume of multimodal interleaved documents remains underutilized for contrastive vision-language representation learning. To fully leverage these unpaired documents, we initially establish a Real-World Data Extraction pipeline to extract high-quality images and texts. Then we design a hierarchical retrieval method to efficiently associate each image with multiple semantically relevant realistic texts. To further enhance fine-grained visual information, we propose an image semantic augmented generation module for synthetic text production. Furthermore, we employ a semantic balance sampling strategy to improve dataset diversity, enabling better learning of long-tail concepts. Based on these innovations, we construct RealSyn, a dataset combining realistic and synthetic texts, available in three scales: 15M, 30M, and 100M. We compare our dataset with other widely used datasets of equivalent scale for CLIP training. Models pre-trained on RealSyn consistently achieve stateof-the-art performance across various downstream tasks, including linear probe, zero-shot transfer, zero-shot robustness, and zero-shot retrieval. Furthermore, extensive experiments confirm that RealSyn significantly enhances contrastive vision-language representation learning and demonstrates robust scalability.

......, , .There are many different standards of analog video. Most are conceptually similar, but they have varying numbers of signal channels, ,They are all based on the analog signal(s) drawing the......

[Figure 3]

[Figure 4]

###### How to utilize interleaved documents for CLIP tranining? How to leverage realistic&synthetic texts to enhance CLIP?

[Figure 5]

[Figure 6]

[Figure 7]

Real-World Data Extraction

Vision Expert Models

[Figure 8]

[Figure 9]

[Figure 10]

Caption

Image

Text

Tag

[Figure 11]

[Figure 12]

Hierarchical Retrieval

Image Semantic Augmented Generation

[Figure 13]

[Figure 14]

Synthetic Text

Realistic Text

[Figure 15]

[Figure 16]

Text Augmentation Supervision

Figure 1: Multimodal interleaved documents are unsuitable for CLIP training. We construct distinct image-text pairs from such documents via retrieval and generation.

and achieves excellent performance across a range of downstream tasks, including image captioning [44, 56], object detection [49, 77], and semantic segmentation [29, 51, 87]. Notably, the representational capacity of such models improves significantly with the scale of the training dataset [48]. Designing a new data paradigm to expand the scale and diversity of image-text datasets, thereby continuously improving the capabilities of pre-trained vision-language representations, represents a significant and challenging issue.

### Keywords

large-scaleimage-textdataset,vision-language representation learning, multimodal interleaved document

### 1 Introduction

The rapid proliferation of mobile networks and social platforms has led to exponential growth in large-scale data, offering a robust foundation for contrastive vision-language representation learning [2, 25–27, 81, 84, 88]. By predicting the correspondence between images and description texts, Contrastive Language-Image Pre-training (CLIP) [62] pre-trains separate uni-modal encoders

In recent years, the introduction of large-scale image-text pair datasets[8,21,64,65,71]hassubstantiallyadvanced vision-language representation learning. Due to download failures and non-English captions in the YFCC [71] dataset, DeCLIP [47] reprocesses a new version of the YFCC15M dataset. LAION400M [65] provides 400 million image-text pairs collected from the web and filtered using CLIP similarity. COYO700M [8] offers 747 million high-quality pairs curated through advanced filtering strategies, including CLIP similarity, watermark detection, and aesthetic scoring. Despite these advancements, a substantial amount of non-paired data, particularly interleaved image-text documents [43, 45, 89] containing multiple images and text paragraphs without explicit correspondence, remains incompatible with conventional vision-language representation learning methods.

* Equal Contribution † Corresponding Author.

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’25, Dublin, Ireland

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755098

198M

336M

285M

[Figure 17]

[Figure 18]

[Figure 19]

Image Extract

Quality Filter

Redundancy Image Filter

①Perceptual① Redundancy Semantic Redundancy

Aspect ratio ratio<1/3 or ratio > 3

②

- ①
- ②

118M

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Interleaved Image-Text Document

[Figure 24]

Image resolution min(width, lenth) < 100

[Figure 25]

2.13B 1.82B

1.16B

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Sentence Extract Redundancy Sentence Filter

Quality Filter

Semantic Filter

Remove Emoji, URL Sentence length

- ①
- ②
- ③ complexity < C1 or

T h e m a t c h b e t w e e n Tottenham Spurs vs Cjelsea will kick off from 16:30.

Information Entropy

###### ①

- ① Perceptual Redundancy A dog sitting next to a door. A dog sitting next to a door.

0.84B

Semantic Redundancy It‘s a vehicular collision. It’s an automobile incident.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- ②

entropy < 0.3

lenth<3 or length > 81 Complexity&Actions have no action

Sentence Perplexity

②

The derby had been played 54 times and the Blues have dominated the Spurs.

perplexity < 30 or perplexity > 200

Figure 2: The Real-World Data Extraction pipeline to extract high-quality images and texts from interleaved image-text documents.

Recent studies [19, 42, 83, 85] aim to enhance the quality of opensource image-text datasets using existing models. For instance, LaCLIP [19] leverages large language models (LLMs) to rewrite raw captions for better alignment with images. CapsFusion [83] finetunes an LLM using a ChatGPT-generated instruction dataset to mitigate hallucination issues. DreamLIP [85] re-captions 30 million images with detailed descriptions using a pre-trained large multimodal model. However, these methods primarily enhance data quality by focusing on synthetic data, thereby overlooking the significance of real-world data. Moreover, the diversity and distribution of synthetic captions produced by these approaches are inherently restricted by the limitations of the underlying generative models.

- • We propose an effective and scalable multimodal interleaved document transformation paradigm for contrastive visionlanguage representation learning.
- • We release RealSyn, a large-scale semantic balanced dataset that integrates both realistic and synthetic texts and is available in three sizes: 15M, 30M, and 100M.
- • We conduct extensive experiments and demonstrate the effectiveness and scalability of our proposed RealSyn dataset for CLIP training.

### 2 Related Work

In this paper, we explore two fundamental questions: 1) How to utilize multimodal interleaved documents for CLIP training.

Large-Scale Pre-training Dataset. In recent years, several largescale image-text datasets [8, 12, 15, 23, 46] collected from the Internet have been released. The YFCC100M [71] dataset provides a comprehensive overview of the evolution of photo and video documentation and sharing from the inception of Flickr in 2004 until early 2014. Due to download failures and non-English captions, DeCLIP [47] reprocesses a new version of the YFCC15M dataset. Additionally, the LAION400M [65] dataset contains 400 million image-text pairs collected from Common Crawl and widely used in vision-language pre-training. Recent advancements have also introduced several large-scale interleaved image-text document datasets [43, 45, 89]. The OBELICS [43] dataset uses a comprehensive filtering strategy and includes 141 million web pages, 353 million associated images, and 115 billion text tokens extracted from Common Crawl. However, due to data format constraints and training inefficiencies, interleaved image-text documents are currently unsuitable for CLIP training.

- 2) How to effectively leverage both realistic and synthetic texts to enhance CLIP performance. As depicted in Figure 1, we initially develop a Real-World Data Extraction pipeline to extract high-quality images and texts from real-world multimodal interleaved documents. Subsequently, we design a hierarchical retrieval method to efficiently associate each image with multiple semantically relevant texts. To improve fine-grained image understanding, we introduce a visual semantic augmented generation module for synthetic text production. Additionally, we implement a semantic balance sampling strategy to enhance dataset diversity and facilitate the learning of long-tail concepts. Leveraging these innovations, we construct the RealSyn dataset, which incorporates both realistic and synthetic texts in three sizes: 15M, 30M, and 100M. We evaluate the RealSyn dataset against other widely used datasets of similar scale for CLIP training. Models pre-trained on RealSyn consistently demonstrate state-of-the-art performance on various downstream tasks, including linear probe, zero-shot transfer, zero-shot robustness, and zero-shot retrieval. Extensive experiments validate that RealSyn is effective for contrastive vision-language representation learning and shows excellent scalability. The main contributions of this paper are summarized as follows:

Vision Language Pre-training. As a pioneering work in visual language pre-training, CLIP has attracted extensive attention due to its powerful zero-shot recognition and exceptional transfer learning performance [24, 31, 55, 66, 69, 74]. Inspired by CLIP, numerous visual-language pre-training works have been published in recent years [47, 57, 79]. SLIP [57] enhances performance by combining

self-supervised learning with CLIP pre-training. DeCLIP [47] increases pre-training efficiency by integrating multi-view supervision across modalities and nearest-neighbor supervision from similar pairs. To mitigate the influence of noisy data, ALIP [79] introduces a gating mechanism that dynamically allocates weights to samples. Despite their advancements, these methods primarily depend on large-scale image-text pairs derived from the Internet. Recent studies [48, 76] demonstrate that the capabilities of CLIP enhance with the expansion of high-quality image-text datasets. Given the extensive use of internet-derived image-text data in existing datasets, there is a pressing need to develop a new data construction paradigm to further expand the scale of high-quality image-text data.

Sentence Database

Text Semantic Clustering

There have a red narrative using these shadows.

Cluster 2 Cluster 3 Cluster N

Cluster 1

Text Encoder

.

...

The artists create a narrative using these shadows.

0.84B

Input Image

[Figure 38]

Realistic Text

Hierarchical Retrieval

[Figure 39]

[Figure 40]

[Figure 41]

Inter-Cluster Intra-Cluster Take that, winter! I hadn 't actually been in the hoop house for several weeks.

Image Encoder

[Figure 42]

Image Semantic Augmented Generation

Synthetic Text

[Figure 43]

Caption Generation

A tent is covered in snow in the woods.

I hadn't actually been in the hoop house for several weeks, but i was ready to take ......

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Synthetic Captions. Recent works [10, 79, 83] indicate that imagetext pairs obtained from websites contain intrinsic noise, which directly impacts the effectiveness of vision-language pre-training. To enhance the quality of existing datasets, LaCLIP [19] uses the in-context learning capability of large language models to rewrite text descriptions associated with each image. CapsFusion [83] employs large language models to refine information from web-based image-text pairs and synthetic captions, improving the quality of multimodal pre-training data. Similarly, DreamLIP [85] generates detailed descriptions for 30 million images using a pretrained large multimodal model. Nevertheless, these methods predominantly focus on synthetic data enhancement, neglecting the importance of real-world data. Furthermore, the diversity and distribution of synthetic captions generated by these methods are intrinsically constrained by the capabilities of the generative models employed.

LLaMA3

[Figure 48]

Semantic Tag Generation

snow|house|winter| blanket|fence

[Figure 49]

Image Embedding Cluster center embedding Intra-cluster text embedding

Figure 3: The architecture of our proposed framework, which constructs distinct image-text pairs from real-world data extracted from interleaved documents via retrieval and generation.

on quality, semantics, and redundancy. Initially, we eliminate sentences based on the following criteria: 1) presence of emojis or URLs; 2) sentences containing fewer than 3 or more than 81 words; and 3) following CAT [61], we retain samples with at least C1 caption complexity and incorporating an action. This phase reduces the corpus size from 2.13B to 1.82B sentences. Then we apply semantic filtering to the remaining sentences, eliminating those with minimal information assessed through information entropy:

- 3 RealSyn Dataset

∑︁𝑛

- 3.1 Real-World Data Extraction

𝑝(𝑥𝑖) log𝑝(𝑥𝑖), (1)

𝜃(𝑥) = −

To transform interleaved image-text documents for vision-language representation learning, we establish a Real-World Data Extraction pipeline (Figure 2) to extract high-quality images and texts. This pipeline consists of three steps: Data Extraction, Image Filtration, and Sentence Filtration.

𝑖=1

where 𝑛 denotes the number of words in a sentence, 𝑥𝑖 represents the 𝑖-th words in the sentences 𝑥, and 𝑝(𝑥𝑖) is the probability of the word 𝑥𝑖 in the entire corpus. Based on human cognition principles and empirical experience, we filter out sentences with a score below 0.3. To further refine the corpus by removing difficult or ambiguous sentences, we use GTP2-large [63] to calculate the perplexity score PPL for each sentence:

Data Extraction. We employ 118M interleaved image-text documents from the OBELICS [43] as the primary data source. All images are extracted and stored in a dedicated image database, while sentences are segmented using the Natural Language Toolkit (NLTK) [5] and stored in a separate sentence database. This process yields 336M images and 2.13B sentences from the interleaved documents.

∑︁𝑡

1 𝑡

PPL(𝑥) = exp −

log𝑝𝜃 (𝑥𝑖 | 𝑥<𝑖) , (2)

𝑖=1

Image Filtration. After extracting 336M images, we apply a twostage filtering process to ensure data quality and reduce redundancy. First, we discard images that meet any of the following criteria: 1) the shorter dimension is fewer than 100 pixels, or 2) the aspect ratio exceeds 3 or is below 1/3. This step removes 51M low-quality images. Next, following CLIP-CID [80], we use the EVA02-CLIP E/14-plus model [68] to extract image embeddings and apply the Union-Find algorithm [70] to eliminate perceptually and semantically redundant images. This step removes an additional 87M images, resulting in a refined dataset of 198M high-quality images. Sentence Filtration. After extracting 2.13B sentences from interleaved image-text documents, we conduct rigorous filtering based

where 𝑡 represents the token number of the sentence, and 𝑝𝜃 (𝑥𝑖 | 𝑥<𝑖) is the likelihood of the𝑖-th token given the previous tokens. We engage three human experts to determine the minimum and maximum values of the perplexity interval for high-quality sentences by comparing sentences of varying perplexity levels. Sentences within the average minimum (30) and maximum perplexity (200) intervals are retained. The overall semantics filtering reduces the corpus to 1.16B sentences. In the final stage, similar to redundancy image filtering, we perform both perceptual and semantic deduplication of sentences. This process results in a refined corpus of 0.84B sentences that include extensive real-world knowledge.

##### Table 1: Linear probe on 20 downstream datasets. Pre-training ViT-B/32 on RealSyn achieves 1.3%-6.9% average performance improvement.

CIFAR100

RESISC45

ImageNet

EuroSAT

Birdsnap

Average

CIFAR10

Country

Food101

SUN397

Aircraft

Flowers

UCF101

Caltech

Memes

STL10

KITTI

SST2

DTD

Cars

Pets

Data Scale Dataset

YFCC 67.2 90.4 70.8 47.7 66.7 23.8 29.7 62.4 65.7 80.1 90.0 94.7 94.9 79.4 75.4 18.4 70.8 48.6 56.2 56.7 64.5 LAION 71.0 93.3 78.1 41.0 66.3 76.9 43.0 71.2 74.5 87.6 88.2 93.6 95.3 82.9 72.2 13.5 75.4 55.7 57.3 59.3 69.8 RealSyn 77.1 94.5 78.7 43.4 71.4 64.7 42.7 71.3 79.9 90.0 88.2 96.4 96.2 87.2 72.4 16.7 79.9 55.7 57.7 64.0 71.4

15M

LAION 76.1 94.5 80.0 47.4 70.3 82.3 45.9 74.7 80.3 89.8 89.5 95.6 95.5 84.5 72.6 15.2 76.6 56.2 60.0 64.3 72.6 RealSyn 81.2 95.4 81.8 48.4 74.5 73.4 45.2 74.2 84.1 91.3 90.6 97.2 96.5 89.2 74.5 19.0 82.6 55.0 56.2 68.5 73.9

30M

LAION 80.2 95.7 82.5 51.3 73.4 85.3 46.1 75.6 83.2 91.1 92.0 96.9 95.2 85.9 68.4 17.4 80.0 57.3 61.4 68.3 74.4 RealSyn 84.2 96.3 83.5 54.0 76.2 77.4 47.6 75.6 86.3 92.1 91.7 97.7 96.8 90.6 73.1 21.1 83.7 57.3 58.9 71.6 75.8

100M

##### Table 2: Zero-shot transfer on 20 downstream datasets. Pre-training ViT-B/32 on RealSyn achieves 2.3%-14.3% average performance improvement.

CIFAR100

RESISC45

ImageNet

EuroSAT

Birdsnap

###### Average

CIFAR10

Country

Food101

SUN397

Aircraft

Flowers

UCF101

Caltech

Memes

STL10

KITTI

SST2

DTD

Cars

Pets

Data Scale Dataset

YFCC 36.3 74.0 40.3 19.4 41.8 2.1 2.3 12.0 19.8 59.8 48.9 87.7 21.2 20.3 23.8 5.1 27.8 47.4 50.1 32.3 33.6 LAION 49.1 85.7 56.9 11.5 45.1 49.9 3.8 25.7 54.6 78.1 30.5 89.5 36.7 36.1 21.7 5.6 38.2 48.8 49.9 37.1 42.7 RealSyn 60.0 85.7 58.3 10.5 56.4 27.6 5.5 33.2 61.7 80.2 31.2 92.4 56.5 56.2 34.0 8.9 52.6 53.3 51.3 43.3 47.9

15M

LAION 58.9 85.9 63.1 17.4 54.8 61.0 4.3 36.4 65.5 82.0 41.3 91.3 40.3 43.7 24.3 7.2 47.4 51.5 50.1 44.9 48.6 RealSyn 67.5 89.0 65.2 15.0 60.6 39.2 7.9 37.8 70.5 84.0 42.2 93.8 59.9 61.9 27.7 10.6 56.7 52.5 50.1 50.9 52.1

30M

LAION 68.9 90.5 68.6 23.6 60.6 68.3 7.8 41.2 74.7 87.1 47.7 94.4 45.6 53.4 23.6 10.4 54.5 51.9 53.3 52.8 53.9 RealSyn 73.5 89.5 68.8 20.1 65.0 48.5 10.2 46.1 76.7 87.6 48.8 94.4 69.0 65.5 24.6 12.1 60.5 52.4 54.1 56.2 56.2

100M

### 3.2 Retrieval and Generation Framework

After extracting high-quality images and sentences from documents, we propose an efficient and scalable framework to retrieve multiple semantically relevant texts for each image and leverage large language models to integrate retrieved realistic text with fine-grained visual information and generate synthetic text. As shown in Figure 3, the architecture of our framework primarily consists of three components: Text Semantic Clustering, Hierarchical Retrieval, and Image Semantic Augmented Generation.

Text Semantic Clustering. To efficiently retrieve multiple semantically relevant texts for each image, we initially encode all sentences using the EVA02-CLIP E/14-plus [68] model. Inspired by Unicom [1], we utilize the standard 𝐾-Means algorithm [34] offline cluster all sentences into a specified number of clusters. To minimize intra-cluster and inter-cluster conflicts, we initially establish the optimal number of cluster centroids through smallscale experiments. Subsequently, we divide the 0.84 billion texts into two million clusters, utilizing efficient feature quantization techniques [35].

Hierarchical Retrieval. Given the prohibitive computational cost of direct semantic text retrieval across 0.84B sentences (requiring over 10,000 GPU-hours on 8×A100), we propose a hierarchical retrieval framework to enhance efficiency. Given an image, we perform inter-cluster retrieval to identify the most relevant cluster center for the image. Subsequently, we conduct intra-cluster retrieval within the identified cluster to obtain multiple real-world sentences semantically matched to the image. This approach achieves scalable retrieval of 198M images and 0.84B sentences within 40 GPU-hours on an 8×A100.

ImageSemanticAugmentedGeneration.Althoughtheretrieved realistic texts achieve satisfactory performance, they exhibit limitations in capturing fine-grained visual semantics. To address this

issue, we introduce the Image Semantic Augmented Generation module. This module initially employs the OFA [75] model to generate a concise caption for each image. We then integrate the open-set image tagging model RAM++ [33], which extracts object detection tags. Considering that RAM++ supports only 4,000 tags, we expand this set to 8,000 by incorporating an additional 4,000 tags derived from real-world sentences. Following CapsFusion [83], we utilize ChatGPT4 Turbo to merge the retrieved realistic texts with concise captions and image tags to construct a 100K instruction dataset (the prompt we used is presented in the supplementary material). Subsequently, we conduct instruction tuning of the LLaMA3-8B model [18] using the LLaMA Factory [86] and deploy the vLLM [40] for large-scale inference. Ultimately, we convert data from 118M multimodal interleaved documents into 198M image-text pairs, where each image is associated with multiple retrieved realistic texts and synthetic texts.

### 3.3 Semantic Balance Sampling

To further improve the quality and diversity of our dataset, we implement semantic balancing sampling across the 198M image-text pairs. Specifically, we use EVA02-CLIP E/14-plus [68] to encode and calculate the cosine similarity between images and synthetic texts. To reduce the impact of OCR-related or mismatched pairs during pre-training, we filter out 29.7M pairs with cosine similarities scores either above 0.61 or below 0.51, as determined through human verification. Inspired by MetaCLIP [32], we introduce an easy but efficient cluster-based semantic balance sampling strategy. We cluster the image embeddings from the remaining 168.3M pairs into 1M centers. To enhance the semantic diversity of our dataset, we randomly select 20, 35, and 180 samples from clusters exceeding these thresholds, while retaining all samples from smaller clusters.

##### Table 3: Zero-shot image-text retrieval performance on Flickr30k and MSCOCO. Pre-training CLIP-B/32 on RealSyn dataset achieves a significant improvement on all metrics.

Text Retrieval Image Retrieval Flickr30k MSCOCO Flickr30k MSCOCO

Data Scale Dataset R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

YFCC 37.1 64.8 75.9 21.3 45.1 57.0 23.5 47.3 58.3 13.2 32.0 43.1 LAION 49.1 76.8 84.5 28.4 53.0 64.9 33.3 60.5 70.9 17.4 38.3 49.7 RealSyn 72.9 91.1 95.1 43.8 69.5 79.6 49.5 76.3 84.6 25.8 50.6 62.5

15M

LAION 59.6 83.5 89.8 35.9 62.4 73.2 42.4 70.1 79.4 22.1 45.5 57.6 RealSyn 76.0 93.3 96.9 48.2 74.6 83.0 54.0 80.0 87.6 29.5 55.2 66.9

30M

LAION 67.5 87.9 93.0 43.3 68.0 78.1 50.4 77.2 85.5 27.1 52.1 63.8 RealSyn 81.6 96.1 97.3 52.3 76.7 85.0 58.8 84.1 90.5 32.5 58.9 70.2

100M

##### Table 4: Zero-shot robustness comparison on different IN-1K val set variants. Pre-training CLIP-B/32 on RealSyn demonstrates superior robustness across all datasets.

Data Scale Dataset IN-V2 IN-A IN-R ObjectNet IN-Sketch Average

YFCC 27.3 12.3 20.8 25.3 6.3 18.4 LAION 30.7 6.0 46.5 28.7 24.3 27.2 RealSyn 37.1 12.5 47.7 35.0 25.4 31.5

15M

LAION 37.5 8.9 54.4 35.5 31.8 33.6 RealSyn 42.9 16.1 56.6 41.5 31.9 37.8

30M

LAION 44.6 12.2 62.5 42.2 37.9 39.9 RealSyn 47.6 19.7 62.5 45.8 37.9 42.7

100M

This approach culminates in the construction of the RealSyn15M, RealSyn30M, and RealSyn100M datasets.

### 4 Experiments and Results

- 4.1 Implementation Details We initially collect 118M interleaved image-text documents from the OBELICS [43] as our primary data source. We use OFA𝑏𝑎𝑠𝑒 [75]

and RAM++𝑙𝑎𝑟𝑔𝑒 [33] to generate brief captions and semantic tags. To validate the dataset performance, we pre-train standard CLIP supervised by the text randomly selected from the three retrieved realistic texts and one synthetic text, inspired by the LaCLIP [19]. During pre-training, we adopt AdamW [53] as the optimizer, with a learning rate of 1e-3 and a weight decay of 0.2. The parameters 𝛽1 and 𝛽2 are set to 0.9 and 0.98, respectively. The input image size is 224×224, and the input text sequence length is 77. The temperature parameter𝜏 is initialized to 0.07. We train 32 epochs with 4096 batch sizes on 8 × A100 (80G) GPUs. Please refer to the supplementary material for more details.

To validate the effectiveness of the RealSyn dataset, we compare RealSyn with the previous datasets across various models and data scales. We compare RealSyn15M with the YFCC15M filtered by DeCLIP [47]. Following ALIP [79], we also compare with LAION15M, LAION30M, and LAION100M (subset randomly selected from LAION400M).

- 4.2 Main Results

Linear Probe. In Table 1, we report the linear probe performance of the ViT-B/32 model across 20 downstream datasets. When pretrained at the 15M scale, RealSyn15M exceeds YFCC15M on 16 of 20

datasets, achieving an average performance increase of 6.9%. Additionally, RealSyn15M outperforms LAION15M on 18 of 20 datasets, with an average improvement of 1.6%. With dataset scaling to 30M and 100M, RealSyn achieves average performance improvements of 1.3% and 1.4% over LAION, respectively. These findings underscore the superior CLIP training efficiency of the RealSyn dataset compared to the widely-used YFCC and LAION datasets across multiple scales.

Zero-shot Transfer. We evaluate the zero-shot transfer performance of the ViT-B/32 model across 20 classification benchmarks using the same prompt templates as SLIP [57]. As indicated in Table 2, RealSyn15M surpasses YFCC15M on 18 of 20 datasets, achieving an average performance improvement of 14.3%. In comparison to LAION15M, RealSyn15M excels on 18 of 20 datasets with an average improvement of 5.2%. Upon expanding the dataset sizes to 30M and 100M, RealSyn achieves average performance improvements of 3.5% and 2.3% compared to LAION, highlighting its efficiency and scalability.

It is important to note that RealSyn demonstrates a significant decrease in performance on certain datasets, such as Cars and Flowers. This reduction is primarily attributed to the unique data distribution of RealSyn, characterized by a scarcity of data for specific concepts, which hampers the model’s ability to effectively learn these concepts. For example, as shown in Figure 4, samples related to cars and flowers represent only 0.9% and 0.4% of the dataset, respectively.

Zero-shot Image-Text Retrieval. In Table 3, we present the zeroshot image-text retrieval performance of the ViT-B/32 model pretrained on different scales of datasets. RealSyn achieves superior results across all evaluation metrics. Specifically, RealSyn15M improvesRecall@1by35.8%&26%onFlickr30K [82]andby22.5%&12.6% on MSCOCO [50]. RealSyn30M improves Recall@1 by 16.4%&11.6% on Flickr30K [82] and by 12.3%&7.4% on MSCOCO [50]. RealSyn100M improves Recall@1 by on 14.6%&8.4% Flickr30K [82] and by 9%&5.4% on MSCOCO [50]. This significant enhancement in cross-modal retrieval performance indicates that the RealSyn dataset effectively improves contrastive vision-language representation learning by incorporating both realistic and synthetic texts, resulting in robust representations and enhanced cross-modal alignment.

Zero-shot Robustness. To further validate the robustness of the RealSyn dataset, we present the zero-shot robustness [62] performance across various IN-1K [17] val set variants in Table 4. The

Flower(0.4%):Color,Tea,Flowers,White

Animal(3.2%):Dog,Dogs,Cat,horse

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Food(4.2%): Salmon, Taco, Hamburger, Beer

Automotive(0.9%):Car,Cars,Engine,Ford

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

Landmark(0.5%):Town,Village,Hall,House

Airplane(3.8%):Airport,Flight,Air,Airline

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

##### Figure 4: A T-SNE [72] projection of LDA [6] topic cluster from a randomly selected 1M samples from RealSyn. RealSyn encompasses a broad range of everyday topics, e.g., animal, food, airplane, etc. We also display representative central images for each identified topic.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

##### Figure 5: Image captioning comparisons on COCO2017 and Flickr30k. B4, MT., RL. and Cd. represent the metric of BLEU [59], METEOR [3], ROUGE-L [3], and Cider [73].

- (a) Richness assessment comparison

[Figure 98]

- (b) Diversity assessment comparison

results indicate that RealSyn significantly enhances the robustness of vision-language pre-training models. Specifically, RealSyn15M shows a performance increase of 13.1% and 4.3% compared to YFCC15M and LAION15M, respectively. When scaled to 30M and 100M, RealSyn achieves additional improvements of 4.2% and 2.8% over LAION. This substantial enhancement in performance is primarily attributable to the utilization of retrieved realistic texts, which surpass the constraints of generative models, and the superior conceptual diversity relative to YFCC and LAION, thus significantly boosting model robustness.

Figure 6: The richness assessment and diversity assessment on different datasets. RealSyn-R1: the most relevant retrieved realistic text. RealSyn-S1: the semantic augmented synthetic text based on RealSyn-R1.

Image Captioning via MLLM. In Figure 5, we present the image captioning performance of the LLaVA-1.5 [52] trained using different datasets (LAION v.s. RealSyn). Initially, we first map visual features into the textual domain using the initial 558k dataset from LLaVA-1.5. We then develop an image captioning dataset from both LAION and RealSyn for instruction tuning. Specifically, we select 1M samples randomly from each dataset and train over two epochs. As depicted in Figure 5, RealSyn significantly outperforms LAION in all evaluation metrics on both the COCO2017 [50] and Flickr30k [82] benchmarks. This notable performance enhancement confirms the higher quality and better image-text alignment of the RealSyn dataset.

### 5 Analysis 5.1 Statistics Analysis

Topic-based Assessment. Following MMC4 [89], we ran LDA [6] on random sampling 1M image-realistic text pairs with 30 topics. Figure 4 presents the proportions and examples for six topics: animal, food, airplane, flower, automotive, and landmark. Notably, the dataset contains minimal samples related to “flower” and “automotive” topics, representing merely 0.4% and 0.9% of the total, respectively. This paucity of examples hinders the model’s ability

[Figure 100]

Figure 7: Data Scaling Analysis. We pretrain ViT-B/32 on RealSyn in different data scales. × represents the results predicted using our data scaling law.

to sufficiently learn these concepts, thereby compromising its performance in the linear probe and zero-shot transfer evaluations on the Flowers and Cars datasets.

Richness Assessment. Figure 6a presents image-text similarity and text token distribution of 15M samples from YFCC15, LAION, RealSyn-R1 (the most relevant retrieved realistic text), and RealSynS1 (the semantic augmented synthetic text based on RealSyn-R1). Compared to datasets sourced from the Internet, RealSyn demonstrates robust similarity metrics even after the removal of OCR data. This effectively explains the significant performance enhancement observed in the CLIP model trained on RealSyn for image-text retrieval tasks. Moreover, both the retrieved realistic texts and synthetic texts contain a larger quantity of words, which can provide a richer textual context that enhances vision-language representation learning.

Diversity Assessment. The RealSyn is constructed based on realworld interleaved image-text documents, which encompasses a wide array of diverse information. Following previous work [41], we randomly select 0.2M samples to calculate the number of unique entities in the caption to assess the data diversity of different datasets. As depicted in Figure 6b, both the retrieved realistic texts and image semantic augmented synthetic texts exhibit a higher number of distinct entities. Such diversity enriches the dataset, facilitating the model’s acquisition of comprehensive knowledge and enhancing both performance and robustness.

Data Scaling Analysis. We present the data scaling law [36] derived from our RealSyn dataset, justifying its scalability over samples. Specifically, we conduct a series of visual-language pretrainings with proposed datasets ranging from 12M to 60M, and fit each performance metric to the inverse of logarithmic functions with respect to the number of millions of training samples 𝑥. Based on the fitting results from these preliminary experiments, we extrapolate each performance scaling law to 100𝑀 samples, and validate their predicted scaling trends with our RealSyn100𝑀 dataset as shown in Figure 7. Notably, as indicated by the coefficients shown in Eq. 3, these performance laws also likely suggest

[Figure 101]

Figure 8: Model scaling capability. We compare the models pre-trained on LAION30M and RealSyn30M.

Table 5: Comparison of semantic balance sampling and random sampling on the 15M dataset.

Linear probe Avg

Transfer Avg

Robustness Avg

Model Dataset

YFCC 64.5 33.6 18.4

LAION 69.8 42.7 27.2 RealSyn-Random 70.7 46.8 30.5 RealSyn-Balance 71.4 47.9 31.5

CLIP-B/32

an upper bound of model capability that a ViT-B/32 could possibly reach through our proposed visual-language pre-training paradigm with multimodal interleaved documents:

−0.21 𝑙𝑜𝑔(𝑥 − 4.23)

Linear Probe: L(𝑥) ≈

+ 0.80

−0.30 𝑙𝑜𝑔(𝑥 − 5.68)

Transfer: L(𝑥) ≈

+ 0.62

(3)

−0.60 𝑙𝑜𝑔(𝑥 − 3.17)

Robustness: L(𝑥) ≈

+ 0.56

Model Scaling Analysis. In Section 4.2, RealSyn exhibits superior performance across various data scales. To further explore the model scaling capability, we present the downstream task performance of three models in Figure 8. Notably, compared to LAION, RealSyn demonstrates steeper slopes in performance curves across linear probe, zero-shot transfer, and robustness, indicative of its superior model scaling capabilities.

Extension to Pure Image. To further extend our transformation paradigm for pure images, we conduct experiments on ImageNet [17]. Initially, we retrieve semantically relevant realistic texts for each ImageNet image from our sentence database and generate image semantic augmented synthetic texts. Then, we pre-train ResNet50 [28] supervised by the text randomly selected from the retrieved realistic texts and synthetic texts. Comparative analysis with SimCLR [11] under identical conditions shows a linear probe average performance enhancement of 2.1% across 12 datasets using our constructed data. Detailed experimental results are provided in the supplementary material.

### 5.2 Ablation Study

Ablation on Semantic Balance Sampling. To demonstrate the efficacy of our proposed semantic balance sampling method, we contrast it with random sampling. As shown in Table 5, semantic balance sampling achieves performance improvements of 0.7%,

##### Table 6: Ablation study examining the quantity and combinations of retrieved realistic texts and corresponding image semantic augmented synthetic texts within a 15M-scale dataset. 𝑇𝑟𝑘: the 𝑘-th retrieved semantic relevant realistic text. 𝑇𝑠𝑘: the image semantic augmented synthetic text for𝑇𝑟𝑘.

Linear probe Avg 70.3 71.0 71.2 70.9 70.6

Linear probe Avg 70.2 70.0 69.9 69.4 69.1

Linear probe Avg

Transfer Avg

Robustness Avg

𝑇𝑟1 𝑇𝑟2 𝑇𝑟3 𝑇𝑟4 𝑇𝑟5

𝑇𝑠1 𝑇𝑠2 𝑇𝑠3 𝑇𝑠4 𝑇𝑠5

𝑇𝑟1 𝑇𝑟2 𝑇𝑟3 𝑇𝑠1

- 70.3 42.4 25.7
- 71.2 46.8 30.7

- 70.2 39.7 24.0

- 71.4 47.9 31.5

[Figure 102]

[Figure 103]

This was unplanned but turned out to be one of our most memorable experiences in Norway

– for both good and bad reasons! Galdhopiggen is over 8,000 feet or 2469m. A biggie BUT with a ski road taking you up really high.... which really isn’t a clever approach approach with a

[Figure 104]

glacier and crevasses on one side and a drop on the other but the snow wasn’t TOO slippy.....

[Figure 105]

[Figure 106]

Image:

Figure 9: Clustering distribution of 15M data obtained from random sampling and semantic balance sampling.

Some exciting but limited views over the edge to thick slabs of snow , clinging to the crags.

To the south , the lyell glacier looked healthy after last winter's heavy snows , with no visible bareice .

Real Text:

Lyell glacier, to the south, appeared healthy after last winter's heavy snows, with no visible bareice. A large cirque at the head of the valley.

Exciting but limited views over the edge to thick slabs of snow, clinging to the crags. Above rest rock the snow ridge relaxes in angle.

1.1%, and 1.0% in linear probe accuracy, zero-shot transfer, and robustness, respectively. Besides, we visualize the data distribution of 15M samples clustered into 1M centers using different sampling strategies. As shown in Figure 9 reveals that semantic balance sampling results in a smoother distribution, thereby enhancing the learning of long-tail concepts.

Syn Text:

##### Figure 10: Visualization of the raw interleaved document, the retrieved realistic text, and synthetic text. Image semanticrelated information is highlighted in green.

Ablation on Realistic Texts and Synthetic Texts. We conduct ablation studies to assess the impact of varying quantities of realistic and synthetic texts on the performance of the CLIP-B/32 model. As shown in Table 6, incrementally increasing the amount of realistic text from one to three enhances model performance, attributed to improved text augmentation that integrates extensive real-world knowledge. However, further increasing this quantity from three to five slightly diminishes performance due to information saturation and the introduction of noise. Conversely, augmenting the number of synthetic texts from one to five incrementally degrades performance, reflecting increased noise introduction. Notably, training with only realistic texts significantly boosts performance, achieving a 71.2% accuracy compared to 69.8% with the LAION15M dataset, underscoring the vital role of real-world knowledge in advancing vision-language representation learning.

“crags”, and “valley”. We provide more visualizations in the supplementary material.

### 6 Conclusion

This paper explores two open-ended questions: 1) How to utilize multimodal interleaved documents for CLIP training. 2) How to effectively leverage both realistic and synthetic texts to enhance CLIP performance. To this end, we first establish a Real-World Data Extraction pipeline to extract high-quality images and texts. Then we design a hierarchical retrieval method to efficiently associate each image with multiple semantically relevant texts. To further enhance fine-grained visual information, we propose an image semantic augmented generation module for synthetic text production. Furthermore, we employ a semantic balance sampling strategy to improve dataset diversity, enabling better learning of long-tail concepts. Based on these innovations, we present RealSyn, a dataset driven by both real and synthetic texts with three sizes: 15M, 30M, and 100M. We compare our dataset with other widely used datasets of equivalent scale for CLIP training. Models pre-trained on RealSyn consistently achieve state-of-the-art performance across various downstream tasks. Furthermore, extensive experiments confirm that RealSyn significantly enhances contrastive vision-language representation learning and demonstrates robust scalability. We hope our work provides insights into vision-language representation learning.

Ablation on the Combination of the Realistic Texts and Synthetic Texts. Table 6 presents the results of ablation experiments on text augmentation using different text types. Introducing image semantic augmented synthetic text, which supplements fine-grained visual semantic information, leads to performance enhancements of 0.2%, 1.1%, and 0.8% in linear probe accuracy, zero-shot transfer, and zero-shot robustness, respectively, compared to using only retrieved realistic texts.

Case Study. In Figure 10, we present the visualization of retrieved realistic text and synthetic text obtained from an interleaved imagetext document using our proposed transformation paradigm. Both realistic and synthetic texts contain extensive descriptive information consistent with the image semantics, such as “lyell glacier”,

### References

- [31] Xiaoxing Hu, Kaicheng Yang, Jun Wang, Haoran Xu, Ziyong Feng, and Yupei Wang. 2025. Decoupled Global-Local Alignment for Improving Compositional Understanding. arXiv preprint arXiv:2504.16801 (2025).
- [32] Xiaoqing Ellen Tan Hu Xu, Saining Xie. 2023. Demystifying CLIP Data. arXiv:2309.16671 (2023).
- [33] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. 2023. Inject semantic concepts into image tagging for open-set recognition. arXiv:2310.15200 (2023).
- [34] Abiodun M. Ikotun, Absalom E. Ezugwu, Laith Abualigah, Belal Abuhaija, and Jia Heming. 2023. K-means clustering algorithms: A comprehensive review, variants analysis, and advances in the era of big data. Inf. Sci. (2023).
- [35] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
- [36] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020).
- [37] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. 2020. The hateful memes challenge: Detecting hate speech in multimodal memes. In NeurIPS.
- [38] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 2013. 3d object representations for fine-grained categorization. In ICCVW.
- [39] Alex Krizhevsky, Geoffrey Hinton, et al. 2009. Learning multiple layers of features from tiny images. Technical Report (2009).
- [40] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In SOSP.
- [41] Zhengfeng Lai, Vasileios Saveris, Chen Chen, Hong-You Chen, Haotian Zhang, Bowen Zhang, Juan Lao Tebar, Wenze Hu, Zhe Gan, Peter Grasch, et al. 2025. Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models. In ICLR.
- [42] Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, and Meng Cao. 2024. VeCLIP: Improving CLIP Training via Visual-enriched Captions. In ECCV.
- [43] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. 2024. Obelics: An open web-scale filtered dataset of interleaved image-text documents. In NeurIPS.
- [44] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.
- [45] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, et al. 2025. OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text. In ICLR.
- [46] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and LingYu Duan. 2024. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. In NeurIPS.
- [47] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. 2022. Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm. In ICLR.
- [48] Zichao Li, Cihang Xie, and Ekin Dogus Cubuk. 2024. Scaling (down) clip: A comprehensive analysis of data, architecture, and training strategies. arXiv preprint arXiv:2404.08197 (2024).
- [49] Jiayi Lin and Shaogang Gong. 2023. Gridclip: One-stage object detection by grid-level clip representation learning. arXiv preprint arXiv:2303.09252 (2023).
- [50] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In ECCV.
- [51] Yuqi Lin, Minghao Chen, Wenxiao Wang, Boxi Wu, Ke Li, Binbin Lin, Haifeng Liu, and Xiaofei He. 2023. Clip is also an efficient segmenter: A text-driven approach for weakly supervised semantic segmentation. In CVPR. 15305–15314.
- [52] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In CVPR.
- [53] I Loshchilov. 2019. Decoupled Weight Decay Regularization. In ICLR.
- [54] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi.

2013. Fine-grained visual classification of aircraft. arXiv:1306.5151 (2013).

- [55] Ségolène Martin, Yunshi Huang, Fereshteh Shakeri, Jean-Christophe Pesquet, and Ismail Ben Ayed. 2024. Transductive Zero-Shot and Few-Shot CLIP. In CVPR.
- [56] Ron Mokady, Amir Hertz, and Amit H Bermano. 2021. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734 (2021).
- [57] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. 2022. Slip: Self-supervision meets language-image pre-training. In ECCV.
- [58] Maria-Elena Nilsback and Andrew Zisserman. 2008. Automated flower classification over a large number of classes. In Sixth Indian Conference on Computer Vision, Graphics & Image Processing.
- [59] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In ACL.

- [1] Xiang An, Jiankang Deng, Kaicheng Yang, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. 2023. Unicom: Universal and Compact Representation Learning for Image Retrieval. In ICLR.
- [2] Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2018. Multimodal machine learning: A survey and taxonomy. TPAMI (2018).
- [3] Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In ACL.
- [4] Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L Alexander, David W Jacobs, and Peter N Belhumeur. 2014. Birdsnap: Large-scale fine-grained visual categorization of birds. In CVPR.
- [5] Steven Bird and Edward Loper. 2004. NLTK: The Natural Language Toolkit. In ACL.
- [6] David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. JMLR (2003).
- [7] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. 2014. Food-101–mining discriminative components with random forests. In ECCV.
- [8] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. 2022. COYO-700M: Image-Text Pair Dataset. https://github. com/kakaobrain/coyo-dataset.
- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. 2021. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR.
- [10] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024. ShareGPT4V: Improving Large Multi-Modal Models with Better Captions. In ECCV.
- [11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In ICML.
- [12] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. 2015. Microsoft coco captions: Data collection and evaluation server. arXiv:1504.00325 (2015).
- [13] Gong Cheng, Junwei Han, and Xiaoqiang Lu. 2017. Remote sensing image scene classification: Benchmark and state of the art. Proc. IEEE (2017).
- [14] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. 2014. Describing textures in the wild. In CVPR.
- [15] Christopher Clark and Matt Gardner. 2017. Simple and effective multi-paragraph reading comprehension. In ACL.
- [16] Adam Coates, Andrew Ng, and Honglak Lee. 2011. An analysis of single-layer networks in unsupervised feature learning. In AISTATES.
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In CVPR.
- [18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv:2407.21783 (2024).
- [19] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. 2024. Improving clip training with language rewrites. In NeurIPS.
- [20] Li Fei-Fei, Rob Fergus, and Pietro Perona. 2004. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPR.
- [21] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. 2024. Datacomp: In search of the next generation of multimodal datasets. In NeurIPS.
- [22] Andreas Geiger, Philip Lenz, and Raquel Urtasun. 2012. Are we ready for autonomous driving? the kitti vision benchmark suite. In CVPR.
- [23] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh.

2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR.

- [24] Tiancheng Gu, Kaicheng Yang, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, and Jiankang Deng. 2024. Rwkv-clip: A robustvision-language representation learner. In EMNLP.
- [25] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. 2024. Regiongpt: Towards region understanding vision language model. In CVPR.
- [26] Ruohao Guo, Liao Qu, Dantong Niu, Yanyu Qi, Wenzhen Yue, Ji Shi, Bowei Xing, and Xianghua Ying. 2024. Open-Vocabulary Audio-Visual Semantic Segmentation. In ACMMM. 7533–7541.
- [27] Wenzhong Guo, Jianwen Wang, and Shiping Wang. 2019. Deep multimodal representation learning: A survey. IEEE Access (2019).
- [28] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In CVPR.
- [29] Wenbin He, Suphanut Jamonnak, Liang Gou, and Liu Ren. 2023. Clip-s4: Language-guided self-supervised semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11207–11216.
- [30] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. 2019. Eurosat: A novel dataset and deep learning benchmark for land use and land

cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2019).

- [60] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. 2012. Cats and dogs. In ICCV.
- [61] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan.

2023. Filtering, distillation, and hard negatives for vision-language pre-training. In CVPR.

- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML.
- [63] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog

(2019).

- [64] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. 2022. LAION-5B: An open large-scale dataset for training next generation image-text models. In NeurIPS.
- [65] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki.

2021. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. In NeurIPS Workshop.

- [66] Shuai Shao, Yu Bai, Yan Wang, Baodi Liu, and Yicong Zhou. 2024. DeIL: Directand-Inverse CLIP for Open-World Few-Shot Learning. In CVPR.
- [67] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 2012. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv:1212.0402

(2012).

- [68] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. 2023. Eva-clip: Improved training techniques for clip at scale. arXiv:2303.15389 (2023).
- [69] Yuwei Tang, Zhenyi Lin, Qilong Wang, Pengfei Zhu, and Qinghua Hu. 2024. AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning. In CVPR.
- [70] Robert Endre Tarjan. 1975. Efficiency of a good but not linear set union algorithm. JACM (1975).
- [71] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. 2016. YFCC100M: The new data in multimedia research. Commun. ACM (2016).
- [72] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. JMLR (2008).
- [73] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In CVPR.
- [74] Jingyun Wang and Guoliang Kang. 2024. Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation. In CVPR.
- [75] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. Ofa: Unifying architectures,

- tasks, and modalities through a simple sequence-to-sequence learning framework. In ICML.
- [76] Xiao Wang, Ibrahim Alabdulmohsin, Daniel Salz, Zhe Li, Keran Rong, and Xiaohua Zhai. 2025. Scaling Pre-training to One Hundred Billion Data for Vision Language Models. arXiv preprint arXiv:2502.07617 (2025).
- [77] Xiaoshi Wu, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Cora: Adapting clip for open-vocabulary detection with region prompting and anchor pre-matching. In CVPR. 7031–7040.
- [78] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba.

2010. Sun database: Large-scale scene recognition from abbey to zoo. In ICCV.

- [79] Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. 2023. Alip: Adaptive language-image pre-training with synthetic caption. In ICCV.
- [80] Kaicheng Yang, Tiancheng Gu, Xiang An, Haiqiang Jiang, Xiangzi Dai, Ziyong Feng, Weidong Cai, and Jiankang Deng. 2025. CLIP-CID: Efficient CLIP Distillation via Cluster-Instance Discrimination. In AAAI.
- [81] Ruilin Yao, Shengwu Xiong, Yichen Zhao, and Yi Rong. 2024. Visual Grounding with Multi-modal Conditional Adaptation. In ACMMM. 3877–3886.
- [82] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. TACL (2014).
- [83] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, and Jingjing Liu. 2024. Capsfusion: Rethinking image-text data at scale. In CVPR.
- [84] Changmeng Zheng, Dayong Liang, Wengyu Zhang, Xiao-Yong Wei, Tat-Seng Chua, and Qing Li. 2024. A Picture Is Worth a Graph: A Blueprint Debate Paradigm for Multimodal Reasoning. In ACMMM. 419–428.
- [85] Kecheng Zheng, Yifei Zhang, Wei Wu, Fan Lu, Shuailei Ma, Xin Jin, Wei Chen, and Yujun Shen. 2024. DreamLIP: Language-Image Pre-training with Long Captions. In ECCV.
- [86] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. In ACL.
- [87] Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, and Yifan Liu. 2023. Zegclip: Towards adapting clip for zero-shot semantic segmentation. In CVPR. 11175– 11185.
- [88] Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, and Junran Wu. 2024. Do LLMs Understand Visual Anomalies? Uncovering LLM’s Capabilities in Zero-shot Anomaly Detection. In ACMMM. 48–57.
- [89] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi.

2024. Multimodal c4: An open, billion-scale corpus of images interleaved with text. In NeurIPS.

This supplementary material introduces the experiment settings and the instruction prompt we used in Section A. Then, we introduce the downstream datasets, detailed model scaling results, comparison with the rewrite datasets, and analyze the combination RealSyn with LAION and detailed results on pure image in Section B. We further analyze our proposed RealSyn dataset by comparison with current datasets and visualize examples in Section C. Finally, we discuss the limitations of this paper in Section D.

- A Detail Experiment Settings

- A.1 Experiment Settings In Table 7, we present the detailed settings used in training CLIP.

Table 7: Hyperparameters used for CLIP pre-training.

Hyperparameter Value Initial temperature 0.07 Weight decay 0.2 Batch size 4096 Learning rate 0.001 Learning rate scheduler OneCycleLR Pct start 0.1 Training epochs 32 GPU 8×A100

- Adam 𝛽1 0.9

- Adam 𝛽2 0.98 Adam 𝜖 10−6

- A.2 Detail Instruction Prompt

The prompt we used for ChatGPT to construct the 100K instruction dataset is present in the following:

"Please merge the information from the given raw text and the synthetic caption with the help of the highly relevant detection tags. The raw caption offers detailed real-world information, yet it suffers from flaws in sentence structure and grammar. The synthetic caption exhibits impeccable sentence structure but often lacks in-depth realworld details and may contain false information. The highly relevant detection tags are provided to enrich the semantic information of the raw caption, while some are redundant and noisy. You are a great information integration and summary expert, you are also good at enriching semantic information. Ensure a well-structured sentence while retaining the detailed real-world information provided in the raw caption. Avoid simply concatenating the sentences and avoid adding external information to describe. Correct and simplify sentences finally. Raw caption:<raw caption>, synthetic caption:<synthetic caption>, and highly relevant detection tags:<detection tags>".

- B Detail External Results

- B.1 Downstream Datasets

To demonstrate the performance of CLIP trained on RealSyn, we compared the linear probe results of CLIP trained on RealSyn,

YFCC [71], and LAION [65] across 20 datasets. These datasets include Food101 [7], CIFAR10 [39], CIFAR100 [39], Birdsnap [4], SUN397 [78], Stanford Cars [38], FGVC Aircraft [54], DTD [14], Pets [60], Caltech101[20], Flowers102[58], SLT10[16], EuroSAT[30], RESISC45 [13], KITTI [22], Country211 [62], UCF101 [67], Hateful Memes [37], SST2 [62], and ImageNet [17]. Details on each dataset and the corresponding evaluation metrics are provided in Table 8.

Table 8: List of linear probe datasets with the data distribution and evaluation metrics.

Dataset Classes Train size Test size Evaluation metric Food101 102 75,750 25,250 accuracy CIFAR10 10 50,000 10,000 accuracy CIFAR100 100 50,000 10,000 accuracy Birdsnap 500 42,138 2,149 accuracy SUN397 397 19,850 19,850 accuracy Cars 196 8,144 8,041 accuracy Aircraft 100 6,667 3,333 mean per class DTD 47 3,760 1,880 accuracy Pets 37 3,680 3,669 mean per class Caltech101 101 3,000 5,677 mean per class Flowers 102 2,040 6,149 mean per class STL10 10 5,000 8,000 accuracy EuroSAT 10 10,000 5,000 accuracy RESISC45 45 3,150 25,200 accuracy KITTI 4 6770 711 accuracy Country211 211 42,200 21,100 accuracy UCF101 101 9,537 1,794 accuracy Memes 2 8,500 500 ROC AUC SST2 2 7,792 1,821 accuracy ImageNet 1000 1,281,167 50,000 accuracy

### B.2 Detailed Model Scaling Results

Linear Probe. In Table 9, we present the detailed linear probe results of different scale CLIP models trained on the 30M dataset. The ViT-L/14 trained on RealSyn30M achieves an average performance improvement of 3.0% across 20 datasets compared to the model trained on the LAION30M.

Zero-shot Transfer. As shown in Table 10, ViT-L/14 trained on our proposed RealSyn30M outperforms LAION30M on 18 of 20 downstream datasets and achieves an average improvement of 5.5%.

Zero-shot Robustness. We present the detailed zero-shot robustness performance in Table 11, compared with LAION30M, the model trained on RealSyn30M boosts average robustness performance by 8.6% on ViT-L/14.

### B.3 Comparison with the Rewrite Dataset.

Due to LaCLIP [19] and Capsfusion [83] do not provide the model weights trained on the same scale dataset as ours, We compared the same scale model ViT-B/16 pre-trained on the RealSyn with DreamLIP [85].

As shown in Table 12, ViT-B/16 pre-trained on RealSyn15M can achieve 1.3% and 8.3% average improvement compared to DreamLIP proposed 15M Rewriting datasets based on YFCC15M [71]. Furthermore, compared with DreamLIP’s proposed 30M, which combines

##### Table 9: Linear probe on 20 downstream datasets. Pre-training different scale CLIP models on RealSyn30M and LAION30M, achieves 1.3%-3.0% average performance improvement.

CIFAR100

RESISC45

ImageNet

EuroSAT

Birdsnap

Average

CIFAR10

Country

Food101

SUN397

Aircraft

Flowers

UCF101

Caltech

Memes

STL10

KITTI

SST2

DTD

Cars

Pets

Model Dataset

LAION 76.1 94.5 80.0 47.4 70.3 82.3 45.9 74.7 80.3 89.8 89.5 95.6 95.5 84.5 72.6 15.2 76.6 56.2 60.0 64.3 72.6 RealSyn 81.2 95.4 81.8 48.4 74.5 73.4 45.2 74.2 84.1 91.3 90.6 97.2 96.5 89.2 74.5 19.0 82.6 55.0 56.2 68.5 73.9

ViT-B/32

LAION 82.1 95.1 81.4 57.5 73.4 87.3 47.1 76.1 84.4 91.5 92.7 96.8 95.6 86.8 70.8 17.6 80.3 59.5 65.6 68.8 75.5 RealSyn 87.5 95.8 82.5 59.4 77.5 81.0 48.7 77.9 88.9 92.5 94.2 98.3 96.9 91.5 70.8 22.1 85.1 60.6 64.7 73.9 77.5

ViT-B/16

LAION 84.7 96.4 83.5 59.2 75.5 88.5 46.6 77.8 85.0 92.6 94.3 97.9 95.9 88.0 71.7 18.7 81.1 58.6 64.6 71.2 76.6 RealSyn 90.3 97.5 86.2 64.3 79.7 83.6 51.4 79.6 90.0 94.5 94.8 98.9 96.6 92.7 73.8 25.0 86.4 63.8 66.1 76.7 79.6

ViT-L/14

##### Table 10: Zero-shot transfer on 20 downstream datasets. Pre-training different scale CLIP models on RealSyn30M and LAION30M,

- achieves 3.5%-5.5% average performance improvement.

Model Dataset

Food101

CIFAR10

CIFAR100

Birdsnap

SUN397

Cars

Aircraft

DTD

Pets

Caltech

Flowers

STL10

EuroSAT

RESISC45

KITTI

Country

UCF101

Memes

SST2

ImageNet

Average

ViT-B/32

LAION 58.9 85.9 63.1 17.4 54.8 61.0 4.3 36.4 65.5 82.0 41.3 91.3 40.3 43.7 24.3 7.2 47.4 51.5 50.1 44.9 48.6 RealSyn 67.5 89.0 65.2 15.0 60.6 39.2 7.9 37.8 70.5 84.0 42.2 93.8 59.9 61.9 27.7 10.6 56.7 52.5 50.1 50.9 52.1

ViT-B/16

LAION 67.6 89.1 63.5 20.8 55.7 66.9 5.4 39.0 70.2 84.9 42.9 94.3 31.1 45.4 34.0 8.7 52.2 54.5 50.6 49.4 51.3 RealSyn 75.8 89.6 64.7 18.9 64.3 48.2 7.9 41.2 76.0 87.5 45.2 95.1 56.8 64.3 27.1 13.1 59.1 54.5 54.0 55.9 54.9

ViT-L/14

LAION 70.8 88.8 69.5 22.8 61.6 69.7 4.9 40.8 68.0 87.3 42.2 95.3 41.5 53.7 25.9 10.4 54.7 54.1 51.8 51.5 53.3 RealSyn 80.7 94.1 73.1 20.9 66.4 53.6 10.1 48.1 72.8 89.4 49.8 96.2 68.5 70.1 32.2 15.3 63.9 54.1 56.9 59.5 58.8

Table 11: Zero-shot robustness comparison. Pre-training different scale CLIP models on RealSyn30M and LAION30M,

- achieves 4.2%-8.6% average performance improvement.

the former shows significant performance gains, with average improvements of 1.3% in linear probe, 3.5% in zero-shot transfer, and a robustness enhancement of 4.2%. The experimental results demonstrate that RealSyn when combined with existing datasets, achieves significant performance improvements, while also validating the robustness and extensibility of RealSyn.

Model Dataset IN-V2 IN-A IN-R ObjectNet IN-Sketch Average ViT-B/32

LAION 37.5 8.9 54.4 35.5 31.8 33.6 RealSyn 42.9 16.1 56.5 41.5 31.9 37.8

Table 13: Data Scaling Comparison. Merged 30M: LAION 15M

LAION 42.4 12.8 60.3 40.2 34.8 38.1 RealSyn 48.0 24.1 63.1 46.7 36.8 43.8

ViT-B/16

+ RealSyn15M.

LAION 45.1 17.1 64.9 43.1 39.0 41.8 RealSyn 52.8 34.7 71.6 50.4 42.4 50.4

ViT-L/14

Linear probe Avg

Transfer Avg

Robustness Avg

Model Dataset

##### Table 12: Comparison with DreamLIP’s rewriting dataset.

LAION15M 69.8 42.7 27.2 RealSyn15M 71.4 47.9 31.5 LAION30M 72.6 48.6 33.6 Merged30M 73.2 48.8 34.2 RealSyn30M 73.9 52.1 37.8

Linear probe Avg

Transfer Avg

ViT-B/16

Model Data Scale Dataset

DreamLIP 73.7 43.3

15M

RealSyn 74.9 51.6 30M

ViT-B/16

DreamLIP 77.2 53.1 RealSyn 77.5 54.9

### B.5 Detailed Results on Pure Image

To further extend our method for pure images, we conduct experiments on ImageNet [17]. For each image, we retrieve three semantically relevant real-world sentences from our pre-constructed sentence database and generate a single semantically augmented synthetic caption based on the top retrieved text. Following SimCLR [11], we utilize 4096 batch size and pre-train ResNet50 [28] for 90 epochs supervised by the text randomly selected from the three retrieved realistic texts and one synthetic text.

YFCC15M, CC12M [9] and CC3M [9] dataset, ViT-B/16 training on RealSyn30M can also outperform it in the linear probe and zero transfer. This significant improvement indicates that realistic knowledge is important for the CLIP model.

### B.4 Combine RealSyn with LAION

As shown in Table 13, integrating RealSyn15M with LAION15M yields average improvements of 0.6% in linear probe, 0.2% in zeroshot transfer across 20 datasets, and a robustness increase of 0.6% compared to LAION30M. Comparing RealSyn30M with LAION30M,

As shown in Table 14, compared with SimCLR [11] under the same conditions, the model trained on our constructed image-text pairs shows an average performance improvement of 2.1% across 12 downstream datasets. The results demonstrate that our method can

##### Table 14: Linear probe on 12 downstream datasets. Pre-training CLIP (ResNet50) on image-text pairs achieves 2.1% performance improvement.

Caltech101

CIFAR100

VOC2007

Birdsnap

###### Average

CIFAR10

Food101

SUN397

Aircraft

Flowers

DTD

Cars

Pets

Model Method

SimCLR 68.4 90.6 71.6 37.4 58.8 50.3 50.3 80.5 74.5 83.6 90.3 91.2 70.6 Real&Syn Texts 72.5 89.1 69.0 57.1 63.6 51.4 48.1 85.5 69.7 90.5 88.1 88.8 72.7

ResNet50

##### Table 15: Current Dataset Comparison. Comparison with large-scale image-text pre-training datasets.

#### Dataset #Images #Avg Tokens / Image #Avg Texts / Image Text Type Source Type

CC12M 12000000 − 1 Realistic Website YFCC15M 15000000 16 1 Realistic Website CapsFusion 120000000 − 1 Synthetic Image-Text Pair LAION400M 400000000 27 1 Realistic Website

RealSyn15M 15239498 40 4 Realistic & Synthetic Interleaved Image-Text RealSyn30M 30328852 38 4 Realistic & Synthetic Interleaved Image-Text RealSyn100M 100862786 36 4 Realistic & Synthetic Interleaved Image-Text

effectively transform pure images into high-quality image-text pairs through retrieval and generation for vision-language pre-training.

- C Further Analysis of RealSyn

- C.1 Compare with Existing Datasets

In Table 15, we compare our proposed RealSyn dataset with existing widely used large-scale image-text pre-training datasets. Compared to previous datasets, our proposed RealSyn dataset provides four textual descriptions for each image, with an average token length of 36-40, significantly higher than LAION400M and YFCC15M. Furthermore, unlike previous datasets, the RealSyn dataset is sourced from real-world interleaved image-text documents and includes both realistic and synthetic texts, thereby expanding the scope for future research exploration.

- C.2 Visualization of Examples In Figure 11 and Figure 12, we visualize additional image-text pairs randomly selected from our proposed RealSyn dataset. 𝑇𝑟𝑘 is the

𝑘-th retrieved semantically relevant real-world sentence and 𝑇𝑠𝑘 is the semantic augmentic caption for 𝑇𝑟𝑘. We also highlighted the image semantic-related information in green and marked the image size below the image.

### D Limitations

To provide more fine-grained visual information, this study employs vision expert models in conjunction with a Large Language Model (LLM) to generate synthetic text. Considering inference costs and efficiency, further exploration of Modified Large Language Models (MLLM) for synthetic text creation is suggested for the community. Additionally, constrained by computational resources, this paper constructs a 100M-scale RealSyn dataset exclusively using OBELICS [43]. Notably, the transformation paradigm presented here is directly applicable to other multimodal document datasets, including MMC4 [89] and OmniCorpus [45].

[Figure 110]

[Figure 111]

Raw Image:

[Figure 112]

[Figure 113]

[Figure 114]

This was a good gig !. grass street was an exceptional group to play at my husbands 6 0 th birthday bash .

The headlining band was a local band he w a s u s i n g o n a katrina benefit album & they had invited him to drop by .

We did our f i rst reunion show in 2 0 1 0 [ after coming together in 2 0 0 9 to celebrate the life of longtime friend and former employee paul ducharme ].

Grass street was an exceptional group to play at my husband 's 6 0 th birthday bash , a n d t h e y p l a y e d instruments like guitars, drums.

(506, 336)

[Figure 115]

It is not often the case that depression is simply a layer on top of a personality , the metaphorical " dark cloud ", but instead it

Lifting the shroud of m y s t e r y o n depression . it 's a topic that has vast implications for human health .

Creators who are yet s t r u g g l i n g t o breakthrough often get into depression .

Depression is not just a superficial layer on top of a personality , but rather it is deeply ingrained , like a dark cloud .

is something deeply ingrained .

(448, 336)

[Figure 116]

This large beach allows guests to grab and hammock or a lounge chair and relax .

Enjoy the beach club and all the equipment on the island !

L o u n g e r s a r e arranged on the hotel 's rocky private beach opposite .

Guests can relax on the large beach with chairs , hammocks , and daybeds under t h e p a l m t r e e s , enjoying the ocean view .

(505, 336)

[Figure 117]

Cheers and enjoy the beach !

I hope the sky is blue at your place . cheers from the beach at coquette point .

Great pics and even in winter a drink on a beach sounds bliss .

Enjoy the sunset while sipping a cup of beer on the beach .

(448, 336)

[Figure 118]

There are currently around 6 , 5 0 0 petrol stations ." this is just the beginning of the infrastructure build out ."

Nevertheless , the infrastructure needed already exists as a network of filling stations .

Additionally , a tank fuel station was built in java in 2 0 0 6 as well .

There are currently around 6 , 5 0 0 petrol stations , which is just the beginning of the infrastructure build out .

(598, 336)

[Figure 119]

There are photos , such as the 1 8 6 3 i m a g e o f t h e construction of the great south road .

figure 2 . excavation o n t h e b e n g a l nagpur railway ( 1 8 9 0 ). figure 2 shows a panoramic view of , again , the bengal nagpur railway .

it was during that period that they made this photograph of forest creek which was being turned over by thousands of e a g e r m i n e r s searching for gold .

1 8 6 3 image of the construction of the great south road is shown in the old b l a c k a n d w h i t e photo , with trees along the road .

(401, 336)

[Figure 121]

Raw Image:

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

My neighbor claims they still have sticky residue on some of their kitchen surfaces today .

Did you just wipe the counter with a soapy substance to clean it ?

The countertop is sprayed with lysol , which destroys the germs , and they are wiped away clean .

A kitchen counter with a sink and a bottle of cleaning agent, there is foam on the sink.

(448, 336)

[Figure 126]

While the sound was intimidating , the sight of the water stream sent shivers down my spine …

The sound of water flowing from barger 's way is ever present .

Flowing water is m e s m e r i z i n g , b l o c k i n g o u t distractions of forest and mind .

Sight of the stream of water with rocks in the forest sent shivers do w n my sp i n e , d e s p i t e t h e intimidating sound .

(503, 306)

[Figure 127]

Last week future posted a photo of himself rocking one of the sweater .

At what point are we going to say , you know what , he stole future 's style and we can 't condone this !?

According to his body statistics , future reaches a height of 6 feet 2 inches and weighs about 8 7 kg .

Last week , future posted a photo of himself wearing a white sweater and sunglasses , looking like a rapper .

(336, 336)

[Figure 128]

N o w t h e r e i s probably a video game where you can virtually cross stitch your heart 's sayings .

And finally , the background image references the trend of cross - stitching funny quotes .

B u t i t w a s a n impulsive moment of pattern deviation that u n l o c k e d t h e subversive potential of cross - stitch for her .

N o w t h e r e i s probably a video game where you can virtually cross stitch your heart 's sayings , such as " love is not an emotion ."

(336, 405)

[Figure 129]

Today , september 4 , is the feast of saint m o s h e h , b e t t e r k n o w n i n m o s t western culture as moses .

T o d a y i n t h e orthodox church we commemorate the lord 's friend and prophet moses , the man of god .

W e p r a y e d t o s t moses so that he may intercede to dispel this pandemic from humanity .

F e a s t o f s a i n t mosheh , also known a s m o s e s , i s celebrated today , september 4 th , in m o s t w e s t e r n cultures . The man in

(336, 448) blue sweater.

[Figure 130]

Photos of birds with their wings spread open , can be a true image of god .

To regain grace he became a bird of flight .

The sight of a bird in flight invoked the feeling of envy in their heart .

Photos of an eagle with their wings spread open , flying in the sky with the s u n s e t i n t h e background , can be a true image of god .

(336, 461)

