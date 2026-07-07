# arXiv:2601.10305v3[cs.CV]25Mar2026

[Figure 2]

## DanQing: An Up-to-Date Large-Scale Chinese VisionLanguage Pre-training Dataset

Hengyu Shen∗, Tiancheng Gu∗, Bin Qin, Lan Wu, Yuling Wu, Shuo Tan, Zelong Sun Jun Wang, Nan Wu, Xiang An, Weidong Cai, Ziyong Feng‡, Kaicheng Yang†

[Figure 3]

DanQing Team, Glint Lab

#### Abstract

Vision-Language Pre-training (VLP) models have achieved remarkable success by leveraging large-scale image-text pairs. While English-centric models like CLIP and SigLIP benefit from massive datasets (e.g., LAION-400M), the development of Chinese VLP remains bottlenecked by the lack of high-quality, large-scale open-source data. In this paper, we present DanQing, a large-scale Chinese cross-modal dataset containing 100 million high-quality image-text pairs curated from Common Crawl. To ensure superior data quality, we develop an effective systematic pipeline comprising data source selection, text refinement, visual diversification, and cross-modal cross-batch filtering, thereby effectively mitigating the intrinsic noise prevalent in web data. Notably, DanQing incorporates data from 2024–2025, enabling models to capture contemporary semantic trends and emerging concepts. Extensive experiments via continued pretraining of SigLIP2 models demonstrate that DanQing consistently outperforms existing Chinese datasets across diverse downstream tasks, including zero-shot classification, cross-modal retrieval, and Chinese-centric large multimodal model tasks. Furthermore, in-depth analysis of DanQing reveals that it exhibits a more balanced semantic distribution and superior scaling capability compared to existing datasets. To facilitate further research in Chinese vision-language pre-training, we will open-source the DanQing dataset under the Creative Common CC-BY-NC 4.0 license.

[Figure 4]

[Figure 5]

Webpage https://deepglint.github.io/DanQing

GitHub https://github.com/deepglint/DanQing ModelScope https://www.modelscope.cn/datasets/deepglint/DanQing

HuggingFace https://huggingface.co/datasets/DeepGlint-AI/DanQing100M

#### 1 Introduction

The proliferation of web-scale data provides a robust foundation for contrastive vision-language representation learning [22]. By aligning dual-encoder architectures through image-text correspondence, frameworks like CLIP [47] have demonstrated remarkable generalization across diverse downstream tasks including image captioning [42; 33; 69], object detection [23; 34; 76], semantic segmentation [32; 49; 65], and cross-modal retrieval [73; 26]. Given the efficacy of CLIP, this promising paradigm has garnered significant attention from both industry and academia as a potential pathway toward next-generation foundational AI models [20; 18].

The success of Vision-Language Pre-training (VLP) is primarily driven by the synergy between architectural innovation and data scaling. On one hand, advanced modeling techniques such as ViT [13] and BERT [12], optimized via InfoNCE [47] or Sigmoid-based [71] contrastive losses, have significantly enhanced the ability of dual-tower models to learn semantically rich multimodal embeddings. On the other hand, the availability of large-scale datasets like LAION-5B [51] and

∗ Equal Contribution. ‡ Team Leader. † Project Leader.

DataComp-1B [18] has provided an essential foundation for VLP. However, while English-centric resources continue to expand, the development of Chinese image-text datasets has substantially lagged. Notably, the most recent dataset [62] is introduced over three years ago. As a result, despite the recognized importance of data scale, the Chinese vision-language pretraining remains limited. Furthermore, existing Chinese resources often suffer from significant data decay, where a high proportion of inaccessible image URLs severely impairs model training and reproducibility. This scarcity of high-quality, persistent data bottlenecks Chinese cross-modal representations.

To bridge this gap, we present DanQing, a large-scale Chinese dataset comprising 100 million high-quality image-text pairs collected after 2024. Specifically, we develop an effective systematic pipeline (Fig. 1) to refine

Dataset Year Language Availability Size Success Rate

CC3M [52] 2018 English Yes 3.1M ≈60% CC12M [6] 2021 English Yes 12M ≈60% RedCaps [11] 2021 English Yes 12M WIT [54] 2021 Multilingual Yes 11.5M YFCC100M [57] 2014 English Yes 100M ≈70% COYO [4] 2022 English Yes 700M LAION-400M [50] 2021 English Yes 400M RealSyn [22] 2025 English Yes 100M -

- 1 billion raw pairs into a high-quality subset. This pipeline consists of data source selection, text refinement, visual diversification, and cross-modal cross-batch filtering, each designed to address distinct sources of noise in web data. As a result, we filter out 90.46% of the raw data, substantially reducing intrinsic noise and enhancing overall dataset quality. Extensive evaluations via continued pre-training of SigLIP2 models demonstrate that DanQing consistently outperforms existing Chinese datasets across diverse downstream tasks, including zero-shot classification, cross-modal retrieval, and Chinese-centric large multimodal model tasks. Furthermore, in-depth analysis of DanQing reveals that DanQing exhibits a more balanced semantic distribution and superior scaling capability compared to existing datasets. The main contributions of this paper are summarized as follows:

- • We develop an effective data filtering pipeline tailored for processing large-scale Chinese imagetext pairs obtained from the Internet.
- • We release an up-to-date large-scale Chinese image-text dataset DanQing, which comprises nearly 100 million Chinese image-text pairs collected after 2024.
- • We conduct extensive experiments across multiple downstream tasks to demonstrate the effectiveness and scalability of DanQing.

- 2 Related Work

Product1M [72] 2021 Chinese Yes 1M WudaoMM [70] 2022 Chinese Yes 5M M6-Corpus [36] 2021 Chinese No 60.5M Wukong [20] 2022 Chinese Yes 100M ≈85% TaiSu [39] 2022 Chinese Yes 166M 100% Zero [62] 2022 Chinese Yes 250M ≈60% DanQing 2025 Chinese Yes 100M 100%

Table 1: Overview of existing VLP datasets.

##### 2.1 Vision-Language Pretraining

As a seminal work in vision-language pre-training, CLIP [47] has demonstrated exceptional zeroshot recognition and transfer capabilities. Building on this paradigm, recent studies have introduced various enhancements [68; 21; 61]. For instance, SLIP [43] integrates self-supervised learning with image-text pretraining to improve representation quality, while ALIP [67] employs a gating mechanism for dynamic sample reweighting to mitigate the impact of noisy data. DFN [15] proposes novel data filtering networks to construct high-quality image-text datasets. Furthermore, MetaCLIP [64] and MetaCLIP2 [8] leverage metadata derived from CLIP’s semantic concepts to curate balanced data subsets. To enable larger batch sizes, SigLIP [71] and SigLIP2 [58] adopt a sigmoid-based loss, eliminating the need for global normalization. In the context of Chinese vision-language pre-training, ChineseCLIP [66] introduces a two-stage framework comprising locked-image tuning followed by contrastive tuning. Similarly, R2D2 [62] enhances representation learning through a preranking-ranking strategy combined with bidirectional distillation. Despite these advances,

[Figure 8]

Figure 1: Overview of the DanQing dataset construction pipeline.

the scarcity of large-scale, high-quality Chinese image-text datasets bottlenecks vision-language pretraining, hindering model scalability and generalization.

- 2.2 Large-Scale Image-Text Dataset

The impressive performance of CLIP [47] across downstream tasks is primarily attributed to the availability of massive, high-quality image-text data. To further advance model capabilities, numerous large-scale image-text pair datasets have been introduced in recent years (Tab. 1). The YFCC100M [57] dataset provides a comprehensive record of photo and video sharing trends on Flickr from its inception in 2004 through early 2014. LAION400M [50] comprises 400 million imagetext pairs sourced from Common Crawl and has become a standard benchmark for vision-language pretraining. COYO-700M [4] collects approximately 10 billion image-alt-text pairs from HTML documents in Common Crawl (from October 2020 to August 2021), employing efficient imageand text-level filtering to remove uninformative pairs at minimal cost. To support data filtering research and benchmarking, DataComp [18] assembled a pool of 12.8 billion image-text pairs for competition tracks, model training, and evaluation. However, these datasets are predominantly based on English image-text pairs, while large-scale Chinese image-text datasets remain scarce. To address this gap, the Wukong [20] dataset, comprising 100 million Chinese image-text pairs collected from the web, has been released. Taisu [39] further advances this effort by introducing an automatic filtering framework, resulting in a large-scale, high-quality Chinese multimodal dataset containing approximately 166 million images and 219 million Chinese captions. Leveraging user click-through rates and diverse textual information for each image, the Zero [62] dataset offers 250 million images and 750 million corresponding Chinese texts, significantly advancing resources for Chinese vision-language pretraining. Despite these advancements, existing Chinese image-text datasets still lag behind their English counterparts in both scale and quality.

- 3 DanQing Dataset

- 3.1 Training Objective of DanQing

The DanQing dataset aims to enhance the Chinese multimodal embedding capabilities of CLIP-style models. Given L2-normalized embeddings v, t ∈ Rd from the image encoder fv(·) and text encoder ft(·), we adopt the SigLIP [58] objective, which reformulates alignment as independent binary classification tasks using a sigmoid loss:

### L = −∑

i,j

Ii=j log σ(sij) + Ii̸=j log(1 − σ(sij)) ,

20M

Width Height

18M

| |
|---|

Min(W, H)

15M

12M

10M

8M

- 0.5M

- 1.0M

- 1.5M

- 2.0M

- 2.5M

- 3.0M

- 3.5M

- 4.0M Avg: 22.0

5M

2M

0

0

5 10 15 20 25 30 35 40 45 50 55 60

200 400 600 800 1000 1200 1400 >1500

(a) Image Resolution Distribution.

(b) Text Chinese words length.

Figure 2: Overview of data characteristics in DanQing.

where sij = (vi · tj)/τ + b is the scaled similarity with bias b. Compared to the standard CLIP [47] cross-entropy over a batch of B image-text pairs, this logistic formulation avoids batch coupling and scales more favorably to large batches and distributed training.

##### 3.2 Curation of DanQing

Data Source Selection. We first collect raw image-text pairs from the Common Crawl (2024–2025). The data collection is partitioned into seven batches and processed in parallel to ensure efficiency. By filtering for the “zho” language tag, we obtain an initial pool of approximately 1.05 billion pairs. To mitigate the inherent noise in web-scale data, we implement a coarse-grained curation based on three criteria: ① Source Reliability: We exclude the pairs originating from websites listed in a manually curated blacklist of low-quality sources. ② Textual Constraints: We retain the data which text contain 5 to 60 Chinese words. ③ Content Safety: We use a lightweight 1M-parameter binary classifier [1] filters unsafe content. This stage yields 706M pairs (∼67%), and subsequent downloading achieve a 67% success rate, resulting in 475M accessible image–text pairs.

Text Refinement. We implement a multi-stage text refinement pipeline across four dimensions: linguistic structure, text quality, information density, and content safety, primarily due to its lower computational overhead. ① Linguistic Structure: We employ FastText [25] to identify and retain Chinese text based on language identification confidence, followed by OpenCC [5] to standardize all content to Simplified Chinese. ②Text Quality: To ensure grammatical and lexical integrity, we discard samples that lack nouns or contain more than five [UNK] tokens after SigLIP2 tokenization [58]. ③ Information Density: Following RealSyn [22], we remove emojis and special characters, and apply

an entropy-based semantic filter H = − ∑iL=0 P(ci) log2 P(ci), where P(ci) denotes the probability of token ci and L denotes the total number of words in the text. Captions with H < 6 × 10−4 are eliminated and this threshold is empirically determined to filter out low-content captions. ④ Content Safety: We utilize a 20M-parameter NSFW detector [2] and the Baidu DataBuilder service to filter advertisements, sensitive political content, and territorial disputes, based on their proven effectiveness in prior work. This process reduces the corpus from 475M to 397M pairs (a 16.4% reduction), significantly improving the signal-to-noise ratio for subsequent training.

Visual Diversification. To rigorously ensure both perceptual quality and semantic diversity, we establish a multi-stage visual diversification pipeline that systematically addresses visual fidelity, information density, perceptual and semantic redundancy, and content safety. ①Visual Fidelity: We retain images with aspect ratios between 1:3 and 3:1 and a minimum edge length greater than 100 pixels. Images with low pixel intensity variation (σ < 2), indicative of uninformative content, are removed. Blurry samples are filtered out by requiring a Laplacian variance (computed via OpenCV [56]) of at least 1000. ②Information Density: Image complexity is quantified via entropy

(H = − ∑255i=0 P(i) log2 P(i), where P(i) denotes the probability of pixel value i) and exclude images with entropy below 3 to remove low-information samples. ③Perceptual and Semantic Redundancy: To suppress duplication, inspired by previous work [68], we extract image embeddings with ChineseCLIP-L14 [66] and compute pairwise cosine distances. Images with cosine distances below an

地域美食/Cuisine (3.68%)

时尚/Fashion (6.11%)

科技/Technology (4.32%)

Key words: 蓝牙, 耳机, 电动汽车, 智能驾驶 (Bluetooth, Headphones, Electric Vehicle, Inteligent Driving)

Key words: 夏日穿搭,流行,新款, 球鞋, 托特包 (Summer styling, Popular,New arrivals, Sneakers, Tote bag)

Key words: 茶叶, 米饭, 火锅, 烧烤, 白酒, 饮料, 咖啡 (Tea, Rice, Hot pot, Barbecue, Baijiu, Beverages, Cofee)

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

家居/Furnishing (3.26%) 体育/Sports (2.93%)

旅游/Tourism (3.18%)

Key words: 建筑, 家具, 设计, 装潢, 材料 (Architecture, Furniture, Design, Decoration, Materials)

Key words: 旅游文化, 景点, 活动, 住宿, 攻略, 摩天轮 (Tourist Atractions, Activities, Accommodation, Travel Guide)

Key words: 体育赛事, 足球, c罗,战报, 篮球 (Sports events, Footbal, Cristiano Ronaldo, Match report, Basketbal)

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 3: Visualization of popular topic in the DanQing dataset, generated via BERTopic [19] on 10M subset.

empirically determined threshold (β = 0.1) are divided into the same set using the Union-Find algorithm [55]. Within each set, we retain only the central image (the image nearest to the centroid of the set) and only the centroid-closest image is retained per set. ④Content Safety: We apply an 86M-parameter NSFW detection model [53] to remove pairs with the highest risk scores, ensuring content safety. Through this process, the dataset size is reduced from 397M to 178M pairs (44.8% retention), substantially improving visual and semantic quality for downstream applications.

Cross-Modal Cross-Batch Filtering. Following LAION400M [50], we leverage an expert model to further refine the dataset based on image-text alignment. Specifically, we compute L2 distance using Chinese-CLIP-L14 [66] and retain pairs within the [1.06, 1.24] interval. This thresholding strategy ensures high-quality alignment: scores below 1.06 indicate weak semantic correlation, while those exceeding 1.24 often correspond to images dominated by OCR text rather than descriptive content. This stage removes 25M pairs. Finally, we apply perceptual and semantic redundancy filtering to the samples across seven batches, eliminating an additional 54 million redundant pairs. As a result, this pipeline produces a curated dataset comprising 99,892,381 high-quality image-text pairs.

##### 3.3 Statistic of DanQing

Data Characteristics. As illustrated in Fig. 2a, we analyze the general characteristics of the DanQing dataset. We assess image resolutions in terms of width, height, and minimum dimension, revealing a broad spectrum of visual scales. While most images fall within the 300 to 500 pixel range, a considerable proportion exceeds 1,024 pixels. This extensive coverage facilitates the extraction of robust, scale-invariant features for vision-language representation learning. Such visual diversity is essential for ensuring generalization to real-world images, where object scales and orientations vary substantially. In addition, we present the distribution of text lengths in Fig. 2b. DanQing contains a total of 2.2B Chinese words, with an average length of 22 words per sample. The length distribution spans a broad range from 5 to 60 tokens, with the majority concentrated between 5 and 40. This wide distribution enables models to learn representations across different levels of textual granularity.

Topic Modeling. To further investigate the semantic diversity of DanQing, we implement a topic modeling pipeline based on BERTopic [19]. Specifically, we randomly sample 10M image-text pairs and extract text embeddings using Chinese-CLIP-L/14 [66]. To address the challenges of high-dimensional clustering, we apply Uniform Manifold Approximation and Projection [41] for dimensionality reduction. Subsequently, we use HDBSCAN [19] to identify distinct semantic clusters, setting the minimum cluster size to 1,000 to ensure cluster stability and reduce noise. We then utilize class-based TF-IDF to extract representative keywords for each topic. Fig. 3 visualizes six prevalent topics, including fashion, technology, cuisine, furnishing, tourism, and sports. These results indicate that DanQing encompasses a wide variety of real-world domains, providing a comprehensive foundation for large-scale vision-language representation learning. For a more detailed topic analysis, please refer to the Appendix C.2.

[48]Country211

[44]Flowers102

[16]Caltech101

[14]VOC2007

[7]RESISC45

[29]CIFAR10

[31]MNIST

[27]Memes

[3]Food101

[28]Cars

[46]Pets

[9]DTD

Dataset

Avg. Model Architecture: SigLIP2-B/32@256

Baseline [58] 77.0 85.1 8.2 35.9 55.1 81.9 37.6 61.9 56.3 76.3 49.4 69.0 57.8 Wukong [20] 78.6 91.7 9.5 42.6 61.2 83.0 61.4 71.3 58.1 75.1 53.8 75.6 63.5 Zero∗ [62] 79.3 92.2 10.8 45.1 64.7 86.3 63.2 76.7 58.9 74.5 49.6 77.3 64.9

- TaiSu∗ [39] 78.5 90.9 5.7 43.5 53.6 83.5 52.4 62.9 53.3 58.9 54.0 77.3 59.5

- DanQing 79.7 93.0 9.9 46.4 66.6 83.4 58.5 78.7 61.4 76.0 54.4 77.1 65.4 Model Architecture: SigLIP2-B/16@256

Baseline [58] 77.3 85.4 10.7 35.3 60.8 83.9 38.1 65.0 59.8 81.0 51.0 71.0 59.9 Wukong [20] 78.4 90.3 12.7 44.8 68.7 81.5 63.6 76.0 59.0 80.8 55.0 78.4 65.8 Zero∗ [62] 79.5 91.3 13.9 45.6 70.5 84.6 65.5 78.9 60.6 80.2 51.0 79.0 66.7 TaiSu∗ [39] 78.6 89.3 7.0 44.6 58.1 82.2 54.3 65.9 55.8 62.1 54.2 79.2 60.9

- DanQing 80.2 93.2 13.3 48.0 71.6 83.5 62.2 81.8 63.5 81.7 53.2 79.6 67.7 Model Architecture: SigLIP2-L/16@256

Baseline [58] 76.7 88.5 15.9 44.8 72.0 80.8 49.7 84.3 63.9 87.4 49.2 68.9 65.2 Wukong [20] 80.3 96.1 20.5 48.2 78.3 84.9 74.3 84.5 65.7 86.5 55.0 78.1 71.0 Zero∗ [62] 82.4 96.3 22.6 48.9 81.9 86.4 75.9 89.5 65.3 87.8 52.0 79.7 72.4 TaiSu∗ [39] 81.7 94.8 13.1 44.3 68.9 74.2 64.5 79.1 59.4 70.7 55.6 79.7 65.5 DanQing 83.5 96.7 22.4 49.2 83.8 85.2 75.0 90.0 64.8 88.7 55.8 79.9 72.9

- Table 2: Zero-shot image classification performance using models pretrained on different datasets. ∗ indicates random sampling of 100 million image-text pairs. The best and second best scores are in boldface and underlined.

#### 4 Experiments and Results

##### 4.1 Implementation Details

To validate the effectiveness of the DanQing dataset, we continue pre-training the SigLIP2 [58] model for 2 epochs using 16 × A800 (80G) GPUs. We employ AdamW [40] as the optimizer, initializing it with a learning rate of 1e-5 and a weight decay of 0.1. The batch size is set to 768 × 16. The momentum parameters β1 and β2 are set to 0.9 and 0.98, respectively. A learning rate warmup strategy is applied during the first 1,000 iterations to ensure training stability. The input image size is 256 × 256, and the input text sequence length is truncated or padded to 64. To ensure a fair comparison in our experiments, we randomly select 100M samples from both the Zero and TaiSu datasets for training.

##### 4.2 Main Results

Zero-shot Classification. As presented in Tab. 2, we perform continual pre-training on three SigLIP2 backbone models (B/32, B/16, and L/16) using the Wukong, Zero, TaiSu, and DanQing datasets. Continual pre-training with these datasets significantly improves model performance, with DanQing yielding the most notable gains. Specifically, DanQing enhances performance by 7.6%, 7.8%, and 7.7% on B/32, B/16, and L/16, respectively. Additionally, compared to the Wukong dataset, DanQing achieves a 1.9% performance improvement across all three backbone models. Similarly, compared to the Zero dataset, DanQing provides average performance improvements of 0.5%, 1.0%, and 0.5% on B/32, B/16, and L/16, respectively. These results highlight the high quality of the DanQing dataset and its effectiveness in Chinese image-text contrastive learning tasks.

Cross-Modal Retrieval. To further validate the effectiveness of the DanQing dataset, we conduct comparisons on cross-modal retrieval tasks. As shown in Tab. 3, on the Flickr30K-CN, MSCOCO-CN, and MUGE datasets, DanQing achieves average retrieval performance improvements of 2.4%&2.4%,

Flickr30K-CN [30] MSCOCO-CN [35] MUGE [36] Text to Image Image to Text Text to Image Image to Text Text to Image Image to Text

Dataset R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 Avg. Model Architecture: SigLIP2-B/32@256

Baseline [58] 45.4 71.2 80.6 67.7 88.9 94.6 49.6 77.2 87.8 51.8 81.1 90.9 38.3 61.7 69.9 35.3 60.7 69.8 67.9 Wukong [20] 49.8 75.8 83.7 68.2 89.8 95.6 54.4 81.6 90.7 56.5 84.8 83.2 55.1 77.9 85.1 44.0 71.2 80.1 74.3 Zero∗ [62] 49.5 76.5 84.4 68.7 90.5 95.1 53.9 84.0 92.1 56.9 84.6 93.3 54.5 77.7 84.9 42.1 69.4 78.5 74.3 TaiSu∗ [39] 60.5 84.2 90.3 77.8 94.4 97.2 65.7 90.7 96.0 65.5 88.9 94.5 56.2 78.2 84.7 44.1 71.2 80.3 78.9 DanQing 54.2 79.0 86.6 73.0 92.2 96.3 60.1 84.5 93.8 61.0 88.3 96.3 54.8 78.1 84.9 45.3 72.1 80.7 76.7

###### Model Architecture: SigLIP2-B/16@256

Baseline [58] 51.3 76.6 84.7 73.5 93.3 96.7 51.9 79.7 89.6 54.7 82.5 91.9 41.6 64.6 73.4 38.9 64.3 73.5 71.3 Wukong [20] 56.5 81.8 88.4 74.8 94.2 97.8 57.5 83.3 92.0 61.0 86.0 93.7 60.1 81.7 87.7 48.8 75.3 83.2 78.0 Zero∗ [62] 58.2 83.7 90.4 74.9 93.4 96.9 58.7 86.0 94.4 60.0 84.8 93.1 59.6 80.8 86.8 46.2 72.9 81.3 77.9 TaiSu∗ [39] 68.2 89.0 93.9 83.8 97.2 99.4 68.8 93.0 97.1 67.1 90.1 95.9 60.3 81.0 86.8 48.4 74.9 83.0 82.1 DanQing 61.1 84.9 90.9 80.6 95.0 97.9 62.3 86.6 94.4 64.7 88.5 96.1 60.4 81.3 87.3 50.3 76.3 83.9 80.1

###### Model Architecture: SigLIP2-L/16@256

Baseline [58] 53.5 78.1 85.5 79.6 95.7 98.3 51.7 79.9 89.0 55.4 81.9 90.5 50.2 71.1 78.5 45.6 70.4 78.5 74.1 Wukong [20] 62.8 86.2 91.5 81.7 96.2 98.5 61.0 85.9 93.5 62.9 88.7 95.1 66.6 84.6 90.1 55.8 80.7 87.4 81.6 Zero∗ [62] 64.3 87.9 93.4 78.4 95.5 98.6 61.6 87.2 94.7 62.1 87.2 94.6 65.9 85.3 90.3 53.9 79.0 86.2 81.5 TaiSu∗ [39] 72.6 91.7 95.8 87.8 98.7 99.7 71.4 92.6 97.3 69.2 91.6 96.8 66.0 85.0 90.0 55.1 80.1 86.7 84.9 DanQing 70.2 90.3 94.7 86.3 98.7 99.6 65.9 90.5 95.4 68.0 92.6 97.4 67.5 84.9 90.1 56.8 81.2 87.5 84.3

- Table 3: Cross-modal retrieval performance on short-caption datasets for models pretrained on various large-scale Chinese image-text datasets. ∗ indicates random sampling of 100 million imagetext pairs. The best and second-best results are highlighted in bold and underlined, respectively.

2.1%&2.2%, and 2.7%&2.8% over the Wukong and Zero datasets across three different backbone models. Notably, TaiSu achieves strong retrieval performance, particularly on Flickr30K-CN and MSCOCO-CN, largely because it augments web-crawled tags with concise synthetic captions generated by OFA-Large [60]. The close alignment between the distribution of these captions and the target benchmarks results in substantial performance gains.

In addition, inspired by recent work [63], we further evaluate long caption-based crossmodal retrieval using the DCICN and DOCCI-CN datasets as shown in Tab. 4. We surprisingly observe that, under the same pre-training context length (up to 64 tokens), the DanQing dataset demonstrates superior long caption-based cross-modal retrieval performance compared to other datasets. With the SigLIP2-L/16@256 model, DanQing achieves average performance improvements of 12.8%, 9.0%, and 8.9% over the Wukong, Zero, and TaiSu datasets, respectively. This improvement is mainly because DanQing exhibits higher semantic density and a greater proportion of high-quality texts (Fig. 4).

DCI-CN [59] DOCCI-CN [45] Text to Image Image to Text Text to Image Image to Text

Dataset R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 Avg. Model Architecture: SigLIP2-B/32@256

- Baseline [58] 7.7 18.2 23.8 8.7 19.3 25.3 11.0 26.5 36.0 14.3 32.1 41.6 22.0 Wukong [20] 10.2 22.7 29.6 11.3 23.8 30.7 16.3 34.8 44.8 15.7 35.7 46.3 26.8 Zero∗ [62] 10.9 24.4 32.0 11.2 23.7 30.9 17.1 35.8 45.8 17.8 38.5 50.1 28.2 TaiSu∗ [39] 11.3 24.0 31.5 12.5 26.2 33.1 16.8 37.1 47.4 16.6 38.8 49.1 28.7 DanQing 13.1 27.4 35.0 12.6 26.1 33.7 19.8 42.0 52.8 18.7 40.4 51.5 31.1

Model Architecture: SigLIP2-B/16@256

- Baseline [58] 8.7 19.6 25.9 10.4 21.0 26.8 13.0 30.2 39.8 16.6 35.8 46.8 24.6 Wukong [20] 12.2 25.3 32.6 12.8 25.8 32.6 17.7 39.1 49.4 18.0 38.9 50.1 29.5 Zero∗ [62] 12.9 26.9 34.8 12.9 25.6 33.0 18.8 39.5 49.5 19.0 40.5 51.7 30.4 TaiSu∗ [39] 13.2 27.1 34.7 14.6 28.6 36.0 19.1 40.5 51.5 18.9 41.1 51.9 31.4 DanQing 15.3 30.7 38.4 15.0 29.3 36.9 23.6 47.3 57.8 22.3 44.9 56.6 34.8

###### Model Architecture: SigLIP2-L/16@256

Baseline [58] 16.7 30.9 37.5 16.3 30.6 38.0 29.3 53.5 64.3 29.0 54.4 64.9 38.8 Wukong [20] 23.1 41.0 48.6 21.8 38.3 46.0 37.1 64.2 74.2 33.3 59.2 69.8 46.4 Zero∗ [62] 24.8 43.0 51.4 24.5 41.6 49.9 37.6 66.4 75.8 38.4 67.4 77.8 49.9 TaiSu∗ [39] 26.6 44.4 52.2 26.0 43.2 51.1 41.1 67.3 76.8 37.1 63.2 73.2 50.2 DanQing 31.3 50.7 58.4 30.5 49.9 58.2 48.7 76.4 84.5 44.8 72.2 81.5 57.3

Table 4: Cross-modal retrieval performance on long-caption datasets for models pretrained on various datasets.

Chinese-Centric Large Multimodal Model Tasks. We employ SigLIP2 variants that continue pretrained on different Chinese image–text datasets as vision encoders in Large Multimodal Models (LMMs) to evaluate their compatibility and utility for Chinese-centric multimodal reasoning.

- 0 (a) Text content word density

- 0.5M

- 1.0M

- 1.5M

- 2.0M DanQing Wukong Zero

| |
|---|

| |
|---|

25 50 75 100 125 150 175 200

0

- 0.5M

- 1.0M

- 1.5M

- 2.0M

- 2.5M

- 3.0M

- 3.5M

60 80 100 120 140 160 180

0

20K

40K

60K

80K

(b) Text perplexit

Figure 4: Text Quality Analysis. Randomly sample 10M subsets from DanQing, Wukong, and Zero, then compare their content word density and perplexity.

MMBench (Dev)

MME-RW CMMMU OCRBench

Dataset CN [38] EN [38] CN [75] eval [74] V2 [17] Avg. Model Architecture: SigLIP2-L/16@256 + Qwen2-7B

Baseline [58] 72.9 73.6 43.1 38.7 15.0 48.7 Wukong [20] 73.5 75.6 43.8 39.7 15.0 49.5 Zero∗ [62] 72.9 75.0 42.3 39.4 15.7 49.1 TaiSu∗ [39] 73.5 75.3 42.8 39.8 15.1 49.3 DanQing 74.0 75.3 45.4 39.7 16.0 50.1

Table 5: Performance of LLaVA-NeXT-style models on Chinese-centric LMM downstream benchmarks, utilizing vision encoders pretrained on various datasets.

Specifically, we strictly adhere to the LLaVA-NeXT [37] training pipeline and data configuration, varying only the vision encoder (SigLIP2-L/16) to isolate the effect of pretraining data on downstream multimodal capabilities. Results in Tab. 5 show that DanQing surpasses existing datasets, achieving a new stateof-the-art average score of 50.1% (vs. 49.5%). These performance improvements demonstrate the higher data quality of the DanQing and highlight its potential for Chinese-centric tasks in LMM architectures.

5 Analysis

- 5.1 Text Quality Analysis

We further explore the text quality of DanQing through two metrics: semantic word density and perplexity (PPL), as illustrated in Fig. 4. Specifically, we randomly sampled 10M texts from DanQing, Wukong, and Zero for comparison. We use the jieba 1 toolkit to identify semantic words (i.e., nouns, verbs, and adjectives) in each sentence and compute their proportions as a measure of semantic density. As illustrated in Fig. 4a, DanQing exhibits significantly higher semantic density than the other datasets, which enables the model to acquire more effective semantic information. Additionally, we compute sentence-level perplexity using a pre-trained Chinese BERT model [10]. As shown in

- Fig. 4b, the number of samples in DanQing with PPL scores within the [50, 200] range is substantially higher than that of the other datasets. This range suggests an optimal level of linguistic complexity (neither overly simplistic nor incoherent), thereby highlighting the superior quality of our dataset for vision-language pre-training.

- 5.2 Scaling Capability

Scaling capability determines the upper bound of vision-language pretraining models. To this end, we compare the data and model scaling capabilities of the proposed DanQing dataset with those of the widely used Wukong dataset, and report the average performance on zero-shot classification and retrieval (long&short caption) tasks in Fig. 5.

Data Scaling. To evaluate the scalability of our proposed DanQing, we pretrain SigLIP2-B/32 for two epochs on varying data scales (10M, 30M, 60M, and 100M) and compare the performance trajectories of DanQing and Wukong. As shown in Fig. 5a, DanQing consistently achieves significant performance gains over Wukong across all data scales, with the improvements becoming more

- 1 https://github.com/fxsjy/jieba

<0.4 0.5 0.6 0.7 0.8 0.9 1.0

###### Classification

###### Retrieval

###### Classification

###### Retrieval

66

75

74

54

DanQing Wukong

| |
|---|

72

64

52

70

70

50

62

65

68

48

60

60

66

46

58

64

55

44

56

62

B(86M) L(303M) So(400M) g-opt(1B)

B(86M) L(303M) So(400M) g-opt(1B)

0M 10M 30M 60M 100M

0M 10M 30M 60M 100M

(a) Data Scaling on SigLIP2-B/32@256

(b) Model Scaling on 30M subset

- Figure 5: Data scaling and model scaling capability comparison between DanQing and Wukong.

pronounced as the scale increases. Notably, Wukong’s retrieval performance plateaus beyond 30M samples, whereas DanQing continues to improve steadily from 30M to 100M, indicating that our dataset provides more effective supervision for large-scale vision-language pretraining.

Model Scaling. To further investigate model scaling, we conduct experiments as illustrated in Fig. 5b. Specifically, we utilize 30M subsets from both DanQing and Wukong to train SigLIP2 models across various scales, including Base (86M), Large (303M), So (400M), and g-opt (1B). The results show that DanQing consistently outperforms Wukong in both classification and retrieval tasks. Moreover, DanQing exhibits a steeper scaling curve, better leveraging increased model capacity for representation learning.

5.3 Image Semantic Balance

0 2000 4000 6000 8000 10000

- 100

- 101

- 102

- 103

- 104 DanQing

Wukong

- Figure 6: Clustering distribution of 10M subsets of DanQing and Wukong.

Fig. 6 illustrates the semantic distribution of images in DanQing compared to the Wukong dataset. For quantitative analysis, we randomly sample 10 million images from each dataset and cluster them into 10k groups using the FAISS library [24]. We rank these clusters by the number of samples they contain. The results indicate that DanQing achieves a significantly more balanced and uniform semantic distribution than Wukong, effectively mitigating the long-tail effect. This increased uniformity suggests broader coverage of the visual manifold, which is

essential for learning rare or long-tail concepts during vision-language pretraining.

##### 5.4 Image-Text Alignment

- In Fig. 7, we illustrate the distribution of image-text similarity scores for 10M-sample subsets of DanQing and Wukong [20]. We employ the state-of-the-art Chinese retrieval model FG-CLIP2-L/16 [63] to extract multimodal features and compute their cosine similarity. The results indicate that DanQing consistently achieves higher similarity scores than Wukong, with significantly more samples exceeding the 0.15 threshold. This demonstrates that DanQing provides stronger semantic consistency between images and texts. It is noteworthy that the DanQing dataset contains a significantly higher proportion of samples in the 0 to 0.05 similarity range compared to Wukong. This is primarily because DanQing comprises data from 2024 and 2025, which includes a substantial amount of novel semantic content (as shown in Fig. 8). These findings help explain why models trained on DanQing demonstrate significant performance improvements in retrieval tasks, further highlighting the dataset’s ability to enrich models with comprehensive semantic knowledge.

- 0.1M

- 0.2M

- 0.3M

- 0.4M

- 0.5M

- 0.6M

- 0.7M

DanQing

Wukong

0

0.00 0.05 0.10 0.15 0.20 0.25 0.30

Figure 7: Similarity distributions for 10M subsets of DanQing and Wukong.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

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

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

(a) “黑神话:悟空” (Black Myth: Wukong) (b) “小米SU7” (Xiaomi SU7)

Figure 8: New concept understanding capability comparison between DanQing with existing datasets. The scores represent the softmax-normalized values of the cosine similarities among three image-text pairs.

##### 5.5 New Concept Understanding

- In Fig. 8, we evaluate the capability of SigLIP2-L/16@256 models pre-trained on different Chinese datasets to understand emergent concepts. Specifically, we select internet buzzwords that appear after 2024, such as “黑神话:悟空” (Black Myth: Wukong) and “小米SU7” (Xiaomi SU7). We pair these keywords with their corresponding ground-truth images, along with several semantically related distractors (e.g., traditional cartoon or TV adaptations of “Wukong”, other Xiaomi products, and unrelated food items). By calculating the image-text similarity scores, we observe that the model trained on DanQing consistently assigns the highest confidence to the correct pairs. This superiority demonstrates that DanQing contains more up-to-date information, effectively enabling models to internalize contemporary knowledge and generalize to recent real-world concepts.

#### 6 Conclusion

In this paper, we present DanQing, a large-scale Chinese image–text dataset comprising approximately 100M pairs that mitigates the scarcity of high-quality cross-modal resources for Chinese vision-language pretraining. We develop a rigorous curation pipeline to ensure quality, filtering raw web data and retaining about 10% as high-quality pairs. Through continued pretraining of SigLIP2 models, DanQingconsistently outperforms existing Chinese datasets across multiple downstream benchmarks. Comprehensive analysis further reveals that DanQingexhibits stronger scaling capability, improved understanding of novel concepts, higher textual quality, and more balanced visual semantics distribution. To facilitate and accelerate future research, we will open-source the DanQing dataset, providing a robust foundation for large-scale Chinese vision–language pretraining.

#### References

- [1] Baidu AI Studio Community. PaddleHub Pornographic Content Detection Model Tutorial, Jan

2021. URL https://aistudio.baidu.com/projectdetail/1444248. Online tutorial demonstrating the use of PaddleHub’s porn_detection_lstm model for text-based content moderation. Last updated on 2021-01-13.

- [2] Baidu AI Studio Community. PaddleHub Pornographic Content Detection Model Tutorial, Jan 2021. URL https://www.paddlepaddle.org.cn/hubdetail?name=porn_detection_ cnn&en_category=TextCensorship. Online tutorial demonstrating the use of PaddleHub’s porn_detection_lstm model for text-based content moderation. Last updated on 2021-01-13.
- [3] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In ECCV, pp. 446–461. Springer, 2014.
- [4] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.
- [5] BYVoid. OpenCC: Open Chinese Convert. https://github.com/BYVoid/OpenCC, 2024.
- [6] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pp. 3558– 3568, 2021.
- [7] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 2017.
- [8] Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, et al. Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062, 2025.
- [9] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild, 2013. URL https://arxiv.org/abs/1311.3618.
- [10] Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. Revisiting pre-trained models for chinese natural language processing. In EMNLP, pp. 657–668, 2020.
- [11] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431, 2021.
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [14] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV, 88(2):303–338, 2010.
- [15] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023.
- [16] Li Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPRW, 2004.
- [17] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.

- [18] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Neurips, 36:27092–27112, 2023.
- [19] Maarten Grootendorst. Bertopic: Neural topic modeling with a class-based tf-idf procedure,

2022. URL https://arxiv.org/abs/2203.05794.

- [20] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Neurips, 35:26418–26431, 2022.
- [21] Tiancheng Gu, Kaicheng Yang, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, and Jiankang Deng. Rwkv-clip: A robust vision-language representation learner. In EMNLP, pp. 4799–4812, 2024.
- [22] Tiancheng Gu, Kaicheng Yang, Chaoyi Zhang, Yin Xie, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, and Jiankang Deng. Realsyn: An effective and scalable multimodal interleaved document transformation paradigm. In ACM MM, pp. 3487–3496, 2025.
- [23] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin andFeature Pyramid Networks for Object Detection Cui. Zero-shot detection via vision and language knowledge distillation. In ICLR, 2022.
- [24] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019.
- [25] Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, and Tomas Mikolov. Fasttext.zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.
- [26] Elias Kempf, Simon Schrodi, Max Argus, and Thomas Brox. When and how does clip enable domain and compositional generalization? In ICML, 2025.
- [27] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Neurips, 2020.
- [28] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In ICCVW, pp. 554–561, 2013.
- [29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [30] Weiyu Lan, Xirong Li, and Jianfeng Dong. Fluency-guided cross-lingual image captioning. In ACM MM, pp. 1549–1557, 2017.
- [31] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.
- [32] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Languagedriven semantic segmentation. In ICLR, 2022.
- [33] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022.
- [34] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In CVPR, 2022.
- [35] Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Coco-cn for cross-lingual image tagging, captioning, and retrieval. TMM, 21(9):2347–2360, 2019.

- [36] Junyang Lin, Rui Men, An Yang, Chang Zhou, Yichang Zhang, Peng Wang, Jingren Zhou, Jie Tang, and Hongxia Yang. M6: Multi-modality-to-multi-modality multitask mega-transformer for unified pretraining. In ACM SIGKDD, pp. 3251–3261, 2021.
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/.
- [38] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, pp. 216–233. Springer, 2024.
- [39] Yulong Liu, Guibo Zhu, Bin Zhu, Qi Song, Guojing Ge, Haoran Chen, GuanHui Qiao, Ru Peng, Lingxiang Wu, and Jinqiao Wang. Taisu: A 166m large-scale high-quality dataset for chinese vision-language pre-training. NIPS, 35:16705–16717, 2022.
- [40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [41] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction, 2020. URL https://arxiv.org/abs/1802.03426.
- [42] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021.
- [43] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets language-image pre-training. In ECCV, pp. 529–544. Springer, 2022.
- [44] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, 2008.
- [45] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, Su Wang, and Jason Baldridge. Docci: Descriptions of connected and contrasting images, 2024. URL https://arxiv.org/abs/ 2404.19753.
- [46] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012.
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.
- [49] Yongming Rao, Wenliang Zhao, Guangyi Light, Jiwen Zhou, Jiwen Lu, et al. Denseclip: Language-guided dense prediction with context-aware prompting. In CVPR, 2022.
- [50] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. Neurips, 35: 25278–25294, 2022.

- [52] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.
- [53] Freepik Company S.L. Eva-based fast nsfw image classifier, 2025. URL https://huggingface. co/Freepik/nsfw_image_detector.
- [54] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In SIGIR, pp. 2443–2449, 2021.
- [55] Robert Endre Tarjan. Efficiency of a good but not linear set union algorithm. JACM, 22(2): 215–225, 1975.
- [56] OpenCV team. Opencv: Open source computer vision library. https://github.com/opencv/ opencv, 2024.
- [57] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.
- [58] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [59] Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana Romero-Soriano. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. In CVPR, 2024.
- [60] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022. URL https://arxiv.org/abs/2202. 03052.
- [61] Yu Wu, Yana Wei, Haozhe Wang, Yongfei Liu, Sibei Yang, and Xuming He. Grounded image text matching with mismatched relation reasoning. In ICCV, pp. 2976–2987, 2023.
- [62] Chunyu Xie, Heng Cai, Jincheng Li, Fanjing Kong, Xiaoyu Wu, Jianfei Song, Henrique Morimitsu, Lin Yao, Dexin Wang, Xiangzheng Zhang, et al. Ccmb: A large-scale chinese cross-modal benchmark. In ACM MM, pp. 4219–4227, 2023.
- [63] Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Ji Ao, Dawei Leng, and Yuhui Yin. Fg-clip 2: A bilingual fine-grained vision-language alignment model. arXiv preprint arXiv:2510.10921, 2025.
- [64] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, ShangWen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. arXiv preprint arXiv:2309.16671, 2023.
- [65] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In CVPR, 2022.
- [66] An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese clip: Contrastive vision-language pretraining in chinese. arXiv preprint arXiv:2211.01335, 2022.
- [67] Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. Alip: Adaptive language-image pre-training with synthetic caption. In ICCV, pp. 2922–2931, 2023.

- [68] Kaicheng Yang, Tiancheng Gu, Xiang An, Haiqiang Jiang, Xiangzi Dai, Ziyong Feng, Weidong Cai, and Jiankang Deng. Clip-cid: Efficient clip distillation via cluster-instance discrimination. In AAAI, volume 39, pp. 21974–21982, 2025.
- [69] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. TMLR, 2022.
- [70] Sha Yuan, Shuai Zhao, Jiahong Leng, Zhao Xue, Hanyu Zhao, Peiyu Liu, Zheng Gong, Wayne Xin Zhao, Junyi Li, and Jie Tang. Wudaomm: A large-scale multi-modal dataset for pre-training models. arXiv preprint arXiv:2203.11480, 2022.
- [71] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pp. 11975–11986, 2023.
- [72] Xunlin Zhan, Yangxin Wu, Xiao Dong, Yunchao Wei, Minlong Lu, Yichi Zhang, Hang Xu, and Xiaodan Liang. Product1m: Towards weakly supervised instance-level product retrieval via cross-modal pretraining. In ICCV, pp. 11782–11791, 2021.
- [73] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In ECCV, pp. 310–325. Springer, 2024.
- [74] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, et al. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2401.11944, 2024.
- [75] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024.
- [76] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based language-image pretraining. In CVPR, 2022.

#### Appendix Overview

The appendix includes the following sections:

- • Appendix A: Provides detailed statistics of data filtration.
- • Appendix B: Visualizes image-text pair examples from the DanQing dataset.
- • Appendix C: Offers additional analytical insights and exploratory studies, including source domain distributions (Sec. C.1), extended visualizations of topic modeling (Sec. C.2), and word cloud analyses (Sec. C.3).

#### A Statistics of Data Filtration

Steps Left Data Num Total Filter % Stage Filter % Left %

Collected image URL and text pairs 1,047,085,609 – – 100.00%

Language Control content Safety 726,334,674 30.63% 69.37% 69.37% Source Reliability Text Constraints 706,069,936 32.57% 2.79% 67.43% Download Success 475,104,485 54.63% 67.29% 45.37%

Data Source Selection

Linguistic Structure (CN-Detect) 467,455,303 55.36% 1.61% 44.64% Linguistic Structure (Font conversion) 467,455,303 55.36% 0.00% 44.64% Text Quality (Stop words) 442,680,171 57.72% 5.30% 42.28% Text Quality (Nouns) 431,878,774 58.75% 2.44% 41.25% Text Quality ([UNK]) 431,619,647 58.78% 0.06% 41.22% Information Density (Entropy) 400,068,251 61.79% 7.31% 38.21% Information Density (Emoji&special chars) 400,068,251 61.79% 0.00% 38.21% Content Safety 397,187,759 62.07% 0.72% 37.93%

Text Refinement

Visual Fidelity 352,663,012 66.32% 11.21% 33.68% Visual Fidelity (Low-textual images) 352,204,550 66.36% 0.13% 33.64% Visual Fidelity (Blurry images) 333,255,945 68.17% 5.38% 31.83% Information Density 316,359,868 69.79% 5.07% 30.21% Perceptual and Semantic Redundancy 186,019,602 82.23% 41.20% 17.77% Content Safety 178,601,215 82.94% 3.25% 17.06%

Visual Diversification

Cross-Modal Alignment Assessment 154,293,590 85.26% 13.61% 14.74% Cross-Batch Redundancy Removal 99,892,381 90.46% 35.26% 9.54%

Cross-Modal Cross-Batch Filtering

Final image-text pairs 99,892,381 90.46% – 9.54%

Table 6: Specific statistics information of the DanQing dataset construction pipeline.

We further illustrate the statistical breakdown of our data construction pipeline in Tab. 6. Starting with an initial collection of approximately 1B raw image-text URLs, we apply a multi-stage filtering process, comprising data source selection, text refinement, visual diversification, and cross-modal cross-batch filtering. This pipeline ultimately yields nearly 100M high-quality image-text pairs (storage space occupies approximately 12TB), effectively filtering out 90% of the original noise to ensure data quality.

#### B Examples in DanQing Dataset

Fig. 9 presents representative examples from the DanQing dataset, comprising images and their corresponding Chinese textual descriptions. Specifically, these image-text pairs encompass a wide range of domains, such as natural scenery, historical literature, and automotive technology, demonstrating the thematic diversity of our dataset. This breadth makes DanQing particularly well-suited for general-purpose Vision-Language Pre-training.

##### Rank Source Domain Total Count % Categories

- 1 alicdn.com 3,555,967 3.56% E-commerce, Cloud Computing
- 2 baidu.com 2,649,928 2.65% Search Engine, Tech
- 3 wp.com 1,582,010 1.58% Blog Hosting (WordPress), CMS
- 4 aliyuncs.com 961,433 0.96% Cloud Computing, Object Storage
- 5 chem17.com 781,994 0.78% Chemical Instruments, E-commerce
- 6 bing.net 741,185 0.74% Search Engine (Bing), Content Delivery
- 7 udn.com.tw 542,113 0.54% News Media, General Info
- 8 faiusr.com 490,843 0.49% Website Builder, Marketing
- 9 sinaimg.cn 404,076 0.40% Social Media (Weibo), Sina Portal
- 10 wixstatic.com 393,168 0.39% Website Builder (Wix), Image Hosting
- 11 126.net 372,122 0.37% Internet Services (NetEase), Email
- 12 360buyimg.com 336,282 0.34% E-commerce (JD.com), Logistics
- 13 vjshi.com 333,778 0.33% Video Assets, Copyright Trading
- 14 bing.com 322,193 0.32% Search Engine, Microsoft, International
- 15 staticflickr.com 319,134 0.32% Photo Community, Social Media
- 16 xiniu.com 286,491 0.29% Enterprise Services, Marketing
- 17 sohu.com 285,513 0.29% General Portal, Media, Video
- 18 myqcloud.com 262,140 0.26% Cloud Computing (Tencent)
- 19 smzdm.com 258,988 0.26% Consumer Guide, E-commerce
- 20 made-in-china.com 256,439 0.26% Cross-border Trade, B2B, Export
- 21 cloudfront.net 249,402 0.25% Content Delivery, Cloud Computing
- 22 sogoucdn.com 231,903 0.23% Search Engine (Sogou)
- 23 toutiaoimg.com 229,255 0.23% News (Toutiao), ByteDance
- 24 hbzhan.com 227,981 0.23% Environmental Industry
- 25 byteimg.com 225,890 0.23% ByteDance, Short Video, News
- 26 qpic.cn 223,192 0.22% Social Media (QQ/WeChat), Tencent
- 27 myapp.com 216,009 0.22% App Store, Mobile Internet
- 28 zdmimg.com 210,646 0.21% Consumer Community
- 29 tom.com 210,437 0.21% General Portal, Internet Services, Media
- 30 itc.cn 204,060 0.20% Sohu Media, Content Delivery (CDN)
- 31 588ku.com 194,423 0.19% Design Materials (Qinku/Wotu)
- 32 qunarzz.com 193,114 0.19% Travel & Tourism (Qunar)
- 33 bdxiguaimg.com 185,956 0.19% Video (Xigua Video), ByteDance
- 34 suning.cn 174,229 0.17% E-commerce (Suning), Retail
- 35 thefastimg.com 168,851 0.17% Image Hosting, General CDN
- 36 gtimg.com 168,577 0.17% Tencent, Games, Social
- 37 iqiyipic.com 167,389 0.17% Video (iQIYI), Entertainment
- 38 myxypt.com 164,791 0.17% Pharmaceutical B2B, Industry Platform
- 39 book.com.tw 161,625 0.16% Book Retail, E-commerce, Culture
- 40 tripcdn.com 155,944 0.16% Travel (Ctrip/Trip.com), International Table 7: Statistics and categories overview of the top 40 image source domains.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

金丝楠乌木官皮箱 代表着“一念 收敛，则万善来同”的艺术。The golden phoebe ebony official trunk embodies the art of "restraining one thought to gather all virtues."

冬奥会使用了绿色环保的建筑材 料，有效降低了建筑过程中的碳 排放。 The Winter Olympics used

明代宋应星所著的《天工开物》 上卷中详细记载了制糖的方法。

在咖啡馆里啜饮一杯怀旧小确幸 的咖啡，享受短暂的惬意时光。

Song Yingxing's Tiangong Kaiwu (Volume 1) from the Ming Dynasty details sugar-making methods.

Sipping a cup of nostalgically comforting coffee, savoring those fleeting moments of simple bliss.

green building materials, effectively reducing carbon emissions during construction.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

巴斯(Bath)是一座风光绮丽的中 世纪小镇，散发着浓郁的历史文 化气息。Bath is a picturesque

董卓是什么人物? 董卓性格特点 是怎么样的?

合众汽车 哪吒S猎装 2024款 纯 电510 MAX版。Hozon Auto Neta S Shooting Brake 2024 Model, Pure Electric 510 MAX Edition.

湿疹食物禁忌 鸡蛋是常见的致敏 原，其中蛋白尤其多致敏原。

Food Restrictions for Eczema: Eggs are a common allergen, and egg whites are especially allergenic.

Who was Dong Zhuo? What were Dong Zhuo's personality traits?

medieval town that exudes a rich historical and cultural atmosphere.

Figure 9: Visualization of image-text pairs in DanQing dataset.

#### C Exploring DanQing

##### C.1 Source Domain Distribution

To further investigate the origins of our data, we identified and ranked the top 40 primary web sources, as detailed in Tab. 7. The results indicate that the majority of image-text pairs originate from widely-used Chinese platforms and applications, such as Alibaba, Baidu, ByteDance, and so on. Furthermore, these sources span diverse categories, including E-commerce, news media, and search engines, demonstrating the heterogeneous nature of our data. This variety validates that DanQing is sourced from a broad spectrum of real-world scenarios, capturing the richness of daily-life multimodal content.

##### C.2 Topic Modeling

We further illustrate the topic distribution of DanQing in Fig. 10, following the same analytical methodology described in Sec. 3.3. The visualized examples encompass both previously mentioned categories and novel domains, such as tourism, design, education, and agriculture, which are prevalent throughout the dataset. This thematic breadth, exemplified by diverse visual content, underscores how DanQing aligns closely with real-world scenarios and everyday life.

##### C.3 Word Cloud

In Fig. 11, we visualize the distribution of Chinese words (comprising one or more tokens) within the DanQing dataset. Specifically, we utilize the jieba 2 words segmentation module to tokenize the text and generate the word cloud. The visualization highlights that the most frequent terms include ”2024”, ”中国” (China), ”游戏” (Game), “美食” (Food), “活动” (Activity), and so on. This distribution demonstrates the inclusion of the newest semantic concepts and diverse daily topics, which are essential for robust, general purpose Vision-Language Pre-training.

- 2 https://github.com/fxsjy/jieba

时尚与穿搭 (6.11%)（Fashion and Outfits）

科技与电子产品 (4.32%)（Technology and Electronic Products） Key words: 科技, 蓝牙, 耳机, 电动汽车, 快充, 智能驾驶 (Technology, Bluetooth, Headphones, Electric Vehicle, Fast Charging, Inteligent Driving)

Key words: 夏日穿搭, 卡其色, 流行, 货号,新款, 球鞋, 托特包 (Summer styling, Khaki, Popular, Product code, New arrivals, Sneakers, Tote bag)

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

婚庆 (2.06%)（Tourism） Key words: 婚庆, 婚纱, 影楼, 刺绣, 婚庆公司 (Wedding, Wedding Dress, Photo Studio, Embroidery, Wedding Company)

设计 (2.03%)（Design） Key words: 包装设计, 品牌设计, 产品设计, 工业设计, 创意设计 (Packaging Design, Brand Design, Product Design, Industrial Design, Creative Design)

[Figure 79]

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

教育 (1.73%) (Education) 农业 (1.45%) Agriculture）

Key words: 学习, 学校, 学生, 教师, 课程, 知识, 考试, 成长 (Learning, School, Student, Teacher, Course, Knowledge, Exam, Growth)

Key words: 农业, 种植, 新能源, 种植技术, 农科云, 基地 (Planting, New Energy, Planting Technology, Agricultural Cloud, Base)

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Figure 10: External topic examples visualization of DanQing dataset generated via the BERTopic [19].

[Figure 104]

Figure 11: Word cloud visualization of 10M subset texts from DanQing.

