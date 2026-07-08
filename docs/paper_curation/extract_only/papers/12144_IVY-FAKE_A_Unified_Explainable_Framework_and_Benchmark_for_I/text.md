## arXiv:2506.00979v6[cs.CV]5May2026

### Ivy-Fake: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection

Changjiang Jiang*, Wenhui Dong* , Zhonghao Zhang, Fengchang Yu, Wei Peng, Xinbin Yuan, Yifei Bi, Ming Zhao, Zian Zhou, Chenyang Si, Caifeng Shan*

Nanjing University+

∗Equal Contributions, Corresponding author: wenhui.dong@smail.nju.edu.cn,cfshan@nju.edu.cn,

+Complete affiliation can be see in Sec. A

The rapid development of Artificial Intelligence Generated Content (AIGC) techniques has enabled the creation of high-quality synthetic content, but it also raises significant security concerns. Current detection methods face two major limitations: (1) the lack of multidimensional explainable datasets for generated images and videos. Existing open-source datasets (e.g., WildFake, GenVideo) rely on oversimplified binary annotations, which restrict the explainability and trustworthiness of trained detectors. (2) Prior MLLM-based forgery detectors (e.g., FakeVLM) exhibit insufficiently fine-grained interpretability in their step-by-step reasoning, which hinders reliable localization and explanation. To address these challenges, we introduce Ivy-Fake, the first large-scale multimodal benchmark for fake image and video detection. It consists of over 106K richly annotated training samples (images and videos) and 5,000 manually verified evaluation examples, sourced from multiple generative models and real-world datasets through a carefully designed pipeline to ensure both diversity and quality. Furthermore, we propose Ivy-xDetector, a multimodel large language model (MLLM) based on reinforcement fine-tuning (RFT), capable of producing explainable reasoning chains and achieving robust performance across multiple fake image and video detection benchmarks.

[Figure 1]

Project Homepage: Link Model: https://huggingface.co/AI-Safeguard/Ivy-Fake Dataset: https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake

##### 1 Introduction

The rapid development of diffusion-based models has triggered an exponential growth in Artificial Intelligence Generated Content (AIGC), like Sora (Brooks et al., 2024), DALL-E (Betker et al., 2023), Imagen (Saharia et al., 2022), and Stable Diffusion (Rombach et al., 2022), which have redefined state-of-the-art performance in text-to-image synthesis. However, these advances also raise significant security concerns, including the misuse of DeepFake (Qureshi et al., 2024), document tampering (Qureshi et al., 2024; Qu et al., 2023, 2024, 2025), and dataset poisoning (Steinhardt et al., 2017; Ying et al., 2026). The increasing realism of synthetic content blurs the boundary between genuine and fabricated media, posing critical challenges for misinformation control, content provenance verification, and the preservation of public trust.

However, most existing approaches primarily focus on binary authenticity classification (Chen et al., 2024b; Hong et al., 2025; Bai et al., 2024; Yan et al., 2025), which limits the human interpretability of model predictions (Qu et al., 2026b,a; Ye et al., 2025; Jiang et al., 2026). Existing benchmarks also remain inadequate for evaluating explainable AIGC detection. Datasets such

[Figure 2]

###### Ivy-Fake

###### [AIGC Detection; Artifact Interpretation]

[Figure 3]

[Figure 4]

ChatGPT: From a visual inspection of these stills: There’s a consistent lighting scheme and natural motion blur. Facial features and body movements appear coherent and without the telltale artifacts ... Based on the visual analysis: the video appears to be real — there are no clear signs of manipulation, deepfake artifacts, or CGI.

[Figure 5]

[Figure 6]

Think: 1. Temporal Analysis: The video ... The transitions are abrupt, and there's a lack of natural temporal flow ... 2.Spatial Analysis:Illegible Letters: In the opening scene (0:00-0:01), the sign on the subway wall (reading something like \"Europan V\") displays characteristics of AI-generated text. Letters appear somewhat misshapen, with inconsistent stroke continuity and clarity, particularly in the smaller text below the main word...Conclusion: The video exhibits multiple synthetic fingerprints. ...

[Figure 7]

Temporal Features

###### Spatial Features

[Figure 8]

[Figure 9]

Multi-frame anomaly detection

Single-frame dynamic anomaly detection

Luminance Discrepancy Awkward Facial Expression Duplicated Component Non-Spatial Relationship

[Figure 10]

Detection Features Impractical Luminosity

Omitted Components Spatial Relationships Chromatic Irregularity

[Figure 11]

Distorted Components Abnormal Texture

Localized Blur Illegible Letters

- Figure 1 Overview of the Ivy-Fake framework: By conducting in-depth analysis of temporal and spatial artifacts, the framework enables explainable detection of AI-generated content.

as AIGCDetectionBenchmark (Zhong et al., 2023) and GenVideo (Chen et al., 2024b) provide only binary labels, while more recent resources like LOKI (Ye et al., 2025) attempt to incorporate fine-grained anomaly annotations across modalities but are still constrained in scale and diversity. On the other hand, current methods such as AIDE (Yan et al., 2025) and Demamba (Chen et al.,

- 2024b) are limited to a single modality—either image or video—without exploring unified detection across both domains. Similarly, existing datasets exhibit the same fragmentation: FakeBench (Liu et al., 2025b) emphasizes explainable fake image detection but omits video content, whereas FakeClue (Wen et al., 2025) provides extensive image-level annotations yet lacks integrated video data. This fragmentation leads to substantial gaps in benchmarking and hinders unified advancement in explainable multimodal AIGC detection.

To address these challenges, we propose Ivy-Fake, a comprehensive dataset designed to evaluate explainable multimodal AIGC detection. Ivy-Fake offers: 1) Diverse Multimodal Data, a large-scale dataset comprising 61,107 annotated images and 45,272 annotated videos for training, along with 5,000 samples for evaluation. 2) Explainable Annotations, rich annotations that extend beyond binary labels to include detailed reasoning, enabling nuanced evaluation of models’ interpretability and explanatory capabilities.

In addition to providing Ivy-Fake, we introduce Ivy-xDetector, a model for detecting AI-generated images and videos and explaining the associated artifacts. Unlike other MLLMs, Ivy-xDetector excels at spotting generative artifacts, spatial in images and both spatial and temporal in videos. By integrating multiple spatial and temporal feature extractors, it detects image-level artifacts and video-level temporal inconsistencies with superior accuracy. Through the incorporation of explainable annotations and carefully curated domain-diverse data, our approach achieves state-of-the-art performance using only a fraction of the data required by existing detectors. While prior image and video classifiers (Chen et al., 2024b; Yan et al., 2025; Zhong et al., 2023) often rely on millions (M-level) of samples, our method attains superior performance across both image and video AIGC detection tasks with fewer than 200K samples.

Our main contributions are summarized as follows:

- Table 1 Task examples of Ivy-Fake.

|Features<br><br>|Sub Dimension|Example<br><br>|
|---|---|---|
|Spatial<br><br>|Impractical Luminosity|Which visual clue reveals an impossible spatial or lighting configuration in the scene? (A) Light direction opposite source. (B) Face without lighting. (C) Dark background, bright face. (D) Bright background, dark face.<br><br>|
| |Localized Blur<br><br>|Which clue shows local blur or focus inconsistency? (A) Clear face but blurred hand. (B) Sharp background, fuzzy hair. (C) Glasses edges blurred, eyes clear. (D) Hand blurred while holding clear object.<br><br>|
| |Illegible Letters|Which part of the document text is tampered or altered? (A) Text blurred or half missing. (B) Letters inconsistent in size. (C) Words overlap or melt together. (D) Signs contain random symbols.<br><br>|
| |Omitted Components<br><br>|Which part of the person looks incomplete or missing? (A) Missing one finger. (B) Two fingers merged. (C) Missing eyebrows. (D) Three pairs of glasses.<br><br>|
| |Spatial Relationships|Where does the spatial relation look unnatural or physically impossible? (A) Arm passes through body. (B) Legs overlap unnaturally. (C) Head detached from neck. (D) Hand holds object from wrong side.<br><br>|
| |Distorted Components|Which part appears deformed or stretched in shape? (A) Finger bent at odd angle. (B) Eye stretched sideways. (C) Mouth warped or uneven. (D) Hand larger than face.<br><br>|
| |Chromatic Irregularity|Which clue shows unnatural or inconsistent colors in the image? (A) Face partly blue under warm light. (B) Green reflection on red shirt. (C) Sky color bleeds into hair. (D) Background color unevenly saturated.<br><br>|
| |Abnormal Texture<br><br>|What visual area shows abnormal or inconsistent texture? (A) Skin looks like plastic. (B) Hair pattern repeats unnaturally. (C) Cloth surface too smooth or glossy. (D) Wall texture mixed with skin.<br><br>|
|Temporal<br><br>|Luminance Discrepancy<br><br>|Which moment shows sudden or inconsistent brightness between frames? (A) Light flickers without reason. (B) Face turns dark while background stays bright. (C) Shadow jumps between frames. (D) Object brightness changes abruptly.<br><br>|
| |Awkward Facial Expression<br><br>|When does the facial movement look unnatural or discontinuous across frames? (A) Smile appears and disappears suddenly. (B) Eyes blink in different directions. (C) Mouth freezes mid-motion. (D) Expression changes without emotion cue.<br><br>|
| |Duplicated Component<br><br>|Where does an object or body part appear twice across frames? (A) Two faces shown in one frame. (B) Hand shadow moves separately. (C) Extra arm appears then fades. (D) Same object duplicated in next frame.<br><br>|
| |Non-Spatial Relationships|When does the action sequence look illogical or out of order? (A) Person closes door before touching it. (B) Cup moves without contact. (C) Smile appears after laughter. (D) Object vanishes then reappears suddenly.<br><br>|

- • We introduce Ivy-Fake, the first large-scale dataset for explainable AIGC detection across both images and videos, comprising 106,379 training samples and 5,000 manually verified test instances. Each entry is enriched with fine-grained visual annotations and textual reasoning to support transparent multimodal evaluation and future research.
- • We propose Ivy-xDetector, a reinforcement learning–based model by Reinforcement Learning with Verifiable Rewards (RLVR) built upon Group Relative Policy Optimization (GRPO), which achieves superior performance on multiple image and video detection benchmarks using fewer than 200K samples, significantly outperforming existing methods that rely on millions of data points.
- • We conduct extensive experiments to validate the effectiveness of both the dataset and the model, demonstrating their substantial impact on improving multimodal large models’ capability in synthetic content detection.

Datasets

Token Length Distribution Across Datasets

Abnormal Texture

FakeClue LOKI MMTD-Set IVY-Fake

Avg: 119.6

|Relationships<br><br>Components<br><br>Awkward Facial Expression<br><br>|Distorted Components<br><br>Omitted<br><br>0%<br><br>10%<br><br>20%<br><br>30%<br><br>40%<br><br>50%<br><br>60%<br><br>70%<br><br>80%<br><br>90%<br><br>|
|---|---|
|Impractical Luminosity<br><br>Discrepancy<br><br>|Spatial<br><br>Chromatic Irregularity<br><br>|

Min: 22

Max: 558

FakeClue

Duplicated

Components

Avg: 226.5

Min: 66

Max: 577

LOKI

Dataset

Non-Spatial Rel

Localized Blur

Avg: 241.0

Min: 10

Max: 515

MMTD

Avg: 451.7

Luminance

Relationships

Min: 63

Max: 800

Ivy-Fake

0 100 200 300 400 500 600 700 800 Token Length

Illegible Letters

(a) Token Length Distribution

(b) Multi-Dimensional Coverage

- Figure 2 Token Length Distributions and Multi-Dimensional Coverage Across Datasets; Left: Distribution of token lengths across datasets; Right: Coverage of multiple dimensions in explainability datasets, extracted using Qwen3-32B (Yang et al., 2025). The Prompt can be seen in appendix.

##### 2 Related Work

###### 2.1 Methods for Synthetic Content Detection

Due to growing concerns about the misuse of synthetic data (Deng et al., 2024; Qu et al., 2025; Li et al., 2025), research on AI-generated content (AIGC) detection has expanded rapidly in recent years (Wang et al., 2026; Park et al., 2025; Shuai et al., 2026; Tan et al., 2026a,b; Yu et al., 2026; Huang et al., 2025a; Zhou et al., 2025b). Most existing models for AI-generated images and videos formulate the task as binary classification, simply predicting whether the content is "real" or "fake." Representative examples include CNN-based AIGVDet (Bai et al., 2024), CNNSpot (Wang et al.,

- 2020), DIRE (Wang et al., 2023), Mirror (Liu et al., 2026) and AIDE (Yan et al., 2025). Meanwhile, several works have explored the application of multimodal large language models (MLLMs) to AIGC detection, including Synartifact (Cao et al., 2024) and Bi-LORA (Keita et al., 2025). However, these approaches largely overlook the importance of interpretability in AIGC detection.

Some efforts attempt to introduce interpretability by leveraging spatial annotations (Qu et al., 2023; Zhu et al., 2025a; Qu et al., 2024, 2026a), frequency-domain artifact analysis (Zhang et al., 2023) or or hybrid attention (Xin et al., 2026b; Kong et al., 2025). Nevertheless, the resulting explanations are often difficult for humans to comprehend, as they lack clarity in natural language. This limitation is particularly evident in the video domain, where AI-generated content frequently exhibits obvious flaws, e.g., incoherent frame transitions and object inconsistency, that are easily noticed and reasoned about by humans (Deng et al., 2024). FakeClue (Wen et al., 2025) introduces the use of visionlanguage models (VLMs) to provide interpretability for image-level detection, but it does not offer a unified framework that integrates both images and videos.

###### 2.2 Datasets for Synthetic Content Detection.

Early datasets for synthetic content detection (Zhang et al., 2025; Jiang et al., 2025b; Ma et al., 2025; Du et al., 2025; Zhu et al., 2025b), such as CNNSpot (Wang et al., 2020), primarily collected fake images generated by GAN-based models (Goodfellow et al., 2014; Zhu et al., 2017; Brock et al., 2018). However, with the advent of more advanced generative architectures like diffusion models (Hertz et al., 2022; Nichol et al., 2021; Ying et al., 2026) and their variants, the authenticity of generated content has significantly increased, making it more challenging for detection models to discern. This has spurred the development of newer datasets, including ArtiFact (Cao et al., 2024), GenImage (Zhu et al., 2023b), GenImagePP (Zhou et al., 2025a) and WildFake (Hong et al., 2025). GenImage (Zhu et al., 2023b) comprises images from the 1000 ImageNet (Russakovsky et al., 2015) categories, generated by eight state-of-the-art generators such as Stable Diffusion (Rombach et al.,

- 2022) and Midjourney. Nevertheless, these datasets predominantly focus on image-based content. More recently, datasets emphasizing interpretability have also been introduced (Zhao et al., 2026a). FakeClue (Wen et al., 2025) contains a large amount of image data with explainability annotations but lacks video data. LOKI (Ye et al., 2025) offers data across 26 different categories and includes 18,000 distinct questions; however, its volume of multimodal data is relatively small and primarily suited for evaluation rather than comprehensive model training (Ji et al., 2025). Therefore, a critical gap exists for a unified benchmark encompassing both image and video modalities to rigorously evaluate the performance of contemporary AIGC detectors.

##### 3 Methodology

###### 3.1 Data Collection

The Ivy-Fake dataset is curated from GenVideo (Chen et al., 2024b), LOKI (Ye et al., 2025), FakeClue (Wen et al., 2025), WildFake (Hong et al., 2025), Kinetics-400 (Kay et al., 2017), and MMFakeBench (Liu et al., 2025b). Since Ivy-Fake is substantially larger than other datasets, we primarily detail our data processing procedures here.

Image/Video Corruption Filtering. The integrity of images and videos is critical for MLLM training. We observed that the official GenVideo (Chen et al., 2024b) repository contained a large number of unreadable videos. To ensure data quality, we employed the official Qwen2.5-VL (Bai et al., 2025) I/O library to filter out corrupted files, discarding images and videos that failed to load.

Image/Video Resolution. To maintain consistency, we only retained images whose resolutions fell within [3,136, 12,845,056] pixels, and videos whose per-frame resolutions were within [100,352, 602,112] pixels, with the additional constraint that total pixels did not exceed 19,267,968. Samples outside these ranges were discarded.

Near-Duplicate Image Filtering. Selecting challenging and diverse samples is crucial for model robustness. We extracted visual features from images and video frames using CLIP (Radford et al.,

- 2021) and computed intra-class similarity scores. Images with excessively high similarity were removed, while videos were left unprocessed. To comprehensively activate and enhance the model’s reasoning capabilities across the task, we propose a high-quality data construction pipeline that integrates explainable detection of AI-generated content.

Public Dataset Sources The Ivy-Fake training set comprises approximately 106K image and video samples collected during the cold-start phase from diverse sources to ensure broad coverage of contemporary generative models. A significant portion was sourced from established public datasets, including GenVideo (Chen et al., 2024b), LOKI (Ye et al., 2025), FakeClue (Wen et al., 2025), WildFake (Hong et al., 2025), Kinetics-400 (Kay et al., 2017), MMFakeBench (Liu et al., 2025b), GenImage (Zhu et al., 2023b), DiffusionForensics (Wang et al., 2023). The cold-start phase aimed to enhance the granularity of interpretable content. To further improve the detector’s accuracy, an additional 87,500 image/video samples were stratified and sampled from multiple datasets for RLVR training, including MSRVTT (Xu et al., 2016) (9,000 videos), GenImage (Zhu et al., 2023b) (32,000 images per category), CnnSpot (Wang et al., 2020) (19,000 samples), DigiFakeAV (Liu et al., 2025a) (2,000 samples), GenVideo (Chen et al., 2024b) (Fake subset) (9,000 samples), DiffusionForensics (Wang et al., 2023) (4,500 samples), WildFake (Hong et al., 2025) (12,000 samples), and synthetic data generated by Qwen-Image (Wu et al., 2025) (634 samples). In total, the dataset used for training comprises 106K samples.

Generated Sources To further enrich the dataset and better capture synthetic images prevalent in real-world scenarios, additional samples were generated using a diverse set of state-of-the-art image generation models. These models span multiple architectures and training paradigms, ensuring broad coverage of visual styles and generative patterns for robust evaluation. Specifically, extra sources includes samples generated by FLUX.1 (Labs et al., 2025), Lora Flux (Labs et al., 2025) and Qwen-Image (Wu et al., 2025).

###### 3.2 Data Annotation

As illustrated in Table 1, “file_type” indicates the modality of the input—either “image” or “video”, and label represents the ground-truth label, assigned as either “real” or “fake”. The distilled explanations were further categorized along two major dimensions to facilitate structured analysis: Spatial Features, which comprises eight sub-dimensions and captures artifacts and inconsistencies observable within individual frames or static images. Temporal Features, which includes four sub-dimensions and describes anomalies associated with motion, temporal coherence, and crossframe consistency. Since still images inherently lack temporal attributes, this category is exclusively applicable to video annotations. These categories were informed by established taxonomies of visual artifacts in generative content, as detailed in prior research (Deng et al., 2024).

###### 3.3 Comparison with Existing Datasets

A comparative overview of Ivy-Fake and several existing AIGC detection datasets is provided in

- Table 2. Notably, Ivy-Fake offers unique advantages by integrating explainable annotations across both image and video modalities, addressing a significant gap in current resources.

- Table 2 Comparison on the different datasets used in binary classification and interpretability tasks. Token lengths are computed using the GPT-4o tokenizer from the tiktoken library.

Dataset Avg Token Generator Modality Dataset Lengths fake real

FakeBench (Liu et al., 2025b) - 6 Image 3K 3K VANE-Bench (Bharadwaj et al., 2024) 101 5 Image* 2K 1K LOKI (Ye et al., 2025) 226.5 ∼16 Image+Video 3K 0 FakeClue (Wen et al., 2025) 119.6 26 Image 68K 36K Ivy-Fake 451.7 >30 Image+Video 53K 78K

We identified a long-tail distribution in the distilled dataset, with a small portion of samples exhibiting token lengths concentrated at both extremes (Xin et al., 2026b). To mitigate the influence of these outliers, we retained only the central 90% of the distribution, filtering out the top and bottom 5% of samples, thereby keeping data with token lengths between 60 and 800. As shown in Figure 2

- panel (a), Ivy-Fake exhibits the longest token length distribution, with an average token length of 451.7. Additionally, to conduct a fine-grained analysis of the semantic dimensions embedded in the explanatory language across different interpretability datasets, we employ Qwen3-32B (Yang et al.,

2025) for dimension extraction. To enhance the accuracy of the extraction, we embed supplementary contextual information from each dataset into the model input. For example, in LOKI (Ye et al., 2025), the bounding box coordinates are wrapped with <bbox></bbox> tags, while metadata such as coordinates, titles, and fine-grained descriptions are also included as input to the model. Moreover, we set the Temperature to 0.8 and perform three extraction passes for each sample, retaining only the dimensions consistently identified across all runs. The final aggregated results are visualized in

- panel (b).

Ivy-Fake outperforms FakeVLM in terms of interpretability. While FakeVLM only explains superficial factors such as lighting and texture, Ivy-Fake considers multiple semantic dimensions, including hair, skin, and background, leading to a more comprehensive understanding of forged content.

This is a fake image. The image shows a man with short, styled hair and a beard, wearing a dark shirt, and making a gesture with his hands. Despite its seemingly authentic appearance, certain features, such as disproportionate textures or odd lighting, hint that this image was generated by AI. The image presents a convincing illusion, yet minor anomalies and inconsistencies in lighting or shadows indicate its synthetic nature. The overall structure and coherence of elements point towards it being computer-generated.

textures

lighting

FakeVLM

<think>\n1. **Spatial Analysis - Impractical Luminosity:**- The lighting ... but also in a synthetic environment where the background is rendered separately or completely blacked out. ...Spatial Analysis - Localized Blur:The subject's face and hands are in sharp focus, while there's a very subtle, almost imperceptible blur behind the hands, ...Spatial Analysis - Illegible Letters: There are no letters or text present in the image. Spatial Analysis - Distorted Components:The subject's anatomy (face, hands, arms, shoulders) appears proportional ...Subtle "Over-Perfection": Spatial Analysis - Abnormal Texture:The skin texture appears smooth but not overly so, with subtle pores visible. The shirt fabric texture is also consistent. Lack of Micro-Details/Noise: A truly real photograph, especially one taken in a studio, would still possess a certain level of photographic noise or grain, even if minimal. The image appears almost too clean.There's no excessive pattern repetition or unnatural smoothness that would strongly indicate synthetic generation. ..the "too perfect" quality rather than overt anomalies.</think>\n<conclusion> fake </conclusion>

face

lighting

...

IvyFake

hands

arms

- Figure 3 Comparison between Ivy-Fake and FakeVLM (Wen et al., 2025) (NeurIPS 2025).

###### 3.4 Quality Control

To mitigate the impact of hallucinations from Gemini 2.5 Pro during question generation, we sampled 8,000 cases for manual review. Each chain-of-thought (CoT) involving Gemini underwent at least two rounds of human verification. A total of ten users, including senior university students and regular participants, contributed to the verification process, which required approximately 1,000 hours in total. Each case was tested by at least two users to ensure robustness across different modalities of synthetic data. After this rigorous filtering, we retained 5,000 cases as the benchmark test set, consisting of 2,500 image samples and 2,500 video samples, with each modality containing 1,250 real and 1,250 fake instances.

###### 3.5 Method

Our preliminary investigations revealed that existing MLLMs exhibit inadequate performance on these tasks. To overcome this limitation, we propose Ivy-xDetector, a multimodal large language model designed explicitly for robust and explainable AIGC detection. The overall training pipeline is illustrated in Figure 4.

- Stage 1: Cold Start for Instruction-Driven Detection and Explainability. Due to limited reasoning ability, existing models struggle to detect AIGC content and generate reliable explanations. To address this, we inject Gemini-generated detection and explanatory COT into the training process, thereby improving the model’s fine-grained artifact perception and explanation quality. This stage aims to initialize Ivy-xDetector with instruction following and explainability capabilities. The resulting model not only performs accurate AIGC detection but also generates coherent, humanunderstandable rationales for its classifications.
- Stage 2: Sparse Rewards in Fine-grained Visual forgery Reasoning via Reinforcement Learning Although the fine-tuned model from Stage 1 demonstrates improved artifact awareness and interpretability, its generalization ability remains limited. To further enhance the model’s capacity for producing consistent and human-understandable explanations, we adopt the Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024). GRPO enables reinforcement learning (Zhou et al., 2026; Zhao et al., 2026b) using only binary classification samples, achieving efficient and data-light optimization.

We construct binary real/fake pairs from multiple datasets, each containing both authentic and synthetic images and videos. Ivy-xDetector takes the visual input and extracts the text enclosed

Stage1 Interpretabality Task Instruction Tuning

SFT

Image

###### 🔥

#### 🔥

❄

Vision Encoder

COT

###### Video

LLM Decoder

: :

|Is the Image/ Video real or<br><br>fake?|
|---|

Prompt

SFT

Tokenizer

|<think>..</think>\ n<conckusion>fake </conclusion>|
|---|

COT

COT

Stage2 Interpretabality Task By GRPO

Scaling AIGC Detection COT

- COT_1

- COT_2

- COT_3

- COT_4

- COT_5

###### 🔥

Is the image/video real or fake?

Policy Model

Advantage

Reward Model

(Filter, Clip

Image/Video Data

❄

higher)

Reference Model

- Figure 4 Overview of the three-stage training pipeline for Ivy-xDetector.

1 K

LGRPO(θ) = −

K

A(k) ℓ(k)(θ) + β KL(πθ(·|x)∥πref(·|x)). (2)

k=1

within the <conclusion>... </conclusion> tags, where the predicted label is either real or fake. The quantitative results of different training stages are presented in Table 7.

###### 3.6 GRPO

Following DeepSeek-R1 (Guo et al., 2025) and TextShield-R1 Qu et al. (2026b), we adapt the Group Relative Policy Optimization (GRPO) (Shao et al., 2024), an online RL algorithm designed to maximize the advantage of generated completions while constraining policy divergence from a reference model. We formalize our training process of Ivy-xDetector using GRPO below. Let p denote a sampled prompt, and let o1,o2,...,on be a group of completions generated by the current policy πθ. For each completion Gi, a reward ri is computed using a rule-based reward function. The group advantage for each completion is then calculated as:

ri − mean(r) std(r)

Aˆi,t =

(1)

where β is a coefficient that balances advantage maximization and KL regularization, and the clipping operator clip(...,1 − ϵ,1 + ϵ) constrains the update magnitude. To regularize policy updates, we estimate the token-level Kullback-Leibler (KL) divergence between the current policy πθ and a reference policy πref.

- Table 3 Performance comparison of models on image and video tasks. “Auto Metrics” include Acc, F1. “GPT Assisted” includes four subjective criteria: Comprehensiveness, Relevance, Detail, and Explanation. Bold indicates the best result, and underline indicates the second best.

Model

Image Video Overall Auto Metrics GPT Assisted Auto Metrics GPT Assisted Auto Metrics GPT Assisted

Acc/F1/ROUGE-L/SIM Com./Rel./Det./Exp. Acc/F1/ROUGE-L/SIM Com./Rel./Det./Exp. Acc/F1/ROUGE-L/SIM Com./Rel./Det./Exp.

Closed-source MLLMs GPT-4o 0.725/0.723/0.108/0.525 2.34/3.20/2.04/3.26 0.448/0.579/0.072/0.451 1.79/2.35/1.67/2.40 0.587/0.663/0.090/0.488 2.07/2.78/1.85/2.83 Gemini-2.5-Flash 0.747/0.737/0.263/0.733 3.94/4.11/4.04/4.09 0.810/0.811/0.246/0.723 4.00/4.37/4.03/4.36 0.779/0.776/0.254/0.728 3.97/4.24/4.04/4.22 Open-source MLLMs 7B-Parameters Models

InternVL3.5-8B 0.605/0.602/0.194/0.680 2.83/3.49/2.69/3.32 0.574/0.588/0.188/0.664 2.75/3.35/2.68/3.28 0.589/0.596/0.191/0.672 2.79/3.42/2.69/3.30 MiMo-VL-7B 0.662/0.637/0.121/0.593 1.99/2.80/1.85/2.89 0.778/0.783/0.112/0.580 2.04/2.91/1.90/3.19 0.720/0.715/0.116/0.586 2.01/2.86/1.87/3.04 Qwen2.5-VL-7B 0.013/0.026/0.006/0.264 1.02/1.03/1.01/1.52 0.092/0.159/0.015/0.280 1.14/1.23/1.12/2.21 0.053/0.096/0.010/0.272 1.08/1.13/1.07/1.86 LLaVA-OneVision-1.5-8B 0.500/0.333/0.080/0.499 1.51/2.62/1.49/2.38 0.500/0.333/0.068/0.481 1.49/2.26/1.37/2.19 0.500/0.333/0.074/0.490 1.50/2.44/1.43/2.28 MiniCPM-V-4.5 0.666/0.680/0.169/0.637 3.20/3.93/3.06/3.60 0.491/0.505/0.152/0.627 2.83/3.66/2.76/3.36 0.579/0.610/0.161/0.632 3.01/3.80/2.91/3.48 3B-Parameters Models

Qwen2.5-VL-3B 0.641/0.612/0.023/0.391 1.19/1.33/1.18/3.28 0.689/0.686/0.017/0.381 1.42/1.56/1.40/3.68 0.665/0.652/0.020/0.386 1.31/1.45/1.29/3.48 Gemma-3-4b-it 0.408/0.477/0.170/0.576 2.55/3.36/2.46/3.11 0.396/0.482/0.149/0.561 2.30/3.03/2.37/2.95 0.402/0.482/0.159/0.568 2.43/3.19/2.42/3.03 InternVL3.5-2B 0.602/0.573/0.177/0.648 2.62/3.29/2.51/3.13 0.435/0.459/0.159/0.631 2.46/3.16/2.42/2.99 0.518/0.518/0.168/0.640 2.54/3.22/2.47/3.06 InternVL3.5-4B 0.651/0.652/0.190/0.660 3.01/3.68/2.95/3.58 0.614/0.617/0.181/0.653 2.93/3.61/2.83/3.52 0.632/0.635/0.186/0.656 2.97/3.64/2.89/3.55 Ivy-xDetector 0.831/0.831/0.283/0.714 3.54/4.04/3.61/3.85 0.897/0.897/0.300/0.726 3.72/4.12/3.75/4.24 0.864/0.864/0.291/0.720 3.63/4.08/3.68/4.05

DKL πθ πref =

πref(oi,t | p,oi,<t) πθ(oi,t | p,oi,<t) − log

πref(oi,t | p,oi,<t) πθ(oi,t | p,oi,<t) − 1. (3)

- 3.7 Reward Model

For effective RL, we employ a rule-based reward that consists of accuracy and format rewards. We introduce a solid accuracy reward for AIGC Detection, which utilizes distinct functions to evaluate binary classification task. This allows for a more appropriate assessment based on the expected answer type.

- • Accuracy Reward: The accuracy reward assigns a score of 1 if the label in <conclusion> exactly matches the ground-truth classification real and fake and 0 otherwise.
- • Format Reward: The format reward assigns a score of 1 if the output strictly follows the structural requirements by enclosing the reasoning within <think></think> tags and the final decision within <conclusion></conclusion> tags, and 0 otherwise.

- 4 Experiments

###### 4.1 Experimental Details

Baselines We primarily evaluate three closed-source models and three open-source models on our Ivy-Fake. The closed-source models are GPT-4o (Achiam et al., 2023) and Gemini2.5 Flash (Team

- et al., 2023). For the open-source models, we select models of comparable size: InternVL3.5 (Chen
- et al., 2024c), Qwen2.5-VL (Bai et al., 2025), MiMo-VL (Team et al., 2025), LLaVA-OneVision1.5 (Contributors, 2025; Li et al., 2024) and MiniCPM-V-4.5 (Yao et al., 2025; Yu et al., 2025). Ivy-xDetector denotes our fully RL-trained model after the GRPO optimization stage.

Evaluation We report standard accuracy (Acc) and macro-averaged F1 score (F1) to assess the model’s ability to distinguish real from fake instances. For the reasoning task, we measure the similarity between the model’s reasoning process and the reference annotations using the ROUGE-L score (Lin, 2004) and BertScore (Zhang et al., 2020; He et al., 2021), which captures the longest common subsequence between predicted and reference texts, reflecting token-level overlap. Since ROUGE-L may fail to fully capture the fidelity of reasoning steps, we adopt an LLM-as-a-judge evaluation paradigm (Zheng et al., 2023), following the FakeBench protocol (Liu et al., 2025b), which assesses model responses along four dimensions: (1) Completeness: It reflects the extent to which the response fully addresses all aspects of the user’s question. More complete responses should incorporate information aligning well with the “golden clues” or reference answers. Incomplete or partially answered responses will receive lower scores. (2) Relevance: Measure how closely the content relates to the original annotation; (3) Level of Detail: Assess whether the response includes enough examples or elaborations; (4) Explanation: Verify the accuracy and consistency of explanations for any causes mentioned. Each response is scored using GPT-4o mini (Achiam et al.,

- 2023), which instructs the model to act as an impartial judge and assign a score from 1 to 5.

Abnormal Fragment

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Raw Videos 0:00:06 0:00:08 0:00:13 0:00:15

0:00 0:00:18

Conclusion Reasoning

Spatial Analysis Temporal Analysis

Reasoning: The video exhibits significant synthetic fingerprints. The lighting, a dominant visual element, ..., across time (nonphysical light interactions). Spatially, the textures, ..., these combined factors are characteristic of digitally manipulated or generated content rather than a direct, unaltered camera capture. Conclusion: fake ( )

Impractical Luminosity (Invisible light source detection & Scene brightness measurement):** Many scenes feature intense illumination without clearly defined or plausible physical sources. For example, at 0:00:06-0:00:08, the subject is brightly lit with strong rim lighting ...

Throughout the video, particularly at 0:00:130:00:15, the lighting exhibits characteristics that are temporally inconsistent with natural light sources. The bright, ethereal glows around the figure and the shifting ...

[Figure 23]

(b) Image: Our VS. FakeClue

- Figure 5 Ivy-xDetector performance on video detection.

###### Our FakeClue

[Figure 24]

Chromatic Irregularity: The reddish-brown coloration present on the dog's predominantly black coat (on the ears) appears in very distinct. While some breeds ... light on black fur. This can be a sign of synthetic imagery ... algorithmically or stylistically.

[Figure 25]

Implementation Details Our model was trained using the AdamW (Loshchilov and Hutter, 2017) optimizer with a cosine learning rate schedule. During the SFT stage, the model was trained for one epoch with a batch size of 1, which ensures stable convergence on large-scale multimodal data. In the subsequent RL stage, we adopted the GRPO algorithm, training for one epoch with the same batch size and a learning rate of 1 × 10−5. Within GRPO, the group size n was set to 8, the warm-up ratio to 0.01, the temperature to 0.9. Training the SFT and RL stages required approximately 10 and 50 hours, respectively, on a system equipped with 32 A100 GPUs.

This is a fake image, ..., despite its seemingly authentic appearance, subtle imperfections in the textures and details, ..., suggest it was artificially created. ( )

[Figure 26]

Abnormal Texture: The most significant anomaly is the texture of the dog's fur. ... The waves and individual strands of fur, ..., lacking the subtle random variations, clumping, ..., where textures are often rendered with excessive, non-organic precision.

[Figure 27]

Impractical Luminosity / Material Property: The highlights on the dog's fur are very smooth, ..., suggesting a non-physical interaction of light with a synthetically rendered material.

[Figure 28]

Raw Images

Raw Images

Reasoning: The image exhibits synthetic fingerprints, primarily ... appear more deliberately rendered than natural. The lighting on the fur also contributes to a non-photorealistic appearance. Conclusion: fake ( )

[Figure 29]

For video-based inputs, we configured the preprocessing pipeline to sample frames at a rate of 1 frame per second, with a maximum of 6 frames per clip to ensure temporal coherence while maintaining computational efficiency. Across the training pipeline, the overall input resolution was constrained by a global upper bound of 6,422,528 pixels (MAX_PIXELS), preventing overflow during multi-GPU parallel training and ensuring consistent input scaling across image and video modalities.

###### 4.2 Main Results

We perform extensive experiments to assess both detection and explanation capabilities. In particular, the proposed method is evaluated on the classification (real/fake) tasks and reasoning tasks for both

- Table 4 Comparison on the Chameleon (Yan et al., 2025). Accuracy (%) of different detectors (rows) in detecting real and fake images. For each training dataset, the first row is overall accuracy, the second row is “fake/real” class accuracy.

CNNSpot FreDect Fusing GramNet LNP UnivFD DIRE PatchCraft NPR AIDE Ivy-xDetector

60.89 57.22 57.09 60.95 58.52 60.42 59.71 56.32 58.13 65.77 73.17 9.86/99.25 0.89/99.55 0.02/99.98 4.76/99.66 7.72/96.70 85.52/41.56 11.86/95.67 3.07/96.35 2.43/100.00 26.80/95.06 67.78/77.22

- Table 5 Comparison on the Genimage (Zhu et al., 2023b). Accuracy (%) of different detectors (rows) in detecting real and fake images from different generators (columns). The best result and the second-best result are marked in bold and underline, respectively. Results of other methods are reported from (Yan et al., 2025).

Method Midjourney SD v1.4 SD v1.5 ADM GLIDE Wukong VQDM BigGAN Mean CNNSpot (Wang et al., 2020) 52.80 96.30 95.90 50.10 39.80 78.60 53.40 46.80 64.21 F3Net (Qian et al., 2020) 50.10 99.90 99.90 49.90 50.00 99.90 49.90 49.90 68.69 DIRE (Wang et al., 2023) 60.20 99.90 99.80 50.90 55.00 99.20 50.10 50.20 70.66 GenDet (Zhu et al., 2023a) 89.60 96.10 96.10 58.00 78.40 92.80 66.50 75.00 81.56 PatchCraft (Zhong et al., 2023) 79.00 89.50 89.30 77.30 78.40 89.30 83.70 72.40 82.30 FRIDA (Bonechi et al., 2025) 85.80 85.00 85.50 84.00 87.30 83.90 86.50 84.80 85.30 AIDE (Yan et al., 2025) 79.38 99.74 99.76 78.54 91.82 98.65 80.26 66.89 86.88 DRCT (Chen et al., 2024a) 91.50 95.00 94.40 79.40 89.20 94.70 90.00 81.70 89.50 Effort (Yan et al., 2024) 82.40 99.80 99.80 78.70 93.30 97.40 91.70 77.60 91.10 ThinkFake-7B (Huang et al., 2025b) 92.50 93.10 95.30 73.10 87.40 93.60 66.20 70.80 84.00 Ivy-xDetector 96.41 97.07 97.14 95.59 95.95 96.92 96.03 95.22 96.32

image and video content using the proposed unified framework. For the classification task, we test our model on both image and video content to detect the synthesic content.

From Table 3, our method demonstrates robust detection and explanation capabilities across both image and video tasks. Overall, closed-source models maintain a clear advantage in classification accuracy and subjective explanation quality. Among them, Gemini-2.5-flash achieves the best performance on both automatic metrics and GPT-assisted evaluation (e.g., reaching Acc/F1 of 0.812/0.812 on video detection and an average explanation score above 4.0), reflecting strong overall detection and reasoning ability. In contrast, GPT-4o maintains relatively high classification accuracy.

For open-source models, most models in the 3B–7B parameter range exhibit relatively low scores on explanation dimensions (< 3.5), indicating limitations in generating reasoning chains and covering multi-dimensional details. Notably, Qwen2.5-VL-3B shows relatively stable performance in explanation relevance and detail (Det./Exp. > 3.2), suggesting that lightweight models still hold potential under specific designs.

Notably, Gemini-2.5-Flash achieves the highest scores across all explanation-related dimensions, with an average of 4.12, surpassing all other models. This advantage can be attributed to the fact that part of our training data was distilled from Gemini-2.5-Pro, a powerful multimodal model capable of handling complex reasoning and explanation tasks. In contrast, our Ivy-xDetector ranks second with an average explanation score of 3.85, demonstrating interpretability comparable to Gemini-2.5-Flash while achieving substantially higher detection accuracy (Acc/F1 = 0.864/0.864) than any other open-source model. Compared with Gemini-2.5-Flash (Acc/F1 = 0.780/0.776), Ivy-xDetector improves accuracy by over 10 percentage points, confirming its robustness in both synthetic image detection and interpretive reasoning tasks.

###### 4.3 Generalization Evaluation

Evaluation on Classification Benchmarks. We conduct evaluations across multiple public leaderboards, including the image-based GenImage 5 and the video-based GenVideo benchmark 6. Per-

- Table 6 Comparisons to the GenVideo (Chen et al., 2024b). F1 score (F1), recall score (R) and average precision (AP) on the many-to-many generalization task. “Demamba-XCLIP-FT” is abbreviated as “Demamba”. Results of other methods are reported from (Chen et al., 2024b).

Model Metric Sora Morph Gen2 HotShot Lavie Show-1 Moon Crafter Model Wild Avg.

Studio Valley Scope Scrape F3Net (Image)

R 0.8393 0.9971 0.9862 0.7757 0.5700 0.3657 0.9952 0.9971 0.8943 0.7678 0.8188 F1 0.5000 0.9406 0.9628 0.8169 0.6988 0.4904 0.9332 0.9688 0.8873 0.8251 0.8024

NPR (Image)

R 0.9107 0.9957 0.9949 0.2429 0.8964 0.5771 0.9712 0.9986 0.9429 0.8780 0.8408 F1 0.2786 0.8441 0.9131 0.3028 0.8627 0.5944 0.8170 0.9164 0.8184 0.8163 0.7164

STIL (Video)

R 0.7857 0.9814 0.9804 0.7600 0.6179 0.5329 0.9936 0.9736 0.9457 0.6501 0.8222 F1 0.3805 0.9068 0.9458 0.7824 0.7232 0.6217 0.9039 0.9433 0.8884 0.7267 0.7823

DeMamba (Video)

R 0.9812 1.0000 0.9986 0.6543 0.9486 0.9886 1.0000 1.0000 0.9286 0.8909 0.9302 F1 0.6407 0.9602 0.9790 0.7539 0.9537 0.9551 0.9557 0.9797 0.9240 0.9120 0.9020

Ivy-xDetector

R 0.7857 0.9371 0.9507 0.9443 0.9550 0.9643 0.9968 0.9857 0.8943 0.9461 0.9528 F1 0.8800 0.9676 0.9747 0.9713 0.9770 0.9818 0.9984 0.9928 0.9442 0.9723 0.9526

- Table 7 Ablation study on different SFT and RL training settings. Base Model denotes the base model Qwen2.5-VL-3B. Evaluation metric: ACC / F1.

Training Stage

Model Name

Chameleon (Image) GenVideo (Video) Ivy-Fake Test

SFT RL

Base Model 59.05 / 53.30 67.70 / 80.74 65.10 / 61.80

- - ✓ 40.31 / 37.99 91.61 / 91.55 73.91 / 73.83

- - ✓ 72.72 / 70.79 78.19 / 77.62 77.00 / 76.66

- - ✓ ✓ 73.17 / 73.14 95.29 / 95.29 86.40 / 86.30

formance results of other competing detectors are taken from AIDE (Yan et al., 2025) and Demamba (Chen et al., 2024b). Our method achieves superior performance on all these leaderboards, demonstrating superior detection accuracy and robust cross-modal consistency.

Evaluation on Unseen Benchmarks. Beyond the datasets aligned with our training sets, we further evaluate the generalization ability of Ivy-xDetector on the Chameleon benchmarkYan et al. (2025), which lies outside the training data distribution, as shown in Table 4. The results confirm that our approach preserves strong generalization capability across unseen generative models and diverse data domains.

Compared to the original leaderboard SOTA method AIDE, our model achieves an overall accuracy of 73.17%. Notably, while the accuracy for the real class decreases from 95.06% to 77.22%, the accuracy for the fake class substantially increases from 26.80% to 67.78%. This suggests that AIDE tends to overfit the real category, whereas our model achieves a more balanced and robust performance across both real and synthetic content.

Ablation Study. To validate the effectiveness of our training framework, we design four ablation variants, as shown in Table 7. As shown, applying SFT alone results in only a slight performance drop on the out-of-domain benchmark Chameleon, while achieving consistent improvements across all other datasets. This suggests that the SFT stage primarily enhances the model’s fine-grained sensitivity to synthetic artifacts. In contrast, applying RL alone yields modest but consistent gains in classification accuracy across benchmarks. The combination of SFT and RL achieves the best balance between accuracy and interpretability.

##### 5 Conclusion

We introduced Ivy-Fake, the first unified, large-scale dataset for explainable AIGC detection across both images and videos, featuring over 106K richly annotated training samples and 5,000 evaluation examples with natural-language reasoning, and proposed Ivy Explainable Detector (Ivy-xDetector), a MLLM that jointly detects and explains synthetic content. Our model sets superior benchmarks in AIGC detection and explainability, and our publicly released resources provide a robust foundation for transparent, trustworthy multimodal analysis. We believe that such fine-grained explainability data Ivy-Fake can catalyze new research directions for the AIGC detection community, enabling deeper understanding of synthetic content and more principled model development.

Limitations: Future work will concentrate on tailoring the benchmark toward domain-specific scenarios, including explainability-focused datasets for DeepFake and document manipulation analysis. Nevertheless, our experiments suggest that the proposed approach exhibits strong scalability and generalization capabilities, indicating the potential for even better performance when applied to larger models such as 32B or 72B.

Broader impacts: The interpretability evaluation framework we propose is currently applied only to the field of Fake Image Detection (Du et al., 2025), which is strictly a binary classification task. In future work, our multi-dimensional evaluation approach can be extended to assess the interpretability of other tasks, such as chain-of-thought evaluation for Table Question Answering (Jiang et al., 2025a), question answering in medical scenarios (Zhao et al., 2026a; Xin et al., 2024, 2026a), GUI Agents (Yuan et al., 2025), Document Generation Ying et al. (2024a,a,b) and video grounding interpretability (Lin et al., 2025).

##### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jianfa Bai, Man Lin, Gang Cao, and Zijie Lou. Ai-generated video detection via spatial-temporal anomaly learning. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pages 460–470. Springer, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science, 2(3):8, 2023.

Rohit Bharadwaj, Hanan Gani, Muzammal Naseer, Fahad Shahbaz Khan, and Salman Khan. Vane-bench: Video anomaly evaluation benchmark for conversational lmms, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Simone Bonechi, Paolo Andreini, and Barbara Toniella Corradini. Who made this? fake detection and source attribution with diffusion features, 2025. https://arxiv.org/abs/2510.27602.

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. arXiv preprint arXiv:2405.19707,

2024. https://openai.com/research/video-generation-models-as-world-simulators. Bin Cao, Jianhao Yuan, Yexin Liu, Jian Li, Shuyang Sun, Jing Liu, and Bo Zhao. Synartifact: Classifying and alleviating artifacts in synthetic images via vision-language model. arXiv preprint arXiv:2402.18068, 2024. Baoying Chen, Jishen Zeng, Jianquan Yang, and Rui Yang. Drct: diffusion reconstruction contrastive training towards universal detection of diffusion generated images. In ICML, ICML’24. JMLR.org, 2024a.

Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, and Huaxiong Li. Demamba: Ai-generated video detection on million-scale genvideo benchmark. arXiv preprint arXiv:2405.19707, 2024b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024c.

LLaVA Community Contributors. Llava-onevision-1.5: Fully open framework for democratized multimodal training. In arxiv, 2025. Jingyi Deng, Chenhao Lin, Zhengyu Zhao, Shuai Liu, Qian Wang, and Chao Shen. A survey of defenses against ai-generated visual

media: Detection, disruption, and authentication. arXiv preprint arXiv:2407.10575, 2024. Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. Shichao Dong, Jin Wang, Jiajun Liang, Haoqiang Fan, and Renhe Ji. Explaining deepfake detection by analysing image matching. In European conference on computer vision, pages 18–35. Springer, 2022. Bo Du, Xuekang Zhu, Xiaochen Ma, Chenfan Qu, Kaiwen Feng, Zhe Yang, Chi-Man Pun, Jian Liu, and Ji-Zhe Zhou. Forensichub:

A unified benchmark & codebase for all-domain fake image detection and localization. arXiv preprint arXiv:2505.11003, 2025. Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua

Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi,

et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. In

International Conference on Learning Representations, 2021. https://openreview.net/forum?id=XPZIaotutsD.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Yan Hong, Jianming Feng, Haoxing Chen, Jun Lan, Huijia Zhu, Weiqiang Wang, and Jianfu Zhang. Wildfake: A large-scale and hierarchical dataset for ai-generated images detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(4): 3500–3508, Apr. 2025. doi: 10.1609/aaai.v39i4.32363. https://ojs.aaai.org/index.php/AAAI/article/view/32363.

Qing Huang, Zhipei Xu, Xuanyu Zhang, and Jian Zhang. Unishield: An adaptive multi-agent framework for unified forgery image detection and localization. arXiv preprint arXiv:2510.03161, 2025a.

Tai-Ming Huang, Wei-Tung Lin, Kai-Lung Hua, Wen-Huang Cheng, Junichi Yamagishi, and Jun-Cheng Chen. Thinkfake: Reasoning in multimodal large language models for ai-generated image detection, 2025b. https://arxiv.org/abs/2509.19841.

Yikun Ji, Yan Hong, Bowen Deng, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang, et al. Zoom-in to sort ai-generated images out. arXiv preprint arXiv:2510.04225, 2025.

Changjiang Jiang, Fengchang Yu, Haihua Chen, Wei Lu, and Jin Zeng. Tabdsr: Decompose, sanitize, and reason for complex numerical reasoning in tabular data. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 3172–3196, 2025a.

Changjiang Jiang, Xinkuan Sha, Fengchang Yu, Jingjing Liu, Jian Liu, Mingqi Fang, Chenfeng Zhang, and Wei Lu. Fake-hr1: Rethinking reasoning of vision language model for synthetic image detection. In ICASSP, 2026. https://arxiv.org/abs/2602. 10042.

Wan Jiang, Jing Yan, Ruixuan Zhang, Xiaojing Chen, Changtao Miao, Zhe Li, Chenhao Lin, Yunfeng Diao, and Richang Hong. Revisiting reconstruction-based ai-generated image detection: A geometric perspective. arXiv e-prints, pages arXiv–2510, 2025b.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset, 2017. https: //arxiv.org/abs/1705.06950.

Mamadou Keita, Wassim Hamidouche, Hessen Bougueffa Eutamene, Abdelmalik Taleb-Ahmed, David Camacho, and Abdenour Hadid. Bi-lora: A vision-language approach for synthetic image detection. Expert Systems, 42(2):e13829, 2025.

Zhenglun Kong, Yize Li, Fanhu Zeng, Lei Xin, Shvat Messica, Xue Lin, Pu Zhao, Manolis Kellis, Hao Tang, and Marinka Zitnik. Token reduction should go beyond efficiency in generative models–from vision, language to multimodality. arXiv preprint arXiv:2505.18227, 2025.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. https://arxiv.org/abs/2506.15742.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2024.

Ziqiang Li, Jiazhen Yan, Ziwen He, Kai Zeng, Weiwei Jiang, Lizhi Xiong, and Zhangjie Fu. Is artificial intelligence generated image detection a solved problem? arXiv preprint arXiv:2505.12335, 2025.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. Junan Lin, Daizong Liu, Xianke Chen, Xiaoye Qu, Xun Yang, Jixiang Zhu, Sanyuan Zhang, and Jianfeng Dong. Audio does matter:

Importance-aware multi-granularity fusion for video moment retrieval. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 6027–6036, 2025.

Jiaxin Liu, Jia Wang, Saihui Hou, Min Ren, Huijia Wu, Long Ma, Renwang Pei, and Zhaofeng He. Beyond face swapping: A diffusion-based digital human benchmark for multimodal deepfake detection, 2025a. https://arxiv.org/abs/2505.16512.

Ruiqi Liu, Manni Cui, Ziheng Qin, Zhiyuan Yan, Ruoxin Chen, Yi Han, Zhiheng Li, Junkai Chen, ZhiJin Chen, Kaiqing Lin, et al. Mirror: Manifold ideal reference reconstructor for generalizable ai-generated image detection. arXiv preprint arXiv:2602.02222, 2026.

Xuannan Liu, Zekun Li, Peipei Li, Huaibo Huang, Shuhan Xia, Xing Cui, Linzhi Huang, Weihong Deng, and Zhaofeng He. Mmfakebench: A mixed-source multimodal misinformation detection benchmark for lvlms. In ICLR, 2025b.

Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. ArXiv, abs/1711.05101, 2017. https://api.

###### semanticscholar.org/CorpusID:3312944.

Xiaochen Ma, Xuekang Zhu, Lei Su, Bo Du, Zhuohang Jiang, Bingkui Tong, Zeyu Lei, Xinyu Yang, Chi-Man Pun, Jiancheng Lv, et al. Imdl-benco: A comprehensive benchmark and codebase for image manipulation detection & localization. Advances in Neural Information Processing Systems, 37:134591–134613, 2025.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Kyoungjun Park, Yifan Yang, Juheon Yi, Shicheng Zheng, Yifei Shen, Dongqi Han, Caihua Shan, Muhammad Muaz, and Lili Qiu.

Vidguard-r1: Ai-generated video detection and explanation via reasoning mllms and rl. arXiv preprint arXiv:2510.02282, 2025. Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining

frequency-aware clues. In European conference on computer vision, pages 86–103. Springer, 2020.

Chenfan Qu, Chongyu Liu, Yuliang Liu, Xinhong Chen, Dezhi Peng, Fengjun Guo, and Lianwen Jin. Towards robust tampered text detection in document image: New dataset and new solution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5937–5946, June 2023.

Chenfan Qu, Yiwu Zhong, Chongyu Liu, Guitao Xu, Dezhi Peng, Fengjun Guo, and Lianwen Jin. Towards modern image manipulation localization: A large-scale dataset and novel methods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10781–10790, June 2024.

Chenfan Qu, Yiwu Zhong, Fengjun Guo, and Lianwen Jin. Revisiting tampered scene text detection in the era of generative ai. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 694–702, 2025.

Chenfan Qu, Yiwu Zhong, Fengjun Guo, and Lianwen Jin. Omni-IML: Towards unified interpretable image manipulation localization. In The Fourteenth International Conference on Learning Representations, 2026a.

Chenfan Qu, Yiwu Zhong, Jian Liu, Xuekang Zhu, Bohan Yu, and Lianwen Jin. Textshield-r1: Reinforced reasoning for tampered text detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8621–8629, 2026b.

Shavez Mushtaq Qureshi, Atif Saeed, Sultan H. Almotiri, Farooq Ahmad, and Mohammed A. Al Ghamdi. Deepfake forensics: a survey of digital forensic methods for multimodal deepfake identification on social media. PeerJ Computer Science, 10, 2024. https://api.semanticscholar.org/CorpusID:270088699.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. https://api.semanticscholar.org/CorpusID:231591445.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. https://arxiv.org/ abs/2402.03300.

Chao Shuai, Zhenguang Liu, Shaojing Fan, Bin Gong, Weichen Lian, Xiuli Bi, Zhongjie Ba, and Kui Ren. When detectors forget forensics: Blocking semantic shortcuts for generalizable ai-generated image detection. arXiv preprint arXiv:2603.09242, 2026.

Jacob Steinhardt, Pang Wei Koh, and Percy Liang. Certified defenses for data poisoning attacks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 3520–3532, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Hao Tan, Jun Lan, Senyuan Shi, Zichang Tan, Zijian Yu, Huijia Zhu, Weiqiang Wang, Jun Wan, and Zhen Lei. Videoveritas: Ai-generated video detection via perception pretext reinforcement learning. arXiv preprint arXiv:2602.08828, 2026a.

Hao Tan, Jun Lan, Zichang Tan, Ajian Liu, Chuanbiao Song, Senyuan Shi, Huijia Zhu, Weiqiang Wang, Jun Wan, and Zhen Lei. Veritas: Generalizable deepfake detection via pattern-aware reasoning. In International Conference on Learning Representations, 2026b.

Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025. https://arxiv.org/abs/2506.03569.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai,

Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to

spot... for now. In ICCV, pages 8695–8704, 2020.

Youqi Wang, Shen Chen, Haowei Wang, Rongxuan Peng, Taiping Yao, Shunquan Tan, Changsheng Chen, Bin Li, and Shouhong Ding. Forgeryvcr: Visual-centric reasoning via efficient forensic tools in mllms for image forgery detection and localization. arXiv preprint arXiv:2602.14098, 2026.

Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In ICCV, pages 22445–22455, 2023.

Siwei Wen, Junyan Ye, Peilin Feng, Hengrui Kang, Zichen Wen, Yize Chen, Jiang Wu, Wenjun Wu, Conghui He, and Weijia Li. Spot the fake: Large multimodal model-based synthetic image detection with artifact explanation. NeurIPS, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. https://arxiv.org/abs/2508.02324.

Lei Xin, Caiyun Huang, Hao Li, Shihong Huang, Yuling Feng, Zhenglun Kong, Zicheng Liu, Siyuan Li, Chang Yu, Fei Shen, et al. Artificial intelligence for central dogma-centric multi-omics: Challenges and breakthroughs. arXiv preprint arXiv:2412.12668, 2024.

Lei Xin, Zhenglun Kong, Fukang Chen, Yuhao Zheng4 Zeheng Wang, and Hao Tang. Dualcpt: Dual-branch modeling for cellular phenotype transition. In AAAI Bridge Program on AI for Medicine and Healthcare, pages 302–312. PMLR, 2026a.

Lei Xin, Yuhao Zheng, Ke Cheng, Changjiang Jiang, Zifan Zhang, and Fanhu Zeng. Hytrec: A hybrid temporal-aware attention architecture for long behavior sequential recommendation. arXiv preprint arXiv:2602.18283, 2026b. https://arxiv.org/abs/ 2602.18283.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016.

Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. A sanity check for AI-generated image detection. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview.net/forum? id=ODRHZrkOQM.

Zhiyuan Yan, Jiangming Wang, Zhendong Wang, Peng Jin, Ke-Yue Zhang, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. Effort: Efficient orthogonal modeling for generalizable ai-generated image detection. In ICML, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. Nat Commun 16, 5509 (2025), 2025.

Junyan Ye, Baichuan Zhou, Zilong Huang, Junan Zhang, Tianyi Bai, Hengrui Kang, Jun He, Honglin Lin, Zihao Wang, Tong Wu, et al. Loki: A comprehensive synthetic data detection benchmark using large multimodal models. ICLR, 2025.

Dehao Ying, Fengchang Yu, Haihua Chen, and Wei Lu. Dig: Complex layout document image generation with authentic-looking text

for enhancing layout analysis. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 3239–3247, 2024a. Dehao Ying, Fengchang Yu, Haihua Chen, and Wei Lu. Fine-grained, accurate data generation and multimodal layout analysis for

academic papers. In Proceedings of the 24th ACM/IEEE Joint Conference on Digital Libraries, pages 1–11, 2024b. Dehao Ying, Fengchang Yu, Haihua Chen, Changjiang Jiang, Yurong Li, and Wei Lu. Beyond human annotation: Recent advances in data generation methods for document intelligence. arXiv preprint arXiv:2601.12318, 2026.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, Bokai Xu, Junbo Cui, Yingjing Xu, Liqing Ruan, Luoyuan Zhang, Hanyu Liu, Jingkun Tang, Hongyuan Liu, Qining Guo, Wenhao Hu, Bingxiang He, Jie Zhou, Jie Cai, Ji Qi, Zonghao Guo, Chi Chen, Guoyang Zeng, Yuxuan Li, Ganqu Cui, Ning Ding, Xu Han, Yuan Yao, Zhiyuan Liu, and Maosong Sun. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe, 2025. https://arxiv.org/abs/2509.18154.

Yangxin Yu, Yue Zhou, Bin Li, et al. Agentfox: Llm agent-guided fusion with explainability for ai-generated image detection. arXiv preprint arXiv:2603.23115, 2026.

Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, et al. Se-gui: Enhancing visual grounding for gui agents via self-evolutionary reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Lingzhi Zhang, Zhengjie Xu, Connelly Barnes, Yuqian Zhou, Qing Liu, He Zhang, Sohrab Amirghodsi, Zhe Lin, Eli Shechtman, and Jianbo Shi. Perceptual artifacts localization for image synthesis tasks. In ICCV, pages 7579–7590, 2023.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. https://openreview.net/forum?id=SkeHuCVFDr.

Yonggang Zhang, Jun Nie, Xinmei Tian, Mingming Gong, Kun Zhang, and Bo Han. Detecting generated images by fitting natural image distributions. arXiv preprint arXiv:2511.01293, 2025.

Ming Zhao, Wenhui Dong, Yang Zhang, XiangZheng, Zhonghao Zhang, Zian Zhou, YUNZHI GUAN, Liukun Xu, Wei Peng, Zhaoyang Gong, Zhicheng Zhang, Dachuan li, Xiaosheng Ma, Yuli Ma, Jianing Ni, Changjiang Jiang, Lixia Tian, Chen Qixin, Xia Kaishun, Pingping Liu, Tongshun Zhang, ZhiqiangLiu, Zhongan Bi, Chenyang Si, Tiansheng Sun, and Caifeng Shan. Spinebench: A clinically salient, level-aware benchmark powered by the spinemed-450k corpus. In The Fourteenth International Conference on Learning Representations, 2026a. https://arxiv.org/abs/2601.12318.

Qiannian Zhao, Chen Yang, Jinhao Jing, Yunke Zhang, Xuhui Ren, Lu Yu, Shijie Zhang, and Hongzhi Yin. Know what you know: Metacognitive entropy calibration for verifiable rl reasoning. arXiv preprint arXiv:2602.22751, 2026b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

Nan Zhong, Yiran Xu, Sheng Li, Zhenxing Qian, and Xinpeng Zhang. Patchcraft: Exploring texture patch for efficient ai-generated image detection. arXiv preprint arXiv:2311.12397, 2023.

Jiang Zhou, Xiaohu Zhao, Xinwei Wu, Tianyu Dong, Hao Wang, Yangyang Liu, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, and Deyi Xiong. Incentivizing parametric knowledge via reinforcement learning with verifiable rewards for cross-cultural entity translation. arXiv preprint arXiv:2604.16881, 2026.

Yue Zhou, Xinan He, KaiQing Lin, Bin Fan, Feng Ding, and Bin Li. Breaking latent prior bias in detectors for generalizable aigc image detection. arXiv preprint arXiv:2506.00874, 2025a.

Ziyin Zhou, Yunpeng Luo, Yuanchen Wu, Ke Sun, Jiayi Ji, Ke Yan, Shouhong Ding, Xiaoshuai Sun, Yunsheng Wu, and Rongrong Ji. Aigi-holmes: Towards explainable and generalizable ai-generated image detection via multimodal large language models. ArXiv, abs/2507.02664, 2025b. https://api.semanticscholar.org/CorpusID:280141523.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

Mingjian Zhu, Hanting Chen, Mouxiao Huang, Wei Li, Hailin Hu, Jie Hu, and Yunhe Wang. Gendet: Towards good generalizations for ai-generated image detection. arXiv preprint arXiv:2312.08880, 2023a.

Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, and Yunhe Wang. Genimage: a million-scale benchmark for detecting ai-generated image. In NeurIPS, NIPS ’23, Red Hook, NY, USA, 2023b. Curran Associates Inc.

Xuekang Zhu, Xiaochen Ma, Lei Su, Zhuohang Jiang, Bo Du, Xiwen Wang, Zeyu Lei, Wentao Feng, Chi-Man Pun, and Ji-Zhe Zhou. Mesoscopic insights: Orchestrating multi-scale & hybrid architecture for image manipulation localization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 11022–11030, 2025a.

Xuekang Zhu, Ji-Zhe Zhou, Kaiwen Feng, Chenfan Qu, Yunfei Wang, Liting Zhou, and Jian Liu. Does the manipulation process matter? rita: Reasoning composite image manipulations via reversely-ordered incremental-transition autoregression. arXiv preprint arXiv:2509.20006, 2025b.

# Appendix

##### A Author and Affiliation List

Nanjing University Wenhui Dong* , Chenyang Si, Caifeng Shan Wuhan University Changjiang Jiang*+, Fengchang Yu Stanford University Wei Peng Ningxia University Zhonghao Zhang Nankai University Xinbin Yuan Georgia Institute of Technology Yifei Bi Jilin University Ming Zhao Zhejiang University Zian Zhou

* Equal contribution. Corresponding author.

+ This work was done while serving as a research intern at Nanjing University.

##### B License

For all datasets in our experiments and datasets, FLUX.1 (Labs et al., 2025) and Lora Flux (Labs et al., 2025) falls under the FLUX.1 [dev] Non-Commercial License.WildFake (Hong et al., 2025) is released under an open-source license, making it freely available for research and non-commercial use; Qwen-Image (Wu et al., 2025) is licensed under Apache 2.0; Sora (Brooks et al., 2024) is a commercial, non-open-source text-to-video model provided via ChatGPT Plus/Pro; user-generated content is owned by the user and allows (subject to usage policies) non-commercial use, but the model itself is closed-source.

Stable Diffusion (Blattmann et al., 2023), while originally open-source under Creative ML OpenRAILM (e.g., versions 1.X, 2.1, SDXL), has transitioned in newer versions (SD3.x, SD3.5) to the Stability AI Community License, which permits free use for individuals or entities with annual revenue under USD 1 million, but requires a paid Enterprise License for larger-scale or commercial usage.

During Ivy-Fake construction and experimentation, we accessed Gemini API and ChatGPT API to ensure annotation and evaluation’s quality and reproducibility. We explicitly state that all data generated using these commercial APIs are used solely for academic research and non-commercial purposes, fully complying with the respective API usage agreements and ethical guidelines. No Gemini or ChatGPT derived content is redistributed for any commercial training, deployment, or monetization purposes.

##### C Related Work

###### C.1 Methods for Synthetic Content Detection

Due to growing concerns about the misuse of synthetic data (Deng et al., 2024), research on AIgenerated content (AIGC) detection has expanded rapidly in recent years. Most existing models

###### Image Authenticity Analyst Assistant (User Prompt)

Is this image real or fake?

- Figure 6 User Prompt Template for Image Data Distillation

for AI-generated images and videos formulate the task as binary classification, simply predicting whether the content is "real" or "fake." Representative examples include CNN-based AIGVDet (Bai et al., 2024), CNNSpot (Wang et al., 2020) and Transformer-based models such as DIRE (Wang

- et al., 2023) and AIDE (Yan et al., 2025). Meanwhile, several works have explored the application of multimodal large language models (MLLMs) to AIGC detection, including Synartifact (Cao
- et al., 2024) and Bi-LORA (Keita et al., 2025). However, these approaches largely overlook the importance of interpretability in AIGC detection.

Some efforts attempt to introduce interpretability by leveraging spatial annotations (Dong et al., 2022) or frequency-domain artifact analysis (Zhang et al., 2023). Nevertheless, the resulting explanations are often difficult for humans to comprehend, as they lack clarity in natural language. This limitation is particularly evident in the video domain, where AI-generated content frequently exhibits obvious flaws, e.g., incoherent frame transitions and object inconsistency, that are easily noticed and reasoned about by humans (Deng et al., 2024). FakeClue (Wen et al., 2025) introduces the use of vision-language models (VLMs) to provide interpretability for image-level detection, but it does not offer a unified framework that integrates both images and videos.

Video Authenticity Analyst Assistant (User Prompt)

Is this video real or fake?

- Figure 7 User Prompt Template for Video Data Distillation

###### C.2 Datasets for Synthetic Content Detection.

Early datasets for synthetic content detection, such as CNNSpot (Wang et al., 2020), primarily collected fake images generated by GAN-based models (Goodfellow et al., 2014; Zhu et al., 2017; Brock et al., 2018). However, with the advent of more advanced generative architectures like diffusion models (Ho et al., 2020; Dhariwal and Nichol, 2021; Nichol et al., 2021) and their variants, the authenticity of generated content has significantly increased, making it more challenging for detection models to discern. This has spurred the development of newer datasets, including ArtiFact (Cao et al., 2024), GenImage (Zhu et al., 2023b), and WildFake (Hong et al., 2025). GenImage (Zhu et al., 2023b) comprises images from the 1000 ImageNet (Russakovsky et al., 2015) categories, generated by eight state-of-the-art generators. Nevertheless, these datasets predominantly focus on image-based content. More recently, datasets emphasizing interpretability have also been introduced. FakeClue (Wen et al., 2025) contains a large amount of image data with explainability annotations but lacks video data. LOKI (Ye et al., 2025) offers data across 26 different categories and includes 18,000 distinct questions; however, its volume of multimodal data is relatively small and primarily suited for evaluation rather than comprehensive model training. Therefore, a critical gap exists for a unified benchmark encompassing both image and video modalities to rigorously evaluate the performance of contemporary AIGC detectors.

###### Image Authenticity Analyst Assistant (System Prompt)

## Role Expert AI system for detecting image by analyzing visual anomalies across spatial plausibility.

## Analysis Dimensions ### Spatial Features: static anomaly detection

- - Impractical Luminosity
- - Scene brightness measurement
- - Invisible light source detection (physical validation)
- - Localized Blur
- - Focus distribution mapping (sharpness gradient)
- - Artificial depth-of-field identification (algorithmic artifacts)
- - Illegible Letters
- - OCR text extraction
- - Character structural integrity (stroke continuity)
- - Distorted Components
- - Anatomical/proportional accuracy (biological/object logic)
- - Physics compliance (material/gravity validation)
- - Omitted Components
- - Object completeness check (edge/detail absence)
- - Partial rendering artifact detection (AI-generated traces)
- - Spatial Relationships
- - Contextual object placement (scene plausibility)
- - Perspective consistency (geometric projection)
- - Chromatic Irregularity
- - Color database comparison (natural distribution)
- - Unnatural hue detection (oversaturation/abrupt gradients)
- - Abnormal Texture
- - Surface pattern regularity (texture repetition)
- - Material property coherence (reflectance/roughness validation)

## Reasoning Step

###### 1. Spatial Analysis

- Analyze static features (e.g., lighting, text, objects)

###### 2. Conclusion: Only real or fake.

- - real: Contains verifiable capture device signatures and natural physical imperfections.
- - fake: Exhibits synthetic fingerprints including but not limited to over-regularized textures and non-physical light interactions.

The assistant first thinks about the reasoning step in the mind and then provides the user with the reason. The reasoning step and conclusion are enclosed within <think> </think> and <conclusion> </conclusion> tags, respectively, i.e., <think> reasoning step here </think> <conclusion> real or fake </conclusion>. <conclusion> content must strictly align with the user-provided authenticity label (real/fake) in both value and semantic context.

- Figure 8 System Prompt Template for Image Data Distillation

##### D GRPO

Following DeepSeek-R1 (Guo et al., 2025), we adapt the Group Relative Policy Optimization (GRPO) (Shao et al., 2024), an online RL algorithm designed to maximize the advantage of generated completions while constraining policy divergence from a reference model. We formalize our training process of Ivy-xDetectorusing GRPO below. Let p denote a sampled prompt, and let o1,o2,...,on be a group of completions generated by the current policy πθ. For each completion Gi, a reward ri is computed using a rule-based reward function. The group advantage for each completion is then calculated as:

ri − mean(r) std(r)

Aˆi,t =

(4)

where β is a coefficient that balances advantage maximization and KL regularization, and the clipping operator clip(...,1 − ϵ,1 + ϵ) constrains the update magnitude. To regularize policy updates, we estimate the token-level Kullback-Leibler (KL) divergence between the current policy πθ and a reference policy πref.

Reward Model. For effective RL, we employ a rule-based reward that consists of accuracy and format rewards. We introduce a solid accuracy reward for AIGC Detection, which utilizes distinct functions to evaluate binary classification task. This allows for a more appropriate assessment based on the expected answer type.

- • Accuracy Reward: The accuracy reward assigns a score of 1 if the label in <conclusion> exactly matches the ground-truth classification real and fake and 0 otherwise.
- • Format Reward: The format reward assigns a score of 1 if the output strictly follows the structural requirements by enclosing the reasoning within <think></think> tags and the final decision within <conclusion></conclusion> tags, and 0 otherwise.

##### E Effect of Incorporating Human-Annotated Labels via gemini 2.5 pro on Accuracy

To assess the impact of human-annotated labels on model performance, we compare the accuracy of final conclusion predictions under two settings: (i) with labels incorporated via the gemini 2.5 pro, and (ii) without labels. The evaluation was conducted on about 1,000 examples from the test set.

###### Annotation Setting Accuracy (Acc)

With Label 1.000 Without Label 0.785

- Table 8 Accuracy of conclusion prediction with and without incorporating labels.

As shown in Table 8, incorporating ground-truth labels results in a substantial performance gain, yielding perfect accuracy (1.000), compared to 0.785 without labels. The drastic performance gap suggests potential limitations in label-free or weakly supervised setups when applied to tasks requiring fine-grained semantic understanding.

- F Data Distribution
- G Additional Experiment

- G.1 Synthetic Image Detection

We evaluated our AIGC detector on the GenImage (Zhu et al., 2023b) and Chameleon (Yan et al.,

2025) benchmarks.

- As shown in Table 5, GenImage comprises seven subsets generated by leading models, i.e., Midjourney, Stable Diffusion v1.4 & v1.5, ADM, GLIDE, Wukong, VQDM, and BigGAN. We compared against five state-of-the-art detectors, which are CNNSpot (Wang et al., 2020), F3Net (Qian et al., 2020), DIRE (Wang et al., 2023), GenDet (Zhu et al., 2023a), PatchCraft (Zhong et al., 2023), and AIDE (Yan et al., 2025).

Notably, our detector surpasses previous state-of-the-art methods such as AIDE (86.88%) and PatchCraft (82.30%) by a large margin, even though it is trained under a unified multimodal paradigm rather than a generator-specific setting. The improvement is especially pronounced on challenging generators like ADM, GLIDE, and BigGAN, where conventional CNN- or patch-based detectors often fail to capture high-frequency inconsistencies or generalized texture patterns.

Notably, our model is trained with only 130K image samples—fewer than GenImage’s 2M+ training images—yet still achieves superior generalization performance.

G.2 Synthetic Video Detection

- As shown in Table 6, we evaluate the performance of our detector on the GenVideo benchmark (Chen et al., 2024b).

Specifically, Ivy-xDetector achieves an average F1 score of 95.26% and recall of 95.28%, substantially surpassing the prior state-of-the-art DeMamba (F1 = 90.20%). The improvement is particularly evident on challenging categories such as HotShot, Lavie, and Show-1, where previous methods tend to overfit to specific generative distributions or temporal artifacts. Our results indicate that the unified MLLM-based detection paradigm not only captures spatial inconsistencies but also learns transferable temporal-spatial representations, leading to more robust generalization across unseen generative models.

Notably, our model is trained with only 64K video samples—over 35× fewer than DeMamba’s 2.295M training videos—yet still achieves superior generalization performance.

- H Prompts

Here we provide the prompts that are mainly used in this study. The default system prompt can be seen in Figure11. As illustrated by the following figures, there are five distillation prompts distillation we used in this paper that mainly can be divided into the following three parts:

Prompt Template for Image Data Distillation: Since image data consists of a single frame, it can be treated as a static instance. Therefore, AIGC detection mainly focuses on identifying spatial anomalies. Detail prompt can be found in Figure 6 and 8.

Prompt Template for Video Data Distillation: Compared to images, video inputs provide continuous multi-frame context. This allows for detection along both spatial and temporal anomaly

dimensions. Dtail prompt can be found in Figure 7 and Figure 9.

GPT Assisted Evaluation Prompt: To assess the quality of model outputs, we design a GPT-based evaluator prompt that scores responses across four dimensions: Completeness, Relevance, Level of Detail, and Explanation. The evaluator receives a structured pair of GroundTruth and ModelOutput, each containing a <think> section (reasoning) and a <conclusion> (final judgment). The model must return a structured JSON object with integer scores (1–5) for each dimension. The prompt is provided in Figure 12.

##### I Training Detail

The RL training process of Ivy-xDetector is illustrated in Fig.10. Our GRPO training exhibits a stable increase in reward, eventually reaching 0.95. However, we also observe that both the clip ratio and the maximum response length surge sharply around 200–300 steps, suggesting that a portion of the training samples may be overly challenging for the model.

##### J Case Study: Qualitative Comparison of Methods

According to Figures 13, 14, 15, and 16, Ivy-xDetector consistently demonstrates superior performance in detecting both spatial and temporal anomalies, providing stronger generalization and robustness compared to existing baselines.

###### Video Authenticity Analyst Assistant (System Prompt)

## Role Expert AI system for detecting videos by analyzing visual anomalies across temporal coherence (inter-frame dynamics) and spatial plausibility (intra-frame logic).

## Analysis Dimensions

- ### 1. Temporal Features: Multi-frame dynamic anomaly detection - Luminance Discrepancy

- - Shadow direction consistency (cross-frame comparison)
- - Light source coordination (temporal validation)
- - Awkward Facial Expression
- - Facial muscle motion continuity (expression dynamics)
- - Emotion-context alignment (temporal coherence)
- - Duplicated Components
- - Repeating element pattern recognition (cross-frame tracking)
- - Natural variation analysis (sequence validation)
- - Non-Spatial Relationships
- - Object interaction physics (motion trajectory validation)
- - Fusion/penetration anomalies (temporal detection)

- ### 2. Spatial Features: Single-frame static anomaly detection

- - Impractical Luminosity
- - Scene brightness measurement (single-frame analysis)
- - Invisible light source detection (physical validation)
- - Localized Blur
- - Focus distribution mapping (sharpness gradient)
- - Artificial depth-of-field identification (algorithmic artifacts)
- - Illegible Letters
- - OCR text extraction (single-frame recognition)
- - Character structural integrity (stroke continuity)
- - Distorted Components
- - Anatomical/proportional accuracy (biological/object logic)
- - Physics compliance (material/gravity validation)
- - Omitted Components
- - Object completeness check (edge/detail absence)
- - Partial rendering artifact detection (AI-generated traces)
- - Spatial Relationships
- - Contextual object placement (scene plausibility)
- - Perspective consistency (geometric projection)
- - Chromatic Irregularity
- - Color database comparison (natural distribution)
- - Unnatural hue detection (oversaturation/abrupt gradients)
- - Abnormal Texture
- - Surface pattern regularity (texture repetition)
- - Material property coherence (reflectance/roughness validation)

## Reasoning Step

###### 1. Temporal Analysis

- Track dynamic features across frames (e.g., shadows, expressions)

###### 2. Spatial Analysis

- Analyze static features per frame (e.g., lighting, text, objects)

###### 3. Conclusion: Only real or fake.

- - real: Contains verifiable capture device signatures and natural physical imperfections.
- - fake: Exhibits synthetic fingerprints including but not limited to over-regularized textures and non-physical light interactions.

The assistant first thinks about the reasoning step in the mind and then provides the user with the reason. The reasoning step and conclusion are enclosed within <think> </think> and <conclusion> </conclusion> tags, respectively, i.e., <think> reasoning step here </think> <conclusion> real or fake </conclusion>. <conclusion> content must strictly align with the user-provided authenticity label (real/fake) in both value and semantic context.

- Figure 9 System Prompt Template for Video Data Distillation

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

###### Figure 10 Final Reinforcement Learning Training Curves.

K

1 K

A(k) ℓ(k)(θ) + β KL(πθ(·|x)∥πref(·|x)). (5)

LGRPO(θ) = −

k=1

πref(oi,t | p,oi,<t) πθ(oi,t | p,oi,<t) − log

πref(oi,t | p,oi,<t) πθ(oi,t | p,oi,<t) − 1. (6)

DKL πθ πref =

Default System Prompt

You are an AI-generated content detector. Given a single media (image or video), classify it as real or fake. Provide detailed reasoning inside the <think>...</think> tags, including your step-by-step thought process.

Your output must begin with <think> n and end with </conclusion>.

Then output exactly one word in lowercase—either real or fake—wrapped in <conclusion>...</conclusion>. Do not include any other words. If uncertain, choose the most likely class.

###### Figure 11 Default System Prompt for Training and Evaluation

###### GPT Assisted Evaluation Prompt

## Role You are an impartial evaluator. Your task is to assess whether a model-generated response accurately and coherently matches a human-annotated reference answer.

Each input contains two structured components:

- - <think>: the reasoning or analytical explanation
- - <conclusion>: the final judgment (e.g., real or fake)

## Evaluation Dimensions You should compare the ModelOutput to the GroundTruth, and assign integer scores from 1 to 5 (no decimals) for the following four dimensions:

- 1. Completeness

- - Does the ModelOutput address all aspects covered in the GroundTruth?
- - More complete responses should include all relevant information, especially key golden¨ clues¨.
- - Incomplete or partially aligned answers should receive lower scores.

- 2. Relevance

- - Does the ModelOutput discuss the same detection dimensions as in the GroundTruth?
- - Temporal features include:
- - Luminance discrepancy
- - Duplicated components
- - Awkward facial expressions
- - Motion inconsistency
- - Spatial features include:
- - Abnormal texture
- - Distorted or omitted components
- - Chromatic irregularity
- - Impractical luminosity
- - Localized blur, etc.
- - Penalize if irrelevant aspects are introduced or relevant ones are missing.

- 3. Level of Detail

- - Does the ModelOutput describe fine-grained visual cues in each dimension?
- - High scores require specific subcomponent elaboration, not just general terms.
- - Penalize vague or generic responses that lack specific observations.

- 4. Explanation

- - Is the reasoning in <think> logically consistent with the <conclusion>?
- - The explanation should provide clear, causally-linked justifications.
- - Penalize if the conclusion contradicts the reasoning or lacks support.

- Figure 12 GPT Assisted Evaluation Prompt

[Figure 38]

###### Figure 13 Image example 1 where Ivy-xDetector successfully detects subtle spatial anomalies missed by baselines.

[Figure 39]

###### Figure 14 Image example 2 illustrating improved robustness of Ivy-xDetector against visually deceptive artifacts.

[Figure 40]

###### Figure 15 Video example 1 showcasing Ivy-xDetector’s superior ability to detect subtle cross-frame temporal artifacts.

[Figure 41]

###### Figure 16 Video example 2 showcasing Ivy-xDetector’s superior ability to detect subtle cross-frame temporal artifacts.

