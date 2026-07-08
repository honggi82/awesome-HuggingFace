## Adapting Vision-Language Models Without Labels: A Comprehensive Survey

Hao Dong∗, Lijun Sheng∗, Jian Liang†, Ran He, Eleni Chatzi, Olga Fink

### arXiv:2508.05547v1[cs.LG]7Aug2025

Abstract—Vision-Language Models (VLMs) have demonstrated remarkable generalization capabilities across a wide range of tasks. However, their performance often remains suboptimal when directly applied to specific downstream scenarios without task-specific adaptation. To enhance their utility while preserving data efficiency, recent research has increasingly focused on unsupervised adaptation methods that do not rely on labeled data. Despite the growing interest in this area, there remains a lack of a unified, task-oriented survey dedicated to unsupervised VLM adaptation. To bridge this gap, we present a comprehensive and structured overview of the field. We propose a taxonomy based on the availability and nature of unlabeled visual data, categorizing existing approaches into four key paradigms: Data-Free Transfer (no data), Unsupervised Domain Transfer (abundant data), Episodic Test-Time Adaptation (batch data), and Online Test-Time Adaptation (streaming data). Within this framework, we analyze core methodologies and adaptation strategies associated with each paradigm, aiming to establish a systematic understanding of the field. Additionally, we review representative benchmarks across diverse applications and highlight open challenges and promising directions for future research. An actively maintained repository of relevant literature is available at https://github.com/tim-learn/Awesome-LabelFree-VLMs.

Index Terms—Unsupervised learning, test-time adaptation, multimodal learning, vision-language models.

I. INTRODUCTION

# V

ISION-language models (VLMs), such as CLIP [1], ALIGN [2], Flamingo [3], and LLaVA [4] have attracted

considerable attention from both academia and industry due to their powerful cross-modal reasoning capabilities. These models learn joint image-text representations from large-scale datasets [5] and have demonstrated impressive zero-shot performance and generalization across a variety of tasks. VLMs have been successfully applied in diverse domains, including autonomous driving [6], robotics [7], anomaly detection [8], and cross-modal retrieval [9].

However, because the pre-training phase cannot capture the full diversity of downstream tasks and environments, adapting VLMs to specific applications remains a fundamental challenge. Early efforts primarily relied on supervised finetuning [10]–[13], which explores more knowledge in annotated

∗ Equal contribution. † Corresponding author. H. Dong and E. Chatzi are with ETH Z¨urich, Switzerland. (Email:

hao.dong@ibk.baug.ethz.ch; chatzi@ibk.baug.ethz.ch).

L. Sheng is with the University of Science and Technology of China. (Email: slj0728@mail.ustc.edu.cn).

J. Liang and R. He are with NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences. (Email: liangjian92@gmail.com; rhe@nlpr.ia.ac.cn).

O. Fink is with EPFL, Switzerland. (Email: olga.fink@epfl.ch). Manuscript received August 8, 2025.

examples. Despite their effectiveness, they still suffer from high annotation costs and performance degradation under distribution shifts [14] between training and test data. To address these limitations, a growing body of work has explored unsupervised adaptation techniques [15]–[20]. These approaches—often referred to as zero-shot inference [21]– [23], test-time methods [18], [24], [25], or unsupervised tuning [17], [26], [27]—aim to improve VLMs’ performance in downstream tasks without relying on costly annotation. Such methods have proven effective across a wide range of applications, including image classification [15], [17], [18], segmentation [16], [28], [29], medical image diagnosis [30], [31], and action recognition [32], [33].

Given the rapid growth of this research area, this survey provides a comprehensive and structured overview of existing unsupervised adaptation methods for VLMs. To the best of our knowledge, we are the first to introduce a taxonomy centered on the availability of unlabeled visual data—an often overlooked yet practically critical factor in real-world deployment. As illustrated in Fig. 1, we categorize existing approaches into four paradigms: (1) Data-Free Transfer [15], [16], [21], which adapts models using only textual class names; (2) Unsupervised Domain Transfer [17], [34], [35], which utilizes abundant unlabeled data from the downstream tasks; (3) Episodic Test-Time Adaptation [18], [24], [36], which adapts models to a batch of test instances; and (4) Online Test-Time Adaptation [19], [23], [25], which addresses the challenge of streaming test data. This taxonomy provides a principled framework for understanding the landscape of unsupervised VLM adaptation, guiding practitioners in selecting suitable techniques. We also believe our taxonomy will facilitate fair comparisons across future work within the same paradigm.

The organization of this survey follows the structure shown in Fig. 2. Sec. II provides an overview of several research topics related to unsupervised learning in the context of VLMs. Sec. III introduces zero-shot inference with VLMs and presents a comprehensive taxonomy based on the availability of unlabeled visual data. The central focus of this survey is discussed in Sec. IV - Sec. VII, where we analyze existing approaches within data-free transfer, unsupervised domain transfer, episodic test-time adaptation, and online testtime adaptation, respectively. Sec. VIII explores a variety of application scenarios that utilize unsupervised techniques and introduces related benchmarks, offering a broader perspective on their practical implications and real-world utility. Finally, we summarize emerging trends in the field and identify key scientific questions that could inspire future work in Sec. IX. Comparison with previous surveys. In recent years, sev-

|class<br><br>names a batch of<br><br>[Figure 1]<br><br>unlabeled data|
|---|

|class<br><br>names|
|---|

|T<br><br>I|
|---|

- 1. Data-Free Transfer

- 2. Unsupervised Domain Transfer

- 3. Episodic Test-Time Adaptation

- 4. Online Test-Time Adaptation

|class<br><br>names abundant<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>unlabeled data|
|---|

|class names streaming (online)<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>unlabeled data<br><br>… …|
|---|

downstream tasks pretrained VLMs

downstream tasks

Fig. 1: Illustration of our taxonomy on unsupervised adaptation with VLMs. We categorize existing unsupervised methods into four task paradigms based on the availability of unlabeled visual data.

eral surveys [37]–[40] have explored various aspects of unsupervised adaptation and fine-tuning of VLMs. Existing works [40]–[42] predominantly focus on unimodal model transfer, providing a thorough analysis of this domain, but they offer limited coverage of VLMs. An early work [37] discusses the pre-training stage of VLMs and briefly analyzes its finetuning method for vision tasks. Another survey [38] discusses the adaptation and generalization of multimodal models, but at a relatively coarse-grained level. A recent work [39] uses generalization to understand VLMs’ downstream tasks and reviews existing methods with a perspective of the parameter space. While these surveys contribute valuable insights, our work distinguishes itself by introducing, for the first time, a taxonomy based on the availability of unlabeled visual data and analyzing cutting-edge technologies in each of these paradigms. We believe this is a novel and crucial contribution to the field, especially in terms of the deployment of VLMs.

supervised objectives [54], [55]. To address limitations in generalization, generative models were employed to synthesize visual features for unseen classes [56]–[58]. Recent research emphasizes generalized settings [59], [60], aiming to improve robustness and fair evaluation across seen and unseen categories. ZSL serves as a foundational principle for unsupervised VLM adaptation, leveraging semantic descriptions (e.g., text prompts) to bridge seen and unseen classes. This enables models to generalize to novel visual concepts without requiring any labeled examples. For further information on ZSL, we refer the reader to the recent survey papers [61], [62].

C. Supervised Fine-Tuning of VLMs

Supervised fine-tuning of VLMs has emerged as a key strategy to adapt pre-trained models to downstream tasks with task-specific supervision. Researchers have increasingly focused on parameter-efficient fine-tuning techniques—such as prompt tuning [10], [12], [63], adapter modules [64], [65], and lightweight task-specific layers [66]—that allow models to adapt to new domains while preserving the generality of their pre-trained features. Moreover, some approaches use large language models (LLMs) to aid in adapting VLMs [21], [67] or adapt VLMs to dense prediction tasks [16], [68]. For further information on supervised fine-tuning of VLMs, we refer the reader to the recent survey papers [38], [69]. Instead of relying on explicit label supervision, this survey focuses on unsupervised VLM adaptation, where models must adapt to downstream tasks without access to annotated data.

II. RELATED RESEARCH TOPICS

- A. Vision-Language Models

Recent progress in VLMs has been remarkable, driven by the integration of large-scale pre-training [1], [43], transformer architectures [44], [45], and massive multimodal datasets [5], [46]. Models such as CLIP [1], ALIGN [2], and Flamingo [3] have pushed the boundaries by learning robust joint representations that bridge the semantic gap between vision and language. These advancements have enabled impressive performance across a range of tasks, including image captioning [47], visual question answering [48], text-to-image synthesis [49], and cross-modal retrieval [50], often exhibiting strong zero-shot and few-shot learning capabilities. For further information on vision-language models, we refer the reader to the recent survey papers [37], [51].

- B. Zero-Shot Learning

D. Source-Free Domain Adaptation

Source-free domain adaptation (SFDA) addresses the practical setting where access to source data is restricted during adaptation [70]. SHOT [71] initiates this paradigm by aligning target features through information maximization and selfsupervised learning. The following works improve class-wise feature structure via neighborhood clustering [72] and contrastive learning [73]. Other approaches leverage prototype refinement [74], self-training [75], and adversarial learning [76] to enhance robustness. SFDA is closely related to unsupervised VLM adaptation, as the original source data used to pretrain VLMs is often massive and impractical to access during

Zero-shot learning (ZSL) aims to recognize unseen classes by leveraging semantic information such as attributes or word embeddings. Early methods relied on learning compatibility functions between visual features and manually defined attributes [52], [53]. Subsequent works introduced embeddingbased approaches that align visual and semantic spaces using

Unsupervised VLM Adaptation

Test-Time Adaptation

| | | |
|---|---|---|
| | | |

Episodic Test-Time Adaptation (Sec. VI)

Online Test-Time Adaptation (Sec. VII)

Unsupervised Domain Transfer (Sec. V)

Data-Free Transfer (Sec. IV)

- • Entropy Minimization
- • Feedback Signal
- • Distribution Alignment
- • Self-Supervised Learning

- • Pseudo-Labeling
- • Memory Mechanisms
- • Distribution Modeling

- • Text Augmentation
- • Image Utilization
- • Network Modification

- • Self-Training
- • Entropy Optimization
- • External Resource Utilization

Fig. 2: Taxonomy of unsupervised adaptation paradigms for vision-language models (VLMs).

adaptation. For further information on SFDA, we refer the reader to the recent survey papers [77], [78].

where cos(·,·) measures the cosine similarity and τ is a temperature parameter.

Prompt Tuning. Instead of relying on manually crafted prompts, prompt tuning methods optimize prompts to improve performance on downstream tasks. Specifically, prompt tuning learns a prompt p = [V ]1[V ]2...[V ]M ∈ RM×d in the text embedding space, where M is the number of tokens and d is the embedding size. Given training data Dtrain = {(xi,yi)} from the downstream task, the objective is to generate text inputs of the form “[V ]1[V ]2...[V ]M[CLASS]“ that provide the model with the most relevant context information. For image classification with cross-entropy loss CE, this optimization can be formulated as:

E. Traditional Test-Time Adaptation

Test-time adaptation (TTA) focuses on adapting a pretrained source model online to address distribution shifts without access to source data or target labels. Online TTA methods [79], [80] update specific model parameters using incoming test samples, leveraging unsupervised objectives such as entropy minimization and pseudo-labeling. Robust TTA methods [81], [82] tackle challenging real-world scenarios, including label shifts, single-sample adaptation, and mixed domain shifts. Meanwhile, continual TTA approaches [83], [84] handle evolving distribution shifts encountered over time, which is particularly relevant in dynamic real-world applications. Although most traditional TTA methods were introduced for vision-only architectures, their core mechanisms, such as entropy minimization and pseudo-labeling, have been repurposed for TTA of VLMs [18], [24], [85]. For a comprehensive review of test-time adaptation, we refer readers to the recent survey papers [40], [41].

p∗ = argmin

CE(p(y|x),y). (2)

E(x,y)∼D

train

p

Taxonomy. In this survey, we introduce a taxonomy that systematically categorizes unsupervised VLM adaptation methods based on the availability of unlabeled visual data during the adaptation process (Fig. 1). The proposed framework defines four distinct adaptation paradigms, each with unique assumptions and challenges. The first, data-free transfer, represents the most constrained setting where no visual data from the downstream task is available. In contrast, unsupervised domain transfer assumes access to an abundant, static collection of unlabeled target data, enabling a more comprehensive, offline adaptation before inference. The final two categories address adaptation that occurs during the testing phase itself. Episodic test-time adaptation operates on a small batch of test instances, adapting the model specifically for that batch. Lastly, online test-time adaptation tackles the most dynamic scenario, where the model must continuously learn from a sequential stream of incoming data points and update itself in realtime. This taxonomy highlights the fundamental differences in data access, computational constraints, and algorithmic design across the spectrum of unsupervised VLM adaptation. In the following sections, we provide a detailed overview of existing approaches within each paradigm.

III. PRELIMINARIES Vision-Language Models (VLMs) typically consist of an image encoder that maps high-dimensional images into a low-dimensional embedding space and a text encoder that generates text representations from natural language. Since the introduction of CLIP [1], numerous improved models have been proposed, including ALIGN [2], EVA-CLIP [86], and SigLIP [87], with CLIP remaining the most widely used model in existing works. CLIP is trained on 400 million image-text pairs and aligns image and text embeddings using contrastive loss. Given a batch of image-text pairs, CLIP maximizes the cosine similarity for matched pairs while minimizing it for unmatched ones. During inference, the class names of a target dataset are embedded using the text encoder with a prompt of the form “a photo of a [CLASS]“, where [CLASS] is replaced with specific class names (e.g., cat, dog, car). The text encoder then generates text embeddings tc for each class c, and the prediction probability for an input image x with embedding fx is computed as:

IV. DATA-FREE TRANSFER

Paradigm description. Data-free transfer in the context of VLMs refers to adapting pre-trained models to downstream tasks without access to any visual data (e.g., images) from the downstream task. This setting is particularly challenging, as it

exp(cos(fx,ty)/τ) C c=1 exp(cos(fx,tc)/τ)

, (1)

p(y|x) =

| | |xternal dataset| | |
|---|---|---|---|---|

MatMul

MatMul

MatMul

MatMul

MatMul

MatMul

Rich descriptions Subclass names Class hierarchy

[Figure 10]

[Figure 11]

[Figure 12]

|Class<br><br>Names<br><br>Class<br><br>Names<br><br>Class<br><br>Names|qu|ery|(|e C|(e.g.( -3,-<br><br>LLMs<br><br>.g., GPT-3, hatGPT)| |
|---|---|---|---|---|---|---|
| | | | | | | |

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

| | |Retrieval| |
|---|---|---|---|
| | | | |

softmax

softmax

softmax

softmax

softmax

softmax

Class Names

Class Names

Class Names

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| | |eneration| |
|---|---|---|---|
| | | | |

###### I

###### I

###### I

[Figure 25]

[Figure 26]

[Figure 27]

MatMul

MatMul

MatMul

MatMul

MatMul

MatMul

###### Support set

|G|e|nerative models| | |
|---|---|---|---|---|

T

T

T

Class Names

Class Names

Class Names

T

T

T

T

T

T

text features

text features

text features

text features

text features

text features

text features

text features

text features

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

prediction

prediction

prediction

prediction

prediction

prediction

[Figure 34]

[Figure 35]

[Figure 36]

I

I

I

I

I

I

I

I

I

| | | | | | |
|---|---|---|---|---|---|

image feature

image feature

image feature

image feature

image feature

image feature

image feature prediction

prediction

prediction

image feature

image feature

Inference Stage

Inference Stage

Inference Stage

Inference Stage

Inference Stage

Inference Stage

Inference Stage

Inference Stage

Inference Stage

(a) Text augmentation.

(c) Network modification. Fig. 3: Three representative strategies of the data-free transfer paradigm.

(b) Image utilization.

TABLE I: Popular strategies along with their representative works of data-free transfer.

Strategies Representative Works Text Augmentation

DCLIP [15], CuPL [21], CHiLS [88], TaI [89].

ReCo [28], SuS-X [90], Priming [91], GenCL [92].

Image Utilization

MaskCLIP [16], CALIP [93], SCLIP [94], ProxyCLIP [95].

Network Modification

relies exclusively on textual category names to guide the adaptation process. As such, data-free transfer is considered the most difficult paradigm within unsupervised VLM adaptation. Despite these difficulties, methods developed for this setting are often highly generalizable and broadly applicable, offering robust solutions for a variety of unsupervised tasks across domains where visual data is scarce, sensitive, or unavailable.

We review existing data-free transfer methods and categorize their strategies into three primary approaches: text augmentation, image utilization, and network modification. These categories are summarized in Table I, and we introduce each strategy in detail along with related methods in the following subsections.

- A. Text Augmentation

In data-free transfer paradigm, where only class names are available, direct inference results in the rich semantic capacity of the text encoder remaining underexploited. To mitigate this limitation, several methods have been proposed that enhance the textual input through text augmentation, aiming to generate more informative representations, as shown in Fig. 3 (a). These augmented texts help unlock the latent knowledge of the text encoder, thereby improving model performance despite the absence of visual data.

Leveraging the powerful capabilities of LLMs, such as GPT-3 [96], several data-free transfer methods [15], [21], [97] have adopted text augmentation strategies to enrich class representations with more informative and discriminative descriptions. These approaches aim to replace simple class names

with richer textual content, thereby improving alignment with visual concepts. For instance, DCLIP [15] and CuPL [21] use GPT-3 to generate multiple semantic descriptors and full descriptive sentences, injecting discriminative knowledge into category representations. REAL-Prompt [98] identifies performance drops for categories with low-frequency terms in the pretraining corpus and addresses this by prompting ChatGPT [99] to substitute them with higher-frequency synonyms that are more familiar to the encoder. MPVR [97] introduces a two-step prompting mechanism, where LLMs first generate task-relevant queries that are then used to derive categoryspecific prompts, improving both relevance and diversity. In domain-specific applications like medical image diagnosis, ChatGPT [99] is used to generate symptom-based descriptions of disease classes [31]. Moreover, Parashar et al. [100] show that replacing scientific species names with common English terms improves classification performance. Interestingly, reliance on LLMs is not strictly necessary. WaffleCLIP [101] demonstrates that even random word augmentations to class names can yield results comparable to those generated with LLMs. In a complementary direction, TAG [102] proposes an out-of-distribution (OOD) detection approach that leverages a scoring mechanism based on permuted prompt templates. This score captures consistency of model predictions across varied phrasings and enables more accurate detection.

Extending this line of work, existing studies [32], [88], [103], [104] have also explored the use of subclass names as an alternative to rich textual descriptions, aiming to describe the semantic scope of each category more precisely. CHiLS [88] leverages GPT-3 [96] to generate subclass names for each original class and makes final predictions by aggregating similarities between the image and both the superclass and its associated subclasses. In the context of semantic segmentation, subclass prompts allow for finer-grained, patch-level alignment with the target superclass, leading to measurable improvements in performance [103]. For video-based action recognition, TEAR [32] decomposes complex actions into multiple subactions, generating concise descriptions for each and forming a robust composite representation by averaging their features. Beyond subclassing, other approaches enrich semantic understanding through category-related attributes, helping to capture

nuanced intra-class variation and enhance image recognition accuracy [104]. Furthermore, EOE [105] extends this strategy to OOD detection by prompting LLMs to generate potential OOD category names, thereby broadening the model’s recognition capacity beyond the training distribution.

Rather than classifying objects across all possible categories, some approaches [106]–[108] simplify the task by organizing classes into clusters or hierarchical structures, decomposing a complex classification problem into a sequence of hierarchical sub-tasks. An early work [106] constructs hierarchical clusters of candidate categories and employs ChatGPT [99] to generate group-specific, discriminative textual descriptions. More recently, Lee et al. [107] leverage textual feature similarity to identify semantically similar classes and then prompt an LLM to generate visual descriptors that distinguish a class from its nearest semantic neighbors. Moving beyond accuracy alone, HAPrompts [108] introduces a hierarchical classification framework to encourage the model to promote better mistakes, encouraging the model to predict semantically related labels when misclassifications occur.

Another line of data-free transfer methods [89], [109], [110] leverage external textual data as a training signal to guide models toward more robust task performance. For example, TaI [89] replaces annotated images with rich textual descriptions for prompt tuning, introducing dual-grained prompts that capture both global semantic context and local discriminative features. In a related approach, TAP [109] constructs classspecific textual descriptions and trains a text-only classifier using cross-entropy loss. This classifier is then integrated with a visual encoder at inference time to enhance recognition accuracy. Going a step further, ProText [110] optimizes deep prompt parameters to steer the text encoder toward extracting meaningful representations from LLM-generated descriptions, which embed extensive linguistic knowledge and fine-grained conceptual distinctions.

- B. Image Utilization

In the absence of visual data, methods that rely solely on textual information face inherent limitations, primarily due to the modality gap that exists within VLMs. To bridge this gap, a growing body of research [28], [91], [111] introduces visual signals by either retrieving relevant images from external datasets or synthesizing them using generative models, as shown in Fig. 3 (b).

Retrieval-based methods attempt to provide visual grounding by leveraging large-scale unlabeled datasets. For example, ReCo [28] retrieves semantically relevant images using CLIP [1] from an external corpus and computes a reference image embedding for each category. This embedding is then used to guide the recognition of corresponding image patches and refine the original zero-shot dense predictions. Neural Priming [91] introduces a new classification head by computing the centroids of retrieved image sets for each category from the pre-training dataset, subsequently integrating this head with the original zero-shot classifier to enhance recognition. Generative approaches extend this direction by synthesizing visual data to simulate examples for downstream tasks. For

instance, Shipard et al. [111] construct a synthetic training set using diffusion models [49], generating a diverse set of images that provide rich visual cues in the absence of real data. AttrSyn [112] further boosts image diversity by leveraging LLMs to generate a wide range of attributes, which guide the generative model to produce class-consistent and discriminative samples. SuS-X [90] combines both retrieval and generation strategies by introducing a visual support set, either constructed from large-scale datasets or generated via advanced diffusion models [49]. This support set enables information integration and provides auxiliary supervision during inference. Finally, in the context of continual learning, GenCL [92] generates synthetic images for novel classes using prompt-guided diffusion models, and then introduces an ensemble-based selector to curate a representative coreset from the generated samples, supporting robust and effective category representation over time.

C. Network Modification

Several data-free methods [16], [29], [93], [113] focus on modifying the network architecture of VLMs to enhance their suitability for downstream tasks, particularly dense prediction such as segmentation, as shown in Fig. 3 (c). While these approaches are primarily developed for classification-oriented VLMs, their architectural enhancements significantly improve dense prediction performance.

A pioneering work, MaskCLIP [16], demonstrates that value embeddings in the final attention layers capture richer local information than global features, making them particularly effective for segmentation tasks. To further refine dense predictions from value embeddings, MaskCLIP introduces a keybased smoothing strategy and a denoising technique. Building on this, CALIP [93] facilitates interaction between visual and textual features through a parameter-free attention module, and achieves improved classification results by ensembling outputs from multiple feature representations. CLIP Surgery [114] advances segmentation by introducing value-value attention to enhance local feature consistency and employing a feature surgery strategy to suppress noisy activations, thus improving both segmentation accuracy and interpretability. GEM [115] generalizes value-value attention to an any-any attention mechanism, enhancing consistency across groups of similar tokens by ensembling outputs from the modified attention applied to key, query, and value embeddings at every transformer layer. SCLIP [94] introduces a correlative selfattention mechanism, which yields spatially covariant features that better preserve fine local details. Finally, ProxyCLIP [95] presents a training-free framework that enhances CLIP’s openvocabulary segmentation by integrating spatially consistent proxy attention maps generated from vision foundation models such as DINO [116] and SAM [117].

V. UNSUPERVISED DOMAIN TRANSFER

Paradigm description. Unsupervised domain transfer for VLMs refers to the adaptation of pre-trained models to downstream tasks with abundant unlabeled data. Compared with data-free transfer, unsupervised domain transfer can use

Pseudo-labeling

Pseudo-

-

###### Entropy Optimization

Class

Class

Class

Class

Class

Class

Class

Class

Class

##### T+

##### T+

##### T+

##### I+

##### I+

##### I+

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

Names T

Names T

Names T

Names T

Names T

Names T

Names T

Names T

Names T

𝑪

𝑪

𝑪

###### 𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = ෍

𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = ෍

𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = ෍

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

reference prediction 𝒑ෝ

𝒑ෝ

𝒑ෝ

𝒄

𝒄

𝒄

𝑪

𝑪

𝑪

LLMs

LLMs

𝒎𝒂𝒙 ℒ𝒄𝒍𝒂𝒔𝒔𝒆𝒏𝒕 = ෍

𝒎𝒂𝒙 ℒ𝒄𝒍𝒂𝒔𝒔𝒆𝒏𝒕 = ෍

𝒎𝒂𝒙 ℒ𝒄𝒍𝒂𝒔𝒔𝒆𝒏𝒕 = ෍

𝒑ഥ𝒄𝒍𝒐𝒈𝒑ഥ𝒄

𝒑ഥ𝒄𝒍𝒐𝒈𝒑ഥ𝒄

𝒑ഥ𝒄𝒍𝒐𝒈𝒑ഥ𝒄

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

𝒄

𝒄

𝒄

ℒ𝑺𝑻

ℒ𝑺𝑻

ℒ𝑺𝑻

ℒ𝑲𝑫

ℒ𝑲𝑫

ℒ𝑲𝑫

MLLMs

MLLMs

pseudo label 𝒚ෝ = 𝒂𝒓𝒈𝒎𝒂𝒙 𝒑ෝ

𝒚ෝ = 𝒑ෝ

𝒚ෝ = 𝒑ෝ

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

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

ℒ𝑺𝑻

ℒ𝑺𝑻

ℒ𝑺𝑻

| | | | | | |
|---|---|---|---|---|---|

##### I

##### I

##### I

##### I

##### I

##### I

##### I

##### I

##### I

prediction 𝒑

prediction 𝒑

prediction 𝒑

prediction

abundant prediction 𝒑 unlabeled data

prediction 𝒑

prediction 𝒑

abundant unlabeled data

abundant unlabeled data

abundant unlabeled data

abundant unlabeled data

abundant unlabeled data

abundant unlabeled data

data

data

(a) Self-training.

(c) External resource utilization. Fig. 4: Three representative strategies of the unsupervised domain transfer paradigm.

(b) Entropy optimization.

- TABLE II: Popular strategies along with their representative works of unsupervised domain transfer.

works [19], [120], [128], [129]. SwapPrompt [19] extends this self-training strategy with another swapped prediction mechanism, letting the two augmented views of the same image provide soft pseudo-label optimization supervision through an EMA-updated prompt for each other. RS-CLIP [130] introduces a curriculum learning framework, beginning with a small subset of high-confidence samples for self-training and progressively incorporating more data as optimization progresses, thus mitigating the noise from early-stage pseudo labels. GTA-CLIP [131] proposes a transductive inference approach to pseudo-labeling, using iteratively refined, attributeaugmented similarities between image and text embeddings to improve label quality. In a related approach, CPL [132] refines candidate pseudo labels by introducing both intra- and inter-instance label to reduce the negative impact of incorrect hard pseudo labels typically produced by VLMs. Another work [133] systematically investigates pseudo-labeling strategies across several unsupervised settings and demonstrates the effectiveness of pseudo-labeling in promoting more balanced and robust performance across diverse categories.

Strategies Representative Works Self-Training

UPL [17], LaFTer [118], MUST [26], ReCLIP [119].

POUF [34], CDBN [120], UEO [27], TFUP-T [121].

Entropy Optimization

Neural Priming [91], PEST [122], PromptKD [123], OTFusion [124].

External Resource Utilization

FARE [125], OTTER [126] ZLaP [35], TransCLIP [127].

Miscellaneous

the unlabeled data of downstream tasks to better grasp the data distribution and thus achieve better performance. The challenges of this paradigm mainly come from the filtering and processing of unlabeled data and the unsupervised alignment of VLM and unlabeled data.

We review existing unsupervised domain transfer methods and categorize their strategies into three primary approaches: self-training, entropy optimization, and external resource utilization. These categories are summarized in Table II, and we introduce each strategy in detail along with related methods in the following subsections.

Inspired by FixMatch [134], several unsupervised domain transfer methods [118], [120] apply both weak and strong augmentations to unlabeled data to enhance consistency learning in VLMs. These approaches typically generate pseudo labels using the weakly augmented views and consider them as supervisory signals for self-training on the strongly augmented ones. LaFTer [118] leverages large language models (LLMs) to generate diverse textual data for training a text classifier, which in turn produces high-quality pseudo labels with weak augmented views for effective self-training. MedUnA [30] proposes a dual-branch architecture, consisting of weak and strong branches for the visual encoder, and jointly optimizes them using a pseudo-labeling objective to enhance medical image classification. NoLA [129] employs a DINO-based labeling network fed with weak augmentations to improve pseudo-label quality for training visual prompts. DPA [135] introduces dual prototype representations for both visual and textual branches, integrating their outputs to generate more robust pseudo labels. Additionally, LP-CLIP [136] incorporates confidence estimates into the pseudo-labeling objective, thereby improving both classification accuracy and calibration.

- A. Self-Training

Self-training is a widely used strategy in unsupervised learning, as shown in Fig. 4 (a), where the ground truth of the training data is absent. Using this approach, unsupervised algorithms always manage to calculate high-quality pseudo labels on unlabeled samples as supervision signals. How to obtain and iteratively refine pseudo labels to make VLM better adapts to the distribution of the unlabeled data is the key challenge for those methods.

UPL [17] is one of the earliest efforts to explore unsupervised domain transfer for VLMs. It selects a small set of highconfidence unlabeled samples for each category and optimizes the prompt parameters using a pseudo-labeling strategy,

argmin

selectLCE(p(x),yˆ), (3)

E(x,yˆ)∼D

p

Rather than filtering high-confidence samples and enhancing consistency between different augmentations, there are also some methods that generate pseudo labels in other ways.

where Dselect represents the selected high-confidence samples and LCE denotes the cross-entropy loss. This selective pseudolabeling approach is later adopted by a number of subsequent

MUST [26] maintains an EMA model to produce high-qulity pseudo labels and incorporates a masked image modeling strategy to improve local image representation learning. PEST [122] enhances pseudo-label quality by ensembling predictions from multiple textual prompts and visually augmented views. ReCLIP [119] learns a projection space to better align visual and textual features and employs self-training with pseudo labels refined using Label Propagation [137]. NtUA [138] constructs a confidence-weighted key-value cache of pseudolabeled features and refines it through knowledge distillation, effectively mitigating label noise in scenarios with limited unlabeled data. Similarly, TFUP-T [121] improves pseudolabel quality by building a cache model with representative samples and refining predictions based on both feature-level and semantic-level similarities. To address the issue of lowconfidence pseudo labels, FST-CBDG [139] employs soft pseudo labels and updates them using a moving average strategy during self-training. For regression tasks, CLIPPR [22] trains an adapter for the image encoder using zeroshot pseudo labels, optimizing performance by minimizing the distance between predicted and prior label distributions.

- B. Entropy Optimization

Entropy optimization is a classic unsupervised learning objective, encouraging the model to make confident predictions on unlabeled data, as shown in Fig. 4 (b). Unlike self-training, entropy optimization is not affected by erroneous pseudolabels and behaves more stably on low-performance tasks. Many algorithms minimize sample-level entropy, adapting the model to the unlabeled data distribution [34], [120], and also maximize category-level marginal entropy to avoid mode collapse [34], [120], [121].

POUF [34] and CDBN [120] optimize textual prompt parameters with sample-level entropy minimization and category-level marginal entropy maximization. An optimal transport objective is also incorporated into POUF for better alignment between the distribution of textual prototypes and the unlabeled data. In order to improve both generalization and out-of-distribution detection performance of VLMs, UEO [27] proposes universal entropy, utilizing marginal prediction instead of sample prediction for entropy maximization to stabilize the optimization process.

- C. External Resource Utilization

Several recent approaches enhance the performance of VLMs by incorporating external resources beyond the available unlabeled data. These resources often include retrievalbased image augmentation, introduction of (multimodal) large language models (MLLMs), and knowledge distillation from powerful VLMs or vision models, as shown in Fig. 4 (c).

Neural Priming [91] adopts a transductive learning paradigm by constructing a retrieval set of images based on category names. For each unlabeled sample, it selects the most visually similar images to form a fine-tuning dataset, thereby adapting the VLM to the target domain. LaFTer [118] leverages GPT-3 [96] to generate diverse textual descriptions, which are then used to train a text classifier tailored for the downstream task.

Similarly, PEST [122] and GTA-CLIP [131] query LLMs such as GPT-3 [96] and LLaMA [140] to create multiple prompts per class. These prompts improve pseudo-label quality through ensemble-based prompt inference. LatteCLIP [141] utilizes LLaVA [4] to generate image captions, which support more accurate textual prototype construction for VLM adaptation.

In the context of knowledge distillation, PromptKD [123] reuses textual features from a larger teacher VLM to guide the training of an image encoder, thereby transferring semantic knowledge. Going a step further, KDPL [142] jointly optimizes prompts in both visual and textual input spaces, balancing performance and efficiency. NtUA [138] improves pseudo-label reliability by incorporating the image encoder of a stronger VLM, enhancing both label quality and confidence estimation. OTFusion [124] aligns VLMs’ embedding with features extracted from powerful vision models (e.g., DINO) via optimal transport to obtain refined predictions.

D. Miscellaneous

There are also several recent approaches that address unsupervised domain transfer for VLMs through diverse strategies [125], [143]. ZPE [144] designs a prompt ensembling strategy that leverages unlabeled data to address frequency biases in words and concepts and assigns appropriate ensembling weights to multiple prompt templates. uCAP [145] formulates image generation as a function of class names and latent, domain-specific prompts. It employs an energy-based likelihood framework to infer optimal prompts from unlabeled data. To enhance adversarial robustness while preserving performance on clean inputs, FARE [125] optimizes the vision encoder to align the features of adversarially perturbed images with those of their clean counterparts as computed by the original VLM. OTTER [126] addresses label distribution mismatch by leveraging optimal transport to align model predictions with an estimated label distribution in the target domain. Moreover, InMaP [146] learns class proxies directly in the vision space using refined pseudo labels derived from text embeddings, thereby narrowing the modality gap between visual and textual representations in VLMs. Subsequent methods exploit various strategies for modeling vision-text features, including label propagation [35] and Dirichlet distributions [147]. TransCLIP [127] proposes a plug-and-play transductive framework that optimizes a KL-regularized objective with an efficient block Majorize-Minimize algorithm, integrating the text-encoder knowledge together.

VI. EPISODIC TEST-TIME ADAPTATION

Paradigm description. Episodic test-time adaptation is a popular learning paradigm in which a pre-trained VLM is adapted at inference time using a single batch of unlabeled test data. The goal is to leverage the knowledge embedded in the pre-trained VLM to accurately predict labels for the current batch, without requiring access to multiple test batches or labeled data during adaptation.

We review existing episodic test-time adaptation methods and categorize their strategies into four primary approaches: entropy minimization, feedback signal, distribution alignment,

Online Computed Batch Data Statistics

Online Computed Batch Data Statistics

[Figure 73]

Online Computed Batch Data Statistics

Class

Class

[Figure 74]

Class

[Figure 75]

Class

[Figure 76]

Names T

[Figure 77]

###### T

[Figure 78]

Names T

###### I

[Figure 79]

###### T

Class

Class

###### I

[Figure 80]

𝜇1,𝜎12 𝜇2,𝜎22 𝜇3,𝜎32

Diffusion Loss

[Figure 81]

Names T

𝜇1,𝜎12 𝜇2,𝜎22 𝜇3,𝜎32

Diffusion Loss

###### T

###### I

###### Entropy Minimization

###### Entropy Minimization

Names

𝜇1,𝜎12 𝜇2,𝜎22 𝜇3,𝜎32

Diffusion Loss

Names

###### Entropy Minimization

Names

𝑪

learnable

𝑪

learnable

###### ＋

𝑪

learnable

###### ＋

###### 𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = −෍

###### ＋

###### 𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = −෍

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

prompt

###### 𝒎𝒊𝒏 ℒ𝒔𝒂𝒎𝒑𝒍𝒆𝒆𝒏𝒕 = −෍

prompt

batch-level unlabeled data

𝒑𝒄𝒍𝒐𝒈𝒑𝒄

batch-level unlabeled data

prompt

[Figure 82]

[Figure 83]

[Figure 84]

a batch of unlabeled data

[Figure 85]

[Figure 86]

[Figure 87]

𝒄

𝒄

𝒄

Align

Align

Align

diffusion model

diffusion model

diffusion model

[Figure 88]

[Figure 89]

Aug.

[Figure 90]

Aug.

condition

condition

Aug.

condition

[Figure 91]

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

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

… I

[Figure 102]

… I

[Figure 103]

###### I

[Figure 104]

###### I

… I

[Figure 105]

###### I

𝜇Ƹ1,𝜎ො12 𝜇Ƹ2,𝜎ො22 𝜇Ƹ3,𝜎ො32

| | | | | | |
|---|---|---|---|---|---|

𝜇Ƹ1,𝜎ො12 𝜇Ƹ2,𝜎ො22 𝜇Ƹ3,𝜎ො32

𝜇Ƹ1,𝜎ො12 𝜇Ƹ2,𝜎ො22 𝜇Ƹ3,𝜎ො32

[Figure 106]

[Figure 107]

[Figure 108]

prediction 𝒑

prediction 𝒑

prediction 𝒑

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

prediction 𝒑

prediction 𝒑

prediction 𝒑

[Figure 115]

[Figure 116]

[Figure 117]

Offline Computed Source Data Statistics

Offline Computed Source Data Statistics

Offline Computed Source Data Statistics

a batch of unlabeled data

batch-level unlabeled data

a batch of unlabeled data

batch-level unlabeled data

batch-level unlabeled data

batch-level unlabeled data

(a) Entropy minimization.

(c) Distribution alignment. Fig. 5: Three representative strategies of the episodic test-time adaptation paradigm.

(b) Feedback signal.

(2) Feedback Signal

(1) Entropy Minimization

(2) Feedback Signal

(2) Feedback Signal

(1) Entropy Minimization

(1) Entropy Minimization

(3) Distribution Alignment

(3) Distribution Alignment

(3) Distribution Alignment

- TABLE III: Popular strategies along with their representative works of episodic test-time adaptation.

mask I[H(pi) ≤ η], where H denotes the entropy of predictions on augmented samples. Instead of applying random augmentations as in TPT, DiffTPT [85], [162] leverages pretrained diffusion models to generate diverse augmentations and employs cosine similarity-based filtering to remove spurious samples. R-TPT [148] employs a reliability-based weighted ensembling strategy to aggregate information from trustworthy augmented views of the test sample. C-TPT [163] optimizes prompts by maximizing text feature dispersion, observing that better-calibrated predictions correlate with higher text feature dispersion. O-TPT [164] improves calibration by enforcing orthogonality constraints on class-specific textual prompt features during tuning to maximize their angular separation. Furthermore, DTS-TPT [33] extends TPT to video data for zero-shot activity recognition.

Strategies Representative Works

TPT [18], DiffTPT [85], R-TPT [148], DTS-TPT [33].

Entropy Minimization

Diffusion-TTA [149], RLCF [150], BPRE [151] .

Feedback Signal

PromptAlign [24], MTA [36], TAPT [152], StatA [153].

Distribution Alignment

Self-TPT [154], InCPL [155], LoRA-TTT [156], T3AL [157].

Self-Supervised Learning

AWT [158], RA-TTA [159], SCAP [160], ZERO [161].

Miscellaneous

Beyond tuning text prompts to minimize entropy, several works also explore visual prompts [165], multimodal prompts [24], low-rank attention weights [166], and learnable noise [167]. For example, PromptAlign [24] uses multimodal prompt learning to align image token distributions between a pre-computed source proxy dataset and test samples. TTL [166] adapts low-rank attention weights during test time through a confidence maximization objective, enabling efficient adaptation without altering prompts or backbone parameters.

and self-supervised learning. These categories are summarized in Table III, and we introduce each strategy in detail along with related methods in the following subsections.

A. Entropy Minimization

Entropy minimization is a widely adopted strategy for TTA [40] by adjusting the model’s parameters to make its output predictions more confident with lower entropy, as shown in Fig. 5 (a). This process encourages the model to produce low-uncertainty outputs for the test data, often improving its performance under distribution shifts.

B. Feedback Signal

Shu et al. [18] introduced test-time prompt tuning (TPT) as the first method for adapting pre-trained VLMs at test time. TPT optimizes a text prompt p = [V ]1[V ]2...[V ]M for each test sample using entropy minimization, combined with confidence selection to ensure consistent predictions across augmented views. Specifically, TPT generates N randomly augmented views of a test image x using a set of random augmentations A and minimizes the entropy of the averaged prediction probability distribution:

Some studies explore leveraging feedback signals from diffusion [149] or CLIP-like models [150], [151] for TTA, as shown in Fig. 5 (b). For example, Diffusion-TTA [149] leverages generative feedback from diffusion models to adapt pre-trained discriminative models at test time by optimizing image likelihood, significantly improving performance across tasks like classification, segmentation, and depth prediction. Diffusion-TTA consists of discriminative and generative modules. Given an image x, the discriminative model fθ predicts task output y (Eq. (1) for VLMs). The task output y is transformed into condition c. For image classification, y represents a probability distribution over C categories, y ∈ [0,1]C,y⊤1C = 1. Given the learned text embeddings of a text-conditional diffusion model for the C categories tj ∈ Rd,j ∈ {1...C}, the diffusion condition is c =

C

p∗ = argmin

p˜p(yi|x)log p˜p(yi|x), (4)

−

p

i=1

N

1 ρN

I[H(pi) ≤ η]pp(y|Ai(x)). (5)

p˜p(yi|x) =

i=1

Here, pp(y|Ai(x)) represents the class probability vector for the i-th augmented view of x under prompt p. TPT selects ρpercentile confident samples with a prediction entropy below a threshold η to filter out noisy predictions, using a confidence

C j=1 yj ·tj. Finally, the generative diffusion model ϵϕ is used to measure the likelihood of the input image, conditioned on c. This consists of using the diffusion model ϵϕ to predict

the added noise ϵ from the noisy image xt and condition c. The image likelihood is maximized using diffusion loss by updating the discriminative and generative model weights via backpropagation:

###### Ldiff = Et,ϵ∥ϵϕ(√α¯tx + √1 − α¯tϵ,c,t) − ϵ∥2, (6)

where α¯t defines how much noise is added at each time step t. Differently, RLCF [150] utilizes CLIP-based feedback through reinforcement learning and employs CLIPScore [168] as a reward signal to provide feedback for VLMs. BPRE [151] mitigates text-conditioned bias by using a quality-aware reward module based on intrinsic visual features, forming a selfevolving feedback loop with prototype refinement to enhance adaptation to distribution shifts.

C. Distribution Alignment

Distribution alignment methods align test sample distributions with known source characteristics or refine representations for improved consistency [169], as shown in Fig. 5 (c). For example, PromptAlign [24] bridges the source-totarget distribution gap by jointly updating multimodal prompts to align the per-layer image-token statistics of augmented test views with offline-computed source statistics through a combined alignment and entropy minimization loss. To enhance adversarial robustness, TAPT [152] adapts statistical alignment by using a loss function that, at inference, aligns the augmented visual embeddings of a test sample with precomputed statistics from both clean and adversarially perturbed images. StatA [153] preserves text encoder knowledge during adaptation by utilizing statistical anchors that penalize deviations from text-derived Gaussian priors.

Complementary to these approaches, MTA [36] employs a robust MeanShift algorithm to identify density modes in the feature space while concurrently optimizing them with an inlierness score to automatically assess each view’s quality. Beyond global distribution alignment, several methods focus on class-aware prototype alignment. PromptSync [170] performs class-aware prototype alignment of the test sample with source class prototypes, weighted by mean class probabilities derived from confident augmented views. Also utilizing class prototypes, TPS [169] pre-computes class prototypes and then, for each test sample, dynamically learns shift vectors to adjust these prototypes directly within the shared embedding space.

- D. Self-Supervised Learning

Self-supervised learning [171], [172] is a powerful technique for learning transferable representations. Self-TPT [154] introduces contrastive prompt tuning as a self-supervised learning strategy, which aims to minimize intra-class distances while maximizing inter-class separation by leveraging contrastive learning principles. Specifically, for each class token, multiple prompt variations are generated by altering the insertion point of the class token (e.g., beginning, middle, or end of the prompt sequence). This creates positive pairs from the same class and negative pairs from different classes,

encouraging the model to learn more robust class representations. The contrastive loss is formulated as:

exp tiτ·tj

4C

log j∈P(i)

, (7)

L = −

4C j=1,j̸=i exp tiτ·tj

i=1

where ti and tj are the projected text features of different views, P(i) denotes the set of positive samples for view i, and τ is a temperature parameter. In contrast, LoRA-TTT [156] updates only the low-rank parameters in the image encoder using a memory-efficient reconstruction loss, computed as the mean squared error of class tokens from top-confidence augmented and masked views, to enhance global feature understanding. In addition, InCPL [155] enables efficient model adaptation by optimizing visual prompts from a few labeled examples through a context-aware unsupervised loss and a cyclic learning strategy. T3AL [157] generates and refines temporal action proposals by first deriving video-level pseudolabels from a pretrained VLM, then using a self-supervised method to create initial proposals, and finally enhancing them with frame-level textual descriptions.

E. Miscellaneous

Beyond the previously discussed approaches, additional techniques have been developed for episodic test-time adaptation of VLMs [173]–[180]. One line of work uses retrieval-based strategies [159], [181], [182]. For instance, XMoRe [181] retrieves relevant captions via a two-step crossmodal retrieval and ensembles image and text predictions using dynamically weighted modal-confidence scores. RATTA [159] utilizes fine-grained text descriptions to guide a two-step retrieval of relevant external images, which are then used in a description-based adaptation process to refine the model’s initial prediction. Another line of work tries to improve the calibration of VLMs [161], [163], [164], [183]. Besides retrieval and calibration, other representative work includes optimal transport [158], spurious features erasing [184], loss landscape [185], counterattack [186], and supportive cliques [160]. Several methods focus on improving CLIP’s dense prediction abilities for open-vocabulary semantic segmentation by addressing its image-level pre-training limitations [28], [29], [103], [115], [187]–[189]. The external knowledge, such as MLLMs and LLMs, can also be used during inference, without requiring additional training or finetuning on task-specific data [190]–[197].

VII. ONLINE TEST-TIME ADAPTATION

Paradigm description. Online test-time adaptation is another TTA paradigm designed for streaming data scenarios, where unlabeled data arrives sequentially in mini-batches. Given a pre-trained VLM, the objective is to adapt the model online to each incoming mini-batch in order to accurately predict its labels under potential distribution shifts. Unlike episodic adaptation, which adapts to each batch independently, online adaptation continuously updates the model by leveraging knowledge accumulated from previously observed

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Pseudo-labeling

Class Names T

[Figure 125]

Pseudo-labeling

[Figure 126]

Class Names T

Class Names

[Figure 127]

T

Pseudo-labeling

[Figure 128]

Class Names T

Class Names

[Figure 129]

T

[Figure 130]

Class Names

[Figure 131]

T

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

learnable prompt

[Figure 139]

[Figure 140]

text features

learnable prompt

learnable prompt

[Figure 141]

[Figure 142]

text features

[Figure 143]

reference prediction 𝒑ෝ

[Figure 144]

text features

reference prediction 𝒑ෝ

reference prediction 𝒑ෝ

[Figure 145]

[Figure 146]

[Figure 147]

prediction 𝒑

pseudo label

prediction 𝒑

pseudo label

prediction 𝒑

pseudo label

𝒚ෝ = 𝒂𝒓𝒈𝒎𝒂𝒙 𝒑ෝ

𝒚ෝ = 𝒂𝒓𝒈𝒎𝒂𝒙 𝒑ෝ

[Figure 148]

[Figure 149]

[Figure 150]

𝒚ෝ = 𝒂𝒓𝒈𝒎𝒂𝒙 𝒑ෝ

update

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

… … streaming unlabeled data

update

I

update

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

… … streaming unlabeled data

… … streaming unlabeled data

I

[Figure 166]

[Figure 167]

I

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

… …

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

###### I

[Figure 178]

[Figure 179]

[Figure 180]

image feature

[Figure 181]

… …

[Figure 182]

[Figure 183]

[Figure 184]

… …

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

###### I

[Figure 191]

###### I

image feature

image feature

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

streaming prediction 𝒑 unlabeled data

memory

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

streaming prediction 𝒑 unlabeled data

streaming prediction 𝒑 unlabeled data

memory

memory

(1) Pseudo-labeling (2) Memory Mechanisms

(a) Pseudo-labeling. (3)DistributionModeling

(c) Distribution(3)Distributionmodeling.Modeling Fig. 6: Three representative strategies of the online test-time adaptation paradigm.

(b) Memory mechanisms. (3) Distribution Modeling

(1) Pseudo-labeling (2) Memory Mechanisms

(1) Pseudo-labeling (2) Memory Mechanisms

- TABLE IV: Popular strategies along with their representative works of online test-time adaptation.

odic weight averaging, and text ensembling. To handle noisy target data, AdaND [215] introduces an adaptive noise detector trained with pseudo-labels from a frozen model to decouple noise detection from classification. DART [198] learns adaptive multimodal prompts (class-specific text and instance-level image) while retaining knowledge from prior test samples. ROSITA [216] employs a contrastive learning objective with dynamically updated feature banks to enhance the discriminability of OOD samples. Finally, TIPPLE [217] adopts a two-stage approach, first using online pseudo-labeling with an auxiliary text classification task and diversity regularization for task-oriented prompt learning, then refining this task-level prompt with a tunable residual for each test instance.

Strategies Representative Works

DART [198], CLIPArTT [199], CLIP-OT [200], WATT [201].

Pseudo-Labeling

TDA [25], DMN [202], DPE [203], BaFTA [204].

Memory Mechanisms

OGA [205], DOTA [206], BCA [207], DN [208].

Distribution Modeling

DynaPrompt [209], TCA [210], ECALP [211], OnZeta [23].

Miscellaneous

mini-batches. This enables more effective and efficient label prediction in dynamic, streaming environments.

B. Memory Mechanisms

We review existing online test-time adaptation methods and categorize their strategies into three primary approaches: pseudo-labeling, memory mechanisms, and distribution modeling. These categories are summarized in Table IV, and we introduce each strategy in detail along with related methods in the following subsections.

Memory-based methods leverage dynamic or static memory structures to store and retrieve feature representations and pseudo-labels from test samples, as shown in Fig. 6 (b). These methods enable progressive refinement of predictions by utilizing confident outputs and historical information, enhancing robustness and adaptability without requiring extensive retraining or backpropagation [218]–[221]. Inspired by Tip-

A. Pseudo-Labeling

- Adapter [222], Karmanov et al. [25] propose a training-free dynamic adapter (TDA) without requiring backpropagation. The core of TDA is a dynamic key-value cache system that stores pseudo-labels and corresponding feature representations from test samples. This cache enables progressive refinement of predictions by leveraging confident test-time outputs, facilitating efficient adaptation. Similarly, DMN [202] leverages static memory for training data knowledge and dynamic memory for online test feature preservation. Boost-
- Adapter [223] leverages a lightweight key-value memory to retrieve features from instance-agnostic historical samples and instance-aware boosting samples. HisTPT [224] constructs three complementary knowledge banks—local, hard-sample, and global—to preserve useful information from previously seen test samples. AdaPrompt [225] introduces a confidenceaware buffer that stores and utilizes only class-balanced, highconfidence samples to ensure the prompt updates are robust and stable.

Pseudo-labeling assigns class labels to unlabeled test samples and optimizes the cross-entropy loss between predictions and pseudo-labels to guide model adaptation, as shown in Fig. 6 (a). However, due to distribution shifts, pseudo-labels may be noisy, which can negatively impact learning. Various methods have been proposed to mitigate this issue. Many methods refine the pseudo-labeling process itself; for instance, IST [212] employs graph-based correction and non-maximum suppression for pseudo-label refinement, stabilizing updates with parameter moving averages. Others, like CLIPArTT [199] dynamically construct text prompts from top-K predicted classes to serve as pseudo-labels, while CLIP-OT [200] utilizes optimal transport for label assignment alongside multitemplate knowledge distillation. CTPT [213] focuses on iterative prompt updates guided by stable class prototypes and accurate pseudo-labels. SwapPrompt [19] proposes a dual-prompt and swapped prediction mechanism for efficient prompt adaptation.

Other works utilize dynamically evolving class prototypes to capture accurate multimodal representations during inference. By continuously updating these prototypes from unlabeled test samples, these methods enhance model adaptability, robustness, and efficiency [226], [227]. For example, DPE [203] simultaneously evolves two sets of prototypes—textual and

Several approaches enhance pseudo-labeling by integrating it with other mechanisms. For instance, SCP [214] uses self-text distillation with conjugate pseudo-labels to improve robustness and minimize overfitting. WATT [201] combines diverse text templates, pseudo-label-based updates with peri-

visual—to progressively capture accurate multimodal representations for target classes during test time. BaFTA [204] uses backpropagation-free online clustering to estimate class centroids and robustly aggregate class embeddings with visualtext alignment, guided by entropy-based reliability for improved zero-shot performance. BATCLIP [228] introduces a projection matching loss to improve alignment between visual class prototypes and text features, and a separability loss to increase the distance between these prototypes for more discriminative features.

C. Distribution Modeling

Distribution modeling methods model the distribution of visual or multimodal features, often using Gaussian estimations, to refine predictions during inference [229]–[231], as shown in Fig. 6 (c). By leveraging probabilistic frameworks and incorporating zero-shot priors, these methods enhance adaptability and robustness without requiring extensive hyperparameter tuning or backpropagation. For instance, OGA [205] models the likelihood of visual features using multivariate Gaussian distributions and incorporates zero-shot priors within a maximum a posteriori estimation framework. Similarly, DOTA [206] estimates Gaussian class distributions to compute Bayes-based posterior probabilities for adaptation, achieves fast inference without gradient backpropagation, and incorporates a human-in-the-loop mechanism to handle uncertain samples and enhance test-time performance. BCA [207] continuously updates text-based class embeddings to align likelihoods with incoming image features and concurrently refines class priors using the resulting posterior probabilities. On the other hand, DN [208] approximates negative sample information using the mean representation of test samples, enhancing alignment with the model’s optimization objective without requiring retraining or fine-tuning.

- D. Miscellaneous

Beyond the previously discussed approaches, additional techniques have been developed for online test-time adaptation of vision-language models [210], [232]–[234]. For instance, DynaPrompt [209] mitigates error accumulation in online adaptation by dynamically selecting and updating prompts per test sample based on entropy and confidence scores, while maintaining an adaptive buffer to add informative prompts and discard inactive ones. ECALP [211] performs inference without task-specific tuning by dynamically expanding a graph over text prompts, few-shot examples, and test samples, using context-aware feature re-weighting to exploit the test sample manifold without requiring additional unlabeled data. OnZeta [23] sequentially processes test images for immediate prediction without storage, using online label learning to model the target distribution and online proxy learning to bridge the image-text modality gap via class-specific vision proxies. Besides, other representative work includes support set [235], token condensation [210], prompt distillation [236], and more [237]–[240].

VIII. APPLICATIONS

- A. Object Classification

Object classification serves as a fundamental task for evaluating VLMs, where the objective is to assign a test object image to one of the candidate category names. In the context of unsupervised adaptation with VLMs, research efforts primarily focus on two aspects: fine-grained generalization and robustness to distribution shifts. To assess finegrained classification performance, commonly used benchmark datasets include Caltech101 [241], OxfordPets [242], StanfordCars [243], Flowers102 [244], Food101 [245], FGVCAircraft [246], SUN397 [247], DTD [248], EuroSAT [249] and UCF101 [250]. To evaluate robustness against distributional shifts, researchers [18], [19], [101] often employ ImageNet [251] along with its variants, such as ImageNetV2 [253], ImageNet-Sketch [255], ImageNet-A [252], and ImageNet-R [254]. Additionally, several studies [27], [34] incorporate datasets traditionally used in domain adaptation to evaluate their methods, such as Office-Home [256], Office [14], and DomainNet [257].

- B. Semantic Segmentation

Semantic segmentation aims to assign a semantic label to each pixel in an image, playing a critical role in applications such as autonomous driving and medical image analysis. Unsupervised segmentation methods based on VLMs primarily focus on general and fine-grained object segmentation benchmarks, including PASCAL VOC 2012 [258], PASCAL Context [259], COCO Stuff [260], ADE20K [261], and COCO-Object [262]. In addition, complex scene understanding datasets such as Cityscapes [263] and KITTI-STEP [264] are often employed to evaluate the performance of unsupervised segmentation approaches. To assess the ability to identify rare concepts, some methods [28] utilize FireNet [265]. Moreover, researchers [16] explore robustness to corruptions [285] to find out whether segmentation algorithms preserve the inherent robustness of VLMs. Segmentation performance is commonly quantified using the mean intersection-over-union (mIoU) metric.

- C. Visual Reasoning

Context-dependent visual reasoning aims to identify whether a test image contains a given concept, based on a small set of support images that include both positive and negative examples. The Bongard-HOI [266] is commonly employed to assess the capability of VLMs to abstract the concept of human-object interaction from a limited number of support examples and accurately classify test samples.

- D. Out-of-Distribution Detection

OOD detection focuses on identifying whether a test sample belongs to an in-distribution (ID) dataset composed of candidate categories, which plays a vital role in safety-critical applications. Based on the degree of similarity between the OOD and ID datasets, OOD detection can be categorized into three main types: far OOD, near OOD, and fine-grained OOD

- TABLE V: Overview of datasets from various tasks used in VLM-based unsupervised learning methods. (DFT=Data-Free Transfer, UDF=Unsuperivsed Domain Transfer, ETTA=Episodic Test-Time Adaptation, OTTA=Online Test-Time Adaptation)

Popularity DFT UDF ETTA OTTA

Dataset Task # Classes # Test sample

Caltech101 [241] Object Classification 100 2,465 ⋆ ⋆ ⋆ ⋆ OxfordPets [242] Object Classification & OOD Detection 37 3,669 ⋆ ⋆ ⋆ ⋆ StanfordCars [243] Object Classification & OOD Detection 196 8,041 ⋆ ⋆ ⋆ ⋆ Flowers102 [244] Object Classification 102 2,463 ⋆ ⋆ ⋆ ⋆ Food101 [245] Object Classification & OOD Detection 101 30,300 ⋆ ⋆ ⋆ ⋆ FGVCAircraft [246] Object Classification 100 3,333 ⋆ ⋆ ⋆ ⋆ SUN397 [247] Object Classification & OOD Detection 397 19,850 ⋆ ⋆ ⋆ ⋆ DTD [248] Object Classification & OOD Detection 47 1,692 ⋆ ⋆ ⋆ ⋆ EuroSAT [249] Object Classification 10 8,100 ⋆ ⋆ ⋆ ⋆ UCF101 [250] Object Classification & Action Recognition 101 3,783 ⋆ ⋆ ⋆ ⋆ ImageNet [251] Object Classification & OOD Detection 1,000 50,000 ⋆ ⋆ ⋆ ⋆ ImageNet-A [252] Object Classification 200 7,500 ⋆ ⋆ ImageNet-V2 [253] Object Classification 1,000 10,000 ⋆ ⋆ ImageNet-R [254] Object Classification 200 30,000 ⋆ ⋆ ImageNet-Sketch [255] Object Classification 1,000 50,889 ⋆ ⋆ Office-Home [256] Object Classification 65 15,588 ⋆ ⋆ ⋆ Office [14] Object Classification 31 4,110 ⋆ ⋆ ⋆ DomainNet [257] Object Classification 345 176,743 ⋆ ⋆ ⋆ PASCAL VOC 2012 [258] Semantic Segmentation 20 1,449 ⋆ ⋆ ⋆ ⋆ PASCAL Context [259] Semantic Segmentation 59 5,105 ⋆ ⋆ ⋆ COCO Stuff [260] Semantic Segmentation 172 4,172 ⋆ ⋆ ⋆ ⋆ ADE20K [261] Semantic Segmentation 150 2,000 ⋆ ⋆ COCO-Object [262] Semantic Segmentation 80 5,000 ⋆ ⋆ ⋆ ⋆ Cityscapes [263] Semantic Segmentation 27 500 ⋆ ⋆ ⋆ KITTI-STEP [264] Semantic Segmentation 19 2,981 ⋆ ⋆ ⋆ ⋆ FireNet [265] Semantic Segmentation - 1,452 ⋆ ⋆ ⋆ ⋆ Bongard-HOI [266] Visual Reasoning 2 13,914 ⋆ ⋆ ⋆ CIFAR-100 [267] OOD Detection 100 10,000 ⋆ ⋆ ⋆ CUB-200-2011 [268] OOD Detection 200 5,794 ⋆ ⋆ ⋆ iNaturalist [269] OOD Detection 5,089 675,170 ⋆ ⋆ ⋆ Places [270] OOD Detection 365 18,250 ⋆ ⋆ ⋆ ImageNet-O [252] OOD Detection 200 2000 ⋆ ⋆ ⋆ OpenImage-O [271] OOD Detection - 17,632 ⋆ ⋆ ⋆ MS-COCO [262] Text-Image Retrieval & Image Captioning - 5,000 ⋆ ⋆ ⋆ Flickr30K [272] Text-Image Retrieval & Image Captioning - 1,000 ⋆ ⋆ ⋆ Fashion-Gen [273] Text-Image Retrieval - 32,528 ⋆ ⋆ ⋆ CUHK-PEDES [274] Text-Image Retrieval - 40,206 ⋆ ⋆ ⋆ ICFG-PEDES [275] Text-Image Retrieval - 54,522 ⋆ ⋆ ⋆ Nocaps [276] Text-Image Retrieval & Image Captioning - 15,100 ⋆ ⋆ ⋆ Guangzhou Dataset [277] Medical Image Diagnosis 2 5,856 ⋆ ⋆ Montgomery Dataset [278] Medical Image Diagnosis 2 138 ⋆ ⋆ Shenzhen Dataset [278] Medical Image Diagnosis 2 662 ⋆ ⋆ BrainTumor Dataset [31] Medical Image Diagnosis 2 593 ⋆ ⋆ ⋆ IDRID Dataset [279] Medical Image Diagnosis 5 516 ⋆ ⋆ ISIC Dataset [280] Medical Image Diagnosis 7 11720 ⋆ ⋆ ⋆ HMDB-51 [281] Action Recognition 51 6,766 ⋆ ⋆ Kinetics-600 [282] Action Recognition 600 480,000 ⋆ ⋆ ActivityNet [283] Action Recognition & Action Localization 200 19,994 ⋆ ⋆ THUMOS14 [284] Action Localization 20 212 ⋆ ⋆ ⋆

detection. Far OOD detection deals with samples that are clearly distinct from the ID distribution. For instance, when datasets such as CIFAR-100 [267], CUB-200-2011 [268], StanfordCars [243], Food101 [245], OxfordPets [242], and ImageNet [251] are used as ID data, datasets like iNaturalist [269], SUN397 [247], Places [270], and DTD [248] serve as typical far OOD sources. Near OOD detection addresses a more challenging setting where the OOD samples share visual similarities with the ID data. Common experimental setups include alternately using ImageNet-10 and ImageNet-20 as ID

and OOD datasets, as well as employing ImageNet-O [252] and OpenImage-O [271] as near OOD sets. Fine-grained OOD detection targets subtle distribution shifts within similar categories. For example, datasets such as CUB-200-2011 [268], StanfordCars [243], Food101 [245], and OxfordPets [242] can be split such that half of the classes are seen as ID data and the other half as OOD data. Evaluation of OOD detection performance is typically conducted using FPR95 and AUROC metrics.

- E. Text-Image Retrieval

Text-image retrieval is a fundamental task in visionlanguage research, where the goal is to retrieve relevant images based on textual queries, or vice versa. The MSCOCO [262] and Flickr30K [272] datasets are among the most widely used benchmarks for evaluating performance in this domain. Besides, several specialized datasets are commonly utilized to assess retrieval performance across different contexts, such as Fashion-Gen [273] from the e-commerce domain, CUHK-PEDES [274], ICFG-PEDES [275] from the person re-identification domain, and Nocaps [276] from the natural image domain. Recall@K serves as the standard metric for assessing the performance of retrieval algorithms.

- F. Image Captioning

Image captioning aims to generate descriptive textual summaries of visual content. In the context of test-time adaptation, researchers [150] evaluate the adaptability of the CLIP model across several benchmark datasets, including MSCOCO [262], Flickr30K [272], and NoCaps [276]. These datasets provide diverse visual and textual contexts, enabling the assessment of how effectively the model can generate relevant captions when exposed to new domains without additional annotated supervision. Captioning performance is evaluated using BLEU, CIDEr, SPICE, and RefCLIPScore metrics.

- G. Beyond Vanilla Object Images

Medical image diagnosis. Medical imaging represents a critical real-world application of VLMs in unsupervised learning settings. Researchers [30], [31] frequently utilize datasets such as the Guangzhou Dataset [277], Montgomery Dataset [278], and Shenzhen Dataset [278], which focus on chest X-ray diagnosis. Moreover, VLM-based unsupervised methods [30], [31] have also been applied to various other diagnostic tasks, including diabetic retinopathy [279], brain tumor detection [31], and skin lesion classification [280].

Videos. Beyond static images, VLMs have also been explored in the context of video-based unsupervised learning. Action recognition benchmarks such as HMDB-51 [281], UCF-101 [250], Kinetics-600 [282], and ActivityNet [283] are commonly used to evaluate performance. In addition, more complex tasks like temporal action localization, which involve both action classification and precise timestamp prediction, are addressed using datasets such as ActivityNet [283] and THUMOS14 [284].

IX. RESEARCH CHALLENGES AND FUTURE DIRECTIONS

Despite significant progress, unsupervised VLM adaptation remains an open and challenging problem. This section outlines key research directions, identifying gaps in the current literature and discussing potential avenues for advancing the field.

- A. Theoretical Analysis

While existing research has largely focused on developing effective unsupervised learning methods, rigorous theoretical analyses are still lacking. Understanding the theoretical complexities of VLMs is crucial for developing more principled adaptation methods. Future research can bridge this gap by providing formal generalization guarantees and characterizing the joint embedding space to explain how cross-modal alignment emerges [286].

- B. Open-world Scenarios

Most existing approaches operate under the closed-set assumption, which presumes identical label spaces across domains. However, in real-world applications, test samples often contain unknown classes, making it essential to detect and handle them effectively. While some recent studies [27], [215], [216] have begun addressing the open-world scenario, this challenging yet practical setting remains underexplored. Further research is needed to develop robust open-world adaptation methods that can generalize across diverse domains while accurately identifying unseen categories. Techniques from out-of-distribution detection [287]–[289] could also be leveraged and adapted to facilitate unknown class detection.

- C. Adversarial Robustness

Although VLMs demonstrate strong generalization capabilities, they remain highly susceptible to adversarial attacks [290]. Several recent studies [290], [291] have drawn inspiration from adversarial training techniques [292] to enhance the robustness of VLMs. However, these approaches typically rely on large amounts of labeled data, leading to substantial annotation costs. Therefore, an important research direction is to explore robust optimization [125] and inference strategies [148] under unsupervised settings, enabling VLMs to operate reliably in complex, real-world environments where adversarial threats are likely and labeled data is scarce.

- D. Privacy Considerations

Privacy and security considerations are increasingly critical for the adaptation of VLMs, particularly in sensitive domains such as autonomous driving [293] and healthcare [294]. During adaptation, models may process proprietary or personal data, raising concerns about data leakage and unauthorized access. Additionally, the adaptation process can expose models to adversarial attacks [292] that exploit vulnerabilities during the update phase, potentially degrading performance or leading to harmful outcomes. To address these challenges, future research should focus on developing privacy-preserving adaptation techniques such as federated learning [295], which enable models to adapt effectively without directly accessing raw data.

- E. Efficient Inference

The deployment of VLMs demands substantial computational resources for inference. A critical research challenge

is to reduce their latency and memory footprint without sacrificing performance. Future work may adapt techniques like quantization [296], pruning [297], and knowledge distillation [298] for the unique cross-modal nature of these models. The central difficulty lies in compressing the model while preserving the delicate vision-language alignments learned during pre-training. Developing novel and efficient architectures is crucial for enabling real-time VLM applications on resourceconstrained hardware and moving these powerful models from the cloud to the edge.

- F. More VLMs Beyond CLIP

While CLIP has become the de facto backbone for unsupervised learning of VLMs, relying solely on its contrastive framework limits architectural and objective diversity. Future research should investigate alternative base models—such as advanced training strategies [87], masked-image modeling with joint text encoders [299], or generative vision-language transformers [300]—to uncover new inductive biases. Moreover, studying how different encoder-decoder pairings impact alignment and transferability will guide the selection of more versatile models. Broadening beyond CLIP will catalyze novel unsupervised paradigms and improve VLM robustness across tasks and domains.

- G. Extension to MLLMs

Another promising research direction is to integrate TTA into MLLMs with test-time scaling [301], [302]. TTA methods enable models to dynamically adjust to distribution shifts during inference, enhancing robustness without retraining. In parallel, test-time scaling techniques allocate additional computational resources at test time—allowing models to ”think” longer or perform deeper reasoning on challenging or outof-distribution inputs [303]. By merging these approaches, an MLLM could not only adapt its predictions based on the incoming data stream but also flexibly scale its inference compute based on sample difficulty. This synergy would offer a balanced trade-off between efficiency and accuracy, especially in real-world applications where both rapid response and high adaptability are critical.

- H. New Downstream Tasks

Although unsupervised learning of VLMs has been extensively studied in image classification and semantic segmentation tasks, its potential in other domains remains largely underexplored, including regression [304], generative models [305], cross-modal retrieval [306], depth completion [307], misclassification detection [308], and image super-resolution [309]. Besides, the potential applications in other fields such as medicine [310] and healthcare [311] remain underexplored and warrant greater attention.

- I. Failure Mode and Negative Transfer

Despite the empirical success of many unsupervised adaptation methods for VLMs, few studies have systematically documented their failure modes or reported instances of

negative transfer. For example, entropy minimization [18], although widely used, can reinforce incorrect predictions when the model exhibits high uncertainty, leading to overconfident misclassifications or even mode collapse. Similarly, prompt generation via LLMs may introduce hallucinated or domaininappropriate descriptions [312], resulting in semantic misalignment with the visual content and degraded performance. In continual adaptation settings [83], the accumulation of erroneous pseudo-labels over time can distort the feature space and destabilize the adaptation process. To advance the field, future research should place greater emphasis on robustness analysis, including the development of metrics to detect adaptation failures and best practices for identifying and reporting instability. Furthermore, sharing negative results or counterexamples can play a critical role in uncovering systematic weaknesses and guiding the design of more resilient and reliable adaptation pipelines.

X. CONCLUSION

In this survey, we have presented a comprehensive and structured overview of the rapidly advancing field of unsupervised vision-language model adaptation. Addressing a notable gap in existing literature, we introduced a novel taxonomy that classifies methods based on the availability of unlabeled visual data, a crucial factor for real-world deployment. By delineating the field into four distinct settings—data-free transfer, unsupervised domain transfer, episodic test-time adaptation, and online test-time adaptation—we provided a systematic framework for understanding the unique challenges and assumptions inherent to each scenario. Within this structure, we analyzed core methodologies and reviewed representative benchmarks, offering a holistic perspective on the state of the art. Finally, we identified several key challenges and directions for future research, including the development of theoretical analysis, the handling of open-world scenarios and privacy considerations, and further exploration of new downstream tasks and application fields. This survey will not only serve as a valuable resource for practitioners seeking to navigate the landscape of unsupervised VLM adaptation but also stimulate further innovation by providing a clear basis for comparison and identifying promising directions for future research.

REFERENCES

- [1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.
- [2] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in ICML, 2021.
- [3] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” in NeurIPS, 2022.
- [4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023.
- [5] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” in NeurIPS, 2022.
- [6] R. Chen, Y. Liu, L. Kong, X. Zhu, Y. Ma, Y. Li, Y. Hou, Y. Qiao, and W. Wang, “Clip2scene: Towards label-efficient 3d scene understanding by clip,” in CVPR, 2023.

- [7] N. M. M. Shafiullah, C. Paxton, L. Pinto, S. Chintala, and A. Szlam, “Clip-fields: Weakly supervised semantic fields for robotic memory,” arXiv preprint arXiv:2210.05663, 2022.
- [8] H. Sun, Y. Cao, H. Dong, and O. Fink, “Unseen visual anomaly generation,” in CVPR, 2025.
- [9] S. Chun, S. J. Oh, R. S. De Rezende, Y. Kalantidis, and D. Larlus, “Probabilistic embeddings for cross-modal retrieval,” in CVPR, 2021.
- [10] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for vision-language models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.
- [11] ——, “Conditional prompt learning for vision-language models,” in CVPR, 2022.
- [12] M. U. Khattak, H. Rasheed, M. Maaz, S. Khan, and F. S. Khan, “Maple: Multi-modal prompt learning,” arXiv preprint arXiv:2210.03117, 2022.
- [13] Z. Wang, J. Liang, L. Sheng, R. He, Z. Wang, and T. Tan, “A hard-tobeat baseline for training-free clip-based adaptation,” in ICLR, 2024.
- [14] K. Saenko, B. Kulis, M. Fritz, and T. Darrell, “Adapting visual category models to new domains,” in ECCV, 2010.
- [15] S. Menon and C. Vondrick, “Visual classification via description from large language models,” in ICLR, 2023.
- [16] C. Zhou, C. C. Loy, and B. Dai, “Extract free dense labels from clip,” in ECCV, 2022.
- [17] T. Huang, J. Chu, and F. Wei, “Unsupervised prompt learning for vision-language models,” arXiv preprint arXiv:2204.03649, 2022.
- [18] M. Shu, W. Nie, D.-A. Huang, Z. Yu, T. Goldstein, A. Anandkumar, and C. Xiao, “Test-time prompt tuning for zero-shot generalization in vision-language models,” in NeurIPS, 2022.
- [19] X. Ma, J. Zhang, S. Guo, and W. Xu, “Swapprompt: Test-time prompt adaptation for vision-language models,” in NeurIPS, 2023.
- [20] L. Sheng, J. Liang, R. He, Z. Wang, and T. Tan, “The illusion of progress? a critical look at test-time adaptation for vision-language models,” arXiv preprint arXiv:2506.24000, 2025.
- [21] S. Pratt, I. Covert, R. Liu, and A. Farhadi, “What does a platypus look like? generating customized prompts for zero-shot image classification,” in ICCV, 2023.
- [22] J. Kahana, N. Cohen, and Y. Hoshen, “Improving zero-shot models with label distribution priors,” arXiv preprint arXiv:2212.00784, 2022.
- [23] Q. Qian and J. Hu, “Online zero-shot classification with clip,” in ECCV, 2024.
- [24] J. Abdul Samadh, M. H. Gani, N. Hussein, M. U. Khattak, M. M. Naseer, F. Shahbaz Khan, and S. H. Khan, “Align your prompts: Testtime prompting with distribution alignment for zero-shot generalization,” in NeurIPS, 2023.
- [25] A. Karmanov, D. Guan, S. Lu, A. El Saddik, and E. Xing, “Efficient test-time adaptation of vision-language models,” in CVPR, 2024.
- [26] J. Li, S. Savarese, and S. C. Hoi, “Masked unsupervised self-training for label-free image classification,” in ICLR, 2023.
- [27] J. Liang, L. Sheng, Z. Wang, R. He, and T. Tan, “Realistic unsupervised clip fine-tuning with universal entropy optimization,” in ICML, 2024.
- [28] G. Shin, W. Xie, and S. Albanie, “Reco: Retrieve and co-segment for zero-shot transfer,” in NeurIPS, 2022.
- [29] S. Hajimiri, I. B. Ayed, and J. Dolz, “Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation,” in WACV, 2025.
- [30] U. Rahman, R. Imam, M. Yaqub, B. B. Amor, and D. Mahapatra, “Can language-guided unsupervised adaptation improve medical image classification using unpaired images and texts?” in ISBI, 2025.
- [31] J. Liu, T. Hu, Y. Zhang, X. Gai, Y. Feng, and Z. Liu, “A chatgpt aided explainable framework for zero-shot medical image diagnosis,” in ICML Workshops, 2023.
- [32] M. Bosetti, S. Zhang, B. Liberatori, G. Zara, E. Ricci, and P. Rota, “Text-enhanced zero-shot action recognition: A training-free approach,” in ICPR, 2024.
- [33] R. Yan, H. Qu, X. Shu, W. Li, J. Tang, and T. Tan, “Dts-tpt: dual temporal-sync test-time prompt tuning for zero-shot activity recognition,” in IJCAI, 2024.
- [34] K. Tanwisuth, S. Zhang, H. Zheng, P. He, and M. Zhou, “Pouf: Promptoriented unsupervised fine-tuning for large pre-trained models,” in ICML, 2023.
- [35] Y. Kalantidis, G. Tolias et al., “Label propagation for zero-shot classification with vision-language models,” in CVPR, 2024.
- [36] M. Zanella and I. Ben Ayed, “On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?” in CVPR, 2024.
- [37] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

- [38] H. Dong, M. Liu, K. Zhou, E. Chatzi, J. Kannala, C. Stachniss, and O. Fink, “Advances in multimodal adaptation and generalization: From traditional approaches to foundation models,” arXiv preprint arXiv:2501.18592, 2025.
- [39] X. Li, J. Li, F. Li, L. Zhu, Y. Yang, and H. T. Shen, “Generalizing vision-language models to novel domains: A comprehensive survey,” arXiv preprint arXiv:2506.18504, 2025.
- [40] J. Liang, R. He, and T. Tan, “A comprehensive survey on test-time adaptation under distribution shifts,” International Journal of Computer Vision, vol. 133, no. 1, pp. 31–64, 2025.
- [41] Z. Wang, Y. Luo, L. Zheng, Z. Chen, S. Wang, and Z. Huang, “In search of lost online test-time adaptation: A survey,” International Journal of Computer Vision, vol. 133, no. 3, pp. 1106–1139, 2025.
- [42] Z. Xiao and C. G. Snoek, “Beyond model adaptation at test time: A survey,” arXiv preprint arXiv:2411.03687, 2024.
- [43] L. Yao, R. Huang, L. Hou, G. Lu, M. Niu, H. Xu, X. Liang, Z. Li, X. Jiang, and C. Xu, “Filip: Fine-grained interactive language-image pre-training,” arXiv preprint arXiv:2111.07783, 2021.
- [44] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in NeurIPS, 2017.
- [45] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.
- [46] X. Chen, H. Fang, T.-Y. Lin, R. Vedantam, S. Gupta, P. Doll´ar, and C. L. Zitnick, “Microsoft coco captions: Data collection and evaluation server,” arXiv preprint arXiv:1504.00325, 2015.
- [47] Q. You, H. Jin, Z. Wang, C. Fang, and J. Luo, “Image captioning with semantic attention,” in CVPR, 2016.
- [48] Q. Wu, D. Teney, P. Wang, C. Shen, A. Dick, and A. Van Den Hengel, “Visual question answering: A survey of methods and datasets,” Computer Vision and Image Understanding, vol. 163, pp. 21–40, 2017.
- [49] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022.
- [50] L. Zhen, P. Hu, X. Wang, and D. Peng, “Deep supervised cross-modal retrieval,” in CVPR, 2019.
- [51] C. Zhou, Q. Li, C. Li, J. Yu, Y. Liu, G. Wang, K. Zhang, C. Ji, Q. Yan, L. He et al., “A comprehensive survey on pretrained foundation models: A history from bert to chatgpt,” International Journal of Machine Learning and Cybernetics, pp. 1–65, 2024.
- [52] C. H. Lampert, H. Nickisch, and S. Harmeling, “Learning to detect unseen object classes by between-class attribute transfer,” in CVPR, 2009.
- [53] A. Farhadi, I. Endres, D. Hoiem, and D. Forsyth, “Describing objects by their attributes,” in CVPR, 2009.
- [54] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. Ranzato, and T. Mikolov, “Devise: A deep visual-semantic embedding model,” in NeurIPS, 2013.
- [55] B. Romera-Paredes and P. Torr, “An embarrassingly simple approach to zero-shot learning,” in ICML, 2015.
- [56] V. K. Verma, G. Arora, A. Mishra, and P. Rai, “Generalized zero-shot learning via synthesized examples,” in CVPR, 2018.
- [57] R. Felix, I. Reid, G. Carneiro et al., “Multi-modal cycle-consistent generalized zero-shot learning,” in ECCV, 2018.
- [58] Y. Xian, S. Sharma, B. Schiele, and Z. Akata, “f-vaegan-d2: A feature generating framework for any-shot learning,” in CVPR, 2019.
- [59] W.-L. Chao, S. Changpinyo, B. Gong, and F. Sha, “An empirical study and analysis of generalized zero-shot learning for object recognition in the wild,” in ECCV, 2016.
- [60] Y. Xian, B. Schiele, and Z. Akata, “Zero-shot learning-the good, the bad and the ugly,” in CVPR, 2017.
- [61] F. Pourpanah, M. Abdar, Y. Luo, X. Zhou, R. Wang, C. P. Lim, X.-Z. Wang, and Q. J. Wu, “A review of generalized zero-shot learning methods,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4051–4070, 2022.
- [62] W. Wang, V. W. Zheng, H. Yu, and C. Miao, “A survey of zero-shot learning: Settings, methods, and applications,” ACM Transactions on Intelligent Systems and Technology, vol. 10, no. 2, pp. 1–37, 2019.
- [63] C. Ma, Y. Liu, J. Deng, L. Xie, W. Dong, and C. Xu, “Understanding and mitigating overfitting in prompt tuning for vision-language models,” arXiv preprint arXiv:2211.02219, 2022.
- [64] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, and Y. Qiao, “Clip-adapter: Better vision-language models with feature adapters,” International Journal of Computer Vision, vol. 132, no. 2, pp. 581–595, 2024.

- [65] R. Zhang, R. Fang, P. Gao, W. Zhang, K. Li, J. Dai, Y. Qiao, and H. Li, “Tip-adapter: Training-free clip-adapter for better visionlanguage modeling,” arXiv preprint arXiv:2111.03930, 2021.
- [66] M. Zanella and I. Ben Ayed, “Low-rank few-shot adaptation of visionlanguage models,” in CVPR Workshops, 2024.
- [67] S. Menon and C. Vondrick, “Visual classification via description from large language models,” arXiv preprint arXiv:2210.07183, 2022.
- [68] Y. Rao, W. Zhao, G. Chen, Y. Tang, Z. Zhu, G. Huang, J. Zhou, and J. Lu, “Denseclip: Language-guided dense prediction with contextaware prompting,” in CVPR, 2022.
- [69] J. Gu, Z. Han, S. Chen, A. Beirami, B. He, G. Zhang, R. Liao, Y. Qin, V. Tresp, and P. Torr, “A systematic survey of prompt engineering on vision-language foundation models,” arXiv preprint arXiv:2307.12980, 2023.
- [70] I. Nejjar, H. Dong, and O. Fink, “Recall and refine: A simple but effective source-free open-set domain adaptation framework,” arXiv preprint arXiv:2411.12558, 2024.
- [71] J. Liang, D. Hu, and J. Feng, “Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation,” in ICML, 2020.
- [72] J. Wen, R. Liu, N. Zheng, Q. Zheng, Z. Gong, and J. Yuan, “Exploiting local feature patterns for unsupervised domain adaptation,” in AAAI, 2019.
- [73] D. Chen, D. Wang, T. Darrell, and S. Ebrahimi, “Contrastive test-time adaptation,” in CVPR, 2022.
- [74] J. N. Kundu, N. Venkat, and R. V. Babu, “Universal source-free domain adaptation,” in CVPR, 2020.
- [75] J. Huang, D. Guan, A. Xiao, and S. Lu, “Model adaptation: Historical contrastive learning for unsupervised domain adaptation without source data,” in NeurIPS, 2021.
- [76] Y. Kim, D. Cho, K. Han, P. Panda, and S. Hong, “Domain adaptation without source data,” IEEE Transactions on Artificial Intelligence, vol. 2, no. 6, pp. 508–518, 2021.
- [77] J. Li, Z. Yu, Z. Du, L. Zhu, and H. T. Shen, “A comprehensive survey on source-free domain adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5743–5762, 2024.
- [78] Y. Fang, P.-T. Yap, W. Lin, H. Zhu, and M. Liu, “Source-free unsupervised domain adaptation: A survey,” Neural Networks, vol. 174, p. 106230, 2024.
- [79] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in ICLR, 2021.
- [80] H. Dong, E. Chatzi, and O. Fink, “Towards robust multimodal open-set test-time adaptation via adaptive entropy-aware optimization,” in ICLR, 2025.
- [81] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Efficient test-time model adaptation without forgetting,” in ICML, 2022.
- [82] T. Gong, Y. Kim, T. Lee, S. Chottananurak, and S.-J. Lee, “Sotta: Robust test-time adaptation on noisy data streams,” in NeurIPS, 2023.
- [83] Q. Wang, O. Fink, L. Van Gool, and D. Dai, “Continual test-time domain adaptation,” in CVPR, 2022.
- [84] J. Song, J. Lee, I. S. Kweon, and S. Choi, “Ecotta: Memoryefficient continual test-time adaptation via self-distilled regularization,” in CVPR, 2023.
- [85] C.-M. Feng, K. Yu, Y. Liu, S. Khan, and W. Zuo, “Diverse data augmentation with diffusions for effective test-time prompt tuning,” in ICCV, 2023.
- [86] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Eva-clip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.
- [87] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in ICCV, 2023.
- [88] Z. Novack, J. McAuley, Z. C. Lipton, and S. Garg, “Chils: Zero-shot image classification with hierarchical label sets,” in ICML, 2023.
- [89] Z. Guo, B. Dong, Z. Ji, J. Bai, Y. Guo, and W. Zuo, “Texts as images in prompt tuning for multi-label image recognition,” in CVPR, 2023.
- [90] V. Udandarao, A. Gupta, and S. Albanie, “Sus-x: Training-free nameonly transfer of vision-language models,” in ICCV, 2023.
- [91] M. Wallingford, V. Ramanujan, A. Fang, A. Kusupati, R. Mottaghi, A. Kembhavi, L. Schmidt, and A. Farhadi, “Neural priming for sampleefficient adaptation,” in NeurIPS, 2023.
- [92] M. Seo, S. Cho, M. Lee, D. Misra, H. Choi, S. J. Kim, and J. Choi, “Just say the name: Online continual learning with category names only via data generation,” arXiv preprint arXiv:2403.10853, 2024.
- [93] Z. Guo, R. Zhang, L. Qiu, X. Ma, X. Miao, X. He, and B. Cui, “Calip: Zero-shot enhancement of clip with parameter-free attention,” in AAAI, 2023.

- [94] F. Wang, J. Mei, and A. Yuille, “Sclip: Rethinking self-attention for dense vision-language inference,” in ECCV, 2024.
- [95] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Proxyclip: Proxy attention improves clip for open-vocabulary segmentation,” in ECCV, 2024.
- [96] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” in NeurIPS, 2020.
- [97] M. J. Mirza, L. Karlinsky, W. Lin, S. Doveh, J. Micorek, M. Kozinski, H. Kuhene, and H. Possegger, “Meta-prompting for automating zeroshot visual recognition with llms,” in ECCV, 2024.
- [98] S. Parashar, Z. Lin, T. Liu, X. Dong, Y. Li, D. Ramanan, J. Caverlee, and S. Kong, “The neglected tails in vision-language models,” in CVPR, 2024.
- [99] O. Blog, “Introducing chatgpt,” 2023.
- [100] S. Parashar, Z. Lin, Y. Li, and S. Kong, “Prompting scientific names for zero-shot species recognition,” in EMNLP, 2023.
- [101] K. Roth, J. M. Kim, A. Koepke, O. Vinyals, C. Schmid, and Z. Akata, “Waffling around for performance: Visual classification with random words and broad concepts,” in ICCV, 2023.
- [102] X. Liu and C. Zach, “Tag: Text prompt augmentation for zero-shot out-of-distribution detection supplementary material,” in ECCV, 2024.
- [103] W. Sun, Y. Du, G. Liu, R. Kompella, and C. G. Snoek, “Trainingfree semantic segmentation via llm-supervision,” arXiv preprint arXiv:2404.00701, 2024.
- [104] M. Moayeri, M. Rabbat, M. Ibrahim, and D. Bouchacourt, “Embracing diversity: Interpretable zero-shot classification beyond one vector per class,” in ACM Conference on Fairness, Accountability, and Transparency, 2024.
- [105] C. Cao, Z. Zhong, Z. Zhou, Y. Liu, T. Liu, and B. Han, “Envisioning outlier exposure by large language models for out-of-distribution detection,” in ICML, 2024.
- [106] Z. Ren, Y. Su, and X. Liu, “Chatgpt-powered hierarchical comparisons for image classification,” in NeurIPS, 2023.
- [107] H. Lee, G. Seo, W. Choi, G. Jung, K. Song, and J. Jung, “Enhancing visual classification using comparative descriptors,” in WACV, 2025.
- [108] T. Liang and J. Davis, “Making better mistakes in clip-based zero-shot classification with hierarchy-aware language prompts,” arXiv preprint arXiv:2503.02248, 2025.
- [109] M. J. Mirza, L. Karlinsky, W. Lin, H. Possegger, R. Feris, and H. Bischof, “Tap: Targeted prompting for task adaptive generation of textual training instances for visual classification,” arXiv preprint arXiv:2309.06809, 2023.
- [110] M. U. Khattak, M. F. Naeem, M. Naseer, L. Van Gool, and F. Tombari, “Learning to prompt with text only supervision for vision-language models,” in AAAI, 2025.
- [111] J. Shipard, A. Wiliem, K. N. Thanh, W. Xiang, and C. Fookes, “Diversity is definitely needed: Improving model-agnostic zero-shot classification via stable diffusion,” in CVPR, 2023.
- [112] S. Wang, L. Song, R. Shimizu, M. Goto, and H. Wu, “Attributed synthetic data generation for zero-shot domain-specific image classification,” in ICME, 2025.
- [113] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Clearclip: Decomposing clip representations for dense vision-language inference,” in ECCV, 2024.
- [114] Y. Li, H. Wang, Y. Duan, J. Zhang, and X. Li, “A closer look at the explainability of contrastive language-image pre-training,” Pattern Recognition, vol. 162, p. 111409, 2025.
- [115] W. Bousselham, F. Petersen, V. Ferrari, and H. Kuehne, “Grounding everything: Emerging localization properties in vision-language transformers,” in CVPR, 2024.
- [116] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in ICCV, 2021.
- [117] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023.
- [118] M. J. Mirza, L. Karlinsky, W. Lin, H. Possegger, M. Kozinski, R. Feris, and H. Bischof, “Lafter: Label-free tuning of zero-shot classifier using language and unlabeled image collections,” in NeurIPS, 2023.
- [119] X. Hu, K. Zhang, L. Xia, A. Chen, J. Luo, Y. Sun, K. Wang, N. Qiao, X. Zeng, M. Sun et al., “Reclip: Refine contrastive language image pre-training with source free domain adaptation,” in WACV, 2024.
- [120] Y. Li, Y. Cao, J. Li, Q. Wang, and S. Wang, “Data-efficient clippowered dual-branch networks for source-free unsupervised domain adaptation,” arXiv preprint arXiv:2410.15811, 2024.

- [121] S. Long, L. Wang, Z. Zhao, Z. Tan, Y. Wu, S. Wang, and J. Wang, “Training-free unsupervised prompt for vision-language models,” arXiv preprint arXiv:2404.16339, 2024.
- [122] J. Huang, J. Zhang, H. Qiu, S. Jin, and S. Lu, “Prompt ensemble self-training for open-vocabulary domain adaptation,” arXiv preprint arXiv:2306.16658, 2023.
- [123] Z. Li, X. Li, X. Fu, X. Zhang, W. Wang, S. Chen, and J. Yang, “Promptkd: Unsupervised prompt distillation for vision-language models,” in CVPR, 2024.
- [124] Q. Xu, W. Chen, Z. Hu, H. Li, and Y. Tai, “Otfusion: Bridging visiononly and vision-language models via optimal transport for transductive zero-shot learning,” arXiv preprint arXiv:2506.13723, 2025.
- [125] C. Schlarmann, N. D. Singh, F. Croce, and M. Hein, “Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models,” in ICML, 2024.
- [126] C. Shin, J. Zhao, S. Cromp, H. Vishwakarma, and F. Sala, “Otter: Effortless label distribution adaptation of zero-shot models,” in NeurIPS, 2024.
- [127] M. Zanella, B. G´erin, and I. Ayed, “Boosting vision-language models with transduction,” in NeurIPS, 2024.
- [128] Y. Zhang, C. Zhang, X. Hu, and Z. He, “Unsupervised prototype adapter for vision-language models,” in PRCV, 2023.
- [129] M. F. Imam, R. F. Marew, J. Hassan, M. Fiaz, A. F. Aji, and H. Cholakkal, “Clip meets dino for tuning zero-shot classifier using unlabeled image collections,” arXiv preprint arXiv:2411.19346, 2024.
- [130] X. Li, C. Wen, Y. Hu, and N. Zhou, “Rs-clip: Zero shot remote sensing scene classification via contrastive vision-language supervision,” International Journal of Applied Earth Observation and Geoinformation, vol. 124, p. 103497, 2023.
- [131] O. Saha, L. Lawrence, G. Van Horn, and S. Maji, “Generate, transduct, adapt: Iterative transduction with vlms,” in ICCV, 2025.
- [132] J. Zhang, Q. Wei, F. Liu, and L. Feng, “Candidate pseudolabel learning: Enhancing vision-language models by prompt tuning with unlabeled data,” in ICML, 2024.
- [133] C. Menghini, A. Delworth, and S. Bach, “Enhancing clip with clip: Exploring pseudolabeling for limited-label prompt tuning,” in NeurIPS, 2023.
- [134] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin, and C.-L. Li, “Fixmatch: Simplifying semisupervised learning with consistency and confidence,” in NeurIPS, 2020.
- [135] E. Ali, S. Silva, and M. H. Khan, “Dpa: Dual prototypes alignment for unsupervised adaptation of vision-language models,” in WACV, 2025.
- [136] C. Laroudie, A. Bursuc, M. L. Ha, and G. Franchi, “Improving clip robustness with knowledge distillation and self-training,” arXiv preprint arXiv:2309.10361, 2023.
- [137] A. Iscen, G. Tolias, Y. Avrithis, and O. Chum, “Label propagation for deep semi-supervised learning,” in CVPR, 2019.
- [138] E. Ali and M. H. Khan, “Noise-tolerant few-shot unsupervised adapter for vision-language models,” in BMVC, 2023.
- [139] H. Yan and Y. Guo, “Lightweight unsupervised federated learning with pretrained vision language model,” in International Workshop on Trustworthy Federated Learning, 2024.
- [140] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [141] A.-Q. Cao, M. Jaritz, M. Guillaumin, R. de Charette, and L. Bazzani, “Latteclip: Unsupervised clip fine-tuning via lmm-synthetic texts,” in WACV, 2025.
- [142] M. Mistretta, A. Baldrati, M. Bertini, and A. D. Bagdanov, “Improving zero-shot generalization of learned prompts via unsupervised knowledge distillation,” in ECCV, 2024.
- [143] Y. Benigmim, M. Fahes, T.-H. Vu, A. Bursuc, and R. de Charette, “Floss: Free lunch in open-vocabulary semantic segmentation,” in ICCV, 2025.
- [144] J. U. Allingham, J. Ren, M. W. Dusenberry, X. Gu, Y. Cui, D. Tran, J. Z. Liu, and B. Lakshminarayanan, “A simple zero-shot prompt weighting technique to improve prompt ensembling in text-image models,” in ICML, 2023.
- [145] A. T. Nguyen, K. S. Tai, B.-C. Chen, S. N. Shukla, H. Yu, P. Torr, T.-P. Tian, and S.-N. Lim, “ucap: An unsupervised prompting method for vision-language models,” in ECCV, 2024.
- [146] Q. Qian, Y. Xu, and J. Hu, “Intra-modal proxy learning for zero-shot visual categorization with clip,” in NeurIPS, 2023.
- [147] S. Martin, Y. Huang, F. Shakeri, J.-C. Pesquet, and I. Ben Ayed, “Transductive zero-shot and few-shot clip,” in CVPR, 2024.

- [148] L. Sheng, J. Liang, Z. Wang, and R. He, “R-tpt: Improving adversarial robustness of vision-language models through test-time prompt tuning,” in CVPR, 2025.
- [149] M. Prabhudesai, T.-W. Ke, A. C. Li, D. Pathak, and K. Fragkiadaki, “Diffusion-tta: Test-time adaptation of discriminative models via generative feedback,” in NeurIPS, 2023.
- [150] S. Zhao, X. Wang, L. Zhu, and Y. Yang, “Test-time adaptation with clip reward for zero-shot generalization in vision-language models,” in ICLR, 2023.
- [151] X. Qiao, P. Huang, J. Yuan, X. Guo, B. Ye, Z. Sun, and X. Li, “Bidirectional prototype-reward co-evolution for test-time adaptation of vision-language models,” arXiv preprint arXiv:2503.09394, 2025.
- [152] X. Wang, K. Chen, J. Zhang, J. Chen, and X. Ma, “Tapt: Testtime adversarial prompt tuning for robust inference in vision-language models,” in CVPR, 2024.
- [153] M. Zanella, C. Fuchs, C. De Vleeschouwer, and I. B. Ayed, “Realistic test-time adaptation of vision-language models,” in CVPR, 2025.
- [154] Y. Zhu, G. Zhang, C. Xu, H. Shen, X. Chen, G. Wu, and L. Wang, “Efficient test-time prompt tuning for vision-language models,” arXiv preprint arXiv:2408.05775, 2024.
- [155] J. Yin, X. Zhang, L. Wu, and X. Wang, “In-context prompt learning for test-time vision recognition with frozen vision-language model,” arXiv preprint arXiv:2403.06126, 2024.
- [156] Y. Kojima, J. Xu, X. Zou, and X. Wang, “Lora-ttt: Low-rank test-time training for vision-language models,” in ICML Workshops, 2025.
- [157] B. Liberatori, A. Conti, P. Rota, Y. Wang, and E. Ricci, “Test-time zero-shot temporal action localization,” in CVPR, 2024.
- [158] Y. Zhu, Y. Ji, Z. Zhao, G. Wu, and L. Wang, “Awt: Transferring visionlanguage models via augmentation, weighting, and transportation,” in NeurIPS, 2024.
- [159] Y. Lee, D. Kim, J. Kang, J. Bang, H. Song, and J.-G. Lee, “RA-TTA: Retrieval-augmented test-time adaptation for vision-language models,” in ICLR, 2025.
- [160] C. Zhang, K. Xu, Z. Liu, Y. Peng, and J. Zhou, “Scap: Transductive test-time adaptation via supportive clique-based attribute prompting,” in CVPR, 2025.
- [161] M. Farina, G. Franchi, G. Iacca, M. Mancini, and E. Ricci, “Frustratingly easy test-time adaptation of vision-language models,” in NeurIPS, 2024.
- [162] C.-M. Feng, Y. He, J. Zou, S. Khan, H. Xiong, Z. Li, W. Zuo, R. S. M. Goh, and Y. Liu, “Diffusion-enhanced test-time adaptation with text and image augmentation,” International Journal of Computer Vision, pp. 1–16, 2025.
- [163] H. S. Yoon, E. Yoon, J. T. J. Tee, M. Hasegawa-Johnson, Y. Li, and C. D. Yoo, “C-tpt: Calibrated test-time prompt tuning for visionlanguage models via text feature dispersion,” in ICLR, 2024.
- [164] A. Sharifdeen, M. A. Munir, S. Baliah, S. Khan, and M. H. Khan, “Otpt: Orthogonality constraints for calibrating test-time prompt tuning in vision-language models,” in CVPR, 2025.
- [165] J. Sun, M. Ibrahim, M. Hall, I. Evtimov, Z. M. Mao, C. C. Ferrer, and C. Hazirbas, “Vpa: Fully test-time visual prompt adaptation,” in ACM-MM, 2023.
- [166] R. Imam, H. Gani, M. Huzaifa, and K. Nandakumar, “Test-time low rank adaptation via confidence maximization for zero-shot generalization of vision-language models,” in WACV, 2024.
- [167] R. Imam, A. Hanif, J. Zhang, K. W. Dawoud, Y. Kementchedjhieva, and M. Yaqub, “Noise is an efficient learner for zero-shot vision-language models,” arXiv preprint arXiv:2502.06019, 2025.
- [168] J. Hessel, A. Holtzman, M. Forbes, R. L. Bras, and Y. Choi, “Clipscore: A reference-free evaluation metric for image captioning,” in EMNLP, 2021.
- [169] E. Sui, X. Wang, and S. Yeung-Levy, “Just shift it: Test-time prototype shifting for zero-shot generalization with vision-language models,” in WACV, 2024.
- [170] A. Khandelwal, “Promptsync: Bridging domain gaps in vision-language models through class-aware prototype alignment and discrimination,” in CVPR Workshops, 2024.
- [171] J. Gui, T. Chen, J. Zhang, Q. Cao, Z. Sun, H. Luo, and D. Tao, “A survey on self-supervised learning: Algorithms, applications, and future trends,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9052–9071, 2024.
- [172] D. Hendrycks, M. Mazeika, S. Kadavath, and D. Song, “Using selfsupervised learning can improve model robustness and uncertainty,” in NeurIPS, 2019.
- [173] M. Liu, T. L. Hayes, M. Mancini, E. Ricci, R. Volpi, and G. Csurka, “Test-time vocabulary adaptation for language-driven object detection,” in ICIP, 2025.

- [174] L. Cai, J. Kang, S. Li, W. Ma, B. Xie, Z. Qin, and J. Liang, “From local details to global context: Advancing vision-language models with attention-based selection,” in ICML, 2025.
- [175] C. Yi, L. Ren, D.-C. Zhan, and H.-J. Ye, “Leveraging cross-modal neighbor representation for improved clip classification,” in CVPR, 2024.
- [176] B. An, S. Zhu, M.-A. Panaitescu-Liess, C. K. Mummadi, and F. Huang, “Perceptionclip: Visual classification by inferring and conditioning on contexts,” in ICLR, 2023.
- [177] Y. Ge, J. Ren, A. Gallagher, Y. Wang, M.-H. Yang, H. Adam, L. Itti, B. Lakshminarayanan, and J. Zhao, “Improving zero-shot generalization and robustness of multi-modal models,” in CVPR, 2023.
- [178] S. Shen, Z. Zhu, L. Fan, H. Zhang, and X. Wu, “Diffclip: Leveraging stable diffusion for language grounded 3d classification,” in WACV, 2024.
- [179] M. M. Rahaman, E. K. Millar, and E. Meijering, “Leveraging visionlanguage embeddings for zero-shot learning in histopathology images,” IEEE Journal of Biomedical and Health Informatics, 2025.
- [180] J. Li, M. Yang, Y. Tian, L. Zhang, Y. Lu, J. Liu, and W. Wang, “Wavedn: A wavelet-based training-free zero-shot enhancement for vision-language models,” in ACM-MM, 2024.
- [181] S. Eom, N. Ho, J. Oh, and S.-Y. Yun, “Cross-modal retrieval meets inference: Improving zero-shot classification with cross-modal retrieval,” arXiv preprint arXiv:2308.15273, 2023.
- [182] A. Conti, E. Fini, M. Mancini, P. Rota, Y. Wang, and E. Ricci, “Vocabulary-free image classification and semantic segmentation,” arXiv preprint arXiv:2404.10864, 2024.
- [183] B. Murugesan, J. Silva-Rodr´ıguez, I. B. Ayed, and J. Dolz, “Robust calibration of large vision-language adapters,” in ECCV, 2024.
- [184] H. Ma, Y. Zhu, C. Zhang, P. Zhao, B. Wu, L.-K. Huang, Q. Hu, and B. Wu, “Spurious feature eraser: Stabilizing test-time adaptation for vision-language foundation model,” in AAAI, 2024.
- [185] A. Li, L. Zhuang, X. Long, M. Yao, and S. Wang, “Test-time loss landscape adaptation for zero-shot generalization in vision-language models,” arXiv preprint arXiv:2501.18864, 2025.
- [186] S. Xing, Z. Zhao, and N. Sebe, “Clip is strong enough to fight back: Test-time counterattacks towards zero-shot adversarial robustness of clip,” in CVPR, 2025.
- [187] M. Wysocza´nska, M. Ramamonjisoa, T. Trzci´nski, and O. Sim´eoni, “Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free,” in WACV, 2024.
- [188] T. Shao, Z. Tian, H. Zhao, and J. Su, “Explore the potential of clip for training-free open vocabulary semantic segmentation,” in ECCV, 2024.
- [189] D. Kang and M. Cho, “In defense of lazy visual grounding for openvocabulary semantic segmentation,” in ECCV, 2024.
- [190] J. Li, H. Li, S. Erfani, L. Feng, J. Bailey, and F. Liu, “Visual-text cross alignment: Refining the similarity score in vision-language models,” in ICML, 2024.
- [191] X. Yin, Q. Wang, B. Cao, and Q. Hu, “S3: Synonymous semantic space for improving zero-shot generalization of vision-language models,” arXiv preprint arXiv:2412.04925, 2024.
- [192] A. Abdelhamed, M. Afifi, and A. Go, “What do you see? enhancing zero-shot image classification with multimodal large language models,” arXiv preprint arXiv:2405.15668, 2024.
- [193] C. Wei, “Enhancing fine-grained image classifications via cascaded vision language models,” in EMNLP Findings, 2024.
- [194] A. Munir, F. Z. Qureshi, M. H. Khan, and M. Ali, “Tlac: Two-stage lmm augmented clip for zero-shot classification,” in CVPR Workshops, 2025.
- [195] K. Miller, S. Mishra, A. Gangrade, K. Saenko, and V. Saligrama, “Sparc: Score prompting and adaptive fusion for zero-shot multi-label recognition in vision-language models,” in CVPR, 2025.
- [196] R. Esfandiarpoor and S. H. Bach, “Follow-up differential descriptions: Language models resolve ambiguities for image classification,” in ICLR, 2023.
- [197] Z. Guo, B. Dong, Z. Ji, J. Bai, Y. Guo, and W. Zuo, “Texts as images in prompt tuning for multi-label image recognition,” in CVPR, 2023.
- [198] Z. Liu, H. Sun, Y. Peng, and J. Zhou, “Dart: Dual-modal adaptive online prompting and knowledge retention for test-time adaptation,” in AAAI, 2024.
- [199] G. A. V. Hakim, D. Osowiechi, M. Noori, M. Cheraghalikhani, A. Bahri, M. Yazdanpanah, I. B. Ayed, and C. Desrosiers, “Clipartt: Adaptation of clip to new domains at test time,” in WACV, 2024.
- [200] S. Mishra, J. Silva-Rodrıguez, I. B. Ayed, M. Pedersoli, and J. Dolz, “Words matter: Leveraging individual text embeddings for code generation in clip test-time adaptation,” arXiv preprint arXiv:2411.17002, 2024.

- [201] D. Osowiechi, M. Noori, G. A. V. Hakim, M. Yazdanpanah, A. Bahri, M. Cheraghalikhani, S. Dastani, F. Beizaee, I. B. Ayed, and C. Desrosiers, “Watt: Weight average test-time adaption of clip,” in NeurIPS, 2024.
- [202] Y. Zhang, W. Zhu, H. Tang, Z. Ma, K. Zhou, and L. Zhang, “Dual memory networks: A versatile adaptation approach for vision-language models,” in CVPR, 2024.
- [203] C. Zhang, S. Stepputtis, K. Sycara, and Y. Xie, “Dual prototype evolving for test-time generalization of vision-language models,” in NeurIPS, 2024.
- [204] X. Hu, K. Zhang, M. Sun, A. Chen, C.-H. Kuo, and R. Nevatia, “Bafta: Backprop-free test-time adaptation for zero-shot vision-language models,” arXiv preprint arXiv:2406.11309, 2024.
- [205] C. Fuchs, M. Zanella, and C. De Vleeschouwer, “Online gaussian test-time adaptation of vision-language models,” arXiv preprint arXiv:2501.04352, 2025.
- [206] Z. Han, J. Yang, J. Li, Q. Hu, Q. Xu, M. Z. Shou, and C. Zhang, “Dota: Distributional test-time adaptation of vision-language models,” arXiv preprint arXiv:2409.19375, 2024.
- [207] L. Zhou, M. Ye, S. Li, N. Li, X. Zhu, L. Deng, H. Liu, and Z. Lei, “Bayesian test-time adaptation for vision-language models,” in CVPR, 2025.
- [208] Y. Zhou, J. Ren, F. Li, R. Zabih, and S. N. Lim, “Test-time distribution normalization for contrastively learned visual-language models,” in NeurIPS, 2023.
- [209] Z. Xiao, S. Yan, J. Hong, J. Cai, X. Jiang, Y. Hu, J. Shen, C. Wang, and C. G. M. Snoek, “Dynaprompt: Dynamic test-time prompt tuning,” in ICLR, 2025.
- [210] Z. Wang, D. Gong, S. Wang, Z. Huang, and Y. Luo, “Is less more? exploring token condensation as training-free test-time adaptation,” in ICCV, 2025.
- [211] Y. Li, Y. Su, A. Goodge, K. Jia, and X. Xu, “Efficient and contextaware label propagation for zero-/few-shot training-free adaptation of vision-language model,” in ICLR, 2025.
- [212] J. Ma, “Improved self-training for test-time adaptation,” in CVPR, 2024.
- [213] F. Wang, Z. Han, X. Liu, Y. Yin, and X. Gao, “Ctpt: Continual testtime prompt tuning for vision-language models,” Pattern Recognition, vol. 161, p. 111300, 2025.
- [214] R. Wang, H. Zuo, Z. Fang, and J. Lu, “Towards robustness prompt tuning with fully test-time adaptation for clip’s zero-shot generalization,” in ACM-MM, 2024.
- [215] C. Cao, Z. Zhong, Z. Zhou, T. Liu, Y. Liu, K. Zhang, and B. Han, “Noisy test-time adaptation in vision-language models,” in ICLR, 2025.
- [216] M. Sreenivas and S. Biswas, “Effectiveness of vision language models for open-world single image test time adaptation,” arXiv preprint arXiv:2406.00481, 2024.
- [217] Z. Lu, J. Bai, X. Li, Z. Xiao, and X. Wang, “Task-to-instance prompt learning for vision-language models at test time,” IEEE Transactions on Image Processing, vol. 34, pp. 1908–1920, 2025.
- [218] F. Huang, J. Jiang, Q. Jiang, H. Li, F. N. Khan, and Z. Wang, “Cosmic: Clique-oriented semantic multi-space integration for robust clip testtime adaptation,” in CVPR, 2025.
- [219] C. Ding, X. Gao, S. Dong, Y. He, Q. Wang, X. Song, A. Kot, and Y. Gong, “Space rotation with basis transformation for training-free test-time adaptation,” arXiv preprint arXiv:2502.19946, 2025.
- [220] R. Wang, H. Zuo, Z. Fang, and J. Lu, “Prompt-based memory bank for continual test-time domain adaptation in vision-language models,” in IJCNN, 2024.
- [221] B. Tong, K. Song, and H. Lai, “Test-time alignment-enhanced adapter for vision-language models,” arXiv preprint arXiv:2411.15735, 2024.
- [222] R. Zhang, W. Zhang, R. Fang, P. Gao, K. Li, J. Dai, Y. Qiao, and H. Li, “Tip-adapter: Training-free adaption of clip for few-shot classification,” in ECCV, 2022.
- [223] T. Zhang, J. Wang, H. Guo, T. Dai, B. Chen, and S.-T. Xia, “Boostadapter: Improving vision-language test-time adaptation via regional bootstrapping,” in NeurIPS, 2024.
- [224] J. Zhang, J. Huang, X. Zhang, L. Shao, and S. Lu, “Historical test-time prompt tuning for vision foundation models,” in NeurIPS, 2024.
- [225] D.-C. Zhang, Z. Zhou, and Y.-F. Li, “Robust test-time adaptation for zero-shot prompt tuning,” in AAAI, 2024.
- [226] H. Zhai, X. Chen, C. Zhang, T. Sha, and R. Li, “Mitigating cache noise in test-time adaptation for large vision-language models,” in ICLR Workshops, 2025.
- [227] J. Yi, R. Pan, J. Yang, and X. Yang, “Mint: Memory-infused prompt tuning at test-time for clip,” arXiv preprint arXiv:2506.03190, 2025.

- [228] S. K. Maharana, B. Zhang, L. Karlinsky, R. Feris, and Y. Guo, “Batclip: Bimodal online test-time adaptation for clip,” in ICCV, 2025.
- [229] Z. Xiao, J. Shen, M. M. Derakhshani, S. Liao, and C. G. Snoek, “Anyshift prompting for generalization over distributions,” in CVPR, 2024.
- [230] S. Cui, J. Xu, Y. Li, X. Tang, J. Li, J. Zhou, F. Xu, F. Sun, and H. Xiong, “Bayestta: Continual-temporal test-time adaptation for vision-language models via gaussian discriminant analysis,” arXiv preprint arXiv:2507.08607, 2025.
- [231] Q. Dai and S. Yang, “Free on the fly: Enhancing flexibility in test-time adaptation with online em,” in CVPR, 2025.
- [232] H. Chen, Y. Xu, Y. Xu, Y. Zhang, and L. Cui, “Test-time medical image segmentation using clip-guided sam adaptation,” in BIBM, 2024.
- [233] M. Lafon, G. A. V. Hakim, C. Rambour, C. Desrosier, and N. Thome, “Cliptta: Robust contrastive vision-language test-time adaptation,” arXiv preprint arXiv:2507.14312, 2025.
- [234] H. Han, A. J. Wang, and F. Liu, “Negation-aware test-time adaptation for vision-language models,” arXiv preprint arXiv:2507.19064, 2025.
- [235] R. Yan, J. Wang, H. Qu, X. Du, D. Zhang, J. Tang, and T. Tan, “Test-v: Test-time support-set tuning for zero-shot video classification,” arXiv preprint arXiv:2502.00426, 2025.
- [236] Q. Zhang, M. Zhao, J. Liu, F. Zhang, Y. Xu, and Z.-J. Zha, “Hierarchical knowledge prompt tuning for multi-task test-time adaptation,” in CVPR, 2025.
- [237] B. Tong, H. Lai, Y. Pan, and J. Yin, “On the zero-shot adversarial robustness of vision-language models: A truly zero-shot and trainingfree approach,” in CVPR, 2025.
- [238] X. Chen, J. Huang, Q. Jiang, F. Huang, X. Fu, J. Jiang, and Z. Wang, “Small aid, big leap: Efficient test-time adaptation for vision-language models with adaptnet,” arXiv preprint arXiv:2506.02671, 2025.
- [239] K. Adachi, S. Yamaguchi, and T. Hamagami, “Uniformity first: Uniformity-aware test-time adaptation of vision-language models against image corruption,” arXiv preprint arXiv:2505.12912, 2025.
- [240] D. Sarkar, A. Chakrabartty, B. Bhanja, and A. Das, “Active test time prompt learning in vision-language models,” 2024. [Online]. Available: https://openreview.net/forum?id=pdzHpQbGrn
- [241] L. Fei-Fei, R. Fergus, and P. Perona, “Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories,” in CVPR Workshops, 2004.
- [242] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. Jawahar, “Cats and dogs,” in CVPR, 2012.
- [243] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3d object representations for fine-grained categorization,” in ICCV Workshops, 2013.
- [244] M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in ICVGIP, 2008.
- [245] L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101–mining discriminative components with random forests,” in ECCV, 2014.
- [246] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi, “Fine-grained visual classification of aircraft,” arXiv preprint arXiv:1306.5151, 2013.
- [247] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba, “Sun database: Large-scale scene recognition from abbey to zoo,” in CVPR, 2010.
- [248] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, “Describing textures in the wild,” in CVPR, 2014.
- [249] P. Helber, B. Bischke, A. Dengel, and D. Borth, “Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217–2226, 2019.
- [250] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012.
- [251] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in CVPR, 2009.
- [252] D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song, “Natural adversarial examples,” in CVPR, 2021.
- [253] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do imagenet classifiers generalize to imagenet?” in ICML, 2019.
- [254] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo et al., “The many faces of robustness: A critical analysis of out-of-distribution generalization,” in ICCV, 2021.
- [255] H. Wang, S. Ge, Z. Lipton, and E. P. Xing, “Learning robust global representations by penalizing local predictive power,” in NeurIPS, 2019.
- [256] H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan, “Deep hashing network for unsupervised domain adaptation,” in CVPR, 2017.

- [257] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang, “Moment matching for multi-source domain adaptation,” in ICCV, 2019.
- [258] M. Everingham, S. A. Eslami, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes challenge: A retrospective,” International Journal of Computer Vision, vol. 111, pp. 98–136, 2015.
- [259] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Urtasun, and A. Yuille, “The role of context for object detection and semantic segmentation in the wild,” in CVPR, 2014.
- [260] H. Caesar, J. Uijlings, and V. Ferrari, “Coco-stuff: Thing and stuff classes in context,” in CVPR, 2018.
- [261] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” International Journal of Computer Vision, vol. 127, pp. 302– 321, 2019.
- [262] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in ECCV, 2014.
- [263] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in CVPR, 2016.
- [264] M. Weber, J. Xie, M. Collins, Y. Zhu, P. Voigtlaender, H. Adam, B. Green, A. Geiger, B. Leibe, D. Cremers et al., “Step: Segmenting and tracking every pixel,” arXiv preprint arXiv:2102.11859, 2021.
- [265] F. Panella, V. Melatti, and J. Boehm, “Firenet dataset,” http://www. firenet.xyz, accessed: 2022-05-17.
- [266] H. Jiang, X. Ma, W. Nie, Z. Yu, Y. Zhu, and A. Anandkumar, “Bongard-hoi: Benchmarking few-shot visual reasoning for humanobject interactions,” in CVPR, 2022.
- [267] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.
- [268] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” 2011.
- [269] G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, and S. Belongie, “The inaturalist species classification and detection dataset,” in CVPR, 2018.
- [270] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, “Places: A 10 million image database for scene recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 6, pp. 1452– 1464, 2017.
- [271] I. Krasin, T. Duerig, N. Alldrin, V. Ferrari, S. Abu-El-Haija, A. Kuznetsova, H. Rom, J. Uijlings, S. Popov, A. Veit et al., “Openimages: A public dataset for large-scale multi-label and multi-class image classification,” Dataset available from https://github. com/openimages, vol. 2, no. 3, p. 18, 2017.
- [272] B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” in ICCV, 2015.
- [273] N. Rostamzadeh, S. Hosseini, T. Boquet, W. Stokowiec, Y. Zhang, C. Jauvin, and C. Pal, “Fashion-gen: The generative fashion dataset and challenge,” arXiv preprint arXiv:1806.08317, 2018.
- [274] S. Li, T. Xiao, H. Li, B. Zhou, D. Yue, and X. Wang, “Person search with natural language description,” in CVPR, 2017.
- [275] Z. Ding, C. Ding, Z. Shao, and D. Tao, “Semantically self-aligned network for text-to-image part-aware person re-identification,” arXiv preprint arXiv:2107.12666, 2021.
- [276] H. Agrawal, K. Desai, Y. Wang, X. Chen, R. Jain, M. Johnson, D. Batra, D. Parikh, S. Lee, and P. Anderson, “Nocaps: Novel object captioning at scale,” in ICCV, 2019.
- [277] D. S. Kermany, M. Goldbaum, W. Cai, C. C. Valentim, H. Liang, S. L. Baxter, A. McKeown, G. Yang, X. Wu, F. Yan et al., “Identifying medical diagnoses and treatable diseases by image-based deep learning,” Cell, vol. 172, no. 5, pp. 1122–1131, 2018.
- [278] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. W´ang, P.-X. Lu, and G. Thoma, “Two public chest x-ray datasets for computer-aided screening of pulmonary diseases,” Quantitative Imaging in Medicine and Surgery, vol. 4, no. 6, p. 475, 2014.
- [279] P. Porwal, S. Pachade, R. Kamble, M. Kokare, G. Deshmukh, V. Sahasrabuddhe, and F. Meriaudeau, “Indian diabetic retinopathy image dataset (idrid): a database for diabetic retinopathy screening research,” Data, vol. 3, no. 3, p. 25, 2018.
- [280] N. C. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler et al., “Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic),” in ISBI, 2018.

- [281] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, “Hmdb: a large video database for human motion recognition,” in ICCV, 2011.
- [282] J. Carreira, E. Noland, A. Banki-Horvath, C. Hillier, and A. Zisserman, “A short note about kinetics-600,” arXiv preprint arXiv:1808.01340, 2018.
- [283] F. Caba Heilbron, V. Escorcia, B. Ghanem, and J. Carlos Niebles, “Activitynet: A large-scale video benchmark for human activity understanding,” in CVPR, 2015.
- [284] H. Idrees, A. R. Zamir, Y.-G. Jiang, A. Gorban, I. Laptev, R. Sukthankar, and M. Shah, “The thumos challenge on action recognition for videos “in the wild”,” Computer Vision and Image Understanding, vol. 155, pp. 1–23, 2017.
- [285] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in ICLR, 2019.
- [286] V. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Y. Zou, “Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning,” in NeurIPS, 2022.
- [287] H. Dong, Y. Zhao, E. Chatzi, and O. Fink, “Multiood: Scaling out-ofdistribution detection for multiple modalities,” in NeurIPS, 2024.
- [288] S. Li, H. Gong, H. Dong, T. Yang, Z. Tu, and Y. Zhao, “Dpu: Dynamic prototype updating for multimodal out-of-distribution detection,” in CVPR, 2025.
- [289] M. Liu, H. Dong, J. Kelly, O. Fink, and M. Trapp, “Extremely simple multimodal outlier synthesis for out-of-distribution detection and segmentation,” arXiv preprint arXiv:2505.16985, 2025.
- [290] C. Mao, S. Geng, J. Yang, X. Wang, and C. Vondrick, “Understanding zero-shot adversarial robustness for large-scale models,” in ICLR, 2023.
- [291] L. Li, H. Guan, J. Qiu, and M. Spratling, “One prompt word is enough to boost adversarial robustness for pre-trained vision-language models,” in CVPR, 2024.
- [292] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” arXiv preprint arXiv:1706.06083, 2017.
- [293] H. Dong, X. Chen, M. Dusmanu, V. Larsson, M. Pollefeys, and C. Stachniss, “Learning-based dimensionality reduction for computing compact and effective local feature descriptors,” in ICRA, 2023.
- [294] A. Qayyum, J. Qadir, M. Bilal, and A. Al-Fuqaha, “Secure and robust machine learning for healthcare: A survey,” IEEE Reviews in Biomedical Engineering, vol. 14, pp. 156–180, 2020.
- [295] W. Bao, T. Wei, H. Wang, and J. He, “Adaptive test-time personalization for federated learning,” in NeurIPS, 2023.
- [296] J. Lee, D. Kim, and B. Ham, “Network quantization with element-wise gradient scaling,” in CVPR, 2021.
- [297] Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell, “Rethinking the value of network pruning,” arXiv preprint arXiv:1810.05270, 2018.
- [298] J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,” International Journal of Computer Vision, vol. 129, no. 6, pp. 1789–1819, 2021.
- [299] G. Kwon, Z. Cai, A. Ravichandran, E. Bas, R. Bhotika, and S. Soatto, “Masked vision and language modeling for multi-modal representation learning,” arXiv preprint arXiv:2208.02131, 2022.
- [300] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou, “Show-o: One single transformer to unify multimodal understanding and generation,” arXiv preprint arXiv:2408.12528, 2024.
- [301] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Cand`es, and T. Hashimoto, “s1: Simple test-time scaling,” arXiv preprint arXiv:2501.19393, 2025.
- [302] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling llm test-time compute optimally can be more effective than scaling model parameters,” arXiv preprint arXiv:2408.03314, 2024.
- [303] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.
- [304] I. Nejjar, Q. Wang, and O. Fink, “Dare-gram: Unsupervised domain adaptation regression by aligning inverse gram matrices,” in CVPR, 2023.
- [305] C. Yang, Y. Shen, Z. Zhang, Y. Xu, J. Zhu, Z. Wu, and B. Zhou, “One-shot generative domain adaptation,” in ICCV, 2023.
- [306] H. Li, P. Hu, Q. Zhang, X. Peng, X. Liu, and M. Yang, “Test-time adaptation for cross-modal retrieval with query shift,” in ICLR, 2024.
- [307] H. Park, A. Gupta, and A. Wong, “Test-time adaptation for depth completion,” in CVPR, 2024.
- [308] H. Dong, M. Liu, J. Liang, E. Chatzi, and O. Fink, “To trust or not to trust your vision-language model’s prediction,” arXiv preprint arXiv:2505.23745, 2025.

- [309] Z. Deng, Z. Chen, S. Niu, T. Li, B. Zhuang, and M. Tan, “Efficient test-time adaptation for super-resolution with second-order degradation and reconstruction,” in NeurIPS, 2023.
- [310] B. C. Kalp´elb´e, A. G. Adaambiik, and W. Peng, “Vision language models in medicine,” arXiv preprint arXiv:2503.01863, 2025.
- [311] K. Zhang, R. Zhou, E. Adhikarla, Z. Yan, Y. Liu, J. Yu, Z. Liu, X. Chen, B. D. Davison, H. Ren et al., “A generalist vision–language foundation model for diverse biomedical tasks,” Nature Medicine, pp. 1–13, 2024.
- [312] V. Rawte, A. Sheth, and A. Das, “A survey of hallucination in large foundation models,” arXiv preprint arXiv:2309.05922, 2023.

Hao Dong is a Ph.D. student at ETH Z¨urich, Z¨urich, Switzerland. He received the B.S. degree from Xi’an Jiaotong University, Xi’an, China, in 2020, and the M.S. degree from Aalto University, Espoo, Finland, in 2022. His research interest lies in multimodal learning and sensor fusion, including their applications in robotics, computer vision, and anomaly detection.

Lijun Sheng received the B.E. degree in Automation from the University of Science and Technology of China (USTC), in July 2020. He is now a Ph.D. candidate in the Department of Automation at the University of Science and Technology of China. His research interests include transfer learning, domain adaptation, and trustworthy AI.

Jian Liang received his B.E. degree in Electronic Information and Technology from Xi’an Jiaotong University in July 2013 and his Ph.D. degree in Pattern Recognition and Intelligent Systems from the National Laboratory of Pattern Recognition (NLPR), CASIA, in January 2019. From June 2019 to April 2021, he worked as a research fellow at the National University of Singapore. He is currently an associate professor at NLPR, CASIA. His research interests include transfer learning, pattern recognition, and computer vision.

Ran He received the BE degree in computer science from the Dalian University of Technology, in 2001, the MS degree in computer science from the Dalian University of Technology, in 2004, and the PhD degree in pattern recognition and intelligent systems from CASIA, in 2009. Since September 2010, he joined NLPR, where he is currently a full professor. His research interests include information theoretic learning, pattern recognition, and computer vision. He serves as the editor board member of IEEE TPAMI, TIP, TIFS, TCSVT, and TBIOM, and serves on the program committee of several conferences. He is also a fellow of IEEE and IAPR.

Eleni Chatzi received the Diploma and M.Sc. degrees in civil engineering from the Department of Civil Engineering, National Technical University of Athens (NTUA), Athens, Greece, in 2004 and 2006, respectively, and the Ph.D. degree (with distinction) from the Department of Civil Engineering and Engineering Mechanics, Columbia University, New York, NY, USA, in 2010. She is currently a Full Professor at the Chair of Structural Mechanics with the Department of Civil, Environmental and Geomatic Engineering, ETH Z¨urich, Z¨urich, Switzerland. Her research spans a broad range of topics, including applications on emerging sensor technologies and structural control, methods for curbing uncertainties in structural diagnostics and lifecycle assessment, as well as advanced schemes for nonlinear/nonstationary dynamics simulations.

Olga Fink received the Diploma degree in industrial engineering from the Hamburg University of Technology, Hamburg, Germany, in 2008, and the Ph.D. degree from ETH Z¨urich, Z¨urich, Switzerland, in 2014. She has been an Assistant Professor of Intelligent Maintenance and Operations Systems with EPFL, Lausanne, Switzerland, since March 2022. Before joining EPFL Faculty, she was an Assistant Professor of Intelligent Maintenance Systems with ETH Z¨urich from 2018 to 2022. From 2014 and 2018, she was heading the research group “Smart Maintenance” with the Zurich University of Applied Sciences, Winterthur, Switzerland. And, she is also a Research Affiliate with the Massachusetts Institute of Technology, Cambridge, CA, USA. Her research focuses on hybrid algorithms fusing physics-based models and deep learning algorithms, hybrid operational digital twins, transfer learning, selfsupervised learning, deep reinforcement learning, and multiagent systems for intelligent maintenance and operations of infrastructure and complex assets.

