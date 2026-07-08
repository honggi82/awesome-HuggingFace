[Figure 1]

[Figure 2]

[Figure 3]

## SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model

[Figure 4]

[Figure 5]

[Figure 6]

ByteDance Douyin SAIL Team, CUHK MMLab

# arXiv:2510.12709v3[cs.IR]2Nov2025

##### Abstract

Multimodal embedding models aim to yield informative unified representations that empower diverse crossmodal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. SAIL-Embedding supports multifaceted multimodal retrieval and classification by accommodating arbitrary modality inputs, includingtranscribedtextualinformation, sampledvisualsemantics, andacquirableaudiosignals. Toenhance training robustness and scalability, we introduce the dynamic hard negative mining and adaptive multi-source data balancing to consolidate domain expertise and capture effective multimodal representations. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model’s adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaborationaware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results demonstrate that SAIL-Embedding achieves state-of-the-art performance compared to other methods in item-to-item and query-to-item retrieval tasks across different intents. Furthermore, we provide comprehensive analysis and ablation studies to reveal the necessity of the proposed modules and components. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.5% in the Douyin-Selected scenario. Furthermore, through clustering quantification, the model is widely applied across diverse situations such as decentralization, recall, pre-ranking, and re-ranking. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.1% AUC gain.

Date: November 4, 2025

##### 1 Introduction

Embedding models aim to generate meaningful dense vector representations of data. From the early exploration of distributed word representations such as Word2Vec [43] and GloVe [45] to the recent development of large language model (LLM)-based embedding models such as Gemini-Embedding [33] and Qwen-Embedding [75], textual embedding models [18, 19, 39] have become a fundamental component of natural language processing and have

Copair-i2i

Live-i2i

For a relevant video retrieval

task, find out an image-textaudio triplet that shares the

Search-q2i

Tag-CLS

For a relevant live-streaming

retrieval task, given the livestreaming content with its

same theme as the given

For a relevant video retrieval task, based on

Motivation-CLS

video with its corresponding text and audio information.

For a video content labeling

text and audio information,

task, based on video screenshots and their

find out an image-text-audio triplet created by the same

several query keywords

This is a task of identifying user creative intent. Based on

provided by the user, find out a video with its text

[Figure 7]

Our baby meals are freshly

extracted content, such as

[Figure 8]

creator or with similar

[Figure 9]

OCR, identify the hierarchical tags that best represent the

creative techniques.

the videos , texts and audio

and audio information

sourced every day….

created by the user, find the label that best fits the user's

that best matches the user's intention.

[Figure 10]

content theme.

[Figure 11]

Let's do some exercise

[Figure 12]

[Figure 13]

intent.

[Figure 14]

Yoga is a practice

[Figure 15]

together…

# Standing Beneath the

[Figure 16]

Anyone playing

that nurtures both body and

[Figure 17]

Starry Sky # The Milky Way

[Figure 18]

games? I want a King of Glory

[Figure 19]

[Figure 20]

mind.

[Figure 21]

[Figure 22]

partner.

[Figure 23]

SAIL-Embedding

# Sports; # Comprehensive Sports;

# Purely social

# Finding a partner

[Figure 24]

# Yoga

[Figure 25]

[Figure 26]

Standing under

[Figure 27]

It's almost time

[Figure 28]

[Figure 29]

the Milky Way, gazing into the

to go offline..

Babies raised

[Figure 30]

[Figure 31]

distance

[Figure 32]

[Figure 33]

[Figure 34]

exclusively on breast milk….

[Figure 35]

- Figure 1 SAIL-Embedding Capability Overview. We present SAIL-Embedding, an omni-modal embedding model adapted from vision-language-audio models. SAIL-Embedding is capable of following instructions and performing various omni-modal embedding tasks, such as Motivation/Tag-CLS (Classification), Search-q2i (Query-to-Item Retrieval), and Copair/Live-i2i (Item-to-Item Retrieval).

enabled a wide range of downstream applications, like retrieval-augmented generation (RAG) [24, 28] and information extraction [31, 44, 52]. With the rapid progress of multimodal learning [3, 20, 34, 55, 56, 59, 64, 65, 68], multimodal embedding models [4, 27, 29, 40, 42, 50, 60, 74, 76] that map heterogeneous modalities into a unified vector space have also emerged as an active research direction, showing great promises in practical scenarios such as short-video, image recommendation, as well as cross-modal search.

Current multimodal embedding models generally follow two design paradigms, as illustrated in Figure 2. The first paradigm, exemplified by CLIP [47] and SigLIP [72], leverages paired multimodal data and employs two large encoders to process each modality independently, with either no fusion or only shallow fusion layers. This design is structurally simple and has proven highly effective for cross-modal retrieval. However, its reliance on shallow fusion restricts the expressiveness and semantic richness of the resulting embeddings. With the recent advances in large language models (LLMs) and multimodal LLMs (MLLMs), a second paradigm has gained increasing attention, as shown in Figure 2(b). This line of work integrates LLMs/MLLMs to achieve deep semantic fusion across modalities [4, 29, 40, 74]. Although still under active investigation, this paradigm has already exhibited strong representational capacity and is rapidly emerging as the de facto choice for a wide range of downstream tasks [17].

However, when deployed for real-world scenarios, the models still face significant limitations that hinder their effectiveness. (1) Limited Modalities: Most existing methods rely on only two modalities—typically images and text—for unimodal or cross-modal retrieval. In contrast, industrial applications often require richer and more comprehensive multimodal representations. For example, a Douyin video contains diverse sources of information: visual signals from the cover frame and keyframes, textual cues from tags and captions, background music as audio, and spoken content transcribed via automatic speech recognition (ASR). Each modality contributes essential semantic information, and omitting or misinterpreting even one can severely impact downstream tasks such as recommendation and search, ultimately degrading user experience. (2) Training Instability: These models are typically built upon multimodal large language models, which require careful architectural design and optimization strategies to ensure stable and efficient training. Developing optimization strategies that unlock the practical values of incentive models within business is an indispensable key. (3) Industrial Domain Gap: Many models are trained and evaluated primarily on open-source datasets, but they often underperform on domain-specific industrial data, such as expressive short videos on Douyin, where data distributions and task requirements differ substantially from academic benchmarks.

To this end, we propose SAIL-Embedding, a powerful omni-modal embedding model of the SAIL families [13, 70],

Text emb.

Vision emb.

ITC / IIC / VTC

Multimodal emb.

Multimodal emb.

ITC / IIC TTC Tasks

IIC / VVC / TTC OOC Tasks

Contrast

Vision Encoder

Text Encoder

LLM/MLLM

LLM/MLLM

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

A girl with pink

[Figure 42]

A girl with pink

A girl with pink hair

[Figure 43]

hair holding a gray

hair holding a gray

holding a gray dog

dog in her arms.

dog in her arms.

in her arms.

Vision Modality Text Modality

Instruction Vision/Text Modality

Instruction Vision/Text/Audio Modality

(a) (b) (c)

- Figure 2 Embedding Architecture Comparison. (a) CLIP-like dual-tower embedding model architecture. (b) LLM/MLLM-based embedding model architecture. (c) SAIL-Embedding model architecture. In contrast, our model accommodates arbitrary modality inputs and can handle diverse downstream tasks. I, V, T, and O represent image, video, text, and omni-modality, respectively.

along with tailored training strategies to ensure stable large-scale optimization and balance modality contributions. SAIL-Embedding can handle arbitrary combinations of inputs from vision, text, and audio modalities, yielding multi-dimensional representation vectors to fulfil diverse real-world business requirements.

To further enhance the model’s robustness during the training process and improve its scalability to handle large-scale and diverse datasets, we introduce dynamic hard negative mining and adaptive multi-source data balancing. The dynamic hard negative mining helps the model focus on distinguishing challenging negative samples, thereby consolidating the model’s understanding of domain-specific knowledge and reducing the risk of misclassification caused by ambiguous samples. Meanwhile, the adaptive multi-source data balancing dynamically learns weights directly from the data distribution to reduce reliance on manual parameter tuning and maintain the trade-off between data quality and distributional diversity.

SAIL-Embedding presents the multi-stage training procedure that learns unified representations across multiple dimensions through content- and collaboration-awareness. Specifically, content-aware progressive training gradually enhances the embeddings’ discriminative power for diverse task demands and the generalization ability to handle unseen scenarios by leveraging diverse, semantically rich data resources. This process endows the model with comprehensive domain knowledge capabilities, including cross-modal semantic integration, scenario-specific content understanding, and complex concept relation. During the collaboration-aware recommendation enhancement phase, we perform multi-dimensional interest-driven sequence-to-item distillation to incorporate the users’ historical behavioural patterns into the multimodal representations. Subsequently, the combined ID-to-item distillation further aggregates user-specific preference signals within the recommendation system, thereby improving the accuracy of item recommendations.

Extensive experimental results are conducted on multiple benchmark datasets covering different application scenarios, demonstrating that the proposed SAIL-Embedding achieves state-of-the-art (SOTA) performance compared to other advanced baseline methods (i.e., traditional unimodal models, dual-tower fusion models, and large vision-language representation methods) in diverse downstream tasks.

In online experiments, SAIL-Embedding demonstrates its effectiveness in the recommender system of Douyin, which brings substantial gains through diverse application pathways. For cold-start scenarios, our model achieves a 0.05% LT gain in total by performing embedding-based item2item (i2i) recall along with engaging embeddings into the recommendation model. Specifically, SAIL-Embedding improves the AUC of the cold-start model by 0.1% when deployed as a target side feature. We also discretize embeddings info semantic tokens, which ultimately deliver ∼0.03% LT gain across different stages of the recommender system, including recall, pre-ranking and re-ranking. Specifically, the AUC of the ranking model can be improved by ∼0.1% with the engagement of both embeddings and sematic tokens.

##### 2 Related Works

We have witnessed remarkable progress in multimodal learning in recent years [2, 3, 6, 20, 22, 37, 38, 57, 58, 66]. Among various research directions, multimodal embedding learning has emerged as a fundamental paradigm, aiming to project heterogeneous modalities—such as images, text, audio, and video—into a shared representation space.

Such unified embeddings enable a broad range of downstream tasks, including cross-modal retrieval [60, 76], video understanding [62, 67], and recommendation [9]. Existing approaches can be broadly categorized into two families: (1) dual-tower architectures, which employ independent modality-specific encoders to map each modality to the joint space; and (2) multimodal large language model (MLLM)-based architectures, which integrate all modalities into a unified Transformer framework for joint modeling.

###### 2.1 Dual-Tower Multimodal Embedding Models

Dual-tower architectures, pioneered by CLIP [47], employ separate encoders for each modality (e.g., a Vision Transformer [14] for images and a Transformer-based language model [54] for text), projecting them into a common embedding space. Training typically relies on large-scale contrastive learning, maximizing the similarity between paired samples and minimizing it for mismatched pairs. This decoupled design allows each encoder to be precomputed and cached, enabling highly efficient retrieval at inference time and scaling well to billions of items. Following CLIP, numerous extensions have been proposed. ALIGN [26] scales the model capacity and dataset size to improve representation quality. AudioCLIP [21] incorporates an additional audio branch, while CLIP4Clip [41] adapts the architecture for video-text retrieval by encoding temporal information. More recently, BLIP [35] and BLIP-2 [36] bridge dual-tower and fusion-style paradigms: BLIP adopts a bootstrapped pre-training strategy that unifies vision-language understanding and generation, whereas BLIP-2 introduces a lightweight Q-Former to better align visual features with frozen large language models. SigLIP [71] further improves contrastive training by replacing the softmax cross-entropy with a sigmoid loss, mitigating the inefficiencies of batch-dependent normalization and allowing more stable training on large-scale noisy datasets. Other works [11, 30] investigate robust pretraining on noisy web-scale data by leveraging advanced filtering, tokenization, and data curation strategies. Despite their efficiency and scalability, dual-tower architectures generally fuse modalities only in the final embedding space, which limits their ability to capture fine-grained token-level interactions, temporal dynamics, or higher-level multimodal reasoning required in complex tasks.

###### 2.2 MLLM-based Multimodal Embedding Models

In contrast, MLLM-based approaches [4, 27, 29, 40, 42, 50, 60, 74, 76] aim to integrate all modalities within a unified sequence modeling framework, leveraging the generative and reasoning capabilities of large language models. Typically, modality-specific encoders (e.g., visual or audio spectrogram encoders) transform raw inputs into embeddings, which are then aligned to the language token space via linear projections or learned adapters. Representative works include VLM2Vec [29] and GME [74]. VLM2Vec [29] generates fixed-dimensional embeddings for arbitrary combinations of images and text under task instructions, building upon Phi-3.5-V [1]. Its successor, VLM2Vec-v2 [42], further extends the framework to support videos and visual documents. GME [74] constructs an MLLM-based dense retriever to enable unified cross-modal search. Subsequent research has sought to improve this paradigm: mmE5 [5] leverages synthetic datasets for stronger multilingual performance, MoCa [4] introduces bidirectional attention through continual pre-training to enhance scalability with both model size and training data, UniMoCo [46] proposes a modality-completion module that infers visual features from text to address modality-missing issues, UniME [16] employs discriminative knowledge distillation from a powerful LLM teacher to improve the embedding quality of the language component, and NoteLLM-2 [73] explores leveraging multimodal large representation models for recommendation.

By naturally modeling cross-modal dependencies through Transformer-based self-attention [54], this paradigm enables deeper semantic understanding, contextual reasoning, and compositionality—capabilities that dual-tower models often lack. Nonetheless, most existing methods remain constrained to image and text modalities, falling short of fully supporting omni-modal understanding.

##### 3 Methodology

In this section, we present the construction of the proposed SAIL-Embedding model. We first describe the collected datasets in § 3.1, together with strategies for balancing heterogeneous data and a hard negative mining method to strengthen representation learning. We then introduce the model architecture in § 3.2, highlighting the fusion of different modalities and the use of prompts to fully leverage the capabilities of multimodal large language models. Finally, we define the training objectives in § 3.3 and outline several techniques designed to improve the training effectiveness.

###### 3.1 Data Preparation and Preprocessing

###### Meta Task Data Partition Design Philosophy Query Target Data Magnitude

Copair-i2i User consumption behaviour-based pair V+T+A V+T+A 2.9B Search-i2i User search behaviour-based pair V+T+A V+T+A 0.6B

Live-i2i Live streaming-based behavioral copair V+T+A V+T+A 1.4B Summary-i2i Ngram-based filtering pair V+T+A V+T+A 1.1B

Item to Item Retrieval

Hashtag-i2i User submission-based pair V+T+A V+T+A 0.05B RSDF-i2i RSDF similarity-based filtering pair V+T+A V+T+A 0.3B

ID-i2i ID clustering-based pair V+T+A V+T+A 0.08B Query to Item Retrieval

Search-q2i Search query and click-based pair T V+T+A 1.8B

Score-q2i Pairs of queries and targets with labeled similarity scores for downstream training T V+T+A 0.6B Classification CLS Item information and corresponding multi-level tags V+T+A T 3.1B

Table 1 SAIL-Embedding Training Data Statistics. V, T, and A represent vision, text, and audio modalities, respectively. The training data encompasses multi-faceted retrieval tasks across queries and items with omni-modal information, as well as multi-level label classification tasks. “RSDF” means the Recommendation-side Dense Features.

###### 3.1.1 Recommendation-aware Data Construction

The data construction goal for SAIL-Embedding is to enable the model to provide omni-modal understanding capabilities for practical applications, meeting recommendation demands across diverse scenarios, such as Douyin videos and Douyin live. We curate a large-scale dataset of over 10B samples, with statistical details provided in Table 1. Mostly, each sample is a pair consisting of a query and a target for CLIP-like contrastive learning. The query can be a video or just a couple of words, as well as the target. Different training datasets are designed with specific philosophies to encompass diverse content and collaboration semantics. To this end, the meta-task categories for training data are summarized as follows:

- 1. Item-to-Item Retrieval: Typically, an item serves as a video containing omni-modal information, encompassing vision, text, and audio modalities. This type of task performs retrieval of candidate items based on various requirements, including user behaviors, video summaries, semantic IDs, and specific business applications.
- 2. Query-to-Item Retrieval: In this task, the query only contains short texts for item search. And the target is a video that is mostly clicked by users from the search results list. Based on these pairs, we further use rule-based and LLM-based methods to generate similarity scores between each query and target, labeling for tasks like the COSENT training [51].
- 3. Classification: The query is typically a given item, with the target being the corresponding multi-level tags. Inspired by UniCL [69] and iCLIP [61], we transform classification datasets into item–label text pairs. This type of task spans a wide range of scenarios to serve different tag recognition needs, such as user motivation and typical image-text classification.

###### 3.1.2 Dynamic Hard Negative Mining

In contrastive learning, the effectiveness of representation learning heavily depends on the quality of both positive and negative samples. While random negatives are often abundant, they tend to be semantically dissimilar to the query, making the discrimination task trivial and limiting the model’s ability to capture fine-grained distinctions. Hard negatives—samples that are challenging to distinguish from positives due to high semantic similarity—play a crucial role in improving model robustness and retrieval performance. However, the notion of “hard” is dataset- and task-dependent, and applying a fixed global similarity threshold often leads to suboptimal results. To address this, we propose a dynamic hard negative mining strategy that adaptively determines the optimal similarity threshold for each dataset.

Formally, let P = {(𝑞𝑖, 𝑡𝑖)}𝑖𝑁=1 denote the set of positive pairs, where 𝑞𝑖 is a query and 𝑡𝑖 is its corresponding target. Negative pairs are constructed via a Cartesian product excluding positives:

###### N = {(𝑞𝑖, 𝑡𝑗) | 𝑖 ≠ 𝑗, (𝑞𝑖, 𝑡𝑗) ∉ P}.

We then compute cosine similarity scores 𝑠𝑖 𝑗 = cos(𝑞𝑖, 𝑡𝑗) for all negative pairs (𝑞𝑖, 𝑡𝑗) ∈ N. Merging positive and negative pairs, we obtain a dataset of (𝑠, 𝑦), where 𝑦 ∈ {1 (positive), 0 (negative)}. For each candidate threshold 𝜆, the

binary prediction is defined as:

𝑦ˆ𝑖 𝑗(𝜆) =

1, 𝑠𝑖𝑗 ≥ 𝜆, 0, 𝑠𝑖𝑗 < 𝜆.

Precision and recall are computed accordingly, and the F1 score is defined as:

2 · Precision(𝜆) · Recall(𝜆) Precision(𝜆) + Recall(𝜆)

F1(𝜆) =

.

The optimal similarity threshold is selected by:

𝜆∗ = argmax

F1(𝜆),

𝜆

which is then used to identify and mine hard negatives dynamically during training. This adaptive strategy ensures datasetand task-specific selection of challenging negatives, improving contrastive representation learning and downstream retrieval performance.

Once 𝜆∗ is determined, we construct the hard negative set H by selecting samples whose similarity scores are less than 𝜆∗ yet among the highest below this threshold:

H = {𝑥𝑖 | 𝑠(𝑥𝑖) < 𝜆∗, 𝑠(𝑥𝑖) is among the highest below 𝜆∗} , (1)

where 𝑠(𝑥𝑖) denotes the similarity score of sample 𝑥𝑖. This filtering step removes overly similar samples, which we regard as false negatives. For each query 𝑞𝑖, we sample positives from P, hard negatives from H and in-batch random negatives from N. The contrastive loss is then defined as:

###### ∑︁𝑁

exp(𝑠𝑖𝑖/𝜏) exp(𝑠𝑖𝑖/𝜏) + (𝑞𝑖,𝑡𝑗)∈H exp(𝑠𝑖𝑗/𝜏) + (𝑞𝑖,𝑡𝑗)∈N exp(𝑠𝑖 𝑗/𝜏)

log

Lcontrast = −

,

𝑖=1

where 𝑠𝑖𝑖 is the similarity of the positive pair, 𝜏 is a temperature hyper-parameter. This formulation adaptively enforces discrimination against semantically close but incorrect targets, thereby improving robustness and generalization of the learned representations.

###### 3.1.3 Adaptive Multi-Source Data Balancing

Conventional multi-source training pipelines often rely on manually assigned dataset mixing ratios, determined by subjective expertise and task intuition. Such heuristic configurations are difficult to validate empirically and may lead to suboptimal generalization. We introduce an adaptive weighting framework that learns dataset-specific sampling weights directly from the data distribution, rather than from human-designed heuristics. The core idea is to measure the semantic similarity between high-quality benchmark validation sets and pre-training datasets, and to translate this similarity into flexible sampling weights for multi-source training. Compared with hard filtering approaches that remove entire samples based on a single-instance similarity score, our method performs soft selection at the dataset level, preserving distributional diversity while maintaining overall data quality. This prevents overfitting to the benchmark domain and avoids distribution collapse, thereby improving generalization to unseen tasks.

Our pipeline works as illustrated in Figure 3. We first construct validation subsets from the training datasets with distributions similar to the downstream test tasks. Given multiple training datasets and one or more benchmark validation datasets, we extract embeddings using an early version of our model. To reduce computation, we randomly sample approximately 10k samples per dataset and apply 𝑘-means clustering to obtain cluster centroids. For each dataset pair, we compute a cosine similarity matrix 𝐶 between their centroids. We then reduce 𝐶 to a scalar similarity score via the Sinkhorn algorithm [8], i.e., a weighted sum 𝑃 ⊙ 𝐶, where 𝑃 is a transport matrix derived from the distance matrix 1 − 𝐶 using the Sinkhorn algorithm. This formulation assigns higher weights to more similar clusters. Next, we apply a “fusion first” strategy to determine the optimal modality combination, preferring fused modalities when available, and otherwise selecting unimodal representations. Finally, we compute the similarity between each training set 𝑀𝑖 and each

Training Datasets

Cluster center of

Cluster center of

| |
|---|

| |
|---|

| | |
|---|---|
| | |

fusion embed. Cluster center of audio embed.

vision embed. Cluster center of text embed.

Fusion sim.

[Figure 44]

Fusion matrix

- A
- B
- C

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

| |
|---|

| |
|---|

Dot product

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Aggregation:Sinkhorn

Vision sim.

|Vision matrix|
|---|

[Figure 49]

Final score

| | | |
|---|---|---|
| | | |
| | | |

[Figure 50]

Similarity

[Figure 51]

Matrix

Samples

[Figure 52]

[Figure 53]

- A
- B
- C

Audio matrix

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

Sum by row Softmax

| | |
|---|---|
| | |

Audio sim.

[Figure 54]

[Figure 55]

[Figure 56]

|Text matrix|
|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

…

| | |
|---|---|
| | |

Text sim.

Validation Datasets

Weight of each training dataset

Cluster centers of emb.

- Figure 3 Illustration of Adaptive Multi-Source Data Balancing. For both training and validation datasets, we compute embeddings for different modalities and cluster them. We then construct a similarity matrix to represent the distance between training and validation clusters, which is further aggregated into a single similarity score. Based on these similarity values, each training dataset is assigned a weight using a Softmax function.

benchmark 𝑁𝑗, producing an 𝑚 × 𝑛 similarity matrix. Averaging and normalizing across benchmarks yields the final adaptive sampling weight for each training set.

This method offers three benefits: (1) it learns weights directly from data distributions, reducing reliance on subjective manual tuning; (2) it is modular and can be generalized to other multi-source learning settings; (3) it maintains a balance between high data quality and distributional diversity, mitigating overfitting and enhancing robustness.

###### 3.2 Architecture Design

###### 3.2.1 Overall Architecture

The core idea of SAIL-Embedding is to transform heterogeneous multimodal information into a unified embedding space, enabling robust cross-modal understanding and retrieval. As illustrated in Figure 4, given an input sample 𝑥 containing audio 𝑎, visual 𝑣, and textual 𝑡 signals, our framework leverages a large language model (LLM) [53, 63] as the central reasoning and integration backbone, which is warmed up following the work [13].

For the text modality, we adopt conventional preprocessing pipelines, including tokenization and mapping each token to its corresponding word embedding via a trainable embedding layer. For visual and audio modalities, we follow the “foreign language” metaphor: each non-text modality is processed by a modality-specific encoder, namely E𝑣 for vision and E𝑎 for audio, to project their raw features into a natural language-compatible embedding space. These embeddings are then aligned in both dimension and semantics before being fed into the LLM for multimodal fusion. The final representation is extracted from the output of the LLM using mean pooling over all token embeddings.

This design allows for flexible integration of diverse modalities and provides a unified interface for knowledge transfer from large pre-trained language models to multimodal scenarios, without requiring extensive architecture modifications.

###### 3.2.2 Text Tokenizer

Real-world short videos contain abundant textual information, such as titles, tags, author labels, OCR texts, and ASR texts. The mainly used fields are described as follows:

1. Title: Title text information of the short video.

Mean Pooling

###### Fusion Module (LLM)

| |
|---|

| |
|---|

#### }

Contrastive Loss

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

SAIL-Embedding

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Target Sample

###### Tokenizer

###### Vision Perceiver

[Figure 65]

Average-based aggregation

[Figure 66]

[Figure 67]

For relevant video retrieval task, find out an image-text-audio triplet that shares the same theme as the given video with its corrseponding text and audio information.

Title Text Nick Name

Vision Encoder

Query tokens

Audio Encoder

Tags

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

ASR Text OCR Text

Sampled Frames

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Vision Modality Audio Modality

Query Sample

Text Modality

Instruction

- Figure 4 Overview of Our SAIL-Embedding Model. For each sample, we extract relevant textual information (e.g., titles, OCR text) and combine it with instructions to obtain textual tokens. Visual and audio information are encoded using dedicated encoders, and the resulting multimodal tokens are concatenated and passed through a fusion module to obtain a unified multimodal embedding. The system is trained by contrasting embeddings from query and target samples.

- 2. OCR Text: Textual information recognized from the short video frames.
- 3. ASR Text: Textual information obtained by converting audio from the short video.
- 4. Nickname: The author’s nickname associated with the short video.
- 5. Tags: Text labels generated by tagging models for each item.

We organize these signals into a unified textual format and concatenate them for downstream modeling. Specifically, the title, ASR, and OCR text from the query and candidates, along with additional nickname and tags, are selected as the text-modal data. The processing workflow first performs cleaning and deduplication, followed by the random dropping of partial fields in practice to accommodate scenarios where original fields are missing during online deployment.

Moreover, to effectively leverage the knowledge of the multimodal language model, we design prompts that guide the model to process multimodal information more effectively. As illustrated in Figure 5. Each prompt consists of three parts. The first is the system prompt, which defines the task and specifies its objective, helping the model better interpret and represent subsequent content. Next, we insert the extracted image, text, and audio tokens as user-provided information. Finally, we append the assistant symbol to indicate that the model should generate an answer. In addition, since tasks vary, the instructions for query and target samples are carefully designed to differ, enabling the system to interpret the provided information appropriately for each case.

###### 3.2.3 Vision Encoding Module

For the visual encoding, we adopt a Vision Transformer (ViT)-based backbone [70]. In the case of video data, each frame is independently patchified, generating a sequence of spatio-temporal tokens. We use a patch size of 14 and resize all frames to a uniform resolution to address resolution variation. Although such dense tokenization preserves fine-grained visual details, it also produces an excessive number of tokens, especially for high-resolution or long-duration videos, thereby imposing a substantial computational burden on downstream fusion.

To address this challenge, we introduce a Visual Perceiver module, inspired by perceiver architectures [25], serving as a learnable bottleneck for token reduction. Concretely, we concatenate the visual tokens with 𝑁𝑞 = 16 learnable latent query tokens, feed them into a Transformer block, and only retain the query token embeddings as the condensed visual

Query

Query

Query

<|im_start|>system

<|im_start|>system

<|im_start|>system

For a relevant video retrieval task, find out an image-text-audio triplet

This is a live content labeling task.

This is a related video content retrieval task.

Based on the live screenshots and

Based on the user's query keywords, the video screenshots and their text and audio

that shares the same theme as the

their text and audio, find the hierarchical label that best represents

given video with its corresponding

information that best match the user's

text and audio information. <|im_end|> <|im_start|>user

the content theme.

intent are extracted.

<|im_end|> <|im_start|>user {audios} {images} {texts}

<|im_end|>

<|im_start|>user

{audios} {images} {texts}

{texts}

<|im_end|>

<|im_end|> <|im_start|>assistant\n<|endoftext|>

<|im_end|>

<|im_start|>assistant\n<|endoftext|>

<|im_start|>assistant\n<|endoftext|>

Target

Target

Target

<|im_start|>system

<|im_start|>system

<|im_start|>system

For a relevant video retrieval task,

This is a live content labeling task.

You are an intelligent query system that

find out an image-text-audio triplet

Given a hierarchical content label, find

refers to the video screenshots and text and

that shares the same theme as the given video with its corresponding

the live screenshots and their text and

audio information browsed by the user, and

audio that match the theme. <|im_end|> <|im_start|>user

recommends query keywords that are most likely to retrieve relevant results to the user.

text and audio information.

<|im_end|> <|im_start|>user {audios} {images} {texts}

<|im_end|> <|im_start|>user {audios} {images} {texts}

{texts} <|im_end|>

<|im_end|> <|im_start|>assistant\n<|endoftext|>

<|im_end|> <|im_start|>assistant\n<|endoftext|>

<|im_start|>assistant\n<|endoftext|>

I2I Task CLS Task Q2I Task

- Figure 5 Illustration of instructions for different tasks. We design task-specific instructions by explicitly defining the task and its objective. Audio, image, and text tokens are then provided jointly to obtain the final results. For query and target tokens, modality-specific adaptations are applied to accommodate their respective modality combinations.

representation. This mechanism preserves essential semantic content while significantly reducing sequence length, thereby improving both efficiency and scalability.

###### 3.2.4 Audio Encoding Module

The audio modality processing typically has effective candidates, such as Qwen-audio [7], Whisper [48], and CLAP models [15]. We empirically employ the CLAP model, which is faster than models like Whisper, to balance the efficiency of online deployment with mitigating the long-tail distribution issue in post-sampling audio sequence lengths. This method can extract high-level acoustic semantics to obtain discriminative audio tokens. The pipeline is designed to accommodate diverse audio clip lengths:

- 1. Short audio (≤ 10s): We apply a repeat-and-pad operation to normalize the length before feature extraction, producing a single 1 × dim audio token.
- 2. Long audio (> 10s): We segment the waveform into consecutive non-overlapping 10s chunks, extract features for each chunk with CLAP, and aggregate them (via mean pooling) into a unified 1 × dim representation.

The resulting single audio token is appended to the multimodal token sequence before fusion, ensuring a consistent representation format regardless of the clip length.

###### 3.2.5 Fusion Module

Let T𝑎, T𝑣, and T𝑡 denote the token sequences obtained from the audio, visual, and textual modalities, respectively. We concatenate the unimodal token sequences to form a single multimodal sequence T = [T𝑎;T𝑣;T𝑡], where [· ; ·] denotes concatenation along the token dimension. The fused sequence T is then fed into a multimodal Transformer fusion module (i.e., an LLM backbone [63]), denoted as F (·), which performs cross-modal reasoning via self-attention layers. To strengthen information exchange across modalities and capture long-range dependencies, we employ a bi-directional attention mechanism. The resulting hidden states are further normalized by a tanh activation, ensuring embeddings

###### Stage I Stage II Stage III

Dataset A

Dataset B Dataset C

[Figure 80]

Sample

[Figure 81]

[Figure 82]

- Batch 0
- Batch 1
- Batch 2

Batch 0

[Figure 83]

- Dataset A

Dataset C

- Dataset B

[Figure 84]

[Figure 85]

[Figure 86]

- Batch 1
- Batch 2

[Figure 87]

Large-scale

Selected

Hard Negative;

Diverse Data

Subset

Clean Subset

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Base Model Finetuned Model Refined Model

(a) (b) (c)

- Figure 6 Illustration of Training Techniques. (a) Our progressive training framework gradually shifts from larger, diverse datasets to smaller, domain-specific datasets, balancing general-world knowledge with downstream specialization while maintaining training stability. (b) The conventional training method mixes heterogeneous data into a single batch. (c) Our stochastic specialization training randomly selects a dataset at each iteration to enhance robustness and specialization.

remain within a bounded range, and subsequently aggregated by mean pooling to form the final multimodal embedding: z = Meanpool tanh(F (T)) .

###### 3.3 Training Strategies

###### 3.3.1 Content-Aware Progressive Training

To adapt general-purpose large models to specific downstream tasks, we adopt a content-aware progressive training framework. As illustrated in Figure 6, each stage uses datasets with different scales and characteristics: earlier stages employ larger and more diverse datasets, while later stages focus on higher-quality data that closely matches the downstream task requirements. Specifically, in the first stage, we train a base model on large-scale and diverse datasets to acquire fundamental multimodal representation capabilities. Next, we fine-tune the model on a subset of datasets more aligned with the target downstream tasks. To further enhance the model’s ability to capture fine-grained distinctions, we construct hard negatives and perform an additional fine-tuning stage to obtain the refined model. This progressive training framework enables the model to balance general-world knowledge with downstream domain-specific knowledge, while maintaining training stability.

###### 3.3.2 Loss Definition

Our training objective follows the contrastive learning paradigm to jointly optimize the multimodal embedding space. The overall loss integrates four complementary components: (i) a Noise-Contrastive Estimation (NCE) loss for global alignment, (ii) a COSENT loss [51] for fine-grained ranking, (iii) a multimodal In-Context Learning (mICL) loss [73] to enhance modality-specific discrimination, and (iv) a late fusion loss [73] to balance visual and textual contributions.

Given a query embedding 𝑒𝑞 and its positive target embedding 𝑒+𝑡 , the NCE loss enforces global alignment by contrasting against in-batch negatives:

exp cos(𝑒𝑞, 𝑒+𝑡 )/𝜏 exp cos(𝑒𝑞, 𝑒+

Lnce = −log

, (2)

𝑡 )/𝜏 + 𝑖 𝐵=1 exp cos(𝑒𝑞, 𝑒−

𝑡 )/𝜏

where 𝜏 is a learnable temperature parameter, initialized differently for each task and each dataset to account for distributional variations.

In real-world applications, storage and computational budgets impose strict constraints on the dimensionality of embedding vectors, making the deployment of high-dimensional representations prohibitive due to memory cost and retrieval latency. To overcome this limitation, we employ Matryoshka Representation Learning (MRL) [32], which enforces multi-granularity supervision on embeddings of different sizes. Specifically, the full 1536-dimensional

embedding is sliced into multiple contiguous sub-vectors of sizes 768 and 128, and each slice, together with the full embedding, is optimized using the same InfoNCE loss objective:

Lnce−mrl = Lnce(z128) + Lnce(z768) + Lnce(z1536),

where z𝑑 denotes the sub-embedding of dimensionality 𝑑. This “nested” training paradigm ensures that all subembeddings preserve strong discriminative power, enabling the system to dynamically trade off accuracy and efficiency at inference time without retraining. Such flexibility is particularly critical for deploying resource-constrained multi-modal embedding models.

For text-oriented pre-training datasets, we incorporate a cosine-similarity–based ranking loss (COSENT) [51] to enforce fine-grained orderings:

Lcosent = log 1 + ∑︁

exp cos(𝑒𝑘,𝑒𝑙)−𝜏cos(𝑒𝑖,𝑒𝑗) . (3)

sim(𝑖, 𝑗)>sim(𝑘,𝑙)

To mitigate modality imbalance, we adopt the multimodal In-Context Learning (mICL) [73]. Instead of compressing multimodal inputs into a single embedding, we separate them into modality-specific embeddings (visual n𝑣 and textual n𝑡), and apply in-batch contrastive learning to encourage consistent alignment across modalities:

Lmicl = L(n𝑣, n+

𝑣) + L(n𝑡, n+

𝑡 ), (4) where each term follows the NCE form but within its respective modality. In addition, we integrate a late fusion mechanism [73] to preserve visual fidelity. Given visual embedding v and multimodal embedding n𝑚, a gated fusion module learns to adaptively combine them:

z = 𝜎 𝑊[v, n𝑚] + 𝑏 , nˆ𝑚 = z ⊙ v + (1 − z) ⊙ n𝑚, (5)

where 𝜎 denotes the sigmoid function. We then introduce a late-fusion contrastive loss Llf over nˆ𝑚 to reinforce multimodal consistency. The final training loss is a weighted combination:

L = Lnce−mrl + 𝜆 Lcosent + 𝛼 Lmicl + 𝛽 Llf, (6) where 𝜆, 𝛼, and 𝛽 are balancing hyperparameters.

###### 3.3.3 Stochastic Specialization Training

In multi-domain training scenarios, as illustrated by Figure 6 (b), data from heterogeneous datasets are often mixed within each iteration to ensure balanced utilization. However, such domain mixing inevitably fragments the effective batch size for each individual dataset, leading to small per-domain batches and elevated gradient variance. Moreover, synchronizing features or statistics across different domains within a single iteration introduces additional communication and processing overhead, which becomes more severe as the number of datasets grows.

To address this, we propose Stochastic Specialization Training, a strategy inspired by meta-learning schemes that improves both supervision focus and computational efficiency. Instead of sampling from all datasets in every iteration, our method stochastically selects a single dataset according to a predefined probability distribution based on methods introduced in Section 3.1.3, and draws the entire batch from it. Across training, all datasets are still visited, but each iteration specializes on one domain, yielding larger per-domain batch sizes while keeping the global batch size unchanged.

This specialization reduces gradient variance within each training step, simplifies the iteration logic, and eliminates the need for dataset-specific processing or inter-domain communication inside the iteration. Furthermore, the approach exhibits strong scalability: adding a new dataset requires only its dataset-specific configuration, without modifying the overall batching or communication pattern.

### ...

[Figure 92]

[Figure 93]

[Figure 94]

Target Item

Query Item

Target Item

| | | | |
|---|---|---|---|
| | | | |

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

### ...

| | |
|---|---|
| | |
| | |
| | |
| | |

Auxiliary Task

ID2item Distillation

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Sequence Encoder/Pooling

Feature Projection

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

Seq2item Distillation

| | | | | |
|---|---|---|---|---|
| | | | | |

ID 1

ID 2 ID 3

(a) (b)

- Figure 7 Recommendation Enhancement Training. We implement the (a) sequence-to-item distillation and (b) ID-to-item distillation to enhance the SAIL-Embedding’s collaboration-aware capabilities.

###### 3.3.4 Dataset-Driven Pattern Matching

To address the challenge of heterogeneous modality availability and imbalance across datasets, we design a modality-aware matching strategy that unifies various contrastive objectives under a general query-to-target framework. Specifically, we generalize the CLIP objective beyond the canonical image-to-text setting to a comprehensive modality-to-modality paradigm, where any modality can serve as the query while the others become potential targets. This formulation enables flexible alignment tasks such as Image-to-Text Contrastive (ITC), Image-to-Image Contrastive (IIC), Video-to-Text Contrastive (VTC), Video-to-Video Contrastive (VVC), Text-to-Text Contrastive (TTC), and Omni-to-Omni Contrastive (OOC), the latter covering arbitrary cross-modal pairs without restriction.

A configurable data processor consolidates heterogeneous raw inputs (e.g., images, video frames, captions, ASR transcripts, OCR tokens) into a standardized set of modalities. For each training sample, we dynamically construct all valid query–target pairs according to predefined matching patterns that account for modality characteristics. Modalities without valid patterns are excluded from the loss to mitigate noise propagation. Unlike static formulations that fix the number of losses per dataset, our dynamic multi-pattern matching evaluates all feasible query–target pairs for each sample within the same forward pass. This not only maximizes the utilization of extracted embeddings but also improves optimization stability, as reflected by smoother convergence in training loss according to our experiments.

###### 3.3.5 Collaboration-aware Recommendation Enhancement Training

To address the demands of downstream recommendation scenarios for capturing user interests and to overcome the limitations of existing single-video content understanding, inspired by [10], we propose a collaboration-aware recommendation enhancement training strategy.

Sequence-to-Item Distillation. In data construction, we select user query sequence-to-target video pairs from four perspectives. Specifically, we first choose the users’ historical video viewing sequence (1k) and filter it based on labels indicating positive interaction behaviours. The most recent item in the sequence is designated as the target video.

- 1. Content-aware Single-Peak Interest Modelling: This part of the sequence data is retained by satisfying videos with at least three positive behaviours to mine items with compact interest distributions. Subsequently, items are further filtered based on similarity in content representations.
- 2. Content-aware Multi-Peak Interest Modelling: This part of the sequence data is retained by satisfying videos with any positive behaviour to mine items with widely distributed interests. Then, we select sequence items and the target item based on the Jaccard coefficient, choosing those with behaviour label thresholds greater than 0.5.

- 3. Collaboration-aware Single-Peak Interest Modelling: This part of the sequence data is retained by satisfying at least three positive behaviours. The Jaccard coefficient is still employed to measure behaviour consistency with the target item. We perform label-based clustering to select items that satisfy the original user interaction distribution.
- 4. Collaboration-aware Multi-Peak Interest Modelling: This part of the sequence data is retained by satisfying at least one positive behaviour. Subsequently, we select items corresponding to the proportional distribution of clustered videos as the final sequence data.

As shown in Figure 7(a), we perform contrastive learning by using the representations of the query sequence and the target video. In practice, we find two effective sequence modeling approaches: the mean pooling and the sequence encoder. The former inputs each video in the query sequence into an embedding model to obtain a sequence of query video embeddings, which are then aggregated via pooling. The latter employs a three-layer transformer module to construct a sequence encoder, using a special token to extract the sequence’s overall representation.

ID-to-Item Distillation. In Figure 7(b), we attempt to directly align the model’s output with the recommendation-side representations. In practice, we jointly utilize multiple ID embeddings of each item from the recommender system, aligning them with the omni-modal representation via feature projection to perform feature distillation. Simultaneously, we introduce an auxiliary i2i retrieval task and perform optimization in a multi-task manner to prevent the representation distribution from overly sacrificing content-aware perception capabilities. The above joint training enhances the model’s ability to aggregate video content and mine user interests, thereby better adapting to recommendation scenarios.

##### 4 Experiments and Results

###### 4.1 Configuration Settings

SAIL-Embedding is trained across multiple NPUs, using DeepSpeed ZeRO2 [49] and gradient checkpointing strategies. In the modality components, the latent dimension, number of layers, and sampling depth of the visual perceiver are set to 1024, 16, and 6 respectively, with the self-attention enabled. Audio tokens extracted from the CLAP model [15] have an original dimension of 512 and are mapped via an adapter to a 1536-dimensional alignment hidden state. For the model optimization, we employ the FusedAdam optimizer, which supports mixed-precision computation and gradient fusion, significantly enhancing computational efficiency while maintaining training accuracy. The initial learning rate and weight decay coefficient are set to 1e-5 and 1e-4 to mitigate overfitting risks. The learning rate schedule employs a cosine annealing strategy, with the variation range constrained between 1e-5 and 6e-6 to prevent gradient explosion. In addition, a warm-up strategy is applied during the initial phase to stabilize optimization, and gradient clipping is used to further ensure training stability by limiting gradient norms. In the seq2item distillation, we default to the mean pooling method and set the sequence length to 10 for efficient training. The sequence dataset representing diverse interests comprises 11M samples. The main task dataset in the ID2item distillation contains 220M samples, while the auxiliary task dataset holds 20M samples.

We compare the CLIP-based model [47] and the standard VLM-based approach [74] mainly on a wide range of item-toitem (i2i) tasks. Both are fine-tuned on the training data in Table 1 to ensure fair comparisons. Additionally, performance comparisons on query-to-item (q2i) tasks further involve unimodal embedding models, i.e., Doubao-Embedding and Qwen3-Embedding [75].

###### 4.2 Evaluation Metrics

In recommendation scenarios, traditional evaluation metrics for multimodal embeddings often fail to capture their true business impact. A multimodal embedding model may achieve strong standalone performance but contribute little to downstream user engagement. Moreover, recommendation systems evolve rapidly, where inefficient trial-and-error without robust pre-deployment evaluation risks suboptimal online performance. These challenges motivate the design of a multi-dimensional evaluation framework tailored to multimodal embedding models in recommendation tasks.

Given a dataset {𝑞𝑖, 𝑡𝑖}𝑖𝑁=1, where 𝑞𝑖 denotes a query item and 𝑡𝑖 its matched target, a multimodal encoder E produces embeddings E(𝑞𝑖), E(𝑡𝑖). The objective is to assess whether the resulting embedding space (Q, T) exhibits desirable properties: structural similarity, stable ranking, and strong discriminability. Our evaluation protocol consists of four complementary dimensions:

Models Scenarios Task Metric

CLIP-based Model [47] VLM-based Model [74] SAIL-Embedding Recall@50 17.83 19.99 20.76

University Sub-i2i

Recall@100 21.27 24.68 24.46

Recall@50 36.64 60.71 62.26 Travel Sub-i2i

Recall@100 45.87 70.61 72.40

Recall@50 75.98 81.32 89.08 Film-i2i

Recall@100 80.40 84.79 91.86

Recall@50 95.31 98.75 98.13 Store Visit (Restaurant)-i2i

Recall@100 96.69 99.07 98.69

Recall@50 16.90 22.09 26.67 Store Visit (City)-i2i

Recall@100 22.00 27.65 32.10

Recall@50 58.00 62.19 62.35 Hot Topic-i2i

Recall@100 74.43 80.56 77.98

Recall@50 71.38 70.17 78.07 Comment-i2i

Content Understanding

Recall@100 82.16 80.57 87.62

Recall@50 74.38 85.30 84.55 Music Play-i2i

Recall@100 79.90 89.25 88.72

Recall@50 71.36 82.52 82.08 Music Gameplay-i2i

Recall@100 77.25 86.98 86.72

Recall@50 67.78 71.15 72.83 Game Tag-i2i

Recall@100 72.15 74.92 76.93

Recall@50 26.54 45.97 52.03 Brand Vehicle-i2i

Recall@100 33.15 52.22 57.34

Recall@50 36.01 66.31 72.54 Brand Phone-i2i

Recall@100 46.84 76.68 81.64

Recall@50 71.00 79.31 82.56 Spot-i2i

Recall@100 75.23 81.72 84.64

Recall@50 67.36 76.12 75.52 Summary-i2i

Recall@100 77.48 85.60 85.56

Recall@50 31.94 48.33 74.87 Sub-i2i

Recall@100 44.59 61.51 81.96 Recall@50 57.27 49.66 56.74

ID Fusetag-i2i

Recall@100 66.99 60.13 67.45

Recall@50 56.72 77.45 80.50 Video Search-i2i

Recall@100 62.23 80.90 84.01 Recall@50 39.41 65.79 68.43

Search

Search-i2i

Recall@100 44.63 70.58 73.00

Recall@50 49.76 42.52 52.46 RSDF-i2i

Recall@100 55.79 48.44 59.06

Recall@50 53.47 66.06 69.17 Copair-i2i

Collaborative Perception

Recall@100 70.6 72.70 75.57 Recall@50 47.44 52.55 54.10

Live-i2i

Recall@100 56.69 62.60 63.80

Table 2 Performance Comparison on i2i Tasks. We consider four categories of realistic applications, including content understanding, search, and collaborative perception scenarios.

- 1. Retrieval Recall. We compute recall@k to quantify retrieval performance:

Recall@k =

Number of relevant items retrieved in top-k Total number of relevant items

.

This is applied to i2i and q2i benchmarks, with 𝑘 ∈ {1, 10, 25, 50, 100}.

- 2. Positive–Negative Separability. For each query, we compute cosine similarity with its ground-truth target and with randomly sampled negatives. The distributional gap between positive and negative similarities reflects the model’s discriminative capability.
- 3. Group-wise Clustering Consistency. We cluster the embeddings of positive and negative samples offline and adopt Normalized Mutual Information (NMI) to measure consistency:

2𝐼(𝐿𝑞; 𝐿𝑡) 𝐻(𝐿𝑞) + 𝐻(𝐿𝑡)

NMI(𝐿𝑞, 𝐿𝑡) =

,

where 𝐿𝑞 and 𝐿𝑡 denote cluster assignments, 𝐼(·; ·) is mutual information, and 𝐻(·) denotes entropy. A higher NMI indicates that positive and negative samples are distinguished by similar feature dimensions in the multimodal semantic space. For example, both positive and negative samples may cluster into the category “anime,” yet user

Task Metric Doubao-Embed. Qwen3-Embed.-4B Qwen3-Embed.-8B [75] CLIP-based Model [47] VLM-based Model [74] SAIL-Embedding Short Video-q2i

Recall@50 61.93 71.73 72.42 74.16 78.53 86.53 Recall@100 64.01 75.17 75.77 76.21 80.66 88.54

Decision-q2i AUC 74.51 69.79 69.39 65.18 67.34 82.44 Longtail-q2i AUC 85.84 83.90 83.96 83.18 84.02 91.22

Search Longtail-q2i AUC 76.86 74.90 75.34 74.82 74.99 83.31

Unbiased-q2i AUC 88.67 86.03 86.11 88.62 88.06 93.86 Unbiased Longtail-q2i AUC 85.17 82.01 82.32 85.06 85.86 89.10

Biased-q2i AUC 87.79 85.92 86.10 88.65 89.02 91.65 Live Summary2i

Recall@50 66.31 70.59 69.79 78.25 81.55 84.33

Recall@100 69.31 74.45 73.85 82.55 85.62 88.16 Live-q2i

Recall@50 59.91 64.23 64.22 74.24 73.91 79.08 Recall@100 64.76 69.78 69.57 79.89 79.34 84.38

Table 3 Performance Comparison on q2i Tasks. We consider retrieval and classification applications across nine tasks.

preferences differ with respect to this content. This consistency highlights two key properties: (1) Effectiveness of modality information — multimodal features capture the actual decision basis underlying user behavior (e.g., a user may like or dislike anime due to its “art style”); (2) Strong interpretability of representations — the observed consistency suggests that multimodal features bear a potential causal relationship with user behavior, rather than reflecting only superficial correlations.

- 4. Ranking Consistency. We assess whether query–target similarity rankings remain stable across spaces. Top-𝑘 overlap and Kendall’s 𝜏 are used:

𝜏 =

𝐶 − 𝐷 𝐶 + 𝐷

,

where 𝐶 and 𝐷 are the numbers of concordant and discordant pairs, respectively. Higher 𝜏 indicates better preservation of relative semantic ordering.

- 5. Bijective Alignment Test. We verify whether embeddings exhibit near one-to-one mapping. Specifically, for each query, we retrieve its most similar target, and from this target, retrieve back its most similar query. The proportion of cases where the indices match quantifies embedding fidelity and bijective consistency.
- 6. AUC Measurement. Additionally, we report the AUC metric on recommendation-oriented tasks.

Our comprehensive evaluation suite provides not only a quantitative measure of embedding quality but also actionable diagnostics on the suitability of multimodal embedding models for integration into large-scale recommendation systems.

- 4.3 Evaluation Dataset Introduction We curate several evaluation benchmarks to assess our models. Below is the detailed introduction: For item2item evaluation, we build three types of benchmarks:

- • Content Understanding. To evaluate the basic ability to capture semantic similarities, we integrate items with similar visual cues and textual information into pairs according to various rules. Except for the test split of training datasets, we collect some out-of-domain i2i benchmarks to evaluate the generalizability of our model. Leveraging the auto-tagging pipelines and expert models, we group items with the same brands or intellectual property, resulting in Brand Vehicle-i2i, Brand Phone-i2i, Film-i2i and Game Tag-i2i. Moreover, we employ MLLMs to summarize key information for items, and group them with the same n-grams or keywords, resulting in Keywords-i2i and Summary-i2i.
- • Search-Based. Taking user actions into consideration, we pair search-and-click items with the same queries. Specifically, Search-i2i consists of items with various forms, e.g., videos and photos. While video search-i2i only contains video items, which are mostly delivered on Douyin.
- • Collaborative Perception. To evaluate models’ ability in recommendation scenarios, we collect item pairs based on the co-occurrence mechanism. We construct datasets for video items and live streaming items, respectively. Besides, we directly leverage dense features of the recommender system to compute the cosine similarity scores between items. Items with scores exceeding a particular score are reserved as positive pairs.

Loss Curve w/ CRE SAIL-Embedding SAIL-Embedding (CRE)

[Figure 101]

[Figure 102]

[Figure 103]

Loss

Step

(a) (b) (c)

- Figure 8 Illustration of the Results with Collaboration-aware Recommendation Enhancement (CRE). (a) Loss curve of the model fine-tuned with distilled ID embeddings. (b) Distribution of similarity scores for positive and negative samples in the original SAIL-Embedding. (c) Distribution of similarity scores after applying CRE.

For query2item evaluation, we build two types of benchmarks:

- • Query2item Retrieval Datasets. Short Video-q2i is an in-domain dataset which is similar to the Search-q2i dataset for training. Besides, we collect two out-of-domain benchmarks from live streaming scenario. Live-q2i utilizes user input search text as queries while Live-summary2i uses the summarized texts of each item.
- • Query2item Classification Tasks. To assess the model’s ability to discriminate similar queries and items, we further collect six embedding-based classification datasets from different downstream scenarios. For evaluation, we acquire cosine similarity scores between all queries and items with their embeddings. The scores are regarded as the binary classification probabilities, which predict whether a query and an item are paired positives or unpaired negatives. The AUC metric is computed as the final result. In these datasets, all items contain only text features.

###### 4.4 Results Analysis

- 4.4.1 Item-to-Item Retrieval Task Comparison Results

- In Table 2, we present comparison results across 21 i2i tasks spanning four realistic application intents, including content understanding, search, and collaborative perception. We primarily perform omni-modal item retrieval. SAILEmbedding significantly and consistently outperforms previous models in search and collaborative perception scenarios. Compared to the CLIP-based model, SAIL-Embedding has richer open-world knowledge and feature fusion capabilities, demonstrating stronger multimodal understanding when handling complex and dynamic business scenarios. Compared to the VLM-based model, our model verifies the importance and effectiveness of the audio modality in multimedia retrieval driven by short videos. Furthermore, the well-designed training strategies also better eliminate the performance gaps of large-scale retrieval models in industrial applications. Under the content understanding-oriented scenario, SAIL-Embedding demonstrates competitive performance across most tasks, suggesting the potential of the proposed embedding model in content-semantic learning.

4.4.2 Query-to-Item Retrieval Task Comparison Results

- In Table 3, we further report results for various retrieval methods across 9 q2i tasks, encompassing both uni-modal and vision-language models. These tasks are categorized into retrieval-oriented and classification-oriented tasks, evaluated by Recall and AUC, respectively. SAIL-Embedding significantly outperforms previous methods across all metrics for every task. A plausible explanation is that the omni-modal architecture promotes unified semantic fusion and resolves modality ambiguity, leading to stronger cross-modal semantic understanding. Furthermore, the dataset-driven pattern matching and stochastic specialization training enhance SAIL-Embedding’s cross-domain generalization capabilities and cross-task adaptability.

Gid-i2i R@50

Gid-i2i R@100

Copair-i2i R@50

Copair-i2i R@100

Models NMI Ken.* Inter. Acc.

VLM-based Model 0.53 68.6 28.29 46.51 42.52 48.44 66.06 72.70 VLM-based Model (CRE) 0.61 68.6 29.62 47.22 46.34 52.18 64.78 70.52 SAIL-Embedding 0.60 73.4 29.73 48.28 52.46 59.06 69.17 75.57 SAIL-Embedding (CRE) 0.65 77.2 31.63 48.94 59.69 66.37 66.83 73.10

- Table 4 CRE Training Quantitative Results. NMI, Ken., Inter., and Acc. represent Normalized Mutual Information in group-wise clustering consistency, Kendall’s coefficient, intersection proportion in ranking consistency, and the proportion of successful hits in the bijective alignment test, respectively. For Ken.*, the values are multiplied by 10K for clarity.

###### 4.4.3 Collaboration-aware Recommendation Enhancement Results

Through Collaboration-aware Recommendation Enhancement (CRE), we adapt SAIL-Embedding to better suit downstream recommendation tasks. To evaluate the effectiveness of this training phase, we report the results under different metrics in Figure 8 and Table 4. As shown in Figure 8, after incorporating Sequence/ID embedding distillation, the model converges stably after around 20k steps. We measure the separability illustrated in Figures 8(b)&(c). Before and after distillation, the model already exhibits good discrimination between positive and negative samples. However, with distillation, the overlap between positive and negative samples is further reduced and shifts rightward, verifying that the strategy effectively enhances the discriminative power of the base model, particularly for hard negatives with similarity scores around 0.5. In addition, the positive sample distribution becomes more compact, indicating that by jointly incorporating some ID embeddings, the model learns correlations among positive samples with higher confidence. This capability reflects exactly the kind of improvement expected in recommendation, where multimodal models should not only preserve separability but also strengthen relevance modeling for positive samples.

Quantitative results are summarized in Table 4. Overall, we observe that the CRE strategy consistently improves performance across different metrics, both for VLM-based models and for our proposed SAIL-Embedding. For the NMI, the ID-distilled version achieves a +5% gain, indicating that positive and negative samples become more separable in clustering, which facilitates learning of user preference boundaries. For the ranking consistency, the distilled model improves the Kendall correlation by +3.8% and the Intersection metric by +1.9%, demonstrating enhanced consistency in retrieval orderings and suggesting stronger downstream recommendation performance. For the bijective alignment test, the distilled model improves by +0.66%, showing that incorporating ID embeddings encourages SAIL-Embedding to form more stable matching structures, which may provide valuable guidance for both recall and ranking stages.

Another interesting finding is that CRE training significantly improves model performance on the Gid-i2i benchmark. For instance, the CRE version of SAIL-Embedding achieves 7.23% and 7.31% improvements over the original model on Recall@50 and Recall@100 metrics. The VLM-based model also achieves an average improvement of 3.78%. This observation confirms that training via seq2item and ID2item distillation can further refine the model’s omni-modal representations to perceive collaborative semantics. This is reasonable since Gid-i2i is a paired testing benchmark constructed from gid embeddings’ relevance, which undergoes streaming updates during online execution. Additionally, we observe some performance degradation on the Copair-i2i benchmark constructed based on item-based representation similarity. We consider this variation tolerable to balance the differing application requirements of content-oriented and collaborative behavior-oriented scenarios in industrial settings.

###### 4.4.4 Sequence Modeling Evaluation Results

To verify the model’s understanding of sequence semantics after sequence distillation, we build two evaluation sets from user historical viewing sequences to target videos, named Vanilla Seq2item and Filtered Seq2item. In the former, we collect 100 historical videos viewed by the user in chronological order. The most recent video is selected as the target, while the remaining 10 videos are randomly sampled from the sequence to form the user query sequence. In the Filtered Seq2item, we extend the collected historical sequence to 1k videos and filter them based on content tags or clustered copair-ids to select items highly consistent with the target video as queries. The retained sequence length remains 10.

During implementation, we progressively perform seq2item and ID2item distillation on the content-aware training of SAIL-Embedding. Experimental results are presented in Table 5. The model after seq2item distillation achieves significant performance gains on the Filtered Seq2item. For instance, the Recall@10 and Recall@25 metrics improved

Filtered Seq2item Vanilla Seq2item R@1 R@10 R@25 R@1 R@10 R@25 SAIL-Embedding 8.41 19.35 25.46 4.35 10.46 14.33

Models

SAIl-Embedding (Seq2item Distillation) 8.28 22.02 30.12 4.84 11.78 17.79 SAIl-Embedding (Seq2item + ID2item Distillation) 9.04 23.77 32.41 5.24 13.36 19.09

- Table 5 Sequence Modeling Evaluation Results. Both test sets evaluate the recall accuracy for the top-1, top-10, and top-25 ranked items. The best results are emphasized in bold.

Ablation Studies of Different Modules/Strategies on i2i Tasks

54

52.29

51.75

52

50.08 50.42

Performance(%)

50

47.44

48

45.93

46

44

42.97 43.02

42.47

42

40

BERTasEncoder LLMasFusion w/BidirectionalAttenton w/Task-specificInstruction w/Full-Finetuning AddDataDiversity w/COSENTLoss w/DataScalingUp w/ProgressiveTraining

- Figure 9 Ablation Studies on i2i Tasks. We systematically report on the effects of different modules and strategies on performance.

by 2.67% and 4.66%, respectively. This observation indicates that the model’s sequence comprehension and content aggregation capabilities have been enhanced. Following further ID2item distillation, SAIL-Embedding consistently achieves higher performance, demonstrating that the combined distillation strategy strengthens the model’s collaborationaware capabilities in recommendation scenarios. On the Vanilla Seq2item, the model ultimately achieves a 4.76% metric improvement. This phenomenon suggests that recommendation-enhanced learning enhances the model’s ability to extract diverse user interests.

###### 4.4.5 Systematic Ablation Studies

To comprehensively analyze the necessity of different components and strategies within the model, we conduct systematic ablation studies on subsets of i2i and q2i tasks. Figures 9 and 10 report average performance across different tasks for intuitive observation. We first establish a baseline performance where the BERT model [12] serves as an encoder for the text modality. Subsequently, the LLM is employed as a fuser to replace the traditional dual-tower pattern. We show consistent improvements across both tasks, with the q2i task achieving a significant gain of 5.01%. This means the LLM enhances the model’s ability to handle complex queries, boosting overall retrieval performance. When the original causal attention in the LLM is replaced with the bidirectional attention, positive incremental gains are demonstrated. The underlying reason is that the bidirectional attention captures global multimodal semantic dependencies, mitigating the semantic bias potentially introduced by the causal attention. This makes it more suitable for the representation embedding scenarios.

In the initial implementation, SAIL-Embedding adopts a unified instruction format. Upon transitioning to task-specific instruction designs, we notice substantial gains. For instance, average performances on i2i and q2i scenarios increase by 2.62% and 1.88%, respectively. In practice, we observe that the instruction-based multi-task training paradigm guides the model to focus on distinct feature details. This enables adaptation to different task demands for feature expressiveness across various aspects of the omni-modal embeddings. Additionally, we investigate the differences between using the Low-Rank Adaptation (LoRA) [23] and full-parameter fine-tuning by fully unlocking parameters from the LLM. The results show that full parameter optimization effectively performs cross-modal semantic matching and retrieval between items, enhancing retrieval and classification performance in the q2i tasks.

On the one hand, when enhancing knowledge coverage and business data diversity during model training, SAILEmbedding achieves significant performance gains on i2i tasks while maintaining results across a wide range of q2i tasks.

Ablation Studies of Different Modules/Strategies on q2i Tasks

86

83.91 83.92

84

82.49

82

Performance(%)

80.79 80.80

80.11

79.69

80

79.17

78

76

74.16

74

72

70

BERTasEncoder LLMasFusion w/BidirectionalAttenton w/Task-specificInstruction w/Full-Finetuning AddDataDiversity w/COSENTLoss w/DataScalingUp w/ProgressiveTraining

- Figure 10 Ablation Studies on q2i Tasks. We systematically report on the effects of different modules and strategies on performance.

Scenarios Stage Feature Gain

Recall SID LT30 + 0.01% Pre-Rank SID LT30 + 0.01%

Feed

Rank SID&Embedding Finish AUC + 0.1%

Re-Rank SID LT30 + 0.01% Message Pushing

Recall SID & Embedding LT30 + 0.03%

Rank SID LT30 + 0.01% Coldstart Recall SID & Embedding LT30 + 0.05%

Recall SID LT7 + 0.4% Rank Embedding LT7 + 0.1%

Douyin-Selected

- Table 6 Online Results in Recommender System. Feed, Message Pushing, and Coldstart are different channels or modules of Douyin. Douyin-Selected is another application with its own recommendation pipeline.

On the other hand, we discover that as data volume scales up, model performance increases accordingly, adhering to the effect of scaling laws. After incorporating the COSENT loss, the q2i tasks achieve a significant 1.69% gain, implying that fine-grained ranking capabilities have been enhanced in text-oriented situations. Furthermore, we modify the all-in-one training procedure used by the baseline to the progressive training strategy. Based on experimental results, we conclude that progressive training enables the model to acquire substantial domain knowledge in the early phases while establishing robust task dependencies and activating corresponding representational knowledge in subsequent phases.

###### 4.4.6 Extensive Online Experiments

We deploy our model in the real-world recommender system to further verify its effectiveness. Our model mainly provides two types of features for downstream use.:

- 1. Embeddings. As mentioned before, our model compresses all information of an item, including video frames, title, OCR, ASR, along with the author’s nickname and other textual tags into a dense embedding. This embedding can be used for similarity-based recall, or be used in modules like SIM.
- 2. Semantic ID (SID). Furthermore, we discretize the embeddings into semantic IDs since most recommenders prefer discrete features. We engage both clustering and vector quantization methods to acquire codebooks with different sizes. The discrete tokens can be used for decentralization as well as the features of candidate items.

Experiments are conducted in various scenarios to verify the effectiveness of our method, and some typical results are shown in Table 6. Firstly, we find that the introduced features can benefit nearly all stages of the recommender system, including recall, pre-rank, rank and re-rank. We believe that engaging features provided by the same embedding model can lead to better consistency through these sequential stages. In addition, different scenarios such as coldstart and message pushing can both be enhanced, achieving 0.05% and 0.04% LT gain respectively. And our model can work on

both Douyin and Douyin-Jingxuan, which are two different applications. All these results have demonstrated that our model is both effective and generalizable for recommendation.

Moreover, we find that SIDs bring more gain than dense embeddings. The reason may come from two aspects: (a) SIDs are much easier to use in ruled-based methods than embeddings, such as rule-based decentralization. (b) SIDs can be encoded into trainable embeddings, like item ID, so that they can be further adapted for recommendation models. However, dense embeddings tend to bring more information than SIDs. To effectively leverage such information is worth discovering for future work.

##### 5 Conclusion

In this work, we present SAIL-Embedding, an omni-modal embedding foundation model tailored for large-scale recommendation scenarios. By unifying vision, text, and audio modalities, SAIL-Embedding overcomes the limitations of limited modalities of existing multimodal methods. Our contributions include a dynamic hard negative mining strategy and an adaptive multi-source data balancing framework, which jointly enhance training robustness and representation quality. Furthermore, we design a content-aware progressive training procedure and a collaborationaware recommendation enhancement module, enabling the model to capture both semantic content and collaborative behavioral signals. Extensive experiments across diverse item-to-item and query-to-item benchmarks demonstrate that SAIL-Embedding achieves state-of-the-art performance and superior generalization to real-world industrial applications. Beyond its empirical effectiveness, our systematic ablations validate the necessity of each proposed component. We believe SAIL-Embedding provides a scalable and versatile foundation for future multimodal retrieval and recommendation systems, and we envision extending this framework to broader downstream tasks such as video understanding, personalized content generation, and cross-domain knowledge transfer.

In future work, we plan to further enhance the integration of vision-language models (VLMs) into recommendation systems. First, we will explore training VLMs aligned with recommendation objectives and constructing generative tasks tailored for recommendation, enabling the model to acquire domain-specific knowledge at earlier stages and strengthen its recommendation capability. Second, during representation learning, we aim to better align model training with recommendation goals by mining more paired data from recommendation signals and behavioral feedback, thereby injecting user preferences into multimodal representations. Finally, we will investigate hard negative mining in recommendation scenarios to improve the robustness of representation learning.

##### 6 Contributor List

All contributors are listed in alphabetical order by last name initial, with equal contributions within each group; unless otherwise noted, all are members of the Douyin SAIL Team.

Core Contributors: Lin Lin Jiefeng Long Zhihe Wan Yuchi Wang (CUHK MMLab) Dingkang Yang Shuang Yang Yueyang Yao

Contributors: Xu Chen Zirui Guo Shengqiang Li Weiran Li Hanyu Li Yaling Mou Yan Qiu Haiyang Yu

Project Leader: Xiao Liang

Supervisors: Xiao Liang Hongsheng Li (CUHK MMLab) Chao Feng

##### 7 Acknowledgments

We sincerely thank Song Chen, Yaodi Du, Xinjie Huang, Shengyu Li, Zhenming Sun, Junyan Yao, Enwei Zhang, Tengfeng Zhang, Zeqin Zhang, and Sheng Zheng for their valuable support and contributions.

##### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Hassan Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Singh Behl, Alon Benhaim, and et al. Phi-3 technical report: A highly capable language model locally on your phone. ArXiv, abs/2404.14219, 2024. URL https://api.semanticscholar.org/CorpusID: 269293048.
- [2] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuning-free framework for video motion and appearance editing. ArXiv, abs/2402.13185, 2024. URL https://api.semanticscholar. org/CorpusID:267760278.
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. URL https://api.semanticscholar.org/CorpusID:261101015.
- [4] Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan Yang, Furu Wei, and Zhicheng Dou. Moca: Modality-aware continual pre-training makes better bidirectional multimodal embeddings. arXiv preprint arXiv:2506.23115, 2025.
- [5] Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. mme5: Improving multimodal multilingualembeddingsviahigh-qualitysyntheticdata. ArXiv, abs/2502.08468, 2025. URLhttps://api.semanticscholar. org/CorpusID:276287871.
- [6] Jiawei Chen, Dingkang Yang, Yue Jiang, Mingcheng Li, Jinjie Wei, Xiaolu Hou, and Lihua Zhang. Efficiency in focus: Layernorm as a catalyst for fine-tuning medical visual language pre-trained models. arXiv preprint arXiv:2404.16385, 2024.
- [7] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.
- [8] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.
- [9] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965, 2025.
- [10] Xiuqi Deng, Lu Xu, Xiyao Li, Jinkai Yu, Erpeng Xue, Zhongyuan Wang, Di Zhang, Zhaojie Liu, Guorui Zhou, Yang Song, et al. End-to-end training of multimodal model and ranking model. arXiv preprint arXiv:2404.06078, 2024.
- [11] Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11157–11168, 2020. URL https://api.semanticscholar. org/CorpusID:219573658.
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
- [13] Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable vision language model training via high quality data curation. arXiv preprint arXiv:2501.05952, 2025.
- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [15] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.
- [16] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. ArXiv, abs/2504.17432,

2025. URL https://api.semanticscholar.org/CorpusID:278033532.

- [17] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432, 2025.

- [18] Michael Gu¨nther, Louis Milliken, Jonathan Geuter, Georgios Mastrapas, Bo Wang, and Han Xiao. Jina embeddings: A novel set of high-performance sentence embedding models. arXiv preprint arXiv:2307.11224, 2023.
- [19] Michael G¨unther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, et al. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. arXiv preprint arXiv:2310.19923, 2023.
- [20] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, and et al. Seed1.5-vl technical report. ArXiv, abs/2505.07062, 2025. URL https://api.semanticscholar.org/CorpusID:278502305.
- [21] Andrey Guzhov, Federico Raue, J¨orn Hees, and Andreas R. Dengel. Audioclip: Extending clip to image, text and audio. ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 976–980, 2021. URL https://api.semanticscholar.org/CorpusID:235624127.
- [22] Minghao Han, Dingkang Yang, Jiabei Cheng, Xukun Zhang, Linhao Qu, Zizhi Chen, and Lihua Zhang. Towards unified moleculeenhanced pathology image representation learning via integrating spatial transcriptomics. arXiv preprint arXiv:2412.00651, 2024.
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [24] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282, 2020.
- [25] AndrewJaegle, FelixGimeno, AndrewBrock, AndrewZisserman, OriolVinyals, andJo˜aoCarreira. Perceiver: Generalperception with iterative attention. ArXiv, abs/2103.03206, 2021. URL https://api.semanticscholar.org/CorpusID:232110866.
- [26] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, 2021. URL https://api.semanticscholar.org/CorpusID:231879586.
- [27] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580, 2024.
- [28] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, 2023.
- [29] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024.
- [30] Siddharth Joshi, Arnav Jain, Ali Payani, and Baharan Mirzasoleiman. Data-efficient contrastive language-image pretraining: Prioritizing data quality over quantity. In International Conference on Artificial Intelligence and Statistics, 2024. URL https://api.semanticscholar.org/CorpusID:268532161.
- [31] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781, 2020.
- [32] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representation learning. In Neural Information Processing Systems, 2022. URL https://api.semanticscholar.org/CorpusID:252683450.
- [33] Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hern´andez Abrego,´ Zhe Li, Kaifeng Chen, Henrique Schechter Vera, et al. Gemini embedding: Generalizable embeddings from gemini. arXiv preprint arXiv:2503.07891, 2025.
- [34] Weixian Lei, Jiacong Wang, Haochen Wang, Xiangtai Li, Jun Hao Liew, Jiashi Feng, and Zilong Huang. The scalability of simplicity: Empirical analysis of vision-language learning with a single transformer. arXiv preprint arXiv:2504.10462, 2025.
- [35] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, 2022. URL https://api.semanticscholar.org/CorpusID:246411402.

- [36] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023. URL https: //api.semanticscholar.org/CorpusID:256390509.
- [37] Mingcheng Li, Dingkang Yang, Yuxuan Lei, Shunli Wang, Shuaibing Wang, Liuzhen Su, Kun Yang, Yuzheng Wang, Mingyang Sun, and Lihua Zhang. A unified self-distillation framework for multimodal sentiment analysis with uncertain missing modalities. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 10074–10082, 2024.
- [38] Mingcheng Li, Dingkang Yang, Yang Liu, Shunli Wang, Jiawei Chen, Shuaibing Wang, Jinjie Wei, Yue Jiang, Qingyao Xu, Xiaolu Hou, et al. Toward robust incomplete multimodal sentiment analysis via hierarchical representation learning. Advances in Neural Information Processing Systems, 37:28515–28536, 2024.
- [39] Shiyu Li, Yang Tang, Shizhe Chen, and Xi Chen. Conan-embedding: General text embedding with more and better negative samples. arXiv preprint arXiv:2408.15710, 2024.
- [40] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024.
- [41] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. Neurocomputing, 508:293–304, 2021. URL https://api.semanticscholar.org/CorpusID: 233296206.
- [42] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025.
- [43] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.
- [44] Bhaskar Mitra, Fernando Diaz, and Nick Craswell. Learning to match using local and distributed representations of text for web search. In Proceedings of the 26th international conference on world wide web, pages 1291–1299, 2017.
- [45] Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543, 2014.
- [46] Jiajun Qin, Yuan Pu, Zhuolun He, Seunggeun Kim, David Z. Pan, and Bei Yu. Unimoco: Unified modality completion for robust multi-modal embeddings. ArXiv, abs/2505.11815, 2025. URL https://api.semanticscholar.org/CorpusID: 278739565.
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [48] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.
- [49] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
- [50] Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. arXiv preprint arXiv:2402.15449, 2024.
- [51] Jianlin Su. Cosent (1): A more effective sentence vector scheme than sentence bert, Jan 2022. URL https://kexue.fm/ archives/8847.
- [52] Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Gaopeng Gou, and Qi Wu. Missing target-relevant information prediction with world model for accurate zero-shot composed image retrieval. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24785–24795, 2025.
- [53] Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [54] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar. org/CorpusID:13756489.

- [55] Jiacong Wang, Bohong Wu, Haiyong Jiang, Xun Zhou, Xin Xiao, Haoyuan Guo, and Jun Xiao. World to code: Multi-modal data generation via self-instructed compositional captioning and filtering. arXiv preprint arXiv:2409.20424, 2024.
- [56] Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025.
- [57] Yuchi Wang, Junliang Guo, Jianhong Bai, Runyi Yu, Tianyu He, Xu Tan, Xu Sun, and Jiang Bian. Instructavatar: Text-guided emotion and motion control for avatar generation. ArXiv, abs/2405.15758, 2024. URL https://api.semanticscholar. org/CorpusID:270045800.
- [58] Yuchi Wang, Shuhuai Ren, Rundong Gao, Linli Yao, Qingyan Guo, Kaikai An, Jianhong Bai, and Xu Sun. Ladic: Are diffusion models really inferior to autoregressive counterparts for image-to-text generation? In North American Chapter of the Association for Computational Linguistics, 2024. URL https://api.semanticscholar.org/CorpusID:269156964.
- [59] Yuchi Wang, Yishuo Cai, Shuhuai Ren, Sihan Yang, Linli Yao, Yuanxin Liu, Yuanxing Zhang, Pengfei Wan, and Xu Sun. Rico: Improving accuracy and completeness in image recaptioning via visual reconstruction. ArXiv, abs/2505.22613, 2025. URL https://api.semanticscholar.org/CorpusID:278959593.
- [60] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pages 387–404. Springer, 2024.
- [61] Yixuan Wei, Yue Cao, Zheng Zhang, Houwen Peng, Zhuliang Yao, Zhenda Xie, Han Hu, and Baining Guo. iclip: Bridging image classification and contrastive language-image pre-training for visual recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2776–2786, 2023.
- [62] Zhi Xu, Dingkang Yang, Mingcheng Li, Yuzheng Wang, Zhaoyu Chen, Jiawei Chen, Jinjie Wei, and Lihua Zhang. Debiased multimodal understanding for human language sequences. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 14450–14458, 2025.
- [63] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [64] Dingkang Yang, Shuai Huang, Haopeng Kuang, Yangtao Du, and Lihua Zhang. Disentangled representation learning for multimodal emotion recognition. In Proceedings of the 30th ACM International Conference on Multimedia (ACM MM), pages 1642–1651, 2022.
- [65] Dingkang Yang, Shuai Huang, Shunli Wang, Yang Liu, Peng Zhai, Liuzhen Su, Mingcheng Li, and Lihua Zhang. Emotion recognition for multiple context awareness. In Proceedings of the European Conference on Computer Vision (ECCV), volume 13697, pages 144–162, 2022.
- [66] Dingkang Yang, Haopeng Kuang, Kun Yang, Mingcheng Li, and Lihua Zhang. Towards asynchronous multimodal signal interaction and fusion via tailored transformers. IEEE Signal Processing Letters, 31:1550–1554, 2024.
- [67] Dingkang Yang, Mingcheng Li, Linhao Qu, Kun Yang, Peng Zhai, Song Wang, and Lihua Zhang. Asynchronous multimodal video sequence fusion via learning modality-exclusive and-agnostic representations. IEEE Transactions on Circuits and Systems for Video Technology, 2024.
- [68] Dingkang Yang, Mingcheng Li, Dongling Xiao, Yang Liu, Kun Yang, Zhaoyu Chen, Yuzheng Wang, Peng Zhai, Ke Li, and Lihua Zhang. Towards multimodal sentiment analysis debiasing via bias purification. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [69] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19163–19173, 2022.
- [70] Weijie Yin, Dingkang Yang, Hongyuan Dong, Zijian Kang, Jiacong Wang, Xiao Liang, Chao Feng, and Jiao Ran. Sailvit: Towards robust and generalizable visual backbones for mllms via gradual feature refinement. arXiv preprint arXiv:2507.01643, 2025.
- [71] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11941–11952, 2023. URL https://api. semanticscholar.org/CorpusID:257767223.
- [72] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [73] Chao Zhang, Haoxin Zhang, Shiwei Wu, Di Wu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, and Enhong Chen. Notellm-2: Multimodal large representation models for recommendation. Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, 2024. URL https://api.semanticscholar.org/CorpusID:270063288.
- [74] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024.
- [75] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.
- [76] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. Vista: Visualized text embedding for universal multi-modal retrieval. arXiv preprint arXiv:2406.04292, 2024.

