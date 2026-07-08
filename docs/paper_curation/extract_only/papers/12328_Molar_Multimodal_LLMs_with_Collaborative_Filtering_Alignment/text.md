## Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation

Yucong Luo, Qitao Qin, Hao Zhang, Mingyue Cheng, Ruiran Yan, Kefan Wang, Jie Ouyang University of Science and Technology of China State Key Laboratory of Cognitive Intelligence Hefei, Anhui, China {prime666,qqt,zh2001,yanruiran,wangkefan,ouyang_jie}@mail.ustc.edu.cn mycheng@ustc.edu.cn

### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

The User has watched [14] North by Northwest [20] Back to the Future [37] Toy Story predict the next moviethe user will watch.

||North by Northwest Back to the Future Toy Story|
|---|
<br><br>LLM Word Embedding<br><br>|[14] [20] [37]|
|---|
<br><br>Traditional Rec Model<br><br>LLM-based Recommender<br><br>Piror LLM-based Recommendation<br><br>Item Text embedding Pre-Alignment Item ID embedding<br><br>Text Metadata ID Number<br><br>(a)|
|---|

Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from contentbased and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLMbased baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available 1.

# arXiv:2412.18176v2[cs.IR]30Dec2024

|Ours<br><br>|North by Northwest<br><br>Back to the Future<br><br>Toy Story<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|
|---|
<br><br>Multimodal LLM User Content embedding<br><br>|[14] [20] [37]|
|---|
<br><br>Traditional Rec Model User ID embedding<br><br>Recommendation Layer<br><br>ID Number<br><br>Multimodal Metadata<br><br>(b)<br><br>Post-Alignment|
|---|

Figure 1: Comparison of LLM-based recommendation methods and our Molar. (a) Existing methods prematurely integrate ID and text modalities into the LLM, leading to limited utilization of multimodal content features. (b) Our approach first processes text and non-text modalities through the LLM to generate rich multimodal representations and then incorporates ID information via post-alignment, ensuring a better balance between multimodal content and collaborative signals.

Among these systems, sequential recommendation (SR) methods (Wang et al., 2019; Zhou et al., 2018; Kang and McAuley, 2018; Cheng et al., 2021) have gained prominence due to their ability to capture dynamic user interests more effectively than traditional collaborative filtering techniques. Mainstream SR approaches are mainly based on IDbased deep learning strategies (Koren et al., 2009; Goldberg et al., 1992), such as matrix factorization and deep sequence neural networks (Cheng et al., 2022). These methods encode users and items as unique identifiers, using historical interaction data to learn sequential behavioral patterns.

Recent advances in large language models (LLMs) (Zhao et al., 2023; Luo et al., 2024) have opened new possibilities for sequential recommendation. With their powerful sequence modeling and multimodal understanding capabilities, LLMs have been explored in two main directions. The first approach (Zhang et al., 2023b; Ren et al., 2024; Friedman et al., 2023) reframes SR as a natural language processing task, allowing LLMs to inter-

### 1 Introduction

In the era of information overload, recommender systems have become essential tools to filter information through various online applications, including e-commerce, advertising, and online video platforms (Resnick and Varian, 1997; Koren, 2008).

1https://anonymous.4open.science/r/Molar-8B06/

pret user sequences and generate recommendations based on their language understanding. The second approach (Ning et al., 2024; Liao et al., 2023) combines LLMs with traditional SR models by integrating ID and text modalities into the LLM backbone at an early stage, as shown in Figure 1a. However, these methods face critical limitations: (1) inadequate integration of multimodal features, leading to information loss from non-textual modalities or misalignment between textual and visual features; and (2) suboptimal utilization of traditional SR models, where early fusion of ID information can cause LLMs to learn shortcuts, overshadowing collaborative filtering signals. These challenges result in a failure to fully exploit the multimodal features of items and the potential of collaborative filtering in SR.

To address these issues, this paper introduces the Molar, a Multimodal large language sequential recommendation framework. We make three main contributions in this paper. First, we propose a Multimodal Item Representation Model (MIRM) based on a multimodal large language model (MLLM) to extract item features from both textual and nontextual modalities. By fine-tuning MIRM on multimodal data, we ensure robust and consistent item embeddings. Second, we design a Dynamic User Embedding Generator (DUEG) that leverages these item embeddings to model user interests and predict future behaviors. This allows for effective user modeling in complex multimodal scenarios. Third, we introduce a post-alignment contrastive learning mechanism that aligns collaborative signals from ID-based and content-based models, preserving the strengths of both representations. This mechanism ensures semantic alignment between user embeddings while enhancing recommendation accuracy.

Experiments conducted on multiple datasets demonstrate the effectiveness of Molar. The framework significantly outperforms traditional SR models and state-of-the-art LLM-based methods, achieving superior results in recommendation accuracy and robustness. Our results show that Molar captures user interests more comprehensively by combining multimodal content and collaborative filtering signals, leading to consistent performance improvements across diverse scenarios.

Our contributions are summarized as follows:

• We propose a Multimodal Item Representation Model (MIRM) to extract robust item embeddings by leveraging multimodal content, includ-

ing both textual and non-textual modalities, ensuring comprehensive item feature modeling.

- • We design a Dynamic User Embedding Generator (DUEG) to effectively model user preferences using multimodal item embeddings, enabling dynamic and accurate user interest prediction.
- • We introduce a post-alignment contrastive learning mechanism to integrate collaborative filtering signals from ID-based and content-based models, preserving their complementary strengths and enhancing recommendation performance.

### 2 Related Work

LLM-Based Recommendation. The success of LLMs such as GPT4 (OpenAI et al., 2024) and LLaMA (Grattafiori et al., 2024) has sparked extensive exploration of their application in recommendation systems. Firstly, LLMs are used to enhance user or item information, such as aligning semantic spaces for user and item profiling or generating training signals for cold-start items (Xi et al., 2024; Ren et al., 2024; Zhang et al., 2024b). Secondly, some studies convert recommendation data into a conversational format, leveraging LLMs’ instruction-following capabilities to predict user behavior (Friedman et al., 2023; Bao et al., 2023; Zhang et al., 2023a). Lastly, LLMs are adapted for recommendation tasks by combining ID-based item embeddings with textual features for hybrid prompting or using them for multi-class classification and regression for rating prediction (Ning et al., 2024; Liao et al., 2023). Although these methods demonstrate the potential of LLMs, improvements over traditional recommendation models remain limited. Prior methods either overlook traditional ID-based models by focusing only on text or introduce ID modalities too early, reducing the effectiveness of collaborative filtering signals during LLM training. Unlike these approaches, we propose a post-alignment mechanism to fuse ID-based collaborative information later in the process, preserving LLMs’ world knowledge while retaining essential collaborative information.

Multimodal Large Language Models. Recent advancements (Peng et al., 2023; Zhang et al., 2024c; Yin et al., 2024) in multimodal pre-training have significantly improved task performance but at the cost of increased computational demands. To address this, researchers are leveraging pre-trained unimodal models, particularly large language models (LLMs) (Kasneci et al., 2023), to develop Mul-

[Figure 7]

Image-Text Alignment Structured Attribute Processing Temporal User Behavior Understanding

[Figure 8]

Point-wise Recommendation Loss

[Figure 9]

[Figure 10]

Alignment Loss

Multimodal Finetuning with Three Objectives :

| | |
|---|---|
| | |

[Figure 11]

[Figure 12]

ID-based User Embedding

Content-based User Embedding

Target Item Embedding

##### Multimodal Item Representation Model

Traditional SR Model

Compress the text and image into embedding. Item information: [Text]. Item Image: [Img]. [Cur_Item]

Dynamic User Embedding Generator

Title: Interview with female big brother at Shanghai B Station Comic Con! Listen to the big brother's heart Tag: Comedy Description: I made a friendly guest appearance at bilibili world on @Crooked Research Society's talk show, and interviewed the ladies in the audience!

[Figure 13]

[𝑒𝑚𝑏 ] [𝑒𝑚𝑏 ] [𝑒𝑚𝑏 ] [𝑒𝑚𝑏 ] ...... [𝑒𝑚𝑏 ] [𝑒𝑚𝑏 ]

[User]

|Multimodal Item Information|
|---|

- Figure 2: Illustration of the Molar framework. The Multimodal Item Representation Model (MIRM) processes multimodal item information to generate item embeddings, while the Dynamic User Embedding Generator (DUEG) models user embeddings based on interaction histories for next-item prediction. First, MIRM is fine-tuned for multimodal feature alignment. Then, a joint optimization framework integrates ID-based and content-based user embeddings using a contrastive learning mechanism to enhance recommendation performance.

timodal Large Language Models (MLLMs) that integrate language with other modalities. The primary challenge lies in achieving effective model collaboration, with a focus on aligning modalities and understanding human intent. MLLMs like GPT-4 (OpenAI et al., 2024) and Gemini (Team

and slower inference speeds. To address these challenges, we propose Molar, a decoupled framework that separates the modeling of items and users for more efficient processing. This separation allows tailored-modeling strategies for each component, improving computational efficiency and recommendation accuracy. Our framework is composed of two key modules: the Multimodal Item Representation Model (MIRM) and the Dynamic User Embedding Generator (DUEG). MIRM is designed to compress multimodal features into compact embeddings, mitigating the computational burden of lengthy token sequences. DUEG then utilizes these embeddings to build user representations that capture dynamic user preferences. Together, these modules enable effective multimodal feature modeling and user preference prediction, setting the foundation for robust sequential recommendation.

- et al., 2023) have demonstrated exceptional capabilities in multimodal comprehension. Some studies (Wang et al., 2024; Lu et al., 2024; Zhang
- et al., 2024a) have concentrated on integrating LLMs with visual encoders. Inspired by these prior studies, we have developed Molar, which utilized MLLMs to align and fuse multimodal information to enhance sequential recommendation.

### 3 Methods

#### 3.1 Problem Formulation

We tackle the task of sequential recommendation, where the goal is to predict the next item In+1 that a user u ∈ U is likely to interact with, given their historical interaction sequence Hu = {I1,I2,...,In} arranged in chronological order. Each item Ii ∈ I comes with multimodal information, such as titles, textual descriptions, and images. Our approach leverages this multimodal information to improve the prediction accuracy of the next interaction.

#### 3.3 Multimodal Item Representation Model.

To effectively extract and encode item features, we introduce the Multimodal Item Representation Model (MIRM), denoted as fI. This encoder leverages a multimodal large language model (MLLM) to combine textual descriptions and images into a unified embedding representation. Although MLLMs excel in understanding text and images, their direct application to feature extraction remains limited. To address this, we append a special identifier, [Cur_Item], to the end of each item’s description, guiding the model to focus on extracting relevant features.

#### 3.2 Molar Overview

Traditional recommendation systems based on Large Language Models (LLMs) often suffer from inefficiencies when handling extensive user histories, as transforming these histories into lengthy token sequences results in high computational costs

The process begins by merging an item’s textual and image attributes into a unified description

L, augmented with a prompt to enhance model comprehension. This description is tokenized and processed by the MLLM, with [Cur_Item] appended at the end of the token sequence {t1,t2,...,tm,[Cur_Item]}. The model’s hidden state corresponding to [Cur_Item] is extracted as the item’s embedding:

⟨embi⟩ = fI(txti,imgi), (1)

where ⟨embi⟩ is the embedding of item Ii, txti is the textual description, imgi is associated image.

To enhance the quality of representations, MIRM undergoes multimodal fine-tuning with three complementary objectives:

Image-Text Alignment. Aligns visual features with textual descriptions, using item images to generate detailed descriptions. This alignment ensures that the model captures meaningful relationships between visual content and textual context, improving its ability to interpret multimodal data cohesively. The fine-tuning data for this objective (Image-Text, IT) uses item images as input and produces detailed textual descriptions as output.

Structured Attribute Processing. Converts structured metadata (e.g., price, material) into natural language descriptions for comprehensive feature encoding. This process allows the model to integrate diverse item attributes into a unified representation, enhancing its flexibility to handle heterogeneous data types. The fine-tuning data (Structured Attributes, SA) uses item titles and metadata (e.g., price, material, size, color) as input to generate detailed textual descriptions as output.

Temporal User Behavior Understanding. Captures temporal dynamics by predicting future items from historical interactions using multimodal inputs. This objective helps the model learn sequential patterns in user behavior, enabling it to better adapt to dynamic user preferences over time. The fine-tuning data (User Behavior, UB) consists of historical item interactions (descriptions and images) as input, with predicted next items as output.

These objectives collectively enable MIRM to produce robust, multimodal embeddings that integrate seamlessly into the subsequent user modeling process, bridging item representation with user preference prediction.

#### 3.4 Dynamic User Embedding Generator

Building on the item embeddings generated by MIRM, we design the Dynamic User Embedding Generator (DUEG), denoted as fU. This module constructs dynamic user representations based on their historical interactions, effectively capturing evolving preferences. Unlike MIRM, DUEG simplifies the structure by removing the word embedding layer from the MLLM, retaining the pretrained parameters for efficient embedding processing.

Given a user’s historical interaction sequence Hu = {I1,I2,...,In}, MIRM transforms each item Ii into an embedding embi. These embeddings are then processed by DUEG, which incorporates a special [User] token to represent the user’s dynamic preferences. This approach enables DUEG to predict the next likely item In+1 based on past interactions, formalized as:

Eu = fU(emb1,emb2,...,embn), (2) where Eu is the user embedding.

To optimize both MIRM and DUEG, we introduce two loss functions:

Point-wise Recommendation Loss. A binary cross-entropy loss distinguishes between positive and negative samples, encouraging accurate nextitem predictions:

Lbce = −(y · log(x) + (1 − y) · log(1 − x)),

(3) where y is the label vector, and x represents predicted logits for positive and negative samples. Details of this loss can be found in the Appendix B.

Alignment Loss. To bridge content-based and ID-based representations, we introduce a postalignment mechanism that uses a contrastive learning objective for both embeddings after DUEG processes multimodal content. This prevents premature integration of collaborative filtering into the LLM, ensuring essential collaborative information is preserved. The contrastive learning objective is defined as follows:

1 |U|

Lalign = −

|U|

exp(s(Euid, Eucon)/τ)

log

exp(s(Euid, Ejcon)/τ)

u=1

j∈K

exp(s(Eucon, Euid)/τ)

+ log

exp(s(Eucon, Ejid)/τ)

j∈K

,

(4)

where τ is the temperature parameter, K is the set of comparative instances containing one positive and K-1 negative examples. s(·,·) denotes the cosine similarity function, and Eid and Econ are the ID-based user embeddings from a traditional sequential recommendation model and the contentbased embeddings from DUEG, respectively.

The final training objective combines two losses:

Ltotal = Lbce + α · Lalign, (5)

where α balances their contributions.

By integrating multimodal item features with collaborative signals, DUEG enhances accuracy and robustness in user preference modeling, enabling a seamless transition to recommendation generation.

### 4 Experiments

In this section, we evaluate our proposed framework, Molar, on three real-world datasets and compare it against several baselines, including traditional sequential recommender models and stateof-the-art LLM-based methods. To assess the effectiveness of Molar, we conduct a comprehensive analysis addressing four research questions. Additionally, we investigate the impact of different DUEGs and the various input data modalities on the results.Furthermore, we perform ablation studies to investigate the impact of fine-tuning strategies and post-alignment, as well as evaluate the performance differences between our LLM-based user modeling approach and alternative user representation methods. The following research questions are explored:

- • RQ1: How does Molar perform compared with traditional sequential recommender models and LLM-based methods?
- • RQ2: What are the differences in performance between the LLM DUEG and alternative DUEGs for user representation?
- • RQ3: How do different data modalities impact the performance of Molar?
- • RQ4: How does the post-alignment model affect the performance of Molar?
- • RQ5: How does fine-tuning the MIRM combined with post-alignment training influence the overall performance of Molar?

Table 1: Statistics of Datasets.

Dataset Amazon PixelRec MovieLens # User 993,087 50,000 6,040 # Item 301,312 82,864 3,706 # Interaction 8,813,442 989,476 1,000,209

#### 4.1 Experimental Settings

- 4.1.1 Datasets and Evaluation Metrics

- • Amazon 2: Collected from the Amazon cloth online shopping platform.
- • PixelRec 3: An image dataset for recommender systems with raw pixels and text.
- • MovieLens 4: A commonly-used movie recommendation dataset that contains user ratings.

For all three datasets, we arrange the interaction sequences in sequential order. We utilize a leaveone-out approach to split the data into training, validation, and testing sets. Detailed statistics of the datasets are provided in Table 1. The evaluation metrics are Normalized Discounted Cumulative Gain (NDCG@K), Recall (Recall@K), which are evaluated on the full amount of data. The abbreviations N, and R are respectively used to denote NDCG, and Recall.

- 4.1.2 Implementation Details

We employ Qwen2vl-2b5 as the backbone model for both MIRM and DUEG (experiments with other MLLM backbones are presented in the appendix A). For each dataset, we create three types of data mixtures, each consisting of 10,000 data points, to fine-tune the MIRM. Additionally, we employ SASRec as the ID-based recommendation model for contrastive learning, with an embedding dimension same as the MIRM.

For all methods involving LLMs, each experiment is trained for a maximum of 5 epochs with a batch size of 128. A learning rate warm-up strategy is employed, initializing the learning rate at 1/100 of its maximum value 1e-4, and dynamically adjusting it over training steps using a cosine scheduler.

#### 4.2 Performance Comparison (RQ1)

In this section, we compare Molar against traditional, content-based, and llm-based baselines, taking into metrics of both NDCG and Recall on Pix-

- 2https://jmcauley.ucsd.edu/data/amazon/
- 3https://github.com/westlake-repl/PixelRec
- 4https://grouplens.org/datasets/movielens/
- 5https://github.com/QwenLM/Qwen2-VL

Table 2: Performance comparison of Molar with baseline models. The underlined values indicate the best and second-best results across all models. The abbreviations N and R represent Normalized Discounted Cumulative Gain (NDCG) and Recall, respectively. Statistically significant improvements are marked with * (p-value << 0.05). Overall, Molar consistently achieves superior performance across all datasets, demonstrating its effectiveness in leveraging multimodal and collaborative filtering features.

Amazon* PixelRec* Movielens* N@10 N@20 R@10 R@20 N@10 N@20 R@10 R@20 N@10 N@20 R@10 R@20

Methods

FPMC 0.1037 0.1059 0.1152 0.1232 0.0107 0.0129 0.0191 0.0290 0.0907 0.1129 0.1708 0.2756 GRU4Rec 0.1029 0.1054 0.1107 0.1190 0.0109 0.0127 0.0189 0.0284 0.0828 0.1081 0.1657 0.2664 SASRec 0.1080 0.1105 0.1188 0.1281 0.0131 0.0149 0.0207 0.0311 0.1116 0.1395 0.2137 0.3245 DuoRec 0.1281 0.1342 0.1406 0.1616 0.0147 0.0181 0.0241 0.0362 0.1530 0.1790 0.2704 0.3738

Traditional

SASRecBert 0.1116 0.1130 0.1275 0.1365 0.0131 0.0161 0.0238 0.0357 0.1172 0.1465 0.2244 0.3407 SASRecV it 0.1142 0.1187 0.1237 0.1373 0.0126 0.0155 0.0211 0.0317 0.1204 0.1499 0.2295 0.3481 SASRecBert+V it 0.1164 0.1179 0.1308 0.1437 0.0136 0.0167 0.0210 0.0315 0.1258 0.1567 0.2382 0.3599

Content-based

CoLLM 0.1298 0.1344 0.1388 0.1602 0.0173 0.0213 0.0296 0.0444 0.1658 0.1880 0.2895 0.4058 HLLM 0.1285 0.1351 0.1457 0.1668 0.0189 0.0232 0.0352 0.0528 0.1652 0.1933 0.2920 0.4037

LLM-based

Ours Molar 0.1407 01478 0.1580 0.1773 0.0197 0.0242 0.0359 0.0539 0.1768 0.2068 0.3124 0.4320

elRec, MovieLens, and Amazon datasets, to showcase the effectiveness and robustness of Molar.

Baselines. FPMC (Rendle et al., 2010), GRU4Rec (Tan et al., 2016), and SASRec (Kang and McAuley, 2018) are traditional sequential recommendation models based on Markov Chains, RNN, and attention mechanisms, respectively. DuoRec (Qiu et al., 2022) employs contrastive learning to extract discriminative information for sequential recommendation. SASRec-Content is a variant of SASRec that directly utilizes content feature representations as sequence inputs. It includes three versions: text-only, image-only, and a combination of text and image. CoLLM (Zhang et al., 2023b) and HLLM (Chen et al., 2024) are sequential recommendation models based on large language models (LLMs), both achieving state-of-the-art performance.

Main Results. We implemented the Molar framework on three datasets. The comparison with baseline models is summarized in Table 2. Key observations are as follows:

(a) Superior Performance Across Datasets. Molar consistently outperforms all baseline models across three datasets. Our method achieves over a 7% improvement in NDCG and Recall metrics on the MovieLens dataset, with similar enhancements on PixelRec and Amazon datasets compared to the strongest baseline. This demonstrates Molar effectively integrates traditional sequential information with the expansive knowledge and reasoning capabilities of LLMs. By leveraging user interaction sequences, Molar retains the strengths of collaborative filtering to capture user behavior patterns, while using MLLMs’ advanced visual and

language understanding to interpret complex user intents and contextual nuances. The synergy between sequential data and LLM-driven insights allows Molar to balance explicit user preferences with implicit, semantically rich information, enhancing recommendation relevance and accuracy.

- (b) Enhanced Multimodal Integration. By efficiently integrating textual and visual features through MLLM, Molar achieves a substantial improvement in recommendation quality. Unlike SASRec-Content, which separately uses Vision Transformers (Vit) (Dosovitskiy, 2020) for images and Bert (Devlin, 2018) for text, Molar leverages the MLLM as MIRM. This integrated approach results in a remarkable 44% improvement due to better item embedding extraction compared to SASRec-Content. The primary reason for this boost is the pretraining capabilities of MLLMs, which align and fuse visual and textual representations. This alignment allows for a deeper understanding of item features, streamlining the information processing pipeline and significantly enhancing the quality of the embeddings, thus improving recommendation performance.
- (c) Combining Semantics with Collaboration. While methods like HLLM and CoLLM improve on ID-based and content-based approaches, they fall short compared to Molar, particularly in NDCG metrics, due to their inability to capture collaborative knowledge essential for recommendations. LLMs excel in processing textual data but struggle with user-user and item-item interactions found in traditional systems. Molar bridges this gap by combining LLMs’ semantic understanding with the collaborative filtering strengths of traditional methods, enabling it to use both deep semantic insights

[Figure 14]

- Figure 3: Performance comparison of different DUEGs. Qwen2vl-2b is used as MIRM for all. The LLM backbone DUEG outperforms traditional DUEGs.

and relational knowledge for more accurate and relevant recommendations.

#### 4.3 Impact of DUEG (RQ2)

We conducted experiments to evaluate various representation methods for DUEGs, including FPMC, SASRec, GRU4Rec, and our proposed LLM Qwen2vl backbone as the DUEG.

As shown in Figure 3, the results indicate that Molar, using an LLM backbone as the DUEG, outperforms all other methods across the three datasets. This not only validates the effectiveness of our innovative user representation approach but also highlights the limitations of relying solely on textual and visual information (e.g., text and image metadata) or sequential information (e.g., behavioral ID tokens). Compared to the highest-performing baseline, SASRec, Molar demonstrates an average improvement of 6.0% on the PixelRec dataset and an even more substantial average improvement of 7.2% on the Amazon dataset. The superior performance of Molar can be attributed to the extensive pretraining of the LLM backbone, which imbues it with comprehensive world knowledge. Additionally, the alignment between pretraining, finetuning, and recommendation systems allows the training process to converge with only a few epochs (5 epochs), whereas other non-LLM baselines require prolonged training periods (e.g., 200 epochs for SASRec) to achieve convergence. These results show the superiority of the Molar architecture in both performance and training efficiency.

#### 4.4 Impact of Input Data Modality (RQ3)

To gain a thorough understanding of how various data modalities influence the performance of Molar, particularly the contribution of multimodal fusion and the integration of complementary knowledge from various modalities, we conducted an in-depth

- Table 3: Performance comparison with different modality inputs. The table highlights the impact of using Image Only, Text Only, and Image + Text inputs for sequential recommendation tasks. The combined modality (Image + Text) consistently achieves the best performance across all evaluation metrics, demonstrating the advantage of multimodal integration.

N@10 N@20 N@50 R@10 R@20 R@50

Image Only 0.0182 0.0217 0.0292 0.0329 0.0512 0.0858 Text Only 0.0181 0.0228 0.0296 0.0335 0.0514 0.0860 Image + Text 0.0197 0.0242 0.0313 0.0359 0.0539 0.0895

- Table 4: Performance comparison of different postalignment models for contrastive learning. Results show that stronger sequential models yield better performance, demonstrating the benefits of post-alignment.

Post-Alignment Model N@10 N@20 R@10 R@20 FPMC 0.0194 0.0237 0.0347 0.0527

GRU4Rec 0.0195 0.0240 0.0360 0.0531 SASRec 0.0197 0.0242 0.0359 0.0539 DuoRec 0.0200 0.0253 0.0371 0.0569

analysis on PixelRec, showing in Table 3.

Our findings demonstrate that multimodal fusion enhances recommendation performance. The fusion of multiple modalities significantly enhances recommendation performance. A comparative analysis of Molar on three input types reveals that the combined text with image input yields the best results. This can be attributed to the unique knowledge contributed by each modality, which cannot be captured by the others. MLLM effectively integrates this complementary information, demonstrating its potential as a robust foundation for multimodal recommendation tasks.

#### 4.5 Impact of Post-Alignment Models (RQ4)

In the process of post-alignment contrastive learning, integrating ID information results in varying degrees of improvement across different traditional sequential recommendation models. To verify the performance impact brought by different traditional sequential recommendation models, we conducted the experiments shown in Table 4.

The experimental results reveal a clear trend: stronger sequential recommendation models like DuoRec better support post-alignment contrastive learning, enhancing the integration of ID-based collaborative filtering signals into LLMs. This integration significantly boosts recommendation accuracy, coverage, and performance. DuoRec’s robust architecture captures richer user-item interaction patterns, enabling the LLM to leverage nuanced ID in-

Table 5: Ablation study on the PixelRec dataset. The table evaluates the impact of different fine-tuning data components (Image-Text, Structured Attributes, User Behavior) and the post-alignment module. Results demonstrate that using all fine-tuning components achieves optimal performance, while removing any single component degrades performance. The postalignment contrastive learning module is shown to be critical for maintaining high recommendation accuracy.

N@10 N@20 N@50 R@10 R@20 R@50 Full Model

###### Molar 0.0197 0.0242 0.0313 0.0359 0.0539 0.0895

Fine-Tuning Data

w/o IT 0.0186 0.0227 0.0298 0.0339 0.0512 0.0841 w/o SA 0.0189 0.0237 0.0302 0.0349 0.0528 0.0859 w/o UB 0.0183 0.0220 0.0287 0.0324 0.0495 0.0828 w/o ALL 0.0180 0.0219 0.0285 0.0313 0.0479 0.0808

Post-Alignment w/o CL 0.0182 0.0225 0.0294 0.0325 0.0496 0.0819

formation for top NDCG and Recall scores. These findings highlight the importance of selecting powerful sequential models for contrastive learning, as they refine the process and ensure coherent ID integration, unlocking the full potential of collaborative filtering for diverse scenarios.

#### 4.6 Ablation Study (RQ5)

To analyze the contributions of components in the proposed Molar method, particularly the finetuning of the MIRM on multimodal data and the post-alignment for user representation, we conducted an ablation study on the PixelRec dataset.

Effect of Fine-Tuning Data for MIRM. The fine-tuning data for MIRM comprises three key components:

- • Image-Text (IT): Input: Item images. Output: Detailed textual descriptions.
- • Structured Attributes (SA): Input: Item title and metadata. Output: Detailed descriptions.
- • User Behavior (UB): Input: Historical item interactions (descriptions and images). Output: Predicted subsequent items.

To evaluate the contribution of each fine-tuning data component, we systematically removed one type of fine-tuning data at a time during stage 1. Additionally, removing all three types (w/o ALL) effectively disables the fine-tuning stage. The results, presented in Table 5, show that the model achieves its best performance when all three fine-tuning data types are used together. Conversely, removing any single type of data leads to a noticeable decline in performance, highlighting the importance of each

component in enhancing the model’s overall effectiveness. Interestingly, the performance when structured attributes (w/o SA) are excluded remains higher than when either image-text (w/o IT) or user behavior (w/o UB) data are omitted. This indicates that, even in the absence of structured attributes, the combination of image-text and user behavior data can effectively fine-tune MIRM, enabling it to learn and align meaningful representations.

Impact of Post-Alignment Contrastive Learning. To examine the role of post-alignment in user representation, we conducted an ablation experiment by removing the post-alignment contrastive learning (w/o CL) module. As shown in Table 5, removing the contrastive learning module significantly decreases performance, particularly in the NDCG metric. This highlights the critical importance of incorporating ID-based information and collaborative signals. Post-alignment ensures effective feature-level communication and interaction between ID-based models and content-based MLLM, enabling the model to fully leverage multimodal information in a collaborative filtering context.

The ablation study highlights the importance of each fine-tuning data component and the postalignment contrastive learning module. Finetuning on multimodal data ensures comprehensive item representations, while post-alignment bridges the gap between ID-based and contentbased models, significantly enhancing the overall performance of the Molar framework.

### 5 Conclusion

In this paper, we introduced Molar, a novel framework for sequential recommendation that bridges the gap between collaborative filtering and multimodal content modeling using large language models (LLMs). While traditional LLM-based approaches excel in semantic understanding, they lack the ability to incorporate collaborative filtering signals, limiting their recommendation performance. To overcome this limitation, Molar integrates multimodal data (textual and non-textual) with ID-based collaborative signals, leveraging an MLLM to generate unified item representations and a post-alignment mechanism to align user embeddings effectively. By combining the strengths of multimodal content modeling and collaborative filtering, Molar captures both user interests and contextual semantics, enabling precise and personalized recommendations. Extensive experimental

results demonstrate that Molar consistently outperforms traditional methods and state-of-the-art LLM-based baselines, validating its ability to fully exploit multimodal data and collaborative signals for sequential recommendation tasks.

Limitation. While Molar effectively integrates multimodal large language models (MLLMs) into sequential recommendation tasks, several limitations remain. First, the framework requires multitask fine-tuning to optimize multimodal representations, which can be time-intensive and may hinder its deployment in real-time applications. Second, due to computational constraints, we are unable to train larger language models, and the quality of the generated multimodal item representations heavily depends on the underlying capabilities of the MLLMs. If the base models are suboptimal, the overall recommendation performance may degrade. In future work, we aim to develop an end-to-end training framework and incorporate more advanced MLLMs with larger parameter sizes to enhance the quality of generated representations, thereby improving the overall efficacy of Molar.

### Acknowledgement

This research was supported by grants from the Provincial Natural Science Foundation of Anhui Province (No. 2408085QF193) and the Fundamental Research Funds for the Central Universities of China (No. WK2150110032, No. PA2024GDSK0112).

### References

Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, pages 1007–1014.

Junyi Chen, Lu Chi, Bingyue Peng, and Zehuan Yuan. 2024. Hllm: Enhancing sequential recommendations via hierarchical large language models for item and user modeling. arXiv preprint arXiv:2409.12740.

Mingyue Cheng, Zhiding Liu, Qi Liu, Shenyang Ge, and Enhong Chen. 2022. Towards automatic discovering of deep hybrid network architecture for sequential recommendation. In Proceedings of the ACM Web Conference 2022, pages 1923–1932.

Mingyue Cheng, Fajie Yuan, Qi Liu, Xin Xin, and Enhong Chen. 2021. Learning transferable user representations with sequential behaviors via contrastive

pre-training. In 2021 IEEE International Conference on Data Mining (ICDM), pages 51–60. IEEE.

Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Alexey Dosovitskiy. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Luke Friedman, Sameer Ahuja, David Allen, Zhenning Tan, Hakim Sidahmed, Changbo Long, Jun Xie, Gabriel Schubiner, Ajay Patel, Harsh Lara, et al. 2023. Leveraging large language models in conversational recommender systems. arXiv preprint arXiv:2305.07961.

David Goldberg, David Nichols, Brian M Oki, and Douglas Terry. 1992. Using collaborative filtering to weave an information tapestry. Communications of the ACM, 35(12):61–70.

Aaron Grattafiori, Abhimanyu Dubey, and Abhinav Jauhri et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Wang-Cheng Kang and Julian McAuley. 2018. Selfattentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pages 197–206. IEEE.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. 2023. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274.

Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 426–434.

Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37.

Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan He. 2023. Llara: Aligning large language models with sequential recommenders. arXiv preprint arXiv:2312.02445.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. 2024. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525.

Yucong Luo, Mingyue Cheng, Hao Zhang, Junyu Lu, and Enhong Chen. 2024. Unlocking the potential of large language models for explainable recommendations. In International Conference on Database

Systems for Advanced Applications, pages 286–303. Springer.

Lin Ning, Luyang Liu, Jiaxing Wu, Neo Wu, Devora Berlowitz, Sushant Prakash, Bradley Green, Shawn O’Banion, and Jun Xie. 2024. User-llm: Efficient llm contextualization with user embeddings. arXiv preprint arXiv:2402.13598.

OpenAI, Josh Achiam, Steven Adler, and Sandhini Agarwal et al. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824.

Ruihong Qiu, Zi Huang, Hongzhi Yin, and Zijian Wang. 2022. Contrastive learning for representation degeneration problem in sequential recommendation. In Proceedings of the fifteenth ACM international conference on web search and data mining, pages 813–823.

Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation learning with large language models for recommendation. In Proceedings of the ACM on Web Conference 2024, pages 3464–3475.

Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web, pages 811–820.

Paul Resnick and Hal R Varian. 1997. Recommender systems. Communications of the ACM, 40(3):56–58.

Yong Kiam Tan, Xinxing Xu, and Yong Liu. 2016. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 17–22.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Shoujin Wang, Liang Hu, Yan Wang, Longbing Cao, Quan Z Sheng, and Mehmet Orgun. 2019. Sequential recommender systems: challenges, progress and prospects. arXiv preprint arXiv:2001.04830.

Yunjia Xi, Weiwen Liu, Jianghao Lin, Xiaoling Cai, Hong Zhu, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, and Yong Yu. 2024. Towards openworld recommendation with knowledge augmentation from large language models. In Proceedings of the 18th ACM Conference on Recommender Systems, pages 12–22.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2024. A survey on multimodal large language models. National Science Review, page nwae403.

Chao Zhang, Shiwei Wu, Haoxin Zhang, Tong Xu, Yan Gao, Yao Hu, and Enhong Chen. 2024a. Notellm: A retrievable large language model for note recommendation. In Companion Proceedings of the ACM on Web Conference 2024, pages 170–179.

Chiyu Zhang, Yifei Sun, Jun Chen, Jie Lei, Muhammad Abdul-Mageed, Sinong Wang, Rong Jin, Sem Park, Ning Yao, and Bo Long. 2024b. Spar: Personalized content-based recommendation via long engagement attention. arXiv preprint arXiv:2402.10555.

Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. 2024c. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601.

Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023a. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001.

Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2023b. Collm: Integrating collaborative embeddings into large language models for recommendation. arXiv preprint arXiv:2310.19488.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for clickthrough rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1059–1068.

### A Impact of Different MLLM Backbone

Table 6: Comparison of Different MLLM Backbone.

MLLM Backbone Training Type N@10 N@20 R@10 R@20 Qwen2-VL-2B Full-tuning 0.0197 0.0242 0.0359 0.0539 InternVL2.5-2B 6 Full-tuning 0.0191 0.0237 0.0349 0.0521 deepseek-vl-1.3b 7 Full-tuning 0.0183 0.0225 0.0334 0.0499 Qwen2-VL-7B LoRA 0.0200 0.0251 0.0369 0.0555 Llama-3.2-11B-Vision 8 LoRA 0.0194 0.0249 0.0357 0.0542

To evaluate the impact of different MLLM backbones on the performance of Molar, we conduct comparative experiments using MLLMs of various backbones and sizes. Due to computational constraints, we are unable to fine-tune models with 7B parameters or larger in full, so we employ LoRA training for models with 7B parameters and above.

As shown in Table 5, for models of the same size, performance variations across different backbones are observed, with Qwen2vl achieving the best results. This suggests that, beyond model size, the choice of backbone plays a crucial role in determining the quality of the recommendations. Interestingly, as the model size increases, there is a consistent improvement in recommendation performance, highlighting the advantages of scaling up model capacity. Even when fine-tuned with LoRA, the 7B Qwen2vl model consistently outperforms the 2B counterpart, indicating that the larger model not only benefits from increased parameters but also capitalizes on the specific architectural strengths of Qwen2vl. These findings suggest that, while model size is an important factor, selecting an appropriate backbone could be equally crucial in optimizing performance, particularly when computational resources are limited.

### B Detailed Point-wise Recommendation Loss.

To enhance the model’s accuracy in predicting the next item, we employ a binary cross-entropy (BCE) loss function. In our training process, each positive sample is paired with a negative sample using a 1:1 negative sampling strategy. The target item embedding calculated from MIRM consists of a positive item (pos) and a negative item (neg). For each pair, we define label vector y = [1,0] and generate predicted logits x = [xpos,xneg]. The BCE loss is calculated as:

Lbce = −(y · log(x) + (1 − y) · log(1 − x))

(6)

Minimizing this loss encourages the model to assign higher probabilities to positive samples and lower probabilities to negative samples, thereby improving its ability to distinguish relevant items for accurate next-item predictions.

