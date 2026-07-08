## The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective

Zhen Qin*, Daoyuan Chen*, Wenhao Zhang, Liuyi Yao, Yilun Huang, Bolin Ding, Yaliang Li†, Shuiguang Deng†, Senior Member, IEEE

### arXiv:2407.08583v2[cs.AI]5Aug2024

Abstract—The rapid development of large language models (LLMs) has been witnessed in recent years. Based on the powerful LLMs, multi-modal LLMs (MLLMs) extend the modality from text to a broader spectrum of domains, attracting widespread attention due to the broader range of application scenarios. As LLMs and MLLMs rely on vast amounts of model parameters and data to achieve emergent capabilities, the importance of data is receiving increasingly widespread attention and recognition. Tracing and analyzing recent data-oriented works for MLLMs, we find that the development of models and data is not two separate paths but rather interconnected. On the one hand, vaster and higher-quality data contribute to better performance of MLLMs; on the other hand, MLLMs can facilitate the development of data. The co-development of multi-modal data and MLLMs requires a clear view of 1) at which development stages of MLLMs specific data-centric approaches can be employed to enhance certain MLLM capabilities, and 2) how MLLMs, utilizing those capabilities, can contribute to multi-modal data in specific roles. To promote the data-model co-development for MLLM community, we systematically review existing works related to MLLMs from the data-model co-development perspective. A regularly maintained project associated with this survey is accessible at https://github.com/modelscope/data-juicer/blob/main/docs/awesome llm data.md.

Index Terms—Multi-Modal Data, Multi-Modal Large Language Models, Data-Centric AI, Data-Model Co-Development

✦

1 INTRODUCTION

# L

ARGE language models (LLMs) demonstrate impressive performances across a wide range of tasks in recent

years, with their associated technologies making significant advancements. Since human senses are not limited to text modality, multi-modal LLMs (MLLMs) have come into view, such as Gemini-1.5 [1] and Sora [2] that are capable of processing inputs or outputs in modalities beyond text, and GPT-4o [3] and NExT-GPT [4] that can even interact between multiple modalities in both input and output. MLLMs have gained widespread attention in the past two years. As shown in Fig. 1, research related to MLLMs has been emerging at an increasing speed since 2023.

The outstanding performance of MLLMs stems from the emergent abilities brought by the scaling up in the number of parameters [5]. Many works indicate that scaling up the model size needs to be complemented by even more massive amounts of data [6], [7], [8], such as scaling law [9], [10]. In light of this, a series of works shift the focus from merely model architectures and training techniques to datacentric approaches that focus on the curation of data [11], [12], [13], [14], [15], [16], which serve as the basis to unlock the potential of large models. From Fig. 1, among existing papers for MLLMs, those related to data-centric approaches

- • * Equal Contributions. Work done during Z. Qin’s internship at Alibaba Group. † Corresponding Authors
- • Z. Qin and S. Deng are with the College of Computer Science and Technology, Zhejiang University. E-mail: {zhenqin, dengsg}@zju.edu.cn
- • D. Chen, W. Zhang, L. Yao, Y. Huang, B. Ding and Y. Li are with Alibaba Group. E-mail: {daoyuanchen.cdy, zwh434786, yly287738, lielin.hyl, bolin.ding, yaliang.li}@alibaba-inc.com

| | | |
|---|---|---|
| | | |

Fig. 1. The trends of cumulative numbers of papers on arXiv 1 related to MLLMs and those related to both MLLMs and data, respectively.

likewise exhibit a strong growth trend on the count and occupy a significantly important portion.

As technical works related to MLLMs continue to emerge, some reviews for MLLMs have gradually appeared as well [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33]. These surveys are mainly conducted from the model-centric perspective, however, the importance of the data needs further emphasis. One recent survey emphasizes the data-centric perspective for MLLMs [34], organizing existing data-centric approaches according to the stages in the proposed data pipeline.

Actually, the development of data and models is intertwined rather than separate. Large-scale and high-quality data enhance model performance, while well-trained models can help further improve the data. It reduces labor costs and expands data quantity, and has been successfully demonstrated by the training of Segment Anything model (SAM)

1. The statistics are obtained with the advanced search of arXiv, merely for reflecting trends and may not be absolutely precise on numerical results, especially for subtle connections to multi-modal data.

[35] which utilizes segmentation masks that need to be annotated for training. As the SAM becomes more proficient with training, it gradually replaces humans in annotation tasks, and thereby a repetitive cycle that improves both the model and the dataset is formed. Such a progressive and virtuous cycle advances MLLM development, i.e., MLLMs benefited from high-quality datasets can help improve the training data, which in turn further enhances the MLLMs.

Data-model co-development is promising for MLLMs but has not been fully investigated. From our investigation, there currently lacks a review for MLLMs from the perspective of data-model co-development. Existing surveys have yet to establish a relationship between data-centric approaches and MLLM capabilities, and have not clearly articulated how the capabilities of MLLMs can assist in datasets. The key to enabling the data-model co-development for MLLMs lies in elucidating which data approaches can enhance each specific capability of MLLMs, as well as understanding the roles that models can assume to improve multi-modal data. Thus, this survey aims to advance the data-model co-development for MLLMs by answering the following research questions with a comprehensive review:

- • RQ1: At which stage throughout the lifecycle of an MLLM can specific data-centric approaches be employed to enhance specific MLLM capabilities?
- • RQ2: What roles can models play in facilitating the curation of multi-modal data, and what specific capabilities of models are leveraged in each case?

To answer these two key research questions, we first propose a novel taxonomy grounded in the data-model codevelopment paradigm for MLLMs. We categorize previous efforts into two principal dual types: the data contributions to models and the reciprocal model contributions to data, establishing their in-depth connections anchored in the capabilities of MLLMs. Subsequently, we provide a comprehensive examination of existing works for MLLMs from the data-model co-development perspective, which uncovers considerable untapped potential for advancing the data-model co-development paradigm, primarily due to the lack of dedicated focus on the synergistic interplay between data and models. Built upon the insights garnered, we delineate several progressive future directions in the data-model co-development of MLLMs, forming a roadmap to better leverage the complementarity between data and models, spanning from infrastructures to various self-boosting degrees of data-model co-development. The main contributions of this survey are three-fold:

- • A New Perspective for MLLM Development: we propose a new taxonomy that emphasizes the synergy between multi-modal data and MLLMs, aiming to mine the mutual benefits for both data and model development. This taxonomy is systematically organized based on the hierarchy of data-related techniques essential for developing MLLMs, offering a clear view of the whole life cycle to advance MLLMs for researchers and developers.
- • An Up-to-date Review for MLLMs from Data-Model CoDevelopment Perspective: we systematically review the fast-growing works on MLLMs and elucidate 1) which MLLM capabilities can be enhanced by specific datacentric approaches, and 2) how the capabilities of well-

trained models in turn nourish multi-modal data. To the best of our knowledge, this is the first survey on MLLMs from data-model co-development perspective.

• A Roadmap for Future MLLMs: focusing on the internal interplay between data and models for MLLMs, we provide a roadmap progressively organized by several advanced and promising directions. With this work, we hope to offer a source of inspiration and guidance to both academic researchers and industry practitioners navigating the evolving landscape of MLLMs.

Organizations. The rest of this survey is organized as follows. Sec. 2 provides preliminaries, including the background, taxonomy, comparisons to related surveys, and MLLM architectures. Sec. 3 discusses data contributions for scaling up MLLMs. Sec. 4 reviews data contributions for improving the usability of MLLMs. Sec. 5 describes the capabilities of models that directly help dataset curation for MLLMs. Sec. 6 explores the applications of models acting as data scientists to assist in dataset curation for MLLMs. Sec. 7 lists some public datasets for MLLMs, with the participation of models for curation indicated. Sec. 8 discusses a roadmap for future development of MLLMs.

2 PRELIMINARY

2.1 Background

Data-model co-development for MLLMs attempts to improve model performance while leveraging this model to optimize the data, with the final objective aimed at a wellperforming MLLM. It is characterized by dynamic training data for a dynamic model [36]. Given the current lack of a formal definition for data-model co-development, we attempt to provide a formal definition with a large multimodal generation model as an example, which is capable of generating outputs in modalities different from the input or in more modalities than the input, such as image-to-text dialog [37] and text-to-image synthesis [38].

Definition 1. (Data-Model Co-Development for Large Generation Model). Let 1) pdata(u) denote an ideal distribution that each real-world data u follows, where u is generally composed by a context c and a response r, denoted by u = {c,r}, 2) w be the large generation model to be trained, which generates a response r′ given a context c′, as r′ = w(c′), 3) D = uD1 ,uD2 ,... be the dataset to train w, which is usually characterized by large scale, 4) D∗ denote the ideally optimal D, and 5) Q(·,·) denote the function that quantifies the similarity between two data samples, with a larger value indicating higher similarity, data-model co-development can be formalized as a bi-level optimization problem, as

E{c,r}∈D∗Q(r,w(c)), (1) s.t. D∗ ∈ arg min

max

w

Eu∼pdata(u),∀x={c,r}∈D − Q(u,x). (2)

D

Eq. (2) curates data for training w. Eq. (1) is the final objective that teaches w to produce data resembling realworld data. Data-model co-development can be classified into two paradigms based on the optimization tools for D: 1) (Self-Boosted Paradigm). The model w to be trained

is also used to improve dataset D, where Eq. (2)

§ 4 For Usability of MLLMs

§3 For Scaling of MLLMs

§ 7 Public Datasets for MLLMs

§4.1 For Instruction Responsiveness

§

| |3.1 For Scaling Up of MLLMs: Larger Datasets<br><br>• Data Acquisition<br>• Data Augmentation<br>• Data Diversity<br><br><br>3.2 For Scaling Effectiveness of MLLMs: Better Subsets<br><br>§<br><br>§<br><br>|[Figure 1]<br><br>|§<br><br>§<br><br>[Figure 2]<br><br>• Prompt Design<br>• Single-Hop • Multi-Hop §4.3 For Ethics<br><br>§4.2 For Reasoning Ability<br><br>• ICL Data<br><br><br>Human-Behavior • Alignment Data| |
|---|---|---|---|---|
| |[Figure 3]<br><br>• Data Condensation<br>• Data Mixture<br>• Data Packing<br>• Cross-Modal Alignment<br>| |§• Data Toxicity §4.4 For Evaluation<br><br>• Understanding<br>• Retrieval<br>• Data Privacy & IP<br>• Generation<br>• Reasoning<br><br><br>|Multi-Modal Data Contributions for MLLMs<br><br>|

- • Datasets with models as data sources
- • Datasets without models as data sources
- • MLLM-Based Data Discovery
- • Modality-Compatibility Detection with MLLMs
- • Automatic Knowledge Transfer for MLLMs

§ 8 Roadmap for Future MLLMs

§8.1 Data-Model Co-Development Infrastructures

- § 8.2 Externally-Boosted MLLM Development

- § 8.3 Self-Boosted MLLM Development

[Figure 4]

[Figure 5]

6 For Insight into Multi-Modal Data 5 For Synthesis of Multi-Modal Data

§ §

- • Self Data Scaling with MLLMs
- • Self Data Condensation with MLLMs
- • RL from Self Feedback of MLLMs

| |§<br><br>6.1 Model as a Data Navigator<br>6.2 Model as a Data Extractor<br><br><br>§<br><br>§<br><br>| |§<br><br>§<br><br>5.1 Model as a Data Creator<br>5.2 Model as a Data Mapper<br><br><br>§<br><br>§<br><br>| |
|---|---|---|---|---|
|Model Contributions Multi-Modal Data|[Figure 6]<br><br>§<br><br>§<br><br>[Figure 7]<br><br>6.3 Model as a Data Analyzer<br>6.4 Model as a Data Visualizer<br><br><br>§<br><br>§<br><br>| |§<br><br>§<br><br>[Figure 8]<br><br>5.3 Model as a Data Filter<br>5.4 Model as a Data Evaluator<br><br><br>§<br><br>§<br><br>| |

[Figure 9]

[Figure 10]

Data Contributions for MLLMs Model Contributions for Data Resources & Future Directions

for

[Figure 11]

[Figure 12]

[Figure 13]

Fig. 2. Taxonomy for MLLMs from the data-model co-development perspective and overview of §3-8 with their inter-relationships. Data contributions for MLLMs (§3 & 4) are organized in an objective-driven manner and ordered according to MLLM development stages, i.e., first scaling up for better performance then improving the usability. Model contributions for data (§5 & 6) are organized by the roles played by models.

and Eq. (1) are usually optimized alternatively. In this paradigm, Eq. (2) can be further reformulated as D∗ ∈ arg minD|w Eu∼pdata(u),∀x={c,r}∈DQ(u,x).

2) (Externally-Boosted Paradigm). As an alternative, D can be curated with a well-trained model w∗ such as GPT-4V, or even human efforts. Accordingly, we can rewrite Eq. (2) as D∗ ∈ arg minD|w∗ Eu∼pdata(u),∀x={c,r}∈DQ(u,x) or D∗ ∈ arg minD|human Eu∼pdata(u),∀x={c,r}∈DQ(u,x). From our survey, the self-boosted paradigm is proven effective for uni-modal LLMs [39] and vision foundation models [35], [36], but still lacks investigation for MLLMs.

Data-model co-development is promising as increasing attention shifts towards data-centric approaches, where data often serves as the primary variable, rather than merely focusing on model architectures [11], [12], [13], [16], [34], [40]. As MLLMs require increasingly large volumes of data, models are gradually used to assist or directly build the data samples. Thus, the development of data and models have become interdependent and inseparable: massive and high-quality data can lead to well-performing MLLMs, and in turn, well-performing MLLMs can help construct more high-quality data. Therefore, it is necessary to understand how data approaches enhance specific capabilities of MLLMs, and how MLLMs assist in data approaches, thereby advancing the data-model co-development for MLLMs.

##### 2.2 Taxonomy

The taxonomy and the relationships between the items are illustrated in Fig. 2. According to our investigations, both the contributions of data to models and vice versa can be categorized into two major types. The data contributions to models are organized in an objective-driven manner and arranged according to their sequence in technical stages of MLLM development. First, to elicit foundational abilities from MLLMs, it is crucial to provide more and higherquality data for MLLM training, aiming at the scaling of

MLLMs (§3). Some works focus on providing large-scale datasets to scale up MLLMs (§3.1), while others enhance the effectiveness of scaling by improving data quality and organizing the data strategically (§3.2). After obtaining an MLLM with foundational capabilities, a series of approaches could be performed around data to enhance its usability from various aspects, including the instruction responsiveness (§4.1), reasoning ability (§4.2) and ethics (§4.3), followed by comprehensive evaluations (§4.4).

As datasets grow in size, their curation gradually relies on well-trained models (MLLMs or their components such as LLMs and foundation models). We review a series of data-centric approaches co-piloted with models and group them into two main categories, organized in a role-driven manner and arranged according to their sequence in a data pipeline, i.e., 1) for the synthesis of data, where models directly participate in data curation to alleviate repetitive tasks for humans by acting as a data creator (§5.1), mapper (§5.2), filter (§5.3), and evaluator (§5.4); and 2) for insights into data, where models perform as data scientists to provide insights on multi-modal data by acting as a data navigator (§6.1), analyzer (§6.3), extractor (§6.2), and visualizer (§6.4).

Based on these investigations, we summarize the public datasets for MLLMs and clarify the models’ participation during data curation (§7). Finally, we identify some progressive future directions for MLLMs, forming a roadmap for the future development of MLLMs (§8).

##### 2.3 Differences from Related Surveys

The popularity of MLLMs has led researchers to catalog existing works. Current surveys on MLLMs mainly focus on model-centric perspectives with the taxonomy based on: 1) training techniques that highlight the training stages and algorithms [17], [18], [19], [20], [21], [22], [23], [24], [25]; 2) MLLM architectures [18], [19], [21], [22], [25], [26], [27], [28], [29], [30]; 3) MLLM capabilities for reasoning [20], [27], [31]

TABLE 1: Qualitative comparisons between our survey and closely related surveys. A survey may appear in more than one row due to the diversity of emphases. Model-centric surveys for uni-modal LLMs are omitted due to their secondary relevance to this survey.

Modality Perspective Taxonomy Highlights References

[17], [18], [19], [20], [21], [22], [23], [24], [25]

Based on Training Techniques Training stages & algorithms for MLLMs

[18], [19], [21], [22], [25], [26], [27], [28], [29], [30]

Based on MLLM Architectures The architectural components of MLLMs

Model-Centric

Reasoning abilities of MLLMs [20], [27], [31] Applications of MLLMs [18], [32]

Multi-Modal

Based on MLLM Capabilities

Key considerations & applications in MLLM System Design

Based on MLLM Systems

[19], [33]

Data approaches for MLLMs across stages in the data pipeline

Data-Centric Based on Data Pipeline

[34]

Data approaches for LLMs across stages in the data pipeline

Based on Data Pipeline

[11], [41], [42]

Uni-Modal Data-Centric

Based on Training Stage Data-centric approaches for each training stage [43] Based on Adopted Techniques Techniques for specific data approaches [12], [44], [45]

- • Matching data-centric approaches to MLLM properties (data contributions for MLLMs)
- • Roles acted by models to facilitate multi-modal data (model contributions for multi-modal data)

Based on Mutual Benefits between Model & Data

Data-Model Co-Development

Multi-Modal

Ours

and applications [18], [32]; or 4) MLLM systems which focus on the key considerations and applications [19], [33]. These model-centric surveys facilitate the development of MLLMs, yet have not given dedicated consideration to data.

The importance of data for LLMs has been emphasized by some LLM surveys from data-centric perspectives, with the taxonomy based on: 1) data approaches across stages in the data pipeline [11], [41], [42]; 2) training stages [43]; or 3) the adopted techniques for specific data-centric approaches [12], [44], [45]. These surveys foster the attention of LLM communities toward a data-centric perspective. One recent survey extends the data-centric perspective from LLMs to MLLMs [34], which organizes existing works according to their sequence in the data pipeline.

Given 1) the potential of data-model co-development for MLLMs, and 2) the fact that existing data-centric surveys fall short of establishing connections between data-centric approaches and specific MLLM capabilities, there is currently a lack of and an urgent need for a survey to articulate the contributions made by multi-modal data and MLLMs to each other. Thus, we provide this up-to-date survey on MLLMs from a data-model co-development perspective, clarifying how data technologies facilitate the development of MLLMs and then demonstrating how models can promote multimodal data technologies. The above comparisons between the related surveys and ours are summarized in Table 1.

##### 2.4 MLLM Architecture

We briefly introduce MLLM architecture for reference. An MLLM typically contains: 1) an LLM such as LLaMA [46]; 2) one or more foundation models (encoders) to encode nontext data (e.g., ViT [47] and CLIP [48]); and 3) one or more projectors to align the encoded features of non-text data with the feature space of LLMs. This setup allows MLLMs to generate textual responses to multi-modal inputs [27]. Further, by incorporating 4) modality-specific generators

such as Stable Diffusion [38], MLLMs can produce multi-modal contents [4], [8]. These components may be trained in different stages [49] and with different types of datasets [37], [50]. We do not focus on the pretraining of LLMs, for which we refer readers to the surveys in Table 1.

#### 3 MULTI-MODAL DATA CONTRIBUTIONS FOR MLLMS: SCALING

Building MLLMs with satisfactory performance requires large-scale and high-quality multi-modal data. This section summarizes existing works contributing to large-scale MLLMs by providing datasets, organized by the logical sequence of top-level design and iterative optimization, i.e., first to scale up the cardinality in multi-modal datasets (§3.1), followed by enhancing the scaling effectiveness of datasets (§3.2). The organization of this section is illustrated in Fig. 3. After introducing existing works, a brief summarization and discussion are provided in Sec. 3.3.

##### 3.1 For Scaling Up of MLLMs: Larger Datasets

The excellent performance of MLLMs benefits from a larger number of parameters, especially when the size of model parameters reaches a certain level, abilities that traditional multi-modal models lack begin to emerge, such as OCR-free math reasoning [27]. Larger models require larger-scale data [9], [51], especially for MLLMs which extend the feature space of inputs and/or outputs over LLMs. This subsection summarizes works scaling up MLLMs by providing large-scale data, focusing on data acquisition (§3.1.1), data augmentation (§3.1.2) and data diversity (§3.1.3).

3.1.1 Data Acquisition

Data acquisition, a.k.a. data collection, acquires raw data to support large-scale datasets. Existing works adopt various data sources, including: crawling from the web [14], [48],

For Encoders and Decoders For Projectors For Fine-Tuning

Data Acquisition (§3.1.1)

| | |
|---|---|
| | |

For Scaling Up of MLLMs: Larger Datasets (§3.1)

Traditional Random Augmentation Generative Augmentation

Data Augmentation (§3.1.2)

Multi-ModalDataContributions

For Single-Modality Perception Abilities For Cross-Modality Cognition Abilities

forMLLMs:Scaling

Data Diversity (§3.1.3)

Data Deduplication Low-Quality Data Filtering Kernel Set Construction

Data Condensation (§3.2.1)

| | |
|---|---|
| | |
| | |
| | |

Mitigating Distribution Bias Exploiting Distribution Bias

Data Mixture (§3.2.1)

For Scaling Effectiveness of MLLMs: Better Subsets (§3.2)

For Better Pretraining For Long-Context Support

Data Packing (§3.2.3)

Joint Embedding Space Text-Centric Anchoring

Cross-Modal Alignment (§3.2.4)

- Fig. 3. Organization of data approaches tailored for scaling of MLLMs.

[52], [53], [54], [55], exploiting existing datasets [49], [53], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], employing human efforts [4], [55], [59], [67], [68], [69], synthetic with well-trained MLLMs such as GPT-4V [4], [37], [54], [56], [63], [64], [67], [69], [70], [71], [72], or simulators [73]. The components of MLLMs may be trained in different stages. Taking LLaVA [49] as an example 1, these stages include: 1) pretraining of encoders and/or generators for basic understanding and/or generation, 2) pretraining of projector(s) for feature space alignment, and 3) fine-tuning of the entire or parts of the MLLM to promote task responsiveness or downstream-task performance. Generally, these stages consume different types of datasets [37], [50]

For Encoders and Decoders. Pretraining of encoders typically relies on massive any-text pairs which are easy to obtain. For example, to train the visual encoder in an MLLM, image-text pairs can be collected from the web [48], merging existing datasets [74] or even purchasing [66]. Similarly, audio-text pairs can also be collected from websites [57]. The significance of scaling up datasets for vision-language pretraining has been experimentally demonstrated by [74].

For Projectors. Pretraining of the projectors for feature alignment usually consumes the instruction data converted from existing any-text pairs. Existing works obtain such data by converting any-text pairs into instruction styles such as conversations with native captions as responses [49], providing fine-grained information such as background elements and notable features [37], or enriching the captions with contextual details such as physical appearance [67].

For Fine-Tuning. Fine-tuning usually consumes carefully designed instructions to further enhance MLLMs’ general instruction-following capabilities or downstream-task

1. The terms of pretraining and fine-tuning originate from [49].

performance. For instruction following, data are typically organized in the form of QA pairs by simulating conversations between humans and assistants, using LLMs such as GPT-4 [49] or MLLMs such as GPT-4V [37], [60], [61]. The instructions could be improved with more advanced tasks such as listing all the items marked in the images one by one [60]. Compared to instruction data for projectors, these data typically contain more elements of reasoning and contextual understanding. MLLMs’ downstream-task performance can be improved by fine-tuning with instruction data targeted at specific downstream tasks such as audio retrieval [58], chart understanding [59], [68], [71], user interface (UI) assessment [55], and high-resolution visual understanding [63].

3.1.2 Data Augmentation

Data augmentation expands and diversifies a dataset by transforming existing data or adding synthetic data, thereby enhancing model generalization.

Traditional Random Augmentation. Data augmentation is widely applied to improve the generalization of deep learning models, such as random cropping, flipping, scaling and color transformation for vision tasks, and random deletion and swapping for text tasks. MLLMs also benefit from these simple yet effective techniques, e.g., random cropping and flipping for vision encoders [75] and projectors [76], [77], and character, word and sentence-level text augmentation for visual instruction tuning [78]. For data involving temporal information such as audio and video, time masking and random scaling could be adopted [79].

Generative Augmentation. Random augmentation can disrupt the semantics in multi-modal datasets. Well-trained LLMs such as LLaMA [46] can be employed to rewrite text descriptions [15], enhancing the diversity of vocabularies and sentence structures while preserving the semantics.

Such a character has benefited text-based person retrieval [80], human-action description [81] and image captions [82]. In some datasets with two (implicit) classes, different types of data may be imbalanced, causing MLLMs to overfit to one class of data. By synthesizing scarce negative samples in a dataset that teaches MLLM to use tools, MLLMs can be taught to determine whether a tool is truly needed [83] instead of always outputting that tools are needed. Similarly, issues with insufficient positive samples in image retrieval datasets can be addressed through caption generation and template-based text modification [84].

- 3.1.3 Data Diversity Diversified data help MLLMs perform well across different situations while alleviating the bias in data distribution.

For Single-Modality Perception Abilities. Encoders for other modalities in MLLMs benefit from diversified data sources, thereby improving the perception ability of MLLMs for individual modalities. It is found that Flamingo [85] trained with a mixture of complementary large-scale multimodal datasets performs significantly better than the models trained on any single dataset. Diverse datasets have a positive impact on the understanding of both audio [58] and images [14]. The diversity of concepts in the dataset is also positively correlated with the accuracy of models [52].

For Cross-Modality Cognition Abilities. Higher task diversity [86] and data-pool diversity [70], [87] improve the cross-modality cognition of MLLMs. Variations in data diversity affect MLLMs’ performance in processing information from different modalities [88]. The diversity of demonstration examples can also improve the effectiveness of incontext learning (ICL) for MLLM reasoning [89].

##### 3.2 For Scaling Effectiveness of MLLMs: Better Subsets

Simply increasing the amount of data is not economical [25]. It is indicated that only exponential data growth can lead to linear performance improvements of models [52]. In addition to the computational expense, low-quality data samples can jeopardize the performance of MLLMs. A welldesigned data filtering/selection strategy can lead to betterperforming models with less token count during training [14], [90], [91]. This section refers to the training cost and performance as the scale effectiveness of multi-modal data. It aims to select and orchestrate the training data to achieve higher token efficiency or better results, including: data condensation (§3.2.1), data mixture (§3.2.2), data packing (§3.2.3) and cross-modal alignment (§3.2.4).

first category mainly focuses on training efficiency and the latter two additionally emphasize MLLM performance.

Data Deduplication. As the scale of datasets continues to grow, the likelihood of having duplicate samples within the dataset increases. When text-to-image MLLMs are employed for dataset construction, training samples may be directly copied [94]. Duplicated data not only wastes valuable computational resources but also raises issues such as concerns on copyright infringement [94]. Traditional deduplication based on hash value can precisely remove the redundant samples, but may suffer the scaling of datasets [95]. To perform deduplication with query efficiency, the criteria for determining duplicates can be appropriately relaxed [96], [97]. Thus, embeddings by models can be used to identify and filter out data pairs that are semantically similar but not identical in data representation [94], [98].

Low-Quality Data Filtering. Data filtering, a.k.a data pruning, eliminates data samples that do not meet certain criteria. It contributes comparable or even better performance than training on the complete data pool while lowering training costs. The assessment of data quality can be achieved with heuristic metrics or evaluation models [99].

Heuristic-based Filtering is easy to implement and often effective, such as discarding samples with over-length captions [14] or low text complexity [100]. Several works focus on designing data filters with fixed training code and computational budget [14], [99], [101], [102]. To avoid bias by a single filter, an ensemble of filters could be employed [99]. Despite being effective, heuristic-based filtering eliminates certain categories of data, lowering data diversity [103].

Model-based Filtering eliminates the reliance on human labor of heuristic-based filtering. The evaluation model can be a pretrained foundation model such as CLIP for imagetext matching [14], [101], [104], or a finetuned MLLM to score the quality of instructions [105]. The neural network for data filtering has also been exploited [106]. Apart from discarding low-quality data samples, some works salvage the wrongly discarded data samples by rewriting the captions with well-trained models [101], [103], since some of the data pairs may be still valuable with appropriate rewriting.

Kernel Set Construction. Contrary to low-quality data filtering, kernel set construction starts from scratch and progressively adds data samples from a candidate pool to the selected training set. It is typically achieved by clustering the training data and selecting representative samples (e.g., those close to the centroid) from each cluster [16], [107]. Experiments indicate that fewer but higher-quality instruction data can enable MLLMs to generate better outputs.

- 3.2.1 Data Condensation Training MLLMs with a high-quality subset may lead to comparable or even superior performance while improving the token efficiency of training [90]. It is theoretically analyzed in active learning contexts that the training with a subset containing n samples could outperform training on the original dataset with N samples (N > n) [92], since the information may be transmitted from the selection strategy to MLLMs. Pruning redundant data can improve the scaling trend on vision datasets [93]. We classify the relevant methods into three categories: data deduplication, low-quality data filtering and kernel set construction, where the

3.2.2 Data Mixture

A dataset can be a mixture of data from multiple domains and even various sources. Different from the approaches focusing on data diversity (§3.1.3), those focusing on data mixture primarily enhance MLLM performance by adjusting the proportion of different categories of data within the given training dataset [60], which is similar to adjusting weights of tasks in multi-task learning.

Mitigating Distribution Bias. Generally, a balanced mixture across different domains leads to better pretraining performance [108]. The ratios of synthetic and original data also have a certain impact [109]. It is indicated that the

mixed pretraining dataset should include more samples from higher quality sources, rather than randomly sampling from all candidate sources [110]. Besides, the strategy of data mixing and the arrangements of model parameter updates also have a significant impact on pertaining [85].

Exploiting Distribution Bias. Although eliminating bias between datasets can help improve model performance, in some cases, biased data distribution can be leveraged beneficially [111]. During fine-tuning, adjusting the proportion of different categories of data in the dataset can meet varying requirements on image generation [66].

- 3.2.3 Data Packing Data packing improves MLLM performance through the optimal arrangement of data samples within each batch, without altering the dataset size. Compared to data mixture (§3.2.2) which combines different types of data at the dataset level, data packing combines them at the batch level.

For Better Pretraining. Including hard negatives in each batch can improve the pretraining of CLIP [112]. Packing visual patches from images of different resolutions into a single sequence allows for variable resolution while maintaining the aspect ratio [113]. It may be the solution of Sora to the variability in latent space dimensions [114].

For Long-Context Support. By strategically combining inputs of different lengths into a single batch, data packing enhances the long-context support of MLLMs. Existing works exploit this by formulating an optimization problem to minimize the disruption of contextual information via arranging the chunked long documents and short documents

- [115], or by arranging mutually relevant documents in a single training sequence as much as possible, long-context utilization can be improved with only brief fine-tuning
- [116]. From our investigations, data packing is promising but not fully investigated in the multi-modal domain.

3.2.4 Cross-Modal Alignment

MLLMs benefit from the alignment between different modalities of data. However, ensuring precise alignment is a labor-intensive task, potentially leading to mismatches

- [117]. This section summarizes existing works to detect and improve the inner-sample cross-modal alignment.

Joint Embedding Space. By mapping different modality data into the same feature space, we can assess their matching with the similarity of the embedding vectors [118]. These methods are typically related to multi-task learning and contrastive learning, such as treating image-text pairs with CLIP scores low as cross-modal mismatches, [14], [99], [101], [102], [103], [106] or frame-level mismatches between video and text [119]. CLIP score helps to assign different levels of alignment capabilities to data samples [120]. Techniques of learning from noised data can also be used to remove data samples with potentially incorrect correspondences [117].

Text-Centric Anchoring. The text can serve as an anchor to enable MLLMs to boost cross-modal alignment [109] with their knowledge. Some studies suggest that the mismatches between images and captions may not always stem from semantic dissimilarity, but rather from the low quality of the captions. Thus, some data samples initially deemed as mismatches can be reused by caption rewriting [103]. Similar approaches also facilitate the construction of BLIP-2

[76] and ALLaVA [37]. Chain-of-thought (CoT) techniques can also improve the text-centric alignment of charts [53].

##### 3.3 Brief Summarization and Discussion

A high-performing large-scale MLLM requires extensive datasets. MLLMs typically employ a modular training paradigm, where different components at different stages may consume different types of datasets (§3.1.1). After datasets are constructed, data augmentation may be needed

- (§3.1.2), which generally employs different strategies or different models for data in different modalities. In both acquisition and augmentation, enhancing the diversity generally benefits MLLMs (§3.1.3). However, optimizing diversity may bias the dataset towards simpler samples, while difficulty-based selection can miss necessary simple data [121], highlighting the trade-offs between data diversity and sample importance. Besides, predictive determining the mixture ratio of various data sources has been studied in LLMs [122]. However, the applicability and efficacy of such approaches for MLLMs remain uncharted territory.

After scaling up datasets, we need to improve the scaling effectiveness. To save computational resources and eliminate the low-quality data samples, we need to condense the dataset (§3.2.1) by deduplication, filtering, and kernel set construction, obtaining a true subset with high-density information. Then, there are two methods of adjusting the proportions of different data for better MLLMs: data mixture (§3.2.2) which works at the dataset level for better task performance and data packing (§3.2.3) which works at the batch level to improve convergence and support long contexts. While these methods should be orthogonal, their combined effects have not been extensively explored. A unique challenge for MLLMs is cross-modal alignment, which involves mapping data into a unified feature space or using text as an anchor. There is a lack of research on using other modalities as anchors, as discussed in Sec. 8.2.2.

4 MULTI-MODAL DATA CONTRIBUTIONS FOR MLLMS: USABILITY

After obtaining an MLLM with the works introduced in Sec. 3, enhancement of its usability is needed. We organize related works revolving around data into three progressive phases, followed by a retrospective, as shown in Fig. 4. An MLLM needs to go through the improvements on responsiveness to human instructions (§4.1), reasoning capabilities for better intelligence (§4.2) and compliance with ethics

- (§4.3). Finally, comprehensive evaluations are needed to review the effectiveness of previous methods and assist in model selection for application. After reviewing existing works, we briefly summarize and discuss them in Sec. 4.5.

##### 4.1 For Instruction Responsiveness of MLLMs

The instruction responsiveness of MLLMs can be improved through better prompt templates (§4.1.1), ICL demonstrations (§4.1.2), or fine-tuning data aligned with human behaviors (§4.1.3), where generally, the first two methods do not require changes to the model parameters.

Prompts for Better Responsiveness Prompts for Dataset Curation

Prompt Design (§4.1.1)

| | |
|---|---|
| | |

Demonstration Creation Demonstration Optimization

For Instruction Responsiveness of MLLMs (§4.1)

ICL Data (§4.1.2)

Human Preference-Oriented Improvement Hallucination Reduction

Human-Behavior Alignment Data (§4.1.3)

Data for Single-Hop Reasoning (§4.2.1) Data for Multi-Hop Reasoning (§4.2.2)

For Reasoning Ability of MLLMs (§4.2)

| | |
|---|---|
| | |

Multi-ModalDataContributions

Data-Centric Attacks Data-Centric Defenses

Data Toxicity (§4.3.1)

forMLLMs:Usability

For Ethics of MLLMs (§4.3)

Privacy Attacks Privacy-Preserving Training License Analysis Model Watermarking

Data Privacy and Intellectual Property (§4.3.2)

Visual Understanding Spatial Understanding Temporal Understanding

Benchmarks for Multi-Modal Understanding (§4.4.1)

Quality of Generation Human-Preference Alignment of Generation

Benchmarks for Multi-Modal Generation (§4.4.2)

For Evaluation of MLLMs (§4.4)

Benchmarks for Multi-Modal Safety of Generation

Retrieval (§4.4.3) Benchmarks for Multi-Modal Reasoning (§4.4.4)

- Fig. 4. Organization of data approaches tailored for the usability of MLLMs.

- 4.1.1 Prompt Design The prompt design optimizes the input context c of MLLMs in Definition 1, helping MLLMs to focus on specific tasks or objectives such as questions, instructions, or backgrounds.

Prompts for Better Responsiveness. With carefully designed templates, MLLMs can be better guided to respond to various tasks. The set-of-mark prompt indicates that simply overlaying identities on image regions can release the visual grounding capability [123] and improve the responses of GPT-4V by enabling better identification of the objects referred to by visual prompts [124]. With special prompts, MLLMs can detect whether their outputs contain harmful content [125]. Users can specify visual regions to MLLMs with specially-designed templates [126], [127]. By packing the transformed time series patches with specific prompts such as statistics, time series could be better predicted [128].

Prompts for Dataset Curation. Given specially designed prompts, MLLMs such as GPT-4V can generate responses including detailed evidence and reasoning, enhancing the response quality in curated instruction datasets [37]. As indicated by [129], the diversity of synthetic data is positively correlated with the diversity of prompts. Thus, it is important to provide a diverse set of prompt templates.

- 4.1.2 ICL Data MLLMs can learn from ICL demonstrations to improve their outputs [130]. Thus, the demonstrations play a crucial role

in enhancing the responsiveness of MLLMs.

Demonstration Creation. ICL demonstrations generally have less quantity compared to training data. Existing works involve manually constructing high-quality ICL demonstrations to directly boost MLLM performance [131] and teach well-trained MLLMs to generate datasets [60]. Some studies suggest that providing ICL demonstrations alone may not be sufficient for MLLM capabilities, and finetuning with ICL datasets might be beneficial [132], [133].

Demonstration Optimization. One study finds that nontext information in ICL demonstrations (e.g., images) typically consumes a large number of tokens which leads to unsatisfactory ICL performance, thus, images in demonstrations are aggregated to the latent space of textual labels to shorten input length [134]. For a similar purpose, some works aggregate the demonstrations into images to avoid inaccurate text descriptions of complex images [135]. One study finds that demonstrations from diverse groups (higher diversity) may lead to better performance [89].

4.1.3 Human-Behavior Alignment Data

Human-behavior alignment for MLLMs aims at making the behavior and decision-making of MLLMs more consistent with human behaviors, aligning the MLLM’s output with human preferences and reducing hallucinations [136].

Human Preference-Oriented Improvement. MLLM outputs can better fit human preference with the help of curated

human feedback datasets [137]. Due to the high cost of manually constructing such datasets, human efforts can often be replaced by well-trained MLLMs such as GPT-4V since these well-trained MLLMs are well-aligned with human preferences [70], [138]. With the feedback, human preferences can be distilled to MLLMs with reinforcement learning (RL) [70], [139] or direct preference optimization (DPO) [140], [141]. Preferences can also be presented in a multi-level form, making MLLMs better align with human preferences by learning from the differences between adjacent levels of preferences [142]. Note that merely pursuing high reward scores does not necessarily increase human preference for MLLMs, a.k.a. reward hacking, which can be calibrated with additional information [143].

Hallucination Reduction. Humans answer questions based on facts and background knowledge rather than fabricating information, while MLLMs may exhibit hallucinations [136], [144]. Hallucinations can be mitigated through fine-tuning with instruction datasets [37], [145] and DPO with feedback datasets [142] curated with GPT-4V. Some works promote the study of hallucinations from a benchmarking perspective, which we discuss in detail in Sec. 4.4.

##### 4.2 For Reasoning Ability of MLLMs

This section summarizes existing efforts on the reasoning abilities of MLLMs with data as an intermediary, organized according to the number of thinking steps required.

- 4.2.1 Data for Single-Hop Reasoning Single-hop reasoning requires a direct relationship between the question and the answer. Some works enhance this capability of MLLMs by curating corresponding datasets, emphasizing chart reasoning including mathematical reasoning and chart redrawing [68], and the understanding of humorous and creative content in videos [146].
- 4.2.2 Data for Multi-Hop Reasoning Multi-modal multi-hop reasoning needs multiple reasoning steps to answer a question with information from multiple modalities. Annotated reasoning steps in data help teach MLLMs to better demonstrate the decision-making process [147]. Providing relevant demonstrations can enhance the multi-modal CoT [89], [148], which benefits robotic reasoning [72] and scientific question answering [65]. Scaling up the corresponding data can significantly enhance the reasoning capabilities of MLLMs in both 2D and 3D spaces [149] as well as temporal localization in videos [150]. Augmenting language models with vision experts and compositional reasoning modules leads to better reasoning performance on complex cross-modal tasks [151], [152]. Additionally, neurosymbolic approaches help in improving reasoning performance on complex visual tasks without training [153].
- 4.3 For Ethics of MLLMs

Ethical considerations for MLLMs are crucial for their usability. It is essential to ensure the generated contents do not contain harmful content (§4.3.1), while guaranteeing that the training and inference do not violate privacy or infringe on intellectual property rights (§4.3.2). These issues can be addressed and resolved from data-centric aspects.

- 4.3.1 Data Toxicity

The integration of other modalities makes MLLMs more vulnerable to attackers [154]. When presented alongside input images with malicious content, MLLMs are more prone to generating sensitive or harmful responses than facing purely text-based inputs [125], [155]. To protect MLLMs from ethical violations and thereby improve their usability, it is necessary to research data toxicity faced by MLLMs.

Data-Centric Attacks. MLLMs can be poisoned with training data rather than directly manipulating their parameters. Some works construct adversarial examples to disrupt foundation models [156], [157], [158], [159], but may not always be effective in the context of MLLMs, since many MLLMs are built with frozen encoders [49]. Some works build malicious data to fine-tune MLLMs to generate harmful responses, with images [160] or audio as carriers [161]. Such jailbreak attacks can be performed with prompt engineering [162]. Adversarial prompts can even be transmitted from malicious agents to benign ones to induce harmful outcomes in a multi-agent system [163]. Image triggers can be planted for backdoor attacks [164].

Data-Centric Defenses. MLLMs can counteract harmful queries by simply adding a simple safety prompt before the queries [165]. Observing that MLLMs can retain the safeguard when images are removed from the inputs, prompting MLLMs to first detect and then remove or convert the malicious input images helps to defend against jailbreak attacks [125]. Additionally, training to distinguish the authenticity with multi-modal information helps MLLMs to avoid producing incorrect and confusing outputs [166].

- 4.3.2 Data Privacy and Intellectual Property

Privacy and intellectual property are crucial for MLLM usability. Dataset providers must address privacy issues during data collection, and MLLM training must filter and remove private information. This ensures compliance with laws and encourages data contributions. For intellectual property, model and data releases typically include licenses that protect innovations and economic interests, providing legal safeguards and market competitiveness. Analyzing these licenses helps avoid intellectual property disputes.

Privacy Attacks. MLLMs can memorize training data during training and may reveal portions of personal information under specific prompts similar to LLMs [167]. Thus, training data must exclude personal information while retaining public information, which is evaluated by a benchmark as a red team [168]. Some works exploit this memory information to extract private data, such as using the alignment information between street images and resident details to identify residents in geospatial systems [169].

Privacy-Preserving Training. MLLMs can be fine-tuned to decline requests for private information [170]. Image synthesis can promote privacy protection while performing data augmentation [171]. Differential privacy (DP) [172] can be viewed as a formal transformation of data, i.e., optimizing the data to retain as much utility as possible while removing private information. Federated learning (FL) [173] can harness edge-side data in a privacy-preserving manner to expand the data scale while protecting privacy, which has demonstrated effectiveness in training LLMs by combining

zero-order optimization [174], [175] and parameter-efficient fine-tuning (PEFT) [175], [176] to reduce memory and communication overhead. Although popular for LLMs, it has not yet been comprehensively explored for MLLMs.

License Analysis. Models and data often come with licenses restricting their use, which are a special type of data tied to the lifecycle of MLLMs and require comprehensive analysis, as current remedies are post hoc such as dataset retractions and modifications [177]. Traditional analysis for open-source software [178], [179] may be inadequate as MLLM projects are composed of several components including software, data and models. A tool named ModelGo is proposed to assess potential legal risks in machine learning projects [180], providing several case studies covering five modalities. This topic is important but still in its early stages.

Model Watermarking. By embedding benign backdoors in the training data, i.e., watermarks, model-service providers can determine whether the outputs are generated with their model, thereby strengthening user compliance with the licenses published by the model-service providers, such as clarifying the adoption of specific embedding models [181]. As more and more multi-modal data are created with MLLMs, the application of watermarking techniques for MLLMs needs further exploration.

##### 4.4 For Evaluation of MLLMs

The evaluation of MLLMs helps identify the improvements brought by specific data or approaches for further optimization, thereby facilitating the usability of MLLMs. This subsection summarizes existing works that curate datasets to benchmark MLLMs’ capabilities in multi-modal contexts, including: understanding (§4.4.1), generation (§4.4.2), retrieval (§4.4.3) and reasoning (§4.4.4).

- 4.4.1 Benchmarks for Multi-Modal Understanding With more comprehensive and accurate multi-modal understanding capabilities, MLLMs can better process and integrate information from different data modalities.

Visual Understanding. Visual understanding of MLLMs can be evaluated with general visual tasks such as image understanding [14] and visual perception [182], as well as specific downstream scenarios such as chart understanding [183], chart reasoning [71], aesthetic assessment [184], visual tasks in intelligent transportation [185] and identifying the attributes from the product descriptions [62].

Spatial Understanding. 3D understanding requires comprehension of spatial information, depth perception, and volume measurement over 2D understanding. Existing related evaluations cover point-cloud spatial understanding [186], region-level and scene-level tasks [187], as well as instruction-following abilities in 3D point clouds [137].

Temporal Understanding. Temporal understanding of MLLMs often refers to the ability to handle tasks difficult to solve within a single frame [188]. According to our survey, benchmarks for temporal understanding are usually accompanied by those for spatial understanding [188], [189].

- 4.4.2 Benchmarks for Multi-Modal Generation Generation tasks have an extraordinarily vast output space, making the evaluation quite challenging. Existing works assess the generation of MLLMs from various perspectives.

Quality of Generation. Some existing works on the evaluation of MLLM generation cover interleaved imagetext generation [190] and video generation [191], [192], [193]. Current evaluations mainly emphasize subject and background consistency in temporal quality, as well as aesthetic quality and motion smoothness [191], motion quality [192], and consistency with action sequence descriptions [193].

Human-Preference Alignment of Generation. Some works focus on evaluating the alignment of MLLMs to human preference [138], [191], [194]. Evaluations can be conducted in a unified standard with scoring and pair and batch ranking [194], or customized evaluation criteria can be applied to each evaluation sample [138]. Considering that MLLMs often experience hallucinations, which result in outputs that are inconsistent with human expectations, some works tailor evaluation for hallucinations [195], [196].

Safety of Generation. Although the LLMs in MLLMs are often equipped with abilities to avoid harmful outputs, MLLMs can be affected by jailbreak attacks [197]. Existing work constructs evaluation datasets encompassing different scenarios [165] and harmful behaviors [198] to comprehensively assess the impact of jailbreak attacks on MLLMs.

- 4.4.3 Benchmarks for Multi-Modal Retrieval The retrieval capabilities of MLLMs help break down barriers between different forms of information, facilitating better search and recommendation functions. Existing benchmarks evaluate such capabilities for audio retrieval [58] and in intelligent transportation scenarios which focus on crossmodality image retrieval [185], and vehicle retrieval, person re-identification and sketch-to-image [199].
- 4.4.4 Benchmarks for Multi-Modal Reasoning Some benchmarks evaluate reasoning abilities of MLLMs that do not require CoT, covering both images [200], [201], [202] and videos [73], [146] from a modality perspective, and include counter-intuitive and surprising information in videos [146], decision-making in autonomous driving [73], and operations related to charts from an application scenario perspective [200], [201]. There are also benchmarks focusing on the multi-modal CoT tasks for MLLMs [203].
- 4.5 Brief Summarization and Discussion

Usually, we need to construct and optimize data to further improve the usability of MLLMs trained with the works presented in Sec. 3 to enhance user satisfaction and protect the interests of involved parties, paralleled to evaluations to review the strengths and weaknesses of MLLMs.

MLLMs’ instruction responsiveness (§4.1) can be improved with better prompt templates (§4.1.1), high-quality ICL demonstrations (§4.1.2), and fine-tuning aligned with human behavior (§4.1.3). While the first two methods may not need to update model parameters, they could consume more tokens for inference. A study on ICL shows that text is the main information source in multi-modal contexts, while images have little impact, indicating to increase in the emphasis on other modalities in ICL demonstrations [204].

To improve the reasoning ability of MLLMs (§4.2), corresponding datasets could be constructed for single-hop (§4.2.1) and multi-hop reasoning (§4.2.2). However, most

current reasoning tasks for MLLMs focus on scenarios with intertwined visual and textual elements. There is a need to explore reasoning applications that deeply integrate information from other modalities to improve MLLMs.

To address ethical concerns (§4.3), it is important to prevent MLLMs from toxic outputs (§4.3.1), privacy leakage and license disputes (§4.3.2). Generally, non-text modalities suscept MLLMs to jailbreak attacks. However, research suggests visual cues can improve MLLMs’ ethical alignment [205], warranting further related explorations. While watermarking and privacy-preserving training are well-studied for LLMs, they have been not fully investigated for MLLMs.

To objectively compare MLLMs and reveal their taskspecific shortcomings, benchmarks are needed (§4.4) on understanding (§4.4.1), generation (§4.4.2), retrieval (§4.4.3) and reasoning (§4.4.4). As the covered tasks continue to expand, data-efficient evaluations [206] for MLLMs may be promising due to the conservation of computing resources.

#### 5 MODEL CONTRIBUTIONS FOR MULTI-MODAL DATA: SYNTHESIS

As the scale of datasets for MLLMs continues to grow, manually constructing datasets becomes a labor-intensive task. As a key participant in the data-model co-development, it is essential to identify the contributions made by models to multi-modal data. This section reviews model efforts to synthesize data in existing works, categorized by the roles the models play, and organized in the order of a common pipeline for building a dataset, including: data creator (§5.1), data mapper (§5.2), data filter (§5.3), and data evaluator (§5.4), as illustrated in Fig. 5. At the end of this section, a brief summarization and discussion are provided (§5.5).

Direct Generation Context-Driven Generation

Model as a Data Creator (§5.1)

| | |
|---|---|
| | |
| | |
| | |

Quality Enhancement Data Augmentation Modality Transformation Summarization Annotation

Multi-ModalData:Synthesis

ModelContributionsfor

Model as a Data Mapper (§5.2)

Based on Logits Based on Assessment

Model as a Data Filter (§5.3)

Quality Assessment Ethic Assessment

Model as a Data Evaluator (§5.4)

- Fig. 5. Overview of the model contributions to multi-modal data in terms of data synthesis, categorized by the roles of models.

##### 5.1 Model as a Data Creator

Models can act as a data creator to extend the cardinality of dataset D, as D = D ∪ {w(c) | D,c ∈ C}m, where m data samples are generated based on candidate contexts in C.

Direct Generation. Models can directly generate the data ultimately used for training. For example, MLLMs can generate text responses to the provided instructions based on the given images [207], charts [200], and 3D inputs [208]. Given existing videos, MLLMs can generate multiturn conversations [209]. Given requirements on topic and row count, charts can be automatically generated [183], [210] with rendering libraries. This method can even be used to generate data containing multi-modal responses [211]. Based on existing images and human-crafted instructions, answers [184], reasoning contexts in steps [61], [72] and results of counterfactual reasoning [202] can also be generated. MLLMs can be employed to provide assessment on data quality [105] and aesthetics [70], thereby building corresponding instruction datasets.

Context-Driven Generation. MLLMs with limited capabilities may output contexts as medians to prompt other models for final generation. MLLMs without vision generation capabilities have been utilized to generate captions [84], [212], [213] that then prompt Stable Diffusion for vision output to create fine-tuning data for compositional retrieval. A similar approach is applied for data that teach MLLMs to edit images following human instructions [214].

##### 5.2 Model as a Data Mapper

Models can enhance data quality by transforming data representations, formalized as: Given a set of mapping functions T = {T1(u,w),T2(u,w),...} based on model w such as rewriting and annotating, ∀T ∈ T is applied to each of the samples in D, i.e., D′ = {T(ui,w) | ui ∈ D,∀T ∈ T }. The transformed data samples may also be merged into D instead of replacement as D = D ∪ D′ for a larger quantity.

Quality Enhancement. MLLMs can rewrite data samples for higher quality or better application, such as rephrasing the labels in VQA datasets to suit compositional image retrieval [215] and assembling short titles into long descriptions [64]. COCO dataset [49] can be improved with more detailed captions for data samples generated by GPT-4.

Data Augmentation. MLLMs and LLMs can be used to rewrite data for augmentation and provide different data representations while preserving the original semantics. To increase the diversity of vocabulary and sentence structures [80], [81], captions in vision-language datasets can be rewritten to generate data with different sentence structures [15].

Modality Transformation. MLLMs can convert data from other modalities into texts, making them understandable for LLMs [209], [216] or enable a text-centric alignment between modalities [109]. These approaches may be promising for semantic communication, where the costs associated with data storage and transmission are greatly alleviated.

Summarization. MLLMs can perform summarization on existing data to facilitate data curation. Time-series data collected from the web can be automatically summarized [56], and the retrieved information in different modalities can be extracted and fused for the final text answers [217].

Annotation. Opposite to summarization, annotation fills in more details between the question and the answer with MLLMs’ world knowledge. For example, MLLMs can complete the reasoning processes between the questions and answers [147], supplement captions with more information

such as factual details [196], and refine human annotations on aesthetics [218]. Data from different modalities in documents can be mapped to enable knowledge linkage [219].

##### 5.3 Model as a Data Filter

Models can be used to filter out data samples in D according to certain criteria as Dfiltered = {ui | ui ∈ D,F(ui,w) = 1}, where F is a filter function returning 0 or 1, and Dfiltered⊆ D.

Based on Logits. Some indicators calculated with the model-generated logits can potentially reflect the quality. For example, CLIP score can reflect the image-text semantic relevance [14], [104], [195], where data samples with low scores can be removed. In sampling-based data selection methods, the logits of the surrogate model can determine the sampling probability of each data instance [92].

Based on Assessment. Indicators calculated from logits require manual thresholds and may not align with human behavior. The assessment provided by MLLMs on data samples can serve as a basis for data filtering. Well-trained MLLMs can be employed to generate numerical scores for image-text datasets [105] to quantify the matching between images and texts, and answer questions in datasets while removing the and wrongly-answered data [202]. MLLMs can also evaluate their self-generated dataset [61].

##### 5.4 Model as a Data Evaluator

Models as data evaluators facilitate data synthesis via feedback, not manipulation. This section summarizes existing works that employ models for sample-level data evaluation.

Quality Assessment. As GPT-4V aligns with humans in terms of evaluation criteria [138], it can provide feedback on data samples in 1) quantitative evaluation, 2) preference ranking, and 3) quality enhancement suggestions [70]. As an example of data-model co-development, MLLMs can provide self-evaluation on the generated contents [61]. However, while MLLMs align with human capabilities in pairranking, their quantitative scores need improvement [194].

Ethic Assessment. Models aligned with human values can uphold ethical standards, helping to check web-crawled and model-generated data for violations. With carefully designed prompt templates, MLLMs can self-check their outputs for harmful content [125]. The success of malicious queries can be assessed with the help of ChatGPT [160].

##### 5.5 Brief Summarization and Discussion

With advanced model capabilities and rising data needs for training MLLMs, models increasingly handle data curation instead of humans. A model can serve as a data creator (§5.1), a mapper to transform existing data samples (§5.2), a filter to exclude low-quality samples (§5.3), and an evaluator to evaluate or rank data samples (§5.4). From our investigation, current works mainly rely on well-trained MLLMs such as GPT-4V. This paradigm may be insufficient to construct MLLMs with top-tier scale and performance. Thus, it is promising to leverage the increasing knowledge and capabilities of the target MLLM while training it.

Navigating for Humans Navigating for Models

Model as a Data Navigator (§6.1)

| | |
|---|---|
| | |
| | |
| | |

Multi-ModalData:Insights

ModelContributionsfor

Key Information Extraction Relationship Extraction

Model as a Data Extractor (§6.2)

Insight Derivation Automated Statistics

Model as a Data Analyzer (§6.3)

Visualization with Rendering End-to-End Visualization

Model as a Data Visualizer (§6.4)

Fig. 6. Overview of the model contributions to multi-modal data in terms of data insights, categorized by the roles of models.

#### 6 MODEL CONTRIBUTIONS FOR MULTI-MODAL DATA: INSIGHTS

In addition to data synthesis, models can provide insights to assist in dataset curation and the development of data pipelines, serving as data scientists. This section summarizes existing applications of models for data science, grouped by model roles and ordered by the sequence of tasks performed by a data scientist for dataset curation, including navigator to assist users in locating the required information from a vast search space (§6.1), extractor to extract key content from vast and disorganized information (§6.2), analyzer to provide data analysis and statistics (§6.3), and visualizer to facilitate data visualization (§6.4), as illustrated in Fig. 6. After the introduction of existing works, a brief summarization and discussion are provided in Sec. 6.5.

##### 6.1 Model as a Data Navigator

The capability of LLMs to retrieve external knowledge has been well-recognized [220], [221], [222]. MLLMs can help data consumers quickly locate the needed data or information, with the understanding and retrieval capabilities.

Navigating for Humans. MLLMs can eliminate the need to separate search for data in different modalities for knowledge retrieval based on multi-modal queries, thereby allowing better utilization of the intertwined contextual information across modalities [223]. MLLMs also facilitate knowledge discovery [224], which guide humans to knowledge yet undiscovered by humanity.

Navigating for Models. By extending data consumers from humans to models, external knowledge can be navigated for more precise responses [223], such as ICL demonstrations [225], external knowledge bases [226], and web video descriptions [227]. MLLM outputs also benefit from the entity relationship in multi-modal knowledge graph [228], [229]. As Definition 1, these methods adopt MLLMs to provide contexts with external information and knowledge.

##### 6.2 Model as a Data Extractor

MLLMs can extract information from unstructured raw data to facilitate further analysis, which is especially important for extensive data. It can be regarded as a more fine-grained

analysis, i.e., the data-sample level, compared to those using MLLMs for analysis at the dataset level (§6.3).

Key Information Extraction. Existing works extract key information from documents [230], [231], and reorganize the text outputs generated by MLLMs into segments and related claims [196]. Named entity recognition for knowledge graph construction can be regarded as a special type of work for key information extraction. With MLLMs, entities can be automatically extracted from texts [232], images [52], or even grounded multi-modal named entity recognition that further bounds the groundings of entities in images [233].

Relationship Extraction. Knowledge graph construction requires understanding syntax and semantics and, thus, typically involves labor-intensive efforts [234]. MLLMs can help the automatic construction of knowledge graphs by extracting the relationship between multi-modal entities [235]. This capability can also help to bind different modalities of data within documents, enabling knowledge association [219].

##### 6.3 Model as a Data Analyzer

Data analysis is one of the responsibilities of data scientists and typically includes identifying data quality issues, understanding data distribution, etc. By jointly leveraging the ability of MLLMs and LLMs to understand human requirements and its capability to summarize data, the workload of data scientists can be significantly reduced [236].

Insight Derivation. MLLMs and LLMs have been validated to be capable of data analysis and insights [77], [237], [238], [239]. MLLMs with visual understanding of mathematical formulas and tables can help capture critically important information in scientific diagrams [240] or scientific literature [241], [242], which can even benefit the preliminary analysis of academic paper quality.

Automated Statistics. With understanding and reasoning abilities, MLLMs can perform automated statistics for charts [68], [183], [200], [201], [210], where the charts for datasets can be generated by third-party software. MLLMs also help derive statistical conclusions from tables or databases [243].

##### 6.4 Model as a Data Visualizer

Data visualization is crucial in data science workflows for data comprehension and insights. MLLMs can provide personalized visualizations and handle unstructured data with understanding, reasoning, and generation capabilities.

Visualization with Rendering. Given visual and text constraints [244], MLLMs can generate layouts to alleviate the manual efforts data scientists expend on visualization styles. Visualization of raw data and analysis results can be achieved by invoking APIs with LLMs serving as a code copilot [245], [246], [247]. This kind of method has been applied in some database management works [236], [248].

End-to-End Visualization. Leveraging the image generation and editing abilities of MLLMs [249], visualizations can be performed in an end-to-end manner according to user requirements, and even be adjusted interactively. This helps visualization be better aligned with human needs and reduces human effort. Given an existing chart as a template, MLLMs can be employed to generate new charts following human instructions [59], [210]. Although promising, this approach may require more exploration to ensure that it can produce results that precisely meet user requirements.

##### 6.5 Brief Summarization and Discussion

With the emerging capabilities of MLLMs, they can help data scientists to navigate data consumers to desired data and knowledge (§6.1), extract information and relationships from data (§6.2), provide data insights and statistics (§6.3), and reduce the manual involvements in data visualization (§6.4). These topics are beginning to take shape, but it will be a while before we can fully rely on MLLMs as data scientists.

#### 7 PUBLIC DATASETS FOR MLLMS

Table 2 lists public datasets for MLLMs based on this survey, which is continuously maintained to stay up-to-date. The operator “-” denotes an order-sensitive mapping relationship. “Pretrain” and “Pretrain(P)” denote the pretraining of encoders/decoders and projectors, respectively.

#### 8 ROADMAP FOR FUTURE MLLMS

From this survey, existing MLLM works of data-model co-development are mainly performed in the externallyboosted paradigm as Definition 1. We provide some promising future directions progressively organized as follows: 1) infrastructure development (§8.1), 2) MLLM research with data-model co-development in an externally-boosted manner, where we try to point out the next phase of this widely applied paradigm (§8.2), and 3) MLLM research with data-model co-development in a self-boosted manner, a less explored but promising paradigm in the foreseeable future (§8.3). These progressive directions form a roadmap for MLLM research in the coming few years.

##### 8.1 Data-Model Co-Development Infrastructures

To advance data-model co-development for MLLMs, more advancements are required in the infrastructure for a more conducive environment. Firstly, scalable data management suites for MLLMs should be implemented to support efficient curation of large-scale multi-modal datasets. Then, efficient hardware and algorithms, e.g., quantization and mixed precision algorithms, can enhance the iteration between MLLM training and synthesis of multi-modal datasets. Finally, there is a pressing need for more seamless integration between existing data-centric and model-centric infrastructures. These tools allow developers to effortlessly design and test prototypes for diverse datasets, models and application scenarios. An initial attempt [260] in this direction builds a sandbox with several built-in experimental scaffolds for data-model co-development of MLLMs.

##### 8.2 Externally-Boosted MLLM Development

Based on convenient infrastructures, we can explore some advanced research topics under the externally-boosted paradigm of data-model co-development by leveraging well-trained MLLMs, including data discovery, modalitycompatibility detection and automatic knowledge transfer, organized in order of first scaling up the data and then improving MLLMs’ usability as that in Sec. 3 and 4.

TABLE 2: Public datasets for MLLMs based on this survey. The top and bottom sections list datasets with and without models as one of the data sources, respectively. “Quantity” indicates the count of data samples/instances. “Merge” denotes adopting/merging some existing datasets.

Modality Data Sources MLLM Stage Quantity Objective Reference

Video-Text MLLMs, Merge Pretrain 70M Video-language pretraining Panda-70M [250] Video-Text BLIP2, Web Pretrain 234M Video-related understanding and generation InternVid [251] Image-Text GPT-3.5 Pretrain 9K Chart understand & reasoning SimChart9K [68] Image-Text GPT-4V, Web, Merge Pretrain, Pretrain(P),

1.2M Pretraining encoder and projector finetuning MLLM with high-quality image-text data

ShareGPT4V [54]

Finetune

Image-Text GPT-4, Merge Pretrain, Finetune 8M Enhancing coverage of chart themes Chart-Sum-QA [53] Image-Text GPT4-V, Merge, Web Pretrain(P), Finetune 1.2M Image understanding ShareGPT4V [54] Image-Text GPT4-V, Merge Pretrain(P), Finetune 664K Modality alignment & instruction following ALLaVA (Data) [37] Image-Text GPT-4, Merge Finetune 400K Mitigating hallucinations LRV-Instruction [145] Video-Text GPT-4V, Merge Finetune 40K Text-video generation ShareGPT4Video [252] Image-Text Genimi-Pro, Merge Finetune 195K Fine-grained image perception DocGemini [63] Image-Text GPT-4V, Merge Finetune 1.6M visual comprehension with textual interaction MDVP-Data [124] Image-Text Gemini-Pro, Web Finetune 9.1M Text-centric VQA Square-10M [61] Image-Text GPT-4 Finetune 75K 3D scene understanding 3DMIT [208] Video-Text GPT-4, Merge Finetune 7K Video understanding VideoChat [209] Image-Text GPT-4 Finetune 160K Chart understanding ChartLlama [210] Image-Text GPT-4, Merge Finetune 32K Complex visual reasoning ComVint [207] Image-Text ChatGPT, Human, Merge Finetune 2.4M Human instruction alignment M3IT [253] Image-Text GPT-4, Merge, Human Finetune 600K Chart understanding MMC-Instruction [71] Video-Text BLIP-2, GRiT, Merge Finetune 100K Video understanding and conversation VideoInstruct-100K [67] Image-Text GPT4-V, Merge Finetune 80K Set-of-Mark understanding SoM-LLaVA [60] Text-Image GPT-4V, Merge Finetune 120M Generation quality on human preference VisionPrefer [70] Video-Text ChatGPT, Merge Finetune 2M Spatial & temporal understanding VideoChat2-IT [188] Image-Text GPT-API, Merge Finetune 196K 3D instruction following LAMM [137] Image-Text ChatGPT, Merge, Web Finetune 39M Chart understanding & downstream tasks ChartSFT [254] Image-Text GPT-3.5, Web Finetune, Eval 107K Misinformation detection in medical fields Med-MMHL [166] Image-Text GPT, Simulation Finetune, Eval 231K Spatial understanding on point clouds 3DBench [186] Image-Text GPT, Merge, Human Finetune, Eval 11.3K Multi-modal CoT M3CoT [203]

Video/Image-Text GPT-4, Merge, Human Eval 19K Spatial & temporal understanding SEED-Bench [189]

Text-Video GPT-4, Merge, Human Eval 200K Generation quality on human preference EvalCrafter [192] Image-Text GPT-4, Human Eval 6.9K Image aesthetic assessment ImplicitAVE [62] Image-Text MLLMs, Merge, Human Eval 4.4K Evaluations on human-preference alignment MLLM-as-a-Judge [194]

Image-Text/Image ChatGPT, Merge Eval 31K Comprehensive evaluation of MLLMs MMT-Bench [199] Image-Text Stable Diffusion, GPT-4 Eval 5K Generation safety MM-SafetyBench [165] Image-Text MLLMs, Merge, Human Eval 1.2K Visual hallucinations VHTest [195] Image-Text ChatGPT, Merge, Human Eval 1.9K Visual hallucinations MHaluBench [196] Image-Text GPT-4, Human Eval 48K Reasoning for charts ChartX [200] Image-Text GPT-3.5, Human Eval 6K Chart understanding ChartY [183]

Image-Text Web, Human Pretrain 10M Chinese video-language pretraining Youku-mPLUG [255] Audio-Text Web Pretrain 4.5K Text-audio retrieval WavText5K [58] Video-Text Web, Human Pretrain 13K Video understanding BLiSS [256] Video-Text Merge, Web Pretrain 101.2M Image-Text interleaved pretraining MMC4 [257] Image-Text Merge Pretrain 12.8M-12.8B Image-language pretraining DataComp [14] Audio-Text Web Pretrain 630K Pretraining of audio & text encoders LAION-Audio-630K [57] Image-Text Web Finetune 21K CoT generation ScienceQA [65]

Audio/Image-Text Merge Finetune 5K Triple-modality instruction following BuboGPT [64] Image-Text Web, Human Eval 130K Performance with long-tailed concepts Let it Wag [52] Text-Video Merge, Human Eval 800 Generation quality on human preference VBench [191] Video-Text Human Eval 11.6k Temporal understanding, perception Perception Test [258] Video-Text Web, Human Eval 312K Reasoning, surprising video understanding FunQA [146] Text-Video Human Eval 11M Generation quality, state transfer WorldNet-Wild [193] Image-Text Merge, Human Eval 32.7K Evaluation for logical operations with charts ChartQA [201] Image-Text Merge, Human Eval 3.8K Evaluation for visual understanding BLINK [182] Image-Text Merge, Human Eval 15K Evaluation for mathematical reasoning MathVerse [259] Image-Text Human Eval 420 Evaluation for open-ended QA MLLM-Bench [138]

- 8.2.1 MLLM-Based Data Discovery

Data discovery fundamentally supports scaling and usability of MLLMs as in Sec. 3.1 and 3.2. It needs understanding the requirements, retrieving relevant data, and statistics to gain an initial outline of data characteristics, often requiring heavy human efforts. As in Sec. 6, existing works reveal the capabilities of MLLMs in terms of knowledge discovery [224] and dataset analysis [243]. With these capabilities, it

is promising to enable automatic data discovery to alleviate human labor and keep the knowledge contained in MLLMs up to date by continually acquiring valuable multi-modal data from the internet, such as images and videos from news. This topic leads to some minor research issues.

Automatic Long-tail Data Discovery. MLLMs may perform sub-optimal with inputs belonging to long-tail domains [52], potentially misclassifying some long-tail as

worthless, wasting data that could be even more meaningful. Thus, it is needed to especially enhance MLLMs in discerning whether data belongs to long-tail domains.

Automatic Compliance with Privacy and Licensing Requirements. Data discovery must comply with privacy and usage regulations [180] Enabling MLLMs to understand licenses and avoid privacy violations in an automatic data discovery pipeline, is a worthwhile research direction.

- 8.2.2 Modality-Compatibility Detection with MLLMs From Sec. 3.2.4, prevalent model-driven techniques for assessing the compatibility across modalities mainly leverage pre-trained foundation models, which occasionally yield inaccurate assessments, as a low compatibility score could stem not from modal mismatches but rather from suboptimal data quality [101]. Text-centric anchoring helps with this by refining captions [103]. Our study reveals that aligning across modalities by MLLMs with non-text anchors remains an underexplored frontier. Expanding anchoring to encompass additional modalities may harness a richer array of signals and broaden the representational landscape, potentially elevating the efficacy of alignment endeavors.
- 8.2.3 Automatic Knowledge Transfer for MLLMs Training MLLMs on datasets generated with top-tier models essentially distills knowledge from them. It still requires human involvement, such as specifying a transfer set. As well-trained MLLMs can provide suggestions to quality enhancement [70], it is promising to explore automatic knowledge transfer, especially transferring between MLLMs that excel in different modalities. Specifically, a teacher MLLM can question a student MLLM to assess its shortcomings, and generate tailored data to bridge these gaps. The key lies in creating an automated pipeline to use well-trained MLLMs for evaluating target models.

##### 8.3 Self-Boosted MLLM Development

The self-boosted paradigm of data-model co-development (§2) does not rely on the availability of well-trained MLLMs, making it more versatile and capable of supporting the development of MLLMs with top-tier scale and performance. Potential points for consideration cover the scaling and usability of MLLMs as discussed in Sec. 3.1, 3.2, 4.

8.3.1 Self Data Scaling with MLLMs

Data Scaling with Single MLLM. Existing works show that alternatively using models to assist in data curation and improving models with the newly collected data can achieve data-model co-development [35]. For any-to-any MLLMs, it is promising to alternatively optimize data in different modalities, such as image recaptioning [37], [76], [101], while optimizing them with the continually optimized/growing data, which alleviates copyright concerns.

Data Scaling with Cooperative MLLMs. Inspired by deep mutual learning [261] which trains multiple models that learn from each other and results in a better performance than training a single model, data scaling may be enabled with multiple cooperative MLLMs, i.e., training multiple MLLMs simultaneously and having them jointly improve the datasets with their different knowledge. A

recent study [260] finds that alternative recaptioning with an image-text model and image regeneration with a text-image model can continuously boost the data quality and quantity [260], which provides a prototype for the exploration cooperative MLLMs that excel in different modalities.

- 8.3.2 Self Data Condensation with MLLMs From Sec. 3.2.1, there are some model-based metrics and assessments [14], [101], [103], [104], [105], [106] that help to condensate multi-modal data without reliance of human heuristics. They essentially use the criteria of one model to assess the value of the data for another model, failing to be tailored for the specific needs of target models.

If we consider MLLMs as intelligent entities, they should best understand what type of data they need. Thus, allowing the MLLM to select the necessary training data for itself is promising, where the evaluation can be performed based on quality assessment [61], [70], quantitative score [105], or ranking among data samples [194]. With this, the condensation can be dynamically adjusted based on the MLLM’s current state during training, as some data, mistakenly perceived as low quality or modality mismatch due to difficulties in model comprehension, can be reused for training after the capabilities of the MLLM are improved.

- 8.3.3 RL from Self Feedback of MLLMs Reinforcement learning (RL) from human feedback (RLHF) aligns LLMs and MLLMs with human preferences based on human feedback [262]. However, reliance on human feedback limits scalability, prompting the rise of RL with artificial intelligence feedback (RLAIF) [263]. Curating human feedback for MLLMs may be more labor-intensive than LLMs due to multiple modalities. Well-trained MLLMs can provide ranking assessment aligned with human feedback [194], enabling AI feedback for multi-modal contents [264]. Just like humans can evaluate the quality of their own statements, allowing an MLLM under training to provide feedback on its own output may be a promising direction. In the early training stage of an MLLM, some human involvement may still be needed. Besides, fine-grained feedback rather than scores could be better for MLLMs [70], [265].

#### 9 CONCLUSIONS

This paper highlights the dual effect and great potential of simultaneously improving both data and models for MLLM development, sketching a new data-model co-development paradigm through a comprehensive and systematic review of existing works for MLLMs. Specifically, we first examine the contributions that data can make to MLLMs by first scaling up MLLMs and then improving the usability of MLLMs. Next, we discuss how models can facilitate data curation by serving for data synthesis as data developers and providing data insights as data scientists. We identify that the current data-model co-development paradigm in MLLMs often still requires well-trained models or human assistance. In light of this, we summarize numerous potential future directions ranging from infrastructure development to selfboosted data-model co-development for MLLMs, furnishing a roadmap for future MLLMs from the data-model codevelopment perspective. Through these efforts, we hope to

offer timely guidance and inspire more innovation in both research and applications of MLLMs, fostering a deeper integration of both data and model advancements.

#### REFERENCES

- [1] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv:2403.05530, 2024. 1
- [2] OpenAI, “Sora: Creating video from text,” https://openai.com/s ora, 2024. 1
- [3] OpenAI, “Hello gpt-4o,” https://openai.com/index/hello-gpt-4 o/, 2024. 1
- [4] S. Wu, H. Fei, L. Qu, W. Ji, and T.-S. Chua, “Next-gpt: Any-to-any multimodal llm,” arXiv:2309.05519, 2023. 1, 2.4, 3.1.1
- [5] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., “A survey of large language models,” arXiv:2303.18223, 2023. 1
- [6] P. Villalobos, J. Sevilla, L. Heim, T. Besiroglu, M. Hobbhahn, and A. Ho, “Will we run out of data? an analysis of the limits of scaling datasets in machine learning,” arXiv:2211.04325, 2022. 1
- [7] S. Goyal, P. Maini, Z. C. Lipton, A. Raghunathan, and J. Z. Kolter, “Scaling laws for data filtering–data curation cannot be compute agnostic,” in CVPR, 2024, pp. 22702–22711. 1
- [8] P. Gao, L. Zhuo, Z. Lin, C. Liu, J. Chen, R. Du, E. Xie, X. Luo, L. Qiu, Y. Zhang et al., “Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers,” arXiv:2405.05945, 2024. 1, 2.4
- [9] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv:2001.08361, 2020. 1, 3.1
- [10] A. Aghajanyan, L. Yu, A. Conneau, W.-N. Hsu, K. Hambardzumyan, S. Zhang, S. Roller, N. Goyal, O. Levy, and L. Zettlemoyer, “Scaling laws for generative mixed-modal language models,” in ICML, 2023, pp. 265–279. 1
- [11] D. Zha, Z. P. Bhat, K.-H. Lai, F. Yang, Z. Jiang, S. Zhong, and X. Hu, “Data-centric artificial intelligence: A survey,” arXiv:2303.10158, 2023. 1, 2.1, 1, 2.3
- [12] S. Salehi and A. Schmeink, “Data-centric green artificial intelligence: A survey,” IEEE Transactions on Artificial Intelligence, pp. 1–18, 2023. 1, 2.1, 1, 2.3
- [13] J. Jakubik, M. V¨ossing, N. Kuhl,¨ J. Walk, and G. Satzger, “Datacentric artificial intelligence,” Business & Information Systems Engineering, pp. 1–9, 2024. 1, 2.1
- [14] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang et al., “Datacomp: In search of the next generation of multimodal datasets,” NeurIPS, vol. 36, 2023. 1, 3.1.1, 3.1.3, 3.2, 3.2.1, 3.2.4, 4.4.1, 5.3, 2, 8.3.2
- [15] L. Fan, D. Krishnan, P. Isola, D. Katabi, and Y. Tian, “Improving clip training with language rewrites,” NeurIPS, vol. 36, 2023. 1, 3.1.2, 5.2
- [16] M. He, Y. Liu, B. Wu, J. Yuan, Y. Wang, T. Huang, and B. Zhao, “Efficient multimodal learning from data-centric perspective,” arXiv:2402.11530, 2024. 1, 2.1, 3.2.1
- [17] K. Carolan, L. Fennelly, and A. F. Smeaton, “A review of multimodal large language and vision models,” arXiv:2404.01322,

2024. 1, 2.3, 1

- [18] D. Caffagni, F. Cocchi, L. Barsellotti, N. Moratelli, S. Sarto, L. Baraldi, M. Cornia, and R. Cucchiara, “The (r)evolution of multimodal large language models: A survey,” arXiv:2402.12451,

2024. 1, 2.3, 1

- [19] M. Xu, W. Yin, D. Cai, R. Yi, D. Xu, Q. Wang, B. Wu, Y. Zhao, C. Yang, S. Wang et al., “A survey of resource-efficient llm and multimodal foundation models,” arXiv:2401.08092, 2024. 1, 2.3, 1
- [20] C. Li, Z. Gan, Z. Yang, J. Yang, L. Li, L. Wang, J. Gao et al., “Multimodal foundation models: From specialists to generalpurpose assistants,” Foundations and Trends® in Computer Graphics and Vision, vol. 16, no. 1-2, pp. 1–214, 2024. 1, 2.3, 1
- [21] D. Zhang, Y. Yu, C. Li, J. Dong, D. Su, C. Chu, and D. Yu, “MMLLMs: Recent advances in multimodal large language models,” arXiv:2401.13601, 2024. 1, 2.3, 1
- [22] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and

Machine Intelligence, pp. 1–20, 2024. 1, 2.3, 1

- [23] Q. Liu, J. Zhu, Y. Yang, Q. Dai, Z. Du, X.-M. Wu, Z. Zhao, R. Zhang, and Z. Dong, “Multimodal pretraining, adaptation, and generation for recommendation: A survey,” arXiv:2404.00621, 2024. 1, 2.3, 1
- [24] Y. Tang, J. Bi, S. Xu, L. Song, S. Liang, T. Wang, D. Zhang, J. An, J. Lin, R. Zhu et al., “Video understanding with large language models: A survey,” arXiv:2312.17432, 2023. 1, 2.3, 1
- [25] Y. Jin, J. Li, Y. Liu, T. Gu, K. Wu, Z. Jiang, M. He, B. Zhao, X. Tan, Z. Gan et al., “Efficient multimodal large language models: A survey,” arXiv:2405.10739, 2024. 1, 1, 3.2
- [26] J. Huang, J. Zhang, K. Jiang, H. Qiu, and S. Lu, “Visual instruction tuning towards general-purpose multimodal model: A survey,” arXiv:2312.16602, 2023. 1, 2.3, 1
- [27] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” arXiv:2306.13549,

2023. 1, 2.3, 1, 2.4, 3.1

- [28] J. Wu, W. Gan, Z. Chen, S. Wan, and S. Y. Philip, “Multimodal large language models: A survey,” in IEEE BigData, 2023, pp. 2247–2256. 1, 2.3, 1
- [29] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 12113–12132, 2023. 1, 2.3, 1
- [30] F. Zhao, C. Zhang, and B. Geng, “Deep multimodal data fusion,” ACM Computing Surveys, vol. 56, no. 9, pp. 1–36, 2024. 1, 2.3, 1
- [31] Y. Wang, W. Chen, X. Han, X. Lin, H. Zhao, Y. Liu, B. Zhai, J. Yuan, Q. You, and H. Yang, “Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning,” arXiv:2401.06805,

2024. 1, 1

- [32] P. Zhou, L. Wang, Z. Liu, Y. Hao, P. Hui, S. Tarkoma, and J. Kangasharju, “A survey on generative ai and llm for video generation, understanding, and streaming,” arXiv:2404.16038, 2024. 1, 1, 2.3
- [33] T. Zhao, L. Zhang, Y. Ma, and L. Cheng, “A survey on safe multimodal learning system,” arXiv:2402.05355, 2024. 1, 2.3
- [34] T. Bai, H. Liang, B. Wan, L. Yang, B. Li, Y. Wang, B. Cui, C. He, B. Yuan, and W. Zhang, “A survey of multimodal large language model from a data-centric perspective,” arXiv:2405.16640, 2024. 1, 2.1, 1, 2.3
- [35] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023, pp. 4015–4026. 1, 2.1, 8.3.1
- [36] H. Xu, S. Xie, P.-Y. Huang, L. Yu, R. Howes, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer, “Cit: Curation in training for effective vision-language data,” in ICCV, 2023, pp. 15180–15189. 2.1, 2.1
- [37] G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li, X. Wan, and B. Wang, “Allava: Harnessing gpt4v-synthesized data for a lite vision-language model,” arXiv:2402.11684, 2024. 2.1, 2.4, 3.1.1, 3.2.4, 4.1.1, 4.1.3, 2, 8.3.1
- [38] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10684–10695. 2.1, 2.4
- [39] X. Li, P. Yu, C. Zhou, T. Schick, L. Zettlemoyer, O. Levy, J. Weston, and M. Lewis, “Self-alignment with instruction backtranslation,” arXiv:2308.06259, 2023. 2.1
- [40] D. Zha, K.-H. Lai, F. Yang, N. Zou, H. Gao, and X. Hu, “Datacentric ai: Techniques and future perspectives,” in KDD, 2023, pp. 5839–5840. 2.1
- [41] Z. Wang, W. Zhong, Y. Wang, Q. Zhu, F. Mi, B. Wang, L. Shang, X. Jiang, and Q. Liu, “Data management for large language models: A survey,” arXiv:2312.01700, 2023. 1, 2.3
- [42] L. Long, R. Wang, R. Xiao, J. Zhao, X. Ding, G. Chen, and H. Wang, “On LLMs-driven synthetic data generation, curation, and evaluation: A survey,” arXiv:2406.15126, 2024. 1, 2.3
- [43] A. Albalak, Y. Elazar, S. M. Xie, S. Longpre, N. Lambert, X. Wang, N. Muennighoff, B. Hou, L. Pan, H. Jeong et al., “A survey on data selection for language models,” arXiv:2402.16827, 2024. 1, 2.3
- [44] J. Wang, B. Zhang, Q. Du, J. Zhang, and D. Chu, “A survey on data selection for llm instruction tuning,” arXiv:2402.05123, 2024. 1, 2.3
- [45] B. Ding, C. Qin, R. Zhao, T. Luo, X. Li, G. Chen, W. Xia, J. Hu, A. T. Luu, and S. Joty, “Data augmentation using llms: Data perspectives, learning paradigms and challenges,” arXiv:2403.02990,

2024. 1, 2.3

- [46] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, “LLaMA: Open and efficient foundation language models,” arXiv:2302.13971, 2023. 2.4, 3.1.2
- [47] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021. 2.4
- [48] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763. 2.4, 3.1.1
- [49] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” NeurIPS, vol. 36, 2023. 2.4, 3.1.1, 1, 4.3.1, 5.2
- [50] P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou, W. Zhang, P. Lu, C. He, X. Yue et al., “Llama-adapter v2: Parameter-efficient visual instruction model,” arXiv:2304.15010, 2023. 2.4, 3.1.1
- [51] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark et al., “Training compute-optimal large language models,” arXiv:2203.15556, 2022. 3.1
- [52] V. Udandarao, A. Prabhu, A. Ghosh, Y. Sharma, P. H. Torr, A. Bibi, S. Albanie, and M. Bethge, “No” zero-shot” without exponential data: Pretraining concept frequency determines multimodal model performance,” arXiv:2404.04125, 2024. 3.1.1, 3.1.3, 3.2, 6.2, 2, 8.2.1
- [53] M. Liu, D. Chen, Y. Li, G. Fang, and Y. Shen, “Chartthinker: A contextual chain-of-thought approach to optimized chart summarization,” arXiv:2403.11236, 2024. 3.1.1, 3.2.4, 2
- [54] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin, “Sharegpt4v: Improving large multi-modal models with better captions,” arXiv:2311.12793, 2023. 3.1.1, 2
- [55] J. Wu, Y.-H. Peng, A. Li, A. Swearngin, J. P. Bigham, and J. Nichols, “Uiclip: A data-driven model for assessing user interface design,” arXiv:2404.12500, 2024. 3.1.1
- [56] F. Jia, K. Wang, Y. Zheng, D. Cao, and Y. Liu, “GPT4MTS: Prompt-based large language model for multimodal time-series forecasting,” in AAAI, vol. 38, no. 21, 2024, pp. 23343–23351. 3.1.1, 5.2
- [57] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in ICASSP, 2023, pp. 1–5. 3.1.1, 2
- [58] S. Deshmukh, B. Elizalde, and H. Wang, “Audio retrieval with WavText5k and clap training,” arXiv:2209.14275, 2022. 3.1.1, 3.1.3, 4.4.3, 2
- [59] P. Yan, M. Bhosale, J. Lal, B. Adhikari, and D. Doermann, “Chartreformer: Natural language-driven chart image editing,” arXiv:2403.00209, 2024. 3.1.1, 6.4
- [60] A. Yan, Z. Yang, J. Wu, W. Zhu, J. Yang, L. Li, K. Lin, J. Wang, J. McAuley, J. Gao et al., “List items one by one: A new data source and learning paradigm for multimodal LLMs,” arXiv:2404.16375,

2024. 3.1.1, 3.2.2, 4.1.2, 2

- [61] J. Tang, C. Lin, Z. Zhao, S. Wei, B. Wu, Q. Liu, H. Feng, Y. Li, S. Wang, L. Liao et al., “TextSquare: Scaling up text-centric visual instruction tuning,” arXiv:2404.12803, 2024. 3.1.1, 5.1, 5.3, 5.4, 2, 8.3.2
- [62] H. P. Zou, V. Samuel, Y. Zhou, W. Zhang, L. Fang, Z. Song, P. S. Yu, and C. Caragea, “ImplicitAVE: An open-source dataset and multimodal llms benchmark for implicit attribute value extraction,” arXiv:2404.15592, 2024. 3.1.1, 4.4.1, 2
- [63] Y.-Q. Yu, M. Liao, J. Wu, Y. Liao, X. Zheng, and W. Zeng, “Texthawk: Exploring efficient fine-grained perception of multimodal large language models,” arXiv:2404.09204, 2024. 3.1.1, 2
- [64] Y. Zhao, Z. Lin, D. Zhou, Z. Huang, J. Feng, and B. Kang, “BuboGPT: Enabling visual grounding in multi-modal LLMs,” arXiv:2307.08581, 2023. 3.1.1, 5.2, 2
- [65] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, “Learn to explain: Multimodal reasoning via thought chains for science question answering,” NeurIPS, vol. 35, pp. 2507–2521, 2022. 3.1.1, 4.2.2, 2
- [66] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long, X. Deng, Y. Zhang, X. Liu, M. Huang, Z. Xiao et al., “Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding,” arXiv:2405.08748, 2024. 3.1.1, 3.2.2
- [67] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” arXiv:2306.05424, 2023. 3.1.1, 2

- [68] R. Xia, B. Zhang, H. Peng, N. Liao, P. Ye, B. Shi, J. Yan, and Y. Qiao, “Structchart: Perception, structuring, reasoning for visual chart understanding,” arXiv:2309.11268, 2023. 3.1.1, 4.2.1, 6.3, 2
- [69] Y.-Y. Li, Y. Bai, C. Wang, M. Qu, Z. Lu, R. Soria, and J. Liu, “Deep learning and llm-based methods applied to stellar lightcurve classification,” arXiv:2404.10757, 2024. 3.1.1
- [70] X. Wu, S. Huang, and F. Wei, “Multimodal large language model is a human-aligned annotator for text-to-image generation,” arXiv:2404.15100, 2024. 3.1.1, 3.1.3, 4.1.3, 5.1, 5.4, 2, 8.2.3, 8.3.2, 8.3.3
- [71] F. Liu, X. Wang, W. Yao, J. Chen, K. Song, S. Cho, Y. Yacoob, and D. Yu, “MMC: Advancing multimodal chart understanding with large-scale instruction tuning,” arXiv:2311.10774, 2023. 3.1.1, 4.4.1, 2
- [72] Y. Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai, Y. Qiao, and P. Luo, “EmbodiedGPT: Vision-language pretraining via embodied chain of thought,” NeurIPS, vol. 36, 2023. 3.1.1, 4.2.2, 5.1
- [73] S. Sreeram, T.-H. Wang, A. Maalouf, G. Rosman, S. Karaman, and D. Rus, “Probing multimodal llms as world models for driving,” arXiv:2405.05956, 2024. 3.1.1, 4.4.4
- [74] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and visionlanguage representation learning with noisy text supervision,” in ICML, 2021, pp. 4904–4916. 3.1.1
- [75] Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, and F. Huang, “mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration,” in CVPR, 2024, pp. 13040–13051. 3.1.2
- [76] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in ICML, 2023, pp. 19730–19742. 3.1.2, 3.2.4, 8.3.1
- [77] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian et al., “mplug-docowl: Modularized multimodal large language model for document understanding,” arXiv:2307.02499,

2023. 3.1.2, 6.3

- [78] D. Chen, J. Liu, W. Dai, and B. Wang, “Visual instruction tuning with polite flamingo,” in AAAI, vol. 38, no. 16, 2024, pp. 17745–

17753. 3.1.2

- [79] T. Vallaeys, M. Shukor, M. Cord, and J. Verbeek, “Improved baselines for data-efficient perceptual augmentation of llms,” arXiv:2403.13499, 2024. 3.1.2
- [80] Z. Li, L. Si, C. Guo, Y. Yang, and Q. Cao, “Data augmentation for text-based person retrieval using large language models,” arXiv:2405.11971, 2024. 3.1.2, 5.2
- [81] R. Chivereanu, A. Cosma, A. Catruna, R. Rughinis, and E. Radoi, “Aligning actions and walking to llm-generated textual descriptions,” arXiv:2404.12192, 2024. 3.1.2, 5.2
- [82] Q. Yu, Q. Sun, X. Zhang, Y. Cui, F. Zhang, Y. Cao, X. Wang, and J. Liu, “Capsfusion: Rethinking image-text data at scale,” in CVPR, 2024, pp. 14022–14032. 3.1.2
- [83] R. Yang, L. Song, Y. Li, S. Zhao, Y. Ge, X. Li, and Y. Shan, “GPT4Tools: Teaching large language model to use tools via selfinstruction,” NeurIPS, vol. 36, 2023. 3.1.2
- [84] Z. Feng, R. Zhang, and Z. Nie, “Improving composed image retrieval via contrastive learning with scaling positives and negatives,” arXiv:2404.11317, 2024. 3.1.2, 5.1
- [85] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” NeurIPS, vol. 35, pp. 23716–23736, 2022. 3.1.3, 3.2.2
- [86] P. Gao, R. Zhang, C. Liu, L. Qiu, S. Huang, W. Lin, S. Zhao, S. Geng, Z. Lin, P. Jin et al., “SPHINX-X: Scaling data and parameters for a family of multi-modal large language models,” arXiv:2402.05935, 2024. 3.1.3
- [87] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan et al., “Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs,” arXiv:2406.16860, 2024. 3.1.3
- [88] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu et al., “PaLM-E: An embodied multimodal language model,” in ICML, 2023, pp. 8469–8488. 3.1.3
- [89] B. Liu, C. Lyu, Z. Min, Z. Wang, J. Su, and L. Wang, “Retrievalaugmented multi-modal chain-of-thoughts reasoning for large language models,” arXiv:2312.01714, 2023. 3.1.3, 4.1.2, 4.2.2

- [90] D. Chen, Y. Huang, Z. Ma, H. Chen, X. Pan, C. Ge, D. Gao, Y. Xie, Z. Liu, J. Gao, Y. Li, B. Ding, and J. Zhou, “Data-juicer: A one-stop data processing system for large language models,” in SIGMOD,

2024. 3.2, 3.2.1

- [91] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu et al., “Lima: Less is more for alignment,” NeurIPS, vol. 36, 2023. 3.2
- [92] G. Kolossov, A. Montanari, and P. Tandon, “Towards a statistical theory of data selection under weak supervision,” in ICLR, 2024. 3.2.1, 5.3
- [93] B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli, and A. Morcos, “Beyond neural scaling laws: beating power law scaling via data pruning,” NeurIPS, vol. 35, pp. 19523–19536, 2022. 3.2.1
- [94] R. Webster, J. Rabin, L. Simon, and F. Jurie, “On the deduplication of laion-2b,” arXiv:2303.12733, 2023. 3.2.1
- [95] O. Jafari, P. Maurya, P. Nagarkar, K. M. Islam, and C. Crushev, “A survey on locality sensitive hashing algorithms and their applications,” arXiv:2102.08942, 2021. 3.2.1
- [96] R. Beaumont, “Clip retrieval: Easily compute clip embeddings and build a clip retrieval system with them,” https://github.com /rom1504/clip-retrieval, 2022. 3.2.1
- [97] L. Theis, W. Shi, A. Cunningham, and F. Husz´ar, “Lossy image compression with compressive autoencoders,” in ICLR, 2022. 3.2.1
- [98] A. Abbas, K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos, “Semdedup: Data-efficient learning at web-scale through semantic deduplication,” arXiv:2303.09540, 2023. 3.2.1
- [99] T.-H. Huang, C. Shin, S. J. Tay, D. Adila, and F. Sala, “Multimodal data curation via object detection and filter ensembles,” arXiv:2401.12225, 2024. 3.2.1, 3.2.4
- [100] F. Radenovic, A. Dubey, A. Kadian, T. Mihaylov, S. Vandenhende, Y. Patel, Y. Wen, V. Ramanathan, and D. Mahajan, “Filtering, distillation, and hard negatives for vision-language pre-training,” in CVPR, 2023, pp. 6967–6977. 3.2.1
- [101] A. Mahmoud, M. Elhoushi, A. Abbas, Y. Yang, N. Ardalani, H. Leather, and A. Morcos, “Sieve: Multimodal dataset pruning using image captioning models,” arXiv:2310.02110, 2023. 3.2.1, 3.2.4, 8.2.2, 8.3.1, 8.3.2
- [102] P. Maini, S. Goyal, Z. C. Lipton, J. Z. Kolter, and A. Raghunathan, “T-mars: Improving visual representations by circumventing text feature learning,” arXiv:2307.03132, 2023. 3.2.1, 3.2.4
- [103] T. Nguyen, S. Y. Gadre, G. Ilharco, S. Oh, and L. Schmidt, “Improving multimodal datasets with image captioning,” NeurIPS, vol. 36, 2023. 3.2.1, 3.2.4, 8.2.2, 8.3.2
- [104] H. Yu, Y. Tian, S. Kumar, L. Yang, and H. Wang, “The devil is in the details: A deep dive into the rabbit hole of data filtering,” arXiv:2309.15954, 2023. 3.2.1, 5.3, 8.3.2
- [105] W. Wang, K. Mrini, L. Yang, S. Kumar, Y. Tian, X. Yan, and H. Wang, “Finetuned multimodal language models are highquality image-text data filters,” arXiv:2403.02677, 2024. 3.2.1, 5.1, 5.3, 8.3.2
- [106] A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. Toshev, and V. Shankar, “Data filtering networks,” arXiv:2309.17425, 2023. 3.2.1, 3.2.4, 8.3.2
- [107] L. Wei, Z. Jiang, W. Huang, and L. Sun, “Instructiongpt4: A 200-instruction paradigm for fine-tuning minigpt-4,” arXiv:2308.12067, 2023. 3.2.1
- [108] H. Xu, S. Xie, X. Tan, P.-Y. Huang, R. Howes, V. Sharma, S.-W. Li, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer, “Demystifying CLIP data,” in ICLR, 2024. 3.2.2
- [109] Y.-D. Tsai, T.-Y. Yen, P.-F. Guo, Z.-Y. Li, and S.-D. Lin, “Text-centric alignment for multi-modality learning,” arXiv:2402.08086, 2024. 3.2.2, 3.2.4, 5.2
- [110] T. Nguyen, G. Ilharco, M. Wortsman, S. Oh, and L. Schmidt, “Quality not quantity: On the interaction between dataset design and robustness of clip,” NeurIPS, vol. 35, pp. 21455–21469, 2022. 3.2.2
- [111] Z. Liu and K. He, “A decade’s battle on dataset bias: Are we there yet?” arXiv:2403.08632, 2024. 3.2.2
- [112] J. Ma, P.-Y. Huang, S. Xie, S.-W. Li, L. Zettlemoyer, S.-F. Chang, W.-T. Yih, and H. Xu, “Mode: Clip data experts via clustering,” arXiv:2404.16030, 2024. 3.2.3
- [113] M. Dehghani, B. Mustafa, J. Djolonga, J. Heek, M. Minderer, M. Caron, A. Steiner, J. Puigcerver, R. Geirhos, I. M. Alabdulmohsin et al., “Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution,” NeurIPS, vol. 36, 2024. 3.2.3

- [114] Y. Liu, K. Zhang, Y. Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y. Huang, H. Sun, J. Gao et al., “Sora: A review on background, technology, limitations, and opportunities of large vision models,” arXiv:2402.17177, 2024. 3.2.3
- [115] H. Ding, Z. Wang, G. Paolini, V. Kumar, A. Deoras, D. Roth, and S. Soatto, “Fewer truncations improve language modeling,” arXiv:2404.10830, 2024. 3.2.3
- [116] K. Staniszewski, S. Tworkowski, S. Jaszczur, H. Michalewski, Ł. Kucinski,´ and P. Miło´s, “Structured packing in llm training improves long context utilization,” arXiv:2312.17296, 2023. 3.2.3
- [117] H. Han, K. Miao, Q. Zheng, and M. Luo, “Noisy correspondence learning with meta similarity correction,” in CVPR, 2023, pp. 7517–7526. 3.2.4
- [118] R. Bavishi, E. Elsen, C. Hawthorne, M. Nye, A. Odena, A. Somani, and S. Ta¸sırlar, “Introducing our multimodal models,” 2023. 3.2.4
- [119] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv:2311.15127, 2023. 3.2.4
- [120] F. Zhao, T. Pang, C. Li, Z. Wu, J. Guo, S. Xing, and X. Dai, “Aligngpt: Multi-modal large language models with adaptive alignment capability,” arXiv:2405.14129, 2024. 3.2.4
- [121] A. Maharana, P. Yadav, and M. Bansal, “D2 pruning: Message passing for balancing diversity & difficulty in data pruning,” in ICLR, 2024. 3.3
- [122] C. Ge, Z. Ma, D. Chen, Y. Li, and B. Ding, “Data mixing made efficient: A bivariate scaling law for language model pretraining,” arXiv:2405.14908, 2024. 3.3
- [123] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v,” arXiv:2310.11441, 2023. 4.1.1
- [124] W. Lin, X. Wei, R. An, P. Gao, B. Zou, Y. Luo, S. Huang, S. Zhang, and H. Li, “Draw-and-understand: Leveraging visual prompts to enable MLLMs to comprehend what you want,” arXiv:2403.20271, 2024. 4.1.1, 2
- [125] Y. Gou, K. Chen, Z. Liu, L. Hong, H. Xu, Z. Li, D.-Y. Yeung, J. T. Kwok, and Y. Zhang, “Eyes closed, safety on: Protecting multimodal llms via image-to-text transformation,” arXiv:2403.09572,

2024. 4.1.1, 4.3.1, 5.4

- [126] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” arXiv:2306.15195, 2023. 4.1.1
- [127] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei, “Kosmos-2: Grounding multimodal large language models to the world,” arXiv:2306.14824, 2023. 4.1.1
- [128] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan et al., “Time-LLM: Time series forecasting by reprogramming large language models,” arXiv:2310.01728,

2023. 4.1.1

- [129] L. Fan, K. Chen, D. Krishnan, D. Katabi, P. Isola, and Y. Tian, “Scaling laws of synthetic images for model training... for now,” in CVPR, 2024, pp. 7382–7392. 4.1.1
- [130] Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang, “Generative multimodal models are incontext learners,” in CVPR, 2024, pp. 14398–14409. 4.1.2
- [131] Z. Li, Q. Xu, D. Zhang, H. Song, Y. Cai, Q. Qi, R. Zhou, J. Pan, Z. Li, V. T. Vu et al., “Groundinggpt: Language enhanced multimodal grounding model,” arXiv:2401.06071, 2024. 4.1.2
- [132] H. Zhao, Z. Cai, S. Si, X. Ma, K. An, L. Chen, Z. Liu, S. Wang, W. Han, and B. Chang, “MMICL: Empowering vision-language model with multi-modal in-context learning,” in ICLR, 2024. 4.1.2
- [133] S. Doveh, S. Perek, M. J. Mirza, A. Alfassy, A. Arbelle, S. Ullman, and L. Karlinsky, “Towards multimodal in-context learning for vision & language models,” arXiv:2403.12736, 2024. 4.1.2
- [134] J. Gao, Q. Qiao, Z. Cao, Z. Wang, and W. Li, “AIM: Let any multi-modal large language models embrace efficient in-context learning,” arXiv:2406.07588, 2024. 4.1.2
- [135] L. Wang, W. Xu, Z. Hu, Y. Lan, S. Dong, H. Wang, R. K.-W. Lee, and E.-P. Lim, “All in an aggregated image for in-image learning,” arXiv:2402.17971, 2024. 4.1.2
- [136] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou, “Hallucination of multimodal large language models: A survey,” arXiv:2404.18930, 2024. 4.1.3
- [137] Z. Yin, J. Wang, J. Cao, Z. Shi, D. Liu, M. Li, X. Huang, Z. Wang, L. Sheng, L. Bai et al., “Lamm: Language-assisted multimodal instruction-tuning dataset, framework, and benchmark,” NeurIPS, vol. 36, 2023. 4.1.3, 4.4.1, 2

- [138] W. Ge, S. Chen, G. Chen, J. Chen, Z. Chen, S. Yan, C. Zhu, Z. Lin, W. Xie, X. Wang et al., “MLLM-bench: Evaluating multimodal llms with per-sample criteria,” arXiv:2311.13951, 2023. 4.1.3, 4.4.2, 5.4, 2
- [139] Y. Chen, K. Sikka, M. Cogswell, H. Ji, and A. Divakaran, “Dress: Instructing large vision-language models to align and interact with humans via natural language feedback,” in CVPR, 2024, pp. 14239–14250. 4.1.3
- [140] L. Li, Z. Xie, M. Li, S. Chen, P. Wang, L. Chen, Y. Yang, B. Wang, and L. Kong, “Silkie: Preference distillation for large visual language models,” arXiv:2405.2312.10665, 2023. 4.1.3
- [141] T. Yu, Y. Yao, H. Zhang, T. He, Y. Han, G. Cui, J. Hu, Z. Liu, H.T. Zheng, M. Sun et al., “RLHF-V: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback,” in CVPR, 2024, pp. 13807–13816. 4.1.3
- [142] M. Zhang and K. Rong, “Automated multi-level preference for mllms,” arXiv:2405.11165, 2024. 4.1.3
- [143] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y. Gui, Y.-X. Wang, Y. Yang et al., “Aligning large multimodal models with factually augmented RLHF,” arXiv:2309.14525, 2023. 4.1.3
- [144] Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, and N. Yu, “Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation,” in CVPR, 2024, pp. 13418–13427. 4.1.3
- [145] F. Liu, K. Lin, L. Li, J. Wang, Y. Yacoob, and L. Wang, “Mitigating hallucination in large multi-modal models via robust instruction tuning,” in ICLR, 2023. 4.1.3, 2
- [146] B. Xie, S. Zhang, Z. Zhou, B. Li, Y. Zhang, J. Hessel, J. Yang, and Z. Liu, “FunQA: Towards surprising video comprehension,” arXiv:2306.14899, 2023. 4.2.1, 4.4.4, 2
- [147] X. Gai, C. Zhou, J. Liu, Y. Feng, J. Wu, and Z. Liu, “MedThink: Explaining medical visual question answering via multimodal decision-making rationale,” arXiv:2404.12372, 2024. 4.2.2, 5.2
- [148] G. Zheng, B. Yang, J. Tang, H.-Y. Zhou, and S. Yang, “DDCoT: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models,” NeurIPS, vol. 36, pp. 5168–5191,

2023. 4.2.2

- [149] J. H. Cho, B. Ivanovic, Y. Cao, E. Schmerling, Y. Wang, X. Weng, B. Li, Y. You, P. Kr¨ahenbuhl,¨ Y. Wang et al., “Language-image models with 3d understanding,” arXiv:2405.03685, 2024. 4.2.2
- [150] H. Chen, X. Wang, H. Chen, Z. Song, J. Jia, and W. Zhu, “Grounding-Prompter: Prompting LLM with multimodal information for temporal sentence grounding in long videos,” arXiv:2312.17117, 2023. 4.2.2
- [151] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang, “Mm-react: Prompting chatgpt for multimodal reasoning and action,” arXiv:2303.11381, 2023. 4.2.2
- [152] P. Lu, B. Peng, H. Cheng, M. Galley, K.-W. Chang, Y. N. Wu, S.C. Zhu, and J. Gao, “Chameleon: Plug-and-play compositional reasoning with large language models,” NeurIPS, vol. 36, 2023. 4.2.2
- [153] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in CVPR, 2023, pp. 14953–14962. 4.2.2
- [154] Y. Fan, Y. Cao, Z. Zhao, Z. Liu, and S. Li, “Unbridled icarus: A survey of the potential perils of image inputs in multimodal large language model security,” arXiv:2404.05264, 2024. 4.3.1
- [155] R. Pi, T. Han, Y. Xie, R. Pan, Q. Lian, H. Dong, J. Zhang, and T. Zhang, “Mllm-protector: Ensuring mllm’s safety without hurting performance,” arXiv:2401.02906, 2024. 4.3.1
- [156] Y. Wang, W. Hu, Y. Dong, and R. Hong, “Exploring transferability of multimodal adversarial samples for vision-language pre-training models with contrastive learning,” arXiv:2308.12636,

2023. 4.3.1

- [157] D. Lu, Z. Wang, T. Wang, W. Guan, H. Gao, and F. Zheng, “Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models,” in ICCV, 2023, pp. 102–

111. 4.3.1

- [158] C. Schlarmann and M. Hein, “On the adversarial robustness of multi-modal foundation models,” in CVPR, 2023, pp. 3677–3685. 4.3.1
- [159] B. He, X. Jia, S. Liang, T. Lou, Y. Liu, and X. Cao, “Sa-attack: Improving adversarial transferability of vision-language pretraining models via self-augmentation,” arXiv:2312.04913, 2023. 4.3.1
- [160] X. Tao, S. Zhong, L. Li, Q. Liu, and L. Kong, “Imgtrojan: Jailbreaking vision-language models with one image,” arXiv:2403.02910,

2024. 4.3.1, 5.4

- [161] E. Bagdasaryan, T.-Y. Hsieh, B. Nassi, and V. Shmatikov, “(ab) using images and sounds for indirect instruction injection in multi-modal llms,” arXiv:2307.10490, 2023. 4.3.1
- [162] Y. Wu, X. Li, Y. Liu, P. Zhou, and L. Sun, “Jailbreaking gpt-4v via self-adversarial attacks with system prompts,” arXiv:2311.09127,

2023. 4.3.1

- [163] Z. Tan, C. Zhao, R. Moraffah, Y. Li, Y. Kong, T. Chen, and H. Liu, “The wolf within: Covert injection of malice into mllm societies via an mllm operative,” arXiv:2402.14859, 2024. 4.3.1
- [164] J. Liang, S. Liang, M. Luo, A. Liu, D. Han, E.-C. Chang, and X. Cao, “Vl-trojan: Multimodal instruction backdoor attacks against autoregressive visual language models,” arXiv:2402.13851, 2024. 4.3.1
- [165] X. Liu, Y. Zhu, J. Gu, Y. Lan, C. Yang, and Y. Qiao, “Mmsafetybench: A benchmark for safety evaluation of multimodal large language models,” arXiv:2311.17600, 2023. 4.3.1, 4.4.2, 2
- [166] Y. Sun, J. He, S. Lei, L. Cui, and C.-T. Lu, “Med-mmhl: A multimodal dataset for detecting human-and llm-generated misinformation in the medical domain,” arXiv:2306.08871, 2023. 4.3.1, 2
- [167] N. Lukas, A. Salem, R. Sim, S. Tople, L. Wutschitz, and S. ZanellaB´eguelin, “Analyzing leakage of personally identifiable information in language models,” in SP. IEEE, 2023, pp. 346–363. 4.3.2
- [168] M. Li, L. Li, Y. Yin, M. Ahmed, Z. Liu, and Q. Liu, “Red teaming visual language models,” arXiv:2401.12915, 2024. 4.3.2
- [169] J. Rao, S. Gao, G. Mai, and K. Janowicz, “Building privacypreserving and secure geospatial artificial intelligence foundation models (vision paper),” in SIGSPATIAL, 2023, pp. 1–4. 4.3.2
- [170] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “GPT-4 technical report,” arXiv:2303.08774, 2023. 4.3.2
- [171] I. Joshi, M. Grimmer, C. Rathgeb, C. Busch, F. Bremond, and A. Dantcheva, “Synthetic data in human analysis: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–20, 2024. 4.3.2
- [172] A. Huang, P. Liu, R. Nakada, L. Zhang, and W. Zhang, “Safeguarding data in multimodal AI: A differentially private approach to clip training,” arXiv:2306.08173, 2023. 4.3.2
- [173] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in AISTATS, 2017, pp. 1273–1282. 4.3.2
- [174] Z. Qin, D. Chen, B. Qian, B. Ding, Y. Li, and S. Deng, “Federated full-parameter tuning of billion-sized language models with communication cost under 18 kilobytes,” in ICML, 2024. 4.3.2
- [175] Z. Ling, D. Chen, L. Yao, Y. Li, and Y. Shen, “On the convergence of zeroth-order federated tuning for large language models,” in KDD, 2024. 4.3.2
- [176] J. Bai, D. Chen, B. Qian, L. Yao, and Y. Li, “Federated fine-tuning of large language models under heterogeneous language tasks and client resources,” arXiv:2402.11505, 2024. 4.3.2
- [177] J. Andrews, D. Zhao, W. Thong, A. Modas, O. Papakyriakopoulos, and A. Xiang, “Ethical considerations for responsible data curation,” NeurIPS, vol. 36, 2023. 4.3.2
- [178] P. Ombredanne, “Free and open source software license compliance: Tools for software composition analysis,” Computer, vol. 53, no. 10, pp. 105–109, 2020. 4.3.2
- [179] D. M. German, Y. Manabe, and K. Inoue, “A sentence-matching method for automatic license identification of source code files,” in ASE, 2010, pp. 437–446. 4.3.2
- [180] M. Duan, Q. Li, and B. He, “Modelgo: A practical tool for machine learning license analysis,” in WWW, 2024, pp. 1158–

1169. 4.3.2, 8.2.1

- [181] Y. Tang, J. Yu, K. Gai, X. Qu, Y. Hu, G. Xiong, and Q. Wu, “Watermarking vision-language pre-trained models for multimodal embedding as a service,” arXiv:2311.05863, 2023. 4.3.2
- [182] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, “Blink: Multimodal large language models can see but not perceive,” arXiv:2404.12390, 2024. 4.4.1, 2
- [183] J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang, “Onechart: Purify the chart structural extraction via one auxiliary token,” arXiv:2404.09987, 2024. 4.4.1, 5.1, 6.3, 2
- [184] Z. Zhou, Q. Wang, B. Lin, Y. Su, R. Chen, X. Tao, A. Zheng, L. Yuan, P. Wan, and D. Zhang, “Uniaa: A unified multimodal image aesthetic assessment baseline and benchmark,” arXiv:2404.09619, 2024. 4.4.1, 5.1
- [185] Y. Shi, F. Lv, X. Wang, C. Xia, S. Li, S. Yang, T. Xi, and G. Zhang, “Open-transmind: A new baseline and benchmark for 1st foundation model challenge of intelligent transportation,” in CVPR, 2023, pp. 6327–6334. 4.4.1, 4.4.3

- [186] J. Zhang, T. Hu, X. Huang, Y. Gong, and D. Zeng, “3dbench: A scalable 3d benchmark and instruction-tuning dataset,” arXiv:2404.14678, 2024. 4.4.1, 2
- [187] M. Li, X. Chen, C. Zhang, S. Chen, H. Zhu, F. Yin, G. Yu, and T. Chen, “M3dbench: Let’s instruct large models with multimodal 3d prompts,” arXiv:2312.10763, 2023. 4.4.1
- [188] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo et al., “MVBench: A comprehensive multi-modal video understanding benchmark,” arXiv:2311.17005, 2023. 4.4.1, 2
- [189] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-Bench: Benchmarking multimodal LLMs with generative comprehension,” arXiv:2307.16125, 2023. 4.4.1, 2
- [190] J. An, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, L. Wang, and J. Luo, “Openleaf: Open-domain interleaved image-text generation and evaluation,” arXiv:2310.07749, 2023. 4.4.2
- [191] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” arXiv:2311.17982, 2023. 4.4.2, 2
- [192] Y. Liu, X. Cun, X. Liu, X. Wang, Y. Zhang, H. Chen, Y. Liu, T. Zeng, R. Chan, and Y. Shan, “Evalcrafter: Benchmarking and evaluating large video generation models,” arXiv:2310.11440, 2023. 4.4.2, 2
- [193] Z. Ge, H. Huang, M. Zhou, J. Li, G. Wang, S. Tang, and Y. Zhuang, “Worldgpt: Empowering llm as multimodal world model,” arXiv:2404.18202, 2024. 4.4.2, 2
- [194] D. Chen, R. Chen, S. Zhang, Y. Liu, Y. Wang, H. Zhou, Q. Zhang, P. Zhou, Y. Wan, and L. Sun, “MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with vision-language benchmark,” arXiv:2402.04788, 2024. 4.4.2, 5.4, 2, 8.3.2, 8.3.3
- [195] W. Huang, H. Liu, M. Guo, and N. Z. Gong, “Visual hallucinations of multi-modal large language models,” arXiv:2402.14683,

2024. 4.4.2, 5.3, 2

- [196] X. Chen, C. Wang, Y. Xue, N. Zhang, X. Yang, Q. Li, Y. Shen, J. Gu, and H. Chen, “Unified hallucination detection for multimodal large language models,” arXiv:2402.03190, 2024. 4.4.2, 5.2, 6.2, 2
- [197] T. Zhao, L. Zhang, Y. Ma, and L. Cheng, “A survey on safe multimodal learning system,” arXiv:2402.05355, 2024. 4.4.2
- [198] Z. Niu, H. Ren, X. Gao, G. Hua, and R. Jin, “Jailbreaking attack against multimodal large language model,” arXiv:2402.02309,

2024. 4.4.2

- [199] K. Ying, F. Meng, J. Wang, Z. Li, H. Lin, Y. Yang, H. Zhang, W. Zhang, Y. Lin, S. Liu et al., “MMT-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI,” arXiv:2404.16006, 2024. 4.4.3, 2
- [200] R. Xia, B. Zhang, H. Ye, X. Yan, Q. Liu, H. Zhou, Z. Chen, M. Dou, B. Shi, J. Yan et al., “Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning,” arXiv:2402.12185, 2024. 4.4.4, 5.1, 6.3, 2
- [201] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, “Chartqa: A benchmark for question answering about charts with visual and logical reasoning,” arXiv:2203.10244, 2022. 4.4.4, 6.3, 2
- [202] L. Zhang, X. Zhai, Z. Zhao, Y. Zong, X. Wen, and B. Zhao, “What if the tv was off? examining counterfactual reasoning abilities of multi-modal language models,” in CVPR, 2024, pp. 21853–21862. 4.4.4, 5.1, 5.3
- [203] Q. Chen, L. Qin, J. Zhang, Z. Chen, X. Xu, and W. Che, “M3cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought,” arXiv:2405.16473, 2024. 4.4.4, 2
- [204] F. B. Baldassini, M. Shukor, M. Cord, L. Soulier, and B. Piwowarski, “What makes multimodal in-context learning work?” in CVPR, 2024, pp. 1539–1550. 4.5
- [205] H. Tu, B. Zhao, C. Wei, and C. Xie, “Sight beyond text: Multi-modal training enhances llms in truthfulness and ethics,” arXiv:2309.07120, 2023. 4.5
- [206] C. Xu, G. Saranathan, M. P. Alam, A. Shah, J. Lim, S. Y. Wong, F. Martin, and S. Bhattacharya, “Data efficient evaluation of large language models and text-to-image models via adaptive sampling,” arXiv:2406.15527, 2024. 4.5
- [207] Y. Du, H. Guo, K. Zhou, W. X. Zhao, J. Wang, C. Wang, M. Cai, R. Song, and J.-R. Wen, “What makes for good visual instructions? synthesizing complex visual reasoning instructions for visual instruction tuning,” arXiv:2311.01487, 2023. 5.1, 2
- [208] Z. Li, C. Zhang, X. Wang, R. Ren, Y. Xu, R. Ma, and X. Liu, “3dmit: 3d multi-modal instruction tuning for scene understanding,” arXiv:2401.03201, 2024. 5.1, 2
- [209] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” arXiv:2305.06355, 2023. 5.1, 5.2, 2

- [210] Y. Han, C. Zhang, X. Chen, X. Yang, Z. Wang, G. Yu, B. Fu, and H. Zhang, “ChartLlama: A multimodal llm for chart understanding and generation,” arXiv:2311.16483, 2023. 5.1, 6.3, 6.4, 2
- [211] J. Zhan, J. Dai, J. Ye, Y. Zhou, D. Zhang, Z. Liu, X. Zhang, R. Yuan, G. Zhang, L. Li et al., “AnyGPT: Unified multimodal LLM with discrete sequence modeling,” arXiv:2402.12226, 2024. 5.1
- [212] G. Gu, S. Chun, W. Kim, H. Jun, Y. Kang, and S. Yun, “Compodiff: Versatile composed image retrieval with latent diffusion,” arXiv:2303.11916, 2023. 5.1
- [213] L. Ventura, A. Yang, C. Schmid, and G. Varol, “Covr: Learning composed video retrieval from web video captions,” in AAAI, vol. 38, no. 6, 2024, pp. 5270–5279. 5.1
- [214] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in CVPR, 2023, pp. 18392–18402. 5.1
- [215] M. Levy, R. Ben-Ari, N. Darshan, and D. Lischinski, “Data roaming and quality assessment for composed image retrieval,” in AAAI, vol. 38, no. 4, 2024, pp. 2991–2999. 5.2
- [216] S. Ma, L. Wang, S. Hou, and B. Yan, “Aligned with llm: a new multi-modal training paradigm for encoding fmri activity in visual cortex,” arXiv:2401.03851, 2024. 5.2
- [217] L. Zhang, Y. Wu, F. Mo, J.-Y. Nie, and A. Agrawal, “Moqagpt: Zero-shot multi-modal open-domain question answering with large language model,” arXiv:2310.13265, 2023. 5.2
- [218] Y. Huang, X. Sheng, Z. Yang, Q. Yuan, Z. Duan, P. Chen, L. Li, W. Lin, and G. Shi, “AesExpert: Towards multi-modality foundation model for image aesthetics perception,” arXiv:2404.09624,

2024. 5.2

- [219] Y. Tang, C.-M. Chang, and X. Yang, “Pdfchatannotator: A humanllm collaborative multi-modal data annotation tool for pdfformat catalogs,” in IUI, 2024, pp. 419–430. 5.2, 6.2
- [220] Y. Zhu, H. Yuan, S. Wang, J. Liu, W. Liu, C. Deng, Z. Dou, and J.-R. Wen, “Large language models for information retrieval: A survey,” arXiv:2308.07107, 2023. 6.1
- [221] Z. Jing, Y. Su, Y. Han, B. Yuan, C. Liu, H. Xu, and K. Chen, “When large language models meet vector databases: A survey,” arXiv:2402.01763, 2024. 6.1
- [222] R. C. Fernandez, A. J. Elmore, M. J. Franklin, S. Krishnan, and C. Tan, “How large language models will disrupt data management,” Proceedings of the VLDB Endowment, vol. 16, no. 11, pp. 3302–3309, 2023. 6.1
- [223] X. Long, J. Zeng, F. Meng, Z. Ma, K. Zhang, B. Zhou, and J. Zhou, “Generative multi-modal knowledge retrieval with large language models,” arXiv:2401.08206, 2024. 6.1
- [224] S. Miret and N. Krishnan, “Are llms ready for real-world materials discovery?” arXiv:2402.05200, 2024. 6.1, 8.2.1
- [225] Y. Luo, Z. Zheng, Z. Zhu, and Y. You, “How does the textual information affect the retrieval of multimodal in-context learning?” arXiv:2404.12866, 2024. 6.1
- [226] D. Caffagni, F. Cocchi, N. Moratelli, S. Sarto, M. Cornia, L. Baraldi, and R. Cucchiara, “Wiki-llava: Hierarchical retrievalaugmented generation for multimodal llms,” arXiv:2404.15406,

2024. 6.1

- [227] J. Xu, Y. Huang, J. Hou, G. Chen, Y. Zhang, R. Feng, and W. Xie, “Retrieval-augmented egocentric video captioning,” arXiv:2401.00789, 2024. 6.1
- [228] Z. Chen, Y. Zhang, Y. Fang, Y. Geng, L. Guo, X. Chen, Q. Li, W. Zhang, J. Chen, Y. Zhu et al., “Knowledge graphs meet multimodal learning: A comprehensive survey,” arXiv:2402.05391,

2024. 6.1

- [229] J. Lee, Y. Wang, J. Li, and M. Zhang, “Multimodal reasoning with multimodal knowledge graph,” arXiv:2406.02030, 2024. 6.1
- [230] V. Perot, K. Kang, F. Luisier, G. Su, X. Sun, R. S. Boppana, Z. Wang, J. Mu, H. Zhang, and N. Hua, “LMDX: Language model-based document information extraction and localization,” arXiv:2309.10952, 2023. 6.2
- [231] A. Biswas and W. Talukdar, “Robustness of structured data extraction from in-plane rotated documents using multi-modal large language models (llm),” Journal of Artificial Intelligence Research, vol. 4, no. 1, pp. 176–195, 2024. 6.2
- [232] H. Wu, Y. Yuan, L. Mikaelyan, A. Meulemans, X. Liu, J. Hensman, and B. Mitra, “Structured entity extraction using large language models,” arXiv:2402.04437, 2024. 6.2
- [233] J. Li, H. Li, D. Sun, J. Wang, W. Zhang, Z. Wang, and G. Pan, “LLMs as bridges: Reformulating grounded multimodal named entity recognition,” arXiv:2402.09989, 2024. 6.2

- [234] B. Zhang and H. Soh, “Extract, define, canonicalize: An llm-based framework for knowledge graph construction,” arXiv:2404.03868,

2024. 6.2

- [235] W. He, H. Ma, S. Li, H. Dong, H. Zhang, and J. Feng, “Using augmented small multimodal models to guide large language models for multimodal relation extraction,” Applied Sciences, vol. 13, no. 22, p. 12208, 2023. 6.2
- [236] M. M. Hassan, A. Knipper, and S. K. K. Santu, “Chatgpt as your personal data scientist,” arXiv:2305.13657, 2023. 6.3, 6.4
- [237] L. Cheng, X. Li, and L. Bing, “Is gpt-4 a good data analyst?” arXiv:2305.15038, 2023. 6.3
- [238] A. Hu, H. Xu, J. Ye, M. Yan, L. Zhang, B. Zhang, C. Li, J. Zhang, Q. Jin, F. Huang et al., “mplug-docowl 1.5: Unified structure learning for ocr-free document understanding,” arXiv:2403.12895,

2024. 6.3

- [239] Y. Luo, R. An, B. Zou, Y. Tang, J. Liu, and S. Zhang, “LLM as dataset analyst: Subpopulation structure discovery with large language model,” arXiv:2405.02363, 2024. 6.3
- [240] A. Hu, Y. Shi, H. Xu, J. Ye, Q. Ye, M. Yan, C. Li, Q. Qian, J. Zhang, and F. Huang, “mplug-paperowl: Scientific diagram analysis with the multimodal large language model,” arXiv:2311.18248,

2023. 6.3

- [241] F. Jiang, K. Wang, and H. Li, “Bridging research and readers: A multi-modal automated academic papers interpretation system,” arXiv:2401.09150, 2024. 6.3
- [242] H. Cai, X. Cai, J. Chang, S. Li, L. Yao, C. Wang, Z. Gao, Y. Li, M. Lin, S. Yang et al., “Sciassess: Benchmarking llm proficiency in scientific literature analysis,” arXiv:2403.01976, 2024. 6.3
- [243] X. Wu, R. Zheng, J. Sha, T.-L. Wu, H. Zhou, M. Tang, K.-W. Chang, N. Peng, and H. Huang, “Daco: Towards applicationdriven and comprehensive data analysis via code generation,” arXiv:2403.02528, 2024. 6.3, 8.2.1
- [244] T. Yang, Y. Luo, Z. Qi, Y. Wu, Y. Shan, and C. W. Chen, “Posterllava: Constructing a unified multi-modal layout generator with llm,” arXiv:2406.02884, 2024. 6.4
- [245] W. Zhang, Y. Shen, W. Lu, and Y. Zhuang, “Data-copilot: Bridging billions of data and humans with autonomous workflow,” arXiv:2306.07209, 2023. 6.4
- [246] V. Dibia, “LIDA: A tool for automatic generation of grammaragnostic visualizations and infographics using large language models,” in ACL, 2023. 6.4
- [247] P.-P. V´azquez, “Are llms ready for visualization?” in PacificVis. IEEE, 2024, pp. 343–352. 6.4
- [248] Y. Wu, Y. Wan, H. Zhang, Y. Sui, W. Wei, W. Zhao, G. Xu, and H. Jin, “Automated data visualization from natural language via large language models: An exploratory study,” PACMMOD, vol. 2, no. 3, pp. 1–28, 2024. 6.4
- [249] Y. He, Z. Liu, J. Chen, Z. Tian, H. Liu, X. Chi, R. Liu, R. Yuan, Y. Xing, W. Wang et al., “Llms meet multimodal generation and editing: A survey,” arXiv:2405.19334, 2024. 6.4
- [250] T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren, M.-H. Yang et al., “Panda70m: Captioning 70m videos with multiple cross-modality teachers,” arXiv:2402.19479, 2024. 2

- [251] Y. Wang, Y. He, Y. Li, K. Li, J. Yu, X. Ma, X. Li, G. Chen, X. Chen, Y. Wang et al., “InternVid: A large-scale video-text dataset for multimodal understanding and generation,” arXiv:2307.06942,

2023. 2

- [252] L. Chen, X. Wei, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, B. Lin, Z. Tang et al., “Sharegpt4video: Improving video understanding and generation with better captions,” arXiv:2406.04325, 2024. 2
- [253] L. Li, Y. Yin, S. Li, L. Chen, P. Wang, S. Ren, M. Li, Y. Yang, J. Xu, X. Sun et al., “M3it: A large-scale dataset towards multi-modal multilingual instruction tuning,” arXiv:2306.04387, 2023. 2
- [254] F. Meng, W. Shao, Q. Lu, P. Gao, K. Zhang, Y. Qiao, and P. Luo, “Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning,” arXiv:2401.02384, 2024. 2
- [255] H. Xu, Q. Ye, X. Wu, M. Yan, Y. Miao, J. Ye, G. Xu, A. Hu, Y. Shi, G. Xu et al., “Youku-mplug: A 10 million large-scale chinese video-language dataset for pre-training and benchmarks,” arXiv:2306.04362, 2023. 2
- [256] B. He, J. Wang, J. Qiu, T. Bui, A. Shrivastava, and Z. Wang, “Align and attend: Multimodal summarization with dual contrastive losses,” in CVPR, 2023, pp. 14867–14878. 2
- [257] W. Zhu, J. Hessel, A. Awadalla, S. Y. Gadre, J. Dodge, A. Fang, Y. Yu, L. Schmidt, W. Y. Wang, and Y. Choi, “Multimodal C4: An open, billion-scale corpus of images interleaved with text,” arXiv:2304.06939, 2023. 2
- [258] V. Patraucean, L. Smaira, A. Gupta, A. Recasens, L. Markeeva, D. Banarse, S. Koppula, M. Malinowski, Y. Yang, C. Doersch et al., “Perception test: A diagnostic benchmark for multimodal video models,” NeurIPS, vol. 36, 2023. 2
- [259] R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K.-W. Chang, P. Gao et al., “Mathverse: Does your multimodal llm truly see the diagrams in visual math problems?” arXiv:2403.14624, 2024. 2
- [260] D. Chen, H. Wang, Y. Huang, C. Ge, Y. Li, B. Ding, and J. Zhou, “Data-juicer sandbox: A comprehensive suite for multimodal data-model co-development,” arXiv:2407.11784, 2024. 8.1, 8.3.1
- [261] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu, “Deep mutual learning,” in CVPR, 2018, pp. 4320–4328. 8.3.1
- [262] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon et al., “Constitutional ai: harmlessness from ai feedback. 2022,” arXiv:2212.08073,

2022. 8.3.3

- [263] H. Lee, S. Phatale, H. Mansoor, T. Mesnard, J. Ferret, K. R. Lu, C. Bishop, E. Hall, V. Carbune, A. Rastogi et al., “Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback,” in ICML, 2024. 8.3.3
- [264] T. Yu, H. Zhang, Y. Yao, Y. Dang, D. Chen, X. Lu, G. Cui, T. He, Z. Liu, T.-S. Chua et al., “Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness,” arXiv:2405.17220, 2024. 8.3.3
- [265] Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. Pont-Tuset, S. Young, F. Yang et al., “Rich human feedback for text-to-image generation,” in CVPR, 2024, pp. 19401–19411. 8.3.3

