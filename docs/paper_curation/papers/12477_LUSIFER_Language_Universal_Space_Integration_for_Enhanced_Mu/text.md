# arXiv:2501.00874v3[cs.CL]7May2025

## LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models

Hieu Man

hieum@uoregon.edu University of Oregon Eugene, Oregon, USA

Nghia Trung Ngo

nghian@uoregon.edu University of Oregon Eugene, Oregon, USA

Viet Dac Lai

viet.lai@adobe.com Adobe Research San Jose, California, USA

Ryan A. Rossi

ryrossi@adobe.com Adobe Research San Jose, California, USA

Franck Dernoncourt

franck.dernoncourt@adobe.com Adobe Research San Jose, California, USA

Thien Huu Nguyen

thienn@uoregon.edu University of Oregon Eugene, Oregon, USA

### Abstract

Recent advancements in large language models (LLMs) based embedding models have established new state-of-the-art benchmarks for text embedding tasks, particularly in dense vector-based retrieval. However, these models predominantly focus on English, leaving multilingual embedding capabilities largely unexplored. To address this limitation, we present LUSIFER, a novel zero-shot approach that adapts LLM-based embedding models for multilingual tasks without requiring multilingual supervision. LUSIFER’s architecture combines a multilingual encoder, serving as a languageuniversal learner, with an LLM-based embedding model optimized for embedding-specific tasks. These components are seamlessly integrated through a minimal set of trainable parameters that act as a connector, effectively transferring the multilingual encoder’s language understanding capabilities to the specialized embedding model. Additionally, to comprehensively evaluate multilingual embedding performance, we introduce a new benchmark encompassing 5 primary embedding tasks, 123 diverse datasets, and coverage across 14 languages. Extensive experimental results demonstrate that LUSIFER significantly enhances the multilingual performance across various embedding tasks, particularly for medium and low-resource languages, without requiring explicit multilingual training data. The code and dataset for training are available at: https://github.com/hieum98/lusifer

### CCS Concepts

• Information systems → Language models.

### Keywords

Large Language Models, Representation Learning, Multilingual Embeddings

ACM Reference Format:

Hieu Man, Nghia Trung Ngo, Viet Dac Lai, Ryan A. Rossi, Franck Dernoncourt, and Thien Huu Nguyen. 2025. LUSIFER: Language Universal

This work is licensed under a Creative Commons Attribution 4.0 International License. SIGIR ’25, Padua, Italy

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1592-1/2025/07 https://doi.org/10.1145/3726302.3730029

Space Integration for Enhanced Multilingual Embeddings with Large Language Models. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, 15 pages. https: //doi.org/10.1145/3726302.3730029

### 1 Introduction

Text embeddings, which provide dense vector representations of textual content [13, 40], have become fundamental building blocks in modern natural language processing. These embeddings encode semantic information and serve as an important component for numerous downstream applications, ranging from information retrieval and document reranking to classification, clustering, and semantic textual similarity assessment. Recently, the significance of high-quality embeddings has been further amplified by their crucial role in retrieval-augmented generation (RAG) systems [27]. RAG architectures enable large language models (LLMs) to dynamically access and integrate external or proprietary knowledge without the need for model parameter updates, substantially enhancing their adaptability and accuracy [17, 33, 60].

The evolution of embedding models has witnessed remarkable advancements,progressingfromstatic word embeddings [52]through contextualized representations [16, 43, 50] to state-of-the-art LLMbased embedding models [62] that harness the sophisticated semantic understanding capabilities of large language models. These developments have substantially enhanced performance across various embedding tasks [35], achieving unprecedented accuracy in semantic similarity and retrieval applications. However, a critical limitation remains: the predominant focus on English in LLM-based embedding models has created a significant disparity in multilingual capabilities. This gap is especially pronounced in medium and low-resource languages, where English-centric models exhibit substantial performance degradation due to insufficient languagespecific training data [56, 65]. While recent advances in multilingual embedding models, particularly those leveraging multilingual pre-trained architectures, have demonstrated promising results in multilingual embedding tasks [10, 29, 63], their reliance on explicit multilingual supervision for embeddings constrains their applicability primarily to languages with abundant training resources, leaving the challenge of true language-agnostic representation largely unaddressed.

To address this challenge, we present LUSIFER, a novel zero-shot approach that adapts English LLM-based embedding models for multilingual tasks without requiring explicit multilingual supervision. Drawing inspiration from recent advances in multimodal integration [31, 34], LUSIFER employs a unique architecture that bridges the gap between multilingual understanding and specialized embedding capabilities. At its core, LUSIFER leverages the robust multilingual representations from XLM-R [11] and introduces a learnable connector mechanism to interface with English-optimized LLM embedding models. This approach enables LUSIFER to effectively transfer the multilingual understanding of XLM-R to the target LLM while inheriting advanced embedding capabilities of the LLM. In this way, LUSIFER can achieve effective multilingual representation capabilities without requiring explicit multilingual training data. We conduct comprehensive evaluations of LUSIFER through extensive experiments across 123 diverse datasets spanning 14 languages, focusing on five fundamental embedding tasks: Classification, Clustering, Reranking, Retrieval, and Semantic Textual Similarity (STS). Our experimental results demonstrate that LUSIFER substantially enhances the performance of English-centric LLMbased embedding models, achieving average improvements of 3.19 points across all tasks, with particularly significant gains observed for medium and low-resource languages (up to 22.15 improvement). To validate LUSIFER’s broader applicability and cross-lingual capabilities, we extend our evaluation to cross-lingual tasks using four comprehensive datasets that encompass over 100 languages, including several critically low-resource languages. LUSIFER significantly outperforms existing English-centric embedding models by 5.75 on average in cross-lingual scenarios. These results demonstrate the effectiveness of our approach in enhancing multilingual representation capabilities without explicit multilingual supervision.

The theoretical foundation for LUSIFER’s effectiveness lies in its ability to create a language-agnostic universal space through the integration of a multilingual encoder [30, 46]. We hypothesize that this universal space serves as a bridge between different languages, enabling the target language model to process semantic information independently of the input language. By mapping these language-neutral representations to the target model’s input space, we conjecture that the target LLM can grasp the semantics of these representations, thereby improving the quality of output embeddings across multiple languages. This mechanism allows the model to become less dependent on the specific language of the input, enabling it to better capture semantic information for embedding tasks in languages it rarely encountered during pretraining. Our empirical analysis using t-SNE visualization supports this hypothesis.

- 2 Related Work

- 2.1 English-centric Embedding Models

Text embedding models have experienced significant advancement in recent years, driven by the evolution of pre-trained language models. Early successes with BERT-based architectures, as demonstrated in Sentence-BERT [50], SimCSE [16], and DPR [22], established the foundation for modern embedding approaches. The field has since progressed to leverage LLMs, with recent works [8, 25, 37, 41, 62] demonstrating substantial improvements in embedding quality and

task performance through the enhanced representational capacity of LLMs [35]. However, these advances primarily benefit highresource language applications, as most state-of-the-art LLM-based embedding models are derived from English-centric foundation models [21, 58] and trained predominantly on English or highresource language datasets [62]. This bias has resulted in a significant performance gap between high-resource and low-resource languages, limiting the global applicability of these models. Our proposed method, LUSIFER, addresses this limitation by enabling effective multilingual representation without multilingual training data.

### 2.2 Zero-shot Multilingual Embedding

MultilingualEmbedding hasevolved through several distinct methodological approaches, each addressing the fundamental challenge of bridging language gaps in embedding tasks. Early successful approaches relied on translation models to enable multilingual understanding [32, 54, 71]. While effective, these methods introduced operational complexity by requiring external translation systems, limiting their practical deployment and scalability. The emergence of multilingual pre-trained language models, particularly XLMR [11], opened new possibilities for multilingual transfer. Recent works have demonstrated promising results by fine-tuning such models with contrastive learning objectives on multilingual data [10, 55, 63]. However, these approaches face two key limitations: they require substantial multilingual training data, and moreover, they do not exploit the sophisticated semantic representations afforded by contemporary English-centric LLM architectures, which have demonstrated superior performance in capturing nuanced semantic relationships.

Recent advances in aligning multilingual and English-centric representations could offer a solution. By combining independently pre-trained representations, a paradigm that has shown remarkable success in multimodal alignment research [3, 31, 34], these works bridge the gap between visual encoders and language models to enhance visual comprehension. As such, similar principles can be applied to align multilingual representations with LLM-based semantic spaces. While related efforts have explored aligning multiple LLMs for improved reasoning capabilities in multilingual settings [7, 70], these approaches primarily target generation tasks and typically require large-scale alignment data. Our work extends these efforts by focusing on embedding tasks and leveraging a minimal set of parameters to align multilingual and English-centric representations, enabling enhanced multilingual representation capabilities without requirement for large-scale multilingual training data.

### 2.3 Multilingual Embedding Benchmarks

The evaluation landscape for multilingual embedding models has historically been fragmented across various benchmarks, each with significant limitations. While existing benchmarks have made valuable contributions, they often exhibit constrained scope: MINERS [66] provides evaluation across multiple languages but is limited to classification and STS tasks with only 11 datasets; XNLI [12], XQuAD [4], and SIB-200 [1] offer broad language coverage but focus exclusively on classification tasks; and MTEB [42], despite

Stage 1: Alignment using English Data

Halign

Xinput

Stage 2: Representation

The Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858...

Finetuning using English Data

LConstrastive

Bi-directional Enabled LLM

LLM

LoRA

[Figure 1]

[Figure 2]

[Figure 3]

Halign

Connector (FF)

Connector (FF)

[Figure 4]

[Figure 5]

Multilingual Encoder

Multilingual Encoder

[Figure 6]

[Figure 7]

Given a question, retrieve passages that answer the question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?

Given a question, retrieve passages that answer the question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?

Xinput

Inference with Multilingual Data

Embx

LLM-based Embedding Model

Halign

Connector (FF)

Multilingual Encoder

Given a question, retrieve passages that answer the question: Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?

Xinput

Figure 1: Overview of LUSIFER. Left: Align a multilingual encoder with the target English-centric LLM only using English data and a minimal set of trainable parameter. Center: End-to-end representation finetune through contrastive learning on English text-embedding tasks using LoRA. Right: During inference, LUSIFER successfully processes text-embedding tasks across multiple languages.

its diverse task selection, primarily addresses high-resource languages. To address these limitations, we introduce a comprehensive evaluation framework that encompasses 5 fundamental embedding tasks—Classification, Clustering, Reranking, Retrieval, and STS—across an extensive collection of 123 datasets spanning 14 languages. This holistic approach enables systematic evaluation across both task and language dimensions, providing unprecedented insights into models’ multilingual capabilities. Furthermore, our benchmark extends beyond traditional multilingual evaluation by incorporating cross-lingual tasks, featuring coverage of over 100 languages, including critically low-resource languages that have been historically underrepresented in existing benchmarks. This extensive coverage allows for a more nuanced understanding of embedding models’ performance across the global linguistic landscape.

### 3 Methodology

Previous works demonstrate that representations of multilingual encoder models exhibit inherent language-agnostic properties, facilitating zero-shot multilingual transfer [30, 46]. Building upon this foundation, we propose LUSIFER, an embedding framework that aligns a multilingual encoder model with a target Englishcentric LLM’s representational space, enabling the target to encode semantics across multiple languages without extensive multilingual training. This section details our architectural design and two-stage training process for LUSIFER.

### 3.1 Model Architecture

The core development of LUSIFER lies in its novel approach to enabling multilingual encoding of target LLMs through efficient representation mapping. As illustrated in Figure 1, LUSIFER’s architecture consists of three key components: (1) a multilingual encoder that functions as a language-universal learner, capturing semantic information for diverse languages, (2) a language-agnostic connector that serves as a minimal parametric bridge between representations, and (3) a target LLM optimized for embedding-specific

tasks. The multilingual encoder processes input from various languages into a shared semantic space, while the connector, designed with minimal trainable parameters, aligns these universal representations with the target LLM’s native representational space. This alignment enables the target LLM embedding model to effectively leverage multilingual understanding without requiring extensive multilingual training data or architectural modifications.

Following successful approaches in multimodal alignment [3, 31, 34], we implement the connector as a 2-layers feed-forward network, FF, augmented with a single trainable token appended to the multilingual encoder’s hidden states. Formally, given input tokens X𝑖𝑛𝑝𝑢𝑡 (with necessary padding), the multilingual encoder’s hidden states H𝑒𝑛𝑐 are transformed to align with the target LLM’s representational space. The resulting aligned hidden states H𝑎𝑙𝑖𝑔𝑛 maintain dimensionality compatibility with the target LLM’s hidden states while extending the sequence length by one (|X𝑖𝑛𝑝𝑢𝑡 | + 1): H𝑎𝑙𝑖𝑔𝑛 = [FF(H𝑒𝑛𝑐);t], where FF is the feed-forward network to align the multilingual encoder’s hidden states with dimension d𝑒 to the target LLM’s hidden states with dimension d𝑡, and t ∈ R𝑑𝑡 is the trainable token. Moreover, we employ a masking mechanism to mask any original padding tokens in H𝑒𝑛𝑐 to prevent their influence on the target LLM’s processing, ensuring the model focuses on meaningful tokens.

### 3.2 Training Pipeline

LUSIFER employs a two-stage training process to achieve optimal multilingual representation capabilities. Both stages only require training on English data, leveraging the multilingual encoder’s inherent language-agnostic properties and embedding advantages of LLMs to facilitate zero-shot multilingual transfer.

Stage 1: Alignment Training. The initial training stage aligns the multilingual encoder’s representations with the target LLM’s embedding space. Specifically, we optimize the connector parameters 𝜃𝑐 and the multilingual encoder parameters 𝜃𝑒 while keeping the target LLM’s parameters fixed, ensuring stable convergence. The training employs two complementary objectives: (1) A masked reconstruction task where we randomly mask 𝑘% of input tokens

Classification

InappropriatenessClassification

AmazonCounterfactualClassification

ArEntail

XNLI

Clustering

StackExchangeClusteringP2P

StackExchangeClustering

GeoreviewClusteringP2P

TERRa

TweetSentimentExtractionClassification MultiEURLEXMultilabelClassification

RuSciBenchGRNTIClassification

RTE3

MasakhaNEWSClusteringS2S

RedditClustering RuSciBenchOECDClusteringP2P

MasakhaNEWSClusteringP2P

RuSciBenchOECDClassification

TwitterURLCorpus

BengaliDocumentClassification

KinopoiskClassification

indonli

SpanishNewsClusteringP2P

RuSciBenchGRNTIClusteringP2P

MedrxivClusteringP2P

SIB200ClusteringS2S

IndicNLPNewsClassification

FinToxicityClassification SIB200Classification

TeluguAndhraJyotiNewsClassification

MLSUMClusteringP2P

IndicReviewsClusteringP2P

MedrxivClusteringS2S

RedditClusteringP2P

KLUE-TC

FrenchBookReviews

IndonesianMongabayConservationClassification SpanishNewsClassification

TwitterSemEval2015

BiorxivClusteringS2S ArxivClusteringS2S

MLSUMClusteringS2S ArxivClusteringP2P

HALClusteringS2S

PawsXPairClassification

BengaliHateSpeechClassification

IndicSentimentClassification

MassiveScenarioClassification

TwentyNewsgroupsClustering

BiorxivClusteringP2P

KorHateSpeechMLClassification

MassiveIntentClassification

VieStudentFeedbackClassification

HeadlineClassification

Retrieval

AmazonPolarityClassification

Banking77Classification

MasakhaNEWSClassification

MultilingualSentimentClassification

WikipediaRetrievalMultilingual

QuoraRetrieval

CQADupstackTexRetrieval

Ko-StrategyQA

TweetEmotionClassification

SprintDuplicateQuestions SCIDOCS

AmazonReviewsClassification

TweetSentimentClassification

IndicQARetrieval

FEVER

MintakaRetrieval

ArguAna

MSMARCO

VieQuADRetrieval

KorSarcasmClassification

MTOPDomainClassification

EmotionClassification MTOPIntentClassification

FarsTail

RiaNewsRetrieval

FiQA2018

Touche2020

BelebeleRetrieval

ClimateFEVER

DBPedia

SciFact

NQ

RuReviewsClassification ImdbClassification

AfriSentiClassification

SentimentAnalysisHindi

GeoreviewClassification

MLQARetrieval

RuBQRetrieval

HotpotQA

NFCorpus

TRECCOVID

SwahiliNewsClassification

KLUESTS

STS

SummEvalFr

SummEval

KorSTS

STSBenchmarkMultilingualSTS

BIOSSES

Reranking

MindSmallReranking

AskUbuntuDupQuestions

WikipediaRerankingMultilingual

SciDocsRR

STS12

STS13 OpusparcusPC

STS13

STS14

STS15

STS16

STS17

STS22

FinParaSTS

StackOverflowDupQuestions

RuSTSBenchmarkSTS

STSBenchmark

IndicCrosslingualSTS

SemRel24STS

SICKFr

SICK-R

#### Figure 2: Overview of tasks and datasets in our benchmark. Crosslingual datasets are marked with a blue shade.

such that X𝑖𝑛𝑝𝑢𝑡 = mask(X,𝑘), training the model to recover the original sequence X𝑙𝑚 = X. (2) An autoregressive completion task that focuses on next-token prediction, where the model learns to generate the target sequence X𝑙𝑚 conditioned on the input context X𝑖𝑛𝑝𝑢𝑡. The training objective for both tasks is formulated as language modeling objective to generate the target sequence X𝑙𝑚 given the input sequence X𝑖𝑛𝑝𝑢𝑡. This objective enables local token-level alignment through masked reconstruction task where the model learns to predict the masked tokens by leveraging the context. In addition, it exploits global semantic alignment through autoregressive completion task that encourages the model to capture semantic information of the input sequences to generate the target sequence. As such, our training strategy learns to align the multilingual encoder’s representations with the target LLM’s embedding space while preserving important semantic information of multilingual input sequences. Our training process is conducted using the standard cross-entropy loss function. This stage aims to establish a strong alignment between the multilingual encoder and the target LLM, enabling the target LLM to effectively process multilingual representations produced by the multilingual encoder.

Stage 2: Representation Finetuning. The second stage improves text representations through a contrastive learning process, effectively teaching the model to distinguish between positive and negative examples. Our approach leverages both in-batch negatives sampled from the current training batch and hard-negative examples specifically curated to enhance model training. Additionally, we incorporate bidirectional attention mechanisms within the target LLM, following recent advances in LLM’s representation learning [8, 25, 37, 41]. This bidirectional context modeling significantly enhances the quality of learned representations by enabling the model to capture both forward and backward dependencies in the input sequence. During this stage, we finetune all components of LUSIFER, including the target LLM, the multilingual encoder,

and the connector parameters, to optimize the model’s representation quality for embedding-specific tasks. The goal of this stage is to improve the quality of text representations by leveraging the advanced embedding capabilities of the target LLM while maintaining the multilingual understanding provided by the multilingual encoder.

The two-stage training process enables LUSIFER to effectively align multilingual representations with the target LLM’s embedding space, enhancing the target LLM’s multilingual representation capabilities without requiring explicit multilingual supervision.

### 4 Experiment

This section presents our experimental methodology and results. We first introduce benchmark datasets and evaluation metrics in Section 4.1, followed by our experimental setup including model implementation, training data, and procedures in Section 4.2. We then present our main findings in Section 4.3 and analyze LUSIFER’s cross-lingual capabilities in Section 4.4. We examine detail performance of LUSIFER across five embedding tasks in Section 4.5 and evaluate LUSIFER’s component effectiveness in Section 4.6. Finally, Section 4.7 visualizes LUSIFER’s representations in multilingual space to demonstrate its language-agnostic capabilities.

### 4.1 Benchmark

Figure 2 illustrates the tasks and datasets in our benchmark. Following [42], our benchmark includes five fundamental embedding tasks, with the evaluation protocol for each task adapted from the respective original papers. The benchmark involves 123 diverse datasets, including 48 Classification datasets, 24 Clustering datasets, 24 Retrieval datasets, 22 Semantic Textual Similarity STS datasets, and 5 Reranking datasets. The main metrics for each task are as follows: Classification: Accuracy, Clustering: V-measure [53], Retrieval: nDCG@10, STS: Pearson correlation based on cosine

Baselines En Es Ru Fr Vi Fa Id Ar Fi Ko Hi Bn Te Sw Avg. Jina-embeddings-v3* [55] 59.84 61.23 62.88 58.94 66.74 78.35 58.51 64.71 73.57 64.96 64.19 61.54 68.96 49.20 63.83 mGTE-base* [73] 60.40 59.65 61.02 56.20 65.81 73.46 56.55 61.97 68.96 61.22 60.81 58.24 63.58 52.57 61.46 BGE-M3* [10] 60.09 60.60 62.37 57.34 70.69 78.97 58.78 64.12 75.60 64.72 64.61 65.31 69.85 54.20 64.80 Multilingual-E5-large* [64] 61.91 61.97 62.91 59.40 71.30 78.08 55.21 63.41 76.53 66.55 63.75 63.67 67.32 51.55 64.54 UDEVER-Bloom-7B* [72] 55.83 56.39 59.73 54.38 64.32 68.70 48.97 55.02 67.60 58.54 55.96 55.13 61.00 47.41 57.78 SimCSE [16] 51.92 51.81 24.90 46.95 31.18 37.12 39.27 29.46 41.64 26.23 25.17 21.54 26.71 38.36 35.16 Contriever [20] 49.29 44.26 26.55 44.05 33.03 39.66 38.33 32.36 45.76 26.47 23.27 22.61 22.64 39.26 34.82 GTE-large [29] 62.29 51.66 33.49 50.13 38.88 44.67 43.07 30.27 51.98 27.02 20.38 22.97 22.75 41.40 38.64 BGE-en-1.5 [68] 63.27 51.65 32.79 50.84 38.50 49.73 43.28 30.81 51.16 31.11 25.28 26.34 23.02 41.96 39.98 E5-large [61] 60.12 52.41 26.81 51.00 37.99 39.47 43.86 31.32 53.59 28.84 24.57 23.48 22.03 43.25 38.48 ST5-XXL [45] 58.81 60.35 44.42 58.50 41.81 24.66 53.43 25.30 52.46 15.43 18.07 17.10 21.63 38.81 37.91 GTR-XXL [44] 58.12 54.39 41.94 53.21 37.96 24.67 50.08 25.14 53.88 15.23 17.35 15.92 22.12 40.57 36.47 E5-Mistral [62] 66.64 61.84 61.30 59.65 58.58 72.55 58.25 54.43 66.97 62.82 56.23 55.10 47.15 50.61 59.44 LUSIFER (Ours) 57.20 60.14 59.82 59.24 67.69 76.17 59.70 55.60 72.83 65.23 62.37 58.43 69.30 53.12 62.63

#### Table 1: Comparative analysis of model performance across multiple languages and tasks. The table presents average metrics for each model, with the highest score for each language emphasized in bold. * denotes the models trained on extensive multilingual data.

Baselines MLQARetrieval BelebeleRetrieval STS17 STS22 IndicCrosslingual Avg. SimCSE [16] 7.41 18.35 39.71 37.95 0.18 20.72 Contriever [20] 9.75 22.94 34.55 41.72 0.03 21.80 GTE-large [29] 16.99 31.82 37.57 53.79 1.59 28.35 BGE-en-1.5 [68] 16.64 31.19 40.40 50.77 1.11 28.02 E5-large [61] 17.04 31.12 37.90 54.31 1.83 28.44 ST5-XXL [45] 20.82 41.68 56.19 59.02 1.76 35.89 GTR-XXL [44] 20.19 38.02 50.83 60.11 2.74 34.38 E5-Mistral [62] 31.54 54.75 81.12 71.37 21.92 52.14 LUSIFER (Ours) 36.68 57.81 81.09 70.49 43.40 57.89

#### Table 2: Cross-lingual evaluation results. The table presents average metrics for each model over all languages of the datasets, with the highest score for each language emphasized in bold.

similarity [49], and Reranking: MAP. Following [24], our benchmark covers 14 languages including 5 high-resource languages: English (en), Spanish (es), Russian (ru), French (fr), Vietnamese (vi); 6 medium-resource languages: Persian (fa), Indonesian (id), Arabic (ar), Finnish (fi), Korean (ko), Hindi (hi); 3 low-resource languages: Bengali (bn), Telugu (te), Swahili (sw).

Additionally, we evaluate models on cross-lingual retrieval tasks where the models need to perform text embedding tasks with queries and documents in different languages. These tasks feature 5 datasets, including Belebele [6], MLQA [26], STS17, STS22 [2], and IndicCrosslingualSTS [48], covering over 100 languages, including critically low-resource languages.

### 4.2 Experimental Setup

Implementation Details. LUSIFER encompasses three key components: a multilingual encoder, a connector, and a target LLM. We employ XLM-R-large [11] as the multilingual encoder, Mistral-7B

[21] as the English-centric target LLM, and a 2-layer feed-forward network with one trainable token as the connector. To facilitate efficient training, we leverage the LoRA framework [19] for training of LUSIFER’s components. Furthermore, we employ GradCache [15], gradient checkpointing, mixed precision training, and FSDP [74] to minimize GPU memory requirements. The LUSIFER architecture and its training code are built on top of the Hugging Face Transformers [67] and Pytorch Lightning libraries [14]. We detail the training hyper-parameters for each stage in Table 3

Training Data. We only train LUSIFER on a diverse public English datasets. For alignment training, we use the combination of the English Wikipedia and questions-answering datasets. Specifically, we use subset of Wikitext-103 [39] and MSMARCO [5] for the masked reconstruction and autoregressive completion tasks, respectively. For representation finetuning, we adopt the retrieval datasets as follows: MS MARCO [5], NQ [23], PAQ [28], HotpotQA [69], SNLI [9], SQuAD [47], ArguAna [59], FiQA [36] and FEVER [57]. To address the lack of hard negatives in these datasets, we

SimCSE

Contriever

SimCSE

GTR-XXL ST5-XXL GTE-large

Contriever

En

GTR-XXL ST5-XXL GTE-large

En

80.95

Sw

Es

BGE-en-1.5

46.66

E5-large

BGE-en-1.5

58.67

70.40

72.33

E5-Mistral

E5-large

Sw

Es

LUSIFER-8B

E5-Mistral

61.75

36.46

35.61

LUSIFER-8B

36.86

45.24

31.66

Te

Ru

52.04

80.53

62.87

27.58

27.75

60.60

22.59

43.70

Te

Ru

42.87

56.70

17.75

29.50

36.55

12.88 40.54

Bn

Fr

59.40

68.04

54.91

22.41

29.02

32.65

22.36

13.58

19.19

21.99

13.86

34.50

44.41

15.23

26.37 7.19

31.04

7.29 31.00

44.82

38.13

Bn

Fr

68.19

74.13

27.63

13.47

Hi

Vi

13.03

26.23

20.55

42.62

51.50

23.16

19.42

32.93 33.68

43.18

32.58

65.23

74.94

41.45

42.86

Ko

Fa

44.01

46.79

32.88

50.25

Hi

Vi

50.76

64.94

37.77

73.73 60.39

Fi

Id

56.84 47.50

66.46

Ar

Ko Id

(a) Classification tasks

(b) Clustering tasks

#### Figure 3: Performance comparison of LUSIFER and baseline models on Classification and Clustering tasks.

###### Hyperparameter Alignment Training Representation Finetuning

Batch size 256 256 Learning rate 1.5e-4 5e-5 Learning rate scheduler cosine cosine Learning rate warm-up ratio 0.1 0.1 Weight decay 0.01 0.01 Grad norm clipping 1.0 1.0 Epochs 2 1 Optimizer AdamW AdamW Float precision bf16-mixed bf16-mixed LoRA rank 16 16 LoRA alpha 32 32 Random mask ratio 0.5 Number of hardnegatives - 7

#### Table 3: Training hyperparameters for each stage.

Stage Dataset Number of Samples Alignment Training Wikitext-103 [39] 100,000

MSMARCO [5] 100,000

Representation Finetuning MS MARCO [5] 100,000 FEVER [57] 100,000 PAQ [28] 100,000 SNLI [9] 100,000

HotpotQA [69] 97,800 SQuAD [47] 97,400

FiQA [36] 6,420 NQ [23] 3,420 ArguAna [59] 1,280

Table 4: Number of samples used in each dataset for training. The number of negative samples is included in the total number of samples.

leverage an encoder-based model [61] to select the hard negatives on those datasets. Refer to Table 4 for the number of samples used in each dataset.

Baselines. We evaluate LUSIFER’s performance across the five fundamental embedding tasks on the benchmark datasets. We make comparisons with a variety of baseline models for embedding tasks which only trained/finetuned on mainly English data. Baselines include the following categories: dense retrieval models with Small Language Model (SLM) backbone: SimCSE [16], Contriever [20], GTE-large [29], BGE-en-1.5 [68], E5-large [61]; and dense retrieval models with Large Language Model (LLM) backbone: GTR-XXL [44], ST5-XXL [45], E5-Mistral [62]. Moreover, we include the following state-of-the-art multilingual embedding models which are trained on extensive multilingual data for reference: Jina-embeddings-v3 [55], mGTE-base [73], BGE-M3 [10], Multilingual-E5-large [64], and UDEVER-Bloom-7B [72].

### 4.3 Main Results

Table 1 presents a comprehensive evaluation of LUSIFER’s performance across 14 diverse languages, demonstrating its capabilities in multilingual representation learning. Our model achieves stateof-the-art performance in 10 out of 14 languages, with an average score of 62.63, surpassing the previous benchmark set by E5-Mistral (59.44) by 3.19 points. This improvement is particularly noteworthy given that E5-Mistral utilizes extensive proprietary synthetic data and multilingual training resources. The performance distribution across different language categories reveals LUSIFER’s robust multilingual capabilities. In high-resource languages, while maintaining competitive performance with established benchmarks, our model shows particular strength in medium and low-resource languages. The most striking improvement is observed in Telugu (te), where

SimCSE

Contriever

GTR-XXL ST5-XXL GTE-large

En

59.95

BGE-en-1.5

E5-large

E5-Mistral

LUSIFER-8B

49.31

41.52

Avg.

Fa

77.09

78.75

62.50

63.26

52.02

47.50

35.40

35.29

43.67

56.68

63.00

72.44

36.08

82.66

86.85

Bn

Fi

56.62

85.22

Hi

#### Figure 4: Performance comparison of LUSIFER and baseline models on Reranking tasks.

LUSIFER achieves a remarkable 22.15 points gain over E5-Mistral, underscoring its effectiveness in enhancing representation capabilities for traditionally underrepresented languages. When compared to existing embedding models with SLM backbones, LUSIFER demonstrates substantial improvements over models like E5-large (38.48) and BGE-en-1.5 (39.98) which are trained on English data only, thus further demonstrating the benefits of combining multilingual encoder and LLM’s English-centric for text-embedding tasks in multilingual settings.

Furthermore, LUSIFER achieves competitive performance (62.63) against state-of-the-art multilingual models such as BGE-M3 (64.80) and Multilingual-E5-large (64.54), despite these models requiring extensive multilingual training data. A key advantage of LUSIFER lies in its resource efficiency. While conventional approaches often rely heavily on extensive multilingual training data, our model achieves comparable or superior performance through its innovative alignment mechanism. This efficiency is particularly valuable in scenarios where multilingual training resources are limited or costly to obtain. The model’s ability to generalize effectively across languages while maintaining high performance demonstrates the robustness of our approach.

### 4.4 Cross-Lingual Evaluation

LUSIFER demonstrates cross-lingual capabilities across multiple languages and tasks, as evidenced by the evaluation results presented in Table 2. Our model achieves a state-of-the-art average score of 57.89, surpassing the previous benchmark set by E5-Mistral (52.14) by a significant margin of 5.75 points. The performance breakdown across different datasets reveals LUSIFER’s consistent superiority. In the MLQARetrieval, LUSIFER achieves a score of 36.68, demonstrating a 5.14 improvement over E5-Mistral’s 31.54. Similarly, on the BelebeleRetrieval dataset, LUSIFER’s score of 57.81 outperforms

E5-Mistral’s 54.75, showcasing its robust cross-lingual retrieval capabilities. For semantic similarity tasks, LUSIFER maintains competitive performance, achieving scores of 81.09 and 70.49 on STS17 and STS22 respectively, closely matching or slightly trailing E5Mistral’s performance. Perhaps the most striking achievement of LUSIFER is its performance on low-resource languages, particularly evident in the IndicCrosslingual dataset. Here, LUSIFER achieves an unprecedented score of 43.40, nearly doubling E5-Mistral’s performance of 21.92. This remarkable improvement demonstrates LUSIFER’s ability to effectively transfer semantic knowledge across language boundaries, particularly benefiting languages with limited resources. These results underscore LUSIFER’s effectiveness in enhancing cross-lingual capabilities through efficient multilingual representation alignment, enabling the model to process textembedding tasks across multiple languages effectively.

### 4.5 Task-Specific Performance

Figure 3, 4, 5 present the performance comparison of LUSIFER and baseline models on Classification, Clustering, Reranking, Retrieval, and STS tasks. LUSIFER consistently outperforms the baseline models across 4 out of 5 tasks, with the largest improvements observed in Clustering and Retrieval tasks, especially in the medium and low-resource languages. However, the performance of LUSIFER in the Reranking tasks is slightly worse than the strongest baseline, E5-Mistral model. This discrepancy may be attributed to the task’s complexity and the information loss in the alignment process between the multilingual encoder and the target LLM. Nevertheless, LUSIFER’s strong performance across a variety of tasks and languages highlights its ability to enhance multilingual representations without relying on explicit multilingual training data.

### 4.6 Ablation Study

To comprehensively evaluate LUSIFER’s architectural design and training methodology, we conducted an extensive ablation study examining the contribution of each major component. We compared LUSIFER against several ablated variants:

- (1) Connector-Only: A simplified version using only the finetuning connector during both alignment training and representation finetuning stages, removing the complex interaction between components.
- (2) Frozen Multilingual Encoder: A variant where the multilingual encoder is freeze throughout training, with only the connector trained during alignment and both connector and target LLM trained during representation finetuning.
- (3) Alignment Only: The model trained solely with alignment training, omitting the representation finetuning stage to assess the importance of the two-stage training process.
- (4) Representation Finetuning Only: A version trained using only representation finetuning without initial alignment training.

We evaluated these variants across our standard benchmark suite, measuring performance on cross-lingual transfer, semantic similarity, and downstream task effectiveness. Table 5 presents the comprehensive results of this analysis. The full LUSIFER model achieved the highest average score of 62.63, significantly outperforming all ablated versions. Breaking down the performance impacts: (i) The

SimCSE

SimCSE

Contriever

Contriever

GTR-XXL ST5-XXL GTE-large

GTR-XXL ST5-XXL GTE-large

En

En

53.45

81.10

BGE-en-1.5

Te

Es

BGE-en-1.5

E5-large

E5-large

Te

Es

E5-Mistral

68.97

66.34

47.04

E5-Mistral

LUSIFER-8B

62.38

81.09

LUSIFER-8B

65.50

54.77

37.15

66.38

Bn

Ru

45.82

57.60

65.43

54.72

29.07

40.59

38.06 46.55

35.58

Bn

Ru

41.86

71.56

30.41

35.42

55.10

20.50

23.70

41.23

20.29

32.83

17.85

14.59

Hi

Fr

12.45

54.52

68.63

54.91

6.61

7.20

34.20

7.02

6.87

1.34

2.21

4.33

1.00

10.92

4.76

4.64

13.67

25.52

12.67

61.39

57.72

74.89

4.77

68.16

83.77

28.73

Hi

Fr

24.69

27.88

43.99

36.54

76.67

61.77

25.98

Ko

Vi

36.58

50.27

52.23

34.81

60.76

43.84

56.17

67.77 22.73

72.89

81.24

52.11

40.88

Ko

Id

84.06

78.49

73.61

Fi

Fa

50.35

32.24 87.56

56.19 68.07

Fi Ar

Ar Id

(a) Retrieval tasks

(b) STS tasks

#### Figure 5: Performance comparison of LUSIFER and baseline models on Retrieval and STS tasks.

Baselines En Es Ru Fr Vi Fa Id Ar Fi Ko Hi Bn Te Sw Avg. LUSIFER (Full) 57.20 60.14 59.82 59.24 67.69 76.17 59.70 55.60 72.83 65.23 62.37 58.43 69.30 53.12 62.63 LUSIFER (Connector Only) 35.53 33.98 42.95 33.54 35.68 57.86 35.55 27.60 48.72 34.45 47.57 41.85 46.50 34.66 44.18 LUSIFER (Frozen Multilingual Encoder) 50.99 58.77 58.30 52.73 62.24 75.88 58.11 41.66 70.75 59.53 62.48 55.53 66.24 49.12 58.74 LUSIFER (Alignment Only) 43.32 38.94 45.12 36.75 41.96 64.60 38.38 33.07 52.78 38.08 53.06 47.84 48.34 40.03 44.45 LUSIFER (Representation Finetuning Only) 49.71 58.76 58.08 51.01 62.11 74.01 57.32 40.95 68.47 57.81 59.74 53.53 63.39 47.03 57.28

#### Table 5: Ablation study results of LUSIFER’s components. The table presents average metrics for each model, with the highest score for each language emphasized in bold.

Connector-Only variant (44.18) showed a 18.45 point, highlighting the importance of allowing the multilingual encoder and the target LLM to be finetuned during training. (ii) The Frozen Multilingual Encoder variant (56.74) performed better than Connector-Only but still fell short by 3.89, demonstrating the value of end-to-end training. (iii) Alignment-Only (44.45) and Representation Finetuning Only (57.28) variants both showed substantial degradation, with drops of 18.18 and 5.35 respectively, indicating the complementary nature of our two-stage training approach, especially the importance of the representation finetuning stage. Through this ablation study, we can attributed that the alignment training stage proves crucial for establishing initial cross-lingual connections, while the representation finetuning stage refines these alignments for specific downstream tasks.

### 4.7 Model Representation Visualization

Figure 6 shows 2D scatter plots of representations from different models for 200 randomly sampled examples from the SIB200 dataset, visualized using t-SNE. The points are colored by the language of the samples. The t-SNE representation of E5-Mistral demonstrates a clearer separation between languages, with distinct clusters for each language. In contrast, the visualization of LUSIFER presents a

[Figure 8]

[Figure 9]

(a) E5-Mistral (language) (b) LUSIFER (language)

#### Figure 6: t-SNE representation of 200 randomly samples from the SIB200 dataset. The points are colored by the languages.

more mixed distribution of languages, with overlapping clusters across different languages. This observation provides insights into LUSIFER’s lingual-agnostic capabilities, highlighting the model’s ability to bridge the gaps between representation spaces of different languages. These results suggest that LUSIFER’s alignment strategy enables the model to comprehend semantics across multiple languages effectively, facilitating zero-shot multilingual transfer. Overall, our experiments confirm the advantages of the representation

alignment strategies in LUSIFER in effectively enabling zero-shot multilingual transfer for LLM-based embedding methods.

### 5 Conclusion

In this work, we propose LUSIFER, a novel framework that enables effective multilingual representation without explicit multilingual training data. LUSIFER aligns a multilingual encoder with a target English-centric LLM through a minimal set of trainable parameters, facilitating zero-shot multilingual transfer. Our experimental results demonstrate that LUSIFER achieves state-of-the-art performance across diverse languages and tasks, outperforming existing baseline models. Moreover, LUSIFER significantly enhances cross-lingual capabilities, enabling the model to process text-embedding tasks across multiple languages effectively.

Our work provides a promising direction for enhancing multilingual representation capabilities in English-centric embedding models, enabling global applicability without requiring extensive multilingual training data. The alignment strategies employed in LUSIFER ensure that the model can comprehend and process semantic information across different languages, making it a versatile tool for various multilingual applications. Furthermore, the ability of LUSIFER to perform zero-shot multilingual transfer opens up new possibilities for natural language processing tasks in lowresource languages, where obtaining large-scale annotated data is often challenging. By leveraging the strengths of both multilingual encoders and English-centric LLMs, LUSIFER bridges the gap between languages, fostering better multilingual representation learning.

In future work, we plan to explore additional alignment strategies and further investigate the impact of LUSIFER’s components on multilingual representation quality. We also aim to extend our framework to support different modalities such as images and audio and evaluate its performance on a wider range of tasks. Additionally, we will explore the integration of LUSIFER with other state-of-the-art models to further enhance its capabilities. Moreover, we anticipate that LUSIFER will facilitate broader applications of LLM embeddings in downstream tasks, ranging from deep context understanding requirements like sentiment analysis [18] to text style comprehension tasks such as authorship attribution [38, 51], thereby contributing to the advancement of natural language processing and information retrieval fields.

### Acknowledgements

This research has been supported by the NSF grant # 2239570. This research is also supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract 202222072200003. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes, notwithstanding any copyright annotation therein.

### References

[1] David Ifeoluwa Adelani, Hannah Liu, Xiaoyu Shen, Nikita Vassilyev, Jesujoba O. Alabi, Yanke Mao, Haonan Gao, and Annie En-Shiun Lee. 2024. SIB-200: A Simple,

Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects. arXiv:2309.07445 [cs.CL] https://arxiv.org/abs/2309.07445

- [2] Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2016. SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and Cross-Lingual Evaluation. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval2016), Steven Bethard, Marine Carpuat, Daniel Cer, David Jurgens, Preslav Nakov, and Torsten Zesch (Eds.). Association for Computational Linguistics, San Diego, California, 497–511. doi:10.18653/v1/S16-1081
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bińkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan.

2022. Flamingo: a Visual Language Model for Few-Shot Learning. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 23716–23736. https://proceedings.neurips.cc/paper_files/paper/2022/file/ 960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf

- [4] Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the Cross-lingual Transferability of Monolingual Representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 4623–4637. doi:10.18653/v1/2020.acl-main.421
- [5] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs.CL] https://arxiv.org/abs/1611.09268
- [6] Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 749–775. doi:10.18653/v1/2024.acl-long.44
- [7] Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, and Partha Talukdar. 2024. LLM Augmented LLMs: Expanding Capabilities through Composition. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id= jjA4O1vJRz
- [8] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. arXiv:2404.05961 [cs.CL] https://arxiv.org/abs/ 2404.05961
- [9] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, Lluís Màrquez, Chris Callison-Burch, and Jian Su (Eds.). Association for Computational Linguistics, Lisbon, Portugal, 632–642. doi:10.18653/v1/D15-1075
- [10] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.

2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]

- [11] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learning at Scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 8440–8451. doi:10.18653/v1/2020.acl-main.747
- [12] Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating Crosslingual Sentence Representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2475–2485. doi:10.18653/v1/D18-1269
- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. doi:10.18653/ v1/N19-1423
- [14] William Falcon and The PyTorch Lightning team. 2024. PyTorch Lightning. doi:10. 5281/zenodo.13254264
- [15] Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. 2021. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021), Anna

- Rogers, Iacer Calixto, Ivan Vulić, Naomi Saphra, Nora Kassner, Oana-Maria Camburu, Trapit Bansal, and Vered Shwartz (Eds.). Association for Computational Linguistics, Online, 316–321. doi:10.18653/v1/2021.repl4nlp-1.31
- [16] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 6894–6910. doi:10.18653/v1/2021.emnlp-main.552
- [17] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL] https://arxiv.org/abs/2312.10997
- [18] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM based system. arXiv:2409.09989 [cs.CL] https://arxiv.org/abs/2409.09989
- [19] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations. https: //openreview.net/forum?id=nZeVKeeFYf9
- [20] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. arXiv:2112.09118 [cs.IR] https://arxiv.org/abs/2112.09118
- [21] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https: //arxiv.org/abs/2310.06825
- [22] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for OpenDomain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 6769–6781. doi:10.18653/v1/2020.emnlp-main.550
- [23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452–466. doi:10.1162/tacl_a_00276
- [24] Viet Dac Lai, Nghia Ngo, Amir Pouran Ben Veyseh, Hieu Man, Franck Dernoncourt, Trung Bui, and Thien Huu Nguyen. 2023. ChatGPT Beyond English: Towards a Comprehensive Evaluation of Large Language Models in Multilingual Learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 13171–13189. doi:10.18653/v1/2023.findings-emnlp.878
- [25] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. arXiv:2405.17428 [cs.CL] https: //arxiv.org/abs/2405.17428
- [26] Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk.

2020. MLQA: Evaluating Cross-lingual Extractive Question Answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 7315–7330. doi:10.18653/v1/2020.aclmain.653

- [27] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/ paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
- [28] Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler, Aleksandra Piktus, Pontus Stenetorp, and Sebastian Riedel. 2021. PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them. Transactions of the Association for Computational Linguistics 9 (2021), 1098–1115. doi:10.1162/ tacl_a_00415
- [29] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281 [cs.CL] https://arxiv.org/abs/2308.03281
- [30] Jindřich Libovický, Rudolf Rosa, and Alexander Fraser. 2020. On the Language Neutrality of Pre-trained Multilingual Representations. In Findings of the Association for Computational Linguistics: EMNLP 2020, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 1663–1674. doi:10.18653/v1/2020.findings-emnlp.150

- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems 36 (2024).
- [32] Jiapeng Liu, Xiao Zhang, Dan Goldwasser, and Xiao Wang. 2020. Cross-Lingual Document Retrieval with Smooth Learning. In Proceedings of the 28th International Conference on Computational Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International Committee on Computational Linguistics, Barcelona, Spain (Online), 3616–3629. doi:10.18653/v1/2020.coling-main.323
- [33] Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu Lee, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Chatqa: Surpassing gpt-4 on conversational qa and rag. arXiv preprint arXiv:2401.10225 (2024).
- [34] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. 2024. Ovis: Structural Embedding Alignment for Multimodal Large Language Model. arXiv:2405.20797 [cs.CV] https://arxiv.org/abs/2405.20797
- [35] Kun Luo, Minghao Qin, Zheng Liu, Shitao Xiao, Jun Zhao, and Kang Liu. 2024. Large Language Models as Foundations for Next-Gen Dense Retrieval: A Comprehensive Empirical Assessment. arXiv:2408.12194 [cs.CL] https://arxiv.org/ abs/2408.12194
- [36] Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. WWW’18 Open Challenge: Financial Opinion Mining and Question Answering. In Companion Proceedings of the The Web Conference 2018 (Lyon, France) (WWW ’18). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 1941–1942. doi:10.1145/3184558.3192301
- [37] Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, and Thien Huu Nguyen. 2024. ULLME: A Unified Framework for Large Language Model Embeddings with Generation-Augmented Learning. arXiv:2408.03402 [cs.CL] https://arxiv.org/ abs/2408.03402
- [38] Hieu Man and Thien Huu Nguyen. 2024. Counterfactual Augmentation for Robust Authorship Representation Learning. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (Washington DC, USA) (SIGIR ’24). Association for Computing Machinery, New York, NY, USA, 2347–2351. doi:10.1145/3626772.3657956
- [39] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer Sentinel Mixture Models. In International Conference on Learning Representations. https://openreview.net/forum?id=Byj72udxe
- [40] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality. arXiv:1310.4546 [cs.CL] https://arxiv.org/abs/1310.4546
- [41] Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2024. Generative Representational Instruction Tuning. arXiv:2402.09906 [cs.CL] https://arxiv.org/abs/2402.09906
- [42] Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023. MTEB: Massive Text Embedding Benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, Andreas Vlachos and Isabelle Augenstein (Eds.). Association for Computational Linguistics, Dubrovnik, Croatia, 2014–2037. doi:10.18653/v1/2023.eacl-main.148
- [43] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y Zhao, Yi Luan, Keith B Hall, Ming-Wei Chang, et al. 2021. Large dual encoders are generalizable retrievers. arXiv preprint arXiv:2112.07899 (2021).
- [44] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y. Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2021. Large Dual Encoders Are Generalizable Retrievers. arXiv:2112.07899 [cs.IR] https://arxiv.org/abs/2112.07899
- [45] Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Cer, and Yinfei Yang. 2021. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models. arXiv:2108.08877 [cs.CL] https://arxiv.org/abs/ 2108.08877
- [46] Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How Multilingual is Multilingual BERT?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Anna Korhonen, David Traum, and Lluís Màrquez (Eds.). Association for Computational Linguistics, Florence, Italy, 4996–5001. doi:10.18653/v1/P19-1493
- [47] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Jian Su, Kevin Duh, and Xavier Carreras (Eds.). Association for Computational Linguistics, Austin, Texas, 2383–2392. doi:10.18653/v1/D16-1264
- [48] Gowtham Ramesh, Sumanth Doddapaneni, Aravinth Bheemaraj, Mayank Jobanputra, Raghavan AK, Ajitesh Sharma, Sujit Sahoo, Harshita Diddee, Mahalakshmi J, Divyanshu Kakwani, Navneet Kumar, Aswin Pradeep, Srihari Nagaraj, Deepak Kumar, Vivek Raghavan, Anoop Kunchukuttan, Pratyush Kumar, and Mitesh Shantadevi Khapra. 2022. Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages. Trans. Assoc. Comput. Linguistics 10 (2022), 145–162. doi:10.1162/TACL_A_00452
- [49] Nils Reimers, Philip Beyer, and Iryna Gurevych. 2016. Task-Oriented Intrinsic Evaluation of Semantic Textual Similarity. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, Yuji Matsumoto and Rashmi Prasad (Eds.). The COLING 2016 Organizing Committee,

- Osaka, Japan, 87–96. https://aclanthology.org/C16-1009
- [50] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 3982–3992. doi:10.18653/v1/D19-1410
- [51] Rafael A. Rivera-Soto, Olivia Elizabeth Miano, Juanita Ordonez, Barry Y. Chen, Aleem Khan, Marcus Bishop, and Nicholas Andrews. 2021. Learning Universal Authorship Representations. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 913–919. doi:10.18653/v1/2021.emnlp-main.70
- [52] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval 3, 4 (2009), 333–389.
- [53] Andrew Rosenberg and Julia Hirschberg. 2007. V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure. In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), Jason Eisner (Ed.). Association for Computational Linguistics, Prague, Czech Republic, 410–420. https://aclanthology.org/D07-1043
- [54] Peng Shi, Rui Zhang, He Bai, and Jimmy Lin. 2021. Cross-Lingual Training of Dense Retrievers for Document Retrieval. In Proceedings of the 1st Workshop on Multilingual Representation Learning, Duygu Ataman, Alexandra Birch, Alexis Conneau, Orhan Firat, Sebastian Ruder, and Gozde Gul Sahin (Eds.). Association for Computational Linguistics, Punta Cana, Dominican Republic, 251–253. doi:10. 18653/v1/2021.mrl-1.24
- [55] Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, and Han Xiao. 2024. jina-embeddings-v3: Multilingual Embeddings With Task LoRA. arXiv:2409.10173 [cs.CL] https://arxiv.org/abs/2409.10173
- [56] Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, and Daniel Cer. 2024. Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval. arXiv:2311.05800 [cs.IR] https://arxiv.org/abs/2311.05800
- [57] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.

2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 809–819. doi:10.18653/v1/ N18-1074

- [58] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971 [cs.CL] https://arxiv.org/abs/2302.13971
- [59] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the Best Counterargument without Prior Topic Knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Iryna Gurevych and Yusuke Miyao (Eds.). Association for Computational Linguistics, Melbourne, Australia, 241–251. doi:10.18653/v1/P18-1023
- [60] Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Instructretro: Instruction tuning post retrievalaugmented pretraining. arXiv preprint arXiv:2310.07713 (2023).
- [61] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 [cs.CL] https://arxiv.org/abs/2212. 03533
- [62] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving Text Embeddings with Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek

- Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 11897–11916. doi:10.18653/v1/2024.acl-long.642
- [63] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672 [cs.CL] https://arxiv.org/abs/2402.05672
- [64] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv preprint arXiv:2402.05672 (2024).
- [65] Zihan Wang, Karthikeyan K, Stephen Mayhew, and Dan Roth. 2020. Extending Multilingual BERT to Low-Resource Languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, Trevor Cohn, Yulan He, and Yang

Liu (Eds.). Association for Computational Linguistics, Online, 2649–2656. doi:10. 18653/v1/2020.findings-emnlp.240

- [66] Genta Indra Winata, Ruochen Zhang, and David Ifeoluwa Adelani. 2024. MINERS: Multilingual Language Models as Semantic Retrievers. arXiv:2406.07424 [cs.CL] https://arxiv.org/abs/2406.07424
- [67] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Qun Liu and David Schlangen (Eds.). Association for Computational Linguistics, Online, 38–45. doi:10.18653/v1/2020. emnlp-demos.6
- [68] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-Pack: Packaged Resources To Advance General Chinese Embedding. arXiv:2309.07597 [cs.CL]
- [69] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D181259
- [70] Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, and Minjoon Seo. 2024. LangBridge: Multilingual Reasoning Without Multilingual Supervision. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 7502–7522. doi:10.18653/v1/2024.acl-long.405
- [71] Bryan Zhang and Amita Misra. 2022. Machine translation impact in E-commerce multilingual search. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, Yunyao Li and Angeliki Lazaridou (Eds.). Association for Computational Linguistics, Abu Dhabi, UAE, 99–109. doi:10. 18653/v1/2022.emnlp-industry.8
- [72] Xin Zhang, Zehan Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, and Min Zhang. 2023. Language Models are Universal Embedders. arXiv preprint arXiv:2310.08232 (2023).
- [73] Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval. arXiv:2407.19669 [cs.CL] https://arxiv.org/abs/2407.19669
- [74] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. arXiv:2304.11277 [cs.DC] https://arxiv.org/abs/2304.11277

### A Detailed Results

In this section, we provide detailed results of LUSIFER and E5Mistral on all benchmark datasets for each language.

En Datasets E5-Mistral LUSIFER AmazonCounterfactualClassification 78.69 72.45 AmazonPolarityClassification 95.91 94.3 AmazonReviewsClassification 55.79 55.46 Banking77Classification 88.23 87.33 EmotionClassification 49.77 74 ImdbClassification 94.78 92.52 MassiveIntentClassification 80.57 75.64 MassiveScenarioClassification 82.39 78 MTOPDomainClassification 96.12 96.81 MTOPIntentClassification 86.11 87.34 ToxicConversationsClassification 69.59 82.84 TweetSentimentExtractionClassification 63.72 72.74 SprintDuplicateQuestions 95.66 90.99 TwitterSemEval2015 81.62 68.49 TwitterURLCorpus 87.75 85.35 ArxivClusteringP2P 50.45 35.6 ArxivClusteringS2S 45.5 22.25 BiorxivClusteringP2P 43.53 39.93 BiorxivClusteringS2S 40.24 29.3 MedrxivClusteringP2P 38.19 41.2 MedrxivClusteringS2S 37.45 35.53 RedditClustering 57.71 39.94 RedditClusteringP2P 66.49 53.4 StackExchangeClustering 73.1 46.41 StackExchangeClusteringP2P 45.91 39.7 TwentyNewsgroupsClustering 54.31 38.5 AskUbuntuDupQuestions 66.98 60.56 MindSmallReranking 32.6 24.55 SciDocsRR 86.33 34.94 StackOverflowDupQuestions 54.91 46.04 ArguAna 61.88 74.15 ClimateFEVER 38.4 29.24 CQADupstackTexRetrieval 42.97 23.22 DBPedia 48.9 17.98 FEVER 87.8 82.77 FiQA2018 56.62 14.91 HotpotQA 75.7 49.04 MSMARCO 43.1 56.43 NFCorpus 38.59 5.48 NQ 63.5 42.95 QuoraRetrieval 89.62 89.1 SCIDOCS 16.27 5.53 SciFact 76.41 66.09 Touche2020 26.39 6.33 TRECCOVID 87.33 18.22

- STS12 79.65 74.26

- STS13 88.43 84.2

- STS14 84.54 77.5

- STS15 90.42 84.95

- STS16 87.68 82.21

- STS17 91.75 81.67 STS22 67.28 71.25 BIOSSES 82.64 84.22 SICK-R 80.76 78 STSBenchmark 88.6 84.18 SummEval 31.4 32.36 Avg. 67.69 57.20

Table 7: Detailed results of E5-Mistral and LUSIFER on the English benchmark datasets.

Ru Datasets E5-Mistral LUSIFER

GeoreviewClassification 46.92 43.79 HeadlineClassification 76.52 79.26 InappropriatenessClassification 59.35 63.15 KinopoiskClassification 60.67 60.57 MassiveIntentClassification 72.06 71.29 MassiveScenarioClassification 76.64 74.49 RuReviewsClassification 64.10 67.40 RuSciBenchGRNTIClassification 60.19 59.51 RuSciBenchOECDClassification 46.30 46.41 GeoreviewClusteringP2P 69.87 59.20 RuSciBenchGRNTIClusteringP2P 52.96 55.00 RuSciBenchOECDClusteringP2P 46.54 49.95 TERRa 57.45 54.24 RiaNewsRetrieval 71.39 49.61 RuBQRetrieval 38.04 43.48 RuSTSBenchmarkSTS 81.79 78.20 STS22 61.32 61.44

Avg. 61.30 59.82

Table 8: Detailed results of E5-Mistral and LUSIFER on the Russian benchmark datasets.

Fa Datasets E5-Mistral LUSIFER

MassiveScenarioClassification 76.37 77.94 MassiveIntentClassification 71.98 73.32 MultilingualSentimentClassification 80.07 80.54 FarsTail 63.49 67.98 WikipediaRerankingMultilingual 75.60 78.75 WikipediaRetrievalMultilingual 67.77 78.49

Avg. 72.55 76.17

- Table 11: Detailed results of E5-Mistral and LUSIFER on the Farsi benchmark datasets.

Id Datasets E5-Mistral LUSIFER

IndonesianMongabayConservationClassification 24.72 25.27 MassiveIntentClassification 69.51 71.38 MassiveScenarioClassification 72.89 74.62 SIB200Classification 80.88 80.44 indonli 50.00 50.22 SIB200ClusteringS2S 46.46 47.50 BelebeleRetrieval 81.10 87.56 SemRel24STS 40.40 40.57

Avg. 58.25 59.70

- Table 12: Detailed results of E5-Mistral and LUSIFER on the Indonesian benchmark datasets.

Ar Datasets E5-Mistral LUSIFER

TweetEmotionClassification 53.74 49.03 ArEntail 77.63 84.15 XNLI 68.00 58.58 MintakaRetrieval 17.15 16.59 MLQARetrieval 28.32 47.90 STS17 75.13 71.44 STS22 61.01 61.54

Avg. 54.43 55.60

- Table 13: Detailed results of E5-Mistral and LUSIFER on the Arabic benchmark datasets.

Fi Datasets E5-Mistral LUSIFER

FinToxicityClassification 53.78 62.23 MassiveIntentClassification 64.15 70.77 MassiveScenarioClassification 67.79 75.02 MultilingualSentimentClassification 72.42 83.59 SIB200Classification 66.57 77.06 WikipediaRerankingMultilingual 86.85 82.65 BelebeleRetrieval 73.89 85.18 WikipediaRetrievalMultilingual 71.90 82.94 OpusparcusPC 91.41 91.63 FinParaSTS 20.97 17.24

Avg. 66.97 72.83

- Table 14: Detailed results of E5-Mistral and LUSIFER on the Finnish benchmark datasets.

Ko Datasets E5-Mistral LUSIFER

MassiveIntentClassification 70.42 69.79 MassiveScenarioClassification 75.12 75.60 KorSarcasmClassification 57.64 55.28 SIB200Classification 72.70 77.89 KorHateSpeechMLClassification 8.49 7.54 PawsXPairClassification 53.10 54.97 KLUE-TC 60.58 63.95 SIB200ClusteringS2S 31.04 46.58 Ko-StrategyQA 63.81 68.66 BelebeleRetrieval 80.09 84.69 KLUE-STS 83.48 84.17 KorSTS 79.28 78.36 STS17 80.97 80.55

Avg. 62.82 65.23

- Table 15: Detailed results of E5-Mistral and LUSIFER on the Korean benchmark datasets.

Hi Datasets E5-Mistral LUSIFER

MTOPIntentClassification 68.84 79.93 SentimentAnalysisHindi 58.98 73.92 MassiveIntentClassification 64.69 71.01 MassiveScenarioClassification 69.71 75.42 SIB200Classification 68.43 75.98 TweetSentimentClassification 37.70 40.78 XNLI 65.04 60.26 IndicReviewsClusteringP2P 40.04 42.40 SIB200ClusteringS2S 27.32 45.62 WikipediaRerankingMultilingual 85.22 78.17 BelebeleRetrieval 69.73 66.76 MintakaRetrieval 18.60 21.53 MLQARetrieval 35.37 54.54 WikipediaRetrievalMultilingual 74.62 75.25 IndicCrosslingualSTS 42.30 58.97 SemRel24STS 73.14 77.34

Avg. 56.23 62.37

- Table 16: Detailed results of E5-Mistral and LUSIFER on the Hindi benchmark datasets.

