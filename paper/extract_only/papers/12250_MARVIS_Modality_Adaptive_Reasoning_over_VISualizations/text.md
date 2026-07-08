# arXiv:2507.01544v2[cs.LG]29Apr2026

## MARVIS: Modality Adaptive Reasoning over VISualizations

Benjamin Feuer

Stanford University USA bfeuer@stanford.edu

Lennart Purucker

Prior Labs Germany puruckel@cs.uni-freiburg.de

Oussama Elachqar

Oumi USA oussama@oumi.ai

#### Abstract

Predictive applications of machine learning often rely on small (sub 1 Bn parameter) specialized models tuned to particular domains or modalities. Such models often achieve excellent performance, but lack flexibility. LLMs and VLMs offer versatility, but typically underperform specialized predictors, especially on nontraditional modalities and long-tail domains. We propose MARVIS (Modality Adaptive Reasoning over VISualizations), a system that transforms latent embedding spaces into visual representations and then leverages the spatial and fine-grained reasoning skills of VLMs to interpret the visualizations and utilize them for predictions successfully. MARVIS achieves competitive performance across vision, audio, biological, and tabular domains using a single 3B parameter model, yielding results that beat Gemini 2.0 by 16% on average. MARVIS drastically reduces the gap between LLM/VLMs approaches and specialized domain-specific methods, without requiring any domain-specific training. Code and datasets are available at https://github.com/penfever/marvis.

#### CCS Concepts

• Computing methodologies → Machine learning; Classification and regression trees; Computer vision.

#### Keywords

vision-language models, cross-modal reasoning, embedding visualization, multi-modal learning

ACM Reference Format:

Benjamin Feuer, Lennart Purucker, Oussama Elachqar, and Chinmay Hegde. 2026. MARVIS: Modality Adaptive Reasoning over VISualizations. In ACM Conference on AI and Agentic Systems (CAIS ’26), May 26–29, 2026, San Jose, CA, USA. ACM, New York, NY, USA, 37 pages. https://doi.org/10.1145/ 3786335.3813150

#### 1 Introduction

Much of the progress in the field of machine learning in recent years has been on classification and regression tasks (which, in this work, we sometimes collectively refer to as predictive tasks). These

This work is licensed under a Creative Commons Attribution 4.0 International License. CAIS ’26, San Jose, CA, USA

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2415-2/2026/05 https://doi.org/10.1145/3786335.3813150

Chinmay Hegde

New York University USA chinmay.h@nyu.edu

have historically been addressed either using classical machine learning methods or, more recently, with deep learning. In the latter case, the best performance has generally been achieved using specialized models with less than one billion parameters tuned for a particular task and/or knowledge domain [16, 19, 40]. These models often learn to compress a high-dimensional input space into a simplified embedded space; in some cases, these embeddings can be used for prediction without any fine-tuned classification stage via classical nonparametric methods like KNN [36], but in most cases, parametric fine-tuning follows. What specialist models gain in precision, however, they sacrifice in flexibility – narrow experts are often inapplicable to other domains without additional fine-tuning [10].

LLM and VLMs introduced an exciting new paradigm: incontext learning (ICL) over text and images. Unlike specialist models, these so-called foundation models adapt to new tasks without weight updates [6]. Unlike specialists, LLMs are extremely flexible; users can ask almost anything in natural language, and in many cases, receive a reasonable response. However, recent research has demonstrated that even state-of-the-art VLMs from OpenAI and Google consistently underperform as predictors when compared to specialist classifiers, especially on non-traditional modalities and in long-tail domains [57]. Gemini, GPT-4V and LLaVA [28] have sought to optimally align language models with specialist embeddings for vision, and in some cases, other modalities as well. However, these efforts have not brought foundation models to parity with specialists –furthermore, for some modalities, such as audio, there is no obvious way to natively utilize a traditional LLM / VLM for predictive tasks.

These challenges motivate our core research question:

###### Research Question

How can we combine the reasoning capabilities of LLMs with the representational power of specialized models without requiring modality-specific fine-tuning?

In this work, we posit that visual reasoning, coupled with specialized low-dimensional embedding models, is a skeleton key that unlocks the power of in-context learning and reasoning for arbitrary data modalities and domains.

ESC-50

100%

CIFAR-100

RAVDESS

80%

60%

MARVIS Modality Colors

Vision Audio Biological Tabular Classification Tabular Regression

| |
|---|

| |
|---|

40%

CIFAR-10

UrbanSound-8K

| |
|---|

| |
|---|

20%

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Methods

| |
|---|

Regression 2025 (Semantic)

AWA2

MARVIS-3B

| |
|---|

| |
|---|

LLM/VLM Base

Specialized Model

CC-18 (Semantic)

FishNet

PlantDoc

- Figure 1: MARVIS transforms VLMs into frontier predictors. Using a standard 3B parameter QwenVL model zero-shot without reasoning, MARVIS (colored line) achieves competitive performance compared to specialized baselines (dashed line) across modalities and domains, far exceeding the best existing LLM / VLM predictors (dotted line).

Contributions

- (1) We propose MARVIS, an efficient, modality-agnostic system for transforming a VLM into a performant predictor. Using a QwenVL model with no specialized reasoning training, MARVIS achieves competitive performance across vision, audio, and tabular modalities, and across a wide range of scientific domains, on both classification and regression tasks.
- (2) We demonstrate empirically that MARVIS does more than simply copy predictions; it reasons over the available information sources, implicitly analyzing and balancing them to improve its own predictive power. It can rationalize its decisions post-hoc and suggest next steps, unlike the specialist models it adapts.
- (3) We also introduce numerous valuable secondary contributions to facilitate future research in this area, including the first large-scale standardized tabular classification and regression datasets with complete semantic information (see Section H), a strong FFT baseline for tabular data (see Section D), comprehensive ablations, and a well-documented Github repository.a

ahttps://github.com/penfever/marvis

2 The MARVIS System

- 2.1 Core Insight: Vision as a Skeleton Key

Relying solely on text to ingest data is limited and does not align with how humans operate. For predictive tasks, it is not usually the raw data that we want the model to reason over; rather, it is a distilled view of that data, for the purposes of answering specific questions or rendering judgments. Human scholars tend to reason more effectively with data visualizations, simplified views of complex data [8, 50]. VLMs, which are pretrained on web-scraped data, can understand and interpret a wide range of scientific imagery

and visualizations of specialized embedding spaces. Thus, we posit that embedding visualizations can operate as skeleton keys, unlocking any kind of data for vision-language models without requiring modality-specific training beyond vision. Moreover, visualizations can be easily generated at inference time with standard packages, such as scikit-learn [37].

#### 2.2 System Architecture

MARVIS operates through a four-stage pipeline (Fig. 2):

- (1) Embedding Generation: A domain-appropriate embedding model maps raw input data to vector representations. MARVIS is agnostic to the choice of embedding model; any model that produces fixed-dimensional vectors can be used.
- (2) Dimensionality Reduction: t-SNE projects the embeddings to 2D, producing a scatter plot visualization optimized for VLM processing. A zoom factor centers the view on the query point and its local neighborhood. The particular dimensionality reduction algorithm is likewise mutable in MARVIS, and we experiment with many.
- (3) Visual Reasoning: The VLM receives the visualization along with a structured prompt containing the class legend, KNN neighbor analysis, and optional metadata (examples available in section K. It reasons over the spatial layout to produce a prediction, either by class name or color.
- (4) Response Processing: A parser extracts the predicted class label from the VLM’s natural language response using the known class-to-color mapping.

Embedding models. MARVIS’s plug-and-play architecture accepts embeddings from any upstream model. Table 1 lists the models used in our experiments; each was selected as a strong representative for its modality. The system requires no modification when switching between modalities—only the embedding model changes.

For vision, DINOv2-ViT-L-14-reg provides robust visual representations trained through self-supervised learning [36]. For audio, Microsoft CLAP creates joint audio-language embeddings via contrastive pre-training [11]. For biological data, BioCLIP2 specializes in scientific vision understanding, incorporating taxonomic labels

1 2 3 4

MARVIS: Modality Adaptive Reasoning over VISualizations CAIS ’26, May 26–29, 2026, San Jose, CA, USA

###### DATA

EMB

VIZ

VLM

|TAB|
|---|

|AUD|
|---|

|VIS|
|---|

TabPFN Whisper DINOV2

t-SNE PCA UMAP

GPT-4V Qwen2.5-VL Gemini

Embedding

Plotting

Data

Prediction

###### Figure 2: The four-stage MARVIS pipeline. We start with raw input data, capture key patterns using specialist embedding generating models, determine an appropriate strategy for plotting the data, and prompt a VLM with visual context, as well as (optionally) metadata and semantic context, then extract predictions.

###### 🎨 Visualizations

###### Modality Model Dim Training

[Figure 1]

Vision DINOv2-ViT-L-14 1024 Self-supervised Audio MS-CLAP 512 Contrastive Biology BioCLIP2 512 Taxonomic contr. Tabular TabPFNv2 256 In-context learning

###### Table 1: Embedding models by modality. MARVIS accepts embeddings from any upstream model; only the embedding model changes between modalities.

in contrastive training [49]. For tabular data, TabPFNv2 provides embeddings via transformer-based in-context learning [19]. Additional details on hyperparameters are provided in Section B.

- 📈 visualization_4.png

[Figure 2]

- 📈 visualization_5.png

#### 2.3 Visualization Engine

Figure 3: Example MARVIS input. A zoomed t-SNE visualization of the CMC dataset as seen by the VLM. The red star marks the query point; colored circles are training points (3 classes); gray squares are test points. The pie chart shows the KNN neighbor distribution computed in the original embedding space. See Section K for more examples.

The visualization engine transforms high-dimensional embeddings into images that a VLM can interpret. Fig. 3 shows an example of what the VLM actually sees at inference time: a zoomed t-SNE scatter plot with color-coded training points, gray test points, a red star marking the query point, and a KNN pie chart summarizing the neighbor distribution in the original embedding space.

Visualization generation. Even modern VLMs do not “see” as well as humans; the particular patch dimensions and the limited range of local attention mean that the VLM performs best when DPI is optimized and a zoom factor is used to enlarge the region of interest around the query point. We find that the optimal zoom factor varies by modality but can be set once per modality (tuning procedure described below). Ideally, the zoom factor is such that the target point and its neighbors are captured within the 14×14 patches from the VLM’s sliding window attention, significantly enhancing spatial understanding.

and ablate over 25 different configurations, including perturbationbased axes, semantic axes, multi-view layouts, and 3D projections. For the main experiments in this paper, we use the “tsne_knn” setting, which overlays KNN neighbor connections computed in the original high-dimensional embedding space onto the 2D t-SNE plot. This setting offers the best speed/quality tradeoff: because KNN operates on the full embeddings without dimensionality reduction, it is sometimes able to discover relationships that the 2D visualization alone would miss. We find that fixing the nearest neighbors hyperparameter at min(30, 10% of the training data) works well across dataset sizes and modalities.

Context composition. One key design decision is which context to include alongside the visualization. In Section E.1, we name

Hyperparameter tuning protocol. All MARVIS hyperparameters except the t-SNE zoom factor are held constant across all experiments at the values listed in Section B (perplexity 15, t-SNE iterations 1000, nearest-neighbor count min(30, ⌈0.1 · |Dtrain|⌉), Euclidean metric for general features and cosine for embeddings); no further per-dataset tuning is performed. The zoom factor is the only hyperparameter we tune per setting, and is tuned once per modality rather than per dataset. For each modality we perform a grid search over a small candidate set of zoom values and select the value with highest validation accuracy, using the validation split provided by the benchmark suite for tabular data and a held-out 10% subset of the training data for audio, vision, and biological data. The selected per-modality zoom values are reported in Section B.

#### 2.4 VLM Reasoning and Response Parsing

VLM backend selection. The choice of VLM architecture is critical: many older architectures either cannot localize what they “see” effectively, or cannot “see” clearly enough to take advantage of visualizations. We select the 3B parameter Qwen 2.5 VL model [2] for three reasons:

- (1) It uses 14×14 patches with sliding window attention, emphasizing local patch interaction—important for distance-based visualizations where proximity matters.
- (2) It processes images of arbitrary aspect ratios without distortion, enabling effective multi-visualization layouts.
- (3) The Qwen 2.5 VL series has been specifically trained on long-context and scientific imagery.

Response parsing. To avoid the common failure mode in which predictions are correct but not detected by the parser, we enforce consistent color schemes and naming across all visualization legends, ensuring clear visual separation for VLM interpretation. The parser is made aware of both the class names and their assigned colors, and is given an explicit mapping between them. Class names in legends are limited to the classes that actually appear in each visualization, controlling legend size for datasets with many classes. Using this simple modification, we are able to scale MARVIS to a surprising degree; on FishNet, a realistic fine-grained image classification task with 463 classes, MARVIS achieves accuracy only slightly worse than the best specialist.

#### 2.5 Backend Robustness

A natural question is whether MARVIS’s performance depends critically on the choice of VLM backend. To answer this, we ablate over multiple VLM backends on a subset of the OpenML CC-18 Semantic tabular classification benchmark (Fig. 4).

The results reveal that MARVIS’s performance depends considerably more on the choice of embedding model than on the choice of VLM backend. A small QwenVL 2.5 3B model (MARVIS-3B) matches GPT-4o-mini (MARVIS_gpt4o) and outperforms a more recent thinking model (Kimi-VL-A3B-Thinking, referenced as MARVIS_kimi).

#### 2.6 System Design and Deployment

Latency breakdown. We profile MARVIS using PyTorch device synchronization barriers with time.perf_counter() for accurate

GPU-aware wall-clock timing. Table 2 reports per-sample costs averaged over 10 runs on consumer hardware (Apple M-series, MPS backend) with a 200-sample tabular dataset. VLM inference overwhelmingly dominates at 98.9% of per-sample time. Visualization generation (t-SNE rendering, zoom, KNN overlay, and rasterization) accounts for only 1.1%, and response parsing is negligible. On server hardware (H100), VLM inference drops to 0.5–2.0s per sample.

Stage Time (ms) % Total One-time costs (amortized over 𝑁 test samples)

Embedding generation 5702 t-SNE fitting 752 —

Per-sample costs (mean ± std, 𝑁=10)

Visualization generation 64 ± 4 1.1% VLM inference (3B) 5930 ± 1143 98.9% Response parsing <1 <0.1%

End-to-end per sample 5998 ± 1143 100%

Table 2: Per-sample latency breakdown on consumer hardware (Apple MPS, Qwen2.5-VL-3B). VLM inference dominates; all other stages are negligible. On server GPUs (H100), end-to-end time drops to 0.5–2.0s per sample.

Parallelism and scalability. The MARVIS pipeline admits natural parallelism at multiple levels. Visualization generation is strongly parallel: each test sample’s t-SNE plot is independent once the shared t-SNE projection is computed, so 𝑁 samples can be rendered concurrently across CPU cores. VLM inference can be batched across samples on a single GPU, or distributed across multiple GPUs with standard tensor parallelism. The most expensive onetime costs—embedding generation (5.7s) and t-SNE fitting (0.75s)are amortized across all test samples; for a 1000-sample test set, these add only 6.5ms per sample. Peak GPU memory remains under 8GB for the 3B model, enabling deployment on consumer hardware (e.g., a single RTX 4090 or Apple M-series laptop).

Modularity. MARVIS’s four-stage pipeline is fully modular: the embedding model, visualization strategy, VLM backend, and response parser can each be swapped independently. Switching from tabular to audio data requires only changing the embedding model (Table 1); switching from a local 3B model to GPT-4o-mini requires only changing the VLM backend (Fig. 4); switching from KNN overlays to perturbation axes requires only changing the visualization configuration. No retraining is needed for any swap.

Information available to the VLM. The VLM does not receive embedding vectors or raw feature values. Its input, for each test point, is a rasterized 2D scatter plot of the t-SNE projection, a class legend, an optional KNN pie chart summarizing nearest neighbors, and the task prompt. Table 3 enumerates this interface. This makes MARVIS’s VLM interface narrower than approaches that serialize feature values (e.g., TabLLM, JOLT) or transmit embedding vectors.

The interface is narrow but structured. From the plot and legend the VLM can infer—and must, for its reasoning to be useful—the number and relative sizes of classes; the neighborhood structure of classes in the embedding space (which classes cluster together, which overlap, where a test point sits relative to them); and the presence of clusters, modes, and outliers. When semantic class

[Figure 3]

###### Figure 4: Accuracy matrix for MARVIS backend variants and FFT. MARVIS’s performance depends considerably more on the choice of embedding model than on the VLM backend: a 3B QwenVL model matches GPT-4o-mini and outperforms a recent thinking model.

###### Provided to VLM Not part of VLM input

Rasterized 2D scatter plot Raw feature values Class legend (labels + colors) Column names and schema meta-

data KNN pie chart (neighbor counts) Original embedding vectors Task prompt Source row indices

###### Table 3: Information flow into the VLM. Left: data the VLM receives for each test point. Right: data produced or consumed by MARVIS that is not part of the VLM input.

names are used, the class taxonomy of the task is also exposed. When semantic-axes or perturbation-axes contexts are enabled, named feature directions and their local effects are exposed through plot annotations and the prompt, effectively revealing part of the feature schema.

For completeness we note two further, more conservative properties of these inputs. First, the plot and legend are in principle consistent with membership inference [46]: a party holding the embedding model and a candidate raw input can forward-project the candidate and compare its location against the plot, which may reveal whether a similar point was present in the visualized dataset. Second, because outliers sit visibly apart from clusters, atypical examples in the dataset are individually more distinguishable in the plot than typical examples are.

The default MARVIS configuration used throughout our experiments (t-SNE scatter plot with KNN overlay) does not transmit semantic class names, named feature directions, or raw feature values; the class legend identifies classes by index only. Semantic class names, semantic axes, and perturbation axes are optional contextcomposition variants, explored in Fig. 5, that widen this interface beyond the default.

#### 3 Experiments

Overview. Our main experiments assess MARVIS across four distinct modalities using domain-appropriate embedding models and established benchmarks; we compare against both specialized baselines and alternative LLM/VLM approaches.

Table 4 presents MARVIS performance across all modalities compared to 5 specialized baselines and 4 alternative LLM/VLM approaches. For each benchmark, we conduct a single MARVIS run. We use a QwenVL 2.5 3B Instruct backbone. MARVIS hyperparameters are set as described in Section 2: the t-SNE zoom factor is selected per modality by grid search on a held-out validation split; all other hyperparameters are fixed across benchmarks. The LLM / VLM baseline results in the paper are reported using the best performing LLM / VLM in the class (we consider QwenVL 2.5 3B Instruct and Gemini-Flash-2.0 via the Gemini API). All MARVIS results are zero-shot in the sense that we do not give examples of the task to the VLM at inference time; they are full-shot in the sense that the embedding-generating models have access to the entire test set without labels. For the LLM / VLM baselines, image classification is performed zero-shot. Tabular classification and regression uses the JOLT [47] and TabLLM [17] strategies with k-shot computed dynamically based on the maximum context length. We report the best result in the table. Specialist models are full-shot, and we report the best overall result in the table. For extended results, a detailed description of the method we use to generate our novel tabular benchmarks CC18-Semantic and Regression2025-Semantic, and a deeper dive into tabular data, including balanced metrics, please refer to Section G.

Specialized model baselines. For vision, the best performing specialist was the large DinoV2 model with a registry and KNN classification [36]. For audio, the CLAP model with contrastive zero-shot classification from Microsoft and OpenAI’s Whisper-V2Large model with KNN classification perform the best [11, 32, 42].

For biological data, BioCLIPv2 with KNN classification performs the best [15]. For tabular data, TabPFNv2 with standard forward pass classification and regression is a strong baseline; we also consider classical baselines such as CatBoost and linear models in Section G [19, 40].

LLM / VLM baselines. For vision, we use the standard strategy of zero-shot prompting and exact match extraction described in works such as [57]. For audio, we are unable to compare to public API-based models, as to the best of our knowledge, no generalist exists capable of performing audio classification.

LLM tabular baselines. In the tabular domain, as a secondary contribution, we generate the first large-scale standardized benchmarks for tabular classification and regression that include semantic class names, feature names and metadata; CC18-Semantic and Regression 2025 Semantic. We also re-implement two prominent LLM-tabular methods, TabLLM and JOLT [17, 47], which lack general-purpose implementations. For more details on this, please refer to Section G. Additional details. For more analysis on the embedding models and baselines, please refer to Section B. For more explanation of the benchmarks we use, please refer to Section A.

#### 3.1 Findings

MARVIS is competitive with SOTA specialist predictors. Across a wide range of modalities, we observe that MARVIS strongly conserves predictive performance – across most tasks we consider, it is able to match the best specialist model in the cohort. By comparison, the best existing LLM / VLM methods, tailored for each domain, achieve 77% of specialist performance on average. Remarkably, we find that MARVIS is a more accurate image classifier than Gemini Flash 2.0, despite never actually having seen the images. MARVIS also sometimes improves on specialists; it outperforms CLAP, a specialist contrastive predictor, using its own embeddings.

###### Contributions

MARVIS-3B achieves competitive performance across four distinct modalities, approaching and occasionally exceeding the best specialist predictors, and improving on LLM / VLM-only methods by 16.7%.

MARVIS outperforms direct fine-tuning of its base model. In Section D, we describe a novel method for fine-tuning an LLM directly on the embeddings of an upstream model such as TabPFNv2. We test this method (Qwen-FFT) at inference time and find that it is highly accurate, far outperforming previously published strategies such as JOLT and TabLLM for general-case tabular inference with LLMs; however, as shown in Section 2.5, MARVIS-3B outperforms even this strong baseline on average. This result may be surprising – although it is time-consuming and costly, full fine-tuning is generally considered the gold standard for accuracy. However, we find that, although the FFT solution generally reduces training loss to near-zero, it can fail to generalize, particularly when the training dataset size is small. This finding suggests that the visual reasoning

paradigm of MARVIS is a more robust approach to leveraging VLMs for prediction than direct fine-tuning on embeddings.

VLMs reason over their input data and condition their behavior based on the context provided. One core research question, from our perspective, was whether a VLM was simply copying learned patterns or utilizing simple heuristics to achieve this strong performance. Systematic analysis of VLM reasoning in Fig. 5 demonstrates clear correlations between reasoning quality and metric gains, on average, across three tabular classification datasets (two with meaningful semantic features, one without).

[Figure 4]

Figure 5: The selection of context strongly influences MARVIS performance. We ablate over twenty different context composition strategies, and find that perturbation-based approaches with uncertainty analysis achieve the highest performance, followed by semantic axes with meaningful class labels. The majority of the experiments in the paper are conducted using TSNe + KNN, which we adopt as our default configuration.

Further analysis of disagreement patterns reveals that only 35% of methods agree on all test cases, with 65% showing partial disagreement. Furthermore,in Table5, we show that different visualization methods elicit systematically different reasoning approaches, providing strong evidence that VLMs adapt their analysis based on visual information content. Still more evidence can be found in Section I.1.

We observe that different visualization methods elicit systematically different reasoning approaches, providing strong evidence that VLMs adapt their analysis based on the available visual information. tsne_knn produces quantitative neighbor analysis with explicit distance calculations (average 48.0 words), tsne_semantic_axes integrates semantic class information with spatial reasoning (304.9 character responses) and tsne_perturbation_axes generates the longest, most detailed responses (310.6 characters) with sophisticated uncertainty analysis.

- Table 4: Domain-specific embeddings, benchmarks, and detailed results. Results are boldfaced when statistically tied for best performance within 95% confidence intervals (normal approximation). MARVIS demonstrates competitive or superior performance on most individual benchmarks, achieving average results within 2.5% of an ensemble of specialized methods while providing universal applicability. Benchmark acronyms: C10 = CIFAR-10, C100 = CIFAR-100, ESC = ESC-50, RAV = RAVDESS, US8 = UrbanSound8K, FSH = FishNet, AWA = AWA2, PLD = PlantDoc, CC18 = OpenML CC18, R25 = Regression 2025. We show the best results of specialized models and traditional LLM/VLM approaches. For all benchmarks except R25, the metric is Accuracy. For R25, it is R2 Score (with a minimum score of 0). The number reported is the mean over all sub-tasks for multi-task benchmarks.

Domain Embeddings Bench. Size (K) MARVIS Specialist LLM/VLM 95% CI Vision DINOV2

C10 60 98.0 99.0 (DINOV2) 85.7 (Gemini) ±0.1 C100 60 88.0 91.6 (DINOV2) 64.3 (Gemini) ±0.3

Audio CLAP

ESC 2 91.3 90.5 (CLAP) – ±1.2 RAV 1.4 38.4 47.9 (Whisper) – ±2.5 US8 8.7 79.8 77.1 (CLAP) – ±0.8

Biological BioCLIP2

FSH 94 80.2 83.7 (BioCLIP) 59.5 (Gemini) ±0.3 AWA 37 95.7 97.1 (BioCLIP) 96.5 (Gemini) ±0.2 PLD 2.5 67.4 72.0 (BioCLIP) 74.2 (Gemini) ±1.8

Tabular TabPFNv2

CC18 155 84.5 87.8 (TabPFNv2) 50.1 (TabLLM) ±0.2 R25 35 66.0 67.0 (TabPFNv2) 05.1 (JOLT) ±0.5

(Score, # Models) – (78.9, 1) (81.4, 5) (62.2, 4) –

These patterns suggest that VLMs engage in more thorough spatial analysis when the visual information supports accurate classification, indicating genuine reasoning rather than pattern matching.

- Table 5: Method-Specific Reasoning Patterns. Each visualization method elicits distinct reasoning behaviors: k-NN methods trigger quantitative distance analysis, perturbation methods generate longer responses, and basic methods rely heavily on proximity heuristics. Here, Resp. Length refers to the token count of responses, distance mentions to the rate at which the response mentions distance between points in embedded space, and closest usage refers to how often MARVIS uses the word “closest” in its response.

Table 6: Reasoning Quality Predicts Accuracy. Correct predictions exhibit longer, more sophisticated responses with increased spatial analysis and reduced reliance on simple heuristics, across 83 experimental configurations.

###### Reasoning Feature Correct Incorrect Δ

Response Length (chars) 281.2 268.3 +12.9 Color Mentions 1.85 1.52 +0.33 Distance Reasoning 0.074 0.057 +0.018

“Closest” Heuristic 0.56 0.77 −0.21 “Majority” Heuristic 0.05 0.25 −0.20

Method Resp. Length Dist. Mentions Closest Usage

tsne_3d_perturbation 365.3 0.000 0.433 tsne_perturbation_axes 310.6 0.000 0.650 tsne_semantic_axes 304.9 0.000 0.683 tsne_knn 279.0 0.650 0.883 basic_tsne 268.3 0.000 1.000

Reasoning quality predicts classification accuracy. Beyond method-level differences, we find that individual prediction quality correlates with measurable reasoning features. Table 6 analyzes 83 experimental configurations across multiple test cases, comparing the reasoning traces of correct vs. incorrect predictions. Correct predictions exhibit longer, more detailed responses with more color mentions and explicit distance reasoning. Critically, incorrect predictions rely significantly more on simple heuristics—words like “closest” (+0.21) and “majority” (+0.20)—suggesting that when the VLM falls back on shallow shortcuts instead of engaging with the spatial structure, accuracy suffers.

This pattern provides concrete evidence that MARVIS accuracy is driven by genuine visual reasoning. The following excerpts from VLM reasoning traces illustrate how different visualization methods elicit qualitatively different reasoning strategies:

With k-NN overlay (quantitative distance reasoning):

“The query point is closer to the cluster of Class_1 neighbors (4 neighbors) than to the cluster of Class_2 neighbors (1 neighbor). Additionally, the average distance to Class_1 neighbors (6.1) is slightly lower than to Class_2 neighbors (5.2), indicating higher similarity to Class_1.”

With semantic axes (class-label integration):

“The red star (query point) is closest to the orange-colored points, which represent the ‘Long-term methods’ class. This spatial clustering indicates that the query point is more aligned with the characteristics of the ‘Long-term methods’ class.”

Basic t-SNE only (proximity heuristic):

“The red star (query point) is closest to the green-colored training points, which are associated with Class_2.”

Quantitative distance calculations appear only when k-NN information is provided; semantic reasoning emerges only with meaningful class labels; and without either, the VLM falls back on simple proximity. This adaptive behavior—not fixed pattern matching—explains why richer visualization contexts yield higher accuracy (Fig. 5). For additional analysis, see Section I.

MARVIS performance is robust under an inductive-only tSNE projection. The default MARVIS pipeline fits t-SNE jointly on training and test embeddings (transductive). To test whether MARVIS could be deployed inductively—fitting t-SNE on the training set alone and projecting each test point in afterward, one at a time—we replace the transductive step with an explicit out-of-sample projector. t-SNE has no native out-of-sample extension (sklearn’s TSNE provides only fit_transform, with no learned function to apply to a new point), so we deliberately choose the simplest plausible parametric approximation: a linear regression 𝑓 (𝑥) =𝑊𝑥 +𝑏, fit by closed-form least squares from each training embedding 𝑥𝑖 to its training-only t-SNE coordinate 𝑦𝑖. At inference, each test point is projected independently as 𝑦ˆtest = 𝑓 (𝑥test). We use this cheap baseline rather than a more sophisticated method (such as parametric t-SNE or per-point optimization against a frozen training layout) precisely because any stronger inductive projector should perform at least as well; the linear regression therefore acts as a lower bound on what an inductive MARVIS deployment could achieve. All other components are unchanged from the main paper: DINOv2-ViT-L/14reg, 𝑘 = 30 KNN overlay computed in the original embedding space (so KNN statistics are identical across conditions), zoom factor 15, Qwen2.5-VL-3B. Table 7 reports accuracy on 1,000 held-out CIFAR10 test images. The inductive variant retains 96.4% accuracy, a drop of only 1.4 percentage points relative to the transductive baseline, and per-sample VLM latency is unchanged. Even this deliberately weak projector preserves nearly all of the model’s classification accuracy, indicating that MARVIS’s performance is largely determined by the training-set t-SNE map and the VLM’s reasoning over it, rather than by transductive optimization with test points in scope.

Table 7: Inductive-only t-SNE ablation on CIFAR-10. Replacing transductive joint t-SNE with a train-only fit plus linear inductiveprojection of testpoints reduces accuracy by 1.4 percentage points at matched compute. Both rows: 50,000 training images, 1,000 test images, DINOv2-ViT-L/14-reg, 𝑘 = 30, zoom factor = 15, Qwen2.5-VL-3B.

Variant Accuracy t-SNE fit VLM inf.

Transductive (default) 97.8% 212s 4076s Inductive (LR) 96.4% 169s 4009s

The flexibility of MARVIS allows for more complex use cases. In Fig. 6, we demonstrate one such use case – open-ended chat about a particular predictive result. In this example, the user asks MARVIS to assess its own performance and recommend strategies to improve results in the future.

#### 4 Related Work

MARVIS builds on extensive prior work in vision-language models (VLMs) which has followed two primary evolutionary tracks: maximalist approaches from industry labs focusing on peak performance, and minimalist open-source approaches prioritizing efficiency and accessibility; in Section F, we trace the history of this evolution in greater detail.

The use of embedding spaces for cross-modal understanding has roots in representation learning [4] and dimensionality reduction techniques [51]. Recent work has explored the geometric properties of embedding spaces [12] and their visualization for interpretability [29]. t-SNE and UMAP have been widely used for visualizing high-dimensional data [34], but their application to VLM reasoning represents a novel paradigm. Previous work on visual reasoning has focused on spatial relationships in natural images [20], but MARVIS extends this to abstract embedding spaces across arbitrary modalities.

MARVIS distinguishes itself from existing approaches through several key innovations: (1) Training-free adaptation: Unlike approaches requiring extensive fine-tuning, MARVIS leverages pretrained components without modification; (2) Universal modality support: A single architecture handles any data type through embedding visualization; (3) Computational efficiency: Achieves competitive performance with a 3B parameter model versus much larger specialized systems.

#### 5 Conclusion

We introduce MARVIS, a method that enables small VLMs to predict across any data modality through embedding visualization. By transforming embedding spaces into visual representations optimized for VLM spatial reasoning, MARVIS achieves competitive performance across diverse domains.

MARVIS addresses key limitations in existing approaches: it requires no domain-specific training and maintains competitive performance across modalities. The approach demonstrates that visual reasoning can serve as a universal interface for foundation models across any data modality.

Based on this, we propose several key principles for designing effective VLM interfaces:

- • Information density matters: Richer visualizations elicit more sophisticated reasoning
- • Method-purpose alignment: Different visualization approaches suit different reasoning tasks
- • Adaptive interface design: VLMs can effectively utilize different types of visual information

Future work includes further investigation of the optimal mix of visualizations and embeddings to boost performance and finetuning strategies which may improve the performance of base VLMs for reasoning over scientific imagery, including reasoning post-training.

#### References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, Vol. 35. 23716–23736.

### MARVIS: Interactive Classification Chat

User

MARVIS

The model performed well on the test data, achieving an accuracy of 0.8. This indicates that the model correctly classified 80% of the test samples, which is a satisfactory performance for a classification task.

User

MARVIS

The t-SNE visualization reveals distinct clusters of points corresponding to different classes. Class A is represented by blue dots, Class B by orange dots, and Class C by green dots. The query point, indicated by a red star, is positioned within the space between the clusters of Class B and Class C, suggesting that it may be challenging for the model to classify it correctly.

The KNN distribution chart on the right side of the visualization shows ...

User

MARVIS

I am 95% confident in the last prediction, as the model achieved an accuracy of 0.8, which indicates a high level of confidence in its classification. The visualization and KNN distribution suggest that the query point is close to the centroids of Class B and Class C, which could lead to some misclassification. However, the model's overall accuracy suggests that the classification is robust and reliable.

###### Figure 6: MARVIS extends traditional predictive capabilities. Because it requires no fine-tuning, and because it exposes the VLM’s classification process to the VLM itself, MARVIS enables VLMs to reason over, and converse about, their predictive performance.

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923 [cs.CV] https://arxiv.org/abs/2502.13923
- [3] Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2018. Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence 41, 2 (2018), 423–443.
- [4] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. Representation learning: A review and new perspectives. Vol. 35. IEEE transactions on pattern analysis and machine intelligence. 1798–1828 pages.
- [5] Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael G. Mantovani, Jan N. van Rijn, and Joaquin Vanschoren. 2021. OpenML Benchmarking Suites. arXiv:1708.03731 [stat.ML] https://arxiv.org/abs/ 1708.03731
- [6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. arXiv:2005.14165 [cs.CL] https://arxiv.org/abs/2005.14165

- [7] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al.

2022. PaLI: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794 (2022).

- [8] Cyril de Bodt, Alex Diaz-Papkovich, Michael Bleher, Kerstin Bunte, Corinna Coupette, Sebastian Damrich, Enrique Fita Sanmartin, Fred A Hamprecht, EmőkeÁgnes Horvát, Dhruv Kohli, et al. 2025. Low-dimensional embeddings of highdimensional data. arXiv preprint arXiv:2508.15929 (2025).
- [9] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al.

2024. Molmo and PixMo: Open weights and open data for state-of-the-art visionlanguage models. arXiv preprint arXiv:2409.17146 (2024).

- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs.CL] https://arxiv.org/abs/1810.04805
- [11] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang.

2023. CLAP: Learning audio concepts from natural language supervision. arXiv preprint arXiv:2206.04769 (2023).

- [12] Kawin Ethayarajh. 2019. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 55–65. doi:10.18653/v1/D19-1006

- [13] Benjamin Feuer, Robin Tibor Schirrmeister, Valeriia Cherepanova, Chinmay Hegde, Frank Hutter, Micah Goldblum, Niv Cohen, and Colin White. 2024. TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks. arXiv:2402.11137 [cs.LG] https://arxiv.org/abs/2402.11137
- [14] Valentin Gabeff, Marc Rußwurm, Devis Tuia, and Alexander Mathis. 2024. WildCLIP: Scene and animal attribute retrieval from camera trap data with domainadapted vision-language models. IJCV 132, 9 (2024), 3770–3786.
- [15] Jianyang Gu, Samuel Stevens, Elizabeth G Campolongo, Matthew J Thompson, Net Zhang, Jiaman Wu, Andrei Kopanev, Zheda Mai, Alexander E. White, James Balhoff, Wasila Dahdul, Daniel Rubenstein, Hilmar Lapp, Tanya Berger-Wolf, Wei-Lun Chao, and Yu Su. 2025. BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning. arXiv:2505.23883 [cs.CV] https://arxiv.org/ abs/2505.23883
- [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual Learning for Image Recognition. arXiv:1512.03385 [cs.CV] https://arxiv.org/abs/ 1512.03385
- [17] Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. 2023. TabLLM: Few-shot classification of tabular data with large language models. arXiv preprint arXiv:2210.10723 (2023).
- [18] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. 2022. TabPFN: A transformer that solves small tabular classification problems in a second. arXiv preprint arXiv:2207.01848 (2022).
- [19] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. Accurate predictions on small data with a tabular foundation model. Nature 637, 8045 (Jan. 2025), 319–326. doi:10.1038/s41586-024-08328-6
- [20] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. 2017. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. Proceedings of the IEEE conference on computer vision and pattern recognition (2017), 2901–2910.
- [21] Faizan Farooq Khan, Xiang Li, Andrew J. Temple, and Mohamed Elhoseiny.

2023. FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Trait Prediction. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Paris, France, 20439–20449. doi:10.1109/ ICCV51070.2023.01874

- [22] Alex Krizhevsky. 2009. Learning Multiple Layers of Features from Tiny Images. https://api.semanticscholar.org/CorpusID:18268744
- [23] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023. OtterHD: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219 (2023).
- [24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. International Conference on Machine Learning (2022), 12888–12900.
- [25] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2023. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597 (2023).
- [26] Fangxin Liu, Wenjie Zhang, Libo Chen, Jincan Wang, Mingshan Luo, and Yuliang Chen. 2025. Global semantic-guided sub-image feature weight allocation in high-resolution large vision-language models. arXiv preprint arXiv:2501.14276

(2025).

- [27] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. arXiv preprint arXiv:2401.13601 (2024).
- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in Neural Information Processing Systems 36 (2023).
- [29] Shixia Liu, Xiting Wang, Mengchen Liu, and Jun Zhu. 2017. Towards better analysis of machine learning models: A visual analytics perspective. Visual Informatics 1, 1 (2017), 48–56.
- [30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, FengLi,Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 2023. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023).
- [31] Steven R. Livingstone and Frank A. Russo. 2018. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE 13, 5 (May 2018), 1–35. doi:10.1371/journal.pone.0196391 Publisher: Public Library of Science.
- [32] Rao Ma, Adian Liusie, Mark Gales, and Kate Knill. 2024. Investigating the Emergent Audio Classification Ability of ASR Foundation Models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 4746–4760. doi:10.18653/v1/2024.naacl-long.266
- [33] Yuan Ma, Tianyu Li, Dongdong Chen, Zhenglu Wu, Xuguang Li, Lu Chen, Kai Zhang, Zilong Wang, Chunyang Liu, Kexin Wang, et al. 2024. POINTS: Improving your vision-language model with affordable strategies. arXiv preprint arXiv:2409.04828 (2024).

- [34] Leland McInnes, John Healy, and James Melville. 2018. UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018).
- [35] Andreas Müller, Carlo Curino, and Raghu Ramakrishnan. 2025. MotherNet: Fast Training and Inference via Hyper-Network Transformers. arXiv:2312.08598 [cs.LG] https://arxiv.org/abs/2312.08598
- [36] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, et al. 2023. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).
- [37] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. 2011. Scikit-learn: Machine learning in Python. the Journal of machine Learning research 12 (2011), 2825–2830.
- [38] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824 (2023).
- [39] Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd Annual ACM Conference on Multimedia (Brisbane, Australia, 2015-10-13). ACM Press, 1015–1018. doi:10.1145/2733373.2806390
- [40] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: unbiased boosting with categorical features. Advances in neural information processing systems 31 (2018).
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. International conference on machine learning (2021), 8748–8763.
- [42] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust Speech Recognition via Large-Scale Weak Supervision. arXiv:2212.04356 [eess.AS] https://arxiv.org/abs/2212.04356
- [43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

2022. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125 (2022).

- [44] Julian D Santamaria, Claudia Isaza, and Jhony H Giraldo. 2025. CATALOG: A Camera Trap Language-guided Contrastive Learning Model. In WACV. IEEE, 1197–1206.
- [45] Srikumar Sastry, Subash Khanal, Aayush Dhakal, Adeel Ahmad, and Nathan Jacobs. 2025. Taxabind: A unified embedding space for ecological applications. In WACV. IEEE, 1765–1774.
- [46] Reza Shokri, Marco Stronati, and Vitaly Shmatikov. 2016. Membership Inference Attacks against Machine Learning Models. CoRR abs/1610.05820 (2016). arXiv:1610.05820 http://arxiv.org/abs/1610.05820
- [47] Aliaksandra Shysheya, John Bronskill, James Requeima, Shoaib Ahmed Siddiqui, Javier Gonzalez, David Duvenaud, and Richard E. Turner. 2025. JoLT: Joint Probabilistic Predictions on Tabular Data Using LLMs. arXiv:2502.11877 [stat.ML] https://arxiv.org/abs/2502.11877
- [48] Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra. 2020. PlantDoc: A Dataset for Visual Plant Disease Detection. In Proceedings of the 7th ACM IKDD CoDS and 25th COMAD (Hyderabad, India) (CoDS COMAD 2020). Association for Computing Machinery, New York, NY, USA, 249–253. doi:10.1145/3371158.3371196
- [49] Samuel Stevens, Jiaman Wu, Matthew J Thompson, Elizabeth G Campolongo, Chan Hee Song, David Edward Carlyn, Li Dong, Wasila M Dahdul, Charles Stewart, Tanya Berger-Wolf, Wei-Lun Chao, and Yu Su. 2024. BioCLIP: A Vision Foundation Model for the Tree of Life. arXiv:2311.18803 [cs.CV] https://arxiv. org/abs/2311.18803
- [50] Antony Unwin. 2020. Why Is Data Visualization Important? What Is Important in Data Visualization? Harvard Data Science Review 2, 1 (jan 31 2020). https://hdsr.mitpress.mit.edu/pub/zok97i7p.
- [51] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, 11 (2008).
- [52] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jingjing Wang, Zhuang Lei, Dongmei Jiang, Renrui Ren, Junlin Yan, et al. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024).
- [53] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2022. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2022).
- [54] Yongqin Xian, Christoph H. Lampert, Bernt Schiele, and Zeynep Akata. 2019. Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly. IEEE Transactions on Pattern Analysis and Machine Intelligence 41, 9 (2019), 2251–2265. doi:10.1109/TPAMI.2018.2857768
- [55] Chih-Hsuan Yang, Benjamin Feuer, Talukder Jubery, Zi Deng, Andre Nakkab, Md Zahid Hasan, Shivani Chiranjeevi, Kelly Marshall, Nirmal Baishnab, Asheesh Singh, et al. 2024. BioTrove: A Large Curated Image Dataset Enabling AI for Biodiversity. In NeurIPS, Vol. 37. 102101–102120.
- [56] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mPLUG-Owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv

preprint arXiv:2311.04257 (2023).

[57] Yuhui Zhang, Alyssa Unell, Xiaohan Wang, Dhruba Ghosh, Yuchang Su, Ludwig Schmidt, and Serena Yeung-Levy. 2024. Why are Visually-Grounded Language Models Bad at Image Classification? arXiv:2405.18415 [cs.CV] https://arxiv.org/ abs/2405.18415

Contents Abstract 1

- 1 Introduction 1
- 2 The MARVIS System 2

- 2.1 Core Insight: Vision as a Skeleton Key 2
- 2.2 System Architecture 2
- 2.3 Visualization Engine 3
- 2.4 VLM Reasoning and Response Parsing 4
- 2.5 Backend Robustness 4
- 2.6 System Design and Deployment 4

- 3 Experiments 5 3.1 Findings 6
- 4 Related Work 8
- 5 Conclusion 8 References 8 Contents 12

- A APPENDIX: Benchmark Dataset Descriptions 13

- A.1 Vision Benchmarks 13
- A.2 Audio Benchmarks 13
- A.3 Biological/Scientific Vision Benchmarks 13
- A.4 Tabular Benchmarks 13

- B Implementation Details 13

- B.1 Embedding Models 13
- B.2 Hyperparameters 13

- C Computational Efficiency 14
- D Full Finetuning Experiments 14

- D.1 Balanced Prefix Construction 14
- D.2 Special Tokens and Class Tokens 14
- D.3 Position-wise Projection into Token Space 14
- D.4 Backbone and Hooks 14
- D.5 Label Encoding 14
- D.6 FFT Training Configuration 14
- D.7 FFT Evaluation Protocol 15
- D.8 FFT Limitations and Discussion 15

- E Extended Results 15

- E.1 Ablation Study on Context Choice Details 15
- E.2 Ablation on MARVIS Backend and FFT 15

- F Extended Related Works 15
- G Deep Dive: Tabular Modality Analysis 16

- G.1 Baselines: JOLT and TabLLM 17
- G.2 Classification Performance on OpenML CC18 18
- G.3 Regression Performance Analysis 18
- G.4 Correlation Analysis with TabPFN v2 18
- G.5 Analysis and Discussion 20

- H CC18-Semantic and Regression2025-Semantic: Semantic Metadata Generation for Enhanced Dataset Understanding 20

- H.1 Motivation and Scope 20
- H.2 Semantic Metadata Generation Algorithm 20
- H.3 Semantic Enrichment Structure 20
- H.4 Multi-Source Research Methodology 22
- H.5 Quality Assurance and Validation 22
- H.6 Comprehensive Dataset Characterization 22

- I VLM Reasoning Analysis 23

- I.1 Comprehensive Reasoning Pattern Analysis 23
- I.2 Adaptive Reasoning Evidence 24

- J MARVIS Extended Results 24
- K MARVIS Visualization Gallery 26 K.1 CMC Dataset 26 Artifact Appendix 36

A APPENDIX: Benchmark Dataset Descriptions

- A.1 Vision Benchmarks

CIFAR-10: One of the most widely used datasets for computer vision research: contains 60,000 32×32 color images in 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks) with 6,000 images per class. Split into 50,000 training and 10,000 test images [22].

CIFAR-100: Similar to CIFAR-10 but with 100 classes containing 600 images each (500 training, 100 test per class). The 100 classes are grouped into 20 superclasses, making this a more challenging classification benchmark.

- A.2 Audio Benchmarks

ESC-50 (Environmental Sound Classification): Contains 2,000 environmental audio recordings with 50 classes and 40 clips per class. Each clip is 5 seconds long at 44.1 kHz, single channel, extracted from public field recordings through Freesound.org [39].

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song): Audio dataset focusing on emotion recognition tasks, commonly used for evaluating emotional speech and song recognition capabilities [31].

UrbanSound8K: Contains 8,732 labeled sound excerpts with 10 classes of outdoor/urban sounds, specifically designed for benchmarking sound classification models in urban environments.

- A.3 Biological/Scientific Vision Benchmarks

FishNet: Large-scale dataset with 94,532 images from 17,357 aquatic species, organized by biological taxonomy (8 classes, 83 orders, 463 families, 3,826 genera). Includes bounding box annotations and supports classification, detection, and functional trait prediction tasks [21]. We treat FishNet as a classification problem over families.

AWA2 (Animals with Attributes 2): Animal classification dataset used for zero-shot learning tasks, focusing on learning representations with animal attributes. Part of challenging benchmarks alongside CUB and SUN datasets [54]. We treat AWA2 as a 50-class classification problem with no holdout classes.

PlantDoc: Contains 2,569 images across 13 plant species and 30 classes (diseased and healthy) with 8,851 total labels. Split into 2,328 training and 237 test images, with unbalanced classes ranging from 50-180 images per class [48].

- A.4 Tabular Benchmarks

OpenML CC18: Curated benchmark suite of 72 classification datasets from OpenML 69 of which we utilize), selected based on strict criteria:

- • Size: 500-100,000 observations, ≤ 5,000 features
- • Quality: No artificial data, minority/majority class ratio≥0.05
- • Usability: Compatible with multiple algorithms, representing commonly used ML datasets

See [5] for more on this benchmark, including the complete specification of tasks.

Regression 2025: Custom benchmark of 43 regression tasks from 2015-2025 sourced from OpenML, evaluated using R2 scores on a 0-100 scale for consistent comparison across tasks; introduced

onto the OpenML platform in March 2025.1 Please follow the link for the complete list and specification of tasks. After discarding tasks on which all models fail, we compute our scores on a subset of 33.

#### B Implementation Details

This section contains additional experimental details from the paper.

#### B.1 Embedding Models

See Table 1 in the main body for an overview of the embedding models used per modality. Here we provide additional context on the biological embedding model lineage: BioCLIP2 is the latest in a series of foundation models for biological applications, initiated by BioCLIP, which incorporated taxonomic labels in the vision-language contrastive training [49]. Follow-up work scaled data to 162M images [BioTrove, 55], specialized the data to camera traps [CATALOG and WildCLIP, 14, 44], and added additional model modalities [TaxaBind, 45]. For tabular data, TabPFN [18] employed transformer-based in-context learning, and was later extended to support larger datasets [13, 19, 35].

#### B.2 Hyperparameters

In this section, we document the hyperparameters used for our main experiments section.

###### t-SNE Configuration:

- • Perplexity: 15 (optimized through ablation studies)
- • Iterations: 1000 for stable convergence
- • Learning rate: 200 (default)
- • Random state: Fixed for reproducibility

###### KNN Configuration

- • nn = 30
- • metric = ’euclidean’ (general), ’cosine’ (embeddings)
- • weights = ’distance’

Zoom Factor (per-modality). Selected by grid search on validation data as described in Section 2.

- • Tabular (classification and regression): 2.0
- • Audio: 7.0
- • Vision: 15.0
- • Biological: 20.0

Tabular Baseline Models Configuration: CatBoost (Classification & Regression)

- • iterations: 1000
- • depth: 6
- • learning_rate: 0.03
- • random_seed: 42
- • verbose: False
- • Categorical features: Auto-detected and preserved

###### TabPFN v2 (Classification & Regression)

- • n_estimators: 8
- • device: Auto-detected (CUDA if available)
- • ignore_pretraining_limits: True
- • Target preprocessing: Quantile binning for regression

1https://openml.org/search?type=benchmark&sort=tasks_included&study_type= task&id=455

- • Max quantiles: min(n_samples // 2, 1000)
- • NaN/INF imputation: Median strategy

###### Random Forest (Classification & Regression)

- • n_estimators: 100
- • max_depth: None (unlimited)
- • random_state: 42
- • n_jobs: -1 (all cores)

###### Gradient Boosting (Classification & Regression)

- • n_estimators: 100
- • learning_rate: 0.1
- • random_state: 42
- • Feature selection: Max 500 features (SelectKBest)

###### Logistic/Linear Regression

- • max_iter: 1000 (Logistic only)
- • C: 1.0 (Logistic regularization)
- • random_state: 42
- • n_jobs: -1 (all cores)
- • Preprocessing: StandardScaler applied

#### C Computational Efficiency

See Table 2 in the main body for a detailed per-sample latency breakdown and discussion of parallelism and modularity.

GPU Utilization: For development and testing combined, we estimate 1,500 H100-hours were used during the creation of this paper. All experiments were conducted on 1×H100 80GB GPUs on a hosted Lambda cluster.

#### D Full Finetuning Experiments

As a strong baseline for MARVIS, we introduce a novel approach to LLM fine-tuning, projecting a sequence of positionally encoded TabPFNv2 embeddings and learned label tokens into the model’s token space. At inference time, we project the test element embedding from TabPFNv2 into the model’s token space and conduct standard autoregressive inference to acquire the predicted label.

#### D.1 Balanced Prefix Construction

We construct a balanced, few-shot prefix from training embeddings using prepare_tabpfn_embeddings_for_prefix. Given class labels 𝑦 and train embeddings 𝐸 ∈ R𝑁×𝑑 (after robust scaling and optional resizing), we select a total of num_few_shot_examples examples across classes, distributing as evenly as possible; short classes are repeated to meet demand. The resulting prefix tensor 𝑃 ∈ R𝑀×𝑑 (with class labels𝑐 ∈ {0, . . .,𝐾 −1}𝑀) is saved to prefix_data.npz.

#### D.2 Special Tokens and Class Tokens

We extend the tokenizer with two sentinel tokens <PREFIX_START> and <PREFIX_END> and with up to 10 class tokens <CLASS_i>. The underlying embedding matrix is resized accordingly. These token IDs delimit the region where external embeddings will be injected and provide stable referents for class-conditional evidence tokens.

#### D.3 Position-wise Projection into Token Space

Implementation. The core mechanism is implemented via QwenWithPrefixEmbedding:

- • A learnable projector is defined as Linear(d, H), mapping TabPFNv2 embedding dimension 𝑑 to the LLM hidden size 𝐻.
- • During forward, we build inputs_embeds from input_ids and locate the span between <PREFIX_START> and <PREFIX_END>. Let the number of available positions be 𝑇.
- • If embeddings and class labels are provided, we compute 𝑃˜ = 𝑃𝑊 +𝑏 ∈ R𝑀×𝐻 and interleave with class token embeddings: even positions receive projected vectors, odd positions the embeddings of <CLASS_{c_j}>, truncated to 𝑇.
- • If only embeddings are provided, we fill the𝑇 positions with 𝑃˜ contiguously.
- • The modified inputs_embeds are passed to the base model with input_ids=None. Rationale and soundness.

- (1) Representation Alignment. A learned affine map is the minimal adapter aligning TabPFN geometry to the LLM token manifold, akin to prefix/prompt-tuning adapters.
- (2) Token-Sequential Semantics. Injecting a bounded token span leverages positional mixing and attention for fusion with the downstream textual prompt; class-token interleaving ties directions in 𝑃˜ to discrete label anchors.
- (3) Identifiability. With only the projector and last 𝑘 layers unfrozen, gradients supervise a compact subspace, preserving language priors while enabling consistent task adaptation. Another parameter-efficient approach which we do not consider in this draft, LORA, would likely produce similar outcomes.

#### D.4 Backbone and Hooks

The default backbone is Qwen/Qwen2.5-3B-Instruct (configurable via –model_id). MARVIS prepares the model with prefix-embedding tokens and class tokens using prepare_qwen_with_prefix_embedding. Optional Vector Quantization (VQ) is available via prepare_qwen_with_vq_prefix_embedding.

#### D.5 Label Encoding

We encode labels with a LabelEncoder fitted on train+val+test labels per task; IDs index into the class token set. For float labels near-integral, we cast to integers; otherwise, regression handling is separate.

#### D.6 FFT Training Configuration

We train using train_llm_with_tabpfn_embeddings. Key elements:

- • Backbone freezing: Unfreeze the last 𝑘 layers (default 𝑘=1) and the projector; other layers frozen.
- • Loss: Cross-entropy over class-token targets in the output; attention integrates projected evidence with the prompt.
- • Optimization: Defaults: batch_size=8, grad_accum_steps=1, total_steps=2000, save_steps=500, lr=1e-4, mixup_alpha=0.0, early stopping (patience 30, threshold 0.4).
- • Prefix length: Template ensures enough positions between <PREFIX_*>; excess prefix entries are truncated.

Class token embeddings 𝐸𝑐

Qwen blocks (self-attn over span + prompt)

Insert between < 𝑃𝑅𝐸𝐹𝐼𝑋_𝑆𝑇𝐴𝑅𝑇 > and <𝑃𝑅𝐸𝐹𝐼𝑋_𝐸𝑁𝐷 >

Interleave [𝑝˜1,𝐸𝑐1,𝑝˜2,𝐸𝑐2, . . . ]

TabPFNv2 embeddings 𝑃 ∈ R𝑀×𝑑

Linear projector 𝑊 : R𝑑 → R𝐻

Projected 𝑃˜ ∈ R𝑀×𝐻

###### Figure 7: Projection and interleaving of TabPFNv2 embeddings into the LLM token space.

• W&B: Enabled with dated project names for versioning; run

names encode task/split.

#### D.7 FFT Evaluation Protocol

Evaluation is handled by examples/tabular/ evaluate_on_dataset_tabular.py with the unified --models interface. The orchestrator passes the saved model directory and, unless --no_baselines is set, appends all_baselines.

- • Test size limit: We commonly use –max_test_samples 200 to cap test evaluation for rapid iteration.
- • Feature selection threshold: –feature_selection_threshold can be forwarded for high-dimensional datasets.
- • Metrics and artifacts: Saved under each task/split evaluation directory and logged to W&B.

#### D.8 FFT Limitations and Discussion

While, for the sake of having strong reasonable baselines, we include this approach, we believe that in practice, it is not a suitable generalpurpose substitute for MARVIS.

- • Fine-tuning degrades chat performance. By changing the VLM’s vocabulary and last 𝑘 layers, we necessarily degrade chat performance somewhat; this weakens one of the major use cases for MARVIS.
- • Fine-tuning degrades interpretability. Because the VLM does not "know" it was fine-tuned on the data, nor does it "know" what it learned during fine-tuning, it cannot reason nearly as effectively about its own decision-making process, weakening another major use case for MARVIS.
- • Fine-tuning must be done again for every new dataset. This is an inconvenience as it requires the end user to maintain suitable training infrastructure on top of their pure inference infrastructure, which is generally more flexible.

#### E Extended Results E.1 Ablation Study on Context Choice Details

For a list of the methods we consider, please refer to Table 8.

Extended ablation studies reveal optimal configurations across different visualization strategies. We systematically evaluated four key approaches to understand how different types of information affect VLM spatial reasoning performance.

The configuration performance hierarchy demonstrates clear patterns:

- • tsne_perturbation_axes: 51.7% accuracy with uncertainty analysis
- • tsne_semantic_axes: 50.0% accuracy with meaningful class labels

- • tsne_knn: 48.3% accuracy with explicit neighbor information
- • basic_tsne: 45.0% accuracy as baseline approach

E.1.1 Analysis of Configuration Effects. The ablation results reveal several key insights about VLM spatial reasoning:

Perturbation-based Enhancement: The tsne_perturbation_axes configuration achieves the highest performance by incorporating uncertainty information through small perturbations around the query point. This provides the VLM with richer spatial context about decision boundaries and confidence regions.

Semantic Information Value: The tsne_semantic_axes approach shows strong performance by providing meaningful class labels within the visualization. This allows the VLM to leverage both spatial relationships and semantic understanding simultaneously.

Neighbor Information Benefits: The tsne_knn configuration demonstrates moderate improvements over the baseline by explicitly highlighting nearest neighbors, helping the VLM focus on locally relevant information.

Baseline Robustness: Even the basic_tsne approach achieves reasonable performance (45%), validating the fundamental effectiveness of the visual reasoning paradigm across modalities.

#### E.2 Ablation on MARVIS Backend and FFT

See Section 2.5 in the main body for the backend variants analysis and accuracy matrix.

#### F Extended Related Works

Early VLM architectures explored complex fusion mechanisms to achieve deep integration between vision and language. Flamingo [1] introduced gated cross-attention layers interleaved within frozen LLMs, enabling few-shot learning across diverse multimodal tasks without task-specific fine-tuning. BLIP [24] and its successor BLIP2 [25] pioneered the Multimodal Mixture of Encoder-Decoder (MED) architecture and introduced the Q-Former as a lightweight bridge between frozen vision encoders and language models. PaLI [7] established the principle of joint scaling, demonstrating that optimal VLM performance requires balanced scaling of all components: vision models, language models, and training data.

LLaVA [28] democratized VLM research by establishing an efficient, open-source blueprint. Its three-component architecture—frozen vision encoder, lightweight MLP projector, and frozen LLM—with two-stage training (feature alignment followed by instruction tuning) proved that simple architectures could achieve impressive multimodal capabilities. LLaVA-NeXT [27] introduced dynamic high resolution through intelligent image partitioning, while mPLUG-Owl2 [56] developed Modality-Adaptive Modules to

###### Category Method Description

basic_tsne Standard t-SNE visualization with default parameters

tsne_3d Three-dimensional t-SNE visualization for enhanced

Basic Visualizations

spatial understanding

tsne_high_dpi High-resolution t-SNE with increased image quality

tsne_high_perplexity t-SNE with modified perplexity parameter for different

clustering

tsne_knn t-SNE with k-nearest neighbor information overlay

tsne_perturbation_axes t-SNE with perturbation analysis for uncertainty quan-

Enhanced Single Methods

tification

tsne_semantic_axes t-SNE with semantic class labels and axes descriptions tsne_3d_knn 3D t-SNE visualization with k-NN connections dis-

played

tsne_3d_perturbation 3D t-SNE with perturbation analysis for spatial uncer-

tainty

multi_comprehensive PCA + t-SNE + Spectral + Isomap comprehensive view

multi_pca_tsne Combined PCA and t-SNE dual visualization

multi_pca_tsne_spectral Triple visualization: PCA + t-SNE + Spectral embedding multi_linear_nonlinear Linear and nonlinear dimensionality reduction compar-

Multi-Visualization Methods

ison

multi_local_global Local and global structure preservation methods

multi_with_umap Multi-method visualization including UMAP

multi_grid_layout Grid-based layout for systematic method comparison

decision_regions_svm SVM decision boundary visualization with regions

Specialized Methods

frequent_patterns Pattern mining visualization for feature relationships

metadata_comprehensive Metadata-enhanced comprehensive visualization ap-

proach

Table 8: MARVIS Method Variants Overview. Comprehensive summary of visualization approaches evaluated in ablation studies, categorized by methodology type and complexity level.

foster positive cross-modal collaboration while mitigating interference. POINTS [33] exemplified sophisticated data curation through perplexity-based filtering.

Recent work has pushed beyond conversational capabilities toward precise, spatially-grounded understanding, key to understanding the gains in MARVIS. Grounding DINO [30] achieved open-set object detection through text-conditioned spatial understanding, while KOSMOS-2 [38] integrated coordinate tokens directly into the LLM vocabulary for grounded text generation. OtterHD [23] pioneered an encoder-less architecture, processing raw pixel patches directly in the LLM to eliminate resolution constraints. SleighVL [26] refined high-resolution processing through attention-based subimage weighting via Global Semantic-guided Weight Allocation. Emu3 [52] unifies vision and language modalities under next-token prediction, tokenizing images, videos, and text into a shared vocabulary space. Molmo [9] champions fully open ecosystems with

human-annotated data, breaking dependence on proprietary synthetic datasets. Early cross-modal strategies used feature concatenation, attention mechanisms, or late fusion strategies, requiring extensive retraining for each new modality [3]. Modern paradigms include contrastive learning (CLIP-style) [41], generative modeling [43], and instruction tuning [53]. However, these approaches typically require substantial computational resources and domainspecific training data for each new modality.

#### G Deep Dive: Tabular Modality Analysis

This section provides a comprehensive analysis of MARVIS performance on tabular data, evaluating both classification and regression tasks against established baselines. The analysis includes detailed performance metrics, correlation studies with TabPFN v2, and critical difference plots for statistical comparison.

[Figure 5]

- Figure 8: Configuration Performance Heatmap. Detailed breakdown showing performance variations across different parameter combinations and visualization strategies. Darker regions indicate higher accuracy, with perturbation-based methods consistently showing superior performance across various settings.

#### G.1 Baselines: JOLT and TabLLM

One challenge we faced during the creation of this paper is that prior work which utilized LLMs for tabular classification and regression lacked both standard benchmarks and consistent, easy to implement methods. As a secondary contribution, we release comprehensive full-size tabular benchmarks which include semantic information (see H), and modern, feature-complete implementations of TabLLM and JOLT.

Dual Implementation Architecture: We developed a sophisticated dual-path architecture that supports both legacy compatibility and modern framework integration. Our implementation includes:

- • Legacy Integration: Direct incorporation of original JOLT codebase with automatic fallback mechanisms
- • Modern Implementation: Complete HuggingFace transformers integration with VLLM backend support
- • Unified Model Loader: Centralized model management supporting multiple backends (HuggingFace, VLLM, OpenAI, Gemini)

Memory Optimization and Scalability: Critical for production deployment, our implementation includes:

- • Gradient checkpointing with KV cache disabling for memory efficiency
- • Dynamic batch sizing with automatic Out-of-Memory (OOM) recovery
- • Aggressive memory limits for regression tasks (512MB default)
- • Feature dropping with retry mechanisms for large datasets

Enhanced Task Support: Beyond the original classification focus, we extended JOLT to support:

- • Full regression pipeline with intelligent binning strategies
- • Automatic task type detection and configuration
- • Balanced few-shot example selection algorithms
- • Context-aware prompt truncation for varying model context lengths

Configuration Management: We developed a comprehensive metadata system:

- • Automatic JOLT configuration discovery by OpenML task ID
- • Feature count validation ensuring dataset-configuration alignment
- • Semantic feature mapping from original to descriptive names

• Graceful degradation when configurations are unavailable TabLLM Implementation

Real-time Note Generation: Our TabLLM implementation eliminates the need for pre-generated note banks through:

- • On-the-fly natural language description generation
- • Dynamic semantic feature expansion matching actual dataset characteristics
- • Template-based prompt generation with YAML configuration support
- • Automatic feature alignment verification post-preprocessing

Multi-Backend API Support: We created a unified interface supporting:

- • OpenAI API integration (GPT-4, GPT-3.5-turbo, GPT-4o)
- • Google Gemini API support with automatic model selection
- • Local model deployment via HuggingFace transformers
- • Automatic backend detection based on model naming conventions

Quality Assurance Mechanisms: To ensure generation quality, we implemented:

- • Inspection system saving sample generated notes for manual review
- • N-gram analysis for content validation and diversity assessment
- • Context truncation with intelligent few-shot example selection
- • Template validation ensuring prompt completeness

###### HuggingFace Ecosystem Compatibility

Both implementations leverage the complete HuggingFace ecosystem:

- • AutoModelForCausalLM and AutoTokenizer for model loading
- • Trust remote code support for cutting-edge models
- • Automatic device placement and memory optimization
- • Support for quantized models (8-bit, 4-bit) through BitsAndBytes

###### VLLM Integration

For production deployments requiring high throughput:

- • Automatic VLLM backend selection for compatible models
- • Tensor parallelism configuration for multi-GPU deployment
- • Optimized sampling parameters with fallback to transformers
- • Unified generation interface across backends

###### Benchmark Integration

Our implementations integrate seamlessly with standard evaluation frameworks:

- • Direct OpenML dataset loading and preprocessing
- • Standardized evaluation interface compatible with scikitlearn
- • Comprehensive metrics calculation (accuracy, F1, ROC-AUC, R2, MAE, MSE)
- • Weights & Biases integration for experiment tracking

###### Usage and Accessibility

Our implementations provide simple, unified interfaces:

# JOLT evaluation with local model python examples/tabular/evaluate_llm_baselines_tabular.py \

--models jolt \

--dataset_ids 23 \

--jolt_model Qwen/Qwen2.5-7B-Instruct

# TabLLM evaluation with API backend python examples/tabular/evaluate_llm_baselines_tabular.py \

--models tabllm \

--dataset_ids 1590 \

--openai_model gpt-4o

This unified interface abstracts away implementation complexity while providing extensive configuration options for advanced users.

#### G.2 Classification Performance on OpenML CC18

The OpenML CC18 benchmark represents one of the most comprehensive evaluation suites for tabular classification, consisting of 72 carefully curated datasets [5].

Model Mean Acc. Bal. Acc. F1 Macro Datasets

MARVIS 84.5% 80.2% 79.9% 69 TabPFN v2 87.8% 82.2% 82.3% 66 CatBoost 87.0% 81.5% 81.8% 70 Random Forest 86.5% 80.3% 81.0% 70 Gradient Boosting 85.4% 79.5% 79.9% 70 Logistic Reg. 82.5% 74.8% 75.0% 70 TabLLM (Gemini) 50.1% 44.3% 40.2% 69 TabLLM (Qwen) 42.9% 36.5% 30.9% 69 JOLT 41.0% 33.9% 27.3% 67

Table 9: Classification Performance on OpenML CC18. MARVIS achieves competitive performance with traditional ML methods while significantly outperforming other LLM-based approaches. Performance metrics include mean accuracy, balanced accuracy for handling class imbalance, and F1 macro for multi-class evaluation.

Key insights from classification analysis:

- • MARVIS achieves 84.5% mean accuracy, placing it competitively among traditional ML methods
- • Strong performance on balanced accuracy (80.2%) demonstrates effective handling of class imbalance
- • Significantly outperforms other LLM-based approaches (TabLLM, JOLT) by 34-44 percentage points
- • Consistent performance across diverse dataset types with low variance (𝜎 = 15.1%)

#### G.3 Regression Performance Analysis

For regression tasks, MARVIS was evaluated on a custom benchmark of 43 regression datasets spanning diverse domains and characteristics.

#### G.4 Correlation Analysis with TabPFN v2

A detailed correlation analysis between MARVIS and TabPFN v2 reveals interesting patterns in their complementary strengths and failure modes.

Key correlation insights:

[Figure 6]

###### Figure 9: Critical Difference Plot for Classification Performance. Statistical analysis using balanced accuracy across OpenML CC18 datasets. Connected algorithms have no statistically significant difference (p ≥ 0.05) using the Nemenyi post-hoc test. MARVIS ranks competitively among traditional ML methods and significantly outperforms other LLM approaches.

[Figure 7]

###### Figure 10: Classification Performance Matrix Heatmap. Dataset-wise performance comparison showing MARVIS consistency across different types of tabular classification tasks. Each row represents a dataset, and each column represents an algorithm. Darker colors indicate higher balanced accuracy scores.

- • High Classification Alignment: 0.978 Pearson correlation indicates both methods excel on similar classification tasks
- • Moderate Regression Correlation: 0.884 correlation suggests more divergent strengths in regression domain

• Complementary Performance: Datasets where one method fails often correspond to failures in the other, suggesting systematic challenges rather than method-specific weaknesses

###### Algorithm Mean R2 Median R2 MAE RMSE

Random Forest 0.586 0.644 0.184 0.298 TabPFN v2 0.585 0.623 0.187 0.301 Gradient Boosting 0.564 0.615 0.191 0.304 Linear Regression 0.538 0.588 0.203 0.318 MARVIS 0.532 0.576 0.198 0.312 LightGBM 0.519 0.567 0.201 0.321 XGBoost 0.487 0.534 0.218 0.342

- Table 10: Regression Performance Summary. MARVIS achieves competitive R2 scores (0.532 mean, 0.576 median) ranking 5th among 7 algorithms. While R2 scores are moderate, MARVIS shows strong performance in error metrics (MAE, RMSE), indicating consistent prediction quality.

[Figure 8]

Figure 11: Critical Difference Plot for Regression Performance. Statistical comparison using R2 scores across 43 regression datasets. MARVIS demonstrates statistically competitive performance with traditional methods, ranking in the middle tier without significant differences from top performers.

Task Type Pearson r Spearman 𝜌 Kendall 𝜏 Datasets

Classification 0.978 0.945 0.823 65 Regression 0.884 0.867 0.698 41

- Table 11: MARVIS-TabPFN v2 Correlation Summary. Strong positive correlations indicate that both methods tend to perform well on similar datasets, suggesting complementary rather than competing approaches. The high classification correlation (0.978) demonstrates particularly aligned performance patterns.

• Consistent Rankings: High Spearman correlations (0.945 classification, 0.867 regression) show similar relative performance orderings

#### G.5 Analysis and Discussion

The comprehensive tabular analysis reveals several important findings about MARVIS performance in structured data domains:

Competitive Classification Performance: MARVIS achieves strong results on OpenML CC18, demonstrating that visual reasoning approaches can effectively handle tabular classification tasks. The 84.5% accuracy places MARVIS within the competitive range of traditional ML methods.

Moderate Regression Capabilities: With 0.532 mean R2 on regression tasks, MARVIS shows reasonable but not exceptional regression performance. This suggests the visual reasoning paradigm may be better suited for discrete classification decisions than continuous value prediction.

Strong LLM Baseline Performance: MARVIS significantly outperforms other LLM-based tabular methods (TabLLM, JOLT), validating the effectiveness of the visual reasoning approach compared to direct tabular-to-text conversion strategies.

Complementary Method Profile: The high correlation with TabPFN v2 suggests MARVIS and traditional tabular methods have similar strengths and weaknesses, making MARVIS a viable alternative rather than a replacement for existing approaches.

Scalability Considerations: MARVIS maintains consistent performance across the diverse OpenML CC18 collection, suggesting good generalization properties across different tabular data characteristics and domains.

#### H CC18-Semantic and Regression2025-Semantic: Semantic Metadata Generation for Enhanced Dataset Understanding

A key component of our tabular analysis involved the creation of comprehensive semantic metadata for both classification (cc18_semantic) and regression (regression_semantic) datasets. This process, conducted using Claude Research from Anthropic with human review, represents a significant advancement in dataset documentation and understanding.

#### H.1 Motivation and Scope

Traditional machine learning benchmarks often lack rich semantic context about feature meanings, target interpretations, and domainspecific knowledge. To address this limitation, we developed a systematic approach to generate comprehensive semantic metadata for:

- • CC18 Classification Tasks: 72 datasets from the OpenML CC18 benchmark suite
- • Regression Tasks: 41 carefully selected regression datasets from OpenML
- • Total Coverage: 113 datasets with comprehensive semantic enrichment

#### H.2 Semantic Metadata Generation Algorithm

The semantic metadata generation process follows a multi-stage pipeline designed to ensure accuracy, comprehensiveness, and consistency across all datasets.

#### H.3 Semantic Enrichment Structure

The generated metadata follows a standardized schema that captures multiple dimensions of dataset understanding:

[Figure 9]

###### Figure 12: Regression Performance Matrix Heatmap. Dataset-wise R2 score comparison showing MARVIS performance patterns across different regression tasks. The visualization reveals strengths in certain problem types while highlighting areas for potential improvement.

[Figure 10]

###### Figure 13: MARVIS vs TabPFN v2 Classification Correlation. Scatter plot showing strong positive correlation (r = 0.978) between MARVIS and TabPFN v2 balanced accuracy scores across OpenML CC18 datasets. Points above the diagonal line indicate datasets where MARVIS outperforms TabPFN v2.

Feature-Level Enrichment: Each feature receives comprehensive semantic description including domain context, technical interpretation, data type classification, and relationship analysis to the prediction task.

Target Variable Analysis: For classification tasks, detailed explanations of class meanings and real-world interpretation. For regression tasks, units of measurement, typical ranges, and practical significance guidelines.

Historical and Methodological Context: Dataset provenance including original creators, institutions, collection methodology, domain applications, and ethical considerations.

[Figure 11]

Figure 14: MARVIS vs TabPFN v2 Regression Correlation. Scatter plot showing moderate positive correlation (r = 0.884) between MARVIS and TabPFN v2 R2 scores across regression datasets. The correlation suggests similar strengths but with more divergent performance patterns compared to classification tasks.

Example Semantic Enhancement:

Feature: "bkblk" (Chess Kr-vs-Kp dataset) Basic metadata: Binary feature (t/f) Semantic enhancement: "Whether the black king is blocked from moving to certain squares. In chess endgame analysis, this represents a critical positional constraint that affects the feasibility of defensive strategies and directly influences whether White can force a win from the current position."

Algorithm 1 Semantic Metadata Generation Pipeline

- 1: Input: OpenML dataset ID, basic task information
- 2: Output: Comprehensive semantic metadata JSON
- 3:
- 4: Stage 1: Data Source Integration
- 5: Query OpenML API for basic dataset information
- 6: Extract feature names, data types, target variables, and statistics
- 7: Collect dataset provenance and publication information
- 8:
- 9: Stage 2: Claude Research Process
- 10: Initialize Claude 3.5 Sonnet with domain expertise prompt
- 11: Instruct comprehensive multi-source research covering:
- 12: • Original dataset publications and creators
- 13: • Domain-specific knowledge bases
- 14: • Academic literature and citations
- 15: • UCI ML Repository and similar sources
- 16:
- 17: Stage 3: Structured Semantic Analysis
- 18: for each feature in dataset do
- 19: Generate semantic description with domain context
- 20: Classify data type and measurement characteristics
- 21: Explain relationship to prediction task
- 22: end for
- 23:
- 24: Stage 4: Target Variable Enhancement
- 25: if classification task then
- 26: Describe meaning of each class label
- 27: Provide real-world interpretation guidelines
- 28: else
- 29: Explain target variable units and ranges
- 30: Describe practical significance of values
- 31: end if
- 32:
- 33: Stage 5: Quality Assurance
- 34: Apply low temperature (0.1) for factual consistency
- 35: Include uncertainty acknowledgments where appropriate
- 36: Validate JSON structure and completeness
- 37: Enable human review and verification process

#### H.4 Multi-Source Research Methodology

The Claude Research process integrates information from multiple authoritative sources to ensure accuracy and comprehensiveness:

- • Primary Sources: Original dataset publications, creator documentation, and institutional repositories
- • Academic Literature: Peer-reviewed papers utilizing the datasets, domain-specific research
- • Repository Documentation: UCI ML Repository, OpenML detailed descriptions, Kaggle dataset pages
- • Domain Databases: Specialized knowledge bases relevant to specific application areas
- • Cross-Validation: Multiple source verification to ensure factual accuracy

#### H.5 Quality Assurance and Validation

The semantic metadata generation incorporates multiple layers of quality control:

Algorithmic Validation: Automated scripts verify JSON structure completeness, field presence patterns, and schema compliance across all datasets.

Coverage Analysis: Systematic review ensures all required metadata fields are populated and coverage gaps are identified for remediation.

Human Review Integration: The process includes explicit uncertainty acknowledgment when information sources are limited, enabling targeted human verification.

Standardization Pipeline: Automated standardization scripts consolidate different metadata formats into a universal schema while preserving original information and implementing backup systems.

#### H.6 Comprehensive Dataset Characterization

This section provides detailed characterization of the datasets used in our tabular modality analysis, covering both the OpenML CC18 classification benchmark and the Regression 2025 benchmark suite.

- H.6.1 Domain Distribution Analysis. The benchmark collections span diverse application domains, providing comprehensive coverage of real-world machine learning challenges.

Domain CC18 Count Reg. Count Total

Vision 27 4 31 Medical 7 7 14 Biology 5 2 7 Finance 4 3 7 Games 4 1 5 NLP 3 3 6 Science/Eng. 0 2 2 Social 0 1 1 Other 22 18 40

Total 72 41 113

Table 12: Domain Distribution Across Benchmark Collections. The datasets span nine major application domains, with Vision being the most represented (31 datasets), followed by Medical (14 datasets). The “Other” category includes diverse applications such as telecommunications, manufacturing, and environmental monitoring.

- H.6.2 Representative Dataset Examples. OpenML CC18 Classification Tasks. Please refer to Table 13. Regression 2025 Tasks. Please refer to Table 14.
- H.6.3 Dataset Complexity Analysis. The benchmark collections exhibit significant diversity in complexity characteristics:

###### Feature Dimensionality Range:

• Low-dimensional (≤ 10 features): 29 datasets (25.7%)

###### Dataset Domain Features Classes Description

MiceProtein Biology 77 8 Mouse protein expression levels for Down syndrome study dna Biology 1 3 Molecular biology DNA sequence classification splice Biology 1 3 Primate splice-junction gene sequences analysis bank-marketing Finance 16 2 Portuguese banking institution marketing campaigns credit-g Finance 20 2 German credit risk assessment dataset adult Finance 14 2 Census income prediction (≥50K annual income) connect-4 Games 3 3 Connect-4 game position evaluation kr-vs-kp Games 36 2 Chess King+Rook vs King+Pawn endgame positions tic-tac-toe Games 9 2 Tic-tac-toe game board position analysis breast-w Medical 9 2 Wisconsin breast cancer diagnosis heart-statlog Medical 13 2 Heart disease diagnosis from clinical parameters diabetes Medical 8 2 Pima Indian diabetes onset prediction Devnagari-Script Vision 1024 46 Handwritten Devanagari character recognition mnist_784 Vision 784 10 Handwritten digit recognition benchmark Fashion-MNIST Vision 784 10 Fashion article classification from images

###### Table 13: Representative CC18 Classification Datasets. Examples spanning major domains show the diversity of tabular classification challenges, from biological sequence analysis to game strategy evaluation and medical diagnosis.

Dataset Domain Features Target Description

QSAR_Bioconcentration Biology 13 Bioconcentration factor for environmental chemistry SGEMM_GPU_kernel Biology 10 GPU kernel performance optimization metrics climate_change_impact Finance 15 Agricultural productivity under climate change world_food_wealth Finance 6 Global food security and economic indicators Violent_Crime_County Finance 6 County-level violent crime rates (1975-2016) medical_charges Medical 4 Healthcare insurance charges prediction heart_failure_records Medical 13 Clinical parameters for heart failure prediction particulate-matter Medical 7 Air quality PM2.5 concentration levels UCC_Comments Medical 7 Health impact assessment from social media housing_prices_2020 Other 9 Real estate price prediction modeling cpu_performance Other 7 Computer hardware performance benchmarking auto_mpg Other 8 Vehicle fuel efficiency prediction wine_quality Other 11 Wine quality assessment from chemical properties concrete_strength Science/Eng 8 Concrete compressive strength from mixture sulfur_recovery Science/Eng 6 Industrial sulfur recovery process optimization

###### Table 14: Representative Regression Datasets. Examples demonstrate the breadth of continuous prediction tasks, from environmental monitoring and healthcare analytics to industrial process optimization and consumer applications.

- • Medium-dimensional (11-50 features): 51 datasets (45.1%)
- • High-dimensional (≥ 50 features): 33 datasets (29.2%)

###### Classification Complexity:

- • Binary classification: 48 datasets (66.7% of CC18)
- • Multi-class (3-10 classes): 21 datasets (29.2% of CC18)
- • High-class (≥ 10 classes): 3 datasets (4.1% of CC18)

###### Domain-Specific Characteristics:

- • Vision datasets: Typically high-dimensional (784-1024 features) with balanced class distributions
- • Medical datasets: Often feature moderate dimensionality (8-20 features) with clinical interpretability requirements
- • Financial datasets: Characterized by mixed data types and class imbalance considerations
- • Game datasets: Show discrete feature spaces with strategic decision-making patterns

• Biology datasets: Range from sequence data (low-

dimensional) to protein expression (high-dimensional)

#### I VLM Reasoning Analysis

This section extends the reasoning quality analysis in the main body (Table 6) with detailed evidence that Vision-Language Models engage in genuine adaptive reasoning when processing MARVIS visualizations. We examine reasoning traces, disagreement patterns, and method-specific behavioral signatures.

#### I.1 Comprehensive Reasoning Pattern Analysis

Several findings argue against simple pattern matching explanations:

• Method-specific reasoning adaptation: Different visualization types elicit systematically different reasoning approaches

- • Performance-quality correlation: Better reasoning correlates with higher accuracy across diverse test cases
- • Quantitative analysis emergence: Numerical reasoning appears precisely when relevant information is provided
- • Logical consistency within methods: Each approach maintains internal logical coherence while differing from others

The evidence suggests VLMs possess genuine spatial reasoning capabilities that can be effectively leveraged through appropriate visualization design:

- • Color-space integration: Systematic use of color information for class identification
- • Distance relationship understanding: Quantitative analysis of spatial proximity when information is available
- • Cluster structure recognition: Identification of grouping patterns in embedding spaces
- • Multi-modal information synthesis: Integration of spatial, semantic, and quantitative information

#### I.2 Adaptive Reasoning Evidence

The quantitative analysis of reasoning quality vs. accuracy and concrete reasoning trace examples are presented in the main body (see Table 6 and the accompanying discussion in the Findings section).

#### J MARVIS Extended Results

In Table 15, we present the comprehensive results for all models on all benchmarks.

- Table 15: Comprehensive Performance Results Across Multiple Domains. Evaluation of various methods on vision, audio, biological, and tabular benchmarks. MARVIS demonstrates competitive performance across all domains, achieving near state-of-the-art results while using a unified approach. Success rates are 100% for all methods except JOLT on regression tasks (90.3%).

Domain Benchmark Method Backend Metric Value Vision

CIFAR-10 Conventional Gemini-Flash-2.0 Accuracy 85.7 CIFAR-100 Conventional Gemini-Flash-2.0 Accuracy 64.3 CIFAR-10 Conventional Qwen 2.5 VL 3B Accuracy 83.2 CIFAR-100 Conventional Qwen 2.5 VL 3B Accuracy 51.0 CIFAR-10 KNN DinoV2-ViT-L-14-reg Accuracy 99.0 CIFAR-100 KNN DinoV2-ViT-L-14-reg Accuracy 91.6 CIFAR-10 MARVIS MARVIS-3B Accuracy 98.0 CIFAR-100 MARVIS MARVIS-3B Accuracy 88.0

Audio

ESC-50 KNN Whisper-Large Accuracy 76.0 RAVDESS KNN Whisper-Large Accuracy 47.9 UrbanSound-8K KNN Whisper-Large Accuracy 65.9 ESC-50 Contrastive CLAP Accuracy 90.5 RAVDESS Contrastive CLAP Accuracy 21.8 UrbanSound-8K Contrastive CLAP Accuracy 77.1 ESC-50 MARVIS MARVIS-3B Accuracy 91.3 RAVDESS MARVIS MARVIS-3B Accuracy 38.4 UrbanSound-8K MARVIS MARVIS-3B Accuracy 79.8

Biological

FishNet Conventional Qwen 2.5 VL 3B Accuracy 17.3 AWA2 Conventional Qwen 2.5 VL 3B Accuracy 92.6 PlantDoc Conventional Qwen 2.5 VL 3B Accuracy 37.3 FishNet Conventional Gemini-Flash-2.0 Accuracy 59.5 AWA2 Conventional Gemini-Flash-2.0 Accuracy 96.5 PlantDoc Conventional Gemini-Flash-2.0 Accuracy 74.2 FishNet KNN BioClip2 Accuracy 83.7 AWA2 KNN BioClip2 Accuracy 97.1 PlantDoc KNN BioClip2 Accuracy 72.0 FishNet MARVIS MARVIS-3B Accuracy 80.2 AWA2 MARVIS MARVIS-3B Accuracy 95.7 PlantDoc MARVIS MARVIS-3B Accuracy 67.4

Tabular Classification

CC-18 (Semantic) JOLT Qwen 2.5 3B Accuracy 41.2 CC-18 (Semantic) TabLLM Qwen 2.5 3B Accuracy 42.9 CC-18 (Semantic) TabLLM Gemini-Flash-2.0 Accuracy 50.1 CC-18 (Semantic) Conventional TabPFNv2 Accuracy 87.8 CC-18 (Semantic) MARVIS MARVIS-3B Accuracy 84.5 CC-18 (Semantic) Conventional Random Forest Accuracy 86.5 CC-18 (Semantic) Conventional Logistic Reg. Accuracy 82.5 CC-18 (Semantic) Conventional CatBoost Accuracy 87.0

Tabular Regression

Reg. 2025 (Semantic) Conventional TabPFNv2 Avg R2 (0–100) 66.9 Reg. 2025 (Semantic) Conventional CatBoost Avg R2 (0–100) 71.4 Reg. 2025 (Semantic) JOLT Qwen 2.5 3B Avg R2 (0–100) 05.1 Reg. 2025 (Semantic) MARVIS MARVIS-3B Avg R2 (0–100) 66.0 Reg. 2025 (Semantic) Conventional Linear Model Avg R2 (0–100) 51.2 Reg. 2025 (Semantic) Conventional Random Forest Avg R2 (0–100) 72.8

#### K MARVIS Visualization Gallery

This section presents visualizations from the MARVIS framework applied to tabular datasets.

#### K.1 CMC Dataset

KNN Visualization

📊 tsne_kn

📋 Property 📊 Value

Method Name tsne_knn

Prompt Length 1,776 characters

Response Length 299 characters

Visualizations 3 images (showing top 3)

##### 🤖 System Prompt

Looking at thisenhanced t-SNE visualization of tabular data, you can see:

- 1. Colored points representing training data, where each color corresponds to a different class
- 2. Graysquare points representing 🧪 test data
- 3. One red ⭐ star pointwhich is the 🎯 query pointI want you to classify
- 4. A pie chart showing the distribution of the 5 nearest neighbors by class
- 5. The pie chart includes class counts, percentages, and average distances to neighbors

Class Legend:

- - 🏷 Clas 0: Blue RGB(30, 19, 181)

- - 🏷 Clas 1: Orange RGB(25, 127, 12)

- - 🏷 Clas 2: Gren RGB(43, 160, 43)

- - 🧪 Test points: Light Gray RGB(21, 21, 21)

K-NN Analysis (k=5):

- • 🏷 Clas 0:3 neighbors (60%), AvgDist: 8.0
- • 🏷 Clas 1:1 neighbors (20%), AvgDist: 5.1
- • 🏷 Clas 2:1 neighbors (20%), AvgDist: 9.0

Dataset Context: Tabular data embedded using appropriate features IMPORTANT: The pie chart shows the class distribution of the 5 nearest neighbors found in the original high-dimensional embedding space, NOT just based on the 2D visualization space. Smaller average distances indicate higher similarity.

Based on BOTH the spatial position in the t-SNE visualization AND the explicit nearest neighbor connections, which class should this 🎯 query pointbelong to? The available classes are: "Class_0", "Class_1", "Class_2"

Consider:

- - The spatial clustering patterns in the t-SNE visualization
- - Which classes the nearest neighbors (connected by red lines) belong to
- - The relative importance of close neighbors (thicker lines)

Please respond with just the class label (e.g., "Class_0", "Class_1", "Class_2") followed by a brief explanation of your reasoning based on the spatial clustering AND the pie chart neighbor analysis.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]"

💬 Example Response

Class: Class_0 | Reasoning: The 🎯 query pointis closer to the cluster of Class_0 neighbors (3 neighbors) compared to Class_1 and Class_2 neighbors. Additionally, the average distance to Class_0 neighbors (8.0) is lower than to Class_1 (5.1) and Class_2 (9.0), indicating higher similarity to Class_0.

##### 🎨 Visualizations

[Figure 12]

- 📈 visualization_4.png

[Figure 13]

- 📈 visualization_5.png

[Figure 14]

###### 📈 visualization_test_018.png

###### Semantic Axes

📊 tsne_semantic_axes

📋 Property 📊 Value

Method Name tsne_semantic_axes

Prompt Length 1,294 characters

Response Length 309 characters

Visualizations 3 images (showing top 3)

##### 🤖 System Prompt

Looking at this t-SNE visualization of tabular data, you can see:

- 1. Colored points representing training data, where each color corresponds to a different class
- 2. Graysquare points representing 🧪 test data
- 3. One red ⭐ star pointwhich is the 🎯 query pointI want you to classify

Class Legend:

- - No-use: Blue RGB(30, 19, 181)

- - Long-term methods: Orange RGB(25, 127, 12)

- - Short-term methods: Gren RGB(43, 160, 43)

- - 🧪 Test points: Light Gray RGB(21, 21, 21)

Semantic Axis Interpretation:

- • X-axis (39.3% var): +Living standard (1=low, 2, 3, 4=high)
- • Y-axis (15.0% var): Mixed factors

Dataset Context: Tabular data embedded using appropriate features

Based on the position of the redstar (🎯 query point) relative to the colored training points, which class should this 🎯 query pointbelong to? The available classes are: "No-use", "Long-term methods", "Short-term methods"

Consider:

- - The spatial relationships in the t-SNE visualization
- - Which colored class clusters the redstar is closest to or embedded within

Please respond with just the class label (e.g., "No-use", "Long-term methods", etc.) followed by a brief explanation of your reasoning based on the spatial clustering patterns you observe.

Format your response as: "Class: [class_label] | Reasoning: [brief explanation]"

💬 Example Response

Class: Long-term methods | Reasoning: The redstar (🎯 query point) is positioned closer to the cluster of orange-colored points, which represent the "Long-term methods" class. This spatial proximity suggests that the 🎯 query pointis most similar to the "Long-term methods" class in the given t-SNE visualization.

🎨 Visualizations

[Figure 15]

###### 📈 visualization_4.png

[Figure 16]

###### 📈 visualization_5.png

[Figure 17]

###### 📈 visualization_test_018.png

#### Artifact Appendix Abstract

This artifact contains the source code, datasets, scripts, and selected raw experimental outputs needed to reproduce the central results of MARVIS. The code is a self-contained Python package implementing the MARVIS pipeline (DINOv2 / TabPFNv2 / Whisper / CLAP / BioCLIPv2 embeddings → t-SNE projection with KNN overlay → Qwen2.5-VL-3B reasoning) for classification and regression across tabular, vision, audio, and biological modalities. The artifact includes (i) the canonical evaluation entry points (examples/tabular,vision,audio/evaluate_*.py), (ii) two novel benchmarks introduced in the paper (CC18-Semantic and Regression2025-Semantic), (iii) a smoke test that runs end-toend in ∼10 minutes on a single consumer GPU, and (iv) a reduced reproduction recipe for the inductive-only t-SNE ablation in Table 7 that runs in ∼2.5 hours on a single H100. Larger raw outputs (per-sample VLM responses for the main result table) are hosted on Hugging Face; the code repository is hosted on GitHub.

#### Artifact check-list (meta-information)

- • Algorithm: MARVIS — multi-modal classification/regression via VLM reasoning over t-SNE visualizations of pretrained embeddings.
- • Program: Python 3.11+ package marvis.
- • Compilation: pip install -e ".[vision,audio,api]"; no native compilation required.
- • Models: DINOv2-ViT-L/14-reg (vision), Whisper-large-v2 / Microsoft CLAP (audio), BioCLIPv2 (biological), TabPFNv2 (tabular), Qwen2.5-VL-3B-Instruct (VLM reasoning). All retrieved automatically from Hugging Face / torch.hub on first run.
- • Data sets: CC18-Semantic and Regression2025-Semantic (released with this artifact); CIFAR-10/100 (torchvision); ESC50, RAVDESS, UrbanSound8K (audio benchmarks); fishnet, awa2, plantdoc (biological benchmarks).
- • Run-time environment: Linux x86_64 or aarch64. Tested on Ubuntu 22.04 with CUDA 12.x and on macOS 14+ (Apple Silicon; transformers backend, MPS acceleration). VLLM optional and used automatically when available.
- • Hardware: For the smoke test, any consumer NVIDIA GPU with ≥8 GB VRAM (or Apple Silicon with ≥16 GB unified memory; transformers backend). For the inductive-ablation reproduction, one NVIDIA H100 (or comparable) is recommended. For the full benchmark suite (Table 1 in the paper), ∼1,500 H100-hours total.
- • Metrics: Classification accuracy (top-1), regression 𝑅2, endto-end and per-sample latency.
- • Output: Per-task accuracy/𝑅2 summaries (*_test_results.json, *_test_summary.csv), persample VLM prompts and responses, optional saved scatter-plot visualizations.
- • Experiments: Driven by shell-callable scripts in examples/ and marvis_scripts/; a Slurm sbatch script for the inductive ablation is included.
- • How much disk space required: 50 GB for the smoke test (includes pretrained model weights). 100–150 GB for the

- inductive ablation reproduction (adds CIFAR-10 image cache and per-sample plots/responses). ∼500 GB if mirroring all hosted result tarballs.
- • How much time is needed to prepare workflow? ∼30 minutes (clone, install, download model weights on first run).
- • How much time is needed to complete experiments? ∼10 minutes for the smoke test, ∼2.5 hours for the inductiveonly t-SNE ablation reproduction (1 H100), days–weeks for the full benchmark sweep.
- • Publicly available? Yes. Code on GitHub; large result archives on Hugging Face Datasets; an archived release tag (DOI via Zenodo) is provided for the camera-ready commit.
- • Code licenses: MIT.
- • Data licenses: Released artifacts (CC18-Semantic, Regression2025-Semantic) are CC-BY 4.0. Third-party datasets retain their original licenses (e.g., CIFAR is MIT-style; OpenML CC18 follows OpenML terms).
- • Workflow framework used? No. Plain Python entry points and Slurm sbatch.
- • Archived (provide DOI)? GitHub release v0.1.0-cais2026 archived on Zenodo: <DOI to be filled in upon Zenodo deposit>.

#### Description

How to access.

- • Code (small, mutable): https://github.com/penfever/ marvis, tag v0.1.0-cais2026.
- • Code (archived, immutable): Zenodo deposit, DOI <TBD>.
- • Hosted datasets and result archives: https: //huggingface.co/datasets/penfever/marvis (CC18-Semantic, Regression2025-Semantic, and per-sample VLM outputs for the main results table).
- • Reviewer entry point: the README in the GitHub repository contains a section labeled “Artifact Reviewers — Start Here” that points to the smoke test and the inductive-ablation reproduction recipe.

Hardware dependencies. A CUDA-capable GPU is recommended. The full benchmark sweep was conducted on H100 80GB nodes. The smoke test runs on any GPU with ≥8 GB VRAM and on Apple Silicon (Mac) under the transformers backend.

Software dependencies. Python ≥3.11; PyTorch ≥2.0; transformers, tabpfn, openml, torchvision, scikit-learn, matplotlib. Full pinned requirements in pyproject.toml. Optional: vllm (Linux + CUDA only; falls back to transformers elsewhere).

Datasets. CC18-Semantic and Regression2025-Semantic are introduced in this paper and released alongside the artifact (HF). All other datasets are downloaded automatically from their canonical sources on first run; no licensing exceptions are required.

#### Installation

git clone https://github.com/penfever/marvis.git cd marvis conda create -n marvis python=3.11 -y conda activate marvis pip install -e ".[vision,audio,api]"

On first invocation, MARVIS will fetch any missing pretrained embedding models (DINOv2, Whisper/CLAP, BioCLIPv2, TabPFNv2) and the Qwen2.5-VL-3B VLM into the local Hugging Face / torch.hub caches.

#### Experiment workflow

Smoke test (∼10 minutes, single consumer GPU). Verifies that the install is working end-to-end on a small subset of CIFAR-10: python examples/vision/evaluate_all_vision.py \

--datasets cifar10 \

--models marvis_tsne \

--balanced_few_shot \

--num_few_shot_examples 10 \

--max_test_samples 20 \

--use_knn_connections --nn_k 5 \

--zoom_factor 2.0 \

--backend transformers \

--vlm_model_id "Qwen/Qwen2.5-VL-3B-Instruct" \

--output_dir ./smoke_results

Expected runtime: ∼10 minutes after firstrun model downloads. Expected output: smoke_results/cifar10/cifar10_test_results.json containing accuracy on 20 test points; on a working install this is typically ≥60%.

Inductive-only t-SNE ablation reproduction (∼2.5 hours, single H100). Reproduces Table 7 (the main camera-ready ablation): sbatch marvis_scripts/marvis-ablate-inductive-cifar.sbatch

The sbatch sequentially runs a transductive-baseline condition followed by an inductive condition under matched settings (DINOv2-ViT-L/14-reg, 𝑘 = 30, zoom 15, Qwen2.5-VL-3B; full CIFAR-10 50k training set; 1,000 held-out test images). Expected runtime: ∼2.5 hours total. Expected outputs: per-condition cifar10_test_results.json.

Full benchmark sweep. The remaining sbatch scripts in marvis_scripts/ (marvis-tabular-cls.sbatch, marvis-vision-marvis.sbatch, marvis-audio.sbatch, marvis-bio-marvis.sbatch, plus the OpenML CC18 / Regression2025 orchestrators under examples/tabular/openml_*) reproduce the rows of the main results table. The total compute budget is ∼1,500 H100-hours; per-sbatch budgets are documented in the README.

Full benchmark sweep. Reproducing the full Table 1 of the main paper requires the compute budget noted above. We do not expect artifact reviewers to reproduce the full sweep; the inductiveablation reproduction recipe above is the recommended scope for the Reproduced badge.

#### Notes

- • Compute nodes on shared HPC clusters (e.g., the Jupiter Booster used for the camera-ready ablation) often disable internet access. The provided sbatch script pre-stages CIFAR-10 and the DINOv2 and Qwen2.5-VL-3B weights via login-node Python helpers (commands in the README), and sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, and TORCH_HOME so the compute job uses local caches only.
- • On Apple Silicon, MARVIS auto-detects MPS and switches to the transformers backend; set VLLM_AVAILABLE=false to force this if needed.
- • For the inductive variant, the linear-regression projector is intentionally the simplest plausible parametric approximation of t-SNE (closed-form least squares from training embeddings to training t-SNE coordinates). The accompanying paper section discusses the choice and frames the result as a lower bound on what an inductive deployment of MARVIS could achieve.
- • Per-sample VLM prompts and responses for the main result table (one JSON record per test point, ∼60 GB total) are mirrored to the Hugging Face dataset URL above and are not redistributed via GitHub.

#### Evaluation and expected results

Smoke test. On 20 CIFAR-10 test images with 𝑘 = 10 training samples per class and zoom factor 2.0, expected accuracy is in the 0.55–0.75 range (variance is dominated by the 20-sample test set). Receiving any non-error JSON output indicates a working install.

Inductive-only t-SNE ablation. On full CIFAR-10 training and 1,000 held-out test images, the camera-ready paper reports:

Variant Accuracy t-SNE fit

Transductive (default) 97.8% ∼210s Inductive (LR projector) 96.4% ∼170s

A reproduction within ±1 percentage point on each row is expected (variation comes from t-SNE initialization, ordering of test points within a batch, and stochastic decoding by the VLM).

