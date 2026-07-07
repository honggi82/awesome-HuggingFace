## OpenAutoNLU: Open Source AutoML Library for NLU

Grigory Arshinov1, Aleksandr Boriskin*,1,2, Sergey Senichev*,1,2, Ayaz Zaripov1, Daria Galimzianova1,3, Daniil Karpov1,†, Leonid Sanochkin1,†

1MWS AI, 2ITMO University, 3MBZUAI

Correspondence: g.arshinov@mts.ai

# arXiv:2603.01824v1[cs.CL]2Mar2026

### Abstract

OpenAutoNLU is an open-source automated machine learning library for natural language understanding (NLU) tasks, covering both text classification and named entity recognition (NER). Unlike existing solutions, we introduce data-aware training regime selection that requires no manual configuration from the user. The library also provides integrated data quality diagnostics, configurable out-of-distribution (OOD) detection, and large language model (LLM) features, all within a minimal lowcode API. OpenAutoNLU source code is available here 1, the demo app is accessible here https://openautonlu.dev.

### 1 Introduction

Text classification and NER are foundational tasks in natural language processing (NLP), underpinning applications from intent detection and sentiment analysis to information extraction and document categorization. Despite their ubiquity, deploying effective models for these tasks remains challenging: practitioners must navigate a complex landscape of competing approaches—full finetuning of pretrained transformers, few-shot learning methods, classical machine learning with embeddings—each suited to different data regimes and resource constraints. Further complications arise from data quality issues, hyperparameter sensitivity, and the need for OOD detection in production settings. Automated machine learning (AutoML) offers a promising solution by automating model selection, hyperparameter tuning, and pipeline design, yet existing frameworks exhibit significant gaps when applied to NLP workloads.

* Equal contribution. † Work done while at MWS AI

1https://github.com/mts-ai/OpenAutoNLU

Figure 1: Text Classification Training Pipeline and Inference Pipeline flow

OpenAuto NLU Approach TAML &

Light AutoML

Auto Gluon

Auto Intent

H2O

TAML & embeddings

Encoder finetuning

Embeddings Encoder

finetuning Scarce train data × Has small-data

Word2Vec

Designed for classes with ≥2 examples Custom search pipeline Flexible API HP presets HP presets Presets &

Adaptable Adapted for small datasets

modes

Universal customizable preset LLM Integrations × × × Zero-shot

customizable configs

Train data augmentations and test set generation

classification

OOD detection × × × Train examples

Train examples are optional, configurable

required, one method

× ✓ × ONNX native;

Optimized for NLP production Requires preprocessing;

inference in two lines

Java, Python, R support

Data quality evaluation × × × × ✓ Named Entity Recognition × × ✓ × ✓ Embedding prompting × × × ✓ ✓

Table 1: Comparison of AutoML frameworks for NLP tasks.

We identify two critical gaps in the current AutoML landscape for NLP. First, ease of use: many frameworks require non-trivial configuration, expose complex abstractions (e.g., table-based predictors or modality-specific modules), and lack simple, unified interfaces for common NLP tasks. Second, NLP-centric design: existing systems do not natively integrate (a) automatic selection of training regimes (full fine-tuning vs. few-shot methods) based on dataset size and label distribution, (b) text-specific data-quality assessment, and (c) unified support for both text classification and NER within a single coherent API.

In this work, we present OpenAutoNLU, an open-source AutoML library designed specifically for NLP. OpenAutoNLU addresses the identified gaps through a text-first architecture that emphasizes simplicity, automation, and reproducibility, and demonstrates competitive or superior performance compared to existing AutoML frameworks in standard intent classification benchmarks. The library provides the following key contributions: Automatic training-regime selection, Integrated data-quality tools, Configurable OOD detection, Unified API for classification and NER, Minimal configuration.

### 2 Background

[Figure 1]

Figure 2: This graph illustrates the ratio between performance in macro F1 on classification tasks and time in seconds took to train the solution. Measures are averaged between four different text classification datasets.

Recently, large language models (LLMs) have also been proposed as general-purpose solutions for text classification and intent understanding, of-

ten via prompting or lightweight adaptation. While such models can provide strong zero- and few-shot performance (Wang et al., 2024), they typically incur substantial computational and monetary costs, as well as higher inference latency due to their size and deployment requirements (e.g., multiple GPU-backed serving or external API calls). These characteristics make LLM-based approaches less suitable for many practical scenarios where models must be deployed at scale (Vajjala and Shimangaud, 2025), integrated on-premise, or executed under strict latency and resource constraints. In contrast, the AutoML frameworks considered in this work aim to deliver competitive performance using comparatively lightweight architectures and training pipelines, which are more amenable to efficient, low-latency deployment.

To our knowledge, some solutions for AutoML have previously been introduced. For example, AutoIntent (Alekseev et al., 2025) is an AutoML framework specifically designed for intent classification. It follows an embedding-centric design, where pretrained sentence encoders are combined with either classical classifiers or neural models, followed by automatic decision threshold optimization. AutoIntent supports supervised outof-distribution (OOD) detection, multi-label classification, and multiple training presets that trade off quality and computational cost. In our experiments, we evaluated classic-light (default), classicmedium, nn-medium, nn-heavy and transformersheavy AutoIntent presets.

AutoGluon (Erickson et al., 2020) is a generalpurpose AutoML framework originally developed for tabular data, relying heavily on multi-layer ensembling, stacking, and bagging of heterogeneous models (tree-based models, linear models, and neural networks).

LightAutoML (Vakhrushev et al., 2021) and H2O AutoML (LeDell et al., 2020) are also mainly designed for tabular data. Text inputs are handled by simple Word2Vec-based vectorization and subsequently processed by standard tabular models. All experiments with these frameworks were used with default configurations.

Feature overview for all frameworks can be found in the Table 1.

### 3 OpenAutoNLU

OpenAutoNLU is a library in the Python language that is designed to be used in low-code environ-

ments. Its simple API allows for quick NLU-model prototyping as well as for robust and flexible production model training. Architecturally, it is a collection of classes that are chained by auto pipelines. The diagram can be found in the Figure 1.

Novelty The principal novelty of OpenAutoNLU lies in its resolution of the data-aware method: rather than exposing the user to a menu of algorithms and hyperparameter grids, the library inspects the label distribution of the supplied dataset and deterministically selects the training method that best fits the data regime. Three regimes are distinguished by the minimum per-class sample count nmin (which were computed on our internal benchmarks): (i) 2 ≤ nmin ≤ 5—

AncSetFit (Pauli et al., 2023), an anchor-based few-shot method that leverages human-readable class descriptions and triplet-loss contrastive learning; (ii) 5 < nmin ≤ 80—SetFit (Tunstall et al., 2022), a sentence-transformer–based few-shot learner with a logistic-regression head; (iii) nmin > 80—full transformer fine-tuning (Wolf et al., 2020) with Optuna-driven (Akiba et al., 2019) hyperparameter optimisation. This design means that a practitioner can move from two labelled examples per intent to a production-grade classifier without changing a single line of client code. A second distinctive feature is the integrated out-of-distribution (OOD) detection layer. Each training method has a companion OOD variantMarginal Mahalanobis distance for the fine-tuning regime, Maximum Softmax Probability for SetFit, and a logit-based “outOfScope” class option—all selectable through a single ood_method configuration flag. Third, OpenAutoNLU ships with an LLM-powered data augmentation and synthetic test generation subsystem: when the number of labelled examples is low, the pipeline can call an external language model to synthesise additional training or evaluation samples, guided by automatic domain analysis.

#### 3.1 Structure

Figure 1 shows the high-level module layout. At the top level, the auto_classes module exposes the four public pipeline classes. Each pipeline inherits from AbstractTrainingPipeline or AbstractInferencePipeline, which manage the lifecycle stages: data loading, data processing, (optional) data quality evaluation, method resolution, training, evaluation, and model export.

The methods module contains the training al-

gorithms, each subclassing a shared Method base class:

- • Finetuner / FinetunerWithOOD—standard transformer fine-tuning with early stopping and optional Optuna HPO.
- • SetFitMethod / SetFitOOD—contrastive sentence-transformer training followed by a logistic-regression classifier head.
- • AncSetFitMethod / AncSetFitOODanchor-label–augmented variant of SetFit for extreme few-shot scenarios.
- • TokenClassificationFinetuner—BIOtagging–based NER with entity-level evaluation.

The data module supplies task-specific data providers (SimpleDataProvider for classification, SimpleNerDataProvider for NER) together with an Augmentex-based (Martynov et al., 2023) character- and word-level augmentation engine. The llm_pipelines module adds LLM-driven data generation, domain analysis, and synthetic test set construction.

Data quality A dedicated data_quality module implements a pluggable evaluator framework. A DynamicTuner trains a model on the training set and records per-sample logits across epochs. Four evaluators consume these signals: Retag (van Halteren, 2000) — flag samples whose predicted label disagrees with the annotated label, revealing likely annotation errors. Uncertainty—identifies samples for which the model’s softmax probability on the gold class falls below a configurable threshold, indicating ambiguity. V-Information (Ethayarajh et al., 2025)—measures the usable information each sample contributes by comparing the trained model’s loss with that of a null model (input replaced by a blank token), flagging low-signal examples. Dataset Cartography (Swayamdipta et al., 2020)—(text classification only) computes persample confidence and variability across training epochs, partitioning the data into easy-to-learn, ambiguous, and hard-to-learn regions and producing a visual data map. We characterize hard-to-learn samples as those highly likely to contain annotation errors. We empirically determined optimal thresholds for the hard-to-learn region.

For token-classification tasks, a Label Aggregation evaluator based on Dawid–Skene (Ustalov

et al., 2024) consensus estimation is used instead: it runs Monte Carlo dropout to obtain multiple “annotator” predictions and aggregates them to detect per-token annotation disagreements. These evaluators can be run independently via the diagnose() method of any training pipeline, returning a DatasetEvaluatorOutput with persample scores and a filtered dataset.

Named Entity Recognition. The NER pipeline accepts two annotation formats – offset-based (start/end character indices) and bracket-based (inline markup) – and converts both to a BIO tagging scheme internally. Stratified train/test splitting is performed at the entity level to preserve label proportions. Evaluation uses nervaluate-based (Batista and Upson, 2025) entity-level precision, recall, and F1, with support for partial entity matching.

Optimized-inference-ready model serialization All methods support ONNX export via SaveFormat.ONNX. The resulting model package bundles the ONNX graph, tokeniser files, label mapping, and a meta.json descriptor. The inference managers automatically detect available hardware (CUDA, CoreML, CPU) and run batched inference with automatic batch-size detection to prevent out-of-memory failures.

#### 3.2 Text Classification Training Pipeline

Data Quality Training corpora often contain mislabeled or uninformative examples that degrade model performance(Swayamdipta et al., 2020). OpenAutoNLU includes an optional data-quality stage that identifies such samples before training using an ensemble of diagnostic methods: dataset cartography (Swayamdipta et al., 2020), which tracks per-sample confidence and variability across training epochs and v-usable information (Ethayarajh et al., 2025), which measures how much learnable signal each example carries. We also use uncertainty quantification and retagging, which flag samples with high predictive uncertainty or model-label disagreement.

Objective and search strategy. For the finetuning regime (nmin > 80), the pipeline optimises the macro-averaged F1 score on a held-out validation split (90/10 stratified by default, can be configured), if user explicitly turns hyperparameter optimisation on. Hyperparameter search is performed with Optuna using a Tree-structured

Parzen Estimator (TPE) sampler (Watanabe, 2025) over a configurable search space that includes learning rate (log-uniform in [10−6,10−3]), per-device batch size, and weight decay, with a default budget of 10 trials. Within each trial, early stopping with patience of 5 evaluation steps prevents overtraining; the best checkpoint is retained automatically.

For few-shot regimes (SetFit and AncSetFit), no explicit hyperparameter search is conducted: the methods use well-tuned defaults (e.g. 20 contrastive iterations, backbone learning rate 10−5), and training is performed on the full (downsampled) dataset.

Data-level optimisation Before method selection, the pipeline performs an adaptive rebalancing step. If the fraction of low-resource classes (those with n ≤ 80) exceeds a configurable threshold (default 0.3), underrepresented classes are upsampled to n = 81 using either Augmentex character/word perturbations or, when enabled, LLM-generated paraphrases. After upsampling, the method resolver is re-evaluated, typically promoting the dataset into the fine-tuning regime. Conversely, when a few-shot method is selected, overrepresented classes are downsampled to the method’s ceiling (80 for SetFit, 5 for AncSetFit) to maintain balanced training.

OOD detection optimisation When an OODenabled variant is selected, the pipeline jointly optimises the classifier and the OOD detector. Synthetic out-of-scope samples are generated by a gibberish dataset generator (a function that produces random sequences of randomly placed letters, imitating words in sentences), and a threshold on the chosen OOD score (Mahalanobis distance or maximum softmax probability) is tuned on the validation data. A user-controllable threshold_factor (default 1.0) allows for tuning the True-positive rate.

LLM-based test set generation In many practical scenarios, users lack a held-out test set to estimate the quality of the model after training. OpenAutoNLU addresses this by offering an optional LLM-powered test set generation module: given the training data, an LLM synthesizes realistic labeled examples that can serve as a proxy evaluation set. This is particularly valuable in lowresource settings where collecting and annotating additional data is costly. Our experiments (Ap-

pendix A.1) show that the evaluation scores on the generated test set closely follow those on a real held-out set (with absolute differences below 5 percentage points of up to 80 examples per class), making the generated set a reliable quality signal when no ground-truth test data follow available.

### 4 Experiments

We evaluate on four intent-classification datasets — banking77 (Casanueva et al., 2020), massive (FitzGerald et al., 2022), hwu64 (Liu et al., 2019), and snips (Coucke et al., 2018) — spanning binary and multi-class label spaces across low- (5–10 examples per class), medium- (81–100), and fulldata regimes. All configurations are averaged over three random seeds with standardized framework presets.

Because OpenAutoNLU provides automatic OOD detection, we use two evaluation protocols: (1) OOD-aware, where OOD samples are included in training as an explicit extra label; and (2) OODin-test, where OOD samples appear only at test time with no dedicated label in train. We compare against AutoIntent (Alekseev et al., 2025), AutoGluon (Erickson et al., 2020), LightAutoML (Vakhrushev et al., 2021), and H2O (LeDell et al., 2020), and report F1-macro, F1-in-scope, and F1OOD. Data splits and implementation details are provided in Appendix E.

To isolate AutoML logic from representation differences, we fix pretrained checkpoints where possible: bert-base-uncased 2 is used as the default backbone when supported. For embedding-centric pipelines requiring E5-style representations (notably AutoIntent), we use intfloat/multilingual-e5large-instruct 3.

This choice ensures consistency across frameworks and isolates the contribution of the OpenAutoNLU logic itself rather than differences in pretrained representations.

#### 4.1 OOD Unaware Regime

The OOD-unaware regime mimics a realistic production setting where a deployed model inevitably encounters out-of-distribution input, despite having received no explicit OOD supervision during training. Concretely, OOD samples are present at test time, but no OOD label is provided during training, and the reported F1-Macro is computed

- 2https://huggingface.co/google-bert/bert-base-uncased
- 3https://huggingface.co/intfloat/multilingual-e5-large

Framework Preset Banking77 HWU64 MASSIVE SNIPS Macro OOD Macro OOD Macro OOD Macro OOD

OpenAutoNLU supervised 0.905 0.362 0.893 0.276 0.875 0.471 0.928 0.782 OpenAutoNLU unsupervised 0.912 0.433 0.890 0.378 0.876 0.515 0.921 0.761

AutoIntent CL & unsupervised 0.869 - 0.829 - 0.755 - 0.786 AutoIntent CL & supervised 0.774 0.156 0.728 0.196 0.683 0.366 0.771 0.662 AutoIntent CM & supervised 0.819 0.179 0.728 0.196 0.683 0.366 0.829 0.707

Table 2: F1-Macro and OOD F1-Score for frameworks with automatic OOD detection. In unsupervised regime no OOD train samples are provided, for supervised method OOD samples are selected as described in Section 4.2. 2 best performing presets for AutoIntent are reported (CL for classic-light and CM for classic-medium). Full datasets were used. Best results per dataset are shown in bold.

exclusively over in-domain classes. This makes the metric a direct measure of how well a framework maintains in-domain classification quality under realistic distributional shift.

OpenAutoNLU achieves the best or tied performance on three of four datasets, with AutoGluon being the only competitor to outperform it — on Banking77 only, and at a considerably higher computational cost (see Figure 2). A detailed perframework breakdown, as well as in-scope F1Macro results across low-, medium-, and full-data regimes under a clean test condition with no OOD samples, are provided in Appendix C.

#### 4.2 OOD detection evaluation

We categorize the OOD samples by semantic distance from in-domain data as previously suggested (Baran et al., 2023):

- 1. Close-OOD: Held-out classes from the same macro-category (scenario) as in-domain classes. Only available for hierarchicallylabeled datasets.
- 2. Mid-OOD: Held-out classes from different macro-categories within the same dataset.
- 3. Far-OOD: Samples from a different dataset with different domain (e.g., Banking77 ↔ HWU64) (Zhan et al., 2021).
- 4. Very-Far-OOD: Synthetically generated gibberish text with no semantic content.

OOD samples are uniformly balanced across all semantic distance categories.

The number of OOD samples in the test set is determined by the 95th percentile of the in-domain class size distribution.

For supervised OOD experiments, OOD samples constitute half of the training set size.

Frameworks with native or explicit support for OOD detection (AutoIntent and OpenAutoNLU) are evaluated under supervised OOD settings, where OOD samples were provided during training and testing. Moreover, OpenAutoNLU was evaluated under an unsupervised OOD regime, where no explicit OOD label was provided during training and OOD samples appear only at test time. We further analyze how OOD detection quality changes when OOD examples are introduced as a labeled class during training. The metrics are shown in the Table 2 and a detailed description is provided in the Appendix D.

### 5 Conclusion

In this work, we present an easy to use NLU-model producing and serving library. We inspected solutions of our competitors, highlighted their points of growth and offered our library as an all-in-one batteries-included low-code NLU solution. Experimentally, we proved our solution to be wellbalanced in terms of the cost and quality of the resulting models. We shown that our Out Of Domain layer is not only unique among other Auto ML libraries in its configurability, but also it is state of the art among solutions that have OOD layers implemented.

### Limitations and Future work

Our data-driven deterministic optimization strategy will be further analyzed by implementing a metamodel that would decide the best combination of training method, augmentation, and OOD method based on more abstract dataset features such as dataset2vec (Jomaa et al., 2021).

### References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A nextgeneration hyperparameter optimization framework. Preprint, arXiv:1907.10902.

Ilya Alekseev, Roman Solomatin, Darina Rustamova, and Denis Kuznetsov. 2025. Autointent: Automl for text classification. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 707–716.

Mateusz Baran, Joanna Baran, Mateusz Wójcik, Maciej Zi˛eba, and Adam Gonczarek. 2023. Classical out-of-distribution detection methods benchmark in text classification tasks. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 119–129.

David Batista and Matthew Antony Upson. 2025. nervaluate.

Iñigo Casanueva, Tadas Temˇcinas, Daniela Gerz, Matthew Henderson, and Ivan Vuli´c. 2020. Efficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pages 38–45, Online. Association for Computational Linguistics.

Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, Maël Primet, and Joseph Dureau. 2018. Snips voice platform: an embedded spoken language understanding system for private-bydesign voice interfaces. Preprint, arXiv:1805.10190.

Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. 2020. Autogluon-tabular: Robust and accurate automl for structured data. arXiv preprint arXiv:2003.06505.

Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. 2025. Understanding dataset difficulty with V-usable information. Preprint, arXiv:2110.08420.

Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2022. Massive: A 1m-example multilingual natural language understanding dataset with 51 typologically-diverse languages. Preprint, arXiv:2204.08582.

Hadi S. Jomaa, Lars Schmidt-Thieme, and Josif Grabocka. 2021. Dataset2vec: Learning dataset meta-features. Preprint, arXiv:1905.11063.

Erin LeDell, Sebastien Poirier, and 1 others. 2020. H2o automl: Scalable automatic machine learning. In Proceedings of the AutoML Workshop at ICML, volume 2020, page 24.

Xingkun Liu, Arash Eshghi, Pawel Swietojanski, and Verena Rieser. 2019. Benchmarking natural language understanding services for building conversational agents. Preprint, arXiv:1903.05566.

Nikita Martynov, Mark Baushenko, Anastasia Kozlova, Katerina Kolomeytseva, Aleksandr Abramov, and Alena Fenogenova. 2023. A methodology for generative spelling correction via natural spelling errors emulation across multiple domains and languages. Preprint, arXiv:2308.09435.

Amalie Pauli, Leon Derczynski, and Ira Assent. 2023. Anchoring fine-tuning of sentence transformer with semantic label information for efficient truly few-shot classification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11254–11264, Singapore. Association for Computational Linguistics.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. 2020. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9275–9293, Online. Association for Computational Linguistics.

Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, and Oren Pereg. 2022. Efficient few-shot learning without prompts. Preprint, arXiv:2209.11055.

Dmitry Ustalov, Nikita Pavlichenko, and Boris Tseitlin.

2024. Learning from Crowds with Crowd-Kit. Journal of Open Source Software, 9(96):6227.

Sowmya Vajjala and Shwetali Shimangaud. 2025. Text classification in the llm era – where do we stand? Preprint, arXiv:2502.11830.

Anton Vakhrushev, Alexander Ryzhkov, Maxim Savchenko, Dmitry Simakov, Rinchin Damdinov, and Alexander Tuzhilin. 2021. Lightautoml: Automl solution for a large financial services ecosystem. arXiv preprint arXiv:2109.01528.

Hans van Halteren. 2000. The detection of inconsistency in manually tagged text. In Proceedings of the COLING-2000 Workshop on Linguistically Interpreted Corpora, pages 48–55, Centre Universitaire, Luxembourg. International Committee on Computational Linguistics.

Zhiqiang Wang, Yiran Pang, Yanbin Lin, and Xingquan Zhu. 2024. Adaptable and reliable text classification using large language models. Preprint, arXiv:2405.10523.

Shuhei Watanabe. 2025. Tree-structured parzen estimator: Understanding its algorithm components and their roles for better empirical performance. Preprint, arXiv:2304.11127.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Li-Ming Zhan, Haowen Liang, Bo Liu, Lu Fan, Albert YS Lam, and Xiao-Ming Wu. 2021. Out-ofscope intent detection with self-supervision and discriminative training. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3521–3532.

### Appendix

### A LLM-generated test set evaluation

The experiments reported in Table A.1 were conducted using GPT-4o-mini as the generative backend. The module is not tied to this specific model: OpenAutoNLU supports any OpenAI APIcompatible endpoint, including locally hosted models.

N-shot Original Generated |∆| (pp) [5, 10] 0.667 0.699 +0.032 [10, 20] 0.655 0.637 -0.017 [20, 40] 0.779 0.759 -0.02 [40, 80] 0.808 0.790 -0.018 [81, 100] 0.709 0.561 -0.148 Full 0.759 0.690 -0.689

Table A.1: Macro-F1 on original vs. LLM-generated test sets. The horizontal rule separates regimes where the generated set is a reliable proxy (|∆| < 5pp) from those where it is not.

### B Threshold selection

The regime boundaries nmin = 5 and nmin = 80 were determined empirically across a collection of both publicly available English datasets and internal non-English datasets, spanning a variety of classification tasks including intent recognition, sentiment analysis, and general text classification across multiple domains. For each candidate threshold, we measured macro-F1 performance of each training method (AncSetFit, SetFit, and full fine-tuning) across data regimes and identified the per-class sample counts at which transitioning to a more dataintensive method yielded consistent performance gains. The boundaries reported in this work reflect the values at which these gains were found to be stable across the majority of evaluated datasets and domains. We note that while these thresholds may not be universally optimal for every possible dataset, they provide a robust default that eliminates the need for manual method selection in the vast majority of practical settings.

### C In-domain classification evaluation

- Table C.1 shows that OpenAutoNLU achieves the best or tied performance on three of four datasets

— HWU64, MASSIVE, and SNIPS — while remaining highly competitive on Banking77 (0.914 vs. AutoGluon’s 0.920). LightAutoML is the closest general-purpose competitor on Banking77 and

Framework Banking77 HWU64 MASSIVE SNIPS OpenAutoNLU 0.912 0.890 0.876 0.921 AutoGluon 0.920 0.902 0.861 0.763 AutoIntent 0.869 0.829 0.755 0.786 LightAutoML 0.907 0.900 0.841 0.761 H2O 0.645 0.642 0.701 0.778

Table C.1: F1-Macro on full datasets under the OODunaware regime. OOD samples are present at test time but excluded from the classification report; scores reflect in-domain classification quality only. Best results per dataset are shown in bold.

HWU64, whereas H2O lags substantially across all benchmarks. The only framework to outperform OpenAutoNLU on any dataset is AutoGluon, which does so only on Banking77 and at a considerably higher computational cost. AutoIntent, despite being purpose-built for intent classification, falls notably short of OpenAutoNLU on all four datasets, with the largest gap observed on MASSIVE (0.755 vs. 0.880).

Table C.2 further reports in-scope F1-Macro under a stricter controlled condition where no OOD samples appear in the test set, spanning low(5–10 shots), medium- (40–80 shots), and full-data regimes. These results are broadly consistent with the OOD-unaware findings: OpenAutoNLU performs strongly in medium- and full-data regimes, particularly on MASSIVE and SNIPS. AutoGluon and LightAutoML are competitive at larger data sizes but degrade significantly in the 5–10 shot setting, while AutoIntent shows the inverse pattern — strongest in low-resource conditions but underperforming as training set size grows. H2O remains the weakest framework across all settings and datasets.

### D OOD detection description

Table 2 reveals several notable patterns across frameworks and regimes. OpenAutoNLU in the unsupervised regime achieves the best overall balance between in-domain and OOD detection performance, leading on both Macro and OOD F1 on Banking77, HWU64, and MASSIVE. Interestingly, introducing labeled OOD samples during training does not consistently improve OpenAutoNLU’s performance: while the supervised regime yields the highest OOD F1 on SNIPS (0.782), it comes at the cost of reduced scores on the remaining datasets. This suggests that OpenAutoNLU’s unsupervised OOD mechanism is already effective without explicit OOD supervision.

###### Framework Banking77 HWU64 MASSIVE SNIPS

5-10 40-80 Full 5-10 40-80 Full 5-10 40-80 Full 5-10 40-80 Full OpenAutoNLU 0.727 0.881 0.920 0.747 0.753 0.902 0.666 0.711 0.886 0.607 0.890 0.953 AutoIntent 0.808 0.848 0.782 0.782 0.830 0.737 0.660 0.751 0.689 0.813 0.826 0.789 AutoGluon 0.598 0.911 0.935 0.625 0.899 0.923 0.508 0.824 0.885 0.696 0.868 0.915 LightAutoML 0.524 0.891 0.922 0.555 0.891 0.920 0.425 0.824 0.865 0.659 0.867 0.913 H2O 0.159 0.612 0.654 0.272 0.632 0.652 0.398 0.616 0.635 0.549 0.810 0.908

- Table C.2: In-scope macro-F1 score comparison across different N-shot settings for intent classification task. Default presets for all frameworks are used.

AutoIntent requires supervised OOD samples to produce any OOD predictions at all, and even then falls substantially behind OpenAutoNLU with OOD F1 gaps as large as 0.304 on Banking77 and 0.144 on MASSIVE. Moreover, AutoIntent’s supervised presets incur a significant drop in Macro F1 relative to its unsupervised baseline, indicating that OOD supervision interferes with its in-domain classification. Overall, OpenAutoNLU provides a more robust and consistent solution for joint indomain classification and OOD detection across all evaluated datasets and regimes.

### E Sampling Implementation Details

We evaluate on four English intent classification datasets: Banking77, HWU64, MASSIVE, and SNIPS. For each, we use the same in-distribution versus out-of-distribution (OOD) setup and the same OOD evaluation protocol.

In-distribution sampling. We apply a flat indistribution vs. OOD sampling regime. Only classes with at least nmin training examples are retained, where nmin is set to the upper bound of the few-shot range in few-shot runs and to 100 in full-data runs. Among these classes, a fixed fraction (80%) is chosen at random as in-distribution; the remainder are treated as mid OOD and never appear in training. Within each in-distribution class, 90% of examples are used for training and 10% for testing. In few-shot experiments, the training set is further reduced by sampling a random number of examples per class within the specified n-shot range; classes that fall below the minimum are dropped.

OOD evaluation setup. The test set is augmented with three OOD categories. Mid OOD consists of held-out classes from the same dataset (the 20% of classes not selected as in-distribution). Far OOD is drawn from a different intent dataset in the same language to simulate cross-domain OOD;

Table E.1: Far OOD source for each primary intent benchmark.

Primary dataset Far OOD source Banking77 HWU64 HWU64 Banking77 MASSIVE Banking77 SNIPS Banking77

the pairing is chosen so that the far OOD source is semantically related but distributionally distinct (see Table E.1). Very far OOD comprises 1,000 synthetically generated English gibberish utterances to probe robustness to nonsensical inputs. The total number of OOD examples in the combined test set is set to the 95th percentile of in-distribution class sizes; this budget is split equally among the three OOD types. Optionally, OOD examples can also be included in the training set (with total OOD volume equal to half the in-distribution training size, again split across the three types) to assess methods that require OOD at train time.

### F Hardware

All experiments were run on a single machine with the following specifications: an Intel Xeon Gold 6448H processor (64 cores), an NVIDIA H100 GPU with 80 GB VRAM, and 756 GB of system RAM. Training time measurements reported in Figure 2 reflect wall-clock time on this hardware configuration. GPU acceleration was used for frameworks that support it; frameworks without GPU support were run on CPU.

