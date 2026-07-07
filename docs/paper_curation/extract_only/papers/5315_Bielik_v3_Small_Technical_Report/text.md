# arXiv:2505.02550v2[cs.LG]8May2025

## BIELIK V3 SMALL: TECHNICAL REPORT

THE BIELIK LLM TEAM

Krzysztof Ociepa1,4, Łukasz Flis1,2, Remigiusz Kinas1, Krzysztof Wróbel1,3,5, Adrian Gwoz´dziej1, 2

1SpeakLeash, 2ACK Cyfronet AGH, 3Jagiellonian University, 4Azurro, 5Enelpol

### ABSTRACT

We introduce Bielik v3, a series of parameter-efficient generative text models (1.5B and 4.5B) optimized for Polish language processing. These models demonstrate that smaller, well-optimized architectures can achieve performance comparable to much larger counterparts while requiring substantially fewer computational resources. Our approach incorporates key innovations, including a custom Polish tokenizer (APT4) that significantly improves token efficiency and Adaptive Learning Rate that dynamically adjusts based on training progress. Trained on a meticulously curated corpus of 292 billion tokens spanning 303 million documents, these models excel across multiple benchmarks, including the Open PL LLM Leaderboard, Complex Polish Text Understanding Benchmark, Polish EQ-Bench, and Polish Medical Leaderboard. The 4.5B parameter model achieves results competitive with models 2-3 times its size, while the 1.5B model delivers strong performance despite its extremely compact profile. These advances establish new benchmarks for parameter-efficient language modeling in less-represented languages, making high-quality Polish language AI more accessible for resourceconstrained applications.

### 1 Introduction

The rapid advancement in natural language processing (NLP) has led to the development of increasingly sophisticated language models that can understand and generate human-like text. These models have shown remarkable success in various linguistic tasks across multiple languages. However, the development of high-performing models for lessresourced languages remains a significant challenge due to the scarcity of large and diverse datasets and computational resources.

Several notable efforts have advanced Polish language modeling in recent years. TRURL 2 Voicelab [2023], a collection of fine-tuned Llama 2 models with 7 billion and 13 billion parameters, was trained on approximately 1 million conversational samples. Qra National Information Processing Institute and Gda´nsk University of Technology [2024] models, comprising continuously pretrained architectures with 1, 7, and 13 billion parameters, leveraged 90 billion tokens of Polish data. More recently, PLLuM, developed by a consortium of Polish academic institutions, introduced models ranging from 8 billion to 70 billion parameters, created through continued pretraining of Llama and Mistral models on Polish corpora. While these initiatives have made important strides, they often face limitations in performance, versatility, or accessibility, frequently requiring significantly larger computational resources for comparable performance.

Building on our previous work with Bielik 7B v0.1 Ociepa et al. [2024] and Bielik 11B v2 Ociepa et al. [2025], we introduce the Bielik v3 series of generative text models optimized specifically for Polish language processing. These models, with sizes of 1.5B and 4.5B parameters, represent a significant advancement in parameter-efficient language modeling. By adopting an innovative approach to model scaling and training, we demonstrate that smaller, well-optimized models can achieve performance comparable to much larger counterparts while requiring substantially fewer computational resources.

Our approach introduces several key technical innovations. First, we employ depth up-scaling to adapt Qwen2.5 models Qwen et al. [2025], replacing the original tokenizer with a custom-developed Polish tokenizer (APT4) that significantly improves token efficiency for Polish texts. Second, we utilize Adaptive Learning Rate, which dynamically adjusts the

learning rate based on training progress and context length. These techniques, combined with comprehensive training on a diverse corpus of 292 billion tokens across 303 million documents, enable the Bielik v3 models to achieve remarkable performance despite their compact size.

Our evaluation demonstrates that Bielik v3 models outperform many larger models across various benchmarks, including the Open PL LLM Leaderboard, Complex Polish Text Understanding Benchmark (CPTUB), Polish EQ-Bench, and Polish Medical Leaderboard. Notably, the 4.5B parameter model achieves results competitive with models 2-3 times its size, while the 1.5B model delivers strong performance despite its extremely compact profile. This efficiency makes Bielik v3 particularly valuable for deployment in resource-constrained environments while maintaining high-quality Polish language capabilities.

In the following sections, we detail the model architecture and tokenizer design of Bielik v3, describe our comprehensive data preparation methodology, discuss the pre-training and post-training processes, and evaluate the models’ performance across multiple benchmarks. We also analyze the models’ limitations and potential biases. Our results demonstrate that Bielik v3 not only advances the state of Polish language understanding but also establishes new benchmarks for parameter-efficient language modeling in less-represented languages.

### 2 Model and Tokenizer

In this section, we introduce the model design and tokenizer, presenting architectural decisions and configurations. resources

#### 2.1 Model Architecture

Parameter 1.5B 4.5B Layers 32 60 Model Dimension 1536 2048 Attention Heads 12 16 Key/Value Heads 2 2 Head Size 128 128 Intermediate Size 8960 11008 Activation Function SwiGLU SwiGLU Attention Bias True True MLP Bias True True Vocabulary Size 32000 32000 Positional Embeddings RoPE (θ = 1000000) RoPE (θ = 1000000) Context Length 8192 32768

Table 1: Architecture details of the 1.5B and 4.5B parameter models.

Bielik v3 models are based on the Transformer architecture Vaswani et al. [2017], with key parameters listed in Table 1. The design integrates several advanced techniques to enhance performance and efficiency.

Self-attention with causal masks Vaswani et al. [2017] enables the model to assign varying importance to different parts of the input sequence. The causal mask ensures that the model only attends to preceding tokens, preserving the autoregressive property essential for language modeling.

Grouped-query attention (GQA) Ainslie et al. [2023] reduces both computational complexity and memory usage while maintaining model quality. It achieves this by using fewer key-value heads than query heads, enabling more efficient handling of long sequences.

SwiGLU activation function Dauphin et al. [2016], Shazeer [2020] combines the Swish activation function with Gated Linear Units (GLU), providing better performance and trainability than traditional activation functions such as ReLU.

Rotary Positional Embeddings (RoPE) Su et al. [2024] enhance the model’s ability to capture relative token positions. Compared to absolute positional embeddings, RoPE supports better generalization to longer sequences and improves performance in tasks requiring positional sensitivity.

Root Mean Square Layer Normalization (RMSNorm) Jiang et al. [2024] normalizes activations within the network, offering greater training stability and slightly faster computation compared to standard Layer Normalization.

Pre-normalization involves applying layer normalization before the self-attention and feed-forward layers. This improves model convergence and overall performance.

The Bielik v3 models, with sizes 1.5B and 4.5B, are adapted from the Qwen2.5 1.5B and 3B models Qwen et al. [2025]. The models were scaled using the Depth Up-Scaling method Kim et al. [2024], the tokenizer was replaced, and further pretraining was conducted, as presented in Figure 1. The decision to build on an existing model rather than developing one from scratch was motivated by the desire to allocate resources efficiently. By focusing on the linguistic adaptation of an already high-performing model, we were able to optimize both time and computational resources. The Qwen2.5 models were selected due to their strong benchmark performance and permissive Apache 2.0 license.

[Figure 1]

Figure 1: Bielik 4.5B v3 model upscaling via Depth Up-Scaling (n = 36, m = 8, s = 56) with tokenizer replacement and outermost layer duplication.

#### 2.2 Tokenizer

Polish English Tokenizer Vocab Size Avg tokens Tokens CpT TpW Tokens CpT TpW

- APT3 31980 480 344 5.22 1.48 615 3.15 1.93

- APT4 32000 503 375 4.78 1.62 631 3.07 1.98 Mistral v0.1 32000 578 747 2.40 3.22 408 4.75 1.28 Qwen2.5 151665 499 625 2.87 2.69 373 5.19 1.17

Table 2: Comparison of token count, characters per token (CpT), and tokens per word (TpW) for the preamble of the Constitution of the Republic of Poland in Polish and English, processed by various tokenizers: APT3 and APT4 (Polish-specific tokenizers), Mistral v0.1 and Qwen2.5 (multilingual tokenizers with limited Polish support).

To enhance tokenization efficiency for the Polish language, we replaced the original tokenizer of the Qwen language model with our custom-developed Polish tokenizer, APT. This modification aimed to reduce the number of tokens

required to represent input and output sequences, thereby enabling the model to handle longer contexts within its attention window and generate outputs more efficiently. Such improvements are particularly beneficial for smaller models, where token budget constraints are more pronounced.

One way to assess the effectiveness of the tokenization process is by analyzing the number of tokens generated for a given input. A lower token count generally indicates more efficient and faster text generation by the language model. The tokenizers used in the Mistral 7B and Qwen2.5 models were not specifically trained for the Polish language. Therefore, we decided to switch to a tokenizer trained primarily for Polish, with some support for English.

In addition to token count, we also considered how the tokenizer segments text—particularly whether it separates digits, punctuation, and special characters into distinct tokens, which can significantly impact the quality of generated responses. As a result, we ultimately chose to use our own APT4 tokenizer, which is a successor to the APT3 tokenizer from the Polish APT3 model Ociepa and Azurro Team [2024].

Adapting the model to the new tokenizer necessitated reinitializing the token embedding matrix to accommodate the altered vocabulary. We evaluated several embedding initialization methods:

- • Random Initialization: Assigns random vectors to new tokens, effectively requiring the model to learn embeddings from scratch, which can be inefficient and slow to converge.
- • Frequency-based Vocabulary Transfer (FVT) Yuan et al. [2022]: Initializes embeddings for new tokens by averaging the embeddings of their constituent subword units from the original tokenizer, leveraging frequency information to inform the transfer.
- • Linear Interpolation (aX + b): Applies a linear transformation to map embeddings from the source tokenizer’s space to the target tokenizer’s space, aiming to preserve relational structures between tokens.
- • WECHSEL Minixhofer et al. [2022]: Utilizes multilingual static word embeddings to identify semantically similar tokens between source and target vocabularies, initializing new token embeddings based on these similarities .
- • FOCUS (Fast Overlapping Token Combinations Using Sparsemax) Dobler and de Melo [2023]: Represents new tokens as sparse combinations of overlapping tokens from the source and target vocabularies, selected based on semantic similarity in an auxiliary embedding space .
- • OFA (One For All) Liu et al. [2023]: Leverages external multilingual word embeddings to initialize unseen subword embeddings, facilitating efficient adaptation of pretrained models to new languages .
- • RAMEN Tran [2020]: Employs alignment techniques, such as bilingual dictionaries or cross-lingual embeddings, to map source language embeddings to target language tokens, aiding in the transfer of pretrained models to new languages.

After comparative analysis, we selected the FOCUS method for initializing embeddings corresponding to the APT tokenizer. FOCUS’s approach of constructing new token embeddings as sparse combinations of semantically similar overlapping tokens proved effective in preserving the model’s performance while accommodating the new tokenizer. To assess the efficacy of this tokenizer replacement and embedding initialization, we monitored the initial training loss of the model, providing insights into the immediate impact of these modifications on the model’s learning dynamics.

After tokenizer change we duplicated the outermost layers of the model twice, allowing room for adaptation to the new embeddings. Next, we froze the entire model except for the duplicated layers and the embeddings, and trained it on a dataset containing 56 billion tokens. After this initial adaptation, we unfroze the entire model and continued training.

We selected the preamble of the Constitution of the Republic of Poland as the benchmark text because it effectively represents the style of Polish formal writing and is also available in an official English translation, enabling meaningful comparison. Table 2 provides a detailed comparison of key metrics such as token count, characters per token (CpT), and tokens per word (TpW), demonstrating the relative performance of different tokenizers on both language versions of the preamble.

### 3 Pre-training

The primary objective of the pre-training phase was to enhance the model’s proficiency in the Polish language, with an emphasis on both accuracy and fluency. To achieve this, we utilized a diverse collection of high-quality Polish texts. These materials underwent rigorous preprocessing and thorough quality evaluation to ensure the highest standards of training data, as shown in Tables 4, 5, and 6.

#### 3.1 Pre-training Data

The pre-training of the Bielik v3 models involved constructing a novel, diverse, and high-quality dataset composed primarily of Polish-language texts. We leveraged resources from the SpeakLeash project SpeakLeash Team [2024]. Using metadata associated with each document—including topical information and various stylometric features—we selected 294 million documents from different datasets, ensuring both high quality and thematic diversity. These selected texts underwent comprehensive cleaning and quality evaluation, as described in Sections 3.1.1 and 3.1.2.

Additionally, we excluded documents where scraping was technically permitted (i.e., not blocked by robots.txt) but where the terms and conditions explicitly prohibited use for training language models. Only documents meeting our stringent quality standards were retained and subsequently tokenized. This meticulous curation resulted in a Polish training corpus of 237 billion tokens.

To improve the model’s adaptation to Polish while mitigating catastrophic forgetting Li et al. [2022], Ostapenko et al. [2022], Ibrahim et al. [2024], we supplemented the dataset with English texts from the SlimPajama dataset Soboleva

- et al. [2023], known for its diversity and quality.

To support the model’s readiness for later training stages, we included the instruction dataset from Section 5.2 as part of the pre-training corpus. Originally intended for supervised fine-tuning (SFT), this data contributed to a more seamless and efficient progression into the subsequent phases of training.

In total, the final training dataset comprised 292 billion tokens (303 million documents).

#### 3.1.1 Data Cleanup

To enhance the quality of the documents, we applied a series of heuristics designed to remove corrupted or irrelevant content, anonymize personal data (including physical addresses, email addresses, phone numbers, and URLs), and resolve encoding or formatting issues. These steps produced cleaner, higher-quality texts that were subsequently subjected to further evaluation.

#### 3.1.2 Quality Evaluation

To develop the training dataset for text quality evaluation, we manually selected and annotated documents, categorizing them into three quality classes: HIGH, MEDIUM, and LOW. The HIGH class represents superior-quality documents, LOW denotes poor-quality texts, and MEDIUM encompasses documents whose quality is ambiguous, falling between high and low standards. This nuanced classification approach addresses the inherent complexities in assessing textual quality.

The Bielik v3 dataset comprises 44344 training documents, 3000 test documents, and 1000 validation documents. In addition to real-world texts, we introduced synthetic samples designed to probe the classifier’s sensitivity to a wide range of quality degradations.

Synthetic Data Categories HIGH-quality synthetics

Well-structured and linguistically fluent Markdown documents, demonstrating high factual coherence, clarity of expression, and grammatical correctness in Polish. These examples represent ideal outputs generated under constrained decoding settings.

MEDIUM-quality synthetics Passages that are locally fluent and structurally plausible but exhibit moderate issues such as inconsistent tense, mild repetition, or topic drift. They reflect borderline cases between usable and discardable outputs.

#### LOW-quality synthetics

Outputs generated under high-temperature sampling (e.g., temperature > 1.1), often formatted correctly in Markdown, but semantically incoherent, factually incorrect, or disorganized. This category also includes examples showing characteristic LLM failures such as looping or hallucinated content despite visually clean structure. Additionally, it covers cases of extremely poor machine translation—especially from languages structurally distant from Polish, such as Chinese—where the resulting text, while superficially grammatical, is conceptually broken, mistranslated, or entirely nonsensical. These are now reliably detected and classified as low-quality. Furthermore, we have introduced metrics aimed at detecting potentially looped or repetitive content, which sometimes occurred during text generation or editing using the Bielik v2 language model. These metrics effectively identify texts that may appear well-formatted and conceptually strong, but contain signs of excessive repetition or looping, raising concerns about content quality despite their otherwise high surface coherence.

These synthetic examples serve as both strong positives and strong negatives, enabling the model to move beyond surface-level features and develop greater sensitivity to subtle semantic and structural failures.

#### Stylometric Feature Set

To support classification, we designed an expanded set of stylometric features, aimed at capturing surface and deep characteristics of text typical for LLM-generated output. While the Bielik v2 classifier relied on 150 stylometric and Markdown-aware descriptors, Bielik v3 increases this to 200, introducing novel features tailored to detect generation artefacts, degraded machine translation, and formatting inconsistencies.

Lexical richness and repetition Features such as unique_word_count, hapax_legomena_ratio, and looping_suspicion help separate repetitive from genuinely varied prose.

Diacritic and encoding hygiene Counters of Polish diacritics, the replacement_char_ratio, and related metrics expose corrupted character conversions.

Sentence- and line-level structure Ratios of interrogative sentences or single-word lines flag unusual formatting that often correlates with low factual quality.

#### Readability indices

The lix and rix scores provide continuous proxies for extremely verbose or overly terse fragments.

NER-based coherence The distribution of entity types (person, organisation, location, miscellaneous) helps detect hallucinations or long stretches of proper nouns without context.

#### Morphosyntactic diversity

Features measuring variation in case, tense, and mood penalise unnaturally uniform or erratically shifting narratives. Part-of-speech–weighted top-word ratios

Balances of nouns, verbs, and adjectives among the most frequent tokens reveal content focus versus padding. Feature extraction and modelling We compute these features with an extended StyloMetrix-inspired pipeline Okulska

- et al. [2023], augmented with Unicode handling and a dependency parser for the new descriptors. Among the classifiers tested, XGBoost again achieved the highest macro-F1 on the validation split; detailed feature-importance rankings and ablations are reported in Tables 3 and 4.

The performance of the Bielik v3 quality classifier was rigorously evaluated on a held-out validation set. The model achieved an overall accuracy of 95% and a macro-average F1-score of 0.85, demonstrating strong and balanced performance across all three quality categories.

Notably, the classifier reached an F1-score of 0.97 for both the HIGH and LOW classes, indicating excellent precision and recall in distinguishing clearly defined outputs. Despite lower recall in the MEDIUM category (0.51), which reflects its inherently ambiguous nature, the model maintains reliable boundaries between high- and low-quality texts—an essential requirement for downstream data curation.

Quality threshold for corpus construction A manual audit of 1000 validation documents confirmed that a predicted probability P(HIGH) > 0.50 and P(MEDIUM) > 80 remains an effective cutoff. Documents below this threshold are excluded from the instruction-tuning corpus used in downstream Bielik v3 training.

With the larger dataset, purpose-built synthetic adversaries, and a richer 200-dimensional feature vector, the Bielik v3 quality classifier more precisely detects both overt and subtle degradations, providing a robust filter for large-scale LLM data.

Best model configuration. The best-performing model, XGB_RegL1L2_d8_n250_lr008_a02_l05, is an XGBoostClassifier trained with the following hyperparameters: n_estimators = 250, max_depth = 8, learning_rate = 0.08, subsample = 0.75, colsample_bytree = 0.8, reg_alpha = 0.2, reg_lambda = 0.5. The model uses the multi:softprob objective, was trained with eval_metric = mlogloss, and a fixed random_state = 42. Label encoding was disabled via use_label_encoder = False.

#### Model Val F1 (macro) Val F1 (weighted) Val Accuracy

XGB_RegL1L2_d8_n250_lr008_a02_l05 0.852303 0.941526 0.946 XGB_AggressiveSubsample_d8_n300_lr007 0.841951 0.938001 0.943 XGB_RegL1_d7_n200_lr01_a05 0.841951 0.938001 0.943 CatBoost 0.841948 0.937998 0.943 XGB_RegL2_d6_n300_lr005_l1 0.838424 0.937647 0.943 XGBoost_nEstimators500_maxDepth6_lr007_gamma15 0.836765 0.936393 0.941 XGB_StrongRegCombo_d7_n400_lr003_g2_mcw4_a01_l01 0.834315 0.933820 0.939 XGBoost_nEstimators400_maxDepth3_lr025_gamma05 0.832164 0.934450 0.938 XGB_VeryDeep_LowLR_d15_n200_lr005_reg 0.830983 0.934794 0.941 HistGradientBoosting 0.828799 0.933364 0.938 XGBoost_nEstimators180_maxDepth12_lr004_minChild5 0.828858 0.931842 0.938 XGBoost_nEstimators100_maxDepth6_lr01 0.823295 0.931375 0.937 XGBoost_nEstimators200_maxDepth8_lr005 0.825671 0.931104 0.937 LightGBM 0.825986 0.930186 0.934 XGBoost 0.817084 0.927873 0.933 MLP_hidden100_relu_adam 0.806844 0.913972 0.915 EBM 0.799630 0.914568 0.921 MLP_hidden100_50_relu_lbfgs 0.794733 0.915277 0.919 TabNet 0.793772 0.917838 0.920 MLP_hidden50_50_tanh_sgd 0.796052 0.912717 0.917 RandomForest_nEstimators300_maxDepth20_minSamples5 0.777787 0.914178 0.926

Table 3: Comparison of model performance on the validation set

Class Precision Recall F1-score Support LOW 0.95 0.98 0.97 461 MEDIUM 0.79 0.51 0.62 74 HIGH 0.95 0.98 0.97 465

Accuracy 0.95 1000 Macro avg 0.90 0.82 0.85 1000 Weighted avg 0.94 0.95 0.94 1000

Table 4: Classification Report (Validation) – Best Quality Classifier Model

Class Precision Recall F1-score Support LOW 0.94 0.97 0.96 1384 MEDIUM 0.77 0.47 0.58 225 HIGH 0.95 0.98 0.97 1391 Accuracy 0.94 3000 Macro avg 0.89 0.81 0.83 3000 Weighted avg 0.93 0.94 0.93 3000

Table 5: Classification Report (Test) – Best Quality Classifier Model

[Figure 2]

Figure 2: Confusion matrix showing test and validation results for the XGBoost classifier.

#### 3.2 Category Classification: Results and Applications

This section presents the performance evaluation of the text category classifier developed to automatically assign documents to one of 120 predefined categories. The model was trained and evaluated on a substantial dataset derived from Polish texts, with results summarized in Table 7.

#### Dataset and Setup

The dataset used for this task comprised a total of 58,294 documents. Following standard machine learning practice, the data was partitioned into a training set and a held-out test set. The training set consisted of 52,464 documents (approximately 90% of the data), while the test set contained 5,830 documents (approximately 10%).

A stratified splitting strategy was employed during partitioning to ensure that the proportional representation of each of the 120 categories was maintained in both subsets. This is crucial for reliable evaluation, especially given the large number of potentially imbalanced classes.

#### Modeling Approach

The core classification model utilized a pipeline architecture. Textual data was first processed using a CountVectorizer to convert documents into numerical feature vectors based on word counts, limiting the vocabulary to the 15,000 most frequent terms. These counts were subsequently transformed into Term Frequency-Inverse Document Frequency (TF-IDF) representations using TfidfTransformer, capturing the relative importance of words across the corpus.

Classification was performed using a Linear Support Vector Classifier (LinearSVC), a robust and efficient algorithm for high-dimensional text data. To enable probability estimates and enhance performance, the LinearSVC was wrapped within a CalibratedClassifierCV using isotonic calibration (method=’isotonic’) via 3-fold crossvalidation on the training data.

#### Performance Evaluation

The final trained pipeline was evaluated on the unseen test set (5,830 documents), demonstrating strong performance across the 120 categories.

The overall accuracy achieved was 94.63%, indicating that the vast majority of documents were correctly classified. The high macro-average scores (all above 0.95) are particularly encouraging, suggesting that the classifier performs well across both common and rare categories.

Detailed per-class performance metrics, confusion matrix visualizations, and feature importance analyses were generated to examine specific strengths and weaknesses, identifying categories that may present greater classification challenges.

#### Feature Mean Abs SHAP

oovs 0.5726 average_lines 0.2524 stop_word_ratio 0.2089 polish_diacritics_per_word 0.1851 non_alpha_word_fractions 0.1617 rix 0.1576 short_line_ratio_20 0.1524 avg_paragraph_length 0.1481 lowercase_ratio_md 0.1468 single_word_line_ratio 0.1449 colons_per_sentence 0.1321 not_allowed_chars_ratio 0.1122 special_chars_ratio_md 0.1003 symbol_to_word_ratio 0.0873 duplicate_line_ratio 0.0860 commas_per_sentence 0.0828 blank_lines_ratio 0.0801 polish_diacritics_ratio 0.0799 char_ratio__ 0.0776 emoticons 0.0759 diacritics_std_dev 0.0739 single_char_ratio 0.0704 contextual_word_repetitions_ratio 0.0699 overall_uppercase_ratio 0.0679 avg_dependency_tree_depth 0.0679 short_line_ratio_10 0.0662 char_ratio_> 0.0616

Table 6: Top Features by Mean Absolute SHAP Value for quality classification

Metric Precision Recall F1-score Support

Macro avg 0.9533 0.9563 0.9543 5830 Weighted avg 0.9471 0.9463 0.9461 5830

Table 7: Category Classification Performance on Test Set (N=5,830)

#### Application and Future Directions

The primary objective behind developing the category classifier, alongside the quality classifier, is to curate a dataset that is both thematically diverse and of the highest textual quality.

This dataset is intended for generating synthetic instruction data, leveraging varied and high-quality document collections. Ensuring both thematic breadth and content excellence will enable the creation of robust synthetic instructions, enhancing the effectiveness and generalization capabilities of downstream instruction-following models.

### 4 Synthetic Data Generation for Instruction Tuning

Building on the high-quality category and quality classification pipelines, we introduce a structured process for generating synthetic instruction datasets. This process ensures thematic diversity, high textual quality, and strategic reuse of imperfect data.

#### 4.1 Generation of High-Quality, Thematically Balanced QA Instructions

The core of the synthetic data generation for QA task process involves creating instruction–response pairs from documents classified as HIGH quality and evenly distributed across thematic categories.

[Figure 3]

Figure 3: Distribution of major thematic categories in the Polish text dataset (≥ 0.9%)

Documents meeting the threshold of P(HIGH) > 0.9 under the Bielik v3 quality classifier are selected. Category labels obtained from the thematic classifier ensure that sampling is performed in a balanced manner, avoiding overrepresentation of any single domain.

From these carefully curated inputs, task-oriented QA instructions are synthesized using controlled prompting techniques, ensuring coverage of a wide variety of instruction types, question styles, and domain-specific nuances. Emphasis is placed on factual coherence, clarity, and naturalness of the generated outputs, maintaining strict alignment with the standards of superior-quality human-authored instructions.

#### 4.2 Data Recycling: Improving Imperfect Texts for Inclusion

Not all documents initially meet the strict inclusion criteria. Texts assessed as having medium or borderline quality undergo a dedicated recycling process before being incorporated into the base training corpus.

Specifically, documents categorized as:

- • HIGH-quality texts with a quality model confidence between 50% and 70%,
- • MEDIUM-quality texts with high internal confidence scores,

are subjected to automated refinement using the Bielik v2.3 model. The recycling stage primarily addresses superficial but systematic defects, including:

- • Spelling errors and typographical mistakes,
- • Formatting irregularities and excessive or missing punctuation,

|Category|Count|
|---|---|
|Other Health Politics, Media & News Sport Travel & Tourism Finance Culinary/Food Religion Electronics Beauty IT & Internet Services Automotive Fashion School/Education Home Offices & Municipalities Construction Books & Literature Games Culture & History Cinema, Movies & Series Law & Law Firms|40,337,136 30,432,887 18,066,594 14,308,164 12,168,387 11,182,892 10,537,704 10,499,588 10,403,465 10,337,631<br><br>9,707,525 9,637,918 9,481,582 8,826,348 8,810,490 8,332,005 8,233,476 8,013,594 7,418,487 7,250,791 6,984,202 6,945,003<br><br>|
|——–| |
|Rescue Services Diving Lotteries Bailiff Services Currency Exchange Plumbing Taxi Security Services Postal Services|132,048 130,770 129,202 128,669 117,801 107,582<br><br>97,234 59,082 52,035<br><br>|

Table 8: Distribution of text categories predicted by the classifier. Note: The "Other" category includes texts where the classifier was uncertain and assigned a prediction with less than 20% confidence, as shown in Figure 3.

• Minor OCR artifacts or inconsistencies, while preserving the underlying semantic content of the text.

Following refinement, these recycled documents are reassessed for quality. Only those meeting the revised thresholds are incorporated into the base model training datasets. This approach maximizes data utilization efficiency while maintaining high quality standards.

- 4.3 Summary of the Synthetic Data Curation Strategy The synthetic data curation framework combines two key pillars:

- • Selective instruction generation from top-tier documents, ensuring thematic diversity and linguistic excellence.
- • Strategic data recycling, repairing and upgrading moderately degraded documents to salvage valuable information without compromising quality.

By combining rigorous filtering and thematic balancing for instruction generation, and quality-preserving data recycling for the base training corpus, we construct datasets optimized respectively for instruction tuning and foundational model training, ensuring broad coverage and high factual reliability.

### 5 Post-training

Upon completing the pre-training phase, we transitioned to the post-training phase, which focused on further improving the model’s performance across several domains, including coding, mathematics, logical reasoning, and instruction following.

- 5.1 Supervised Fine-Tuning

To better understand human behavior, the first step involves supervised fine-tuning (SFT), where a pretrained language model is adapted using dialogue-style datasets consisting of prompts and corresponding replies. The subsequent sections provide a detailed overview of the methods applied during this training process.

- 5.1.1 Masked Tokens

We introduced a masked token strategy where the loss function is applied selectively to certain parts of the output. Specifically, we masked the user instructions and control tokens from contributing to the loss Shi et al. [2024]. This approach ensures that training focuses exclusively on the meaningful content tokens, avoiding unwanted optimization signals from non-content parts.

- 5.1.2 Adaptive Learning Rate

Since instruction lengths vary widely, the number of tokens influencing the loss computation can fluctuate. To maintain consistent training dynamics, we implemented an adaptive learning rate (ALR) Ociepa et al. [2024], scaling the base learning rate (LR) according to the square root of the ratio between the current batch’s token count (T) and a reference batch size (BS):

ALR = LR ·

T BS

(1)

- 5.2 Supervised Fine-Tuning Data

The dataset of instructions and dialogues in the Polish language, created for the Bielik 11B v2 release, was cleaned by removing low-quality samples and those containing various errors and defects. Some existing instructions were also regenerated, and completely new instructions were created using the Bielik 11B v2.3 model. The resulting dataset used for training included over 19 million instructions, totaling more than 13 billion tokens.

- 5.3 Supervised Fine-Tuning Hyperparameters

Training employed the AdamW optimizer with parameters β1 = 0.9, β2 = 0.95, and a weight decay of 0.05. The learning rate followed a cosine decay schedule, beginning at 7 × 10−6 and gradually decreasing to 6 × 10−7, with a warmup phase of 50 iterations.

We used a global batch size of 128, with each local batch consisting of a single sample. Gradient clipping was performed with a maximum norm of 1.0, and mixed-precision training was enabled using the bfloat16 format.

To improve efficiency, we applied sample packing, combining multiple dataset samples into a single sequence until reaching the maximum allowed sequence length. The model was trained for 1.2 epochs with a maximum context length of 8,192 tokens.

- 5.4 Preference Learning

- 5.4.1 Preference Training Methods

In our exploration of post-training methodologies for aligning Bielik v3 model with human preferences, we conducted extensive evaluations of various preference optimization techniques. Building upon established methods such as Direct Preference Optimization (DPO), its penalized variant DPO-P, and Odds Ratio Preference Optimization (ORPO), we introduced and assessed a novel approach: Simple Preference Optimization (SimPO).

Direct Preference Optimization (DPO) simplifies the reinforcement learning from human feedback (RLHF) paradigm by eliminating the need for an explicit reward model. It directly optimizes the policy to prefer responses aligned with human preferences.

πθ(yw | x) πref(yw | x) − log

πθ(yl | x) πref(yl | x)

LDPO(πθ;πref) = −E(x,y

w,yl)∼D log σ β log

Here, x represents the input prompt, yw and yl are the preferred and less preferred responses, respectively, πθ denotes the model’s policy, πref is the reference policy, β is a scaling parameter, and σ is the sigmoid function. DPO addresses the complexity of RLHF by providing a stable and computationally efficient alternative that directly incorporates preference data into the training process.

DPO with Penalty (DPO-P) extends DPO by introducing a penalty term to account for uncertainty in preference data, mitigating overfitting to noisy or ambiguous annotations. This penalization adjusts the loss function to reduce the influence of uncertain samples, enhancing the robustness of the model to imperfect preference data. The DPO-P loss function is formulated as:

πθ(yw | x) πref(yw | x) − log

πθ(yl | x) πref(yl | x) − λ · max 0,log

πref(yw | x) πθ(yw | x)

LDPOP(πθ;πref) = −E(x,y

w,yl)∼D log σ β log

Here, λ is a weighting factor for the penalty term. DPO-P addresses the challenge of aligning models without the overhead of additional reference models or complex training procedures.

Odds Ratio Preference Optimization (ORPO) ORPO integrates preference alignment into the supervised fine-tuning (SFT) phase by incorporating an odds ratio-based penalty. This approach eliminates the need for a separate reference model and simplifies the training pipeline. The ORPO loss function is formulated as:

πθ(yw | x) πθ(yl | x)

LORPO = LNLL + λ · log

Where LNLL is the negative log-likelihood loss, and λ is a weighting factor for the odds ratio term. ORPO addresses the challenge of aligning models without the overhead of additional reference models or complex training procedures.

Simple Preference Optimization (SimPO) further streamlines preference optimization by utilizing the average logprobability of a sequence as an implicit reward, removing the necessity for a reference model. It introduces a target reward margin γ to enhance the separation between preferred and less preferred responses. The SimPO objective is defined as:

1 |yl|

1 |yw|

LSimPO = −E(x,y

log πθ(yw | x) −

log πθ(yl | x) − γ

w,yl)∼D log σ β

Here, |y| denotes the length of the response, ensuring length normalization, and γ serves as the target reward margin. SimPO addresses the computational and memory inefficiencies associated with reference models, offering a more efficient and scalable solution for preference alignment.

After conducting extensive evaluations across multiple Polish benchmarks—including we observed that the penalized variant of Direct Preference Optimization (DPO-P) consistently outperformed other methods such as DPO, ORPO, and SimPO. Despite the simplicity and computational efficiency offered by SimPO, DPO-P demonstrated superior alignment with human preferences, particularly in tasks requiring nuanced reasoning and factual accuracy.

#### 5.4.2 Preference Dataset

Compared to the previous Bielik 11B v2 release, for the final phase of post-training with reinforcement learning, we introduced several key enhancements to our Polish preference instruction dataset and response generation pipeline. First, we significantly expanded the dataset to 126,000 instructions (only in Polish language), enriching its diversity and complexity. We also broadened the range of alignment categories to include function calling and tool calling tasks, while substantially increasing the volume of instructions focused on reasoning and mathematics. In addition, we incorporated a large number of conversational instructions to better reflect realistic interaction patterns. In the generation of preferred and rejected responses, we used a broader set of language models, notably including DeepSeek-V3-0324, alongside models used previously. Despite these improvements, we maintain a similar approach to instruction creation and curation, combining manual authoring, perturbation-based enhancement, rigorous deduplication, quality evaluation with reward metamodels, and manual inspection. These refinements ensure that the dataset not only grows in scale but also in quality, better supporting downstream alignment training.

#### 5.4.3 DPO-Positive Hyperparameters

For DPO-Positive (DPO-P) training, we set the loss function parameters to β = 0.1 and λ = 2.5, in accordance with best practices aimed at stabilizing preference-based optimization and preserving the quality of preferred outputs.

The optimization was carried out using the AdamW algorithm, configured with β1 = 0.9, β2 = 0.95, and no weight decay. A fixed learning rate of 5 × 10−7 was used, preceded by 50 warmup steps. The total number of training steps amounted to 1,800.

Training was performed with a global batch size of 64, where each local batch consisted of a single example. Gradient clipping was applied with a maximum norm of 1.0. Mixed-precision training was employed using the bfloat16 format.

#### 5.5 Reinforcement Learning

For Reinforcement Learning (RL) training stage we employed Group Relative Policy Optimization (GRPO) Shao

- et al. [2024] to fine-tune our Bielik-4.5B-v3 language model, utilizing a curated dataset of 12,000 Polish mathematical problems with verifiable solutions (RLVR). This subset was selected from a larger corpus of approximately 100,000 Polish-language math problems developed during the Bielik v3 project. The training process was conducted on an 8×GPU H100 cluster (Athena - Cyfronet) using the Volcano Engine Reinforcement Learning (VERL) framework Sheng

- et al. [2024] , which offers scalable and modular support for large language model (LLM) reinforcement learning workflows.

GRPO is a reinforcement learning algorithm designed to enhance reasoning capabilities in LLMs by addressing limitations found in traditional methods like Proximal Policy Optimization (PPO). Unlike PPO, which relies on a separate value function (critic) to estimate the expected return, GRPO eliminates the need for this component, thereby reducing memory consumption and computational complexity. Instead, GRPO evaluates multiple responses generated for the same prompt, computes their rewards using a reward model, and calculates the advantage of each response relative to the group’s average reward. This group-based advantage estimation allows the model to update its policy by favoring responses that outperform the average, leading to more stable and efficient training.

In GRPO, for each input prompt q, the model generates a group of G responses {a1,a2,...,aG}. Each response ai is evaluated using a reward function R(q,ai), which assesses the quality of the response. The mean µ and standard deviation σ of the rewards within the group are computed as follows:

µ =

G

1 G

R(q,ai), σ =

i=1

G

1 G

(R(q,ai) − µ)2

i=1

The advantage Ai of each response ai is then calculated by normalizing its reward relative to the group’s statistics:

R(q,ai) − µ σ

Ai =

This group-relative advantage estimation allows the model to identify which responses are better or worse compared to others in the same group, without requiring an explicit value function.

The policy is updated by maximizing the following objective function, which incorporates the advantage and a clipping mechanism to ensure stable updates:

1 G

L(θ) =

G

min(ri(θ)Ai,clip(ri(θ),1 − ϵ,1 + ϵ)Ai)

i=1

θ(ai|q)

Here, ri(θ) = π

πθold(ai|q) is the probability ratio between the new and old policies, and ϵ is a hyperparameter that controls the extent of clipping.

Additionally, GRPO incorporates a Kullback-Leibler (KL) divergence penalty to prevent the updated policy from deviating too much from a reference policy πref, typically the pre-trained model before fine-tuning. The KL penalty is added to the loss function as follows:

##### Ltotal(θ) = L(θ) − β · KL[πθ ∥ πref]

Where β is a coefficient that balances the importance of the KL penalty.

By leveraging group-relative advantages and eliminating the need for a value function, GRPO offers a more efficient and stable approach to fine-tuning LLMs, particularly in tasks that require complex reasoning, such as mathematical problem-solving.

The application of GRPO in our training regimen not only improved the model’s performance on mathematical tasks but also enhanced its capabilities in other reasoning-intensive areas. This suggests that GRPO effectively strengthens the model’s general reasoning abilities, making it a valuable approach for fine-tuning LLMs across diverse domains.

The optimization settings included a learning rate of 1 × 10−6, a global batch size of 128 (with a local batch size of 16 per GPU). To ensure stable policy updates, KL divergence regularization was applied with a coefficient of 0.001, using the low-variance KL loss type.

#### 5.6 Model Merging

Similar to the Bielik 11B v2 release, we applied a range of merging strategies to refine model quality; however, the leading method throughout the development of Bielik 11B v3 is the linear merging approach. This method served as the primary technique for improving model quality both after Supervised Fine-Tuning (SFT) and after Reinforcement Learning from Human Feedback (RLHF) stages. By consistently employing linear combinations of models at each phase, we ensured stable integration of improvements while maintaining the stylistic and functional consistency of the model outputs.

### 6 Evaluation

We evaluated the Bielik v3 models on several benchmarks to assess their performance across different language understanding and generation tasks, as detailed in the following subsections.

The models were evaluated on the following benchmarks:

- • Open PL LLM Leaderboard (Polish)
- • Polish EQ-Bench (Polish)
- • CPTUB Leaderboard (Polish)
- • Polish Medical Leaderboard (Polish)
- • Polish Linguistic and Cultural Competency Benchmark (PLCC) (Polish)
- • Open LLM Leaderboard
- • MixEval
- • Berkeley Function-Calling Leaderboard

#### 6.1 Open PL LLM Leaderboard

The Open PL LLM Leaderboard, based on the Open LLM Leaderboard v1 [Beeching et al., 2023a], evaluates models on various NLP tasks, including: sentiment analysis, categorization, short answer question answering, and text classification, but does not test their conversational capabilities [Wróbel et al., 2024, Ociepa et al., 2024]. The leaderboard utilizes the lm-evaluation-harness framework for model evaluation [Gao et al., 2024].

#### Tasks:

- • polemo2: Sentiment analysis of online consumer reviews across four domains (medicine, hotels, products, university) with four-class labeling (positive, negative, neutral, ambiguous) [Koco´n et al., 2019]; metric: accuracy.
- • klej-ner: Named entity recognition in sentences containing single-type entities, classifying into six categories (no entity, place, person, organization, time, geographical name) [Rybak et al., 2020]; metric: accuracy.
- • 8tags: Topic classification of social media headlines into eight categories (film, history, food, medicine, motorization, work, sport, technology) [Dadas et al., 2020]; metric: accuracy.
- • belebele: Machine reading comprehension for question answering [Bandarkar et al., 2024]; metric: accuracy.

- • dyk: Question answering based on human-annotated pairs from Wikipedia’s "Did You Know" section [Marcinczuk et al., 2013]; metric: binary F1.
- • ppc: Text similarity assessment using manually labeled sentence pairs (exact paraphrases, close paraphrases, non-paraphrases) [Dadas, 2022]; metric: accuracy.
- • psc: Summarization of news articles [Ogrodniczuk and Kope´c, 2014]; metric: binary F1.
- • cbd: Text classification for cyberbullying and hate-speech detection [Ptaszynski et al., 2023]; metric: macro F1.
- • polqa: Open-domain question answering from the "Jeden z dziesi˛eciu" TV show, with and without context (abstractive QA/RAG) [Rybak et al., 2024]; metric: accuracy, levenshtein.
- • poquad: Context-based extractive question answering (QA/RAG) [Tuora et al., 2023]; metric: levenshtein.
- • eqbench: emotional intelligence benchmark [Paech, 2024]; metric: custom.

Most of the tasks are multiple-choice tests, which means that the model chooses the correct answer from a set of options. They are implemented as two types of tests:

- • Loglikelihood: We choose the highest probability token from the given set, e.g., ABCD. These tests are suitable for base models.
- • Generate: Model generates answer freely.

All tasks are evaluated in both 0-shot and 5-shot settings, with the average score across all tasks normalized by baseline scores.

It is important to note that PLLuM models are not included in this leaderboard, as they were trained on training portions of the tasks used in the benchmark (except for the Belebele and EQ-Bench tasks), unlike all other models present on the leaderboard, to the best of our knowledge. The authors of the datasets used in the benchmark are primarily PLLuM consortium members.

As shown in Table 9, the Bielik-4.5B-v3 model achieves an impressive average score of 54.94, making it competitive with much larger models. This result is particularly noteworthy considering it outperforms several models with significantly more parameters, such as Qwen2.5-7B (53.35), EuroLLM-9B (50.03), and SOLAR-10.7B-v1.0 (47.54). The smaller Bielik-1.5B-v3 model achieves a score of 31.48, which is comparable to Qwen2.5-1.5B (31.83) and Llama-3.2-3B (31.89), despite its compact size.

The instruction-tuned models demonstrate substantial improvements over their base counterparts. As shown in Table 10, Bielik-4.5B-v3.0-Instruct achieves a score of 56.13, outperforming Qwen2.5-7B-Instruct (54.93) and Mistral-NemoInstruct-2407 (55.27) despite having fewer parameters. Most impressively, Bielik-1.5B-v3.0-Instruct scores 41.36, exceeding the performance of Qwen2.5-3B-Instruct (41.23) with approximately half the parameters, and coming close to Phi-4-mini-instruct (43.30) which has more than twice the parameter count.

These results demonstrate the effectiveness of our training approach for the Bielik v3 models, which achieve remarkable parameter efficiency and strong performance on Polish language tasks. When considering the performance-to-parameter ratio, both the 1.5B and 4.5B models represent significant advancements in resource-efficient language modeling for Polish.

#### 6.2 Polish EQ-Bench

The Polish Emotional Intelligence Benchmark, a localized Polish version of the original EQ-Bench Paech [2024], evaluates language models’ emotional intelligence capabilities across various dimensions of emotional understanding and response. This benchmark assesses models’ ability to comprehend, interpret, and appropriately respond to emotionally complex situations in Polish language contexts.

The results in Table 11 highlight the performance of Bielik v3 models on the emotionally nuanced Polish EQ-Bench. The Bielik-4.5B-v3.0-Instruct achieves a score of 53.58, which is particularly impressive for its parameter count. Despite having only 4.8B parameters, it outperforms several much larger models including PLLuM-12B-chat (52.26) and multiple PLLuM models with significantly more parameters. It also performs comparably to EuroLLM-9B-Instruct (54.10) with nearly half the parameters.

This efficiency is remarkable when considering the emotional intelligence capabilities achieved with substantially fewer parameters than comparable models. The performance gap between Bielik-4.5B-v3.0-Instruct and larger models like Bielik-11B-v2 variants (approximately 15-18 points difference) reflects the trade-offs between model size and performance, while demonstrating that even compact models can exhibit meaningful emotional intelligence capabilities.

Qwen2.5-72B 72.7 67.38 Qwen2.5-32B 32.8 66.73 Qwen-72B 72.7 66.02 Qwen2.5-14B 14.8 62.71 Meta-Llama-3-70B 70.6 62.07

- Qwen1.5-72B 72.7 61.11 Meta-Llama-3.1-70B 70.6 60.87 Mixtral-8x22B-v0.1 141.0 60.75 Mistral-Small-24B-Base-2501 24.0 59.90

- Qwen1.5-32B 32.8 58.71 Bielik-11B-v2 11.2 58.14
- Qwen2.5-7B 7.0 53.35 EuroLLM-9B 9.2 50.03 Qwen-7B 7.0 49.39 SOLAR-10.7B-v1.0 10.7 47.54 Mistral-Nemo-Base-2407 12.2 47.28 internlm2-20b 20.0 47.15 Bielik-4.5B-v3 4.8 45.47

- Qwen2.5-3B 3.0 44.59 Meta-Llama-3.1-8B 8.0 43.77 Meta-Llama-3-8B 8.0 43.30 Qwen1.5-72B 72.3 39.51 Mistral-7B-v0.3 7.0 38.88 Mistral-7B-v0.2 7.0 38.81

- Qwen1.5-7B 7.0 37.92 Bielik-7B-v0.1 7.2 34.34 Qra-13b 13.0 33.90 Llama-3.2-3B 3.0 31.89
- Qwen2.5-1.5B 1.5 31.83 Bielik-1.5B-v3 1.6 31.48 Qra-7b 7.0 16.60

Table 9: Open PL LLM Leaderboard results for base models (5-shot evaluation)

The Bielik-1.5B-v3.0-Instruct model, with just 1.6B parameters, achieves a more modest score of 13.88, comparable to PLLuM-12B-nc-instruct (13.11) despite having only about 13% of the parameters.

When considering the performance gradient across model sizes, we observe that the Bielik-4.5B-v3.0-Instruct achieves 76% of the performance of our best Bielik-11B-v2.5-Instruct model (72.00) with only 43% of the parameters. This efficient scaling pattern demonstrates the effectiveness of our training approach in balancing performance with computational efficiency across the Bielik model family.

These results demonstrate that the specialized training methodologies employed for Bielik v3 models enable them to achieve competitive performance on emotionally nuanced tasks despite their compact size, highlighting the effectiveness of the model architecture and training approach used for Polish language emotional intelligence capabilities.

#### 6.3 Complex Polish Text Understanding Benchmark (CPTUB)

The Complex Polish Text Understanding Benchmark (CPTUB) Sowa et al. [2024] is specifically designed to evaluate language models’ proficiency in interpreting complex Polish texts. Unlike traditional tasks that focus on explicit meaning, CPTUB assesses the models’ capacity to understand implied meanings and handle cognitively challenging questions. The benchmark comprises two main components:

- • Implicatures: Evaluates a model’s ability to interpret implied meanings, including sarcasm, idiomatic expressions, and phraseological compounds. This component tests sensitivity to nuanced, context-dependent inferences through three subtasks:

- – Sentiment: Correctly identifying the emotional tone beyond literal expressions
- – Language understanding: Interpreting the underlying intentions of text authors
- – Phraseology: Recognizing and explaining fixed or semi-fixed expressions whose meanings cannot be inferred from their individual components

Mistral-Large-Instruct-2411 123.0 69.84 Meta-Llama-3.1-405B-Instruct-FP8 405.0 69.44 Mistral-Large-Instruct-2407 123.0 69.11 Qwen2.5-72B-Instruct 72.7 67.92 QwQ-32B-Preview 32.8 67.01 Llama-3.3-70B-Instruct 70.6 66.40

- Qwen2-72B-Instruct 72.7 65.87 Bielik-11B-v2.3-Instruct 11.2 65.71 Bielik-11B-v2.2-Instruct 11.2 65.57 Meta-Llama-3.1-70B-Instruct 70.6 65.49 Bielik-11B-v2.1-Instruct 11.2 65.45 Mixtral-8x22B-Instruct-v0.1 141.0 65.23 Bielik-11B-v2.0-Instruct 11.2 64.98 Meta-Llama-3-70B-Instruct 70.6 64.45
- Qwen3-32B 32.8 64.24 Llama-4-Scout-17B-16E-Instruct 109.0 64.21 Bielik-11B-v2.5-Instruct 11.2 63.95 Mistral-Small-24B-Instruct-2501 24.0 62.97 phi-4 14.7 62.57 Qwen3-14B 14.8 62.24 Mistral-Small-Instruct-2409 22.2 61.41 Qwen2.5-32B-Instruct 32.8 61.21 Qwen2.5-14B-Instruct 14.8 59.91 aya-23-35B 35.0 56.37

Bielik-4.5B-v3.0-Instruct 4.8 56.13 Qwen3-8B 8.2 55.78 Qwen3-4B 4.0 55.49 Mistral-Nemo-Instruct-2407 12.2 55.27 Qwen2.5-7B-Instruct 7.6 54.93 Mistral-7B-Instruct-v0.3 7.2 47.74 Mistral-7B-Instruct-v0.2 7.2 45.95 Bielik-7B-Instruct-v0.1 7.2 44.70 Phi-4-mini-instruct 3.8 43.30

Bielik-1.5B-v3.0-Instruct 1.6 41.36 Qwen2.5-3B-Instruct 3.1 41.23 Qwen3-1.7B 2.0 38.34 Mistral-7B-Instruct-v0.1 7.0 33.11 Qwen2.5-1.5B-Instruct 1.5 31.89

Table 10: Open PL LLM Leaderboard results for instruction-tuned models (5-shot evaluation)

- • Tricky Questions: Assesses a model’s capability to address challenging questions characterized by logical puzzles, semantic ambiguity, logical inconsistencies, absurdity, and humor. This component specifically targets the model’s reasoning skills and ability to avoid hallucinations when faced with ambiguous or nonsensical queries.

As shown in Table 12, the Bielik v3 models demonstrate impressive performance on this challenging benchmark, particularly in relation to their parameter counts. The Bielik-4.5B-v3.0-Instruct model achieves an overall score of 3.38, which is remarkable for a model with only 4.8B parameters. This positions it in the same performance tier as significantly larger models, including several with over 10x the parameter count. Several key observations can be made about the Bielik v3 models’ performance:

- 1. Exceptional parameter efficiency: Bielik-4.5B-v3.0-Instruct (3.38) outperforms phi-4 (3.30) despite having only about a third of the parameters (4.8B vs. 14.7B). It also surpasses all PLLuM models regardless of size, including PLLuM-8x7B variants with nearly 10x more parameters.
- 2. Strong implicature handling: Bielik-4.5B-v3.0-Instruct shows particularly strong performance in implicatures (3.68), exceeding even some Bielik-11B-v2 variants and models like Mixtral-8x22B-Instruct-v0.1 (3.67). This suggests superior understanding of nuanced Polish linguistic features like idioms and contextual meaning.

Model Parameters (B) Score Mistral-Large-Instruct-2407† 123.0 78.07 Mistral-Large-Instruct-2411† 123.0 77.29 Meta-Llama-3.1-405B-Instruct-FP8 405.0 77.23 gpt-4o-2024-08-06 Unknown 75.15 gpt-4-turbo-2024-04-09 Unknown 74.59 Mistral-Small-Instruct-2409 22.2 72.85 Llama-PLLuM-70B-chat 70.6 72.56 Meta-Llama-3.1-70B-Instruct 70.6 72.53 Bielik-11B-v2.5-Instruct 11.2 72.00 Qwen2-72B-Instruct 72.7 71.23 Meta-Llama-3-70B-Instruct 70.6 71.21 gpt-4o-mini-2024-07-18 Unknown 71.15 Qwen2.5-32B-Instruct 32.8 71.15 Bielik-11B-v2.3-Instruct 11.2 70.86 Llama-3.3-70B-Instruct 70.6 70.73 Llama-PLLuM-70B-instruct 70.6 69.99 WizardLM-2-8x22B 141.0 69.56 Qwen2.5-14B-Instruct 14.8 69.17 Bielik-11B-v2.2-Instruct 11.2 69.05

- Bielik-11B-v2.0-Instruct 11.2 68.24 glm-4-9b-chat 9.0 61.79 Mistral-Nemo-Instruct-2407 12.2 61.76
- Bielik-11B-v2.1-Instruct 11.2 60.07 EuroLLM-9B-Instruct 9.2 54.10 Bielik-4.5B-v3.0-Instruct 4.8 53.58 PLLuM-12B-chat 12.2 52.26 PLLuM-8x7B-nc-chat† 46.7 47.29 Llama-PLLuM-8B-chat 8.0 46.20 Llama-3.2-3B-Instruct 3.2 46.19 PLLuM-8x7B-chat 46.7 45.22 PLLuM-8x7B-nc-instruct† 46.7 41.75 PLLuM-8x7B-instruct 46.7 39.55 PLLuM-12B-instruct 12.2 36.21 Qwen2.5-3B-Instruct 3.1 35.87 PLLuM-12B-nc-chat† 12.2 35.41 Llama-PLLuM-8B-instruct 8.0 31.59 Qwen2.5-1.5B-Instruct 1.5 27.63 Llama-3.2-1B-Instruct 1.2 17.82 gemma-1.1-2b-it 2.5 16.47 Bielik-1.5B-v3.0-Instruct 1.6 13.88 PLLuM-12B-nc-instruct† 12.2 13.11 †Models with a non-commercial license.

Table 11: Polish EQ-Bench results for various models.

- 3. Phraseology strength: The Bielik-4.5B-v3.0-Instruct model scores 3.67 in phraseology, notably higher than many larger models including all Bielik-11B variants. This indicates exceptional ability to understand Polish idiomatic expressions and fixed phrases.
- 4. Sentiment analysis competence: Both Bielik v3 models perform well in sentiment analysis, with the 1.5B model scoring 3.53 - higher than many larger models including Qwen2.5-3B-Instruct (2.95) despite having half the parameters.
- 5. Tricky questions challenges: The area with the most room for improvement is in handling tricky questions, where Bielik-4.5B-v3.0-Instruct scores 2.46. This is consistent with the pattern seen across most models, as this category tests complex reasoning abilities that typically benefit from larger model scales.

The performance of Bielik-1.5B-v3.0-Instruct is similarly impressive within its parameter class. At just 1.6B parameters, it achieves an overall score of 2.36, outperforming models with substantially more parameters like Phi-4-mini-instruct (2.17 with 3.8B parameters) and performing comparably to Qwen2.5-3B-Instruct (2.50 with 3.1B parameters). Its

Model Params (B) Overall Implicatures Senti- Language Phrase- Tricky Average Average ment Understanding ology Questions

DeepSeek-R1 685.0 4.14 4.14 4.49 4.35 3.60 4.12 Mistral-Large-Instruct-2411† 123.0 4.00 4.10 4.33 3.98 3.99 3.72 Qwen2.5-72B-Instruct 72.7 3.95 3.99 4.08 3.97 3.93 3.81 Mistral-Large-Instruct-2407† 123.0 3.93 4.03 4.23 4.00 3.86 3.65 Llama-4-Maverick-17B-128E-Instruct-FP8 402.0 3.93 3.99 4.39 4.11 3.48 3.76 gemma-3-27b-it 27.4 3.81 3.90 3.88 3.79 4.03 3.53 Meta-Llama-3-70B-Instruct 70.6 3.78 3.81 4.13 3.82 3.47 3.71 Qwen2.5-32B-Instruct 32.8 3.75 3.80 3.81 3.57 4.04 3.59 Llama-4-Scout-17B-16E-Instruct-FP8 402.0 3.75 3.94 4.10 3.81 3.90 3.19 Bielik-11B-v2.3-Instruct 11.2 3.63 3.77 3.97 3.79 3.55 3.22

- Bielik-11B-v2.1-Instruct 11.2 3.61 3.66 3.96 3.92 3.47 3.47 Mixtral-8x22B-Instruct-v0.1 141.0 3.56 3.67 3.78 3.68 3.55 3.24 Qwen2.5-14B-Instruct 14.8 3.55 3.62 3.91 3.57 3.37 3.34 Llama-PLLuM-70B-chat 70.6 3.53 3.63 3.94 3.61 3.35 3.21 Bielik-11B-v2.5-Instruct 11.2 3.48 3.67 4.01 3.86 3.13 2.91
- Bielik-11B-v2.2-Instruct 11.2 3.46 3.57 3.72 3.73 3.25 3.12 Bielik-4.5B-v3.0-Instruct 4.8 3.38 3.68 3.76 3.61 3.67 2.46 Llama-PLLuM-70B-instruct 70.6 3.33 3.56 3.78 3.63 3.26 2.63 phi-4 14.7 3.30 3.50 3.72 3.54 3.24 2.72 Bielik-11B-v2.0-Instruct 11.2 3.26 3.61 3.97 3.75 3.13 2.20 PLLuM-12B-nc-chat† 12.2 3.15 3.33 3.22 3.23 3.54 2.62 PLLuM-12B-chat 12.2 3.14 3.32 3.32 3.21 3.43 2.59 PLLuM-8x7B-nc-instruct† 46.7 3.11 3.56 3.88 3.59 3.22 1.76 PLLuM-12B-instruct 12.2 3.09 3.49 3.71 3.17 3.59 1.90 Qwen2.5-7B-Instruct 7.62 3.07 3.23 3.56 3.03 3.10 2.58 PLLuM-8x7B-nc-chat† 46.7 3.03 3.44 3.76 3.48 3.08 1.80 Meta-Llama-3.1-8B-Instruct 8.0 3.01 3.31 3.97 3.38 2.58 2.11 PLLuM-8x7B-instruct 46.7 3.01 3.51 3.59 3.47 3.46 1.51 PLLuM-8x7B-chat 46.7 3.01 3.41 3.44 3.45 3.35 1.78 Meta-Llama-3-8B-Instruct 8.0 3.00 3.17 3.33 3.15 3.04 2.48 Llama-PLLuM-8B-chat 8.0 2.92 3.14 3.13 2.93 3.36 2.25 Bielik-7B-Instruct-v0.1 7.2 2.88 3.13 3.59 3.48 2.32 2.16 Llama-PLLuM-8B-instruct 8.0 2.82 3.20 3.24 2.90 3.46 1.66 gemma-2-2b-it 2.6 2.65 2.80 3.40 2.90 2.10 2.21 Qwen2.5-3B-Instruct 3.1 2.50 2.73 2.95 2.46 2.80 1.81 Bielik-1.5B-v3.0-Instruct 1.6 2.36 2.75 3.53 2.33 2.38 1.22 Phi-4-mini-instruct 3.8 2.17 2.46 2.69 2.43 2.25 1.30 Llama-3.2-3B-Instruct 3.2 2.00 2.26 2.76 2.30 1.72 1.22 EuroLLM-1.7B-Instruct 1.7 1.76 2.10 2.24 1.79 2.26 0.76 Qwen2.5-1.5B-Instruct 1.5 1.76 2.12 2.79 1.35 2.23 0.66 †Models with a non-commercial license.

- Table 12: Complex Polish Text Understanding Benchmark (CPTUB) results across different evaluation categories

strong sentiment analysis score (3.53) is particularly noteworthy, matching or exceeding many models with 3-7x more parameters.

These results highlight the effectiveness of the specialized training methodologies employed for the Bielik v3 models, particularly the focus on Polish-specific data curation and the innovative techniques described in previous sections, as shown in Tables 1 and 2. The models demonstrate that through careful optimization, even relatively small models can achieve competitive performance on complex linguistic tasks that traditionally favor much larger architectures.

#### 6.4 Polish Medical Leaderboard

The Polish Medical Leaderboard evaluates language models on Polish Board Certification Examinations (Pa´nstwowy Egzamin Specjalizacyjny, PES) from years 2018-2022. This benchmark assesses models’ medical knowledge and reasoning capabilities in a Polish-language medical context, using datasets from speakleash/PES-2018-2022, which is based on amu-cai/PES-2018-2022 Pokrywka et al. [2024].

###### Model Parameters (B) Average (%)

Meta-Llama-3.1-405B-Instruct-FP8 405.0 69.20 Mistral-Large-Instruct-2407† 123.0 64.28 Qwen2.5-72B-Instruct 72.7 63.89 Meta-Llama-3.1-70B-Instruct 70.6 61.75 Qwen2-72B-Instruct 72.7 61.35 Meta-Llama-3-70B-Instruct 70.6 57.51 Qwen2.5-32B 32.8 55.69 Qwen2.5-32B-Instruct 32.8 54.52 Qwen2.5-14B-Instruct 14.8 49.60 Bielik-11B-v2.5-Instruct 11.2 44.85 GLM-4-9b-chat 9.0 44.54 Mistral-Small-Instruct-2409 22.2 43.60 Bielik-4.5B-v3.0-Instruct 4.8 43.55 Bielik-11B-v2.3-Instruct 11.2 43.26

- Bielik-11B-v2.1-Instruct 11.2 43.16
- Bielik-11B-v2.2-Instruct 11.2 43.05 Qwen2.5-7B-Instruct 7.6 42.69 Bielik-11B-v2.0-Instruct 11.2 41.53 Meta-Llama-3.1-8B-Instruct 8.0 40.60 Mistral-Nemo-Instruct-2407 12.2 40.36 Bielik-11B-v2 11.2 39.98 Qwen2.5-3B-Instruct 3.0 37.72 Bielik-1.5B-v3.0-Instruct 1.6 34.63 Qwen2.5-1.5B-Instruct 1.5 32.64 Mistral-7B-Instruct-v0.3 7.0 31.24 †Models with a non-commercial license.

- Table 13: Polish Medical Leaderboard results (5-shot setting) showing model performance on Polish Board Certification Examinations.

Bielik v3’s performance: In the Polish Medical Leaderboard (Table 13), the Bielik v3 models demonstrate impressive medical reasoning capabilities relative to their model size:

- • Bielik-4.5B-v3.0-Instruct achieves a score of 43.55%, which is remarkably close to Bielik-11B-v2.5-Instruct (44.85%) despite having less than half the parameters
- • The smaller Bielik-1.5B-v3.0-Instruct scores 34.63%, outperforming Qwen2.5-1.5B-Instruct (32.64%) and significantly outperforming larger models like Mistral-7B-Instruct-v0.3 (31.24%)
- • Notably, Bielik-4.5B-v3.0-Instruct matches or outperforms several much larger models, including MistralSmall-Instruct-2409 (43.60%) which has 22.2B parameters

Performance context: The benchmark highlights several important insights about Bielik v3’s medical capabilities:

- • Both Bielik v3 models achieve strong parameter efficiency, with the 4.5B model performing at nearly the same level as models 2-3 times its size
- • The models show effective cross-domain generalization from general Polish language understanding to specialized medical knowledge, despite not having domain-specific medical training
- • The gap between Bielik v3 models and top-performing models like Meta-Llama-3.1-405B-Instruct (69.20%) reflects the expected scaling relationship between model size and specialized domain knowledge

These results emphasize the effectiveness of the training methodologies employed for the Bielik v3 models, enabling strong performance on specialized domain knowledge even at smaller parameter counts. This makes the Bielik v3 models particularly valuable for practical applications where computational efficiency must be balanced with domain-specific performance.

#### 6.5 Polish Linguistic and Cultural Competency Benchmark (PLCC)

The Polish Linguistic and Cultural Competency Benchmark (PLCC) [Dadas et al., 2025] is a specialized evaluation framework that tests language models’ grasp of Polish cultural knowledge and context. Unlike conventional NLP

assessments, PLCC delves deeper into cultural understanding through 600 carefully designed questions spanning six key domains: history, geography, culture & tradition, art & entertainment, grammar, and vocabulary.

The benchmark’s questions probe models’ familiarity with various aspects of Polish heritage, including cultural references, historical milestones, traditional customs, folklore, literary works, and contemporary popular culture. These elements are crucial for achieving authentic language comprehension that goes beyond mere text processing. The questions vary in complexity from widely known information to specialized regional knowledge, presented in both multiple-choice and open-ended formats that demand precise responses with specific facts, dates, names, or concepts.

The PLCC results (Table 14) reveal several insights about Bielik v3 models’ cultural competency:

Parameter efficiency: Bielik-4.5B-v3.0-Instruct achieves 42.33% on PLCC despite its small size, outperforming several larger models including Qwen-2.5-14B (26.67%) and Phi-4 (29.17%). Similarly, Bielik-1.5B-v3.0-Instruct reaches 27.50%, demonstrating strong performance for its compact parameter count.

Size-performance tradeoff: While Bielik v3 models score lower than their larger Bielik v2 counterparts (Bielik11B-v2.2-Instruct at 63.00%), they provide viable alternatives for resource-constrained deployments while maintaining reasonable cultural knowledge capabilities.

Competitive small-model performance: Bielik-4.5B-v3.0-Instruct shows competitive results compared to models with similar parameter counts, highlighting the effectiveness of our training methodology in preserving cultural knowledge despite parameter reduction.

Progressive improvements: The performance gap between Bielik-1.5B-v3.0-Instruct (27.50%) and Bielik-4.5Bv3.0-Instruct (42.33%) demonstrates the benefits of additional parameters for cultural knowledge retention, while both maintain efficient footprints.

#### 6.6 Open LLM Leaderboard

The Open LLM Leaderboard [Beeching et al., 2023b] evaluates models on various English language tasks, providing insights into the model’s performance across different linguistic challenges.

#### 6.7 MixEval

MixEval [Ni et al., 2024] is an English-language benchmark grounded in verified data, created to assess Large Language Models (LLMs) both efficiently and reliably. Its main characteristics include:

- 1. Built from a collection of pre-existing benchmark datasets
- 2. Demonstrates strong alignment with Chatbot Arena rankings, showing a 0.96 correlation
- 3. Executes locally with minimal overhead, requiring just 6% of the time and cost of MMLU

This benchmark offers a dependable and fast approach for evaluating LLMs, making it a practical choice for continuous performance tracking and model comparison. Results are presented in Table 17.

#### 6.8 Berkeley Function-Calling Leaderboard

The Berkeley Function-Calling Leaderboard (BFCL) Yan et al. [2024] is designed to measure the proficiency of language models in accurately invoking functions (tools) using realistic input data. This evaluation is critical for determining how effectively models can interact with APIs and external systems—an essential skill for deploying LLMs in areas such as software engineering, data processing, and automated workflows.

The benchmark employs Abstract Syntax Tree (AST) metrics to gauge function call correctness across a range of test types:

- • Expert Curated (Non-live) dataset: A collection of hand-crafted, static test cases developed by domain experts to assess function calling in controlled environments
- • User Contributed (Live) dataset: Real-time, user-submitted examples that reflect function calling in authentic, dynamic situations

###### Model Parameters (B) Average Score (%)

Gemini-2.5-Pro-Exp-03-25 Unknown 89.50 DeepSeek-R1 685.0 76.00 DeepSeek-v3-0324 685.0 71.00 DeepSeek-v3 685.0 69.17 PLLuM-8x7B-nc-chat† 46.7 68.17 Llama-3.1-Tulu-3-405B 405.0 63.83

- Bielik-11B-v2.2-Instruct 11.2 63.00
- Bielik-11B-v2.3-Instruct 11.2 62.17 GPT-4.1-mini-2025-04-14 Unknown 62.17 Bielik-11B-v2.1-Instruct 11.2 61.00 Llama-3.1-405B 405.0 60.00 PLLuM-12B-nc-chat† 12.2 59.50 Llama-PLLuM-70B-chat 70.6 58.50 Llama-4-Maverick 402.0 58.17 Command-A-03-2025† 111.0 56.17 Mistral-Large-2407† 123.0 54.17 PLLuM-8x7B-chat 46.7 54.17 Mistral-Large-2411† 123.0 52.00 WizardLM-2-8x22B 141.0 51.50 Qwen-Max Unknown 50.83 Command-R-Plus-08-2024† Unknown 50.17 Mixtral-8x22B 141.0 49.83 Command-R-Plus-04-2024† Unknown 49.33 Llama-3.3-70B 70.6 48.83 Llama-3.1-70B 70.0 47.83 Gemma-3-27B 27.4 47.33 PLLuM-12B-chat 12.2 47.00 Bielik-7B-Instruct-v0.1 7.0 46.67 Mistral-Small-3.1-24B-2503 24.0 43.33

- Llama-3.0-70B 70.0 43.00 Gemma-2-27B 27.0 42.67 Bielik-4.5B-v3.0-Instruct 4.8 42.33 Llama-4-Scout 109.0 41.50 EuroLLM-9B 9.0 41.00 Qwen-2.5-72B 72.7 39.17 Mistral-Small-24B-2501 24.0 39.00 Llama-PLLuM-8B-chat 8.0 38.50 Qwen3-32B 32.8 37.67 Mixtral-8x7B 46.7 35.33 Qwen-2.5-32B 32.8 30.50 Qwen3-14B 14.8 30.33 Gemma-2-9B 9.0 29.17 Phi-4 14.7 29.17 Bielik-1.5B-v3.0-Instruct 1.6 27.50 Qwen-2.5-14B 14.8 26.67 Qwen3-8B 8.2 26.00 Mistral-Nemo 12.2 23.00 Command-R-7B† 7.0 22.83
- Llama-3.1-8B 8.0 22.67 Mistral-7B-v0.3 7.2 21.83 Ministral-8B 8.0 20.67 Qwen-2.5-7B 7.0 17.67 †Models with a non-commercial license.

- Table 14: Polish Linguistic and Cultural Competency Benchmark (PLCC) results for open-source models. Closed proprietary models have been excluded from this comparison.

• Multi-turn interactions: Evaluates the model’s capability to preserve and utilize conversational context over multiple exchanges

###### Model AVG arc_challenge hellaswag truthfulqa_mc2 mmlu winogrande gsm8k

Qwen1.5-14B 66.70 56.57 81.08 52.06 69.36 73.48 67.63 Bielik-11B-v2 65.87 60.58 79.84 46.13 63.06 77.82 67.78 Qwen-14B 65.86 58.28 83.99 49.43 67.70 76.80 58.98 Meta-Llama-3-8B 62.62 60.24 82.23 42.93 66.70 78.45 45.19 Bielik-4.5B-v3 61.02 51.19 73.01 45.63 61.32 71.35 63.61

- Mistral-7B-v0.1 60.97 59.98 83.31 42.15 64.16 78.37 37.83
- Mistral-7B-v0.2 60.37 60.84 83.08 41.76 63.62 78.22 34.72 Bielik-1.5B-v3 53.64 46.93 64.28 42.47 55.13 63.38 49.66 Bielik-7B-v0.1 49.98 45.22 67.92 47.16 43.20 66.85 29.49

Table 15: Open LLM Leaderboard results for base models

Model AVG arc_challenge hellaswag truthfulqa_mc2 mmlu winogrande gsm8k

SOLAR-10.7B-Instruct-v1.0 74.20 71.08 88.16 71.43 66.21 83.58 64.75 Phi-3-medium-4k-instruct 73.45 67.32 85.76 57.71 77.83 72.69 79.38

- Bielik-11B-v2.2-Instruct 69.86 59.90 80.16 58.34 64.34 75.30 81.12
- Bielik-11B-v2.3-Instruct 69.82 59.30 80.11 57.42 64.57 76.24 81.27 Bielik-11B-v2.1-Instruct 69.82 59.56 80.20 59.35 64.18 75.06 80.59 openchat-3.5-0106-gemma 69.42 64.68 81.08 54.93 64.69 78.30 72.86 Bielik-11B-v2.0-Instruct 68.04 58.62 78.65 54.65 63.71 76.32 76.27 Meta-Llama-3-8B-Instruct 66.87 60.75 78.55 51.65 67.07 74.51 68.69 Mistral-7B-Instruct-v0.2 65.71 63.14 84.88 68.26 60.78 77.19 40.03 Bielik-4.5B-v3-Instruct 64.89 56.06 73.90 50.79 63.66 71.19 73.69 gemma-7b 64.29 61.09 82.47 44.91 66.03 78.45 52.77 Qwen1.5-32B-Chat 62.95 66.04 85.49 66.95 74.99 77.19 7.05 Qwen1.5-14B-Chat 62.27 58.70 82.27 60.36 68.57 73.09 30.63 Bielik-1.5B-v3-Instruct 56.64 48.38 65.03 42.47 54.59 65.35 62.85 Qwen1.5-7B-Chat 55.15 55.89 78.56 53.54 61.65 67.72 13.57 Mistral-7B-Instruct-v0.1 54.96 54.52 75.63 56.28 55.38 73.72 14.25 Bielik-7B-Instruct-v0.1 51.26 47.53 68.91 46.18 49.47 65.51 29.95

Table 16: Open LLM Leaderboard results for selected instruction-tuned models

- • Relevance detection: Determines whether the model appropriately triggers a function in cases where at least one relevant function should be used. Multiple valid function calls may exist; correctness of arguments is not strictly enforced—only that a relevant function is invoked
- • Irrelevance detection: Tests the model’s ability to refrain from invoking any functions when none are applicable. Models should either justify why no function is suitable or respond without initiating a function call

### 7 Limitations and Biases

Bielik v3 series of models can produce factually incorrect output, and should not be relied on to produce factually accurate data. Our models were trained on various public datasets. While great efforts have been taken to clear the training data, it is possible that this model can generate lewd, false, biased or otherwise offensive outputs.

### 8 Conclusion

In this technical report, we introduce the Bielik v3 series of generative text models for Polish language processing, including 1.5B and 4.5B parameter variants. These models represent a substantial advancement in Polish language AI, offering remarkable parameter efficiency while maintaining strong performance across diverse linguistic tasks, as demonstrated in our comprehensive evaluations in Tables 9, 10, 11, 12, and 13.

Key contributions of our work include:

###### Model MixEval-Hard MixEval

Qwen1.5-72B-Chat 48.3 84.1 LLaMA-3-8B-Instruct 45.6 75.0

- Bielik-11B-v2.1-Instruct 45.0 74.6 Qwen1.5-32B-Chat 43.3 81.0 Bielik-11B-v2.3-Instruct 43.2 73.0 Bielik-11B-v2.0-Instruct 40.2 72.1
- Bielik-11B-v2.2-Instruct 39.7 72.4 Mistral-7B-Instruct-v0.2 36.2 70.0 Bielik-4.5B-v3-Instruct 29.6 55.3 Bielik-1.5B-v3-Instruct 24.8 46.6

Table 17: MixEval benchmark results comparing Bielik v3 models against other instruction-tuned models

Model Non-Live Non-Live Non-Live Non-Live Live Live Live Live Parallel

Python Multiple Parallel Parallel Simple Multiple Parallel Multiple Simple AST AST AST Multiple AST AST AST AST AST

Open-Mistral-Nemo-2407 (Prompt) 92.00% 93.50% 89.50% 84.50% 77.91% 74.45% 87.50% 66.67% Gemma-3-12b-it (Prompt) 94.00% 95.00% 90.00% 73.00% 84.88% 70.85% 87.50% 62.50% Open-Mistral-Nemo-2407 (FC) 91.25% 93.50% 85.50% 85.00% 77.13% 69.61% 75.00% 70.83% Bielik-11B-v2.5-Instruct (FC) 95.00% 97.50% 87.50% 87.00% 77.13% 77.21% 43.75% 66.67% Bielik-4.5B-v3.0-Instruct (FC) 94.00% 92.50% 82.00% 86.00% 70.16% 68.66% 50.00% 54.17% Qwen2.5-3B-Instruct (Prompt) 91.50% 90.50% 79.50% 79.00% 69.77% 66.48% 56.25% 62.50% Qwen2.5-3B-Instruct (FC) 96.00% 92.00% 73.50% 76.50% 74.03% 72.08% 62.50% 45.83% Qwen2.5-1.5B-Instruct (FC) 92.25% 87.00% 81.50% 75.50% 74.03% 66.10% 50.00% 45.83% Qwen2.5-1.5B-Instruct (Prompt) 89.00% 86.00% 70.00% 66.50% 70.54% 59.26% 56.25% 41.67% Bielik-1.5B-v3.0-Instruct (FC) 77.00% 85.00% 69.50% 63.00% 61.63% 58.69% 50.00% 41.67% Bielik-11B-v2.3-Instruct (Prompt) 87.50% 93.50% 47.00% 50.00% 72.87% 69.71% 43.75% 54.17%

Table 18: Comprehensive summary of model performance on the Berkeley Function-Calling Leaderboard subtasks. Bielik models demonstrate strong results across a variety of subtasks, excelling especially in Non-Live Python Simple AST and Non-Live Multiple AST categories, while also maintaining consistent outcomes in Live Simple and Multiple AST tasks.

- 1. Innovative Architectural Decisions: Building upon the Qwen2.5 architecture, we implemented depth upscaling and replaced the tokenizer with our custom APT4 tokenizer optimized for Polish, resulting in more efficient token usage.
- 2. Data Quality Focus: We developed sophisticated quality classification systems with 95% accuracy to ensure our training corpus consisted of high-quality Polish texts balanced across 120 thematic categories.
- 3. Training Methodology Innovations: Our techniques include Adaptive Learning Rate, which significantly improved model performance, particularly for Polish-specific linguistic patterns.
- 4. Impressive Performance Efficiency: The 4.5B parameter model achieves results competitive with models 23× larger across multiple benchmarks, while the 1.5B model delivers strong performance despite its extremely compact profile. This efficiency makes Bielik v3 particularly valuable for deployment in resource-constrained environments while maintaining high-quality Polish language capabilities.
- 5. Benchmark Excellence: On the Open PL LLM Leaderboard, CPTUB, Polish Medical Benchmark, and EQBench, Bielik v3 models consistently outperform many larger models, demonstrating exceptional efficiency.

These models provide a powerful foundation for Polish language applications across various domains, from general conversational AI to specialized fields such as medicine and law. By prioritizing parameter efficiency without sacrificing quality, Bielik v3 enables broader deployment on resource-constrained systems while advancing the state of Polish language AI.

Future work will focus on further enhancing capabilities for complex reasoning, exploring additional efficiency improvements, and expanding domain-specific knowledge. We believe the Bielik v3 models establish a new benchmark for efficient, high-quality language models for less-resourced languages.

### Acknowledgements

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2024/017214 and PLG/2025/018338.

The model could not have been created without the commitment and work of the entire SpeakLeash team, whose contribution is invaluable. Thanks to the hard work of many individuals, it was possible to gather a large amount of content in Polish and establish collaboration between the open-science SpeakLeash project and the HPC center: ACK Cyfronet AGH. Individuals who contributed to the creation of the model through their commitment to the open-science SpeakLeash project: Sebastian Kondracki, Marek Magry´s, Szymon Mazurek, Mieszko Cholewa, Igor Ciuciura, Szymon Baczy´nski, Jacek Chwiła, Dominika Basaj, Kuba Sołtys, Karol Jezierski, Anna Przybył, Agnieszka Ratajska, Witold Wydma´nski, Izabela Babis, Nina Babis, and many other wonderful researchers and enthusiasts of the AI world.

### References

Voicelab. Trurl 2 models), 2023. URL https://huggingface.co/Voicelab. National Information Processing Institute and Gda´nsk University of Technology. Qra models, 2024. URL https:

//huggingface.co/OPI-PG.

Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoz´dziej, and Remigiusz Kinas. Bielik 7b v0.1: A polish language model – development, insights, and evaluation, 2024. URL https://arxiv.org/abs/2410.18565.

Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoz´dziej, and Remigiusz Kinas. Bielik 11b v2 technical report, 2025. URL https://arxiv.org/abs/2505.02410.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. URL https: //api.semanticscholar.org/CorpusID:13756489.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, Singapore, December 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.emnlp-main.298. URL https://aclanthology.org/2023.emnlp-main.298.

Yann Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International Conference on Machine Learning, 2016. URL https://api.semanticscholar.org/CorpusID: 16119010.

Noam Shazeer. Glu variants improve transformer, 2020. URL https://arxiv.org/abs/2002.05202.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. ISSN 09252312. doi:https://doi.org/10.1016/j.neucom.2023.127063. URL https://www.sciencedirect.com/science/ article/pii/S0925231223011864.

Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, and David Z. Pan. Pre-rmsnorm and pre-crmsnorm transformers: equivalent and efficient pre-ln transformers. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2024. Curran Associates Inc.

Sanghoon Kim, Dahyun Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. SOLAR 10.7B: Scaling large language models with simple yet effective depth up-scaling. In Yi Yang, Aida Davani, Avi Sil, and Anoop Kumar, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 23–35, Mexico City, Mexico, 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.naacl-industry.3.

Krzysztof Ociepa and Azurro Team. Introducing apt3-1b-base: Polish language model, 2024. URL https://azurro.

##### pl/apt3-1b-base-en. Accessed: 2024-09-30.

X. Yuan, Y. Li, and Y. Liu. Frequency-based vocabulary transfer for efficient tokenizer adaptation in multilingual pretrained models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track. Association for Computational Linguistics, 2022.

Benjamin Minixhofer, Fabian Paischer, and Navid Rekabsaz. Wechsel: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 2022.

Konstantin Dobler and Gerard de Melo. Focus: Effective embedding initialization for monolingual specialization of multilingual models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023). Association for Computational Linguistics, 2023.

Yihong Liu, Peiqin Lin, Mingyang Wang, and Hinrich Schütze. Ofa: A framework of initializing unseen subword embeddings for efficient large-scale multilingual continued pretraining. arXiv preprint arXiv:2311.08849, 2023.

Ke Tran. From english to foreign languages: Transferring pretrained language models. arXiv preprint arXiv:2002.07306, 2020.

SpeakLeash Team. Speakleash a.k.a spichlerz!, 2024. URL https://www.speakleash.org. Accessed: 2024-09-30. Dingcheng Li, Zheng Chen, Eunah Cho, Jie Hao, Xiaohu Liu, Xing Fan, and Chenlei Guo. Overcoming catastrophic

forgetting during domain adaptation of seq2seq language generation. In North American Chapter of the Association for Computational Linguistics, 2022. URL https://api.semanticscholar.org/CorpusID:249268165.

Oleksiy Ostapenko, Timothee Lesort, Pau Rodriguez, Md Rifat Arefin, Arthur Douillard, Irina Rish, and Laurent Charlin. Continual learning with foundation models: An empirical study of latent replay. In Sarath Chandar, Razvan Pascanu, and Doina Precup, editors, Proceedings of The 1st Conference on Lifelong Learning Agents, volume 199 of Proceedings of Machine Learning Research, pages 60–91. PMLR, 22–24 Aug 2022. URL https: //proceedings.mlr.press/v199/ostapenko22a.html.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. Simple and scalable strategies to continually pre-train large language models, 2024. URL https://arxiv.org/abs/2403.08763.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 6 2023. URL https: //huggingface.co/datasets/cerebras/SlimPajama-627B.

Inez Okulska, Daria Stetsenko, Anna Kołos, Agnieszka Karli´nska, Kinga Gła˛bi´nska, and Adam Nowakowski. Stylometrix: An open-source multilingual tool for representing stylometric vectors, 2023. URL https://arxiv.org/ abs/2309.12810.

Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, and Aldo Lipani. Instruction tuning with loss over instructions, 2024. URL https://arxiv.org/abs/2405.14394.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard (2023-2024). https://huggingface.co/spaces/ open-llm-leaderboard-old/open_llm_leaderboard, 2023a.

Krzysztof Wróbel, SpeakLeash Team, and Cyfronet Team. Open pl llm leaderboard. https://huggingface.co/

##### spaces/speakleash/open_pl_llm_leaderboard, 2024.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/12608602.

Jan Koco´n, Piotr Miłkowski, and Monika Za´sko-Zieli´nska. Multi-level sentiment analysis of PolEmo 2.0: Extended corpus of multi-domain consumer reviews. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 980–991, Hong Kong, China, November 2019. Association for Computational Linguistics. doi:10.18653/v1/K19-1092. URL https://www.aclweb.org/anthology/K19-1092.

Piotr Rybak, Robert Mroczkowski, Janusz Tracz, and Ireneusz Gawlik. KLEJ: Comprehensive benchmark for polish language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1191–1201, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb. org/anthology/2020.acl-main.111.

Sławomir Dadas, Michał Perełkiewicz, and Rafał Po´swiata. Evaluation of sentence representations in Polish. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 1674–1680, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology.org/ 2020.lrec-1.207.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.44.

Michał Marcinczuk, Marcin Ptak, Adam Radziszewski, and Maciej Piasecki. Open dataset for development of polish question answering systems. In Proceedings of the 6th Language & Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics, Wydawnictwo Poznanskie, Fundacja Uniwersytetu im. Adama Mickiewicza, 2013.

Sławomir Dadas. Training effective neural sentence encoders from automatically mined paraphrases. In 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 371–378, 2022. doi:10.1109/SMC53654.2022.9945218.

Maciej Ogrodniczuk and Mateusz Kope´c. The Polish Summaries Corpus. In Proceedings of the Ninth International Conference on Language Resources and Evaluation, LREC 2014, 2014.

Michal Ptaszynski, Agata Pieciukiewicz, Pawel Dybala, Pawel Skrzek, Kamil Soliwoda, Marcin Fortuna, Gniewosz Leliwa, and Michal Wroczynski. Expert-annotated dataset to study cyberbullying in polish language. Data, 9(1):1, 2023.

Piotr Rybak, Piotr Przybyła, and Maciej Ogrodniczuk. PolQA: Polish question answering dataset. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LRECCOLING 2024), pages 12846–12855, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.

##### org/2024.lrec-main.1125.

Ryszard Tuora, Aleksandra Zwierzchowska, Natalia Zawadzka-Paluektau, Cezary Klamra, and Łukasz Kobyli´nski. Poquad-the polish question answering dataset-description and analysis. In Proceedings of the 12th Knowledge Capture Conference 2023, pages 105–113, 2023.

Samuel J. Paech. Eq-bench: An emotional intelligence benchmark for large language models, 2024. URL https:

##### //arxiv.org/abs/2312.06281.

Jan Sowa, Magdalena Krawczyk, Natalia Nadolna, Anna Zieli´nska, Maria Filipkowska, Agnieszka Kosiak, Marta Kania, Krzysztof Wróbel, Remigiusz Kinas, Szymon Baczy´nski, SpeakLeash Team, and Cyfronet Team. Complex polish text understanding benchmark. https://huggingface.co/spaces/speakleash/cptu_bench, 2024. Jakub Pokrywka, Jeremi Kaczmarek, and Edward Gorzela´nczyk. Gpt-4 passes most of the 297 written polish board

certification examinations, 2024. URL https://arxiv.org/abs/2405.01589. Sławomir Dadas, Małgorzata Gr˛ebowiec, Michał Perełkiewicz, and Rafał Po´swiata. Evaluating polish linguistic and cultural competency in large language models, 2025. URL https://arxiv.org/abs/2503.00995.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https://huggingface.co/spaces/ open-llm-leaderboard-old/open_llm_leaderboard, 2023b.

Jinjie Ni, Fuzhao Xue, Xiang Yue, Yuntian Deng, Mahir Shah, Kabir Jain, Graham Neubig, and Yang You. Mixeval: Deriving wisdom of the crowd from LLM benchmark mixtures. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/ 2024/hash/b1f34d7b4a03a3d80be8e72eb430dd81-Abstract-Conference.html.

###### Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Berkeley function calling leaderboard. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_ calling_leaderboard.html, 2024.

