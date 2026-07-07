arXiv:2505.02410v2[cs.CL]8May2025

BIELIK 11B V2 TECHNICAL REPORT

THE BIELIK LLM TEAM

Krzysztof Ociepa1,4, Łukasz Flis1,2, Remigiusz Kinas1, Krzysztof Wróbel1,3,5, Adrian Gwoz´dziej1,2

1SpeakLeash, 2ACK Cyfronet AGH, 3Jagiellonian University, 4Azurro, 5Enelpol

# ABSTRACT

We present Bielik 11B v2, a state-of-the-art language model optimized for Polish text processing. Built on the Mistral 7B v0.2 architecture and scaled to 11B parameters using depth up-scaling, this model demonstrates exceptional performance across Polish language benchmarks while maintaining strong cross-lingual capabilities. We introduce two key technical innovations: Weighted Instruction Cross-Entropy Loss, which optimizes learning across diverse instruction types by assigning qualitybased weights to training examples, and Adaptive Learning Rate, which dynamically adjusts based on context length. Comprehensive evaluation across multiple benchmarks demonstrates that Bielik 11B v2 outperforms many larger models, including those with 2-6 times more parameters, and significantly surpasses other specialized Polish language models on tasks ranging from linguistic understanding to complex reasoning. The model’s parameter efficiency and extensive quantization options enable deployment across various hardware configurations, advancing Polish language AI capabilities and establishing new benchmarks for resource-efficient language modeling in less-represented languages.

# 1 Introduction

The rapid advancement in natural language processing (NLP) has led to the development of increasingly sophisticated language models that can understand and generate human-like text. These models have shown remarkable success in various linguistic tasks across multiple languages. However, the development of high-performing models for lessresourced languages remains a significant challenge due to the scarcity of large and diverse datasets and computational resources.

Several notable efforts have advanced Polish language modeling in recent years. TRURL 2, a collection of fine-tuned Llama 2 models with 7 billion and 13 billion parameters, was trained on approximately 1 million conversational samples. Qra models, comprising continuously pretrained architectures with 1, 7, and 13 billion parameters, leveraged 90 billion tokens of Polish data. More recently, PLLuM, developed by a consortium of Polish academic institutions, introduced models ranging from 8 billion to 70 billion parameters, created through continued pretraining of Llama and Mistral models on Polish corpora. While these initiatives have made important strides, many still face limitations in performance, versatility, or accessibility due to restrictive licensing or require significantly larger computational resources for comparable performance.

Building on our previous work with Bielik 7B v0.1, we introduce Bielik 11B v2, a state-of-the-art language model optimized specifically for Polish text processing. Developed as a collaborative effort between the SpeakLeash openscience project and ACK Cyfronet AGH, this model represents a significant advancement in both scale and capabilities. Bielik 11B v2 is built on the Mistral 7B v0.2 architecture and scaled to 11 billion parameters using depth up-scaling, striking an optimal balance between performance and computational efficiency.

Our approach introduces two key technical innovations: Weighted Instruction Cross-Entropy Loss, which optimizes learning across diverse instruction types by assigning quality-based weights to training examples, and Adaptive Learning Rate, which dynamically adjusts based on context length. These techniques, combined with comprehensive training on a diverse corpus of 198 billion tokens, enable Bielik 11B v2 to outperform many larger models, including those with 2-6 times more parameters, across a variety of benchmarks such as the Open PL LLM Leaderboard, Polish MT-Bench, and Polish Linguistic and Cultural Competency Benchmark (PLCC).

In the following sections, we detail the architecture of Bielik 11B v2, describe our dataset preparation methodology, discuss the pre-training and post-training processes, and evaluate the model’s performance across multiple benchmarks. We also present various quantization options that enable deployment across different hardware configurations. Our results demonstrate that Bielik 11B v2 not only advances the state of Polish language understanding but also establishes new benchmarks for parameter-efficient language modeling in less-represented languages.

# 2 Architecture

Parameter Value Layers 50 Model Dimension 4096 Attention Heads 32 Key/Value Heads 8 Head Size 128 Intermediate Size 14336 Activation Function SwiGLU Vocabulary Size 32128 Positional Embeddings RoPE (θ = 1000000) Context Length 32768

Table 1: Model architecture.

The Bielik 11B v2 model is based on the Transformer architecture Vaswani et al. [2017], with key parameters listed in Table 1. The design integrates several advanced techniques to enhance performance and efficiency.

Self-attention with causal masks Vaswani et al. [2017] enables the model to assign varying importance to different parts of the input sequence. The causal mask ensures that the model only attends to preceding tokens, preserving the autoregressive property essential for language modeling.

Grouped-query attention (GQA) Ainslie et al. [2023] reduces both computational complexity and memory usage while maintaining model quality. It achieves this by using fewer key-value heads than query heads, enabling more efficient handling of long sequences.

SwiGLU activation function Dauphin et al. [2017], Shazeer [2020] combines the Swish activation function with Gated Linear Units (GLU), providing better performance and trainability than traditional activation functions such as ReLU.

Rotary Positional Embeddings (RoPE) Su et al. [2024] enhance the model’s ability to capture relative token positions. Compared to absolute positional embeddings, RoPE supports better generalization to longer sequences and improves performance in tasks requiring positional sensitivity.

Root Mean Square Layer Normalization (RMSNorm) Jiang et al. [2023a] normalizes activations within the network, offering greater training stability and slightly faster computation compared to standard Layer Normalization.

Pre-normalization involves applying layer normalization before the self-attention and feed-forward layers. This improves model convergence and overall performance.

The Bielik 11B v2 model is adapted from the Mistral 7B v0.2 model Jiang et al. [2023b], scaled using the Depth Up-Scaling method Kim et al. [2024], and further pretrained. Beginning with the original 32-layer model (n = 32), we duplicated its layers and removed the final 8 and initial 8 layers at the junction (m = 7), resulting in a 50-layer model (s = 50), as shown in Figure 1. We chose to upscale to 50 layers in order to reach a model size of approximately 11B parameters, which can still be run comfortably on consumer-grade GPUs with up to 24GB of VRAM.

This decision to build on an existing model, rather than developing one from scratch, was driven by the desire to allocate resources efficiently. By focusing on the linguistic adaptation of an already high-performing model, we were able to optimize both time and computational resources. The Mistral 7B v0.2 model was selected due to its strong benchmark performance and permissive Apache 2.0 license.

We retained the original tokenizer from the Mistral 7B model, which includes a vocabulary of 32,000 tokens. The only modification was the addition of special tokens required for the chat template, bringing the total vocabulary size to 32,128 tokens.

[Figure 1]

Figure 1: Depth up-scaling with n = 32, m = 7, and s = 50.

# 3 Pre-training

The primary objective of the pre-training phase was to enhance the model’s proficiency in the Polish language, with an emphasis on both accuracy and fluency. To achieve this, we utilized a diverse collection of high-quality Polish texts. These materials underwent rigorous preprocessing and thorough quality evaluation to ensure the highest standards of training data.

## 3.1 Pre-training Data

The pre-training of the Bielik v2 model involved constructing a novel, diverse, and high-quality dataset composed primarily of Polish-language texts. We leveraged resources from the SpeakLeash project SpeakLeash Team [2024]. Using metadata associated with each document—including topical information and various stylometric features—we selected 41 million documents from different datasets, ensuring both high quality and thematic diversity. These selected texts underwent comprehensive cleaning and quality evaluation, as described in Sections 3.1.1 and 3.1.2.

Additionally, we excluded documents where scraping was technically permitted (i.e., not blocked by robots.txt) but where the terms and conditions explicitly prohibited use for training language models. Only documents meeting our stringent quality standards were retained and subsequently tokenized. This meticulous curation resulted in a Polish training corpus of 90 billion tokens.

To improve the model’s adaptation to Polish while mitigating catastrophic forgetting Li et al. [2022], Ostapenko et al.

- [2022], Ibrahim et al. [2024], we supplemented the dataset with English texts from the SlimPajama dataset Soboleva et al. [2023], known for its diversity and quality.

To better align the model with the objectives of later training phases, we also incorporated the instruction dataset described in Section 4.1 into the pre-training data. Although originally constructed for supervised fine-tuning (SFT), this dataset facilitated a smoother and more effective transition to subsequent stages of training.

In total, the final training dataset comprised 198 billion tokens (96 million documents).

## 3.1.1 Data Cleanup

To enhance the quality of the documents, we applied a series of heuristics designed to remove corrupted or irrelevant content, anonymize personal data (including physical addresses, email addresses, phone numbers, and URLs), and resolve encoding or formatting issues. These steps produced cleaner, higher-quality texts that were subsequently subjected to further evaluation.

## 3.1.2 Quality Evaluation

To develop the training dataset for text quality evaluation, we manually selected and annotated documents, categorizing them into three quality classes: HIGH, MEDIUM, and LOW. The HIGH class represents superior-quality documents; LOW denotes poor-quality texts, and MEDIUM encompasses documents whose quality is ambiguous, falling between high and low standards. This nuanced classification approach addresses the inherent complexities of assessing textual quality.

For the Bielik v2 model, the annotated dataset was significantly expanded, comprising 20 000 training samples, 1500 test samples, and 500 validation samples. Each document was represented by a vector of 150 carefully selected stylometric features. Beyond standard linguistic and structural metrics (such as frequencies of verbs, nouns, sentences, and punctuation marks), the updated feature set placed special emphasis on characteristics relevant to Markdown-formatted texts.

Newly introduced Markdown-focused features include: counts of ordered list items, hyperlinks, images, inline code fragments, code blocks, blockquotes, horizontal rules, and special Markdown characters. In addition, detailed analysis of table formatting—such as pipe character frequency and header-to-body alignment—was incorporated.

This comprehensive feature set was developed following the methodology inspired by the StyloMetrix tool Okulska et al.

- [2023], and enhanced specifically for Markdown textual analysis. Using this enriched stylometric and Markdown-aware representation, we evaluated multiple machine learning algorithms, with the XGBoost classifier emerging as the most effective for distinguishing between the defined quality categories. A ranked list of the most influential features, based on mean absolute SHAP values, is presented in Table 5.

The performance of the Bielik v2 model was rigorously evaluated on held-out validation and test sets. On the validation set, the model achieved an overall accuracy of 86% and a macro-average F1-score of 0.79, indicating robust performance across all three quality levels. On the test dataset, the classifier reached an overall accuracy of 94% and a macro-average F1-score of 0.85, underscoring its strong ability to differentiate particularly between HIGH- and LOW-quality documents.

To determine a reliable threshold for identifying high-quality texts suitable for inclusion in downstream training corpora, we conducted a manual analysis of 1000 documents. This evaluation confirmed that a predicted probability above 90% for the HIGH-quality class serves as a practical cutoff. Documents below this threshold were systematically excluded from the final training dataset for the Bielik v2 model.

Alternative high-capacity model. The XGB_HighN_LowLR_d7_n1000_lr002 configuration corresponds to an XGBoostClassifier trained with n_estimators = 1000, max_depth = 7, learning_rate = 0.02, subsample = 0.7, and colsample_bytree = 0.7. As with the best model, it uses the multi:softprob objective, with eval_metric = mlogloss, use_label_encoder = False, and a fixed random_state = 42.

## Model Val F1 (macro) Val F1 (weighted) Val Accuracy

XGB_HighN_LowLR_d7_n1000_lr002 0.789482 0.928575 0.938 XGB_AggressiveSubsample_d8_n300_lr007 0.796013 0.928040 0.934 LightGBM 0.798175 0.927704 0.934 XGB_RegL1L2_d8_n250_lr008_a02_l05 0.793790 0.926026 0.934 CatBoost 0.788047 0.924801 0.932 HistGradientBoosting 0.788035 0.924784 0.932 XGBoost_nEstimators400_maxDepth3_lr025 0.791143 0.924473 0.930 XGBoost 0.763592 0.917313 0.926 MLP_hidden100_relu_adam 0.768963 0.913112 0.918 MLP_hidden50_50_tanh_sgd 0.741196 0.905938 0.914 EBM 0.725163 0.904523 0.916 ExtraTrees_nEstimators200_maxDepthNone 0.730563 0.901927 0.920 RandomForest_nEstimators200_maxDepthNone 0.729183 0.899997 0.918 MLP_hidden200_tanh_adam_alpha1e-3 0.746473 0.898842 0.900 RandomForest_nEstimators300_maxDepth20 0.727778 0.898033 0.916 RandomForest_nEstimators150_maxDepth15 0.724613 0.896521 0.914 MLP_hidden50_25_10_logistic_adam 0.741392 0.895995 0.896 MLP_hidden100_50_relu_lbfgs 0.739666 0.895068 0.898 RandomForest_nEstimators100_maxDepth10 0.669666 0.896601 0.908 KNN_neighbors7_weightsUniform_norm1 0.750744 0.884859 0.902 KNN_neighbors5_weightsDistance_norm1 0.732194 0.880066 0.884

Table 2: Comparison of models performance

Using these features as input, we trained an XGBoost classifier. This machine learning model effectively captured complex patterns in the data, achieving strong results on both validation and test sets.

Detailed performance metrics for the validation set are presented in Table 3, while results for the test set are summarized in Table 4.

[Figure 2]

Figure 2: Confusion matrix showing test and validation results for the XGBoost classifier.

Class Precision Recall F1-score Support LOW 0.95 0.98 0.97 233 MEDIUM 0.69 0.32 0.44 34 HIGH 0.94 0.99 0.96 233 Accuracy 0.94 500 Macro avg 0.86 0.76 0.79 500

- Weighted avg 0.93 0.94 0.93 500 Table 3: Classification Report (Validation) – Best Model

Class Precision Recall F1-score Support

LOW 0.94 0.98 0.96 682 MEDIUM 0.84 0.48 0.61 120 HIGH 0.96 0.98 0.97 698

Accuracy 0.94 1500 Macro avg 0.91 0.82 0.85 1500

- Weighted avg 0.94 0.94 0.94 1500 Table 4: Classification Report (Test) – Best Model

To determine an appropriate threshold for selecting high-quality documents, we conducted a manual analysis of 1,000 samples. Based on this evaluation, we set the threshold for the HIGH category at a predicted probability exceeding 90%. Documents falling below this threshold were excluded from the Bielik model’s final training dataset.

## 3.1.3 Category Classification: Results and Applications

This section discusses the performance and practical implications of the text category classifier, designed to automatically label Polish-language documents with one of 120 predefined thematic categories. The classifier plays a crucial role not only in organizing data for training base models but also in generating balanced instruction sets covering diverse domains.

## Feature Mean Abs SHAP

oovs 0.5637 non_alpha_word_fractions 0.3562 stop_word_ratio 0.3155 average_lines 0.2147 colons_per_sentence 0.1583 short_line_ratio_20 0.1578 special_chars_ratio_md 0.1570 commas_per_sentenced 0.1021 lowercase_ratio_md 0.1003 overall_uppercase_ratio 0.1002 avg_paragraph_length 0.0984 avg_word_length 0.0955 noun_ratio 0.0906 char_ratio__ 0.0906 adj_ratio 0.0904 uppercase_ratio_md 0.0838 ratio_of_bulletpoints 0.0822 char_ratio_> 0.0816 punct_frequency 0.0804 duplicate_line_ratio 0.0788 blank_lines 0.0771 emoticons 0.0754 single_char_ratio 0.0711 entropy 0.0710 blank_lines_ratio 0.0705

Table 5: Top Features by Mean Absolute SHAP Value

## Dataset and Setup

The classification model was trained on a comprehensive dataset of 35,944 Polish-language documents. A standard 90/10 train-test split was applied, resulting in 32,349 samples for training and 3,595 for evaluation.

To ensure the representativeness of the full label set, stratified sampling was employed, preserving the category distribution across the two subsets. This approach is especially important for tasks involving a large number of categories, some of which may be infrequent or closely related.

## Modeling Approach

The classification pipeline starts by converting textual data into numeric vectors using a CountVectorizer, restricted to the 10,000 most informative and frequently occurring features. This dimensionality has proven sufficient for effectively separating the 120 categories.

The term counts were then transformed into TF-IDF representations via the TfidfTransformer, highlighting the relative importance of terms across documents.

Classification was performed using a Linear Support Vector Classifier (LinearSVC), known for its efficiency and robustness on high-dimensional data. To produce calibrated probability estimates, the classifier was wrapped with CalibratedClassifierCV using isotonic regression and 3-fold cross-validation.

## Performance Evaluation

The trained model achieved an overall accuracy of 94.44% on the held-out test set of 3,595 documents, demonstrating strong generalization across the wide thematic range.

These consistently high scores indicate that the classifier performs reliably across both high- and low-frequency classes. Importantly, most misclassifications tend to occur among semantically related categories (e.g., Transportation, Automotive, and Travel), which are not problematic in practical downstream use.

## Category Distribution

The underlying dataset covers a broad range of topics, with category distribution reflecting real-world domain diversity. As shown in Figure 3, the most prominent categories include Other (23.3%), Law (8.8%), and Politics, Media & News (8.8%), followed by domains such as Finance, Health, and Education.

While the long-tail nature of the distribution is evident, the dataset remains well-balanced overall. Categories with lower representation (e.g., below 1.5%) were excluded from the analysis in the figure to improve readability, though they were retained in the training process. This distribution ensures the classifier is exposed to sufficient variability, while preserving robustness even across less frequent or closely related classes.

Metric Precision Recall F1-score Support

Macro avg 0.9451 0.9444 0.9436 3595 Weighted avg 0.9447 0.9441 0.9426 3595

Table 6: Category Classification Performance on Test Set (N=3,595)

## Application and Future Directions

This classifier enables the construction of thematically diverse and well-balanced datasets in the Polish language, which are essential for both model pretraining and the generation of high-quality, targeted instruction sets.

Its ability to maintain category balance while minimizing critical misclassifications allows it to serve as a robust preprocessing step for synthetic instruction generation. These instructions benefit from broad topical coverage and linguistic quality, strengthening downstream models’ performance in instruction-following tasks across various domains.

## 3.2 Training Hyperparameters

We employed the AdamW optimizer Loshchilov and Hutter [2019] with hyperparameters β1 = 0.9, β2 = 0.95, and a weight decay of 0.1. The learning rate followed a cosine decay schedule, starting at 2 × 10−5 and decreasing to 9 × 10−6, with a warm-up period of 50 iterations. Training was conducted over 97,250 iterations.

We utilized Megatron-LM with a global batch size of 128 and a tensor parallelism degree of 4. The gradient clipping norm was set to 1.0, and mixed-precision training was enabled using bfloat16. The model was trained on 198 billion tokens over two epochs, with a maximum context length of 32,768 tokens.

## 3.3 Training process monitoring

To ensure the expected quality of the pre-training process, we carefully monitored the progression of model performance on the OpenLLM PL and OpenLLM EN benchmarks. Throughout training, model checkpoints were saved at regular intervals and benchmarked accordingly. In our experience, degradation in benchmark performance over time typically indicates dataset-related issues and is rarely reversible.

Performance evaluation based on the benchmarks mentioned above was conducted in parallel with the training process on a separate set of compute nodes to avoid interference. This approach enabled early detection of training issues without having to wait for an epoch to complete, resulting in significant compute time savings.

# 4 Post-training

After completing the pre-training phase, we proceeded to the post-training phase, which aimed to enhance the model’s performance in various domains, including coding, mathematics, logical reasoning, and instruction following.

## 4.1 Post-training Data

Due to the lack of a sufficiently large and open dataset of instructions and dialogues in the Polish language, we began constructing our own dataset, which is continuously expanded and refined by human annotators. This dataset is manually curated through the creation of instructions and dialogues, ensuring high-quality, relevant content specifically tailored to the Polish language.

[Figure 3]

Figure 3: Distribution of major thematic categories in the Polish text dataset (≥ 1.7%)

To complement the manually annotated data, we also generated additional instructions and dialogues using Mixtral 8x22B Jiang et al. [2024]. The resulting dataset used for training included over 20 million instructions, totaling more than 10 billion tokens.

The distribution of instructions and dialogues across categories is presented in Figure 4, and the weighted version, based on the weights described in Section 4.2.3, is shown in Figure 5.

For the final phase of post-training with reinforcement learning using Direct Preference Optimization (DPO) Rafailov et al. [2023a] and (DPO-P) Pal et al. [2024], we constructed a dedicated dataset consisting of 72,000 preference instructions. To support alignment training, we developed a comprehensive instruction data generation pipeline designed to ensure high-quality, diverse, and reliable datasets in Polish. The pipeline combines manual efforts with automated processes, incorporating techniques to maximize task variety and rigorously control data quality throughout each stage.

- • Step 1: Instruction Creation. The instruction creation phase involved two approaches. First, domain experts manually authored instructions across a wide range of topics and complexity levels. Second, a perturbationbased augmentation strategy was applied: existing English-language instruction datasets were translated and

[Figure 4]

Figure 4: Distribution of instruction categories in the SFT dataset.

[Figure 5]

Figure 5: Weighted distribution of instruction categories in the SFT dataset.

perturbed into Polish using language models. This allowed us to expand the instruction pool by introducing meaningful linguistic and semantic variations while preserving the core task intent.

- • Step 2: Instruction Deduplication. To increase diversity and minimize redundancy, all instructions were subjected to a deduplication process. We employed a combination of Locality-Sensitive Hashing (LSH), cosine similarity calculations, and MiniHash fingerprinting techniques. Depending on the configuration selected by domain experts, these methods were applied individually or in combination, ensuring that similar or duplicate instructions were effectively removed from the dataset.
- • Step 3: Generation of Chosen and Rejected Response Pairs. Following instruction preparation, we generated pairs of preferred (chosen) and less preferred (rejected) responses. This was achieved through two main methods: manual labeling by human annotators and synthetic generation using large language models, including Mistral, Bielik v0.1, and LLaMA-based models. Manual creation allowed for fine-grained quality distinctions, while automated generation enabled efficient scaling of the dataset.
- • Step 4: Quality Evaluation with Metamodels. Each response pair was evaluated using reward metamodels based on the LLaMA 70B and Mixtral8x22B. The metamodels assigned quality scores on a scale from 0 to 10, considering attributes such as factual accuracy, completeness, clarity, and depth of reasoning. This evaluation provided an essential quantitative assessment of response quality for subsequent filtering.
- • Step 5: Filtering. The dataset was filtered according to strict quality control rules. Pairs in which the chosen and rejected responses received identical scores were discarded. Likewise, pairs with a score distance of less than two points were removed to avoid ambiguous training examples. Additionally, if a rejected response achieved a higher score than the chosen response, the labels were swapped to maintain logical consistency.
- • Step 6: Dataset Cleaning. After filtering, the dataset underwent a cleaning phase to ensure format consistency and correctness. Samples were reformatted into a unified structure, and entries containing API errors, malformed outputs, or serious formatting issues were removed. This stage ensured the structural and semantic integrity of the dataset.
- • Step 7: Manual Inspection and Correction. Finally, the cleaned dataset was subjected to a full manual review. Annotators validated the quality of instruction-response pairs, corrected mislabeled examples, and adjusted evaluation scores where necessary. This manual inspection phase was crucial for maximizing dataset reliability and ensuring its suitability for downstream RLHF training.

It is important to note that our alignment objective was not centered on safety-related aspects of model responses. Instead, our focus was on tailoring the model to produce outputs in a preferred style, encompassing appropriate dialogue formatting, mathematical expressions, storytelling, bullet points, headings, and other stylistic elements. To achieve this, we endeavored to create a highly diversified set of tasks and, subsequently, to align the selected responses to a consistent stylistic standard.

Furthermore, the range of tasks included in the instruction dataset is continuously expanding. New tasks are generated based on the ongoing observation of user interactions with the Bielik Chat system. By analyzing real-world conversations, we identify gaps, emerging user needs, and opportunities for stylistic or functional improvements, which in turn inform the creation of additional instructions and response patterns. This iterative approach ensures that the dataset evolves dynamically, staying aligned with user expectations and practical usage scenarios.

## 4.2 Supervised Fine-Tuning

Previous research has shown that low-quality instructions negatively impact a model’s benchmark performance, with findings indicating that poorly constructed instructions degrade model capabilities Zhou et al. [2023]. These studies demonstrated that smaller, high-quality instruction datasets often outperform larger, noisier ones. To mitigate this issue, we implemented several improvements, as summarized below, while continuing to use the aforementioned datasets.

## 4.2.1 Masked Tokens

We adopted a masked token strategy, applying the loss function selectively to specific parts of the output. In particular, we masked the loss for user instructions and control tokens Shi et al. [2024]. This method ensures that these tokens do not contribute to the total loss during training, allowing the model to concentrate on learning from the actual content tokens.

## 4.2.2 Adaptive Learning Rate

Instruction lengths can vary significantly, causing fluctuations in the number of tokens used to compute the loss. To ensure that each instruction had a consistent influence during training, we utilized an adaptive learning rate (ALR)

Ociepa et al. [2024], where the learning rate (LR) is scaled based on the square root of the ratio between the number of tokens in the current batch (T) and a baseline batch size (BS):

T BS

ALR = LR ·

(1)

## 4.2.3 Weighted Instruction Cross-Entropy Loss

To fine-tune our language model on a dataset containing instruction-response pairs of varying quality, we applied the Weighted Instruction Cross-Entropy Loss (WICEL) Ociepa et al. [2024]. This technique assigns a quality-based weight wi ∈ (0,1] to each training example, enabling the model to prioritize higher-quality data during optimization. The weighted loss is calculated as:

L = −wi ·

T

log πθ(yt|x,y<t) (2)

t=1

where (x,y) denotes the instruction-response pair, T is the length of the response, and πθ is the fine-tuned model. This approach increases training robustness and efficiency by incorporating lower-quality samples without allowing them to dominate the learning process.

## 4.3 Supervised Fine-Tuning Hyperparameters

We used the AdamW optimizer with hyperparameters β1 = 0.9, β2 = 0.95, and a weight decay of 0.05. The learning rate followed a cosine decay schedule, starting at 7 × 10−6 and decaying to 6 × 10−7, with 50 warmup iterations.

Our configuration used a global batch size of 128, composed of local batches of size 1. Gradient clipping was applied with a threshold of 1.0, and training was performed using mixed precision with the bfloat16 format.

We employed a sample packing technique, which concatenates multiple samples from the dataset into a single sequence, up to the maximum sequence length. The model was trained for 3 epochs with a maximum context length of 8,192 tokens.

## 4.4 Reinforcement Learning

To align the model with user preferences, we experimented with several techniques: DPO Rafailov et al. [2023b], PPOSchulman et al. [2017], KTO Ethayarajh et al. [2024], ORPO Hong et al. [2024] and SiMPO Meng et al. [2024]. As part of the project, we also developed a Polish reward model based on preference data. However, training with Proximal Policy Optimization (PPO) using this reward model did not yield performance improvements over the DPO approach. Ultimately, the DPO-Positive (DPO-P) method Pal et al. [2024] was adopted.

Direct Preference Optimization (DPO) is a technique that aligns language models with human preferences by directly optimizing a loss function based on pairwise comparisons between preferred (chosen) and less preferred responses (rejected). However, a notable limitation of DPO is its potential to inadvertently decrease the absolute probability of preferred responses, especially when the differences between response pairs are minimal. This phenomenon arises because DPO focuses solely on the relative preference between responses, without explicitly encouraging the model to maintain or increase the likelihood of preferred outputs.

πθ(yw | x) πref(yw | x) − log

πθ(yl | x) πref(yl | x)

(3)

LDPO(πθ;πref) = −E(x,y

w,yl)∼D log σ β log

To address this issue, DPO-Positive (DPO-P) introduces an additional term to the loss function that penalizes reductions in the probability of preferred responses relative to a reference model. This modification ensures that the model not only favors preferred responses over less preferred ones but also maintains or enhances the generation probability of these preferred responses. Empirical studies have demonstrated that DPO-P outperforms standard DPO, particularly in scenarios where response pairs have small edit distances, by mitigating the degradation of preferred response probabilities and improving overall model alignment with human preferences.

πref(yw | x) πθ(yw | x)

πθ(yw | x) πref(yw | x) − log

πθ(yl | x) πref(yl | x) − λ · max 0,log

LDPOP(πθ;πref) = −E(x,y

w,yl)∼D log σ β log

(4)

This approach leveraged both generated and manually corrected examples, which were scored by a metamodel. A dataset of over 72,000 examples, varying in length to capture different aspects of response style, was used. This dataset was filtered and assessed by the reward model to select instructions with an appropriate level of distinction between preferred and rejected responses. A key innovation introduced in DPO-P was the incorporation of multi-turn conversations.

## 4.4.1 DPO-Positive Hyperparameters

During DPO-Positive (DPO-P) training, the loss function was parameterized with β = 0.05 and λ = 2.5, following the recommendations for stabilizing preference optimization and ensuring retention of high-quality preferred responses.

We used the AdamW optimizer with hyperparameters β1 = 0.9, β2 = 0.95, and no weight decay. The learning rate was constant and set to 7 × 10−7, with 50 warmup iterations. Training was conducted over a total of 3,800 iterations.

Our configuration used a global batch size of 64, composed of local batches of size 1. Gradient clipping was applied with a threshold of 1.0, and training was performed using mixed precision with the bfloat16 format.

## 4.5 Model Merging

To leverage the diverse capabilities of models fine-tuned under varying conditions, we developed our framework for merging models and tested multiple strategies, including Linear (Model Soups) Wortsman et al. [2022], TIES (TIES-Merging: Resolving Interference When Merging Models) Yadav et al. [2023], Model Stock Jang et al. [2024]. Each of these methods offers distinct approaches to combining model parameters, aiming to integrate strengths from different models while mitigating potential conflicts or redundancies.

We systematically evaluated these merging techniques on a suite of Polish language benchmarks. Among the methods tested, the Linear merging approach, assigning equal weights (1.0) to each model, consistently yielded the most favorable results. This method involved merging three models obtained from different phases of the same DPO-Positive (DPO-P) training process, effectively capturing varied response characteristics developed during training.

To further explore the potential of model merging, we conducted an experiment by introducing the original base model into the mixture of the three DPO-P fine-tuned models. This addition aimed to assess whether incorporating the foundational model could enhance the merged model’s performance or stability.

Beyond post-alignment merging, we also applied merging techniques during earlier stages of model development, specifically between different Supervised Fine-Tuning (SFT) runs. This strategy aimed to consolidate improvements from various SFT iterations, thereby enhancing the base model’s quality prior to alignment through reinforcement learning from human feedback (RLHF).

## 4.6 Layer-Wise Model Quality Monitoring

To ensure the robustness and generalization capabilities of our models throughout the training process, we employed WeightWatcher Martin and Mahoney [2021], an open-source diagnostic tool designed to analyze deep neural networks without relying on training or test data. WeightWatcher examines the spectral properties of weight matrices in each layer, leveraging the theory of Heavy-Tailed Self-Regularization (HTSR) to assess layer quality.

[Figure 6]

Figure 6: Alpha Histogram, Correlation Flow and Scale Flow charts for Bielik-11B models.

A key metric provided by WeightWatcher is the power-law exponent α, which quantifies the degree of heavy-tailedness in the eigenvalue distribution of a layer’s weight matrix. Empirical studies suggest that well-trained layers typically exhibit α values within the range of 2 to 6, with lower values indicating better generalization properties.

During our experiments, we applied WeightWatcher to each model checkpoint obtained from different phases of the DPO-Positive (DPO-P) training process. This analysis enabled us to identify layers that were potentially over- or under-trained, providing insights into the internal dynamics of the model’s learning process. By monitoring the evolution of α values across layers and training stages, we could make informed decisions regarding training adjustments and model selection, ultimately contributing to the development of models with improved performance and stability.

# 5 Quantization

In order to support a wider range of hardware configurations and deployment scenarios, we provide several quantized variants of our instruction-tuned models. Quantization significantly reduces model size and inference latency, making it possible to run models on resource-constrained devices, albeit with a potential trade-off in output quality.

We have prepared the following quantized versions of our models:

- • GGUF
- • GPTQ
- • HQQ
- • AWQ
- • EXL2
- • MLX
- • Quanto
- • FP8 (compatible with vLLM and SGLang; optimized for Ada Lovelace and Hopper architectures)
- • INT8 W8A8 (vLLM INT8 quantization with 8-bit weights and 8-bit activations)

While quantized models enable faster and more efficient inference, users should be aware that they may exhibit a decrease in response quality compared to the full-precision versions.

In particular, the GGUF variants have been further optimized through a two-step process designed to preserve model quality after quantization:

- • To minimize the degradation in output quality, we applied a post-training calibration procedure based on an importance matrix. The importance matrix assigns weights to model activations according to their contribution to the model’s output, as estimated on a specially constructed bilingual Polish-English calibration dataset. This dataset was curated to reflect the target linguistic and task distributions. During calibration, layers or parameters deemed more important are preserved more precisely during quantization, leading to improved output consistency and better preservation of model capabilities, especially in multilingual settings.
- • Although the main model weights were quantized to low-bit formats (e.g., INT4), the embedding layers and the final output projection layer were retained in FP16 precision. This selective retention helps maintain the expressiveness of the input and output spaces, mitigating quality loss associated with aggressive quantization.

This calibration and quantization strategy improves the performance of GGUF models, making them suitable for deployment on resource-constrained devices while minimizing the loss in response quality.

# 6 Evaluation

We conducted a comprehensive evaluation of the Bielik 11B v2 models across multiple benchmarks to assess their performance in various linguistic tasks and languages. This section presents our evaluation results, comparing Bielik models against three key reference groups:

- 1. Leading models across all parameter sizes, including Qwen [Yang et al., 2024, Team, 2024a], Llama [Grattafiori et al., 2024], Mistral [Jiang et al., 2023b], Aya 23 [Aryabumi et al., 2024], and Gemma [Team, 2024b, 2025]
- 2. Models with similar parameter counts to Bielik, such as EuroLLM [Martins et al., 2024], Teuken [Ali et al., 2024], Salamandra [Gonzalez-Agirre et al., 2025], and Occiglot [Avramidis et al., 2024]

- 3. Other recent Polish language models including PLLuM [Consortium, 2025] and Qra

For clarity, the tables presented throughout this section contain selected subsets of models from each leaderboard, chosen to provide meaningful comparisons. Complete and up-to-date results can be found on the respective leaderboards’ online platforms.

The models were evaluated on the following benchmarks:

- • Open PL LLM Leaderboard (Polish)
- • Polish MT-Bench (Polish)
- • Polish EQ-Bench (Polish)
- • CPTUB Leaderboard (Polish)
- • Polish Medical Leaderboard (Polish)
- • Polish Linguistic and Cultural Competency Benchmark (PLCC) (Polish)
- • LLMzSzŁ (LLMs Behind the School Desk) (Polish)
- • European LLM Leaderboard
- • EuroEval
- • Open LLM Leaderboard
- • Open LLM Leaderboard v2
- • MixEval
- • Berkeley Function-Calling Leaderboard
- • FLORES200 Translation Benchmark (see Appendix 26)
- • BenCzechMark (see Appendix B)
- • Portuguese Benchmark (Open PT LLM Leaderboard) (see Appendix C)

It is important to note that Bielik 11B v2.5-Instruct is being released concurrently with this article. As such, comprehensive benchmark results for this latest version are not yet available across all evaluation frameworks mentioned above. The results presented in this section include data for Bielik 11B v2.5-Instruct where available, while predominately showcasing performance metrics for earlier v2 family models. Complete benchmark results for Bielik 11B v2.5-Instruct will be updated as evaluations are completed.

## 6.1 Open PL LLM Leaderboard

The Open PL LLM Leaderboard, based on the Open LLM Leaderboard v1 [Beeching et al., 2023a], evaluates models on various NLP tasks, including: sentiment analysis, categorization, short answer question answering, and text classification, but does not test their conversational capabilities [Wróbel et al., 2024, Ociepa et al., 2024]. The leaderboard utilizes the lm-evaluation-harness framework for model evaluation [Gao et al., 2024].

## Tasks:

- • polemo2: Sentiment analysis of online consumer reviews across four domains (medicine, hotels, products, university) with four-class labeling (positive, negative, neutral, ambiguous) [Koco´n et al., 2019]; metric: accuracy.
- • klej-ner: Named entity recognition in sentences containing single-type entities, classifying into six categories (no entity, place, person, organization, time, geographical name) [Rybak et al., 2020]; metric: accuracy.
- • 8tags: Topic classification of social media headlines into eight categories (film, history, food, medicine, motorization, work, sport, technology) [Dadas et al., 2020]; metric: accuracy.
- • belebele: Machine reading comprehension for question answering [Bandarkar et al., 2024]; metric: accuracy.
- • dyk: Question answering based on human-annotated pairs from Wikipedia’s "Did You Know" section [Marcinczuk et al., 2013]; metric: binary F1.
- • ppc: Text similarity assessment using manually labeled sentence pairs (exact paraphrases, close paraphrases, non-paraphrases) [Dadas, 2022]; metric: accuracy.

Qwen2.5-72B 72.7 67.38 Qwen2.5-32B 32.8 66.73 Qwen-72B 72.7 66.02 Qwen2.5-14B 14.8 62.71 Meta-Llama-3-70B 70.6 62.07 Qwen1.5-72B 72.7 61.11 Meta-Llama-3.1-70B 70.6 60.87 Mixtral-8x22B-v0.1 141.0 60.75 Mistral-Small-24B-Base-2501 24.0 59.90

- Qwen1.5-32B 32.8 58.71 Bielik-11B-v2 11.2 58.14
- Qwen2.5-7B 7.0 53.35 EuroLLM-9B 9.2 50.03 Qwen-7B 7.0 49.39 SOLAR-10.7B-v1.0 10.7 47.54 Mistral-Nemo-Base-2407 12.2 47.28 internlm2-20b 20.0 47.15 Meta-Llama-3.1-8B 8.0 43.77 Meta-Llama-3-8B 8.0 43.30 Qwen1.5-72B 72.3 39.51 Mistral-7B-v0.3 7.0 38.88 Mistral-7B-v0.2 7.0 38.81 Qwen1.5-7B 7.0 37.92 Bielik-7B-v0.1 7.2 34.34 Qra-13b 13.0 33.90 Llama-3.2-3B 3.0 31.89 Qra-7b 7.0 16.60

Table 7: Open PL LLM Leaderboard results for base models (5-shot evaluation)

- • psc: Summarization of news articles [Ogrodniczuk and Kope´c, 2014]; metric: binary F1.
- • cbd: Text classification for cyberbullying and hate-speech detection [Ptaszynski et al., 2023]; metric: macro F1.
- • polqa: Open-domain question answering from the "Jeden z dziesi˛eciu" TV show, with and without context (abstractive QA/RAG) [Rybak et al., 2024]; metric: accuracy, levenshtein.
- • poquad: Context-based extractive question answering (QA/RAG) [Tuora et al., 2023]; metric: levenshtein.
- • eqbench: emotional intelligence benchmark [Paech, 2023]; metric: custom.

Most of the tasks are multiple-choice tests, which means that the model chooses the correct answer from a set of options. They are implemented as two types of tests:

- • Loglikelihood: We choose the highest probability token from the given set, e.g., ABCD. These tests are suitable for base models.
- • Generate: Model generates answer freely.

All tasks are evaluated in both 0-shot and 5-shot settings, with the average score across all tasks normalized by baseline scores.

It is important to note that PLLuM models are not included in this leaderboard, as they were trained on training portions of the tasks used in the benchmark (except for the Belebele and EQ-Bench tasks), unlike all other models present on the leaderboard, to the best of our knowledge. The authors of the datasets used in the benchmark are primarily PLLuM consortium members.

As shown in Table 7, Bielik-11B-v2 achieved an impressive average score of 58.14, making it competitive with much larger models. It obtains similar scores to Qwen1.5-32B (58.71) despite having significantly fewer parameters, and shows a substantial improvement over its predecessor, Bielik-7B-v0.1, which scored 34.34.

The instruction-tuned versions show further performance improvements. The latest results from the Open PL LLM Leaderboard (Table 8) highlight the outstanding performance of the Bielik models. The Bielik-11B-v2.3-Instruct

Mistral-Large-Instruct-2411 123.0 69.84 Meta-Llama-3.1-405B-Instruct-FP8 405.0 69.44 Mistral-Large-Instruct-2407 123.0 69.11 Qwen2.5-72B-Instruct 72.7 67.92 QwQ-32B-Preview 32.8 67.01 Llama-3.3-70B-Instruct 70.6 66.40

- Qwen2-72B-Instruct 72.7 65.87 Bielik-11B-v2.3-Instruct 11.2 65.71 Bielik-11B-v2.2-Instruct 11.2 65.57 Meta-Llama-3.1-70B-Instruct 70.6 65.49 Bielik-11B-v2.1-Instruct 11.2 65.45 Mixtral-8x22B-Instruct-v0.1 141.0 65.23 Bielik-11B-v2.0-Instruct 11.2 64.98 Meta-Llama-3-70B-Instruct 70.6 64.45
- Qwen3-32B 32.8 64.24 Llama-4-Scout-17B-16E-Instruct 109.0 64.21 Bielik-11B-v2.5-Instruct 11.2 63.95 Mistral-Small-24B-Instruct-2501 24.0 62.97 phi-4 14.7 62.57 Qwen3-14B 14.8 62.24 Mistral-Small-Instruct-2409 22.2 61.41 Qwen2.5-32B-Instruct 32.8 61.21 Qwen2.5-14B-Instruct 14.8 59.91 aya-23-35B 35.0 56.37 Qwen3-8B 8.2 55.78 Qwen3-4B 4.0 55.49 Mistral-Nemo-Instruct-2407 12.2 55.27 Qwen2.5-7B-Instruct 7.6 54.93 Mistral-7B-Instruct-v0.3 7.2 47.74 Mistral-7B-Instruct-v0.2 7.2 45.95 Bielik-7B-Instruct-v0.1 7.2 44.70 Qwen2.5-3B-Instruct 3.1 41.23 Qwen3-1.7B 2.0 38.34 Mistral-7B-Instruct-v0.1 7.0 33.11 Qwen2.5-1.5B-Instruct 1.5 31.89

Table 8: Open PL LLM Leaderboard results for instruction-tuned models (5-shot evaluation)

achieves a remarkable score of 65.71, placing it among the top performers and outperforming many models with significantly more parameters.

Several key observations emerge from these results:

- 1. The entire Bielik-11B-v2.x family performs exceptionally well, with scores ranging from 64.98 to 65.71, placing them ahead of many larger models including Meta-Llama-3-70B-Instruct (64.45).
- 2. The Bielik-11B-v2.3-Instruct serves as the flagship model, with the Bielik-11B-v2.2-Instruct version following closely at 65.57.
- 3. The latest Bielik-11B-v2.5-Instruct model achieves a strong score of 63.95. While slightly lower than its v2.3 predecessor, this version incorporates enhanced function-calling capabilities and other architectural improvements that trade off some benchmark performance for improved practical functionality.
- 4. When considering parameter efficiency, Bielik models are particularly impressive, outperforming many models with 2-6 times more parameters.
- 5. The evolution from Bielik-7B-Instruct-v0.1 (44.70) to the v2 series shows a dramatic improvement of more than 20 percentage points.

## 6.1.1 Quantization Performance

Model quantization is essential for deploying large language models efficiently on consumer hardware. The Open PL LLM benchmark provides valuable insights into how different quantization methods affect Bielik’s performance, showing its practical utility in resource-constrained environments.

Model Params (B) Average (%) Bielik-11B-v2.3-Instruct.Q8_0.gguf 11.2 65.76 Bielik-11B-v2.3-Instruct 11.2 65.71 Bielik-11B-v2.3-Instruct.Q6_K.gguf 11.2 65.26 Bielik-11B-v2.3-Instruct.IQ3_XXS.gguf.IQ 11.2 64.89 Bielik-11B-v2.3-Instruct.Q4_K_M.gguf 11.2 64.76 Bielik-11B-v2.3-Instruct.Q4_K_M.gguf.IQ 11.2 64.71 Bielik-11B-v2.3-Instruct.IQ2_XXS.gguf.IQ 11.2 61.34 Mistral-Nemo-Instruct-2407 12.2 55.27 Bielik-11B-v2.3-Instruct.IQ1_M.gguf.IQ 11.2 52.09 Mistral-7B-Instruct-v0.3 7.2 47.74 Bielik-7B-Instruct-v0.1 7.2 44.70

Table 9: Open PL LLM benchmark results for Bielik-11B-v2.3-Instruct across different quantization methods, compared to other models. Higher scores are better.

Resilience to quantization: Bielik-11B-v2.3-Instruct demonstrates remarkable resilience across different quantization methods (as shown in Table 9):

- • The Q8_0 quantized version (65.76%) actually slightly outperforms the original full-precision model (65.71%)
- • More aggressive quantization with Q6_K (65.26%), IQ3_XXS (64.89%), and Q4_K_M (64.76%) maintains over 98% of the original performance
- • Even at extreme compression with IQ2_XXS (61.34%), Bielik preserves 93% of its original capability
- • The most aggressive IQ1_M quantization (52.09%) still outperforms both Mistral-7B-Instruct-v0.3 (47.74%) and Bielik-7B-Instruct-v0.1 (44.70%)

Practical implications: These results have significant implications for real-world applications:

- • Bielik can be effectively deployed on consumer-grade hardware with minimal performance degradation
- • The Q4_K_M quantization offers an excellent balance of model size reduction and performance preservation
- • Even highly compressed versions remain competitive with larger models, enabling deployment in memoryconstrained environments
- • The model architecture appears inherently robust to quantization effects, suggesting effective weight distribution during training

This quantization analysis demonstrates Bielik’s practical utility beyond benchmark scores, showing that its strong performance can be maintained even under significant compression. This makes the model particularly valuable for applications where deployment efficiency is critical without sacrificing Polish language understanding capabilities.

## 6.2 Polish MT-Bench

MT-bench Zheng et al. [2023] is a tool designed to test the ability of language models (LLMs) to conduct two-step conversations and follow instructions. It covers typical use cases and focuses on challenging questions to differentiate the capabilities of various models. Eight main categories of user queries were identified, which were used to construct MT-bench:

- • writing
- • role-playing
- • information extraction
- • reasoning
- • mathematics
- • coding
- • knowledge / hard sciences / stem
- • knowledge / humanities / social sciences

##### Model Params (B) Score Coding Extract. Human. Math Reason. Role-play STEM Writing

gemma-3-27b-it 27.4 9.28 8.10 9.90 10.00 8.25 8.40 9.95 9.95 9.70 Mistral-3.1-24B 24.0 9.18 8.30 9.80 10.00 7.85 9.00 9.40 9.90 9.15 phi-4 14.7 9.07 7.60 9.30 9.95 7.70 9.55 9.20 10.00 9.25 gemma-3-12b-it 12.0 8.97 8.25 9.55 10.00 7.45 7.75 9.45 10.00 9.30 Qwen2.5-32B-Instruct 32.8 8.86 7.95 9.90 9.65 7.60 9.10 8.30 9.70 8.65 Qwen2-72B-Instruct 72.7 8.78 7.80 9.80 9.75 6.50 8.85 9.20 9.55 8.75 Mistral-Small-2501 24.0 8.72 7.95 9.90 9.70 7.83 7.90 9.05 9.50 7.95 Mistral-Large-2407 123.0 8.66 6.75 9.90 9.40 7.80 8.70 8.70 9.35 8.70 aya-expanse-32b 32.0 8.62 5.75 8.40 10.00 6.60 8.95 9.70 9.95 9.60

- gemma-2-27b-it 27.4 8.62 7.45 9.60 10.00 7.80 6.85 8.70 9.80 8.75 Mistral-Small-2409 22.2 8.56 7.10 9.15 10.00 7.00 7.90 8.90 9.65 8.80 Bielik-11B-v2.3-Instruct 11.2 8.56 6.25 9.43 9.50 7.70 8.35 8.75 8.98 9.50 Qwen2.5-14B-Instruct 14.8 8.33 6.70 9.25 9.18 8.10 7.55 8.50 9.60 7.75
- gemma-3-4b-it 4.0 8.22 5.40 8.40 9.90 7.40 6.25 9.45 9.65 9.30 Meta-Llama-3.1-405B 405.0 8.17 7.25 9.85 9.65 6.25 5.80 8.70 8.65 9.20 Meta-Llama-3.1-70B 70.6 8.15 6.25 9.85 9.50 6.00 6.15 8.80 9.55 9.10

- Bielik-11B-v2.2-Instruct 11.2 8.12 5.05 9.30 9.40 6.45 6.90 9.03 9.45 9.35 Bielik-11B-v2.1-Instruct 11.2 8.00 5.40 9.13 9.20 6.15 6.25 9.45 8.90 9.50 aya-expanse-8b 8.0 7.76 4.90 8.05 9.65 4.35 6.85 9.25 9.75 9.30 gpt-3.5-turbo Unknown 7.72 6.00 8.15 9.75 6.85 5.20 8.65 9.25 9.10 Mixtral-8x7b 46.7 7.64 5.20 8.15 9.45 5.65 5.80 8.95 8.55 9.35 Bielik-11B-v2.0-Instruct 11.2 7.56 5.60 8.65 9.43 5.50 6.05 7.75 8.78 8.75 Mistral-Nemo-2407 12.2 7.37 5.85 8.95 9.50 6.70 5.80 7.45 8.30 6.40 Llama-PLLuM-70B-chat 70.6 6.75 4.80 9.45 8.80 2.90 5.20 6.60 8.20 8.05 openchat-3.5-0106-gemma 7.0 6.51 5.35 6.90 8.80 4.55 5.40 7.98 8.48 7.05 PLLuM-12B-nc-chat† 12.2 6.47 4.55 7.20 9.50 2.30 4.80 6.75 9.10 7.55 PLLuM-8x7B-nc-chat† 46.7 6.43 4.10 8.40 7.48 3.35 4.95 6.90 8.90 7.40 PLLuM-8x7B-chat 46.7 6.30 4.55 8.00 8.60 3.45 4.90 6.25 8.20 7.10 Llama-PLLuM-8B-chat 8.0 6.05 3.65 6.30 9.50 2.75 5.35 6.15 7.50 7.20 PLLuM-12B-chat 12.2 5.81 3.05 6.55 9.30 2.65 3.90 5.00 8.00 8.00 Bielik-7B-Instruct-v0.1 7.2 5.40 3.00 4.35 8.48 4.10 6.15 7.83 6.90 7.85 dolphin-2.9.1-llama-3-8b 8.0 5.24 4.60 6.15 8.80 4.80 3.30 7.40 6.35 5.50 Polka-Mistral-7B-SFT 7.0 4.43 2.95 5.25 5.60 2.95 2.45 4.90 6.80 5.25 trurl-2-7b 7.0 2.75 1.80 3.50 3.95 1.70 2.05 3.30 2.65 3.15 †Models with a non-commercial license.

Table 10: Polish MT-Bench results for various models across all evaluation categories.

The evaluation of responses is performed by a metamodel. In the case of MT-Bench, this is the GPT-4 model. By using a metamodel, we can verify responses from open-ended questions, e.g., write an article about hybrid cars. The model evaluates the content of the response, the quality of facts used, creativity, etc.

The Polish MT-Bench Kinas et al. [2024] has been completely polonized. Each task was first machine-translated and then verified. Additionally, we introduced Polish accents, e.g., instead of describing a vacation in Hawaii, we suggested the location - Mazury (Masuria). In our language version, many changes were introduced to transfer the test into Polish linguistic realities.

The latest results from the Polish MT-Bench (Table 10) reveal several key insights about the Bielik models’ performance:

Top-tier performance: Bielik-11B-v2.3-Instruct achieves a remarkable score of 8.56, placing it in the top tier of models alongside much larger architectures. It matches the performance of Mistral-Small-Instruct-2409 (22.2B) and outperforms larger models including Meta-Llama-3.1-405B-Instruct (405B) and Meta-Llama-3.1-70B-Instruct (70B).

Efficiency advantage: When considering parameter count efficiency, Bielik-11B-v2.3-Instruct demonstrates exceptional performance. It achieves comparable or better results than models with 2-6 times more parameters, including Qwen2.5-14B-Instruct (14B) and aya-expanse-32b (32B).

Category strengths: Examining performance across categories, Bielik-11B-v2.3-Instruct shows particular strength in mathematics (7.70), reasoning (8.35), and writing (9.50). The writing score is especially noteworthy, matching

Model Parameters (B) Score Mistral-Large-Instruct-2407† 123.0 78.07 Mistral-Large-Instruct-2411† 123.0 77.29 Meta-Llama-3.1-405B-Instruct-FP8 405.0 77.23 gpt-4o-2024-08-06 Unknown 75.15 gpt-4-turbo-2024-04-09 Unknown 74.59 Mistral-Small-Instruct-2409 22.2 72.85 Llama-PLLuM-70B-chat 70.6 72.56 Meta-Llama-3.1-70B-Instruct 70.6 72.53 Bielik-11B-v2.5-Instruct 11.2 72.00 Qwen2-72B-Instruct 72.7 71.23 Meta-Llama-3-70B-Instruct 70.6 71.21 gpt-4o-mini-2024-07-18 Unknown 71.15 Qwen2.5-32B-Instruct 32.8 71.15 Bielik-11B-v2.3-Instruct 11.2 70.86 Llama-3.3-70B-Instruct 70.6 70.73 Llama-PLLuM-70B-instruct 70.6 69.99 WizardLM-2-8x22B 141.0 69.56 Qwen2.5-14B-Instruct 14.8 69.17 Bielik-11B-v2.2-Instruct 11.2 69.05

- Bielik-11B-v2.0-Instruct 11.2 68.24 glm-4-9b-chat 9.0 61.79 Mistral-Nemo-Instruct-2407 12.2 61.76
- Bielik-11B-v2.1-Instruct 11.2 60.07 PLLuM-12B-chat 12.2 52.26 PLLuM-8x7B-nc-chat† 46.7 47.29 Llama-PLLuM-8B-chat 8.0 46.20 PLLuM-8x7B-chat 46.7 45.22 PLLuM-8x7B-nc-instruct† 46.7 41.75 PLLuM-8x7B-instruct 46.7 39.55 PLLuM-12B-instruct 12.2 36.21 PLLuM-12B-nc-chat† 12.2 35.41 Llama-PLLuM-8B-instruct 8.0 31.59 PLLuM-12B-nc-instruct† 12.2 13.11 †Models with a non-commercial license.

Table 11: Polish EQ-Bench results for various models.

gemma-3-27b-it despite the latter having over twice the parameters. This balanced performance across all categories indicates versatility rather than specialization in particular areas.

Comparison with other Polish models: The performance gap between Bielik and other dedicated Polish language models is substantial. Bielik-11B-v2.3-Instruct outperforms the best PLLuM model (Llama-PLLuM-70B-chat, 6.75) by 1.81 points despite having less than 16% of its parameters. Even more striking, it surpasses PLLuM-12B-nc-chat (6.47) by 2.09 points despite similar parameter counts.

Evolutionary improvements: The progression from Bielik-7B-Instruct-v0.1 (5.40) to Bielik-11B-v2.3-Instruct (8.56) demonstrates a dramatic 3.16-point improvement. Even within the v2 series, each iteration shows clear advancements, with v2.3 improving 0.44 points over v2.2, 0.56 points over v2.1, and 1.00 points over v2.0.

These results highlight Bielik-11B-v2.3-Instruct’s position as a state-of-the-art model for Polish language processing, delivering performance comparable to much larger models while maintaining efficiency in parameter count.

## 6.3 Polish EQ-Bench

The Polish Emotional Intelligence Benchmark, which is a localized Polish version of the original EQ-Bench [Paech, 2023], evaluates the emotional intelligence capabilities of language models across various dimensions of emotional understanding and response.

- Bielik-11B-v2.3-Instruct demonstrates exceptional emotional intelligence capabilities with a score of 70.86 (as shown in Table 11), outperforming many larger models including WizardLM-2-8x22B (141B parameters). This places it just 1.67 points behind Qwen2.5-32B-Instruct despite having only a third of the parameters.

The newest addition to the family, Bielik-11B-v2.5-Instruct, further improves on these results with a score of 72.00, surpassing both Qwen2-72B-Instruct (71.23) and Meta-Llama-3-70B-Instruct (71.21) despite having just a fraction of their parameters. This positions it less than 0.53 points behind Meta-Llama-3.1-70B-Instruct (72.53), showcasing the effectiveness of the latest training methodology.

In the comparison with other Polish language models Bielik-11B-v2.3-Instruct outperformed most of PLLuM model variants except the largest 70B models. It surpasses:

- • PLLuM-12B-chat by 18.60 points (70.86 vs 52.26)
- • PLLuM-8x7B-nc-chat by 23.57 points (70.86 vs 47.29) (MoE architecture with 46.7B effective parameters)
- • Smaller PLLuM variants by 25-57 points

Among the Bielik variants, v2.3-Instruct demonstrates clear superiority over its predecessors, with a 1.81 point improvement over v2.2-Instruct, 2.62 points over v2.1, and a substantial 10.79 point advantage over v2.0. The gap between Bielik and the top-performing models (Mistral-Large and Llama-3.1-405B) is remarkably narrow considering the significant difference in model scale.

## 6.4 Complex Polish Text Understanding Benchmark (CPTUB)

The Complex Polish Text Understanding Benchmark (CPTUB) [Sowa et al., 2024] is specifically designed to evaluate language models’ proficiency in interpreting complex Polish texts. Unlike traditional tasks that focus on explicit meaning, CPTUB assesses the models’ capacity to understand implied meanings and handle cognitively challenging questions. The benchmark comprises two main components:

- • Implicatures: Evaluates a model’s ability to interpret implied meanings, including sarcasm, idiomatic expressions, and phraseological compounds. This component tests sensitivity to nuanced, context-dependent inferences through three subtasks:

- – Sentiment: Correctly identifying the emotional tone beyond literal expressions
- – Language understanding: Interpreting the underlying intentions of text authors
- – Phraseology: Recognizing and explaining fixed or semi-fixed expressions whose meanings cannot be inferred from their individual components

- • Tricky Questions: Assesses a model’s capability to address challenging questions characterized by logical puzzles, semantic ambiguity, logical inconsistencies, absurdity, and humor. This component specifically targets the model’s reasoning skills and ability to avoid hallucinations when faced with ambiguous or nonsensical queries.

As shown in Table 12, Bielik models demonstrate strong performance on this challenging benchmark, particularly in understanding implied meanings in Polish text. The best-performing variant, Bielik-11B-v2.3-Instruct, achieves an overall score of 3.63, placing it ahead of many larger models, including Mixtral-8x22B-Instruct-v0.1 (3.56).

Several observations can be made from these results:

- 1. Bielik-11B-v2.3-Instruct and Bielik-11B-v2.1-Instruct show particularly strong performance in sentiment analysis (3.97 and 3.96) and language understanding (3.79 and 3.92), indicating robust capabilities in interpreting emotional cues and author intentions in complex Polish texts.
- 2. The latest Bielik-11B-v2.5-Instruct achieves a score of 3.48, with notable strength in sentiment analysis (4.01) and language understanding (3.86). Its performance in tricky questions (2.91) is somewhat lower than previous versions, reflecting its optimization focus on other capabilities.
- 3. The performance gap between Bielik models and similarly-sized models is notable. For instance, Bielik-11Bv2.3-Instruct (3.63) substantially outperforms phi-4 (3.30) despite having fewer parameters.
- 4. When compared to other Polish language models, Bielik models show clear superiority over the PLLuM family. Bielik-11B-v2.3-Instruct (3.63) significantly outperforms even the largest PLLuM model, Llama-PLLuM-70Bchat (3.53), despite having only about 16% of its parameters.

Model Params (B) Overall Implicatures Senti- Language Phrase- Tricky Average Average ment Understanding ology Questions

DeepSeek-R1 685.0 4.14 4.14 4.49 4.35 3.60 4.12 Mistral-Large-Instruct-2411† 123.0 4.00 4.10 4.33 3.98 3.99 3.72 Qwen2.5-72B-Instruct 72.7 3.95 3.99 4.08 3.97 3.93 3.81 Mistral-Large-Instruct-2407† 123.0 3.93 4.03 4.23 4.00 3.86 3.65 Llama-4-Maverick-17B-128E-Instruct-FP8 402.0 3.93 3.99 4.39 4.11 3.48 3.76 gemma-3-27b-it 27.4 3.81 3.90 3.88 3.79 4.03 3.53 Meta-Llama-3-70B-Instruct 70.6 3.78 3.81 4.13 3.82 3.47 3.71 Qwen2.5-32B-Instruct 32.8 3.75 3.80 3.81 3.57 4.04 3.59 Llama-4-Scout-17B-16E-Instruct 109.0 3.75 3.94 4.10 3.81 3.90 3.19 Bielik-11B-v2.3-Instruct 11.2 3.63 3.77 3.97 3.79 3.55 3.22

- Bielik-11B-v2.1-Instruct 11.2 3.61 3.66 3.96 3.92 3.11 3.47 Mixtral-8x22B-Instruct-v0.1 141.0 3.56 3.67 3.78 3.68 3.55 3.24 Qwen2.5-14B-Instruct 14.8 3.55 3.62 3.91 3.57 3.37 3.34 Llama-PLLuM-70B-chat 70.6 3.53 3.63 3.94 3.61 3.35 3.21 Bielik-11B-v2.5-Instruct 11.2 3.48 3.67 4.01 3.86 3.13 2.91
- Bielik-11B-v2.2-Instruct 11.2 3.46 3.57 3.72 3.73 3.25 3.12 Llama-PLLuM-70B-instruct 70.6 3.33 3.56 3.78 3.63 3.26 2.63 phi-4 14.7 3.30 3.50 3.72 3.54 3.24 2.72

- Bielik-11B-v2.0-Instruct 11.2 3.26 3.61 3.97 3.75 3.13 2.20 PLLuM-12B-nc-chat† 12.2 3.15 3.33 3.22 3.23 3.54 2.62 PLLuM-12B-chat 12.2 3.14 3.32 3.32 3.21 3.43 2.59 PLLuM-8x7B-nc-instruct† 46.7 3.11 3.56 3.88 3.59 3.22 1.76 PLLuM-12B-instruct 12.2 3.09 3.49 3.71 3.17 3.59 1.90 Qwen2.5-7B-Instruct 7.62 3.07 3.23 3.56 3.03 3.10 2.58 PLLuM-8x7B-nc-chat† 46.7 3.03 3.44 3.76 3.48 3.08 1.80 Meta-Llama-3.1-8B-Instruct 8.0 3.01 3.31 3.97 3.38 2.58 2.11 PLLuM-8x7B-instruct 46.7 3.01 3.51 3.59 3.47 3.46 1.51 PLLuM-8x7B-chat 46.7 3.01 3.41 3.44 3.45 3.35 1.78 Meta-Llama-3-8B-Instruct 8.0 3.00 3.17 3.33 3.15 3.04 2.48 Llama-PLLuM-8B-chat 8.0 2.92 3.14 3.13 2.93 3.36 2.25 Bielik-7B-Instruct-v0.1 7.2 2.88 3.13 3.59 3.48 2.32 2.16 Llama-PLLuM-8B-instruct 8.0 2.82 3.20 3.24 2.90 3.46 1.66 †Models with a non-commercial license.

- Table 12: Complex Polish Text Understanding Benchmark (CPTUB) results across different evaluation categories

5. In the challenging "Tricky Questions" category, which tests reasoning and hallucination avoidance, Bielik11B-v2.1-Instruct achieves the best score among the Bielik family at 3.47, demonstrating its robustness against ambiguous or nonsensical queries.

These results highlight the effectiveness of Bielik models in handling complex linguistic phenomena specific to the Polish language, suggesting that the training methodology has successfully captured nuanced aspects of Polish language understanding beyond simple translation or basic comprehension tasks.

## 6.5 Polish Medical Leaderboard

The Polish Medical Leaderboard evaluates language models on Polish Board Certification Examinations (Pa´nstwowy Egzamin Specjalizacyjny, PES) from years 2018-2022. This benchmark assesses models’ medical knowledge and reasoning capabilities in a Polish-language medical context, using datasets from speakleash/PES-2018-2022, which is based on amu-cai/PES-2018-2022 Pokrywka et al. [2024].

Bielik’s performance: In the Polish Medical Leaderboard (Table 13), Bielik-11B-v2.3-Instruct achieves a score of 43.26%, demonstrating solid medical reasoning capabilities in Polish:

- • The latest Bielik-11B-v2.5-Instruct shows further improvement with a score of 44.85%, marking the best performance among all Bielik variants
- • The model performs competitively among models of similar size, outperforming Meta-Llama-3.1-8B-Instruct (40.60%)

##### Model Parameters (B) Average (%)

Meta-Llama-3.1-405B-Instruct-FP8 405.0 69.20 Mistral-Large-Instruct-2407† 123.0 64.28 Qwen2.5-72B-Instruct 72.7 63.89 Meta-Llama-3.1-70B-Instruct 70.6 61.75 Qwen2-72B-Instruct 72.7 61.35 Meta-Llama-3-70B-Instruct 70.6 57.51 Qwen2.5-32B 32.8 55.69 Qwen2.5-32B-Instruct 32.8 54.52 Qwen2.5-14B-Instruct 14.8 49.60 Bielik-11B-v2.5-Instruct 11.2 44.85 GLM-4-9b-chat 9.0 44.54 Mistral-Small-Instruct-2409 22.2 43.60 Bielik-11B-v2.3-Instruct 11.2 43.26

- Bielik-11B-v2.1-Instruct 11.2 43.16
- Bielik-11B-v2.2-Instruct 11.2 43.05 Qwen2.5-7B-Instruct 7.6 42.69 Bielik-11B-v2.0-Instruct 11.2 41.53 Meta-Llama-3.1-8B-Instruct 8.0 40.60 Mistral-Nemo-Instruct-2407 12.2 40.36 Bielik-11B-v2 11.2 39.98 †Models with a non-commercial license.

- Table 13: Polish Medical Leaderboard results (5-shot setting) showing model performance on Polish Board Certification Examinations.

- • The entire Bielik-11B-v2.x family shows consistent performance in the 41-45% range
- • Bielik models outperform Mistral-Nemo-Instruct models

Performance context: The benchmark highlights the following insights about Bielik’s medical capabilities:

- • Bielik demonstrates respectable medical reasoning without domain-specific training
- • The model’s performance points to strong cross-domain generalization from general Polish language understanding to specialized medical knowledge
- • The gap between Bielik (43.26%) and top-performing models like Meta-Llama-3.1-405B-Instruct (69.20%) is expected given the significant difference in model scale (11.2B vs 405B parameters)

These results highlight Bielik’s versatility across different knowledge domains, achieving competitive performance on specialized medical examinations despite not being specifically trained for medical applications. This versatility makes Bielik suitable for a wide range of practical applications in Polish language processing.

## 6.6 Polish Linguistic and Cultural Competency Benchmark (PLCC)

The Polish Linguistic and Cultural Competency Benchmark (PLCC) [Dadas et al., 2025] evaluates language models’ knowledge of Polish cultural context, moving beyond traditional NLP tasks to assess cultural understanding. The benchmark consists of 600 manually crafted questions across six categories: history, geography, culture & tradition, art & entertainment, grammar, and vocabulary.

PLCC questions assess knowledge of Polish cultural references, historical events, traditions, folklore, literature, and pop culture—competencies essential for genuine language understanding beyond surface-level text processing. Questions range from commonly known facts to region-specific cultural knowledge, with both closed and open-ended formats requiring specific facts, dates, names, or concepts in responses.

The PLCC results (Table 14) reveal several insights about Bielik models’ cultural competency:

Strong cultural grounding: Bielik-11B-v2.2-Instruct achieves a remarkable 63.00% average score across all six categories, trailing behind larger models including DeepSeek variants (69.17-76.00%) and PLLuM-8x7B-nc-chat

- (68.17%), while outperforming many other models despite having fewer parameters. This demonstrates Bielik’s strong grasp of Polish cultural nuances despite its modest parameter count.

##### Model Parameters (B) Average Score (%)

Gemini-2.5-Pro-Exp-03-25 Unknown 89.50 DeepSeek-R1 685.0 76.00 DeepSeek-v3-0324 685.0 71.00 DeepSeek-v3 685.0 69.17 PLLuM-8x7B-nc-chat† 46.7 68.17 Llama-3.1-Tulu-3-405B 405.0 63.83

- Bielik-11B-v2.2-Instruct 11.2 63.00
- Bielik-11B-v2.3-Instruct 11.2 62.17 GPT-4.1-mini-2025-04-14 Unknown 62.17 Bielik-11B-v2.1-Instruct 11.2 61.00 Llama-3.1-405B 405.0 60.00 PLLuM-12B-nc-chat† 12.2 59.50 Llama-PLLuM-70B-chat 70.6 58.50 Llama-4-Maverick 402.0 58.17 Command-A-03-2025† 111.0 56.17 Mistral-Large-2407† 123.0 54.17 PLLuM-8x7B-chat 46.7 54.17 Mistral-Large-2411† 123.0 52.00 WizardLM-2-8x22B 141.0 51.50 Qwen-Max Unknown 50.83 Command-R-Plus-08-2024† Unknown 50.17 Mixtral-8x22B 141.0 49.83 Command-R-Plus-04-2024† Unknown 49.33 Llama-3.3-70B 70.6 48.83 Llama-3.1-70B 70.0 47.83 Gemma-3-27B 27.4 47.33 PLLuM-12B-chat 12.2 47.00 Bielik-7B-Instruct-v0.1 7.0 46.67 Mistral-Small-3.1-24B-2503 24.0 43.33

- Llama-3.0-70B 70.0 43.00 Gemma-2-27B 27.0 42.67 Llama-4-Scout 109.0 41.50 EuroLLM-9B 9.0 41.00 Qwen-2.5-72B 72.7 39.17 Mistral-Small-24B-2501 24.0 39.00 Llama-PLLuM-8B-chat 8.0 38.50 Mixtral-8x7B 46.7 35.33 Qwen-2.5-32B 32.8 30.50 Gemma-2-9B 9.0 29.17 Phi-4 14.7 29.17 Qwen-2.5-14B 14.8 26.67 Mistral-Nemo 12.2 23.00 Command-R-7B† 7.0 22.83
- Llama-3.1-8B 8.0 22.67 Mistral-7B-v0.3 7.2 21.83 Ministral-8B 8.0 20.67 Qwen-2.5-7B 7.0 17.67 †Models with a non-commercial license.

- Table 14: Polish Linguistic and Cultural Competency Benchmark (PLCC) results for open-source models. Closed proprietary models have been excluded from this comparison.

Efficiency advantage: With just 11B parameters, Bielik models outperform significantly larger models, including Llama-3.1-405B (60.00%) and Llama-3.3-70B (48.83%). This efficiency highlights the effectiveness of Bielik’s training methodology in capturing cultural knowledge without requiring massive parameter counts.

Comparative Polish model performance: Bielik models show competitive performance against specialized Polish language models. While PLLuM-8x7B-nc-chat (non-commercial license) leads with 68.17%, Bielik’s 63.00% significantly outperforms most other PLLuM variants including PLLuM-12B-chat (47.00%) and Llama-PLLuM-8B-chat (38.50%).

Strength across categories: Bielik-11B-v2.2-Instruct demonstrates particular strength in history (77%) and geography (72%) categories, showcasing its robust knowledge of Polish historical events and geographical features. Performance in grammar (53%) suggests areas for further improvement.

Evolutionary improvement: The progression from Bielik-7B-Instruct-v0.1 (46.67%) to Bielik-11B-v2.2-Instruct (63.00%) demonstrates a substantial 16.33 percentage point improvement, highlighting significant advances in the model’s cultural understanding across versions.

These results underscore Bielik’s proficiency in handling culturally-specific Polish knowledge, outperforming many larger models and demonstrating that effective training on culturally relevant data can yield strong results without requiring massive parameter counts.

## 6.7 LLMzSzŁ (LLMs Behind the School Desk)

The LLMzSzŁ benchmark (LLM-y za Szkolna˛ Ława˛, LLMs Behind the School Desk) Jassem et al. [2025] is a comprehensive evaluation framework for Polish language models based on a collection of Polish national exams. It incorporates both academic and professional tests extracted from the archives of the Polish Central Examination Board, providing a realistic assessment of models’ reasoning abilities in an educational context.

The benchmark includes a diverse range of exams with questions spanning multiple disciplines. The academic exams cover subjects like Polish Language, Mathematics, Nature, Biology, and Physics, while the professional exams include specialized fields such as Arts, Mechanical/Mining/Metallurgical, and Agriculture/Forestry disciplines.

|Exam Type|Number of Questions|
|---|---|
|8th-Grade Exam Middle School Exam High School Exam Professional Exam<br><br>|50 175 377 18219|

Bielik’s performance: On the LLMzSzŁ benchmark (Table 15), Bielik models demonstrate strong performance, with

- Bielik-11B-v2.1-Instruct achieving the highest score of 57.52 among the Bielik family. Key observations include:

- • The entire Bielik-11B-v2.x family performs consistently well, with scores ranging from 55.61 to 57.52
- • Bielik models achieve these scores despite having significantly fewer parameters than the top-performing models
- • There is a substantial improvement from Bielik-7B-v0.1 (39.15) to the v2 series (55.14+), highlighting the effectiveness of the v2 training methodology
- • Bielik models significantly outperform Polish-focused models like Qra (34.85) and Trurl (40.22), as well as PLLuM 12B and 8B variants, while only trailing behind the larger PLLuM 8x7B models (46.7B parameters) and PLLuM 70B models (70.6B parameters)

Performance across exam types: Bielik models show interesting patterns across different exam categories:

- • Strong performance on Professional Exams (56.67-56.93), where practical knowledge and domain-specific understanding are crucial
- • More balanced performance across academic exams compared to some larger models that excel in particular exam types
- • Stronger results on 8th-Grade Exams (45.73-49.70) compared to Middle School Exams (47.36-48.62) and High School Exams (46.10-47.53), suggesting effective handling of more accessible educational content

Parameter efficiency: With only 11 billion parameters, Bielik achieves impressive results relative to much larger models:

- • Outperforms PLLuM-12B-nc-chat (53.40) and matches DeepSeek-R1-Distill-Qwen-32B (55.80) and Qwen2.514B (55.31) despite having fewer parameters
- • Demonstrates superior Polish language understanding compared to general models with similar parameter counts

Model Params Overall 8th-Grade Middle School High School Professional (B) Score Exams Exams Exams Exams

Qwen2.5-72B-Instruct 72.7 69.06 68.73 74.56 73.17 67.59 Qwen2.5-72B 72.7 68.50 65.24 77.36 69.49 67.24 Mistral-Large-Instruct-2407† 123.0 67.17 51.89 71.69 63.41 65.52 Llama-3.3-70B-Instruct 70.6 67.13 46.69 66.43 61.20 66.11 Mistral-Large-Instruct-2411† 123.0 66.60 50.44 71.06 63.14 65.11 Meta-Llama-3.1-70B-Instruct 70.6 66.59 53.81 63.23 62.72 65.39 Llama-PLLuM-70B-base 70.6 64.56 38.69 55.72 52.78 63.85 Llama-PLLuM-70B-chat 70.6 64.42 46.48 53.18 51.62 63.95 Qwen2.5-32B 32.8 61.04 55.56 68.57 64.47 59.72 PLLuM-8x7B-nc-chat† 46.7 60.52 38.69 48.94 41.08 59.62 PLLuM-8x7B-nc-base† 46.7 57.82 30.98 50.25 40.95 56.89

- Bielik-11B-v2.1-Instruct 11.2 57.52 45.73 47.36 46.10 56.93 Bielik-11B-v2.3-Instruct 11.2 57.40 47.88 48.62 46.55 56.74
- Bielik-11B-v2.2-Instruct 11.2 57.36 49.70 47.87 47.53 56.67 DeepSeek-R1-Distill-Qwen-32B 32.0 55.80 53.71 66.64 61.57 54.52 Bielik-11B-v2.0-Instruct 11.2 55.61 46.84 42.84 46.24 55.16 Qwen2.5-14B 14.8 55.31 45.50 63.31 54.86 54.19 Bielik-11B-v2 11.2 55.14 32.32 49.49 40.75 54.60 PLLuM-12B-nc-chat† 12.2 53.40 43.88 32.95 37.94 52.70 PLLuM-8x7B-chat 46.7 52.80 35.18 44.37 41.08 51.84 PLLuM-8x7B-base 46.7 52.24 30.35 44.10 44.22 51.41 EuroLLM-9B-Instruct 9.0 51.35 30.65 40.01 35.31 50.47 PLLuM-12B-nc-base† 12.2 50.55 35.88 41.46 37.41 50.15 PLLuM-12B-chat 12.2 49.69 36.17 38.09 40.15 48.66 Llama-PLLuM-8B-chat 8.0 47.68 37.36 31.35 38.86 47.18 Meta-Llama-3.1-8B-Instruct 8.0 47.41 39.81 44.50 37.06 47.17 PLLuM-12B-base 12.2 46.47 40.81 37.83 36.36 45.48 Llama-PLLuM-8B-base 8.0 46.32 27.98 38.88 37.55 45.44 Mistral-Nemo-Instruct-2407 12.2 45.54 37.99 42.30 37.68 44.71 Meta-Llama-3-8B-Instruct 8.0 44.83 38.14 38.89 38.70 43.97 Meta-Llama-3.1-8B 8.0 44.21 34.17 34.68 40.54 43.13 Mistral-Nemo-Base-2407 12.2 42.16 33.36 34.83 35.11 40.91 Meta-Llama-3-8B 8.0 41.38 44.03 38.45 38.72 40.14 Bielik-7B-Instruct-v0.1 7.2 40.77 35.68 30.48 36.58 40.05 Mistral-7B-Instruct-v0.2 7.2 40.75 33.03 45.17 30.80 40.06 Trurl-2-13b 13.0 40.22 34.77 33.78 34.63 39.61 Trurl-2-13b-8bit 13.0 40.23 34.43 35.17 34.31 39.64 Bielik-7B-v0.1 7.2 39.15 30.32 33.87 32.51 38.42 Trurl-2-13b-academic 13.0 34.89 30.80 31.95 30.43 34.51 Qra-13b 13.0 34.85 31.66 28.23 32.85 33.77 Trurl-2-7b 7.0 32.30 22.53 28.86 31.43 32.32 Trurl-2-7b-8bit 7.0 31.86 18.53 26.88 31.95 32.08 Qra-7b 7.0 29.07 19.65 25.89 29.27 28.54 EuroLLM-1.7B-Instruct 1.7 25.61 29.31 26.54 26.04 25.54 Qra-1b 1.0 25.47 30.22 26.41 25.23 25.20 †Models with a non-commercial license.

- Table 15: LLMzSzŁ benchmark results showing model performance on Polish national exams in decreasing order of overall score. The table displays performance across different exam types: 8th-Grade Exams, Middle School Exams, High School Exams, and Professional Exams (averaged by years).

• Shows remarkable performance-to-parameter ratio, achieving 83% of Qwen2.5-72B-Instruct’s score with only 15% of its parameters

These LLMzSzŁ results highlight Bielik’s exceptional capabilities in understanding and reasoning about Polish educational content, from basic school materials to professional certification exams. This performance is particularly impressive considering the model’s relatively modest size, demonstrating the effectiveness of specialized training for Polish language tasks.

##### Model Params (B) Average ARC GSM8K HellaSwag MMLU TruthfulQA

Gemma-2-27b-Instruct 27.0 0.71 0.74 0.77 0.71 0.68 0.63 Meta-Llama-3.1-70B-Instruct 70.6 0.70 0.72 0.69 0.74 0.77 0.59 Bielik-11B-v2.3-Instruct 11.2 0.66 0.69 0.68 0.71 0.63 0.62 Mixtral-8x7B-Instruct-v0.1 46.7 0.63 0.67 0.52 0.67 0.63 0.65 c4ai-command-r-35B-v01 35.0 0.61 0.66 0.49 0.71 0.60 0.56 Mistral-Nemo-Instruct-12.2B 12.2 0.60 0.64 0.58 0.61 0.59 0.60 EuroLLM-9B-Instruct 9.0 0.59 0.68 0.48 0.67 0.57 0.52 Mixtral-8x7B-v0.1 46.7 0.58 0.66 0.46 0.66 0.63 0.51 Gemma-2-9b-Instruct 9.0 0.58 0.68 0.45 0.61 0.59 0.58 Meta-Llama-3.1-8B-Instruct 8.0 0.57 0.57 0.59 0.58 0.57 0.55

Table 16: European LLM Leaderboard results for Polish language (top 10 models). Full results in Appendix A.

## 6.8 European LLM Leaderboard

The European LLM Leaderboard Thellmann et al. [2024] evaluates language models across multiple European languages, testing their understanding, reasoning, and generation capabilities. The benchmark assesses models using standardized tests like ARC, GSM8K, HellaSwag, MMLU, and TruthfulQA, providing a comprehensive view of performance across diverse linguistic contexts.

The European LLM Leaderboard results (Table 16) reveal Bielik-11B-v2.3-Instruct’s strong performance in Polish language tasks, placing it third among all evaluated models with an average score of 0.66. This positions it behind only Gemma-2-27b-Instruct (0.71) and Meta-Llama-3.1-70B-Instruct (0.70), while outperforming significantly larger models like Mixtral-8x7B-Instruct-v0.1 (0.63) with 47B parameters.

Balanced performance across tasks: Bielik demonstrates remarkable consistency across different evaluation dimensions:

- • Strong scientific reasoning capabilities in ARC (0.69)
- • Impressive mathematical problem-solving in GSM8K (0.68), nearly matching Llama-3.1-70B-Instruct (0.69)
- • Excellent common sense understanding with HellaSwag (0.71), on par with Gemma-2-27b-Instruct
- • Solid factual accuracy with TruthfulQA (0.62), outperforming larger models including Llama-3.1-70B-Instruct (0.59)
- • Competitive broad knowledge in MMLU (0.63), comparable to Mixtral-8x7B models

Cross-lingual capabilities: Beyond Polish, Bielik demonstrates robust cross-lingual transfer to other languages without specific training (Appendix A):

- • German: Average score of 0.62, showing stronger mathematical reasoning (GSM8K: 0.65) than many larger models
- • Czech: Average score of 0.60, particularly strong in mathematical reasoning (GSM8K: 0.60), outperforming Mixtral-8x7B-Instruct-v0.1 (0.50)

Translation performance: In the FLORES200 translation benchmark, Bielik shows asymmetric capabilities:

- • Stronger performance translating into Polish (BLEU: 15.31) than from Polish (BLEU: 11.72)
- • Excellent results with English-Polish translation (BLEU: 21.93 to Polish, 28.32 from Polish)
- • Strong performance with linguistically similar languages like Czech (BLEU: 19.30 to Polish)

These results demonstrate Bielik’s exceptional capabilities as a relatively small yet powerful model, achieving competitive performance against much larger models across a diverse range of tasks. The model’s efficiency is particularly evident in its strong performance despite having significantly fewer parameters than most top-performing alternatives.

## 6.9 EuroEval Leaderboard

EuroEval Nielsen [2023] is a comprehensive benchmark framework for evaluating language models across multiple European languages. Developed over the course of three years, it has become a standard evaluation benchmark for

##### Model Params (B) Rank EN DE FR IT ES NL PL DA SV NO

- Gemini-1.5-Pro-002 Unknown 1.38 1.61 1.27 1.48 1.57 1.35 1.40 1.22 1.33 1.29 1.27
- Gemini-2.0-Flash-001 Unknown 1.39 1.70 1.26 1.61 1.38 1.40 1.37 1.25 1.30 1.28 1.31 GPT-4o Unknown 1.50 1.92 1.37 1.80 1.51 1.45 1.44 1.32 1.37 1.41 1.39 Gemma-3-27B-it 27.4 1.74 2.09 1.55 2.04 2.22 1.70 1.81 1.52 1.68 1.63 1.60 Mistral-Small-24B 24.0 1.89 2.17 1.52 2.72 2.55 2.13 1.95 1.60 1.75 1.82 1.71 Gemma-3-12B-it 12.0 1.90 2.31 1.65 2.42 2.55 1.87 2.04 1.66 1.80 1.76 1.79 Gemma-2-27B-it 27.4 1.95 2.20 1.79 2.65 2.66 1.81 2.09 1.73 1.87 1.90 1.83 Gemma-2-9B-it 9.0 2.13 2.48 1.92 2.74 2.80 1.97 2.23 1.77 1.95 1.98 1.92 Bielik-11B-v2.3-Instruct 11.2 2.22 2.38 1.79 3.12 3.27 2.40 2.31 1.41 2.18 2.25 2.12 Meta-Llama-3.1-8B-Instruct 8.0 2.34 2.74 2.07 2.92 3.04 2.48 2.40 1.88 2.10 2.12 2.06

Table 17: EuroEval leaderboard results showing model ranking based on average performance across European languages (lower is better). The table includes performance scores for English (EN), German (DE), French (FR), Italian (IT), Spanish (ES), Dutch (NL), Polish (PL), Danish (DA), Swedish (SV), and Norwegian (NO). Bielik-11B-v2.3Instruct demonstrates competitive performance, especially considering its parameter efficiency and notably strong results in Polish.

various research institutions and organizations throughout Europe. The benchmark supports assessment of a wide range of model types, including encoders, decoders, encoder-decoders, base models, and instruction-tuned models.

- Bielik-11B-v2.3-Instruct’s results on the EuroEval leaderboard (Table 17) demonstrate solid performance with a rank of 2.22, placing it competitively among similarly sized models. While larger models like Gemini and GPT variants achieve stronger absolute scores, Bielik’s performance is noteworthy given its parameter efficiency.

Slavic language advantage: As expected, Bielik demonstrates exceptional performance in Polish (1.41), its primary target language, outperforming all other models including those with significantly more parameters. This highlights the effectiveness of specialized training on Polish data.

Cross-lingual performance by language family: Bielik shows a consistent pattern across language families:

- • Strong performance in Germanic languages (English: 2.38, German: 1.79, Dutch: 2.31, Danish: 2.18, Swedish: 2.25, Norwegian: 2.12)
- • Moderate performance in Romance languages (French: 3.12, Italian: 3.27, Spanish: 2.40)

This pattern aligns with the model’s training focus, which emphasizes Polish and English, with Germanic languages generally sharing more linguistic features with Polish than Romance languages.

Parameter efficiency: With just 11B parameters, Bielik-11B-v2.3-Instruct achieves a rank that competes with considerably larger models like Gemma-2-9B-it and Meta-Llama-3.1-8B-Instruct. This efficiency highlights the effectiveness of Bielik’s training methodology in producing a compact yet capable multilingual model.

Linguistic proximity advantage: Bielik demonstrates stronger performance in German (1.79) compared to other non-Slavic languages, which may reflect linguistic similarities between Polish and German, both belonging to European language families with shared historical influences.

These EuroEval results complement other benchmark findings, confirming Bielik’s position as a highly efficient multilingual model with competitive performance across European languages, especially considering its parameter count relative to larger competitors. The model’s superior performance in Polish demonstrates the value of specialized language model development targeted at specific languages.

## 6.10 Open LLM Leaderboard

The Open LLM Leaderboard [Beeching et al., 2023b] evaluates models on various English language tasks, providing insights into the model’s performance across different linguistic challenges.

The results from the Open LLM Leaderboard (Table 18) demonstrate the impressive performance of Bielik-11B-v2 across various NLP tasks. With an average score of 65.87, it outperforms Meta-Llama-3-8B (62.62), Mistral-7B-v0.1 (60.97), and Mistral-7B-v0.2 (60.37), while remaining competitive with Qwen models of similar size. Note that this

##### Model AVG arc_challenge hellaswag truthfulqa_mc2 mmlu winogrande gsm8k

Qwen1.5-14B 66.70 56.57 81.08 52.06 69.36 73.48 67.63 Bielik-11B-v2 65.87 60.58 79.84 46.13 63.06 77.82 67.78 Qwen-14B 65.86 58.28 83.99 49.43 67.70 76.80 58.98 Meta-Llama-3-8B 62.62 60.24 82.23 42.93 66.70 78.45 45.19

- Mistral-7B-v0.1 60.97 59.98 83.31 42.15 64.16 78.37 37.83
- Mistral-7B-v0.2 60.37 60.84 83.08 41.76 63.62 78.22 34.72 Bielik-7B-v0.1 49.98 45.22 67.92 47.16 43.20 66.85 29.49

Table 18: Open LLM Leaderboard results for base models

Model AVG arc_challenge hellaswag truthfulqa_mc2 mmlu winogrande gsm8k

SOLAR-10.7B-Instruct-v1.0 74.20 71.08 88.16 71.43 66.21 83.58 64.75 Phi-3-medium-4k-instruct 73.45 67.32 85.76 57.71 77.83 72.69 79.38 Bielik-11B-v2.5-Instruct 71.42 61.95 80.71 53.17 67.44 79.72 85.52

- Bielik-11B-v2.2-Instruct 69.86 59.90 80.16 58.34 64.34 75.30 81.12
- Bielik-11B-v2.3-Instruct 69.82 59.30 80.11 57.42 64.57 76.24 81.27 Bielik-11B-v2.1-Instruct 69.82 59.56 80.20 59.35 64.18 75.06 80.59 openchat-3.5-0106-gemma 69.42 64.68 81.08 54.93 64.69 78.30 72.86 Bielik-11B-v2.0-Instruct 68.04 58.62 78.65 54.65 63.71 76.32 76.27 Meta-Llama-3-8B-Instruct 66.87 60.75 78.55 51.65 67.07 74.51 68.69 Mistral-7B-Instruct-v0.2 65.71 63.14 84.88 68.26 60.78 77.19 40.03 gemma-7b 64.29 61.09 82.47 44.91 66.03 78.45 52.77 Qwen1.5-32B-Chat 62.95 66.04 85.49 66.95 74.99 77.19 7.05 Qwen1.5-14B-Chat 62.27 58.70 82.27 60.36 68.57 73.09 30.63 Qwen1.5-7B-Chat 55.15 55.89 78.56 53.54 61.65 67.72 13.57 Mistral-7B-Instruct-v0.1 54.96 54.52 75.63 56.28 55.38 73.72 14.25 Bielik-7B-Instruct-v0.1 51.26 47.53 68.91 46.18 49.47 65.51 29.95

Table 19: Open LLM Leaderboard results for selected instruction-tuned models

table presents a selected subset of models from the full leaderboard, chosen specifically to provide relevant comparisons to Bielik models.

Key observations:

- 1. Bielik-11B-v2 shows particularly strong performance in mathematical reasoning (gsm8k) with a score of 67.78, comparable to Qwen1.5-14B (67.63) and significantly better than other models.
- 2. It performs well in mmlu (63.06) and winogrande (77.82), demonstrating balanced capabilities across knowledge-intensive and reasoning tasks.
- 3. Compared to its predecessor, Bielik-7B-v0.1, the v2 model shows substantial improvements in all categories, with a remarkable 15.89-point increase in the average score.

The instruction-tuned versions show further improvement over their base counterparts (Table 19). The top performers include SOLAR-10.7B-Instruct-v1.0 (74.20) and Phi-3-medium-4k-instruct (73.45), which lead the pack in overall performance. This table presents a subset of models from the full leaderboard, selected to provide meaningful comparisons with Bielik models.

However, the Bielik models demonstrate impressive results relative to their parameter count. The newest member of the family, Bielik-11B-v2.5-Instruct, achieves the highest score among Bielik models at 71.42, approaching the performance of top models while outperforming many larger competitors. Other variants like Bielik-11B-v2.3-Instruct (69.82) and Bielik-11B-v2.2-Instruct (69.86) achieve scores that are competitive with models like openchat-3.5-0106-gemma

- (69.42) while substantially outperforming models of similar or even larger size like Mistral-7B-Instruct-v0.2 (65.71) and Qwen1.5-32B-Chat (62.95).

Particularly noteworthy is Bielik’s exceptional performance in mathematical reasoning (gsm8k), where Bielik-11B-v2.5Instruct excels with a score of 85.52, even surpassing the top overall performers SOLAR-10.7B-Instruct-v1.0 (64.75) and Phi-3-medium-4k-instruct (79.38). This demonstrates Bielik’s specialized strength in this challenging domain.

Model Avg (%) IFEval BBH MATH GPQA MuSR MMLU-PRO Qwen2.5-72B-Instruct 47.98 86.38 61.87 59.82 16.67 11.74 51.40 Qwen2.5-32B-Instruct 46.60 83.46 56.49 62.54 11.74 13.50 51.85 Mistral-Large-Instruct-2411† 46.52 84.01 52.74 49.55 24.94 17.22 50.69 Llama-3.3-70B-Instruct 44.85 89.98 56.56 48.34 10.51 15.57 48.13 Qwen2.5-14B-Instruct 41.31 81.58 48.36 54.76 9.62 10.16 43.38 Qwen2.5-7B-Instruct 35.20 75.85 34.89 50.00 5.48 8.45 36.52 Phi-3-medium-4k-instruct 33.10 64.23 49.38 19.56 11.52 13.05 40.84 Mistral-Small-Instruct-2409 29.92 62.83 40.56 20.39 11.07 10.23 34.43 Bielik-11B-v2.3-Instruct 28.33 55.83 38.06 20.85 12.08 16.01 27.16 Bielik-11B-v2.2-Instruct 27.98 55.52 36.96 26.81 10.85 10.11 27.63 Phi-3-mini-4k-instruct 27.56 54.77 36.56 16.39 10.96 13.12 33.58 Bielik-11B-v2.1-Instruct 27.20 50.90 36.29 26.66 11.63 10.52 27.18 Qwen2.5-3B-Instruct 27.16 64.75 25.80 36.78 3.02 7.57 25.05 Mistral-Nemo-Instruct-2407 24.67 63.80 29.68 12.69 5.37 8.48 27.97 Bielik-11B-v2.0-Instruct 24.66 52.52 33.77 11.86 8.95 14.74 26.12 Meta-Llama-3-8B-Instruct 23.91 74.08 28.24 8.69 1.23 1.60 29.60 Mixtral-8x7B-Instruct-v0.1 23.82 55.99 29.74 9.14 7.05 11.07 29.91 Bielik-11B-v2 15.99 23.81 27.82 7.85 5.15 7.56 23.75 PLLuM-12B-chat 15.35 32.14 21.32 1.81 1.34 14.67 20.80 PLLuM-12B-base 14.67 28.21 21.24 2.87 5.37 10.98 19.34 Llama-PLLuM-8B-chat 14.61 35.15 16.28 3.40 1.90 11.86 19.10 PLLuM-12B-nc-chat† 14.60 28.34 23.01 1.21 4.36 12.92 17.75 PLLuM-12B-nc-base† 11.42 24.05 19.39 2.19 2.68 2.90 17.32 †Models with a non-commercial license.

Table 20: Open LLM Leaderboard v2 results across different evaluation tasks

## 6.11 Open LLM Leaderboard v2

The Open LLM Leaderboard v2 [Fourrier et al., 2024] is an updated benchmark suite designed to evaluate large language models across a more diverse and challenging set of tasks. Unlike the original leaderboard that focused primarily on multiple-choice questions, v2 incorporates more complex reasoning, instruction following, and specialized knowledge evaluations:

- • IFEval - Tests a model’s ability to follow explicit instructions precisely
- • BBH (Big Bench Hard) - Includes 23 challenging tasks covering algorithmic reasoning, language understanding, and advanced world knowledge
- • MATH Lvl 5 - Features high-school level competition mathematics problems requiring sophisticated problemsolving skills
- • GPQA - Contains graduate-level questions crafted by PhD-level domain experts across scientific disciplines
- • MuSR - Presents multistep soft reasoning problems requiring integration of reasoning abilities with long-range context understanding
- • MMLU-PRO - Offers a refined version of MMLU with 10 choices instead of 4, requiring more sophisticated reasoning and deeper domain knowledge

As shown in Table 20, Bielik models demonstrate competitive performance in this more challenging benchmark suite. The best-performing variant, Bielik-11B-v2.3-Instruct, achieves an average score of 28.33%, placing it in the same tier

- as models like Mistral-Small-Instruct-2409 (29.92%) and Phi-3-mini-4k-instruct (27.56%). Several observations can be made from these results:

- 1. Bielik models excel particularly in MuSR tasks, with Bielik-11B-v2.3-Instruct scoring 16.01% - higher than many larger models including Qwen2.5-72B-Instruct (11.74%) and Qwen2.5-14B-Instruct (10.16%). This suggests strong performance in complex reasoning scenarios requiring integration with long-range context.
- 2. In GPQA (graduate-level questions), Bielik-11B-v2.3-Instruct (12.08%) outperforms several larger models, including Qwen2.5-14B-Instruct (9.62%) and is competitive with Qwen2.5-32B-Instruct (11.74%), demonstrating efficient knowledge encoding relative to its parameter count.

##### Model MixEval-Hard MixEval

Qwen1.5-72B-Chat 48.3 84.1 LLaMA-3-8B-Instruct 45.6 75.0

- Bielik-11B-v2.1-Instruct 45.0 74.6 Qwen1.5-32B-Chat 43.3 81.0 Bielik-11B-v2.3-Instruct 43.2 73.0 Bielik-11B-v2.0-Instruct 40.2 72.1
- Bielik-11B-v2.2-Instruct 39.7 72.4 Bielik-11B-v2.5-Instruct 39.0 68.6 Mistral-7B-Instruct-v0.2 36.2 70.0

Table 21: MixEval results

- 3. The significant gap between Bielik models and Polish-focused PLLuM models is notable, with even the base Bielik-11B-v2 (15.99%) outperforming all PLLuM variants, and the best Bielik-11B-v2.3-Instruct (28.33%) nearly doubling the performance of the best PLLuM model (15.35%).
- 4. The consistent progression across versions (v2.0 to v2.3) demonstrates the effectiveness of the training methodology, with each version showing improvements in most categories.

These results confirm that Bielik models not only perform well in classification and multiple-choice tasks (as shown in the original Open LLM Leaderboard) but also in more complex reasoning and instruction-following scenarios.

## 6.12 MixEval

MixEval [Ni et al., 2024] is a ground-truth-based English benchmark designed to evaluate Large Language Models (LLMs) efficiently and effectively. Key features of MixEval include:

- 1. Derived from off-the-shelf benchmark mixtures
- 2. Highly capable model ranking with a 0.96 correlation to Chatbot Arena
- 3. Local and quick execution, requiring only 6% of the time and cost compared to running MMLU

This benchmark provides a robust and time-efficient method for assessing LLM performance, making it a valuable tool for ongoing model evaluation and comparison.

The results (Table 21) show that all Bielik-11B models perform competitively on the MixEval benchmark. Bielik-11Bv2.1-Instruct achieves the best performance among the Bielik models with scores of 74.6 on MixEval and 45.0 on MixEval-Hard, comparable to LLaMA-3-8B-Instruct. Bielik-11B-v2.3-Instruct follows with scores of 73.0 and 43.2 on MixEval and MixEval-Hard respectively. The newest Bielik-11B-v2.5-Instruct shows slightly lower scores (68.6 and 39.0), reflecting its optimization for function-calling and other specialized capabilities that may trade off performance on these specific benchmarks. All Bielik variants significantly outperform Mistral-7B-Instruct-v0.2 on both metrics, demonstrating their improved capabilities despite being based on a similar architecture.

## 6.13 Berkeley Function-Calling Leaderboard

The Berkeley Function-Calling Leaderboard (BFCL) Yan et al. [2024] evaluates language models’ ability to call functions (tools) accurately using real-world data. This benchmark is particularly important for assessing how well models can interface with external systems and APIs, a crucial capability for practical applications of LLMs in software development, data analysis, and task automation.

The benchmark uses Abstract Syntax Tree (AST) evaluation metrics to assess function call accuracy across several categories:

- • Expert Curated (Non-live) dataset: Static examples curated by experts to evaluate function calling on controlled scenarios
- • User Contributed (Live) dataset: Dynamic, interactive scenarios submitted by users to test real-world function calling patterns
- • Multi-turn interactions: Testing the model’s ability to maintain context across conversation turns

Model Non-Live Non-Live Non-Live Non-Live Live Live Live Live Parallel

Python Multiple Parallel Parallel Simple Multiple Parallel Multiple Simple AST AST AST Multiple AST AST AST AST AST

Open-Mistral-Nemo-2407 (Prompt) 92.00% 93.50% 89.50% 84.50% 77.91% 74.45% 87.50% 66.67% Gemma-3-12b-it (Prompt) 94.00% 95.00% 90.00% 73.00% 84.88% 70.85% 87.50% 62.50% Open-Mistral-Nemo-2407 (FC) 91.25% 93.50% 85.50% 85.00% 77.13% 69.61% 75.00% 70.83% Bielik-11B-v2.5-Instruct (FC) 95.00% 97.50% 87.50% 87.00% 77.13% 77.21% 43.75% 66.67% Qwen2.5-3B-Instruct (Prompt) 91.50% 90.50% 79.50% 79.00% 69.77% 66.48% 56.25% 62.50% Qwen2.5-3B-Instruct (FC) 96.00% 92.00% 73.50% 76.50% 74.03% 72.08% 62.50% 45.83% Qwen2.5-1.5B-Instruct (FC) 92.25% 87.00% 81.50% 75.50% 74.03% 66.10% 50.00% 45.83% Qwen2.5-1.5B-Instruct (Prompt) 89.00% 86.00% 70.00% 66.50% 70.54% 59.26% 56.25% 41.67% Bielik-11B-v2.3-Instruct (Prompt) 87.50% 93.50% 47.00% 50.00% 72.87% 69.71% 43.75% 54.17%

Table 22: Detailed breakdown of Berkeley Function-Calling Leaderboard subtask performance across models. Bielik models show competitive performance on numerous subtasks, with particularly strong results in Non-Live Python Simple AST, Non-Live Multiple AST categories, and consistent performance on Live Simple and Multiple AST tasks.

- • Relevance detection: Assessing whether models correctly invoke functions when appropriate. These scenarios present at least one relevant function that should be called, though there could be multiple valid ways to invoke it. Models are expected to output some function call relevant to the user query, without necessarily checking for parameter correctness
- • Irrelevance detection: Evaluating if models correctly abstain from calling functions when unnecessary. In these scenarios, none of the available functions are relevant to the user query. Models are expected to either explain why no function is applicable or simply respond without making a function call

Detailed subtask performance: The fine-grained subtask results (Table 22) show Bielik models’ performance across specific function-calling tasks:

- • Strong performance in Non-Live Python Simple AST (87.50-95.00%) and Non-Live Multiple AST (93.5097.50%)
- • Good performance on Live Simple AST (72.87-77.13%) and Live Multiple AST (69.71-77.21%)
- • Opportunities for improvement in parallel AST tasks, particularly Live Parallel AST (43.75%)

Performance context: Despite being primarily optimized for Polish language understanding rather than function calling, Bielik demonstrates foundational capabilities in parsing and executing function calls through natural language. The table includes results for both Bielik-11B-v2.3-Instruct using the prompt-based approach and Bielik-11B-v2.5Instruct with native function-calling capabilities (FC), showing our progress in this area. It’s important to note that Bielik has been evaluated only on a subset of all BFCL tasks, as certain task types have not yet been addressed in training. Work is ongoing to expand Bielik’s capabilities to cover the complete range of function-calling tasks. The implementation of native function-calling capabilities in v2.5 represents an important step forward, though further improvements are needed particularly for handling parallel AST tasks where performance is currently lower compared to other subtasks.

# 7 Limitations and Biases

The Bielik v2 series of models may produce factually incorrect output and should not be relied upon to generate completely accurate information in all contexts. These models were trained on diverse public datasets, and despite our extensive efforts to clean and filter the training data, they may occasionally generate content that is biased, offensive, or factually inaccurate. Users should apply appropriate caution and verification when deploying these models, particularly in sensitive or high-stakes applications.

# 8 Conclusion

In this technical report, we have introduced the Bielik 11B v2 series of models (Table 23), specifically designed for Polish language processing. These models represent a significant advancement in language model capabilities for Polish, demonstrating competitive performance against much larger models across various benchmarks.

Model name Training OpenLLM PL MT-Bench OpenLLM v2 OpenLLM Release process avg (5-shot) score avg avg date

Mistral-7B-v0.2 pre-training 38.81 - - 60.78 Mar 2024 Bielik-11B-v2 cont pre-training 58.14 - 15.99 65.87 Aug 2024

- Bielik-11B-v2.0-Instruct SFT 64.98 7.56 24.66 68.04 Aug 2024
- Bielik-11B-v2.1-Instruct SFT, DPOP 65.45 8.00 27.20 69.82 Aug 2024
- Bielik-11B-v2.2-Instruct SFT, DPOP 65.57 8.12 27.98 69.86 Aug 2024
- Bielik-11B-v2.3-Instruct SFT, DPOP, MERGE 65.71 8.56 28.33 69.82 Aug 2024 Bielik-11B-v2.5-Insturct SFT, DPOP, MERGE 63.95 - - 71.42 May 2025

Table 23: List of released v2 family models with source model for base line reference

The Bielik 11B v2 models incorporate several technical innovations, including Weighted Instruction Cross-Entropy Loss and Adaptive Learning Rate, which have proven effective for balancing learning across different instruction types and optimizing the training process. By building upon the Mistral 7B architecture and applying depth up-scaling, we’ve created models that achieve state-of-the-art results for their parameter size.

Our evaluation across multiple benchmarks demonstrates that Bielik models excel in Polish language tasks while maintaining strong performance in cross-lingual scenarios. The models show particularly impressive results in the Open PL LLM Leaderboard, Polish MT-Bench, and Polish Linguistic and Cultural Competency Benchmark (PLCC), often outperforming much larger specialized Polish language models.

The Bielik models also demonstrate remarkable parameter efficiency, achieving comparable results to models with significantly more parameters. This efficiency, combined with our comprehensive quantization options, makes these models practical for deployment across a wide range of hardware configurations.

Future work will focus on further enhancing the models’ capabilities in specialized domains, improving cross-lingual transfer to other Slavic languages, and expanding their function-calling abilities. The Bielik 11B v2 series represents an important step forward in democratizing access to high-quality language models for Polish, providing powerful tools for diverse linguistic applications.

# Acknowledgements

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2024/017214 and PLG/2025/018338.

The model could not have been created without the commitment and work of the entire SpeakLeash team, whose contribution is invaluable. Thanks to the hard work of many individuals, it was possible to gather a large amount of content in Polish and establish collaboration between the open-science SpeakLeash project and the HPC center: ACK Cyfronet AGH. Individuals who contributed to the creation of the model through their commitment to the open-science SpeakLeash project: Sebastian Kondracki, Marek Magry´s, Szymon Mazurek, Mieszko Cholewa, Igor Ciuciura, Paweł Kiszczak, Szymon Baczy´nski, Jacek Chwiła, Maria Filipkowska, Jan Maria Kowalski, Dominika Basaj, Kuba Sołtys, Karol Jezierski, Kacper Milan, Jan Sowa, Len Krawczyk, Marta Seidler, Agnieszka Ratajska, Krzysztof Koziarek, Szymon Pepli´nski, Zuzanna Dabi´c, Filip Bogacz, Agnieszka Kosiak, Izabela Babis, Nina Babis, and many other wonderful researchers and enthusiasts of the AI world.

# A European LLM Leaderboard - Additional Results

This appendix presents detailed results from the European LLM Leaderboard for German and Czech languages, complementing the Polish language results presented in the main text.

## A.1 German Language Results

In German language evaluation (Table 24), Bielik-11B-v2.3-Instruct shows strong performance with an average score of 0.62, positioning it in the upper range among models with similar parameter counts. Despite being primarily trained on Polish and English data, the model demonstrates impressive cross-lingual transfer capabilities to German. It achieves particularly strong results in mathematical reasoning (GSM8K: 0.65) and factual accuracy (TruthfulQA: 0.59), outperforming several models with similar or larger parameter counts such as Mistral-7B-Instruct-v0.3 (0.55),

Meta-Llama-3.1-70B-Instruct 70.6 0.71 0.73 0.74 0.75 0.78 0.57 Gemma-2-27b-Instruct 27.0 0.71 0.77 0.78 0.72 0.69 0.59 Phi-3-medium-14B-4k-Instruct 14.0 0.69 0.71 0.78 0.69 0.72 0.56 Phi-3-medium-14B-128k-Instruct 14.0 0.67 0.70 0.72 0.68 0.71 0.55 Mixtral-8x7B-Instruct-v0.1 46.7 0.67 0.71 0.58 0.75 0.67 0.65 Mistral-Nemo-Instruct-12.2B 12.2 0.65 0.69 0.65 0.70 0.64 0.56 c4ai-command-r-35B-v01 35.0 0.63 0.69 0.52 0.75 0.61 0.55 Mixtral-8x7B-v0.1 46.7 0.62 0.70 0.51 0.74 0.68 0.50 Bielik-11B-v2.3-Instruct 11.2 0.62 0.64 0.65 0.62 0.60 0.59 Mistral-Nemo-Base-12.2B 12.0 0.61 0.68 0.52 0.72 0.65 0.50 Gemma-2-9b-Instruct 9.2 0.60 0.71 0.49 0.63 0.61 0.58 Meta-Llama-3-8B-Instruct 8.0 0.59 0.6 0.63 0.58 0.58 0.55 Meta-Llama-3.1-8B-Instruct 8.0 0.59 0.61 0.58 0.62 0.62 0.52 EuroLLM-9B-Instruct 9.2 0.58 0.69 0.47 0.69 0.58 0.50 Mistral-7B-Instruct-v0.3 7.2 0.55 0.62 0.39 0.61 0.54 0.57 Mistral-7B-Instruct-v0.2 7.2 0.54 0.61 0.32 0.61 0.53 0.64 Meta-Llama-3.1-8B 8.0 0.54 0.59 0.4 0.63 0.59 0.5 Teuken-7B-sigma-v05 7 0.53 0.62 0.31 0.69 0.47 0.58 Pharia-1-LLM-7B-control-aligned 7.5 0.48 0.62 0.16 0.67 0.48 0.50 Vicuna-13b-v1.5 13.0 0.48 0.55 0.26 0.59 0.50 0.52 Teuken-7B-instruct-research-v0.4 7.5 0.48 0.62 0.13 0.64 0.45 0.54 Teuken-7B-instruct-commercial-v0.4 7.5 0.46 0.61 0.09 0.64 0.45 0.50 Salamandra-7b-instruct 7.8 0.45 0.61 0.09 0.64 0.47 0.45 Pharia-1-LLM-7B-control 7.5 0.45 0.62 0.06 0.66 0.45 0.46 Teuken-7B-base-v0.55 7.5 0.44 0.57 0.09 0.64 0.45 0.45 Salamandra-7b 7.8 0.42 0.60 0.00 0.64 0.43 0.45 EuroLLM-1.7B-Instruct 1.7 0.36 0.50 0.09 0.49 0.27 0.45 EuroLLM-1.7B 1.7 0.34 0.48 0.04 0.48 0.26 0.43

Table 24: European LLM Leaderboard results for German language.

Meta-Llama-3.1-8B (0.54), and even Vicuna-13b-v1.5 (0.48). This performance indicates Bielik’s effectiveness in transferring its capabilities to Germanic languages despite its Slavic-focused training.

## A.2 Czech Language Results

In Czech language evaluation (Table 25), Bielik-11B-v2.3-Instruct achieves a strong average score of 0.60, placing it fifth overall and outperforming all models with comparable parameter counts. This demonstrates exceptional cross-lingual transfer to Czech, another West Slavic language. The model’s performance is particularly impressive in mathematical problem-solving with a GSM8K score of 0.60, surpassing much larger models including Mixtral-8x7B-Instruct-v0.1 (0.50) and c4ai-command-r-35B-v01 (0.48).

Bielik significantly outperforms all 7B and smaller models, including Mistral-7B-Instruct-v0.3 (0.50), specialized European models like Teuken and Salamandra, and even Vicuna-13b-v1.5 (0.43) despite its larger parameter count. The factual accuracy score (TruthfulQA: 0.58) matches that of Meta-Llama-3.1-70B-Instruct despite Bielik having only about 15% of its parameters. These results highlight Bielik’s efficient architecture and the benefits of its Polish pretraining for cross-lingual transfer to related Slavic languages.

## A.3 FLORES200 Translation Benchmark Results

The FLORES200 translation results for Bielik-11B-v2.3-Instruct (Table 26) show asymmetric performance across language pairs with Polish. The model achieves its best results with English-Polish (21.93/28.32) and demonstrates strong performance with linguistically similar West Slavic languages like Czech (19.30/14.58). It also performs well with major European languages such as Portuguese (19.10/19.76), French (18.97/19.06), and German (19.18/14.93). However, performance drops significantly with Baltic and Finno-Ugric languages, suggesting limitations in crosslinguistic transfer to more distant language families.

Meta-Llama-3.1-70B-Instruct 70.6 0.71 0.72 0.75 0.72 0.77 0.58 Gemma-2-27b-Instruct 27.4 0.70 0.75 0.75 0.70 0.68 0.61 Mixtral-8x7B-Instruct-v0.1 46.7 0.62 0.68 0.50 0.66 0.63 0.63 c4ai-command-r-35B-v01 35.0 0.60 0.67 0.48 0.71 0.60 0.55 Bielik-11B-v2.3-Instruct 11.2 0.60 0.63 0.60 0.59 0.59 0.58 Mistral-Nemo-Instruct-12.2B 12.2 0.59 0.62 0.55 0.59 0.59 0.59 Mixtral-8x7B-v0.1 46.7 0.58 0.66 0.43 0.65 0.65 0.52 Gemma-2-9b-Instruct 9.0 0.58 0.67 0.46 0.59 0.59 0.60 Meta-Llama-3.1-8B-Instruct 8.0 0.57 0.58 0.58 0.57 0.58 0.53 EuroLLM-9B-Instruct 9.0 0.57 0.68 0.45 0.67 0.58 0.50 Mistral-7B-Instruct-v0.3 7.0 0.50 0.58 0.32 0.54 0.51 0.56 Teuken-7B-sigma-v05 7.0 0.50 0.59 0.26 0.65 0.45 0.56 Mistral-7B-Instruct-v0.2 7.0 0.50 0.57 0.28 0.55 0.50 0.59 Teuken-7B-instruct-research-v0.4 7.0 0.45 0.59 0.11 0.62 0.42 0.53 Salamandra-7b-instruct 7.0 0.45 0.59 0.11 0.62 0.46 0.47 Teuken-7B-instruct-commercial-v0.4 7.0 0.44 0.56 0.11 0.61 0.43 0.51 Vicuna-13b-v1.5 13.0 0.43 0.48 0.20 0.51 0.46 0.49 Salamandra-7b 7.0 0.41 0.59 0.00 0.63 0.41 0.43 Occiglot-7b-eu5 7.0 0.40 0.46 0.18 0.48 0.44 0.45 Occiglot-7b-eu5-Instruct 7.0 0.40 0.48 0.13 0.48 0.43 0.46 EuroLLM-1.7B-Instruct 2.0 0.35 0.49 0.08 0.48 0.26 0.44 EuroLLM-1.7B 2.0 0.33 0.46 0.03 0.48 0.25 0.43 Pharia-1-LLM-7B-control 7.0 0.27 0.27 0.01 0.31 0.30 0.44 Pharia-1-LLM-7B-control-aligned 7.0 0.27 0.27 0.02 0.31 0.31 0.43

Table 25: European LLM Leaderboard results for Czech language.

Language to Polish from Polish Bulgarian 17.83 6.81 Czech 19.30 14.58 Danish 17.81 15.24 German 19.18 14.93 Greek 5.49 0.50 English 21.93 28.32 Estonian 6.13 1.53 Finnish 12.76 4.14 French 18.97 19.06 Hungarian 15.59 7.36 Italian 16.54 13.07 Lithuanian 7.99 1.28 Latvian 5.85 0.88 Dutch 14.76 13.07 Portuguese 19.10 19.76 Romanian 18.30 15.47 Slovak 17.65 6.60 Slovenian 17.37 12.53 Spanish 16.17 17.50 Swedish 17.58 14.51 Average 15.31 11.36

Table 26: FLORES200 translation benchmark BLEU scores for Bielik-11B-v2.3-Instruct

# B BenCzechMark Benchmark Results

BenCzechMark (BCM) is a comprehensive Czech-centric benchmark for Large Language Models introduced by Fajcik et al. [2024]. It encompasses 50 challenging tasks across 8 categories, evaluating models on diverse linguistic capabilities in native Czech. The benchmark’s scoring system employs a Duel Win Score (DWS) mechanism that statistically compares each model against others, providing a robust evaluation methodology.

Model Params Avg Czech Math Factual Lang. NER NLI Reading Sentiment (B) (%) Lang. Reason. Know. Model. Comp.

Meta-Llama-3.1-405B-Instruct 406.0 90.2 98.5 91.7 88.6 93.3 97.2 89.1 93.2 70.4 Qwen2.5-72B 72.7 85.7 75.3 94.9 83.6 83.6 96.3 84.0 92.6 75.5 Meta-Llama-3.3-70B-Instruct 70.6 80.0 86.1 75.9 78.8 69.9 92.6 82.8 79.0 75.0 Qwen2.5-72B-Instruct 72.7 78.9 80.2 94.0 80.7 69.0 69.4 86.1 72.8 79.2 Meta-Llama-3.1-70B-Instruct 70.6 78.6 80.2 60.2 79.1 81.7 88.9 84.6 78.4 75.9 Meta-Llama-3.1-70B 70.6 76.5 70.1 55.1 75.9 90.5 97.2 68.8 83.3 71.3 Qwen2.5-32B-Instruct 32.8 73.9 74.7 91.7 68.0 62.0 65.7 84.0 68.5 76.9 Mistral-Large-Instruct-2411 123.0 73.8 85.2 83.3 86.5 6.5 82.4 83.4 82.1 81.0 Qwen2-72B-Instruct 72.7 72.5 72.2 83.3 78.6 60.6 80.6 66.1 74.1 64.4 Mixtral-8x22B-Instruct-v0.1 141.0 71.5 69.1 59.3 70.6 83.3 70.4 73.5 67.3 78.2 Bielik-11B-v2.3-Instruct 11.2 49.5 46.9 31.0 48.7 25.9 54.6 67.1 64.2 57.4 Google/Gemma-2-27b 27.2 46.9 51.2 52.3 43.7 63.4 2.8 53.0 38.9 69.9 Meta-Llama-3-8B-Instruct 8.0 44.0 45.4 13.0 35.2 33.6 63.0 48.4 56.2 57.4 EuroLLM-9B-Instruct 9.2 34.7 43.8 11.6 36.2 85.2 14.8 29.3 29.6 27.3 EuroLLM-9B 9.2 31.4 32.4 8.8 36.0 88.7 16.7 24.5 23.5 20.4 Mistral-7B-Instruct-v0.3 7.3 32.6 29.0 4.2 33.1 37.3 38.9 39.9 35.8 42.6 CohereForAI/aya-23-8B 8.0 37.4 27.2 5.6 31.0 53.5 40.7 34.0 45.1 62.0 salamandra-7b-instruct 7.8 29.4 24.1 2.8 25.1 70.1 11.1 29.5 16.7 56.0 csmpt7b 6.7 16.0 4.6 0.0 2.6 80.6 9.3 18.5 3.1 9.3 CSTinyLlama-1.2B 1.2 11.3 1.2 1.4 2.9 76.2 4.6 2.8 0.0 1.4

- Table 27: BenCzechMark results showing performance across all 8 categories. The table includes top-performing models, various mid-sized models (including Bielik-11B-v2.3-Instruct ranked 21st overall), and specialized Czech models for comparison. All values are percentages, with higher scores indicating better performance.

Bielik-11B-v2.3-Instruct achieves an average score of 49.5% on the BenCzechMark leaderboard (Table 27), placing it

- at position 21 among evaluated models. This is a respectable performance for a model of its size (11.2B parameters) when compared to much larger models like Llama-3.1-405B-Instruct (406B) and Qwen2.5-72B (72.7B) that dominate the top positions.

Category-specific performance: Bielik shows varying capabilities across different task categories:

- • Strongest performance in Natural Language Inference (67.1%) and Reading Comprehension (64.2%), demonstrating good reasoning abilities in Czech
- • Moderate performance in Sentiment Analysis (57.4%) and NER (54.6%), showing capability in understanding emotional context and entity recognition in Czech text
- • Lower scores in Czech Language Understanding (46.9%), Factual Knowledge (48.7%), and Math Reasoning (31.0%), indicating areas for improvement in future versions

Comparative analysis: Despite being primarily designed for Polish language processing, Bielik demonstrates reasonable cross-lingual transfer to Czech:

- • Outperforms several similarly-sized models in Reading Comprehension and NLI tasks
- • Shows stronger reasoning capabilities in NLI compared to some larger models
- • Significantly outperforms native Czech models (BUT-FIT/csmpt7b and BUT-FIT/CSTinyLlama-1.2B) in overall performance and most categories, despite these models having been specifically trained on Czech data
- • Demonstrates the value of instruction tuning and Slavic language pretraining for cross-lingual transfer between related languages

These results highlight Bielik’s ability to generalize across Slavic languages, while also identifying specific areas where future versions could be enhanced to better support Czech language capabilities. The model’s performance is particularly notable given that it was not specifically optimized for Czech language tasks, unlike the specialized Czech models in the benchmark that excel in language modeling but lag significantly in reasoning-based tasks.

Model Avg ENEM BLUEX OAB ASSIN2 ASSIN2 FAQUAD HateBR PT Hate tweetSent Exams RTE STS NLI Speech BR

- Bielik-11B-v2.1-Instruct 73.45 72.29 62.59 48.56 93.50 79.82 82.73 79.58 73.97 67.99
- Bielik-11B-v2.2-Instruct 73.45 72.57 62.73 49.02 93.67 80.50 80.32 79.60 73.75 68.85
- Bielik-11B-v2.3-Instruct 73.08 71.73 62.31 48.66 93.29 80.49 80.63 79.58 73.13 67.89 Bielik-11B-v2.0-Instruct 72.57 70.75 61.47 47.84 92.59 80.14 83.75 77.88 72.06 66.61 Bielik-11B-v2 68.68 68.86 58.55 48.70 92.76 77.99 61.37 74.05 67.26 68.58 Mistral-Nemo-Instruct-2407 73.76 72.08 61.47 53.80 93.22 75.18 82.34 77.96 74.30 73.48

- Table 28: Performance of Bielik models on Portuguese language benchmarks (higher is better). The table shows scores for: ENEM (Brazilian National High School Exam), BLUEX, OAB Exams (Brazilian Bar Association), ASSIN2 RTE (Recognizing Textual Entailment), ASSIN2 STS (Semantic Textual Similarity), FAQUAD NLI (Natural Language Inference), HateBR, PT Hate Speech, and tweetSentBR.

# C Portuguese Benchmark Results

The Open PT LLM Leaderboard Garcia [2024] aims to provide a comprehensive benchmark for the evaluation of Large Language Models (LLMs) in the Portuguese language across a variety of tasks and datasets. The benchmark evaluates models on 9 key tasks using the Eleuther AI Language Model Evaluation Harness, a unified framework to test generative language models on a large number of different evaluation tasks. As shown in Table 28, Bielik models perform competitively on Portuguese language tasks despite being primarily optimized for Polish, demonstrating strong cross-lingual transfer abilities to other Romance languages.

# References

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, Singapore, 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.emnlpmain.298. URL https://aclanthology.org/2023.emnlp-main.298.

Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 933–941. PMLR, 2017. URL http://proceedings.mlr.press/v70/dauphin17a.html.

Noam Shazeer. Glu variants improve transformer, 2020. URL https://arxiv.org/abs/2002.05202.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. ISSN 09252312. doi:https://doi.org/10.1016/j.neucom.2023.127063. URL https://www.sciencedirect.com/science/ article/pii/S0925231223011864.

Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, and David Z. Pan. Pre-rmsnorm and pre-crmsnorm transformers: Equivalent and efficient pre-ln transformers. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023b. URL https://arxiv.org/abs/2310.06825.

Sanghoon Kim, Dahyun Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. SOLAR 10.7B: Scaling large language models with simple yet effective depth up-scaling. In Yi Yang, Aida Davani, Avi Sil, and Anoop Kumar, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 23–35, Mexico City, Mexico, 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.naacl-industry.3.

SpeakLeash Team. Speakleash a.k.a spichlerz!, 2024. URL https://www.speakleash.org. Accessed: 2024-09-30. Dingcheng Li, Zheng Chen, Eunah Cho, Jie Hao, Xiaohu Liu, Fan Xing, Chenlei Guo, and Yang Liu. Overcoming

catastrophic forgetting during domain adaptation of seq2seq language generation. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5441–5454, Seattle, United States, 2022. Association for Computational Linguistics. doi:10.18653/v1/2022.naacl-main.398. URL https://aclanthology.org/2022.naacl-main.398.

Oleksiy Ostapenko, Timothee Lesort, Pau Rodriguez, Md Rifat Arefin, Arthur Douillard, Irina Rish, and Laurent Charlin. Continual learning with foundation models: An empirical study of latent replay. In Sarath Chandar, Razvan Pascanu, and Doina Precup, editors, Proceedings of The 1st Conference on Lifelong Learning Agents, volume 199 of Proceedings of Machine Learning Research, pages 60–91. PMLR, 2022. URL https://proceedings.mlr.

### press/v199/ostapenko22a.html.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. Simple and scalable strategies to continually pre-train large language models, 2024. URL https://arxiv.org/abs/2403.08763.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023. URL https: //huggingface.co/datasets/cerebras/SlimPajama-627B.

Inez Okulska, Daria Stetsenko, Anna Kołos, Agnieszka Karli´nska, Kinga Gła˛bi´nska, and Adam Nowakowski. Stylometrix: An open-source multilingual tool for representing stylometric vectors, 2023. URL https://arxiv.org/ abs/2309.12810.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. URL https://arxiv.org/abs/2401.04088.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/hash/ a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive, 2024. URL https://arxiv.org/abs/2402.13228.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. LIMA: less is more for alignment. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_ files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html.

Zhengxiang Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, and Aldo Lipani. Instruction tuning with loss over instructions. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC,

Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/ 7ffb43adf37b3eeaba559098bc084cc6-Abstract-Conference.html.

Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoz´dziej, and Remigiusz Kinas. Bielik 7b v0.1: A polish language model – development, insights, and evaluation, 2024. URL https://arxiv.org/abs/2410.18565.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023b. URL http://papers.nips.cc/paper_files/paper/2023/hash/ a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv preprint, abs/1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. ArXiv preprint, abs/2402.01306, 2024. URL https://arxiv.org/abs/2402.01306.

Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic preference optimization without reference model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11170–11189, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.emnlp-main.626. URL https://aclanthology.org/2024.emnlp-main.626/.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/ e099c1c9699814af0be873a175361713-Abstract-Conference.html.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 23965–23998. PMLR, 2022. URL https://proceedings.mlr.press/v162/ wortsman22a.html.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html.

Dong-Hwan Jang, Sangdoo Yun, and Dongyoon Han. Model stock: All we need is just a few fine-tuned models, 2024. URL https://arxiv.org/abs/2403.19522.

Charles H. Martin and Michael W. Mahoney. Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning. J. Mach. Learn. Res., 22:165:1–165:73, 2021. URL http: //jmlr.org/papers/v22/20-410.html.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. ArXiv preprint, abs/2407.10671, 2024. URL https://arxiv.org/abs/2407.10671.

Qwen Team. Qwen2.5: A party of foundation models, 2024a. URL https://qwenlm.github.io/blog/qwen2.5/. Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha

Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte

Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel

Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Kelly Marchisio, Sebastian Ruder, Acyr Locatelli, Julia Kreutzer, Nick Frosst, Phil Blunsom, Marzieh Fadaee, Ahmet Üstün, and Sara Hooker. Aya 23: Open weight releases to further multilingual progress, 2024.

Gemma Team. Gemma. 2024b. doi:10.34740/KAGGLE/M/3301. URL https://www.kaggle.com/m/3301. Gemma Team. Gemma 3. 2025. URL https://goo.gle/Gemma3Report.

Pedro Henrique Martins, Patrick Fernandes, João Alves, Nuno M. Guerreiro, Ricardo Rei, Duarte M. Alves, José Pombal, Amin Farajian, Manuel Faysse, Mateusz Klimaszewski, Pierre Colombo, Barry Haddow, José G. C. de Souza, Alexandra Birch, and André F. T. Martins. Eurollm: Multilingual language models for europe, 2024. URL https://arxiv.org/abs/2409.16235.

Mehdi Ali, Michael Fromm, Klaudia Thellmann, Jan Ebert, Alexander Arno Weber, Richard Rutmann, Charvi Jain, Max Lübbering, Daniel Steinigen, Johannes Leveling, Katrin Klug, Jasper Schulze Buschhoff, Lena Jurkschat, Hammam Abdelwahab, Benny Jörg Stein, Karl-Heinz Sylla, Pavel Denisov, Nicolo’ Brandizzi, Qasid Saleem, Anirban Bhowmick, Lennard Helmer, Chelsea John, Pedro Ortiz Suarez, Malte Ostendorff, Alex Jude, Lalith Manjunath, Samuel Weinbach, Carolin Penke, Oleg Filatov, Shima Asaadi, Fabio Barth, Rafet Sifa, Fabian Küch, Andreas Herten, René Jäkel, Georg Rehm, Stefan Kesselheim, Joachim Köhler, and Nicolas Flores-Herr. Teuken-7b-base & teuken-7b-instruct: Towards european llms, 2024. URL https://arxiv.org/abs/2410.03730.

Aitor Gonzalez-Agirre, Marc Pàmies, Joan Llop, Irene Baucells, Severino Da Dalt, Daniel Tamayo, José Javier Saiz, Ferran Espuña, Jaume Prats, Javier Aula-Blasco, Mario Mina, Adrián Rubio, Alexander Shvets, Anna Sallés, Iñaki Lacunza, Iñigo Pikabea, Jorge Palomar, Júlia Falcão, Lucía Tormo, Luis Vasquez-Reina, Montserrat Marimon, Valle Ruíz-Fernández, and Marta Villegas. Salamandra technical report, 2025. URL https://arxiv.org/abs/2502.

### 08489.

Eleftherios Avramidis, Annika Grützner-Zahn, Manuel Brack, Patrick Schramowski, Pedro Ortiz Suarez, Malte Ostendorff, Fabio Barth, Shushen Manakhimova, Vivien Macketanz, Georg Rehm, and Kristian Kersting. Occiglot at WMT24: European open-source large language models evaluated on translation. In Barry Haddow, Tom Kocmi, Philipp Koehn, and Christof Monz, editors, Proceedings of the Ninth Conference on Machine Translation, pages 292–298, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.wmt-1.23. URL https://aclanthology.org/2024.wmt-1.23/.

PLLuM Consortium. Pllum: A family of polish large language models. 2025. Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero,

Lewis Tunstall, and Thomas Wolf. Open llm leaderboard (2023-2024). https://huggingface.co/spaces/ open-llm-leaderboard-old/open_llm_leaderboard, 2023a.

Krzysztof Wróbel, SpeakLeash Team, and Cyfronet Team. Open pl llm leaderboard. https://huggingface.co/

### spaces/speakleash/open_pl_llm_leaderboard, 2024.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 2024. URL https:

### //zenodo.org/records/12608602.

Jan Koco´n, Piotr Miłkowski, and Monika Za´sko-Zieli´nska. Multi-level sentiment analysis of PolEmo 2.0: Extended corpus of multi-domain consumer reviews. In Mohit Bansal and Aline Villavicencio, editors, Proceedings of the

23rd Conference on Computational Natural Language Learning (CoNLL), pages 980–991, Hong Kong, China,

2019. Association for Computational Linguistics. doi:10.18653/v1/K19-1092. URL https://aclanthology. org/K19-1092.

Piotr Rybak, Robert Mroczkowski, Janusz Tracz, and Ireneusz Gawlik. KLEJ: Comprehensive benchmark for Polish language understanding. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1191–1201, Online, 2020. Association for Computational Linguistics. doi:10.18653/v1/2020.acl-main.111. URL https://aclanthology.org/2020.

### acl-main.111.

Slawomir Dadas, Michał Perełkiewicz, and Rafał Po´swiata. Evaluation of sentence representations in Polish. In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1674–1680, Marseille, France, 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https: //aclanthology.org/2020.lrec-1.207.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand and virtual meeting, 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.44.

Michał Marcinczuk, Marcin Ptak, Adam Radziszewski, and Maciej Piasecki. Open dataset for development of polish question answering systems. In Proceedings of the 6th Language & Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics, Wydawnictwo Poznanskie, Fundacja Uniwersytetu im. Adama Mickiewicza, 2013.

Sławomir Dadas. Training effective neural sentence encoders from automatically mined paraphrases. In 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 371–378, 2022. doi:10.1109/SMC53654.2022.9945218.

Maciej Ogrodniczuk and Mateusz Kope´c. The Polish summaries corpus. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 3712–3715, Reykjavik, Iceland, 2014. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf.

Michal Ptaszynski, Agata Pieciukiewicz, Pawel Dybala, Pawel Skrzek, Kamil Soliwoda, Marcin Fortuna, Gniewosz Leliwa, and Michal Wroczynski. Expert-annotated dataset to study cyberbullying in polish language. Data, 9(1):1, 2023.

Piotr Rybak, Piotr Przybyła, and Maciej Ogrodniczuk. PolQA: Polish question answering dataset. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 12846–12855, Torino, Italia, 2024. ELRA and ICCL. URL https://aclanthology.

### org/2024.lrec-main.1125.

Ryszard Tuora, Aleksandra Zwierzchowska, Natalia Zawadzka-Paluektau, Cezary Klamra, and Łukasz Kobyli´nski. Poquad-the polish question answering dataset-description and analysis. In Proceedings of the 12th Knowledge Capture Conference 2023, pages 105–113, 2023.

Samuel J. Paech. Eq-bench: An emotional intelligence benchmark for large language models, 2023. URL https:

### //arxiv.org/abs/2312.06281.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llmas-a-judge with mt-bench and chatbot arena. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html.

Remigiusz Kinas, Filipkowska Maria, SpeakLeash Team, and Cyfronet Team. Mt-bench pl. https://huggingface.

### co/spaces/speakleash/mt-bench-pl, 2024.

Jan Sowa, Magdalena Krawczyk, Natalia Nadolna, Anna Zieli´nska, Maria Filipkowska, Agnieszka Kosiak, Marta Kania, Krzysztof Wróbel, Remigiusz Kinas, Szymon Baczy´nski, SpeakLeash Team, and Cyfronet Team. Complex polish text understanding benchmark. https://huggingface.co/spaces/speakleash/cptu_bench, 2024.

Jakub Pokrywka, Jeremi Kaczmarek, and Edward Gorzela´nczyk. Gpt-4 passes most of the 297 written polish board certification examinations, 2024. URL https://arxiv.org/abs/2405.01589.

Sławomir Dadas, Małgorzata Gr˛ebowiec, Michał Perełkiewicz, and Rafał Po´swiata. Evaluating polish linguistic and cultural competency in large language models, 2025. URL https://arxiv.org/abs/2503.00995.

Krzysztof Jassem, Michał Ciesiółka, Filip Grali´nski, Piotr Jabło´nski, Jakub Pokrywka, Marek Kubis, Monika Jabło´nska, and Ryszard Staruch. Llmzszł: a comprehensive llm benchmark for polish, 2025. URL https://arxiv.org/abs/ 2501.02266.

Klaudia Thellmann, Bernhard Stadler, Michael Fromm, Jasper Schulze Buschhoff, Alex Jude, Fabio Barth, Johannes Leveling, Nicolas Flores-Herr, Joachim Köhler, René Jäkel, and Mehdi Ali. Towards multilingual llm evaluation for european languages, 2024. URL https://arxiv.org/abs/2410.08928.

Dan Nielsen. ScandEval: A benchmark for Scandinavian natural language processing. In Tanel Alumäe and Mark Fishel, editors, Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa), pages 185–201, Tórshavn, Faroe Islands, 2023. University of Tartu Library. URL https://aclanthology.org/2023. nodalida-1.20.

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https://huggingface.co/spaces/ open-llm-leaderboard-old/open_llm_leaderboard, 2023b.

Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. Open llm leaderboard v2.

### https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard, 2024.

Jinjie Ni, Fuzhao Xue, Xiang Yue, Yuntian Deng, Mahir Shah, Kabir Jain, Graham Neubig, and Yang You. Mixeval: Deriving wisdom of the crowd from LLM benchmark mixtures. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/ 2024/hash/b1f34d7b4a03a3d80be8e72eb430dd81-Abstract-Conference.html.

Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Berkeley function calling leaderboard. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_ calling_leaderboard.html, 2024.

Martin Fajcik, Martin Docekal, Jan Dolezal, Karel Ondrej, Karel Beneš, Jan Kapsa, Pavel Smrz, Alexander Polok, Michal Hradis, Zuzana Neverilova, Ales Horak, Radoslav Sabol, Michal Stefanik, Adam Jirkovsky, David Adamczyk, Petr Hyner, Jan Hula, and Hynek Kydlicek. Benczechmark : A czech-centric multitask and multimetric benchmark for large language models with duel scoring mechanism, 2024. URL https://arxiv.org/abs/2412.17933.

Eduardo A. S. Garcia. Open portuguese llm leaderboard. https://huggingface.co/spaces/eduagarcia/open_

pt_llm_leaderboard, 2024.

