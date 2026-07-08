# arXiv:2508.18579v2[cs.LG]29Aug2025

## DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model

Mohammadreza Ghaffarzadeh-Esfahani1, Ali Motahharynia1,2,*, Nahid Yousefian1, Navid Mazrouei1, Jafar Ghaisari3, and Yousof Gheisari1

1Regenerative Medicine Research Center, Isfahan University of Medical Sciences, Isfahan, Iran 2Isfahan Neuroscience Research Center, Isfahan University of Medical Sciences, Isfahan, Iran 3Department of Electrical and Computer Engineering, Isfahan University of Technology, Isfahan, Iran *Corresponding author: Ali Motahharynia, Regenerative Medicine Research Center, Isfahan University of Medical Sciences, Isfahan, 81746 73461, Iran, Tel: +98 313 668 7087, Email: alimotahharynia@gmail.com, ORCID: 0000-0002-1140-3257

Abstract

Drug discovery is a complex and resource-intensive process, making early prediction of approval outcomes critical for optimizing research investments. While classical machine learning and deep learning methods have shown promise in drug approval prediction, their limited interpretability constraints their impact. Here, we present DrugReasoner, a reasoning-based large language model (LLM) built on the LLaMA architecture and finetuned with group relative policy optimization (GRPO) to predict the likelihood of smallmolecule approval. DrugReasoner integrates molecular descriptors with comparative reasoning against structurally similar approved and unapproved compounds, generating predictions alongside step-by-step rationales and confidence scores. DrugReasoner achieved robust performance with an AUC of 0.732 and an F1 score of 0.729 on the validation set and 0.725 and 0.718 on the test set, respectively. These results outperformed conventional baselines, including logistic regression, support vector machine, and k-nearest neighbors and had competitive performance relative to XGBoost. On an external independent dataset, DrugReasoner outperformed both baseline and the recently developed ChemAP model, achieving an AUC of 0.728 and an F1-score of 0.774, while maintaining high precision and balanced sensitivity, demonstrating robustness in real-world scenarios. These findings demonstrate that DrugReasoner not only delivers competitive predictive accuracy but also enhances transparency through its reasoning outputs, thereby addressing a key bottleneck in AI-assisted drug discovery. This study highlights the potential of reasoning-augmented LLMs as interpretable and effective tools for pharmaceutical decision-making.

Keywords: Chain-of-thought reasoning; Drug approval; Drug discovery; Large language model; Reinforcement learning

### 1 Introduction

Drug discovery is a complex and costly process, often requiring over a decade and substantial financial investment, reaching approximately 879 million dollars, to bring a single compound from discovery to market [1,2]. This makes early screening and accurate prediction of a drug candidate’s eventual success crucial for optimizing resource allocation. In recent years, computational methods, including classical machine learning and deep learning have contributed to improve drug discovery processes from structural validation to toxicity and efficacy screening [3]. A notable example is ChemAP [4], which applies knowledge distillation from a teacher model that integrates multi-modal information into a student model capable of predicting drug approval from chemical structures alone. However, such models remains constrained by their limited interpretability, making it challenging to build trust for model-driven decisions [5].

Large language models (LLMs) have recently emerged as transformative tools in artificial intelligence (AI), demonstrating broad capabilities across a wide range of domains through training on massive and diverse datasets [6].A central advancement that has enhanced their precision and problem-solving capacity is the incorporation of chain-of-thought (CoT) reasoning, which allows LLMs to simulate human-like reasoning and improve the interpretability of their outputs [7].These reasoning abilities have attracted considerable interest in the field of drug discovery, where the complexity of biological systems and the need for integrative analysis present significant challenges [8]. Recent efforts, including frameworks such as DrugReAlign [9] and DrugAgent [10], have applied agentic AI approaches that extend LLMs with specialized tools for information retrieval, knowledge integration, and decision support. By combining CoT reasoning with structured access to domain-specific data, these systems enable novel applications such as drug repurposing. On the other hand, models like MolReasoner [11] and Mol-R1 [12] have focused on leveraging these reasoning capabilities to fine-tune models for de novo molecular design, enabling the generation of novel chemical structures with desirable pharmacological properties, highlighting a paradigm shift toward reasoning-augmented LLMs in drug discovery.

In this study, we introduce DrugReasoner, an interpretable LLM built on the LLaMA architecture [13], designed to predict the likelihood of a drug approval based on molecular features. The reasoning capability of DrugReasoner is central to this task, as approval decisions depend on integrating complex, multifaceted evidence into a coherent judgment. By fine-tuning the model with group relative policy optimization (GRPO) [14] on a curated dataset containing both approved and unapproved compounds, as well as structurally similar approved and unapproved molecules for each entry, DrugReasoner is able to not only produce accurate predictions but also to articulate its decision-making process. Comparative evaluation against baseline and recent approval predictor models demonstrates that DrugReasoner achieves both strong predictive performance and enhanced interpretability, underscoring the importance of reasoning-augmented LLMs for transparent and efficient AI-assisted drug discovery.

### 2 Results

In order to address the challenge of drug approval prediction, we developed DrugReasoner, an algorithm based on Llama-3.1-8B-Instruct. By fine-tuning it using GRPO with customized reward functions, this framework is designed to generate CoT reasoning and accurately predict small-molecule approval.

The model first analyzes the molecular features of the query compound, then compares them with the molecular features of the five most similar approved and unapproved molecules from the training set. Following this, it outputs a binary label (approved/unapproved) along with an explanatory rationale (Fig. 1).

Dataset

System prompt

Data collection

###### Data processing

You are a chemist specializing in drug discovery and molecular modeling... For each compound, you are provided with:

[Figure 1]

- - Molecular descriptors and properties calculated via RDKit.

- - The RDKit-derived descriptors and properties of the  ve most similar approved and  ve most similar unapproved small molecules (based on structural similarity).

[Figure 2]

Custom dataset

5 most similar approved SMILES

Follow these steps:

- 1. Analyze the compound's physicochemical properties.

- 2. Compare the compound with its most similar approved and unapproved counterparts ....

- 3. Think step by step and use your knowledge of molecular approval criteria and the patterns ... Based on your analysis, label the compound as either:

MOLFORMER Embedding

2255 approved SMILES

[Figure 3]

- - approved

- - unapproved Then assign a con dence score between 0 and 1, where:

- - A score near 1.0 indicates high con dence in your prediction (clear, strong evidence).

- - A score near 0.0 indicates high uncertainty (ambiguous, con icting, or weak evidence).

Approved + Unapproved SMILES

Similarity search

Physiochemical feature extraction

Respond in the following format: <think> (Your step-by-step scienti c reasoning) </think> <label> (approved or unapproved) </label> <score> (Numeric con dence score between 0 and 1) </score>

2255 unapproved SMILES

5 most similar unaproved SMILES

Models assessment

DrugReasoner development

Llama-3.1-8B-Instruct

- - Correctness
- - XML format
- - Soft format compliance
- - Interpretability
- - Con dence-alignment

Validation set Test set

External set

1.0

1.0

1.0

| |
|---|

[Figure 4]

0.8

0.8

0.8

Reward

0.6

0.6

0.6

GRPO

Score

Score

Score

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

DrugReasoner

AUC F1

AUC F1 AUC F1

Logistic regression KNN SVM XGBoost ChemAp DrugReasoner

- Figure 1: Schematic representation of DrugReasoner development and assessment. The upper panel illustrates dataset preparation and processing. The lower panel shows model training with group relative policy optimization (GRPO) using customized reward functions, followed by comparative evaluation against other models on validation, test, and external datasets.

#### 2.1 DrugReasoner is effectively trained to interpret small-molecule approval

DrugReasoner was fine-tuned on a dataset of 2,255 approved and 2,255 unapproved small molecules, using customized reward functions that encouraged accurate predictions and coherent reasoning. Training was performed over 14,500 optimization steps, during which the model generated four outputs per input. The reward diagram is presented in Fig. 2A (Supplementary File 1). Performance was monitored every 500 steps on the evaluation set. Checkpoint 12,500 was selected as the final model based on the AUC and other evaluation metrics and complete adherence (100%) to the expected output structure (Fig. 2B, Supplementary File 2). In addition, the model was required to produce a confidence score for each prediction. This score stabilized at 0.87 toward the end of training. An example of model input formatting and reasoning output is shown in supplementary file 3.

#### 2.2 DrugReasoner demonstrated reasonable performance in comparison to baseline models

DrugReasoner outperformed four baseline models, k-nearest neighbors (KNN), logistic regression, support vector machine (SVM), and XGBoost [15] on the validation set, achieving the highest overall balance of metrics (AUC = 0.732, accuracy = 0.732, recall = 0.721, precision = 0.738, specificity = 0.742, F1 = 0.729, Fig. 2C, Supplementary Files 4 and 5). It maintained robust performance on the test set (AUC = 0.725, accuracy = 0.725) and achieved the highest recall (0.702). Furthermore, it matched XGBoost’s F1-score (0.718) with competitive precision (0.735) and specificity (0.748) (Fig. 2D, Supplementary Files 4 and 5).

#### 2.3 DrugReasoner outperformed baseline and recently developed approval prediction models on the external dataset

On the external independent dataset, containing 17 approved and 8 unapproved drugs (ChemAP study dataset), DrugReasoner outperformed all baseline and ChemAP models (Supplementary Files 4 and 5). It achieved the highest AUC (0.728), F1-score (0.774), and precision (0.857), while maintaining a balanced accuracy (0.720) with strong recall (0.706) and specificity (0.750).

- A
- B
- C D

- 1
- 2
- 3
- 4
- 5

Reward

EMA

0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145

Steps (*100)

0.9

0.8

Evaluationmetrics

Validsamples(%)

0.7

0.6

0.5

F1 Precision Recall AUC

Speci city Sample validity (%)

0.4

0.3

- 97

- 97.5
- 98

98.5

99

- 99.5
- 100

0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145

Steps (*100)

Validation set metrics Test set metrics

1.0

1.0

0.8

0.8

0.6

0.6

Score

Score

0.4

0.4

0.2

0.2

0.0

0.0

AUC F1 Recall Speci city Precision AUC F1 Recall Speci city Precision

E

External set metrics

1.0

0.8

0.6

Score

0.4

Logistic regression KNN SVM XGBoost

0.2

ChemAp

DrugReasoner

0.0

AUC F1 Recall Speci city Precision

- Figure 2: DrugReasoner assessment. (A) Reward trajectory during group relative policy optimization (GRPO) training with exponential moving average (EMA). (B) Evaluation metrics of model on the validation set. (C) Comparative performance of DrugReasoner and baseline models on the validation set. (D) Comparative performance on the test set. (E) Comparative performance of DrugReasoner with baseline models and ChemAP on an independent external dataset.

In contrast, conventional machine learning baselines showed weaker discriminative ability (AUC = 0.529 to 0.618) and minimal sensitivity (≤ 0.235) despite perfect specificity. ChemAP exhibited lower AUC (0.64), recall (0.529), and specificity (0.75), indicating its limited generalizability and highlighting DrugReasoner’s superior real-world applicability.

- 3 Discussion

In this work, we presented DrugReasoner, a Llama-3.1-8B-Instruct algorithm, fine-tuned using GRPO and customized reward functions to predict drug approval outcomes while providing interpretable reasoning. The model achieved superior performance on the validation set compared to baseline models (logistic regression, KNN, SVM, and XGBoost). On the test set, it surpassed or showed competitive results compared to these models. On an external dataset, however, DrugReasoner significantly outperformed both baselines and the recently developed model, ChemAP [4]. These results highlight DrugReasoner’s enhanced predictive accuracy,

generalizability, and promise as a decision-support model in the drug discovery process.

The integration of reasoning capabilities into LLMs has emerged as a game-changing tool in computational drug discovery, enabling more interpretable and context-aware predictions across molecular tasks. For instance, models like CoTox [16] apply CoT reasoning to predict molecular toxicity by incorporating biological pathways and gene ontology, while MolReasoner [11] enhances interpretability through synthetic CoT samples and reinforcement learning to link chemical structures with linguistic descriptions. Similarly, frameworks such as DrugPilot [17] and DrugAgent [10] employ multi-agent collaboration and reasoning to automate workflows. They integrate domain-specific tools, addressing challenges in data processing and tasks like drug-target interaction modeling. Mol-R1 [12] further advances explicit long-CoT reasoning for molecule generation via iterative adaptation and distillation strategies, improving performance in knowledge-intensive domains. In this context, DrugReasoner extends reasoning-augmented LLMs to drug approval prediction by applying GRPO-fine-tuned CoT reasoning using molecular feature comparisons with structurally similar approved and unapproved compounds to generate binary outcomes, confidence scores, and explanatory rationales. By doing so, our model bridges predictive performance with interpretability in a domain where transparent decision-making is crucial.

Traditional machine learning approaches to approval prediction like DrugApp [18] have relied on diverse features, including molecular, physicochemical, clinical trial, and patent data, often achieving robust performance via ensemble methods. However, features like patent or clinical trial data are typically available only post-trial, limiting their use in early discovery. ChemAP addresses this gap by predicting approval from chemical structures via knowledge distillation, enriching semantic knowledge to distinguish approved from unapproved drugs. Furthermore, DrugReasoner introduces an LLM-based paradigm that operates on molecular features (derived solely from structures) to enhance prediction with interpretable reasoning. This enables superior results on external datasets compared to ChemAP and conventional baselines. By emphasizing coherent rationales and confidence scoring, our model not only improves predictive accuracy but also enhances trustworthiness, offering a new paradigm for early-stage drug discovery.

Despite promising results, this study has limitations that should be addressed in future studies. To mitigate the risk of potential data leakage from LLMs pretrained on vast internet datasets, we excluded simplified molecular-input line-entry system (SMILES) representations, relying instead on molecular features for improved interpretability. While this reduced the risk of bias, it may also have constrained the model’s ability to capture fine-grained structural information. Computational constraints further limited our exploration of larger models. Extending the context window, which could have allowed the inclusion of SMILES for similar approved and unapproved molecules, and systematic hyperparameter searches during GRPO fine-tuning could further enhance the reasoning performance. Future work should address these constraints by integrating structural data in a controlled manner, optimizing hyperparameters, scaling model size, and context length to improve reasoning and predictive accuracy.

Overall, DrugReasoner represents a significant advancement toward interpretable reasoningaugmented LLMs for drug approval prediction. By combining CoT reasoning with GRPO fine-tuning and molecular feature comparisons, it achieves strong performance and generates explanatory rationales across validation, test, and external datasets. Its ability to operate solely on molecular features while maintaining generalizability, highlights its potential to accelerate pharmaceutical decision-making. As such, DrugReasoner establishes a foundation for transparent and trustworthy AI tools, capable of guiding critical investment and research decisions from the earliest stages of drug discovery.

### 4 Methods

#### 4.1 Dataset Preparation

A dataset of small molecules was obtained from the ChEMBL database (version 35) [19] including approved molecules in Phase IV clinical trials and molecules in the preclinical phase. Child molecules were replaced by their parent compounds to avoid duplication. A subset of 2,255 approved and 2,255 unapproved molecules was created by random undersampling of the unapproved class. The SMILES structure was retrieved, and strings were canonicalized using RDKit (version 2023.9.5) [20].

The dataset was partitioned into training, validation, and test subsets (8:1:1) using a stratified sampling strategy to maintain class distribution across all splits. The validation set was used for hyperparameter optimization, and the test set was reserved for evaluating the final model. Additionally, an independent external dataset used in the ChemAP paper was used to assess real-world performance. This dataset contained 20 approved and 8 unapproved drugs. Three approved drugs overlapping with the training, validation, or test sets were removed, and the remaining molecules were used for external evaluation.

#### 4.2 Data Processing

##### 4.2.1 Molecular embedding

SMILES representations were embedded using MOLFORMER [21], a pretrained transformer model trained on SMILES strings via masked language modeling. MOLFORMER incorporates linear attention with rotary positional embeddings to enhance scalability and contextual encoding. Each molecule was tokenized using its tokenizer and embeddings were computed in batches of 64 with attention masking, mapping each molecule into a 768-dimensional embedding space. Final molecular embeddings were derived by applying mean pooling over the last hidden states, weighted by the attention mask to account for variable sequence lengths and padding.

##### 4.2.2 XGBoost similarity search

To enhance the dataset and improve model performance, for each molecule, we identified the five most similar approved and the five most similar unapproved molecules. Similarity was computed using XGBoost (version 3.0.2), trained on the MOLFORMER embeddings within the training set only. The model was trained as a binary classifier (approved vs. unapproved) on the training set using molecular embeddings from MOLFORMER as input features, with log-loss (binary cross-entropy) as the evaluation metric. Hyperparameter optimization was performed using Optuna (version 3.6.1) [22] on 0.1 of the training set with the tree-structured parzen estimator sampler and median pruner for early stopping. The search space included learning rate, maximum tree depth, L1 and L2 regularization, and subsampling ratios. A total of 1,000 trials were conducted, with early stopping of 20 rounds based on validation log-loss. The optimal hyperparameter configuration was used to train the final XGBoost model on the full training set.

The trained XGBoost model was used to generate leaf embeddings for each molecule by recording the index of the terminal leaf node within each decision tree for a given input. Pairwise hamming distances between leaf embeddings quantified topological similarity in the decision space. In the intra-training set, we implemented a self-comparison procedure, where for each molecule in the training set, the five most similar approved and unapproved compounds were identified based on their leaf proximity. For validation and test sets, a query-based similarity function was employed. Each molecule was embedded via MOLFORMER, and its leaf trajectory was compared against the training set to identify the top five most similar molecules of each class.

##### 4.2.3 Feature extraction

To prevent data leakage—given that the base model was trained on extensive internet-scale data—we excluded SMILES strings, which could allow the model to directly recognize known molecules. Instead, we used only computed molecular features which also improved interpretability. For all molecules, including query molecules and their neighbors, we computed physicochemical and structural descriptors using RDKit. These included molecular weight, LogP, topological polar surface area, hydrogen bond donors and acceptors, rotatable bonds, molecular refractivity, chiral centers, heavy atoms, ring counts, and formal charge. Additionally, structural alert checks were performed using pan assay interference compounds and Brenk filters to assess undesirable substructures.

##### 4.2.4 Prompt instruction Each input sample included:

- • RDKit descriptors of the candidate molecule
- • RDKit descriptors of the five most similar approved molecules
- • RDKit descriptors of the five most similar unapproved molecules

These features were encoded in a structured prompt, where the system prompt defined a domain-specific instruction set to simulate expert chemical reasoning Supplementary Files 3.

The model output consisted of three fields in structured XML tags:

- • <think>: Explanation and comparative reasoning
- • <label>: Binary decision (approved or unapproved)
- • <score>: Confidence score (0.0 to 1.0) This schema ensured interpretability and allowed structured reward modeling.

#### 4.3 DrugReasoner development

We fine-tuned Llama-3.1-8B-Instruct [23] using GRPO trainer from hugging face transformer reinforcement learning (TRL) library (version 0.15.2) [24] to perform CoT reasoning on molecular features.

Training utilized the unsloth library (version 2025.3.19) [25] with:

- • Maximum sequence length: 4,096 tokens (prompt + completion, 2,048 each)
- • 4-bit quantization [26]
- • Low-rank adaptation (LoRA) [27] with rank 32, applied to key attention projection layers (q proj, k proj, v proj, o proj, gate proj, up proj, down proj) to reduce memory overhead.

##### 4.3.1 Reinforcement learning with GRPO

The model was trained using GRPO trainer for 14,500 steps, generating four outputs per prompt during training, which were scored using multi-objective rewards. Training was performed with a learning rate of 5 × 10−6, Adam optimizer parameters β1 = 0.9 and β2 = 0.99, a weight decay of 0.1, 100 warmup steps, and a cosine learning rate scheduler. All other variables set to the default of the library. Optimization employed the paged adamw 8bit optimizer [28] with a per-device batch size of 4 and gradient clipping at a maximum norm of 0.1.

Group relative policy optimization works by generating multiple possible responses or a group (Kj) of possible responses (actions = {aj1,aj2,...,ajKj}) from the current policy (πθ) for each input (sj). Each response in the group is evaluated and assigned a reward score (Rjk) based on the defined reward function. The average reward across the group (R¯j) serves as a baseline (Eq. 1), and the advantage (Ajk) of each response is calculated as the difference between its reward and this group mean (Eq. 2).

Kj

1 Kj

R¯j =

Rjk (1)

k=1

Ajk = Rjk − R¯j (2)

Using this advantage metric, the model updates its parameters to reinforce responses with above-average rewards while discouraging those with below-average performance. To ensure training stability, a clipped surrogate objective (Ljk(θ)) is applied, where the clipping function (clip(.,1−ϵ,1+ϵ)) prevents updates from deviating excessively from the reference policy. Here, πθold is the policy before the update and ϵ is a small hyperparameter (Eq. 3 and 4).

πθ(ajk|sj) πθold(ajk|sj)

rjk(θ) =

(3) Ljk(θ) = min(rjk(θ)Ajk,clip(rjk(θ),1 − ϵ,1 + ϵ)Ajk) (4)

This optimization is guided by a specialized loss function (L(θ)) that also incorporates a KL-divergence penalty (DKL) to prevent abrupt policy changes and ensure stable learning (Eq.

- 5).

L(θ) = −

j k

Ljk(θ) + β

j

DKL(πθold(·|sj)||πθ(·|sj)) (5)

##### 4.3.2 Reward function

We implemented a multi-faceted reward function strategy composed of five distinct objectives to perform the scientific reasoning task.

- 1. Correctness: Compares predicted label to ground truth (2.0 for correct, 0.0 otherwise). This function also logs intermediate outputs for qualitative analysis.
- 2. XML format: Assesses whether the response follows the specified format (presence and singularity of <think>, <label>, and <score> tags). Each correctly placed start (<>) and end (</>) tag contributes 0.125, summing to a maximum of 0.75 (6×0.125).
- 3. Soft format compliance: A regular expression-based reward function that returns 0.5 if the response approximately matches the expected format, meaning it includes the required structural elements (e.g., specific tags) in the correct order with allowances for minor whitespace variations, extra text, or small ordering changes, and 0.0 if key elements are missing or the structure is significantly incorrect.
- 4. Interpretability: Assigns 0.5 if the extracted label is semantically valid (i.e., approved or unapproved).
- 5. Confidence-alignment: Evaluates consistency between the predicted label and its confidence score:

- • Correct with high confidence (≥ 0.7): 2.0

- • Correct with moderate confidence (0.4− < 0.7): 1.0
- • Correct with low confidence (< 0.4): 0.0
- • Incorrect with low confidence (< 0.4): 1.0
- • Incorrect with moderate confidence (0.4− < 0.7): 0.5
- • Incorrect with high confidence (≥ 0.7): 0.0

Each reward function returned a scalar value, which was used to score the model’s generated completions during GRPO training. Reward signals were recorded at each step for inspection and debugging Fig. 2A, Supplementary File 1.

##### 4.3.3 Checkpoint selection

Training was performed for a total of 14,500 steps. To identify the optimal checkpoint for inference, we tracked the trajectory of reward scores throughout training. In addition, every 500 steps, we performed inference using top p = 0.9, top k = 9, and temperature = 1, evaluating the model on the validation set with metrics including, AUC, F1, precision, recall, specificity, and accuracy. The checkpoint at 12,500 steps achieved optimal performance and was selected as the final model Fig. 2B, Supplementary File 2.

#### 4.4 Model evaluation

DrugReasoner was evaluated against four conventional baselines: logistic regression, SVM, KNN, and XGBoost. To assess real-world applicability, we compared DrugReasoner with ChemAP using the external validation dataset described in the ChemAP study. All baseline models, along with ChemAP and DrugReasoner, were evaluated on this external dataset.

#### 4.5 Hardware and training time

Training was performed on a single NVIDIA V100 GPU (32 GB VRAM) with an AMD CPU (64 GB RAM, 4 cores). Total training time was approximately 794.3 hours. Baseline models were evaluated on CPU-only hardware.

- 5 Data availability

All data generated or analyzed during this study are included in the manuscript and supporting files. The drug approval prediction dataset used in this study is publicly available at: “Moreza009/drug approval prediction” on Hugging Face.

- 6 Code availability

The checkpoints and code for assessment of drug approval are publicly available at https:// github.com/mohammad-gh009/DrugReasoner and “Moreza009/Llama-DrugReasoner” Explore the interactive user interface of DrugReasoner at This notebook.

- 7 Acknowledgment Not applicable.
- 8 Funding No funding was received for this study or its publication.
- 9 Competing interest The authors declare no competing interests.
- 10 Authors contribution

Conceptualization: M.G, A.M, Y.G. Dataset preparation: M.G, A.M, N.Y. Model development: M.G, A.M. Model assessment: M.G, A.M, N.Y, J.G. Data interpretation: All authors. Drafting original manuscript: M.G, A.M, N.Y, N.M. Revising the manuscript: A.M, J.G, Y.G. All the authors have read and approved the final version for publication and agreed to be responsible for the integrity of the study.

### References

- [1] Sertkaya, A., Beleche, T., Jessup, A., & Sommers, B. D. (2024). Costs of Drug Development and Research and Development Intensity in the US, 2000-2018. JAMA Network Open, 7(6), e2415445. https://doi.org/10.1001/jamanetworkopen.2024.15445
- [2] Singh, N., Vayer, P., Tanwar, S., Poyet, J.-L., Tsaioun, K., & Villoutreix, B. O. (n.d.). Frontiers — Drug discovery and development: Introduction to the general public and patient groups. https://doi.org/10.3389/fddsv.2023.1201419
- [3] Blanco-Gonz´lez, A., Cabez´n, A., Seco-Gonz´lez, A., Conde-Torres, D., AnteloRiveiro, P., Pin˜eiro, A.,´ & Garcia-Fandino, R. (2023). The Role of AI in Drug Discovery: Challenges, Opportunities, and Strategies. Pharmaceuticals, 16(6), 891. https://doi.org/10.3390/ph16060891
- [4] Cho, C., Lee, S., Bang, D., Piao, Y., & Kim, S. (2024). ChemAP: Predicting drug approval with chemical structures before clinical trial phase by leveraging multimodal embedding space and knowledge distillation. Scientific Reports, 14(1), Article 1. https://doi.org/10.1038/s41598-024-72868-0
- [5] Interpretable AI in healthcare: Clinical applicability and usability. (2025, August 25). Nature. https://www.nature.com/collections/fifhafhibf
- [6] Naveed, H., Khan, A. U., Qiu, S., Saqib, M., Anwar, S., Usman, M., Akhtar, N., Barnes, N., & Mian, A. (2024). A Comprehensive Overview of Large Language Models (No. arXiv:2307.06435). arXiv. https://doi.org/10.48550/arXiv.2307.06435
- [7] Chen, Q., Qin, L., Liu, J., Peng, D., Guan, J., Wang, P., Hu, M., Zhou, Y., Gao, T., & Che, W. (2025). Towards Reasoning Era: A Survey of Long Chainof-Thought for Reasoning Large Language Models (No. arXiv:2503.09567). arXiv. https://doi.org/10.48550/arXiv.2503.09567
- [8] Lin, A., Ye, J., Qi, C., Zhu, L., Mou, W., Gan, W., Zeng, D., Tang, B., Xiao, M., Chu, G., Peng, S., Wong, H. Z. H., Zhang, L., Zhang, H., Deng, X., Li, K., Zhang, J., Jiang, A., Li, Z., & Luo, P. (2025). Bridging artificial intelligence and biological sciences: A comprehensive review of large language models in bioinformatics. Briefings in Bioinformatics, 26(4), bbaf357. https://doi.org/10.1093/bib/bbaf357
- [9] Wei, J., Zhuo, L., Fu, X., Zeng, X., Wang, L., Zou, Q., & Cao, D. (2024). DrugReAlign: A multisource prompt framework for drug repurposing based on large language models. BMC Biology, 22(1), 226. https://doi.org/10.1186/s12915-024-02028-3
- [10] Liu, S., Lu, Y., Chen, S., Hu, X., Zhao, J., Lu, Y., & Zhao, Y. (2025). DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration (No. arXiv:2411.15692). arXiv. https://doi.org/10.48550/arXiv.2411.15692
- [11] Zhao, G., Li, S., Lu, Z., Cheng, Z., Lin, H., Wu, L., Xia, H., Cai, H., Guo, W., Wang, H., Xu, M., Zhu, S., Ke, G., Zhang, L., & Gao, Z. (2025). MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs (No. arXiv:2508.02066). arXiv. https://doi.org/10.48550/arXiv.2508.02066
- [12] Li, J., Wang, W., Zhang, Q., Li, J., Zhang, D., Zheng, C., Zhang, S., Wei, X., & Li, Q. (2025a). Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery (No. arXiv:2508.08401). arXiv. https://doi.org/10.48550/arXiv.2508.08401

- [13] Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., & Ma, Z. (2024). The Llama 3 Herd of Models (No. arXiv:2407.21783). arXiv. https://doi.org/10.48550/arXiv.2407.21783
- [14] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (No. arXiv:2402.03300). arXiv. https://doi.org/10.48550/arXiv.2402.03300
- [15] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785
- [16] Park, J., Park, Y., Song, M., Park, S., Lee, D., Baek, S., & Kang, J. (2025). CoTox: Chainof-Thought-Based Molecular Toxicity Reasoning and Prediction (No. arXiv:2508.03159). arXiv. https://doi.org/10.48550/arXiv.2508.03159
- [17] Li, K., Wu, Z., Wang, S., Wu, J., Pan, S., & Hu, W. (2025). DrugPilot: LLMbased Parameterized Reasoning Agent for Drug Discovery (No. arXiv:2505.13940). arXiv. https://doi.org/10.48550/arXiv.2505.13940
- [18] F, C., & T, D. (n.d.). Machine learning-based prediction of drug approvals using molecular, physicochemical, clinical trial, and patent-related features. PubMed. Retrieved August 18, 2025, from https://pubmed.ncbi.nlm.nih.gov/36444655/?dopt=Abstract
- [19] Zdrazil, B., Felix, E., Hunter, F., Manners, E. J., Blackshaw, J., Corbett, S., de Veij, M., Ioannidis, H., Lopez, D. M., Mosquera, J. F., Magarinos, M. P., Bosc, N., Arcila, R., Kizil¨ren, T., Gaulton, A., Bento, A. P., Adasme, M. F., Monecke, P., Landrum, G. A., & Leach, A. R. (2024). The ChEMBL Database in 2023: A drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Research, 52(D1), D1180–D1192. https://doi.org/10.1093/nar/gkad1004
- [20] Release 2023 09 5 (Q3 2023) Release · rdkit/rdkit. (n.d.). GitHub. Retrieved August 17, 2025, from https://github.com/rdkit/rdkit/releases/tag/Release 2023 09 5

- [21] Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., & Das, P. (2022). Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence, 4(12), 1256–1264. https://doi.org/10.1038/s42256-022-005807
- [22] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework (No. arXiv:1907.10902). arXiv. https://doi.org/10.48550/arXiv.1907.10902
- [23] Meta-llama/Llama-3.1-8B-Instruct · Hugging Face. (2024, December 6). https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- [24] von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., & Gallou´edec, Q. (2025). TRL: Transformer Reinforcement Learning (Version 0.21) [Python]. https://github.com/huggingface/trl (Original work published 2020)
- [25] Daniel Han, M. H., & team, U. (2023). Unsloth [Computer software]. http://github.com/unslothai/unsloth

- [26] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs (No. arXiv:2305.14314). arXiv. https://doi.org/10.48550/arXiv.2305.14314
- [27] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.

(2021). LoRA: Low-Rank Adaptation of Large Language Models (No. arXiv:2106.09685). arXiv. https://doi.org/10.48550/arXiv.2106.09685

- [28] Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2022). 8bit Optimizers via Block-wise Quantization (No. arXiv:2110.02861). arXiv. https://doi.org/10.48550/arXiv.2110.02861

