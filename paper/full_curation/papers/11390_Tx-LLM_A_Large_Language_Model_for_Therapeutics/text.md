### arXiv:2406.06316v1[cs.CL]10Jun2024

# Tx-LLM: A Large Language Model for Therapeutics

Juan Manuel Zambrano Chaves∗,1, Eric Wang∗,2, Tao Tu2, Eeshit Dhaval Vaishnav, Byron Lee, S. Sara Mahdavi2, Christopher Semturs1, David Fleet2, Vivek Natarajan†,1 and Shekoofeh Azizi†,‡,2

1Google Research, 2Google DeepMind

Developing therapeutics is a lengthy and expensive process that requires the satisfaction of many different criteria, and AI models capable of expediting the process would be invaluable. However, the majority of current AI approaches address only a narrowly defined set of tasks, often circumscribed within a particular domain. To bridge this gap, we introduce Tx-LLM, a generalist large language model (LLM) fine-tuned from PaLM-2 which encodes knowledge about diverse therapeutic modalities. Tx-LLM is trained using a collection of 709 datasets that target 66 tasks spanning various stages of the drug discovery pipeline. Using a single set of weights, Tx-LLM simultaneously processes a wide variety of chemical or biological entities (small molecules, proteins, nucleic acids, cell lines, diseases) interleaved with free-text, allowing it to predict a broad range of associated properties, achieving competitive with state-of-the-art (SOTA) performance on 43 out of 66 tasks and exceeding SOTA on 22. Among these, Tx-LLM is particularly powerful and exceeds best-in-class performance on average for tasks combining molecular SMILES representations with text such as cell line names or disease names, likely due to context learned during pretraining. We observe evidence of positive transfer between tasks with diverse drug types (e.g., tasks involving small molecules and tasks involving proteins), and we study the impact of model size, domain finetuning, and prompting strategies on performance. We believe Tx-LLM represents an important step towards LLMs encoding biochemical knowledge and could have a future role as an end-to-end tool across the drug discovery development pipeline.

###### 1 Introduction

Developing therapeutics is a risky enterprise, as 90% of clinical trial candidates fail, and even successful therapeutics typically take 10-15 years and 1-2 billion dollars until approval [1, 2]. Perhaps the most daunting obstacle in this process is that a successful therapeutic must simultaneously satisfy numerous criteria. For example, a drug should interact with its proposed target, ultimately leading to the desired therapeutic effect and clinical efficacy. At the same time, the drug should be non-toxic and have drug-like properties (e.g., solubility, permeability, suitable pharmacokinetics, and pharmacodynamics). In clinical trials, unexpected off-target effects and interactions may counterbalance the effects of an otherwise promising drug candidate [3]. Furthermore, practical considerations regarding small molecule synthesis, or biological molecule developability must be taken into account.

Given the expense of experimentally assessing each of these characteristics, curated collections of experimental data paired with machine learning models for predicting therapeutic properties are useful as initial screening steps [4]. To aid the development of these models, the Therapeutics Data Commons (TDC) was developed as a resource containing AI-ready datasets and benchmarks for a diverse range of therapeutics-related tasks such as drug-target binding prediction or drug toxicity prediction [5, 6]. Current state-of-the-art (SOTA) models for TDC datasets are largely focused on individual tasks with one approach being to train a library of specialist models and call upon different specialists for each step of the therapeutic development pipeline [5]. However, specialist models lack awareness of other tasks in the therapeutic development pipeline, which may in turn limit their ability to contextualize and improve performance.

Large language models (LLMs) have emerged as useful systems for encoding information from large-scale data

∗ Equal contributions. Contributions made as a student researcher. † Equal leadership. ‡ Corresponding author: shekazizi@google.com

###### Therapeutics Data Commons (TDC) Therapeutics Instruction Tuning (TxT)

y

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

66 Tasks x

709 Datasets 10 Million (Instructions, Answer) Pairs

Small molecules

Target discovery

[Figure 5]

[Figure 6]

###### !

Macromolecules

Efﬁcacy & safety

[Figure 7]

[Figure 8]

Cell & gene therapies

Manufacturing

Classiﬁcation Generation

###### Classiﬁcation

Generation

Input Output

Drug Y

Given a product SMILES string, predict the reactant SMILES string.

Given a drug SMILES string, predict whether it (A) is not mutagenic (B) Is mutagenic

NC1C2ccccc2C(=O)N1Cc 1ccccc1

CCOC(=O)CBr

Cc1nnsc1SCC(=O)O 0 CCn1c(N)nc2ccccc21 0 O=C(O)CC(O)C(=O)O 1 … …

Product: NC1C2ccccc2C(=O)N1Cc1ccccc1

[NH2:1][C:2]1[S:3][C:4]([S:7] [C:8]#[N:9])=[CH:5][N:6]=1

[NH2:1][C:2]1[S:3][ CH:4]=[CH:5][N:6 ]=1.[S-:7][...

Drug: Cc1nnsc1SCC(=O)O Answer: (A)

Answer: CCOC(=O)CBr

… …

Regression

Regression

Given two drug SMILES strings and a cell line, predict the drug synergy from 0 to 1000.

Drug1 Drug2 Cell Line ID Synergy

C(=O)(N)NO N.N.Cl[Pt+2]Cl 786-0 1.059215 C(=O)(N)NO N.N.Cl[Pt+2]Cl ACHN -1.159521 CS(=O)(=O)CCCCO N.N.Cl[Pt+2]Cl OVCAR3 2.033875 … … … …

- Drug 1: C(=O)(N)NO
- Drug 2: N.N.Cl[Pt+2]Cl Cell line: 786-0, renal cell adenocarcinoma Answer: 562

###### TxT Prompt

Instructions: Answer the following question about drug synergy

Context: Synergy is a dimensionless measure of deviation of an observed drug combination response from the expected effect of non-interaction. Drug combinations were tested against cancer cell lines...

Prompted drug

- Drug 1

- Drug 2

Dataset(TDC)

Prompted drug

Question: Given two drug SMILES strings and a cell line description, predict the normalized drug synergy from 000 to 1000, where 000 is minimum drug synergy and 1000 is maximum drug synergy.

Prompted cell line

Cell Line ID

- Drug1 SMILES: C(=O)(N)NO

- Drug2 SMILES: N.N.Cl[Pt+2]Cl Cell line description: 786-0, renal cell adenocarcinoma Answer: 562

Target

Synergy

- Drug1 SMILES: CS(=O)(=O)CCCCO

- Drug2 SMILES: N.N.Cl[Pt+2]Cl Cell line description: OVCAR3, ovarian cell adenocarcinoma Answer: 678

Aggregated TDC & Literature Search

[Figure 9]

Literature Search …

- Figure 1 | Overview of the Tx-LLM. (top) Datasets from the Therapeutic Data Commons are used to construct the Therapeutics instruction Tuning (TxT) collection. The original tabular datasets contain a variety of drug types including small molecules, macro-molecules such as proteins and nucleic acids, cells, and genes. The tasks encompass a broad range of areas relevant to drug discovery and development such as predicting targets, evaluating efficacy and safety, and predicting ease of manufacturing. TxT interleaves free-text instructions with string representations of molecules, such as SMILES strings for small molecules or amino acid sequences for proteins. TxT is used to prompt and finetune Tx-LLM to solve classification, regression, or generation tasks. (bottom) Example of a TxT prompt for predicting drug synergy. The prompt is composed of Instructions, Context, and a Question using information from the corresponding TDC dataset and/or literature search and may also contain exemplars to aid in-context learning.

and communicating the information using language. Interestingly, LLMs have shown promise for multiple types of tasks such as multiple-choice question answering [7], time series prediction [8], and regression [9]. Furthermore, LLMs can be effectively adapted to specific domains such as medicine or chemistry with in-context learning or finetuning [10–14]. We hypothesized that tasks across the therapeutic development pipeline, even those involving diverse drug types such as small molecules and protein sequences, could be combined to train a generalist LLM with improved performance on individual tasks while using the same set of weights for all tasks.

In this work, we develop and introduce such a generalist model, Tx-LLM, by representing therapeutics as strings and finetuning the PaLM-2 base LLM on a diverse set of classification, regression, and generation tasks (Figure 1). Our key contributions are as follows:

- • Tx-LLM performs above or near SOTA for 43 out of 66 tasks. For datasets combining molecular string representations with text, Tx-LLM is especially effective and exceeds SOTA on average for these, likely due to context learned during LLM pretraining.
- • Interestingly, we find evidence of positive transfer between datasets with diverse drug types, as training on datasets including biological sequences improves performances on molecular datasets.
- • We perform ablation studies and observe that scale, domain finetuning, and prompting strategies also significantly impact model performance.
- • The proposed Tx-LLM shows promise as an end-to-end therapeutic development assist, allowing one to query a single model for multiple steps of the development pipeline (Figure 2).

###### 2 Related works

Large language models (LLMs) Since the advent of transformer-based models [15], LLMs have become increasingly powerful at a variety of natural language processing tasks [16, 17]. LLMs are trained using self-supervised learning on large-scale text corpi and have been shown to encode information while also generalizing to unseen tasks. Interestingly, it has recently been shown that LLMs are able to perform regression on diverse tasks using only textual representations of mathematical parameters and values [9].

Specialist models for therapeutics Therapeutics have been represented in a variety of ways. Molecules can be naturally represented as graphs, and graph neural networks (GNNs) have been applied for a variety of prediction or generation tasks [18–24]. A notable application of GNNs was the discovery of Halicin, an antibiotic which was effective against previously pan-resistant bacterial strains [25]. Molecules are also commonly represented using binary vectors (fingerprints), which capture the local environment of each atom in the molecule within a predefined radius and can be input into a variety of models [26–28]. Proteins and nucleic acids are conveniently represented using their amino acid or nucleotide sequences, and which can then be encoded in multiple ways to predict properties such as fitness [29], binding [30, 31], and secondary structure [32]. The introduction of AlphaFold and its successors significantly advanced the field of structure prediction [33–36], which may allow further developments in areas such as design of structure-based drugs [37] and vaccines [38].

Language models for biology and chemistry Language models have also emerged as tools for biology and chemistry. While many LLMs have shown limited performance for chemistry [39, 40], LlaSMol recently showed that finetuning LLMs for chemistry achieved near-SOTA performance on multiple tasks [41]. Beyond chemical property prediction, Chemcrow used LLMs for robotics-guided organic synthesis without human interaction [42]. Protein language models such as ESM [43, 44], Unirep [45], and ProtGPT2 [46] have used self-supervised pretraining (such as masked token prediction) to generate protein embeddings that were useful for downstream tasks such as predicting stability and functional effects of mutations. It has also been proposed that the probabilities assigned to amino acids during masked token prediction correlate to fitness [47] and provide an efficient landscape for computational protein engineering [48]. ProtLLM combined protein sequence encoders with language encoders [49], and BioT5 combined molecular string representations with protein names, sequences, or structures to train LLMs on various prediction tasks [50]. At the cellular level, scGPT [51], GenePT [52], and Geneformer [53] represent cells as rank ordered lists of highly expressed genes for tasks such as cell type annotation or therapeutic target discovery.

## Discovery Development

Target Identiﬁcation and Validation

Lead Discovery & Compound Screening

Pre-clinical Clinical

Drug-Target Interaction

Dose In vivo Toxicity Efﬁcacy

ExampleDatasetsPromptsAnswersExampleTasks

Gene-Disease Association

Pharmacokinetics Drug-like Properties In vitro Toxicity

High Throughput Screening

Protein-Protein Interaction

Drug-Drug Interaction

Reaction Yields Prediction

BindingDB SARS-CoV-2 USPTO

Caco2 Wang hERG Tox21

DisGeNET HuRI

Phase I, II, III trial ClinTox

[Figure 10]

Predict association score from 0 to 1 between

Given

Predict whether

[Figure 11]

Given

[Figure 12]

[Figure 13]

[Figure 14]

predict whether phase II trial will be successful for Type 2 Diabetes Mellitus

and

blocks the hERG gene, causing cardiotoxicity

and Type 2 Diabetes Mellitus

predict binding afﬁnity

##### Tx-LLM

0.85 KD 0.05 μM No Yes

- Figure 2 | Tx-LLM may be effective for end-to-end therapeutic development. Tx-LLM is a single model that can be queried for multiple steps of the therapeutic development process, covering tasks from early-stage target discovery to late-stage clinical trial approval. We list example tasks associated with each stage of the therapeutic development pipeline, example datasets in TDC that correspond to these tasks, and example prompts that can be used to query Tx-LLM. For illustration, the example prompts are geared towards discovering new small molecules against targets associated with type 2 diabetes, and the datasets associated with the example prompts are shown in bold.

###### 3 Methods

###### 3.1 Datasets

We assembled TxT, a collection of 709 drug discovery datasets comprising 66 tasks formatted for instruction tuning. We sourced our datasets from TDC, a publicly available repository that offers a wide variety of tasks spanning the drug discovery process. The median dataset size was 11,000, and the distribution of dataset sizes is illustrated in Figure A.1. We excluded a small number of datasets found in TDC for various reasons, which are detailed in Table A.1.

Each dataset in TxT is formatted as a text prompt comprised of four components (instructions, context, question, answer), illustrated in Figure 1 and Tables A.3 to A.5. Instructions consisted of a short sentence describing the task at hand, such as “Answer the following question about drug properties”. For each dataset, we crafted context, i.e., free-text descriptions providing additional information that grounds the question in a relevant biochemical setting. Contexts were 2-3 sentences long, sourced from TDC dataset descriptions, and manually complemented based on a brief literature search of the topic. For specialized assays describing a specific experimental condition, such as ToxCast, additional information for contexts were obtained from publicly available assay descriptions [54, 55]. The question is a succinct query that specifies the specific property being asked, and interleaves English text with text-based representations of therapeutics (e.g., “Does the following molecule cross the blood brain barrier? <molecule>”. The format of answers varied depending of the type of task.

Datasets in TxT fell into one of three categories:

- (i) Binary classification questions were formatted as a prediction of a single property of a therapeutic with two possibilities, yes/no (e.g., whether a drug is toxic).
- (ii) Regression questions were formatted as a prediction of a single property of a therapeutic on a continuous scale (e.g. drug-target binding affinity). To leverage the token-based, and not float-based, representation in existing language models, we uniformly binned the labels between 0 and 1000 and instruct Tx-LLM to predict the bin label. On evaluation, the predicted bin was transformed back to the original numeric label space.
- (iii) Generation we focused on one generation task, which consists of predicting reactants of a chemical reaction given the product, sourced from the USPTO dataset [56].

String representations of diverse types of therapeutics in TxT fell into one of the following categories:

- (i) SMILES Small molecules were represented with their SMILES string.
- (ii) Amino acid Proteins and peptides were represented with their amino acid sequences. Multiple Histopatibility Complex molecules, such as those found in the in the MHC1 IEDB IMGT Nielsen [57] and MHC2 IEDB Jensen [58] datasets were represented using their pseudo-sequences (only showing residues that are in contact with a peptide), T cell receptors were represented using their CDR3 hypervariable loops.
- (iii) Nucleotide Nucleic acids were represented with their nucleotide sequence.
- (iv) Amino acid + SMILES Multi-instance datasets containing both proteins and small molecules used the protein amino acid sequence and the molecular SMILES string.
- (v) Nucleotide + Amino acid Multi-instance datasets containing both nucleic acids and proteins used the nucleotide sequence and protein amino acid sequence.
- (vi) SMILES + Text Multi-instance datasets containing small molecules and other feature types used the molecular SMILES string and English text to represent the additional features, such as disease or cell line names and descriptions. Notable datasets in this category include Phase I, Phase II, and Phase III clinical trial datasets. These datasets contain information about the SMILES strings of candidate drugs, the names of targeted diseases for various clinical trial phases, and whether the trial ultimately received approval.
- (vii) Amino acid + Text Multi-instance datasets containing proteins and other feature types used the protein amino acid sequence and text to represent the other features. DisGeNET, which contained protein sequences and disease names, was the only dataset in this category.

For each dataset, data splits were constructed using TDC functions with recommended split methods (random, scaffold, cold-start, combination, temporal), which are indicated in Tables A.7 and A.8. For datasets in the ADMET, DrugCombo, or DTI DG leaderboards, we followed the leaderboard-specific instructions for generating splits with a seed of 1.

###### 3.2 Modeling

Base LLM Tx-LLM was initiated from PaLM-2 [59], the second generation of Google’s LLM trained using the Pathways accelerator orchestration system [60]. PaLM-2 models used in this work were trained at sizes S and M.

Few-shot prompting Few-shot prompting [16] involves providing example inputs and outputs in the prompt. Based on prior evidence showing the benefits of using a mixture of 0 and few-shot tasks [61], we constructed TxT as a mixture of of 70% 0-shot and 30% few-shot prompts with the number of few shots randomly chosen between 1 and 10. If the shots caused the prompt to exceed the maximum length, then the number of shots was reduced until the length was below the maximum. The shots were selected using random datapoints in the training dataset. On evaluation, we also considered nearest neighbor shots, but we strictly used random shots during training because the datapoints were often more similar within the training set than across training and test sets (Figure A.2).

Finetuning We finetuned a non-instruction tuned variant of PaLM-2 using TxT training data. We trained a single model across all TDC datasets using dataset mixture ratios proportional to the number of datapoints in each dataset. Tx-LLM generally refers to Tx-LLM (M) models trained across all TDC datasets. We explored various key ablations using subsets of TDC datasets for comparison using a smaller, Tx-LLM (S) due to constraints on computational resources. Hyperparameters for finetuning are listed in Table A.2.

###### 3.3 Evaluation

State-of-the-art performance SOTA values reported in this work are primarily derived from academic literature. For TDC datasets included in the ADMET, DrugCombo, and DTI DG leaderboards, SOTA values were obtained directly from the respective leaderboard. For datasets not featured in a leaderboard, SOTA values were determined through a manual literature review.

Few-shot prompting We evaluated the performance of Tx-LLM with 0 and few-shot prompting, varying the number of shots and whether the shots were selected based on random datapoints or nearest neighbor datapoints. Nearest neighbors were identified using representations of the query therapeutic(s). For small molecules, nearest neighbors were identified using Tanimoto similarities with Morgan fingerprints, whereas for amino acid and nucleotide sequences percent sequence identities were used. Morgan fingerprints and Tanimoto similarities were calculated using RDKit and Chemfp [62, 63], and multiple sequence alignments for percent sequence identities were calculated using Clustal Omega [64]. For validation, shots were selected from the training sets, whereas shots for test sets were selected from the combined training and validation sets.

Metrics We report Tx-LLM performances on TDC datasets using the preferred metric for each task as defined by [6]. Metrics for binary classification datasets include area under the receiver operating characteristic curve (AUROC), area under the precision-recall curve (AUPRC), and accuracy. Metrics for regression datasets include the Spearman correlation coefficient, Pearson correlation coefficient, mean absolute error (MAE), and mean squared error (MSE). The metric for the USPTO generation dataset is set accuracy, where for each reaction set overlaps between the generated reactants compared to the ground-truth reactants were assigned a score of 1 if there was perfect overlap, and 0 otherwise.

Statistical tests Given performances on TDC datasets from two models, we sought to determine whether one model was significantly better considering all datasets. To do this, we performed a non-parametric Wilcoxon signed-rank test on the performances from both models and report the p-value. In order to account for the differences in magnitudes for MAE and MSE metrics, we normalized all performances by the mean of the performances from both models. We also reversed the sign of MAEs and MSEs because lower MAEs and MSEs correspond to better performances.

Data contamination analysis PaLM-2 was trained on diverse data [59] that may contain information about

therapeutics in TDC. To study this, we analyzed the percent overlap between TDC dataset features and the PaLM-2 training data. To illustrate an example, consider the case of a TDC dataset containing SMILES strings for molecules paired with target amino acid sequences. For each TDC datapoint, two searches over the PaLM-2 training data were performed using (i) the full SMILES string and (ii) the full amino acid sequence, and the datapoint was considered overlapping if either the SMILES string or amino acid sequence were found in the PaLM-2 training data. The minimum match length was set to the SMILES/amino acid sequence length, up to a maximum of 512 characters due to infrastructural constraints. Our analysis looks for direct character overlaps and does not account for different representations of the same feature, such as molecules being represented as SMILES strings in TDC but being referred to by name in the PaLM-2 training data.

###### 4 Results

###### 4.1 Performance on TDC datasets

The performance of Tx-LLM on TDC datasets is summarized in Figure 3 and Tables A.7 and A.8. Out of 66 TDC tasks, Tx-LLM performed near or exceeding SOTA on 43 datasets. From these 43 datasets, Tx-LLM outperformed SOTA on 22 (12 binary classification datasets and 10 regression datasets) and performed near SOTA (defined as within 10% of SOTA) for another 21 (12 binary classification datasets and 9 regression datasets). Notably, these results were achieved using the same set of model weights without any task-specific optimizations.

Tx-LLM is particularly effective at combining SMILES and text TDC datasets containing features involving SMILES strings for molecules and text for other features (such as disease name or cell line name and description) tended to perform near or exceeding SOTA more frequently than datasets containing other features types (Figure 3 and Tables A.7 and A.8). To quantify this trend, we calculated the median relative difference of Tx-LLM performance from SOTA (defined as (Tx-LLM performance - SOTA) / SOTA) for each feature type (SMILES + Text, Nucleotide + Amino acid, SMILES, Amino acid, Amino acid + SMILES, Nucleotide) in Table A.6. In calculating the relative differences, signs were reversed for MAE and MSE metrics because lower MAEs and MSEs correspond to better performances. Amino acid + Text was not included because only the DisGeNET dataset, which we did not find a SOTA for, had this feature type.

SMILES + Text was the only feature type yielding a positive median relative difference, suggesting that it was the only feature type which tended to exceed SOTA on average. This performance may be due to the text representations for diseases and cell lines, both because text is a natural representation for a LLM and because the base LLM may have learned context about these in its pretraining. The ability to exceed SOTA may also be in part due to the difficulty in representing these features with non-LLM models. For example, SOTA models for the clinical trial approval datasets (phase 1, phase 2, and phase 3) represent diseases as nodes in an interaction graph [65], which may contain less information than context learned from LLM pretraining.

Overall, these results suggest that finetuned LLMs may be particularly effective for tasks involving both a drug and a target that can be represented in text (such as disease name or cell line name). In a sense, a finetuned LLM is an intermediate model between a domain-specific model, such as a GNN which is effective for representing molecules but less effective for other features, and a base LLM that does not understand molecular SMILES strings [39] but does contain diverse knowledge about physiology. Thus, a finetuned LLM may be an ideal model for tasks involving both.

Limitations of Tx-LLM for datasets without textual features In contrast, Tx-LLM underperformed SOTA on small molecule datasets solely using SMILES strings (Table A.6). This suggests that SOTA models representing molecules as graphs may be more effective than those relying only on SMILES strings, as SMILES strings have limitations such as non-uniqueness [66]. Additionally, LLM-generated SMILES strings may not follow proper SMILES grammar, which renders them invalid and possibly constitute hallucinations. In our work, datasets involving protein amino acid sequences perform similarly as those involving SMILES strings relative to SOTA models (Table A.6), which is less intuitive because proteins are naturally represented as sequences. One consideration is that SOTA models often encode evolutionary or structural information that is not learned by pretraining a LLM on natural language text. For example, BiComp-DTA uses evolutionary information from combining alignment-free and alignment-based methods in its protein encoding to predict

AUROC

AUPRC

Accuracy

1.0

1.0

1.0

0.9

0.8

0.8

Tx-LLMPerformance

Tx-LLMPerformance

Tx-LLMPerformance

0.8

0.6

0.6

0.7

0.4

0.4

0.6

0.2

0.2

0.5

0.0

0.0

0.4

0.4 0.5 0.6 0.7 0.8 0.9 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

SOTA

SOTA

SOTA

Generation Accuracy

Spearman

Pearson

1.0

1.0

1.0

0.8

0.8

0.8

Tx-LLMPerformance

Tx-LLMPerformance

Tx-LLMPerformance

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

SOTA

SOTA

SOTA

logMAE

logMSE

0.1

SMILES

0.2

1.0

Tx-LLMPerformance

Tx-LLMPerformance

Amino acid

0.3

0.5

Nucleotide

0.4

Amino acid + SMILES

0.0

0.5

Amino acid + Text

0.6

SMILES + Text

0.5

Nucleotide + Amino acid

0.7

1.0

0.8

1.0 0.5 0.0 0.5 1.0

0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1

SOTA

SOTA

- Figure 3 | Comparison of Tx-LLM’s performance with SOTA. Tx-LLM is evaluated on each dataset in TDC, and comparison with SOTA for different metrics is illustrated in panels. Datasets are colored by their feature types indicated in the legend, and marker sizes illustrate the number of data points in the task on a log scale. The larger shaded area in green indicates where Tx-LLM outperforms SOTA, while the narrower orange shaded area indicates where Tx-LLM is near SOTA (defined as within 10%). MAE and MSE values are log-transformed because the magnitudes of these values depend on the units of the outputs. Generation accuracy is the fraction of correct SMILES strings in the USPTO generation task.

drug-target dissociation constants, and Kinnings et al. [67] perform structure-based docking calculations to predict drug-target IC50s.

Evidence of data contamination affecting Tx-LLM performance was not observed We analyzed the percent overlap between TDC datasets and the PaLM-2 training data in Table A.17. The large majority of datasets do not have any overlap with the PaLM-2 training data, while 7 datasets have some overlap. For these 7 datasets, we also filter the test set to remove the overlapping datapoints, and the performances on the filtered test sets do not decrease relative to the unfiltered test sets. We caution that our contamination analysis looks for direct character overlaps and thus may miss overlapping features that are represented in different forms between the PaLM-2 training data and TDC datasets. In particular, molecules are represented as SMILES strings in TDC but likely referred to by name in natural language text used to train a generic LLM.

###### 4.2 Evidence of positive transfer across datasets with diverse drug types

To assess positive transfer in Tx-LLM, we trained a Tx-LLM (S) model only on TDC datasets containing molecules (excluding datasets involving other drug types such as proteins and nucleic acids) and compared the performance on small molecule datasets against the Tx-LLM (S) trained on all TDC datasets. The results in Figure 4 illustrate that the model trained on all datasets performs better than the model trained on small molecule datasets when evaluated on 43 out of 56 small molecule datasets.

To determine whether this improved performance was statistically significant, we performed a Wilcoxon signed-rank test on the small molecule dataset performances from both models (see Methods). The difference in performances between the model trained on all datasets and the model trained on small molecule datasets was highly significant (ρ = 1.4 × 10−5), showing evidence of positive transfer between datasets involving diverse drug types and datasets involving small molecules. This is interesting given that other drug types such as proteins and nucleic acids are represented using their sequences, which are quite different from the SMILES strings used to represent small molecules. This may be because small molecules and proteins are more similar to each other than to natural language, so additional training steps away from PaLM-2 are ultimately helpful.

A model trained only on datasets in the ADMET benchmark group, which was a subset of small molecule datasets, was also evaluated in Tables A.15 and A.16 in order to study positive transfer between small molecule datasets and ADMET datasets. For datasets in the ADMET benchmark, the model trained on small molecule datasets was the best model for only 4 ADMET datasets, while the model trained on ADMET datasets was the best model for 7 ADMET datasets. Interestingly, the model trained on all datasets was the best model for 11 ADMET datasets, suggesting that positive transfer between datasets with diverse drug types may be more impactful than positive transfer within small molecule datasets. This may support the use of a generalist model that expresses many drug types with the same string representation, rather than using separate representations for each drug type.

###### 4.3 Ablations

Model size and finetuning We compared performances on TDC datasets for models of multiple sizes (Tx-LLM (S) and Tx-LLM (M)) to examine the effect of model scale in Figures A.3 and A.4 and Tables A.9 and A.10. Additionally, we evaluated PaLM 2 (S) and PaLM 2 (M) as baseline generalist models in order to study the effect of biochemical domain finetuning. Tx-LLM (M) outperformed Tx-LLM (S) on 57 out of 66 TDC datasets, suggesting that scale is beneficial within our tested sizes (ρ = 1.65 × 10−7, Wilcoxon signed-rank test). Domain finetuning is also significant, as Tx-LLM outperforms PaLM-2 on 60 datasets for the S models (ρ = 1.86×10−10, Wilcoxon signed-rank test) and on 63 datasets for the M models (ρ = 3.58×10−11, Wilcoxon signed-rank test).

Number of shots and shot selection We compared varying the number of shots in the prompt as well as whether the shots are selected from random datapoints or nearest neighbor datapoints in order to study the extent to which Tx-LLM could exploit in-context learning in Figures A.3 and A.4 and Tables A.11 and A.12. We observed that the best performances were not clearly skewed toward a particular prompting strategy, and pairwise comparisons did not yield statistically significant differences (ρ > 0.05, Wilcoxon signed-rank test). This is consistent with previous observations that zero-shot and few-shot prompting yielded marginal

AUROC

AUPRC

Accuracy

1.0

1.0

1.0

0.9

0.8

0.8

0.8

0.6

0.6

0.7

Alldatasets

Alldatasets

Alldatasets

0.4

0.4

0.6

0.5

0.2

0.2

0.4

0.0

0.0

0.3

- -0.4

- -0.2

- -0.4

- -0.2

0.2

0.1

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

-0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0

-0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0

Molecule datasets

Molecule datasets

Molecule datasets

Generation Accuracy

Spearman

Pearson

1.0

1.0

1.0

0.8

0.8

0.8

0.6

0.6

0.6

Alldatasets

Alldatasets

Alldatasets

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

- -0.4

- -0.2

- -0.4

- -0.2

- -0.4

- -0.2

-0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0

-0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0

-0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0

Molecule datasets

Molecule datasets

Molecule datasets

logMAE

logMSE

1.2

1.5

1.0

1.2

0.8

SMILES

1.0

Alldatasets

Alldatasets

0.6

Amino acid + SMILES

0.8

SMILES + Text

0.4

0.5

0.2

0.2

0.0

0.0

-0.2

-0.2

-0.2 0.0 0.2 0.5 0.8 1.0 1.2 1.5

-0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2

Molecule datasets

Molecule datasets

- Figure 4 | Tx-LLM shows evidence of positive transfer across datasets with diverse drug types. Performance of Tx-LLM (S) finetuned and evaluated on small molecule datasets. “All datasets” indicates a Tx-LLM (S) model finetuned on all TDC datasets, and “Molecule datasets” indicates a Tx-LLM (S) model finetuned on datasets containing molecules (datasets involving other drug types such as proteins or nucleic acids are not included in training). Datasets are colored by their feature types indicated in the legend, and marker sizes illustrate the number of data points in the task on a log scale. The larger shaded area in green indicates where “All datasets” is better than “Molecule dataset” (showing evidence of positive transfer), while the narrower orange shaded area indicates where the performance of “Molecule datasets” is near the performance of “All dataset” (defined as within 10%). MAE and MSE values are log-transformed because the magnitudes of these values depend on the units of the outputs. Generation accuracy is the fraction of correct SMILES strings in the USPTO generation dataset.

differences in performance when evaluated on a single task [61].

Context removal We constructed a unique context for each TDC dataset that provides background information for Tx-LLM. We studied the effect of removing this context in Tables A.13 and A.14. We observed that removing the context reduced performance in 49 out of 66 datasets, which was also statistically significant (ρ = 4.9 × 10−6, Wilcoxon signed-rank test). Given that we trained a single set of weights for all TDC datasets, this suggests that providing context may be a useful way to allow generalist LLMs to become effective predictors for specific tasks. This was especially true for the ToxCast dataset, which contained numerous subtasks corresponding to predicting toxicity in various assays. Each ToxCast assay draws from the same set of small molecules and measures toxicity, but the toxicity label differs based on the particular assay used. Without providing assay-specific information in the context, the model would have no way of differentiating subtasks with different labels.

###### 5 Discussion

To the best of our knowledge, Tx-LLM is the first LLM trained on a wide variety of TDC datasets including small molecules, proteins, nucleic acids, cells, and diseases all in a single model. Interestingly, we found that including datasets without small molecules in our training, such as those only using proteins, enhances performance on small molecule datasets compared to models trained only on small molecule datasets. A previous generalist AI model finetuned for the biomedical domain (Med-PaLM M [12]) observed positive transfer between chest X-ray report generation and chest X-ray classification tasks. There, the positive transfer was between tasks that were relatively closely related, as both tasks involved chest X-rays. However, the positive transfer observed in Tx-LLM is between molecular tasks using SMILES strings and protein tasks using amino acid sequences, which are quite different.

Although LLMs have demonstrated strong general language and reasoning capabilities, their effectiveness has been notably limited when it comes to tasks requiring specialized knowledge in chemistry [39, 68]. For example, it was found that InstructGPT [68] rarely produced invalid SMILES strings, possibly suggesting that SMILES strings were encountered during training, but InstructGPT was unable to correctly predict the molecule name from the SMILES string. Thus, while non-finetuned LLMs may be able to capture grammar, understanding deeper connections is difficult, and domain finetuning may be necessary to achieve strong performance for therapeutics.

LLMs have previously been challenged by mathematical problems and regression, as it has been observed that smaller models have underperformed on mathematical benchmarks, which may be related to lossy compression of numbers in the model hidden states [69]. However, Tx-LLM is often effective at performing regression, exceeding SOTA or achieving near-SOTA performance on 19 out of 29 regression datasets. In other work, Gruver et al. [8] found that LLMs could be used for time series prediction, which was attributed to the LLMs’ proclivity for periodicity often found in time series. LLMs have also been implemented as universal regressors [9], and LLMs finetuned for the chemistry domain have been used to perform regression tasks on molecules [41]. Taken together, these suggest that applying LLMs to regression problems may be a promising area to explore in multiple domains.

Since Tx-LLM is a generalist model trained on a wide variety of tasks, we propose that it can have a future role for end-to-end therapeutic development spanning early-stage development such as target discovery to late-stage development such as clinical trial approval. Figure 2 illustrates an example for the case of developing small molecule drugs against type 2 diabetes. Here, Tx-LLM can be effective at identifying genes associated with type 2 diabetes, predicting binding affinities of many small molecules against the protein target, predicting toxicities of the selected small molecules, and finally predicting the probability of clinical trial approval. Tx-LLM can also be used for other tasks that are not illustrated, such as predicting drug permeability and drug synthesis reactions.

However, at this stage, Tx-LLM is in the research stage with scope for further improvement as the model is not an effective predictor for every task. In particular, the integration of the Gemini family of models [70] with Tx-LLM remains an interesting possibility to enhance performance. Experimental validation and subsequent screening steps also remain an essential, complementary part of the therapeutic development pipeline. Overall, we believe the methodology behind Tx-LLM (including the creation of TxT and the development of fine-tuned

LLMs) represents a promising step towards using AI to contextualize and enhance many aspects of therapeutic development in future.

One important limitation of our work is that Tx-LLM is not instruction-tuned to follow natural language because we were primarily interested in the accuracy of its predictions and did not want to limit this ability by also constraining Tx-LLM to follow natural language. Therefore, unlike LLMs finetuned for medical question answering [10–12], Tx-LLM is unable to explain its predictions to the user. This limits the benefits gained by training over a wide variety of tasks and may be an interesting area for future work.

Additionally, it is important to note that LLMs are trained on large datasets, which increase the potential for data contamination and may result in overestimating their generalization. For Tx-LLM, our data contamination analysis suggests that there is little overlap between the PaLM-2 training data and our evaluation data, and filtering the overlapping datapoints does not result in decreased performance. Nonetheless, our analysis does not account for different formats between the PaLM-2 training data and TDC data (e.g. using molecule names vs SMILES strings), and prospectively analyzing the performance of Tx-LLM over time may be another avenue to study the effect of data contamination.

###### 6 Conclusion

Therapeutic development is an expensive process that involves many potential types of therapeutics as well as many criteria for approval. AI models may be useful tools for reducing the failure rate of therapeutic development by providing initial screening for multiple aspects of the development pipeline. Although further development and validation is required, we believe Tx-LLM represents a notable advance towards a single generalist AI that can contextualize and aid many aspects of development ranging from target discovery to manufacturing.

###### Acknowledgments

This project was a collaboration between teams at Google Research and Google DeepMind. We thank David Belanger for the feedback and insight which significantly contributed to the enhancement of this report. We also thank Sami Lachgar, Lauren Winer, Maggie Shiels, Jessica Valdez, Jane Park, Jon Small, Aaron Abood, Rishad Patel, Uchechi Okereke, Annisah Um’rani, Alan Karthikesalingam, Anil Palepu, and Juraj Gottweis for their valuable insights, technical support and feedback during our research. We are also grateful to Zoubin Ghahramani, Raia Hadsell, Jon Shlens and Joelle Barral for their support during the course of this project.

###### Data availability

Datasets utilized for developing, benchmarking, and evaluation of Tx-LLM are publicly accessible with appropriate permissions. The Therapeutics Data Commons (TDC) datasets are accessible via their website.

###### Competing interests

This study was funded by Alphabet Inc and/or a subsidiary thereof (‘Alphabet’). T. T., C. S., D. F., V. N., and S. A. are employees of Alphabet and may own stock as part of the standard compensation package.

###### References

- 1. Sun, D., Gao, W., Hu, H. & Zhou, S. Why 90% of clinical drug development fails and how to improve it? en. Acta Pharmaceutica Sinica B 12, 3049–3062. issn: 22113835. https://linkinghub.elsevier.com/retrieve/pii/S2211383522000521

(2024) (July 2022).

- 2. Hinkson, I. V., Madej, B. & Stahlberg, E. A. Accelerating Therapeutics for Opportunities in Medicine: A Paradigm Shift in Drug Discovery. Frontiers in Pharmacology 11, 770. issn: 1663-9812. https://www.frontiersin.org/article/10.3389/fphar. 2020.00770/full (2024) (June 2020).
- 3. Lin, A., Giuliano, C. J., Palladino, A., John, K. M., Abramowicz, C., Yuan, M. L., Sausville, E. L., Lukow, D. A., Liu, L., Chait, A. R., Galluzzo, Z. C., Tucker, C. & Sheltzer, J. M. Off-target toxicity is a common mechanism of action of cancer drugs undergoing clinical trials. en. Science Translational Medicine 11, eaaw8412. issn: 1946-6234, 1946-6242. https://www.science.org/doi/10.1126/scitranslmed.aaw8412 (2024) (Sept. 2019).
- 4. Sadybekov, A. V. & Katritch, V. Computational approaches streamlining drug discovery. en. Nature 616, 673–685. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-023-05905-z (2024) (Apr. 2023).
- 5. Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C., Sun, J. & Zitnik, M. Artificial intelligence foundation for therapeutic science. en. Nature Chemical Biology 18, 1033–1036. issn: 1552-4450, 1552-4469. https://www.nature.com/articles/s41589-022-01131-2 (2024) (Oct. 2022).
- 6. Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C., Sun, J. & Zitnik, M. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development 2021. arXiv: 2102.09548 [cs.LG].
- 7. Robinson, J. & Wingate, D. Leveraging Large Language Models for Multiple Choice Question Answering in The Eleventh International Conference on Learning Representations (2023). https://openreview.net/forum?id=yKbprarjc5B.
- 8. Gruver, N., Finzi, M., Qiu, S. & Wilson, A. G. Large Language Models Are Zero-Shot Time Series Forecasters 2023. arXiv: 2310.07820 [cs.LG].
- 9. Song, X., Li, O., Lee, C., Yang, B., Peng, D., Perel, S. & Chen, Y. OmniPred: Language Models as Universal Regressors

2024. arXiv: 2402.14547 [cs.LG].

- 10. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., Payne, P., Seneviratne, M., Gamble, P., Kelly, C., Babiker, A., Schärli, N., Chowdhery, A., Mansfield, P., Demner-Fushman, D., Agüera Y Arcas, B., Webster, D., Corrado, G. S., Matias, Y., Chou, K., Gottweis, J., Tomasev, N., Liu, Y., Rajkomar, A., Barral, J., Semturs, C., Karthikesalingam, A. & Natarajan, V. Large language models encode clinical knowledge. en. Nature 620, 172–180. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-023-06291-2 (2024) (Aug. 2023).
- 11. Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Hou, L., Clark, K., Pfohl, S., Cole-Lewis, H., Neal, D., Schaekermann, M., Wang, A., Amin, M., Lachgar, S., Mansfield, P., Prakash, S., Green, B., Dominowska, E., y Arcas, B. A., Tomasev, N., Liu, Y., Wong, R., Semturs, C., Mahdavi, S. S., Barral, J., Webster, D., Corrado, G. S., Matias, Y., Azizi, S., Karthikesalingam, A. & Natarajan, V. Towards Expert-Level Medical Question Answering with Large Language Models

2023. arXiv: 2305.09617 [cs.CL].

- 12. Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.-C., Carroll, A., Lau, C., Tanno, R., Ktena, I., Mustafa, B., Chowdhery, A., Liu, Y., Kornblith, S., Fleet, D., Mansfield, P., Prakash, S., Wong, R., Virmani, S., Semturs, C., Mahdavi, S. S., Green, B., Dominowska, E., y Arcas, B. A., Barral, J., Webster, D., Corrado, G. S., Matias, Y., Singhal, K., Florence, P., Karthikesalingam, A. & Natarajan, V. Towards Generalist Biomedical AI 2023. arXiv: 2307.14334 [cs.CL].
- 13. Zhang, W., Wang, Q., Kong, X., Xiong, J., Ni, S., Cao, D., Niu, B., Chen, M., Zhang, R., Wang, Y., Zhang, L., Li, X., Xiong, Z., Shi, Q., Huang, Z., Fu, Z. & Zheng, M. Fine-tuning Large Language Models for Chemical Text Mining preprint (Chemistry, Feb. 2024). https://chemrxiv.org/engage/chemrxiv/article-details/65baa07b9138d2316124f224 (2024).
- 14. Jablonka, K. M., Schwaller, P., Ortega-Guerrero, A. & Smit, B. Leveraging large language models for predictive chemistry. en. Nature Machine Intelligence 6, 161–169. issn: 2522-5839. https://www.nature.com/articles/s42256-023-00788-1 (2024) (Feb. 2024).
- 15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. & Polosukhin, I. Attention Is All You Need 2023. arXiv: 1706.03762 [cs.CL].
- 16. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I. & Amodei, D. Language Models are Few-Shot Learners in Advances in Neural Information Processing Systems (eds Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. & Lin, H.) 33 (Curran Associates, Inc., 2020), 1877–1901. https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
- 17. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. & Liu, P. J. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research 21, 1–67. http://jmlr.org/papers/v21/20-074.html (2020).
- 18. Torng, W. & Altman, R. B. Graph Convolutional Neural Networks for Predicting Drug-Target Interactions. en. Journal of Chemical Information and Modeling 59, 4131–4149. issn: 1549-9596, 1549-960X. https://pubs.acs.org/doi/10.1021/acs. jcim.9b00628 (2024) (Oct. 2019).
- 19. Stärk, H., Ganea, O., Pattanaik, L., Barzilay, R. & Jaakkola, T. Equibind: Geometric deep learning for drug binding structure prediction in International conference on machine learning (2022), 20503–20521.
- 20. Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., Li, Z., Luo, X., Chen, K., Jiang, H. & Zheng, M. Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. en. Journal of Medicinal Chemistry 63, 8749–8760. issn: 0022-2623, 1520-4804. https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959 (2024) (Aug. 2020).
- 21. Heid, E. & Green, W. H. Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction. en. Journal of Chemical Information and Modeling 62, 2101–2110. issn: 1549-9596, 1549-960X. https://pubs.acs.org/doi/10.1021/acs.jcim.1c00975 (2024) (May 2022).
- 22. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K. & Barzilay, R. Analyzing Learned Molecular Representations for Property Prediction. en. Journal of Chemical Information and Modeling 59, 3370–3388. issn: 1549-9596, 1549-960X. https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237 (2024) (Aug. 2019).

- 23. Morrone, J. A., Weber, J. K., Huynh, T., Luo, H. & Cornell, W. D. Combining Docking Pose Rank and Structure with Deep Learning Improves Protein–Ligand Binding Mode Prediction over a Baseline Docking Approach. Journal of Chemical Information and Modeling 60. PMID: 32077698, 4170–4179. eprint: https://doi.org/10.1021/acs.jcim.9b00927. https://doi.org/10.1021/acs.jcim.9b00927 (2020).
- 24. Mohr, B., Shmilovich, K., Kleinwächter, I. S., Schneider, D., Ferguson, A. L. & Bereau, T. Data-driven discovery of cardiolipinselective small molecules by computational active learning. en. Chemical Science 13, 4498–4511. issn: 2041-6520, 2041-6539. http://xlink.rsc.org/?DOI=D2SC00116K (2024) (2022).
- 25. Stokes, J. M., Yang, K., Swanson, K., Jin, W., Cubillos-Ruiz, A., Donghia, N. M., MacNair, C. R., French, S., Carfrae, L. A., Bloom-Ackermann, Z., Tran, V. M., Chiappino-Pepe, A., Badran, A. H., Andrews, I. W., Chory, E. J., Church, G. M., Brown, E. D., Jaakkola, T. S., Barzilay, R. & Collins, J. J. A Deep Learning Approach to Antibiotic Discovery. en. Cell 180, 688–702.e13. issn: 00928674. https://linkinghub.elsevier.com/retrieve/pii/S0092867420301021 (2024) (Feb. 2020).
- 26. Ran, B., Chen, L., Li, M., Han, Y. & Dai, Q. Drug-Drug Interactions Prediction Using Fingerprint Only. en. Computational and Mathematical Methods in Medicine 2022 (ed Wei, L.) 1–14. issn: 1748-6718, 1748-670X. https://www.hindawi.com/ journals/cmmm/2022/7818480/ (2024) (May 2022).
- 27. Tayyebi, A., Alshami, A. S., Rabiei, Z., Yu, X., Ismail, N., Talukder, M. J. & Power, J. Prediction of organic compound aqueous solubility using machine learning: a comparison study of descriptor-based and fingerprints-based models. en. Journal of Cheminformatics 15, 99. issn: 1758-2946. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00752-6

(2024) (Oct. 2023).

- 28. Belenahalli Shekarappa, S., Kandagalla, S. & Lee, J. Development of machine learning models based on molecular fingerprints for selection of small molecule inhibitors against <span style="font-variant:small-caps;">JAK2</span> protein. en. Journal of Computational Chemistry 44, 1493–1504. issn: 0192-8651, 1096-987X. https://onlinelibrary.wiley.com/doi/10.1002/jcc. 27103 (2024) (June 2023).
- 29. Louie, R. H. Y., Kaczorowski, K. J., Barton, J. P., Chakraborty, A. K. & McKay, M. R. Fitness landscape of the human immunodeficiency virus envelope protein that is targeted by antibodies. Proceedings of the National Academy of Sciences 115, E564–E573. eprint: https://www.pnas.org/doi/pdf/10.1073/pnas.1717765115. https://www.pnas.org/doi/abs/10.1073/ pnas.1717765115 (2018).
- 30. Gfeller, D., Schmidt, J., Croce, G., Guillaume, P., Bobisse, S., Genolet, R., Queiroz, L., Cesbron, J., Racle, J. & Harari, A. Improved predictions of antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0 reveal potent SARS-CoV-2 CD8+ T-cell epitopes. en. Cell Systems 14, 72–83.e5. issn: 24054712. https://linkinghub.elsevier.com/ retrieve/pii/S2405471222004707 (2024) (Jan. 2023).
- 31. Motmaen, A., Dauparas, J., Baek, M., Abedi, M. H., Baker, D. & Bradley, P. Peptide-binding specificity prediction using fine-tuned protein structure prediction networks. Proceedings of the National Academy of Sciences 120, e2216697120. eprint: https://www.pnas.org/doi/pdf/10.1073/pnas.2216697120. https://www.pnas.org/doi/abs/10.1073/pnas.2216697120

(2023).

- 32. Wang, Y., Liu, Y., Wang, S., Liu, Z., Gao, Y., Zhang, H. & Dong, L. ATTfold: RNA Secondary Structure Prediction With Pseudoknots Based on Attention Mechanism. Frontiers in Genetics 11, 612086. issn: 1664-8021. https://www.frontiersin. org/articles/10.3389/fgene.2020.612086/full (2024) (Dec. 2020).
- 33. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S. A. A., Ballard, A. J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., Back, T., Petersen, S., Reiman, D., Clancy, E., Zielinski, M., Steinegger, M., Pacholska, M., Berghammer, T., Bodenstein, S., Silver, D., Vinyals, O., Senior, A. W., Kavukcuoglu, K., Kohli, P. & Hassabis, D. Highly accurate protein structure prediction with AlphaFold. en. Nature 596, 583–589. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-021-03819-2 (2024) (Aug. 2021).
- 34. Tunyasuvunakool, K., Adler, J., Wu, Z., Green, T., Zielinski, M., Žídek, A., Bridgland, A., Cowie, A., Meyer, C., Laydon, A., Velankar, S., Kleywegt, G. J., Bateman, A., Evans, R., Pritzel, A., Figurnov, M., Ronneberger, O., Bates, R., Kohl, S. A. A., Potapenko, A., Ballard, A. J., Romera-Paredes, B., Nikolov, S., Jain, R., Clancy, E., Reiman, D., Petersen, S., Senior, A. W., Kavukcuoglu, K., Birney, E., Kohli, P., Jumper, J. & Hassabis, D. Highly accurate protein structure prediction for the human proteome. en. Nature 596, 590–596. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-021-03828-1

(2024) (Aug. 2021).

- 35. Senior, A. W., Evans, R., Jumper, J., Kirkpatrick, J., Sifre, L., Green, T., Qin, C., Žídek, A., Nelson, A. W. R., Bridgland, A., Penedones, H., Petersen, S., Simonyan, K., Crossan, S., Kohli, P., Jones, D. T., Silver, D., Kavukcuoglu, K. & Hassabis, D. Improved protein structure prediction using potentials from deep learning. en. Nature 577, 706–710. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-019-1923-7 (2024) (Jan. 2020).
- 36. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., Bodenstein, S. W., Evans, D. A., Hung, C.-C., O’Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z.,

Žemgulyt˙e, A., Arvaniti, E., Beattie, C., Bertolli, O., Bridgland, A., Cherepanov, A., Congreve, M., Cowen-Rivers, A. I., Cowie, A., Figurnov, M., Fuchs, F. B., Gladman, H., Jain, R., Khan, Y. A., Low, C. M. R., Perlin, K., Potapenko, A., Savy, P., Singh, S., Stecula, A., Thillaisundaram, A., Tong, C., Yakneen, S., Zhong, E. D., Zielinski, M., Žídek, A., Bapst, V., Kohli, P., Jaderberg, M., Hassabis, D. & Jumper, J. M. Accurate structure prediction of biomolecular interactions with AlphaFold 3. en. Nature. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-024-07487-w (2024) (May 2024).

- 37. Ren, F., Ding, X., Zheng, M., Korzinkin, M., Cai, X., Zhu, W., Mantsyzov, A., Aliper, A., Aladinskiy, V., Cao, Z., Kong, S., Long, X., Man Liu, B. H., Liu, Y., Naumov, V., Shneyderman, A., Ozerov, I. V., Wang, J., Pun, F. W., Polykovskiy, D. A., Sun, C., Levitt, M., Aspuru-Guzik, A. & Zhavoronkov, A. AlphaFold accelerates artificial intelligence powered drug discovery: efficient discovery of a novel CDK20 small molecule inhibitor. en. Chemical Science 14, 1443–1452. issn: 2041-6520, 2041-6539. https://xlink.rsc.org/?DOI=D2SC05709C (2024) (2023).
- 38. Higgins, M. K. Can We AlphaFold Our Way Out of the Next Pandemic? en. Journal of Molecular Biology 433, 167093. issn: 00222836. https://linkinghub.elsevier.com/retrieve/pii/S002228362100317X (2024) (Oct. 2021).
- 39. White, A. D., Hocky, G. M., Gandhi, H. A., Ansari, M., Cox, S., Wellawatte, G. P., Sasmal, S., Yang, Z., Liu, K., Singh, Y. & Peña Ccoa, W. J. Assessment of chemistry knowledge in large language models that generate code. Digital Discovery 2, 368–376. http://dx.doi.org/10.1039/D2DD00087C (2 2023).
- 40. Guo, T., Guo, K., Nan, B., Liang, Z., Guo, Z., Chawla, N. V., Wiest, O. & Zhang, X. What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks in Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2023). https://openreview.net/forum?id=1ngbR3SZHW.

- 41. Yu, B., Baker, F. N., Chen, Z., Ning, X. & Sun, H. LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset 2024. arXiv: 2402.09391 [cs.AI].
- 42. M. Bran, A., Cox, S., Schilter, O., Baldassari, C., White, A. D. & Schwaller, P. Augmenting large language models with chemistry tools. en. Nature Machine Intelligence. issn: 2522-5839. https://www.nature.com/articles/s42256-024-00832-8

(2024) (May 2024).

- 43. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J. & Fergus, R. Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences. PNAS. https: //www.biorxiv.org/content/10.1101/622803v4 (2019).
- 44. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., Dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S. & Rives, A. Evolutionary-scale prediction of atomic-level protein structure with a language model. en. Science 379, 1123–1130. issn: 0036-8075, 1095-9203. https://www.science.org/doi/10.1126/science. ade2574 (2024) (Mar. 2023).
- 45. Alley, E. C., Khimulya, G., Biswas, S., AlQuraishi, M. & Church, G. M. Unified rational protein engineering with sequence-based deep representation learning. en. Nature Methods 16, 1315–1322. issn: 1548-7091, 1548-7105. https: //www.nature.com/articles/s41592-019-0598-1 (2024) (Dec. 2019).
- 46. Ferruz, N., Schmidt, S. & Höcker, B. ProtGPT2 is a deep unsupervised language model for protein design. en. Nature Communications 13, 4348. issn: 2041-1723. https://www.nature.com/articles/s41467-022-32007-7 (2024) (July 2022).
- 47. Hie, B., Zhong, E. D., Berger, B. & Bryson, B. Learning the language of viral evolution and escape. Science 371, 284–288. eprint: https://www.science.org/doi/pdf/10.1126/science.abd7331. https://www.science.org/doi/abs/10.1126/science. abd7331 (2021).
- 48. Hie, B. L., Shanker, V. R., Xu, D., Bruun, T. U. J., Weidenbacher, P. A., Tang, S., Wu, W., Pak, J. E. & Kim, P. S. Efficient evolution of human antibodies from general protein language models. en. Nature Biotechnology 42, 275–283. issn: 1087-0156, 1546-1696. https://www.nature.com/articles/s41587-023-01763-2 (2024) (Feb. 2024).
- 49. Zhuo, L., Chi, Z., Xu, M., Huang, H., Zheng, H., He, C., Mao, X.-L. & Zhang, W. ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training 2024. arXiv: 2403.07920 [q-bio.BM].
- 50. Pei, Q., Zhang, W., Zhu, J., Wu, K., Gao, K., Wu, L., Xia, Y. & Yan, R. BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations 2024. arXiv: 2310.07276 [cs.CL].
- 51. Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N. & Wang, B. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. en. Nature Methods. issn: 1548-7091, 1548-7105. https://www.nature.com/ articles/s41592-024-02201-0 (2024) (Feb. 2024).
- 52. Chen, Y. & Zou, J. GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT. bioRxiv. eprint: https://www.biorxiv.org/content/early/2024/03/05/2023.10.16.562533.full.pdf (2024).
- 53. Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., Mantineo, H., Brydon, E. M., Zeng, Z., Liu, X. S. & Ellinor, P. T. Transfer learning enables predictions in network biology. en. Nature 618, 616–624. issn: 0028-0836, 1476-4687. https://www.nature.com/articles/s41586-023-06139-9 (2024) (June 2023).
- 54. Kim, S., Chen, J., Cheng, T., Gindulyte, A., He, J., He, S., Li, Q., Shoemaker, B. A., Thiessen, P. A., Yu, B., et al. PubChem 2023 update. Nucleic acids research 51, D1373–D1380 (2023).
- 55. ToxCast and Tox21 Summary Files from invitrodb_v3. https://www.epa.gov/chemical-research/toxicity-forecastertoxcasttm-data (2015).
- 56. Lowe, D. Chemical reactions from US patents (1976-Sep2016). https://figshare.com/articles/dataset/Chemical_reactions_ from_US_patents_1976-Sep2016_/5104873 (June 2017).
- 57. Nielsen, M. & Andreatta, M. NetMHCpan-3.0; improved prediction of binding to MHC class I molecules integrating information from multiple receptor and peptide length datasets. en. Genome Medicine 8, 33. issn: 1756-994X. https: //genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0288-x (2024) (Dec. 2016).
- 58. Jensen, K. K., Andreatta, M., Marcatili, P., Buus, S., Greenbaum, J. A., Yan, Z., Sette, A., Peters, B. & Nielsen, M. Improved methods for predicting peptide binding affinity to <span style="font-variant:small-caps;">MHC</span> class <span style="font-variant:small-caps;">II</span> molecules. en. Immunology 154, 394–406. issn: 0019-2805, 1365-2567. https://onlinelibrary.wiley.com/doi/10.1111/imm.12889 (2024) (July 2018).
- 59. Google et al. PaLM 2 Technical Report 2023. arXiv: 2305.10403 [cs.CL].
- 60. Barham, P., Chowdhery, A., Dean, J., Ghemawat, S., Hand, S., Hurt, D., Isard, M., Lim, H., Pang, R., Roy, S., Saeta, B., Schuh, P., Sepassi, R., Shafey, L. E., Thekkath, C. A. & Wu, Y. Pathways: Asynchronous Distributed Dataflow for ML

2022. arXiv: 2203.12533 [cs.DC].

- 61. Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J. & Roberts, A. The Flan Collection: Designing Data and Methods for Effective Instruction Tuning 2023. arXiv: 2301.13688 [cs.AI].
- 62. Landrum, G. RDKit: Open-Source Cheminformatics Software. https://github.com/rdkit/rdkit/releases/tag/Release_2016_ 09_4 (2016).
- 63. Dalke, A. The chemfp project. en. Journal of Cheminformatics 11, 76. issn: 1758-2946. https://jcheminf.biomedcentral. com/articles/10.1186/s13321-019-0398-8 (2024) (Dec. 2019).
- 64. Sievers, F., Wilm, A., Dineen, D., Gibson, T. J., Karplus, K., Li, W., Lopez, R., McWilliam, H., Remmert, M., Söding, J., Thompson, J. D. & Higgins, D. G. Fast, scalable generation of high-quality protein multiple sequence alignments using Clustal Omega. en. Molecular Systems Biology 7, 539. issn: 1744-4292, 1744-4292. https://www.embopress.org/doi/10. 1038/msb.2011.75 (2024) (Jan. 2011).
- 65. Fu, T., Huang, K., Xiao, C., Glass, L. M. & Sun, J. HINT: Hierarchical interaction network for clinical-trial-outcome predictions. Patterns 3, 100445. issn: 2666-3899. https://www.sciencedirect.com/science/article/pii/S2666389922000186

(2022).

- 66. Wigh, D. S., Goodman, J. M. & Lapkin, A. A. A review of molecular representation in the age of machine learning. WIREs Computational Molecular Science 12, e1603. eprint: https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1603. https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1603 (2022).
- 67. Kinnings, S. L., Liu, N., Tonge, P. J., Jackson, R. M., Xie, L. & Bourne, P. E. A Machine Learning-Based Method To Improve Docking Scoring Functions and Its Application to Drug Repurposing. en. Journal of Chemical Information and Modeling 51, 408–419. issn: 1549-9596, 1549-960X. https://pubs.acs.org/doi/10.1021/ci100369f (2024) (Feb. 2011).

- 68. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J. & Lowe, R. Training language models to follow instructions with human feedback 2022. arXiv: 2203.02155 [cs.CL].
- 69. Zhu, F., Dai, D. & Sui, Z. Language Models Understand Numbers, at Least Partially 2024. arXiv: 2401.03735 [cs.CL].
- 70. Team, G. et al. Gemini: A Family of Highly Capable Multimodal Models 2024. arXiv: 2312.11805 [cs.CL].

#### Appendix

###### A.1 Additional data details

- Table A.1 | Excluded TDC datasets and reasons for exclusion. Dataset name Reason for exclusion QM7b Prediction of quantum properties is not closely related to therapeutic development. QM8 Prediction of quantum properties is not closely related to therapeutic development. QM9 Prediction of quantum properties is not closely related to therapeutic development. IEDB Jespersen Amount of data is small, and token prediction is more difficult to implement in a LLM

than binary classification. PDB Jespersen Amount of data is small, and token prediction is more difficult to implement in a LLM

than binary classification. DrugBank DDI Large number of possible labels is difficult to implement in a LLM. TWOSIDES Large number of possible labels is difficult to implement in a LLM. USPTO Catalyst Large number of possible labels is difficult to implement in a LLM. MOSES No clear metric. ZINC No clear metric. ChEMBL No clear metric. USPTO 50K Subset of USPTO. USPTO Reaction Same dataset as USPTO.

- Table A.2 | Hyperparameters used for Tx-LLM finetuning. Hyperparameter Tx-LLM (S) Tx-LLM (M)

Learning rate 3 × 10−5 1 × 10−4 Dropout rate 0.05 0.15 Batch size 256 256 Max token input length 2048 2048 Max token output length 512 512

- Table A.3 | Example of prompts for binary classification datasets.

Instructions: Answer the following question about drug properties. Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system. Question: Given a drug SMILES string, predict whether it (A) does not cross the BBB (B) crosses the BBB Drug SMILES: CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21 Answer: (B)

Instructions: Answer the following question about peptide-MHC binding. Context: In the human body, T cells monitor the existing peptides and trigger an immune response if the peptide is foreign. To decide whether or not if the peptide is not foreign, the peptide must bind to a major histocompatibility complex (MHC) molecule. Therefore, predicting peptide-MHC binding affinity is pivotal for determining immunogenicity. In some experiments, the peptide binding is measured against cells that express multiple MHCs, so the peptide could be binding any one of the possible MHCs. Class 1 MHC molecules bind to peptides that are usually 8-14 amino acids long and activate CD8 T cells. Question: Given the amino acid sequence of the peptide and possible pseudo amino acid sequences of MHC 1, predict whether the peptide (A) does not bind to any of the MHCs (B) binds to any of the MHCs Peptide amino acid sequence: QLADETLLKV Possible MHC pseudosequences: YFAMYGEKVAHTHVDTLYVRYHYYTWAEWAYTWY Answer: (B)

Instructions: Answer the following question about miRNA protein interactions. Context: MicroRNAs (miRNAs) are, small non-coding RNAs with 18–25 nucleotides, which are central regulators at the post-transcriptional level in both animals and plants. Perfect or near-perfect complementary binding of miRNAs and their target mRNA negatively regulates gene expression by accelerating mRNA degradation or suppressing mRNA translation. Question: Given the miRNA mature sequence and target amino acid sequence, predict whether (A) the miRNA and target do not interact (B) the miRNA and target interact miRNA sequence: UUCCUGUCAGCCGUGGGUGCC Target amino acid sequence: MSVNMDELRHQVMINQFVLAAGCAADQAKQLLQAAHWQFETALSTFF QETNIPNSHHHHQMMCTPSNTPATPPNFPDALAMFSKLRASEGLQSSNSPMTAAACSPPANFSPFWASSPPSHQAPWIPPSSPTTFHHLHRPQPTWPPGAQQGGAQQKAMAAMDGQR Answer: (A)

Instructions: Answer the following question about clinical trials. Context: Clinical trial is the most time and cost-consuming step in the drug discovery process. Phase 1 clinical trials test the safety and basic properties of a new drug or treatment in a small group of people for the first time. Optimizing and designing trials with machine learning could drastically lead to the speedup of delivery of life-saving therapeutics to patients. Clinical trial outcome prediction is a machine learning task that aims to forecast the outcome of clinical trials, such as the approval rate of a drug or treatment. It utilizes various clinical trial features, including the drug’s molecular structure and patient disease. Question: Given a drug SMILES string and disease, predict if the phase 1 trial (A) would not be approved (B) would be approved Drug SMILES: COC1=NC(N)=NC2=C1N=CN2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O Disease: Chronic myeloproliferative disease Answer: (A)

- Table A.4 | Example of prompts for regression and generation datasets.

Instructions: Answer the following question about drug properties. Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue. Question: Given a drug SMILES string, predict its normalized Caco-2 cell effective permeability from 000 to 1000, where 000 is minimum permeability and 1000 is maximum permeability. Drug SMILES: O=C(O)COC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl Answer: 788

Instructions: Answer the following question about drug responses. Context: The same drug compound could have various levels of responses in different patients. To design drug for individual or a group with certain characteristics is the central goal of precision medicine. In experiments, IC50s of drugs were measured against cancer cell lines. Question: Given a drug SMILES string and a cell line description, predict the normalized drug sensitivity from 000 to 1000, where 000 is minimum drug sensitivity and 1000 is maximum drug sensitivity. Drug SMILES: CN1C=C(C2=CC=CC=C21)/C=C\3/C4=C(C=CC=N4)NC3=O Cell line description: SNU-1, stomach cell sourced from cancer Answer: 615

Instructions: Answer the following question about drug target interactions. Context: Drug-target binding is the physical interaction between a drug and a specific biological molecule, such as a protein or enzyme. This interaction is essential for the drug to exert its pharmacological effect. The strength of the drug-target binding is determined by the binding affinity, which is a measure of how tightly the drug binds to the target. Kd is the dissociation constant of a drug-target complex. It is the concentration of drug at which half of the drug-target complexes have dissociated. A lower Kd value indicates a stronger binding affinity. Question: Given the target amino acid sequence and compound SMILES string, predict their normalized binding affinity Kd from 000 to 1000, where 000 is minimum Kd and 1000 is maximum Kd. Drug SMILES: O=S(=O)(O)c1cccc2cccc(Nc3ccccc3)c12 Target amino acid sequence: MATVQQLEGRWRLVDSKGFDEYMKELGVGIALRKMGAMAKPDC IITCDGKNLTIKTESTLKTTQFSCTLGEKFEETTADGRKTQTVCNFTDGALVQHQEWDGKESTITRKLKDGKLVVECVMNNVTCTRIYEKVE Answer: 397

Instructions: Answer the following question about reactions. Context: Retrosynthesis is the process of finding a set of reactants that can synthesize a target molecule, i.e., product, which is a fundamental task in drug manufacturing. The target is recursively transformed into simpler precursor molecules until commercially available "starting" molecules are identified. In a data sample, there is only one product molecule, reactants can be one or multiple molecules. Question: Given a product SMILES string, predict the reactant SMILES string. Product SMILES: [CH2:12]1[C:7]2([CH2:6][CH2:5][O:15][CH2:1][CH2:8]2)[CH2:13][CH2:14][O:10][C:11]1=[O:17] Answer: [CH:1]12B[CH:5]([CH2:6][CH2:7][CH2:8]1)CCC2.[O:10]1[CH2:14][CH2:13][CH2:12] [CH2:11]1.[OH-:15].[Na+].[OH:17]O.Cl

- Table A.5 | Example of a 10-shot prompt for a binary classification dataset.

Instructions: Answer the following question about drug properties. Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system. Question: Given a drug SMILES string, predict whether it (A) does not cross the BBB (B) crosses the BBB Drug SMILES: CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: CN1C(=O)CN=C(c2ccccc2F)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: CN1C(=S)CN=C(c2ccccc2)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: CP(C)(=O)CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: CN1C(=O)CN=C(c2ccccc2)c2cc([N+](=O)[O-])ccc21 Answer: (B) Drug SMILES: CCN(CC)CCN1C(=O)CN=C(c2ccccc2F)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1CC1CC1 Answer: (B) Drug SMILES: C#CCN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21 Answer: (B) Drug SMILES: O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1CC(F)(F)F Answer: (B) Drug SMILES: CCS(=O)(=O)CCN1C(=O)CN=C(c2ccccc2F)c2cc(Cl)ccc21 Answer: (B)

Drug SMILES: CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21 Answer: (B)

- Figure A.1 | Distribution of TDC dataset sizes, aggregated over train, validation, and test sets. For datasets containing multiple subtasks, such as ToxCast which contains data for more than 600 different assays, the dataset size is calculated by summing over the sizes for each subtask.

Train Validation Test

2.5

| |
|---|

| |
|---|

2.0

1.5

PDF

1.0

0.5

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Tanimoto similarity of 10 nearest neighbors

- Figure A.2 | Distribution of the Tanimoto similarities for the 10 nearest neighbors in the AMES dataset. Nearest neighbors are calculated from the training set for training and validation sets, and from both the training and validation sets for the test set.

###### A.2 Additional results

- Table A.6 | Median relative difference of Tx-LLM (M) performance from SOTA for datasets grouped by feature type. The relative difference is defined as (Tx-LLM performance - SOTA) / SOTA. The signs were reversed for MAE and MSE metrics because lower MAE and MSE values correspond to better performances.

Feature type Median relative difference of Tx-LLM performance from SOTA

SMILES + Text 0.048 Nucleotide + Amino acid -0.007 Amino acid -0.080 SMILES -0.082 Amino acid + SMILES -0.482 Nucleotide -0.888

- Table A.7 | Tx-LLM (M) performance compared with SOTA for each binary classification dataset, along with the feature types and metric type. Performances that are better than SOTA are bolded.

Dataset name Feature type Split method Metric SOTA Tx-LLM

PAMPA NCATS SMILES Scaffold AUROC 0.900 [1] 0.668 HIA Hou SMILES Scaffold AUROC 0.988 [2] 0.990 Pgp Broccatelli SMILES Scaffold AUROC 0.935 [3] 0.939 Bioavailability Ma SMILES Scaffold AUROC 0.748 [4] 0.702 BBB Martins SMILES Scaffold AUROC 0.915 [5] 0.882 CYP2C19 Veith SMILES Scaffold AUROC 0.890 [6] 0.895 CYP2D6 Veith SMILES Scaffold AUPRC 0.739 [7] 0.659 CYP3A4 Veith SMILES Scaffold AUPRC 0.904 [7] 0.840 CYP1A2 Veith SMILES Scaffold AUPRC 0.900 [8] 0.914 CYP2C9 Veith SMILES Scaffold AUPRC 0.839 [7] 0.788 CYP2C9 Substrate CarbonMangels SMILES Scaffold AUPRC 0.441 [9] 0.436 CYP2D6 Substrate CarbonMangels SMILES Scaffold AUPRC 0.736 [7] 0.600 CYP3A4 Substrate CarbonMangels SMILES Scaffold AUROC 0.662 [10] 0.647 hERG SMILES Scaffold AUROC 0.874 [4] 0.909 AMES SMILES Scaffold AUROC 0.871 [3] 0.786 DILI SMILES Scaffold AUROC 0.925 [3] 0.882 Skin Reaction SMILES Scaffold AUROC 0.840 [11] 0.615 Carcinogens Lagunin SMILES Scaffold Accuracy 0.770 [12] 0.786 Tox21 SMILES Scaffold AUROC 0.961 [13] 0.882 ClinTox SMILES Scaffold AUROC 0.948 [14] 0.863 herg central SMILES Scaffold AUROC 0.860 [15] 0.888 hERG Karim SMILES Scaffold Accuracy 0.770 [16] 0.745 ToxCast SMILES Scaffold AUROC 0.777 [14] 0.792 SARSCoV2 Vitro Touret SMILES Scaffold AUROC 0.640 [17] 0.601 SARSCOV2 3CLPro Diamond SMILES Scaffold AUROC 0.800 [18] 0.712 HIV SMILES Scaffold AUROC 0.851 [19] 0.732 SAbDab Chen Amino acid Random AUPRC 0.510 [20] 0.473 HuRI Amino acid Cold-start AUPRC 0.724 [21] 0.753 miRTarBase Nucleotide +

Random Accuracy 0.804 [22] 0.799

Amino acid

MHC1 IEDB IMGT Nielsen Amino acid Random AUROC 0.986 [23] 0.907 MHC2 IEDB Jensen Amino acid Random AUROC 0.940 [24] 0.863 weber Amino acid Cold-start AUROC 0.870 [25] 0.743

- phase1 SMILES + Text

Cold-start AUROC 0.576 [26] 0.667

- phase2 SMILES + Text

Cold-start AUROC 0.645 [26] 0.676

- phase3 SMILES + Text

Cold-start AUROC 0.723 [26] 0.728 butkiewicz SMILES Random AUROC 0.840 [27] 0.566

- Table A.8 | Tx-LLM (M) performance compared with SOTA for each regression and generation dataset, along with the feature types and metric type. Performances that are better than SOTA are bolded. Datasets for which we did not find a SOTA are marked as N/A.

Dataset name Feature type Split method

Metric SOTA Tx-LLM

Caco2 Wang SMILES Scaffold MAE 0.285 [2] 0.432 Lipophilicity AstraZeneca SMILES Scaffold MAE 0.467 [28] 0.587 Solubility AqSolDB SMILES Scaffold MAE 0.761 [28] 0.987 PPBR AZ SMILES Scaffold MAE 7.788 [28] 9.108 VDss Lombardo SMILES Scaffold Spearman 0.627 [29] 0.609 Half Life Obach SMILES Scaffold Spearman 0.547 [30] 0.448 Clearance Hepatocyte AZ SMILES Scaffold Spearman 0.440 [31] 0.385 Clearance Microsome AZ SMILES Scaffold Spearman 0.625 [2] 0.413 LD50 Zhu SMILES Scaffold MAE 0.552 [32] 0.618 USPTO Yields SMILES Random Pearson 0.361 [33] 0.070 Buchwald Hartwig SMILES Random Pearson 0.786 [33] 0.905 TAP Amino acid Random MAE N/A 4.983 Leenay Nucleotide Random Spearman 0.740 [34] 0.083 BindingDB kd Amino acid +

Cold-start Pearson 0.712 [35] 0.391 BindingDB ic50 Amino acid +

SMILES

Cold-start Spearman 0.637 [36] 0.311 BindingDB ki Amino acid +

SMILES

Cold-start Pearson 0.840 [37] 0.726 BindingDB Patent Amino acid +

SMILES

Temporal Pearson 0.588 [38] 0.531 DAVIS Amino acid +

SMILES

Cold-start MSE 0.219 [39] 0.704 KIBA Amino acid +

SMILES

Cold-start MSE 0.154 [39] 0.548 DisGeNET Amino acid +

SMILES

Random MAE N/A 0.057

Text

GDSC1 SMILES + Text Random Pearson 0.860 [40] 0.887 GDSC2 SMILES + Text Random Pearson 0.860 [40] 0.900 DrugComb CSS SMILES + Text Combination MAE 16.858 [41] 14.057 OncoPolyPharmacology SMILES + Text Combination Pearson 0.730 [42] 0.552 Protein SAbDab Amino acid Random MAE N/A 1.268 DrugComb HSA SMILES + Text Combination MAE 4.453 [41] 4.118 DrugComb Loewe SMILES + Text Combination MAE 9.184 [41] 17.381 DrugComb Bliss SMILES + Text Combination MAE 4.560 [41] 4.104 DrugComb ZIP SMILES + Text Combination MAE 4.027 [41] 3.777 USPTO SMILES Random Generation

0.415 [43] 0.239

Accuracy

- Table A.9 | Performances on binary classification datasets for PaLM 2 (S), PaLM 2 (M), Tx-LLM (S) and Tx-LLM (M). The best performances are bolded. Dataset name Metric PaLM 2 (S) PaLM 2 (M) Tx-LLM (S) Tx-LLM (M)

PAMPA NCATS AUROC 0.640 0.661 0.646 0.668 HIA Hou AUROC 0.837 0.711 0.942 0.990 Pgp Broccatelli AUROC 0.791 0.848 0.909 0.939 Bioavailability Ma AUROC 0.492 0.564 0.605 0.702 BBB Martins AUROC 0.732 0.616 0.805 0.882 CYP2C19 Veith AUROC 0.608 0.627 0.877 0.895 CYP2D6 Veith AUPRC 0.170 0.198 0.605 0.659 CYP3A4 Veith AUPRC 0.544 0.544 0.800 0.840 CYP1A2 Veith AUPRC 0.590 0.594 0.906 0.914 CYP2C9 Veith AUPRC 0.336 0.401 0.750 0.788 CYP2C9 Substrate CarbonMangels AUPRC 0.340 0.308 0.403 0.436 CYP2D6 Substrate CarbonMangels AUPRC 0.380 0.575 0.643 0.600 CYP3A4 Substrate CarbonMangels AUROC 0.560 0.604 0.637 0.647 hERG AUROC 0.701 0.756 0.879 0.909 AMES AUROC 0.614 0.563 0.785 0.786 DILI AUROC 0.643 0.741 0.727 0.882 Skin Reaction AUROC 0.431 0.553 0.564 0.615 Carcinogens Lagunin Accuracy 0.821 0.714 0.857 0.786 Tox21 AUROC 0.406 0.610 0.858 0.882 ClinTox AUROC 0.387 0.471 0.818 0.863 herg central AUROC 0.491 0.509 0.880 0.888 hERG Karim Accuracy 0.570 0.555 0.724 0.745 ToxCast AUROC 0.455 0.530 0.779 0.792 SARSCoV2 Vitro Touret AUROC 0.580 0.556 0.512 0.601 SARSCOV2 3CLPro Diamond AUROC 0.371 0.619 0.755 0.712 HIV AUROC 0.436 0.494 0.686 0.732 SAbDab Chen AUPRC 0.437 0.545 0.390 0.473 HuRI AUPRC 0.509 0.501 0.705 0.753 miRTarBase Accuracy 0.502 0.499 0.765 0.799 MHC1 IEDB IMGT Nielsen AUROC 0.548 0.557 0.913 0.907 MHC2 IEDB Jensen AUROC 0.617 0.604 0.781 0.863 weber AUROC 0.666 0.680 0.738 0.743

- phase1 AUROC 0.524 0.490 0.624 0.667
- phase2 AUROC 0.529 0.519 0.639 0.676
- phase3 AUROC 0.527 0.508 0.701 0.728 butkiewicz AUROC 0.504 0.499 0.574 0.566

- Table A.10 | Performances on regression and generation datasets for PaLM 2 (S), PaLM 2 (M), Tx-LLM (S) and Tx-LLM (M). The best performances are bolded.

Dataset name Metric PaLM 2 (S) PaLM 2 (M) Tx-LLM (S) Tx-LLM (M)

Caco2 Wang MAE 0.457 1.680 0.621 0.432 Lipophilicity AstraZeneca MAE 1.108 1.189 0.779 0.587 Solubility AqSolDB MAE 2.536 5.427 0.931 0.987 PPBR AZ MAE 13.104 32.447 11.138 9.108 VDss Lombardo Spearman -0.062 0.174 0.497 0.609 Half Life Obach Spearman -0.033 0.380 0.525 0.448 Clearance Hepatocyte AZ Spearman 0.240 0.075 0.256 0.385 Clearance Microsome AZ Spearman 0.337 0.024 0.385 0.413 LD50 Zhu MAE 0.823 0.971 0.808 0.618 USPTO Yields Pearson 0.001 0.002 0.042 0.070 Buchwald Hartwig Pearson 0.333 0.089 0.682 0.905 TAP MAE 2.480 3.536 5.075 4.983 Leenay Spearman 0.004 -0.010 0.048 0.083 BindingDB kd Pearson 0.089 0.087 0.317 0.391 BindingDB ic50 Spearman 0.033 -0.114 0.326 0.311 BindingDB ki Pearson 0.055 0.026 0.565 0.726 BindingDB Patent Pearson -0.022 0.031 0.474 0.531 DAVIS MSE 5.102 5.145 0.564 0.704 KIBA MSE 8.530 8.777 0.709 0.548 DisGeNET MAE 0.088 0.134 0.059 0.057 GDSC1 Pearson -0.042 0.006 0.876 0.887 GDSC2 Pearson -0.058 0.010 0.896 0.900 DrugComb CSS MAE 27.159 24.66 14.740 14.057 OncoPolyPharmacology Pearson 0.056 0.017 0.418 0.552 Protein SAbDab MAE 1.282 1.236 1.432 1.268 DrugComb HSA MAE 5.885 4.485 4.311 4.118 DrugComb Loewe MAE 16.456 20.865 17.428 17.381 DrugComb Bliss MAE 5.675 4.565 4.425 4.104 DrugComb ZIP MAE 5.735 4.742 4.047 3.777 USPTO Generation Accuracy 0.000 0.000 0.220 0.239

Table A.11 | Performances on binary classification datasets for varying few-shot prompting strategies with Tx-LLM (S). The number of shots is varied between 0, 1, 5, and 10, and the shots are either chosen randomly or based on the nearest neighbors (KNN). Nearest neighbors are determined by Tanimoto similarities for molecules and sequence identity for proteins and nucleotides. The best performances are bolded.

Dataset name Metric 0-shot 1-shot random

5-shot random

10-shot random

1-shot KNN

5-shot KNN

10-shot KNN

PAMPA NCATS AUROC 0.677 0.650 0.649 0.633 0.671 0.662 0.646 HIA Hou AUROC 0.956 0.960 0.947 0.939 0.953 0.952 0.942 Pgp Broccatelli AUROC 0.906 0.903 0.908 0.908 0.905 0.908 0.909 Bioavailability Ma AUROC 0.607 0.616 0.619 0.597 0.611 0.606 0.605 BBB Martins AUROC 0.807 0.806 0.800 0.799 0.811 0.808 0.805 CYP2C19 Veith AUROC 0.874 0.875 0.876 0.875 0.875 0.877 0.877 CYP2D6 Veith AUPRC 0.598 0.600 0.601 0.604 0.604 0.605 0.605 CYP3A4 Veith AUPRC 0.801 0.802 0.802 0.799 0.803 0.803 0.800 CYP1A2 Veith AUPRC 0.906 0.907 0.906 0.906 0.907 0.906 0.906 CYP2C9 Veith AUPRC 0.751 0.748 0.752 0.750 0.748 0.751 0.750 CYP2C9 Substrate CarbonMangels AUPRC 0.416 0.408 0.423 0.411 0.412 0.414 0.403 CYP2D6 Substrate CarbonMangels AUPRC 0.624 0.634 0.624 0.624 0.637 0.648 0.643 CYP3A4 Substrate CarbonMangels AUROC 0.645 0.645 0.640 0.639 0.646 0.641 0.637 hERG AUROC 0.869 0.868 0.868 0.877 0.873 0.875 0.879 AMES AUROC 0.780 0.784 0.781 0.783 0.781 0.783 0.785 DILI AUROC 0.697 0.708 0.702 0.715 0.717 0.713 0.727 Skin Reaction AUROC 0.572 0.566 0.544 0.549 0.570 0.552 0.564 Carcinogens Lagunin Accuracy 0.839 0.857 0.875 0.893 0.857 0.857 0.857 Tox21 AUROC 0.857 0.856 0.858 0.856 0.858 0.858 0.858 ClinTox AUROC 0.801 0.796 0.800 0.814 0.811 0.807 0.818 herg central AUROC 0.878 0.879 0.880 0.880 0.880 0.879 0.880 hERG Karim Accuracy 0.711 0.709 0.714 0.715 0.714 0.725 0.724 ToxCast AUROC 0.777 0.778 0.778 0.777 0.779 0.779 0.779 SARSCoV2 Vitro Touret AUROC 0.525 0.526 0.519 0.542 0.521 0.525 0.512 SARSCOV2 3CLPro Diamond AUROC 0.723 0.739 0.744 0.748 0.717 0.746 0.755 HIV AUROC 0.675 0.683 0.691 0.686 0.675 0.686 0.686 SAbDab Chen AUPRC 0.369 0.390 0.385 0.373 0.399 0.385 0.390 HuRI AUPRC 0.698 0.705 0.705 0.705 0.705 0.705 0.705 miRTarBase Accuracy 0.765 0.765 0.765 0.765 0.765 0.765 0.765 MHC1 IEDB IMGT Nielsen AUROC 0.912 0.913 0.913 0.913 0.913 0.913 0.913 MHC2 IEDB Jensen AUROC 0.775 0.780 0.780 0.783 0.778 0.778 0.781 weber AUROC 0.737 0.738 0.738 0.738 0.738 0.738 0.738

- phase1 AUROC 0.628 0.634 0.627 0.628 0.631 0.628 0.624
- phase2 AUROC 0.641 0.642 0.640 0.638 0.642 0.640 0.639
- phase3 AUROC 0.693 0.696 0.697 0.692 0.695 0.697 0.701 butkiewicz AUROC 0.581 0.548 0.582 0.593 0.558 0.538 0.574

Table A.12 | Performances on regression and generation datasets for varying few-shot prompting strategies with Tx-LLM (S). The number of shots is varied between 0, 1, 5, and 10, and the shots are either chosen randomly or based on the nearest neighbors (KNN). Nearest neighbors are determined by Tanimoto similarities for molecules and sequence identity for proteins and nucleotides. The best performances are bolded.

Dataset name Metric 0-shot 1-shot random

5-shot random

10-shot random

1-shot KNN

5-shot KNN

10-shot KNN

Caco2 Wang MAE 0.605 0.621 0.627 0.613 0.616 0.597 0.621 Lipophilicity AstraZeneca MAE 0.803 0.812 0.791 0.782 0.818 0.793 0.779 Solubility AqSolDB MAE 0.899 0.918 0.915 0.921 0.919 0.917 0.931 PPBR AZ MAE 10.814 10.727 10.854 11.024 10.935 11.129 11.138 VDss Lombardo Spearman 0.496 0.496 0.508 0.487 0.486 0.488 0.497 Half Life Obach Spearman 0.494 0.502 0.503 0.489 0.523 0.499 0.525 Clearance Hepatocyte AZ Spearman 0.255 0.285 0.252 0.258 0.285 0.303 0.256 Clearance Microsome AZ Spearman 0.401 0.406 0.401 0.406 0.403 0.386 0.385 LD50 Zhu MAE 0.815 0.811 0.809 0.809 0.810 0.807 0.808 USPTO Yields Pearson 0.010 0.044 0.040 0.038 0.044 0.042 0.042 Buchwald Hartwig Pearson 0.736 0.802 0.809 0.491 0.800 0.808 0.682 TAP MAE 5.150 5.092 5.137 5.029 5.087 5.075 5.075 Leenay Spearman 0.036 0.035 0.034 0.063 0.032 0.055 0.048 BindingDB kd Pearson 0.314 0.314 0.320 0.305 0.320 0.318 0.317 BindingDB ic50 Spearman 0.331 0.327 0.327 0.327 0.327 0.326 0.326 BindingDB ki Pearson 0.569 0.574 0.568 0.568 0.576 0.565 0.565 BindingDB Patent Pearson 0.483 0.474 0.474 0.474 0.474 0.474 0.474 DAVIS MSE 0.561 0.561 0.570 0.561 0.561 0.564 0.564 KIBA MSE 0.743 0.711 0.715 0.718 0.707 0.709 0.709 DisGeNET MAE 0.060 0.059 0.059 0.059 0.059 0.059 0.059 GDSC1 Pearson 0.876 0.876 0.875 0.875 0.876 0.876 0.876 GDSC2 Pearson 0.895 0.895 0.895 0.895 0.895 0.895 0.896 DrugComb CSS MAE 14.779 14.775 14.729 14.749 14.781 14.743 14.740 OncoPolyPharmacology Pearson 0.423 0.427 0.427 0.418 0.432 0.424 0.418 Protein SAbDab MAE 1.399 1.384 1.427 1.406 1.398 1.420 1.432 DrugComb HSA MAE 4.298 4.297 4.300 4.312 4.296 4.299 4.311 DrugComb Loewe MAE 17.425 17.454 17.435 17.424 17.455 17.440 17.428 DrugComb Bliss MAE 4.284 4.284 4.293 4.352 4.285 4.294 4.425 DrugComb ZIP MAE 4.014 4.021 4.035 4.045 4.022 4.035 4.047 USPTO Generation

0.225 0.221 0.212 0.209 0.221 0.220 0.220

Accuracy

- Table A.13 | Performances on binary classification datasets with Tx-LLM (S) using 10-shot KNN prompting, with and without context. The best performances are bolded.

Dataset name Metric 10-shot KNN no context 10-shot KNN with context

PAMPA NCATS AUROC 0.664 0.646 HIA Hou AUROC 0.905 0.942 Pgp Broccatelli AUROC 0.884 0.909 Bioavailability Ma AUROC 0.592 0.605 BBB Martins AUROC 0.797 0.805 CYP2C19 Veith AUROC 0.875 0.877 CYP2D6 Veith AUPRC 0.596 0.605 CYP3A4 Veith AUPRC 0.805 0.800 CYP1A2 Veith AUPRC 0.898 0.906 CYP2C9 Veith AUPRC 0.750 0.750 CYP2C9 Substrate CarbonMangels AUPRC 0.358 0.403 CYP2D6 Substrate CarbonMangels AUPRC 0.645 0.643 CYP3A4 Substrate CarbonMangels AUROC 0.639 0.637 hERG AUROC 0.868 0.879 AMES AUROC 0.747 0.785 DILI AUROC 0.633 0.727 Skin Reaction AUROC 0.529 0.564 Carcinogens Lagunin Accuracy 0.857 0.857 Tox21 AUROC 0.828 0.858 ClinTox AUROC 0.759 0.818 herg central AUROC 0.876 0.880 hERG Karim Accuracy 0.728 0.724 ToxCast AUROC 0.719 0.779 SARSCoV2 Vitro Touret AUROC 0.550 0.512 SARSCOV2 3CLPro Diamond AUROC 0.765 0.755 HIV AUROC 0.668 0.686 SAbDab Chen AUPRC 0.415 0.390 HuRI AUPRC 0.703 0.705 miRTarBase Accuracy 0.765 0.765 MHC1 IEDB IMGT Nielsen AUROC 0.912 0.913 MHC2 IEDB Jensen AUROC 0.786 0.781 weber AUROC 0.738 0.738 phase1 AUROC 0.608 0.624 phase2 AUROC 0.635 0.639 phase3 AUROC 0.691 0.701 butkiewicz AUROC 0.621 0.574

- Table A.14 | Performances on regression and generation datasets with Tx-LLM (S) using 10-shot KNN prompting, with and without context. The best performances are bolded.

Dataset name Metric 10-shot KNN no context 10-shot KNN with context

Caco2 Wang MAE 0.713 0.621 Lipophilicity AstraZeneca MAE 0.765 0.779 Solubility AqSolDB MAE 1.101 0.931 PPBR AZ MAE 34.763 11.138 VDss Lombardo Spearman 0.489 0.497 Half Life Obach Spearman 0.389 0.525 Clearance Hepatocyte AZ Spearman 0.219 0.256 Clearance Microsome AZ Spearman 0.401 0.385 LD50 Zhu MAE 0.818 0.808 USPTO Yields Pearson 0.041 0.042 Buchwald Hartwig Pearson 0.655 0.682 TAP MAE 5.711 5.075 Leenay Spearman 0.036 0.048 BindingDB kd Pearson 0.309 0.317 BindingDB ic50 Spearman 0.318 0.326 BindingDB ki Pearson 0.547 0.565 BindingDB Patent Pearson 0.466 0.474 DAVIS MSE 0.564 0.564 KIBA MSE 0.704 0.709 DisGeNET MAE 0.060 0.059 GDSC1 Pearson 0.875 0.876 GDSC2 Pearson 0.818 0.896 DrugComb CSS MAE 21.656 14.740 OncoPolyPharmacology Pearson 0.380 0.418 Protein SAbDab MAE 1.433 1.432 DrugComb HSA MAE 4.374 4.311 DrugComb Loewe MAE 18.923 17.428 DrugComb Bliss MAE 4.843 4.425 DrugComb ZIP MAE 5.315 4.047 USPTO Generation Accuracy 0.221 0.220

- Table A.15 | Performance of Tx-LLM (S) finetuned on different datasets and evaluated with 10-shot KNN prompting on binary classification datasets. "All datasets" indicates a Tx-LLM (S) model finetuned on all TDC datasets, "molecule datasets" indicates a Tx-LLM (S) model finetuned on datasets containing molecules (datasets only involving other drug types such as proteins or nucleic acids are not included in training), and "ADMET datasets" indicates a Tx-LLM (S) model finetuned on datasets in the ADMET benchmark, which only contains molecules.

Dataset name Feature type In ADMET

Metric All datasets

Molecule datasets

ADMET datasets

HIA Hou SMILES Yes AUROC 0.942 0.928 0.915 Pgp Broccatelli SMILES Yes AUROC 0.909 0.917 0.852 Bioavailability Ma SMILES Yes AUROC 0.605 0.667 0.713 BBB Martins SMILES Yes AUROC 0.805 0.860 0.843 CYP2D6 Veith SMILES Yes AUPRC 0.605 0.522 0.418 CYP3A4 Veith SMILES Yes AUPRC 0.800 0.737 0.736 CYP2C9 Veith SMILES Yes AUPRC 0.750 0.698 0.609 CYP2C9 Substrate CarbonMangels SMILES Yes AUPRC 0.403 0.455 0.328 CYP2D6 Substrate CarbonMangels SMILES Yes AUPRC 0.643 0.670 0.683 CYP3A4 Substrate CarbonMangels SMILES Yes AUROC 0.637 0.740 0.683 hERG SMILES Yes AUROC 0.879 0.857 0.753 AMES SMILES Yes AUROC 0.785 0.726 0.672 DILI SMILES Yes AUROC 0.727 0.628 0.524

- phase1 SMILES + Text No AUROC 0.624 0.556 0.534
- phase2 SMILES + Text No AUROC 0.639 0.522 0.503
- phase3 SMILES + Text No AUROC 0.701 0.477 0.492 PAMPA NCATS SMILES No AUROC 0.646 0.702 0.717 CYP2C19 Veith SMILES No AUROC 0.877 0.829 0.797 CYP1A2 Veith SMILES No AUPRC 0.906 0.861 0.607 Skin Reaction SMILES No AUROC 0.564 0.627 0.388 Carcinogens Lagunin SMILES No Accuracy 0.857 0.875 0.714 Tox21 SMILES No AUROC 0.858 0.811 0.662 ClinTox SMILES No AUROC 0.818 0.778 0.411 herg central SMILES No AUROC 0.880 0.840 0.663 hERG Karim SMILES No Accuracy 0.724 0.691 0.654 ToxCast SMILES No AUROC 0.779 0.758 0.632 SARSCoV2 Vitro Touret SMILES No AUROC 0.512 0.571 0.570 SARSCOV2 3CLPro Diamond SMILES No AUROC 0.755 0.715 0.690 HIV SMILES No AUROC 0.686 0.633 0.421 butkiewicz SMILES No AUROC 0.574 0.645 0.498 miRTarBase Nucleotide +

No Accuracy 0.765 0.499 0.502

Amino acid

SAbDab Chen Amino acid No AUPRC 0.390 0.694 0.735 HuRI Amino acid No AUPRC 0.705 0.513 0.513 MHC1 IEDB IMGT Nielsen Amino acid No AUROC 0.913 0.675 0.650 MHC2 IEDB Jensen Amino acid No AUROC 0.781 0.718 0.714 weber Amino acid No AUROC 0.738 0.689 0.688

- Table A.16 | Performance of Tx-LLM (S) finetuned on different datasets and evaluated with 10-shot KNN prompting on regression and generation datasets. "All datasets" indicates a Tx-LLM (S) model finetuned on all TDC datasets, "molecule datasets" indicates a Tx-LLM (S) model finetuned on datasets containing molecules (datasets only involving other drug types such as proteins or nucleic acids are not included in training), and "ADMET datasets" indicates a Tx-LLM (S) model finetuned on datasets in the ADMET benchmark, which only contains molecules.

Dataset name Feature type In ADMET

Metric All datasets

Molecule datasets

ADMET datasets

Caco2 Wang SMILES Yes MAE 0.621 0.578 0.555 Lipophilicity AstraZeneca SMILES Yes MAE 0.779 0.815 0.813 Solubility AqSolDB SMILES Yes MAE 0.931 0.989 0.929 PPBR AZ SMILES Yes MAE 11.138 11.394 11.441 VDss Lombardo SMILES Yes Spearman 0.497 0.338 0.412 Half Life Obach SMILES Yes Spearman 0.525 0.316 0.267 Clearance Hepatocyte AZ SMILES Yes Spearman 0.256 0.154 0.262 Clearance Microsome AZ SMILES Yes Spearman 0.385 0.354 0.519 LD50 Zhu SMILES Yes MAE 0.808 0.749 0.713 GDSC1 SMILES + Text No Pearson 0.876 0.075 0.146 GDSC2 SMILES + Text No Pearson 0.896 0.120 0.135 DrugComb CSS SMILES + Text No MAE 14.740 32.394 26.871 OncoPolyPharmacology SMILES + Text No Pearson 0.418 0.097 0.159 DrugComb HSA SMILES + Text No MAE 4.311 10.883 5.127 DrugComb Loewe SMILES + Text No MAE 17.428 26.516 12.242 DrugComb Bliss SMILES + Text No MAE 4.425 11.984 5.603 DrugComb ZIP SMILES + Text No MAE 4.047 11.367 5.351 USPTO Yields SMILES No Pearson 0.042 0.015 0.007 Buchwald Hartwig SMILES No Pearson 0.682 0.283 0.554 USPTO SMILES No Generation

0.220 0.158 0.000

Accuracy

Leenay Nucleotide No Spearman 0.048 -0.006 -0.014 DisGeNET Amino acid + Text No MAE 0.059 0.511 0.432 BindingDB kd Amino acid +

No Pearson 0.317 0.126 -0.001 BindingDB ic50 Amino acid +

SMILES

No Spearman 0.326 0.004 0.156 BindingDB ki Amino acid +

SMILES

No Pearson 0.565 -0.057 -0.049 BindingDB Patent Amino acid +

SMILES

No Pearson 0.474 0.010 -0.030 DAVIS Amino acid +

SMILES

No MSE 0.564 14.955 5.631 KIBA Amino acid +

SMILES

No MSE 0.709 5.950 3.391

SMILES

TAP Amino acid No MAE 5.075 6.491 5.024 Protein SAbDab Amino acid No MAE 1.432 1.864 1.067

- Table A.17 | The percent of each dataset’s test set containing features that also exist in the PaLM-2 training data, excluding datasets with no overlap at all.

Dataset name Percent overlap with PaLM-2 training data

Metric Unfiltered performance

Filtered performance

TAP 10.42 MAE 4.983 4.560 HuRI 5.93 AUPRC 0.753 0.756

SAbDab Chen 5.39 AUPRC 0.473 0.529

phase3 0.28 AUROC 0.723 0.727 BindingDB kd 0.09 Pearson 0.391 0.391

miRTarBase 0.04 Accuracy 0.804 0.799 DisGeNET 0.02 MAE 0.057 0.057

200

| || | |
|---|---|
| | |
| | |
| | |
<br><br>| | |
|---|---|
| | |
| | |
| | |
<br><br>| | |
|---|---|
| | |
| | |
| | |
<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

%ImprovementoverPaLM2(S)

150

100

50

0

50

100

150

200

PaLM 2 (M) Tx-LLM (S) Tx-LLM (M)

Model Variant

###### Figure A.3 | The percent improvement of PaLM 2 (M), Tx-LLM (S), and Tx-LLM (M) compared to PaLM 2 (S) for all TDC tasks.

10

%Improvementover0-shot

5

| | | | |
|---|---|---|---|
| | | | |
| | | | |

0

5

Random KNN

10

1 5 10

Number of Shots

###### Figure A.4 | The percent improvement of few-shot prompting over 0-shot prompting with Tx-LLM (S). The number of shots and the shot selection method (random or KNN) are varied.

###### References

- 1. Siramshetty, V., Williams, J., Nguyen, Ð.-T., Neyra, J., Southall, N., Mathé, E., Xu, X. & Shah, P. Validating ADME QSAR Models Using Marketed Drugs. SLAS DISCOVERY: Advancing the Science of Drug Discovery 26. PMID: 34176369, 1326–1336. eprint: https://doi.org/10.1177/24725552211017520. https://doi.org/10.1177/24725552211017520 (2021).
- 2. Li, A. & Huang, D. ADMET property prediction with Oloren ChemEngine 2022.
- 3. Turon, G., Hlozek, J., Woodland, J. G., Kumar, A., Chibale, K. & Duran-Frigola, M. First fully-automated AI/ML virtual screening cascade implemented at a drug discovery centre in Africa. en. Nature Communications 14, 5736. issn: 2041-1723. https://www.nature.com/articles/s41467-023-41512-2 (2024) (Sept. 2023).
- 4. Bera, S., Dent, J., Gill, G., Stolman, A. & Wu, B. SimGCN for TDC Benchmarks 2022.
- 5. Fontenot, R., Kathad, U., McDermott, J., Sturtevant, D., Sharma, P. & Carr, P. Predicting a Compounds Blood-BrainBarrier Permeability with Lantern Pharma’s AI and ML Platform, RADR 2023.
- 6. Plonka, W., Stork, C., Šícho, M. & Kirchmair, J. CYPlebrity: Machine learning models for the prediction of inhibitors of cytochrome P450 enzymes. en. Bioorganic & Medicinal Chemistry 46, 116388. issn: 09680896. https://linkinghub.elsevier. com/retrieve/pii/S0968089621003965 (2024) (Sept. 2021).
- 7. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V. S. & Leskovec, J. Pre-training Graph Neural Networks. CoRR abs/1905.12265. arXiv: 1905.12265. http://arxiv.org/abs/1905.12265 (2019).
- 8. Plonka, W., Stork, C., Šícho, M. & Kirchmair, J. CYPlebrity: Machine learning models for the prediction of inhibitors of cytochrome P450 enzymes. en. Bioorganic & Medicinal Chemistry 46, 116388. issn: 09680896. https://linkinghub.elsevier. com/retrieve/pii/S0968089621003965 (2024) (Sept. 2021).
- 9. Turon, G., Hlozek, J., Woodland, J. G., Kumar, A., Chibale, K. & Duran-Frigola, M. First fully-automated AI/ML virtual screening cascade implemented at a drug discovery centre in Africa. en. Nature Communications 14, 5736. issn: 2041-1723. https://www.nature.com/articles/s41467-023-41512-2 (2024) (Sept. 2023).
- 10. Huang, K., Fu, T., Glass, L. M., Zitnik, M., Xiao, C. & Sun, J. DeepPurpose: a deep learning library for drug–target interaction prediction. Bioinformatics 36, 5545–5547. issn: 1367-4803. eprint: https://academic.oup.com/bioinformatics/articlepdf/36/22-23/5545/50716143/btaa1005.pdf. https://doi.org/10.1093/bioinformatics/btaa1005 (Dec. 2020).
- 11. Alves, V. M., Muratov, E., Fourches, D., Strickland, J., Kleinstreuer, N., Andrade, C. H. & Tropsha, A. Predicting chemically-induced skin reactions. Part I: QSAR models of skin sensitization and their application to identify potentially hazardous compounds. Toxicology and Applied Pharmacology 284, 262–272. issn: 0041-008X. https://www.sciencedirect. com/science/article/pii/S0041008X14004529 (2015).
- 12. Lagunin, A., Filimonov, D., Zakharov, A., Xie, W., Huang, Y., Zhu, F., Shen, T., Yao, J. & Poroikov, V. Computer-Aided Prediction of Rodent Carcinogenicity by PASS and CISOC-PSCT. QSAR & Combinatorial Science 28, 806–810. eprint: https: //onlinelibrary.wiley.com/doi/pdf/10.1002/qsar.200860192. https://onlinelibrary.wiley.com/doi/abs/10.1002/qsar.200860192

(2009).

- 13. Shermukhamedov, S., Mamurjonova, D. & Probst, M. Structure to Property: Chemical Element Embeddings and a Deep Learning Approach for Accurate Prediction of Chemical Properties 2023. arXiv: 2309.09355 [physics.chem-ph].
- 14. Li, P., Li, Y., Hsieh, C.-Y., Zhang, S., Liu, X., Liu, H., Song, S. & Yao, X. TrimNet: learning molecular representation from triplet messages for biomedicine. Briefings in Bioinformatics 22, bbaa266. issn: 1477-4054. eprint: https://academic.oup. com/bib/article-pdf/22/4/bbaa266/39144778/bbaa266.pdf. https://doi.org/10.1093/bib/bbaa266 (Nov. 2020).
- 15. Korotcov, A., Tkachenko, V., Russo, D. P. & Ekins, S. Comparison of Deep Learning With Multiple Machine Learning Methods and Metrics Using Diverse Drug Discovery Data Sets. en. Molecular Pharmaceutics 14, 4462–4475. issn: 1543-8384, 1543-8392. https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.7b00578 (2024) (Dec. 2017).
- 16. Karim, A., Lee, M., Balle, T. & Sattar, A. CardioTox net: a robust predictor for hERG channel blockade based on deep learning meta-feature ensembles. en. Journal of Cheminformatics 13, 60. issn: 1758-2946. https://jcheminf.biomedcentral. com/articles/10.1186/s13321-021-00541-z (2024) (Dec. 2021).
- 17. Liu, Y., Wu, Y., Shen, X. & Xie, L. COVID-19 Multi-Targeted Drug Repurposing Using Few-Shot Learning. Frontiers in Bioinformatics 1, 693177. issn: 2673-7647. https://www.frontiersin.org/articles/10.3389/fbinf.2021.693177/full (2024) (June 2021).
- 18. Haneczok, J. & Delijewski, M. Machine learning enabled identification of potential SARS-CoV-2 3CLpro inhibitors based on fixed molecular fingerprints and Graph-CNN neural representations. Journal of Biomedical Informatics 119, 103821. issn: 1532-0464. https://www.sciencedirect.com/science/article/pii/S1532046421001507 (2021).
- 19. Li, J., Cai, D. & He, X. Learning Graph-Level Representation for Drug Discovery (Sept. 2017).
- 20. Chen, X., Dougherty, T., Hong, C., Schibler, R., Zhao, Y. C., Sadeghi, R., Matasci, N., Wu, Y.-C. & Kerman, I. Predicting Antibody Developability from Sequence using Machine Learning. bioRxiv. eprint: https://www.biorxiv.org/content/early/ 2020/06/20/2020.06.18.159798.full.pdf. https://www.biorxiv.org/content/early/2020/06/20/2020.06.18.159798 (2020).
- 21. Raimondi, D., Simm, J., Arany, A. & Moreau, Y. A novel method for data fusion over entity-relation graphs and its application to protein–protein interaction prediction. Bioinformatics 37, 2275–2281. issn: 1367-4803. eprint: https://academic.oup. com/bioinformatics/article-pdf/37/16/2275/50339388/btab092.pdf. https://doi.org/10.1093/bioinformatics/btab092 (Feb. 2021).
- 22. Wong, L., You, Z.-H., Guo, Z.-H., Yi, H.-C., Chen, Z.-H. & Cao, M.-Y. MIPDH: A Novel Computational Model for Predicting microRNA–mRNA Interactions by DeepWalk on a Heterogeneous Network. en. ACS Omega 5, 17022–17032. issn: 2470-1343, 2470-1343. https://pubs.acs.org/doi/10.1021/acsomega.9b04195 (2024) (July 2020).
- 23. Gfeller, D., Schmidt, J., Croce, G., Guillaume, P., Bobisse, S., Genolet, R., Queiroz, L., Cesbron, J., Racle, J. & Harari, A. Improved predictions of antigen presentation and TCR recognition with MixMHCpred2.2 and PRIME2.0 reveal potent SARS-CoV-2 CD8+ T-cell epitopes. en. Cell Systems 14, 72–83.e5. issn: 24054712. https://linkinghub.elsevier.com/ retrieve/pii/S2405471222004707 (2024) (Jan. 2023).
- 24. Motmaen, A., Dauparas, J., Baek, M., Abedi, M. H., Baker, D. & Bradley, P. Peptide-binding specificity prediction using fine-tuned protein structure prediction networks. Proceedings of the National Academy of Sciences 120, e2216697120. eprint: https://www.pnas.org/doi/pdf/10.1073/pnas.2216697120. https://www.pnas.org/doi/abs/10.1073/pnas.2216697120

(2023).

- 25. Weber, A., Born, J. & Rodriguez Martínez, M. TITAN: T-cell receptor specificity prediction with bimodal attention networks. Bioinformatics 37, i237–i244. issn: 1367-4803. eprint: https://academic.oup.com/bioinformatics/articlepdf/37/Supplement\_1/i237/50694051/btab294.pdf. https://doi.org/10.1093/bioinformatics/btab294 (July 2021).
- 26. Fu, T., Huang, K., Xiao, C., Glass, L. M. & Sun, J. HINT: Hierarchical interaction network for clinical-trial-outcome predictions. Patterns 3, 100445. issn: 2666-3899. https://www.sciencedirect.com/science/article/pii/S2666389922000186

(2022).

- 27. Vu, O., Mendenhall, J., Altarawy, D. & Meiler, J. BCL::Mol2D—a robust atom environment descriptor for QSAR modeling and lead optimization. en. Journal of Computer-Aided Molecular Design 33, 477–486. issn: 0920-654X, 1573-4951. http://link.springer.com/10.1007/s10822-019-00199-8 (2024) (May 2019).
- 28. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K. & Barzilay, R. Analyzing Learned Molecular Representations for Property Prediction. Journal of Chemical Information and Modeling 59. PMID: 31361484, 3370–3388. eprint: https: //doi.org/10.1021/acs.jcim.9b00237. https://doi.org/10.1021/acs.jcim.9b00237 (2019).
- 29. Boral, N., Ghosh, P., Goswami, A. & Bhattacharyya, M. Accountable Prediction of Drug ADMET Properties with Molecular Descriptors. bioRxiv. eprint: https://www.biorxiv.org/content/early/2022/07/02/2022.06.29.115436.full.pdf. https://www.biorxiv.org/content/early/2022/07/02/2022.06.29.115436 (2022).
- 30. Euclia. https://https://github.com/euclia/public-models. 2023.
- 31. Boral, N., Ghosh, P., Goswami, A. & Bhattacharyya, M. Accountable Prediction of Drug ADMET Properties with Molecular Descriptors. bioRxiv. eprint: https://www.biorxiv.org/content/early/2022/07/02/2022.06.29.115436.full.pdf. https://www.biorxiv.org/content/early/2022/07/02/2022.06.29.115436 (2022).
- 32. Huang, D., Chowdhuri, S. (, Li, A., Li, A., Agrawal, A., Gano, K. & Zhu, A. A Unified System for Molecular Property Predictions: Oloren ChemEngine and its Applications preprint (Chemistry, Oct. 2022). https://chemrxiv.org/engage/ chemrxiv/article-details/6350b9d186473a47d31a8492 (2024).
- 33. Probst, D., Schwaller, P. & Reymond, J.-L. Reaction classification and yield prediction using the differential reaction fingerprint DRFP. en. Digital Discovery 1, 91–97. issn: 2635-098X. http://xlink.rsc.org/?DOI=D1DD00006C (2024)

(2022).

- 34. Leenay, R. T., Aghazadeh, A., Hiatt, J., Tse, D., Roth, T. L., Apathy, R., Shifrut, E., Hultquist, J. F., Krogan, N., Wu, Z., Cirolia, G., Canaj, H., Leonetti, M. D., Marson, A., May, A. P. & Zou, J. Large dataset enables prediction of repair after CRISPR–Cas9 editing in primary T cells. en. Nature Biotechnology 37, 1034–1037. issn: 1087-0156, 1546-1696. https://www.nature.com/articles/s41587-019-0203-2 (2024) (Sept. 2019).
- 35. Kalemati, M., Zamani Emani, M. & Koohi, S. BiComp-DTA: Drug-target binding affinity prediction through complementary biological-related and compression-based featurization approach. en. PLOS Computational Biology 19 (ed Schlessinger, A.) e1011036. issn: 1553-7358. https://dx.plos.org/10.1371/journal.pcbi.1011036 (2024) (Mar. 2023).
- 36. Kinnings, S. L., Liu, N., Tonge, P. J., Jackson, R. M., Xie, L. & Bourne, P. E. A Machine Learning-Based Method To Improve Docking Scoring Functions and Its Application to Drug Repurposing. en. Journal of Chemical Information and Modeling 51, 408–419. issn: 1549-9596, 1549-960X. https://pubs.acs.org/doi/10.1021/ci100369f (2024) (Feb. 2011).
- 37. Wei, B. & Gong, X. DeepPLA: a novel deep learning-based model for protein-ligand binding affinity prediction. bioRxiv. eprint: https://www.biorxiv.org/content/early/2021/12/03/2021.12.01.470868.full.pdf. https://www.biorxiv.org/content/ early/2021/12/03/2021.12.01.470868 (2021).
- 38. Lam, H. T., Sbodio, M. L., Galindo, M. M., Zayats, M., Fernández-Díaz, R., Valls, V., Picco, G., Ramis, C. B. & López, V. Otter-Knowledge: benchmarks of multimodal knowledge graph representation learning from different sources for drug discovery 2023. arXiv: 2306.12802 [cs.LG].
- 39. Pei, Q., Wu, L., Zhu, J., Xia, Y., Xie, S., Qin, T., Liu, H., Liu, T.-Y. & Yan, R. Breaking the barriers of data scarcity in drug–target affinity prediction. Briefings in Bioinformatics 24, bbad386. issn: 1477-4054. eprint: https://academic.oup. com/bib/article-pdf/24/6/bbad386/54034646/bbad386.pdf. https://doi.org/10.1093/bib/bbad386 (Oct. 2023).
- 40. Lind, A. P. & Anderson, P. C. Predicting drug activity against cancer cells by random forest models based on minimal genomic information and chemical properties. en. PLOS ONE 14 (ed Olier, I.) e0219774. issn: 1932-6203. https: //dx.plos.org/10.1371/journal.pone.0219774 (2024) (July 2019).
- 41. Xia, F., Shukla, M., Brettin, T., Garcia-Cardona, C., Cohn, J., Allen, J. E., Maslov, S., Holbeck, S. L., Doroshow, J. H., Evrard, Y. A., Stahlberg, E. A. & Stevens, R. L. Predicting tumor cell line response to drug pairs with deep learning. en. BMC Bioinformatics 19, 486. issn: 1471-2105. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2509-3

(2024) (Dec. 2018).

- 42. Preuer, K., Lewis, R. P. I., Hochreiter, S., Bender, A., Bulusu, K. C. & Klambauer, G. DeepSynergy: predicting anticancer drug synergy with Deep Learning. Bioinformatics 34, 1538–1546. issn: 1367-4803. eprint: https://academic.oup. com/bioinformatics/article-pdf/34/9/1538/48915102/bioinformatics\_34\_9\_1538.pdf. https://doi.org/10.1093/ bioinformatics/btx806 (Dec. 2017).
- 43. Zheng, S., Rao, J., Zhang, Z., Xu, J. & Yang, Y. Predicting Retrosynthetic Reactions Using Self-Corrected Transformer Neural Networks. en. Journal of Chemical Information and Modeling 60, 47–55. issn: 1549-9596, 1549-960X. https: //pubs.acs.org/doi/10.1021/acs.jcim.9b00949 (2024) (Jan. 2020).

