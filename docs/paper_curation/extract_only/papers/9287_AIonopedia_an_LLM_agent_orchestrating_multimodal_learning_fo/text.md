# AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery

Yuqi Yin1#, Yibo Fu2#, Siyuan Wang3#, Peng Sun1, Hongyu Wang4, Xiaohui Wang4, Lei Zheng5, Zhiyong Li2*, Zhirong Liu1*, Jianji Wang2*, and Zhaoxi Sun4*

1College of Chemistry and Molecular Engineering, Peking University, Beijing 100871, China 2Henan Key Laboratory of Green Chemistry, Collaborative Innovation Center of Henan Province for Green Manufacturing of Fine Chemicals, Key Laboratory of Green Chemical Media and Reactions, Ministry of Education, School of Chemistry and Chemical Engineering, Henan Normal University, Xinxiang, Henan 453007, P. R. China 3School of Computer Science, Shanghai Jiao Tong University, Shanghai 201100, China 4Faculty of Synthetic Biology, Shenzhen University of Advanced Technology, Shenzhen 518107, China 5Shanghai Frontiers Science Center of Artificial Intelligence and Deep Learning and NYU-ECNU Center for Computational Chemistry, NYU-Shanghai, 1555 Century Avenue, Pudong New Area, Shanghai 200062, China

*To whom correspondence should be addressed: Zhiyong Li yli@htu.edu.cn Zhirong Liu LiuZhiRong@pku.edu.cn Jianji Wang jwang@htu.edu.cn Zhaoxi Sun z.sun@suat-sz.edu.cn

Abstract The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical

efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.

Keywords: Ionic Liquids, Gas Absorption, Solute Extraction, Large Language model, Agent

- 1. Introduction. Ionic liquids (IL) are salts composed of cations and anions, defined as having melting points below 100 ℃

at ambient pressure. Owing to their low volatility, high thermal stability, absorption capacity, electrochemical advantages and other favorable properties, they have found widespread use across many applications.1-6 By designing and screening the constituent cation-anion pairs, ILs offer exceptional tunability.7-11 This tunability, however, results in a vast combinatorial space of potential cation-anion pairs. Yet for precisely this reason, selecting ILs that satisfy the requisite physicochemical property criteria for specific application scenarios remains the key bottleneck to practical deployment.

Traditionally, the design of ILs has relied on the expert knowledge. However, any given physicochemical property typically arises from multiple intertwined interactions, making precise control difficult. For example, lengthening the alkyl chain of [RMIM]+ reduces Coulombic interactions and increases entropy, both of which tend to lower the melting point, while strengthening van der Waals interactions, which tends to raise it.12-18 Alongside expert knowledge, several computational approaches can aid researchers, ranging from molecular dynamics (MD) simulation and quantum chemistry calculations to simple linear regressions such as the Abraham model19. However, these methods often suffer from prohibitive computational cost, limited accuracy, or narrow domains of applicability.20-23

To address these challenges, deep learning provides a new approach, enabling data‑driven models to leverage existing experimental results for rapid, generalizable inference while maintaining strong accuracy. In the previous IL studies researchers chose to use neural networks including descriptor-based Multi-Layer Perceptrons (MLP)24-26, SMILES-sequence Recurrent Neural Networks (RNN)27-29, Convolutional Neural Networks (CNN) on 1D/2D molecular representations25,30,31, and message-passing Graph Neural Networks (GNN) for molecular graphs32-34. Nevertheless, three problems remain to be addressed. First, unlike organic chemistry and related areas where experimental measurements are abundant, labeled data for ILs are scarce.3537 A major challenge, therefore, lies in how to utilize unlabeled datasets to compensate for the scarcity of labeled data. Second, molecular data are intrinsically multimodal38, and we must integrate these modalities more effectively to represent IL systems with higher fidelity. Third, workflow automation is lacking, since conventional chemical pipelines are fragmented.39 We therefore seek methods that can efficiently process data and perform diverse tasks.

Concurrently, the rapid development of Large Language Models (LLMs) in recent years offers a different perspective from task-specific deep learning models, which can be used to address the aforementioned issues. Since 2018, numerous works represented by BERT40 and GPT41 have proposed that we can leverage selfsupervised training to utilize vast amounts of text data to improve the performance of downstream Natural

Language Processing (NLP) tasks. Researchers in chemistry have drawn inspiration from these NLP approaches to develop various LLMs, such as encoder-based ChemBERTa42, decoder-based ChemLLM43, and encoder-decoder-based MolT544, among others. Building on the success in NLP, computer vision researchers have also joined the effort, with multimodal works such as CLIP45, which excels in image-text understanding. In chemistry, works that align modalities based on the inherent multimodality of molecules have been emerging continuously. Examples include MMFRL46, which uses molecular graphs and five other modalities, and PointGAT47, which enhances GNN performance with additional 3D representations. To further enhance the capabilities of LLMs, works such as Toolformer48 combine them with external tools to form agents, enabling LLMs to automate pipeline execution and independently make decisions to solve problems. Chemical researchers have also adopted this approach, with projects like Coscientist49 significantly accelerating the development of automated experimentation.

Building on this background, we introduce AIonopedia, to our knowledge the first efficient LLM-based intelligent agent tailored to ILs. By interacting with various specialized modules, it orchestrates the execution of multiple IL-related pipelines. AIonopedia can autonomously search and process data, enabling an end-toend solution to IL research problems. Its core module, the property predictor, is the first LLM-augmented multimodal domain foundation model for ILs reported to date. This module follows a two-step training paradigm of modality alignment and fine-tuning, effectively leveraging unlabeled data from molecular graphs, SMILES sequences, and physicochemical descriptors to enhance performance. Motivated by limited species coverage and the lack of high-quality data relevant to important real-world scenarios, we compile a new IL dataset for fine-tuning that contains the largest collection of known IL solute-solvent interaction data. Our method consistently achieves superior performance across a wide range of property datasets, while also demonstrating robust Out-Of-Distribution (OOD) generalization. Built on this property predictor, we develop two complementary pipelines: an IL modification pipeline that performs anion replacement and cation sidechain edits and is evaluated on literature-reported systems, and a hierarchical search pipeline that combines traversal and molecular similarity search for large-scale molecular screening and design. We further validate the screening pipeline in wet-lab experiments, confirming its effectiveness in real-world settings. In summary, the introduction of AIonopedia provides a novel and efficient tool for IL research, advancing data-driven and automated approaches in chemistry.

- 2. Methodology. Inspired by a series of chemistry-domain LLM-agent studies49-51, we designed AIonopedia. The complete

workflow that includes the agent is presented in Fig. 1A.

- 2.1. Overview of the AIonopedia. At the core of the tool-invocation pipeline is a planner powered by GPT-552, OpenAI’s State-of-the-Art

(SOTA) reasoning model. The exceptional performance of GPT-5, validated on benchmarks such as Aider Polyglot53, AIME 202554, and MMMU55, underpins the agent's capabilities. Following the ReAct56 methodology, the planner iteratively combines reasoning and acting to interact with six specialized tools: web searcher, PubChem searcher, SMILES canonicalizer, data processor, property predictor, and molecule searcher. The model cycles through the steps of Thought, Action, and Observation. During the Thought step, the model engages in reasoning by receiving information from the user prompt and using zero-shot chain-of-thought (CoT)57 to perform logical inferences. In the Action step, the model selects the appropriate tool and provides the necessary inputs based on the reasoning from the Thought step to execute current task. Finally, in the Observation step, the model receives the output from the tool and uses this information for the next iteration of the Thought step. By iteratively repeating these steps, the model effectively combines reasoning with actions to accomplish complex tasks.

The web searcher module utilizes an LLM-controlled fused search architecture. Queries from the planner are processed by the searcher's internal GPT-5 and are sent to the Serper API58 to retrieve results from Google Scholar. For each retrieved article, the module then attempts to use Semantic Scholar API59 to obtain more detailed abstract information. Following this, a general web search is employed as a fallback mechanism, which ensures the tool can adapt to diverse scenarios, ranging from IL paper searches to general information retrieval. Importantly, such a search-and-retrieval framework not only compensates for the inherent incompleteness of the model’s internal knowledge, but also markedly reduces hallucination by grounding responses in externally verifiable sources. This Retrieval-Augmented Generation (RAG)60 capability enhances both the factual reliability and adaptability of the overall agent system.

In contrast to the versatile web searcher, the PubChem searcher module is specialized for chemical structure retrieval, converting molecule or ion names/synonyms into SMILES strings. While it first attempts a standard PubChem search61, it also leverages an internal LLM to address database gaps for ionic species. The model provides reasoning to generate or correct results, such as converting a retrieved neutral form into its proper charged state.

The retrieved information is then prepared by two dedicated components. The data processor, a Python code interpreter, handles the processing of data and results. Subsequently, the SMILES canonicalizer, an RDKit62 tool, normalizes the inputs for the pipelines. Once this preparation is complete, the planner extracts

the essential data and inputs it into the property predictor. This essential data primarily includes the IL's structural information and corresponding information such as temperature. The predictor is a multimodal foundation model (the technical details will be explained in Section 2.2 below) that fuses IL molecular sequences with their graph representations, trained on an IL dataset containing ~100000 samples. We curated the dataset from existing literature by using automated scripts for an initial retrieval of approximately 10000 papers, subsequently applying LLMs for data extraction (such as text retrieval and OCR), and performing manual verification. The predictor currently estimates two broad classes of properties: the solute-solvent interactions related to ILs and their bulk characteristics (as shown in Fig. 1B). The former includes solvation free energy (ΔG), transfer free energy, and hydration free energy, while the latter encompasses the melting point, surface tension, viscosity, and mass density of bulk ILs.

Considering the real-world demand in chemical experiments for identifying ideal solvent and solute candidates, we designed a molecule searcher module to effectively explore the chemical space. Given the complexity of IL systems and the scarcity of relevant data, generative models often fail to produce chemically realistic molecules efficiently. To overcome this, we have transformed the IL discovery task into a heuristic search problem, enabling a more controlled and feasible navigation of this space. As illustrated in Fig. 1C, this module's pipeline leverages the property predictor to first identify the Top-K optimal ionic pairs or solutes from our property dataset. These serve as starting points for a beam search conducted in both our dataset of known IL systems and external databases (e.g., PubChem), guided by Tanimoto similarity63. This approach enables the discovery of more potential IL candidates within the model's generalization range. The most promising of these are then prioritized for wet-lab validation.

In summary, this ReAct-driven pipeline empowers AIonopedia to move seamlessly from information gathering to property prediction and IL screening, offering a powerful, end-to-end solution to accelerate the traditional research workflow.

[Figure 1]

: User prompt

modules based on GPT-5 modules based on AIonopedia-trained models

- (A)
- (B) (D)
- (C)

###### AIonopedia

Knowledge extraction & curation

###### Wet-lab validation

Data processor

Property predictor

[Figure 2]

###### Python code interpreter: · Process data and results

###### Multimodal-LLM based predictor:

[Figure 3]

acquisition of primary literatures

· Solute-Solvent interaction · Bulk property

multi-step experimental operations

SMILES canonicalizer

Molecule searcher

Planner

[Figure 4]

RDKit tool: · Normalize input for pipelines

GPT-5 plan & execute: · Reason and choose the action

###### Exhaustive/Beam search:

Expert-led, LLM/script-assisted processing

· Search for optimal solute-solvent pairs

· Multi-level hierarchical search

Pubchem searcher

Web searcher

[Figure 5]

[Figure 6]

Cross-source retrieval: · Query scholarly databases

Name to SMILES: · Query PubChem by names/synonyms · Handle edge cases and disambiguate via LLM

property data aggregation

+ web for ILs · Return concise summaries with references

results characterization

###### Solute-Solvent interaction

Property predictor

Solvation

###### Transfer Hydration

- 1. Modality alignment stage

- 2. Momentum Graph-Text Contrastive Loss

[Figure 7]

[Figure 8]

[Figure 9]

Organic solvent/Water

Gas

Gas

Graph-Text Matching Loss

Labeled Supervised Loss

Ionic liquid

Ionic liquid

Water

[Figure 10]

[Figure 11]

[Figure 12]

pre-defined system features output

FFN

FFN

Graph-Text Grounding Loss

concatenate

###### Bulk property

cross-modal embedding cross-modal embedding

Surface tension

Melting point Mass density Viscosity

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Cross attention module

Cross attention module

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Attention

Attention

###### Molecule searcher

sequence embedding

fused graph embedding

[Figure 42]

[Figure 43]

[Figure 44]

###### Momentum Graph-Text Contrastive Loss

Graph fusion encoder

LLM projector

per-dataset ILs existing ILs / PubChem database

Large Language Model

[Figure 45]

⊕

[Figure 46]

raw multi-graph embedding

Exhaustive search

Property predictor

GNN projector

sequence representation

Temp projector

Graph transformer

###### …

temperature

Beam search

top K results

solute graph

anion graph

cation graph

organic solvent graph

[Figure 47]

[Figure 48]

graph queue

molecule similarity e.g.

[Figure 49]

…

[Figure 50]

InfoNCE Loss

[Figure 51]

BMIM

Momentum graph encoder

LLM encoder

HMIM

Avg

language modal

graph modal

3. Finetuning stage

property prediction

Momentum LLM encoder

concatenate New FFN

Graph encoder

cross-modal embedding

cross-modal embedding

InfoNCE Loss

… language queue

Two-tower base model

……

- Fig. 1. (A) The overall workflow incorporating AIonopedia, illustrating the closed-loop process from dataset collection to final application in wet-lab validation. (B) Property categories covered in the dataset. (C) The architecture of the molecule searcher, based on beam search with Tanimoto similarity. (D) The architecture of

the property predictor, a multimodal foundation model for the IL domain developed via a two-stage training strategy.

- 2.2. Multi-stage training of the multi-modal LLM. To capture rich molecular semantics, our property predictor is inspired by a series of multimodal

alignment works45,64-68, which utilize contrastive learning to train a dual-tower multimodal model for molecule text and graph. The language view offers effortless, multi‑molecule context, whereas the graph view preserves topology with permutation‑equivariant embeddings. Contrastively aligning them fuses these complementary strengths, giving every molecule a unified, chemically grounded augmentation. At a high level, the model consists of an LLM-based language encoder and a graph-transformer69-based graph encoder, with cross-modality attention modules stacked on top to fuse the two views. Unlike traditional single molecule contrastive learning, we treat the entire multi‑molecule system as a single, holistic sample (e.g. 1-Butanol in [BMPyrr]+[B(CN)4]- at 298.15K). This enables deeper information fusion than merely concatenating per-molecule embeddings at the output layer. An architecture overview of the predictor is presented in Fig. 1D.

Although our labeled dataset is limited in size, it spans a wide range of supervised properties, whereas unlabeled molecular data are far more abundant. To leverage both sources effectively, we adopt a two‑stage training strategy, beginning with a modality alignment stage followed by a finetuning stage. For the alignment phase of the model, we gathered a significantly larger unlabeled molecule dataset and performed random sampling of these molecules to compose synthetic data samples, which are random combinations of cations, anions, organic solvents, and solutes prepared at a fixed temperature. Each sample is annotated with pseudolabels that integrate pre-computed physicochemical descriptors with categorical tags specifying the molecular composition and system temperature, driving the self-supervised training. Detailed information on the dataset is provided in Section 3.1 below.

Once the synthetic set is prepared, we encode its language modality with an LLM to obtain the corresponding embeddings, which serve as one branch of the contrastive objective. While these LLMs are typically based on decoder-only architectures, they possess strong representation capabilities owing to their vast number of parameters. Particularly, LLMs trained on scientific corpora to augment domain-specific capabilities are intrinsically adapted to tasks requiring the comprehension of chemical problems and molecular representation. The LLMs were fine-tuned using the Low-Rank Adaptation (LoRA)70 method during the training process, which significantly reduced the computational resources and GPU memory consumption required for training. Correspondingly, the four types of molecular data (cation, anion, solute, organic solvent)

for the graph modality are fed into the same graph encoder. The extracted encodings of both modalities are transformed using a projector, consistent with the approach of LLaVA-1.571, where the projector is implemented as a two-layer MLP. Next, the molecular graphs of all constituent molecules are concatenated with the temperature feature and passed through a lightweight transformer‑based graph‑fusion encoder, which integrates the signals into a single graph‑modality embedding for the complete system. After obtaining the embeddings for both modalities, we take the last token from each as the representation and apply momentum contrastive learning72 with InfoNCE loss73.

On top of encoders, we add two cross-attention decoders to fuse modalities, where each modality's embeddings serve as queries for the other's key-value pairs. To guide both encoders and decoders, three loss functions are introduced. Among these, Graph-Text Grounding (GTG) task employs an autoregressive cross-entropy loss, computed by the decoder that receives text queries. Meanwhile, graph-text matching (GTM) task employs a binary-classification loss to determine whether the two modalities originate from the same molecular system. Apart from the two losses previously used in BLIP-266, a supervised loss which combines a MSE term with a cross-entropy term in a weighted sum is utilized to quantify the discrepancy between model outputs and pseudo labels. The GTM and supervised branches share the concatenated final tokens from both modalities, which are then fed into two separate Feed-Forward Networks (FFN).

In the fine‑tuning phase, we replace the alignment phase FFNs with task‑specific regression heads and continue training all model parameters, effectively specializing the model for the different property regression tasks.

- 3. The multi-property IL dataset. We collected and assembled massive labeled IL datasets from a vast body of literature to ensure the

robustness of our property predictor. Fig. 2A presents the number of data points and the diversity of IL species in each dataset on a logarithmic scale. Although the organic-water transfer dataset contains no IL species (plotted as 1 on the log scale), we include it because subtracting these values from the IL-water results yields the IL-organic transfer ΔG. All datasets were designed for broad substance diversity to prevent any single class of similar molecules from dominating training process and introducing bias.

[Figure 52]

- Fig. 2. (A) The bar plot of dataset samples and IL counts. From left to right: solvation ΔG, transfer ΔG (IL/water), transfer ΔG (organic/water), mass density, viscosity, surface tension, melting point, and the modality-alignment dataset. (B) 1D and 2D KDE distributions for six training properties (the training data for hydration ΔG is indirectly provided by solvation ΔG and transfer ΔG data). (C) Molecular similarity heatmap of finetuning dataset (1159 cations, 287 anions, and 328 neutral molecules). (D) Comparison of tSNE dimensionality reduction results using specific descriptors and molecular similarity.
- 3.1. Synthetic data for the modality alignment. In contrast to ILBERT74 and other works27,36 in the field of ILs, we did not leverage ultra-large databases

such as ZINC2075 directly in Stage 1. Fewer than 10000 ILs are known so far, and the most comprehensive resource, ILThermo76, encompasses only about 3000 entries. Consequently, introducing an excessively broad spectrum of non-IL ion species during pretraining or alignment would inevitably impart undue prior bias. At the same time, given that our language encoder has already been pretrained, further scaling with ultra-large datasets is not necessary. Instead, we adopted a quality-over-quantity strategy and performed self-supervised modality alignment training on carefully curated IL systems. By augmenting the diversity of existing ILs through similarity screening in large databases and combinatorially sampling the components to reduce redundancy, we generated 2.8 million synthetic virtual-system data points. The resulting dataset falls into four broad categories, corresponding to IL-solute interactions, organic solvent-solute interactions, IL bulk properties including temperature, and IL bulk properties excluding temperature.

The pseudo labels, used as targets for Stage 1 labeled supervised loss training, comprised a 21dimensional descriptor representation for the four types of molecules, the system’s temperature feature, and a 4-dimensional one-hot vector encoding the format classification. The descriptors include the number of hydrogen bond donors, the number of hydrogen bond acceptors, the number of rotatable bonds, the polar surface area,77 the number of atomic stereocenters, the octanol-water partition coefficient log P and molecular reactivity with the Crippen’s approach,78 the fraction of sp3 carbon, the number of rings, the number of heterocycles, the number of aromatic rings, the number of aromatic heterocycles, the number of spiro atoms, the molecular weight, the number of heteroatoms, the number of heavy atoms, the kappa1&2&3 shape indices79, the Balaban J index80 and the Bertz CT index81. These chemical properties provide informative characterizations of a given molecule.

Additionally, for each molecule we constructed a graph object and traversed the atom indices in the order defined by the canonical SMILES. The node features represent atomic properties including atom type, degree, hybridization, implicit valence, aromaticity, formal charge, and hydrogen bonding potential, while the edge features capture bond characteristics including bond type, stereochemistry, conjugation, and ring participation. By leveraging synthetic unlabeled data in quantities far exceeding the labeled data, we substantially improved the model’s performance across all metrics. The comparison results are provided in Section 4.4.

- 3.2. Experimental datasets. We included one-dimensional label distributions of datasets for six properties, alongside their

temperature-expanded, two-dimensional counterparts, as shown in Fig. 2B. Since melting points have no temperature dependence and transfer ΔG are only reported in the literature at 298K, their 2D distributions are

omitted. Fig. 2C shows the Tanimoto similarity heatmap for all molecules used during fine-tuning, including 1159 cations, 287 anions, and 328 neutral molecules. To comprehensively characterize each molecule, we computed four fingerprint types: ECFP, MACCS, atom-pair and PubChem, which are based on RDKit and chemfp82, and applied hierarchical clustering to enhance visualization clarity. The low overall internal similarity, as indicated by the predominantly dark matrix with sparse highlights, underscores the breadth of our chemical coverage.

For the datasets we collected on solvation ΔG and biphasic transfer ΔG properties, they cover about 80 ionic solvents and 150 solutes, which are the largest datasets covering these properties as far as we know. Given the thermodynamic link allowing hydration ΔG to be determined from the solvation and transfer ΔG, our dataset intrinsically defines the hydration behavior of solutes. To explicitly evaluate this implicit hydration behavior, we generated ten hypothetical IL systems per solute by sampling novel ion combinations from the solvent-solute dataset, excluding any known pairs to prevent data leakage. These virtual ILs are used solely to construct a held-out hydration ΔG benchmark (Section 4.3), on which all models are evaluated.

Aside from solute-solvent interactions involved in the above-mentioned datasets, we additionally curate datasets for bulk properties. The considered bulk properties include mass density, viscosity, surface tension and melting points. Compared to the aforementioned datasets, our bulk properties dataset covers a larger number of ILs, with nearly 6,000 systems in total, encompassing approximately 1,200 cations and 300 anions.

Additionally, Fig. 2D presents t-SNE projections based on both our descriptor set and traditional molecular-similarity metrics, revealing a pronounced separation between IL-forming ions and neutral, gas- or drug-like molecules. Crucially, the markedly sharper separation achieved with our descriptors unequivocally validates their effectiveness in capturing the key chemical distinctions between these categories. To verify local concordance, we also mark the locations of 1-butyl-3-methylimidazolium ([Bmim]+) and 1-hexyl-3methylimidazolium ([Hmim]+) in each t-SNE plot. Under our descriptors, these two ions remain proximate, consistent with their underlying chemical similarity.

##### 4. Results.

- 4.1. Data extraction capability of the agent. Reliable automated data acquisition for ILs critically depends on mapping informal notations in the

literature to standardized molecular representations. In the field of ILs, researchers commonly use abbreviations to replace the lengthy ion names83. Because these abbreviations lack standardized forms and are sparsely indexed in databases, they are difficult for conventional workflows to handle (e.g., N-

octyltrimethylammonium may be abbreviated as both [C8TA]+ and [N1118]+). To systematically characterize this data-extraction challenge, we define a benchmark task that requires methods to convert an ion abbreviation directly into its corresponding canonical SMILES and full name.

We used a dataset containing 50 ions to compare the results of four web-enabled SOTA LLMs, as shown in Fig. 3A. Given the diversity of ion full names and the potential non-uniqueness of canonical SMILES (e.g. due to delocalized charge), the results were evaluated by human experts. We reported metrics in decreasing order of difficulty: (i) Canonical SMILES Match: direct match of the expected SMILES. (ii) Structural Match: match of the expected SMILES after manual canonicalization. (iii) Full Name Match: correct full name provided.

Among these, GPT-5 significantly outperformed the other LLMs on all metrics and was therefore selected as the core LLM for AIonopedia. To further evaluate the efficacy of our agent, we compared the performance of the full AIonopedia system against GPT-5 with varying degrees of tool integration, as shown in Fig. 3B. Owing to its rich tool interactions and coordinated communication across multiple LLMs, AIonopedia comprehensively surpassed the compared baselines. It proved particularly effective on the strictest metric (i), boosting the accuracy to 94.7%, and was thus validated as a more effective problem-solving approach.

## (A) (B)

### LLM + Web Search Benchmark

### Tool Integration Benchmark (GPT-5)

- 0.8
- 1

Accuracy

0.6

0.4

0.2

0

Canonical SMILES Match

Structural Match

Full Name Match

- 0.8
- 1

Accuracy

0.6

0.4

0.2

0

Canonical SMILES Match

Structural Match

Full Name Match

GPT-5 Gemini 2.5 pro

Claude 4.5 sonnet DeepSeek-R1

AIonopedia GPT-5 GPT-5(w/o web)

Fig. 3. The model effect on the abbreviation interpretation for ions in ILs. (A) Performance comparison of four different LLMs84-87 equipped with a base search module. (B) Effect of tool integration on GPT-5 performance.

- 4.2. Evaluation of the property predictor across the various property datasets.

In this study, we selected Meta AI's Galactica88 series, Alibaba's Qwen389 series as well as Google DeepMind’s Gemma390 series to initialize the language encoder of the property predictor. Galactica and Qwen3 both explicitly state in their technical reports that they were trained on scientific texts to enhance domain knowledge, whereas Gemma3 serves as a general‑purpose LLM control that makes no such claim. This setup allows us to assess how domain-specific pre-training affects downstream property prediction. We then benchmarked these property predictor variants against a comprehensive suite of chemistry-domain baselines. This suite was carefully selected to be representative across two key dimensions: architecture, including both encoder- and decoder-based models, and modality, covering both unimodal and multimodal LLMs.

We employ multi-level 5-fold cross-validation splits for each dataset to mitigate overfitting91. For IL bulkproperty datasets, we adopted cation-based and IL-based splits; for the ternary solute-solvent interaction dataset, we additionally introduced a split based on all three components. Hyperparameters derived from training with the strictest cation-based split were subsequently applied to the other splits. The detailed results are shown in Table 1-3.

Considering the training cost and to ensure a fair comparison with prior work, we fine‑tuned most LLMs and multimodal LLMs using LoRA. For each model, we followed the default LoRA settings from their respective GitHub repositories, except for PRESTO92, where we lowered the default rank to match that used in works such as LlasMol93. This adjustment was made because PRESTO typically utilizes a much larger LoRA rank compared to other methods, which can lead to significantly higher resource consumption. Since SPMM68 and ILBERT employ smaller, lightweight BERT40 and RoBERTa94 encoders, we utilized full finetuning instead. Furthermore, since data analysis in the IL domain typically relies on traditional machine learning methods or lightweighted neural networks rather than large pretrained models, we included an MLP baseline for comparison. This baseline was trained on the physicochemical descriptors predefined during our modality-alignment phase, confirming that our model’s performance does not depend solely on these pseudo-labels.

Our model achieved the best performance across all metrics on nearly every dataset. In particular, the Qwen3-0.6b based model delivered the best average results, ranking first on 20 evaluated metrics and showing especially striking performance on several solute-solvent pair datasets (Table 2). By contrast, excluding the smallest version based on Galactica-125m, among the other three models of similar size, the Gemma3-1b version performed the worst due to its lack of training on scientific texts. Thanks to LoRA training, our model’s number of trainable parameters is roughly on par with the fully fine‑tuned BERT‑based models and lower than several approaches that employ 7-8 billion‑parameter models. Among the baselines, ILBERT delivered the

strongest overall performance, likely because it was pretrained on a large volume of ionic data, and its results on the melting‑point dataset were especially impressive. However, both encoder‑style models, ILBERT and SPMM, fall far behind the other models on the viscosity dataset. We attribute this gap to the strong length sensitivity of viscosity: the two encoders aggregate features with a 1‑D CNN or a single [CLS] token, making them inherently less responsive to sequence length than decoder‑style LLMs that read the final token. Although ILBERT’s original paper reports higher viscosity accuracy than our reproduction, we believe the difference arises from variations in dataset collection and from our stricter cation‑based split. In contrast, the MLP baseline sidesteps the issue because its input descriptors include molecular weight.

To give a general survey, we averaged the three evaluation metrics for each dataset together with the results from the different split schemes, yielding an overall rank for every model across the seven datasets. These ranks were then visualised as radar charts that compare the four AIonopedia variants with one another and with the baseline models, as shown in Fig. 4, which visually validates the leading position of the Qwen30.6b variant.

Table 1. Comparison of AIonopedia with baselines under cation-based splits; the organic solvent/water transfer ΔG dataset uses a solvent-based split.

IL/water transfer ΔG (kcal/mol)

viscosity (mPa·s) (log scale)

organic solvent/water transfer ΔG (kcal/mol)

solvation ΔG (kcal/mol)

melting point (K)

surface tension (mN/m)

mass density (g/cm3)

RMSE (↓)

0.294±0.036 (0.2939)

3.86±0.41 0.0327±0.0032 0.530±0.039

AIonopedia (Galactica-125m) 0.364±0.098 0.483±0.123 40.4±3.9

AIonopedia (Galactica-1.3b) 0.322±0.136 0.457±0.108 40.3±3.8 0.298±0.038 3.77±0.24 0.0330±0.0047 0.529±0.051

0.294±0.031 (0.2943）

3.62±0.35 0.0333±0.0068 0.534±0.038

AIonopedia (Qwen3-0.6b) 0.328±0.130 0.441±0.117 39.9±3.4

AIonopedia (Gemma3-1b) 0.325±0.136 0.459±0.095 40.6±3.2 0.305±0.026 3.66±0.33 0.0330±0.0060 0.538±0.061 MLP (pretrain descriptors) 0.417±0.106 0.755±0.093 45.5±5.3 0.411±0.048 4.48±0.48 0.0437±0.0067 0.878±0.052 MolCA67 (Galactica-1.3b) 0.421±0.078 0.732±0.176 46.9±1.9 0.384±0.049 5.81±0.47 0.0525±0.0064 0.846±0.222 T5chem95,96 0.379±0.104 0.576±0.104 44.9±2.5 0.317±0.038 3.94±0.23 0.0357±0.0043 0.702±0.067 Molinst (Llama3-8b)97,98 0.352±0.098 0.510±0.159 42.4±3.2 0.310±0.036 4.08±0.43 0.0331±0.0031 0.678±0.076 SPMM (full finetune)68 0.445±0.090 0.531±0.118 46.4±3.9 0.588±0.018 4.67±0.36 0.0488±0.0053 0.670±0.098 LlaSMol (Mistral-7b)93,99 0.337±0.133 0.492±0.150 42.6±4.9 0.310±0.019 3.85±0.27 0.0340±0.0027 0.674±0.087 PRESTO (Vicuna v1.57b)92,100

0.400±0.144 0.472±0.120 41.2±4.6 0.309±0.031 3.97±0.46 0.0442±0.0046 0.723±0.124

ILBERT (full finetune)74 0.334±0.105 0.459±0.169 39.7±4.3 0.536±0.025 3.96±0.28 0.0377±0.0067 0.636±0.031

IL/water transfer ΔG

viscosity (log scale)

organic solvent/water transfer ΔG

Pearson r (↑) solvation ΔG

melting point

surface tension mass density

AIonopedia (Galactica-125m) 0.9540±0.0213 0.9798±0.0103 0.7186±0.0305 0.9098±0.0198 0.9123±0.0120 0.9845±0.0038 0.9754±0.0044 AIonopedia (Galactica-1.3b) 0.9592±0.0300 0.9819±0.0081 0.7201±0.0321 0.9065±0.0211 0.9123±0.0109 0.9839±0.0051 0.9751±0.0059 AIonopedia (Qwen3-0.6b) 0.9563±0.0279 0.9837±0.0080 0.7191±0.0283 0.9095±0.0146 0.9187±0.0097 0.9835±0.0065 0.9742±0.0050 AIonopedia (Gemma3-1b) 0.9581±0.0279 0.9830±0.0066 0.7113±0.0178 0.9022±0.0033 0.9162±0.0083 0.9840±0.0056 0.9734±0.0067

MLP 0.9253±0.0326 0.9462±0.0061 0.6354±0.0584 0.8109±0.0313 0.8686±0.0168 0.9718±0.0078 0.9282±0.0129 MolCA 0.9319±0.0174 0.9488±0.0192 0.5977±0.0521 0.8579±0.0164 0.7815±0.0436 0.9598±0.0079 0.9411±0.0219 T5chem 0.9510±0.0251 0.9813±0.0088 0.6824±0.0279 0.9023±0.0189 0.9059±0.0111 0.9830±0.0043 0.9632±0.0104 Molinst 0.9510±0.0204 0.9757±0.0136 0.6697±0.0469 0.8978±0.0179 0.8917±0.0251 0.9839±0.0031 0.9559±0.0123 SPMM 0.9230±0.0284 0.9773±0.0102 0.6165±0.0404 0.5746±0.0527 0.8683±0.0086 0.9659±0.0077 0.9584±0.0168 LlaSMol 0.9531±0.0284 0.9771±0.0117 0.6654±0.0395 0.8971±0.0150 0.9026±0.0187 0.9830±0.0035 0.9575±0.0126 PRESTO 0.9366±0.0348 0.9790±0.0090 0.6969±0.0482 0.8984±0.0072 0.8996±0.0128 0.9709±0.0055 0.9493±0.0193 ILBERT 0.9517±0.0277 0.9797±0.0136 0.7204±0.0381 0.6446±0.0464 0.9015±0.0177 0.9793±0.0062 0.9629±0.0060

IL/water transfer ΔG

viscosity (log scale)

organic solvent/water transfer ΔG

Kendall τ (↑) solvation ΔG

melting point

surface tension mass density

AIonopedia (Galactica-125m) 0.843±0.027 0.883±0.016 0.515±0.029 0.803±0.022 0.778±0.020 0.911±0.010 0.878±0.014 AIonopedia (Galactica-1.3b) 0.865±0.043 0.890±0.016 0.508±0.046 0.800±0.025 0.772±0.031 0.912±0.015 0.881±0.016 AIonopedia (Qwen3-0.6b) 0.860±0.046 0.893±0.011 0.505±0.022 0.800±0.021 0.780±0.022 0.908±0.015 0.879±0.012 AIonopedia (Gemma3-1b) 0.864±0.041 0.892±0.014 0.500±0.028 0.795±0.010 0.776±0.022 0.911±0.010 0.877±0.019 MLP 0.781±0.041 0.793±0.016 0.466±0.027 0.683±0.030 0.706±0.016 0.870±0.017 0.768±0.018 MolCA 0.802±0.017 0.821±0.027 0.395±0.056 0.710±0.021 0.624±0.039 0.843±0.023 0.807±0.033 T5chem 0.843±0.030 0.885±0.011 0.471±0.035 0.783±0.020 0.746±0.027 0.907±0.005 0.849±0.017 Molinst 0.853±0.030 0.874±0.013 0.460±0.055 0.775±0.022 0.740±0.036 0.904±0.013 0.835±0.016 SPMM 0.778±0.033 0.875±0.021 0.400±0.050 0.410±0.049 0.692±0.020 0.853±0.029 0.845±0.025 LlaSMol 0.854±0.049 0.868±0.018 0.452±0.049 0.774±0.014 0.759±0.020 0.903±0.023 0.833±0.020 PRESTO 0.825±0.057 0.877±0.015 0.484±0.051 0.775±0.014 0.749±0.012 0.874±0.015 0.825±0.029 ILBERT 0.844±0.047 0.883±0.019 0.514±0.042 0.455±0.047 0.735±0.034 0.889±0.008 0.850±0.010

Table 2. Comparison of AIonopedia with baselines under IL-based and ternary-component-based splits on the solute-solvent interaction datasets; the organic solvent/water transfer ΔG dataset uses a solute-solventbased split.

solvation ΔG (ternary-component split) (kcal/mol)

IL/water transfer ΔG (ternary-component split) (kcal/mol)

organic solvent/water transfer ΔG (kcal/mol)

solvation ΔG (kcal/mol)

IL/water transfer ΔG (kcal/mol)

RMSE (↓)

AIonopedia (Galactica-125m) 0.304±0.030 0.473±0.100 0.155±0.013 0.373±0.043 0.415±0.028 AIonopedia (Galactica-1.3b) 0.321±0.029 0.460±0.121 0.126±0.012 0.265±0.029 0.399±0.030

0.393±0.035 (0.3930)

AIonopedia (Qwen3-0.6b) 0.309±0.044 0.440±0.132 0.124±0.016 0.238±0.013

0.393±0.032 (0.3934）

AIonopedia (Gemma3-1b) 0.311±0.039 0.459±0.121 0.125±0.017 0.250±0.013

MLP 0.441±0.065 0.880±0.249 0.361±0.023 0.742±0.043 0.818±0.035 MolCA 0.400±0.073 0.630±0.051 0.341±0.021 0.644±0.056 0.724±0.012 T5chem 0.371±0.027 0.588±0.137 0.240±0.013 0.507±0.041 0.560±0.019 Molinst 0.352±0.038 0.510±0.124 0.178±0.019 0.275±0.005 0.468±0.032 SPMM 0.425±0.016 0.550±0.111 0.307±0.023 0.427±0.082 0.575±0.053 LlaSMol 0.315±0.037 0.591±0.132 0.160±0.016 0.336±0.036 0.513±0.033 PRESTO 0.413±0.144 0.530±0.088 0.160±0.032 0.259±0.022 0.541±0.089 ILBERT 0.323±0.051 0.452±0.124 0.147±0.024 0.260±0.030 0.404±0.042

solvation ΔG (ternary-component split)

IL/water transfer ΔG (ternary-component split)

organic solvent/water transfer ΔG

Pearson r (↑) solvation ΔG IL/water transfer ΔG

AIonopedia (Galactica-125m) 0.9625±0.0056 0.9801±0.0104 0.9904±0.0021 0.9880±0.0028 0.9849±0.0025 AIonopedia (Galactica-1.3b) 0.9601±0.0051 0.9807±0.0116 0.9939±0.0014 0.9944±0.0011 0.9861±0.0021

AIonopedia (Qwen3-0.6b) 0.9615±0.0099 0.9816±0.0113 0.9942±0.0018 0.9953±0.0006 0.9862±0.0024 AIonopedia (Gemma3-1b) 0.9625±0.0076 0.9803±0.0118 0.9940±0.0018 0.9950±0.0007 0.9860±0.0023 MLP 0.9185±0.0237 0.9200±0.0490 0.9444±0.0085 0.9465±0.0062 0.9360±0.0113 MolCA 0.9340±0.0203 0.9626±0.0062 0.9527±0.0056 0.9616±0.0069 0.9529±0.0038 T5chem 0.9512±0.0060 0.9785±0.0123 0.9815±0.0030 0.9874±0.0023 0.9786±0.0028 Molinst 0.9504±0.0082 0.9763±0.0121 0.9868±0.0027 0.9929±0.0002 0.9798±0.0030 SPMM 0.9272±0.0040 0.9789±0.0085 0.9633±0.0046 0.9892±0.0005 0.9713±0.0044 LlaSMol 0.9605±0.0091 0.9681±0.0158 0.9893±0.0023 0.9892±0.0022 0.9755±0.0044 PRESTO 0.9349±0.0371 0.9743±0.0095 0.9890±0.0043 0.9936±0.0009 0.9721±0.0092 ILBERT 0.9585±0.0108 0.9801±0.0110 0.9909±0.0030 0.9935±0.0015 0.9849±0.0028

solvation ΔG (ternary-component split)

IL/water transfer ΔG (ternary-component split)

organic solvent/water transfer ΔG

Kendall τ (↑) solvation ΔG IL/water transfer ΔG

AIonopedia (Galactica-125m) 0.870±0.019 0.883±0.016 0.944±0.002 0.910±0.008 0.906±0.005 AIonopedia (Galactica-1.3b) 0.863±0.019 0.888±0.018 0.956±0.002 0.938±0.005 0.914±0.004 AIonopedia (Qwen3-0.6b) 0.866±0.031 0.895±0.020 0.962±0.002 0.944±0.002 0.916±0.005 AIonopedia (Gemma3-1b) 0.870±0.014 0.885±0.020 0.963±0.002 0.943±0.003 0.915±0.006 MLP 0.788±0.017 0.772±0.040 0.816±0.009 0.798±0.011 0.780±0.015 MolCA 0.804±0.031 0.849±0.010 0.829±0.007 0.847±0.010 0.828±0.007 T5chem 0.848±0.016 0.880±0.018 0.919±0.003 0.907±0.007 0.884±0.004 Molinst 0.847±0.022 0.874±0.015 0.946±0.003 0.931±0.003 0.890±0.009 SPMM 0.780±0.015 0.878±0.022 0.847±0.006 0.913±0.002 0.897±0.005 LlaSMol 0.860±0.017 0.851±0.027 0.941±0.006 0.908±0.010 0.877±0.013 PRESTO 0.830±0.057 0.867±0.017 0.956±0.003 0.938±0.003 0.870±0.023 ILBERT 0.861±0.016 0.887±0.020 0.947±0.002 0.936±0.008 0.911±0.007

- Table 3. Comparison of AIonopedia with baselines under IL-based splits on the bulk property datasets.

RMSE (↓) melting point (K) viscosity (mPa·s) (log scale) surface tension (mN/m) mass density (g/cm3)

0.249±0.018 (0.2493)

3.63±0.27 0.0272±0.0029

AIonopedia (Galactica-125m) 38.4±3.0

0.249±0.020 (0.2492)

3.42±0.27 0.0267±0.0030

AIonopedia (Galactica-1.3b) 38.5±2.5

AIonopedia (Qwen3-0.6b) 38.9±3.5 0.250±0.017 3.47±0.35 0.0266±0.0028 AIonopedia (Gemma3-1b) 39.6±3.0 0.251±0.020 3.44±0.32 0.0264±0.0032 MLP 44.4±3.6 0.368±0.022 4.33±0.31 0.0427±0.0036 MolCA 48.2±4.0 0.332±0.011 5.26±0.25 0.0499±0.0028 T5chem 43.4±3.2 0.288±0.025 3.77±0.28 0.0323±0.0013 Molinst 39.7±3.0 0.268±0.018 3.50±0.44 0.0280±0.0024 SPMM 45.5±3.3 0.545±0.023 4.59±0.37 0.0417±0.0026 LlaSMol 42.1±3.2 0.267±0.016 3.59±0.13 0.0289±0.0021 PRESTO 42.4±2.2 0.268±0.019 3.63±0.34 0.0468±0.0045 ILBERT 37.2±3.2 0.507±0.019 3.51±0.20 0.0323±0.0028

Pearson r (↑) melting point viscosity (log scale) surface tension mass density AIonopedia (Galactica-125m) 0.7470±0.0380 0.9393±0.0080 0.9194±0.0160 0.9895±0.0021 AIonopedia (Galactica-1.3b) 0.7476±0.0471 0.9392±0.0105 0.9272±0.0127 0.9898±0.0021 AIonopedia (Qwen3-0.6b) 0.7408±0.0613 0.9389±0.0093 0.9242±0.0184 0.9897±0.0021 AIonopedia (Gemma3-1b) 0.7289±0.0533 0.9378±0.0104 0.9261±0.0162 0.9899±0.0024 MLP 0.6537±0.0473 0.8575±0.0192 0.8790±0.0199 0.9727±0.0060 MolCA 0.5728±0.0392 0.8949±0.0079 0.8187±0.0321 0.9641±0.0058 T5chem 0.7202±0.0401 0.9279±0.0103 0.9185±0.0131 0.9862±0.0022

Molinst 0.7254±0.0370 0.9265±0.0108 0.9210±0.0199 0.9885±0.0020 SPMM 0.6435±0.0419 0.6593±0.0282 0.8742±0.0214 0.9761±0.0019 LlaSMol 0.6767±0.0567 0.9271±0.0101 0.9182±0.0100 0.9877±0.0022 PRESTO 0.6806±0.0214 0.9270±0.0118 0.9156±0.0183 0.9669±0.0099 ILBERT 0.7620±0.0291 0.7034±0.0196 0.9214±0.0101 0.9849±0.0023

Kendall τ (↑) melting point viscosity (log scale) surface tension mass density AIonopedia (Galactica-125m) 0.546±0.031 0.834±0.015 0.800±0.021 0.931±0.004 AIonopedia (Galactica-1.3b) 0.543±0.051 0.838±0.016 0.807±0.014 0.933±0.004

0.934±0.005 (0.9337)

AIonopedia (Qwen3-0.6b) 0.548±0.048 0.835±0.017 0.804±0.022

0.934±0.005 (0.9336）

AIonopedia (Gemma3-1b) 0.536±0.049 0.833±0.015 0.807±0.021

MLP 0.472±0.046 0.708±0.018 0.721±0.018 0.876±0.009 MolCA 0.373±0.032 0.754±0.012 0.662±0.021 0.863±0.013 T5chem 0.516±0.046 0.815±0.012 0.781±0.018 0.918±0.005 Molinst 0.524±0.035 0.812±0.015 0.791±0.021 0.928±0.003 SPMM 0.440±0.049 0.462±0.022 0.720±0.010 0.884±0.004 LlaSMol 0.475±0.064 0.809±0.015 0.787±0.022 0.924±0.004 PRESTO 0.488±0.023 0.814±0.023 0.792±0.011 0.889±0.015 ILBERT 0.568±0.025 0.497±0.016 0.772±0.019 0.910±0.004

(A) (B)

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

- Fig. 4. (A) Average performance ranks of the four AIonopedia models across all evaluated datasets. (B) Average performance ranks of the best‑performing variant, AIonopedia (Qwen3‑0.6 b), compared with baselines across the same datasets.

- 4.3. System-specific benchmarks and comparison with traditional simulation baselines. Molecular simulations as a classical computational approach have been extensively applied to study the

bulk-phase properties and solvation behavior of ionic liquids, offering atomistic insights into their structural organization, transport mechanisms, and thermodynamic features.101-104 However, such all-atom simulations are often prohibitively slow, limited in their coverage of chemical space, and constrained by accessible time and length scales. Further, many experimentally measurable properties (e.g., viscosity and long-time relaxation dynamics) remain challenging to reproduce accurately within feasible simulation durations.

To examine the performance of AIonopedia in more depth, we further compare the predictions of the property predictor on the solute-solvent interaction dataset against our MD simulations performed with GROMACS105. Because the simulation for an IL system typically requires several hours to days of time, running simulations across the entire set of property datasets was infeasible. Therefore, we confined the computational evaluation to five specific systems with varying chemical compositions and temperatures and their temperature‑dependent subsets. Here, we adopted the IL‑based split, kept the training hyper‑parameters identical to those used for the cross‑validation split, and report the results in Table 4, with also the aforementioned large‑language‑model and traditional machine‑learning baselines

Overall, the computational method performs consistently across all datasets, whereas the other approaches vary markedly with the dataset split. For this method, the RMSE ranges from 0.524 to 0.897 kcal mol⁻¹, all favorably lie within the chemical precision. Remarkable, most machine‑learning models outperform the MD method, and our AIonopedia models again achieved the best performance on most metrics (e.g. with the RMSE

- as low as 0.060-0.464 kcal/mol for AIonopedia (Qwen3-0.6b)). Among fived studied systems, the ion pair [Quin8]+[TF₂N]- is absent as a pair in the training set, but both constituent ions are well represented in other IL systems, so almost all learning‑based models give near-perfect prediction on this system (with a Pearson r>0.99, except r=0.97 for MLP and r=0.94 for MolCA). For the two other nitrogen‑containing pairs, [BMIM]+[BETI]- and [EMIM]+[TF₂N]- with constituent ions appearing less frequently in the training data, several baselines show a noticeable drop in accuracy. Conversely, the phosphonium cations and nonfluorinated anions that form the remaining two ILs are scarcely represented in the dataset. The anion in [P66614]+[L-Lact]- is entirely absent from the training data, and neither ion in [P4442]+[DEP]- appears at all, rendering both corresponding test sets strongly OOD. On these two most difficult systems our method exhibits a pronounced advantage over all alternatives. SPMM, though weaker than our model on every other benchmark, attains comparable accuracy on the [P4442]+[DEP]- dataset. All remaining models demonstrate clear over-fitting. Especially ILBERT, which is previously the top‑ranked model, performs poorly on [P4442]+[DEP]- and even worse on [P66614]+[L‑Lact]-. In the latter case its RMSE climbs above 1 kcal mol⁻¹, surpassing the accepted boundary for chemical accuracy106,107.

To further benchmark AIonopedia models against the traditional method, we built a small mass‑density

set for representing bulk properties and a separate dataset of hydration ΔG. Experiments on the mass‑density set also kept the cation‑split hyper‑parameters. For hydration ΔG, we created ten de‑duplicated virtual IL solvents per solute, drawn from the solute-solvent dataset. Each baseline used two separately trained models, one for solvation ΔG and one for transfer ΔG (IL/water). The baseline obtained hydration ΔG by subtraction, then averaged the ten solvent values. This protocol slightly underestimates performance variance but reduces small‑sample bias and better matches real‑world usage. On both tasks our model was best, or statistically tied for best, as shown in Table 5.

- Table 4. Comparison of AIonopedia with computational baseline and prior baselines on solvation ΔG datasets for fixed systems.

298K [BMIM]+ [BETI]-

298K [EMIM]+ [TF2N]-

298K [QUIN8]+ [TF2N]-

328K [QUIN8]+ [TF2N]-

338K [QUIN8]+ [TF2N]-

298K [P66614]+ [L-Lact]-

328K [P4442]+ [DEP]-

348K [P4442]+ [DEP]-

RMSE(kcal/mol) (↓)

AIonopedia (Galactica-125m)

0.387±0.018 0.259±0.005 0.320±0.091 0.088±0.007 0.441±0.042 0.073±0.008 0.075±0.006 0.512±0.029

AIonopedia (Galactica-1.3b)

###### 0.380±0.030 0.220±0.005 0.350±0.045 0.060±0.003 0.369±0.058 0.051±0.003 0.051±0.003 0.457±0.047

AIonopedia (Qwen3-0.6b)

0.438±0.032 0.242±0.023 0.281±0.094 0.064±0.009 0.361±0.053 0.060±0.007 0.062±0.008 0.464±0.029

AIonopedia (Gemma3-1b)

0.405±0.008 0.230±0.022 0.305±0.050 0.062±0.011 0.433±0.060 0.056±0.008 0.054±0.010 0.493±0.047

MD simulation 0.806±0.052 0.524±0.042 0.731±0.052 0.682±0.068 0.697±0.019 0.897±0.045 0.597±0.055 0.547±0.012 MLP 0.558±0.024 0.405±0.014 0.573±0.087 0.226±0.016 0.451±0.046 0.220±0.007 0.220±0.007 0.564±0.018 MolCA 0.280±0.017 0.405±0.125 0.769±0.039 0.372±0.021 0.534±0.023 0.340±0.025 0.336±0.026 0.656±0.024 T5chem 0.327±0.017 0.281±0.022 0.429±0.061 0.132±0.013 0.639±0.024 0.125±0.008 0.123±0.006 0.750±0.029 Molinst 0.341±0.038 0.227±0.039 0.705±0.035 0.068±0.013 0.443±0.136 0.063±0.010 0.064±0.008 0.581±0.139 SPMM 0.358±0.012 0.445±0.050 0.449±0.126 0.309±0.030 0.365±0.032 0.159±0.028 0.283±0.032 0.581±0.045 LlaSMol 0.341±0.037 0.277±0.085 0.412±0.133 0.077±0.014 0.648±0.058 0.069±0.014 0.071±0.016 0.746±0.061 PRESTO 0.422±0.021 0.809±0.024 0.531±0.050 0.075±0.006 0.469±0.072 0.065±0.009 0.064±0.008 0.562±0.074 ILBERT 0.355±0.035 0.261±0.048 1.210±0.391 0.086±0.009 0.532±0.087 0.062±0.007 0.058±0.008 0.640±0.079

298K [BMIM]+ [BETI]-

298K [EMIM]+ [TF2N]-

298K [QUIN8]+ [TF2N]-

328K [QUIN8]+ [TF2N]-

338K [QUIN8]+ [TF2N]-

298K [P66614]+ [L-Lact]-

328K [P4442]+ [DEP]-

348K [P4442]+ [DEP]-

Pearson r (↑)

AIonopedia (Galactica-125m)

0.9952±0.0005 0.9832±0.0022 0.9815±0.0047 0.9962±0.0009 0.9450±0.0065 0.9969±0.0006 0.9969±0.0005 0.9018±0.0030

AIonopedia (Galactica-1.3b)

0.9946±0.0004 0.9904±0.0012 0.9844±0.0023 0.9982±0.0002 0.9664±0.0056 0.9984±0.0002 0.9984±0.0002 0.9250±0.0039

AIonopedia (Qwen3-0.6b)

0.9943±0.0006 0.9908±0.0028 0.9718±0.0043 0.9987±0.0004 0.9685±0.0077 0.9985±0.0004 0.9984±0.0004 0.9300±0.0038

AIonopedia (Gemma3-1b)

0.9949±0.0005 0.9905±0.0013 0.9655±0.0090 0.9986±0.0005 0.9574±0.0049 0.9986±0.0004 0.9986±0.0004 0.9180±0.0053

MD simulation 0.8710±0.0171 0.9309±0.0115 0.7684±0.0248 0.7957±0.0418 0.8940±0.0089 0.7495±0.0441 0.8231±0.0311 0.8761±0.0056 MLP 0.9486±0.0026 0.9540±0.0045 0.8117±0.0232 0.9706±0.0034 0.9098±0.0093 0.9734±0.0018 0.9729±0.0019 0.8451±0.0091 MolCA 0.9876±0.0029 0.9602±0.0275 0.7441±0.0310 0.9369±0.0066 0.9053±0.0057 0.9474±0.0059 0.9471±0.0059 0.8294±0.0032 T5chem 0.9931±0.0008 0.9776±0.0047 0.9451±0.0195 0.9931±0.0009 0.9331±0.0037 0.9931±0.0007 0.9926±0.0009 0.8783±0.0032 Molinst 0.9938±0.0008 0.9862±0.0039 0.8582±0.0164 0.9978±0.0008 0.9341±0.0093 0.9981±0.0009 0.9980±0.0007 0.8709±0.0157

SPMM 0.9910±0.0017 0.9813±0.0029 0.9364±0.0247 0.9939±0.0011 0.9722±0.0069 0.9964±0.0008 0.9965±0.0008 0.9012±0.0201 LlaSMol 0.9926±0.0011 0.9795±0.0122 0.9578±0.0120 0.9973±0.0003 0.9308±0.0112 0.9976±0.0009 0.9972±0.0012 0.8747±0.0071 PRESTO 0.9927±0.0010 0.7868±0.0132 0.8809±0.0256 0.9968±0.0007 0.9267±0.0085 0.9973±0.0008 0.9974±0.0007 0.8690±0.0120 ILBERT 0.9922±0.0019 0.9813±0.0057 0.8766±0.0478 0.9954±0.0009 0.9321±0.0097 0.9974±0.0005 0.9978±0.0006 0.8952±0.0118

298K [BMIM]+ [BETI]-

298K [EMIM]+ [TF2N]-

298K [QUIN8]+ [TF2N]-

328K [QUIN8]+ [TF2N]-

338K [QUIN8]+ [TF2N]-

298K [P66614]+ [L-Lact]-

328K [P4442]+ [DEP]-

348K [P4442]+ [DEP]-

Kendall τ (↑)

AIonopedia (Galactica-125m)

- 0.945±0.002 0.935±0.004 0.907±0.027 0.950±0.006 0.813±0.010 0.952±0.007 0.953±0.0061 0.797±0.007

AIonopedia (Galactica-1.3b)

- 0.946±0.002 0.922±0.003 0.905±0.014 0.970±0.004 0.846±0.014 0.966±0.006 0.963±0.003 0.831±0.010

AIonopedia (Qwen3-0.6b)

0.950±0.004 0.943±0.009 0.860±0.012 0.975±0.009 0.850±0.017 0.973±0.004 0.965±0.010 0.833±0.010

AIonopedia (Gemma3-1b)

0.952±0.002 0.933±0.009 0.853±0.027 0.970±0.007 0.828±0.009 0.975±0.001 0.967±0.004 0.817±0.006

MD simulation 0.678±0.030 0.733±0.027 0.574±0.033 0.620±0.053 0.705±0.015 0.555±0.045 0.651±0.032 0.702±0.011 MLP 0.849±0.004 0.864±0.008 0.617±0.025 0.850±0.010 0.753±0.007 0.873±0.005 0.860±0.011 0.724±0.003 MolCA 0.903±0.016 0.875±0.032 0.577±0.026 0.809±0.013 0.753±0.010 0.843±0.013 0.833±0.011 0.717±0.006 T5chem 0.931±0.005 0.900±0.006 0.798±0.042 0.933±0.008 0.785±0.008 0.935±0.005 0.937±0.004 0.762±0.005 Molinst 0.937±0.008 0.933±0.009 0.661±0.009 0.960±0.006 0.787±0.015 0.966±0.001 0.961±0.003 0.757±0.018 SPMM 0.932±0.004 0.941±0.003 0.810±0.036 0.929±0.007 0.876±0.014 0.946±0.005 0.954±0.005 0.838±0.020 LlaSMol 0.937±0.007 0.933±0.009 0.818±0.031 0.950±0.008 0.785±0.013 0.958±0.007 0.956±0.011 0.764±0.012 PRESTO 0.941±0.009 0.866±0.015 0.720±0.034 0.961±0.006 0.785±0.017 0.954±0.008 0.949±0.006 0.765±0.012 ILBERT 0.943±0.008 0.941±0.004 0.679±0.048 0.934±0.008 0.786±0.010 0.958±0.006 0.954±0.009 0.772±0.011

- Table 5. Comparison of AIonopedia with computational baseline and prior baselines on small mass density dataset and hydration ΔG dataset.

mass density hydration ΔG RMSE (g/cm3) Pearson r Kendall τ RMSE (kcal/mol) Pearson r Kendall τ

metric

AIonopedia (Galactica-125m) 0.0120±0.0006 0.9969±0.0005 0.956±0.006 0.966±0.027 0.9427±0.0030 0.909±0.008 AIonopedia (Galactica-1.3b) 0.0110±0.0006 0.9975±0.0003 0.962±0.006 0.802±0.016 0.9611±0.0012 0.940±0.007

0.966±0.006 （0.9662）

AIonopedia (Qwen3-0.6b) 0.0106±0.0015 0.9974±0.0009

0.749±0.018 0.9672±0.0018 0.948±0.002

AIonopedia (Gemma3-1b) 0.0124±0.0008 0.9963±0.0004

0.966±0.005 （0.9659）

0.801±0.015 0.9627±0.0016 0.932±0.003

MD simulation 0.0670±0.0002 0.9800±0.0002 0.886±0.008 1.001±0.005 0.9452±0.0005 0.781±0.004 MLP 0.0385±0.0055 0.9747±0.0085 0.769±0.060 1.321±0.019 0.8940±0.0044 0.737±0.006 MolCA 0.0321±0.0039 0.9798±0.0022 0.931±0.015 1.035±0.191 0.9371±0.0214 0.825±0.048 T5chem 0.0132±0.0014 0.9963±0.0008 0.947±0.005 1.005±0.018 0.9382±0.0022 0.897±0.005 Molinst 0.0141±0.0014 0.9959±0.0005 0.930±0.016 0.969±0.037 0.9523±0.0049 0.901±0.009 SPMM 0.0299±0.0041 0.9855±0.0007 0.857±0.005 0.832±0.044 0.9610±0.0033 0.893±0.005 LlaSMol 0.0118±0.0006 0.9969±0.0005 0.944±0.009 0.727±0.084 0.9698±0.0079 0.935±0.008 PRESTO 0.0156±0.0014 0.9950±0.0012 0.948±0.011 1.035±0.0261 0.9340±0.0033 0.885±0.008 ILBERT 0.0193±0.0002 0.9925±0.0001 0.843±0.006 0.711±0.058 0.9752±0.0055 0.933±0.006

##### 4.4. Ablation study of the multi-modal LLM.

To further assess the contribution of different components in our property predictor to the performance, we conducted an ablation study with the smallest model variant, AIonopedia (Galactica‑125m). The results are summarized in Table 6 where the IL‑based split was adopted and all training hyper‑parameters kept identical. The default model attains the top score on most datasets.

We first examined the contribution of the training losses. The earlier MLP baseline has showed that supervision from pseudo‑labels alone could not deliver adequate performance, so we evaluated the remaining loss terms here. Specifically, we removed the supervised loss and retained only the three BLIP‑2 losses, contrastive loss, GTG loss, and GTM loss. The results reveal that a purely unsupervised graph-text semantic objective fails to capture some molecular information (e.g. the RMSE of solvation ΔG and IL/water transfer ΔG increase 22% and 38%, respectively), whereas adding physicochemical‑property supervision does guide the model toward a deeper understanding of the system.

Next, we evaluated the impact of modality alignment. Completely omitting Phase 1 produced the largest loss in performance, the RMSE on the IL/water transfer ΔG dataset increases from 0.47 kcal/mol to 0.81 kal/mol, almost doubling. We also removed either the graph or the text modality separately. Both omissions impaired accuracy, but the decline was more pronounced when the text modality was excluded.

- Table 6. Performance metrics of the property predictor in the ablation study.

organic solvent/water transfer ΔG (kcal/mol)

IL/water transfer ΔG (kcal/mol)

melting point (K)

viscosity (mPa·s)

surface tension (mN/m)

mass density (g/cm3)

solvation ΔG (kcal/mol)

RMSE

AIonopedia (Galactica-125m)

###### 0.304±0.030 0.473±0.100 38.4±3.0 0.249±0.018 3.63±0.27 0.0272±0.0029 0.415±0.028

w/o supervised loss 0.370±0.047 0.652±0.094 41.2±3.8 0.264±0.017 3.85±0.27 0.0287±0.0025 0.475±0.029 w/o pretrain 0.362±0.036 0.807±0.056 44.4±3.3 0.287±0.011 4.29±0.31 0.0313±0.0023 0.654±0.024 w/o graph modal 0.344±0.0810 0.463±0.128 39.8±3.1 0.257±0.017 3.69±0.27 0.0273±0.0030 0.426±0.031 w/o language modal 0.379±0.010 0.485±0.410 40.9±3.2 0.271±0.107 3.98±0.23 0.0298±0.0036 0.438±0.027

organic solvent/water transfer ΔG

IL/water transfer ΔG

melting point viscosity surface tension mass density

Pearson r solvation ΔG

AIonopedia (Galactica-125m)

###### 0.9625±0.0056 0.9801±0.0104 0.7470±0.0380 0.9393±0.0080 0.9194±0.0160 0.9895±0.0021 0.9849±0.0025

w/o supervised loss 0.9502±0.0144 0.9692±0.0119 0.7101±0.0499 0.9334±0.0089 0.9118±0.0195 0.9887±0.0019 0.9808±0.0026 w/o pretrain 0.9496±0.0104 0.9467±0.0110 0.6406±0.0444 0.9215±0.0050 0.8882±0.0215 0.9867±0.0021 0.9650±0.0020 w/o graph modal 0.9574±0.0156 0.9807±0.0112 0.7256±0.0463 0.9358±0.0088 0.9134±0.0195 0.9892±0.0023 0.9844±0.0025 w/o language modal 0.9441±0.0037 0.9766±0.0131 0.7015±0.0310 0.9262±0.0097 0.9008±0.0208 0.9870±0.0029 0.9824±0.0026

organic solvent/water transfer ΔG

IL/water transfer ΔG

melting point viscosity surface tension mass density

Kendall τ solvation ΔG

AIonopedia (Galactica-125m)

0.931±0.004 (0.9314)

0.906±0.005 (0.9061)

0.870±0.019 0.883±0.016 0.546±0.031 0.834±0.015 0.800±0.021

w/o supervised loss 0.848±0.017 0.871±0.010 0.496±0.046 0.816±0.015 0.786±0.026 0.926±0.003 0.892±0.005 w/o pretrain 0.841±0.010 0.796±0.019 0.449±0.057 0.798±0.014 0.741±0.027 0.918±0.004 0.847±0.005

0.931±0.004 (0.9312)

0.906±0.004 (0.9056)

w/o graph modal 0.856±0.035 0.885±0.020 0.517±0.054 0.827±0.015 0.790±0.024

w/o language modal 0.809±0.013 0.878±0.022 0.500±0.040 0.809±0.017 0.758±0.024 0.923±0.006 0.898±0.004

- 4.5. IL modification and screening for gas absorption: from literature calibration to wet-lab validation. Traditional IL design typically relies on the prior knowledge of domain experts. By defining a family of

structurally related ILs as the prior, researchers can confine exploration to a smaller chemical space for targeted modification and optimization. Accordingly, we introduce a practice-aligned workflow where AIonopedia refines a specified IL via two representative strategies: anion replacement and cation side-chain engineering. In the former, we fix the cation and substitute candidate anions; in the latter, we fix the anion and systematically modify the cation side chain. We selected one targeted example for each to execute this workflow. During the process, AIonopedia iteratively performs reasoning and computation to verify whether its hypotheses are correct. To prevent cheating, the study ensured that no dataset leakage occurred and that online queries were disabled. The iteration budget was also set to five to restrict computational trials and probe the agent’s reasoning ability.

For anion replacement, we started from [EMIM]+[SCN]- and tasked AIonopedia with optimizing CO2 absorption. The agent identified the anion [TCB]- within five iterations. The other anions explored along the way yielded solvation ΔG of CO2 (kcal/mol) calculated at 298 K, 1 atm, consistent with the corresponding absorption-capacity ranking: [SCN]- (−0.5964) < [DCA]- (−0.7336) < [TCM]- (−1.3686) < [TF₂N]- (−1.6346) < [TCB]- (−1.7204)108-111. Accordingly, solvation ΔG provides a thermodynamically motivated proxy for absorption capacity.

For cation side-chain engineering, we used [EMIM]+[TF2N]- as the starting point to optimize NH3 uptake. With targeted prompting and a few iterations, AIonopedia converged on [EtOHIM]+ as the preferred cation. The calculated solvation ΔG of NH3 for the other cationic variants also tracked the expected absorptioncapacity order: [EMIM]+ (−1.8748) < [EtOHMIM]+ (−1.9520) < [EIM]+ (−1.9692) < [EtOHIM]+ (−2.1151).112

While the workflow is effective for IL modification, dependence on prior knowledge constrains the agent and the traditional research paradigm, limiting the discovery of wholly new IL systems and applications. To assess how AIonopedia aids IL discovery in real wet lab settings, we defined an extremely rigorous task: the model must achieve zero-shot generalization to screen ILs for NH3 absorption. Unlike prior work such as ILBERT, which often validates using systems closely resembling ILs previously reported for the same

application, we manually excluded all ILs reported for NH3 absorption and their close analogues from the data. As a result, the model explores previously unexplored regions of IL chemical space.

Candidate screening was guided by solvation ΔG minimization, which we previously showed to track absorption capacity. In contrast to traditional ILs with nitrogen-centered cations, we discovered the first IL with phosphorus-centered cations applicable to NH3 absorption without imposing a priori family constraints, represented by [P4442]+ [DEP]-. Guided by the predictions, we synthesized this IL and evaluated its NH3 absorption performance using a gravimetric method. Prior to testing, the IL was dried in a vacuum oven at 70 ℃ for 24 h. Approximately 100 mg of IL was weighed into a quartz crucible placed in the measurement cell, where an ionizing fan was used to eliminate electrostatic charges before the cell was sealed. At 100 ℃ and ambient pressure, the sample was purged with helium at 50 cm3/min for about 8 h, with repeated weighing until mass equilibrium, to determine the activated sample mass. Measurements were carried out at 25 ℃ with a total gas flow of 50 cm3/min. During the absorption stage, the NH3 partial pressure was stepped from 5% to 95% in 5% increments. During desorption, it was stepped from 95% to 5% in 10% increments. Equilibrium

- at each step was defined by a mass fluctuation within 0.1 mg, with an upper equilibration time of 100 min. The experimental results show that the ammonia absorption capacity increased with NH3 partial pressure and reached equilibrium at 95%, giving an equilibrium uptake of 1.80 mol/mol. Upon decreasing the NH3 partial pressure, the uptake did not decline, indicating a strong interaction between the IL and NH3. Compared with literature data, this IL exhibits a high absorption capacity. Notably, this phosphorus-centered IL features excellent structural tunability, offering ample scope for targeted performance optimization via cation sidechain engineering and anion replacement. The experimental setup and results are shown in Fig. 5 and Table 7.

#### (A) (B)

###### Gas absorption working principle diagram

Gases

[Figure 67]

[Figure 68]

- 0.4

0.8

1.2

- 1.6
- 2

Microbalance

|[Figure 69]<br><br>absorption chamber|
|---|

[Figure 70]

+-NHuptake of[P][DEP](mol/mol)34442

|0.00000 g|
|---|

abs ch

Vacuum chamber

NH3 He

25 ℃

temperature sensor

Weighing hanging wire

Gas pipelines

MFC 2 MFC 1

Constant temperature absorption chamber

precision ﬂow meter

(B)

[Figure 71]

[Figure 72]

[Figure 73]

25 ℃

Sample crucible ×8

0

Constant temperature circulating water tank

Absorption Desorption

0 20 40 60 80 100

Gas concentration (Vi/V)

- Fig. 5. (A) schematic of the multi-station gravimetric gas/vapor sorption instrument. (B) Ammonia absorption–desorption isotherm of [P4442]+ [DEP]- at 298 K.

- Table 7. Reported absorption capacity of aprotic ILs ammonia absorbent. materials T/℃ Pressure/kPa ammonia uptake (mol NH3/mol IL)

[P4442]+[DEP]- 25 101 1.80 (this work)

[C4C1IM]+[DBP]- 40 153 0.28113 [C4C1IM]+[DMP]- 40 113 0.25113 [C2C1IM]+[DEP]- 40 103 0.20113 [C2C1IM]+[DMP]- 40 101 0.22113 [C1C1IM]+[DMP]- 50 221 0.35113 [C2MIM]+[TF2N]- 40 171 0.097114 [C2MIM]+[BF4]- 40 140 0.14114 [C4MIM]+[BF4]- 40 180 0.25114 [C6MIM]+[BF4]- 30 230 0.37114

[C2MIM]+[SCN]- 30 100 0.18115 [C4MIM]+[SCN]- 30 100 0.19115 [C6MIM]+[SCN]- 40 100 0.20115 [C4MIM]+[TF2N]- 40 100 0.28116 [C4MIM]+[DCA]- 30 567 2.01117

[C4MMIM]+[TF2N]- 30 536 1.60117

[BMMIM]+[DCA]- 30 560 1.61117 [C4IM]+[SCN]- 30 151 2.60117 [C4IM]+[NO3]- 30 100 1.50117

[C2MIM]+[FAP]- 25 101 0.49118 [C2MIM]+[TFO]- 25 101 0.48118

##### 5. Discussion

In this paper, we develop AIonopedia, a transformative LLM agent designed to address a critical need in the field of ILs. AIonopedia delivers a fully automated IL research workflow that spans from raw data acquisition through molecular screening and design. This agent transforms what used to be a fragmented, manual process into a seamless end-to-end pipeline, materially accelerating discovery for domain experts.

At the core of AIonopedia is a multimodal contrastive learning paradigm that unlocks the value of largescale unlabeled corpora while unifying three complementary molecular modalities for training: molecular graphs, SMILES sequences, and physicochemical descriptors. This design not only lifts overall performance beyond competing SOTA approaches, including prior IL-specific methods, chemical-domain unimodal LLMs and multimodal LLMs, but also dramatically strengthens OOD generalization. Consequently, our model can scale to broad species screening with confidence, whereas alternative methods struggle.

In parallel with this modeling effort, we construct a novel, large-scale labeled dataset for ILs. The resource contains more than double the number of pure IL species found in ILthermo, the largest preceding database, and includes the largest known collection of solute-solvent interaction data. This richer supervision expands coverage across chemical space and provides a far more stringent test bed for evaluating ILs as next-generation solvents.

Furthermore, we employed two complementary approaches to the design and discovery of ionic liquids. IL modification used iterative computation and reasoning to realize anion replacement and cation side-chain edits, with results validated on literature-reported datasets. In parallel, IL screening adopted a hierarchical search architecture to balance chemical-space exploration with verification-oriented reliability. We assess the screening pipeline in a challenging, application-driven wet-lab setting by posing NH3 absorption as a zeroshot task and enforcing a literature-agnostic protocol that excludes all previously reported ILs from the search space, thereby restricting exploration to completely new chemistries. Even under this extreme OOD regime, the method pinpoints the first IL with phosphorus-centered cations for NH3 absorption, underscoring its strong promise for ionic-liquid discovery.

In conclusion, this work introduces AIonopedia not merely as a tool, but as a robust, validated framework that fundamentally accelerates the discovery pipeline for ILs. By successfully bridging the gap from automated data handling to SOTA multimodal prediction and real-world experimental validation, we have demonstrated a powerful new paradigm for AI-driven materials science. The contributions presented here serve as the foundation for our ultimate goal of engineering a fully autonomous agent capable of proposing hypotheses, analyzing data, and even directing automated experiments. This endeavor will fulfill the vision of AIonopedia as a true ‘AI-ion-wikipedia’, which is a comprehensive, dynamic, and interactive knowledge resource for the entire research community.

##### References

- 1 Dupont, J. et al. Ionic Liquids in Metal, Photo-, Electro-, and (Bio) Catalysis. Chemical Reviews 124, 5227-5420

(2024). https://doi.org:10.1021/acs.chemrev.3c00379

- 2 Numpilai, T., Pham, L. K. H. & Witoon, T. Advances in Ionic Liquid Technologies for CO2 Capture and Conversion: A Comprehensive Review. Industrial & Engineering Chemistry Research 63, 19865-19915 (2024). https://doi.org:10.1021/acs.iecr.4c02072

- 3 Ovejero-Pérez, A., Nakasu, P. Y. S., Hopson, C., Costa, J. M. & Hallett, J. P. Challenges and opportunities on the utilisation of ionic liquid for biomass pretreatment and valorisation. npj Materials Sustainability 2, 7 (2024). https://doi.org:10.1038/s44296-024-00015-x

- 4 Shivani, Thakur, R. C., Thakur, A., Sharma, A. & Sharma, R. Unravelling the prospects of electrolytes containing ionic liquids and deep eutectic solvents for next generation lithium batteries. Journal of Energy Chemistry 105, 482-500 (2025). https://doi.org:https://doi.org/10.1016/j.jechem.2025.01.060

- 5 Meshram, P., Agarwal, N. & Abhilash. A review on assessment of ionic liquids in extraction of lithium, nickel, and cobalt vis-à-vis conventional methods. RSC Advances 15, 8321-8334 (2025). https://doi.org:10.1039/D4RA08429B

- 6 Shamshina, J. L. & Rogers, R. D. Ionic Liquids: New Forms of Active Pharmaceutical Ingredients with Unique, Tunable Properties. Chemical Reviews 123, 11894-11953 (2023). https://doi.org:10.1021/acs.chemrev.3c00384

- 7 Alreshidi, M. A. et al. A review on the evolution of ionic liquids: Sustainable synthesis, applications, and future prospects. Materials Today Sustainability 31, 101160 (2025). https://doi.org:https://doi.org/10.1016/j.mtsust.2025.101160

- 8 Beil, S. et al. Toward the Proactive Design of Sustainable Chemicals: Ionic Liquids as a Prime Example. Chemical Reviews 121, 13132-13173 (2021). https://doi.org:10.1021/acs.chemrev.0c01265

- 9 Ong, H. C. et al. Recent advances in biodiesel production from agricultural products and microalgae using ionic liquids: Opportunities and challenges. Energy Conversion and Management 228, 113647 (2021). https://doi.org:https://doi.org/10.1016/j.enconman.2020.113647

- 10 Silva, W., Zanatta, M., Ferreira, A. S., Corvo, M. C. & Cabrita, E. J. Revisiting Ionic Liquid Structure-Property Relationship: A Critical Analysis. International Journal of Molecular Sciences 21, 7745 (2020).
- 11 Greer, A. J., Jacquemin, J. & Hardacre, C. Industrial Applications of Ionic Liquids. Molecules 25, 5207 (2020).
- 12 Philippi, F. & Welton, T. Targeted modifications in ionic liquids – from understanding to design. Physical Chemistry Chemical Physics 23, 6993-7021 (2021). https://doi.org:10.1039/D1CP00216C

- 13 Bejaoui, Y. K. J. et al. Insights into structure–property relationships in ionic liquids using cyclic perfluoroalkylsulfonylimides. Chemical Science 14, 2200-2214 (2023). https://doi.org:10.1039/D2SC06758G

- 14 Tsuzuki, S., Matsumoto, H., Shinoda, W. & Mikami, M. Effects of conformational flexibility of alkyl chains of cations on diffusion of ions in ionic liquids. Physical Chemistry Chemical Physics 13, 5987-5993 (2011). https://doi.org:10.1039/C0CP02087G

- 15 Tokuda, H., Hayamizu, K., Ishii, K., Susan, M. A. B. H. & Watanabe, M. Physicochemical Properties and Structures of Room Temperature Ionic Liquids. 2. Variation of Alkyl Chain Length in Imidazolium Cation. The Journal of Physical Chemistry B 109, 6103-6110 (2005). https://doi.org:10.1021/jp044626d

- 16 Tsuzuki, S. Factors Controlling the Diffusion of Ions in Ionic Liquids. ChemPhysChem 13, 1664-1670 (2012). https://doi.org:https://doi.org/10.1002/cphc.201100870

- 17 López-Martin, I., Burello, E., Davey, P. N., Seddon, K. R. & Rothenberg, G. Anion and Cation Effects on Imidazolium Salt Melting Points: A Descriptor Modelling Study. ChemPhysChem 8, 690-695 (2007). https://doi.org:https://doi.org/10.1002/cphc.200600637

- 18 MacFarlane, D. R., Kar, M. & Pringle, J. M. Fundamentals of Ionic Liquids: From Chemistry to Applications. (Wiley-VCH Verlag GmbH & Co. KGaA, 2017).
- 19 Abraham, M. H. Scales of solute hydrogen-bonding: their construction and application to physicochemical and biochemical processes. Chemical Society Reviews 22, 73-83 (1993). https://doi.org:10.1039/CS9932200073

- 20 Bradley, J.-C., Abraham, M. H., Acree, W. E. & Lang, A. S. I. D. Predicting Abraham model solvent coefficients. Chemistry Central Journal 9, 12 (2015). https://doi.org:10.1186/s13065-015-0085-4

- 21 Iftimie, R., Minary, P. & Tuckerman, M. E. Ab initio molecular dynamics: Concepts, recent developments, and future trends. Proceedings of the National Academy of Sciences 102, 6654-6659 (2005). https://doi.org:10.1073/pnas.0500193102

- 22 Cohen, A. J., Mori-Sánchez, P. & Yang, W. Challenges for Density Functional Theory. Chemical Reviews 112, 289320 (2012). https://doi.org:10.1021/cr200107z

- 23 Durrant, J. D. & McCammon, J. A. Molecular dynamics simulations and drug discovery. BMC Biol 9, 71 (2011). https://doi.org:10.1186/1741-7007-9-71

- 24 Datta, R., Ramprasad, R. & Venkatram, S. Conductivity prediction model for ionic liquids using machine learning. The Journal of Chemical Physics 156, 214505 (2022). https://doi.org:10.1063/5.0089568

- 25 Zafer, A., Phu, N., Xiaoqi, C. & Kah Chun, L. Room temperature ionic liquids viscosity prediction from deeplearning models. Energy Materials 3, 300039 (2023). https://doi.org:10.20517/energymater.2023.38

- 26 Sheikhshoaei, A. H., Sanati, A. & Khoshsima, A. Deep learning models to predict CO2 solubility in imidazoliumbased ionic liquids. Scientific Reports 15, 26445 (2025). https://doi.org:10.1038/s41598-025-12004-8

- 27 Qu, M. et al. Machine learning-driven generation and screening of potential ionic liquids for cellulose dissolution. Journal of Cheminformatics 17, 78 (2025). https://doi.org:10.1186/s13321-025-01018-z

- 28 Ali, M. et al. Prediction of CO2 solubility in Ionic liquids for CO2 capture using deep learning models. Scientific Reports 14, 14730 (2024). https://doi.org:10.1038/s41598-024-65499-y

- 29 Zhang, K., Wu, J., Yoo, H. & Lee, Y. Machine Learning-based approach for Tailor-Made design of ionic Liquids: Application to CO2 capture. Separation and Purification Technology 275, 119117 (2021). https://doi.org:https://doi.org/10.1016/j.seppur.2021.119117

- 30 Zhang, R. et al. Modelling enzyme inhibition toxicity of ionic liquid from molecular structure via convolutional neural network model. SAR QSAR Environ Res 34, 789-803 (2023). https://doi.org:10.1080/1062936x.2023.2255517

- 31 Gugulothu, N., Mohan, M., Marland, M. R., Smith, J. C. & Kidder, M. K. Deep Learning Approaches for Predicting the Surface Tension of Ionic Liquids. Journal of Chemical Information and Modeling 65, 5856-5867

(2025). https://doi.org:10.1021/acs.jcim.5c00158

- 32 Baran, K. & Kloskowski, A. Graph Neural Networks and Structural Information on Ionic Liquids: A Cheminformatics Study on Molecular Physicochemical Property Prediction. The Journal of Physical Chemistry B 127, 10542-10555 (2023). https://doi.org:10.1021/acs.jpcb.3c05521

- 33 Feng, H., Qin, L., Zhang, B. & Zhou, J. Prediction and Interpretability of Melting Points of Ionic Liquids Using Graph Neural Networks. ACS Omega 9, 16016-16025 (2024). https://doi.org:10.1021/acsomega.3c09543

- 34 Rittig, J. G., Ben Hicham, K., Schweidtmann, A. M., Dahmen, M. & Mitsos, A. Graph neural networks for temperature-dependent activity coefficient prediction of solutes in ionic liquids. Computers & Chemical Engineering 171, 108153 (2023). https://doi.org:https://doi.org/10.1016/j.compchemeng.2023.108153

- 35 Koutsoukos, S., Philippi, F., Malaret, F. & Welton, T. A review on machine learning algorithms for the ionic liquid chemical space. Chemical Science 12, 6820-6843 (2021). https://doi.org:10.1039/D1SC01000J

- 36 Chen, G., Song, Z., Qi, Z. & Sundmacher, K. Generalizing property prediction of ionic liquids from limited labeled data: a one-stop framework empowered by transfer learning. Digital Discovery 2, 591-601 (2023). https://doi.org:10.1039/D3DD00040K

- 37 Beckner, W., Ashraf, C., Lee, J., Beck, D. A. C. & Pfaendtner, J. Continuous Molecular Representations of Ionic Liquids. The Journal of Physical Chemistry B 124, 8347-8357 (2020). https://doi.org:10.1021/acs.jpcb.0c05938

- 38 David, L., Thakkar, A., Mercado, R. & Engkvist, O. Molecular representations in AI-driven drug discovery: a review and practical guide. Journal of Cheminformatics 12, 56 (2020). https://doi.org:10.1186/s13321-020-00460-5

- 39 Schilling-Wilhelmi, M. et al. From text to insight: large language models for chemical data extraction. Chemical Society Reviews 54, 1125-1150 (2025). https://doi.org:10.1039/D4CS00913D

- 40 Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 (2018). <https://ui.adsabs.harvard.edu/abs/2018arXiv181004805D>.

- 41 OpenAI. Improving language understanding by generative pre-training, <https://openai.com/research/language-

- unsupervised/> (2018).
- 42 Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv:2010.09885 (2020). <https://ui.adsabs.harvard.edu/abs/2020arXiv201009885C>.

- 43 Zhang, D. et al. ChemLLM: A Chemical Large Language Model. arXiv:2402.06852 (2024). <https://ui.adsabs.harvard.edu/abs/2024arXiv240206852Z>.

- 44 Edwards, C. et al. Translation between Molecules and Natural Language. arXiv:2204.11817 (2022). <https://ui.adsabs.harvard.edu/abs/2022arXiv220411817E>.

- 45 Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020

(2021). <https://ui.adsabs.harvard.edu/abs/2021arXiv210300020R>.

- 46 Zhou, Z., Li, Y., Hong, P. & Xu, H. Multimodal fusion with relational learning for molecular property prediction. Communications Chemistry 8, 200 (2025). https://doi.org:10.1038/s42004-025-01586-z

- 47 Zhang, R., Yuan, R. & Tian, B. PointGAT: A Quantum Chemical Property Prediction Model Integrating Graph Attention and 3D Geometry. Journal of Chemical Theory and Computation 20, 4115-4128 (2024). https://doi.org:10.1021/acs.jctc.3c01420

- 48 Schick, T. et al. Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv230204761S>.

- 49 Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. Nature 624, 570-578 (2023). https://doi.org:10.1038/s41586-023-06792-0

- 50 M. Bran, A. et al. Augmenting large language models with chemistry tools. Nature Machine Intelligence 6, 525-535

(2024). https://doi.org:10.1038/s42256-024-00832-8

- 51 Chen, K. et al. Chemist-X: Large Language Model-empowered Agent for Reaction Condition Recommendation in Chemical Synthesis. arXiv:2311.10776 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv231110776C>.

- 52 OpenAI. Introducing GPT-5, <https://openai.com/index/introducing-gpt-5/> (2025).

- 53 Aider‑AI. Aider Polyglot Coding Leaderboard, <https://aider.chat/docs/leaderboards/> (2025).

- 54 Balunović, M., Dekoninck, J., Petrov, I., Jovanović, N. & Vechev, M. MathArena: Evaluating LLMs on Uncontaminated Math Competitions. arXiv:2505.23281 (2025). <https://ui.adsabs.harvard.edu/abs/2025arXiv250523281B>.

- 55 Yue, X. et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. arXiv:2311.16502 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv231116502Y>.

- 56 Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629 (2022). <https://ui.adsabs.harvard.edu/abs/2022arXiv221003629Y>.

- 57 Kojima, T., Gu, S. S., Reid, M., Matsuo, Y. & Iwasawa, Y. Large Language Models are Zero-Shot Reasoners. arXiv:2205.11916 (2022). <https://ui.adsabs.harvard.edu/abs/2022arXiv220511916K>.

- 58 Dev, S. Serper API, <https://serper.dev/> (2025).

- 59 Kinney, R. M. et al. The Semantic Scholar Open Data Platform. ArXiv abs/2301.10140 (2023).
- 60 Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 (2020). <https://ui.adsabs.harvard.edu/abs/2020arXiv200511401L>.

- 61 Kim, S. et al. in Nucleic Acids Research Vol. 51 D1373-D1380 (2022).
- 62 rdkit/rdkit: 2025_03_4 (Q1 2025) Release v. Release_2025_03_4 (Zenodo, 2025).
- 63 Bender, A. & Glen, R. C. Molecular similarity: a key technique in molecular informatics. Organic & Biomolecular Chemistry 2, 3204-3218 (2004). https://doi.org:10.1039/B409813G

- 64 Li, J. et al. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems 34, 9694-9705 (2021).
- 65 Yu, J. et al. CoCa: Contrastive Captioners are Image-Text Foundation Models. arXiv:2205.01917 (2022).

- <https://ui.adsabs.harvard.edu/abs/2022arXiv220501917Y>.

66 Li, J., Li, D., Savarese, S. & Hoi, S. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv:2301.12597 (2023).

- <https://ui.adsabs.harvard.edu/abs/2023arXiv230112597L>.

- 67 Liu, Z. et al. MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter. arXiv:2310.12798 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv231012798L>.

- 68 Chang, J. & Ye, J. C. Bidirectional generation of structure and properties through a single molecular foundation model. Nature Communications 15, 2323 (2024). https://doi.org:10.1038/s41467-024-46440-3

- 69 Shi, Y. et al. Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. arXiv:2009.03509 (2020). <https://ui.adsabs.harvard.edu/abs/2020arXiv200903509S>.

- 70 Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 (2021). <https://ui.adsabs.harvard.edu/abs/2021arXiv210609685H>.

- 71 Liu, H., Li, C., Li, Y. & Lee, Y. J. Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv231003744L>.

- 72 He, K., Fan, H., Wu, Y., Xie, S. & Girshick, R. Momentum Contrast for Unsupervised Visual Representation Learning. arXiv:1911.05722 (2019). <https://ui.adsabs.harvard.edu/abs/2019arXiv191105722H>.

- 73 van den Oord, A., Li, Y. & Vinyals, O. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748 (2018). <https://ui.adsabs.harvard.edu/abs/2018arXiv180703748V>.

- 74 Qiu, Y. et al. Large chemical language models for property prediction and high-throughput screening of ionic liquids. Digital Discovery 4, 1505-1517 (2025). https://doi.org:10.1039/D5DD00035A

- 75 Irwin, J. J. et al. ZINC20—A Free Ultralarge-Scale Chemical Database for Ligand Discovery. Journal of Chemical Information and Modeling 60, 6065-6073 (2020). https://doi.org:10.1021/acs.jcim.0c00675

- 76 Dong, Q. et al. ILThermo: A Free-Access Web Database for Thermodynamic Properties of Ionic Liquids. Journal of Chemical & Engineering Data 52, 1151-1159 (2007). https://doi.org:10.1021/je700171f

- 77 Ertl, P., Rohde, B. & Selzer, P. Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions and Its Application to the Prediction of Drug Transport Properties. Journal of Medicinal Chemistry 43, 3714-3717 (2000). https://doi.org:10.1021/jm000942e

- 78 Wildman, S. A. & Crippen, G. M. Prediction of Physicochemical Parameters by Atomic Contributions. Journal of Chemical Information and Computer Sciences 39, 868-873 (1999). https://doi.org:10.1021/ci990307l

- 79 Hall, L. H. & Kier, L. B. in Reviews in Computational Chemistry Reviews in Computational Chemistry 367-422

(1991).

- 80 Balaban, A. T. Highly discriminating distance-based topological index. Chemical Physics Letters 89, 399-404

(1982). https://doi.org:https://doi.org/10.1016/0009-2614(82)80009-2

- 81 Bertz, S. H. The first general index of molecular complexity. Journal of the American Chemical Society 103, 35993601 (1981). https://doi.org:10.1021/ja00402a071

- 82 Dalke, A. The chemfp project. Journal of Cheminformatics 11, 76 (2019). https://doi.org:10.1186/s13321-0190398-8

- 83 Niedermeyer, H., Hallett, J. P., Villar-Garcia, I. J., Hunt, P. A. & Welton, T. Mixtures of ionic liquids. Chemical Society Reviews 41, 7780-7802 (2012). https://doi.org:10.1039/C2CS35177C

- 84 Comanici, G. et al. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv:2507.06261 (2025). <https://ui.adsabs.harvard.edu/abs/2025arXiv250706261C>.

- 85 Claude Sonnet 4.5 (Anthropic, 2025).
- 86 DeepSeek-AI. DeepSeek-R1-0528 Release, <https://api-docs.deepseek.com/news/news250528> (2025).

- 87 DeepSeek-AI et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948 (2025). <https://ui.adsabs.harvard.edu/abs/2025arXiv250112948D>.

- 88 Taylor, R. et al. Galactica: A Large Language Model for Science. arXiv:2211.09085 (2022). <https://ui.adsabs.harvard.edu/abs/2022arXiv221109085T>.

- 89 Yang, A. et al. Qwen3 Technical Report. arXiv:2505.09388 (2025). <https://ui.adsabs.harvard.edu/abs/2025arXiv250509388Y>.

- 90 Team, G. et al. Gemma 3 Technical Report. arXiv:2503.19786 (2025). <https://ui.adsabs.harvard.edu/abs/2025arXiv250319786G>.

- 91 Racki, A. & Paduszyński, K. Recent Advances in the Modeling of Ionic Liquids Using Artificial Neural Networks. Journal of Chemical Information and Modeling 65, 3161-3175 (2025). https://doi.org:10.1021/acs.jcim.4c02364

- 92 Cao, H. et al. PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes. arXiv:2406.13193

(2024). <https://ui.adsabs.harvard.edu/abs/2024arXiv240613193C>.

- 93 Yu, B., Baker, F. N., Chen, Z., Ning, X. & Sun, H. LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset. arXiv:2402.09391 (2024). <https://ui.adsabs.harvard.edu/abs/2024arXiv240209391Y>.

- 94 Liu, Y. et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692 (2019). <https://ui.adsabs.harvard.edu/abs/2019arXiv190711692L>.

- 95 Christofidellis, D. et al. Unifying Molecular and Textual Representations via Multi-task Language Modelling. arXiv:2301.12586 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv230112586C>.

- 96 Raffel, C. et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv:1910.10683 (2019). <https://ui.adsabs.harvard.edu/abs/2019arXiv191010683R>.

- 97 Fang, Y. et al. Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. arXiv:2306.08018 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv230608018F>.

- 98 Dubey, A. et al. The Llama 3 Herd of Models. arXiv:2407.21783 (2024). <https://ui.adsabs.harvard.edu/abs/2024arXiv240721783D>.

- 99 Jiang, A. Q. et al. Mistral 7B. arXiv:2310.06825 (2023). <https://ui.adsabs.harvard.edu/abs/2023arXiv231006825J>.

- 100 LMSYS. FastChat: We released Vicuna v1.5 based on Llama 2 with 4K and 16K context lengths. (2023). <https://github.com/lm-sys/FastChat>.

- 101 Ishii, Y. & Matubayasi, N. Self-Consistent Scheme Combining MD and Order-N DFT Methods: An Improved Set of Nonpolarizable Force Fields for Ionic Liquids. Journal of Chemical Theory and Computation 16, 651-665

(2020). https://doi.org:10.1021/acs.jctc.9b00793

- 102 Ishii, Y., Matubayasi, N. & Washizu, H. Nonpolarizable Force Fields through the Self-Consistent Modeling Scheme with MD and DFT Methods: From Ionic Liquids to Self-Assembled Ionic Liquid Crystals. The Journal of Physical Chemistry B 126, 4611-4622 (2022). https://doi.org:10.1021/acs.jpcb.2c02782

- 103 Mangin, T., Schurhammer, R. & Wipff, G. Liquid–Liquid Extraction of the Eu (III) Cation by BTP Ligands into Ionic Liquids: Interfacial Features and Extraction Mechanisms Investigated by MD Simulations. J. Phys. Chem. B

(2022).

- 104 Wang, X. et al. Multi-temperature charge scaling of ionic solvents: Disparate responses of thermodynamic properties. J. Mol. Liq., 125445 (2024). https://doi.org:https://doi.org/10.1016/j.molliq.2024.125445

- 105 Abraham, M. J. et al. GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers. SoftwareX 1-2, 19-25 (2015). https://doi.org:https://doi.org/10.1016/j.softx.2015.06.001

- 106 Pople, J. A. Quantum Chemical Models (Nobel Lecture). Angewandte Chemie International Edition 38, 1894-1902

(1999). https://doi.org:https://doi.org/10.1002/(SICI)1521-3773(19990712)38:13/14<1894::AIDANIE1894>3.0.CO;2-H

- 107 Martin, J. M. L. & de Oliveira, G. Towards standard methods for benchmark quality ab initio thermochemistryW1 and W2 theory. Journal of Chemical Physics 111, 1843-1856 (1999). https://doi.org:10.1063/1.479454

- 108 Torralba-Calleja, E., Skinner, J. & Gutiérrez-Tauste, D. CO2 Capture in Ionic Liquids: A Review of Solubilities and Experimental Methods. Journal of Chemistry 2013, 473584 (2013). https://doi.org:https://doi.org/10.1155/2013/473584

- 109 Kim, J. E., Kim, H. J. & Lim, J. S. Solubility of CO2 in ionic liquids containing cyanide anions: [c2mim][SCN], [c2mim][N(CN)2], [c2mim][C(CN)3]. Fluid Phase Equilibria 367, 151-158 (2014). https://doi.org:https://doi.org/10.1016/j.fluid.2014.01.042

- 110 Carvalho, P. J., Kurnia, K. A. & Coutinho, J. A. P. Dispelling some myths about the CO2 solubility in ionic liquids. Physical Chemistry Chemical Physics 18, 14757-14771 (2016). https://doi.org:10.1039/C6CP01896C

- 111 Safarov, J. et al. Carbon dioxide solubility in 1-butyl-3-methylimidazolium-bis(trifluormethylsulfonyl)imide over a

- wide range of temperatures and pressures. The Journal of Chemical Thermodynamics 67, 181-189 (2013). https://doi.org:https://doi.org/10.1016/j.jct.2013.08.008
- 112 Jiang, H.-Y. et al. Advanced Materials for NH3 Capture: Interaction Sites and Transport Pathways. Nano-Micro Letters 16, 228 (2024). https://doi.org:10.1007/s40820-024-01425-1

- 113 Zhu, X. et al. Efficient absorption of ammonia with dialkylphosphate-based ionic liquids. New Journal of Chemistry 45, 20432-20440 (2021). https://doi.org:10.1039/D1NJ02018H

- 114 Yokozeki, A. & Shiflett, M. B. Ammonia Solubilities in Room-Temperature Ionic Liquids. Industrial & Engineering Chemistry Research 46, 1605-1610 (2007). https://doi.org:10.1021/ie061260d

- 115 Zeng, S. et al. Efficient and reversible absorption of ammonia by cobalt ionic liquids through Lewis acid–base and cooperative hydrogen bond interactions. Green Chemistry 20, 2075-2083 (2018). https://doi.org:10.1039/C8GC00215K

- 116 Shang, D. et al. Protic ionic liquid [Bim][NTf2] with strong hydrogen bond donating ability for highly efficient ammonia absorption. Green Chemistry 19, 937-945 (2017). https://doi.org:10.1039/C6GC03026B

- 117 Shang, D. et al. Enhanced NH3 capture by imidazolium-based protic ionic liquids with different anions and cation substituents. Journal of Chemical Technology & Biotechnology 93, 1228-1236 (2018). https://doi.org:https://doi.org/10.1002/jctb.5467

- 118 Makino, T. & Kanakubo, M. NH3 absorption in Brønsted acidic imidazolium- and ammonium-based ionic liquids. New Journal of Chemistry 44, 20665-20675 (2020). https://doi.org:10.1039/D0NJ04743K

