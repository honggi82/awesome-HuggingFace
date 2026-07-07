### arXiv:2509.21320v3[cs.CL]14Dec2025

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 22]

## SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines

###### Yizhou Wang1,2 ♦♠, Chen Tang1,2 ♦♠, Han Deng1,2 ♠, Jiabei Xiao1,2 †, Jiaqi Liu1 †, Jianyu Wu1,7 †, Jun Yao1,4 †, Pengze Li1,6 †, Encheng Su1,4 †, Lintao Wang1,3†, Guohang Zhuang1, Yuchen Ren1,3, Ben Fei1,2, Ming Hu1, Xin Chen1, Dongzhan Zhou1, Junjun He1, Xiangyu Yue2, Zhenfei Yin8, Jiamin Wu1,2, Qihao Zheng1, Yuhao Zhou1, Huihui Xu1, Chenglong Ma1, Yan Lu1,2, Wenlong Zhang1, Chunfeng Song1, Philip Torr8, Shixiang Tang1,2 ♣, Xinzhu Ma1♣, Wanli Ouyang1,2 and Lei Bai1

1Shanghai Artificial Intelligence Laboratory, 2The Chinese University of Hong Kong, 3The University of Sydney, 4University of Science and Technology of China, 5Fudan University, 6Shanghai Jiao Tong University, 7University of North Carolina at Chapel Hill, 8University of Oxford

We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence–text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at https:

//huggingface.co/SciReason and https://github.com/open-sciencelab/SciReason.

###### Highlights

- • Adaptive Scientific Reasoning. We obverse not all scientific tasks necessitate explicit reasoning—for instance, property prediction tasks often admit direct answers, whereas retrosynthesis and design tasks require elaborate reasoning. Our model can exploit the robustness of answerbased matching to generate reliable CoT supervision only where reasoning is essential, which explicitly differentiates between two task modes: “instant” tasks, where the original supervised fine-tuning (SFT) data is preserved, and “thinking” tasks, where the data is fully replaced with CoT-augmented counterparts. This principled separation enables the model to adaptively allocate reasoning capacity, producing coherent CoT rationales for complex scientific tasks while preserving efficiency and accuracy on simpler, non-reasoning tasks.
- • Task Grouped Rewarding. Typical reward functions for reinforcement learning on typical LLMs lack explicit signals for encouraging and correcting scientific reasoning patterns across diverse thinking tasks, e.g., knowledge extraction, property prediction and classification, sequence generation and design. To tackle these diverse scientific tasks, we propose to group various scientific rewards into three categories: distance-based reward functions for scientific predictions, matching-based reward functions for scientific retrieval and extraction, and scientific toolingverified reward functions that are based on professional scientific tools. Such innovative grouping enables our model share the common knowledge among different tasks and simplifies the reinforcement learning framework.
- • Scientific Reward Softening. Scientific tasks typically return True(1)/False(0), in which case we experimentally observe making the reinforcement learning difficult to converge. In this paper,

♠: Co-first Author ♦: Student Project Lead †: Core Contributor ♣: Corresponding Author Please send correspondence regarding this report to tangshixiang@pjlab.org.cn

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 44]

###### Aspect SciReasoner NatureLM Bio-T5/T5+ UniGenX

Scientific Reasoning

✓ ✗ ✗ ✗

Scientific Task Coverage

Scientific Translation

10 3 2 0

Text/knowledge Extraction

6 0 0 0

Property Prediction

22 0 1 4

Property Classification

49 10 13 0

Generation and Design

16 17 5 18 Total 103 30 21 22 Pretraining corpus

- 140B sequence-centric tokens:
- – 90% sequence-based data
- – 10% general text

80B biochemical tokens:

84.5M structural samples:

###### 206B diverse tokens:

- – Scientific text
- – Pure scientific sequences
- – Sequence–text interleaved pairs
- – Sequence–sequence interleaved pairs

– Completely composed of proteins and small molecules data

- – 99% protein-structure and materials data
- – 1% small molecule data

SFT corpus 40M QA samples 5M QA samples 2.8M QA samples 24M task-related samples Reasoning corpus

NA NA NA

570K answer-aligned longform scientific CoT data and 72K difficulty-filtered RL data

Table 1 | Comparison emphasizing reasoning ability and coverage breadth. Our model explicitly elicits scientific reasoning and supports five capability families with up to 103 tasks; While other methods either only emphasize cross-domain sequence-based discovery or support limited disciplines, they do not pay attention to scientific reasoning and end-to-end workflow coverage.

we develop a method to continuously scale reward signals. Specifically, we transform task-specific metrics, such as language-based scoring metrics (e.g., BERTScore) or quantitative measures (e.g., RMSE) for regression tasks, into a normalized [0, 1] range. This continuous reward signal effectively promotes stable training and significantly boosts the model’s performance on complex scientific reasoning tasks.

• Comprehensive Scientific Capability. Our model supports a broad scientific workflow, spanning five major families and encompassing up to 100 tasks:

- – Scientific Translation: Bidirectional conversion between human text and scientific representations for interoperability.
- – Text & Knowledge Extraction: Literature-aware modeling that turns unstructured text into structured evidence for read–reason–design cycles.
- – Property Prediction: Accurate estimation of continuous scientific endpoints across chemistry, biology, and materials.
- – Property Classification: Robust categorization of molecular, biological, and material properties into discrete classes.
- – Sequence Generation & Design: Constraint-aware, controllable generation that reuses predictive structure for efficient design loops.

We obtain state-of-the-art performance on 54 tasks and rank among top-2 on 101 tasks, demonstrating that a single backbone integrating natural language with multi-representation scientific data can enable cross-domain generalization beyond fixed task menus and reduce the fragmentation seen in specialist pipelines. We highlight the superiority of our model in Tab. 1

##### Contents

- 1 Introduction 3
- 2 Pre-Training 6

- 2.1 Pretraining Data Types 6
- 2.2 Training Protocol 8

- 3 Post-Training 9

- 3.1 Supervised Fine-tuning 9

- 3.1.1 Data 9
- 3.1.2 Training Protocol 10

- 3.2 Reasoning-based Reinforcement Learning 10

- 3.2.1 Data 10
- 3.2.2 Training Protocol 11

- 4 Evaluation and Application 13

- 4.1 Task Overview 13
- 4.2 Baseline 13
- 4.3 Scientific Translation Tasks 13
- 4.4 Scientific Knowledge Extraction and Question Answering 17
- 4.5 Property Prediction Tasks 19
- 4.6 Property Classification Tasks 23
- 4.7 Generation and Design Tasks 30
- 4.8 Reasoning Examples 36

- 4.8.1 Material Task: SNUMAT Classification 36
- 4.8.2 DNA Task: Promoter-300 37
- 4.8.3 Protein Task: Solubility 37

- 4.9 Effectiveness of Pretraining 38

- 5 Conclusion 40 References 40

###### 1. Introduction

Large foundation models (LFMs) have demonstrated tremendous potential in modeling natural language [1, 2, 3], vision [4, 5, 6, 7, 8], audio [9, 10, 11, 12] and multimodal [13, 14, 15, 16] tasks. Recent works such as GPT-5 [17], Qwen-3 [18], LLaMA-4 [19], DeepSeek-v3 [20], and Gemini-

###### 2.5 [21] have achieved remarkable performance across diverse domains, benefiting from advances in transformer architectures [22, 23, 24], large-scale pretraining [25], and instruction tuning [26, 27, 28]. While these models excel at general-purpose reasoning [29, 30, 31], summarization [32, 33, 34], and dialogue [35, 36, 37], their adaptation to scientific disciplines such as chemistry, biology, healthcare, and materials science remains an emerging frontie [38, 39, 40, 41, 42, 43, 44]. Scientific problems often involve diverse, domain-specific representations [45, 46, 47], ranging from molecular representation (e.g., SMILES [48], SELFIES [49]) and protein sequences (e.g., FASTA [50]) to genomic data, which differ fundamentally from natural language in structure and semantics. Bridging this gap requires foundation models capable of integrating symbolic, numerical, and sequence-based knowledge with natural language understanding, enabling applications such as molecular property prediction [51, 52, 53, 54], biomolecule engineering [55, 56, 57, 58, 59], and the interpretation of complex experimental data [60, 61, 62, 63, 64]. Existing works on scientific large models can be generally classified into two parts:

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 66]

Model

v8

[Figure 67]

[Figure 68]

[Figure 69]

Data Types

Pre-training

|True/False Question Open-Ended Question<br><br>...<br><br>Prediction<br><br>Classification<br><br>[Figure 70]<br><br>Molecular Prediction Material Property Prediction Biology Multi-omics Regression<br><br>...<br><br>Biology Multi-omics Property Classification<br><br>•Solubility Classification<br>•Antibody-Antigenqing<br><br><br>...<br><br>Translation<br><br>Generation and Design<br><br>QA<br><br>Translation between : SMILES and Formula IUPAC and Formula<br><br>...<br><br>[Figure 71]<br><br>[Figure 72]<br><br>Unconditional Generation Tasks Conditional Material Generation Conditional Molecular Generation<br><br>Function-guided Protein Design<br><br>MoleculeDesign<br><br>...<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]|
|---|

[Figure 77]

[Figure 78]

SFT

| |[Figure 79]|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]|
|---|---|---|
| |[Figure 83]|[Figure 84]<br><br>Sequence-Sequence Sequence-Text Pair<br><br>[Figure 85]|
| | | |

[Figure 86]

[Figure 87]

Reasoning

[Figure 88]

[Figure 89]

[Figure 90]

Task

[Figure 91]

Data

[Figure 92]

[Figure 93]

[Figure 94]

Scientific Text Pure Sequence General Text

Chain of Thought <think> ...<think>

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Instant Thinking

Application

Pre-training Instruction Finetuing Reasoning RL

[Figure 101]

What are the key structural differences between DNA and RNA?

[Figure 102]

[Figure 103]

What is the primary monomer ... a common macromolecular material ...

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

{"input": "<dna>CAGGTCAG...</dna> I need to identify any epigenetic marks in this DNA."}

[Figure 112]

The primary monomer units used in the synthesis...

[Figure 113]

The main structural differences between DNA and RNA are...

Sequence

General Texts

Pure

Sequence Text pair

[Figure 114]

[Figure 115]

sequence pair

[Figure 116]

[Figure 117]

... Sequence

What is the molecular formula

What determines the threedimensional structure of a protein?

[Figure 118]

and molar mass of aspirin?

[Figure 119]

<think>DNA sequences themselves do not directly contain epigenetic marks...1.DNA Methylation Data... 2.Histone Modification Data: ... 3.RNA Expression Data... 4.Protein Binding Data....5.ChIP-Seq or Bisulfite Sequencing Data...</think>

###### Pretraining corpus

[Figure 120]

[Figure 121]

The molecular formula of aspirin is C₉H₈O₄, and ...

The three-dimensional structure of ...

Multi-discipline corpus

Mixed

|data<br><br>re-weighting|
|---|

[Figure 122]

[Figure 123]

###### warm-up stage

scientific text general text paired sequence text

The DNA sequence displays characteristic patterns of epigenetic modifications, confirming their presence.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

What is the role of C. elegans in the lipoylation pathway?

[Figure 128]

[Figure 129]

[Figure 130]

Reasoning corpus

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Expression of the enzymes M01F1.3, ZC410.7, and ....

###### SFT-style

[Figure 135]

Satisfy complex reasoning or design tasks

[Figure 136]

[Figure 137]

Satisfy basic tasks Satisfy QA or Prediction tasks

[Figure 138]

[Figure 139]

[Figure 140]

- Figure 1 | We present a unified scientific LLM framework organized into three stages (Pre-training, SFT, and Reasoning). In the pre-training stage, a mixed “warm-up” corpus aligns the base model with scientific syntax and sequences, combining scientific and general text with pure sequences and paired sequence-text / sequence-sequence data spanning DNA/RNA, proteins, small molecules, and materials. The SFT stage aggregates more than 100 curated scientific sub-tasks across these domains to adapt the model to actionable research objectives such as property prediction and sequence-text transformations. The reasoning stage introduces a rationale-augmented corpus and supports two usage modes—Instant (fast answers) and Thinking (step-by-step deliberation)—to solve complex problems.

- • Specialist Large Models. Bio-T5 families [65, 66], ChemLLM [67], ChemVLM [68], ChemMLLM [69], Chem3DLLM [70], ProLLama [71], BioReason [72] adopt pretrained large language models (e.g., T5 [73], LLaMA-2 [74], Qwen-3 [18], InternLM [3]) to solve the discipline-specific tasks [75, 76, 77]. They typically collect curated, domain-specific corpora [78, 79, 80] and task-aligned instruction data [81], e.g., sequence–property pairs; protein/DNA/RNA sequences with structural or functional annotations; reaction and synthesis records; and multi-form molecular representations (e.g., SMILES/IUPAC/SELFIES).
- • Multi-discipline Large Models. NatureLM [82] and UniGenX [83] adopt a pretrain-thenfinetune scheme to learn a shared model for both life science and materials science. Galactica [84] is designed to support scientific knowledge mining from massive academic literatures. Biology-Instructions [85] collects a large-scale dataset to comprehend multi-omics biology.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 162]

Although these methods and other homologous techniques [86, 70, 87, 88, 68, 89, 90, 91] demonstrate the large models can successfully handle scientific tasks, key gaps remain in cross-domain generalization, grounding to physical and experimental constraints, and systematic coverage of heterogeneous scientific modalities. For specialist models, their narrow scope and customized pipelines limit transfer outside the target subfield, fragment the model ecosystem, and, more importantly, require retraining to support unseen tasks or modalities. For multi-discipline large models [82, 83], they typically focus on limited tasks (e.g., NatureLM supports only 30 tasks) and therefore fail to cover the long tail of scientific problems or generalize compositionally beyond a pre-specified task list. Furthermore, they still produce shallow pattern-based predictions rather than explicit multi-step reasoning traces that mirror how scientists derive conclusions from data and theory. We therefore aim to close these gaps by unifying multi-representation scientific data with natural language in a single backbone, scaling instruction coverage to a substantially broader and more diverse task mixture, and grounding reasoning via long-form thinking process to ensure fidelity and verifiability.

In this work, we propose the first scientific reasoning large language model that couples multirepresentation pretraining with instruction-driven alignment and reasoning-inducing post-training. Our model ingests natural language alongside scientific sequences (DNA/RNA/protein), molecular strings (e.g., SMILES/IUPAC/SELFIES) and material representation through task-aware tokenization mapped into a shared backbone. Our training process involves pretraining, SFT-based post-training and RL-based post-training. Pretraining first aligns heterogeneous corpus (e.g., sequence-structuretext), while instruction tuning abstracts diverse tasks into a small set of consistent input–output schemas that encourage compositional generalization. Finally, a reasoning-inducing phase equips the model with deliberate and verifiable problem solving: we bootstrap long-form rationales via an annealed cold-start procedure, and finish with reinforcement learning using physics- and task-aware rewards (factuality, stoichiometry consistency, constraint satisfaction), yielding reliable chain-ofthought solutions. Our method supports a broad range of the following tasks. We organize capabilities into five families that align with the scientific workflow and enable cross-domain generalization beyond prior systems (e.g., limited task menus as in NatureLM), as follows:

- • Scientific Translation. Ensures interoperability between human-readable text and machineoriented scientific representations, enabling faithful bidirectional conversion that preserves semantics and physical meaning. Compared with earlier systems confined to a few format mappings, this category scales to many representations and thus supports data augmentation, tool chaining, and transparent human–AI collaboration.
- • Text and Knowledge Extraction. Focuses on natural language understanding and reasoning tasks that operate directly on diverse scientific text. This evaluates the model’s ability to identify scientific entities, extract structured knowledge, and answer domain-specific questions from literature. By isolating linguistic comprehension from sequence modeling, it probes whether the model can capture the semantics of scientific discourse, resolve complex entity relations, and deliver accurate answers across diverse contexts.
- • Scientific Property Prediction. Provides a unified interface for estimating continuous endpoints (e.g., biochemical, materials), producing continuous signals that can be used for screening and prioritization. Unlike prior models tied to a small set of benchmarks, our formulation flexibly accommodates heterogeneous continuous targets across disciplines, enhancing transfer and compositional generalization.
- • Scientific Property Classification. Handles categorical endpoints (binary, multi-class, and hierarchical) such as solubility, stability, functional annotations, or phase labels. It delivers high-fidelity and domain-general capabilities over categorical properties, facilitating practical applications such as candidate triage, property screening, and informed design decisions across scientific domains.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 184]

• Scientific Sequence Generation and Design. Closes the loop from inference to actionable design under multi-objective and constraint-aware settings (e.g., synthesizability, stability, safety). This family goes beyond template matching by supporting controllable, cross-domain generation that reuses learned structure from prediction tasks, yielding better sample efficiency than siloed, single-discipline models.

The overall illustration of this work is shwon in Fig. 1. In summary, we introduce a unified scientific LLM that couples scaled multi-representation pretraining with instruction-aligned I/O schemas and a verifiable reasoning stage. Task-aware tokenization aligns heterogeneous sequences and notations; a solve–check protocol with retrieval and domain tools, followed by reinforcement learning with physics- and task-aware rewards, enforces unit/stoichiometry and constraint consistency. The resulting single backbone supports prediction, design, translation, and literature-grounded extraction within a read–reason–design loop, delivering strong cross-domain generalization across chemistry, proteins, genomics, and materials; the following sections detail data, training, and evaluation.

###### 2. Pre-Training

###### 2.1. Pretraining Data Types

To ensure downstream results on scientific sequence generation, property prediction, and sequencemanipulation tasks, we design our pretraining strategy around four complementary types:

- 1. Scientific Text. We harvest long and short prose from PubMed [92], PubChem [93], textbooks, and materials databases (e.g., crystal-structure annotations), as well as solution threads from StackExchange for a specific discipline. This corpus grounds the model’s understanding of domain-specific reasoning and terminology needed for scientific Q&A and context-aware property inference.
- 2. Pure Sequence. To capture the contextual dependencies of biomolecular and chemical sequences, we collect:

- • DNA/RNA: Serving as the fundamental blueprint of genetic information, pure nucleotide sequences allow the model to directly learn base-level patterns and long-range dependencies without being influenced by external structural or functional annotations.
- • Protein: As the direct products of gene expression, amino acid sequences encode the full spectrum of structural and functional diversity in proteins, enabling the model to capture residue-level motifs and evolutionary conservation signals.
- • Small Molecules: We collect Simplified Molecular Input Line Entry System (SMILES) strings from PubChem. Furthermore, to ensure that the model can learn various representations of molecules beyond SMILES, we also adopt International Union of Pure and Applied Chemistry (IUPAC) and Self-Referencing Embedded Strings (SELFIES) names to achieve enhanced molecule representations. For SMILES sequences, we warp them with the <SMILES>...</SMILES> tags.

- 3. Sequence–Sequence and Sequence–Text Pairs. Beyond standard sequence collections, we enrich the training data with cross-discipline examples drawn from life sciences (e.g. protein structure), materials chemistry, and scientific literature. We hope this hybrid strategy preserves broad language fluency while instilling deep domain-specific proficiency. To simulate the diversity of cross-disciplinary queries, we develop:

• Intra-discipline pairs: e.g. multi-protein homology judgment based on sequence patterns. Such data enhances the model’s ability to discern subtle evolutionary and structural relation-

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 206]

[Figure 207]

- Figure 2 | Composition of the 206B-token scientific pretraining corpus across five scientific domains: protein data, chemical data, genomic data, transcriptomic data and materials data.

ships within a single domain, improving performance in specialized sequence comparison and classification.

- • Cross-discipline pairs: e.g. prediction of small-molecule binding to a protein segment. By bridging cross-disciplinary representations, these pairs could improve the capacity to integrate different knowledge for tasks such as drug discovery and biomolecular engineering.
- • Multi-representation of the same entity: concurrent IUPAC, SELFIES, and SMILES representations of a molecule. This could promote translation and reasoning between complementary notations, enabling robust understanding and cross-format interoperability.
- • Text–data pairs: combined prompts and labels for molecular, nucleic, or protein properties. Pairing natural language descriptions with scientific sequences enables rich bidirectional interaction, empowering downstream natural language understanding (e.g., protein or molecule interpretation) and language-driven generation of scientific sequences.

4. General Text. In order to maintain the language capabilities of the model itself, we adopt C4 dataset as our general text corpus. Compared with the existing methods [82], the pretraining data they use contains only about 10% of natural text, and almost entirely relies on pure sequence data; therefore, they lack coherent conversation capabilities and cross-domain reasoning abilities.

###### Data Preparation Details

DNA. We retrieve about 200 billion bases from NCBI [94], sampling at an average of 1,000 bp fragments per organism to produce 30 billion <dna></dna>-tagged tokens and is represented by the FASTA format. Essential metadata (e.g., organism name and genomic coordinates) is included to anchor the model’s interpretation of each sequence’s context.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 229]

RNA. We collect about 20 billion RNA tokens from RNAcentral [95], reflecting diverse transcriptomic contexts. Then, we warp the collected RNA sequence (FASTA format) with the <rna></rna> tag. The basic information of the RNA sequence (e.g., Family) is also retrieved and is used to guide the model’s understanding of sequence context.

Protein. We collect protein sequences from UniRef50 and UniRef90 (mammalia) and warp the raw sequence with <protein></protein> tag, this yields about 40 billion protein tokens. We also pair sequence fragments with concise natural-language descriptions, and sample pure text abstracts from PubMed for additional context.

Small Molecules. To ensure reliable representation and alignment of small molecules, we curate diverse textual resources, construct multiple molecular encodings, and implement rigorous token validation.

- • Text: abstracts from PubChem and “long-form” chemistry Q&A from StackExchange.
- • Multi-Representations: To ensure comprehensive molecular characterization, we utilize multiple chemical representations. These are obtained either directly from established databases (e.g., SMILES-to-IUPAC from PubChem) or generated via specialized toolkits (e.g., SMILES-to-SELFIES using the SELFIES library [49]).
- • Tagging: We apply a multistage filter: regex based on RDKit [96] / SELFIES [49] toolkits, small LLM validation and manual review to eliminate ambiguous tokens (e.g., “In”, “Is”) and ensure high-fidelity tagging.

Materials. We collect raw material data from publicly available material databases, including Material Projects [97], GNoME [47], SNUMAT [98], etc. The fields of raw data include the crystal structure files (Crystallographic Information File, CIFs), chemical composition, space-group and crystal-system information, material type (e.g., bulk or slab), thermodynamic metrics (formation energy, energy above hull), electronic and mechanical properties (band gap, density, elastic moduli), magnetic properties, per-site coordination details, bond-length descriptions, etc.

To convert the raw data into the pretraining corpus, we generate task-specific templates by prompting an LLM to convert structural annotations into natural-language descriptions, then score and select the best templates for large-scale synthetic data generation.

Fig. 2 summarizes the composition of our training corpus across five scientific domains and their internal subdivisions. Specifically, protein sequences account for the largest share (55 billion tokens, 33.1%), followed by chemistry data (50 billion tokens, 30.1%), DNA genome fragments (30 billion tokens, 18.1%), RNA transcripts (26 billion tokens, 15.7%), and materials-science texts (5 billion tokens, 3.0%).

###### 2.2. Training Protocol

We train the model for one epoch using bfloat16 mixed-precision. The base models are Qwen3 [18] 1.7B and 8B, respectively. Training batches consisted of 16 and 32 sequences per device for 1.7B model and 8B model, respectively. We set the initial learning rate to 2 × 10−4 and accumulated gradients over two steps, applying a linear warm-up over the first 1% of total steps. We especially arrange the data for the following training stages:

• At the warm-up stage, we adopt a mixed form of scientific text, general text and paired sequence

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 251]

[Figure 252]

- Figure 3 | Training loss over train tokens for the 1.7B and 8B models. The model does not exhibit any saturation phenomenon during the training process.

text data. This could help the model rapidly adapt to both domain-specific and general linguistic patterns before the main training phase.

• At the end of the training, we include SFT-style training data to refine the model’s capability to follow complex, domain-specific instructions and produce task-aligned outputs. This stage serves to align the model’s generative behavior with SFT-like reasoning and formatting standards, thereby enhancing its utility in real-world scientific applications.

For the training system, we use 128 A800 GPUs for 1.7B model and 256 A800 GPUs for 8B model. The training loss curves are shown in Fig. 3.

###### 3. Post-Training

###### 3.1. Supervised Fine-tuning

###### 3.1.1. Data

In the supervised fine-tuning (SFT) stage, we integrated about 100 scientific sub-tasks spanning multiple disciplines, including chemistry (Mol-Instructions [81], ChemData [67], USPTO50k [99], SMolInstruct [100], etc.), DNA, RNA, proteins (Biology-Instructions [85], PEER [101], OPI [102], GUE [103], BEACON [104], etc.), materials (LLM4Mat [105], Material Projects [97]) and general instruction texts from OIG [106]. Moreover, cross-sequence interaction tasks and natural language–based scientific question answering are categorized within the cross-domain corpus. To warp the scientific sequences with discipline-related tags (i.e., <SMILES>...</SMILES>, <dna>...</dna>, <rna>...</rna>, <protein>...</protein>, etc.), the data preparation pipeline involved:

- • Extracting scientific sequences from chemistry, DNA, RNA, protein, and macromolecule datasets via rule-based scripts tailored to each dataset format.
- • Conducting secondary annotation using domain-specific checkpoints from pre-trained singlediscipline models.
- • Filtering and adding scientific tags with the assistance of small LLMs.
- • Converting all data into a unified dialogue format for SFT training.

We show the data distribution of supervised finetuning stage in Tab. 2.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 274]

Category Training samples (M) Proportion Materials 8.278 20.79% General Text 10.873 27.30% DNA&RNA 5.476 13.75% Protein 2.473 6.21% Molecule 12.556 31.53% Cross Domain 0.168 0.42% Total 39.824 100.00%

- Table 2 | Distribution of supervised fine-tuning (SFT) training samples across five major scientific domains and additional general text data, reported in millions and proportions. Each domain undergoes tailored preprocessing, domain-specific annotation, and tagging (e.g., <SMILES>...</SMILES>, <dna>...</dna>, <protein>...</protein>) before being converted into a unified dialogue format for SFT. The corpus comprises 63.35 million samples.

###### 3.1.2. Training Protocol

We train the model for five epochs using 16-bit bfloat16 precision. A per-device batch size of 32 is used, with gradient accumulation over two steps to achieve a larger effective batch size. The initial learning rate is set to 5 × 10−5, and we apply a linear warm-up over the first 5% of total steps. Inputs are tokenized lazily with right-side truncation to a maximum sequence length of 8192 tokens, and the Liger-kernel [107] is enabled for efficiency. The training process is accelerated using DeepSpeed ZeRO Stage 2 [108] and FlashAttention implementation [109].

Based on the SFT dataset, we train four single-discipline models in DNA/RNA, protein, materials science, and chemical molecules. However, during multi-disciplinary joint training, we observe substantial performance degradation in certain sub-tasks due to data imbalance after dataset merging. To address this, we adopt two strategies: (1) we conduct large-scale pre-training followed by augmentation with additional SFT data—with materials science receiving further data expansion—and (2) data re-weighting, which adjusts task-level proportions to mitigate the impact of low-quality instructions or underrepresented tasks. This re-weighting approach effectively alleviates performance decline in mixed-dataset training.

###### 3.2. Reasoning-based Reinforcement Learning

###### 3.2.1. Data

Cold Start Before reinforcement learning (RL), we construct a correct-only chain-of-thought (CoT) corpus as a cold-start prior. For each sub-task, we use DeepSeek-R1-Distill-Qwen-32B to generate CoT,

answer pairs and target 𝑇=500 correctly answered items under a hard evaluation budget of 𝐶=50,000 prompts. We first draw 6,000 random prompts, evaluate correctness, and retain the CoT attached to correct answers. If after 𝑁 evaluated items the tally is 𝑆 < 𝑇, we estimate the additional sample size via the empirical accuracy ˆ𝑝=𝑆/𝑁 as 𝑛add = ⌈(𝑇 − 𝑆)/max(ˆ𝑝, 𝜀)⌉ (with small 𝜀>0 for numerical stability), then repeat sampling–generation–adjudication until either 𝑆 ≥ 𝑇 or 𝑁 ≥ 𝐶; if the cap is reached, we keep all accumulated correct examples. Correctness is adjudicated by direct comparison of the generator’s final answer to SFT ground truth for auto-gradable tasks (exact match or task-specific tolerance), and by an external GPT-3.5-turbo judge for semantic equivalence on free-form tasks (e.g., protein descriptions). This adaptive, budgeted, correct-only harvesting mitigates rationale noise and yields a reliable initialization for subsequent RL.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 296]

Reinforcement Learning We curate reinforcement learning (RL) data at the level of each sub-task using an empirical solve-rate filter that targets medium-difficulty instances. Let 𝐷train𝑘 denote the k-th sub-task’s training set. For every example 𝑥 ∈ 𝐷train𝑘 :

- 1. Stochastic rollouts. We generate 𝑁 = 8 independent rollouts from 𝑀0 on 𝑥 using stochastic decoding with do_sample=True. To balance output diversity against format fidelity, we adopt an adaptive temperature 𝑇 tuned once per sub-task prior to collection: we initialize at 𝑇 = 0.9 (the DAPO default setting) and increase 𝑇 if the eight-way samples show insufficient diversity (e.g., low uniqueness of normalized responses or highly repetitive content); conversely, if outputs become overly diffuse and induce frequent format non-compliance or invalid parses, we decrease 𝑇. During the final eight rollouts for each 𝑥, we hold 𝑇 fixed and vary only the random seed, keeping all other decoding parameters constant.
- 2. Empirical solve rate. For rollout 𝑖, compute a correctness indicator 𝑓𝑖(𝑥) ∈ {0, 1} using the sub-task’s canonical metric (e.g., exact match or a thresholded score), and aggregate

ˆ𝑝(𝑥) =

1 8

∑︁8

𝑖=1

𝑓𝑖(𝑥). (1)

- 3. Mid-difficulty filter. Retain examples with partial success:

𝑆 = 𝑥 ∈ 𝐷train𝑘 0.125 < ˆ𝑝(𝑥) < 0.875 . (2)

This removes items that are nearly always solved (ˆ𝑝 ≈ 1) or nearly always failed (ˆ𝑝 ≈ 0), which typically yield weaker gradients for policy improvement.

- 4. Sampling. Uniformly sample without replacement 𝐾 = 1000 examples from 𝑆. If |𝑆| < 1000, include all of 𝑆 and top up to 𝐾 = 1000 by uniformly sampling from 𝐷train \ 𝑆.

The resulting 1,000-example set serves as the RL training pool for that sub-task.

Focusing RL on neither-trivial-nor-impossible instances yields stronger and more stable improvement: items with ˆ𝑝 ∈ (0.125, 0.875) (i) expose the policy to cases where exploration can flip outcomes, (ii) avoid vanishing advantages on saturated positives, and (iii) reduce uninformative gradients from uniformly failed cases.

###### 3.2.2. Training Protocol

Cold Start Before reinforcement learning, we apply an annealed cold-start (ACS) adaptation to a the model after SFT stage to (i) endow it with the ability to produce long, coherent chains of thought (CoT) on tasks that genuinely require stepwise reasoning, while (ii) preserving concise, direct-answer behavior on tasks that do not. We therefore partition the SFT task suite into thinking and instant categories. For thinking tasks, we replace the original SFT samples with their CoT-augmented counterparts (teacher-forced rationales followed by the final answer). For instant tasks, we retain the original direct-answer SFT data unchanged. Moreover, we have observed that task-level replacement outperforms within-task mixing of CoT and non-CoT targets. Mixing presents inconsistent sequence lengths, styles, and termination signals for near-identical instructions, weakening the model’s calibration about when to elaborate. Replacement yields a cleaner conditional distribution, i.e., inputs from thinking tasks consistently map to CoT-style trajectories, thereby improving both the incidence and quality of generated rationales without confusing non-reasoning behavior.

To minimize behavioral drift from SFT while shaping rationale production where appropriate, the model is trained with a reduced learning rate and a short schedule relative to SFT (i.e., small-step updates). This “low-temperature” adjustment biases the model toward emitting structured reasoning only when the input distribution indicates a reasoning task.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 318]

|0 200 400 600 800<br><br>Training<br><br>0.60<br><br>0.65<br><br>0.70<br><br>0.75<br><br>0.80<br><br>AveragedRewardScore<br><br>Scientific Rewa<br><br>Figure 4 | Scientific reward during DAPO training progres<br><br>Reinforcement Learning. We adopt DAPO [110] heterogeneous scientific tasks. For each prompt we compute group-standardized advantages:<br><br>𝑟𝑖,𝑡(𝜃) =<br><br>𝜋𝜃(𝑜𝑖,𝑡 | 𝑞, 𝑜𝑖,<𝑡) 𝜋𝜃old(𝑜𝑖,𝑡 | 𝑞, 𝑜𝑖,<𝑡)<br><br>, 𝐴ˆ?<br><br>and maximize the PPO-style clipped objective:<br><br>JDAPO(𝜃) = 𝔼(𝑞,𝑎)∼D,{𝑜<br><br>𝑖}𝑖𝐺=1∼𝜋𝜃old (·|𝑞)<br><br>1<br><br>𝐺 𝑖=1 |𝑜𝑖|<br><br>∑︁𝐺<br><br>𝑖=1<br><br>∑︁|𝑜𝑖|<br><br>𝑡=1<br><br>min 𝑟𝑖,𝑡(𝜃)𝐴ˆ𝑖,<br><br>s.t. 0 < {𝑜𝑖 | is_equivalent(𝑎,<br><br>with the asymmetric Clip-Higher schedule to preserve that filters degenerate groups with identical reward Unlike the original rule-based binary reward, many information extraction, and numeric regression) do therefore replace the binary signal with a reward softe a common [0, 1] scale:<br><br>𝑅soft𝑖 = 𝑔 𝑚𝑖(𝑞, ? where 𝑔(·) is a monotone calibration ensuring compar a group-specific quality metric. For grouping scheme, w prediction tasks, matching-based rewards for retrieval leveraging professional scientific software.<br><br>We show the training dynamics of scientific reward in|1000 1200 1400 1600<br><br>Steps<br><br>rd during DAPO<br><br>s.<br><br>as the RL algorithm to optimize the policy on draw 𝐺 candidates from the old policy 𝜋𝜃old<br><br>?,𝑡 =<br><br>𝑅𝑖 − mean({𝑅𝑖}𝐺𝑖=1) std({𝑅𝑖}𝐺𝑖=1)<br><br>. (3)<br><br>𝑡, clip 𝑟𝑖,𝑡(𝜃), 1 − 𝜀low, 1 + 𝜀high 𝐴 ˆ𝑖,𝑡<br><br>𝑜𝑖)} < 𝐺,<br><br>(4)<br><br>exploration, together with Dynamic Sampling s and a token-level policy-gradient reduction.<br><br>of our tasks (free-form semantic judgment, not admit a single exact-match criterion. We ning scheme that maps the grouped metrics to<br><br>?𝑖, A(𝑞)) (5) able reward magnitudes across tasks and 𝑚𝑖 is<br><br>e employ distance-based rewards for scientific and extraction tasks, and tool-verified rewards<br><br>Fig 4.<br><br>|
|---|---|
| |12|

AveragedRewardScore

,

a

w

w

W

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 340]

###### 4. Evaluation and Application

###### 4.1. Task Overview

Translation. The tasks in this group involve converting information between different scientific formats or representations, for example translating between IUPAC names and SMILES, generating descriptive text from molecular structures, or performing the inverse transformation. These capabilities are essential to ensure interoperability within the scientific data ecosystem and to enable effective human-machine communication.

Text and Knowledge Extraction/Question Answering. Applying natural language processing to scientific literature, these tasks include chemical entity recognition, interaction extraction, and various formats of question answering (multiple-choice, open-ended, true/false). They integrate AI methods with literature analysis to facilitate rapid extraction of knowledge from scientific texts.

Property Prediction and Classification. These tasks span a wide range of predictive challenges, from molecular properties (e.g., solubility, toxicity) to materials characteristics (e.g., MP classification, properties derived from JARVIS-DFT), DNA/RNA predictions (e.g., epigenetic markers, transcription factor binding in human), and protein functions (e.g., catalytic activity, stability). Their scope—encompassing chemistry, materials, nucleic acids, and proteins—highlights the universal demand for accurate predictive models in scientific discovery.

Generation and Design. This category focuses on the unconditional generation tasks and the conditional generation tasks. It includes unconditional generation of molecules, proteins, RNA, and materials, as well as targeted design challenges such as rational antibody design (RAbD) and programmable RNA switches. Progress in this domain is critical for accelerating the discovery of new compounds and biomolecules.

###### 4.2. Baseline

For general-purpose closed-source models, we compare against Gemini-2.5-pro∗ and GPT-o3†‡; for general-purpose open-source models, we use GPT-oss-120B and DeepSeek-r1§. As specialist baselines, we consider two categories: (i) non-language models, represented by Transformer encoderbased architectures (denoted as non-LLM Experts); and (ii) language models, represented by the best-performing specialist large language model on this task (denoted as Specialist LLMs). Please refer to the Appendix for details.

###### 4.3. Scientific Translation Tasks

Scientific translation evaluates whether a model preserves scientific semantics while mapping across modalities—symbolic encodings and formulae, natural-language text, and biological sequences. The goal is not to invent new facts but to produce an equivalent rendering that remains invertible and respects domain constraints (e.g., stoichiometry, valence, sequence–function coherence). This suite

∗Gemini-2.5-pro version: 2025-06-17 †GPT-o3 version: 2025-04-16. Results are provided in Appendix Table. 10 to Table 15. ‡Since some test tasks involve extremely large-scale test sets, to avoid substantial testing costs, for closed-source models,

we randomly sampled 1,000 samples from tasks with a test set size exceeding 1,000 for testing. Meanwhile, we also present the performance of our models on these 1,000 samples for comparison.

§DeepSeek-r1 version: 2025-05-28.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 362]

Gemini 2.5-pro

GPT oss

Non-LLM Experts

Specialist LLMs

Task Metric

SciReasoner-1.7B SciReasoner-8B Molecular Representation Translation

Top1

Split Match ↑ 4.70%† 0.17% — 29.00% 49.52%|50.20%† 56.63%|57.10%† SMILES to Formula

SMILES to IUPAC

Top1

Element Match ↑ 55.20%† 19.88% — 93.20% 95.92%|87.1%† 96.39%|96.00%† IUPAC to SMILES

Top1

Split Match ↑ 38.20%† 7.58% — 70.10% 78.85%|86.50%† 84.40%|90.70%† IUPAC to Formula

Top1

Element Match ↑ 88.50%† 33.24% — 87.90% 91.81%|88.10%† 92.65%|93.00%†

Natural Language Translation of Molecule Molecular Description ROUGE-L ↑ 0.23 0.05 0.75 0.29 0.75 0.78 Molecular Captioning MENTOR ↑ 0.42† 0.15 — 0.45 0.59|0.59† 0.60|0.58†

###### Protein Functional Translation

CASPSimilarSeq ROUGE-L↑ 0.01 0.02 — 0.74 0.83 0.85 IDFilterSeq ROUGE-L↑ 0.01 0.03 — 0.70 0.77 0.82 UniProtSeq ROUGE-L↑ 0.01 0.03 — 0.71 0.84 0.88 Mol-Instructions ROUGE-L↑ 0.21† 0.07 0.68 0.44 0.83|0.82† 0.99|0.98†

- Table 3 | Evaluation results of scientific translation tasks. Bold indicates the best performance, and underline indicates the second best. The names of expert models and specialist LLMs for each task are provided in Appendix Table. 10. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely largescale test sets exist for certain tasks, which would lead to substantial testing costs.

therefore probes compositional reasoning, ontology grounding, and robustness to notation/paraphrase as a modality-agnostic measure of semantic fidelity. The results are shown in Tab. 3.

Molecular Representation Translation. Following SMolInstruct [100], we evaluate the model’s ability to convert faithfully among common chemical representations, i.e., SMILES strings, IUPAC names, and empirical formulae. The tasks cover SMILES↔IUPAC, SMILES↔Formula, and IUPAC↔Formula, with performance judged by exact-match accuracy and structure-level equivalence after canonicalization.

- • SMILES to IUPAC Translation. This task takes a molecule’s SMILES string as input and converts it into its corresponding IUPAC name. This is a fundamental challenge in cheminformatics, as it requires the model to interpret a simplified linear notation of a chemical structure and convert it into the standardized naming convention used by chemists.
- • SMILES to Formula Translation. This task takes a molecule’s SMILES string as input and converts it into its corresponding molecular formula. The model receives a SMILES string as input and generates the complete molecular formula, which specifies the number of atoms for each element in the molecule. This task is crucial for computational chemistry, as it bridges the gap between a structural representation and the molecule’s elemental composition.
- • IUPAC to SMILES Translation. This task converts a chemical name into its corresponding SMILES string. The model takes a molecule’s IUPAC name as input and generates the SMILES string, which is a concise text-based representation of the molecule’s chemical structure. This task is fundamental for many cheminformatics applications, as it allows for the conversion of human-readable names into a format that computers can process.
- • IUPAC to Formula Translation. This task takes a molecule’s IUPAC name as input and converts it into its corresponding molecular formula. The model’s output is a list of all elements and the number of atoms for each element in the molecule. This task is crucial for chemical informatics, as it links a molecule’s systematic name to its elemental composition.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 384]

Molecular Captioning Case Input

Please provide a brief introduction to this molecule. <SMILES> CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(C)=O </SMILES>

Reference The molecule is a 1,2-diacyl-sn-glycero-3-phosphocholine in which the acyl groups at positions 1 and 2 are specified as (9Z)-hexadecenoyl and acetyl respectively. It has a role as a mouse metabolite. It derives from a palmitoleic acid.

Our model The molecule is a 1,2-diacyl-sn-glycero-3-phosphocholine in which the 1and 2-acyl groups are palmitoleoyl and acetyl respectively. It has a role as a mouse metabolite. It is functionally related to a palmitoleic acid.

GPT-5 This molecule is an unsaturated analog of platelet-activating factor (PAF), consisting of a glycerol backbone esterified at the sn-1 position with hexadec9-enoic acid, acetylated at the sn-2 position, and linked at the sn-3 position to a phosphocholine group. The presence of a cis double bond in the long acyl chain introduces unsaturation, which may influence its membrane dynamics and biological activity. As a member of the PAF family, it is potentially involved in inflammatory signaling and immune modulation.

- Figure 5 | Comparison of molecular captioning outputs.

Evaluation Metric. We evaluate performance using two metrics: Top-1 Split Match and Top-1 Element Match.

- • Split Match. This metric assesses correctness by parsing the predicted formula 𝑃 and the ground-truth formula 𝐺 into structured components according to a predefined splitting rule. Let judge_string_split_match(𝑃, 𝐺) denote a Boolean function that returns true if 𝑃 and 𝐺 are equivalent under this split-based comparison. The metric is computed as

𝑀split =

1

𝑁

∑︁𝑁

𝑖=1

𝕀 ∃𝑗 judge_string_split_match(𝑃𝑖, 𝐺𝑖,𝑗) ,

where 𝕀(·) is the indicator function, 𝑃𝑖 the 𝑖-th prediction, and {𝐺𝑖,𝑗} the corresponding set of references.

- • Element Match. This metric provides a more permissive comparison by disregarding element order and only checking whether the predicted and reference formulas contain identical counts of each element. For example, C5H8 and H8C5 are considered equivalent. Define 𝐶(𝐹) as the function that maps a formula 𝐹 to its multiset of element counts. The metric is given by

###### ∑︁𝑁

1

𝕀 ∃𝑗 𝐶(𝑃𝑖) = 𝐶(𝐺𝑖,𝑗) .

𝑀element =

𝑁

𝑖=1

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 406]

SMILES: CC1=NC=C[N-]1.CC1=NC=C[N-]1.[Zn+2] Reference: zinc;2-methylimidazol-3-ide Our output: zinc;2-methylimidazol-3-ide

GPT-5: zinc bis(5-methylpyridin-2-ide)

LLaSMol: 2-[(E)-2-[2-hydroxy-2-(dimethylamino)acetyl]-3-oxocyclopent-3-en-1-ylidene]acetic acid Gemini2.5-Pro: bis(2-methylimidazolato)zinc(II)

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

(a) Ours (b) GPT-5 (c) Ours (d) GPT-5

2D Chemical Structure 3D Ball-and-Stick Model

###### Figure 6 | Case study of “ZIF-8” in Metal Organic Framework (MOF).

Natural Language Translation of Molecule. To bridge the gap between chemical representations and human-readable knowledge, we evaluate tasks that translate molecular structures into natural language descriptions.

- • Molecular Description. This is a generative task designed to describe a molecule based on its chemical structure. The model takes a molecule’s SMILES string as input. Its goal is to generate a natural language description that includes the molecule’s properties, functions, and a summary of its effects. This task is crucial for making complex chemical information accessible and understandable to a broader audience. This task comes from Mol-Instructions [81].
- • Molecular Captioning. Similar to the Molecular Description task, this task requires the model to output the SMILES string’s explanation, using data from SMolInstruct [100].

Evaluation Metric. We use ROUGE-L and Metric for Evaluation of Translation with Explicit ORdering (MENTOR) as metrics.

- As shown in Fig. 5, our model matches the reference on every core fact while avoiding unsupported inferences. It correctly identifies the lipid class as a 1,2-diacyl-sn-glycero-3-phosphocholine, specifies the exact acyl substituents (palmitoleoyl at sn-1 and acetyl at sn-2), reproduces the biological role (mouse metabolite), and links it to palmitoleic acid—all with normalized terminology and no embellishment.
- As shown in Fig. 6, we visualize the generated 2D chemical structure and 3D ball-and-stick model for CC1=NC=C[N-]1.CC1=NC=C[N-]1.[Zn+2], which is called “ZIF-8‘’ in Metal Organic Framework (MOF). Since MOF is a coordination compound, there is a “Coordinate bond” in MOF. Our method generates the IUPAC name of ZIF-8 successfully, while GPT-5 could not. Moreover, LLaSMol and Gemini2.5-Pro generate IUPAC names without meaningful 2D chemical structures. In addition to “Coordinate bond”, we also evaluate our model for generating molecules with only “covalent bonds”. As depicted in Fig. 7, our method is able to successfully translate SMILES to IUPAC name. However, the IUPAC name generated by GPT-5 yields the opposite chiral isomer, and the IUPAC names generated by LLaSMol and Gemini2.5-Pro are not consistent with the SMILES.

Protein Functional Translation. For the given protein sequence, we prompt the model to describe the function of the protein in a clear and coherent manner within the context of the natural language domain. It thus tests both the model’s reasoning when inferring functional roles from sequence data

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 437]

Create Date: 2025-07-28 SMILES: C[C@H](C(=O)N[C@H](CC1=CC=CC=C1)C(=O)C(=O)NCC2=CC=CC=N2)NC(=O)C3=CC(=C(C=C3)Cl)Cl Reference: 3,4-dichloro-N-[(2R)-1-[[(2R)-3,4-dioxo-1-phenyl-4-(pyridin-2-ylmethylamino)butan-2-yl]amino]-1-oxopropan-2-yl]benzamide

###### Our output: 3,4-dichloro-N-[(2R)-1-[[(2R)-3,4-dioxo-1-phenyl-4-(pyridin-2-ylmethylamino)butan-2-yl]amino]-1-oxopropan-2-yl]benzamide

GPT-5: (2S)-2-[[(2S)-2-benzyl-3-(pyridin-2-ylmethylcarbamoyl)propanoyl]amino]-N-(3,4-dichlorophenyl)propanamide LLaSMol: (2Z,5Z)-2-chloro-5-[2-[(2Z,5Z)-2-chloro-5-[(2-methyl-3-oxo-3-phenylpropanoyl)amino]penta-2,4-dienoyl]-3-oxo-3-phenylpropanoyl] penta-2,4-dienoic acid Gemini2.5-Pro: (4S)-4-{(2S)-2-(3,4-dichlorobenzamido)propanamido}-4-phenyl-N-(pyridin-2-ylmethyl)-2,3-dioxobutanamide

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

Ours GPT-5 LLaSMol Gemini2.5-Pro

- Figure 7 | Case study of SMILES→IUPAC from latest PubChem data. Given the input SMILES (top), our model generates the exact reference IUPAC name (shown in blue), preserving the 3,4dichloro-benzamide core, the pyridin-2-ylmethylamino 1,3-dicarbonyl motif, and the (2R)/(2R) stereochemistry; the reconstructed structure (bottom left) is therefore congruent with the input. In contrast, competing LLMs (GPT-5, LLaSMol, Gemini-2.5-Pro) yield chemically inconsistent names that alter substituents, lose halogens, change chain length, or flip chirality, leading to mismatched reconstructions.

and its proficiency at producing accurate, fluent text. These tasks include four different data for testing, (i) CASPSimilarSeq: the sequences that are similar to the sequences released by CASP14. (ii) IDFilterSeq: this version uses ID-filtered sequences to reduce redundancy. (iii) UniProtSeq: this version’s protein sequences are from the UniProt database. (iv) Mol-ins: The protein sequence of this version is derived from the Mol-Instructions [81] benchmark. Additionally, this test set requires the model to also output subcellular localization information. We therefore consider the above four data sources are four sub-tasks of Protein Functional Translation.

Summary of Evaluation Results. Our models consistently outperform baselines. The SciReasoner-

- 8B model achieves near-ceiling accuracy in formula recovery (>93%), markedly improves challenging SMILES-IUPAC translations, and delivers the highest scores in molecular captioning (MENTOR 0.61) and protein functional translation (ROUGE-L up to 0.98). These results highlight strong semantic fidelity and scalability, establishing a new frontier for scientific translation tasks.

###### 4.4. Scientific Knowledge Extraction and Question Answering

Unlike our sequence-based tasks that require models to process structured biological inputs such as DNA, RNA, protein sequences, or SMILES strings, the tasks grouped under Scientific Knowledge Extraction and QA focus purely on natural language understanding and reasoning. They evaluate our model’s ability to extract scientific knowledge, recognize scientific entities, and answer domain-specific questions directly from unstructured text. Furthermore, this isolates linguistic comprehension from sequence modeling, allowing us to probe whether models can capture the semantics of scientific literature, resolve complex entity relations, and deliver accurate answers across diverse contexts. The results are listed in Tab. 4.

Chemical Entity Recognition. This task entails detecting textual mentions of chemical substances from the inputs and assigning them to specific entities (e.g., chemical compounds). This poses chal-

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 463]

SciReasoner 8B Scientific Knowledge Extraction and QA

Gemini 2.5-pro

GPT oss

Non-LLM Experts

Specialist LLMs

SciReasoner 1.7B

Task Metric

Chemical Entity Recognition F1 ↑ 0.71 0.55 — 0.75 0.81 0.92 Chemical–Protein Interaction Extraction F1 ↑ 0.12 0.05 — 0.22 0.46 0.83 Chemical–Disease Interaction Extraction F1 ↑ 0.35 0.23 — 0.40 0.66 0.97 General Multiple Choice Question ACC ↑ 0.94 0.87 — 0.65 0.92 0.99 True/False Question ACC ↑ 0.62 0.67 — 0.55 0.64 0.68 Open-Ended Question BertScore ↑ 0.84 0.80 — 0.84 0.83 0.87

- Table 4 | Evaluation results of scientific knowledge extraction and QA tasks. Bold indicates the best performance, and underline indicates the second best. The names of expert models and specialist LLMs for each task are provided in Appendix Table. 11.

lenges for models since the heterogeneity of chemical nomenclature. Accordingly, the performance of this task provides a sensitive probe of an LLM’s domain competence in biomedical text understanding.

Chemical-Protein Interaction Extraction This task evaluates language models on chemical–protein interaction (CPI) annotation: given biomedical text snippets, the model must (i) detect mentions of chemical entities and protein/gene targets, (ii) link candidate pairs, and (iii) assign an interaction label that captures mechanism and polarity (e.g., binding). The task requires grounding each prediction to supporting evidence spans and correctly handling context , thereby probing biochemical and pharmacological knowledge under realistic nomenclature variability.

Chemical-Disease Interaction Extraction This task probes extraction of chemical-induced disease (CID) relations from biomedical literature. For the input, the model must (i) detect mentions of chemicals and diseases, (ii) link candidate pairs, and (iii) decide whether the passage asserts a causal, adverse relation in which exposure to the chemical induces or exacerbates the disease, citing the supporting span. Accurate CID extraction enables downstream applications in toxicology, pharmacovigilance, and disease-mechanism mapping.

General Multiple Choice Question The task is to select the correct option for each question, probing factual knowledge, conceptual understanding, and applied reasoning relevant to biology, chemistry, and other relevant fields.

True/False Question Following the practice in Mol-Instructions [81], this task requires the language models to answer research questions with positive/negative/uncertain results, without being given the references from PubMed.

Open-Ended Question Open-ended questions require free-form responses to input content related to biomedicine, without following a fixed format. Open-ended QA tests genuine biomedical understanding, demands clear self-contained explanations, mirrors real use.

Summary of Evaluation Results Scaling from SciReasoner-1.7B to SciReasoner-8B yields consistent gains across scientific knowledge extraction and QA. The SciReasoner-8B model surpasses prior Specialist LLMs on every tasks. The most pronounced improvements appear on relation extraction (chemical–protein and chemical–disease), reflecting stronger entity linking, synonym/acronym disambiguation, and evidence grounding; chemical entity recognition also benefits, indicating sharper

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 485]

SciReasoner 8B Molecular Property Regression

Gemini 2.5-pro

GPT oss

Non-LLM Experts

Specialist LLMs

SciReasoner 1.7B

Task Metric

ESOL RMSE ↓ 1.28 3.45 — 1.04 1.21 1.08 LIPO RMSE ↓ 1.21 1.65 — 1.01 0.94 0.82 Physicochemical Prediction MAE ↓ 0.027 1.329 — 0.013 0.003 0.002

###### Material Property Regression

MPRegression

1.32† 0.95 5.32 0.76 4.98|4.67† 5.12|5.21† SNUMATRegression 1.07† 1.00 1.83 1.16 2.18|2.18† 1.96|2.22† JARVISDFT 1.09† 0.88 4.10 0.70 5.11|5.28† 5.21|5.59† JARVISQETB 0.53† 0.50 59.44 1.05 138.59|148.71† 126.61|137.61† GNoME 1.28† 0.73 15.60 0.50 19.05|18.46† 22.03|21.81† hMOF 0.65† 0.67 1.48 0.66 1.05|1.04† 0.99|0.99† Cantor-HEA 0.77† 0.45 8.40 0.87 7.40|7.38† 7.18|7.02† QMOF 1.24† 0.86 2.05 0.93 8.55|8.76† 8.07|7.94† OQMD 1.20† 0.65 6.02 1.16 5.29|5.50† 5.43|5.67† OMDB 0.82† 0.93 1.55 1.01 1.40|1.41† 1.39|1.42†

- MAD

- MAE ↑

Biology Multi-omics Property Regression Fluorescence Spearman ↑ -0.43† 3.20 69.00 2.57 69.36|70.82† 66.24|68.49† StabilityBiology-Instructions Spearman ↑ -9.15† -7.79 79.00 60.25 63.25|55.67† 64.41|59.03† Thermostability Spearman ↑ 5.47† 6.29 78.00 45.07 56.25|53.17† 58.55|54.45† Enhancer Activity Prediction PCC ↑ -4.24† -1.29 68.00 57.24 65.07|70.20† 64.39|52.08† APA Isoform Prediction R2 ↑ 0.13† 0.01 50.82 59.01 87.94|86.19† 85.87|87.92† Mean Ribosome Loading Prediction R2 ↑ 0.00† 0.02 78.00 47.64 50.82|54.56† 60.44|64.5† Programmable RNA Switches R2 ↑ 0.01† 0.01 55.67 26.65 39.00|38.1† 43.55|29.20† CRISPR On Target Prediction Spearman ↑ -3.64 6.21 44.10 2.87 14.35 25.81 siRNA Efficiency Prediction Mixed-score ↑ 38.75† 40.99 49.38 42.92 62.56|64.13† 59.67|61.94†

- Table 5 | Evaluation results of property prediction tasks. Bold indicates the best performance, and underline indicates the second best. The names of expert models and specialist LLMs for each task are provided in Appendix Table. 12. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely largescale test sets exist for certain tasks, which would lead to substantial testing costs.

domain-specific span detection. In QA, multiple-choice performance approaches a ceiling, open-ended questions and true/false questions remain stable yet competitive with strong general-purpose models. Overall, scaling primarily boosts entity-dense, relation-heavy reading while preserving high performance on general scientific QA.

###### 4.5. Property Prediction Tasks

We benchmark our single instruction-tuned model on a unified suite of composition-to-property and sequence-to-property regressions spanning materials, molecules, and multi-omics. Inputs include chemical formulas, SMILES, and DNA/RNA/protein sequences, with targets covering structural, electronic, thermodynamic, and biochemical properties. The results are listed in Tab. 5.

Molecular Property Regression. We consider three molecular regression tasks from SMolInstruct [100] and Mol-Instructions [81]: ① ESOL Prediction, ② LIPO Prediction, and ③ Physicochemical Prediction. These tasks perform single-output regression to quantitatively predict fundamental molecular characteristics directly from biochemical representations.

• Estimated Solubility (ESOL) Prediction. The ESOL Property Prediction task on property prediction for a molecule. The model receives a chemical structure in the form of a SMILES string. Its goal is to predict the aqueous solubility for that molecule. This is a regression task, requiring a single continuous value as output. The predicted value represents the logarithm of

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 507]

solubility (logS) in mol/L.

- • Lipophilicity (LIPO) Prediction. The LIPO Property Prediction task predicts the octanol-water distribution coefficient (logD). The model takes a molecule’s SMILES string as input. It calculates the logD value under the specific condition of pH 7.4. The final output is a single numerical value for this coefficient.
- • Physicochemical Prediction. Physicochemical prediction leverages a molecule’s structural details to estimate its core physicochemical traits. By inferring unmeasured attributes, it guides the early prioritization of promising candidates and reduces experimental burden.

The above task takes an SMILES string as input, and then requires the model to output a specific property, as shown in the following example for the Physicochemical Prediction task.

|Example:<br><br>Instructions: Could you give me the LUMO energy value of this molecule? <SMILES> CC1=CC2C(CO)C2C1 </SMILES> Response: 0.0191|
|---|

Evaluation Metric. For ESOL Prediction and LIPO Prediction, we use RMSE as the evaluation metric. For Physicochemical Prediction, we use MAE as the evaluation metric.

Material Property Regression Follow LLM4MAT [105], we evaluate the model using data sources from multiple databases, i.e., ① MPregression [111], ② SNUMATregression [98], ③ JARVIS-QETB [112], ④ JARVIS-DFT [113, 114], ⑤ GNoME [47], ⑥ Cantor-HEA [115], ⑦ QMOF [116, 117], ⑧ OQMD [118],⑨ OMDB [118], and ⑩ hMOF [119], organized into a “structure-to-property” task designed to predict various crystal properties based on input chemical formulas.

- • For Materials Project (MP), we predict band gap (eV), unit-cell volume (Å3), formation energy per atom (eV/atom), energy above hull (eV/atom), total energy per atom (eV/atom), thermodynamic stability flag, mass density (g/cm3), atomic density (atoms/Å3) and Fermi energy (eV).
- • For SNUMAT, we predict both HSE and GGA band gaps (eV), the corresponding optical band gaps (eV), spin–orbit coupling flag.
- • For Jarvis-QETB, we predict energy per atom (eV/atom), indirect gap (eV), Fermi energy (eV) and final energy (eV).
- • For Jarvis-DFT, our model outputs formation energy per atom (eV/atom), band gap (eV), spectroscopic limited maximum efficiency (SLME), electronic spillage, total energy (eV), static dielectric constant (x-direction) at the meta-GGA TBmBJ level of theory (MEPSX), maximum electric field gradient (V/Å2), average electron mass, maximum piezoelectric coefficient (C/m2 and pC/N), dielectric constant, n- and p-type Seebeck coefficients (µV/K), power factors (µW/cm·K2), exfoliation energy (meV/atom), bulk and shear moduli (GPa), modified Becke–Johnson band gap (eV) and energy above hull (eV/atom).
- • For GNoME, we predict formation and decomposition energies (eV/atom), band gap (eV), corrected total energy (eV), volume (Å3) and density (g/cm3).
- • For Cantor-HEA, we estimate formation energy per atom (eV/atom), energy above hull (eV/atom), volume per atom (Å3/atom) and energy per atom (eV/atom).
- • For QMOF, we predict total energy (eV), band gap (eV), largest cavity diameter and pore-limiting diameter (both Å).
- • For OMDB and OQMD, we predict band gap (eV) and, we also predict formation energy (eV/atom) for OQMD.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 529]

[Figure 530]

[Figure 531]

(a) Density (b) FEPA

- Figure 8 | GNoME distributions (Density vs FEPA)

[Figure 532]

(a) Ehull (b) EPA

[Figure 533]

- Figure 9 | Materials Project distributions (Ehull vs EPA)

• For hMOF, we predict maximum and minimum CO2 adsorption capacities (mmol/g), largest cavity diameter (Å), pore-limiting diameter (Å), void fraction, and both gravimetric (m2/g) and volumetric (m2/cm3) surface areas.

We follow the instruction template in LLM4MAT [105].

|Example:<br><br>Instructions: You are a material scientist. Look at the chemical composition and predict the requested property. Respond in a JSON format. chemical formula: KPrMnNbO6 property name: volume Response: {volume : 128.62405239223665}|
|---|

Evaluation Metric. For the above tasks, the metric is:

𝑚 𝑗=1 |𝑆𝑗|MADMAE𝑗𝑗

###### ∑︁𝑛

###### ∑︁𝑛

1

1

- MAD

- MAE

MAD =

|𝑦𝑖 − ¯𝑦| MAE =

|ˆ𝑦𝑖 − 𝑦𝑖|

(6)

=

𝑚 𝑗=1 |𝑆𝑗|

𝑛

𝑛

𝑖=1

𝑖=1

As shown in Fig. 8, and Fig. 9, our model is capable of predicting a broad spectrum of structural, electronic and thermodynamic properties across multiple materials databases.

Biology Multi-omics Property Regression For multi-omics (i.e., DNA, DNA and protein) tasks, we consider 9 regression tasks in Biology-Instructions [85]: ① Fluorescence Prediction, ② Stability Prediction, ③ Thermostability Prediction, ④ Enhancer Activity Prediction, ⑤ Alternative Aolyadenylation (APA) Isoform Prediction, ⑥ Mean Ribosome Loading Prediction, ⑦ Programmable RNA Switches, ⑧ CRISPR On Target Prediction, and ⑨ siRNA Efficiency Prediction, they perform single-output or multi-output regression to predict diverse biological properties.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 555]

[Figure 556]

[Figure 557]

(a) Enhancer Activity Prediction (b) Mean Ribosome Loading Prediction

- Figure 10 | Biology Multi-omics distributions.

- • Fluorescence Prediction. This task presents a regression task to predict the fluorescence of higher-order green fluorescent protein (GFP) mutants. Each protein sequence is mapped to the logarithm of its fluorescence intensity.
- • Stability Prediction. This task presents a regression task to predict the intrinsic stability of proteins. The model maps each protein sequence to a continuous stability score reflecting its ability to maintain its fold above a specific concentration threshold (e.g., the EC50 value).
- • Thermostability Prediction. This task is to predict protein thermostability, defined as a protein’s stability at elevated temperatures.
- • Enhancer Activity Prediction. This task is a multi-output regression task to predict the activity levels of enhancer regions from DNA sequences. The model output two numeric values corresponding to the enhancer’s housekeeping and developmental activity levels.
- • Alternative Aolyadenylation (APA) Isoform Prediction. This tesk presents a regression task to predict APA isoform usage from RNA sequences. The model outputs a proportion between 0 and 1 for each isoform, which represents its relative expression and captures the variability in polyadenylation signal processing.
- • Mean Ribosome Loading Prediction. This task presents a regression task to predict ribosome loading efficiency from RNA sequences. The model outputs a continuous value representing the mean ribosome loading, reported to two decimal places. Accurate prediction of ribosome loading is essential for understanding how cis-regulatory elements, particularly 5’ untranslated regions (UTRs), influence translation efficiency.
- • Programmable RNA Switches. This task addresses a multi-label regression task to predict the behavior of programmable RNA switches from their sequences. The model outputs three continuous values representing the “ON”, “OFF”, and “ON/OFF” states, each reported to two decimal places.
- • CRISPR On Target Prediction. This task involves a regression task to predict the on-target knockout efficacy of single guide RNA (sgRNA) sequences for use in CRISPR systems.
- • siRNA Efficiency Prediction. The model predicts the gene-silencing efficiency of modified small interfering RNA (siRNA). Based on the siRNA and corresponding target sequences, the model outputs a numeric value representing the percentage of mRNA remaining after treatment.

Evaluation Metric. For Fluorescence Prediction, Stability Prediction, Thermostability, and CRISPR On Target Prediction, the Spearman score is used as the evaluation metric. For APA Isoform Prediction, Mean Ribosome Loading Prediction, and Programmable RNA Switches, the R2 score is adopted as the evaluation metric.

For Enhancer Activity Prediction, PCC is used as the evaluation metric:

𝑛 𝑖=1(𝑥𝑖 − 𝑥¯)(𝑦𝑖 − ¯𝑦)

𝑛 𝑖=1(𝑥𝑖 − 𝑥¯)2√︃

√︃

PCC =

(7)

𝑛 𝑖=1(𝑦𝑖 − ¯𝑦)2

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 579]

For siRNA Efficiency Prediction, we consider the metric of Mixed-Score:

Range_MAE

MAE 100 ) + 50% · F1 · (1 −

Mixed-Score = 50% · (1 −

100 ) where (8) F1 = 2 ·

###### ∑︁𝑚

Precision · Recall Precision + Recall

1

and Range_MAE =

|𝑦𝑗 − ˆ𝑦𝑗| (9)

𝑚

𝑗=1

Summary of Evaluation Results Across materials, molecules, and multi-omics, scaling our model from SciReasoner-1.7B to SciReasoner-8B yields consistent accuracy gains, i.e., lower MAE/RMSE for numeric regressions and higher rank-based correlations (e.g., Spearman/PCC) for sequence-toproperty tasks. Relative to baselines, the SciReasoner-8B model matches or surpasses the strongest non-LLM encoder baselines and/or Specialist LLMs on multiple benchmarks in Tab. 5, especially on long-context or multi-output targets.

- • Materials (composition→property). The SciReasoner-8B model improves across MP, SNUMAT, JARVIS, GNoME, Cantor-HEA, OQMD/OMDB, QMOF, and hMOF, with notable MAE drops on harder thermodynamic/electronic properties (e.g., energy-above-hull, band gap, density/surfacearea). On a subset of properties (see Tab. 5), SciReasoner-8B ties or outperforms encoder-based baselines / Specialist LLMs.
- • Molecules (SMILES→property). For ESOL and LIPO (RMSE↓) and physicochemical endpoints (MAE↓), SciReasoner-8B consistently surpasses SciReasoner-1.7B, with the largest gains on endpoints sensitive to subtle substructure/charge effects. In several cases, SciReasoner-8B attains or exceeds Specialist LLMs, while SciReasoner-1.7B remains competitive but lags at the extremes (very hydrophilic/hydrophobic regimes).
- • Biology multi-omics (DNA/RNA/protein→property). The SciReasoner-8B model yields broad lifts in correlation-based metrics—Fluorescence, Stability/Thermostability, CRISPR on-target, and Mean Ribosome Loading—and improves composite outcomes such as the siRNA Mixed-Score. The largest SciReasoner-1.7B→SciReasoner-8B jumps appear in multi-output or context-dependent tasks (e.g., Enhancer Activity with housekeeping/developmental readouts, APA isoform usage).

###### 4.6. Property Classification Tasks

Property classification tasks evaluate a model’s capability to assign discrete categorical labels to small molecular, material, genetic, and multi-omics inputs, thereby enabling systematic assessment of structure–property relationships across various domains. We show the results of property classification in Tab. 6 and Tab. 7.

Molecular Classification Molecular classification tasks evaluate a model’s ability to infer discrete biological or pharmacological properties of molecules directly from their structural representations, enabling rapid assessment of drug permeability, toxicity, antiviral potential, and adverse effect risks that are central to drug discovery and safety evaluation.

• Blood-Brain Barrier Permeability (BBBP) Prediction. The BBBP property prediction task is designed to predict whether a given molecule can penetrate the blood-brain barrier. The model uses the SMILES string of a molecule as input. It then outputs a binary classification result, indicating if the molecule is blood-brain barrier permeable or not. This is a crucial property for drug discovery, as it helps determine if a potential drug can be effective for central nervous system targets.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 601]

Gemini 2.5-pro

GPT oss

Non-LLM Expert Models

Specialist LLMs

Task Metric

SciReasoner-1.7B SciReasoner-8B

Molecular Classification BBBP ACC@1 ↑ 71.57% 46.70% — 75.10% 79.70% 82.74% Clintox ACC@1 ↑ 26.39% 52.78% — 93.10% 92.36% 91.67% HIV ACC@1 ↑ 26.7%† 69.86% — 96.70% 96.79%|92.40%† 96.81%|95.90%† Sider ACC@1 ↑ 67.7%† 38.25% — 70.70% 70.00%|66.20%† 68.53%|68.70%†

###### Material Property Classification

MPclassification AUC↑ 0.58† 0.53 0.72 0.51 0.70|0.69† 0.72|0.71† SNUMATclassification AUC↑ 0.59† 0.60 0.72 0.58 0.69|0.69 0.66|0.66

###### Gene-centric Association Classification

Precision↑ 0.01 0.03 — 0.39 0.41 0.41

gSymbol2Tissue

Recall↑ 0.01 0.09 — 0.91 0.82 0.83 F1 score↑ 0.01 0.04 — 0.53 0.51 0.52

Precision↑ 0.00 0.01 — 0.36 0.75 0.81

gSymbol2Cancer

Recall↑ 0.00 0.02 — 0.32 0.72 0.81 F1 score↑ 0.00 0.01 — 0.32 0.73 0.81

Precision ↑ 0.00 0.01 — 0.27 0.68 0.80

gName2Cancer

Recall ↑ 0.00 0.04 — 0.26 0.67 0.79 F1 score ↑ 0.00 0.01 — 0.25 0.67 0.79

- Table 6 | Evaluation results of molecular classification tasks, material property classification tasks, and gene-centric association classification tasks. Bold indicates the best performance, and underline indicates the second best. The names of expert models and specialist LLMs for each task are provided in Appendix Table. 13. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely large-scale test sets exist for certain tasks, which would lead to substantial testing costs.

- • Clinical Toxicity (ClinTox) Prediction. The ClinTox property prediction task is designed to assess the toxicity of a molecule. The model takes the molecule’s SMILES string as input. It then provides a binary classification indicating whether the molecule is toxic or not based on its chemical structure. This prediction is vital in the early stages of drug development to filter out potentially harmful compounds.
- • HIV Prediction. The HIV property prediction task determines whether a molecule can inhibit the replication of the HIV virus. The model uses a molecule’s SMILES string as input. It then performs a binary classification to predict if the molecule is an HIV replication inhibitor or not. This is an essential step in identifying potential antiviral drugs for treating HIV/AIDS.
- • Side Effect Resource (SIDER) Prediction. The SIDER property prediction task predicts whether a molecule causes a specific side effect. The model takes a molecule’s SMILES string as input. It then provides a binary classification indicating if the molecule is associated with a given side effect. This task is crucial for drug safety, helping researchers to identify potential adverse drug reactions early in the development process.

Material Property Classification We consider two material property classification tasks, MPclassification and SNUMATclassification, which evaluate the ability of a model to classify various crystal properties of an input chemical formula.

• For the Materials Project (MP), it consists of two sub-tasks. One of them is to predict whether the gap of the crystal is formed directly, and the other is to predict whether a crystal is stable.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 623]

|Example:<br><br>Instructions: You are a material scientist. Look at the chemical composition and predict whether the material is stable. Respond in JSON. chemical formula: KPrMnNbO6 property name: is_stable Response: {is_stable: False}|
|---|

• For the SNUMAT dataset, it outputs direct/indirect gap flag.

|Example:<br><br>Instructions: You are a material scientist. Given the chemical composition, predict whether the band gap is direct (True) or indirect (False). Respond in JSON. chemical formula: PbS property name: is_direct Response: {is_direct: True}|
|---|

Evaluation Metric. We use AUC as the evaluation metric for these two tasks.

Gene-centric Association Classification. We assess gene-centric knowledge with three multi-label tasks mapping a gene identifier (symbol or official name) to tissue expression profiles or curated gene–cancer associations as in OPI [102].

- • Tissue Location Prediction from Gene Symbol (gSymbol2Tissue). This task involves predicting the tissue expression profile of a protein-coding gene given its gene symbol. Predicting spatial expression patterns aids in understanding protein-specific biological roles and informs biomarker discovery and therapeutic targeting.
- • Cancer Prediction from Gene Symbol (gSymbol2Cancer). Given a gene symbol, this task asks the model to identify associated cancer types, using data from the Cancer Gene Census. This task evaluates the model’s capability to mine gene-cancer associations from existing biomedical knowledge, supporting cancer gene discovery and precision oncology efforts.
- • Cancer Prediction from Gene Name (gName2Cancer). Given a gene symbol, this task asks the model to identify associated cancer types, using data from the Cancer Gene Census. This task evaluates the model’s capability to mine gene-cancer associations from existing biomedical knowledge, supporting cancer gene discovery and precision oncology efforts.

Evaluation Metric. We use Precision, Recall, and F1 score as the evaluation metrics for these three tasks. We present the case study of the gSymbol2Cancer task as follows:

Biology Multi-omics Property Classification For biology-related (DNA/RNA/protein) classification tasks, we include the following 26 tasks from the Biology-Instructions [85] and Mol-Instructions [81]. These tasks span binary and multi-label classification problems covering molecular interactions, regulatory elements, and functional annotation.

Specifically, we first consider 11 binary classification task:

• Solubility Classification. For a given protein sequence, this task addresses a binary classification task to predict whether a it is soluble or insoluble.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 645]

SciReasoner 8B Biology Multi-omics Property Classification (binary)

Gemini 2.5-pro

GPT oss

Non-LLM Expert Models

Specialist LLMs

SciReasoner 1.7B

Task Metric

SolubilityBiology-Instructions ACC ↑ 51.4† 51.22 77.00 63.02 62.67|67.3† 71.61|69.4† Antibody-Antigen MCC ↑ −5.05† −2.78 54.90 1.06 41.59|45.43† 45.5|48.84† RNA-Protein MCC ↑ 1.20† −5.23 93.2 74.26 86.47|85.97† 84.97|88.8† Epigenetic Marks Prediction MCC ↑ 0.52† 0.84 58.83 3.64 14.16|18.09† 17.24|21.26†

- TF-Mouse0 MCC ↑ 0.20† 4.75 64.23 26.50 41.79|39.77† 43.52|40.42†

- TF-Mouse1 MCC ↑ −1.24† −0.38 86.28 66.96 68.95|72.07† 71.03|75.98†

- TF-Mouse2 MCC ↑ 0.41† 6.99 81.28 63.98 71.57|69.08† 75.07|68.52†

- TF-Mouse3 MCC ↑ 2.72† −9.63 73.49 32.02 44.81|45.10† 52.98|53.31†

- TF-Mouse4 MCC ↑ 6.61† −2.07 52.66 9.18 27.34|28.45† 28.62|28.20† Enhancer-Promoter Interaction MCC ↑ −3.75† −1.69 53.59 4.13 8.18|6.75† 14.40|8.12†

Promoter-300all MCC ↑ 24.12 -3.56 91.01 78.97 82.94 83.01 Promoter-300notata MCC ↑ 16.65 -2.40 94.34 83.45 88.46 87.04 Promoter-300tata MCC ↑ 11.98 13.04 79.43 38.50 44.38 51.37 Promoter-coreall MCC ↑ 12.38 -3.57 70.92 59.98 61.49 62.56 Promoter-corenotata MCC ↑ 15.64 -6.37 71.58 62.85 61.72 63.52 Promoter-coretata MCC ↑ 13.02 17.97 78.15 39.36 53.67 46.94

- TF-Human0 MCC ↑ -3.46 3.24 71.99 56.56 59.33 61.07

- TF-Human1 MCC ↑ 12.23 3.85 76.06 61.17 62.22 63.47

- TF-Human2 MCC ↑ 2.29 5.19 66.52 46.92 47.80 48.02

- TF-Human3 MCC ↑ -8.44 -1.27 58.54 34.16 38.96 40.89

- TF-Human4 MCC ↑ 7.56 15.06 77.43 51.86 56.43 56.10 SolubilityPEER ACC ↑ 58.40† 10.91 74.37 71.40 70.14|70.8† 73.99|75.4† StabilityPEER ACC ↑ 49.60† 38.21 58.30 63.50 68.41|68.90† 64.70|64.3† Human PPI ACC ↑ 52.74 58.2 85.09 84.8 76.79 73.42 Yeast PPI ACC ↑ 51.52 44.90 66.16 60.40 53.81 60.41

Biology Multi-omics Property Classification (multi-label) Protein Function ROUGE-L 0.20† 0.05 0.57 0.43 0.63|0.63 0.97|0.96 Domain/Motif ROUGE-L 0.19† 0.02 0.53 0.46 0.56|0.56 0.97|0.90 FunctionEC Fmax↑ 3.3† 22.86 88.9 19.79 60.65|60.33† 77.26|77.56† Non-coding RNA Func. Classif. ACC ↑ 17.8† 11.54 85.73 63.09 80.39|88.8† 86.65|93.4† Modification Prediction AUC ↑ 51.19 50.05 84.00 59.06 53.06 59.94 Subcellular Localization Accuracy↑ 0.02 0 — 0.78 0.84 0.91 Fold Type Accuracy↑ 0.01 0 — 0.61 0.34 0.48

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0.59 0.48 0.49

0.37 0.34 0.35

0.40 0.39 0.39

0.55 0.55 0.54

Numerical EC (New-392)

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0.58 0.47 0.49

0.07 0.07 0.07

0.10 0.10 0.10

0.21 0.23 0.22

Numerical EC (Price-149)

Precision↑ Recall↑ F1↑

0.04 0.03 0.03

0 0.02 0.01

-

0.81 0.74 0.76

0.83 0.80 0.81

###### 0.86 0.83 0.84

Keywords (CASPSimilarSeq)

Precision↑ Recall↑ F1↑

0.03 0.02 0.01

0 0.02 0.01

-

0.84 0.80 0.82

0.86 0.85 0.85

###### 0.89 0.89 0.88

Keywords (IDFilterSeq)

Precision↑ Recall↑ F1↑

0.01 0 0

0.01 0.03 0.01

-

0.86 0.82 0.83

0.85 0.90 0.88

###### 0.92 0.91 0.91

Keywords (UniProtSeq)

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

-

0.76 0.74 0.75

0.81 0.81 0.80

###### 0.86 0.86 0.86

GO Terms (CASPSimilarSeq)

###### 0.83 0.82 0.81

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

-

0.74 0.73 0.72

0.78 0.77 0.77

GO Terms (IDFilterSeq)

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

-

0.76 0.74 0.74

0.84 0.82 0.81

0.88 0.87 0.86

GO Terms (UniProtSeq)

- Table 7 | Evaluation results of biology multi-omics property classification tasks. Bold: best, underline: the second best. Details of expert models and specialist LLMs are provided in Appendix Table. 14. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task to avoid substantial testing costs.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 667]

Cancer Prediction from Gene Symbol Case Input

Get the cancer name that is related to the given gene symbol: PLAG1

Reference salivary adenoma, lipoblastoma Our model salivary adenoma; lipoblastoma GPT-5 Pleomorphic adenoma (salivary gland).

- Figure 11 | Case study of cancer prediction from a gene symbol (PLAG1). The reference lists two associated tumors: salivary adenoma and lipoblastoma. Our model reproduces both exactly (precision

= 1.00, recall = 1.00, F1 = 1.00). GPT-5 returns only pleomorphic adenoma (salivary gland), which maps to the salivary adenoma category but misses lipoblastoma (precision = 1.00, recall = 0.50), indicating partial coverage.

- • Antibody-Antigenqing. This is a binary classification task to predict interactions between antibody and antigen sequences. The ultimate goal is to leverage this understanding to predict effective antibodies against new viral variants.
- • RNA-Protein Interaction Prediction. This is a binary classification task to predict interactions between non-coding RNAs (ncRNAs) and proteins based on their respective sequences. Most ncRNAs must interact with proteins to perform their biological functions. Therefore, identifying these interactions is crucial for elucidating the mechanisms underlying ncRNA-mediated biological activities .
- • Epigenetic Marks Predictions. For a given DNA sequence, this binary classification task predicts the presence of chemical modifications in DNA sequence.
- • TF-Mouse. This is a binary classification task, to determine whether specific regions with transcription factors binding in the DNA sequences or not. These transcription binding sites (TBS) are critical for controlling the initiation, enhancement, or repression of transcription.
- • Enhancer-Promoter Interaction Prediction. This task addresses a binary classification task focused on identifying interactions between enhancer and promoter regions within a given pair of DNA sequences. Predicting these enhancer-promoter interactions is crucial for understanding the complex regulatory networks that govern gene activity.
- • TF-Human Prediction. This is a binary classification task, to determine whether specific regions with transcription factors binding in the DNA sequences or not.
- • Promoter-300&Promoter-Core. These two tasks are binary classification problems focused on identifying the presence or absence of promoter regions in DNA sequences. The first task, Promoter Detection 300, involves detecting these regions within a 300 base pair (bp) window that includes both the core promoter and surrounding regulatory elements. The second task, Promoter Detection Core, focuses on detecting a shorter, core sequence (typically 50-100 bp) located directly upstream of the transcription start site.
- • Human PPI. This task presents a binary classification task to predict protein-protein interactions (PPI) in humans. Based on pairs of human proteins, the model outputs Positive/Negetive sentences indicating whether the two proteins interact .
- • Yeast PPI. This task presents a binary classification task to predict protein-protein interactions (PPI) in yeast. Based on pairs of yeast proteins, the model outputs Positive/Negetive sentences indicating whether the two proteins interact.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 689]

In summary, the above binary classification task is to evaluate whether the given DNA/RNA/protein sequence is positive or negative for a specific task. Using solubility classification as an example:

|Example:<br><br>Instructions: <protein>MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQSVT SRGDLATCPRGFAVTGCTCGSACGSWDVRAETTCHCQCAGMDWTGARCCRVQPLEHHHHHH</protein> This task predicts if a protein is likely to be soluble or insoluble. Can this protein be expected to be soluble? Response: Solubility is not predicted for this protein.|
|---|

More example please refer to the Appendix. Then, we consider 15 multi-label&multi-class classification tasks:

- • Domain/Motif Classification. This task requires models to pinpoint folded structural domains and conserved motifs within a protein sequence. These annotations provide a concise information of protein architecture that aids functional inference and downstream analyses.
- • Protein Function classification. Protein function classification maps an input protein sequence onto standardized annotations across the three GO ontologies: cellular component, biological process, and molecular function. This could enable high-throughput, comprehensive annotation of novel proteins.
- • Function-EC Classification. For a give protein sequence, this multi-label classification task predicts enzyme function by annotating protein sequences with their corresponding Enzyme Commission (EC).
- • Non-coding RNA Function Classification. This task presents a multi-class classification task for assigning non-coding RNA (ncRNA) sequences to one of 13 functional classes (e.g., ’tRNA’, ’miRNA’, ’riboswitch’). Accurate classification is essential for understanding the diverse regulatory roles of ncRNAs in biological processes and disease.
- • Modification Prediction. This task addresses a multi-label classification task to predict posttranscriptional modifications in RNA sequences, identifying one or more types from a set of 12 common modifications (e.g., “m6A”, “m1A”, “m5C”). The precise identification of these modification sites is essential for understanding RNA’s regulatory mechanisms and its roles in diverse biological processes.
- • Fold Type Classification. This task involves assigning a protein to one of 1195 fold types based on its amino acid sequence. Fold types reflect fundamental tertiary structural patterns, even across proteins with low sequence similarity.
- • Subcellular Localization Classification. This task requires predicting the subcellular localization (e.g., nucleus, cytoplasm, membrane) of a protein in eukaryotic cells based on its sequence. Subcellular location is tightly linked to protein function and disease relevance, making this task vital for biological interpretation and drug discovery.
- • Numerical EC Classification. This task entails predicting the GO terms associated with a protein sequence. GO annotations describe protein functions across three dimensions: molecular function, biological process, and cellular component. Unlike the Function-EC Classification task, this task outputs four numerical levels of EC number system. Moreover, since the test data comes from two distinct sets, i.e., New-392 and Price-149, we therefore consider these two datasets are two sub-tasks of Numerical EC Classification. We add extra spaces between the numbers and the periods to make sure the the four numbers are tokenized properly, for example,

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

(a) GO (CASPSimilarSeq) (b) GO (IDFilterSeq) (c) GO (UniProtSeq)

- Figure 12 | Gene Ontology (GO) term classification measured as the number of correctly predicted attributes per sequence (higher is better). Box–whisker plots compare two model sizes (SciReasoner8B, blue; SciReasoner-1.7B, red) on three evaluation sets: (A) CASPSimilarSeq, (B) IDFilterSeq, and (C) UniProtSeq. Diamonds mark the mean. Across all datasets, the SciReasoner-8B model shows higher medians and upper quartiles than the SciReasoner-1.7B model, with longer upper whiskers (reaching 9–10 correct terms vs. 5–8), indicating more high-quality predictions. Gains are most pronounced on IDFilterSeq and UniProtSeq, suggesting stronger generalization beyond CASP-similar sequences.

|Example:<br><br>Instructions: What is the input protein sequence’s EC identifier? <protein>MTAEPATKKIKLELSDPSEPLTQSDVIAFQKEALFRCINRRRVDFEALRKQYELSRRE...</pro tein> Response: 2 . 3 . 2 . 27|
|---|

- • Gene Ontology (GO) Terms Classification. This task entails predicting the GO terms associated with a protein sequence. GO annotations describe protein functions across three dimensions: molecular function, biological process, and cellular component. This tasks include three different data for testing, (i) CASPSimilarSeq: the sequences whose are similar to the sequences released by CASP14. (ii) IDFilterSeq: this version uses ID-filtered sequences to reduce redundancy. (iii) UniProtSeq: this version’s protein sequences are from the UniProt database. We therefore consider the above three data soucres are three sub-tasks of GO Terms Classification.
- • Function Keywords Predictions. This task targets the prediction of UniProtKB keywords associated with a protein, spanning 10 categories: biological process, cellular component, coding sequence diversity, developmental stage, disease, domain, ligand, molecular function, posttranslational modification, and technical term. These keywords provide concise yet informative summaries of protein characteristics and functions. Similar to the Terms Classification, we regard the (i) CASPSimilarSeq, (ii) IDFilterSeq, and (iii) UniProtSeq are three sub-tasks of Function Keywords Prediction.

In Fig. 12, we compare Gene Ontology (GO) term classification performance between the SciReasoner1.7B and SciReasoner-8B models across three evaluation sets. The box–whisker plots show that the SciReasoner-8B model consistently achieves higher medians and upper quartiles of correctly predicted attributes per sequence, with upper whiskers extending to 9–10 correct terms. These improvements are most evident on IDFilterSeq and UniProtSeq, highlighting stronger generalization beyond CASP-similar sequences.

We present the case study of the Function Keywords Prediction task in Fig. 13.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 736]

Function Keywords Prediction Case Input

Please give the functional keywords of the following protein sequences. <protein>MVEAFCATWKLTNSQNFDEYMKALGVGFATRQVGNVTKPTVIISQEGDKVV IRTLSTFKNTEISFQLGEEFDETTADDRNCKSVVSLDGDKLVHIQKWDGKETNFVREIKDGKMVMTLTFGDV VAVRHYEKA</protein>

Reference 3D-structure ; Acetylation ; Alternative splicing ; Cytoplasm ; Direct protein

sequencing ; Lipid-binding ; Reference proteome ; Transport

###### Our model Acetylation; Cytoplasm; Direct protein sequencing; Lipid-binding; Reference proteome; Transport

GPT-5 Binding; Metal-binding; Oxidoreductase

- Figure 13 | Case study of functional keyword prediction for a single protein sequence. The curated reference lists eight UniProtKB keywords. Our model correctly recovers six of them (Acetylation, Cytoplasm, Direct protein sequencing, Lipid-binding, Reference proteome, Transport), yielding 100% precision and 75% recall. By contrast, GPT-5 returns generic terms (Binding, Metal-binding, Oxidoreductase) with no overlap with the reference.

Summary of Evaluation Results Across all classification tasks, scaling from SciReasoner-1.7B to SciReasoner-8B consistently improves performance, often surpassing GPT baselines (GPT-o3/oss) and approaching specialist systems.

- • Molecular Classification. On BBBP, ClinTox, HIV, and SIDER, the 8B model achieves higher accuracy and AUC than 1.7B, reducing toxicity false positives and improving antiviral recall, showing stronger capture of structure–activity relationships.
- • Material Property Classification. For MPclassification and SNUMATclassification, 8B improves AUC by about 3% over 1.7B, narrowing the gap with expert material models and outperforming general-purpose baselines.
- • Gene-centric Association Classification. On gSymbol2Tissue, gSymbol2Cancer, and gName2Cancer, the 8B model gains up to 15 F1 over 1.7B, recovering curated gene–disease associations more completely than GPT-5 or smaller models.
- • Biology Multi-omics Classification. The largest relative gains appear here: 8B improves MCC/accuracy by 2–10 points across several tasks, and achieves state-of-the-art in multi-label GO and UniProt keyword prediction, with broader and higher-quality annotation coverage.

Overall, 1.7B → 8B scaling delivers robust gains across molecules, materials, genes, and multi-omics, positioning the 8B model as best or second-best on most tasks and validating unified cross-domain pretraining for discrete property inference.

###### 4.7. Generation and Design Tasks

We leverage the generation and design tasks to assess models’ ability to create novel scientific entities—ranging from molecules, proteins, RNAs to materials—either in an unconditional manner or under explicit structural and functional constraints, thereby providing a rigorous test of generative

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 758]

SciReasoner 8B Unconditional Generation

Gemini 2.5-pro

GPT oss

Non-LLM Experts

Specialist LLMs

SciReasoner 1.7B

Task Metric

Validity ↑ 0.568 0.952 — 0.968 0.864 0.971 Unique ↑ 0.042 0.318 — 0.966 0.988 0.982

Uncondi. Molecular Generation

Avg. Length↑ 128.87 318.91 — 284.50 280.10 284.70 Diversity↑ 0.62 1.00 — 0.97 1.00 0.91

Uncondi. Protein Generation

Validity↑ 0.93 0.99 — — 0.80 0.99 Uncondi. RNA Generation Avg. MFE↓ N/A -417.58 — -177.1 -178.85 -200.14 Uncondi. Material Generation SMACT↑ 89.70 N/A — 66.07 55.06 88.18

###### Conditional Material Generation

SMACT ↑ N/A N/A — 83.36 86.07 85.85 Precision ↑ N/A N/A — 98.44 99.89 99.96

Composition to Material

Novelty ↓ N/A N/A — 97.13 85.64 84.81 Bulk to Material

SMACT ↑ N/A N/A — 94.75 88.92 87.49 Success rate ↑ N/A N/A — — 7.12 7.05

Conditional Small Molecular Generation Forward Synthesis Prediction Exact Match ↑ 0.32† 0.33 — 0.63 0.69|0.81 0.75|0.83 Forward Reaction Prediction Exact Match ↑ 0.55 0.36 0.86 0.54 0.96 0.99 Reagent Prediction Exact Match ↑ 0.04 0.00 0.26 0.13 0.19 0.22 RetrosynthesisMol-Instructions Exact Match ↑ 0.20 0.05 0.64 0.41 0.72 0.83 RetrosynthesisUSPTO-50K Exact Match ↑ 0.23† 0.12 0.61 0.70 0.68|0.72 0.72|0.74 RetrosynthesisSMol-Instruction Exact Match ↑ 0.12† 0.08 — 0.33 0.41|0.43† 0.46|0.47† Molecule Generation Exact Match ↑ 0.25† 0.03 — 0.19 0.39|0.34† 0.48|0.46† Description-Guided Molecule Exact Match ↑ 0.06 0.04 0.11 0.002 0.10 0.12

###### Conditional Biochemical Generation

Max.

Function-guided Protein Design

Normalized SW ↑ 0.00† 0.00† 0.97 0.43 0.96|0.90† 1.00|1.00† Catalytic Activity ROUGE-L ↑ 0.25† 0.02 0.72 0.52 0.77|0.78† 0.99|0.95†

- Table 8 | Evaluation results of generation and design tasks. N/A indicates that the model cannot generate samples that meet the required specifications. Bold indicates the best performance, and underline indicates the second best. The names of expert models and specialist LLMs for each task are provided in Appendix Table. 15. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely largescale test sets exist for certain tasks, which would lead to substantial testing costs.

priors and controllable design capabilities across diverse scientific domains. We show the results of generation and design tasks in Tab. 8.

Unconditional Generation Tasks. To probe the model’s generative prior, we evaluate unconditional generation across four domains: RNA sequences, inorganic material compositions, small molecules, and proteins. In this setting, the model receives no task-specific conditioning beyond minimal format requirements (e.g., valid alphabet/grammar, length or syntax ranges) and must sample plausible candidates directly from its learned distribution. Furthermore, these tasks could provide seed candidates for downstream, property-conditioned design.

• Unconditional RNA generation. The unconditional RNA generation task aims to produce novel RNA sequences without any input conditions, using the four types of nucleotides. We show the visualization of unconditional RNA generation in Fig. 14.

|Example:<br><br>Instructions: Please generate a novel RNA sequence of length 50 nt.|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

- Figure 14 | Visualization of Unconditional RNA Generation. The pLDDT scores of these three RNA sequences are 74.6, 72.3, and 71.5, respectively. These values indicate that the generated structures achieve moderate confidence and highlight the model’s ability to produce diverse yet structurally plausible candidates.

|Response: AUGGCUACUGGUAACCGGAAUCCGGAUUCCGGAAUCCGGAAUCCGGAAUCCGGA|
|---|

- • Unconditional Material Generation. The unconditional material generation task aims to generate novel material compositions without any input constraints.

|Example:<br><br>Instructions: Produce a material composition with any bulk modulus. Response: {composition: "Li2O·TiO2·MnO2", bulk_modulus: 120.5 GPa}|
|---|

- • Unconditional Small Molecular Generation. This task aims to produce novel molecules that are both chemically plausible and structurally diverse, without any guiding input conditions.

|Example:<br><br>Instructions: Generate a molecule with <SMILES>. Response: <SMILES> CC(=O)CCCCCC(C)C </SMILES>|
|---|

- • Unconditional Protein Generation. This task is a generative task focused on creating new protein sequences or structures without any specific constraints. The purpose of this task is to test our model’s understanding of the fundamental principles of protein biology and to generate completely new proteins with biological feasibility from scratch.

###### Evaluation Metric.

• For unconditional molecular generation, 800 sequences are randomly and independently generated, and the proportions of valid sequences and unique sequences is evaluated by using RDKit.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 805]

- • For unconditional protein generation, 500 sequences are generated independently at random, and the average length, diversity (non-repetition rate), and validity (whether only containing valid characters) of these sequences are evaluated.
- • For unconditional RNA generation, 5000 sequences are generated independently at random, and the average minimum free energy (MFE) of the valid sequences is evaluated.
- • For unconditional material generation, 5000 materials are generated independently at random, and the percentage of valid materials is evaluated by Semiconducting Materials by Alternative Chemical Toolkit (SMACT).

Conditional Material Generation Tasks In conditional material generation tasks, the model is required to generate candidate material compositions that satisfy given constraints, such as specific elemental requirements or target physical properties. These tasks not only assess the model’s generative capability but also its ability to adhere to underlying chemical and physical principles in material design.

- • Composition to Material Generation. This task asks the model to design materials containing specific elements (e.g., Li, Ti, Mn, Fe, and O).

|Example:<br><br>Instructions: Given F, In, Tc, Zr, design a material composition. Response: {composition: "FeInTcZrO4", stability: "metastable"}|
|---|

- • Bulk to Material Generation. In this task, the model is asked to generate materials with a desired bulk modulus.

|Example:<br><br>Instructions: Given a bulk modulus of 75.321 GPa, propose a material composition. Response: {composition: "Al3Ti", predicted_bulk_modulus: 74.8 GPa}|
|---|

Evaluation Metrics. In addition to SMACT, we evaluate model performance using three complementary metrics: Precision, Novelty, and Success Rate.

- • Precision quantifies the proportion of correctly predicted elements relative to the total number of ground-truth elements, averaged across all samples. It reflects how accurately the predicted material matches the specified composition.
- • Novelty measures the proportion of predictions that do not appear in the ground-truth material set, thereby capturing the model’s ability to generate unseen candidates.
- • Success Rate denotes the fraction of predictions that exactly reproduce all ground-truth elements of a sample. It assesses whether the model can fully recover the intended material.

Conditional Small Molecular Generation Tasks We evaluate controllable, chemistry-aware generation under explicit conditioning signals, including reactants/reagents, textual descriptions, scaffolds, and target property constraints. These tasks probe whether a model can follow chemical instructions, satisfy constraints (e.g., validity, synthesizability, and target properties), and balance novelty with correctness, and we follow the task settings in Mol-Instruction [81] and SMolInstruct [100].

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 827]

###### Reactant Prediction

Instruction: Can you suggest some reagents that might have been used in the given chemical reaction? <SMILES> COC(=O)CC1=CC=CC=C1OCC2=CSC(CCC=3N=C(C4=CC=CC=C4)OC=3C)=N2 </SMILES>

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

Ours LLaSMol Gemini2.5-Pro

Reference

###### Retrosynthesis Prediction

Instruction: With the provided product, recommend some probable reactants that were likely used in its production. <SMILES>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8]) =[O:10])[cH:19][c:18]2[cH:17][cH:16]1 </SMILES>

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

Ours LLaSMol Gemini2.5-Pro

Reference

- Figure 15 | Visualization comparison on molecular generation tasks. Top (Reactant Prediction): given a product SMILES (instruction shown), models propose likely reagents. Bottom (Retrosynthesis Prediction): given a product SMILES, models suggest probable reactants/precursors. Our model recovers reactants that align with the reference chemistry, whereas baseline models propose off-target species.

- • Forward Synthesis Prediction. The Forward Synthesis task predicts the product of a chemical reaction given a set of reactants and reagents. The model takes the SMILES strings of the starting materials as input. Its goal is to generate the most probable product molecule, effectively simulating a chemical reaction. This task is fundamental to drug discovery and materials science, as it helps chemists predict the outcome of a synthesis and explore new reaction pathways. The Molecule Generation task is a generative task that creates a molecule based on a given set of properties and constraints. The model receives a natural language description as input. Its goal is to generate a valid SMILES string for a molecule that satisfies all the specified conditions. This task is highly valuable in drug discovery, as it allows for the design of novel molecules with specific desired characteristics.
- • Description Guided Molecule Design. This is a text-conditioned generation task that produces candidate molecules that satisfy explicit, user-specified criteria expressed in natural language (e.g., target activity). By conditioning the generative process on these constraints, the chemical search space is narrowed toward compounds consistent with the desired profile, improving the efficiency of design and optimization in applications such as drug discovery.
- • Forward Reaction Prediction. Forward reaction prediction forecasts the expected products of a chemical reaction from specified reactants and reagents.
- • Reagent Prediction. Reagent prediction identifies the optimal catalysts, solvents and auxiliary agents for any given reaction.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 857]

• Retrosynthesis Prediction. Retrosynthesis prediction task is to invert the synthetic process—beginning with the target molecule and working backward to identify viable precursors and reaction sequences.

Evaluation Metric. The exact match metric enforces the strictest criterion: a prediction is considered correct only if the predicted molecular formula string is identical to one of the ground-truth formulas in both element symbols and their order. Formally,

###### ∑︁𝑁

1

𝑀exact =

𝕀 ∃𝑗 𝑃𝑖 = 𝐺𝑖,𝑗 ,

𝑁

𝑖=1

where 𝑁 denotes the total number of samples, 𝑃𝑖 the predicted formula for the 𝑖-th sample, {𝐺𝑖,𝑗} the corresponding set of ground-truth formulas, and 𝕀(·) the indicator function that equals 1 if the condition holds and 0 otherwise.

We visualize the outputs in Fig. 15. Specifically, compared with LLaSMol and Gemini-2.5-Pro, our method recovers reagents/reactants that more closely match the reference chemistry in both reactantand retrosynthesis-prediction settings.

Conditional Biochemical Generation Tasks Follow Mol-Instructions [81], we assess the model’s capacity to generate biochemical outputs conditioned on specific inputs, ranging from function-guided protein design to enzyme reaction description from sequences.

- • Function-guided Protein Design. Given a functional prompt (e.g., “DNA binding”), the model outputs a corresponding amino-acid sequence wrapped in <protein> tags. This evaluates its ability to map explicit functional requirements to viable sequences. Function-driven protein design speeds up directed evolution, enzyme engineering and drug-target discovery.
- • Catalytic Activity Prediction. Catalytic activity prediction aims to infer the exact chemical transformation an enzyme performs directly from its amino acid sequence.

|Example:<br><br>Instructions: Using the protein sequence supplied, identify and describe the enzymatic catalytic activity, with emphasis on the chemical reaction it accelerates: <protein>...</protein> Response: Based on the provided protein sequence, the enzyme appears to facilitate the chemical reaction: (-)-secoisolariciresinol + NADP(+) = (+)-lariciresinol + H(+) + NADPH.|
|---|

Evaluation Metric. For Function-guided protein design, we use Normalized Smith-Waterman score (Normalized SW) as the metric.

Summary of Evaluation Results Scaling from SciReasoner-1.7B to SciReasoner-8B strengthens both unconditional and conditional generation across molecules, proteins, RNAs, and materials.

• Unconditional generation. SciReasoner-8B attains SOTA-level molecular validity while retaining very high uniqueness; it also improves material feasibility (SMACT) beyond prior Specialist LLMs. RNA samples exhibit lower (better) free energies than SciReasoner-1.7B and the LLM baseline.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

- Figure 16 | Structural overlay of the proteins: the reference structure is shown in gray and SciReasoner8B’s prediction in green.

- • Conditional material design. With element-set constraints, SciReasoner-8B maintains high chemical validity with near-perfect precision and improved (lower) novelty; under bulk-modulus conditioning, validity is comparable to SciReasoner-1.7B with a small trade-off in success rate.
- • Conditional molecular generation. SciReasoner-8B consistently surpasses both SciReasoner1.7B and prior Specialist LLMs across forward synthesis/reaction prediction, reagent selection, retrosynthesis, and text-guided molecule design, with the largest gains in retrosynthesis and top-1 exact matching.
- • Conditional Protein Design. Function-guided protein generation reaches near-ceiling sequence similarity and substantially outperforms prior LLM baselines. As shown in 16, SciReasoner-8B can generate proteins that are very close to the reference protein. More results please refer to the Appendix.

Overall, scaling primarily amplifies constraint satisfaction and chemical/structural validity under conditioning, while also strengthening the generative prior in unconditional regimes.

###### 4.8. Reasoning Examples

This section provides a detailed exposition of the model’s reasoning capabilities across diverse scientific domains. Each example illustrates the sophisticated chain-of-thought processing employed by the model to arrive at precise predictions and assessments, underscoring its efficacy in tackling complex scientific inquiries. These cases, ranging from materials science to molecular biology, demonstrate the model’s ability to interpret intricate inputs and generate well-supported conclusions. For each instance, the input prompt, the model’s detailed thinking content, the final answer, and a reference are provided to offer a transparent view of the complete reasoning trajectory.

###### 4.8.1. Material Task: SNUMAT Classification

- As shown in Fig. 17, the SNUMAT Classification example showcases a sophisticated ability to infer complex electronic properties from basic chemical information. Given the chemical formula RbCuCl3 and a request for its Direct_or_indirect_HSE, the model initiates its reasoning by correctly deconstructing the formula into its ionic components. Critically, it accurately deduces the +2 oxidation state of copper, which leads to the identification of its d9 electronic configuration. By integrating these pieces of information—ionic nature, specific oxidation states, and key electronic configurations—the model constructs a logical pathway to its final, correct prediction. This example demonstrates the model’s capacity to not just identify individual chemical features, but to connect them in a physically meaningful sequence to determine a material’s fundamental electronic character.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 904]

SNUMAT Classification

Input. You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: property_name:predicted_property_value. Answer as precise as possible and in as few words as possible. chemical formula: RbCuCl3 property name: Direct_or_indirect_HSE Thinking Content:

<think> The compound RbCuCl3 is analyzed based on its chemical formula. It consists of Rb+, Cu, and Cl− ions, likely forming ionic or coordination bonds. The Cu likely has a +2 oxidation state, giving it a d9 configuration. Considering its probable crystal structure and electronic arrangement, RbCuCl3 is predicted to be an indirect bandgap semiconductor due to the ionic bonding and the potential for indirect electron transitions in its structure. { "Direct_or_indirect_HSE": "indirect"} </think>

###### Answer:

{Direct_or_indirect_HSE : Indirect}

###### Reference:

Indirect

- Figure 17 | SNUMAT Classification Example. This figure demonstrates the model’s reasoning process for predicting the bandgap type of the material RbCuCl3. The thinking content highlights a logical chain of deductions: from analyzing the constituent ions and chemical bonding, to identifying the crucial +2 oxidation state and resulting d9 electronic configuration of the Cu ion.

###### 4.8.2. DNA Task: Promoter-300

- As shown in Fig. 18, the model is tasked with assessing the promoter potential of a 300 bp DNA sequence. The reasoning sequence meticulously evaluates several critical characteristics pertinent to promoter regions. It correctly identifies the sequence length as being within the typical range for promoters and notes the GC-rich content and the presence of CpG islands—both established indicators of promoter activity. While acknowledging the absence of an explicit TATA box, the model prudently suggests the potential for other transcription factor binding sites. Importantly, it recommends further bioinformatics analysis using specialized tools or databases, demonstrating a nuanced understanding of the empirical requirements for definitive promoter identification. This structured, multi-faceted analysis highlights the model’s ability to synthesize biological cues into a well-reasoned assessment, concluding with a biologically plausible interpretation.

4.8.3. Protein Task: Solubility

- As shown in Fig. 19, the Solubility prediction task exemplifies the model’s analytical prowess in protein biochemistry. Provided with a protein sequence, the model systematically dissects its characteristics to predict solubility. The reasoning primarily focuses on the high proportion of hydrophilic and charged amino acids (e.g., Arginine, Lysine, Glutamic acid, Aspartic acid), identifying them as key factors that promote solubility through favorable interactions with water. It further reinforces this conclusion by noting the absence of long hydrophobic regions or transmembrane domains, which are common drivers of aggregation and insolubility. This detailed consideration of amino acid composition and its implications for protein-solvent interactions underscores a sophisticated understanding of the

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 926]

Promoter-300

Input. <dna>TTCTTTTTCCGGATTTTCAGCCGGGTCTTCCGGGGATGGAGAGCAAAAGGACTTGG TGCTCTCGGAGAGAGCCTGCAGGGGGCGGGGTGTTGGTAGAAGGAGTGGGAGGGGTTTTACTGTAG CAAGGCGGGGCCTCAGTTGAGTGGGTGGGGCCTGGAGGCAACTAACGCGGCATAGGGCGAGGGCG GGGCCGGCAGGCCGGGGCGGGGCTGGGAAGTAGTCGGGCGGGGTTGTGAGACGCCGCGCTCAGCT TCCATCGCTGGGCGGTCAACAAGTGCGGGCCTGGCTCAGCGCGGGGGG</dna>: Could you assess the potential of this 300 bp DNA as a promoter region? Thinking Content:

<think> The 300 bp DNA sequence exhibits several features that suggest it might function as a promoter region, though definitive confirmation requires further analysis. Here’s a structured summary of the evaluation:

- 1. Length and Position: The sequence length of 300 bp is within the typical range for promoter regions, located upstream of the transcription start site.

- 2. GC Content: The sequence is GC-rich, which is characteristic of GC-rich promoters, potentially indicating a strong promoter element.

- 3. CpG Islands: The presence of multiple CG dinucleotides suggests a CpG island, commonly found in promoter regions, supporting the possibility of being a promoter.

- 4. Transcription Factor Binding Sites: While specific motifs like the TATA box are not evident, the sequence may contain other transcription factor binding sites that could drive gene expression.
- 5. Conservation Analysis: Without checking against known promoters, it’s difficult to assess conservation, but this could provide additional evidence.
- 6. Recommendation for Further Analysis: To confirm its potential as a promoter, it is recommended to use bioinformatics tools like PromoScanner or compare it with known promoters in databases. In conclusion, while the sequence shows features consistent with a promoter region, such as appropriate length, GC content, and CpG islands, further analysis with specialized tools is necessary for confirmation. </think>

###### Answer:

Promoter motifs are detectable in the given DNA fragment.

###### Reference:

The 300 bp DNA sequence includes a recognizable promoter region.

- Figure 18 | DNA Promoter Region Assessment Example. This figure demonstrates the model’s structured approach to evaluate a 300 bp DNA sequence for its potential as a promoter region. The thinking content meticulously breaks down the analysis into key biological features and provides a recommendation for further experimental validation.

biophysical principles governing protein solubility.

###### 4.9. Effectiveness of Pretraining

We ablate the contribution of pretraining on ten representative tasks that span gene-regulatory sequence understanding (promoter detection at 300bp and core promoters at 50–100 bp), transcription factor binding (binary classification), small-molecule property prediction (ESOL; regression), naturallanguage protein function generation, two sequence–sequence interaction tasks (antibody–antigen and RNA–protein; binary), alternative APA isoform usage (regression), GO prediction, and gene-name to cancer mapping. Metrics are Matthews correlation coefficient (MCC), ROUGE-L, precision, and R2 (“higher is better”), and RMSE for ESOL (“lower is better”).

Pretraining yields consistent gains on all tasks. Across the nine “higher is better” tasks, the median absolute improvement is +15.0 percentage points (pp) and the mean improvement is +22.0 pp. The ESOL regression error drops by 98.7% (RMSE 94.54→1.2194.54→1.21). The largest

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 948]

Solubility

Input. <protein>MDAQTIAPGFESVAELFGRFLSEDREYSAQLAAYHRGVKVLDISGGPHRRPDSVTG VFSCSKGVSGLVIALLVQDGFLDLDAEVVKYWPEFGAEGKATITVAQLLSHQAGLLGVEGGLTLAEYNNS ELAAAKLAQMRPLWKPGTAFGYHALTIGVFMEELCRRITGSTLQEIYEQRIRSVTGAHFFLGLPESEEPRY ATLRWAADPSQPWIDPASHFGLSANSAVGDILDLPNLREVRAAGLSSAAGVASAEGMARVYAAALTGLA ANGDRAAVAPLLSEETIQTVTAEQVFGIDRVFGETSCFGTVFMKSHARSPYGSYRAFGHDGASASLGFA DPVYELAFGYVPQQAEPGGAGCRNLELSAAVRKAVTELAQ</protein> [Solubility Prediction Binary Classification] Will the protein generated from this sequence be soluble? Thinking Content:

<think> Based on the analysis of the provided protein sequence, the protein is likely soluble. Here’s a concise summary of the reasoning:

- 1. Amino Acid Composition: The sequence contains a significant number of hydrophilic and charged amino acids such as Arginine (R), Lysine (K), Glutamic acid (E), and Aspartic acid (D). These residues promote solubility by interacting with water.

- 2. Absence of Long Hydrophobic Regions: While there are some hydrophobic residues, they do not form extensive stretches that could lead to aggregation or membrane insertion, which might reduce solubility.

- 3. No Obvious Transmembrane Domains: The sequence lacks regions indicative of transmembrane domains, which are typically hydrophobic and contribute to insolubility.

- 4. Presence of Charged Residues: The abundance of charged residues suggests potential for ionic interactions with water, enhancing solubility.

- 5. Aromatic Residues: While aromatic residues can sometimes reduce solubility, their distribution in this sequence does not appear to be a significant issue. Given these factors, the protein is predicted to be soluble. Conclusion: The protein is likely soluble based on its composition and sequence analysis. </think>

###### Answer:

The solubility task forecasts a protein’s ability to dissolve in solution. This protein is highly soluble.

###### Reference:

This task evaluates if a protein will dissolve in a solution. Yes, this protein is soluble.

- Figure 19 | Protein Solubility Prediction Example. This figure illustrates the model’s reasoning for predicting protein solubility. The thinking content details an analysis of key factors, such as the high proportion of hydrophilic and charged amino acids and the absence of long hydrophobic regions or transmembrane domains, which collectively lead to the prediction of a soluble protein.

absolute gains appear on sequence–sequence interaction and regulatory tasks (APA, RNA–protein, antibody–antigen), indicating that pretraining most strongly enhances representations that require long-range dependencies, cross-sequence compatibility, and motif composition. Promoter detection at 300bp benefits more than core-promoter detection at 50–100bp, consistent with the hypothesis that broader context amplifies the value of pretraining. Language-centric tasks (function generation; structured GO terms; gene-name→cancer) also improve steadily, showing transfer from pretrained biochemical/biomedical knowledge to text- and ontology-grounded outputs.

Tab. 9 summarizes per-task results. For proportion-like metrics reported in percent in the raw logs, we rescale to [0,1][0,1]. Absolute deltas are expressed in pp for those metrics and in raw units for RMSE; relative deltas follow the usual convention of (after-before)/before(after-before)/before for higher-is-better metrics and (before-after)/before(before-after)/before for RMSE.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 970]

Task Metric Before After Δ (pp / RMSE) Relative Δ Promoter Detection MCC 0.451 0.719 26.8 ↑59.4% Core Promoter Detection MCC 0.515 0.589 7.4 ↑14.4% TF-Human MCC 0.447 0.529 8.2 ↑18.3% ESOL RMSE ↓ 94.54 1.21 -93.33 ↓98.7% Protein Function Generation ROUGE-L 0.755 0.823 6.7 ↑8.9% Antibody–Antigen Interaction MCC 0.129 0.416 28.7 ↑221.9% RNA–Protein Interaction MCC 0.507 0.865 35.8 ↑70.6% APA Isoform Usage R2 0.311 0.879 56.8 ↑182.3% GO (CASP-SimilarSeq) Precision 0.690 0.840 15.0 ↑21.7% GeneName → Cancer Precision 0.610 0.740 13.0 ↑21.3%

- Table 9 | Ablation of pretraining on 10 tasks. Larger is better for MCC/ROUGE-L/Precision/R; lower is better for RMSE. Values are on the [0,1][0,1] scale after rescaling percent-formatted entries; “pp” denotes percentage points.

Notes. (i) Percent-like entries in the raw results (e.g., 12.92) were rescaled to proportion (e.g., 0.12920.1292). (ii) “pp” =100×(after-before)=100×(after-before) for non-RMSE metrics. (iii) Relative Δ is computed with respect to the “before” score; for RMSE we report relative reduction.

###### 5. Conclusion

We introduced a scientific language foundation model that bridges general-purpose large language modeling with the heterogeneous data and reasoning workflows of the natural sciences. Our framework couples multi-representation pre-training with instruction-driven alignment and a reasoning stage that supports both fast responses and step-by-step deliberation. By mapping natural language, DNA/RNA/protein sequences, molecular strings, and materials representations into a shared backbone via task-aware tokenization and consistent input–output schemas, the model moves beyond narrow, discipline-specific solutions and limited task menus. This design enables a single system to support four core capability families central to scientific work—property prediction/classification, sequence generation and design under constraints, scientific translation across formats, and text/knowledge extraction—while grounding its outputs through retrieval and tool execution for improved fidelity and verifiability.

###### References

- [1] Jason Wei et al. “Finetuned language models are zero-shot learners”. In: arXiv preprint arXiv:2109.01652 (2021).
- [2] Tom Brown et al. “Language models are few-shot learners”. In: Advances in neural information processing systems 33 (2020), pp. 1877–1901.
- [3] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. 2023.
- [4] Maxime Oquab et al. “Dinov2: Learning robust visual features without supervision”. In: arXiv preprint arXiv:2304.07193 (2023).
- [5] Jing Shao et al. “Intern: A new learning paradigm towards general vision”. In: arXiv preprint arXiv:2111.08687 (2021).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 992]

- [6] Shixiang Tang et al. “Humanbench: Towards general human-centric perception with projector assisted pretraining”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, pp. 21970–21982.
- [7] Yuanzheng Ci et al. “Unihcp: A unified model for human-centric perceptions”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023, pp. 17840–17852.
- [8] Yizhou Wang et al. “Hulk: A universal knowledge translator for human-centric tasks”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).
- [9] Shansong Liu et al. “Mumu-llama: Multi-modal music understanding and generation via large language models”. In: arXiv preprint arXiv:2412.06660 3.5 (2024), p. 6.
- [10] Alexandre Défossez et al. “Moshi: a speech-text foundation model for real-time dialogue”. In: arXiv preprint arXiv:2410.00037 (2024).
- [11] Qingkai Fang et al. “Llama-omni: Seamless speech interaction with large language models, 2024”. In: URL https://arxiv. org/abs/2409.06666 ().
- [12] Zalán Borsos et al. “Audiolm: a language modeling approach to audio generation”. In: IEEE/ACM transactions on audio, speech, and language processing 31 (2023), pp. 2523–2533.
- [13] Jean-Baptiste Alayrac et al. “Flamingo: a visual language model for few-shot learning”. In: Advances in neural information processing systems 35 (2022), pp. 23716–23736.
- [14] Peng Wang et al. “Ofa: Unifying architectures, tasks, and modalities through a simple sequenceto-sequence learning framework”. In: International conference on machine learning. PMLR. 2022, pp. 23318–23340.
- [15] Jaemin Cho et al. “Unifying vision-and-language tasks via text generation”. In: International Conference on Machine Learning. PMLR. 2021, pp. 1931–1942.
- [16] Ming Hu et al. “Ophclip: Hierarchical retrieval-augmented learning for ophthalmic surgical video-language pretraining”. In: arXiv preprint arXiv:2411.15421 (2024).
- [17] OpenAI. Introducing GPT-5. Accessed: 2025-08-12. 2025. url: https://openai.com/ index/introducing-gpt-5/.
- [18] An Yang et al. “Qwen3 technical report”. In: arXiv preprint arXiv:2505.09388 (2025).
- [19] Meta AI. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. Accessed: 2025-08-12. 2025. url: https://ai.meta.com/blog/llama-4-multimodalintelligence/.
- [20] Aixin Liu et al. “Deepseek-v3 technical report”. In: arXiv preprint arXiv:2412.19437 (2024).
- [21] Google DeepMind. Gemini 2.5: Our most intelligent AI model. Last updated March 26, 2025. Accessed: 2025-08-12. 2025. url: https://blog.google/technology/googledeepmind/gemini-model-thinking-updates-march-2025/.
- [22] Kai Han et al. “A survey on vision transformer”. In: IEEE transactions on pattern analysis and machine intelligence 45.1 (2022), pp. 87–110.
- [23] Alexey Dosovitskiy et al. “An image is worth 16x16 words: Transformers for image recognition at scale”. In: arXiv preprint arXiv:2010.11929 (2020).
- [24] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing systems 30 (2017).
- [25] Jared Kaplan et al. “Scaling laws for neural language models”. In: arXiv preprint arXiv:2001.08361 (2020).
- [26] Baolin Peng et al. “Instruction tuning with gpt-4”. In: arXiv preprint arXiv:2304.03277 (2023).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1014]

- [27] Haotian Liu et al. “Visual instruction tuning”. In: Advances in neural information processing systems 36 (2023), pp. 34892–34916.
- [28] Wenliang Dai et al. “Instructblip: Towards general-purpose vision-language models with instruction tuning”. In: Advances in neural information processing systems 36 (2023), pp. 49250– 49267.
- [29] Jason Wei et al. “Chain-of-thought prompting elicits reasoning in large language models”. In: Advances in neural information processing systems 35 (2022), pp. 24824–24837.
- [30] Shunyu Yao et al. “Tree of thoughts: Deliberate problem solving with large language models”. In: Advances in neural information processing systems 36 (2023), pp. 11809–11822.
- [31] Maciej Besta et al. “Graph of thoughts: Solving elaborate problems with large language models”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 38. 16. 2024, pp. 17682–17690.
- [32] Tianyi Zhang et al. “Benchmarking large language models for news summarization”. In: Transactions of the Association for Computational Linguistics 12 (2024), pp. 39–57.
- [33] Liyan Tang et al. “Evaluating large language models on medical evidence summarization”. In: NPJ digital medicine 6.1 (2023), p. 158.
- [34] Yang Zhang et al. “A comprehensive survey on process-oriented automatic text summarization with exploration of llm-based methods”. In: arXiv preprint arXiv:2403.02901 (2024).
- [35] Ehsan Hosseini-Asl et al. “A simple language model for task-oriented dialogue”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 20179–20191.
- [36] Hongru Wang et al. “A survey of the evolution of language model-based dialogue systems”. In: arXiv preprint arXiv:2311.16789 (2023).
- [37] Tao Gong et al. “Multimodal-gpt: A vision and language model for dialogue with humans”. In: arXiv preprint arXiv:2305.04790 (2023).
- [38] Ming Hu et al. “A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers”. In: arXiv preprint arXiv:2508.21148 (2025).
- [39] Jiaqi Wei et al. “From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery”. In: arXiv preprint arXiv:2508.14111 (2025).
- [40] Lei Bai et al. “Intern-s1: A scientific multimodal foundation model”. In: arXiv preprint arXiv:2508.15763 (2025).
- [41] Tianbin Li et al. “Gmai-vl & gmai-vl-5.5 m: A large vision-language model and a comprehensive multimodal dataset towards general medical ai”. In: arXiv preprint arXiv:2411.14522 (2024).
- [42] Jin Ye et al. “Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 94327–94427.
- [43] Yutao Hu et al. “OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024, pp. 22170–22183.
- [44] Chenglong Ma et al. “MedITok: A Unified Tokenizer for Medical Image Synthesis and Interpretation”. In: arXiv preprint arXiv:2505.19225 (2025).
- [45] Josh Abramson et al. “Accurate structure prediction of biomolecular interactions with AlphaFold 3”. In: Nature 630.8016 (2024), pp. 493–500. doi: 10.1038/s41586-024-07487w.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1036]

- [46] Jonathan M Stokes et al. “A deep learning approach to antibiotic discovery”. In: Cell 180.4

(2020), pp. 688–702.

- [47] Amil Merchant et al. “Scaling deep learning for materials discovery”. In: Nature 624.7990

(2023), pp. 80–85.

- [48] David Weininger. “SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules”. In: Journal of chemical information and computer sciences 28.1 (1988), pp. 31–36.
- [49] Mario Krenn et al. “Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation”. In: Machine Learning: Science and Technology 1.4 (2020), p. 045024.
- [50] William R Pearson and David J Lipman. “Improved tools for biological sequence comparison.” In: Proceedings of the National Academy of Sciences 85.8 (1988), pp. 2444–2448.
- [51] W Patrick Walters and Regina Barzilay. “Applications of deep learning in molecule generation and molecular property prediction”. In: Accounts of chemical research 54.2 (2020), pp. 263– 270.
- [52] Jianbo Qiao et al. “A self-conformation-aware pre-training framework for molecular property prediction with substructure interpretability”. In: Nature Communications 16.1 (2025), p. 4382.
- [53] Longlong Li et al. “Kolmogorov–Arnold graph neural networks for molecular property prediction”. In: Nature Machine Intelligence (2025), pp. 1–9.
- [54] Yizhen Zheng et al. “Large language models for scientific discovery in molecular property prediction”. In: Nature Machine Intelligence (2025), pp. 1–11.
- [55] Kevin M Ulmer. “Protein engineering”. In: Science 219.4585 (1983), pp. 666–671.
- [56] Romas J Kazlauskas and Uwe T Bornscheuer. “Finding better protein engineering strategies”. In: Nature chemical biology 5.8 (2009), pp. 526–529.
- [57] Minkyung Baek et al. “Accurate prediction of protein structures and interactions using a three-track neural network”. In: Science 373.6557 (2021), pp. 871–876.
- [58] Mingchen Li et al. “Harnessing A Unified Multi-modal Sequence Modeling to unveil ProteinDNA Interdependency”. In: bioRxiv (2025), pp. 2025–02.
- [59] Muhammad Shaban et al. “A Foundation Model for Spatial Proteomics”. In: arXiv preprint arXiv:2506.03373 (2025).
- [60] Abdenour Hadid, Tanujit Chakraborty, and Daniel Busby. “When geoscience meets generative AI and large language models: Foundations, trends, and future challenges”. In: Expert Systems 41.10 (2024), e13654.
- [61] Zhengyang Lin et al. “Large language models reveal big disparities in current wildfire research”. In: Communications Earth & Environment 5.1 (2024), p. 168.
- [62] Yuhao Zhou et al. “Scientists’ First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning”. In: arXiv preprint arXiv:2506.10521 (2025).
- [63] Xuming He et al. “RadarQA: Multi-modal Quality Analysis of Weather Radar Forecasts”. In: arXiv preprint arXiv:2508.12291 (2025).
- [64] Fengxiang Wang et al. “OmniEarth-Bench: Towards Holistic Evaluation of Earth’s Six Spheres and Cross-Spheres Interactions with Multimodal Observational Earth Data”. In: arXiv preprint arXiv:2505.23522 (2025).
- [65] Qizhi Pei et al. “Biot5: Enriching cross-modal integration in biology with chemical knowledge and natural language associations”. In: arXiv preprint arXiv:2310.07276 (2023).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1058]

- [66] Qizhi Pei et al. “Biot5+: Towards generalized biological understanding with iupac integration and multi-task tuning”. In: arXiv preprint arXiv:2402.17810 (2024).
- [67] Di Zhang et al. “Chemllm: A chemical large language model”. In: arXiv preprint arXiv:2402.06852 (2024).
- [68] Junxian Li et al. “Chemvlm: Exploring the power of multimodal large language models in chemistry area”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. 1. 2025, pp. 415–423.
- [69] Qian Tan et al. “ChemMLLM: Chemical Multimodal Large Language Model”. In: arXiv preprint arXiv:2505.16326 (2025).
- [70] Lei Jiang et al. “Chem3DLLM: 3D Multimodal Large Language Models for Chemistry”. In: arXiv preprint arXiv:2508.10696 (2025).
- [71] Liuzhenghao Lv et al. “Prollama: A protein large language model for multi-task protein language processing”. In: IEEE Transactions on Artificial Intelligence (2025).
- [72] Adibvafa Fallahpour et al. “BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model”. In: arXiv preprint arXiv:2505.23579 (2025).
- [73] Colin Raffel et al. “Exploring the limits of transfer learning with a unified text-to-text transformer”. In: Journal of machine learning research 21.140 (2020), pp. 1–67.
- [74] Hugo Touvron et al. “Llama 2: Open foundation and fine-tuned chat models”. In: arXiv preprint arXiv:2307.09288 (2023).
- [75] Wei Li et al. “Ophora: A large-scale data-driven text-guided ophthalmic surgical video generation model”. In: arXiv preprint arXiv:2505.07449 (2025).
- [76] Ying Chen et al. “Slidechat: A large vision-language assistant for whole-slide pathology image understanding”. In: Proceedings of the Computer Vision and Pattern Recognition Conference. 2025, pp. 5134–5143.
- [77] Yanzhou Su et al. “Gmai-vl-r1: Harnessing reinforcement learning for multimodal medical reasoning”. In: arXiv preprint arXiv:2504.01886 (2025).
- [78] Anubhav Jain et al. “Commentary: The Materials Project: A materials genome approach to accelerating materials innovation”. In: APL materials 1.1 (2013).
- [79] Helen M Berman et al. “The protein data bank”. In: Nucleic acids research 28.1 (2000), pp. 235–242.
- [80] Kim Pruitt et al. “The reference sequence (RefSeq) database”. In: The NCBI handbook 2

(2012).

- [81] Yin Fang et al. “Mol-instructions: A large-scale biomolecular instruction dataset for large language models”. In: arXiv preprint arXiv:2306.08018 (2023).
- [82] Yingce Xia et al. “Naturelm: Deciphering the language of nature for scientific discovery”. In: arXiv e-prints (2025), arXiv–2502.
- [83] Gongbo Zhang et al. “Unigenx: Unified generation of sequence and structure with autoregressive diffusion”. In: arXiv preprint arXiv:2503.06687 (2025).
- [84] Ross Taylor et al. “Galactica: A large language model for science”. In: arXiv preprint arXiv:2211.09085 (2022).
- [85] Haonan He et al. “Biology Instructions: A Dataset and Benchmark for Multi-Omics Sequence Understanding Capability of Large Language Models”. In: arXiv preprint arXiv:2412.19191

(2024).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1080]

- [86] Chaoqi Liang et al. “Rethinking the bert-like pretraining for dna sequences”. In: arXiv preprint arXiv:2310.07644 1 (2023).
- [87] Peng Ye et al. “Genomics-FM: Universal Foundation Model for Versatile and Data-Efficient Functional Genomic Analysis”. In: bioRxiv (2024), pp. 2024–07.
- [88] Chen Tang et al. “UniSTD: Towards Unified Spatio-Temporal Learning across Diverse Disciplines”. In: Proceedings of the Computer Vision and Pattern Recognition Conference. 2025, pp. 29213–29224.
- [89] Chaoqi Liang et al. “scWGBS-GPT: A Foundation Model for Capturing Long-Range CpG Dependencies in Single-Cell Whole-Genome Bisulfite Sequencing to Enhance Epigenetic Analysis”. In: bioRxiv (2025), pp. 2025–02.
- [90] Lifeng Qiao et al. “Model decides how to tokenize: Adaptive dna sequence tokenization with mxdna”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 66080–66107.
- [91] Hao Du et al. “Split and Merge Proxy: pre-training protein inter-chain contact prediction by mining rich information from monomer data”. In: (2023).
- [92] National Library of Medicine (NLM). PubMed. Launched 1996; accessed: 2025-09-12. 1996. url: https://pubmed.ncbi.nlm.nih.gov/.
- [93] National Center for Biotechnology Information (NCBI). PubChem. Launched 2004; accessed: 2025-09-12. 2004. url: https://pubchem.ncbi.nlm.nih.gov/.
- [94] National Center for Biotechnology Information. NCBI Home. Accessed: 2025-08-12. 2025. url: https://www.ncbi.nlm.nih.gov/.
- [95] RNAcentral Consortium. RNAcentral: A comprehensive database of non-coding RNA sequences. Launch announced September 10, 2014; accessed: 2025-08-12. 2014. url: https:// rnacentral.org/.
- [96] RDKit Team. RDKit: Open-Source Cheminformatics Software. Accessed: 2025-09-06. url: https://www.rdkit.org/.
- [97] Materials Project Team. The Materials Project. Launch announced 2011; accessed: 2025-09-06.

2011. url: https://next-gen.materialsproject.org/.

- [98] SNUMAT Team. SNU material data center. Accessed: 2025-09-06. url: https://www. snumat.com/.
- [99] Nadine Schneider, Nikolaus Stiefl, and Gregory A Landrum. “What’s what: The (nearly) definitive guide to reaction role assignment”. In: Journal of chemical information and modeling 56.12 (2016), pp. 2336–2346.
- [100] Botao Yu et al. “Llasmol: Advancing large language models for chemistry with a large-scale, comprehensive, high-quality instruction tuning dataset”. In: arXiv preprint arXiv:2402.09391

(2024).

- [101] Minghao Xu et al. “Peer: a comprehensive and multi-task benchmark for protein sequence understanding”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 35156– 35173.
- [102] Hongwang Xiao et al. “OPI: An Open Instruction Dataset for Adapting Large Language Models to Protein-Related Tasks”. In: Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges.
- [103] Zhihan Zhou et al. “Dnabert-2: Efficient foundation model and benchmark for multi-species genome”. In: arXiv preprint arXiv:2306.15006 (2023).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1102]

- [104] Yuchen Ren et al. “Beacon: Benchmark for comprehensive rna tasks and language models”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 92891–92921.
- [105] Andre Niyongabo Rubungo et al. “LLM4Mat-bench: benchmarking large language models for materials property prediction”. In: Machine Learning: Science and Technology 6.2 (2025), p. 020501.
- [106] Huu Nguyen et al. The Open Instruction Generalist (OIG) Dataset. First published March 10, 2023 (LAION blog); Hugging Face version updated March 20, 2023; accessed August 12,

2025. 2023. url: https://huggingface.co/datasets/laion/OIG.

- [107] Pin-Lun Hsu et al. “Liger kernel: Efficient triton kernels for llm training”. In: arXiv preprint arXiv:2410.10989 (2024).
- [108] Jeff Rasley et al. “Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters”. In: Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020, pp. 3505–3506.
- [109] Tri Dao. “Flashattention-2: Faster attention with better parallelism and work partitioning”. In: arXiv preprint arXiv:2307.08691 (2023).
- [110] Qiying Yu et al. “Dapo: An open-source llm reinforcement learning system at scale”. In: arXiv preprint arXiv:2503.14476 (2025).
- [111] A Jain et al. The Materials Project: A Materials Genome Approach to Accelerating Materials Innovation. APL Materials, 1 (1): 011002, 2013. 2013.
- [112] Kevin F Garrity and Kamal Choudhary. “Fast and accurate prediction of material properties with three-body tight-binding model for the periodic table”. In: Physical review materials 7.4

(2023), p. 044603.

- [113] Kamal Choudhary et al. “High-throughput identification and characterization of twodimensional materials using density functional theory”. In: Scientific reports 7.1 (2017), p. 5179.
- [114] Kamal Choudhary et al. “Computational screening of high-performance optoelectronic materials using OptB88vdW and TB-mBJ formalisms”. In: Scientific data 5.1 (2018), pp. 1– 12.
- [115] Kangming Li et al. “Efficient first principles based modeling via machine learning: from simple representations to high entropy materials”. In: Journal of Materials Chemistry A 12.21 (2024), pp. 12412–12422.
- [116] Andrew S Rosen et al. “Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery”. In: Matter 4.5 (2021), pp. 1578–1597.
- [117] Andrew S Rosen et al. “High-throughput predictions of metal–organic framework electronic properties: theoretical challenges, graph neural networks, and data exploration”. In: npj Computational Materials 8.1 (2022), p. 112.
- [118] Scott Kirklin et al. “The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies”. In: npj Computational Materials 1.1 (2015), pp. 1–15.
- [119] Christopher E Wilmer et al. “Large-scale screening of hypothetical metal–organic frameworks”. In: Nature chemistry 4.2 (2012), pp. 83–89.
- [120] Amir Shanehsazzadeh, David Belanger, and David Dohan. “Is transfer learning necessary for protein landscape prediction?” In: arXiv preprint arXiv:2011.03443 (2020).
- [121] Emanuele Rossi et al. “ncRNA classification with graph convolutional networks”. In: arXiv preprint arXiv:1905.06515 (2019).

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1124]

- [122] Joshua Meier et al. “Language models enable zero-shot prediction of the effects of mutations on protein function”. In: Advances in neural information processing systems 34 (2021), pp. 29287– 29303.
- [123] Bernardo P de Almeida et al. “DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers”. In: Nature genetics 54.5 (2022), pp. 613–624.
- [124] Nicholas Bogard et al. “A deep neural network for predicting and engineering alternative polyadenylation”. In: Cell 178.1 (2019), pp. 91–106.
- [125] Paul J Sample et al. “Human 5’ UTR design and variant effect prediction from a massively parallel translation assay”. In: Nature biotechnology 37.7 (2019), pp. 803–809.
- [126] Nicolaas M Angenent-Mari et al. “A deep learning approach to programmable RNA switches”. In: Nature communications 11.1 (2020), p. 5057.
- [127] Han Xu et al. “Sequence determinants of improved CRISPR sgRNA design”. In: Genome research 25.8 (2015), pp. 1147–1157.
- [128] Yuchen Ren et al. “COMET: Benchmark for Comprehensive Biological Multi-omics Evaluation Tasks and Language Models”. In: arXiv preprint arXiv:2412.10347 (2024).
- [129] Sameer Khurana et al. “DeepSol: a deep learning framework for sequence-based protein solubility prediction”. In: Bioinformatics 34.15 (2018), pp. 2605–2613.
- [130] Jie Zhang et al. “Predicting unseen antibodies’ neutralizability via adaptive graph neural networks”. In: Nature Machine Intelligence 4.11 (2022), pp. 964–976.
- [131] Yong Han and Shao-Wu Zhang. “ncRPI-LGAT: prediction of ncRNA-protein interactions with line graph attention network framework”. In: Computational and Structural Biotechnology Journal 21 (2023), pp. 2286–2295.
- [132] Xiaoping Min et al. “Predicting enhancer-promoter interactions by deep learning and matching heuristic”. In: Briefings in Bioinformatics 22.4 (2021).
- [133] Liang He et al. “Sfm-protein: Integrative co-evolutionary pre-training for advanced protein sequence representation”. In: arXiv preprint arXiv:2410.24022 (2024).
- [134] Jin Su et al. “Saprot: Protein language modeling with structure-aware vocabulary”. In: BioRxiv

(2023), pp. 2023–10.

- [135] Zitao Song et al. “Attention-based multi-label neural networks for integrated prediction and interpretation of twelve widely occurring RNA modifications”. In: Nature communications 12.1 (2021), p. 4011.
- [136] Tianhao Yu et al. “Enzyme function prediction using contrastive learning”. In: Science 379.6639

(2023), pp. 1358–1363.

- [137] He Cao et al. “InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery”. In: Proceedings of the 31st International Conference on Computational Linguistics. 2025, pp. 354–379.
- [138] Yuqiang Han et al. “Retrosynthesis prediction with an iterative string editing model”. In: Nature Communications 15.1 (2024), p. 6404.
- [139] Carl Edwards et al. “Translation between Molecules and Natural Language”. In: 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022. 2022.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1146]

Appendix

[Figure 1147]

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

• ta

－

－

＝＝

＝三

璧 孛

n 巳

[Figure 1148]

， 零

fi

：：

a1巴巴nc产L b气

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

Specialist LLMs

# 孛

Task Metric

SciReasoner-1.7B SciReasoner-8B Molecular Representation Translation

t0

严

皇

[Figure 1150]

[Figure 1151]

Top1

SMILES to IUPAC

Split Match ↑ 4.70%† 1.50%† 2.20% 0.17% — 29.00%(LlaSMol-7B [100]) 49.52%|50.20%† 56.63%|57.10%† SMILES to Formula

昙2se 5

Top1

Element Match ↑ 55.20%† 64.81%† 34.00% 19.88% — 93.20%(LlaSMol-7B [100]) 95.92%|87.1%† 96.39%|96.00%† IUPAC to SMILES

[Figure 1153]

[Figure 1154]

mv:rsl?

Top1

Split Match ↑ 38.20%† 21.82%† 26.9% 7.58% — 70.10%(LlaSMol-7B [100]) 78.85%|86.50%† 84.40%|90.70%† IUPAC to Formula

。

fH gon

Top1

Element Match ↑ 88.50%† 76.50%† 68.80% 33.24% — 87.90%(LlaSMol-7B [100]) 91.81%|88.10%† 92.65%|93.00%†

[Figure 1156]

k

gon

Natural Language Translation of Molecule Molecular Description ROUGE-L ↑ 0.23 0.07 0.06 0.05 0.75(BioT5+ [66]) 0.29(Mol-Instructions-7B [81]) 0.75 0.78 Molecular Captioning MENTOR ↑ 0.42† 0.28† 0.25 0.15 — 0.45(LlaSMol-7B [100]) 0.59|0.59† 0.60|0.58†

#### 零

5HY酶

[Figure 1160]

[Figure 1161]

###### Protein Functional Translation

CASPSimilarSeq ROUGE-L↑ 0.01 0.01 0.02 0.02 — 0.74 (OPI-Galactica-6.7B [102]) 0.83 0.85 IDFilterSeq ROUGE-L↑ 0.01 0.01 0.01 0.03 — 0.70 (OPI-Galactica-6.7B [102]) 0.77 0.82 UniProtSeq ROUGE-L↑ 0.01 0.02 0.01 0.03 — 0.71 (OPI-Galactica-6.7B [102]) 0.84 0.88 Mol-Instructions ROUGE-L↑ 0.21† 0.08† 0.05 0.07 0.68(BioT5+ [66]) 0.44(Mol-Instructions-7B [81]) 0.83|0.82† 0.99|0.98†

＠

[Figure 1162]

[Figure 1163]

- Table 10 | Evaluation results of scientific translation tasks. Bold indicates the best performance, and underline indicates the second best. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely large-scale test sets exist for certain tasks, which would lead to substantial testing costs.

[Figure 1169]

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

• ta

－

－

＝＝

＝三

璧 孛

n 巳

[Figure 1170]

， 零

fi

：：

a1巴巴nc产L b气

# 孛

t0

严

皇

[Figure 1172]

[Figure 1173]

昙2se 5

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

[Figure 1175]

Task Metric

Specialist LLMs SciReasoner-1.7B SciReasoner-8B Scientific Knowledge Extraction and QA

[Figure 1176]

mv:rsl?

。

fH gon

Chemical Entity Recognition F1 ↑ 0.71 0.66 0.05 0.55 — 0.75(Mol-Instructions-7B [81]) 0.81 0.92 Chemical–Protein Interaction Extraction F1 ↑ 0.12 0.06 0.05 0.05 — 0.22(Mol-Instructions-7B [81]) 0.46 0.83 Chemical–Disease Interaction Extraction F1 ↑ 0.35 0.33 0.18 0.23 — 0.40(Mol-Instructions-7B [81]) 0.66 0.97 General Multiple Choice Question ACC ↑ 0.94 0.71 0.26 0.87 — 0.65(Mol-Instructions-7B [81]) 0.92 0.99 True/False Question ACC ↑ 0.67 0.62 0.66 0.66 — 0.55(Mol-Instructions-7B [81]) 0.64 0.68 Open-Ended Question BertScore ↑ 0.84 0.82 0.79 0.80 — 0.84(Mol-Instructions-7B [81]) 0.83 0.87

[Figure 1178]

k

gon

#### 零

5HY酶

[Figure 1182]

[Figure 1183]

- Table 11 | Evaluation results of scientific knowledge extraction and QA tasks. Bold indicates the best performance, and underline indicates the second best.

＠

[Figure 1184]

[Figure 1185]

[Figure 1191]

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

• ta

－

－

＝＝

＝三

璧 孛

n 巳

[Figure 1192]

， 零

fi

：：

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

Specialist LLMs

Task Metric

SciReasoner-1.7B SciReasoner-8B Molecular Property Regression

a1巴巴nc产L b气

ESOL RMSE ↓ 1.28 1.58 1.48 3.45 — 1.04(LlaSMol-7B [100]) 1.21 1.08 LIPO RMSE ↓ 1.21 1.44 2.00 1.65 — 1.01(LlaSMol-7B [100]) 0.94 0.82 Physicochemical Prediction MAE ↓ 0.027 1.879 2.384 1.329 — 0.013(Mol-Instructions-7B [81]) 0.003 0.002

孛

t0

严

皇

[Figure 1194]

[Figure 1195]

###### Material Property Regression

1.32† 1.22† 0.66 0.95 5.32(MatBERT-109M [105]) 0.76(Gemma 2-9b-it:5S [105]) 4.98|4.67† 5.12|5.21† SNUMATRegression 1.07† 1.14† 1.12 1.00 1.83(MatBERT-109M [105]) 1.16(Llama 2-7b-chat:5S [105]) 2.18|2.18† 1.96|2.22† JARVISDFT 1.09† 1.2† 0.79 0.88 4.10(MatBERT-109M [105]) 0.70(Llama 2-7b-chat:5S [105]) 5.11|5.28† 5.21|5.59† JARVISQETB 0.53† 0.53† 0.35 0.50 59.44(LLM-Prop-35M [105]) 1.05(Llama 2-7b-chat:5S [105]) 138.59|148.71† 126.61|137.61† GNoME 1.28† 1.19† 0.84 0.73 15.60(LLM-Prop-35M [105]) 0.50(Llama 2-7b-chat:5S [105]) 19.05|18.46† 22.03|21.81† hMOF 0.65† 0.80† 0.61 0.67 1.48(LLM-Prop-35M [105]) 0.66(Llama 2-7b-chat:5S [105]) 1.05|1.04† 0.99|0.99† Cantor-HEA 0.77† 0.91† 0.53 0.45 8.40(LLM-Prop-35M [105]) 0.87(Llama 2-7b-chat:5S [105]) 7.40|7.38† 7.18|7.02† QMOF 1.24† 1.41† 0.51 0.86 2.05(MatBERT-109M [105]) 0.93(Llama 2-7b-chat:5S [105]) 8.55|8.76† 8.07|7.94† OQMD 1.20† 1.25† 0.77 0.65 6.02(LLM-Prop-35M [105]) 1.16(Llama 2-7b-chat:5S [105]) 5.29|5.50† 5.43|5.67† OMDB 0.82† 0.89† 0.88 0.93 1.55(MatBERT-109M [105]) 1.01(Llama 2-7b-chat:5S [105]) 1.40|1.41† 1.39|1.42†

MPRegression

昙2se 5

[Figure 1197]

[Figure 1198]

mv:rsl?

- MAD

- MAE ↑

。

fH gon

[Figure 1200]

k

gon

#### 零

5HY酶

[Figure 1204]

Biology Multi-omics Property Regression Fluorescence Spearman ↑ -0.43† 0.78† 0.72 3.20 69.00(ShallowCNN [120]) 2.57(ChatMultiOmics-8B [85]) 69.36|70.82† 66.24|68.49† StabilityBiology-Instructions Spearman ↑ -9.15† -7.96 -1.65 -7.79 79.00(Evoformer [121]) 60.25(ChatMultiOmics-8B [85]) 63.25|55.67† 64.41|59.03† Thermostability Spearman ↑ 5.47† 15.00 -3.65 6.29 78.00(ESM-1v [122]) 45.07(ChatMultiOmics-8B [85]) 56.25|53.17† 58.55|54.45† Enhancer Activity Prediction PCC ↑ -4.24† 0.49† -2.27 -1.29 68.00(DeepSTARR [123]) 57.24(ChatMultiOmics-8B [85]) 65.07|70.20† 64.39|52.08† APA Isoform Prediction R2 ↑ 0.13† 0.24† 0.03 0.01 50.82(APARENT [124]) 59.01(ChatMultiOmics-8B [85]) 87.94|86.19† 85.87|87.92† Mean Ribosome Loading Prediction R2 ↑ 0.00† 0.00† 0.06 0.02 78.00(Optimus [125]) 47.64(ChatMultiOmics-8B [85]) 50.82|54.56† 60.44|64.5† Programmable RNA Switches R2 ↑ 0.01† 0.15† 0.08 0.01 55.67(MLP-O [126]) 26.65(ChatMultiOmics-8B [85]) 39.00|38.1† 43.55|29.20† CRISPR On Target Prediction Spearman ↑ -3.64 -2.21 0.74 6.21 44.10(SCC [127]) 2.87(ChatMultiOmics-8B [85]) 14.35 25.81 siRNA Efficiency Prediction Mixed-score ↑ 38.75† 34.02 17.68 40.99 49.38(Meta-Biomol [128]) 42.92(ChatMultiOmics-8B [85]) 62.56|64.13† 59.67|61.94†

[Figure 1205]

＠

[Figure 1206]

[Figure 1207]

- Table 12 | Evaluation results of property prediction tasks. Bold indicates the best performance, and underline indicates the second best. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely large-scale test sets exist for certain tasks, which would lead to substantial testing costs.

[Figure 1213]

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

• ta

－

－

＝＝

＝三

璧 孛

n 巳

[Figure 1214]

， 零

fi

：：

a1巴巴nc产L b气

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

Specialist LLMs

Task Metric

SciReasoner-1.7B SciReasoner-8B

孛

Molecular Classification BBBP ACC@1 ↑ 71.57% 78.68% 59.90% 46.70% — 75.10%(LlaSMol-7B [100]) 79.70% 82.74% Clintox ACC@1 ↑ 26.39% 47.92% 70.14% 52.78% — 93.10%(LlaSMol-7B [100]) 92.36% 91.67% HIV ACC@1 ↑ 26.7%† 77.70%† 89.3% 69.86% — 96.70%(LlaSMol-7B [100]) 96.79%|92.40%† 96.81%|95.90%† Sider ACC@1 ↑ 67.7%† 70.50%† 61.2% 38.25% — 70.70%(LlaSMol-7B [100]) 70.00%|66.20%† 68.53%|68.70%†

t0

严

皇

[Figure 1216]

[Figure 1217]

昙2se 5

[Figure 1219]

###### Material Property Classification

[Figure 1220]

mv:rsl?

MPclassification AUC↑ 0.58† 0.56† 0.57 0.53 0.72(MatBERT-109M [105]) 0.51(Gemma 2-9b-it:5S [105]) 0.70|0.69† 0.72|0.71† SNUMATclassification AUC↑ 0.59† 0.63† 0.57 0.60 0.72(LLM-Prop-35M [105]) 0.58(Gemma 2-9b-it:5S [105]) 0.69|0.69 0.66|0.66

。

fH gon

[Figure 1222]

###### Gene-centric Association Classification

k

gon

Precision↑ 0.01 0.38 0.03 0.03 — 0.39(OPI-Galactica-6.7B [102]) 0.41 0.41

#### 零

gSymbol2Tissue

Recall↑ 0.01 0.68 0.09 0.09 — 0.91(OPI-Galactica-6.7B) [102] 0.82 0.83 F1 score↑ 0.01 0.45 0.04 0.04 — 0.53(OPI-Galactica-6.7B) [102] 0.51 0.52

5HY酶

[Figure 1226]

[Figure 1227]

Precision↑ 0.00 0.11 0.00 0.01 — 0.36(OPI-Galactica-6.7B [102]) 0.75 0.81

gSymbol2Cancer

Recall↑ 0.00 0.10 0.02 0.02 — 0.32(OPI-Galactica-6.7B [102]) 0.72 0.81 F1 score↑ 0.00 0.10 0.00 0.01 — 0.32(OPI-Galactica-6.7B [102]) 0.73 0.81

＠

Precision ↑ 0.00 0.11 0.00 0.01 — 0.27(OPI-Galactica-6.7B [102]) 0.68 0.80

[Figure 1228]

[Figure 1229]

gName2Cancer

Recall ↑ 0.00 0.10 0.03 0.04 — 0.26(OPI-Galactica-6.7B [102]) 0.67 0.79 F1 score ↑ 0.00 0.10 0.00 0.01 — 0.25(OPI-Galactica-6.7B [102]) 0.67 0.79

- Table 13 | Evaluation results of molecular classification tasks, material property classification tasks, and gene-centric association classification tasks. Bold indicates the best performance, and underline indicates the second best. † denotes that only randomly sampled 1000 test samples were utilized for evaluation in the corresponding task. This measure is adopted because extremely large-scale test sets exist for certain tasks, which would lead to substantial testing costs.

[Figure 1235]

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

SciReasoner 8B Biology Multi-omics Property Classification (binary)

Specialist LLMs

SciReasoner 1.7B

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

Task Metric

• ta

－

－

＝＝

＝三

SolubilityBiology-Instructions ACC ↑ 51.40† 50.83 51.82 51.22 77.00(DeepSol [129]) 63.02(ChatMultiOmics-8B [85]) 62.67|67.3† 71.61|69.4† Antibody-Antigen MCC ↑ −5.05† 1.32 -6.05 −2.78 54.90(DeepAAI [130]) 1.06(ChatMultiOmics-8B [85]) 41.59|45.43† 45.5|48.84† RNA-Protein MCC ↑ 1.20† −4.40 -15.97 −5.23 93.2(ncRPI-LGAT [131]) 74.26(ChatMultiOmics-8B [85]) 86.47|85.97† 84.97|88.8† Epigenetic Marks Prediction ↑ 0.52† 2.40† 3.23 0.84 58.83(DNABert-2 [103]) 3.64(ChatMultiOmics-8B [85]) 14.16|18.09† 17.24|21.26†

璧 孛

n 巳

[Figure 1236]

， 零

fi

：：

a1巴巴nc产L b气

- TF-Mouse0 MCC ↑ 0.20† 11.93† 4.70 4.75 64.23(DNABert-2 [103]) 26.50 (ChatMultiOmics-8B [85]) 41.79|39.77† 43.52|40.42†

- TF-Mouse1 MCC ↑ −1.24† −3.27† 1.95 −0.38 86.28(DNABert-2 [103]) 66.96 (ChatMultiOmics-8B [85]) 68.95|72.07† 71.03|75.98†

- TF-Mouse2 MCC ↑ 0.41† 5.97† 21.34 6.99 81.28(DNABert-2 [103]) 63.98 (ChatMultiOmics-8B [85]) 71.57|69.08† 75.07|68.52†

- TF-Mouse3 MCC ↑ 2.72† 3.00† 4.68 −9.63 73.49(DNABert-2 [103]) 32.02 (ChatMultiOmics-8B [85]) 44.81|45.10† 52.98|53.31†

- TF-Mouse4 MCC ↑ 6.61† 3.58† 1.44 −2.07 52.66(DNABert-2 [103]) 9.18 (ChatMultiOmics-8B [85]) 27.34|28.45† 28.62|28.20† Enhancer-Promoter Interaction MCC ↑ −3.75† −3.24 0 −1.69 53.59(EPI-DLMH [132]) 4.13(ChatMultiOmics-8B [85]) 8.18|6.75† 14.40|8.12†

孛

t0

严

皇

[Figure 1238]

[Figure 1239]

Promoter-300all MCC ↑ 24.12 13.04 17.37 -3.56 91.01(NT-2500M-multi [103]) 78.97(ChatMultiOmics-8B [85]) 82.94 83.01 Promoter-300notata MCC ↑ 16.65 20.31 16.60 -2.40 94.34(DNABert-2 [103]) 83.45(ChatMultiOmics-8B [85]) 88.46 87.04 Promoter-300tata MCC ↑ 11.98 15.12 12.60 13.04 79.43(NT-2500M-multi [103]) 38.50(ChatMultiOmics-8B [85]) 44.38 51.37 Promoter-coreall MCC ↑ 12.38 -3.18 -12.13 -3.57 70.92(DNABert 3-mer [103]) 59.98(ChatMultiOmics-8B [85]) 61.49 62.56 Promoter-corenotata MCC ↑ 15.64 18.43 -13.70 -6.37 71.58(NT-2500M-multi [103]) 62.85(ChatMultiOmics-8B [85]) 61.72 63.52 Promoter-coretata MCC ↑ 13.02 16.07 11.90 17.97 78.15(DNABert 3mer [103]) 39.36(ChatMultiOmics-8B [85]) 53.67 46.94

昙2se 5

[Figure 1241]

[Figure 1242]

mv:rsl?

。

fH gon

[Figure 1244]

- TF-Human0 MCC ↑ -3.46 2.41 5.97 3.24 71.99(DNABert-2 [103]) 56.56(ChatMultiOmics-8B [85]) 59.33 61.07

- TF-Human1 MCC ↑ 12.23 11.08 4.42 3.85 76.06(DNABert-2 [103]) 61.17(ChatMultiOmics-8B [85]) 62.22 63.47

- TF-Human2 MCC ↑ 2.29 7.20 0.61 5.19 66.52(DNABert-2 [103]) 46.92(ChatMultiOmics-8B [85]) 47.80 48.02

- TF-Human3 MCC ↑ -8.44 0.00 -0.40 -1.27 58.54(DNABert-2 [103]) 34.16(ChatMultiOmics-8B [85]) 38.96 40.89

- TF-Human4 MCC ↑ 7.56 15.66 7.61 15.06 77.43(DNABert-2 [103]) 51.86(ChatMultiOmics-8B [85]) 56.43 56.10 SolubilityPEER ACC ↑ 58.40† 53.0† 57.58 10.91 74.37(BioT5+ [66]) 71.40(NatureLM [82]) 70.14|70.8† 73.99|75.4† StabilityPEER ACC ↑ 49.60† 45.40† 50.63 38.21 58.30(SFM-Protein [133]) 63.50(NatureLM-8B [82]) 68.41|68.90† 64.70|64.3† Human PPI ACC ↑ 52.74 57.4 51.48 58.2 85.09(BioT5+ [66]) 84.8(NatureLM-8B [82]) 76.79 73.42 Yeast PPI ACC ↑ 51.52 50.00 56.60 44.90 66.16(BioT5+ [66]) 60.40(NatureLM-8B [82]) 53.81 60.41

k

gon

#### 零

5HY酶

[Figure 1248]

[Figure 1249]

＠

[Figure 1250]

[Figure 1251]

Biology Multi-omics Property Classification (multi-label) Protein Function ROUGE-L 0.20† 0.07† 0.05 0.05 0.57(BioT5+ [66]) 0.43(Mol-Instructions-7B [81]) 0.63|0.63 0.97|0.96 Domain/Motif ROUGE-L 0.19† 0.04† 0.03 0.02 0.53(BioT5+ [66]) 0.46(Mol-Instructions-7B [81]) 0.56|0.56 0.97|0.90 FunctionEC Fmax↑ 3.3† 4.49† 5.27 22.86 88.9(SaProt-GearNet [134]) 19.79(ChatMultiOmics-8B [85]) 60.65|60.33† 77.26|77.56† Non-coding RNA Func. Classif. ACC ↑ 17.8† 6.20† 10.31 11.54 85.73(GCN [121]) 63.09(ChatMultiOmics-8B [85]) 80.39|88.8† 86.65|93.4† Modification Prediction AUC ↑ 51.19 50.96 49.92 50.05 84.00(MultiRM [135]) 59.06(ChatMultiOmics-8B [85]) 53.06 59.94 Subcellular Localization Accuracy↑ 0.02 0.08 0.05 0 — 0.78(OPI-Galactica-6.7B [102]) 0.84 0.91 Fold Type Accuracy↑ 0.01 0.01 0.01 0 — 0.61(OPI-Llama-3.1-8B-Instruct) [102] 0.34 0.48

Precision↑ Recall↑ F1↑

0 0 0

0.01 0 0

0 0 0

0 0 0

0.59(CLEAN [136])

0.37(OPI-Llama-3.1-8B-Instruct) [102])

0.40 0.39 0.39

0.55 0.55 0.54

- 0.48(CLEAN [136])

- 0.49(CLEAN [136])

- 0.34(OPI-Llama-3.1-8B-Instruct) [102]
- 0.35(OPI-Llama-3.1-8B-Instruct) [102]

Numerical EC (New-392)

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0 0 0

0 0 0

0.58(CLEAN [136]) 0.47(CLEAN [136]) 0.49(CLEAN [136])

0.07(OPI-Llama-3.1-8B-Instruct) [102] 0.07(OPI-Llama-3.1-8B-Instruct) [102] 0.07(OPI-Llama-3.1-8B-Instruct) [102]

0.10 0.10 0.10

0.21 0.23 0.22

Numerical EC (Price-149)

Continued on next page

[Figure 1257]

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

Specialist LLMs

SciReasoner 1.7B

SciReasoner 8B

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

上悔人工智能实验室

上泭人工智能实验室

Task Metric

• ta

Precision↑ Recall↑ F1↑

0.04 0.03 0.03

0.21 0.19 0.19

0 0.04 0.01

0 0.02 0.01

-

0.81(OPI-Galactica-6.7B [102]) 0.74(OPI-Galactica-6.7B [102]) 0.76(OPI-Galactica-6.7B [102])

0.83 0.80 0.81

###### 0.86 0.83 0.84

－

－

＝＝

＝三

Keywords (CASPSimilarSeq)

璧 孛

n 巳

[Figure 1258]

Precision↑ Recall↑ F1↑

0.03 0.02 0.01

0.17 0.16 0.16

0.01 0 0

0 0.02 0.01

-

0.84(OPI-Galactica-6.7B [102]) 0.80(OPI-Galactica-6.7B [102]) 0.82(OPI-Galactica-6.7B [102])

0.86 0.85 0.85

###### 0.89 0.89 0.88

， 零

Keywords (IDFilterSeq)

fi

：：

a1巴巴nc产L b气

Precision↑ Recall↑ F1↑

0.01 0 0

0.2 0.18 0.18

0.01 0 0

0.01 0.03 0.01

-

0.86(OPI-Galactica-6.7B [102])

0.85 0.90 0.88

###### 0.92 0.91 0.91

- 0.82(OPI-Galactica-6.7B [102])
- 0.83(OPI-Galactica-6.7B [102])

Keywords (UniProtSeq)

孛

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0 0 0

0 0 0

-

0.76(OPI-Galactica-6.7B [102])

0.81 0.81 0.80

###### 0.86 0.86 0.86

t0

严

- 0.74(OPI-Galactica-6.7B [102])
- 0.75(OPI-Galactica-6.7B [102])

GO Terms (CASPSimilarSeq)

皇

[Figure 1260]

[Figure 1261]

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0 0 0

0 0 0

-

0.74(OPI-Galactica-6.7B [102]) 0.73(OPI-Galactica-6.7B [102]) 0.72(OPI-Galactica-6.7B [102])

0.78 0.77 0.77

###### 0.83 0.82 0.81

GO Terms (IDFilterSeq)

昙2se 5

Precision↑ Recall↑ F1↑

0 0 0

0 0 0

0 0 0

0 0 0

-

0.76(OPI-Galactica-6.7B [102]) 0.74(OPI-Galactica-6.7B [102]) 0.74(OPI-Galactica-6.7B [102])

0.84 0.82 0.81

0.88 0.87 0.86

[Figure 1263]

[Figure 1264]

GO Terms (UniProtSeq)

mv:rsl?

。

- Table 14 | Evaluation results of biology multi-omics property classification tasks. Dashed lines separate tasks that report multiple metrics (Precision/Recall/F1). Bold indicates the best performance, and underline indicates the second best.

fH gon

[Figure 1266]

k

gon

#### 零

5HY酶

[Figure 1270]

[Figure 1271]

＠

[Figure 1272]

[Figure 1273]

[Figure 1279]

Gemini 2.5-pro

GPT o3

DeepSeek r1

GPT oss

Non-LLM Expert Models

SciReasoner 8B Unconditional Generation

Specialist LLMs

SciReasoner 1.7B

歹二

三主

ShanghsiA巾JicialJniclligcacc!..a坎皿IOry

昙 =

荨主

Sha,ghaAmr,cnllmclhgenceIAboralOry

Sha01ghaiArtificialhttclligcnccLabornto乃＇

震主一

上消人工智能实验室

Task Metric

上悔人工智能实验室

上泭人工智能实验室

• ta

－

－

＝＝

＝三

Validity ↑ 0.568 0.715 0.698 0.952 — 0.968(NatureLM-8B [82]) 0.864 0.971 Unique ↑ 0.042 0.364 0.093 0.318 — 0.966(NatureLM-8B [82]) 0.988 0.982

Unconditional Molecular Generation

璧 孛

n 巳

[Figure 1280]

Avg. Length↑ 128.87 153.56 260.20 318.91 — 284.50(NatureLM-8B [82]) 280.10 284.7 Diversity↑ 0.62 0.99 0.93 1.00 — 0.97 (NatureLM-8B [82]) 1.00 0.91

， 零

Unconditional Protein Generation

fi

：：

Validity↑ 0.93 0.97 1.00 0.99 — — 0.80 0.99 Unconditional RNA Generation Avg. MFE↓ N/A -151.70 N/A -417.58 — -177.1(NatureLM-8x7B [82]) -178.85 -200.14 Unconditional Material Generation SMACT↑ 89.70 N/A N/A N/A — 66.07(NatureLM-8x7B [82]) 55.06 88.18

a1巴巴nc产L b气

###### Conditional Material Generation

孛

t0

SMACT ↑ N/A N/A N/A N/A — 83.36(NatureLM-8B [82]) 86.07 85.85 Precision ↑ N/A N/A N/A N/A — 98.44(NatureLM-8B [82]) 99.89 99.96

严

皇

Composition to Material Generation

[Figure 1282]

[Figure 1283]

Novelty ↓ N/A N/A N/A N/A — 97.13(NatureLM-1B [82]) 85.64 84.81 Bulk to Material Generation

SMACT ↑ N/A N/A N/A N/A — 94.75(NatureLM-8x7B [82]) 88.92 87.49 Success rate N/A N/A N/A N/A — — 7.12 7.05

昙2se 5

Conditional Small Molecular Generation Forward Synthesis Prediction Exact Match ↑ 0.32† 0.27† 0.10 0.33 — 0.63(LlaSMol-7B [100]) 0.69|0.81 0.75|0.83 Forward Reaction Prediction Exact Match ↑ 0.55 0.58 0.48 0.36 0.86(BioT5+ [66]) 0.54(InstructMol-GS-6.9B [137]) 0.96 0.99 Reagent Prediction Exact Match ↑ 0.04 0.01 0.01 0.00 0.26(BioT5+ [66]) 0.13(InstructMol-GS-6.9B [137]) 0.19 0.22 RetrosynthesisMol-Instructions Exact Match ↑ 0.20 0.16 0.05 0.05 0.64(BioT5+ [66]) 0.41(InstructMol-GS-6.9B [137]) 0.72 0.83 RetrosynthesisUSPTO-50K Exact Match ↑ 0.23† 0.07† 0.03 0.12 0.61(EditRetro [138]) 0.70(NatureLM-8B [82]) 0.68|0.72 0.72|0.74 RetrosynthesisSMol-Instruction Exact Match ↑ 0.12† 0.04† 0.01 0.08 — 0.33(LlaSMol-7B [100]) 0.41|0.43† 0.46|0.47† Molecule Generation Exact Match ↑ 0.25† 0.16† 0.19 0.03 — 0.19(LlaSMol-7B [100]) 0.39|0.34† 0.48|0.46† Description-Guided Molecule Design Exact Match ↑ 0.06 0.04 0.03 0.04 0.11(MolT5 [139]) 0.002(Mol-Instructions-7B [81]) 0.10 0.12

[Figure 1285]

[Figure 1286]

mv:rsl?

。

fH gon

[Figure 1288]

k

gon

#### 零

5HY酶

[Figure 1292]

[Figure 1293]

###### Conditional Biochemical Generation

Function-guided Protein Design Max. Normalized SW ↑ 0.00† 0.00† 0.00 0.00† 0.97(BioT5+ [66]) 0.43(Mol-Instructions-7B [81]) 0.96|0.90† 1.00|1.00† Catalytic Activity ROUGE-L ↑ 0.25† 0.09† 0.04 0.02 0.72(BioT5+ [66]) 0.52(Mol-Instructions-7B [81]) 0.77|0.78† 0.99|0.95†

＠

[Figure 1294]

[Figure 1295]

- Table 15 | Evaluation results of generation and design tasks. N/A indicates that the model cannot generate samples that meet the required specifications. Bold indicates the best performance, and underline indicates the second best.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1322]

###### Protein Generation

[Figure 1323]

[Figure 1324]

[Figure 1325]

- Figure 20 | Additional Visualization for Protein Generation task (a).

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

- Figure 21 | Additional Visualization for Protein Generation task (b).

Translation between SMILES and IUPAC

Create Date: 2025-07-28 SMILES: C[C@H](C(=O)N[C@H](CC1=CC=CC=C1)C(=O)C(=O)NCC2=CC=CC=N2)NC(=O)C3=CC(=C(C=C3)Cl)Cl Reference: 3,4-dichloro-N-[(2R)-1-[[(2R)-3,4-dioxo-1-phenyl-4-(pyridin-2-ylmethylamino)butan-2-yl]amino]-1-oxopropan-2-yl]benzamide

Our output: 3,4-dichloro-N-[(2R)-1-[[(2R)-3,4-dioxo-1-phenyl-4-(pyridin-2-ylmethylamino)butan-2-yl]amino]-1-oxopropan-2-yl]benzamide

GPT-5: (2S)-2-[[(2S)-2-benzyl-3-(pyridin-2-ylmethylcarbamoyl)propanoyl]amino]-N-(3,4-dichlorophenyl)propanamide Mol-Instruct: (2Z,5Z)-2-chloro-5-[2-[(2Z,5Z)-2-chloro-5-[(2-methyl-3-oxo-3-phenylpropanoyl)amino]penta-2,4-dienoyl]-3-oxo-3phenylpropanoyl]penta-2,4-dienoic acid Gemini2.5-Pro: (4S)-4-{(2S)-2-(3,4-dichlorobenzamido)propanamido}-4-phenyl-N-(pyridin-2-ylmethyl)-2,3-dioxobutanamide

[Figure 1330]

Ours

[Figure 1331]

GPT-5

- Figure 22 | Additional Visualization for SMILES-to-IUPAC Translation task

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1353]

###### Task Examples

|Example 1:<br><br>Prompt: How soluble is <SMILES> CC(C)Cl </SMILES> ? Response: Its log solubility is -1.41 mol/L.|
|---|

ESOL This task has 112 test samples.

|Example 1:<br><br>Prompt: Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <SMILES> NC(=O)C1=CC=CC=C1O </SMILES> . Response: 1.090|
|---|

LIPO This task has 420 test samples.

|Example 1:<br><br>Prompt: Is blood-brain barrier permeability (BBBP) a property of <SMILES> CCNC(=O)/C=C/C1=CC=CC(Br)=C1 </SMILES>? Response: Yes|
|---|

BBBP This task has 197 test samples.

|Example 1:<br><br>Prompt: Is <SMILES> COC[C@@H](NC(C)=O)C(=O)NCC1=CC=CC=C1 </SMILES> toxic? Response: No|
|---|

Clintox This task has 144 test samples.

|Example 1:<br><br>Prompt: Can <SMILES> CC1=CN(C2C=CCCC2O)C(=O)NC1=O </SMILES> serve as an inhibitor of HIV replication? Response: No|
|---|

HIV This task has 4107 test samples.

|Example 1:<br><br>Prompt: Are there any known side effects of <SMILES> CC1=CC(C)=C(NC(=O)CN(CC(=O )O)CC(=O)O)C(C)=C1Br </SMILES> affecting the heart? Response: No|
|---|

SIDER This task has 2860 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1375]

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: KPrMnNbO6property name: volume Response: {volume : 128.62405239223665}|
|---|

MPregression This task has 82241 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Na2Ti3Cl8property name: Band_gap_HSE Response: {Band_gap_HSE : 1.583856}|
|---|

SNUMATregression This task has 4152 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: YReW2O8property name: formation_energy_peratom Response: {formation_energy_peratom : -2.15925}|
|---|

JARVISDFT This task has 54647 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: BeMoRhproperty name: final_energy Response: {final_energy : -169.1866374093}|
|---|

JARVISQETB This task has 249596 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1397]

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Yb3Eu7(TlPd2)2property name: Density Response: {Density : 8.2778}|
|---|

GNoME This task has 216455 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Zn4C84N16H22O17property name: max_co2_adsp Response: {max_co2_adsp : 8.95639}|
|---|

hMOF This task has 92925 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Al2Co1Cr1Mn1Ni2Si1property name: e_above_hull Response: {e_above_hull : 0.4014893125}|
|---|

Cantor-HEA This task has 33608 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Zn12C48H88O80property name: lcd Response: {lcd : 3.40559}|
|---|

QMOF This task has 3064 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1419]

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: Sm2CuH6property name: e_form Response: {e_form : -0.3328793676}|
|---|

OQMD This task has 192799 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: C4H5(CH)3(CH2)3C5H5NCH3C3NH4H3(CO)2C4NH4SO2property name: bandgap Response: {bandgap : 2.886436}|
|---|

OMDB This task has 1213 test samples.

|Example 1:<br><br>Prompt: Please evaluate the following protein sequence and provide an explanation of the enzyme’s catalytic activity, including the chemical reaction it facilitates: <protein>MVKKRIAVIGAGISGLGAIKCCLDEDLEPTCFERNDDIGGLWKF QKNASEKMPSIYRSVTINTSKEMMCFSDFPIPDHFPNYMHNSKLMDYFRMYAKRFSLLDYIRFKTTVRSVRKRPDFHIH GQWDVVVETDGKQESLVFDGVLVCSGHHTDPHLPLKSFPGIEKFEGCYFHSREYKSPEDYVGKRIIVVGIGNSGVDIAV ELGRVAKQVFLSTRRGSWILHRVWNNGYPMDSSFFTRFHSFLQKILTTEAVNKYLEKTLNSRFNHAHYGLQPQHRPLSQ HPTISDDLPNHIISGKVQVKPNVKEFTGTDVHFDDGTVEENIDVVIFATGYSISFPFLGDLIAVTDNEVSLYKLMFPPD LEKPTLAVIGLIQPLGIILPIAELQSRWAVRVFKGLSKLPSVKAMKADMDQRKKAMEKRYVKTARHTIQVDHIEYMDEI ASLAGVKPNLLLLFLSDPTLAMEVFFGPCTPYQYRLQGPGKWDGARRAILTQRERIIKPLKTRITSEKSRSAPGLFWIK MALFGLAFLVPSLTYFSYICQ</protein> Response: Upon reviewing the provided protein sequence, the corresponding enzyme’s catalytic activity is identified as the following chemical reaction: H(+) + hexan-3-one + NADPH + O2 = ethyl butanoate + H2O + NADP(+).|
|---|

Catalytic Activity This task has 1601 test samples.

|Example 1:<br><br>Prompt: Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule? <SMILES> O=C1OCC2CC1C2 </SMILES> Response: 0.2756|
|---|

Physicochemical Prediction This task has 1999 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1441]

|Example 1:<br><br>Prompt: <protein>SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVP WPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVRFEGDTLVNRIELKGIDFKEDGNI LGHKLEYNYNSHNVYIMADKQKIGIKVNFRIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRD HMVLLEFVTAAGITHGMDELYK</protein> [Fluorescence Prediction Regression] What is the anticipated fitness of this green fluorescent protein mutant? Response: The fluorescence intensity for this GFP mutant is calculated as 3.67.|
|---|

Fluorescence This task has 27217 test samples.

|Example 1:<br><br>Prompt: <protein>TTIKVNGQEYTVPLSPEQFAKAAKKRWPDYEVQIHGNTVKVTR</protein> What is the expected stability score for this sequence? Response: The stability score determined for this sequence is 0.53.|
|---|

Stability This task has nan test samples.

|Example 1:<br><br>Prompt: <protein>MNYARFITAASAARNPSPIRTMTDILSRGPKSMISLAGGLPNPNMFPFKTAVITV ENGKTIQFGEEMMKRALQYSPSAGIPELLSWLKQLQIKLHNPPTIHYPPSQGQMDLCVTSGSQQGLCKVFEMIINPGDNV LLDEPAYSGTLQSLHPLGCNIINVASDESGIVPDSLRDILSRWKPEDAKNPQKNTPKFLYTVPNGNNPTGNSLTSERKKE IYELARKYDFLIIEDDPYYFLQFNKFRVPTFLSMDVDGRVIRADSFSKIISSGLRIGFLTGPKPLIERVILHIQVSTLHP STFNQLMISQLLHEWGEEGFMAHVDRVIDFYSNQKDAILAAADKWLTGLAEWHVPAAGMFLWIKVKGINDVKELIEEKAV KMGVLMLPGNAFYVDSSAPSPYLRASFSSASPEQMDVAFQVLAQLIKESL</protein> The task evaluates protein thermostability using mass spectrometry to measure melting points in human cells. How is the thermostability of this protein assessed? Response: The task uses mass spectrometry to measure protein melting curves and predict thermostability in human cells. The thermostability measure recorded for this protein is 56.77|
|---|

Thermostability This task has 1336 test samples.

|Example 1:<br><br>Prompt: <dna>TGTCGCTCCCATTTCGTCAAATGTTGCGTGCTAATTCGCTTGCCTTCTGGCGGCTATTT TTGTTTTGATATAATTTATATTTCTCGCTCTTTACTTATCTTTTTTGCTTTTTTTTTGTATTTTGTTTGCCGCTTACACT TCACATTTGCCTTCGGGGGCCGTCGCTTAAAAATAAACGCACACACACACTCGCGGCATTCACATATTTATTTTTGGTTT CTCCAACTTTGTACGCTCTGCGTTTTCTTC</dna> [Regression: Enhancer activity prediction] Please provide an enhancer activity prediction for this sequence. Response: The enhancer activity prediction yields: HK -0.68, Dev -1.26|
|---|

Enhancer Activity Prediction This task has 41186 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1463]

|Example 1:<br><br>Prompt: <rna>CTGCTTGGAGTTTACATTGAGGAAGCTGGTAACTGACCTTCAAAGCTAAAATATAAAAC TATTTGGGAAGTATGAAACGATGTCTCGTGATCTGGTGTACCCTTATCCCTGTGACGTTTGGCCTCTGACAATACTGGTA TAATTGTAAATAATGTCAAACTCCGTTTTCTAGCAAGTATTAAGGGA</rna> [Isoform prediction] What is the likelihood that this sequence ends in a proximal isoform? Response: The processing of 3’ UTR regions in the sequence results in a proximal isoform proportion of 0.33.|
|---|

APA Isoform Prediction This task has 49755 test samples.

|Example 1:<br><br>Prompt: <rna>TAATCGGTTCTAAATACGATTAGTG</rna> What ribosome loading rate is expected from the sequence? Response: The sequence impacts ribosome loading during translation initiation with an efficiency of 0.63|
|---|

Mean Ribosome Loading Prediction This task has 7600 test samples.

|Example 1:<br><br>Prompt: <rna>GGGCTTGGGGAAAGTGGACAGAGATGATTATTAAACCAAACACACAAACGCACTAATAA TCATCTCTGTCCACTTTCCCCAAGAACAGAGGAGACTTGGGATGAGTGGACAGAACCTGGCGGC AGCGCAAAAGATGCGTAAAGGAGAA</rna> [Programmable RNA switches prediction] How does the sequence influence the ON and OFF states of the RNA switch? Response: The RNA switch with this sequence is predicted to have an ON state of 0.02, an OFF state of 0.47, and an ON/OFF ratio of 0.06|
|---|

Programmable RNA Switches This task has 9154 test samples.

|Example 1:<br><br>Prompt: <rna>TTCACTGCGCAGCCTGGCATTGG</rna> What gene disruption success rate is associated with the sequence? Response: The predicted gene disruption level for this sequence is 0.26.|
|---|

CRISPR On Target Prediction This task has 416 test samples.

|Example 1:<br><br>Prompt: <sirna>a Af a c u Uf g a a u g a a Af c Gf a c u u c u c</sirna> <rna>ACGTACCTGCAGATCTACGGGGTCCTGGAGCACCCTGAAGACTCAAGACCAAAAGACTGTCAGGA AGGCAGAGTGCAGAGCAATCCACTGTCCAAGACCACACGACTTCGAGAACGGGGAATACTGGCCCCGGTCTCCCTACTAC AATGTGAGTGATGAGATCTCTTTCCACTGCTATGACGGTTACACTCTCCGGGGCTCTGCCAATCGCACCTGCCAAGTGAA TGGCCGATGGAGTGGGCAGACAGCGATCTGTGACAACGGAGCGGGGTACTGCTCCAACCCGGGCATCCCCATTGGCACAA|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1485]

|GGAAGGTGGGCAGCCAGTACCGCCTTGAAGACAGCGTCACCTACCACTGCAGCCGGGGGCTTACCCTGCGTGGCTCCCAG CGGCGAACGTGTCAGGAAGGTGGCTCTTGGAGCGGGACGGAGCCTTCCTGCCAAGACTCCTTCATGTACGACACCCCTCA AGAGGTGGCCGAAGCTTTCCTGTCTTCCCTGACAGAGACCATAGAAGGAGTCGATGCTGAGGATGGGCACGGCCCAGGGG AACAACAGAAGCGGAAGATCGTCCTGGACCCTTCAGGCTCCATGAACATCTACCTGGTGCTAGATGGATCAGACAGCATT GGGGCCAGCAACTTCACAGGAGCCAAAAAGTGTCTAGTCAACTTAATTGAGAAGGTGGCAAGTTATGGTGTGAAGCCAAG ATATGGTCTAGTGACATATGCCACATACCCCAAAATTTGGGTCAAAGTGTCTGAAGCAGACAGCAGTAATGCAGACTGGG TCACGAAGCAGCTCA</rna> What is the likely mRNA remaining percentage after siRNA treatment? Response: The sequence leads to an mRNA remaining percentage of 28.00 after siRNA treatment.|
|---|

siRNA Efficiency Prediction This task has 6688 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: KPrMnNbO6property name: is_stable Response: {is_stable : False}|
|---|

MPclassification This task has 20575 test samples.

|Example 1:<br><br>Prompt: You are a material scientist. Look at the chemical formula of the given crystalline material and predict its property. The output must be in a json format. For example: {property_name:predicted_property_value}. Answer as precise as possible and in as few words as possible.chemical formula: PbSproperty name: SOC Response: {SOC : True}|
|---|

SNUMATclassification This task has 4152 test samples.

|Example 1:<br><br>Prompt: Please conduct a domain/motif search on the given protein sequence and provide your findings. The sequence is: <protein>MGTTTTHKFDRPLRLPPLPCPFPSEVNPYVEQVDKETLEWLIDSEMLDDAETVERYRQAKYGWLSARTYP YAEHHTLRLVSDWCVWLFAFDDAFCESDRRAAEIARALPQLYAVLEDLDVGSEVDDVFAKSLLEIKGRIAAYGDDEQLD RWRNVTKDYLFAQVWEAANREDEVVPSLEDYIFMRRRTGAMLTVFALIDVASGRSLSADEWRHPGMRAITESANDVVVW DNDLISYAKESNSGNSRNNLVNVLAEHRHYSRQEAMEEIGEMRNQAIADMVAVRPSLEALGSDAVLAYVRGLEFWISGSV DYSLTSSRYTDAWRTARQPSIR</protein> Response: Based on computational analysis, the provided sequence potentially contains the following protein domains or motifs: DDXXXE motif domains|
|---|

Domain/Motif This task has 1400 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1507]

|Example 1:<br><br>Prompt: Analyze the following amino acid sequence, and determine the function of the resulting protein, its subcellular localization, and any biological processes it may be part of: <protein>MANHKSTQKSIRQDQKRNLINKSRKSNVKTFLKRVTLAINAGDKKVASEALSAAHSKLAKAANKG IYKLNTVSRKVSRLSRKIKQLEDKI</protein> Response: The protein with the amino acid sequence is expected to exhibit rRNA binding, structural constituent of ribosome, contributing to the translation. It can be typically found in the ribonucleoprotein complex, ribosome of the cell.|
|---|

Protein Function This task has 3494 test samples.

|Example 1:<br><br>Prompt: <protein>MPLGLFSSGKAQVLCDEKIPGGKKKEPKQLSENKCKGVTLKLDHTRVTIEKQIAE<br><br>GGFAIVYVASDRKNNKFALKRQFTKDNEKQLEACCREHSFLKQCIGHKNIVEFVDSYTNCLGNGIWECMLLTEYHQKNVL QLMNERISQNQYLTNDEILSIFTDLCEAVSFIHNRPQPIIHRDLKVENVLISSHKPPHYVLCDFGSATTQILSVEKYGVE YVKSEVERNTTMCYRSPEMIDFYSGLEIGLKSDIWALGVLLYRLCFFCVPFEESPLAIQSVNYQFPSVPNIPDEIKVLIY MLLDIDVNRRPSIYQTSVLAFEANHRKPLSEEIQNKKCTDAVPSLKSCIQLMRDGSNPRNKRDSSPRNPEAPPIQSSSKM ASLSQQVPSISNISMPSGSGTVETSVAPRLRPKATTVVPNVPSISPVPPVGLPHLRLPSKGSTDETDGSQVRKVPIDFHH RQSFSGEEQLKPAAEADSAGPLSCPLIKPTDLGFTDLDKPALPRDRAQTDGKRRLPHESDIIFQQQHRRNVSDTSQISRS AFKPYSSQQTTSKTSSQVVRSVEDMSQRQNGGSGEWNPFLVAPFSNNSISRKDGQESAFMMDDSHFGMVFDEIRRKEIPA ELDSETSSIDSRDPFGAAPFDQLTVSTSSSAQPVSLPPGSVHLHTSIAPSPLLRHVLTKAPPPVPRRTTSRGFQV</pro tein> Is this sequence associated with a soluble protein?<br><br>Response: The protein predicted from this sequence is not soluble.|
|---|

Solubility This task has 2001 test samples.

|Example 1:<br><br>Prompt: <protein>MGSSHHHHHHSQDLENLYFQGSMNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLL TKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAG FTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRTKRVITTFRTGTWDAYKNLRKKLEQLYNRYKDPQDENKIGIDGIQQFC DDLALDPASISVLIIAWKFRAATQCEFSKQEFMDGMTELGCDSIEKLKAQIPKMEQELKEPGRFKDFYQFTFNFAKNPGQ KGLDLEMAIAYWNLVLNGRFKFLDLWNKFLLEHHKRSIPKDTWNLLLDFSTMIADDMSNYDEEGAWPVLIDDFVEFARPQ IAGTKSTTV</protein> [EC Number Prediction Multilabel Classification] What are the donor and acceptor molecules in this enzyme’s reaction? Response: Donor molecules utilized by enzymes are identified by their EC numbers. Donor molecules it utilizes are identified by EC number EC3.2.1.17,EC3.2.1.-.|
|---|

Function-EC This task has 1919 test samples.

|Example 1:<br><br>Prompt: Ab-Ag interaction detection: Any binding between <protein>EVQLVESGANVVRPGGSLRLSCKASGFIFENFGFSWVRQAPGKGLQWVAGLNWNGGDTRYADSVKGRFRMS|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1529]

|RDNSRNFVYLDMDKVGVDDTAFYYCARGTDYTIDDAGIHYQGSGTFWYFDLWGRGTLVSVSSXEIVLAQSPGTLSLSPGE RATLSCRASHNVHPKYFAWYQQKPGQSPRLLIYGGSTRAAGIPGKFSGSGSGTDFTLTISRVDPEDFAVYYCQQYGGSPY TFGQGTKVEIKX</protein> and <protein>MRVMGIQKNYPPLWRWGMIIFWIMMICNAKQMWVTVYYGVPVWR DASTTLFCASDAKAYDPEAHNAWATHACVPTDPNPQEMQLKNVTENFNMWKNDMVEQMHVDIISLWDQSLKPCVKLTPLC VTLNCTDVRNRTLNYTYNNSTSNISLVSSDMEGEIKNCSFNMTTELRDKNQKVYALFYRQDVIQNGNNNSSYYRLINCNT SAMTQACPKVSFNPIPIHYCAPAGFAILKCNDKEFNGTGPCKNVSTVQCTHGIKPVISTQLLLNGSIAEEEIVIRSENIT NNAKTIIVQLNEPVEINCTRPNNNTRKSVRIGPGQTFYATGDIIGNIRQAHCNVSRAKWNETLHKVAEQLRTYFKNKTII FTNHSGGDVELKTHMFNCGGEFFYCNTTALFNSTWDNNSTREDNNSTGNDTITLQCRIKQIVRMWQRSGQAMYAPPIQGV IKCKSNITGLLLTRDGGNNESTNETFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTHAKRRVVQREKRAVGLGAVFFGF LGAAGSTMGAASITLTVQARQLLSGIVQQQSNLLKAIEAQQHLLKLTVWGIKQLQARVLALERYLGDQQLLGIWGCSGKL ICTTNVPWNSSWSNKTYGDIWDNMTWLQWDKEISNYTNIIYALIEESQNQQEKNEQDLLALDKWASLWNWFDISKWLWYI KIFIMIVGGLIGLRIVFTIITIINRVRQGYSPLSFQTLTHHQREPDRPERIEEEGGEQDKDRSIRLVGGFLALAWDDLRN LCLFIYHRLRDFVLIAARAVSLLGHSSLKGLRLGWEALKYLWNLLAYWGKELKNSAVNLLDTVAIAVANWTDRVIEIGQR AGRAILNIPRRIRQGAERALV</protein>?<br><br>Response: Yes. Based on their primary structures, a physical interaction appears plausible.|
|---|

Antibody-Antigen This task has 3301 test samples.

|Example 1:<br><br>Prompt: I need to know if <rna>TGCTTCGTGCTTGCTTGAAGTTAAACCCGGCGGGGGAACGA TTGTTCCCACTGAACGTTCCCACAGGAGAACCCTTGGAGACACTACCTTTCCCACAGAATTTCATTCAAGGAGCTCAAAG GCCAGGAAATCTTTTGAAAAGGTACCATCTACCTGGAGGGCATCATCCTGAGTGAGAGGACAAGCAAAAAGAGGAATGGA AGACTAGATACACAGCTAAATGGAAGGAGAACTTCTGGCCTAAATGCAATAAGGAAATCTATGTGGCTACCGAGAGCAAC AGGACACTCCTGACATTGTGTAACCCAAAGATAGAAACATGCAATTGTGCTACATGGCCCTCCAGGCTCTGGAAAAACCA CAATGGCAAAGCAGCTAATGTTAGAATGGTCAGAGCGTAAGCAGGCCCAGATTTTCTCATGTGCCTTCTACATCAGCTGC AGGGAAGTAAATAATACTAAACTTTGCACTTTTGCCCACCTCCTCTCCATGGACAATCC</rna> has any functional or structural interaction with <protein>HVLFEHAVGYALLALKEVEEISLLQPQVE ESVLNLGKFHSIVRLVAFCPFASSQVALENANAVSEGVVHEDLRLLLETHLPSKKKKVLLGVGDPKIGAAIQEELGYNCQ TGGVIAEILRGVRLHFHNLVKGLTDLSACKAQLGLGHSYSRAKVKFNVNRVDNMIIQSISLLDQLDKDINTFSMRVREWY GYHFPELVKIINDNATYCRLAQFIGNRRELNEDKLEKLEELTMDGAKAKAILDASRSSMGMDISAIDLINIESFSSRVVS LSEYRQSLHTYLRSKMSQVAPSLSALIGEAVGARLIAHAGSLTNLAKYPASTVQILGAEKALFRALKTRGNTPKYGLIFH STFIGRAAAKNKGRISRYLANKCSIASRIDCFSEVPTSVFGEKLREQVEERLSFYETGEIPRKNLDVMKEAMVQAEEAAA EITRKLEKQEKKRLKKKK</protein>. Response: The interaction between the RNA and protein is predicted to be positive.|
|---|

RNA-Protein This task has 4164 test samples.

|Example 1:<br><br>Prompt: <dna>CAACGGTACCCAATGGTTCCATCTCTAAGTCGTTAGGGCCAATCGTGGGCGGTATGCTA TTGTCACCTTCATCTTCGCTGTCTTCCTCACTCAACTTGAAAGTGGAAGAATCATTATTGTGCATTGTGAAACCTCCAGC ATTGGAAAATCTGTGTTTTAAGCTGGTGGACCTTGAAACCAAAAACAAGAAACATTGAAATGCAATGGCGGAGTGTAGTG AAGGAACAGCACCAAACACAATAGGAGACATGTGAAATCCTTTAGTGTTCAAATGAGTACCCAAATGGCTATCTACTCTT ATCAAACCGGCAGCAAACCCTTCCTGGGTGTAATTAACATGTTCGGTGTCATCAATACCGTACAGATGAGTAAACCATGG GGAGGCCATTGGAACAAGCAGATGGGTAAGGACCCCCGCTATATTTTGTAATCCTAGGGCAAAACTAAAGCATTTTAAAG TCCCCGGTGGTTGGAAGACGTATAAGTATACAGCTGTTAAA</dna> [Classification: epigenetic marks prediction] I need help with an Epigenetic Mark Prediction task. Here’s|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1551]

|the sequence: Response: EMP, or Epigenetic Marks Prediction, aims to identify epigenetic changes, successfully confirmed in yeast DNA sequences.|
|---|

Epigenetic Marks Predictions This task has 28741 test samples.

|Example 1:<br><br>Prompt: <dna>CTGCTGCTGCTGCTGGCTCTGGCTTTTGTTGTTGATGGTGACATCATCATTTAAAAAAA AACATATGCTGATCACTGTATGAAAGTTCAACAATGTTAGTG</dna>: I need a detailed review of transcription factor binding sites in this DNA sequence. Response: Upon examination, I can confirm that the mouse DNA sequence piece contains TF binding sites, which is crucial for gene expression.|
|---|

TF-Mouse This task has 10005 test samples.

|Example 1:<br><br>Prompt: <rna>AGCTACGGCCATATCACCCTGAACATGCCTGATCTCGTCTGATCTCAGAAGCTAAGCAC GGTAGGGCCTGGTTAGTACTTGGATGGAAGACCACCTGGGAATACCAGGTGCTGTAGGGG</rna> How would you classify this sequence within non-coding RNA classes? Response: The RNA sequence is categorized under the 5S_rRNA family based on its structural characteristics.|
|---|

###### Non-coding RNA Function Classification This task has 2600 test samples.

|Example 1:<br><br>Prompt: <rna>TAAAGCACTTAACATACATAACTTACTGTGGCTCAGAGTAAGCGCTTAGTAAAAATTGG AGTTATTTTTCAATAGATTCATTATCTATAATAAAGGAGAAA</rna> What does the sequence indicate about its potential RNA modifications?Please choose from the following options:[’Am’, ’Cm’, ’Gm’, ’Um’, ’m1A’, ’m5C’, ’m5U’, ’m6A’, ’m6Am’, ’m7G’, ’Psi’, ’AtoI’, ’none’] Response: AtoI.The RNA modifications predicted for the sequence are AtoI.|
|---|

Modification Prediction This task has 1200 test samples.

|Example 1:<br><br>Prompt: Analyze the interaction potential between <dna>CCTCCTGGGGCTGTTTACCAGAG<br><br>CCTTAACTAGGTGCGACCTGTTGATAGCTGGTGTAGTCCCAGGAACCAGGATTTGGATGGAGGGTACAATGAGGCTCTGA GCTGCCTTCTCCCCTTCCAGGGCTGGGGGCAGCAGCCATGCCTACCTGGGGGGCCCGCCCCGCGTCCCCAGATCGCTTTG CGGTGTCTGCGGAGGCTGAGAACAAGGTTCGGGAACAGCAGCCCCATGTGGAGCGCATCTTCAGCGTGGGGGTGAGCGTC CTTCCGAAGGACTGTCCGGACAACCCCCACATCTGGCTGCAGCTGGAGGGCCCCAAGGAAAACGCCAGCAGAGCCAAGGT GAACGCCTTCTCTCCCCCATCCCTCCAGGCACCAAGGACGCTTTCCCCCAGGGCGGAGGAGAGCAGGGCCGAGGAGAGCA|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1573]

|GGGAAGAGGAGGGGCCTGGGGAGGATCTCCACTTCTTGGTTGTCTCACTGGCAACCTCAGTGCCCCTGAGTTTTGAG</d na> and <dna>GGTCGTGCAGGTAAAACATCCCAAGCACCACTTCTTTCAGCAGGCGGCAAAGGAGCGGCCAGGGCCG AGGGCACTGGGACTGCAGCAGCCCCGACAAGGAGCCGTTCTCCATGAATTTAGTCACCAGAGCCGGCTTGGGATCTTGGT CCCAGTTCACCTTCTCGATAACCCCTTCTAGGCGCAGCACGAATTCGTTATCCAGACTTGCCATGGCCTTGACCTCCCTG GATATCGCCTTCCTACACTCCAGGAGAGAGCTGGAGTCGCACCGGGGTCGTGGGAAAATCCCTCCCTTCGCCATTCAGGC CCCAGAGCACAGTGGCTCCACCTTTTTGGCCAGATTGCGGCTGGGGTCAACCGGGGTCACTCACGAGTTTACGATCTTGA CCGCCACATCGTAGCCCCACTTCCTATGTTGCGCCCGGAACACTGTGCCGAACCCGCCTTTGCCGACGAGCTCCTGGTTC TCCAGTTCCTCGATGGACACCAAGGGGGCGGGG</dna>.<br><br>Response: This DNA sequence suggests a high probability of promoter-enhancer interaction.|
|---|

###### Enhancer-Promoter Interaction Prediction This task has 1762 test samples.

|Example 1:<br><br>Prompt: <dna>AGAAACCCTATGAATGCAGTCTTTGTAGGAAAGCTTTTTTTGAGAAGTCGGAGCTAATTAGACAT CTGAGAACTCATACAGGAGAAAAACCTTATGAATGC</dna>: Assist me in locating transcription factor binding sites in this DNA sequence. Response: My biological analysis indicates that TF binding sites are present in the given DNA fragment.|
|---|

TF-Human This task has 28741 test samples.

|Example 1:<br><br>Prompt: <dna>TGTCGCGCCCTCCCGGGCTCGGCCACTCTGCCCTTCCCTGCCCCAGCGCAGGTGAGCTGGCGCTT<br><br>GTCCTTAACCTTCTTCAACGACGAGGCGGGACGGAGGCTCGGACGCCTCCCCAGCCTCTGGGCTGGCGCAAGTCGACTGC GGTTCGTCCTCAGTCCACGATGAAGTCATAGGGCTGTGGGTCCGTGTGTCTGTGACGTCAGAAGATGGGTCCTGGGATGT CACTGCAGGCCCGGGCGGGGACCTGGGCTCACGGAAGGGCCGGGGGGACCCGCAGGGTAGAAAAAGGAAGTTGTC</dna >: Could you determine if this sequence contains a promoter?<br><br>Response: The DNA fragment includes identifiable promoter characteristics.|
|---|

Promoter-300 This task has 5920 test samples.

|Example 1:<br><br>Prompt: <dna>CTCTTGCCTTCCGGCCCGGCGCCCGATTTCCGCCTTCCGACCCAGCTGTGGGCTGCGCCCCACGC CAGCC</dna> Does this sequence function as a core promoter? Response: Yes, core promoter elements have been found in the given sequence.|
|---|

Promoter-core This task has 5000 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1595]

|Example 1:<br><br>Prompt: Based on their primary structures, is an interaction between <protein>MSGEENPASKPTPVQDVQGDGRWMSLHHRFVADSKDKEPEVVFIGDSLVQLMHQCEIWRELFSPLHALNF GIGGDGTQHVLWRLENGELEHIRPKIVVVWVGTNNHGHTAEQVTGGIKAIVQLVNERQPQARVVVLGLLPRGQHPNPLR EKNRQVNELVRAALAGHPRAHFLDADPGFVHSDGTISHHDMYDYLHLSRLGYTPVCRALHSLLLRLLAQDQGQGAPLLE PAP</protein> and <protein>MWSPEREAEAPAGGDPAGLLPPEWEEDEERMSFLFSAFKRSREVNSTDWDSK MGFWAPLVLSHSRRQGVVRLRLRDLQEAFQRKGSVPLGLATVLQDLLRRGELQRESDFMASVDSSWISWGVGVFLLKPL KWTLSNMLGDNKVPAEEVLVAVELLKEKAEEVYRLYQNSPLSSHPVVALSELSTLCANSCPDERTFYLVLLQLQKEKRV TVLEQNGEKIVKFARGPRAKVSPVNDVDVGVYQLMQSEQLLSRKVESLSQEAERCKEEARRACRAGKKQLALRSL KAKQRTEKRIEALHAKLDTVQGILDRIYASQTDQMVFNAYQAGVGALKLSMKDVTVEKAESLVDQIQELCDTQDEVSQT LAGGVTNGLDFDSEELEKELDILLQDTTKEPLDLPDNPRNRHFTNSVPNPRISDAELEAELEKLSLSEGGLVPSSKSPK RQLEPTLKPL</protein> plausible in a human cell? Please answer ’Yes’ if there is a stable physical association between two proteins and ’No’ if not. Response: It is likely that these proteins interact. The answer is Yes.|
|---|

Human PPI This task has 237 test samples.

|Example 1:<br><br>Prompt: Based on the primary amino acid chains, is there evidence for a stable physical association between yeast protein A (<protein>MSNYPLHQACMENEFFKVQELLHSKPSLLLQKDQDGRIPLHWSVSFQAHEITSFLLSKMENVNLDDYPDD SGWTPFHIACSVGNLEVVKSLYDRPLKPDLNKITNQGVTCLHLAVGKKWFEVSQFLIENGASVRIKDKFNQIPLHRAASV GSLKLIELLCGLGKSAVNWQDKQGWTPLFHALAEGHGDAAVLLVEKYGAEYDLVDNKGAKAEDVALNEQVKKFFLNNV </protein>) and yeast protein B (<protein>METIDIQNRSFVVRWVKCGRGDVINYQIKPLKKSIEVG IYKKLKSSVDDHASAVHIAPDTKTLLDYTTKSLLHKGSSSNIEEHHRRSSQHSHSSSNGSDNKRKERSYSSLSISGIQQQ SQEIPLREKLSASGFTLVKRVGNVSGNTMVQGDLEVKDTDYYYAFILDNSSSKNAKKKILFNASVINGDNQSMISTRSTP PARPTALSRTSTQQDMLFRVGQGRYLQGYLLKKRRKRLQGFKKRFFTLDFRYGTLSYYLNDHNQTCRGEIVISLSSVSAN KKDKIIIIDSGMEVWVLKATTKENWQSWVDALQTCFDDQFEDKDTSTLEENPDILDDDKEVINKSSPQDHDHLTPTATTK SALSHRQHTQKDMDDIYVPLPSESYATFSMNLRLIQQRLEQCKKDSLSYKPTTLHQRSEGLNGTHSSSSVFTNNRVSSFN HSSSGMTSSDSLASEEVPSNKTYIEHALYNQLADLEVFVSRFVTQGEVLFKDHQILCKKAKDTRVSLTSYLSENDEFFDA EEEISRGVIILPDTEDDINNIVEETPLLGKSDQNEFTKEVQLSGSEQIASSSVESYTTNDENHSRKHLKNRHKNRRRGHP HHQKTKSAQSSTETFTSKDLFALSYPKSVTRRNDIPEAAASPPSLLSFLRKNVGKDLSSIAMPVTSNEPISILQLISETF EYAPLLTKATQRPDPITFVSAFAISFLSIYRDKTRTLRKPFNPLLAETFELIREDMGFRLISEKVSHRPPVFAFFAEHLD WECSYTVTPSQKFWGKSIELNNEGILRLKFKTTGELFEWTQPTTILKNLIAGERYMEPVNEFEVHSSKGDKSHILFDKAG MFSGRSEGFKVSIIPPPSSNRKKETLAGKWTQSLANETTHETIWEVGDLVSNPKKKYGFTKFTANLNEITEIEKGNLPPT DSRLRPDIRAYEEGNVDKAEEWKLKLEQLQRERRNKGQDVEPKYFEKVSKNEWKYITGPKSYWERRKKHDWSDISQLW </protein>)? Please answer ’Yes’ if there is a stable physical association between two proteins and ’No’ if not. Response: Interaction: Yes. The answer is Yes.|
|---|

Yeast PPI This task has 394 test samples.

|Example 1:<br><br>Prompt: : Return the EC number of the protein sequence.<br><br><protein>MSILEWFWNILCGMAQYLTFSKNLTNDNLVNIYVKSNVGGTISVNLDPKSDIKNVKELVAPKLGLEPDDVK IIFAGKELLDSTVIEVLDFFSDILHAVKVNKKIKNVIPDKPLCETLEELHQLNDQKNVESIEESNLKNEGKNKAHFFIYC ANPCKKINTGKLRVCCSECKHGAFTVDTDPQSWADVLDKNKITGVCNNVGCEGLYAKFYFKCASHPSQGENDTAVPLNLI KRNHKKIPCLACTDICDPVLVFSCDNRHVTCLECFKNYCGSRLKDRQFLSHPDFGYTLPCPAGCSNSFIEEVHHFRLLTD|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1617]

|AQYEQYHRFATEEFILQAGGVLCPQPGCGQGILIDQNCNRVQCSCGYVFCGKCLEGFHLGECLNPTDVPFLSQNCDYPLD PEKLEKARWDEASSTVIKVLTKPCPKCRTSTERAGGCMHMICTRANCGFHWCWVCQGPWERDCMASHWFG</protein> Response: : 2 . 3 . 2 . 31|
|---|

Numerical EC (New-392) This task has 382 test samples.

|Example 1:<br><br>Prompt: Return the EC number of the protein sequence. <protein>MAIPPYPDFRSAAF LRQHLRATMAFYDPVATDASGGQFHFFLDDGTVYNTHTRHLVSATRFVVTHAMLYRTTGEARYQVGMRHALEFLRTAFL DPATGGYAWLIDWQDGRATVQDTTRHCYGMAFVMLAYARAYEAGVPEARVWLAEAFDTAEQHFWQPAAGLYADEASPDW QLTSYRGQNANMHACEAMISAFRATGERRYIERAEQLAQGICQRQAALSDRTHAPAAEGWVWEHFHADWSVDWDYNRHD RSNIFRPWGYQVGHQTEWAKLLLQLDALLPADWHLPCAQRLFDTAVERGWDAEHGGLYYGMAPDGSICDDGKYHWVQA ESMAAAAVLAVRTGDARYWQWYDRIWAYCWAHFVDHEHGAWFRILHRDNRNTTREKSNAGKVDYHNMGACYDVLLWAL DAPGFSKESRSAALGRP</protein> Response: 5 . 3 . 1 . 7|
|---|

NumericalEC (Price-149) This task has 149 test samples.

|Example 1:<br><br>Prompt: Through sequence analysis, forecast the protein’s subcellular<br><br>positioning in eukaryotic cells. <protein>AGFPEQEPEPKFWNDWAQKTLDKALSLQTLNKNKAQN LILFLGDGMGVPTVTAARILKGQLRGQPGEEGQLEMDKFPFVALSKTYNTNAQVADSAGTATAYLCGVKANEGTVGVSAA AVRSQANTTQGNEVTSILRWAKDAGKSIGIVTTTRVNHATPSAAYAHCVDRDWYSDNEMPADAVEAGCKDIARQLFENIP DIDVIMGGGRKYMYPKNTTDVEYPGQPKHSGTRKDGRNLVKEWVDRNTEKKGHYVWNKKDLLSLNPTKVDYLLGLFEPAD LPYDLERNKETDPSLSEMVEVAIKILRRNPNGFYLLVEGGRIDHGHHEGKDKQAIHEAVEMDRAIGRADLMTSTSDTLTV VTADHSHLFSFGGYTPRGNEIFGLAAFISDVDQKPFTAILYGNGPGYKLVNGARENVSTVDYQDNSYLAQAAVPLSSETH GGEDVAVFAKGPMAHLLHGVHEQNYIPHAMAYAACIGQNR</protein><br><br>Response: membrane|
|---|

Subcellular Localization This task has 2772 test samples.

|Example 1:<br><br>Prompt: Analyze the provided protein sequence and predict the UniProtKB<br><br>keywords associated with it. <protein>MAAADGGGPGGASVGTEEDGGGVGHRTVYLFDRREKESELG DRPLQVGERSDYAGFRACVCQTLGISPEEKFVITTTSRKEITCDNFDETVKDGVTLYLLQSVNQLLLTATKERIDFLPHY DTLVKSGMYEYYASEGQNPLPFALAELIDNSLSATSRNIGVRRIQIKLLFDETQGKPAVAVIDNGRGMTSKQLNNWAVYR LSKFTRQGDFESDHSGYVRPVPVPRSLNSDISYFGVGGKQAVFFVGQSARMISKPADSQDVHELVLSKEDFEKKEKNKEA IYSGYIRNRKPSDSVHITNDDERFLHHLIIEEKEKDSFTAVVITGVQPEHIQYLKNYFHLWTRQLAHIYHYYIHGPKGNE IRTSKEVEPFNNIDIEISMFEKGKVPKIVNLREIQDDMQTLYVNTAADSFEFKAHVEGDGVVEGIIRYHPFLYDRETYPD DPCFPSKLKDEDDEDDCFILEKAARGKRPIFECFWNGRLIPYTSVEDFDWCTPPKKRGLAPIECYNRISGALFTNDKFQV STNKLTFMDLELKLKDKNTLFTRILNGQEQRMKIDREFALWLKDCHEKYDKQIKFTLFKGVITRPDLPSKKQGPWATYAA IEWDGKIYKAGQLVKTIKTLPLFYGSIVRFFLYGDHDGEVYATGGEVQIAMEPQALYDEVRTVPIAKLDRTVAEKAVKKY VEDEMARLPDRLSVTWPEGDELLPNEVRPAGTPIGALRIEILNKKGEAMQKLPGTSHGGSKKLLVELKVILHSSSGNKEI ISHISQHGGKWPYWFKKMENIQKLGNYTLKLQVVLNESNADTYAGRPLPSKAIKFSVKEGKPEKFSFGLLDLPFRVGVPF NIPLEFQDEFGHTSQLVTDIQPVLEASGLSLHYEEITKGPNCVIRGVTAKGPVNSCQGKNYNLKVTLPGLKEDSQILKIR|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1639]

|LLPGHPRRLKVKPDSEILVIENGTAFPFQVEVLDESDNITAQPKLIVHCKFSGAPNLPVYVVDCSSSGTSILTGSAIQVQ NIKKDQTLKARIEIPSCKDVAPVEKTIKLLPSSHVARLQIFSVEGQKAIQIKHQDEVNWIAGDIMHNLIFQMYDEGEREI NITSALAEKIKVNWTPEINKEHLLQGLLPDVQVPTSVKDMRYCQVSFQDDHVSLESAFTVRPLPDEPKHLKCEMKGGKTV QMGQELQGEVVIIITDQYGNQIQAFSPSSLSSLSIAGVGLDSSNLKTTFQENTQSISVRGIKFIPGPPGNKDLCFTWREF SDFIRVQLISGPPAKLLLIDWPELKESIPVINGRDLQNPIIVQLCDQWDNPAPVQHVKISLTKASNLKLMPSNQQHKTDE KGRANLGVFSVFAPRGEHTLQVKAIYNKSIIEGPIIKLMILPDPEKPVRLNVKYDKDASFLAGGLFTDFMISVISEDDSI IKNINPARISMKMWKLSTSGNRPPANAETFSCNKIKDNDKEDGCFYFRDKVIPNKVGTYCIQFGFMMDKTNILNSEQVIV EVLPNQPVKLVPKIKPPTPAVSNVRSVASRTLVRDLHLSITDDYDNHTGIDLVGTIIATIKGSNEEDTDTPLFIGKVRTL EFPFVNGSAEIMSLVLAESSPGRDSTEYFIVFEPRLPLLSRTLEPYILPFMFYNDVKKQQQMAALTKEKDQLSQSIVMYK SLFEASQQLLNEMKCQVEEARLKEAQLRNELKIHNIDIPTTQQVPHIEALLKRKLSEQEELKKKPRRSCTLPNYTKGSGD VLGKIAHLAQIEDDRAAMVISWHLASDMDCVVTLTTDAARRIYDETQGRQQVLPLDSIYKKTLPDWKRSLPHFRNGKLYF KPIGDPVFARDLLTFPDNVEHCETVFGMLLGDTIILDNLDAANHYRKEVVKITHCPTLLTRDGDRIRSNGKFGGLQNKAP PMDKLRGMVFGAPVPKQCLILGEQIDLLQQYRSAVCKLDSVNKDLNSQLEYLRTPDMRKKKQELDEHEKNLKLIEEKLGM TPIRKCNDSLRHSPKVETTDCPVPPKRMRREATRQNRIITKTDV</protein><br><br>Response: 3D-structure ; Acetylation ; Alternative splicing ; Chromatin regulator ; Chromosome ; Direct protein sequencing ; Disease variant ; DNA damage ; DNA repair ; DNA-binding ; Hydrolase ; Hypogonadotropic hypogonadism ; Isopeptide bond ; Kallmann syndrome ; Microphthalmia ; Phosphoprotein ; Reference proteome ; Ubl conjugation|
|---|

###### Keywords-CASPSimilarSeq This task has 184 test samples.

|Example 1:<br><br>Prompt: Please list the UniProtKB keywords associated with the provided<br><br>protein sequence. <protein>MTSVAKVYYSQTTQTESRPLMGPGIRRRRVLTKDGRSNVRMEHIADKRFLYL KDLWTTFIDMQWRYKLLLFSATFAGTWFLFGVVWYLVAVAHGDLLELDPPANHTPCVVQVHTLTGAFLFSLESQTTIGYG FRYISEECPLAIVLLIAQLVLTTILEIFITGTFLAKIARPKKRAETIRFSQHAVVASHNGKPCLMIRVANMRKSLLIGCQ VTGKLLQTHQTKEGENIRLNQVNVTFQVDTASDSPFLILPLTFYHVVDETSPLKDLPLRSGEGDFELVLILSGTVESTSA TCQVRTSYLPEEILWGYEFTPAISLSASGKYIADFSLFDQVVKVASPSGLRDSTVRYGDPEKLKLEESLREQAEKEGSAL SVRISNV</protein><br><br>Response: ATP-binding ; Cell membrane ; Deafness ; Disease variant ; Epilepsy ; Intellectual disability ; Ion channel ; Ion transport ; Membrane ; Nucleotide-binding ; Potassium ; Potassium transport ; Reference proteome ; Transmembrane ; Transmembrane helix ; Transport ; Voltage-gated channel|
|---|

Keywords-IDFilterSeq This task has 1112 test samples.

|Example 1:<br><br>Prompt: Analyze the given protein sequence and generate the relevant<br><br>UniProtKB keywords. <protein>MWLKLFFLLLYFLVLFVLARFFEAIVWYETGIFATQLVDPVALSFKKLKT ILECRGLGYSGLPEKKDVRELVEKSGDLMEGELYSALKEEEASESVSSTNFSGEMHFYELVEDTKDGIWLVQVIANDRSP LVGKIHWEKMVKKVSRFGIRTGTFNCSSDPRYCRRRGWVRSTLIMSVPQTSTSKGKVMLKEYSGRKIEVEHIFKWITAHA ASRIKTIYNAEHLKEEWNKSDQYWLKIYLFANLDQPPAFFSALSIKFTGRVEFIFVNVENWDNKSYMTDIGIYNMPSYIL RTPEGIYRYGNHTGEFISLQAMDSFLRSLQPEVNDLFVLSLVLVNLMAWMDLFITQGATIKRFVVLISTLGTYNSLLIIS WLPVLGFLQLPYLDSFYEYSLKLLRYSNTTTLASWVRADWMFYSSHPALFLSTYLGHGLLIDYFEKKRRRNNNNDEVNAN NLEWLSSLWDWYTSYLFHPIASFQNFPVESDWDEDPDLFLERLAFPDLWLHPLIPTDYIKNLPMWRFKCLGVQSEEEMSE GSQDTENDSESENTDTLSSEKEVFEDKQSVLHNSPGTASHCDAEACSCANKYCQTSPCERKGRSYGSYNTNEDMEPDWLT WPADMLHCTECVVCLENFENGCLLMGLPCGHVFHQNCIVMWLAGGRHCCPVCRWPSYKKKQPYAQHQPLSNDVPS</pro|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1651]

[Figure 1652]

[Figure 1653]

[Figure 1654]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1661]

|tein> Response: Endoplasmic reticulum ; Membrane ; Metal-binding ; Reference proteome ; Transferase ; Transmembrane ; Transmembrane helix ; Ubl conjugation pathway ; Zinc ; Zinc-finger|
|---|

Keywords-UniProtSeq This task has 4562 test samples.

|Example 1:<br><br>Prompt: Please predict the folding type of the protein sequence, the prediction range is from 0 to 1194. <protein>MLTAEEKAAVTGFWGKVDVDVVGAQALGRLLVVY PWTQRFFQHFGNLSSAGAVMNNPKVKAHGKRVLDAFTQGLKHLDDLKGAFAQLSGLHCNKLHVNPQNFRLLGNVLALVV ARNFGGQFTPNVQALFQKVVAGVANALAHKYH</protein> Response: 0|
|---|

Fold Type This task has 3243 test samples.

|Example 1:<br><br>Prompt: Predict the GO term based on the protein sequence. <protein>MAAADGGGPGGASVGTEEDGGGVGHRTVYLFDRREKESELGDRPLQVGERSDYAGFRACVCQTLGISPEEK FVITTTSRKEITCDNFDETVKDGVTLYLLQSVNQLLLTATKERIDFLPHYDTLVKSGMYEYYASEGQNPLPFALAELIDN SLSATSRNIGVRRIQIKLLFDETQGKPAVAVIDNGRGMTSKQLNNWAVYRLSKFTRQGDFESDHSGYVRPVPVPRSLNSD ISYFGVGGKQAVFFVGQSARMISKPADSQDVHELVLSKEDFEKKEKNKEAIYSGYIRNRKPSDSVHITNDDERFLHHLII EEKEKDSFTAVVITGVQPEHIQYLKNYFHLWTRQLAHIYHYYIHGPKGNEIRTSKEVEPFNNIDIEISMFEKGKVPKIVN LREIQDDMQTLYVNTAADSFEFKAHVEGDGVVEGIIRYHPFLYDRETYPDDPCFPSKLKDEDDEDDCFILEKAARGKRPI FECFWNGRLIPYTSVEDFDWCTPPKKRGLAPIECYNRISGALFTNDKFQVSTNKLTFMDLELKLKDKNTLFTRILNGQEQ RMKIDREFALWLKDCHEKYDKQIKFTLFKGVITRPDLPSKKQGPWATYAAIEWDGKIYKAGQLVKTIKTLPLFYGSIVRF FLYGDHDGEVYATGGEVQIAMEPQALYDEVRTVPIAKLDRTVAEKAVKKYVEDEMARLPDRLSVTWPEGDELLPNEVRPA GTPIGALRIEILNKKGEAMQKLPGTSHGGSKKLLVELKVILHSSSGNKEIISHISQHGGKWPYWFKKMENIQKLGNYTLK LQVVLNESNADTYAGRPLPSKAIKFSVKEGKPEKFSFGLLDLPFRVGVPF... (We hide the remaining seq uences for readability)</protein> Response: Barr body ; site of double-strand break ; ATP binding ; ATP hydrolysis activity ; DNA binding ; protein homodimerization activity ; chromosome organization ; dosage compensation by inactivation of X chromosome ; double-strand break repair ; inactivation of X chromosome by heterochromatin formation ; negative regulation of double-strand break repair via homologous recombination ; nose development ; positive regulation of DNA repair ; positive regulation of double-strand break repair via nonhomologous end joining|
|---|

GO-CASPSimilarSeq This task has 184 test samples.

|Example 1:<br><br>Prompt: Given a protein sequence, predict the corresponding Gene Ontology term that describes its molecular function, biological process, and cellular component. <protein>MTSVAKVYYSQTTQTESRPLMGPGIRRRRVLTKDGRSNVRMEHIADKRFLYLKDLWTTF|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1683]

|IDMQWRYKLLLFSATFAGTWFLFGVVWYLVAVAHGDLLELDPPANHTPCVVQVHTLTGAFLFSLESQTTIGYGFRYISEE CPLAIVLLIAQLVLTTILEIFITGTFLAKIARPKKRAETIRFSQHAVVASHNGKPCLMIRVANMRKSLLIGCQVTGKLLQ THQTKEGENIRLNQVNVTFQVDTASDSPFLILPLTFYHVVDETSPLKDLPLRSGEGDFELVLILSGTVESTSATCQVRTS YLPEEILWGYEFTPAISLSASGKYIADFSLFDQVVKVASPSGLRDSTVRYGDPEKLKLEESLREQAEKEGSALSVRISNV </protein><br><br>Response: astrocyte projection ; basolateral plasma membrane ; cell body ; ciliary base ; plasma membrane ; presynapse ; ATP binding ; ATP-activated inward rectifier potassium channel activity ; inward rectifier potassium channel activity ; adult walking behavior ; cellular response to potassium ion ; central nervous system myelination ; glutamate reuptake ; non-motile cilium assembly ; potassium ion homeostasis ; potassium ion import across plasma membrane ; potassium ion transport ; regulation of long-term neuronal synaptic plasticity ; regulation of monoatomic ion transmembrane transport ; regulation of resting membrane potential ; visual perception|
|---|

GO-IDFilterSeq This task has 1112 test samples.

|Example 1:<br><br>Prompt: For a given protein sequence, identify the appropriate Gene Ontology term that corresponds to its molecular function, biological process, and cellular component. <protein>MWLKLFFLLLYFLVLFVLARFFEAIVWYETGIFATQLVDPVALSFKKLKT ILECRGLGYSGLPEKKDVRELVEKSGDLMEGELYSALKEEEASESVSSTNFSGEMHFYELVEDTKDGIWLVQVIANDRSP LVGKIHWEKMVKKVSRFGIRTGTFNCSSDPRYCRRRGWVRSTLIMSVPQTSTSKGKVMLKEYSGRKIEVEHIFKWITAHA ASRIKTIYNAEHLKEEWNKSDQYWLKIYLFANLDQPPAFFSALSIKFTGRVEFIFVNVENWDNKSYMTDIGIYNMPSYIL RTPEGIYRYGNHTGEFISLQAMDSFLRSLQPEVNDLFVLSLVLVNLMAWMDLFITQGATIKRFVVLISTLGTYNSLLIIS WLPVLGFLQLPYLDSFYEYSLKLLRYSNTTTLASWVRADWMFYSSHPALFLSTYLGHGLLIDYFEKKRRRNNNNDEVNAN NLEWLSSLWDWYTSYLFHPIASFQNFPVESDWDEDPDLFLERLAFPDLWLHPLIPTDYIKNLPMWRFKCLGVQSEEEMSE GSQDTENDSESENTDTLSSEKEVFEDKQSVLHNSPGTASHCDAEACSCANKYCQTSPCERKGRSYGSYNTNEDMEPDWLT WPADMLHCTECVVCLENFENGCLLMGLPCGHVFHQNCIVMWLAGGRHCCPVCRWPSYKKKQPYAQHQPLSNDVPS</pro tein> Response: endoplasmic reticulum ; endoplasmic reticulum membrane ; endoplasmic reticulum quality control compartment ; metal ion binding ; ubiquitin protein ligase activity ; ubiquitin-protein transferase activity ; central nervous system development ; endoplasmic reticulum mannose trimming ; protein ubiquitination ; ubiquitin-dependent ERAD pathway|
|---|

GO-UniProtSeq This task has 4562 test samples.

|Example 1:<br><br>Prompt: According to the relevant data reported in the scientific literature, the Gene symbol of a given protein can be used to predict which tissues the protein is expressed in. TMEM198 Response: adrenal gland , appendix , breast , bronchus , cerebral cortex , colon , fallopian tube , gallbladder , kidney , nasopharynx , prostate , rectum , salivary gland , seminal vesicle , skeletal muscle , small intestine , stomach , urinary bladder|
|---|

gSymbol2Tissue This task has 2181 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1705]

|Example 1:<br><br>Prompt: Get the cancer name that is associated with the given gene symbol. MLLT1 Response: AL|
|---|

gSymbol2Cancer This task has 148 test samples.

|Example 1:<br><br>Prompt: Return the cancer name associated with the given gene name. myeloid/lymphoid or mixed-lineage leukemia (trithorax homolog, Drosophila); translocated to, 1 (ENL) Response: AL|
|---|

gName2Cancer This task has 148 test samples.

|Example 1:<br><br>Prompt: Translate the given SMILES formula of a molecule <SMILES> CCC(C)C1CNCCCNC1 </SMILES> into its IUPAC name. Response: <IUPAC> 3-butan-2-yl-1,5-diazocane </IUPAC>|
|---|

S2I This task has 2993 test samples.

|Example 1:<br><br>Prompt: Could you provide the SMILES for <IUPAC> 4-ethyl-4-methyloxolan-2-one </IUPAC> ? Response: Of course. It’s <SMILES> CCC1(C)COC(=O)C1 </SMILES> .|
|---|

I2S This task has 2993 test samples.

|Example 1:<br><br>Prompt: Given the SMILES representation <SMILES> S=P1(N(CCCl)CCCl)NCCCO1 </SMILES>, what would be its molecular formula? Response: It is C7H15Cl2N2OPS .|
|---|

S2F This task has 2993 test samples.

|Example 1:<br><br>Prompt: What is the molecular formula of the compound with this IUPAC name <IUPAC> 2,5-diphenyl-1,3-oxazole </IUPAC> ? Response: C15H11NO|
|---|

I2F This task has 2993 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1727]

|Example 1:<br><br>Prompt: Could you provide a description of this molecule? <SMILES> C/C=C/C1=CC2=C(C=C1)OC(=C2C)C3=CC=C(C=C3)OC4=C(C=CC(=C4)C5=C(C6=C(O5)C=CC(=C6) /C=C/C)C)O </SMILES> Response: The molecule is a natural product found in Piper aequale with data available.|
|---|

Molecular Description This task has 1000 test samples.

|Example 1:<br><br>Prompt: Describe this molecule: <SMILES> CCOC(=O)C1=CN=CN1[C@H](C)C1=CC=CC=C1 </SMILES> Response: The molecule is an imidazole derivative with short-acting sedative, hypnotic, and general anesthetic properties. Etomidate appears to have gamma-aminobutyric acid (GABA) like effects, mediated through GABA-A receptor. The action enhances the inhibitory effect of GABA on the central nervous system by causing chloride channel opening events which leads to membrane hyperpolarization.|
|---|

Molecule Captioning This task has 2538 test samples.

|Example 1:<br><br>Prompt: Examine the given protein sequence and share a brief overview of its attributes: <protein>MKIVLASNNQGKLAELKAMLAPLGVQLLRQAELGIPEAAEPFRTFVENALAKARHA SALSGLPALADDAGLCVEAFGGLPGVDTAFYAVQFGYAKGDANNVKALLAQMAGITDRRAALVSTLVAVRSA EDPEPLIACGRVAGEVALEPMGSNGFGFDPVMFIPEFGQTFAQLPVEVKNANSHRGKATRQMMALMRER WIT </protein> Response: A concise description of the protein with the specified amino acid sequence includes: Pyrophosphatase that catalyzes the hydrolysis of nucleoside triphosphates to their monophosphate derivatives, with a high preference for the non-canonical purine nucleotides XTP (xanthosine triphosphate), dITP (deoxyinosine triphosphate) and ITP. Seems to function as a house-cleaning enzyme that removes non-canonical purine nucleotides from the nucleotide pool, thus preventing their incorporation into DNA/RNA and avoiding chromosomal lesions.|
|---|

###### Protein Functional Translation-Molins This task has 2633 test samples.

|Example 1:<br><br>Prompt: Predict the general functional description of the protein sequence.<br><br><protein>MAAADGGGPGGASVGTEEDGGGVGHRTVYLFDRREKESELGDRPLQVGERSDYAGFRACVCQTLGISPEEK FVITTTSRKEITCDNFDETVKDGVTLYLLQSVNQLLLTATKERIDFLPHYDTLVKSGMYEYYASEGQNPLPFALAELIDN SLSATSRNIGVRRIQIKLLFDETQGKPAVAVIDNGRGMTSKQLNNWAVYRLSKFTRQGDFESDHSGYVRPVPVPRSLNSD ISYFGVGGKQAVFFVGQSARMISKPADSQDVHELVLSKEDFEKKEKNKEAIYSGYIRNRKPSDSVHITNDDERFLHHLII EEKEKDSFTAVVITGVQPEHIQYLKNYFHLWTRQLAHIYHYYIHGPKGNEIRTSKEVEPFNNIDIEISMFEKGKVPKIVN|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1749]

|LREIQDDMQTLYVNTAADSFEFKAHVEGDGVVEGIIRYHPFLYDRETYPDDPCFPSKLKDEDDEDDCFILEKAARGKRPI FECFWNGRLIPYTSVEDFDWCTPPKKRGLAPIECYNRISGALFTNDKFQVSTNKLTFMDLELKLKDKNTLFTRILNGQEQ RMKIDREFALWLKDCHEKYDKQIKFTLFKGVITRPDLPSKKQGPWATYAAIEWDGKIYKAGQLVKTIKTLPLFYGSIVRF FLYGDHDGEVYATGGEVQIAMEPQALYDEVRTVPIAKLDRTVAEKAVKKYVEDEMARLPDRLSVTWPEGDELLPNEVRPA GTPIGALRIEILNKKGEAMQKLPGTSHGGSKKLLVELKVILHSSSGNKEIISHISQHGGKWPYWFKKMENIQKLGNYTLK LQVVLNESNADTYAGRPLPSKAIKFSVKEGKPEKFSFGLLDLPFRVGVPFNIPLEFQDEFGHTSQLVTDIQPVLEASGLS LHYEEITKGPNCVIRGVTAKGPVNSCQGKNYNLKVTLPGLKEDSQILKIRLLPGHPRRLKVKPDSEILVIENGTAFPFQV EVLDESDNITAQPKLIVHCKFSGAPNLPVYVVDCSSSGTSILTGSAIQVQNIKKDQTLKARIEIPSCKDVAPVEKTIKLL PSSHVARLQIFSVEGQKAIQIKHQDEVNWIAGDIMHNLIFQMYDEGEREINITSALAEKIKVNWTPEINKEHLLQGLLPD VQVPTSVKDMRYCQVSFQDDHVSLESAFTVRPLPDEPKHLKCEMKGGKTVQMGQELQGEVVIIITDQYGNQIQAFSPSSL SSLSIAGVGLDSSNLKTTFQENTQSISVRGIKFIPGPPGNKDLCFTWREFSDFIRVQLISGPPAKLLLIDWPELKESIPV INGRDLQNPIIVQLCDQWDNPAPVQHVKISLTKASNLKLMPSNQQHKTDEKGRANLGVFSVFAPRGEHTLQVKAIYNKSI IEGPIIKLMILPDPEKPVRLNVKYDKDASFLAGGLFTDFMISVISEDDSIIKNINPARISMKMWKLSTSGNRPPANAETF SCNKIKDNDKEDGCFYFRDKVIPNKVGTYCIQFGFMMDKTNILNSEQVIVEVLPNQPVKLVPKIKPPTPAVSNVRSVASR TLVRDLHLSITDDYDNHTGIDLVGTIIATIKGSNEEDTDTPLFIGKVRTLEFPFVNGSAEIMSLVLAESSPGRDSTEYFI VFEPRLPLLSRTLEPYILPFMFYNDVKKQQQMAALTKEKDQLSQSIVMYKSLFEASQQLLNEMKCQVEEARLKEAQLRNE LKIHNIDIPTTQQVPHIEALLKRKLSEQEELKKKPRRSCTLPNYTKGSGDVLGKIAHLAQIEDDRAAMVISWHLASDMDC VVTLTTDAARRIYDETQGRQQVLPLDSIYKKTLPDWKRSLPHFRNGKLYFKPIGDPVFARDLLTFPDNVEHCETVFGMLL GDTIILDNLDAANHYRKEVVKITHCPTLLTRDGDRIRSNGKFGGLQNKAPPMDKLRGMVFGAPVPKQCLILGEQIDLLQQ YRSAVCKLDSVNKDLNSQLEYLRTPDMRKKKQELDEHEKNLKLIEEKLGMTPIRKCNDSLRHSPKVETTDCPVPPKRMRR EATRQNRIITKTDV</protein><br><br>Response: Non-canonical member of the structural maintenance of chromosomes (SMC) protein family that plays a key role in epigenetic silencing by regulating chromatin architecture (By similarity). Promotes heterochromatin formation in both autosomes and chromosome X, probably by mediating the merge of chromatin compartments (By similarity). Plays a key role in chromosome X inactivation in females by promoting the spreading of heterochromatin . Recruited to inactivated chromosome X by Xist RNA and acts by mediating the merge of chromatin compartments: promotes random chromatin interactions that span the boundaries of existing structures, leading to create a compartment-less architecture typical of inactivated chromosome X (By similarity). Required to facilitate Xist RNA spreading (By similarity). Also required for silencing of a subset of clustered autosomal loci in somatic cells, such as the DUX4 locus . Has ATPase activity; may participate in structural manipulation of chromatin in an ATP-dependent manner as part of its role in gene expression regulation . Also plays a role in DNA repair: localizes to sites of DNA double-strand breaks in response to DNA damage to promote the repair of DNA double-strand breaks . Acts by promoting non-homologous end joining (NHEJ) and inhibiting homologous recombination (HR) repair .|
|---|

Protein Functional Translation-CASPSimilarSeq This task has 184 test samples.

|Example 1:<br><br>Prompt: Predict the functional description of the protein sequence below. <protein>MTSVAKVYYSQTTQTESRPLMGPGIRRRRVLTKDGRSNVRMEHIADKRFLYLKDLWTTFIDMQWRYKLLLF SATFAGTWFLFGVVWYLVAVAHGDLLELDPPANHTPCVVQVHTLTGAFLFSLESQTTIGYGFRYISEECPLAIVLLIAQL VLTTILEIFITGTFLAKIARPKKRAETIRFSQHAVVASHNGKPCLMIRVANMRKSLLIGCQVTGKLLQTHQTKEGENIRL NQVNVTFQVDTASDSPFLILPLTFYHVVDETSPLKDLPLRSGEGDFELVLILSGTVESTSATCQVRTSYLPEEILWGYEF TPAISLSASGKYIADFSLFDQVVKVASPSGLRDSTVRYGDPEKLKLEESLREQAEKEGSALSVRISNV</protein> Response: May be responsible for potassium buffering action of glial cells in|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1771]

|the brain. Inward rectifier potassium channels are characterized by a greater tendency to allow potassium to flow into the cell rather than out of it. Their voltage dependence is regulated by the concentration of extracellular potassium; as external potassium is raised, the voltage range of the channel opening shifts to more positive voltages. The inward rectification is mainly due to the blockage of outward current by internal magnesium. Can be blocked by extracellular barium and cesium (By similarity). In the kidney, together with KCNJ16, mediates basolateral K(+) recycling in distal tubules; this process is critical for Na(+) reabsorption at the tubules.|
|---|

Protein Functional Translation-IDFilterSeq This task has 1112 test samples.

|Example 1:<br><br>Prompt: Predict the general functional description of the protein sequence.<br><br><protein>MWLKLFFLLLYFLVLFVLARFFEAIVWYETGIFATQLVDPVALSFKKLKTILECRGLGYSGLPEKKDVREL VEKSGDLMEGELYSALKEEEASESVSSTNFSGEMHFYELVEDTKDGIWLVQVIANDRSPLVGKIHWEKMVKKVSRFGIRT GTFNCSSDPRYCRRRGWVRSTLIMSVPQTSTSKGKVMLKEYSGRKIEVEHIFKWITAHAASRIKTIYNAEHLKEEWNKSD QYWLKIYLFANLDQPPAFFSALSIKFTGRVEFIFVNVENWDNKSYMTDIGIYNMPSYILRTPEGIYRYGNHTGEFISLQA MDSFLRSLQPEVNDLFVLSLVLVNLMAWMDLFITQGATIKRFVVLISTLGTYNSLLIISWLPVLGFLQLPYLDSFYEYSL KLLRYSNTTTLASWVRADWMFYSSHPALFLSTYLGHGLLIDYFEKKRRRNNNNDEVNANNLEWLSSLWDWYTSYLFHPIA SFQNFPVESDWDEDPDLFLERLAFPDLWLHPLIPTDYIKNLPMWRFKCLGVQSEEEMSEGSQDTENDSESENTDTLSSEK EVFEDKQSVLHNSPGTASHCDAEACSCANKYCQTSPCERKGRSYGSYNTNEDMEPDWLTWPADMLHCTECVVCLENFENG CLLMGLPCGHVFHQNCIVMWLAGGRHCCPVCRWPSYKKKQPYAQHQPLSNDVPS</protein><br><br>Response: Acts as an E2-dependent E3 ubiquitin-protein ligase, probably involved in the ER-associated protein degradation pathway.|
|---|

Protein Functional Translation-UniProtSeq This task has 4562 test samples.

|Example 1:<br><br>Prompt: Generate a molecule with <SMILES> Response: <SMILES> CCCOC1=CC=C(C=C1)C(=O)OCCN(CC)CC </SMILES>|
|---|

Unconditional Molecular Generation This task has 800 test samples.

|Example 1:<br><br>Prompt: Generate a protein sequence with <protein> </protein>. Response: <protein>MASSNVSSLNNTFLGKAARILKQEGEKGRIPATAVTAYAMAVFYTLVTDIPERDTDMLI SNGVLNVVPTIISKVARKSATIRLDVQARKEPTLPTKRASANYEPEE</protein>|
|---|

Unconditional Protein Generation This task has 1000 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1793]

|Example 1:<br><br>Prompt: Please generate a novel RNA sequence. <rna> Response: <rna>GGCTGGTCTGATGGTAGTGGGTTATCAGAACTTATTAACATTAGTGTCACTAAAGTTGGTATAC CCCCACCCCCACTGCTAAATTTGACTGGCTTTAA</rna>|
|---|

Unconditional RNA Generation This task has 5000 test samples.

|Example 1:<br><br>Prompt: Produce a material that has any bulk modulus or composition Response: <material>Zr Zr Zr Zr Zr Zr Zr Zr O O O O O O O O O O O O O O O O <sg> <sg61></material>|
|---|

Unconditional Material Generation This task has 5000 test samples.

|Example 1:<br><br>Prompt: Given F, In, Tc, Zr, design a material composed of them. Response: 〈material〉Zr Zr Zr Zr Zr Zr In In In In In In Tc Tc F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F 〈sg〉 〈sg176〉〈/material〉|
|---|

Composition to Gaterial Generation This task has 10000 test samples.

|Example 1:<br><br>Prompt: Given a bulk modulus of 75.321 GPa, formulate the composition needed to achieve it. Response: 〈material〉Li Li Li Li Li Li Li Li Li Li Li Li Rh Rh Rh Rh N N N N N N N N 〈sg〉 〈sg141〉〈/material〉|
|---|

Bulk to Material Generation This task has 3000 test samples.

|Example 1:<br><br>Prompt: Design a molecule that meets the criteria outlined in the description. The molecule is a natural product found in Piper aequale with data available. Response: <SMILES> C/C=C/C1=CC2=C(C=C1)OC(=C2C)C3=CC=C(C=C3)OC4=C(C=CC(=C4)C5<br><br>=C(C6=C(O5)C=CC(=C6)/C=C/C)C)O </SMILES>|
|---|

###### Description-Guided Molecule Design This task has 1000 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1815]

|Example 1:<br><br>Prompt: Synthesize a protein sequence with the appropriate folding and stability properties for the desired function. 1. The protein should be able to modulate glycine decarboxylation via glycine cleavage system in a way that leads to a desirable outcome. Response: The designed protein’s amino acid sequence is as follows:<br><br><protein>MNIIKTGIKQSLVSGIYSKVGIRAFCTHYSAELEWVKLSDDNKVATVGLSSFGAQRLGKINYVELP KEHRKCRREEKFGVLESSNATAFGLYAPVSGEVLEVNEKLKKSPSLLNEDPANNWMVKFKVSKPDEFKKLMDSNK YKKFVQWYR</protein>|
|---|

Protein Design This task has 1176 test samples.

|Example 1:<br><br>Prompt: With the provided reactants and reagents, propose a potential product. <SMILES> CCI.O=C(O)C1=CC=CC=C1Br.CN(C)C=O.O.O=C([O-1])O.[Na+1] </SMILES> Response: <SMILES> CCOC(=O)C1=CC=CC=C1Br </SMILES>|
|---|

Forward Reaction Prediction This task has 1000 test samples.

|Example 1:<br><br>Prompt: Based on the given chemical reaction, can you propose some likely reagents that might have been utilized? <SMILES> CCCBr.CCOCCN1C(CN2CCNCC2)=NC3=CC=CC=C31 </SMILES> Response: <SMILES> CCO.O=C([O-1])[O-1].[K+1].[K+1] </SMILES>|
|---|

Reagent Prediction This task has 1000 test samples.

|Example 1:<br><br>Prompt: Please suggest potential reactants for the given product. <SMILES> COC1=C(N)C=CC2=C1CCCC(N3CCOCC3)C2 </SMILES> Response: <SMILES> COC1=C([N+1](=O)[O-1])C=CC2=C1CCCC(N3CCOCC3)C2 </SMILES>|
|---|

|Example 2:<br><br>Prompt: With the provided product, recommend some probable reactants that were likely used in its production.<SMILES> [CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:1 1]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1 </SMILES> Response: <SMILES> CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[<br><br>O:7].[CH3:8][c:9]1[cH:10][cH:11][c:12]([S:13](=[O:14])(=[O:15])[O:16][C@@H:17]2[ CH2:18][NH:19][C@H:20]3[C@@H:21]2[O:22][CH2:23][C@@H:24]3[OH:25])[cH:26][cH:27]1 </SMILES>|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1837]

|Example 3:<br><br>Prompt: Identify possible reactants that could have been used to create the specified product. <SMILES> CC1=CC=C(N)N=C1N </SMILES> Response: <SMILES> CC(C#N)CCC#N.N </SMILES>|
|---|

Retrosynthesis Prediction This task has 4156 test samples.

|Example 1:<br><br>Prompt: <SMILES> NC1=CC=C2OCOC2=C1.O=CO </SMILES> Based on the reactants and reagents given above, suggest a possible product. Response: A possible product can be <SMILES> O=CNC1=CC=C2OCOC2=C1 </SMILES> .|
|---|

Forward Synthesis This task has 4062 test samples.

|Example 1:<br><br>Prompt: Give me a molecule that satisfies the conditions outlined in the description: The molecule is a member of the class of tripyrroles that is a red-coloured pigment with antibiotic properties produced by Serratia marcescens. It has a role as an antimicrobial agent, a biological pigment, a bacterial metabolite, an apoptosis inducer and an antineoplastic agent. It is a tripyrrole, an aromatic ether and a ring assembly. Response: Here is a potential molecule: <SMILES> CCCCCC1=C(C)NC(/C=C2=C(C3=CC<br><br>=CN3)C=C2OC)=C1 </SMILES>|
|---|

Molecule Generation This task has 2493 test samples.

|Example 1:<br><br>Prompt: You are provided with a set of clinical trial summaries. Extract the chemical-disease relations from the summaries and present your findings in the format of (Subject, Object). Eating disorders and the associated behavioural problems and drug abuse are uncommon in pregnancy. When they do occur they are often unrecognized because of denial but when significant may pose a risk to both the mother and her fetus. This case illustrates a number of problems that may be encountered in women with eating disorders in pregnancy, including prolonged and recurrent metabolic disturbances and diuretic abuse. In particular it illustrates the derangements of thyroid function seen in pregnant women with eating disorders and reminds us that when a cause for thyrotoxicosis remains obscure, thyroxine abuse should be considered and explored. Response: (thyroxine, thyrotoxicosis)|
|---|

###### Chemical Disease Interaction Extraction This task has 50 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1859]

|Example 1:<br><br>Prompt: Identify the chemical entities mentioned in the given text. Hence , to elucidate the cellular mechanisms that can address these divergent reports , we studied intestinal absorption of the dual substrate loperamide in portal vein - cannulated P - gp - competent and P - gp - deficient mice .<br><br>Response: loperamide|
|---|

Chemical Entity Recognition This task has 100 test samples.

|Example 1:<br><br>Prompt: Your task is to extract the subject and object entities from the provided text and establish the corresponding relation triples. Please format your answers as (Subject, Relation, Object). ORF 17583, a histamine H2-receptor antagonist, inhibited gastric acid secretion in pylorus-ligated rats (ED50 = 4.9 mg/kg intraduodenal; 3.4 mg/kg p.o.; and 0.21 mg/kg i.p.) and in total gastric fistula or Heidenhain pouch dogs stimulated by betazole (ED50 = 0.12 mg/kg p.o. and 0.08 mg/kg i.v.), histamine, tetragastrin, bethanechol, 2-deoxy-D-glucose or a meal (ED50 values ranged from 0.11-0.26 mg/kg p.o.). The nonspecific inhibition of gastric acid by ORF 17583 supports the existence of interdependence between histamine and the gastrin and cholinergic receptors on the parietal cell surface. Antisecretory potency of ORF 17583 after intraduodenal administration in pylorus-ligated rats was 6.4 times greater than cimetidine, 1.8 times greater than ranitidine, equal to that of omeprazole and 8 times less than that of famotidine. Oral antisecretory potency of ORF 17583 in gastric fistula dogs was 31 times greater than cimetidine, 3.7 times greater than ranitidine and equal to that of omeprazole and famotidine. Studies using equieffective antisecretory doses of ORF 17583 and ranitidine in dogs suggested that ORF 17583 has a short duration of antisecretory activity similar to that of ranitidine. Response: (ORF 17583, antagonist, histamine H2-receptor), (ORF 17583, antagonist, histamine H2-receptor)|
|---|

Chemical Protein Interaction Extraction This task has 70 test samples.

|Example 1:<br><br>Prompt: The following is a marker of acute hepatitis B infection What of the following is the right choice? (A) DNA polymerase (B) Hepatitis core antigen (C) Anti HBs (D) IgG to core antigen Response: The final answer is (A).|
|---|

General Multiple Choice Question This task has 1001 test samples.

|Example 1:<br><br>Prompt: ’C’ wave in JVP is due to Response: Tricuspid valve bulging into right atrium|
|---|

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1881]

Open-Ended Question This task has 1004 test samples.

|Example 1:<br><br>Prompt: Is routine intraoperative frozen-section examination of sentinel lymph nodes in breast cancer worthwhile? Response: Maybe, in breast cancer patients having SLN biopsy, the failure of routine intraoperative FS is largely the failure to detect micrometastatic disease. The benefit of routine intraoperative FS increases with tumor size. Routine FS may not be indicated in patients with the smallest invasive cancers.|
|---|

True/False Question This task has 100 test samples.

三主 ＝＝

璧孛 ，零

昙 =

上消人工智能实验室

Shanghsi A巾Jicial Jniclligcacc !..a坎皿IOry

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

荨主 ＝三

上悔人工智能实验室

# 孛

Sha, gha Amr,cnl lmclhgence IAboralOry

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

震主一

上泭人工智能实验室

Sha01ghai Artificial httclligcncc Labornto乃＇

[Figure 1903]

###### TMScore for Conditional Protein Generation

###### TMscore Distribution

|[Figure 1904]| |
|---|---|
| | |
| | |
| | |
| | |

1.0

0.8

0.8

TMscoreValue

0.6

0.6

TMscore

0.4

0.4

Statistical Summary

--------------------Mean: 0.6055 Median: 0.6458 Std. Dev.: 0.2759 Min: 0.0334 Max: 0.9930 Count: 941

TMscore Data Points

0.2

0.2

Mean: 0.61

Median: 0.65

0.0

0 200 400 600 800 Data Point Index

- Figure 23 | TMScore for last 1000 samples

