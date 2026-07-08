# arXiv:2606.22138v1[cs.CL]20Jun2026

[Figure 1]

[Figure 2]

###### Towards a Comprehensive Biological Foundation Model Spanning the Modality Matrix of Sequences, Structures, and Language

[Figure 3]

Qizhi Pei1,2∗, Zhimeng Zhou3,4∗, Yi Duan1,2∗, Yiyang Zhao2,3,4, Wei Li5,2, Han Guo2, Liang He6, Chengping Li6, Chang-Yu Hsieh3, Conghui He2, Rui Yan7†, Lijun Wu2† 1Gaoling School of Artificial Intelligence, Renmin University of China, 2OpenDataLab, Shanghai Artificial Intelligence Laboratory, 3Zhejiang University, 4Shanghai Innovation Institute, 5East China Normal University, 6Zhongguancun Academy, 7School of Artificial Intelligence, Wuhan University

We present BioMatrix, the first multimodal foundation model that natively integrates sequences, structures, and natural language for both molecules and proteins within a single decoder-only architecture. Existing biological foundation models pursue native multimodality and broad entity coverage separately: those that fuse multiple modalities under a shared objective remain confined to a single entity type, while those spanning multiple entity types either omit explicit structural modeling or rely on adapter-based designs in which the model cannot natively generate the very modalities it can read. BioMatrix closes this gap by mapping molecular sequences (supporting both SMILES and SELFIES notations), molecular structures, protein sequences, protein structures, and natural language into a shared discrete token space through a unified tokenization scheme, so that all modalities are consumed and produced uniformly under a single next-token prediction objective—without external encoders, projection adapters, or modality-specific output heads. Built upon the Qwen3 language model (1.7B and 4B), BioMatrix is continually pretrained on 304.4 billion tokens spanning general and domainspecific text, sequence and structure views of molecules and proteins, and cross-modal corpora that interleave biomolecular entities with scientific text and link distinct entities through molecule–protein and protein–protein interaction data. After tuning on a comprehensive suite of downstream applications covering 80 tasks across 6 categories—encompassing single-entity and multi-entity understanding and generation tasks across and within modalities—BioMatrix achieves state-of-the-art or competitive performance on 77 out of 80 tasks, demonstrating that a single, natively multimodal generalist model can effectively match or surpass specialized approaches across a wide range of biological tasks.

Correspondence: Rui Yan (rui.yan@whu.edu.cn) and Lijun Wu (wulijun@pjlab.org.cn) Links: GitHub: QizhiPei/biomatrix HuggingFace: QizhiPei/biomatrix

Seq.Und.

1DGen

3D Cond.

Annotation

Seq.Gen.

Name

94.8

95.5

94.5

93.6

85.4 77.7 82.5 82.0

85.2 85.1 85.7

82.2

89.6

98.8

83.2

85.3

Property

85.0

3DGen

49.9

81.8

49.0 29.4

49.2

Backbone

87.6 86.2 86.2 84.9

Mol-Inst.

30.8

35.7

75.3

91.4

91.4

100.0

94.5

67.6

97.0

66.5 54.7

95.6

[Figure 4]

[Figure 5]

96.3

46.2

61.5 60.2 60.0

55.4

64.8

92.5

70.5 63.4

MolQA

Caption

70.2

93.0

59.0

73.8

70.7 63.7

Know.Mine

Co-gen.

97.9

12.0

13.0

65.1 56.4

97.7

Molecule

Protein

12.2

14.7

48.0

39.0

54.8

66.4 67.7

61.7

54.6

75.2

76.4 70.9

61.5

Text2Mol

57.7 62.9

Custom

75.5

63.1

58.4 57.9

49.0 72.1

60.5

50.4

66.7

57.9

56.1

54.7

57.0 84.0

Text2Prot

Inv.Fold

87.0

84.0 93.0

Synthesis

Opt.

Struct. QA

Folding

Editing

BioMatrix-4B BioMatrix-1.7B Best non-BioMatrix Second-best non-BioMatrix

*Equal Contribution. †Corresponding Authors. 1

#### Contents

- 1 Introduction 3
- 2 Unified Multimodal Tokenization 6

- 2.1 Molecule Structure Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2 Protein Structure Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3 Vocabulary Extension . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.4 Embedding Initialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 3 Multimodal Continual Pretraining 11

- 3.1 Pretraining Corpus . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.2 Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4 Instruction Tuning across Biological Tasks 14

- 4.1 Task Taxonomy and Data Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 4.2 Data Format Unification and Prompt Diversification . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.3 Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 5 Evaluation Setup 17
- 6 Molecule Tasks 18

- 6.1 1D Molecular Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 6.2 3D Molecular Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- 7 Protein Tasks 33

- 7.1 1D Protein Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 7.2 3D Protein Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- 8 Interaction Tasks 45

- 8.1 1D Interaction Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- 8.2 3D Interaction Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

- 9 Discussion 48

- 9.1 Why Train Task-Group Variants Rather Than One Monolithic SFT Model . . . . . . . . . . . . . . . 48
- 9.2 SMILES vs. SELFIES: Complementary, Not Interchangeable . . . . . . . . . . . . . . . . . . . . . . 48
- 9.3 Where Scaling from 1.7B to 4B Helps, and Where It Does Not . . . . . . . . . . . . . . . . . . . . . . 49
- 9.4 Native Multimodality Pays Off Most on Cross-Modal and Cross-Entity Tasks . . . . . . . . . . . . 51
- 9.5 Tokenization Is the Bottleneck on Fine-Grained 3D Geometry . . . . . . . . . . . . . . . . . . . . . . 51

- 10 Conclusion 52
- 11 Limitations 52

#### 1 Introduction

Foundation models trained on large-scale data have achieved remarkable success in natural language processing and beyond, demonstrating strong capabilities in reasoning, generation, and cross-task generalization [9, 142, 39, 155]. Extending this paradigm to the biological and chemical sciences holds transformative potential for drug discovery, protein engineering, and molecular design [150, 151, 97]. Biological entities, however, are not defined by a single representation. A molecule or protein may be specified as a sequence, realized as a spatial structure, and interpreted through natural-language descriptions of function, activity, disease relevance, or experimental context. These views play complementary roles: sequences provide compact symbolic identity, structures determine physical interaction and function, and language carries both human design intent and the accumulated knowledge of the literature. Biological tasks also rarely involve one entity in isolation; drug discovery, protein engineering, and mechanism understanding routinely depend on molecule–protein and protein–protein relationships. A general biological foundation model should therefore support both native multimodality—treating sequence, structure, and language as first-class inputs and outputs of the same generative model—and broad entity coverage across molecules, proteins, and their interactions. Despite rapid progress on individual fronts, no existing model meets both requirements at once.

Existing efforts toward such a unified model can be organized along two largely independent axes: how biomolecular content is coupled with natural language, and how broadly different biomolecular entity types are covered. Progress along each axis has so far been pursued in isolation, and the two have yet to be reconciled within a single architecture.

- Axis 1: coupling biomolecular content with natural language. Work along this axis broadly falls into two paradigms. The adapter-based paradigm keeps domain encoders and a language model as separate components, as in BioMedGPT [79] and 3D-MoLM [63]. While this paradigm benefits from specialized encoders, it introduces an input–output asymmetry: biomolecular content enters the LLM as continuous encoder features but must be emitted through the LLM’s text vocabulary, making native generation of non-textual modalities such as molecular or protein structures difficult without external decoders. The native-tokenization paradigm instead discretizes biomolecular content into tokens consumed under the same next-token objective as language, mirroring the shift from late-fusion adapters to early-fusion token streams in vision–language modeling [120, 21, 50]. In biology, this direction is enabled by line notations such as SMILES, SELFIES, and amino acid strings, as well as recent SE(3)-invariant geometry tokenizers for molecules [32] and proteins [42, 124, 114, 99]. Existing models in this paradigm remain partial: BioT5 [95, 96], NatureLM [137], and SciReasoner [132] unify sequences with text but omit explicit structure, while ESM3 [42] includes sequence and structure but covers only proteins.
- Axis 2: covering multiple biomolecular entity types. A parallel line of work pursues unified modeling across biomolecular entities themselves, often without natural language. AlphaFold3 [4] jointly represents proteins, nucleic acids, and molecules for structure prediction, and ODesign [149] extends this direction to all-atom generative design. More recently, LOGOS [61] from the Tongyi team broadens the scope further by encoding diverse scientific objects under a shared grammar, including proteins, antibodies, small molecules, reactions, materials, pockets, and protein–ligand complexes. This grammar-based formulation is complementary to our goal: it emphasizes symbolic scientific object descriptions and contact/constraint patterns, whereas BioMatrix unifies natural language, biomolecular sequences, and explicit structure tokens in a single decoder-only next-token interface. Overall, these models show that broad cross-entity coverage is feasible, but their token spaces and objectives remain largely separate from language-grounded prediction over both sequence and structure. Consequently, design intents must be specified through structured configurations rather than functional descriptions, and the models cannot directly leverage the textual knowledge available in biological literature.

The remaining gap. As Table 1 makes concrete, native multimodality and broad entity coverage have so far been pursued separately: models that fuse sequence, structure, and text are confined to a

- Table 1: Comparison of modality coverage across multimodal biological foundation models. △: structural/spatial interactions are encoded as discrete token sequences rather than explicit 3D coordinates or geometric inputs.

Molecule Protein Text 1D Seq 3D Struct 1D Seq 3D Struct NL

Model

ESM3 [42] ✗ ✗ ✓ ✓ ✓ 3D-MoLM [63] ✓ ✓ ✗ ✗ ✓ AlphaFold3 [4] ✓ ✓ ✓ ✓ ✗ ODesign [149] ✗ ✓ ✓ ✓ ✗ BioT5/BioT5+ [95, 96] ✓ ✗ ✓ ✗ ✓ BioMedGPT [79] ✓ ✗ ✓ ✗ ✓ NatureLM [137] ✓ ✗ ✓ ✗ ✓ SciReasoner [132] ✓ ✗ ✓ ✗ ✓ LOGOS [61] ✓ △ ✓ △ ✗

###### BioMatrix ✓ ✓ ✓ ✓ ✓

single entity type, while models that span multiple entity types either lack natural language or omit explicit structural modeling. With mature sequence, text, and SE(3)-invariant geometry tokenizers now available, the remaining challenge is integration: reconciling heterogeneous tokenizers into one shared vocabulary and curating supervision dense enough to bind sequences, structures, language, and cross-entity relationships under a unified next-token objective.

We close this gap with BioMatrix, a multimodal foundation model that, to our knowledge, is the first to natively integrate sequence, structure, and natural language for both molecules and proteins within a single decoder-only architecture. Built upon Qwen3 [142] (1.7B and 4B), BioMatrix serializes all supported modalities into one discrete token stream: SMILES and SELFIES for molecular sequences, SELFIES-aligned tokens for molecular conformations, residue-level sequence tokens and GCP-VQVAE structure tokens for proteins, and the original language-model vocabulary for natural language. Modality-specific boundary tokens allow these segments to be freely interleaved, so tasks that are usually architecturally distinct—captioning, text-conditioned design, folding, inverse folding, structure generation, and interaction prediction—become different conditional generation patterns under one next-token objective.

Several design choices make this interface practical rather than merely conceptual. For molecules, we adapt MolStrucTok [32] with a branch-decoupled decoder and compose frequent SELFIES–structure pairs into joint tokens, binding chemical identity to local geometry without exploding the vocabulary. For proteins, we retain a factored representation between residue tokens and GCP-VQVAE [99] structure tokens, preserving residue-level sequence–structure alignment while avoiding a prohibitively large joint residue–structure vocabulary. To train the resulting model, we construct a 304.4-billion-token corpus spanning scientific text, molecule- and protein-centric multi-view data, interleaved biomedical text, and molecule–protein/protein–protein interaction data, using stochastic composition and protein three-view instances to expose both isolated and jointly grounded modality views.

We evaluate the resulting backbone after instruction tuning on 80 tasks across 6 categories, covering molecular generation and prediction, molecule–text translation, molecular conformer generation, protein understanding and design, protein structure tasks, and biomolecular interaction prediction. BioMatrix achieves state-of-the-art (SOTA) or competitive performance on 77 out of 80 tasks, with the largest gains on settings that exercise the unified design, including text-based molecule generation, property-conditioned molecular conformer generation, protein sequence–structure co-generation, and structure-grounded molecule–protein affinity prediction. Our main contributions are as follows:

[Figure 6]

[Figure 7]

Text Token Molecule 1D Token Molecule 3D Token Protein 1D Token Protein 3D Token

The molecule is a 2hydroxy fatty acid anion that is the conjugate base of 2-hydroxy-3methylpentanoic acid …

Protein 3D De-Tokenizer

Text De-Tokenizer

Molecule 3D De-Tokenizer

… …

### BioMatrix Backbone (Qwen3)

…

…

|Unified Vocabulary (Shared Latent Space)|
|---|

Protein 1D Tokenizer

Text Tokenizer

Molecule 1D Tokenizer

Text Tokenizer

Text Tokenizer

Prompt: Generate a molecule 3D structure with HOMO energy (homo) = -0.2530.

###### Prompt:

Prompt: Please provide a brief introduction to this molecule.

VLSEGEWQLV LHVWAKVEAD VAGHGQDILI …

[C][C][C][Branch1][C] [C][C][Branch1][C][O] [C][=Branch1][C][=O][O-1]

Given a protein amino acid sequence, predict its 3D structure.

- Figure 1: Overview of BioMatrix. Natural language, molecular line notations (SMILES, SELFIES), molecular structures, protein sequences, and protein structures are mapped into a shared discrete token space via a unified tokenization scheme, processed by a single Qwen3-based decoder backbone under next-token prediction, and decoded back to their respective modalities through modality-specific de-tokenizers.

- • A unified architecture along both axes. We propose BioMatrix, the first foundation model that simultaneously achieves native multimodality (sequences, structures, and natural language) and broad entity coverage (molecules, proteins, and their interactions) within a single decoder-only architecture, supporting both single- and multi-entity understanding and generation under one nexttoken prediction objective.
- • A unified tokenization scheme. We design a tokenization scheme that consolidates molecular sequences (in both SMILES and SELFIES), protein sequences, molecular and protein structures, and natural language into a single shared discrete vocabulary, enabling heterogeneous biomolecular inputs and outputs to be handled uniformly without external encoders or projection adapters.
- • Empirical evidence at scale. We curate a 304.4-billion-token pretraining corpus spanning withinmodality and cross-modality data, and instruction-tune across 80 tasks in 6 categories. Built on a single unified pretrained backbone, BioMatrix matches or surpasses specialized baselines on 77 out of 80 tasks, providing the first empirical evidence that the two axes above can be unified without compromising per-task performance.

###### Molecule Structure Tokenizer Protein Structure Tokenizer

###### 1

###### 2

###### 3

###### 4

###### 1

###### 2

###### 3

###### 4

3D molecule conformer

Local spherical descriptors

Vector quantization

Molecule 3D tokens

Structure encoder (GNN / GCPNet)

Vector quantization

Protein 3D tokens

Protein backbone / 3D structure

Codebook (512 entries)

Joint 1D + 3D tokens

Codebook (4096 entries)

3D structure tokens

[Figure 8]

<S 21>

###### ...

###### ...

[Figure 9]

[H 3]

r

θ

<S 8>

[Figure 10]

φ

[C 0]

<S 42>

⋮

Quantized vector

Quantized vectors

[#C 6]

<S 17>

###### ⋮

r distance θ polar angle φ azimuth

[#N 497]

1D sequence tokens

Backbone 3D coordinates

Residue / node embeddings

Discrete index

###### Discrete index

<A A> <A G> <A L> ...

[Figure 11]

[Figure 12]

Reconstruct residue representations

Reconstruct local descriptors (r, θ, abs(φ), sgn(φ))

Reconstruct backbone coordinates

Reconstruct 3D structure

Decoder

Decoder

Main tokenization flow Reconstruction (decoder) Molecule pipeline Protein pipeline

- Figure 2: 3D structure tokenization pipelines in BioMatrix. The molecule structure tokenizer converts each 3D conformer into SELFIES-aligned joint 1D–3D tokens through local spherical descriptors and a 512-entry vector-quantized codebook, while the protein structure tokenizer encodes backbone geometry into per-residue structure tokens using a GCPNet-based encoder and a 4096-entry codebook. Dashed arrows indicate the reconstruction paths used by the corresponding decoders.

#### 2 Unified Multimodal Tokenization

A central design goal of BioMatrix is to process sequences, structures, and natural language for both molecules and proteins within a single decoder-only model. To this end, we adopt a unified tokenization scheme that maps all modalities into a shared discrete token space, so that heterogeneous biomolecular inputs can be handled by the same next-token prediction objective.

Concretely, natural language and sequence-like representations of molecules (SMILES, SELFIES) and proteins (amino acid sequences) are tokenized as ordinary token sequences, while molecular conformations and protein backbones are discretized into structural codes by two dedicated structure tokenizers: we build on MolStrucTok [32] for molecules (Section 2.1, where we additionally propose a branch-decoupled decoder) and directly adopt GCP-VQVAE [99] for protein backbones (Section 2.2).

- Figure 2 summarizes these two structure-tokenization pipelines. These structural codes, together with the sequence and language tokens, are then integrated into a single extended Qwen3 vocabulary alongside modality-specific control tokens that delimit each modality segment (Section 2.3). Finally, the embeddings of all newly added tokens are initialized through a description-based scheme (Section 2.4).

###### 2.1 Molecule Structure Tokenizer

For molecular structures, we adopt MolStrucTok [32] as our structure tokenizer, which discretizes continuous atomic coordinates into a compact set of SE(3)-invariant structural codes that can be seamlessly consumed by a language model. We briefly review the components relevant to BioMatrix here, and refer readers to the original paper for full architectural and training details. Compared to the original design, we make two modifications tailored to our setting: we drop the understanding descriptors (bond lengths and bond angles to nearest neighbors) and retain only the generation descriptors, which we find sufficient for faithful structure reconstruction; and we replace the original monolithic decoder with a branch-decoupled variant that disentangles the heterogeneous geometric targets and

noticeably improves reconstruction fidelity. We describe both changes alongside the corresponding components below.

Geometric input. MolStrucTok represents a molecular conformer as an augmented line notation in which every atom token is paired with a local geometric descriptor on top of its chemical identity. Following the order induced by the underlying SELFIES sequence, each atom vi is assigned a local spherical coordinate frame anchored at a focal atom and two reference atoms, all selected via 2D topological proximity to vi. Within this frame, the position of vi is encoded by three scalars: the radial distance di to the focal atom, the polar angle θi, and the azimuthal angle φi. Because the frame is constructed purely from intrinsic graph topology and relative geometry, the resulting descriptors are inherently invariant to global translation and rotation. Since θi ∈ [0, π] and φi ∈ (−π, π], to normalize their ranges, φi is further decomposed into its absolute value and sign, yielding a four-dimensional input

gi = (di, θi, abs(φi), sign(φi)). (1) As noted above, we omit the understanding features (bond lengths and bond angles to the four nearest neighbors) used in the original MolStrucTok and keep only these generation descriptors gi.

Encoding pipeline. The continuous descriptors gi are discretized by a VQ-VAE [123]. After lognormalizing the length component and rescaling the angular components to a unit range, an MLP

encoder maps each gi into a latent embedding, which is then assigned to its nearest entry in a learnable codebook of size 512. The selected code index serves as the structural token for atom vi, yielding a per-atom sequence of discrete geometric codes whose order follows that of the underlying SELFIES.

Decoding pipeline and branch-decoupled decoder. At decoding time, each codebook entry is passed through a decoder that reconstructs the four-dimensional descriptor gi, from which 3D coordinates are recovered by sequentially traversing the molecule and reconstructing each local spherical frame. The original MolStrucTok decoder regresses all four components with a shared trunk followed by a single output head. We observe, however, that the four entries of gi are heterogeneous in both physical meaning and numerical scale—di is a positive length, θi and abs(φi) are bounded angles, and sign(φi) is a binary indicator—and that forcing a shared trunk to jointly regress all of them introduces interference among these targets and limits reconstruction accuracy. We therefore replace the monolithic decoder with a branch-decoupled variant: the shared trunk is removed, and each of the four targets—length, polar angle, azimuthal magnitude, and azimuthal sign—is predicted by its own dedicated MLP head operating directly on the quantized code. A symmetric modification on the encoder side yielded no measurable benefit, suggesting that the bottleneck lies in disentangling heterogeneous outputs rather than in encoding them. With this change alone, the average percomponent RMSD improves by roughly 0.1A˚ on QM9-2014 [101] over the original tokenizer, while leaving the codebook size, encoder, and overall token interface unchanged.

Integration into BioMatrix. For each molecule, we run the MolStrucTok encoder over its conformation to obtain a per-atom sequence of discrete structural codes drawn from the 512-entry codebook, aligned one-to-one with the corresponding SELFIES atom tokens. Their integration into BioMatrix’s joint vocabulary is described in Section 2.3.

###### 2.2 Protein Structure Tokenizer

For protein structures, we adopt GCP-VQVAE [99] as-is, without architectural modification, since it already offers strong reconstruction fidelity and generalization to unseen proteins among publicly available open-source structure tokenizers at comparable codebook sizes. Specifically, we use its Large variant throughout BioMatrix. Conceptually, GCP-VQVAE plays the same role for proteins

that MolStrucTok plays for molecules: it converts a continuous backbone into a sequence of discrete, pose-invariant structural symbols, and provides a decoder that maps those symbols back to backbone coordinates. The key difference is the unit of tokenization. Rather than assigning one code per atom, GCP-VQVAE assigns one code per residue, so a protein chain of length L becomes a length-L sequence of structural tokens aligned with the amino acid sequence.

Geometric input. GCP-VQVAE operates on protein backbone coordinates, taking the (N, Cα, C) atoms of each residue as input. Each residue is treated as a node in a protein graph, with geometric features derived from the relative positions and orientations of neighboring backbone frames rather than from absolute Cartesian coordinates. This is important because a structure token should describe the local fold of a residue, not the arbitrary global rotation or translation of the protein in a coordinate file. By restricting itself to backbone geometry, the tokenizer captures the global fold and local secondary-structure arrangement of a protein while remaining agnostic to side-chain conformations, which keeps the discrete vocabulary compact and broadly applicable to both experimentally resolved and predicted structures.

Encoding pipeline. The backbone graph is first processed by a strictly SE(3)-equivariant GCPNet encoder [88]. GCPNet maintains scalar features together with vector features, and performs message passing in local geometric frames so that the representation can retain directional and chiral information while remaining well behaved under global rigid motions. The resulting per-residue embeddings summarize each residue’s local backbone environment as well as information propagated from nearby residues. A transformer encoder then refines these residue embeddings with global sequence-level context. Finally, a vector-quantization layer replaces each continuous residue embedding with the nearest entry in a learnable codebook of size 4,096. The selected code index is the protein structure token for that residue. Thus the encoder maps

{(xiN,xiCα,xiC)}iL=1 −→ (z1, . . . , zL), zi ∈ {1, . . . ,4,096}, (2) where each zi is a discrete summary of the backbone geometry around residue i.

Decoding pipeline. The decoder performs the inverse operation. Each discrete code zi is first mapped back to its corresponding codebook vector, producing a sequence of continuous embeddings with the same length as the protein chain. A transformer decoder then predicts a rigid frame for each residue, parameterized by a translation and a rotation represented with a continuous 6D rotation head. These predicted residue frames are applied to an idealized local backbone template containing the N, Cα, and C atoms, yielding reconstructed backbone coordinates for the full chain. In this sense, the code sequence does not directly store Cartesian coordinates; it stores enough local geometric information for the decoder to recover a globally coherent backbone after the residue-level frames are assembled. Training uses reconstruction losses on the recovered coordinates and geometry together with the standard VQ codebook and commitment objectives, so the learned codes form a compact geometric vocabulary that can be generated autoregressively by BioMatrix.

Integration into BioMatrix. For each protein, we run the GCP-VQVAE Large encoder over its backbone structure to obtain a length-L sequence of discrete per-residue structural tokens drawn from the 4,096-entry codebook, one token per residue. Because this sequence is residue-aligned, it can be paired directly with the dedicated amino-acid tokens introduced in Section 2.3: folding becomes generation of z1, . . . , zL from amino-acid tokens, while inverse folding becomes generation of aminoacid tokens from z1, . . . , zL. At output time, generated structure-token sequences are passed through the GCP-VQVAE decoder to recover backbone coordinates. Their integration into BioMatrix’s joint vocabulary is described in Section 2.3.

###### 2.3 Vocabulary Extension

Category Modality Example

1D SMILES <|mol smi start|> C # CC # N <|mol smi end|> 1D SELFIES <|mol sfi start|> [C] [#C] [C] [#N] <|mol sfi end|> 3D Structure <|mol 3d start|> [H 3] [C 0] [#C 6] [C 132] [#N 497] <|mol 3d end|>

Molecule

1D AA Sequence <|prot aa start|> <A M> <A R> <A A> <A K> <A W> <|prot aa end|> 3D Structure <|prot 3d start|> <S 4012> <S 153> <S 2091> <S 2440> <S 3117> <|prot 3d end|>

Protein

- Table 2: Examples of wrapped sequences for each modality in the extended vocabulary. Each molecule or protein segment is delimited by a dedicated pair of start/end special tokens, while natural language text is left unwrapped and serves as the default carrier modality. Each light-blue box corresponds to a single token produced by the tokenizer on the wrapped segment. Molecular 3D tokens are formed by composing each SELFIES token with its 3D code from MolStrucTok [32]; protein 3D tokens form a standalone set produced by GCP-VQVAE [99].

To accommodate the full spectrum of biomolecular modalities within a single decoder-only model, we extend the original Qwen3 vocabulary along two orthogonal axes: modality-specific tokens that encode molecular and protein sequence and structure representations, and structural control tokens that delimit each modality segment within a mixed-modality sequence. Table 2 summarizes the full set of additions, which we describe below.

1D tokens. For natural language, we directly reuse the Qwen3 tokenizer without modification. For molecules, SMILES strings are tokenized directly by the Qwen3 tokenizer since SMILES is a widely used line notation that Qwen3 has already been exposed to during pretraining. SELFIES, in contrast, is a more recent and less prevalent notation whose bracketed atomic symbols are not natively recognized, and we add them as new vocabulary entries. We retain both SMILES and SELFIES as parallel molecular line notations rather than committing to a single one since our downstream experiments (Section 6.1) reveal that the two notations exhibit complementary strengths. For proteins, we introduce a dedicated vocabulary of 26 residue-level tokens covering the 20 standard amino acids together with non-standard and unknown residue symbols. Although amino acid letters already lie within Qwen3’s character set, its tokenizer merges them into variable-length subword pieces, which would break the one-to-one correspondence between residues and the per-residue 3D tokens produced by GCP-VQVAE (Section 2.2). Enforcing single-token-per-residue encoding preserves this alignment and directly supports downstream tasks that operate on residue-level sequence–structure correspondences, such as folding (mapping each AA token to its structural token) and inverse folding (the reverse direction).

- 3D tokens. The molecular and protein structure vocabularies are inherited from the two structure tokenizers introduced in Sections 2.1 and 2.2, but we integrate them into BioMatrix’s vocabulary in two different ways, guided by how many joint sequence–structure entries each entity type would actually need.

For molecules, we compose each MolStrucTok code with its aligned SELFIES atom token into a joint entry, following the original MolStrucTok design, so that every structure-tokenized molecule token simultaneously carries chemical identity and local geometry. In principle, pairing the 925 unique SELFIES atom tokens observed in our pretraining corpus with the 512 MolStrucTok codes could yield up to ∼470K combinations, but in practice the vast majority never occur. We therefore enumerate joint tokens from the pretraining data in descending order of frequency and greedily retain

the smallest prefix that covers more than 99% of occurrences, which yields 11,294 joint entries. This composed vocabulary is compact enough to be learned reliably while keeping tail combinations, whose embeddings would be under-trained, out of the vocabulary.

For proteins, we keep the 4,096 GCP-VQVAE codes as a standalone set of per-residue structure tokens, parallel to the 26 residue-level amino-acid tokens (standard amino acids plus non-standard and unknown symbols). Applying the same 99%-coverage procedure to AA–structure pairs would require 71,737 joint entries (nearly half the size of the original Qwen3 vocabulary, 151,669), and a long tail of these combinations occurs only a handful of times in the pretraining corpus, leaving their embeddings severely under-trained. We therefore opt for a factored representation in which each residue is described by an AA token and a structure token independently.

Structural control tokens. To let the model distinguish heterogeneous content within a single token stream, we further introduce five pairs of start/end special tokens that mark the boundaries of each non-text biomolecular segment (rightmost column of Table 2). Natural language is left unwrapped and serves as the default carrier modality, while molecular and protein segments, whether sequence- or structure-based, are explicitly enclosed by their corresponding control-token pair. This design provides a clean interface between continuous prose and structured biomolecular content, and allows text, molecular representations, and protein representations to be interleaved within one training instance under BioMatrix’s unified next-token prediction framework.

###### 2.4 Embedding Initialization

Extending Qwen3’s vocabulary with the modality-specific tokens and structural control tokens introduced in Section 2.3 adds several thousand new entries to both the input embedding table and, when not tied, the output projection. Initializing these new rows from scratch (e.g., Gaussian noise) discards the well-trained semantic geometry of the underlying language model and slows down the early stages of continual pretraining. We therefore adopt a simple description-based initialization scheme that grounds every newly added token in the pretrained embedding space of Qwen3 itself.

Description-based initialization. For each new token t, we first construct a short natural-language description desc(t) that summarizes its semantic role. The description is then tokenized by the original Qwen3 tokenizer into a sequence of subword ids (s1, . . . , sk), and the new embedding is initialized as the mean of the corresponding pretrained input embeddings, plus a small isotropic Gaussian perturbation:

k

1 k

##### ∑

EQwen3[sj] + ϵ, ϵ ∼ N 0, 1d I , (3)

et =

j=1

where EQwen3 denotes the pretrained Qwen3 input embedding matrix and d is the embedding dimension. The noise term breaks symmetry between tokens whose descriptions happen to share subword ids, while keeping each new embedding within the typical scale of the pretrained vocabulary.

Per-category descriptions. The description desc(t) is constructed in a category-specific manner that reflects how much prior semantic content each token actually carries:

- • Structural control tokens are mapped to short phrases describing the modality boundary they mark, e.g., <|mol smi start|> → Begins a molecule SMILES string, <|prot 3d end|> → Ends a protein 3D structure. This grounds each delimiter in the same semantic space as the surrounding natural-language context.

- • Protein 1D tokens are mapped to the full English name of their corresponding amino acid (e.g., <A A> → Alanine, <A W> → Tryptophan), with non-standard or ambiguous symbols mapped to descriptive phrases such as Selenocysteine or any amino acid.

- • Molecular 1D, molecular 3D, and protein 3D tokens have no informative natural-language gloss—a SELFIES atom token or a VQ codebook index is essentially an opaque symbol. For these categories, we use the raw token string as its own description, which yields a near-uniform initialization centered on the subword decomposition of the token literal while still benefiting from the noise term to break ties.

Discussion. This scheme requires no additional training or auxiliary models: it reuses the frozen Qwen3 embedding table as a fixed semantic encoder for the description strings, and writes the resulting vectors directly into the expanded embedding rows before continual pretraining begins. In our preliminary experiments, we find that this description-based initialization provides a more stable starting point than random initialization in the early phase of continual pretraining.

#### 3 Multimodal Continual Pretraining

To equip BioMatrix with joint understanding over sequences, structures, and language for both molecules and proteins, we perform continual pretraining on a large-scale, heterogeneous corpus that spans general and scientific text, molecular and protein data with both sequence and structure views, and cross-entity resources linking distinct biomolecular entities or wrapping them into free-form scientific text. All modalities are serialized into a shared token space via the unified tokenization scheme described in Section 2, with each modality segment explicitly delimited by the modality-specific start/end tokens. In this section, we first describe the composition and curation of the pretraining corpus (Section 3.1), and then report the training dynamics of BioMatrix under this data mixture

- (Section 3.2).

###### 3.1 Pretraining Corpus

We organize the corpus into four categories: (i) general- and scientific-domain text (Section 3.1.1); (ii) molecule-centric data, where within-molecule sequence, structure, and textual views are jointly exposed (Section 3.1.2); (iii) protein-centric data, organized analogously to the molecular case (Section 3.1.3); and (iv) cross-entity and interleaved data that link distinct biomolecular entities or wrap them into scientific prose (Section 3.1.4).

For molecular and protein data, we adopt two complementary strategies that operate at different levels of instance construction: stochastic composition governs how fields are sampled and ordered within a single instance, while the three-view instance pattern controls how multiple instances are emitted from the same biomolecular entity. Stochastic composition applies to every molecule- and protein-centric instance. For each instance, the inclusion of optional fields is controlled by predefined probabilities, and the ordering of interchangeable components (e.g., property fields, annotation entries, or co-present modalities) is randomly permuted across samples. For example, a single PubChem molecule may yield an instance that randomly samples a subset of SMILES, SELFIES, tokenized conformation, IUPAC name, textual description, physicochemical properties and permutes their order. This design prevents the model from overfitting to any fixed input schema and encourages it to jointly learn associations among heterogeneous representations [90]. The three-view instance pattern is applied specifically to protein-centric data sources that provide both sequences and structures. For each such protein, we emit (i) a sequence-only instance containing the amino acid sequence, (ii) a structure-only instance containing the tokenized backbone, and (iii) a joint instance containing both modalities with their relative order randomly permuted, exposing the model to both isolated and jointly grounded views of each protein. Protein data sources providing only sequences emit the sequence view alone; sources that additionally carry functional annotations further emit a separate composite instance that combines those annotations with the available modalities under the stochastic composition strategy. We apply this pattern to proteins because sequence and structure provide largely complementary, residue-level-

Edu

###### Text

PPIRef

FineWeb-

STITCH

USPTO

| | | |25.6B| | |79.7B|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

STITCH

BindingDB

General

Text 105.3B

Cross-entity 48.0B

FineFineWeb

S2ORC

Scientific

0 20 40 60 80

PubMed

Tokens (Billion)

Protein

Swiss-Prot

36.0B

1D

BioRxiv

38.5B

3D

AFDB

2.9B

Other

Token

0 10 20 30 40

Tokens (Billion)

304.4B

Molecule

32.1B

1D

PubMed

RCSBPDB

17.6B

3D

Other

24.0B

TrEMBL

0 5 10 15 20 25 30 35 40

Tokens (Billion)

PubChem

Cross-entity

Protein 77.4B

Molecule 73.7B

1D

17.1B

UniRef50

MolTextNet PCQM4Mv2

3D

11.4B

ZINC20

PubChemQC

Other

19.5B

0 5 10 15 20

Tokens (Billion)

- Figure 3: Overview of the continual pretraining corpus composition. Token budget distribution across four data categories — general and scientific text, molecule-centric data, protein-centric data, and cross-entity/interleaved data — spanning sequence, structure, and textual modalities, totaling 304.4 billion tokens.

aligned views of the same entity, and we want the model to handle each modality both independently and jointly.

In total, our pretraining corpus comprises approximately 304.4 billion tokens, with text and protein data contributing the largest shares, followed by molecular and cross-entity data. Figure 3 summarizes the full token budget across categories and modalities.

###### 3.1.1 Text Corpora

General-domain text. We use FineWeb-Edu [76], a large-scale, education-oriented filtered subset of CommonCrawl, as our source of general-domain natural language, which helps prevent catastrophic forgetting of the model’s general language abilities during continual pretraining.

Scientific-domain text. To strengthen the model’s scientific language proficiency, we curate text from two complementary sources. From FineFineWeb [81], a domain-categorized web corpus, we select the four subsets most relevant to our target disciplines: biology, chemistry, medicine, and health. To complement web-sourced text with more rigorous scientific writing, we additionally extract a large collection of full-article texts from PubMed [135], covering peer-reviewed literature in molecular biology, pharmacology, and clinical research.

###### 3.1.2 Molecular Data

PubChem. We collect all compounds available in PubChem [54] as of December 2025 and construct multi-view training instances that jointly expose the model to structure, properties, and annotations. For each molecule, we extract up to three representations—canonical SMILES, SELFIES, and a tokenized conformation produced by our molecular structure tokenizer (Section 2.1)—together with available physicochemical properties (e.g., molecular weight, hydrogen bond donor/acceptor counts, topological polar surface area, XLogP3) and textual annotations such as IUPAC names and natural language descriptions. These heterogeneous components are concatenated into a single training instance following the stochastic composition strategy described above. Since PubChem’s conformer coverage

is limited and its conformations are generated by lightweight force fields, we further incorporate DFT-level conformations from PCQM4Mv2 [45] and PubChemQC [83], processed through the same structure tokenizer and merged into the corpus under the same composition strategy.

MolTextNet. To strengthen molecule–text alignment, we incorporate MolTextNet [154], a corpus of approximately 2.5M molecule–description pairs derived from ChEMBL35 [148], whose descriptionssynthesized by GPT-4o-mini from structural, physicochemical, bioactivity, and synthesizability annotationsare substantially longer and more informative than prior molecule–text resources. For each entry, we canonicalize the SMILES, derive the corresponding SELFIES, and assemble the three modalities into a single training instance under the same stochastic composition strategy.

###### 3.1.3 Protein Data

UniRef50. We include protein sequences from UniRef50 [118] to provide broad coverage of the protein sequence space.

RCSB PDB. We incorporate experimentally determined protein structures from the RCSB Protein Data Bank [13]. For each chain, we extract the amino acid sequence and discretize its backbone geometry into structural tokens via our protein structure tokenizer (Section 2.2), then apply the three-view instance pattern.

UniProt/Swiss-Prot. We use the manually curated Swiss-Prot subset of UniProt [1] as our primary source of annotated proteins, pairing each entry with its predicted structure from the latest release of the AlphaFold Protein Structure Database (AFDB) [14] when available. Beyond sequence and structure, Swiss-Prot provides rich functional metadata: protein name, description, organism, taxonomic lineage, subcellular localization, keywords, and free-text comments on function, catalytic activity, and biological context. For each protein, we serialize these annotations together with the available sequence and structure views into a composite training instance, and additionally emit the three-view instance pattern so that the model is exposed to both isolated and jointly grounded views.

UniProt/TrEMBL. To scale up coverage of the protein sequence–function space, we further include the automatically annotated TrEMBL subset of UniProt [1]. As most TrEMBL entries lack resolved or predicted structures at our processing time, we retain only the amino acid sequence together with its functional annotations, with evidence codes and literature citations removed. TrEMBL provides broad taxonomic and functional coverage that complements the smaller but more richly grounded Swiss-Prot samples.

AlphaFold Protein Structure Database. Beyond the Swiss-Prot predictions already paired with UniProt annotations, we incorporate two additional curated subsets from the AFDB [14]: the model organism proteomes (covering 16 reference species including human, mouse, E. coli, and S. cerevisiae) and the global health proteomes (covering pathogens relevant to neglected tropical and infectious diseases). To further scale structural diversity, we additionally sample approximately 130M predicted structures from the full AFDB, representing over 60% of all entries in the database and covering a substantial fraction of the known protein universe. All entries are processed through the same structure tokenizer and emit the three-view instance pattern.

###### 3.1.4 Cross-Entity and Interleaved Data

While Sections 3.1.2 and 3.1.3 already expose the model to within-entity cross-modal alignment (e.g., a molecule paired with its own conformation and description), this section focuses on data that link distinct biomolecular entities to each other or wrap them into free-form scientific text.

Interleaved biomedical text. To encourage tight cross-modal grounding between natural language and biomolecular representations, we construct an interleaved corpus in which biomedical entities mentioned in scientific text are directly paired with their structural representations. Source documents are drawn from four complementary corpora—PubMed abstracts [135], bioRxiv abstracts [109], S2ORC [74], and the USPTO-Applications patent corpus [19, 75]—jointly covering biomedical literature, preprints, broad scientific publications, and chemical patents. We apply BERN2 [117], a neural biomedical named entity recognition and normalization tool, to detect mentions of molecules and proteins and link them to canonical database identifiers. For each recognized entity, we retrieve the corresponding structural representations—SMILES and SELFIES for molecules, amino acid sequences for proteins—and append them inline after the entity mention, wrapped with the appropriate special tokens. This exposes the model to natural language contexts in which textual descriptions co-occur with their underlying structural representations, facilitating fine-grained alignment between biomedical terminology and biomolecular entities.

Biomolecular interaction data. To equip the model with knowledge of biomolecular interactions, we incorporate curated interaction datasets spanning both molecule–protein and protein–protein pairs in sequence-only and structure-grounded forms. For molecule–protein interactions, we collect sequence-level pairs from BindingDB [69], STITCH [89], and the jglaser binding affinity dataset [34], and structure-grounded complexes from CrossDocked2020 [31]. For protein–protein interactions, we use AlphaSeq [2] as the sequence-level source and PPIRef [17] as the structure-grounded source.

###### 3.2 Training Dynamics

We conduct continual pretraining on top of the LLaMA-Factory [152] framework. We train both BioMatrix-1.7B and BioMatrix-4B on 64 NVIDIA H100 GPUs with a global batch size of 1,024 and a maximum sequence length of 8,192 tokens, using the AdamW optimizer with a peak learning rate of 2.0 × 10−4 under a cosine schedule with 2,000 warmup steps. Training runs for approximately 36.4K steps (1 epoch), consuming the full 304.4B-token corpus described in Section 3.1. The resulting training loss curves are shown in Figure 4.

On loss spikes. A closer inspection of the curves reveals several transient loss spikes for both model sizes, concentrated primarily in the early-to-middle phase of training. In all cases, the loss recovers to its pre-spike trajectory within a small number of steps without any manual intervention (e.g., learning-rate rewinding or optimizer-state rollback). To assess whether these spikes cause any latent degradation that is not visible in the training loss itself, we evaluated checkpoints before and after each prominent spike on a held-out validation set. Across all evaluated spikes, we observe no meaningful degradation: pre- and post-spike checkpoints achieve essentially comparable validation loss, and subsequent checkpoints continue to improve smoothly as training proceeds.

Given that the diversity of modalities inherently increases the risk of optimization instability, we interpret the observed spikes as localized optimization perturbations (most likely triggered by occasional high-gradient batches arising from the highly heterogeneous mixture of text, sequence tokens, and structure tokens) rather than genuine deterioration of the learned representations.

#### 4 Instruction Tuning across Biological Tasks

After multimodal continual pretraining, we perform supervised instruction tuning to align BioMatrix with a broad spectrum of downstream biological tasks and to elicit its ability to follow natural-language instructions over heterogeneous biomolecular inputs and outputs. In this section, we first describe the task taxonomy and data sources of the instruction-tuning corpus (Section 4.1), then detail how

3.0

1.7B

2.8

4B

2.6

2.4

Trainingloss

2.2

2.0

1.8

1.6

1.4

0 5000 10000 15000 20000 25000 30000 35000 Training steps

Figure 4: Training loss curves of BioMatrix-1.7B and BioMatrix-4B during multimodal continual pretraining. Both models converge stably despite occasional transient loss spikes, which leave downstream performance essentially unaffected.

heterogeneous datasets are unified into a consistent instruction format with diversified prompts

- (Section 4.2), and finally report the training configuration (Section 4.3).

###### 4.1 Task Taxonomy and Data Sources

We curate a comprehensive instruction-tuning corpus that spans three biological entity scopesmolecule, protein, and interaction—and, within each scope, covers both sequence-based and structurebased tasks. Table 3 summarizes the full task taxonomy, comprising 80 tasks drawn from a diverse collection of public benchmarks; this taxonomy is used consistently throughout the paper, with the paragraphs below giving a scope-level overview and Section 6, 7, and 8 reporting evaluation results organized along the same sub-task structure. Table 4 reports the overall training data statistics grouped by domain and modality, and a detailed per-sub-task breakdown is further provided in Table 31.

Molecular tasks. For molecules, we cover a broad range of sequence-level tasks, including unconditional generation, name conversion, property prediction, captioning, text-based generation, forward and retrosynthesis, editing, customized generation, optimization, and question answering. For molecular structures, we include unconditional and property-conditioned conformer generation tasks.

Protein tasks. For proteins, we cover sequence understanding, annotation prediction, knowledge mining, text-based protein design, and unconditional sequence generation. For protein structures, we include structure understanding, folding and inverse folding, sequence–structure co-generation, and unconditional structure generation.

Interaction tasks. Beyond single-entity tasks, we additionally include molecule–protein and protein– protein interaction tasks in both sequence-only and structure-grounded forms, covering binding affinity prediction and interaction modeling at both levels.

Table 3: Overview of downstream tasks used for instruction tuning. The table defines 6 task categories spanning three domains (molecule, protein, interaction) across 1D and 3D modalities. The final entry (MegaScience, Text domain) is included only during instruction tuning to preserve general scientific reasoning capability, and is not part of the evaluation suite.

Domain Modality Task Datasets

Unconditional Sequence Generation MOSES [98], GuacaMol [16] Name Conversion SMolInstruct [146] Property Prediction SMolInstruct [146]

Molecule Captioning SMolInstruct [146], CheBI-20 [24], KnowMol-100K [143] Text-Based Molecule Generation SMolInstruct [146], KnowMol-100K [143]

1D

Molecule

Forward Synthesis SMolInstruct [146]

Retrosynthesis SMolInstruct [146], USPTO-50K [106] Molecule Editing OpenMolIns/S2-Bench [60]

Molecule Optimization OpenMolIns/S2-Bench [60], MolOpt-Instructions [144] Customized Molecule Generation OpenMolIns/S2-Bench [60]

Molecule Question Answering MoleculeQA [78] 3D

Unconditional Structure Generation QM9-2014 [101] Conditional Structure Generation QM9-2014 [101]

Sequence Understanding Open Protein Instructions [138] Annotation Prediction Open Protein Instructions [138], Mol-Instructions [28] Knowledge Mining Open Protein Instructions [138]

1D

Text-Based Protein Design CAMEO [41], Molinst-SwissProtCLAP [28] Unconditional Sequence Generation DPLM-2 [131]

Protein

Structure Understanding PFUD [82] Structure Prediction (Folding) DPLM-2 [131]/PDB Date [131]

3D

Inverse Folding DPLM-2 [131]/PDB Date [131] Sequence–Structure Co-Generation DPLM-2 [131]

Unconditional Backbone Generation DPLM-2 [131]

Molecule–Protein Interaction BindingDB [68], PDBBindv2019 [70], CASF-2016 [116]

1D

Interaction

Protein–Protein Interaction Yeast [40], Human [94], PPI Affinity [86] 3D Molecule–Protein Interaction PDBBindv2020 [129]

Text – – MegaScience [27]

General scientific reasoning. Beyond the biology-specific tasks enumerated in Table 3, we additionally incorporate MegaScience [27] into the instruction-tuning mixture as a source of general scientific reasoning supervision. Including this corpus helps preserve BioMatrix’s broad scientific reasoning and instruction-following capabilities inherited from the base Qwen3 model, and prevents them from being eroded by the heavy concentration of domain-specific biomolecular tasks.

###### 4.2 Data Format Unification and Prompt Diversification

The collected datasets exhibit substantial heterogeneity in their original formats, and only a fraction is natively distributed in instruction-style format. To bring them into a uniform interface compatible with BioMatrix, we apply three transformations as follows.

Modality serialization with special tokens. Whenever an input or output field contains a biomolecular entity, we serialize it using the unified tokenization scheme and wrap it with the corresponding modality-specific start/end tokens, following the same protocol as in continual pretraining (Table 2).

Conversion to instruction format. For datasets that are not natively in instruction form, we manually design conversion pipelines that map each raw example into a (instruction, input, response) triplet. Depending on the task type, classification labels are verbalized into natural-language category names,

Table 4: Training data statistics of instruction tuning. Domain Modality #Train Molecule

1D (SMILES) 9,407,182 1D (SELFIES) 9,425,897

3D 390,928 Protein

1D 3,009,742 3D 1,272,911

1D 67,369 3D 18,013

Interaction

Text – 1,253,230 Total – 24,845,248

regression targets are formatted as numerical strings with task-appropriate units and precision.

Prompt template diversification. Using a single fixed prompt per task tends to produce models that overfit to the surface form of the instruction and generalize poorly to paraphrased or unseen phrasings at inference time. To mitigate this, for each sub-task in Table 3 we manually curate a pool of semantically equivalent prompt templates that vary in wording, sentence structure, and level of formality, while preserving the underlying task intent. Each training instance then samples one template uniformly at random from its task-specific pool to fill the instruction field.

###### 4.3 Training Configuration

Instruction tuning is also performed using the LLaMA-Factory framework [152]. We instruction-tune both BioMatrix-1.7B and BioMatrix-4B using the AdamW optimizer. The peak learning rate is set to

- 5.0 × 10−5 under a cosine schedule with a linear warmup over the first 10% of total steps. We use sample-level attention masking during training. The maximum sequence length is 2,048 tokens.

#### 5 Evaluation Setup

Our evaluation is designed to answer the two empirical questions:

- Q1: Feasibility of unified tokenization. Can biomolecular structures be discretized into tokens that share a vocabulary with sequences and natural language, and decoded under a single next-token objective, while still supporting high-quality structural generation, prediction, and understanding across both molecules and proteins?
- Q2: Competitiveness against task-specialized models. Does a single unified backbone—covering molecules, proteins, and their interactions across sequence, structure, and text within one vocabularymatch or surpass models that are purpose-built or purpose-trained for individual tasks?

To answer these questions, we organize the evaluation along the same taxonomy used for instruction tuning (Table 3): single-entity molecular tasks across sequence and structure settings (Section 6), singleentity protein tasks organized analogously (Section 7), and molecule–protein as well as protein–protein interaction tasks (Section 8).

We emphasize that BioMatrix shares a single tokenization scheme, a single continually-pretrained backbone, and a single next-token training objective across every result reported in this section. The instruction-tuning stage, in contrast, is organized into several task-group-specific variants whose

grouping is determined by data composition and per-task evaluation considerations. We make this division fully explicit in Table 30.

Evaluated variants. All variants of BioMatrix evaluated in this section are instruction-tuned from the same continually-pretrained backbones and use the same unified tokenization scheme. The instructiontuning stage, however, is organized into task-group-specific variants summarized in Table 30, and the numbers reported correspond to these task-group variants. For practical convenience, we release a single all-tasks merged SFT model trained on the union of all sub-task corpora with mild oversampling of small-data sub-tasks (see Section 9.1).

Baselines. For each task, we compare BioMatrix against representative baselines drawn from the prior literature. The selected baselines span two complementary types of comparison: (a) generalist biological foundation models that, like BioMatrix, attempt to cover multiple task categories under a shared framework (e.g., BioT5+ [96], NatureLM [137], SciReasoner [132], 3D-MoLM [63], ESM3 [42], ProtTeX [82]), against which we test whether BioMatrix advances the multi-task frontier within the unified-tokenization paradigm; and (b) task-specialized state-of-the-art models that represent the strongest published numbers on individual benchmarks (e.g., NExT-Mol [73], MolGPT [11], ESMFold [65], ProteinMPNN [23], Uni-Mol [153]), against which we test whether BioMatrix’s task-group variants, derived from the unified backbone, can match dedicated specialists. The specific baselines considered for each task are listed alongside the corresponding results in Sections 6–8. We additionally report both BioMatrix-1.7B and BioMatrix-4B throughout to provide a within-family scale comparison under an otherwise identical recipe.

Evaluation protocol. For every sub-task in Table 3, we follow the official data split and evaluation metric provided by its source benchmark. All inference is performed with vLLM [57]. We adopt two decoding configurations depending on the nature of the task. For tasks whose evaluation does not involve output diversity—such as property prediction, name conversion, annotation prediction, question answering—we use greedy decoding with temperature 0 to obtain a deterministic prediction. For tasks whose evaluation explicitly involves sampling diverse outputs—such as unconditional molecule generation or text-based protein design—we instead use nucleus sampling with top-p = 1.0, and adaptively tune the sampling temperature on a per-task basis to balance fidelity and diversity according to the conventions of each benchmark.

#### 6 Molecule Tasks

We evaluate BioMatrix on molecular tasks spanning both sequence-based and structure-based settings, as summarized in Figure 5. Section 6.1 covers tasks that take or produce molecular line notations together with natural language, while Section 6.2 covers tasks that involve molecular geometry.

- 7 Molecule Editing

Instruction + input molecule

-> edited molecule

- 8 Molecule Optimization

Input molecule + objective

-> improved molecule

- 9 Customized Molecule Generation Constraints -> generated molecule

- 10 Molecule Question Answering

- 1 Unconditional Sequence Generation

No condition

-> valid SMILES / SELFIES

- 2 Name Conversion IUPAC / formula / sequence conversion
- 3 Property Prediction Molecule sequence

-> property value / label

- 4 Molecule Captioning

SMILES / SELFIES

-> natural-language description

- 5 Text-based Molecule Generation

Text requirement

-> molecule sequence

- 11 Unconditional Structure Generation

No condition

-> molecule 3D structure

- 12 Conditional Structure Generation

Target property

-> molecule 3D structure

1D token SMILES / SELFIES 3D token Molecule structure Text Natural-language instruction or answer

[Figure 13]

- 6 Forward & Retrosynthesis Reactants -> product Product -> reactants

[Figure 14]

###### BioMatrix Molecule Interface

SMILES / SELFIES tokens + 3D structure tokens + text prompts

[Figure 15]

[Figure 16]

Question + molecule context

-> text answer

###### Example Unified Token Stream

[C] [=O] [O] ... text prompt [C 0] [H 3] [#N 497] ... text answer

Figure 5: Overview of the molecular task suite evaluated in this section.

###### Key takeaways on molecule tasks.

- • Strong parameter efficiency against larger LLM baselines. Across sequence-level molecular tasks—spanning unconditional generation, name conversion, property prediction, captioning, text-based generation, forward and retrosynthesis, editing, customized generation, optimization, and question answering—BioMatrix achieves the best or second-best result on the vast majority of metrics among LLM-based methods, often surpassing baselines that are several times larger in parameter count.
- • Native multimodality unlocks gains on cross-modal tasks. The improvements over prior specialists are most pronounced on tasks that explicitly bridge modalities or conditioning signals: text-based molecule generation, molecule captioning, and especially property-conditioned conformer generation, where BioMatrix reduces MAE on electronic-structure targets by roughly 3–4× over the strongest prior method. This pattern directly validates the value of placing sequence, structure, and text in a single discrete token space.
- • SMILES and SELFIES are complementary, not interchangeable. By natively supporting both notations under an otherwise identical recipe, BioMatrix reveals a clean task–representation match: SELFIES dominates tasks rewarding distribution-level chemical validity (unconditional generation, property optimization), while SMILES dominates tasks rewarding surface-level structural anchoring (customized generation, synthesis prediction). Maintaining both notations is therefore a feature, not a redundancy.
- • Tokenization, not modeling capacity, bounds fine-grained geometric fidelity. On unconditional conformer generation, BioMatrix matches diffusion-based baselines on distributional and dihedral metrics but exhibits a residual gap on bond-length precision. A single lightweight MMFF refinement closes most of this gap, indicating that the limitation lies in the irreducible quantization error of finite structural codebooks rather than in next-token modeling itself.

###### 6.1 1D Molecular Tasks

###### 6.1.1 Unconditional Generation

This task evaluates the model’s ability to learn the underlying distribution of drug-like molecules purely from data, without any conditioning signal. The model is asked to generate 10,000 molecules from scratch as line-notation strings. We evaluate on two widely used benchmarks, MOSES [98] and GuacaMol [16], following their official protocols. Both benchmarks assess the generated set along three standard axes: validity, uniqueness, and novelty. MOSES additionally reports internal diversity (IntDiv1/IntDiv2), which measures the average pairwise dissimilarity among generated molecules via Tanimoto similarity over molecular fingerprints, with higher values indicating that the generated set spans a broader range of distinct structures rather than collapsing onto similar ones.

- Table 5: Performance comparison of unconditional 1D molecule generation task across 10,000 samples on the MOSES [98] dataset. The baseline results are derived from [127].

Models Validity Unique Novelty IntDiv1 IntDiv2 Graph-Based

JT-VAE [53] 1.0 0.999 0.914 0.855 0.849 GraphINVENT [85] 0.964 0.998 – 0.857 0.851 Digress [125] 0.857 1.0 0.936 0.855 0.849

Sequence-Based

CharRNN [107] 0.975 0.999 0.842 0.856 0.850 AAE [84] 0.936 1.0 0.793 0.855 0.850 VAE [55] 0.977 0.998 0.700 0.856 0.850 LatentGAN [100] 0.897 0.997 0.949 0.857 0.850 MolGPT [11] 0.988 1.0 0.821 0.855 0.849 Sc2Mol [64] 0.631 0.990 0.986 0.866 0.872 OMG-GPT [127] 0.990 1.0 0.903 0.855 0.849

BioMatrix-1.7B (SMILES) 0.951 1.0 0.941 0.873 0.866 BioMatrix-1.7B (SELFIES) 0.995 1.0 0.891 0.865 0.858 BioMatrix-4B (SMILES) 0.983 1.0 0.954 0.867 0.860 BioMatrix-4B (SELFIES) 0.998 1.0 0.925 0.868 0.861

As shown in Tables 5 and 6, both BioMatrix-1.7B and BioMatrix-4B produce essentially valid and fully unique sets on both benchmarks, with validity reaching 0.951–0.999 and uniqueness at or near 1.0. On GuacaMol, BioMatrix-1.7B (SELFIES) attains the highest validity (0.999) among all sequenceand graph-based methods, while BioMatrix-4B (SELFIES) reaches 0.998 validity together with 1.000 uniqueness and 0.986 novelty, jointly outperforming the strongest dedicated molecular generator OMG-GPT [127] (0.984 / 1.000 / 0.975) across all three axes and remaining competitive with MolGPT [11] (0.979 / 0.999 / 1.000). On MOSES, BioMatrix-4B (SELFIES) likewise delivers the best overall balance, with validity of 0.998, uniqueness of 1.000, novelty of 0.925, and competitive internal diversity (IntDiv1=0.868, IntDiv2=0.861), improving over OMG-GPT (0.990 / 1.000 / 0.903 / 0.855 / 0.849) on every axis except validity, where the two are essentially tied.

- Table 6: Performance comparison of unconditional 1D molecule generation task across 10,000 samples on the GuacaMol [16] dataset. The baseline results are derived from [127].

Models Validity Unique Novelty Graph-Based

Digress [125] 0.852 1.0 0.996 NAGVAE [58] 0.929 0.955 1.0

Sequence-Based SMILES LSTM [108] 0.959 1.0 0.912 VAE [55] 0.870 0.999 0.974 AAE [84] 0.822 1.0 0.998 ORGAN [38] 0.379 0.841 0.687 MolGPT [11] 0.979 0.999 1.0 OMG-GPT [127] 0.984 1.0 0.975 BioMatrix-1.7B (SMILES) 0.994 0.999 0.957 BioMatrix-1.7B (SELFIES) 0.999 1.0 0.977 BioMatrix-4B (SMILES) 0.991 1.0 0.978 BioMatrix-4B (SELFIES) 0.998 1.0 0.986

BioMatrix occupies a more balanced validity–novelty–diversity operating point than prior generators. Existing baselines tend to trade one axis for another: Sc2Mol attains the highest novelty (0.986) and internal diversity (IntDiv2 = 0.872) on MOSES but at the cost of validity collapsing

- to 0.631, meaning most of its “novel” outputs are not chemically valid. BioMatrix instead keeps validity (≥0.951), uniqueness (≈1.0), novelty (≥0.891), and internal diversity (IntDiv1/2 above the ∼0.855 baseline cluster) jointly high. The SELFIES variants additionally guarantee syntactic validity by construction (0.995–0.999), suggesting that the unified next-token objective combined with SELFIES tokenization produces a generated set that is simultaneously valid, distinct, and broad in chemical-space coverage—without the need for hand-tuned trade-offs.

###### 6.1.2 Name Conversion

This task evaluates the model’s ability to translate between different molecular naming conventions, testing whether the model has internalized the bidirectional mapping between structured line notations (SMILES / SELFIES) and human-readable identifiers (IUPAC names, molecular formulas). Following the protocol of SMolInstruct [146], we evaluate on five sub-tasks that collectively probe both directions of this mapping: IUPAC-to-formula (I2F), IUPAC-to-sequence (I2S), sequence-to-formula (S2F), and sequence-to-IUPAC (S2I), where the sequence refers to SMILES for the SMILES variant of BioMatrix and SELFIES for the SELFIES variant. For each sub-task we report exact match accuracy (EM) as the primary metric, and additionally report chemical validity for I2S, which produces line-notation strings that must parse into legal molecules.

- As shown in Table 7, BioMatrix-4B establishes state-of-the-art performance across the board among LLM-based methods, achieving the best result on four of five metrics and remaining competitive on S2I EM with the strongest generalist baseline SciReasoner-8B while using roughly half the parameters. The improvement over SciReasoner-8B is most pronounced on I2S EM, rising from 84.40% to 92.83%. At the 1.7B scale, BioMatrix surpasses SciReasoner-1.7B on all EM metrics, indicating that the advantage is not simply a function of model size.

- Table 7: Performance comparison of molecular name conversion task on the SMolInstruct [146]. The baseline results are derived from [146, 132, 156].

I2F I2S S2F S2I EM ↑ EM ↑ Validity ↑ EM ↑ EM ↑ Task-specific, non-LLM based models Literature SOTA 97.90 73.50 99.40 100.00 56.50 General / molecular LLM baselines

Model

GPT-4 [5] 8.70 3.30 84.20 4.80 0.00 GPT-oss [6] 33.24 7.58 – 19.88 0.17 Claude Opus 3 [10] 34.60 17.70 90.20 9.20 0.00 LlaSMol Mistral [146] 87.90 70.10 99.60 93.20 29.00 SciReasoner-1.7B [132] 91.81 78.85 – 95.92 49.52 SciReasoner-8B [132] 92.65 84.40 – 96.39 56.63 BioMedGPT-Mol-8B [156] 91.40 76.50 – 95.70 44.90

BioMatrix-1.7B (SMILES) 79.45 87.22 99.12 99.39 44.75 BioMatrix-1.7B (SELFIES) 79.42 78.39 99.78 99.46 42.58 BioMatrix-4B (SMILES) 94.09 92.83 99.75 99.28 55.21 BioMatrix-4B (SELFIES) 94.12 83.10 99.82 99.31 53.40

###### 6.1.3 Property Prediction

This task evaluates the model’s ability to predict physicochemical and biological properties of molecules directly from their line-notation representations, testing whether the model has internalized the mapping from molecular structure to functional behavior. Following the protocol of SMolInstruct [146], we evaluate on six widely used MoleculeNet [136] benchmarks spanning two regression tasks—ESOL and Lipo, measured by root-mean-square error (RMSE)—and four classification tasks—BBBP, ClinTox, HIV, and SIDER, measured by accuracy.

- As shown in Table 8, BioMatrix achieves the best or second-best result among LLM-based methods on five of six benchmarks, and surpasses the task-specific literature SOTA on three of them (HIV, SIDER, and ESOL for the 1.7B variant). BioMatrix-1.7B already outperforms the much larger SciReasoner8B and BioMedGPT-Mol-8B on four of six benchmarks, again indicating that the advantage is not primarily driven by model scale. The four classification benchmarks (BBBP, ClinTox, HIV, SIDER) yield tightly clustered results across all four BioMatrix variants, indicating stable performance on these tasks. The two regression benchmarks show larger fluctuations—ESOL RMSE in particular ranges from 0.810

to 1.740 across variants without a clean pattern—which we attribute to the higher sensitivity of RMSE

- as a continuous, unbounded metric compared to classification accuracy, rather than to systematic differences between variants.

6.1.4 Molecule Captioning and Text-based Molecule Generation

These two tasks jointly evaluate the model’s bidirectional alignment between molecular structures and natural-language descriptions: captioning asks the model to produce a textual description given a molecule, while text-based generation asks it to produce a molecule given a textual description. Following the protocol of SMolInstruct [146], we evaluate captioning with METEOR against reference descriptions, and text-based generation with exact match accuracy (EM), fingerprint-based Tanimoto similarity (FTS) to the reference molecule, and chemical validity of the generated line notations.

- As shown in Table 9, BioMatrix achieves the best result among LLM-based methods on all four metrics

- Table 8: Performance comparison of property prediction task on the SMolInstruct [146]. The baseline results are derived from [146, 132, 156]

Model

ESOL Lipo BBBP ClinTox HIV SIDER

RMSE ↓ RMSE ↓ ACC ↑ ACC ↑ ACC ↑ ACC ↑ Task-specific, non-LLM based models Literature SOTA 0.819 0.612 85.30 92.40 97.00 70.00 General / molecular LLM baselines

GPT-4 [5] 2.570 1.545 62.90 50.00 59.60 57.60 GPT-oss [6] 3.45 1.65 46.70 52.78 69.86 38.25 Claude Opus 3 [10] 1.036 1.194 75.10 41.70 76.40 67.00 LlaSMol Mistral [146] 1.150 1.010 74.60 93.10 96.70 70.70 SciReasoner-1.7B [132] 1.21 0.94 79.70 92.36 96.79 70.00 SciReasoner-8B [132] 1.080 0.820 82.74 91.67 96.81 68.53 BioMedGPT-Mol-8B [156] 0.916 0.973 87.60 93.10 – –

BioMatrix-1.7B (SMILES) 0.810 0.860 81.62 90.91 97.43 72.39 BioMatrix-1.7B (SELFIES) 0.860 0.850 82.70 92.31 97.32 72.14 BioMatrix-4B (SMILES) 1.240 0.780 84.86 93.71 97.56 72.35 BioMatrix-4B (SELFIES) 1.740 0.840 85.95 94.41 97.48 72.48

- Table 9: Performance comparison of molecule captioning (METEOR) and text-based molecule generation (EM, FTS, Valid) tasks on the SMolInstruct [146] dataset. The baseline results are derived from [146, 132, 156].

Model METEOR ↑ EM ↑ FTS ↑ Validity ↑ Task-specific, non-LLM based models Literature SOTA 51.5 31.70 73.20 95.30 General / molecular LLM baselines GPT-4 [5] 18.8 6.40 42.60 81.40 GPT oss [6] 15.00 3.00 – – Claude Opus 3 [10] 21.9 12.30 57.60 92.60 LlaSMol Mistral [146] 45.2 19.20 61.70 99.70 SciReasoner-1.7B [132] 59.0 39.00 – – SciReasoner-8B [132] 60.0 48.00 – – BioMedGPT-Mol-8B [156] 51.5 29.60 77.50 98.90 BioMatrix-1.7B (SMILES) 60.2 56.35 80.76 99.72 BioMatrix-1.7B (SELFIES) 59.7 53.94 79.81 99.82 BioMatrix-4B (SMILES) 61.4 65.07 85.47 99.52 BioMatrix-4B (SELFIES) 61.5 63.98 84.75 99.47

and additionally surpasses the task-specific literature SOTA on three of them (METEOR, EM, and FTS). The gains are most pronounced on text-based generation, where BioMatrix-4B (SMILES) improves EM from the prior LLM best of 48.00% (SciReasoner-8B) to 65.07%, and FTS from 77.50% (BioMedGPT-Mol8B) to 85.47%. BioMatrix-1.7B already outperforms every LLM baseline on all four metrics, including the much larger SciReasoner-8B and BioMedGPT-Mol-8B.

###### 6.1.5 Forward Synthesis and Retrosynthesis

These two tasks jointly evaluate the model’s understanding of chemical reactions in both directions: forward synthesis asks the model to predict the product given a set of reactants, while retrosynthesis asks it to predict plausible reactants given a target product. Following the protocol of SMolInstruct [146], we report exact match accuracy (EM), fingerprint-based Tanimoto similarity (FTS) to the reference molecule, and chemical validity of the generated line notations for both directions.

- Table 10: Performance comparison of forward synthesis and retrosynthesis tasks on the SMolInstruct [146] dataset. The baseline results are derived from [146, 132, 156].

Forward Synthesis Retrosynthesis

Model

EM ↑ FTS ↑ Validity ↑ EM ↑ FTS ↑ Validity ↑ Task-specific, non-LLM based models Literature SOTA 78.70 92.20 100.00 47.00 77.50 99.70 General / molecular LLM baselines GPT-4 [5] 1.60 40.50 87.00 0.00 33.40 42.60 GPT-oss [6] 33.00 – – 8.00 – – Claude Opus 3 [10] 3.70 45.70 97.00 1.10 46.20 94.80 LlaSMol Mistral [146] 63.30 84.90 99.80 32.90 70.40 100.00 SciReasoner-1.7B [132] 69.00 – – 41.00 – – SciReasoner-8B [132] 75.00 – – 46.00 – – BioMedGPT-Mol-8B [156] 67.20 88.50 99.80 32.40 75.20 99.90 BioMatrix-1.7B (SMILES) 72.75 92.22 99.74 42.69 73.81 99.77 BioMatrix-1.7B (SELFIES) 62.78 84.24 99.93 39.32 71.97 99.93 BioMatrix-4B (SMILES) 77.94 89.88 99.93 45.16 74.88 99.93 BioMatrix-4B (SELFIES) 69.08 87.01 99.85 42.90 73.63 99.93

- As shown in Table 10, BioMatrix achieves the best or second-best result among LLM-based methods on all six metrics. The SMILES variants consistently outperform the SELFIES variants on EM, with BioMatrix-4B (SMILES) reaching 77.94% on forward synthesis EM and 45.16% on retrosynthesis EMsubstantially ahead of the prior LLM best (SciReasoner-8B at 75.00% / 46.00%) on forward synthesis while remaining competitive on RS.

6.1.6 Molecule Editing

This task evaluates the model’s ability to perform localized structural modifications to a given molecule under a natural-language instruction—adding, deleting, or substituting a chemical component while leaving the rest of the molecule intact. It thereby probes a finer-grained capability than text-based generation: the model must locate the target substructure, apply the requested edit, and emit a chemically valid output that differs from the input only in the intended way. Following the protocol of S2-Bench [60], we evaluate on three sub-tasks—component addition (AddComp), component deletion (DelComp), and component substitution (SubComp)—and we report success rate (SR) and weighted success rate (WSR), where the latter additionally weights each successful edit by its structural similarity to the input molecule, so that high WSR requires the edit to be both successful and localized.

- As shown in Table 11, BioMatrix establishes new state-of-the-art results on AddComp and DelComp across both metrics. On AddComp, BioMatrix-4B (SMILES) reaches an SR of 0.9294 and a WSR of 0.6447, improving over the strongest baseline Llama3.1-8B (OpenMolIns-xlarge) [60] by a substantial margin on both axes despite using fewer than half the parameters. On DelComp, BioMatrix-4B

- Table 11: Performance comparison of molecule editing task on the S2-Bench [60]. The baseline results are derived from [60].

AddComp DelComp SubComp

Model

SR ↑ WSR ↑ SR ↑ WSR ↑ SR ↑ WSR ↑

GPT-4o [50] 0.6188 0.4197 0.7012 0.4234 0.7992 0.5774 GPT-4-turbo [92] 0.6990 0.4848 0.7244 0.4154 0.7778 0.5696 Claude-3.5 [8] 0.6832 0.4794 0.5414 0.3615 0.8104 0.5924 Gemini-1.5-pro [121] 0.7058 0.4794 0.7590 0.4515 0.7148 0.5103 Llama3.1-8B (OpenMolIns-xlarge) [60] 0.7790 0.5273 0.8640 0.5327 0.6100 0.4535

BioMatrix-1.7B (SMILES) 0.8816 0.6100 0.9524 0.5760 0.6034 0.4556 BioMatrix-1.7B (SELFIES) 0.8522 0.5853 0.9626 0.5693 0.5858 0.4369 BioMatrix-4B (SMILES) 0.9294 0.6447 0.9568 0.5774 0.6082 0.4600 BioMatrix-4B (SELFIES) 0.9124 0.6272 0.9666 0.5724 0.5990 0.4501

(SELFIES) attains the highest SR of 0.9666, with all four BioMatrix variants exceeding the prior best across both SR and WSR. BioMatrix-1.7B already outperforms every baseline on these two sub-tasks. On SubComp, however, BioMatrix trails the strongest proprietary baseline Claude-3.5 [8], whose SR of 0.8104 remains well above the best BioMatrix result. We attribute this asymmetry to the intrinsic difficulty of substitution as an editing operation: unlike addition or deletion, which involve a single localized modification, substitution requires the model to jointly perform a deletion and an insertion

- at the same site while preserving the surrounding chemical context, leaving substantially less room for error. The fact that even the much larger Llama3.1-8B baseline drops to an SR of 0.6100 on this sub-task (well below its own AddComp and DelComp scores) suggests that SubComp is broadly challenging for open-source instruction-tuned models.

###### 6.1.7 Customized Molecule Generation

This task evaluates the model’s ability to generate molecules that satisfy explicit structural constraints—a specified atom count, bond count, or set of functional groups—rather than reproducing a particular molecule described in free text. Compared to text-based molecule generation (Section 6.1.4), where each prompt describes a single target molecule, customized generation specifies only a structural recipe and admits any chemically valid molecule satisfying it. This shifts the task from one of imitation to one of constrained enumeration: the model must internalize what each constraint means at the structural level (e.g., counting atoms of a given element, recognizing functional-group substructures) and then produce a molecule that simultaneously satisfies the constraint and remains chemically valid. We follow the protocol of S2-Bench [60] and evaluate on three sub-tasks—atom-number control (AtomNum), bond-number control (BondNum), and functional-group inclusion (FuncGroup)—reporting success rate (SR) and weighted success rate (WSR), where the latter additionally weights each successful generation by its novelty against a large external reference database (Zinc-250K [112]), so that high WSR requires the model to satisfy the constraints with structurally novel molecules.

- As shown in Table 12, BioMatrix-4B (SMILES) achieves the best result on AtomNum across both metrics, lifting AtomNum SR to 0.2774 and clearly surpassing every proprietary and open-source baseline. On BondNum, BioMatrix-4B (SMILES) reaches an SR of 0.1104, essentially matching the strongest baseline Llama3.1-8B (OpenMolIns-xlarge) [60] (0.1196) despite using roughly half the parameters. A similar pattern holds on FuncGroup, where Llama3.1-8B retains the lead at SR 0.3548 and BioMatrix-4B (SMILES) reaches 0.2714, remaining competitive with the proprietary frontier models.

The most striking observation is the large gap between the SMILES and SELFIES variants across all three sub-tasks: on FuncGroup, for example, BioMatrix-4B (SMILES) reaches an SR of 0.2714

- Table 12: Performance comparison of customized molecule generation task on the S2-Bench [60]. The baseline results are derived from [60].

AtomNum BondNum FuncGroup SR ↑ WSR ↑ SR ↑ WSR ↑ SR ↑ WSR ↑

MolCustom

GPT-4o [50] 0.1998 0.1339 0.0650 0.0412 0.2330 0.1518 GPT-4-turbo [92] 0.1702 0.1190 0.0774 0.0488 0.2180 0.1440 Claude-3.5 [8] 0.1928 0.1335 0.1058 0.0697 0.2364 0.1556 Gemini-1.5-pro [121] 0.1742 0.1202 0.0708 0.0462 0.2486 0.1659 Llama3.1-8B (OpenMolIns-xlarge) [60] 0.1186 0.0811 0.1196 0.0807 0.3548 0.2268

BioMatrix-1.7B (SMILES) 0.2176 0.1471 0.0996 0.0702 0.2408 0.1497 BioMatrix-1.7B (SELFIES) 0.0680 0.0492 0.0274 0.0178 0.0098 0.0062 BioMatrix-4B (SMILES) 0.2774 0.1899 0.1104 0.0795 0.2714 0.1703 BioMatrix-4B (SELFIES) 0.0262 0.0184 0.0056 0.0041 0.0048 0.0032

whereas BioMatrix-4B (SELFIES) collapses to 0.0048. This pattern, which is largely absent from the imitation-style tasks evaluated earlier, points to a structural mismatch between SELFIES and the kind of constraints this task imposes, and the mismatch manifests differently across the three sub-tasks.

For AtomNum, the difficulty is that the number of atom tokens in a SELFIES string does not in general equal the number of heavy atoms in the resulting molecule. SELFIES uses control tokens such as [Branch1] and [Ring1] that are each followed by one token whose role is not to contribute an atom, but to act as an index into the SELFIES alphabet that encodes the branch length or ring size. Concretely, the two SELFIES strings [C][C][Branch1][O][C][C][C] and [C][C][C][C][C][C][Ring1][N] each contain six [C] tokens, yet decode to pentane (CCCCC, five carbons) and cyclohexane (C1CCCCC1, six carbons) respectively: in the first string, the [C] token immediately following [Branch1] is consumed as a branch-length index and contributes no atom, while in the second string the [N] token following [Ring1] encodes the ring size and likewise contributes no atom. A model cannot therefore satisfy a constraint such as “contains six carbons” by simply emitting six [C] tokens, whereas in SMILES every heavy-atom character is a heavy atom.

For BondNum, the difficulty is that the markers needed to count bonds in SELFIES are not standalone tokens the model can enumerate. In SMILES, non-single bonds appear as dedicated characters—= for double and # for triple—and ring-closing bonds as matching pairs of digits, so checking a constraint like “contains two double bonds and one triple bond” reduces to counting = and # characters. In SELFIES, by contrast, the same information is encoded in two ways that both obscure it. First, double and triple bonds are fused into the following atom token rather than existing as standalone tokens: acetylene is written C#C in SMILES (one #) but [C][#C] in SELFIES, with the triple-bond marker now baked into the second atom token. Second, ring-closing bonds are encoded through [Ring1] together with an index token specifying the ring size, and the bond type of the ring closure is itself carried as a prefix on that same index token (e.g., the benzene fragment in benzoic acid ends with [Ring1][=Branch1], where [=Branch1] serves simultaneously as a ring-size index and as an indicator that the closing bond is a double bond). Counting bonds from a SELFIES string therefore requires parsing the internal structure of several distinct token classes rather than aggregating over one, whereas in SMILES every non-default bond has a dedicated, countable surface form.

For FuncGroup, the difficulty lies in surface-form instability. Common functional groups in the benchmark pool (benzene rings, carboxyl groups, amides, and so on) correspond in SMILES to short, recurring substrings: every benzene ring is written as c1ccccc1 (or its kekulized variants), every carboxyl group as C(=O)O, every primary amide as C(=O)N, regardless of where they appear in the molecule. In SELFIES, by contrast, the same functional group can be split across discontiguous

token positions depending on its surrounding context. In benzoic acid (OC(=O)c1ccccc1), the benzene ring encodes into a contiguous block [C][=C][C][=C][C][=C][Ring1][=Branch1]. In biphenyl (c1ccc(cc1)-c1ccccc1), however, the encoder chooses to realize the first benzene ring as a ring whose tail is wrapped inside a branch: the first four ring atoms appear as [C][=C][C][=C], followed by a [Branch1][Branch1] control pair that opens a branch containing the remaining two ring atoms and the ring-closing marker ([C][=C][Ring1][=Branch1]), after which the second benzene ring follows as its own contiguous block. The same chemical substructure (a benzene ring) therefore has no single canonical token signature in SELFIES, and a model cannot anchor a constraint such as “contains a benzene ring” to one recognizable token pattern.

SELFIES is great for open-ended generation, but this same feature makes it unsuitable for tasks with strict structural constraints. Across all three customized-generation sub-tasks, the SELFIES variants of BioMatrix collapse to near-zero success rates while the SMILES variants achieve stateof-the-art or competitive results. The underlying failure modes—atom tokens reinterpreted as branch/ring indices, bond types folded into neighboring atom tokens, and substructures split across non-contiguous positions—all stem from SELFIES’s context-dependent token semantics. This same property guarantees the syntactic validity that benefits unconditional generation (Section 6.1.1), but it prevents the model from anchoring structural constraints to recognizable token patterns.

###### 6.1.8 Molecule Optimization

This task evaluates the model’s ability to modify a given molecule under a natural-language instruction so as to improve one or more target properties while keeping the modified molecule structurally close to the original. Compared to molecule editing (Section 6.1.6), which specifies a structural operation (add / delete / substitute a component), optimization specifies a property-level objective and leaves the model free to choose which substructures to modify in order to reach it. We evaluate BioMatrix on two complementary benchmarks: S2-Bench [60], which focuses on single-property optimization, and MolOpt [144], which additionally covers multi-property joint optimization.

Single-property optimization. We follow the protocol of S2-Bench [60] and evaluate on three singleproperty optimization sub-tasks corresponding to three widely used molecular properties: octanolwater partition coefficient (logP), molar refractivity (MR), and quantitative estimate of drug-likeness (QED). For each sub-task, the model is given an input molecule together with an instruction specifying a target property and the desired direction of change, and is asked to return a modified molecule. We report success rate (SR) and weighted success rate (WSR), where the latter additionally weights each successful optimization by its structural similarity to the input molecule.

- As shown in Table 13, BioMatrix establishes new state-of-the-art performance on logP and MR across both metrics, with all four BioMatrix variants surpassing every proprietary and open-source baseline by a clear margin. On QED, BioMatrix remains competitive with the strongest baseline Llama3.1-8B (OpenMolIns-xlarge) [60], with BioMatrix-4B (SMILES) trailing it slightly on both SR and WSR.

Single- and multi-property optimization. We complement the S2-Bench evaluation with the MolOptInstructions benchmark [144], which extends molecule optimization in two directions: it enlarges the set of single-property objectives to cover drug-relevant attributes beyond logP/MR/QED—namely QED, hydrogen bond acceptor count (acceptor+), hydrogen bond donor count (donor+), aqueous solubility (solubility+), blood–brain barrier permeability (bbbp+), and hERG inhibition (herg-, where the goal is to decrease the value). It additionally introduces multi-property objectives (sol+&acc+, qed+&bbbp+) that require the model to optimize two attributes jointly. Following the original protocol of [144], each task is evaluated under two success criteria. Under the loose criterion (L), an optimization is counted as successful whenever the optimized property is shifted in the requested direction relative to the input. Under the strict criterion (S), the property must additionally clear a per-property threshold

- Table 13: Performance comparison of single-property molecule optimization task on the S2-Bench [60]. The baseline results are derived from [60].

logP MR QED SR ↑ WSR ↑ SR ↑ WSR ↑ SR ↑ WSR ↑

MolOpt

GPT-4o [50] 0.7190 0.4735 0.6864 0.4407 0.3952 0.2442 GPT-4-turbo [92] 0.7662 0.5351 0.7388 0.5039 0.3946 0.2599 Claude-3.5 [8] 0.7970 0.5678 0.6962 0.4951 0.5361 0.3775 Gemini-1.5-pro [121] 0.7712 0.5415 0.7876 0.5312 0.4704 0.2859 Llama3.1-8B (OpenMolIns-xlarge) [60] 0.8822 0.5877 0.6982 0.4673 0.8648 0.5825

BioMatrix-1.7B (SMILES) 0.9780 0.6635 0.9590 0.6436 0.8210 0.5569 BioMatrix-1.7B (SELFIES) 0.9660 0.6483 0.9702 0.6527 0.7980 0.5401 BioMatrix-4B (SMILES) 0.9838 0.6625 0.9708 0.6502 0.8368 0.5672 BioMatrix-4B (SELFIES) 0.9734 0.6495 0.9776 0.6546 0.8170 0.5504

(e.g., ∆QED ≥ 0.1, ∆acceptor ≥ 1); for solubility specifically, the strict criterion further demands that the optimized value land within a prescribed range around the input value rather than merely move in the right direction, making it the most demanding sub-task in the benchmark. We report the fraction of generations satisfying the optimization objective (Correct) together with the fraction that are chemically valid (Validity), under both criteria.

- As shown in Table 14, BioMatrix-4B (SELFIES) achieves the best Correct and Validity scores on the majority of metric–task combinations under both criteria, with the remaining top positions filled by other BioMatrix variants. The improvement over the prior best baseline DrugAssist [144] is consistent across both single- and multi-property tasks, and is particularly visible on hERG inhibition (widely regarded as one of the hardest properties to optimize due to its complex and data-scarce structure–activity relationships) and on the strict solubility setting, the most demanding sub-task in the benchmark, where the strict criterion requires landing inside a prescribed value range rather than merely shifting in the right direction. The advantage of BioMatrix on this benchmark also holds at the smaller scale: BioMatrix-1.7B (SELFIES) already surpasses every baseline on Correct for most tasks.

The same SELFIES property that hurts customized generation helps property optimization. Across all eight MolOpt-Instructions sub-tasks and under both loose and strict success criteria, the SELFIES variants of BioMatrix dominate the Correct and Validity columns, with BioMatrix-4B (SELFIES) leading on the majority of metric–task combinations. This stands in direct contrast to the customized-generation results in Section 6.1.7, where the SELFIES variants collapsed to near-zero. The asymmetry reflects a clean task–representation match: optimization requires producing valid molecules with shifted properties, which SELFIES guarantees by construction and supports by abstracting away surface-form bookkeeping, whereas customized generation requires anchoring constraints to surface tokens, which SELFIES’s context-dependent semantics actively obscure.

###### 6.1.9 Molecule Question Answering

This task evaluates the model’s factual accuracy in molecular knowledge, testing whether the model can correctly answer fine-grained questions about specific molecules rather than producing fluent but factually incorrect descriptions. We evaluate on MoleculeQA [78], the largest molecule-domain QA benchmark to date, which is designed to expose factual errors in molecular descriptions that traditional lexical-similarity metrics (e.g., BLEU, ROUGE) fail to detect. Following an MMLU [43]-style multiple-choice format, MoleculeQA covers four sub-categories of molecular knowledge: Structure, Source, Property, and Application, with overall accuracy (Total) reported as the primary metric.

- Table 14: Performance comparison of molecule optimization task on the MolOpt-Instructions [144] dataset. The baseline results are derived from [144].

Task Model Correct (L) ↑ Validity (L) ↑ Correct (S) ↑ Validity (S) ↑

GPT-3.5-turbo [91] 0.15 0.97 0.15 0.96 BioMedGPT-LM [79] 0.15 0.34 0.09 0.32 DrugAssist [144] 0.76 0.99 0.63 0.97

qed+

BioMatrix-1.7B (SMILES) 0.73 0.84 0.51 0.76 BioMatrix-1.7B (SELFIES) 0.87 0.95 0.688 0.95 BioMatrix-4B (SMILES) 0.81 0.87 0.59 0.77 BioMatrix-4B (SELFIES) 0.93 0.99 0.74 0.98

GPT-3.5-turbo [91] 0.04 0.98 0.06 0.96 BioMedGPT-LM [79] 0.18 0.45 0.13 0.39 DrugAssist [144] 0.71 0.97 0.67 0.96

acceptor+

BioMatrix-1.7B (SMILES) 0.82 0.86 0.72 0.84 BioMatrix-1.7B (SELFIES) 0.83 0.91 0.75 0.93 BioMatrix-4B (SMILES) 0.89 0.91 0.84 0.88 BioMatrix-4B (SELFIES) 0.96 0.99 0.93 0.97

GPT-3.5-turbo [91] 0.10 0.98 0.04 0.95 BioMedGPT-LM [79] 0.17 0.46 0.09 0.46 DrugAssist [144] 0.72 0.98 0.76 0.95

donor+

BioMatrix-1.7B (SMILES) 0.91 0.94 0.85 0.89 BioMatrix-1.7B (SELFIES) 0.80 0.91 0.74 0.93 BioMatrix-4B (SMILES) 0.94 0.95 0.92 0.93 BioMatrix-4B (SELFIES) 0.95 0.98 0.93 0.98

GPT-3.5-turbo [91] 0.16 0.94 0.05 0.95 BioMedGPT-LM [79] 0.18 0.27 0.09 0.35 DrugAssist [144] 0.80 0.98 0.41 0.98

solubility+

BioMatrix-1.7B (SMILES) 0.96 0.97 0.63 0.80 BioMatrix-1.7B (SELFIES) 0.90 0.98 0.51 0.87 BioMatrix-4B (SMILES) 0.94 0.94 0.58 0.71 BioMatrix-4B (SELFIES) 0.97 1.00 0.64 0.90

GPT-3.5-turbo [91] 0.10 0.97 0.10 0.95 BioMedGPT-LM [79] 0.16 0.26 0.07 0.22 DrugAssist [144] 0.82 0.99 0.61 0.98

bbbp+

BioMatrix-1.7B (SMILES) 0.95 0.97 0.79 0.89 BioMatrix-1.7B (SELFIES) 0.90 0.98 0.62 0.94 BioMatrix-4B (SMILES) 0.93 0.94 0.81 0.89 BioMatrix-4B (SELFIES) 0.94 1.00 0.67 0.99

GPT-3.5-turbo [91] 0.13 0.98 0.15 0.97 BioMedGPT-LM [79] 0.13 0.20 0.12 0.18 DrugAssist [144] 0.71 0.99 0.67 0.98

herg-

BioMatrix-1.7B (SMILES) 0.69 0.83 0.54 0.77 BioMatrix-1.7B (SELFIES) 0.84 0.96 0.72 0.93 BioMatrix-4B (SMILES) 0.71 0.85 0.64 0.82 BioMatrix-4B (SELFIES) 0.94 0.99 0.85 0.99

GPT-3.5-turbo [91] 0.09 0.92 0.02 0.91 BioMedGPT-LM [79] 0.10 0.29 0.07 0.32 DrugAssist [144] 0.50 0.95 0.27 0.95

sol+&acc+

BioMatrix-1.7B (SMILES) 0.83 0.90 0.21 0.59 BioMatrix-1.7B (SELFIES) 0.74 0.93 0.23 0.77 BioMatrix-4B (SMILES) 0.89 0.92 0.24 0.50 BioMatrix-4B (SELFIES) 0.91 0.99 0.34 0.78

GPT-3.5-turbo [91] 0.09 0.96 0.06 0.95 BioMedGPT-LM [79] 0.16 0.35 0.11 0.36 DrugAssist [144] 0.65 0.99 0.41 0.98

qed+&bbbp+

BioMatrix-1.7B (SMILES) 0.70 0.86 0.45 0.73 BioMatrix-1.7B (SELFIES) 0.84 0.96 0.43 0.91 BioMatrix-4B (SMILES) 0.73 0.85 0.46 0.72 BioMatrix-4B (SELFIES) 0.88 1.00 0.46 0.96

- Table 15: Performance comparison of molecule question answering task on the MoleculeQA [78]. The baseline results are derived from [78].

Model Structure Source Property Application Total GPT-3.5-1106-turbo [91] 25.60 37.60 28.04 32.22 29.29 Galactica-6.7B [119] 32.35 41.92 31.05 28.21 33.96 Mol-Instructions-7B [28] 37.46 47.36 32.69 29.88 38.37 BioMedGPT-LM-7B [79] 54.19 60.01 38.85 40.9 52.23 GPT-4-1106-preview [5] 60.94 50.19 35.57 43.91 53.47 MolT5-base [24] 58.01 65.85 45.14 42.24 55.39 MoMu-base [113] 61.58 65.3 43.78 43.07 57.43 BioT5-base [95] 65.98 69.24 49.11 40.73 62.03 MolCA-1.3B [71] 71.12 70.98 47.81 43.17 64.79 BioMatrix-1.7B (SMILES) 79.28 74.76 43.23 44.41 70.07 BioMatrix-1.7B (SELFIES) 79.05 75.06 43.91 44.91 70.15 BioMatrix-4B (SMILES) 83.36 77.96 47.88 46.24 73.78 BioMatrix-4B (SELFIES) 83.26 76.55 47.20 45.58 73.24

- As shown in Table 15, BioMatrix-4B establishes new state-of-the-art performance on the Total accuracy (73.78%) as well as on the Structure, Source, and Application sub-categories, substantially surpassing the prior best MolCA-1.3B (64.79%). The improvement is most pronounced on the Structure subcategory, where BioMatrix-4B reaches 83.36% versus the prior best of 71.12%, indicating that BioMatrix captures structural knowledge of molecules with notably higher fidelity. BioMatrix-1.7B already outperforms every baseline on Total accuracy and on the Structure, Source, and Application subcategories, including specialist baselines such as BioT5-base and MolCA-1.3B that are explicitly designed for molecule–text understanding.

One observation worth noting is that the Property sub-category remains the hardest for all models, with BioMatrix-4B reaching 47.88%—competitive with but not exceeding the prior best BioT5-base (49.11%). We attribute this to the fact that property-related questions often require predicting over physicochemical attributes (e.g., precise solubility values or activity ranges), which is a fundamentally different skill from the factual recall that dominates the other three sub-categories.

###### 6.2 3D Molecular Tasks

###### 6.2.1 Unconditional 3D Structure Generation

This task evaluates the model’s ability to jointly generate a chemically valid molecular graph and a physically plausible conformer, without any conditioning signal. We evaluate on QM9-2014 [101] following the protocol of [73], reporting two sets of metrics that target the 2D graph and 3D geometry of the generated molecules, respectively. The 2D-level metrics comprise atom stability, molecule stability, and validity & completeness (V&C) for chemical validity; validity & uniqueness (V&U) and validity & uniqueness & novelty (V&U&N) for diversity; and similarity to nearest neighbor (SNN), fragment similarity (Frag), scaffold similarity (Scaf), and Fr´echet ChemNet Distance (FCD) for distributional similarity to the test set. The 3D-level metrics comprise atom stability and FCD computed on the 2D graphs reconstructed from the predicted coordinates, together with the maximum mean discrepancy (MMD) of bond lengths, bond angles, and dihedral angles against the reference distribution. Training-set statistics are also included as a reference upper bound.

Results are shown in Table 16. On the 2D side, BioMatrix delivers strong and well-rounded perfor-

- Table 16: Performances comparison of unconditional 3D molecule generation across 10,000 samples on the QM9-2014 [101] dataset. The baseline results are derived from [73]. † denotes post-hoc refinement via MMFF force-field optimization.

- 2D-Metric FCD ↓ AtomStable ↑ MolStable ↑ V&C ↑ V&U ↑ V&U&N ↑ SNN ↑ Frag ↑ Scaf ↑

Train 0.063 0.999 0.988 0.989 0.989 0.000 0.490 0.992 0.946 MolGPT [11] 0.461 0.982 0.976 0.977 0.937 0.763 0.523 0.958 0.923 MolGen [29] 0.085 1.000 0.988 1.000 0.955 0.479 0.500 0.988 0.934 CDGS [46] 0.798 0.997 0.951 0.951 0.936 0.860 0.493 0.973 0.784 JODO [47] 0.138 0.999 0.988 0.990 0.960 0.780 0.522 0.986 0.934

- MiDi [126] 0.187 0.998 0.976 0.980 0.954 0.769 0.501 0.979 0.882 EQGAT-diff [59] 2.157 1.000 0.972 1.000 0.996 0.695 0.479 0.949 0.707 NExT-Mol [73] 0.070 1.000 0.989 1.000 0.967 0.802 0.530 0.992 0.945

- BioMatrix-1.7B 0.064 1.000 0.988 1.000 0.948 0.749 0.495 0.991 0.944

- BioMatrix-4B 0.066 1.000 0.988 1.000 0.949 0.742 0.494 0.991 0.942

3D-Metric FCD ↓ AtomStable ↑ Bond length ↓ Bond angle ↓ Dihedral angle ↓

Train 0.877 0.994 5.44E-04 4.65E-04 1.78E-04 G-SchNet [33] 2.386 0.957 3.62E-01 7.27E-02 4.20E-03 G-SphereNet [80] 6.659 0.672 1.51E-01 3.54E-01 1.29E-02 EDM [44] 1.285 0.986 1.30E-01 1.82E-02 6.64E-04 MDM [48] 4.861 0.992 2.74E-01 6.60E-02 2.39E-02 JODO [47] 0.885 0.992 1.48E-01 1.21E-02 6.29E-04 MiDi [126] 1.100 0.983 8.96E-01 2.08E-02 8.14E-04 EQGAT-diff [59] 1.519 0.988 4.09E-01 1.91E-02 1.14E-03 NExT-Mol [73] 0.879 0.993 1.15E-01 7.32E-03 1.95E-04

BioMatrix-1.7B 1.040 0.897 1.05E+00 3.16E-02 4.76E-04

- BioMatrix-4B 1.012 0.897 1.05E+00 3.09E-02 6.09E-04 BioMatrix-1.7B† 0.226 0.985 1.15E+00 4.38E-02 2.19E-04 BioMatrix-4B† 0.225 0.985 1.16E+00 4.18E-02 2.41E-04

mance: BioMatrix-1.7B attains FCD=0.064, essentially matching the training reference (0.063) and the strongest diffusion-based baseline NExT-Mol (0.070), and is within 0.001–0.002 of the training distribution on Frag (0.991 vs. 0.992) and Scaf (0.944 vs. 0.946). Atom stability, molecule stability, and V&C all reach 1.000 / 0.988 / 1.000, on par with or above every baseline. The two axes where BioMatrix trails NExT-Mol are V&U&N (0.749 vs. 0.802) and SNN (0.495 vs. 0.530), both reflecting a generated distribution that sits slightly closer to the training set than NExT-Mol’s.

On the geometric side, BioMatrix shows a more nuanced picture. In the default decoding setting, the raw conformers exhibit competitive dihedral-angle MMD but comparatively loose bond-length fidelity (MMD=1.05) and lower atom stability (0.897). The bond-length MMD is highly sensitive to small shifts in the distributional mean, and we therefore interpret this number as indicating a modest systematic offset in predicted bond lengths rather than a catastrophic failure of geometry modeling. We attribute this primarily to two structural limitations of the MolStrucTok-based reconstruction pipeline. First, its 512-entry codebook must quantize a four-dimensional continuous descriptor space, imposing an intrinsic quantization error on every per-atom geometric code that no amount of upstream modeling can recover. Second, coordinates are reconstructed autoregressively by sequentially traversing the molecule and rebuilding each local spherical frame atom-by-atom, which causes peratom decoding errors to accumulate along the traversal and occasionally produces geometrically implausible configurations; this stands in contrast to diffusion-based approaches such as NExT-Mol, which denoise all atomic coordinates jointly and therefore enjoy a global geometric consistency that autoregressive reconstruction does not afford.

Autoregressive structure reconstruction trades global geometric consistency for unification, but the gap is largely closable post-hoc. BioMatrix’s raw conformers match diffusion-based baselines on dihedral-angle MMD but exhibit looser bond-length fidelity (MMD = 1.05) and lower atom stability (0.897) than NExT-Mol, reflecting irreducible quantization error from a finite codebook compounded by per-atom error accumulation along the autoregressive traversal. A single lightweight MMFF refinement step closes most of this gap (FCD: 1.04 → 0.23, atom stability: 0.897 → 0.985), indicating that the limitation is concentrated in fine-grained geometric consistency rather than chemical plausibility, and that combining a unified token-level generator with cheap geometric post-processing is a practical path to high-fidelity structural outputs.

###### 6.2.2 Property-conditioned 3D Structure Generation

This task evaluates the model’s ability to generate molecular conformers whose quantum-chemical properties match a prescribed target value, testing whether a model has internalized the mapping from molecular structure to physical observables rather than merely reproducing the training distribution. Following the protocol of [73], we condition generation on each of six quantum properties from QM9-2014 [101]: dipole moment µ, polarizability α, heat capacity Cv, HOMO energy εHOMO, LUMO energy εLUMO, and HOMO–LUMO gap ∆ε. To quantify controllability, we follow prior work [73] and partition the QM9-2014 training set into two disjoint halves of 50K molecules each, Da and Db: an auxiliary property predictor ϕc [44] is trained on Da and used to estimate the property values of generated molecules, while BioMatrix is trained on Db. We then report the mean absolute error (MAE) between the specified target value and the prediction of ϕc on each generated sample. Following [73], we additionally include ϕc’s own MAE on Db as a lower bound (L-Bound) that reflects the predictor’s intrinsic error and is unreachable by any generator.

- Table 17: Performance (MAE) comparison of conditional 3D molecule generation across 10,000 samples for each property on the QM9-2014 [101] dataset. The baseline results are derived from [73].

Method µ (D) α (Bohr3) Cv mol cal K εHOMO (meV) εLUMO (meV) ∆ε (meV)

L-Bound 0.043 0.09 0.040 39 36 65 EDM [44] 1.123 2.78 1.065 371 601 671 EEGSDE [12] 0.777 2.50 0.941 302 447 487 GeoLDM [141] 1.108 2.37 1.025 340 522 587 JODO [47] 0.628 1.42 0.581 226 256 335 NExT-Mol [73] 0.507 1.16 0.512 205 235 297

BioMatrix-1.7B 0.211 0.53 0.474 53 59 84 BioMatrix-4B 0.211 0.47 0.288 57 54 81

As shown in Table 17, BioMatrix substantially advances the state of the art across all six properties. BioMatrix-4B attains the best MAE on five of six targets and the second-best on the remaining one, with BioMatrix-1.7B occupying the complementary positions and itself already surpassing every baseline on every property. On the electronic-structure targets in particular, the improvement over the strongest prior method NExT-Mol [73] is especially pronounced: MAE drops from 205 to 53meV (εHOMO), 235 to 54meV (εLUMO), and 297 to 81meV (∆ε), representing roughly a 3–4× error reduction. On the remaining three properties, BioMatrix-4B likewise more than halves NExT-Mol’s error (e.g., µ:

- 0.507 → 0.211; α: 1.16 → 0.470).

Joint tokenization of properties and geometry yields outsized gains on property-conditioned conformer generation. On the three electronic-structure targets of QM9-2014, BioMatrix-4B reduces MAE over the strongest prior method NExT-Mol by roughly 3–4× (εHOMO: 205 → 53 meV; εLUMO: 235 → 54 meV; ∆ε: 297 → 81 meV), and more than halves the error on the remaining three. The relative improvement here is substantially larger than what we observe on unconditional conformer generation against the same baseline, suggesting that exposing target property values in the same discrete token space as the structure under a single next-token objective is a particularly effective form of conditioning.

#### 7 Protein Tasks

We next evaluate BioMatrix on protein tasks spanning both sequence-based and structure-based settings, as summarized in Figure 6. Section 7.1 covers tasks that take or produce amino acid sequences together with natural language, while Section 7.2 covers tasks involving backbone geometry.

[Figure 17]

- 6 Structure Understanding AA sequence + 3D structure + question

-> text answer

- 7 Structure Prediction (Folding) AA sequence -> 3D backbone structure
- 8 Inverse Folding 3D backbone structure -> AA sequence
- 9 Sequence-Structure Co-generation

Generate sequence and structure jointly seq -> struct or struct -> seq

- 10 Unconditional Backbone Generation

###### Sequence Understanding

- 1
- 2

- 3
- 4
- 5

AA sequence -> EC number / fold type / localization

[Figure 18]

###### BioMatrix Protein Interface

###### Annotation Prediction

[Figure 19]

1D AA tokens + 3D structure tokens + text prompts

AA sequence -> GO / keywords / function text, catalytic activity, domains, motifs

[Figure 20]

###### Knowledge Mining

Gene symbol or name

- -> tissue / cancer association

Text-based Protein Design

Natural-language requirements

- -> AA sequence

Unconditional Sequence Generation

No condition

- -> foldable AA sequence

Example Unified Token Stream

<A M> <A K> <A W> ... text prompt

###### 1D token

<S 4012> <S 153> <S 2091> ... text answer

No condition

-> designable protein backbone

Amino Acid (AA) sequence 3D token Protein structure Text Natural-language instruction or answer

1D token

Figure 6: Overview of the protein task suite evaluated in this section.

###### Key takeaways on protein tasks.

- • Parameter-efficient state-of-the-art across protein understanding. Across sequence understanding, annotation prediction, and knowledge mining, BioMatrix consistently matches or exceeds the performance of substantially larger baselines (including the 8B-scale SciReasoner and OPI-Llama variants) at both the 1.7B and 4B scales, while frontier general-purpose LLMs such as GPT-O3 and DeepSeek-R1 fail to generalize to these domains.
- • Residue-aligned tokenization makes folding and inverse folding strict symmetric duals. By preserving a strict one-to-one correspondence between each AA token and its per-residue structure token, folding and inverse folding reduce to clean token-to-token alignment problems on the same vocabulary. BioMatrix achieves competitive folding accuracy and state-of-the-art amino acid recovery on inverse folding without any specific architecture or design.
- • Designable backbones and foldable sequences emerge as the default mode of the nexttoken prior. On sequence–structure co-generation, unconditional backbone generation, and unconditional sequence generation, BioMatrix matches or surpasses dedicated specialists (RFDiffusion, FrameDiff, FoldFlow) and the strongest multimodal baseline DPLM-2 on every standard self-consistency metric, in some cases approaching the native-PDB reference. The consistency of this advantage across decoding orders and conditioning settings suggests that high-quality joint generation is a natural property of the unified token space.
- • Text-grounded protein design without specialist architecture. On text-based protein design, BioMatrix generates sequences that are simultaneously diverse, low-redundancy, and structurally plausible across both CAMEO and MolInst-SwissProtCLAP, remaining competitive with task-specialized baselines under a single unified backbone.

###### 7.1 1D Protein Tasks

###### 7.1.1 Sequence Understanding

This suite of tasks evaluates whether a model can infer high-level biological properties directly from protein primary sequences, including enzyme function annotation, structural family recognition, and cellular destination prediction. Concretely, we consider four sequence understanding subsets from the OPI benchmark [138]: EC Number Prediction, in which the model predicts Enzyme Commission (EC) annotations from sequence alone; Fold Type Prediction, which evaluates hierarchical structural classification at the family, superfamily, and fold levels; and subcellular localization prediction, which measures whether the model can identify the cellular compartment in which a protein is expected to reside. Together, these tasks probe complementary aspects of protein sequence understanding, including sequence-to-function mapping, sequence-to-structure abstraction, and sequence-to-localization reasoning. Following the evaluation protocol established in [138], we employ specific metrics tailored to the nature of each task: Precision, Recall, and the F1-score are utilized for EC number prediction to account for the multi-level classification structure of the task, whereas Accuracy is reported for fold type identification and subcellular localization tasks. These metrics collectively measure the proficiency of the model in recognizing conserved sequence motifs and capturing the global structural and functional implications encoded within the primary sequence.

Results for EC number prediction are summarized in Table 18. The proposed models demonstrate a substantial improvement in performance compared to existing LLM baselines. Notably, BioMatrix-

- 1.7B achieves an F1-score of 34.34% on the CLEAN EC Number Price dataset, which significantly outperforms SciReasoner-1.7B (10.00%) and even surpasses the SciReasoner-8B (22.00%) by a large margin. On the CLEAN EC Number New split, BioMatrix-4B remains competitive with SciReasoner-8B (F1: 43.54% vs. 54.00%) despite using roughly half the parameters, with comparable Recall (44.13% vs. 44.00%) but lower Precision. These results suggest that BioMatrix is particularly effective on the more challenging Price split, while remaining competitive on the New split at substantially smaller scale. As shown in Table 19, BioMatrix also demonstrates strong performance on structural and localization

- Table 18: Performance comparison of protein sequence understanding tasks across EC number prediction benchmark. The baseline results are derived from [138, 132]

Model

CLEAN EC Number Price CLEAN EC Number New Precision Recall F1 Precision Recall F1

OPI-Galactica-6.7B [138] 2.68 2.68 2.68 27.00 26.63 25.96 OPI-Llama-3.1-8B-Instruct [138] 7.38 7.38 7.38 37.24 33.74 34.68 SciReasoner-1.7B [132] 10.00 10.00 10.00 40.00 39.00 39.00 SciReasoner-8B [132] 21.00 23.00 22.00 55.00 44.00 54.00

BioMatrix-1.7B 33.89 35.57 34.34 42.86 43.37 43.03 BioMatrix-4B 27.74 28.86 28.08 43.24 44.13 43.54

- Table 19: Performance comparison of fold type and subcellular localization tasks on the OPI benchmark. The baseline results are derived from [138, 132].

Fold type prediction Subcellular localization Family level Superfamily level Fold level Hold-out

Model

Gemini 2.5-pro [35] – – – 2.00 GPT-O3 [93] – – – 8.00 DeepSeek-R1 [39] – – – 5.00 OPI-Galactica-6.7B [138] 49.00 13.00 8.00 78.00 OPI-Llama-3.1-8B-Instruct [138] 61.00 15.00 10.00 42.00 SciReasoner-1.7B [132] – – – 84.00 SciReasoner-8B [132] 83.40 28.63 15.04 91.00

BioMatrix-1.7B 85.84 29.51 16.71 89.90 BioMatrix-4B 87.25 30.22 15.60 90.69

tasks. In fold type prediction, BioMatrix-4B achieves state-of-the-art Accuracy at both the Family level with 87.25% and the Superfamily level with 30.22%, consistently outperforming all baselines. Even at the more challenging Fold level, which requires remote homology detection from distant evolutionary sequences, BioMatrix-1.7B attains an Accuracy of 16.71%, the highest among all evaluated models. For subcellular localization, while general-purpose models such as GPT-O3 and DeepSeek-R1 fail to generalize to this domain (scoring below 10%), BioMatrix-4B achieves a high Accuracy of 90.69%, approaching the performance of SciReasoner-8B (91.00%) despite having only half the parameter count. Collectively, these results highlight the strong capability of BioMatrix to infer global structural and functional properties from primary sequences.

BioMatrix consistently achieves strong performance and high parameter efficiency across diverse protein sequence understanding benchmarks. Across tasks covering enzyme function prediction, structural classification, and subcellular localization, BioMatrix matches or exceeds the performance of substantially larger baselines, demonstrating the effectiveness of the training strategy used in BioMatrix. These results indicate that BioMatrix is highly effective in capturing both fine-grained functional motifs and global structural patterns from primary protein sequences, further highlighting the potential of BioMatrix as a strong foundation for comprehensive proteomic annotation, particularly for inferring complex biological properties from sequence alone.

###### 7.1.2 Annotation Prediction

This task evaluates the model’s ability to generate biological annotations for a given protein sequence, with target outputs ranging from controlled-vocabulary labels (e.g., Gene Ontology terms, UniProt keywords) to free-form descriptions of function, catalytic activity, and domain composition. We evaluate on a combined suite of 13 sub-tasks derived from the OPI [138] and Mol-Instructions [28] datasets. The OPI components focus on high-level biological classifications, including Gene Ontology (GO) terms, Keywords, and Function. To broaden the scope of functional characterization, we further incorporate protein-oriented tasks from Mol-Instructions, specifically those focused on predicting biological functions, enzymatic catalytic activities, structural domains and functional motifs, and broader physiological roles. We treat these tasks as an extended part of the annotation-prediction category, as they require the model to map primary amino acid sequences to fine-grained biochemical properties. Following the evaluation protocol of [138, 28], we use metrics that are appropriate for each output format. For categorical tasks, including keyword prediction and GO term prediction, we report precision, recall, and F1-score to evaluate label prediction quality. For the OPI Function task and all Mol-Instructions subsets, we report ROUGE-L to measure the semantic consistency between the generated text and reference descriptions. Results are summarized in Table 20 and Table 21.

From the results, we observe substantial performance gap between BioMatrix and general-purpose LLMs. Frontier models such as GPT-O3 and DeepSeek-R1 produce near-zero F1 scores on categorical tasks, indicating a fundamental inability to map sequences to structured biological ontologies without external assistance. In contrast, BioMatrix-4B achieves a Keyword F1 of 91.26% and a GO term F1 of 86.33% on the UniProtSeq dataset. These results indicate that, through specialized training, the model has progressed beyond simple pattern recognition and has internalized complex hierarchical relationships within biological databases. Results on the stratified test sets of OPI further demonstrate that the proposed model maintains high fidelity even as sequence similarity to the training distribution decreases. BioMatrix-4B consistently outperforms SciReasoner-8B on these challenging splits, suggesting that the model has developed robust structural–functional intuition rather than relying on shallow sequence memorization. On the Mol-Instructions suite, BioMatrix establishes a new state-of-the-art performance level. Specifically, BioMatrix-4B achieves a ROUGE-L score of 79.20% in General Function prediction, substantially exceeding the performance of tool-augmented models and larger-scale baselines such as Mol-Instructions-7B. This improvement is particularly evident in specialized domains such as Catalytic Activity, where BioMatrix-1.7B achieves a ROUGE-L of 73.45%, demonstrating a strong capability to generate expert-level biochemical descriptions. Notably, both scale variants of BioMatrix consistently outperform existing specialized architectures and even high-parameter models across all protein-related metrics. These findings highlight the potential of BioMatrix as a powerful autonomous annotator that delivers high-quality functional insights directly from primary sequences.

###### 7.1.3 Knowledge Mining

To evaluate the logic and reasoning capabilities of models within the protein domain, we introduce the Knowledge Mining (KM) category. This category assesses the capability of the model to extract and associate high-level biological knowledge—such as tissue specificity and disease associations—directly from protein contexts (e.g., gene symbols or names), moving beyond simple sequence-to-function mapping. Following the OPI benchmark [138], this task focuses on open-ended knowledge extraction from protein-related text and identifiers. We evaluate models across three key subsets: gSymbol2Tissue (predicting tissue location from gene symbols), gSymbol2Cancer (predicting cancer associations from gene symbols), and gName2Cancer (predicting cancer associations from gene names). These datasets are derived from structured biological databases, including the Human Protein Atlas and COSMIC. As these tasks require precise entity retrieval, we employ Precision, Recall, and the F1-score as the primary evaluation metrics.

The results are summarized in Table 22. In the gSymbol2Tissue task, BioMatrix-1.7B achieves a Precision

- Table 20: Performance comparison of annotation prediction tasks on the OPI dataset. The baseline results are derived from [138, 132].

Keywords GO terms Function Precision Recall F1 Precision Recall F1 Rouge-L

Model Testing data

CASPSimilarSeq 4.00 3.00 3.00 0.00 0.00 0.00 1.00 IDFilterSeq 3.00 2.00 1.00 0.00 0.00 0.00 1.00 UniProtSeq 1.00 0.00 0.00 0.00 0.00 0.00 1.00

Gemini 2.5-pro [35]

CASPSimilarSeq 21.00 19.00 19.00 0.00 0.00 0.00 1.00 IDFilterSeq 17.00 16.00 16.00 20.00 18.00 18.00 1.00 UniProtSeq 0.00 18.00 18.00 0.00 0.00 0.00 2.00

GPT-O3 [93]

CASPSimilarSeq 0.00 4.00 1.00 0.00 0.00 0.00 2.00 IDFilterSeq 0.00 0.00 1.00 0.00 0.00 0.00 1.00 UniProtSeq 1.00 0.00 0.00 0.00 0.00 0.00 1.00

DeepSeek-R1 [39]

CASPSimilarSeq 0.00 2.00 1.00 0.00 0.00 0.00 2.00 IDFilterSeq 0.00 2.00 1.00 0.00 0.00 0.00 3.00 UniProtSeq 1.00 3.00 1.00 0.00 0.00 0.00 3.00

GPT-OSS [6]

CASPSimilarSeq 81.20 73.60 76.43 76.13 74.92 74.76 74.30 IDFilterSeq 83.77 80.19 80.70 74.04 72.74 72.07 70.14 UniProtSeq 85.96 81.96 82.76 76.38 73.73 73.58 71.33

OPI-Galactica-6.7B [138]

CASPSimilarSeq 42.02 50.57 43.85 11.13 9.36 9.90 75.24 IDFilterSeq 67.62 69.05 66.50 66.86 62.87 63.04 47.86 UniProtSeq 76.06 74.89 73.74 71.50 68.97 68.49 51.44

OPI-Llama-3.1-8B-Instruct [138]

CASPSimilarSeq 83.00 80.00 81.00 81.00 81.00 80.00 83.00 IDFilterSeq 86.00 85.00 85.00 78.00 77.00 77.00 77.00 UniProtSeq 85.00 90.00 88.00 84.00 82.00 81.00 84.00

SciReasoner-1.7B [132]

CASPSimilarSeq 86.00 83.00 84.00 86.00 86.00 86.00 85.00 IDFilterSeq 89.00 89.00 88.00 83.00 82.00 81.00 82.00 UniProtSeq 92.00 91.00 91.00 88.00 87.00 86.00 88.00

SciReasoner-8B [132]

CASPSimilarSeq 83.87 82.06 81.71 83.59 81.82 82.21 81.96 IDFilterSeq 89.03 88.81 87.73 84.70 83.41 82.61 83.61 UniProtSeq 92.42 91.33 91.02 88.23 86.73 86.11 89.05

BioMatrix-1.7B

CASPSimilarSeq 84.64 81.52 81.66 83.52 80.38 81.13 82.10 IDFilterSeq 90.08 88.57 88.04 84.58 83.21 82.44 84.35 UniProtSeq 92.77 91.38 91.26 88.40 86.96 86.33 89.15

BioMatrix-4B

of 41.64%, which outperforms all baseline models, including the 8B-parameter SciReasoner-8B (41.00%) and OPI-Llama-3.1-8B (40.02%). While general-purpose frontier models, such as Gemini 2.5-pro and DeepSeek-R1, exhibit near-zero performance regarding cancer-related associations. This stark contrast suggests that even the most advanced generic LLMs fail to internalize specialized scientific facts, likely due to a lack of targeted exposure during the pre-training phase and the strict exact-match constraints inherent to biological identifiers.

- Table 21: Performance comparison of protein-related tasks on the Mol-Instructions dataset. The baseline results are derived from [26]. Results are reported in ROUGE-1 / ROUGE-L. † indicates that results obtained by supervised fine-tuning on SciReasoner [132] data based on our CPT model.

Protein Function

Catalytic Activity

Domain Motif

General Function

Model

Avg. Supervised Finetuning

BioMedGPT [79] 5.98 / 4.28 7.97 / 6.28 1.81 / 1.81 4.84 / 4.12 5.15 / 4.12 ProtT3 [72] 15.46 / 10.81 17.36 / 12.41 16.83 / 12.22 19.49 / 15.96 17.28 / 12.85 Prot2Text [3] 16.56 / 11.61 18.24 / 13.05 11.49 / 9.37 49.14 / 47.38 23.86 / 20.35 Qwen2.5-3B-SFT [49] 40.74 / 30.85 41.45 / 34.52 42.60 / 32.13 33.95 / 25.68 39.69 / 30.79

Text-based Reasoning

BioMedGPT-R1 [79] 35.16 / 26.80 27.64 / 22.22 30.60 / 23.33 27.78 / 20.82 30.30 / 23.29 Qwen2.5-3B-R1 [49] 49.10 / 38.04 61.33 / 46.54 51.01 / 41.10 42.56 / 32.37 51.00 / 39.51

Online LLM Baseline

DeepSeek-R1 [39] 26.65 / 19.68 21.86 / 18.02 26.09 / 19.70 16.53 / 10.76 22.78 / 17.04 Kimi-k2 [122] 25.08 / 17.32 22.99 / 17.97 27.28 / 20.07 22.30 / 14.80 24.41 / 17.54 Qwen3 [142] 18.53 / 12.62 15.93 / 12.27 15.13 / 11.28 18.59 / 11.94 17.05 / 12.03

Multi-Source RAG

DeepSeek-R1 [39] 38.23 / 25.68 32.36 / 23.75 30.35 / 21.46 28.54 / 16.73 32.37 / 21.91 Kimi-k2 [122] 22.30 / 14.14 18.56 / 13.15 21.64 / 17.12 45.41 / 26.44 26.98 / 17.71 Qwen3 [142] 37.42 / 24.52 50.66 / 40.67 16.44 / 12.44 42.61 / 26.34 36.78 / 25.99

Tool-Powered Reasoning

DeepSeek-R1 [39] 59.71 / 38.31 47.95 / 34.15 48.36 / 34.36 58.11 / 35.70 53.53 / 35.63 Kimi-k2 [122] 57.98 / 35.73 67.68 / 46.90 39.76 / 28.56 60.08 / 36.78 56.38 / 36.99 Qwen3 (PFUA) [142] 66.43 / 44.29 72.32 / 54.18 54.98 / 44.26 63.60 / 41.90 64.33 / 46.16

GPT-OSS [6] - / 5.00 - / 2.00 - / 2.00 - / 7.00 - / 4.00 BioT5+ [96] - / 57.00 - / 72.00 - / 53.00 - / 68.00 - / 62.50 Mol-Instructions-7B [28] - / 43.00 - / 52.00 - / 46.00 - / 44.00 - / 46.25 SciReasoner-data-1.7B† 49.67 / 42.90 48.41 / 46.99 52.66 / 49.70 30.12 / 26.23 45.22 / 41.46 SciReasoner-data-4B† 59.93 / 53.21 53.87 / 52.54 48.30 / 45.46 69.07 / 67.60 57.95 / 54.70

BioMatrix-1.7B 68.46 / 61.88 74.09 / 73.45 56.16 / 52.55 79.00 / 78.03 69.43 / 66.48 BioMatrix-4B 68.56 / 62.09 73.46 / 73.06 60.36 / 56.22 80.19 / 79.20 70.64 / 67.64

BioMatrix exhibits superior biological knowledge synthesis and exceptional parameter efficiency, particularly in resolving complex entity-disease associations. In the Knowledge Mining suite, BioMatrix consistently outperforms general-purpose frontier models and achieves state-of-the-art results on cancer association tasks, effectively overcoming the domain-gap barriers where generic LLMs fundamentally fail. Notably, BioMatrix-4B delivers performance on par with or superior to SciReasoner-8B, demonstrating that strategic alignment with specialized biological datasets can effectively compensate for a smaller parameter footprint. These findings suggest that BioMatrix possesses a robust capacity for resolving synonymous biological nomenclature and navigating intricate clinical metadata.

###### 7.1.4 Text-based Protein Design

Text-based protein design evaluates whether a model can generate amino-acid sequences that are simultaneously consistent with a natural-language description of function, evolutionarily plausible as

- Table 22: Performance comparison of knowledge mining tasks on the OPI dataset. The baseline results are derived from [138, 132].

Model

gSymbol2Tissue gSymbol2Cancer gName2Cancer Precision Recall F1 Precision Recall F1 Precision Recall F1

Gemini 2.5-pro [35] 1.00 1.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 GPT-O3 [93] 38.00 68.00 45.00 11.00 10.00 10.00 11.00 10.00 10.00 DeepSeek-R1 [39] 3.00 9.00 4.00 0.00 2.00 0.00 0.00 3.00 0.00 GPT-OSS [6] 3.00 9.00 4.00 1.00 2.00 1.00 1.00 4.00 1.00 OPI-Galactica-6.7B [138] 39.17 90.77 53.03 35.55 31.89 32.29 27.28 25.54 25.33 OPI-Llama-3.1-8B-Instruct [138] 40.02 93.56 54.66 28.90 27.01 26.64 27.86 27.07 26.59 SciReasoner-1.7B [132] 41.00 82.00 51.00 75.00 72.00 73.00 68.00 67.00 67.00 SciReasoner-8B [132] 41.00 83.00 52.00 81.00 81.00 81.00 80.00 79.00 79.00

BioMatrix-1.7B 41.64 88.00 53.89 60.25 63.32 60.26 76.63 77.05 76.18 BioMatrix-4B 40.62 86.86 52.89 81.27 79.55 79.53 79.22 79.05 79.07

protein sequences, and likely to fold into stable structures. In other words, this task examines crossmodal semantic grounding from text to protein sequence while also assessing whether the generated sequences remain biophysically realistic, rather than merely matching the textual prompt at a superficial level. We follow the evaluation protocol adopted in ProDVa [66] and report results on two text–protein benchmarks introduced in prior work: the CAMEO subset and the MolInstruct-SwissProtCLAP subset. The CAMEO split contains protein design prompts associated with experimentally characterized proteins and is primarily used to evaluate the general quality of text-to-protein generation under realistic functional descriptions. The MolInstruct-SwissProtCLAP subset is constructed from protein–text pairs curated from MolInstruct and SwissProtCLAP, and provides a complementary benchmark with broader textual annotations and function-oriented descriptions. Following previous work, all methods are evaluated under the same text-conditioned generation setting, and the generated sequences are further folded by ESMFold for structure-based assessment.

- Table 23: Performance comparison for the text-conditioned protein design task on the CAMEO dataset. Baseline results are collected from [66].

Seq. Plausibility Foldability Lang. Alignment (%) Seq. Div.

Models

PPL ↓ Rep ↓ pLDDT ↑ % > 70 ↑ PAE ↓ % < 10 ↑ ProTrek ↑ Key. Rec. ↑ Retr. Acc. ↑ ↑

ProteinDT [67] 1405.70 0.11 38.70 0.20 26.25 0.00 3.89 0.05 7.43 99.72 ProteinDTFT [67] 1860.43 0.04 38.66 1.04 23.90 0.42 6.28 1.08 16.57 99.32 Pinal [22] 584.22 0.15 66.50 47.21 14.57 33.53 14.57 30.46 51.68 82.72 PAAG [147] 2571.40 0.02 33.14 0.00 23.31 0.00 5.21 0.23 7.10 99.02 PAAGFT [147] 2004.01 0.04 41.53 1.12 24.34 0.46 3.46 0.01 7.82 99.87 Chroma [51] 1322.37 0.03 61.66 28.96 13.01 39.03 2.97 0.11 6.57 97.21 ESM3 [42] 279.78 0.33 59.79 31.49 17.40 21.37 3.76 5.49 11.97 96.77 ProDVa [66] 656.04 0.01 75.88 77.00 6.39 83.88 14.43 30.34 44.77 98.58 BioMatrix-1.7B 762.80 0.09 65.64 50.43 12.94 45.30 7.20 0.98 8.42 97.78 BioMatrix-4B 631.89 0.07 60.92 43.36 15.85 30.55 8.42 4.73 12.23 88.95

Results on the CAMEO and MolInstruct-SwissProtCLAP subsets are shown in Table 23 and Table 24. We assess model performance from four perspectives: sequence plausibility, measured by perplexity (PPL; lower is better) and repetition rate (Rep; lower is better) [134]; foldability, measured by mean pLDDT [4], the percentage of designs with pLDDT greater than 70, mean predicted aligned error

- Table 24: Performance comparison for the text-conditioned protein design task on the MolinstSwissProtCLAP dataset. Baseline results are collected from [66].

Seq. Plausibility Foldability Lang. Alignment (%) Seq. Div.

Models

PPL ↓ Rep ↓ pLDDT ↑ % > 70 ↑ PAE ↓ % < 10 ↑ ProTrek ↑ EvoLlama ↑ Retr. Acc. ↑ ↑

ProteinDT [67] 1576.23 0.07 38.29 0.98 25.13 0.40 1.20 40.57 9.28 99.23 ProteinDTFT [67] 1213.38 0.04 51.42 25.61 18.57 23.92 13.89 52.84 47.29 79.87 Pinal [22] 308.97 0.13 75.25 68.97 10.96 58.44 17.50 53.42 57.95 82.96 PAAG [147] 2782.70 0.02 28.39 0.07 25.38 0.10 1.29 34.39 7.06 99.15 PAAGFT [147] 1332.35 0.04 50.37 23.86 19.96 21.99 10.04 49.69 33.66 86.09 Chroma [51] 1370.21 0.03 59.18 20.17 15.03 28.62 2.10 40.10 7.33 96.13 ProDVa [66] 415.63 0.02 76.86 76.35 8.66 68.06 17.40 51.10 59.07 83.29 BioMatrix-1.7B 581.42 0.05 69.76 60.18 11.25 56.34 9.24 37.56 23.14 86.08 BioMatrix-4B 482.08 0.06 71.89 64.01 11.07 56.98 11.70 38.40 31.08 80.31

(PAE) [4], and the percentage of designs with PAE smaller than 10 after ESMFold [65] prediction; language alignment, measured by ProTrek [115] score, keyword recovery, and retrieval accuracy; and sequence diversity (Seq. Div.), which quantifies the diversity of generated sequences across prompts. Overall, BioMatrix demonstrates strong and balanced performance across both benchmarks, with complementary strengths at different model scales. On the CAMEO subset, the 1.7B-parameter version of BioMatrix shows stronger structural quality, reaching 65.64 pLDDT, 50.43% of designs above the threshold of 70, 12.94 PAE, and 45.30% of designs below PAE 10. These results highlight its strength in foldability and rank it as the second-best method overall on the % < 10 metric. At the same time, the 4B-parameter version delivers clear advantages in sequence plausibility and text alignment, reducing PPL to 631.89 and outperforming the 1.7B-parameter version on ProTrek similarity, keyword recall, and retrieval accuracy, which highlights the ability of BioMatrix to capture semantically meaningful textto-protein relationships while maintaining competitive generation quality. This advantage becomes even more evident on the MolInstruct-SwissProtCLAP subset. The 4B-parameter version consistently improves over the 1.7B-parameter version in sequence plausibility, language alignment, and foldability, reducing PPL from 581.42 to 482.08 and increasing ProTrek similarity, EvoLlama score, and retrieval accuracy from 9.24, 37.56, and 23.14 to 11.70, 38.40, and 31.08, respectively. Specialist baselines such as ProDVa and Pinal remain ahead on foldability and certain alignment metrics, which is expected given their dedicated design on this task; BioMatrix remains competitive under a single unified backbone.

BioMatrix consistently generates diverse, low-redundancy, and structurally plausible proteins across both text-conditioned design benchmarks. Across CAMEO and MolInst-SwissProtCLAP, BioMatrix maintains strong foldability, with pLDDT values exceeding 60 on CAMEO and approaching 70 on MolInst-SwissProtCLAP. This structural quality is accompanied by low sequence repetition and high sequence diversity, indicating that BioMatrix retains broad generative capacity rather than relying on narrow or degenerate design modes. The consistency of these patterns across benchmarks with distinct prompt distributions and sequence compositions further highlights the robustness of BioMatrix for open-ended protein generation from natural-language instructions.

###### 7.1.5 Unconditional Sequence Generation.

This task evaluates the model’s ability to generate foldable amino acid sequences from scratch—i.e., sequences whose ESMFold-predicted structures are high-confidence rather than disordered or randomlike. Following the protocol adopted by DPLM-2 [131], we draw unconditional AA sequences from the model, fold each one with ESMFold [65], and report the mean pLDDT over the resulting predictions as the primary structural-plausibility indicator. Higher pLDDT corresponds to higher confidence that

the generated sequence encodes a well-defined fold; random or low-complexity sequences typically produce pLDDT in the 30–50 range, while plausibly foldable sequences concentrate above 80.

We apply the same length-alignment protocol as in Section 7.2.4. Table 27 shows that BioMatrix-1.7B and BioMatrix-4B produce sequences whose ESMFold-predicted structures reach mean pLDDT of 85.28 and 85.01 respectively. Both numbers exceed DPLM-650M [130] (83.25) and DPLM-2-650M (82.25), and are almost 50 points above EvoDiff [7] (35.85), whose low pLDDT reflects sequences that ESMFold cannot confidently fold.

###### 7.2 3D Protein Tasks

###### 7.2.1 Structure Understanding

This task evaluates the model’s ability to answer natural-language questions about a protein given both its amino acid sequence and backbone structure—testing whether the model can draw on structural context to describe protein properties in free-form text. We evaluate on the Protein Function Understanding Dataset (PFUD) [82], a structure-based question-answering benchmark that covers six categories of inquiry: molecular function, subcellular location, biological process, domains and motifs, a summary of functional characteristics, and composite queries that jointly ask about multiple attributes. Since the answers are free-form descriptions rather than class labels, we follow the original protocol and report BLEU-2 and ROUGE-1/2/L scores against the reference responses.

- Table 25: Performance comparison of protein structure understanding task on the PFUD [82] dataset. The baseline results are derived from [82].

Model BLEU-2 ROUGE-1 ROUGE-2 ROUGE-L

Llama3-Instruct [36] 2.08 15.91 2.67 5.81 BioMedGPT-LM-10B [79] 2.41 18.91 2.99 14.89

- Llama2-molinst-7B [28] 26.25 45.24 23.47 38.15
- Llama3-AAseq-FT [36] 37.10 59.65 37.72 52.50 ProtT3-FT [72] 40.79 61.97 42.53 56.98 ProtTeXLlama3 [82] 41.54 63.46 43.17 57.89

BioMatrix-1.7B 45.17 63.26 43.92 57.88 BioMatrix-4B 46.07 63.73 44.45 58.38

As summarized in Table 25, BioMatrix-4B sets a new state-of-the-art across all four metrics, while BioMatrix-1.7B remains highly competitive with the strongest prior baseline ProtTeXLlama3 [82]—a larger model built on Llama3-8B that also incorporates structural tokens—and even surpasses it on BLEU-2 and ROUGE-2. BioMatrix-4B further extends the margin. We attribute this to BioMatrix’s exposure to richly annotated protein corpora during continual pretraining—in particular Swiss-Prot entries paired with their AFDB-predicted structures and serialized alongside functional metadata such as subcellular localization, keywords, and free-text comments (Section 3.1.3)—which grounds the joint sequence–structure token space in the same functional vocabulary.

###### 7.2.2 Structure Prediction (Folding)

This task evaluates the model’s ability to predict a protein backbone structure directly from its amino acid sequence—cast in BioMatrix as autoregressive decoding of per-residue structure tokens conditioned on the AA-token sequence in the joint vocabulary (Section 2.3). We evaluate on the PDB date split introduced by Multiflow [18] and report TM-score and RMSD between predicted and reference backbones. Unlike encoder-based folding models that structurally guarantee an output of the same length as the input sequence, an autoregressive decoder terminates whenever it emits an

- Table 26: Performance comparison of structure prediction (folding) and inverse folding tasks on the PDB date split [18] dataset. The baseline results are derived from [131].

Folding Inverse Folding Models RMSD ↓ TMscore ↑ AAR ↑ scTM ↑ ESMFold [65] 2.84 0.93 – – MultiFlow [18] 15.64 0.53 37.74 0.94 ESM3 [42] 4.94 0.87 49.50 0.94 DPLM-2-150M [131] 8.35 0.76 48.83 0.89 DPLM-2-650M [131] 5.67 0.83 54.80 0.91 DPLM-2-3B [131] 4.54 0.86 61.67 0.92

BioMatrix-1.7B 5.49 0.84 75.20 0.85 BioMatrix-4B 5.38 0.84 75.50 0.86

end-of-structure token, which does not in general align with the residue count of the query. Since RMSD is only defined between backbones of matching length, we address this mismatch purely at inference time rather than through any architectural modification: for each input sequence we draw 200 samples under nucleus sampling with a deliberately elevated temperature to promote length diversity, and retain a sample whose decoded structure length matches the input. This procedure is applied uniformly to all BioMatrix results reported in Table 26 and does not involve any oracle selection on TM/RMSD—the filter is strictly on length.

As shown in Table 26, BioMatrix sits firmly in the top tier of multimodal generalist models on this task. Both BioMatrix-1.7B (TM = 0.84 / RMSD = 5.49) and BioMatrix-4B (TM = 0.84 / RMSD = 5.38) surpass DPLM-2 at the 150M and 650M scales and closely track DPLM-2-3B (TM = 0.86 / RMSD = 4.54), despite operating under a shared vocabulary that must simultaneously accommodate molecules, proteins, and text rather than a protein-only token space. The residual gap to ESMFold (TM = 0.93 / RMSD = 2.84), a dedicated folding model whose architecture and training objective are tailored end-to-end to this single task, reflects the expected trade-off between a unified next-tokenprediction backbone and a hand-designed folding specialist, rather than a failure of the unification hypothesis itself. We attribute BioMatrix’s competitiveness here to the residue-aligned tokenization described in Section 2.3, which preserves a strict one-to-one correspondence between each AA token and its per-residue GCP-VQVAE structure token, turning folding into a clean token-level alignment problem that is directly compatible with next-token prediction.

###### 7.2.3 Inverse Folding

This task evaluates the reverse direction of folding: given a backbone structure, predict an amino acid sequence that would plausibly fold into it. In BioMatrix, this is cast as autoregressive decoding of AA tokens conditioned on the per-residue GCP-VQVAE structure-token sequence in the joint vocabulary (Section 2.3), making folding and inverse folding strictly symmetric operations on the same residuealigned token interface. We evaluate on the same PDB date split as in the folding task, and report two complementary metrics that target two distinct notions of success. Amino acid recovery (AAR) measures per-residue agreement between the predicted sequence and the native reference, and directly quantifies how well the model reproduces the original sequence. Self-consistency TM-score (scTM) instead measures the TM-score between the native backbone and the ESMFold [65]-predicted structure of the generated sequence. Because many different sequences can adopt the same fold, AAR and scTM are only loosely correlated and the two metrics reward different behaviors. As in the folding task, an autoregressive decoder does not structurally guarantee that its output length matches the query, and AAR is only defined when the predicted AA sequence has the same length as the input backbone; we

therefore apply the same length-filtering protocol as above, drawing 50 nucleus-sampled predictions per input with an elevated temperature and retaining a length-matched sample.

- As shown in Table 26, BioMatrix achieves the highest AAR by a substantial margin: BioMatrix-4B reaches 75.50% and BioMatrix-1.7B reaches 75.20%. The high AAR indicates that BioMatrix accurately recovers the native residue identity at each position, which we attribute to two compounding factors: the residue-aligned vocabulary design of Section 2.3 turns inverse folding into a clean token-to-token mapping problem in which each GCP-VQVAE structure token is paired one-to-one with the AA token it should predict; and the protein CPT data (Section 3.1.3) directly exposes the model to the joint AA–structure token distribution under the same next-token objective used at inference. On scTM, BioMatrix-4B and BioMatrix-1.7B report 0.86 and 0.85 respectively, trailing DPLM-2-3B (0.92), ESM3 and MultiFlow (both 0.94). Both values remain well above the 0.5 threshold above which two structures are generally considered to adopt the same fold [139], and sit in a range commonly associated with high-confidence structural agreement in protein design evaluations, indicating that BioMatrix’s designed sequences do fold back to the native backbone under ESMFold. Together with the AAR result, this suggests that BioMatrix prioritizes residue-level fidelity to the native sequence, which is not guaranteed to coincide with the ESMFold-friendly sequences that scTM directly favors.

Folding and inverse folding form a clean symmetry test of the residue-aligned tokenization, with BioMatrix prioritizing native-sequence fidelity. Under a single shared vocabulary and next-token objective, BioMatrix achieves competitive folding accuracy (TM = 0.84) and state-of-the-art amino acid recovery on inverse folding (75.50% for BioMatrix-4B vs. 61.67% for DPLM-2-3B), without any direction-specific architecture or decoder head. The high AAR alongside slightly lower scTM (0.85–

- 0.86, still well above the 0.5 fold-equivalence threshold) indicates that BioMatrix’s inverse-folding behavior favors per-residue fidelity to the native sequence over ESMFold-friendly redesigns.

7.2.4 Sequence–structure Co-generation.

This task evaluates the model’s ability to jointly produce an amino acid sequence and a backbone that are mutually consistent—i.e., the generated sequence should plausibly fold into the generated structure—without any held-out test set or external conditioning. Following the evaluation protocol of DPLM-2 [131], we assess consistency through the self-consistency TM-score (scTM) and RMSD (scRMSD) between the generated backbone and the ESMFold [65]-predicted structure of the generated sequence, together with the mean pLDDT of that prediction as a structural-confidence indicator. BioMatrix supports two complementary decoding orders on the same joint vocabulary (Section 2.3): seq → struct, where the AA sequence is sampled first and the per-residue GCP-VQVAE structure tokens are decoded conditioned on it, and struct → seq, which proceeds in the reverse order.

The DPLM-2 baselines operate under absolute length control, generating exactly 100 samples at each of the target lengths {100,200,300,400,500}. An autoregressive decoder cannot enforce an exact output length by construction; we therefore draw 5000 unconditional samples under nucleus sampling and, for each target length, retain the 100 samples whose generated length lies closest to that target. No oracle filtering on scTM, scRMSD, or pLDDT is applied at any stage—selection is strictly on proximity to the target length. This protocol is used identically for all three tasks: co-generation, unconditional structure generation (Section 7.2.5), and unconditional sequence generation (Section 7.1.5).

As shown in Table 27, BioMatrix variants deliver strong self-consistency in both directions and improve substantially over DPLM-2-650M at matched or larger scale. In the seq → struct direction, BioMatrix-

- 1.7B and BioMatrix-4B reach scTM=0.965 with scRMSD of 2.81 and 2.80 respectively, versus 0.907 /

- 6.34 for DPLM-2-650M under the same decoding order. In the struct → seq direction the improvement is even more pronounced: BioMatrix-1.7B attains scTM = 0.979 / scRMSD = 1.50 and BioMatrix-4B attains 0.977 / 1.49, surpassing DPLM-2-650M’s 0.921 / 4.97 and MultiFlow’s 0.930 / 3.21, and edging above the native-PDB reference (0.904 / 4.62) on this metric combination. Mean pLDDT tells a consistent

- Table 27: Performance comparison of structure-sequence co-generation, unconditional backbone generation, and unconditional sequence generation tasks. The baseline results are derived from [131].

scTM ↑ scRMSD ↓ pLDDT ↑ Structure-sequence co-generation.

Native PDB protein 0.904 4.623 – ESM3-Open-1.4B [42] (seq → struct) 0.624 24.180 – MultiFlow [18] 0.930 3.208 79.447 DPLM-2-650M [131] (seq → struct) 0.907 6.337 82.246 DPLM-2-650M [131] (struct → seq) 0.921 4.969 81.910 DPLM-2-650M [131] (co-generation) 0.925 3.899 82.686

BioMatrix-1.7B (seq → struct) 0.965 2.811 85.276 BioMatrix-1.7B (struct → seq) 0.979 1.497 88.800 BioMatrix-4B (seq → struct) 0.965 2.800 85.012 BioMatrix-4B (struct → seq) 0.977 1.488 86.092

Unconditional backbone generation. (sequence predicted by ProteinMPNN)

Native PDB struct. (seq. from PMPNN) 0.969 0.864 – FrameDiff [145] 0.818 3.919 – FoldFlow [15] 0.540 7.965 – RFDiffusion [133] 0.914 1.969 – DPLM-2-650M [131] 0.945 4.451 –

BioMatrix-1.7B 0.956 2.986 – BioMatrix-4B 0.963 3.067 –

Unconditional sequence generation. (structures predicted by ESMFold)

EvoDiff [7] – – 35.846 DPLM-650M [130] – – 83.252 DPLM-2-650M [131] – – 82.246

BioMatrix-1.7B – – 85.276 BioMatrix-4B – – 85.012

story, rising from 82.25–82.69 for the DPLM-2-650M variants to 85.01–88.80 for BioMatrix.

###### 7.2.5 Unconditional Structure Generation

This task evaluates the model’s ability to generate protein backbones from scratch that are designablei.e., backbones for which a plausible amino acid sequence can be designed that folds back to the target structure with high fidelity. Following DPLM-2 [131], we first draw backbone structures from the model without any conditioning, then use ProteinMPNN [23] to propose sequences for each generated backbone, fold those sequences with ESMFold [65], and finally report scTM and scRMSD between the generated backbone and the ESMFold-predicted structure of the ProteinMPNN-designed sequence. We apply the same length-alignment protocol as in Section 7.2.4.

As summarized in Table 27, BioMatrix achieves the highest designability among all evaluated methods, including dedicated structure-diffusion specialists. BioMatrix-4B attains scTM=0.963 with scRMSD=3.07, and BioMatrix-1.7B reaches 0.956 / 2.99. These scTM values exceed RFDiffusion [133] (0.914), FrameDiff [145] (0.818), FoldFlow [15] (0.540), and DPLM-2-650M (0.945), and sit close to the native-PDB upper bound of 0.969. On scRMSD, BioMatrix trails RFDiffusion (1.97) but clearly outperforms DPLM-2-650M (4.45), with a 1.4–1.5A˚ absolute improvement at comparable scale. Taken

together, these numbers place BioMatrix on the joint scTM–scRMSD frontier despite generating backbones through autoregressive next-token prediction over a shared vocabulary rather than through a structure-specific diffusion or flow-matching process.

A unified next-token backbone reaches the joint scTM–scRMSD frontier on protein structure generation, without any structure-specific objective. Across sequence–structure co-generation, unconditional backbone generation, and unconditional sequence generation, BioMatrix matches or surpasses dedicated diffusion specialists (RFDiffusion, FrameDiff, FoldFlow) and the strongest multimodal baseline DPLM-2-650M on every standard self-consistency metric. On co-generation in particular, BioMatrix-1.7B in the struct→seq direction reaches scTM = 0.979 / scRMSD = 1.50, edging above the native-PDB reference (0.904 / 4.62). The consistency of this advantage—across decoding orders, with and without conditioning—suggests that designable backbones and foldable sequences emerge as the default mode of the next-token prior under residue-aligned tokenization, rather than as artifacts of any particular conditioning setup.

#### 8 Interaction Tasks

Beyond single-entity tasks, we evaluate BioMatrix on molecule–protein and protein–protein interaction tasks, which require jointly predicting over two biomolecular entities at either the sequence or the structure level. If molecules and proteins genuinely share a unified token space and the cross-entity pretraining corpus (Section 3.1.4) has effective transfer, then a variant instruction-tuned from this shared backbone should be able to reason over heterogeneous biomolecular pairs without any pairspecific architecture.

###### Key takeaways on interaction tasks.

- • No pair-specific architecture is required. Across molecule–protein and protein–protein interaction tasks in both sequence-only and structure-grounded regimes, BioMatrix achieves state-of-the-art or highly competitive performance simply by placing two heterogeneous biomolecular entities side-by-side in the same token stream and processing them under the unified next-token objective—without any pair-specific architecture, explicit pocket encoding, geometric cross-attention, or fusion module that prior specialist models rely on.
- • Cross-entity pretraining alone is enough to surpass task-specialized SOTA. On both PDBBindv2019/CASF-2016 and BindingDB, BioMatrix matches or exceeds previously reported literature SOTA.
- • Structure-grounded affinity prediction validates the unified geometric token space. On the structure-conditioned PDBBindv2020 benchmark, BioMatrix outperforms dedicated specialists that rely on trigonometry-aware cross-attention between ligand and pocket geometry. This directly demonstrates that placing the ligand’s SELFIES-aligned geometry tokens and the protein’s per-residue structure tokens in a shared vocabulary is sufficient for joint structuregrounded prediction across heterogeneous entities.

###### 8.1 1D Interaction Tasks

###### 8.1.1 Molecule–protein interaction.

This task evaluates the model’s ability to predict the binding affinity between a small molecule and a target protein purely from their sequence-level representations without access to the co-crystallized complex structure. We evaluate on two datasets: the PDBBindv2019 refined set with the CASF-2016 core set as the held-out test [70, 116] (abbreviated as PDB in Table 28), and BindingDB [68] (abbreviated as BDB), both measured by pKd and evaluated with root-mean-square error (RMSE).

- Table 28: Performance comparison of 1D interaction tasks on the PEER [140] benchmark. The baseline results are also derived from [140]. * denotes the model is used as a feature extractor with frozen pre-trained weights.

Yst Hum Aff PDB BDB ACC ↑ ACC ↑ RMSE ↓ RMSE ↓ RMSE ↓ Literature SOTA – – – 1.181 1.340

Model

DDE [104] 55.83 62.77 2.908 – – Moran [30] 53.00 54.67 2.984 – – LSTM [102] 53.62 63.75 2.853 1.457 1.572 Transformer [102] 54.12 59.58 2.499 1.455 1.566 CNN [110] 55.07 62.60 2.796 1.376 1.497 ResNet [102] 48.91 68.61 3.005 1.441 1.565 ProtBert [25] 63.72 77.32 2.195 1.562 1.549 ProtBert* [25] 53.87 83.61 2.996 1.457 1.649 ESM-1b [103] 57.00 78.17 2.281 1.559 1.556 ESM-1b* [103] 66.07 88.06 3.031 1.368 1.571

BioMatrix-1.7B 62.44 82.28 2.446 1.268 1.284 BioMatrix-4B 62.18 83.12 2.350 1.268 1.030

- As shown in Table 28, BioMatrix establishes a new state of the art on both datasets. On PDBBind/CASF2016, both BioMatrix-1.7B and BioMatrix-4B achieve an RMSE of 1.268, clearly outperforming most baselines including the pretrained protein language model ESM-1b [103]. On BindingDB, BioMatrix-4B further drives RMSE down to 1.030, substantially ahead of every baseline. Notably, BioMatrix also surpasses the previously reported literature SOTA results (1.181 on PDB and 1.340 on BDB).

Large-scale cross-entity continual pretraining matches or surpasses task-specialized SOTA on sequence-based affinity prediction. On BindingDB (RMSE = 1.030), BioMatrix clearly surpasses the prior literature SOTA (1.340), and on PDBBindv2019/CASF-2016 (RMSE = 1.268) it remains competitive with the literature SOTA (1.181) while outperforming every reported LLM and protein-

- encoder baseline. BioMatrix achieves this as a single generalist model with no task-specific architecture, indicating that the cross-entity pretraining corpus described in Section 3.1.4 already
- encodes ligand–target binding signal at a scale and granularity that approaches—and on BindingDB exceeds—specialized architectures trained from scratch on individual benchmarks.

###### 8.1.2 Protein–protein Interaction.

This task evaluates the model’s ability to predict properties of two interacting proteins jointly from their amino acid sequences alone. We consider three datasets in Table 28: two binary classification benchmarks—Yeast [40] (Yst) and Human [94] (Hum), each asking whether a given pair of proteins interacts—and one regression benchmark, PPI Affinity [86] (Aff), which targets the binding affinity of a protein–protein complex. We report classification accuracy (ACC) for the former two and RMSE for the latter.

Across all three subsets, BioMatrix delivers strong and consistent performance. On the classification benchmarks, BioMatrix-4B attains 62.18% on Yeast and 83.12% on Human, surpassing every nonpretrained baseline (DDE, Moran, LSTM, CNN, ResNet) by a substantial margin and remaining competitive with the much larger ESM-1b (66.07% / 88.06%). On the PPI Affinity regression task, BioMatrix-4B achieves an RMSE of 2.350, better than all non-pretrained baselines and within close

- Table 29: Performance comparison of 3D interaction (affinity prediction) task on the PDBBindv2020 [129] dataset. The baseline results are derived from [37, 77].

Model RMSE ↓ Pearson ↑ Spearman ↑ MAE ↓

TransformerCPI [20] 1.741 0.576 0.540 1.404 MONN [62] 1.438 0.624 0.589 1.143 PIGNet [87] 2.640 0.511 0.489 2.110 IGN [52] 1.433 0.698 0.641 1.169 HOLOPROT [111] 1.546 0.602 0.571 1.208 STAMP-DPI [128] 1.658 0.545 0.411 1.325 EGNN [105] 1.445 0.648 0.598 1.141 EGNN + TargetDiff [37] 1.374 0.680 0.637 1.118 TANKBind [77] 1.346 0.726 0.703 1.070

BioMatrix-1.7B 1.310 0.720 0.717 1.041 BioMatrix-4B 1.260 0.737 0.712 0.972

range of the top specialist ProtBert (2.195) [25]. These results show that BioMatrix matches the performance of protein-specialized encoders on pairwise protein interaction prediction tasks.

###### 8.2 3D Interaction Tasks

###### 8.2.1 Molecule–protein Interaction.

This task extends sequence-based molecule–protein affinity prediction to the structure-grounded regime, where both the ligand and the protein pocket are additionally provided as spatial structures, requiring the model to jointly predict over sequence and geometry of two distinct entities. We evaluate on PDBBindv2020 [129] following the time split [37, 77], which provides co-crystallized protein– ligand complexes annotated with experimentally measured binding affinities. The underlying affinity value may correspond to a dissociation constant (Kd), an inhibition constant (Ki), or a half-maximal inhibitory concentration (IC50), all expressed in molar concentration units; we take their negative base-10 logarithm as a unified regression target. Performance is assessed along four complementary axes: root-mean-square error (RMSE) and mean absolute error (MAE) quantify absolute prediction accuracy, while Pearson and Spearman correlations measure how well the predicted affinities preserve the ranking of the ground-truth values.

As reported in Table 29, BioMatrix-4B achieves the best result on three of the four metrics (RMSE = 1.260, Pearson=0.737, MAE=0.972) and the second best on Spearman (0.712), while BioMatrix-1.7B attains the best Spearman (0.717) and complementary second-best positions on the remaining metrics.

Cross-entity geometric reasoning emerges from shared tokenization, with no pair-specific architecture required. On structure-grounded molecule–protein affinity prediction (PDBBindv2020), BioMatrix-4B achieves the best result on three of four metrics and outperforms TankBind, the strongest specialist baseline that uses trigonometry-aware cross-attention between ligand and pocket geometry. The ligand’s SELFIES-aligned geometry tokens and the protein’s per-residue structure tokens are simply placed side-by-side in the same token stream and processed under the unified next-token objective, with no explicit pocket encoding, geometric cross-attention, or fusion module. This directly validates that placing two heterogeneous biomolecular entities in a shared discrete token space is sufficient for joint structure-grounded reasoning.

#### 9 Discussion

Across the 80 tasks reported in Sections 6, 7, and 8, several cross-cutting patterns emerge. We collect them here as observations rather than prescriptions, in the hope that they inform the design of future biological foundation models.

###### 9.1 Why Train Task-Group Variants Rather Than One Monolithic SFT Model

A natural question raised by Table 30 is why we instruction-tune BioMatrix as a small set of taskgroup-specific variants rather than as a single all-tasks SFT model. The grouping is the result of an empirical observation we encountered during development: in our setting, mixing heterogeneous biological tasks into a single SFT run rarely produces mutual gains, and on small-data tasks it actively destabilizes performance.

We attribute this to a structural difference between biological SFT and the kind of large-scale instruction tuning that has become standard in NLP. In a typical NLP SFT corpus, each instance can plausibly be treated as a distinct “task” in its own right: the surface form, the input distribution, the reasoning trajectory, and the output style all vary instance-to-instance, and the resulting diversity makes a single model absorb millions of heterogeneous instances without one task dominating another. Biological SFT data, by contrast, is much more pattern-monolithic at the instance level. Within a sub-task such as retrosynthesis, forward synthesis, or binding-affinity regression, every instance shares essentially the same input and output schema; the diversity is concentrated across sub-tasks rather than within them. As a consequence, when we mix all sub-tasks into a single SFT run, the model is exposed to a small number of dominant patterns rather than a long tail of heterogeneous ones, and the gradient from large-data sub-tasks easily overwhelms the gradient from small-data sub-tasks, with the latter showing high run-to-run variance and frequent degradation relative to dedicated training.

For the open-source release, we publish a single all-tasks SFT model on the union of all sub-task corpora, with mild oversampling of small-data sub-tasks to mitigate the imbalance described above. This unified variant is competitive on the large-data sub-tasks but still trails the task-group variants on a non-trivial fraction of small-data ones, and we have not been able to fully close this gap within the recipe explored here.

###### 9.2 SMILES vs. SELFIES: Complementary, Not Interchangeable

Because BioMatrix natively supports both SMILES and SELFIES as parallel molecular sequence notations, the molecular task suite in Section 6.1 doubles as a controlled, large-scale comparison of the two notations under an otherwise identical training recipe. The picture that emerges is that neither notation dominates the other: each has a regime in which it is clearly preferable, and the regime is determined by a single underlying property—whether the task requires surface-level structural anchoring or distribution-level chemical validity.

###### SELFIES wins where validity-by-construction matters. On unconditional molecular generation

- (Section 6.1.1), the SELFIES variants of BioMatrix attain 0.995–0.999 validity on both MOSES and GuacaMol, essentially saturating the metric, while SMILES variants sit at 0.951–0.994. On the MolOptInstructions benchmark (Section 6.1.8), where the model must shift a target property in a prescribed direction while remaining chemically valid, the SELFIES variants of BioMatrix likewise achieve often higher validity than their SMILES counterparts—BioMatrix-4B (SELFIES) reaches 0.98–1.00 across all eight sub-tasks, whereas the corresponding SMILES variants range from 0.71 to 0.97—and simultaneously dominate the Correct metric under both loose and strict success criteria.

The validity advantage of SELFIES stems directly from a fundamental property of the formalism: every syntactically well-formed SELFIES string is guaranteed to decode into a chemically valid molecule,

regardless of the token sequence the model produces [56]. In principle, this guarantee should yield 100% validity. The residual gap observed in practice (up to ∼5% on some MolOpt sub-tasks, <0.5% on unconditional generation) arises not from SELFIES decoding failures but from edge cases at the generation level: the model occasionally emits tokens that fall outside the expected SELFIES pattern (e.g., producing text fragments, or malformed bracket structures that do not constitute a parseable SELFIES string), or the generated sequence is truncated by the maximum decoding length before a complete molecule is formed, leaving a partial string that cannot be decoded. The wider gap on MolOpt compared to unconditional generation is expected: optimization prompts require the model to jointly interpret a natural-language property instruction and produce a valid SELFIES output, increasing the probability of format-level errors relative to the simpler unconditional setting where the model only needs to continue in SELFIES mode. These are failures of the language model’s sequence generation rather than of the SELFIES formalism itself, and the consistently higher validity of SELFIES over SMILES across both benchmarks confirms that the grammatical guarantee provides a meaningful safety net even when the model’s generation is imperfect.

In both settings, the task is structurally well-matched to SELFIES’s design: the model is rewarded for producing any valid molecule satisfying a high-level objective, and SELFIES’s bracketed grammar guarantees the underlying syntactic legality without the model having to bookkeep ring closures, valence, or branch parentheses.

SMILES wins where surface-level structural anchoring matters. The contrast is sharpest on customized molecule generation (Section 6.1.7), where the SELFIES variants collapse to near-zero success rates across all three sub-tasks (e.g., FuncGroup SR drops from 0.2714 for BioMatrix-4B (SMILES) to

- 0.0048 for BioMatrix-4B (SELFIES)), and on forward synthesis and retrosynthesis (Section 6.1.5), where BioMatrix-4B (SMILES) reaches 77.94/45.16 EM versus 69.08/42.90 for the SELFIES counterpart. As discussed in Section 6.1.7, the failure modes here all stem from the same source: SELFIES’s contextdependent token semantics actively obscure the surface patterns that the task requires the model to anchor onto. Atom counts are confounded by branch/ring index tokens, bond-type counts are folded into neighboring atom tokens, and functional-group substructures lose any single canonical token signature. SMILES, with its one-character-per-heavy-atom and dedicated bond/ring symbols, exposes precisely the regularities these tasks reward.

Tasks where the choice is essentially neutral. On property prediction, molecule captioning, textbased generation, molecule editing, single-property optimization on S2-Bench, and MoleculeQA, the two notations track each other to within metric noise. These are tasks in which neither validity-byconstruction nor surface-level pattern matching is the bottleneck: the model is either reading molecular sequence content as input (with SMILES and SELFIES carrying equivalent information modulo the tokenizer) or generating short outputs whose correctness is dominated by semantic content rather than by the grammar of the notation.

###### 9.3 Where Scaling from 1.7B to 4B Helps, and Where It Does Not

Our two within-family checkpoints (BioMatrix-1.7B and BioMatrix-4B), trained under an otherwise identical recipe, allow a clean per-task readout of how much each capability benefits from a ∼2.4× increase in parameter count. The results are markedly heterogeneous, and we find it useful to group sub-tasks into three regimes.

Strongly scaling tasks. The largest and most consistent gains from 1.7B to 4B appear on tasks that combine non-trivial knowledge recall with structured generation. Molecular name conversion

- (Section 6.1.2) gains 14–15 absolute points on I2F EM (79.45 → 94.09 for SMILES) and 5–6 on I2S EM. Text-based molecule generation (Section 6.1.4) gains 9–10 absolute EM points (56.35 → 65.07 for SMILES), while customized generation shows a large relative gain despite remaining a difficult

###### Where Scaling from 1.7B to 4B Helps

Score uses only BioMatrix-4B vs. BioMatrix-1.7B; lower-is-better metrics are converted to relative error reduction.

+0.7%

1D Gen Name Property

+9.8%

+1.6%

+2.2%

Caption Text2Mol Synthesis

+15.5%

Molecule 1D avg +7.2%

+6.6%

+2.5%

Editing

+8.0%

Opt. Custom MolQA

+19.8%

+5.2%

-3.1%

3D Gen 3D Cond.

Molecule 3D

- avg +3.0%

Protein 1D avg +1.9%

Protein 3D avg +0.2%

Interaction

- avg +4.3%

+9.2%

-1.3%

Seq. Und. Annotation Mol-Inst. Know. Mine Text2Prot

+0.1%

+1.7%

+11.1%

-1.9%

+0.9%

Struct. QA Folding Inv. Fold

+0.0%

+0.4%

- -0.2%

+0.7%

- -0.3%

Co-gen. Backbone Seq. Gen.

+0.3%

PPI cls. Affinity 1D Binding RMSE PDBBind 3D

+3.9%

+9.9%

+3.0%

−5 0 5 10 15 20

Relative gain of BioMatrix-4B over BioMatrix-1.7B (%)

Positive gain Negative gain Near parity band

Figure 7: Relative gain from scaling BioMatrix from 1.7B to 4B. The score compares only the two BioMatrix scales: higher-is-better metrics use the relative score increase over BioMatrix-1.7B, while lower-is-better metrics use the corresponding relative error reduction.

low-success regime. Forward synthesis EM gains ∼5 points (72.75 → 77.94), optimization improves across the aggregate success metrics, and MoleculeQA gains ∼3.5 absolute points on Total accuracy. On the protein side, the clearest relative gain is knowledge mining, where the 4B model improves substantially over the 1.7B variant. We interpret these as tasks where the bottleneck is the model’s ability to compress a large, heterogeneous training distribution into reusable factual and structural knowledge—a regime in which additional parameters reliably help.

Weakly scaling or saturated tasks. A second cluster of tasks shows little to no improvement from scaling, typically because the 1.7B variant has already saturated the relevant metric or because the task is dominated by benchmark noise. Unconditional molecular generation is essentially saturated at

- 1.7B (validity, uniqueness, and novelty all sit at or near their ceilings on both MOSES and GuacaMol). Property prediction and molecule captioning show only modest relative gains, and the aggregate

annotation and structure-QA scores on the protein side are likewise close to parity between the two scales. Sequence understanding (EC number, fold type, subcellular localization) and inverse folding AAR are within 1 point between the two scales. These flat scaling trends should therefore not be read as poor task performance: in several cases BioMatrix is already at or near the best reported result, leaving little headroom for a larger checkpoint to show a measurable gain.

Anti-scaling and high-variance tasks. A small number of tasks exhibit non-monotone behavior between the two scales. The most visible case is regression on small datasets: ESOL RMSE rises from 0.810 (BioMatrix-1.7B SMILES) to 1.240 (BioMatrix-4B SMILES), and a similar but smaller fluctuation appears on Lipo. On EC number prediction (Price split) and a few interaction sub-tasks, the 1.7B variant slightly outperforms the 4B variant. Furthermore, interaction tasks should not be read as a strong scaling regime despite occasional positive deltas in Figure 7: they are mostly small-data, single-task SFT variants, where optimization noise and benchmark variance can be comparable to the apparent gain.

###### 9.4 Native Multimodality Pays Off Most on Cross-Modal and Cross-Entity Tasks

A subtler but, we believe, important pattern across the evaluation is where the unified-tokenization design pays off most relative to specialized baselines. On tasks that operate within a single modality and a single entity type—unconditional molecule generation, sequence-only protein understanding, classification on MoleculeNet—BioMatrix matches but does not dramatically exceed dedicated specialists; the unification does not hurt, but it also does not unlock new headroom. On tasks that explicitly cross modalities or entities, by contrast, the gains over the strongest specialist baselines widen substantially: property-conditioned molecular conformer generation, text-based molecule generation, sequence-based molecule–protein affinity prediction, and structure-grounded molecule–protein affinity prediction.

Taken together, the data directly validate the effectiveness of this design: the value of placing sequences, structures, and natural language for both molecules and proteins in a single discrete token space is not primarily that it improves any individual single-modality task, but that it makes the cross-modal and cross-entity tasks tractable under one next-token objective, without bolt-on encoders, projectors, or fusion modules.

###### 9.5 Tokenization Is the Bottleneck on Fine-Grained 3D Geometry

A final cross-cutting observation concerns the limits of the current tokenizers on geometry. On tasks evaluated at the level of distributions or backbone topology—unconditional molecular conformer generation on graph-aggregated metrics, unconditional protein backbone generation, sequence–structure co-generation, structure understanding—BioMatrix matches or surpasses dedicated diffusion specialists. On tasks evaluated at the level of fine-grained per-atom geometry, however, a residual gap remains: bond-length MMD on QM9-2014 is loose relative to NExT-Mol (Section 6.2.1), and folding RMSD trails ESMFold on the PDB date split (Section 7.2.2). Both gaps are concentrated in the geometric reconstruction step rather than in upstream modeling: a single MMFF refinement closes most of the QM9 gap (FCD: 1.04 → 0.23, atom stability 0.897 → 0.985), suggesting that the limitation lies in the irreducible quantization error of finite codebooks combined with autoregressive coordinate reconstruction, rather than in the language model’s ability to learn the geometric distribution.

Two implications follow. First, the unified-tokenization paradigm is not in tension with high-fidelity structural outputs—a cheap geometric post-processing step recovers most of the precision an end-toend diffusion specialist provides, and pairing a unified token-level generator with such a refinement step is a practical design pattern. Second, advances in structural tokenization—larger or hierarchical codebooks, joint tokenization of side chains, or non-autoregressive decoding schemes that reconstruct

coordinates globally rather than atom-by-atom—are likely to translate directly into improvements for any model built on top of them.

#### 10 Conclusion

We presented BioMatrix, a multimodal foundation model that natively integrates sequences, structures, and natural language for both molecules and proteins within a single decoder-only architecture. By consolidating all modalities into a shared discrete token space, BioMatrix handles heterogeneous biomolecular inputs and outputs uniformly under a single next-token prediction objective, without external encoders, projection adapters, or modality-specific output heads. Through continual pretraining on a large-scale multimodal corpus and instruction tuning across a comprehensive task suite, BioMatrix achieves state-of-the-art or competitive performance on the vast majority of evaluated benchmarks, demonstrating that a single generalist backbone can match or surpass specialized models across a broad spectrum of biological tasks. We hope BioMatrix serves as a useful step toward unified biological foundation models that bridge sequences, structures, and language across diverse biomolecular entities.

#### 11 Limitations

Despite the breadth and consistency of the empirical results, BioMatrix has several limitations that we believe are important to acknowledge, and that point to concrete directions for future work.

Molecular and protein 3D structures are tokenized in disjoint spaces. BioMatrix employs two structure tokenizers that operate independently of each other, with codebooks anchored in disjoint geometric reference frames. While this design works well within a single biomolecular entity, neither tokenizer is equipped to express the relative pose between a small molecule and a protein pocket. As a result, BioMatrix cannot natively represent or generate biomolecular complexes, which limits its applicability to tasks such as structure-based docking and pocket-conditioned ligand design. A unified 3D tokenizer that operates on a shared geometric reference frame across entity types is the next step.

Potential overlap between continual pretraining and SFT/evaluation data. We did not perform dedicated entity-level filtering between the continual pretraining corpus and the downstream SFT/evaluation corpora. Consequently, molecules, proteins, and their associated annotations appearing in SFT or evaluation tasks may have been implicitly observed during continual pretraining, either through related database entries, literature text, or cross-entity resources. This is a practical limitation of building broad biological foundation models from public molecular and protein corpora, where the same entities and metadata are repeatedly reused across databases and benchmarks. Our goal in this work is therefore not to claim a strictly contamination-free benchmark setting, but to build a strong and comprehensive base model that the community can further post-train on custom tasks with task-specific data curation.

General-purpose capabilities are partially eroded by domain specialization. Although we incorporate FineWeb-Edu and MegaScience into the pretraining and instruction-tuning corpora to mitigate catastrophic forgetting, the heavy concentration of biomolecular tokens in the 304.4B-token corpus inevitably shifts the model away from the general language and reasoning capabilities of the underlying Qwen3-Base. We have not systematically benchmarked BioMatrix on general-purpose instruction-following or reasoning suites, and we expect a measurable gap relative to the original Qwen3 checkpoints on such tasks.

Task-group SFT rather than a single unified SFT model. As detailed in Section 9.1, we train a small set of task-group-specific SFT variants rather than one monolithic instruction-tuned model, because in our setting the pattern-monolithic nature of biological SFT data causes large-data sub-tasks to overwhelm small-data ones when mixed naively. While we release a single all-tasks SFT model with mild oversampling for practical use, this unified variant still trails the task-group variants on a non-trivial fraction of small-data sub-tasks. Closing this gap (via more principled data balancing, curriculum design, or training dynamics tailored to highly imbalanced multi-task biological corpora) is an open problem that we have not fully solved within the recipe explored here.

Coverage of biomolecular entities and modalities. BioMatrix covers small molecules and proteins, but does not currently model other biologically important entities such as nucleic acids (DNA, RNA), carbohydrates, or lipids. Extending the unified tokenization scheme to these additional entities and modalities is conceptually compatible with the framework presented here, and we view it as a natural direction for scaling the paradigm.

#### References

- [1] Uniprot: the universal protein knowledgebase in 2023. Nucleic acids research, 51(D1):D523–D531, 2023.
- [2] A-Alpha Bio. Open-AlphaSeq: Open protein–protein interaction affinity datasets, 2025. URL https: //huggingface.co/datasets/aalphabio/open-alphaseq.
- [3] Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos, and Michalis Vazirgiannis. Prot2text: Multimodal protein’s function generation with gnns and transformers. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 10757–10765, 2024.
- [4] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024.
- [5] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [6] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.
- [7] Sarah Alamdari, Nitya Thakkar, Rianne van den Berg, Alex Lu, Nicolo Fusi, Ava Amini, and Kevin Yang. Protein generation with evolutionary diffusion: sequence is all you need. In NeurIPS 2023 Generative AI and Biology (GenBio) Workshop.
- [8] Anthropic. Claude 3.5 Sonnet model card addendum, 2024. URL https://www-cdn.anthropic.com/ fed9cc193a14b84131812372d8d5857f8f304c52/Model Card Claude 3 Addendum.pdf.

- [9] Anthropic. Claude Opus 4.6 system card. Technical report, Anthropic, February 2026. URL https: //www.anthropic.com/claude-opus-4-6-system-card.
- [10] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1(1):4, 2024.
- [11] Viraj Bagal, Rishal Aggarwal, P. K. Vinod, and U. Deva Priyakumar. Molgpt: Molecular generation using a transformer-decoder model. J. Chem. Inf. Model., 62(9):2064–2076, 2022.
- [12] Fan Bao, Min Zhao, Zhongkai Hao, Peiyao Li, Chongxuan Li, and Jun Zhu. Equivariant energy-guided SDE for inverse molecular design. In ICLR. OpenReview.net, 2023.
- [13] Helen M Berman, John Westbrook, Zukang Feng, Gary Gilliland, Talapady N Bhat, Helge Weissig, Ilya N Shindyalov, and Philip E Bourne. The protein data bank. Nucleic acids research, 28(1):235–242, 2000.

- [14] Damian Bertoni, Maxim Tsenkov, Paulyna Magana, Sreenath Nair, Ivanna Pidruchna, Marcelo Querino Lima Afonso, Adam Midlik, Urmila Paramval, Dare Lawal, Ahsan Tanweer, et al. Alphafold protein structure database 2025: a redesigned interface and updated structural coverage. Nucleic Acids Research, 54 (D1):D358–D362, 2026.
- [15] Avishek Joey Bose, Tara Akhound-Sadegh, Guillaume Huguet, Kilian Fatras, Jarrid Rector-Brooks, ChengHao Liu, Andrei Cristian Nica, Maksym Korablyov, Michael M. Bronstein, and Alexander Tong. Se(3)stochastic flow matching for protein backbone generation. In ICLR. OpenReview.net, 2024.
- [16] Nathan Brown, Marco Fiscato, Marwin H. S. Segler, and Alain C. Vaucher. Guacamol: Benchmarking models for de novo molecular design. J. Chem. Inf. Model., 59(3):1096–1108, 2019.
- [17] Anton Bushuiev, Roman Bushuiev, Petr Kouba, Anatolii Filkin, Marketa Gabrielova, Michal Gabriel, Jir´ı Sedl´ar, Tom´as Pluskal, Jir´ı Damborsk´y, Stanislav Mazurenko, and Josef Sivic. Learning to design proteinprotein interactions with enhanced generalization. In ICLR. OpenReview.net, 2024.
- [18] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi S. Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In ICML, Proceedings of Machine Learning Research, pages 5453–5512. PMLR / OpenReview.net, 2024.
- [19] He Cao, Yanjun Shao, Zhiyuan Liu, Zijing Liu, Xiangru Tang, Yuan Yao, and Yu Li. PRESTO: progressive pretraining enhances synthetic chemistry outcomes. In EMNLP (Findings), Findings of ACL, pages 10197–

10224. Association for Computational Linguistics, 2024.

- [20] Lifan Chen, Xiaoqin Tan, Dingyan Wang, Feisheng Zhong, Xiaohong Liu, Tianbiao Yang, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, Mingyue Zheng, and Arne Elofsson. Transformercpi: improving compoundprotein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinform., 36(16):4406–4414, 2020.
- [21] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [22] Fengyuan Dai, Shiyang You, Yudian Zhu, Yuan Gao, Lihao Fu, Xibin Zhou, Jin Su, Chentong Wang, Yuliang Fan, Xiaoxiao Ma, et al. Toward de novo protein design from natural language. BioRxiv, pages 2024–08, 2024.
- [23] Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J Ragotte, Lukas F Milles, Basile IM Wicky, Alexis Courbet, Rob J de Haas, Neville Bethel, et al. Robust deep learning–based protein sequence design using proteinmpnn. Science, 378(6615):49–56, 2022.
- [24] Carl Edwards, Tuan Manh Lai, Kevin Ros, Garrett Honke, Kyunghyun Cho, and Heng Ji. Translation between molecules and natural language. In EMNLP, pages 375–413. Association for Computational Linguistics, 2022.
- [25] Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, Ghalia Rehawi, Yu Wang, Llion Jones, Tom Gibbs, Tamas Feher, Christoph Angerer, Martin Steinegger, et al. Prottrans: toward understanding the language of life through self-supervised learning. IEEE transactions on pattern analysis and machine intelligence, 44(10): 7112–7127, 2021.
- [26] Chuanliu Fan, Zicheng Ma, Huanran Meng, Aijia Zhang, Wenjie Du, Jun Zhang, Yi Qin Gao, Ziqiang Cao, and Guohong Fu. Interleaved tool-call reasoning for protein function understanding, 2026. URL https://arxiv.org/abs/2601.03604.
- [27] Run-Ze Fan, Zengzhi Wang, and Pengfei Liu. Megascience: Pushing the frontiers of post-training datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025.
- [28] Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Huajun Chen. Mol-instructions: A large-scale biomolecular instruction dataset for large language models. In ICLR. OpenReview.net, 2024.
- [29] Yin Fang, Ningyu Zhang, Zhuo Chen, Lingbing Guo, Xiaohui Fan, and Huajun Chen. Domain-agnostic molecular generation with chemical feedback. In ICLR. OpenReview.net, 2024.

- [30] Zhi-Ping Feng and Chun-Ting Zhang. Prediction of membrane protein types based on the hydrophobic index of amino acids. Journal of protein chemistry, 19(4):269–275, 2000.
- [31] Paul G. Francoeur, Tomohide Masuda, Jocelyn Sunseri, Andrew Jia, Richard B. Iovanisci, Ian Snyder, and David Ryan Koes. Three-dimensional convolutional neural networks and a cross-docked data set for structure-based drug design. J. Chem. Inf. Model., 60(9):4200–4215, 2020.
- [32] Kaiyuan Gao, Yusong Wang, Haoxiang Guan, Zun Wang, Qizhi Pei, John Hopcroft, Kun He, and Lijun Wu. Tokenizing 3d molecule structure with quantized spherical coordinates. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 291–301, 2026.
- [33] Niklas W. A. Gebauer, Michael Gastegger, and Kristof Schutt.¨ Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. In NeurIPS, pages 7564–7576, 2019.
- [34] J Glaser. Binding affinity training data set, 2021. URL https://huggingface.co/datasets/jglaser/binding affinity.

- [35] Google DeepMind. Gemini 2.5: Our most intelligent AI model. Google DeepMind Blog, March 2025. URL https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/. Accessed: 2025-08-12.
- [36] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [37] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. In ICLR. OpenReview.net, 2023.
- [38] Gabriel Lima Guimaraes, Benjamin Sanchez-Lengeling, Carlos Outeiral, Pedro Luis Cunha Farias, and Al´an Aspuru-Guzik. Objective-reinforced generative adversarial networks (organ) for sequence generation models. arXiv preprint arXiv:1705.10843, 2017.
- [39] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [40] Yanzhi Guo, Lezheng Yu, Zhining Wen, and Menglong Li. Using support vector machine combined with auto covariance to predict protein–protein interactions from protein sequences. Nucleic acids research, 36(9): 3025–3030, 2008.
- [41] Jurgen¨ Haas, Alessandro Barbato, Dario Behringer, Gabriel Studer, Steven Roth, Martino Bertoni, Khaled Mostaguir, Rafal Gumienny, and Torsten Schwede. Continuous automated model evaluation (cameo) complementing the critical assessment of structure prediction in casp12. Proteins: Structure, Function, and Bioinformatics, 86:387–398, 2018.
- [42] Thomas Hayes, Roshan Rao, Halil Akin, Nicholas J Sofroniew, Deniz Oktay, Zeming Lin, Robert Verkuil, Vincent Q Tran, Jonathan Deaton, Marius Wiggert, et al. Simulating 500 million years of evolution with a language model. Science, 387(6736):850–858, 2025.
- [43] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR. OpenReview.net, 2021.
- [44] Emiel Hoogeboom, Victor Garcia Satorras, Cl´ement Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In ICML, Proceedings of Machine Learning Research, pages 8867–8887. PMLR, 2022.
- [45] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. OGB-LSC: A large-scale challenge for machine learning on graphs. In NeurIPS Datasets and Benchmarks, 2021.
- [46] Han Huang, Leilei Sun, Bowen Du, and Weifeng Lv. Conditional diffusion based on discrete graph structures for molecular graph generation. In AAAI, pages 4302–4311. AAAI Press, 2023.
- [47] Han Huang, Leilei Sun, Bowen Du, and Weifeng Lv. Learning joint 2-d and 3-d graph diffusion models for complete molecule generation. IEEE Trans. Neural Networks Learn. Syst., 35(9):11857–11871, 2024.

- [48] Lei Huang, Hengtong Zhang, Tingyang Xu, and Ka-Chun Wong. MDM: molecular diffusion model for 3d molecule generation. In AAAI, pages 5105–5112. AAAI Press, 2023.
- [49] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, An Yang, Rui Men, Fei Huang, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-coder technical report. CoRR, abs/2409.12186, 2024.
- [50] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [51] John B Ingraham, Max Baranov, Zak Costello, Karl W Barber, Wujie Wang, Ahmed Ismail, Vincent Frappier, Dana M Lord, Christopher Ng-Thow-Hing, Erik R Van Vlack, et al. Illuminating protein space with a programmable generative model. Nature, 623(7989):1070–1078, 2023.
- [52] Dejun Jiang, Chang-Yu Hsieh, Zhenxing Wu, Yu Kang, Jike Wang, Ercheng Wang, Ben Liao, Chao Shen, Lei Xu, Jian Wu, et al. Interactiongraphnet: A novel and efficient deep graph representation learning framework for accurate protein–ligand interaction predictions. Journal of medicinal chemistry, 64(24):18209–18232, 2021.
- [53] Wengong Jin, Regina Barzilay, and Tommi S. Jaakkola. Junction tree variational autoencoder for molecular graph generation. In ICML, Proceedings of Machine Learning Research, pages 2328–2337. PMLR, 2018.
- [54] Sunghwan Kim, Jie Chen, Tiejun Cheng, Asta Gindulyte, Jia He, Siqian He, Qingliang Li, Benjamin A Shoemaker, Paul A Thiessen, Bo Yu, et al. Pubchem 2025 update. Nucleic acids research, 53(D1):D1516–D1525, 2025.
- [55] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.
- [56] Mario Krenn, Florian H¨ase, AkshatKumar Nigam, Pascal Friederich, and Al´an Aspuru-Guzik. Selfreferencing embedded strings (SELFIES): A 100% robust molecular string representation. Mach. Learn. Sci. Technol., 1(4):45024, 2020.
- [57] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, pages 611–626. ACM, 2023.
- [58] Youngchun Kwon, Dongseon Lee, Youn-Suk Choi, Kyoham Shin, and Seokho Kang. Compressed graph representation for scalable molecular graph generation. J. Cheminformatics, 12(1):58, 2020.
- [59] Tuan Le, Julian Cremer, Frank No´e, Djork-Arn´e Clevert, and Kristof T. Schutt.¨ Navigating the design space of equivariant diffusion-based generative models for de novo 3d molecule generation. In ICLR. OpenReview.net, 2024.
- [60] Jiatong Li, Junxian Li, Weida Wang, Yunqing Liu, Changmeng Zheng, Dongzhan Zhou, Xiao-yong Wei, and Qing Li. Speak-to-structure: Evaluating llms in open-domain natural language-driven molecule generation. arXiv preprint arXiv:2412.14642, 2024.
- [61] Mingyang Li, Yurou Liu, Jieping Ye, Bing Su, Ji-Rong Wen, and Zheng Wang. Speaking the language of science: Toward a general-purpose generative foundation model for the natural sciences. arXiv preprint arXiv:2606.16905, 2026.
- [62] Shuya Li, Fangping Wan, Hantao Shu, Tao Jiang, Dan Zhao, and Jianyang Zeng. Monn: a multi-objective neural network for predicting compound-protein interactions and affinities. Cell systems, 10(4):308–322, 2020.
- [63] Sihang Li, Zhiyuan Liu, Yanchen Luo, Xiang Wang, Xiangnan He, Kenji Kawaguchi, Tat-Seng Chua, and Qi Tian. Towards 3d molecule-text interpretation in language models. In ICLR. OpenReview.net, 2024.
- [64] Zhirui Liao, Lei Xie, Hiroshi Mamitsuka, and Shanfeng Zhu. Sc2mol: a scaffold-based two-step molecule generator with variational autoencoder and transformer. Bioinform., 39(1), 2023.
- [65] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023.

- [66] Nuowei Liu, Jiahao Kuang, Yanting Liu, Tao Ji, Changzhi Sun, Man Lan, and Yuanbin Wu. Protein design with dynamic protein vocabulary. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.
- [67] Shengchao Liu, Yutao Zhu, Jiarui Lu, Zhao Xu, Weili Nie, Anthony Gitter, Chaowei Xiao, Jian Tang, Hongyu Guo, and Anima Anandkumar. A text-guided protein design framework. arXiv preprint arXiv:2302.04611, 2023.
- [68] Tiqing Liu, Yuhmei Lin, Xin Wen, Robert N. Jorissen, and Michael K. Gilson. Bindingdb: a web-accessible database of experimentally determined protein-ligand binding affinities. Nucleic Acids Res., 35(DatabaseIssue):198–201, 2007.
- [69] Tiqing Liu, Linda Hwang, Stephen K Burley, Carmen I Nitsche, Christopher Southan, W Patrick Walters, and Michael K Gilson. Bindingdb in 2024: a fair knowledgebase of protein-small molecule binding data. Nucleic acids research, 53(D1):D1633–D1644, 2025.
- [70] Zhihai Liu, Minyi Su, Li Han, Jie Liu, Qifan Yang, Yan Li, and Renxiao Wang. Forging the basis for developing protein–ligand interaction scoring functions. Accounts of chemical research, 50(2):302–309, 2017.
- [71] Zhiyuan Liu, Sihang Li, Yanchen Luo, Hao Fei, Yixin Cao, Kenji Kawaguchi, Xiang Wang, and Tat-Seng Chua. Molca: Molecular graph-language modeling with cross-modal projector and uni-modal adapter. In EMNLP, pages 15623–15638. Association for Computational Linguistics, 2023.
- [72] Zhiyuan Liu, An Zhang, Hao Fei, Enzhi Zhang, Xiang Wang, Kenji Kawaguchi, and Tat-Seng Chua. Prott3: Protein-to-text generation for text-based protein understanding. In ACL (1), pages 5949–5966. Association for Computational Linguistics, 2024.
- [73] Zhiyuan Liu, Yanchen Luo, Han Huang, Enzhi Zhang, Sihang Li, Junfeng Fang, Yaorui Shi, Xiang Wang, Kenji Kawaguchi, and Tat-Seng Chua. Next-mol: 3d diffusion meets 1d language modeling for 3d molecule generation. In ICLR. OpenReview.net, 2025.
- [74] Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel S. Weld. S2ORC: the semantic scholar open research corpus. In ACL, pages 4969–4983. Association for Computational Linguistics, 2020.
- [75] Daniel Lowe. Chemical reactions from us patents, 2017.
- [76] Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- [77] Wei Lu, Qifeng Wu, Jixian Zhang, Jiahua Rao, Chengtao Li, and Shuangjia Zheng. Tankbind: Trigonometryaware neural networks for drug-protein binding structure prediction. In NeurIPS, 2022.
- [78] Xingyu Lu, He Cao, Zijing Liu, Shengyuan Bai, Leqing Chen, Yuan Yao, Hai-Tao Zheng, and Yu Li. Moleculeqa: A dataset to evaluate factual accuracy in molecular comprehension. In EMNLP (Findings), Findings of ACL, pages 3769–3789. Association for Computational Linguistics, 2024.
- [79] Yizhen Luo, Jiahuan Zhang, Siqi Fan, Kai Yang, Yushuai Wu, Mu Qiao, and Zaiqing Nie. Biomedgpt: Open multimodal generative pre-trained transformer for biomedicine. arXiv preprint arXiv:2308.09442, 2023.
- [80] Youzhi Luo and Shuiwang Ji. An autoregressive flow model for 3d molecular geometry generation from scratch. In ICLR. OpenReview.net, 2022.
- [81] Ge M-A-P, Zhang, Xinrun Du, et al. Finefineweb: A comprehensive study on fine-grained domain web corpus, December 2024. URL [https://huggingface.co/datasets/m-a-p/FineFineWeb](https:// huggingface.co/datasets/m-a-p/FineFineWeb).
- [82] Zicheng Ma, Chuanliu Fan, Zhicong Wang, Zhenyu Chen, Xiaohan Lin, Yanheng Li, Shihao Feng, Ziqiang Cao, Jun Zhang, and Yi Qin Gao. Prottex: Structure-in-context reasoning and editing of proteins with large language models. J. Chem. Inf. Model., 65(13):6599–6612, 2025.
- [83] Nakata Maho. The pubchemqc project: A large chemical database from the first principle calculations. In AIP conference proceedings, volume 1702. AIP Publishing, 2015.
- [84] Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.

- [85] Roc´ıo Mercado, Tobias Rastemo, Edvard Lindel¨of, Gunter¨ Klambauer, Ola Engkvist, Hongming Chen, and Esben Jannik Bjerrum. Graph networks for molecular design. Mach. Learn. Sci. Technol., 2(2):25023, 2021.
- [86] Iain H. Moal and Juan Fern´andez-Recio. SKEMPI: a structural kinetic and energetic database of mutant protein interactions and its use in empirical models. Bioinform., 28(20):2600–2607, 2012.
- [87] Seokhyun Moon, Wonho Zhung, Soojung Yang, Jaechang Lim, and Woo Youn Kim. Pignet: a physicsinformed deep learning model toward generalized drug–target interaction predictions. Chemical Science, 13

(13):3661–3673, 2022.

- [88] Alex Morehead and Jianlin Cheng. Geometry-complete perceptron networks for 3d molecular graphs. Bioinform., 40(2), 2024.
- [89] Shrimon Mukherjee, Madhusudan Ghosh, and Partha Basuchowdhuri. Deepglstm: Deep graph convolutional network and lstm based approach for predicting drug-target binding affinity. In Proceedings of the 2022 SIAM International Conference on Data Mining (SDM), pages 729–737. SIAM, 2022.
- [90] Natalia Neverova, Christian Wolf, Graham W. Taylor, and Florian Nebout. Moddrop: Adaptive multi-modal gesture recognition. IEEE Trans. Pattern Anal. Mach. Intell., 38(8):1692–1706, 2016.
- [91] OpenAI. Introducing ChatGPT. https://openai.com/blog/chatgpt, November 2022.
- [92] OpenAI. New models and developer products announced at DevDay. https://openai.com/blog/ new-models-and-developer-products-announced-at-devday, November 2023.
- [93] OpenAI. Openai o3 and o4-mini system card. https://openai.com/index/o3-o4-mini-system-card/, April 2025.
- [94] Xiao-Yong Pan, Ya-Nan Zhang, and Hong-Bin Shen. Large-scale prediction of human protein- protein interactions from amino acid sequence based on latent topic features. Journal of proteome research, 9(10): 4992–5001, 2010.
- [95] Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, and Rui Yan. Biot5: Enriching cross-modal integration in biology with chemical knowledge and natural language associations. In EMNLP, pages 1102–1123. Association for Computational Linguistics, 2023.
- [96] Qizhi Pei, Lijun Wu, Kaiyuan Gao, Xiaozhuan Liang, Yin Fang, Jinhua Zhu, Shufang Xie, Tao Qin, and Rui Yan. Biot5+: Towards generalized biological understanding with IUPAC integration and multi-task tuning. In ACL (Findings), Findings of ACL, pages 1216–1240. Association for Computational Linguistics, 2024.
- [97] Qizhi Pei, Zhimeng Zhou, Kaiyuan Gao, Jinhua Zhu, Yue Wang, Zun Wang, Tao Qin, Lijun Wu, and Rui Yan. Leveraging biomolecule and natural language through multi-modal learning: A survey. arXiv preprint arXiv:2403.01528, 2024.
- [98] Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. Molecular sets (moses): a benchmarking platform for molecular generation models. Frontiers in pharmacology, 11: 565644, 2020.
- [99] Mahdi Pourmirzaei, Alex Morehead, Farzaneh Esmaili, Jarett Zida Ren, Mohammadreza Pourmirzaeioliaei, and Dong Xu. Gcp-vqvae: A geometry-complete language for protein 3d structure. In NeurIPS 2025 AI for Science Workshop.
- [100] Oleksii Prykhodko, Simon Johansson, Panagiotis-Christos Kotsias, Josep Arus-Pous,´ Esben Jannik Bjerrum, Ola Engkvist, and Hongming Chen. A de novo molecular generation method using latent vector based generative adversarial network. J. Cheminformatics, 11(1):74, 2019.
- [101] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014.
- [102] Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Xi Chen, John F. Canny, Pieter Abbeel, and Yun S. Song. Evaluating protein transfer learning with TAPE. In NeurIPS, pages 9686–9698, 2019.
- [103] Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function emerge from scaling

- unsupervised learning to 250 million protein sequences. Proc. Natl. Acad. Sci. USA, 118(15):e2016239118, 2021.
- [104] Vijayakumar Saravanan and Namasivayam Gautham. Harnessing computational biology for exact linear b-cell epitope prediction: a novel amino acid composition-based feature descriptor. Omics: a journal of integrative biology, 19(10):648–658, 2015.
- [105] Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In ICML, Proceedings of Machine Learning Research, pages 9323–9332. PMLR, 2021.
- [106] Nadine Schneider, Nikolaus Stiefl, and Gregory A. Landrum. What’s what: The (nearly) definitive guide to reaction role assignment. J. Chem. Inf. Model., 56(12):2336–2346, 2016.
- [107] Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 4(1):120–131, 2018.
- [108] Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 4(1):120–131, 2018.
- [109] Richard Sever, Samantha Hindle, Ted Roeder, Sol Fereres, Olaya Fern´andez Gayol, Sanchari Ghosh, Martina Proietti Onori, Emma Croushore, Kevin-John Black, Linda Sussman, et al. biorxiv: the preprint server for biology. BioRxiv, page 833400, 2019.
- [110] Amir Shanehsazzadeh, David Belanger, and David Dohan. Is transfer learning necessary for protein landscape prediction? arXiv preprint arXiv:2011.03443, 2020.
- [111] Vignesh Ram Somnath, Charlotte Bunne, and Andreas Krause. Multi-scale representation learning on proteins. In NeurIPS, pages 25244–25255, 2021.
- [112] Teague Sterling and John J. Irwin. ZINC 15 - ligand discovery for everyone. J. Chem. Inf. Model., 55(11): 2324–2337, 2015.
- [113] Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, and Ji-Rong Wen. A molecular multimodal foundation model associating molecule graphs with natural language. arXiv preprint arXiv:2209.05481, 2022.
- [114] Jin Su, Chenchen Han, Yuyang Zhou, Junjie Shan, Xibin Zhou, and Fajie Yuan. Saprot: Protein language modeling with structure-aware vocabulary. In ICLR. OpenReview.net, 2024.
- [115] Jin Su, Xibin Zhou, Xuting Zhang, and Fajie Yuan. A trimodal protein language model enables advanced protein searches. Nature Biotechnology, 2025. doi: 10.1038/s41587-025-02836-0. URL https://doi.org/10. 1038/s41587-025-02836-0.
- [116] Minyi Su, Qifan Yang, Yu Du, Guoqin Feng, Zhihai Liu, Yan Li, and Renxiao Wang. Comparative assessment of scoring functions: The CASF-2016 update. J. Chem. Inf. Model., 59(2):895–913, 2019.
- [117] Mujeen Sung, Minbyul Jeong, Yonghwa Choi, Donghyeon Kim, Jinhyuk Lee, and Jaewoo Kang. BERN2: an advanced neural biomedical named entity recognition and normalization tool. Bioinform., 38(20):4837–4839, 2022.
- [118] Baris E Suzek, Hongzhan Huang, Peter McGarvey, Raja Mazumder, and Cathy H Wu. Uniref: comprehensive and non-redundant uniprot reference clusters. Bioinformatics, 23(10):1282–1288, 2007.
- [119] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.
- [120] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [121] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [122] Kimi Team. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534, 2025.

- [123] A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NIPS, pages 6306–6315, 2017.
- [124] Michel Van Kempen, Stephanie S Kim, Charlotte Tumescheit, Milot Mirdita, Jeongjae Lee, Cameron LM Gilchrist, Johannes S¨oding, and Martin Steinegger. Fast and accurate protein structure search with foldseek. Nature biotechnology, 42(2):243–246, 2024.
- [125] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. In The Eleventh International Conference on Learning Representations.
- [126] Cl´ement Vignac, Nagham Osman, Laura Toni, and Pascal Frossard. Midi: Mixed graph and 3d denoising diffusion for molecule generation. In ECML/PKDD (2), Lecture Notes in Computer Science, pages 560–576. Springer, 2023.
- [127] Huibin Wang, Zehui Wang, Minghua Shi, Zixian Cheng, and Ying Qian. Enhancing unconditional molecule generation via online knowledge distillation of scaffolds. Molecules, 30(6):1262, 2025.
- [128] Penglei Wang, Shuangjia Zheng, Yize Jiang, Chengtao Li, Junhong Liu, Chang Wen, Atanas Patronov, Dahong Qian, Hongming Chen, and Yuedong Yang. Structure-aware multimodal deep learning for drugprotein interaction prediction. J. Chem. Inf. Model., 62(5):1308–1317, 2022.
- [129] Renxiao Wang, Xueliang Fang, Yipin Lu, and Shaomeng Wang. The pdbbind database: collection of binding affinities for protein- ligand complexes with known three-dimensional structures. Journal of medicinal chemistry, 47(12):2977–2980, 2004.
- [130] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion language models are versatile protein learners. In ICML, Proceedings of Machine Learning Research, pages 52309–52333. PMLR / OpenReview.net, 2024.
- [131] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. DPLM-2: A multimodal diffusion protein language model. In ICLR. OpenReview.net, 2025.
- [132] Yizhou Wang, Chen Tang, Han Deng, Jiabei Xiao, Jiaqi Liu, Jianyu Wu, Jun Yao, Pengze Li, Encheng Su, Lintao Wang, et al. Scireasoner: Laying the scientific reasoning ground across disciplines. arXiv preprint arXiv:2509.21320, 2025.
- [133] Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with rfdiffusion. Nature, 620(7976):1089–1100, 2023.
- [134] Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. In ICLR. OpenReview.net, 2020.
- [135] Jacob White. Pubmed 2.0. Medical reference services quarterly, 39(4):382–387, 2020.
- [136] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9

(2):513–530, 2018.

- [137] Yingce Xia, Peiran Jin, Shufang Xie, Liang He, Chuan Cao, Renqian Luo, Guoqing Liu, Yue Wang, Zequn Liu, Yuan-Jyue Chen, et al. Nature language model: deciphering the language of nature for scientific discovery. arXiv preprint arXiv:2502.07527, 2025.
- [138] Hongwang Xiao, Wenjun Lin, Hui Wang, Zheng Liu, and Qiwei Ye. Opi: An open instruction dataset for adapting large language models to protein-related tasks. In Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges, 2024.
- [139] Jinrui Xu and Yang Zhang. How significant is a protein structure similarity with tm-score = 0.5? Bioinform., 26(7):889–895, 2010.
- [140] Minghao Xu, Zuobai Zhang, Jiarui Lu, Zhaocheng Zhu, Yangtian Zhang, Chang Ma, Runcheng Liu, and Jian Tang. PEER: A comprehensive and multi-task benchmark for protein sequence understanding. In NeurIPS, 2022.

- [141] Minkai Xu, Alexander S. Powers, Ron O. Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. In ICML, Proceedings of Machine Learning Research, pages 38592–38610. PMLR, 2023.
- [142] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [143] Zaifei Yang, Hong Chang, Ruibing Hou, Shiguang Shan, and Xilin Chen. Knowmol: Advancing molecular large language models with multi-level chemical knowledge. arXiv preprint arXiv:2510.19484, 2025.
- [144] Geyan Ye, Xibao Cai, Houtim Lai, Xing Wang, Junhong Huang, Longyue Wang, Wei Liu, and Xiangxiang Zeng. Drugassist: a large language model for molecule optimization. Briefings Bioinform., 26(1), 2025.
- [145] Jason Yim, Brian L. Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay, and Tommi S. Jaakkola. SE(3) diffusion model with application to protein backbone generation. In ICML, Proceedings of Machine Learning Research, pages 40001–40039. PMLR, 2023.
- [146] Botao Yu, Frazier N Baker, Ziqi Chen, Xia Ning, and Huan Sun. Llasmol: Advancing large language models for chemistry with a large-scale, comprehensive, high-quality instruction tuning dataset. Conference on Language Modeling, 2024.
- [147] Chaohao Yuan, Songyou Li, Geyan Ye, Yikun Zhang, Long-Kai Huang, Wenbing Huang, Wei Liu, Jianhua Yao, and Yu Rong. Annotation-guided protein design with multi-level domain alignment. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 1855–1866, 2025.
- [148] Barbara Zdrazil, Eloy Felix, Fiona Hunter, Emma J Manners, James Blackshaw, Sybilla Corbett, Marleen De Veij, Harris Ioannidis, David Mendez Lopez, Juan F Mosquera, et al. The chembl database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic acids research, 52(D1):D1180–D1192, 2024.
- [149] Odin Zhang, Xujun Zhang, Haitao Lin, Cheng Tan, Qinghan Wang, Yuanle Mo, Qiantai Feng, Gang Du, Yuntao Yu, Zichang Jin, et al. Odesign: A world model for biomolecular interaction design. arXiv preprint arXiv:2510.22304, 2025.
- [150] Qiang Zhang, Keyan Ding, Tianwen Lv, Xinda Wang, Qingyu Yin, Yiwen Zhang, Jing Yu, Yuhao Wang, Xiaotong Li, Zhuoyi Xiang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Mengyao Zhang, Jinlu Zhang, Jiyu Cui, Renjun Xu, Hongyang Chen, Xiaohui Fan, Huabin Xing, and Huajun Chen. Scientific large language models: A survey on biological & chemical domains. ACM Comput. Surv., 57(6):161:1–161:38, 2025.
- [151] Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. A comprehensive survey of scientific large language models and their applications in scientific discovery. In EMNLP, pages 8783–8817. Association for Computational Linguistics, 2024.
- [152] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In ACL (3), pages 400–410. Association for Computational Linguistics, 2024.
- [153] Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, and Guolin Ke. Uni-mol: A universal 3d molecular representation learning framework. In ICLR. OpenReview.net, 2023.
- [154] Yihan Zhu, Gang Liu, Eric Inae, and Meng Jiang. Moltextnet: A two-million molecule-text dataset for multimodal molecular learning. arXiv preprint arXiv:2506.00009, 2025.
- [155] Yicheng Zou, Dongsheng Zhu, Lin Zhu, Tong Zhu, Yunhua Zhou, Peiheng Zhou, Xinyu Zhou, Dongzhan Zhou, Zhiwang Zhou, Yuhao Zhou, et al. Intern-s1-pro: Scientific multimodal foundation model at trillion scale. arXiv preprint arXiv:2603.25040, 2026.
- [156] Chenyang Zuo, Siqi Fan, and Zaiqing Nie. Biomedgpt-mol: Multi-task learning for molecular understanding and generation. arXiv preprint arXiv:2512.04629, 2025.

## Appendix

Table 30: Instruction-tuning grouping of BioMatrix.

Task family Variant Tasks covered Rationale for grouping

Both benchmarks probe the same underlying capability—learning the grammar of 1D molecular sequences and exploring chemical space with high validity and diversity—but each defines its own train/test split and computes novelty against its own training set. Following the official protocol of each benchmark, we train one variant per benchmark on its respective training set.

Unconditional generation (MOSES, GuacaMol; trained separately, one variant per benchmark)

Vgenmol-1D

1D molecule

Name conversion, property prediction, captioning, text-based generation, forward synthesis, retrosynthesis, editing, optimization, customized generation, and question answering

Heterogeneous supervised tasks with abundant data; co-training improves cross-task generalization.

Vmixmol-1D

Both subtasks rely on the same underlying capability of generating chemically valid 2D graphs together with physically plausible 3D conformers; propertyconditioned generation simply adds a quantum-property prefix to the same generation process. Co-training the two as a single mixture lets the model amortize the shared geometric learning signal while still learning to ground generation on the conditioning property.

Unconditional and propertyconditioned 3D structure generation

3D molecule Vmol-3D

The understanding-oriented tasks share the same amino acid sequence input format and are co-trained as a single mixture. They are separated from the generation-oriented design tasks because mixing understanding and generation objectives degrades overall performance.

Sequence understanding, annotation prediction, knowledge mining

Vundprot-1D

1D protein

Within the design group, CAMEO and Molinst-SwissProtCLAP are trained separately because their sequence distributions differ substantially, and mixing the two corpora degrades performance on both benchmarks.

Text-based protein design (one variant per ProDVa benchmark: CAMEO, MolinstSwissProtCLAP)

Vdesprot-1D

Each 1D/3D interaction subtask trained as a separate variant (1D: PDBBind/CASF-2016, BindingDB, Yeast-PPI, Human-PPI, PPI-Affinity; 3D: PDBBindv2020)

Each subtask has a small training set with high evaluation variance; mixing across subtasks degrades performance noticeably.

Interaction Vinter∗

- 63

Table 31: Detailed training and test data statistics for each sub-task.

Domain Modality Task Name #Train #Test Source

MOSES 1,936,962 10,000 OMG-GPT [127] GuacaMol 1,260,508 10,000 OMG-GPT [127]

Unconditional Sequence Generation

IUPAC-to-formula (I2F) 300000 2993 SMolInstruct [146] IUPAC-to-sequence (I2S) 274053 2764 SMolInstruct [146] sequence-to-formula (S2F) 274053 2764 SMolInstruct [146] sequence-to-IUPAC (S2I) 274053 2764 SMolInstruct [146]

Name Conversion

ESOL 888 112 SMolInstruct [146] Lipo 3359 420 SMolInstruct [146] BBBP 1489 185 SMolInstruct [146] ClinTox 1131 143 SMolInstruct [146] HIV 30370 3813 SMolInstruct [146] SIDER 20240 2340 SMolInstruct [146]

Property Prediction

SMolInstruct 55611 2327 SMolInstruct [146] CheBI-20 26407 3300 MolT5 [24] KnowMol-100K 399996 – KnowMol-100K [143]

Molecule Captioning

SMolInstruct 55611 2282 SMolInstruct [146] CheBI-20 26407 3300 MolT5 [24] KnowMol-100K 99999 – KnowMol-100K [143]

Text-Based Molecule Generation

1D

Molecule

AtomNum 133334 5000 S2-Bench [60] BondNum 133334 5000 S2-Bench [60] FuncGroup 133334 5000 S2-Bench [60]

Customized Molecule Generation

Forward Synthesis SMolInstruct 971809 4062 SMolInstruct [146] Retrosynthesis

SMolInstruct 941735 4156 SMolInstruct [146] USPTO-50K 40008 5007 USPTO-50K [106]

AddComp 133333 5000 S2-Bench [60] DelComp 133333 5000 S2-Bench [60] SubComp 133333 5000 S2-Bench [60]

Molecule Editing

qed+ 145991 8122 MolOpt-Instructions [144] Continued on next page

BioMatrix

Molecule Optimization

- 64

Table 31 – Continued from previous page

###### Domain Modality Task Name #Train #Test Source

acceptor+ 154492 8583 MolOpt-Instructions [144] donor+ 154492 8583 MolOpt-Instructions [144] solubility+ 154493 8583 MolOpt-Instructions [144] bbbp+ 154493 8583 MolOpt-Instructions [144] herg- 154493 8582 MolOpt-Instructions [144] others 263501 – MolOpt-Instructions [144] logP 133333 5000 S2-Bench [60] MR 133333 5000 S2-Bench [60] QED 133333 5000 S2-Bench [60]

Structure 32176 3113 MoleculeQA [78] Source 11062 1343 MoleculeQA [78] Property 4838 731 MoleculeQA [78] Usage 1917 599 MoleculeQA [78]

Molecule Question Answering

Unconditional Structure Generation QM9-2014 97,834 10,000 NExT-Mol [73]

QM9-2014-µ 48,849 10,000 NExT-Mol [73] QM9-2014-α 48,849 10,000 NExT-Mol [73] QM9-2014-Cv 48,849 10,000 NExT-Mol [73] QM9-2014-εHOMO 48,849 10,000 NExT-Mol [73] QM9-2014-εLUMO 48,849 10,000 NExT-Mol [73] QM9-2014-∆ε 48,849 10,000 NExT-Mol [73]

3D

Conditional Structure Generation

CLEAN EC number new

392 OPI [138] CLEAN EC number price 149 OPI [138] Subcellular localization 11,230 2772 OPI [138] Fold holdout

227,362

Sequence Understanding

718 OPI [138] Superfamily holdout 1,254 OPI [138] Family holdout 1,272 OPI [138]

12,311

CASPSimilarSeq function

184 OPI [138] IDFilterSeq function 1,112 OPI [138] UniProtSeq function 4,562 OPI [138]

451,618

CASPSimilarSeq go

184 OPI [138] Continued on next page

451,618

1D

Annotation Prediction

Protein

BioMatrix

- 65

Table 31 – Continued from previous page

###### Domain Modality Task Name #Train #Test Source

IDFilterSeq go 1,112 OPI [138] UniProtSeq go 4,562 OPI [138]

CASPSimilarSeq keywords

184 OPI [138] IDFilterSeq keywords 1,112 OPI [138] UniProtSeq keywords 4,562 OPI [138]

451,618

Catalytic Activity 51,573 1,601 Mol-Instructions [28] Domain Motif 43,700 5,876 Mol-Instructions [28] Protein Function 110,689 3,494 Mol-Instructions [28] General Function 83,939 2,633 Mol-Instructions [28]

Gene name to cancer 590 148 OPI [138] Gene symbol to cancer 590 148 OPI [138] Gene symbol to tissue 8,723 2,181 OPI [138]

Knowledge Mining

CAMEO 391,933 507 ProDVa [66] Molinst-SwissProtCLAP 712,248 5,876 ProDVa [66]

Text-Based Protein Design

Unconditional Sequence Generation PDB, AFDB 220,285 500 DPLM-2 [131]

Structure Understanding PFUD 391,771 7,702 ProtTex [82] Structure Prediction (Folding) PDB, AFDB 220,285 447 DPLM-2 [131] Inverse Folding PDB, AFDB 220,285 447 DPLM-2 [131] Sequence–Structure Co-Generation PDB, AFDB 220,285 500 DPLM-2 [131] Unconditional Structure Generation PDB, AFDB 220,285 500 DPLM-2 [131]

3D

BindingDB 7,900 5,230 PEER [140] PDBBindv2019 16,434 285 PEER [140]

Molecule–Protein Interaction

1D

Yeast 4,945 394 PEER [140] Human 35,669 237 PEER [140] PPI 2,421 326 PEER [140]

Interaction

Protein–Protein Interaction

3D Molecule–Protein Interaction PDBBindv2020 18,013 359 PDBBindv2020 [129] Text – – MegaScience 1,253,230 – MegaScience [27]

BioMatrix

