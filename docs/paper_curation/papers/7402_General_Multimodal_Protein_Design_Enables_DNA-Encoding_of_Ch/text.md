## arXiv:2604.05181v1[cs.LG]6Apr2026

# General Multimodal Protein Design Enables DNA-Encoding of Chemistry

Jarrid Rector-Brooks1,2,3†*, Théophile Lambert1,4†, Marta Skreta2,3†, Daniel Roth1† Yueming Long1, Zi-Qi Li1, Xi Zhang2,5, Miruna Cretu6, Francesca-Zhoufan Li1 Tanvi Ganapathy1, Emily Jin7, Avishek Joey Bose2,8, Jason Yang1 Kirill Neklyudov2,3,9, Yoshua Bengio2,3,10, Alexander Tong11 Frances H. Arnold1*, Cheng-Hao Liu1,12†*

1California Institute of Technology 2Mila – Québec AI Institute 3Université de Montréal 4Université Paris-Saclay 5McGill University 6University of Cambridge 7University of Oxford 8Imperial College London 9Institut Courtois 10LawZero 11AITHYRA 12FutureHouse

*Correspondence: chl@caltech.edu, jarrid.rector-brooks@mila.quebec, frances@cheme.caltech.edu †Equal contribution

Abstract Evolution is an extraordinary engine for enzymatic diversity, yet the chemistry it has explored remains a narrow slice of what DNA can encode. Deep generative models can design new proteins that bind ligands, but none have created enzymes without pre-specifying catalytic residues. We introduce DISCO, a multimodal model that co-designs protein sequence and 3D structure around arbitrary biomolecules, as well as inference-time scaling methods that optimize objectives across both modalities. Conditioned solely on reactive intermediates, DISCO designs diverse heme enzymes with novel active-site geometries. These enzymes catalyze newto-nature carbene-transfer reactions, including alkene cyclopropanation, spirocyclopropanation, B–H, and C(sp3)–H insertions, with high activities exceeding those of engineered enzymes. Random mutagenesis of a selected design further confirmed that enzyme activity can be improved through directed evolution. By providing a scalable route to evolvable enzymes, DISCO broadens the potential scope of genetically encodable transformations.

Code & Models | https://github.com/DISCO-design/DISCO

1

###### Introduction

Enzymes catalyze chemical reactions under mild conditions with high efficiency and specificity, underpinning applications in chemical manufacturing, drug synthesis, and emerging therapeutic modalities. [1, 2] Enzyme engineering has typically relied on iterative rounds of mutation and screening to accumulate beneficial sequence changes toward desired functions – an evolutionary process that can enable enzymes to perform functions not previously known in nature. [3] However, every such campaign requires an initial protein with measurable activity. For well-studied transformations, suitable starting points can be found among natural proteins or their close variants; but for many desirable new-to-nature chemistries, nature offers no obvious candidates. Identifying one remains largely a matter of chemical intuition – a laborious process fundamentally limited by what evolution has already sampled. How can we systematically design functional enzymes for chemical transformations that have no precedent in known biology?

Generative deep learning models have transformed protein design, successfully enabling the creation of novel binders and the scaffolding of pre-defined structural motifs. [4–6] However, de novo design of enzymes for new-to-nature reactions remains largely unrealized. Current computational pipelines face two fundamental limitations. First, they rely heavily on pre-specified, fixed geometric arrangements of active-site residues or theozymes. [7–9] This precludes reactions for which no motif or precise mechanism is known and prevents the discovery of novel active-site geometries. Second, experimentally validated generative pipelines predominantly operate sequentially: they first generate a backbone and then devise a sequence to match the backbone design (inverse folding). [7, 10, 11] Because sequence and structure jointly determine function, this decoupled paradigm cannot leverage sequence-based objectives during critical stages of backbone formation, and vice versa.

We introduce DISCO (DIffusion for Sequence-structure CO-design), a general multimodal framework that simultaneously designs protein sequences and 3D structures de novo. DISCO designs can be conditioned on and co-folded with arbitrary biomolecules, without pre-defined residue motifs. A novel multimodal Feynman-Kac corrector method allows DISCO to optimize sequence and structure objectives together. DISCO achieves state-of-the-art in silico results in generating binders for diverse biomolecular targets with fine-grained property control. Applied to new-to-nature catalysis experimentally, DISCO generates diverse, evolvable carbene transferases with novel active sites. No inverse-folding is required. These enzymes have activities that surpass typical starting points for directed evolution and, in selected cases, extensively evolved variants. Ultimately, DISCO enables the discovery of biocatalysts and protein motifs previously unknown to biology, expanding the searchable space of DNA-encoded chemical reactivity.

###### Multimodal protein generation with DISCO

DISCO models discrete sequences and 3D structures as a joint distribution that is denoised through a unified generative process (Fig. 1). We achieve this by training a single deep neural network that employs a masked discrete diffusion process for sequences and, in parallel, leverages continuous diffusion for 3D atomic coordinates. By independently sampling noise per modality during training, one can provably learn the joint reverse process using only unimodal diffusion losses [12]: the standard denoising score matching loss for coordinates and the masked diffusion language model loss for sequences. We trained DISCO using only data from the Protein

#### 2 DISCO

###### A

3 Applications

Inputs

1

[Figure 1]

New-to-nature enzyme catalysis

Amino acid sequence(s)

Inference steering

Multimodal diffusion

MV··

[Figure 2]

[Figure 3]

Condition tokens

Molecular SMILES

c1··

·· m m m m ·· ·· m E m A ·· ·· S E D K ··

[Figure 4]

[Figure 5]

DNA/RNA sequence(s)

[Figure 6]

[Figure 7]

Molecule binder

DNA binder RNA binder

Bonds & geometries

[Figure 8]

Specific binder

###### B

Cross-modal recycling

[Figure 9]

+ MTKAF·· ·· m m m ··

Joint multimodal diffusion

Input embedder

Pairformer

[Figure 10]

Pair

pLM

Input conditions

+

Single

Inputs

[Figure 11]

D

C

[Figure 12]

[Figure 13]

Steering

✓ Refolding

✓ Confidence

DISCO

✓ Burial

[Figure 14]

…

✓ Contacts

Rxn. intermediate geometry & bonds

m m m

[Figure 15]

✓ Charges

✓ …

Masked sequence

✓ Newresidue motifs Wet-lab expression, screening, evolution

…

x1 xt x0

Figure 1. Multimodal protein design workflow with DISCO. (A) Inference overview, highlighting the arbitrary molecular context, sequence-structure co-generation with sequence correction, and representative applications. (B) Architecture overview. (C) Multimodal inference-time steering schematics with Feynman-Kac corrector. (D) Design-to-test workflow for generating enzymes with new active sites from reactive intermediates.

Data Bank without special filtering, avoiding any selection biases introduced when models are trained exclusively on “designable” structures that folding algorithms can successfully predict.

DISCO’s architecture comprises an input embedder that produces single and pair representations of all input modalities, a frozen protein language model (pLM) that encodes partial sequences, [13] a Pairformer stack that contextualizes these representations, and a joint diffusion module built on atom-level attention that predicts both sequence logits and denoised coordinates of the backbone (Fig. 1B). [14] We inject SE(3) symmetries softly through data augmentation rather than architectural constraints.

Central to our approach is a cross-modal recycling mechanism that conditions each generation step on four distinct encodings: the model’s current predicted clean sequence xˆ0seq and structure xˆ0struct, as well as the current noised sequence xτseq and structure xτstruct. We train a structure encoder [15] and employ the pLM to inject sequence information. This bidirectional conditioning ensures that sequence predictions are informed by emerging structural features while structural predictions adapt to evolving sequence identity.

DISCO’s inference strategy proved critical to generation quality (Fig. 2B, Fig. S4). Standard masked diffusion—iteratively unmasking tokens without revision [16]—yields sequences that fail to fold into their generated geometries. By enabling self-correction during sequence inference [17, 18] and introducing a novel sequence temperature mechanism that smooths the amino acid distribution for overconfident tokens early in the trajectory, we drastically improve co-designability. Combining this with noisy guidance [12] allows DISCO to achieve state-of-the-art performance in sequence-structure co-design. On unconditional monomer design, approximately 90% of generated sequences refold to within 2 Å RMSD of their designed backbone structures using ESMFold (Fig. 2A, Fig. S5, Table S4). [19] Compared to baselines, DISCO achieves the highest sequence and structure diversity and novelty without sacrificing co-designability (Fig. 2A).

###### Conditioning on arbitrary biomolecular contexts

DISCO flexibly conditions on arbitrary biomolecular contexts, including small molecules, metallocofactors, reactive intermediates, and nucleic acids. Non-protein molecules are represented identically to proteins – with discrete tokens for chemical identity and corresponding atomic coordinates. The model simultaneously unmasks protein sequences while denoising the coordinates of targets ranging from small molecules and metal cofactors to reactive intermediates and nucleic acids. These contexts co-fold with the designed protein throughout the generative trajectory: DISCO adjusts the coordinates of conditioning biomolecules while designing the protein, capturing conformational changes induced by molecular interactions and extending generation beyond fixed, pre-defined atomic scaffolds.

To benchmark this capability, we designed a new benchmark, STUDIO-179, a library of 179 natural and non-natural ligands spanning applications in catalysis, pharmaceuticals, luminescence, and sensing. The library encompasses rigid molecules (e.g., the pollutant tetrachlorodibenzodioxin), large or flexible molecules (e.g., the cofactor CoQ10), and metals/metalloclusters (e.g., [4Fe-4S]). For each ligand, we quantify the fraction of generated designs that are both structurally diverse and co-designable, defining co-designable as when the protein backbone and all ligand centroids have an RMSD < 2 Å upon refolding with Chai-1. [20] Compared to baselines, DISCO generates the highest proportion of diverse, co-designable complexes for 178

[Figure 16]

[Figure 17]

###### A B

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>C|Sapropterin CoQ10<br><br>[Figure 26]<br><br>Heme B + PLP<br><br>TCDD Erythromycin<br><br>[Figure 27]<br><br>[Figure 28]<br><br>DNA: 7RCC RNA: 6YMC<br><br>Rocuronium SAM<br><br>[Figure 29]<br><br>[Figure 30]<br><br>C60×3<br><br>D|
|---|---|
|[Figure 31]<br><br>[Figure 32]<br><br>6 S-S bonds<br><br>5 S-S bonds<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>E<br><br>| |
|[Figure 36]<br><br>[Figure 37]<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
<br><br>Aldosterone and not cortisone<br><br>[Figure 38]<br><br>F<br><br>Valine and not proline| |

[Figure 39]

[Figure 40]

[Figure 41]

Figure 2. DISCO generates diverse, designable protein sequences and structures across a wide variety of functions evaluated in silico. (A) In unconditional monomer generation, DISCO outperforms existing methods in terms of co-designability, novelty, and diversity (400 samples). (B) Ablation of unconditional monomer generation with DISCO to identify best inference-time practices. (C) DISCO outperforms existing methods for conditional generation across diverse targets, including natural and non-natural molecules, multi-ligands, DNA, and RNA. (D) Examples of designed (blue) and refolded (white) protein complexes show high agreement. (E) FKC-MM enables the joint steering of sequence and structure, as demonstrated by two multimodal rewards: disulfide bridges and cation-π interactions. Example generated proteins with high disulfide bridge counts are shown at right. (F) FKC-SG generates more proteins with predicted binding specificity (a large separation between binding sites of on- and off-targets) compared to existing filtering methods. Example refolded structures are shown at right (on-target in pink, off-target in yellow).

of 179 evaluated ligands (Figs. 2C, 2D, Figs. S10, S11, S25). Our formulation naturally supports multi-context conditioning—such as designing proteins to interface with multiple distinct ligands simultaneously—again yielding high-quality complexes (Fig. 2D).

Extending beyond small molecules, we evaluated DISCO’s performance on macromolecular interfaces. DISCO successfully generated co-designable proteins predicted to bind to both sequence-specific DNA and RNA sequences, outperforming existing models (Fig. 2C). Note that because the frozen pLM in DISCO was not trained on multimeric complexes, we constrained our current evaluation to non-protein-binding tasks.

###### Multimodal inference steering using Feynman-Kac Correctors

The multimodality of DISCO enables inference-time scaling with both sequence and structure signals (Fig. 1C). Rather than relying on inefficient generate-and-filter (best-of-N) approaches, which often fail when desired traits are rare, we utilize the Feynman-Kac Corrector (FKC) framework to incorporate target properties into the generation process directly and tilt sample distributions toward having desirable traits without necessarily invoking expensive reward oracles. [21, 22] We derive two novel FKC approaches.

First, FKC - Multimodal (FKC-MM) allows for reward tilting with functions defined jointly over both discrete sequences and continuous structures. When using multimodal reward functions for increasing disulfide bond content or cation-π interactions, FKC-MM generates samples with enriched target properties compared to structure-only FKC or unconditional generations (Fig. 2E). Notably, while unconditional generation underrepresents cysteines (Fig. S6), the top 2% of designable, 100-amino acid proteins produced by FKC-MM contain six disulfide bonds – a density matched by only the top 0.2% of comparable training proteins (Fig. S9). These results underscore the benefit of being able to generate sequences alongside protein structures.

We further introduce FKC - Specificity Guidance (FKC-SG) to steer DISCO towards binding exclusively to a target molecule while avoiding a structurally similar decoy. This is achieved by sampling from a tilted distribution that encourages samples that are likely under the on-target model while penalizing samples that are likely under the off-target model (Section A.7). We evaluated FKC-SG on discriminating two dissimilar molecules (e.g., biotin from pyridoxal 5’-phosphate (PLP)), as well as very similar molecules, such as aldosterone from cortisone (two constitutional isomeric steroids) and valine from proline (two hydrophobic, aliphatic amino acids). We find that FKC-SG generates proteins that have high separation between on- and off-target complexes (Fig. 2F), improving upon filters like best-of-N, which in some cases do not produce any hits passing in silico filters.

###### DISCO designs exhibit realistic protein features with novel, complementary motifs

DISCO’s designs capture the complex statistical properties of proteins. In unconditional generation, samples display natural amino-acid compositions (Fig. 3A, Fig. S6), diverse secondary structures (Fig. 3B, Fig. S7), favorable local Ramachandran geometries (Fig. 3C), reasonable radii of gyration (Fig. S8), and native-like surface properties including appropriate hydrophobicity and net charge (Fig. 3D). The designs further exhibit high long-range contact order (Fig. 3E), indicating complex, well-connected topologies rather than trivial folds often seen in de novo designs. [23] These properties extend to conditional generation (Figs. S12–S17).

[Figure 42]

[Figure 43]

[Figure 44]

B C

[Figure 45]

A

[Figure 46]

[Figure 47]

E

D

F G H

[Figure 48]

[Figure 49]

Novel binding site example

[Figure 50]

139K

191K

106E

[Figure 51]

41E

125S

213S

42S 45S

110S 113S

RMSD5-residue = 5.04 Å to Q76IH4

M

[Figure 52]

- I
- J

- K
- L

[Figure 53]

[Figure 54]

[Figure 55]

Reference conformers

Cofactor F420 Carbene

Generated conformers

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Figure 3. DISCO designs biophysically realistic proteins with novel motifs, flexible ligands, and target-informed sequences. (A-E) Unconditional generation results: (A) Amino acid type distribution as compared to PDB and inverse-folded samples. (B) Secondary structure diversity. (C) Ramachandran plot. (D) Surface hydrophobicity and net charge. (E) Distribution of long-range contact order. (F-M) Conditional generation results: (F) Folddisco motif similarity searches reveal no close matches to designed residue motifs in AlphaFoldDB. (G) Example active-site (purple) compared to the closest known motif (yellow). (H) Distribution of global sequence/structure novelty. (I) DISCO designs binding sites with consideration of ligand properties, as illustrated by matching lipophilicity. (J) Example design satisfying ligand constraints, building four disulfide bonds to a [4Fe-4S] cluster. (K) DISCO-generated conformers are valid as evaluated by PoseBusters, even if they deviate from the reference. (L) DISCO samples different conformers. (M) Examples of sampled conformers that are rare or absent in the training set.

We hypothesized that jointly generating sequence and structure allows the designed binding site to adapt its chemistry and geometry as the target adjusts its conformation, yielding a mutually compatible interface. This is a capability unique to co-design, allowing DISCO to construct chemically appropriate microenvironments without inverse folding or any pre-specified residue motif. Indeed, ligand-conditioned generation yields pockets that are quantitatively responsive to ligand identity: binding-site lipophilicity correlates with ligand hydrophobicity (Fig. 3I, Figs. S20, S21), appropriate coordinating residues emerge for specific cofactors (Fig. 3J, Fig. S27), and sufficiently large cavities form to avoid steric clashes (Fig. S19). Prior co-design models capture these physiochemical trends less consistently (Fig. S22), which aligns with their lower co-designability and diversity. Furthermore, DISCO samples valid ligand conformers beyond the reference input (Figs. 3K–M, Figs. S17, S26), by selectively exploring rotatable bonds while preserving rigid geometries.

These complementary pockets generated by DISCO exhibit striking novelty and diversity. For coordination spheres up to ten residues, motif similarity searches using Folddisco [24] show that most binding motifs in co-designable generations lack natural homologs in AlphaFoldDB (novel motif defined as no match or RMSD > 3 Å; Figs. 3F, 3G, Fig. S28). Clustering the generated motifs via pairwise distance shows over 90% cluster diversity (Figs. S23, S24). DISCO also generates globally novel sequences, and many folds have no close homologs in the Protein Data Bank (Fig. 3H). Such novelty and diversity are only relevant when the motifs are foldable and physicochemically realistic – standards DISCO meet while expanding well beyond the space of known global scaffolds and local motifs.

###### Designing enzymes for new-to-nature biocatalysis

Enzyme design is among the most challenging applications for conditional protein generation, especially for new-to-nature catalytic activities that lie outside the model’s training distribution. Carbene transfer reactions, unexplored in known natural enzymes, have proven valuable for expanding the catalytic repertoire of biology. [25] While directed evolution of cytochrome P450, cytochrome c, and various globin variants has produced numerous carbene transferases, identifying initial activity for a desired reaction remains challenging and laborious, usually limited to screening a limited set of known scaffolds. Previous computational approaches have designed heme enzymes for only a handful of carbene reactions using physics-based theozymes and pre-specified active-site residue geometries [26–28]—a strategy that is expensive, biased by the choice of motif, and difficult to generalize, especially when the catalytic mechanism is not known. We hypothesized that DISCO’s ability to co-generate sequence and structure around arbitrary molecules could bypass residue motif or theozyme specification.

Carbene-transfer reactions proceed via formation of an iron–carbenoid intermediate, most often generated by nitrogen extrusion from a diazo precursor. This intermediate subsequently transfers the carbene fragment to a substrate through diverse pathways, including alkene cyclopropanation, or insertion into C–H, B–H, Si–H, or N–H bonds. [25, 29] Formation of the heme–carbene intermediate is generally the rate-determining step (>20 kcal/mol, Fig. S3). [30] Therefore, we conditioned DISCO on DFT-derived geometries and bonding patterns of the hemecarbene precursor complex rather than on a complete transition-state model. This choice reflects a deliberate simplification: rather than fixing an exact transition state calculated in vacuo, we let DISCO’s co-folding mechanism sample conformations of the reactive intermediate that would

be compatible with the protein it is constructing. This allows us to bypass the theozyme-based design entirely, where the limited mechanistic understanding for new-to-nature reactions may preclude the design of an accurate theozyme. DISCO is free to explore catalytic solutions without being constrained by human assumptions about the protein residues required for activity or a fixed transition state geometry.

From ∼ 104 in silico generated sequence-structure pairs – one to two orders of magnitude fewer than recent structure-based pipelines [7, 10] – we applied a computational filter (Fig. 1D) combining predicted structures (AlphaFold 3 [14] and Chai-1 [20]), confidence measures such as chain pAE and ipTM, active-site contact counts, solvent exposure, net charge, and exposed hydrophobicity, yielding 90 designs for experimental testing (Section A.10). Neither the sequences nor the structures were redesigned after generation. Computational analysis confirms that the designed active sites represent new residue geometries (Fig. 3F, >80% with >3 Å RMSD in the five nearest residues). Moreover, while some designs closely recapitulate the reference geometry (RMSD < 1 Å), a substantial fraction adopt more divergent conformations (RMSD > 2 Å), primarily through rotations about the iron–carbenoid bond (Figs. 3L, 3M). This indicates that DISCO actively explores conformational space even for reaction intermediates. The designed enzymes are highly diverse: no pair shares more than 50% sequence identity, and the 90 designs form 75 distinct structural clusters (TM-score < 0.5).

To assess the catalytic generality, we selected four substrates spanning distinct carbene transfer reactions: p-methoxystyrene (1a) with ethyl diazoacetate (EDA, 5a) for cyclopropanation, [31] 1,3-dimethylimidazol-2-ylidene borane (2a) with ethyl 2-diazopropanoate (EDP, 5b) for B–H insertion, [32] 1-phenylpyrrolidine (3a) with 5a for C(sp3)–H insertion, [33] and tert-butyl-3methyleneazetidine-1-carboxylate (4a) with 5a for spirocyclopropanation of a pharmaceutically relevant scaffold. [34]

All 90 designs were screened in whole-cell format in Escherichia coli across all four reactions. To control for background reactivity from endogenous hemoproteins, we overexpressed an unrelated protein (TrpB from Thermotoga maritima) under identical conditions. Top hits were selected for validation (Fig. 4). Several reactions exhibited high background activity from endogenous E. coli hemoproteins, which complicated identification of lower-activity designs, especially since overexpressing these designs may disrupt heme homeostasis and further reduce their activity. We note that hemin alone has no detectable activity for the spirocyclopropanation or the C(sp3)–H insertion reaction.

Remarkably, screening across our selected carbene reactions confirmed that the designed variants were not only functional (2.2 – 66% exceeded the TmTrpB control) but, in several cases, exhibited high catalytic activity. For the cyclopropanation of p-methoxystyrene, the best design achieved 72% yield and a total turnover number (TTN) of 4,050 with a 99:1 diastereomeric ratio (Fig. S32), surpassing both early evolved P411 enzymes and the recently reported designed enzyme PNC2 which scaffolded a helix bundle around a porphyrin-based theozyme (Fig. 4A). [27, 31] For B–H insertion, the top variant achieved 98% yield and 5,170 TTN (Fig. 4B), far exceeding both the prior starting point (120 TTN) and the laboratory-evolved variant (2490 TTN). [32] Most notably, for C(sp3)–H insertion – a highly challenging transformation whose previous engineering campaign required 14 rounds of directed evolution and for which mechanistic uncertainty precludes theozyme construction – our best design reached 42% yield and 2,360 TTN for the alkylation of 1-phenylpyrrolidine, rivaling the performance of previously evolved P411-CHF catalysts on this substrate (2,030 TTN) (Fig. 4C). [33] The sterically and

[Figure 65]

[Figure 66]

[Figure 67]

- A

Alkene cyclopropanation

B-H insertion

C(sp3)-H alkylation

Spirocyclopropanation

- B
- C
- D

[Figure 68]

- 1a

5a

1b

- 2a 2b
- 3a 3b
- 4a 4b

Hemin: 96 TTN Evolved P450H2-5-F10: 364 TTN Theozyme-based design PNC2: 630 TTN

[Figure 69]

[Figure 70]

[Figure 71]

5b

[Figure 72]

Hemin: 80 TTN Rma cyt c: 120 TTN Evolved Rma cyt c: 2490 TTN

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Hemin: ND Evolved P411-CHF: 2030 TTN

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Hemin: ND Evolved ApePgb: 2200 TTN

- Figure 4. DISCO-designed carbene transferases (dCTs) catalyze multiple carbene transfer reactions with high activity at room temperature, including (A) p-methoxystyrene cyclopropanation. (B) B–H bond insertion of carbenes. (C) C(sp3)–H alkylation. (D) Spirocyclopropanation of an unsaturated exocyclic N-heterocycle. Each panel shows a reaction scheme, results for hemin/previous evolved variants, plate-based screening data, and yield and TTN results from selected validations. ND: Not Detectable.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

### A 77R B

36Y

33Y 58R

[Figure 85]

[Figure 86]

116L

91H

78H 105L

[Figure 87]

68L

78L

[Figure 88]

RMSD5-residue = 7.40 Å

[Figure 89]

to Q2JM39

[Figure 90]

dCT-H11

[Figure 91]

[Figure 92]

PDB: 3CRJ TM-score: 0.81 Seq. id.: 0.21

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

C

dCT-F9 dCT-G9 dCT-H10

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

PDB: 7C7E TM-score: 0.51 Seq. id.: 0.09

PDB: 5MSV TM-score: 0.52 Seq. id.: 0.05

PDB: 4RFS TM-score: 0.50 Seq. id.: 0.05

[Figure 101]

No full match for 5 closest residues

[Figure 102]

- Figure 5. Novelty of selected, designed enzymes and evaluation of evolvability. (A) dCT-H11 design (green/pink) structure and the AlphaFold 3/Chai predicted structures (grey), alongside the closest PDB match by TM-score. The inset shows the five-residue active-site arrangement predicted by AlphaFold 3 (purple), compared to the closest motif in AlphaFoldDB (yellow). (B) Mutational landscape of error-prone PCR variants in plate-based screening of dCT-H11 for the spirocyclopropanation reaction. (C) Designed and predicted structures of dCT-F9, dCT-G9, dCTH10, alongside the closest PDB matches by TM-score. The five closest active-site residues yielded no full match in AlphaFoldDB.

electronically demanding spirocyclopropanation proved more challenging, [34] yielding fewer active variants with modest activity. Although enantioselectivity was not a design criterion, modest enantioselectivities were observed across all reactions, with enantiomeric excesses up to 35% for the spirocyclopropanation (Figs. S29–S31). Notably, enzymes with a preference for either enantiomer were identified for three reactions.

To probe the evolvability of DISCO’s designs, we selected the spirocyclopropanation reaction, where design alone had yielded only modest activity. We subjected the designed carbene transferase dCT-H11 to a single round of error-prone PCR and screened approximately 700 mutants (Fig. 5B). Around 35 variants displayed noticeably improved activity relative to the parent enzyme with divergent enantioselectivity, either enhancing the parent’s preference (from +35% to +49% ee) or inverting it entirely (+35% to -35% ee, Fig. S33). The top variants harbor substitutions scattered across the protein, consistent with the long-range epistatic effects characteristic of natural enzyme evolution. These results suggest that DISCO’s designs can occupy regions of sequence space with accessible uphill paths for further optimization.

The top hits span diverse folds with novel motifs (Fig. 5). The closest structural match to

dCT-H11 is a TetR-family transcription factor from the extremophile Haloarcula marismortui (PDB ID: 3CRJ, TM-score: 0.81) that lacks known catalytic activity. dCT-H11 shares only 21% sequence identity with 3CRJ, suggesting that DISCO repurposed a topology not pre-specified for catalysis. The most similar residue motif to the five closest active-site residues, drawn from AlphaFoldDB, shows a large mismatch (RMSD > 7 Å), confirming that the geometry of the catalytic residues is itself novel. Other designs are even more divergent: dCT-F9 (closest match: 5MSV, TM-score 0.52, 5% sequence identity) and dCT-G9 (closest match: 7C7E, TMscore 0.51, 9% sequence identity) adopt folds that are substantially more remote from known structures, with no corresponding motif identified in AlphaFoldDB based on their five nearest active-site residues (Fig. 5C). Notably, none of the closest structural matches correspond to naturally heme-binding proteins, highlighting the ability of the model to capture underlying biochemical principles and repurpose unrelated protein folds for heme binding and carbene transfer reactions. Together, these results demonstrate that the model can create novel proteins with emergent functions, extending beyond the boundaries of natural protein space.

###### General biomolecular design unlocks new-to-nature reactivities

These results establish a general platform for functional protein design in which sequence and 3D structure are generated as a coupled object and flexibly conditioned on any biomolecules. Computationally, careful inference alignment across modalities via cross-modal recycling, self-correction, and entropy regularization enables DISCO to generate diverse, high-quality proteins that outperform existing co-generation methods, while inference-time steering on both sequence and structure turns generative sampling into a controllable search process. Experimentally, conditioning only on reaction intermediates was sufficient to produce promiscuous carbene transferases across four new-to-nature reactions, bypassing transition-state calculations, theozyme scaffolding, and extensive wet-lab screening. From just 90 genes, top designs achieved 98% yield for B–H insertion and >2,300 TTN for C(sp3)–H insertion, exceeding previously evolved biocatalysts. Importantly, the designed active-site residues are novel, exhibit stereoselectivity, and occupy accessible fitness landscapes that support subsequent evolution.

The ability to genetically encode arbitrary chemical transformations requires global discovery with local optimization. DISCO provides a global search for diverse, evolvable enzymes. Yet the reactions explored here represent only a small subset of synthetically valuable transformations not found in nature. Mastering increasingly complex mechanisms likely requires closing the loop between generative discovery and directed evolution, informed by richer biophysical constraints. More broadly, co-designing proteins with arbitrary biomolecular partners charts a path toward programmable biological systems whose complexity rivals that of natural evolution, and whose function serves human intent.

###### References

- [1] R. Buller, et al., From nature to industry: Harnessing enzymes for biocatalysis. Science 382 (6673), eadh8615 (2023).
- [2] J. C. Reisenbauer, K. M. Sicinski, F. H. Arnold, Catalyzing the future: recent advances in chemical synthesis using enzymes. Curr. Opin. Chem. Biol. 83, 102536 (2024).
- [3] K. Chen, F. H. Arnold, Engineering new catalytic activities in enzymes. Nat. Catal. 3 (3), 203–213 (2020).
- [4] J. L. Watson, et al., De novo design of protein structure and function with RFdiffusion. Nature 620 (7976), 1089–1100 (2023).
- [5] J. Wang, et al., Scaffolding protein functional sites using deep learning. Science 377 (6604), 387–394 (2022).
- [6] M. Pacesa, et al., One-shot design of functional protein binders with BindCraft. Nature 646 (8084), 483–492 (2025).
- [7] W. Ahern, et al., Atom-level enzyme active site scaffolding using RFdiffusion2. Nat. Methods 23 (1), 96–105 (2026).
- [8] A. Lauko, et al., Computational design of serine hydrolases. Science 388 (6744), eadu2454

(2025).

- [9] A. H.-W. Yeh, et al., De novo design of luciferases using deep learning. Nature 614 (7949), 774–780 (2023).
- [10] J. Butcher, et al., De novo design of all-atom biomolecular interactions with rfdiffusion3. bioRxiv (2025).
- [11] H. Stark, et al., BoltzGen: Toward Universal Binder Design. bioRxiv (2025).
- [12] K. Rojas, Y. Zhu, S. Zhu, F. X. F. Ye, M. Tao, Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces, in International Conference on Machine Learning (2025).
- [13] X. Wang, et al., Diffusion language models are versatile protein learners, in International Conference on Machine Learning (2024).
- [14] J. Abramson, et al., Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630 (8016), 493–500 (2024).
- [15] J. Dauparas, et al., Atomic context-conditioned protein sequence design using LigandMPNN. Nat. Methods 22 (4), 717–723 (2025).
- [16] S. S. Sahoo, et al., Simple and Effective Masked Diffusion Language Models, in Advances in Neural Information Processing Systems 37 (2024).
- [17] F. Z. Peng, et al., Path planning for masked diffusion model sampling. arXiv preprint arXiv:2502.03540 (2025).

- [18] G. Wang, Y. Schiff, S. S. Sahoo, V. Kuleshov, Remasking Discrete Diffusion Models with Inference-Time Scaling, in Advances in Neural Information Processing Systems 38 (2025).
- [19] Z. Lin, et al., Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379 (6637), 1123–1130 (2023).
- [20] C. D. team, et al., Chai-1: Decoding the molecular interactions of life. BioRxiv pp. 2024–10

(2024).

- [21] M. Skreta, et al., Feynman-Kac correctors in diffusion: Annealing, guidance, and product of experts, in International Conference of Machine Learning (2025).
- [22] M. Hasan, et al., Discrete feynman-kac correctors. arXiv preprint arXiv:2601.10403 (2026).
- [23] D. Listov, C. A. Goverde, B. E. Correia, S. J. Fleishman, Opportunities and challenges in design and optimization of protein function. Nat. Rev. Mol. Cell Biol. 25 (8), 639–653 (2024).
- [24] H. Kim, R. S. Kim, M. Mirdita, J. Yoon, M. Steinegger, Structural motif search across the protein-universe with Folddisco. bioRxiv pp. 2025–07 (2025).
- [25] Y. Yang, F. H. Arnold, Navigating the Unnatural Reaction Space: Directed Evolution of Heme Proteins for Selective Carbene and Nitrene Transfer. Acc. Chem. Res. 54 (5), 1209–1225

(2021).

- [26] I. Kalvet, et al., Design of Heme Enzymes with a Tunable Substrate Binding Pocket Adjacent to an Open Metal Coordination Site. J. Am. Chem. Soc. 145 (26), 14307–14315 (2023).
- [27] K. Hou, et al., De novo design of porphyrin-containing proteins as efficient and stereoselective catalysts. Science 388 (6747), 665–670 (2025).
- [28] W. Huang, et al., De Novo Design, Directed Evolution and Computational Study of HemeBinding Helical Bundle Protein Catalysts for Biocatalytic Enantioselective Ge–H Insertion. J. Am. Chem. Soc. 147 (44), 40869–40878 (2025).
- [29] C. Damiano, P. Sonzini, E. Gallo, Iron catalysts with N-ligands for carbene transfer of diazo reagents. Chem. Soc. Rev. 49 (14), 4867–4905 (2020).
- [30] M. Garcia-Borràs, et al., Origin and Control of Chemoselectivity in Cytochrome c Catalyzed Carbene Transfer into Si–H and N–H bonds. J. Am. Chem. Soc. 143 (18), 7114–7123 (2021).
- [31] P. S. Coelho, E. M. Brustad, A. Kannan, F. H. Arnold, Olefin Cyclopropanation via Carbene Transfer Catalyzed by Engineered Cytochrome P450 Enzymes. Science 339 (6117), 307–310

(2013).

- [32] S. B. J. Kan, X. Huang, Y. Gumulya, K. Chen, F. H. Arnold, Genetically programmed chiral organoborane synthesis. Nature 552 (7683), 132–136 (2017).
- [33] R. K. Zhang, et al., Enzymatic assembly of carbon–carbon bonds via iron-catalysed sp 3 C–H functionalization. Nature 565 (7737), 67–72 (2019).
- [34] J. L. Kennemur, Y. Long, C. J. Ko, A. Das, F. H. Arnold, Enzymatic stereodivergent synthesis of azaspiro [2. y] alkanes. J. Am. Chem. Soc. 147 (31), 27165–27171 (2025).

- [35] J. Ho, A. Jain, P. Abbeel, Denoising Diffusion Probabilistic Models, in Advances in Neural Information Processing Systems 33 (2020).
- [36] D. T. Gillespie, A general method for numerically simulating the stochastic time evolution of coupled chemical reactions. Journal of Computational Physics 22 (4), 403–434 (1976).
- [37] M. Ren, T. Zhu, H. Zhang, Carbonnovo: Joint design of protein structure and sequence using a unified energy-based model, in International Conference on Machine Learning (2024).
- [38] J. Shi, K. Han, Z. Wang, A. Doucet, M. Titsias, Simplified and generalized masked diffusion for discrete data, in Advances in Neural Information Processing Systems 37 (2024).
- [39] J. Jumper, et al., Highly accurate protein structure prediction with AlphaFold. Nature 596 (7873), 583–589 (2021).
- [40] S. Nie, et al., Large Language Diffusion Models, in Advances in Neural Information Processing Systems 38 (2025).
- [41] T. Karras, et al., Guiding a diffusion model with a bad version of itself, in Advances in Neural Information Processing Systems 37 (2024).
- [42] H. Chang, H. Zhang, L. Jiang, C. Liu, W. T. Freeman, Maskgit: Masked generative image transformer, in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 11315–11325.
- [43] S. Lin, B. Liu, J. Li, X. Yang, Common diffusion noise schedules and sample steps are flawed, in Proceedings of the IEEE/CVF winter conference on applications of computer vision

(2024), pp. 5404–5411.

- [44] J. Dauparas, et al., Robust deep learning–based protein sequence design using ProteinMPNN. Science 378 (6615), 49–56 (2022).
- [45] M. Steinegger, J. Söding, MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat. Biotechnol. 35 (11), 1026–1028 (2017).
- [46] M. Van Kempen, et al., Fast and accurate protein structure search with Foldseek. Nat. Biotechnol. 42 (2), 243–246 (2024).
- [47] C. Bannwarth, S. Ehlert, S. Grimme, GFN2-xTB—An accurate and broadly parametrized self-consistent tight-binding quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. J. Chem. Theory Comp. 15 (3), 1652–1671 (2019).
- [48] T. Geffner, et al., La-proteina: Atomistic protein generation via partially latent flow matching. arXiv preprint arXiv:2507.09466 (2025).
- [49] S. L. Lisanza, et al., Multistate and functional protein design using RoseTTAFold sequence space diffusion. Nat. Biotechnol. 43 (8), 1288–1298 (2025).
- [50] T. Hayes, et al., Simulating 500 million years of evolution with a language model. Science 387 (6736), 850–858 (2025).

- [51] R. Krishna, et al., Generalized biomolecular modeling and design with RoseTTAFold All-Atom. Science 384 (6693), eadl2528 (2024).
- [52] W. Kabsch, C. Sander, Dictionary of protein secondary structure: Pattern recognition of hydrogen-bonded and geometrical features. Biopolymers 22 (12), 2577–2637 (1983).
- [53] F. Neese, The ORCA program system. WIRES Comput. Molec. Sci. 2 (1), 73–78 (2012).
- [54] T. H. Allen, T. Kawamoto, S. Gardner, S. J. Geib, D. P. Curran, N-heterocyclic carbene boryl iodides catalyze insertion reactions of N-heterocyclic carbene boranes and diazoesters. Org. Lett. 19 (13), 3680–3683 (2017).
- [55] M. P. Doyle, D. Van Leusen, W. H. Tamblyn, Efficient alternative catalysts and methods for the synthesis of cyclopropanes from olefins and diazo compounds. Synthesis 1981 (10), 787–789 (1981).
- [56] D. G. Gibson, et al., Enzymatic assembly of DNA molecules up to several hundred kilobases. Nat. Methods 6 (5), 343–345 (2009).
- [57] Y. Long, et al., LevSeq: Rapid generation of sequence-function data for directed evolution and machine learning. ACS Synth. Biol. 14 (1), 230–238 (2024).
- [58] C. E. Boville, et al., Engineered Biosynthesis of β-Alkyl Tryptophan Analogues. Angew. Chem. Int. Ed. 57 (45), 14764–14768 (2018).

###### Acknowledgments

The authors thank Sabine Brinkmann-Chen, Olexa Bilaniuk, Almer van der Sloot, Luca Ambrogioni, Jessica Wu, Dejan Stancˇevic´, Danyal Rehman, Guillaume Huguet, Ella Miray Rajaonson, Fred Zhangzhi Peng, Andrei Rekesh, Ariane Mora, Julia Reisenbauer for very fruitful discussions and Ziyan Zhang for help with compound synthesis.

Funding. The authors acknowledge funding from Amgen, AMD, Wendy and Eric Schmidt Foundation, Amazon Web Services, the Natural Sciences and Engineering Research Council of Canada, National Research Council Canada, Microsoft, IVADO, and Canada First Research Excellence Fund. We thank Twist Bioscience for support with gene fragment synthesis. The work is enabled by high-performance computing resources at Mila, AMD, and the Digital Research Alliance of Canada. J.R.B. is supported by the St-Pierre-Larochelle Scholarship in AI for the Environment. D.R. thanks the Alexander von Humboldt Foundation for a Feodor Lynen Postdoctoral Fellowship. M.S. is supported by the IVADO 2025 Postdoctoral Research Funding Program.

Author contributions. All authors participated in the review and editing of the paper.

- • Conceptualization: CHL, JRB, AT, FHA
- • Model development: JRB, CHL, AT, MS, MC, NZ, AJB
- • Inference steering: MS, JRB, KN, CHL
- • Benchmarks, ablations, and baselines: JRB, CHL, MC, JY, EJ
- • Design filtering: CHL, FZL
- • Experimental characterization: TL, DR, YL, ZL, TG
- • Funding acquisition: FHA, CHL, YB
- • Supervision: CHL, FHA, AT, YB, KN
- • Writing – original draft: CHL, JRB, TL, MS

Competing interests. There are no competing interests to declare.

Data and materials availability. Code and model checkpoints associated with this paper are available at https://github.com/DISCO-design/DISCO. Analyzed results, including protein sequence, structure, and functional outcomes, will be available in the supplementary information or in an online database. Lyophilized cell powders are available upon request under a materials transfer agreement.

###### Supplementary Information

General Multimodal Protein Design Enables DNA-Encoding of Chemistry

###### Contents

###### A In silico methods and results 20

- A.1 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.2 Theoretical foundations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- A.2.1 Continuous-space diffusion modeling . . . . . . . . . . . . . . . . . . . . . 20
- A.2.2 Discrete-Space diffusion modeling . . . . . . . . . . . . . . . . . . . . . . . 21
- A.2.3 Multi-modal diffusion modeling . . . . . . . . . . . . . . . . . . . . . . . . 22

- A.3 Data pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- A.4 Model architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- A.4.1 Denoiser architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- A.4.2 Cross-modal encoding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.4.3 Diffusion module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.4.4 Diffusion conditioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.4.5 Atom attention decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- A.5 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- A.5.1 Structure diffusion losses . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.5.2 Sequence diffusion losses . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.5.3 Full training loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- A.5.4 Training hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- A.5.5 Model selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- A.6 Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- A.6.1 Inference sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- A.6.2 Sequence self-correction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- A.6.3 Entropy adaptive sequence temperature . . . . . . . . . . . . . . . . . . . . 34
- A.6.4 Inference schedule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.6.5 Conditional generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- A.6.6 Noisy guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- A.6.7 Inference hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- A.7 Inference steering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- A.7.1 Feynman-Kac Correctors - Specificity Guidance . . . . . . . . . . . . . . . 41
- A.7.2 FKC-SG experimental details . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- A.7.3 FKC-SG evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- A.7.4 Feynman-Kac Correctors - Multimodal . . . . . . . . . . . . . . . . . . . . 43
- A.7.5 FKC-MM reward functions . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

- A.8 In-silico evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

- A.8.1 Unconditional protein generation . . . . . . . . . . . . . . . . . . . . . . . 48
- A.8.2 Sequence diversity and novelty . . . . . . . . . . . . . . . . . . . . . . . . . 48

- A.8.3 Structural diversity and novelty . . . . . . . . . . . . . . . . . . . . . . . . 48
- A.8.4 Conditional evaluation of small-molecule ligands . . . . . . . . . . . . . . 49
- A.8.5 Conditional evaluation: nucleic acid binding . . . . . . . . . . . . . . . . . 49
- A.8.6 Baseline methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- A.8.7 Binding motif novelty . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- A.8.8 Binding motif diversity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

- A.9 Density Functional Theory calculations . . . . . . . . . . . . . . . . . . . . . . . . 51
- A.10 Design filtering for enzymes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- A.11 Additional in-silico results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54

- A.11.1 Additional unconditional generation results . . . . . . . . . . . . . . . . . 54
- A.11.2 Additional conditional generation results . . . . . . . . . . . . . . . . . . . 57

###### B In vitro methods and results 72

- B.1 Materials and instrumentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
- B.2 Cloning and transformation in 96-well plates . . . . . . . . . . . . . . . . . . . . . 73
- B.3 Protein expression and reaction screening in plates . . . . . . . . . . . . . . . . . . 73
- B.4 Protein expression and whole-cell validation reactions . . . . . . . . . . . . . . . . 74

- B.4.1 Analysis by GC-FID (compounds 1b - 4b) . . . . . . . . . . . . . . . . . . . 75
- B.4.2 Analysis by LC–MS (compound 4b for directed evolution) . . . . . . . . . 75
- B.4.3 Analysis of enantioselectivity . . . . . . . . . . . . . . . . . . . . . . . . . . 76

- B.5 Hemochrome assay . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
- B.6 Selectivity results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
- B.7 Directed evolution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
- B.8 DNA sequences of the best designs . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
- B.9 Calibration curves for GC-FID analysis . . . . . . . . . . . . . . . . . . . . . . . . 83

- B.9.1 Calibration curves for 1b . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
- B.9.2 Calibration curves for 2b . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
- B.9.3 Calibration curves for 3b . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
- B.9.4 Calibration curves for 4b . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86

- B.10 Calibration curves for LC-MS analysis . . . . . . . . . . . . . . . . . . . . . . . . . 87
- B.11 Synthesis of product standard 3b . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
- B.12 HPLC–UV chromatograms for enantioselectivity determination . . . . . . . . . . 90
- B.13 Chiral GC–FID traces for enantioselectivity determination . . . . . . . . . . . . . 106

###### A In silico methods and results

###### A.1 Notation

Denote the space of real numbers in dimension d as Rd. Let V = {1, . . . ,V} be a finite vocabulary set. We designate the final element of this set as a specialized mask token V = m, whereas the remaining V − 1 elements in V form the set of natural amino acids, including one unknown amino acid UNK. We use the diffusion time convention where time τ = 0 is associated with the data distribution pdata and τ = 1 is associated with pnoise. We describe proteins with L residues with backbone coordinates xstruct and sequence xseq as x = (xstruct, xseq) ∈ (Xstruct, Xseq) := (RL×4×3, VL) = X where X denotes the space of protein backbone coordinates with associated amino acid types. However, for simplicity, whenever it is clear from context, we also omit superscripts struct and seq to avoid notational clutter.

We use τ to denote inference time, with t(τ) to denote structure time and r(τ) to denote sequence time. We also use the shorthand t to denote t(τ) and r to denote r(τ).

Let ∆d := {v ∈ Rd : vi ≥ 0,i = 1, . . . , d, ∑id=1 vi = 1} represent the d-dimensional probability simplex. Each point on u ∈ ∆d corresponds to a categorical distribution Cat(j; u) = uj for j ∈ V. Let δ(x) ∈ ∆d be the one-hot distribution that places all its mass on x. The data distribution pdata is provided as an empirical distribution on n sequences in the form of a training set D = {x}n ⊂ X. We denote for x ∈ V, δ(x) ∈ ∆d given by Cat(j; δ(x)) = 1 if j = x and 0 otherwise. Finally, we reserve superscripts for set indexing purposes, e.g. xi,i ∈ [d], while subscripts are used to represent positions in time of a discrete sample xrseq,r ∈ [0,1].

###### A.2 Theoretical foundations

For a multi-modal space where elements contain multiple components, we describe the theoretical framework for modeling such systems, first individually, then in combination.

###### A.2.1 Continuous-space diffusion modeling

Generative modeling in continuous space systems can be formulated as the simulation of a Stochastic Differential Equation (SDE) forward in time to destroy information, then reverse in time to generate new samples from the target distribution.

In particular, during training, information is gradually destroyed from the training dataset by simulation with samples x ∼ pdata(x) using the following SDE:

dxt = ft(xt)dt + gtdWt, xt=0 ∼ pdata(x) (S1)

where ft(xt) : [0,1] × Xstruct → Xstruct is generally some pre-defined linear drift, Wt is the standard Wiener process, gt ∈ R≥0 is the diffusion coefficient, a scaling factor for the noise. The drift ft and the diffusion gt are chosen such that at pt=1 ≈ N(0, I).

Given this forward process, a reverse-time denoising SDE can be defined in the opposite direction,

dxt = −ft(xt) + gt2∇ log pt(xt) dt + gtdWt, xt=1 ∼ N(0, I) (S2)

where pt is the marginal distribution induced by the stochastic process in (S1). By training a model of the score function ∇ log pt(xt), the SDE can be reversed to generate points approxi-

mately from the data distribution. This leads to a natural parameterization using a denoising network Dθ(xt, t) which can be used to generate new samples using the following reverse-time denoising SDE:

Dθ(xt, t) − xt σt2

dxt = −ft(xt) + gt2

dt + gtdWt, xt=1 ∼ N(0, I) (S3)

where σt2 = 0 t gr2dr represents the total noise of the system at time t. It is possible to learn Dθ(xt) using the standard denoising score matching loss [35]:

Et,x0∼p

data,xt∼pt(xt|x0)

1

σt2 ∥Dθ(xt, t) − x0∥22 (S4)

###### A.2.2 Discrete-Space diffusion modeling

In the discrete-space setting, we require a slightly different treatment of discrete variables. We follow the discrete diffusion literature, which defines a similar noising and denoising process based on a masking process. In this case, continuous-time Markov chains (CTMC) or jump processes describe transitions on the discrete state spaces. In this section, we focus on the discrete time setting where time is divided up into T sub-intervals such that r˜ ∈ [0, T) represents normalized time r := r˜/T. This process can also be described in the fully continuoustime setting, which requires more rigorous treatment and can be found in related works such as [12, 17].

We focus on “masked” discrete diffusion modeling where information is destroyed by transitioning to a special mask token m independently for each xri ̸= m:

L

L

###### ∏

###### ∏

pt(xri|x0i ) =

Cat(xri; αrδ(x0i ) + (1 − αr)δ(m)). (S5)

pr(xr|x0) =

i=1

i=1

where αr describes the noise schedule similarly to gt in the continuous case, and is specified such that α0 = 1 and α1 = 0. This describes a process where at time r a fraction αr of the tokens are in their original state and 1 − αr of the tokens are in the masked state. The simplest conditional reverse transition kernel for a token xri is described by:

 

Cat(xri−1; δ(xri)) xri ̸= m Cat xri−1; (1−αr−1)δ(m)+(αr−1−αr)δ(x0i)

qt(xri−1|xri, x0i ) =

(S6)

1−αr xri = m.



This leads to a natural parameterization for the transitions of a discrete time generative model. We first define a (possibly) time-dependent denoiser network Dr,θ : VL × [0,1] → ∆L that predicts the probabilities of a clean sample x0 ∼ Dr,θ(xr):

 

Cat(xri−1; δ(xri)) xri ̸= m Cat xri−1; (1−αr−1)δ(m)+(αr−1−αt)D

pr,θ(xri−1|xri, Dri,θ(xr)) =

(S7)

i r,θ(xr)

1−αr xri = m.



where Dθi represents the predicted distribution over tokens for the i-th index of the sequence. Using this reverse parameterization, as T → ∞, there exists an evidence lower bound (ELBO) to

the log likelihood of the data distribution:

 

  dr (S8)

L

1 0

1 1 − αr

dαr

###### ∑

δ(x0i )T log Dri,θ(xr)

log pθ(x0) ≥ −

dr ·

Exr∼pr(·|x0)

i=1,xri=m

This can be used to train Dθ using standard stochastic gradient-descent type algorithms. It can be thought of as training a masked discrete diffusion model by optimizing a weighted

cross-entropy loss over masking levels αr weighted by ddrαr · 1−1αr.

However, there exist several major limitations of the denoising process in (S6), which is that as T → ∞,

- 1. Lack of Order Control. An analytic Gillespie-style sampler [36] shows that the denoising proceeds by uniformly randomly selecting a masked position, offering no control over the generation order of tokens.
- 2. Token Commitment. (S7) reveals that once unmasked, a specific token is committed to and never revisited. This means that if “mistakes” are made in the generation process, they can never be corrected. These two issues have led to a variety of solutions involving alternative reverse-time

processes for discrete generation. We use a path planning (P2) approach [17] which allows for “planned” unmasking and remasking of tokens during the diffusion process.

P2 departs from the standard MDM inference procedure in (S7) where the reverse-time transition pr,θ(xri−1|xri, Dri,θ(xr)) is independently sampled for each index by instead assigning the likelihood of denoising index i as a function of a separate planner GU which represents the probability of unmasking each index. This addresses limitation (1) of the standard denoising process. To address limitation (2), P2 also includes a remasking planner GM which represents the probability of an already unmasked token being first remasked then denoised to (potentially) a new value. This can be described with the following transition kernel, where z ∼ Dθ(xtr) is a sampled instantiation from the denoiser, and x¯r represents xr after remasking according to GM.

 

Cat xri−1; αr−1−αr

1−αr GUi (z, xr)δ(zi) xri = m Cat xri−1; (αr−1−αr)GiM(z,xr)

pr,θ(xri−1|xr, z) =

(S9)

(1−αr)(1−Cat(xri,Dθi(x¯r)))Dθi (x¯r) xri ̸= m,



in practice, we use a uniform GU and GM, as we find that a GM which depends on the confidence of the Dθ reduces structural diversity.

###### A.2.3 Multi-modal diffusion modeling

Having described separate processes for structure and sequence, it now remains to combine them to perform cogeneration in Diffusion for Sequence-Structure Co-design (DISCO).

We describe a protein x with L residues with sequence component xseq ∈ VL and backbone structure component xstruct ∈ (R4)L as x = (xseq, xstruct). A noisy protein with sequence noise level r and structure noise level t is then denoted as xr,t = (xrseq, xtstruct). We define the conditional forward process pr,t|0(xr,t|x0) to factorize over modality

pr,t|0(xr,t|x0) := pr|0(xrseq|x0seq)pt|0(xtstruct|x0struct) (S10)

where pr|0(xrseq|x0i,seq) and pt|0(xtstruct|x0struct) are defined similarly to the individual cases i.e.

pr|0(xrseq|x0seq) :=

L

Cat(xri,seq; αrδ(x0seq) + (1 − αr)δ(m)) (S11)

###### ∏

i=1

and pt|0(xtstruct|x0struct) is defined implicitly through simulation of the forward SDE

dxtstruct = ft(xtstruct)dt + gtdWt, xtstruct=0 = x0struct. (S12) To reverse this process, we need to define a reverse over both spaces that depends on

increments ∆s and ∆t.

p(xr−∆r,t−∆t|xr,t) = pr−∆r(xrseq−∆

|xr,t)pt−∆t(xtstruct−∆

t |xr,t) (S13) where pt−∆t(xtstruct−∆

r

|xr,t) is governed by the reverse-time SDE

t

dxtstruct = −ft(xtstruct) + gt2∇xstruct

log pt(xr,t) dt + gtdWt (S14)

t

and p(xrseq−∆

|xr,t) = ∏iL=1 p(xri,seq−∆

|xr,t) with

r

r

These equations govern the dynamics of the reverse process, but we still need to parameterize both the structure score relative to the partially noised xr,t as well as the sequence denoiser relative to the same state. To do this, we use a single model Dθ(xr,t). This allows us to perform inference along any trajectory of r, t pairs from (1,1) to (0,0) for valid generation up to numerical and learning errors. We use the following generative process given our denoising model Dθ with structure output component Dθstruct and sequence logit output component Dθseq:

 

1−αr (z, xr,t)δ(xri) xri,seq,t = m Cat xri−1; (αr−1−αr)GiM(z,xr,t)

Cat xri−1; αr−1−αr

ps−∆s,θ(xri−∆

r|xr,t, z) =

Dθi,seq(xr,t,r, t) xri ̸= m,



(1−αr)(1−Cat(xri,Dθi,seq(xr,t,r,t))

(S15)

Dθstruct(xr,t,r, t) − xstruct σt2

dxtstruct = −ft(xtstruct) + gt2

dt + gtdWt (S16)

Finally, it remains to define the training scheme for Dθ. Here, we combine independent losses and noising for each component individually. Specifically, we define independent distributions and losses over separate domains as in (S4) and (S8), but replace the modality-specific denoisers with our joint denoiser Dθ, which operates on a noisy input containing both modalities:

  

  

  

  

L

1

1 1−αr

dαr

δ(xseq)T log Dθi,seq(xr,t)

###### ∑

σt2∥Dθstruct(xr,t)−xstruct∥2−

dr ·

Er,t,x∼p

dataxr,t∼p(xr,t|x)

i=1 xti,seq=m

(S17)

These tools provide the necessary theory to perform multimodal training and inference. While theoretically any schedule over r, t can be used, we lock together the two schedules during inference only by defining the schedules in terms of a unifying time variable τ.

###### A.3 Data pipeline

We trained on the weighted PDB dataset described in AlphaFold 3 with a cutoff date of 202109-30. Data processing, including clustering and filtering, follows the procedures described in Sections 2.5.3 and 2.5.4 of the AlphaFold 3 Supplementary Material. The dataset was obtained by downloading with the Protenix scripts/database/download_training_data.sh script from commit a4059a6894c826b7bb81ba753cbdb89029313244.

We sample either single chains or chain-pair interfaces, with interfaces defined as pairs of chains with minimum heavy atom separation less than 5A˚. For r ∈ {chain, interface}, the dataset is weighted as

βr Nclust

w ∝

(αprot ∗ nprot + αnuc ∗ nnuc + αligand ∗ nligand)

with βchain = 0.5, βinterface = 1, αprot = 3, αnuc = 3, αligand = 1, and Nclust defined by the Protenix dataset clusters.

Cropping is performed per sample, where, as in AlphaFold 3 we use a contiguous crop, a spatial crop, and a spatial interface crop. For detailed algorithms, please refer to Section 2.7 of the AlphaFold 3 Supplementary Material [14]. We crop with the following weights: 20% contiguous cropping, 40% spatial cropping, and 40% spatial interface cropping.

###### A.4 Model architecture

While our architecture is broadly based on AlphaFold 3, we make a number of changes to enable protein sequence and structure co-design. Unless otherwise specified, we make use of the AlphaFold 3 architectural components and the interested reader may refer to the AlphaFold

- 3 Supplementary Information for detailed module-wise algorithms. As in AlphaFold 3, our model takes the form of a conditional diffusion model. However, unlike AlphaFold 3, we remove a number of the conditioning modules and add other domain-specific modules. We begin by removing the Template Module from the architecture as well as the MSA Module as computing MSAs for an ever evolving protein sequence trajectory during inference is computationally intractable. To maintain a source of evolutionary information, we replace the MSA Module with a protein language model, in our case DPLM 650M [13].We present the main model architecture in Algorithm 1.

Our tokenization scheme sees each amino acid residue and nucleotide correspond to a single token, while for other molecules, we encode each heavy atom as a separate token. Input features correspond to those used in AlphaFold 3, with the exception of all MSA and template features (see Table 5 in the AlphaFold 3 S.I. [14]). To obtain the initial trunk features, we feed the input features to the InputFeatureEmbedder (Algorithm 2 in AlphaFold 3 S.I.)

###### A.4.1 Denoiser architecture

The core of our model is the Denoiser (Algorithm 1), which iteratively refines the single (si) and pair (zij) representations over a series of cycles (Ncycle = 4). We first derive the initial features, {sinputsi }, using the InputFeatureEmbedder based on the input features {f∗}. The initial single and pair representations, siniti ∈ Rcs and zinitij ∈ Rcz, are generated via linear projections of the embedded inputs. We enrich the initial pair representations with relative position encodings and token bond features.

During each cycle c ∈ [1, . . . , Ncycle], the pair representation is first updated using a layernormalized linear projection. For cycles beyond the first (c > 1), the architecture employs a CrossModalEncode module to update representations based on the current sequence and structure states, xlseq and xlstruct, as well as skip connections. The model then integrates evolutionary information via the PlmModule, our drop-in replacement for the MSA Module of AlphaFold 3 and set to be DPLM 650M [13], updating the pair representations using features from the underlying protein language model. To encode chain breaks as input for the pLM, we insert a length 25 glycine linker between all protein chains. To obtain the single representations, we average the hidden states for each transformer block, while for the pair representations, we concatenate all attention weights for each attention block. Following an update to the single representation si, both representations are processed through the PairformerStack. As we believe that for de novo design we require a lighter pairformer than for typical folding tasks, we reduce the number of blocks in the Pairformer from 48 in AlphaFold 3 to 8. The outputs of the first cycle are cached as skip connections ({sskipi } and {zskipij }) for use in subsequent modules.

After completing all cycles, a final CrossModalEncode is applied before passing the representations to the DiffusionModule to produce the denoised sequence and structure estimates, xˆ0seq and xˆ0struct.

###### A.4.2 Cross-modal encoding

To inform the trunk representations with the current trajectory of the diffusion process and to better encourage self-consistency between both sequence and structure modalities, the CrossModalEncode module (Algorithm 2) integrates intermediate sequence and structure states. It first calls the DiffusionModule to obtain a preliminary estimate of the fully denoised states, xˆ0seq and xˆ0struct.

The noisy and denoised sequences are encoded using the frozen DPLM 650M, while the noised and denoised structures are encoded via a separate structure encoder, in our case a Message-Passing Neural Network following the LigandMPNN architecture [15], yielding embeddings useq0 , useql , vseq0 , and vseql , respectively. The structure encoder weights are initialized from LigandMPNN weights and subsequently trained end-to-end along with the rest of the weights in the overall model, with the structure encoder embeddings being passed through a final layer norm and linear layer before being added to the single representation. These are subsequently added to the single representation si. Inspired by CarbonNovo [37], a distance map Dij is computed from the step 0 structure estimate xˆ0struct using a distance one-hot encoder. This spatial information is injected directly into the pair representation zij.

###### A.4.3 Diffusion module

The DiffusionModule (Algorithm 3) dictates the generation of both sequence and structural updates. It begins by conditioning the trunk representations on the current structure noise (tˆ) and sequence noise (rˆ) via DiffusionConditioning. The noisy structure coordinates, xlnoisy, are then scaled to dimensionless vectors with approximately unit variance (rlnoisy).

To process atom-level geometries, the model utilizes an AtomAttentionEncoder which aggregates atomic features into coarse-grained, sequence-local token representations {ai}. These token representations are then passed through a 24-block DiffusionTransformer featuring full self-attention.

Following the transformer, the architecture decodes the token activations back to the atomic level using the AtomAttentionDecoder. This decoder is run twice with different configurations for the sequence and structure outputs, with each configuration being a separately trained AtomAttentionDecoder head:

- 1. Structure Update: Run with seq_mode = False to compute coordinate updates (rlupdate).
- 2. Sequence Logits: Run with seq_mode = True to predict sequence logits (xllogits), where in sequence mode we scatter add the atom-level embeddings back to the token level before finally predicting the sequence logits.

Finally, the predicted positional update rlupdate is rescaled and combined with the input noisy positions using the following schedule equation to yield the output structure xlout:

σdata · tˆ σdata2 + tˆ2

σdata2 σdata2 + tˆ2

xlnoisy +

rlupdate (S18)

xlout =

###### A.4.4 Diffusion conditioning

The DiffusionConditioning block (Algorithm 4) modulates the input representations based on the current noise levels. The pair representations zij are concatenated with relative position encodings and processed through a 2-layer Transition block.

For the single representations si, trunk outputs and input features are concatenated and projected. As the sequence and structure times may be different, we opt to embed them separately to help the model distinguish between sequence and structure noise levels. We embed the structure and sequence noise levels using a FourierEmbedding layer. The structure noise tˆ is embedded logarithmically as 14 log(tˆ/σdata), while the sequence noise rˆ is embedded directly. Both noise embeddings are added to si before it undergoes a 2-layer Transition block.

###### A.4.5 Atom attention decoder

The AtomAttentionDecoder (Algorithm 5) maps the processed token-level features back to the required output space. Token activations ai are broadcast to their constituent atoms and summed with the corresponding atom-level skip connections qskipl . This representation is passed through a 3-block, 4-head AtomTransformer using cross-attention.

If the decoder is operating in sequence mode (seq_mode = True), the updated atomic features are aggregated back into token-level representations using a scatter-add operation. The final output yˆl is then generated via layer normalization and a linear projection without biases.

###### A.5 Training

All training hyperparameters are detailed in Table S1. The model was trained with distributed data parallel across 32 L40S GPUs for 11 days for 160,000 training steps. We treated any atomic position with probability of being resolved less than 0.9 as unresolved and mask out these atoms (or residues) when evaluating the losses. Dropout is only applied in the Pairformer module, and we fully freeze the pre-trained DPLM 650M weights during training. We use a series of losses to train DISCO as described below. DISCO is comprised of 888 million parameters in total, with 235 million trainable parameters.

###### Algorithm 1 Denoiser

- 1: function DENOISER({f∗}, xlseq, xlstruct, Ncycle = 4, cs = 384, cz = 128)
- 2: {sinputsi } = INPUTFEATUREEMBEDDER({f∗})
- 3: siniti = LINEARNOBIAS(sinputsi ) siniti ∈ Rcs
- 4: zinitij = LINEARNOBIAS(sinputsi ) + LINEARNOBIAS(sinputsj ) zinitij ∈ Rcz
- 5: zinitij += RELATIVEPOSITIONENCODING({f∗})
- 6: zinitij += LINEARNOBIAS(fijtoken_bonds)
- 7: {zˆij}, {sˆi}, {sskipi }, {zskipij } = 0, 0, 0, 0
- 8: for all c ∈ [1, . . . , Ncycle] do
- 9: zij = zinitij + LINEARNOBIAS(LAYERNORM(zˆij)) zij ∈ Rcz
- 10: if c > 1 then
- 11: {si}, {zij} = CROSSMODALENCODE({si}, {zij}, xlseq, xlstruct, {sskipi }, {zskipij })
- 12: end if
- 13: {zij} += PLMMODULE({fSi}, {zij}, {sinputsi })
- 14: si = siniti + LINEARNOBIAS(LAYERNORM(sˆi)) si ∈ Rcs
- 15: {si}, {zij} = PAIRFORMERSTACK({si}, {zij})
- 16: {sˆi}, {zˆij} ← {si}, {zij}
- 17: if c == 1 then
- 18: {sskipi }, {zskipij } = {sˆi}, {zˆij}
- 19: end if
- 20: end for
- 21: {si}, {zij} = CROSSMODALENCODE({si}, {zij}, xlseq, xlstruct, {sskipi }, {zskipij })
- 22: {xˆ0seq}, {xˆ0struct} = DIFFUSIONMODULE({f∗}, {sinputsi }, {si}, {zij}, {sskipi }, {zskipij }, xlseq, xlstruct)
- 23: end function

###### Algorithm 2 Cross-Modal Encoding

- 1: function CROSSMODALENCODE({si}, {zij}, xlseq, xlstruct, {sskipi }, {zskipij }, xlseq, xlstruct)
- 2: {xˆ0seq}, {xˆ0struct} = DIFFUSIONMODULE({f∗}, {sinputsi }, {si}, {zij}, {sskipi }, {zskipij }, xlseq, xlstruct)
- 3: useq0 = LINEARNOBIAS(LAYERNORM(PLMENCODE(xˆ0seq)))
- 4: useql = LINEARNOBIAS(LAYERNORM(PLMENCODE(xˆlseq)))
- 5: vseq0 = LINEARNOBIAS(LAYERNORM(STRUCTUREENCODE(xˆ0struct)))
- 6: vseql = LINEARNOBIAS(LAYERNORM(STRUCTUREENCODE(xˆlstruct)))
- 7: Dij = LINEARNOBIAS(DISTANCEONEHOTENCODER(xˆ0struct))
- 8: si = si + useq0i + useqli + vstruct0i + vstructli
- 9: zij = zij + Dij
- 10: return {si}, {zij}
- 11: end function

- Algorithm 3 Diffusion Module

- 1: function DIFFUSIONMODULE( {xlnoisy struct}, {xlseq}, tˆ, rˆ, {f∗}, {sinputsi }, {strunki }, {ztrunkij },

{sskipi }, {zskipij }, σdata = 16, catom = 128, catompair = 16, ctoken = 768) # Conditioning

- 2: {si}, {zij} = DIFFUSIONCONDITIONING(tˆ,rˆ, {f∗}, {sinputsi }, {strunki }, {ztrunkij }, σdata) # Scale positions to dimensionless vectors with approximately unit variance.
- 3: rlnoisy = xlnoisy tˆ2 + σdata2 rlnoisy ∈ R3

- 4: {strunki }+ = LINEARNOBIAS(LAYERNORM({sskipi }))
- 5: {ztrunkij }+ = LINEARNOBIAS(LAYERNORM({zskipij })) # Sequence-local Atom Attention and aggregation to coarse-grained tokens
- 6: {ai}, {qskipl }, {cskipl }, {pskiplm } = ATOMATTENTIONENCODER(

{f∗}, {rlnoisy}, {strunki }, {zij}, catom, catompair, ctoken

) ai ∈ Rctoken # Full self-attention on token level.

- 7: ai += LINEARNOBIAS(LAYERNORM(si))
- 8: {ai} ← DIFFUSIONTRANSFORMER({ai}, {si}, {zij}, βij = 0, Nblock = 24, Nhead = 16)
- 9: ai ← LAYERNORM(ai) # Broadcast token activations to atoms and run Sequence-local Atom Attention
- 10: {rlupdate} = ATOMATTENTIONDECODER({ai}, {qskipl }, {cskipl }, {pskiplm },seq_mode = False)
- 11: {xllogits} = ATOMATTENTIONDECODER({ai}, {qskipl }, {cskipl }, {pskiplm },seq_mode = True) # Rescale updates to positions and combine with input positions
- 12: xlout =

σdata2

σdata2 + tˆ2 · xlnoisy +

σdata · tˆ σdata2 + tˆ2

· rlupdate

- 13: return {xlout}, {xllogits}
- 14: end function

- Algorithm 4 Diffusion Conditioning

- 1: function DIFFUSIONCONDITIONING(tˆ, rˆ, {f∗}, {sinputsi }, {strunki }, {ztrunkij }, σdata, cz = 128, cs = 384)

# Pair conditioning

- 2: zij = concat([ztrunkij , RELATIVEPOSITIONENCODING({f∗})])
- 3: zij ← LINEARNOBIAS(LAYERNORM(zij)) zij ∈ Rcz
- 4: for all b ∈ [1,2] do
- 5: zij += TRANSITION(zij, n = 2)
- 6: end for # Single conditioning, t is structure noise, r is sequence noise
- 7: si = concat([strunki , sinputsi ])
- 8: si ← LINEARNOBIAS(LAYERNORM(si)) si ∈ Rcs
- 9: nstruct = FOURIEREMBEDDING 1

4 log(tˆ/σdata), 256

- 10: nseq = FOURIEREMBEDDING(rˆ, 256)
- 11: si += LINEARNOBIAS(LAYERNORM(nstruct)) + LINEARNOBIAS(LAYERNORM(nseq))
- 12: for all b ∈ [1,2] do
- 13: si += TRANSITION(si, n = 2)
- 14: end for
- 15: return {si}, {zij}
- 16: end function

- Algorithm 5 Atom Attention Decoder

- 1: function ATOMATTENTIONDECODER({ai}, {qskipl }, {cskipl }, {pskiplm },seq_mode)
- 2: # Broadcast per-token activations to per-atom activations and add the skip connection
- 3: ql = LINEARNOBIAS(atok_idx(l)) + qskipl
- 4: # Cross attention transformer
- 5: {ql} = ATOMTRANSFORMER({ql}, {cskipl }, {pskiplm }, Nblock = 3, Nhead = 4)
- 6: # Scatter add atom embeddings back to token embeddings if sequence mode
- 7: if seq_mode then
- 8: ql = SCATTERADD(ql,tok_idx(l))
- 9: end if
- 10: # Map to positions update or logits
- 11: yˆl = LINEARNOBIAS(LAYERNORM(ql))
- 12: return {yˆl}
- 13: end function

###### A.5.1 Structure diffusion losses

The losses used for the structure diffusion module are largely the same as those used in AlphaFold 3. We recapitulate them here for convenience. The final structure loss is a weighted combination of the following loss.

MSE Loss. The main loss used for the structure portion of our diffusion model takes the form of a weighted aligned MSE loss to the predicted denoised structure. We first perform a rigid alignment of the ground truth structure ⃗xlGT

{⃗xlGT-aligned} = weighted_rigid_align({⃗xlGT}, {⃗xlstruct}, {wl}) (S19) where the weights wl are defined as

wl = 1 + flis_dnaαdna + flis_rnaαrna + flis_ligandαligand (S20) and αdna = αrna = 5 and αligand = 10. The MSE loss is then defined as

LMSE =

1 3

mean

l

wl ⃗xlstruct −⃗xlGT-aligned

2

(S21)

Notably, we do not take the loss on any atomic positions which are unresolved.

Smooth LDDT Loss. The Smooth LDDT Loss measures the accuracy of predicted atomic coordinates relative to ground truth structures. For each pair of atoms, we compute the difference between predicted and ground truth pairwise distances, then score each pair using a differentiable approximation of the discrete LDDT thresholds via a sum of sigmoid εlm ∈ [0,1]. Pairs are masked by an inclusion radius that is modality-aware: nucleotide atoms (DNA/RNA) are evaluated within 30 Å, while all other atoms are evaluated within 15 Å. The final loss is one minus the masked mean per-pair score, such that a loss of 0 corresponds to perfect coordinate recovery. The Smooth LDDT loss is detailed in Algorithm 6.

Full Structure Diffusion Loss. The full structure diffusion loss is a weighted combination of the MSE and smooth LDDT losses with

tˆ2 + σdata2

(tˆ+ σdata)2LMSE + Lsmooth_lddt (S22)

Lstruct =

where tˆ is the sampled noise level and σdata is a constant based on the variance of the data (we use σdata = 16). During training the structure noise level tˆ is sampled from σdata · exp(−1.2 + 1.5 · N(0,1)). Before computing the structure diffusion losses, we apply an optimal ground truth chain assignment as described in Subsection 4.2 of the AlphaFold 3 S.I.

###### A.5.2 Sequence diffusion losses

While the structure diffusion module is a weighted combination of multiple losses, for the sequence diffusion head, we instead make use exclusively of the masked diffusion loss from (S8), introduced in [16, 38]. In our paper we use a linear schedule αr = 1 − r so that the loss weight ddrαr · 1−1αr = −1r. When computing the loss, we mask out unreliable residues. In particular, we mask out any residues for which any of its constituent backbone atoms have occupancy less than 0.8, any residue for which any backbone atom is unresolved, and any residue which is part

- Algorithm 6 Smooth LDDT Loss

- 1: function SMOOTHLDDTLOSS({xl}, {xlGT}, {flis_dna}, {flis_rna})
- 2: # Compute distances between all pairs of atoms
- 3: δxlm ← ∥xl − xm∥
- 4: δxlmGT ← ∥xlGT − xmGT∥
- 5: # Compute distance difference for all pairs of atoms
- 6: δlm ← δxlmGT − δxlm
- 7: εlm ← 14 sigmoid 2 1 − δlm + sigmoid(1 − δlm) + sigmoid(2 − δlm) + sigmoid(4 − δlm)

- 8: # Restrict to bespoke inclusion radius
- 9: flis_nucleotide ← flis_dna + flis_rna
- 10: clm ← δxlmGT < 30Å flis_nucleotide + δxlmGT < 15Å 1 − flis_nucleotide
- 11: # Compute mean, avoiding self term
- 12: lddt = mean

l̸=m

(clm εlm) / mean

l̸=m

(clm)

- 13: return 1 − lddt
- 14: end function

of a histag. We define a residue to be part of a histag if it is part of a consecutive run of 5 or more histidine residues.

In parameterizing our log denoiser Dθseq, we utilize the SUBS parameterization from [16], wherein as the reverse process never predicts a m token, we substitute −∞ in for the logit corresponding to the m token. During training, we sample the sequence time r uniformly from [0,1]. When masking the sequence, we ensure that all amino acid residue information is not leaked to the model by masking all reference information features in addition to any sequence identity features. Specifically, we replace the reference position, mask and charge features with masked versions of themselves. We set the masked reference position to be the average reference conformer position for each backbone atom, averaged over all 20 standard amino acid residues, while for the masked reference charge and mask, we set them to be identical to those of the glycine backbone atoms.

- A.5.3 Full training loss

The full training loss is a weighted combination of the sequence and structure losses, as well as a distogram loss wherein our distogram head and loss are identical to AlphaFold 2 [39].

Ltotal = αseqLseq + αMSELMSE + αsmooth_lddtLsmooth_lddt + αdistogramLdistogram (S23) where we set αseq = 1, αMSE = 4, αsmooth_lddt = 4, αdistogram = 0.03.

- A.5.4 Training hyperparameters We present all training hyperparameters in Table S1.
- A.5.5 Model selection

We choose our final checkpoint by evaluating EMA checkpoints on the unconditional generation task. We select the checkpoint with the best tradeoff of % co-designability and structural cluster

###### Co-Designability & Structural Diversity vs Checkpoint

| |
|---|

120

| |
|---|

| |
|---|

70

| |
|---|

| |
|---|

| |
|---|

NStructuralClusters

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Co-Designable(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

100

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

40

| |
|---|

60

30

20

Co-Designable (%)

40

| |
|---|

N Structural Clusters

10

| |
|---|

0 1 × 105 2 × 105 3 × 105 4 × 105 5 × 105 6 × 105

Training Step

Figure S1. Co-designability increases while structural diversity decreases as training progresses.

designability. Interestingly, in Figure S1 we observe that structural diversity peaks around 200k training steps and decreases throughout the rest of training, indicating overfitting. We select checkpoint 160k as our final model, given its superior tradeoffs. We note that the inference settings used to generate this plot were those of Table S3, which yields improved structural clusters with slightly degraded co-designability.

Table S1. Model and Training Hyperparameters

###### Category Hyperparameter Value

Global Dimensions & Training

Global Single Representation Dimension (cs) 384 Global Pair Representation Dimension (cz) 128 Global Gradient Clipping Norm 10.0 Global EMA Decay Rate 0.999 Global Sequence Batch Size 32 Global Structure Batch Size 96 Global Crop Size 384

Optimizer (Adam) Adam Learning Rate 0.00018

- Adam β1 0.9
- Adam β2 0.95 Adam Weight Decay 1 × 10−8

Learning Rate Scheduler Scheduler Warmup Steps 1000 Continued on next page

Table S1 – continued from previous page Category Hyperparameter Value Scheduler Warmup Type Linear Scheduler Decay Factor 0.95 Scheduler Decay Every N Steps 50000

Core Model Architecture

Model Max Number of Cycles (Ncycle) 4 PLM PLM Base Model DPLM 650M

Input Embedder Token Dimension (ctoken) 384 Input Embedder Atom Dimension (catom) 128 Input Embedder Atom Pair Dimension (catompair) 16 Relative Position rmax 32 Relative Position smax 2

Pairformer Stack

Pairformer Number of Blocks 8 Pairformer Number of Attention Heads 16 Pairformer Dropout 0.25

Diffusion Module

Diffusion Token Dimension (ctoken) 768 Diffusion σdata 16 Diffusion Transformer Number of Blocks 24 Diffusion Transformer Number of Attention Heads 16 Atom Encoder Number of Blocks 3 Atom Encoder Number of Attention Heads 4 Structure Atom Decoder Number of Blocks 3 Structure Atom Decoder Number of Attention Heads 4 Sequence Atom Decoder Number of Blocks 3 Sequence Atom Decoder Number of Attention Heads 4

###### A.6 Inference

We found that inference techniques are critical for model performance, observing in our experiments that the co-designability metrics can scale from 16% to 88% with the same checkpoint, depending on inference settings. Below, we detail the inference pipeline and progressively introduce each technique responsible for this improvement.

###### A.6.1 Inference sampling

Our inference-time sampling scheme is shown in Algorithm 7, where the structure inference procedure is largely based off of that of AF3.

- Algorithm 7 SampleDiffusion

Require: {f∗}, Structure Noise Schedule [c0, c1, . . . , cT], Sequence Time Schedule [r0,r1,...,rT], γ0 = 0.8, γmin = 1.0, noise scale λ = 0.1, step scale η = 1.5, do_noisy_guidance = false

- 1: xlstruct ∼ c0 · N (0,I3) xl ∈ R3
- 2: xlseq = [m,m, . . . ,m]
- 3: for each (cτ,rτ) ∈ zip([c1, . . . , cT], [r1, . . . ,rT]) do
- 4: {xlstruct} ← CENTRERANDOMAUGMENTATION({xlstruct})
- 5: γ = γ0 if cτ > γmin else 0
- 6: tˆ = cτ−1(γ + 1)
- 7: ξl = λ tˆ2 − c2τ−1 · N (0,I3) ξl ∈ R3

- 8: xlstruct,noisy = xlstruct + ξl
- 9: {xlstruct,denoised}, {xlseq,logits} = DENOISER({xlstruct,noisy}, {xlseq}, tˆ, rτ, {f∗})
- 10: if do_noisy_guidance then
- 11: {xlstruct,denoised}, {xlseq,logits} = NOISYGUIDANCE({xlstruct,noisy}, {xlseq}, tˆ, rτ, {f∗})
- 12: end if
- 13: δl = (xl − xldenoised)/tˆ
- 14: dt = cτ − tˆ
- 15: xlstruct ← xlstruct,noisy + η · dt · δl
- 16: xlseq = PATHPLANNINGSTEP(xlseq,logits, bτ)
- 17: end for
- 18: return {xl}

###### A.6.2 Sequence self-correction

Standard masked diffusion inference is limited due to the structure of its reverse process. Specifically, the masked diffusion reverse process requires that once a particular token is unmasked, it retains its value throughout the rest of the inference trajectory. This, however, prevents the model from correcting its mistakes and, for optimal performance, requires the model to have learned perfectly how to generate a particular sequence in every n! possible unmasking orderings. Given limited compute budgets learning, such a model is impractical, and significant performance gains have been observed by allowing a masked discrete diffusion model to self-correct its errors [17, 18, 40].

We similarly find that using standard masked discrete diffusion inference performs poorly. To remedy, this we adopt a random remasking strategy [17] wherein at each sequence time r we uniformly choose r percent of tokens to be masked and 1 − r percent to be unmasked with the full procedure summarized in Algorithm 8. This allows the model to correct mistakes it made during the inference trajectory, especially at critical structural residues such as those of loopy regions of a structure and at regions where the structure shifts from disordered to structured.

###### A.6.3 Entropy adaptive sequence temperature

We observed that the model tends to select particular overconfident residues early on during an inference trajectory, which harms overall co-designability. To remedy this, we applied a novel adaptive temperature scheme for the sequence logits, whereby the distribution of overly confident tokens is flattened early in the inference trajectory.

- Algorithm 8 Path Planning Sampling Require: x1 ← [m, . . . ,m], Seq length N, Denoiser Dθseq, Schedule κ, Num sampling steps T

- 1: for t ∈ 1, T−1

T ,...,0 do

- 2: Plan:
- 3: z ∼ Dθseq(xt)
- 4: UnmaskPos ← RandSampleUniform(⌊κ(t) · N⌋, {1, . . . , N})
- 5: MaskPos ← {1, . . . , N} \ UnmaskPos
- 6: Denoise:
- 7: xtj ←

zj if j ∈ UnmaskPos ∧ xtj = m m if j ∈ MaskPos

- 8: end for
- 9: return xL

Hmax − H(n) Hmax

τrentropy,(n) = 1 + rγβ

(S24)

where H(n) is the entropy of the nth token’s distribution induced by the denoiser µθ, Hmax = − log(20) is the maximum entropy of a distribution over all amino acids, β is a constant controlling the strength of the entropy scaling, and γ controls how quickly we turn off entropy adaptive tempering. We also maintain a global logit temperature τrglobal for which we use a step function.

 

0.8 if r ≥ 0.2 0.1 if r < 0.2

τrglobal =



The final logit temperature at time r for token position n is then

(S25)

τr(n) = τrglobal · τrentropy,(n) (S26)

Hmax − H(n) Hmax

= τrglobal + τrglobalrγβ

(S27)

where the entropy term approaches 0 as the sequence time r → 0. Writing the logits for the nth token induced by the denoiser as {xseq,logits,(n)} we finally obtain the tempered distribution we draw sequence samples from prτ(xi(n)) ∝ exp(xseq,logits,(n)/τr(n)). We find that the application of both these sequence temperature methods in conjunction is critical for performance.

###### A.6.4 Inference schedule

Figure S2 illustrates the three noise schedules evaluated as inference settings for joint sequencestructure co-design. The naïve schedule (left) pairs the default AlphaFold 3 structure noise schedule with a linear sequence masking schedule, serving as our baseline. The most codesignable schedule (middle) and most diverse schedule (right) modify the sequence masking trajectory, with the latter employing a mask_rate = t1.5 to concentrate unmasking toward the end of the denoising process, encouraging greater sequence diversity. Consistent with prior works on diffusion-based protein structure design [4], we found that the noise scale λ applied to the structure diffusion process is critical for sample quality—excessively large noise scales

###### Naïve Schedule

###### Most Co-Designable Schedule

###### Most Diverse Schedule

4000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

3500

0.8

3000

SequenceMask%

StructureNoise

2500

- 0.6

2000

0.4

1500

1000

0.2

500

0.0

0

1.0 0.8 0.6 0.4 0.2 0.0

1.0 0.8 0.6 0.4 0.2 0.0

1.0 0.8 0.6 0.4 0.2 0.0

Time

Figure S2. Various noise schedules used as inference settings. The leftmost plot displays a naïve schedule representing the default AlphaFold 3 settings paired with a linear sequence schedule. The middle plot is the schedule used in our most co-designable settings, while the right is our most diverse settings (using mask_proportion = t1.5 for sequence).

destabilize generation, while overly small values collapse diversity. In practice, we set λ = 0.1 for all reported experiments, which we found to strike the best balance between structural plausibility and sequence diversity across our benchmark tasks.

###### A.6.5 Conditional generation

Conditional generation of proteins in complex with other biomolecules extends naturally from our framework. During inference, the conditioning molecule (e.g. a small-molecule ligand or nucleic acid chain) is provided as an input feature: its reference conformer geometry and bond information are supplied through the standard AlphaFold 3 conditioning features, and its atomic coordinates are allowed to be denoised throughout the reverse diffusion trajectory. The protein sequence and structure are generated jointly in the same setting as above, while the conditioning molecule’s features participate in attention and pairformer computations, allowing the model to shape the protein around the target binding partner (and vice versa). No additional training or fine-tuning is required.

###### A.6.6 Noisy guidance

Noisy guidance [12] is a guidance mechanism that arises naturally from a multimodal diffusion framework with decoupled noise schedules. We detail the entire noisy guidance algorithm in

- Algorithm 9. Standard Classifier-Free Guidance (CFG) interpolates between a conditional and unconditional score with guidance strength ω:

ωsθ(xtstruct, t, c) + (1 − ω)sθ(xtstruct, t,∅). (S28)

Noisy guidance exploits the fact that we train with a decoupled noise schedule for sequence and structure to replace the unconditional score with a conditional score evaluated at a partially noised condition xρseq at noise level ρ > r:

ωsstructθ (xtstruct, xrseq, t,r) + (1 − ω)sstructθ (xtstruct, xρseq, t, ρ), ρ > r. (S29)

This recovers vanilla CFG when r = 0 and ρ = 1, but more generally allows the guiding model to be a partially conditioned score rather than a fully unconditional one. This is conceptually related to Autoguidance [41], which uses a degraded model as the corrector to reduce errors in

- Algorithm 9 Multimodal Noisy Guidance

Require: {xlstruct,noisy}, {xlseq}, tˆ, rτ, {f∗}, guidance weight ω, noisy times ψseq > rτ, ψstruct > tˆ,

rescaling weight ϕ, guidance intervals (ξminstruct, ξmaxstruct), (ξminseq , ξmaxseq )

- 1: # Conditional denoiser pass at current noise levels
- 2: {xcondstruct,denoised}, {ℓseqcond} = DENOISER({xlstruct,noisy}, {xlseq}, tˆ,rτ, {f∗})
- 3: # Structure noisy guidance: corrupt sequence
- 4: if tˆ ∈ (ξminstruct, ξmaxstruct) then
- 5: {xlseq,ψ} ← CORRUPT({xlseq}, ψseq)
- 6: {xngstruct,denoised},_ = DENOISER({xlstruct,noisy}, {xlseq,ψ}, tˆ, ψseq, {f∗})
- 7: σcond ← std {xcondstruct,denoised} , σng ← std {xngstruct,denoised}
- 8: xlstruct,rescaled ← xngstruct,denoised ·

σcond σng

- 9: {xlstruct,denoised} ← ϕ · xlstruct,rescaled + (1 − ϕ) · xcondstruct,denoised
- 10: else
- 11: {xlstruct,denoised} ← {xcondstruct,denoised}
- 12: end if
- 13: # Sequence noisy guidance: corrupt structure
- 14: if rτ ∈ (ξminseq , ξmaxseq ) then
- 15: ξlstruct = (ψstruct)2 − tˆ2 · N (0,I3)

- 16: {xlstruct,ψ} ← {xlstruct,noisy} + ξlstruct
- 17: _, {ℓseqng } = DENOISER({xlstruct,ψ}, {xlseq}, ψstruct,rτ, {f∗})
- 18: {ℓseq,denoisedl } = ω · {ℓseqcond} + (1 − ω) · {ℓseqng }
- 19: else
- 20: {ℓseq,denoisedl } ← {ℓseqcond}
- 21: end if
- 22: return {xlstruct,denoised}, {ℓseq,denoisedl }

the conditional score — here the same effect is achieved by degrading the condition rather than the model. Noisy guidance can also be applied to unconditional joint generation by adaptively choosing ρ > r as both modalities are being generated simultaneously.

For the discrete modality, the guidance interpolation is performed in logit space rather than directly averaging the discrete scores. Concretely, given conditional and unconditional logits ℓcond and ℓuncond, the guided discrete score is

sseqθ ← softmax(ω · ℓcond + (1 − ω) · ℓuncond) , (S30)

rather than a geometric average of the softmax outputs as proposed in some prior discrete guidance work. This arithmetic interpolation in logit space has been shown to carry theoretical advantages over direct probability-space averaging [42].

In practice, we find that in continuous space, the magnitude of the conditional score does not match the magnitude of the noised score, or equivalently, the conditional and noised structure denoiser outputs. To remedy this, we take the approach of [43] and rescale the guided structure denoiser output based on the ratio of standard deviations of conditional and noised structure denoiser outputs Dθstruct,cond, Dθstruct,ng as

σcond = std(Dθstruct,cond), σng = std(Dθstruct,ng ) (S31)

σcond σng

Dθstruct,rescaled = Dθstruct,cond ·

(S32)

Dθstruct,final = ϕ · Dθstruct,rescaled + (1 − ϕ) · Dθstruct,cond (S33)

Similar to the usage of CFG, we only apply noisy guidance within a guidance interval t ∈ (ξminstruct, ξmaxstruct) and r ∈ (ξminseq , ξmaxseq ). To enable multimodal noisy guidance, for each modality, we fix a constant modality and add noise to the other modality before feeding both through the model to produce denoiser predictions on the current timestep and noisy inputs before adjusting the respective score function based on the modality we noised. Finally, we found that setting the noisy sequence and structure times ψseq and ψstruct to constants worked best and performed experiments using this.

###### A.6.7 Inference hyperparameters

We specify the hyperparameters used for both the unconditional benchmark in Table S2 and those for the conditional benchmark in Table S3. We note that we used different parameters for the unconditional and conditional settings as we found that a different set of parameters yielded more co-designable, diverse structures while slightly degrading the % co-designability, with each setting sitting at different points on the Pareto frontier.

Table S2. Unconditional Inference Hyperparameters

###### Category Hyperparameter Value

Model Model Number of Cycles (Ncycle) 4

Inference Noise Scheduler Noise Scheduler σmax 160.0 Noise Scheduler σmin 0.0004 Noise Scheduler ρ 7 Noise Scheduler σdata 16.0

Structure Diffusion Sampling

- Sampling γ0 0.8 Sampling γmin 1.0 Sampling Noise Scale (λ) 0.1 Sampling Step Scale (η) 1.5 Sampling Number of Steps (Nstep) 200

Noisy Guidance Noisy Guidance Enabled True

Noisy Guidance ωstruct 1.5 Noisy Guidance ωseq 2.0 Noisy Guidance Unconditional Seq. Time 0.6 Noisy Guidance Unconditional Struct. Time 0.8 Noisy Guidance Rescale ϕ 0.7 Noisy Guidance Guidance End Time 0.3 Noisy Guidance Guidance Start Time 0.8

Sequence Sampling Strategy

Seq. Sampling Strategy Path Planning Seq. Sampling Score Type Random

 

0.8 if t ≥ 0.2 0.1 if t < 0.2

Seq. Sampling Logits Temperature



Seq. Sampling Entropy Adaptive β 1.0 Seq. Sampling Entropy Adaptive γ 1.0

Sequence Noise Scheduler Seq. Noise Scheduler Schedule t

Table S3. Conditional Inference Hyperparameters

###### Category Hyperparameter Value

Model Model Number of Cycles (Ncycle) 4

Inference Noise Scheduler Noise Scheduler σmax 160.0 Noise Scheduler σmin 0.0004 Noise Scheduler ρ 7 Noise Scheduler σdata 16.0

Structure Diffusion Sampling

- Sampling γ0 1.6 Sampling γmin 1.0 Sampling Noise Scale (λ) 0.1 Sampling Step Scale (η) 1.5 Sampling Number of Steps (Nstep) 200

Noisy Guidance Noisy Guidance Enabled False Sequence Sampling Strategy

Seq. Sampling Strategy Path Planning Seq. Sampling Score Type Random

 

0.8 if t ≥ 0.2 0.1 if t < 0.2

Seq. Sampling Logits Temperature



Seq. Sampling Entropy Adaptive Temp. False Sequence Noise Scheduler Seq. Noise Scheduler Schedule t1.5

###### A.7 Inference steering

Inference-time steering enables the adaptation of a single pretrained model to diverse and evolving design objectives at sampling time, without the need for retraining. Desired properties are incorporated directly into the generation process, tilting the distribution of samples at each step. Common guidance approaches rely on heuristic score modifications in diffusion models, which only approximately target conditional distributions and often compromise control over the resulting marginals. In this work, we instead consider the Feynman-Kac Corrector (FKC) framework [21], which enables principled sampling from desired target distributions. In the continuous case, this is done by updating samples using a weighted stochastic differential equation (SDE). The Feynman-Kac PDE describes the time-evolution of the density of samples as

∂ptFK(x) ∂t

= −⟨∇, ptFK(x)vt(x)⟩ +

gt2 2

∆ptFK(x) + ptFK(x)(ht(x) − ht(x)ptFK(x)dx) , (S34)

where the first two terms correspond to the Fokker-Planck PDE and the last term is a reweighting PDE, assigning each sample a time-dependent log-weight wt based on the weighting function ht(x). In the discrete setting [22], the analogous time-evolution of weighted marginals can be described by weighted Forward Kolmogorov Equations (FKEs) of the form.

∂ptFK(i) ∂t

###### = ∑

(At(j,i)pt(j) − At(i, j)pt(i)) + pt(i) ht(i) − Ept(i)ht(i) , (S35)

j̸=i

where At(i, j) is the rate matrix. To sample from ptFK(x) using FKC, three general steps are required at every timestep t. We

start with a batch of K particles (or samples); at each timestep, we do the following to each particle k:

- 1. Update its state xt based on the reverse process using Eq. (S16) (continuous) or Eq. (S15) (discrete).
- 2. Compute its log-weight update log wtk+dt = log wtk + ht(xk)dt.
- 3. Normalize the weights and use them to resample the particles in the batch. The state and weight updates are derived based on the target distribution we wish to sample

from. In Section A.7.1 and Section A.7.4, we demonstrate two use cases of FKC steering and write the state and weight updates for each.

###### A.7.1 Feynman-Kac Correctors - Specificity Guidance

The goal of specificity guidance is to generate protein samples that preferentially bind to a desired ligand (on-target) while avoiding binding to an undesired one (off-target). We formalize this as sampling from the target density.

pSGt,β(x) ∝ qont (x)(1−β)qofft (x)β (S36)

with β < 0. By setting β < 0, the off-target distribution acts as a repulsive term, encouraging samples that are likely under qon while penalizing those that are likely under qoff. In Section A.7.1, we present FKC - Specificity Guidance (FKC-SG) and define the corresponding state and weight updates to sample from pSGt,β(x). This corollary follows from FKC - Geometric Average defined in [21].

###### FKC-SG

Consider two diffusion models qont (x), qofft (x) and inverse temperature β < 0. The weighted SDE corresponding to the repulsion of marginals pSGt,β(x) ∝ qont (x)(1−β)qofft (x)β is

dxt = gt2((1 − β)∇ log qont (xt) + β∇ log qofft (xt))dt − ft(xt)dt + gtdWt , (S37) dwt =

gt2 2

2

β(β − 1) ∇ log qont (xt) − ∇ log qofft (xt)

dt (S38)

In Algorithm 10, we show how this is implemented in our sampling framework. Importantly, since our model does not enforce invariance to rotations and translations, one cannot naively combine the scores from the two distributions since they may not necessarily have the same frame of reference. Thus, we first compute the translation and rotation matrices qofft (xt) to the space of qont (xt) using Kabsch alignment before doing guidance or computing weights. Furthermore, we only do steering on the protein state and use the ligand state for conditioning. In Algorithm 10, we denote operations restricted to the protein or ligand components using xprotstruct and xligstruct, respectively. Similarly, operations corresponding to the on-target and off-target states are denoted by xon and xoff.

- Algorithm 10 SampleDiffusion with FKC-SG

Require: {f∗}, Structure Noise Schedule [c0, c1, . . . , cT], Sequence Time Schedule [r0,r1,...,rT], γ0 = 0.8, γmin = 1.0, noise scale λ = 0.1, step scale η = 1.5, guidance_on_sequence = true, inverse temperature β = −0.5, resample = true, number of particles K = 4

- 1: {xkstruct,l,on}kK=1, xkstruct,l ∼ N (0, c20I3) ∀l ∈ {1, . . . , L} xkstruct,l ∈ R3
- 2: xoffstruct = xonstruct.clone()
- 3: {xkseq,l }2k=K1, xkseq,l = m ∀l ∈ {1, . . . , L}
- 4: {w}kK=1 = 1/K
- 5: R = I3
- 6: t = 0
- 7: for each (cτ,rτ) ∈ zip([c1, . . . , cT], [r1,...,rT]) do
- 8: γ = γ0 if cτ > γmin else 0
- 9: tˆ = cτ−1(γ + 1)
- 10: {ξl,k}kK=1, ξl = λ tˆ2 − c2τ−1 · N (0,I3)∀l ∈ {1, . . . , L}

- 11: xonstruct,noisy = xonstruct + ξ
- 12: xprot,offstruct,noisy ← APPLYTRANSFORM(xprot,onstruct,noisy, R, t) ▷ rotate on-prot struct to off-prot space
- 13: {xstruct,denoised}, {xseq,logits} = DENOISER({xstruct,noisy}, {xseq}, tˆ, rτ, {f∗})
- 14: dt = cτ − tˆ # update protein state
- 15: (R, t) ← KABSCH(xprot,onstruct,noisy, xprot,offstruct,noisy) ▷ maps on-prot to off-prot space
- 16: (R−1, t−1) ← INVERTTRANSFORM(R, t) ▷ maps off-prot to on-prot space
- 17: x˜prot,offstruct,denoised ← APPLYTRANSFORM(xprot,offstruct,denoised, R−1, t−1)
- 18: δprot, on ← (xprot,struct,noisyon − xprot,struct,denoisedon )/tˆ
- 19: δprot,off ← (xprot,struct,noisyon − x˜prot,offstruct,denoised)/tˆ
- 20: δprot,on ← (1 − β)δprot, on + βδprot, off ▷ guidance for on-target (repel off-target)
- 21: xprot,onstruct ← xprot,onstruct,noisy + η · dt · δprot,on # update ligand state
- 22: δlig ← (xligstruct,noisy − xligstruct,denoised)/tˆ
- 23: xligstruct ← xligstruct,noisy + η · dt · δlig # calculate scores for on- and off-target protein states
- 24: scoreprot,on ← (xprot,struct,noisyon − xprot,struct,denoisedon )/tˆ2
- 25: scoreprot,off ← (xprot,struct,noisyon − x˜prot,offstruct,denoised)/tˆ2 # optionally adjust sequence logits
- 26: if guidance_on_sequence then
- 27: xonseq,logits = (1 − β)xonseq,logits + (1 − β)xoffseq,logits
- 28: end if
- 29: xonseq = PATHPLANNINGSTEP(xonseq,logits, bτ) # compute weights & resample particles in batch
- 30: dw = −β(β − 1) · scoreprot,on − scoreprot,off

2

· tˆ

- 31: w ← w + dw · dt

- 32: if resample then
- 33: wk = wk/ ∑iK wi ▷ normalize weights
- 34: ℓ ∼ Cat(K | w) ▷ sample index
- 35: xkstruct,prot,on ← xℓstruct,prot,on
- 36: xkstruct,lig,on ← xℓstruct,lig,on
- 37: xkstruct,lig,off ← xℓstruct,lig,off
- 38: xkseq ← xℓseq
- 39: wk ← 1/K ▷ re-initialize weights
- 40: end if
- 41: xseq ← CONCAT([xonseq, xonseq])
- 42: end for
- 43: return xonstruct, xonseq

###### A.7.2 FKC-SG experimental details

We investigate the ability of FKC-SG to generate specific-binding proteins for three sets of on/off-targets. The first pair is amino acids valine (on-target) and proline (off-target); the second is steroids aldosterone (on-target) and cortisone (off-target); the third is biotin (on-target) and pyridoxal 5’-phosphate (PLP, off-target). For each, we sweep over inverse temperatures β ∈ {−0.5, −1.0, −2.0}. Previous work also found that resampling at every step reduced sample diversity significantly due to variance of the weights[21], and so we also searched for the best time point to stop resampling (resample = false when τ ∈ {0.4,0.6,0.8,1.0}). Finally, we tried particle batch sizes of K ∈ {2,4,8}. We found that β = −0.5 worked best, as well as setting resample = false when τ = 0.6. For the valine/proline pair, we used 4 particles and for the other two pairs, we found 8 particles were better.

###### A.7.3 FKC-SG evaluation

For each ligand pair, we generated K · 200 protein structures and sequences of length 150 using DISCO with FKC-SG or DISCO on its own. We refolded all K · 200 sequences with the on-target using Chai-1 and aligned them with the structure generated by DISCO. If the complex was designable (i.e. RMSDs of both the ligand centroids and proteins were less than 2 Å), we kept the structure. We then folded each off-target with the sequence and computed the RMSD between the ligand centroids after aligning the complex with the protein folded with the on-target. For each batch of K particles, we kept the sample with the largest ligand centroid RMSD (max reward) for further evaluation (max. 200 samples for each method if each batch had one designable complex). For each protein sequence, we folded it three times with Chai-1 for each of the on/off-targets. For each sequence, we measured the pairwise ligand centroid RMSD between all remaining Chai-1 folds (9 comparisons) and kept the minimum RMSD. A sequence was considered to show separation between the on- and off- targets if this minimum RMSD exceeded a fixed threshold (6 Å). We report the number of such specific sequences for each condition.

###### A.7.4 Feynman-Kac Correctors - Multimodal

We observe that the reward tilting framework of unimodal Feynman-Kac correctors in continuous and discrete spaces can be extended to the multimodal case. To this end, we derive in Theorem A.7.4 expressions for the augmented drift and rate matrices as well as the weight term

- h. We consider the one-dimensional discrete case as the extension to multi-dimensional discrete objects is trivial.

FKC-MM Reward-Tilting Identify pτ(xτ,i) := pt(τ),s(τ)(xt(τ),i). For the reward tilted marginals qτ(x,i) ∝

pτ(x,i) exp(βτR(x,i)) the following equations holds

gτ2

gτ2 2

∂qτ(xt,i) ∂τ

= t˙(τ) − ∇, qτ(xt,i)(−fτ(x) + gτ2∇ log pτ(x) + βτ

∆qτ(x,i)

###### 2 ∇R(x,i) +

###### +s˙(τ) ∑

AτR(i, j, x)qτ(x,i) − AτR(y,i, x)qτ(x, j)

j̸=i

+ qτ(x,i) hτ(x,i) − Ey,j∼pτ(y,j)[hτ(y, j)] (S39)

gτ2

hτ(x,i) = t˙(τ) βτ∇xR(x,i),

2 ∇x log pτ(x,i) − fτ(x)

∂βτ ∂τ

###### +s˙(τ)∑

AτR(i, j, x) − Aτ(i, j, x) +

R(x,i) (S40)

j̸=i

exp(βτR(x, j)) exp(βτR(x,i))

AτR(i, j) = Aτ(i, j, x)

(S41)

This proof follows closely the steps for unimodal reward-tilted FKC in continuous and discrete space, specifically Proposition D.6 of [21] and Theorem 3.5 of [22]. We begin by denoting a time schedule for each modality t(τ) and s(τ) with t corresponding to the continuous modality and s to the discrete modality. We then define

pτ(x,i) := pt(τ),s(τ)(x,i) (S42) qτ(x,i) :=

1 Zτ

pτ(x,i)exp(βτR(x,i)),Zτ = ∑

pτ(x,i) exp(βτR(x,i))dx (S43)

i

Breaking apart the time derivative

∂ ∂τ

∂ ∂t

∂ ∂s

pτ(x,i) = t˙(τ)

pt,s(x,i) + s˙(τ)

pt,s(x,i) (S44) From Lemma 3 of [12] we have

gt2 2

∂ ∂t

∆pt,s(x,i) (S45)

###### pt,s(x,i) = −∇ · (vt(x)pt,s(x,i)) +

∂ ∂s

###### pt,s(x,i) = ∑

[As(x, j,i)pt,s(x, j) − As(x,i, j)pt,s(x,i)] (S46)

j̸=i

where in the latter we define As(x,i,i) to be the negative sum over the off-diagonal. Consider the time derivative of the change in log density

∂ ∂τ

∂ ∂τ

log qτ(x,i) =

(log pτ(x,i) + βτR(x,i) − log Zτ)

∂ ∂t

∂ ∂s

∂βτ ∂τ

∂ ∂τ

###### = t˙(τ)

log pt,s(x,i) + s˙(τ)

log pt,s(x,i) +

R(x,i) −

log Zτ

(S47) We break up the expression by modality, considering the continuous modality first

gt2 2

gt2 2 ∥∇ log pt,s(x,i)∥2

∂ ∂t

∆ log pt,s(x,i) +

log pt,s(x,i) = −⟨∇, vt(x)⟩ − ⟨∇ log pt,s(x,i), vt(x)⟩ +

gt2 2

gt2 2 ∥∇ log qt,s(x,i)∥2

∆ log qt,s(x,i) +

= −⟨∇, vt(x)⟩ − ⟨∇ log qt,s(x,i), vt(x)⟩ +

gt2 2

gt2 2

+ βτ∇R(x,i), vt(x) − gt2∇ log pt,s(x,i) −

∆R(x,i)

βτ∇R(x,i) − βτ

We will eventually evaluate the change in marginal density ∂pτ∂τ(x,i) and to do so we will

use the term qt,s(x,i)∂∂t log pt,s(x,i) which we derive now. Making use of the identities p(x)⟨∇, f(x)⟩ = ⟨∇, f(x)p(x)⟩ − ⟨f(x), ∇p(x)⟩ for vector field f and scalar field p, that

∆p(x) = p(x)(∆ log p(x) + ∥ log p(x)∥2), and that ∂p∂t(tx) = pt(x)∂∂t log pt(x) we can evaluate the time derivative of the non-log density

∂ ∂t

qt,s(x,i)

log pt,s(x,i) = −qt,s(x,i)⟨∇, vt(x)⟩ − qt,s(x,i)⟨∇ log qt,s(x,i), vt(x)⟩

gt2 2

∆ log qt,s(x,i) + ∥∇ log qt,s(x,i)∥2

+ qt,s(x,i)

gt2 2

+ qt,s(x,i) βτ∇R(x,i), vt(x) − gt2∇ log pt,s(x,i) −

βτ∇R(x,i)

gt2 2

∆R(x,i)

− qt,s(x,i)βτ

= −⟨∇, vt(x)qt,s(x,i)⟩ + ⟨∇qt,s(x,i), vt(x)⟩ − ⟨∇qt,s(x,i), vt(x)⟩

gt2 2

∆qt,s(x,i) + qt,s(x,i)hcontt,s (x,i)

+

σt2 2

∆qt,s(x,i) + qt,s(x,i)hcont_pret,s (x,i)

= −⟨∇, vt(x)qt,s(x,i)⟩ +

To help with performance during inference, we can add the reward gradient to the drift. To do so we recall that for a scalar field p(x) and vector fields f(x) and h(x) we have that ⟨∇, p(x)f(x)⟩ + ⟨∇, p(x)g(x)⟩ = ⟨∇, p(x)(f(x) + h(x))⟩

gt2 2

∂ ∂t

∆qt,s(x,i) + qt,s(x,i)hcont_pret,s (x,i) − ⟨∇, qt,s(x,i)a∇R(x,i)⟩ + ⟨∇, qt,s(x,i)a∇R(x,i)⟩

qt,s(x,i)

log pt,s(x,i) = −⟨∇, vt(x)qt,s(x,i)⟩ +

gt2 2

∆qt,s(x,i) + qt,s(x,i)hcontt,s (x,i)

= −⟨∇, qt,s(x,i)(vt(x) + a∇R(x))⟩ +

where

gt2 2

hcontt,s (x,i) = a∆R(x,i) + a⟨∇ log qt,s(x,i), ∇R(x,i)⟩ − βτ

∆R(x,i)

gt2 2

+ βτ∇R(x,i), vt(x) − gt2∇ log pt,s(x,i) −

βτ∇R(x,i)

Plugging in vt(x) = −ft(x) + gt2∇ log pt,s(x,i) and a = βτgt2/2 we have

gt2 2 ∇R(x,i))

∂ ∂t

log pt,s(x,i) = − ∇, qt,s(x,i)(−ft(x) + gt2∇ log pt,s(x,i) + βτ

qt,s(x,i)

gt2 2

∆qt,s(x,i) + qt,s(x,i)hcontt,s (x,i)

+

gt2 2

gt2 2

gt2 2 ∇ log qt,s(x,i)

hcontt,s (x,i) = βτ

∆R(x,i) − βτ

∆R(x,i) + βτ∇R(x,i),

gt2 2

βτ∇R(x,i)

+ βτ∇R(x,i), −ft(x) −

gt2 2 ∇ log(qt,s(x,i)/ exp(βτR(x,i))

= βτ∇R(x,i), −ft(x) +

gt2 2 ∇ log pt,s(x,i)

= βτ∇R(x,i), −ft(x) +

We now consider the discrete modality

pt,s(x, j) pt,s(x,i) − As(x,i, j)

∂ ∂s

logpt,s(x,i) = ∑

As(x, j,i)

j̸=i

exp(βτR(x, j)) exp(βτR(x,i))

qt,s(x, j) qt,s(x,i) − As(x,i, j)

###### = ∑

As(x, j,i)

j̸=i

:=AsR(x,j,i)

qt,s(x, j) qt,s(x,i) − AsR(x,i,j) +∑

###### = ∑

AsR(x, j,i)

AsR(x,i, j) − As(x,i, j)

j̸=i

j̸=i

Again, we consider the weighted term

∂ ∂s

logpt,s(x,i) = ∑

AsR(x, j,i)qt,s(x, j) − AsR(x,i, j)qt,s(x,i)

qt,s(x,i)

j̸=i

+qt,s(x,i)∑

AsR(x,i, j) − As(x,i, j)

j̸=i

:=gtdiscrete,s (x,i)

###### Finally we have

gt2 2 ∇R(x,i))

∂qτ(x,i) ∂τ

= −t˙(τ) ∇, qt,s(x,i)(−ft(x) + gt2∇ log pt,s(x,i) + βτ

gt2 2

+ t˙(τ)

∆qt,s(x,i) +s˙(τ)∑

AsR(x, j,i)qt,s(x, j) − AsR(x,i, j)qt,s(x,i)

j̸=i

+ qt,s(x,i)(ht,s(x,i) − Ex,i∼qt,s(x,i)[ht,s(x,i)]) ht,s(x,i) = t˙(τ)hcontt,s (x,i) + s˙(τ)hdiscretet,s (x,i) +

∂βτ ∂τ

R(x,i)

gt2 2 ∇ log pt,s(x,i)

= t˙(τ) βτ∇R(x,i), −ft(x) +

∂βτ ∂τ

+s˙(τ)∑

AsR(x,i, j) − As(x,i, j) +

R(x,i)

j̸=i

###### A.7.5 FKC-MM reward functions

A central motivation for joint sequence–structure guidance is that many biophysically meaningful objectives are inherently multimodal: they depend simultaneously on residue identity (sequence) and inter-residue geometry (structure), and thus multimodal steering would be favorable compared to sequence-only or structure-only methods that are available to current two-stage design methods. We designed two such rewards—disulfide bond formation and cation-π interactions—that exemplify this coupling.

Both rewards share the same form. Given residue-type masks ϕi, ψi ∈ {0,1} derived from the sequence, the soft contact count between eligible pairs is

cij = σ

d0 − ∥ci − cj∥ τ · mtypeij · 1[|i − j| ≥ ∆min] · 1[i < j], (S48)

where σ(·) is the sigmoid function, ci are reconstructed Cβ coordinates, d0 is a distance threshold, and τ controls sigmoid sharpness. The per-sample reward aggregates these contacts with a log-sum form to enable better gradient flow:

R =

L

###### ∑

log 1 + ∑j cij . (S49)

i=1

Disulfide bond reward. Cβ positions are reconstructed from the backbone N, Cα, C atoms of each residue using the ideal tetrahedral placement. Disulfide Cβ–Cβ distances peak at ∼3.8 Å in the PDB. We set d0 = 4.5 Å, τ = 0.3 Å, ∆min = 3, and mSSij = 1[si=Cys] · 1[sj=Cys].

Cation-π reward. Cation-π Cβ–Cβ distances span ∼6–8 Å. We define C+ = {Arg, Lys} and Aπ = {Phe, Tyr, Trp, His} (His treated as aromatic only), with mcat-ij π = 1[si∈C+]·1[sj∈Aπ]+ 1[si∈Aπ] · 1[sj∈C+], d0 = 8.0 Å, τ = 0.5 Å, ∆min = 4.

###### A.8 In-silico evaluation

###### A.8.1 Unconditional protein generation

We sampled 100 structures at each of four target lengths (70, 100, 200, and 300 residues) using DISCO with noisy guidance, as discussed above. Each generated structure was paired with the sequence produced jointly by the model. A design was classified as co-designable if the generated sequence from the model can fold into the generated structure, where the threshold is root-mean-square deviation (RMSD) of < 2.0 Å of the backbone, as folded by ESMFold [19]. The mean predicted local distance difference test (pLDDT) score was extracted from ESMFold and averaged over all residues for each design. We note that novelty/diversity metrics are only valid for co-designable samples (e.g. random, unfoldable sequences will have high novelty/diversity, but they are not useful), but other analyses (e.g. secondary structure, radius of gyration, etc.) are conducted for all samples as they reflect how well the model captures protein statistics.

To compare generated sequence properties against inverse-folding sequences (Figure 3, Figure S6), ProteinMPNN [44] was applied to each generated backbone, producing 8 candidate sequences per structure. Each was refolded with ESMFold, and the sequence yielding the lowest backbone RMSD was retained for comparison.

Secondary structure was assigned using the tool DSSP. Eight-state assignments were reduced to three states: helix (H, G, I → H), strand (E, B → E), and coil (all others → C).

The relative contact order is defined by

1

L· N ∑

RCO =

|i − j|, (S50)

contacts

where L is the chain length and N is the total number of qualifying contacts. A long-range contact between residues i and j was recorded if

dCα(i, j) < 8.0 Å and |i − j| ≥ 12. (S51)

###### A.8.2 Sequence diversity and novelty

Sequence diversity was quantified by clustering co-designable sequences with MMseqs2 [45] easy-cluster:

|mmseqs easy-cluster input.fasta clust tmp \<br><br>--min-seq-id 0.3 -c 0.8 --cov-mode 0|
|---|

The number of clusters divided by the total number of samples (400) served as the diversity metric. Sequence novelty was assessed by searching against UniRef50 at maximum sensitivity:

|mmseqs easy-search input.fasta uniref50 result.m8 tmp --gpu 1|
|---|

Sequence novelty was computed as the average of 1 − seq. identity.

###### A.8.3 Structural diversity and novelty

Structural diversity was measured by clustering co-designable backbones with Foldseek [46] easy-cluster using TM-score–based alignment:

|foldseek easy-cluster input_dir clust tmp \<br><br>--alignment-type 1 --cov-mode 0 \<br><br>--min-seq-id 0 --tmscore-threshold 0.5|
|---|

The number of clusters divided by the total number of samples (400) served as the diversity metric.

Structural novelty was evaluated by searching each backbone against the PDB:

|foldseek easy-search input_dir pdb result.m8 tmp \<br><br>--alignment-type 1 --tmscore-threshold 0.0|
|---|

Structure novelty was computed as the average of 1 − TM-score.

###### A.8.4 Conditional evaluation of small-molecule ligands

We assembled a curated set of 179 chemically diverse molecules (cofactors, metals, druglike compounds, fluorophores, saccharides, pollutants, metabolites, etc.), including multiple different molecules. We call this benchmark STUDIO-179.

For each molecule, we optimized the geometry of RDKit ETKDG conformer with the semiempirical method GFN2-xTB[47]. The RDKit-derived bond information and the optimized geometry were used as the input bond/reference conformer information to the model. 50 protein–ligand complexes were generated at each of three target lengths (150, 200, and 250 residues) using DISCO with the inference settings specified in Table S3.

Co-designability was assessed by refolding each complex with Chai-1 [20]. A design was considered co-designable if both:

RMSDbackbone(Xdesign, XChai) < 2.0 Å, (S52) ∥r¯designℓ − r¯Chaiℓ ∥ < 2.0 Å ∀ ligand ℓ, (S53)

where r¯ℓ denotes the centroid of ligand ℓ. For multi-ligand systems, Eq. (S53) was required to hold for every ligand independently.

A protein–ligand steric clash was flagged if any protein heavy atom in the backbone structure

i and ligand heavy atom j satisfied dij < rivdW + rvdWj − 0.5 Å, where rvdW denotes the van der Waals radius and the 0.5 Å tolerance accounts for minor overlap in experimental structures.

For logP and pocket hydrophobicity analysis, the binding site is defined as the set of protein residues whose backbone atoms lie within 10 Å of any heavy atom of the cofactor/ligand, retaining up to the 10 closest residues sorted by minimum protein–ligand distance.

###### A.8.5 Conditional evaluation: nucleic acid binding

We selected DNA/RNA sequences that were released in the PDB after our training cut-off date. Each DNA chain was cropped to a maximum length of 10. Designed protein–nucleic acid complexes were refolded with Chai-1. We followed the experimental design of the RFDiffusion3 DNA benchmark and for all DNA and RNA targets generated 100 samples with lengths sampled uniformly between lengths 50 to 80. Co-designability was assessed by first aligning nucleotide reference atoms between the design and the Chai-1 prediction, using phosphorus (P) atoms as the primary anchor.

A design was considered co-designable if

RMSDbackbone(global-aligned)(Xdesign, XChai) < 2.0 Å, (S54)

ensuring that the protein refolds correctly and maintains the designed spatial relationship with its nucleic acid partner.

###### A.8.6 Baseline methods

For unconditional monomer generation, we benchmarked against La-Proteina[48], ProteinGenerator[49], ESM3[50], RFdiffusion3[10], and BoltzGen[11]. For conditional generation (small-molecule and nucleic acid binding), only RFdiffusion3 and BoltzGen were included, as the remaining baselines do not natively support ligand- or nucleic acid–conditioned generation. We additionally included RFDiffusion All-Atom [51] as a baseline for small molecule conditioning. For each designed backbone for RFDiffusion All-Atom, one sequence was generated using LigandMPNN[15], and was selected for downstream evaluation. All baselines were run with publicly released code and recommended default parameters; the same downstream evaluation pipeline was applied identically. For RFDiffusion3, BoltzGen, and RFDiffusion All-Atom to allow for comparison to DISCO we require each to co-fold the ligand or nucleic acid positions along with the protein backbone.

###### A.8.7 Binding motif novelty

To assess whether DISCO generates novel binding motifs, we searched designed active sites against known structures using Folddisco[24] with default parameters. For each co-designable protein–ligand complex, the binding motif was defined as the k protein residues closest to the ligand (by side chain–ligand heavy-atom distance of refolded structure with Chai-1). Searches were performed on Chai-1 refolded structures (which provide side-chain coordinates required by Folddisco) against the afdb_swissprot_v4 Foldcomp database. Folddisco calculations are only carried out for co-designable structures, to avoid searching for ligands whose binding site cannot be agreed upon by Chai-1 and DISCO.

- A binding motif was classified as novel if either: (i) Folddisco returned no match, or (ii)

Folddisco returned a full match to all k query residues but the matched-residue RMSD exceeded 3.0 Å.

###### A.8.8 Binding motif diversity

We now turn to assess the diversity of the generated binding motifs. The binding motif’s definition is the same as Section A.8.7. Given two K-residue motifs (P, τP) and (Q, τQ) from co-designable proteins sharing the same ligand type, we report the aggregated per-sample metric of nearest-neighbour diversity NNi = minj̸=i d(i, j) for a metric d detailed below. Furthermore, we also report

(i) Cα-RMSD. Both coordinate sets are centred and then iteratively aligned using the Kabsch algorithm [52] with the Hungarian (linear-sum-assignment) optimal matching. At each of n = 3 iterations, the current assignment is solved on the squared-distance cost matrix, and the optimal rotation is recomputed on the matched pairs. The final RMSD is dRMSD =

1 K ∑kK=1 ∥pk − qσ(k)∥2, where σ is the terminal assignment.

(ii) Chem-cost. A combined spatial–chemical distance is used. Each residue is assigned a chemical class (hydrophobic, polar, positive, negative, or special). A chemical distance matrix Dijchem is defined as 0 for identical residue types, 0.5 for same class, and 1.0 otherwise. The

Hungarian assignment is solved on Cij = α ∥pi − qj∥2 + β Dijchem (α = 1, β = 5), alternated with Kabsch alignment for n = 3 iterations. The reported score is the mean per-residue combined

distance under the final matching: dchem = K1 ∑k α ∥pk − qσ(k)∥ + β Dkchem,σ(k) .

(iii) 1 − lDDT. After the spatial (RMSD) assignment σ is established, the internal distance matrices DijP = ∥pi − pj∥ and DijQ = ∥qσ(i) − qσ(j)∥ are compared. For every upper-triangular pair (i, j), the absolute difference |DijP − DijQ| is tested against thresholds {0.5,1.0, 2.0,4.0} Å. The local distance difference test score is lDDT = (preserved pairs)/(|pairs| × |thresholds|), and we report 1 − lDDT so that larger values indicate greater diversity.

- (iv) Frobenius distance. This rotation-invariant metric compares the internal Cα distance

matrices directly. An optimal residue permutation is found by iteratively solving a Hungarian assignment on the row-permuted distance-matrix cost (n = 3 iterations). The Frobenius distance

is dF = (K1

2) ∑i<j(DijP − DπQ(i),π(j))2.

- (v) Motif cluster diversity. With all pairwise Cα-RMSD values between motifs using iterative

Hungarian matching with Kabsch alignment, we perform complete-linkage agglomerative clustering with a distance threshold of t = 2.0Å. The cluster ratio is defined as r = nclusters/ndesigns, where r ≈ 1 indicates that nearly every design adopts a structurally distinct active-site configuration, while r ≈ 0 indicates that designs collapse onto a small number of shared motif geometries.

###### A.9 Density Functional Theory calculations

Density functional theory (DFT) calculations were performed with ORCA 6.0.1 using the hybrid B3LYP functional in combination with the def2-TZVP basis set [53]. Dispersion effects were included with Grimme’s D3 correction and Becke–Johnson damping (D3BJ).Unless otherwise noted, all species were treated as triplet-state open-shell systems and were therefore computed using unrestricted DFT. Solvent effects were modeled with the SMD implicit solvation model using ethyl acetate as the solvent, chosen to approximate the dielectric environment of the enzyme active site.

For carbene insertions, we optimized the pre-reactive intermediate corresponding to the approach of the carbene precursor to the heme iron center. With heme, multiple carbene insertion intermediates were computed. The heme cofactor was modeled with a 5-methylimidazole axial ligand; this ligand was included in the electronic-structure model but removed in the subsequent design stage. All reaction intermediates were geometry optimized and characterized by vibrational frequency analysis. For the calculation of energy barriers, transition state geometries were located using the NEB-TS method and frequency analysis indicates that they possess a single imaginary frequency along the reaction coordinate.

###### A.10 Design filtering for enzymes

For every enzyme design campaign, DISCO generated on the order of 104 sequence–structure pairs conditioned on the target intermediate structure (including reference conformer and bonds), where DISCO generates the structure for the reactive intermediate. These candidates were subjected to a multi-stage filtering pipeline combining orthogonal structure-prediction oracles, physicochemical filters, and diversity-aware selection.

Each candidate was first refolded with Chai-1, and structures satisfying the co-designability

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Figure S3. DFT-calculated reaction mechanism for the carbene-insertion reactions using Feporphyrin as the cofactor. All design campaigns used intermediate 1 reoptimized with heme b.

criterion (Eq. (S53)) were advanced to a second round of structure prediction with AlphaFold 3. In both rounds, the protein was co-folded with the reactive intermediate to assess activesite geometry. AlphaFold 3 predictions were additionally performed with only the cofactor (omitting the intermediate). A design was retained only if it was co-designable under both Chai-1 and AlphaFold 3.

Dual-oracle passing designs were subjected to the following filters:

- 1. Contacts. The number of protein heavy atoms within 4.0 Å of any intermediate heavy atom was required to be ≥ 5.
- 2. Active-site burial. Three complementary metrics assessed ligand burial. First, the SASA burial fraction,

Acomplex Afree

fburial = 1 −

, (S55)

was required to exceed 0.5. Second, we computed a solid-angle enclosure score. From each of up to 15 ligand heavy-atom origins (selected by farthest-point sampling), n = 500 rays were cast uniformly over S2 via a Fibonacci lattice. A ray from origin o was considered blocked by protein atom j if

rvdWj + rprobe ∥pj − o∥

cos∠(rˆ, dˆj) ≥ cos arctan

, (S56)

where rˆ is the ray direction, dˆ j the unit vector from o to protein atom pj, rvdWj its van der Waals radius, and rprobe = 1.4 Å. The enclosure from each origin is the fraction of blocked rays; the worst-case (minimum) across all origins was required to exceed 0.5. Third, the maximum exposure angle was estimated from the centroid origin by clustering unblocked rays via greedy single-linkage and reporting the bounding-cone half-angle of the largest cluster; this was required to be ≤ 65◦.

- 3. Metal coordination. For metalloenzyme campaigns, proper coordination geometry of catalytic metal ions was verified (correct number and identity of coordinating residues, bond lengths within ±0.5 Å of ideal values).
- 4. Surface hydrophobicity. The fraction of solvent-exposed hydrophobic residues (Ala, Val, Ile, Leu, Met, Phe, Trp, Pro) was required to be <50% to reduce aggregation propensity.
- 5. Net charge. The formal net charge at pH 7 was checked to be within ±15. Designs were then ranked within each cluster by a simple composite confidence score:

S = 0.5 · ipTM + 0.25 · pTM − 0.5 · PAEchain + 1.0 · ipTMchain, (S57)

where ipTM and pTM are the AlphaFold 3 global interface and predicted TM-scores, PAEchain is the pairwise aligned error between the protein and ligand chains, and ipTMchain is the pairwise interface pTM between the ligand and protein chains. In addition to the composite score, hard cutoffs were enforced: ipTM ≥ 0.7, pTM ≥ 0.7, and chain-pair ipTM ≥ 0.5.

Passing designs were clustered at both the sequence level (MMseqs2, 30% identity) and the structure level (Foldseek, TM-score 0.5). Within each sequence cluster, at most 2 sequences were retained. Within each structure cluster, at most 8 structures were retained.

###### A.11 Additional in-silico results

###### A.11.1 Additional unconditional generation results

We note that as co-designability is very high for unconditional generation results, sample analysis of generated proteins (e.g. amino acid distribution, secondary structure distribution, radius of gyration, relative contact order, etc.) show no qualitative changes when filtered for co-designability.

Table S4. Unconditional generation metrics. Bold indicates best per column; underline indicates second best. Co-designable structural and sequence diversity are the most informative metrics because they reflect both sample quality and the absence of mode collapse, whereas one should not pursue the maximization of co-designability alone, as folding models cannot refold every PDB protein within 2 Å RMSD. Note that only DISCO, BoltzGen and RFDiffusion3 can condition on other biomolecules.

Struct. Diversity

Seq. Diversity

Struct. Novelty

Seq. Novelty

Method Co-designable

pLDDT

ESM3 0.145 0.040 0.113 0.697 0.806 0.533 ProteinGenerator 0.355 0.118 0.273 0.173 0.998 0.779 La-Proteína 0.895 0.293 0.790 0.173 0.879 0.917 BoltzGen 0.798 0.065 0.310 0.132 0.865 0.882 RFDiffusion3 0.460 0.135 0.450 0.166 0.926 0.766

DISCO (ours) 0.878 0.343 0.855 0.211 0.961 0.879

###### Co-Designability vs Steps

###### Structural Diversity vs Steps

| |
|---|

| |
|---|

80

NStructuralClusters

| |
|---|

100

Co-Designable(%)

| |
|---|

70

| |
|---|

| |
|---|

| |
|---|

60

80

50

60

| |
|---|

40

- 1 cycle

- 2 cycles

- 3 cycles

- 4 cycles

- 1 cycle

- 2 cycles

- 3 cycles

- 4 cycles

30

40

| |
|---|

20

20

10

| |
|---|
| |

| |
|---|

| |
|---|

0

25 50 75 100 125 150 175 200

25 50 75 100 125 150 175 200

Number of Steps

Number of Steps

Figure S4. Co-designability and number of structure clusters for different numbers of steps and recycles. Performed on the unconditional task with the settings of Table S2 and noisy guidance disabled.

100

89.7%

88.5% 88.9%

82.7%

80

Co-designable(%)

60

40

20

0

70 100 200 300

Sequence length

40

37.4%

35

%Co-designableClusters

31.2%

30

25

20

15.3%

15

10

8.2%

5

0

70 100 200 300

Sequence length

###### Figure S5. Distribution of unconditional generation co-designability and the proportion of codesignable clusters across protein lengths.

[Figure 114]

###### Figure S6. Amino acid distribution for unconditional generation using our sequence–structure model. Shown are sequences designed by our model, sequences obtained by inverse folding our generated structures with ProteinMPNN, and sequences from 50,000 randomly sampled PDB structures from the training data.

[Figure 115]

###### Figure S7. Distributions of long-range contact types in unconditionally generated samples. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

[Figure 116]

###### Figure S8. Distributions of radius of gyration in unconditionally generated samples. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

89.7%

80

Percentageofchains(%)

60

40

20

3.7% 2.5% 2.6%

0.9% 0.3% 0.1% 0.0% 0.0% 0.0% 0.0%

0

0 1 2 3 4 5 6 7 8 9 10

Number of disulfide bonds

- Figure S9. Distribution of the number of disulfide bonds present in each protein chain of the PDB training data, where the length of the chain is less than or equal to 100.

###### A.11.2 Additional conditional generation results

We note that while co-designability is not as high as unconditional generation results, sample analyses of generated proteins generally showed no qualitative changes when filtered for co-designability. The only significant change can be seen in Fig. S13.

With the exception of motif novelty and diversity (and unless otherwise noted), all sample analysis are evaluated without co-designability filtering. Like unconditional generation analysis, general properties (e.g. conformer PoseBusters validity, steric clashes with ligands, logP of ligand vs pocket chemical features) reflect whether the model captures reasonable chemical fidelity; once this is satisfied and the protein is co-designable, only then does a motif’s novelty and diversity matter (e.g. one can easily sample random pockets that do not fold).

Natural

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

40

%Co-designable

30

clusters

20

10

0

TestosteroneKetocholesterolStrychnine[3Fe-4S]CoQ1GMSBPheaCortisolAldosteroneVitaminK3ABAActinoninCoelenterazinePhenolStaurosporineCoumarinShikimicacidHg SalicylicacidHCA-CaroteneCTP LuciferinNicotinamideGDP PaclitaxelcGMPAdenosineTDPAconitineTetrodotoxinNADPFAD GABASinefunginRiboflavinBH4PentahydroxyflavoneInsP3SirohemeVitaminCCiguatoxin1VitaminB9ThiamineLacticacidParomomycin

Non-natural

50

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

%Co-designable

40

clusters

30

20

10

0

PTMradicalNHCimidazoliumMirexDEHPPCB128TBTOTCAQuinacridonePFOSDIIDBrodifacoumThioxanthonePFOADimethoxythioxanthone4CzIPNCarbamazepineMabiqMes-Acr+CarfentanilNHCtriazoliumIsotonitazeneAcenaphthenequinoneDCDPSEosinYIndigoIsoindigoCiprofloxacinDarunavirAzaxanthineMuskxyleneNHCthiazoliumOQOCHAPSOD4Propargyl-dideazafolicacidNirmatrelvirCyclothiazidePPACK IohexolMPT IAI

DISCO

RFDiffusion3

Ligand

RFAA

BoltzGen

- Figure S10. Benchmark performance across the remaining set of STUDIO-179, ranging from natural to non-natural ligands. We observe that for all ligands DISCO generates the highest proportion of diverse, co-designable sequence-structure pairs.

Natural Non-natural Multi-ligand

100

| |
|---|

| |
|---|

80

Co-designable(%)

60

40

20

0

150 200 250

Sequence length

70

Natural Non-natural Multi-ligand

| |
|---|

| |
|---|

60

%Co-designableClusters

50

40

30

20

10

0

150 200 250

Sequence length

###### Figure S11. Distribution of average conditional generation co-designability and the proportion of co-designable clusters across protein lengths, where each scatter point corresponds to a ligand type.

0.5

0.5

DISCO PDB

DISCO PDB

| |
|---|

| |
|---|

0.4

0.4

Frequency

Frequency

0.3

0.3

0.2

0.2

0.1

0.1

0.0

0.0

Positive Negative

Positive Negative

Non-polar (aliphatic)

Non-polar (aromatic)

Polar (uncharged)

Non-polar (aliphatic)

Non-polar (aromatic)

Polar (uncharged)

0.12

DISCO PDB

| |
|---|

0.10

0.08

Frequency

0.06

0.04

0.02

0.00

GLU LYS LEU ALA VAL ILE GLY ASP ARG SER THR ASN PHE PRO TYR MET HIS GLN TRP CYS

- Figure S12. Distribution of amino acids for conditional generation using our sequence–structure model. Results are aggregated from all benchmark ligands. Shown are sequences designed by our model and sequences from 50,000 randomly sampled PDB structures from the training data. Top left shows all generated samples, top right shows only co-designable samples, and bottom shows all generated samples, but split into specific amino acids. Note that despite reduced co-designability, there is no significant change in the distribution of the types of amino acids.

[Figure 117]

[Figure 118]

- Figure S13. Secondary structure distribution for conditional generation using our sequence–structure model. Top shows results from all benchmark ligands without filtering, and bottom shows the effect of filtering by co-designability. We include a comparison against 50,000 randomly sampled PDB structures from the training data. Because current folding methods successfully refold only about 60% of the PDB training set, co-designability is likely a biased filter.

[Figure 119]

[Figure 120]

- Figure S14. Distributions of relative contact order for conditional generation using our sequence–structure model (top) and long-range contact types (bottom). Results are aggregated from all benchmark ligands without co-designability filtering. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

1.0

60

##### Surfacehydrophobicfraction

40

0.8

##### Netsurfacecharge

20

0.6

0

0.4

20

0.2

40

60

0.0

DISCO PDB

DISCO PDB

- Figure S15. Distribution of surface hydrophobicity and net charge for conditional generation using our sequence–structure model. Results are aggregated from all benchmark ligands without codesignability filtering. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

180

[Figure 121]

DISCO

120

60

(°)

0

60

120

180

180 120 60 0 60 120 180

(°)

180

[Figure 122]

PDB

120

60

(°)

0

60

120

180

180 120 60 0 60 120 180

(°)

- Figure S16. Ramachandran plot for conditional generation using our sequence–structure model. Results are aggregated from all benchmark ligands without co-designability filtering. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

0.25

DISCO PDB

| |
|---|

0.20

Density

0.15

0.10

0.05

0.00

0 5 10 15 20 25 30 35 40

Radius of Gyration (Å)

- Figure S17. Distributions of radius of gyration for conditional generation using our sequence–structure model. Results are aggregated from all benchmark ligands without filtering. We include a comparison against 50,000 randomly sampled PDB structures from the training data.

[Figure 123]

- Figure S18. PoseBusters validity breakdown for conditional generation results without codesignability filtering.

[Figure 124]

###### Figure S19. Average rate of any steric clash between the designed protein structure and the ligand(s), grouped by ligand type, without co-designability filtering.

[Figure 125]

###### Figure S20. Number of rotatable bonds compared to the median of conformer RMSD (with standard deviation), split by ligand type, without co-designability filtering.

[Figure 126]

- Figure S21. The strip plot shows the average binding site residue composition, split by ligand type, without co-designability filtering. Note that the charged-residue outliers correspond to metals that require charged coordination.

0.5

80

2logPvshydrophobicbindingsite(R)

0.4

No-clashrate(%)

75

0.3

70

0.2

65

60

0.1

DISCO RFDiffusion3 BoltzGen

DISCO RFDiffusion3 BoltzGen

Figure S22. Comparison of chemical fidelity indicators of binding motif and ligand interaction, averaged over all ligand types that all three models share (i.e. no multi-ligands). Top shows the correlation of pocket hydrophobicity with ligand lipophilicity. Bottom shows whether the designed proteins have any backbone clashes with the ligand. Together with the co-designability cluster plot shown in the main text, these results establish that DISCO yields realistic binding pockets that respond to ligand’s chemical and physical properties.

1.0

###### Motifclusterdiversity(clusters/samples)

0.8

0.6

0.4

0.2

Natural ligands

Non-natural ligands

Carbene

0.0

3 4 5 6 7 8 9 10

Number of closest residues (K)

Figure S23. Distribution of motif cluster diversity (averaged per ligand type), across K closest residues to the ligand.

- 1

- 2

- 3

- 4

- 5

Nearest-neighbourCA-RMSD(Å)

Natural ligands Non-natural ligands Carbene

Nearest-neighbourFrobeniusdistance(Å)

- 1

- 2

- 3

- 4

- 5

- 6

Natural ligands Non-natural ligands Carbene

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

Nearest-neighbourchem-cost

Natural ligands Non-natural ligands Carbene

0.7

Nearest-neighbour1lDDT

0.6

0.5

0.4

0.3

Natural ligands Non-natural ligands Carbene

###### Figure S24. Violin plot showing the full distribution of per-design nearest-neighbour diversity of the 5 closest residues to the ligand, each overlaid strip point shows a per-ligand median.

[Figure 127]

[Figure 128]

###### Figure S25. Additional conditional design examples. Left shows conditioning on luciferin, a natural luminescent molecule, in a hybrid helix and sheet topology. Right shows a substituted naphthalene diimide, a non-natural electron acceptor, in a helix bundle.

[Figure 129]

[Figure 130]

###### Figure S26. Additional generated conformer examples, compared to the reference conformer. Left shows bacteriopheophytin A, and right shows Nirmatrelvir.

[Figure 131]

[Figure 132]

###### Figure S27. An additional generated motif: the bonding geometries of a class I type II Cu center (2 His, 2 Cys, 1 Glu residues in tetrahedral coordination with Cu[2+]). This suggests the model can build informed residues to satisfy chemical constraints.

[Figure 133]

[Figure 134]

###### Figure S28. Additional generated motifs: two novel motifs (purple) aligned to the closest known motif (beige) in AlphaFoldDB. This indicates the model can build new residue motifs. Note that many generated motifs yield no match via Folddisco.

###### B In vitro methods and results

###### B.1 Materials and instrumentation

All chemicals were obtained from commercial suppliers and used without further purification. Racemic product standards 1b, 2b, 4b were prepared according to literature-known methods.[34, 54, 55]

M9-N buffer was used for enzymatic reactions and was prepared from a 5X stock solution of Na2HPO4·7H2O (64 g), KH2PO4 (15 g), and NaCl (2.5 g) dissolved in 1 L of double-distilled water (ddH2O). The working solution (1X) was prepared by dilution with ddH2O, followed by addition of CaCl2 (1 mL, 0.1 M) and MgSO4 (2 mL, 1.0 M). The pH was adjusted to 8.0.

The TAE buffer used in DNA gel electrophoresis was prepared from a 50X stock solution made by dissolving Tris free base (242 g), disodium EDTA (18.61 g), and glacial acetic acid (57.2 mL) in 1 L of ddH2O. The 1X TAE buffer was then prepared by diluting 40 mL of the 50X stock solution with 1960 mL of ddH2O.

PCR amplifications were performed using the Phusion® High-Fidelity PCR Kit (New England Biolabs) on an Eppendorf Mastercycler X50s.

Gibson Assembly reactions were carried out using an in-house solution containing isothermal master mix, T5 exonuclease, Phusion® DNA polymerase, and Taq DNA ligase (New England Biolabs).[56]

Gene fragments (without adapters) were ordered from Twist Bioscience, where the vendor was chosen for speed, cost and quality reasons.

T7 Express competent Escherichia coli (BL21) (New England Biolabs) were used for all transformations. All NMR spectra were obtained at the Caltech Liquid NMR Facility. NMR spectra were collected on a Bruker Prodigy 400 MHz instrument equipped with a cryoprobe operating at 400 MHz and 101 MHz for 1H and 13C, respectively. 1H spectra are referred to residual CDCl3 solvent signals referenced at δ 7.26 ppm. Data for 1H NMR are reported as follows: chemical shift (δ ppm), integration, multiplicity (s = singlet, d = doublet, t = triplet, q = quartet, p = pentad, sext = sextet, hept = heptet, m = multiplet, br s = broad singlet), and coupling constant (Hz).

High-performance liquid-chromatography mass spectroscopy (HPLC–MS) for standard reaction screening analysis was performed using an Agilent 1200 series instrument equipped with an Agilent Poroshell C18 column (Agilent Poroshell 120 EC-C18, 4.6 x 50 mm, 2.7 µm) with a Poroshell 120 guard column (4.6 x 5 mm, 2.7 µm). Water and acetonitrile (MeCN) with 0.1% (v/v) HPLC grade glacial acetic acid were used as eluents.

Normal phase chiral HPLC–UV data was collected with an Agilent 1260 Infinity HPLC equipped with a Daivel Chiralpak IC column (4.6 x 250 mm, 5 µm) and a Daivel Chiralpak IB column (4.6 x 250 mm, 5 µm) run in series (for products 2b and 3b) or a Daicel Chiralcel OD-H (product 1b). Hexanes and isopropanol were used as eluents.

Gas chromatography (GC) analyses were conducted on an Agilent Technologies 7820A GC system equipped with flame ionization detectors (FID). An Agilent J&W HP-5 column was used for achiral analyses, while an Agilent J&W CycloSil-B column was employed for chiral

separations. GC–MS analyses were carried out on an Agilent 7820A GC coupled to an Agilent 5977B MSD using an Agilent J&W HP-5MS UI column.

Chromatographic purification was accomplished by flash chromatography using a Biotage Isolera One instrument with Sfär High-Capacity Duo columns using AMD Silica Gel 60, 230–400 mesh. Ethyl acetate in hexanes was used as eluents.

Centrifugation was carried out in an Allegra 25R tabletop centrifuge equipped with a TS-5.1-500 swinging bucket rotor or a S5700 microplate swing bucket rotor.

###### B.2 Cloning and transformation in 96-well plates

The pET22b(+) backbone (Novagen) was PCR-linearized, digested with DpnI, gel-purified and confirmed to produce minimal background transformation. All genes were inserted between the NdeI and XhoI restriction sites. Ordered gene fragments were delivered in lyophilized form (Twist Bioscience) and resuspended in PCR-grade water to a concentration of 50 ng/µL. We note that for design sequences that did not begin with a methionine, an ATG codon was added as a start codon. A His-tag with the amino acid sequence LEHHHHHH was appended to the end of each sequence. Assembly reactions were performed in 96-well format using Gibson Assembly. Each reaction contained 1 µL insert DNA, 1 µL vector (100 ng/µL), and 5 µL Gibson Assembly mix, followed by incubation at 50 °C for 1 h. Transformation was performed by incubating 5 µL BL21(DE3) competent cells with the assembly mixture on ice for 30 min, followed by heat shock at 42 °C for 10 s. After recovery in LB medium (100 µL), 10 µL was used to inoculate 500 µL LB medium supplemented with 100 µg/mL carbenicillin (LBcarb), and cultures were grown overnight at 37 °C. Transformants were verified by in-house sequencing (LevSeq), showing ∼95% correct sequences.[57] The remaining ∼5% likely resulted from sequencing, assembly, or transformation errors. Variants were organized in 96-well plates, including: negative control: β-subunit of tryptophan synthase from Thermotoga maritima (TmTrpB, wells A1, B2, C3, D4) and positive control: evolved protoglobin from Aeropyrum pernix 5312[34] (wells E5 and F6). Glycerol stocks were prepared for long-term storage.

###### B.3 Protein expression and reaction screening in plates

Glycerol stocks were used to inoculate 500 µL LBcarb in 96-well plates, covered with a sterile, breathable film and grown at 37 °C with 220 rpm shaking. From stationary-phase cultures, 50 µL was transferred to 900 µL TBcarb and incubated for 2 h before induction with 50 µL of isopropyl β-D-1-thiogalactopyranoside (IPTG) (0.5 mM final) and 5-aminolevulinic acid (ALA) (1 mM final) in TBcarb. Cells were harvested by centrifugation (4,000×g, 10 min) and resuspended in 380 µL M9-N buffer (pH 8.0) supplemented with 10 g/L glucose by orbital shaking at 900 rpm. The plates were then transferred into a Coy anaerobic chamber (∼ 0 – 70 ppm O2) to set up the reactions anaerobically. Reactions were initiated by adding 20 µL of a substrate solution in MeCN ([substrate]final = 10 mM, [EDA]final = 15 mM, final MeCN concentration: 5%). Plates were sealed with adhesive aluminum foil and incubated overnight at room temperature with shaking (700 rpm).

###### Workup procedures (reaction dependent):

- For 1b, 3b and 4b: The plates were removed from the Coy chamber, and HPLC-grade MeCN (1200 µL) was added to each well. The resulting suspensions were mixed by shaking on an

orbital shaker at 500 rpm for 20 min at room temperature, and the plates were then centrifuged (5,000×g, 15 min, 4 °C) to remove protein and cell debris from the supernatant. The supernatant (200 µL/well) was transferred to a polypropylene 96-well microtiter plate (Agilent) and sealed with an Easy Pierce Heat Sealing Foil (Thermo Scientific™) and subjected to reversed-phase HPLC–MS analysis using methods details in Table S5, Table S6, and Table S7. The product MS peak was integrated using the built-in analysis.

- Table S5. LC-MS methods for analysis of compound 1b. Solvent A: H2O + 0.1% AcOH, solvent B: MeCN + 0.1% AcOH. Flow rate: 1.5 mL/min.

|Time (min)<br><br>|A (%)|B (%)|
|---|---|---|
|0.0 2.5<br><br>|50.0 50.0|50.0 50.0<br><br>|

- Table S6. LC-MS methods for analysis of compound 3b. Solvent A: H2O + 0.1% AcOH, solvent B: MeCN + 0.1% AcOH. Flow rate: 1.5 mL/min.

|Time (min)|A (%)<br><br>|B (%)|
|---|---|---|
|0.0 0.4 0.5 2.1 2.5 2.6 3.0<br><br>|85.0 85.0 10.0 5.0 5.0 85.0 85.0<br><br>|15.0 15.0 90.0 95.0 95.0 15.0 15.0|

- Table S7. LC-MS methods for analysis of compound 4b. Solvent A: H2O + 0.1% AcOH, solvent B: MeCN + 0.1% AcOH. Flow rate: 1 mL/min.

|Time (min)|A (%)|B (%)<br><br>|
|---|---|---|
|0.0 3.0|50.0 50.0<br><br>|50.0 50.0|

- For 2b: The plates were removed from the Coy chamber, and 400 µL of a 1 mM solution of 1,2,3-trimethoxybenzene dissolved in 1:1 (v/v) EtOAc:cyclohexane were added to each well. Plates were sealed using a 96-well sealing mat and then mixed by shaking on an orbital shaker (900 rpm) for 20 min. Plates were then centrifuged (5,000×g, 10 min, room temperature), and the organic layer (200 µL/well) was transferred to a 400 µL glass insert within a 2.0 mL screw-cap vial for GC–MS analysis. Analyses were performed on an Agilent J&W HP-5MS UI column using the following oven program: 120 °C (1 min), ramped at 35 °C/min to 250 °C, and then ramped at 75 °C/min to 300 °C (hold 0.2 min) for a total run time of approximately 11.2 min. The internal standard and product peaks were integrated using built-in MassHunter Quantitative Analysis software (Agilent).

###### B.4 Protein expression and whole-cell validation reactions

Cultures expressing enzyme variants with promising activity were streaked from the corresponding glycerol stocks onto LBcarb agar plates. A single colony was used to inoculate 5 mL of LBcarb and grown overnight at 37 °C with shaking at 250 rpm. Subsequently, 0.5 mL of the starter

culture were used to inoculate 50 mL of TBcarb in a 125 mL sterile Erlenmeyer flask. In parallel, plasmid DNA was isolated from the starter culture using a Monarch®Plasmid Miniprep Kit (New England Biolabs), and the sequence of each variant was confirmed by Sanger sequencing (Quintara Biosciences, Culver City, CA). Cultures were grown at 37 °C (220 rpm) to an OD600 of 0.6–0.8, cooled on ice for 30 min, and induced with IPTG (0.5 mM final) and ALA (1.0 mM final). Expression proceeded at 22 °C for 20–24 h. Cells were harvested (4,000×g, 5 min), and pellets were resuspended in M9-N buffer (10 g/L glucose) and normalized to OD600 = 30. Aliquots (380 µL) were transferred to microcentrifuge tubes and placed in a Coy anaerobic chamber. Reactions were initiated by adding 20 µL substrate solution ([substrate]final = 10 mM, [EDA]final = 15 mM, final MeCN concentration: 5%). Reactions were incubated overnight at room temperature (700 rpm), and each variant was tested in triplicate.

###### B.4.1 Analysis by GC-FID (compounds 1b - 4b)

Tubes were removed from the Coy chamber, and 400 µL of a 1:1 (v/v) EtOAc:cyclohexane containing an internal standard (see section Calibration curves for GC-FID analysis) were added to each tube. Tubes were vortexed and then centrifuged (20,000×g, 10 min, room temperature), and the organic layer (200 µL) was transferred to a 400 µL glass insert within a 2.0 mL screw-cap vial for GC–FID analysis. Analyses were performed on an Agilent J&W HP-5 column using the following oven program: 100 °C (2 min), ramped at 50 °C/min to 300 °C (hold 2 min) for a total run time of 10 min. The internal standard and product peaks were integrated using built-in analysis software (Agilent) and quantified using calibration curves.

###### B.4.2 Analysis by LC–MS (compound 4b for directed evolution)

Tubes were removed from the Coy chamber, and 1.2 mL of MeCN was added to each tube. After mixing, the tubes were centrifuged (20,000xg, 15 minutes, 4 °C), 300 µL of the supernatant and 50 µL of a stock solution of papaverine hydrochloride (internal standard, 6.5 mg in 450 mL 2:1 MeCN:H2O) were transferred into glass inserts within a 2.0 mL screw-cap vial for reversedphase HPLC-MS analysis (method details are in table S8.). The internal standard (papaverine hydrochloride) and product peaks were integrated using built-in analysis software (Agilent) and quantified using calibration curves prepared and measured at the same time as sample analysis (see section Calibration curves for LC-MS analysis).

Table S8. LC-MS methods for analysis of compound 4b (for directed evolution). Solvent A: H2O + 0.1% AcOH, solvent B: MeCN + 0.1% AcOH. Flow rate: 1 mL/min.

|Time (min)<br><br>|A (%)|B (%)|
|---|---|---|
|0.0 1.0 3.15 3.35 3.50 4.00|95.0 35.0 5.0 5.0 95.0 95.0<br><br>|5.0 65.0 95.0 98.0 5.0 5.0|

- B.4.3 Analysis of enantioselectivity

For 1b - 3b

Tubes were then removed from the Coy chamber, and 400 µL of a 1:1 (v/v) EtOAc:cyclohexane mixture were added to each tube. Tubes were vortexed and then centrifuged (20,000×g, 10 min, room temperature), and the organic layer (200 µL) was transferred to a 400 µL glass insert within a 2.0 mL screw-cap vial for chiral HPLC-UV analysis. Analyses were performed using a Chiralpak IC column (4.6 x 250 mm, 5 µm) and a Chiralpak IB column (4.6 x 250 mm,

- 5 µm) run in series for products 2b and 3b. Conditions for 2b: 30% isopropanol/hexanes, flow rate = 1.0 mL/min, runtime = 35 min, and UV-Vis detection at λ = 235 nm. For 3b: 6% isopropanol/hexanes, flow rate = 0.6 mL/min, runtime = 35 min, and UV-Vis detection at λ = 235 nm. For 1b, a Chiralcel OD-H column was used. Conditions: 6% isopropanol/hexanes, flow rate = 0.6 mL/min, runtime = 35 min, and UV-Vis detection at λ = 235 nm. The product peaks were integrated using built-in analysis software (Agilent).

###### For 4b (for directed evolution)

Residual samples (1.3 mL) worked up for LC-MS analysis were concentrated to dryness under a steady stream of compressed air. A 1:1 (v:v) cyclohexane:EtOAc mixture (400 µL) was then added to the tubes. The tubes were vortexed and then centrifuged (20,000×g, 10 min, room temperature), and the organic layer (200 µL) was transferred to a 400 µL glass insert within a 2.0 mL screw-cap vial for chiral GC-FID analysis. Analyses were performed on an Agilent CycloSil-B column using the following oven program: 50 °C (0 min), ramped at 10 °C/min to 170 °C (hold 19 min), then at 30 °C/min to 200 °C (hold 8 min), for a total run time of 40 min.

###### B.5 Hemochrome assay

Protein concentration in the cell was determined using the hemochrome assay on the cell lysate. Lysate was obtained by sonication using a Qsonica Q500 sonicator (9 min total, 1 s on, 2 s off, 35% amplitude, on wet ice). The cell debris was removed by centrifugation (5,000×g,

- 10 min, 4 °C). To a cuvette, 500 µL of the lysate and 500 µL of Solution I [0.2 M NaOH, 40%

(v/v) pyridine, 0.5 mM K3Fe(CN)6] were added. The UV–Vis spectrum (500–600 nm) of the oxidized state Fe(III) was recorded immediately. Sodium dithionite (10 µL of 0.5 M solution in 0.5 M NaOH) was added and mixed thoroughly by pipetting, and the UV–Vis spectrum of the reduced state Fe(II) was recorded immediately. The protein concentration was calculated using the extinction coefficient and dilution factor:

ε557reduced−540oxidized = 23.98 mM−1cm−1 (S58)

###### B.6 Selectivity results

[Figure 135]

- Figure S29. Enantiomeric excess (%) of methoxystyrene cyclopropanation product (trans, 1b). The corresponding enantiomeric ratio of dCT-H11 is 61:39.

[Figure 136]

- Figure S30. Enantiomeric excess (%) of the product of B–H insertion 2b. The corresponding enantiomeric ratio of dCT-F9 is 63:37.

[Figure 137]

###### Figure S31. Enantiomeric excess (%) of the product of C–H insertion 3b. The corresponding enantiomeric ratio of dCT-H11 is 40:60.

[Figure 138]

###### Figure S32. Diastereomeric excess (%) of methoxystyrene cyclopropanation product (trans vs cis, 1b) in selected validation results. The corresponding diastereomeric ratio of dCT-H11 is 99:1 (trans:cis).

###### B.7 Directed evolution

Random mutations were introduced using error-prone PCR (epPCR). PCRs were run using Taq polymerase (New England Biolabs) in the presence of 300, 400, and 500 µM MnCl2 with the primers shown in table S9 following a reported procedure.[58] Primers 005 and 006 were used to amplify the DNA region encoding the full-length protein of interest. Primers 007 and 008 were used to amplify the backbone (pET22b(+) vector) fragment containing a gene encoding for ampicillin resistance using Phusion™ polymerase and the template plasmid DNA. PCR products were isolated, and fragments were then assembled into a circular plasmid using the Gibson assembly protocol, incubating at 50 °C for 1 hour. Then, 2.5 µL of the Gibson product were used to transform 10 µL of competent E. coli cells. Transformed cells were supplemented with 150 µL SOC medium and immediately plated on LBcarb agar plates. Plates were incubated at 30 °C for 16–18 hours. Single colonies were then used to inoculate 500 µL LBcarb 96-well plates; eight wells on each plate were inoculated with single colonies of the parent enzyme (parent controls), and two wells were inoculated with a sterile toothpick (sterile controls). Expression, reaction setup and work-up in 96-well plates was done as described in section B.3. A total of eight plates were screened by reversed-phase HPLC–MS, and sequences were verified by in-house sequencing (LevSeq).[57] The highest-performing variants were validated in triplicate as described in section B.3, and the enantioselectivity was measured (see figure S33).

Table S9. Primers used in error-prone PCR.

Primers Sequence (5’ → 3’)

- 005 GAA ATA ATT TTG TTT AAC TTT AAG AAG GAG ATA TAC ATA TG
- 006 GCC GGA TCT CAG TGG TGG TGG TGG TGG TGC TCG AG
- 007 CAT ATG TAT ATC TCC TCC TTA AAG TTA AAC AAA ATT ATT TC
- 008 CTC GAG CAC CAC CAC CAC CAC CAC TGA GAT CCG GC

Table S10. Mutations of highest-performing variants from error-prone PCR.

Variant Mutations dCT-r1-B7 A2P L68R E73D F82L N90I L103P S167F

- dCT-r1-D1 E39D D110E T121A F173S E191V H199L dCT-r1-E3 A66V W98R E144 V156A
- dCT-r1-E12 V26A K27N M43L R77C F120Y G150D I152T L161P R180C L195H
- dCT-r1-F1 V8E K27E W98R L114P L165F

[Figure 139]

###### Figure S33. Selected validation results of the evolution of the spirocyclopropanation reaction 4b. The parent (dCT-H11) is shown in gray. The corresponding enantiomeric ratios of dCT-H11, dCT-r1-E12, and dCT-r1-F1 are 68:32, 75:25 and 25:75, respectively.

###### B.8 DNA sequences of the best designs

>dCT-H11 ATGGCGCTGCCAGATATCGAAGTGTACCCGGATCTGGAACTGAGCGAGCGTCAGCGT GAACGCTTAGTCAAAATAGTGAAAGCGGCCATTGAAGAAATCTATGAACGCGGTTA CGAAGCCACGTCGATGGAAGATATCGCCGAGCGCGCCGGCATTTCACGTGAGACTGT TCGCTACTATTTTCCTTCTAAAGAGGCTCTGCTGAAAGCAGCTGCAGAATTTTTAGCG CGCCGTCATGAACGTTTCTTCGAAGAATGGAGTAACATTAATAGCCCCGAAGAGCTT GAAGATTGGTTGCAGAAACTGCTTGATAGCTTGGAACGGGTCGATCTGGAGAAACTG GTGAACCTGGCGGACTTCACCGAAGCAGAGCCGGAGCTCAAAGAGAAAATCAAAG AACTGCTGTATCATCGGCTGAAAAAACTCTTGGAAGAAGGGAAGAAAAAGGGCCAA ATTAAAGAGGACGTGGATGTTGAAGAACTGACCAAGGAATTACTGTCCATGATTGAT CAGCTGTTTAGTACGTATATCAAAACACGTGACCGCGCGGAATTTGAAAAGCTCAAA AAGGAGATTTTGGACCTTTTACGTAAACACGTAGTTCTCGAGCACCACCACCACCAC CACTGA

>dCT-H10 ATGGGGGTGGCCGCGGAATTAGAACGTGCAGCGCGGGCTTTACGTGAAGCTCTTCGC GCAGCTGTTCGTGAGGCTGGTTTACGGCTGCCTGGTGAGGAAGCCGAAGCGTTCGAA CGCCTGCTGGCCTTACATTGCGAGGTGGCTGACGGTCTGTTGCGCGTTCTGGCGGCAG CGTTAACTCGTGATCCTGCAGAGTGTGCTGCTGCTGCGCGTGCCGTAGGTGAATTGGC TCTGCGCCTGTTGGAAGCAAGTGCTGCGCTTGTTGTACCTCCAGGACGTGCAGCGCTT GCGCGTGCGTTTGTTGAACGGTTGCGTGGCTTTGTCGAAGGCTTCGTGGCGTTAGCGG AGCGTGTTGGCGGAGCGTCTGCAGATGCTTTAGAGGAAGCAGCTCGTCGTCTGGCGG CCGCTGCTTTGGCAGCAGATCGTGTAGTTCTGGCCTTCTTTGAAGAAGCGCTTGCTGA AGCCGAGCGTGCGGGCGATGAACGTTACGCGCGCTTATGCCGTGAACGTCTGGCAGT GTATCCTCGCGTGAGCCGTGCGGCATTAGACGCGTTTGCGGCATTAGCAGTTCGTCTG ATTCGCGCCGCTGCTCGTGCGCTGGCCCTCGAGCACCACCACCACCACCACTGA

- >dCT-F9 ATGTCTTATGAAGAGCTGCTCGAGGAAATTGCACGCGAGATCCTGCGCAGTGCGCGC GCTGTGTATGAGGGATCAATCGAAGAACATCGTGAACATGTTCGCCGTGCACGCGAA TTGCTTGAGCGTCTGGGCAATGTCCGTGTAGTTGTCGAAGTGCGTGATCCGGATGGCG TTCCCCTGTTAGTAGTCCGTTTTGAAGTGGAAAACGGCGAATGGCGTATCGACGCGAT ATACAACCAAGTGTCGCGCTTAACCGATCAGGAGTTCTACGAGTTCGTGCGGGACAA TCCGGGTATTGACCCACGGGAACTCCTGCCTCACCTTGCCGAATTTGTTCGTCGCCTG AAAGAATCCGGTAGCGTTGAGGTGCGGGTACGCATTGAATTGACGGATGAAGAAGT GGTGTTAGAACTGGTCGAACGTCTGGAGGGGCGTCCGGCCGAAGAACTGTATCTCGA GCACCACCACCACCACCACTGA
- >dCT-G9 ATGGAATACGATTATAAAGCTCGCAGCCTTAAAAAAGCCGAGTTACTGACTAACGCT CTGAAAGGTGAAAACGCTTCGCCAGAAGTTAAAGCGTTTGTAGATGCGCTTAAAAAG AAAGTGGGAGATGATATTGTGCCGGCGGTCGAAGCACTGAAGAATGTTAGCGAAGA CCTCAAGAAAGAAATTGTAGAGCTGCTTCGTAAAGCATTGGATAGTGCCAAAACCA

AGGACGAACTGATCGCCGCGTTAACCAAACTGCATGGCTTAGTCTTCGACTCCCTGG GGCTGCCGAATGCCGATGCAATAATCAAACAGGTGACCGATGCCCTGAAGCAGAAG TCTCTGGAGGAGCTCAAAGAAGTGTTGAAAACAGTCGTTGATGGTCACACGGGCAAA ACGGACGAAAAACAAGCAATCGAAAAAGCAATGGCGTTGGCCGCGGCGATTTTTTC ACTCGAGCACCACCACCACCACCACTGA

>dCT-G1 ATGACTTTTGAAGAACTTAAAAAGAAAACGAAAGAGGAGGTGCGTGAAATTGTTGA AAGTTGCGATCCTAAAGAACTTGTCGAGCGTCTCCAGGAACTGGTGCGTGAAACACA TAACTCGGTGCTTCCGCCGACCATAGTAGAATTTCAAAGCAATCTGTTTGAACGGCTG CTGGCTGCGAAAACGGTCGAAGAGAAAGCGCTGTTGCTGGAGAAGTTGGTTTTTATT GGTATCTATGTCAAGATCGTTCGCAAATACGATGAGGAGCTGTTCAAAAAGGTACAG GCCATTTTACGGCGTTTCTTGGAGGAAATCGACACCTTAACCGAAGAAAAAATTAAA GAATTCCTGGATCGCATTTATGAACTGGTGGAACCACTGAAACGCGCCTGGCTCGAA GAAGGCGAGAAAAACGCACGCAAAGTGCTGGAAGCAATCAAGAAATTATGTCTCGA GCACCACCACCACCACCACTGAGA

###### B.9 Calibration curves for GC-FID analysis

- For 1b, 3b and 4b: The analyte of interest and internal standard (1,2-diphenylethane) were dissolved in a 1:1 (v:v) solution of cyclohexane and EtOAc to create two separate 100 mM solutions. Seven different standard-to-internal standard (S/IS) ratios were prepared: 1:1, 0.5:1, 0.25:1, 0.125:1, and 0.0625:1 by mixing specific volumes of the 100 mM solutions in a GC vial. The total volume for each sample was adjusted to 500 µL by adding cyclohexane:EtOAc (1:1, v/v). An aliquot of the prepared solutions was transferred to a 400 µL glass insert within a 2.0 mL screw-cap vial and subjected to GC-FID analysis with an achiral stationary phase.
- For 2b: The analyte of interest was prepared under conditions analogous to those used for enzymatic reactions. A 200 mM stock solution in MeCN was prepared, from which successive stocks of 100, 20, 10, 2 and 1 mM were generated. From each stock, 20 µL were added to 380 µL of reaction buffer (M9-N, pH 8.0, 10 g/L glucose), followed by 400 µL of a 1:1 (v/v) cyclohexane:EtOAc solution containing the internal standard (1,2,3-trimethoxystyrene, 1 mM). Tubes were vortexed and centrifuged (20,000×g, 10 min, room temperature), and 200 µL of the organic layer were transferred to a 400 µL glass insert in a 2.0 mL screw-cap vial for GC–FID analysis. The IS and S peaks were integrated using built-in analysis software (Agilent). Calibration curves were generated by plotting the known concentrations of the product, based on prepared S/IS ratios, as a function of the ratio of its FID signal (peak area) relative to that of the internal standard. The slope of the calibration curve was used to calculate the product yield in enzymatic reactions.

###### B.9.1 Calibration curves for 1b

[Figure 140]

- Figure S34. Calibration curve of the cis isomer of 1b using 1,2-diphenylethane as IS.

[Figure 141]

- Figure S35. Calibration curve of the trans isomer of 1b using 1,2-diphenylethane as IS.

###### B.9.2 Calibration curves for 2b

[Figure 142]

- Figure S36. Calibration curve of 2b using 1,2,3-trimethoxystyrene as IS.

B.9.3 Calibration curves for 3b

[Figure 143]

- Figure S37. Calibration curve of 3b using 1,2-diphenylethane as IS.

###### B.9.4 Calibration curves for 4b

[Figure 144]

Figure S38. Calibration curve of Compound 4b using 1,2-diphenylethane as IS.

###### B.10 Calibration curves for LC-MS analysis

A stock solution of tert-butyl-3-methyleneazetidine-1-carboxylate (4a) was prepared in acetonitrile (1 mM final concentration) and used to create successive stocks of 10, 30, 50, 100, 200, 300 µM concentration. An aliquot of these stocks (300 µL) and 50 µL of a stock solution of papaverine hydrochloride (internal standard, 6.5 mg in 2:1 MeCN:H2O) were transferred to 400 µL glass inserts in 2.0 mL screw-cap vials for LC–MS analysis.

[Figure 145]

Figure S39. Calibration curve of 4b on LC-MS for directed evolution assay.

###### B.11 Synthesis of product standard 3b

[Figure 146]

Figure S40. Synthesis of 3b.

- Step 1: A sealed tube flushed with nitrogen was charged with pyrrolidine (5 mmol), potassium carbonate (2 equiv, 10 mmol), copper (I) iodide (0.25 mmol, 5 mol%), iodobenzene (1.2 equiv,

6 mmol) and DMF (7.5 ml). The mixture was heated at 90 °C for 48 hours, then cooled to room temperature. Water was added, and the pH value was adjusted to <3 with concentrated HCl. The aqueous phase was extracted four times with ethyl acetate. The combined organic layers were washed with brine, dried over magnesium sulfate, filtered and concentrated under reduced pressure. Purification by flash column chromatography (0 to 100% EtOAc/hexane gradient) afforded the product.

- Step 2: A round-bottom flask was charged with carboxylic acid (1.0 equiv), ethanol (1.1 equiv), and DMAP (0.1 equiv). Dichloromethane was added (0.4 M), and the mixture was stirred

[Figure 147]

Figure S41. 1H NMR (400 MHz, CDCl3) of 3b.

vigorously. DIC (1.1 equiv) was then added dropwise via syringe, and the reaction mixture was stirred until consumption of the acid was complete, as determined by TLC. The mixture was filtered through a fritted funnel and rinsed with CH2Cl2/Et2O. The solvent was removed under reduced pressure, and the crude product was purified by flash column chromatography to afford the corresponding ester as a yellow oil.

1H NMR (400 MHz, CDCl3) δ 7.26 – 7.22 (m, 2H), 6.71 – 6.65 (m, 1H), 6.65 – 6.55 (m, 2H), 4.20 – 4.15 (m, 3H), 3.46 – 3.39 (m, 1H), 3.22 – 3.13 (m, 1H), 2.79 (dd, J = 15.0, 2.9 Hz, 1H), 2.26 – 2.17 (m, 1H), 2.17 – 1.94 (m, 4H), 1.28 (t, J = 7.1 Hz, 3H). 13C NMR (101 MHz, CDCl3) δ 172.1, 146.5, 129.5, 129.4, 121.6, 116.0, 111.9, 55.4, 47.9, 37.8, 31.0, 23.1, 14.3.

Spectroscopic data are in agreement with the literature.[33]

[Figure 148]

###### Figure S42. 13C NMR (101 MHz, CDCl3) of 3b.

###### B.12 HPLC–UV chromatograms for enantioselectivity determination

[Figure 149]

- Figure S43. HPLC-UV chromatogram of an authentic standard of compound 1b.

[Figure 150]

- Figure S44. HPLC-UV chromatogram of compound 1b (reaction with dCT-B9).

[Figure 151]

###### Figure S45. HPLC-UV chromatogram of compound 1b (reaction with dCT-F3).

[Figure 152]

###### Figure S46. HPLC-UV chromatogram of compound 1b (reaction with dCT-F9).

[Figure 153]

###### Figure S47. HPLC-UV chromatogram of compound 1b (reaction with dCT-G1).

[Figure 154]

###### Figure S48. HPLC-UV chromatogram of compound 1b (reaction with dCT-G9).

[Figure 155]

###### Figure S49. HPLC-UV chromatogram of compound 1b (reaction with dCT-H5).

[Figure 156]

###### Figure S50. HPLC-UV chromatogram of compound 1b (reaction with dCT-H10).

[Figure 157]

- Figure S51. HPLC-UV chromatogram of compound 1b (reaction with dCT-H11).
- Figure S52. HPLC-UV chromatogram of compound 1b (reaction with TmTrpB control).

[Figure 158]

[Figure 159]

- Figure S53. HPLC-UV chromatogram of compound 1b (reaction with ApePgb 5312 control).
- Figure S54. HPLC-UV chromatogram of an authentic standard of compound 3b.

[Figure 160]

[Figure 161]

###### Figure S55. HPLC-UV chromatogram of compound 3b (reaction with dCT-B9).

[Figure 162]

###### Figure S56. HPLC-UV chromatogram of compound 3b (reaction with dCT-F3).

[Figure 163]

###### Figure S57. HPLC-UV chromatogram of compound 3b (reaction with dCT-F9).

[Figure 164]

###### Figure S58. HPLC-UV chromatogram of compound 3b (reaction with dCT-G1).

[Figure 165]

###### Figure S59. HPLC-UV chromatogram of compound 3b (reaction with dCT-G9).

[Figure 166]

###### Figure S60. HPLC-UV chromatogram of compound 3b (reaction with dCT-H5).

[Figure 167]

###### Figure S61. HPLC-UV chromatogram of compound 3b (reaction with dCT-H10).

[Figure 168]

###### Figure S62. HPLC-UV chromatogram of compound 3b (reaction with dCT-H11).

[Figure 169]

###### Figure S63. HPLC-UV chromatogram of compound 3b (reaction with TmTrpB control).

[Figure 170]

###### Figure S64. HPLC-UV chromatogram of compound 3b (reaction with ApePgb 5312 control).

[Figure 171]

- Figure S65. HPLC-UV chromatogram of an authentic product standard of compound 2b.
- Figure S66. HPLC-UV chromatogram of compound 2b (reaction with dCT-B9).

[Figure 172]

[Figure 173]

###### Figure S67. HPLC-UV chromatogram of compound 2b (reaction with dCT-F3).

[Figure 174]

###### Figure S68. HPLC-UV chromatogram of compound 2b (reaction with dCT-F9).

[Figure 175]

###### Figure S69. HPLC-UV chromatogram of compound 2b (reaction with dCT-G1).

[Figure 176]

###### Figure S70. HPLC-UV chromatogram of compound 2b (reaction with dCT-G9).

[Figure 177]

###### Figure S71. HPLC-UV chromatogram of compound 2b (reaction with dCT-H5).

[Figure 178]

###### Figure S72. HPLC-UV chromatogram of compound 2b (reaction with dCT-H11).

[Figure 179]

###### Figure S73. HPLC-UV chromatogram of compound 2b (reaction with TmTrpB control).

[Figure 180]

###### Figure S74. HPLC-UV chromatogram of compound 2b (reaction with ApePgb 5312 control).

###### B.13 Chiral GC–FID traces for enantioselectivity determination

[Figure 181]

Figure S75. Chiral GC-FID trace for racemic standard of 4b.

[Figure 182]

Figure S76. Chiral GC-FID trace for an enantiopure standard of (R)-4b.

[Figure 183]

Figure S77. Chiral GC-FID trace for 4b (reaction with dCT-H11).

[Figure 184]

- Figure S78. Chiral GC-FID trace for 4b (reaction with evolved variant dCT-r1-B7).

[Figure 185]

###### Figure S79. Chiral GC-FID trace for 4b (reaction with evolved variant dCT-r1-E3).

[Figure 186]

###### Figure S80. Chiral GC-FID trace for 4b (reaction with evolved variant dCT-r1-E12).

[Figure 187]

###### Figure S81. Chiral GC-FID trace for 4b (reaction with evolved variant dCT-r1-F1).

[Figure 188]

###### Figure S82. Chiral GC-FID trace for 4b (reaction with evolved variant dCT-r1-D1).

