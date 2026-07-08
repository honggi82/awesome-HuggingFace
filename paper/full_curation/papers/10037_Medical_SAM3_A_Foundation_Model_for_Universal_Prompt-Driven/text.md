# arXiv:2601.10880v1[cs.CV]15Jan2026

## Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation

Chongcong Jiang1*, Tianxingjian Ding1*, Chuhan Song2*, Jiachen Tu3, Ziyang Yan4†, Yihua Shao5, Zhenyi Wang1, Yuzhang Shang1, Tianyu Han6, and Yu

Tian1†

- 1 University of Central Florida, Orlando, USA
- 2 University College London, London, UK

3 University of Illinois Urbana-Champaign, Champaign, USA 4 University of Trento, Trento, Italy

- 5 The Hong Kong Polytechnic University, China
- 6 University of Pennsylvania, Philadelphia, USA yu.tian2@ucf.edu

Abstract. Promptable segmentation foundation models such as SAM3 have demonstrated strong generalization capabilities through interactive and concept-based prompting. However, their direct applicability to medical image segmentation remains limited by severe domain shifts, the absence of privileged spatial prompts, and the need to reason over complex anatomical and volumetric structures. Here we present Medical SAM3, a foundation model for universal prompt-driven medical image segmentation, obtained by fully fine-tuning SAM3 on large-scale, heterogeneous 2D and 3D medical imaging datasets with paired segmentation masks and text prompts. Through a systematic analysis of vanilla SAM3, we observe that its performance degrades substantially on medical data, with its apparent competitiveness largely relying on strong geometric priors such as ground-truth-derived bounding boxes. These findings motivate full model adaptation beyond prompt engineering alone. By fine-tuning SAM3’s model parameters on 33 datasets spanning 10 medical imaging modalities, Medical SAM3 acquires robust domain-specific representations while preserving prompt-driven flexibility. Extensive experiments across organs, imaging modalities, and dimensionalities demonstrate consistent and significant performance gains, particularly in challenging scenarios characterized by semantic ambiguity, complex morphology, and long-range 3D context. Our results establish Medical SAM3 as a universal, text-guided segmentation foundation model for medical imaging and highlight the importance of holistic model adaptation for achieving robust prompt-driven segmentation under severe domain shift. Code and model will be made available at https://github.com/AIM-Research-Lab/ Medical-SAM3.

Keywords: Medical Image Segmentation · Foundation Models · FineTuning · SAM3

* Co-first authors. † Corresponding to Z. Yan and Y. Tian.

[Figure 1]

###### Input Modalitites Text Prompts

[Figure 2]

[Figure 3]

[Figure 4]

Fetal Head

[Figure 5]

Liver

[Figure 6]

Retinal Blood Vessel

[Figure 7]

Ultrasound Laparoscopy Fundus

[Figure 8]

[Figure 9]

[Figure 10]

Lung

[Figure 11]

Nuclei

[Figure 12]

Skin Lesion

[Figure 13]

CT Pathology Dermatoscopy

[Figure 14]

[Figure 15]

[Figure 16]

Polyp

[Figure 17]

Bone Tumor

[Figure 18]

Colonoscopy X-Ray MRI Thyroid Gland

[Figure 19]

Segmentation Outputs

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

### Medical SAM3

Prompt-driven Foundation Model

[Figure 27]

[Figure 28]

[Figure 29]

- Fig. 1. Universal medical image segmentation via text prompting with Medical SAM3. Our proposed model unifies diverse medical imaging modalities—ranging from radiology (CT, MRI, X-Ray) to optical imaging (Fundus, Dermoscopy, Endoscopy) and pathology—into a single framework.

#### 1 Introduction

Medical image segmentation aims to delineate clinically relevant structures and abnormalities in medical images at the pixel or voxel level. By enabling objective quantification of disease extent and anatomical changes, segmentation supports lesion assessment, surgical or radiotherapy planning, and longitudinal

follow-up [35,12,62]. Despite remarkable progress in deep learning, many models remain optimized for specific tasks and data distributions, making adaptation to new modalities, anatomies, pathologies, or clinical sites challenging [16,37,57]. This reliance on expert dense annotation and dataset-specific optimization limits scalability and hinders deployment in long-tail rare conditions and heterogeneous real-world settings, especially under distribution shift.

Methodologically, the field has been dominated by fully supervised specialist models trained with dense annotations. Convolutional neural networks (CNNs) and vision transformers (ViTs) have achieved strong performance for medical segmentation [54,46,8,18], and automated pipelines further reduce manual tuning [24]. However, these advances largely remain within a dataset-centric paradigm and do not readily generalize across modalities and clinical sites, motivating promptable foundation models that provide a more unified and scalable interface for segmentation.

Segmentation foundation models offer a promising alternative, aiming to generalize across tasks through prompt-based interaction while reducing taskspecific retraining [56]. The Segment Anything Model (SAM) [29] demonstrated remarkable zero-shot generalization in natural images via visual prompts, and subsequent models such as SAM3 [7] extend this paradigm with concept-based prompting. However, a critical gap remains. Medical images differ substantially from natural scenes in acquisition protocols and semantic structure, often leading to unstable performance under zero-shot or lightly adapted settings [20,45,23]. More crucially, many previous foundation models achieve competitive results only by relying on ground-truth-derived bounding boxes, essentially utilizing oracle localization cues [41,79,75,58]. While effective for interactive refinement, such privileged geometric priors largely remove the localization challenge and reduce the problem to boundary refinement, which may confound comparisons when geometric priors are not available at deployment. In real deployments, boxes must be provided by a clinician or an upstream detector. Without such cues, text-only prompts often degrade sharply under severe domain misalignment[89,40,59], motivating holistic model adaptation for robust, prompt-driven medical segmentation.

To address these challenges, we present Medical SAM3, a universal promptdriven foundation model for medical image segmentation obtained by holistically adapting SAM3 on large-scale, heterogeneous 2D and 3D medical datasets with paired segmentation masks and text prompts. By moving beyond lightweight adapters and reducing reliance on pre-defined geometric cues (e.g., bounding boxes), Medical SAM3 learns robust domain-specific representations while preserving promptable flexibility under severe domain shift. We further conduct a systematic diagnostic study of vanilla SAM3 in medical settings and evaluate Medical SAM3 across both internal validation tasks and external validation tasks spanning diverse organs, modalities, and dimensionalities. Across this suite, Medical SAM3 achieves state-of-the-art performance and supports a spatial-promptfree, semantic-driven paradigm for medical image segmentation. In summary, our contributions are threefold: (i) we introduce Medical SAM3 by holistically

adapting SAM3 for universal, text-guided medical segmentation without privileged spatial prompts; (ii) we provide a diagnostic study that characterizes the failure modes of vanilla SAM3 under severe domain shift and its reliance on geometric cues; and (iii) we curate a large-scale text–image–mask aligned medical segmentation corpus and establish strong results through extensive internal and external evaluations across diverse organs, modalities, and 2D/3D settings.

#### 2 Related Works

Specialist Medical Image Segmentation. Fully supervised specialist models remain the dominant paradigm in medical image segmentation. Early encoder– decoder CNNs and their variants, represented by FCN and U-Net, establish strong inductive biases for dense prediction and are widely extended with attention and redesigned skip connections [73,39,54,49,91,69,70,71]. For volumetric imaging, 3D architectures directly model spatial context in CT and MRI, including 3D U-Net and V-Net [9,46]. Beyond architecture design, automated training pipelines such as nnU-Net substantially reduce manual engineering and provide strong baselines across datasets [24]. Large scale multi-organ segmentation systems further demonstrate that broad anatomical coverage can be achieved when sufficient annotations and standardized pipelines are available [78,60]. More recently, Transformer based designs improve global context modeling for medical segmentation, including hybrid and fully Transformer architectures [8,19,18,83,90,72]. In parallel, selective state space models (SSMs), exemplified by Mamba, have been explored to capture long range dependencies with improved efficiency, inspiring Mamba-based medical segmentation architectures such as U-Mamba, SegMamba, VM-UNet, and Swin-UMamba [15,42,82,55,36].

Text Guided and Open Vocabulary Segmentation. Text guided segmentation in general vision is commonly approached by aligning dense visual features with language representations to enable open vocabulary mask prediction [40,52,84,11,33]. Referring expression segmentation further studies phrase grounded masks through explicit cross modal fusion [61,77,85]. These lines of work provide complementary perspectives on semantic conditioning and prompt design that are relevant to text based target specification in medical segmentation.

Promptable Segmentation Foundation Models. Interactive medical segmentation predates recent foundation models and commonly improves an automatic prediction with lightweight user inputs, such as clicks or scribbles, as exemplified by DeepIGeoS [74]. SAM introduces a promptable interface via a prompt encoder and a mask decoder, enabling segmentation conditioned on spatial prompts [29], and SAM 2 extends this design with memory for streaming image and video settings [53]. Medical adaptations of SAM style models have been studied through supervised domain adaptation and parameter efficient customization, including MedSAM and Medical SAM Adapter [41,79]. Extending

Text Prompt

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Prompt Encoder

Detector

⊕

Output for frame t

Mask detected in frame t

Mask propagated from frame t-1

[Figure 35]

[Figure 36]

[Figure 37]

Image Encoder

Tracker

Memory Bank

- Fig. 2. Overview of Medical SAM3. Medical SAM3 takes a text prompt and medical images (2D or slice-based 3D) as input. A detector segments target instances in the current frame, while an optional tracker propagates masks across frames via a memory bank. The final prediction is produced by merging detected and propagated masks, supporting semantic-driven segmentation without privileged spatial prompts.

promptable segmentation to volumetric data has been explored from different perspectives, including learning native 3D promptable models such as SAMMed3D and using memory mechanisms for 3D image/video settings such as MedSAM2 [75,43]. In addition, universal medical segmentation has been investigated via prompt driven multi task learning and minimal interaction paradigms, including UniSeg, MedUniSeg, and One-Prompt Segmentation [87,86,80]. Most recently, concept based prompting has been introduced in SAM3 to broaden conditioning beyond purely geometric cues [7], and large vocabulary medical segmentation driven by text prompts has also been explored [89].

- 3 Method

In this paper, we propose a full fine-tuning strategy to adapt SAM3 [7], a largescale promptable segmentation foundation model, to medical imaging under severe domain shift. Unlike parameter-efficient or partial fine-tuning approaches, we update all model parameters to enable comprehensive domain adaptation. Crucially, we introduce no architectural modifications to SAM3. Figure 2 illustrates our training pipeline for 2D and 3D modalities, which follows SAM3’s detector–tracker design for sequential inputs. At frame t, Medical SAM3 combines a mask detected by the detector with a mask propagated from slice t−1 by the tracker, and updates the memory bank for subsequent propagation.

##### 3.1 Unified Input Formulation

Medical imaging spans a wide spectrum of departments, encompassing natively planar modalities such as Histopathology, Fundus photography, Dermatology,

and Projection Radiography (X-ray). To harmonize these heterogeneous data sources into a generalist foundation model, we unify these modalities within a common 2D feature space. By treating each medical scan as a high-fidelity 2D image, we maximize the model’s applicability across diverse clinical workflows without being constrained by inconsistent 3D acquisition geometries. This strategy not only simplifies the integration of diverse clinical workflows but also enables the perception backbone to prioritize high-resolution spatial features (at 1008 × 1008 pixels), which are often compromised in computationally heavy volumetric frameworks.

To leverage the 33 diverse datasets during joint training, we structure each sample into a text-driven triplet (I,M,t), where I is the image, M is the corresponding mask, and t is the text prompt derived directly from the dataset’s clinical labels. Unlike traditional segmentation models that require a fixed, closed-set label space, our approach exploits the semantic flexibility of the pre-trained text encoder. By associating masks with their native clinical nomenclature, the model learns to associate varied terminology with their corresponding visual features. This strategy avoids the need for complex label re-mapping while allowing the model to internalize a vast range of anatomical and pathological descriptors across disparate medical domains.

##### 3.2 Stratified Tuning

Medical images contain critical diagnostic details that necessitate high spatial resolution. We maintain a training resolution of 1008 × 1008 pixels to align with the high-frequency spatial priors inherited from the original large-scale pre-training. This ensures that the positional embeddings remain synchronized with the perception backbone.

To mitigate the significant domain gap between natural and medical textures without catastrophic forgetting, we employ Layer-wise Learning Rate Decay (LLRD). For a base learning rate ηbase, the learning rate ηl for the l-th layer of the vision backbone is defined as:

ηl = ηbase · γL−l (1)

where L = 12 is the total number of layers and γ = 0.85 is the decay factor. This stratified strategy allows shallow layers to retain general-purpose visual primitives, such as edges and textures, while forcing deeper layers to specialize in complex medical semantics.

##### 3.3 Text-Driven Semantic Alignment

In practical clinical environments, the requirement for manual bounding boxes as spatial priors often creates a bottleneck, as it assumes the clinician has already identified the target’s precise location. To maximize the utility of Medical SAM3 as an autonomous assistant, we transition from a prompt-dependent paradigm to a strictly text-driven semantic alignment strategy. By utilizing clinical concepts

- as the sole input during training, we force the model to develop an intrinsic spatial awareness that bridges abstract medical nomenclature with pixel-level morphological features.

This alignment process is formulated as a semantic-to-spatial distillation task. Without the crutch of a bounding box, the transformer decoder must learn to treat the text embedding ztxt = Etxt(c) not merely as a class label, but as a discriminative spatial query. Through this pure text-driven supervision, the model is compelled to identify long-range correlations between high-level clinical descriptors (e.g., “irregular mass,” “calcified node”) and specialized pathological textures within the vision backbone’s feature maps. This global-to-local reasoning path ensures that the linguistic manifold and the visual manifold are explicitly aligned. Consequently, at inference time, the model can interpret conceptual keywords and autonomously perform zero-shot localization, effectively simulating a clinician’s cognitive process of translating a diagnostic term into a visual search.

##### 3.4 Set-Prediction Objective

We optimize the model using a multi-task objective that jointly supervises instance discovery and semantic segmentation. Given predicted queries {yˆi}Ni=1 and ground-truth instances {yj}Mj=1, we establish a one-to-one assignment π via bipartite Hungarian matching. To address potential sparse supervision in medical scenes, an auxiliary one-to-many (O2M) matcher πo2m is employed to enhance training stability. The total objective is:

Ltotal = Lfind(π) + λo2mLfind(πo2m) + Lseg, (2) where all terms are normalized by the batch-wise matched instance count.

Finding Loss. For matched queries, Lfind supervises classification, presence, and localization:

+ λgLgiou), (3)

Lfind(π) = λceLce + λprLpres + 1{j̸=∅}(λℓ

1Lℓ1

where Lce is a focal-style classification loss and Lpres supervises query presence. The box regression terms (ℓ1 and GIoU) are computed only for positive assignments (j ̸= ∅).

Segmentation Loss. To ensure precise mask boundaries—critical for clinical quantification—the segmentation loss Lseg combines pixel-wise and structural terms:

Lseg = λfLfocalseg + λdLdice + λspLseg-pres, (4)

where Ldice improves boundary adherence and Lseg-pres provides semantic presence supervision.

#### 4 Experiments

##### 4.1 Datasets

To develop a prompt-driven foundation model with strong generalization to medical segmentation tasks, we fine-tune Medical SAM3 on a diverse multi-domain collection assembled from publicly accessible datasets, where each sample is paired with a segmentation mask and a text prompt that is manually curated or derived from dataset labels.

As shown in Table 1, the collected corpus encompasses 33 datasets across 10 imaging modalities—including radiography (CXR and X-ray/angiography), ultrasound, endoscopy, pathology, fundus, dermoscopy, microscopy, virtual microscopy, electron microscopy, and others—amounting to a total of 76,956 images and 263,705 mask annotations. Radiography is the largest contributor with 40,160 images, dominated by large-scale CXR collections. Ultrasound and endoscopy/fetoscopy form two mid-sized groups with 12,179 and 12,887 images, respectively, while the remaining modalities provide long-tail diversity that improves coverage of appearance, acquisition, and annotation styles. The median annotation area varies by several orders of magnitude, ranging from 58 px in electron microscopy nuclei to over one million pixels in chest radiographs, highlighting substantial scale variation in segmentation targets. For consistency, we standardize all datasets to an 85/15 split for training and validation with a fixed seed of 42, yielding approximately 65.4k training images and 11.5k validation images.

Our evaluation protocol is designed to rigorously test robustness under domain shift, we evaluate Medical SAM3 on 10 internal validation tasks derived from held-out splits of the fine-tuning corpus. Complementing this, we conduct external validation on 7 segmentation tasks that were entirely excluded from the model development pipeline.

Unifying these diverse benchmarks within a prompt-driven framework requires structuring each sample as an (image, mask, text) triplet. While our architecture natively supports prompts of varying granularity, spanning from broad categories to detailed descriptive attributes, we prioritize atomic clinical concepts in this study to establish a consistent baseline. We define a dataset-specific label taxonomy and map it to a unified vocabulary of canonical concept names. Single-class datasets are assigned a single global concept; Multi-class datasets use a one to one mapping from label indices to anatomy or pathology terms defined by the dataset specification. This label-to-text dictionary ensures that each segmentation mask is paired with a consistent and standardized prompt across datasets.

##### 4.2 Experimental Settings

For both training and evaluation, we use text-only prompts for all internal and external tasks. Prompts are instantiated by applying the label-to-text mapping protocol in Sec. 4.1, resulting in a single canonical concept term per class for both

training and evaluation. We compare the results of the original SAM3 [7] and our Medical SAM3. The original SAM3 is evaluated using the official checkpoint without any additional training on medical data. Medical SAM3 is initialized from the same SAM3 checkpoint and obtained by full parameter fine tuning on the training splits described in Sec. 4.1.

All training and testing are implemented in PyTorch with distributed data parallelism using the NCCL backend. Experiments are conducted on one node with four NVIDIA H100 GPUs with 80GB memory. We train for up to 10 epochs and select the final checkpoint based on internal validation performance. We optimize with AdamW using β1 = 0.9 and β2 = 0.999. We use group-wise learning rates of 3 × 10−4 for the decoder, segmentation head, and dot-product scoring, 5 × 10−5 for the vision backbone, 5 × 10−5 for the language backbone, and 1 × 10−4 for the geometry prompt encoder. The learning rate schedule uses linear warmup followed by an inverse-square-root decay. Training uses only text prompts paired with ground-truth segmentation masks, without any spatial or interactive prompts such as points or bounding boxes. Model selection is based on performance on the internal validation set. We follow the set-prediction objective in Sec. 3.4 for instance discovery and mask prediction. We use focal Hungarian matching with an auxiliary one-to-many branch to improve assignment stability under sparse supervision and severe foreground–background imbalance. All matching and loss hyperparameters are summarized in Table 2.

During evaluation, we select the highest-confidence mask generated from the text prompt. For multi-class scenarios, we query each class independently and resolve overlaps via pixel-wise maximal confidence, yielding a single nonoverlapping semantic map. This strategy ensures consistent predictions suited for text-only deployment. We report Dice coefficient and Intersection-over-Union (IoU) as primary metrics.

- Table 2. Matching and loss hyperparameters used in the set-prediction objective.

###### Block Param Val. Param Val.

wcls 2.0 wbox 5.0 wgiou 2.0 αmatch 0.25 γmatch 2 stable false

O2O matcher BinaryHungarianMatcherV2

top-k 4 threshold 0.4 αo2m 0.3 λo2m 2.0

O2M matcher BinaryOneToManyMatcher

λce 20.0 λpr 20.0 αcls 0.25 γcls 2 pos. weight 10 padded Nq 200 λℓ1 5.0 λg 2.0

Lfind

αseg 0.6 γseg 2.0 λf 20.0 λd 30.0 λsp 1.0

Lseg

#### 5 Results

CVC-Clinic 87.9

Breast Cancer 43.8

COph100 63.1

HC18 92.6

Intraretinal Fluid 85.0

DRIVE 55.8

ETIS-Larib 86.1

M2CAI 88.1

PS-FH-AOP'23 91.6

TN3K 40.8

PH2 92.7

FetoPlac 77.0

PAPILA 99.4

STARE 54.4

SegThy 78.5

GlaS'15 88.2

CHASE-DB1 62.6

CVC-Clinic 81.2

Breast Cancer 35.7

COph100 46.6

HC18 86.9

Intraretinal Fluid 75.2

DRIVE 39.2

ETIS-Larib 79.3

M2CAI 81.5

PS-FH-AOP'23 84.8

TN3K 32.7

PH2 87.5

FetoPlac 64.3

PAPILA 98.7

STARE 37.8

SegThy 66.2

GlaS'15 80.7

CHASE-DB1 45.7

Medical SAM3 (Ours) SAM3

Fig. 3. Radar chart overview of segmentation performance. Results are split by internal validation (top) and external generalization (bottom), reporting Dice (left) and IoU (right) scores. The red area (Medical SAM3) significantly covers the blue area (SAM3) in all scenarios, aligning with the metrics in Table 3.

Internal validation on held-out splits. Table 3 (top) reports results on 10 internal held-out splits. Medical SAM3 improves over the original SAM3 on all tasks, increasing the average Dice from 54.0% to 77.0% and the average IoU from 43.3% to 67.3%. These gains highlight that full-parameter fine-tuning strengthens medical domain visual priors and improves text-to-mask alignment, enabling reliable localization even when only a class name is provided. The improvements are most pronounced for small, thin, or low-contrast targets where text-only prompting is particularly challenging. For retinal vessel segmentation, performance increases substantially on DRIVE from 24.8% to 55.8% Dice and on COph100 from 34.1% to 63.1% Dice, indicating better boundary adherence for fine vascular structures. We also observe strong gains on modality-specific targets with large appearance shifts, including fetal head segmentation on PS-FHAOP’23 from 65.7% to 91.6 Dice and placental vessel segmentation on FetoPlac from 56.6% to 77.0% Dice. Overall, the consistent gains across all internal heldout splits indicate that Medical SAM3 achieves strong in-domain adaptation under a text-only prompting setting.

- Table 3. Quantitative comparison on internal (10) and external (7) testing datasets. We report Dice and IoU (%).

Dataset Dice (%) IoU (%)

SAM3 Ours ∆ SAM3 Ours ∆ Internal datasets (10)

PS-FH-AOP’23 65.7 91.6 +25.9 50.3 84.8 +34.5 DRIVE 24.8 55.8 +31.0 14.2 39.2 +25.0 COph100 34.1 63.1 +29.0 22.1 46.6 +24.6 Breast Cancer 16.3 43.8 +27.5 11.6 35.7 +24.0 Intraretinal Fluid 62.0 85.0 +23.1 50.4 75.2 +24.8 M2CAI 67.7 88.1 +20.4 54.5 81.5 +27.0 FetoPlac 56.6 77.0 +20.5 42.9 64.3 +21.4 GlaS’15 68.9 88.2 +19.4 59.8 80.7 +21.0 SegThy 57.3 78.5 +21.2 48.4 66.2 +17.8 PAPILA 86.2 99.4 +13.1 78.7 98.7 +20.1

Avg. (Internal) 54.0 77.0 +23.0 43.3 67.3 +24.0 External datasets (7)

TN3K 4.2 40.8 +36.6 3.4 32.7 +29.3 HC18 23.9 92.6 +68.7 17.3 86.9 +69.6 CVC 0.0 87.9 +87.9 0.0 81.2 +81.2 ETIS 0.0 86.1 +86.1 0.0 79.3 +79.3 PH2 18.4 92.7 +74.3 14.9 87.5 +72.6 CHASE 17.9 62.6 +44.7 9.8 45.7 +35.9 STARE 18.6 54.4 +35.8 10.3 37.8 +27.5

Avg. (External) 11.9 73.9 +62.0 8.0 64.4 +56.4

In digital pathology, Medical SAM3 markedly improves breast cancer tissue segmentation from 16.3% to 43.8% Dice and gland segmentation on GlaS’15 from 68.9% to 88.2% Dice, showing robust adaptation to stain and texture variations under the same protocol. On high-contrast targets where the baseline is already strong, Medical SAM3 maintains or further boosts accuracy, with PAPILA reaching 99.4% Dice and 98.7% IoU.

External validation under domain shift. To assess zero-shot generalization, we evaluate Medical SAM3 on seven external datasets that are excluded from training, spanning ultrasound, endoscopy, and fundus photography: TN3K, HC18, CHASE_DB1, STARE, CVC-Clinic, ETIS-Larib, and PH2. Table 3 (bottom) shows consistent improvements over the original SAM3 across all tasks, with average Dice increasing from 11.9% to 73.9% and average IoU rising from 8.0% to 64.4%. The most striking recovery occurs in endoscopic polyp segmentation (CVC and ETIS), where the baseline SAM3 suffers catastrophic failure due to weak text-visual alignment; in contrast, Medical SAM3 successfully grounds the target, achieving 87.9% and 86.1% Dice, respectively. Similarly, in ultrasound (HC18) and dermatology (PH2) tasks, the model overcomes domain gaps

to boost performance by over 68%, proving its capability to reliably localize anatomical structures in unseen domains without additional adaptation.

Image GT SAM 3 Medical SAM3

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Fig. 4. Visualization of the segmentation performance of SAM3 and Medical SAM3

Qualitative Results Figure 4 provides representative visual comparisons between SAM3 and Medical SAM3 under the same text-only prompting protocol. Across diverse modalities, the original SAM3 frequently fails to localize the target anatomy, producing either near-empty masks or severe over-segmentation that collapses to large foreground regions. This behavior is particularly evident for thin and low-contrast structures such as retinal vessels, where SAM3 outputs noisy masks with widespread false positives, while Medical SAM3 recovers fine

vascular branches with substantially cleaner boundaries. Similar improvements are observed on endoscopic targets, where SAM3 tends to miss small regions or yields fragmented predictions, whereas Medical SAM3 produces coherent masks that better match the ground truth. On dermoscopy, Medical SAM3 also delineates lesion extent more accurately and avoids the spurious background activations seen in SAM3.

#### 6 Discussion and Conclusion

Our study reveals that the main bottleneck for universal medical segmentation with promptable foundation models is not the availability of a prompt interface, but the reliability of semantic grounding under domain shift. While strong geometric cues in medical imaging can often simplify segmentation into a boundary refinement task, relying solely on text prompts exposes the more critical challenge: mapping clinical concepts to spatially precise masks across heterogeneous appearances.

The consistent performance gains of Medical SAM3 indicate that robust text grounding is attainable when adaptation is treated as a holistic representation problem rather than merely a prompt-engineering problem. In particular, the improvements observed across diverse modalities point to a shared latent structure that can be learned when the model is forced to align high-level language concepts with localization-relevant visual features. This perspective also helps explain why failures are most visible on small, thin, or low-contrast targets: such cases demand stronger coupling between semantics and spatial evidence, and are less forgiving to misalignment.

These findings have substantial implications for both evaluation and deployment. Benchmarking should explicitly distinguish interactive settings (where users or upstream detectors provide spatial hints) from deployment-consistent semantic-only settings; otherwise, comparisons may be confounded by privileged localization priors. From a systems standpoint, a text-driven interface is attractive precisely because it offers a unified way to query segmentation targets across departments and modalities. However, realizing this promise requires standardized prompt protocols and careful handling of terminology and label granularity, since clinical language is inherently variable.

Despite these advancements, several limitations persist. First, full adaptation

- at high resolution can be computationally demanding, motivating future work on parameter-efficient strategies and distillation without sacrificing robustness. Second, while a planar representation improves universality across inconsistent acquisition geometries, it may underutilize native volumetric continuity; native

- 3D prompting and explicit inter-slice consistency constraints are promising directions. Third, our current evaluation prioritizes atomic concept prompts; extending to synonym-robust, attribute-rich, and compositional prompts will be important for real clinical usage. Finally, broader multi-center validation and reliability analyses, such as uncertainty estimation, are necessary to quantify deployment readiness.

Overall, Medical SAM3 supports a semantic-driven paradigm for universal medical segmentation and highlights that robust promptability in medicine is primarily an alignment and adaptation challenge. Future progress will likely come from combining scalable multi-domain training, richer clinical language handling, and efficiency-oriented adaptation to enable practical and trustworthy deployment.

#### References

- 1. ACOUSLIC Consortium: ACOUSLIC-AI: Abdominal circumference ultrasound image dataset. Grand Challenge (2024)
- 2. Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A.: Dataset of breast ultrasound images. Data in Brief 28, 104863 (2020)
- 3. Ali, S., et al.: PolypGen: A multi-center polyp detection and segmentation dataset for generalisability assessment. Scientific Data 10(1), 75 (2023)
- 4. Araújo, T., Aresta, G., Castro, E., Rouco, J., Aguiar, P., Eloy, C., Polónia, A., Campilho, A.: Classification of breast cancer histology images using convolutional neural networks. PloS one 12(6), e0177544 (2017)
- 5. Bano, S., Casella, A., Vasconcelos, F., Qayyum, A., et al.: FetReg: Fetoscopic placental vessel segmentation and registration. Medical Image Analysis 76, 102330

(2022)

- 6. Candemir, S., Jaeger, S., Palaniappan, K., Musco, J.P., Singh, R.K., Xue, Z., Karargyris, A., Antani, S., Thoma, G., McDonald, C.J.: Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Transactions on Medical Imaging 33(2), 577–590 (2014)
- 7. Carion, N., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)
- 8. Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y.: Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306 (2021)
- 9. Çiçek, Ö., Abdulkadir, A., Lienkamp, S., Brox, T., Ronneberger, O.: 3d u-net: learning dense volumetric segmentation from sparse annotation. In: MICCAI. pp. 424–432. Springer (2016)
- 10. Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al.: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). arXiv preprint arXiv:1902.03368 (2019)
- 11. Ding, Z., Wang, J., Tu, Z.: Maskclip: Mask transformer for open-vocabulary universal image segmentation. In: CVPR (2023)
- 12. Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M., Thrun, S.: Dermatologist-level classification of skin cancer with deep neural networks. Nature 542(7639), 115–118 (2017)
- 13. Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A., Rajpoot, N.: PanNuke: An open pan-cancer histology dataset for nuclei instance segmentation and classification. European Congress on Digital Pathology pp. 11–19 (2019)
- 14. Gómez-Flores, W., Cervantes-Sánchez, F., Escalante-Ramírez, B.: BUS-UCLM: A breast ultrasound dataset for lesion detection and classification. Pattern Recognition Letters 155, 33–40 (2022)

- 15. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023)
- 16. Guan, H., Liu, M.: Domain adaptation for medical image analysis: a survey. IEEE Transactions on Biomedical Engineering 69(3), 1173–1185 (2021)
- 17. Hatamizadeh, A., Hosseini, H., Patel, N., Choi, J., Pole, C.C., Hoeferlin, C.M., Schwartz, S.D., Terzopoulos, D.: RAVIR: A dataset and methodology for the semantic segmentation of retinal arteries and veins in infrared reflectance imaging. IEEE Journal of Biomedical and Health Informatics 26(7), 3272–3283 (2022)
- 18. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R., Xu, D.: Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In: MICCAI. Springer (2022)
- 19. Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R., Xu, D.: Unetr: Transformers for 3d medical image segmentation. In: WACV. pp. 574–584. IEEE (2022)
- 20. He, S., Bao, R., Grant, P.E., Ou, Y.: Accuracy of segment-anything model (sam) in medical image segmentation: a comprehensive evaluation. International Journal of Computer Assisted Radiology and Surgery 19(1), 31–46 (2024)
- 21. van den Heuvel, T.L.A., de Bruijn, D., de Korte, C.L., van Ginneken, B.: Automated measurement of fetal head circumference using 2d ultrasound images. PloS One 13(8), e0200412 (2018)
- 22. Hong, W.c., et al.: CholecSeg8k: A semantic segmentation dataset for laparoscopic cholecystectomy based on CholecT50. In: arXiv preprint arXiv:2012.12453 (2020)
- 23. Huang, Y., Yang, X., Liu, L., Zhou, H., Chang, A., Zhou, X., Chen, R., Yu, J., Chen, J., Chen, C., et al.: Segment anything model for medical images? Medical Image Analysis 92, 103061 (2024)
- 24. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18(2), 203–211 (2021)
- 25. Jha, D., Smedsrud, P.H., Riegler, M.A., Halvorsen, P., de Lange, T., Johansen, D., Johansen, H.D.: Kvasir-SEG: A segmented polyp dataset. In: International Conference on Multimedia Modeling. pp. 451–462. Springer (2020)
- 26. Kaggle: Ultrasound nerve segmentation challenge. https://www.kaggle.com/c/ ultrasound-nerve-segmentation (2016), brachial plexus segmentation in ultrasound images
- 27. Kainz, P., Urschler, M., Schulter, S., Wohlhart, P., Lepetit, V.: You should use regression to detect cells. Medical Image Computing and Computer-Assisted Intervention (MICCAI) pp. 276–283 (2015)
- 28. Kashani, A.H., et al.: Automated intraretinal cystoid fluid segmentation using optical coherence tomography images and deep learning. Translational Vision Science & Technology 10(14), 23 (2021)
- 29. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. In: ICCV. pp. 4015–4026 (2023)
- 30. Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R., Sellés-Navarro, I., PalazónCabanes, A., Sancho-Gómez, J.L.: PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment. Scientific Data 9(1), 291 (2022)
- 31. Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., Sethi, A.: A dataset and a technique for generalized nuclear segmentation for computational pathology. IEEE Transactions on Medical Imaging 36(7), 1550–1560 (2017)

- 32. Levinshtein, A., et al.: UroCell: A dataset for electron microscopy segmentation of urinary bladder cells. Data in Brief 30, 105522 (2020)
- 33. Liang, F., Wu, B., Dai, X., Li, K., Zhao, Y., Zhang, H., Zhang, P., Vajda, P., Marculescu, D.: Open-vocabulary semantic segmentation with mask-adapted clip. In: CVPR (2023)
- 34. Lin, Z., Wei, D., Liao, J., Xu, X., Bhagat, S., et al.: NucMM-Z: A dataset for nuclear segmentation in zebrafish brain using electron microscopy. MICCAI (2021)
- 35. Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., Van Der Laak, J.A., Van Ginneken, B., Sánchez, C.I.: A survey on deep learning in medical image analysis. Medical Image Analysis 42, 60–88 (2017)
- 36. Liu, J., Yang, H., Caverly, H.Y., et al.: Swin-umamba: Mamba-based unet with imagenet-based pretraining. In: MICCAI (2024)
- 37. Liu, Q., Chen, C., Qin, J., Dou, Q., Heng, P.A.: Feddg: Federated domain generalization on medical image segmentation via episodic learning in continuous frequency space. In: CVPR. pp. 1013–1023 (2021)
- 38. Liu, Y., Tian, Y., Wang, C., Chen, Y., Liu, F., Belagiannis, V., Carneiro, G.: Translation consistent semi-supervised segmentation for 3d medical images. IEEE Transactions on Medical Imaging (2024)
- 39. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR. pp. 3431–3440 (2015)
- 40. Lüddecke, T., Ecker, A.: Image segmentation using text and image prompts. In: CVPR. pp. 7086–7096 (2022)
- 41. Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Medsam: Segment anything in medical images. Nature Communications 15, 654 (2024)
- 42. Ma, J., Li, F., Wang, B.: U-mamba: Enhancing long-range dependency for biomedical image segmentation. In: MICCAI (2024)
- 43. Ma, J., Zhu, Y., Wang, B.: Medical sam 2: Segment medical images as video via segment anything model 2. arXiv preprint arXiv:2408.00874 (2025), updated version of MedSAM-2
- 44. Maier-Hein, L., et al.: Can masses of non-experts train highly accurate image classifiers? a crowdsourcing approach to instrument segmentation in laparoscopic images. Medical Image Computing and Computer-Assisted Intervention (MICCAI)

(2014)

- 45. Mazurowski, M.A., Dong, H., Gu, H., Yang, J., Konz, N., Zhang, Y.: Segment anything model for medical image analysis: an experimental study. Medical Image Analysis 89, 102918 (2023)
- 46. Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 3DV. pp. 565–571. IEEE (2016)
- 47. Moreira, I.C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M.J., Cardoso, J.S.: INbreast: Toward a full-field digital mammographic database. Academic Radiology 19(2), 236–248 (2012)
- 48. Ngoc Lan, P., et al.: BKAI-IGH NeoPolyp: A colonoscopy dataset for colorectal polyp detection and segmentation. IEEE Access 9, 163026–163039 (2021)
- 49. Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B., et al.: Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999 (2018)
- 50. PCMMD Consortium: PCMMD: Plasma cell multiple myeloma dataset for cell detection and segmentation. Scientific Data (2025), microscopy dataset for plasma cell segmentation

- 51. Popescu, D., Diaconu, A.M., Deac, A., Stanciu, O., Dogaru, R., Bacila, C.: ARCADE: Automatic region-based coronary artery disease diagnostics using x-ray angiography images. Medical Image Analysis 83, 102636 (2023)
- 52. Rao, Y., Zhao, W., Liu, G., Lu, J., Zhou, J.: Denseclip: Language-guided dense prediction with context-aware prompting. In: CVPR. pp. 18082–18091 (2022)
- 53. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)
- 54. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI. pp. 234–241. Springer (2015)
- 55. Ruan, J., Xiang, S.: Vm-unet: Vision mamba unet for medical image segmentation. arXiv preprint arXiv:2402.02491 (2024)
- 56. Shao, Y., He, H., Li, S., Chen, S., Long, X., Zeng, F., Fan, Y., Zhang, M., Yan, Z., Ma, A., et al.: Eventvad: Training-free event-aware video anomaly detection. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 2586–2595 (2025)
- 57. Shao, Y., Liang, S., Ling, Z., Yan, M., Liu, H., Chen, S., Yan, Z., Zhang, C., Qin, H., Magno, M., et al.: Gwq: Gradient-aware weight quantization for large language models. arXiv preprint arXiv:2411.00850 (2024)
- 58. Shao, Y., Lin, D., Zeng, F., Yan, M., Zhang, M., Chen, S., Fan, Y., Yan, Z., Wang, H., Guo, J., et al.: Tr-dq: Time-rotation diffusion quantization. arXiv preprint arXiv:2503.06564 (2025)
- 59. Shao, Y., Lin, X., Long, X., Chen, S., Yan, M., Liu, Y., Yan, Z., Ma, A., Tang, H., Guo, J.: Icm-fusion: In-context meta-optimized lora fusion for multi-task adaptation. arXiv preprint arXiv:2508.04153 (2025)
- 60. Shao, Y., Xu, Y., Long, X., Chen, S., Yan, Z., Liu, H., Wang, Y., Tang, H., Yang, Y.: Accidentblip: Agent of accident warning based on ma-former. In: 2025 IEEE Intelligent Vehicles Symposium (IV). pp. 2156–2161. IEEE (2025)
- 61. Shao, Y., Yan, M., Liu, Y., Chen, S., Chen, W., Long, X., Yan, Z., Li, L., Zhang, C., Sebe, N., et al.: In-context meta lora generation. arXiv preprint arXiv:2501.17635

(2025)

- 62. Shen, D., Wu, G., Suk, H.I.: Deep learning in medical image analysis. Annual Review of Biomedical Engineering 19, 221–248 (2017)
- 63. Sirinukunwattana, K., Pluim, J.P., Chen, H., Qi, X., Heng, P.A., Guo, Y.B., Wang, L.Y., Matuszewski, B.J., Brber, E., et al.: Gland segmentation in colon histology images: The GlaS challenge contest. Medical Image Analysis 35, 489–502 (2017)
- 64. Sirinukunwattana, K., Raza, S.E.A., Tsang, Y.W., Snead, D.R., Cree, I.A., Rajpoot, N.M.: Locality sensitive deep learning for detection and classification of nuclei in routine colon cancer histology images. IEEE Transactions on Medical Imaging 35(5), 1196–1206 (2016)
- 65. Society for Imaging Informatics in Medicine: SIIM-ACR pneumothorax segmentation challenge. In: Kaggle Competition (2019)
- 66. Song, S., et al.: CT2US: Cross-modal supervision for abdominal organ segmentation. Medical Image Analysis 82, 102603 (2022)
- 67. Staal, J., Abramoff, M.D., Niemeijer, M., Viergever, M.A., Van Ginneken, B.: Ridge-based vessel segmentation in color images of the retina. IEEE Transactions on Medical Imaging 23(4), 501–509 (2004)
- 68. Tahir, A.M., Chowdhury, M.E.H., Khandakar, A., Rahman, T., Qiblawey, Y., Khurshid, U., Kiranyaz, S., Ibtehaz, N., Rahman, M.S., Al-Maadeed, S., et al.: COVID-QU-Ex dataset: A large-scale collection of covid-19, non-covid, and lung opacity chest x-ray images. Informatics in Medicine Unlocked 30, 100893 (2022)

- 69. Tian, Y., Liu, F., Pang, G., Chen, Y., Liu, Y., Verjans, J.W., Singh, R., Carneiro, G.: Self-supervised pseudo multi-class pre-training for unsupervised anomaly detection and segmentation in medical images. Medical image analysis 90, 102930

(2023)

- 70. Tian, Y., Pang, G., Liu, F., Chen, Y., Shin, S.H., Verjans, J.W., Singh, R., Carneiro, G.: Constrained contrastive distribution learning for unsupervised anomaly detection and localisation in medical images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 128–140. Springer (2021)
- 71. Tian, Y., Pang, G., Liu, Y., Wang, C., Chen, Y., Liu, F., Singh, R., Verjans, J.W., Wang, M., Carneiro, G.: Unsupervised anomaly detection in medical images with a memory-augmented multi-level cross-attentional masked autoencoder. In: International workshop on machine learning in medical imaging. pp. 11–21. Springer

(2023)

- 72. Tian, Y., Shi, M., Luo, Y., Kouhana, A., Elze, T., Wang, M.: Fairseg: A large-scale medical image segmentation dataset for fairness learning using segment anything model with fair error-bound scaling. arXiv preprint arXiv:2311.02189 (2023)
- 73. Tian, Y., Wen, C., Shi, M., Afzal, M.M., Huang, H., Khan, M.O., Luo, Y., Fang, Y., Wang, M.: Fairdomain: Achieving fairness in cross-domain medical image segmentation and classification. In: European Conference on Computer Vision. pp. 251–271. Springer (2024)
- 74. Wang, G., Zuluaga, M.A., Li, W., Pratt, R., Patel, P.A., Aertsen, M., Doel, T., David, A.L., Deprest, J., Ourselin, S., Vercauteren, T.: Deepigeos: A deep interactive geodesic framework for medical image segmentation. IEEE TPAMI 41(7), 1559–1572 (2018)
- 75. Wang, H., Guo, S., Ye, J., Deng, Z., Ren, Y., Li, Y., Wan, X.: Sam-med3d. arXiv preprint arXiv:2310.15161 (2023)
- 76. Wang, X., et al.: BTXRD: Bone tumor x-ray dataset for detection and segmentation. Scientific Data (2025), bone tumor X-ray dataset
- 77. Wang, Z., Lu, Y., Li, Q., Tao, X., Guo, Y., Gong, M., Liu, T.: Cris: Clip-driven referring image segmentation. In: CVPR. pp. 11686–11695 (2022)
- 78. Wasserthal, J., Breit, H.C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D.T., Cyriac, J., Yang, S., et al.: Totalsegmentator: Robust segmentation of 104 anatomic structures in ct images. Radiology: Artificial Intelligence 5(5) (2023)
- 79. Wu, J., Fu, R., Fang, H., Liu, Y., Wang, Z., Xu, Y., Jin, Y., Arbel, T.: Medical sam adapter: Adapting segment anything model for medical image segmentation. Medical Image Analysis 102, 103547 (2025)
- 80. Wu, J., Min, X.: One-prompt to segment all medical images. In: CVPR (2024)
- 81. Wunderling, T., Golla, B., Poudel, P., Taber, C., Gockel, M., Modersitzki, J.: SegThy: A novel dataset for thyroid segmentation in ultrasound images. International Journal of Computer Assisted Radiology and Surgery 12(8), 1405–1414

(2017)

- 82. Xing, Z., Ye, T., Yang, Y., Liu, G., Zhu, L.: Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation. In: MICCAI (2024)
- 83. Yan, Z., Dong, W., Shao, Y., Lu, Y., Liu, H., Liu, J., Wang, H., Wang, Z., Wang, Y., Remondino, F., et al.: Renderworld: World model with self-supervised 3d label. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 6063–6070. IEEE (2025)

- 84. Yan, Z., Li, L., Shao, Y., Chen, S., Wu, Z., Hwang, J.N., Zhao, H., Remondino, F.: 3dsceneeditor: Controllable 3d scene editing with gaussian splatting. arXiv preprint arXiv:2412.01583 (2024)
- 85. Yang, Z., Wang, J., Tang, Y., Chen, K., Zhao, H., Torr, P.H.: Lavt: Language-aware vision transformer for referring image segmentation. In: CVPR. pp. 18155–18165

(2022)

- 86. Ye, Y., Chen, Z., Zhang, J., Xie, Y., Xia, Y.: Meduniseg: 2d and 3d medical image segmentation via a prompt-driven universal model. arXiv preprint arXiv:2410.05905 (2024)
- 87. Ye, Y., Xie, Y., Zhang, J., Chen, Z., Xia, Y.: Uniseg: A prompt-driven universal segmentation model as well as a strong representation learner. In: MICCAI. pp. 508–518. Springer (2023)
- 88. Zhang, X., et al.: COph100: A comprehensive ophthalmic fundus image dataset for deep learning. Scientific Data (2023), fundus image dataset for retinal vessel segmentation
- 89. Zhao, Z., et al.: One model to rule them all: Towards universal segmentation for medical images with text prompts. arXiv preprint arXiv:2312.17183 (2025), also referred to as SAT
- 90. Zhou, H.Y., Guo, J., Zhang, Y., Yu, L., Wang, L., Yu, Y.: nnformer: Interleaved transformer for volumetric segmentation. arXiv preprint arXiv:2109.03201 (2021)
- 91. Zhou, Z., Rahman Siddiquee, M.M., Tajbakhsh, N., Liang, J.: Unet++: A nested u-net architecture for medical image segmentation. In: MICCAI. pp. 3–11. Springer

(2018)

Table 1. Summary of datasets used for training. The table reports the number of images and annotations across distinct medical modalities, and the median annotation area represents the typical scale of the segmentation targets.

Dataset Images Anns Median Area (px) CXR (3 datasets)

COVID-QU-Ex 33,920 67,839 7,542 Chest Xray Masks and Labels 704 1,415 1,022,508 Chest X-Ray Pneumothorax 290 370 7,183

X-ray (2 datasets)

BTXRD 3,746 2,273 14,746 ARCADE 1,500 2,316 1,472

###### Ultrasound (7 datasets)

BUSI 647 647 17,348 BUS-UCLM 264 281 24,102 US-Nerve 2,323 2,323 6,954 ACOUSLIC 300 300 58,330 ps-fh-aop-2023 4,000 3,999 8,048 CT2USforKidneySeg 4,586 4,601 11,807 SegThy 59 130 2,118

###### Endoscopy (6 datasets)

CholecSeg8k 8,080 112,521 749 m2caiSeg 614 804 102,493 PolypGen 1,710 2,003 70,698 BKAI-IGH NeoPolyp 1,000 1,117 39,138 Kvasir-SEG 1,000 1,060 34,489

###### Pathology (4 datasets)

PanNuke 2,540 21,978 811 MonuSeg 82 2,887 172 Breast Cancer Segmentation 151 3,495 15,524 GlaS@MICCAI’2015 165 1,538 10,904

###### Virtual Microscope (2 datasets)

MUCIC Colon Tissue 60 12,396 1,688 MUCIC HL60 Granulocytes 240 3,987 4,857

###### Electron Microscopy (2 datasets)

NucMM-Z 62 581 58 UroCell 5 163 241

Microscopy (1 dataset) PCMMD 3,517 3,519 26,227

###### Fundus/OCT (5 datasets)

Intraretinal Cystoid Fluid 1,459 4,601 202 PAPILA 488 488 172,486 COph100 324 324 7,168 RAVIR Dataset 23 141 3,638 DRIVE 20 20 28,176

###### Dermoscopy & Others (2 datasets)

ISIC_2018 2,594 2,594 429,033 FetoPlac 483 994 6,456

###### Total 76,956 263,705 –

