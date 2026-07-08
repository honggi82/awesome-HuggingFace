## arXiv:2511.15186v2[cs.CV]26Mar2026

### Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset

Geon Choi1,† Hangyul Yoon1,† Hyunju Shin2 Hyunki Park2 Sang Hoon Seo2 Eunho Yang1,3 Edward Choi1,* 1KAIST 2Samsung Medical Center 3AITRICS

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Figure 1. Examples of the instruction-guided CXR lesion segmentation task. Given text instructions for various lesion types and locations of interest, ROSALIA, a VLM trained on our MIMIC-ILS dataset, can: (A) segment lesions in a specified location, (B) segment lesions globally, and (C) detect empty-target cases. As can be seen in (A), ROSALIA correctly ignores the unrequested lesion in the left lung.

#### Abstract

The applicability of current lesion segmentation models for chest X-rays (CXRs) has been limited both by a small number of target labels and the reliance on complex, expert-level text inputs, creating a barrier to practical use. To address these limitations, we introduce instruction-guided lesion segmentation (ILS), a medical-domain adaptation of referring image segmentation (RIS) designed to segment diverse lesion types based on simple, user-friendly instructions. Under this task, we construct MIMIC-ILS, the first large-scale instructionanswer dataset for CXR lesion segmentation, using our fully automated multimodal pipeline that generates annotations from CXR images and their corresponding reports. MIMICILS contains 1.1M instruction-answer pairs derived from 192K images and 91K unique segmentation masks, covering seven major lesion types. To empirically demonstrate its utility, we present ROSALIA, a LISA model fine-tuned on the MIMIC-ILS dataset. ROSALIA can segment diverse lesions and provide textual explanations in response to user instructions. The model achieves high accuracy in our newly proposed task, highlighting the effectiveness of our pipeline and the value of MIMIC-ILS as a foundational resource for pixel-level CXR lesion grounding. The dataset and model are available at https://github.com/checkoneee/ROSALIA.

#### 1. Introduction

Medical imaging is an essential technique in modern medicine, enabling accurate diagnosis and appropriate treatment. Among various imaging modalities, chest X-ray (CXR) is one of the most common examinations due to its high accessibility and rapid acquisition [4]. Radiologists reach a diagnosis by integrating visual evidence from CXRs with their clinical knowledge, and describe these findings in a text format known as a radiology report. A key step in this diagnostic process is identifying the precise location and boundary of a lesion—an abnormal region with pathological changes [6]. This task is labor-intensive and demands substantial clinical expertise and analytical precision.

To alleviate physicians’ workload in localizing pathological regions, there is a growing demand for automated lesion segmentation models in CXRs. Recently, vision–language models (VLMs) equipped with segmentation modules [24, 25, 40] have emerged as a promising solution for referring image segmentation (RIS), as they can interpret diverse userspecific needs expressed through natural language instructions. However, despite the success of such VLMs in generaldomain RIS, their application to CXRs remains limited. Although prior studies [16, 29] have explored CXR lesion segmentation using text prompts, they are limited to a single

†Equal Contribution *Correspondence to

Table 1. Existing CXR datasets with spatial annotations for pathologic lesions.

Spatial Annotation

Dataset # Images

Instruction-Answer Pair # Annotations Type Multi-Lesion Method

VinDr-CXR [37] 15K 9K Bounding Box ✓ Manual ✗ Padchest-GR [8] 4.6K 7.7K Bounding Box ✓ Manual ✗ MS-CXR [3] 1K 1.2K Bounding Box ✓ Manual ✗ TBX-11K [32] 12K 1.2K Bounding Box ✗ Manual ✗ SIIM-ACR [46] 13K 2.7K Segmentation Mask ✗ Manual ✗ QaTa-COV19 [9] 121K 9.3K Segmentation Mask ✗ Semi-Automated ✗ Danilov et al. [7] 1.4K 0.6K Segmentation Mask ✓ Manual ✗

MIMIC-ILS (Ours) 192K 91K Segmentation Mask ✓ Fully-Automated ✓

lesion type (e.g., COVID-19) and moreover require long, detailed expert-level medical descriptions based on tailored CXR review (e.g., “Bilateral pulmonary infection, two infected areas, upper right lung and upper left lung.”) as input. Such constraints make them impractical not only for physicians who aim to segment diverse lesion types across various anatomical subregions before closely reviewing the image themselves, but especially for non-experts who can hardly interpret CXR images at all.

To address these limitations, we propose a more userfriendly task, namely instruction-guided lesion segmentation (ILS). In this task, the model is required to process diverse user instructions, ranging from prompts that specify the lesion type and target location, to requests that look for abnormalities globally. If the requested lesion is not present, the model should reliably report its absence. Additionally, the model should be able to provide textual descriptions regarding a lesion’s location or type, even if not explicitly prompted by the user. However, a dataset to support such a versatile task has been unavailable, as constructing a suitable dataset for training and evaluation poses significant challenges—most notably the need for expert-curated mask annotations. Moreover, accurately pairing these masks with precise textual instructions in terms of anatomical locations and specific lesion types remains a highly complex task.

In this work, we introduce the first fully automated pipeline for constructing a large-scale ILS dataset for CXRs. The central challenge is: “How can we derive lesion masks and corresponding instruction-answer text pairs from raw images that contain no explicit annotations?” To address this, we leverage radiology reports as a key source of information for each image. Using paired image–report data, our two-stage pipeline integrates pre-trained vision models and large language models (LLMs) to extract high-confidence anomalous regions and structured textual information. By exploiting the consistency between these heterogeneous modalities, we generate high-quality lesion masks and diverse instruction–answer pairs. Applying our novel framework to MIMIC-CXR [21, 22]—a large, publicly available CXR–report dataset—we constructed MIMIC-ILS, a largescale dataset consisting of 1.1M samples derived from 192K images and 91K lesion masks (Table 1).

Although several datasets [3, 7–9, 32, 37, 46] have tried to introduce spatial annotations in the CXR domain, they are unsuitable for direct use in our ILS task (Table 1). Most provide only coarse bounding-box localization or single lesion type masks that are limited in scale due to reliance on expert annotations. Moreover, they also lack explicit links between mask annotations and textual instructions. MIMIC-ILS bridges these gaps by offering large-scale instruction–answer pairs, each paired with an auto-labeled segmentation mask and a detailed lesion profile. Despite being constructed entirely without human intervention, expert evaluations report a high acceptance rate of over 95% for this dataset.

Leveraging MIMIC-ILS, we fine-tune LISA [24] to develop ROSALIA (RadiOlogy Segmentation Assistant trained on a Lesion-grounded Instruction-Answer dataset), the first VLM designed for ILS task in CXRs. Given simple and concise user instructions, ROSALIA generates segmentation masks and textual descriptions (Fig. 1). It supports a wide range of tasks, including specific segmentation (e.g., “Segment the pneumonia in the right lung.”), generic segmentation (e.g., “Segment the opacity.”), and absence confirmation (e.g., “There is no atelectasis in the left lung base.”). This flexibility enables ROSALIA to effectively address diverse user needs, delivering tailored outputs for each request.

In summary, our contributions are threefold:

- • We introduce a novel automated pipeline that generates lesion masks and corresponding instructions directly from CXRs without any human intervention. Using only image–report pairs, our method produces a large-scale dataset without requiring explicit manual processing.
- • Applying our framework to MIMIC-CXR, we construct MIMIC-ILS, the first dataset for instruction-guided lesion segmentation (ILS) in CXRs. The resulting dataset is further validated by medical experts, confirming its high quality and the reliability of the construction process.
- • To validate the utility of MIMIC-ILS, we introduce ROSALIA, the first VLM designed for ILS in CXRs. Trained on our million-scale dataset, ROSALIA interprets user instructions across diverse lesion types and locations, producing accurate lesion masks and descriptive outputs. These results demonstrate the value of our dataset and model in advancing fine-grained lesion grounding for CXR analysis.

#### 2. Related Work

##### 2.1. Lesion Segmentation and Datasets

Lesion segmentation aims to generate masks corresponding to abnormal regions in medical images. Typically, models are trained on datasets where radiologists have directly annotated lesion masks. For CT and MRI, several studies [18, 19] have utilized public datasets that provide diverse tumor masks [1, 2, 14]. In contrast, such pixel-level annotations are scarce in the CXR domain. While some datasets provide only bounding boxes [8, 37], those that offer segmentation masks usually focus on a single lesion type [9, 32, 46]. Consequently, existing models trained on these datasets are limited in their effective segmentation range for CXRs [47]. Our work directly addresses this gap by constructing a comprehensive, multi-type lesion segmentation dataset for CXRs.

##### 2.2. Referring Image Segmentation

Referring image segmentation (RIS) is the task of segmenting a target specified by text. Early approaches to this task focused on aligning image features with text labels to generate corresponding masks [28, 42, 45, 48]. More recently, advancements in VLMs have enabled researchers to extend their reasoning capabilities to segmentation [24, 25, 40]. These models can generate an appropriate mask based on complex instructions that require real-world knowledge, such as “Segment the object richest in vitamin C in this photo.”

Similar research has emerged in the medical domain, but current approaches remain limited. They usually rely on simple prompts including class labels (e.g., “a computerized tomography of a tumor”) [5, 31], which cannot handle sentence-level instructions. In the CXR domain specifically, recent VLMs have been trained using free-form text that describes the location and number of lesions [16, 29]. These approaches, however, expect users to have already reviewed the CXR image, thus providing expert-level descriptions as input. In contrast, our model allows users to obtain the lesion mask, its presence or absence, and type information even without having to interpret the CXR images first.

#### 3. Automatic Dataset Construction

This section outlines our approach to automatically constructing a large-scale dataset for training a model that generates both lesion segmentation masks and corresponding textual descriptions in response to user instructions. The main challenges in this process are: (1) generating lesion masks directly from raw CXR images without explicit image annotations, (2) aligning appropriate instruction–answer texts with the obtained masks, and (3) ensuring that the entire pipeline operates in a fully automated, human-free manner. To address these challenges, our framework first extracts textual and spatial information from image–report pairs and generates lesion masks followed by a verification process (Sec. 3.1). Using the verified lesion masks, we then construct diverse

instruction–answer pairs (Sec. 3.2).

##### 3.1. Grounded Lesion Mask Generation

To construct our dataset, we use MIMIC-CXR [21, 22], a large collection of CXR images paired with radiology reports. Each report is written by a radiologist and provides visual descriptions of the corresponding CXR image. Based on this dataset, we generate grounded lesion masks through four sequential steps as illustrated in Fig. 2: (1) Report structuring and location mapping; (2) Spatial information extraction; (3) Lesion mask generation; (4) Location verification. The details of each step are provided in Appendix A.

Report Structuring and Location Mapping. The first step employs LLMs to convert radiology reports into a structured form for later steps. Specifically, we instruct an LLM to transform each sentence describing an abnormal finding into a six-element tuple consisting of the following categories: entity, sentence index, presence, certainty, location, and predicted lesion type. The location element is then mapped to one or more anatomical labels to ensure compatibility with the segmentation model used in subsequent processes. For example, if the second sentence in a given radiology report is “The lower lung opacity is pneumonia.”, its corresponding output is (opacity, 2, positive, definitive, [right lung base, left lung base], pneumonia). Here, the term “lower lung” in the original report is mapped to “right lung base” and “left lung base”.

Spatial Information Extraction. The second step extracts spatial information from CXRs using three distinct models:

- (1) RadEdit [38], a diffusion-based image editing model;
- (2) CXAS [41], an anatomy segmentation model; and (3) a pretrained YOLO model for CXR lesion detection [36]. These models are used respectively to generate an anomaly map, anatomy masks, and lesion box masks, which serve as visual cues for lesion mask generation in the subsequent steps.

RadEdit takes an input image x ∈ RH×W containing a lesion and the text prompt “No acute cardiopulmonary process” and outputs an edited image xˆ from which the lesion has been removed. We derive xano ∈ [0,1]H×W as:

x − xˆ Imax

xano =

,

where Imax is the maximum possible pixel intensity (i.e., 255 for an 8-bit image). xano provides morphological information about hyperintense lesions, which are areas that appear brighter than the normal lung field. From xano, we define anomaly map A as:

A = {(i,j) | (xano)i,j ≥ τano}, (1)

where (i,j) represents a pixel coordinate and τano is a threshold for anomaly pixels.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 2. An overview of grounded lesion mask generation. (Top-left) Textual information is extracted from the radiology report during the report structuring and location mapping. (Bottom-left and Center) Pretrained vision models are also employed to produce spatial information. (Right) Finally, a lesion mask is generated by integrating this information. The verification step then confirms the grounded location (l1), identifies the empty location (l3) for negative sample generation, and discards the reported-but-ungrounded location (l2).

CXAS produces anatomy masks corresponding to each location element in the previously derived structured report tuples. We denote these masks as coordinate sets {Mi}ni=1, where n is the number of anatomical labels mapped in the previous step. Each Mi contains the pixel coordinates for a specific anatomy, serving as a spatial approximation of the lesion location mentioned in the radiology report.

(c1) sufficient overlap with {Mi}ni=1; (c2) a high confidence score; (c3) a high internal signal ratio from A (i.e., the ratio of the intersection area between the box mask and A to the area of the box mask); and (c4) a sufficient size relative to either Lr or Ll. Conditions c1 and c2 ensure that the boxes align with the reported locations and are likely to correspond to true lesions. However, the A can contain false negatives (i.e., coordinates that belong to actual lesion areas but are missing from A), which may result in excessively small or even empty masks. To mitigate this issue, conditions c3 and c4 are used to retain only those boxes that contain strong lesion signals and are large enough to allow meaningful segmentation. Once the appropriate lesion box masks are selected, we extract from A the connected components (i.e., the individual, contiguous ‘islands’ in 2D space) that intersect with these selected masks. This component then undergoes a post-processing step involving small, noisy mask removal to produce the final, refined lesion mask Mlesion (see Appendix A.5 for further details).

In parallel, the pretrained YOLO model is applied to x to detect a diverse range of lesions. It outputs bounding boxes that not only specify the locations of potential lesions but also assign a confidence score to each detection. From these results, we construct a set of lesion box masks, {Bj}mj=1, where m denotes the number of detected boxes. Each Bj represents the pixel coordinates enclosed by a bounding box, accompanied by a confidence score conf B

###### ∈ [0,1].

j

Lesion Mask Generation. With the three visual cues extracted from the previous step, the initial lesion masks can be generated. Here, the anomaly map A plays a central role, representing a composite signal of all hyperintense lesions. We decompose this signal into individual masks and align them with the specific lesions described in the report. During this process, the anatomy masks {Mi}ni=1, lesion box masks {Bj}mj=1, and the right and left lung masks (Lr and Ll) are jointly used to select high-quality mask candidates.

Location Verification. In the final step, we explicitly verify whether each lesion mask generated by Algorithm 1 has been successfully grounded to the structured report. To assess the grounding status, we define three types of locations: reported location, grounded location and empty location. The reported location is a set of anatomical labels extracted from the previous location mapping with LLMs. Based on this set, the grounded location is defined as a subset of the reported

The core of this filtering process, outlined in Algorithm 1, selectively retains only appropriate candidates from the initially detected lesion box masks, based on four conditions:

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 3. Instruction–answer pair generation process using the example report, “Bibasilar atelectasis. Cardiomegaly.” We utilize the elements extracted from the previous lesion mask generation process (see Fig. 2), indicated by the dashed box. Structured tuples (A&B in the top left) are converted to text instructions and mapped to their corresponding ground-truth masks and textual descriptions. Invalid instructions for lesions which lack a corresponding mask are excluded (colored as red), and only valid instructions are retained (colored as green). (ET: entity, PS: presence, CT: certainty, RL: reported location, GL: grounded location, EL: empty location)

Algorithm 1: Lesion Mask Generation

Input: Anomaly map A, anatomy masks {Mi}ni=1, lesion

box masks {Bj}mj=1 with confidences {conf B

}mj=1, right lung mask Lr, left lung mask Ll

j

Output: Final lesion mask Mlesion

- 1 Mlesion ← ∅;
- 2 Munion ← ni=1 Mi;
- 3 foreach Bj ∈ {Bj}mj=1 do

- 4 c1 ←

|Bj ∩ Munion| |Bj ∪ Munion|

≥ τanatomy;

- 5 c2 ← conf B

j

≥ τconf;

- 6 c3 ←

|Bj ∩ A| |Bj|

≥ τsignal;

- 7 c4 ←

|Bj ∩ Lr| |Bj ∪ Lr|

≥ τsize ∨

|Bj ∩ Ll| |Bj ∪ Ll|

≥ τsize ;

- 8 if c1 ∧ c2 ∧ c3 ∧ c4 then

- 9 C ← FindIntersectingComponent(Bj, A);
- 10 if C is not empty then

- 11 Mnew ← Refine(C);
- 12 Mlesion ← Mlesion ∪ Mnew;

- 13 return Mlesion

location that spatially overlaps with a generated lesion mask, confirming successful localization of the reported finding.

This location is derived from the anatomy masks {Mi}ni=1 that intersect with the selected lesion box masks during the lesion mask generation. Finally, we introduce an empty location, which refers to a lung region with no reported lesions and is used to generate negative samples.

##### 3.2. Instruction-Answer Pair Generation

With the information extracted from the previous process (i.e., grounded lesion mask generation), we build our dataset for seven major lesion types found in CXRs: cardiomegaly, pneumonia, atelectasis, opacity, consolidation, edema, and effusion. These lesions are not only the most frequently mentioned in radiology reports, but also clinically significant to be common annotation targets in other medical datasets [3, 21, 22, 43]. For each lesion, we construct positive instruction-answer pairs, which include a ground-truth lesion mask. Negative pairs using an empty mask are also generated to enable the model to confirm the absence of lesions. An example of this pair generation process is shown in Fig. 3. Please refer to Appendix B and C for the lesion descriptions and specific dataset generation process.

Instruction Types and Limitations. We consider three types of segmentation instructions (Table 2). A basic instruction specifies both the segmentation target and its location. The location can be a broad region (such as left lung or right lung), one of eight more specific zones (apical, upper, mid, and lower zones for each lung), or a combination of these regions. In contrast, a global instruction specifies only the segmentation target. A lesion inference instruction asks the model to predict the type of lesion represented by an opacity within a given location. The generation of these instructions is inherently constrained by the grounded lesion mask generation. For example, a global instruction becomes invalid if

the generated mask captures only part of the lesion. To address this, our framework dynamically produces only those instruction–answer pairs that are valid given the grounding information available for each image.

Table 2. Templates for each question type. Each type includes answer templates for both positive and negative cases, with the negative answers positioned in the last row of each cell.

Type Role Template

Instruction Segment the [Target] in the [Location]. Answer

Basic

[SEG] [SEG] There is no [Target] in the [Location].

Instruction Segment the [Target]. Answer

Global

[SEG] It is located in the [Location]. [SEG] There is no [Target].

Segment the opacity in the [Location] and predict its type.

Instruction

Lesion Inference

[SEG] It is highly suggestive of [Lesion]. [SEG] It possibly reflects [Lesion]. [SEG] There is no opacity in the [Location].

Answer

Instruction Generation. The instruction generation process begins by creating a basic instruction for each grounded lesion. Next, we determine whether a global instruction can be generated. The global instruction is created only when the grounded location and the reported location are identical. Separately, we generate lesion inference instructions by transforming the basic instructions for pneumonia, atelectasis, and edema, replacing these specific lesion types with opacity. This transformation is motivated by the fact that these findings are all specific types of “opacity,” a more fundamental visual concept in medical imaging. Negative samples are generated by (1) selecting lesion types that are not mentioned or explicitly negated in the radiology report; or (2) utilizing empty locations to substitute the original location in the basic instruction of a positive sample.

Answer Generation. Each answer consists of a lesion mask and a textual description. The answer lesion masks for positive pairs are determined differently depending on whether they are organ-level or localized abnormalities. For cardiomegaly, we utilize a heart mask as its corresponding lesion mask since this condition is defined by the state of a specific organ [12]. In contrast, localized abnormalities (e.g., pneumonia or effusion) can appear in variable locations, so for these findings, we use the lesion masks generated in Sec. 3.1. For negative pairs, an empty mask is used. As for the textual description, it is also provided for both positive and negative samples. Specifically, the answer template for lesion inference incorporates a certainty level.

#### 4. MIMIC-ILS Dataset

Our final dataset, MIMIC-ILS, consists of 1.1M instructionanswer pairs (135K positive and 930K negative samples) derived from 192K MIMIC-CXR images. This final image

set is obtained by first filtering out low-quality images (e.g., images with extreme contrast issues), and then excluding any images for which no instruction-answer pairs are generated through our pipeline in Sec. 3. The positive samples are generated from 91K unique lesion masks, where each mask can be associated with multiple instruction–answer samples. The resulting dataset covers seven distinct lesion types, and the overall statistics are presented in Fig. 4. Following the official MIMIC-CXR split, the dataset is divided into 1M training samples, 8.2K validation samples, and 12K test samples. Details on quality control and a distribution of MIMIC-ILS are presented in Appendix D and E.

180K

Positive

160K

151K 148K 148K 154K 151K 151K

Negative

140K

120K

100K

80K

60K

40K

40K

34K

24K

21K

20K

14K

10K

9K

3K

0

CA PN AT OP CO ED EF

Figure 4. Distribution of MIMIC-ILS dataset. The y-axis indicates the number of samples, and the x-axis represents the lesion type. (CA: cardiomegaly, PN: pneumonia, AT: atelectasis, OP: opacity, CO: consolidation, ED: edema, EF: effusion)

Human Evaluation. To assess the quality of MIMIC-ILS, an expert review was conducted by four radiation oncologists specializing in lesion contouring on medical images. For the test set samples, clinicians classified each case as either acceptable or unacceptable based on mask quality. Positive cases were reviewed by all experts, while negatives were split among them. Any sample judged unacceptable by at least one expert was excluded from the final test set, and the results are summarized in Table 3. Among the 10.7K mask samples initially reviewed, 96.4% were rated as acceptable and finally included in the test set. More details on the expert profiles and quality assessment are provided in Appendix E.

Table 3. Acceptance rate and number of evaluated samples for the human evaluation. Each sample corresponds to a unique combination of lesion mask, target, and location.

Total Positive Negative Rate (%) # Samples Rate (%) # Samples Rate (%) # Samples

Expert

- Expert A 96.1 4,090 95.6 1,841 96.5 2,249

- Expert B 97.2 4,028 96.0 1,841 98.3 2,187

- Expert C 98.7 4,041 99.8 1,841 97.8 2,200

- Expert D 97.6 4,065 96.9 1,841 98.2 2,224 Overall 96.4 10,701 90.1 1,841 97.7 8,860

#### 5. Model Training

Using MIMIC-ILS, we train our ILS model, ROSALIA. The model adopts the architecture of LISA [24], which demonstrated strong zero-shot language-guided segmentation performance in the general domain. As illustrated in Fig. 5, the

architecture integrates a VLM backbone with the Segment Anything Model (SAM) [23]. The VLM processes both the image and the input instruction to produce a special token, [SEG], along with its textual description. This [SEG] token embedding is then passed to SAM together with the input image for mask prediction. Within SAM, the frozen image encoder extracts embeddings from the image, and the mask decoder integrates these embeddings with the hidden embedding of [SEG] token to generate the final mask.

[Figure 19]

[Figure 20]

- Figure 5. Overview of ROSALIA. The architecture integrates a VLM with the SAM. The VLM takes a CXR image and a segmentation instruction as input, generating both a textual description and a special [SEG] token. The hidden embedding of this [SEG] token is then passed to SAM’s decoder to produce the final mask.

The overall loss function L consists of two components:

(1) a language loss and (2) a mask loss. It is formulated as: L = λtxtLtxt + Lmask,

Lmask = λbceLbce + λdiceLdice.

Ltxt denotes the autoregressive cross-entropy loss for the answer text, and Lmask represents the segmentation loss computed between the ground-truth mask and the predicted foreground probability map, which combines the binary crossentropy loss Lbce and the DICE [34] loss Ldice. The λtxt, λbce, and λdice are coefficients for each loss term.

#### 6. Experiments

##### 6.1. Implementation Details

Training Details. ROSALIA is built on the LISA-7B architecture and is fine-tuned from its original checkpoint [24]. Following LISA, we adopted LLaVA [30] as the VLM backbone and employed the largest version of SAM (SAM-H). LoRA [15] fine-tuning was applied to the VLM with a rank of 128 and an alpha of 256, while the mask decoder was fully fine-tuned. The epochs and the initial learning rate were set to 15 and 0.0003, respectively, using the AdamW optimizer [33]. The total batch size was 256, and the ratio of positive to negative samples was maintained at 1:1 in each mini-batch. The loss coefficients λtxt, λbce, and λdice were set to 0.5, 5, and 1, respectively, and the DICE loss was computed only for positive samples. Further model training details are described in the Appendix F.

Baseline Models. Since we present the first dataset for ILS in CXRs, no existing models have been directly trained on our proposed task. Nonetheless, we evaluated several models from both the general domain (LISA [24], Text4Seg [25], PixelLM [40]) and the medical domain (BiomedParse [47], RecLMIS [16], IMIS-Net [5]), which can take an image and text as input to produce a segmentation output.

Evaluation Metrics. We used three metrics to evaluate model performance. For positive samples, we used Intersection-over-Union (IoU)–based measures: gIoU and cIoU [24]. gIoU is the average IoU across samples, while cIoU is the ratio of total intersection to total union across the dataset. For negative cases, we used empty-target accuracy (N-Acc.), the proportion of samples correctly predicted to have no masks [44].

##### 6.2. Main Results

Table 4 presents the results of the baselines and our proposed modelontheMIMIC-ILStestset. Whileexisting VLM-based segmentation models from both the general and medical domains struggle with the ILS task, ROSALIA (LISA-7B finetuned on MIMIC-ILS) achieves notably high performance. In particular, not only do these baselines yield low IoU scores on positive cases, but they also frequently fail on empty-target cases, where the N-Acc. rate is nearly zero in most instances. These results highlight the need for a dedicated dataset to effectively address the ILS task in CXRs. Furthermore, the strong results of ROSALIA on the physician-verified test set demonstrate that the training set of MIMIC-ILS serves as a high-quality resource—even without manual expert filtering.

Table 4. Segmentation results (%) on the MIMIC-ILS test set. “NAcc.” denotes the accuracy of correctly predicting empty targets. ¶ indicates medical domain baselines. The best and second-best results are marked in bold and underline, respectively.

Model gIoU cIoU N-Acc. LISA-7B [24] 8.3 12.8 0.7

LISA-13B [24] 8.9 12.2 0.0 Text4Seg [25] 6.1 10.3 20.6

PixelLM-7B [40] 9.2 11.8 0.0 PixelLM-13B [40] 12.8 15.4 0.0 BiomedParse¶ [47] 23.8 18.5 0.6

RecLMIS¶ [16] 22.4 19.5 0.0

IMIS-Net¶ [5] 9.8 11.8 21.6 ROSALIA (Ours) 71.2 75.6 91.8

Table 5 presents the performance of ROSALIA across different lesion types. The overall gIoU exceeds 0.7, indicating that more than 80% of the regions overlap between the predicted and ground-truth masks when the two are of similar size. Even for the lesion type with the lowest gIoU, the score remains above 0.55, suggesting over 70% regional overlap under similar mask sizes between the ground truth and predictions.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

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

- Figure 6. Visualized inference results of ROSALIA and baseline models. The first three rows show results for positive cases, while the last row presents results for negative cases with an empty target mask. Additional examples are demonstrated in Appendix G.

Table 5. Segmentation performance (%) of ROSALIA for each lesion type.

empty-target cases. Additionally, Fig. 7 demonstrates the outputs produced from diverse instructions applied to the same input image. Although multiple lesions coexist in the image, ROSALIA accurately interprets each instruction and generates results tailored to the user’s specific request. This highlights the model’s ability to handle diverse lesion types and locations of interest.

Lesion gIoU cIoU N-Acc.

Cardiomegaly 89.0 89.0 85.8 Pneumonia 57.2 60.4 97.1 Atelectasis 60.2 58.7 91.7

Opacity 60.5 64.2 85.0 Consolidation 61.9 65.6 91.2

[Figure 57]

[Figure 58]

[Figure 59]

Edema 64.8 66.6 92.2 Effusion 60.3 59.6 90.4

Total 71.2 75.6 91.8

We also evaluate the accuracy of text responses across different question types, as shown in Table 6. A response is considered correct only when both the template and all variables for each question type (denoted by square brackets in Table 2) exactly match the structured ground-truth information. Despite this strict criterion, ROSALIA achieves high accuracy across most question types (see Appendix G for text accuracy of each lesion type).

Figure 7. Examples of outputs from different instructions applied to the same image. Among the multiple lesions present, ROSALIA can selectively segment only the lesion and location of interest.

#### 7. Conclusion

Table 6. Text response accuracy (%) of ROSALIA.

In this study, we introduce MIMIC-ILS, the first dataset for instruction-guided lesion segmentation in CXRs, along with ROSALIA, a VLM developed for this new task. Our automated pipeline enables the construction of this millionscale dataset, and expert evaluations show a remarkably high acceptance rate, confirming the quality and reliability of our fully human-free data generation process. Trained on MIMICILS, ROSALIA demonstrates a comprehensive ability to generate accurate lesion segmentations and textual responses across diverse user instructions. These findings indicate that MIMIC-ILS and ROSALIA offer a strong foundation for advancing research on fine-grained lesion grounding in the CXR domain.

Type Overall Basic Global Lesion Inf.

Positive 90.7 95.4 93.7 75.1 Negative 95.3 96.9 82.3 90.6

Total 94.4 96.8 88.8 84.8

##### 6.3. Qualitative Results

Fig. 6 presents qualitative examples from each model for the ILS task. The baseline models largely fail, either producing entirely incorrect masks or segmenting the whole anatomical regions (e.g., the left or right lung). In contrast, ROSALIA accurately segments only the lesion specified in the instruction within the designated region and correctly identifies

Acknowledgments. This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants (No.RS-2019II190075, No.RS-2022-II220984, No.RS-2024-00457882, No.RS-2025-02304967, No.RS-2025-02305581), the Korea Health Industry Development Institute (KHIDI) grant (No.RS-2025-02213750), and National Research Foundation of Korea (NRF) grant (NRF-2020H1D3A2A03100945, No.RS-2023-00209060), and the Medical Scientist Training Program funded by the Korean government (MSIT, MOHW).

#### References

- [1] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-Schneider, Bennett A Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, et al. The medical segmentation decathlon. Nature communications, 13(1):4128, 2022. 3
- [2] Patrick Bilic, Patrick Christ, Hongwei Bran Li, Eugene Vorontsov, Avi Ben-Cohen, Georgios Kaissis, Adi Szeskin, Colin Jacobs, Gabriel Efrain Humpire Mamani, Gabriel Chartrand, et al. The liver tumor segmentation benchmark (lits). Medical image analysis, 84:102680, 2023. 3
- [3] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Stephanie Hyland, Maria Wetscherek, Tristan Naumann, Aditya Nori, Javier Alvarez-Valle, et al. Making the most of text semantics to improve biomedical vision–language processing. In European conference on computer vision, pages 1–21. Springer, 2022. 2, 5
- [4] Joshua Broder. Imaging the chest: the chest radiograph. Diagnostic imaging for the emergency physician, page 185, 2011. 1
- [5] Junlong Cheng, Bin Fu, Jin Ye, Guoan Wang, Tianbin Li, Haoyu Wang, Ruoyu Li, He Yao, Junren Cheng, JingWen Li, et al. Interactive medical image segmentation: A benchmark dataset and baseline. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20841–20851,

2025. 3, 7

- [6] Sherri de Coronado, Margaret W Haber, Nicholas Sioutos, Mark S Tuttle, and Lawrence W Wright. Nci thesaurus: using science-based terminology to integrate cancer research results. In MEDINFO 2004, pages 33–37. IOS Press, 2004. 1
- [7] Viacheslav V Danilov, Alex Proutski, Alex Karpovsky, Alexander Kirpich, Diana Litmanovich, Dato Nefaridze, Oleg Talalov, Semyon Semyonov, Vladimir Koniukhovskii, Vladimir Shvartc, et al. Indirect supervision applied to covid19 and pneumonia classification. Informatics in Medicine Unlocked, 28:100835, 2022. 2
- [8] Daniel Coelho de Castro, Aurelia Bustos, Shruthi Bannur, Stephanie L Hyland, Kenza Bouzid, Maria Teodora Wetscherek, Maria Dolores S´anchez-Valverde, Lara JaquesP´erez, Lourdes P´erez-Rodr´ıguez, Kenji Takeda, et al. Padchest-gr: A bilingual chest x-ray dataset for grounded radiology report generation. NEJM AI, 2(7):AIdbp2401120,

2025. 2, 3

- [9] Aysen Degerli, Serkan Kiranyaz, Muhammad EH Chowdhury, and Moncef Gabbouj. Osegnet: Operational segmentation network for covid-19 detection using chest x-ray images. In 2022 IEEE International Conference on Image Processing (ICIP), pages 2306–2310. IEEE, 2022. 2, 3
- [10] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23

(2):304–310, 2015. 2

- [11] Nicolas Gaggion, Candelaria Mosquera, Martina Aineseder, Lucas Mansilla, Diego Milone, and Enzo Ferrante. Chexmask database: a large-scale dataset of anatomical segmentation masks for chest x-ray images. 1
- [12] Nicolas Gaggion, Lucas Mansilla, Candelaria Mosquera, Diego H. Milone, and Enzo Ferrante. Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis. IEEE Transactions on Medical Imaging, 2022. 6
- [13] Nicol´as Gaggion, Candelaria Mosquera, Lucas Mansilla, Julia Mariel Saidman, Martina Aineseder, Diego H Milone, and Enzo Ferrante. Chexmask: a large-scale dataset of anatomical segmentation masks for multi-center chest x-ray images. Scientific Data, 11(1):511, 2024. 1
- [14] Nicholas Heller, Sean McSweeney, Matthew Thomas Peterson, Sarah Peterson, Jack Rickman, Bethany Stai, Resha Tejpaul, Makinna Oestreich, Paul Blake, Joel Rosenberg, et al. An international challenge to use artificial intelligence to define the state-of-the-art in kidney and kidney tumor segmentation in ct imaging., 2020. 3
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 7

- [16] Xiaoshuang Huang, Hongxiang Li, Meng Cao, Long Chen, Chenyu You, and Dong An. Cross-modal conditioned reconstruction for language-guided medical image segmentation. IEEE Transactions on Medical Imaging, 2024. 1, 3, 7
- [17] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, pages 590–597, 2019. 2
- [18] Yankai Jiang, Zhongzhen Huang, Rongzhao Zhang, Xiaofan Zhang, and Shaoting Zhang. Zept: Zero-shot pan-tumor segmentation via query-disentangling and self-prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11386–11397, 2024. 3
- [19] Yankai Jiang, Wenhui Lei, Xiaofan Zhang, and Shaoting Zhang. Unleashing the potential of vision-language pretraining for 3d zero-shot lesion segmentation via maskattribute alignment. arXiv preprint arXiv:2410.15744, 2024. 3
- [20] Alistair Johnson, Matt Lungren, Yifan Peng, Zhiyong Lu, Roger Mark, Seth Berkowitz, and Steven Horng. Mimic-cxr-

- jpg-chest radiographs with structured labels. PhysioNet, 101: 215–220, 2019. 1
- [21] A Johnson, T Pollard, R Mark, S Berkowitz, and S Horng. Mimic-cxr database (version 2.1. 0). physionet. rrid: Scr 007345, 2024. 2, 3, 5, 1

- [22] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a deidentified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019. 2, 3, 5, 1
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 7
- [24] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579– 9589, 2024. 1, 2, 3, 6, 7
- [25] Mengcheng Lan, Chaofeng Chen, Yue Zhou, Jiaxing Xu, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Text4seg: Reimagining image segmentation as text generation. arXiv preprint arXiv:2410.09855, 2024. 1, 3, 7
- [26] Hyungyung Lee, Geon Choi, Jung-Oh Lee, Hangyul Yoon, Hyuk Gi Hong, and Edward Choi. Cxreasonbench: A benchmark for evaluating structured diagnostic reasoning in chest x-rays. arXiv preprint arXiv:2505.18087, 2025. 6
- [27] Hyungyung Lee, Geon Choi, Jung-Oh Lee, Hangyul Yoon, Hyuk Gi Hong, and Edward Choi. CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays (version 1.0.1). PhysioNet, 2025. RRID:SCR 007345. 6

- [28] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Ren´e Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022. 3
- [29] Zihan Li, Yunxiang Li, Qingde Li, Puyang Wang, Dazhou Guo, Le Lu, Dakai Jin, You Zhang, and Qingqi Hong. Lvit: language meets vision transformer in medical image segmentation. IEEE transactions on medical imaging, 43(1):96–107,

2023. 1, 3

- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 7
- [31] Jie Liu, Yixiao Zhang, Jie-Neng Chen, Junfei Xiao, Yongyi Lu, Bennett A Landman, Yixuan Yuan, Alan Yuille, Yucheng Tang, and Zongwei Zhou. Clip-driven universal model for organ segmentation and tumor detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 21152–21164, 2023. 3
- [32] Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, and Ming-Ming Cheng. Rethinking computer-aided tuberculosis diagnosis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2646–2655,

2020. 2, 3

- [33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7

- [34] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016. 7
- [35] Duc Nguyen, DungNB, Ha Q. Nguyen, Julia Elliott, NguyenThanhNhan, and Phil Culliton. Vinbigdata chest x-ray abnormalities detection. https://kaggle.com/competitions/ vinbigdata- chest- xray- abnormalities- detection,

2020. Kaggle. 1

- [36] Dung Nguyen, Minh Khoi Ho, Huy Ta, Thanh Tam Nguyen, Qi Chen, Kumar Rav, Quy Duong Dang, Satwik Ramchandre, Son Lam Phung, Zhibin Liao, Minh-Son To, Johan Verjans, Phi Le Nguyen, and Vu Minh Hieu Phan. Localizing before answering: A benchmark for grounded medical visual question answering. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI 2025), pages 7670–7676, 2025. 3
- [37] Ha Q Nguyen, Khanh Lam, Linh T Le, Hieu H Pham, Dat Q Tran, Dung B Nguyen, Dung D Le, Chi M Pham, Hang TT Tong, Diep H Dinh, et al. Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations. Scientific Data, 9(1): 429, 2022. 2, 3, 1
- [38] Fernando P´erez-Garc´ıa, Sam Bond-Taylor, Pedro P Sanchez, Boris van Breugel, Daniel C Castro, Harshita Sharma, Valentina Salvatelli, Maria TA Wetscherek, Hannah Richardson, Matthew P Lungren, et al. Radedit: stress-testing biomedical vision models via diffusion image editing. In European Conference on Computer Vision, pages 358–376. Springer,

2024. 3, 1

- [39] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3505–3506,

2020. 9

- [40] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26374–26383, 2024. 1, 3, 7
- [41] Constantin Seibold, Alexander Jaus, Matthias A Fink, Moon Kim, Simon Reiß, Ken Herrmann, Jens Kleesiek, and Rainer Stiefelhagen. Accurate fine-grained segmentation of human anatomy in radiographs via volumetric pseudo-labeling. arXiv preprint arXiv:2306.03934, 2023. 3, 1
- [42] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. Advances in Neural Information Processing Systems, 35:33754–33767,

2022. 3

- [43] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2097–2106,

2017. 5, 2

- [44] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized segmentation

- via multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3858–3869, 2024. 7
- [45] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18134–18144, 2022. 3
- [46] Anna Zawacki, Carol Wu, George Shih, Julia Elliott, Mikhail Fomitchev, Mohannad Hussain, ParasLakhani, Phil Culliton, and Shunxing Bao. Siim-acr pneumothorax segmentation. https://kaggle.com/competitions/siim-acrpneumothorax-segmentation, 2019. Kaggle. 2, 3
- [47] Theodore Zhao, Yu Gu, Jianwei Yang, Naoto Usuyama, Ho Hin Lee, Tristan Naumann, Jianfeng Gao, Angela Crabtree, Jacob Abel, Christine Moung-Wen, et al. Biomedparse: a biomedical foundation model for image parsing of everything everywhere all at once. arXiv preprint arXiv:2405.12971,

2024. 3, 7

- [48] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European conference on computer vision, pages 696–712. Springer, 2022. 3

# Appendix

### <Table of Contents>

- A. Grounded Lesion Mask Generation 1

- A.1. Report Pre-Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.2. Large Language Models and Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.3. Vision Models and Characteristics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.4. Thresholds for Lesion Mask Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.5. Lesion Mask Post-Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.6. Empty Location . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- A.7. Sample Discarding and Pipeline Recall . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

- B. Lesion Types 2
- C. Instruction-Answer Pair Generation 5

- C.1. Positive Instruction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.2. Negative Instruction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.3. Clinical Utility of MIMIC-ILS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- D. Quality Control 6

- D.1. Chest X-rays . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- D.2. Lung and Heart Masks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- D.3. Cardiomegaly . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- E. MIMIC-ILS Dataset 6

- E.1. Details for Dataset Splits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.2. Quality Assessment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.3. Details on Expert Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.4. Data Generation Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- F. Model Training Details 9
- G. Additional Experimental Results 9

- G.1. Lesion-Wise Text Accuracy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- G.2. Additional Qualitative Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

#### A. Grounded Lesion Mask Generation

##### A.1. Report Pre-Processing

Textual information for the CXR images from MIMIC-CXR was extracted from their corresponding radiology reports. From these raw reports, we extract the findings, impression, and last paragraph sections following the official MIMIC report pre-processing code. We then adhere to a hierarchical fallback logic to select a single representative text section for each study: the impression section is used if the findings section is missing, and the last paragraph is used if the impression is also absent. Studies that lack all three of these sections are excluded.

##### A.2. Large Language Models and Prompts

To extract information from the pre-processed report section, our pipeline employs two distinct large language models (LLMs). The initial report structuring step utilizes MistralSmall-3.1-24B-Instruct-2503. Using the prompt shown in Figure 9, we extract lesion information from the report as six-element tuples. For the subsequent location mapping step, we employ medgemma-27b-text-it, which is specialized in the medical domain. Using the prompt shown in Figure 10, we normalize the lesion’s location in a two-step process. In compliance with the PhysioNet credentialed data use agreement for MIMIC-CXR, both models were run on our local GPU setup.

##### A.3. Vision Models and Characteristics

Pretrained HybridGNet. We utilized a HybridGNet model, pretrained on the CheXMask dataset [11–13], to segment the right lung, left lung, and heart. It demonstrates robust segmentation performance for these three organs, even in challenging CXRs from patients with severe conditions characterized by dense opacities.The resulting masks serve multiple, distinct roles in our pipeline. The heart mask is used directly as the ground-truth lesion mask for cardiomegaly. The right and left lung masks serve two purposes: they are used as the (Lr,Ll) inputs in Algorithm 1, and they are also merged with the heart mask to define the editing region for RadEdit. Details regarding the use of this model were omitted from the main text for brevity.

RadEdit. This diffusion-based image editing model takes a chest X-ray image and a text prompt as input [38]. To transform the input into a normal-appearing image, we used the standard prompt on which RadEdit was trained: “No acute cardiopulmonary process”. Additionally, it requires a mask specifying the editing region. For this, we used the merged masks from the pretrained HybridGNet described above. Notably, we used the original MIMIC-CXR dataset [21, 22], which contains DICOM files, rather than the MIMIC-CXRJPG version [20]. This is because RadEdit was trained on the

original MIMIC-CXR, and we observed that inputting the histogram-equalized MIMIC-CXR-JPG images significantly degraded the quality of the edited image.

CXAS. Designed for anatomy segmentation in CXRs, this model is capable of segmenting 159 anatomical region classes [41]. Specifically, in our research, we input the opacity-removed images (the output of RadEdit) into CXAS to segment the anatomy. This is because CXAS tends to produce lower-quality anatomy masks for patients with significant opacities.

Pretrained YOLO. For lesion detection, we employed a YOLO model, specifically utilizing the checkpoint from the submitted solution in the VinBigData Chest X-ray Abnormalities Detection competition [35, 37]. Although this model can detect various types of lesions (aortic enlargement, atelectasis, calcification, consolidation, ILD, infiltration, lung opacity, nodule/mass, other lesion, pleural effusion, pleural thickening, pneumothorax, pulmonary fibrosis), we filtered its outputs to retain only those findings considered hyperintense lesions. We therefore excluded aortic enlargement, other lesion, and pneumothorax from the detection categories.

##### A.4. Thresholds for Lesion Mask Generation

The thresholds in Equation 1 and Algorithm 1 were carefully calibrated to ensure maximum mask quality. The final values were determined through an iterative process involving multiple quality checks by a physician, who identified the settings that maximized the yield of high-quality masks. The finalized thresholds are summarized in Table 7. With the exception of edema, the threshold settings are identical for all other lesion types. We set the threshold values for edema lower than for other lesion types because it tends to spread widely throughout the lungs.

Table 7. Threshold values used to generate the lesion masks. ‘General’ lesions refer to all lesions other than edema.

Threshold General Edema

τano 0.10 0.01 τanatomy 0.25 0.25 τconf 0.20 0.01 τsignal 0.20 0.20 τsize 0.10 0.10

##### A.5. Lesion Mask Post-Processing

To further enhance the quality of the final lesion masks, additional post-processing steps were applied. Sequential erosion and dilation operations are used to remove small, scattered noise. Although omitted from the main text for brevity, this

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Figure 8. An example of detailed lesion mask generation where the report mentions “Areas of streaky opacity are again seen in the upper lobes.” but the mask is grounded only to the right upper lobe. In the top row, the yellow lesion box mask in the left lung is discarded due to insufficient overlap with the green upper lung mask. The remaining box mask in the right lung is then used to filter the anomaly map. Intermediate post-processing steps, including noise removal and mask expansion, are applied to enhance the final mask quality.

noise removal step is also performed prior to filtering the anomaly map with the lesion box mask. We also expanded the lesion masks to include adjacent pixels with similar intensity values for more complete segmentation. Furthermore, specifically for effusions at the lung base, we incorporated the lower portion of the lung masks from the pretrained HybridGNet to ensure clean coverage extending to the costophrenic angle. The detailed process is illustrated in Figure 8.

##### A.6. Empty Location

We extract an “empty location” during the location verification step. An empty location is defined as a lung region where a specific lesion is not present. We designated a lung region as an empty location if it did not overlap at all with the anatomy masks corresponding to the reported location. To identify locations that are truly free of any reported lesions, we compute this not only for the seven major lesions but for all lesions mentioned in the report, and utilize this information in a subsequent data generation step.

##### A.7. Sample Discarding and Pipeline Recall

In our pipeline, samples that did not meet the strict crossmodel consistency criteria were excluded, and the resulting recall rate is reported in Table 8. While the recall can be flexibly increased by relaxing these criteria, our primary

goal is to generate high-confidence samples rather than to maximize recall. Since training the LISA on a higher-recall dataset led to degraded performance on the MIMIC-ILS test set (gIoU: 54.3%, cIoU: 61.4%, N-Acc: 95.8%), we opted for a high-threshold setting.

Table 8. Recall comparison for different threshold values.

Lesion atel. pneu. effu. opac. edem. cons. Avg. Recall (default τ) 13.4 28.5 16.0 21.6 58.0 31.9 28.3 Recall (half τ) 28.7 49.8 32.6 40.3 82.1 50.6 47.3

#### B. Lesion Types

We construct our dataset around seven major lesion types commonly observed in CXRs, identified through discussions with board-certified physicians: opacity, consolidation, pnuemonia, atelectasis, edema, cardiomegaly, and effusion. These disease categories are widely utilized in many CXR-related studies [10, 17, 43]. First, we include opacity and consolidation, which are high-level, comprehensive terms referring to hyperintense lesions. These broad categories can be mapped to specific lung lesion types, including pneumonia, atelectasis, and edema. We also include two major non-lung disease categories: cardiomegaly, the enlargement of the heart, and (pleural) effusion, the accumulation of fluid in the pleural space.

###### Figure 9. A prompt template for report structuring.

###### Figure 10. A prompt template for location mapping. Step 2 illustrates the scenario following the ‘lung’ mapping from Step 1.

#### C. Instruction-Answer Pair Generation

##### C.1. Positive Instruction

Basic Instruction. The instructions for positive samples are generated directly from the grounded lesion mask generation results. For instance, if a definitive finding of pneumonia has a grounded location of right lung base and left lung base, we generate a basic instruction: “Segment the pneumonia in the right lung base and left lung base.”. However, if the finding’s certainty is tentative, indicating the lesion’s presence is not definitive, we substitute it with the more general term opacity to create the basic instruction. For example, the previous instruction becomes: “Segment the opacity in the right lung base and left lung base.”. As listed in Table 9, the target location can be specified as a single area—either a broad region or a specific lung zone—or as a combination of these areas.

Table 9. List of valid target locations for basic instructions. Locations are categorized into broad regions and specific lung zones.

Lung Region Location Name Broad Regions

right lung

left lung

right apical zone lung right upper zone lung right mid zone lung

Lung Zones (Right)

right lung base

left apical zone lung left upper zone lung left mid zone lung

Lung Zones (Left)

left lung base

Global Instruction. A global instruction is used to segment all instances of a lesion across the entire lung, without specifying a location. To create a valid global instruction, the generated lesion must cover all lesions cited in the report; this ensures the mask can serve as a complete ground truth. Therefore, we only generate global instructions when the grounded location, where masks were actually generated, and the reported location, the complete area mentioned in the report, are identical. If the lesion type is cardiomegaly, we always generate this instruction type. This is because cardiomegaly represents a condition of the heart itself, rather than a lesion that can appear in variable locations.

Lesion Inference Instruction. We generate lesion inference instructions to enable the model to infer the specific lesion type from an opacity at a given location. These instructions are generated for findings regardless of their original certainty level, as the certainty is instead reflected in the ground-truth text description. We selected pneumonia, atelectasis, and

edema as the target lesion types for this task. This choice reflects clinical reporting practices, where radiologists often describe these specific findings using an inferential process. In contrast, other major lesion types are typically stated directly. For example, a report rarely states, “There is an opacity in the left lung. It is highly suggestive of effusion.”; instead, the finding is stated directly as “Left lung effusion.”

##### C.2. Negative Instruction

Negative instructions are generated in two main scenarios. First, we generate instructions for lesions that are either never mentioned or negated (e.g., “no pneumonia”) in the report. For these findings, we create a negative instruction, which can be either a basic type by randomly assigning a lung region (e.g., “Segment the pneumonia in the left lung.”), or a global type (e.g., “Segment the pneumonia”). The second method involves pairing a target lesion with a randomly selected empty location. For example, if right lung apex and left lung apex are empty locations, we can generate the instruction, “Segment the atelectasis in the right lung apex.” To prevent an excessive number of negative samples, our logic restricts the generation to a maximum of one negative instruction per lesion type for each study.

##### C.3. Clinical Utility of MIMIC-ILS

Diversity of Instructions. The functional scope of the dataset is driven primarily by the diversity of disease–anatomy combinations rather than by the number of instruction templates. Linguistic diversity can be readily addressed by paraphrasing existing instructions in MIMIC-ILS. To demonstrate this feasibility, we used Qwen3-Next-80BA3B-Instruct to paraphrase each original instruction into nine variants reflecting three user personas (medical experts, laypersons, and AI developers). LISA trained on this enriched dataset still demonstrate strong performance (gIoU: 67.3%, cIoU: 73.1%, N-Acc: 96.5%) on the paraphrased test set.

Usability for Laypersons. Users without medical expertise cannot be expected to visually identify lesions in a scan to provide basic instructions. However, because MIMIC-ILS incorporates negative cases for absence confirmation, a model trained on this dataset naturally overcomes this limitation. By iteratively querying the model across various anatomical locations, the system can autonomously verify the presence of a lesion—outputting a precise segmentation mask if it exists, or confirming its absence otherwise. Similarly, global instructions (e.g., “Segment the opacity”) offer an intuitive way for users to make broad inquiries. Coupled with the model’s robust handling of negative cases, these capabilities ensure that users can effectively obtain screening results without ever needing to visually inspect the image themselves.

#### D. Quality Control

##### D.1. Chest X-rays

We exclusively utilized Posteroanterior (PA) and Anteroposterior (AP) view images from the MIMIC-CXR dataset. However, even within these designated views, the dataset contains noisy samples, including mislabeled lateral views, non-chest X-rays, or images with severe anatomical truncation. To ensure data quality, we leveraged metadata from CXReasonBench [26, 27]. This dataset was meticulously constructed from frontal view images within the MIMIC-CXR dataset that had verified high image quality. Specifically, we utilized its pre-extracted information such as the count of extractable CXAS anatomy masks and indicators of full chest visibility to identify and exclude these problematic images beforehand.

##### D.2. Lung and Heart Masks

Lung and heart masks are a critical component for the construction of MIMIC-ILS. However, both models can produce erroneous results: the pretrained HybridGNet occasionally generates abnormal masks, and CXAS (even when applied to RadEdit-processed images) also generates suboptimal masks. To address this, we cross-referenced the masks from both models and excluded cases with significant discrepancies, interpreting this as a failure in either the HybridGNet or CXAS segmentation. Specifically, we determined that large differences in the outermost x-coordinates of the lung masks or the lowermost y-coordinates of the heart masks would cause problems for subsequent grounded lesion mask generation, and thus excluded these studies.

##### D.3. Cardiomegaly

To generate reliable negative samples for cardiomegaly, we measured the cardiothoracic ratio (CTR) using the right lung, left lung, and heart masks generated by the pretrained HybridGNet. We then filtered these samples, exclusively including those with a CTR of 0.45 or less in our final negative dataset. This 0.45 threshold was calibrated by a physician who analyzed the distribution of CXRs across different CTR intervals to establish a clinically sound cutoff.

#### E. MIMIC-ILS Dataset

##### E.1. Details for Dataset Splits

The data splits and distribution by instruction type for MIMIC-ILS are presented in Tables 10, 11, and 12. Note that the counts for the test set reflect the final numbers after excluding cases that were rejected during the quality assessment process.

##### E.2. Quality Assessment

A rigorous quality assessment was conducted on the test split by four physicians. All positive samples were reviewed by all four physicians, while the negative samples were divided among them for evaluation. The reviewers were provided with an CXR image, the lesion type, and the mapped anatomical location text generated from our information-grounding process, along with the corresponding ground-truth radiology report. They were then asked to mark each pair as either “Acceptable” or “Not Acceptable” on a review sheet. The evaluation process was conducted independently for each expert, ensuring that no reviewer could access the others’ evaluation results.

##### E.3. Details on Expert Evaluation

The expert evaluations were conducted by four physicians, all experienced radiation oncologists with extensive training in lesion contouring. Their professional backgrounds are as follows: Experts A and B are board-certified physicians with 9 and 7 years of clinical experience, respectively, while Experts C and D are resident doctors, each with 6 years of clinical experience. Also, the lesion-level acceptance rate in human evaluation are shown in Table 13.

##### E.4. Data Generation Examples

With our proposed data generation pipeline, we can produce high-quality lesion masks and their corresponding instruction–answer pairs from grounded information. Figure 11 illustrates representative examples across various lesion types and anatomical locations, including both positive and negative cases, along with their corresponding structured information.

- Table 10. Number of generated instruction-answer pairs per lesion and template type in MIMIC-ILS train split.

Lesion # IAs

Basic Global Lesion Inference pos neg pos neg pos neg

cardiomegaly 63,153 0 0 39,108 24,045 0 0 pneumonia 158,059 4,542 145,317 511 3,147 4,542 0 atelectasis 166,935 8,846 128,943 3,331 16,969 8,846 0

opacity 156,807 9,113 73,619 1,532 274 0 72,269 consolidation 154,955 3,428 144,489 379 6,659 0 0

edema 182,233 14,150 145,251 5,390 3,292 14,150 0 effusion 162,998 10,244 114,375 3,713 34,666 0 0

Total 1,045,140 50,323 751,994 53,964 89,052 27,538 72,269

- Table 11. Number of generated instruction-answer pairs per lesion and template type in MIMIC-ILS validation split.

Basic Global Lesion Inference pos neg pos neg pos neg

Lesion # IAs

cardiomegaly 539 0 0 332 207 0 0 pneumonia 1,211 28 1,130 2 23 28 0 atelectasis 1,316 75 986 33 147 75 0

opacity 1,225 75 581 13 0 0 556 consolidation 1,213 30 1,137 3 43 0 0

edema 1,469 129 1,124 59 28 129 0 effusion 1,273 416 885 22 287 0 0

Total 8,246 416 5,843 464 735 232 556

Table 12. Number of generated instruction-answer pairs per lesion and template type in MIMIC-ILS test split.

Basic Global Lesion Inference pos neg pos neg pos neg

Lesion # IAs

cardiomegaly 965 0 0 803 162 0 0 pneumonia 1,767 60 1,596 8 43 60 0 atelectasis 1,842 110 1,466 45 111 110 0

opacity 1,753 174 779 26 5 0 769 consolidation 1,756 69 1,612 10 65 0 0

edema 2,274 283 1,551 103 54 283 0 effusion 1,878 156 1,312 54 356 0 0

Total 12,235 852 8,316 1,049 796 453 769

Table 13. Acceptance rate (%) by lesion type for each expert in the human evaluation.

Expert A Expert B Expert C Expert D Pos Neg Total Pos Neg Total Pos Neg Total Pos Neg Total

Lesion

Cardiomegaly 97.7 97.1 97.7 99.2 100.0 99.2 100.0 100.0 100.0 99.4 100.0 99.4 Pneumonia 90.0 98.5 97.3 97.1 99.7 99.4 98.6 99.5 99.4 98.6 98.8 98.7 Atelectasis 97.2 98.8 98.4 79.9 99.5 94.2 100.0 99.0 99.3 99.3 97.3 97.8 Opacity 92.6 92.3 92.4 96.5 96.5 95.1 99.5 92.7 96.0 96.0 96.6 96.3 Consolidation 97.2 95.7 95.9 100.0 97.2 97.6 100.0 98.3 98.5 100.0 99.0 99.2 Edema 95.8 95.1 95.4 97.9 100.0 99.0 100.0 97.3 98.5 89.8 98.2 94.4 Effusion 89.8 96.6 94.1 89.3 97.3 94.3 99.5 97.6 98.3 95.4 98.8 97.6

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

###### Figure 11. Examples of final generated samples in our MIMIC-ILS dataset.

#### F. Model Training Details

In our experiments, training the LISA-7B–based model took approximately two and a half days on two NVIDIA H100 GPUs for 15 epochs. Using the DeepSpeed package [39], we trained the model with the DeepSpeed Stage-2 configuration and a WarmupDecayLR scheduler, with 100 warmup steps and a minimum and maximum learning rate of 0 and 0.0003, respectively. For inference on the test set, which contains 12K examples, segmentation alone takes about 20 minutes, whereas segmentation with text outputs requires approximately 1.5 hours. During training, each input image had a 50% chance of being processed with histogram equalization.

#### G. Additional Experimental Results

##### G.1. Lesion-Wise Text Accuracy

The text-response accuracy of ROSALIA for each lesion type is summarized in Table 14. Across most lesion and question types, the model consistently achieves high accuracy, similar to the segmentation performance reported in Table 5. For lesion-inference questions, CXR alone typically cannot provide a definitive diagnosis and often requires additional examinations (e.g., blood tests or cultures). As a result, radiologists generally provide only a differential diagnosis based on visual findings. Given this inherent uncertainty in CXR interpretation, improvements in accuracy for lesion-inference questions are naturally limited. Nevertheless, we evaluated how well the trained model on our dataset can perform on this question type and leave further advancements in this direction to future work.

Table 14. Text response accuracy (%) of ROSALIA across different question and lesion types.

Lesion Overall Basic Global Lesion Inf. Cardiomegaly 96.0 - 96.0 -

Pneumonia 96.3 99.2 72.6 36.7 Atelectasis 92.9 96.9 69.2 69.1

Opacity 91.5 92.2 93.6 90.5 Consolidation 97.3 97.6 92.0 -

Edema 93.4 96.2 74.5 85.5 Effusion 94.5 96.9 85.9 -

Total 94.4 96.8 88.8 84.8

##### G.2. Additional Qualitative Examples

We present additional qualitative examples comparing ROSALIA with baseline models in Figure12. Unlike the baselines, ROSALIA produces accurate segmentation outputs tailored to diverse user instructions. In addition, examples that include both text responses and segmentation outputs are shown in Figure 13. In these cases as well, ROSALIA provides highly factual text responses alongside precise segmentation results.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

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

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

###### Figure 12. Qualitative comparison of segmentation results between ROSALIA and baseline models.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### Figure 13. Examples of textual responses generated by ROSALIA. All generated text responses correctly match the ground-truth answers, and both the segmentation and textual outputs in this figure are rated as good examples by medical experts.

