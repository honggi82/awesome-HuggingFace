# arXiv:2503.15264v1[cs.CV]19Mar2025

## LEGION: Learning to Ground and Explain for Synthetic Image Detection

Hengrui Kang1,2*, Siwei Wen3,2*, Zichen Wen1,2*, Junyan Ye4,2, Weijia Li4,2† Peilin Feng3,2, Baichuan Zhou2, Bin Wang2, Dahua Lin2,5, Linfeng Zhang1, Conghui He2,5†

1Shanghai Jiao Tong University 2Shanghai Artificial Intelligence Laboratory 3Beihang University 4Sun Yat-Sen University 5SenseTime Research Project Page: https://opendatalab.github.io/LEGION E-mail: liweij29@mail.sysu.edu.cn, heconghui@pjlab.org.cn

### Abstract

|Defender|Defender & Controller|
|---|---|
|[Figure 1]<br><br>Real Fake<br><br>Classifier<br><br>Deepfake Detection Traditional IFDL<br><br>Recent Work (MLLMs-Based)<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>CNNSpot、UniFD … SPAN、HiFi-Net …<br><br>[Figure 6]<br><br>[Figure 7]<br><br>Tamper-only<br><br>[Figure 8]<br><br>[Figure 9]<br><br>No explanation<br><br>[Figure 10]<br><br>|FFAA (24’08)<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>Domain: Face-only No Segmentation|
|---|
<br><br>|Fakeshield (24’10)<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Fogery: Temper-only<br><br>[Figure 21]|
|---|
<br><br>|ForgeryGPT (24’10)<br><br>[Figure 22]<br><br>Fogery: Temper-only<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]|
|---|
<br><br>|SIDA (24’12)<br><br>No Mask for Fully Synthetic Ones<br><br>[Figure 28]<br><br>Focus More on Temper<br><br>[Figure 29]|
|---|
<br><br>[Figure 30]|LEGION (Ours)<br><br>Detection<br><br>Real<br><br>Fake<br><br>Analysis<br><br>Region 1 Loca. Mask Expl.<br><br>Detection Grounding Explanation<br><br>[Figure 31]<br><br>[Figure 32]<br><br>But, how to serve as a Controller<br><br>|[Figure 33]<br><br>Regeneration<br><br>Prompt<br><br>LEGION<br><br>T2I Model<br><br>[Figure 34]<br><br>Expl. Feedback|
|---|
<br><br>As a Defender, we can<br><br>|[Figure 35]<br><br>Inpainting<br><br>LEGION<br><br>Inpainting<br><br>Mask & Expl. Guidance<br><br>[Figure 36]<br><br>[Figure 37]<br><br>Regional|
|---|
<br><br>Region 2 Loca. Mask Expl.<br><br>[Figure 38]<br><br>Fully Synth.|

The rapid advancements in generative technology have emerged as a double-edged sword. While offering powerful tools that enhance convenience, they also pose significant social concerns. As defenders, current synthetic image detection methods often lack artifact-level textual interpretability and are overly focused on image manipulation detection, and current datasets usually suffer from outdated generators and a lack of fine-grained annotations. In this paper, we introduce SynthScars, a high-quality and diverse dataset consisting of 12,236 fully synthetic images with human-expert annotations. It features 4 distinct image content types, 3 categories of artifacts, and fine-grained annotations covering pixel-level segmentation, detailed textual explanations, and artifact category labels. Furthermore, we propose LEGION (LEarning to Ground and explain for Synthetic Image detectiON), a multimodal large language model (MLLM)-based image forgery analysis framework that integrates artifact detection, segmentation, and explanation. Building upon this capability, we further explore LEGION as a controller, integrating it into image refinement pipelines to guide the generation of higher-quality and more realistic images. Extensive experiments show that LEGION outperforms existing methods across multiple benchmarks, particularly surpassing the second-best traditional expert on SynthScars by 3.31% in mIoU and 7.75% in F1 score. Moreover, the refined images generated under its guidance exhibit stronger alignment with human preferences. The code, model, and dataset will be released.

Figure 1. Comparison with Existing Image Forgery Detection Methods. LEGION not only serves as a Defender, enabling multi-task forgery analysis, but also functions as a Controller, facilitating high-quality image generation.

has evolved rapidly, producing diverse and detailed synthetic images. While it enhances creativity, simplifies design, and addresses data scarcity, it also poses risks such as privacy violations, copyright disputes, and the spread of misinformation. This duality underscores both its transformative power and the ethical dilemmas it introduces. Many researchers have focused on the risks of image generation technology, developing methods and benchmarks [27, 61] for detecting synthetic images to mitigate societal harm. Yet, a thorough review of synthetic image detection research reveals notable limitations.

(I) Challenges in Synthetic Image Detection Datasets. While datasets represented by OpenForensics [24] contain a large volume of data, they primarily consist of outdated synthetic images generated using early GAN techniques. These images are of poor quality, riddled with noticeable artifacts, and largely limited to cartoon or anime styles, whose artificial nature is easily discernible. Consequently, models trained on such datasets struggle to effectively detect realistic synthetic images. The RichHF-18K dataset [28] uses point annotations for artifacts in synthetic images, offering

### 1. Introduction

From GANs [10, 21] to diffusion [13, 14, 32, 36, 37] and autoregressive models [49, 51], image generation technology

*Equal Contribution. †Corresponding authors.

low positional accuracy and poor edge delineation. Meanwhile, tampering-based datasets like SID-Set [17] provide full object contour annotations, but their approach is less generalizable as modern synthetic artifacts typically affect only small object regions.

##### (II) Limitations of Synthetic Image Detection and Ar-

tifact Localization Methods. Most traditional methods, such as PAL4VST [67], rely on low-level structural cues, effectively identifying texture disruptions but struggling with artifacts that require global reasoning, such as violations of physical laws governing lighting and shadows. Some works [17, 18, 26, 58] have introduced multimodal large language models (MLLMs) to address this problem. However, homogenization of research directions has limited further progress, as these methods predominantly focus more on tampered images with limited exploration of fully synthetic ones, where artifacts are more complex, less constrained by real-world references, and rarely studied from an interpretability perspective.

##### (III) Can Synthetic Image Detection Methods Ad-

vance Image Generation? Current synthetic image detection methods mitigate the societal risks posed by image synthesis technologies by identifying and localizing artifacts in synthetic images, thereby positioning the detection technology as a Defender. However, image generation is a double-edged sword; focusing solely on its negative implications does not fully harness the potential of synthetic image detection. From this perspective, we aim to inspire a paradigm shift in the design of synthetic image detection methods—from crafting a defender to developing a Controller. This entails not only detecting and localizing artifacts in synthetic images but also guiding image generators to produce more realistic and natural images. By doing so, we can foster the controlled advancement of image generation technologies.

To address these limitations, we present SynthScars, a challenging dataset for synthetic image detection, excluding outdated, low-quality, and cartoon-style images. It features fine-grained annotations with irregular polygons to precisely outline artifacts, along with detailed classifications and explanations. This dual-layer annotation—spatial and explanatory—elevates the dataset’s value for advancing synthetic image detection research. Moreover, to achieve in-depth interpretability, we propose LEGION, a comprehensive image forgery analysis framework tailored for fully synthetic images. By leveraging the powerful prior knowledge, reasoning and expression ability of MLLMs, it achieves strong generalization across different domains and impressive robustness against various perturbations. Furthermore, we explore the potential of using forgery explanation as feedback to enhance the generation of higherquality and more realistic images. Specifically, instead of positioning LEGION as a Defender, we employ it as

a Controller, and construct two iterative optimization pipelines via image regeneration and inpainting, respectively. For image regeneration, artifact explanations from our model iteratively refine the prompt. For image inpainting, detected artifact masks and corresponding explanations guide region-by-region selective refinement, progressively reducing artifact areas and enhancing image authenticity. The overall comparison with previous methods is illustrated in Figure 1.

The main contributions of this paper are as follows:

- • We introduce SynthScars, a challenging dataset for synthetic image detection, featuring high-quality synthetic images with diverse content types, as well as finegrained pixel-level artifact annotations with detailed textual explanations.
- • We propose LEGION, a comprehensive image forgery analysis framework for artifact localization, explanation generation, and forgery detection, which effectively aids human experts to detect and understand image forgeries.
- • Extensive experiments demonstrate that LEGION achieves exceptional performance on four challenging benchmarks. Comparisons with 19 existing methods show that it achieves state-of-the-art performance on the vast majority of metrics, exhibiting strong robustness and generalization ability.
- • We position LEGION not only as a Defender against ever-evolving generative technologies but also as a Controller that guides higher-quality and more realistic image generation. Qualitative and quantitative experiments on image regeneration and inpainting show the great value of LEGION in providing feedbacks for progressive artifact refinement.

### 2. Related Work

#### 2.1. Synthetic Image Detection and Localization

Traditional detection approaches, based on CNNs and transformers, have treated synthetic image detection as a binary classification task, leveraging spatial or frequency domain features [5, 8, 20, 47, 55, 59]. However, these methods often struggle with generalization across diverse generators and robustness against various types of perturbations. More importantly, they suffer from a lack of interpretability, with the generation of natural language explanations for specific anomaly causes remaining unexplored.

Recently, many works [11, 12, 65, 67] have extended the initial binary classification to a more complicated artifact localization task. For example, some studies have utilized gradient [41, 42, 44] or attention maps [15] to reveal potential anomalous regions. Alternatively, other scholars have focused on constructing datasets annotated with detailed artifact segmentation masks. However, these methods primarily focus on detecting forgeries from inpainting [65] or

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

###### Physical Distortion Structure

ObjectSceneAnimalHuman

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Non-Realistic Styles

Outdated Generators

- (i) Cartoon & Watercolor style
- (ii) Meaningless for our task

- (i) Unrecognized images
- (ii) Obvious and uniform artifacts

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

The neck and head of the left male appear cartoonish, not realistic. The right male is the same.

The light and shadow in the

The bird has deformed claws with some digits missing. The branch behind it is broken off abruptly.

front of table are blocked by the wall, differing from normal optics.

[Figure 66]

[Figure 67]

[Figure 68]

Coarse-Grained Annotation

Manipulation Limitations

- (i) Inaccurate point annotation
- (ii) Ambiguous edge definition

- (i) Strongly contour-dependent
- (ii) Non-universal and simple

###### (a) Diverse contents (b) Fine-grained artifacts (c) Previous drawbacks

- Figure 2. SynthScars Datasets. (a) shows image cases across four diverse content types. (b) presents annotation cases across different fine-grained artifact types. (c) enumerates drawbacks of previous datasets, which SynthScars perfectly addresses.

#### 2.3. Guided Image Refinement

human manipulation [16, 34, 52], and overlook the more challenging task of identifying AI-generated traces, which involve inconsistencies across multiple aspects, including image content, structure, style, and other intrinsic features, showing greater flexibility, diversity, and complexity.

Existing image generation models support multimodal conditional generation, making text-guided regeneration and artifact-aware inpainting based on segmentation masks and explanations possible. Early works [3, 13] employed textdriven approaches for conditional image generation. Other approaches, such as ControlNet [66], extended this to support multimodal conditional inputs, such as masks and edge maps, enabling more targeted and controllable refinement processes. Recently, methods based on LLMs [43, 63] have emerged. For instance, Idea2Img [60] constructs an agent system that leverages GPT-4V’s rethinking capability to progressively refine the prompt for text-to-image (T2I) models and iteratively guides image regeneration, improving both image quality and text-image alignment.

#### 2.2. Multimodal Large Language Models

Building on the success of large language models (LLMs) [1, 50], multimodal large language models (MLLMs) [25, 30, 70] extend capabilities by integrating vision and text processing, achieving remarkable performance in comprehensive tasks. Some benchmarks, such as FakeBench [27] and LOKI [61], have demonstrated the great potential of MLLMs in synthetic image detection, showing that they can provide more interpretable and context-aware detection results. Moreover, some MLLM-based general visual segmentation models [23, 40] have made significant progress, accurately locating objects based on semantic information.

Some works have explored region-level image forgery analysis to guide inpainting models in revising anomalous regions. For example, PAL4Inpaint[65] and PAL4VST [67] inject segmentation masks as conditional input into inpainting models[45, 69] or refiners like SDXL [39]. However, lacking textual artifact explanations, they often resort to simple methods like object removal, failing to preserve semantics. We extend this idea by integrating our image forgery analysis framework, LEGION, into the pipeline to provide generation models with precise artifact cues.

Recent works have explored interpretable synthetic image analysis, providing textual explanations to assist human judgment. For instance, FFAA [18] enhances robustness through its proposed multi-answer intelligent decision system, but is limited to facial data and lacks support for the localization task. Fakeshield [58] carefully designs some modules to generalize across various tampering types, yet lacks analysis for full synthetic images. ForgeryGPT [26] proposes a customized LLM architecture and a novel framework that captures high-order forensic knowledge correlations of forged images from diverse feature spaces, enabling explainable generation and interactive dialogue. Although SIDA explores both tampered and fully synthetic images, it only provides artifact segmentation results for tampered ones. In summary, most methods focus mainly on tampering analysis and have not explored the more complex task of localizing AI-generated artifacts.

### 3. SynthScars Dataset 3.1. Motivation

Recent advancements in generative AI have made it easier to create sophisticated synthetic and tampered content, but existing detection datasets face significant limitations: (i) Outdated Content: Benchmarks like ProGAN [9] rely on early GANs, producing low-fidelity images easily distinguishable from photorealistic outputs of modern models

like Stable Diffusion 3.51 and FLUX2. (ii) Domain Mismatch: Some datasets focus on anime-style or syntheticcartoon images, creating a gap with natural photographic content crucial for real-world applications. (iii) Annotation Issues: Datasets like RichHF-18K [28] use sparse point annotations, sacrificing spatial precision and introducing ambiguity during training, especially for subtle or complex manipulations. (iv) Contour Dependency: Recent datasets, such as SID-Set [17], focus on manipulations with clearcut boundaries, failing to represent real-world forgeries involving irregular or contextually blended alterations, limiting model generalizability.

#### 3.2. Dataset Construction

To address the limitations outlined in Section 3.1, we construct a comprehensive dataset SynthScars through a rigorous pipeline designed to maximize real-world relevance and annotation precision, shown in Figure 2. Specifically, we aggregate and curate samples from multiple public datasets including RichHF-18K [28], Chameleon [59], FFAA [18], and others, following the protocols outlined below.

Data Preprocessing and Quality Control. To ensure a balanced and high-quality synthetic dataset, we first perform source sampling by clustering latent feature representations from a pretrained ResNet-50 [22], followed by uniform sampling from each cluster to mitigate dataset-specific biases while retaining diversity in manipulation types and semantic content. Subsequently, we address limitations (i) and (ii) through a multistage filtering process using Qwen2VL-72B-Instruct [53], which removes low-quality samples (e.g., blurred or compressed artifacts), non-photorealistic content (e.g., anime-style images), and samples exhibiting conspicuous synthetic patterns.

Dual Fine-grained Annotation. To address limitations (iii) and (iv), we adopt irregular polygon masks for annotating artifacts in synthetic images. This fine-grained annotation approach enables precise labeling of artifacts of any shape, size, or location within the image. It also aligns more closely with the characteristics of current synthetic data, where entire objects are rarely artifacts. Compared to pointbased annotations, irregular polygon masks offer superior accuracy in localizing artifact regions. Additionally, [33] categorizes artifacts in synthetic images into three types: physics, distortion, and structure. Inspired by this, we incorporate fine-grained artifact categorization into our annotation process, ensuring systematic and precise labeling of different artifact types. This approach enhances the clarity and organization of our dataset while providing a deeper understanding of synthetic artifact characteristics, enabling more targeted analysis and model training.

- 1https://github.com/Stability-AI/sd3.5
- 2https://github.com/black-forest-labs/flux

#### 3.3. Dataset Statistics

Compared to existing synthetic image datasets, Table 1 highlights SynthScars’ unique combination of pixel-level masks, textual explanations, and artifact-type labels for all samples, setting a new benchmark in synthetic image analysis with 100% valid annotations. For detailed dataset statistics, including image categories and artifact annotations, please refer to the Appendix A.

Dataset Pixel-level Mask Explanation Artifact Type Annotator Valid Sample

CNNSpot [54] - 0 CIFAKE [2] - 0 UniFD [38] - 0

GenImage [71] - 0 Chamelon [59] - 0

AI-Face [29] - 0

PAL4VST [67] Human 10168 RichHF-18K [28] Human 11140

LOKI [61] Human 229 MMTD-Set [58]

–

GPT-4 0 FF-VQA [18] GPT-4 0 SID-Set [17]

–

GPT-4 0

SynthScars Human 12236

Table 1. Comparison with Existing Image Forgery Datasets. The last column shows the number of samples fully synthesized by common generators, with realistic style and valid masks.

–

denotes that only masks of tampered images are provided.

### 4. Method

Recent research, such as LISA [23] and GLaMM [40], have demonstrated the potential of multimodal large language models (MLLMs) in performing general image segmentation. Inspired by this work, we introduce the similar idea to the fully synthetic image forgery analysis task to address the limitations discussed in Section 2. To this end, we propose LEGION, a multi-task image forgery analysis framework that supports deepfake detection, artifact localization, and explanation generation. Furthermore, to enhance the quality and realism of current image generation techniques, we build two training-free pipelines based on LEGION for image regeneration and inpainting, respectively.

#### 4.1. LEGION Architecture

LEGION consists of four core components: (i) Global Image Encoder, (ii) LLM, (iii) Grounding Image Encoder, and (iv) Pixel Decoder, as shown in Figure 3 (a). Each is described in detail in the following sections. First, to extract global features from the input image, we employ ViT-H/14 CLIP as the global image encoder (Eg).

Deepfake Detection. For this task, we follow the conventional approach and formulate it as a binary classification problem, where an image is categorized as either real or fake. We leverage the CLS token from the global feature representation and pass it through a two-layer MLP to predict the detection result. Specifically, given an input image xi, it is first encoded into a feature vector Ix = Eg(xi) ∈ RD

v. We then extract the CLS token (denoted as CLS(·))

|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>Global Image Encoder<br><br>Prompt<br><br>Please provide a detailed analysis of artifacts in this photo, considering <Diverse Artifact Prior>. Output with interleaved segmentation masks for the corresponding parts of the answer.<br><br>[Figure 72]<br><br>[Figure 73]<br><br>Grounding Image Encoder<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>[Figure 74]|
|---|
<br><br>CLS Token<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>Pixel Decoder<br><br>[Figure 78]<br><br>[Figure 79]<br><br>V-L LLM<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>L-P<br><br>Real Fake<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|
<br><br>IMG Tokens<br><br>OUT Prompts<br><br>[Figure 85]<br><br>…<p>Location</p><SEG>…<br><br>R1:Puppy’s right clump<br>R2:Puppy‘s left ear<br><br><br>[Figure 86]<br><br>Final Output<br><br>Detection: Fake Analysis:<br><br>[GLOBAL]<br><br>[LOCAL]<br><br>MLP<br><br>Image Forgery Analysis<br><br>[Figure 87]<br><br>[Figure 88]<br><br>A fluffy puppy with a folded left ear sits on a blanket, with a small black mass resembling a tail nearby.<br><br>Puppy's left clump: The little black mass resembles a puppy's tail. Puppy‘s left ear: …<br><br>|
|---|

|[Figure 89]<br><br>[Figure 90]<br><br>Prompt<br><br>Initial/Revised<br><br>[Figure 91]<br><br>[Figure 92]<br><br>Image Forgery Analysis<br><br>Iteratively<br><br>[Figure 93]<br><br>[Figure 94]<br><br>T2I Model<br><br>Regeneration<br><br>Explanation Feedback<br><br>Revision<br><br>[Figure 95]<br><br>|Memory|
|---|
<br><br>|Initial Feedback|
|---|
<br><br>|Iter 1 Feedback|
|---|
<br><br>|Iter N-1 Feedback|
|---|
<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>Text Refiner|
|---|

(b) Image Regeneration Pipeline

||Region 1 Loca.|Mask| |Expla.|
|---|---|---|---|
|Region 2 Loca.|Mask| |Expla.|
|[Figure 99]| | | |
|Region 3 Loca.|Mask| |Expla.|
|Region 4 Loca.|Mask| |Expla.|
<br><br>[Figure 100]<br><br>[Figure 101]<br><br>Image<br><br>Initial/Inpainted<br><br>[Figure 102]<br><br>[Figure 103]<br><br>Image Forgery<br><br>Analysis<br><br>[Figure 104]<br><br>[Figure 105]<br><br>Inpainting Model<br><br>Inpainting<br><br>Processing Regional Triplets<br><br>Region by Region<br><br>All Regions Done<br><br>Iteratively|
|---|

(a) LEGION Framework (c) Image Inpainting Pipeline

- Figure 3. Architecture Overview. (a) Our proposed framework for image forgery analysis, LEGION. (b) and (c) shows two pipelines for image generation. T2I in (b) is short for text-to-image, and Loca. and Expla. in (c) denotes Location and Explanation, respectively.

Training. We adopt a two-stage independent training strategy. In stage 1, we first train the artifact localization and explanation generation tasks by optimizing segmentation performance using a weighted combination of Binary CrossEntropy (BCE) and Dice loss, while employing CrossEntropy (CE) loss to evaluate the discrepancy between the predicted explanation and the ground truth annotations. In stage 2, we focus on enhancing the model’s forgery detection capability using typical CE loss for classification. The loss of each stage can be formulated as:

from this feature representation and pass it through an MLP classifier (denoted as MLP(·)) to obtain the probability distribution yd over the real and fake classes:

###### yd = MLP CLS(Ix) . (1)

Explanation Generation. To further provide user-friendly natural language explanations, we incorporate a vicunabased LLM (L) to facilitate vision-language alignment. We design a prompt template: “The <image> provides an overview of the image.” + xp, where xp represents the specialized prompt designed for image forgery analysis. Specifically, we first pass the remaining 256 tokens (excluding the CLS token, denoted as I

- Ls1 = λbceLBCE(M,Mˆ ) + λdiceLDice(M,Mˆ )

+ λceLCE(ye,yˆe),

- Ls2 = LCE(yd,yˆd).

(4)

′

x) from the CLIP global image encoder through a vision-to-language (V-L) projection layer (Pvl), which replaces the <image> token in the prompt template. The processed prompt is then concatenated with the forgery analysis prompt xp shown in Figure 3 (a) to construct the final prompt, guiding the LLM to generate textual explanations ye:

#### 4.2. Image Refinement Pipeline

Generation and detection technologies mutually reinforce each other, driving their co-evolution. Inspired by [67], we extend LEGION from a forgery Defender to a generation Controller, enabling artifact-free refinement without additional training data. By providing guidance and feedback to generation models, LEGION facilitates the progressive elimination of potential artifacts, thus enhancing the final output’s quality and realism. To achieve this, we explore two refinement strategies: one leveraging prompt revision for image regeneration, and the other employing inpainting techniques to selectively correct artifact regions. Regeneration. We employ an iterative generation approach by combining prompt revision with a text-to-image (T2I) model, as illustrated in Figure 3(b). Given an initial image I0 and prompt P0, we feed this prompt into T2I model (denoted as Regen(·)) to regenerate an updated image I1, which may be poorly generated artifact regions. I1 is subsequently analyzed using our LEGION framework, which returns detailed explanations of artifact anomalies (e.g. fingers are de-

′

x) . (2)

ye = L xp,Pvl(I

Artifact Localization. To obtain pixel-level artifact masks, we apply a pretrained SAM encoder as our grounding image encoder (El) and design our pixel decoder (D) with reference to the SAM decoder. The output of the LLM will append a specialized token <SEG> after every description of artifact locations (e.g., the cat’s ears), and then a languageto-prompt (L-P) projection layer(Plp) is used to transform the text embeddings related to <SEG> token (vseg) into the decoder’s feature space. Ultimately, D produces the binary masks (M) through the following equation:

M = D El (xi),Plp(vseg) , s.t.,Mi ∈ {0,1}. (3)

|Method<br><br>|Source<br><br>|SynthScars Object Animal Human Scene<br><br>mIoU F1 mIoU F1 mIoU F1 mIoU F1|LOKI<br><br>mIoU F1<br><br>|RichHF-18K mIoU F1|
|---|---|---|---|---|
|HiFi-Net⋆ [12] TruFor⋆ [11] PAL4VST⋆* [67]<br><br>|CVPR23 CVPR23 ICCV23<br><br>|43.74 0.45 45.28 0.03 46.21 0.84 45.90 0.04 46.99 14.82 48.45 17.57 49.02 15.43 48.93 12.64 50.46 19.25 52.55 21.61 59.18 35.70 52.55 19.14<br><br>|39.60 2.41<br><br>46.55 16.70<br><br>47.34 11.58<br><br><br>|44.96 0.39<br><br>48.41 18.03<br><br>49.88 14.78<br><br><br>|
|Ferret□ [62] Griffon□ [64] LISA-v1-7B⋆* [23]<br><br>|ICLR24 ECCV24 CVPR24|30.10 20.81 27.17 15.78 25.54 13.87 30.64 13.79 38.54 23.40 27.76 18.58 23.04 14.81 35.83 14.47 35.49 23.70 32.44 18.77 34.11 17.50 37.56 18.31<br><br>|24.50 18.88 21.96 20.41 31.10 9.29<br><br>|26.52 16.22 28.13 18.19 35.90 21.94|
|InternVL2-8B□ [7] Qwen2-VL-72B□ [53]<br><br>|CVPR24 -|41.08 13.36 41.22 7.83 41.21 3.91 41.68 7.55<br><br>33.89 23.25 32.46 21.98 26.92 14.75 39.00 18.17|42.03 10.06 26.62 20.99<br><br>|39.90 9.58 27.58 19.02|
|LEGION⋆ (Ours)|-|54.62 29.90 54.52 27.43 60.82 39.44 53.67 24.51<br><br>|48.66 16.71<br><br>|50.07 17.41<br><br>|

- Table 2. Performance Comparison of Artifact Localization on SynthScars and Two Benchmarks in Unseen Domains. ∗ denotes methods fine-tuned on SynthScars, while others use pre-trained weights due to unavailable training code. ⋆ and □ represent the models that output segmentation masks and bounding boxes, respectively. Grayed approaches predict most of the image as artifacts and are only for reference, not included in the comparison.

formed). These explanations will be recorded in a memory bank (M). Next, a text refiner (denoted as Revise(·)) revises the initial prompt P0 by rethinking the historical records in the memory bank, enriching the descriptions of regions related to artifacts. This results in a revised prompt P1, which is then used to generate the next iteration of the image. This iterative process continues multiple times, progressively refining both the image and prompt. Specifically, for the t-th iteration (t being a positive integer in the range 0 to N-1), the recurrence formula can be expressed as:

Mt = Mt−1 + LEGION(It), Pt+1 = Revise(Pt,Mt),

It+1 = Regen(Pt+1).

(5)

Inpainting. We also construct a pipeline to facilitate the inpainting process, as illustrated in Figure 3(c), by using an inpainting model to iteratively remove artifacts and progressively enhance image quality. Compared to regeneration, this approach better preserves non-artifact regions since the image is not entirely regenerated but selectively refined only for anomalous areas. Specifically, given an input image I0, we first perform image forgery analysis using our LEGION framework, and organize the feedbacks into a region-wise triplet set (A), where each region is represented

- as (Li,Mi,Ei), with L, M, E denoting the location, mask, and explanation, respectively. Each identified artifact region is then refined using the inpainting model (denoted as Inpaint(·)) by feeding corresponding mask and textual explanation into the model. This is a region-by-region process until all regions are done, finally yielding the refined image. By iterating the above process multiple times, we can obtain higher-quality and more realistic images. For the image It
- at the t-th iteration, this process can be formulated as: At = LEGION(It) = {(Li, Mi, Ei) | i = 1, 2, . . . , k}

(6)

k

Mi · Inpaint(It, Mi, Ei) − It .

It+1 = It +

i=1

### 5. Experiments

#### 5.1. Experimental Setup

Implementation Details We adopt the pretrained weights of GLaMM due to its strong capability in the grounded conversation generation task it proposed. In training stage 1, we fine-tune GLaMM using LoRA on 8 NVIDIA A100 GPUs with α = 8 and a batch size of 2 per device. The initial learning rate is set to 1e-4, and the weight hyperparameters λce, λdice, and λbce in Eq 4, which balance the three types of loss, are set to 1.0, 0.2, and 0.4, respectively. In training stage 2, we directly train the MLP on ProGAN using 8 NVIDIA A100 GPUs, with an initial learning rate of 1e-3 and a batch size of 64 per device.

Evaluation Metrics. For the artifact localization task, we evaluate the segmentation performance by reporting the mean Intersection over Union (mIoU) of foreground and background regions, as well as the overall F1 scores, following [11, 12, 67]. Additionally, to assess the alignment between the generated textual explanations and ground truth, we employ two metrics: ROUGE-L, which is based on n-gram overlap and sequence matching, and Cosine Similarity Score (CSS), which computes the cosine similarity between text embeddings obtained from a pre-trained language model, following FakeShield [58] (more details in Appendix B.2). In the image refinement section, we assess the quality and realism of regenerated and inpainted images using the Human Preference Score (HPS) proposed in [56].

#### 5.2. Localization Evaluation

To evaluate artifact localization performance, we use training split of SynthScars as training data and test on its test split for in-domain assessment. We further examine models’ generalization performance to unseen domains using LOKI [61] and RichHF-18K [28], which have been filtered to retain only images with realistic styles. We com-

###### LEGION (Ours)

###### PAL4VST InternVL2-8B Ground Truth

pare LEGION with state-of-the-arts (SOTAs) for fully synthetic artifact localization. Baselines include traditional expert models such as HiFi-Net [12], TruFor [11], and PAL4VST [67], as well as VLMs for object-grounding, including Ferret [62], Griffon [64], and LISA [23]. Moreover, we benchmark general-purpose MLLMs that have shown great performance across various tasks, including InternVL2 [7], Qwen2-VL [53] and DeepSeek-VL2 [57].

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

The thumb of the left hand is missing, water droplets and white spots appear between the fingers, and the size of the fingers is not proportional.

grainy texture and glare across the entire image, asymmetrical buildings and horizontal line distortions in the background……

The number of fingers is abnormal, and the fingers are twisted and abnormally shaped.

Artifact Explanation Unavailable for Expert Model

Results in Table 2 demonstrate that LEGION achieves SOTA performance across all three evaluation datasets, despite its F1 score on RichHF-18K being slightly lower than that of LISA-v1-7B and TruFor. Compared with traditional experts, LEGION outperforms the strongest expert model, PAL4VST, by 10.65 points in F1 score for Object category on SynthScars, and also consistently surpassing it on the other two datasets. For object-grounding VLMs and general-purpose MLLMs, we observe that these models struggle with artifact localization due to the lack of pretraining and prior knowledge specific to this task. This leads to two extreme behaviors: some models fail to identify foreground regions altogether (e.g., DeepSeek-VL2), while others overestimate artifacts, treating most of the image as a huge artifact (e.g., Ferret, Griffon, and Qwen2-VL), resulting in low mIoU but artificially high F1 scores due to extreme foreground recall. InternVL2 and LISA fine-tuned on SynthScars exhibit these extremes less severely, yet LEGION still outperforms both across the majority of metrics. The visualization comparison of various methods is presented in Figure 4.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

reflections on the car‘s surface, slight asymmetry in shadows and distorted text on the license plate……

The car license plate text is distorted and unreadable.

Artifact Explanation Unavailable for Expert Model

The license plate number was distorted and unrecognizable.

Figure 4. Comparison of artifact segmentation and explanations across different methods: PAL4VST, InternVL2-8B, and our proposed LEGION, alongside the ground truth.

rameters. In our evaluation, we test on SynthScars and LOKI which contain detailed artifact explanations, measuring ROUGE-L for surface-level structural alignment and CSS for semantic equivalence to jointly assess both lexical coherence and contextual fidelity. Since SynthScars has a fixed response format and LEGION trained on it demonstrates structural advantages, we convert the outputs of others to the same format for fair comparison. The experimental results are presented in Table 3, indicating the superior performance of LEGION across both datasets.

Robustness Study. We also systematically compare the localization performance between LEGION and PAL4VST under three types of perturbations. The results can be found in Table 10 of Appendix B, indicating the robustness of our model under strong interference conditions—a critical capability unattainable by conventional approaches.

|HPS<br><br>|Regeneration<br><br>|Inpainting|
|---|---|---|
|Pre-refined Score (Avg.) Post-refined Score (Avg.)<br><br>|31.24 33.36|29.57 30.20<br><br>|
|Growth Rate ↑|6.98%<br><br>|2.14%|

Table 4. HPS Comparison Before and After Refinement in Regeneration and Inpainting. Scores are normalized to the range of 0–100 for better visualization and comparison.

#### 5.3. Explanation Performance

|Method Qwen2-VL [53]|Date 24.09|Params 72B|SynthScars<br><br>ROUGE-L↑ CSS↑ 25.84 58.15|LOKI<br><br>ROUGE-L↑ CSS↑ 11.80 37.64|
|---|---|---|---|---|
|LLaVA-v1.6 [31]|24.01|7B|29.61 61.75|16.07 41.07|
|InternVL2 [7]|24.07|8B|25.93 56.89|10.10 39.62|
|Deepseek-VL2 [57]|24.12|27B|25.50 47.77|6.70 28.76|
|GPT-4o [19]|24.12|-|22.43 53.55|9.61 38.98|
|LEGION (Ours)|25.03|8B|39.50 72.60|18.55 45.96|

#### 5.4. Detection Performance

Following previous works [4, 38, 47, 48, 68], we train LEGION on ProGAN [9] and evaluate its cross-generator generalization on the UniversalFakeDetect benchmark [38]. As shown in Table 5, LEGION achieves the highest accuracy on GANs, CRN, and IMLE, secures competitive secondplace accuracy on SITD, and maintains comparable detection performance in other generators.

- Table 3. Comparison of Multimodal Models in Artifact Explanation Generation. Metrics are normalized to the range of 0–100 for better visualization and comparison.

To better assess the model interpretability and capability to generate natural language explanations for artifacts, we conduct a comparative analysis of the latest released open-source (e.g., DeepSeek-VL2) and closed-source models (e.g., December,2024 updated GPT-4o) with varying pa-

#### 5.5. Image Refinement Cases

To evaluate LEGION’s effectiveness in guiding generation models toward higher-quality, more realistic images, we randomly sample 200 images from the SynthScars test set.

|Method|GANs<br><br>|Deepfakes|Perceptual Loss CRN IMLE<br><br>|Low Level Vision SITD SAN|Diffusion|
|---|---|---|---|---|---|
|Co-occurence [35] Freq-spec [68] CNNSpot [54] Patchfor [4] UniFD [38] LDGard [46] FreqNet [47] NPR [48] LEGION (Ours)<br><br>|75.17 75.28 85.29 69.97 95.25 89.17 94.23 94.16 97.01<br><br>|59.14 45.18 53.47 75.54 66.60 58.00 97.40 76.89 63.37<br><br>|73.06 87.21 53.61 50.98 86.31 86.26 72.33 55.30 59.50 72.00 50.74 50.78 71.92 67.35 50.00 50.00 90.78 98.93<br><br>|68.98 60.42 47.46 57.12 66.67 48.69 75.14 75.28 63.00 57.50 62.50 50.00 88.92 59.04 66.94 98.63 79.44 57.76<br><br>|85.53 69.00 58.63 72.54 82.02 89.79 83.34 94.54 83.10<br><br>|

###### Vanilla Iteration 1 Iteration 2 Iteration 3

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

Wall Color (L)

Wall Color (L) Window Shape (R)

Wall Color (L) Window Shape (R)

Wall Color (L)

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Window Shape (R)

Window Shape (R)

[Figure 134]

[Figure 135]

[Figure 136]

Detect Feedback

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

###### Table 5. Comparison of Synthetic Image Detection on UniversalFakeDetect Benchmark. All methods are trained on ProGAN.

Visual Mask

###### Case 1 (Easy): Style Distortion Adjustment

[Figure 141]

[Figure 142]

Artifact Type: Distortion Solution: Style Adaptation

The reflection of the house does not match the color and shape of the house. Physical Laws Violation!

[Figure 143]

Explanation

[Figure 144]

Figure 6. Case Study of Image Inpainting.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Model Feedback

At a global level, I have found: A cartoon-style man with curly hair and glasses is depicted, featuring noticeable asymmetry between two sides of his face. In detail, I have found the following artifacts: ……

model to refine the region. Two rounds of optimization are needed to correct the structure and achieve a natural form.

Iteration 1 Prompt

Initial Prompt

......, natural lighting, realistic style, focus on facial details, warm atmosphere.

A person with curly hair wearing glasses and a yellow top.

##### 5.5.2. Image Inpainting under Feedback Guidance

###### Case 2 (Hard): Detailed Structure Reshape

[Figure 149]

[Figure 150]

[Figure 151]

Leveraging the powerful inpainting capability of the SDXL model3, we explore region-level image restoration guided by LEGION’s feedback, including artifact masks and anomaly explanations. For each image, we perform 3 iterations of the inpainting pipeline. Figure 6 illustrates the iterative refinement process for correcting reflections in a synthetic image. In the original image, the left reflection on the water mismatches the wall’s color, while the right reflection contains an unrealistic window shape, violating physical laws. Through multiple iterations, LEGION progressively identifies entire reflection region, highlighting color and shape discrepancies to guide the inpainting process. By the 3-rd iteration, the artifacted region is successfully refined, achieving high-quality restoration. Notably, such physical artifacts requiring global reasoning are often challenging for previous localization models to detect. This underscores LEGION’s strong capability in addressing artifact localization in fully synthetic images.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

###### Iteration 2 Prompt

###### Initial Prompt Iteration 1 Prompt

...... Her left hand is adorned with a ring, and she wears elegant earrings. The backdrop is softly blurred ......

...... Her left hand is adorned with a diamond ring, and she wears elegant earrings. The background is softly blurred ......

A woman with short hair wearing a ring and earrings poses against a blurred background.

Figure 5. Case studies of Image Regeneration. (Top) Style Distortion Adjustment, (Bottom) Detailed Structure Reshape.

Following [6], we compute the average HPS scores for images before and after refinement (for both regeneration and inpainting) and measure the average growth rate of each refinement process. The results are summarized in Table 4.

##### 5.5.1. Prompt Revision and Image Regeneration

We consider the inherent gap between the original image generator and the advanced refiner Stable Diffusion 3.5, which may independently lead to significant quality improvements unrelated to LEGION’s guidance. To control external influences, we first generate a concise one-sentence description of the original image using GPT-4o, then use Stable Diffusion 3.5 to regenerate the image as the refinement starting point. Finally, we perform two rounds of iterative regeneration to produce the final result. Figure 5 showcases two image regeneration cases. Case 1: LEGION detects a cartoonish style in the original image and refines the prompt with constraints like “natural lighting” and “realistic style”. After one refinement round, the image becomes significantly more realistic. Case 2: The woman’s left pinky finger is deformed in the initial image. Guided by LEGION, subsequent prompts add hand-specific details, enabling the

### 6. Conclusion

In this work, we explore the challenging task of fully synthetic image forgery analysis by introducing SynthScars dataset, which features diverse and high-difficulty artifact instances that are not constrained by local contours and require global understanding. To enable more fine-grained artifact analysis, we propose LEGION, a MLLM-based framework that enhances interpretability compared to traditional methods. Experimental results show that it achieves superior performance and strong robustness across multiple benchmarks and evaluation metrics. It also demonstrates

3https : / / huggingface . co / diffusers / stable diffusion-xl-1.0-inpainting-0.1

strong potential as a controller for guided image generation and inpainting from both qualitative and quantitative perspectives. While our proposed dataset and method address some key limitations in image forgery analysis, the inherent diversity and flexibility of fully synthetic artifacts leave ample room for further improvements in the future. We also hope more researchers to explore this domain and contribute to the responsible and ethical use of generative AI.

### 7. Appendix

##### Contents of the Appendices:

- Section A. Detailed Construction Process and Statistics

of the SynthScars Dataset.

- Section B. More Experimental Details for Image Forgery

Analysis Task.

- Section C. Robustness Comparison Between LEGION

and Expert Model under Various Perturbations.

- Section D. Additional Visual Examples of LEGION.
- Section E. Limitations and Failure Case Analysis.

### A. SynthScars Dataset

#### A.1. Artifact Definition

Inspired by [33], we categorize artifacts in synthetic images into three types: physics, distortion, and structure. To eliminate subjective differences among annotators and to clarify and standardize the criteria for artifact classification during the annotation process, we established a guideline that explicitly defines the nature and scope of various artifacts, as shown in Table 8.

#### A.2. Annotation Details

We recruited 12 experienced annotators with higheducation backgrounds and provided them with dedicated training before the annotation task. They were required to strictly follow the guideline for data annotation and discard samples where artifacts were entirely imperceptible to human eyes. The annotation process for 12,236 samples in SynthScars took a total of 240 hours and underwent multiple rounds of quality inspection.

#### A.3. Data Curation

To obtain high-quality, deceptive, and challenging synthetic images, we carry out a multistage filtering process using Qwen2-VL-72B-Instruct [53], which removes lowquality samples (e.g., blurred or compressed artifacts), nonphotorealistic content (e.g., cartoonish or watercolor-style images), and samples exhibiting conspicuous synthetic patterns. Specifically, we designed a prompt, as shown in the Table 9, for the model to sequentially inspect each data sample against the given criteria. Only samples that meet all the standards are retained.

Image Content Human Object Animal Scene Total

Train 6253 1940 1183 1860 11236 Test 587 162 134 117 1000

Total 6840 2102 1317 1977 12236

- Table 6. Statistics on Image Content. SynthScars encompasses a diverse range of real-world scenarios, including 12,236 fully synthesized images from different generators.

Artifact Type Physics Distortion Structure Total

Train 1431 1249 21233 23913 Test 111 136 2406 2653

Total 1542 1385 23639 26566

- Table 7. Statistics on Artifact Types. SynthScars classifies artifacts into three fine-grained anomaly types, and contains a total of 26,566 artifact instances.

- A.4. Dataset Statistics

As shown in Table 6, SynthScars includes 12,236 fully synthesized images across diverse real-world scenarios, with 11,236 training and 1,000 test samples categorized into human, object, animal, and scene. The dataset features 26,566 artifact instances (Table 7), annotated with irregular polygon masks and classified into three types: physics-related (6%), distortion (5%), and structural anomalies (89%).

- B. Experimental Details

- B.1. Prompt Design

When designing the prompt, in order to fully unleash the LLM’s broad reasoning ability, we incorporated prior knowledge of different artifacts (denoted as <Diverse Artifact Prior>). Specifically, it consists of common cases from the three types of artifacts we defined, guiding the model to examine the image from the corresponding perspectives. To provide a concrete example, we define it as follows:

Physics artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), Structure artifacts (e.g., deformed objects, asymmetry, or distorted text), and Distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation)

- B.2. Explanation Evaluation

Following Fakeshield [58], we use paraphrase-MiniLM-L6v24 from HuggingFace as our text embedding model to transform the outputs into semantic feature space.

4https://huggingface.co/sentence-transformers/ paraphrase-MiniLM-L6-v2

##### Artifact Definition

###### 1. Physics

- (a) Optical Display: These artifacts arise from inconsistencies in the propagation and reflection of light, violating fundamental optical principles. They can occur across different objects and scenes, leading to unrealistic visual effects. Common cases include incorrect reflections, shadows, and light source positioning errors, causing synthetic images to deviate from real-world optical phenomena.
- (b) Physical Law Violations: These artifacts result from the failure to adhere to fundamental physical laws during image synthesis. They typically manifest as illogical scenes, such as water flowing upward or objects floating in mid-air, which contradict natural laws.
- (c) Space and Perspective: These artifacts stem from inaccuracies in object proportions and spatial relationships during image generation, leading to inconsistencies with real-world perspective rules. Examples include incorrect depth perception, mismatched object sizes, or spatial distortions that prevent accurate perspective alignment.

###### 2. Structure

- (a) Deformed Objects: These artifacts arise when the shape or structure of objects is distorted due to errors in the generative model. Contributing factors include geometric inconsistencies, texture mapping errors, and rendering issues.
- (b) Asymmetrical Objects: These artifacts occur when an object exhibits unnatural asymmetry, deviating from expected structural balance.
- (c) Incomplete/Redundant Structures: These artifacts appear as missing or excessive structural components, leading to unrealistic representations of objects.
- (d) Illogical Structures: These artifacts involve the generation of unrecognizable or non-existent objects, as well as the appearance of elements that should not logically exist within the given context.
- (e) Text Distortion and Illegibility: These artifacts include warped, irregular, or unrecognizable text, affecting the readability and coherence of textual content within the generated image.

###### 3. Distortion

- (a) Color and Texture: These artifacts result from errors in color rendering or color space conversion, leading to unnatural hues, inappropriate saturation, or other inconsistencies in color perception.
- (b) Noise and Blurring: These artifacts are associated with image noise reduction and clarity enhancement processes. They may arise when algorithms fail to effectively remove noise or introduce excessive blurring, causing local details to appear distorted or unnatural.
- (c) Artistic Style: These artifacts occur when synthetic images exhibit unintended stylization, such as cartoonish or painterly appearances that deviate from realistic textures. Such distortions are often caused by errors in style transfer or texture generation algorithms.

Table 8. Artifact Definition. We clearly define three types of artifacts and require annotators to strictly follow this guideline for annotation.

### C. Robustness Study

We compare the artifact localization performance between LEGION and PAL4VST (the strongest expert model from Table 2) on SynthScars under three types of distortion. Table 10 reveal that Gaussian noise induces the most severe performance degradation, followed by Gaussian blur, while JPEG compression exhibits the least negative effects. Notably, as intensity increases, LEGION remains stable, while PAL4VST degrades sharply, highlighting our model’s superior robustness under strong interference—an unattainable ability for traditional expert models.

### D. More Visual Examples

In addition to comparing LEGION’s predictions with other methods, including multi-modal large language models and

expert models, this section provides an extended visualization of artifact segmentation masks and their corresponding explanations. As shown in Figure 7, LEGION excels in predicting artifacts on highly realistic synthetic images, achieving both positional and contour accuracy in segmentation. The accompanying explanations are insightful, highlighting not only the location of the artifact but also offering a plausible rationale for its artificial nature. These results highlight LEGION’s ability to deliver precise artifact detection alongside interpretable insights, enhancing the transparency and trustworthiness of synthetic image generation.

### E. Limitations and Analysis

While our model demonstrates promising results in detecting and segmenting artifacts on AI-generated images, there remain areas for improvement. A qualitative analysis of

##### System Prompt

You are a helpful assistant. Analyze the given images based on the following three criteria and assign one label to each image. You only need to return the label for each image without providing any additional explanations: Evaluation Criteria

###### 1. Clarity

- (a) The image should be well-lit, sharp, and visually clear without blurriness, noise, or distortion.
- (b) The image must not show obvious signs of artificial manipulation, such as pixelated edges or unnatural distortions.

###### 2. Safety

- (a) The image must not contain violence, blood, gore, explicit sexual content, hate symbols, discriminatory elements, or any harmful or inappropriate material.
- (b) Any content that could evoke strong negative emotions or discomfort should be classified as unsafe.

###### 3. Realism

- (a) The image should look realistic and have a photo-like appearance.
- (b) It must not be cartoonish, animated, or heavily stylized in an artistic manner.

Labeling Task Assign one of the following labels to each image:

- 1. Acceptable: If the image meets all three criteria;
- 2. Rejected[Clarity]: If the image is unclear, blurry, or distorted;
- 3. Rejected[Safety]: If the image contains unsafe or inappropriate content;
- 4. Rejected[Realism]: If the image is stylized, animated, or lacks realism.

##### User Prompt

Please strictly follow the instructions to label the input image: {image}

Table 9. Curation Prompt. Only samples that meet all the standards are retained.

|Distortion<br><br>|PAL4VST mIoU F1|LEGION (Ours) mIoU F1<br><br>|
|---|---|---|
|No Distortion<br><br>|56.10 29.21|59.41 36.96|
|JPEG Comp. (QF = 50) JPEG Comp. (QF = 35) JPEG Comp. (QF = 20)<br><br>|55.95 28.85 55.55 27.60 55.01 (-1.9%) 26.36 (-9.8%)<br><br>|57.78 33.97<br>58.04 34.08<br><br><br>57.91 (-2.5%) 34.28 (-7.3%)|
|Gaussian Noise (σ = 0.1)<br><br>Gaussian Noise (σ = 0.2)<br><br>Gaussian Noise (σ = 0.3)<br><br><br>|56.01 28.96 54.42 25.16 52.91 (-5.7%) 21.11 (-27.7%)|57.31 33.00 56.77 32.52 56.49 (-4.9%) 32.12 (-13.1%)<br><br>|
|Gaussian Blur (Ksize = 5) Gaussian Blur (Ksize = 9) Gaussian Blur (Ksize = 15)|55.62 27.76 54.58 25.23 53.24 (-5.1%) 22.30 (-23.7%)<br><br>|57.75 33.78 57.27 32.63 57.50 (-3.2%) 33.52 (-9.3%)|

Table 10. Robustness Comparison Under Different Perturbations. LEGION significantly outperforms the strongest existing expert model under severe JPEG compression (denoted as JPEG Comp.), Gaussian noise, and Gaussian blur (Ksize represents kernel size). Values in parentheses indicate degradation ratios, with the more robust method highlighted in green, otherwise in red.

as minor distortions or unnatural textures, can be difficult to perceive even for the human eye. We argue that the model is being overwhelmed by the sheer volume of information, leading to a prioritization of more prominent anomalies at the expense of smaller, less conspicuous ones.

failure cases reveals two primary challenges. First, in scenarios with high scene complexity and a multitude of elements, our model sometimes tends to miss subtle artifact regions. As illustrated in Figure 8, the predicted masks may incompletely cover the areas affected by anomalies, particularly when the artifacts are intertwined with intricate background details. Second, the model struggles with detecting very subtle artifacts that occupy a small image area, especially in human portraits. These artifacts, often manifesting

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

The cat‘s right hind limb is abnormally long, swollen in the middle, with a foot extending from the tip of the tail.

The girl had only three fingers on her left hand, the tail finger was missing……

The cat's tongue is red and umbrella-shaped, which looks very abrupt.

The woman‘s left hand and hair are strangely twisted together. The woman’s left hand and hair are strangely twisted together.

The middle section of the outer ear rim of the right

The foot is twisted and has an abnormal shape.

ear is missing. There is an extra fingertip on the top of the right thumb……

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

The cat's tail is too long and flowing, and it appears abruptly in the picture, defying the laws of physics.

The woman's hands has an extra finger. The woman's right ear is fused with her hair.

The fingers are deformed and twisted, and the length and thickness of the

The man's right hand has an extra finger on the index finger and a missing tail finger.

The cat's tongue is bright red, which is inconsistent with the color of its fur.

The bird has an extra claw on its left foot.

fingers are significantly different from those of the same age.

- Figure 7. More Visualization of Artifact Segmentation Masks and Corresponding Explanations for Identified Artifacts. The figure illustrates a qualitative comparison between the ground truth (Top row) and the corresponding predictions obtained from our proposed model (Bottom row).

[Figure 169]

[Figure 170]

The table has an extra leg on the left side.

[Figure 171]

[Figure 172]

The middle cat's body has a colorful outfit, which has an unnatural appearance.

[Figure 173]

[Figure 174]

The table on the left side has an extra leg.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

The character’s left eye is deformed, the pupil is fused to the lower eyelid, and the tail of the eye is missing.

The woman's right hand is deformed and twisted, with an unnatural appearance.

The left side of the sofa is missing the foot. The coffee table has too many legs for the norm.

The black cat is missing its right eye. The black cat‘s mouth is missing. The black cat is missing its right ear……

The legs of the chair and the table are abnormally fused together, looking like a whole, and appearing deformed.

The fingers of the woman‘s left hand were fused together, indistinguishable from her five fingers……

Teeth are different sizes and have an unnatural appearance. There is a bright light in the left eye and the right inner eyelid is blurred.

[Figure 179]

The man's right eye lacked luster, and the white of his eye was too pale, in stark contrast to the left.

[Figure 180]

The character's left eye is deformed with the left eye. The character's left ear is deformed and distorted

- Figure 8. Examples of failures in complex scenes and intricate small artifacts. Each case includes artifacts segmentation mask and corresponding explanations. The first row depicts the ground truth, while the second row shows the corresponding predictions generated by our model.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023. 3
- [2] Jordan J Bird and Ahmad Lotfi. Cifake: Image classification and explainable identification of ai-generated synthetic images. IEEE Access, 12:15642–15650, 2024. 4
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 3
- [4] Lucy Chai, David Bau, Ser-Nam Lim, and Phillip Isola. What makes fake images detectable? understanding properties that generalize. In Computer vision–ECCV 2020: 16th European conference, Glasgow, UK, August 23–28, 2020, proceedings, part XXVI 16, pages 103–120. Springer, 2020. 7, 8
- [5] Jiaxuan Chen, Jieteng Yao, and Li Niu. A single simple patch is all you need for ai-generated image detection, 2024. 2
- [6] Zhaorun Chen, Yichao Du, Zichen Wen, Yiyang Zhou, Chenhang Cui, Zhenzhen Weng, Haoqin Tu, Chaoqi Wang, Zhengwei Tong, Qinglan Huang, et al. Mj-bench: Is your multimodal reward model really a good judge for text-toimage generation? arXiv preprint arXiv:2407.04842, 2024. 8
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 6, 7
- [8] Junxian Duan, Yuang Ai, Jipeng Liu, Shenyuan Huang, Huaibo Huang, Jie Cao, and Ran He. Test-time forgery detection with spatial-frequency prompt learning. International Journal of Computer Vision, pages 1–16, 2024. 2
- [9] Hongchang Gao, Jian Pei, and Heng Huang. Progan: Network embedding via proximity generative adversarial network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1308–1316, 2019. 3, 7
- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 1
- [11] Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa Verdoliva. Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20606–20615,

2023. 2, 6, 7

- [12] Xiao Guo, Xiaohong Liu, Zhiyuan Ren, Steven Grosz, Iacopo Masi, and Xiaoming Liu. Hierarchical fine-grained image forgery detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3155–3165, 2023. 2, 6, 7

- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 3
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [15] Jinbin Huang, Chen Chen, Aditi Mishra, Bum Chul Kwon, Zhicheng Liu, and Chris Bryan. Asap: Interpretable analysis and summarization of ai-generated image patterns at scale. arXiv preprint arXiv:2404.02990, 2024. 2
- [16] Yihao Huang, Felix Juefei-Xu, Qing Guo, Yang Liu, and Geguang Pu. Fakelocator: Robust localization of ganbased face manipulations. IEEE Transactions on Information Forensics and Security, 17:2657–2672, 2022. 3
- [17] Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He, Xingyu Zhao, Bei Peng, Baoyuan Wu, Xiaowei Huang, and Guangliang Cheng. Sida: Social media image deepfake detection, localization and explanation with large multimodal model. arXiv preprint arXiv:2412.04292, 2024. 2, 4
- [18] Zhengchao Huang, Bin Xia, Zicheng Lin, Zhun Mou, and Wenming Yang. Ffaa: Multimodal large language model based explainable open-world face forgery analysis assistant. arXiv preprint arXiv:2408.10072, 2024. 2, 3, 4
- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7
- [20] Yonghyun Jeong, Doyeon Kim, Youngmin Ro, and Jongwon Choi. Frepgan: robust deepfake detection using frequencylevel perturbations. In Proceedings of the AAAI conference on artificial intelligence, pages 1060–1068, 2022. 2
- [21] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 1
- [22] Brett Koonce and Brett Koonce. Resnet 50. Convolutional neural networks with swift for tensorflow: image recognition and dataset categorization, pages 63–72, 2021. 4
- [23] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024. 3, 4, 6, 7
- [24] Trung-Nghia Le, Huy H Nguyen, Junichi Yamagishi, and Isao Echizen. Openforensics: Large-scale challenging dataset for multi-face forgery detection and segmentation inthe-wild. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10117–10127, 2021. 1
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 3

- [26] Jiawei Li, Fanrui Zhang, Jiaying Zhu, Esther Sun, Qiang Zhang, and Zheng-Jun Zha. Forgerygpt: Multimodal large language model for explainable image forgery detection and localization. arXiv preprint arXiv:2410.10238, 2024. 2, 3

- [27] Yixuan Li, Xuelin Liu, Xiaoyang Wang, Shiqi Wang, and Weisi Lin. Fakebench: Uncover the achilles’ heels of fake images with large multimodal models. arXiv preprint arXiv:2404.13306, 2024. 1, 3
- [28] Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19401–19411, 2024. 1, 4, 6
- [29] Li Lin, Xin Wang, Shu Hu, et al. Ai-face: A million-scale demographically annotated ai-generated face dataset and fairness benchmark. arXiv preprint arXiv:2406.00783, 2024. 4
- [30] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 7
- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,

2022. 1

- [33] Melanie Mathys, Marco Willi, and Raphael Meier. Synthetic photography detection: A visual guidance for identifying synthetic images created by ai. arXiv preprint arXiv:2408.06398, 2024. 4, 9
- [34] Ghazal Mazaheri and Amit K Roy-Chowdhury. Detection and localization of facial expression manipulations. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1035–1045, 2022. 3
- [35] Lakshmanan Nataraj, Tajuddin Manhar Mohammed, Shivkumar Chandrasekaran, Arjuna Flenner, Jawadul H Bappy, Amit K Roy-Chowdhury, and BS Manjunath. Detecting gan generated fake images using co-occurrence matrices. arXiv preprint arXiv:1903.06836, 2019. 8
- [36] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1
- [37] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 1

- [38] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards universal fake image detectors that generalize across generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24480– 24489, 2023. 4, 7, 8
- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3

- [40] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024. 3, 4
- [41] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626,

2017. 2

- [42] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013. 2
- [43] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, et al. Dreamsync: Aligning textto-image generation with image understanding feedback. arXiv preprint arXiv:2311.17946, 2023. 3
- [44] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319–3328. PMLR, 2017. 2
- [45] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2149– 2159, 2022. 3
- [46] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, and Yunchao Wei. Learning on gradients: Generalized artifacts representation for gan-generated images detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12105–12114, 2023. 8
- [47] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5052–5060, 2024. 2, 7, 8
- [48] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28130–28139, 2024. 7, 8
- [49] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 1
- [50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [51] Michael Tschannen, Andr´e Susano Pinto, and Alexander Kolesnikov. Jetformer: An autoregressive generative model

of raw images and text. arXiv preprint arXiv:2411.19722,

2024. 1

- [52] Junke Wang, Zuxuan Wu, Jingjing Chen, Xintong Han, Abhinav Shrivastava, Ser-Nam Lim, and Yu-Gang Jiang. Objectformer for image manipulation detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2364–2373,

2022. 3

- [53] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4, 6, 7, 9
- [54] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020. 4, 8
- [55] Yuan Wang, Kun Yu, Chen Chen, Xiyuan Hu, and Silong Peng. Dynamic graph learning with content-guided spatialfrequency relation reasoning for deepfake detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7278–7287, 2023. 2
- [56] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 6

- [57] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding, 2024. 7
- [58] Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, and Jian Zhang. Fakeshield: Explainable image forgery detection and localization via multi-modal large language models. The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 4, 6, 9
- [59] Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. A sanity check for aigenerated image detection. The Thirteenth International Conference on Learning Representations, 2025. 2, 4
- [60] Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Idea2img: Iterative self-refinement with gpt-4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541, 2023. 3
- [61] Junyan Ye, Baichuan Zhou, Zilong Huang, Junan Zhang, Tianyi Bai, Hengrui Kang, Jun He, Honglin Lin, Zihao Wang, Tong Wu, et al. Loki: A comprehensive synthetic data detection benchmark using large multimodal models. arXiv preprint arXiv:2410.09732, 2024. 1, 3, 4, 6
- [62] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and

- Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. The Twelfth International Conference on Learning Representations, 2023. 6, 7
- [63] Qifan Yu, Juncheng Li, Wentao Ye, Siliang Tang, and Yueting Zhuang. Interactive data synthesis for systematic vision adaptation via llms-aigcs collaboration. arXiv preprint arXiv:2305.12799, 2023. 3
- [64] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In European Conference on Computer Vision, pages 405–422. Springer, 2024. 6, 7
- [65] Lingzhi Zhang, Yuqian Zhou, Connelly Barnes, Sohrab Amirghodsi, Zhe Lin, Eli Shechtman, and Jianbo Shi. Perceptual artifacts localization for inpainting. In European Conference on Computer Vision, pages 146–164. Springer,

2022. 2, 3

- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 3
- [67] Lingzhi Zhang, Zhengjie Xu, Connelly Barnes, Yuqian Zhou, Qing Liu, He Zhang, Sohrab Amirghodsi, Zhe Lin, Eli Shechtman, and Jianbo Shi. Perceptual artifacts localization for image synthesis tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7579– 7590, 2023. 2, 3, 4, 5, 6, 7
- [68] Xu Zhang, Svebor Karaman, and Shih-Fu Chang. Detecting and simulating artifacts in gan fake images. In 2019 IEEE international workshop on information forensics and security (WIFS), pages 1–6. IEEE, 2019. 7, 8
- [69] Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. arXiv preprint arXiv:2103.10428, 2021. 3
- [70] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3
- [71] Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, and Yunhe Wang. Genimage: A million-scale benchmark for detecting ai-generated image. Advances in Neural Information Processing Systems, 36:77771–77782, 2023. 4

