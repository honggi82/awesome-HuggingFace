# arXiv:2601.11522v1[cs.CV]16Jan2026

## UniX: Unifying Autoregression and Diffusion for Chest X-Ray Understanding and Generation

Ruiheng Zhang1,∗, Jingfeng Yao2,∗, Huangxuan Zhao1,∗,†, Hao Yan1, Xiao He1, Lei Chen2, Zhou Wei1, Yong Luo1, Zengmao Wang1, Lefei Zhang1, Dacheng Tao3, Bo Du1,†

1Wuhan University 2Huazhong University of Science and Technology 3Nanyang Technological University

### Abstract

Despite recent progress, medical foundation models still struggle to unify visual understanding and generation, as these tasks have inherently conflicting goals: semantic abstraction versus pixel-level reconstruction. Existing approaches, typically based on parameter-shared autoregressive architectures, frequently lead to compromised performance in one or both tasks. To address this, we present UniX, a next-generation unified medical foundation model for chest X-ray understanding and generation. UniX decouples the two tasks into an autoregressive branch for understanding and a diffusion branch for highfidelity generation. Crucially, a cross-modal self-attention mechanism is introduced to dynamically guide the generation process with understanding features. Coupled with a rigorous data cleaning pipeline and a multi-stage training strategy, this architecture enables synergistic collaboration between tasks while leveraging the strengths of diffusion models for superior generation. On two representative benchmarks, UniX achieves a 46.1% improvement in understanding performance (Micro-F1) and a 24.2% gain in generation quality (FD-RadDino), using only a quarter of the parameters of LLM-CXR. By achieving performance on par with task-specific models, our work establishes a scalable paradigm for synergistic medical image understanding and generation. Codes and models are available at https://github.com/ZrH42/UniX.

### 1. Introduction

In recent years, vision–language pretraining–based medical foundation models have shown remarkable success in both understanding [1, 4, 6, 19, 24, 29, 30, 42] and generation [2, 3, 9, 26, 38, 39] tasks. As research progresses, medical image understanding and generation are increas-

###### Quantitative Results: Better Performance

[Figure 1]

[Figure 2]

Uni. Model

###### UniX

| |[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|---|
|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]| |

LLM-CXR HealthGPT

Und. Model

RGRG

LLaVA-Med

GPT-4V

Gen. Model

SD V3.5 Lumina 2.0 Flux.1-Dev

Avg. Metrics of All Tasks

UniX LLM-CXR

Qualitative Results: Ground Truth Comparison

Task1: Medical Understanding Task2: Medical Generation

[Figure 12]

[Figure 13]

Clinical Report

There is significant opacification of the right hemithorax, mostly secondary to a large pleural effusion. Underlying

parenchymal consolidation cannot

be excluded. The left lung is well inflated with no focal consolidation. Cardiomediastinal silhouette is midline and within normal limits.

Gound Truth

[Figure 14]

[Figure 15]

UniX Report Gound Truth

FINDINGS: The tube is relatively high and could be advanced by 1 to 2 cm. Bilateral pleural effusions,

FINDINGS: The endotracheal tube tip is 2 cm above the carina. The nasogastric tube tip is in the

###### moderate pulmonary

distal stomach. There is

edema and mild cardiomegaly are present. The nasogastric tube shows normal course.

cardiomegaly and bilateral pleural effusions. There is mild pulmonary edema. There is no pneumothorax.

LLM-CXR

UniX

Figure 1. The quantitative and qualitative results of UniX. Quantitative results show UniX’s superiority over existing unified and single-task medical foundation models in understanding and generation. Qualitatively, UniX enables multi-disease X-ray interpretation and high-fidelity medical image generation.

ingly seen as interconnected tasks, where semantic reasoning and visual synthesis can mutually reinforce each other. This insight has spurred the development of unified medical foundation models [15, 17, 22, 46] that aim to integrate

both capabilities within a single framework.

However, unified modeling of these two capabilities is inherently challenging, given their fundamentally different objectives of semantic abstraction versus pixel-level reconstruction. Existing efforts, such as LLM-CXR [17], often employ parameter sharing and joint multi-task heads for integrated learning. This approach, unfortunately, can introduce task competition and feature interference, which degrades performance in both understanding and generation. HealthGPT [22] mitigates this issue through task-specific H-LoRA modules, offering a structured compromise but not a fundamental solution. Moreover, most current unified medical foundation models still rely on discretized generation paradigms, whose outputs are constrained by vocabulary granularity and fail to recover fine structural details in medical images. A straightforward alternative is to attach a diffusion model to a pre-trained vision–language model [8, 11]. While this improves generative quality to some extent, it fails to fully exploit understanding features to guide generation, thereby underutilizing the potential of a unified architecture.

Through systematic analysis, we identify two intrinsic limitations in existing unified medical foundation models. First, understanding and generation possess conflicting objectives. Understanding requires semantic abstraction, whereas generation demands pixel-level reconstruction. Jointly learning these opposing goals in a shared feature space causes interference. Second, a paradigm mismatch exists between discrete autoregression and continuous imaging. Discrete methods inherently struggle to capture the fine-grained structural details of medical images. Consequently, prior works often resort to superficial stacking or cascading to combine these tasks. This strategy achieves unification only in form and fails to exploit deep architectural synergy between the two capabilities.

To address these challenges, we propose UniX. This framework fundamentally resolves the tension between semantic processing and visual synthesis. We adopt a decoupled dual-branch architecture to eliminate the understanding-generation conflict. An autoregressive branch focuses on semantic abstraction, while a separate branch handles pixel-level reconstruction. To bridge the paradigm mismatch, the generation branch leverages diffusion models. This design captures the continuous nature of medical images and avoids the granularity loss inherent in discrete tokenization. Finally, we introduce a cross-modal self-attention mechanism to ensure architectural synergy. Unlike superficial stacking, this module dynamically injects understanding features into the diffusion process. This effectively links semantic reasoning with high-fidelity generation.

To further improve data quality and training efficiency, we implement a rigorous data cleaning pipeline and adopt

a stagewise optimization strategy. In the first stage, we freeze the generation branch and train only the understanding branch to acquire medical image interpretation capabilities. In the second stage, we freeze the understanding branch and pre-train the generation branch to learn basic image generation. In the third stage, we continue to freeze the understanding branch and fine-tune the generation branch for high-resolution image generation. Thanks to our architectural design and optimization strategy, UniX achieves dual-task modeling with significantly fewer parameters, maintaining strong vision-language understanding while attaining high-quality generation performance.

The main contributions of this paper are summarized as follows:

- • We propose UniX, a next-generation unified medical foundation model that structurally decouples yet coordinates understanding and generation. To our knowledge, it is among the first efforts to integrate autoregressive and diffusion paradigms in the medical imaging field.
- • We introduce a cross-modal self-attention mechanism to bridge the understanding and generation branches. This module seamlessly integrates the understanding features as contextual conditions, providing dynamic, contentaware guidance throughout the generation process.
- • Experiments on chest X-ray report generation and image synthesis tasks show that UniX uses only a quarter of the parameters of LLM-CXR, yet improves understanding performance (Micro-F1) by 46.1% and generation performance (FD-Raddino) by 24.2%, achieving performance comparable to single-task medical foundation models.

### 2. Related Work

#### 2.1. Single-task Medical Foundation Model

Medical foundation models for single tasks primarily center on two major objectives. The first category focuses on image understanding. These models handle tasks such as disease diagnosis, knowledge-based question answering, and report generation. Early studies [6, 24, 42] mainly targeted disease classification. They employed CNNs [16] or Transformers [32] to extract imaging features, followed by task-specific heads for classification, segmentation, or prediction.

Recently, with the rise of multimodal large language models [12, 18, 33, 40], medical foundation models [1, 4, 19, 29, 30] have evolved toward more clinically meaningful applications, such as medical question answering and report generation. These models typically combine a visual encoder with a large language model, processing multimodal tokens in an autoregressive way.

The second category centers on image generation. These models address tasks like synthesis, super-resolution, and inpainting. With recent advances in generative AI [7, 23, 27,

Model Architecture

##### Data Processing

[Figure 16]

[Figure 17]

Vision Encoder

AutoRegressive

FINDINGS: No focal consolidation, pleural effusion, pneumothorax, or pulmonary edema identified. The heart size is top normal. Mediastinal and hilar contours are unremarkable. IMPRESSION: No radiographic evidence for acute cardiopulmonary process. Cleaned Reports

Text Prompt

[Figure 18]

Medical Image

Medical Reports

[Figure 19]

Text Tokenizer

Cross-Modal Self-Attention

[Figure 20]

Medical Prompts

[Figure 21]

self-judgment

[Figure 22]

VAE & Noise

Latent Diffusion

Initial Reports

Synthetic Image

Medical Image

Training Pipeline

Stage 1. Medical Understanding SFT Stage 2. Medical Generation PT Stage 3. Medical Generation FT

[Figure 23]

[Figure 24]

[Figure 25]

Medical Report Low Resolution High Resolution

[Figure 26]

[Figure 27]

[Figure 28]

VAE

VAE

VAE

REP. Align

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Cross-Modal Self-Attention

[Figure 33]

[Figure 34]

Cross-Modal Self-Attention

Cross-Modal Self-Attention

Und. Branch Gen. Branch

Und. Branch Gen. Branch

Und. Branch Gen. Branch

RadDino

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

Siglip VAE

Siglip VAE

Text Text Text

Siglip VAE

- Figure 2. Model Architecture. UniX comprises two decoupled yet synergistic branches: an autoregressive understanding branch for semantic encoding, and a diffusion-based generation branch for visual synthesis. To enable effective collaboration between them, we introduce a cross-modal self-attention mechanism that allows semantic features to dynamically guide the generation process. Data Processing and Training Pipeline. To fully exploit the potential of this architecture, we design a rigorous data cleaning pipeline and a three-stage training strategy. This strategy progressively freezes the branches during different stages, ensuring efficient knowledge transfer and stable training.

37, 41, 43, 47], diffusion-based approaches have become the dominant paradigm for high-fidelity medical image generation. Single-task generative models can produce realistic, instruction-aligned images, helping to expand datasets and mitigate long-tail issues. Some studies [2, 9] further demonstrate that synthetic data can enhance the performance of visual understanding models, revealing a potential synergy between generation and understanding.

#### 2.2. Unified Medical Foundation Model

Unified models [5, 11, 20, 34–36] aim to handle both understanding and generation within a single architecture. They represent a promising next step for medical foundation models. Existing unified medical foundation models [15, 17, 46] typically adopt a shared Transformer backbone with multi-task heads. However, this parameter-sharing strategy faces inherent conflicts. Understanding tasks require compressing and abstracting information, while generative tasks demand preserving and reconstructing details. These opposing objectives cause feature interference and limit overall performance.

To address this, HealthGPT [22] introduces H-LoRA modules to separate task-specific parameters. This im-

proves performance but still lags behind specialized singletask models. In addition, most unified medical foundation models rely on discrete generation methods based on visual bag-of-words [10, 31]. Since these approaches compress continuous pixel data into a fixed codebook, they inevitably discard high-frequency details and subtle texture variations. Consequently, they struggle to capture continuous, fine-grained pathological patterns. As a result, image fidelity remains limited.

In contrast, UniX introduces a dual-branch architecture that integrates autoregressive and diffusion paradigms, echoing insights from BAGEL [7] on the value of bottleneck-free multimodal interaction. This design resolves the objective conflict through architectural decoupling. The understanding branch focuses on semantic comprehension, while the diffusion branch specializes in highfidelity image generation. The two branches interact via cross-modal self-attention, allowing understanding features to guide the generation process dynamically. Through this synergy, UniX achieves strong performance comparable to single-task models while maintaining high parameter efficiency.

### 3. Method

In this section, we introduce UniX, a next-generation medical foundation model designed to achieve decoupled yet synergistic learning between Chest X-ray understanding and generation.

As shown in Figure 2, our model contains two core components: an autoregressive understanding branch and a diffusion-based generation branch. The understanding branch, built on a vision-language model, handles semantic abstraction and report reasoning. The generation branch is built directly upon the inherited LLM backbone from the understanding branch, and it specializes in synthesizing high-fidelity Chest X-ray images. A cross-modal selfattention module connects the two, allowing dynamic feature exchange and semantic conditioning during generation.

#### 3.1. Understanding via Autoregression

The understanding branch formulates multimodal comprehension as an autoregressive sequence modeling problem. This formulation aligns naturally with medical report generation, where the model must reason over both visual and textual contexts in a causal manner.

Concretely, we define a multimodal token sequence S = [V, Tin, Tout], where V , Tin, and Tout denote the visual tokens, input textual tokens, and output textual tokens respectively. Let m be the starting index of Tout in S, and n be the ending index of S. The cross-entropy loss is computed over the autoregressive predictions for all tokens in Tout:

n−1

log p(Si+1|S≤i;ωu), (1)

LCE = −

i=m

where n is the index of the last token in the sequence S, and ωu denotes parameters of the understanding branch. This design allows the model to jointly capture visual semantics and linguistic reasoning within a unified space.

#### 3.2. Generation via Latent Diffusion

The generation branch adopts a latent diffusion framework that reconstructs medical images from high-level semantics extracted by the understanding branch. Instead of operating in pixel space, diffusion is performed in a VAE-encoded latent space, which greatly improves efficiency and stability.

Given a latent variable xt sampled from the noisy distribution pt(x) at time t, the model learns to estimate the target velocity field ut(x) by minimizing the mean-square error:

t(x) ∥vt (x;ωg) − ut (x)∥2 , (2)

LMSE = Et,p

where ωg represents the parameters of the generation branch. The semantic embeddings from the understanding

branch act as conditioning inputs, enabling disease-specific synthesis and improved lesion localization.

#### 3.3. Cross-Modal Self-Attention

To enable semantically informed visual generation, we introduce a cross-modal self-attention mechanism [7] that facilitates bidirectional information flow between the understanding and generation branches. Unlike conventional cross-attention, which conditions one modality on a static context, our formulation performs joint self-attention over a unified multimodal token sequence. This design allows semantic representations from the understanding branch to directly modulate the generative trajectory, while also permitting generative states to feed back into the semantic space.

Let the unified sequence be S = [Tin, N], where Tin denotes the textual tokens produced by the understanding branch and N denotes the noise-conditioned latent embeddings from the generation branch. For each token Si, we compute modality-specific projections for queries, keys, and values as:

{Qi,Ki,Vi} = δu(i)W{uq,k,v}Si + δg(i)W{gq,k,v}Si, (3) where the modality selectors δu(i) and δg(i) are defined as:

1, Si ∈ Tin, 0, Si ∈ N,

δg(i) = 1 − δu(i). (4)

δu(i) =

This formulation yields two distinct parameter spaces for understanding and generation tokens, while maintaining a shared attention operation across the unified sequence. The resulting attention map is computed in standard form:

Attn(S) = softmax

QK⊤ √

d

V, (5)

but all cross-modal interactions are learned implicitly through the joint attention scores rather than through explicit conditioning.

This mechanism synchronizes an autoregressive branch for understanding and a diffusion branch for high-fidelity generation, ultimately improving the fidelity and clinical consistency of the generated images.

#### 3.4. Three-Stage Training Pipeline

As shown in Figure 2, we adopt a three-stage training strategy to progressively align the understanding and generation branches.

• Stage 1: Medical Understanding Supervised FineTuning. In this stage, the generation branch is frozen. We fine-tune the visual encoder, visual connector, and language model backbone in the understanding branch using paired medical images and reports. This step helps

###### Hyper Parameters Stage One Stage Two Stage Three

Learning rate 1e-4 2e-4 1e-4 LR scheduler Constant Constant Constant

Resolution 384 256 512 Use REPA – True False

REPA loss weight – 0.5 –

Batch Size 256 256 256 Weight decay 0.0 0.0 0.0

Gradient norm clip 1.0 1.0 1.0 Optimizer AdamW(0.9, 0.95, 1e−15) Warm-up steps 80 2K 0 Training steps 3840 75K 5K

- Table 1. Hyper-parameter settings for three training stages. Note that we have introduced weights for the multiple loss functions to make them more accessible. Specifically, the “REPA loss weight” refers to the weight ratio between the MSE loss and the REPA loss.

the model learn the semantic correspondence between images and text. As a result, the understanding branch gains strong abilities in medical image interpretation and report generation. It also serves as a high-level semantic feature provider for the generation branch in later stages.

- • Stage 2: Medical Generation Pretraining. Here, we freeze the understanding branch and pre-train the generation branch on text–low-resolution image pairs. To accelerate convergence, we apply Representation Alignment [44], aligning the eighth-layer hidden states of the generation branch’s language model with RadDino image features using a similarity objective. This design enables the generation branch to better utilize high-level semantics from the understanding branch for low-resolution medical image synthesis.
- • Stage 3: Medical Generation Fine-Tuning. We maintain the same freezing strategy as in Stage 2 and fine-tune the generation branch using text–high-resolution image pairs. During this stage, we extend the positional encoding of the generation branch and remove feature-level supervision. After fine-tuning, the generation branch can synthesize high-resolution medical images with improved report–image alignment, clearer lesion depiction, and higher visual fidelity.

### 4. Experiments

This section, we describe how UniX exploits its decoupled autoregressive–diffusion dual-branch design through data processing, model configuration, and a three-stage training pipeline.

#### 4.1. Implementation Details

• Data Details. We conduct experiments on the MIMICCXR dataset [14]. For the understanding branch, we

use frontal-view radiographs and refine the paired reports with the DeepSeek large language model. The full cleaning pipeline is provided in the supplementary material. Following the official split, we obtain 163,344 image–report pairs for training and 2,365 for testing. For the generation branch, we follow the processing and split protocol of ChexGenBench [9], resulting in 237,387 training and 4,352 test pairs. Images are resized to 384×384 for understanding fine-tuning, 256×256 for generation pretraining, and 512×512 for generation fine-tuning. Aspect ratios are preserved by padding when needed.

- • Model Details. Both branches are partially initialized from Janus-Pro [5]. For the understanding branch, we adopt siglip-large-patch16-384 [45] as the visual encoder. It produces 1024-dimensional embeddings, which are mapped to the 2048-dimensional LLM space through a two-layer MLP. The language backbone contains 24 transformer layers and incorporates QK normalization and QKV bias. For the generation branch, we use a 16× downsampled, 16-channel VAE for encoding and decoding. Two single-layer MLPs provide bidirectional projection between the 16-dimensional latent space and the 2048-dimensional language space. The generation backbone follows the same initialization strategy as the understanding branch.
- • Training Details. All models are trained with fullparameter fine-tuning on eight NVIDIA L20 GPUs. The complete hyperparameter configuration is listed in Table 1.

#### 4.2. Decoupled Architecture for Task Separation

We conduct extensive comparisons among understandingonly, generation-only, and unified medical foundation models. Qualitative examples of UniX’s performance in both understanding and generation are shown in Figures 3 and 4. As observed, UniX produces precise reports and highfidelity images. We next present detailed results for understanding and generation tasks.

For understanding tasks, we primarily evaluate the medical reliability of generated reports using the CheXbert F1 score [28] between generated and ground-truth reports. Additional evaluation metrics, including BLEU [25], Radgraph [13] and ROUGE-L [21], are provided in the supplementary material. For generation tasks, we measure generation quality with FD-RadDino and KD-RadDino, assess image–text consistency with the Alignment Score, and evaluate accuracy and diversity using the four PRDC metrics.

###### 4.2.1. Understanding

The training dynamics of the understanding branch are detailed in the Supplementary Material. As training progresses, the cross-entropy loss decreases steadily, and the model’s report generation ability improves rapidly at first

CheXbert — “uncertain” as negative↑ CheXbert — “uncertain” as positive↑ Micro F1-14 Micro F1-5 Macro F1-14 Macro F1-5 Micro F1-14 Micro F1-5 Macro F1-14 Macro F1-5 Signal-task Medical Foundation Model

Model Und. Params

GPT-4V – 35.5 25.8 20.4 19.6 35.6 33.3 25.3 29.6 Med-PaLM M 84B 53.6 57.9 39.8 51.6 – – – – LLaVA-Rad 7B 57.3 57.4 39.5 47.7 57.3 60.2 44.0 53.3 LLaVA-Med 7B 27.2 22.0 15.5 16.6 27.3 24.4 18.7 20.5 FlamingoCXR 3B – – – – 51.9 58.0 – – PromptMRG <1B 15.3 6.0 7.8 3.5 15.0 6.9 8.4 4.1 RGRG <1B 38.9 47.2 23.7 40.8 37.4 49.0 24.4 42.7

Unified Medical Foundation Model

LLM-CXR 12B – – – – 36.0 – 21.1 – HealthGPT 3.8B 24.2 25.1 14.6 18.5 25.5 28.2 17.5 22.6 UniX 1.5B 53.6 56.6 33.2 47.3 52.6 57.9 35.5 49.8

- Table 2. Comparison of various medical foundation models on X-ray understanding tasks. The data reveals that UniX achieves a substantial improvement in understanding over the unified medical foundation model. Notably, it delivers performance comparable to a larger, single-task medical foundation model, despite having fewer parameters.

FINDINGS: In comparison with the study of ___, there is little change in the substantial enlargement of the cardiomediastinal silhouette and moderate pulmonary edema with bilateral pleural effusions. Monitoring and support devices remain in place.

[Figure 46]

FINDINGS: Chest PA and lateral radiographs redemonstrate mild interstitial edema and mild cardiomegaly. No signs of aspiration and no change from prior CXR.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

FINDINGS: There is substantial enlargement of the cardiomediastinal silhouette and moderate pulmonary edema with bilateral pleural effusions. Monitoring and support devices are present.

[Figure 52]

FINDINGS: Mild interstitial edema and mild cardiomegaly are present. No signs of aspiration.

FINDINGS: The right-sided pacemaker leads are in stable position. There is mild-to-moderate pulmonary edema. No pleural effusions. No focal parenchymal opacities concerning for pneumonia. Moderate cardiomegaly is present. No pneumothorax.

Generated Reports

FINDINGS: The endotracheal tube tip is 4.5 cm above the carina. The left IJ central line and nasogastric tube are in position. There is cardiomegaly and bilateral pleural effusions. There is moderate pulmonary edema.

Generated Reports

FINDINGS: No previous images. The nasogastric tube is not coiled, however it extends only to the distal esophagus. This information has been conveyed to Dr. ___, who is covering for Dr. ___, by telephone at 8:45 on ___. The heart is normal in size and there is no evidence of pneumonia, vascular congestion, or pleural effusion.

[Figure 53]

[Figure 54]

FINDINGS: No focal consolidation is seen there is no pleural effusion or pneumothorax. The cardiac and mediastinal silhouettes are unremarkable. IMPRESSION: No acute cardiopulmonary process.

[Figure 55]

[Figure 56]

FINDINGS: The nasogastric tube is not coiled but extends only to the distal esophagus. The heart is normal in size and there is no evidence of pneumonia, vascular congestion, or pleural effusion.

[Figure 57]

FINDINGS: No focal consolidation is seen there is no pleural effusion or pneumothorax. The cardiac and mediastinal silhouettes are unremarkable. IMPRESSION: No acute cardiopulmonary process.

[Figure 58]

FINDINGS: The lungs are clear without focal consolidation. No pleural effusion or pneumothorax is seen. The cardiac and mediastinal silhouettes are unremarkable. IMPRESSION: No acute cardiopulmonary process.

FINDINGS: The nasogastric tube tip lies in the distal stomach. The side hole is probably proximal to the esophagogastric junction. No evidence of acute focal pneumonia or vascular congestion.

Generated Reports

Generated Reports

- Figure 3. Demonstration of Data Processing and Report Generation Efficacy. The application of large language models enables the purification of raw data by eliminating extraneous information. This process ensures that the model prioritizes and extracts pertinent information related to disease diagnosis.

before gradually plateauing. Although continued finetuning can further reduce the loss, we observed that key performance metrics begin to decline, indicating that the model tends to overfit specific patterns rather than learning general principles.

Table 2 summarizes the understanding performance of various medical foundation models. UniX achieves better performance than unified models while using substantially fewer parameters. Compared with single-task models, it outperforms models of similar scale and approaches the performance of much larger ones. We exclude medical agent systems here since they typically build upon existing medical foundation models and rely on multi-model collaboration. A comparison between UniX and such agents is

provided in the supplementary material. 4.2.2. Generation

We observed that the mean squared error for the generation branch decreases rapidly early on and then slows in both stages. During medical generation pre-training, key metrics show limited improvement between 25K and 50K steps but increase sharply from 50K to 75K steps. The corresponding loss curves and a detailed analysis are included in the Supplementary Material.

Table 3 reports the generation results. Compared with LLM-CXR, UniX delivers clear improvements in both image quality and image–text alignment. It also consistently outperforms single-task models, reflecting the advantage of its decoupled yet collaborative architecture. Notably, UniX

###### Model Gen. Params Resolution FD-RadDino↓ KD-RadDino↓ Alignment Score↑ Precision↑ Recall↑ Density↑ Coverage↑

Signal-task Medical Foundation Model

Flux.1-Dev ∗ 2.6B 1024 122.400 0.144 0.036 0.420 0.008 0.125 0.326 Lumina 2.0 ∗ 2.5B 1024 101.198 0.110 0.121 0.574 0.014 0.256 0.170 SD V3.5 Medium ∗ 2.5B 1024 91.302 0.103 0.044 0.632 0.205 0.401 0.244 SD V2-1 ∗ 0.86B 512 186.530 0.413 0.197 0.530 0.049 0.180 0.038 RadEdit 0.86B 512 69.695 0.033 0.677 0.397 0.544 0.150 0.285 Sana ∗ 0.6B 512 54.225 0.016 0.695 0.674 0.614 0.520 0.548 Pixart Sigma ∗ 0.6B 512 60.154 0.023 0.697 0.666 0.522 0.506 0.506

Unified Medical Foundation Model

LLM-CXR 12B 256 71.243 0.061 0.319 0.782 0.041 0.671 0.459 UniX 1.5B 256 65.208 0.051 0.251 0.675 0.243 0.366 0.419 UniX 1.5B 512 54.022 0.024 0.635 0.736 0.479 0.536 0.550

- Table 3. Comparison of various medical foundation models on X-ray generation tasks. Under a standardized benchmark, UniX matches the output quality of single-task medical models. Furthermore, it demonstrates exceptional performance in both accuracy and diversity. HealthGPT was not included in this test due to the lack of publicly available text-to-image generation code. ∗ indicates that all these generative models from the natural image domain were fine-tuned on the same X-ray dataset.

###### (A) Normal (B) Different Severity (C) Different Area

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

No acute cardiopulmonary.

Severely enlarged heart size

Mildly enlarged heart size

Left pleural effusion.

Right pleural effusion.

Bilateral pleural effusion.

###### (D) Multi-findings Generation

[Figure 65]

[Figure 66]

[Figure 67]

Cardiomegaly is extensive. Bibasal areas of atelectasis are noted. No appreciable pleural effusion or pneumothorax seen.

Cardiomegaly is accompanied by pulmonary vascular congestion and interstitial edema. Small left pleural effusion and right pleural effusion are present.

Moderate cardiomegaly is noted. Increased interstitial markings are seen throughout the lungs. There is no confluent consolidation or effusion. There is no acute osseous abnormality.

[Figure 68]

[Figure 69]

[Figure 70]

There may have been a modest decrease in small residual right pleural effusion. Right basal atelectasis is still substantial. No pneumothorax. Left lung clear. Heart size top normal, unchanged. Left PIC line ends low in the SVC.

Cardiomegaly is substantial. Large right pleural effusion is demonstrated. Left lung is clear. There is no pneumothorax.

Feeding tube ends low in the stomach. Moderate left lower lobe atelectasis. Small left pleural effusion has decreased. No pneumothorax. Right lung clear. Normal postoperative appearance to cardiomediastinal silhouette.

- Figure 4. Qualitative Examples from UniX. (A)-(C) illustrate the model’s precise control over the attributes of generated findings, including their severity and location. In (D), the model successfully synthesizes a complex radiographic scene containing multiple findings that are consistent with a full clinical report, highlighting its ability to process and integrate extensive contextual information.

performs on par with the strong baseline Sana. Even though Sana is also fine-tuned on the target dataset, UniX matches its performance and achieves comparable, and in some metrics slightly superior. This demonstrates that our unified approach maintains top-tier generation quality without compromising semantic consistency.

We further evaluate pathology-specific generation qual-

ity in Table 4. UniX achieves consistently higher fidelity across a wide range of lesion types, capturing subtle pathological cues and preserving clinically relevant details. In these fine-grained tasks, UniX remains highly competitive with Sana. This is particularly noteworthy given that UniX balances a unified multi-task objective whereas Sana focuses on a specialized generation task. Consequently, the

FD-RadDino↓ At Cd Cn Ec Fc Fr LL LO NF PE PO PN PT SD Signal-task Medical Foundation Model

Model Gen. Params Resolution

RadEdit 0.86B 512 63.38 62.79 136.59 76.94 155.97 197.58 184.11 61.90 67.88 60.60 215.92 114.66 151.34 53.10 Pixart Sigma ∗ 0.6B 512 59.27 60.39 133.96 73.93 155.53 179.44 174.63 56.83 48.74 59.05 210.90 108.42 150.55 51.61 Sana ∗ 0.6B 512 51.03 54.68 127.46 67.84 147.00 172.32 163.14 49.23 44.60 49.80 199.45 88.52 141.99 46.51

Unified Medical Foundation Model

LLM-CXR 12B 256 71.57 71.37 136.65 83.18 148.28 168.50 163.22 66.93 64.62 67.83 200.84 108.04 147.52 67.54 UniX 1.5B 256 63.34 63.39 129.32 73.88 150.25 177.68 165.88 58.31 60.58 58.55 201.53 105.96 141.63 57.61 UniX 1.5B 512 52.19 51.70 122.84 64.36 142.23 176.35 156.81 49.15 45.71 48.06 191.65 99.31 135.48 47.04

- Table 4. Generation Performance per Pathology. Within the unified medical foundation model, UniX dominates the comparison, achieving top performance in 13 out of the 14 categories. ∗ indicates that all these generative models from the natural image domain were fine-tuned on the same X-ray dataset.

Configs Metrics

Train Branch Data Ratio Micro-F1↑ Macro-F1↑ FD↓ KD↓ Gen 0 : 1 53.2 36.0 62.114 0.041

- Und & Gen 1 : 1 43.7 29.5 65.747 0.047

- Und & Gen 1 : 2 42.9 28.3 76.125 0.064 Und & Gen 1 : 4 44.9 28.4 76.108 0.064 Und & Gen 0 : 1 13.9 6.3 74.815 0.054

- Table 5. Impact of Joint Dual-Branch Optimization. Freezing the understanding branch yields fast generative gains without harming comprehension. Unfreezing it without understanding data severely degrades comprehension and offers no generative benefit. Mixing both data types mitigates this degradation but slows generative learning.

#### 5.2. Impact of Joint Dual-Branch Optimization

To study how joint fine-tuning affects understanding and generation, we design multiple medical image generation experiments, each fine-tuned for 2K generation steps. We evaluate three strategies:

- 1) Unfreeze only the generation branch.
- 2) Unfreeze both branches and train with a mixture of understanding and generation data, where we vary the mixing ratio.
- 3) Unfreeze both branches but train without any understanding data.

The results in Table 5 lead to several observations. First, the fine-tuned understanding branch should not be involved in subsequent training of the generation branch. Fully freezing the understanding branch yields rapid gains in generation performance without harming understanding accuracy. In contrast, unfreezing the branch without providing understanding data is detrimental; it severely degrades understanding performance and offers no benefit to generation. Adjusting the ratio of understanding and generation data can partially mitigate the drop in understanding performance, as the semantic supervision stabilizes the updated parameters. However, this setup forces the model to balance two competing objectives within the same updates, which slows the acquisition of strong generative capability.

results demonstrate strong fine-grained visual synthesis capability and robust performance under diverse diagnostic conditions.

### 5. Ablations

#### 5.1. Impact of Data Cleaning on Understanding

Figure 3 illustrates the critical role of data cleaning with DeepSeek in mitigating hallucinations. Raw hospital reports often contain significant noise, such as underscores, technical metadata, and conversational fillers, which complicates the alignment between visual features and textual descriptions. By employing targeted prompts to strip away these non-diagnostic elements, we construct a cleaner and more semantic-dense target for the model. This preprocessing step is crucial because it forces the model to attend strictly to clinically relevant patterns during training, resulting in generated reports that are factually grounded and free from structural hallucinations.

### 6. Conclusion

We present UniX, a next-generation unified medical foundation model that achieves architectural decoupling and coordination for Chest X-Ray understanding and generation. Existing unified medical foundation models neither resolve the intrinsic conflict nor fully exploit the strengths of different modeling paradigms. To address these limitations, we design a dual-branch architecture that combines autoregressive understanding with diffusion-based generation. This structure decouples the two tasks and prevents mutual interference. We introduce a cross-modal

self-attention mechanism that aligns both branches and enables understanding features to guide generation dynamically. With these designs, UniX delivers stronger understanding and generation performance than prior unified medical foundation models while using fewer parameters, and it reaches competitiveness with dedicated single-task medical foundation models. We hope that UniX offers a new perspective for advancing medical foundation models.

### References

- [1] Yaowei Bai, Ruiheng Zhang, Yu Lei, Jingfeng Yao, Shuguang Ju, Chaoyang Wang, Wei Yao, Yiwan Guo, Guilin Zhang, Chao Wan, et al. From bench to bedside: A deepseekpowered ai system for automated chest radiograph interpretation in clinical practice. arXiv preprint arXiv:2507.19493,

2025. 1, 2

- [2] Christian Bluethgen, Pierre Chambon, Jean-Benoit Delbrouck, Rogier Van Der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P Langlotz, and Akshay S Chaudhari. A vision–language foundation model for the generation of realistic chest x-ray images. Nature Biomedical Engineering, 9(4):494–506, 2025. 1, 3
- [3] Pierre Chambon, Christian Bluethgen, Jean-Benoit Delbrouck, Rogier Van der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P Langlotz, and Akshay Chaudhari. Roentgen: vision-language foundation model for chest x-ray generation. arXiv preprint arXiv:2211.12737, 2022. 1
- [4] Juan Manuel Zambrano Chaves, Shih-Cheng Huang, Yanbo Xu, Hanwen Xu, Naoto Usuyama, Sheng Zhang, Fei Wang, Yujia Xie, Mahmoud Khademi, Ziyi Yang, et al. Towards a clinically accessible radiology foundation model: openaccess and lightweight, with automated evaluation. arXiv preprint arXiv:2403.08002, 2024. 1, 2
- [5] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 3, 5

- [6] Ziwei Cui, Jingfeng Yao, Lunbin Zeng, Juan Yang, Wenyu Liu, and Xinggang Wang. Lkcell: Efficient cell nuclei instance segmentation with large convolution kernels. arXiv preprint arXiv:2407.18054, 2024. 1, 2
- [7] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 3, 4
- [8] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499,

2023. 2

- [9] Raman Dutt, Pedro Sanchez, Yongchen Yao, Steven McDonagh, Sotirios A Tsaftaris, and Timothy Hospedales. Chexgenbench: A unified benchmark for fidelity, privacy

- and utility of synthetic chest radiographs. arXiv preprint arXiv:2505.10496, 2025. 1, 3, 5
- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3
- [11] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024. 2, 3
- [12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2
- [13] Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P Lungren, Andrew Y Ng, et al. Radgraph: Extracting clinical entities and relations from radiology reports. arXiv preprint arXiv:2106.14463, 2021. 5
- [14] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a deidentified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019. 5
- [15] Tackeun Kim, Jihang Kim, Leonard Sunwoo, and Edward Choi. Unixgen: a unified vision-language model for multiview chest x-ray generation and report generation. arXiv preprint arXiv:2302.12172, 2023. 1, 3
- [16] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 2002. 2
- [17] Suhyeon Lee, Won Jun Kim, Jinho Chang, and Jong Chul Ye. Llm-cxr: instruction-finetuned llm for cxr image understanding and generation. arXiv preprint arXiv:2305.11490,

2023. 1, 2, 3

- [18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2
- [19] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36:28541–28564,

2023. 1, 2

- [20] Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, et al. Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29767–29779, 2025. 3
- [21] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 5

- [22] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li, Wanggui He, Hao Jiang, Mengze Li, Xiaohui Song, et al. Healthgpt: A medical large visionlanguage model for unifying comprehension and generation via heterogeneous knowledge adaptation. arXiv preprint arXiv:2502.09838, 2025. 1, 2, 3
- [23] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [24] DongAo Ma, Jiaxuan Pang, Michael B Gotway, and Jianming Liang. A fully open ai foundation model applied to chest radiography. Nature, pages 1–11, 2025. 1, 2
- [25] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318,

2002. 5

- [26] Fernando P´erez-Garc´ıa, Sam Bond-Taylor, Pedro P Sanchez, Boris van Breugel, Daniel C Castro, Harshita Sharma, Valentina Salvatelli, Maria TA Wetscherek, Hannah Richardson, Matthew P Lungren, et al. Radedit: stress-testing biomedical vision models via diffusion image editing. In European Conference on Computer Vision, pages 358–376. Springer, 2024. 1
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [28] Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y Ng, and Matthew P Lungren. Chexbert: combining automatic labelers and expert annotations for accurate radiology report labeling using bert. arXiv preprint arXiv:2004.09167, 2020. 5
- [29] Ryutaro Tanno, David GT Barrett, Andrew Sellergren, Sumedh Ghaisas, Sumanth Dathathri, Abigail See, Johannes Welbl, Charles Lau, Tao Tu, Shekoofeh Azizi, et al. Collaboration between clinicians and vision–language models in radiology report generation. Nature Medicine, 31(2):599–608,

2025. 1, 2

- [30] Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. Nejm Ai, 1(3):AIoa2300138, 2024. 1, 2
- [31] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [33] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2

- [34] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 3
- [35] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.
- [36] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [37] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, et al. Pixel-perfect depth with semantics-prompted diffusion transformers. arXiv preprint arXiv:2510.07316, 2025. 3
- [38] Ziyang Xu, Huangxuan Zhao, Ziwei Cui, Wenyu Liu, Chuansheng Zheng, and Xinggang Wang. Most-dsa: Modeling motion and structural interactions for direct multi-frame interpolation in dsa images. arXiv preprint arXiv:2407.07078, 2024. 1
- [39] Ziyang Xu, Huangxuan Zhao, Wenyu Liu, and Xinggang Wang. Garamost: Parallel multi-granularity motion and structural modeling for efficient multi-frame interpolation in dsa images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 28530–28538, 2025. 1
- [40] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 2023. 2
- [41] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37:56166–56189, 2024. 3
- [42] Jingfeng Yao, Xinggang Wang, Yuehao Song, Huangxuan Zhao, Jun Ma, Yajie Chen, Wenyu Liu, and Bo Wang. Evax: A foundation model for general chest x-ray analysis with self-supervised learning. arXiv preprint arXiv:2405.05237,

2024. 1, 2

- [43] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 3
- [44] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 5
- [45] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. 5

- [46] Ziyang Zhang, Yang Yu, Yucheng Chen, Xulei Yang, and Si Yong Yeo. Medunifier: Unifying vision-and-language

pre-training on medical data with vision generation task using discrete visual representations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29744–29755, 2025. 1, 3

[47] Ya Zou, Jingfeng Yao, Siyuan Yu, Shuai Zhang, Wenyu Liu, and Xinggang Wang. Turbo-vaed: Fast and stable transfer of video-vaes to mobile devices. arXiv preprint arXiv:2508.09136, 2025. 3

