# arXiv:2601.19798v1[cs.CV]27Jan2026

## Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision

###### Youtu-VL Team∗

Despite the significant advancements represented by Vision-Language Models (VLMs), current architectures often exhibit limitations in retaining fine-grained visual information, leading to coarsegrained multimodal comprehension. We attribute this deficiency to a suboptimal training paradigm inherent in prevailing VLMs, which exhibits a text-dominant optimization bias by conceptualizing visual signals merely as passive conditional inputs rather than supervisory targets. To mitigate this, we introduce Youtu-VL, a framework leveraging the Vision-Language Unified Autoregressive Supervision (VLUAS) paradigm, which fundamentally shifts the optimization objective from “visionas-input” to “vision-as-target.” By integrating visual tokens directly into the prediction stream, Youtu-VL applies unified autoregressive supervision to both visual details and linguistic content. Furthermore, we extend this paradigm to encompass vision-centric tasks, enabling a standard VLM to perform vision-centric tasks without task-specific additions. Extensive empirical evaluations demonstrate that Youtu-VL achieves competitive performance on both general multimodal tasks and vision-centric tasks, establishing a robust foundation for the development of comprehensive generalist visual agents.

Code: https://github.com/TencentCloudADP/youtu-vl Model: https://huggingface.co/collections/tencent/youtu

[Figure 2]

[Figure 3]

|Semantic Segmentation|
|---|

|Referring Segmentation|
|---|

Visual Grounding

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Object Detection

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

GUI Agent

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

MultiImage

[Figure 44]

[Figure 45]

[Figure 46]

Semantic Segmentation

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

|Pose Estimation|
|---|

|Object Detection|
|---|

|Visual Grounding|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Referring Segmentation

OCR

Depth Estimation

General VQA

|General VQA|
|---|

|GUI Agent|
|---|

|Depth Estimation|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

Multimodal Reasoning Object Counting

Pose Estimation

Image Classification

[Figure 62]

[Figure 63]

Figure 1. Youtu-VL achieves competitive performance on both general multimodal tasks and vision-centric tasks. The concentric rings illustrate the capability scope of different models across various tasks. Colored regions indicate that the model supports the corresponding task, while white regions denote a lack of support. Unlike prior models that exhibit functional gaps, Youtu-VL accommodates a comprehensive range of vision-centric and multimodal tasks via a standard architecture, achieving competitive performance without relying on task-specific modules.

*Full author list in contributions.

### 1 Introduction

Previous Paradigm: Vision as Input (Text-DominantAutoregressiveSupervision)

Youtu-VL Paradigm: Vision as Target (Vision-LanguageUnifiedAutoregressiveSupervision)

Image Text

Image

Text

Previous VLM

Youtu-VL

Text Supervision Only

Image-Text Unified Supervision

- Figure 2. Comparison between the previous "vision as input" paradigm and the Youtu-VL "vision as target" paradigm. The left panel shows the previous text-dominant VLM, which relies solely on text supervision. The right panel illustrates the Youtu-VL paradigm, which incorporates Vision-Language Unified Autoregressive Supervision (VLUAS), treating vision as a target to achieve unified supervision for both image and text.

Vision-Language Models (VLMs) have achieved significant proficiency in multimodal tasks by integrating the understanding capabilities of Large Language Models (LLMs) with pre-trained visual encoders. By aligning visual features to linguistic semantics, these architectures have become the standard for generalist applications ranging from image captioning to visual reasoning [Bai et al., 2025, Wang et al., 2025].

However, a fundamental limitation persists in current VLM research: the retention of fine-grained visual information. We attribute this deficiency to a text-dominant optimization bias inherent in prevailing training paradigms. In typical architectures, visual signals are conceptualized merely as passive conditional inputs, while the optimization process is driven solely by autoregressive text generation objectives [Liu et al., 2023, Assran et al., 2023]. Consequently, the model is implicitly encouraged to discard visual details deemed redundant for coarse-grained text generation, inevitably creating an information bottleneck that hinders dense perception capabilities.

To address the aforementioned optimization bottleneck, we introduce Youtu-VL, a framework built upon the Vision-Language Unified Autoregressive Supervision (VLUAS) paradigm. As illustrated in Figure 2, diverging from the prevailing “vision-as-input” strategies, we expand the textual lexicon into a unified multimodal vocabulary Vunified via a learned visual codebook. Central to this design is our Synergistic Vision Tokenizer, which fuses high-level semantic concepts with low-level geometric structures to produce discrete codes, thereby providing dense semantic visual supervision. This enables the model to treat visual signals as supervisory targets rather than passive conditions. By integrating these visual tokens into the prediction stream, Youtu-VL enforces the unified autoregressive supervision of visual details and linguistic content, ensuring the preservation of fine-grained information typically discarded by standard text-generation objectives.

Furthermore, we extend this unified paradigm to encompass a comprehensive suite of vision-centric tasks employing a standard architecture without task-specific modules. We categorize these capabilities into two distinct streams: text-based prediction and dense prediction. For text-based prediction tasks such as object detection and visual grounding, we implement an axis-specific vocabulary with absolute pixel coordinates, enabling the model to generate precise bounding boxes directly as textual tokens without normalization artifacts. Conversely, targeting pixel-level tasks like semantic segmentation and depth estimation, we utilize the model’s native logit representations. Through multi-task autoregressive vision supervision, our approach enables high-quality dense prediction directly from these raw logits without extra task-specific heads or embeddings. This design unifies the inference pipeline, allowing a standard VLM to seamlessly transition between high-level reasoning and low-level dense perception without structural additions.

Our contributions are summarized as follows:

|[Figure 66]| |
|---|---|
| | |

###### NTP-M for Multi-label and Multi-Task

[Figure 67]

Visual Codebook

|[Figure 68]|
|---|

[Figure 69]

seg. tokens

ℒ = ℒℓ𝓅𝒾𝓅𝓈 + ℒℊ𝒶𝓃 + ℒ𝓋𝓆 + ℒℯ𝓃𝓉

Mask Decoders binary masks (e.g. SAM)

dog, chair, BG, depth10, depth11, depth12, depth20, depth22, depth50, depth60, depth70

###### Hello world! We are Youtu-VL

text tokens

[Figure 70]

###### Text

Frozen SigLip2

Frozen DINOv3

[Figure 71]

[Figure 72]

Vision Encoder

vision tokens

H W

NTP-M (ours)

Retrieval binary masks (e.g. UFO)

H H

Text Tokenizer

[Figure 73]

[Figure 74]

###### Projector

|W|
|---|

|W|
|---|

seg. tokens

|W|
|---|

chair NTP (standard)

[Figure 75]

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[EOT]|
|---|

|[BOS]|
|---|

|[BOI]|
|---|

|···|
|---|

|[EOI]|
|---|

|[BOT]|
|---|

|Hello|
|---|

|world|
|---|

|!|
|---|

|We|
|---|

|are|
|---|

|VL|
|---|

|Youtu|
|---|

|-|
|---|

|Softmax(<br><br>QK<br><br>d<br><br>)V|
|---|

text tokens

losses for the huge vocabulary set v

Binary polygons (e.g. VisionLLM)

[Figure 81]

[Figure 82]

C

Other Solutions

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

pi : positives

(irrelevant vocabulary >> positives)

[Figure 83]

|[Figure 84]|
|---|

|[Figure 85]|
|---|

[Figure 86]

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[EOT]|
|---|

|[BOI]|
|---|

|···|
|---|

|[EOI]|
|---|

|[BOT]|
|---|

|Hello|
|---|

|world|
|---|

|!|
|---|

|We|
|---|

|are|
|---|

|Youtu|
|---|

|-|
|---|

|VL|
|---|

|[EOS]|
|---|

MLP

[Figure 89]

Learnable Codebook

N irelev: relevant negatives (top k)

vision tokens

avg

|[EOT]|
|---|

|[BOT]|
|---|

|Penguin|
|---|

|-|
|---|

|at|
|---|

|-|
|---|

|<x_233>|
|---|

|<y_123>|
|---|

|<x_321>|
|---|

|<y_300>|
|---|

|[EOS]|
|---|

|tree snow … d20 d60|
|---|

|sky tree snow … d10 d100|
|---|

|penguin snow QQ … d5 d50|
|---|

|outuV sky … d10 d1000|
|---|

|L tree snow tree … d10|
|---|

|···|
|---|

No Additions (Ours)

…

LNTP-M

text tokens

[Figure 90]

avg

losses for all patches

Decoder

Training Inference

- Figure 3. Overview of the Youtu-VL Framework. Left: The architecture integrates a Vision Encoder and Youtu-LLM via a Spatial Merge Projector, operating under the proposed VLUAS paradigm for unified autoregressive modeling. Middle: The Synergistic Vision Tokenizer. We construct a unified vocabulary by fusing semantic and geometric features via cross-attention, optimized with perception and adversarial losses. Right: Dense prediction mechanism. Our proposed NTP-M enables robust multi-label supervision with a relevant negative sampling. Unlike conventional approaches, Youtu-VL achieves direct dense prediction without auxiliary decoders or task-specific tokens.

- • Vision-Language Unified Autoregressive Supervision (VLUAS). We propose a paradigm shift from text-only supervision to generative unification. By treating visual tokens as optimization targets, we mitigate text-dominant bias and enforce the retention of fine-grained visual details.
- • Vision-Centric Predictions from the Standard Architecture. We treat image and text tokens with equivalent autoregressive status, empowering Youtu-VL to perform vision-centric tasks for both dense vision prediction (e.g., segmentation, depth) and text-based prediction (e.g., grounding, detection) within a standard VLM architecture, eliminating the need for task-specific additions.
- • Empirical Performance. Extensive evaluations demonstrate that Youtu-VL achieves competitive performance on both general multimodal tasks and vision-centric tasks, establishing a robust baseline for generalist visual agents.

### 2 Architecture and Methodology

We introduce Youtu-VL, a novel framework built on our proposed Vision-Language Unified Autoregressive Supervision (VLUAS) and vision-centric predictions from the standard architecture. As illustrated in the left panel of Figure 3, Youtu-VL comprises three core components: a vision encoder, a vision-language projector designed to map visual features into a unified vision-language token space, and a LLM.

Large Language Model (LLM). The LLM employed in Youtu-VL is a self-developed model built upon the architecture of Youtu-LLM [Lu et al., 2026]. Building upon this foundation, we introduce the VLUAS paradigm, which extends the original text vocabulary with a dedicated image-token vocabulary. To achieve this, we employ a multi-stage training pipeline. In Stage 3, our Vision Tokenizer is trained to construct the visual codebook, forming the visual portion of Vunified (Figure 3, Middle). In Stage 4, we apply the proposed multi-label NTP objective directly to the LLM’s autoregressive outputs over vision tokens, enabling dense prediction supervision without introducing auxiliary decoders or task-specific tokens (Figure 3, Right).

Vision Encoder. Our vision encoder is built upon SigLIP-2 [Tschannen et al., 2025]. Specifically, we adopt the siglip2-so400m-patch16-naflex variant. This architecture incorporates 2D Rotary Position

Embedding (RoPE) [Su et al., 2023] according to spatial shapes for positional encoding. It further employs window attention for efficiency, with global attention inserted every 8 layers. In addition, it leverages FlashAttention [Dao et al., 2022] through the cumulative sequence length mechanism to handle variablelength sequences within a batch. Importantly, our design allows the encoder to process images at native resolution, which provides flexibility for high-resolution inputs without requiring resizing to a fixed scale.

Vision-Language Projector. The Vision-Language Projector employs a Spatial Merge operation to concatenate adjacent 2 × 2 patch features [Bai et al., 2023]. This effectively reduces the token count to one-quarter (1/4) of its original size, thereby shortening the input sequence length for the subsequent LLM. Finally, a two-layer Multi-Layer Perceptron (MLP) projects these compressed features into the LLM’s input space.

##### 2.1 Vision-Language Unified Autoregressive Supervision Paradigm

The VLUAS paradigm bridges the modality gap by extending the conventional textual lexicon into a Unified Image-Text Vocabulary (Vunified). Under this formulation, visual signals are represented as tokens and trained with the same next-token prediction objective as language, which enables a unified autoregressive modeling interface and supports direct token-level supervision for dense prediction. This section details the construction of this vocabulary via a specialized tokenizer, followed by the formulation of our unified autoregressive objective.

Construction of Unified Image-Text Vocabulary. To realize the VLUAS paradigm, we expand the conventional textual lexicon into a unified image-text vocabulary (Vunified) by introducing a vision tokenizer that maps an image to a sequence of discrete indices through vector quantization. We use these discrete indices as prediction targets, which casts visual learning into the same next-token prediction form as language modeling and enables direct token-level dense supervision via cross-entropy. In parallel, the conditioning pathway remains continuous: the input image is encoded into continuous embeddings by a vision encoder, projected to the LLM hidden dimension, and concatenated with text embeddings as context. This preserves fine-grained visual cues in the context and avoids propagating quantization error into the conditioning signal. A purely continuous alternative would require regressing high-dimensional vectors, departing from the token-based autoregressive interface and making supervision less aligned with standard LLM training. This asymmetric design therefore combines information-preserving continuous conditioning with stable, vocabulary-based autoregressive supervision.

For the vision tokenizer representation, it is important to preserve both high-level semantics and fine spatial structure. Semantic encoders tend to be spatially coarse, whereas pixel-level objectives can retain excessive high-frequency redundancy and encourage texture shortcuts. We address this by designing a synergistic vision tokenizer that forms dense vision-language aligned features before discretization. Specifically, we leverage two frozen foundational encoders with complementary properties: SigLIP-2 [Tschannen et al., 2025] provides rich language-aligned semantics, while DINOv3 [Siméoni et al., 2025] offers boundaryconsistent local correspondences via self-distillation, which helps maintain spatial structure. Although DINOv3 contains dense semantics as well, it is not explicitly aligned to language; combining it with SigLIP-2 allows us to bind language-aligned semantic content to geometry-aware spatial layout.

Concretely, we employ a cross-attention fusion mechanism that probes semantic features under structural constraints. We project the feature maps into a shared manifold to define Query (Q), Key (K), and Value (V) matrices:

Q = HgeoWQ, K = HsemWK, V = HsemWV, (1) where Hgeo and Hsem denote the hidden states from the structural (DINOv3) and semantic (SigLIP-2)

encoders, respectively. The fused synergistic representation Zsyn is computed via cross-attention:

Zsyn = Softmax

QK⊤ √dk

V. (2)

Prior to quantization, we concatenate Zsyn with the original structural features Hgeo along the channel dimension, project the composite representation via an MLP, and discretize it using Index Backpropagation Quantization (IBQ) [Shi et al., 2025]. This process maps continuous vectors to the nearest prototype in a learnable codebook C = {ck}kK=1, configured with a vocabulary size of K = 150,000 and embedding dimension D = 768.

The tokenizer is optimized end-to-end to reconstruct input images from these discrete codes. We employ a compound objective function Ltok that balances perceptual fidelity with codebook usage:

(3)

Ltok = λpLlpips + λgLgan

###### + Lvq + λeLent

Codebook Optimization

Perceptual & Adversarial Fidelity

where Llpips enforces textural realism via perceptual similarity, and Lgan denotes the adversarial discriminator loss. To preclude codebook collapse, we integrate a vector quantization loss Lvq alongside an entropy regularization term Lent. We set the loss weights to λp = 1, λg = 1, and λe = 0.1. A pivotal design choice in our strategy is the deliberate exclusion of standard pixel-wise ℓ1 reconstruction loss. We posit that ℓ1 minimization creates a “texture bias”, encouraging the model to memorize high-frequency noise as a shortcut, thereby bypassing high-level semantic abstraction. By relying exclusively on perceptual and adversarial constraints, we compel the codebook to encode structural semantics rather than mere pixel statistics. Empirically, this approach yields a codebook utilization rate of 97.74% on our composite dataset. The learned indices form the visual vocabulary Vimg, which, when merged with the textual vocabulary, establishes the unified foundation Vunified = Vtext ∪ Vimg for our VLUAS paradigm.

Vision-Language Unified Autoregressive Supervision (VLUAS) Paradigm. The core contribution of Youtu-VL is the transition from a discriminative “vision-as-input” paradigm to a generative “vision-as-target” paradigm. We reformulate the training objective to model the joint probability of vision and language sequences autoregressively. To maximize performance, we employ an asymmetric representation strategy for inputs and targets. For the input context, we utilize continuous visual embeddings extracted by the SigLIP-2 encoder. These features are linearly projected to the LLM dimension and concatenated with textual embeddings, retaining maximum signal fidelity by eliminating quantization errors at the input stage.

Formally, given a multimodal sequence S containing textual tokens T = {t1, . . . , tN} and visual tokens V = {v1, . . . , vM}, the model optimizes the probability of the next token si given the context s<i, regardless of modality. The unified objective function LVLUAS is a weighted summation of textual and visual prediction losses:

LVLUAS = Ltext + λLimage (4)

where λ is a hyperparameter balancing task gradients (empirically set to 0.5). The individual loss components are calculated using standard cross-entropy:

Ltext = − ∑

logP(ti | s<i;θ), Limage = − ∑

log P(vj | s<j; θ) (5)

i∈It

j∈Iv

Here, It and Iv denote the indices of text and image tokens in the flattened sequence, respectively. By enforcing Limage, Youtu-VL is compelled to simultaneously reconstruct visual details and linguistic content. This resolves the text-dominant optimization bias, preserves fine-grained visual information typically lost in discriminative approaches.

##### 2.2 Vision-Centric Predictions from the standard VLM

Text Prediction for Vision-Centric Tasks. We categorize vision-centric tasks into two paradigms: text prediction and dense prediction. The text prediction paradigm encompasses tasks including image classification, object counting, visual grounding, object detection, pose estimation, and polygon-based segmentation. Central to the text prediction paradigm is the precise localization of objects via coordinate prediction. We realize it with three designs. (1) Axis-specific vocabulary: we expanded the tokenizer’s vocabulary by introducing 2048 coordinates for both the X-axis and the Y-axis (e.g., <x_0>). This design significantly reduces the sequence length required for coordinate representation (wo. x & y indicators, commas, and long numbers). It also effectively mitigates ambiguity between X and Y values, particularly in multi-coordinate scenarios. (2) Absolute pixel coordinates: the model operates directly on absolute pixel coordinates rather than normalized relative coordinates. This design eliminates the need for coordinate renormalization during training and inference, thereby preventing potential interference caused by the model attempting to learn implicit scaling mappings. (3) Parsing tokens: for precise results parsing, some parsing tokens have been integrated into the vocabulary (refer to Appendix A.1).

Building upon this vocabulary design, we formulate specific vision tasks as follows. (1) Visual grounding is presented as the prediction of four coordinate tokens XYXY defining the bounding box. (2) Object detection extends the grounding formulation by appending a text category token before the coordinate sequence. The framework supports flexible query modes, ranging from single and multiple target classes, also supporting "detect all" commands for open-world scenarios. (3) Human pose estimation is formulated as the sequential prediction of multiple keypoint coordinates. (4) Polygon-based segmentation can be implemented by predicting a sequence of points to outline object boundaries (compressed within 20 points to avoid very long cases). (5) Object counting: we supported direct regression of a numerical token, or a “detect-then-count” approach (see Figure 15) for verifiable accuracy.

Dense Prediction from the standard VLM. Our goal is to empower standard VLMs to perform mainstream vision-centric tasks, thereby reconciling general-purpose VLMs with these critical yet often overlooked capabilities. It is evident that text generation alone cannot encompass all modalities, particularly pixel or patch-level dense prediction tasks such as semantic segmentation and depth estimation. Conventional approaches typically resort to incorporating additional task-specific decoders [Rasheed et al., 2024, Wu

- et al., 2024, Liu et al., 2025] and corresponding task embedding tokens [Tang et al., 2025]. However, such add-ons inevitably fragment the unified architecture and complicate both training and inference pipelines, thereby undermining practicality. Inspired by the token activation map [Li et al., 2025], we contend that a standard VLM is inherently a dense predictor capable of yielding dense predictions directly from the output vision tokens without auxiliary modules. Our inference scheme relies on two core mechanisms: (1) Predict categories directly within the unified vocabulary. (2) Take the vision token logits corresponding to the predicted category token indices, with the argmax operation to obtain the results.

For semantic segmentation, categories are standard text tokens. For depth estimation, we discretize the depth range into bins (e.g., 1–1000), supporting both linear quantization for multiple closed sets and logarithmic quantization for open-world tasks (e.g., depth anything scenario). Note that we do not predict each bin name; we apply argmax across all bin vocabularies. In comparison, predicting classes in semantic segmentation is necessary since the general text vocabulary is too large to argmax. To handle categories represented by multiple tokens (e.g., sub-word units), we aggregate their alignment scores by computing the mean of the raw logits Zv associated with each category k (Sk denotes the set of token indices corresponding to the k-th semantic category). These aggregated logits are reshaped (R) into a spatial grid and upsampled via bilinear interpolation (I) to recover pixel-level granularity. Finally, the dense prediction map M is derived by taking the argmax (A) over the predicted categories:

1

M = A I R Z ¯ , where Zˆ k =

|Sk| ∑

Zv,t . (6)

t∈Sk

To obtain high-quality pixel-level outputs, a Dense CRF [Krähenbühl and Koltun, 2011] can be optionally employed as a post-processing step following interpolation. Additionally, a softmax operation with temperature τ can be applied to the aggregated logits Zˆ k to regulate the distribution sharpness, thereby controlling the relative contribution of logits to the color values. This operation is optional for semantic segmentation, while depth estimation is not suggested. Another effective strategy to boost performance is image zooming, which leverages the model’s native resolution capability to process finer details. Finally, the mask is converted to a string and appended to the conversation via an open-ended language interface for both input and output.

Autoregressive Vision Loss for Multi-Label and Multi-Tasks. A core contribution of our work is the autoregressive supervision of vision tokens. We extend the next-token prediction (NTP) paradigm to vision tokens to facilitate a tighter alignment between image and text representations, ensuring robust dense prediction performance rather than restricting supervision solely to textual tokens. Acknowledging that a single image patch often encapsulates multiple semantic labels and task targets (see Figure 3), we employ a multi-label supervision objective for vision tokens. We term this NTP-M, a variant of the standard NTP adapted for multi-label and multi-task scenarios. Specifically, we transition from a single-label to a multi-label framework by constructing a multi-hot target vector for each image patch, where the token IDs corresponding to all relevant objects, tasks, and potential granularities are set to 1, while the remaining vocabulary indices are assigned 0. Unlike the standard NTP, which models a categorical distribution via Softmax, we model the probability of each token in the vocabulary independently. Consequently, we redefine the generative objective p(Y|Xv, Xinstruct) as the joint probability of independent Bernoulli trials over the vocabulary:

|V|

L

σ(zi,v)yi,v · (1 − σ(zi,v))(1−yi,v) , (7)

#### ∏

∏

p(Y|Xv, Xinstruct) =

v=1

i=1

where Xv, Xinstruct are the vision token input and instruction (prompt) input, respectively. L denotes the sequence length of vision tokens, |V| is the vocabulary size, σ(zi,v) represents the sigmoid-activated probability for token v at vision patch i, and yi,v ∈ {0, 1} indicates the ground truth presence of the corresponding target. Note that Xinstruct can influence the dense prediction process; for example, the prompt before the image guides depth estimation for the depth range of a certain camera.

Given the extensive vocabulary size inherent to VLMs, significant class imbalance poses a challenge to effective supervision. To address this, we implement a robust multi-label next-token prediction loss LNTP-M that replaces the naive averaging strategy to ensure stable convergence. Our core insight lies in independent positive-negative averaging coupled with relevant negative sampling. Specifically, we decouple the processing of positive and negative samples. We first calculate the mean loss for all valid positive samples. For negative samples, we rank them based on their predicted probabilities p (post-Sigmoid) in Eq. 7 and compute the average loss only for the top-k relevant negatives. This approach prevents the gradient dilution often caused by averaging an overwhelming number of irrelevant, low-response negative samples. Distinct from standard online hard example mining [Shrivastava et al., 2016], which ranks positive and negative samples jointly, our method prevents positive samples from being overshadowed or excluded by the sheer volume of hard negatives, thereby enhancing convergence efficiency. Furthermore, to accommodate incomplete annotations (e.g., instances with semantic labels but missing depth information), we apply a validity mask to exclude the corresponding indices from the averaging process. We formally define the sets for positive samples and relevant negative samples to reflect the sorting and masking logic. Let pi,j = σ(zi,j) be the Sigmoid-activated probability for the j-th token at step i. Let Mi be the set of valid indices for the current task (masking out missing annotations). The set of positive indices is defined as Pi = {j ∈ Mi | yi,j = 1}. For the negative samples, let the candidate set be Ci = {j ∈ Mi | yi,j = 0}. We define the relevant negative set Nihard as the subset of Ci with size k that maximizes the sum of predicted probabilities (i.e., the top-k sorting operation):

Nihard = argmax

#### ∑

pi,j . (8)

S⊂Ci,|S|=k

j∈S

The final loss LNTP-M is the summation of the independent averages over these sets:

 −

  . (9)

L

1

1

∑

|Pi| ∑

k ∑

log(pi,j) −

log(1 − pi,j)

LNTP-M =

i=1

j∈Pi

j∈Nihard

### 3 Pre-training

- 3.1 Training Recipe

The training of Youtu-VL follows a progressive, multi-stage training recipe. This pipeline is structured into four sequential phases, evolving from establishing a robust language foundation (Stages 1 and 2) to multimodal foundation pre-training. (Stage 3), and culminating in versatile task adaptation (Stage 4). The details of this training recipe are described in Figure 4. During Stages 3 and 4, we deploy a dual-stream supervision strategy to enforce comprehensive visual supervision. For general multimodal data, we impose the autoregressive visual reconstruction loss Limage (defined in Section 2.1), compelling the model to predict visual tokens as intrinsic generation targets alongside text. Complementarily, for vision-centric data, we incorporate the specialized loss LNTP-M (detailed in Section 2.2), which is essential for facilitating fine-grained dense perception capabilities.

[Figure 96]

Figure 4. The pre-training recipe for Youtu-VL. The top panel illustrates the evolution of the data mixture from Stage 1 to Stage 4: Stages 1 and 2 exclusively utilize pure text data to establish a strong linguistic foundation, while Stages 3 and

###### 4 progressively enhance multimodal capabilities. The bottom panel presents the learning rate schedule aligned with the training stages.

Stages 1 and 2: Language Backbone Pre-training. The Youtu-VL language backbone was trained through two distinct phases: Commonsense Pre-training (Stage 1) and STEM- and Coding-centric Pre-training (Stage

- 2). Collectively, these stages encompassed approximately 10T tokens of pure text data, empowering the model with exceptional proficiency in general reasoning, STEM, and coding tasks. Detailed specifications of the data mixture and training recipe are provided in the Youtu-LLM technical report [Lu et al., 2026].

- Stage 3: Multimodal Foundation Pre-training. To empower the model with robust visual perception and multimodal comprehension, we utilize a comprehensive multimodal dataset comprising image-caption pairs and data for vision-centric tasks. This diverse mixture facilitates the acquisition of broad world knowledge, cross-modal contextual understanding, and proficiency in various visual perception tasks. During this stage, all components of Youtu-VL—including the LLM backbone, the Vision Encoder, and the Projector—are fully trainable in an end-to-end manner. To preserve and further enhance the linguistic capabilities of the

- backbone, we incorporate the high-quality text corpus from the General Mid-Training phase of Youtu-LLM. By mixing this text data with multimodal samples, the model is trained on a total of approximately 1.8T tokens, balancing visual and linguistic capabilities.
- Stage 4: Versatile Task Adaptation. This phase specializes the model with capabilities across a wide range of tasks, including General VQA, OCR, STEM, GUI, Detection, Segmentation, Grounding, and Pose Estimation. The entire model undergoes end-to-end training on a diverse, high-quality multimodal instruction dataset comprising approximately 0.6T tokens, enabling it to generalize effectively to diverse user instructions. Furthermore, we incorporate a substantial volume of high-quality, synthetic short Chainof-Thought (CoT) data. This stage empowers the model with long-context understanding and generation, alongside fundamental logical reasoning skills.

##### 3.2 Pre-Training Data

###### 3.2.1 Vision-Centric Data

Text Data. Most of the vision-centric tasks described in Section 2.2 are achieved through text predictions. The majority of the data is constructed based on open-source datasets, combined with some internal data and synthetic data. We have elaborated on the data details according to the tasks. (1) Visual grounding: Beyond leveraging widely adopted open-source grounding datasets, we curated a diverse set of grounding queries derived from object detection and scene graph datasets. Furthermore, we employed an automated annotation pipeline to synthesize additional grounding data from large-scale image-text pairs. (2) Object detection: We utilized multiple open-source object detection datasets and formulated detection tasks with varying requirements based on bounding box annotations, enabling the model to recognize a broad spectrum of object categories. Additionally, we synthesized a denser and more diverse dataset to further bolster the model’s detection capabilities. (3) Polygon-based segmentation targets referring expression and instance tasks. We compress points within 20 for efficient training and avoid out of the window size.

For concept understanding and keypoint perception, the data include: (1) Object counting: In addition to open-source counting datasets, we augmented our training set by synthesizing counting data derived from visual grounding and detection data. Two inference variants are supported: directly counting and a detect-then-count manner. (2) Image classification: We leveraged open-source classification datasets to establish the model’s classification proficiency. This enables the recognition of not only common objects and scenes but also fine-grained biological taxonomic concepts. (3) Human pose estimation: We formulate pose estimation as a unified keypoint prediction problem, directly regressing joint coordinates for all persons in an image within a single forward pass.

Dense Labeled Data. Dense labeled data is primarily derived from open-source datasets and synthetic data. We have elaborated on the data processing strategies according to the specific dense prediction tasks. (1) Semantic segmentation: We structured the training data by mapping numeric mask labels to text descriptions. During the dataloading phase, labels are converted into token indices, where valid category indices are assigned positive labels, and unlabeled or invalid regions are explicitly ignored during loss calculation. (2) Instance and referring segmentation: We integrated binary segmentation with detection and grounding tasks to support high-quality scenarios. We employed visual prompts via random-colored bounding boxes and assigned foreground (<FG>) and background (<BG>) vocabularies to enable high-quality box-prompted segmentation. Furthermore, we applied diverse augmentation techniques, including padding, cropping, and resizing, to enhance model robustness. (3) Depth estimation: We implemented a linear quantization pipeline to map valid depth ranges into discrete bins (1–1000) for a known camera, with label 0 for ignore labels. Prompts are processed before the image to allow the model to adapt to varying camera parameters, and augmentation involves color jitter and random cutout, using original values for high-resolution data

and fixed-ratio resizing (e.g., 3x) for low-resolution sources. The predictions are then dequantized to the real depth for evaluation.

1. Object Detection

Open Det. Data

Copy Paste & Class Binding

Grounding

Mask

Models

Segmentator

Labeled Data

[Figure 99]

2. Semantic Segmentation

[Figure 100]

[Figure 101]

Open Seg. Data

Copy Paste & Crop & Binding

Mask Segmentator

Mask Labeler

###### Dataset for Open World Scenarios

Data

Massive Vision- Augmentation

3. Depth Estimation

Centric Data

Open Depth Data

Depth Models

Size Align Quantification

Unlabeled Data

- Figure 5. Data synthesis pipeline for open-world scenarios. The framework processes massive vision-centric data through two parallel branches: (top) object detection and semantic segmentation, utilizing grounding models for raw data and data binding strategies for labeled data; and (bottom) depth estimation, employing depth models and quantization. The pipeline applies specific augmentations to generate a comprehensive dataset for open-world tasks.

Data Pipeline for Open World Scenarios. The open-scene dataset is constructed by processing massive vision-centric data through two primary branches to support object detection, segmentation, and depth estimation. For object detection and semantic segmentation, the pipeline handles both raw and labeled data. Raw data is processed via grounding models and segmentators, while labeled data undergoes data binding to handle single, multiple, or sets of classes. This synthesis addresses four requirements: specifying positive categories, mixing positive and negative categories (for closed-set scenarios), and generating dense scenes. These inputs result in class-merged and class-sampled outputs. Specifically, the "arbitrary category" scenario employs a Copy-Paste strategy where transparent objects undergo random resizing and rotation before being densely placed on backgrounds. For depth estimation, the pipeline similarly splits processing. Raw data is passed through depth models to generate pseudo-labels using open-source models. Labeled data undergoes resize and quantization to simulate stable camera parameters with a fixed focal length of 2000 pixels. We utilize log-uniform quantization with flexible prompt placement to define a valid depth range of 0.5m to 100m; consequently, input images with differing focal lengths represent relative depth and require scaling. Finally, the outputs from both branches pass through task-specific augmentations (detection, segmentation, and depth) and cropped binary segmentation to yield the final dataset for open world scenarios.

###### 3.2.2 Image Caption and Knowledge Data

Our data acquisition pipeline begins with a massive aggregation of open-source image-text pairs. To ensure the integrity and robust foundation of this corpus, we implemented a rigorous multi-stage filtration protocol. This initial phase involved strict image-text alignment filtering via CLIP scores [Radford et al., 2021, Wei

- et al., 2025, Fang et al., 2023], the excision of NSFW content, and resolution constraints. Following the pruning of unavailable links and corrupted data, this preliminary cleaning yielded a raw corpus comprising approximately 5 trillion tokens.

Building upon this foundation, we sought to maximize information density and training efficiency through a three-pronged strategy: Concept-Balanced Sampling, Rare Class Mining, and Knowledge-Injected Recaptioning.

• Concept-Balanced Sampling. Inspired by the methodologies proposed in MetaCLIP [Chuang et al.,

- 1. Concept-balanced Sampling
- 2. Rare Class Mining
- 3. Knowledge-injected Recaptioning

General Concapts (CLIP Retrieval,

Saturation)

Named Entities (Keyword Matching, Verification)

Bilingual Lexicon

LLM Taxonomy

HighKnowledge Subset

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

CLIP Score Filter

[Figure 107]

[Figure 108]

MD5 Deduplication

NSFW

Excision

[Figure 109]

JEPA Model

Semantic Dense Score

Retrieve Rare Instances

Long-tail Subset

Resolution Constraints

Massive Image-Text

High-quality

Repetition Detection

Dataset

Paris

Fine-tuned Captioner

Rewrite with Knowledge

Dense Descriptions

- Figure 6. The workflow for synthesizing knowledge-dense image caption and knowledge data. Starting from massive raw image-text pairs, the pipeline proceeds through three main stages: (1) a multi-stage filtration protocol to ensure basic quality; (2) a core enhancement phase featuring Concept-based Sampling, Rare Class Mining, and Knowledge-Injected Recaptioning to maximize information density and diversity; and (3) a final purification stage for deduplication.

2025], we moved beyond random sampling to a rigorous, ontology-driven approach. We curated a comprehensive bilingual lexicon (Chinese and English) derived from multiple encyclopedic sources. LLMs were leveraged to taxonomize each entry into general concepts (e.g., "soccer," "cinematography") or named entities (e.g., specific movie titles, historical figures). For general concepts, we employed CLIP-score retrieval to mine the training data, continuing until a pre-defined saturation threshold was met for each concept. For named entities, where factual precision is paramount, we utilized stringent keyword matching and entity verification to ensure correct association. This balanced sampling strategy allowed us to effectively extract a subset of high-knowledge-density data from the otherwise sparse and noisy raw distribution.

- • Rare Class Mining via JEPA Score. Concept-based sampling is inherently biased towards common entities, leaving the significant long-tail of the data distribution underexplored. To rectify this, we leveraged a JEPA-based density metric to identify and retrieve rare instances located in the sparse regions of the latent space. Specifically, we utilized state-of-the-art JEPA (Joint-Embedding Predictive Architecture) series models to compute a "semantic density score" for each data point, akin to methods discussed in [Balestriero et al., 2025]. By identifying samples in low-density regions of the latent space, we selectively retained rare but semantically unique examples based on adaptive thresholds, ensuring the model is exposed to a diverse visual manifold.
- • Knowledge-Injected Recaptioning. While large-scale open-source image-captioning datasets have fueled recent progress, they frequently suffer from brevity and noise. To elevate the model’s finegrained visual understanding, we developed a specialized "Knowledge-Injected Captioner" by finetuning a multimodal model. This captioner was deployed to rewrite the dataset, transforming sparse web texts into dense, detailed, and visually grounded descriptions. This process not only hallucinates less but also aligns the textual modality precisely with the visual content.

Finally, the curated dataset underwent a purification stage. We applied MD5 checksum deduplication to remove redundancy and utilized repetition detection heuristics to eliminate cyclical generation artifacts. The culmination of this pipeline is a clean, compact, and high-fidelity dataset containing approximately 1 trillion tokens, serving as the cornerstone of our visual-language alignment.

###### 3.2.3 Optical Character Recognition (OCR) Data

To bolster proficiency in OCR and fine-grained chart understanding, we curated a high-quality dataset focusing on visual perception and reasoning. Initially, we integrated existing open-source human-annotated datasets to establish a baseline capability. However, we identified significant limitations in these legacy resources: (1) Brevity, where responses are often overly concise; (2) Monotony, characterized by a lack of diversity in prompt patterns; and (3) Absence of Intermediate Steps, as human annotations typically provide final answers without the intermediate reasoning process (i.e., short Chain-of-Thought). This absence leads to suboptimal training efficiency and limited generalization to complex, unseen queries. Consequently, we shifted our strategy to synthesize a large-scale, verbose dataset, which constitutes the core of our training corpus.

Our data synthesis pipeline employs a three-pronged approach to ensure high quality and diversity. First, to leverage vast amounts of unlabeled data, we aggregated a collection of raw PDFs, academic charts, and real-world document images. We utilized an LLM to generate diverse questions from various perspectives and difficulty levels, followed by a powerful VLM to produce detailed, descriptive answers, creating rich instruction-following pairs. Second, targeting the deficiencies in existing human-labeled data, we introduced a logic-consistency refinement pipeline. An LLM evaluates the logical coherence of existing QA pairs; for samples deemed opaque, a VLM regenerates the response with explicit step-by-step reasoning. This process is safeguarded by a closed-loop verification mechanism where an LLM validates that the new reasoning path remains consistent with the original ground truth. Finally, to further scale the training data, we constructed a massive synthetic OCR dataset from pure text corpora using a robust rendering engine. This engine applies randomized typographical augmentations and simulates real-world acquisition conditions—specifically by superimposing complex background textures (e.g., paper grain, shadows) and applying physical distortions (e.g., blur, noise, affine transformations, and rotation)—to generate millions of clean, accurate, and visually diverse training samples.

###### 3.2.4 Science, Technology, Engineering, and Mathematics (STEM) Data

The STEM dataset is designed to enhance capabilities in image-based analysis and reasoning. The dataset spans a wide range of visually intensive STEM scenarios, including geometric diagrams, plots and charts, physics illustrations, chemical structures, engineering schematics, and exam-style problem figures. To construct a dataset capable of supporting complex reasoning, we implemented a sophisticated curation pipeline comprising three key stages:

###### 1. Multi-Dimensional Quality Filtering 2. Synthesis and Consistency Verification 3. Visual-Grounded Question Expansion

Has Ground

Yes

Visual

Truth (GT)?

Filtration (Edit-distance,

Question

Augmented

Yes

VLM Resynthesis

Relevance

LLM Verification

Prediction VLM (Image+Answer

Pairs

[Figure 111]

[Figure 112]

LLM Dedupe, Relevance)

(Add Reasoning Steps)

No

-> New Question)

Satisfies High Scoring Thresholds? System (VLMs & LLMs)

Re-evaluate

Answer

LLM Question Variation

against Original

Correctness

Diverse

(Prompt Perturbation)

VLM Formulate

High-Fidelity

LLM Extract

Image Caption Data

STEM-related Datasets

Educational Questions

STEM Corpus

Knowledge Entities

VLM Answer Queries

No

Reasoning Completeness

Keep

Consistency Filter

Render Text on Images (OCR Style, Real-world Simulation)

Superior Candidate

(Reasoning Invariance)

- Figure 7. Data construction pipeline for STEM. The pipeline consists of three key phases: (1) Multi-Dimensional Quality Filtering to ensure high visual and logical standards; (2) Synthesis and Consistency Verification to enhance reasoning details and ensure fidelity; and (3) Visual-Grounded Question Expansion to augment the dataset with diverse queries and real-world simulations.

###### • Multi-Dimensional Quality Filtering. We initiated the process by aggregating diverse STEM-related

datasets. To purge low-quality samples, we deployed a scoring system utilizing both VLMs and LLMs. Each sample was rigorously evaluated across three critical dimensions: visual relevance, answer correctness, and reasoning completeness. Only data satisfying high thresholds in all metrics were retained for the subsequent refinement phases.

- • Synthesis and Consistency Verification. Recognizing that legacy data often lacks detailed reasoning steps, we utilized a VLM to resynthesize answers conditioned on the image and question. To ensure the fidelity of these synthetic responses, we adopted a bifurcated verification strategy. For samples with existing ground truth (GT), an LLM verified the semantic alignment between the synthesized answer and the original answer. For synthetic datasets lacking ground truth, we implemented a verification protocol. We employed an LLM to generate multiple semantically equivalent but syntactically distinct variations of the original question (i.e., prompt perturbation). The VLM was then tasked with answering these diverse queries. We enforced a strict consistency filter, retaining only those samples where the model demonstrated reasoning invariance—yielding consistent answers across all query variations. This ensures that the generated knowledge is stable. Finally, we re-evaluated the new, verbose answers against the original ones using our quality metrics, preserving the superior candidate to maximize information density.
- • Visual-Grounded Question Expansion. A single question often fails to exhaust the knowledge embedded in a complex image. To rectify this, we fine-tuned a specialized "Question Prediction VLM" that takes an image and an answer as input to predict the corresponding question. This allowed us to generate diverse, meaningful queries for each image. These augmented pairs underwent filtration via edit-distance heuristics, LLM-based deduplication, and visual relevance scoring. Additionally, we repurposed our Image Caption data by using LLMs to extract knowledge entities from captions, prompting a VLM to formulate educational questions based on these points. Finally, to simulate real-world usage scenarios (e.g., photographing exam papers), we adopted the rendering strategy from our OCR pipeline. Textual questions were dynamically rendered onto the images, creating a robust dataset that mimics the visual noise and layout of physical documents.

Through the synergy of these pipelines, we ultimately constructed a high-fidelity STEM corpus characterized by rigorous reasoning chains, diverse query perspectives, and robust visual grounding, significantly enhancing the model’s capabilities in STEM.

###### 3.2.5 Graphical User Interface (GUI) Data

To empower Youtu-VL with agentic capabilities for autonomous GUI interaction, we conduct continual pre-training using a dual-stream data curation pipeline composed of single-turn grounding data and long-horizon interaction trajectories.

- • Granular Perception & Grounding: To establish robust atomic UI understanding and operational capability, we aggregate single-turn data from various open-source datasets. We implement a rigorous data production and filtering process, utilizing an ensemble LLM-as-a-judge framework [Lin et al., 2025] to ensure label correctness. This high-fidelity data is crucial for enhancing the model’s finegrained capabilities, specifically in element description, dense captioning, and dense grounding, thereby enabling a robust understanding of diverse user interfaces.
- • Sequential Interaction Dynamics: To advance the policy model’s reasoning regarding environmental dynamics and multi-step interaction logic, we synthesize large-scale, cross-platform trajectories spanning desktop, mobile, and web environments [Xie et al., 2024, Shi et al., 2025] using large-scale sandbox systems. This synthesis pipeline is designed to maximize task diversity and simulate realistic operation environments across different applications. Throughout the synthesis process, we implement strict

- privacy and access control protocols to ensure the produced trajectories in the sandbox are free from personally identifiable information and unauthorized content access.
- • Reward-Guided Hybrid Verification: We enforce strict quality control across both data streams through a hierarchical filtering strategy. Candidates are ranked and filtered based on scores from our ensemble judging framework [Lin et al., 2025]: the highest-scoring trajectories undergo rigorous human annotation to construct a premium dataset for post-training, while the remaining validated data is utilized for continual pre-training.

###### 3.2.6 Pure Text Data

To preserve the model’s fundamental language abilities, we incorporated the mid-training data used by Youtu-LLM, encompassing domains such as Web, Encyclopedia, STEM, and Coding. Comprehensive details regarding data acquisition and curation can be found in the Youtu-LLM technical report [Lu et al., 2026].

###### Ablation Study: w/ VLUAS vs w/o VLUAS

Stage 3 Stage 4

0.65

AvgScore(27Benchmarks)

0.60

0.55

0.50

0.45

0.40

w/o VLUAS

0.35

w/ VLUAS

0.30

0.0 0.1 0.2 0.3 0.4 0.5 0.6

Training Tokens (T)

- Figure 8. Ablation Study on VLUAS Effectiveness. Comparative scaling curves of the model trained with (red) and without (blue) the proposed Unified Pre-training (VLUAS) strategy. The results indicate a critical divergence in scaling behavior: while the baseline model (wo/ VLUAS) exhibits clear signs of performance saturation during the later phases of Stage 3, the VLUAS-enhanced model maintains a superior performance trajectory with higher data efficiency. This consistent gap across both Stage 3 and Stage 4 empirically validates that incorporating visual supervision significantly alleviates data saturation and effectively raises the upper bound of multimodal capabilities.

##### 3.3 Analysis of Pre-training Paradigm

Standard VLM benchmarks predominantly employ zero-shot settings. However, pre-training checkpoints often lack instruction alignment, leading to format mismatches that obscure the model’s intrinsic capabilities. To decouple representation quality from instruction-following skills, we adopt a few-shot evaluation protocol aligned with recent foundational LLM assessment methodologies [DeepSeek-AI et al., 2025]. Specifically, we employ in-context learning to standardize output formats, thereby enabling a fair assessment of the base model’s progress without relying on explicit instruction tuning.

Utilizing this rigorous protocol, we analyze the scaling dynamics of multimodal understanding across Stages 3 and 4. As illustrated in Figure 8, the average performance across 27 multimodal benchmarks highlights a distinct divergence between our VLUAS paradigm and the conventional text-dominant supervision baseline. The text-dominant model exhibits signs of performance saturation early in Stage 3, pointing to a fundamental bottleneck when supervision is confined to the linguistic modality. Conversely, VLUAS sustains a steep

learning trajectory, effectively bypassing this premature plateau. This advantage extends into Stage 4; even with improved data quality, VLUAS demonstrates superior data efficiency and consistently widens the performance gap. These empirical results confirm that integrating generative visual supervision serves not merely as an auxiliary task, but fundamentally elevates the model’s capability ceiling.

##### 3.4 Analysis of Data Scaling

###### Data Scaling of Image-Text Unified Pre-Training

0.80

Stage 3 Stage 4

0.75

AvgScore(27Benchmarks)

0.70

0.65

0.60

0.55

0.50

0.45

w/ VLUAS

0.40

0.0 0.5 1.0 1.5 2.0 2.5

Training Tokens (T)

(a) Scalability of Unified Pre-training.

Stage 3

###### Stage 4

= 0.102

= 0.079

0.60

0.30

Avg.Error

0.50

0.40

1013

1013

GFLOPs

GFLOPs

(b) Power-Law Scaling Dynamics.

- Figure 9. Scaling analysis of the VLUAS paradigm. (a) The scaling trajectory of the average performance across 27 multimodal benchmarks with respect to training tokens. The model demonstrates robust and continuous capability growth throughout the training process across distinct data regimes (Stage 3 and Stage 4), effectively leveraging massivescale data without exhibiting empirical saturation. (b) The analysis of average evaluation error (1 −Score) versus training compute (GFLOPs) on a log-log scale. The training dynamics in both Stage 3 and Stage 4 strictly adhere to neural scaling laws, characterized by scaling exponents of α ≈ 0.102 and α ≈ 0.079, respectively. This precise linear relationship confirms the predictability and computational efficiency of the VLUAS paradigm.

To systematically evaluate the scalability and stability of our pre-training, we conducted a comprehensive analysis spanning 2.4T tokens. As illustrated in Figure 9a, the model exhibits a robust, monotonic positive correlation between the unified training volume and multimodal performance (averaged across 27 benchmarks). The training trajectory spans two distinct data regimes, Stage 3 (Multimodal Foundation Pre-training) and Stage 4 (Versatile Task Adaptation), where our optimization strategy consistently translates increased computational investment into tangible capability gains. Throughout the pre-training lifecycle, encompassing both the foundation learning on mixed text-visual corpora in Stage 3 and the high-quality instruction tuning in Stage 4, the average benchmark score demonstrates a sustained upward trend, rising from approximately 0.43 to a final convergence of over 0.74. This continuous growth confirms that our pretraining effectively leverages massive-scale data, maintaining high sample efficiency without encountering the early saturation or instability often observed in multi-task optimization.

By evaluating how evaluation error (ϵ) scales with training compute (GFLOPs), we quantify the predictability of the model’s performance trajectory. As visualized in the log-log plots of Figure 9b, our unified pre-training process closely follows neural scaling laws, fitting the power-law equation L(C) ∝ C−α. Specifically, the Stage

- 3 regime exhibits a steep scaling exponent of α ≈ 0.102, indicating that the integration of general text corpora with multimodal samples facilitates rapid information absorption and high data efficiency in the initial phase. Upon transitioning to Stage 4, the model enters a refined scaling regime with α ≈ 0.079. While the exponent naturally moderates as performance approaches the irreducible error floor, the sustained linearity on the logarithmic scale serves as strong empirical evidence that our training infrastructure maintains computational efficiency even during complex task adaptation and long-context CoT generation, effectively converting massive resources into predictable model improvements.

w. Vision Supervision w/o Vision Supervision

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

Input Images Youtu-VL-4B Qwen2.5-VL-3B Qwen3-VL-4B

Youtu-VL-4B w/o

- Figure 10. Visualization of Vision Token Representations. This figure compares the Principal Component Analysis (PCA) visualizations [Oquab et al., 2023] of the last-layer hidden states of vision tokens. We contrast Youtu-VL-4B (with vision token supervision) against three models without such supervision: Youtu-VL-4B w/o, Qwen2.5-VL-3BInstruct [Yang et al., 2024], and Qwen3-VL-4B-Instruct [Bai et al., 2025]. Leveraging vision token supervision, our model exhibits outstanding feature separation and visualization quality compared to VLMs lacking this supervision.

##### 3.5 Visual Representation Analysis

To evaluate the quality of visual representations, we performed PCA on the last-layer hidden states of the vision tokens output by the LLM and projected them back onto the spatial dimensions of the original images,

- as shown in Figure 10. For VLMs lacking vision token supervision, we observe a clear progression from Qwen2.5-VL to Qwen3-VL; the latter exhibits superior representation quality with greater distinctiveness between objects, which positively correlates with its improved model performance. Notably, our YoutuVL-4B (without vision supervision) demonstrates representation capabilities comparable to Qwen3-VL. However, upon integrating vision token supervision, the visual representations of Youtu-VL-4B show significant improvement, characterized by clear semantic structures and sharp object separation. This qualitative result provides compelling evidence for the effectiveness and necessity of our proposed vision supervision, especially in the visual representation aspect.

### 4 Post-training

##### 4.1 Supervised Fine-Tuning

###### 4.1.1 Training Recipe

The supervised fine-tuning aims to refine the model’s capability to understand complex instructions, enhance reasoning, and align with human preferences. During this phase, we extend the context window from the initial 16K tokens to 32K tokens. We employ the AdamW [Loshchilov and Hutter] optimizer with a cosine

learning rate scheduler. Specifically, we set a 5% warmup ratio, with the learning rate decaying from a peak of 2 × 10−5 to 2 × 10−6.

###### 4.1.2 Data Curation

We are dedicated to curating a comprehensive, high-quality SFT dataset that encompasses a diverse spectrum of multimodal tasks. To achieve this, we implemented a multi-source data acquisition strategy:

- • High-Quality Mining from Pre-training Corpora. To extract high-value instruction-following data from our massive pre-training corpus, we employed a stratified sampling strategy. We first leveraged a VLM to assess the quality and alignment of individual samples. To further ensure task diversity, we established a fine-grained keyword taxonomy. By balancing samples against this taxonomy and filtering based on VLM scores, we curated a premium subset from the raw pool, effectively maximizing data efficiency.
- • Refined Open-Source Data. Existing open-source datasets are often limited by simple "Yes/No" or single-word answers, which are insufficient for training robust reasoning capabilities. We treated these datasets merely as a source of images and prompts, while synthesizing entirely new target responses. By employing a rigorous rewrite-and-expand pipeline, we prompted a powerful VLM to analyze the visual details and generate comprehensive, paragraph-level descriptions and reasoning steps. This approach effectively "up-scales" the information density of the original data by an order of magnitude.
- • Grounding
- • Detection
- • Counting
- • Segmentation

###### Vision-centric Tasks

###### STEM Tasks

General Tasks

[Figure 133]

[Figure 134]

[Figure 135]

- • Science
- • Technology
- • Engineering
- • Mathematics

- • OCR
- • Chart
- • General VQA
- • Instruction following

Perception RL Reasoning RL General RL

Reward System

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

SFT Model Rule-based Reward LLM-based Reward RL Model

Figure 11. The Architecture of our multi-stage reinforcement learning framework.

##### 4.2 Reinforcement Learning

To further unlock our model’s potential beyond the supervised fine-tuning phase, we employ a multistage reinforcement learning framework. As illustrated in Figure 11, our proposed framework follows a sophisticated three-stage training process:

- • Perception RL. The first stage is dedicated to enhancing the model’s fine-grained visual perception. During this phase, we focus on vision-centric tasks, such as visual grounding, object detection, semantic segmentation, object counting, etc. By utilizing RL objectives tailored for these tasks, our model is optimized to precisely localize and interpret structural elements within an image.

- • Reasoning RL. The second stage targets augmenting the model’s complex reasoning capabilities. We leverage a curated STEM (Science, Technology, Engineering, and Mathematics) dataset that integrates both text-only and multimodal data. By employing verifiable reward signals to guide the model, we effectively bridge the gap between fundamental visual recognition and sophisticated problem-solving.
- • General RL. Building upon the reinforcement learning on verifiable STEM tasks, the third stage shifts the focus towards broader generalization across a wider range of tasks. By incorporating a diverse set of general VQA challenges, including OCR, chart understanding, instruction following, etc., we further improve the model’s generalization capabilities.

###### 4.2.1 Data Curation

To ensure the stability and efficiency of our multi-stage reinforcement learning framework, we implement a rigorous data curation pipeline. Our objective is to construct a diverse, high-quality and most importantly, verifiable dataset across various domains, including vision-centric tasks, STEM, OCR, chart understanding, general VQA, and complex instruction following. The curation process consists of the following four steps:

- • Task Categorization. We begin by categorizing samples collected from a broad spectrum of opensource multimodal datasets into the following domains: vision-centric tasks, STEM, OCR, chart understanding, general VQA, and complex instruction-following. This categorization enables us to tailor domain-specific reward signals that align with the requirements of each task.
- • Verifiability-driven Filtering. To ensure the precision of reward signals during the RL process, we further filter out samples that cannot be easily verified. Specifically, we first classify collected samples from each task domain into objective and subjective types. Objective samples are prioritized for training, as they provide deterministic ground-truth labels that facilitate automated and high-fidelity verification. Furthermore, we filter out samples that are inherently difficult to verify or susceptible to random guessing, such as multiple-choice, true/false, and multi-question prompts.
- • Quality Assurance. To guarantee the correctness of the response used for training, we employ a consensus-based validation mechanism. We utilize a suite of existing models to perform crossverification on the candidate data. Only samples where a clear consensus is reached are retained.
- • Complexity Calibration. Finally, we conduct offline difficulty grading to optimize the learning curriculum. For each candidate sample, we generate eight independent responses using our most advanced SFT model. Samples where all responses are correct are discarded.

###### 4.2.2 Reward System

As summarized in Table 1, we design task-specific reward signals to provide reliable optimization targets across different tasks. For vision-centric tasks (e.g., grounding, detection, segmentation, counting), rewards are computed by comparing structured predictions against deterministic annotations via standard metrics like IoU or mAP, yielding high-fidelity and fully verifiable feedback. In STEM domains, we enforce a strict output format to facilitate automated answer extraction and apply rule-based validation. For cases where rule-based validation fails, we additionally perform a consistency check with a strong LLM-based verifier. For OCR and chart understanding, we utilize exact string matching and edit distance for spotting and parsing tasks, respectively. To address more general VQA tasks, we use LLM-based consistency checking against reference answers and model responses. Finally, for complex instruction-following tasks, we employ an LLM-as-a-judge with detailed rubrics and reference answers to obtain a scalar reward.

Domain Task Reward Design

IoU-based bounding box reward. Full score if IoU ≥ 0.5, otherwise proportional to IoU.

Grounding

mAP over IoU thresholds {0.50:0.95}. Predictions of unseen classes are treated as false positives.

Detection

Vision-Centric

Exact match for small counts. Relative error-based soft reward for large counts.

Counting

Class-label IoU for semantic segmentation. Polygon IoU or AP@50 for referring and instance segmentation.

Segmentation

Rule-based answer validation. LLM-based consistency checking when needed.

STEM Math & Science

Spotting Exact string matching between predicted and ground-truth text. Parsing Reward based on edit distance between structured outputs. VQA Perform consistency checking with LLM.

OCR& Chart

General VQA VQA Perform consistency checking with LLM. Complex Instructions VQA

Perform LLM-based scoring with detailed rubrics and reference answers.

Table 1. Details of the reward functions used for different tasks.

Beyond task-specific rewards, following Youtu-LLM [Lu et al., 2026], we incorporate two auxiliary rewards to prevent common failure modes. (i) Language Consistency Reward discourages code-switching and mixed-language outputs by penalizing responses whose dominant language deviates from the pre-specified target language. (ii) Repetition Detection Reward penalizes degenerate generations with excessive n-gram repetition, mitigating cases where the model loops and fails to terminate.

###### 4.2.3 Training Recipe

Training Configuration. We follow the setup adopted in DAPO [Yu et al., 2025]. The maximum context length is set to 32,768 tokens. In each training stage, we sample 6,144 rollouts of the current policy. Policy optimization is performed using a mini-batch size of 1,536, with 4 gradient update steps per training iteration. The learning rate of the policy network is fixed at 1 × 10−6. To balance training stability and policy exploration, the clipping range is set to [0.20,0.24], which helps prevent overly aggressive policy updates while encouraging sufficient exploration during training. In addition, to suppress excessively long generations, we introduce a soft overlong punishment mechanism. Specifically, we analyze the distribution of model output lengths and set the overlong threshold accordingly. Following previous work [Qi et al., 2025], we remove the KL penalty term during training and adopt FP16 training. Empirically, these design choices lead to faster convergence and more stable training dynamics.

Reward-Variance-Aware Sampling. Directly applying standard sampling strategies in reinforcement learning often leads to training instability. When advantage values approach zero, the corresponding policy gradients diminish, reducing effective gradient magnitudes and increasing sensitivity to noise. This issue is further amplified by the introduction of soft overlong punishment, which assigns negative rewards to some samples and causes zero-reward samples without length penalties to dominate mini-batch updates. In addition, unlike prior settings with discrete rewards (e.g., binary 0/1 signals), our framework employs continuous reward values. This problem is especially pronounced in vision-centric tasks, where conventional reward-based sampling heuristics retain many samples with highly similar rewards, resulting in insufficient reward discrimination. To address these issues, we propose a variance-based dynamic sampling strategy. Specifically, for each rollout group Oq = {o1, . . . , oG} generated for prompt q, we compute the variance of the corresponding rewards and discard groups whose reward variance falls below a predefined threshold. In addition, we require that at least one sample within a rollout group receives a positive reward, which helps

avoid misleading optimization signals when the overall reward signal is non-positive and further improves training stability. This strategy effectively removes sample groups with insufficient reward discrimination, improving sample quality and stabilizing policy optimization.

Consistent Sampling. The mismatch between the training backend and the inference engine has been identified in previous work as a critical factor affecting the effectiveness of reinforcement learning. To mitigate this issue, we follow Youtu-LLM [Lu et al., 2026] and adopt a consistent sampling strategy to improve the stability of training. For a given prompt q, the KL metric K(q) is used to measure the divergence between the training policy and the rollout policy.

K(q) = Ea∼πrollout(·|q)

πθ(a|q) πrollout(a|q) − 1 − log

πθ(a|q) πrollout(a|q)

. (10)

For each prompt q, a rollout group Oq = {o1, . . . , oG} generated for the prompt is admitted for training only when the corresponding policy divergence satisfies K(q) ≤ τK. Rollout groups that violate this condition are directly filtered out during the sampling stage, ensuring that the model is optimized exclusively on samples with constrained policy drift.

Based on the above sampling strategies, the final reinforcement learning objective is defined as:

###### J (θ) = Eq∼D,{o

i}iG=1∼πrollout(·|q)

G

1 ∑iG=1 |oi|

∑

i=1

|oi|

min ri,t(θ) Aˆi,t, clip ri,t(θ),1 − ϵlow,1 + ϵhigh A ˆi,t (11)

#### ∑

t=1

R(oi) > 0, K(q) ≤ τK, Var {R(oi)}iG=1 > τV. (12)

s.t. max

1≤i≤G

### 5 Evaluation

We conducted an extensive evaluation of Youtu-VL on a suite of 30 vision-centric and 45 general multimodal benchmarks. To our knowledge, this is the pioneering effort to unify the assessment of a VLM across dozens of distinct tasks, ranging from grounding, detection, classification, counting, segmentation, depth estimation, and pose estimation to visual query answering, OCR, and GUI operations. Addressing the limitations of existing frameworks [Li et al., 2024, Duan et al., 2024, Zhang et al., 2024] in handling such diversity, we extended VLMEvalKit [Duan et al., 2024] by integrating additional vision-centric and plain text evaluation tasks, thereby establishing a holistic and unified evaluation pipeline. As illustrated in the two bar charts at the bottom of Figure 1, we showcase the model’s proficiency across various dimensions in comparison to other models. Specifically for the general multimodal benchmarks, we calculated the average score for each category using only those benchmarks where results were available for all three models. Detailed empirical findings for each evaluation benchmark are provided in the following subsections.

##### 5.1 Vision-Centric Tasks

Existing benchmarks have inconsistent label set definitions across datasets. To resolve this, we include the benchmark name in the prompt to provide necessary context. Meanwhile, we support manually specifying

label sets (e.g., for semantic segmentation). Given that full label sets are often too large for prompt context windows, the name of the benchmark is the main type. For practical inference tasks, users can supply a custom label set to identify target objects or some flexible prompts, supporting both present and absent categories. Evaluation details and prompts are given in Appendix A.1. The evaluation results are given in Table 2 with experiment analysis below.

Visual Grounding. We evaluate Youtu-VL on the standard RefCOCO/+/g benchmarks [Yu et al., 2016] to assess its grounding capabilities. Youtu-VL demonstrates outstanding grounding performance, achieving an average score of 91.8% across all RefCOCO splits. In comparison, the average result is 89.4% for InternVL-

- 3.5-4B and 91.6% for the proprietary Seed1.5-VL [Guo et al., 2025]. These results demonstrate Youtu-VL’s proficiency in visual grounding tasks.

Object Detection. For object detection, Youtu-VL generates direct textual outputs without extra heads. This approach yields competitive performance on COCO [Lin et al., 2014] using the 2017 val split. Specifically, Youtu-VL achieves 47.1% mAP, while the result of GiT [Wang et al., 2024] is 46.7%. It also maintains comparable performance to UFO [Tang et al., 2025] (48.9% mAP), despite UFO being a larger, task-aware model that benefits from dense proposals and parallel decoding. Similarly, VisionLLM-v2 [Wu et al., 2024] gains benefits from its task-specific decoder with a 56.7% mAP. In comparison to these methods, Youtu-VL requires no additional training or inference modifications while still achieving competitive performance.

Semantic Segmentation. For semantic segmentation, Youtu-VL directly predicts dense predictions from the output logits of vision tokens. This approach demonstrates strong versatility, while other general-purpose VLMs like Qwen3-VL and InternVL-3.5 do not support these dense prediction tasks. In terms of performance, Youtu-VL achieves significant improvements over vision-centric models; for instance, on the ADE20k [Zhou et al., 2017] dataset, it attains 54.2 mIoU, while the results of GiT is 47.8%. Furthermore, Youtu-VL proves competitive results against the classic specialist method Mask2Former [Cheng et al., 2022] and CLIP-based VLMs [Lan et al., 2024, Qiu et al., 2025, Liu et al., 2024, Zhou et al., 2022]. Youtu-VL also demonstrates strong performance on the COCOStuff [Caesar et al., 2018] datasets, without fine-tuning like SAN [Xu et al., 2023] (52.5% vs. 45.7% in mIoU). In addition to quantitative analysis, segmentation also supports open scenarios, such as specifying certain categories and as many segmentation categories as possible. These capabilities are not supported by classic specialists and vision-centric VLMs. We conducted a qualitative analysis and visualization of this in Appendix C.

Referring Expression Segmentation. In this task, Youtu-VL employs a grounding-then-segmentation manner to generate precise segmentation masks based on specific textual descriptions. Specifically, we generate the grounding box and draw on the image, then crop to segment the foreground semantics. This operate do not rely on any extra decoder, like SAM or other task-specific heads, while significantly beyond the polygon-based referring expression segmentation. This method yields state-of-the-art performance among comparable models. On the RefCOCO [Yu et al., 2016] val set, Youtu-VL achieves 80.7% mIoU, while the results are 80.0% [Tang et al., 2025], 76.6% [Wu et al., 2024], 80.5% [Liu et al., 2025], and 79.3% [Ouyang et al., 2025] for other settings. Our method does not rely on predicting extra mask embedding tokens or extra heads, which is simple yet effective. These results prove that a generalist model can achieve pixel-level precision without relying on the extra task-specific tokens or decoders with high-quality results.

Depth Estimation. This task is similar to semantic segmentation, where the class names are replaced with the names of quantized bins. Youtu-VL predicts dense depth maps directly from monocular images. Unlike most general-purpose and vision-centric VLMs (e.g., Qwen3-VL [Bai et al., 2025], GiT [Wang et al., 2024]), which lack support for this spatial-related task, Youtu-VL integrates this capability seamlessly. On the NYUv2 [Silberman et al., 2012] dataset, Youtu-VL achieves a δ1 of 90.4%, which is slightly lower than the 8B-parameter UFO model (δ1 at 93.6%). Because no task-specific fine-tuning is applied with less parameters. In comparison, the performance of another VLM-based method DepthLLM-3B [Cai et al., 2025], is 86.8%, which requires inference for each point. In contrast, our model obtains depth information intrinsically without any additional inference. Youtu-VL demonstrates strong performance on the DDAD [Guizilini et al.,

|Benchmarks<br><br>|General-Purpose VLMs<br><br>Youtu-VL Qwen3-VL InternVL-3.5 4B (instruct) 4B (instruct) 4B|Vision-Centric VLMs<br><br>†UFO GiT *VisionLLM v2<br><br>8B 756M 7B<br><br>|Classic Specialists<br><br>*VLM *Non-VLM<br><br>- -|
|---|---|---|---|
|Visual Grounding RefCOCO val<br><br>RefCOCO testA<br>RefCOCO testB RefCOCO+ val<br><br><br>RefCOCO+ testA<br>RefCOCO+ testB RefCOCOg val RefCOCOg test<br>|93.6 90.7 92.5 95.2 92.2 94.3 90.8 86.7 88.2 90.1 82.9 87.6 93.9 89.4 92.3 85.4 75.6 81.6 92.2 87.3 89.6 92.9 87.7 89.3<br><br>|91.8 - 90.0 94.3 - 93.1 87.5 - 87.1<br><br>86.9 - 81.1 91.3 - 87.3 80.6 - 74.5<br><br>87.9 - 85.0<br><br>88.6 - 86.4<br><br><br>|92.6 [36] 90.5 [37] 94.3 [36] 93.1 [37]<br><br>91.4 [36] 88.2 [37]<br><br>88.7 [38] 82.7 [37]<br><br>92.2 [38] 88.9 [37] 83.2 [38] 75.9 [37]<br><br>89.2 [38] 86.1 [37] 89.3 [36] 87.0 [37]<br><br><br>|
|Object Detection COCO val<br><br>|47.1 - -|48.9 46.7 56.7|63.7 [39] 63.1 [40]|
|Semantic Segementation ADE20k Cityscapes Context59 VOC20 COCOStuff<br><br>|54.2 × × 70.4 × × 60.4 × × 92.5 × × 52.5 × ×|54.5 47.8 52.3<br><br>- 61.8 -<br><br>- 63.3 -<br><br>- - -<br><br><br>30.2 49.1 -<br><br>|38.4 [41] 56.4 [42] 42.0 [43] 83.3 [42] 63.6 [41] 60.8 [44] 97.1 [45] -<br>39.6 [46] 45.7 [47]<br>|
|Referring Segementation RefCOCO val<br><br>RefCOCO testA<br><br>RefCOCO testB RefCOCO+ val<br><br><br>RefCOCO+ testA<br><br>RefCOCO+ testB RefCOCOg val RefCOCOg test<br><br><br>|80.7 × × 82.0 × ×<br><br>78.4 × × 76.2 × ×<br>79.6 × × 71.4 × ×<br><br><br>76.5 × ×<br>76.6 × ×<br>|80.0 × 76.6<br><br>81.6 × 79.3<br><br><br>78.1 × 74.3<br><br>76.7 × 64.5<br><br>79.9 × 69.8<br><br><br>72.3 × 61.5<br><br>75.5 × 70.7<br><br>76.3 × 71.2<br><br><br>|80.5 [14] 79.3 [48] 82.6 [14] 81.2 [48] 76.9 [14] 77.8 [48] 74.3 [14] 69.5 [48] 78.9 [14] 75.6 [48] 68.4 [14] 63.0 [48]<br><br>76.3 [14] 71.3 [48]<br>77.0 [14] 72.0 [48]<br>|
|Depth Estimation NYUv2 Cityscapes DDAD<br><br>|90.4 × × 92.7 × × 87.6 × ×|93.6 × ×<br><br>- × ×<br><br>- × ×<br><br><br>|86.8 [49] 98.8 [50] - 92.1 [51] 74.7 [49] 88.2 [50]|
|Human Pose MPII<br><br>|89.1 × ×|× × -<br><br>|89.3 [52] 93.3 [53]|
|Image Classification ImageNet-ReaL|89.3 - -|× × ×|91.1 [54] 91.2 [55]<br><br>|
|Object Counting TallyQA-Simple TallyQA-Complex CountBench<br><br>|85.1 79.0 77.6 74.4 64.0 66.4 88.6 78.4 79.4|× × × × × × × × ×<br><br>|84.9 [56] 86.3 [57] 72.3 [56] 77.1 [57] 83.1 [56] 93.8 [44]|

- Table 2. Comparison of Youtu-VL with general-purpose VLMs (standard architecture), vision-centric VLMs (taskspecific design involved), and classic specialists (single or certain tasks). “×” indicates that this capacity is not supported by the model, “-” means results are not available or reported. Models with “*” use task-specific decoders (e.g., Deform-DETR in VisionLLM v2 [13] beyond the valina VLM architecture). “†” indicates extra task-specific tokens (e.g., multiple <mask> tokens in UFO [15]). Results in gray indicate task-specific fine-tuning on a single dataset, which usually brings higher results. Best results in each setting are marked in bold. “Classic Specialists” indicates models focusing on a few specific tasks using VLMs (e.g., CLIP [19]) or Non-VLMs (e.g., Mask2Former [42], DINO [40], GroundingDINO [58], UniDepthV2 [50], SAM3 [44]). We report the results from the model whose parameters are closest to ours, and general-purpose VLMs indicate their Instruct version. We have compared with additional specialists in Appendix B.1 and Appendix B.2 for reference with their differences.

2020] benchmark, a δ1 of 87.6%, while DepthLLM-3B’s result is 74.7%. Our result is close to the specialist model UniDepth-v2 [Piccinelli et al., 2025] (88.2%). This establishes Youtu-VL as a comprehensive model capable of understanding 3D information alongside 2D semantic tasks.

Human Pose Estimation. Youtu-VL realizes human pose estimation by directly regressing keypoint coordinates within a single generative framework, avoiding the task-specific heads of specialist methods. On MPII [Andriluka et al., 2014], Youtu-VL achieves 89.1% (PCKh@0.5) following the evaluation protocol of the specialist model ViTPose Xu et al. [2022] (93.3%). This result is close to LocLLM [Wang et al., 2024] (89.3% on the specific keypoints). Despite the comparable performance, Youtu-VL is a versatile Vision-Language Model (VLM) that operates without the need for explicit pose estimation priors. Detailed evaluation information can be found in Appendix A.1.

Image Classification. In image classification tasks, Youtu-VL is prompted to generate a phrase identifying the primary object category within an image. On ImageNet-ReaL [Beyer et al., 2020] (ImageNet-1k with corrected ground-truth), Youtu-VL achieves a Top-1 accuracy of 89.3%. This performance approaches the level of leading specialist models, demonstrating the model’s efficiency in object recognition.

Object Counting. This task is evaluated by directly outputting numerical counts of objects without any options. In CountBench [Paiss et al., 2023], Youtu-VL achieves an accuracy of 88.6%, while the result of Qwen3-VL-4B is 78.4% (answer prediction without option). In simple object counting scenarios (TallyQAsimple) [Acharya et al., 2019], Youtu-VL reaches 85.1% accuracy, performing on par with the generalist model Omni-SMoLA (86.3%). For complex counting tasks (TallyQA-complex), Youtu-VL secures a leading score of 74.4%, while the performance of the PaliGemma [Beyer et al., 2024] is 72.3%. This result is close to the specialist model Omni-SMoLA [Wu et al., 2024] (77.1%). Benefiting from extensive training on localization data, Youtu-VL exhibits superior object counting performance.

##### 5.2 General Multimodal Tasks

In this section, we systematically evaluate Youtu-VL across several dimensions, including general visual question answering (VQA), multimodal reasoning, real-world and multi-image understanding, OCR and document understanding, hallucination suppression, and text-centric tasks. For benchmarks that are not officially reported for Qwen3-VL, we additionally run evaluations using its publicly released checkpoints to enable a more comprehensive comparison. In general, our model consistently outperforms or maintains parity with leading models of comparable size across a wide range of multimodal tasks.

General Visual Question Answering. To evaluate the general-purpose VQA capabilities of Youtu-VL, we conduct extensive experiments on a comprehensive suite of benchmarks, including MMBench_V11 (EN/CN) [Liu et al., 2023], MMStar [Chen et al., 2024], MME [Fu et al., 2023], CVBench (2D/3D) [Zhu et al., 2025], ScienceQA [Lu et al., 2022], the SEED-Bench series [Li et al., 2023, 2024,], and MMVet [Yu et al., 2024], as detailed in Table 3. Across these benchmarks, Youtu-VL demonstrates consistently strong performance on general visual understanding tasks that require joint perception and lightweight reasoning. On MMBench, the model achieves scores of 83.9 on the English split and 83.6 on the Chinese split, indicating stable multilingual visual–language alignment. On MMStar and MME, which emphasize broad visual reasoning and robustness across diverse question types, Youtu-VL obtains comparatively high scores, reflecting effective integration of visual perception and knowledge grounding. Furthermore, the model delivers impressive performance on CVBench-2D and CVBench-3D, suggesting solid capability in handling structured visual representations and core vision-related reasoning. Performance on MMVet is comparatively lower, indicating room for improvement on more complex instruction-style VQA scenarios that demand stronger compositional reasoning and response calibration.

Multimodal Reasoning & Math We evaluate the models across a broad spectrum of multimodal reasoning

|Benchmarks|Qwen3-VL 8B (instruct)<br><br>|InternVL-3.5 Qwen3-VL Youtu-VL 4B 4B (instruct) 4B (instruct)|
|---|---|---|
|General VQA MMBenchCN MMBenchEN MMStar MME (/2800)<br><br>CVBench2d<br><br>CVBench3d ScienceQAval SEEDBenchIMG SEEDBench2 MMVet<br><br><br>|84.7 84.5 70.9 – – – – – – –<br><br>|– 83.5 83.6 80.3 83.9 83.9 65.0 69.8 71.1 2272 2309∗ 2384<br><br>– 79.1∗ 80.4<br>– 92.4∗ 93.0<br>– 94.7∗ 97.0<br>– 77.0∗ 76.9<br>– 75.9∗ 74.5<br>– 68.3∗ 64.6<br>|
|Multimodal Reasoning & Math VisuLogic MMMUval MMMU-Pro CMMMUval MathVistamini MathVersemini LogicVista VLMsAreBlind<br><br>|22.5 69.6 55.9 – 77.2 62.1 55.3 74.0|– 19.0 25.7 66.6 67.4 61.1<br><br>– 53.2 43.0<br><br>– 54.6∗ 52.6<br><br><br>77.1 73.7 76.5 45.8 46.8 56.5 41.8 53.2 52.4<br><br>– 71.9 88.9|
|Hallucination HallusionBench CRPEexist CRPErelation POPE|61.1 – – –<br><br>|44.8 57.6 59.1<br><br>– 95.6∗ 96.9 75.0 71.0∗ 72.2 88.9 89.3∗ 86.4|
|OCR-related Understanding AI2Dtest<br><br>InfoVQAval TextVQAval DocVQAval ChartQAtest OCRBench<br><br>SEEDBench2Plus CharXivDQ CharXivRQ|85.7 83.1 – 96.1 89.6 896 – 83.0 46.4<br><br>|82.6 84.1 85.6 78.0 80.3 79.1 77.9 80.8∗ 79.6 92.4 95.3 94.4 86.0 84.6 85.3 822 881 813 69.4 71.5∗ 71.3 71.1 76.2 79.4 39.6 39.7 43.8|
|Multi-image & Real-world BLINK RealWorldQA<br><br>MMERealWorldEN MMERealWorldCN<br><br>|69.1 71.5 – –|58.1 65.8 64.3<br><br>66.3 70.9 74.6 – 63.0∗ 61.5<br><br>59.8 61.3∗ 63.5<br>|
|GUI Agent ScreenSpot Pro OSWorld<br><br>|54.6 33.9<br><br>|- 59.5 59.6 – 26.2 38.8|
|Text-Centric MMLU-Pro MMLU-Redux C-Eval MuSR IFEval DROP(F1) BBH GPQA-Diamond<br><br>|71.6 84.9 – – 83.7 – – –|– 67.1 56.5<br><br>– 81.5 76.8<br><br><br>71.9 76.5 69.1<br><br>– 46.6 58.3<br><br>– 82.3 76.9<br><br>– 85.0 79.3<br><br>– 84.8 71.9<br><br>– 42.9 39.8<br>|

- Table 3. Comparison of Youtu-VL with the state-of-the-art VLMs on various multimodal benchmarks. All numbers are reported or reproduced under the official evaluation protocols. “–” indicates that the corresponding number is not available or not reported. The superscript “*” indicates that the result was not reported and has been reproduced using our evaluation system.

benchmarks, including VLMs Are Blind [Rahmanzadehgervi et al., 2025], VisuLogic [Xu et al., 2025], the MMMU family (MMMU [Yue et al., 2024], MMMU-Pro [Yue et al., 2024], CMMMU [Zhang et al., 2024]), MathVista [Lu et al., 2024], MathVerse [Zhang et al., 2024], and LogicVista [Xiao et al., 2024]. These benchmarks collectively assess image-grounded logic, abstract reasoning, mathematical understanding, and knowledge-intensive multimodal problem-solving.

As shown in Table 3, Youtu-VL achieves strong results on several reasoning-oriented benchmarks. Notably, it achieves scores of 25.7% on VisuLogic and 88.9% on VLMs Are Blind, demonstrating robust visual reasoning capabilities. Together with its competitive performance on LogicVista, these results suggest that Youtu-VL is effective at handling logical relations, rule-based reasoning, and image-dependent inference. On mathrelated benchmarks, Youtu-VL achieves a score of 56.5% on MathVerse and maintains stable performance on MathVista, demonstrating formidable multimodal mathematical reasoning capabilities.

On large-scale academic benchmarks such as MMMU, MMMU-Pro, and CMMMU, which resemble multidisciplinary university-level examinations and emphasize long-tail domain knowledge and long-context language reasoning, Youtu-VL attains mid-to-high score levels (e.g., around 60 on MMMU). While these scores are generally competitive, there remains a modest performance gap on the most knowledge-intensive and context-heavy tasks. This suggests that, beyond its solid multimodal perception and reasoning, future improvements could focus on expanding specialized domain coverage and further strengthening longcontext reasoning stability. Overall, the current results show that Youtu-VL already provides advanced and reliable multimodal reasoning capabilities, particularly in scenarios demanding genuine image-grounded inference, abstract logical judgment, and structured visual–symbolic reasoning.

Hallucination. We evaluate hallucination suppression and instruction alignment using HallusionBench [Guan

- et al., 2023], POPE [Li et al., 2023], and CRPE [Wang et al., 2023, 2024], benchmarks specifically designed to probe adversarial image–question inconsistency as well as fine-grained object existence and relation judgment. As shown in Table 3, Youtu-VL demonstrates a significantly reduced propensity for hallucination. On HallusionBench, which deliberately presents scenarios where questions contradict image content, Youtu-VL achieves 59.1%. This improvement is particularly pronounced in cases where the question suggests the presence of objects clearly absent from the image, indicating that Youtu-VL prioritizes verification against visual

evidence over reliance on linguistic priors. On CRPEexist and CRPErelation, Youtu-VL delivers commendable performance, with residual errors primarily stemming from intrinsically ambiguous boundary cases, such as minute objects or severe occlusions. Overall, Youtu-VL maintains competitive instruction-following capabilities while exhibiting superior resistance to hallucination and stricter image-grounded alignment, favoring cautious responses when visual evidence is insufficient or contradictory. We attribute this robustness to the integration of effective visual supervision, which compels the model to ground its predictions strictly in visual information.

OCR-related Understanding & Document QA. We evaluate fine-grained OCR capabilities and documentlevel question answering on AI2D [Kembhavi et al., 2016], InfoVQA [Mathew et al., 2021], TextVQA [Singh et al., 2019], DocVQA [Mathew et al., 2021], ChartQA [Masry et al., 2022], OCRBench [Liu et al., 2023], SEEDBench2Plus [Li et al., 2024], and the CharXiv description and reasoning subsets [Wang et al., 2024]. These benchmarks cover a spectrum of tasks ranging from dense text recognition to structured document analysis and chart-centric vision–language integration.

As shown in Table 3, Youtu-VL delivers strong performance on OCR-centric benchmarks, especially in settings that require higher-level semantic understanding and reasoning over recognized text. In particular, it achieves a score of about 79% on CharXivDQ and around 44% on CharXivRQ, providing clear gains of several points over comparable 4B-level systems on both descriptive and reasoning-style chart understanding. These tasks demand more than basic text extraction; they require structured interpretation of scientific charts and figures, with joint reasoning over axes, legends, curves, and surrounding textual explanations. Qualitative inspection further suggests that Youtu-VL tends to capture document layout and visual–textual relations in a more globally consistent way, often producing explanations that are both complete and logically organized.

On widely used OCR-heavy benchmarks such as TextVQA, DocVQA, ChartQA, OCRBench, SEEDBench2Plus, AI2D, and InfoVQA, Youtu-VL maintains a generally competitive score profile, with performance that is stable across diverse data sources and task formats. While there remain a few datasets on which its scores are slightly behind the very best reported numbers, the overall results indicate that Youtu-VL is already effective

- at handling information-dense visual contexts and at bridging low-level text recognition with higher-level document and chart intelligence.

Multi-image & Real-world Understanding. We assess multi-image reasoning and real-world scene understanding on BLINK [Fu et al., 2024], RealWorldQA [xAI, 2024], and MMERealWorld (EN and CN) [Zhang

- et al., 2024]. As summarized in Table 3, Youtu-VL delivers competitive performance across these benchmarks. Specifically, Youtu-VL delivers strong performance on real-world and multi-image benchmarks. On RealWorldQA, it reaches a score of 74.6, indicating robust competence in handling high-resolution photographs with small objects, cluttered backgrounds, and complex object–scene interactions. On MMERealWorld, Youtu-VL attains solid scores in both English and Chinese settings, and its relatively higher score on the Chinese split is particularly notable given that the model is trained primarily on English data. This crosslingual robustness suggests that the underlying visual perception is largely language-agnostic, benefiting from effective visual supervision that helps the model focus on fine-grained details while preserving global scene context. On BLINK, which emphasizes fundamental multi-image matching and comparison, Youtu-VL maintains competitive score levels, showing that it can align and compare information across images in a consistent manner. Overall, these results indicate that Youtu-VL provides strong and reliable real-world visual understanding while preserving solid multi-image reasoning abilities, with remaining gaps mainly appearing in more context-heavy, conversational settings rather than in core visual perception.

GUI Agent. We evaluated the GUI agentic performance of Youtu-VL using two primary benchmarks: ScreenSpot Pro [Li et al., 2025] and OSWorld [Xie et al., 2024]. ScreenSpot Pro serves as a static, single-turn grounding benchmark designed to assess the model’s fundamental capacity for instruction parsing and precise coordinate prediction. Its inclusion of diverse operating systems and specialized cross-platform applications ensures a comprehensive measurement of the model’s perceptual generalization. In contrast, OSWorld provides a holistic evaluation of dynamic, closed-loop interaction. By utilizing a live Ubuntu sandbox with authentic software installations, OSWorld requires the agent to execute multi-turn reasoning and adapt to real-time state transitions. This environment accurately reflects "computer use" capabilities by testing the model’s ability to maintain long-horizon goals within a functional operating system. Our empirical results demonstrate that Youtu-VL achieves SoTA performance across both benchmarks, most notably on OSWorld, where it achieved a success rate of 38.8. These metrics suggest that Youtu-VL possesses superior grounding-to-action mapping, effectively translating visual perceptions into valid system sequences rather than merely identifying UI elements. The substantial performance improvement highlights the model’s resilience against “error compounding”—a common failure mode in multi-turn tasks where a single misstep leads to an unrecoverable state. Furthermore, these results demonstrate that Youtu-VL can handle complex, real-world computer use scenarios with greater efficiency and robustness than baseline models.

Text-Centric Tasks The final section of Table 3 presents the evaluation results of Youtu-VL on pure text benchmarks. Given that Youtu-VL is architected with a primary focus on multimodal capabilities—particularly vision-centric tasks—a discernible performance gap exists when compared to state-of-the-art models. Specifically, on IFEval [Zhou et al., 2023], which assesses instruction-following capabilities, Youtu-VL achieves a score of 76.9, demonstrating a competent ability to adhere to complex constraints despite not reaching the ceiling of top-tier language models. On MMLU-Redux [Gema et al., 2024], the model attains a score of 76.8, indicating a robust foundation in general knowledge and reasoning; however, its performance on GPQA-Diamond [Rein et al., 2023] (39.8) highlights remaining challenges in handling expert-level domain reasoning. In summary, while Youtu-VL maintains a functional baseline for textual interaction, there remains significant headroom for improvement in language capabilities, which serves as a key objective for future iterations.

### 6 Conclusion

Discussion. In this work, we present Youtu-VL, a framework that fundamentally reshapes the optimization landscape for Vision-Language Models. By introducing the Vision-Language Unified Autoregressive Supervision (VLUAS) paradigm, we effectively mitigate the text-dominant optimization bias inherent in traditional

architectures. Transitioning from a static “vision-as-input” dependency to a generative “vision-as-target” objective, Youtu-VL incentivizes the model to simultaneously predict fine-grained visual details and highlevel linguistic semantics, thereby bridging the gap between coarse understanding and dense perception. Crucially, our results demonstrate that this unified objective enables a standard VLM architecture to natively execute diverse vision-centric tasks without reliance on task-specific decoders or auxiliary heads. This illustrates that high-fidelity sensory perception can be modeled end-to-end within a single generalist transformer. Our findings mark a critical inflection point in the evolution of VLMs: moving beyond mere cross-modal alignment toward structural unification. We contend that true multimodal intelligence emerges not from stacking specialist modules, but from the synergistic integration of perception and reasoning. Youtu-VL thus serves not merely as a competitive baseline, but as a foundational blueprint for the next generation of Generalist Visual Agents.

Limitation. While Youtu-VL achieves competitive performance across various benchmarks, it faces certain systemic challenges. First, the current visual representation granularity remains a bottleneck for highprecision tasks on low-resolution inputs. Second, the model’s performance in specialized geometry-aware tasks (e.g., depth and pose estimation) is still constrained by its sensitivity to sensor intrinsics and the diversity of training distributions, limiting its zero-shot robustness in out-of-distribution environments. Furthermore, while Youtu-VL excels in general perception, its high-level cognitive abilities, particularly in complex mathematical reasoning and dense knowledge retrieval, require further optimization. Addressing these foundational constraints remains a key objective for future development of the Youtu-VL framework.

### Contributions and Acknowledgments

We would like to express our sincere gratitude to all contributors, including those not listed in the paper, for their invaluable support and efforts. The contributors within each group are listed in no particular order.

Core Contributors Zhixiang Wei Yi Li Zhehan Kan Xinghua Jiang Zuwei Long Shifeng Liu Hongze Shen Wei Liu Xiaoyu Tan Haojia Lin Yubo Zhu Qianyu Li Di Yin Haoyu Cao Weibo Gu Xin Li♠ Yinsong Liu Deqiang Jiang Xing Sun† Yunsheng Wu

Contributors Mingkong Tang Shuangyin Liu Lexiang Tang Haodong Lin Junru Lu Jiarui Qin Lingfeng Qiao Ruizhi Qiao Bo Ke Jianfeng He Ke Li Yangning Li Yunhang Shen Mengdan Zhang Peixian Chen Kun Yin Bing Liu Yunfei Wu Huang Chen Zhongpeng Cai Xiaotian Li Youtu-VL Team

### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang,

♠Project Lead †Corresponding author: winfredsun@tencent.com

Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

- [2] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.
- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv:2304.08485, 2023.
- [4] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023. URL https://arxiv.org/abs/2301.08243.
- [5] Junru Lu, Jiarui Qin, Lingfeng Qiao, Yinghui Li, Xinyi Dai, Bo Ke, Jianfeng He, Ruizhi Qiao, Di Yin, Xing Sun, Yunsheng Wu, Yinsong Liu, Shuangyin Liu, Mingkong Tang, Haodong Lin, Jiayi Kuang, Fanxu Meng, Xiaojuan Tang, Yunjia Xi, Junjie Huang, Haotong Yang, Zhenyi Shen, Yangning Li, Qianwen Zhang, Yifei Yu, Siyu An, Junnan Dong, Qiufeng Wang, Jie Wang, Keyu Chen, Wei Wen, Taian Guo, Zhifeng Shen, Daohai Yu, Jiahao Li, Ke Li, Zongyi Li, and Xiaoyu Tan. Youtu-llm: Unlocking the native agentic potential for lightweight large language models, 2026. URL https: //arxiv.org/abs/2512.24618.
- [6] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025. URL https://arxiv.org/abs/2502.14786.
- [7] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/abs/2104.09864.
- [8] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness, 2022. URL https://arxiv.org/abs/2205.14135.
- [9] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.
- [10] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. Dinov3, 2025. URL https://arxiv.org/abs/2508.10104.

- [11] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization, 2025. URL https://arxiv.org/abs/2412. 02692.
- [12] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
- [13] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Zhe Chen, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. Advances in Neural Information Processing Systems, 37: 69925–69975, 2024.
- [14] Ye Liu, Zongyang Ma, Junfu Pu, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Unipixel: Unified object referring and segmentation for pixel-level visual reasoning. arXiv preprint arXiv:2509.18094, 2025.
- [15] Hao Tang, Chenwei Xie, Haiyang Wang, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Ufo: A unified approach to fine-grained visual perception via open-ended language interface. arXiv preprint arXiv:2503.01342, 2025.
- [16] Yi Li, Hualiang Wang, Xinpeng Ding, Haonan Wang, and Xiaomeng Li. Token activation map to visually explain multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 48–58, October 2025.
- [17] Philipp Krähenbühl and Vladlen Koltun. Efficient inference in fully connected crfs with gaussian edge potentials. Advances in neural information processing systems, 24, 2011.
- [18] Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769, 2016.
- [19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [20] Zhixiang Wei, Guangting Wang, Xiaoxiao Ma, Ke Mei, Huaian Chen, Yi Jin, and Fengyun Rao. Hq-clip: Leveraging large vision-language models to create high-quality image-text datasets and clip models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22447–22456, 2025.
- [21] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023.
- [22] Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, et al. Meta clip 2: A worldwide scaling recipe. arXiv preprint arXiv:2507.22062, 2025.
- [23] Randall Balestriero, Nicolas Ballas, Mike Rabbat, and Yann LeCun. Gaussian embeddings: How jepas secretly learn your data density, 2025. URL https://arxiv.org/abs/2510.05949.
- [24] Haojia Lin, Xiaoyu Tan, Yulei Qin, Zihan Xu, Yuchen Shi, Zongyi Li, Gang Li, Shaofei Cai, Siqi Cai, Chaoyou Fu, et al. Cuarewardbench: A benchmark for evaluating reward models on computer-using agent. arXiv preprint arXiv:2510.18596, 2025.
- [25] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37: 52040–52094, 2024.

- [26] Yuchen Shi, Yuzheng Cai, Siqi Cai, Zihan Xu, Lichao Chen, Yulei Qin, Zhijian Zhou, Xiang Fei, Chaofan Qiu, Xiaoyu Tan, et al. Youtu-agent: Scaling agent productivity with automated generation and hybrid policy optimization. arXiv preprint arXiv:2512.24615, 2025.
- [27] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437.
- [28] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [29] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2.5 technical report. arXiv:2412.15115, 2024.
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations.
- [31] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [32] Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025.
- [33] Bo Li, Peiyuan Zhang, Kaichen Zhang, Fanyi Pu, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Accelerating the development of large multimoal models, March 2024. URL https://github.com/EvolvingLMMs-Lab/lmms-eval.
- [34] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multimodality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198– 11201, 2024.

- [35] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. URL https://arxiv.org/abs/2407.12772.
- [36] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15325–15336, 2023.
- [37] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.
- [38] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. arXiv preprint arXiv:2305.11172, 2023.
- [39] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv:2208.10442, 2022.
- [40] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In The Eleventh International Conference on Learning Representations.
- [41] Congpei Qiu, Yanhao Wu, Wei Ke, Xiuxiu Bai, and Tong Zhang. Refining CLIP’s spatial awareness: A visual-centric perspective. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=38No4B8sx6.
- [42] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.
- [43] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Proxyclip: Proxy attention improves clip for open-vocabulary segmentation. In European Conference on Computer Vision, pages 70–88. Springer, 2024.
- [44] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [45] Yong Liu, Sule Bai, Guanbin Li, Yitong Wang, and Yansong Tang. Open-vocabulary segmentation with semantic-assisted calibration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3491–3500, 2024.
- [46] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European conference on computer vision, pages 696–712. Springer, 2022.
- [47] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for openvocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2945–2954, 2023.
- [48] Shuyi Ouyang, Ziwei Niu, Hongyi Wang, Yen-Wei Chen, and Lanfen Lin. Region-aware anchoring mechanism for efficient referring visual grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24192–24202, 2025.
- [49] Zhipeng Cai, Ching-Feng Yeh, Hu Xu, Zhuang Liu, Gregory Meyer, Xinjie Lei, Changsheng Zhao, Shang-Wen Li, Vikas Chandra, and Yangyang Shi. Depthlm: Metric depth from vision language models. arXiv preprint arXiv:2509.25413, 2025.

- [50] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025.
- [51] Pardis Taghavi, Reza Langari, and Gaurav Pandey. Swinmtl: A shared architecture for simultaneous depth estimation and semantic segmentation from monocular camera images. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4957–4964. IEEE, 2024.
- [52] Dongkai Wang, Shiyu Xuan, and Shiliang Zhang. Locllm: Exploiting generalizable human keypoint localization via large language model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 614–623, 2024.
- [53] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35:38571–38584, 2022.
- [54] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171, 2024.
- [55] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.
- [56] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.
- [57] Jialin Wu, Xia Hu, Yaqing Wang, Bo Pang, and Radu Soricut. Omni-smola: Boosting generalist multimodal models with soft mixture of low-rank experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14205–14215, 2024.
- [58] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chun yue Li, Jianwei Yang, Hang Su, Jun-Juan Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv:2303.05499, 2023.
- [59] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.
- [60] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [61] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.
- [62] Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, and Liwei Wang. Git: Towards generalist vision transformer through universal language interface. In European Conference on Computer Vision, pages 55–73. Springer, 2024.
- [63] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.
- [64] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1209–1218, 2018.
- [65] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.

- [66] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2485–2494, 2020.
- [67] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, pages 3686–3693, 2014.
- [68] Lucas Beyer, Olivier J Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. Are we done with imagenet? arXiv preprint arXiv:2006.07159, 2020.
- [69] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3170–3180, 2023.
- [70] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.
- [71] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [72] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024.
- [73] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.
- [74] Nannan Zhu, Yonghao Dong, Teng Wang, Xueqian Li, Shengjun Deng, Yijia Wang, Zheng Hong, Tiantian Geng, Guo Niu, Hanyan Huang, et al. Cvbench: Benchmarking cross-video synergies for complex multimodal reasoning. arXiv preprint arXiv:2508.19542, 2025.
- [75] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.
- [76] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv:2307.16125, 2023.
- [77] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seedbench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.
- [78] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024.
- [79] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.
- [80] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind: Failing to translate detailed visual features into words, 2025. URL https://arxiv.org/abs/2407.06581.
- [81] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models, 2025. URL https://arxiv.org/abs/2504.15279.

- [82] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [83] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [84] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, et al. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2401.11944, 2024.
- [85] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.
- [86] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.
- [87] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.
- [88] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models, 2023.
- [89] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [90] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv preprint arXiv:2308.01907, 2023.
- [91] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024.
- [92] Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. ArXiv, abs/1603.07396, 2016.
- [93] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2582–2591, 2021.
- [94] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.
- [95] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.
- [96] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv:2203.10244, 2022.
- [97] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. arXiv:2305.07895, 2023.

- [98] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024.
- [99] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.
- [100] xAI. Realworldqa: A benchmark for real-world spatial understanding. https://huggingface.co/ datasets/xai-org/RealworldQA, 2024. Accessed: 2025-04-26.
- [101] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024.
- [102] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 8778–8786, 2025.
- [103] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. CoRR, abs/2311.07911, 2023.
- [104] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? CoRR, abs/2406.04127, 2024.
- [105] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level Google-proof Q&A benchmark. CoRR, abs/2311.12022, 2023.
- [106] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
- [107] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014.
- [108] Yuan Liu, Haodong Duan, Bo Li Yuanhan Zhang, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023.
- [109] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, ChengLin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/ s11432-024-4235-6. URL http://dx.doi.org/10.1007/s11432-024-4235-6.
- [110] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv:2311.16502, 2023.
- [111] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. In Advances in Neural Information Processing Systems, volume 34, pages 12077–12090, 2021.
- [112] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. 2022.

- [113] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-toimage diffusion models for visual perception. ICCV, 2023.
- [114] Yi Li, Hualiang Wang, Yiqun Duan, Jiheng Zhang, and Xiaomeng Li. A closer look at the explainability of contrastive language-image pre-training. Pattern Recognition, 162:111409, 2025. ISSN 0031-

3203. doi: https://doi.org/10.1016/j.patcog.2025.111409. URL https://www.sciencedirect.com/ science/article/pii/S003132032500069X.

- [115] Chanyoung Kim, Dayun Ju, Woojung Han, Ming-Hsuan Yang, and Seong Jae Hwang. Distilling spectral graph for object-context aware open-vocabulary semantic segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15033–15042, 2025.
- [116] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15116–15127, 2023.
- [117] David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, and Amir Zamir. 4m: Massively multimodal masked modeling. Advances in Neural Information Processing Systems, 36:58363–58408, 2023.
- [118] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19175–19186, 2023.
- [119] Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser-Nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, and Amjad Almahairi. Jack of all tasks master of many: Designing general-purpose coarse-to-fine vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14076–14088, 2024.
- [120] Chang-Bin Zhang, Yujie Zhong, and Kai Han. Mr. detr: Instructive multi-route training for detection transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9933–9943, 2025.
- [121] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790, 2021.
- [122] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.
- [123] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In European Conference on Computer Vision, pages 405–422. Springer, 2024.

### A Evaluation Details

##### A.1 Vision-Centric Benchmarks and Prompts

In order to ensure better reproducibility, we have provided the evaluation details for the benchmarks of vision-centric tasks, along with specific prompts. All results were tested using a zero-shot approach, without utilizing few-shot examples, without the thinking mode (e.g., CoT).

###### A.1.1 Visual Grounding

RefCOCO series [59]. For visual grounding, we require the model to directly localize the queried object by predicting its bounding box. The specific prompt is as follows:

|<image> Question: Please provide the bounding box coordinate of the region this sentence describes: the person bottom left Answer: <box><x_155><y_154><x_221><y_206></box>|
|---|

Visual Grounding for the Open World. Except for specific benchmarks, Youtu-VL also supports various open-scene prompts and objectives. We support some flexible prompts and Chinese prompts. For visual grounding, we provide some prompt examples for reference, and their Chinese versions are also suggested. {keyword} indicates the target needed to ground, which is usually a word or phrase.

|Can you return the bounding box for "{keyword}" in this image? Output the bounding box for "{keyword}" in this image. Perform object grounding for "{keyword}" with bounding box. Can you return the bounding box for {keyword} in this image? Output the bounding box for {keyword} in this image. Perform object grounding for {keyword} with bounding box. Can you return the bounding box for {keyword} in this image? Output the bounding box for {keyword} in this image. Perform object grounding for {keyword} with bounding box. Please provide the bounding box coordinate of the region this sentence describes:<br><br>{keyword}. Please get the bounding box coordinate for the target: {keyword}. Get bounding box for {keyword}. Identify the bounding box for "{keyword}" in this image. Locate and return the bounding box coordinates of {keyword}. Find the bounding box around the object described as "{keyword}". Provide the coordinates of the bounding box for {keyword}. Show me the bounding box for the object labeled "{keyword}". Draw the bounding box for {keyword} and provide its coordinates. Extract the bounding box for "{keyword}" from the image. Where is the bounding box for {keyword} located in this image? Mark the bounding box surrounding {keyword} and return its position. Return the coordinates of the bounding box that encloses {keyword}. Give me the bounding box details for the item described as "{keyword}". Please highlight the bounding box of {keyword} and share the coordinates. Get the spatial bounding box for the object "{keyword}". Locate {keyword} in the image and provide its bounding box. Identify the region bounding box corresponding to {keyword}. Provide bounding box information for the entity named "{keyword}". Find and output the bounding box for {keyword} in this picture. Output the spatial coordinates of the bounding box for {keyword}. Can you specify the bounding box around {keyword} in the image? Please return the bounding box of the object referred to as "{keyword}". Find the object "{keyword}" and give me its bounding box coordinates.|
|---|

###### A.1.2 Object Detection

COCO val [61]. For evaluation on COCO, we prompt the model to produce detections using the COCO label space, without specifying any particular category in advance. For detections generated via a naïve autoregressive output format, we parse each prediction into an object category and a bounding box. For each predicted box, when computing mAP, we set its confidence score to be the box area. Because our model is trained on higher-resolution images, our main results are obtained by aggregating predictions from multiple upscaled versions of each COCO image, specifically at 1×, 2×, 3×, and 4×. Bounding boxes predicted at multiple scales are merged using non-maximum suppression (NMS) with an IoU threshold of 0.7. The specific prompt for each image is as follows:

|<image> Question: Detect all objects in the provided image. Answer: <ref>spoon</ref><box><x_87><y_103><x_929><y_934></box><ref>bowl</ref> <box><x_85><y_1409><x_887><y_2094></box><box><x_84><y_97><x_2073><y_1466></box><br><br>...<ref>dining table</ref><box><x_89><y_105><x_2069><y_2082></box>|
|---|

Object Detection for the Open World. The difference between object detection and visual grounding lies in the fact that grounding supports the localization of a single object, while detection supports multiple objects of a single category, as well as multiple objects of multiple categories. We perform detection for given categories, including combinations of single/multiple categories that may or may not be present in the image. {keyword} is a series of categories, such as “dog” or “dog, cat, person, tree”. The specific prompts are as follows, and we support their Chinese versions:

|Please detect {keyword} in the image. Please detect “{keyword}” in the image. Please detect: {keyword} Detect and highlight {keyword} in the picture. Can you detect {keyword} in this image? Can you detect “{keyword}”? Detect objects from: {keyword}. Detect all {keyword} visible in the photo. Detect all “{keyword}” in the photo. Detect all objects in these classes: {keyword} Perform detection for “{keyword}” Run object detection for {keyword} in this picture. Apply detection: {keyword}. Use image detection to locate {keyword}. Carry out detection for all {keyword}. Carry out detection for “{keyword}”. Perform detection for “{keyword}”. Apply detection: {keyword}. Run detection for any {keyword} visible. Execute detection for classes: {keyword}. Detect {keyword} in this image. Please perform detection on {keyword} shown. Run detection algorithms to find {keyword}. Detect any visible {keyword} in this photo. Detect the presence of {keyword} in the image. Find and detect {keyword} within the picture. Detect all objects labeled as {keyword}. Perform a detailed detection for {keyword}. Detect {keyword} and outline them in the image. Locate and detect “{keyword}” precisely. Run a detection pass for {keyword} here.|
|---|

If no category keywords are entered, we also support detecting the main categories in the image. This method can be referred to as ’detect anything’ to some extent. However, there may be issues like missing less prominent categories. We will continue to optimize this capability to enhance the generalization ability in open-world scenarios. Specific prompts include, but are not limited to:

|Detect all object categories present in this picture. Run object detection and output all found objects. Detect objects across the entire image area. Please detect every object and report the results. Run a full object detection to capture all items. Detect all objects in the frame and list them. Apply object detection and return all object labels. Detect and annotate all objects you can see. Identify all objects present and provide detection data. Run object detection to find every object in view. Detect all objects in the picture and summarize them. Perform object detection over the entire image. Identify all items in this image using object detection. Detect all visible objects and provide their labels. Run object detection to detect all objects shown. Detect all objects within this image and list categories. Please perform object detection and list all findings. Detect all objects in this image without specifying extra constraints. Detect all. Detect it. Detect all objects. Run object detection. Return all object categories found. Detect all objects with locations. Detect objects. Show all detected objects. Run object detection. Return all detections. Enumerate all objects detected. Detect all object categories present. Output all detected objects. Perform detection. Detect all objects shown.|
|---|

###### A.1.3 Image Classification

ImageNet-ReaL [68]. During evaluation, we instruct the model to output a single word or short phrase describing the category of the dominant object in the image. A prediction is counted as correct if it matches the ground-truth label. The specific prompt is as follows:

|<image> Question: What is the category of the primary object in this image? Answer the question with a single word or phrase. Answer: puck|
|---|

###### A.1.4 Object Counting

TallyQA [70] & CountBench [69]. For the counting evaluation, we require the model to output a single numeric value corresponding to the number of queried objects. A prediction is considered correct if and only if this number exactly matches the ground-truth count. For evaluation, the input image is upsampled by a factor of 2 or 2.5. The specific prompt is as follows:

|<image> Question: How many aum symbols are there in the image? Output the final answer as a single number. Answer: 9|
|---|

###### A.1.5 Semantic Segmentation

ADE20K [63]. In our testing of the ADE20K dataset (test), we employed an open vocabulary method. The prompts were constructed using the format "Segment: class1, class2, class3, ..." where the class names are all lowercase English terms representing the various categories. To assist the model in querying each class individually without fitting to specific instructions, we randomly shuffled the class names for each image. Given that the patch size undergoes significant downsampling after token merging (to 32), and considering the small resolution of images in this benchmark, we scaled each image up by four times its original resolution for testing. The results for the detected categories are formatted as <ref>class1</ref><ref>class2</ref><ref>class3</ref>, making them easier to parse. The pixel-level outputs are listed at the end as <mask>RLE string</mask>, with the numbering corresponding to the order of the class names as they appear. “RLE string” indicates the pixel labels compressed by Run-Length Encoding. The length of the 1D mask string is the same as the pixel number. Users need to reshape it to the 2D raw image size for further usage. For evaluation without background, a softmax with temperature 0.2 and DenseCRF [17] is applied to the logits to refine the pixel-level predictions. This operation is applied to all tasks related to semantic segmentation during evaluation, and is optional in real practice. Here’s a specific example:

|<image> Question: Segment: tree, towel, tank, armchair, refrigerator, countertop, blanket, railing, hood, bathtub, radiator, chandelier, canopy, arcade machine, trade name, case, wardrobe, oven, sidewalk, toilet, chair, wall, door, land, fireplace, windowpane, plant, waterfall, swimming pool, computer, animal, apparel, bannister, poster, flag, step, chest of drawers, box, book, light, tower, ceiling, bulletin board, palm, bicycle, barrel, earth, buffet, person, bus, stairs, stove, stool, basket, washer, rug, monitor, stairway, cushion, pot, minibike, awning, desk, grass, clock, runway, sea, bookcase, sky, food, hovel, seat, van, road, fountain, streetlight, coffee table, painting, ball, car, bed , cradle, airplane, kitchen island, house, ottoman, column, rock, blind, sconce, fan, escalator, lake, boat, booth, stage, sofa, base, swivel chair, mirror, pole, shower, conveyer belt, table, bench, tray, glass, lamp, dishwasher, crt screen, plaything, pool table, water, screen door, truck, bridge, pillow, building, ship, signboard, traffic light, microwave, screen, path, vase, mountain, pier, field, bottle, sculpture, floor, hill, river, sand, flower, sink, grandstand, shelf, dirt track, ashcan, skyscraper, counter, cabinet, television receiver, fence, bar, plate, tent, curtain, bag, without the background class. Answer: The target categories include <ref>tree</ref><ref>sidewalk</ref><ref> ... </ref><ref>pot</ref><ref>signboard</ref>, numbered sequentially starting from 0, without the background class.<mask>RLE string</mask>|
|---|

Note that we evaluated ADE20k using a randomly sampled prompt from “Semantic Segmentation for the Open World” to demonstrate prompt flexibility, whereas other benchmarks used a fixed prompt for simplicity.

Cityscapes [106]. The evaluation of Cityscapes (val) is consistent with the output format of ADE20K; the only difference in the input prompt is the category name (shuffle the names, too). The distinction lies in the resolution, which is multiplied by 2 instead of 4. To reduce computation, we performed a 2x2 non-overlapping crop of the images for testing. The specific prompts are as follows:

|<image> Question: Segment: train, motorcycle, vegetation, person, wall, terrain, pole, sky, traffic light, fence, bicycle, road, traffic sign, rider, building, truck, sidewalk, bus, car, without the background class. Answer: The target categories include <ref>road</ref><ref>sidewalk</ref><ref>car</ref>, numbered sequentially starting from 0.<mask>RLE string</mask>|
|---|

Context59 [107]. The Pascal context (val) benchmark follows the aforementioned prompt and output protocols, with the input resolution upscaled to 3× its original size. Context59 comprises 59 foreground categories, excluding background classes. To accommodate scenarios requiring background identification, the model supports an optional configuration where the phrase "without the background class" is omitted from the prompt. In this mode, a background channel (represented by <OTHERS>) is integrated into the post-processing pipeline: logits are scaled by a factor of 0.25 before a sigmoid activation, followed by the addition of a constant background score (0.5) to facilitate the argmax operation. However, for standard evaluation on Context59, we adhere to the default softmax-based refinement with a temperature of 0.2 and DenseCRF, consistent with our protocol for other semantic segmentation tasks. A representative prompt is illustrated below:

|<image> Question: Segment: window, ceiling, dog, person, ground, keyboard, cloth, bus, bag, boat, sheep, wall, bicycle, snow, platform, grass, flower, computer, floor, truck, bottle, light, car, curtain, sign, bird, pottedplant, tree, cat, table, door, bed, food, train, sidewalk, bench, bedclothes, sofa, mountain, rock, water, building, aeroplane, plate, track, cabinet, horse, chair, cup, fence, road, tvmonitor, motorbike, sky, book, mouse, cow, wood, shelves, without the background class. Answer: The target categories include <ref>ground</ref><ref>grass</ref> ... <ref>rock</ref><ref>water</ref><ref>sky</ref>, numbered sequentially starting from 0.<mask>RLE string</mask>|
|---|

VOC20. VOC20, or PASCAL VOC, contains 20 foreground classes without background classes. The test resolution multiples, prompt format, and output format are consistent with Context59. A specific example is as follows:

|<image> Question: Segment: train, sofa, sheep, horse, bird, cat, car, cow, boat, aeroplane, diningtable, pottedplant, chair, dog, bottle, person, motorbike, bicycle, tvmonitor, bus, without the background class. Answer: The target categories include <ref>car</ref><ref>boat</ref><ref>person</ref> <ref>bus</ref>, numbered sequentially starting from 0.<mask>RLE string</mask>|
|---|

COCOStuff [64]. CocoStuff uses the same testing protocol as ADE20K, except that the resize scale is 3. The rest of the prompt format (including label shuffling) and output format are consistent. A specific example is as follows:

|<image> Question: Segment: wall-tile, ground-other, surfboard, moss, fire hydrant, towel, backpack, couch, floor-wood, mirror-stuff, rock, handbag, door-stuff, paper, salad, gravel, door, plant-other, hill, snowboard, eye glasses, snow, person, tennis racket, wall-concrete, playingfield, plastic, wall-wood, bird, banana, carrot, bed, sandwich, furniture-other, knife, rug, plate, vase, elephant, clouds, kite, tv, cell phone, fork, apple, straw, stone, frisbee, suitcase, fence, cake, clock, train, grass, wood, donut, curtain, cow, cloth, floor-other, toaster, tree, giraffe, sports ball, shelf, flower, building-other, potted plant, carpet, fruit, window-blind, spoon, railroad, pillow, traffic light, bush, desk-stuff, sheep, bear, railing, bottle, hat, blender, baseball glove, sea, sky-other, hair drier, microwave, parking meter, zebra, tent, mouse, skis, counter, desk, banner, tie, oven, branch, scissors, structural-other, dirt, keyboard, fog, cupboard, wall-panel, wall-brick, floor-stone, food-other, wall-other, dining table, napkin, stop sign, cat, net, mirror, leaves, bus, house, bowl, mud, stairs, truck, laptop, table, textile-other, clothes, sand, window, vegetable, pizza, floor-tile, wine glass, waterdrops, river, road, floor-marble, cardboard, refrigerator, window-other, wall-stone, platform, cup, mountain, street sign, shoe, umbrella, car, book, chair, skyscraper, cabinet, skateboard, ceiling-other, blanket, toothbrush, bench, orange, light, hot dog, bridge, pavement, bicycle, solid-other, ceiling-tile, teddy bear, water-other, motorcycle, broccoli, horse, cage, mat, baseball bat, dog, roof, boat, metal, sink, hair brush, remote, toilet, airplane, without the background class. Answer: The target categories include <ref>rock</ref><ref>plant-other</ref> <ref>bird</ref><ref>grass</ref><ref>bush</ref><ref>bear</ref><ref>dirt</ref> <ref>water-other</ref>, numbered sequentially starting from 0.<mask>RLE string</mask>|
|---|

Semantic Segmentation for the Open World. Open set segmentation and specific benchmarks differ in the output content and interpretation results. Open sets support single objects, multiple objects, and a collection of labels. If there is no improvement without the background, the model will output a label of <OTHERS>, which will activate the sigmoid + threshold interpretation method. Specifically, the logits are divided by 4 and then passed through sigmoid, with the background class threshold defaulting to 0.5. The corresponding accuracy for open sets is generally not high, and fixed thresholds can sometimes lead to inaccuracies in the background class. We can achieve more accurate open-set segmentation results using the detection-then-segmentation type. The specific approach is to first invoke detection, then draw each box on the image, apply a padding of 1.2, resize, and then segment the foreground and background. In contrast, directly outputting semantic segmentation results is faster but relatively lower in quality, though it can additionally support stuff classes. The specific prompts for direct semantic segmentation are as follows (the Chinese version is also supported):

|Please perform segmentation for the following classes: {keyword}. Can you identify and segment these objects: {keyword}? I’m looking for {keyword}. Please segment them in the image. Segment the image based on these keywords: {keyword}. Mark the regions corresponding to: {keyword} in the image. Identify the boundaries of these objects: {keyword}. Please label and segment the following categories: {keyword}. Show me the segmentation mask for: {keyword}. Create segmentation maps for these items: {keyword}. Locate and segment the following objects: {keyword}. Segment the image to highlight: {keyword}. Distinguish and segment these classes: {keyword}. Please provide segmentation for objects: {keyword}. Identify and separate the following elements: {keyword}. Generate segmentation for the specified classes: {keyword}. Semantic segmentation for {keyword}. Perform semantic segmentation for {keyword}. Run seamntic segmentation: {keyword}. Segment: {keyword}. Semantic segmentation: {keyword}. Find {keyword} and segment. Segment these: {keyword}. Segment {keyword}. Segment {keyword}. Segment and label {keyword}. Please segment {keyword}. Could you please segment objects classified as: {keyword}? Provide a detailed segmentation of the following: {keyword}. Perform pixel-level segmentation for: {keyword}. Mark and segment the specified classes: {keyword}. Generate a segmented output focusing on: {keyword}. Show segmentation results highlighting: {keyword}. Can you segment and annotate the following keywords: {keyword}? Provide segmentation masks specifically for: {keyword}. Please segment any visible {keyword} in the image. Produce a segmentation mask isolating {keyword}. Segment and highlight the regions occupied by {keyword}. Perform semantic segmentation targeting: {keyword}. Segment and classify the following entities: {keyword}. Mark the presence and segment the {keyword} in the image. Generate masks that segment the {keyword} clearly. Create a detailed segmentation for the objects: {keyword}. Generate segmentation annotations for: {keyword}. Provide a detailed semantic segmentation of the following categories: {keyword}. Perform pixel-level semantic segmentation for: {keyword}. Create precise semantic segmentation boundaries around these classes: {keyword}. Generate a semantic segmentation output focusing on: {keyword}. Segment {keyword} in the image. Show semantic segmentation for {keyword}. Find and segment {keyword}. Please segment {keyword}. Segment {keyword} categories. Show {keyword} segmentation. Classify and segment {keyword}." }. Generate segmentation annotations for: {keyword}.|
|---|

{keyword} indicates some category names such as “dog, cat, tree, soft”. We also support “segment anything” type without specific keyword assignment. It is important to note that this model does not necessarily segment all elements in detail, but rather segments some significant main objects. Since background classes will also be output, sigmoid processing will be applied, with a background threshold of 0.5. The combined version of detection and semantic segmentation can address issues with inaccurate backgrounds. Suggested prompts include:

|Please segment all objects present in the image. I want to segment everything in the image. Segment every object visible in the photo. Segment all objects in this image. Segment all visible elements in the photo. Segment all objects in the image. Identify and segment all items. Segment everything visible. Find and segment all objects. Label all objects in the image. Segment all elements. Perform semantic segmentation for this image. Do semantic segmentation for this image. Segment all. Conduct semantic segmentation for all. Perform semantic segmentation for this image. Do semantic segmentation for this image. Segment all. Segment this image. Segment it. Segment everything. Segment all Semantic segmentation for all. Semantic segmentation. Run semantic segmentation Segment this image Conduct semantic segmentation for all. Apply semantic segmentation to the image. Run semantic segmentation on this image. Execute semantic segmentation for all objects. Carry out semantic segmentation for everything. Perform full semantic segmentation. Complete semantic segmentation for all items.|
|---|

###### A.1.6 Referring Expression Segmentation

RefCOCO series [59] (polygon). The basic version of referring expression segmentation is implemented using polygons. This mode is a simple version, but not evaluated.

|<image> Question: Can you segment “bowl behind the others can only see par” in this image? Answer: The segmentation result is <ins><poly><x_468><y_3><x_471><y_46><x_473><y_83><x_521> <y_107><x_581><y_117><x_640><y_106><x_640><y_105><x_640><y_1></poly></ins>.|
|---|

This is the basic referring segmentation prompt. Note that <ins> represents an object, and an object may have multiple <poly>, indicating different parts.

RefCOCO series [59] (grounding-then-segmentation). Due to the precision limitations from the number of points, we recommend first using grounding and then semantic segmentation. Specifically, after grounding outputs bounding boxes on the original image, we draw boxes on the image with random colors and crop the image using a padding ratio of 1.2 (an extra 0.2). We then resize the shorter side to 1280 for semantic segmentation to distinguish between foreground and background classes (supporting lower resolutions for acceleration). Similar to semantic segmentation, DenseCRF is also utilized as a post-processing step (not used with polygons). “RLE string” in the output indicates the pixel labels compressed by Run-Length Encoding in the string format. The evaluation metric is cIoU. Examples based on Polygon and semantic segmentation are shown below:

In the testing, we actually used grounding prompts, then drew boxes on the image and performed a 1.2 padding crop. After setting the short side to 1280, we executed the following semantic segmentation commands, and then filled the segmentation results back into the original image for testing.

|<image> Question: Can you segment “Segment the code ” in this image? Answer: The segmentation result is <ins><poly><x_468><y_3><x_471><y_46><x_473><y_83><x_521> <y_107><x_581><y_117><x_640><y_106><x_640><y_105><x_640><y_1></poly></ins>.|
|---|

This is the basic referring segmentation prompt. Note that <ins> represents an object, and an object may have multiple <poly>, indicating different parts.

|<image> Question: Please provide the bounding box coordinate of the region this sentence describes: the person bottom left Answer: <box><x_155><y_154><x_221><y_206></box>|
|---|

|<image> Question: Please provide the bounding box coordinate of the region this sentence describes: bowl behind the others can only see part Answer: <box><x_54><y_0><x_361><y_141></box> <cropped image> #Draw the box, padding, and resize. Question: Segment the core target. Answer: The results are 0 for <ref><BG></ref> and 1 for <ref><FG></ref>.<mask>RLE string</mask>|
|---|

Note that the mask result is from the cropped image. We fill it back to the raw image for evaluation via cIoU.

Referring Expression Segmentation for the Open World. Referring segmentation and points are the keywords of this task, which will activate the model to output polygon segmentation results for a single

target. The approach based on grounding + semantic segmentation can refer to the flexible prompts of grounding.

|Can you segment "{keyword}" via points? Can you segment "{keyword}" in the manner of referring segmentation? Referring segment for "{keyword}". Referring segment for {keyword}. Referring expression segmentation for {keyword}.’ Can you segment {keyword} via points? Can you segment {keyword} in the manner of referring segmentation? Outline {keyword} via points. Outline {keyword} via the polygon or points. Segment {keyword} via points. Outline "{keyword}" via points. Use points to segment {keyword}. Use points to segment "{keyword}". Use point to segment {keyword}. Use point to segment "{keyword}". Please {keyword}. Use points to segment "{keyword}". Outline "{keyword}" via the polygon or points. Segment "{keyword}" via points. Perform referring segmentation for the keywords: {keyword}. Perform referring segmentation for "{keyword}". Please segment "{keyword}" in the image using polygon or points. Could you outline "{keyword}" with points? Draw the boundary of "{keyword}" via points. Generate a referring segmentation mask for "{keyword}". Segment {keyword} based on referring expression. Segment "{keyword}" using points outlining. Please perform referring expression segmentation for {keyword}. Use polygon or points to segment "{keyword}". Draw a polygon or points to segment "{keyword}" in the image. Referring segmentation for the object "{keyword}". Segment {keyword} in this picture by outlining with polygon or points. Please extract the polygon or pointsal region corresponding to "{keyword}|
|---|

###### A.1.7 Depth Estimation

NYUv2 [65]. The depth estimation test has four key points: (1) The category IDs are preset from <custom_1> to <custom_1000>, eliminating the need to predict category names, which increases speed. (2) Instead of using DenseCRF, we first upsample by a factor of two, followed by argmax, and then resize to the original image size. Appropriate resizing before argmax improves results. (3) Real depth needs to be dequantized. For NYUv2, we use uniform linear quantization, mapping real depths from 0 to 10 meters to 1 to 1000. Invalid depths are set to 0 and ignored during training. During testing, we dequantize to the actual depths and exclude invalid depths. (4) The prompt must precede the image so that the model can learn the potential quantization methods. The output format is a string of depth names corresponding to the pixel sizes after argmax and resizing. For NYUv2, since the images are relatively small, we performed a threefold resize based on the original resolution. The example is given below:

|Question: Please estimate the depth of this image from the NYUv2 dataset. <image> Answer: This is the <depth>.<mask>RLE string</mask>|
|---|

Cityscapes [106]. The testing criteria for Cityscapes are the same as for NYUv2. The difference lies in quantizing the effective depth from 0-80m to 1-1000, and since the resolution is sufficient, no resizing was done. Besides, we set invalid regions from the left and bottom pars following previous works. Specific examples are as follows:

|Question: Please estimate the depth of this image from the Cityscapes dataset. <image> Answer: This is the <depth>.<mask>RLE string</mask>|
|---|

DDAD [66]. The testing criteria for DDAD are the same as for NYUv2. The difference lies in quantizing the effective depth from 0.05-120m to 1-1000, and since the resolution is sufficient, no resizing was done. Specific examples are as follows:

|Question: Please estimate the depth of this image from the DDAD dataset. <image> Answer: This is the <depth>.<mask>RLE string</mask>|
|---|

Depth Estimation for the Open World. Depth estimation in open world scenes defaults to Log-uniform Quantization, mapping real depths ranging from 0.5 to 100m to 1 to 1000. Values outside this range are set to 0 (IGNORE). During testing, the predicted values need to be de-quantized to obtain the real depth; otherwise, only relative depth is obtained. Additionally, we require inputs to be resized to a focal length of 2000 pixels to simulate a camera with a default focal length. Discrepancies in actual focal length may lead to biased predicted depths, resulting in relative depth outputs. Unlike specific datasets that require prompts to be placed in advance, our open-set supports prompts both before and after the image. It is important to note that this task relies on a large amount of training data, and the capabilities of depth anything still need improvement. However, we have demonstrated that the standard VLM model can accommodate several different quantization methods, inherently building a spatial depth perception without requiring additional structures. We recommend using the following prompts and their Chinese translation versions:

|Estimate the depth. Estimate the depth in the default range. Predict the depth map. Estimate the depth map. What the depth map? Execute depth estimation. Run depth estimation. Generate a depth map. Calculate the depth. Provide depth estimation. Perform depth prediction. Create the depth map. Get the depth map for the image. Compute the depth information. Estimate distance using depth. Run a depth map generation. Can you predict the depth map? Show me the depth estimation. Depth map prediction, please. Extract depth from the image. Apply depth estimation.|
|---|

###### A.1.8 Human Pose Estimation

MPII [67]. We evaluate our model on the MPII validation set using the standard PCKh@0.5 metric. Following the evaluation setting of ViTPose [53], where results are reported on images cropped using ground-truth person bounding boxes, the input images are constructed from ground-truth bounding box crops. As our model predicts multiple human poses per image in a single forward pass, we adopt a post-processing strategy to enable one-to-one evaluation. Specifically, for each image, we compute the keypoint center of the ground-truth instance and select the predicted pose whose keypoint center is spatially closest to it, discarding all other predictions. The selected prediction is then evaluated against the corresponding ground-truth pose using the standard PCKh metric.

|<image> Question: Detect all persons and their poses from the image within the class set of MPII Human Pose Dataset. The output format should be a JSON-like string, containing person instances. Each person instance is enclosed in <person>...</person> tags. Within each person instance, provide the bounding box using <box>...</box> tags and their 16 keypoints using <kpt>...</kpt> tags. The bounding box is defined by <x_x1><y_y1><x_x2><y_y2> tags, and each keypoint is defined by <x_...><y_...><v_...> tags, where x, y are coordinates and v is visibility. The joints must follow the standard MPII ordering. Answer: <person><box><x_457><y_187><x_548><y_315></box> <kpt><x_531><y_793><v_1.0></kpt> <kpt><x_624><y_659><v_1.0></kpt> ...<kpt><x_602><y_281><v_1.0></kpt> </person>|
|---|

Note. In the structured output, each coordinate and visibility value (e.g., <x_*>, <y_*>, <v_*>) is represented as a dedicated special token and is directly learned during training. For brevity, only a subset of keypoints is shown in the example above; in practice, the model predicts the complete set of 16 keypoints for each detected person following the MPII joint ordering.

##### A.2 General Multimodal Benchmarks

We provide the detailed evaluation settings and prompts for the general multimodal benchmarks reported in the main paper. All evaluations were conducted using a refined VLM evaluation framework based on the VLMEvalKit [34]. Specifically, we enhanced the framework to address the limitations of the original exactmatching protocol used in certain Multiple-Choice Question (MCQ) benchmarks. To improve evaluation accuracy, we integrated an LLM-based judging mechanism that leverages a large language model to robustly extract and parse final answers from the model’s generated responses.

Implementation Details. To ensure optimal performance across diverse tasks, we tailored the visual resolution settings based on the benchmark requirements. For the majority of benchmarks that demand finegrained visual details (e.g., DocVQA [95], MMMU [82], MMBench [108], RealWorldQA [100]), we employed a dynamic high-resolution strategy. Specifically, we set the minimum number of patches to 1,280 (default is 64), with a maximum of 65,536 patches. For benchmarks less sensitive to extreme resolution or where standard resolution suffices (e.g., MathVista [85], OCRBench [109], ScienceQA [75], HallusionBench [88]), we retained the default setting (min_num_patches=64).

Regarding the evaluation protocol, particularly for MCQ benchmarks, we adopted a hybrid judging strategy to ensure accuracy. We first attempt to extract the answer using exact matching. If the extraction fails or the model’s output is ambiguous, we utilize an LLM-based judge to parse the response.

Prompt Specifics. Regarding prompting strategies, we categorized benchmarks based on their task nature. For benchmarks requiring complex reasoning (e.g., MathVerse [86], MMMU, VisuLogic [81]) or real-world analysis (e.g., RealWorldQA), we selectively activated Chain-of-Thought (CoT) prompting (e.g., "Think step-by-step"). For hallucination and OCR tasks, we employed strict constraints to ensure concise and precise outputs. The detailed configurations of the prompts are provided below.

VisuLogic [81]. For this benchmark, which focuses on complex visual logical reasoning, we utilized a specialized prompt that explicitly instructs the model to reason step-by-step before providing the final answer in a boxed format.

|<image> {Question}<br><br>Solve the complex visual logical reasoning problem through step-by-step reasoning.Think about the reasoning process first and answer the question following this format: Answer: \boxed{LETTER} Think step-by-step.|
|---|

MathVerse & Mathvista & LogicVista [86, 85, 87]. To tackle complex reasoning in these benchmarks, we incorporated the CoT strategy to elicit structured geometric and mathematical derivations.

|<image> {Question} Think step-by-step.|
|---|

MMMU (VAL & Pro_Standard) [83, 110]. For the massive multi-discipline understanding benchmarks, we appended a standard CoT instruction after the options to encourage detailed reasoning.

|<image> Hint: {Hint} Question: {Question} Options:<br><br>A. {OptionA}<br>B. {OptionB}<br><br><br>... Please select the correct answer from the options above. Think step-by-step.|
|---|

MMMU (Pro_V) [83]. The vision-only protocol of MMMU-Pro. In this configuration, redundant textual prompts are removed and replaced by CoT instructions to elicit depth-first visual analysis and logical deduction.

|<image> Think step-by-step.|
|---|

RealWorldQA [100]. Although this is a VQA task, we found that spatial and physical understanding benefits significantly from reasoning. Therefore, we forced the "Think step-by-step" instruction.

|<image> {Question} Think step-by-step.|
|---|

MMBench & CV-Bench [71, 74]. For general perception and reasoning, we utilized the standard CircularEval strategy provided by VLMEvalKit, augmented with a step-by-step thinking instruction.

|<image> Hint: {Hint} Question: {Question} Options: A. {OptionA}<br><br>... Please select the correct answer from the options above. Think step-by-step.|
|---|

MME-RealWorld [101]. For this benchmark, we used a detailed system prompt to guide the model in selecting the best option, ending with a specific suffix to induce the answer generation.

|<image> {Question} {Options}<br><br>Select the best answer to the above multiple-choice question based on the image. Respond with the letter (A, B, C, D, or E) of the correct option. The best answer is:|
|---|

HallusionBench & POPE [88, 89]. To rigorously test for hallucinations, we disabled CoT and enforced a strict "Yes/No" constraint to prevent the model from generating evasive or verbose responses.

|<image> {Question} Please answer yes or no.|
|---|

DocVQA, InfoVQA, TextVQA, ChartQA & AI2D [95, 93, 94, 96]. For document understanding and OCR-related tasks, we appended a constraint to ensure the output is a short phrase or single word, facilitating accurate metric calculation.

|<image> {Question} Answer the question with a single word or phrase.|
|---|

ScienceQA & SEEDBench [75, 76]. For these general multiple-choice benchmarks, we used a direct answer prompt without CoT to evaluate the model’s direct knowledge retrieval capabilities.

|<image> Hint: {Hint} Question: {Question} Choices: A. {OptionA}<br><br>... Answer the question with a single word or phrase.|
|---|

OSWorld [25]. For the OSWorld benchmark, we use the prompts below to evaluate the candidate models.

|System: # Tools You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags: <tools> {"type": "function", "function": {"name_for_human": "computer_use", "name":<br><br>"computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.<br><br>This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.<br><br>Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn’t open, try wait and taking another screenshot.<br><br>The screen’s resolution is {Resolution}. Whenever you intend to move the cursor to click on an element like an icon, you should<br><br>consult a screenshot to determine the coordinates of the element before moving the cursor.<br><br>If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.<br><br>Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don’t click boxes on their edges unless asked.", "parameters":<br><br>{"properties": {"action": {"description": " ‘key‘: Performs key down presses on the arguments passed in order, then performs key<br><br>releases in reverse order. ‘type‘: Type a string of text on the keyboard. ‘mouse_move‘: Move the cursor to a specified (x, y) pixel coordinate on the screen. ‘left_click‘: Click the left mouse button at a specified (x, y) pixel coordinate on<br><br>the screen.<br><br>‘left_click_drag‘: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.<br><br>‘right_click‘: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.<br><br>‘middle_click‘: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.<br><br>‘double_click‘: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.<br><br>‘triple_click‘: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it’s the closest action). ‘scroll‘: Performs a scroll of the mouse scroll wheel. ‘hscroll‘: Performs a horizontal scroll (mapped to regular scroll). ‘wait‘: Wait specified seconds for the change to happen. ‘terminate‘: Terminate the current task and report its completion status. ‘answer‘: Answer a question.<br><br>", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by ‘action=key‘.", "type": "array"}, "text": {"description": "Required only by ‘action=type‘.", "type": "string"}, "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"}, "pixels": {"description": "The amount of scrolling.", "type": "number"}, "time": {"description": "The seconds to wait.", "type": "number"}, "status": {"description": "The status of the task.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}} </tools>|
|---|

|For each function call, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call> # Response format Response format for every step: 1) Action: a short imperative describing what to do in the UI. 2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}. Rules: - Output exactly in the order: Action, <tool_call>. - Be brief: one sentence for Action. - Do not output anything else outside those parts. - If finishing, use action=terminate in the tool call. User: <image> Please generate the next move according to the UI screenshot, instruction and previous actions. Instruction: {Question} Previous actions:<br><br>{Prevision Action List}|
|---|

ScreenSpot Pro [102]. For the ScreenSpot Pro benchmark, we used the following prompt to evaluate the candidate models.

|System: # Tools You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags: <tools>{ "name":"computer_use", "description": "Use a mouse to interact with a computer. The screen’s resolution is {Resolution}." "notes": "Click with the cursor tip centered on targets; avoid edges unless asked. Do not use other tools (type, key, scroll, left_click_drag). Only left_click and mouse_move are allowed. If you can’t find the element, terminate and report failure.", "parameters":{ "type":"object", "required":["action"], "properties":{ "action":{ "type":"string", "enum":["mouse_move","left_click"], "description":"The action to perform." }, "coordinate":{ "type":"array", "description":"(x, y): pixels from left/top. Required for action=mouse_move and action=left_click." } } } } } </tools> For each function call, return a json object with function name and arguments within <tool_call>. . .</tool_call> XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call> Response format for every step:<br><br>1) Action: a short imperative describing what to do in the UI.<br>2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}. Rules:<br><br><br>- Output exactly in the order: Action, <tool_call>.<br>- Be brief: one sentence for Action.<br>- Do not output anything else outside those parts.<br>- If finishing, use action=terminate in the tool call.<br>|
|---|

|User: <image> You are given a screenshot of a desktop GUI and an instruction describing a target UI element. Your goal is to choose exactly one action using the computer_use tool to click on that target.<br><br>Task instruction: {instruction}<br><br>You must output exactly one step following the Response format, with a single computer_use tool call whose ‘coordinate‘ corresponds to the center of the target element in a {resolution} relative coordinate system (x from left to right, y from top to bottom).|
|---|

### B Extended Experiments and Comparison

Vision-centric tasks encompass a wide variety of settings, making comprehensive comparison challenging. To enable thorough evaluation beside Table 2, we provide two additional supplementary tables—Table 4 and Table 5, focusing on segmentation and grounding, respectively. In these tables, we systematically compare against five representative settings: (1) vision specialist models, (2) CLIP-based vision–language models, (3) vision generalist architectures, (4) multimodal LLMs with task-specific additions, and (5) our standard multimodal LLM, which uses no extra modules, heads, or task embeddings. Our results show that, despite its architectural simplicity, our approach achieves competitive or state-of-the-art performance across diverse settings. This demonstrates that a standard MLLM, when equipped with proper supervision and training, can serve as a highly effective and universal predictor, combining strong performance, minimal design, and broad applicability.

##### B.1 Comparison With Dense Prediction Methods

- The results presented in Table 4 demonstrate the superior performance of our proposed Youtu-VL across multiple dense prediction benchmarks, highlighting its capability as a robust vision-language generalist without requiring complex architectural additions.

In the task of semantic segmentation, Youtu-VL achieves remarkable results across all five datasets. Notably, it attains 54.2 mIoU on ADE20K, compared with 47.8 mIoU of the vision generalist models like GiT [62] and 32.1 mIoU of the CLIP-based methods such as SAN [47]. Crucially, distinct from standard Multimodal LLMs that are typically unable to perform these tasks (indicated by “×”), Youtu-VL demonstrates a unique capability to handle fine-grained dense prediction directly.

In the depth estimation tasks, Youtu-VL shows competitive results. It achieves 90.4 on NYUv2, while the result of DepthLLM [49] is 87.6. Our method just needs to process the image once, while DepthLLM needs to draw each point and inference multiple times. We also fine-tune the model like the vision specialist model SwinMTL [51] and achieve a 92.7 δ1 (vs. 92.1 of SwinMTL). While the specialist model UniDepthv2 [50] performs higher on NYUv2, the performance on DDAD is quite close (88.2 vs. 87.6), suggesting this general-purpose model sometimes meets the high performance of task-specific models.

Youtu-VL demonstrates strong performance in the referring expression segmentation task. On the RefCOCO

Semantic Segmentation Depth Estimation Referring Segmentation (RefCOCO) Methods Additions ADE COCOstuff Context59 Cityscapes VOC20 NYUv2 DDAD Cityscapes val testA testB val+ testA+ testB+ val-g test-g Vision Specialist Models Segformer (MiT-B5) [111] MLP Decoder 51.0 46.7 - 82.4 - × × × × × × × × × × × Mask2Former (Swin-L) [112] Pixel Decoder 56.0 - - 83.3 - × × × × × × × × × × × UniDepth-v2 (ViT-L) [50] Depth Decoder - - - - - 98.8 88.2 - × × × × × × × × SwinMTL (Swin-B) [51] MLP Decoder - - - - 76.41 - - 92.1 × × × × × × × × RVG (ViT-B) [48] MLP Decoder × × × × × × × × 79.4 81.2 77.8 69.5 75.7 63.0 71.3 72.1 VPD (UNet) [113] Denoising Decoder 53.7 - - - - 96.4 - - 73.3 - - 62.7 - - 62.0 CLIP-based Vision Language Models

MaskCLIP [46] Attn Adaptation 12.3 16.9 26.2 25.6 62.9 × × × × × × × × × × × CLIPSurgery [114] Consistent Attn 16.1 21.9 29.3 31.4 77.5 × × × × × × × × × × × CASS [115] VFM Graph 20.4 26.7 40.2 39.4 87.8 × × × × × × × × × × × SAN (ViT-L) [47] Decoupled Head 32.1 45.8 57.7 - 94.6 × × × × × × × × × × ×

Vision Generalist Models X-Decoder (DaViT-d5) [116] X-Decoder 58.1 - 60.4 81.7 97.7 × × × - - - - - - - 4M (ViT-L) [117] Task-specific Heads 53.4 - - - - 94.4 - - × × × × × × × × SEEM (DaViT-d5) [117] SEEM-Decoder - - - - - × × × - - - - - - 65.6 BEIT-3 (Multiway-T) [118] None 62.8 - - - - × × × × × × × × × × × GiT [62] Parallel Decoding 47.8 49.1 63.3 61.8 - × × × × × × × × × × × SAM3 [44] DETR-like Decoder 13.8 - 60.8 65.2 - - - - 75.5 77.6 71.0 67.3 71.1 63.4 73.4 74.0

###### VLM with Additions

GLaMM (Vicuna-7B) [12] SAM/Pixel Decoder × × × × × × × × 79.5 83.2 76.9 72.6 78.7 64.6 74.2 74.9 UniPixel (Qwne2.5-VL-3B) [12] SAM Decoder × × × × × × × × 80.5 82.6 76.9 74.3 78.9 68.4 76.3 77.0 VisionLLM v2 (Swin-T) [13] Deform-DETR 52.3 - - - - x x x 76.6 79.3 74.3 64.5 69.8 61.5 70.7 71.2 UFO (InternVL2.5-8B) [15] Mask Tokens 54.5 30.2 - - - 93.6 - - 80.0 81.6 78.1 76.7 79.9 72.3 75.5 76.3

Standard VLM InternVL-3.5 (4B) [2] None × × × × × × × × × × × × × × × × Qwen3-VL (4B) [1] None × × × × × × × × × × × × × × × × DepthLLM (Qwne2.5-VL-3B) [49] None × × × × × 86.8 74.7 - × × × × × × × × VistaLLM (Vicuna-7B) [119] None × × × × × × × × 74.5 76.0 72.7 69.1 73.7 64.0 69.0 70.9 Youtu-VL (4B, Ours) None 54.2 52.2 60.4 70.4 92.5 90.4 87.6 92.7 80.7 82.0 78.4 76.2 79.6 71.4 76.5 76.6

- Table 4. Quantitative comparison on dense prediction tasks. We evaluate the performance on Semantic Segmentation (ADE20K, COCO-Stuff, Pascal Context, Cityscapes, VOC2012), Depth Estimation (NYUv2, DDAD, Cityscapes), and Referring Segmentation (RefCOCO, RefCOCO+, RefCOCOg). The table compares our proposed Youtu-VL against Vision Specialist Models, CLIP-based methods, Vision Generalist Models, and other Multimodal LLMs (both with architectural additions and standard setups). "Additions" refers to the usage of additional task-specific decoders, heads, or task tokens. “×” denotes that the method is not applicable and “-” indicates results are not available. Results in gray indicate task-specific fine-tuning on a single dataset, which usually brings higher results. Best results in each setting are marked in bold.

val, Youtu-VL achieves cIoU of 80.7, while the baseline results are 79.5 for GLaMM [12] and 80.5 for UniPixel [14], which requires an extra SAM decoder. Besides, the method based on the polygon prediction is merely 74.5. The UFO [15] achieves an 80.0 cIoU (80.7 of Youtu-VL), while this method needs extra mask token embeddings for a retrieval process. Compared with the above methods. Our method is straightforward yet highly effective, designed to be applicable to standard VLMs without requiring extra architectural modifications.

##### B.2 Comparison on Localization Tasks

- The results presented in Table 5 demonstrate that Youtu-VL achieves superior performance in localization tasks without requiring complex architectural additions. This success is attributed to the synergistic interplay among detection, counting, and grounding capabilities, which establishes leading performance within a standard VLM framework.

In object detection, Youtu-VL achieves 47.1 mAP on the COCO validation split, underscoring the potential of standard VLMs in handling dense localization tasks. Furthermore, performing SFT exclusively on detection data starting from Stage 3 boosts performance to 48.0 mAP, effectively bridging the gap with specialized architectures. Regarding visual grounding, Youtu-VL secures leading results across all splits of the RefCOCO series. Similarly, in counting tasks, the model exhibits remarkable performance on benchmarks such as CountBench, TallyQA-simple, and TallyQA-complex.

###### Methods Detection Counting Grounding

COCO val CountBench TallyQA-Simple TallyQA-Complex val testA testB val+ testA+ testB+ val-g test-g Vision Specialist Models

Mr. DETR++ (Swin-L) [120] 58.4 x x x x x x x x x x x Omni-SMoLA (PaLI-XFT) [57] x - 86.3 77.1 x x x x x x x x MDETR (ENB3) [121] x x x x 87.5 90.4 82.7 81.1 85.5 73.0 83.4 83.3

Vision Generalist Models BEIT-3 (Multiway-T) [118] 63.7 x x x x x x x x x x x Florence-2 (DaViT-B) [122] 43.4 x x x 93.4 95.3 92.0 88.3 92.9 83.6 91.2 91.7 Grounding DINO (Swin-L) [37] 62.6 x x x 90.6 93.2 88.2 82.8 89.0 75.9 86.1 87.0 SAM3 (Perception Encoder) [44] 56.4 93.8 - - - - - - - - - VLM with Additions

GLaMM (Vicuna-7B) [12] x x x x 79.5 83.2 76.9 72.6 78.7 64.6 74.2 74.9 VisionLLM v2 (Swin-T) [13] 56.7 - - - 87.9 91.2 84.3 77.6 83.8 70.2 82.9 84.1 UFO (InternVL2.5-8B) [15] 48.9 x x x 93.1 94.8 89.2 87.7 92.1 82.3 88.2 89.2

###### Standard VLM

Griffon (LLama2-13B) [123] 24.8 x x x 90.1 93.4 86.1 84.8 90.5 77.8 86.1 87.2 InternVL-3.5 (4B) [2] - 79.4 77.6 66.4 92.5 94.3 88.2 87.6 92.3 81.6 89.6 89.3 Qwen3-VL (4B) [1] - 78.4 79.0 64.0 90.7 92.2 86.7 82.9 89.4 75.6 87.3 87.7 Youtu-VL (4B, Ours) 47.1 (48.0∗) 88.6 85.1 74.4 93.6 95.2 90.8 90.1 93.9 85.4 92.2 92.9

- Table 5. Quantitative comparison on localization tasks. We evaluate the performance on Object Detection (COCO 2017 val split), Counting (CountBench, TallyQA-Simple, TallyQA-Complex), and Visual Grounding (RefCOCO, RefCOCO+, RefCOCOg). The table compares our proposed Youtu-VL against Vision Specialist Models, Vision Generalist Models, and other Multimodal LLMs (both with architectural additions and standard setups). “×” denotes that the method is not applicable and “-” indicates results are not available. Best results in each setting are marked in bold. “∗” indicates the performance of the model obtained after fine-tuning exclusively on detection data starting from Stage 3.

### C Case Study

##### Object Detection

[Figure 182]

User Instruction: Detect all objects in the provided image.

##### Model Response

<ref>cup</ref><box><x_46><y_27><x_476><y_719></box><ref>fork</ref><box><x_432><y_342>

- <x_680><y_1077></box><ref>cake</ref><box><x_576><y_532><x_1223><y_952></box> <ref>orange</ref><box><x_702><y_290><x_1066><y_554></box>

Figure 12. Qualitative example of object detection from the COCO dataset.

##### Object Detection

[Figure 184]

User Instruction: Detect all objects in the provided image.

##### Model Response

<ref>truck</ref><box><x_0><y_0><x_515><y_1204></box><ref>car</ref><box><x_1308>

- <y_515><x_1717><y_732></box><box><x_1820><y_522><x_2239><y_792></box><box><x_2180> <y_639><x_2239><y_783></box><box><x_284><y_632><x_472><y_801></box><box>

- <x_1011><y_774><x_2239><y_1663></box><ref>traffic light</ref><box><x_1439><y_9>

- <x_1509><y_55></box><box><x_1300><y_34><x_1368><y_80></box><box><x_483><y_207>

- <x_520><y_304></box><ref>parking meter</ref><box><x_1837><y_511><x_1874><y_580> </box><box><x_852><y_987><x_1057><y_1582></box><ref>bird</ref><box><x_715><y_745>

- <x_1058><y_1098></box>

Figure 13. Qualitative example of object detection from the COCO dataset.

##### Open-World Detection

[Figure 186]

User Instruction: Detect all objects.

##### Model Response

<ref>bird</ref><box><x_209><y_246><x_232><y_271></box><box><x_557><y_297><x_598>

- <y_337></box><box><x_245><y_163><x_297><y_191></box><box><x_91><y_41><x_163>

- <y_64></box><box><x_87><y_104><x_144><y_126></box><box><x_345><y_150><x_432>

- <y_179></box><box><x_284><y_98><x_351><y_134></box><box><x_409><y_72><x_466>

- <y_92></box><box><x_485><y_51><x_566><y_88></box><box><x_462><y_298><x_493> <y_318></box><box><x_308><y_57><x_383><y_75></box><ref>zebra</ref><box><x_463> <y_151><x_497><y_210></box><box><x_540><y_141><x_567><y_208></box><box><x_68> <y_167><x_233><y_282></box><box><x_24><y_177><x_140><y_266></box>

Figure 14. Qualitative example of open-world detection from the Objects365 dataset.

##### Detect-then-Count

[Figure 188]

User Instruction: How many strawberry can you see in the image? First detect each one, then count the number.

##### Model Response

<ref>Strawberry</ref><box><x_1625><y_288><x_1889><y_629></box><box><x_1321><y_408> <x_1621><y_650></box><box><x_957><y_484><x_1257><y_776></box><box><x_681><y_577> <x_946><y_806></box><box><x_493><y_620><x_698><y_821></box><box><x_1325><y_624> <x_1640><y_898></box><box><x_1638><y_630><x_1881><y_867></box><box><x_995><y_781> <x_1315><y_1063></box><box><x_718><y_791><x_1000><y_1082></box><box><x_534><y_824> <x_757><y_1052></box><box><x_1578><y_871><x_1829><y_1163></box><box><x_1253><y_894> <x_1570><y_1208></box><box><x_465><y_1007><x_803><y_1322></box><box><x_1061><y_1039> <x_1271><y_1251></box><box><x_786><y_1043><x_1039><y_1293></box><box><x_1270><y_1178> <x_1569><y_1509></box><box><x_1568><y_1180><x_1819><y_1503></box><box><x_936><y_1217> <x_1250><y_1606></box><box><x_633><y_1256><x_949><y_1621></box><box><x_394><y_1292>

- <x_641><y_1598></box><box><x_1564><y_1498><x_1792><y_1754></box><box><x_1266><y_1504>

- <x_1563><y_1784></box><box><x_504><y_1477><x_703><y_1687></box><box><x_1029><y_1470> <x_1283><y_1701></box><box><x_792><y_1502><x_968><y_1715></box>The answer is 25

Figure 15. Qualitative example of detect-then-count from a photo taken by an author from Youtu-VL Team.

##### Semantic Segmentation

[Figure 190]

User Instruction: Label all river, tree, pot, tank, blind, bicycle, bannister, lake, screen, barrel, mirror, swimming pool, stairway, coffee table, bottle, bar, grass, plant, ball, fence, streetlight, sky, cushion, chair, microwave, stove, awning, rock, palm, sculpture, airplane, shower, conveyer belt, canopy, booth, radiator, flower, ashcan, fountain, house, bag, stage, minibike, screen door, escalator, oven, case, rug, plaything, poster, washer, pole, animal, swivel chair, ship, sand, bench, sidewalk, path, table, shelf, runway, counter, tower, bed , truck, basket, sofa, railing, crt screen, door, land, lamp, signboard, sconce, waterfall, ottoman, wardrobe, bus, countertop, kitchen island, column, hood, fan, glass, seat, base, stool, boat, hovel, bookcase, water, tent, traffic light, bathtub, tray, toilet, earth, sink, dishwasher, building, cabinet, pool table, apparel, painting, flag, trade name, armchair, sea, light, road, computer, monitor, desk, bridge, chandelier, fireplace, book, skyscraper, person, pillow, blanket, grandstand, windowpane, van, vase, towel, curtain, stairs, field, wall, hill, arcade machine, clock, pier, refrigerator, cradle, step, bulletin board, box, chest of drawers, mountain, food, television receiver, dirt track, plate, floor, buffet, ceiling, car in the image, without the background class.

##### Model Response

The target categories include <ref>sky</ref><ref>road</ref><ref>car</ref><ref>tree</ref> <ref>building</ref><ref>awning</ref><ref>fountain</ref><ref>plant</ref>, numbered sequentially starting from 0, without the background class.<mask>RLE string</mark>

Figure 16. Qualitative example of semantic segmentation from the ADE20k dataset.

##### Semantic Segmentation

[Figure 192]

User Instruction: Create segmentation maps for these items: countertop, flower, hood, sky, shower, path, dishwasher, tank, skyscraper, railing, chest of drawers, arcade machine, painting, bar, building, sand, mountain, bannister, shelf, bus, bed , mirror, table, step, pier, refrigerator, floor, bag, minibike, conveyer belt, awning, canopy, runway, flag, book, basket, waterfall, water, car, apparel, bulletin board, pillow, signboard, washer, van, rug, plaything, chair, fountain, stairs, plate, animal, bottle, stove, door, blanket, stool, wall, buffet, trade name, fence, clock, box, dirt track, cushion, counter, radiator, seat, escalator, cradle, screen door, booth, ashcan, house, sofa, cabinet, coffee table, blind, fireplace, vase, swimming pool, bathtub, lamp, column, bicycle, toilet, hill, bridge, tower, ship, sea, pole, wardrobe, plant, tray, ceiling, windowpane, tree, television receiver, computer, light, oven, bench, grass, streetlight, monitor, river, bookcase, boat, rock, poster, person, food, curtain, sink, pool table, sidewalk, crt screen, earth, palm, lake, land, screen, stage, truck, desk, swivel chair, pot, traffic light, fan, chandelier, sculpture, ottoman, road, glass, hovel, base, towel, kitchen island, stairway, field, microwave, airplane, sconce, barrel, tent, case, grandstand, armchair, ball, without the background class.

##### Model Response

The target categories include <ref>car</ref><ref>signboard</ref><ref>sidewalk</ref> <ref>sky</ref><ref>road</ref><ref>building</ref><ref>pole</ref><ref>streetlight</ref>, numbered sequentially starting from 0, without the background class.<mask>RLE string</mark>

Figure 17. Qualitative example of semantic segmentation from the ADE20k dataset.

##### Visual Grounding

[Figure 194]

User Instruction: Please provide the bounding box coordinate of the region this sentence describes: the black cat sitting on a wooden toilet

##### Model Response

<box><x_76><y_146><x_213><y_339></box>

Figure 18. Qualitative example of visual grounding from the RefCOCO dataset.

##### Visual Grounding

[Figure 196]

User Instruction: Please provide the bounding box coordinate of the region this sentence describes: the container holding fresh vegetables

##### Model Response

<box><x_0><y_0><x_309><y_330></box>

Figure 19. Qualitative example of visual grounding from the RefCOCO dataset.

##### Referring Segmentation

[Figure 198]

- User Instruction 1: Please provide the bounding box coordinate of the region this sentence describes: the laptop left top

- Model Response 1

<box><x_0><y_13><x_234><y_245></box> (Operations: Draw box -> Crop image -> Resize)

User Instruction 2: Segment the core target.

- Model Response 2 The results are 0 for <ref><BG></ref> and 1 for <ref><FG></ref>.<mask>RLE string</mask>

Figure 20. Qualitative example of referring expression segmentation from the RefCOCO dataset.

##### Depth Estimation

[Figure 200]

User Instruction: Please estimate the depth of this image from the NYUv2 dataset

##### Model Response

This is the <depth>.<mask>RLE str</mask>

Figure 21. Qualitative example of depth estimation from the NYUv2 dataset.

##### Depth Estimation

[Figure 201]

User Instruction: Please estimate the depth of this image from the NYUv2 dataset

##### Model Response

This is the <depth>.<mask>RLE str</mask>

Figure 22. Qualitative example of depth estimation from the NYUv2 dataset.

##### Human Pose Estimation

[Figure 203]

User Instruction: Detect all persons and their poses from the image within the class set of MPII Human Pose Dataset. The output format should be a JSON-like string, containing person instances. Each person instance is enclosed in <person>...</person> tags. Within each person instance, provide the bounding box using <box>...</box> tags and their 16 keypoints using <kpt>...</kpt> tags. The bounding box is defined by <x_x1><y_y1><x_x2><y_y2> tags, and each keypoint is defined by

- <x_...><y_...><v_...> tags, where x, y are coordinates and v is visibility. The joints must be in this specific order: (0) right_ankle, (1) right_knee, (2) right_hip, (3) left_hip, (4) left_knee, (5) left_ankle, (6) pelvis, (7) thorax, (8) upper_neck, (9) head_top, (10) right_wrist, (11) right_elbow, (12) right_shoulder,

(13) left_shoulder, (14) left_elbow, (15) left_wrist. Please output all detected persons with their bounding boxes and keypoints.

Model Response

<person><box><x_306><y_240><x_410><y_361></box><kpt><x_342><y_1084></kpt><kpt>

- <x_349><y_868></kpt><kpt><x_353><y_712></kpt><kpt><x_272><y_707></kpt><kpt><x_261>
- <y_864></kpt><kpt><x_262><y_1078></kpt><kpt><x_313><y_710></kpt><kpt>

- <x_328><y_391></kpt><kpt><x_338><y_362></kpt><kpt><x_379><y_239></kpt><kpt><x_394>
- <y_703></kpt><kpt><x_403><y_566></kpt><kpt><x_396><y_393></kpt><kpt>

- <x_257><y_388></kpt><kpt><x_216><y_522></kpt><kpt><x_269>

...

- <x_1364><y_224><x_1468><y_340></box><kpt><x_1471><y_1104></kpt><kpt><x_1483>
- <y_846></kpt><kpt><x_1488><y_686></kpt><kpt><x_1393><y_689></kpt><kpt><x_1398>

- <y_860></kpt><kpt><x_1409><y_1112></kpt><kpt><x_1442><y_688></kpt><kpt><x_1439>

- <y_381></kpt><kpt><x_1430><y_341></kpt><kpt><x_1401><y_223></kpt><kpt><x_1465>

- <y_602></kpt><kpt><x_1551><y_574></kpt><kpt><x_1529><y_382></kpt><kpt><x_1354>

- <y_379></kpt><kpt><x_1342><y_539></kpt><kpt><x_1333><y_624></kpt></person>

69

Figure 23. Qualitative example of human pose estimation from generated image (HY).

##### Creative Writing

[Figure 205]

User Instruction: 请根据这张图片，创作一首七言绝句

##### Model Response

雪岭晨曦染赤霞， 枯枝剪影立寒坡。 云海漫卷千山静， 一抹金辉破夜韶。

- Figure 24. Qualitative example of visual-language creative generation. The model perceives the visual aesthetics of the snowy sunrise and composes a coherent Chinese poem matching the atmosphere.

##### Mathematical Reasoning

[Figure 207]

User Instruction: Find the length of AC in the isosceles triangle ABC.

##### Model Response

###### Phase 1: Comprehensive Information Extraction

- • Textual Input: The problem specifies triangle ABC is isosceles and asks for the length of AC.
- • Visual Input:

- – Side AB = 9x − 1, side BC = 5x − 0.5, and side AC = 4x + 1.
- – Pink tick marks on sides AC and BC indicate these two sides are equal (a standard geometric notation for congruent segments).

###### Phase 2: Strategic Problem Setup

- • Goal: Determine the value of x using the equality of AC and BC, then compute AC.
- • Assumption: Since the tick marks confirm AC = BC, this is the correct pair to equate.
- • Relevant Principle: In an isosceles triangle, two sides are equal. Here, AC = BC implies 4x + 1 = 5x − 0.5.

###### Phase 3: Rigorous Solution Execution

- 1. Set up the equation based on the equality of AC and BC: 4x + 1 = 5x − 0.5
- 2. Solve for x: Subtract 4x from both sides:

1 = x − 0.5 Add 0.5 to both sides:

x = 1.5

- 3. Substitute x = 1.5 into the expression for AC: AC = 4x + 1 = 4 × 1.5 + 1 = 6 + 1 = 7
- 4. Verification:

- • Compute BC with x = 1.5: BC = 5x − 0.5 = 5 × 1.5 − 0.5 = 7.5 − 0.5 = 7
- • Both AC and BC equal 7, confirming the solution is consistent with the diagram’s tick marks.

###### Phase 4: Solution Validation

- • The computed value x = 1.5 satisfies the condition AC = BC, as required by the visual notation.
- • Substituting back confirms all side lengths align with the given expressions.
- • No contradictions exist between the diagram and calculations.

Therefore, the final answer is 7.

- Figure 25. Qualitative example of mathematical reasoning. The model successfully extracts geometric constraints (isosceles triangle properties) from the image, formulates the algebraic equation, and solves for the specific side length step-by-step.

##### Chart & Table Understanding

[Figure 209]

User Instruction: What is the average amount of goods where the second letter in the name is ’e’?

##### Model Response

We are asked to find the average amount of goods where the second letter in the name is ‘e’. First, let’s identify which items in the chart are goods (purple bars) and whose name contains the letter ‘e’ as the second letter. Looking at the list:

- 1. Petroleum products — purple bar → 69.6B Name: “Petroleum” → letters: P-E-T-O-L-I-M-E-N-T → second letter is E
- 2. Other business services — orange bar → not goods
- 3. Pearls, precious and semiprecious stones — purple bar → 39.3B Name: “Pearls” → P-E-A-R-L-S → second letter is E
- 4. Transport — orange bar → not goods
- 5. Machinery and appliances — purple bar → 25.4B Name: “Machinery” → M-A-C-H-I-R-E-Y → second letter is A “appliances” → A-P-P-L-I-C-A-T-I-O-N-S → second letter is P

→ So overall, second letter is A or P, not E

- 6. Iron and steel — purple bar → 22.9B Name: “Iron” → I-R-O-N → second letter is R “steel” → S-T-E-E-L → second letter is T

→ Not E

- 7. Organic chemicals — purple bar → 22.1B Name: “Organic” → O-R-G-A-N-I-C → second letter is R “chemicals” → C-H-E-M-I-C-A-L-S → second letter is H

→ Not E

So only two goods meet the condition:

- • Petroleum products: 69.6B
- • Pearls, precious and semiprecious stones: 39.3B

Now, compute the average:

Average = (69.6 + 39.3)/2 = 108.9/2 = 54.45

Final Answer: 54.45B

- Figure 26. Qualitative example of chart understanding and multi-step reasoning. The model accurately interprets visual cues (color legends) to distinguish data categories, applies character-level text analysis to filter labels, and performs arithmetic operations on the extracted values to derive the answer.

##### Visual Perception

[Figure 211]

User Instruction: This is a color vision test chart. Please carefully observe the numbers in the picture and answer in the following format: Numbers: what you see.

##### Model Response

Numbers: 86

- Figure 27. Qualitative example of fine-grained visual perception. The model accurately recognizes the number hidden in the Ishihara color test plate, demonstrating the capability to distinguish subtle chromatic differences and patterns to identify the target information.

Mathematical OCR & LaTeX Conversion

[Figure 212]

User Instruction: 能把图中公式整理为latex格式吗

Model Response

以下是图中公式整理后的LaTeX 格式：

B(M) =

3 Lhost(M)

M

Mmin

dn dm

dm

1

0

dxsub × [1 + Bsub(m)] Lsub(m, xsub)xsub2

- Figure 28. Qualitative example of mathematical formula recognition. The model accurately parses the complex spatial structure (integrals, fractions, subscripts) and specific symbols from the image, translating them into syntactically correct LaTeX code.

##### GUI Agent

[Figure 214]

User Instruction: Shut the popup window.

##### Model Response

Action: "Click the OK button on the error popup dialog to close it." <tool_call> {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1693, 755]}} </tool_call>

- Figure 29. Qualitative example of GUI agent performance. The ground-truth target is highlighted in green, and the model’s click location is shown in red. The model demonstrates robust, precise interaction with interface elements across multiple windows. The screenshot is selected from the ScreenSpot Pro [102] evaluation set.

##### GUI Agent

[Figure 216]

User Instruction: Alter the warmup steps value.

##### Model Response

Action: "Click on the value ’2000’ for the warmup parameter in the OptimArgs dataclass to prepare for editing it." <tool_call> {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [714, 800]}} </tool_call>

- Figure 30. Qualitative example of GUI agent performance. This case demonstrates the model’s ability to interpret instructions within a specific context and execute precise edits accordingly in the editor environment. The screenshot is selected from the ScreenSpot Pro [102] evaluation set.

