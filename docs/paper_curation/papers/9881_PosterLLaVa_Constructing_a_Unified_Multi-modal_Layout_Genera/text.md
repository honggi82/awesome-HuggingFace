## PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM

Tao Yang†, Yingmin Luo†, Zhongang Qi‡, Yang Wu, Member, IEEE, Ying Shan, and Chang Wen Chen‡, Fellow, IEEE

### arXiv:2406.02884v3[cs.CV]26Nov2024

Abstract—Layout generation is the keystone in achieving automated graphic design, requiring arranging the position and size of various multi-modal design elements in a visually pleasing and constraint-following manner. Previous approaches are either inefficient for large-scale applications or lack flexibility for varying design requirements. Our research introduces a unified framework for automated graphic layout generation, leveraging the multi-modal large language model (MLLM) to accommodate diverse design tasks. In contrast, our data-driven method employs structured text (JSON format) and visual instruction tuning to generate layouts under specific visual and textual constraints, including user-defined natural language specifications. We conducted extensive experiments and achieved stateof-the-art (SOTA) performance on public multi-modal layout generation benchmarks, demonstrating the effectiveness of our method. Moreover, recognizing existing datasets’ limitations in capturing the complexity of real-world graphic designs, we propose two new datasets for much more challenging tasks (user-constrained generation and complicated poster), further validating our model’s utility in real-life settings. Marking by its superior accessibility and adaptability, this approach further automates large-scale graphic design tasks. Finally, we develop an automated text-to-poster system that generates editable SVG posters based on users’ design intentions, bridging the gap between layout generation and real-world graphic design applications. This system integrates our proposed layout generation method as the core component, demonstrating its effectiveness in practical scenarios. The code and datasets are open-sourced on https://github.com/posterllava/PosterLLaVA.

Index Terms—Media arts, Layout Generation, Poster Generation, Multi-modal Generation, Large Language Models

I. INTRODUCTION

# F

OR diverse sorts of graphic design (commercial posters, mobile app UIs, webpages, video thumbnails, etc.), layout

plays a critical role in structuring visual and textual elements to captivate audiences and communicate intended messages. This task has required designers to create layouts manually, demanding their extensive expertise and experience. For largescale designing tasks, the efficiency of this strategy is far from expected.

The most naive idea for massive graphic design generation is to utilize pre-design templates and replace content according to requirements. However, the production and selection of templates still involve human labor, and mechanically applying the inappropriate layout template can lead to obtrusive designs.

†: Authors contributed equally to this research. ‡: Corresponding authors. Tao Yang, Changwen Chen are with Hong Kong Polytechnic University. Tao Yang, Yingmin Luo, Zhongang Qi, and Ying Shan are with Tencent

PCG ARC Lab. Yang Wu is with Tencent AI Lab. Manuscript received April 19, 2005; revised August 26, 2015.

###### Task Description

Background Element Type & Number

[Figure 1]

title x 1 subtitle x 1 underlay x 2 decoration x 2

"Please generate a layout according to the background image and design elements."

item logo x 3 item x 3

User Requirement

- 1. "The title should be at the top, and the subtitle_0 should be at the bottom."
- 2. "The item_0, item_1, and item_2 are left aligned and each is matched with an item logo."
- 3. "The decoration_0 should be on the upper left side of the title."

Vision Head LLM

[Figure 2]

[Figure 3]

Code Format Layout

Graphic Design

[Figure 4]

| | |
|---|---|
| | |

JSON

|{<br><br>"width": 1242, "height": 1660, "elements": [<br><br>{"type": "background", "boxes": [0, 0, 1242, 1660], "content": "path_to_image"}, {"type": "title", "boxes": [0.1302, 0.0023, 0.8922, 0.1332], "content": "美食安利"},<br><br>... ]<br><br>}| |
|---|---|
| | |

Fig. 1: The overall framework of our proposed content-aware layout generation method. Adopting the multi-modal LLM [4] as the central processing unit, we embed information from visual and textual domains to generate a reasonable and visually pleasing graphic layout. The result is encoded in JSON format and can be rendered into a real-world poster.

Previous researchers attempted to frame layout generation as an optimization problem, tackling it with heuristic algorithms [1]–[3]. However, these methods hinge on crafting welldesigned energy functions, a task that still depends heavily on design expertise, and the form of which usually lacks generality across different applications.

With the advance in deep learning, researchers are glad to embrace data-driven methods [5]–[10] in layout generation. Most of these works focus on adopting the latest generative architecture but overlook the necessary conditional requirements for layout. This limits their applicability in real-world scenarios, which frequently demand the integration of complex multi-modal conditions. Recently, more and more researchers have recognized the importance of multi-modal conditions and started to explore content-aware layout generation. For visual conditions, CGL-GAN [11] and DS-GAN [12] take an innovative step to incorporate the semantic information on background images as conditions in layout generation, and some later work [13], [14] also consider the content of foreground elements as conditions. For textual conditions, some preliminary attempts [6], [15], [16] generate layouts under given graphic conditions. However, the introduced constrained optimization processes or specific intermediate representations

strengthen the training or labeling complexity. An efficient end-to-end framework that can directly translate natural language instructions into desired layouts is still needed.

Although previous approaches have demonstrated progress on certain datasets, most of them rely on highly customized network architectures that lack universality. Such specificity necessitates substantial modifications or complete redesigns to accommodate new or varied layout design challenges. Recognizing this limitation, we develop a unified framework named PosterLLaVa (see Fig. 1) for layout generation task, inspired by the simplicity and effectiveness of the recently published multi-modal instruction tuning [4], [17]–[21] method. Pretrained with numerous amounts of unlabelled corpora and finetuned with instruction-following data, MLLMs (Multi-modal Large Language Models) are capable of handling multiple vision-language tasks (e.g., VQA [4], [18], visual grounding [20], [21], etc.) according to the given instructions and their knowledge. For layout generation, we first show how layout information can be represented by structure natural language in JSON format. With this representation, we can measure the performance of PosterLLaVa on established content-aware generation datasets and compare it with previous benchmarks. To tackle the multi-modal condition inputs, we utilize the pretrained visual head of LLaVa [22] to convert input image to representation adapted to textual token space and finetuning the LLM [23] to interpret and generate layout data. With the LLM as the central processing unit, our model can manage a wide range of layout generation tasks through simple modifications of the input instructions, eliminating any need for changes in model architecture. Moreover, user requirements presented by natural language can be seamlessly integrated into the generation instructions, enhancing the model’s responsiveness to specific design needs.

The main contribution of our work can be summarized as follows.

- 1) A Unified Layout Generation Tool We propose a unified content-aware layout generation method using multimodal LLMs, adaptable across various design scenarios through simple modifications of input instructions. Our approach is validated across multiple public datasets and two new datasets proposed in this paper, showcasing its superior performance and versatility.
- 2) New Datasets for Real-world Applications Recognizing the shortcomings of existing content-aware layout generation datasets in handling real-world demands, we collect data for two challenging tasks. The first is a graphic layout dataset named QB-Poster, composed of 5,188 samples designed with prevalence on Chinese social networks. This dataset is characterized by its intricate geometric relationships between sufficient types of content. The second is the UC-Poster, which is the largest poster layout dataset composed of 64,520 layouts, each constrained by both a background image and 3 textual user requirements. Through comparative analysis with the latest comparable method, our method demonstrates remarkable adaptability and effectiveness in capturing the distribution of complicated real-world layouts.
- 3) An Automatic Text-to-editable-poster System With

PotserLLaVa as the fundamental component, we build a novel pipeline named PosterGen to interpret the user’s intention and then create a visually appealing poster design to convey the key information. PosterGen utilizes the latest LLMs and T2I diffusion models. We provide experiments to prove its production quality.

II. RELATED WORK A. Automatic Graphic Layout Generation

Rule-based Methods Before the appearance of deep learning, layout generation has been studied for decades [1], [2], [24], [25]. Typically, Yin et al. [24] proposed a series of principles according to widely accepted aesthetic or informationconveying rules and a heuristic algorithm to minimize the overall energy function. These methods do not require training. Instead, they perform a runtime searching process during every inference. The true complexity of these methods lies in the design of the energy function, which requires a lot of design experience and expertise. Moreover, these functions must be manually re-designed when encountering a new design element or applied to a different styled layout (e.g., from UI to commercial poster).

Content-agnostic Layout Generation Neural networks offer researchers a way to formulate designing principles implicitly from numerous data, saving human efforts. Most early works [5], [7], [8], [26]–[28] focus on generating visually reasonable layouts for mobile UIs, documents, and magazine pages. LayoutGAN [26] employs the GAN (Generative Adversarial Network) paradigm and designs a differentiable rendering process for connecting the visual and graphic domains. LayoutVAE [5] and CanvasVAE [29] adopt the VAE (Variational Auto-Encoder) paradigm, while more recent works adopt the auto-regressive architecture [7]–[9] or the diffusion architecture [10], [28], [30]. Despite their achievement on unconditioned layout generation tasks, they are hard to use in real-world scenarios.

Content-aware Layout Generation Recently, some other works [11]–[14], [31] have paid their attention to commercialstyle posters, in which case the graphic designs are usually based on a non-empty background image. CGL-GAN [11] contributes a large dataset with around 60k Chinese commercial posters and proposes to learn with a transformer-based GAN network receiving a saliency map and the inpainted background as input. Similarly, PosterLayout [12] tackles the problem with a CNN-LSTM network with saliency map as input. [32] adopts a C-VAE (Conditional Variational AutoEncoder) to predict the layout. LayoutDETR [13] design a DETR-like [33] to utilize the pre-trained objects detection model and integrate both GAN and VAE for layout generation. They also include pre-trained ViT [34] and BETR [35] as visual and textual encoders to get embedded features of the design elements. Interestingly, some work [6], [15], [16] also attempted to generate layouts following specific constraints. Primitively, LayoutGAN++ [6] introduces an additional constrained optimization process based on the Lagrangian multiplier method to get the desired layout. Then, LayoutFormer++ [15] and Parse-then-place [16] design a specific

intermediate representation to handle various constraints. The latter also studies the text-to-layout problem, which includes implicitly expressed user requirements and is very similar to ours.

- B. Multi-modal Large Language Models and Application

LLMs (Large Language Models) [23], [36], [37] have achieved remarkable success across a wide range of natural language processing (NLP) tasks. With billions of parameters, these models derive extensive knowledge from pre-training on vast unlabeled text corpora. Various instruction-tuning methods have been investigated to enhance the ability of LLMs to comprehend and execute natural language instructions [38], [39]. While LLMs have proven adept at understanding and generating text, multi-modal LLMs have been facilitated by incorporating additional modalities like visual and auditory data [4], [17], [18]. A prevalent approach involves injecting LLMs with multi-modal information and leveraging their robust reasoning capabilities.

LLMs-assisted Layout Generation Layouts, which can be encoded in formats such as XML or JSON, are ideally suited to be processed by pre-trained Large Language Models (LLMs). Previous works have used domain-specific data to strengthen their code generation ability. LayoutNUWA [40] fine-tunes the LLaMa [41] and CodeLLaMa [42] to the content-agnostic layout generation task, achieving the SOTA performance in multiple content-agnostic layout datasets. LayoutPrompter [43] introduces an interesting training-free approach, leveraging RAG (Retrieval-Augmented Generation) to strengthen the incontext learning ability of GPT [36], dynamically sourcing examples from a dataset. However, this retrieval-centric strategy is limited to open-domain generation. These works overlook the visual domain feature or translate it into hard tokens before feeding into LLM, potentially resulting in severe information loss. To tackle this weakness, we include the latest proposed multi-modal technique - visual instruct tuning [4] to fine-tune a pre-trained large model, which accepts the visual information with a pre-trained and aligned visual adaptation head [44]. For the layout-to-image generation, interestingly, some contemporaneous work like LayoutGPT [45] and TextDiffuser-2 [46] also adopt LLMs, showing a promising production pipeline for LLM-based graphic design.

- C. Graphic Design Generation

Most of the papers [5], [11], [12], [26] mentioned previously have conducted research solely on layout generation, but the subsequent task i.e., from a layout to a complete graphic design still requires human effort. This includes theme-related images, copywriting, and font design. Considering this purpose, some early works [14], [47] proposed naive schemes. Given a text prompt, Text2poster [47] advises retrieving the background image with similar semantics and rendering the text element on it with the LSTM-generated layout and predefined font types, sizes, and colors. HPCVTG [14] develops a thumbnail generation system, given a video and its text description, it retrieves the frame with similar semantic meanings and renders the GPT-summarized text element with predefined

font types and colors, while layouts are optimized with the EA (Evolution Algorithm). Recently, glyph-conditioned diffusion models [48]–[52] have been utilized for scene-text generation or editing, showcasing a surprisingly better performance than vanilla stable diffusion. Nonetheless, most publicly available models face trouble when tackling text with numerous characters and small font sizes.

COLE [53] is the latest published work on generating whole-stage graphic layouts, which builds a system with diffusion model [54] and LLM layout planner [23]. However, it relies on a closed-sourced graphic design dataset, which requires large human effort to construct. The unavailability of the dataset and code has restricted the re-productivity of this well-designed method. For public availability considerations, OpenCOLE [55] reproduced and simplified COLE’s pipeline with a publicly available dataset (i.e. Crello [29]), offering an open-sourced alternative with compatible performance. Nevertheless, it suggests fine-tuning both the diffusion model and topography-LLM, still consumes considerable computational resources.

III. METHODOLOGY

- A. Multi-modal Layout Tokenization

Assuming that all complicated attributes and art styles have their default values, we can explicitly represent the information of a graphic design Lj by defining the position (xi,yi), size (hi,wi), and content Ii of every element. The position and size can be further expressed as bounding box format if rotation and irregular shapes are not involved. The class labels ci of elements are explicitly given to excavate the relationship between different kinds of elements. We got the following representation of a poster:

Lj = {(xi,yi,hi,wi),ci,Ii}Ni=0 (1) in which N represents the number of elements. For previous papers, most consider Lj as a numeric form, which means solving the problem in a continuous space. However, we design the following process to tokenize Lj and feed it into LLMs to predict the next token. First, we normalized the bounding box coordinates with the background width and height to facilitate multi-resolution generation. Each coordinate data value of the bounding box vector is truncated to K decimal places to avoid redundancy. For class label ci, we use the corresponding text label instead, for example {text,logo,underlay} regarding the PosterLayout [12] dataset. Finally, for image elements, Iimgi is encoded by a pre-trained vision header, which is composed of a ViT [34] encoder and a linear projection head, namely

h(Iimgi ) = WTCLIP(Iimgi ). (2)

and the content Itxt of the text element is inherently in a text format.

- B. Training Scheme

To facilitate the learning of tokenized layout data, we adopt the training scheme proposed by Liu et al. [4], i.e.,

USER: <image>

Please help me to place <N> foreground elements over the background of <resolution> to craft a <domain name>. Remember to avoid unbalance, overlap, misalignment, and occlusion of semantic-meaningful objects on the background image. Return the result by filling in the following JSON file while keeping the number and types of elements unchanged. The initial JSON is defined as: <masked json>, in which each design element is represented by a bounding box described as [left, top, right, bottom], and each coordinate is a contiguous number in 0-1. The user constraints are defined as: <constraints>, which should be adopted as compulsory design requirements.

ASSISTANT: Sure! Here is the design result: <json>.

TABLE I: Prompt template for applying visual instruction tuning on content-aware generation task. The placeholder tokens in bold type are replaced with specific information during training or inference.

the visual instruction tuning. The original paper, focusing on general visual-language tasks, recommends fine-tuning a pretrained LLM [41] by two phrases: 1. pre-training for feature alignment, and 2. end-to-end fine-tuning. The alignment phase usually requires numerous image-text pairs to adapt visual information into language space, and the fine-tuning phase requires relatively less data to acquire instruction-following outputs. Recognizing that the primary challenge in layout generation resides in decoding the semantic and geometric relationship between graphic elements, we streamline the training process by using the pre-trained linear projection layer to skip the feature alignment phase. This allows us to reduce training expenditure while maintaining comparable performance with the full-trained model.

C. Prompt Template

We introduce the following prompt template for the adopting end-to-end fine-tuning phase of visual instruction tuning in various content-aware layout generation tasks. The template is described in Tab. I. The pre-trained vision head converts the background image into soft tokens (as Eq. 2 shows) to get <image>. <N> is replaced with the exact number of design elements, and <resolution> is replaced with the canvas resolution. We use a domain indicator <domain name> to distinguish different tasks and datasets. For example, ”commercial poster” for CGL dataset and ”advertising banner” for ad banner dataset. The ground-truth layout information is expressed by textual representation through the process introduced in Sec. III-A and arranged in JSON format (as Fig. 1) to replace <json>. For human instruction, we delete bounding boxes and preserve the category labels to get the <masked json>. As for user-constrained generation tasks, the constraints are given as <constraints>.

IV. EXPERIMENT

Implementation Details Most experiments are conducted on 8 NVIDIA A10 GPUs and can be finished within 12 hours. The MLLM checkpoint adopted is the full-tuning 7B version of LLaVa-v1.5 [22], which is trained with LLaMa-2 [23] 7B as the base model with visual instruction tuning. For most of the following layout datasets, we fine-tune the MLLM with one epoch. But for the ad banners dataset, considering its tiny scale, we find that the model needs at least 3 epochs to converge. For the adaptation into the QB-Poster dataset, we adopt the pre-trained model on all training sets of Ad

TABLE II: An overall description of the content-aware layout generation datasets. QB-Poster is the complicated real-world poster dataset proposed in this paper, which outperforms previous datasets in both annotation categories and box numbers per poster.

Dataset Train Test Classes Boxes/img Total Boxes

CGL dataset 60,548 1,000 4 4.87 265,818 PosterLayout 9,974 905 3 4.73 47,024

Ad Banner 7,672 1,000 8 2.23 16,593

YouTube 10,000 1,000 3 5.88 67,223 QB-Poster 4,675 513 10 15.17 78,723

Banner, CGL, and PosterLayout as a starting point to enhance its performance. We increase the max token from 2048 to 4096 as the token length grows with the element number. For other training or inference hyper-parameters, we apply the default recipe recommended by LLaVa [4].

A. Result on Public Content-aware Layout Dataset

Dataset Description As mentioned in Section II-A, contentaware layout generation, is still in its early stages. For learning this task, a corresponding background image and the coordinates of ground-truth layout bounding boxes are needed. We extensively investigated datasets published in previous literature. Available public datasets are listed in Tab. II.

CGL dataset [11], one of the pioneering content-aware poster layout collections, comprises 60,548 training samples and 1,000 test samples collected from e-commerce platforms. The design elements are divided into 4 categories: logo, text, underlay, and embellishment. The class labels and the bounding boxes of the elements for each poster in the training set are manually annotated, while the test set only includes the background image. Techniques such as image inpainting [56] and saliency detection [57] are needed to obtain additional visual information. Recognizing the limitations of the CGL dataset, particularly its repetitive content and scarcity of complex layouts featuring over ten elements, Hsu et al. [12] introduce PosterLayout, offering 9,974 poster-layout pairs for training and 905 background images for testing. LayoutDETR [13] contributes an ad banner dataset with multi-modal information, containing 7,672 samples divided into training and testing subsets in a 9: 1 ratio. The background images are either from the Pitt Image Ads dataset or Google Image, and the bounding boxes, categories, and text content are automatically extracted

- TABLE III: Results comparison on PosterLayout dataset. Evaluations are conducted under PosterLayout’s [12] settings. Previous results are copied for comparison.

Methods

Content-aware Geometric

Uti⇑ Occ⇓ Rea⇓ Val ⇑ Ove ⇓ Ali ⇓ Undl ⇑ Unds ⇑ Ground-Truth 0.2222 0.1900 0.1522 0.9999 0.0001 0.0002 0.9965 0.9912

Content-aware Methods

CGL-GAN 0.2257 0.1546 0.1715 0.7066 0.0605 0.0062 0.8624 0.4043 DS-GAN [12] 0.2541 0.2088 0.1874 0.8788 0.0220 0.0046 0.8315 0.4320 LayoutPrompter [43] 0.2597 0.0992 0.1723 0.9992 0.0036 0.0036 0.8986 0.8802 PosterLLaVa(Ours) 0.2628 0.1649 0.1142 1.0000 7.7e-5 0.0002 1.0000 1.0000

- TABLE IV: Results comparison on CGL-GAN dataset. Evaluations are conducted under CGL-GAN’s [11] settings. Previous results are copied for comparison. † indicates that we apply BASNet [58] for saliency detection rather than PFPN [57] since the pre-trained link of the latter one expires.

Content-aware Geometric

Methods

Rcom ⇓Rshm ⇓Rsub ⇓Rove ⇑Rund ⇑Rali ⇑Rocc ⇑ Content-unaware Methods

LayoutTransformer [7] 40.92 21.08 1.310 0.0156 0.9516 0.0049 VTN [8] 41.77 22.21 1.323 0.0130 0.9628 0.0047 -

#### Content-aware Methods

ContentGAN [27] 45.59 17.08 1.143 0.0397 0.8626 0.0071 93.4 CGL-GAN [11] 35.77 15.47 0.805 0.0233 0.9359 0.0098 99.6 PDA-GAN [31] 33.55 12.77 0.688 0.0290 0.9481 0.0105 99.7 PosterLLaVa(Ours) 34.80 8.214 0.277† 2.4e-10 1.0000 0.0008 100

by OCR automatically. However, unlike CGL and PosterLayout, this dataset contains banners with multi-resolutions. The YouTube [14] dataset is another newly proposed dataset focusing on video thumbnail generation. The design elements are images and texts extracted from the original video, and the ground-truth layouts are generated using an evolution algorithm with hand-crafted design principles as objectives.

Evluation Metrics For a fair comparison with results published in previous papers, we first adopt the original evaluation measurements for each dataset. The metrics used are similar for CGL-dataset [11] and PosterLayout [12] dataset. The calculation of content-aware metrics is related to background or saliency image: the Rcom and Rea represent the readability of text elements; Rshm, Rsub, Occ represents the occlusion of semantic meaningful or saliency region on the background, while Uti indicates the utility of non-saliency region. The geometric metrics are only related to the predicted bounding boxes: Rove, Ove represents the overlap ratio; Rund, Undl and Unds indicates whether the underlays are correctly placed under texts, and Ali represents the alignment; Rocc and Val indicates the valid (e.g., non-empty) layout ratio. For Ad Banner [13] and YouTube [14] dataset, similarity metrics are included since the ground-truth layouts are available. This is achieved by measuring the FID (Frechet Inception Distance) or IoU between the generated layout/image and the corresponding ground-truth. VB in the YouTube dataset represents Visual Balance, which represents whether the overall placement is balanced. Please refer to the original papers for detailed explanations of these metrics.

Result Comparison The results presented in Tab. III, IV, V, and VI demonstrate that our method outperforms existing

- TABLE V: Results comparison on the ad banner dataset under LayoutDETR’s [13] settings. Results of previous methods are copied for comparison, among which PosterLLaVa achieves SOTA performance in all metrics except misalignment.

Methods

Similarity Geometric Layout FID⇓

Image FID⇓

IoU ⇑

DocSim ⇑

Overlap ⇓

Misalign (×10−2) ⇓ Ground-Truth - - - - 0.035 1.889

Content-unaware Methods

LayoutGAN++ [6] 4.25 28.40 0.163 0.130 0.104 0.759 READ 4.45 32.10 0.177 0.141 0.093 2.867 Vinci 38.97 58.12 0.104 0.143 0.243 0.271 LayoutTransformer [7] 5.47 39.70 0.080 0.115 0.127 3.632

Content-aware Methods

CGL-GAN [11] 4.69 30.50 0.154 0.127 0.116 1.191 ICVT [32] 12.54 30.11 0.163 0.137 0.423 0.682 LayoutDETR-VAE [13] 3.25 27.47 0.216 0.152 0.119 1.737 PosterLLaVa(Ours) 2.37 24.87 0.242 0.158 0.029 1.161

- TABLE VI: Results comparison on the Youtube dataset under HPCVTG’s [14] settings. Previous results are copied for comparison. PosterLLaVa shows promising performance in reducing overlap and saliency occlusion.

Similarity Geometric mIoU ⇑

Methods

FID ⇓

VB ⇓

Overlap ⇓

Misalign ⇓

Occlusion ⇓

Ground-Truth - - 0.93 6.29 1.55 5.88

#### Content-unaware Methods

LayoutGAN++ [6] 4.06 145.7 6.01 151.02 1.52 21.23 LayoutTransformer [7] 11.42 59.89 6.53 76.15 0.06 18.38

Content-aware Methods HPCVTG [14] 14.16 18.50 2.13 47.51 3.25 14.41 PosterLLaVa(Ours) 27.50 12.14 3.10 8.17 0.49 7.24

approaches, both content-unaware and content-aware, by a significant margin. In the Ad Banner dataset, our model exhibits improvements across all metrics except Misalign. For the PosterLayout dataset, our method markedly enhances geometric metrics, whereas LayoutPrompter [43] achieves a better trade-off between utility and occlusion. This is understandable because all previous methods incorporate additional input (i.e., saliency maps pre-processed by the saliency detector), while our method relies solely on the original background image. Similarly, in the CGL dataset, our method outperforms other approaches, particularly in geometric measurements. These results confirm the effectiveness of our method across various datasets and metrics.

B. Result on the Proposed Datasets

User-constrained Layout Generation Although contentaware layout generation has been a valuable step toward realworld applications, realistic graphic design problems often involve more conditionality. User constraint is one of them, usually including optional suggestions or mandatory opinions for graphic design products. These constraints, typically articulated in natural language, introduce even more complexity due to their potential ambiguity. As Section II-A mentioned, several previous works [6], [15], [16] have explored similar topics. Yet a comprehensive end-to-end solution that seamlessly

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

DS-GAN

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

LayoutPrompter

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

PosterLLaVa(Ours)

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

HPCVTG

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

PosterLLaVa(Ours)

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

LayoutPrompter

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

PosterLLaVa(Ours)

- Fig. 2: Qualitative results on the PosterLayout (top), Youtube (middle), and QB-Poster (bottom) datasets. PosterLLaVa achieves the highest overall generation quality on all three datasets.

integrates visual content with natural language constraints is still required. Our methodology, leveraging large multi-modal models, is inherently equipped to bridge this gap.

To this end, we propose a new dataset to validate the constrained generation ability of our approach. Firstly, we ask human annotators to write 3 user constraints according to the original poster layout in the CGL [11] validation set (6,006 samples), which are later used as test samples in this experiment. Then, with these high-quality human-annotated constraints serving as in-context learning examples, we utilize ChatGPT to generate constraints automatically. This approach enables us to expand our constraint dataset to include the entire training corpus of the CGL dataset and the PosterLayout

dataset, thereby assembling an enormous training dataset of 64,520 samples to mirror the diverse demands of real-world graphic design tasks. The synthesized user-constrained poster dataset is called UC-Poster for short.

A New Real-world Poster Dataset A notable limitation of existing content-aware datasets is their oversimplification. Typically, these datasets feature layouts with no more than 15 design elements, categorized into fewer than 5 types. Such simplicity falls short of conveying sufficient semantic information and mirroring the complexity of designs employed in real-world graphic designs.

To better align with the demands of real-life scenarios, we collect a new dataset named QB-Poster with a much more

TABLE VII: Results comparison on the QB-Poster (up) and UC-Poster dataset (down). In both datasets, PosterLLaVa wins the overall comparison, surpassing all previous methods by better trade-offs between different metrics.

Similarity Content-aware Geometric Constraint

Methods

Image FID⇓

IoU⇑ Uti⇑ Occ⇓ Rea⇓ Val ⇑ Ove ⇓ Ali ⇓ Undl ⇑ Unds ⇑ VB⇓ Vio⇓

#### QB-Poster dataset

DS-GAN [12] 85.19 0.0558 0.5048 0.4146 0.1995 1.0000 0.1541 0.0034 0.3094 0.1627 0.0287 CGL-GAN [11] 67.10 0.0373 0.2908 0.3904 0.1800 0.9959 0.1375 0.0040 0.3726 0.0600 0.0956 ICVT [32] 97.59 0.0231 0.1121 0.3629 0.1442 0.9599 0.4666 0.0018 0.4673 0.3617 0.2903 LayoutDM [10] 159.3 0.0144 0.2218 0.4096 0.1850 0.9980 0.2240 0.0003 0.4736 0.3618 0.1223 LayoutPrompter [43] 96.86 0.0195 0.2467 0.4504 0.1956 0.9509 0.0233 0.0004 0.2686 0.1501 0.2784 PosterLLaVa(Ours) 35.97 0.1996 0.2656 0.3377 0.1659 0.9949 0.0117 4.75e-5 0.9418 0.9141 0.1221 -

#### UC-Poster dataset

LayoutPrompter [43] 20.29 0.0961 0.2024 0.2846 0.1038 0.8512 0.0014 0.0018 0.3916 0.2906 0.0781 0.4130 PosterLLaVa(Ours) 3.823 0.1996 0.1751 0.0924 0.1000 0.9432 0.0014 0.0003 0.9962 0.9944 0.0662 0.1156

LayoutPrompter PosterLLaVa(Ours)

[Figure 56]

[Figure 57]

- 1. "All elements should be at the bottom of the background image."
- 2. "text_0 needs to be below logo_0. "
- 3. "All elements should be centered horizontally within the background image."

[Figure 58]

[Figure 59]

[Figure 60]

- 1. "All elements should be placed at the bottom of the background image."
- 2. "text_0 is below logo_0."
- 3. "underlay_0 needs to be able to override text_1."

[Figure 61]

[Figure 62]

[Figure 63]

- 1. "underlay_0 needs to be able to override element text_0."
- 2. "text_1 and text_2 should be centered horizontally within the background image."
- 3. "underlay_1 needs to be able to override element text_3."

[Figure 64]

- Fig. 3: Qualitative results on the User-constrained Poster dataset. The user requirement texts are shown on the left side, and the bold requirement means it was violated by either method.

complicated style. As shown in Tab. II, the elements per poster and geometric complexity of QB-Poster surpass other datasets significantly. This includes 5,188 poster-layout pairs, with 4,675 for training and 513 for testing. The dataset categorizes design elements into 10 categories: title, subtitle, item logo, item, item title, object, text background, decoration, frame, and text. These fine-grained class labels reveal the design pattern of elements and provide the algorithm with additional semantic information. Text elements are organized using a hierarchical classification to indicate their levels of importance. Meanwhile, visual elements are categorized as decoration, text background, object, and frame, which respectively identify decorative icons, underlays, semantically significant objects within background images, and the canvas area.

Baseline and Evaluation Metrics We include a wide range

of previous content-aware layout generation approaches to ensure a comparison on the proposed QB poster benchmark, including GAN-based [11], [12], VAE-based [32], diffusionbased [10], and LLM-based (empowered by in-context learning) [43]. To reproduce LayoutPrompter [43], we use the gpt3.5-turbo-instruct instead of the text-davinci-003 (advised by the original paper) since the latter has been abandoned by OpenAI. For comparison on the UC-Poster dataset, we only include LayoutPrompter since other methods cannot support textual user requirements as input. We extend the original LayoutPrompter by concatenating the extracted saliency bounding box and the constraint texts as inputs to take both background semantics and user requirements into consideration.

For evaluation metrics, we integrate the evaluation metrics used in public datasets. Namely, we adopt Image FID and the bounding box IoU from LayoutDETR [13] to measure similarity between generated layouts and the ground-truth layouts; utility, occlusion, and readability from PosterLayout [12] to measure content-aware quality; validity, overlap, alignment, underlay (loose), underlay (strict), and VB (visual balance) from PosterLayout and HPCVTG [14] to measure geometric quality. Most importantly, to measure whether the model follows the input constraints, we sample a subset (50 layouts) of the test set and ask human annotators to verify the average violation ratio of the constraint, marked as violation. The overall result is shown in Tab VII.

Result Comparison Our proposed PosterLLaVa generally outperforms others in comparisons on the QB-Poster and UCPoster datasets. In QB-Poster, though some traditional methods win in some particular metrics, our PosterLLaVa achieves the best overall trade-offs. In UC-Poster, our method surpasses LayoutPrompter by a clear margin.

The overall comparison shows that our method excels especially in similarity measurements, the utility/occlusion trade-off, and underlay measurements. Therefore, PosterLLaVa tends to place text boxes in the corresponding areas with background color and dialogue bubbles, roughly with an accuracy of 90%. The vision encoder enables this capability and is crucial for developing the poster generation system, PosterGen, which will be discussed in the subsequent section.

Theme Colors

Background

[Figure 65]

[Figure 66]

Background Depiction

poster background, cats, collage imagery, cat breeds, paw prints, playful motifs, sense of joy and celebration, event notice, ...

###### T2I Module

(HunyuanDiT, FLUX)

###### Design Intention

Create an Instagram advertisement for SVW Group's International Cat Day Festival at Reeves Hall. Encourage people to attend the festival by highlighting the chance to find a pet friend and offering lectures on pet protection and cat grooming workshops. Date and location information: Aug 8, 2018, London, Reeves Hall.

|x 3 : ['Cat Fes @ Reeves', 'Cat Day Fes!', 'Find Your Pet!'] x 3 : ['Aug 8. 18 London', 'Events & Adoption', 'Learn Love at SVW']<br><br>titlex 3 subtitle<br><br>Text Contents<br><br>Layout Module<br><br>(PosterLLaVa)|[Figure 67]| |
|---|---|---|
| | | |
| | | |

Analysis Module

Rendering Module

(GPT-4)

Layout

###### Font Attributes Color Adjustment

{'title': {'font': 'Pacifico', 'variant' : 'regular'}, 'subtitle': {'font': 'Lato', 'variant' : '300'}}

{'title': {'fill': '#0e0f20', 'sw' : 0, 'stroke': ' '}, 'subtitle': {'fill': '#822016', 'sw': 0, 'stroke': ' '}}

GPT4

(WCAG Standard)

Generated Desgin

[Figure 68]

- Fig. 4: The text-to-poster generation system PosterGen, in which graphic design generation can be decomposed into a) intention analysis, b) text-to-image background generation, c) content-aware layout generation, and d) text attributes generation (font, color, etc.). As illustrated, PosterGen can correctly interpret the user’s design intention to create high-quality background images and display key information in an attention-grabbing location.

V. ABLATION STUDIES

We design several ablation experiments to verify the necessity of our proposed method on the following dimensions. We assume that 1. Considering the small scale of the existing content-aware dataset (<100,000 samples), the generation performance of the model is positively correlated to the number of training samples and model size; 2. the multimodal information used should contribute to the generated layout quality. The ad banner dataset is selected for ablation because it is the most lightweight but still contains sufficient multi-modal information, and the metrics used are stable (in contrast, the reliability of utility and occlusion scores highly depends on the quality of saliency detection).

Result The result shown in Tab. VIII demonstrates the assumption proposed above. For extra training data, we apply the whole training set of CGL, PosterLayout, and ad banner datasets (78,194 samples in total) for fine-tuning, which improves all geometric measurements. Surprisingly, it also improves the similarity metrics except for Layout FID, which reveals the generality in content-aware generation datasets. Furthermore, the similarity measurement continues to increase by upgrading the pre-trained LLaVa model from 7B to 13B. For multi-modal information, we reduce the visual input (i.e., background image) and textual input (i.e., text element content), respectively, and both of these degrade the overall performance (with a slight improvement in overlap metric, probably because the reduction of information has lower the learning difficulty). These results together demonstrate the effectiveness of utilizing multi-modal large models in content-aware layout generation tasks, and with whose enormous learning capacity, the corresponding demand for more high-quality layout data.

VI. POSTERGEN: AN AUTOMATED TEXT-TO-EDITABLE POSTER SYSTEM WITH MULTI-LANGUAGE SUPPORT

In Section III, we propose PosterLLaVa to tackle the challenge of layout generation given the background image and element categories and validate its effectiveness on a wide range of datasets. However, in real-world applications, the bounding-box-based representation is rarely the final goal of automated art design, because layout generation is only a part

TABLE VIII: Ablation Studies conducted on ad banner dataset [13]. Results demonstrate the necessity of applying large models, large datasets, and multi-modal information in contentaware layout generation.

Similarity Geometric Layout FID⇓

Methods

Misalign (×10−2) ⇓

Image FID⇓

IoU ⇑

DocSim ⇑

Overlap ⇓

PosterLLaVa(Ours) 2.37 24.87 0.242 0.158 0.029 1.161 + extra training data 3.91 24.40 0.251 0.160 0.027 0.949 + 7B→13B LLM 2.78 23.86 0.262 0.156 0.026 1.676

- - textual info 2.98 25.14 0.225 0.115 0.021 1.522

- - visual info 8.27 40.59 0.092 0.115 0.020 2.193

of automotive art design. In real-world scenarios, the business pipeline often begins with vague design requirements (such as user intent, product descriptions, news, or event announcements) and ends with a complete poster design, which needs the teamwork of product managers, graphic designers, frontend programmers, etc. and costing a lot of time and human effort. Despite this pressing demand, existing works [10]–[12], [28], focusing on polishing the layout generation algorithm, provide limited insight into the downstream transition from layout to complete poster design. There is a severe lack of accessible open-source tools to facilitate this process and save human effort in the process of graphic designing.

Noticing this fact, we further developed a poster generation system (named PosterGen) based on PosterLLaVa to facilitate automated text-to-poster design. The overall framework of PosterGen is shown in Fig. 4. The system can be generally divided into 3 stages: 1) intention analysis: our system utilizes an analysis module (i.e., GPT-3.5 in our implementation) to analyze the background information and the intention given by the user. This step will refine these intentions into three groups of more detailed information: prompts for background generation; copy-writings and categories, namely the text elements to be presented on the design; and font suggestions, including font family and variant. 2) Background generation: subsequently, a pre-trained text-to-image diffusion model is adopted to generate a high-quality background using the refined prompts. 3) Layout generation: with the background

SD3 DALL-E3 COLE OpenCOLE

FLUX.1 Ours

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

- (a) Design a Facebook event cover for a Valentine's Day sweets sale with a collage of various cakes and desserts. The cover should emphasize the romantic theme and the discounts available, with the text 'Valentine's Sale' and 'up to 30% off' prominently displayed. Encourage customers to shop now.
- (b) Create a billboard advertisement for a luxury perfume sold on an eBay store named @storename.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

(c) Create an Instagram advertisement for a new baby formula brand called PINGUIN that emphasizes the importance of infant nutrition. The image features an adorable baby drinking from a bottle in pink.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- (d) Create an Instagram ad for a camping festival and hiking event. The ad should promote the event and provide information about registration including a website. In addition, offer tips for surviving while hiking.
- (e) Create a Facebook advertisement for a store selling natural essential oils that promote healing for the body and soul. Highlight the use of floral and plant ingredients in the product. Encourage viewers to visit the store.
- (f) Create an Instagram advertisement for SVW Group's International Cat Day Festival at Reeves Hall. Encourage people to attend the festival by highlighting the chance to find a pet friend and offering lectures on pet protection and cat grooming workshops. Date and location information: Aug 8, 2018, London, Reeves Hall.

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

- Fig. 5: The qualitative comparison on DESIGNERINTENSION [53] benchmark with recently proposed poster generation method COLE [53] and OpenCOLE [55], and text-to-image models [59]–[61] (GPT-4 based prompt augmentation are adopted as advised by [53]). Our method has better editability than vanilla T2I schemes and outperforms competitors with better background quality, text readability, and fewer training resources.

image, copywriting, and corresponding categories, we adopt our PosterLLaVa to produce reasonable layout suggestions. 4) Font color selection: we extract the theme color from the generated background with clustering and call the GPT-3.5 again for harmonious text color suggestions. These initial colors are then filtered and fine-tuned according to the lightness distribution of their corresponding background patches (see Appendix for details of WCAG standard-based color adjusting scheme) to improve readability. Our observation finds that font types can be directly inferred from the intention. Differently, these color selection steps are separated independently because they are related to multi-modal information and thus cannot be paralleled or integrated into step one like font selection. Assuming the copywriting doesn’t involve line breaks, the optimal font size can be inferred using the bonding box (containing size and location), text content, and font type. Finally, the rendering module combines all the above information to create a text-overlaid poster image.

A. Key Observations

Our system is built upon two key empirical observations about the zero-shot inference capability of large models: 1) some diffusion models (e.g., HunyuanDiT [63], FLUX

- [61]) can generate background images with placeholders to facilitate the text illustration step later, simply by inserting special tokens like ”poster background”, and ”blank-leaving” in prompts. 2) Some LLMs (e.g. GPT [37], LLaMa3 [64]) have preliminary knowledge of public-available fonts. This includes

distinguishing whether a given font belongs to the serif or sans-serif category, handwritten or printed style, and assessing the suitability of these categories for the current theme, which is sufficient to make reasonable font choices considering the given design intention and audiences’ preference. These two observations allow us to build our PosterGen system with PosterLLaVa as the only fine-tuned model, minimizing the need for expensive training procedures. In contrast, recently proposed approaches (COLE [53] and OpenCOLE [55] recommend fine-tuning both the diffusion model and the LLM to solve background generation and typography selection, requiring numerous computational and data resources. In the following experiment (see Fig. 7), we are surprised to observe that our zero-shot method even surpasses these methods by a clear margin concerning the three dimensions of graphics & images, design layout, and topography & color. This demonstrates that the quality of existing graphic design datasets is still far from expectations, and over-relying on them can lead to a degradation in models’ generation quality.

B. Qualitative Evaluation

Evaluation with DESIGNERINTENTION Samples The corresponding qualitative comparisons are shown in Fig.5, in which we select 6 intentions (a-f) and show the design results of 6 methods. Generally speaking, T2I methods have better image quality, but their drawbacks are also clear. As suggested by COLE [53], for the T2I methods, we apply

SD3 DALL-E3 OpenCOLE

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

- (e) Create a '浪漫情人节' card with a young couple in love. The card should have the message '你是我的今天和所有明天' to express love and affection.
- (f) Design a poster for a '返校滑板车促销', targeting parents who want to provide a '有趣安全的骑行' for their children to school. Offer a '六折促销' on the best scooters for kids.

- (a) Create a movie ticket announcing the screening of '午夜纽 约' by 'Seline媒体'. The ticket provides information about the movie '剧院位置', '时间', '座位号', and the '票价'. The ticket number is '编号: 0504-9807-6040'.
- (b) Design a Facebook event cover for a '情人节甜品特卖' with a collage of various cakes and desserts. The cover should emphasize the romantic theme and the discounts available, with the text '情人节特卖' and '高达七折优惠' prominently displayed. Encourage customers to shop now.

FLUX.1 Ours

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

- (g) Design a poster to promote a '慈善单车骑行' to '为人道主 义事业筹集资金和支持'. The event will be held on '6月26日' and the website for '注册和捐款' is 'www.funbike.com'.
- (h) Design a '豪华邮轮旅行' advertisement that promotes '全 包套餐'. Include the '公司名称' and '网站URL'.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

- Fig. 6: The qualitative comparison of Chinese text illustration, showcasing the multi-lingual adaptability of the proposed method. COLE [53] is excluded due to its public inaccessibility. Latest T2I methods [60]–[62] still cannot produce meaningful Chinese characters; OpenCOLE [55] fails to render since the fonts they included cannot support Chinese characters; while our method stably generates high-quality samples.

GPT-4 augmentation to enrich and expand the original design intention into detailed prompts. This method enriches the painting details. However, this method sometimes results in pure images without text like in cases b, c, and e; or, strange image compositions and unnecessary scenes, like in cases b and e. Remarkably, DALLE-3 often generates overcomplicated design layouts, like in case a, d, and f, distracting the users’ attention. Different from SD-XL [59], the SOTA T2I methods [60]–[62], can produce images with clear English titles now, but the small print content is still unrecognizable (like DALLE-3 in case f) or inundated with non-sense typos (like SD3 and FLUX in case d and f). In contrast, the text-to-editable-graphic-design methods are distinctively more readable. The text color selection of COLE and OpenCOLE is stiffness, with black and white appearing in a dominantly large possibility. OpenCOLE chooses extremely thin font types and small font sizes, and meanwhile, the text is often placed out of the placeholder (like in case a and d), further reducing its readability. Our method generally outperforms all the baselines with high image quality and style consistency, reasonable and readable font type/color selection, and attractive layout design with all text elements placed accordingly - benefited by our proposed PosterLLaVa model trained on numerous highquality layout data.

Evaluation of Chinese Samples One distinct advantage of our method is its multilingual ability, which is a benefit of exploring the zero-shot font selection ability of GPT. We show some qualitative examples in Fig 6 for comparison by replacing the copywriting (but NOT the whole intention)

with Chinese translations to create posters illustrated with Chinese characters. COLE [53] is excluded because of the inaccessibility of source code to reproduce their method. OpenCOLE [55] includes 753 types of fonts, but all with Latin character sets, causing the garbled characters when trying to render Chinese. Though supporting English characters to some extent, the T2I method fails to generate recognizable and meaningful Chinese characters. This is due to the difference in the total number of different languages and has been discussed in previous works [48], [51]. Another interesting phenomenon is that due to data distribution, these T2I methods sometimes cannot distinguish whether the user wants Chinese glyphs or Chinese styles, like SD3 and FLUX in case a, as well as SD3 and DALLE-3 in case c - in which we are trying to illustrate themes of western concepts but with Chinese characters. Our method undoubtedly wins in the multi-lingual comparison by producing high-quality Chinese posters aligned with various kinds of intentions. For instance, in case f, it chooses to use a cute font to align with the theme of children’s product promotion.

C. Quantitative Evaluation

GPT-4o Evaluations The evaluation of graphic design is nontrivial, due to its subjectivity and non-uniqueness. COLE [53] provides the DESIGNERINTENTION benchmark, which contains 500 textual design intentions from different areas (e.g., ad, poster, social media). OpenCOLE [55] opensource the evaluation code using GPT-4 and suggests evaluating on a 200-sample subset of the benchmark. We generally follow

DeepFloyd IF

Stable-Diffusion 3

FLUX.1 dev

###### COLE

###### OpenCOLE

PosterGen(Ours)

Content Relevance Typography Color

Content Relevance Typography Color

Content Relevance Typography Color

Content Relevance Typography Color

Content Relevance Typography Color

Content Relevance Typography Color

8.54

8.34

8.06 7.35

7.90 7.68

7.72 7.27

10.0

10.0

10.0

10.0

10.0

10.0

7.71

9.0

9.0

9.0

9.0

9.0

9.0

7.12

6.83 6.39

8.0

8.0

8.0

8.0

8.0

8.0

7.0

7.0

7.0

7.0

7.0

7.0

Design Layout

Design Layout

Design Layout

Design Layout

Design Layout

Design Layout

7.51

8.15

8.41

7.90

7.93

8.37

6.78

6.92

7.64

7.07

7.18

7.99

7.40

7.41

8.36

8.40

8.69

8.71

Graphics Images

Graphics Images

Graphics Images

Graphics Images

Graphics Images

Graphics Images

Innovation

Innovation

Innovation

Innovation

Innovation

Innovation

- Fig. 7: The quantitative comparison (with GPT-4o evaluated scores) between SOTA text-to-image commercial models [60]–

- [62], SOTA graphic designer generation approaches [53], [55] and our method on DESIGNERINTENSION [53] benchmark. The proposed PosterGen generally outperforms others, with the simplest pipeline and lowest training efforts.

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

- Fig. 8: The quantitative comparison on DESIGNERINTENSION [53] benchmark through human-annotated user studies. We sample different intentions and methods to create pair-wise comparison and compute the ultimate score with Elo [65] rating mechanism.

the evaluation pipeline advised by OpenCOLE: render graphic designs into images and use GPT-4o [37] to evaluate the quality of the graphic designs to improve efficiency and save expenses. The comparison includes five dimensions: a) design and layout, b) content relevance and effectiveness,

- c) typography and color scheme, d) graphic and images, and e) innovation. COLE and OpenCOLE are included as SOTA graphic design generation methods for comparison, T2I methods with enhanced text rendering ability [60]–[62], [66] are also taken into consideration, even though they cannot produce veritable graphic designs, but generate non-editable picture prototypes instead. According to the experiment result shown in Fig 7, PosterGen wins the comparison of all five dimensions among text-editable methods (shown in bold font), especially in graphics and images, layout, and typography. Our method achieved a notable high score (that is, 8.54) in terms of content relevance, indicating its ability to analyze intent and user preferences. As a pure T2I model, FLUX performs surprisingly well considering layout and topography, which are, unfortunately, encoded in pixel images implicitly and cannot be easily extracted or edited. All included methods have room for improvement in terms of innovation. Noted that we ignore DALLE3 in Fig/7 it’s hard to use GPT-4o to evaluate images generated by DALLE3 [60], which seems to have some inner specific safety checker and fails to decode with a possibility of over 80%.

User Studies We extend the quantitative evaluation by conducting human-based user studies. We hire two groups of people on the crowd-sourcing platform, including 15 annotators. They are divided into two groups, 5 are with graphic design experience and 10 are not. Besides, different from giving absolute scores ranging from 1 to 10 as GPT did, we randomly sampled image pairs and asked human subjects to compare and choose the winner. We believe the pair-wise comparison is more intuitive for humans to perform. Finally, we compute the Elo [65] score for each model regarding 5 dimensions, related details are discussed in the appendix. As illustrated in Fig. 8, we can have the following observations: 1. Due to numerous high-quality training data, the overall quality of the latest T2I method is conspicuously superior to the graphic design generation method, among which FLUX is the most remarkable one, winning in most comparisons. 2. The average quality of our method is better than DeepFloyd IF [66], and approximately compatible with SD3 [62] and DALLE-3 [60], which constitutes the best average-quality graphic design generation method. Moreover, its information communication is extremely prominent, even outperforming FLUX [61] in both the design and non-design groups; its layout quality surpasses SD3 and DALLE3 in the non-design group evaluation, which reflects the effectiveness of our proposed layout generation model and text attributes polishing algorithm. 3. Regardless of the dimension, the design group

has a greater preference for the T2I method than the nondesign group, which indicates the necessity of collecting more high-quality graphic design data and enhancing the generation pipeline. Generally speaking, this result is in accord with the previous GPT evaluation (more analysis are included in the appendix).

VII. CONCLUSION

Content-aware layout generation is a highly multi-modal problem. Utilizing the latest multi-modal large model instruction fine-tuning techniques, we propose a method named PoserLLaVa that represents multi-modal layout information as tokens, which are then processed by a Large Language Model (LLM). The proposed method achieves SOTA performance across multiple content-aware layout generation datasets. Additionally, by surveying existing content-aware layout generation datasets, we identify significant shortcomings in the current public datasets, namely the lack of user-constrained data and complicated data, both of which are crucial in realworld applications. We further collect two new datasets to bridge this gap, the user-constrained poster dataset and the QB-Poster, based on which we verify the extended ability of our method. In summary, to achieve large-scale automated production, high-quality multi-modal layout data and a unified learning approach are still under demand, for which our method paves the way.

REFERENCES

- [1] M. Rajasekharan, B. A. Peters, and T. Yang, “A genetic algorithm for facility layout design in flexible manufacturing systems,” International journal of Production research, vol. 36, no. 1, pp. 95–110, 1998.
- [2] A. Baykaso˘glu and N. N. Gindy, “A simulated annealing algorithm for dynamic layout problem,” Computers & Operations Research, vol. 28, no. 14, pp. 1403–1426, 2001.
- [3] J. Macedo, D. Lopes, J. Correia, P. Machado, and E. Costa, “Evolving visually-diverse graphic design posters,” in International Conference on Computational Intelligence in Music, Sound, Art and Design (Part of EvoStar). Springer, 2024, pp. 265–278.
- [4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” arXiv preprint arXiv:2304.08485, 2023.
- [5] A. A. Jyothi, T. Durand, J. He, L. Sigal, and G. Mori, “Layoutvae: Stochastic scene layout generation from a label set,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9895–9904.
- [6] K. Kikuchi, E. Simo-Serra, M. Otani, and K. Yamaguchi, “Constrained graphic layout generation via latent optimization,” in Proceedings of the 29th ACM International Conference on Multimedia, 2021, pp. 88–96.
- [7] K. Gupta, J. Lazarow, A. Achille, L. S. Davis, V. Mahadevan, and A. Shrivastava, “Layouttransformer: Layout generation and completion with self-attention,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1004–1014.
- [8] D. M. Arroyo, J. Postels, and F. Tombari, “Variational transformer networks for layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13642–13652.
- [9] X. Kong, L. Jiang, H. Chang, H. Zhang, Y. Hao, H. Gong, and I. Essa, “Blt: bidirectional layout transformer for controllable layout generation,” in European Conference on Computer Vision. Springer, 2022, pp. 474– 490.
- [10] N. Inoue, K. Kikuchi, E. Simo-Serra, M. Otani, and K. Yamaguchi, “Layoutdm: Discrete diffusion model for controllable layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10167–10176.
- [11] M. Zhou, C. Xu, Y. Ma, T. Ge, Y. Jiang, and W. Xu, “Compositionaware graphic layout gan for visual-textual presentation designs,” arXiv preprint arXiv:2205.00303, 2022.

- [12] H. Y. Hsu, X. He, Y. Peng, H. Kong, and Q. Zhang, “Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6018–6026.
- [13] N. Yu, C.-C. Chen, Z. Chen, R. Meng, G. Wu, P. Josel, J. C. Niebles, C. Xiong, and R. Xu, “Layoutdetr: Detection transformer is a good multimodal layout designer,” arXiv preprint arXiv:2212.09877, 2022.
- [14] T. Yang, F. Wang, J. Lin, Z. Qi, Y. Wu, J. Xu, Y. Shan, and C. Chen, “Toward human perception-centric video thumbnail generation,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 6653–6664.
- [15] Z. Jiang, J. Guo, S. Sun, H. Deng, Z. Wu, V. Mijovic, Z. J. Yang, J.G. Lou, and D. Zhang, “Layoutformer++: Conditional graphic layout generation via constraint serialization and decoding space restriction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18403–18412.
- [16] J. Lin, J. Guo, S. Sun, W. Xu, T. Liu, J.-G. Lou, and D. Zhang, “A parse-then-place approach for generating graphic layouts from textual descriptions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 23622–23631.
- [17] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” arXiv preprint arXiv:2301.12597, 2023.
- [18] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.
- [19] R. Zhang, J. Han, A. Zhou, X. Hu, S. Yan, P. Lu, H. Li, P. Gao, and Y. Qiao, “Llama-adapter: Efficient fine-tuning of language models with zero-init attention,” arXiv preprint arXiv:2303.16199, 2023.
- [20] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi et al., “mplug-owl: Modularization empowers large language models with multimodality,” arXiv preprint arXiv:2304.14178, 2023.
- [21] W. Wang, Z. Chen, X. Chen, J. Wu, X. Zhu, G. Zeng, P. Luo, T. Lu, J. Zhou, Y. Qiao et al., “Visionllm: Large language model is also an open-ended decoder for vision-centric tasks,” arXiv preprint arXiv:2305.11175, 2023.
- [22] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” arXiv preprint arXiv:2310.03744, 2023.
- [23] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.
- [24] W. Yin, T. Mei, and C. W. Chen, “Automatic generation of social media snippets for mobile browsing,” in Proceedings of the 21st ACM international conference on Multimedia, 2013, pp. 927–936.
- [25] S. Ma and C. W. Chen, “Automatic creation of magazine-page-like social media visual summary for mobile browsing,” in 2016 IEEE International Conference on Image Processing (ICIP). IEEE, 2016, pp. 469–473.
- [26] J. Li, J. Yang, A. Hertzmann, J. Zhang, and T. Xu, “Layoutgan: Generating graphic layouts with wireframe discriminators,” arXiv preprint arXiv:1901.06767, 2019.
- [27] X. Zheng, X. Qiao, Y. Cao, and R. W. Lau, “Content-aware generative modeling of graphic design layouts,” ACM Transactions on Graphics (TOG), vol. 38, no. 4, pp. 1–15, 2019.
- [28] J. Zhang, J. Guo, S. Sun, J.-G. Lou, and D. Zhang, “Layoutdiffusion: Improving graphic layout generation by discrete diffusion probabilistic models,” arXiv preprint arXiv:2303.11589, 2023.
- [29] K. Yamaguchi, “Canvasvae: Learning to generate vector graphic documents,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5481–5489.
- [30] M. Hui, Z. Zhang, X. Zhang, W. Xie, Y. Wang, and Y. Lu, “Unifying layout generation with a decoupled diffusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1942–1951.
- [31] C. Xu, M. Zhou, T. Ge, Y. Jiang, and W. Xu, “Unsupervised domain adaption with pixel-level discriminator for image-aware layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10114–10123.
- [32] Y. Cao, Y. Ma, M. Zhou, C. Liu, H. Xie, T. Ge, and Y. Jiang, “Geometry aligned variational transformer for image-conditioned layout generation,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 1561–1571.
- [33] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision. Springer, 2020, pp. 213– 229.

- [34] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [35] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.
- [36] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.
- [37] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [38] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27730–27744, 2022.
- [39] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language model with self generated instructions,” arXiv preprint arXiv:2212.10560, 2022.
- [40] Z. Tang, C. Wu, J. Li, and N. Duan, “Layoutnuwa: Revealing the hidden layout expertise of large language models,” arXiv preprint arXiv:2309.09506, 2023.
- [41] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [42] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin et al., “Code llama: Open foundation models for code,” arXiv preprint arXiv:2308.12950, 2023.
- [43] J. Lin, J. Guo, S. Sun, Z. J. Yang, J.-G. Lou, and D. Zhang, “Layoutprompter: Awaken the design ability of large language models,” arXiv preprint arXiv:2311.06495, 2023.
- [44] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [45] W. Feng, W. Zhu, T.-j. Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang, “Layoutgpt: Compositional visual planning and generation with large language models,” arXiv preprint arXiv:2305.15393, 2023.
- [46] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser2: Unleashing the power of language models for text rendering,” arXiv preprint arXiv:2311.16465, 2023.
- [47] C. Jin, H. Xu, R. Song, and Z. Lu, “Text2poster: Laying out stylized texts on retrieved images,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 4823–4827.
- [48] J. Ma, M. Zhao, C. Chen, R. Wang, D. Niu, H. Lu, and X. Lin, “Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation,” arXiv preprint arXiv:2303.17870, 2023.
- [49] Y. Yang, D. Gui, Y. Yuan, W. Liang, H. Ding, H. Hu, and K. Chen, “Glyphcontrol: Glyph conditional control for visual text generation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [50] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser: Diffusion models as text painters,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [51] Y. Tuo, W. Xiang, J.-Y. He, Y. Geng, and X. Xie, “Anytext: Multilingual visual text generation and editing,” arXiv preprint arXiv:2311.03054, 2023.
- [52] Z. Liu, W. Liang, Z. Liang, C. Luo, J. Li, G. Huang, and Y. Yuan, “Glyph-byt5: A customized text encoder for accurate visual text rendering,” arXiv preprint arXiv:2403.09622, 2024.
- [53] P. Jia, C. Li, Z. Liu, Y. Shen, X. Chen, Y. Yuan, Y. Zheng, D. Chen, J. Li, X. Xie et al., “Cole: A hierarchical generation framework for graphic design,” arXiv preprint arXiv:2311.16974, 2023.
- [54] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [55] N. Inoue, K. Masui, W. Shimoda, and K. Yamaguchi, “Opencole: Towards reproducible automatic graphic design generation,” arXiv preprint arXiv:2406.08232, 2024.
- [56] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky,

- “Resolution-robust large mask inpainting with fourier convolutions,” arXiv preprint arXiv:2109.07161, 2021.
- [57] B. Wang, Q. Chen, M. Zhou, Z. Zhang, X. Jin, and K. Gai, “Progressive feature polishing network for salient object detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 12128–12135.
- [58] X. Qin, Z. Zhang, C. Huang, C. Gao, M. Dehghan, and M. Jagersand, “Basnet: Boundary-aware salient object detection,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
- [59] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [60] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo et al., “Improving image generation with better captions,” Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, vol. 2, no. 3, p. 8, 2023.
- [61] B. F. Labs, “Flux.1,” https://github.com/black-forest-labs/flux, 2024, accessed: 2024-10-03.
- [62] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first International Conference on Machine Learning, 2024.
- [63] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long, X. Deng, Y. Zhang, X. Liu, M. Huang, Z. Xiao et al., “Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding,” arXiv preprint arXiv:2405.08748, 2024.
- [64] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.
- [65] A. E. Elo and S. Sloan, The rating of chessplayers : past and present. Ishi Press International, 2008. [Online]. Available: https://cir.nii.ac.jp/crid/1130282270181653248
- [66] D. Lab, “Deepfloyd if,” https://github.com/deep-floyd/IF, 2023.

