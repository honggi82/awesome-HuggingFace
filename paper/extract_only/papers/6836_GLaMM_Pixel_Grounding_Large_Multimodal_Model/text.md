## GLaMM: Pixel Grounding Large Multimodal Model

# arXiv:2311.03356v3[cs.CV]2Jun2024

Hanoona Rasheed1*, Muhammad Maaz1*, Sahal Shaji1, Abdelrahman Shaker1, Salman Khan1,2 Hisham Cholakkal1, Rao M. Anwer1,3, Eric Xing1,4, Ming-Hsuan Yang5,7, Fahad S. Khan1,6

1Mohamed bin Zayed University of AI, 2Australian National University, 3Aalto University 4Carnegie Mellon University, 5University of California - Merced, 6Linköping University, 7Google Research

hanoona.bangalath@mbzuai.ac.ae, muhammad.maaz@mbzuai.ac.ae https://github.com/mbzuai-oryx/groundingLMM, https://grounding-anything.com

### Abstract

[Figure 1]

Describe the image in detail. Please output interleaved segmentation mask

[Figure 2]

[Figure 3]

Large Multimodal Models (LMMs) extend Large Language Models to the vision domain. Initial LMMs used holistic images and text prompts to generate ungrounded textual responses. Recently, region-level LMMs have been used to generate visually grounded responses. However, they are limited to only referring to a single object category at a time, require users to specify the regions, or cannot offer dense pixel-wise object grounding. In this work, we present Grounding LMM (GLaMM), the first model that can generate natural language responses seamlessly intertwined with corresponding object segmentation masks. GLaMM not only grounds objects appearing in the conversations but is flexible enough to accept both textual and optional visual prompts (region of interest) as input. This empowers users to interact with the model at various levels of granularity, both in textual and visual domains. Due to the lack of standard benchmarks for the novel setting of visually Grounded Conversation Generation (GCG), we introduce a comprehensive evaluation protocol with our curated grounded conversations. Our proposed GCG task requires densely grounded concepts in natural scenes at a large-scale. To this end, we propose a densely annotated Grounding-anything Dataset (GranD) using our proposed automated annotation pipeline that encompasses 7.5M unique concepts grounded in a total of 810M regions available with segmentation masks. Besides GCG, GLaMM also performs effectively on several downstream tasks, e.g., referring expression segmentation, image and region-level captioning and vision-language conversations.

The image showcases a large, white building with a red roof, surrounded by a well-manicured lawn and palm trees. The sky is visible over the building, the pavement, and the grass. The grass is also seen extending to the pavement.

Figure 1. Grounded Conversation Generation with GLaMM. Our multimodal conversational model can provide natural language responses grounded at the pixel-level in the input image. Different levels of granularity are depicted in the output groundings, e.g., things (building, tree), stuff (grass, sky, pavement), and object parts (roof as a subpart of the building) alongside the object attributes (white house, red roof, well-manicured lawn) and object relationships (grass extending to the pavement, sky over the building). Existing LMMs, open-source (e.g., LLaVa, miniGPT4, Shikra, Kosmos-2) and closed-source (e.g., GPT4-V, Bard), do not offer pixel-level grounded conversational capability.

forts like [6, 8, 22, 29, 52, 61] demonstrate effective textual responses based on input images. Although these models are sophisticated, they cannot still ground their responses in the visual context. Such grounding is crucial for advanced applications like detailed visual understanding, interactive embodied agents, and localized content manipulation. Recent efforts have started to address this limitation by enabling models to process user-defined regions specified via bounding boxes [5, 31, 35, 36, 57].

### 1. Introduction

Fueled by the generative AI wave, Large Multimodal Models (LMMs) have emerged as a pivotal advancement, bridging the gap between vision and language tasks [2]. Initial ef-

*Equal contribution.

A few recent works have explored grounded text response generation [5, 21, 35, 59] but do not provide detailed pixel-level groundings. Parallel to these, efforts have been made in the referring segmentation literature to ground textual descriptions in natural images [21]. However, they are limited to grounding a single object and cannot engage in natural, coherent conversations, thereby restricting their practical applicability in interactive tasks that demand a deep understanding of both visual and textual content. To address these limitations of existing works, we introduce Grounding LMM (GLaMM), that simultaneously provides in-depth region understanding, pixel-level groundings, and conversational abilities through an end-to-end training approach (see Fig. 1 and Tab. 1).

To address the lack of benchmarks for visually grounded conversations, we introduce the novel task of Grounded Conversation Generation (GCG). The GCG task aims to produce natural language responses interleaved with object segmentation masks. This challenging task unifies several existing tasks in computer vision that are typically treated in isolation, i.e., referring expression segmentation, image and region-level captioning, phrase grounding, and vision-language conversations. Thereby, our unified model and proposed pretraining dataset can effectively transfer to several downstream tasks (referring expression segmentation, region-level captioning, image captioning, and conversational-style QA). We present GLaMM as the first model specifically designed for this challenging task. Unlike prior works, GLaMM can work with both textual and visual prompts and can generate visually grounded outputs, thus offering a versatile user experience.

Detailed region-level understanding requires the laborious process of collecting large-scale annotations for image regions. We propose an automated pipeline to annotate the large-scale Grounding-anything Dataset (GranD) to alleviate the manual labeling effort. Leveraging the automated pipeline with dedicated verification steps, GranD comprises 7.5M unique concepts anchored in 810M regions, each with a segmentation mask. Using state-ofthe-art vision and language models, the dataset annotates SAM [18] images through a multi-level hierarchical scheme that enhances annotation quality. With 11M images, 84M referring expressions, and 33M grounded captions, GranD sets a new benchmark in comprehensiveness. In addition to the automatically generated dataset for the GCG, we provide the first high-quality dataset for grounded conversations obtained by revamping the existing manually annotated datasets [16, 37, 49] for GCG using GPT-4 [34] incontext learning. We refer to the high-quality dataset as GranDf, denoting its suitability for fine-tuning.

Our work has three main contributions:

• We present GLaMM, the first model capable of generating natural language responses seamlessly integrated

with object segmentation masks. Unlike existing models, GLaMM accommodates textual and visual prompts, facilitating enhanced multimodal user interaction.

- • Recognizing the lack of standardized benchmarks for visually grounded conversations, we propose the new Grounded Conversation Generation (GCG) task. We also introduce a comprehensive evaluation protocol to measure the efficacy of models for GCG that unifies multiple isolated tasks, filling a significant gap in the literature.
- • To facilitate model training and evaluation, we create Grounding-anything Dataset (GranD), a large-scale densely annotated dataset. Developed using an automatic annotation pipeline and verification criteria, it encompasses 7.5M unique concepts grounded in 810M regions.

Additionally, we propose GranDf, a high-quality dataset explicitly designed for the GCG task finetuning, by repurposing existing open-source datasets.

### 2. Related Work

LMMs provide a versatile interface for a diverse array of tasks, encompassing language and vision. Prominent models such as BLIP-2 [24], LLaVA [29], InstructBLIP [6] and MiniGPT-4 [61] first conduct image-text feature alignment followed by instruction tuning. Other representative works include Otter [22], mPLUG-Owl [52], LLaMaAdapter [56], Video-ChatGPT [32], InternGPT [31]. However, these approaches lack region-specific understanding.

Recent works like Kosmos-2 [35], Shikra [5], GPT4RoI [57], VisionLLM [44], Ferret [53] and All-Seeing [45] aim to allow region-specific conversation. Some methods [5, 35, 45, 53] input location bins and bounding boxes with image data for region-level understanding, relying on the LLM exclusively for interpreting these regions. GPT4RoI advances this by using spatial boxes and RoI-aligned features for input and training on region-text pairs. BuboGPT [59] utilizes an off-the-shelf grounding model [30] and matches the groundings with the language response. In contrast, LISA [21] utilizes embeddings from the vision language model and the SAM [18] decoder to generate output segmentation masks. However, LISA cannot comprehend specific image regions or handle multiple instances.

To classify the LMM landscape, methods can be partitioned into four distinct categories (see Tab. 1 - separated via dotted lines). The first encompasses models effective in textual responses but lacking in region-specific capabilities [6, 8, 22, 29, 51, 52, 61]. In contrast, among models that handle region inputs or offer visual grounding, three more categories emerge. The first of these incorporates external vision modules [31, 59], and the next relies exclusively on LMMs for region understanding [5, 35, 36, 44]. The last category combines specialized vision modules with LMMs, trained end-to-end for a comprehensive understanding of regions [21, 45, 57]. Our approach belongs to the

Input / Output

Method Image

Region Pixel-Wise Multi-turn End-End Region Multi-Region Enc. / Dec. Grounding Conversation Model

MM-REACT (arXiv-23) [51] / / / LLaVA (NeurIPS-23) [29] / / / miniGPT4 (arXiv-23) [61] / / / mPLUG-OWL (arXiv-23) [52] / / / LLaMA-Adapter v2 (arXiv-23) [8] / / / Otter (arXiv-23) [22] / / / Instruct-BLIP (arXiv-23) [6] / / / InternGPT (arXiv-23) [31] / / / Bubo-GPT (arXiv-23) [59] / / / Vision-LLM (arXiv-23) [44] / / / Det-GPT (arXiv-23) [36] / / / Shikra (arXiv-23) [5] / / / Kosmos-2 (arXiv-23) [35] / / / GPT4RoI (arXiv-23) [57] / / / ASM (arXiv-23) [45] / / / LISA (arXiv-23) [21] / / / GLaMM (ours) / / /

Table 1. Comparison of recent Large Multimodal Models (LMMs) emphasizing their capabilities for region-level understanding. The Input denotes models that can process regions defined by users via bounding boxes, with Multi-Region indicating models that can handle multiple such regions. The Output represents models capable of delivering grounded responses. While some methods employ external vision modules for region understanding, others rely solely on the LMM, which may result in imprecise localization. However, a few integrate specialized vision modules and LMMs, as indicated by the Region Enc./Dec.. The End-End Model distinction separates models that leverage LMMs for region understanding from those employing external modules. Pixel-wise Grounding highlights models that can respond with segmentation masks, and Multi-turn Conversation represents models that can hold an interactive dialogue with the user. Among these, our proposed GLaMM stands out by offering comprehensive region understanding, pixel-wise grounding in its responses, conversational capabilities, and an end-to-end training approach.

last category and distinctly offers pixel-level grounding together with multi-turn conversations and the flexibility to operate on both input images and specific regions. Further, we provide large-scale instance-level grounded visual understanding dataset that allows generalizability of GLaMM to multiple vision-language tasks.

### 3. Method

Existing Large Multimodal Models (LMMs) either generate ungrounded text or are restricted by limitations such as single-object grounding, user-specified region inputs, or the lack of dense pixel-level object grounding (see Tab. 1). Our Grounding LMM (GLaMM) aims to overcome these limitations by generating natural language responses seamlessly integrated with object segmentation masks. This enables a visually grounded human-machine conversation.

#### 3.1. GLaMM Architecture

GLaMM consists of five core components: i) Global Image Encoder, ii) Region Encoder, iii) LLM, iv) Grounding Image Encoder, and v) Pixel Decoder. These components are cohesively designed to handle both textual and optional visual prompts (image level and region), allowing for interaction at multiple levels of granularity and generating

grounded text responses (Fig. 2). These blocks together enable scene-level, region-level, and pixel-level grounding, as explained next. Refer Appendix A.2 for training details.

Scene-Level Understanding: To achieve a holistic understanding of the scene, we employ ViT-H/14 CLIP [38] as our global image encoder (I), in conjunction with a vicunabased LLM (L) and a vision-to-language (V-L) projection layer (f). Specifically, given an image ximg and a text instruction xt, the image is first encoded into a feature vector Ix = I(ximg) ∈ RD

v and projected to language space f(Ix) ∈ RD

t. The LLM then integrates both the projected image features and the text instruction to generate output yt:

yt = L f(Ix),xt .

This maps image features to language space, enabling GLaMM to offer holistic scene understanding, achieved through specific prompts like, “The <image> provides an overview of the image. Could you please give me a detailed description of the image?” The <image> token is replaced with 256 tokens from the CLIP global image encoder.

Region-Level Understanding: Building on the shortcomings of existing models that can handle only image-level visual inputs, and in alignment with recent work [57], the

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

| | |[Figure 8]|
|---|---|---|
| | | |

Grounding Image Encoder

Global Image Encoder

Pixel Decoder

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

| |
|---|

A large hot air balloon

[Figure 16]

LLM

[Figure 17]

Region Encoder

V-L L-P

[Figure 18]

A large body of water

Image-region prompts Output prompts

| |
|---|

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

| |
|---|

Grounded Conversation Generation

| |
|---|

Can you give a detailed description of this image?

A large, colorful hot air balloon is flying over the river. The sky is overarching the river, trees, and buildings.

Referring Expression Segmentation

Image Level Captioning

Region Level Captioning

Phrase Grounding

Please describe the given image.

[Figure 23]

Can you segment the balloon in this given picture?

[Figure 24]

Can you segment phrases in, “A <hot air balloon> over the <river>”

Could you tell me more about this region?

A hot air balloon is seen in the air, flying over a river. The balloon, high up in the sky, is yellow and red in color, with a bridge and trees visible in the background.

[Figure 25]

[Figure 26]

[Figure 27]

Sure, the segmentation is

[Figure 28]

[Figure 29]

A yellow, red and blue hot air balloon

A <hot air balloon> over the <river>

- Figure 2. GLaMM’s architecture. The figure illustrates our model architecture, showcasing its ability to offer scene-level understanding, region-level interpretation, and pixel-level grounding. Top: The core components of GLaMM, including the global image encoder, region encoder, LLM, grounding image encoder, and pixel decoder, are cohesively tailored for vision-language tasks across different granularities. The vision-to-language (V-L) projection layer efficiently maps image features into the language domain, and the pixel decoder utilizes the language-to-prompt (L-P) projection layer, transforming text embeddings related to segmentation into the decoder space. A major feature of GLaMM is its ability to perform our newly introduced Grounded Conversation Generation (GCG) task. This highlights the model’s capability to anchor specific phrases to corresponding segmentation masks in the image. Bottom: The diverse downstream applications of GLaMM, including referring expression segmentation, region-level captioning, image-level captioning, and phrase grounding.

with a specialized token, <SEG>. Prompts, such as “Please segment the ‘man in red’ in the given image," trigger the model to generate responses with corresponding <SEG> tokens. A language-to-prompt (L-P) projection layer (g) transforms the last-layer embeddings corresponding to <SEG> tokens (lseg) into the decoder’s feature space. Subsequently, P produces binary segmentation masks M,

region encoder (R) extends the model’s capability to interpret and interact with user-specified regions in an image. This component constructs a hierarchical feature pyramid from four selected CLIP global image encoder layers, followed by RoIAlign [10] to generate a 14x14 feature map. Combining these features yields a unified region-of-interest (RoI) representation. To facilitate region-targeted responses from GLaMM, we augment the existing vocabulary with a specialized token <bbox>. This is integrated into a prompt like, “The <image> provides an overview of the image. Can you provide a detailed description of the region <bbox>?”. Here the <bbox> token is replaced with the RoI extracted features.

M = P g(lseg),V(ximg) , s.t.,Mi ∈ {0,1}.

Using an end-to-end training approach, GLaMM excels in region understanding, pixel-level grounding, and conversational capabilities. However, due to the lack of standard benchmarks for the novel setting of generating visually grounded detailed conversations, we introduce a novel task, Grounded Conversation Generation (GCG), and a comprehensive evaluation protocol as explained next.

For the region-level understanding, alongside the global image features Ix, we also take user-specified regions r as inputs, encoded as Rx = R(Ix,r), followed by projection to language space through the same V-L projection layer f employed in scene-level understanding. We augment the text instruction xt by replacing <bbox> tokens with the corresponding region features to obtain x′t = [xt ← f(Rx)]. The LLM then generates the output yt as,

#### 3.2. Grounded Conversation Generation (GCG)

The objective of the GCG task is to construct image-level captions with specific phrases directly tied to corresponding segmentation masks in the image. For example, “<A man> and <a boy> sit on <a bench> next to <an old white car>.”, shown in Fig. 3 (left), features how each bracketed phrase (highlighted in the image) is anchored to a unique image segmentation mask. This creates a densely annotated caption that aligns textual descriptions with visual regions, enriching the image’s contextual interpretation.

yt = L f(Ix),x′t .

Pixel-Level Grounding: Utilizing the grounding image encoder denoted as V and the pixel decoder represented as P, GLaMM facilitates fine-grained pixel-level object grounding, allowing it to ground its responses visually. We instantiate V with a pretrained SAM encoder [18] and design P based on a SAM decoder-like architecture. To activate the pixel-level grounding, our model’s vocabulary is augmented

GCG Output Representation: A sample prompt for querying the model in this task is: “Could you please

[Figure 30]

[Figure 31]

[Figure 32]

A soccer player in a red uniform is about to kick the ball while a player in a white uniform is trying to block the shot.

A woman in a navy blue jacket and hat with a hair ribbon in her hair.

A man and a boy sit on a bench next

to an old white car.

- Figure 3. Qualitative results of GLaMM on grounded conversation generation (GCG). Given user queries, the LMM generates textual responses and grounds objects, object parts, attributes, and phrases using pixel-level masks, showing its detailed understanding.

give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.” The model generates a detailed caption along with interleaved segmentation masks, employing the format “<p>A man</p><SEG> and <p>a boy</p><SEG> sit on <p>a bench</p><SEG> next to <p>an old white car</p><SEG>.” We use special tokens, namely <p>, </p> and <SEG>, to delineate the start and end of each phrase and its corresponding region mask, respectively.

Our GranD dataset is meticulously constructed using a stage-wise annotation pipeline, capturing annotations that range from fine-grained specifics to high-level context. This enables the automatic generation of densely annotated captions well-suited for the GCG task, thereby significantly facilitating GLaMM’s training for this task. Some qualitative results of our model on the GCG task are shown in Fig. 3.

Evaluation Criteria: We introduce a benchmarking suite for GCG, with a validation set of 2.5K images and a test set of 5K images. Four key aspects are evaluated: i) generated dense caption quality, ii) mask-to-phrase correspondence accuracy, iii) generated mask quality, and iv) region-specific grounding ability. Metrics include METEOR and CIDEr for captions, class-agnostic mask AP for grounding, mask IoU for segmentation, and mask recall for region-specific grounding (refer to Appendix A.1 for details).

Having delineated the architecture of GLaMM and the intricacies of the GCG task, it becomes imperative to address the scarcity of large-scale annotated data for regionlevel understanding. We next focus on devising a new, densely annotated dataset to optimize the model’s performance and overcome this data limitation.

- 4. Data Annotation Pipeline

level details. It aims to overcome challenges in image understanding and dense pixel-level grounding, thereby expanding capabilities of visual instruction tuning in LMMs. The pipeline contains four distinct levels (see Fig. 4).

i) Level-1 focuses on object localization and provides semantic labels, segmentation masks, attributes, and depth information. ii) Level-2 defines relationships between detected objects. iii) Level-3 organizes information from the first two levels into a hierarchical scene graph, used to generate dense captions using LLM with in-context examples. iv) Level-4 offers enriched contextual information for a deeper understanding of the scene, going beyond what’s observed (e.g., historical information of a landmark). Please refer to Appendix A.4 for pipeline implementation details.

#### 4.1. Object Localization and Attributes (Level-1)

- In level-1, the focus is on detailed object identification within images. First, object-bounding boxes are identified using multiple SoTA object detection models. Classagnostic NMS is applied to each model to filter out false positives. After this step, bounding boxes from different models are compared using IoU, with a bounding box retained as an object only if detected by at least two other detection models. We also generate attributes for each filtered object using region-based vision-language models and incorporate depth information to contextualize each object’s relative position within the scene.

4.2. Relationships and Landmarks (Level-2)

- In level-2, multiple short textual descriptions of the overall scene are generated. Phrases extracted from these descriptions are grounded to specific objects in level-1 to form relationships. These relationships articulate connections between multiple objects or define an object’s role within the scene. Further, each scene is assigned a landmark category that includes a primary and a more specific sub-category (see Tab. 7 in Appendix A.4.1).

We introduce our automated annotation pipeline used to create the Grounding-anything Dataset (GranD). GranD is a comprehensive, multi-purpose image-text dataset offering a range of contextual information, from fine-grained to high-

[Figure 33]

Objects and Attributes

Scene Graph

[Figure 34]

dog, pub dog, a brown and white dog dog collar, black color, chain collar bell, cowbell steps, stairs, the steps of a building sack, a large white bag with black writing

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Relationships and Landmarks

Dense Grounded Caption

A dog sitting on the steps

A large brown dog is sitting on the steps of a building. It is wearing a black chain dog collar. The collar has a cowbell attached to it. There is a bag in the background with black writings on it.

A large brown dog wearing a chain collar

Cowbell attached to dog collar

Landmarks: Outdoor – Urban Landscape

Extra Context

Dogs, especially pugs and bulldogs, have been a part of human families for thousands of years, serving as loyal companions. They have been bred for specific traits, making them popular pets. Dogs have been trained for various tasks, including assisting people with disabilities and serving as search and rescue animals. Dog collars, often bearing identification tags, are essential for keeping pets safe and ensuring they can be returned home if lost. Cowbells, once used to signal the arrival of a cow, have been repurposed as dog collars, providing a distinct sound to help locate a dog if it wanders off. In outdoor urban landscape, dogs are often found sitting on steps, as they may choose to rest in spots that offer a good view of their surroundings.

Level-4

Level-1

Level-3

Level-2

Scene Graph & Dense Captioning

Extra Contextual Insights

Object locatlization and attributes

Relationships

§ Short Captions and Phrase extraction § Grounding expression § Landmarks

§ Hierarchical Scene Graph § In-contex Learning with LLM § Verification Pipeline

§ Image Tagging and Object Detection § Open Vocabulary Detection § Region Attribute Detection

§ Lanmark Details § History and Background § Precautionary Measures

- Figure 4. Automatic Annotation Pipeline of the Grounding-anything Dataset (GranD). Comprising four levels, this pipeline plays a pivotal role in generating GranD’s 7.5M unique concepts grounded in 810M regions. level-1 details objects and attributes, level-2 includes short captions and relational markers, level-3 builds a scene graph, hierarchically organizing information from earlier levels to facilitate LLM for grounded dense captions, level-4 provides additional historical and societal context for a richer visual understanding.

#### 4.3. Scene Graph and Dense Captioning (Level-3)

In level-3, object attributes and labels from level-1 are combined with the relationships and phrases obtained from level-2 to form a hierarchical scene graph. This structured data serves as a query for LLM to generate dense image captions. To provide additional context, depth values and bounding box coordinates are used to assign each object to specific spatial layers within the scene, such as immediate foreground, foreground, midground, or background. Additionally, short scene-level captions are incorporated into the scene graph to enhance LLMs’ contextual understanding.

Dense Captioning Verification: To enhance the fidelity of the LLM-generated dense captions, we implement an automatic verification pipeline using chain-of-thoughts prompting. This pipeline produces a checklist of objects derived from the generated dense caption expected to be present in the image. The associated caption is flagged as inaccurate if any object specified in the checklist is absent from the scene graph. Such captions are then regenerated, incorporating feedback from the initial assessment.

#### 4.4. Extra Contextual Insights (Level-4)

Level-4 builds on the scene graph from level-3 to obtain a more detailed visual understanding. we query LLM to extract extended contextual insights beyond basic object identification and relationships, including details about the landmarks, historical context, guidelines for interacting with the scene, and even predictive elements about future events. To facilitate this, we prompt LLM with in-context examples.

Dataset Images Regions Concepts Tokens Captions†

COCO [25] 0.1M 0.9M 80 - LVIS [9] 0.1M 1.5M 1,203 - Objects365 [42] 0.6M 10.1M 365 - Open Images [20] 1.5M 14.8M 600 - BigDetection [4] 3.5M 36.0M 600 - V3Det [43] 0.2M 1.5M 13,029 - VG [19] 0.1M 0.3M 18,136 51.2M SA-1B [18] 11M 1.1B - - AS-1B [45] 11M 1.2B 3.5M 132.2B GranD (Ours) 11M 810M 7.5M 5.0B 33M

Table 2. GranD versus existing datasets. GranD uniquely provides three †grounded captions per image with segmentation masks for every region. AS-1B is shaded to denote its concurrent, non-public status at the time of this publication.

Utilizing our automated annotation pipeline, we annotate a corpus of 11M SAM images [18], which are inherently diverse, high-resolution, and privacy-compliant. The resulting dataset comprises 810M regions, each associated with a segmentation mask, and includes 7.5M unique concepts. Further, the dataset features 84M referring expressions, 22M grounded short captions, and 11M densely grounded captions. To our knowledge, this is the first dataset of this scale generated entirely through an automated annotation pipeline (see Tab. 2 for details and Fig. 15 in Appendix for dataset sample visualizations).

#### 4.5. Building GranDf for GCG

Motivated by the need for higher-quality data in fine-tuning stage, we introduce GranDf. It contains 214K imagegrounded text pairs with 2.5K validation and 5K test sam-

Validation Set Test Set M C AP50 mIoU Recall M C AP50 mIoU Recall

Model

BuboGPT [59] 17.2 3.6 19.1 54.0 29.4 17.1 3.5 17.3 54.1 27.0 Kosmos-2 [35] 16.1 27.6 17.1 55.6 28.3 15.8 27.2 17.2 56.8 29.0 LISA* [21] 13.0 33.9 25.2 62.0 36.3 12.9 32.2 24.8 61.7 35.5 GLaMM† 15.2 43.1 28.9 65.8 39.6 14.6 37.9 27.2 64.6 38.0 GLaMM 16.2 47.2 30.8 66.3 41.8 15.8 43.5 29.2 65.6 40.8

###### Method refCOCO refCOCO+ refCOCOg

val testA testB val testA testB val(U) test(U)

CRIS [47] 70.5 73.2 66.1 65.3 68.1 53.7 59.9 60.4 LAVT [50] 72.7 75.8 68.8 62.1 68.4 55.1 61.2 62.1 GRES [26] 73.8 76.5 70.2 66.0 71.0 57.7 65.0 66.0 X-Decoder [63] - - - - - - 64.6 SEEM [64] - - - - - - 65.7 LISA-7B [21] 74.9 79.1 72.3 65.1 70.8 58.1 67.9 70.6 GLaMM 79.5 83.2 76.9 72.6 78.7 64.6 74.2 74.9

Table 3. Performance on GCG Task: Metrics include METEOR (M), CIDEr (C), AP50, mIoU, and Mask Recall. LISA* denotes LISA adapted for GCG. GLaMM† denotes training excluding 1K human annotated images. GLaMM shows better performance.

Table 4. Qualitative Assessment of GLaMM in ReferringExpression Segmentation: Performance across refCOCO, refCOCO+, and refCOCOg in generating accurate segmentation masks based on text-based referring expressions surpasses that of closely related work, including LISA which is specifically designed for this task.

ples. GranDf comprises two primary components: one subset is manually annotated, and the other subset is derived by re-purposing existing open-source datasets.

We extend open-source datasets—namely Flickr-

- 30K [37], RefCOCOg [16], and PSG [49] by generating compatible GCG annotations. For RefCOCOg, we use the dataset’s referring expressions and their connected masks. These expressions offer concise descriptions of distinct objects in the image. With the aid of GPT-4, we seamlessly blend these referring expressions with contextual information from COCO captions, crafting detailed yet accurate grounded captions while preserving the original referring expressions. This ensures zero error in matching phrases with their corresponding segmentation masks. This technique yields approximately 24K GCG samples. For PSG, we leverage the dataset’s triplet structures, which describe relations between two objects in a scene. These triplets are integrated with COCO captions using GPT-4, resulting in densely annotated captions that can be mapped to segmentation masks. This gives us around
- 31K additional GCG samples. For Flickr-30K, we use the 158K Flickr captions and their referring expressions alongside associated bounding boxes. These boxes are then accurately segmented using HQ-SAM [17].

In addition, we contribute a minor, high-quality manual annotation set to benchmark the GCG task. Using GranD’s automatic annotations as a base, annotators refine referring expressions to match SAM GT masks, yielding around 1000 focused samples for training and 1000 for evaluation (refer to Appendix D and Fig. 14 in Appendix for designed prompts and dataset visualizations).

### 5. Experiments

We perform quantitative evaluations of GLaMM on six benchmarks: i) Grounded Conversation Generation (GCG), ii) referring-expression segmentation, iii) region-level captioning, iv) image-level captioning, v) conversational-style question answering and vi) phrase grounding. We present the first four benchmarks next, and the remaining are discussed in Appendix B. Grounded Conversation Generation (GCG). We pretrain GLaMM on GranD dataset followed by fine-tuning on the GranDf dataset. The results are presented in Tab. 3 on both validation and test splits of the GranDf dataset (refer to Sec. 3.2 and Sec. 4.5 for details). GLaMM shows improved performance compared to baseline methods. Pretrained models for BuboGPT and Kosmos-2 are sourced from official releases, and LISA is adapted and trained on the GranDf dataset for the GCG task. GLaMM† denotes the variant trained on GranDf dataset excluding the 1000 human-annotated images. Qualitative results are shown in Fig. 3 and supplementary Fig. 7. Referring Expression Segmentation. In this task, the model processes an image and a text-based referring expression to output a segmentation mask. The prompt used is, “Please segment the <referring expression> in the image." The model responds with “Sure, it is <SEG>.", where the <SEG> token is decoded to obtain the mask. We achieve better results over recent works like LISA on the refCOCO, refCOCO+, and refCOCOg validation and test sets in Tab. 4. This demonstrates the efficacy of our GranD dataset, offering the model extensive concept vocabulary during pre-training (refer to Fig. 5 (middle) and supplementary Fig. 8 for qualitative results).

Region Level Captioning. In this task, models generate

[Figure 40]

[Figure 41]

[Figure 42]

Can you segment the McDonald's cup in this image?

Could you give a comprehensive explanation of what can be found within this picture?

Can you please describe this region?

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

| |
|---|

[Figure 47]

Can you please segment the smaller cup now?

[Figure 48]

[Figure 49]

[Figure 50]

A Yellow Volkswagen Beetle

[Figure 51]

Two girls in costume stand in front of a black curtain with a smiling face and skeleton head on it. They are next to a poster of a man in a tuxedo.

[Figure 52]

Is this car a new model ?

[Figure 53]

No, it is an old model

- Figure 5. Qualitative results of GLaMM’s performance across downstream tasks. The figure showcases examples from three tasks: region-level understanding (left), referring-expression segmentation (center), and image-level captioning (right), demonstrating its capabilities in offering in-depth region understanding, pixel-level groundings, and conversational abilities through an end-to-end training approach.

sual Genome and refCOCOg, using METEOR and CIDEr metrics with results presented in Tab. 5. GLaMM shows improved results over GRiT and GPT4RoI after fine-tuning and demonstrates robust zero-shot performance, highlighting the significance of GranD’s region-text pairs (refer to Fig.5 (left) and supplementary Fig. 9 for qualitative results). Image Level Captioning. For this task, GLaMM responds to queries like, “Could you please give me a detailed description of the image?" with a textual description. We evaluate GLaMM’s zero-shot performance on Flickr30k [37] and NoCap [1] datasets, with Tab. 6 showing its favorable performance against recent image captioning models and other LMMs (refer to Fig. 5 (right) and supplementary Fig. 10 for qualitative results).

###### Model refCOCOg Visual Genome

METEOR CIDEr METEOR CIDEr

GRIT [48] 15.2 71.6 17.1 142 Kosmos-2 [35] 14.1 62.3 - GPT4RoI [57] - - 17.4 145.2 GLaMM (ZS) 15.7 104.0 17.0 127.0 GLaMM (FT) 16.2 106.0 19.7 180.5

- Table 5. Performance of GLaMM in Region-Level Captioning: Metrics include METEOR and CIDEr scores, assessed on Visual Genome and refCOCOg Datasets, exhibiting competitive results.

Model NoCap Flickr30k

CIDEr SPICE CIDEr SPICE

VinVLM [55] 95.5 13.5 - LEMON [12] 106.8 14.1 - SimVLM [46] 110.3 14.5 - CoCa [54] 120.6 15.5 - BLIP [23] 113.2 14.7 - BLIP-2 [24] 121.6 15.8 - InstructBLIP [6] 123.1 - 82.8 Shikra-13B [5] - - 73.9 -

- Kosmos-1 [13] - - 67.1 14.5
- Kosmos-2 [35] - - 66.7 GLaMM 106.8 15.8 95.3 18.8

- Table 6. Performance of GLaMM in Zero-Shot Image Captioning: Assessed on Flickr30k and NoCap datasets, showing favorable results compared to recent models in the field. region-specific captions given an image, a user-specified region via a bounding box and related text. We utilize a prompt like, “Can you provide a detailed description of the region <bbox>?”, to instruct the model for this task, where the special token <bbox> is replaced with the actual region representations. We evaluate GLaMM on Vi-

### 6. Conclusion

We introduce GLaMM, the first model capable of generating natural language responses intertwined with object segmentation masks, allowing for enhanced multimodal user interactions. Recognizing the lack of standardized benchmarks for visually grounded conversations, we introduce the novel task of Grounded Conversation Generation and establish a comprehensive evaluation protocol. To facilitate research and model development, we create the Grounding-anything Dataset (GranD), a large-scale, densely annotated dataset with 7.5 million unique concepts grounded in 810 million regions. Our automated annotation pipeline ensures the reliability and scalability of this dataset used for our model. In addition to these contributions, we curated a dataset specifically tailored for the GCG task (GranDf) by leveraging existing open-source datasets, establishing a high-quality fine-tuning dataset to develop visually grounded conversations. Our model performs well on downstream tasks besides GCG, including region and image captioning, referring segmentation, and vision-language conversations.

### References

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In ICCV, 2019. 8
- [2] Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundational models defining a new era in vision: A survey and outlook. arXiv:2307.13721, 2023. 1
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023. 3
- [4] Likun Cai, Zhi Zhang, Yi Zhu, Li Zhang, Mu Li, and Xiangyang Xue. Bigdetection: A large-scale benchmark for improved object detector pre-training. In CVPR, 2022. 6
- [5] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv:2306.15195, 2023. 1, 2, 3, 8
- [6] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv:2305.06500, 2023. 1, 2, 3, 8
- [7] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. arXiv:2303.11331, 2023. 1
- [8] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. Llamaadapter v2: Parameter-efficient visual instruction model. arXiv:2304.15010, 2023. 1, 2, 3
- [9] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In CVPR,

2019. 6

- [10] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In ICCV, 2017. 4
- [11] Matthew Honnibal and Ines Montani. spaCy: Industrialstrength Natural Language Processing in Python. 2020. 1
- [12] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In CVPR,

2022. 8

- [13] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv:2302.14045, 2023. 8
- [14] Xinyu Huang, Youcai Zhang, Jinyu Ma, Weiwei Tian, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Lei Zhang. Tag2text: Guiding vision-language model via image tagging. arXiv:2303.05657, 2023. 1

- [15] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetrmodulated detection for end-to-end multi-modal understanding. In ICCV, 2021. 1
- [16] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 2, 7
- [17] Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu. Segment anything in high quality. arXiv:2306.01567, 2023. 7
- [18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 2, 4, 6
- [19] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 2017. 6
- [20] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV,

2020. 6

- [21] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv:2308.00692, 2023. 2, 3, 7, 1
- [22] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv:2305.03726, 2023. 1, 2, 3
- [23] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

- 2022. 8

[24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

- 2023. 2, 8, 1, 3

- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 6
- [26] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In CVPR, 2023. 7
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv:2310.03744, 2023. 3
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv:2310.03744, 2023. 1, 2, 3
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 2, 3

- [30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv:2303.05499, 2023. 2
- [31] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, Kunchang Li, Zhe Chen, Xue Yang, Xizhou Zhu, Yali Wang, Limin Wang, Ping Luo, Jifeng Dai, and Yu Qiao. Interngpt: Solving visioncentric tasks by interacting with chatgpt beyond language.

- arXiv:2305.05662, 2023. 1, 2, 3

[32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models.

- arXiv:2306.05424, 2023. 2

- [33] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In ECCV, 2022. 1
- [34] OpenAI. Gpt-4 technical report. arXiv:2303.08774, 2023. 2, 1, 7
- [35] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv:2306.14824, 2023. 1, 2, 3, 7, 8
- [36] Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Lewei Yao, Jianhua Han, Hang Xu, Lingpeng Kong, and Tong Zhang. Detgpt: Detect what you need via reasoning. arXiv:2305.14167, 2023. 1, 2, 3
- [37] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV,

2015. 2, 7, 8

- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3
- [39] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE TPAMI, 2020. 1
- [40] Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alex Smola, and Xu Sun. Prompt pretraining with twenty-thousand classes for open-vocabulary visual recognition. arXiv:2304.04704, 2023. 1
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. arXiv:2112.10752, 2021. 3
- [42] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, 2019. 6

- [43] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3det: Vast vocabulary visual detection dataset. arXiv:2304.03752,

2023. 6

- [44] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an openended decoder for vision-centric tasks. arXiv:2305.11175,

2023. 2, 3

- [45] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv:2308.01907, 2023. 2, 3, 6
- [46] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv:2108.10904, 2021. 8
- [47] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In CVPR, 2022. 7
- [48] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv:2212.00280, 2022. 8, 1
- [49] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In ECCV, 2022. 2, 7
- [50] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR,

- 2022. 7

[51] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv:2303.11381,

- 2023. 2, 3

- [52] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality. arXiv:2305.03726, 2023. 1, 2, 3
- [53] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv:2310.07704, 2023. 2
- [54] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv:2205.01917, 2022. 8
- [55] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Making visual representations matter in visionlanguage models. arXiv:2101.00529, 2021. 8
- [56] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao.

- Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv:2303.16199, 2023. 2
- [57] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv:2307.03601, 2023. 1, 2, 3, 8
- [58] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv:2306.03514, 2023. 1
- [59] Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. Bubogpt: Enabling visual grounding in multi-modal llms. arXiv:2307.08581, 2023. 2, 3, 7
- [60] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv:2306.05685, 2023. 1, 2, 3
- [61] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv:2304.10592, 2023. 1, 2, 3
- [62] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In ICCV, 2023. 1
- [63] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. In CVPR, 2023. 7
- [64] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. In NeurIPS, 2023. 7

## GLaMM: Pixel Grounding Large Multimodal Model Supplementary Material

We provide supplementary material for a deeper understanding and more analysis related to the main paper, arranged as follows:

- 1. Additional implementation details (Appendix A)
- 2. Additional downstream tasks (Appendix B
- 3. Additional qualitative results (Appendix C)
- 4. Dataset visualizations (Appendix D)
- 5. Limitations and future work (Appendix E)
- 6. Ethics and societal impact (Appendix F)

### A. Additional Implementation Details

#### A.1. Evaluation Metrics

Mask Recall: To quantify region-specific grounding, we propose a ‘mask recall’ metric, utilizing a two-tiered validation approach. Initially, predicted masks are mapped to ground-truth masks via a one-to-one set assignment, followed by IoU computation for these pairs. Pairs surpassing a 0.5 IoU threshold proceed to a textual similarity assessment using BERT. A pair is considered a true positive (TP) only if both IoU and BERT similarity exceed their 0.5 thresholds; otherwise, it is classified as a false positive (FP). The mask recall is subsequently calculated using the standard formula, normalizing the number of TPs by the total ground-truth mask count.

#### A.2. Model Architecture and Training

In all of our experiments, we use Vicuna LLM [60] with 7B parameters. The design of region encoder is motivated from GPT4RoI [57] and grounding image encoder and pixel decoder are inspired from LISA [21]. The V-L and L-P layers are implemented using 2 layer MLP with GELU activation as in LLaVA-v1.5 [28]. We use PyTorch to implement our GLaMM and use Deepspeed zero-2 optimization during training.

Specifically, our model is trained using two types of losses: auto-regressive cross-entropy loss for text generation and a linear combination of per-pixel binary crossentropy loss and DICE loss for segmentation. During training, the global image encoder and grounding image encoder are kept frozen and the region encoder, projection layers (VL and L-P) and the pixel decoder are fully finetuned, while the LLM is LORA finetuned with α = 8. Our codes and pretrained models will be publicly released.

##### A.2.1 Pretraining on GranD

During pretraining GLaMM is trained on GranD dataset for referring expression segmentation, region-level captioning,

image-level captioning and grounded conversation generation (GCG) tasks simultaneously. We use a batch size of 160 and train for a total of 35K iterations during pretraining. We use LORA-8 for efficiently adapting the LLM and initialize the pretraining from GPT4RoI [57] for faster convergence. In the experiment tables in Section. 5, we refer to this model as GLaMM (ZS) which is obtained after pretraining on GranD.

#### A.3. Finetuning on Downstream Tasks

We finetune GLaMM on multiple downstream tasks including GCG, referring expression segmentation, region-level captioning and image-level captioning. For GCG, we finetune our model on GranDf dataset. A batch size of 160 is used and the model is trained for 5K iterations in total. It is worth noting that GranDf dataset is a combination of multiple open-source datasets that we repurposed for GCG task using GPT4 [34]. Please refer to Appendix. D for the prompts designed to query GPT4 for constructing GranDf dataset, along with the dataset visualizations.

For referring expressions segmentation, we finetune GLaMM on refCOCO, refCOCO+ and refCOCOg datasets. We represent this model as GLaMM (FT) in Tab. 4. Similarly, for region-level captioning, GLaMM (FT) is finetuned on refCOCOg and Visual Genome datasets. For imagelevel captioning, we fine tune GLaMM on LLaVA-Instruct150K [29] dataset. For LLaVA-bench, the model is finetuned on LLaVA-Instruct-80K [29] instruction set. We use eight NVIDIA A100-40GB GPUs in all of our pretraining and finetuning experiments.

#### A.4. Automated Dataset Annotation Pipeline

Our automated annotation pipeline incorporates diverse state-of-the-art models at various levels. For Level-1, we use Tag2Text [14] and RAM [58] for image tagging, CoDETR [62], EVAv02 [7], OWL-ViT [33], and POMP [40] for object localization, GRiT [48] and GPT4RoI [57] for attribute generation, and MiDAS [39] for depth estimation. Level-2 leverages BLIP-2 [24] and LLaVA-v1.5 [28, 29] for scene descriptions and landmark categorization, SpaCy [11] for phrase extraction, and MDETR [15] for phrase grounding. For both Level-3 and Level-4, we use Vicuna-v1.5 [60] with 13B parameters, supplemented with in-context examples. Please refer to Appendix A.4 for further details on implementation and LLM prompts used across different pipeline levels.

We design a fully automated dataset annotation pipeline using multiple hierarchical levels in the visual domain to construct GranD dataset. The segmentation masks for

Prompt: The provided prompt is a scene graph, which is a structured representation of a scene detailing its various elements and their relationships. The scene graph consists of:

- 1. Layers of Depth: The scene is divided into different layers based on proximity - 'Immediate Foreground', 'Foreground', 'Midground', and 'Background'. Each layer depicts objects or entities at that depth in the scene.
- 2. Groups: Within each depth layer, objects and entities are clustered into groups, sometimes with specific attributes.
- 3. Relationships: This section illustrates the interactions or spatial relationships between various objects or groups.
- 4. Landmarks: It gives a broader view or categorization of the scene, defining its overarching theme or environment.

---

- ##Example - 1:

- Prompt: {scene_graph_1} Desired caption: {dense_caption_1}

-----##Example - 2:

- Prompt: {scene_graph_2} Desired caption: {dense_caption_2}

-----Please provide a simple and straightforward 2-4 sentence image caption based on the following scene graph details: {scene_graph}. Create the caption as if you are directly observing the image. Do not mention the use of any source data like 'The relationship indicates ...' or 'No relations specified'.

- (a) Illustration of LLM in-context learning for dense captioning used in the construction of our GranDdataset.
- (b) Illustration of LLM in-context learning for extra contextual insights used in the construction of our GranDdataset.

Prompt:

- ##Example - 1:

- Prompt: {scene_graph_1}

- Additional context: {caption_1}

-----##Example - 2: Prompt: {scene_graph_2}

- Additional context: {caption_2}

-----Provide context based on the typical usage, history, potential dangers, and other interesting aspects surrounding the general theme presented by the objects and elements in the following scene graph: {scene_graph}

Limit the response to one paragraph with 5-7 sentences. DO NOT mention, refer to, or hint about "objects", "scene", or "scene graph". ONLY focus on explaining use cases, history, potential dangers, etc.

- Figure 6. Prompts used to construct GranD dataset. The figure shows the prompts used to query Vicuna [60] to generate dense captions and the extra context in our automated training pipeline. We provide in-context examples to guide the LLM.

##### A.4.1 LLM Prompts and In-context Learning

most of the regions are obtained from SAM [18] annotations by comparing our detected labeled regions with SAMprovided class-agnostic regions. For the remaining regions that do not match with any of the SAM regions, we run SAM model with a bounding box query to obtain masks.

Landmark categorization: We use LLaVA-v1.5-13B [28] model to assign landmark categories to each image. Please refer to Tab. 7 for primary and fine categories used.

Main category Fine Category Indoor scene Living space, Work space, Public space, In-

Our automated annotation pipeline utilizes only opensource models and incorporates a feedback loop using the chain of thoughts prompting via LLM. As it does not require feedback from the human in the loop, it can be scaled to generate dense noisy labels for a larger number of images, which can then be used to pretrain a larger LMM. Given the availability of enough compute power, this could be a step towards building a larger generic large multi-modal model. We will release our GranD dataset along with the implementation of our automated dataset annotation pipeline for further research. Below we present the LLM prompts we use at different levels of our automated dataset annotation pipeline.

dustrial space Outdoor scene Urban landscape, Rural landscape, Natural landscape

Transportation scene

Road, Airport, Train station, Port and harbor

Sports and recreation scene

Sporting venue, Recreational area, Gym and fitness center

Table 7. Summary of landmark categories and their corresponding fine-grained categories. We use LLaVA-v1.5 [28] for assigning landmark categories to images.

Dense Captioning: We arrange objects, attributes and relationships hierarchically to construct a visual scene graph,

that is used to query Vicuna-v1.5-13B [60] model along with in-context examples to generate dense captions. The designed prompt is shown in Fig. 6 (a).

Extra Context: We query Vicuna-v1.5-13B model to generate additional context about the visual scene. The prompt designed for this purpose is shown in Fig. 6 (b).

### B. Additional Downstream Tasks

#### B.1. Phrase Grounding

In order to adapt the GLaMM model for phrase grounding, we repurpose the GCG dataset to suit this particular task. Specifically, the answers in the GCG dataset are now used as questions, and the parts of the captions containing groundings are regarded as phrases. The model is subsequently trained to locate pixel-level groundings for these phrases, which are enclosed within <p> and </p> tokens. The results of this adaptation are shown in the following figure.

[Figure 54]

#### B.2. Conversational Style Question Answering

We evaluate our model on the LLaVA-Bench [28, 29] that uses GPT-4 for evaluation of models. This benchmark tests the model on three different types of tasks: conversation question-answering, detailed descriptions, and complex reasoning tasks. The evaluation provides insights into the model’s conversational and reasoning capabilities. The results in Tab. 8 present a comparison of GLaMM with previous open-source models. We note that GLaMM performance is on par with the recently released LLaVA1.5 which leverages additional data for vision-to-language alignment. Qualitative results are shown in Fig. 11 and Fig. 13.

Method LLM LLaVAW

BLIP-2 [24] Vicuna-13B 38.1 InstructBLIP [6] Vicuna-7B 60.9 Qwen-VL [3] Qwen-7B 63.4 Qwen-VL-Chat [3] Qwen-7B 58.6 LLaVA-1.5 [27] Vicuna-7B 63.4 GLaMM Vicuna-7B 63.3

Table 8. Evaluation of GLaMM on conversational style QA using LLaVA-Bench. The table compares GLaMM’s performance with previous open-source models in conversation questionanswering, detailed descriptions, and complex reasoning tasks.

### C. Additional Qualitative Results

In this section, we provide more qualitative examples to better understand the capacity of GLaMM.

#### C.1. Grounded Conversation Generation (GCG)

- Fig. 7 shows qualitative results of GLaMM finetuned on

GranDf dataset. The model could produce dense captions and provide dense pixel-level groundings of the caption.

C.2. Referring Segmentation

- Fig. 8 shows the effectiveness of GLaMM in understanding the natural language query and segmenting the corresponding objects. Note that GLaMM can also segment multiple objects via multi-round conversations.

C.3. Region-level Captioning

- Fig. 9 shows the qualitative results of GLaMM for regionlevel understanding. Our model can generate detailed descriptions about the user-specified regions in an image.

C.4. Image-level Captioning

- Fig. 10 shows GLaMM’s qualitative results on captioning tasks. Our model can generate dense captions for images.

#### C.5. Conditional Image Generation

- Fig. 12 shows GLaMM’s seamless integration for generative tasks. We use the Stable Diffusion inpainting model stable-diffusion-xl-1.0-inpainting [41] for this task. We first generate a segmentation mask using our GlaMM model based on the user query. This segmentation mask along with the user prompt is given as the input to the Stable Diffusion inpainting model, which generates the final output.

C.6. Conversations

- Fig. 13 illustrates the unique functionality of GLaMM to engage in multi-purpose task conversations. GLaMM is a generic conversational model that can accept prompts in the form of text and/or region and can answer in the form of

[Figure 55]

[Figure 56]

[Figure 57]

The image features a large, old building with a roof, situated on a grassy field. A tree is also present on the grass. The sky is seen overarching the building.

The image showcases a boat sailing on a river. The sky is overarching the boat, bridge, and the buildings.

A person in a black jacket and a straw hat is cooking some food.

- Figure 7. Qualitative results of GLaMM’s performance in grounded conversation generation. The figure shows how GLaMM seamlessly generates detailed responses, grounding phrases using pixel-level masks showing its detailed understanding.

[Figure 58]

[Figure 59]

Can you segment the woman’s hair in this image

[Figure 60]

Now please segment the hat.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Can you segment Ramos in this image ?

[Figure 66]

Now can you segment the football ?

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Can you segment what the person is using to ride ?

Now can you please segment the sea.

[Figure 72]

[Figure 73]

- Figure 8. Qualitative results of GLaMM’s capability in referring expression segmentation. The figure illustrates how GLaMM effectively translates text-based referring expressions into corresponding segmentation masks. Leveraging its training on the GranD dataset, the model can provide pixel-grounded reasoning and operate across various levels of granularity.

They are wrestling in the sand.

What are they doing ?

It could be potentially dangerous for the woman to be using her cellphone while crossing the street. She might not be paying full attention to the traffic and might be more susceptible to accidents.

Can you please describe the region ?

[Figure 74]

Why does he need this ?

[Figure 75]

[Figure 76]

[Figure 77]

What is she doing in this image ?

[Figure 78]

Do you think this is dangerous ?

[Figure 79]

[Figure 80]

[Figure 81]

| |
|---|

A man sitting on a motorcycle

[Figure 82]

| |
|---|

[Figure 83]

It is a helmet and he needs it for protection while riding his motorcycle.

| |
|---|

She is using her cellphone while crossing

the street.

[Figure 84]

This is an image from India.

[Figure 85]

[Figure 86]

Can you tell me which country this image is from, given the context ?

[Figure 87]

[Figure 88]

| |
|---|

[Figure 89]

- Figure 9. Qualitative illustration of GLaMM’s performance in region-level captioning. The figure demonstrates GLaMM’s ability to generate region-specific captions adeptly, translating the intricate details from designated regions into coherent textual descriptions, enriched by its training on the comprehensive GranD dataset. This capability, combined with the inherent reasoning abilities of LLMs, enables it to tackle reasoning-based visual questions about these regions.

[Figure 90]

[Figure 91]

[Figure 92]

The image showcases a large, ornate ceiling with various

The image features a large building with a sign that reads “TESCO”. A light is also visible, hanging from the building. The sky is seen over the building.

The image shows a large elephant statue with tusks, standing on a wooden floor. There is also a skull of an elephant in front of the elephant statue.

religious paintings. The ceiling is part of a church’s interior, featuring a vibrant artwork.

- Figure 10. Qualitative results of GLaMM on image-level captioning tasks. The figure shows the capabilities of GLaMM in generating detailed and context-aware captions for a diverse range of images. On the left, GLaMM demonstrates its proficiency in text recognition within images; it accurately identifies and incorporates specific textual information, such as the brand name "TESCO," into its caption. In the middle image, GLaMM’s capability to discern subtleties in visual content is showcased. It can effectively distinguish between live entities and inanimate objects, such as differentiating a living creature from a statue. On the right, the figure demonstrates GLaMM’s competence in reasoning about complex visual scenes. It can analyze and describe intricate details and interactions within an image, reflecting a deep understanding of both the individual elements and the overall context of the scene.

Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer

A man in a fishing boat uses an oar to navigate the water.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

What can you tell me about the

region indicated in the image?

[Figure 97]

[Figure 98]

| |
|---|

[Figure 99]

| |
|---|

“A straw Hat”

[Figure 100]

Why does he need the hat ?

[Figure 101]

[Figure 102]

The man is wearing the hat to protect his head from the sun.

Can you segment the water body ?

[Figure 103]

[Figure 104]

[Figure 105]

Sure, here’s the segmentation.

[Figure 106]

Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.

A woman in a white hat and apron is presenting a bowl to a man in a red shirt.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Can you please describe this region ?

[Figure 111]

[Figure 112]

| |
|---|

A Table with a large number of bowls on it

[Figure 113]

Why do we need so many bowls ?

[Figure 114]

[Figure 115]

This is a restaurant setting where the bowls are used to serve different dishes. It is a common practice in some cultures to serve a variety of small dishes or appetizers in bowls to customers.

Can you segment what he is using Sure, the segmentation result is. for cooking ?

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

| |
|---|

- Figure 11. Multimodal conversational interactions facilitated by GLaMM. The figure showcases GLaMM engaging in multi-turn dialogues, providing detailed descriptions, addressing region-specific inquiries, and presenting grounded conversations. This effectively highlights its adaptability in intricate visual-language interactions and robustly retaining reasoning capabilities inherent to LLMs.

[Figure 121]

[Figure 122]

Can you please segment the person wearing the orange jacket ?

Can you please segment the yacht ?

[Figure 123]

[Figure 124]

Sure, It is.

Sure, It is.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Generate “An airplane lands in the water surrounded by green hills.”

Generate “Spider man riding a bike on the street.”

[Figure 131]

[Figure 132]

Sure, Here's the new image

Sure, Here's the new image

[Figure 133]

[Figure 134]

- Figure 12. Qualitative results of GLaMM on conditional image generation. The figure shows the integration of GLaMM with an image generation model (stable diffusion). GlaMM first generates the segmentation mask (e.g. "yacht" in the left image and "person wearing orange jacket" in the right image) which is used along with a text prompt as input to the diffusion model to generate the desired images.

The image suggests that the two individuals are dressed up in animal masks and playing accordions as a form of entertainment or performance. This could be part of a street performance, a parade, or a themed event. The use of animal masks adds an element of humor and novelty to their appearance, drawing attention and curiosity from passersby. The accordions serve as a musical accompaniment, adding to the overall spectacle and providing a lively, upbeat atmosphere.

Why do you think they wear masks and play the accordion ?

A white horse mask.

Can you please segment the snow around him?

Why does he need the jacket and

cap ?

A blue knit cap

Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer

A child in a blue jacket is playing with a fox in the snow.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

What can you tell me about the region indicated in the image?

[Figure 139]

[Figure 140]

| |
|---|

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

It is cold outside the jacket and the cap provides warmth.

[Figure 145]

[Figure 146]

[Figure 147]

Sure, here’s the segmentation.

[Figure 148]

Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.

Two people wearing animal masks playing accordions.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Can you please describe this region ?

[Figure 153]

[Figure 154]

| |
|---|

[Figure 155]

[Figure 156]

[Figure 157]

Can you segment the man wearing the red shoes ?

[Figure 158]

[Figure 159]

[Figure 160]

Sure, the segmentation result is.

[Figure 161]

| |
|---|

[Figure 162]

| |
|---|

- Figure 13. Multimodal conversational with GLaMM. The figure shows multimodal conversations generated through GLaMM. The model is flexible enough to process multimodal inputs and respond with multimodal outputs in a single conversation.

Prompt: You are given five base captions for an image briefly describing the image from different perspective. You are also given a number of relationships between objects in the image. Each relationship consists of [subject, relation/verb, object]. Note that each subject and object follows the format <entitiy name><object number in image>. For example, if there are five persons and two tables, they will be formatted as person-1, person-2,.., person-5 and table-1, table-2.

Provide a concise image caption that straightforwardly describes objects/things visible in the image, using provided relationships. Use the base captions for context understanding, but it is not mandatory to include them verbatim in the final description. Break down the description into shorter sentences. Importantly keep the names of the subject and object unchanged. Break down the description into shorter, clear sentences. Do not add any extra information. Keep it brief.

‘person in truck’, ‘car parked on pavement-merged’, ‘truck parked on dirt-merged’, ‘house beside tree-merged’, ‘dirtmerged beside pavement-merged’

‘person about to hit sports ball’, ‘person swinging tennis racket’, ‘person running on playingfield’, ‘banner attached to fence-merged’, ‘playingfield beside fence-merged’

‘person attached to dog’, ‘person on bed’ , ‘dog on bed’, ‘dog attached to blanket’, ‘bed in front of wall’, ‘curtain attached to wall’, ‘cabinet attached to curtain’

[Figure 163]

[Figure 164]

[Figure 165]

The image depicts a person on a bed, lovingly hugging a dog. Both the person and the dog are on the bed, with the dog also attached to a blanket. Further aspects of the room include a curtain attached to a wall, and a cabinet attached to this curtain.

A large commercial truck is parked on a dirt area, with its lights on. A person is sitting inside this truck. Close by the truck, a car is parked on the pavement. There is a house nestled beside a tree that's near the dirt and pavement.

A person is running on a playing field situated beside a fence. This individual is about to hit a sports ball, swinging a tennis racket in preparation. Attached to the fence is a banner.

- (a) Samples from our GranD𝑓 dataset: Illustrating the repurposing of the OpenPSGdataset for the GCG task.
- (b) Samples from our GranD𝑓 dataset: Illustrating the repurposing of the RefCOCO-g dataset for the GCG task.
- (c) Samples from our GranD𝑓 dataset: Illustrating the repurposing of the Flickr-30k dataset for the GCG task.

Prompt: Create a concise image caption that straightforwardly describes things visible in the image, using provided phrases verbatim without altering them in any way. Do not infer actions or relationships not explicitly mentioned in the phrases. Use the base captions for contextual understanding, but it is not mandatory to include them verbatim in the final description. Preserve the exact wording of the provided phrases in the final caption. Break down the description into shorter, clear sentences, but without sacrificing the exact wording of the provided phrases. Construct a simple, coherent, and succinct caption that accurately uses the exact phrases, organizing the information into shorter sentences without assuming or inferring additional actions or relationships.

[Figure 166]

‘ a tupperware box filled with fruit’, ‘a plastic container with sliced radishes, green peppers, cucumbers and a sauce’, ‘slices of pizza are also visibly kept within a plastic container’

‘a woman with glasses and a purple shirt is having a meal’, ‘a man with a mustache wearing a blue shirt with food on his plate’, ‘the man with the glasses on’

‘a laptop computer sitting to the left of a red book’, ‘ laptop on the right side of the table with a scrabble game open’

[Figure 167]

[Figure 168]

A tupperware box filled with fruit features prominently on the table, along with a plastic container with sliced radishes, green peppers, cucumbers and a sauce. Several slices of pizza are also visibly kept within a plastic container.

A woman with glasses and a purple shirt is having a meal at a table with a man with a mustache wearing a blue shirt with food on his plate and another man, the man with the glasses on.

A laptop computer sitting to the left of a red book. There is also a laptop on the right side of the table with a scrabble game open.

[Figure 169]

[Figure 170]

[Figure 171]

A small group of three enjoy the view of the water as a small boy wanders off and two companions enjoy a walk along the shoreline.

A man in a striped shirt poses with a blond girl in a black apron.

Toddler sits on carpet in living room touching guitar .

- Figure 14. Dataset samples from GranDf. The figure shows the GPT4 [34] prompts used and the created dataset samples from Grandf dataset. This repurposed human-annotated dataset provides rich semantics to GLaMM for GCG task.

|[Figure 172]<br><br>[Figure 173]|
|---|

Life buoy, Life belt, Life jacket: A red and white lifesaver.

Laptop, Laptop computer: Laptop on a brown table.

[Figure 174]

|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|
|---|

Jersey: A boy wearing blue shirt.

Boy, Person: A boy in blue shirt. The boy is bending.

Person, Woman, Girl: A girl wearing orange t-shirt, the girl is bending.

Boy, Person: A boy wearing blue shirt. The boy is kneeling.

Chair: A black chair with a girl in an orange shirt leaning on it.

[Figure 184]

Mannequin, Person: A mannequin with blue clothes lying on the floor.

Wood floor, Classroom: A first aid course in classroom.

Dense Caption: A group of children are practicing a first aid course in a classroom. A mannequin wearing a blue shirt, surrounded by young boys, including two wearing blue shirts. There are various items on the table, such as a laptop, projector, and markers. A chair and a blackboard with a red and white lifesaver jacket is also seen in the room. The boys are bending over the mannequin, possibly practicing CPR, while a women watches.

young boys

Additional Context: The image depicts a first aid training session, likely aimed at teaching children basic life-saving techniques like CPR. Given the classroom setting and the presence of a projector and laptop, it may be part of a structured educational program. First aid courses like this are crucial for empowering people to handle emergencies effectively, reducing the severity of injuries or even saving lives. However, improper technique can be potentially dangerous, emphasizing the importance of qualified supervision, in this case, provided by the watching woman. The lifesaver jacket on the blackboard hints at a broader scope of training, possibly including water safety. The brown table with a laptop on it serves as a functional workspace, allowing for remote work or study in a cozy environment.

|[Figure 185]<br><br>[Figure 186]|
|---|

Person, Woman: A girl wearing holding pink bow and bag.

Flower arrangement: Pink flowers on the sidewalk.

Person, Girl: A girl carrying bags and a cellphone.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Scarf: A burgundy scarf around a womans neck.

[Figure 192]

Cellphone: A cellular phone.

[Figure 193]

Bag, Backpack: A gray and black backpack with a white tag.

[Figure 194]

[Figure 195]

Person, Man: A man in jacket carrying a backpack.

Jacket: A girl wearing a jacket.

[Figure 196]

[Figure 197]

Coat, Jacket: A woman in gray coat with a backpack.

Shopping bag, Plastic bag: The woman holding a cell phone.

[Figure 198]

[Figure 199]

Cellphone: A cellular phone.

[Figure 200]

Jeans: Blue bag on the womans arm.

Bag, backpack: A white and gray backpack.

Two women

Dense Caption: Two women with backpacks are taking a selfie with cellphone in front of a flower-covered wall, enjoying their time together in the city. One of them is wearing a gray jacket and a white and gray backpack, while the other is holding a umbrella and a black backpack with a white tag. They are surrounded by potted plants and a tall planter of flowers.

Additional Context: In the urban landscape, individuals often carry various bags and backpacks to store their belongings, such as handbags, shopping bags, and backpacks. These bags are usually made of durable materials like canvas or nylon and come in different colors, sizes, and styles. Some people prefer to carry a scarf or a jacket to protect themselves from the elements, while others wear jeans or trousers for comfort and convenience. Outdoor spaces in the city may feature potted plants, flower arrangements, and other decorative elements to enhance the aesthetic appeal of the area.. Cell phones and other electronic devices have become essential for communication and accessing information on-the-go. In outdoor settings, people often use these devices to capture memories, stay connected with others, and navigate their surroundings.

- Figure 15. Dataset samples from GranD. The figure shows a few samples from the GranD dataset, generated using the automated annotation pipeline. It provides multiple semantic labels and attributes for detected objects, along with the grounded dense caption and additional context.

### D. Dataset Visualization

text and/or segmentation masks. Note that our model is not explicitly trained to handle such scenarios, and this behavior emerges mainly due to our pretraining on GranD dataset, where an image is presented to LMM in different contexts.

In this section, we provide additional dataset samples of our GranD and GranDf datasets to better understand the func-

tionalities they offer. Please see Fig. 15 and Fig. 14.

### E. Limitations and Future Work

The large-scale automated pipeline provides dense labelings that are important for our pretraining but still contains some noise. A high-quality, clean dataset could help further improve the pretrained representations, although this comes at a significantly higher annotation cost. A potential research direction is to develop a cost-effective annotation pipeline aimed at reducing noise in dense labeling. Additionally, expanding the GLaMM framework to include modalities such as video and 3D is also a future research direction.

### F. Ethics and Societal Impact

Our Grounding-anything Dataset (GranD) utilizes SAM images that have de-identified personal information, with all faces and license plates obscured. To the best of our knowledge, the dataset does not portray any strong biases or discrimination. We urge for the responsible use of GranD and GLaMM, promoting research progress while safeguarding privacy.

### G. Acknowledgement

The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at Alvis partially funded by the Swedish Research Council through grant agreement no. 2022-06725, the LUMI supercomputer hosted by CSC (Finland) and the LUMI consortium, and by the Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre.

