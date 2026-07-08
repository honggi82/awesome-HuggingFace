## One missing piece in Vision and Language: A Survey on Comics Understanding

# arXiv:2409.09502v2[cs.CV]8Jan2025

EMANUELE VIVOLI, Computer Vision Center, UAB, Spain and MICC, University of Florence, Italy MOHAMED ALI SOUIBGUI, Computer Vision Center, UAB, Spain ANDREY BARSKY, Computer Vision Center, UAB, Spain ARTEMIS LLABRÉS, Computer Vision Center, UAB, Spain MARCO BERTINI, MICC, University of Florence, Italy DIMOSTHENIS KARATZAS, Computer Vision Center, UAB, Spain

Vision-language models have recently evolved into versatile systems capable of high performance across various tasks, often in zero-shot settings. Comics Understanding, a complex and multifaceted field, stands to benefit from these advances greatly. As a medium, comics combine rich visual and textual narratives, challenging AI models with tasks that span image classification, object detection, instance segmentation, and deeper narrative comprehension through sequential panels. However, the unique structure of comics—marked by stylistic variation, reading order complexity, and non-linear storytelling—poses distinct challenges. In this survey, we provide a comprehensive review of Comics Understanding from both dataset and task perspectives. Our contributions include: (1) analyzing the unique structure and elements of the comics medium; (2) surveying key datasets and tasks in comics research; (3) introducing the Layer of Comics Understanding (LoCU) framework, a novel taxonomy for redefining vision-language tasks in comics; (4) categorizing existing methods using the LoCU framework; and (5) identifying research challenges and future directions for applying vision-language models to comics. This survey pioneers a task-oriented framework for comics intelligence, aiming to guide future research by addressing gaps in data and task definition. A project associated with this survey is available at https://github.com/emanuelevivoli/awesome-comics-understanding.

CCS Concepts: • Computing methodologies → Information extraction; Computer vision representations; Image representations; Computer graphics; Artificial intelligence; Natural language processing. Keywords: Comics understanding, Vision and Language tasks, Comics, Manga, Sequential Visual Art ACM Reference Format:

Emanuele Vivoli, Mohamed Ali Souibgui, Andrey Barsky, Artemis Llabrés, Marco Bertini, and Dimosthenis Karatzas. 2025. One missing piece in Vision and Language: A Survey on Comics Understanding. 1, 1 (January 2025), 37 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

#### 1 INTRODUCTION

Comics serve as a sophisticated medium that combines visual and textual elements to tell stories [177]. Applying recent machine learning tools to understand comics is complicated. This complexity comes from its unique domain that, driven by the author’s creativity, exhibits unique variations in both style and content. For example, modeling the drawing style is complicated, the reading order changes from one book to another, and, unlike natural images, comics frequently depict transformations that defy physical laws, adding another layer of complexity.

Motivated by these challenges, advanced artificial intelligence (AI) approaches have been increasingly applied to Comics Understanding. Given that AI thrives on tackling complex and diverse tasks, many researchers are now focusing on problems like object detection [210], semantic segmentation [12], optical character recognition (OCR) [10], recurrence of characters and objects in

Authors’ addresses: Emanuele Vivoli, Computer Vision Center, UAB, Barcelona, Spain and MICC, University of Florence, Florence, Italy, evivoli@cvc.uab.cat; Mohamed Ali Souibgui, Computer Vision Center, UAB, Barcelona, Spain; Andrey Barsky, Computer Vision Center, UAB, Barcelona, Spain; Artemis Llabrés, Computer Vision Center, UAB, Barcelona, Spain; Marco Bertini, MICC, University of Florence, Florence, Italy; Dimosthenis Karatzas, Computer Vision Center, UAB, Barcelona, Spain.

varying contexts [109], pose estimation [9], depth estimation [13], and integration of visual and textual narratives [82]. Comics also include complex narrative elements such as satire and irony and spatial and temporal dynamics within the story that have not been deeply investigated. Many of these tasks, individually or in combination, represent the cutting edge of advanced AI, particularly in the area of Multimodal Vision-Language understanding [99, 101, 116, 222, 231].

Historically, in machine learning (and deep learning), innovation has been driven by the available data, both in terms of tasks and more capable models. As new types of annotated data and more complex tasks emerged, the need for innovative approaches grew, particularly in domains that combine multiple modalities. The domain of multimodal learning, especially the convergence of vision and language, has witnessed significant advancements. These advancements include the shift away from independent uni-modal representations [83, 158] toward the alignment and integration of different modalities [23, 26, 104, 183, 214, 244] and the creation of multimodal embedding space combining aligned uni-modal inputs [56, 243], with early and late fusion. Trained on extensive multimodal datasets, these models have shown remarkable abilities in tasks like multimodal retrieval [56, 158, 243], visual question answering, image captioning [104], reasoning [23, 214], and retrieval-augmented multimodal generation [232] as well as image generation [54, 225].

[Figure 1]

Fig. 1. Number of publications on Comics and Manga (from Google Scholar). The publications have been filtered by keywords (manga, comics, graphic novels), topics, and journals (computer science).

The comics domain is uniquely well-suited to driving advancements in these types of multimodal reasoning models. Research in comics has extensively explored a range of questions, from the human ability to derive meaning from sequential images [32] to machine interpretation of comic strips, particularly through closure tasks [82]. With the continuous advancement in AI models and the intensive interest in harvesting vast knowledge from Comics, a lot of work is being published constantly. This is evidenced by a great number of recent papers as shown in Fig. 1. These contributions shifted the landscape around Comics Understanding, leaving previous surveys outdated [9]. Nowadays, the Comics Understanding research community lacks a comprehensive survey that catalogs these papers and the various challenges in the Vision-Language domain.

To address this need, we introduce in this paper the novel Layer of Comics Understanding (LoCU) framework. This taxonomy, illustrated in Table 1, delineates tasks by their input/output modalities and the spatio-temporal dimensions necessary for reasoning over comic data, inviting a structured and progressive approach to comics analysis. Motivated by the uniquely broad task distribution of the comics medium, this survey aggregates and examines all extant research on machine learning in comics, aimed at highlighting critical unresolved issues in the field. It is the first to review the breadth of comics research in a hierarchical structure, and to our knowledge, the first to propose a big-picture taxonomy of comics intelligence from a task-oriented perspective.

Within this framework, the survey enumerates the state of the art in Comics Understanding tasks, including image classification, object detection, semantic segmentation, style transfer,

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Prompt

Prompt bbox

[Figure 18]

[Figure 19]

| |
|---|

Single image

[Figure 20]

[Figure 21]

Single Image

Image CLS

[Figure 22]

Panels Characters Textboxes

[Figure 23]

[Figure 24]

Super Resolution

[Figure 25]

comic

Augmentation

characterstextbox

| |
|---|

| |
|---|

| |
|---|

| |
|---|

panel character

[Figure 26]

| |
|---|

| |
|---|

| |
|---|

bbox

[Figure 27]

textbox

Grounding

Faces

|[Figure 28]|
|---|
|[Figure 29]|

|[Figure 30]|
|---|

face

| |
|---|
| |
| |

| |
|---|
| |
| |

Text-Char. ass.

Analysis

...

Emotion CLS

[Figure 31]

###### Tagging

| |
|---|

[Figure 32]

angry

Style transfer

[Figure 33]

class

boat

Model

Objects

[Figure 34]

Model

Objects Detection

| |
|---|

[Figure 35]

Model

Model

[Figure 36]

Action CLS

| |
|---|

| |
|---|

[Figure 37]

| |
|---|

| |
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

flying

Vectorization

Dialog Transcr.

Character Re-Id.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Page Stream

Depth estimation

[Figure 49]

[Figure 50]

front page first page story page

| |
|---|

Single Image

[Figure 51]

green dress

Multiple images

Single Image

[Figure 52]

Segm.

Free vocab. Ground.

Translation

[Figure 53]

[Figure 54]

bbox

Prompt

|A panel with two men, one falling down the grass and the other with an extreme surprise face sunning away from a flying bullet.<br><br>class| |
|---|---|
| | |

Text 2 Image Retrieval

[Figure 55]

Text

[Figure 56]

[Figure 57]

panel character

Segmentation

Modification

[Figure 58]

textbox

###### Model

face

Retrieval

...

[Figure 59]

[Figure 60]

|[Figure 61]| |
|---|---|
| | |

boat

| |In the panel, three girls are talking each others. The one in the background wear a red dress and [...]|
|---|---|
| | |

[Figure 62]

Image 2 Text Retrieval

Image Impainting

Single image

Model

|[Figure 63]| |
|---|---|
| | |

Model

[Figure 64]

Image + [prompt] (masks)

Instance Segmentation

[Figure 65]

Composed Image Retrieval Image

[Figure 66]

|turn it into a still from a western|
|---|

|but fighting|
|---|

Image editing via Text instructions

|[Figure 67]| |
|---|---|
| | |

Images Texts

+ Instruction

Databases

Single Image

Composed image+text

answer

|caption|
|---|

| |[Figure 68]|
|---|---|
| | |

| |yellow|
|---|---|
| | |

###### Image 2 Text Generation

[Figure 69]

[Figure 70]

[Figure 71]

3D Character Generation

[Figure 72]

|story description|
|---|

Understanding

VQA

question

|[Figure 73]|
|---|

Generation

|character description|
|---|

|what is the color of the floor ?|
|---|

answer

Synthesis

| |the one with red dress|
|---|---|
| | |

[Figure 74]

[Figure 75]

[Figure 76]

Grounded Image Captioning

question

[Figure 77]

| | |
|---|---|
| | |
|[Figure 78]| |
| | |

|which lady is in background?|
|---|

Visual Reasoning

Model

Story 2 Video Generation

Character images

Images

Model

Model

dialog

[Figure 79]

| |Q: what color is the floow? A: yellow Q: why he is probably not at Brown’s ? A: Because he went home to eat.|
|---|---|
| | |

dialog

[Figure 80]

[Figure 81]

|[Figure 82]|
|---|

|Q: what color is the floow? A: yellow Q: why he is probably not at Brown’s ?|
|---|

Text 2 Image Generation

at home, read new paper #at home, The newspaper says there is a treasure house in the forest.[...]

[Figure 83]

[Figure 84]

Single Image

Multi-page Story Generation

[Figure 85]

|at home, read new paper #at home, The newspaper says there is a treasure house in the forest.[...]|
|---|

Visual Dialog

| | |
|---|---|
| | |
| |[Figure 86]|
| | |

Scene-graph Generation

Full-Book Story description

Story description

- Fig. 2. Visualization of the tasks in Layers of Comics Understanding. From panel level to multipage, from unimodal to multimodal, and from simplest to more complex.

multimodal retrieval, synthesis, and generation. Our comprehensive analysis uncovers fundamental gaps—specifically in data availability and task definition.

In summary, the main contributions of this work are threefold. First, it presents a systematic review of Comics Understanding tasks. This is the first survey of comics tasks for the Vision-Language domain, which provides a big picture of this promising research field with a comprehensive summary and categorization of existing studies. Second, it studies the up-to-date progress of Comics Understanding, including a comprehensive benchmarking and discussion of existing work over multiple public datasets. Third, it shares several research challenges and potential research directions that could be pursued in VLMs for Comics Understanding.

The rest of this survey is organized as follows. Section 2 introduces the Comics medium as a composition of Visual-Language tasks, and several related surveys. Section 3 describes the Foundations of Comics, including their structure, similarities, and differences among various types. Section 4 introduces existing tasks and the commonly used datasets in Comics Understanding, also focusing on evaluation procedures. In Section 5 we propose the Layer of Comics Understanding taxonomy (LoCU), and for each layer a collection of tasks grouped by difficulty. In Section 6 we start with Layer

- 1: Tagging and Augmentation, Section 7 with Layer 2: Grounding, Analysis, and Segmentation, in Section 8 with Layer 3: Retrieval and Modification, Section 9 with Layer 4: Understanding, and in Section 10 with Layer 5: Generation and Synthesis. Finally, we conclude with Section 11.
- 2 BACKGROUND

This section first presents the development of the comics research field based on the learning paradigm and how it evolves towards more complex tasks. Then, we introduce the main limitations encountered by the research community, as well as discuss several related surveys to highlight the scope and contributions of this work.

#### 2.1 The Comics Research Epochs

The development of the comics field can be broadly divided into three phases, including (1) Traditional Machine Learning, (2) Deep Learning, and (3) Modern Foundational Models. In what follows, we

introduce, compare, and analyze the three paradigms and how they shaped Comics advancements in detail.

Traditional Machine Learning. Early machine learning approaches heavily depended on feature engineering, utilizing hand-crafted features [3, 27, 197], which were capable of, e.g., classify the style of a manga using SVMs [28], retrieve a manga character from a sketch with edge orientation histograms [131] or segment comic balloons with energy functions and active contour models [166]. However, these paradigms require domain experts to craft effective features for specific tasks, which do not cope well with complex tasks and have poor generalizability. Moreover, the feature engineering obstacles the extension of the Comics Understanding to more tasks.

Deep Learning approaches. With the development of deep learning, researchers have achieved major improvements by using end-to-end trainable deep neural networks (DNNs). These models eliminate the need for complex feature engineering and shift the focus to designing network architectures that learn features automatically. Applying these paradigms to comics came as a natural step. Models became better at detection [145, 157] and segmentation [44, 47, 141], and more complex tasks started being investigated and solved (e.g. visual and textual closure [82]). However, the turn from traditional machine learning toward deep learning raises a new grand challenge: the laborious collection of large-scale, task-specific, and crowd-labeled data in DNN training. Moreover, while deep learning methods excel in specific tasks, they are often difficult to adapt to multi-task. In the domain of comics, where data is scarce, protected by copyright, and challenging to annotate, these constraints have significantly hindered the development of more generalized, multi-task models for Comics Understanding.

Modern Foundational models. The era of foundational models marks a significant evolution in the application of AI to comics, with models that can learn from vast amounts of unstructured data to perform a variety of complex tasks [118, 149, 158]. These models relax previous limitations by “solving” simple yet important tasks as a zero-shot counter effect, e.g. panel, character, and text-box detection [170]. By leveraging these foundational models, more advanced tasks can be proposed and takled [169, 207], opening the doors to yet unexplored more complex ones.

#### 2.2 Relevant Surveys

To the best of our knowledge, this is the first survey that reviews comics understanding publications with a task-oriented approach for Vision-Language. Several relevant surveys have been conducted, examining broader aspects of visual art [176] and, more specifically, comics [8, 9, 135]. Notably, Augereau et al. [9] categorized comic research into three primary areas: (i) content analysis, (ii) content generation and adaptation, and (iii) user interaction, which remain relevant entry points for understanding this field’s research and applications.

In contrast, building on recent advancements, our survey focuses on two key aspects: 1) the recent progress in Comics Understanding tasks and datasets, and 2) the application of Vision-Language tasks to the unique characteristics of comics.

#### 3 COMICS FOUNDATIONS

Our preliminary analysis prioritizes understanding the structural intricacies of comics. The term “comics” broadly denotes the entire medium in its uncountable form, while “a comic” refers to a singular entity within this medium, such as a comic book or strip. The structure of comic books has been a subject of extensive study, examining aspects like their commonalities, layout structures [35], content arrangement, narrative transitions [31–34, 38], and stylistic narrative variations [4, 37, 95].

Typically, comics are presented in single or double-page formats, influenced by the artist’s style. They narrate stories across various genres or series, such as superheroes and science fiction tales, sometimes interlinking through crossovers. Notably, artistic consistency within a series or genre,

- Table 1. Overview of Vision-Language Tasks of the Layers of Comics Understanding. The ranking is based on input and output modalities and dimensions, as illustrated in Supplementary Materials.

Layer Category Task Input Output

- 0

- (Sec.5)

View

- (Sec. 5.1) Basic Image Viewing (BIV) Text Command Image Display Image Classification (I-CLS) Image Tag

Emotion Classification Comic Panels/Images Emotion Labels Action Detection Multiple Panels Tag

Tagging

- (Sec. 6.1) Page Stream Segmentation (PSS) Images Tags sequence

Image Super-Resolution (ISR) Image Image Style Transfer (ST) Image Image

Vectorization Comic Panels/Images Vector Image

1

- (Sec.6) Augmentation

(Sec. 6.2)

Depth Estimation Comic Panels/Images Depth Map Object Detection Tag/s + Image Bounding Boxes

Grounding Character Re-identification (Character ID) Multiple Panels Tag 2

- (Sec. 7.1) Grounding (IG) [Prompt] + Image Bounding Boxes Character-Balloon Association (Speaker ID) Character + Balloons Tag

Analysis Dialog transcription Image Text

- (Sec. 7.2) Translation Image Text

- (Sec.7)

Segmentation

- (Sec. 7.3) Instance Segmentation (IS) [Prompt] + Image Segments Image-Text Retrieval (IR) Text Image

Retrieval Text-Image Retrieval (TR) Image Text

- (Sec. 8.1) Composed Image Retrieval (CIR) Text + Image Image Image Inpainting (II) Text + [prompt] + Image Image

3

- (Sec.8) Modification

(Sec. 8.2) Image Editing via Text (IET) Text + Image Image

Visual Question Answering (VQA) Image + Text Text Visual Dialog (VisDial) Image + Dialog + Text Text

4

- (Sec.9)

Understanding

(Sec. 9.1) Visual Reasoning (VR) Image + Text Text Image-2-Text Generation (I2T) Image Text

Grounded Image Captioning (GIC) Image Text + Bbox Text-2-Image Generation (T2I) Text Image

Scene Graph Generation for Captioning Comic Panel Scene Graph

Generation (Sec. 10.1)

Sound Generation from Single Panel Single Comic Panel Sound/Audio 3D Model Generation from Images (3DGI) Collection of Images 3D Model

Video Generation from Text (VGT) Complex Long Text Video

5

- (Sec.10) Synthesis (Sec. 10.2) Narrative-Based Complex Scene Generation (NCSG) Detailed Narrative Text Series of Images

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

Objects detection Panels sequence

Linking

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

| |
|---|

| |
|---|

5

| |
|---|

| |
|---|

4

1

| |
|---|

[Figure 116]

| |
|---|

| |
|---|

| |
|---|

2 3

| |
|---|

[Figure 117]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Repeating character

Multiple pages

Reading Order

| |
|---|

| |
|---|

| |
|---|

| |
|---|

6

| | | | |
|---|---|---|---|
| | | | |
| | | | |

8 10

11

[Figure 118]

###### 1 2 3 4 14 15 16

[Figure 119]

[Figure 120]

Dialog

7 9

Characters Names

Narrator: "As the tramp steamer [...]" Sailor 1: "THAT DISGUISE DOESN'T [...]" McWhustle: "HOOT, LADDIE! BEFORE [...]" McWhustle: "HOOT, MON! WE'RE AGROUND" Captain: "THERE ARE NO REEFS IN THESE [...]"

- 1
- 2
- 3
- 4

Narrator, Sailor 1, McWhustle, Captain, Matey, Sailor 2, Sailor 3, Sailor 4

- Fig. 3. Anatomy of Comic Page Elements: This illustration delineates the various components found on comic pages including object detection, linkages between elements, the designated reading order of texts and panels, as well as key textual features like character names—all arranged in a logical sequence throughout the panels and pages.

often achieved through collaborative efforts, is a key feature. Each comic page (as illustrated in Figure 3) encapsulates individual scenes within panels of diverse types - from splash panels for opening scenes to widescreen panels for expansive narratives. These panels convey interactions and dialogues through speech or thought balloons, and expressive onomatopoeia, enriching the

- Table 2. Overview of Comic/Manga Datasets and Tasks, including information on availability, published year, source, and properties (languages, number of books and pages). Accessibility is marked with ✗ for no longer existing datasets, ✗ for existing but inaccessible datasets, and ✓ for accessible ones. The link directs to project websites, while  links to dataset websites. The tasks identifiers corresponds to Image Captioning (IC), Object Detection (OD), Re-Identification (RI), Linking (L), Segmentation (S), Dialog Generation (DG), and Analysis (A).

From Name Year Acc. Lang. Origin # books # pages IC OD RI L S DG A

Fahad18 [90] 2012 ✗ - - - 586 x x eBDtheque [62]  2013 ✓ GB, FR, JP 1905-2012 25 100 x x

[62] sun70 [195] 2013 ✗ FR - 6 60 x x

Ho42 [73] 2013 ✗ - - - 42 x

[62] SSGCI [100]  2016 ✗ GB, FR, JP 1905-2012 - 500 x Sequencity [137] 2017 ✗ GB, JP - - 140000 x BAM! [218] 2017 ✗ - - - 2500000 x x COMICS [82]  2017 ✓ GB 1938-1954 3948 198657 x

[52] JC2463 [157] 2017 ✗ JP - 14 2463 x AEC912 [157] 2017 ✗ GB, FR - - 912 x GCN [45]  2017 ✗ GB, JP 1978-2013 253 38000 x x

[137] Sequencity612 [137] 2017 ✗ GB, JP - - 612 x [52] Comics3w [66] 2017 ✗ JP, GB - 103 29845 x

comics2k [78]  2018 ✗ - - - - x DCM772 [138]  2018 ✓ GB 1938-1954 27 772 x [52] Manga109 [52, 145]  2018 ✓ JP 1970-2010 109 21142 x x x x [137] Sequencity4k [141] 2020 ✗ GB, FR, JP - - 4479 x [82] EmoRecCom [143]  2021 ✓ GB 1938-1954 - - x

BCBId [48]  2022 ✓ BD - 64 3327 x [52] COO [10]  2022 ✓ JP 1970-2010 109 10602 x [82] COMICS-Text+ [186]  2022 ✓ GB 1938-1954 3948 198657 x

VLRC [35]  2023 ✗ JP, FR, GB, 6+ 1940-present 376 7773 x

PopManga [170]  2024 ✓ GB 1990-2020 25 1925 x x x x mix CoMix [207]  2024 ✓ GB, FR 1938-2023 100 3800 x x x x

narrative experience. Artistic variations in style and color also play a significant role in comics, ranging from detailed realism to abstract forms, with the color palette setting the mood and atmosphere. Despite rigorous analyses [30], no significant differences have been identified in terms of attention, subjectivity, and viewpoint between Manga and American comics. Hence, these distinctions are better explored within their distinct historical, cultural, and artistic contexts:

American Comics: With a rich history dating back to the 1930s, American comics are celebrated for their colorful and high-quality presentations, primarily known for their superhero narratives. They adhere to a top-to-bottom, left-to-right reading format and are characterized by a dialogue-rich and brisk storytelling style.

Manga: Emerging in post-World War II Japan and influenced by diverse sources, including Japanese traditional art and American comics, manga is recognized for its unique black-and-white style, though colored variants exist. Covering a broad range of genres, Manga is read right-to-left, top-to-bottom. This style is marked by a distinctive “Japanese Visual Language” that is consistent across genres [36].

#### 4 TASKS AND DATASETS

This section summarizes the commonly used Comics datasets, as detailed in Table 2. Some of these datasets are either intended for training or evaluation, depending on their sizes.

#### 4.1 Annotations overview

The available current comics datasets were introduced for different tasks. From Table 2 it is noticeable that the available datasets vary significantly in size, languages, origin, and -more importantlyannotations, making them cover different tasks.

eBDtheque [62] offers comprehensive annotations in high quality for Character, Balloon, and Panel detection, along with Text-Character association. This makes it usable for object detection and linking tasks but it comes with a drawback: it is composed of only 100 pages. In contrast, the COMICS dataset [82] focuses on a set of tasks called closure, based on automatic panel and text detection over a large number of pages. However, it was introduced with poor annotations. Only recently some works [186] have improved OCR text detection on the COMICS dataset. In the same vein, [211] shows that improving OCR accuracy (using Textract) could lead to marginal accuracy gain in closure tasks for the COMICS dataset. Both [186, 211] released the novel OCR annotations to the research community. It is to note also that some of the dataset with high-quality annotations and large number of pages are not accessible. For instance, BCBId [48], though promising high-quality annotations, falls short in delivery. Similarly, VLRC [35], while rich in annotations, lacks digital applicability due to its focus on physical books. Manga109 [52] is the first to provide high-level annotations, covering more than 20k pages. The dataset has been improved over several iterations, further expanding the scope of the dataset by adding annotations for onomatopoeias [10] and dialog [105]. The current state of Manga109 is the result of many years of annotation and improvement by subsequent developments. More recently, following the Manga109 wave, PopManga [170] has been proposed as an English-language alternative, including annotations for recognition and linking, but on a smaller scale (2k images).

In summary, while datasets for Comics Understanding exist, there is a need for more approaches to standardize and unify their annotations. The Comics Dataset Framework [210] proposed a unification of detection annotations as well as a standardization of metrics and settings used to train and measure models. The work was subsequently extended in CoMix [207], where mainly English-language comics (and manga) were selected and the annotations extended to the task of single-page dialogue transcription. In this work, performances of several state-of-the-art models, including Magi [170], CLIP [158], DINOv2 [149], and GPT4 [148], were measured, across various datasets, for the first time.

However, as highlighted in the first column of Table 2, the complexity of annotations and tasks is limited. While object detection and linking are integral tasks to understanding the narrative and dialogue in comics, we must advance toward more complex tasks found in Vision-Language research. These will be discussed within our taxonomy in the next section.

#### 4.2 Datasets overview

The domain of Comics Understanding encompasses a range of datasets focused on processing and examining comic media. A review of various datasets, as detailed in Table 2, reveals that several are no longer accessible (marked with ✗) or have restricted access (✗). Those available (✓) often require

approval from respective research groups to obtain them. This is a common practice in the field of “sharing copyrighted materials” as in fact they can be used only for research purposes in the majority of cases. If the copyright does not allow for redistribution, it is a common practice to release annotations and a script to download and structure the dataset accordingly (as for [170]). As we can see from Table 2, many datasets share common origins (column “From”).

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

###### COMICS DCM EBD MANGA

Fig. 4. Example of different comic datasets, from Black-and-White to Color, from comics-style to Manga-style.

Notably, these datasets vary significantly in terms of style, quality, and size. The common languages of these datasets are English, French, and Japanese and none of the available data seems to be collected from recent books (after 2010), probably due to copyright limits. We found the above limitation to be one of the main issues in the comic domain: (almost) every work proposes its own (sometimes private) dataset on which results are drawn with little comparison, making these works hard to replicate. Datasets are limited in size and the model’s code and weights are rarely shared. Luckily, this has started changing recently [170, 207, 210].

In Table 3 we report, for every dataset, the declared numbers of annotations available. On the right-most column, we report the number of classes annotated, which correspond to different class objects for the detection, or the number of classes for classification annotations. For instance, in Fahad18 there are 18 characters, thus the number of classes is 18. However, in eBDtheque, there are 5 classes, but only panel, character, and face annotations. This is because the characters are separated into three sub-categories: humanlike, animal-like, and object-like. Here, we describe the datasets chronologically and provide an overview of the image style in Figure 4.

Table 3. Number of single instance annotation (declared) in papers. The “Obj. cls” stands for Object-based classification, and refers to the number of classes available to classify the detected instances in fine-grained classes. Colored images are indicated with , while black and white images with . The only dataset that provide Onomatopoeias annotations is PopManga (61465).

Mode / Name Panel Char. Face Text Cls

Fahad18 [90] - 18 - - 18 Ho42 [73] 200 100 - - eBDtheque [62] 850 1550 - 1092 5 SSGCI [100] - 50 - - 1 COMICS [82, 186] 1229664 - - 2498657 2 Comics3w [65] 159529 - - - 1 JC2463 [157] - - 15801 - 2 AEC912 [157] - - 8184 - 2 DCM772 [138] 4470 10757 5438 - 4 Manga109 [10, 52, 145] 103900 157152 118715 147918 4 EmoRecCom [143] 10199 - - - 8 PopManga [170] - 18783 - 20843 2

eBDtheque: The eBDtheque1 [62] includes 100 pages from 20 books, mainly in French, with some English and Japanese comics. [165] labeled 850 panels, 1,092 balloons, 1,550 characters, and 4,691 text lines, offering expert annotations. Balloon labels include instance segmentation and classification (speech, thought, narration, illustrative).

/ CoMix [207] 22176 48903 32625 37923 4

Manga109: The Manga1092 [52] dataset includes 109 manga volumes from 93 authors, with copyrighted material obtained under a “restrictive term-of-use”, allowing use for research and publications with proper citation. The mangas, published between the 1970s and 2010s, span 12 genres. The dataset was extended by COO [10], which added onomatopoeia annotations (5.8 per page on average), and Manga-Dialog [105], which linked text boxes to characters.

COMICS: The COMICS3 [82] dataset contains 1,229,664 panels with automatic textbox transcriptions from golden-age American comics (1938-1954). It includes 3,948 books from the Digital Comics Museum and manually annotated bounding boxes for 500 pages and 1,500 textboxes. Later works improved OCR transcriptions [186, 211], with pipelines developed for better OCR processing and transcript extraction. [1] added dialog prediction based on "persona" information.

DCM772: The DCM7724 [138] contains 772 (pages) images from 27 golden-age comic books from the same COMICS source. Annotations, however, are more precise as they contain panels, the associated characters bounding boxes, and also face boxes.

- 1http://ebdtheque.univ-lr.fr/registration
- 2http://www.manga109.org/index_en.php
- 3https://obj.umiacs.umd.edu/comics/index.html
- 4https://git.univ-lr.fr/crigau02/dcm_dataset

#### 4.3 Datasets for Evaluation

Across the datasets presented in Table 2, two recent datasets are provided as evaluation sets. In particular, PopManga proposes two splits (validation and test) while CoMix enhances annotations for various existing datasets among which also PopManga.

PopManga: The PopManga5 [170] dataset contains English manga titles from the most popular mangas. The dataset contains two test splits: test-seen with 1,136 pages and test-unseen with 789 pages, obtained from 15 and 10 books respectively. The test sets have been annotated with Text (both textboxes and onomatopoeias annotated as text), and Characters.

CoMix: The CoMix6 [207] dataset contains English and French comics and manga titles from the most popular datasets: eBDtheque, COMICS, DCM, and PopManga. The dataset contains two splits: validation (available with annotations) and the held-out test split (available only through the evaluation server7). It contains 3.8k images across 100 books, and annotations for multi-task, from detection and linking to dialog generation.

#### 5 TAXONOMY: LOCU

Developing a comprehensive taxonomy for comics within the Vision-Language (VL) domain requires a careful overview of existing work in VL surveys and tasks. Our approach synthesizes insights from various benchmarks and taxonomies, forming the foundation for our unique classification – the Layers of Comics Understanding (LoCU).

Relevant Taxonomies. Inspiration for this taxonomy comes from multimodal benchmarks like MMMU [234], which highlight the need for a structure addressing modal complexity and task-specific intricacies in comics. The dimensional analysis in video understanding from [50] also informs our spatial and temporal considerations for comics. In VL evaluation, [51] reframes tasks as Visual Question Answering (VQA), but lacks a comprehensive structure for comics-related tasks. Challenges in multimodal learning from [11] guide our classification of VL tasks, with a task-centric focus. [103] provides a foundational classification of VL tasks into grounding, retrieval, understanding, and generation. Their work is crucial in shaping our approach, and we build upon their framework by extending these categories to account for comic-specific tasks. In particular, we add categories like tagging, augmentation, analysis, segmentation, modification, and synthesis, which better capture the diversity of tasks observed in comic studies and expand the structure to fully represent the unique challenges in comics understanding.

In Table 1, we present 31 Vision-Language multimodal tasks, categorized according to the modalities involved, forming the backbone of the LoCU framework. This taxonomy, spanning 10 distinct task groups across five layers, aims to capture the multifaceted nature of Comics Understanding. Layer 0, representing fundamental image processing and viewing capabilities, sets the stage for increasingly complex interactions in the subsequent layers. More details are provided in Supplementary Materials. This structured approach allows for a comprehensive and nuanced understanding of the range and complexity of tasks within the comics domain, laying a foundation for future exploration and innovation in comic analysis and understanding.

#### 6 LAYER 1: TAGGING AND AUGMENTATION

##### The first layer comprises tasks with unimodal input (single image or sequence of images) and unimodal output (Images or Text): Tagging and Augmentation.

- 5https://github.com/ragavsachdeva/Magi
- 6https://emanuelevivoli.github.io/CoMix-dataset 7Accessible at Robust Reading Competition website

#### 6.1 Tagging

Tagging tasks predominantly involve classification outputs, albeit with varying input types (Fig. 5).

[Figure 141]

Single image

Image CLS

[Figure 142]

comic

###### Tagging

Emotion CLS

Model

angry

- 6.1.1 Image Classification. Definition: The task of Image Classification requires classifying an image into one of several predefined categories. In the domain of comics, the task can be applied at the panel or page level. In both cases, the classes vary from artist names, comic/manga styles, or image type (whether it’s an ad, a front page, or a story page).

Early efforts in this domain, such as those by Chu et al. [28], focused on classifying manga panels by artistic style using Support Vector Machines (SVM) and manga-specific feature vectors. Their feature vectors are built on edge-detected lines creating a 20-dimensional vector from elements such as angle, orientation, the density of segments, etc. Hiroe et al. [70] employed a novel method using SVMs to classify comic books based on the frequency of exclamation marks, demonstrating marks-per-page histogram as an innovative approach to content approximation. Daiku et al. [40] expanded the scope to page-level classification within manga stories using Convolutional Neural Networks (CNN), distinct from Page Stream Segmentation (discussed in Subsection 6.1.4). Jiang et al. [84] introduced the Consensus Style Centralizing Auto-Encoder (CSCAE) for style classification, particularly distinguishing between Shonen (for boy) and Shojo (for girl) manga styles using robust style feature representations. They used low-rank and group sparsity constraints for consensus for the classification of manga character faces. Lastly, in [202] Terauchi et al. presented an approach for classifying manga based on the author’s unique style, employing a Variational Autoencoder (VAE) for this purpose. They also proposed a “four-scene comics story dataset” for this classification task into Mow, Seinen, and Shonen touches (styles). In a more recent work [221], the authors tackled the challenge of (multi-)genre classification at the story level in Manga, introducing a Panel-Pageaware Comic genre classification model that takes page sequences as input and produces class-wise probabilities. Their proposal relyied on the attention mechanism to jointly attend page features and panel boxes, then pooled into a transformer encoder. The transformer classification token is merged with a Graph Convolution Network (GCN) over the labels graph, which predicts the probability distribution of the labels for the comic stories.

- 6.1.2 Emotion Classification. Definition: Emotion Classification in comics involves categorizing a single panel or an entire page (looking both at the image and the text) into predefined emotional categories, enhancing the understanding of the narrative’s emotional context.

[Figure 143]

Action CLS

[Figure 144]

[Figure 145]

flying

[Figure 146]

[Figure 147]

[Figure 148]

Page Stream

front page first page story page

Multiple images

Segm.

Fig. 5. Illustration of Tagging tasks.

Tanaka et al. [199] conducted a seminal study on the relationship between speech balloon shapes and emotions in comics, using the Manga109 dataset to analyze how different balloon styles convey emotions. The EmoReCon challenge at ICDAR 2021 [143] pushed the field forward by tasking participants with classifying 8 emotions from comic panels using a specially designed dataset. One winning team employed a three-level fusion strategy (early, mid, and late), combining EfficientNetB3 [198] for visual embeddings and RoBERTa [121] for textual embeddings. Their midfusion approach outperformed other methods, showcasing the power of multimodal fusion. Another top team used an early fusion method, integrating ResNet50 image features with OCR-extracted text processed by a transformer encoder. They predicted emotions using a weighted average of cls tokens, achieving results comparable to the previous team. Beyond character emotions, Sanches et al. [173] analyzed readers’ emotions by tracking physiological signals. By monitoring changes in

the reader’s physiological responses, they classified manga into three genres: comedy, romance, and horror. Yang et al. [227] introduced two new datasets, MangaAD+ and MangaEmo+, based on Manga109. MangaAD+ adds unannotated elements like shop names and ads, while MangaEmo+ includes annotations for 8 emotions on each page for multi-label classification. Using a modified Faster R-CNN for text detection and EfficientNet for visual features, they tackled the emotion classification task, though neither dataset has been made publicly available.

- 6.1.3 Action Detection. Definition: Action Detection in comics is characterized by the analysis of sequences of consecutive panels. This task involves processing multiple images to classify the depicted sequence into one of several predefined action categories.

Unique to Action Detection is its multimodal input nature, where each panel in the sequence contributes to a holistic understanding of the action. Unlike single image analysis, this task requires the model to interpret a series of images, each adding context and continuity to the narrative. Despite being popular for Videos [245], this task has not yet been explored in comics apart from image-cloze task [82] where the model is challenged to select the correct next panel from two options. In the context of clozure, many works have proposed valid solutions, starting from LSTM-based original solution of [82], the BERT-style multimodal and multigranularity embedding of [188], and the Multimodal T5-based approach of [211]. However, while elementary, this task underscores the potential for more sophisticated Panel sequence analysis (and Action Detection) in comics, a domain where sequential imagery plays a pivotal role. Recently, [77] propose panel-level benchmarks about text and character recognition, as well as visual-cloze tasks in manga-style comics benchmarking recent open-source MLLMs models.

- 6.1.4 Page Stream Segmentation. Definition: Page Stream Segmentation (PSS) in comics involves dividing a comic book into different sections or stories. In its simplest form, this entails categorizing pages with tags like cover, regular page, ads, or last page. In a more advanced form, PSS extracts key information such as story titles, authors, illustrators, prices, characters, and synopses.

Though PSS is not a new concept in document analysis, it has seen various methods evolve over time. Traditional approaches like Support Vector Machines (SVM) for single-page classification [58] have given way to Convolutional Neural Networks (CNNs) [217] and more recent attention-based multimodal fusion techniques [41], which have significantly improved the understanding of page layouts and structures in documents. However, PSS remains largely unexplored in the comics domain, where the task is complicated by the unique structure of comics. A single volume may contain multiple interwoven stories, ads, and narrative text, creating a more complex segmentation challenge. This complexity requires more sophisticated methods for segmenting and understanding the diverse content of comic books. An emerging area of interest in PSS for comics is not just segmenting the pages but also tagging each segment with metadata such as story titles, authors, and synopses.

[Figure 149]

This expands traditional image classification into a more detailed, story-oriented approach, offering a promising field of study.

Single Image

Super Resolution

[Figure 150]

Augmentation

[Figure 151]

Style transfer

[Figure 152]

Model

[Figure 153]

[Figure 154]

Vectorization

#### 6.2 Augmentation

Augmentation tasks, in the context of comics, involve image manipulation techniques aimed at enhancing or altering the visual presentation (Fig. 6). The primary tasks

[Figure 155]

[Figure 156]

Depth estimation

Fig. 6. Illustration of Augmentation tasks.

identified under this category are Super-Resolution (SR), Style-Transfer (ST), Vectorization, and Depth Estimation. Each of these tasks revolves around processing a single image input to produce an enhanced or stylistically altered output.

- 6.2.1 Super-Resolution. Definition: The Super-Resolution task in comics primarily targets the enhancement of image quality, especially for images that are compressed, photograph-based, or scanned.

Recent advancements in manga super-resolution (SR) focus on preserving visual integrity during enhancement. Yao et al. [229] employ deep learning to classify and tailor SR models for manga screentone, ensuring semantic preservation. Dai et al. [39] introduce the Structured Fusion Attention Network (SFAN), which optimizes feature extraction and reconstruction quality through attention modules, achieving superior performance on datasets like Manga109. These methods enhance image quality while maintaining the artistic nuances critical to manga, effectively balancing technical efficiency with artistic integrity.

- 6.2.2 Style-Transfer. Definition: Style-transfer in comics encompasses a range of image-to-image modifications, including photo-to-comic transformation, image-to-image translation, and colorization, characterized by similar input and output attributes.

Photo-to-Comic Transformation: Photo-to-comic has mainly involved mangas and involves replacing colors and textures with halftone patterns. Traditionally, it often consists of manually selecting and adjusting the colors. Methods have evolved from basic pattern generation to sophisticated techniques capturing structural and tonal similarities. Classic approaches relied on algorithmic solutions and were limited in their ability to handle multiple screentone. Recent advancements by Zhang et al. [240] utilize deep learning for semantic region segmentation in cartoon images, followed by screentone application. However, these methods are primarily effective for color-based inputs, with limited applicability to line-based images.

Image-to-Image Translation: Topal et al. [203] explored style transfer as a transfer for the detection task. Specifically, they used style transfer networks like CycleGAN and CartoonGAN to train a detection network, which was then fine-tuned on specific comic datasets for character and face detection. This approach -using style transfer techniques to enhance performances in other tasks- has also been explored in different applications [42, 78]. In particular, the multi-task learning method for dense predictions in comic panels has been used as a medium for comic panel reconfiguration [14] and depth estimation [13].

Colorization: Colorization in comics, particularly manga, involves transforming black-andwhite images into full-color pages, requiring both artistic precision and technical expertise. Furusawa et al. [53] introduced a method that allows users to add color dots to guide an automated process, giving users the flexibility to influence the final output while ensuring the system aligns with their vision. Hensman et al. [67] used Conditional GANs (cGANs) to colorize manga panels by tailoring the model to each reference image, allowing versatile style adaptations but requiring a new GAN for every image, increasing computational demands. Tsubota et al. [205] explored

screentone synthesis, essential for manga, by comparing image translation techniques like pix2pix [79] on the Manga109 dataset. Their approach, using a U-Net architecture [102], stands out by explicitly considering screentone labels, preserving the screentone patterns integral to manga art. Golyadkin et al. [57] introduced a two-stage method for manga page colorization:

###### Colorization

###### Screening

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

««

Fig. 7. Colorization vs. Screening

Pixel2Style2Pixel [164] for color drafts, followed by a conditional GAN to refine the output. This method elegantly combines user-provided color hints and draft colorizations, yielding high-quality results that surpass previous methods. Jiramahapokee et al. [88] further refined the process using a multi-encoder Variational Autoencoder (VAE) to ensure color consistency and integrated CIELAB interpolation [236] to enhance color saturation and authenticity. Their approach produced detailed and visually consistent colorizations, particularly in shading quality.

Screening: Screening, or adding screentone to black-and-white manga drawings, has been a focus of several studies. A notable contribution by Wu et al. [219] introduced a two-stage architecture for this process. The first stage involves generating grayscale shading using a six-layer Swintransformer [122], which then serves as input for the second stage, a reference-based screentone generation module. This algorithmic process extracts screentone patterns from a reference manga and applies them to line drawings, considering the shading nuances provided by the first stage.

These advancements in colorization represent a significant contribution to the domain of comics, particularly manga, where colorization can profoundly impact the narrative and aesthetic appeal of the stories.

- 6.2.3 Vectorization. Definition: Vectorization in the context of comics, particularly manga, involves transforming raster images into vector formats. This process includes operations like screening, structural lines extraction, and converting images into vector graphics, each with unique methodologies and applications.

Structural Lines Extraction: This process, structural lines extraction, focuses on removing screentones to leave only the defining lines of figures and shapes. It can be seen as the reverse operation of the screening, which corresponds to adding the screentone to a structural line image. In this task, Li et al. [102] developed a CNN-based method using a U-Net architecture with residual blocks to effectively extract structural lines from manga pages. This method is adaptable for various applications, including manga retargeting and colorization, and can handle diverse patterns and textures, resulting in clean skeleton sketches.

Manga Vectorization: Various works [96, 125, 178] have proposed different approaches for this task. In particular, Yao et al. [230] developed a method involving adaptive binarization and screentone detection, refining borders, estimating lighting, and compensating for missing strokes, leading to high-quality, resolution-independent rendering. A recent innovative approach, MARVEL by Su et al. [191], employs Reinforcement Learning to train an agent in selecting drawing primitives that recreate raster manga images in vector format. While effective, this approach works with squared manga patches and may result in visual gaps in certain renderings. Moreover, this method is not convenient in shape rendering, as it uses primitives and cannot manipulate them. MARVEL represents an interesting exploration into the use of machine learning for vectorization, highlighting both the potential and the complexities of this task.

- 6.2.4 Depth Estimation. Definition: Depth Estimation within the comic domain refers to the process of predicting the spatial distance between objects and the viewer in a two-dimensional comic panel. It serves a crucial purpose in the reconfiguration of comic panels and unlocks potential developments in digital artistry.

To date, this need has arisen primarily when adapting comics for digital media like smartphones, tablets, and e-readers, where the original layout may not fit the screen optimally. Key methods for reconfiguring panels include: (i) Adjusting panel proportions and rescaling them to fit the new medium while maintaining narrative and visual integrity, and (ii) Cropping panels to preserve

essential elements. Both approaches rely on depth estimation to identify and prioritize key components. A major challenge in this area is the lack of ground truth data for comics. To address this, Bhattacharjee et al. [13] employed Image-to-Image style-transfer, transforming comic images to resemble natural ones, enabling depth estimation using Laplacian edges and feature-based GANs. Building on this, a follow-up study [14], based on their earlier work [15], introduced a Swin Transformer for multitasking, including depth estimation and object segmentation. However, the multi-decoder-head architecture poses scalability challenges due to task complexity. Depth estimation is crucial for adapting comics to various digital formats, ensuring key visual elements are retained while preserving the original artistic intent.

#### 6.3 Satellite Tasks

Some of the tasks that fit under the Augmentation umbrella have not found a space in our taxonomy, e.g. the task of de-warping. The task involves correcting distortions typically found in scanned comic pages. Addressing the de-warping challenge, a recent study by Garai et al. [55] proposed a novel mathematical model to describe the warping process. This model estimates distortion factors based on the boundaries of panels in a comic document image, aiming to rectify these warps effectively. While their approach yielded promising qualitative results, particularly in the image areas, challenges persist in handling text distortion, where some distortions proved unrecoverable.

#### 7 LAYER 2: GROUNDING, ANALYSIS, AND SEGMENTATION

The second layer of our LoCU framework comprises three distinct task groups: grounding, analysis, and segmentation. These tasks dig deeper into the intricate components of comics, focusing on detailed elements like panels, text, and characters and their ordering and associations.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

#### 7.1 Grounding

[Figure 167]

[Figure 168]

Grounding in comics involves identifying and classifying specific elements within a comic’s panel, such as text, characters, and objects (Fig. 8). Under this task name umbrella, many individual tasks belong such as Detection, Character Re-identification, and sentence Grouding.

[Figure 169]

[Figure 170]

Prompt

Panels Characters Textboxes

[Figure 171]

[Figure 172]

| |
|---|

| |
|---|

class bbox

Objects Detection

[Figure 173]

| |
|---|

panel character

[Figure 174]

Faces

|[Figure 175]|
|---|
|[Figure 176]|

|[Figure 177]|
|---|

| |
|---|
| |
| |

| |
|---|
| |
| |

Grounding

textbox

[Figure 178]

face

[Figure 179]

Objects

...

boat

Model

Character Re-Id.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 180]

[Figure 181]

Free vocab. Grounding

[Figure 182]

- 7.1.1 Detection. Definition: In Detection task, given an image and a set of class names (or tags), the goal is to find all instances of those classes in the image and locate them with bounding boxes.

green dress

Single Image

Fig. 8. Illustration of Grounding tasks.

Panels. Initial studies in digital Manga, such as those by Ponsard et al. [155] and Arai et al. [5], emphasized automated segmentation and panel detection, underscoring the importance of maintaining reading order. Arai et al. [6] further developed these ideas with a focus on real-time detection and extraction techniques, which evolved into more sophisticated SVG-based approaches as explored in [190]. [72] contributed advanced methods for panel extraction using region growing techniques and mathematical morphology, combined with speech balloon detection.

Text. Jomaa et al. [89] employed a tracking algorithm for panel extraction, while speech balloons were identified using Robert’s edge detection operator and classified with thresholding. Liu et al. [120] focused on balloon extraction based on text character locations. Pal et al. [150] performed text extraction using SIFT local features, Space Pyramid Matching (SPM), and SVM on eBDtheque and Bangla comics images. In [7] authors contributed to text extraction in Manga109 combining region proposal generation with SVM classifiers. Chu et al. [29] and Qin et al. [157] modified existing

architectures like Faster R-CNN for detecting text and faces in comics. In particular, [29] proposed additional aspect ratio regions, while [157] built two comics datasets, namely JC2463 (Manga black and white style) and AEC912 (colored American comics style) to test face detection evaluation.

Face. Interestingly, in [224] authors benchmarked Fast R-CNN, Faster R-CNN and SSD in panel, character, face, and balloon detection. They used the Manga109 dataset, discovering that Fast R-CNN works better in panels when there are clear boundaries, Faster R-CNN when boundaries are not well defined, and SSD needs already cut panels as it is limited for full-page complex mangas. So far, many methods have used classical detection architectures and applied them to the comic domain to extract elements such as panels, characters, text, etc. A new direction is drawn by [140] challenged the conventional four-edge polygon representation of comic panels, suggesting segmentation-based approaches using U-Net for more accurate extraction. Here, authors performed extraction through segmentation, employing a U-Net that classifies pixels into background, panels, and borders.

Onomatopoeia. Recently, the focus has shifted to onomatopoeia detection, a relatively unexplored area in comic analysis. The release of the COO dataset by Baek et al. [10] has paved the way for new research in this domain. Authors proposed a modified version of M4C [75] designed to link truncated onomatopoeias together. Following this, Yang et al. [227] proposed text augmentation techniques for onomatopoeias using EfficientNet. Sharma et al. [179] furthered this research by introducing their dataset for panel and character detection, employing a Faster R-CNN model. However, they did not provide comparative analysis with other models.

Benchmarks. Lately, Dutta et al. [46, 48] introduced the BCBId dataset, benchmarking existing datasets (eBDtheque, Manga109, and DCM) and employing YOLO architectures for panel and character detection. They presented comparisons with existing methods, showing enhanced performance in detecting panels and characters across multiple datasets. However, it is unclear if these results are truly comparable. This is due to the fact that in previous (and contemporary) works, authors don’t provide neither information about the settings used (in terms of train-val-split), nor about hyperparameters and model settings to replicate the weights. Only recently, Vivoli et al. [210] fairly compare models across various datasets and styles, unifying annotations for multiple detection classes and providing model weights, code, and data to replicate.

- 7.1.2 Character Re-Identification. Definition: Character Re-Identification, also known as character retrieval in comics analysis, has evolved from focusing on visual features alone to incorporating textual information alongside character images.

Initial studies by Khan et al. [90] and Amirshahi et al. [3] employed techniques like PHOG and SIFT keypoints for character retrieval. Iwata et al. [81], inspired by Sun et al. [196], adapted these methods for manga characters, focusing on character-level identification rather than entire page regions, and further explored their applications in [80]. However, Li et al. [109] highlighted the limitations of these handcrafted features, particularly their sensitivity to pose, expression, and shape variations. Li et al. [109] addressed manga character retrieval and verification, the latter being the task of classifying whether two images belong to the same character. They introduced a dual loss approach—combining dual ring loss and dual adaptive re-weighting loss—to address long-tailed distributions and quality variations in manga data, improving robustness. In animated cartoons, where characters frequently change shape and color, Nir et al. [144] proposed a SelfSupervised Learning (SSL) and Multi-object Tracking (MOT) approach to cluster and link character proposals, facilitating character re-identification across scenes. Similarly, Zhang et al. [241] tackled the challenges of character re-identification in manga, where faces are similar but body features differ. They introduced FSAC (Face-body and Spatial-temporal Associated Clustering), which

refines clustering through face-body graphs and spatial-temporal relationships, using a triplet loss to handle artistic exaggerations. To address occlusion in manga, Zhang et al. [237] developed Occlusion-Aware Manga Character Re-identification (OAM-ReID), using self-paced contrastive learning. Their model synthesizes occluded data, such as speech balloons and incomplete bodies, and combines contrastive, occlusion, and re-identification losses, training a ViT and TransReID on 10 Manga109 books. Soykan et al. [187] introduced "Identity-aware SimCLR" for character re-identification, leveraging SimCLR [25] with data augmentations to capture both local (face) and global (body) features. Similarly, Sachdeva et al. [170] proposed Relationformer [182], a multi-task Transformer model for character clustering and linking bounding boxes at the page level. Finally, Ahmad et al. [2] introduced an enhanced YOLOv5 combined with XGBoost for object detection, achieving state-of-the-art performance across multiple datasets, emphasizing the importance of well-annotated data and augmentation techniques in improving detection tasks.

- 7.1.3 Sentence-based Grounding. Definition: Sentence-based grounding is a Vision-Language task that links textual content to specific visual elements within a panel. In comics, this task translates into grounding the sentence elements both into a single panel and a full page. Moreover, the complexity of the task varies depending on what is the sentence to be grounded: a panel description or a dialogue or narrative description.

In image and video Multimedia processing various works tackle the task with custom CNNs and LSTMs [110] approaches, in an unsupervised manner [152] or with custom multimodal fusion approaches [118, 181]. However, the only work that used a similar approach (GroundingDINO) in a zero-shot setting is [170] which used it to automatically annotate the training set.

[Figure 183]

[Figure 184]

[Figure 185]

#### 7.2 Analysis

[Figure 186]

Analysis tasks involve a deeper examination of the relationship between text and visual elements in comics, including character-text association, panel sorting, dialog transcription, and translation (Fig. 9).

[Figure 187]

Prompt bbox

[Figure 188]

[Figure 189]

[Figure 190]

| |
|---|

characterstextbox

| |
|---|

| |
|---|

Text-Char. association

| |
|---|

| |
|---|

###### Analysis

| |
|---|

Model

| |
|---|

Dialog Transciption

[Figure 191]

- 7.2.1 Text-character association. Definition: The task of text-character association in comics aims to link textual content, usually dialogue, to the corresponding speaking character. This task often involves a two-step process: first identifying panels, text boxes, and character boxes, and then associating the text with its speaker.

[Figure 192]

[Figure 193]

Translation

Single Image

Fig. 9. Illustration of Analysis tasks.

Rigaud et al. [168] addressed the text-character association task using geometric graph analysis and anchor point selection, developing a distance-based algorithm that utilized both global (page-level) and single-frame (panel-level) information, relying on Euclidean distances and balloon tail cues. Building on this, Nguyen et al. [139] introduced ComicMTL, a multitask learning approach based on a modified Mask R-CNN with an added "PairPool" head for association prediction. Their method tackled multiple tasks—panel and character detection, balloon segmentation, text recognition, and text-speaker association—demonstrating effectiveness on DCM772 and eBDtheque datasets. Omori et al. [147] took a novel approach by focusing on the continuity of comics, particularly reading order, to associate text with off-panel speakers. However, their approach raised scalability and generalizability concerns. Li et al. [105] expanded the Manga109 dataset by introducing the Manga109 Dialog dataset, refining speaker detection. They categorized text-character links into “easy” (character in the panel) and “hard” (character off-panel), employing Faster R-CNN for detection and association classification. Despite progress, handling alternating speaker dialogue

remains challenging, pointing to the potential of NLP models for complex scenarios. Sachdeva et al. [170] recently proposed a multi-task Transformer model, Relationformer [182], with specialized MLP heads for detection and linking. Their two-step, end-to-end solution first detects and then links characters and text, though their model’s benchmarking, especially against YOLO-based models, remains limited, focusing only on GroundingDino [118] in a zero-shot setting. Lastly, in [189] built on top of Comic-MTL [142] a Mask R-CNN based model for detecting and linking textboxes and characters, but did not fully compared with previous models.

Character-text association is a crucial yet challenging task in comics analysis, requiring an understanding of narrative and visual-textual interplay. Future research could benefit from advanced NLP techniques to manage complex dialogue scenarios, and more unified benchmarks and replicable methodologies are needed, as highlighted by the LoCU framework, to support comparative analysis and innovation in this field.

- 7.2.2 Panel Sorting. Definition: Sorting individual comic panels into their correct narrative order presents a unique challenge, both for humans and automated systems.

Ueno et al. [206] conducted a study revealing that while people can relatively easily identify the first and last panels in a sequence, accuracy diminishes with the presentation of all panels from four-panel strips. This suggests the complexity involved in discerning narrative continuity in comics. The same study attempted to automate panel sorting using an AlexNet-like architecture. However, the CNN’s performance was poor, indicating the difficulty of the task and the need for more advanced or specialized algorithms in panel sorting.

- 7.2.3 Dialog transcription. Definition: Dialog transcription in comics aims to create an automatic transcript of dialogues, identifying the speaker and the sequence of speech.

The process typically involves several steps: (i) detecting panels, text boxes, and character boxes, (ii) associating text with speakers, and (iii) sorting text boxes according to their reading order to generate a coherent dialogue transcript. This task has been addressed by [105] for Japanese Mangas, proposing the needed annotations for Manga109, and more recently by [170] for English Mangas called PopManga. However, a few limitations arise from these approaches: sorting the text boxes is always done as a postprocessing operation, depending only on algorithmic approaches. Cases in which balloons alternate among close panels are inherently mistaken. Recently, Sachdeva et al. [169] proposed Magiv2 which tackled the problem of dialog transcription adding also names (from a name bank) allowing it to operate on the whole collection of pages. In CoMix [207], a first benchmark for this task was proposed, authors enhanced the previous annotations and proposed a new metric for character naming (based on ANLS) and dialog transcription (based on Hungarian matching and edit distance).

- 7.2.4 Translation. Definition: The task of translation in comics involves not only linguistic challenges but also graphical and spatial ones, requiring an integrated approach that respects the visual and narrative structure of the comic. Text appears in various forms such as balloons, onomatopoeias, and scene text, each necessitating different strategies. For instance, Balloon Text must fit within the original speech balloon, preserving visual integrity, while Onomatopoeias may require both linguistic translation and graphic replacement to maintain stylistic consistency.

Recently, several works have addressed these complexities by integrating multimodal information and context-aware techniques. [69] were among the first to incorporate context obtained from manga images into a multimodal translation framework, developing a comprehensive system

for fully automated manga translation. To improve text image translation performance, [124] proposed a Multi-Teacher Knowledge Distillation method, effectively distilling knowledge from a pipeline model into an end-to-end Text Image Machine Translation (TIMT) model. The advent of Large Language Models (LLMs) has further advanced the field; [228] explored leveraging LLMs for manga translation tasks, demonstrating the potential of these models in handling complex scenarios involving both text and visual elements. Similarly, [113] proposed a methodology that leverages the vision component of multimodal LLMs to improve translation quality, investigating the impact of translation unit size and context length, and introducing a token-efficient approach along with a new evaluation dataset. Inspired by human translation workflows, [185] presented an approach to completely automate the process of translating graphic novels, addressing the challenges of automating different aspects of the translation workflow. Focusing on character identification and speaker prediction, [108] proposed an iterative multimodal framework that employs multimodal information for both tasks, allowing machines to identify characters and predict speaker names based solely on unannotated comic images. Furthermore, [115] introduced the task of character-centric story generation, presenting the first model capable of generating visual stories with consistently grounded and coreferent character mentions, emphasizing the importance of character consistency and coreference in narrative understanding and translation. Despite these tasks being tackled as image-to-text translation, they have origins in image-to-image translation due to possible changes in visual components influenced by cultural elements, as discussed by [91]. This indicates that future research may focus more on end-to-end image-to-image translation and transcreation, where both textual and visual elements are adapted to enhance cultural relevance.

[Figure 194]

- 7.3 Segmentation

Segmentation in comic analysis, particularly instance segmentation, presents a nuanced approach to identifying and isolating various elements within a comic panel (Fig. 10).

Prompt

class bbox

[Figure 195]

panel character

### Segmentation

textbox

face

...

boat

Model

|[Figure 196]| |
|---|---|
| | |

Instance Segmentation

- 7.3.1 Instance segmentation. Definition: Instance segmentation in comics extends beyond the scope of typical object detection. Rather than just identifying and classifying elements within bounding boxes, this task involves generating precise pixel-wise masks for each instance.

[Figure 197]

|[Figure 198]| |
|---|---|
| | |

Single Image

Fig. 10. Illustration of Segmentation tasks.

Instance segmentation, as opposed to detection, offers more accurate panel separation and representation in comics, which is crucial when reconfiguring or reformatting them for different layouts. This precision ensures that each panel is cleanly isolated, avoiding remnants of adjacent panels or the original layout—a common issue with less precise bounding boxes. One of the key challenges in instance segmentation for comics lies in the diversity of artistic styles and the complexity of layouts. Comics often feature intricate designs, overlapping elements, and various configurations such as grids, staggered layouts, bleeds, and insets, as noted by Cohn [35]. These complex panel arrangements make it difficult to segment panels accurately using bounding boxes, especially when panel boundaries are not clearly defined. In a recent work [98], authors proposed a pipeline of detection models (YOLO) and segmentations (GroundingDINO) for segmenting objects in comic panels. Despite the idea being simple and elegant -using in a cascade approach a YOLO detector and prompting SAM with the detected bounding boxes- the approach has been only evaluated for the detection tasks as no Segmentation ground truth exists for the

used datasets. Moreover, as noticed in [210], results are not comparable due to different evaluation datasets/splits/criteria.

Although current datasets do not include semantic segmentations, the 2024 “AI for Visual Art” workshop [12] introduced a challenge on comic semantic segmentation, signaling the growing importance of this task in comic analysis.

#### 8 LAYER 3: RETRIEVAL AND MODIFICATION

Layer 3 in the Layers of Comics Understanding focuses on advanced image and text understanding through tasks such as Retrieval (uni-modal and multi-modal) and Modification.

#### 8.1 Retrieval

The task of retrieval, in the domain of the comic, addresses two main retrieval settings in: unimodal and cross-modal retrieval, which do not differ, in practice, from the vision-language retrieval tasks (Fig. 11). These tasks aim to retrieve an instance of the same modality of the query (unimodal) or a different modality from the query (cross-modal), using either text or images as the query. An additional task emerged recently in the realm of Vision-Language, called “Composed image retrieval”, which combines in the query both image and text.

[Figure 199]

[Figure 200]

Text

Text 2 Image Retrieval

|A panel with two men, one falling down the grass and the other with an extreme surprise face sunning away from a flying bullet.| |
|---|---|
| | |

###### Model

###### Retrieval

|[Figure 201]| |
|---|---|
| | |

| |In the panel, three girls are talking each others. The one in the background wear a red dress and [...]|
|---|---|
| | |

Image 2 Text Retrieval

[Figure 202]

Single image

Composed Image Retrieval

|but fighting|
|---|

Images Texts

Composed image+text

Databases

Fig. 11. Illustration of Retrieval tasks.

- 8.1.1 Unimodal Retrieval. Definition: Unimodal retrieval in comics involves searching for a specific media element (image or text) within a database, using a query of the same format (image or text, respectively).

Pioneering works in this area have explored various aspects of comic retrieval. Wu et al. [220] developed a comic retrieval system that leveraged OCR and external text. One significant article [196] focused on retrieving full pages or books containing similar elements. This approach is particularly relevant in copyright violation contexts. The authors implemented feature vectors for each image, storing them in a database, and used these vectors to find the most similar documents. Their feature representation method was based on the Histogram of Oriented Gradients (HOG), which outperformed the traditional SIFT method. Notably, they applied HOG not to the entire images but selectively to Regions Of Interest (ROIs) identified in the comics, such as faces or specific markings. Building on this approach, authors in [81] utilized a similar methodology for manga character retrieval. They focused on identifying and extracting distinctive features from characters, enhancing the ability to retrieve specific characters from large databases.

- 8.1.2 Cross-modal retrieval. Definition: Cross-modal retrieval in comics encompasses tasks where the query and the retrieval results are from different modalities, such as using sketches or text queries for retrieving a comic image.

Sketch-based Retrieval. Matsui et al. [131] introduced a framework for retrieving manga images from sketches using the Fine Multi-scale Edge Orientation Histogram (FMEOH). This method indexed different-sized squares on a page and enabled efficient retrieval from sketch queries. Further advancing this area, authors [132] proposed a two-step system that first processes manga and then retrieves images based on sketches. This framework was one of the first to employ the Manga109 dataset and included margin labeling, objectness-based edge orientation histograms, and approximate nearest neighbor search. Narita et al. [136] explored CNN-based

feature extraction for sketches and manga images (without and with screentone, respectively), demonstrating improvements over algorithmic approaches.

Multimodal Learning and Character Retrieval. Nguyen et al. [142] proposed a multitask multimodal approach for character image and text learning, aiming at tasks such as character retrieval and emotion recognition. Their method focused on leveraging both visual and verbal information in manga content, in a CLIP-like style aiming at performing various character tasks (e.g. detection, retrieval, emotion classification), trained uniquely with visual information and verbal information in manga image content. Later, in [216] authors introduced ComicLib, a dataset for comic sketch research, providing benchmarks across various tasks like colorization, generation, and retrieval. They conducted extensive comparison experiments with other datasets (e.g. QuickDraw [63]) to provide a benchmark for ComicLib on common tasks like colorization (Style2Paints [239]), generation (DCGAN [159]), retrieval (image-to-image with ResNet), detection (with Faster R-CNN [163], YOLOv3 [162], and SSD [119]) and recognition. However, the dataset’s availability is unclear.

Text-to-Image Retrieval. Shen et al. [180] presented a system for text-to-image content retrieval within Manga frames. This multi-staged system integrates object detection, text recognition, and a vision-text encoder to facilitate efficient search and retrieval of dialogues and scenes. Their method is composed of an object detection model for identifying text and frame bounding boxes (DETRbased [19]), a Vision Encoder-Decoder model for text recognition (DiT [204]), a text encoder for embedding text (multilingual DistilBERT [175]), and a vision-text encoder with unified embedding space (CLIP [158]). They experimented with Japanese and GPT-48translate/rephrased English sentences, on the annotated panels from “Dollgun Book”. Despite being a composition of existing systems on panel level, on a small distribution of data, the work is a first direction to more complex Comic retrieval.

#### 8.2 Modification

In the realm of comics, modification tasks such as image inpainting and image editing play a crucial role in streamlining the storytelling process and enhancing creative expression (Fig. 12).

[Figure 203]

Image Impainting

[Figure 204]

[Figure 205]

###### Modification

[Figure 206]

[Figure 207]

[Figure 208]

Model

Image + [prompt] (masks)

[Figure 209]

[Figure 210]

Image editing via Text instructions

|turn it into a still from a western|
|---|

- 8.2.1 Image Inpainting and Editing. Definition: Image inpainting and editing in the context of comics involve various techniques and tools that allow for the alteration or enhancement of comic panels, facilitating more dynamic and engaging storytelling.

Image + Instruction

Fig. 12. Illustration of Modification tasks.

In the realm of comic inpainting and editing, CodeToon [193] offers a code-driven storytelling tool that automatically generates comics from code snippets through metaphorical interpretation. It simplifies the transition from story ideation to comic format, addressing common challenges faced by comic creators. ComicScript [215] provides a suite of operations for interactive comic creation, allowing users to add/remove panels, change perspectives, and support branching narratives. It also integrates data manipulation to enhance engagement. Professional creators validated its potential by crafting interactive comics, showcasing the tool’s ability to personalize storytelling. Both tools highlight the growing role of computational methods in enhancing comic creation and editing workflows.

8Using the website https://platform.openai.com/

#### 8.3 Future Tasks

- 8.3.1 Personalized Image retrieval. Definition: Personalized Image Retrieval (PIR) focuses on developing person-centric models that can efficiently retrieve specific comic panels based on a compound query involving both image and text inputs [97]. PIR aims to identify and retrieve images or videos that correspond to a compound query, which typically includes an image of a person’s face combined with a textual scene or action description.

In the realm of comics, this translates to the capability of retrieving panels where a specific character appears, given an image of the character, the character’s name, or a description of the character. This task becomes increasingly complex when the retrieval criteria include contextspecific actions (e.g., a character eating pizza or jumping out of a window) or interactions with other characters.

#### 9 LAYER 4: ADVANCED VISUAL-LANGUAGE UNDERSTANDING

In the fourth layer, we propose in the Layer of Comics Understanding a set of tasks related to multimodal understanding, which comprehends often reasoning through a text-rich image given a text prompt and one or more images as input. However, the only task that has been explored in comics already is Visual Question Answering, which under its umbrella groups different definitions of VQA in comics.

#### 9.1 Understanding

This layer investigates tasks that require a deeper integration of visual elements and language, advancing beyond basic identification or retrieval to more complex forms of comprehension and reasoning within comics (Fig. 13).

###### Understanding

Visual Question Answering

answer

question

|[Figure 211]|
|---|

| |yellow|
|---|---|
| | |

|what is the color of the floor ?|
|---|

question

|which lady is in background?|
|---|

answer

Model

Visual Reasoning

| |the one with red dress|
|---|---|
| | |

dialog

dialog

|[Figure 212]|
|---|

| |Q: what color is the floow? A: yellow Q: why he is probably not at Brown’s ? A: Because he went home to eat.|
|---|---|
| | |

|Q: what color is the floow? A: yellow Q: why he is probably not at Brown’s ?|
|---|

Visual Dialog

- 9.1.1 Visual Question Answering. Definition: Given an image-question pair, Visual Question Answer requires answering a question based on the image. Most studies treat VQA as a classification problem on a predefined answer set, but its general definition would be an open answer where no specific set of options is provided.

The first notable attempt in the domain of the comic is by [134] with ComicVQA. However, their approach, more akin to retrieval based on specific “Doraemon” comics datasets, diverges from the typical VQA framework. Later, Sumi et al. [194] presented a system, ComicsQA, with a unique twist, converting user QA interactions into comic stories, reflecting the user’s situation and offering solutions. The creation of a comprehensive VQA dataset specifically for comics, encompassing single-panel, single-page, and multi-page formats, remains an open challenge. In a recent survey [235], Zeng et al. show the massive work that has been carried out on VQA on multi-modal data across different mediums (movies, comics, etc.) and Multimodal Machine Comprehension [171, 172]. This, together with multiple datasets across various mediums [16, 129, 184, 200, 200, 209] are laying the groundwork for advancing VQA also in comics.

- 9.1.2 Visual Entailment. Definition: Visual entailment (VE) in comics is an area yet to be deeply explored. This task involves assessing whether a given image semantically entails the accompanying text. The challenge lies in interpreting the visual narrative in conjunction with textual elements to ascertain semantic consistency or correlation. It can be seen as a simplified version of the text-closure task where we

Fig. 13. Illustration of Understanding tasks.

aim at correctly classifying whether the image is represented by the given text, however, depending on the provided text, VE could quickly lead to complex scenarios. Recently, [76] investigated the performances of Multimodal models on comics with contradictory narratives, where each comic consists of two panels that create a humorous contradiction. The paper introduces the YesBut benchmark and assesses Multimodal Models’ limited capabilities in this reasoning task.

#### 9.2 Future Tasks

In the context of Understanding, a number of Vision Language tasks could be of interest in comics, both for pertaining and fine-tuning.

- 9.2.1 Visual Dialog. Definition: The task of Visual Dialogs (VisDial) corresponds to answering a specific question, having an image, a question about the image, and the previous dialog history about the image.

In comics, Visual dialog is an area still in its infancy and requires constructing a narrative dialogue based on visual cues within the comic panels. This task demands a comprehensive understanding of the storyline, character interactions, and visual symbolism to generate contextually relevant dialogues.

- 9.2.2 Visual Reasoning. Definition: Visual reasoning in comics extends beyond simple question answering to include the ability to interpret and analyze the visual content in depth. This task necessitates understanding the objects, their interactions, and the underlying narrative structure within the comic panels. It’s a sophisticated blend of visual comprehension and logical deduction, aimed at uncovering deeper layers of meaning in the comics.

10 LAYER 5: GENERATION AND SYNTHESIS The last layer explores the creative frontier in comics analysis, where the synthesis and generation of comics from various media sources play a pivotal role.

#### 10.1 Generation

The field of Generation in comics has seen lots of attraction on Manga style comics rather than American ones. Their popularity has attracted the research community towards creating comics from various media such as video, images, and text descriptions (Fig. 14).

|caption|
|---|

###### Image 2 Text Generation

[Figure 213]

[Figure 214]

[Figure 215]

|story description|
|---|

###### Generation

character description

[Figure 216]

[Figure 217]

[Figure 218]

Grounded Image Captioning

In the panel, [three girls] are talking each others. [The one in the background] wear a [red dress] and [white shoes] with [red socks]. She is walking away on a [yellow floor]. [The one

| | |
|---|---|
| | |
|[Figure 219]| |
| | |

Images

in the foreground] is wearing [...]

Model

[Figure 220]

[Figure 221]

[Figure 222]

Text 2 Image Generation

Single Image

- 10.1.1 Comics generation from other media. Definition: The task of generating comics from other media comprehends an umbrella of tasks with various input types (video, audio, charts, and comics itself) whose output is a comic panel, a sequence of panels, a page, or a book.

|at home, read new paper #at home, The newspaper says there is a treasure house in the forest.[...]| |
|---|---|
| | |

[three girls]

| | |
|---|---|
| | |
|[Figure 223]| |
| | |

Scene-graph Generation

[The one in the background]

[The one in the foreground]

Story description

[white shoes]

[red dress]

[yellow floor]

[red socks]

Fig. 14. Illustration of Generation tasks.

One of the earliest attempts at comic generation from other media is the “Comic Live Chat” by Matsuda et al. [130], which transformed video meetings into comics. This approach involved selecting keyframes and rendering the dialogues of meeting participants in a comic format. Tanapichet et al. [201] proposed a system for creating comic strips from cartoon animations using optical flow techniques, demonstrating a novel approach to narrative translation from one visual medium to another. Hoashi et al. [74] worked on creating manga previews by detecting panels and text balloons, while in [21, 43] authors explored generating manga faces and caricatures from real

images. Cao et al. [17] and Herranz et al. [68] introduced methods for comic-like video summarization. These systems developed pipelines for layout proposal, image reconfiguration, and balloon creation, treating comic creation as a series of steps from keyframe selection to final panel layout. On the same task, to automatically create comics from video content, Jing et al. [87] pioneered video transformation using TV show subtitles and a speaker detection algorithm to generate comic layouts optimized through an energy-based metric. This approach was a significant step forward compared to earlier methods like [213]. Giving it more time, a groundbreaking approach by Pesko et al. [154] employed neural style transfer, creating visually compelling comic representations from video-sourced data9. While their startup has since become closed-sourced, their initial approach signifies a significant leap in comic generation technology. Yang et al. [226] presented a system to generate comics from movies, employing a multi-page layout framework and emotion-aware balloon generation, showcasing the potential of comprehensive systems in comic generation. To extract keyframes from movies, they used GIST [146] similarity between two frames. To detect the speaker, they perform lip motion analysis and use that to associate balloons with talking characters. Finally, the movie frames are rendered in a comic-layout page after the adaption to comic style. Moreover, in [238] authors represent conversational videos to comics “fixed-layout” representation, developing a system that learns guidance field which provides a prior prediction of the possible positions of word balloons while making the word balloons not overlap with other nonverbal information (e.g. hand gestures, visual clues in the background, etc). Their input is raw video, thus the text dialog is captured from audio and rendered in the location chosen from their method. Comics have been used also in education [20, 111, 192] to teach and explain in an easier way different concepts. An example of this is [242] where authors proposed a system that crafts data stories from a collection of user-created charts, using a comic-style panel to imply the underlying sequence and logic of data-driven narratives.

- 10.1.2 Image-to-Text generation. Definition: Image-to-Text is a general umbrella term that covers tasks from captioning, to textbox prediction in comics, masking a specific part of the dialog.

Motivated by supporting the Blind or Low Vision community (BLV) - also known as People with Visual Impairements (PVI) - Ramaprasad et al. [161] developed a two-step method to create natural language descriptions of comic strips. This method first extracts information about panels, characters, and text using computer vision techniques, followed by the use of a multimodal large language model (MLLM) to generate descriptions. This work stands as one of the few promoting MLLM specifically for comic strip captioning, despite using “out of the box” architectures like Grounding DINO [118] to extract arbitrary elements, CLIP [158] to match character images with name-descriptions, and LLaVa [117] as MLLM. Vivoli et al. [211] proposed an advanced version of the text-cloze task with image-to-text generation, increasing the challenge level. They adopted the VL-T5 model [26] using two configurations with similar performances but different model sizes: with BLIP-2 visual backbone (1.3B) or a ResNet-50 fine-tuned with SimCLR on comics (0.2B). They surpassed previous state of the art in the standard “text-cloze” task, and propose benchmarks in the generation version of the task. Authors in [1] presented a similar task but incorporated character descriptions and transcriptions, emphasizing a language model-centric approach. Guo et al. [59] introduced the multi-modal manga complement task. This innovative task combines visual sequences from comic pages with corresponding text dialogues, challenging the model to complete the narrative appropriately. They propose an effective method with CLIP as Feature Encoding, a custom cross-attention module called Fine-grained Visual Prompt Generation, and a transformer-based encoder-decoder architecture called Dialog complement. In a recent work

9GitHub repository: https://github.com/maciej3031/comixify

[170], authors proposed a Relationformer-based architecture that generates dialog for manga pages. Their approach includes detection, matching, and sorting algorithms to produce an ordered dialog that aligns with the visual narrative. However, all these works are limited by the dialog appearing in the comic, and in the context of describing a comic, not only the dialog is important, but also the scene description. Recent works, to fill this gap, have approached the task by exploring single panel captioning. In [167], the authors explored prompt engineering with contextual information for closed-sourced Multimodal LLMs aiming at generating an accurate text description of a single panel and subsequently of the full story. They successfully demonstrated to retain many of the important details in the caption, but the exploration was limited to a couple of story panels. In [208], authors took a step further proposing a new metric for comic panel captioning called “attribute retaining metric” (ARM) which assesses whether all the objects and attributes of the panel have been identified. They propose a pipeline with open-source Multimodal LLMs and zero-shot grounding models for generating dense and grounded captions for more than 1M panels. However, in the ARM metric only the attributes are taken into consideration, which identifies the need to consider both locations and object/attribute names. Finally, [76] focuses on comics with contradictory narratives, where each comic consists of two panels that create a humorous contradiction, and introduces the “YesBut” benchmark, which comprises tasks of varying difficulty aimed at assessing MLLMs capabilities in recognizing and interpreting these comics.

- 10.1.3 Text-to-Image generation. Definition: The Text-to-Image task comprehends generating an image, or a sequence of images, from a text description.

In the field of Text-to-Image generation, early work by Jin et al. [85] utilized GANs to generate anime-style character faces, demonstrating the potential of GANs for stylized image generation. Inoue et al. [78] advanced the field with domain transfer and pseudo-labeling techniques for image generation. StoryGAN [106] was a groundbreaking development in story visualization, employing a sequential conditional GAN with a context encoder to track story flow and a story-level discriminator. Later improvements included DuCo-StoryGAN [127], which used a dual learning framework for better semantic consistency, and VLStoryGAN [126], which incorporated recursive architectures to handle structured text inputs. Melistas et al. [133] proposed a pipeline for generating synthetic graphic novels, using GPT-2 for text and StyleGAN2 for image synthesis, trained on a large manga collection. Proven-Bessel et al. [156] developed ComicGAN, a text-to-image GAN for generating single-panel comics in the style of Dilbert, while FastGAN [114], enhanced for small datasets, was applied to manga faces [71] with improved FID scores. Yu et al. [233] introduced a multilingual text-to-image model for webtoon generation, showing the flexibility of GANs across languages. However, even fine-tuning on Korean MSCOCO resulted in abstract images that differed from authentic comics. Diffusion models have also made strides in story visualization. StoryDALL-E [128] used autoregressive Transformers for improved model tuning. Everaert et al. [49] adapted Stable Diffusion for comic generation by tweaking the initial latent tensor, showing how diffusion models can be tailored for comic styles. Despite limitations in text generation, diffusion models successfully generated comic panels aligned with COMIC data distributions. The consistency of characters and style across multiple images, especially in storytelling contexts, remains a key challenge [64]. Recent work by Jin et al. [86] leveraged large language models (LLMs) and finetuned Stable Diffusion to generate new manga content. Using ChatGPT to generate storylines and dialogues for a hypothetical continuation of the manga One Piece, they fine-tuned Stable Diffusion with LoRA and ControlNet to match the layout, style, and aesthetic of the original manga. The final product included LLM-generated dialogue integrated into comic panels. The subfield of story

visualization within text-to-image generation is rapidly evolving, driven by the integration of powerful LLMs and new image synthesis techniques. Key challenges include maintaining character and style consistency across multiple panels and developing unified benchmarks to track advancements across the field.

#### 10.2 Synthesis

The group of Synthesis tasks, compared to the “generation” one, refers to the creation of more complex, structured, and often temporally extended outputs, such as creating a full graphic novel from a complex storyline or merging multiple elements (characters, scenes, dialogues) in a consistent way across multiple pages (Fig. 15).

| |[Figure 224]|
|---|---|
| | |

3D Character Generation

[Figure 225]

###### Synthesis

[Figure 226]

Story 2 Video Generation

Character images

Model

|at home, read new paper #at home, The newspaper says there is a treasure house in the forest.[...]| |
|---|---|
| | |

[Figure 227]

[Figure 228]

Multi-page Story Generation

[Figure 229]

Full-Book Story description

- 10.2.1 3D generation from images. Definition: The task of generating 3D models from 2D comic illustrations presents unique challenges. Unlike conventional portrait illustrations, comics often involve stylized and exaggerated features, adding complexity to this already intricate process.

While there are advancements in using 3D character assets for tasks like pose estimation [93], re-targetting [92, 94], and character reposing [112], the field lacks scalable training resources due to the unavailability of suitable 3D character assets. In light of these issues, Chen et al. [24] addressed this gap by formalizing the stylized reconstruction task with the introduction of the AnimeRecon benchmark and the Vroid dataset. The Vroid dataset, in particular, provides a wealth of 3D assets for scalable training in this domain. A significant part of their work involves solving the challenge of contour removal from illustrations, which is a crucial step in achieving accurate 3D reconstruction from stylized comic images. While there are numerous works on 3D reconstruction from line drawings in general [123, 174], the specific application in comics is underexplored, likely due to the absence of suitable datasets and ground truth references for comics. It is worth noting that the exploration of 3D reconstruction from comic images holds great potential. It not only enriches the visual experience of comics but also opens up possibilities for animations, interactive media, and virtual reality applications within the comic domain.

- 10.2.2 Video generation. Definition: Video generation in comic refers to the synthesis of video content from static comic panels. This process represents a significant leap in merging traditional comic art with dynamic multimedia.

Fig. 15. Illustration of Synthesis tasks.

Recent advancements in video generation from both images [212, 223] and text [60] have made significant strides, though defining accurate metrics and benchmarks remains a challenge [50]. These developments are rapidly evolving within the broader Vision and Language community. However, in the domain of comics, video generation is still in its early stages. One of the first attempts came from Cao et al. [18], who introduced a framework to animate manga panels. Their method classifies panels based on motion and emotion, using this information to animate characters and backgrounds. The resulting video includes both intra-panel animations and transitions between panels. Later, Gupta et al. [61] expanded on this with C2VNet, which generates panel-audio videos by cropping panels and adding a highlighting effect to speech balloons to simulate character dialogue. They also introduced IMCDB, a dataset of annotated Indian Mythological Comics in English, to support research in this field. While these efforts represent initial steps, they are still far from the capabilities seen in Vision-Language video generation, such as producing fully animated video clips with sound from a single image. The integration of comics with video and audio elements

offers exciting potential for enhancing storytelling and audience engagement. Future research could pave the way for interactive and immersive comic experiences, blending traditional art with modern digital technologies.

- 10.2.3 Narrative-based Comic Generation. Definition: The task involves generating a comic (whether a panel, strip, or page) from a plain text description. The term “Narrative-based” highlights that the text not only describes the scene but also conveys the story’s narration.

The pioneering efforts in the realm of narrative-based comic generation commenced with the introduction of StoryGAN by Li et al. [107], which marked a significant advancement in generating image sequences from storylines using a sequential conditional GAN framework. This innovation incorporated a Context Encoder, which dynamically tracked the story flow, laying the groundwork for future developments in this area. Building on this foundation, Maharana et al. [127] enhanced visual quality and coherence by incorporating a dual learning framework that utilized video captioning, a copy-transform mechanism, and MART-based transformers. This approach significantly improved the semantic alignment between the story and generated images, thus enriching the story visualization process. In terms of character coherence and continuity, Chen et al. [22] focused on maintaining consistent character portrayal throughout the narrative. They adapted Vector-Quantized Variational Autoencoders (VQ-VAE) with a text-to-visual-token architecture, ensuring character coherence in the visual narratives. Furthering this direction, Maharana et al. [128] introduced the concept of story continuation, where the generated visual story is conditioned on a source image, allowing narratives to incorporate new characters more fluidly. They also introduced the DiDeMoSV dataset, which provided a platform for exploring these complex narrative structures. Recent advancements in story visualization are exemplified by Pan et al. [151], with presenting AR-LDM, a latent diffusion model that significantly raised the standards for visual quality in natural image datasets like VIST. This model was auto-regressively conditioned on history captions and generated images, capturing complex interactions between frames. Rahman et al. [160] introduced a framework with a visual memory module that implicitly captured actor and background context, generating frames that were not only of high visual quality but also consistent with the story. Peng et al. [153] proposed PCSG, a diffusion-based text-to-image synthesis framework with controllable plugins for character consistency, scene layout specification, and character pose specification, further enhancing the personalized aspects of story visualization. These emerging trends and future directions in narrative-based comic generation showcase a notable shift towards more sophisticated, context-aware, and visually coherent storytelling techniques, potentially leading to automated comic creation tools that adapt to various narrative styles and complexities.

#### 10.3 Future Tasks

10.3.1 Comics to Scene graph.

Definition: Transforming comics into scene graphs is an emerging area of research. This task involves dissecting comic panels to identify and represent the relationships between different elements within the panel in a graph format.

The last work that tackles this task is [170], which employs a Relationformer architecture [182], explicitly designed for image-to-graph tasks. However, the authors convert the full page into different types of nodes (panels, text, characters) and only consider two types of arches: characterscharacters (re-identification) and text-characters (speaker identification). The inherently complexity of the comics, such as the elements in the image (e.g. the background and foreground objects) have not been considered yet. The task has yet to be explored to fine-grained graph creation from single panels, full pages, or a book.

#### 11 CONCLUSION

In reviewing the diverse and innovative efforts in the realm of comic research and technology, it becomes evident that this field is in a state of dynamic and exciting evolution. The array of tasks and applications that researchers have embarked upon reflects not only the complexity of comics as a medium but also their immense potential as a bridge between artistic expression and technological advancement. We have covered numerous works and categorized them into distinct layers based on a proposed taxonomy. We have aimed to distill the key topics, methods, challenges, and emerging directions that can shape future research in Comics Understanding. In the future, we will continue to monitor advancements in this field, and we will update our findings in the following repository: https://github.com/emanuelevivoli/awesome-comics-understanding.

#### REFERENCES

- [1] Harsh Agrawal, A. Mishra, Manish Gupta, and M. . 2023. Multimodal Persona Based Generation of Comic Dialogs. Annual Meeting of the Association for Computational Linguistics (2023), 14150–14164.
- [2] Tasweer Ahmad and Maximilian Schich. 2023. Toward cross-domain object detection in artwork images using improved YoloV5 and XGBoosting. IET Image Process. 17 (2023), 2437–2449.
- [3] Seyed Ali Amirshahi, Michael Koch, Joachim Denzler, and Christoph Redies. 2012. PHOG analysis of self-similarity in esthetic images. Proc SPIE 8291 (2012), 46–.
- [4] Pei Soo Ang. 2021. book review: Who Understands Comics? Questioning the Universality of Visual Language Comprehension. Visual Communication 23 (2021), 388 – 390.
- [5] K. Arai and H. Tolle. 2010. Method for Automatic E-Comic Scene Frame Extraction for Reading Comic on Mobile Devices. 2010 Seventh International Conference on Information Technology: New Generations (2010), 370–375.
- [6] K. Arai and H. Tolle. 2011. Method for Real Time Text Extraction of Digital Manga Comic. The International Journal on the Image 4 (2011), 669–676.
- [7] Yuji Aramaki, Yusuke Matsui, T. Yamasaki, and K. Aizawa. 2016. Text detection in manga by combining connectedcomponent-based and region-based classifications. 2016 IEEE International Conference on Image Processing (ICIP)

(2016), 2901–2905.

- [8] Olivier Augereau, M. Iwata, and K. Kise. 2017. An Overview of Comics Research in Computer Science. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 54–59.
- [9] Olivier Augereau, M. Iwata, and K. Kise. 2018. A survey of comics research in computer science. J. Imaging 4 (2018), 87.
- [10] Jeonghun Baek, Yusuke Matsui, and K. Aizawa. 2022. COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts. European Conference on Computer Vision (2022), 267–283.
- [11] T. Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2017. Multimodal Machine Learning: A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence 41 (2017), 423–443.
- [12] Deblina Bhattacharjee and Bahar Aydemir. 2024. ECCV workshop: AI for Visual art. https://github.com/IVRL/AI4VA.
- [13] Deblina Bhattacharjee, Martin Nicolas Everaert, M. Salzmann, and S. Süsstrunk. 2021. Estimating Image Depth in the Comics Domain. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2021), 1111–1120.
- [14] Deblina Bhattacharjee, S. Süsstrunk, and M. Salzmann. 2023. Dense Multitask Learning to Reconfigure Comics. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2023), 5646–5655.
- [15] Deblina Bhattacharjee, Tong Zhang, Sabine Süsstrunk, and Mathieu Salzmann. 2022. MulT: An End-to-End Multitask Learning Transformer. ArXiv abs/2205.08303 (2022).
- [16] Ali Furkan Biten, Rubèn Pérez Tito, Andrés Mafla, Lluís Gómez, Marçal Rusiñol, Ernest Valveny, C. V. Jawahar, and Dimosthenis Karatzas. 2019. Scene Text Visual Question Answering. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019), 4290–4300.
- [17] Ying Cao, Antoni B. Chan, and Rynson W. H. Lau. 2012. Automatic stylistic manga layout. ACM Transactions on Graphics (TOG) 31 (2012), 1 – 10.
- [18] Ying Cao, X. Pang, Antoni B. Chan, and Rynson W. H. Lau. 2017. Dynamic Manga: Animating Still Manga via Camera Movement. IEEE Transactions on Multimedia 19 (2017), 160–172.
- [19] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-End Object Detection with Transformers. ArXiv abs/2005.12872 (2020).
- [20] Francisco Castro, Sangho Suh, J. E, Weena Naowaprateep, and Yang Shi. 2022. Developing Comic-based Learning Toolkits for Teaching Computing to Elementary School Learners.

- [21] I. Chang and Ruei-Min Cheng. 2011. Caricaturation for human face pictures. 2011 International Conference on Machine Learning and Cybernetics 4 (2011), 1702–1707.
- [22] Hong Chen, Rujun Han, Te-Lin Wu, Hideki Nakayama, and Nanyun Peng. 2022. Character-centric Story Visualization via Visual Planning and Token Alignment. Conference on Empirical Methods in Natural Language Processing (2022), 8259–8272.
- [23] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. 2023. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. ArXiv abs/2310.09478 (2023).
- [24] Shuhong Chen, Kevin Zhang, Yichun Shi, Heng Wang, Yiheng Zhu, Guoxian Song, Sizhe An, Janus Kristjansson, X. Yang, and Matthias Zwicker. 2023. PAniC-3D: Stylized Single-view 3D Reconstruction from Portraits of Anime Characters. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 21068–21077.
- [25] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. 2020. A Simple Framework for Contrastive Learning of Visual Representations. ArXiv abs/2002.05709 (2020).
- [26] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. 2021. Unifying Vision-and-Language Tasks via Text Generation. International Conference on Machine Learning (2021), 1931–1942.
- [27] Yun-Feng Chou and Zen-Chung Shih. 2011. Comic character animation using Bayesian estimation. Computer Animation and Virtual Worlds 22 (2011).
- [28] W. Chu and Ying-Chieh Chao. 2014. Line-Based Drawing Style Description for Manga Classification.
- [29] W. Chu and Chih-Chi Yu. 2018. Text Detection in Manga by Deep Region Proposal, Classification, and Regression. 2018 IEEE Visual Communications and Image Processing (VCIP) (2018), 1–4.
- [30] Neil Cohn. 2011. A different kind of cultural frame: An analysis of panels in American comics and Japanese manga. Image and narrative 12 (2011), 120–134.
- [31] Neil Cohn. 2013. Beyond speech balloons and thought bubbles: The integration of text and image, Vol. 2013. 35 – 63.
- [32] Niel Cohn. 2013. Visual Narrative Structure. Cognitive Science 37, 3 (2013), 413–452.
- [33] Neil Cohn. 2014. The architecture of visual narrative comprehension: the interaction of narrative structure and page layout in understanding comics. Frontiers in Psychology 5 (2014).
- [34] Neil Cohn. 2016. A multimodal parallel architecture: A cognitive framework for multimodal interactions. Cognition 146 (2016), 304–323.
- [35] Neil Cohn. 2023. The Visual Language Research Corpus (VLRC) Project. (2023). https://dataverse.nl/dataset.xhtml? persistentId=doi:10.34894/LWMZ7G
- [36] Neil Cohn and Sean Ehly. 2016. The vocabulary of manga: Visual morphology in dialects of Japanese Visual Language. Journal of Pragmatics 92 (2016), 17–29.
- [37] Neil Cohn, Ryan Taylor, and Kaitlin Pederson. 2017. A Picture is Worth More Words Over Time: Multimodality and Narrative Structure Across Eight Decades of American Superhero Comics. Multimodal Communication 6 (2017), 19 – 37.
- [38] Neil Cohn, David Wagner, T. Foulsham, and J. E. Drury. 2014. The cognition of comics: What "comics" can tell us about the mind. Cognitive Science 36 (2014).
- [39] Yaonan Dai, Jiuyang Yu, Tianhao Hu, Yang Lu, and Xiaotao Zheng. 2022. Structured Fusion Attention Network for Image Super-Resolution Reconstruction. IEEE Access 10 (2022), 31896–31906.
- [40] Yuki Daiku, Olivier Augereau, M. Iwata, and K. Kise. 2017. Comic Story Analysis Based on Genre Classification. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 60–65.
- [41] Mehmet Arif Demirtacs, Berke Oral, Mehmet Yasin Akpinar, and Onur Deniz. 2022. Semantic Parsing of Interpage Relations. 2022 26th International Conference on Pattern Recognition (ICPR) (2022), 1579–1585.
- [42] Jinhong Deng, Wen Li, Yuhua Chen, and Lixin Duan. 2020. Unbiased Mean Teacher for Cross-domain Object Detection. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020), 4089–4099.
- [43] Xuexiong Deng, Mu Li, and Yueling Zhang. 2010. Research and development of the generation in Japanese manga based on frontal face image, In 2010 IEEE 11th International Conference on Computer-Aided Industrial Design & Conceptual Design 1. 2010 IEEE 11th International Conference on Computer-Aided Industrial Design & Conceptual Design 1 1 (2010), 729–732.
- [44] David Dubray and Jochen Laubrock. 2019. Deep CNN-Based Speech Balloon Detection and Segmentation for Comic books. 2019 International Conference on Document Analysis and Recognition (ICDAR) (2019), 1237–1243.
- [45] Alexander Dunst, Rita Hartel, and Jochen Laubrock. 2017. The Graphic Narrative Corpus (GNC): Design, Annotation, and Analysis for the Digital Humanities. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 15–20.
- [46] Arpita Dutta and Samit Biswas. 2019. CNN Based Extraction of Panels/Characters from Bengali Comic book Page Images. 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW) 1 (2019), 38–43.

- [47] Arpita Dutta, Samit Biswas, and A. Das. 2021. CNN-based segmentation of speech balloons and narrative text boxes from comic book page images. International Journal on Document Analysis and Recognition (IJDAR) 24 (2021), 49 – 62.
- [48] Arpita Dutta, Samit Biswas, and A. Das. 2022. BCBId: first Bangla comic dataset and its applications. International Journal on Document Analysis and Recognition (IJDAR) 25 (2022), 265 – 279.
- [49] Martin Nicolas Everaert, Marco Bocchio, Sami Arpa, Sabine Süsstrunk, and Radhakrishna Achanta. 2023. Diffusion in Style. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (2023), 2251–2261.
- [50] Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. 2024. AIGCBench: Comprehensive Evaluation of Image-toVideo Content Generated by AI. ArXiv abs/2401.01651 (2024).
- [51] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. 2023. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. ArXiv abs/2306.13394 (2023).
- [52] Azuma Fujimoto, Toru Ogawa, Kazuyoshi Yamamoto, Yusuke Matsui, T. Yamasaki, and K. Aizawa. 2016. Manga109 dataset and creation of metadata.
- [53] Chie Furusawa, Kazuyuki Hiroshiba, Keisuke Ogaki, and Yuri Odagiri. 2017. Comicolorization: semi-automatic manga colorization.
- [54] Rinon Gal, Yuval Alaluf, Y. Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and D. Cohen-Or. 2022. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. ArXiv abs/2208.01618 (2022).
- [55] Arpan Garai, Arpita Dutta, and Samit Biswas. 2022. Automatic dewarping of camera-captured comic document images. Multimedia Tools and Applications 82 (2022), 1537–1552.
- [56] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. ImageBind One Embedding Space to Bind Them All. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 15180–15190.
- [57] Maksim Golyadkin and Ilya Makarov. 2023. Robust Manga Page Colorization via Coloring Latent Space. IEEE Access 11 (2023), 111581–111597.
- [58] Albert Gordo, Marçal Rusiñol, Dimosthenis Karatzas, and Andrew D. Bagdanov. 2013. Document Classification and Page Stream Segmentation for Digital Mailroom Applications. 2013 12th International Conference on Document Analysis and Recognition (2013), 621–625.
- [59] Hongcheng Guo, Boyang Wang, Jiaqi Bai, Jiaheng Liu, Jian Yang, and Zhoujun Li. 2023. M2C: Towards Automatic Multimodal Manga Complement. Conference on Empirical Methods in Natural Language Processing (2023), 9876–9882.
- [60] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. 2023. Photorealistic Video Generation with Diffusion Models. European Conference on Computer Vision (2023), 393–411.
- [61] Vaibhav Gupta, Vinay Detani, Vivek Khokar, and Chiranjoy Chattopadhyay. 2021. C2VNet: A Deep Learning Framework Towards Comic Strip to Audio-Visual Scene Synthesis. IEEE International Conference on Document Analysis and Recognition (2021), 160–175.
- [62] C. Guérin, Christophe Rigaud, Antoine Mercier, Farid Ammar-Boudjelal, K. Bertet, Alain Bouju, J. Burie, Georges Louis, J. Ogier, and A. Revel. 2013. eBDtheque: A Representative Database of Comics. 2013 12th International Conference on Document Analysis and Recognition (2013), 1145–1149.
- [63] David R Ha and D. Eck. 2017. A Neural Representation of Sketch Drawings. ArXiv abs/1704.03477 (2017).
- [64] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. 2024. DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion. ArXiv abs/2407.12899 (2024).
- [65] Zheqi He, Yafeng Zhou, Yongtao Wang, and Zhi Tang. 2017. SReN: Shape Regression Network for Comic Storyboard Extraction. AAAI Conference on Artificial Intelligence (2017), 4937–4938.
- [66] Zheqi He, Yafeng Zhou, Yongtao Wang, Siwei Wang, Xiaoqing Lu, Zhi Tang, and Lingyi Cai. 2018. An End-to-End Quadrilateral Regression Network for Comic Panel Extraction. Proceedings of the 26th ACM international conference on Multimedia (2018).
- [67] Paulina Hensman and K. Aizawa. 2017. cGAN-Based Manga Colorization Using a Single Training Image. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 72–77.
- [68] Luis Herranz, J. Calic, J. Sanchez, and M. Mrak. 2012. Scalable Comic-Like Video Summaries and Layout Disturbance. IEEE Transactions on Multimedia 14 (2012), 1290–1297.
- [69] Ryota Hinami, Shonosuke Ishiwatari, Kazuhiko Yasuda, and Yusuke Matsui. 2020. Towards Fully Automated Manga Translation. ArXiv abs/2012.14271 (2020). https://api.semanticscholar.org/CorpusID:229679890
- [70] Sotaro Hiroe and S. Hotta. 2017. Histogram of Exclamation Marks and Its Application for Comics Analysis. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 66–71.
- [71] Komei Hiruta, Ryusuke Saito, Taro Hatakeyama, Atsushi Hashimoto, and Satoshi Kurihara. 2022. Conditional GAN for Small Datasets. 2022 IEEE International Symposium on Multimedia (ISM) (2022), 278–281.

- [72] Anh Khoi Ngo Ho, J. Burie, and J. Ogier. 2012. Panel and Speech Balloon Extraction from Comic books. 2012 10th IAPR International Workshop on Document Analysis Systems (2012), 424–428.
- [73] Hoang Nam Ho, Christophe Rigaud, J. Burie, and J. Ogier. 2013. Redundant structure detection in attributed adjacency graphs for character detection in comics books, In IAPR International Workshop on Graphics Recognition. IAPR International Workshop on Graphics Recognition.
- [74] K. Hoashi, C. Ono, Daisuke Ishii, and Hiroshi Watanabe. 2011. Automatic preview generation of comic episodes for digitized comic search. Proceedings of the 19th ACM international conference on Multimedia (2011).
- [75] Ronghang Hu, Amanpreet Singh, Trevor Darrell, and Marcus Rohrbach. 2019. Iterative Answer Prediction With Pointer-Augmented Multimodal Transformers for TextVQA. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019), 9989–9999.
- [76] Zhe Hu, Tuo Liang, Jing Li, Yiren Lu, Yunlai Zhou, Yiran Qiao, Jing Ma, and Yu Yin. 2024. Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions. ArXiv abs/2405.19088 (2024). https: //api.semanticscholar.org/CorpusID:270095233
- [77] Hikaru Ikuta, Leslie Wöhler, and Kiyoharu Aizawa. 2024. MangaUB: A Manga Understanding Benchmark for Large Multimodal Models. ArXiv abs/2407.19034 (2024). https://api.semanticscholar.org/CorpusID:271534288
- [78] Naoto Inoue, Ryosuke Furuta, T. Yamasaki, and K. Aizawa. 2018. Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition

(2018), 5001–5009.

- [79] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. 2016. Image-to-Image Translation with Conditional Adversarial Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), 5967–5976.
- [80] M. Iwata, Eiki Imazu, and K. Kise. 2015. Similarity learning based on pool-based active learning for manga character retrieval. 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR) (2015), 437–442.
- [81] M. Iwata, Atsushi Ito, and K. Kise. 2014. A Study to Achieve Manga Character Retrieval Method for Manga Images. 2014 11th IAPR International Workshop on Document Analysis Systems (2014), 309–313.
- [82] Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan L. Boyd-Graber, Hal Daumé, and L. Davis.

2016. The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic book Narratives. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), 6478–6487.

- [83] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. International Conference on Machine Learning (2021), 4904–4916.
- [84] Shuhui Jiang, Ming Shao, Chengcheng Jia, and Y. Fu. 2018. Learning Consensus Representation for Weak Style Classification. IEEE Transactions on Pattern Analysis and Machine Intelligence 40 (2018), 2906–2919.
- [85] Yanghua Jin, Jiakai Zhang, Minjun Li, Yingtao Tian, Huachun Zhu, and Zhihao Fang. 2017. Towards the Automatic Anime Characters Creation with Generative Adversarial Networks. ArXiv abs/1708.05509 (2017).
- [86] Ze Jin and Zorina Song. 2023. Generating coherent comic with rich story using ChatGPT and Stable Diffusion. ArXiv abs/2305.11067 (2023).
- [87] Guangmei Jing, Yongtao Hu, Yanwen Guo, Yizhou Yu, and Wenping Wang. 2015. Content-Aware Video2Comics With Manga-Style Layout. IEEE Transactions on Multimedia 17 (2015), 2122–2133.
- [88] Tawin Jiramahapokee. 2023. inkn’hue: Enhancing Manga Colorization from Multiple Priors with Alignment MultiEncoder VAE. ArXiv abs/2311.01804 (2023).
- [89] H. Jomaa, M. Awad, and Lina Ghaibeh. 2015. Panel Tracking for the Extraction and the Classification of Speech Balloons. International Conference on Image Analysis and Processing (2015), 394–405.
- [90] F. Khan, R. Anwer, Joost van de Weijer, Andrew D. Bagdanov, M. Vanrell, and Antonio M. López. 2012. Color attributes for object detection. 2012 IEEE Conference on Computer Vision and Pattern Recognition (2012), 3306–3313.
- [91] Simran Khanuja, Sathyanarayanan Ramamoorthy, Yueqi Song, and Graham Neubig. 2024. An image speaks a thousand words, but can everyone listen? On image transcreation for cultural relevance. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Miami, Florida, USA, 10258–10279. https://aclanthology.org/2024.emnlp-main.573
- [92] Pramook Khungurn. 2022. Pkhungurn/Talking-Head-Anime-2-Demo. https://github.com/pkhungurn/talking-headanime-2-demo
- [93] Junho Kim, Minjae Kim, Hyeonwoo Kang, and Kwanghee Lee. 2019. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. ArXiv abs/1907.10830 (2019).
- [94] Kangyeol Kim, S. Park, Jaeseong Lee, Sunghyo Chung, Junsoo Lee, and J. Choo. 2021. AnimeCeleb: Large-Scale Animation CelebHeads Dataset for Head Reenactment. European Conference on Computer Vision (2021), 414–430.
- [95] Bien Klomberg, Irmak Hacımusaoğlu, and Neil Cohn. 2022. Running through the Who, Where, and When: A Cross-cultural Analysis of Situational Changes in Comics. Discourse Processes 59 (2022), 669 – 684.
- [96] J. Kopf and Dani Lischinski. 2011. Depixelizing pixel art. ACM SIGGRAPH 2011 papers (2011).

- [97] Bruno Korbar and Andrew Zisserman. 2022. Personalised CLIP or: how to find your vacation videos. British Machine Vision Conference (2022), 639.
- [98] Eleanna Kouletou, Vassilis Papavassiliou, and Vassilis Katsouros. 2024. Investigating Neural Networks and Transformer Models for Enhanced Comic Decoding, In Document Analysis and Recognition – ICDAR 2024 Workshops: Athens, Greece, August 30–31, 2024, Proceedings, Part I (Athens, Greece). Document Analysis and Recognition – ICDAR 2024 Workshops: Athens, Greece, August 30–31, 2024, Proceedings, Part I, 138–153. https://doi.org/10.1007/978-3-031-706455_10
- [99] Hugo Laurençon, Andr’es Marafioti, Victor Sanh, and Léo Tronchon. 2024. Building and better understanding vision-language models: insights and future directions. ArXiv abs/2408.12637 (2024).
- [100] Thanh Nam Le, M. Luqman, Anjan Dutta, P. Héroux, Christophe Rigaud, C. Guérin, P. Foggia, J. Burie, J. Ogier, J. Lladós, and Sébastien Adam. 2018. Subgraph spotting in graph representations of comic book images. Pattern Recognit. Lett. 112 (2018), 118–124.
- [101] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li.

2024. LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild. https://llava-vl.github.io/blog/ 2024-05-10-llava-next-stronger-llms/

- [102] Chengze Li, Xueting Liu, and T. Wong. 2017. Deep extraction of manga structural lines. ACM Transactions on Graphics (TOG) 36 (2017).
- [103] Feng Li, Hao Zhang, Yi-Fan Zhang, S. Liu, Jian Guo, L. Ni, Pengchuan Zhang, and Lei Zhang. 2022. Vision-Language Intelligence: Tasks, Representation Learning, and Large Models. ArXiv abs/2203.01922 (2022).
- [104] Junnan Li, Dongxu Li, S. Savarese, and Steven C. H. Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. International Conference on Machine Learning (2023), 19730–19742.
- [105] Yingxuan Li, K. Aizawa, and Yusuke Matsui. 2023. Manga109Dialog: A Large-Scale Dialogue Dataset for Comics Speaker Detection. 2024 IEEE International Conference on Multimedia and Expo (ICME) (2023), 1–6.
- [106] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, L. Carin, David Edwin Carlson, and Jianfeng Gao. 2018. StoryGAN: A Sequential Conditional GAN for Story Visualization. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018), 6322–6331.
- [107] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, L. Carin, David Edwin Carlson, and Jianfeng Gao. 2018. StoryGAN: A Sequential Conditional GAN for Story Visualization. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018), 6322–6331.
- [108] Yingxuan Li, Ryota Hinami, Kiyoharu Aizawa, and Yusuke Matsui. 2024. Zero-Shot Character Identification and Speaker Prediction in Comics via Iterative Multimodal Fusion, In ACM Multimedia. ACM Multimedia. https: //api.semanticscholar.org/CorpusID:269294071
- [109] Yonggang Li, Yafeng Zhou, Yongtao Wang, Xiaoran Qin, and Zhi Tang. 2021. Dual Loss for Manga Character Recognition with Imbalanced Training Data. 2020 25th International Conference on Pattern Recognition (ICPR) (2021), 2166–2171.
- [110] Zhihui Li, Lina Yao, Xiaoqin Zhang, Xianzhi Wang, S. Kanhere, and Huaxiang Zhang. 2019. Zero-Shot Object Detection with Textual Descriptions. AAAI Conference on Artificial Intelligence (2019), 8690–8697.
- [111] Antonio Alexandre Lima, Marcello Montillo Provenza, and M. A. Nunes. 2022. Comics as a Pedagogical Tool for Teaching, In 2022 XVII Latin American Conference on Learning Technologies (LACLO). 2022 XVII Latin American Conference on Learning Technologies (LACLO) (2022), 1–7.
- [112] Zuzeng Lin, Ailin Huang, Zhewei Huang, Chen Hu, and Shuchang Zhou. 2022. Collaborative Neural Rendering using Anime Character Sheets. International Joint Conference on Artificial Intelligence (2022), 5824–5832.
- [113] Philip Lippmann, Konrad Skublicki, Joshua B. Tanner, Shonosuke Ishiwatari, and Jie Yang. 2024. Context-Informed Machine Translation of Manga using Multimodal Large Language Models. https://api.semanticscholar.org/CorpusID: 273821926
- [114] Bingchen Liu, Yizhe Zhu, Kunpeng Song, and A. Elgammal. 2021. Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis. ArXiv abs/2101.04775 (2021).
- [115] Danyang Liu, Mirella Lapata, and Frank Keller. 2024. Generating Visual Stories with Grounded and Coreferent Characters. ArXiv abs/2409.13555 (2024). https://api.semanticscholar.org/CorpusID:272770579
- [116] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. Improved Baselines with Visual Instruction Tuning. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 26286–26296.
- [117] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. ArXiv abs/2304.08485

(2023).

- [118] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chun-yue Li, Jianwei Yang, Hang Su, Jun-Juan Zhu, and Lei Zhang. 2023. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ArXiv abs/2303.05499 (2023).

- [119] W. Liu, Dragomir Anguelov, D. Erhan, Christian Szegedy, Scott E. Reed, Cheng-Yang Fu, and A. Berg. 2015. SSD: Single Shot MultiBox Detector. European Conference on Computer Vision (2015), 21–37.
- [120] Xueting Liu, Chengze Li, Haichao Zhu, T. Wong, and Xuemiao Xu. 2015. Text-aware balloon extraction from manga. The Visual Computer 32 (2015), 501 – 511.
- [121] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. ArXiv abs/1907.11692

(2019).

- [122] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and B. Guo. 2021. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021), 9992–10002.
- [123] Z. Lun, Matheus Gadelha, E. Kalogerakis, Subhransu Maji, and Rui Wang. 2017. 3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks. 2017 International Conference on 3D Vision (3DV) (2017), 67–77.
- [124] Cong Ma, Yaping Zhang, Mei Tu, Yang Zhao, Yu Zhou, and Chengqing Zong. 2023. Multi-Teacher Knowledge Distillation For Text Image Machine Translation. ArXiv abs/2305.05226 (2023). https://api.semanticscholar.org/ CorpusID:258564898
- [125] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Y. Fu, and Humphrey Shi. 2022. Towards Layer-wise Image Vectorization. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), 16293–16302.
- [126] A. Maharana and Mohit Bansal. 2021. Integrating Visuospatial, Linguistic, and Commonsense Structure into Story Visualization. ArXiv abs/2110.10834 (2021).
- [127] A. Maharana, Darryl Hannan, and Mohit Bansal. 2021. Improving Generation and Evaluation of Visual Stories via Semantic Consistency. ArXiv abs/2105.10026 (2021).
- [128] A. Maharana, Darryl Hannan, and Mohit Bansal. 2022. StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation. ArXiv abs/2209.06192 (2022).
- [129] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. 2022. InfographicVQA, In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2582–2591.
- [130] Misa Matsuda, I. Tanev, and K. Shimohara. 2010. Comic Live Chat communication tool based on concept of downgrading, In Proceedings of SICE Annual Conference 2010. Proceedings of SICE Annual Conference 2010 (2010), 2775–2778.
- [131] Yusuke Matsui, K. Aizawa, and Yushi Jing. 2014. Sketch2Manga: Sketch-based manga retrieval. 2014 IEEE International Conference on Image Processing (ICIP) (2014), 3097–3101.
- [132] Yusuke Matsui, Kota Ito, Yuji Aramaki, Azuma Fujimoto, Toru Ogawa, T. Yamasaki, and K. Aizawa. 2015. Sketch-based manga retrieval using manga109 dataset. Multimedia Tools and Applications 76 (2015), 21811 – 21838.
- [133] Thomas Melistas, Giannis Siglidis, Fivos Kalogiannis, and Ilan Manouach. 2021. A Deep Learning Pipeline for the Synthesis of Graphic Novels. International Conference on Innovative Computing and Cloud Computing (2021), 256–265.
- [134] Yukihiro Moriyama, Byeongseon Park, Shinnosuke Iwaoki, and Mitsunori Matsushita. 2016. Designing a questionanswering system for comic contents.
- [135] Malik Nairat. 2021. Generative Comics - A Computational Approach to Creating Comics Material. https://gupea.ub.gu. se/handle/2077/69634
- [136] Rei Narita, Koki Tsubota, T. Yamasaki, and K. Aizawa. 2017. Sketch-Based Manga Retrieval Using Deep Features. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 49–53.
- [137] Nhu-Van Nguyen, Christophe Rigaud, and J. Burie. 2017. Comic Characters Detection Using Deep Learning. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 41–46.
- [138] Nhu-Van Nguyen, Christophe Rigaud, and J. Burie. 2018. Digital Comics Image Indexing Based on Deep Learning. J. Imaging 4 (2018), 89.
- [139] Nhu-Van Nguyen, Christophe Rigaud, and J. Burie. 2019. Comic MTL: optimized multi-task learning for comic book image analysis. International Journal on Document Analysis and Recognition (IJDAR) 22 (2019), 265 – 284.
- [140] Nhu-Van Nguyen, Christophe Rigaud, and J. Burie. 2019. What do We Expect from Comic Panel Extraction? 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW) 1 (2019), 44–49.
- [141] Nhu-Van Nguyen, Christophe Rigaud, A. Revel, and J. Burie. 2020. A learning approach with incomplete pixel-level labels for deep neural networks. Neural networks : the official journal of the International Neural Network Society 130

(2020), 111–125.

- [142] Nhu-Van Nguyen, Christophe Rigaud, A. Revel, and J. Burie. 2021. Manga-MMTL: Multimodal Multitask Transfer Learning for Manga Character Analysis. IEEE International Conference on Document Analysis and Recognition (2021), 410–425.
- [143] Nhu-Van Nguyen, Xuan-Son Vu, Christophe Rigaud, Lili Jiang, and J. Burie. 2021. ICDAR 2021 Competition on Multimodal Emotion Recognition on Comics Scenes. IEEE International Conference on Document Analysis and

- Recognition (2021), 767–782.
- [144] O. Nir, Gal Rapoport, and Ariel Shamir. 2022. CAST: Character labeling in Animation using Self-supervision by Tracking. Computer Graphics Forum 41 (2022).
- [145] Toru Ogawa, Atsushi Otsubo, Rei Narita, Yusuke Matsui, T. Yamasaki, and K. Aizawa. 2018. Object Detection for Comics using Manga109 Annotations. ArXiv abs/1803.08670 (2018).
- [146] A. Oliva and A. Torralba. 2001. Modeling the Shape of the Scene: A Holistic Representation of the Spatial Envelope. International Journal of Computer Vision 42 (2001), 145–175.
- [147] Yuga Omori, Kota Nagamizo, and Daisuke Ikeda. 2022. Algorithms for estimation of comic speakers considering reading order of frames and texts. 2022 12th International Congress on Advanced Applied Informatics (IIAI-AAI) (2022), 367–372.
- [148] OpenAI. 2023. GPT-4 Technical Report.
- [149] M. Oquab, Timothée Darcet, Théo Moutakanni, Huy Q. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Huijiao Xu, H. Jégou, J. Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision. ArXiv abs/2304.07193 (2023).
- [150] S. Pal, J. Burie, U. Pal, and J. Ogier. 2016. Line-wise text identification in comic books: A support vector machine-based approach. 2016 International Joint Conference on Neural Networks (IJCNN) (2016), 3995–4000.
- [151] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. 2022. Synthesizing Coherent Story with AutoRegressive Latent Diffusion Models. 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

(2022), 2908–2918.

- [152] Letitia Parcalabescu and A. Frank. 2020. Exploring Phrase Grounding without Training: Contextualisation and Extension to Text-Based Image Retrieval. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2020), 4137–4146.
- [153] Wenxuan Peng, Peter Schaldenbrand, and Jean Oh. 2023. Personalized Comic Story Generation.
- [154] Maciej Pesko, Adam Svystun, Pawel Andruszkiewicz, Przemyslaw Rokita, and T. Trzciński. 2018. Comixify: Transform video into a comics. ArXiv abs/1812.03473 (2018).
- [155] C. Ponsard. 2009. Enhancing the Accessibility for All of Digital Comic books. e Minds Int. J. Hum. Comput. Interact. 1

(2009).

- [156] Ben Proven-Bessel, Zilong Zhao, and L. Chen. 2021. ComicGAN: Text-to-Comic Generative Adversarial Network. ArXiv abs/2109.09120 (2021).
- [157] Xiaoran Qin, Yafeng Zhou, Zheqi He, Yongtao Wang, and Zhi Tang. 2017. A Faster R-CNN Based Method for Comic Characters Face Detection. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 01

(2017), 1074–1080.

- [158] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and I. Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. International Conference on Machine Learning (2021), 8748–8763.
- [159] Alec Radford, Luke Metz, and Soumith Chintala. 2015. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. CoRR abs/1511.06434 (2015).
- [160] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, S. Tulyakov, Shweta Mahajan, and L. Sigal. 2022. Make-A-Story: Visual Memory Conditioned Consistent Story Generation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), 2493–2502.
- [161] Reshma Ramaprasad. 2023. Comics for Everyone: Generating Accessible Text Descriptions for Comic Strips. ArXiv abs/2310.00698 (2023).
- [162] Joseph Redmon, S. Divvala, Ross B. Girshick, and Ali Farhadi. 2015. You Only Look Once: Unified, Real-Time Object Detection. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015), 779–788.
- [163] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 39 (2015), 1137–1149.
- [164] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and D. Cohen-Or. 2020. Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020), 2287–2296.
- [165] Christophe Rigaud. 2014. Segmentation and indexation of complex objects in comic book images. (Segmentation et indexation d’objets complexes dans les images de bandes dessinées). Electronic Letters on Computer Vision and Image Analysis 14 (2014), 59–60.
- [166] Christophe Rigaud, J. Burie, J. Ogier, Dimosthenis Karatzas, and Joost van de Weijer. 2013. An Active Contour Model for Speech Balloon Detection in Comics. 2013 12th International Conference on Document Analysis and Recognition

(2013), 1240–1244.

- [167] Christophe Rigaud, Jean-Christophe Burie, and Samuel Petit. 2024. Toward accessible comics for blind and low vision readers. ArXiv abs/2407.08248 (2024). https://api.semanticscholar.org/CorpusID:271097563
- [168] Christophe Rigaud, N. L. Thanh, J. Burie, J. Ogier, M. Iwata, Eiki Imazu, and K. Kise. 2015. Speech balloon and speaker association for comics and manga understanding. 2015 13th International Conference on Document Analysis and Recognition (ICDAR) (2015), 351–355.
- [169] Ragav Sachdeva, Gyungin Shin, and Andrew Zisserman. 2024. Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names. ArXiv abs/2408.00298 (2024).
- [170] Ragav Sachdeva and Andrew Zisserman. 2024. The Manga Whisperer: Automatically Generating Transcriptions for Comics. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), 12967–12976.
- [171] Pritish Sahu, Karan Sikka, and Ajay Divakaran. 2021. Challenges in Procedural Multimodal Machine Comprehension: A Novel Way To Benchmark. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2021), 526–535.
- [172] Pritish Sahu, Karan Sikka, and Ajay Divakaran. 2021. Towards Solving Multimodal Comprehension. ArXiv abs/2104.10139 (2021).
- [173] Charles Lima Sanches, Olivier Augereau, and K. Kise. 2016. Manga content analysis using physiological signals. Proceedings of the 1st International Workshop on coMics ANalysis, Processing and Understanding (2016).
- [174] Aditya Sanghi, P. Jayaraman, Arianna Rampini, J. Lambourne, Hooman Shayani, Evan Atherton, and Saeid Asgari Taghanaki. 2023. Sketch-A-Shape: Zero-Shot Sketch-to-3D Shape Generation. ArXiv abs/2307.03869 (2023).
- [175] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv abs/1910.01108 (2019).
- [176] I. Santos, Luz Castro, Nereida Rodriguez Fernandez, Alvaro Torrente Patino, and A. Carballal. 2021. Artificial Neural Networks and Deep Learning in the Visual Arts: a review. Neural Computing and Applications 33 (2021), 121 – 157.
- [177] Byu Scholarsarchive, A. Manning, and Scott McCloud. 2018. Understanding Comics: The Invisible Art.
- [178] P. Selinger. 2003. Potrace : a polygon-based tracing algorithm.
- [179] Rishabh Sharma and Vinay Kukreja. 2023. CPD: Faster RCNN-based DragonBall Comic Panel Detection, In International Conference on Communication Systems and Network Technologies. 2023 IEEE 12th International Conference on Communication Systems and Network Technologies (CSNT) (2023), 786–790.
- [180] Conghao Shen, Violet Z. Yao, and Yixin Liu. 2023. MaRU: A Manga Retrieval and Understanding System Connecting Vision and Language. ArXiv abs/2311.02083 (2023).
- [181] Haozhan Shen, Tiancheng Zhao, Mingwei Zhu, and Jianwei Yin. 2023. GroundVLP: Harnessing Zero-shot Visual Grounding from Vision-Language Pre-training and Open-Vocabulary Object Detection. ArXiv abs/2312.15043 (2023).
- [182] Suprosanna Shit, Rajat Koner, Bastian Wittmann, J. Paetzold, I. Ezhov, Hongwei Li, Jia-Yu Pan, Sahand Sharifzadeh, Georgios Kaissis, Volker Tresp, and Bjoern H Menze. 2022. Relationformer: A Unified Framework for Image-to-Graph Generation. European Conference on Computer Vision (2022), 422–439.
- [183] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. 2021. FLAVA: A Foundational Language And Vision Alignment Model. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), 15617–15629.
- [184] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach.

2019. Towards VQA Models That Can Read. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019), 8309–8318.

- [185] Sandeep Singh, Kang Dept, of, Geet kiran Kaur, and Sanjay Singla. 2024. The Future of Graphic Novel Translation: Fully Automated Systems. 2024 International Conference on Knowledge Engineering and Communication Systems (ICKECS) 1 (2024), 1–8. https://api.semanticscholar.org/CorpusID:271747919
- [186] Gürkan Soykan, Deniz Yuret, and T. M. Sezgin. 2022. A Comprehensive Gold Standard and Benchmark for Comics Text Detection and Recognition. ArXiv abs/2212.14674 (2022).
- [187] Gurkan Soykan, Deniz Yuret, and T. M. Sezgin. 2023. Identity-Aware Semi-Supervised Learning for Comic Character Re-Identification. ArXiv abs/2308.09096 (2023).
- [188] Gurkan Soykan, Deniz Yuret, and T. Metin Sezgin. 2024. ComicBERT: A Transformer Model and Pre-training Strategy for Contextual Understanding in Comics, In IEEE International Conference on Document Analysis and Recognition. IEEE International Conference on Document Analysis and Recognition. https://api.semanticscholar.org/CorpusID: 272694691
- [189] Gurkan Soykan, Deniz Yuret, and T. Metin Sezgin. 2024. Spatially Augmented Speech Bubble to Character Association via Comic Multi-task Learning, In IEEE International Conference on Document Analysis and Recognition. IEEE International Conference on Document Analysis and Recognition. https://api.semanticscholar.org/CorpusID:272694702
- [190] Chung-Yuan Su, Ray-I Chang, and Jen-Chang Liu. 2011. Recognizing Text Elements for SVG Comic Compression and Its Novel Applications. 2011 International Conference on Document Analysis and Recognition (2011), 1329–1333.

- [191] H. Su, Xuefeng Liu, Jianwei Niu, Jiahe Cui, Ji Wan, Xinghao Wu, and Nana Wang. 2024. MARVEL: Raster Gray-Level Manga Vectorization via Primitive-Wise Deep Reinforcement Learning. IEEE Transactions on Circuits and Systems for Video Technology 34 (2024), 2677–2693.
- [192] Sangho Suh, Celine Latulipe, Ken Jen Lee, Bernadette Cheng, and E. Law. 2021. Using Comics to Introduce and Reinforce Programming Concepts in CS1.
- [193] Sangho Suh and Jian Zhao. 2022. CodeToon: Story Ideation, Auto Comic Generation, and Structure Mapping for Code-Driven Storytelling. Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology

(2022).

- [194] Y. Sumi. 2017. ComicQA: contextual navigation aid by hyper-comic representation. Proceedings of the 19th International Conference on Information Integration and Web-based Applications & Services (2017).
- [195] Weihan Sun, J. Burie, J. Ogier, and K. Kise. 2013. Specific Comic Character Detection Using Local Feature Matching. 2013 12th International Conference on Document Analysis and Recognition (2013), 275–279.
- [196] Weihan Sun and K. Kise. 2011. Similar Manga Retrieval Using Visual Vocabulary Based on Regions of Interest. 2011 International Conference on Document Analysis and Recognition (2011), 1075–1079.
- [197] Weihan Sun and K. Kise. 2013. Detection of exact and similar partial copies for copyright protection of manga. International Journal on Document Analysis and Recognition (IJDAR) 16 (2013), 331–349.
- [198] Mingxing Tan and Quoc V. Le. 2019. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv abs/1905.11946 (2019).
- [199] Hideki Tanaka, Ryosuke Yamanishi, and Jun-ichi Fukumoto. 2015. Relation Analysis between Speech Balloon Shapes and their Serif Descriptions in Comic. 2015 IIAI 4th International Congress on Advanced Applied Informatics (2015), 229–233.
- [200] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images. ArXiv abs/2301.04883 (2023).
- [201] Pakpoom Tanapichet, N. Cooharojananone, and R. Lipikorn. 2011. Automatic comic strip generation using extracted keyframes from cartoon animation. 2011 International Symposium on Intelligent Signal Processing and Communications Systems (ISPACS) (2011), 1–6.
- [202] Akira Terauchi, N. Mori, and Miki Ueno. 2019. Analysis Based on Distributed Representations of Various Parts Images in Four-Scene Comics Story Dataset. 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW) 1 (2019), 50–55.
- [203] Baris Batuhan Topal, Deniz Yuret, and T. M. Sezgin. 2022. Domain-Adaptive Self-Supervised Pre-Training for Face & Body Detection in Drawings. ArXiv abs/2211.10641 (2022).
- [204] Hugo Touvron, M. Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv’e J’egou. 2020. Training data-efficient image transformers & distillation through attention. International Conference on Machine Learning

(2020), 10347–10357.

- [205] Koki Tsubota, Daiki Ikami, and K. Aizawa. 2019. Synthesis of Screentone Patterns of Manga Characters. 2019 IEEE International Symposium on Multimedia (ISM) (2019), 212–2123.
- [206] Miki Ueno and H. Isahara. 2017. Story Pattern Analysis Based on Scene Order Information in Four-Scene Comics. 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR) 03 (2017), 78–83.
- [207] Emanuele Vivoli, Marco Bertini, and Dimosthenis Karatzas. 2024. CoMix: A Comprehensive Benchmark for Multi-Task Comic Understanding. ArXiv abs/2407.03550 (2024).
- [208] Emanuele Vivoli, Niccoló Biondi, Marco Bertini, and Dimosthenis Karatzas. 2024. ComiCap: A VLMs pipeline for dense captioning of Comic Panels. ArXiv abs/2409.16159 (2024). https://api.semanticscholar.org/CorpusID:272831696
- [209] Emanuele Vivoli, Ali Furkan Biten, Andrés Mafla, Dimosthenis Karatzas, and Lluís Gómez. 2022. MUST-VQA: MUltilingual Scene-text VQA. ECCV Workshops (2022), 345–358.
- [210] Emanuele Vivoli, Irene Campaioli, Mariateresa Nardoni, Niccoló Biondi, Marco Bertini, and Dimosthenis Karatzas.

2024. Comics Datasets Framework: Mix of Comics datasets for detection benchmarking. IEEE International Conference on Document Analysis and Recognition (2024), 154–167.

- [211] Emanuele Vivoli, Joan Lafuente Baeza, Ernest Valveny Llobet, and Dimosthenis Karatzas. 2024. Multimodal Transformer for Comics Text-Cloze, In Document Analysis and Recognition - ICDAR 2024, Elisa H. Barney Smith, Marcus Liwicki, and Liangrui Peng (Eds.). Document Analysis and Recognition - ICDAR 2024, 128–145.
- [212] Cong Wang, Jiaxi Gu, Panwen Hu, Songcen Xu, Hang Xu, and Xiaodan Liang. 2023. DreamVideo: High-Fidelity Image-to-Video Generation with Image Retention and Text Guidance. ArXiv abs/2312.03018 (2023).
- [213] Meng Wang, Richang Hong, Xiao-Tong Yuan, Shuicheng Yan, and Tat-Seng Chua. 2012. Movie2Comics: Towards a Lively Video Content Presentation. IEEE Transactions on Multimedia 14 (2012), 858–870.
- [214] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023. CogVLM: Visual Expert for Pretrained Language Models. ArXiv abs/2311.03079 (2023).

- [215] Zezhong Wang, Hugo Romat, Fanny Chevalier, N. Riche, Dave Murray-Rust, and Benjamin Bach. 2021. Interactive Data Comics. IEEE Transactions on Visualization and Computer Graphics PP (2021), 1–1.
- [216] Xin Wei, Haifeng Zhao, Ziyu Gao, Yan Zhang, Jun Zhou, Zhixiang Chen, and Qidi Lu. 2022. ComicLib: A New Large-Scale Comic Dataset for Sketch Understanding, In International Conference on Digital Image Computing: Techniques and Applications. 2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA) (2022), 1–8.
- [217] Gregor Wiedemann and Gerhard Heyer. 2017. Page Stream Segmentation with Convolutional Neural Nets Combining Textual and Visual Features. ArXiv abs/1710.03006 (2017).
- [218] Michael J. Wilber, Chen Fang, Hailin Jin, Aaron Hertzmann, J. Collomosse, and Serge J. Belongie. 2017. BAM! The Behance Artistic Media Dataset for Recognition Beyond Photography. 2017 IEEE International Conference on Computer Vision (ICCV) (2017), 1211–1220.
- [219] Huisi Wu, Ziheng Ma, Wenliang Wu, Xueting Liu, Chengze Li, and Zhenkun Wen. 2023. Shading-Guided Manga Screening From Reference. IEEE Transactions on Visualization and Computer Graphics 30 (2023), 4941–4954.
- [220] Yejun Wu. 2010. Searching Digital Political Cartoons. 2010 IEEE International Conference on Granular Computing

(2010), 541–545.

- [221] Chenshu Xu, Xuemiao Xu, Nanxuan Zhao, Weiwei Cai, Huaidong Zhang, Chengze Li, and Xueting Liu. 2023. PanelPage-Aware Comic Genre Understanding. IEEE Transactions on Image Processing 32 (2023), 2636–2648.
- [222] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S. Ryoo, Shrikant B. Kendre, Jieyu Zhang, Can Qin, Shu Zhen Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, and Ran Xu. 2024. xGen-MM (BLIP-3): A Family of Open Large Multimodal Models. ArXiv abs/2408.08872 (2024).
- [223] Wilson Yan, Andrew Brown, Pieter Abbeel, Rohit Girdhar, and S. Azadi. 2023. Motion-Conditioned Image Animation for Video Editing. ArXiv abs/2311.18827 (2023).
- [224] Hideaki Yanagisawa, Takuro Yamashita, and Hiroshi Watanabe. 2018. A study on object detection method from manga images using CNN. 2018 International Workshop on Advanced Image Technology (IWAIT) (2018), 1–4.
- [225] Ling Yang, Zhilong Zhang, Shenda Hong, Runsheng Xu, Yue Zhao, Yingxia Shao, Wentao Zhang, Ming-Hsuan Yang, and Bin Cui. 2022. Diffusion Models: A Comprehensive Survey of Methods and Applications. Comput. Surveys 56

(2022), 1 – 39.

- [226] Xin Yang, Zongliang Ma, Letian Yu, Ying Cao, Baocai Yin, Xiaopeng Wei, Qiang Zhang, and Rynson W. H. Lau. 2021. Automatic Comic Generation with Stylistic Multi-page Layouts and Emotion-driven Text Balloon Generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 17 (2021), 1 – 19.
- [227] Yi-Ting Yang and W. Chu. 2023. Manga Text Detection with Manga-Specific Data Augmentation and Its Applications on Emotion Analysis. Conference on Multimedia Modeling (2023), 29–40.
- [228] Zhishen Yang, Tosho Hirasawa, Edison Marrese-Taylor, and Naoaki Okazaki. [n.d.]. Large Language Models as Manga Translators: A Case Study. https://api.semanticscholar.org/CorpusID:271271650
- [229] Chih-Yuan Yao, Husan-Ting Chou, Yu-Sheng Lin, and Kuo-Wei Chen. 2023. Screentone-Aware Manga Super-Resolution Using DeepLearning. ArXiv abs/2305.08325 (2023).
- [230] Chih-Yuan Yao, Shih-Hsuan Hung, Guo-Wei Li, I-Yu Chen, Reza Adhitya, and Yu-Chi Lai. 2017. Manga Vectorization and Manipulation with Procedural Simple Screentone. IEEE Transactions on Visualization and Computer Graphics 23

(2017), 1070–1084.

- [231] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qi-An Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. ArXiv abs/2408.01800 (2024).
- [232] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, J. Leskovec, Percy Liang, M. Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Retrieval-Augmented Multimodal Language Modeling. ArXiv abs/2211.12561 (2023).
- [233] K. Yu, Hyoungju Kim, Jeongin Kim, Chanjun Chun, and Pankoo Kim. 2023. A Study on Generating Webtoons Using Multilingual Text-to-Image Models. Applied Sciences (2023).
- [234] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 9556–9567.
- [235] Chengchang Zeng, Shaobo Li, Qin Li, Jie Hu, and Jianjun Hu. 2020. A Survey on Machine Reading Comprehension: Tasks, Evaluation Metrics, and Benchmark Datasets. ArXiv abs/2006.11880 (2020).

- [236] Max Zeyen, Tobias Post, H. Hagen, J. Ahrens, D. Rogers, and R. Bujack. 2018. Color Interpolation for Non-Euclidean Color Spaces. 2018 IEEE Scientific Visualization Conference (SciVis) (2018), 11–15.
- [237] Ci-Yin Zhang and Wei-Ta Chu. 2023. Occlusion-Aware Manga Character Re-identification with Self-Paced Contrastive Learning.
- [238] Heng Zhang, Lifeng Zhu, Qingdi Chen, Ai-Guon Song, and L. Yu. 2023. Augmenting Conversations With Comic-Style Word Balloons. IEEE Transactions on Human-Machine Systems 53 (2023), 367–377.
- [239] Lvmin Zhang, Yi Ji, and Xin Lin. 2017. Style Transfer for Anime Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN. 2017 4th IAPR Asian Conference on Pattern Recognition (ACPR) (2017), 506–511.
- [240] Lvmin Zhang, Xinrui Wang, Qingnan Fan, Yi Ji, and Chunping Liu. 2021. Generating Manga from Illustrations via Mimicking Manga Creation Workflow. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

(2021), 5638–5647.

- [241] Zhimin Zhang, Zheng Wang, and Wei Hu. 2022. Unsupervised Manga Character Re-identification via Face-body and Spatial-temporal Associated Clustering. ArXiv abs/2204.04621 (2022).
- [242] Jian Zhao, Shenyu Xu, Senthil K. Chandrasegaran, Chris Bryan, F. Du, Aditi Mishra, Xin Qian, Yiran Li, and K. Ma.

2021. ChartStory: Automated Partitioning, Layout, and Captioning of Charts into Comic-Style Narratives. IEEE Transactions on Visualization and Computer Graphics 29 (2021), 1384–1399.

- [243] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, Hongfa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Liejie Yuan. 2023. LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment. ArXiv abs/2310.01852 (2023).
- [244] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. MiniGPT-4: Enhancing VisionLanguage Understanding with Advanced Large Language Models. ArXiv abs/2304.10592 (2023).
- [245] Yi Zhu, Xinyu Li, Chunhui Liu, M. Zolfaghari, Yuanjun Xiong, Chongruo Wu, Zhi Zhang, Joseph Tighe, R. Manmatha, and Mu Li. 2020. A Comprehensive Study of Deep Video Action Recognition. ArXiv abs/2012.06567 (2020).

