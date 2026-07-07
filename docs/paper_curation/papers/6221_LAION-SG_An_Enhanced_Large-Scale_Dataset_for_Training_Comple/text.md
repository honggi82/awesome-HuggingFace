# arXiv:2412.08580v2[cs.CV]13Dec2024

## LAION-SG: An Enhanced Large-Scale Dataset for Training Complex Image-Text Models with Structural Annotations

Zejian Li1 Chenye Meng1 Yize Li2 Ling Yang3 Shengyuan Zhang1 Jiarui Ma1 Jiayi Li2 Guang Yang4 Changyuan Yang4 Zhiyuan Yang4 Jinxiong Chang5 Lingyun Sun1

1 Zhejiang University 2 Jiangnan University 3 Peking University 4 Alibaba Group 5 Ant Group 1 {zejianlee,mengcy}@zju.edu.cn

One or two relations

Three or four relations More than four relations

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

LAION Images

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

SDXL

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

SDXL-SG

###### (a) (b) (c) (d) (e) (f)

Annotation CloudsMountainsurroundreflectedmountainin lake

Bear surrounded by grass

Person hold sword Birds fly above clouds Person above mountains

Person facing mountain Tent placed on mountain Sun illuminating mountain

Table paced onrug Chair placed around table Window adjacent to brick Fruit placed on table Vase placed on table Flower inside vase

Person holding basket Fence encloses house Chicken near house Person looking at chicken Plant grows side of house

Grass

Figure 1. The generated images given by the text-to-image (T2I) model SDXL [31] and SDXL-SG, a T2I model with structural annotation guidance given different numbers of relations. For one or two relations, both models can generate accurately. When dealing with three or four relations, the T2I model fails to generate the relations “hold” and “facing”. For cases with more than four relations, the limitations of the T2I model become more pronounced. In (e) and (f), three and two relations are incorrectly generated, respectively. In comparison, SDXL-SG accurately captures the relations as shown in the generated images.

### Abstract

Recent advances in text-to-image (T2I) generation have shown remarkable success in producing high-quality images from text. However, existing T2I models show decayed performance in compositional image generation involving multiple objects and intricate relationships. We attribute this problem to limitations in existing datasets of image-text pairs, which lack precise inter-object relationship annotations with prompts only. To address this problem, we con-

struct LAION-SG, a large-scale dataset with high-quality structural annotations of scene graphs (SG), which precisely describe attributes and relationships of multiple objects, effectively representing the semantic structure in complex scenes. Based on LAION-SG, we train a new foundation model SDXL-SG to incorporate structural annotation information into the generation process. Extensive experiments show advanced models trained on our LAIONSG boast significant performance improvements in complex scene generation over models on existing datasets.

We also introduce CompSG-Bench, a benchmark that evaluates models on compositional image generation, establishing a new standard for this domain. Our annotations with the associated processing code, the foundation model and the benchmark protocol are publicly available at https://github.com/mengcye/LAION-SG.

### 1. Introduction

Text-to-image (T2I) generative models [7, 9, 24, 31, 33, 36, 41, 59] have made significant strides, showcasing an impressive ability to generate high-quality images from text prompts. However, when reviewing T2I generation models, it is observed that they are generally effective for simple scene generation but notably deteriorate when handling complex scenes—such as those involving multiple objects and intricate relations between them ( Fig. 1). We attribute this limitation to a lack of emphasis on complex inter-object associations within existing text-image datasets. Prior T2I works have primarily focused on architectural improvements, which fail to address this underlying issue.

Scene graphs (SG) provide a structured description of image content. A scene graph consists of nodes (representing objects and attributes) and edges (depicting relationships between objects). Compared to the sequential description of text, SGs offer compact, structured approaches describing complex scenes, enhancing annotation efficiency. SGs also allow for more precise specification of specific objects associated attributes and their relationships, a feature critical for generating complex scenes. However, existing scene graph datasets are relatively small in scale (e.g., COCO-Stuff [4] and Visual Genome [20]), while largescale datasets primarily consist of text annotations only.

Our work focuses on compositional image generation via scene graphs (SG2CIM). We construct LAION-SG dataset, a significant extension of LAION-Aestheics V2 (6.5+) [38] with high-quality, high-complexity scene graph annotations. Our annotations feature multiple objects, attributes and relationships describing images of high visual quality. Therefore, our LAION-SG better encapsulates the semantic structure of complex scenes, supporting improved generation for intricate scenarios. The advantage of LAION-SG in complex scene generation is validated in further experiments with multiple metrics on semantic consistency.

With LAION-SG, we train existing models and propose a new baseline for generating complex scenes with SGs. To build the baseline, we use SDXL [31] as the backbone model and train an auxiliary SG encoder to incorporate SG within the image generation process. Specifically, the SG encoder employs a graph neural network (GNN) [37] to extract scene structure in graphs, thereby optimizing the SG embeddings. These embeddings are then fed into the backbone model to yield high-quality complex images. Our ap-

proach efficiently enhances the model’s ability to generate complex scenes and provides a foundational model.

Finally, we establish CompSGen Bench, a benchmark for complex scene generation evaluation. Using this benchmark, we evaluate existing state-of-the-art (SOTA) models alongside our baseline variants trained on COCO-Stuff [4], Visual Genome [20], and our LAION-SG. Both quantitative and qualitative results demonstrate models trained on LAION-SG outperforming other model trained on COCOStuff and Visual Genome as in current T2I and SG2IM baselines. We conclude that LAION-SG dataset significantly enhances complex scene generation. Our LAION-SG dataset represents a pioneering effort in annotating complexity on existing image datasets and holds potential for wider applications for scene perception and synthesis.

In summary, our contributions are as follows:

- • We introduce a new scene graph (SG)-based dataset, LAION-SG, for complex scene image generation. This dataset includes high-quality SG annotations with multiple objects, their attributes, and numerous relationships, enhancing generative models’ capability to handle complex scenes and improving the complexity and fidelity of generated images.
- • We fine-tune a new efficient foundation model on the proposed dataset. The model demonstrates heightened sensitivity to image content awareness and competitive performance in complex scene generation, setting a new baseline for SG-image understanding.
- • We establish CompSGen Bench, a benchmark for complex scene generation evaluation with several metrics, and conduct extensive experiments to validate the effectiveness of our dataset and baseline.

### 2. Related Work

Compositional Image Generation. Text-to-image generation [7, 9, 24, 31, 33, 36, 41, 59] has advanced significantly, particularly through diffusion models [16, 34]. However, the sequential format of the textual data imposes limitations on image generation. This limitation is particularly evident when generating compositional images that involve multiple objects and associated relationships [25, 51, 54].

Previous studies have explored various methods to enhance the controllability of the text-to-image diffusion model. Compositional Diffusion [28] breaks down complex text prompts into multiple easily generated segments, but it is limited to conjunction and negation operators. Attendand-Excite [6] guides pre-trained diffusion models to generate all entities in the text through immediate reinforcement activation, yet it still faces attribute leakage issues.

Other approaches use additional spatial conditions to improve generation. GLIGEN [23] integrates bounding boxes by adding trainable gated self-attention layers while freezing the original weights. Ranni [13] incorporates a semantic

panel containing layouts, colors, and keypoints parsed from Large Language Model (LLM) for training diffusion models. However, these efforts require costly training.

Universal Guidance [3] utilizes a well-trained object detector and constructs new losses to enforce the generated images to match the positional guidance. BoxDiff [48] encourages the desired objects to appear in specified regions by calculating losses based on the maximum values in the cross-attention maps. RealCompo [53] achieves a balance between the realism and complexity of images by adjusting the influence of textual and layout guidance in the denoising process. Nevertheless, these methods rely on the accuracy of the initial bounding boxes, despite some works [12, 25, 52] attempt to directly generate spatial layout information based on textual conditions using LLM.

For text-to-image generation, all of these methods mainly focus on model improvement, failing fundamentally to address the limitations imposed from the dataset.

Image Generation from Scene Graphs (SG2IM) [19, 20] involves creating images based on structured representations of scenes, where objects and their relationships are explicitly defined as a graph [49].

The traditional SG2IM methods typically consist of two stages [1, 10, 19, 28]. They first transform the input scene graph into a layout, which serves as an auxiliary for image refinement using a generative model. These methods are effective for generating images with a few objects and simple relationships, but they often generate confusingly when relations become abstract and the number of objects increases.

Therefore, some methods propose directly learning the alignment from scene graphs to images [11, 42, 46, 55]. SGDiff [50] pre-trains an SG encoder using contrastive learning, which is combined with Stable Diffusion (SD) to generate images. SG-Adapter [40] is trained to fine tune SD, incorporating SG information into the text-to-image process through attention layers. R3CD [27] enhances large-scale diffusion models by leveraging SG transformers to learn abstract interactions and improve image generation. These methods overcome the issues associated with sequential text conditions, enhancing the model’s semantic expressive capacity. However, the current SG-image datasets do not match the scale and quality of text-image datasets, resulting in a quality bottleneck for SG2IM methods.

Large-Scale Image-Text Datasets and Benchmarks. Previous datasets, such as MS-COCO [26], Visual Genome [20], and ImageNet [8], are primarily constructed through manual annotation. These datasets are highly accurate in terms of labeling but limited in scale due to the considerable costs associated with manual annotation. To address the limitations, some approaches propose automatically extracting and filtering data from websites. The Conceptual Captions dataset (CC3M) [39], for instance, contains approximately 3.3 million image-text pairs, and

its extended version, CC12M [5], increases the dataset size to 12 million pairs. LAION-5B [38] further expands dataset scale, comprising around 5.85 billion image-text pairs, making it one of the largest publicly available imagetext datasets. LAION-Aesthetics, a subset of LAION-5B, is curated for high visual quality and intended to support image generation and aesthetic research. However, it does not always ensure textual descriptions that accurately reflect image content. Thus we enhance LAION-Aesthetics with structured annotations in the form of Scene Graphs (SG), constructing a high-quality, large-scale dataset tailored for generating compositional images.

In text-to-image generation, several benchmarks comprehensively assess model performance across various aspects. T2I-CompBench [17] provides 6,000 compositional prompts covering attribute binding, object relationships, and complex compositions. HRS-Bench evaluates models on 13 skills across 50 scenarios, including accuracy, robustness, and bias [2]. HEIM assesses 12 dimensions, such as text-image alignment, aesthetics, and multilinguality [22]. VISOR focuses on spatial relationships with its SR2D dataset [14], and HPS v2 enables comparison based on human preferences [45]. These benchmarks only focus on text-based image generation. To fill the gap in this domain, we are the first to propose a complex scene generation benchmark based on scene graphs.

### 3. Dataset and Benchmark

A large-scale, high-quality dataset is essential for learning compositional image generation. However, existing largescale T2I datasets, such as LAION, describe content beyond the images (as illustrated in Fig. 5), misleading the generation. In contrast, SG datasets tend to focus more specifically on the actual content within images, namely the objects and relationships. Nonetheless, current SG datasets, such as COCO and VG, are relatively small in scale and have limited object and relationship types, making them insufficient for compositional image generation.

To address this, we propose LAION-SG dataset, which is a large-scale, high-quality, open-vocabulary SG dataset. We also introduce the Complex Scene Generation Benchmark (CompSGen Bench) to evaluate models’ performance .

#### 3.1. Dataset Construction

Our LAION-SG dataset is built on high-quality images in LAION-Aesthetic V2 (6.5+) [38] with automated annotation performed using GPT-4o [30]. LAION-Aesthetics V2 (6.5+) is a subset of LAION-5B [38], comprising 625,000 image-text pairs with predicted aesthetic scores over 6.5, curated using the LAION-Aesthetics Predictor V2 model. During our construction, only 540,005 images are available in the dataset due to copyright or other issues.

|[Figure 19]<br><br>Multimodal Large Language Model<br><br>|1. Object Recognition|
|---|
<br><br>|2. Attribution Recognition|
|---|
<br><br>|3. Relation Recognition|
|---|
<br><br>|4. Other constraints|
|---|
|
|---|

[Figure 20]

[Figure 21]

LAION Image

Scene Graph

|• Adjective, no concrete object.<br>• Every object has at least one attribute and can have multiple.<br><br>[Figure 22]<br><br>stone<br><br>castle<br><br>for ﬁed<br><br>• Format : [int(item id), "attribute1", "attribute2", ...].<br><br><br>[Figure 23]<br><br>|person|
|---|
<br><br>male young<br><br>[Figure 24]<br><br>|person|
|---|
<br><br>book−holding ❌ ✅<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

|• Give each object you see a unique id, ensuring distinctly recognition.<br>• Format : item_int (id).<br><br>[Figure 27]<br><br>Item_id: 3<br><br>Item_id: 2 Item_id: 0<br><br>Item_id: 1 Item_id: 4<br><br>• Assign multiple identical objects with different ids but the same name.<br><br>[Figure 28]<br><br>| |
|---|
<br><br>Item_id: 3 label: tree<br><br>| |
|---|
<br><br>Item_id: 2 label: tree<br><br>• Skip some objects if there are too many.<br><br><br>[Figure 29]|
|---|

|• No simple relation.<br>• Use more precise verbs (no "overlaps" etc.). Don't repeat.<br>• Format: [int(id), "relation", int(id)].<br><br><br>[Figure 30]<br><br>| | | |below|
|---|---|---|---|
| | | | |
| | |tag| |
<br><br>collar<br><br>[Figure 31]<br><br>❌<br><br>[Figure 32]<br><br>| | | | |
|---|---|---|---|
| | |tag| |
<br><br>collar hang from<br><br>[Figure 33]<br><br>pepper<br><br>garlic<br><br>bag overlaps<br><br>overlaps<br><br>[Figure 34]<br><br>pepper<br><br>garlic<br><br>bag ﬂanked by<br><br>near<br><br>[Figure 35]<br><br>❌<br><br>[Figure 36]<br><br>✅<br><br>[Figure 37]<br><br>✅|
|---|

|• For people, use "person" and give attributes (e.g. gender and age).<br>• Objective, no personification, no associations.<br>• STRICTLY follow the format: "items" : ["person_0", "book_1",], "attributes" : [[0, "male", "young"], [1, "small", "rectangular"]], "relations" : [[0, "holding", 1]]<br><br><br>[Figure 38]<br><br>boy person<br><br>[Figure 39]<br><br>male<br><br>young<br><br>[Figure 40]<br><br>❌ ✅<br><br>[Figure 41]|
|---|

- Figure 2. The construction pipeline of LAION-SG dataset. 1) Identify the objects in the image and assign a unique ID to each. 2) The attributes must be abstract adjectives and should not include specific objects. Each object may have one or more attributes. 3) The relations between objects should be as specific as possible, avoiding simple relations. Use more precise verbs, minimizing repetition. 4) For people, label the object as “person” and include attributes such as gender and age. Avoid anthropomorphism or associations, and provide an objective description of what is observed in the image.

0 20 40 60 80 100 120 140

Length of Annotation

0.0

0.2

0.4

0.6

0.8

1.0

SG-IoU+

Annotation

Original Text

Scene Graph

0 20 40 60 80 100 120 140

Length of Annotation

0.0

0.2

0.4

0.6

0.8

1.0

Entity-IoU+

Annotation

Original Text

Scene Graph

0 20 40 60 80 100 120 140

Length of Annotation

0.0

0.2

0.4

0.6

0.8

1.0

Relation-IoU+

Annotation

Original Text

Scene Graph

- Figure 3. The annotation length and accuracy characteristics of LAION-SG compared to the LAION-Aesthetics. Compared to text, the scene graph, as a more compact form, has a longer length and its accuracy is more concentrated in high-scoring areas. This suggests that our LAION-SG annotation more accurately reflects the image information and contains richer semantics.

Through prompt engineering, we devised a set of specific requirements for scene graph annotations to ensure comprehensiveness, systematic structure, and precision in the annotation results. Fig. 2 illustrates the detailed construction pipeline of LAION-SG. Each component plays a crucial role in achieving high-quality automated annotation.

hierarchy is accurately represented.

Second, the attribute section mandates that each object must have at least one abstract adjective attribute, while avoiding the use of other objects as attributes. This design is especially important in complex scenes as it helps differentiate objects’ appearance, state, and characteristics from the background and other elements, maintaining consistency and clarity in annotations. By avoiding the confusion between specific objects and abstract attributes, the annotations become more interpretable and generalizable.

First, as scene graphs typically contain multiple objects and their relations, the prompt requires “identification of as many objects, attributes, and their relations within the image as possible”. This design encourages that all objects and interactions in a scene are annotated. Each object is assigned a unique ID, even for multiple objects of the same type, ensuring that the entirety of the scene’s structure and

In the relation section, we specify the use of concrete verbs to describe relations between objects rather than relying solely on spatial orientation. This is because rela-

(a)

###### Relation Object Attribute Scene Graph

>20 words

>20 words

0-5 words >20 words

0-10words

10-20 words

10-20 words

0.79% 12.46%

8.07%

3.53% 16.45%

10.39%

>30 words

13.65%

28.66%

0-5 words

0-5 words

10-20 words

43.33%

45.71%

37.66% 32.15%

10-20 words

41.04% 40.62%

36.69%

5-10 words

5-10 words

28.80%

5-10 words

20-30 words

Top 10 Relations (Count) Top 10 Attributes (Count)

(b)

80058 (3.78%) 65592 (3.1%)

405106 (7.36%)

surrounded by adjacent to surround holding near

tall small large

252111 (4.58%) 233905 (4.25%)

64350 (3.04%)

46762 (2.21%) 44244 (2.09%)

228585 (4.15%) 184718 (3.35%)

green young

41435 (1.96%) 39117 (1.85%)

181236 (3.29%) 174447 (3.17%)

placed on attached to

female wooden

35265 (1.66%) 33144 (1.56%)

170711 (3.1%) 168367 (3.06%)

worn by reflect part of

male white distant

32033 (1.51%)

121379 (2.2%)

- Figure 4. The annotation distribution of LAION-SG. (a) The length of the scene graph lies in a wide range. Our annotation provides more specific information compared to single-word descriptions, while also avoiding the inefficiency in model learning caused by excessively lengthy annotations. (b) The top 10 relations and attributes represent only a small percentage of the total distribution, indicating that LAION-SG covers a highly diverse range of annotations, showcasing its large scale and open vocabulary.

##### Annotation # Objects (w/o Proper Noun) Length SG-IoU+↑ Ent.-IoU+↑ Rel.-IoU+↑

LAION Caption 5.33±3.94 (2.02±3.01) 19.0±19.7 0.306 0.631 0.557 LAION-SG (Ours) 6.39±4.17 32.2±20.3 0.422 0.810 0.749

Table 1. The number of objects and length per sample, and the average accuracy for 300 samples across different annotation types.

tions are often more critical in scene graphs than mere spatial information. By using precise verbs like “standing on” or “holding”, we capture dynamic interactions within the scene, which is essential for complex scene generation.

Leveraging these prompts with the multimodal large language model GPT-4o, we generate annotations representing scene graphs. Our annotation is expect to achieve accuracy for every object, attribute, and relationship, thoroughly covering each detail in the scene and providing robust data support for subsequent compositional image generation tasks.

#### 3.2. LAION-SG Dataset

By performing the construction strategy, we develop LAION-SG, a large-scale, high-quality dataset containing 540,005 SG-image pairs annotated with objects, attributes, and relationships. This dataset is divided into a training set of 480,005 samples, a validation set of 10,000 samples, and a test set of 50,000 samples. We present statistics comparing the original LAION-Aesthetics text-to-image dataset with our LAION-SG dataset as follows.

As shown in Tab. 1, in the original LAION-Aesthetics caption, the average number of objects per sample is 5.33, with 38% of these being proper nouns that offer limited guidance during model training. For our SG annotations, the average number of objects per sample increases to 6.39, excluding abstract proper nouns and focusing on specific nouns that reflect true semantic relationships. LAION-SG

contains 20% more object information than the original LAION-Aesthetics dataset, and this advantage increases to 216% when excluding proper nouns.

Additionally, we calculated the relationship between length and accuracy for different annotations. The annotation length for text is defined as the number of tokens in the prompt, while for SG as the total number of nodes and edges. We leverage SG-IoU+, Entity-IoU+, and RelationIoU+ introduced in Sec. 5.2 to measure annotation accuracy.

The average annotation length for original captions and our scene graphs is 19.0 and 32.2, respectively, with SG achieving higher accuracy across all three metrics. Fig. 3 visualizes the length and accuracy of samples for both annotation types. Note that a scene graph is a more structured and compact form of annotation compared to text. Even so, the SG length is still significantly longer than sparse text, and its accuracy is also much higher. This demonstrates that our LAION-SG dataset contains richer, more nuanced, and precise semantic features, which can greatly enhance model performance in image generation, fundamentally addressing the challenges of generating complex scenes.

Furthermore, we analyze the length distribution of scene graphs in LAION-SG in Fig. 4 (a). Most objects are described by 0-5 (45.72%) or 5-10 (41.04%) words, with a smaller proportion described by 10-20 (12.46%) words or more than 20 (0.79%) words. This range is reasonable, of-

fering a more precise expression than a single word while avoiding excessive length that could hinder model learning efficiency. In terms of the overall scene graph, the proportions of word counts in the ranges 0-10, 10-20, 20-30, and more than 30 are 10.39%, 32.15%, 28.80%, and 28.66%, respectively. These statistics reflect the richness, detail, and flexibility of annotations in LAION-SG.

- Fig. 4 (b) presents the top 10 most frequent relations and

attributes in LAION-SG. The most commonly included relation terms are generally specific in semantics like “surrounded by”, “adjacent to” and “holding”. The most frequent relation is “surrounded by”, occurring 80,058 times and accounting for 3.78% of all relations. The most common attribute is “tall”, representing 7.36% of all attributes, while the second most common, “small”, accounts for only

- 4.58%. The tenth-ranked relation and attribute each make up only 1.51% and 2.2% of their respective totals. These data indicate the annotations in LAION-SG are highly diverse and broadly covered, as even the most frequently used descriptors represent only a small percentage of the total.

#### 3.3. Complex Scene Generation Benchmark

To evaluate model performance on compositional image generation, we propose Complex Scene Generation Benchmark (CompSGen Bench). From the 50,000-image test set, we select samples with over four relations as complex scenes, and get a total of 20,838 samples. We calculate FID [21] , CLIP score [32], and three accuracy metrics [40] to assess models’ performance.

FID measures the overall quality of generated images, while the CLIP Score calculates the similarity between the generated and ground truth images. The complex scene evaluation consists of three metrics: SG-IoU, Entity-IoU, and Relation-IoU. They represent the overlap between the generated images and the real annotations in terms of scene graphs, objects, and relations, respectively. Sec. 5.4 shows the test results for different models on CompSGen Bench.

### 4. Foundation Model

As the complexity of the image increases, the generated results of T2I models become more difficult to control (Fig. 1). We introduce a foundation model to address the challenges of compositional image generation in T2I task. Our model is built on top of Stable Diffusion XL (SDXL) [31] and incorporates SG information via graph neural networks (GNN) [37].

An SG consists of multiple triples and single objects. Our baseline initializes each triple and single object separately using the CLIP text encoder ET(·). For single objects, the initialization result from CLIP serves as the final representation, denoted as es. For SG triples, each of them is encoded by CLIP to yield a corresponding triple embedding et = ET(triplesg). Our SG encoder extracts object

and relation embeddings as the nodes and edges and inputs them into the GNN to optimize the SG embedding. More calculation details can be found in supplementary material.

If a relation contains multiple words, each word contributes an edge connecting the nodes of the two related objects. Attributes are treated as separate nodes connected to their respective objects. After processing with the GNN, we obtain a refined triple embedding, denoted as er.

To stabilize the training, we introduce a learnable scaling factor α to control the strength of the refined embedding. α is initialized as zero and updated throughout training. Finally, all triple embeddings are concatenated with singleobject embeddings to form the SG embedding esg, which is fed into the U-Net of SDXL for iterative noise prediction.

esg = f(sg) = concat(et + αer,es) (1) Given an image x and its corresponding SG, the SG en-

coder is trained with:

L = EE(x),sg,ϵ,t[∥ ϵ − ϵθ(zt,t,f(sg)) ∥22] (2)

Here, E(·) denotes the CLIP image encoder, and zt represents the latent code of the image. ϵθ represents the predicted noise, parameterized by θ. f(sg) encapsulates the SG embedding output from SG encoder of our baseline. We train the parameters of SG encoder to minimize the gap between the predicted and added noise, thereby fitting the distribution of compositional images.

### 5. Experiments

#### 5.1. Implementation and Baselines

We compare our trained SDXL-SG with three advanced baselines: SDXL [31], SG-Adapter [40], and SGdiff [50], following their evaluation settings. For SDXL-SG, we initialize the scene graph embeddings using OpenCLIP ViTbigG/14 [18] and CLIP ViT-L/14 [32], similar to SDXL. These embeddings are refined with a 5-layer SG Encoder, each with 512 input and output dimensions. The refined embeddings are subsequently injected into SDXL for image generation. For our training, we employ the Adam optimizer with a learning rate of 5e-4, training for one epoch on the full LAION-SG dataset. All training processes are conducted on eight NVIDIA RTX 4090D GPUs.

#### 5.2. Datasets and Evaluation Metrics

We train existing models and SDXL-SG on COCO-Stuff, Visual Genome (VG), and LAION-SG datasets. We evaluate the generated results based on image quality and accuracy in target content within complex scenes. Fr´echet Inception Distance (FID) [21] measures the overall visual quality of generated images, while the CLIP Score evaluates the similarity between generated and ground truth (GT) images. For assessing complex scenes, we calculate SG-IoU,

Original LAION Caption

Scene Graph

GT SDXL SGDiff SG-Adapter SDXL-SG

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

(a) (b) (c) (d)

[Figure 47]

John Singer Sargent. Paul Helleu Sketching His Wife Alice.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(e) (f) (g) (h)

[Figure 53]

Look dandy d‘Oscar Wilde.

Generation errors in Human body parts.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

(i) (j) (k) (l)

[Figure 59]

| | | | |
|---|---|---|---|
| | | | |

Still life with BellPeppers.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

(m) (n) (o) (p)

[Figure 65]

Brooklyn Botanic Garden, Engagement Session, NY, Cherry Blossoms, ...

- Figure 5. Visual comparison on LAION-SG. The compared methods include T2I model (SDXL [31]) and SG2IM models (SGDiff [50] and SG-Adapter [40]). The first column shows the original caption from LAION-Aesthetics. The second column displays the scene graph from our LAION-SG. The last five columns show ground truth images and images generated by different models. Objects or relations are highlighted with the same color in scene graphs and generated images to show SDXL-SG successfully captures the complex scenes.

Entity-IoU, and Relation-IoU [40]. They respectively indicate the degree of consistency between the scene graphs, objects and relations derived from the generated image and the real annotation. Higher scores indicate stronger model capability in generating complex scenes.

with more accurate objects and relations, even for relatively complex scenarios. For instance, in the first row, where the relationship is “male person painting female person”, both (a) and (b) fail to generate “painting”, and (c) generates two female figures, whereas SDXL-SG accurately and qualitatively generates the relations provided by the scene graph. Figures (f)-(p) illustrate more examples where ours outperforms existing baselines in relation generation.

To evaluate the annotation quality, we propose SG-IoU+, Entity-IoU+, and Relation-IoU+. Images are generated using scene graphs or LAION captions. The SG list, entity list, and relation list are then extracted from both the generated and GT images. The consistency between the corresponding lists of the two images is calculated to assess the annotation accuracy. The extraction is performed using GPT-4o. Given the high cost, we compute the average for 300 samples as the result. More calculation details can be found in the supplementary material.

Additionally, existing T2I and SG2IM models more frequently produce incorrect generations in (e). Other errors include erroneous number of generated objects such as bag in the green box in (i), person in the blue box in (m) and (n) or attribute errors such as bag in the yellow box in (j). SDXL-SG avoids these mistakes as in (d), (h), (l) and (p).

#### 5.4. Quantitative Results

#### 5.3. Qualitative Results

We compared results of SDXL [31], other SG2IM methods including SGDiff [50] and SG-Adapter [40], and SDXLSG. Each is tested when trained on different datasets. The original SGDiff [50] introduces bounding box as auxiliary data during training. For fair comparison, we train SGDiff without bounding box with the official implementation. We used FID to evaluate the quality of generated images. Fine-tuning pre-trained T2I models inevitably in-

- Fig. 5 displays 1024×1024 images generated on LAIONSG. Each row shows an original caption from the LAIONAesthetics dataset, a scene graph, the corresponding ground truth image, and images generated by different baseline models. The corresponding elements in the scene graph and the images are highlighted in matching colors.

Our foundation model SDXL-SG can generate scenes

Method Dataset FID↓ SGIoU↑

Ent.IoU↑

Rel.IoU↑

SDXL LAION 19.3 0.371 0.813 0.780 SGDiff w/o bbox

COCO 47.8 0.435 0.841 0.816 VG 35.2 0.529 0.801 0.795 LS 32.2 0.531 0.855 0.830

COCO 34.9 0.485 0.840 0.833 VG 39.5 0.515 0.803 0.782 LS 31.3 0.538 0.866 0.852

SGAdapter

COCO 30.0 0.497 0.842 0.833 VG 21.9 0.546 0.813 0.800 LS 20.1 0.558 0.884 0.856

SDXLSG (Ours)

- Table 2. Results on COCO-Stuff, Visual Genome and LAION-SG (LS). The first and second best is in bold and underlined.

Method FID↓ CLIP↑ SGIoU↑

Ent.IoU↑

Rel.IoU↑

SDXL 25.2 0.700 0.226 0.753 0.658 SGDiff 35.8 0.690 0.304 0.787 0.698 SG-Adapter 27.8 0.681 0.314 0.771 0.693 SDXL-SG (Ours)

26.7 0.698 0.340 0.792 0.703

- Table 3. The results of existing T2I and SG2IM models, as well as our baseline model, on the Complex Scene Generation Benchmark. The best is in bold, and the second best is underlined.

creases FID scores [35, 40, 44]. We also measure SG-IoU, Entity-IoU, and Relation-IoU [40].

As demonstrated in Tab. 2, our baseline achieves the best performance among all candidates in both image quality and accuracy. Notably, the SG-IoU of T2I model is significantly lower than that of SG2IM models, indicating that text provides far less control in the image generation process compared to structured annotations. This highlights the necessity of constructing a large-scale, highquality structured annotation dataset. Furthermore, for the same model, results trained on LAION-SG consistently outperformed those trained on COCO and VG. This suggests that our LAION-SG dataset is more effective than previous SG-image datasets due to its higher annotation quality.

Additionally, we evaluate the complex scene generation capability of baseline models on the CompSGen Bench (Sec. 3.3). As shown in Tab. 3, our baseline outperforms the existing SG2IM model in terms of image quality, similarity to the ground truth (GT) image, and content accuracy. Compared to SDXL, our FID score does not increase significantly even after fine-tuning, which typically raises FID scores. The slightly lower CLIP score (only by 0.02) compared to SDXL is due to pre-trained CLIP models incorporate abstract prior information such as background and historical knowledge. In contrast, our focus is on generating

Method Prop. FID↓ SGIoU↑

Ent.IoU↑

Rel.IoU↑

10% 31.6 0.522 0.794 0.790 20% 24.3 0.524 0.804 0.793 50% 22.9 0.535 0.800 0.796

SGAdapter

100% 21.9 0.546 0.813 0.800

10% 27.3 0.530 0.874 0.837 20% 24.5 0.533 0.877 0.838 50% 22.2 0.547 0.876 0.849

SDXLSG (Ours)

##### 100% 20.1 0.558 0.884 0.856

Table 4. Results of ablation study. Prop. denotes data proportion.

accurate and realistic content for complex scenes. SDXLSG significantly outperforms SDXL on accuracy metrics including SG-IoU, Entity-IoU and Relation-IoU.

We also conduct a quantitative analysis (detailed in Sec. 3.2 ) and a user study (detailed in supplementary material) to verify the effectiveness and strong correlation with human perception of structured annotations, indicating their significant advantage in expressing image content compared to sequential text.

#### 5.5. Ablation Study

We conduct ablation studies to demonstrate the positive impact of LAION-SG. We train SDXL-SG variants on 10%, 20%, 50%, and 100% samples of LAION-SG. The total training iterations remain constant across all settings for fairness. As the sample size increases, the model’s capability to generate compositional images improves significantly (Tab. 4). Notably, in the 10% LAION-SG ablation, where the data volume is smaller than that of VG, the model’s FID and Entity-IoU scores still outperform the results trained on VG, with SG-IoU and Relation-IoU scores remaining roughly comparable (Tab. 2). This indicates LAION-SG not only provides a data volume advantage but also features higher quality in images and annotations. This enhances model training efficiency and significantly improves performance in compositional image generation.

### 6. Conclusion

In this paper, we introduce LAION-SG, a high-quality, large-scale dataset with structured annotation designed for training complex image-text models. Compared to existing text-image datasets, LAION-SG provides more precise descriptions of objects, attributes, and relations, effectively capturing the semantic structure of complex scenes. Based on LAION-SG, we train a new foundational model SDXL-SG, integrating structured annotation into the generation process. Experiments demonstrate models trained on LAION-SG significantly outperform those trained on existing datasets. We also propose CompSG-Bench, a

benchmark for evaluating model performance in compositional image generation based on scene graph, establishing a new standard for complex scene generation. In summary, LAION-SG represents a pioneering effort in annotating complexity on image datasets and holds great potential for broader scene perception and synthesis applications.

### References

- [1] Oron Ashual and Lior Wolf. Specifying object attributes and relations in interactive scene generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4560–4568, 2019. 3

- [2] Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20041–20053, 2023. 3

- [3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 843–852, 2023. 3

- [4] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1209–1218, 2018. 2

- [5] Soravit Changpinyo, Piyush Kumar Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale imagetext pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3557–3567,

2021. 3

- [6] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Trans. Graph., 42(4), 2023. 2

- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. ArXiv, abs/2403.04692, 2024. 2

- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255, 2009. 3

- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, pages 8780–8794, 2021. 2

- [10] Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and MCMC. In Proceedings of the 40th

- International Conference on Machine Learning, pages 8489– 8510, 2023. 3
- [11] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. In Eleventh International Conference on Learning Representations (ICLR 2023), 2023. 3

- [12] Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Xuehai He, S Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. In Advances in Neural Information Processing Systems, pages 18225–18250, 2023. 3

- [13] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4744–4753, 2024. 2

- [14] Tejas Gokhale, Hamid Palangi, Besmira Nushi, Vibhav Vineet, Eric Horvitz, Ece Kamar, Chitta Baral, and Yezhou Yang. Benchmarking spatial relationships in text-to-image generation. ArXiv, abs/2212.10015, 2023. 3

- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 16

- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851, 2020. 2, 16

- [17] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In Advances in Neural Information Processing Systems, pages 78723–78747, 2023. 3

- [18] Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 6, 16
- [19] Justin Johnson, Agrim Gupta, and Li Fei-Fei. Image generation from scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1219–1228, 2018. 3

- [20] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123(1):32–73, 2017. 2, 3, 15

- [21] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, and Percy S Liang. Holistic evaluation of text-to-image models. In Advances in Neural Information Processing Systems, pages 69981–70011, 2023. 6

- [22] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, and Percy S Liang. Holistic evaluation of text-to-image models. In Advances in Neural Information Processing Systems, pages 69981–70011, 2023. 3

- [23] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22511–22521, 2023. 2

- [24] Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24142–24153, 2024. 2

- [25] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llmgrounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. ArXiv, abs/2305.13655, 2023. 2, 3

- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755,

2014. 3

- [27] Jinxiu Liu and Qi Liu. R3cd: Scene graph to image generation with relation-aware compositional contrastive control diffusion. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3657–3665, 2024. 3

- [28] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439, 2022. 2, 3

- [29] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 16

- [30] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, et al. Gpt-4 technical report,

2024. 3, 14

- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. ArXiv, abs/2307.01952,

2023. 1, 2, 6, 7, 12, 16

- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763, 2021. 6, 16

- [33] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. ArXiv, abs/2204.06125, 2022. 2

- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2022. 2, 16

- [35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22500–22510, 2023. 8

- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, pages 36479–36494, 2022. 2

- [37] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1): 61–80, 2008. 2, 6, 12

- [38] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems, pages 25278–25294, 2022. 2, 3, 16

- [39] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 3

- [40] Guibao Shen, Luozhou Wang, Jiantao Lin, Wenhang Ge, Chaozhe Zhang, Xin Tao, Yuanhui Zhang, Pengfei Wan, Zhong ming Wang, Guangyong Chen, Yijun Li, and Ying cong Chen. Sg-adapter: Enhancing text-to-image generation with scene graph guidance. ArXiv, abs/2405.15321, 2024. 3, 6, 7, 8, 15

- [41] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Trans. Graph., 43: 52:1–52:18, 2024. 2

- [42] Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, and Xiaodong Lin. Compositional text-to-image synthesis with attention map control of diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5544–5552, 2024. 3

- [43] Yunnan Wang, Ziqiang Li, Wenyao Zhang, Zequn Zhang, Baao Xie, Xihui Liu, Wenjun Zeng, and Xin Jin. Scene graph disentanglement and composition for generalizable complex image generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 12

- [44] Zixiao Wang, Farzan Farnia, Zhenghao Lin, Yunheng Shen, and Bei Yu. On the distributed evaluation of generative models. ArXiv, abs/2310.11714, 2024. 8

- [45] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. CoRR, abs/2306.09341, 2023. 3

- [46] Yang Wu, Pengxu Wei, and Liang Lin. Scene graph to image synthesis via knowledge consensus. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2856– 2865, 2023. 3

- [47] Yinwei Wu, Xianpan Zhou, Bing Ma, Xuefeng Su, Kai Ma, and Xinchao Wang. Ifadapter: Instance feature control for grounded text-to-image generation. ArXiv, abs/2409.08240,

2024. 12

- [48] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7452–7461,

2023. 3

- [49] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5410–5419, 2017. 3

- [50] Ling Yang, Zhilin Huang, Yang Song, Shenda Hong, G. Li, Wentao Zhang, Bin Cui, Bernard Ghanem, and Ming-Hsuan Yang. Diffusion-based scene graph to image generation with masked contrastive pre-training. ArXiv, abs/2211.11138,

2022. 3, 6, 7

- [51] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal LLMs. In Proceedings of the 41st International Conference on Machine Learning, pages 56704–56721, 2024. 2

- [52] Tianjun Zhang, Yi Zhang, Vibhav Vineet, Neel Joshi, and Xin Wang. Controllable text-to-image generation with gpt-

4. ArXiv, abs/2305.18583, 2023. 3

- [53] Xinchen Zhang, Ling Yang, Yaqi Cai, Zhaochen Yu, Kaini Wang, Jiake Xie, Ye Tian, Minkai Xu, Yong Tang, Yujiu Yang, and Bin Cui. Realcompo: Balancing realism and compositionality improves text-to-image diffusion models. ArXiv, abs/2402.12908, 2024. 3

- [54] Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. ArXiv, abs/2410.07171, 2024. 2

- [55] Yangkang Zhang, Chenye Meng, Zejian Li, Pei Chen, Guang Yang, Changyuan Yang, and Lingyun Sun. Learning object consistency and interaction in image generation from scene graphs. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23, pages 1731–1739, 2023. 3

- [56] Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc++: Advanced multi-instance generation controller for image synthesis. ArXiv, abs/2407.02329, 2024. 12

- [57] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6818–6828, 2024. 12

- [58] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. ArXiv, abs/2410.12669, 2024. 12

- [59] Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, and Jinhui Xu. Shifted diffusion for textto-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10157–10166, 2023. 2

### Supplementary Material S1. Details of SDXL-SG

Text-to-image (T2I) generation can produce highly detailed results. However, when the given text describes a relatively complex scene (e.g., an image containing multiple objects or multiple relationships between objects), the output of T2I models often falls short of expectations. Through experiments, we found that incorporating scene graph (SG) information during the image generation can significantly improve the model’s ability to generate compositional images. Based on this observation, we introduce a foundation model to alleviate restrictions on text-to-image generation.

Our model is based on the SDXL [31] architecture, integrating SG information into the generation process through graph neural networks (GNN) [37]. As shown in Fig. S6, we initialize each single object and triple of SG separately using the CLIP text encoder ET(·) to get their embeddings es and et. Specifically, for the triple embedding, it includes representations of objects eo

, relations er

, and object attributes eo

ij

k

represents the embedding of the k-th object in the SG, er

nam. Here, eo

k

represents the embedding of the j-th word in the i-th relation, as some relations in LAIONSG annotations may contain multiple words (e.g., “grown by”), and eo

ij

nam denotes the embedding of the m-th attribute word of the n-th object, as an object’s attributes may consist of multiple words (e.g., “tall wooden building”).

This structured SG input is then fed into the GNN. Objects serve as nodes, and relations act as edges. After that, we obtain a refined triple embedding er, which can be represented as:

###### er = GNN(ET(triplesg)) (S3)

We introduce an α factor to control the strength of the refined triple embedding, ensuring stable learning for the model. The optimized triple embedding is represented as:

###### et′ = et + αer (S4)

Finally, all triple embeddings are concatenated with singleobject embeddings to form the SG embedding esg, which is fed into the U-Net of SDXL for iterative noise prediction.

###### esg = f(sg) = concat(et′,es) (S5)

We employ SDXL [31] as the pretrained framework. The model learns SG knowledge at time step t by:

L = EE(x),sg,ϵ,t[∥ ϵ − ϵθ(zt,t,f(sg)) ∥22] (S6)

As introduction in Sec. S7.1, our training is conducted in the latent space to enhance efficiency. f(sg) encapsulates the SG embedding output from SG encoder of our baseline. The training process dynamically adjusts parameters of SG

encoder to minimize the gap between the predicted and added noise, which can reduce L, improving the model’s capability to handle compositional image generation.

Our architecture is designed to be lightweight and efficient. The generation time for 100 images at a resolution of 1024×1024 is measured. Our baseline model takes an average of 17.19 seconds per image, while the original SDXL model takes 16.70 seconds, both running on a single RTX 4090D GPU. Moreover, our SgEncoder model has a parameter count of 14.70M, which is only 0.23% of the approximately 6.6B parameters of the original SDXL, demonstrating its exceptional lightweight advantage. The inference time increases by less than 3%, and the parameter growth is negligible, making the additional computational cost almost insignificant. However, the improvement in output accuracy is substantial.

### S2. Discussion on Complex Scene Generation

Complex scene generation is a challenging task attracting attention from the community of image generation. MIGC (Multi-Instance Generation Controller) [57] and MIGC++ [56] adopts a strategy of generating individual instances separately and then integrating them, while incorporating multimodal descriptions for attributes (text and images) and localization (bounding boxes and masks). By incorporating appearance tokens and an instance semantic map, IFAdapter [47] enhances the fidelity of fine-grained features in multi-instance generation while ensuring spatial precision. 3DIS [58] decouples the multi-instance generation task into two stages: depth map generation and detail rendering. By combining depth-driven layout control with training-free fine-grained attribute rendering, it significantly enhances instance positioning accuracy and detail representation.

These works effectively address the challenges of multiinstance compositional generation. However, they primarily control the generated objects at the spatial level and fail to resolve inaccuracies in generating abstract semantic relationships between objects, such as “holding” or “riding”.

In contrast, [43] disentangles layouts and semantics from scene graphs, leveraging variational autoencoders and diffusion models to significantly enhance instance relationship modeling and fine-grained control in complex scene generation. This approach enhances the model’s understanding and representation of abstract semantics through the structured form of scene graphs. Nevertheless, it still faces generation bottlenecks due to dataset limitations.

Therefore, we propose the LAION-SG dataset to fundamentally address the challenges of complex scene generation at the data level. Simultaneously, we introduce a baseline model that enables simple and efficient generation of complex scenes based on scene graphs.

𝑒

𝑒

[Figure 66]

⊙

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

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

Single Object Single Embedding

SG Embedding

CLIP

𝞪

∗

+

𝑒

𝑒

|GNN 🔥<br><br>[Figure 86]|
|---|

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

❄

Scene Graph SG Triple

Triple Embedding

Refined Embedding

[Figure 103]

|𝑒<br><br>𝑒<br><br>𝑒<br><br>𝑒<br><br>𝑒<br><br>𝑒<br><br>𝑒 𝑒|
|---|

|SD|XL|
|---|---|

[Figure 104]

❄

Image

Figure S6. The architecture of our foundation model. Concatenation is indicated by ⊙ and multiplication by ∗.

LAION Image Original LAION Text Scene Graph

trial, users are presented with three images: the LAION image and two images generated from the original LAION caption and the scene graph respectively. Users are asked to choose the image from the latter two that best matched the content of the LAION image. We invite 10 participants, with a 1:1 gender ratio and ages ranging from 20 to 30. They come from diverse backgrounds, including computer science, design, and human-computer interaction (HCI).

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

The result of user study is shown in Fig. S7. A total of 63% of participants preferred the images generated from the scene graph, while only 37% chose those from the text prompt. This indicates that, compared to sequential text annotations, structured annotations have an overwhelming advantage in expressing image content.

Please select the image closest to the LAION image from those generated from the original LAION caption and the scene sraph.

Notification to Human Subjects. We present the notification to subjects to inform the collection and user of data before the experiments.

Annotation Original Caption Scene Graph User Preference 37% 63%

Dear volunteers, we would like to express our thankfulness for your support to our study. We study an image generation algorithm, which translates scene graphs to realistic images.

- Figure S7. The result of user study. Top: We present images generated from original captions and scene graphs to users and ask them to choose the one that better aligns with the content of the LAION image. Bottom: Across 100 validation image pairs, users showed a strong preference for the results generated from scene graphs.

All information about your participation in the study will appear in the study record. All information will be processed and stored according to the local law and policy on privacy. Your name will not appear in the final report. When referred to your data provided, only an individual number assigned to you is mentioned.

### S3. User Study

Beyond objective metrics, whether the results align with human cognition is also crucial. We conduct a user study to compare which annotation type generates images that better align with human perception.

We respect your decision whether you want to be a volunteer for the study. If you decide to participate in the study, you can sign this informed consent form.

We randomly select 100 text-sg-image triplets. In each

...

span over

[Figure 111]

|[Figure 112]<br><br>person<br><br>|headdress|
|---|
<br><br>adorn|
|---|

rainbow

], "relations": [

valley

{

"triple_id": 0, "item1": 1, "relation": "adorn", "item2": 0, "global_relation_id": 2118510

(a) (b)

},

... ]

- Figure S8. Example images from the LAION-Aesthetics dataset.

},

The use of users’ data was approved by the Institutional Review Board of the main authors’ affiliation.

And for the image in Fig. S8 (b), its highlighted portion corresponds to the following scene graph, with other parts omitted.

### S4. Examples of LAION-SG

{

Given an image, we employ a multimodal large language model, GPT-4o [30], to perform automated scene graph annotation. Our pipeline focuses on assigning distinct ids to different objects, identifying attributes for each object, labeling relations between objects, and adhering to other specified constraints. The annotations are strictly output in the designated format. Here, we provide two specific examples. For the image in Fig. S8 (a), the highlighted portion corresponds to the following scene graph, with other parts omitted.

"img_id": "483868", "name": "694108219422834467.jpg", "caption_ori": "Yosemite’s Rainbow.

Yosemite National Park, California.", "score": "6.544332504272461", "url": "https://photos.smugmug.com/..." "items": [

{

"item_id": 0, "label": "rainbow", "attributes": [

{

"img_id": "482063", "name": "minus83166520...", "caption_ori": "Page 90 of Girl...", "score": "6.720815181732178", "url": "https://stories...", "items": [

"colorful", "arc-shaped"

], "global_item_id": 3213781

},

... {

{

"item_id": 0, "label": "person", "attributes": [

"item_id": 4, "label": "valley", "attributes": [

"young", "female"

"green", "vast"

],

], "global_item_id": 3213785

- "global_item_id": 3201686

}, {

"item_id": 1, "label": "headdress", "attributes": [

"ornate", "white"

],

- "global_item_id": 3201687

},

...

], "relations": [

{

"triple_id": 0, "item1": 0, "relation": "span over", "item2": 4, "global_relation_id": 2126675

},

},

},

... ]

### S5. Details of Accuracy Metrics

We leverage SG-IoU, Entity-IoU, and Relation-IoU [40] to measure the model’s ability to generate complex scenes. Specifically, we use GPT-4 to extract scene graph lists from the generated images, with each list consisting of triples in the form ⟨sn,rn,on⟩. From this SG list, we derive the Entity and Relation lists and calculate the intersection over union (IoU) between the derived lists and the real annotations. Higher scores indicate stronger model capability in generating complex scenes.

Furthermore, we propose SG-IoU+, Entity-IoU+, and Relation-IoU+ to evaluate the annotation accuracy. Detailedly, we first generate two images: one using the original LAION captions and the other using scene graph from LAION-SG. Then for the real image and the two generated images, we extract the lists of SGs, relations and entities from each image with GPT-4o again. Taking the lists of SGs as an example, the IoU scores is calculated between the list SG generated image and that of the real image. Also the IoU beween the caption generated and the real is calculated. This IoU evaluates the extent that the generated images and the real image are similar along the SG structure, thus reflecting the annotation accuracy. It is the hight the better. Such IoU is also calculated on the lists of relations and entities.

### S6. Discussion on Annotation

#### S6.1. Hallucinations of GPT-4o

In our annotation process, GPT-4o occasionally exhibits hallucination phenomena, generating information that does not actually exist. Through a random check of 100 annotation samples, we find that approximately 1% contain such issues. These issues typically manifest as annotations that do not strictly adhere to the image content but instead rely on semantic inference to incorrectly label objects that are not present. For example, in Fig. S9, the GT image only shows one visible earring, while the other earring is occluded. However, GPT-4o erroneously infer its presence based on semantic reasoning.

Although the limitations of current multimodal large models make it challenging to completely avoid such problems, the quality of the original LAION annotation of the GT image in Fig. S9 is relatively low, further hindering the generation of complex scenes. Nevertheless, our annotation process strives to ensure the accurate description of entities and relationships within images, thereby maintaining a high overall annotation quality.

Original LAION Caption Scene Graph Labeled Image

[Figure 113]

blonde adult female

worn by

worn by worn by

person

“564full grace kelly 3”

earring earring

sparkly garment sparkly

shiny pink

- Figure S9. GPT-4o occasionally exhibits hallucination phenomena, labeling objects that do not exist in the image. For example, in a case where the image shows only one earring, GPT-4o incorrectly labels a nonexistent second earring. However, despite these issues, the overall quality of our annotations still surpasses that of LAION’s original annotations.

[Figure 114]

person_0 be looking at person_1 …

person_0 person_1

[Figure 115]

|stick|
|---|

person

person hold stick …

(a) (b)

- Figure S10. GPT-4o occasionally makes errors during the annotation process. For example, in (a), GPT-4o misidentifies the relationship, incorrectly assuming that person 0 is looking at person 1, which in fact is wrong. In (b), GPT-4o misclassifies a small object, labeling the item held by the person as a stick, when it is actually an umbrella.

#### S6.2. Failure Examples

Through prompt engineering, we leverage GPT-4o to perform large-scale, high-quality scene graph annotations for images. While the majority of these annotations accurately describe the entities and their relationships in the images, a small number of errors still occur. In a random sample of 100 annotations, approximately 2% contain one mislabeled annotation, including inaccurate relationship descriptions (as shown in Fig. S10(a)) or entity recognition errors (as shown in Fig. S10(b)).

#### S6.3. Abstract Attribute

We follow the guidelines of previous work [20] and try to ensure that attributes in the scene graph are abstract adjectives rather than specific objects.

### S7. More Information of Foundation Model

#### S7.1. A Brief Introduction of Stable Diffusion XL

SDXL (Stable Diffusion XL) [31] is an advanced latent diffusion model (LDMs) [34] primarily designed for generating high-resolution images based on text prompts. It builds on the fundamentals of diffusion models [16] by utilizing a two-stage process: initially generating images from noise and then refining these images to enhance quality.

SDXL operates in a compressed latent space rather than the pixel space directly, using an autoencoder to encode an input image into a lower-dimensional latent space and then applying the diffusion process in this space. This approach is computationally efficient and enables the generation of high-quality, detailed images with fewer resources compared to pixel-based diffusion models [16]. The model’s architecture consists of an autoencoder and a UNet-based diffusion network that performs the denoising operations.

To interpret text prompts with high fidelity, SDXL integrates two text encoders (OpenCLIP ViT-bigG [18] and CLIP ViT-L [32]). These encoders convert the textual input into feature representations, which are then concatenated and used to condition the diffusion process, thereby allowing the model to follow text prompts more accurately.

The SDXL diffusion process is a series of denoising steps in which the model progressively reduces noise from an initial noise-filled image until a clear image is produced. This iterative process can be represented mathematically by

xt = √αt · x0 + √1 − αt · ϵ (S7)

Here xt is the noisy image at step t. αt is a noise decay factor for each time step. x0 represents the clean, noise-free image. And ϵ is random Gaussian noise added to the image at each step. Each step is controlled by a learned model, ϵθ, that predicts and subtracts noise from the image, allowing it to converge on a high-quality result as t → 0.

SDXL’s training objective is to minimize the mean squared error between the predicted noise and the actual noise added to the image. The conditional term is introduced through classifier-free guidance [15], a mechanism that combines conditional information with the noise predictions from unconditional generation. This enables the model to better follow prompt details when generating images.

Specifically, the conditional loss function in SDXL can be represented as

0),c,ϵ∼N(0,I),t[∥ ϵ − ϵθ(zt,t,τ(c)) ∥22], (S8)

L = EE(x

where E(x0) and zt is latent representations of the original image and its noisy version at timestep t, c and τ(c) is the input condition and its latent embedding, and ϵθ(zt,t,τ(c)) represents the model’s noise prediction under

condition c. Additionally, the conditional term c includes spatial conditions like size and crop settings, enabling the model to adapt to various resolutions and framing needs. By minimizing this error, SDXL learns how to progressively remove noise and refine images accurately across various levels of initial noise.

To enhance the visual quality of generated images, SDXL includes a refinement model that operates in the latent space. This model further refines the output using SDEdit [29], an image-to-image process where noise is temporarily reintroduced and then denoised to improve quality.

#### S7.2. Additional Results

Successful and Failure Examples. We provide additional experimental results. Fig. S11 presents more successful cases, while Fig. S12 shows some failure cases, including object misalignment, incorrect object shape generation, and errors in object appearance generation.

Applications in Image Editing. We also offer a straightforward image editing application: by editing attributes, objects, or relationships in the scene graph, corresponding images can be generated. Fig. S13 shows the results of this approach.

### S8. Limitation

We summarize statistics on the types of objects annotated in the LAION-Aesthetics [38] and LAION-SG datasets. Among 10,000 samples, LAION-Aesthetics contains 12,263 distinct object types, which reduces to 5,811 after excluding proper nouns. In comparison, LAION-SG includes 1,429 types, all of which are common words without any proper nouns. This difference reflects a limitation of LAION-SG, as its vocabulary distribution is relatively less extensive. Furthermore, since LAION-SG focuses on scene graph that describe specific content within images, it is less sensitive to abstract cues such as historical context or stylistic elements. Integrating these control factors into the scene graph-to-image process remains a promising direction for future research.

Compared to manual annotation workflows, automated method significantly improves efficiency and reduces costs. However, it slightly lacks the precision achievable through human annotation, as discussed in Sec. S6.1 and Sec. S6.2. This limitation, however, is an inherent aspect of automated processes.

### S9. Social Impact

Scene graph to image generation holds great potential to benefit diverse fields, from content creation and education to virtual reality and simulation. By enabling the generation of realistic images from structured descriptions, this

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

- Figure S11. Additional results of successful examples.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

- Figure S12. Additional results of failure examples.

technology democratizes creative processes, allowing individuals with limited artistic skills to visualize complex ideas efficiently. Moreover, it can facilitate accessibility for users with disabilities, providing new ways to interact with visual content.

However, the technology also poses challenges, such as potential misuse for generating misleading, harmful content and negative bias. To mitigate these risks, the dataset and methods proposed in this work prioritize ethical considerations, including content moderation and bias reduction.

###### Origin Image Edit Attribute Edit Object Edit Relation

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

green grass person riding horse

yellow grass person riding horse

green grass person riding motorcycle

green grass person walking to horse

Figure S13. Additional results of applications in image editing: achieved by editing attributes, objects, or relationships in the scene graph.

Future research and collaboration across disciplines are essential to ensure that such technologies align with societal values while maximizing their positive impact.

