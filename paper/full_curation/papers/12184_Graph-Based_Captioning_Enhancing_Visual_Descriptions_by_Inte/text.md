## Graph-Based Captioning: Enhancing Visual Descriptions by Interconnecting Region Captions

# arXiv:2407.06723v2[cs.CV]26Feb2025

Yu-Guan Hsieh1* Cheng-Yu Hsieh2* Shih-Ying Yeh3*

yuguan@spellbrush.com cydhsieh@cs.washington.edu kblueleaf@gapp.nthu.edu.tw

Louis Béthune4 Hadi Pouransari4 Pavan Kumar Anasosalu Vasu4

{l_bethune, mpouransari, panasosaluvasu}@apple.com

Chun-Liang Li4 Ranjay Krishna2* Oncel Tuzel4 Marco Cuturi4

ranjay@cs.washington.edu, {chunliang_li, otuzel, cuturi}@apple.com

1 Spellbrush 2 University of Washington 3 National Tsing Hua University 4 Apple Code: https://github.com/apple/ml-gbc

Dataset: https://huggingface.co/graph-based-captions

#### Abstract

Humans describe complex scenes with compositionality, using simple text descriptions enriched with links and relationships. While vision-language research has aimed to develop models with compositional understanding capabilities, this is not reflected yet in existing datasets which, for the most part, still use plain text to describe images. In this work, we propose a new annotation strategy, graph-based captioning (GBC) that describes an image using a labeled graph structure, with nodes of various types. The nodes in GBC are created through a two-stage process: first, identifying and describing entity nodes; second, linking these nodes by highlighting compositions and relations among them. Since all GBC nodes hold plain text descriptions, GBC retains the flexibility found in natural language, but can also encode hierarchical information in its edges. We demonstrate that GBC can be produced automatically, using off-the-shelf multimodal LLMs and object detection models, by building a new dataset GBC10M that gathers GBC annotations for about 10M images of the CC12M dataset. Through CLIP training on GBC10M, we show that leveraging GBC nodes’ annotations—particularly those in composition and relation nodes—significantly boosts the model’s performance across various benchmarks compared to when other annotations are used. To further explore the opportunities provided by GBC, we also investigate the use of GBC as middleware

*Work done at Apple.

for text-to-image generation, and show the extra benefits of incorporating the graph structure in this task.

#### 1. Introduction

The availability of huge paired image/caption datasets has revolutionized our ability to develop advanced multimodal models, enabling a range of tasks such as efficient textto-image synthesis, text-guided image manipulation, and fine-grained image understanding through multimodal large language models [2, 6, 49, 56, 65, 68]. The quality and granularity of these datasets plays, therefore, a crucial role. While quality can be addressed by filtering out data [22, 30, 71] or with a simple recaptioning strategy [19, 21, 48, 60], there is ample interest in the community to provide more detailed, fine-grained information for each image [4, 9, 20, 56]. To obtain better annotations, we draw inspiration from compositionality, a fundamental characteristic of human perception that is reflected in the natural language used to describe our surroundings [5, 13, 15, 27, 38, 39]. Compositionality plays an especially important role when examining larger images found in the wild, which have a rich coarse-to-fine, hierarchical structure, commonly represented by a scene graph [40]. While scene graphs have been successfully applied to image retrieval [40, 72], generation [23, 59], and pre-training [35, 37], the scale of scene graph dataset is typically small. For instance, Visual Genome [45] only contains around 100k images.

###### Image Captions

###### Captions for the Node "Snowy Tree 1"

- • [Detailed] The image captures a serene winter scene featuring a tall brick water tower as its central subject. The tower stands majestically against a backdrop of a clear blue sky lightly speckled with clouds ... (incomplete)
- • [Short] A water tower stands tall against a sky, surrounded by snowy trees.

Captions for the Node "Snowy Trees"

- • [Composition] Snowy tree 2, which is located in the top left corner, appears to be slightly larger than snowy tree 1, which is situated in the bottom right corner.
- • The snowy trees are a striking contrast against the clear sky, creating a serene winter scene

- • A snowy tree stands prominently in the foreground, its branches heavily laden with white snow. The tree's silhouette contrasts sharply against the overcast sky, creating a striking visual effect.

[Figure 1]

[Figure 2]

Captions for the Node "Sky / Water Tower / Snowy Trees"

- • The sky forms a backdrop behind both the water tower and the snowy trees.

- Figure 1: An illustration of our proposed graph-based captions. The image node, entity nodes, composition nodes, and relation nodes are respectively colored in red, blue, green, and yellow. The color texts in the captions correspond to the labels of the outgoing edges, which are summarized as node labels in the figure. More examples are provided in Appendix C.3.

Contributions. To overcome the limitations of existing datasets and annotation formats that either struggle to represent the hierarchical nature of scenes or are of small size and lack flexibility in their description, this paper makes a series of contributions as summarized below.

###### 1. We propose graph-based captioning (GBC), a new

vision-language data format that captions images with a graph-based structure akin to scene graphs while retaining the flexibility and intuitiveness of plain text description. GBC contains four types of nodes: (1) an image node with captions of the entire image, (2) entity nodes that contain descriptions of individual objects, (3) composition nodes that link objects in the images of the same type, and (4) relation nodes that describe the spatial (“the tree is to the left of the tower”) or semantic (“The branch is covered in snow”) relationships between objects of different types. Importantly, encoding non-hierarchical relations as nodes rather than edges enables us to capture relationships involving more than two objects (§ 3.1).

2. We design a workflow to produce GBC annotations at scale. Our approach combines a multimodal large lan-

guage model (MLLM) with an open-vocabulary detection model. Initially, the MLLM generates both short and detailed captions for the entire image, which are used to identify entities. The detection model is then applied to locate bounding boxes for each identified entity. This process is recursively applied to create a GBC for each detected proposal. Finally, the MLLM is prompted to generate composition and relation captions that link multiple entity nodes together (§ 3.2).

###### 3. We create large-scale GBC dataset containing 10

million images with ≈ 534 words per image using the aforementioned workflow. While ours is the first visionlanguage dataset that contains structured captions, a few recent datasets contain dense annotations, and only [85] has a scale that is similar to ours. Our dataset is released under the CC BY-NC 4.0 license (§ 3.3).

###### 4. We demonstrate the benefit of GBC via CLIP train-

ing experiments. Concretely, we show that the diversity of captions found in GBC nodes improves CLIP model performance across image-to-text retrieval, text-to-image retrieval, compositionality, and semantic segmentation tasks, while retaining comparative performance on zero-shot ImageNet

classification. Remarkably, we observe that composition and relation nodes, which can only be obtained through the GBC workflow, boost performance. Moreover, we perform ablation on the influence of annotation format on retrieval performance using a set-aside test set from GBC. In this case, we see that an architecture that is tailored to the GBC format provides comparable or even better performance than that obtained when describing an image with detailed captions, suggesting that GBC can be a promising alternative to traditional image captioning formats (§ 4).

5. We demonstrate the benefit of GBC as middleware for text-to-image generation. By breaking down the textto-image generation process into the subtasks of text-toGBC and GBC-to-image, we provide users with a powerful intermediary for fine-grained image manipulation. The textto-GBC subtask is handled by a lightweight language model, while the GBC-to-image subtask can be completed through a training-free approach. Importantly, we show that the inclusion of graph information significantly enhances the performance of a baseline method that otherwise performs poorly when only bounding box information is used (§ 5).

#### 2. Related works

In this section, we discuss related works on vision-language datasets. We refer the readers to Appendix A for works that are specific to CLIP [68] and text-to-image models.

Vision-language datasets. First vision-language datasets were manually built using human annotations, such

- as Flickr30k [93], COCO [55] and Visual Genome [45]. This yielded annotations of high quality, but unfortunately of short length, and in limited amounts (with no dataset containing more than 130k images). Several studies have then demonstrated the benefits of using larger scale datasets obtained by crawling the web, such as YFCC100M [78], RedCaps [17], or Wikipedia-based image-text dataset (WIT) [75]. The quality of these data became a concern when it was noticed that in some situations the caption was only loosely related (or not related at all) with the image, which can be detrimental to the overall performance [70]. This motivated researchers to use automatic filtering procedures to select higher-quality data samples, like in Localized Narratives [66] or Conceptual Captions (CC3M) [74], and its successor CC12M [8]. These efforts have reached billion scale with LAION-5B [71], and LAION-CAT [67]. In a similar vein, Meta-CLIP [89] reproduces the processing of the seminal CLIP paper [68] on a subset of the Common Crawl dataset, SemDeDup [1] relies on the embeddings provided by foundation models to filter data and remove duplicates, while DFN [22] uses filtering networks trained on high quality data to extract subsets of Common Crawl.

VL datasets with dense captioning. It was noticed

recently that using entirely generated captions from raw images, as in DAC [19] and AS-1B [85], could improve results over filtering approaches. These datasets are characterized by their long and detailed captions that describe every element within a scene. Complementing these efforts, Urbanek et al. [80] introduced DCI, a dataset featuring similarly dense annotations but curated by humans and on a smaller scale. Alternatively, DOCCI [62] focuses on a set of only 15k high quality, high resolution, paired image-captions, manually selected and annotated by one of the authors, with typical caption length of more than 135 words. In the ImageInWords [28] dataset, captions are iteratively improved by humans, on top of previously human or machine annotated captions, yielding 9K densely captioned images.

#### 3. Improving image annotations with GBC

We introduce in this section our new captioning format to represent an image, explain how we can use any off-theshelf multimodal large language model (MLLM) and openvocabulary detection model to obtain such captions, and briefly describe the two datasets GBC1M, and GBC10M that we construct following the proposed workflow. Additional details about the data preparation process and the datasets can be found in the Appendices B and C.

##### 3.1. Representing an image with GBC

To encode the structured information contained in an image, we propose to represent each image as a directed acyclic graph (DAG), denoted as G = (V,E). Each node of the graph v ∈ V is associated with a bounding box. Starting with the root node, which corresponds to the entire image (image node), other nodes can either hold a set of objects (composition node and relation node), or a single object in the image (entity node). Moreover, to benefit from the expressive power of natural language descriptions and to ensure smooth integration of our annotations into the existing ecosystems of methods that rely primarily on image-text pairs, we label each node v with a set of captions Cv = {C1,...,Cnv}.

The edges, on the other hand, are used to encode the hierarchy between the nodes. More specifically, there is an edge e ∈ E from u to v only if the content associated to v is part of the content associated to u. This relation is also reflected by the edge label Le which should appear in the captions of the source node u and be able to represent the object(s) associated to the target node.1

An examplar GBC, generated automatically through our

1Ideally, we would also like to distinguish between multiple appearances of the same text in a caption. However, this is not explicitly handled by our current dataset construction workflow so we omit it here.

[Figure 3]

- Figure 2: Our image annotation process involves four types of queries that are performed in two separate stages, with the detection model serves to single out the regions that are used for different queries.

workflow is provided in Figure 1. It should be noted that the only manual addition in that graph comes from the "title label" of each node, which is obtained by taking the union of labels found in its incoming edges. Compared to the standard scene graph annotation, the use of node captions provides flexibility to describe complex concepts, while the underlying graph still captures the inherent structure of the image. Our dataset, whose construction is detailed in Sec. 3.2 next, includes several different types of captions tailored to the structure of the DAG. At the root image node, we provide both detailed and short captions to cater to varying levels of granularity. Captions at composition nodes and relation nodes explicitly describe the arrangement and interaction of multiple objects, while the captions at the entity nodes provide detailed description of a single object.

##### 3.2. GBC dataset construction workflow

We show how to produce GBC annotations automatically, using any pre-trained MLLM and open-vocabulary detection model. This results in a workflow that is comparable, in compute time and complexity, to that of other widespread recaptioning approaches. At a high level, we use a MLLM model to provide captions and identify potential entity nodes, followed by a detection model to provide bounding box coordinates for these entities.

Data annotation. Our overall process to annotate a single image is shown in Figure 2. To account for the different types of nodes, we design four query templates as listed below:

• Image query: We ask the model to provide detailed caption for the image, identify prominent elements, and summarize the long caption with a concise one that contains all these elements. The identified elements are then passed to the detection model to obtain the bounding

boxes.

- • Entity query: For each bounding box, we crop out the region and ask the model whether a specific object appears in the cropped image. Moreover, we also ask the model to describe the object and identify prominent elements of the object when it is present. The identified elements are again passed to detection models for detection.
- • Composition query: In the case where multiple bounding boxes are returned for a single type of object, we ask the model to describe the composition of these objects with an annotated image.
- • Relation query: For image or entity nodes with more than two children, we ask the model to describe the relations between its children.

Provided that there is no guarantee that all the detected objects would end up as a node in the graph—consider the case where the MLLM says that the object is not present or just fails to reply in the correct format—we split the entire process into two stages, and we only perform composition queries and relation queries after discovering all the entity nodes. Finally, to improve efficiency and to reduce redundant information, we train two dedicated classifier on top of Jina Embeddings [32] to decide whether a piece of text is suitable for object detection and whether two texts can represent the same object in an image. The former is applied to every identified element while the later results in merging of nodes when a new query targets a region that has already been queried with similar texts.

##### 3.3. GBC1M and GBC10M

Following the process outlined in Sec. 3.2, we annotate the CC12M dataset [8] with graph-based captions using LLaVA-1.6 [56, 57] as the MLLM and Yolo-World [12] as the open-vocabulary detection model. Specifically, we

GBC1M GBC10M

# Images 1,013,592 10,138,757 # Vertices / Image 12.12 12.24 # Edges / Image 22.28 21.81 # Captions / Image 17.40 17.67 # Words / Image 593.14 533.98 Average Graph Diameter 4.55 4.41

- Table 1: Key statistics of the GBC1M and GBC10M datasets. We report number of images, average number of vertices, edges, captions, and words per image, and average graph diameter.

construct two sets of annotations: GBC1M for a subset of around 1M of images, with all the queries performed with the Yi-34B version of LLaVA-1.6, and GBC10M for a subset of around 10M of images, with LLaVA-1.6 Yi-34B for image and composition queries, and LLaVA-1.6 Mistral-7B for entity and relation queries.2

We provide statistics of the above two datasets in Tab. 1. We note that these two datasets have very similar per-image statistics, with the number of words being the only exception,

- as LLaVA-1.6 Yi-34B tends to provide longer descriptions than LLaVA-1.6 Mistral-7B. Moreover, our datasets use an average number of around 500 words to describe each image. This is comparable to other dataset with rich annotations such as DCI (1111 words/img) [80] and DOCCI (136 words/img) [62].

#### 4. CLIP training with GBC

We present in this section a comprehensive set of experiments to compare different image annotation schemes from a CLIP training perspective. We first show that compared to existing annotation schemes, GBC annotations can bring improvements on a range of benchmarks across classification, retrieval, and dense prediction task. Then, we demonstrate how GBC allows one to encode denser, more descriptive textual information to better represent images on retrieval tasks. Additional experimental details and further ablations are respectively provided in Appendices E and F.

##### 4.1. Annotation formats

We outline below the different types of image annotations that are considered in our experiments, each providing different opportunities to leverage information from the image.

Short caption. Each image is paired with a short caption, as in common image-text datasets.

2Our larger dataset does not cover the entire CC12M both because some images were no longer accessible at the time we accessed the images, and because we discard images for which the MLLM model’s reply to the image query does not comply with the prescribed format.

Long caption. One can improve image description using a longer caption. The long captions that we use in our experiments are of 110 words on average, as compared to short captions, of only 28 words on average. We extend the context length of text encoders in CLIP models from 77 to 512 for this setup.

Region captions. Alternatively, more captions can be provided for an image, especially those that describe a specific region of the image. While this format includes all region captions, it does not include the relational information between region descriptions found in GBC.

Graph-based captions. Finally, we consider the GBC format as proposed in Sec. 3.1. The GBC format includes region captions, but also provides additional information, stored in relation and composition nodes. With this in mind, we explore three different ways to leverage GBC annotations:

- • A direct way to leverage GBC is to treat captions for all nodes in the graph as positive texts for the image, i.e. as what we do for region captions. We refer to such method as GBC-captions.
- • Another strategy is to traverse from the root image node through the graph and concatenate the captions at the visited vertices into a single long caption. We then train a CLIP model with 512 context length in the standard fashion. We refer to this method as GBC-concat.
- • To fully benefit from the graph information, when available, we introduce additional cross-attention layers to leverage the graph topology (see Appendix D.1 for details). This allows us to encode the entire graph into a text embedding that also contains the information about the graph structure. We refer to this method as GBC-graph.

As GBC annotation encapsulates all existing short, long, and region caption formats, we are able to instantiate all the setups by using only a subset of annotation available in our curated GBC10M dataset. Specifically, taking only the short or detailed caption at the root image node creates the short and long caption setup, respectively. To mimic the region caption setup, we drop the relation and composition captions from GBC annotations. Additionally, to isolate the impact of using long captions and the additional captions uncovered by GBC, we omit long captions from both the region caption setup and the three GBC setups. By adapting GBC into these configurations, we ensure consistent text annotation quality across different methods.

##### 4.2. Experimental setup

We perform CLIP training on our GBC10M dataset, while leaving out 10,151 samples as the test set. Following common practice, we use the CLIP score computed by a pretrained CLIP model [22] to filter our training set, discarding the 5% of captions with the lowest scores for each type. In

|Annotation|Flickr-1k T2I I2T|MSCOCO-5k T2I I2T<br><br>|ImageNet<br><br>|SugarCrepe|ADE20K|
|---|---|---|---|---|---|
|CC12M|46.4 64.6|25.0 39.4|39.2|72.9<br><br>|41.7|
|Short Long Region<br><br>GBC-captions GBC-concat GBC-graph|56.3 73.2<br>56.4 75.2 58.3 76.6 60.6 79.3 56.1 76.0 58.0 76.9<br><br><br>|30.7 46.7<br>31.8 50.1 31.5 49.1 34.1 51.9 31.4 48.5 31.9 49.2<br><br><br>|38.8 39.6 38.5 40.8 39.0 38.4<br><br>|76.0 77.0 75.6 76.7 75.7 74.4<br><br>|42.0 42.8 43.5 45.0 42.1 43.8|

- Table 2: Comparative performance on various existing benchmarks when trained using different annotation schemes. For retrieval tasks we report Recall@1, and for ADE20K we report the mIOU. As a baseline, we also report performance of a model trained on the same set of images using original CC12M captions. The highest scores for each task are highlighted in bold, and the second-highest scores are underlined.

addition, we retain the original CC12M captions associated with each image. Specifically, in all setups, both the original caption and the short synthetic caption are used as positive texts for the image during training. This prevents the severe distribution shifts that could occur from using only long or region captions when evaluating on standard benchmarks.

Objective. To pair an image with multiple captions in training CLIP models, we adopt a multiple-positive contrastive loss in the spirit of LaCLIP [21] and DAC [19]. Briefly speaking, compared to standard CLIP objective, the multiple-positive loss sums over the loss on each positive captions of an image while all the captions from the images in the same batch are used in the normalization term.

Model and hyperparmeters. We use the standard CLIP ViT-B/16 model, with the only difference of longer context length of text encoder for long caption and GBCconcat, and a replacement of the vanilla transformer block by our dedicated attention block in text encoder for GBCgraph. We fix the global batch size (i.e. number of images in each batch) to 4,096 for all the methods. The models are trained for 45,000 steps with AdamW and cosine scheduler

- at a learning rate of 10−3. This roughly correspond to 20 epochs of training. We evaluate at the EMA checkpoint at epoch 10, as we observe that further training provides little to no improvement in performance across the benchmarks.

##### 4.3. Evaluations on existing benchmarks

We compare the CLIP models derived from different annotation schemes on an array of evaluation benchmarks, including: Flickr-1k [64] and MSCOCO-5k [55] for zeroshot retrieval, ImageNet [69] for zero-shot classification, SugarCrepe [36] for compositional understanding evaluation, and ADE20k [104] for semantic segmentation that measures models’ dense prediction performances. Table 2 illustrates our results, from which we draw the following two key insights.

|Annotation|Short T2I I2T|Long T2I I2T|GBC-captions T2I I2T<br><br>|GBC-concat T2I I2T|GBC-graph T2I I2T|
|---|---|---|---|---|---|
|Short Long Region<br><br>GBC-captions GBC-concat GBC-graph|85.8 86.2<br>86.4 87.5<br><br><br>85.3 86.1<br>86.8 87.6 86.1 86.5 84.8 85.7<br>|85.0 87.2 95.4 95.7 85.5 88.2 87.2 89.6 92.7 93.5 85.5 88.0|57.3 37.0 44.3 33.1 91.5 79.3 91.3 80.9 57.5 37.9 90.8 79.8|87.4 88.2 90.5 91.4<br><br>89.5 90.0<br>90.1 91.0 94.6 94.9 89.6 90.5<br>|- -<br>- -<br>- -<br>- -<br>- -<br><br><br>95.9 96.1<br><br>|

Table 3: Image and text retrieval performance on GBC test set when trained and evaluated using different types of annotations (Rows: models trained from different annotations; Columns: Evaluations on different annotations). For each trained model, we highlight the best evaluation performance in T2I and I2T retrievals in bold. For the GBC-captions column, we include all the captions from our graph by default except in cases where training is done solely on the region annotations. In this case, excluding relation and composition captions, as done during training, results in better performance.

GBC annotation leads to clear performance gains by encoding relational information. Table 2 demonstrates that training models with more detailed textual information, such as long captions or region captions, consistently enhances downstream performance, particularly in retrieval tasks and dense prediction. However, the most significant improvements are seen with GBC-captions, which augment traditional region captions with relational and compositional descriptions, yielding about a 5% recall increase for retrieval tasks and a 3% boost in mIOU in segmentation task compared to using only short captions. Given that the GBC workflow is uniquely positioned to provide such relational information, this demonstrates that GBC captures valuable insights not present in conventional captions.

How the captions are used matters. Compared to GBC-captions, the improvements achieved by GBC-concat and GBC-graph on these benchmarks are of a smaller margin. This indicates that the way GBC annotations is used significantly impacts performance. Specifically, this worse performance is likely due to a mismatch between training and evaluation. For instance, the graph information that would benefit GBC-graph most is not provided in any of these benchmarks. We address this discrepancy below.

##### 4.4. Evaluation on GBC test set

To assess the effectiveness of different annotation formats, we provide the model with these annotations at test time. Annotations that better describe the images should, ideally, result in better retrieval performance when they are used. Specifically, we use our own test set and consider performing retrieval with the various types of annotations presented in Sec. 4.1. Note, however, that when using region captions or GBC-captions, no single text embedding can naturally encompass all the relevant information. To address this limitation, we perform retrieval based on the average CLIP score between the image embedding and the text embeddings

of the provided captions in this setup. We report our results in Tab. 3, where the rows correspond to the annotations at training time and the columns correspond to the annotation

- at test time. Unsurprisingly, we see a strong tendency that when a model is trained with a certain annotation format, it performs the best when we use the same format for retrieval. Among the few exceptions, we note that models trained to pair with shorter captions may have better performance when concatenation of short captions is provided at test time. This leads us to the following two observations.

Denser textual information benefits retrieval. The table clearly shows that training with richer annotations—such as long captions, GBC-concat, or GBC-graph—enhances retrieval performance. This improvement suggests that these methods provide a more effective representation of the images. Specifically, GBC-graph yields the best performance, indicating that the proposed GBC format consists in a viable alternative to the commonly used detailed captions.

Simple augmentation during training does not allow to exploit additional information when available. Our observations from Sec. 4.3 show that treating all captions as independent positives yields the best performance on existing benchmarks. However, there is no evidence that this method could harness the richer information from multiple captions when they are provided together in test time. Indeed, whether we use average CLIP score or concatenation, the retrieval performance of these methods significantly lags behind those methods that are trained directly with captions that individually encompass rich information.

#### 5. Text-to-image generation with GBC

While current image generation models mostly rely on natural language instructions to create images, the inherent ambiguity in natural language can limit their ability to produce content that aligns precisely with the user’s intent. Additionally, natural languages may lack the expressive power needed to convey all the nuances present in an image. To overcome these challenges, a number of works have introduced middleware between text and image to enable more fine-grained control over the generated content [25, 54, 61, 77, 90, 92]. In this section, we complement this line of work by showing how the graph information in GBC can further enhance the image generation process. Specifically, we explain how GBC can be used as middleware for text-to-image generation by dividing the process into two distinct tasks: text-to-GBC and GBC-to-image. For simplicity, we discard all relation nodes and composition captions, and assume there is only one caption per node. Additional details and experimental results are reported in Appendices D, G and H.

Text-to-GBC. To generate GBC descriptions from a user-provided prompt, we encode GBC in natural languages

|User Input<br><br>Node #0 image (Image Node) type: image<br><br>is_leave: False desc: An illustration with beautiful scenery of galaxy faraway in the background. The sky is filled with stars and nebulae. In the middle of the image, there is a tall building with multiple levels illustrated by warm lights. parents: bbox: left: 0.0000000, top: 0.0000000, right: 1.0000000, bottom: 1.0000000|
|---|

|Node #1 sky type: entity is_leave: True desc: The sky in this image is a beautiful blend of colors, with hues of pink and orange... parents: #0(image: sky) bbox: left: 0.0000000, top: 0.0000000, right: 1.0000000, bottom: 0.7968206<br><br>Generated Nodes<br><br>Node #2 nebulae type: entity is_leave: True desc: The image showcases a vibrant nebula, characterized by its rich red hue... parents: #0(image: nebulae) bbox: left: 0.0002560, top: 0.0000000, right: 1.0000000, bottom: 0.7933800<br>Node #3 building type: entity is_leave: True desc: A tall building with numerous windows, set against a backdrop of a vibrant night sky... parents: #0(image: building) bbox: left: 0.1817661, top: 0.4052215, right: 0.8178599, bottom: 0.7993608<br>|
|---|

[Figure 4]

Figure 3: An example of generated graph from our 200M prompt generation model.

and train a small language model of 200M parameters to generate the entire GBC graph from the short prompt contained in the image node. We find that as we are focusing on a very specific task, this already gives satisfying results. We show an example of the generated graph in 3. The small size of the additional model ensures that we add minimum overhead to the entire image generation process.

GBC-to-image. Our image generation experiments are based on SDXL [65], a latent text-conditional diffusion model. We consider various methods that generate images using different subsets of information from GBC and demonstrate that providing graph information helps generate images that better align with user intent.

- 1. Text only: For this, we simply generate images by concatenating prompts from different nodes in a BFS order (same as GBC-concat for CLIP training).
- 2. Text and bounding boxes: Methods for incorporating bounding boxes in image generation include training-free alternatives such as MultiDiffusion-type approaches [3, 54, 90], BoxDiff [88], and DenseDiffusion [42], as well as training-based alternatives like GLIGEN [53]. In this work, we implement a simple training-free baseline that encodes each text prompt independently and manipulates cross-attention masks so that each image patch only attends to a prompt when its corresponding bounding box contains that image patch [18]. This approach is more efficient than previous methods but often underperforms, as the image generation process may disregard additional prompts even though we attend to them in cross-attention.
- 3. Text, bounding boxes, and graph: We find that the baseline approach presented in the previous bullet point can be drastically improved when graph information is available. For this, we implement the following modifi-

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

[Figure 18]

[Figure 19]

[Figure 20]

Figure 4: Images generated using GBC prompts with different algorithms. Some algorithms use only a strict subset of GBC information. We note that although more advanced methods for generating images from region prompts exist, our goal here is to highlight how incorporating additional graph information can enhance a simple, training-free approach that might otherwise perform poorly when only bounding box information is exploited. Image prompts are provided for the second example using IP adapter [91]. The method that only leverages prompts and graph does not work for the third example as the depth of the corresponding graph is greater than 1.

cations of the algorithm:

- • Improved cross-attention mask: The graph structure introduces a natural hierarchy among prompts. To ensure that image patches focus on the most finegrained descriptions, we design the mask so that a patch attends to a prompt only if its bounding box contains the patch and none of its descendant prompts do so.
- • Confining each node’s content to its bounding box: When a bounding box is provided for a node associated with a label L (i.e., L is the label of one of the edges pointing to the node), we want to restrict L from appearing in regions outside the bounding box. This is achieved by masking out L from the parent prompt that points to the node using the edge labeled L in cross-attention and by adding L to negative prompts for patches outside the bounding box in question.
- • Encoding prompts with parent context (optional): In some cases, encoding each prompt with additional contextual information can be beneficial. To achieve this, we concatenate each prompt with its parent prompt before passing it through the text encoder and then mask out the parent prompt in cross-attention to reduce its influence.

4. Text and graph: We note that the algorithm described

above can be extended to support arbitrary segmentation masks for each node. Additionally, segmentation masks for each object in the description can be derived from cross-attention scores, provided that these objects are accurately generated by the model. Using this approach, we implement a method similar to the one proposed by Ge et al. [29] that generates images based solely on text and graph information. We find that this method performs well for star graphs (i.e., where every entity prompt has global prompt as its parent), but can be hardly generalized to more complex graph topologies.

We compare the four approaches via qualitative examples in Figure 4. As expected, the method incorporating all the information from GBC performs the best. In particular, we can generate a red banana and a yellow apple without encountering attribute binding issues when we leverage the underlying graph structure (first and forth image). However, such attribute binding issues can arise when the global and individual prompts are encoded together (third image) or when they are attended to simultaneously in cross-attention (second image). On the contrary, in the cat and dog example, encoding individual prompts with their parent prompts proves beneficial, resulting in a higher success rate for generating both animals in the image. Notably, even with the base global prompt, SDXL struggles to generate both a cat and a dog simultaneously. The success rate of the approach

that relies solely on text and graph is hence affected, as it assumes the base model can generate all key objects described. In the same example, we demonstrate that image prompts can be provided for individual nodes if the underlying model can handle them. Finally, our last example illustrates that our approach can extend beyond star graphs, generating images based on more complex GBC prompts, confirming the advantages of incorporating graph structures in GBC.

#### 6. Conclusion

We propose graph-based captioning (GBC) as a new image-text annotation format and curated the GBC1M and GBC10M datasets using modern MLLMs and detection models. Building on CLIP training, we introduce several baseline methods to leverage these datasets, demonstrating that models trained with GBC annotations achieve superior performance across various benchmarks compared to those trained with traditional annotation formats. For text-to-image generation, we show that using GBC as middleware enables finer control in image generation due to the rich information provided by the graph structure. In summary, our work demonstrates that GBC provides a versatile and powerful foundation for developing more advanced vision-language models across various applications, enhancing both imagetext representation learning and text-to-image generation.

#### References

- [1] Amro Kamal Mohamed Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Dataefficient learning at web-scale through semantic deduplication. In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2023. 3
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022. 1
- [3] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. In International Conference on Learning Representations, 2023. 7, 16
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions, 2023. 1, 16
- [5] Léon Bottou. From machine learning to machine reasoning. Machine learning, 94(2):133–149, 2014. 1
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 1
- [7] Likun Cai, Zhi Zhang, Yi Zhu, Li Zhang, Mu Li, and Xiangyang Xue. Bigdetection: A large-scale benchmark for improved object detector pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4777–4787, 2022. 28
- [8] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR,

2021. 3, 4

- [9] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 1
- [10] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 45
- [11] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5343–5353, 2024. 16
- [12] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-

- vocabulary object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2024. 4
- [13] Noam Chomsky and Morris Halle. Some controversial questions in phonological theory. Journal of linguistics, 1(2): 97–138, 1965. 1
- [14] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open- mmlab/ mmsegmentation, 2020. 45
- [15] MJ Cresswell. Logics and Languages. Methuen, 1973. 1
- [16] Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and Ashok Cutkosky. The road less scheduled. arXiv preprint arXiv:2405.15682,

2024. 50

- [17] Karan Desai, Gaurav Kaul, Zubin Trivadi Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. 3
- [18] Regional Prompter Developpers. Regional prompter github page. https://github.com/hako-mikan/sdwebui-regional-prompter, 2023. Accessed: 202411-01. 7
- [19] Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, and Leonid Karlinsky. Dense and aligned captions (DAC) promote compositional reasoning in VL models. In Neural Information Processing Systems, 2023. 1, 3, 6, 16, 28, 41, 47
- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 1
- [21] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving CLIP training with language rewrites. In Neural Information Processing Systems, 2023. 1, 6, 16, 41, 47
- [22] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In International Conference on Learning Representations, 2024. 1, 3, 5
- [23] Azade Farshad, Yousef Yeganeh, Yu Chi, Chengzhi Shen, Böjrn Ommer, and Nassir Navab. Scenegenie: Scene graph guided diffusion models for image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 88–98, 2023. 1, 16
- [24] Pedro F Felzenszwalb and Daniel P Huttenlocher. Efficient graph-based image segmentation. International journal of computer vision, 59:167–181, 2004. 42
- [25] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4744–4753, 2024. 7, 16

- [26] Enrico Fini, Pietro Astolfi, Adriana Romero-Soriano, Jakob Verbeek, and Michal Drozdzal. Improved baselines for vision-language pre-training. Transactions on Machine Learning Research, 2023. Featured Certification. 16
- [27] Jerry A Fodor and Zenon W Pylyshyn. Connectionism and cognitive architecture: A critical analysis. Cognition, 28 (1-2):3–71, 1988. 1
- [28] Roopal Garg, Andrea Burns, Burcu Karagol Ayan, Yonatan Bitton, Ceslee Montgomery, Yasumasa Onoe, Andrew Bunner, Ranjay Krishna, Jason Baldridge, and Radu Soricut. Imageinwords: Unlocking hyper-detailed image descriptions. arXiv preprint arXiv:2405.02793, 2024. 3
- [29] Songwei Ge, Taesung Park, Jun-Yan Zhu, and Jia-Bin Huang. Expressive text-to-image generation with rich text. In IEEE International Conference on Computer Vision (ICCV), 2023. 8
- [30] Sachin Goyal, Pratyush Maini, Zachary C Lipton, Aditi Raghunathan, and J Zico Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. arXiv preprint arXiv:2404.07177, 2024. 1
- [31] Jiatao Gu, Ying Shen, Shuangfei Zhai, Yizhe Zhang, Navdeep Jaitly, and Joshua M Susskind. Kaleido diffusion: Improving conditional diffusion models with autoregressive latent modeling. In Advances in neural information processing systems, 2024. 16, 51
- [32] Michael Günther, Louis Milliken, Jonathan Geuter, Georgios Mastrapas, Bo Wang, and Han Xiao. Jina embeddings: A novel set of high-performance sentence embedding models. arXiv preprint arXiv:2307.11224, 2023. 4
- [33] Hasan Abed Al Kader Hammoud, Hani Itani, Fabio Pizzati, Philip Torr, Adel Bibi, and Bernard Ghanem. Synthclip: Are we ready for a fully synthetic clip training? arXiv preprint arXiv:2402.01832, 2024. 16
- [34] Laura Hanu and Unitary team. Detoxify. Github. https://github.com/unitaryai/detoxify, 2020. 26
- [35] Roei Herzig, Alon Mendelson, Leonid Karlinsky, Assaf Arbelle, Rogerio Feris, Trevor Darrell, and Amir Globerson. Incorporating structured representations into pretrained vision \& language models using scene graphs. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 1, 16
- [36] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in Neural Information Processing Systems, 36, 2024. 6, 44
- [37] Yufeng Huang, Jiji Tang, Zhuo Chen, Rongsheng Zhang, Xinfeng Zhang, Weijie Chen, Zeng Zhao, Zhou Zhao, Tangjie Lv, Zhipeng Hu, et al. Structure-clip: Towards scene graph knowledge to enhance multi-modal structured representations. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2417–2425, 2024. 1, 16
- [38] Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise? Journal of Artificial Intelligence Research, 67:757–795, 2020. 1
- [39] Theo MV Janssen and Barbara H Partee. Compositionality. In Handbook of logic and language, pages 417–473. Elsevier, 1997. 1

- [40] Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. Image retrieval using scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3668–3678, 2015. 1
- [41] Justin Johnson, Agrim Gupta, and Li Fei-Fei. Image generation from scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1219–1228, 2018. 16
- [42] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7701–7711, 2023. 7, 16
- [43] Michael Kirchhof, James Thornton, Pierre Ablin, Louis Béthune, Eugene Ndiaye, and Marco Cuturi. Sparse repellency for shielded generation in text-to-image diffusion models. arXiv preprint arXiv:2410.06025, 2024. 51
- [44] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 28
- [45] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 1, 3, 28, 31, 34
- [46] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020. 28, 31
- [47] Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 2024. 51
- [48] Zhengfeng Lai, Haotian Zhang, Bowen Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, and Meng Cao. Veclip: Improving clip training via visual-enriched captions,

2024. 1, 16

- [49] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 1
- [50] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 16
- [51] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu

- Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 16
- [52] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390– 23400, 2023. 16
- [53] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 7, 16
- [54] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. LLMgrounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. Transactions on Machine Learning Research, 2024. 7, 16
- [55] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 3, 6, 28, 44
- [56] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2023. 1, 4
- [57] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 4
- [58] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 50
- [59] Rameshwar Mishra and AV Subramanyam. Scene graph to image synthesis: Integrating clip guidance with graph conditioning in diffusion models. arXiv preprint arXiv:2401.14111, 2024. 1
- [60] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving multimodal datasets with image captioning. In Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 1, 16
- [61] Weili Nie, Sifei Liu, Morteza Mardani, Chao Liu, Benjamin Eckart, and Arash Vahdat. Compositional text-to-image generation with dense blob representations. In International Conference on Machine Learning, 2024. 7, 16
- [62] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. Docci: Descriptions of connected and contrasting images. arXiv preprint arXiv:2404.19753, 2024. 3, 5
- [63] Nobuyuki Otsu et al. A threshold selection method from gray-level histograms. Automatica, 11(285-296):23–27,

1975. 42

- [64] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik.

- Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 6, 44
- [65] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 1, 7
- [66] Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part V 16, pages 647–664. Springer,

2020. 3

- [67] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distillation, and hard negatives for vision-language pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6967–6977, 2023. 3
- [68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 3, 16, 41
- [69] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 6, 44
- [70] Shibani Santurkar, Yann Dubois, Rohan Taori, Percy Liang, and Tatsunori Hashimoto. Is a caption worth a thousand images? A study on representation learning. In The Eleventh International Conference on Learning Representations, 2022. 3
- [71] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 1, 3
- [72] Sebastian Schuster, Ranjay Krishna, Angel Chang, Li FeiFei, and Christopher D Manning. Generating semantically precise scene graphs from textual descriptions for improved image retrieval. In Proceedings of the fourth workshop on vision and language, pages 70–80, 2015. 1
- [73] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019. 28
- [74] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, im-

- age alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 3
- [75] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2443–2449, 2021. 3
- [76] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 16
- [77] Omost Team. Omost github page. https://github. com/lllyasviel/Omost, 2024. Accessed: 2024-11-

01. 7, 16, 51

- [78] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016. 3
- [79] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 16
- [80] Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana Romero-Soriano. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. arXiv preprint arXiv:2312.08578, 2023. 3, 5, 45
- [81] Stefan Van der Walt, Johannes L Schönberger, Juan NunezIglesias, François Boulogne, Joshua D Warner, Neil Yager, Emmanuelle Gouillart, and Tony Yu. scikit-image: image processing in python. PeerJ, 2:e453, 2014. 42
- [82] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Fastvit: A fast hybrid vision transformer using structural reparameterization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 45
- [83] Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, and Oncel Tuzel. Clip with quality captions: A strong pretraining for vision tasks. arXiv preprint arXiv:2405.08911, 2024. 45
- [84] Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, and Oncel Tuzel. Mobileclip: Fast image-text models through multi-modal reinforced training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 16
- [85] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In The Twelfth International Conference on Learning Representations, 2023. 2, 3, 28
- [86] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.

- Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 17
- [87] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In European conference on computer vision, 2018. 45
- [88] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7452–7461, 2023. 7, 16
- [89] Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In The Twelfth International Conference on Learning Representations, 2023. 3
- [90] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In International Conference on Machine Learning, 2024. 7, 16
- [91] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for textto-image diffusion models. arXiv preprint arxiv:2308.06721,

- 2023. 8

[92] Shih-Ying Yeh, Sang-Hyun Park, Giyeong Oh, Min Song, and Youngjae Yu. Tipo: Text to image with text presampling for prompt optimization. arXiv preprint arXiv:2411.08127,

- 2024. 7, 16, 50, 51

- [93] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 3
- [94] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 16
- [95] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. In International Conference on Machine Learning, pages 25994–26009. PMLR, 2022. 16
- [96] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 16
- [97] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. arXiv preprint arXiv:2403.15378, 2024. 16
- [98] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 16
- [99] Bo Zhao, Lili Meng, Weidong Yin, and Leonid Sigal. Image generation from layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8584–8593, 2019. 16

- [100] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Unicontrolnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 2023. 16
- [101] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22490–22499, 2023. 16
- [102] Kecheng Zheng, Yifei Zhang, Wei Wu, Fan Lu, Shuailei Ma, Xin Jin, Wei Chen, and Yujun Shen. Dreamlip: Languageimage pre-training with long captions. arXiv preprint arXiv:2403.17007, 2024. 16
- [103] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based languageimage pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16793–16803, 2022. 16
- [104] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302–321, 2019. 6, 44, 45

### Appendix

#### Table of Contents

###### A. Related works, limitations, and societal impact 16

- A.1. Additional related works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2. Limitations and perspectives . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.3. Societal impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

###### B. Dataset construction 17

- B.1. Query templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2. Text classifiers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.3. Other details for data annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.4. Computation cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

###### C. Dataset information 26

- C.1. Data release and licensing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.2. Dataset statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.3. Examples from GBC10M . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

###### D. Algorithm details 41

- D.1. Structure-aware hierarchical attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- D.2. Multi-positive contrastive loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- D.3. Sampling algorithm for GBC-to-image generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42

###### E. Experimental details for CLIP training 43

- E.1. Data filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- E.2. Dynamic batch size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- E.3. Hyperparameters for CLIP training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- E.4. Computation cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- E.5. Evaluation details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44

###### F. Additional results and experiments for CLIP training 45

- F.1. Retrieval with multiple captions using maximum CLIP score . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- F.2. Retrieval with long captions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- F.3. Matching compute resource for training with short captions . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- F.4. The importance of multi-positive contrastive loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- F.5. Impact of caption type on CLIP training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- F.6. Impact of the underlying graph on retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- F.7. Evaluating at non-EMA checkpoints . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48

###### G. Experimental details for text-to-image generation 50

- G.1. Text-to-GBC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- G.2. GBC-to-image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

###### H. Additional results for text-to-image generation 51

- H.1. GBC-to-image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- H.2. Text-to-image with GBC as middleware . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51

###### I. Image attributions 59

#### A. Related works, limitations, and societal impact

This appendix delves deeper into the broader context of our study, examines additional related works, discusses the limitations of our methodologies, and explores its potential societal impacts.

- A.1. Additional related works In this section, we include works related to CLIP training and text-to-image generation.

CLIP with recaptioning. CLIP [68] is a seminal vision-language model that utilizes text and image encoders to generate joint latent representations. While there is an extensive body of literature on CLIP training—ranging from modifications in the objective [50, 94], data augmentation techniques [26, 52], to training procedures [76, 96]—it is impossible to cover all developments comprehensively here. Among these, particularly relevant to our work is the recent trend that highlights the benefits of enhancing caption quality through dedicated models. For instance, VeCLIP [48] enriches image alt-text with outputs from LLaVA, while similar recaptioning strategies have also been explored by Doveh et al. [19], Nguyen et al. [60], and Vasu et al. [84]. On the other hand, LaCLIP [21] employs LLaMA [79] to rewrite captions. Going further, SynthCLIP [33] leverages a dataset with entirely generated captions and images for CLIP training.

CLIP with additional annotations. There has been a plethora of research on training CLIP models with diverse annotations such as long captions, region captions, and scene graphs. As for long captions, DreamLIP [102] proposes to sample sub-captions from the long description to construct multiple positive pairs, while Long-CLIP [97] addresses CLIP’s 77-token limitation by modifying the positional encoding to accommodate longer text sequences during fine-tuning. Meanwhile, region annotations with varying granularity have been considered by works including GLIP [51], X-VLM [95], and RegionCLIP [103]. Their objectives match features of image crops to their specific descriptions. Efforts that aim to improve CLIP training with the help of scene graphs include CLIP-SGVL [35] and Structure-CLIP [37]. The former integrates scene graphs to define additional objective for image encoder, while the later uses scene graphs to guide the generation of negative captions, and to enrich the text encoder with additional contextual information.

Image generation with layout conditioning. Numerous works have studied image generation with layout conditioning, with the conditioning frequently represented as a single prompt with grounding information or bounding boxes along with their respective annotations on top of a global prompt. Early methods primarily focused on training generation models directly with such conditions [99], and some even employed scene graph conditioning and only used layouts as intermediaries for generation [23, 41]. More recent works, such as [101], have effectively integrated layout information into modern image generation models using similar strategies. However, the advent of text-to-image models trained on large-scale paired image/caption datasets has rendered training-from-scratch approaches less necessary. Instead, training additional adapters [53, 61] or controlnets [98, 100] is often sufficient to incorporate layout information into image generation.

Another thread of work focuses on training-free approaches. MultiDiffusion [3] proposes to run the diffusion model sampling process in parallel for different regions and combine them on the fly. This method ensures that each region contained the desired information when individual prompt is simple enough, and has been further refined in LLM-grounded diffusion [54] and RPG-DiffusionMaster [90]. The downside of these approaches is the need for running the sampling multiple times, making the total cost proportional to the number of annotated regions in the image. Another set of approaches manipulates cross-attention. Chen et al. [11] distinguish between forward and backward guidance methods. Forward approaches involve direct intervention in cross-attention scores, while backward approaches update intermediate states at each sampling step to minimize a specific energy function. They found that the backward approach generally outperforms the forward approach. However, it is also more computationally expensive due to the additional gradient steps involved. The baseline methods we implement can be seen as variants of the forward approach. More advanced modulations to improve forward approaches have been proposed by Kim et al. [42] for image generation based on segmentation mask information. On the other hand, BoxDiff [88] can be considered a variant of the backward approach.

Text-to-image generation with additional middleware. It has become increasingly common to transform user-provided prompts into more detailed prompts before feeding them into text-to-image models [4, 92]. To enhance the model’s ability to generate images with complex compositions, other works have focused on incorporating grounding information into user prompts. This can be achieved through various means, such as region annotations [54, 90], semantic panels [25], dense blob representations [61], or Canvas [77]. Additionally, Gu et al. [31] proposed conditioning the final generation on both the initial prompt and the generated middleware, demonstrating promising results.

##### A.2. Limitations and perspectives

We discuss below the limitations of our works from three different perspectives, the procedure and format, the datasets, and the experiments. These limitations also naturally point to several future directions that are to be explored.

- A.2.1. Limitation concerning the GBC procedure and format

While GBC remains a versatile high-level annotation format that in principle applies to any image, its design is inherently tied to the coarse-to-fine and compositional nature of natural images. This design orientation means that GBC is not necessarily the most suitable for certain types of images such as scientific imagery, homogeneous patterns, or abstract art. Specifically, scientific imagery often requires annotations that convey precise, quantifiable data rather than relational or descriptive text. This limitation highlights the need for tailored approaches to different visual content categories to address their unique characteristics.

- A.2.2. Limitation concerning the GBC datasets

Our datasets are curated with the help of LLaVA and Yolo-World, and hence inherit their limitations. This includes but is not limited to, the bias and hallucination from LLaVA captioning, incorrectly identified objects from Yolo-World, and the inability of Yolo-World to recognize certain object category (see Appendix C.3 for concrete examples). Moreover, our approach mainly distinguishes between objects of the same type via composition nodes. Yet, we believe that there is a more effective strategy than merely assigning numbers to these objects.

- A.2.3. Limitation concerning our experiments

Our experiments on CLIP training and retrieval tasks are of relatively small scale and do not fully uncover the potential of the GBC annotation format, especially as we believe that what GBC offers may not be fully represented in traditional benchmarks. Our text-to-image experiments focus on training-free approaches for image generation, which is inherently limited by the underlying model’s capabilities. This limitation is evident in the model’s difficulty generating multiple objects accurately, even with simple prompts, and the attribute binding issues that arise within the text encoder. A training-based approach would likely mitigate these challenges. Lastly, we did not investigate the application of GBC for training MLLMs, which could be a promising avenue for future exploration.

##### A.3. Societal impact

Our paper introduces the GBC datasets and procedure, both aimed at advancing the development of multimodal models. Specifically, the structured approach of GBC, designed to provide detailed descriptions, may help overcome representational biases inherent in existing captioning pipelines, offering more accurate descriptions of images. The potential benefits of these advancements extend across a range of applications, such as assistive technologies and scientific research. However, alongside these benefits, there are challenges including the potential spread of misinformation and concerns about privacy. A comprehensive discussion of these broader societal impacts, both positive and negative, extends beyond the immediate focus of our methodological study.

#### B. Dataset construction

In this appendix, we provide all the missing details about our dataset construction process that are not mentioned in Secs. 3.2 and 3.3.

##### B.1. Query templates

To make the MLLM models fulfill the tasks described in Sec. 3.2, we perform COT prompting [86] with few-shot examples. The four templates for our queries are shown in Figure 5 to 11. We make the following remarks concerning the design of our prompts.

Prompt structure. We craft these prompts with the help of ChatGPT, which results in prompts that might be more complicated than necessary. Meanwhile, we did notice that the inclusion of few-shot examples is crucial for the model to adhere to the required output formats. Given that using always the same few-shot examples might significantly bias the model’s

output, it could be beneficial to randomly retrieve examples from a diverse pool for each query, but we did not pursue this exploration.

[Single] and [Multiple] annotations. Since a detection model could output multiple candidate bounding boxes for an input text, we ask the MLLM to annotate each identified element with either [single] or [multiple]. We then proceed with slightly different algorithms in the two cases, to encourage the selection of either only one, or multiple bounding boxes. In particular, we use respectively an NMS threshold of 0.05 and 0.2 for objects labeled with [single] and [multiple]. However, these labels do not necessarily dictate the final count of bounding boxes; multiple boxes may still be selected for items labeled [single], and vice versa.

Dynamically filled-in elements. To ensure that the response of the MLLM is relevant, the prompts are dynamic and reflect the content of the current image (the image query being the only exception). Such information comes from previous queries and can be naturally retrieved for different queries. The only nonobvious part is the hard coded hints for composition queries, which we explain below.

Hard coded hints for composition queries. After numerous attempts, we observe that LLaVA-1.6 struggles with accurately describing the composition of multiple objects in a scene, even when these objects are annotated with bounding boxes. To overcome this limitation, we guide the models with hints generated programmatically using a set of predefined rules. Specifically, we begin by constructing a Euclidean minimum spanning tree based on the centers of the bounding boxes. We then select a random node as the root and perform a Depth-First Search (DFS) on the tree. During this search, we interleave descriptions of the edges, which detail the geometric relations between two objects based on the positions of their bounding boxes, with node descriptions. These node descriptions are added when an object is located at a particular extremity of the composition, such as the rightmost or top-left position.

##### B.2. Text classifiers

Both of our text classifiers are trained for binary classification using logistic loss. To determine whether a piece of text is suitable for object detection, we utilize a single linear layer added on top of the Jina Embedding.3 For the task of assessing whether two texts can represent the same object, we concatenate their Jina embeddings and process them through an MLP. This MLP includes layer normalization, a hidden layer that expands the input dimensionality by a factor of four, followed by SiLU activation and the final linear layer. The dataset for the training of our text classifiers are prepared with the help of ChatGPT.

##### B.3. Other details for data annotation

We incorporate LLaVA-1.6 into our pipeline using llama.cpp.4 Moreover, to speed up the annotation process, we utilize models quantized at different precision levels: the vision encoders at 6-bit precision, the LLM component of LLaVA-1.6 Mistral-7B at 5-bit precision, and the LLM component of LLaVA-1.6 Yi-34B at 3-bit precision.5 We use the default hyperparameters for inference except for a temperature of 0.1 and context window of size of 5952 (note that LLaVA-1.6 can use up to 2880 image tokens). We discard any responses that do not comply with our required format.

As for the object detection model, we use YOLO-Worldv2-X trained with input resolution of 640 × 640.6 We set the confidence threshold to 0.05 and retain a maximum of six bounding boxes for each input text, selecting those with the highest confidence scores. We exclude any region whose size is smaller than 5,000. To prevent repetitive descriptions of the same element, we keep only those bounding boxes that occupy less than 80% of the current image region for detections arising from entity queries. Regarding node merging, we consider two bounding boxes to be overlapping if their intersection occupies more than 85% of the area of each bounding box involved.

- 3https://huggingface.co/jinaai/jina-embeddings-v2-small-en Accessed: 2024-05-01
- 4https://github.com/ggerganov/llama.cpp Accessed: 2024-05-01
- 5https://huggingface.co/cmp-nct/llava-1.6-gguf Accessed: 2024-05-01
- 6https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_

pretrain-8698fbfa.pth Accessed: 2024-05-01

Query Template for Image Query

###### System message

As an AI visual assistant, your role is to conduct an in-depth analysis of images and articulate a detailed account of the visible elements. You must then distill this information into a precise and concise caption that accurately reflects the content of the image.

Step-by-Step Process: Detailed Caption:

- - Conduct a thorough examination of the image to note all elements present, including main subjects, minor objects, background details, and any text.
- - Prepare a detailed caption that accounts for all these elements, emphasizing the whole objects within the scene. Top-Level Element Identification:
- - Identify and format concrete objects: Begin by identifying concrete objects within the image that are detectable by object recognition models. Each identified object should be formatted as [object_name][node_type] where [node_type] is either [single] or [multiple]: - [single]: Applied to items that appear only once in the image, represented as a unique entity within its context, such as a [cat][single] or a [chair][single]. This category is used regardless of the object’s size or location in the frame and is intended for items that are not repeated elsewhere in the image. For example, a [stop sign][single] on a street corner or a [tree][single] in a field.
- - [multiple]: Applied to items that are present more that once within the image, emphasizing their plurality. Examples include [dogs][multiple] playing in a park,

[chairs][multiple] in a café, [park benches][multiple] along a pathway, [girls][multiple] on a street, [pillows][multiple] on a couch, [paintings][multiple] on a wall, and [lights][multiple] across a ceiling.

- - Entire objects only: When identifying elements within an image, only include objects that stand alone as the main subjects. Avoid breaking down the top-level objects into smaller components.
- - Grouping similar items: When general items, such as houses, trees, players, or people, appear multiple times in the image, they should be grouped together under a single [multiple] label rather than described separately. This approach applies even if these items might have been described individually in the detailed caption.
- - No abstraction: Do not include abstract qualities like colors (blue, red, white), patterns, or expressions.
- - No numbering: Do not use any number to label objects. Just use [houses][multiple].
- - No directional description: Do not use positional terms for individual elements. Instead, group similar items under a single [multiple] label, like [cowboys][multiple]. Concise Formatted Caption:
- - Use the identified elements to construct a concise formatted caption. Use brackets to denote each identified object, following the [object_name][node_type] format. The object name should only appear in the bracket.
- - Restrict the number of elements mentioned in the concise caption to avoid overcrowding and ensure clarity. Prioritize the inclusion of key elements that define the scene or the subject’s essence.
- - The concise caption should contain at most two sentences. Example Adjustments:
- - Character attributes: When analyzing an image featuring a person with distinctive attributes such as armor or tattoos, focus on the person as a whole rather than the individual attributes. The correct annotation would be [person][single], encompassing all aspects of the person appearance without breaking them down into separate elements.
- - Architectural features: In the case of architectural elements, avoid itemizing components like the roof, windows, or door if they contribute to the overall structure of a building. For a singular building in the image, use [house][single]. If the image depicts a series of buildings, such as a row of houses with varying designs, annotate them collectively as [houses][multiple], regardless of their individual features.
- - Groups of similar objects: For scenes containing groups of similar objects or individuals, such as girls playing in a park, group them under a single [multiple] label. Even if the individuals are engaged in different activities or have distinct appearances, they should be annotated as [girls][multiple] to emphasize their collective presence within the scene. Similarly, even if multiple dogs or chairs have different colors, they should be labeled as [dogs][multiple] and [chairs][multiples].

Figure 5: The system prompt used for image query (first half).

Query Template for Image Query

###### System message (continued)

Example Captions: For an image featuring multiple elements like a logo: Detailed Caption: A design showcasing a prominent grey ’N’ at the top, with three smaller NEO Business Bank logos directly below it, two colored squares positioned to the bottom left, and a line of text to the bottom right detailing the availability of various file formats for the design. Top-Level Element Identification:

- - [’N’][single]
- - [Logos][multiple]
- - [Squares][multiple]
- - [Text][single] Concise Formatted Caption: A design showcasing grey [’N’][single] positioned over NEO Business Bank [logos][multiple], accompanied by colored [squares][multiple] and [text][single] at the bottom. For an illustration of a zebra:

Detailed Caption: An animated zebra stands upright on two legs, waving in a welcoming manner, next to a wooden signpost at the beginning of a dirt path. This path leads to a quaint wooden cabin with a thatched straw roof, surrounded by a simple wooden fence. In the background, there’s another similar cabin. The scene is completed by a clear sky overhead and multiple trees dotting the landscape, contributing to the lush greenery. Contextual Considerations: The zebra’s legs are part of its overall form and should not be listed separately. Top-Level Element Identification:

- - [Zebra][single]
- - [Signpost][single]
- - [Dirt path][single]
- - [Cabins][multiple]
- - [Trees][multiple]
- - [Sky][single] Concise Formatted Caption: An animated [zebra][single] waves next to a wooden [signpost][single] on a [dirt path][single] that leads towards wooden [cabins][multiple], with [trees][multiple] enhancing the lush greenery under a clear [sky][single]. For a photo of two men on street:

Detailed Caption: A photo of two men standing side by side on a city street. The man on the left has long hair and is wearing a beige blazer over a white shirt with black trousers. He is smiling and looking directly at the camera. The man on the right has short hair and is dressed in a gray blazer over a black shirt with gray trousers. He also smiles at the camera. They are standing on a sidewalk lined with shops and buildings, suggesting they are in a commercial or urban area. The lighting suggests it might be late afternoon or early evening. Contextual Considerations: The two men, despite their distinct appearances and attire, should be grouped together under a single label since they both fall under the category of "men". Top-Level Element Identification:

- - [Two men][multiple]
- - [Sidewalk][single]
- - [Shops][multiple]
- - [Buildings][multiple]
- - [City street][single] Concise Formatted Caption: [Two men][multiple] stand side by side on a [sidewalk][single] along a [city street][single], lined with [shops][multiple] and [buildings][multiple], each dressed in coordinated blazers and trousers.

###### User message:

Following the instruction, please provide a detailed caption and a concise formatted caption for the given image. Note that it is crucial for you to put elements that can be detected and should be further described in picture in brackets as [object_name][node_type] in the concise formatted caption.

Figure 6: The system and user prompts used for image query (second half).

Query Template for Entity Query

###### System message

Your task is to perform an in-depth analysis of a cropped image focusing on a requested object, like a "house". The process involves a step-by-step evaluation to identify the object’s presence, describe its features, craft concise captions, and assess any prominent objects.

Process Overview: Verify Object Presence:

- - Examine the image to determine if the specified object, or any instance of it, is present.
- - State the presence with "Object Present: Yes" or "Object Present: No". Provide Appropriate Caption (If Object Is Present):
- - Provide a detailed description of the object, focusing solely on its features without reference to other elements in the image.
- - The description should contain at most 50 words. Assessment of Prominent Objects:
- - Evaluate the described features to determine if any stand out for further description and are detectable by an object detection model. This is crucial for complex objects such as ’man’, ’woman’, ’family’, ’couple’, ’cat’, or ’house’, where components or distinctive attributes are significant. For example, when analyzing ’woman’, consider elements like dress [single], skirt [single], or hair [single] as prominent features. For simpler objects like ’cup’ or ’chair’, additional descriptions may not be needed. Identification of Prominent Features (If Applicable):
- - If there are prominent features identified, list and format these features for potential detection by an object detection model.
- - Ensure these features are parts or components of the main object and not the entire object itself.
- - Use [single] for unique, standalone items, and [multiple] for features present more than once, such as roof [single] or windows [multiple].
- - Group similar items under a single [multiple] label rather than describing them separately, even if individual descriptions were provided in the detailed caption. For example, multiple distinct windows in a house should be labeled as windows [multiple] rather than individually enumerated.
- - For groups like families or couples, identify members separately (e.g., man [single], woman [single]) rather than as a collective unit. This contrasts with grouping similar inanimate objects (e.g., windows [multiple]), where individual distinction isn’t necessary.
- - Consistency with the caption: Ensure that the features identified as [single] or [multiple] are also mentioned in the caption. Example Responses:

- Example 1: Object Not Present

Object Presence: No Detailed Caption: N/A Prominent Features: N/A Identification of Prominent Features: N/A

- Example 2: Object Present Without Prominent Features (requested object: "cup")

Object Presence: Yes Detailed Caption: A simple ceramic cup on a wooden table. The cup has a smooth, unadorned surface and a standard curved handle on one side. Prominent Features: No Identification of Prominent Features: N/A

Figure 7: The system prompt used for entity query (first half).

Query Template for Entity Query

###### System message (continued)

- Example 3: Object Present With Prominent Features (requested object: "family")

Object Presence: Yes Detailed Caption: A family of four is captured enjoying a sunny day in the park. The father, in casual attire, is engrossed in setting up a picnic spot, while the mother, donned in a summer dress, is laying out a feast on a blanket. Nearby, two children, a boy and a girl, engage in playful antics; the boy is kicking a football with fervor, and the girl, adorned in a light frock, is gleefully chasing bubbles. Prominent Features: Yes Identification of Prominent Features:

- - Father: [single]
- - Mother: [single]
- - Boy: [single]
- - Girl: [single]

- Example 4: Object Present With Prominent Features (requested object: "car")

Object Presence: Yes Detailed Caption: A vintage car in pristine condition, with shiny chrome bumpers and classic spoke wheels. The car’s body is painted in a vibrant red, and the leather interior is visible through the clear windows. A unique hood ornament adorns the front, adding to the car’s elegance. Prominent Features: Yes Identification of Prominent Features:

- - Chrome bumpers: [single]
- - Wheels: [multiple]
- - Hood ornament: [single]

###### User message:

Please assess the image focusing on ’{}’. Start by confirming its presence with ’Object Present: Yes’ or ’Object Present: No’. If present, describe its key features in a detailed caption with at most 50 words. Then, evaluate if any aspects stand out for further emphasis, stating ’Prominent Features: Yes’ or ’No’ while preferring "Yes". If yes, list a few notable features in brackets, applying [single] or [multiple] as appropriate. Importantly, do not include ’{}’ in features. Instead, you should break it down.

[Figure 21]

###### Example filled-in elements

- • boat
- • boat

Figure 8: The system and user prompts used for entity query (second half). The placeholders ‘{}’ are dynamically filled with the name of the object, i.e., the label of the associated incoming edge.

Query Template for Composition Query

###### System message

Your role is to analyze images containing objects within pre-labeled bounding boxes and describe the compositional arrangement of these objects based on provided hints. You will then provide general descriptions that apply to all the objects collectively.

Input Image Format Explanation:

- - The image will feature objects of interest, each enclosed within a bounding box.
- - Each bounding box will be numbered centrally to uniquely identify it.
- - The objects will be similar in nature (e.g., all dogs) and positioned within a scene. Utilizing Hints for Analyzing Composition:
- - Begin by reviewing the hints provided regarding the spatial arrangement of the objects.
- - These hints may specify the relative positions of objects (e.g., "Object 3 is in the top right corner").
- - Use the hints to guide your description of how the objects relate to each other within their bounding boxes. Output Format:
- - Composition Description: Start with "Composition:" followed by a description informed by the hints and using the bounding box numbers. This description should elucidate the spatial arrangement of the objects as per the hints.
- - General Descriptions: Provide observations that apply to all objects within the specified group, excluding unrelated elements or background details. Preface this section with "General descriptions:". Additional Guidelines:
- - Describe the spatial arrangement of objects without inferring spatial relations from the sequence of numbers.
- - Utilize clear spatial language to articulate the composition.
- - The description should reflect the actual visual composition, not the order of numbers in the bounding boxes. Examples: Example for 3 Dogs in Bounding Boxes: Query Prompt: "Please describe the composition of the 3 dogs in the bounding boxes, followed by some general descriptions that apply to all dogs." System Response:

Composition: Dog 3 is in front, with dog 2 to the left and dog 1 to the right. General descriptions:

- - The three dogs are aligned in a row on the grass.
- - They share similar sizes and features, suggesting they may be from the same breed. Additional Examples:

For 5 Flowers in a Garden Bed in Bounding Boxes: Composition: Flower 4 takes a central position, flanked by flower 2 and flower 3 on either side, while flower 1 and flower 5 bookend the arrangement at the outer edges. General descriptions:

- - Each flower is in full bloom, indicating a peak growing season.

For 2 Cats in a Window in Bounding Boxes: Composition: Cat 1 is positioned on the left side of the window sill, while cat 2 is curled up on the right. General descriptions:

- - Both cats are basking in the sunlight coming through the window.
- - Their relaxed postures suggest a shared sense of comfort and tranquility.

Figure 9: The system prompt used for composition query.

Query Template for Composition Query

###### User message:

Please describe the composition of the {} in the bounding boxes, followed by some general descriptions that apply to all {}. The composition should include {} and be based on the following hints (do not mention hints or bounding boxes in the response). {}

###### Example filled-in elements

- • boats
- • boats
- • boat 2, boat 3, boat 1
- • - boat 2 is on the right side of the composition

- - boat 2 is to the right of boat 3
- - boat 3 is to the right of boat 1
- - boat 1 is on the left side of the composition

[Figure 22]

- Figure 10: The user prompt used for composition query. The placeholders ‘{}’ are dynamically filled with the name of the object, the labels of the out edges (in the form of the name of the object plus number), and hard coded hints that are obtained using the positions of the bounding boxes.

Query Template for Relation Query

###### System message

Your role involves analyzing the spatial and direct interactions between pre-identified elements within an image, described through annotations like [beach], [turquoise waters], [people], [shoreline], [lush vegetation]. Your task is to objectively describe how these elements are related or positioned relative to each other within the scene.

Steps for Identifying Objective Relations:

- 1. Review Annotated Elements: Start by examining the list of annotated elements. Understand the nature of each element as it is presented in the image.
- 2. Identify Spatial Positions: Determine the spatial positioning of these elements in relation to each other. Focus on direct relationships such as touching, overlapping, or proximity without implying any further interpretation.
- 3. Describe Direct Interactions: Look for and describe any direct interactions between elements, such as one element supporting another, blocking, or leading into another.
- 4. Format Relations List: Provide your findings as a list of bullet points. Each bullet should detail a direct and observable relationship between two or more elements, using their annotated identifiers for reference. Example Relations Based on Annotated Elements: For elements: [beach], [turquoise waters], [people], [shoreline], [lush vegetation], you might reply:

- - The [people] are standing on the [beach], with the [lush vegetation] to their left.
- - [Turquoise waters] lap against the [beach] at the [shoreline], with [people] scattered along its edge.
- - [Lush vegetation] flanks the left side of the [beach], providing a natural border.
- - The [shoreline] separates the [beach] from the [turquoise waters].
- - To the right of the [lush vegetation], the [beach] stretches towards the [turquoise waters]. For another set of elements: [eagle], [snake], [wings], you might reply:
- - The [eagle] has its [wings] spread above the [snake].
- - The [snake] is positioned below the [eagle].
- - The [eagle]’s claws are near or in contact with the [snake]. Guidelines for Reporting Relations:

- 1. Ensure descriptions are based solely on visible or directly inferable relationships without adding interpretations or assumptions.
- 2. Maintain clarity and precision in articulating the spatial and interactional dynamics between elements.
- 3. Stick to objective descriptions that reflect the physical and observable aspects of the elements’ relationships.
- 4. Only answer the list of bullet points without anything before or after.
- 5. Do not include any bullet point with 1 or even 0 elements.

- - Visible Relationships Only: Report relationships that are clearly depicted in the image. If no clear relationships are visible, state "No visible relationships."
- - Objective Descriptions: Keep descriptions factual and based solely on what can be seen in the image.
- - Avoid Assumptions: Do not infer or assume any relationships that aren’t clearly shown in the image.
- - Bullet Point Format: Present each observable relationship as a separate bullet point, avoiding any descriptive text not related to the direct relationships.
- - No Relation Inference: Refrain from implying relationships or positions that are not explicitly shown. If elements are simply present without any discernible interaction, it is acceptable to say "Elements are present without visible interaction."
- - Avoid Single Element Points: Do not include bullet points that mention only one element or have no elements at all. Each bullet point must reference the relationship between two or more elements.

###### User message:

Please infer and list of at most {} relations between {} in this images.

[Figure 23]

###### Example filled-in elements:

- • 4
- • eiffel tower, sky, trees, boat, water

- Figure 11: The system and user prompts used for relation query. The placeholders ‘{}’ are dynamically filled with a random number between 2 and the number of involved objects, and the names of these objects.

- B.4. Computation cost We list below the major computation cost of our data preparation process.

- • GBC1M: With our processing pipeline, it takes an average of around 3 minutes to annotate each image on an A100 80G when all the queries are performed with LLaVA-1.6 Yi-34B. As a result, annotating 1 million images took us around 6 days with 300 A100 80Gs.
- • GBC10M: The average annotation time per image on an A100 80G is improved to 1 minute when relation and entity queries are performed with LLaVA-1.6 Mistral-7B. This process is about twice slower on a V100 32G. In this regard, our GBC12M dataset was compiled in roughly 6 days using 500 A100 80Gs and 1,000 V100 32Gs.

We also compute CLIP score for each caption using the DFN-5B model.7 This computation takes around 3 hours for every 10,000 images on a V100 32G.

#### C. Dataset information

In this appendix, we provide information about dataset release, dataset statistics, and visualizations of a few examples from our GBC10M dataset.

##### C.1. Data release and licensing

Our datasets are available at https://huggingface.co/graph-based-captions, released under the CC BY-NC 4.0 license. Following CC12M, we include URLs to images along with captions generated through our GBC procedure, all stored in JSON lines format. Comprehensive documentation including a dataset card and croissant metadata is also provided in the data repository.

Personal identifiable information and offensive content. Our dataset comprises only captions generated by MLLM models (LLaVA 1.6 Yi-34B and LLaVA 1.6 Mistral-7B), which were trained on carefully curated data. The images, sourced from CC12M, are generally free from offensive content. In particular, CC12M is the result of a filtering operation involving adult content detection on images and their captions. While CC12M images may include human faces, we do not host the images directly; only the URLs are provided. Additionally, we conduct toxicity check with Detoxify [34] on a subset of examples in GBC dataset and find no harmful contents. While it was not possible to manually examine all the samples produced by GBC pipeline, we believe that the protective measures of the source dataset and model are sufficient to avoid both harmful content, and privacy leakages.

##### C.2. Dataset statistics

In this section, we provide statistical insights into the GBC1M and GBC10M datasets. In particular, we zoom in on the statistics at image, vertex, edge, and caption levels, and present distributions of several key metrics including for example caption length, region size, and CLIP score. Since most of these metrics exhibit long-tailed distributions, we often group excessively large values into a single histogram bin for better visualization.

###### C.2.1. Image and graph statistics

We first look at the sizes of the images and of the annotation graphs, i.e. the numbers of vertices and edges in these graphs and their diameters (which is measured as the length of the longest path in a directed graph). The distributions of these metrics are shown in Figures 12 and 13. We see that the image size has a very long-tailed distribution, with the majority of images having around 786 × 786 pixels. Conversely, the distributions of graph diameters are more similar to that of a Poisson or a binomial distribution, with most of the graphs having a diameter between 3 and 6. Finally, as one could expect, the numbers of vertices and edges share quite similar distributions.

While we expect the size of a graph to reflect the inherent complexity of an image, we acknowledge that our annotations are influenced by the biases of the used models. In particular, we observe that our annotation process tends to yield larger graph for natural images compared to other types of images such as artworks or graphic designs.

7https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378 Accessed: 2024-05-01

Number of Pixels per Image

Graph Diameter

1e5

1e5

4.0

Max: 7.50e+07

Mean: 879266.01

Mean: 4.55

3.5

2.0

3.0

1.5

2.5

Counts

Counts

2.0

1.0

1.5

1.0

0.5

0.5

0.0

0.001 1.667 3.333 5.000 6.666 8.332 1e6

0.0

1 3 5 7 9 11 13 15

Number of Vertices per Image

Number of Edges per Image

1e5

1e5

1.2

1.4

Mean: 12.13

Mean: 22.28

1.0

1.2

1.0

0.8

Counts

Counts

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

1 9 18 27 36 45-196

0 18 36 54 72 91-372

Figure 12: Distributions of metrics at image and graph level in the GBC1M Dataset.

Number of Pixels per Image

Graph Diameter

1e6

1e6

4.0

2.5

Max: 1.07e+08

Mean: 879878.32

Mean: 4.41

3.5

2.0

3.0

2.5

Counts

1.5

Counts

2.0

1.5

1.0

1.0

0.5

0.5

0.0

0.000 1.667 3.333 5.000 6.666 8.332 1e6

0.0

1 3 5 7 9 11 13 15

Number of Vertices per Image

Number of Edges per Image

1e6

1e6

1.2

Mean: 12.24

Mean: 21.81

1.0

1.0

0.8

0.8

Counts

Counts

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

1 9 17 25 33 42-187

0 16 32 48 64 81-508

Figure 13: Distributions of metrics at image and graph level in the GBC10M Dataset.

Dataset # Images # Regions / Image COCO [55] 123,000 7 Visual Genome [45] 108, 249 42 Objects365 [73] 638,000 16 Open Images [46] 1.7M 8 BigDetection [7] 3.5M 10 SA-1B [44] 11M 100 AS-1B [85] 11M 110 DCI [19] 7,805 40 GBC1M (ours) 1.1M 11 GBC10M (ours) 10.1M 11

- Table 4: Comparison of number of regions per image among several vision-language datasets with region-based annotations. We use the statistics reported in the original paper although some datasets, such as COCO and Open Images have been updated after their initial release. Moreover, for Open Images we report the number for the training set with bounding box annotation [46, Tab. 5]. For DCI, we compute the average number of regions per image ourselves using their open-sourced dataset with 7,805 images as this number is not reported in [19].

###### C.2.2. Vertex statistics

We have shown previously that our datasets contain an average of 12 vertices per graph. This translates to 11 regions per image after excluding the root node that represents the entire image. We compare this number with several other vision-language datasets with region-based annotations in Tab. 4. As one can see, this number aligns well with many of these datasets, particularly those used for detection, such as COCO and Object365. However, it lags behind compared to Visual Genome and more recent datasets with dense annotations, such as AS-1B and DCI. We believe this discrepancy can be attributed to both the top-down design of our annotation process, which tends to overlook less significant components of the images, and the limitations of the detection model used. Notably, both AS-1B and DCI utilize Segment Anything [44] to identify regions of interest. Segment Anything is trained on SA-1B, which has much denser annotations compared to the object detection datasets used for training Yolo-World.

We next examine how this number is distributed across the different types of nodes that are present in our graphs. For this, we plot the distributions of the numbers of composition nodes, relation nodes, entity nodes, and leaves (i.e. the nodes without any children) in Figures 14 and 15. As seen in the figures, a large number of vertices are entity nodes, which focus on describing a single object. In spite of this, we still have an average number of 4 vertices per graph that are dedicated to describing the composition or relationships between multiple objects.

To complete our investigation, we visualize the distributions of the sizes of the vertices’ bounding boxes in Figures 16 and 17. We note that most of the regions have small relative size (smaller than 0.1). This is also observed in other datasets such as Visual Genome [45, Fig. 15] and Open Images [46, Fig. 20]. Relation nodes, whose bounding boxes are defined as the minimum bounding box containing the union of all the involved objects’ bounding boxes, have sizes that spread more uniformly across different ratios. We also observe a large number of entity nodes with bounding boxes that have a relative size close to 1. This likely corresponds to background objects that spans across the entire image, such as “sky” or “grass”.

###### C.2.3. Edge statistics

Our datasets feature an average of 22 edges per graph. We analyze the origins of these edges in Figure 18, which shows their distributions across different types of source vertices. The figure indicates that the image node is responsible for a large proportion of these edges, suggesting that many of the entities that we identify directly come from the image caption. This is natural provided that an image often contains many objects, while it is less common to need further decomposition of a single object for detailed description. Besides this, these figures also indicate the number of entities that are involved in our composition and relation descriptions. Notably, we see that most of these descriptions only contain 2 or 3 objects, with few of them involving more than 4 objects. In contrast, we observe a relatively large number of entity nodes with 4 outgoing edges, and we believe this can be attributed to the bias caused by the few-shot examples provided in our query template.

Number of Composition Nodes per Image

Number of Relation Nodes per Image

1e5

1e5

- 0

- 1

- 2

- 3

- 4

- 5

2.00

Mean: 1.10

Mean: 2.61

1.75

1.50

1.25

Counts

Counts

1.00

0.75

0.50

0.25

0.00

0 1 2 3 4 5 6 7-30

0 2 4 6 8 10 12-46

Number of Entity Nodes per Image

Number of Leaves per Image

1e5

1e5

1.0

1.2

Mean: 7.41

Mean: 5.77

0.8

1.0

0.6

0.8

Counts

Counts

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 3 6 9 12 15 18 21 24 27 30-132

1 3 5 7 9 11 13 15-101

Figure 14: Distributions of vertex numbers across different types of vertices in the GBC1M Dataset.

Number of Composition Nodes per Image

Number of Relation Nodes per Image

1e6

1e6

2.00

Mean: 1.15

Mean: 2.76

1.75

- 0

- 1

- 2

- 3

- 4

1.50

1.25

Counts

Counts

1.00

0.75

0.50

0.25

0.00

0 1 2 3 4 5 6 7-35

0 1 2 3 4 5 6 7 8 9 10-67

Number of Entity Nodes per Image

Number of Leaves per Image

1e6

1e6

1.0

1.2

Mean: 7.33

Mean: 6.01

0.8

1.0

0.8

0.6

Counts

Counts

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 3 6 9 12 15 18 21 24 27-129

1 3 5 7 9 11 13 15-102

Figure 15: Distributions of vertex numbers across different types of vertices in the GBC10M Dataset.

Absolute Bounding Box Size

1e6

Image Node Composition Node Relation Node Entity Node

4.0

| |
|---|

3.5

| |
|---|

| |
|---|

3.0

Max: 7.50e+07

2.5

Counts

2.0

1.5

1.0

0.5

0.0

0.0008 0.2006 0.4005 0.6003 0.8002 1.0000 1e6

1e6 Relative Bounding Box Size

| |Image Node Composition Node Relation Node Entity Node<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

3.0

2.5

2.0

Counts

1.5

1.0

0.5

0.0

0.0 0.2 0.4 0.6 0.8 1.0

- Figure 16: Distribution of bounding box sizes in the GBC1M Dataset. We show both the absolute size (number of pixels) and the relative size (normalized by image size).

0.0003 0.2002 0.4002 0.6001 0.8001 1.0000 1e6

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

4.0

Counts

1e7

Max: 1.07e+08

Absolute Bounding Box Size

Image Node Composition Node Relation Node Entity Node

| |
|---|

| |
|---|

| |
|---|

| |Image Node Composition Node Relation Node Entity Node<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

0.0 0.2 0.4 0.6 0.8 1.0

0.0

0.5

1.0

1.5

2.0

2.5

3.0

Counts

1e7 Relative Bounding Box Size

- Figure 17: Distribution of bounding box sizes in the GBC10M Dataset. We show both the absolute size (number of pixels) and the relative size (normalized by image size).

| |Image Node Composition Node Relation Node Entity Node<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

0 2 4 6 8 10 12 14 16-125

- 0
- 1
- 2
- 3
- 4
- 5
- 6

Counts

1e6 Number of Out Edges

(a) GBC1M

| |Image Node Composition Node Relation Node Entity Node<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

0 2 4 6 8 10 12 14 16-254

- 0
- 1
- 2
- 3
- 4
- 5
- 6

Counts

1e7 Number of Out Edges

(b) GBC10M

- Figure 18: Distributions of number of outgoing edges across different types of vertices in the GBC1M (left) and GBC10M (right) datasets.

−0.1531 −0.0020 0.1491 0.3002 0.4512 0.6023

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

Counts

1e6 dfn5b-h-patch14-378 Score Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption

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

(a) GBC1M

−0.1875 −0.0260 0.1355 0.2970 0.4585 0.6200

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

Counts

1e7 dfn5b-h-patch14-378 Score Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption

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

(b) GBC10M

- Figure 19: Distributions of DFN-5B CLIP scores across different types of captions in the GBC1M (left) and GBC10M (right) datasets.

Caption Type # Captions # Words / Caption # Tokens / Caption CLIP score

17.4 24.5 0.36 Image Short 28.1 35.3 0.33 Image Detail 110.3 130.9 0.26

Image Original

1,013,592

GBC1M

Entity 7,512,638 37.5 46.3 0.29 Composition 1,117,935 35.8 44.1 0.23 Multi-Entity 3,487,562 17.8 23.1 0.25 Relation 3,493,543 22.0 27.2 0.30

17.4 24.6 0.36 Image Short 28.1 35.3 0.33 Image Detail 110.3 130.9 0.26

Image Original

10,138,757

GBC10M

Entity 74,354,424 33.9 42.1 0.28 Composition 11,621,125 36.2 44.5 0.22 Multi-Entity 36,359,826 17.9 23.2 0.24 Relation 36,606,028 11.5 15.3 0.28

- Table 5: Key caption statistics of the GBC1M and GBC10M datasets across different types of captions. We use the DFN-5B CLIP model to compute the CLIP scores.

We also provide analysis for the edge labels. These edge labels should represent the objects that are associated to their respective target vertices. In particular, during our annotation process, we use these labels as input of the detection model to obtain the bounding boxes of the entity nodes. In Figures 21 and 23, we plot the distributions of the numbers of words and tokens contained in the edge labels. As expected, most of the time we use only 1 or 2 words to represent the entities.

We next study the content of these labels. To this end, we plot the distribution of (i) the 20 most common edge labels at the in-edges of the entity nodes, reflecting the content of these entity nodes, and (ii) the 20 most common edge label pairs when pairing the in- and out-edges of the entity node, reflecting the situation where we zoom in on an object to further describe a part of it. The corresponding histograms are presented in Figures 20 and 22. From these plots, we see that the most common objects from our datasets are “tree”, “sky”, “man”, “woman”, “table”, and “building”, among others. This distribution aligns well with the ones reported for existing datasets, cf. [45, Fig. 22] and [46, Tab. 11]. Furthermore, while the occurrence of certain labels and label pairs, such as (“woman”, “hair”), may be influenced by our system prompts, others like (“bed”, “pillows”) are widely present despite not being included in our prompts. This suggests potential biases in either the model or the dataset itself.

###### C.2.4. Caption statistics

For statistics at the caption level, we first complete Tab. 1 by providing distribution of CLIP scores on the two datasets in Figure 19, and distribution of number of captions, words, and tokens per image in Figures 24 and 25. In particular, the significant variation in CLIP score distributions across different caption types motivates our decision to perform CLIP-filtering independently for each type, as mentioned in Sec. 4.2.

Going further, we report the average number of words and tokens per caption across different types of captions in Figure 27, 29, and Tab. 5. We can see that except for the detailed image captions, most captions indeed contain fewer than 77 tokens. Tab. 5 additionally reveals that we have near 2.5 times more region captions (i.e. entity and multi-entity captions) than the total of relation and composition captions. However, as we have seen in Sec. 4.3 and will further ablate in Appendix F.5, these relation and composition captions, unique to our dataset, are crucial for the performance improvement that we observe across different evaluations.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

tree

sky hair

shirt person

man window woman

building dress table

text chair suit house plant

wall flower

pants bed

TopIn-EdgeLabelsofEntityNodes

leafe jacket t-shirt water

trees

car floor

cloud leaves

branch bowl plate

windows pillow

roof balcony

character door skirt

uniform hand street

dog desk eyes book

countertop hat tie flowers

0.0 0.5 1.0 1.5 2.0 Counts 1e5

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

woman, hair man, shirt woman, dress

man, suit bed, pillows

man, hair

building, windows tree, branches bed, comforter

person, hair bed, bedspread person, shirt bed, headboard dress, skirt suit, shirt woman, blonde hair

dress, bodice house, windows

TopIn-OutEdgeLabelPairsofEntityNodes

man, tie t-shirt, text

dress, sleeves sky, clouds house, roof

suit, tie chair, backrest house, porch

man, pants tree, leaves man, beard

building, balconies person, pants player, jersey

chair, armrests dress, lace detailing

plant, leaves

man, jacket couple, man

woman, blouse man, t-shirt tree, trunk girl, hair couple, woman person, jacket suit, suit jacket

woman, shirt bride, dress forest, trees

building, flat roof

garden, plants kitchen, countertops

0.00 0.25 0.50 0.75 1.00 1.25 1.50 Counts 1e5

- Figure 20: Distributions of the 20 most common in-edge labels and in-/out-edge label pairs at entity nodes in the GBC1M dataset. We remove numbers from the edge labels for the computation of their occurrences in these plots.

1 2 3 4 5 6-19

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

Counts

1e7

Mean: 1.40

Number of Words per Edge Text

3 4 5 6-29

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

Counts

1e7

Mean: 3.51

Number of Tokens per Edge Text

- Figure 21: Distributions of numbers of words/tokens in each edge label in the GBC1M dataset. To compute the number of tokens we use

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
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

tree window woman

sky man

person

hair building

table

shirt chair dress

text leafe

wall house jacket

roof bed suit

TopIn-EdgeLabelsofEntityNodes

plant flower t-shirt

car water

trees

door leaves

balcony pants floor windows sign

street plate cloud

bowl

character book desk

cabinet dog countertop jersey eyes helmet tie branch beard hat

0.0 0.5 1.0 1.5 2.0 2.5 Counts 1e6

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

woman, hair

woman, dress building, windows

man, shirt house, roof man, suit

man, beard house, windows

house, porch man, hair building, roof

man, tie person, hair man, jacket

tree, branches dress, sleeves

player, jersey building, balcony tree, leaves

forest, trees TopIn-OutEdgeLabelPairsofEntityNodes

person, jacket dress, neckline

person, shirt

tree, trunk t-shirt, text

bed, pillows

dress, skirt man, mustache

kitchen, cabinets bed, headboard plant, leaves woman, blouse kitchen, island building, sign

chair, armrests building, balconies

girl, hair

man, pants house, balcony

car, wheels couple, man suit, tie

house, door woman, jacket character, hair

woman, necklace chair, backrest couple, woman

man, glasses kitchen, countertop

0.00 0.25 0.50 0.75 1.00 1.25 1.50 Counts 1e6

- Figure 22: Distributions of the 20 most common in-edge labels and in-/out-edge label pairs at entity nodes in the GBC10M dataset. We remove numbers from the edge labels for the computation of their occurrences in these plots.

1 2 3 4 5 6-57

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

Counts

1e8

Mean: 1.40

Number of Words per Edge Text

3 4 5 6-70

0.0

0.2

0.4

0.6

0.8

1.0

1.2

Counts

1e8

Mean: 3.51

Number of Tokens per Edge Text

- Figure 23: Distributions of numbers of words/tokens in each edge label in the GBC1M dataset. To compute the number of tokens we use

Number of Captions per Image

Number of Words per Image

Number of Tokens per Image

1e5

1e5

1e5

1.4

1.4

1.4

Mean: 17.40

Mean: 593.14

Mean: 730.76

1.2

1.2

1.2

1.0

1.0

1.0

0.8

0.8

0.8

Counts

Counts

Counts

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

2 16 30 45 59 74-420

10 462 914 1366 1818 2270-8562

16 576 1136 1696 2256 2817-19709

Figure 24: Distributions of numbers of captions, words, and tokens per image in the GBC1M Dataset.

Number of Captions per Image

Number of Words per Image

Number of Tokens per Image

1e6

1e6

1e6

1.4

1.4

1.4

Mean: 17.68

Mean: 533.98

Mean: 664.82

1.2

1.2

1.2

1.0

1.0

1.0

0.8

Counts

Counts

Counts

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

2 15 29 42 56 70-616

8 394 780 1166 1552 1938-9781

14 499 985 1471 1957 2443-13391

Figure 25: Distributions of numbers of captions, words, and tokens per image in the GBC10M Dataset.

We conclude this part by showing the distribution of the 20 most common words and trigrams that appear in our captions in Figures 26 and 28, with stop words removed when considering the word distributions. The frequent appearances of colors among the top words again align with the distribution reported in Visual Genome [45, Fig. 24]. In addition, phrases like “appears to be”, “possibly”, and “the image captures” that commonly appear in our data, reflect LLaVA’s use of GPT-generated data during instruction tuning.

##### C.3. Examples from GBC10M

As a complement to the dataset statistics presented in the previous section, we showcase a few illustrative examples from GBC10M in Figures 30 and 31. These examples demonstrate the varying levels of graph complexity across our dataset. The number of nodes varies from just a few (first example in Figure 30) to over 10 (third example in Figure 30 and the example in Figure 31). In most cases, this complexity aligns with the visual complexity of the corresponding image.

On the other hand, these examples also reveal limitations arising from the object detection models used. For instance, in the Messe example from Figure 30, the detection model incorrectly identifies a standing priest as a “kneeling figure”. Similarly, in Figure 31, two of the three nodes labeled “trunk” are derived from tree nodes and erroneously associated with the elephant’s trunk or other non-trunk objects on the elephant. These limitations become particularly severe in the Regalia example of Figure 30, where the presence of more specific objects like crowns, scepters, bracelets, and earrings leads to frequent confusion by the object detection model.

Next, we focus on the captions associated with these images. A subset of these captions is presented in Tabs. 6 and 7. We observe that hallucination is particularly important for detailed captions. These erroneous descriptions can then be inherited by the shorter captions derived from them. We also note there are situations where the model describes an object that actually does not exist in the corresponding region of the image, such as the caption for “scepter 1” in the Regalia example. As we can see from the figure, in the corresponding bounding box, there is no scepter visible but only a crown on a wooden base.

In spite of these inaccuracies in object detection and captioning, the overall graph structure and captions still align well with the images. On top of this, the granularity of our descriptions significantly enhances the descriptive power of our dataset, allowing for a more nuanced understanding of the visual content.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

appears white

suggesting design visible

right side blue

black image

left

vibrant suggests

possibly

top light

stands positioned

surface background scene

TopWordsinCaptions

red part

indicating

green wearing

featuring

sky color

creating

large person

made tree hair

adding composition

might multiple

dark window

overall shirt smooth

two clear

wooden located natural

text

0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 Counts 1e6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

appears to be

part of a a sense of on the left

to be made be made from

it has a it might be

to be a the left side

made from a contrasts with the

a touch of the right side

on the right side of the are part of

at the top a close-up view

close-up view of of the composition view of a the image captures image captures a at the bottom positioned on the in the background of a larger

and appears to TopTrigramsinCaptions

in front of is positioned on

that contrasts with be part of suggesting they are appear to be

is wearing a with no visible

suggesting it might with a smooth it appears to arranged in a it could be in the top

adding a touch to the scene with a white dressed in a

2 is positioned

are arranged in against a backdrop

0.0 0.5 1.0 1.5 2.0 Counts 1e6

Figure 26: Distributions of the 20 most common words and trigrams that appear in the captions of the GBC1M dataset.

1e6 Number of Words per Caption Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption

4.0

| |
|---|

| |
|---|

3.5

| |
|---|

| |
|---|

3.0

| |
|---|

2.5

| |
|---|

Counts

2.0

1.5

1.0

0.5

0.0

0 40 80 120 160 200-3276

1e6 Number of Tokens per Caption

| |Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

4.0

3.5

3.0

2.5

Counts

2.0

1.5

1.0

0.5

0.0

2 41 81 120 160 200-10362

Figure 27: Distributions of numbers of words/tokens across different types of captions in the GBC1M dataset. To compute the number of

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

appears image

white visible

right design

side

black positioned

wearing left top

blue suggesting

large green view

red close-up

background surface

TopWordsinCaptions

person made scene

vibrant tree part light

sky composition

window

hair showcases

possibly located

two suggests

features wooden multiple

color building

bottom front dark

creating indicating

small leaves

text

0 1 2 3 4 Counts 1e7

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

appears to be a close-up view

close-up view of view of a

it has a to be made

the image showcases image showcases a

part of a on the left

to be a it appears to

a sense of

the left side be made of on the right

the right side

at the top side of the

is wearing a

in front of be made from

in the bottom TopTrigramsinCaptions

the image shows of the composition image shows a at the bottom in the background and appears to it might be made from a arranged in a of a larger

the image captures positioned on the image captures a

is positioned on are arranged in

in the image with a white

appear to be

are part of with no visible with a smooth

a smooth surface in the top be part of

front of the suggesting they are

a person wearing

0.0 0.5 1.0 1.5 2.0 2.5 3.0 Counts 1e7

Figure 28: Distributions of the 20 most common words and trigrams that appear in the captions of the GBC10M dataset.

1e7 Number of Words per Caption Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption

| |
|---|

- 0

- 1

- 2

- 3

- 4

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

Counts

0 40 80 120 160 200-3492

1e7 Number of Tokens per Caption

4.0

| |Original Image Caption Short Image Caption Detail Image Caption Entity Caption Composition Caption Multi-Entity Caption Relation Caption<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

3.5

3.0

2.5

Counts

2.0

1.5

1.0

0.5

0.0

2 41 81 120 160 200-6787

- Figure 29: Distributions of numbers of words/tokens across different types of captions in the GBC10M dataset. To compute the number of

Flame:

[Figure 24]

[Figure 25]

Messe:

[Figure 26]

[Figure 27]

[Figure 28]

Regalia:

[Figure 29]

- Figure 30: Example images and graphs from the GBC10M dataset. For ease of visualization we do not show all the bounding boxes.

- Elephant:
- Figure 31: An example image with its graph from the GBC10M dataset. For ease of visualization do not show all the bounding boxes.

[Figure 30]

[Figure 31]

|Short Captions<br><br>|Flame [Figure 30]. A flame with yellow base and blue peak emerges from a metal object against a dark background.<br><br>Messe [Figure 30]. A priest holds a chalice aloft while another figure kneeling figure kneels on the floor, set against a backdrop of architectural details and ornamentation within what appears to be a religious setting.<br><br>Regalia [Figure 30]. Two people stand behind a display case containing a crown, scepter with a blue gem, and a golden orb with a red gem, all under natural light from a window or glass panel within an indoor setting.<br><br>Elephant [Figure 31]. A man rides atop a bench strapped on a elephant drinking from a riverbank, surrounded by trees under a clear sky.|
|---|---|
|Detailed Captions|Flame [Figure 30]. The image captures a close-up view of a blue flame emanating from a small metal object, which appears to be a lighter or torch. The flame has a vibrant yellow hue at its base, transitioning to a bright blue at its peak. The flame’s shape is irregular with wisps extending outward from its core, suggesting it’s in motion or has been recently ignited. The metal object has a cylindrical shape with a pointed tip from which the flame emerges. The background is dark, providing a stark contrast that accentuates the flame’s colors and form.<br><br>Messe [Figure 30]. The image portrays a religious scene set within what appears to be a church or chapel. At the center of the composition stands a priest, dressed in traditional religious attire with a red robe and a white cowl. He holds a golden chalice aloft with both hands, suggesting he may be performing a sacrament or ritual. To his right, another figure, possibly another priest or religious figure, kneels on the floor, seemingly in prayer or reverence. The background features ornate architectural details, including arches and intricate patterns on the walls, indicative of Gothic or similar architectural styles. The overall atmosphere suggests a solemn or sacred moment within a religious ceremony or service.<br><br>Regalia [Figure 30]. The image captures a scene where two individuals are standing behind a display case containing various items. The display case houses a collection of ornate jewelry pieces, including a crown with intricate detailing, a scepter with a blue gem at its top, and a golden orb with a red gem. The individuals are dressed in pink shirts and are positioned behind the display case, which has a reflective surface. The background suggests an indoor setting with a window or glass panel allowing natural light to illuminate the scene.<br><br>Elephant [Figure 31]. The image captures a serene scene at a riverbank where a man is riding on the back of a large elephant. The elephant, with its majestic gray skin, is partially submerged in the water, drinking from it. The man, dressed in casual attire, sits comfortably on a wooden bench strapped securely on the elephant’s back. The bench is adorned with colorful cushions for added comfort during the ride. The backdrop features lush greenery with trees lining the riverbank, adding to the tranquil atmosphere of the scene. The overall setting suggests a peaceful interaction between humans and nature.|

###### Table 6: Corresponding synthetic detailed and short captions for the examples shown in Figures 30 and 31. We highlight the objects described in the children nodes in dark blue and mark some erroneous descriptions in italic.

|Entity Captions|Flame—flame [Figure 30]. A bright yellow flame emanates from a blue lighter. The flame is intense and vibrant, with a distinct orange hue at the center. It appears to be burning steadily, casting a warm glow.<br><br>Messe—priest [Figure 30]. A priest dressed in traditional religious attire, which includes a red robe with gold trim, a white cassock, and a red sash. He holds a cross in his right hand.<br><br>Regalia—scepter 1 [Figure 30]. A gold scepter with a cross atop it. The scepter has intricate designs and patterns throughout its length.<br><br>Elephant—elephant [Figure 31]. A large elephant with a long trunk is seen walking through a body of water. Its skin appears rough and wrinkled, typical of elephants. The elephant has small tusks and large ears that are characteristic of this species.<br><br>|
|---|---|
|Relation Captions|Flame—flame/metal object [Figure 30]. The flame is positioned above the metal object. Messe—priest/kneeling figure [Figure 30]. The priest is standing in front of the kneeling figure. Regalia—crown/scepter [Figure 30]. The scepter is positioned next to the crown. Elephant—elephant/riverbank/trees [Figure 31]. The elephant is standing near the riverbank with trees in the background.<br><br>|
|Composition Captions<br><br>|Messe—kneeling figure [Figure 30]. Kneeling figure 1, positioned at the top right, appears to be in a state of prayer or reverence, while kneeling figure 2, located at the bottom left, seems to be in a similar posture, possibly indicating a shared moment of devotion or reflection.<br><br>Regalia—scepter [Figure 30]. Scepter 2, which is in the bottom left corner, has a golden handle with a blue gemstone at its center, while scepter 1, positioned above scepter 2, features a golden handle with a red gemstone at its center. Both scepters are ornate, with intricate designs and a regal appearance.<br><br>Elephant—riverbank [Figure 31]. Riverbank 1 is positioned above Riverbank 2, with Riverbank 2 located at the bottom of the composition.|
|Multi-Entity Captions|Messe—kneeling figure [Figure 30]. Both figures are depicted in a posture commonly associated with prayer or worship, suggesting a religious or spiritual context for their actions.<br><br>Regalia—scepter [Figure 30]. The gemstones in their handles add a touch of elegance and value to each scepter.<br><br>Elephant—riverbank [Figure 31]. They are situated near a body of water, which suggests a peaceful, natural setting.|

###### Table 7: Some example relational and region captions for the examples shown in Figures 30 and 31. We highlight the objects described in the children nodes in dark blue and mark some erroneous descriptions in italic.

- D. Algorithm details This appendix provides missing details about our architectures, CLIP training objective, and sampling algorithms.

##### D.1. Structure-aware hierarchical attention

We propose a simple text encoder architecture to incorporate structural information encoded in GBC graph along with node captions. Specifically, we present structure-aware hierarchical attention (SAHA) block which treats each caption as an individual sample, and introduces an additional cross-attention layer that enforces the captions to attend to their children.

Formally, we consider a caption graph GC = (C,EC) with vertices C = v∈V Cv and edges EC ⊆ C × C such that (C,C′) ∈ EC if and only if C ∈ Cu, C′ ∈ Cv, e = (u,v) ∈ E, and the label Le is included within the caption C. In words, each vertex in the graph represents a caption from a node of the original graph and there is an edge from one caption to another only if the second caption describes part of the first caption. After tokenization of the captions, we can map the edge labels to a set of token positions of the source caption, which we write as Pe. Then, the target caption annotates the source caption via the tokens at positions Pe. Therefore, we can simply consider cross-attention with queries from these tokens and keys and values from the target caption. We illustrate this idea in Figure 32, where we zoom in on the additional cross-attention layer (SACA) on the right side of the figure.

To express this via mathematical formula, we denote by NC the children of caption C in caption graph GC and write the

features of C in the input of our structure-aware cross-attention (SACA) layer as XC = [xC1 ,...,xCnC]. Then, the SACA layer maps each feature vector xCi to

′

′

′∈NC 1i∈P(C,C′) MHA(xCi ,XC

,XC

) min(1, C′∈NC 1i∈P(C,C′))

SACA(xCi ) = C

, (1)

where MHA implements the standard multi-head attention mechanism. Note that we average across the results from all the relevant captions that describe this token, as we show in the figure.

Our text encoder then stacks a number of SAHA blocks, effectively interleaving the vanilla self-attention layers that process, local, intra-caption information, with the SACA layers that process global, inter-caption information in a structure-aware manner. Furthermore, the model acts as a classic text encoder in the absence of edges in the graph, i.e. when EC = ∅.

As a side note, we highlight that with SAHA, information is only propagated from each node to its direct parent within a block. Consequently, the number of blocks must exceed the depth of the GC to ensure that information reaches the root node from all levels of the graph.

Complexity analysis. To estimate the computational complexity of our approach, we assume that the captions have a fixed sequence length n. Then, implementing SACA using masked cross-attentions between captions leads to a total complexity of O(|EC|n2). Additionally, we must account for the self-attention operations, resulting in a combined complexity of O(|C|n2 + |EC|n2). Provided that many of the graphs in our dataset are trees, we have |EC| = |C| − 1 and the complexity simplifies to O(|C|n2). In contrast, a naive approach that performs self-attention on the concatenated set of all captions would result in a significantly higher complexity of O(|C|2n2).

##### D.2. Multi-positive contrastive loss

To pair multiple positive captions to an image, we extend standard contrastive loss [68] into multiple-positive contrastive loss, as also considered in prior studies [19, 21]. Specifically, consider a batch of N images {Ii}Ni=1, where each image Ii is associated with Mi captions {Ti,j}M

- i
- j=1, we utilize the following loss function to account for multiple positive texts per image:

Mi

N

1 Z

S(Ii,Ti,j) S(Ii,Ti,j) + Nk=1,k̸=i M

, (2)

LI = −

log

- k
- l=1 S(Ii,Tk,l)

i=1

j=1

where S(I,T) = exp(cos(I,T)/τ), τ is a learnable temperature parameter, and Z = Ni=1 Mi is a normalizer. On the other hand, each caption still only has one paired image. Therefore, we use the standard contrastive loss on for text-to-image

|Caption a|
|---|

|Caption b|
|---|

|Image|
|---|

|Sky|
|---|

|Snowy Trees|
|---|

|Water Tower|
|---|

‧‧‧

| | | | |
|---|---|---|---|
| |S| | |

| | | | |
|---|---|---|---|
| | |S| |

Residual FeedForward PreNorm

+

+

| | |
|---|---|
|Image| |

| | |
|---|---|
|Sky| |

| | |
|---|---|
|Snowy Trees| |

| | |
|---|---|
|Water Tower| |

Mean

Multi Head Attention

‧‧‧

q k v

| | |
|---|---|
|S| |

###### MHA

MHA MHA

SACA Block

Query

| | | | | |
|---|---|---|---|---|
| |S| | | |

| | |
|---|---|
|Edge 1| |

| | |
|---|---|
|Edge 2| |

| | |
|---|---|
|Edge 3| |

S

Key-Value From Other Edges

‧‧‧

LayerNorm

Residual Self Attention PreNorm

S S

Caption a Caption b

| | |
|---|---|
|Edge (ca, cb)| |

‧‧‧

Image Sky Snowy Trees

Water Tower

- Figure 32: Illustration of the proposed SAHA block when applied to the graph shown in Figure 1. For the sake of simplicity, we assume here there is only one caption per node.

alignment:

N

1 Z

LT = −

i=1

Mi

S(Ii,Ti,j)

. (3)

log

N k=1 S(Ik,Ti,j)

j=1

##### D.3. Sampling algorithm for GBC-to-image generation

In Sec. 5, we did not provide details on how the negative prompts are handled and how the images are generated when only graph and text information from GBC are available. We explain them below.

Negative GBC prompts. Given a base negative prompt Cneg, we create a negative GBC prompt for each positive GBC prompt by preserving the same graph structure and bounding boxes, while replacing the caption of each node with Cneg. For each node, we then prepend the base negative prompt with the concatenated labels of the outgoing edges e1,...,ek from that node, resulting in "Le

,Cneg". During sampling, we apply the same cross-attention masks based on bounding boxes and graph structure for the negative GBC prompt. Putting together, the above ensures that we prevent Le from appearing outside the region associated with the vertex that edge e points to using the negative prompt, as mentioned in Sec. 5.

,...,Le

1

k

Sampling with only graph and text from GBC. Our overall procedure for sampling images from GBC without bounding box information is illustrated in Figure 33. We restrict ourselves to star graphs, and split the images into non-overlapping segments, each corresponding to a distinct node, based on cross-attention scores. Concretely, for each set of tokens Pe within the global prompt that maps to a child caption, we store the cross attention scores Ae ∈ R(w×h) that averages across all the tokens in Pe and across all the cross-attention layers of the same query dimension (w × h). We focus especially on the lower-resolution cross-attention maps—for SDXL with an output resolution of 1024 × 1024, this corresponds to w = h = 32.

n, where n is the number of leaves in the star graph. For each of these maps Ae, we regard it as an image of size w × h and apply Felzenszwalb’s graph-based image segmentation algorithm [24] with the default hyperparameters of skimage [81] to get the image segments Se = S1e,...,Sme

At each sampling step, we then get a set of cross-attention maps Ae

,...,Ae

1

.

e

We then combine the segmentation maps to form S1,...,Sm, where two image patches belong to the same segment if and only if they appear together in the same segment in each of the segmentation maps Se

n. Finally, we merge the segments to get S1′,...,Sn′ +1 by assigning each segment Si to one of the n + 1 vertices. This is achieved by using Otsu’s method [63] to determine a threshold for each leaf, based on the average cross-attention scores of each segment across both image patches and relevant text tokens. If a segment is initially assigned to multiple leaves, we retain only the assignment to the leaf with the highest normalized score (all average scores are normalized to the range [0,1]). Segments unassigned to any leaf are instead assigned to the root node.

,...,Se

1

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Figure 33: To sample images from GBC without bounding box information, we infer a segmentation map that maps image patches to graph vertices using cross-attention scores. We first run the sampling process for half of the total number of time steps to get the initial map. We then restart the process with the provided map, with the map being updated dynamically during the second half of the sampling process.

The aforementioned segmentation provides a cross-attention mask for each time step, which we apply to both the lowerresolution (32 × 32) and higher-resolution (64 × 64) cross-attention layers within the UNet (assuming an output resolution of 1024 × 1024). To maintain stability in the mask application, we further employ an EMA version of the masks with a momentum of 0.9. To generate the initial mask, we first run the sampling process using only the global prompt for T/2 steps, with the EMA mask calculated from step T/4 to T/2. We then restart the process with the same initial noise, applying the obtained EMA mask during the first half of sampling. During the second half, we update the EMA mask dynamically and apply it to the UNet at each step.

- E. Experimental details for CLIP training This appendix provides additional details on the experiments presented in Sec. 4.

##### E.1. Data filtering

For the computation of CLIP score, we split any caption that contains more than 77 tokens into individual sentences, compute the score for each of these sentences, and compute the average of these scores. Then, we start by filtering out images whose short synthetic captions have CLIP scores that are lower than the 5% quantile. After this, we consider three filtering strategies depending on the annotation formats.

Long caption. In this case, we just further filter out a portion of original captions and long captions with the lowest CLIP scores (by considering the 5% quantiles from the non-filtered dataset).

GBC. Naive CLIP filtering and tokenizer truncation could break the graph structure as some of the edge labels would not appear in the captions of its source node anymore after these operations. We address this issue by filtering out the captions following the reverse of a topological ordering of the graph, drop a node along with its in edges when all its captions and children get filtered, and otherwise, if necessary, add bag-of-words captions that collects edge labels from the remaining out edges of a node to ensure all these labels still appear in some captions of this node. Moreover, we split the captions whose length are longer than 77 tokens into concatenations of sentences that fit within this limit, and drop any caption which contains sentences that are of more than 77 tokens.

Short and Region. We follow the strategy mentioned in GBC, but use selected types of captions. Moreover, bag-of-words captions are not used.

We remark that the filtering procedure is only applied to the training set, and not the GBC test set.

| |Short Long Region GBC-captions GBC-concat GBC-graph|
|---|---|
|Training time (hr)|22.7 19.7 29.9 38.9 20.3 42.0|

Table 8: CLIP model training time for 45,000 iterations with different annotation formats.

Hyperparameters Values Data augmentation RRC Crop size 224×224 Train iterations 45k Global batch size 4,096 Optimizer AdamW Min / max learning rate {1e-6, 1e-3} LR. decay schedule type Cosine Warmup iterations 1,000 Weight decay rate 0.05 EMA factor 0.9995

Table 9: Hyperparameters for CLIP model training. RRC stands for RandomResizedCrop

Hyperparameters Values Data augmentation RRC Crop size 512×512 Train iterations 160k Global batch size 16 Optimizer AdamW Peak learning rate

[5e-4, 2e-4, 1e-4, 7e-5, 5e-5] LR. decay schedule type Polynomial Warmup iterations 1,500 Weight decay rate 0.01

Table 10: Training hyperparameters for semantic segmentation experiments on ADE20k. RRC stands for RandomResizedCrop.

##### E.2. Dynamic batch size

Given the varying sizes of our graph, setting a fixed number of images per batch could result in out-of-memory errors unless we opt for a conservatively small batch size. To overcome this challenge, we implement a dynamic batching strategy for the setups where the number of captions per image is in principle unbounded. This encompasses notably region, GBC-captions, and GBC-graph. With this strategy, we ensure that the number of captions, and, in the case of GBC-graph, the number of edges, that are included in each batch do not exceed a certain limit. In this regard, the batch size that we report in Sec. 4 is actually just an upper bound on the number of images included in each batch. More specifically, we set this limit based on the number of average captions/edges per image in the filtered dataset. For example, for GBC-captions and GBC-graph we have in average 17.61 captions per graph. We thus set the limit on caption number to 18 × 64 = 1152 on each GPU (as mentioned in Appx. E.4, we use 64 GPUs for most of our experiments, which gives a batch size of 64 per GPU).

##### E.3. Hyperparameters for CLIP training

We used a consistent set of hyperparameters for all model training runs, as detailed in Table 9. The sole exception is training with original CC12M captions, where we used a larger batch size of 8,192 to ensure the model sees a comparable number of texts as during training with both short synthetic and original captions. For this specific setup with the larger batch size, we reported evaluation results from the EMA checkpoint at the end of epoch 15, for it achieving the best performance among the evaluated checkpoints. For GBC-graph, we drop the edges with probability 0.5 so that the model also learns how to match images with short captions.

##### E.4. Computation cost

We train all CLIP models on A100-80G GPUs. As training with different annotation formats requires varying size of GPU memory, we use different total numbers of GPUs to ensure the same batch size. Specifically, we utilize 16 GPUs for training with Short captions, and utilize 64 GPUs for training with all other annotation formats. We list the corresponding time required for training with different annotation formats in Tab. 8.

##### E.5. Evaluation details

Our evaluation uses the validation set of ImageNet-1k [69] and the test sets of Flickr30k [64] and MS-COCO [55]. For SugarCrepe [36] we report the average performance across all variants. As for ShareGPT4V [104], we use a subset of size

|Annotation<br><br>|T2I I2T|
|---|---|
|Short Long GBC-captions GBC-concat GBC-graph|71.5 78.1<br>72.9 80.1 86.4 87.3<br>73.3 79.7 85.2 85.9<br>|

Table 11: Image and text retrieval performance on GBC test when using max CLIP score over all the captions.

|Annotation<br><br>|DCI-Long T2I I2T<br><br>|DCI-concat T2I I2T|ShareGPT4V-15k T2I I2T|
|---|---|---|---|
|Long GBC-captions GBC-concat|53.6 53.3 42.5 43.3 51.4 52.3|63.3 65.8<br>64.3 63.8 69.0 70.8<br>|93.4 93.9 78.7 82.1 89.5 91.4|

Table 12: Image and text retrieval performance on DCI and a 15k subset of ShareGPT4Vcap100k of our models trained on longer captions. We also include GBC-captions that by design can only handle short captions as a baseline for comparison.

15,295 from ShareGPT4V-cap100k. These images were also used for LLaVA training.8 When each image is paired with multiple captions, we only select one of them. The evaluation setups with ADE20K and DCI are more involved, as we explain below.

Evaluation on ADE20K. We evaluate the quality of CLIP models’ image encoder for dense prediction tasks like image segmentation by performing full finetuning on ADE20k [104] dataset. We follow the same setup as described in [82, 83] where we use a ViTDet style feature pyramid network with UperNet [87] head. All models were trained using the MMSegmentation library [14]. We sweep through peak learning rate for all the results reported in the paper and the ranges are listed in Table 10.

Evaluation on DCI. We perform text-to-image and image-to-text evaluations on DCI [80] using either long captions or concatenated captions. The long captions are marked as extra_caption in the released DCI dataset. We filter out samples with empty long captions, resulting in a subset of 7,602 images for evaluation with long captions. Regarding evaluation with concatenated captions, we leverage the full set of 7,805 images. We retain masks containing summary captions (these are masks with bounding boxes larger than 224 × 224). If the human-annotated caption contains fewer than 77 tokens and is longer than the first summary caption, we use it. Otherwise, we use the first summary caption. For concatenation, we follow the Breadth-First Search (BFS) order based on the provided tree structure between the masks.

#### F. Additional results and experiments for CLIP training

In this appendix, we present additional experiments that we performed but did not present in the main paper due to space constraints.

##### F.1. Retrieval with multiple captions using maximum CLIP score

An alternative to the mean CLIP score we considered in Tab. 3 is to take the maximum, for which we report the results in Tab. 11. Compared to taking the average, using the maximum is more robust to low CLIP scores, and thus gives better results when the model is not trained to match the image with all local captions, as with Short, Long caption, and GBC-concat. 9 Nonetheless, despite these differences, the overall retrieval performance still significantly lags behind that achieved using a single caption.

##### F.2. Retrieval with long captions

To complement the results presented in Tab. 2, we evaluate the retrieval performance of our extended context models on datasets with richer annotations. We focus specifically on ShareGPT4V [10], which offers GPT-style detailed captions closely resembling those obtained from LLaVA, and DCI [80], containing human-annotated detailed and region captions. The latter allows us to perform retrieval using either detailed captions or concatenated short captions, as we did in Section 4.4. Our results shown in Tab. 12 demonstrates that the close caption distribution with ShareGPT4V effectively enables strong retrieval results for models trained on our long captions. However, potentially due to the distribution shift, all the models perform badly on DCI retrieval with long captions. In this setup, using concatenated captions for training and retrieval significantly outperformed other baselines, indicating the broader benefit of the concatenation approach.

8https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain Accessed: 2024-05-14 9In GBC-graph, the entire graph is encoded, so we do pair the image with all the captions during training.

|Annotation|Hyperparameter<br><br>Epoch Batch size # Tokens<br><br>|Evaluation results<br><br>ImageNet Flickr COCO<br><br>ShareGPT4V<br><br>DCIconcat<br><br>GBC test|
|---|---|---|
|Short|10 4,096<br><br>77 512<br><br>10 28 16,384 77 40<br><br>|38.8 64.8 38.7 79.1 57.5 87.8<br>39.0 64.7 39.3 86.7 56.4 89.7 33.2 59.1 34.7 74.0 52.3 83.4<br><br>40.0 67.4 38.7 80.6 58.1 88.6 39.0 65.7 37.3 79.9 57.8 88.6<br><br><br>|
|GBC-captions GBC-concat GBC-graph|10 4,096 77|40.8 70.0 43.0 80.4 64.1 91.2 39.0 66.1 40.0 90.5 69.9 94.8 38.4 67.5 40.6 78.0 61.3 96.0<br><br>|

Table 13: Comparative performance across various benchmarks when we perform CLIP training on short captions with different hyperparameters. For ease of reference, we also include the results from the methods that use GBC annotations. We report the average image and text Recall@1 for all retrieval benchmarks. Specifically, as explained in Sec. 4.4, we perform retrieval using various annotation formats for GBC test. We thus report here the average of the highest image and text Recall@1 scores. The number of iterations is consistently set at 45,000, corresponding to 20 epochs with a batch size of 4,096 and 76 epochs with a batch size of 16,384.

##### F.3. Matching compute resource for training with short captions

All our models presented in Sec. 4 used 8 nodes for training, except for the models trained on short captions, which only used 2 nodes. This raises the question of whether the performance gap could be bridged by providing more computational resources to this setup. To address this, we specifically considered two modifications that would naturally necessitate using more nodes for training with short captions: (i) extending the context length to 512, as done for training with Long and GBC-concat captions, and (ii) using a batch size that is four times larger, i.e. a batch size of 16,384 instead of 4,096. All other hyperparameters remained unchanged. We then trained the models on 8 nodes, each with 8 GPUs, as in the other setups, which resulted in training times of 18 and 48 hours for the two modifications respectively. The evaluation results are presented in Tab. 13.

Training with extended context length. Provided that the models are only trained with short captions, we do not expect any tangible benefit from extending the context length. Yet, surprisingly, while this is indeed the case for classic benchmarks such as ImageNet, Flickr, and COCO, we do observe a significant performance boost on ShareGPT4V retrieval, suggesting that the longer context length is still beneficial for retrieval with long caption even though the model is not explicitly trained for this task. On the other hand, we do not observe any benefit when evaluated using concatenated caption from DCI. Finally, we also get a slight performance improvement on GBC test, and it turns out this improved performance is achieved by performing retrieval using the long caption. This is in line with the performance gain that we observe for the ShareGPT4V benchmark.

Training with larger batch size. More interestingly, CLIP is known to perform better when trained with a large batch size, so we might be able to bridge the performance gap by simply including more images and captions in each batch. To enable a fair comparison for this setup, we report evaluation results from three checkpoints at varying training stages in Tab. 13. These checkpoints are chosen to align with key training milestones.

- • Number of images seen: We consider the EMA checkpoint at the end of epoch 10 to align the number of images seen.
- • Number of iterations: We include the EMA checkpoint at the end of epoch 40 to compare models at a fixed number of training iterations.
- • Best performing checkpoint: Additionally, we report results for the EMA checkpoint at the end of epoch 28, as it gives the best performance among all evaluated checkpoints (see Figure 34).

As we can see from the table, while the use of a larger batch size indeed leads to better performance on ImageNet and Flickr, the results still lag behind those achieved with GBC-captions. This discrepancy underscores the importance of including multiple captions per image to enhance performance.

|Annotation|ImageNet Flickr-1k MSCOCO-5k SugarCrepe Average Drop<br><br>|
|---|---|
|Short Long GBC-captions|38.8 → 35.2 64.8 → 61.0 38.7 → 36.7 76.0 → 74.4 -2.75<br>39.6 → 30.5 65.8 → 56.8 40.1 → 33.5 77.0 → 74.0 -6.93<br>40.8 → 31.9 70.0 → 58.3 43.0 → 33.2 76.7 → 73.3 -8.45<br>|

Table 14: Performance degradation across different annotation types when switching from multi-positive contrastive loss to standard contrastive loss with randomly sampled positive captions. For Flickr-1k and MSCOCO-5k we report the average image and text Recall@1.

|Annotation<br><br>|Flickr-1k T2I I2T<br><br>|MSCOCO-5k T2I I2T|DCI-concat T2I I2T<br><br>|ImageNet|SugarCrepe|ADE20K|
|---|---|---|---|---|---|---|
|Short Region GBC-relation GBC-captions|56.3 73.2 58.3 76.6 60.4 76.5 60.6 79.3<br><br>|30.7 46.7<br>31.5 49.1 34.8 52.5 34.1 51.9<br><br><br>|57.5 57.5<br><br>61.8 61.5<br>62.0 61.4 64.1 63.4<br>|38.8 38.5 41.5 40.8<br><br>|76.0 75.6 76.4 76.7<br><br>|42.0 43.5 44.5 45.0<br><br>|

Table 15: Comparative performance on various existing benchmarks when trained using different subsets of GBC-captions.

##### F.4. The importance of multi-positive contrastive loss

We next look into the influence of the objective function when an image is paired with multiple captions. Instead of employing the multi-positive contrastive loss introduced in Appendix D.2, we can use a standard contrastive loss with a single randomly sampled caption paired with each image. Tab. 14 presents the evaluation results for both the models trained with the original objective (left side of the arrow), and this new, sampled, objective (right side of the arrow).

The table clearly shows a performance decline across all the considered annotation formats and benchmarks when sampling is applied, as also observed by Doveh et al. [19] and Fan et al. [21]. The performance drop is particularly important when the captions vary significantly (e.g., long versus short captions, or image versus region captions), and when many captions are involved. More surprisingly, this alternative loss does not lead to improvement but rather to performance degradation when we increase the number of captions paired with each image. We conjecture this is because the additional captions that we consider here are less relevant for these specific benchmarks, leading to a worse performance when they are forced to be treated as positive in the sampled objective.

Overall, these results confirm the importance of our multi-positive contrastive loss in leveraging the presence of multiple captions for an image.

##### F.5. Impact of caption type on CLIP training

To further highlight the value of relation and composition captions from GBC, we trained a CLIP model using only these captions alongside short image captions. As shown in the third row of Tab. 15, these captions, despite being more than twice as scarce as region captions, not only provided a larger performance gain than using only region captions, but sometimes even enabled the model to achieve comparable or better performance than using all captions combined. This underscores the significant benefit of the relational captions from GBC datasets.

Looking closely, we note that region captions primarily benefit retrieval and dense prediction tasks, while relation and composition captions improve performance across the board. While using all captions remains the best approach for most benchmarks, the marginal improvement from region captions hints at the potential for more efficient training with these captions through alternative training objectives.

##### F.6. Impact of the underlying graph on retrieval

In this part, we investigate how much GBC-graph relies on the underlying graph structure for retrieval. For this, we probe the performance of our model when the graph is modified either in the mapped tokens or in the connectivity patterns. In terms of the mapped tokens, we consider

• Last token: For any edge from a caption C to another caption C′, we mapped the information of C′ to the last token before

|Annotation<br><br>|Short<br><br>T2I I2T|GBC-graph Star graph Line graph<br><br>Groundtruth Last token Random token T2I I2T T2I I2T T2I I2T T2I I2T T2I I2T<br><br>|
|---|---|---|
|GBC-graph|84.8 85.7|95.9 96.1 93.1 93.6 95.2 95.7 94.7 95.0 92.2 92.4<br><br>|

- Table 16: Image and text retrieval performance on the GBC test set when the model is trained using GBC-graph and evaluated across various underlying graph structures.

|Annotation<br><br>|ImageNet Flickr-1k MSCOCO-5k SugarCrepe ShareGPT4V-15k GBC test|
|---|---|
|CC12M|37.1 50.5 27.9 39.4 47.2 49.5<br><br>|
|Short Long Region<br><br>GBC-captions GBC-concat GBC-graph|37.5 62.0 36.1 74.5 77.3 86.9<br>38.5 64.5 38.4 75.8 93.5 95.5<br><br><br>40.3 68.6 40.8 76.1 78.9 91.8<br><br>41.3 70.6 43.1 76.4 80.1 91.9<br><br><br>38.2 63.4 37.1 74.9 89.8 96.1<br><br>39.7 68.1 40.8 75.3 78.2 96.2<br><br><br>|

- Table 17: Comparative performance on various existing benchmarks when trained using different annotation schemes. Unlike the other tables that report performance for EMA checkpoints, this table presents the performance at the final non-EMA checkpoints obtained from the end of training.

the summary token in C.10

• Random token: For each edge, we randomly map the information to one token in the source caption. As for the connectivity pattern, we investigate

- • Star graph: All the captions are mapped to the short image synthetic caption.
- • Line graph: We map each caption to its next caption in a list (ordered as in GBC-concat following the BFS order), with the short image synthetic caption being the first in the list.

The results are shown in Tab. 16. Since random-token mapping consistently leads to better result than last-token mapping, we only report results for this in the case of star graph and line graph. First of all, we observe that no matter which graph is given, we always achieve better performance than retrieval with only short caption, suggesting that the model is always able to exploit the additional captions to some extent. Furthermore, employing random-token mapping, whether with the groundtruth graph topology or the star graph, yields performance that closely matches that of using the groundtruth graph with correct mapping (interestingly, when using star graph the performance is also very close to that obtained with GBC-concat, see Tab. 3). This suggests that the specific retrieval task we are examining is not highly dependent on the provided mapping and topology. However, we do believe the mapping and topology could play a significant role in other tasks or when more fine-grained distinctions between images are necessary.

##### F.7. Evaluating at non-EMA checkpoints

For the sake of completeness, we also perform evaluation on the non-EMA checkpoints, with results shown in Tab. 17 and Figure 35. Comparing Figure 34 with Figure 35, we see that while EMA checkpoints may experience a drop in performance during later training stages, non-EMA checkpoints typically exhibit best performance at the final training checkpoint. Consequently, our evaluations in Tab. 17 are based on these last checkpoints. From the evaluation results, we observe a similar trend in the performance comparison of annotation formats with non-EMA checkpoints as with EMA ones, confirming the validity of our previous claims. Finally, we also note that the use of larger batch size when training with short captions is only beneficial when we consider EMA checkpoints.

10We also experimented with mapping the information to the summary token but this completely destroys the performance.

ImageNet

Flickr-1k

0.7

0.40

AverageImageandTextRecall@1

| |
|---|

| |
|---|

0.6

0.35

| |
|---|

| |
|---|

0.30

0.5

Accuracy

0.25

0.4

0.20

Short Short [bs16384]

Short Short [bs16384]

0.3

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.15

Long

Long

0.2

Region

Region

0.10

GBC-captions

GBC-captions

0.1

GBC-concat

GBC-concat

0.05

GBC-graph

GBC-graph

10000 20000 30000 40000 # Iterations

10000 20000 30000 40000 # Iterations

MSCOCO-5k

GBC test

1.0

0.4

| |
|---|

AverageImageandTextRecall@1

AverageImageandTextRecall@1

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.8

| |
|---|

| |
|---|

0.3

0.6

Short

Short

0.2

0.4

Short [bs16384]

Short [bs16384]

Long

Long

Region

Region

0.1

0.2

GBC-captions

GBC-captions

GBC-concat

GBC-concat

GBC-graph

GBC-graph

0.0

10000 20000 30000 40000 # Iterations

10000 20000 30000 40000 # Iterations

- Figure 34: Benchmark performances on ImageNet, Flickr-1k, MSCOCO-5k, and GBC test for EMA checkpoints of models trained with different annotations / hyperparameters. For GBC test we use different formats for retrieval at test time and average the highest scores that are respectively obtained for text-to-image and image-to-text retrievals.

10000 20000 30000 40000 # Iterations

0.10

0.15

0.20

0.25

0.30

0.35

0.40

Accuracy

| |
|---|

| |
|---|

| |
|---|

ImageNet

Short

Short [bs16384]

Long

Region

GBC-captions

GBC-concat

GBC-graph

10000 20000 30000 40000 # Iterations

0.2

0.3

0.4

0.5

0.6

0.7

AverageImageandTextRecall@1

| |
|---|

| |
|---|

| |
|---|

Flickr-1k

Short

Short [bs16384]

Long

Region

GBC-captions

GBC-concat

GBC-graph

10000 20000 30000 40000 # Iterations

0.10

0.15

0.20

0.25

0.30

0.35

0.40

0.45

AverageImageandTextRecall@1

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

MSCOCO-5k

Short

Short [bs16384]

Long

Region

GBC-captions

GBC-concat

GBC-graph

10000 20000 30000 40000 # Iterations

0.4

0.5

0.6

0.7

0.8

0.9

AverageImageandTextRecall@1

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

GBC test

Short

Short [bs16384]

Long

Region

GBC-captions

GBC-concat

GBC-graph

- Figure 35: Benchmark performances on ImageNet, Flickr-1k, MSCOCO-5k, and GBC test for non-EMA checkpoints of models trained with different annotations / hyperparameters. For GBC test we use different formats for retrieval at test time and average the highest scores that are respectively obtained for text-to-image and image-to-text retrievals.

- G. Experimental details for text-to-image generation This appendix provides additional details on the experiments presented in Sec. 4.

- G.1. Text-to-GBC We provide below detailed information on data processing and model training for our GBC prompt generation model.

Data format and dataset. As illustrated in Figure 3, each node in our GBC graph is encoded in plain text as follows

Node #<id> <name> type: <type> is_leave: <True|False> desc: <description> parents: #<parent_id>(<parent_name>: <name>) bbox: <bbox>

The elements of the above format are defined as:

- • <id> is a unique numerical identifier for the node.
- • <name> is the node’s unique string identifier as assigned in our dataset.
- • <type> indicates the node type (image, entity, composition, or relation).
- • <is_leave> specifies whether the node is a leaf node.
- • <description> contains the textual description of the node content. For image node we use the short caption while for the remaining we use the first caption stored in the node.
- • <parent_id> and <parent_name> identify the parent node(s).
- • <bbox> represents the bounding box coordinates.

Our training dataset is derived from GBC10M by processing each graph G = (V,E) into a sequence of node descriptions encoded in the aforementioned format. Moreover, we ensure that this node sequence forms a topological order of the original graph so that the hierarchical structure of the graph is respected. The numerical identifier <id> of each node naturally corresponds to its position in this sequence. We also remove all relation nodes and composition captions to align with our GBC-to-image generation pipeline.

Training Configuration. Our text-to-GBC model is built upon TIPO-200M [92]. This lightweight architecture ensures efficient processing while maintaining high-quality graph generation capabilities. We train the model using the scheduler-free AdamW optimizer [16, 58] for one epoch, which amounts to approximately 20,000 steps with a fixed global batch size of 512. The learning rate is set to 5e-5, with a warmup period of 100 steps and a weight decay of 0.1. We use β1 = 0.9, β2 = 0.99 for AdamW, and limit the context length to a maximum of 4,096 tokens.

- G.2. GBC-to-image

For sampling from SDXL, we use Euler sampler with T = 24 sampling steps and a cfg scale of 6. All prompts are either truncated or padded to 77 tokens. The base negative prompt is set to "low quality, worst quality".

Regarding Figure 33, we observe that encoding prompts independently performs better for the first example, while encoding them with contextual information from parent prompts works better for the second example. Accordingly, we adopt the respective strategy for these two examples when only prompt and graph information is provided.

- H. Additional results for text-to-image generation In this appendix, we present additional results for text-to-image generation experiments.

##### H.1. GBC-to-image

Unbiased generations from different methods for the prompts considered in Figure 4 are shown in Figures 36 to 38. These results confirm our observations in Sec. 5.

- In Figure 36, we additionally observe that restricting banana and apple from appearing outside the bounding boxes could

lead to the appearance of ambiguous or incomplete objects. These artifacts seem to be the result of the model attempting to generate apples or bananas but being constrained by the bounding box, causing the object to morph into unnatural or distorted forms. Given how different text tokens and image patches exchange information in text encoder and in the self-attention and convolutional layers of UNet, this leakage of information seems unavoidable with our current approach.

- In Figure 37, we see that vanilla SDXL struggles to generate a cat and a dog with prompt concatenation. Similarly, with the

naive approach that only leverages bounding box information with forward cross-attention control, the model still fails to generate one of the two animals with high probability.

##### H.2. Text-to-image with GBC as middleware

We next study the possibility of combining our text-to-GBC and GBC-to-image pipeline. For GBC-to-image, we use the algorithm that exploits all information from GBC and encode each prompt with parent prompts as contextual information. In Figure 39, we first show that our prompt generation model is able to generate complex graphs from a simple prompt. However, we find the our image generation algorithm fail to generate images that adhere to such complex GBC prompt for the following reasons.

- • The generated GBC is not perfect, potentially due the small size of our prompt generation model. While each node’s description is correct for their corresponding object, inconsistencies can arise between the descriptions of different nodes. For instance, in the last example of Figure 39, a cat is described simply as an animal rather than as a mechanical cat. Moreover, the generated bounding boxes can also contradict the descriptions. We believe the latter could be improved by encoding bounding boxes differently, as for example done in [77].
- • The image generation algorithm struggles with bounding boxes that have large overlapping areas when they are on disjoint paths of the graph. This limitation arises because the algorithm relies on the graph hierarchy to manage overlapping bounding boxes and determine their priority. As far as we are aware, no existing training-free approach can effectively handle highly complex overlapping bounding boxes of this kind.
- • Not all important objects from the seed prompt are guaranteed to receive additional descriptions from our prompt generation model. Combined with the previously mentioned limitation, this can result in certain objects failing to appear in the generated image. For example, in the third example of Figure 39, the frog is absent from the final images.

To circumvent the above limitations while still benefiting from the advantages of using GBC as middleware, we restrict the process to simpler graphs and adjust the prompt generation to ensure the model explicitly produces detailed descriptions for a select number of key objects. The results, shown in Figure 40, demonstrate that the generated images align more closely with both the seed prompt and the intermediate GBC. However, some failure cases remain, such as the absence of a frog in two images for the third example or the generation of a mechanical dog instead of a mechanical cat in two images for the forth example.

GBC as middleware enhances image diversity. It has been reported that naively prompting the text-to-image models could often result in images with low diversity from a single prompt [43, 47]. Using additional middleware is an effective way to address this issue as it enables the model to generate a diverse set of intermediate representations from the same seed prompt [31, 92]. We verify that GBC as middleware can indeed also improve the diversity of generated images in Figures 41 and 42. On the downside, as noted previously, this approach may slightly compromise prompt adherence. Interestingly, for the fourth example, SDXL fails to generate a mechanical cat using only the plain text prompt. In contrast, employing GBC that provides additional descriptions of the mechanical cat as middleware successfully ensures its inclusion in the generated images.

[Figure 40]

(a) Generated with concatenation of text prompts.

[Figure 41]

(b) Generated with text prompts and bounding box information.

[Figure 42]

(c) Generated with text prompts, graph, and bounding box information. Prompts are encoded independently.

[Figure 43]

(d) Generated with text prompts, graph, and bounding box information. Prompts are encoded with parent prompts as context information.

[Figure 44]

(e) Generated with text prompts and graph information. Prompts are encoded independently. Figure 36: Non cherry-picked generations for the first example presented in Figure 4.

[Figure 45]

(a) Generated with concatenation of text prompts.

[Figure 46]

(b) Generated with text and image prompts and bounding box information.

[Figure 47]

(c) Generated with text and image prompts, graph, and bounding box information. Prompts are encoded independently.

[Figure 48]

- (d) Generated with text and image prompts, graph, and bounding box information. Prompts are encoded with parent prompts as context information.

[Figure 49]

- (e) Generated with text and image prompts and graph information. Prompts are encoded with parent prompts as context information. Figure 37: Non cherry-picked generations for the second example presented in Figure 4.

[Figure 50]

(a) Generated with concatenation of text prompts.

[Figure 51]

(b) Generated with text prompts and bounding box information.

[Figure 52]

(c) Generated with text prompts, graph, and bounding box information. Prompts are encoded independently.

[Figure 53]

(d) Generated with text prompts, graph, and bounding box information. Prompts are encoded with parent prompts as context information. Figure 38: Non cherry-picked generations for the third example presented in Figure 4.

[Figure 54]

###### Figure 39: Example generations with GBC as middleware, presented without cherry picking. Our prompt generation model can generate complex graphs when used naively. This, however, cannot be handled by our image generation algorithm, leading to unsatisfying results. For better visualization, bounding boxes are shown for only one of the 6 generated images. Some generated prompts are provided in Tab. 18.

[Figure 55]

###### Figure 40: Example generations with GBC as middleware, presented without cherry picking. Here, we manipulate the generation process to start from a certain object (lamp, turtle, frog, or mechanical cat). For better visualization, bounding boxes are shown for only one of the 6 generated images. Some generated prompts are provided in Tab. 18.

|Example Generated Prompts for Figure 39<br><br>|Lamp. A vintage table lamp with a yellow shade. The lamp has a sturdy base with intricate detailing. It features a classic design with a metal frame and a ribbed texture.<br><br>Bookshelf. A large wooden bookshelf filled with various books. The bookshelf has multiple shelves, each holding an assortment of books with different sizes and colors. Some books are stacked horizontally while others are arranged vertically.<br><br>Pond. A serene pond with a wooden dock extending into it. The water is calm, reflecting the surrounding greenery. A few lily pads are scattered across the surface, adding to the tranquil atmosphere.<br><br>Dock. A wooden dock extends into a body of water. The dock has a sturdy structure with visible planks and supports. A small boat is docked at one end, with a single oar resting against it.<br><br>Mushrooms. The mushrooms are likely part of a larger forest ecosystem, contributing to the overall environment.<br><br>Steam. The image showcases a serene stream meandering through a lush forest. The stream is flanked by verdant greenery, with various plants and trees lining its path. The water appears calm and clear, reflecting the surrounding foliage. The stream’s banks are adorned with fallen leaves, hinting at the season being autumn.<br><br>Cat. The image showcases a stylized depiction of a cat. It is characterized by a sleek body with smooth muscles and a distinctive mask-like pattern on its head. The cat’s ears are pointed upwards, and it has a small, round tail. Its eyes are closed, giving it a serene appearance.<br><br>Wooden door. A wooden door with a visible grain pattern and a slightly worn appearance. It has a rectangular shape with a central panel flanked by two smaller panels. A small window or peephole is located near the top center.|
|---|---|
|Example Generated Prompts for Figure 40|Lamp. A vintage table lamp with a gold base and a white lampshade. The lamp has a classic design with a curved neck and a floral pattern at the base.<br><br>Leather armchair. A vintage leather armchair with a rich brown color. The chair has a plush seat cushion and armrests. It features a tufted backrest with decorative buttons, adding to its classic design.<br><br>Turtle. A large turtle with a rough, textured shell is seen resting on a log. Its head is slightly raised, and it appears calm and relaxed.<br><br>Frog. A frog with vibrant green skin sits perched on a mushroom. Its large eyes are wide open, and it appears to be looking directly at the camera.<br><br>Mechanical cat. A mechanical cat with intricate gears and cogs visible on its body. The cat has a sleek design with a streamlined head and tail. Its eyes are glowing with a yellowish hue.<br><br>Head. The image showcases a close-up view of a head, which appears to be that of a mechanical or robotic robot. It has a sleek, metallic finish with a shiny surface that reflects light. The head features two eyes that are glowing with a yellowish hue, giving it a somewhat eerie or futuristic appearance. There are no visible mouth or nose, and no other distinguishable facial features are apparent.|

###### Table 18: Examples of generated prompts that we obtain in Figures 39 and 40 with our prompt generation model. We highlight the objects described in the children nodes in dark blue.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Figure 41: Example generations from the seed prompts in Figures 39 and 40 when SDXL is used naively with Euler sampling.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Figure 42: Example generations from the seed prompts in Figures 39 and 40 when GBC is used as middleware with the manipulation introduced in Figure 40.

#### I. Image attributions

All the images that we show in this paper come from Wikimedia Commons. We provide in Tab. 19 the exact source urls and license for each of the images. The urls to the CC BY-SA 2.0, CC BY-SA 3.0, and GFDL 1.2 licenses are respectively https: //creativecommons.org/licenses/by-sa/2.0, https://creativecommons.org/licenses/bysa/3.0/, and https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt.

###### Image Source URL License

- Figure 1 https : / / commons . wikimedia . org / wiki / File:Tartu_raudteejaama_veetorn,_2010. JPG

CC BY-SA 3.0

- Figure 2 https : / / commons . wikimedia . org / wiki / File:Eiffel_Tower_from_north_Avenue_ de_New_York,_Aug_2010.jpg

CC BY-SA 3.0

Figure 4

Public domain

Corgi https : / / commons . wikimedia . org / wiki / File:Fawn_and_white_Welsh_Corgi_puppy_ standing_on_rear_legs_and_sticking_ out_the_tongue_(cropped).jpg

CC BY-SA 2.0

Cat https : / / fr . m . wikipedia . org / wiki / Fichier:Orange_tabby_cat_sitting_on_ fallen_leaves-Hisashi-01A.jpg

- Figure 30

Flame https : / / commons . wikimedia . org / wiki / File:Flametest--Na.swn.jpg

CC BY-SA 3.0

Messe https : / / commons . wikimedia . org / wiki / File:Messe_mit_Wandlungskerze_Beuron. jpg

Public domain

Regalia https : / / commons . wikimedia . org / wiki / File:Crown,_sceptre,_orb_%26_key_of_ the_King_of_Sweden_2014.jpg

Public domain

- Figure 31 https : / / commons . wikimedia . org / wiki / File:Indian-Elephant-444.jpg

GFDL 1.2

Table 19: Source URLs and licenses of the images shown in this paper.

