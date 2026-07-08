## MoDE: CLIP Data Experts via Clustering

Jiawei Ma1∗,2 Po-Yao Huang1 Saining Xie3 Shang-Wen Li1 Luke Zettlemoyer1,4 Shih-Fu Chang2 Wen-Tau Yih1 Hu Xu1+ 1FAIR, Meta 2Columbia University 3New York University 4University of Washington

[Figure 1]

# arXiv:2404.16030v1[cs.CV]24Apr2024

### Abstract

Tree guard to stop the cats The tiger reaches up to a tree trunk in a wooded area

[Figure 2]

The success of contrastive language-image pretraining (CLIP) relies on the supervision from the pairing between images and captions, which tends to be noisy in webcrawled data. We present Mixture of Data Experts (MoDE) and learn a system of CLIP data experts via clustering. Each data expert is trained on one data cluster, being less sensitive to false negative noises in other clusters. At inference time, we ensemble their outputs by applying weights determined through the correlation between task metadata and cluster conditions. To estimate the correlation precisely, the samples in one cluster should be semantically similar, but the number of data experts should still be reasonable for training and inference. As such, we consider the ontology in human language and propose to use finegrained cluster centers to represent each data expert at a coarse-grained level. Experimental studies show that four CLIP data experts on ViT-B/16 outperform the ViT-L/14 by OpenAI CLIP and OpenCLIP on zero-shot image classification but with less (<35%) training cost. Meanwhile, MoDE can train all data expert asynchronously and can flexibly include new data experts. The code is available at https: //github.com/facebookresearch/MetaCLIP/ tree/main/mode.

a cat with its front paws stretched up against the tree

A picture took in a national park

f(⋅ |𝑐)

f(⋅ |𝑐)

CLIP f(⋅)

MoDE (Ours)

f(⋅ |𝑐)

Figure 1. For an image-caption pair, the caption may describe limited visual content or even be unrelated, and such noises unavoidably hurt the quality of negative examples to learning a single model. We propose to uncover the clusters from training data, where 1) the pairs with similar images but different captions are assigned to different clusters and 2) the samples in each cluster are of related meanings, and learn a Data Expert for each cluster. These experts are then selectively ensembled for inference.

The key to the success of contrastive vision-language representation learning lies in the creation of quality negative examples for training [8, 14]. A single image can be depicted by texts with different meanings (i.e., semantics), covering multiple details and interpretations, as illustrated in Fig. 1. Because the paired caption usually describes limited visual content, it is common to see that two similar images have drastically different textual descriptions, especially in noisy web-crawled data. When those imagecaption pairs are sampled in the same batch, captions of other images become false negatives — acceptable captions yet being treated as negative descriptions of the target image. Conversely, if only dissimilar image-caption pairs are sampled, the contrastive learning problem becomes trivial. Incorporating hard negatives [8, 37, 49] (e.g., incorrect yet similar captions that share many words of a correct textual description) in training batches has often been shown to improve the model performance.

### 1. Introduction

Contrastive Language-Image Pretraining (CLIP) learns versatile vision-language representations which are transferable across diverse downstream tasks. Existing models, such as OpenAI CLIP [39], OpenCLIP [44] and MetaCLIP [50], are trained with a large collection of webcrawled image-caption pairs. Specifically, for each image, its paired caption is viewed as a positive example, and the captions of all the other images are viewed as negatives. The model then learns to project both images and captions into a shared space, where the embedding of the positive caption is drawn closer to the image embedding, compared to the embeddings of all the other negative captions.

In this work, we introduce the Mixture of Data Experts (MoDE) framework (shown in Fig. 1-bottom) via clustering. MoDE separates false negative samples into different

∗ Research done while Jiawei Ma was an intern at FAIR.

+ Project Lead.

clusters and groups the pairs with similar semantics, which mitigates noise from false-negative captions while incorporating a more challenging set of hard-negative examples, thereby enhancing vision-language pre-training. MoDE consists of two main steps: (1) the training data (i.e., imagecaption pairs) is first clustered into several disjoint subsets by the captions; each cluster is then used to train a model following the standard contrastive learning method. In this way, each model is specialized by the training data in one cluster and thus termed as a Data Expert. (2) When applied to downstream tasks, such as image classification, the task metadata (i.e., class names), are first compared to the centroid of each data cluster to determine which data expert needs to be activated. Selected data experts are then used to create the embeddings of the test image and classes. The class with the highest ensembled similarity is then output as the classification result.

Empirically, MoDE outperforms several state-of-the-art vision-language models when applied to multiple standard benchmarks, including +3.7% on image classification in CLIP benchmark [34,39], +3.3% on image-to-text retrieval and +2.7% on text-to-image retrieval on COCO [29]. The superiority of MoDE can be attributed to better trained individual data expert models, due to the fact that examples in the same cluster, when used for contrastive learning, provide more quality negatives. Because captions in the same cluster are different but semantically similar (e.g., “a cat climbs a tree”, “a tiger reaches up to a tree”), they become challenging negative examples when compared with images that are not the originally paired ones. On the other hand, it is also less likely to encounter a false negative case where a very different caption validly describes the same image (e.g., “tree guards to stop the cats” in Fig. 1). MoDE is also uniquely positioned for large-scale training when billions of image-caption pairs are available. As each data expert uses only a fraction of the whole dataset, it can be more easily trained with fewer compute resources asynchronously. From experiments across different ViT [6] model scales, we show that four ViT-B/16 data experts can outperform the single ViT-L/14 model by OpenAI CLIP [39] and OpenCLIP [43] on image classification but requires much less (<35%) training cost. In summary, our contributions are:

- • We investigate the quality negative samples in contrastive language-image pretraining, and in particular, the noise of false negatives in web-crawled image-caption pairs.
- • We propose the MoDE framework to learn a system of CLIP data experts via clustering, and adaptively ensemble data experts for downstream tasks at inference time.
- • Extensive experimental study has demonstrated the effects in zero-shot transfer benchmarks with low training cost. MoDE can include new data experts flexibly and is thus beneficial for continual pre-training.

### 2. Related Work

Contrastive Language Image Pretraining (CLIP) aims to learns robust & transferable visual representations from large-scale data. Scaling up [19, 38] existing approaches and improving the effectiveness is critical. Recent progress in the field involves the exploration of regularization techniques [53] and hyperbolic embedding methods [4] but they require significant effort for data annotation. Data curation is then proposed to remove noisy web-crawled imagecaption pairs. Additionally, methods like image masking [28] and concise captions [27] efficiently decrease memory demands, enabling the use of larger batch sizes and model sizes. However, a trade-off between training cost and effectiveness still exists. Following the studies [23, 41] in contrastive learning [2, 16], recent work investigated negative samples in CLIP training but still focuses on image side [30, 48]. The noise exhibited in captions [51] is then overlooked. In this study, we tackle the data noise and the discovery of negative samples via clustering. Rather than training a single model, we asynchronously train multiple data experts and then directly ensemble them for inference adaptively, which also shows benefits for model scaling.

Mixture-of-Expert (MoE) trains a set of sub-models and a routing module. Originally, each expert is defined as an entire network [18, 21], and a single model is selected for each data adaptively. As restricting to hard model selection may limit the practicality, deep mixture of expert [7], is then proposed where the MoE layer is set to softly ensemble layer outputs via weighted sum, which is then investigated with different architectures [9,25] in various tasks [40,45]. However, all expert models are still trained on the same data simultaneously, resulting in much heavier training costs. Recently, BTM [13, 26] proposes to learn expert models on different document types (e.g., papers, posts) separately but is only validated on language models. Meanwhile, both MoE and BTM can only determine the model routing for each input separately. Instead, MoDE generalizes to tasklevel adaptation and ensembles the models by task metadata (e.g., class names in classification task [3]).

Inference-Time Adaptation adapts a pre-trained model quickly and effectively to new tasks. Initially, transductive learning [10] is studied and leverages all unlabeled test data for model update. To mitigate the dependence on the presumed distribution of test data, test-time training [11,42,47] is developed to generate individual models for each input. Subsequent explorations into meta-learning [15,31,46] introduced a separate module (i.e., meta-learner) that can adapt the pre-trained model for each task with a few annotated examples. MoDE has inference-time task adaptation but without annotation or parameter update.

[Figure 3]

[Figure 4]

###### Data Expert Learning (MoDE-4)

###### Inference-Time Task Adaptation

c1

c4

Ensemble Weights

s1

s2

[0.2 0.0 0.6 0.2 ] Routing

- s7
- s8

!(⋅ |%!) !(⋅ |%") !(⋅ |%#) !(⋅ |%$)

s3

c1 c2 c3 c4

###### Data Clustering

c2 c3

s5

s4

Task Metadata (e.g., ‘dog’, ‘cat’)

s1 s2 s3 s4 s5 s6 s7 s8

s6

- Figure 2. Framework of MoDE via clustering. (Left) We perform a two-step clustering on captions to decide clusters / conditions for data experts. The colored scatter plots are fine-grained clusters and the circles are clusters at coarse-grained level. (Right) Each coarse-grained cluster (c) conditions the learning of one data expert f(·|c) and all data experts (colored boxes) are learned asynchronously. For inference, the similarity between task metadata and fine-grained cluster centers ({s}) is used to decide the routing of data experts. To keep reasonable training cost, all data experts can be initialized with a model partially trained on all data without clustering (omitted for simplicity).

[Figure 5]

[Figure 6]

###### Data Expert Learning (MoDE-4)

###### Inference-Time Task Adaptation

Ensemble Weights

[ 0.2 0.0 0.6 0.2 ]

!(⋅ |%!) !(⋅ |%") !(⋅ |%#) !(⋅ |%$)

Routing

### 3. CLIP Data Experts

task at test time requires detailed description (e.g., recognize the “cat” species instead of just “animal”), the conditions should be representative such that the correlation with tasks can be precisely modeled for reliable data experts selection; 2) the number of conditions should be reasonable since each condition is used to learn one data expert. As each condition is represented by a cluster, the ideals of representative likely ask for more fine-grained clustering whereas the latter may require for fewer data experts.

c1 c2 c3 c4

Data Clustering

Task Metadata (e.g., ‘dogs’)

For contrastive image-language pre-training, the model is trained to accurately align each image with the captions describing the visual content. In a manner of divide-andconquer [1], for each CLIP data expert training on one cluster, we reduce the amount of false negatives and increase the hard negatives within each mini-batch. In this way, we mitigate noise exhibited in web-crawled image-caption pairs and make the model training more effective.

s1 s2 s3 s4 s5 s6 s7 s8

Instead, motivated by the ontology in human language, we propose to capture such a hierarchical structure via clustering, i.e., determine the condition of a data expert at the coarse-grained level and represent it via the set of finegrained clusters. For simplicity, we design a two-step Kmeans clustering. We first employ fine-grained clustering to locate each cluster whose samples are of similar semantics, such that the fine-grained cluster centers are representative (Step 1), and then group fine-grained clusters to determine coarse-grained clustering among data for data experts’ specialization (Step 2). In this way, instead of using a single coarse-grained center, the condition is symbolized by the fine-grained cluster centers. The features for clustering are extracted from captions and the details are studied in Sec. 5.

As shown in Fig. 2, on top of the established CLIP training that learns a single dense CLIP model f(·) (Sec. 3.1), we propose to learn a set of CLIP data experts {f(·|c)} via unsupervised clustering (Sec. 3.2) and each CLIP data expert f(·|c) is trained on the cluster c (Sec. 3.3). In this way, the conditioned data expert f(·|c) is less sensitive to the noise from other clusters and can be effectively trained among the data of coherent semantics. For each evaluation task, by measuring the correlation between the task metadata (e.g., class names) and the conditions, the outputs can be jointly decided by multiple data experts (Sec. 3.4).

#### 3.1. Background: Vanilla CLIP Training

CLIP [39] learns separate vision and language encoders with a joint vision-language embedding space. By contrasting positive pairs from negative samples within the same batch, CLIP can accurately model the similarity of the image and caption in each pair. We denote CLIP as f (xv,xl) for an image-caption input (xv,xl), and simplify CLIP model as f(·). As a reminder, instead of learning a single dense CLIP model f(·), we propose to learn a set of CLIP data expert models independently given a set of conditions C, i.e., {f(·|c)|c ∈ C}.

- Step 1: Fine-grained Clustering. As the amount of pretrain data D is huge (hundreds of millions to billions level for CLIP [39]), it could be inefficient to train K-means over all pre-training data. Instead, we first uniformly sample a subset from the pre-training set: D′ ∼ D and |D′| ≪ |D|. Then, we perform K-means training [33] over D′:

S ← K-means(D′), (1)

where S is a set of learned cluster centers. Note that the number of fine-grained clusters m = |S| can be substantially large such that the cluster center of each cluster well represents coherent semantic information for each cluster.

- Step 2: Coarse-grained Clustering.To efficiently allocate the training/inference of a data expert, we perform a second

#### 3.2. Clustering

This subsection discusses how to formulate conditions C, and how to use clustering to automatically discover conditions for data experts from the pre-train set. In a nutshell, the desiderata for the conditions are twofold: 1) as each

round, i.e., coarse-grained, K-means clustering on top of fine-grained cluster centers S:

C ← K-means(S), (2)

where each coarse-grained cluster center c ∈ C is the condition for a data expert. We denote n = |C| as the number of data experts where n ≪ m, and Sc as set of finegrained clusters assigned to the data expert f(·|c) where

- S = ∪c∈CSc.

3.3. Data Experts Training

Next, we formulate training data for each data expert. We first collect the data assigned for each fine-grained cluster s: Ds = {d|s = arg mins∈S(∥ed − es∥22) and d ∈ D}, where ed and es are the embeddings for training example d and fine-grained cluster center s respectively. To train a data expert f(·|c), its corresponding CLIP training data is:

Dc =

s∈Sc

Ds. (3)

For convenience, we use MoDE-n to indicate the system with n CLIP data experts. For training efficiency, all data experts are specialized from the same seed CLIP model that is partially trained over the entire set D. Then, each data expert f(·|c) is trained only on Dc.

3.4. Inference Time Task-Adaptation

As our framework conditions the model expertise on clusters to train data experts, it also gives multiple models to choose from during inference (instead of the only choice on a single CLIP model). This gives the room to adapt different data experts to various downstream tasks.

We propose a simple approach to adapt data experts (no parameter updates) to downstream tasks using the task metadata. Intuitively, this approach routes each downstream task adaptively and efficiently to data experts during inference. For simplicity, we formulate the data experts routing as a weighted sum of data experts’ outputs. Formally, given an evaluation task T, the output of CLIP data experts is

c∈C

f(·|c)p(c|T), (4)

where p(c|T) is the normalized weight for the data expert f(·|c), i.e., c∈C p(c|T) = 1. The weight is proportional to the correlation, i.e., similarity, between metadata of task

- T and condition c. Below we provide simple implementations for zero-shot classification and retrieval, respectively. Zero-Shot Classification. To have accurate routing, we leverage fine-grained cluster centers S in Step 1 to route a task to data experts. We treat the set of class names L as metadata, and define the similarity matrix between classes and data experts as A ∈ R|L|×m. To compute A, we first compute el as the embedding for class l ∈ L via the same

encoder for the embedding of fine-grained cluster center es. Then each entry is defined as

Al,s = exp(−∥el − es∥22/λ), (5)

where λ ∈ R+ is a temperature to sharpen the similarities. Further, the weight routing to a data expert f(·|c) is proportional to

Al,s). (6)

p(c|T) ∝ exp(

l∈L s∈Sc

In practice, we found that using the nearest neighboring fine-grained cluster center (arg maxs∈S Al,s) for each class l ∈ L is good enough to reduce noises in routing.

Zero-Shot Retrieval. The retrieval tasks consist of text retrieval and image retrieval. For text retrieval where each image is used to retrieve a text from a large corpus Q, we leverage Q as metadata to build similarity matrix A ∈ R|Q|×m. Similar to the classification task, the weights for ensembling can be naturally adopted for MoDE:

Aq,s), (7)

p(c|T) ∝ exp(

q∈Q s∈Sc

where each entry Aq,s is computed as exp(−∥eq−es∥22/λ), where eq is the embedding for text q. For image retrieval where each text q retrieves an image separately, we treat the retrieval by text q as an independent task Tq such that the ensembling weights are then p(c|Tq) ∝ exp( s∈S

Aq,s).

c

### 4. Experiment

#### 4.1. Data

We use the datasets collected in MetaCLIP [50] for evaluation and conduct experiments on image-caption pairs at two scales: 400M (similar to the scale in OpenAI CLIP), and 2.5B to scale MoDE. All images are pre-processed with face-blurring and de-duplication against benchmarks.

#### 4.2. Training Setup

Clustering Setup. We use the pre-trained language model SimCSE [12] to extract the embeddings for all captions where the advantages of language encoders over CLIP encoders are studied in Sec. 5.3. We use balanced Kmeans [32] for both of the two unsupervised clustering steps. We set the number of fine-grained clusters m = 1024, and report performance for both MoDE-2 and MoDE4 below to directly show the improvement by increase the number of data expert models on all evaluation tasks.

Data Experts Training Setup. We follow OpenAI CLIP’s hyper-parameters [39] for fair comparison and train on the same budget of 12.8B image-caption pairs (32 epochs of 400M), with a global batch size of 32,768. We train MoDE under 3 scales: for ViT-B/32 and ViT-B/16, we use 64

HatefulMemes

Caltech-101

Kinetics700

Country211

RESISC45

CIFAR100

FER-2013

Food-101

CIFAR10

ImageNet

EuroSAT

SUN397

UCF101

CLEVR

GTSRB

Flowers

Average

Aircraft

MNIST

STL-10

PCAM

KITTI

SST2

CUB

DTD

Cars

Pets

ViT-B/32

|OpenAI CLIP 56.6 OpenCLIP 57.6 MetaCLIP 58.2|63.4 83.7 89.8 65.1 53.7 62.0 59.7 19.6 44.0 87.2 87.4 66.9 48.2 46.6 97.1 44.9 61.0 32.6 28.7 17.2 62.5 63.9 48.0 23.6 56.4 58.6 62.9 80.7 90.7 70.6 61.2 66.4 79.2 16.7 54.5 86.5 90.7 66.1 37.4 48.2 95.6 52.2 58.0 42.0 38.0 14.8 50.1 63.0 42.8 22.5 53.3 52.3 65.5 80.6 91.3 70.2 63.4 63.0 70.7 26.8 52.8 88.7 91.9 68.5 41.5 35.9 95.4 52.6 64.2 35.8 30.7 17.2 55.5 66.1 45.4 30.6 56.4 53.4<br><br>|
|---|---|
|MoDE-2 58.6 MoDE-4 59.0<br><br>|66.1 81.2 90.9 70.5 65.2 63.0 72.0 28.3 53.5 89.4 92.3 68.2 45.2 33.5 95.4 51.9 63.7 34.9 34.2 17.3 54.3 65.9 45.5 29.3 56.6 54.6 66.4 82.3 91.3 70.9 67.0 63.7 73.8 30.1 52.6 89.9 92.1 69.2 37.9 33.2 95.7 53.5 64.1 35.2 33.9 17.1 58.4 66.6 45.9 30.0 58.0 54.5<br><br>|

ViT-B/16

|OpenAI CLIP 59.6 OpenCLIP 60.4 MetaCLIP 61.1|68.3 88.8 90.8 68.2 55.6 64.0 64.6 24.0 45.1 88.9 89.1 69.4 51.8 53.0 98.2 54.8 65.5 43.3 21.7 22.8 56.3 68.5 52.3 25.5 58.7 60.5 67.1 85.8 91.7 71.4 65.3 69.2 83.6 17.4 51.0 89.2 90.8 66.5 66.3 46.1 97.0 52.2 65.7 43.5 23.7 18.1 51.7 67.0 46.2 33.9 54.5 54.4 70.8 86.8 90.1 66.5 70.8 66.6 74.1 27.9 55.9 90.4 93.8 72.3 47.8 44.6 97.2 55.4 68.8 43.8 33.4 22.6 52.9 68.0 49.5 22.8 54.8 60.6<br><br>|
|---|---|
|MoDE-2 61.8 MoDE-4 62.1<br><br>|71.2 87.2 91.3 67.4 71.7 66.8 75.5 29.9 57.0 90.5 94.1 73.0 51.0 44.9 97.2 55.4 68.7 44.5 32.9 22.7 52.9 67.2 49.4 28.1 56.0 60.1 71.6 87.8 91.4 68.9 74.7 67.2 77.3 32.6 56.2 91.3 93.9 74.9 43.7 46.6 97.2 54.4 70.0 44.0 29.8 22.9 55.7 68.6 50.0 29.7 55.2 58.0<br><br>|

ViT-L/14

|OpenAI CLIP 65.7 OpenCLIP 64.5 MetaCLIP 67.1<br><br>|75.5 93.0 95.6 78.3 63.3 66.8 77.8 31.3 55.3 93.6 93.3 79.3 76.4 56.9 99.4 61.9 70.9 50.6 19.2 31.9 50.1 75.7 60.2 22.3 59.7 68.9 72.7 90.0 94.7 78.0 73.9 72.4 89.5 24.7 60.2 91.6 93.6 73.0 76.1 54.3 98.1 63.9 69.6 49.9 16.0 23.0 51.7 71.5 51.6 25.4 55.3 56.0<br><br>76.2 90.7 95.5 77.4 75.9 70.5 84.7 40.4 62.0 93.7 94.4 76.4 61.7 46.5 99.3 59.7 71.9 47.5 29.9 30.9 70.1 75.5 57.1 35.1 56.6 65.6<br><br><br>|
|---|---|
|MoDE-2 67.1 MoDE-4 67.2<br><br>|76.5 91.1 95.9 77.8 76.7 70.6 85.1 40.9 62.4 93.9 94.8 76.8 63.0 46.2 99.4 57.8 71.7 47.4 26.7 31.1 69.9 75.6 57.3 33.1 56.6 65.5 76.3 91.2 95.7 77.9 78.3 70.7 85.6 41.8 62.4 94.0 94.5 77.1 62.6 46.6 99.2 57.7 72.0 47.3 26.8 31.3 71.5 76.0 57.3 30.6 56.6 65.5<br><br>|

- Table 1. Performance on CLIP benchmark [34,39] by models trained on 400M image-caption pairs. MoDE-2 and MoDE-4 consistently outperform the MetaCLIP Baseline and MoDE-4 achieves the best score on average.

Average

ImageNet

Food-101

CIFAR10

CIFAR100

CUB

SUN397

Cars

Aircraft

DTD

Pets

Caltech-101

Flowers

MNIST

FER-2013

STL-10

EuroSAT

RESISC45

GTSRB

KITTI

Country211

PCAM

UCF101

Kinetics700

CLEVR

HatefulMemes

SST2

ViT-B/32

|OpenCLIP 61.5 MetaCLIP 59.8<br><br>|66.6 82.0 93.6 75.8 66.0 68.3 86.0 23.9 56.1 90.5 91.9 70.5 70.0 50.4 96.6 49.3 65.7 49.3 32.7 16.7 51.7 64.9 45.6 24.2 52.4 57.2<br><br>67.6 82.6 95.2 77.7 67.8 66.8 77.2 26.9 58.9 90.9 92.5 69.7 42.7 48.3 96.3 49.9 66.5 39.2 29.3 17.7 50.0 68.0 47.6 19.4 53.5 53.1<br><br><br>|
|---|---|
|MoDE-2 61.2 MoDE-4 61.7<br><br>|68.7 84.1 95.3 78.6 69.5 67.0 80.8 30.9 60.6 91.0 92.9 71.9 40.8 50.4 96.3 51.3 67.9 44.2 31.4 18.3 51.3 69.0 47.4 23.2 52.6 54.4<br><br>68.8 85.8 95.2 79.0 74.4 67.5 83.3 29.5 60.3 91.9 92.9 72.1 49.7 46.9 96.4 50.3 66.8 51.6 28.5 19.6 50.1 68.4 48.3 21.6 52.6 52.2<br><br><br>|

ViT-B/16

|OpenCLIP 62.4 MetaCLIP 63.5<br><br>|70.2 86.2 94.9 76.9 70.5 70.6 88.2 26.6 56.3 90.4 93.1 71.0 65.8 53.3 97.9 55.2 68.3 48.3 11.9 20.3 51.2 68.1 48.9 24.8 53.0 59.5 72.1 88.3 95.7 79.0 71.4 68.5 82.9 30.3 62.1 91.7 93.3 73.9 66.1 47.0 98.4 51.1 71.1 46.6 16.6 22.7 50.5 73.0 52.5 30.8 57.4 59.0<br><br>|
|---|---|
|MoDE-2 65.0 MoDE-4 67.2<br><br>|73.6 89.5 96.0 81.4 76.5 69.0 85.7 35.9 63.5 93.4 93.4 75.5 59.2 46.4 98.3 50.0 72.0 50.1 34.9 23.9 50.8 71.2 52.1 31.2 59.1 58.4<br><br>74.2 91.6 96.5 82.0 80.9 71.2 88.9 42.2 63.0 93.6 93.6 78.9 66.8 49.0 98.5 53.8 71.5 57.5 32.4 26.7 61.7 73.8 53.9 27.4 57.0 59.4<br><br><br>|

ViT-L/14

|OpenCLIP 65.7 MetaCLIP 69.8|74.0 88.6 95.8 78.3 73.5 73.5 91.4 34.6 61.2 92.7 93.3 74.4 64.4 53.9 98.5 58.6 71.9 51.6 26.1 24.4 58.0 73.3 52.0 27.4 55.1 60.4 79.2 93.4 97.6 84.2 80.1 73.8 88.7 44.6 68.1 94.7 95.4 81.8 64.4 55.1 99.3 59.2 74.6 56.3 29.7 34.0 67.3 81.6 62.0 25.9 58.0 66.7<br><br>|
|---|---|
|MoDE-2 70.4 MoDE-4 71.2<br><br>|79.5 93.5 97.6 85.0 82.9 74.0 90.9 49.0 69.5 95.0 95.3 81.8 69.7 53.7 99.2 63.3 75.2 59.0 29.8 33.9 62.3 81.7 62.4 24.0 56.6 65.4 79.4 94.0 97.8 85.6 83.5 74.2 91.2 48.7 69.1 95.6 95.6 81.4 71.4 54.3 99.3 61.0 76.5 63.3 34.7 34.0 70.9 81.6 62.2 24.6 55.7 66.7<br><br>|

- Table 2. Performance on CLIP benchmark [34,39] by models trained on billion-scale dataset (OpenCLIP: 2.3B, MetaCLIP/MoDE: 2.5B). MoDE-2 and MoDE-4 consistently outperform the MetaCLIP Baseline and MoDE-4 achieves the best score on average.

Nvidia V100 GPUs with a per GPU batch size of 512, and for ViT-L/14, we use 128 GPUs with a 256 per GPU batch size. To maintain a reasonable training cost, we start MoDE training from the 27th epoch (out of 32 epochs) of a partially trained MetaCLIP as the seed model and all data experts share the same seed model to save computes.

#### 4.3. Evaluation

Zero-Shot Image Classification. We follow the evaluation protocol in CLIP benchmark [34,39,50] and use the same class names & prompts by OpenAI CLIP. For fair comparison, MetaCLIP [50] naturally serves as the single dense baseline. The checkpoints of OpenAI CLIP (WIT400M data) [39] and OpenCLIP (LAION-400M data, LAION-2B data) [44] are also re-evaluated for fair comparison.

The framework MoDE has shown consistent performance gain across model scales and data scales. Firstly,

we compare the models learned from 400M-scale dataset in Table 1, and summarize the results by different model scales. MoDE achieves consistent performance gain where increasing the number of data experts results in better performance. Next, we study the scaling property of MoDE on 2.5B image-text pairs. From Table 2, comparing against MetaCLIP [50], the advantage of MoDE to learn four data expert models is better revealed on scaling training data: +1.9% on B/32, +3.7% on B/16, and +1.4% on L/14. Lastly, we increase the number of data experts. As shown in Fig. 3, the performance can be kept improving when we increase the number of data experts where MoDE-16 ViT-B/32 can outperform the MetaCLIP ViT-B/16 baseline.

Notably, MoDE provides an efficient and scalable approach to consume large-scale data without a large batch size that requires more GPUs (384 Nvidia A100 GPUs) as in OpenCLIP. As shown in Table 2, based on ViT-B/16 with

Approach ViT Avg. IN-Sk IN-V2 IN-A IN-O IN-R Avg. IN-Sk IN-V2 IN-A IN-O IN-R OpenAI CLIP

Text Retrieval Image Retrieval COCO Flickr30k COCO Flickr30k R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 OpenCLIP

Approach ViT

49.4 42.3 56.0 31.5 47.8 69.4 - - - - - -

OpenCLIP 50.6 49.4 55.1 21.7 53.5 73.4 52.9 53.7 58.1 26.3 50.0 76.4 MetaCLIP 52.2 53.3 57.6 28.6 46.8 74.8 54.4 56.0 59.6 29.9 48.3 78.1

B/32

56.3 79.8 87.1 84.1 96.2 98.3 39.3 65.4 75.6 66.7 88.4 93.1

MoDE-2 53.0 53.9 57.9 29.4 48.0 75.7 55.2 57.1 60.5 31.2 48.4 79.0 MoDE-4 53.4 54.4 58.5 30.8 47.6 76.0 56.5 57.6 61.6 34.2 49.2 80.0

MetaCLIP 55.2 78.9 86.5 80.7 95.2 97.3 38.1 64.1 74.3 65.1 87.7 92.7 MoDE-2 56.7 80.2 87.5 82.8 95.1 98.2 39.5 65.3 75.3 66.4 89.0 93.6 MoDE-4 57.4 80.1 87.3 82.9 95.6 97.7 39.9 66.1 75.7 66.7 88.4 93.3

B/32

OpenAI CLIP

56.0 48.3 61.9 50.0 42.3 77.7 - - - - - -

Average Accuracy on CLIP Benchmark

[Figure 7]

OpenCLIP 54.8 52.4 59.7 33.2 50.7 77.9 56.7 56.1 62.3 38.2 46.3 80.6 MetaCLIP 57.7 57.9 62.6 47.0 39.2 81.8 60.1 60.2 65.0 49.5 41.6 84.2

OpenCLIP

59.5 81.8 88.6 86.2 98.0 99.5 42.3 67.7 77.1 69.8 90.4 94.6

[Figure 8]

[Figure 9]

B/16

Average Accuracy on CLIP Benchmark

MetaCLIP 59.4 80.6 87.8 85.5 97.4 98.9 41.4 67.2 76.9 70.7 90.8 94.5 MoDE-2 60.7 82.6 89.0 87.3 97.6 99.2 43.1 68.6 77.8 72.1 91.8 95.3 MoDE-4 62.7 82.9 89.8 89.4 98.0 99.6 44.1 69.5 78.7 72.6 91.8 95.4

MoDE-2 58.4 58.5 63.2 47.9 39.9 82.3 62.3 62.4 66.5 52.0 45.2 85.5 MoDE-4 59.0 58.8 63.7 49.2 40.4 82.9 63.3 62.8 67.1 55.7 44.5 86.6

B/16

OpenAI CLIP

64.1 59.6 69.8 70.7 32.3 87.9 - - - - - -

OpenCLIP

63.3 83.9 90.8 89.5 98.7 99.4 46.5 71.1 79.8 75.5 92.9 95.9

OpenCLIP 59.6 59.6 65.5 46.5 42.0 84.7 62.2 63.3 67.8 53.9 38.7 87.4 MetaCLIP 63.8 65.0 69.8 66.4 28.9 88.9 67.2 68.9 72.6 72.3 30.2 92.1

MetaCLIP 64.4 85.0 91.3 90.1 98.6 99.3 47.1 71.4 80.3 76.5 93.6 96.5 MoDE-2 65.2 85.3 91.6 90.9 98.9 99.6 47.9 72.1 80.6 77.2 93.7 96.6 MoDE-4 65.5 85.4 91.8 91.2 99.0 99.7 48.2 72.4 80.7 77.6 93.7 96.7 Pretrain Data OpenCLIP:2.3B; MetaCLIP/MoDE:2.5B

L/14

L/14

[Figure 10]

MoDE-2 64.0 65.2 70.0 66.9 28.9 89.0 67.6 69.3 72.8 73.0 30.6 92.3 MoDE-4 64.1 65.3 70.1 66.8 29.4 89.0 68.2 69.9 73.3 74.0 31.3 92.7 Pre-Train Data 400M Image-Caption Pairs OpenCLIP:2.3B; MetaCLIP/MoDE:2.5B

Average Accuracy on CLIP Benchmark

B/32 B/16 L/14

Diameter

B/32 B/16 L/14

Table 4. Zero-shot Retrieval. Entries in blue are the best ones. Results by model trained on 400M pairs can be found in the Suppl.

Diameter

- Table 3. Zero-Shot Robustness Evaluation. The results are separated by the scale of pre-train set. Entries in blue are the best ones.

MetaCLIP OpenAI CLIP OpenCLIP

###### MoDE-n (Ours)

MoDE-n (Ours) OpenCLIP

Baseline OpenAI CLIP (MoDE-1)

GPU Days

MoDE-1 (MetaCLIP)

GPU Hours

m 2 22 25 28 29 210 211

Average Accuracy on CLIP Benchmark

[Figure 11]

Average Accuracy on CLIP Benchmark

[Figure 12]

MoDE-1 (MetaCLIP, ViT-B/16)

# Data Experts (VIT-B/32)

MoDE-1 (MetaCLIP, ViT-B/32)

B/32 B/16 L/14

Diameter

MetaCLIP OpenAI CLIP OpenCLIP

- Figure 3. Average accuracy CLIP benchmark with increased number of data expert models in MoDE (Pretrain set: 2.5B pairs).

MoDE-n (Ours)

GPU Days

Figure 4. Summary of average accuracy on CLIP benchmark and pretraining cost (GPU-Hours). The diameter is proportional to the model size, different approaches are color-coded.

a batch size of 32K, the MoDE-2 with two data expert models is on par with the ViT-L/14 model by OpenCLIP [43], while 4 data expert models can outperform the ViT-L/14 by

- 1.5% on CLIP benchmark dataset. Nevertheless, MoDE requires much less pretraining cost. As summarized in Fig. 4, MoDE-4 ViT-B/16 only requires less-than-35% of GPUHours used for OpenAI CLIP ViT-L/14. Compared with OpenCLIP trained on LAION-2B data, MoDE-8 ViT-B/32 data experts can even outperform a single ViT-B/16 model by OpenCLIP but only use 31% of its GPU-Hours. In this way, our approach demonstrates great potential for efficient CLIP pretraining with limited GPUs in future. Zero-Shot Robustness. In addition, to show a consistent gain on different tasks in the CLIP benchmark, we further validate the benefits towards robustness of MoDE in variants of ImageNet zero-shot classification. As summarized in Table 3, though there are systematic gaps across variants of ImageNet, learning a set of data experts can improve the zero-shot accuracy on all five variants over the MetaCLIP Baseline for all model scales, and increasing the number of data experts can still introduce consistent gain. For the accuracies on IN-A and IN-O, the gap between baseline and other approaches is mitigated clearly by MoDE. Finally, MoDE-4 achieves the highest average accuracy of all dataset variants among all compared methods. Zero-Shot Retrieval. We follow OpenCLIP [43] and reports the image/text retrieval results on COCO [29] and

Flickr30k [52]. The compared models are trained on billion-scale datasets. As shown in Table 4, learning data experts can improve the scores consistently across all model sizes, on COCO, in particular, +3.3% and +2.7% in R@1 for image-to-text and text-to-image retrieval respectively by ViT-B/16 models, and we achieve the best performance. For the performance gap between MetaCLIP Baseline and OpenCLIP, e.g., text retrieval on Flickr30k by ViT-B/32 models, the gap can also be mitigated clearly.

### 5. Discussion

We first analyze the importance of clustering (Sec. 5.1) and then study the MoDE design (Secs. 5.2 and 5.3). Finally, we investigate the potential of our approach in other important research directions (Secs. 5.4 and 5.5).

#### 5.1. Effectiveness of Clustering

As MoDE ensembles the data experts learned from different clusters, we are first interested in the effects of clustering and consider two variants for ablation.

Though model ensembling [22] can provide gains over a single model, we are interested in how a naive ensem-

HatefulMemes

Caltech-101

Kinetics700

Country211

RESISC45

CIFAR100

FER-2013

Food-101

CIFAR10

ImageNet

EuroSAT

SUN397

UCF101

CLEVR

GTSRB

Flowers

Average

Aircraft

MNIST

STL-10

PCAM

KITTI

SST2

CUB

DTD

Cars

Pets

400M Image-Caption Pairs

- MetaCLIP 58.2 65.5 80.6 91.3 70.2 63.4 63.0 70.7 26.8 52.8 88.7 91.9 68.5 41.5 35.9 95.4 52.6 64.2 35.8 30.7 17.2 55.5 66.1 45.4 30.6 56.4 53.4 Random-2 57.7 64.9 80.7 91.4 69.6 59.8 63.0 72.3 28.3 52.3 88.7 91.9 69.4 38.1 30.8 95.4 52.9 62.9 33.2 36.1 17.3 54.4 65.7 44.7 27.1 56.2 53.0

Full-2 58.3 65.9 81.0 91.2 69.9 63.8 63.3 71.0 27.3 52.3 88.9 91.8 69.2 42.9 33.3 95.4 52.5 64.6 35.8 31.2 17.0 56.1 67.0 45.5 28.7 57.5 53.5

MoDE-2 58.6 66.1 81.2 90.9 70.5 65.2 63.0 72.0 28.3 53.5 89.4 92.3 68.2 45.2 33.5 95.4 51.9 63.7 34.9 34.2 17.3 54.3 65.9 45.5 29.3 56.6 54.6 2.5B Image-Caption Pairs

- MetaCLIP 59.8 67.6 82.6 95.2 77.7 67.8 66.8 77.2 26.9 58.9 90.9 92.5 69.7 42.7 48.3 96.3 49.9 66.5 39.2 29.3 17.7 50.0 68.0 47.6 19.4 53.5 53.1 Random-2 60.0 67.4 82.4 95.0 77.8 68.1 66.6 77.0 26.5 58.3 91.0 92.3 69.0 45.4 47.8 96.2 50.4 66.2 43.8 30.0 17.7 50.0 67.8 47.4 20.2 53.8 52.1

Full-2 60.0 67.8 82.6 95.2 77.7 68.4 66.7 77.7 27.7 58.6 90.9 92.5 69.9 43.6 48.7 96.4 50.1 66.0 41.7 28.2 17.9 50.0 68.4 47.7 19.3 53.9 52.8 MoDE-2 61.2 68.7 84.1 95.3 78.6 69.5 67.0 80.8 30.9 60.6 91.0 92.9 71.9 40.8 50.4 96.3 51.3 67.9 44.2 31.4 18.3 51.3 69.0 47.4 23.2 52.6 54.4

Table 5. Ablation Study for performance gain via Clustering by VIT-B/32.

|Approach<br><br>|CLIP Avg. ImageNet<br><br>|CLIP Avg. ImageNet|
|---|---|---|
|MetaCLIP OneStep-2 CoarseCluster-2 MoDE-2<br><br>|58.2 65.6 58.0 65.0<br><br>58.5 66.1<br><br>58.6 66.1<br><br><br>|59.8 67.7<br><br>59.8 67.6<br><br>60.6 68.6<br><br>61.2 68.7<br><br><br>|

|CoarseCluster-4 MoDE-4<br><br>|58.7 66.2<br><br>59.0 66.4<br><br><br>|61.3 68.5 61.7 68.8<br><br>|
|---|---|---|
|Pre-Train Dataset|400M Image-Caption Pairs|2.5B Image-Caption Pairs|

Table 6. Ablation study for Clustering Strategy by ViT-B/32.

bling of models trained on similar distribution performs compared to MoDE with data specialization. In Table 5, we train two ViT-B/32 CLIP models on the same training data without clustering, and then average the model outputs for prediction (Full-2). This achieves a similar performance as the baseline. Thus, the clustering is essential for MoDE.

Furthermore, we randomly split the training data into two subsets, and specialize a data expert for each subset (Random-2). For a fair comparison, we mimic the size of subsets by MoDE-2 in the random splitting, and all data experts use the same seed model. As the data split is not obtained through clustering, we still only use the average of model outputs for evaluation. However, though Random2 can provide small improvement when trained on 2.5B image-caption pairs (60.0 vs. 59.8), there is a noticeable drop when training on the 400M pairs (57.7 vs. 58.2).

#### 5.2. Clustering Strategy

Instead of obtaining the data clusters in a single step, MoDE employs a two-step clustering strategy to discover the centers of fine-grained cluster S, which are used to properly model the correlation between task metadata and the conditions (Sec. 3.2). We provide ablation studies below to demonstrate this necessity for model ensembling.

Firstly, we evaluate the one-step clustering alternative, i.e., m = n, and for simplicity, we only learn two data experts (OneStep-2) based on ViT-B/32. As shown in Table 6, we summarize the average score on the CLIP benchmark and stand out the accuracy of ImageNet as it has the most number of classes. As the cluster centers are not representative enough to model the correlation with task metadata,

model ensembling in OneStep-2 can even result in a slight drop. We do observe that each data expert alone can outperform MetaCLIP Baseline on different tasks in the CLIP benchmark but it is difficult to pick correctly.

Then, we follow the two-step clustering but alter the number of fine-grained clusters m in the first step. As plotted in Fig. 5, we summarize the results of MoDE-2 trained on 400M image-caption pairs. With increasing m, we observed that the average accuracy on the CLIP evaluation benchmark improves consistently. Though the performance can be improved slightly when m is increased from 1024 to 2048, the computational cost during data clustering is also higher. We set m = 1024 in the main experiments.

Lastly, as another piece of evidence, we keep m as 1024 but use the coarse-grained cluster centers in Step 2, to determine the ensembling weights (CoarseCluster). As shown in Table 6 , as the meta clusters are not representative enough to obtain good ensembling weight, the resulting accuracy improvement is trivial. When we increase the number of data experts from 2 to 4, the gap between CoarseCluster-4 and MoDE-4 is even enlarged, which further demonstrates the importance of using fine-grained clusters to determine the ensembling weight for data experts in our MoDE.

#### 5.3. Embeddings for Clustering

We further validate the importance of using language embeddings. In addition to SimCSE [12] language embedding, we investigate the following embeddings for clustering: (1) image embedding from the open-sourced DINOv2 [36]; (2) image and/or text embeddings from the seed model (i.e., the partially trained MetaCLIP checkpoints on the 27th epoch). When the image embeddings are used for clustering, for each test image, we use its similarity with all fine-grained cluster centers to determine the logits ensemble weights. When both image and text embeddings are used, we use their concatenation as the feature for clustering. Without loss of generality, we compare with MoDE-2 trained on 400M pairs and set m = 1024 for fair comparison. We summarize the scores in Table 7 and report the zero-shot accuracy CLIP benchmark and ImageNet.

[Figure 13]

[Figure 14]

Average Accuracy on CLIP Benchmark

[Figure 15]

Average Accuracy on CLIP Benchmark

|Approach<br><br>|B/32 B/16 L/14 G/14|
|---|---|
|OpenAI CLIP OpenCLIP MetaCLIP Ours<br><br>|63.3 68.4 75.6 -<br><br>66.6 70.2 75.3 80.1<br>67.6 72.1 79.2 71.4 75.3 80.3 -<br><br><br>|

Modality Model CLIP Eval. ImageNet

Image DINOv2 58.1 65.2 Image CLIP Seed 58.3 64.7

B/32 B/16 L/14

Diameter

Image & Lang. CLIP Seed 58.4 65.5 Lang. CLIP Seed 58.3 65.4 Lang. SimCSE [12] 58.6 66.1

MoDE-n (Ours) OpenCLIP

Baseline OpenAI CLIP (MoDE-1)

MoDE-1 (MetaCLIP)

GPU Hours

m 2 22 25 28 29 210 211

Table 8. ImageNet Zero-shot Accuracy via Prior-based Data Ranking.

[Figure 16]

Average Accuracy on CLIP Benchmark

Figure 5. Ablation on # of clusters in Step 1.

Table 7. Ablation Study on Embedding Types.

MoDE-1 (MetaCLIP, ViT-B/16)

Firstly, by using image embeddings for clustering, the resulting models underperform MetaCLIP, in particular on ImageNet, and we believe the main reason is that the image embedding contains low-level details. As such, the cluster centers are not representative of model ensembling.

coders as the representation and feed it into a linear layer for classification. To maintain reasonable training cost, only linear probing is considered where we exclusively train the linear classifier from scratch and fix all vision encoders. As shown in Table 9, our MoDE achieves consistent and clear performance gain over MetaCLIP Baseline.

# Data Experts (VIT-B/32)

MoDE-1 (MetaCLIP, ViT-B/32)

Furthermore, utilizing the language embeddings from the seed model yields only marginal performance improvement. This suggests that the CLIP embedding may still fall short of discerning high-level semantic correlations. This occurs as the language embeddings are influenced by image embeddings, potentially overlooking high-level semantics not depicted in corresponding images. For example, abstract concepts such as “travel”, “product”, and “politics” may lack corresponding visual elements. In contrast, the SimCSE text embeddings pretrained on large text corpora can understand abstract concepts for clustering. As another evidence, we compare the clustering based on language embeddings and use TF-IDF embeddings as reference. As the TF-IDF embedding is determined by on the frequency of words, the clusters on TF-IDF embeddings shown in Fig. 6 can only group captions with the same words, and struggle to provide abstract concepts via discrete text tokens. In contrast, using SimCSE embeddings can group the captions with coherent semantics (e.g., food).

|Model<br><br>|Linear Probe∗ B/32 B/16 L/14<br><br>|Linear Probe B/32 B/16 L/14|
|---|---|---|
|MetaCLIP MoDE-2 MoDE-4|69.3 73.3 80.3<br><br>68.9 73.8 80.6<br><br>69.1 74.5 80.7<br><br><br>|67.5 73.8 82.3 71.3 76.9 83.9 74.1 79.6 84.7<br><br>|

∗: Initialize classifier with language embeddings as in OpenCLIP [43].

Table 9. Performance comparison on ImageNet via linear probing.

As shown in Table 10, we evaluate all vision encoders by MoDE-4 ViT-B/16 independently and report the accuracy via linear probing and finetuning (i.e., all parameters are trained). Linear probing on the concatenated features achieves higher score than finetuning a single model (79.6 Vs. 76.7) but with much less training cost.

Embedding TF-IDF SimCSE

- Cluster A “Star Wars Rebels”; “Converse sneakers alta in tela Chuck Taylor All Star Eva Lift”

“Egg noodles with garlic and tomato” “The best steak ever”

- Cluster B “Impact of Samsung‘s most recent data breach unknown”; “Unknown”; “Unknown”

|Data Experts|Zero-Shot Linear Probe∗ Linear Probe Finetune<br><br>|
|---|---|
|MetaCLIP<br><br>0<br>1<br>2<br>3<br>|72.1 73.3 73.8 76.7 63.3 66.4 67.3 75.7 68.5 71.3 72.0 76.9 65.2 68.2 68.8 76.3 72.9 74.9 74.2 77.2<br><br>|

“outdoor fire pit at a cabin in the smoky mountains” “Wildfire, grass in flame and fume”

Cluster (Food) “Egg noodles with garlic and tomato” “The best steak ever”

###### Cluster (Wildfire)

‘fireman with flame in the wild nature’ “graphic locates multiple wildfires on a map of the western U.S.”

SimCSE

∗: Initialize classifier with language embeddings as in OpenCLIP [43].

Cluster (‘Unknown’) ‘Impact of Samsung‘s most recent data breach unknown’; ‘Unknown’; ‘Unknown’;

Cluster (‘Star’) ‘Star Wars Rebels’; ‘Converse sneakers alta in tela Chuck Taylor All Star Eva Lift’

Table 10. Evaluation for each data expert in MoDE-4 ViT-B/16.

TF-IDF

In addition, by comparing among vision encoders, the data expert achieving higher zero-shot accuracy also hits the best score in both linear probing and finetuning, indicating a consistent correlation benefited from the strong encoder initialization. In this way, by training data expert on each coarse-grained cluster, we increase the quality negative within each mini-batch to learn stronger vision encoders effectively. Finally, the parameters can also be averaged and then used as initialization of a single network for finetuning, and more details can be found in the Supp.

Figure 6. Representative instances for each cluster.

#### 5.4. Application of Vision Encoders

Besides zero-shot generalization, the set of vision encoders can also be directly ensembled in downstream application. Notably, all vision encoders are equally weighted, and we do not need any cluster center (i.e., clusterindependent), which is generalizable to the case where the language metadata such as class names is not available.

#### 5.5. Training Priority of Data Experts

Firstly, we ensemble the encoder outputs and use ImageNet classification for evaluation. Specifically, for each image, we concatenate the outputs from all (n) vision en-

As the data experts can be trained asynchronously, MoDE introduces flexibility in the data expert training pri-

[Figure 17]

[Figure 18]

Average Accuracy on CLIP Benchmark

Besides, as detailed in Suppl., MoDE can also be aligned for a set of downstream tasks, e.g., CLIP benchmarks.

[Figure 19]

In summary, MoDE can be applied to different types of downstream tasks. Meanwhile, the coarse-level clustering in the second step tentatively assumes the fine-grained clusters should be split into disjoint groups with overlap. We believe the fine-grained clusters can also be grouped flexibly and we leave it for future work.

One Data Expert All Data Experts

Figure 7. CLIP benchmark accuracy by MoDE-n when the data experts based on ViT-B/32 are developed in order and added to the system progressively. The pre-train set contains 2.5B pairs.

### 6. Conclusion

The success of CLIP depends on the quality negative samples. As the false negative noise in web-crawled pairs hurts training effectiveness, scaling CLIP on large-scale data presents unique challenges in terms of training efficiency and computational bottlenecks. To this end, we have presented Mixture of Data Experts (MoDE) to asynchronously train a group of data experts. Each expert model is trained on a set of fine-grained clusters where the data in each cluster is of coherent semantics and all data experts are trained individually. During inference, the outputs are selectively ensembled based on the requirements for each task and modeled by the correlation between task metadata and fine-grained cluster centers. Empirically, MoDE significantly outperforms OpenCLIP and OpenAI CLIP on standard benchmarks with less than 35% training cost. Furthermore, the image embeddings extracted by all data experts can be combined to enhance the representation of visual information. We plan to adapt MoDE for generative models in the future.

ority. Below we demonstrate the robustness and effectiveness of MoDE when the data experts are trained in order.

Firstly, we rank the conditions, i.e., coarse-level clusters, to determine the training priority of data experts. This is useful when the computational resource is not sufficient to learn a giant dense model or all data experts together. We use the diversity of fine-grained clusters as a reference, and first train the model on the condition with the largest range, i.e., the average distance between fine-grained clusters and the coarse-grained center. As shown in Fig. 7, we vary the total number of ViT-B/32 data experts, i.e., n, from 2 to 32 and summarize the average accuracy on the CLIP benchmark. When the data experts are gradually included, the performance keeps increasing.

In this way, instead of learning from all data simultaneously, MoDE enables progressive integration of new data experts, enabling dynamic updates. MoDE holds promise for applications such as online and continual learning. With each new set of data, it has the flexibility to update a pretrained data expert, or to learn a new data expert. This is particularly valuable when the incoming data are unprecedented to the existing expert system. We leave the trade-off between catastrophic forgetting [24] and effective adaption as the futrure work.

### Acknowledgement

The authors would like to thank Xinlei Chen and Margaret Li for constructive discussion.

### References

- [1] Richard E Blahut. Fast algorithms for signal processing. Cambridge University Press, 2010. 3
- [2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020. 2
- [3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 2
- [4] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, pages 7694–7731. PMLR, 2023. 2
- [5] Inderjit S Dhillon and Dharmendra S Modha. Concept decompositions for large sparse text data using clustering. Machine learning, 42:143–175, 2001. 14

At the same time, we can also select the clusters given the task metadata as prior following the retrieval-enhanced setup [17]. When the metadata is accessible, we use the SimCSE [12] to extract their embeddings and retrieve the nearest fine-grained clusters for each of them. Then, the data expert trained on the selected clusters is of highest training priority, and we only train that single data expert for evaluation while the rest clusters can be left for future continual MoDE pretraining if needed. We take ImageNet as an example where the 1000 class names are used to retrieve clusters. As summarized in Table 8, adapting our approach can improve the efficiency of network training significantly and can even escalate the performance along the model scale in most cases. For example, our ViT-B/16 outperforms the L/14 models by OpenAI CLIP/ OpenCLIP and our ViT-L/14 even outperforms the ViT-G/14 in OpenCLIP.

- [6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2
- [7] David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314, 2013. 2
- [8] Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, and Sanja Fidler. VSE++: improving visual-semantic embeddings with hard negatives. In British Machine Vision Conference 2018, BMVC 2018, Newcastle, UK, September 3-6, 2018, page 12. BMVA Press, 2018. 1
- [9] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022. 2
- [10] Alex Gammerman, Volodya Vovk, and Vladimir Vapnik. Learning by transduction. arXiv preprint arXiv:1301.7375,

2013. 2

- [11] Yossi Gandelsman, Yu Sun, Xinlei Chen, and Alexei Efros. Test-time training with masked autoencoders. Advances in Neural Information Processing Systems, 35:29374–29385,

2022. 2

- [12] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, pages 6894–6910. Association for Computational Linguistics (ACL), 2021. 4, 7, 8, 9, 12
- [13] Suchin Gururangan, Margaret Li, Mike Lewis, Weijia Shi, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Scaling expert language models with unsupervised domain discovery. arXiv preprint arXiv:2303.14177, 2023. 2
- [14] Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 297–304, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR. 1
- [15] Guangxing Han, Jiawei Ma, Shiyuan Huang, Long Chen, and Shih-Fu Chang. Few-shot object detection with fully cross-transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5321–5330, 2022. 2
- [16] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 2
- [17] Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. Retrieval-enhanced contrastive vision-text models. arXiv preprint arXiv:2306.07196, 2023. 9
- [18] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991. 2

- [19] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 2

- [20] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019. 14
- [21] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994. 2
- [22] Michael I Jordan and Tom M Mitchell. Machine learning: Trends, perspectives, and prospects. Science, 349(6245):255–260, 2015. 6
- [23] Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. Hard negative mixing for contrastive learning. Advances in Neural Information Processing Systems, 33:21798–21809, 2020. 2
- [24] James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka GrabskaBarwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. CoRR, abs/1612.00796, 2016. 9
- [25] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. 2
- [26] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306,

- 2022. 2

[27] Xianhang Li, Zeyu Wang, and Cihang Xie. An inverse scaling law for clip training. arXiv preprint arXiv:2305.07017,

- 2023. 2

- [28] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390– 23400, 2023. 2
- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2, 6
- [30] Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, and Chunyuan Li. Learning customized visual models with retrieval-augmented knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15148–15158, 2023. 2
- [31] Jiawei Ma, Hanchen Xie, Guangxing Han, Shih-Fu Chang, Aram Galstyan, and Wael Abd-Almageed. Partner-assisted learning for few-shot image classification. In Proceedings

- of the IEEE/CVF International Conference on Computer Vision, pages 10573–10582, 2021. 2
- [32] Mikko I Malinen and Pasi Fr¨anti. Balanced k-means for clustering. In Structural, Syntactic, and Statistical Pattern Recognition: Joint IAPR International Workshop, S+ SSPR 2014, Joensuu, Finland, August 20-22, 2014. Proceedings, pages 32–41. Springer, 2014. 4
- [33] Tom M Mitchell. Machine learning, 1997. 3
- [34] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets language-image pretraining. In European Conference on Computer Vision, pages 529–544. Springer, 2022. 2, 5
- [35] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems, 35:9564– 9576, 2022. 12
- [36] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 7
- [37] Mandela Patrick, Po-Yao Huang, Yuki Markus Asano, Florian Metze, Alexander G. Hauptmann, Jo˜ao F. Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 1
- [38] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei Yu, Jiahui Yu, Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, et al. Combined scaling for zero-shot transfer learning. Neurocomputing, 555:126658, 2023. 2
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 4, 5
- [40] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andr´e Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021. 2
- [41] Vin Sachidananda, Ziyi Yang, and Chenguang Zhu. Global selection of contrastive batches via optimization on sample permutations. In Proceedings of the 40th International Conference on Machine Learning, 2023. 2
- [42] Stephan R Sain. The nature of statistical learning theory,

1996. 2

- [43] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2, 6, 8

- [44] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 1, 5
- [45] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In International Conference on Learning Representations, 2016. 2
- [46] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017. 2
- [47] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In International conference on machine learning, pages 9229–

9248. PMLR, 2020. 2

- [48] Chen-Wei Xie, Siyang Sun, Xiong Xiong, Yun Zheng, Deli Zhao, and Jingren Zhou. Ra-clip: Retrieval augmented contrastive language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19265–19274, 2023. 2
- [49] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP, 7-11 November, 2021, pages 6787–6800. Association for Computational Linguistics, 2021. 1
- [50] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. arXiv preprint arXiv:2309.16671, 2023. 1, 4, 5
- [51] Yuncong Yang, Jiawei Ma, Shiyuan Huang, Long Chen, Xudong Lin, Guangxing Han, and Shih-Fu Chang. TempCLR: Temporal alignment representation with contrastive learning. In The Eleventh International Conference on Learning Representations, 2023. 2
- [52] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 6
- [53] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2
- [54] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022. 12

HatefulMemes

|Average<br><br>|ImageNet<br><br>Food-101<br><br>CIFAR10<br><br>CIFAR100<br><br>CUB<br><br>SUN397<br><br>Cars<br><br>Aircraft<br><br>DTD<br><br>Pets<br><br>Caltech-101<br><br>Flowers<br><br>MNIST<br><br>FER-2013<br><br>STL-10<br><br>EuroSAT<br><br>RESISC45<br><br>GTSRB<br><br>KITTI<br><br>Country211<br><br>PCAM<br><br>UCF101<br><br>Kinetics700<br><br>CLEVR<br><br>HatefulMemes<br><br>SST2|
|---|---|
|MetaCLIP 59.8|67.6 82.6 95.2 77.7 67.8 66.8 77.2 26.9 58.9 90.9 92.5 69.7 42.7 48.3 96.3 49.9 66.5 39.2 29.3 17.7 50.0 68.0 47.6 19.4 53.5 53.1<br><br>|
|MoDE-2 61.2 MoDE-4 61.7 MoDE-8 63.4 MoDE-16 64.0 MoDE-32 64.0<br><br>|68.7 84.1 95.3 78.6 69.5 67.0 80.8 30.9 60.6 91.0 92.9 71.9 40.8 50.4 96.3 51.3 67.9 44.2 31.4 18.3 51.3 69.0 47.4 23.2 52.6 54.4<br><br>68.8 85.8 95.2 79.0 74.4 67.5 83.3 29.5 60.3 91.9 92.9 72.1 49.7 46.9 96.4 50.3 66.8 51.6 28.5 19.6 50.1 68.4 48.3 21.6 52.6 52.2<br><br><br>69.3 88.1 95.6 80.1 76.0 68.2 87.7 46.7 60.9 91.2 93.4 77.1 46.5 47.2 97.1 58.3 67.7 52.7 27.4 18.5 50.1 68.6 48.2 25.2 53.3 52.1<br><br>70.7 88.4 96.1 80.5 80.8 67.9 87.1 44.6 59.9 92.2 93.2 79.4 50.1 49.8 97.1 60.3 67.7 48.5 26.1 18.9 55.3 68.1 48.3 25.7 54.1 51.9 69.6 88.2 95.9 80.8 80.1 68.3 88.9 44.1 59.9 92.6 93.5 83.0 42.9 46.9 97.4 56.2 67.3 48.8 30.7 19.2 58.0 68.2 48.1 30.6 53.5 50.2<br><br><br>|

Caltech-101

Kinetics700

Country211

RESISC45

CIFAR100

FER-2013

Food-101

CIFAR10

ImageNet

EuroSAT

SUN397

UCF101

CLEVR

GTSRB

Flowers

Average

Aircraft

STL-10

MNIST

PCAM

KITTI

SST2

CUB

DTD

Cars

Pets

Table 11. Performance details of Fig. 3 when scaling MoDE-n based on ViT-B/32 on 2.5B image-caption pairs.

Text Retrieval Image Retrieval Text Retrieval Image Retrieval COCO Flickr30k COCO Flickr30k COCO Flickr30k COCO Flickr30k

Approach

R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 ViT-B/32

|OpenAI CLIP OpenCLIP MetaCLIP<br><br>|50.2 75.0 83.5 78.9 94.9 98.2 52.5 76.8 84.7 78.8 94.1 97.0<br><br>51.8 76.4 84.7 77.8 93.5 97.1<br><br><br>|30.4 56.0 66.9 58.8 83.6 90.0 35.3 60.9 71.7 61.7 85.5 90.9 35.9 61.8 72.1 62.3 85.5 91.5|- - - - - 56.3 79.8 87.1 84.1 96.2 98.3 55.2 78.9 86.5 80.7 95.2 97.3<br><br>|- - - - - 39.3 65.4 75.6 66.7 88.4 93.1 38.1 64.1 74.3 65.1 87.7 92.7|
|---|---|---|---|---|
|MoDE-2 MoDE-4<br><br>|53.3 76.7 84.8 78.6 94.3 96.9 53.7 77.2 85.1 78.5 94.9 96.8<br><br>|36.4 62.1 72.6 63.0 86.1 91.8 36.7 62.5 73.0 63.6 86.4 91.7<br><br>|56.7 80.2 87.5 82.8 95.1 98.2<br><br>57.4 80.1 87.3 82.9 95.6 97.7<br>|39.5 65.3 75.3 66.4 89.0 93.6 39.9 66.1 75.7 66.7 88.4 93.3<br><br>|

ViT-B/16

|OpenAI CLIP OpenCLIP MetaCLIP<br><br>|52.4 76.7 84.6 86.2 98.0 99.5<br><br>55.4 79.7 86.9 83.4 96.8 98.5<br><br>56.4 79.9 87.1 85.7 97.2 98.7<br><br><br>|33.1 58.4 69.0 69.8 90.4 94.6 38.4 63.6 73.9 65.7 88.3 93.0 40.0 65.3 75.3 67.6 89.6 94.2<br><br>|- - - - - 59.5 81.8 88.6 86.2 98.0 99.5 59.4 80.6 87.8 85.5 97.4 98.9<br><br>|- - - - - 42.3 67.7 77.1 69.8 90.4 94.6 41.4 67.2 76.9 70.7 90.8 94.5|
|---|---|---|---|---|
|MoDE-2 MoDE-4<br><br>|57.5 80.3 87.6 86.5 97.0 98.8 57.7 81.1 88.1 86.6 97.5 98.8<br><br>|40.4 65.6 75.6 68.7 89.4 94.2<br><br>41.0 66.2 75.8 68.7 90.0 94.2<br><br><br>|60.7 82.6 89.0 87.3 97.6 99.2 62.7 82.9 89.8 89.4 98.0 99.6<br><br>|43.1 68.6 77.8 72.1 91.8 95.3<br><br>44.1 69.5 78.7 72.6 91.8 95.4<br><br><br>|

ViT-L/14

|OpenAI CLIP OpenCLIP MetaCLIP<br><br>|56.3 79.4 86.6 85.2 97.4 99.2<br><br>59.7 82.2 89.4 87.6 97.8 99.5<br><br>60.0 82.9 89.4 86.2 98.1 99.6<br><br><br>|36.5 61.0 71.1 64.9 87.2 92.0 43.0 68.0 77.4 70.2 90.9 94.6 43.8 68.7 77.8 73.4 92.3 95.7<br><br>|- - - - - -<br><br>63.3 83.9 90.8 89.5 98.7 99.4<br>64.4 85.0 91.3 90.1 98.6 99.3<br>|- - - - - -<br><br>46.5 71.1 79.8 75.5 92.9 95.9<br><br>47.1 71.4 80.3 76.5 93.6 96.5<br>|
|---|---|---|---|---|
|MoDE-2 MoDE-4<br><br>|60.8 82.6 89.4 87.7 98.1 99.5 60.6 82.9 89.1 86.7 97.9 99.7<br><br>|44.2 68.6 77.8 73.4 92.5 95.8<br><br>44.2 68.6 77.8 73.5 92.1 95.9<br><br><br>|65.2 85.3 91.6 90.9 98.9 99.6 65.5 85.4 91.8 91.2 99.0 99.7<br><br>|47.9 72.1 80.6 77.2 93.7 96.6<br><br>48.2 72.4 80.7 77.6 93.7 96.7<br><br><br>|
|Pre-Train Dataset:|400M Image-Caption Pairs| |OpenCLIP:2.3B, MetaCLIP / MetaCLIP: 2.5B| |

Table 12. Zero-shot Retrieval. The results are separated by the scale of pre-train set. Entries in blue are the best ones.

### A. Full Results

Below we provide the complete results of MoDE reported in Sec. 4 if mentioned. Firstly, we compare the performance on CLIP evaluation benchmark, and reports the scores by scaling up the number of coarse-grained clusters in Table 11. When more data experts are learned, the average accuracy on CLIP benchmark keeps improving.

Secondly, we summarize the results for zero-shot retrieval in Table 12. The results are separated by the scale of pre-train dataset. Consistently, our approach can outperform the MetaCLIP Baseline in all cases. MoDE also achieves the best score in most cases.

We noticed the work LiMoE [35] which follows conventional Deep Mixture of Expert models and trains a stack of Transformer MoE layers on all 3.6B image-caption pairs [54]. However, the number of parameters in a single LiMoE network is much larger than a single dense baseline. As all of the network parameters are trained synchronously, it will cause huge memory usage. Meanwhile, comparing with MoDE-4 trained on different data clusters while the total pre-train set has only about 2.5B image-caption pairs, our system is more flexible and also achieve better results consistently.

| |ViT-B/32 ViT-B/16 ViT-L∗|
|---|---|
|classification (ImageNet)|LiMoE 67.5 73.7 78.6 MoDE-4 68.9 74.3 79.4<br><br>|
|text retrieval (CoCo)|LiMoE 45.7 51.3 55.7 MoDE-4 57.4 62.7 65.6<br><br>|
|image retrieval (CoCo)<br><br>|LiMoE 31.0 36.2 39.6 MoDE-4 39.9 44.1 48.2|

∗: LiMoE uses L/16 and MoDE uses L/14.

Table 13. Comparison with LiMOE [54]

### B. Ablation Study Details for Clustering

Firstly, for ablation details on Clustering Strategy in Sec. 5.2, we show details in Table 14 for Table 6 and Table 15 for Fig. 5.

Then, for the embedding types, we provide the details of MoDE-2 in Table 16. We note that the SimCSE [12] can be trained via unsupervised or supervised ways. The unsupervised training strategy utilizes dropout masks to generate two views from the same sentence to build positive pair while the latter one uses two sentences which are of similar semantic meaning as positive samples to each other. Regardless the training strategy, we found the average score on CLIP benchmark is the same.

HatefulMemes

Caltech-101

Kinetics700

Country211

RESISC45

CIFAR100

FER-2013

Food-101

CIFAR10

ImageNet

EuroSAT

SUN397

UCF101

CLEVR

GTSRB

Flowers

Average

Aircraft

MNIST

STL-10

PCAM

KITTI

SST2

CUB

DTD

Cars

Pets

400M Image-Caption Pairs

- MetaCLIP 58.2 65.5 80.6 91.3 70.2 63.4 63.0 70.7 26.8 52.8 88.7 91.9 68.5 41.5 35.9 95.4 52.6 64.2 35.8 30.7 17.2 55.5 66.1 45.4 30.6 56.4 53.4

- OneStep-2 58.0 65.0 80.4 91.3 69.9 62.2 62.5 69.0 27.1 52.7 88.5 91.7 67.3 40.2 32.3 95.0 54.8 63.9 36.2 36.6 16.7 54.5 66.4 45.1 26.4 57.9 54.0 CoarseCluster-2 58.5 66.1 81.1 91.0 70.6 65.3 63.1 71.8 27.1 53.5 89.0 92.2 68.7 45.2 33.5 95.4 52.0 63.7 34.9 34.2 17.3 54.3 66.1 45.5 29.3 56.6 54.6 MoDE-2 58.6 66.1 81.2 90.9 70.5 65.2 63.0 72.0 28.3 53.5 89.4 92.3 68.2 45.2 33.5 95.4 51.9 63.7 34.9 34.2 17.3 54.3 65.9 45.5 29.3 56.6 54.6 CoarseCluster-4 58.7 66.2 82.2 91.2 70.8 67.4 63.2 73.7 28.2 54.0 89.7 92.2 69.8 38.1 33.2 95.6 53.5 64.0 35.2 33.8 17.8 53.2 66.4 45.7 29.9 57.1 53.3 MoDE-4 59.0 66.4 82.3 91.3 70.9 67.0 63.7 73.8 30.1 52.6 89.9 92.1 69.2 37.9 33.2 95.7 53.5 64.1 35.2 33.9 17.1 58.4 66.6 45.9 30.0 58.0 54.5 2.5B Image-Caption Pairs

MetaCLIP 59.8 67.7 82.6 95.2 77.7 67.8 66.8 77.2 26.9 58.9 90.9 92.5 69.7 42.7 48.3 96.3 49.9 66.5 39.2 29.3 17.7 50.0 68.0 47.6 19.4 53.5 53.1

- OneStep-2 59.8 67.6 82.3 94.8 77.5 67.8 66.3 76.8 26.4 58.1 90.9 92.0 68.7 45.1 47.6 96.1 50.1 65.9 43.6 29.8 17.6 49.7 67.7 47.2 20.2 53.6 51.8 CoarseCluster-2 60.6 68.6 81.9 95.1 77.8 68.7 68.0 77.8 27.3 57.2 90.3 92.6 68.4 44.7 50.3 96.3 50.7 67.2 47.1 33.2 18.4 50.6 69.6 48.2 19.1 52.5 53.2 MoDE-2 61.2 68.7 84.1 95.3 78.6 69.5 67.0 80.8 30.9 60.6 91.0 92.9 71.9 40.8 50.4 96.3 51.3 67.9 44.2 31.4 18.3 51.3 69.0 47.4 23.2 52.6 54.4

CoarseCluster-4 61.3 69.1 83.9 95.1 78.1 73.1 67.5 82.2 27.8 60.4 90.9 92.9 70.1 49.7 46.9 96.2 50.4 66.4 50.4 28.4 18.8 50.0 68.8 48.1 21.6 52.7 52.9 MoDE-4 61.7 68.8 85.8 95.2 79.0 74.4 67.5 83.3 29.5 60.3 91.9 92.9 72.1 49.7 46.9 96.4 50.3 66.8 51.6 28.5 19.6 50.1 68.4 48.3 21.6 52.6 52.2

- Table 14. Performance details for ablation study on clustering strategy in Table 6 (Sec. 5.2). The experiments are performed on ViT-B/32. The results are separated by the scale of pre-train set.

|Average|ImageNet<br><br>Food-101<br><br>CIFAR10<br><br>CIFAR100<br><br>CUB<br><br>SUN397<br><br>Cars<br><br>Aircraft<br><br>DTD<br><br>Pets<br><br>Caltech-101<br><br>Flowers<br><br>MNIST<br><br>FER-2013<br><br>STL-10<br><br>EuroSAT<br><br>RESISC45<br><br>GTSRB<br><br>KITTI<br><br>Country211<br><br>PCAM<br><br>UCF101<br><br>Kinetics700<br><br>CLEVR<br><br>HatefulMemes<br><br>SST2<br><br>|
|---|---|
|MetaCLIP 58.2|65.5 80.6 91.3 70.2 63.4 63.0 70.7 26.8 52.8 88.7 91.9 68.5 41.5 35.9 95.4 52.6 64.2 35.8 30.7 17.2 55.5 66.1 45.4 30.6 56.4 53.4<br><br>|
|m = 2 58.0 m = 4 58.2 m = 32 58.4 m = 256 58.4 m = 512 58.5 m = 1024 58.6 m = 2048 58.7<br><br>|65.0 80.4 91.3 69.9 62.2 62.5 69.0 27.1 52.7 88.5 91.7 67.3 40.2 32.3 95.0 54.8 63.9 36.2 36.6 16.7 54.5 66.4 45.1 26.4 57.9 54.0<br><br>65.2 81.2 91.1 70.1 62.6 63.0 72.1 28.2 53.5 89.1 92.1 69.0 37.7 33.5 95.3 53.0 63.6 36.0 35.5 17.6 54.2 65.8 45.0 28.2 56.4 55.0<br><br>66.0 81.2 91.4 70.7 64.9 63.1 71.7 28.0 53.2 88.8 92.1 69.0 38.1 32.9 95.4 53.1 64.2 36.3 34.0 17.4 54.1 66.0 45.6 29.0 56.3 55.2<br><br><br>65.9 80.6 91.2 70.2 64.2 63.5 70.5 27.2 52.6 88.8 92.2 68.5 40.0 35.2 95.3 53.5 64.0 39.5 34.9 17.2 53.7 66.0 45.6 28.3 56.0 54.4<br><br>65.9 81.2 91.2 70.3 64.6 63.6 72.0 29.0 52.5 89.2 92.0 69.7 40.0 35.3 95.5 52.9 63.2 39.1 34.9 17.1 53.7 66.1 45.2 27.4 56.0 54.6<br><br><br>66.1 81.2 90.9 70.5 65.2 63.0 72.0 28.3 53.5 89.4 92.3 68.2 45.2 33.5 95.4 51.9 63.7 34.9 34.2 17.3 54.3 65.9 45.5 29.3 56.6 54.6 65.8 81.4 91.2 70.4 66.1 63.3 72.1 29.6 51.5 89.1 92.4 70.2 43.0 33.2 95.1 53.1 63.8 32.9 32.7 17.1 57.9 66.7 45.2 31.5 56.1 54.9<br><br><br>|

- Table 15. Performance details of MoDE-2 when ablating the number of finegrained clusters in Step 1 (Fig. 5 in Sec 5.2). Experiments are performed on ViT-B/32 on 400M image-caption pairs.

|Average|ImageNet<br><br>Food-101<br><br>CIFAR10<br><br>CIFAR100<br><br>CUB<br><br>SUN397<br><br>Cars<br><br>Aircraft<br><br>DTD<br><br>Pets<br><br>Caltech-101<br><br>Flowers<br><br>MNIST<br><br>FER-2013<br><br>STL-10<br><br>EuroSAT<br><br>RESISC45<br><br>GTSRB<br><br>KITTI<br><br>Country211<br><br>PCAM<br><br>UCF101<br><br>Kinetics700<br><br>CLEVR<br><br>HatefulMemes<br><br>SST2<br><br>|
|---|---|
|MetaCLIP 58.2|65.5 80.6 91.3 70.2 63.4 63.0 70.7 26.8 52.8 88.7 91.9 68.5 41.5 35.9 95.4 52.6 64.2 35.8 30.7 17.2 55.5 66.1 45.4 30.6 56.4 53.4<br><br>|
|DINOv2 58.1 Image (CLIP Seed) 58.3 Image & Lang. (CLIP Seed) 58.4 Lang. (CLIP Seed) 58.3 SimCSE-UnSup 58.6 SimCSE-Sup 58.6<br><br>|65.2 80.5 91.2 70.3 63.4 63.1 69.8 26.5 51.6 89.0 91.8 68.1 41.0 36.4 95.2 53.4 63.0 37.3 35.0 16.7 53.7 65.6 45.4 26.8 56.0 53.5<br><br>64.7 80.6 91.3 70.7 63.0 63.0 70.8 27.4 53.4 87.8 92.1 68.9 42.2 33.2 95.2 53.6 62.4 38.8 34.4 16.9 61.6 65.9 45.2 20.3 57.8 55.6<br><br>65.5 80.3 91.3 70.2 63.4 63.0 70.3 27.7 52.0 88.7 91.8 68.3 40.0 35.3 95.1 54.4 64.4 38.9 36.0 16.7 54.0 66.2 45.7 27.4 56.6 54.6 65.2 80.7 91.3 69.8 64.8 62.6 71.9 26.9 52.3 88.8 91.7 68.6 39.0 34.1 95.2 54.1 63.1 38.1 33.8 16.8 54.8 66.1 45.2 27.6 57.5 55.8<br><br><br>65.7 80.3 91.4 69.6 64.4 63.0 71.8 26.6 52.0 88.9 92.1 69.2 41.0 37.7 95.4 54.4 64.2 39.0 35.1 17.3 53.5 66.3 45.6 26.8 56.8 55.5<br><br>66.1 81.2 90.9 70.5 65.2 63.0 72.0 28.3 53.5 89.4 92.3 68.2 45.2 33.5 95.4 51.9 63.7 34.9 34.2 17.3 54.3 65.9 45.5 29.3 56.6 54.6<br><br><br>|

- Table 16. Performance details on CLIP evaluation benchmark for ablating the embedding types for clustering (Table 7 in Sec. 5.3). The experiments evaluate MoDE-2 based on ViT-B/32 on 400M image-caption pairs.

Meanwhile, when both image and language embeddings are used for clustering, we concatenate their embeddings and we experimentally found that adding the language and image embeddings pair-wisely cannot result in meaningful cluster. However, at inference time, the ensembling weights should be calculated for all image-class pairs in the zeroshot classification task, which is computational heavy but provides very limited gain over the baseline.

### C. Robustness in Training Priority

For the retrieval-enhanced setup, besides using the class names of a single dataset to retrieve the most important finegrained data clusters, we can also use the class names of all

tasks in CLIP benchmark. The detailed results are summarized in Table 17.

### D. Robustness of Vision Encoders

For emsembling over model outputs, we can also add the model outputs element-wisely. However, as all vision encoders are separately trained, the learned embedding spaces are not necessarily aligned with each other. As a result, ensembling via element-wise addition does not introduce gain, e.g., for MoDE-4 with ViT-B/16 encoders, the accuracy is only 74.5 compared with 79.6 in Table 9.

Finally, in addition to directly aggregate the feature outputs by all data experts, the parameters learned in MoDE

HatefulMemes

Caltech-101

Kinetics700

Country211

RESISC45

CIFAR100

FER-2013

Food-101

CIFAR10

ImageNet

EuroSAT

SUN397

UCF101

CLEVR

GTSRB

Flowers

Average

Aircraft

MNIST

STL-10

PCAM

KITTI

SST2

CUB

DTD

Cars

Pets

ViT-B/32

|OpenAI CLIP 56.6 OpenCLIP 61.5 MetaCLIP 59.8|63.4 83.7 89.8 65.1 53.7 62.0 59.7 19.6 44.0 87.2 87.4 66.9 48.2 46.6 97.1 44.9 61.0 32.6 28.7 17.2 62.5 63.9 48.0 23.6 56.4 58.6<br><br>66.6 82.0 93.6 75.8 66.0 68.3 86.0 23.9 56.1 90.5 91.9 70.5 70.0 50.4 96.6 49.3 65.7 49.3 32.7 16.7 51.7 64.9 45.6 24.2 52.4 57.2<br><br>67.6 82.6 95.2 77.7 67.8 66.8 77.2 26.9 58.9 90.9 92.5 69.7 42.7 48.3 96.3 49.9 66.5 39.2 29.3 17.7 50.0 68.0 47.6 19.4 53.5 53.1<br><br><br>|
|---|---|
|Ours 61.9<br><br>|70.1 85.4 95.7 80.1 74.4 67.0 81.2 36.4 58.5 91.4 93.5 72.7 44.7 42.2 96.8 53.0 69.1 41.8 35.8 18.6 58.7 69.8 48.9 21.7 49.7 51.3<br><br>|

ViT-B/16

|OpenAI CLIP 59.6 OpenCLIP 62.4 MetaCLIP 63.5<br><br>|68.3 88.8 90.8 68.2 55.6 64.0 64.6 24.0 45.1 88.9 89.1 69.4 51.8 53.0 98.2 54.8 65.5 43.3 21.7 22.8 56.3 68.5 52.3 25.5 58.7 60.5 70.2 86.2 94.9 76.9 70.5 70.6 88.2 26.6 56.3 90.4 93.1 71.0 65.8 53.3 97.9 55.2 68.3 48.3 11.9 20.3 51.2 68.1 48.9 24.8 53.0 59.5 72.1 88.3 95.7 79.0 71.4 68.5 82.9 30.3 62.1 91.7 93.3 73.9 66.1 47.0 98.4 51.1 71.1 46.6 16.6 22.7 50.5 73.0 52.5 30.8 57.4 59.0<br><br>|
|---|---|
|Ours 64.8<br><br>|74.0 89.8 96.3 81.2 76.2 69.4 85.3 39.1 58.4 92.8 93.8 75.9 57.4 48.3 98.6 54.8 72.3 46.5 28.0 23.3 50.0 74.3 53.4 29.2 57.8 58.4<br><br>|

ViT-L/14

|OpenAI CLIP 65.7 OpenCLIP 65.7 MetaCLIP 69.8<br><br>|75.5 93.0 95.6 78.3 63.3 66.8 77.8 31.3 55.3 93.6 93.3 79.3 76.4 56.9 99.4 61.9 70.9 50.6 19.2 31.9 50.1 75.7 60.2 22.3 59.7 68.9 74.0 88.6 95.8 78.3 73.5 73.5 91.4 34.6 61.2 92.7 93.3 74.4 64.4 53.9 98.5 58.6 71.9 51.6 26.1 24.4 58.0 73.3 52.0 27.4 55.1 60.4 79.2 93.4 97.6 84.2 80.1 73.8 88.7 44.6 68.1 94.7 95.4 81.8 64.4 55.1 99.3 59.2 74.6 56.3 29.7 34.0 67.3 81.6 62.0 25.9 58.0 66.7<br><br>|
|---|---|
|Ours 70.0<br><br>|79.4 93.7 97.7 85.0 81.6 73.8 89.2 47.5 68.3 95.7 95.4 83.8 69.5 52.9 99.4 62.4 74.1 59.1 29.3 34.3 58.4 81.8 62.2 23.9 57.1 65.1<br><br>|

- Table 17. Performance on CLIP evaluation benchmark via in Retrieval-Enhanced setup. The class names of all 26 tasks are jointly used to determine the data clusters. OpenCLIP is trained on LAION-2B with 2.3B image-caption pairs. OpenAI CLIP is trained on WIT400M and its results are included here for complete result summary purpose only.

can also be ensembled via averaging and then used as initialization of a single network for finetuning. As shown in Table 18, we use ViT-B/32 vision encoder, and achieve consistent gain over MetaCLIP Baseline.

|Approach|Accuracy|
|---|---|
|MetaCLIP MoDE-2 MoDE-4 MoDE-8 MoDE-16 MoDE-32|73.7 74.0 74.2 73.9 74.1 74.1<br><br>|

Table 18. Accuracy on ImageNet via parameter averaging.

### E. Implementation Detail

Clustering. We first sample 100M captions from the 400M image-caption pairs to learn the cluster centers in an unsupervised manner. Then, we use nearest neighbor to determine the cluster assignment for all other samples in the 400M as well as 2.5B dataset. We also observed that the cluster centers learned by using less than 2M samples can also result in similar clustering assignments using spherical K-means clustering [5] via FAISS [20]. In practice, we observed that the balanced K-means clustering algorithm does not necessarily enforce strict balance regarding the distribution of the clusters. For example, for the two coarse-grained clusters on 400M dataset used to train MoDE-2, the number of samples for each cluster are around 170M and 230M respectively. Consequently, as mentioned for Random-2 in Sec. 5.1, mimic the size of subsets by MoDE-2 in the random splitting for fair comparison.

Similarity matrix. For task-level adaptation, as mentioned in Sec. 3.4, we use the nearest neighbor fine-grained

cluster (arg maxs∈S Al,s) for each class l ∈ L. In other words, we apply a maximum filter for each row, i.e., Al, where the non-maximum values are reset as 0, i.e., Al,s′ = 0 if s′ ̸= sˆ where sˆ = arg maxs∈S Al,s. Then, we scale the raw distance value 5 times, i.e., setting the temperature (divisor) as 0.2, according to our experimental cross validation.

Routing Weights. As described in Eq. (6), the routing weight p(c|T) of a data expert f(·|c) is essentially obtained via softmax normalization. At inference time, we note the routing weights should be reasonably distant from each other. Consequently, given the classification task with the class names L, we use the number of classes |L| to roughly adjust the weights. Firstly, when |L| is small, e.g., |L| < 10, though only one data expert can be activated, the selection could be sensitive to noisy routing. Then, we soften the values in A by multiplying exp(0.5 − |L|) to ensemble two data experts in most cases. On the other hand, when |L| is large, e.g., |L| > 200, the normalized weights tend to be over-smooth, we thus use a much smaller temperature by dividing the λ by log(|L|). Then, we can only select a few data experts and have low-entropy routing weights.

