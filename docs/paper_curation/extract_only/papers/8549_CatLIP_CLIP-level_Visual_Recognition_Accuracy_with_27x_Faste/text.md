## CatLIP: CLIP-level Visual Recognition Accuracy with 2.7× Faster Pre-training on Web-scale Image-Text Data

Sachin Mehta1 Maxwell Horton1 Fartash Faghri1 Mohammad Hossein Sekhavat1 Mahyar Najibi1 Mehrdad Farajtabar1 Oncel Tuzel1 Mohammad Rastegari1

# arXiv:2404.15653v1[cs.CV]24Apr2024

### Abstract

Contrastive learning has emerged as a transformative method for learning effective visual representations through the alignment of image and text embeddings. However, pairwise similarity computation in contrastive loss between image and text pairs poses computational challenges. This paper presents a novel weakly supervised pre-training of vision models on web-scale imagetext data. The proposed method reframes pretraining on image-text data as a classification task. Consequently, it eliminates the need for pairwise similarity computations in contrastive loss, achieving a remarkable 2.7× acceleration in training speed compared to contrastive learning on web-scale data. Through extensive experiments spanning diverse vision tasks, including detection and segmentation, we demonstrate that the proposed method maintains high representation quality. Our source code along with pre-trained model weights and training recipes is available at https://github.com/apple/corenet.

### 1. Introduction

The pre-training of computer vision models on large-scale datasets has enabled models to learn intricate patterns, features, and representations, leading to their effective generalization across diverse visual tasks (e.g., Krizhevsky et al., 2012; Kornblith et al., 2019; Kirillov et al., 2023). The pretraining methods offer different trade-offs. Fully supervised methods benefit from well-defined optimization objectives, enabling the model to learn strong semantic representations (Zhai et al., 2022a; Dehghani et al., 2023). However, obtaining large-scale manually labeled datasets for such methods is time-consuming and expensive. In contrast, unsupervised methods offer scalability to large datasets as they do not rely on manual labels (Chen et al., 2020; He et al., 2022). Yet,

Sachin Mehta led this project and made significant technical contributions. 1Apple.

Image encoder

Image encoder

Binary crossentropy loss

Contrastive loss

Alice is working on a computer.

Extract Synsets

Alice is working on a computer.

Text encoder

Synset vocab

(a) CatLIP

(b) CLIP

Image Relative pre-training Top-1 Accuracy Encoder time (GPU hours) (%)

CLIP

1.00× 84.4 CatLIP (Ours) 0.37× 84.3 CLIP

ViT B/16

1.00× 86.9 CatLIP (Ours) 0.36× 86.7

ViT H/16

(c) Pre-training time vs. transfer learning on ImageNet-1k.

Figure 1. CatLIP is 2.7× faster to pre-train than CLIP while maintaining down-stream accuracy. For a fair comparison, we calculate GPU hours by training CatLIP and CLIP with a batch size of 65k for one epoch of DataComp-1.3B on the same hardware configuration. Finetuning accuracy of CatLIP and CLIP is reported on ImageNet-1k dataset. Here, represents trainable backbones.

achieving competitive performance compared to supervised methods often requires the use of either larger models or longer training. Weakly-supervised approaches strike a balance by incorporating noisy supervision (e.g., captions associated with an image) while maintaining scalability (Singh

- et al., 2022; Jia et al., 2021). Notably, contrastive learning using weakly labeled data is a transformative method, especially with the increasing availability of both small- and large-scale weakly labeled image-text datasets in the public domain (Thomee et al., 2016; Sharma et al., 2018; Gadre
- et al., 2023; Schuhmann et al., 2022).

Contrastive Language-Image Pretraining (CLIP) (Radford et al., 2021) aligns image-text pairs by encouraging the similarity between input image-text pairs and penalizes similarity between unpaired samples. The process involves considering multiple negative pairs for every positive pair, leading to a large number of pairwise comparisons and making it computationally intensive. In comparison, supervised learning approaches leverage simpler and more

[dog]NN [catching]VBG [frisbee]NN

POS Tagging

Extract Noun

[dog]NN [frisbee]NN

1000

dog catching frisbee

No.ofImageNet-1ksynsets

- 103

- 104

- 105

- 106

- 107

Dataset

Dataset

No.ofsynsetoccurences

DataComp-1.3B

CC3M

Caption

800

CC3M

DataComp-1.3B

WordNet

Extract Synsets

600

ImageText Dataset

400

Create

Vocab

200

Labels (person, car)

Find Match

Extract Synsets

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0 200 400 600 800 1000

Synset similarity threshold

ImageNet-1k synset Id

Target dataset

Synset vocabulary

(a)

(b)

(c)

- Figure 2. ImageNet-1k labels appear frequently in image-text datasets, contributing to increased zero-shot classification accuracies of CLIP models. (a) shows the process of finding labels of the target dataset in the image-text dataset. (b) shows the number of ImageNet-1k synsets at different similarity thresholds between synset vocabulary and ImageNet-1k synset. Exact match (similarity score of 1.0) for approximately 40% of ImageNet-1k labels is found in the DataComp-1.3B captions. (c) illustrates occurrences of ImageNet-1k synsets in image-text datasets, with larger datasets exhibiting more samples per synset. Here, the vocabulary pruning threshold, Vτ, is set to 500.

computationally efficient objective functions, such as the cross-entropy loss. This avoids the need for a global collection of negative samples for effective learning, addressing computational challenges inherent in contrastive learning. Zhai et al. (2023) reduce the computational overhead of contrastive loss in CLIP by reducing the dependence of the loss on the entire batch. However, they still have to compute pairwise similarities.

Effectively harnessing the advantages of both web-scale image-text datasets and image classification frameworks for faster training while preserving representation quality remains an open question. We begin our investigation by looking into the presence of ImageNet-1k (Deng et al., 2009) classes in both small- and large-scale image-text datasets that are commonly used for training CLIP. Many of the ImageNet-1k classes emerge within the CC3M (Sharma et al., 2018) and DataComp-1.3B (Gadre et al., 2023) captions, often appearing as either exact matches or closely related concepts (Figure 2b). Moreover, they manifest in multiple instances, underscoring the knowledge of objects within these datasets (Figure 2c).

Based on these observations, we introduce a method that addresses the trade-off between efficiency and scalability on a weakly labeled web-scale image-text dataset (Figure 1a). In contrast to contrastive pre-training, we conceptualize imagetext pre-training as a classification problem where multilabels are obtained by extracting nouns from text captions. We call the proposed method, CatLIP (Categorical Loss for Image-text Pre-training). Figure 1c shows that CatLIP is 2.7× more efficient to train than CLIP while preserving downstream performance on the ImageNet-1k dataset. We present extensive experiments in Sections 4 and 5 to demonstrate the effectiveness of the proposed method.

The main contributions of our paper are:

1. We introduce a novel approach for accelerating the pre-

training of vision models on image-text data (Section 3). To the best of our knowledge, this is the first method to cast pre-training on image-text data as classification task.

- 2. CatLIP improves the accuracy with data and model scaling (Section 4). Noteworthy are the results on smallscale image-text data, where the performance of model improves with longer training when trained with CatLIP as compared to CLIP (Section 3.2).
- 3. Standard approach for transfer learning involves initializing the model backbone with pre-trained weights and randomly initializing the classifier. Because target labels can be a subset of CatLIP’s vocabulary, it enables the extraction of embeddings associated with target task labels from the classification layer of the pre-trained model. These embeddings can then be utilized to initialize the classification layer in the target task, facilitating dataefficient transfer learning (Section 4.2; Figure 5).
- 4. Through extensive experiments spanning various downstream tasks, including object detection and semantic segmentation (refer to Section 5), we demonstrate the effectiveness of representations learned by CatLIP, showing comparable performance to CLIP. As an instance, Mask R-CNN (He et al., 2017) employing a vision transformer (ViT B/16; (Dosovitskiy et al., 2020)) backbone achieved a mean average precision score of 49.9 on COCO (Lin et al., 2014) for both CatLIP and CLIP. We highlight that CLIP requires 2.7× more time for pre-training compared to CatLIP on DataComp-1.3B (Figure 1c).

### 2. Related Work

Learning visual representations at scale. Several studies have demonstrated the benefits of pre-training models on large-scale datasets. The first line of research focuses on leveraging large-scale labeled datasets for training vision models (e.g., ImageNet-1k, JFT 300M (Dosovitskiy et al.,

2020), and JFT-3B (Zhai et al., 2022a)). This approach relies on explicit supervision, enabling models to acquire robust semantic representations. However, it comes with the drawback of requiring extensive manual labeling for constructing large-scale datasets. The second line of research focuses on leveraging unlabeled data (Chen et al., 2020; He et al.,

- 2022). While these approaches offer scalability, achieving competitive performance against supervised models often necessitates the use of larger models or longer training. The third line of research focuses on weak supervision (Radford et al., 2021; Jia et al., 2021; Singh et al., 2022), aiming to strike a trade-off between fully supervised learning, which provides good accuracy, and unsupervised learning, which offers scalability.

Our work falls in the category of weakly-supervised learning from image-text data. Unlike previous works that align image-text pairs using contrastive learning, we formulate learning on image-text datasets as a classification problem. This paradigm shift in pre-training allows us to address the computational bottlenecks in contrastive learning and reduce pre-training time by 2.7× while preserving accuracy on downstream tasks.

Efficient contrastive image-text pre-training. The majority of methods for pre-training on image-text data follow contrastive learning (e.g., Radford et al., 2021; Jia et al., 2021; Xu et al., 2021; Schuhmann et al., 2022; Cherti et al.,

- 2023). However, this approach is computationally expensive due to the computation of pairwise similarities for each image and text embedding. While some efforts have been made to enhance the training efficiency of image-text pre-training, they may not necessarily address the specific bottlenecks associated with contrastive learning.

Pre-trained encoders. LiT (Zhai et al., 2022b) aligns image and text embeddings with a frozen image encoder and a learnable text encoder. APE (Rosenfeld et al., 2022) uses frozen image and text encoders to extract image and text embeddings, which are then aligned using a small multi-layer perceptron. Although these methods may not explicitly tackle the computational challenges associated with contrastive learning, they reduce alignment time by optimizing fewer parameters, and are complementary to our work.

ViT-specific optimizations. FLIP (Li et al., 2023) extends masked autoencoders (He et al., 2022) for contrastive learning. It involves randomly masking image tokens and subsequently aligning text tokens with unmasked image tokens using contrastive learning. Hardware-level optimizations (Dao et al., 2022; Rabe & Staats, 2021) further reduce the memory footprint of attention in transformers and can aid in accelerating CLIP training. While this model-specific strategy accelerates training, it doesn’t directly tackle the bottlenecks associated with contrastive learning. These methods

complement our work, offering potential enhancements in training efficiency for specific models.

Better objective functions. SigLIP (Zhai et al., 2023) mitigated the computational overhead of the softmax-based contrastive loss in CLIP by calculating local pairwise similarities through a sigmoid loss. However, the extent to which local sigmoid loss enhances training efficiency in comparison to softmax-based loss remains unclear. UniCL (Yang et al., 2022) tries to unify supervised and weaklysupervised datasets by constructing image-text-label triplets. While it can improve accuracy, it still needs to compute pair-wise similarities, making the pre-training slower and more expensive.

In contrast to existing contrastive image-text pre-training approaches, our work reframes image-text pre-training as a classification task, significantly reducing pre-training time while preserving CLIP-level accuracy in downstream tasks.

Improving large-scale pre-training. There are works, not specifically tailored to image-text pre-training, that aim to enhance pre-training efficiency. Examples include the use of better optimizers (Rajbhandari et al., 2020; Chen et al., 2023), adopting distributed training strategies (Zhao et al., 2023), employing mixed-precision training techniques (Micikevicius et al., 2017), and incorporating gradient accumulation (Chen et al., 2016). These efforts are complementary to our work.

### 3. CatLIP: Contrastive to Categorical Learning

For a given image-text dataset D = (Ii,Ti) | i ∈ [1,N] with N pairs of images I and text captions T, CLIP aligns the image and text embeddings obtained from independent image and text encoders. This alignment is achieved through a contrastive loss which calculates pair-wise similarities between image and text embeddings. However, the computation of global pair-wise similarities is computationally demanding.

This paper casts image-text pre-training as a classification task. Our approach involves extracting nouns from text captions and mapping them to WordNet synsets (Miller, 1995) (Section 3.1). We conduct pre-training on the CC3M dataset using a vision transformer model of Dosovitskiy et al. (2020), and assess it’s performance through linear probing accuracy on the ImageNet-1k dataset. Unlike CLIP, the proposed method exhibits non-saturation with extended training (Section 3.2).

##### 3.1. Caption-derived classification labels

The abundance of noisy image-text data crawled from the web has enabled large-scale contrastive pre-training, leading to models improved generalization on visual recogni-

108

Dataset

No.ofsynsetoccurences

DataComp-1.3B

106

CC3M

104

102

100

0 10000 20000 30000 40000

Synset Id

(a) Synset distribution

30000

Dataset

No.ofuniquesynsets

25000

DataComp-1.3B

CC3M

20000

15000

10000

5000

0

100 500 1000 2500 5000

Vocabulary pruning threshold

(b) No. of synsets vs. Vτ

[Figure 1]

[Figure 2]

CC3M DataComp-1.3B

(c) Top 1000 synset names

- Figure 3. Analysis of extracted WordNet synsets in image-text datasets. Larger datasets typically contain a greater number of synsets, indicating increased content diversity in larger datasets.

tion tasks. However, it is important to note that image-text pre-training is computationally intensive, particularly when compared with supervised pre-training for image classification on a similar scale of data (e.g., JFT-3B). On a positive note, the process of collecting image-text datasets at webscale is more cost-effective compared to collecting labeled datasets.

Leveraging the benefits of both web-scale image-text datasets and image classification frameworks effectively for faster training while preserving representation quality is an open question. We performed a simple analysis to answer this question. An overview of our approach is shown in Figure 2a. For a given text caption T = (w1,··· ,wm) with a sequence of m words, we tag each word wi with partof-speech tag posi, where i is the position of word in a text caption. We then extract nouns and map them to WordNet synset as:

ExtractSynset(T) = {f(wi) | posi is a noun∀i = 1..m} (1) where f is a function that maps wi to WordNet synset S.

To examine if the object labels of downstream classification dataset are present in image-text dataset D, we extracted synset S¯ using Equation (1) for the object labels y in the downstream classification dataset D¯ =

(I¯i,yi) | i ∈ [1,M] with M pairs of images I¯ and object classification labels y. Synsets S ∈ V and S¯ are similar if sim(S,S¯) ≥ α, where sim is a WordNet’s path similarity function and α is a pre-defined similarity threshold.

The statistics for different pre-training corpora are summarized in Figure 3. With the increase in pre-training imagetext dataset size (from CC3M to DataComp-1.3B), the number of unique synsets increases (Figure 3b). This is expected, as a larger dataset typically contains a more diverse range of content, leading to a greater variety of unique objects and patterns. However, we observe that this also (1) increases the overlap with the labels in the downstream classification dataset (i.e., ImageNet-1k; Figure 2b) and (2) increases number of samples per synset (Figure 2c). This suggests

that the larger pre-training dataset have more connections and commonalities with the labels in the downstream classification datasets, ultimately resulting in improvements in zero-shot accuracy. It’s worth noting that because the model has been exposed to similar images of labels (or objects) in the downstream classification dataset during pre-training, manual template-based image retrieval (a.k.a., zero-shot image classification) in CLIP may not be truly zero-shot. The notion of zero-shot image classification remains subject to debate. However, the analysis in Figure 3 introduces an intriguing question: Can we cast image-text pre-training as a classification problem?

##### 3.2. CatLIP pre-training

We propose CatLIP, a weakly supervised pre-training method based on WordNet synset supervision to answer this question. As shown in Figure 1a, CatLIP extracts multiple synsets from input text captions and subsequently pretrains the image classification models using a binary cross-entropy loss, effectively shifting the problem from noisy image-text alignment to noisy image classification.

Synset vocabulary. The distribution of synsets in imagetext datasets follows a long-tail distribution (Figure 3a). We create a vocabulary V : S → C by counting the number of occurrences C of the synset S in given image-text dataset. Because some of the synsets will be less frequent, we prune V by storing synsets whose count C is greater than a predefined vocabulary pruning threshold Vτ. To determine a good value of Vτ, we train CatLIP with the base version of vision transformer (ViT B/16) as an image backbone on the CC3M dataset and evaluated the linear probe accuracy on the ImageNet-1k dataset. Results in Appendix B shows that Vτ = 100 or 500 is a good pruning threshold. In our experiments, we use Vτ = 500.

CatLIP vs. CLIP. Considering pre-training on image-text datasets as a classification problem, can CatLIP serve as a viable alternative to CLIP pre-training? To evaluate the

60

Top-1Acc.(%)

55

CLIP

50

CatLIP (Ours)

45

40

25 50 75 100

Pre-training epochs

(a) Acc. vs. pre-training epochs

60

CLIP

Top-1Acc.(%)

CatLIP (Ours)

55

50

4x less compute

45

40

0 50 100 150 200 250 300

Pre-training GPU hours

(b) Acc. vs. pre-training time

- Figure 4. CatLIP vs. CLIP. Unlike CLIP, CatLIP benefits from longer training on CC3M dataset and requires significantly less pre-training computation. Here, (a) shows the top-1 linear probe accuracy of CLIP and CatLIP on ImageNet-1k as a function of CC3M pre-training epochs while (b) shows the linear probe accuracy as a function of pre-training duration. Here, each dot in a graph represents an independently trained model on CC3M. See Appendix A for training details.

comparative advantages of CatLIP over CLIP, we conduct pre-training experiments with CatLIP and CLIP on smallscale image-text dataset, i.e., CC3M, using ViT B/16 as an image backbone. We note that discussions on large-scale data and model scaling experiments are in Section 4. We make following observations from results in Figure 4.

First, the accuracy of the ViT B/16 model pre-trained with CatLIP and linear probed on ImageNet-1k improves as training is extended. In contrast, the accuracy of CLIP plateaus after a certain training duration. This observation aligns with previous works (e.g., Mu et al., 2022), reinforcing the notion that CLIP requires large pre-training datasets to achieve optimal performance.

Second, CatLIP demonstrates faster pre-training as compared to CLIP. This is likely because CatLIP optimizes only an image backbone during pre-training while CLIP optimizes both image and text backbones. Consequently, CatLIP has several advantages over CLIP in terms of training efficiency: (1) it has faster step time (forward-backward pass) due to fewer network parameters to optimize, (2) requires less GPU memory for the same batch size, (3) benefits from GPU-optimized loss function implementations, and (4) has less communication overhead in multi-node training as it does not require synchronization between GPUs for computing global pair-wise similarities in contrastive loss.

### 4. Data and Model Scaling in CatLIP

Standard pre-training methods scale both data and model size to enhance accuracy. Section 3 shows that CatLIP is effective on small-scale datasets. This section studies the effect of data (Section 4.1) and model (Section 4.2) scaling on CatLIP (see Appendix A for training details). We assess the quality of learned representations within the framework of transfer learning through (1) linear probing (LP), achieved by training a linear classifier on frozen image

Table 1. Data scaling in CatLIP. Scaling the image-text dataset from three million to 1.3 billion samples improves transfer learning accuracy of ViT B/16, both with linear probing (LP) and full finetuning (FT) on ImageNet-1k and Places365.

ImageNet-1k Top-1 (%) Places365 Top-1 (%) LP FT LP FT

Dataset # samples

CC3M 2.9 M 58.0 67.3 49.1 54.5 DataComp-1.3B 1.3 B 80.1 84.3 53.6 59.2

backbone weights and (2) full fine-tuning (FT), achieved by fine-tuning the entire model. We do LP and FT evaluations on two standard image classification datasets, i.e., ImageNet1k and Places365 (Zhou et al., 2017a). For training details including hyper-parameters, see Appendix A.

##### 4.1. Data scaling

We pre-train CatLIP with ViT B/16 as an image backbone on two publicly available image-text datasets (i.e., CC3M and DataComp-1.3B). Table 1 shows the linear probing (LP) and fine-tuning (FT) results of these pre-trained models. Increasing the data size from three million samples in CC3M to 1.3 billion samples in DataComp-1.3B improves the accuracy on transfer datasets. Consistent with previous works, this observation indicates that scaling data improves model’s representation quality when pre-trained with CatLIP.

##### 4.2. Model scaling

We pre-train CatLIP with different variants of vision transformer (ViT-B/16, ViT-L/16, and ViT-H/16) on the DataComp-1.3B dataset. Figure 5 shows the linear probe evaluation on the ImageNet-1k and Places365 datasets for different variants of ViT, respectively. We make following observations.

Representation quality improves with model size. Scaling up the model size improves the representation quality. However, the performance of the largest model (i.e., ViT H/16) starts to saturate on CatLIP (ViT B/16 vs. ViT L/16 vs. ViT H/16: 80.8 vs. 85.2 vs. 86.1; Figure 5a). These results on ImageNet-1k are consistent with scaling laws for ViT on large-scale supervised datasets (Zhai et al., 2022a). We also observe similar trends on Places365 dataset (Figure 5b).

Better transfer learning with bigger models. Linear probing on task-specific datasets shows that bigger models exhibit better transfer learning capabilities, especially in small data regime (red color curves in Figure 5). For example, the linear probe top-1 accuracy of ViT B/16, ViT L/16, and ViT H/16 when trained with 1% of ImageNet-1k data is about 16%, 26%, and 36% respectively. This highlights their capacity to leverage pre-existing knowledge and generalize effectively on small datasets.

ViT B/16 ViT L/16 ViT H/16

80

80

80

Top-1Acc.(%)

Top-1Acc.(%)

Top-1Acc.(%)

60

60

60

40

40

40

Classiﬁer Initialization

Classiﬁer Initialization

Classiﬁer Initialization

Random Init Transfer Init

Random Init Transfer Init

Random Init Transfer Init

20

20

20

1 5 10 15 20 25 50 75 100

1 5 10 15 20 25 50 75 100

1 5 10 15 20 25 50 75 100

Training data (%)

Training data (%)

Training data (%)

60

50

Top-1Acc.(%)

40

30

Classiﬁer Initialization

Random Init Transfer Init

20

1 5 10 15 20 25 50 75 100

Training data (%)

(a) ImageNet-1k

60

50

Top-1Acc.(%)

40

30

Classiﬁer Initialization

Random Init Transfer Init

20

1 5 10 15 20 25 50 75 100

Training data (%)

60

50

Top-1Acc.(%)

40

30

Classiﬁer Initialization

Random Init Transfer Init

20

1 5 10 15 20 25 50 75 100

Training data (%)

(b) Places365

- Figure 5. Linear probe transfer learning with CatLIP is more data efficient. Here, transfer learning is achieved by training a linear classifier on downstream tasks for 30 epochs with frozen image backbone features (each dot in a graph is an independent run). Linear classifier is initialized either using a standard method (Random Init) or transferred classifier embeddings from a pre-trained model (Transfer Init; ours). Initializing classifier with Transfer Init delivers better accuracy than Random Init, especially in small data regime.

Data efficient transfer learning. Many target labels in downstream classification datasets are English nouns. Therefore, we can map target labels and the synset vocabulary V based on their WordNet path similarity score, as shown in Figure 2a. Subsequently, extracting the embeddings associated with target labels from the classification layer of the pre-trained model, for initializing the classification layer of the target task, is a natural and logical step. We refer to this classifier initialization as Transfer Init and compare it with random classifier initialization (Random Init).

We make two observations from results in Figure 5: (1) In the context of the ImageNet-1k dataset, all downstream task labels are subset of V (Figure 2b). Consequently, Transfer Init yields significantly better accuracy, especially in small data regimes (Figure 5a). This performance improvement aligns with expectations as the pre-training on large corpora has exposed the model to images of similar objects. (2) In the case of the Places365 dataset, Transfer Init continues to result in improved accuracy, however, gains are less substantial compared to observation (1). This disparity is attributed to the fact that Places365 have target labels that are either not a subset of V (e.g., archaeological excavation)1 or comprise a combination of multiple synsets in V (e.g.,

1Initialized with random embeddings.

restaurant kitchen)2. As a consequence, model requires more data to adapt to these unseen labels or linearly interpolated embeddings, resulting in a less pronounced improvement compared to more aligned labels in ImageNet.

##### 4.3. Comparison with existing pre-training methods

Our weakly supervised method, CatLIP, is compared with state-of-the-art methods in Table 2. We compare pretraining parameter count, pre-training data size, image resolution during pre-training and fine-tuning, and top-1 accuracy on two standard datasets: ImageNet-1k and Places365. We group these methods into two categories: supervised and weakly-supervised. We classify models pre-trained on JFT as supervised because limited information is available about these datasets and their annotation process. (Zhai et al., 2022a) briefly mentions that JFT-3B has 30k classes, and is collected using a semi-automatic annotation pipeline, implying a manual curation and annotation process. Therefore, we consider JFTs as a supervised dataset, similar to (Singh et al., 2022).

Table 2 shows that CatLIP delivers competitive performance compared to existing pre-training methods. For instance, ViT B/16 pre-trained with CatLIP and CLIP on

2Initialized with an average embedding.

- Table 2. Transfer learning accuracies of ViT models pre-trained on different datasets using supervised and weakly-supervised methods. Transfer learning is achieved through fine-tuning the entire model on downstream classification tasks. Our weakly-supervised models achieve the best performance on both ImageNet-1k and Places365 classification tasks. Here, we include models pre-trained on JFT under supervised pre-training as limited information is available about their semi-automatic labeling method, similar to (Singh et al., 2022). † ViT-22B uses frozen image encoder. Here, WIT means web-crawled image-text dataset. ⋆ CoCa is a hybrid approach that uses labels from JFT-3B to create captions for image-text training along with ALIGN data. Therefore, it is not directly comparable to other approaches.

Resolution Top-1 accuracy

Model Params Pretraining

Pre. Fine. ImageNet-1k Places365 Supervised pre-training

ViT B/16 (Dosovitskiy et al., 2020) 87 M ImageNet-21k 224 384 84.0 58.2 ViT L/16 (Dosovitskiy et al., 2020) 305 M ImageNet-21k 224 384 85.2 59.0 ViT L/16 (Dosovitskiy et al., 2020) 305 M JFT 300M 224 512 87.8 ViT H/14 (Dosovitskiy et al., 2020) 634 M JFT 300M 224 512 88.6 ViT L/16 (Zhai et al., 2022a) 305 M JFT 3B 224 384 88.5

- ViT G/14 (Zhai et al., 2022a) 1.9 B JFT 3B 224 518 90.5 ViT-22B† (Dehghani et al., 2023) 22 B JFT 4B 224 224 89.5 Weakly supervised pre-training

ViT B/16 (Singh et al., 2022) 87 M IG 3.6B 224 384 85.3 59.1 ViT L/16 (Singh et al., 2022) 305 M IG 3.6B 224 512 88.1 60.7

- ViT H/16 (Singh et al., 2022) 633 M IG 3.6B 224 518 88.6 60.7 ALIGN EfficientNet-L2 (Jia et al., 2021) – ALIGN 1.8B 289 600 88.6 FLIP ViT-B (Li et al., 2023) 329 M LAION 2B 224 224 87.1 FLIP ViT-H (Li et al., 2023) 756 M LAION 2B 224 224 87.7 OpenAI CLIP ViT B/16 (Radford et al., 2021) 149 M WIT 400M 224 224 85.3 RangeAugment CLIP ViT B/16 (Mehta et al., 2022) 149 M WIT 1.2B 224 224 84.3 RangeAugment CLIP ViT H/16 (Mehta et al., 2022) 756 M WIT 1.2B 224 224 86.9 OpenCLIP ViT B/16 (Cherti et al., 2023) 149 M LAION 2B 224 224 85.5 OpenCLIP ViT L/14 (Cherti et al., 2023) 329 M LAION 2B 224 224 87.3 OpenCLIP ViT H/14 (Cherti et al., 2023) 756 M LAION 2B 224 224 87.6 CoCa ⋆ (Yu et al., 2022) 2.1 B JFT 3B + ALIGN 288 91.0 CLIP ViT B/16 (Our repro.) 149 M DataComp-1.3B 224 224 84.4 58.4 CatLIP ViT B/16 (Ours) 105 M DataComp-1.3B 224 224 84.3 59.2 CatLIP ViT L/16 (Ours) 329 M DataComp-1.3B 224 224 86.5 60.3 CatLIP ViT H/16 (Ours) 663 M DataComp-1.3B 224 224 86.7 60.2 CatLIP ViT B/16 (Ours) 105 M DataComp-1.3B 224 512 86.1 59.4 CatLIP ViT L/16 (Ours) 329 M DataComp-1.3B 224 512 88.6 60.7 CatLIP ViT H/16 (Ours) 663 M DataComp-1.3B 224 512 88.8 61.1

DataComp-1.3B achieves 84.3% and 84.4% top-1 accuracy on ImageNet-1k, and 58.4% and 59.2% on Places365 respectively. It is important to note that pre-training with CatLIP is 2.7× faster than CLIP (Figure 1c). These results suggest that CatLIP is an efficient and effective alternative to contrastive pre-training on large-scale image-text datasets.

Additionally, ViT variants trained with CatLIP are competitive to other publicly available CLIP variants, such as OpenCLIP (Cherti et al., 2023) and RangeAugment CLIP (Mehta et al., 2022), despite being pre-trained on different datasets. Also, comparison with (Singh et al., 2022) is interesting because it is also trained as a multi-label classifier on Instagram images, utilizing hashtags as labels for the images. Remarkably, CatLIP achieves similar performance to (Singh et al., 2022), but with about 2.8× smaller pre-training data.

### 5. Task Generalization

Many vision pre-training methods commonly use a singlelabel classification benchmark (e.g., ImageNet-1k) for evaluation. However, real-world images are inherently contextrich. These images often comprises of complex backgrounds, intricate scenes, and multiple objects, presenting a more nuanced and detailed visual environment than what is typically addressed in single-label classification tasks (e.g., ImageNet-1k).

To assess the effectiveness of CatLIP in such real-world scenarios, we conduct experiments on three challenging tasks: (1) multi-label object classification (Section 5.1), (2) semantic segmentation (Section 5.2), and (3) object detection (Section 5.3). Our observations show that models pre-trained with CatLIP achieve similar or better accuracy compared to CLIP on these tasks. This emphasizes that CatLIP ef-

- Table 3. Multi-label object classification on COCO. For reference, we’ve included results from state-of-the-art methods with complex architectures and loss functions in gray cells. Here, † denotes that results are from (Xu et al., 2023).

Model Image Encoder mAP (in %)

ML-Decoder† (Ridnik et al., 2023) ViT L 90.6 ADDS (Xu et al., 2023) ViT L 91.8

CLIP (Our impl.) ViT B/16 87.4 CatLIP (Ours)

ViT B/16 87.9 ViT L/16 90.6

fectively learns high-quality representations, showcasing its versatility and robustness in addressing complex visual recognition scenarios.

##### 5.1. Multi-label object classification

We evaluate the performance of models pre-trained with CatLIP on the multi-object classification task using the COCO dataset. This dataset consists of approximately 118,000 training images categorized into 80 classes, with an average of around 3 labels per image. We evaluate the performance in terms of mean average precision (mAP) on the validation set, consisting of 5,000 images.

Table 3 shows that CatLIP achieves comparable performance to CLIP. Remarkably, CatLIP, trained with a simple binary cross-entropy loss, delivers competitive accuracy compared to existing methods employing more complex techniques, such as asymmetric loss functions (Ridnik et al., 2023) and pyramidal feature extraction (Xu et al., 2023).

##### 5.2. Semantic segmentation

We evaluate the semantic segmentation performance on ADE20k (Zhou et al., 2017b). ADE20k comprises 150 semantic categories with approximately 20,000 training and 2,000 validation images. We use Deeplabv3 (Chen et al., 2017) as a segmentation head. We use mean intersection over union (mIoU) metric to measure accuracy.

The results in Table 4 show that DeepLabv3 with CatLIP’s ViT B/16 as an image encoder achieves competitive performance compared to CLIP’s ViT B/16. Interestingly, DeepLabv3 with CatLIP’s ViT image encoders delivers similar or better performance than existing methods that employ more complex segmentation heads. For example, CatLIP’s ViT L/16 with DeepLabv3 as segmentation head is about 1% more accurate than ViT Adaptor (Chen et al., 2022) with UpperNet (Xiao et al., 2018) segmentation head with similar number of network parameters.

##### 5.3. Object detection and instance segmentation

We evaluate the object detection and instance segmentation performance on the COCO dataset with Mask R-CNN as

- Table 4. Semantic segmentation on ADE20k. For reference, we have included results from state-of-the-art methods (MMSeg (Contributors, 2020), SegViT (Zhang et al., 2022), and ViT Adaptor (Chen et al., 2022)) with complex architectures (UpperNet (Xiao et al., 2018) and Semantic FPN (Kirillov et al., 2019)) in gray.

Model Image encoder Seg. head # params. Resolution mIoU MMSeg ViT B UpperNet 144 M 512 × 512 47.73 SegViT ViT L SegViT 344 M 640 × 640 54.60 ViT Adaptor ViT L Semantic FPN 332 M 512 × 512 52.90 ViT Adaptor ViT L UpperNet 332 M 512 × 512 53.40 CLIP (Our impl.) ViT B/16 DeepLabv3 99 M 512 × 512 49.70

CatLIP

ViT B/16

DeepLabv3

99 M

512 × 512

50.10 ViT L/16 320 M 54.46 ViT H/16 652 M 55.63

- Table 5. Object detection and instance segmentation on COCO with Mask R-CNN. As a reference and for fair comparison, we include results from ViTDet (Li et al., 2022) without any information propagation strategies.

Model

Image COCO

encoder APBox APMask ViTDet

ViT B 48.9 43.9 ViT L 52.9 47.2

CLIP (Our impl.) ViT B/16 49.9 43.7 CatLIP

ViT B/16 49.9 43.6 ViT L/16 52.7 46.0 ViT H/16 53.4 46.4

detection head. The dataset comprises of 118,000 training and 5000 validation images. We measure the accuracy in terms of mean average precision metric (mAP @ IoU of 0.50:0.95).

Results are given in Table 5. We see that Mask R-CNN with CatLIP’s ViT B/16 image backbone delivers similar performance to CLIP’s ViT B/16 backbone. We also observe that Mask RCNN’s mAP increases when we scale image backbone from ViT B/16 to ViT H/16, and we obtain results similar to existing method (Li et al., 2022). These results show that CatLIP learns high quality representations.

- 6. Conclusion

This paper introduces a weakly-supervised pre-training method for vision models using publicly available web-scale image-text data. By recasting pre-training as a classification task, the proposed method addresses the computational challenges associated with contrastive learning, leading to an impressive 2.7× training speed-up on web-scale data while preserving transfer learning accuracies across varieties of visual recognition tasks, including detection and segmentation. This work contributes significantly to the efficient and effective pre-training of vision models. We hope this work will facilitate efficient pre-training research on web-scale noisy data.

### Broader Impact

Large-scale pre-training on image-text datasets have led to a series of breakthroughs in computer vision and related fields. However, large-scale pre-training is still challenging because it requires significant computational resources. This paper provides insights into the efficiency of image-text pre-training, and proposes a reformulation with significantly lower computational requirements. We show the training efficiency gains while preserving transfer accuracies on several visual recognition tasks. We will open-source our code and pre-trained models to enable future research. We hope that our work will make research in pre-training accessible. Otherwise, we expect that it will have the same broader impact, both positive and negative, as the field as a whole.

### References

Chen, L.-C., Papandreou, G., Schroff, F., and Adam, H. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.

Chen, T., Xu, B., Zhang, C., and Guestrin, C. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PMLR, 2020.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023.

Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., and Qiao, Y. Vision transformer adapter for dense predictions. arXiv preprint arXiv:2205.08534, 2022.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829, 2023.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pp. 7480–7512. PMLR, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Gadre, S. Y., Ilharco, G., Fang, A., Hayase, J., Smyrnis, G., Nguyen, T., Marten, R., Wortsman, M., Ghosh, D., Zhang, J., et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.

He, K., Gkioxari, G., Doll´ar, P., and Girshick, R. Mask rcnn. In Proceedings of the IEEE international conference on computer vision, pp. 2961–2969, 2017.

He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., and Girshick, R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.

Kirillov, A., Girshick, R., He, K., and Doll´ar, P. Panoptic feature pyramid networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6399–6408, 2019.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.

Contributors, M. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https:// github.com/open-mmlab/mmsegmentation, 2020.

Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2661–2671, 2019.

Dao, T., Fu, D., Ermon, S., Rudra, A., and R´e, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Li, Y., Mao, H., Girshick, R., and He, K. Exploring plain vision transformer backbones for object detection. In European Conference on Computer Vision, pp. 280–296. Springer, 2022.

Li, Y., Fan, H., Hu, R., Feichtenhofer, C., and He, K. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23390–23400, 2023.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740– 755. Springer, 2014.

Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Mehta, S. and Rastegari, M. Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer. arXiv preprint arXiv:2110.02178, 2021.

Mehta, S., Naderiparizi, S., Faghri, F., Horton, M., Chen, L., Farhadi, A., Tuzel, O., and Rastegari, M. Rangeaugment: Efficient online augmentation with range learning. arXiv preprint arXiv:2212.10553, 2022.

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.

Miller, G. A. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39–41, 1995.

Mu, N., Kirillov, A., Wagner, D., and Xie, S. Slip: Selfsupervision meets language-image pre-training. In European Conference on Computer Vision, pp. 529–544. Springer, 2022.

Rabe, M. N. and Staats, C. Self-attention Does Not Need O(n2) Memory. arXiv preprint arXiv:2112.05682, 2021.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.

Ridnik, T., Sharir, G., Ben-Cohen, A., Ben-Baruch, E., and Noy, A. Ml-decoder: Scalable and versatile classification head. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 32–41, 2023.

Rosenfeld, E., Nakkiran, P., Pouransari, H., Tuzel, O., and Faghri, F. Ape: Aligning pretrained encoders to quickly learn aligned multimodal representations. arXiv preprint arXiv:2210.03927, 2022.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Sharma, P., Ding, N., Goodman, S., and Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2556– 2565, 2018.

Singh, M., Gustafson, L., Adcock, A., de Freitas Reis, V., Gedik, B., Kosaraju, R. P., Mahajan, D., Girshick, R., Doll´ar, P., and Van Der Maaten, L. Revisiting weakly supervised pre-training of visual perception models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 804–814, 2022.

Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., and Li, L.-J. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.

Xiao, T., Liu, Y., Zhou, B., Jiang, Y., and Sun, J. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV), pp. 418–434, 2018.

Xu, H., Ghosh, G., Huang, P.-Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., and Feichtenhofer, C. Videoclip: Contrastive pre-training for zero-shot videotext understanding. arXiv preprint arXiv:2109.14084, 2021.

Xu, S., Li, Y., Hsiao, J., Ho, C., and Qi, Z. Open vocabulary multi-label classification with dual-modal decoder on aligned visual-textual features, 2023.

Yang, J., Li, C., Zhang, P., Xiao, B., Liu, C., Yuan, L., and Gao, J. Unified contrastive learning in image-text-label space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19163– 19173, 2022.

Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y. Coca: Contrastive captioners are imagetext foundation models. arXiv preprint arXiv:2205.01917, 2022.

Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12104–12113, 2022a.

Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., and Beyer, L. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18123–18133, 2022b.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. arXiv preprint arXiv:2303.15343, 2023.

Zhang, B., Tian, Z., Tang, Q., Chu, X., Wei, X., Shen, C., et al. Segvit: Semantic segmentation with plain vision transformers. Advances in Neural Information Processing Systems, 35:4971–4982, 2022.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017a.

Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., and Torralba, A. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 633–641, 2017b.

### A. Training Details

Pre-training on image-text datasets. Table 6 lists the hyper-parameters used for pre-training on CC3M and DataComp1.3B with CatLIP. Following (Mehta et al., 2022), we use a multi-scale variable batch sampler (Mehta & Rastegari, 2021) with a base image resolution of 224 × 224. We use AdamW (Loshchilov & Hutter, 2017) with default β values as an optimizer and binary cross-entropy loss as an objective function. We use cosine learning rate schedule (Loshchilov & Hutter, 2016). For augmentation, we use random resized crop with default scale and aspect ratios, random horizontal flipping, and RangeAugment (Mehta et al., 2022).

Transfer learning on image classification. Table 7 lists the hyper-parameters used for pre-training on ImageNet-1k, Places365, and COCO. We use a multi-scale variable batch sampler with a base image resolution of 224 × 224. We use AdamW with default β values as an optimizer. For single-label classification (ImageNet-1k and Places365), we use cross-entropy loss with label smoothing as an objective function. For multi-label classification (COCO), we use binary cross entropy as an objective function. We use cosine learning rate schedule. For high resolution experiments in Table 2 and multi-label classification experiments in Table 3, we use 512 × 512 as the base image resolution.

Transfer learning on semantic segmentation and object detection. Table 7 lists the hyper-parameters used for pretraining on ADE20k for semantic segmentation and COCO for object detection. For segmentation, we use random short size resize, random crop, and horizontal flipping as augmentation along with RangeAugment. We use a fixed-size batch sampler with a base image resolution of 512 × 512. We use AdamW with default β values as an optimizer. We use cross-entropy as an objective function and cosine schedule for annealing learning rate.

For detection, we use the same augmentations as used in ViTDet (Li et al., 2022). We use a multi-scale variable batch sampler with a base image resolution of 1024 × 1024. We use AdamW with default β values as an optimizer. We use the same loss function as used in ViTDet for regressing bounding box regression and object name classifcation. For learning

Table 6. Pre-training hyper-parameters for CLIP and CatLIP. (a) CC3M experiments in Figure 4

(b) DataComp-1.3B

Hyperparameter CLIP CatLIP

Backbone ViT B/16 ViT B/16 Max. Epochs 25/50/75/100 25/50/75/100 LR scheduler Cosine Cosine Max. LR 0.0005 0.002 Min. LR 0.000001 0.00002 Optimizer AdamW AdamW AdamW β’s (0.9, 0.98) (0.9, 0.999) Weight decay 0.2 0.2 Batch size per GPU 512 512 # A100 GPUs 4 4 A100 GPU Memory 40 GB 40 GB

Hyperparameter CLIP CatLIP

ViT B/16 ViT B/16 ViT L/16 ViT H/16

Max. Iterations 200000 200000 200000 200000 LR scheduler Cosine Cosine Cosine Cosine Max. LR 0.0005 0.001 0.0006 0.0004 Min. LR 0.000001 0.00001 0.000006 0.000004 Optimizer AdamW AdamW AdamW AdamW AdamW β’s (0.9, 0.98) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) Weight decay 0.2 0.2 0.2 0.2 Batch size per GPU 256 1024 512 256 # A100 GPUs 256 64 128 256 A100 GPU Memory 80 GB 80 GB 40 GB 40 GB

Table 7. Transfer learning hyper-parameters image classification tasks. Here, LP and FT means linear probing and full finetuning respectively.

(a) LP on ImageNet-1k and Places365.

Hyperparameter Value

Backbone ViT B/L/H Epochs 30 LR scheduler Cosine Max. LR 0.00003 Min. LR 0.000001 Optimizer AdamW Weight decay 0 Batch size per GPU 256 # A100 GPUs 1 A100 GPU Memory 80 GB

(b) FT on ImageNet-1k and Places365.

Hyperparameter Value

Backbone ViT B/L/H Epochs 10 LR scheduler Cosine Max. LR 0.00003 Min. LR 0.000003 Optimizer AdamW Weight decay 0.05 Batch size per GPU 128 # A100 GPUs 1 A100 GPU Memory 80 GB

(c) FT on COCO.

Hyperparameter Value

Backbone ViT B/L/H Epochs 10 LR scheduler Cosine Max. LR 0.00001 Min. LR 0.000001 Optimizer AdamW Weight decay 0.05 Batch size per GPU 32 # A100 GPUs 1 A100 GPU Memory 80 GB

Table 8. Transfer learning hyper-parameters for semantic segmentation and object detection. (a) Semantic segmentation on ADE20k.

(b) Object detection on COCO.

#### Hyperparameter Value

Seg. Head DeepLabv3 Image Backbone ViT B/L/H Epochs 50 LR scheduler Cosine Max. LR 0.00003 Min. LR 0.000003 Optimizer AdamW Weight decay 0.1 Batch size per GPU 4 # A100 GPUs 8 A100 GPU Memory 80 GB

###### Hyperparameter Value

Det. Head Mask R-CNN Image Backbone ViT B/L/H Epochs 25/25/20 Optimizer AdamW Weight decay 0.1 LR scheduler Multi-step Max. LR 0003 LR step size 0.1 LR step interval (22, 24) / (22, 24) / (17, 19) Layer-wise LR decay 0.7/0.8/0.9 Stochastic dropout 0.1/0.4/0.5 Batch size per GPU 2 # A100 GPUs 64 A100 GPU Memory 80 GB

rate annealing, we use multi-step learning rate schedule.

### B. Additional results

Effect of vocabulary pruning threshold. The distribution of synsets in image-text datasets demonstrates a long-tail pattern, necessitating the pruning of infrequently occurring synsets. In Figure 6, we show the results obtained by pruning the synset vocabulary V at various thresholds, denoted as Vτ. For each value of Vτ, we train an independent model using CatLIP on CC3M. Here, we use ViT-B as the image encoder.

To determine the optimal pruning threshold, we conduct linear probing experiments on ImageNet-1k. Notably, larger values of Vτ lead to a significant decline in performance, likely attributed to the removal of crucial information about objects at such elevated thresholds. We found that Vτ = 100 or Vτ = 500 are good pruning thresholds. Therefore, in our experiments involving both CC3M and DataComp-1.3B, we use Vτ = 500 as the threshold.

Data efficient transfer learning. Section 4 highlights the effectiveness of Transfer Init, showcasing its effictiveness in transferring information from the pre-trained classification layer using CatLIP to the downstream task’s classification layer through linear probing on ImageNet-1k. In Figure 7, a comparable trend to linear probing is observed when finetuning the entire model on ImageNet-1k at varying data percentages. However, these trends are not as pronounced as in linear probing, as anticipated. This discrepancy can be attributed to the backbone’s adaptability to new labeled data during fine-tuning, in

58

Top-1LinearProbeAcc.(%)

56

54

52

50

100 500 1000 2500 5000

Vocabulary pruning threshold

- Figure 6. Linear probing accuracy on ImageNet-1k as a function of vocabulary pruning threshold Vτ. Here, each data point represents a ViT B/16 model pretrained on CC3M at different values of vocabulary pruning thresholds Vτ.

80

80

80

Top-1Acc.(%)

Top-1Acc.(%)

Top-1Acc.(%)

60

60

60

80

80

70

40

40

40

Classiﬁer Initialization Random

Classiﬁer Initialization

Classiﬁer Initialization

60

70

1 5 10

1 5 10

Random Init Transfer Init

Random Init Transfer Init

20

20

20

Transfer (Ours)

0

0

0

1 5 10 15 20 25 50 75 100

1 5 10 15 20 25 50 75 100

1 5 10 15 20 25 50 75 100

Training data (%)

Training data (%)

Training data (%)

(a) ViT-B

(b) ViT-L

(c) ViT-H

- Figure 7. Transfer accuracy of models pre-training using CatLIP on ImageNet-1k. Transfer learning is achieved by finetuning the entire model on ImageNet-1k for 10 epochs. Linear classifier is initialized either using a standard method (Random Init) or transferred classifier embeddings from a pre-trained model (Transfer Init). Transfer Init delivers better accuracy than Random Init, especially in small data regimes.

- 15

- 16

- 17

- 18

- 19

- 20

Model

ViT B ViT L ViT H

Loss

1 2 3 4 5 6 7 8 9 10 11 12

Epoch

Figure 8. Training loss vs. epoch for different variants of ViT models, pre-trained with CatLIP on DataComp-1.3B.

contrast to the frozen backbone in linear probing.

Pre-training loss. Training loss during pre-training on DataComp-1.3B is shown in Figure 8 for different variants of ViT. Loss decreases with increase in model size. This observed correlation aligns with the transfer learning accuracy of different ViT models on downstream tasks (Sections 4 and 5). Importantly, the loss is not saturated, suggesting that longer training will further improve performance.

