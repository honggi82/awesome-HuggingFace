## Transferable and Principled Efficiency for Open-Vocabulary Segmentation

Jingxuan Xu1, Wuyang Chen2, Yao Zhao1,3, Yunchao Wei1,3 1Beijing Jiaotong University 2Simon Fraser University 3Peng Cheng Laboratory

# arXiv:2404.07448v3[cs.CV]17Sep2024

### Abstract

Recent success of pre-trained foundation vision-language models makes Open-Vocabulary Segmentation (OVS) possible. Despite the promising performance, this approach introduces heavy computational overheads for two challenges: 1) large model sizes of the backbone; 2) expensive costs during the fine-tuning. These challenges hinder this OVS strategy from being widely applicable and affordable in real-world scenarios. Although traditional methods such as model compression and efficient fine-tuning can address these challenges, they often rely on heuristics. This means that their solutions cannot be easily transferred and necessitate re-training on different models, which comes at a cost. In the context of efficient OVS, we target achieving performance that is comparable to or even better than prior OVS works based on large vision-language foundation models, by utilizing smaller models that incur lower training costs. The core strategy is to make our efficiency principled and thus seamlessly transferable from one OVS framework to others without further customization. Comprehensive experiments on diverse OVS benchmarks demonstrate our superior trade-off between segmentation accuracy and computation costs over previous works. Our code is available on https://github.com/Xujxyang/OpenTrans

### 1. Introduction

Open-vocabulary segmentation has been developed to address the limitations of traditional semantic segmentation [2, 22, 42, 50], which cannot identify categories beyond its training set, by leveraging human-like recognition of novel categories using arbitrary text descriptions. This method enables the segmentation of arbitrary categories from text inputs, expanding its use in areas like image editing and human-robot interaction. Early models used cross-modal alignment between pixel and text embeddings [25, 43, 48], while recent methods have improved performance with region-level alignment [10, 15, 19, 28, 44]. These methods learn from base class embeddings and are often paired with a frozen pre-trained vision-language foundation model, such as CLIP [37], to assist in re-classification, with strate-

[Figure 1]

Figure 1. Comparison with popular open-vocabulary segmentation works on Pascal-Context [36] (with Resnet50 as backbone). The red points represent our works, the orange points stand for traditional two-stage open-vocabulary segmentation [10, 44], the blue point stands for single-stage open-vocabulary segmentation [17], the green denotes a novel dual-channel prediction model with frozen backbone [47], the purple represent traditional convolutional segmentation networks [2].

gies that incorporate knowledge distillation or a frozen CLIP backbone to maintain a generalizable representation while fine-tuning on known classes.

Despite the promising performance of recent OVS frameworks, these methods introduce heavy computational over-

heads. The core bottleneck to the efficiency of recent OVS frameworks is rooted in the challenging balance between the preservation of the generalization ability of pretrained CLIP model versus the overhead of model size and training costs. First, as the pretrained CLIP model supports the extraction of generalizable features of unseen images, the entire heavy image encoder has to be inherited for the OVS task. This introduces the requirement for the large model size of the CLIP image encoder. Second, after inheriting from CLIP, one needs to fine-tune together with the segmentation head. This further requires expensive training computations, and careful and delicate fine-tuning to preserve the pretrained generalizable knowledge in the CLIP image encoder. These challenges hinder this OVS strategy from being more widely applicable and affordable in real-world scenarios.

Traditional methods, including model compression [3, 4, 13] and efficient fine-tuning [20, 21, 39], can mitigate these challenges. For example, iterative magnitude pruning can find semantic-aware lottery tickets that introduce highly sparse networks adapted to downstream tasks. Adapterbased fine-tuning can partially update pretrained models by introducing lightweight modules. Despite the efficiency, however, these methods come with a price of heuristics: their solutions cannot be easily transferred and have to be retrained on different models. One has to re-run the pruning to find task-specific and model-specific sparse masks that can hardly be transferred to different frameworks. During the fine-tuning, adapter layers have to be empirically customized and inserted into pretrained models, which introduces extra computation costs. Therefore, we are motivated to ask our core question:

Can we design principled methods to make OVS efficient, and seamlessly transferable to different frameworks?

This work targets achieving performance as competitive as large vision-language foundation models but using smaller models with less training costs. We propose a Transferable Open-vocabulary segmentation technic (dubbed OpenTrans) , to establish a principled and seamlessly transferable efficiency across various OVS frameworks, eliminating the need for additional customization. This enables us to achieve the “one design, transferable to all” fashion. First, to address the large model size of the CLIP image encoder, we explore small subnetwork by iterative magnitude pruning. To avoid overfitting the seen OVS classes and finding sparse masks specific to OVS frameworks, we prune the CLIP image encoder without semantic supervision. This strategy enables our pruning approach to discover highly transferable subnetwork that seamlessly adapt to the latest OVS frameworks without requiring any modifications, avoiding any further pruning costs. Second, to address the heavy OVS fine-tuning computation, we propose to only select partial layers to update. To enable a “plug-and-play” strategy for layer-wise selective fine-tuning, we target analyzing the quality of pre-

trained layers without any data or any forward/backward process. We investigate the spectrum of pretrained weights, freeze layers with heavy tails, and only update those with light tails. This principled analysis not only further accelerates the training with fewer computations, but also introduces zero overhead in the pretrained model and can be directly adopted in different OVS frameworks. We conduct comprehensive experiments on diverse OVS benchmarks, our method can employ 21.2M smaller networks (Parameters), 59.11P less training costs (Training FLOPs).

We summarize our contributions below:

- • Transferable sparse model: We reduce the backbone for 21.2M smaller in parameters and reduce 94.6G FLOPs. We avoid extra pruning costs on diverse latest OVS frameworks by directly transferring our subnetwork.
- • Principled efficient fine-tuning: We design a plug-andplay layer-wise selection approach to enable efficient finetuning that reduces the training costs in OVS.
- • Superior mIoU-efficiency trade-off: Extensive experiments demonstrate that we achieve comparable or even improved mIoU over diverse OVS benchmarks with significantly reduced computation costs for both training and inference.

### 2. Related work

#### 2.1. Vision-Language Pre-training Model

Vision-language pre-training is a pre-training approach that integrates language and vision domains. In its early stages [26, 27], research in this field was initially limited to smaller datasets, requiring subsequent fine-tuning on downstream tasks. Recent studies emphasize the advantages of leveraging the extensive pool of web data accessible to researchers. Through the implementation of contrastive learning, CLIP [37] establishes meaningful connections between images and their corresponding captions, achieving notable performance in cross-modal alignment. Alin [23] adopts a straightforward dual-encoder architecture and employs a contrastive loss to align the visual and linguistic representations of image and text pairs. The advent of these methods has played a substantial role in advancing Open-Vocabulary Segmentation. Despite the promising performance, visionlanguage pre-trained models still impose significant computational overheads, primarily attributed to the large sizes of the backbone. This poses challenges in terms of wide applicability and affordability. In contrast, we utilize CLIP as a source of open-domain capabilities, but propose efficiency to its image encoder with transferable sparse masks.

#### 2.2. Open-Vocabulary Segmentation

The purpose of OVS is to segment diverse classes, including those that are inaccessible during the training process. The developmental trajectory of OVS can be delineated into two

phases: pre- and post-introduction of vision-language pretraining models. Research conducted before the advent of vision-language pre-training models, as observed in works such as [1, 43], often yield suboptimal results. Therefore, integrating these models into OVS emerges as a highly advantageous strategy [11, 24, 28]. Just after CLIP came out, Simbaseline [44] adopts proposal masks for cropping the input image and leverages CLIP to extract region-level features. Zegformer [10] utilizes CLIP as its encoder and incorporates MaskFormer [5] for extracting mask proposals, both demonstrating improved performance in OVS. Recently, CAT-Seg [7] introduces a spatial and class aggregation network, integrating multi-modality guidance features to enhance open-vocabulary pixel classification effectiveness, achieving state-of-the-art performance on part datasets. FC-CLIP [47] proposes a structure with two classifiers and freezes the CLIP backbone to uphold open-vocabulary recognition, also attaining state-of-the-art performance on certain datasets. Han et al. [17] develop an efficient framework that avoids relying on the extra computational burden of the CLIP model. In our research, our focus is on advancing principled efficiency for OVS frameworks. This is accomplished through the implementation of transferable sparse masks and efficient fine-tuning, employing spectrum analysis of pretrained weights.

#### 2.3. Parameter-Efficient Transfer Learning

Parameter-Efficient Transfer Learning (PETL) is a research direction focused on minimizing the computational burden associated with adapting large pretrained models to new tasks. This is achieved by selectively updating specific parameters, eliminating the necessity to update the entire model. Adapters [20, 39] are compact bottleneck modules that are inserted within transformer layers.Empirical studies have demonstrated that using layer normalization layers alone for training adapters is adequate to achieve optimal performance during the fine-tuning process. SideAdapters [40, 45] partitions the trainable parameters of the backbone model and forms a distinct side network. The designated task for this side network is to adapt the entire model to novel tasks, leading to its specific design tailored to different structures and tasks. Following a similar pattern, LoRA [21] introduces trainable rank decomposition matrices into a pre-trained model that remains fixed. Although the majority of advancements in PETL have been observed in the NLP domain, researchers also extend the application of this technique to the fields of vision-language (VL) [41, 45, 49, 52]. However, the adapter-based efficient fine-tuning methods still have to introduce extra layers or modules to the pretrained layers, which require extra design and customization efforts. Instead, we aim to propose a principled and plug-and-play efficient fine-tuning method that can be seamlessly adopted in different OVS frameworks.

### 3. Method

#### 3.1. Preliminaries

General OVS Pipeline. OVS aims at training a segmentation model capable of segmenting arbitrary classes using text descriptions. Most of the existing methods decoupled OVS into two steps: mask proposals generation and mask proposals classification. They first involve MaskFormer[6] to generate class-agnostic mask proposals, followed by leveraging CLIP’s transferability to classify these proposals. The mask proposals classification process can be defined as:

Ci = vi T = softmax((vi ∗t1),(vi ∗t2)···(vi ∗t|C|)),

(1) where C is the classification score. vi is the vision embeddings for i mask proposals, which executed with either masked crops [10, 44] or masked attention [45]. T is the text embeddings from CLIP-pretrained text encoder. Building upon this inspiration, Our framework maintains the utilization of the Mask2Former architecture..

During training, in order to preserve the generalizability of CLIP, we employ a distillation loss [17] (“Knowledge Distillation Loss” in Fig. 2, denoted as LKD) to distill the CLIP’s knowledge to our backbone. The distillation loss guarantees that the distance between text embeddings and visual embeddings remains consistent across all pairwise categories. The alignment of the CLIP text space with the image feature space enhances the performance of open vocabulary, while being agnostic to any semantic class. Refer to the supplementary material for details. In our work, we introduced this loss to enhance the open-domain characteristics of our pruning method.

Out Target. In this work, we target introducing two levels of efficiency to current OVS works: both model and training efficiency. More importantly, our core contribution lies in not only establishing these efficiency principles but also ensuring their transferability across various OVS frameworks. We will introduce two steps (see Fig. 2):

- 1. Model efficiency (Sec. 3.2): We prune the heavy CLIP image encoder without semantic awareness to make the backbone transferable to different OVS frameworks.
- 2. Training efficiency (Sec. 3.3): During fine-tuning, we only partially update selective layers of the backbone to further reduce training costs.
- 3.2. Model Efficiency

In this section, our main objective is to improve model efficiency through model pruning. Specifically, we aim to answer the question: Can we find efficient subnetwork that can be seamlessly transferred to different OVS frameworks? We focus on finding a subnetwork within the model’s backbone that allows for the seamless transfer of its sparse masks across different OVS works.

[Figure 2]

- Figure 2. Overview of OpenTrans(Ours). We introduce principled and transferable efficiency in two folds. Step1: we prune the heavy CLIP image encoder without semantic awareness, this turns the backbone into a semantic-agnostic transferable subnetwork. This allows us to seamlessly transfer subnetwork to other OVS Frameworks, such as Deeplabv3 [2], Mask2former [6] and FC-CLIP [47]. Step2: during fine-tuning, we further explore and prioritize principled efficiency by introducing layer-wise heavy-tail spectrum analysis. This method involves selectively updating layers with light-tail spectra in their pretrained weights, while keeping layers with heavy-tail spectra frozen.

Pruning method. To identify the pruning mask for the backbone, we utilize the classical Iterative Magnitude Pruning (IMP) approach, as described in the LTH (Lottery Ticket Hypothesis) literature [3, 4, 13, 14]. The pruning of the model’s backbone is carried out in a two-step process. First, we train an unpruned network as the pre-training model. Then, we remove a portion of weights with the globally smallest magnitudes [4, 18, 38].

We acknowledge the crucial role of the knowledge distillation loss LKD in aligning feature spaces between CLIP text and vision features, which significantly impacts the model’s open-vocabulary performance. Hence, during the pruning process to find the pruning mask, we solely utilize the knowledge distillation loss and exclude any semantic supervision. This approach also allows us to remain agnostic to any specific OVS framework. More importantly, it helps us discover a pruning mask on one OVS framework that can be transferable to other frameworks without any further customization, while still preserving the open-domain capabilities in CLIP without any risk of overfitting to specific semantic classes.

We denote our backbone as f(x;θ), where x is the input and θ ∈ Rd as its parameters. The subnetwork of the backbone can be described as f(x;m ⊙ θ) with a pruned binary mask m ∈ {0,1}d, where ⊙ is the element-wise product. We depict our pruning outlines in Algorithm 1, the sparsity level indicates the remaining weights of the model.

Subnetwork Transfer. Upon obtaining the pruning mask

Algorithm 1 Semantic-agnostic Model Pruning.

- 1: Set the initial mask to m = 1d, with pre-trained parameters θ.
- 2: repeat
- 3: Train f(x; m ⊙ θ) for t iterations with only the distillation loss LKD.
- 4: Prune p% of remaining weights in θ and update m accordingly.
- 5: until the sparsity of m reaches the desired sparsity level s.
- 6: Return f(x; m ⊙ θ).

through IMP, we proceed to transfer it to other OVS frameworks (Fig. 3) by applying it to different segmentation backbones [2, 6, 47]. Importantly, no fine-tuning or additional adjustments are required when incorporating our subnetwork into these alternative frameworks, as it seamlessly integrates with the model.

To demonstrate that our transferable subnetwork can be widely adopted, we choose to study three representative and diverse OVS frameworks:

- • Traditional CNN-based OVS: In the traditional convolutional segmentation model, CLIP text embeddings can be

utilized as the classifier. Thus, in Eq. (1), the vi represents the vision embeddings for the i-th pixel. We choose DeeplabV3 [2] as our transfer target, which incorporates the ASPP module to enhance its performance.

- • Mask2Former-based OVS: This line of work separates the process of acquiring masks and predicting mask cat-

[Figure 3]

- Figure 3. We transfer our semantic-agnostic sparse masks to different OVS pipelines. Left: Traditional CNN-based frameworks: these frameworks commonly employ simple upsampling techniques. Middle: Mask2Former-based frameworks, the approach utilizes a decoupled mask and class segmentation head. Right: OVS with extra classifiers: some frameworks propose to leverage a frozen CLIP classifier to further utilize the pretrained visual knowledge (with the blue box indicating frozen classifier layers of the model).

egories, resulting in higher quality masks compared to CNN methods. We choose Han et al. [17] as our transfer target.

• OVS with extra classifiers: Beyond Mask2Former, many recent OVS frameworks additionally utilize an extra pretrained classifier obtained from CLIP. We choose FCCLIP [47] as our target, which uses an out-of-vocabulary classifier during inference and greatly helps in OVS.

In our experiments (Sec. 4.2), we will show that by adopting our transferable subnetwork, we can facilitate the efficiency of all three types of OVS frameworks. After transferring the subnetwork, the sparsified pretrained backbone will be fine-tuned with CLIP text embeddings in the next section.

#### 3.3. Training Efficiency

In current OVS frameworks, heavy training costs are a common issue. The fine-tuning stage typically updates all layers without considering the quality of different pretrained weights, resulting in inefficiencies. In this section, our objective is to address the question: Can we make the fine-tuning of OVS frameworks principled and efficient? To achieve this, we propose a principled and explainable metric that guides the fine-tuning of pretrained OVS models. Importantly, our approach does not introduce any extra parameters and does not incur any overhead during training.

##### 3.3.1 Fine-tuning with Selective Layers

We target a plug-and-play efficient fine-tuning, with minimal extra computation costs and without introducing any extra layers. Our strategy is simple and principled: during fine-tuning process, we selectively calculate gradients and

[Figure 4]

Figure 4. We partially fine-tune layers of our pretrained backbone by analyzing heavy-tail spectrum of pretrained weights. We first fine-tune the whole model for 104 iterations. Then, we compute α values for all layers of the model. During fine-tuning, we freeze layers with small α values (indicating good pretrained quality) and only fine-tune layers with large α (bad pretrained quality).

update layers that are still under-trained (weights of worse pretrained quality), while freeze and skip calculating gradients of layers that are already well-trained (weights of good pretrained quality), thus saving computation costs during fine-tuning. We will defer how to determine the quality of pretrained layers in Sec. 3.3.2. We depict our selective layer-wise fine-tuning in Fig. 4.

##### 3.3.2 Layer Selections via Heavy-tail Analysis

We assess the quality of pretrained weights and determine under-trained/well-trained layers by analyzing their heavytail behaviors in the weights spectrums.

The fundamental concept of heavy-tail theory [31] is that in the empirical spectral density (ESD) of weight matrices, heavy-tail structures emerge as these matrices become more correlated and effectively capture diverse correlations in data during the optimization process [30–33, 35]. A key practical implication of this theory is the ability to forecast model performance by determining the power-law coefficient from the ESDs, which only requires the weights of the model. Studies, such as Yang et al. [46], have shown that lower coefficients typically indicate greater test accuracy.

Imagine a neural network composed of L layers, each associated with weight matrices W1, W2, ···, WL. For every weight matrix Wi, which has a dimension of N × M, assuming N ≥ M, the correlation matrix is defined as Σi = Wi⊤Wi. The eigenvalues of Σi are represented as {λj}Mj=1, where λj equals the square of the singular values of Wi, denoted as {σj}Mj=1. The largest eigenvalue of the correlation matrix Σi is referred to as λi,max. The term ESD (empirical spectral density) for the weight matrix Wi describes the empirical distribution of the eigenvalues of Σi, often visualized through a histogram. The density function that models the ESD, denoted as p(λ), is considered within the range (λmin,λmax). For a power law, p satisfies

###### p(λ) ∝ λ−α, λmin < λ < λmax. (2)

- Table 1. Model efficiency via subnetwork transfer. By discovering semantic-agnostic sparse masks, we can directly transfer the subnetwork to different OVS frameworks, significantly reducing their model sizes and computation costs, while preserving their OVS performance after fine-tuning. Random means trasnfering random subnetworks.

|Model|COCO Cityscapes ADE20K-150 ADE20K-847 PAS-20 PC-59 PC-459<br><br>|Params. FLOPs|
|---|---|---|
|Han et al. [17] Random Ours<br><br>|46.0 33.9 16.6 2.5 71.2 39.0 7.1 37.4 28.7 13.2 2.2 60.1 33.2 5.8 42.5 31.7 15.8 2.6 64.6 35.1 6.4|44.1M 268.2G 22.9M 173.3G 22.9M 48.1%↓ 173.3G 35.4%↓<br><br>|
|DeeplabV3[2] Random Ours<br><br>|26.3 20.3 8.8 - 44.1 23.9 4.1 17.9 16.3 6.4 - 30.2 16.5 2.7 34.8 24.3 10.8 - 55.2 28.9 5.2|40.3M 241.3G 19.1M 146.8G 19.1M 52.6%↓ 146.8G 39.2%↓<br><br>|
|FC-CLIP [47] Random Ours<br><br>|58.7 53.2 23.3 7.1 89.5 50.5 12.9 52.8 50.0 17.2 3.2 85.5 44.8 8.7 56.8 52.1 19.1 4.2 87.6 47.4 9.9|39.0M 200.1G 17.8M 105.6G 17.8M 54.4%↓ 105.6G 47.2%↓<br><br>|

α is essentially the slope of the tail of the ESD of the weights on a log-log scale.

A smaller α indicates a slow decay of large eigenvalues in the ESD, which tends to lead to a better quality of pretrained weights [33]. Therefore, during fine-tuning, we only update layers with large α values (i.e. weights that are not welltrained), and will freeze layers with small α values (i.e. pretrained weights of good quality). We update 50% layers of top (larger) α values, and freeze the other 50% layers of smaller α values, which helps prevent overfitting.

### 4. Experiments

Sufficient experiments are conducted to verify the effectiveness of the model. We first separately study the contribution from our transferable subnetworks (in Sec. 4.2) and our principled layer-selective fine-tuning (in Sec. 4.3). We then show our final results (in Sec. 4.4) to demonstrate our superior balance between OVS accuracy and efficiency.

#### 4.1. Implementation Details

Datasets. We conduct experiments in the following popular OVS datasets. COCO-Panoptic [29] is a large-scale semantic segmentation dataset with 133 classes. ADE20K [51] is a large-scale dataset for semantic segmentation, it contains more than 20k training images and 2k validation images. This dataset is divided into two splits. ADE20K-150 comprises 150 semantic classes, while ADE20K-857 encompasses 857 classes. Cityscapes [8] is a dataset for urban scene understanding, its’ validation set contains 500 highresolution images. PASCAL datasets [12, 36] contains two splits of datasets. Pascal VOC 2012 contains 1,449 validation images from 20 classes. Pascal Context is an extensive of Pascal VOC 2010, and it has 5005 validation images. We utilize both the commonly used PC-59 version and the more challenging PC-459 version.

We evaluate our OpenTrans in a cross dataset setting following previous works [7, 17, 45, 47]. Our subnetwork is trained on COCO and evaluated on the others. Training Strategy. Our implementation is inspired by the

work of Han et al. [17]. Throughout the model training process, we consistently utilize four NVIDIA RTX 4090 Ti GPUs. The input image is resized to 512 × 512. During the IMP stage outlined in Algorithm 1, we set the learning rate to 0.0001 without implementing any specific step schedule. The parameters s and p in the algorithm indicate a sparsity level and pruning ratio of 10%, respectively. When transferring the subnetwork to different architectures, we perform fine-tuning for several iterations. The experimental settings are adjusted accordingly, depending on the specific architectures being used. For a thorough understanding of the experimental details, please refer to the supplementary material.

Parameters and FLOPs. For precise and equitable computation, we employed a uniform calculation methodology to assess parameters and FLOPs across all tasks. Our evaluation encompasses all model parameters during testing. FLOPs were consistently computed using the initial graph from the COCO validation set with dimensions 800 × 1216.

#### 4.2. Model Efficiency via Subnetwork Transfer

In this section, we employ three exemplary methodologies [2, 17, 47] to evaluate the efficiency of the transferable subnetwork transfer strategy. We compare these three representative methods with original approaches and random subnetwork transfer strategies.

Comparisons with original OVS approaches. We prune once but contribute to the efficiency of peer works via subnetwork transfer. Application of our subnetwork yields notable reductions in Params and FLOPs for models such as Han et al., DeeplabV3, and FC-CLIP, as delineated in Tab. 1. This means that all our three models share the same subnetwork, which validates the transferability of our method. Specifically, for Han et al., we achieve a reduction in Params from 44.1M to 22.9M, and FLOPs from 268.2G to 173.3G. Despite being nearly half the size, the mIoU only saw a marginal 2.2% reduction in Cityscapes and 0.8% in ADE20K-150. In the case of DeeplabV3, we successfully halved the model size while surpassing its original performance. The improvements amounted to a 4% increase in mIoU for Cityscapes

- Table 2. Efficient Fine-tuning. By analyzing the heavy-tail spectrum in pretrained weights, our layer-selective strategy can reduce training costs in OVS frameworks in a principled manner. Here we choose Han et al. [17] as our test-bed. “random” indicates randomly freezing 50% layers, “α∗” means freezing 50% layers with top (greater) α values, and “α” means freezing 50% layers with bottom (smaller) α values.

|Traing method<br><br>|COCO Cityscapes ADE20K-150 ADE20K-847 PAS-20 PC-59 PC-459<br><br>|Training FLOPs|
|---|---|---|
|Han et al. Han et al.(random) Han et al.+α∗ Han et al.+α<br><br>|46.0 33.9 16.6 2.5 71.2 39.0 7.1<br><br>44.5 33.5 16.4 2.5 70.1 38.5 7.2<br>45.3 33.6 16.7 2.7 73.2 39.2 7.3<br><br><br>47.2 34.0 17.3 2.9 74.0 39.9 7.7<br>|181.4 P<br><br>164.5 P 9.3%↓ 172.3 P 5.0%↓ 159.6 P 12.0%↓<br><br>|

- Table 3. Balance: OpenSeg vs. Efficiency. We compare OVS performance on popular image segmentation datasets. By adopting our methods, we can significantly boost the efficiency of popular OVS frameworks while preserving their OVS performance.

|Model|Backbone Training Set|COCO PAS-20 PC-59 ADE20K-150<br><br>|Params. FLOPs|
|---|---|---|---|
|ZS3Net [1] SPNet [43] LSeg [25] ZegFormer [10] Cat-Seg [7] OVSeg [28] LSeg+ [16] OpenSeg [16] Simbaseline [44] DeeplabV3 [2] Han et al. [17] FC-CLIP [47]<br><br>|R-101 PASCAL-15 R-101 PASCAL-15 R-101 PASCAL-15 R-50 COCO Stuff R-101 COCO Stuff R-101 COCO Stuff R-101 COCO Panoptic R-101 COCO Panoptic R-50 COCO Panoptic R-50 COCO Panoptic R-50 COCO Panoptic R-50 COCO Panoptic|- 38.3 19.4 -<br><br>- 18.3 - 24.3<br><br>- 47.4 - -<br><br>- 80.7 42.8 16.4<br><br>- 93.7 57.5 27.2<br><br>- 89.2 53.3 24.0<br><br>- 59.0 36.0 13.0<br><br><br>36.9 - 36.9 15.3 39.5 - 40.1 14.4 26.3 44.1 23.9 8.8<br><br>46.0 71.2 39.0 16.6 58.7 89.5 50.5 23.3<br><br>|- -<br>- -<br>- -<br><br><br>63.9M 1128.7G<br><br>60.8M 3847.8G<br>61.1M 1116.2G<br><br><br>- -<br>- -<br><br><br>64.6M 1060.6G 40.3M 241.3G 44.1M 268.2G 39.0M 200.1G<br>|
|Ours (Han et al.) Ours (DeeplabV3) Ours (FC-CLIP)|R-50 COCO Panoptic R-50 COCO Panoptic R-50 COCO Panoptic<br><br>|42.3 64.8 35.7 15.3 34.2 54.6 28.8 10.7 56.8 87.3 47.4 18.8|22.9M 173.3G 19.1M 146.8G 17.8M 105.6G<br><br>|

and a 2% boost for ADE20K-150. Compared with the original FC-CLIP model, we are able to compress it to 17.95M Params and 105.6G FLOPs, surpassing even the compactness achieved by Han et al. Although FC-CLIP emphasizes the significance of freezing the CLIP backbone, it is noteworthy that the subnetwork we utilized exerted a minimal impact on the model’s performance in the task of OVS.

Comparisons with random subnetwork transfer. The results for the Random, characterized by equivalent sparsity to our transferable subnetwork, are omitted in Tab. 1. It is evident that ours significantly outperforms Random across all datasets, This highlights the effectiveness of our proposed method in preserving the transferability of performance from previous OVS approaches.

In a word, the proposed transferable subnetwork significantly decreases computational redundancy with minimal impacts on performance.

- 4.3. Efficient Fine-tuning with Layer Selections

the layers within the backbone. Through the identification of layers with small α values as high-quality layers, excluding them from training results in enhanced performance not only on the training dataset but also significant improvements compared to the original model across all evaluated OVS datasets. Notably, we achieve 0.7%, 2.8%, and 0.9% improvement on ADE20K-150, PAS-20 and PC-59 in terms of mIoU. In comparison to randomly freezing half of the layers without training involvement, our proposed method consistently surpasses the performance of the original model on all datasets. The freezing method allows for the conservation of training FLOPs, leading to more efficient resource utilization, as detailed in Tab. 4. It becomes apparent that through fine-tuning the identified poorly performing layers, the network can attain improved training results while utilizing fewer parameters.

#### 4.4. Stronger Balance: OpenSeg vs. Efficiency

To assess the open-vocabulary performance of our proposed method, we compare its effectiveness against state-of-theart techniques on various well-known image segmentation datasets [8, 12, 29, 36, 51]. The results presented in Tab. 3 illustrate the outcomes of employing our two previously proposed methods for enhancing model and training efficiency.

Based on the computation method outlined in Sec. 3.3, we can compute the α value for each layer of the backbone. The model training is initiated for 104 iterations before performing the α derivation and freezing specific layers.

In Tab. 2, we adopt diverse approaches to freeze half of

- Table 4. Reduction of training FLOPs with our methods (ResNet50). “pruning” means our transferable subnetworks in Sec 3.2. “α” means selective layer-wise fine-tuning in Sec 3.3.

|Model|Training FLOPs<br><br>|
|---|---|
|Han et al. Han et al.+α Han et al.+α+pruning<br><br>|181.4 P<br><br>159.6 P 12.0%↓ 122.2 P 32.6%↓|

These findings showcase that our model not only attains impressive speed but also demonstrates robust OVS capabilities. To elaborate, our method achieves a notable 56.8% mIoU on the training set COCO, encompassing 133 categories. It further attains an 18.8% mIoU on the ADE20K dataset with 150 categories and secures 47.4% and 9.9% mIoU on Pascal Context with 49 categories and 459 categories, respectively.

In comparison to the state-of-the-art method Cat-Seg across multiple datasets, our approach reveals a performance gap of only 6.4% mIoU on the PAS-20 dataset and 8.4% mIoU on the ADE20k-150 dataset. Notably, we achieve a remarkable reduction of over 240% in model parameters, coupled with a nearly 40-fold improvement in computational speed. Han et al. pioneer the elimination of the two-stage process, demonstrating commendable performance and efficiency. In contrast, our approach showcases advantages of 2.2% mIoU on ADE20K-150 and 8.4% mIoU on PC-59. Moreover, we achieve more than a twofold improvement in both parameter efficiency and operational speed.

#### 4.5. Training cost

The layer-wise fine-tuning method we proposed, has proven to be effective across OVS works. Moreover, the combination of this layer-wise fine-tuning method with our subnetwork further significantly amplifies training efficiency. In Tab. 4, our focus is exclusively on the training FLOPs of the entire model, given that our subnetwork and layer-wise fine-tuning methods are specifically applied to the backbone component. The incorporation of α results in a substantial reduction of 12.0% in the training parameters for the model. When coupled with subnetwork, the training parameters are further diminished to only 32.6% of their original value. A comprehensive breakdown of the training FLOPs calculation will be provided in the appendix section.

#### 4.6. Qualitative Results

Fig. 5 displays several visualization results obtained from our method. There exists a notable disparity in the quality of masks obtained by various baselines. In intricate scenes, the conventional convolution segmentation head exhibits subpar performance. Moreover, the unique structural design of FC-CLIP facilitates a more precise assignment of unseen category labels to the generated masks, enhancing its capacity to predict both foreground and background categories.

[Figure 5]

Figure 5. Visualizations of examples on the PC-59 validation set by our models (trained on COCO panoptic training set, zero-shot evaluated on the PC-59 validation set).

### 5. Conclusion

In this paper, we introduce a transferable subnetwork approach that markedly reduces the size and computational demands of popular OVS models. Additionally, we propose a principled layer-selective fine-tuning method, enabling efficient fine-tuning of OVS models by only updating undertrained layers. Through extensive experiments on various OVS datasets, we illustrate that our model strikes a commendable balance between performance and efficiency. We hope our method can serve as a solid baseline and help advance the future research of OVS. Some current limitations and potential future works: 1) adaptation of our method on larger convolutional backbones (e.g. ConvNeXt-Large) and ViT-based backbones; 2) more fine-grained selection of weight elements or channels for fine-tuning, going beyond the layer level; 3) extension to other open-vocabulary tasks, such as open-set object detection.

### References

- [1] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick P´erez. Zero-shot semantic segmentation. Advances in Neural Information Processing Systems, 32, 2019. 3, 7
- [2] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017. 1, 4, 6, 7
- [3] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, and Michael Carbin. The lottery ticket hypothesis for pre-trained bert networks. Advances in neural information processing systems, 33:15834–15846,

2020. 2, 4

- [4] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Michael Carbin, and Zhangyang Wang. The lottery tickets hypothesis for supervised and self-supervised pre-training in computer vision models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16306–16316, 2021. 2, 4
- [5] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34:17864–17875, 2021. 3
- [6] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 3, 4
- [7] Seokju Cho, Heeseong Shin, Sunghwan Hong, Seungjun An, Seungjun Lee, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation, 2023. 3, 6, 7
- [8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 6, 7
- [9] Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13(1):795–828,

2012. 4

- [10] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11583–11592, 2022. 1, 3, 7
- [11] Zheng Ding, Jieke Wang, and Zhuowen Tu. Open-vocabulary universal image segmentation with maskclip. arXiv preprint arXiv:2208.08984, 2022. 3
- [12] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88:303–338, 2010. 6, 7
- [13] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018. 2, 4

- [14] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pages 3259–3269. PMLR, 2020. 4
- [15] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Open-vocabulary image segmentation. arXiv preprint arXiv:2112.12143, 2021. 1
- [16] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European Conference on Computer Vision, pages 540–557. Springer, 2022. 7
- [17] Kunyang Han, Yong Liu, Jun Hao Liew, Henghui Ding, Jiajun Liu, Yitong Wang, Yansong Tang, Yujiu Yang, Jiashi Feng, Yao Zhao, et al. Global knowledge calibration for fast openvocabulary segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 797– 807, 2023. 1, 3, 5, 6, 7
- [18] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015. 4
- [19] Shuting He, Henghui Ding, and Wei Jiang. Semanticpromoted debiasing and background disambiguation for zeroshot instance segmentation. In CVPR, 2023. 1
- [20] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019. 2, 3
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 3
- [22] Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, and Wenyu Liu. Ccnet: Criss-cross attention for semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 603–612,

2019. 1

- [23] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021. 2
- [24] Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, and Humphrey Shi. Learning mask-aware clip representations for zero-shot segmentation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [25] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In International Conference on Learning Representations, 2022. 1, 7
- [26] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI conference on artificial intelligence, pages 11336– 11344, 2020. 2

- [27] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 121–137. Springer, 2020. 2
- [28] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7061–7070, 2023. 1, 3, 7
- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6, 7
- [30] Charles H Martin and Michael W Mahoney. Traditional and heavy tailed self regularization in neural network models. In International Conference on Machine Learning, pages 4284– 4293, 2019. 5
- [31] Charles H Martin and Michael W Mahoney. Heavy-tailed universality predicts trends in test accuracies for very large pre-trained deep neural networks. In SIAM International Conference on Data Mining, pages 505–513. SIAM, 2020. 5
- [32] Charles H Martin and Michael W Mahoney. Post-mortem on a deep learning contest: a Simpson’s paradox and the complementary roles of scale metrics versus shape metrics. Technical Report Preprint: arXiv:2106.00734, 2021.
- [33] Charles H Martin and Michael W Mahoney. Implicit selfregularization in deep neural networks: Evidence from random matrix theory and implications for learning. Journal of Machine Learning Research, 22(165):1–73, 2021. 5, 6
- [34] Charles H Martin and Michael W Mahoney. Implicit selfregularization in deep neural networks: Evidence from random matrix theory and implications for learning. Journal of Machine Learning Research, 22(165):1–73, 2021. 4
- [35] Charles H Martin, Tongsu Serena Peng, and Michael W Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications, 12(1):1–13, 2021. 5
- [36] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014. 1, 6, 7
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2
- [38] Alex Renda, Jonathan Frankle, and Michael Carbin. Comparing rewinding and fine-tuning in neural network pruning. arXiv preprint arXiv:2003.02389, 2020. 4

- [39] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016. 2, 3
- [40] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Lst: Ladder side-tuning for parameter and memory efficient transfer learning. Advances in Neural Information Processing Systems, 35: 12991–13005, 2022. 3
- [41] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5227–5237, 2022. 3
- [42] Yunchao Wei, Huaxin Xiao, Honghui Shi, Zequn Jie, Jiashi Feng, and Thomas S Huang. Revisiting dilated convolution: A simple approach for weakly-and semi-supervised semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7268–7277,

2018. 1

- [43] Yongqin Xian, Subhabrata Choudhury, Yang He, Bernt Schiele, and Zeynep Akata. Semantic projection network for zero-and few-label semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8256–8265, 2019. 1, 3, 7
- [44] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for openvocabulary semantic segmentation with pre-trained visionlanguage model. In European Conference on Computer Vision, pages 736–753. Springer, 2022. 1, 3, 7
- [45] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2945– 2954, 2023. 3, 6
- [46] Yaoqing Yang, Ryan Theisen, Liam Hodgkinson, Joseph E Gonzalez, Kannan Ramchandran, Charles H Martin, and Michael W Mahoney. Evaluating natural language processing models with generalization metrics that do not need access to any training or testing data. arXiv preprint arXiv:2202.02842,

2022. 5

- [47] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 4, 5, 6, 7
- [48] Hui Zhang and Henghui Ding. Prototypical matching and open set rejection for zero-shot semantic segmentation. In ICCV, pages 6974–6983, 2021. 1
- [49] Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tipadapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930, 2021. 3
- [50] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2881–2890, 2017. 1
- [51] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k

dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017. 6, 7

[52] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022. 3

## Transferable and Principled Efficiency for Open-Vocabulary Segmentation Supplementary Material

### 6. Experimental Details

#### 6.1. Fine-tuning Settings After Transfer

This section provides a detailed introduction to the experimental settings for all three transfer experiments. The experimental settings used for Han et al. and Deeplabv3 are identical. We trained the models on the COCO panoptic dataset for 50,000 iterations using four 4090 Ti GPUs. The training was performed with a batch size of 28, utilizing the SGD optimizer with an initial learning rate of 0.00015. The learning rate was reduced by a factor of 0.1 at 40,000 iterations and 45,000 iterations. For FC-CLIP, we conducted training for 368,750 iterations using eight 4090 Ti GPUs. The training was performed with a batch size of 16, utilizing the AdamW optimizer with an initial learning rate of 0.0001. The learning rate was reduced by a factor of 0.1 at 327,778 iterations and 355,092 iterations.

#### 6.2. Training FLOPs Calculation

To calculate the training FLOPs of our model backbone, we consider both the forward and backward processes. When calculating the backward FLOPs, we need to compute gradients for both model parameters and the hidden features. Hence, the FLOPs involved in the backward process will be doubled compared with the FLOPs of the forward process1. Suppose the inference FLOPs of the model is C. We provide a comprehensive calculation approach below where our subnetwork has a sparsity of 10% and 50% of the layers are frozen during fine-tuning:

- 1. Standard Fine-tuning: forward FLOPs 1×C, backward FLOPs 2×C.
- 2. Model with layer-wise fine-tuning: Frozen layers: forward FLOPs 1×C, backward FLOPs 1×C. Active layers: forward FLOPs 1×C, backward FLOPs 2×C.

- 3. Model with subnetwork: forward FLOPs 0.1×C, backward FLOPs 1.1×C (1 for hidden features, 0.1 for sparse weights).
- 4. Model with layer-wise fine-tuning and subnetwork: Frozen layers: forward FLOPs 0.1×C, backward FLOPs 1×C. Active layers: forward FLOPs 0.1×C, backward FLOPs 1.1×C (1 for hidden features, 0.1 for sparse weights).

To calculate the FLOPs of specific layers, we utilize the “get model complexity info” function provided by the ptFLOPs library2.

- 1https://epochai.org/blog/backward-forward-FLOP-ratio
- 2https://pypi.org/project/ptflops/

- 6.3. Visualizations of Heavy-tail Behaviors in Efficient Fine-tuning

Our Fig. 4 provides a comprehensive overview of the layerwise fine-tuning implementation process. Fig. 6 showcases the α value for each layer in our backbone before fine-tuning (after pruning, before adopting sparse masks). Layers of α below the red horizontal line (median value of α across layers) will be frozen during fine-tuning. Additionally, Fig. 7 illustrates the changes in the median of α values throughout the entire fine-tuning process under our training method. It is evident that the α value of most layers remains relatively stable, leading us to not dynamically calculate α or adjust our fine-tuning layers in the main experiment.

- 7. Futher Experiments

- 7.1. Ablation Study on Different Knowledge Distillation Losses

In the method presented in Sec. 3.1, a loss function is introduced to align the text with the vision feature space. This loss, referred to as text-guided knowledge distillation (TGKD), facilitates the distillation of knowledge from textual information into visual embeddings. The process of text-guided knowledge distillation can be formulated as follows:

LTGKD =

1 N

N

i=1

N

j=1

(∥Vi−R(I, Mj)∥−∥T (Yi)−T (Yj)∥) , (3)

where T denotes the CLIP text encoder, R denotes the CLIP image encoder, Vi denotes the visual embeddings and Yi is the category name of i-th ground truth region. I represents the input image, while M represents the mask corresponding to the respective category. Correspondingly, a vision-guided knowledge distillation approach has also been proposed, which can be formulated as follows:

LV GKD =

1 N

N

i=1

N

j=1

(∥Vi −R(I, Mj)∥−∥Vi −Vj∥) , (4)

Tab. 5 provides a comprehensive comparison of the results obtained from pruning using two distinct distillation losses. The analysis demonstrates that the utilization of text-guided distillation loss significantly enhances the OVS performance of the model.

- 7.2. Ablation Study on Pruning Strategies

We further study two pruning strategies. The first strategy is to naively apply IMP to prune the model using all training

[Figure 6]

- Figure 6. Layer-wise α values in model backbone before fine-tuning used in Sec. 4.3. Specifically, the blue curve represents the calculated α value for each layer of Resnet, while the red line represents their median. During the fine-tuning process, we freeze the layers that have values smaller than the median.

[Figure 7]

- Figure 7. The median value of layer-wise α during the fine-tuning process. We can see α remains relatively stable without significant changes.

- Table 6. Ablation study of different configurations for pruning with both distillation and segmentation loss. Pruning ratio (p) and the number of training iterations (t) used in Algorithm 1 were studied.

|p t<br><br>|COCO Cityscapes ADE20K-150 PAS-20 PC-59|
|---|---|
|0.1 5000<br><br>0.1 2500<br><br>0.2 5000<br><br><br>|40.6 29.1 15.0 60.1 33.0 36.9 28.9 13.8 54.8 32.0 32.1 27.8 11.6 50.4 27.8|

- Table 7. Ablation study of different configurations for pruning with only distillation loss followed by segmentation fine-tuning (our strategy in Sec. 3). Pruning ratio (p) and the number of training iterations (t) used in Algorithm 1 were studied.

|p t|COCO Cityscapes ADE20K-150 PAS-20 PC-59<br><br>|
|---|---|
|0.1 5000 0.1 1000 0.1 100 0.3 1000 0.5 1000|41.8 31.8 15.4 63.2 35.1<br><br>41.9 32.7 15.0 63.9 35.1<br><br><br>42.4 31.0 14.3 63.2 34.8 40.4 28.9 13.7 61.4 33.6 39.8 30.3 13.6 60.7 33.3<br><br>|

- Table 5. Comparison of different distillation loss: LTGKD vs. LV GKD.

without semantic supervision exhibit the ability to be further fine-tuned in the original model and can also be effectively transferred to other tasks, yielding superior experimental results (as shown in Tab. 1).

|Distillation Loss<br><br>|COCO PC-59 ADE20k-150|
|---|---|
|Text-guided (LTGKD) Vision-guided (LV GKD)|42.5 35.1 15.8 40.0 32.6 14.2<br><br>|

losses (distillation loss plus mask loss and classification loss). The second strategy, which is proposed in our Sec. 3.2, first obtains subnetworks based on the semantic-agnostic distillation loss, and subsequently fine-tunes the model for a specific number of iterations.

Tab. 8 provides experimental details. The results demonstrate that the adopted pruning with fine-tuning method not only improves OVS performance but also requires fewer training iterations. Additionally, the subnetworks obtained

Ablation Study on Pruning Configurations. We also study the best configuration of t and p of two pruning strategies discussed above. Ablation studies on the first pruning strategy (distillation + segmentation loss) are shown in Tab. 6. It is important to highlight that reducing the value of t or increasing the value of p has a substantial impact on the performance, leading to a noticeable drop.

Additionally, for our pruning strategy (pruning with distillation only, followed by segmentation fine-tuning), different

[Figure 8]

###### Figure 8. More visualizations of examples on the PC-59. “GT”: ground truth.

- Table 8. We compare the segmentation performance (mIoU) of two strategies for finding sparse models: 1) pruning with both distillation and segmentation loss; 2) our pruning (with only distillation loss) followed by segmentation fine-tuning (Sec. 3). The “Training Iters” parameter represents the total number of training iterations required by each of the two methods.

Pruning Method COCO PC-59 ADE20k-150 Training Iters.

|Pruning (distillation + segmentation loss)| |40.6 33.0 15.0| |105000|
|---|---|---|---|---|
|Pruning (distillation only) + Segmentation Fine-tuning| |42.5 35.1 15.8| |95000|

- Table 9. CKA similarity between CLIP image encoder and DeeplabV3 backbones.

DeeplabV3 Sparse DeeplabV3 (Ours) CKA (vs. CLIP) 0.361 0.512

Table 10. Ablation study of freezing ratio

|Ratio|COCO Cityscapes ADE-150 ADE-847 PAS-20 PC-59 PC-459<br><br>|
|---|---|
|0.25 0.75 0.5<br><br>|47.5 34.5 17.3 2.9 72.5 39.6 7.6<br><br>46.5 34.8 17.4 2.8 72.6 38.9 7.5<br>47.2 34.0 17.3 2.9 74.0 39.9 7.7<br>|

[Figure 9]

Figure 9. α changes before and after finetuning.

configurations are studied in Tab. 7. Initially, a pruning rate (p) of 0.1 was used, while different values of t were employed. The performance on the COCO dataset was found to be very similar across different t values. However, when t was reduced to a very small value, a significant drop in performance on the OVS datasets was observed. This observation further supports the claim that the subnetwork discovery method proposed in Sec. 3.2 is advantageous for the OVS task. If t is kept constant and the pruning rate is increased, it will result in IMP finding subnetwork with lower quality. It is worth noting that the final sparsity of the subnetworks discovered by different pruning rates will vary, making direct comparisons between them relatively unfair. However, this disparity is inevitable given the different pruning rates employed.

#### 7.3. Ablation Study Implementation details

In this section, we conduct several ablations to justify the design choices of our proposed methods. Compare with other compression methods In Tab. 1, we

mainly compare our method with random pruning, since it can also be transferred to different models without incurring any additional costs. In Tab. 8, we compare our method with the established pruning technique IMP. Our approach outperforms IMP in terms of both performance and training time. Moreover, our subnetwork exhibits transferability, allowing for faster training across different segment architectures.

OVS capability analysis Our core method leverages the benefits of our transferable subnetwork to improve OVS performance and enable enhanced open-set knowledge distillation from CLIP. As shown in Tab. 9, by adopting our subnetwork, DeeplabV3 can achieve more similar features with CLIP than the baseline, measured by Centered Kernel Alignment (CKA) similarity averaged over layers [9].

Analysis of freezing layers As shown in Fig. 9, compared with early layers, α of deeper layers undergo more changes during fine-tuning, gradually becoming more “well-trained” as their αs decrease [34].

Based on the implicit self-regularization in deep networks [34], weight matrices with α < 2 are generally considered “over-trained” and more prone to overfitting. Therefore, in our supplementary Figure 6, we observe certain layers with α < 2. Freezing these layers during fine-tuning provides benefits, as it helps prevent overfitting. We also provide different ratios of freezing layers in Tab. 10. Users can adjust this ratio flexibly according to their own needs.

#### 7.4. More Qualitative Results

Building upon the findings in Sec. 4.6, we present additional qualitative results in this section, along with a comparison to the ground truth. Fig. 8 illustrates a specific case where our model demonstrates robust OVS performance by accurately labeling some parts that are not labeled in the ground truth. This exemplifies the effectiveness of our model in accurately predicting labels even in challenging scenarios where ground truth annotations may be incomplete.

