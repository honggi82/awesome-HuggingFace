## ARKit LabelMaker: A New Scale for Indoor 3D Scene Understanding

Guangda Ji1 Silvan Weder1 Francis Engelmann1,2 Marc Pollefeys1 Hermann Blum1,3

1ETH Z¨urich 2Stanford University 3University of Bonn & Lamarr Institute

# arXiv:2410.13924v2[cs.CV]20Mar2025

### Abstract

Neural network performance scales with both model size and data volume, as shown in both language and image processing. This requires scaling-friendly architectures and large datasets. While transformers have been adapted for 3D vision, a ‘GPT-moment’ remains elusive due to limited training data. We introduce ARKit LabelMaker, a large-scale real-world 3D dataset with dense semantic annotation that is more than three times larger than prior largest dataset. Specifically, we extend ARKitScenes [4] with automatically generated dense 3D labels using an extended LabelMaker pipeline [37], tailored for large-scale pre-training. Training on our dataset improves accuracy across architectures, achieving state-of-the-art 3D semantic segmentation scores on ScanNet and ScanNet200, with notable gains on tail classes. Our code is available at labelmaker.org and our dataset at huggingface.

### 1. Introduction

Recent advancements in deep learning on language [5, 26, 27] and 2D vision [1, 28, 29] have made tremendous progress, primarily driven by the abundance of training data available on the web for these modalities. Scaling this type of large-scale training to billions of data points has revealed surprising properties [38] and resulted in unprecedented performance gains, enabling entirely new applications. However, this approach is not directly applicable to 3D scene understanding, where real-world 3D data lacks web-scale abundance and demands labor-intensive ground truth annotations.

While recent efforts aim to reduce this dependency through self-supervision [13, 49], distillation [23, 45], denoising [35], or open-set scene understanding [10, 18, 33, 40, 44], state-of-the-art 3D segmentation methods [21, 31, 40, 46] still rely on some level of direct supervision. Consequently, annotated data remains essential for learning these tasks, and constructing datasets of comparable scale to those in language and image generation remains a significant challenge. In this paper, we contribute the largest 3D real-world indoor semantic dataset and investigate key

###### Datasets Size

train

ScanNet++ [43] (100 classes)

val test

S3DIS [2] (13 classes)

ScanNet/ScanNet200 [9, 30] (20/200 classes)

ARKit LabelMaker (186 classes)

num scenes 0 1000 2000 3000 40000 5000 6000

Figure 1. Our LabelMaker annotation data creates the world’s largest real-world 3D scene annotation dataset.

questions in 3D scene understanding: Is real-world data preferable to synthetic data? How can labeling efforts be minimized? Do current models benefit from increased realworld data?

To address these questions, we leverage ARKitScenes [4], a large-scale 3D indoor dataset consisting of 3D reconstructions and RGB-D frames captured with consumer tablets. Although these scenes are annotated with 3D object bounding boxes, they lack the per-point annotations necessary for training competitive 3D segmentation models. To overcome this limitation, we augment the dataset with per-point semantic labels created through an automated pipeline. This approach enables us to produce the significantly larger dataset compared to prior 3D semantic segmentation dataset (see Fig. 1). Our dataset is suitable for (pre-)training any 3D semantic segmentation model. To validate the effectiveness of these extensive yet imperfect annotations, we use them to re-train various models and conduct comprehensive evaluations on widely-used 3D semantic segmentation benchmarks [9, 30].

More specifically, we build on top of the recent LabelMaker [37] pipeline, which we extend into LabelMakerV2 with more and updated base models, a more general input data structure, as well as deployment scripts for large clusters through docker or SLURM. Using this pipeline, we process the entire ARKitScenes dataset, which takes 48’000 GPU hours on Nvidia 3090 GPUs. We further scale our pipeline beyond ARKitScenes to arbitrary scenes by integrating the iOS app Scanner 3D into LabelMakerV2, enabling automatic annotation of scenes recorded with consumer iPhones. In experiments, we use our automatically

generated ARKitScenes labels to pre-train the currently most-used 3D segmentation methods, MinkowskiNet [7] and PTv3 [40]. We find that without labeling any data manually, extending the scale of real-world training data improves the performance of both models on multiple benchmarks, or achieves the same performance as with even more synthetic training data.

Download & Preprocessing

OVSeg

| | |
|---|---|
|Base Models| |
| | |

Grounded-SAM

CMX Model

H H A

C M X

OmniDepth

Consensus

In summary, we answer the following research question: “Does large-scale pre-training with automatic labels show similar trends in 3D as it does for language and image tasks?” through the following key contributions:

InternImage

Point-Lifting

Mask3D

: Video Visualization

- • Generating the largest existing real-world 3D dataset with dense semantic annotations on 186 classes.
- • Improving over state-of-the-art PointTransformer on ScanNet200 by 2.1%, on tail classes by 5.5% mIoU.
- • Trade-off analysis between training in unsupervised settings, on synthetic data, and on auto-labeled real-world data providing guidance for future data scaling efforts.

Figure 2. Dependency graph of the LabelMakerV2 pipeline. Our LabelMakerV2 pipeline has a clear dependency structure that has to be handled in the distributed processing of the data. This has to be especially respected when recovering from job failure. There, our recovery strategy checks for unfinished jobs in the dependency graph before submitting any new jobs to avoid unnecessarily wasting compute resources. The boxes with thick green frame donotes visualizable tasks. These are used during inspection and job quality assurance.

### 2. Related Works

them, such as the popular MinkowskiNet [7]. Mix3D [21] enhances MinkowskiNet through effective 3D data augmentation techniques. PonderV2 [49] explores self-supervised learning from RGB-D data to improve the performance of the MinkowskiNet architecture. Point-based methods includes [3, 11, 16, 17, 24, 25, 34, 42]. However, there is a recent shift from models based on point-wise convolutions to point-based transformer models [15, 22, 31, 47]. Notable examples include PointTransformer [47] and its successors PTv2 [39], and PTv3 [40], which are developed towards better efficiency and scalability. PointPromptTraining [41](PPT) introduces a novel training paradigm enabling the simultaneous training of multiple datasets with diverse label spaces. Combining PTv3 with PPT achieves state-of-the-art performance on the ScanNet/ScanNet200 semantic segmentation benchmark.

Datasets for 3D semantic segmentation. 3D semantic segmentation classifies each point in a 3D point cloud into a set of predefined semantic categories. Prominent datasets for training and evaluation include ScanNet [9]/ScanNet200 [30] consisting of 1.5k scenes, and the Stanford 3D Indoor Scene Dataset [2] (S3DIS), which comprises 6 large-scale indoor areas with 271 rooms. Both datasets include RGB-D frames captured in the real world. In addition to real-world datasets, Structured3D [48] is a photo-realistic synthetic dataset with 3.5K house designs, and Replica [32] provides 18 high-quality reconstructed scenes. ARKitScenes [4] is the most extensive collection of indoor scenes to date, with 5047 scans of 1661 unique scenes. RGB-D data is recorded with an Apple iPad equipped with a built-in LiDAR scanner. High-quality surface reconstruction and the bounding box for object detection are also provided. However, per-point annotations for semantic or instance segmentation are not included, so it cannot be directly used for training 3D segmentation models.

In this paper, we address a key limitation of existing datasets for 3D semantic segmentation: their small scale. We hypothesize that this constraint hampers the performance of commonly used models.

LabelMaker. Weder et al. [37] is an automatic 3D semantic segmentation annotation pipeline that consolidates outputs from state-of-the-art 2D and 3D segmentation models with an additional feature for translating frame-wise 2D labels into consistent 3D point cloud labels. In this work, we employ an enhanced version of LabelMaker to create 3D semantic segmentation annotations for ARKitScenes.

### 3. Method

#### 3.1. LabelMaker Revisted

As we build on LabelMaker [37], we briefly review its key steps. LabelMaker is an automatic pipeline for 2D and 3D semantic annotation, producing labels comparable in quality to human annotations [9]. It automatically generates semantic labels by leveraging an ensemble of base models to predict pixel-level semantics for each frame in an RGB-D trajectory. Since the base models predict segmentations in different label spaces (based on their training data), their predicted semantic labels are then mapped to a unified la-

3D semantic segmentation models. Deep learning models for processing 3D input data can be classified into three main categories: voxel-based, point-based, and transformer-based methods. Voxel-based methods transform points into fixed-sized voxel grids before processing

bel space. Only through this mapping, the different base models can be used in a subsequent ensemble. Thus, [37] defined a mapping from every label space into a carefully curated label space based on wordnet synkeys [20]. After mapping all base model predictions to the unified label space, they are aggregated into a single consensus per frame of the RGB-D trajectory. This is the first stage of per-frame denoising. As the RGB-D trajectory provides multi-view information of the scene, the individual frames can be further denoised by lifting 2D predictions onto 3D points and performing per-point voting. The final labels can either be directly used as 3D labels for 3D semantic segmentation or projected into 2D and be used for training or evaluating 2D semantic segmentation models. In this paper, we improve this pipeline to robustly scale to large-scale datasets and show its benefit for pretraining 3D semantic segmentation models. In the following, we describe the improvements in more detail.

#### 3.2. Improving LabelMaker for Scaling

While LabelMaker [37] introduced an automatic labeling tool that produces annotations comparable to human annotators, we enhance the pipeline with two modifications to further improve its performance, ensuring robust highquality annotation for large-scale datasets. The complete pipeline is shown in Figure 2.

Integrating Grounded-SAM. LabelMaker [37] employs several state-of-the-art base models in its ensemble, but does not utilize Segment Anything (SAM) [14], a 2D segmentation model trained on large-scale datasets that generalizes robustly across diverse scenarios. To scale LabelMaker to any environment, we aim to integrate this prior into the pipeline. However, efficiently leveraging this model for semantic segmentation is not straightforward. To address this, Grounded SAM combines Grounding DINO [19] with SAM [14]. Grounding DINO predicts instance bounding boxes based on semantic labels or natural language, while SAM generates high-quality segmentation masks for these boxes. We integrate this model by adapting it to LabelMaker’s unified label space, allowing it to contribute as an additional vote in the ensemble.

Aligning to Gravity. For optimal performance, many semantic segmentation models require the gravity direction to be aligned with the coordinate system used during training. However, large-scale datasets are not inherently gravityaligned. For instance, in ARKitScenes, occasional phone rotations introduce inconsistencies in the orientation of 2D images. Passing these misoriented images to LabelMaker’s base models degrades performance and leads to misclassifications, such as confusing the floor with the ceiling (see Fig. 3). Therefore, we project the sky direction, corresponding to the z-axis of ARKit’s pose coordinate system (derived from the IMU), onto each 2D frame. We then compute the

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Without gravity alignment With gravity alignment Figure 3. Qualitative Evaluation of Gravity Alignment. LabelMaker annotation with and without gravity alignment. Without gravity alignment, floors may be misclassified as walls, walls as ceilings, as well as other orientation-dependent objects.

angle α between the sky direction and the upward direction. Given this angle, we rotate the image by k · π2, where k = arg mins(|sπ2 − α|) to align the sky direction roughly upward, and rotate the predicted segmentation back to its original orientation after inference to align it with its coordinate system.

Omission of NeuS. In LabelMaker [37], an implicit surface model, NeuS [36] with a semantic head, is trained per scene as an optional 3D lifting and 2D denoising step. We omit this step due to its high computational cost. Moreover, NeuS optimizes its own scene geometry, which is poorly constrained in ‘inside-out’ scans of ARKitScenes, making point cloud label generation more complex than a simple coordinate lookup. Instead, we retain per-point voting from [37], as it proved the most stable in our initial exploration.

### 4. Results

#### 4.1. Baselines

We evaluate the effectiveness of our ARKitScenes LabelMaker dataset using two well-established and distinct network architectures: MinkowskiNet [7] and PointTransformer [8, 39–41]. MinkowskiNet remains the foundation of many top-performing models in 3D semantic segmentation benchmarks, with several modifications [21, 49] pro-

Dataset #train #val #test real #classes Structured3D 6519 - 1697 25 S3DIS 406 - - 13 ScanNet/ScanNet200 1201 312 100 20 / 200 ScanNet++ 230 50 50 100 ARKit LabelMaker 4471 274 274 186

- Table 1. Dataset Size. We provide by far the largest real-world labeled training dataset compared to existing real-world datasets. We provide automatically generated per-point semantic annotations for 4471 training scenes and 548 validation scenes.

posed to enhance its performance. PointTransformer [40], a more recent architecture, achieves state-of-the-art results on the ScanNet and ScanNet200 benchmarks. Given that transformers generally benefit from large-scale training data, we also train on this architecture. From these two architectures, we derive three relevant baselines:

Vanilla MinkowskiNet. This is the standard MinkowskiNet model based on [7], which most 3D semantic segmentation methods compare to. In this paper, we use the commonly used ‘Res16UNet34C‘variant of MinkowskiNet to guarantee fair comparison to all other baselines.

Mix3D [21] is a data augmentation method for large-scale 3D scene segmentation that creates new training samples by merging two augmented scenes, effectively placing object instances into novel, out-of-context environments. This approach encourages models to infer semantics from local structures rather than relying on overall scene context. MinkowskiNet shows significant performance improvements when trained with Mix3D.

PonderV2 [49]. Addressing the scarcity of 3D annotations involves two strategies, unsupervised feature learning and automated pseudo-labeling. PonderV2 [49] represents the former, using neural rendering objectives to learn 3D features without semantic annotations. Our work explores the latter, generating large-scale pseudo-labels through automatic annotation. While PonderV2 currently achieves state-of-the-art results in unsupervised settings, we compare both paradigms to quantify the relative merits of label-free feature learning versus pseudo-label-driven supervision for scaling 3D segmentation models.

PointTransformerV3 (PTv3) [40] is a recently proposed method to accelerate transformer architectures and enable large-scale training by jointly training multiple datasets with diverse label spaces. This contrasts with the LabelMaker [37] approach, which translates label spaces to a common one before training. The combination of PTv3 and PPT achieves state-of-the-art performance on the ScanNet/ScanNet200 semantic segmentation benchmarks.

#### 4.2. Datasets and Evaluation Metrics

ScanNet [9] comprises 1513 densely annotated scans across 707 distinct indoor scenes, totaling 2.5 million RGBD frames. It stands as one of the most widely used and influential benchmark datasets for indoor 3D scene understanding. It is annotated by humans using the NYU40 label space and evaluated on a subset of 20 classes from NYU40.

ScanNet200 [30]. While only 20 classes are used in the ScanNet benchmark, the original dataset is annotated with many more classes. ScanNet200 [30] leverages these annotations and organizes them into a new benchmark with 200 classes that are of higher-resolution than the original ScanNet classes. Given the large-number of different categories generated by our LabelMakerV2 pipeline, we also pre-train the models for this task and evaluate them on the ScanNet200 benchmark.

ScanNet++ [43] is a dataset of 460 high-resolution 3D indoor scenes with dense semantic and instance annotations, captured using a high-precision laser scanner and registered images from a DSLR camera and RGB-D streams. It focuses on long-tail and multi-labeled annotations. Models are typically evaluated on 100 classes.

Structured3D [48] is a large-scale indoor synthetic RGBD dataset featuring 6519 training scenes and 1697 test scenes. It is annotated with a label space of 25 classes. Structured3D is only used in PTv3+PPT joint training and we adopt pre-processed version of these two datasets from [40].

Matterport3D [6] is a large-scale RGB-D dataset containing 10,800 panoramic views from 194,400 RGB-D images across 90 building-scale scenes. We map the ScanNet200 label space to Matterport3D and use this dataset for zeroshot evaluation of our trained model.

ARKit LabelMaker (Ours). This is the dataset generated with our method described above. The resulting dataset contains 5019 scenes, from which we take 4471 for training and 548 for validation according to the official trainval split provided by the original ARKitScenes [4] dataset. For every scene, we created 3D point cloud associated perpoint semantic labels in the original LabelMaker wordnet label space (186 classes). In some experiments, we project the dataset from wordnet label space to ScanNet200 label space to train it together with ScanNet200 dataset. We denote this converted dataset as ARKit LabelMakerSN200. To increase efficiency and make the experimental settings comparable to previous studies, we perform down-sampling on 3D meshes to a voxel size of 2 cm. Normal information is preserved and down-sampled simultaneously. Table 1 illustrates the scale of each dataset. ARKit LabelMaker dataset is the largest annotated real-world indoor semantic dataset.

Val mIoU Test mIoU all head common tail all head common tail MinkUNet [7]

Method Training Data

vanilla ScanNet200 27.2 49.8 20.8 11.2 25.3 46.3 15.4 10.2 fine-tune (Ours) ARKit LabelMakerSN200 → ScanNet200 28.1 51.6 22.3 10.6 27.4 49.0 19.4 9.4 co-training (Ours) ARKit LabelMakerSN200 + ScanNet200 28.2 52.5 22.4 9.8 - - - -

PTv3 [40]

vanilla ScanNet200 35.2 56.5 30.1 19.3 37.8 - - fine-tune (Ours) ARKit LabelMakerSN200 → ScanNet200 36.4 56.7 31.2 21.6 - - - fine-tune (Ours) ARKit LabelMaker → ScanNet200 37.0 57.0 32.6 21.7 38.4 58.2 30.9 22.2 PPT [40] ScanNet200 + S3DIS + Structure3D → ScanNet200 36.0 - - - 39.3 59.2 33.0 21.6 PPT(Our ablation) ScanNet + ScanNet200 + Structure3D → ScanNet200 36.3 56.6 31.7 20.7 - - - PPT(Ours) ScanNet + ScanNet200 + ScanNet++ + Structure3D + ARKit LabelMaker 37.5 58.8 33.3 20.4 41.4 61.0 32.2 27.1

- Table 2. 3D Semantic Segmentation Scores on ScanNet200 [30]. To investigate our large-scale dataset also helps with long-tail categories, we evaluate it on the ScanNet200 dataset. For both MinkowskiNet [7] and PointTransformerv3 [40], we compare it to vanilla training as well as training procedure proposed in [40]. We can show that common neural networks benefit from pre-training on automatically generated large-scale annotations. We also report the head, common and tail classes mean IoU.

Method Training Data val test MinkUNet [7]

vanilla ScanNet 72.4 73.6 PonderV2 [49] ScanNet (self-supervised) → ScanNet 73.5 Mix3D [21] ScanNet 73.6 78.1 fine-tune (Ours) ARKit LabelMakerSN200 → ScanNet 74.1 -

PTv3 [40]

vanilla ScanNet 77.5 77.9 fine-tune (Ours) ARKit LabelMakerSN200 → ScanNet 78.9 fine-tune (Ours) ARKit LabelMaker → ScanNet 78.0 79.0 PPT [40] ScanNet + S3DIS + Structure3D 78.6 79.4 PPT (Ours) ScanNet+ ScanNet200 + ScanNet++ + Structure3D + ARKit LabelMaker 79.1 79.8

- Table 3. 3D Semantic Segmentation Scores. Comparing training strategies for two top-performing models (PointTransformerV3 [40] and MinkowskiNet [7]) on ScanNet20 [9]. Adding ARKit LabelMakerSN200 through pre-training and co-training improves the performance for both models. With PonderV2 [49] and Mix3D [21], we compare large-scale pretraining to two other training strategies. Large-scale pre-training is superior to both, extensive data augmentation (Mix3D) and self-supervised pre-training (PonderV2).

Metrics. We follow the standard metrics of the ScanNet 3D semantic segmentation task and compute the mean and per-class intersection-over-union (IoU), the mean per-class accuracy (mAcc) and the total per-point accuracy (tAcc).

- 4.3. Experiment Settings

epochs. If the label space is changed for fine-tuning, we replace the classification head and exclusively train it with the same learning rate setting until convergence while the rest of the model is fixed. Then, the entire network undergoes fine-tuning with a learning rate of 0.001, while other settings are kept unchanged.

For PTv3 [40], we follow [40] employing the AdamW optimizer with OneCycleLR for 800 epochs of training. Similar to the fine-tuning of MinkowskiNet, we initially freeze the backbone and then solely train the classification head until convergence. Then, we fine-tune on ScanNet or ScanNet200 with a reduced learning rate of 0.0006. Besides ARKit LabelMakerSN200, we also pre-train PTv3 with ARKit LabelMaker in wordnet label space as the mapping from wordnet to ScanNet200 may reduce class diversity.

Co-training with ScanNet200. With this experiment, we investigate whether ARKit LabelMaker can be seamlessly combined with existing datasets to increase dataset size and improve model performance. To this end, we merge ARKit LabelMakerSN200 with ScanNet200 and train a MinkowskiNet from scratch. Due to resource constraints, we conduct this experiment only with MinkowskiNet. The training setup follows the exact pre-training procedure described earlier for MinkowskiNet.

We adopt three approaches to evaluate the effectiveness of our ARKitScenes LabelMaker dataset.

Pre-training. To investigate whether automatic labels are useful to learn strong features from imperfect annotations, we pre-train both, MinkowskiNet and PointTransformerV3, on our generated ARKit LabelMakerSN200 dataset. Afterwards, we fine-tune the pretrained models on the ScanNet and ScanNet200 dataset, respectively.

Joint-training. We employ PTv3+PPT for joint training across multiple datasets and label spaces. In addition to ScanNet/ScanNet200, ScanNet++, and Structured3D, we incorporate our ARKit LabelMaker dataset. To maximize semantic class exposure, we adopt LabelMaker’s WordNet label space. Our training setup follows the exact PTv3+PPT configuration from [40], using the AdamW optimizer with a OneCycleLR scheduler and a learning rate of 0.05. We also integrate the LabelMaker WordNet label space into the normalization layer and final classification head.

For MinkowskiNet, we employ the Res16UNet34C architecture as our backbone model. During pre-training, we utilize the AdamW optimizer with a learning rate of 0.01 and OneCycleLR scheduler, training the network for 600

Pretrain Dataset fine-tune val mIoU scratch 32.76 10% ARKit LabelMaker 33.30 (+0.54) 20% ARKit LabelMaker 35.29 (+2.53) 50% ARKit LabelMaker 36.29 (+3.53) 100% ARKit LabelMaker 37.04 (+4.28) Structured3D (re-sampled to ARKit LabelMaker size) 32.21

- Table 4. PTv3 [40] pretrained on different portion of ARKit LabelMaker dataset and Structured3D resampled to be the same size

- as ARKit LabelMaker. We then fine-tune on ScanNet200 and report validation mIoU. These experiments show the scaling potential of PTv3 and the advantage of real-world over synthetic data.

Method mIoU mAcc tAcc ScanNet’s Annotation 17.7 21.3 70.6 LabelMakerV1 16.3 20.3 75.0 LabelMakerV2 w/o GSAM 17.6 21.0 77.1 LabelMakerV2 w/ GSAM 18.5 22.7 76.9

Table 5. Evaluation of LabelMaker on 5 ScanNet scenes. We use the same ScanNet scenes in the original LabelMaker’s paper with manual annotation in WordNet label space. Our LabelMakerV2 improves over ScanNet’s manual annotation.

4.4. Experiments

In Table 3, we present the results for the ScanNet dataset. For MinkowskiNet [7], we show that pre-training on our large-scale, real-world ARKit LabelMakerSN200 dataset not only significantly improves the mean intersection-overunion compared to vanilla training but also outperforms other pre-training variants. Similar trends are observed in Table 2, which reports results on the ScanNet200 dataset. Co-training MinkowskiNet on ScanNet200 further confirms that our ARKit LabelMakerSN200 dataset enhances training without introducing a domain gap relative to ScanNet200.

Comparison to Self-supervision. Table 3 shows that pretraining on our imperfect yet automatically generated labels outperforms self-supervised pre-training (PonderV2 [49]) and extensive data augmentation (Mix3D [21]). This highlights the importance of direct supervision with large-scale training data for effective 3D segmentation.

Comparison to Training on Synthetic Data. Table 2 shows that PTv3 pre-trained on ARKit LabelMaker achieves comparable or superior performance to large-scale multi-dataset joint training. The approach in [40] relies heavily on Structure3D, a synthetically generated dataset, for pretraining, motivating a deeper analysis in Table 4. We compare pretraining on different subset sizes of ARKit LabelMaker to an equivalent amount of synthetic data from Structure3D [48]. Even at smaller scales, pretraining with real-world ARKit LabelMaker data proves more effective than synthetic data, which can even degrade performance

- at limited sizes. These findings strongly indicate that autolabeling real-world recordings is significantly more effec-

LabelMakerV2 Method Variant mIoU mAcc tAcc w/o GSAM w/o g-alignment 7.1 10.0 45.1 w/o GSAM w/ g-alignment 10.3 13.4 70.6

w/ GSAM w/o g-alignment 7.9 11.0 44.9 w/ GSAM w/ g-alignment 12.9 15.7 73.6

Table 6. We annotate three ARKitScenes scenes and perform ablation study of GSAM and gravity alignment. Our results shows that both GSAM and gravity alignment are helpful to the performance. This demonstrate the contribution of our novelty in improving LabelMaker’s pipeline. We also provide the visualization of ground truth labels and predictions of these scenes in Figure 5.

tive per data point than relying on synthetic data.

Effect on Long-tail Classes. For Point Transformer, integrating our ARKit LabelMaker into the joint training of PTv3+PPT results in a noticeable boost in validation and test mIoU. This version of PTv3 achieves state-of-the-art performance on both ScanNet and ScanNet200, with the latter benefiting significantly from improved tail class mIoU (Table 2, Figure 4). When comparing fine-tuned and vanilla models on the ScanNet200 validation set, our trained model shows a performance gain of +0.5% on head classes and +2.4% on tail classes. Compared to PTv3-PPT trained without ARKit LabelMaker, our model achieves a +0.8% gain on head classes and +5.5% on tail classes on the ScanNet200 test set. This effect persists even in a zero-shot setting on the Matterport3D dataset (Table 7). Including ARKit LabelMaker in training yields similar performance gains across Top-40 to Top-160 class sets. For MinkowskiNet, we do not observe a consistent improvement in tail class performance. On ScanNet200, validation mIoU slightly declines. In zero-shot evaluation on Matterport3D, the performance gain from adding ARKit LabelMakerSN200 over MinkowskiNet trained solely on ScanNet200 diminishes from Top-40 to Top-160 class sets.

#### 4.5. Ablation Studies

Different Training Regimes. Between pre-training with ARKit LabelMaker and co-training with aligned labels, we observe no significant difference in Table 2. Co-training is primarily relevant for MinkowskiNet, as PTv3 allows for joint training across different output label spaces through PPT [40]. Increasing the dataset scale further through joint training on multiple datasets generally leads to better performance than pre-training solely on ARKit LabelMaker.

Evaluation of LabelMakerV2. In Table 5, we evaluate our updated LabelMakerV2 pipeline on five ScanNet scenes with five manual annotations in LabelMaker’s WordNet label space, sourced from the original LabelMaker. The current pipeline achieves higher accuracy than even ScanNet’s original annotations. In Table 6, we conduct an ablation study to assess the performance improvements from inte-

toilet seat cover dispenser

Vanilla PTv3 PTv3-PPT with ALC (Ours)

soap dispenser

papercutter

| |
|---|

paper bag

tray

sign

|tube cup<br><br>bar<br><br>papertowelroll<br><br>guitar|100<br><br>101<br><br>102<br><br>103<br><br>clock<br><br>bucket<br><br>tissuebox<br><br>toasteroven|
|---|---|
|bowl<br><br>showerhead<br><br>closetrod<br><br>hairdryer<br><br>headphones|scale<br><br>plunger<br><br>handicapbar<br><br>dumbbell|

p

toiletpaperdispenser

laundrydetergent

ironingboard

broom

soapdish

paper

case of water bottles

extinguisher fire

projector

showerfloor

mouse

showercurtainrod b

lightswitch

ll

show

coatrack

coffee kettle

power outlet

toaster

dustpan

ball

vent

- Figure 4. Correctly predicted tail class points on ScanNet200 validation set. We compare the number of correctly predicted points of selected tail class in ScanNet200 validation sets between PTv3 trained from scratch and the PTv3-PPT trained with our datasets. With our dataset, Point Transformer gains more ability to detect rase classes. Tail classes that are not predicted by any models are ignored in this plot, and we present the full tail class performance difference in the supplementary.

3D Scene

Manual Annotations

LabelMaker auto-labels (Ours)

OpenScene [23]

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

- Figure 5. Visualization on ARKitScenes. From left to right: 3D scene, ground truth annotation (black regions indicate unannotated areas), LabelMaker annotations, OpenScene predictions.

grating Grounded-SAM and gravity alignment. For this study, we manually annotate three ARKitScenes and provide additional visualizations in Figure 5. Figure 3 qualitatively demonstrates how gravity alignment corrects prediction errors in a video with continuous viewpoint rotation.

Scaling Potential of LabelMaker and PTv3. In Table 4, we conduct an ablation study by varying the number of pretraining samples from the ARKit LabelMaker dataset and evaluating fine-tuning performance on ScanNet200. The log-linear relationship between loss and data size is well-

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 6. Self-captured scenes and the semantic segmentation generated by LabelMakerv2. This figure provides a qualitative impression of the segmentation quality obtained by LabelMakerV2. The color mapping is defined here.

documented in large language models. In our experiments we cannot find such trend for validation loss at either the pre-training or fine-tuning stage. Instead, we find a loglinear relation between validation mIoU and dataset size from 20% onwards, which in turn indicates further scaling opportunities for even larger datasets ( Figure E3 ). We also run a pre-training experiment with the same data volume from the synthetic Structured3D dataset, which showed no performance gain. This further highlights the importance of our pipeline.

#### 4.6. Transferability to other Domains

Efficacy in Downstream task. Since ARKit LabelMaker is auto-labeled for 3D semantic segmentation, an important question is how useful this data generation approach is for other related 3D perception tasks. To assess this, we evaluate the general features learned through supervised training on our dataset by using PTv3 as the backbone and finetuning PointGroup [12] for 3D instance segmentation on ScanNet and ScanNet200. Results in Table 8 show a substantial performance improvement on ScanNet200 when incorporating our ARKit LabelMaker dataset. This suggests that auto-labeling as a method for scaling data is not limited to a single task but can serve as a general pretraining objective for learning robust features, akin to how ImageNet pretraining has become a standard for image encoders.

Zero-shot Evaluation on Matterport3D. We perform inference on the Matterport3D [6] dataset using the ScanNet200 label space, and map the labels to the Top40/80/160 NYU classes as done in [23]. PTv3 trained with ARKit LabelMaker surpasses both directly supervised models and OpenScene [23], an open-vocabulary model. These

label space Top-40 NYU Classes Top-80 NYU Classes Top-160 NYU Classes method mIoU mAcc tAcc mIoU mAcc tAcc mIoU mAcc tAcc MinkowskiNet [7] trained on Matterport3D (supervised) - 50.8 - - 33.4 - - 18.4 MinkowskiNet trained on ScanNet200 33.1 43.5 74.5 19.9 28.4 72.3 11.7 17.5 71.5 MinkowskiNet trained on ARKit LabelMakerSN200 + ScanNet200 38.9 49.6 77.5 23.2 32.7 75.3 13.6 20.6 74.5 OpenScene[23] - 50.9 - - 34.6 - - 23.1 vanilla PTv3 trained on ScanNet200 36.2 46.1 77.3 22.0 29.8 75.0 13.0 18.8 74.1 PTv3-PPT without ARKit LabelMaker (Our reproduction) 41.4 51.2 79.9 27.0 36.2 77.7 16.3 24.4 76.8 PTv3-PPT trained with ARKit LabelMaker (Ours) 43.8 53.6 80.6 29.1 38.2 78.5 17.3 26.0 77.6

- Table 7. Zero-shot evaluation on Matterport3D [6] test set region segmentation. We evaluate our trained PTv3 model on the ScanNet200 label space and map it to the top-40/80/160 NYU classes. Our models outperform fully-supervised MinkowskiNet and OpenScene.

ScanNet [9] ScanNet200 [30]

Method mAP25 mAP50 mAP mAP25 mAP50 mAP vanilla PTv3 PointGroup (from [40] 77.5 61.7 40.9 40.1 33.2 23.1 vanilla PTv3 PointGroup (reproduced) 77.1 62.9 41.1 37.9 30.6 21.0 PTv3-PPT PointGroup pretrained [40] 78.9 63.5 42.1 40.8 34.1 24.0 PTv3-PPT PointGroup pretrained w/ ARKit LabelMaker (Ours) 78.5 62.6 41.4 42.9 34.9 24.5

- Table 8. Instance segmentation scores on ScanNet(200) [9, 30]. Our dataset improves instance segmentation as a downstream task. Pre-training with our dataset increases performance on both ScanNet and ScanNet200, with a significant gain on ScanNet200.

the original LabelMaker [37], our improved pipeline does not achieve perfect accuracy. While [37] demonstrated that its annotations are comparable to crowd-sourced human labels, training on noisy data always carries the risk of introducing systematic errors. For safety-critical applications, rigorous evaluation on accurately annotated data remains essential when leveraging tools like ours for training data generation. Does large-scale pretraining with automatic labels exhibit similar trends as seen in language and image generation tasks? Our results suggest so, showing measurable improvements across multiple architectures and tasks when (pre)training on ARKit LabelMaker. However, while real-world data proves significantly more effective than synthetic data, the scale of generated 3D data remains orders of magnitude smaller than that of image datasets. Our pipeline facilitates data collection, making it easier to expand training datasets as more scans become available.

results indicate that ARKit LabelMaker has sufficient scale and diversity of real-world appearance to strongly improve model generalization.

Data Scaling via Mobile Integration. While LabelMaker demonstrates success in large-scale 3D training on ARKitScenes, the dataset’s diversity remains limited. To scale our pipeline to arbitrary real-world environments, we integrate the iOS app Scanner 3D into LabelMaker, leveraging modern mobile devices’ ubiquitous RGB-D capture capabilities. This integration enables automatic annotation of scenes recorded with consumer iPhones. We validate scalability by processing two self-captured scenes, a kitchen and a fireplace, recorded using an iPhone 12 Max in a holiday cottage. Figure 6 showcases reconstructed scenes and their semantic segmentations, confirming the pipeline’s effectiveness for diverse real-world settings.

### 5. Conclusion

In this paper, we introduced ARKit LabelMaker, the largest real-world 3D RGB-D dataset with 3D semantic annotations, generated automatically using an enhanced version of LabelMaker [37], which we call LabelMakerV2. While these labels are inherently imperfect due to automation, we demonstrate their effectiveness in pre-training widely used 3D semantic segmentation models, significantly outperforming traditional, self-supervised, and augmentationheavy training approaches. This aligns with trends in language and image generation, where scaling up training data has led to substantial performance gains. To support further data collection for training and evaluation, we also provide integration with an existing iOS app.

#### 4.7. Limitations & Broader Impact

While we enhance LabelMaker [37] with an improved point cloud pipeline, we omit the generation of 2D segmentation maps. The computational cost of NeRF-based lifting across the entire ARKitScenes dataset exceeds our available resources, requiring approximately 12 additional GPU hours per scene. An interesting future research direction would be to explore more efficient 2D lifting methods and assess whether training 2D models on this data yields performance gains comparable to those observed for 3D models. Furthermore, 20 ARKitScenes scenes are excluded from processing due to missing pose data. Since LabelMakerV2 relies on accurate poses, future iterations of the software stack could incorporate techniques to reconstruct missing pose information, enabling broader dataset coverage. Like

Acknowledgements. We thank Xiaoyang Wu, author of PointTransformerV3 [40], for valuable and helpful advice. This project is partially supported by an SNSF Postdoc.Mobility fellowship, the ETH Foundation through a CareerSeed grant, and the Lamarr Institute.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Iro Armeni, Ozan Sener, Amir R. Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3D Semantic Parsing of Large-Scale Indoor Spaces. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 1, 2
- [3] Matan Atzmon, Haggai Maron, and Yaron Lipman. Point Convolutional Neural Networks by Extension Operators. ACM Transactions On Graphics (TOG), 2018. 2
- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, et al. ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data. In International Conference on Neural Information Processing Systems (NeurIPS), 2021. 1, 2, 4
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language Models are Few-shot Learners. International Conference on Neural Information Processing Systems (NeurIPS),

2020. 1

- [6] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGBD Data in Indoor Environments. International Conference on 3d Vision (3dV), 2017. 4, 7, 8
- [7] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2, 3, 4, 5, 6, 8
- [8] Pointcept Contributors. Pointcept: A Codebase for Point Cloud Perception Research. https://github.com/ Pointcept/Pointcept, 2023. 3
- [9] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1, 2, 4, 5, 8
- [10] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views. In International Conference on Learning Representations (ICLR),

2024. 1

- [11] Binh-Son Hua, Minh-Khoi Tran, and Sai-Kit Yeung. Pointwise Convolutional Neural Network. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [12] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, ChiWing Fu, and Jiaya Jia. PointGroup: Dual-Set Point Group-

- ing for 3D Instance Segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 7
- [13] Li Jiang, Zetong Yang, Shaoshuai Shi, Vladislav Golyanik, Dengxin Dai, and Bernt Schiele. Self-supervised Pretraining with Masked Shape Prediction for 3D Scene Understanding. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [14] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment Anything. arXiv preprint arXiv:2304.02643, 2023. 3
- [15] Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, and Jiaya Jia. Stratified Transformer for 3D Point Cloud Segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [16] Loic Landrieu and Martin Simonovsky. Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [17] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. PointCNN: Convolution on Xtransformed Points. In International Conference on Neural Information Processing Systems (NeurIPS), 2018. 2
- [18] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [19] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding Dino: Marrying Dino with Grounded Pre-training for Open-set Object Detection. European Conference on Computer Vision (ECCV), 2024. 3
- [20] George A. Miller. Wordnet: a lexical database for english. Commun. ACM, 38(11):39–41, 1995. 3
- [21] Alexey Nekrasov, Jonas Schult, Or Litany, Bastian Leibe, and Francis Engelmann. Mix3d: Out-of-context Data Augmentation for 3D Scenes. In International Conference on 3d Vision (3dV), 2021. 1, 2, 3, 4, 5, 6
- [22] Chunghyun Park, Yoonwoo Jeong, Minsu Cho, and Jaesik Park. Fast Point Transformer. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [23] Songyou Peng, Kyle Genova, Chiyu ”Max” Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser. OpenScene: 3D Scene Understanding with Open Vocabularies. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 7, 8
- [24] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
- [25] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. In International Conference on Neural Information Processing Systems (NeurIPS), 2017. 2

- [26] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. OpenAI blog, 2018. 1
- [27] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019. 1
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1
- [29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In International Conference on Computer Vision and Pattern Recognition (CVPR),

2022. 1

- [30] David Rozenberszki, Or Litany, and Angela Dai. LanguageGrounded Indoor 3D Semantic Segmentation in the Wild. In European Conference on Computer Vision (ECCV), 2022. 1, 2, 4, 5, 8
- [31] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3D for 3D Semantic Instance Segmentation. In International Conference on Robotics and Automation (ICRA), 2023. 1, 2
- [32] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica Dataset: A Digital Replica of Indoor Spaces. arXiv,

2019. 2

- [33] Ayca Takmaz, Alexandros Delitzas, Robert W Sumner, Francis Engelmann, Johanna Wald, and Federico Tombari. Search3D: Hierarchical Open-Vocabulary 3D Segmentation. IEEE Robotics and Automation Letters (RA-L), 2025. 1
- [34] Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Fran¸cois Goulette, and Leonidas J. Guibas. KPConv: Flexible and Deformable Convolution for Point Clouds. In International Conference on Computer Vision (ICCV), 2019. 2
- [35] Mathias Vogel, Keisuke Tateno, Marc Pollefeys, Federico Tombari, Marie-Julie Rakotosaona, and Francis Engelmann. P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising. In European Conference on Computer Vision (ECCV),

2024. 1

- [36] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. International Conference on Neural Information Processing Systems (NeurIPS), 2021. 3
- [37] Silvan Weder, Hermann Blum, Francis Engelmann, and Marc Pollefeys. LabelMaker: Automatic Semantic Label Generation from RGB-D Trajectories. In International Conference on 3d Vision (3dV), 2024. 1, 2, 3, 4, 8

- [38] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682,

- 2022. 1

[39] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer V2: Grouped Vector Attention and Partition-based Pooling. In International Conference on Neural Information Processing Systems (NeurIPS),

- 2022. 2, 3

- [40] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler, Faster, Stronger. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 4, 5, 6, 8
- [41] Xiaoyang Wu, Zhuotao Tian, Xin Wen, Bohao Peng, Xihui Liu, Kaicheng Yu, and Hengshuang Zhao. Towards Largescale 3D Representation Learning with Multi-dataset Point Prompt Training. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3
- [42] Yifan Xu, Tianqi Fan, Mingye Xu, Long Zeng, and Yu Qiao. SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters. In European Conference on Computer Vision (ECCV), 2018. 2
- [43] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. In International Conference on Computer Vision (ICCV), 2023. 1, 4
- [44] Gonca Yilmaz, Songyou Peng, Marc Pollefeys, Francis Engelmann, and Hermann Blum. OpenDAS: Open-Vocabulary Domain Adaptation for 2D and 3D Segmentation. arXiv preprint arXiv:2405.20141, 2024. 1
- [45] Yuanwen Yue, Anurag Das, Francis Engelmann, Siyu Tang, and Jan Eric Lenssen. Improving 2D Feature Representations by 3D-Aware Fine-tuning. In European Conference on Computer Vision (ECCV), 2024. 1
- [46] Yuanwen Yue, Sabarinath Mahadevan, Jonas Schult, Francis Engelmann, Bastian Leibe, Konrad Schindler, and Theodora Kontogianni. Agile3d: Attention Guided Interactive Multiobject 3D Segmentation. International Conference on Learning Representations (ICLR), 2024. 1
- [47] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point Transformer. In International Conference on Computer Vision (ICCV), 2021. 2
- [48] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling. In European Conference on Computer Vision (ECCV), 2020. 2, 4, 6
- [49] Haoyi Zhu, Honghui Yang, Xiaoyang Wu, Di Huang, Sha Zhang, Xianglong He, Tong He, Hengshuang Zhao, Chunhua Shen, Yu Qiao, and Wanli Ouyang. PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm. arXiv preprint arXiv:2310.08586, 2023. 1, 2, 3, 4, 5, 6

## ARKit LabelMaker: A New Scale for Indoor 3D Scene Understanding Supplementary Material

### A. Dataset Class Statistics

Dataset Statistics of ARKit LabelMaker. In Figure A1, we present the point count for each class in the LabelMaker WordNet label space. Our dataset maintains a substantial data distribution even across tail classes.

- 103

- 104

- 105

- 106

- 107

- 108

- 109

#points

apparel

fireplace

picture

blanket

basket

fridge

radiator

backpack

handcart

bed

bathtub

blind

microwave

display

plant

teddy

table

chair

lamp

cabinet

door

stand

sofa

slipper

bicycle

wall

curtain

range_hood

floor

ashcan

mat

coffee_table

ceiling

doorframe

bottle

book

stairway

step

vase

bannister

wardrobe

desk

person

oven

toilet

window

countertop

washer

sink

towel

stool

stove

bookshelf

cushion

swivel_chair

footstool

mirror

chest_of_drawers

clothes_hamper

shower_stall

box

armchair

- 103

- 104

- 105

- 106

- 107

- 108

- 109

#points

baggage

jacket

projector

bucket

bag

bag

paper

loudspeaker

ladder

laptop

kettle

fan

guitar

coffee_maker

railing

jar

sleeping_bag

tray

plate

piano

food

candle

rack

light

faucet

pool_table

coatrack

dumbbell

printer

stapler

bench

bowl

mug

hamper

dish_rack

booth

paper_towel

sack

decoration

seat

clock

shirt

toaster

computer

dispenser

switch

column

shoe

soap_dispenser

kitchen_island

dishwasher

chopping_board

blackboard

furniture

windowsill

step_stool

shower_curtain

signboard

toilet_tissue

mailbox

alarm_clock

- 103
- 104
- 105
- 106
- 107
- 108
- 109

| | | |
|---|---|---|
| | | |
| | | |
| | | |
|treadmill<br><br>| |dais|

#points

rod

bathrobe

rope

tire

iron

treadmill

umbrella

shredder

beanbag

broom

remote_control

control

bar

frying_pan

gat

pipe

tripod

duplicator

dais

hat

dryer

dustpan

paper_cutter

banner

plunger

coat

crate

soap

coat_hanger

tennis_racket

pitcher

brush

hand_blower

soap_dish

ironing_board

scale

heater

earphone

drinking_fountain

vent

vacuum

container

metronome

fire_alarm

machine

water_heater

dartboard

vending_machine

envelope

toothbrush

music_stand

water_cooler

telephone

dollhouse

helmet

mouse

computer_keyboard

sewing_machine

fire_extinguisher

toolbox

thermostat

thermos

Figure A1. Number of points for each ARKitLabelMaker class.

### B. PTv3 Results on ScanNet++

We also report the training and evaluation results of PTv3 on ScanNet++ in Table B1. Unfortunately, the numbers are not fully comparable, because we were so far unable to reporduce the validation results of PTv3. When the authors of [40] released PTv3’s performance on ScanNet++, they expanded Structured3D’s training set from 6,519 to 18,348 samples, which we refer to as Structured3D v2. Due to limited computational resources, we could not train with this updated version of Structured3D yet. We will update ScanNet++ results once the new result is available. Our pretraining and joint-training (PPT) experiments show performance gains over vanilla PTv3, with PTv3-PPT achieving similar improvements to the original PTv3-PPT but with significantly less training data.

### C. Tail Classes Performance of PTv3

We give a detailed plot of the number of correctly predicted tail class points on ScanNet200 validation set in Figure C2.

PTv3 Variant Training Data #Data val mIoU

vanilla ScanNet++ 713 41.8 fine-tune (Ours) ARKit LabelMakerSN200 → ScanNet++ 4471 → 713 42.5 PPT [40] ScanNet200 + ScanNet++ + Structure3Dv2 45868 45.3† PPT (Ours) ScanNet200 + ScanNet++ + ARKit LabelMaker 11168 44.5 PPT (Ours) ScanNet+ ScanNet200 + ScanNet++ + Structure3D + ARKit LabelMaker 30386 44.6

Table B1. Semantic Segmentation Scores on ScanNet++ [43]. We evaluated the efficacy of our ARKit LabelMaker dataset on the ScanNet++ benchmark using both pre-training and joint training methods. †: this number comes from Wu et al..

- 100

- 101

- 102

- 103

- 104

Vanilla PTv3 PTv3-PPT with ALC (Ours)

#points

0

paperbag

projector

bucket

broom

handicapbar

speaker

luggage

paper

bar

ball

calendar

guitar

cup

closetrod

tray

plate

candle

papertowelroll

hat

dustpan

plunger

crate

foldedchair

dumbbell

tube

hairdryer

sign

bowl

papercutter

coffeekettle

coatrack

cdcase

scale

purse

guitarcase

clock

vent

poster

soapdish

showercurtainrod

toaster

ironingboard

stuffedanimal

headphones

keyboardpiano

waterpitcher

powerstrip

firealarm

waterbottle

toiletpaperholder

soapdispenser

lightswitch

mouse

storagecontainer

musicstand

showerhead

poweroutlet

toiletpaperdispenser

toasteroven

showerfloor

fireextinguisher

caseofwaterbottles

laundrydetergent

alarmclock

tissuebox

toiletseatcoverdispenser

Figure C2. Correctly predicted tail class points on ScanNet200 validation set. We compare the number of correctly predicted points of tail class in ScanNet200 validation sets between PTv3 trained from scratch and the PTv3-PPT trained with our datasets. With our dataset, Point Transformer gains more ability to detect rase classes.

### D. Detailed process of applying LabelMaker to ARKitScenes

ARKitScenes is one of the largest indoor 3D scenes dataset. It consists of 5047 parsable scenes of various size. We consider a scene parsable if the RGB-D trajecotry comes with associated pose data. Processing these scenes with our improved LabelMaker pipeline requires deliberate engineering with the following goals: a) Bring the data in to the format required by LabelMaker [37] b) Robust processing to not waste compute on failures, c) Improved parallelization to speed up processing. d) Accurate resource estimation to prevent waste of compute resources and longer job waiting time. e) Fast failure identification and results inspection.

Transforming the data LabelMaker [37] requires a specific data format to be able to reliably process all data. All trajectories require posed RGB-D data and a denoised 3D model that is used by Mask3D. ARKitScenes comprises data of varying resolutions and sampling rates. Depth data is captured at 256×192 and 60 FPS, while the RGB frames

are recorded at 640 × 480 and 30 FPS. Therefore, synchronization is required to process the data with LabelMaker. To this end, we match each RGB frame with the closest depth frame in time and we resize the depth frame to RGB frame’s resolution. Pose data, originally at 10 FPS, is interpolated using rotation splines to synchronize with each RGB frame. To obtain a 3D mesh of each scene that can be processed by Mask3D, we reconstruct the 3D model by fusing the synchornized posed RGB-D data using TSDF fusion and then extract mesh with marching cube algorithm. We empirically chose a voxel size of 8mm and a truncation distance of 4cm for fusion.

Building a scalable pipeline LabelMaker [37] can be decomposed into individual modules such as the individual base models, the consensus computation, and the 3D lifting. This modular nature allows to build a scalable pipeline using popular high-performance computing toolboxes. As the different base models have different runtimes, we had to leverage dependency management system that can handle different dependencies of the pipeline steps. This architecture allows us to effectively leverage large-scale computing and distribute the processing across many different nodes.

In more detail, we decompose the pipeline into several jobs (ordered by dependency) for each scene:

- 1. Preprocessing: Downloading the original scene data, transforming it into LabelMaker format, and run TSDF fusion to get the 3D mesh of the scene.
- 2. Forwarding 2D images or 3D meshes to each base models: Grounded-SAM, Mask3D, OVSeg, CMX, InternImage. (all jobs depends on step 1.)
- 3. Getting the consensus label from base models’ labels. (depends on all elementary jobs in step 2.)
- 4. Lifting the 2D consensus label into 3D. (depends on step 3.)
- 5. Rendering the outputs of base models or consensus into videos for visualization. (depends on steps 2. or 3.)
- 6. Post-processing, including removing temporary files and get statistics of each tasks. (depends on all steps above) Optimizing compute resource scheduling. ARK-

itScenes contains scenes of a wide range of sizes, spanning from a minimum of 65 frames to a maximum of 13796, and different parts of the pipeline scale differently with increasing scene size. To figure out the minimum resources requirements, we select 11 scenes of varied sizes uniformly distributed within the minimum and maximum range and record their resources usage. While most jobs are not sensitive to scene size and can suffice with a fixed resource allocation, the base models exhibit greater sensitivity to scene size. We interpolate resource needs with respect to scene size and summarize the empirical numbers into Table D2. Through this, we ensure that we request minimal-required resources, so that we have lowest job waiting time and less idle compute power.

Task #CPU RAM Time GPU

Download & Prepossessing 2 24G 4h Video Rendering 8 32G 30min Grounded-SAM 2 12G 6h 3090 ×1 OVSeg 2 8G 8h 3090 ×1 InternImage 2 10G 8h 3090 ×1 Mask3D 8 16G 1h 30min 3090 ×1 OmniData 8 8G 2h 3090 ×1 HHA 18 9G 2h CMX 2 8G 3h 3090 ×1

Consensus 16 16G 2h Point Lifting 2 72G 4h -

Table D2. Requested resources for each task. We report the average resources required by the individual steps of the LabelMakerv2 pipeline. The required cores, RAM, and GPU time varies across the different jobs. Through our job scheduling mechanism, we ensure that the required compute is optimially distributed across all jobs.

Assuring the quality of the individual processings. In order to assure high-quality labels produced by our improved pipeline, we have built tooling to efficiently check for failures of the processed scenes. To this end, we store the logs and statistics of each job and built visualization tools for this data as well as the intermediate predictions. This allows us to conveniently browse at scale through the predictions and identify individual failures.

Failure handling and compute resource optimization. When doing large-scale processing on a high-performance compute cluster, a common issue is the failure of jobs. This can happen for several reasons such as node crashing, unexpected memory usage, and many more. Therefore, the processing pipeline has to be robust to these failures. Additionally, compute should not be wasted when recovering from these failures. Therefore, we designed a simple yet effective strategy for efficiently recovering from job failures. Before every restart is triggered for a scene, we analyze both the logs and file system to identify the jobs that have not finished for this scene. Once these jobs have been identified, we only resubmit these jobs. This ensures that no compute resource is used in rerunning completed tasks.

### E. Log-Linear Performance Relation in Data Scaling

#### E.1. Implementation Details

We optimize the code to deploy each individual piece of the pipeline of LabelMakerV2 as individual jobs to a GPU cluster, with SLURM as a dependency manager between the pipeline pieces. To optimize the overall execution time, it is therefore important to be able to estimate the processing time of each piece of the pipeline at the point of job submission. ARKitScenes contains scenes of a wide range of sizes, spanning from a minimum of 65 frames to a maximum of

[Figure 23]

Figure E3. Relation of Validation mIoU against training data percentage of ARKit LabelMaker. This figure shows the validation mIoU on ScanNet200 after fine-tuning with respect to the percentage of ARKit LabelMaker data used in pre-training. This figure shows a log-linear relationship.

13796, and different parts of the pipeline scale differently with increasing scene size. To figure out the minimum resources requirements, we select 11 scenes of varied sizes uniformly distributed within the minimum and maximum range and record their resources usage. While most jobs are not sensitive to scene size and can suffice with a fixed resource allocation, the base models exhibit greater sensitivity to scene size. We interpolate resource needs with respect to scene size and summarize the empirical numbers in the Appendix. Through this, we ensure that we request minimalrequired resources, so that we have lowest job waiting time and less idle compute power.

We use a CentOS 7 based SLURM cluster to process all the data, which is capable of handling task dependencies and parallel processing. Before submitting jobs for a single scene, we first check the progress of each job and generate a SLURM script to submit only those unfinished jobs. This ensures that no compute resource is used in rerunning completed tasks.

We employ test time augmentation by forwarding all models twice, with Mask3D using two different random seeds and other models being mirror flipped. Following the practice of LabelMaker [37], we assign equal weights to each model when calculating the consensus, although these weights are configurable in our pipeline code. Since we are primarily interested in the 3D labels that can be used for pre-training 3D semantic segmentation models, SDFStudio training and rendering are omitted due to their lengthy processing times. Further, we enhance the pipeline by storing both the most and second most voted predictions alongside their respective vote counts. This information is useful when we want to investigate on the uncertainty across the base models. We leave the exploitation of this information as potential future direction of research.

