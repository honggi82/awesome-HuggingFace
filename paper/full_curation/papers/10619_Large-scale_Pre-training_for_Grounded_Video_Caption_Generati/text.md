## Large-scale Pre-training for Grounded Video Caption Generation

# arXiv:2503.10781v3[cs.CV]9Sep2025

Evangelos Kazakos1, Cordelia Schmid2, Josef Sivic1 1Czech Institute of Informatics, Robotics and Cybernetics at the Czech Technical University in Prague 2Inria, Ecole´ normale sup´erieure, CNRS, PSL Research University

https://ekazakos.github.io/grounded_video_caption_generation/

[Figure 1]

###### a woman is stirring something in a wok by using a spatula

Figure 1. Output of our GROunded Video caption gEneration (GROVE) model on an instructional video. The model outputs a video-level caption (bottom) with key noun phrases in the caption coloured and localised (grounded) in the video by temporally consistent bounding boxes (top). Note how the objects are consistently annotated (with the same color) despite changes in scale and viewpoint and how the person is marked as occluded (orange box not present) in frames 1 and 4 when the person (or their hand) is not visible.

1

### Abstract

We propose a novel approach for captioning and object grounding in video, where the objects in the caption are grounded in the video via temporally dense bounding boxes. We introduce the following contributions. First, we present a large-scale automatic annotation method that aggregates frame-level captions grounded with bounding boxes into temporally dense and consistent annotations. We apply this approach on the HowTo100M dataset to construct a large-scale pre-training dataset, named HowToGround1M. We also introduce a Grounded Video Caption Generation model, dubbed GROVE, and pre-train the model on HowToGround1M. Second, we introduce iGround–a dataset of 3513 videos with manually annotated captions and dense spatio-temporally grounded bounding boxes. This allows us to measure progress on this challenging problem, as well as to fine-tune our model on this small-scale but high-quality data. Third, we demonstrate that our approach achieves state-of-the-art results on the proposed iGround dataset, as well as on the VidSTG, ActivityNet-Entities, GroundingYouTube, and YouCookInteractions datasets. Our ablations demonstrate the importance of pre-training on our automatically annotated HowToGround1M dataset followed by fine-tuning on the manually annotated iGround dataset and validate the key technical contributions of our model. The dataset and code are available at https://ekazakos.github.io/ grounded_video_caption_generation/.

### 1. Introduction

We seek to generate grounded captions for a given input video. As illustrated in Figure 1, the task is challenging as it entails both (i) generating the natural language caption for the video and (ii) predicting temporally dense bounding boxes for multiple noun phrases from the caption. Compared to grounded captioning in still images [6, 20, 29, 46, 49], the task in the video domain has the additional difficulty that objects might disappear in some frames because of occlusions and we need to produce temporally dense and consistent bounding boxes across the frames of the input video. This problem is important as spatio-temporal grounding of individual objects with natural language descriptions on a large scale is one of the key steps to advance areas such as human-robot interaction and embodied perception [17, 22, 28, 34, 52].

Despite the advances on the grounded video caption generation problem [21, 44, 50] one of the key limiting factors hindering further progress is the lack of suitable large-scale videos datasets with captions densely grounded with multiple spatio-temporal boxes in the video. Existing datasets are restricted to localising a single spatio-temporal tube for each short textual description [5, 37, 38, 48], have limited temporal consistency as they provide bounding boxes for only a few sparsely sampled frames per video [10, 13, 50], or are limited to a specific domain such as egocentric videos [8, 10, 19].

In this work, we address this key limitation by the following three contributions. First, to address the issue of limited

Dataset Annot. type

Multiple frames

Multi-object grounding

Num. videos

Num. instances

VidSTG [48] Manual ✓ ✗ 36.2K 9.9M HC-STVG [38] Manual ✓ ✗ 10.1K 1.5M ActivityNet-Entities [50] Manual ✗ ✓ 37.4K 93.6K HowToGround1M (Ours) Automatic ✓ ✓ 1M 80.1M iGround (Ours) Manual ✓ ✓ 2K 236.9K

- Table 1. Comparison of our two datasets iGround and HowToGround1M with state-of-the-art video grounding datasets.

training data, we introduce a large-scale automatic annotation method leveraging an existing model for grounded still-image captioning together with an LLM to summarize frame-level captions into video-level captions. The LLM is also tasked to perform temporally consistent bounding box annotation, associating frame-level phrases that correspond to objects with video-level phrases, resulting in a video-level caption grounded with multiple object tubes with consistent natural language labels. We apply this approach to videos from the HowTo100M [23] dataset, which results in a new large-scale pre-training dataset, namely HowToGround1M, of 1M videos for this problem. This automatic annotation method is coupled with a proposed GROunded Video Caption gEneration model, called GROVE. The key technical contributions of this model include: (i) spatio-temporal adapters, which enable efficient modeling of spatio-temporal information in video; (ii) a bounding box decoder that outputs temporally coherent bounding boxes in video and (iii) a temporal objectness head that explicitly models objects that temporary leave the frame or are occluded. We pre-train the GROVE model on the proposed large-scale automatically annotated HowToGround1M dataset. Second, we introduce a new manually annotated dataset for the grounded caption generation task, which we name iGround. The dataset contains 3513 videos and more than 230,000 annotated object bounding boxes. We split this dataset into train/val/test sets (2013/500/1000, respectively). This allows us to measure progress on this challenging problem, as well as to fine-tune our model on small-scale but high-quality data. Third, our results demonstrate that our GROVE model achieves state-ofthe-art performance on the proposed iGround dataset, as well as on the VidSTG [48], ActivityNet-Entities [50], GroundingYouTube [4], and YouCook-Interactions [37] datasets. We perform extensive ablations that demonstrate the importance of pre-training using our automatically annotated HowToGround1M dataset followed by fine-tuning on the manually annotated iGround dataset and validate the key technical contributions of our model. For additional details, see the appendix.

### 2. Related Work

Image-based grounded data generation. Recent efforts in multi-modal learning have focused on grounding text to images and comprehending referring expressions [6, 20, 29, 31, 46, 49]. Given the scale of these models,

large-scale training data are essential, yet manual annotation is prohibitively expensive. To address this, methods typically leverage pre-trained models for automatic annotation. Several approaches rely on LLM-driven data generation [6, 46], where GPT-4 [27] is used to create grounded dialogue datasets. Some methods pair instruction-tuned captions with bounding boxes [46], while others generate QA pairs from existing bounding boxes and captions [6]. Other methods leverage NLP techniques, such as extracting noun chunks via Part-of-Speech tagging and aligning them with bounding boxes using a pre-trained grounding model [29]. More complex multi-stage annotation pipelines integrate multiple pretrained models to generate large-scale pseudolabeled datasets [31]. We build on this line of work but extend it to the video domain.

Datasets for spatio-temporal grounding in video. The existing datasets [5, 12, 37, 38, 48, 50] for spatio-temporal video grounding are relatively small scale as they rely on manual annotation, which is tedious and time consuming. Typically, the existing datasets also focus on localizing a single spatio-temporal tube for each short textual description [5, 37, 38, 48], which can be a limiting factor in instructional videos where multiple objects are often manipulated. In contrast, both our automatically annotated HowToGround1M dataset as well as our manually-labelled iGround dataset contain multiple spatio-temporal bounding boxes grounding multiple objects described in the caption. We compare our HowToGround1M dataset as well as our iGround dataset with exisiting video grounding datasets in Table 1. ActivityNetEntities [50] is closest to ours in that it provides grounded captions – multiple objects per frame for annotated noun phrases in the caption. Nevertheless, the authors annotate a single frame per object in the video segment while both our automatically annotated HowToGround1M as well as our manually annotated iGround dataset have densely annotated frames per video segment. Moreover, HowToGround1M has the largest number of videos across all datasets and the largest number of annotated instances. More statistics and analysis of our proposed datasets is provided in appendix D.

Spatio-temporal grounding in video. Spatio-temporal grounding [5, 11, 14, 18, 36–38, 42, 43, 47, 48] aims to predict a single spatio-temporal tube enclosing an event described in a natural language query. Early approaches relied on object detection features [38, 48]. [48] introduced a spatio-temporal graph encoder, while [38] and [43] adapted transformers for multi-modal grounding. Some works employed contrastive learning on large-scale instructional videos, leveraging weak supervision from HowTo100M [5, 37]. Others explored architectural innovations, such as twostream models for appearance and motion [18] or contextguided decoding for object-centric grounding [11]. However, these methods lack a text generation component, as they assume the query is given, and they cannot generate multiple

[Figure 2]

- Figure 2. A method for automatic annotation of spatio-temporally grounded captions. In the first stage (left), we apply a still-image grounded caption generation model on individual video frames producing temporally inconsistent outputs. In the second stage (middle), the captions from individual frames are aggregated into a single video-level caption describing the most salient actions/objects in the video. Third (right), individual frame-level phrases and bounding boxes are associated over time into a temporally consistent and dense labelling of object bounding boxes over the video.

spatio-temporal tubes for multiple objects. Our approach addresses this issue with the GROVE model, a novel automatic annotation method, and a manually-labeled dataset for fine-tuning and evaluation. More pertinent to our work is [50], which separately performs captioning and grounding. In contrast, our GROVE model jointly (i) captions and (ii) grounds noun phrases with temporally dense bounding boxes while handling occlusions. Concurrent to our work is [26], which targets mask-level video grounding and trains on semi-automatic annotations from existing benchmarks. In contrast, we focus on bounding boxes, pre-train with large-scale fully automatic annotations and fine-tune on human-annotated data. We outperform both [26] and [50] .

### 3. Large-scale generation of grounded captions

In this section, we first introduce our automatic annotation method for generating a large-scale dataset for grounded video caption generation (Sec. 3.1). We then describe HowToGround1M (Sec. 3.2), our dataset created by applying this method to videos from HowTo100M [23, 35].

#### 3.1. Automatic annotation method

We describe our method for generating an automatically annotated dataset for grounded video caption generation. Given an unlabelled dataset of videos depicting humans interacting with objects and/or other humans, the goal of the method is to generate both video-level captions describing what is happening in the video and temporally dense and consistent bounding boxes grounded to the noun phrases from the caption that describe the main objects in the video. We leverage foundation LMMs and LLMs as they have been pretrained on large-scale datasets and provide a rich source of information. Our method consists of three steps, as shown in Figure 2: i) frame-wise grounded caption generation, ii) video-level caption aggregation and iii) temporally consistent annotation of objects. We describe these steps next.

Stage 1: Frame-wise grounded caption generation. We

begin by generating grounded captions for individual video frames, producing text descriptions alongside bounding boxes for noun phrases referring to objects. As no video models exist for this task, we use image-based models in a frame-by-frame manner. We adopt GLaMM [31], which excels in image-based grounded captioning. Since GLaMM outputs segmentation masks, we convert them to bounding boxes. An example output is shown in Figure 2 (left). In the experiments, we also compare an alternative approach that combines frame-level captioning [41] with open-vocabulary object detection [24] to obtain candidate captions and bounding boxes in individual frames.

- Stage 2: Video-level caption aggregation. The previous stage provides outputs for individual frames, which are, however, not temporally consistent. We address this issue in Stage 2. From the frame-level captions, we generate a videolevel caption that highlights the most salient actions and objects while ensuring consistency in noun phrase annotations across frames. We achieve this by prompting Llama2 [39], as described next. Since the grounded captioning model produces lengthy captions with extraneous details, we first extract Subject-Verb-Object (SVO) triplets and relevant adpositional phrases using Part-of-Speech (POS) tagging. These structured triplets serve as input to the LLM, encapsulating subjects, actions, and relevant objects. To enhance accuracy, we perform in-context learning by providing example pairs of frame-level SVO triplets and their corresponding video-level captions (see Figure 12 for details). Given a new set of SVO triplets, the LLM generates a video-level caption and tags noun phrases corresponding to objects of interest within <p></p> tags. Figure 2 (middle) illustrates this process. In experiments, we compare the above approach with providing the LLM with full captions from Stage 1 instead of extracting Subject-Verb-Object triplets from the caption to assess the impact of additional context.
- Stage 3: Temporally consistent bounding box annotation. While Stage 2 ensures a consistent video-level caption,

[Figure 3]

- Figure 3. An overview of our GROVE model for grounded video caption generation. Dashed red rectangles outline the key technical contributions enabling grounded caption generation in video and include: (i) spatio-temporal adapters; (ii) the bounding box decoder and (iii) the temporal objectness head. VL: vision-to-language projection, LQ: language-to-query projection.

noun phrase annotations remain inconsistent across frames due to the use of an image-based model. To resolve this, we introduce temporal labeling of objects, ensuring that bounding boxes corresponding to the same object are consistently labeled throughout the video using the video-level noun phrases. These consistently labeled bounding boxes form video object tracks. We formulate this as a text classification task and prompt the LLM with in-context learning. The input consists of frame-level noun phrases to be classified and the video-level noun phrases serving as class labels. Figure 13 provides details on the prompt. Figure 2 (right) illustrates this process. An alternative approach is to use a visual tracker, such as [15], for associating bounding boxes over time, which we compare against in Sec. 6.2.

Upon completing all three stages, we obtain videos with automatically generated captions, grounded bounding boxes, and temporally coherent labels aligned with the noun phrases in the caption. Additional details are provided in appendix F.

#### 3.2. The HowToGround1M dataset

We apply our automatic annotation method to Internet instructional videos from the HowTo100M dataset [23]. We choose HowTo100M due to its diversity of actions, scenes, objects and lighting conditions. We apply our approach to clips obtained using time stamps from the HowToCaption [35] version of the dataset. These clips contain meaningful events in the video identified based on LLM analysis of the associated narrations. In detail, we randomly sample 1M video clips from HowTo100M videos using start/end timestamps from HowToCaption. We ensured there is no video

ID overlap with any of the downstream datasets, including iGround. The videos from HowTo100M have variable frame rates usually ranging in 25-30 fps, and we process the videos at 5fps. The majority of the clips are 8 seconds long with a spatial resolution of 455×256 pixels. We run our automatic annotation method on this set of data to obtain our HowToGround1M pre-training dataset. The resulting dataset contains 1M videos, with 1M captions containing 3.2M noun phrases. The captions contain 18.6k unique terms and 142k unique noun phrases. Overall, the dataset contains 43.6M annotated frames with 80.1M bounding boxes. This dataset is used to pre-train the GROVE model, described next.

### 4. The GROVE Model

We introduce a GROunded Video caption gEneration model, called GROVE, see Figure 3. The input to the model is a video clip with T frames (left) and the output is a natural language caption (right) together with N spatio-temporal bounding boxes localizing the individual noun phrases in the video together with a objectness score indicating the presence / absence of the object in a specific frame (top). We build on the GLaMM [31] model, which is a state-of-the-art method for still image-based grounded caption generation, and extend it to the video domain. The key technical components enabling grounded caption generation in videos are (shown in dashed red rectangles in Figure 3): i) the spatiotemporal adapters with pooling which enable modelling temporal information efficiently; ii) the bounding box decoder which allows re-using large-scale pretrained decoder weights [31]; and iii) the temporal objectness head for mod-

elling objects that temporary leave the frame or are occluded. Details are given below. We assess the importance of these components in Section 6. Additional details of the GROVE model and the training procedure are in appendix E.

Spatio-temporal adapters and pooling. The visual information is encoded using the Global Video Encoder Ve(·), which represents the video globally for captioning, and the Grounding Video Encoder Vg(·), which represents the finegrained details for grounding. We build these encoders by adapting the respective pre-trained image encoders [31]. We achieve this by interleaving spatio-temporal adapter layers (denoted as a()) between the image-based encoder layers. To stabilise training, we add residual connections and introduce a learnable parameter that is multiplied by the adapter’s output and starts from 0 at the beginning of training [1]. By doing so, at the beginning of training the adapter’s output is effectively cancelled out and the network observes only the original encoder’s output. As training progresses, the learnable parameter is tuned and the network automatically adjusts the contribution of the adapter based on the gradients of the loss. In detail, the adapter layer performs a(o) = o + tanh(α) × f(o), where o is the output of the preceding encoder layer, α is the tunable parameter that is initialised to 0 and passes through a tanh activation and f(·) is the adapter layer. As feeding the full video tokens oe to the LLM is computationally prohibitive, we introduce a spatio-temporal pooling function after the output of Ve(·), i.e., op = p(oe).

Bounding box decoder and prediction head. We adapt the pre-trained mask decoder [16, 31] for bounding box decoding. We focus on bounding boxes (rather than full pixel-level masks) as they are easier and cheaper to manually annotate yet provide good localization accuracy for compact objects, which are the main focus of this work. We transform the mask decoder to a bounding box decoder by using the embedded detection tokens as queries, and the visual features of the Grounding Video Encoder as keys/values, resulting in an output that has same length as the detection tokens, allowing us to predict a bounding box for each detection token that corresponds to a noun phrase in the caption. Importantly, while Vg(·) performs video processing, we apply the crossattention in a frame-wise fashion to predict objects at each frame of the input video. We employ a bounding box prediction head on the output of the bounding box decoder, od. It is an MLP that predicts bounding box coordinates for the embedded detection tokens at each frame: pbb = hbb(od), where pbb ∈ RT×N

d×4 are the bounding box predictions and hbb(·) is the bounding box head.

Temporal objectness head. As discussed previously, one major challenge for videos is that objects might disappear and reappear in different frames of the video. To address this, we introduce a temporal objectness head. Different than objectness predictions in image-based object detection, the

purpose of this head is to predict whether an object is visible or not at a given frame of a video: ptobj = htobj(od), where ptobj ∈ RT×N

d×1 are the temporal objectness scores and htobj(·) is the temporal objectness head. During inference, we threshold ptobj and for each frame we select only the bounding boxes for which the temporal objectness scores pass the threshold.

### 5. Manually annotated iGround dataset

For the iGround dataset, we select 3513 clips from the HowTo100M dataset [23] by sampling ‘interesting’ videos that typically include dynamic events or actions that are clear and distinguishable. In those events/actions, people usually interact with objects. We make sure that the 3513 clips do not overlap with those in the HowToGround1M dataset described in section 3. Moreover, we ensure that there is no video ID overlap with any of the downstream datasets. We split the set into a training set (2013 clips), a validation set (500 clips), and a test set (1000 clips), ensuring no overlap in video IDs between them. The video annotation itself consists of 3 steps. The first step entails watching the video and providing a natural language description of what is happening in the video and the objects that are being manipulated. Note, that we are interested in the active objects, i.e. objects that humans interact with, rather than densely describing all objects in the scene. In the second step, bounding boxes are annotated for all visible instances of humans/objects mentioned in the caption that has been provided in the previous step. Finally, each bounding box is annotated with a short phrase or word that should match exactly a short phrase/word from the caption. More information on the annotation procedure, including annotation guidelines for the raters and mechanisms to ensure the consistency and overall quality of the raters’ annotations, can be found in appendix G.

### 6. Experiments

In this section we introduce evaluation datasets and metrics, compare the proposed approach with the state of the art on three benchmark datasets, analyze the effect of the pretraining dataset size and ablate the key components of the proposed model and automatic annotation procedure.

#### 6.1. Datasets and evaluation metrics

We evaluate the proposed approach on three datasets, the newly introduced grounded video caption generation dataset iGround as well as the established VidSTG [48] and ActivityNets-Entities [50] video grounding benchmarks.

iGround. We build on the metrics for grounding captions in still images [31] and adapt them to our task in videos. These include METEOR [2] and CIDEr [40] for the quality of the captions, AP50 for the grounding accuracy, and recall [31] that combines (i) IoU between ground truth (GT)

Method METEOR CIDER AP50 Recall

- a. GLaMM [31] 11.9 29.9 20.8 19.3
- b. GROVE - PT (Ours) 14.3 50.6 27.0 22.5
- c. GROVE - PT+FT (Ours) 21.4 83.5 31.7 26.2

All

- d. Automatic annotation 13.8 40.0 27.1 20.4
- e. GROVE - PT (Ours) 14.3 50.6 33.6 24.3
- f. GROVE - FT (Ours) 21.0 77.7 15.8 18.1
- g. GROVE - PT+FT (Ours) 21.4 83.5 40.0 28.7

Center

- Table 2. Grounded video caption generation on manuallyannotated iGround test set. Pre-training on our new large-scale HowToGround1M dataset followed by finetuning on manuallyannotated iGround training data (PT+FT) clearly outperforms pretraining only (PT) and finetuning only (FT) as well as the GLaMM baseline [31] (a.) and directly applying automatic annotation (d.). We show center frame (“Center”) and all frame (“All”) evaluation.

and predicted bounding boxes as well as (ii) the similarity of embeddings of GT and predicted noun phrases that correspond to bounding boxes. The aim of the recall metric is to assess the rate of positive predictions. A prediction is considered positive if both the bounding box IoU and the noun phrase similarity are above a certain threshold. We propose a video-level evaluation setting for AP50 and recall for the grounded video caption generation task where the metrics are calculated per video and averaged across videos. ActivityNet-Entities. We follow [50] and report F1all, F1all per sent, F1loc, and F1loc per sent. In F1all, a region prediction is considered correct if the associated noun phrase is both correctly predicted (exact match) and correctly localised (IoU > 0.5). F1loc considers only localisation accuracy ignoring errors in the generated noun phrases. In these metrics, accuracies are averaged across noun categories while in F1all per sent and F1loc per sent accuracies are averaged across sentences.

VidSTG. We follow [14, 36, 43, 51] and assume that the videos are temporally segmented to the events of interest (using the available start/end timestamps) and report msIoU, defined as the average IoU across frames, between the predicted and ground truth bounding boxes for the target event. Implementation details. All implementation details including architectural choices of the GROVE model as well as training and inference details can be found in appendix E.

#### 6.2. Comparison with the state of the art

iGround. The results on our human-annotated iGround test set are shown in Table 2. We compare the results of the proposed GROVE model with our automatic annotation method (described in Section 3) and (still image) GLaMM [31]. The automatic annotation method is a natural fit for a baseline for this task as it performs image-based grounded captioning followed by video-level aggregation without any training. Comparing with [31] aims to assess the benefits of our method in comparison to still-image grounding. Since [31]

Method FT msIoU PG-V-L (13B) [25] ✗ 35.1 GLaMM [31]

Method msIoU

STVGBert [36] 47.3 TubeDETR [43] 59.0 STCAT [14] 61.7 DenseVOC [51] 61.9 GROVE FT (Ours) 61.3 GROVE PT+FT (Ours) 63.7

+ SAM2 [32] ✗ 38.6 GROVE ✗ 43.0

VideoGLaMM[26] ✓ 39.7 GROVE ✓ 55.5

Table 3. State-of-the-art comparison of spatial grounding on the VidSTG [48] test set (declarative sentences). All models use ground truth temporal localization. Large-scale pretraining (PT+FT) results in an improvement over fine-tuning only (FT) for our model GROVE.

Table 4. State-of-the-art comparison of spatial grounding on the VidSTG [48] test set (interrogative sentences). All models use ground truth temporal localization. GROVE outperforms all competitors both in a pre-training only setting (✗) and when fine-tuned on VidSTG (✓).

Method F1all F1all per sent F1loc F1loc per sent

GVD [50] 07.10 17.30 23.80 59.20 GROVE FT (Ours) 09.51 21.15 30.96 68.79 GROVE PT+FT (Ours) 13.39 24.08 45.04 77.29

Table 5. Results on the validation set of ActivityNet-Entities [50]. Large-scale pretraining (PT+FT) results in an improvement over fine-tuning only (FT) for our model GROVE.

runs per frame, it does not provide a video-level caption and noun phrases; these differ across frames (see Sec. 3). Thus, we use predictions of [31] for the center frame of each video. We call this the “Center” set-up in Table 2. The “All” set-up considers video-level caption and all frames of the video for bounding box localisation. GROVE, pre-trained on our automatically annotated HowToGround1M dataset (b.), significantly outperforms the [31] baseline (a.) with an improvement of 20 points in CIDEr and 6.2 points in AP50. This demonstrates that temporal context is crucial for this task. Results further improve with finetuning (c., PT+FT). In the all-frame evaluation, the GROVE model pretrained on the automatically annotated HowToGround1M dataset (e.) provides significant performance improvements over directly obtaining predictions on the test set using the automatic annotation method (d.). This indicates that the GROVE model can correct (smooth out) some of the noise in the automatic annotations during the large-scale training. We obtain further major performance improvements by fine-tuning GROVE on our manually-annotated iGround dataset, showing that large-scale pre-training with automatic labels followed by fine-tuning on a small-scale but highly accurate dataset (PT+FT, c. and g.) is a good recipe for this task. Finally, the underperformance of GROVE that is directly fine-tuned on iGround without pre-training (FT, f.) provides further evidence of the importance of pre-training, particularly when the fine-tuning dataset is small.

ActivityNet-Entities and VidSTG. We adapt the GROVE model for the spatio-temporal video grounding task on

[Figure 4]

- Figure 4. Qualitative examples showing predictions of our GROVE model. Please note that GROVE is able to: (i) produce video-level natural language captions describing the main action in the video; (ii) ground multiple objects; and (iii) produce spatio-temporally consistent bounding box predictions. Please note that the second row shows an example of model prediction that is partly incorrect as the blue box, while temporally consistent, does not depict a “yarn”. More results including failure modes are discussed in appendix A.

1k 10k 100k 1M

32%47%63%78%94%

CIDEr

1k 10k 100k 1M

Pre-training dataset size

25%29%34%38%43%

AP50

1k 10k 100k 1M

20%22%24%26%28%

Recall

PT

PT+FT

- Figure 5. Results after pre-training (PT) vs. after fine-tuning and pre-training (PT+FT) as a function of the pre-training dataset size. Results are reported on the iGround validation set.

the ActivityNet-Entities and VidSTG datasets. The details are in appendix E. Tables 3, 4 and 5 show the comparison of GROVE with published results on VidSTG and ActivityNet-Entities. On VidSTG declarative sentences (Table 3), GROVE achieves the best performance despite not being designed for this task. On VidSTG interrogative sentences, (Table 4), GROVE outperforms all previous work, both when it is fine-tuned on VidSTG (✓), but also when it is only pre-trained on HowToGround1M pre-training set (✗). On ActivityNet-Entities (Table 5), GROVE significantly outperforms the GVD baseline, demonstrating its effectiveness even with sparse annotations.

Other datasets. In Table 8, we evaluate GROVE on YouCook-Interactions [37] and GroundingYouTube [4], outperforming the previous SOTA by large margins.

Qualitative Results. We show qualitative results of the GROVE model on several example videos in Figures 1 and

- 4. More qualitative results are in appendix A.

#### 6.3. Effects of pre-training dataset size

We study the scaling behaviour of the GROVE model by varying the size of the pre-training data. Specifically, we pre-train GROVE on 1k, 10k, 100k and 1M videos sampled from our HowToGround1M dataset. The performance is measured on the iGround validation set, where we report CIDEr, AP50 and recall. The results are shown in Figure 5. For each metric, we report the performance of the pre-trained model for different amounts of pre-training data, but also the performance of the fine-tuned model when initialised from the pre-trained models and fine-tuned on the manually annotated iGround training set. The results show that as we scale the pre-training dataset, both the pre-trained and the fine-tuned models continue to improve consistently across all metrics. This finding is important, as it verifies that obtaining automatic annotations at a large scale is beneficial for pretraining. It also signifies that fine-tuning using small-scale but high-quality data is most efficient when the model is

Unfreeze AD METEOR CIDEr AP50 Recall

✓ ✓ 19.7 92.6 42.0 26.9 ✓ ✗ 19.7 88.9 39.2 26.4 ✗ ✓ 19.2 82.2 36.8 25.9

- Table 6. Ablation of spatio-temporal adapters (AD) and unfreezing the bounding box decoder and projection layers (unfreeze). We report results on the iGround validation set.

0.0 0.1 0.2 0.3 0.4 0.5 Temporal objectness threshold

39.0%40.0%41.0%42.0% AP50

AP50

26.0%27.0%28.0%29.0%

Recall

Recall

Figure 6. Benefits of temporal objectness. AP50 (left) and recall (right) of our model for different temporal objectness thresholds. Results are reported on the iGround validation set.

Auto. annotation METEOR CIDER AP50 Recall Average

- a. Proposed 12.3 31.7 26.9 19.3 22.5
- b. Alt. Stage 1 (F) 12.4 36.8 20.1 14.5 20.9
- c. Alt. Stage 1 (V) 16.8 23.1 14.6 14.2 17.2
- d. Alt. Stage 2 11.9 28.7 27.8 17.6 21.5
- e. Alt. Stage 3 12.3 31.7 23.2 16.0 20.8

- Table 7. Comparison of our automatic annotation approach (row a.) vs. several alternatives (rows b.-e.) on the iGround validation set. See text for details.

pre-trained at the largest scale.

Benefits of pre-training. The benefits of large-scale pretraining on the HowToGround1M dataset are clearly demonstrated on three different tasks and datasets, as shown in Table 2 (iGround), Table 3 (VidSTG) and Table 5 (ActivityNet-Entities), where the model pre-trained on the HowToGround1M dataset and finetuned on each specific dataset reaches state-of-the-art results.

#### 6.4. Ablation analysis

Here we provide the ablation of the key components of the GROVE model as well as evaluate variants of the automatic annotation method.

Model ablations. We ablate the key components of our GROVE model. We report results on the iGround validation set after pre-training with fine-tuning, which achieves the best results. In Table 6, we ablate the spatio-temporal adapters and the training (unfreezing) of the bounding box decoder and projection layers (VL and LQ in Fig. 3). To mitigate overfitting, we keep the visual backbones and LLM

frozen, while training the embedding and output layers of the LLM to accommodate the modified vocabulary.

Next, we assess the importance of the temporal objectness head. Figure 6 shows the performance of the GROVE model in AP50 and recall when varying the threshold for the temporal objectness from 0.0 (completely removing the temporal objectness head) to 0.5. This threshold regulates the sensitivity of the model to detecting whether an object leaves the frame (or is occluded). These results demonstrate that by increasing the temporal objectness threshold we obtain substantial benefits in AP50 while sacrificing only a negligible amount of recall. This demonstrates the importance of modelling objects that temporally leave the frame.

Ablations of automatic annotation. We ablate each stage of our automatic annotation method and show results in Table 7. In “b. Alt. Stage 1 (F)”, we replace our Stage 1 with the GIT [41] image captioner + Llama-3 [9] noun phrase extractor + OWLv2 [24] object detector, which yields slightly crisper captions but weaker grounding because the captioner and detector are not trained jointly; a video-level variant (VideoLlama-3 [45] instead of GIT), row “c. Alt. Stage 1 (V)”, shows the same pattern. Feeding full captions in Stage 2 (row “d. Alt. Stage 2”) instead of using subject-verbobject triplets (row “a. Proposed”) raises AP50: full-captions yield fewer predictions as the LLM trims its output to the most salient objects, reducing recall but improving precision– hence the slightly higher AP50. Adding CoTracker3 in Stage 3 (row “e. Alt. Stage 3”) further degrades grounding due to tracker drift during viewpoint changes. Our proposed automatic annotation method (row “a. Proposed”) scores the best on average; a more detailed analysis is in appendix B.

### 7. Conclusion

We introduced a large-scale automatic annotation method to generate densely grounded captions in video, which we used to construct the large-scale HowToGround1M video dataset. We developed GROVE, a new model for grounded video captioning, which we pre-trained on this large-scale data. To enable rigorous evaluation, we introduced iGround, a dataset with high-quality manual annotations, which also serves as a fine-tuning resource. Our experiments demonstrate the effectiveness of our pre-training strategy, the importance of scaling the pre-training dataset as well as the benefits of fine-tuning on smaller, high-quality data. Our approach not only sets the state of the art on the new manually annotated iGround dataset but also outperforms existing methods on the VidSTG, ActivityNet-Entities, GroundingYoutube and YouCook-Interactions benchmarks. We believe this work provides a strong foundation for future research in grounded video captioning.

Acknowledgments. This work was supported by the Ministry of Education, Youth and Sports of the Czech

Republic through the e-INFRA CZ (ID:90140). This research also received the support of projects EXA4MIND, ELLIOT, CLARA and ERC FRONTIER, funded by the European Union’s Horizon Europe Research and Innovation Programme, under Grant Agreements N° 101092944, N° 101214398, N° 101136607 and N° 101097822. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. Furthermore, this work was funded in part by the French government under management of Agence Nationale de la Recherche as part of the “France 2030” program, reference ANR-23-IACL-0008 (PR[AI]RIE-PSAI projet), and the ANR project VideoPredict (ANR-21FAI1-0002-01). Cordelia Schmid would like to acknowledge the support by the K¨orber European Science Prize.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Bi´nkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Kar´en Simonyan. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 5
- [2] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 2005. 5
- [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. 16
- [4] Brian Chen, Nina Shvetsova, Andrew Rouditchenko, Daniel Kondermann, Samuel Thomas, Shih-Fu Chang, Rogerio Feris, James Glass, and Hilde Kuehne. What when and where? selfsupervised spatio-temporal grounding in untrimmed multiaction videos from narrated instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 7, 12
- [5] Brian Chen, Nina Shvetsova, Andrew Rouditchenko, Daniel Kondermann, Samuel Thomas, Shih-Fu Chang, Rogerio Feris, James Glass, and Hilde Kuehne. What, when, and where?

– self-supervised spatio-temporal grounding in untrimmed multi-action videos from narrated instructions. arXiv preprint arXiv:2303.16990, 2024. 1, 2

- [6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195,

2023. 1, 2

- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao

- Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 17
- [8] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen. Epic-kitchens visor benchmark: Video segmentations and object relations. In Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks, 2022. 1
- [9] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 8, 12

- [10] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, and et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
- [11] Xin Gu, Heng Fan, Yan Huang, Tiejian Luo, and Libo Zhang. Context-guided spatio-temporal video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [12] De-An Huang, Shyamal Buch, Lucio Dery, Animesh Garg, Li Fei-Fei, and Juan Carlos Niebles. Finding “it”: Weaklysupervised reference-aware visual grounding in instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [13] Jingwei Ji, Ranjay Krishna, Li Fei-Fei, and Juan Carlos Niebles. Action genome: Actions as compositions of spatiotemporal scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2020. 1

- [14] Yang Jin, Yongzhi Li, Zehuan Yuan, and Yadong Mu. Embracing consistency: A one-stage approach for spatio-temporal video grounding. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 6
- [15] N. Karaev, I. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. arXiv preprint arXiv:2410.11831, 2024. 4, 12
- [16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 5, 17
- [17] Zongmian Li, Jiri Sedlar, Justin Carpentier, Ivan Laptev, Nicolas Mansard, and Josef Sivic. Estimating 3d motion and forces of human-object interactions from internet videos. International Journal of Computer Vision (IJCV), 130(2):363–383,

2022. 1

- [18] Zihang Lin, Chaolei Tan, Jian-Fang Hu, Zhi Jin, Tiancai Ye, and Wei-Shi Zheng. Collaborative static and dynamic vision-language streams for spatio-temporal video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [19] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi.

- Hoi4d: A 4d egocentric dataset for category-level humanobject interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
- [20] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. arXiv preprint arXiv:2404.13013, 2024. 1, 2
- [21] Chih-Yao Ma, Yannis Kalantidis, Ghassan AlRegib, Peter Vajda, Marcus Rohrbach, and Zsolt Kira. Learning to generate grounded visual captions without localization supervision. In Proceedings of the European Conference on Computer Vision (ECCV), page 353–370, 2020. 1
- [22] Robert McCarthy, Daniel CH Tan, Dominik Schmidt, Fernando Acero, Nathan Herr, Yilun Du, Thomas G Thuruthel, and Zhibin Li. Towards generalist robot learning from internet video: A survey. arXiv preprint arXiv:2404.19664, 2024. 1
- [23] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 2, 3, 4, 5
- [24] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. arXiv preprint arXiv:2306.09683, 2024. 3, 8, 12
- [25] Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, and Fahad Khan. Pg-video-llava: Pixel grounding large videolanguage models. arXiv preprint arXiv:2311.13435, 2023. 6
- [26] Shehan Munasinghe, Hanan Gani, Wenqi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, and Salman Khan. Videoglamm : A large multimodal model for pixel-level visual grounding in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference (CVPR),

2025. 3, 6

- [27] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2024. 2
- [28] Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik. Learning to imitate object interactions from internet videos. arXiv preprint arXiv:2211.13225, 2022. 1
- [29] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 1, 2
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), 2021. 17
- [31] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan.

- GLaMM: Pixel grounding large multimodal model. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3, 4, 5, 6, 12, 17
- [32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. 6
- [33] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 16
- [34] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2018. 1
- [35] Nina Shvetsova, Anna Kukleva, Xudong Hong, Christian Rupprecht, Bernt Schiele, and Hilde Kuehne. Howtocaption: Prompting llms to transform video annotations at scale. arXiv preprint arXiv:2310.04900, 2023. 3, 4
- [36] Rui Su, Qian Yu, and Dong Xu. STVGBert: A visuallinguistic transformer based framework for spatio-temporal video grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2, 6
- [37] Reuben Tan, Bryan A. Plummer, Kate Saenko, Hailin Jin, and Bryan Russell. Look at what im doing: Self-supervised spatial grounding of narrations in instructional videos. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 1, 2, 7, 12
- [38] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatiotemporal video grounding with visual transformers. IEEE Transactions on Circuits and Systems for Video Technology, 32(12):8238–8249, 2021. 1, 2
- [39] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [40] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 5
- [41] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. GIT: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022. 3, 8, 12
- [42] Syed Talal Wasim, Muzammal Naseer, Salman Khan, MingHsuan Yang, and Fahad Shahbaz Khan. Videogrounding-dino: Towards open-vocabulary spatio-temporal video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [43] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. TubeDETR: Spatio-temporal video ground-

- ing with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 6
- [44] Mihai Zanfir, Elisabeta Marinoiu, and Cristian Sminchisescu. Spatio-temporal attention models for grounded video captioning. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. 1
- [45] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 8, 12
- [46] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models. arXiv preprint arXiv:2312.02949, 2023. 1, 2
- [47] Zhu Zhang, Zhou Zhao, Zhijie Lin, Baoxing Huai, and Jing Yuan. Object-aware multi-branch relation networks for spatiotemporal video grounding. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2020. 2
- [48] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. Where does it exist: Spatio-temporal video grounding for multi-form sentences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1, 2, 5, 6, 17
- [49] Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. BuboGPT: Enabling visual grounding in multi-modal llms. arXiv preprint arXiv:2307.08581,

2023. 1, 2

- [50] Luowei Zhou, Yannis Kalantidis, Xinlei Chen, Jason J Corso, and Marcus Rohrbach. Grounded video description. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1, 2, 3, 5, 6, 17
- [51] X. Zhou, A. Arnab, C. Sun, and C. Schmid. Dense video object captioning from disjoint supervision. arXiv preprint arXiv:2306.11729, 2023. 6
- [52] Kateryna Zorina, Justin Carpentier, Josef Sivic, and Vladim´ır Petr´ık. Learning to manipulate tools by aligning simulation to video demonstration. IEEE Robotics and Automation Letters, 7(1):438–445, 2021. 1

### Appendix

To complement the main paper, this appendix assembles additional results, analyses, and implementation details. Sections A and B provide additional qualitative visualizations and expanded quantitative metrics, respectively. Section C discusses the model and data limitations. Section D reports comprehensive dataset statistics. Section E details the GROVE architecture and training setup, while Section F describes our automatic annotation pipeline. Section G outlines the human-annotation protocol, and Section H lists the exact prompts used to curate spatio-temporally grounded captions.

### A. Additional qualitative results

Figures 8 and 9 show qualitative results of our GROVE model (Section 4), pre-trained on the HowToGround1M dataset and finetuned on the iGround training set (2013 examples). The results are shown on the iGround test set. In the figures’ captions we discuss some of the benefits of our model. Additional qualitative results showcasing the predictions of our approach overlaid over the input videos are shown in the supplementary video (available at https://ekazakos.github.io/grounded_ video_caption_generation/). Figure 10 shows the main failure modes of our model.

### B. Additional quantitative results

Detailed analysis for the ablations of automatic annotation. We replace each stage of our automatic annotation method with an alternative. Results are shown in Table 7. In Stage 1, we replace the still-image model [31] with an alternative still-image grounded caption generation method. This approach leverages GIT [41] for frame-level captioning, Llama3 [9] for extracting noun phrases from the caption, and OWLv2 [24] for their bounding box localisation within each frame. We call this alternative “b. Alt. Stage 1 (F)”. We also evaluate a video-level variant “c. Alt. Stage 1 (V)”, where we replace the GIT captioner with VideoLlama3 [45]. To ablate Stage 2 (“d. Alt. Stage 2”), we provide the LLM with full captions from Stage 1 instead of extracting SubjectVerb-Object triplets from the caption to assess the impact of additional context. To ablate Stage 3 (“e. Alt. Stage 3”), we incorporate CoTracker3 [15], a SOTA visual point tracking method to provide temporal association of bounding boxes across frames. Using 5 uniformly sampled frames and their bounding box predictions from Stage 1, we track objects in between with CoTracker3 and associate the resulting tracks with noun phrases from the caption.

Results are reported in Table 7, where we compare the alternative automatic annotation methods on the iGround validation set. The frame-level alternative Stage 1 (row b.) performs better in captioning due to GIT’s superior perfor-

mance but performs noticeably worse for grounding. This is because our Stage 1 still-image grounding model [31] is explicitly trained for grounding, unlike GIT, Llama3, and OWLv2, which are not trained jointly and may underperform due to various factors–such as Llama3 extracting nongroundable noun phrases or OWLv2 missing objects. A similar trend is observed for the video-level alternative Stage 1 (row c.). Compared to our proposed method, the alternative Stage 2 (row d.) underperforms across all metrics except AP50. This is because the full-caption input yields fewer predictions as the LLM trims its output to the most salient objects, reducing recall but improving precision–hence the slightly higher AP50. In contrast, the SVO-based input in our proposed automatic annotation method leads to slightly longer captions (12 vs.11 words) with more noun phrases (3.3 vs. 3.0), leading to more object predictions and higher recall. This reflects a typical precision-recall trade-off. The alternative Stage 3 (row e.) underperforms in grounding due to tracker drift caused by abrupt viewpoint changes. Overall, on average (the last column in Table 7), our proposed method achieves the best performance.

Comparison with the state of the art on YouCookInteractions and GroundgYouTube datasets. In Table 8, we evaluate GROVE on YouCookInteractions and GroundingYouTube datasets, outperforming the previous SOTA by large margins.

Method YouCook-Interactions GroundingYouTube

What When and Where (S3D) [4] 53.98 60.62 What When and Where (CLIP) [4] 58.35 56.98 GROVE 68.67 72.14

Table 8. Comparison with SOTA on YouCook-Interactions [37] and GroundgYouTube[4] datasets.

### C. Limitations

Although our proposed datasets and model advance the state of the art in grounded video captioning and spatio-temporal sentence grounding, they also reveal avenues for future exploration, stemming from the following limitations.

Scaling to long videos. Despite achieving state-of-the-art results on VidSTG by running inference in a sliding-window manner over videos up to three minutes, the training phase remains memory-bound: we can supply the model with only eight frames per clip. This is sufficient for the short clips in HowToGround1M and iGround (8-10 seconds), where eight frames corresponds to about 1 fps sampling. For VidSTG’s much longer videos, however, uniform sampling of only eight frames introduces large gaps between frames and prevents the model from seeing fine-grained temporal dependencies during training. Closing this discrepancy will require methods that can train directly on larger frame spans or more efficient representations of extended temporal context (e.g.

Method METEOR CIDER AP50 Recall

Auto. annotation 12.3 31.7 26.9 19.3 GROVE 19.7 92.6 42.0 26.9

All

Auto. annotation 08.1 07.3 22.3 14.7 GROVE 14.4 41.3 36.0 18.9

Hard

- Table 9. Results of our automatic annotation method (Auto. annotation) and the complete proposed model (GROVE) on the entire iGround validation set (All) and for a subset (about 10% of data) with challenging similar referring expressions (Hard).

memory).

Complex referring expressions. We examined the iGround validation set and discovered that roughly 10% of its videos contain more than one object whose referring expressions (and appearance) are highly similar. We designate this challenging portion of the data as the “Hard” subset. In Table 9, we compare GROVE with our automatic annotation method in this subset. Although GROVE still surpasses the automatic annotation method on this subset, the marked drop in performance of both methods on the “Hard” subset (comparing to “All”, i.e. the full validation set) reveals that reliably disambiguating closely related referring expressions remains still a challenge. These results suggest opportunities to refine both the model architecture and the automatic annotation method to better handle such fine-grained cases.

- D. Dataset Statistics

Table 10 reports the statistics of both the HowToGround1M pre-training dataset and the iGround manually annotated set. Word clouds of the natural language descriptions from those datasets are shown in Figure 7.

Statistic HowToGround1M iGround Avg num frames per video 44.6 40.1 Avg duration (seconds) 7.9 8.0 Avg num instances per video 80.1 118.1 Total num instances 80,092,775 421,588 Avg box width × height 243.7 × 172.6 174.9 × 135.5 Avg tube length (frames) 6.4 29.0 Avg caption length (words) 12.1 15.4

Table 10. Statistics of HowToGround1M and iGround datasets.

- E. Details of the GROVE model

[Figure 5]

- (a)

[Figure 6]

- (b)

Figure 7. Word cloud for (a) HowToGround1M dataset and (b) iGround dataset.

Model architecture. Figure 3 shows the different components of our approach. The Global Video Encoder, Ve(·), outputs video features, oe, which are pooled spatio-temporally, resulting in the video prompts. These are projected to a language embedding space with V L(·). The LLM, LM(·), ingests a multimodal prompt consisting of video and language tokens. The LLM is prompted to generate a caption

for the video by tagging the noun phrases that correspond to objects and appending them with detection tokens (shown with red and green in the LLM’s generated caption in Figure 3). The LLM’s output hidden states that correspond to the generated caption are projected to queries (using LQ(·)). The queries corresponding to the detection tokens are fed to the bounding box decoder D(·). The Grounding Video Encoder, Vg(·), outputs fine-grained video features, which are also fed to the decoder. The decoder performs crossattention frame-wise between the queries and the outputs of Vg(·), og, which are used as keys/values. Finally, the prediction heads output bounding box predictions and temporal objectness scores for each object at each frame. This objectness score is used to predict the presence/absence of the object in each video frame and is of major importance for the grounded video caption generation task. Details about the visual backbones Ve(·) and Vg(·) as well as details about the LLM LM(·) including the format of its multimodal inputs and its vocabulary are given next.

Projection layers. We project the outputs of the Global Video Encoder and the output hidden states of the LLM with MLPs, op′ = V L(op) and oq = LQ(ol), where V L(·) projects the visual features to an embedded language space,

while LQ(·) projects the LLM’s hidden states to queries. op′ is the LLM’s visual input while oq is input to the bounding box decoder that is described next.

Backbones. GROVE consists of two video encoders and a

[Figure 7]

- Figure 8. Qualitative results of our GROVE model on the (unseen) iGround test set. The colour-coded sentence fragments are spatiotemporally localised in the video with the bounding boxes colour coded with the same colour. The results demonstrate that: (i) our model can localise even small objects such as a pen or a tooth brush; (ii) objects are consistently labelled across frames despite changes of viewpoint or scale; (iii) the model focuses on the human and the interacted objects; (iv) the model can successfully ground multiple objects in the video. Additional results are shown in the supplementary video (available at https://ekazakos.github.io/grounded_video_ caption_generation/). 14

[Figure 8]

- Figure 9. Additional qualitative results of our GROVE model on the (unseen) iGround test set. The colour-coded sentence fragments are spatio-temporally localised in the video with the bounding boxes colour coded with the same colour. In addition to the model’s properties discussed in Fig. 8, GROVE is capable of predicting whether an object is present in a certain frame via the temporal objectness head; in the second example there are no bounding box predictions for the hand in the first three frames while in the fourth example there are no predictions for the hand and the screwdriver in the second and fifth frame. Additional results are shown in the supplementary video (available at https://ekazakos.github.io/grounded_video_caption_generation/).

[Figure 9]

- Figure 10. Qualitative results for the main failure modes of our GROVE model on the (unseen) iGround test set. The colour-coded sentence fragments are spatio-temporally localised in the video with the bounding boxes colour coded with the same colour. We identify four main failure modes: (i) temporal objectness mispredicts the presence of an object (first row, last frame for the knife), (ii) inaccurate predictions of object location (second row, third and last frames for the spatula), (iii) misclassification of object (third row, model predicts “radish” for the pumpkin), and (iv) misclassification of action (last example, model predicts “watering” for planting).

multimodal LLM as its main backbones. The Global Video Encoder Ve(·), takes as input a video v ∈ RT×H1×W1 and produces an output oe ∈ RT×

p ×Wp1, where p is the patch size of the underlying visual transformer. Its purpose is to provide a holistic representation of the video that will be ingested by the LLM. The Grounding Video Encoder Vg(·), takes as input a video v ∈ RT×H2×W2, where W2 > W1 and H2 > H1. It produces og ∈ RT×

###### H1

p ×Wp2. og is used to ground phrases from the caption to the visual content, which is performed by the bounding box decoder that is described later. The input video to the Grounding Video Encoder is of larger spatial resolution than that of the Global Video Encoder for enhanced localisation capability. Finally, the LLM LM(·) takes as input a multimodal sequence s ∈ RL×D and produces an output ol of the same size. Its input is of the form The <video> provides an overview of the video. Could you please give me a description of the video? Please respond with interleaved bounding boxes for the corresponding parts of the answer. <video> is replaced by the output of Ve(·), and therefore the LLM ingests mixed language and visual tokens. We also augment the LLM’s

###### H2

vocabulary with a detection token <DET>, prompting the model to generate responses with <DET> tokens by the phrases that correspond to objects to be detected in the video.

Loss function. Our loss function is a combination of a language modelling loss and losses relevant to video object detection. The language modelling loss is a Cross-Entropy loss applied on ol. For object detection, we follow DETR [3] and use a gIoU loss [33] and an L1 loss applied on pbb. Different than [3], the losses are applied per frame and summed over frames. Moreover, the losses are applied only to the objects that appear in the frame (rather than each object in the caption) using the ground-truth temporal objectness scores. The representation that we use for the bounding boxes is [x,y,w,h] and their coordinates are normalised with the dimensions of the video. Finally, we employ a binary cross-entropy loss on ptobj. Our loss is, hence, defined

as:

LLM = CE(ol) (1) LgIoU = gIoU(pbb,gtbb) (2)

LL1 = L1(pbb,gtbb) (3) Ltobj = BCE(ptobj,gttobj) (4)

L = λLM × LLM + λgIoU × LgIoU (5) + λL1 × LL1 + λtobj × Ltobj, (6)

where gtbb are the ground truth boxes and gttobj are the ground truth objectness scores and λ are the weights for the losses.

Training/inference. We realise the Global Video Encoder Ve(·) with a CLIP-L [30] model with an input of 336×336 and a patch size of 14. The Grounding Video Encoder Vg(·) is instantiated with a SAM [16] encoder and the bounding box decoder D(·) is a SAM-based decoder, the same as in GLaMM [31]. The LLM LM(·) is a Vicuna-7B model [7]. During training we keep Ve(·), Vg(·) and LM(·) frozen. Vg(·) originally takes as input 1024× 1024 images. As this is too large to fit in memory for videos, we instead use 512×512 video spatial resolution, while we interpolate the positional encodings of Vg(·) and fine-tune them. Adapters are 3D spatiotemporal convolutional layers with a kernel of size 3 × 3 × 3 and a stride of 1. We apply adapters to every 3 layers of Ve(·) and to all global attention layers of Vg(·). The bounding box head hbb is an MLP with two FC layers and a ReLU activation function in between, while the temporal objectness head htobj is a linear layer. Both prediction heads employ a sigmoid activation function. We apply a threshold of 0.5 to the temporal objectness scores. Both the adapters and the prediction heads are randomly initialised. We use T = 8 frames for the videos during both training and testing. During training we perform random sparse sampling of frames by splitting the video in 8 segments and randomly drawing a frame from each segment while during testing we pick the centre frame of each segment.

We train GROVE for 20 epochs using a batch size of 128. We use a learning rate of 5 × 10−5 with warmup for the first 100 training steps and linearly decay the learning rate for the rest of training. We do not apply any weight decay or spatial data augmentation. We use λLM = 1,λgIoU = λL1 = λtobj = 2.

Details of VidSTG and ActivityNet-Entities experiments. For VidSTG [48] and ActivityNet-Entities [50], we do not use the temporal objectness head. That is because in VidSTG the spatio-temporal tubes are continuous within the segments’ boundaries, while ActivityNet-Entities provides annotations for a single frame per object and in the rest of the frames the objects might still be present but without annotation, and thus should not be modelled as absent. As the task

in VidSTG entails predicting the spatio-temporal bounding boxes given a short description, we provide the short descriptions as input to our GROVE model during both training and inference in a teacher-forcing setup. For evaluating GROVE on VidSTG without observing any VidSTG data during training (GROVE with FT: ✗, Table 4), we pre-train GROVE on HowToGround1M. Each HowToGround1M caption is rewritten–by prompting Llama-3–into both of VidSTG’s sentence styles, declarative and interrogative. Every transformed sentence is then paired with a single bounding box per frame, chosen as the box of the first subject or object it mentions. This supervision reshapes HowToGround1M’s annotation distribution to mirror VidSTG’s, allowing GROVE to achieve strong performance without relying on any VidSTG training data.

### F. Details of the automatic annotation method

Multiple people in the video. Our automatic annotation method can handle multiple subjects in a video as long as one of the two following conditions are met: a) the subjects are described with a distinct language, e.g. ‘man with green jumper’ and ‘man with blue shirt’, or b) the subjects are within a Subject-Verb-Object relationship even when described with the same terms, e.g. (‘person’, ‘dances’, ‘with’, ‘person’) which would produce ‘A person dances with another person’. If neither conditions are met, the caption aggregation (Stage 2) may merge the two subjects into one. Association of verbs and objects is naturally performed through the Subject-Verb-Object triplets. For example, given two relationships: (‘man’, ‘cuts’, ‘onions’) and (‘woman’, ‘stirs’, ‘food’, ‘in’, ‘pot’). The LLM-based caption aggregation step (Stage 2) has sufficient information to associate the man with the action of cutting the onions and the woman with stirring the food.

Additional details of Stage 3. We provide additional details of the procedure of Stage 3 using the example from Figure 2, right. The object in the woman’s hands is described as ‘a green beverage’ and ‘a glass of green liquid’ across different frames. Stage 2 has provided the video-level noun phrases ‘a woman’ and ‘a beverage’. Stage 3 is formulated as a classification problem where each one of ‘a green beverage’ and ‘a glass of green liquid’ are the inputs to be classified in one of the classes {‘a woman’, ‘a beverage’, ∅} and thus associated with the right bounding box. The class ∅ represents the “None” class, i.e. when an input does not belong to any of the known classes and it is useful for noisy inputs.

### G. Protocol for human annotations

In Figure 11, we describe the annotation guidelines for annotating the training/validation/test sets of the iGround dataset.

The annotation criteria have been extensively discussed with the annotation provider and the annotators have been

|Annotation Guidelines:<br><br>1. Video Selection:<br><br>• You will be provided with a larger set of videos than needed.<br>• Your first task is to select clips that are considered ‘interesting’ based on criteria that will be discussed further. An ‘interesting’ video typically includes dynamic events or actions that are clear and distinguishable despite the low video quality. In those events/actions people usually interact with objects, e.g. ‘A man is cutting an onion using a knife’. ‘Non-interesting’ events are typically static, e.g. a person simply standing/sitting and talking. Non-interesting events are also events with ambiguous actions taking place, i.e. generic/abstract actions that cannot be described concisely or actions for which the annotator is unsure about what is happening in the video.<br><br><br>2. Video Annotation:<br><br><br>• For each selected video clip, write a concise, one-sentence description of the main event taking place in the clip. If the action is too complex, use at most two sentences for describing it, but prioritise one-sentence descriptions.<br>• Focus only on the objects that humans interact with rather than describing densely every object in the scene.<br>• To enrich the language descriptions, also describe properties of objects such as color, shape, etc, e.g. ‘blue cup’ or ‘red onion’. It is not strictly necessary to always describe the object’s property but only when deemed important by the annotator.<br>• When you are unsure about the object being used, you can simply describe it as ‘object’. If object is unknown but the category of the object is known, please describe the object using its category, e.g. ‘food’.<br>• When there are two or more humans in the scene, use one of their characteristics to distinguish them, e.g. ‘the woman in the red shirt standing next to the woman in the green shirt is putting a strawberry on a cocktail glass’.<br>• If there are multiple actions happening consecutively, describe all of them and their associated objects. E.g. ‘a person is doing action-1 using object-1, then doing action-2 with an object-2’. As shown in the example, you can use ‘then’ for connecting temporally adjacent actions.<br>• Provide bounding boxes for humans/different objects mentioned in your description. These bounding boxes should be applied to all frames where the objects are visible.<br>• Label each bounding box with a short phrase directly from your sentence description (e.g., ‘a brown dog’, ‘person´s hands’).<br>• It is not necessary that each object appears in each frame of the video. For example, a person might be using a tool, then leaving it down and using another tool. In this case, you would annotate with bounding boxes the first tool for the first half of the video and the second tool for the second half. Another common case is that objects or the person might disappear and then reappear. In this case, again all instances of the object must be annotated, so you should be careful about objects leaving the scene as they might enter the scene again later.<br>• If there are many small objects, e.g. mushrooms in a pan, use a single bounding box labelled as ‘mushrooms’.<br>• There are cases where two or more bounding boxes are needed for objects of the same type: a) one bounding box for each human hand when both are used to perform an action, b) one bounding box for each tool/container/appliance etc of the same type that the human is using, e.g. when they are placing food in two dishes, or pouring the content of a shaker in two cocktail glasses.<br>• Descriptions: Must be accurate and written in fluent English. Suitable for either native speakers or highly proficient English speakers.<br>• Bounding Boxes: Ensure that bounding boxes accurately encompass the objects for the entirety of their visibility within the clip. The bounding boxes should be consistent and smooth across frames, maintaining size and position as closely as possible given the movement of the object and video quality. An exception is when there are abrupt viewpoint changes of the camera, which might result in objects abruptly changing position and size across neighbouring frames.<br>|
|---|

Figure 11. Annotation guidelines for the manually annotated iGround dataset.

trained based on those criteria prior to commencing the annotation process. We have also performed a pilot annotation project with the annotation provider on 10 video clips with several rounds of careful checking and feedback. Moreover, the annotation provider performed regular quality reviews on the annotations to ensure that the annotation criteria have been met.

### H. Prompts for automatic curation of spatiotemporally grounded captions

The full prompt for the Stage 2 (Video-level caption aggregation) of our automatic annotation approach (Section 3) is shown in Figure 12 and the full prompt for Stage 3 (Temporally consistent bounding box annotation) in Figure 13.

###### System Instructions

Generate a dynamic, video-level description based on frame-level inputs. The inputs include actions performed in individual frames in the form of Subject-Verb-Object (SVO) triplets along with prepositions and prepositional objects. The SVO triplets describe how actions are performed and how objects interact. Your output should be a concise narrative in 1 sentence, focusing on the most salient actions depicted across the frames. Enclose the exact text of relevant objects within <p></p> tags.

###### Input format:

[[‘subject’: ‘subject_text’, ‘verb’: ‘action_text’, ‘object’: ‘object_text’, ‘prepositions_objects’: [(’preposition’, ‘prepositional_object’)],],]

Output format: A Python dictionary with a key ‘CAPTION’, and as a value a dynamic description of the video content.

Infer motion from static descriptions. E.g. ‘image shows a person holding a spoon and a bowl’ implies ‘person is stirring food in a bowl’. Enclose the human and the most frequent object that is used to perform the action within <p></p> tags. If there is no human, enclose the two most frequent objects within <p></p> tags.

- User Input 1 SVO:

[[‘image’, ‘shows’, ‘cup’], [‘bowl’, ‘is’]], [[‘person’, ‘holding’, ‘spoon’], [‘spoon’, ‘is’, ‘bowl’], [[‘image’, ‘shows’, ‘spoon’, (‘inside’, ‘bowl’)]], [[‘person’, ‘seen’], [‘person’, ‘holding’, ‘spoon’], [‘spoon’, ‘used’],

[‘spoon’, ‘stir’, ‘food’, (‘in’, ‘bowl’)]], [[‘person’, ‘holding’, ‘spoon’], [‘spoon’, ‘is’, ‘bowl’]], [[‘person’, ‘holding’, ‘spoon’], [‘spoon’, ‘is’, ‘bowl’]], [[‘person’, ‘holding’, ‘spoon’], [‘spoon’, ‘is’, ‘bowl’]], [‘image’, ‘shows’, ‘spoon’, (‘in’, ‘bowl’)]], [[‘image’, ‘shows’, ‘bottle’], [‘bottle’, ‘positioned’, (‘beside’, ‘bowl’)]], [[‘image’, ‘shows’, ‘bottle’], [‘bottle’, ‘positioned’, (‘beside’, ‘cup’)]], [[‘image’, ‘shows’, ‘bottle’], [‘image’, ‘placed’, (‘on’, ‘counter’)],

[‘bottle’, ‘positioned’, (‘beside’, ‘bowl’)]]]

- Assistant Response 1 {‘CAPTION’: ‘<p>A person</p> is stirring <p>food in a bowl</p> using a spoon’}

User Input 2 SVO:

[[‘hand’, ‘using’, ‘cutting board’]], [[‘woman’, ‘using’, ‘cutting board’], [‘woman’, ‘make’, ‘craft project’]], [[‘child’, ‘using’, ‘craft cutter’], [‘child’, ‘cut’, ‘object’]], [[‘child’, ‘using’, ‘craft cutter’], [‘child’, ‘cut’, ‘paper’]], [[‘woman’, ‘using’, ‘craft cutter’], [‘woman’, ‘cut’, ‘object’]], [[‘woman’, ‘using’, ‘scissors pair’], [‘woman’, ‘cut’, ‘piece’, (‘of’, ‘paper’)]], [[‘hand’, ‘using’, ‘scissors pair’], [‘hand’, ‘cut’, ‘piece’, (‘of’, ‘paper’)]], [[‘woman’, ‘using’, ‘scissors pair’], [‘woman’, ‘cut’, ‘piece’, (‘of’, ‘paper’)]], [[‘woman’, ‘using’, ‘craft cutter’], [‘woman’, ‘cut’, ‘object’]], [[‘woman’, ‘using’, ‘craft cutter’], [‘woman’, ‘cut’, ‘plate’]]]

- Assistant Response 2 {‘CAPTION’: ‘<p>A woman</p> is cutting <p>an object</p> using a craft cutter’}

New User Input SVO: {input_svo}

Figure 12. The full prompt for Stage 2 (Video-level caption aggregation) of our automatic annotation approach (Section 3).

System Instructions You are tasked with classifying humans and objects to a set of given categories. Input format: Human/Object (string), set of categories (lists of strings). Output format: A Python dictionary with a key ‘CATEGORY’, and as a value the predicted category of the human/object. Use ‘None’ if the human/object doesn‘t belong to any of the categories. DO NEVER classify a human as the object category and vice versa.

###### User Input 1

Input: ‘person‘ Categories: [‘a woman’, ‘her hair’]

- Assistant Response 1 {‘CATEGORY’: ‘a woman’}

User Input 2 Input: ‘table’ Categories: [‘a person’, ‘a bowl’]

- Assistant Response 2 {‘CATEGORY’: ‘None’}

User Input 3

Input: ‘a piece of food on a plate’ Categories: [‘a woman’, ‘a meal’]

- Assistant Response 3 {‘CATEGORY’: ‘a meal’}

User Input 4 Input: ‘a hand’ Categories: [‘a person’, ‘food on a plate’]

- Assistant Response 4 {‘CATEGORY’: ‘a person’}

User Input 5 Input: ‘a man in a white shirt and black apron is also present’ Categories: [‘a person’, ‘food’]

- Assistant Response 5 {‘CATEGORY’: ‘a person’}

New User Input Input: {input_object} Categories: {input_categories}

Figure 13. The full prompt for Stage 3 (Temporally consistent bounding box annotation) of our automatic annotation approach (Section 3).

