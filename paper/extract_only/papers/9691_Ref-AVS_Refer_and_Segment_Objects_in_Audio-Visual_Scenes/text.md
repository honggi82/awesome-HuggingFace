# arXiv:2407.10957v1[cs.CV]15Jul2024

## Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes

Yaoting Wang1† , Peiwen Sun2† , Dongzhan Zhou3† , Guangyao Li1 , Honggang Zhang2 , and Di Hu 1,4

1 Gaoling School of Artificial Intelligence, Renmin University of China, China yaoting.wang@outlook.com {guangyaoli,dihu}@ruc.edu.cn 2 Beijing University of Posts and Telecommunications, Beijing, China {sunpeiwen,zhhg}@bupt.edu.cn 3 Shanghai Artificial Intelligence Laboratory, Shanghai, China zhoudongzhan@pjlab.org.cn 4 Engineering Research Center of Next-Generation Search and Recommendation

Abstract. Traditional reference segmentation tasks have predominantly focused on silent visual scenes, neglecting the integral role of multimodal perception and interaction in human experiences. In this work, we introduce a novel task called Reference Audio-Visual Segmentation (RefAVS), which seeks to segment objects within the visual domain based on expressions containing multimodal cues. Such expressions are articulated in natural language forms but are enriched with multimodal cues, including audio and visual descriptions. To facilitate this research, we construct the first Ref-AVS benchmark, which provides pixel-level annotations for objects described in corresponding multimodal-cue expressions. To tackle the Ref-AVS task, we propose a new method that adequately utilizes multimodal cues to offer precise segmentation guidance. Finally, we conduct quantitative and qualitative experiments on three test subsets to compare our approach with existing methods from related tasks. The results demonstrate the effectiveness of our method, highlighting its capability to precisely segment objects using multimodal-cue expressions. Dataset is available at https://gewu-lab.github.io/Ref-AVS.

Keywords: Referring Audio-Visual Segmentation · Audio-Visual Segmentation · Multimodal Learning

### 1 Introduction

In the real world, visual scenes are often accompanied by diverse multimodal information1, including audio and text modalities. This additional multimodal

† Equal contribution. Corresponding author.

1 To clarify, we refer to the sound and vision in the scene as multimodal information, while the attribute description of audio and vision (e.g., louder and on the right) in the reference expression is named multimodal cues.

[Figure 1]

- Fig. 1: Comparison of the Ref-AVS task with other related tasks. Ref-AVS challenges machines to locate objects of interest in the visual space using multimodal cues, just like humans do in the real world.

information offers valuable spatiotemporal and semantic cues, aiding individuals in locating objects of interest. Similarly, the ability to locate objects based on multimodal cues plays a crucial role in the functioning of machines. For example, stage cameras should be able to locate the musician playing instruments, while movie cameras should locate the superhero in front of the yelling villain. In the above scenarios, using a single modality for segmentation alone cannot achieve the desired behavior.

To date, researchers have studied the reference segmentation problem from somewhat limited scenarios. As shown in the coordinate system in Fig. 1, there are currently three mainstream developments in segmentation methods for different modalities. Firstly, from the perspective of visual cues, Video Object Segmentation (VOS) [10, 31–33, 41] emerges with annotated first frame mask as the reference, guiding the segmentation of specific objects in subsequent video frames. While VOS can achieve arbitrary object segmentation in videos, it heavily relies on precise annotation of the first frame, which can be challenging and time-consuming in practice. Secondly, from the perspective of text cues, Referring Video Object Segmentation (R-VOS) [9, 12, 14, 20] explores the segmentation ability by using attribute descriptions as guidance. As R-VOS has successfully replaced the mask annotation used in VOS with natural language, it provides a more accessible and user-friendly reference form but the capability to locate objects in more natural dynamic audio-visual scenes is still limited. In contrast, from the perspective of audio cues, Audio-Visual Segmentation (AVS) [27, 38, 45–48] leverages audio as a guide to segment sounding objects. This approach effectively addresses the challenge of locating objects in dynamic audio-visual scenes. However, AVS still faces limitations in its ability to focus on general objects that do not generate sound and cannot effectively locate objects of specific interest.

In simple terms, current work falls short of enabling machines to locate objects of interest in natural dynamic audio-visual scenes. For instance, as illus-

trated in Fig. 1, how can machines accurately locate a person who is genuinely playing a musical instrument over time? This requires machines to infer which instrument is making a sound and who is playing that instrument. Though Yan et al. [42] utilize either audio or language modality for multimodal reference segmentation, their method does not integrate multimodal information simultaneously nor support the aforementioned scenario. Therefore, introducing a task to explore the possibility of locating interested objects in natural dynamic audiovisual scenes holds significant practical potential.

In this paper, we propose a pixel-level segmentation task called Referring Audio-Visual Segmentation (Ref-AVS), which requires the network to densely predict whether each pixel corresponds to the given multimodal-cue expression, including dynamic audio-visual information. Top-left of Fig. 1 highlights the distinctions between Ref-AVS and previous tasks. The Ref-AVS task poses a greater challenge as it requires the network to accurately locate and segment objects in a more complex and stereoscopic modality space. To this end, a computation model with comprehensive multimodal understanding capabilities is necessary. To foster research advancements in this field, we introduce the Ref-AVS Bench, the pioneering benchmark that focuses on locating and segmenting objects of interest using referring multimodal-cue expressions. Considering the complexity of real-world audio-visual scenes, we collect about 4,000 audible video clips from YouTube of which more than 60% are multi-source sound scenarios. More than 20,000 reference expressions are collected and verified by experts to ensure accuracy and reliability, which adopt multimodal cues to describe objects in diverse and dynamic audio-visual scenes. Moreover, a special unseen test set is considered to evaluate the model’s generalization ability in the growing demand for zero-shot scenarios.

Our contributions can be summarized as follows:

- – We propose Ref-AVS as a challenging scene understanding task that segments objects of interest with multimodal-cue expressions and provide the corresponding Ref-AVS benchmark for performance training and validation.
- – We design an end-to-end framework for Ref-AVS that efficiently processes the multimodal cues with a crossmodal transformer, serving as a feasible research framework for future development.
- – We conduct extensive experiments to demonstrate the advantages of considering multimodal cues for visual segmentation, which also indicate the superiority of our approach in all subsets.

### 2 Related Work

#### 2.1 Referring Video Object Segmentation (R-VOS)

The main objective of R-VOS is to perform object segmentation in streaming videos based on given natural language expressions. Initially proposed in 2018, A2D-Sentence [14] aims to segment actors in video content based on

descriptions of their actions. Subsequently, Refer-DAVIS17 [20] and JHMDBSentences [12] use extra language as references to replace previous VOS settings. To facilitate large-scale R-VOS training, Seo et al. [36] develop the ReferYouTube-VOS dataset upon the YouTube-VOS-2019 dataset [41] and Henghui et al. [9] build the MeViS for complex motion expression. Overall, the primary R-VOS methods have evolved from semi-supervised VOS and incorporate attention mechanisms to enhance modalities interaction. For example, MAttNet [44] decomposes the expression information into language- and word-level attention, while URVOS [36] introduces crossmodal attention and memory attention. ReferFormer [40] and MTTR [1] utilize the crossmodal transformer to connect language expression with image regions. These datasets typically offer expressions that encompass the action or appearance attributes of the target object. However, these expressions primarily focus on visual information, which fails to meet the increasing multimodal demands in today’s diverse audio-visual scenes.

#### 2.2 Audio-Visual Segmentation (AVS)

The multimodal segmentation in audio-visual scenes, which is of great concern to us, has received some attention in AVS [13,38,47]. AVS aims to obtain finer pixel-level masks corresponding to sound-emitting objects in the visual space. In contrast, previous studies on audio-visual localization [17,37] use heat maps for coarse localization in an unsupervised learning manner. The existing works on AVS can be broadly categorized into fusion-based methods [13, 18, 23, 25, 26,47] and prompt-based methods [27,29,30,38]. The former primarily focuses on localizing sounding objects by fusing audio and visual features, while the latter emphasizes constructing effective audio prompts for the visual foundation model. However, AVS simply segments all sound-emitting objects in the visual space, without the flexibility to combine the multimodal cues to segment specific objects of interest.

#### 2.3 Language-aided Audio-Visual Scene Understanding

Currently, there is a limited number of public tasks offering datasets for audiovisual scene comprehension accompanied by language assistance. Two notable examples of such datasets are Music-AVQA [22] and AVQA [43], both encompassing audio, visual, and language information. These datasets contain rich audio-visual components and focus on annotations for textual questions and answers, emphasizing temporal (Before/After), spatial cues (Left/Right), etc.

Researchers [4,19,21] acknowledge the significance of the provided question, as it guides the feature extraction process for both audio and visual signals. Consequently, they have made efforts to identify pertinent segments by evaluating the semantic similarity between the question and temporal segments. Furthermore, [21] has been dedicated to locating crucial areas by leveraging semantic similarity between questions and visual tokens. Clearly, existing explorations have significantly advanced the research on language-aided audio-visual scene understanding. Nevertheless, the works that rely on the aforementioned datasets

are still unable to offer pixel-level scene understanding. This encourages us to begin exploring fine-grained segmentation of dynamic audio-visual scenes in the real world by constructing appropriate datasets.

### 3 Refer-AVS Dataset

#### 3.1 Object Category

To ensure a diverse range of referred objects, we have carefully selected a wide variety of audible objects in 48 categories as well as a smaller selection of static, unsoundable objects in 3 categories. For the objects that can produce sound, we have chosen 20 categories of musical instruments, 8 of animals, 15 of machines, and 5 of humans. In the special case of humans, considering their diverse appearances, voices, and actions, we have employed the concept of morphology and classified them into 5 categories, based on age and gender.

#### 3.2 Video Selection

During the process of video collection, we employed the techniques presented in [3,47] to ensure the alignment of audio and visual snippets with the intended semantics. All videos were sourced from YouTube under the Creative Commons license, and each video was trimmed to a duration of 10 seconds. Throughout the manual collection process, we deliberately avoided videos falling into several categories (detailed in the appendix): 1) Videos with a large number of instances of the same semantics; 2) Videos characterized by extensive editing and camera switching; 3) Non-realistic videos containing synthetic artifacts.

To raise the alignment with real-world distributions, we carefully select videos that contribute to the diversification of scenes within our dataset. Specifically, we focus on selecting videos that involve interactions between multiple objects, such as musical instruments, people, vehicles, etc. The rich combination of categories indicates that our dataset is not limited to a narrow set of scenarios but rather encompasses a broad spectrum of real-life scenes where such objects are likely to naturally appear together. Refer to the supplementary materials for details.

In addition to diversity, we also carefully filter the videos to ensure that the dataset includes scenes with greater complexity and a larger number of objects. Specifically, 56% of the total videos contain two or more objects, while 13% of the total videos contain three or more objects.

#### 3.3 Expressions

The diversity of expressions is one of the core factors in the assembly of the RefAVS dataset. The expressions consist of three dimensions of information, namely audio, vision, and time. The auditory dimension incorporates characteristics such

- as volume and rhythm while the visual dimension encompasses attributes like the appearance and spatial configuration of objects. We also leverage temporal

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

- Fig. 2: The illustration of our Ref-AVS benchmark. Note,for , the sound volume increases successively from silent to loud. Our benchmark is meticulously designed to encompass multimodal expressions from multiple dimensions. By combining various types of modality expressions, we achieve a dataset that exhibits great diversity.

cues to generate references with sequential hints, e.g., “the one making sound first” or “the one appearing later”. By integrating auditory, visual, and temporal information, we craft a rich array of expressions that not only accurately reflect multimodal scenarios but also meet users’ specific needs for precise references.

- Fig. 2 illustrates examples of the combination of different modalities. The accuracy of expressions is also a core concern. We follow three rules to

generate high-quality expressions: 1) Uniqueness: The object referred to by an expression must be unique and an expression cannot refer to multiple objects simultaneously. 2) Necessity: Complex expressions can be used for reference, but each adjective in the sentence should narrow down the scope of target objects, avoiding unnecessary and redundant descriptions of the object. 3) Clarity: Certain expression templates involve subjective factors, such as “the __ with a louder sound”. The use of such expressions should only occur when the situation is clear enough to avoid ambiguous references.

Besides diversity and accuracy, we have graded the difficulty of expressions based on the number of cues involved, with the proportions of easy, medium, and hard samples in the overall dataset being 20%, 60%, and 20%, respectively. This gradation could be a great benefit to future research like curriculum learning. Please refer to the supplementary material for more details.

#### 3.4 Segmentation Masks

We divide each 10-second video into ten equal 1-second snippets and the objective of annotation is to acquire the first frame’s mask for each snippet. For these sampled frames, the ground truth labels are binary masks indicating the target object, according to expressions and multimodal information.

To obtain such a mask, initially, we manually select the pivotal frames for each 10-second video in which the target object is present. These pivotal frames

###### Table 1: Datasets comparison with related tasks.

Task Dataset Modality Video Frame Object Expression Object/Video Pub.

Flickr-SoundNet [35] A+V 5,000 5,000 - - - CVPR’2018 AVL

VGG-SS [2] A+V 5,158 5,158 - - - CVPR’2021

AVS [47] A+V 12,356 82,972 13,500∼ - 1.09∼ ECCV’2022 AVS

VPO [45] A+V - 34,874 40,000∼ - 1.14 (Object/Frame) CVPR’2024 J-HMDB Sentences [12] T+V 928 928 928 928 1.28 CVPR’2018

A2D Sentences [14] T+V 3,782 11,936 4,825 6,656 1 CVPR’2018

- Refer-DAVIS16 [20] T+V 50 3,455 50 100 1 ACCV’2018
- Refer-DAVIS17 [20] T+V 90 13,543 205 1,544 2.27 ACCV’2018

R-VOS

Refer-Youtube-VOS [36] T+V 3,975 11,936 7,451 27,899 1.86 ECCV’2020

MeViS [9] T+V 2,006 443,000 8,171 28,570 4.28 ICCV’2023 Ref-AVS Ref-AVS Bench A+T+V 4,002 40,020 6,888 20,261 1.72 ECCV’2024

may occur at the beginning, middle, or end of the video, depending on when the target object is most clearly visible. Subsequently, we utilize Grounding SAM [34] to segment and label the pivotal frames, which will be manually checked and corrected subsequently. This process allows us to generate masks and labels for multiple target objects within the pivotal frames. Once the masks for the pivotal frames are established, we apply a tracking algorithm [7] to track the target object across the preceding and subsequent frames and obtain the ultimate mask and label for the target object in a span of 10 frames.

#### 3.5 Dataset Statistics

In addition, we compare Refer-AVS with other popular audio-visual benchmarks in the Tab. 1. The Flickr-SoundNet [35] and VGG-SS [2] benchmarks are labeled

- at a patch level through bounding boxes and possess around 5,000 frame-level annotations. Compared with our dataset with pixel-level annotations, these benchmarks have a significantly lower quantity of annotations. Regarding the AVS dataset [45–47], the videos in our dataset contain a higher average number of objects about 1.72 objects per video, implying that our scenarios are more challenging with multiple sound sources and multiple semantics. Within such scenarios, our Ref-AVS benchmark is particularly valuable as it demonstrates the ability to effectively focus on objects of real interest. Our dataset also has more uniform video durations and a more refined video selection process than AVS. When it comes to datasets for R-VOS tasks, we maintain a consistent advantage in the number of videos compared to other datasets [9,12,14,20,36]. Moreover, we possess a considerable volume of data encompassing a large number of objects, expressions, and complex scenes.

Overall, our Ref-AVS dataset encompasses a substantial collection of 20,000 expressions and pixel-level annotations spread across 4,000 videos, totaling more than 11 hours. As a result, we believe that this benchmark adequately fulfills the requirements for facilitating research on the Ref-AVS task, while also presenting a significant level of challenge in this domain. We aim to continuously expand the dataset for broader community needs, like training larger foundation models.

Table 2: Dataset split of Ref-AVS Bench.

|Subset|all<br><br>|train val test|test test test (seen) (unseen) (null)|
|---|---|---|---|
|Category† Video Object Expression|52 4,002 6,888 20,261|39 39 52 2,908 276 818 5,366 518 1,004<br><br>14,117 1,349 4,770|39 13 1 292 269 257 478 269 257 2,288 1,454 1,028|

† The categories here contains “background”.

#### 3.6 Dataset Split

As shown in Tab. 2, the complete dataset is divided into three sets, i.e., a training set of 2908 videos, a validation set of 276 videos, and a test set of 818 videos. The videos in the test set and their corresponding annotations undergo a meticulous review and re-annotation process conducted by experienced workers. In order to thoroughly evaluate the models’ performance in the Ref-AVS task, the test set is further divided into three distinct subsets, each serving a specific purpose.

Seen: The first test subset, referred to as the “seen test set”, comprises object categories that have already appeared in the training set. The purpose of establishing this subset is to evaluate the model’s fundamental performance and assess how well it generalizes to familiar object categories.

Unseen: To address the growing demand for the generalization capabilities of various models in an open-world scenario, an additional test set was created specifically to assess the model’s ability to generalize to unseen audio-visual scenes. This “unseen test set” consists of object categories that did not appear in the training set, although their super-categories (e.g., animal, vehicle) may have been present in the training set. The purpose of this test set is to evaluate the model’s capacity to generalize to novel object categories while leveraging its understanding of broader super-categories.

Null: A “null reference problem” arises when an expression refers to an object that either does not exist or is not visible in the given context. A model that accurately understands expression guidance should not segment any object in scenarios of a null reference [39]. Based on this consideration, we have developed a “null subset” to test the model’s robustness against such challenges. This subset comprises object categories that were present during training but are paired with expressions that do not correlate, ensuring all objects in the video frames are irrelevant to the given reference expression. Therefore, the true masks for this subset are empty, and the model should refrain from segmenting any object.

### 4 Expression Enhancing with Multimodal Cues

#### 4.1 Overall Architecture

Ref-AVS is designed to locate objects of interest in dynamic audio-visual scenes by utilizing multimodal cues. To solve this challenging task, we propose the

[Figure 6]

- Fig. 3: The overall architecture of our method EEMC. We utilize cached memory to preserve temporal information, enabling the model to capture temporal mutations. Modality encoding provides a more clear context for the multimodal cue features Qm, using the modality identify tokens [aud] and [vis]. Lastly, we achieve efficient prompting of the visual foundation model by employing cross-attention, where mask queries q serve as the target and multimodal cue features Qm act as the source.

Expression Enhancing with Multimodal Cues (EEMC) method, which focuses on integrating multimodal information in the dynamic audio-visual scenes into reference expressions with the corresponding multimodal cues. This approach enables the formation of a comprehensive multimodal reference feature. Additionally, we employ an attention mechanism to utilize the multimodal reference cues as a prompt for the visual foundation model, thereby facilitating the final segmentation process.

#### 4.2 Multimodal Representations

Audio: Similar to the video processing, we divide the audio input into clips at 1-second intervals. Audio representations FA ∈ Rt×d

A is encoded from VGGish [15,16], where t represents the duration of the audio in seconds and matches the number of video frames. The audio representations are extracted offline and the audio encoder is not fine-tuned.

Visual: We sample t frames from the video input at 1-second intervals and extract visual representations FV ∈ Rt×d

V ×H×W using a pre-trained Swin-base model [11]. The visual encoder is not fine-tuned. Expression: We use RoBERTa [8, 28] as our text encoder to extract textual expression features FT ∈ RL×d

T , where L denotes the length of the expression. The expression representations are obtained off-the-shelf without fine-tuning.

#### 4.3 Temporal Bi-modal Transformer

Temporal A-T & V-T Fusion: This module is introduced to retrieve the expression-related information of each modality. Firstly, for the convenience of the following multimodal fusion, we prepare the projected audio feature FA′ ∈ Rt×1×d

V , downsampled and flattened visual feature FV ′ ∈ Rt×(W8 ×H8 )×dV , and projected text feature FT′ ∈ Rt×L×d

V which is expanded along the temporal dimension. Subsequently, the features are combined to yield the expression-related multimodal features FˆA, FˆV and modality-aware expression features FˆT

, FˆT

, with the same dimension as FA′, FV ′, FT′, respectively:

A

V

FˆM,FˆT

= MF(Concat(FM′;FT′)),M ∈ {A,V } (1)

M

where Concat(·) denotes the concatenate operation, and MF(·) denotes the modality fusion function, which is employed as self-attention.

Cached Memory: Although the aforementioned fusion method can combine cross-modal information, it does not give much emphasis to strong temporal dynamics. To address this issue, we introduce a straightforward cached memory CA ∈ Rt×2×d

T to explicitly capture the temporal mutations as they occur. The cache memory is required to store the mean modality features in the temporal domain from the beginning to the current moment. Then, the multimodal cues feature QA and QV output from temporal-aware transformers can be calculated as

,CV ∈ Rt×(W×H+1)×d

T

QτM = (β + 1)FˆMτ − βCMτ ,M ∈ {A,V }, (2)

- 0, if τ = 1,
- 1

CMτ =

(3)

τ−1 i=0

FˆMi , if τ > 1,

τ−1

while τ is the time stamp of the specific feature at the temporal dimension and β is a hyper-parameter to control the sensitivity to temporal dynamics. When the current audio or visual features do not change significantly compared to the mean of past features (i.e., FˆMτ = CMτ ), QτM remains nearly unchanged. However, in cases where the current change is drastic (i.e., FˆMτ ̸= CMτ ), the cached memory can amplify the difference in the current feature, leading to an output with salient features. It is important to note that text cue feature QT inherently possesses a highly abstracted semantic nature, which means manipulation may not be necessary at this stage. Therefore, at this stage, we combine and average the original textual features with the text features enhanced by multimodal information, resulting in a comprehensive enhanced text feature QT = mean(FT′ +FˆT

+FˆT

). Finally, we obtain the processed cue feature of each modality QA, QV and QT. Modality Encoding: The modality encoding approach involves incorporating modality identify tokens [aud] ∈ R1×d

A

V

V for audio modality and [vis] ∈ R1×d

V

for visual modality into the following multimodal cues integration process. Since only two tokens are required to split the sequence into three segments, we only need to manipulate the audio and visual cue features.

- 4.4 Prompting with Multimodal Cues

In this Prompting with Multimodal Cues (PMC) phrase, when we obtain the final multimodal cues features, we can concatenate the multimodal cues together and then apply self-attention to obtain comprehensive multimodal cues Qm ∈ Rt×(H×W+L+3)×d

V through the Multimodal Integration Transformer:

Qm = MF(Concat([QA;[aud];QV ;[vis];QT])), (4)

where MF is the multimodal fusion function serving as the self-attention for multimodal cues interaction. Then we utilize these comprehensive multinational cues to prompt the learnable mask queries q ∈ RN×d

V in the transformer-based segmentation decoder with cross-attention, where N is the number of mask queries:

qQ = CATF(query = q,key = Qm,value = Qm), (5) where qQ ∈ RN×d

V is the updated mask queries. CATF(·) is the cross-attention transformer with q and Qm as the information target and source, respectively.

- 5 Experiments

- 5.1 Implementation Details

Mask2Former [5] serves as our visual foundation model to provide the commonly used transformer-based segmentation decoder. By default, in this work, all input video frames are scaled to 384 × 384. The shape of the visual features is [H = 64,W = 64,dV = 256], and we employ 8-fold downsampling to reduce the computation costs. The audio features with dA = 128 are extracted from the mono-channel waveform. The text features have a shape of [L = 25,dT = 768]. For convenience, we map all modality feature dimensions to dV . Hyper-parameter β is set to 1 by default. Transformer layers NL of the Temporal Bi-modal Transformer, Multimodal Integration Transformer and the cross-attention transformer CATF are set to 4 by default. The number of mask queries Nq is fixed to 100.

- 5.2 Evaluation Metrics

To conduct a comprehensive evaluation of our Ref-AVS method, we employ the Jaccard index (J ) and F-score (F) as performance metrics. Additionally, we introduce S in the null reference test set to assess the efficacy of expression guidance in the model. S denotes the square root of the ratio between the predicted mask area and the background area, a higher value indicates a larger proportion of the mask relative to the background area, suggesting a lack of precise expression guidance provided to the model.

Table 3: Performance on Ref-AVS Bench. We use three test subsets to evaluate the comprehensive ability of Ref-AVS methods. Mix is the average performance of Seen and Unseen test set. We also use Null test set to evaluate the robustness of multimodal-cue expression guidance.

Seen Unseen Mix(S+U) Null Pub. Method Task J (%) F J (%) F J (%) F S (↓) AVSBench [47]

21.22 0.457 27.67 0.509 24.45 0.483 -

AVS

ECCV’2022

+text 23.20 0.511 32.36 0.547 27.78 0.529 0.208 AVSegFormer [13]

29.16 0.423 32.71 0.451 30.94 0.437 -

AVS

AAAI’2024

+text 33.47 0.470 36.05 0.501 34.76 0.486 0.171 GAVS [38]

24.85 0.455 27.86 0.460 26.36 0.457 -

AVS

AAAI’2024

+text 28.93 0.498 29.82 0.497 29.38 0.498 0.190 ReferFormer [40]

27.46 0.477 24.94 0.504 26.2 0.491 -

Ref-VOS

CVPR’2022

+audio 31.31 0.501 30.40 0.488 30.86 0.495 0.176 R2VOS [24]

21.34 0.309 23.91 0.431 22.63 0.370 -

Ref-VOS

ICCV’2023

+audio 25.01 0.410 27.93 0.498 26.47 0.454 0.183 EEMC Ref-AVS 34.20 0.513 49.54 0.648 41.87 0.581 0.007 ECCV’2024

#### 5.3 Quantitative Results

We compare our Ref-AVS method with existing approaches in the relevant field using our Ref-AVS benchmark. The results from the seen test set demonstrate the superior performance of our Ref-AVS method on this dataset, outperforming other methods by a significant margin. To ensure fairness, we equip the other methods with the same inputs from both the audio and visual modalities as ours. However, even with this additional information, these methods still fail to achieve satisfactory results. The result indicates that simple modal fusion is inadequate for addressing the challenges of understanding multimodal cues within the Ref-AVS task. Instead of directly incorporating audiovisual information, our approach selects textual representations as carriers of multimodal information because they contain rich semantics and cues that are contextually relevant to the current audiovisual environment.

We conduct tests on the unseen test set and the null reference test set to explore the generalization and multimodal cues-following ability. Our method still maintains a leading position in the unseen test set because we leverage text with highly abstract semantic capabilities as a multimodal information carrier, instead of directly fusing audio and visual information. Therefore, the obtained multimodal cues can provide more robust semantic guidance. On the null test set, we obtain leading results among all other methods, indicating that our method can perceive multimodal cues to a reasonably accurate degree. Compared to other transformer-based and query-based frameworks, AVSBench, as a classic AVS method, obtains inferior results. This may be attributed to the model simply fuses multimodal cues and visual features in a progressive fashion, resulting in relatively weak guidance over the visual space.

[Figure 7]

- Fig. 4: Segmentation masks of the referred objects in the Ref-AVS Bench. Note, for , the sound volume increases successively from silent to loud. We compare our

[Figure 8]

[Figure 9]

[Figure 10]

methodology with AVSegformer and ReferFormer. More cases can be found in the supplementary material.

#### 5.4 Qualitative Results

We visualize the segmentation masks on the test set of the Ref-AVS Bench and compare them with AVSegFormer and ReferFormer, as shown in Fig. 4. From these qualitative results, we observe that both ReferFormer in the Ref-VOS task and AVSegFormer in the AVS task fail to accurately segment the object described in the expression. Specifically, AVSegformer struggles to fully comprehend the expression and tends to directly generate the sound source. This issue is exemplified in the bottom-left sample, where AVSegformer erroneously segments the vacuum cleaner instead of the boy. On the other hand, Ref-VOS may not adequately understand the audio-visual scene and thus mistakenly identify the toddler as the piano player, as shown in the top-right sample. In contrast, our Ref-AVS method demonstrates a superior capability to simultaneously process multi-modal expressions and scenes, enabling it to accurately interpret the user instruction and segment the intended object of interest.

#### 5.5 Ablation Study

We conduct ablation studies to investigate the impact of two modalities (audio and text) information on the Ref-AVS task, as well as the effectiveness of the proposed method. As shown in the Tab. 4, we observe that in setting ②, removing the text information results in a significant performance degradation of 11.44% in

###### Table 4: Ablation study on modalities and method modules.

Seen Unseen Mix Null J (%) F J (%) F J (%) F S(↓)

Setting

- ① EEMC 34.20 0.513 49.54 0.648 41.87 0.581 0.007
- ② (-) Text 23.67 0.468 37.19 0.641 30.43 0.555 0.011
- ③ (-) Audio 27.27 0.492 43.64 0.656 35.46 0.574 0.010

- ④ (-) Cached Memory 25.30 0.494 36.73 0.660 31.05 0.577 0.009
- ⑤ (-) Modality Encoding 27.20 0.470 44.97 0.637 36.09 0.554 0.009

J and 2.60% in F. This degradation is much higher compared to setting ③, where the removal of the audio information results in a combined mixed drop of 6.41% in J and 0.70% in F. This phenomenon primarily results from the clarity and directness of textual information as a reference source. In contrast, when relying solely on audio information, the model often overlooks the referential content and instead focuses on identifying objects visually associated with audible behaviors. We also conduct experiments to examine the function of each module in this framework. The cached memory is designed to track significant changes in the temporal domain. Meanwhile, the purpose of modality encoding is to extract more distinct and thorough features from multimodal cues, enhancing modal perception. Results from settings ④ and ⑤ in Tab. 4 demonstrate the effectiveness of these modules.

### 6 Conclusion

In this work, we introduce a novel task, Ref-AVS, which requires the machine to segment objects of interest in dynamic audio-visual scenes based on expressions that incorporate multimodal cues and temporal information. To support this field, we develop the first benchmark, Ref-AVS Bench, for performance training and evaluation. Our method, EEMC, establishes a robust baseline with an advanced multimodal cues fusion module. We compare our method with several existing approaches on the Ref-AVS Bench dataset and demonstrate its promising performance in accurately locating objects using multimodal reference expressions. Essentially, Ref-AVS leverages audio, visual, and textual data to enrich real-world scene comprehension. Our task has potential in applications such as attention focusing and object editing in immersive audio-visual scenarios like extended reality, thereby enhancing user interaction experiences. In the future, we plan to further expand the scale of the current dataset to meet the growing data demands in the era of large language models.

### Acknowledgements

This research was supported by National Natural Science Foundation of China (NO.62106272), and Public Computing Cloud, Renmin University of China. This research was partially supported by Shanghai Artificial Intelligence Laboratory.

### References

- 1. Botach, A., Zheltonozhskii, E., Baskin, C.: End-to-end referring video object segmentation with multimodal transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4985–4995 (2022)
- 2. Chen, H., Xie, W., Afouras, T., Nagrani, A., Vedaldi, A., Zisserman, A.: Localizing visual sounds the hard way. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16867–16876 (2021)
- 3. Chen, H., Xie, W., Vedaldi, A., Zisserman, A.: Vggsound: A large-scale audiovisual dataset. In: ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 721–725. IEEE (2020)
- 4. Chen, Z., Wang, L., Wang, P., Gao, P.: Question-aware global-local video understanding network for audio-visual question answering. IEEE Transactions on Circuits and Systems for Video Technology (2023)
- 5. Cheng, B., Choudhuri, A., Misra, I., Kirillov, A., Girdhar, R., Schwing, A.G.: Mask2former for video instance segmentation (2021)
- 6. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1290–1299

(2022)

- 7. Cheng, H.K., Oh, S.W., Price, B., Schwing, A., Lee, J.Y.: Tracking anything with decoupled video segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1316–1326 (2023)
- 8. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805

(2018)

- 9. Ding, H., Liu, C., He, S., Jiang, X., Loy, C.C.: Mevis: A large-scale benchmark for video segmentation with motion expressions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2694–2703 (October 2023)
- 10. Ding, H., Liu, C., He, S., Jiang, X., Torr, P.H., Bai, S.: Mose: A new dataset for video object segmentation in complex scenes. arXiv preprint arXiv:2302.01872

(2023)

- 11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
- 12. Dutta, U.K., Harandi, M., Sekhar, C.C.: Unsupervised deep metric learning via orthogonality based probabilistic loss. IEEE Transactions on Artificial Intelligence 1(1), 74–84 (2020)
- 13. Gao, S., Chen, Z., Chen, G., Wang, W., Lu, T.: Avsegformer: Audio-visual segmentation with transformer (2023)
- 14. Gavrilyuk, K., Ghodrati, A., Li, Z., Snoek, C.G.: Actor and action video segmentation from a sentence. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 5958–5966 (2018)
- 15. Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M., Ritter, M.: Audio set: An ontology and human-labeled dataset for audio events. In: 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). pp. 776–780. IEEE (2017)

- 16. Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B., et al.: Cnn architectures for large-scale audio classification. In: 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). pp. 131–135. IEEE (2017)
- 17. Hu, D., Qian, R., Jiang, M., Tan, X., Wen, S., Ding, E., Lin, W., Dou, D.: Discriminative sounding objects localization via self-supervised audiovisual matching. Advances in Neural Information Processing Systems 33, 10077–10087 (2020)
- 18. Huang, S., Li, H., Wang, Y., Zhu, H., Dai, J., Han, J., Rong, W., Liu, S.: Discovering sounding objects by audio queries for audio visual segmentation (2023)
- 19. Jiang, Y., Yin, J.: Target-aware spatio-temporal reasoning via answering questions in dynamics audio-visual scenarios. arXiv preprint arXiv:2305.12397 (2023)
- 20. Khoreva, A., Rohrbach, A., Schiele, B.: Video object segmentation with language referring expressions. In: Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part IV 14. pp. 123–141. Springer (2019)
- 21. Li, G., Hou, W., Hu, D.: Progressive spatio-temporal perception for audio-visual question answering. In: Proceedings of the 31st ACM International Conference on Multimedia. p. 7808–7816. MM ’23, Association for Computing Machinery, New York, NY, USA (2023). https://doi.org/10.1145/3581783.3612293, https: //doi.org/10.1145/3581783.3612293
- 22. Li, G., Wei, Y., Tian, Y., Xu, C., Wen, J.R., Hu, D.: Learning to answer questions in dynamic audio-visual scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19108–19118 (2022)
- 23. Li, K., Yang, Z., Chen, L., Yang, Y., Xun, J.: Catr: Combinatorial-dependence audio-queried transformer for audio-visual video segmentation. arXiv preprint arXiv:2309.09709 (2023)
- 24. Li, X., Wang, J., Xu, X., Li, X., Raj, B., Lu, Y.: Robust referring video object segmentation with cyclic structural consensus. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22236–22245 (2023)
- 25. Ling, Y., Li, Y., Gan, Z., Zhang, J., Chi, M., Wang, Y.: Hear to segment: Unmixing the audio to guide the semantic segmentation (2023)
- 26. Liu, J., Ju, C., Ma, C., Wang, Y., Wang, Y., Zhang, Y.: Audio-aware queryenhanced transformer for audio-visual segmentation (2023)
- 27. Liu, J., Wang, Y., Ju, C., Zhang, Y., Xie, W.: Annotation-free audio-visual segmentation. arXiv preprint arXiv:2305.11019 (2023)
- 28. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V.: Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019)
- 29. Ma, J., Sun, P., Wang, Y., Hu, D.: Stepping stones: A progressive training strategy for audio-visual semantic segmentation. IEEE European Conference on Computer Vision (ECCV) (2024)
- 30. Mo, S., Tian, Y.: Av-sam: Segment anything model meets audio-visual localization and segmentation. arXiv preprint arXiv:2305.01836 (2023)
- 31. Perazzi, F., Khoreva, A., Benenson, R., Schiele, B., Sorkine-Hornung, A.: Learning video object segmentation from static images. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2663–2672 (2017)
- 32. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., SorkineHornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 724–732 (2016)

- 33. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbeláez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017)
- 34. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Zhang, L.: Grounded sam: Assembling open-world models for diverse visual tasks (2024)
- 35. Senocak, A., Oh, T.H., Kim, J., Yang, M.H., So Kweon, I.: Learning to localize sound source in visual scenes. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)
- 36. Seo, S., Lee, J.Y., Han, B.: Urvos: Unified referring video object segmentation network with a large-scale benchmark. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV

16. pp. 208–223. Springer (2020)

- 37. Tian, Y., Shi, J., Li, B., Duan, Z., Xu, C.: Audio-visual event localization in unconstrained videos. In: Proceedings of the European Conference on Computer Vision (ECCV) (September 2018)
- 38. Wang, Y., Liu, W., Li, G., Ding, J., Hu, D., Li, X.: Prompting segmentation with sound is generalizable audio-visual source localizer. arXiv preprint arXiv:2309.07929 (2023)
- 39. Wang, Y., Sun, P., Li, Y., Zhang, H., Hu, D.: Can textual semantics mitigate sounding object segmentation preference? IEEE European Conference on Computer Vision (ECCV) (2024)
- 40. Wu, J., Jiang, Y., Sun, P., Yuan, Z., Luo, P.: Language as queries for referring video object segmentation (2022)
- 41. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327

(2018)

- 42. Yan, S., Zhang, R., Guo, Z., Chen, W., Zhang, W., Li, H., Qiao, Y., He, Z., Gao, P.: Referred by multi-modality: A unified temporal transformer for video object segmentation (2023)
- 43. Yang, P., Wang, X., Duan, X., Chen, H., Hou, R., Jin, C., Zhu, W.: Avqa: A dataset for audio-visual question answering on videos. In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 3480–3491 (2022)
- 44. Yu, L., Lin, Z., Shen, X., Yang, J., Lu, X., Bansal, M., Berg, T.L.: Mattnet: Modular attention network for referring expression comprehension. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1307–1315

(2018)

- 45. Yuanhong, C., Yuyuan, L., Hu, W., Fengbei, L., Chong, W., Gustavo, C.: A closer look at audio-visual semantic segmentation. arXiv preprint arXiv:2304.02970

(2023)

- 46. Zhou, J., Shen, X., Wang, J., Zhang, J., Sun, W., Zhang, J., Birchfield, S., Guo, D., Kong, L., Wang, M., Zhong, Y.: Audio-visual segmentation with semantics. arXiv preprint arXiv:2301.13190 (2023)
- 47. Zhou, J., Wang, J., Zhang, J., Sun, W., Zhang, J., Birchfield, S., Guo, D., Kong, L., Wang, M., Zhong, Y.: Audio-visual segmentation. In: European Conference on Computer Vision (2022)
- 48. Zhou, X., Zhou, D., Hu, D., Zhou, H., Ouyang, W.: Exploiting visual context semantics for sound source localization. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5199–5208 (2023)

## Ref-AVS: Refer and Segment Objects in Audio-Visual Scenes (Supplementary Material)

Yaoting Wang1† , Peiwen Sun2† , Dongzhan Zhou3† , Guangyao Li1 , Honggang Zhang2 , and Di Hu 1,4

1 Gaoling School of Artificial Intelligence, Renmin University of China, China yaoting.wang@outlook.com {guangyaoli,dihu}@ruc.edu.cn 2 Beijing University of Posts and Telecommunications, Beijing, China {sunpeiwen,zhhg}@bupt.edu.cn 3 Shanghai Artificial Intelligence Laboratory, Shanghai, China zhoudongzhan@pjlab.org.cn 4 Engineering Research Center of Next-Generation Search and Recommendation

### A Dataset

#### A.1 More statistics and examples

The histogram in the diagram below represents the distribution of videos containing different categories of target objects. We can observe that the majority of the data is abundant and diverse. A small portion of the data is reserved as an unseen test set, whose categories have never appeared in the training set. Taking a broader view of the object categories, we can see that most objects are visually and acoustically related, with a few instances of silent and static data. We believe the diversity of object categories and quantity of the dataset will benefit future research in this challenging task.

As shown in Fig. 1, most videos in the dataset consist of challenging multisource audio. Language reference expressions containing multimodal cues can assist us in locating objects of interest within audio-visual scenes.

#### A.2 Data collection pipeline

We carefully designed the rules to select the videos in the dataset, as mentioned in Sec.3.2 of the main text. Videos falling into one of the following categories will NOT be selected:

1. Videos with a large number of instances of the same semantics. Such videos often exhibit numerous overlapping instances and audio entanglement, making it challenging to accurately describe each instance, even for humans;

†: Equal contribution. : Corresponding author.

[Figure 11]

###### Fig. 1: Data examples in our dataset Ref-AVS Bench. Use five time-steps for clarity.

[Figure 12]

- Fig. 2: Illustration of the categories distribution for our Ref-AVS dataset, where seen and unseen categories are provided. The object classes were carefully chosen to ensure diversity and capture the distributional characteristics observed in real-world scenarios.

- 2. Videos characterized by extensive editing and camera switching. These videos tend to showcase diverse scenery, causing constant changes in the relative positions of instances and the audio source. Consequently, the expressions are likely to be unclear and objects are more difficult to refer to.
- 3. Non-realistic videos containing synthetic artifacts. Animation and handdrawn videos involve secondary processing of object sound and images as part of their artistic expression.

Following the rules above, we obtain the initial videos for further processing. The overall data flow process is illustrated in Fig. 3.

The process starts with the selection of a pivot frame from the video, which is then used to detect and segment it using a foundation segmentation model. Then, the scene switch detection step is included to identify when the subject or context changes in the video. If a scene switch is detected, the process repeats, starting with the selection of a new pivot frame. This is followed by manual labeling for mask refinement and spread-out tracking for more accurate expression tracking. Moreover, the expression is then augmented, which involves LLM to increase the diversity of the dataset and then split it into train and test sets. The train set is used to train a model, while the test set is used to evaluate the model’s performance. Manual refinement is also applied to the validation set and test set.

The goal of the entire pipeline design process is to incorporate as many efficient frozen and off-the-shelf models as possible to minimize manual annotation costs while improving efficiency and accuracy. However, experienced annotators are still irreplaceable and are assigned to critical stages to ensure quality control and address specific issues such as tracking inefficiency caused by minor scene switches. Finally, manual inspections on the validation (val) and test datasets are conducted by skilled researchers to ensure that they meet the community’s requirements in terms of accuracy and quantity. For all videos in our dataset (train

[Figure 13]

- Fig. 3: Dataset collection pipeline. The pipeline has played a significant role in ensuring the efficiency and cost-effectiveness of the overall process, leading to the successful acquisition of high-quality samples.

& test), we require annotators to select quality masks from the auto-generated candidates. During test set verification, 156 (3.27%) masks are manually corrected with less than 20% pixel, indicating the quality of the mask annotation is acceptable to model training.

#### A.3 More details for expressions

Templates. The initial expressions are generated via pre-defined templates, as shown in Fig. 4. The templates are created by integrating messages from the audio, visual, and temporal dimensions. To avoid confusion, we provide the annotators with several examples so that they can fully understand the templates.

We further define multiple placeholders that should be replaced in the templates, which can help the annotators produce suitable expressions more easily. The placeholders are summarized in Tab. 1. The usage of templates and placeholders enables flexible control and statistical analysis of the expressions, ensuring the quality of the dataset.

In addition to the existing templates, we have granted experienced annotators and researchers access to a limited set of non-template expressions. This small collection of expressions brings higher flexibility beyond the constraints imposed by the used templates.

Expression grade. We have implemented a practical grading system within the expression template, as shown in the third column of 4. This grading system serves a dual purpose: it enables annotators and researchers to evaluate the difficulty of the data, while also establishing a strong foundation for future curriculum learning and related research endeavors.

Uniqueness, necessity, and clarity. As mentioned in Sec.3.3 in the main part, we provide three rules for expression annotation, namely uniqueness, necessity, and clarity. We introduce three cases to demonstrate these rules to the annotators, as shown in Fig. 5, Fig. 6, and Fig. 7, respectively.

##### Supp

|Reference|Type|Grade|Expressions|
|---|---|---|---|
|1. Semantic|Semantic|1|[object]. Example: [sheep]. Explaination: This expression can be used only when there is only one sheep in the video. If there are two sheep, unique representation cannot be achieved, and this template cannot be used.|
|2. Visual<br><br>|Spacial|1|The [object.1] [spacial] the [object.2]. Example: The [man] [on the right of] the [man]. Explaination: It represents the spatial relationship between two objects in a simple manner.|
| |Action|2|The [object.1] [verb] [a_an] [object.2].<br><br>Example1: "The [guitar] [being played by] a [man]." .<br>Example2: "The [man] [holding] a [guitar]." Explaination: The interaction relationship between two objects.<br>|
| | |1|The [object] [verb]. Example: "The [man] [runing]". Explaination: The behavior of the object|
| |Appearance|1|The [surface] [object]. Example: The [red] [car]. Example: The [old] [man]. Explaination: The appearance of the object|
| |Appearance + Action|2|The [person] [look] and [verb] [object.2]. Example: The [man] [in a red T-shirt] and [playing] the [guitar]. Explaination: Both the appearance and behavior of the object.|
| |Spacial + Temporal|1|The [object] appearing through the entire video. Example: The [car] appearing through the entire video. Explaination: The specific object that consistently appears in the video frame.|
| | |1|The [time] [object] appearing in the video. Example: The [first] [woman] appearing in the video. Explaination: An object of category A, A1, appears first in the video, followed by another object of category A, A2. Used to refer to the situation where A1 appears.|
| | |2|The [object.1] appearing [sequence] [a_an] [object.2]. Example: The [car] appearing [before] the [track]. Explaination: An object appears before or after another object.|
| | |2|The [object.1] appearing [duration] than [a_an] [object.2]. Example: The [car] appearing [longer] than a [track]. Explaination: Description of the comparison of screen time.|
|3. Audio<br><br>|Acoustic|1|The sounding [object]. Example: The sounding [handpan] Explaination: If there is a piano and two hand drums. The piano and one of the hand drums produce sound. In this case, the template can refer to the [object] that produces sound. "Handpan" is used to distinguish the instruments in the video, and "sounding" is used to differentiate the hand drum that produces sound. Note: In this scenario, avoid using "the sounding [piano]" because there is only one piano in the image, and the information/clue provided by "sounding" is limited compared to the word "[piano]" (the model might take shortcuts in learning).|
| |Acoustic|2|The sounding object. Example: The sounding object. Explaination: Suitable for locating the object that produces the unique sound among multiple objects of the same class. Note: Unlike the previous example, it does not specify the category of the object.|
| |Acoustic + Action|4|The object making the [volume.sup] sound. Example: The object making the [loudest] sound|
| | | |The object making [volume] sound than the [object.2]. Example: The object making louder sound than the [piano].|
| | | |The object making a sound by [verb]. Example: the [man] making a sound by [sing a song].|
| |Acoustic + Temporal|5|The object valume is [volume] over the time. Example: The object valume is [larger] over the time. Explaination: For example, it can refer to a car approaching from a distance, with the sound getting louder and louder.|
| | | |The object rhythm is [rhythm] over the time. Example: The object rhythm is [faster] over the time.|
| | | |The [time] object to make a sound. Example: The [first] object to make a sound. Explaination: In a concert, if the piano produces sound before the violin, you can use this template, and the object you would choose is the piano.|
| | | |The object that making a sound [sequence] the [object.2]. Example: The object that making a sound [before] the [guitar]. Explaination: Scenario: A person sings first and then plays the guitar. In this case, the object you should focus on is the [person].|
| | | |The object that keeps making sound at all times.|
| | | |The object making [duration] sound duration than the [object.2]. Example: The object making [longer] sound duration than the [man]. Explaination: If the duration of a lady's speech is longer than that of a gentleman, the object you should focus on is the lady.|
| | | |The object making the [duration.sup] sound duration. Example: The object making longest sound duration.|
| | | |The object making [rhythm] rhythm than the [object.2]. The object making faster rhythm than the piano.|
| | | |The object making the [rhythm.sup] rhythm. The object making fastest rhythm than the piano.|
| | | |The object making the [continuity] sound. Example: A person plays the drums with pauses in between - non-continuous. Example: A person plays the violin without pauses in between - continuous.|
|4. AudioVisual|Acoustic + Temporal + Spacial<br><br>|3|The [time] sounding [object] held by a [person]. Example: The [first] sounding [guitar] held by a [boy].|
| | | |The [object.1] [spacial] the [time] sounding [object.2]. Example: The [man] [near] the [last] sounding [instrument].|
| |Acoustic + Temporal + Appearance<br><br>|1|The [surface] [object.1] that makes the sound with rhythm [rhythm] than [object.2]. Example: The [white] [car] that makes the sound with rhythm [faster] than [bus].|
| | | |The [surface] [object.1] that makes [volume] sound than [object.2]. Example: The [yellow] [bus] that makes [lower] sound than [car].|
| |Acoustic + Spacial|5<br><br>|The [object.1] [spacial] the sounding [object.2]. Example: the [man] [near] the sounding [piano].|
| | | |The sounding [object.1] [spacial] the [object.2]. Example: the sounding [piano] [near] the man.|
| |Acoustic + Apperance|3|The [surface] and sounding [object]. Example: The [wearing the glasses] and sounding [man].|
| |Acoustic + Apperance + Spacial| |The [surface] [object.1] [spacial] the sounding [object.2] Example: The [young] [woman] [near] the sounding [piano].|

- Fig. 4: This chart shows the template we offered to the annotators. The third column of the chart shows a comprehensive evaluation of the grading system for expressions.

|Type|Placeholder|sub-label|Semantic|
|---|---|---|---|

###### Table 1: The list of placeholders. The annotators need to choose the proper words to replace the placeholder.

Placeholder Sub-label Semantic [temporal]

[sequence] before, after; [duration] longer, shorter;

[volume] louder, lower; [continuity] intermittent, continuous

[acoustic]

[left-right] on the left of, on the right of; [up-down]

on, under; above, beneath; [distance] near, far; [back-front] in front of, behind;

[spacial]

[color] red, green, ...; [shape] square, circle; [look] in a red t-shirt, wearing glasses, ...;

[appearance]

[action] [verb] running, holding, being held by, ... [object]

[obj] erhu, clock, piano, ...; [person] Specific category, e.g., man, boy, girl, ...;

[Figure 14]

- Fig. 5: Uniqueness: The object referred to by an expression must be unique and an expression cannot refer to multiple objects simultaneously.

#### A.4 Dataset diversity

Fig. 8a visualizes the co-occurrence of objects in our dataset, where we can observe a dense web of connections spanning various categories, such as musical instruments, people, vehicles, etc. The rich combination of categories indicates that our dataset is not limited to a narrow set of scenarios but rather encompasses a broad spectrum of real-life scenes where such objects are likely to naturally appear together. Fig. 8c demonstrates that 56% of the total videos contain two or more objects, while 13% of the total videos contain three or more objects. The distribution of expressions in Fig. 8b shows that expressions that combine two or more types of information account for nearly half.

#### A.5 Video duration

The video length in our dataset is 10s, which is a common setting in most audio-visual localization datasets (e.g., AVSBench, VGG-SS, and AVE) and is

[Figure 15]

- Fig. 6: Necessity: Complex expressions can be used for reference, but each adjective in the sentence should narrow down the scope of target objects, avoiding unnecessary and redundant descriptions of the object.

considered to contain ample audio-visual events. We collect videos from realworld scenes with high diversity to support our task, including various temporal contents (e.g., temporal order and event duration). As for the longer time interval setting, we test our model with a 90-second video, as shown in Fig. 9. Despite being trained on 10s videos, our model exhibits generalization ability to videos of longer duration. Specifically, the output sticks to the object making a louder sound at the moment (piano first and then violin). However, building robust A-V correlations with long-range dependencies can still be challenging, which may be mitigated by grasping global audio-visual information and developing time-aware fusion methods. We leave the exploration for future work.

#### A.6 Dataset ablation

- In Tab. 2, we conduct further analysis for our dataset in terms of expression complexity and dimensions. We use the grade and length to reflect The complexity and difficulty of the expressions. Results in the first and second row of Tab. 2 show that mIoU decreases with harder expressions. As for the dimension, results in the last row of Tab. 2 illustrate that the acoustic subset is harder than the spatial subset. Moreover, the temporal dimension cannot exist alone, and adding temporal references may assist the segment process, indicating that time-aware modules like cached memory may be beneficial to this task.

### B Method

#### B.1 Learning objectives

To enhance the quality of the mask during the model training process, we utilize both the binary cross-entropy loss and dice loss as optimization techniques:

Lmask = λbce · Lbce + λdice · Ldice. (1)

[Figure 16]

- Fig. 7: Clarity: Certain expression templates involve subjective factors, such as “the __ with a louder sound”. The use of such expressions should only occur when the situation is clear enough to avoid ambiguous references.

- Table 2: Dataset ablation and dimension analysis. Higher grade produce lower performance, same as the expression length.

Grade 1+2 3 4 5 mIoU 39.27 37.78 35.31 33.39

Exp. len (l) l <= 4 4 < l <= 8 8 < l <= 12 12 < l mIoU 39.66 37.00 33.74 29.61

Dimension ①spatial ②acoustic ①+temporal ②+temporal mIoU 33.83 26.07 41.01 32.30

Further, we adopt the mask classification loss to compose the final loss: Lseg = Lmask + λcls · Lcls. (2)

The parameters λbce,λdice,λcls in the loss are set to 5, 5 and 2 respectively, following the commonly used setting in mask transformer [6] works.

- B.2 More ablation of our method

- In Tab. 3, we conduct ablation experiments for text and audio encoders and find RoBERTa and VGGish work better.

- Table 3: Ablation of different text encoder and audio encoder. (T) and (A) indicate the modified text and audio encoder.

RoBERTa+VGGish (T)Flan-T5 (T)LanguageBind (A)LanguageBind

mIoU 40.87 35.17 33.96 38.23 F-score 0.581 0.547 0.540 0.561

[Figure 17]

[Figure 18]

- (b) Multimodal cues distribution of expression

[Figure 19]

- (c) Objects over video distribution.

(a) Object co-occurrence map

- Fig. 8: The distribution for both expressions and objects. We include multiple modalities combination and consider multiple objects appear in the same video.

[Figure 20]

- Fig. 9: Ref-AVS on a long video around 90 seconds. In the video, the suonding object is changing along the temporal dimension.

#### B.3 Implementation detail on comparison method

To incorporate missing audio or text modality information into other R-VOS or AVS methods, we employ linear projection to ensure that the newly added modality has the same feature dimensions as the existing modalities. Subsequently, we use an addition operation to combine the missing modality information with the original modality information. The remaining components of these methods remain unchanged in their default settings. To ensure fairness, all methods utilize the same audio encoder (VGGish [15,16]) and language encoder (RoBERTa [8,28]).

[Figure 21]

###### Fig. 10: An example of the multi-instance scene. Text reference can help to disambiguate the real object of interested.

- B.4 Supplementary video and more qualitative cases

The attached file contains an MP4 video comprising seven segments. These segments encompass a variety of scenes (indoor, outdoor, concerts, etc.), sound scenarios (single-source and multi-source), and temporal characteristics (temporally related and unrelated). We utilize our Ref-AVS Bench model to refer to and segment these videos, guided by the corresponding expressions. It is evident that our Ref-AVS model, Expression Enhancing with Multimodal Cues (EEMC), performs well in most cases. However, there are instances where the segmentation results are not as accurate, particularly in complex situations. Nevertheless, the video demo effectively showcases the challenges of the Ref-AVS task and demonstrates the effectiveness of our approach.

A variety of multi-instance scenes are covered in our dataset, including instruments, dialogue, etc. As shown in Fig. 10, our model successfully segments the target guitar. However, multi-instance scenes still pose challenges for our task. As shown in Fig. 11(c), the model generates a partial mask for the man who is not talking. The results illustrate that multi-instance scenes merit further research, especially in complicated scenes such as noisy backgrounds.

We provide more failure cases in Fig. 11(a), (b), (d) to demonstrate the challenge of our dataset. The model can be improved for complex scenery, occlusions, background noise, and wrong output in an all-mute scene.

[Figure 22]

###### Fig. 11: The failure cases for our model. Ref-AVS is a challenging task and many aspects is still in need of exploration.

#### B.5 System latency

The inference speed of our method is 2 FPS on 1×RTX 3090. Pruning and quantization can be used to accelerate time-critical applications.

