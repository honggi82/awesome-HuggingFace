## UniVTG: Towards Unified Video-Language Temporal Grounding

Kevin Qinghong Lin1, Pengchuan Zhang2, Joya Chen1, Shraman Pramanick3, Difei Gao1, Alex Jinpeng Wang1, Rui Yan1, Mike Zheng Shou1

1Show Lab, National University of Singapore 2Meta AI 3Johns Hopkins University

### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

# arXiv:2307.16715v2[cs.CV]18Aug2023

When do the kids dancing?

Video Temporal Grounding (VTG), which aims to ground target clips from videos (such as consecutive intervals or disjoint shots) according to custom language queries (e.g., sentences or words), is key for video browsing on social media. Most methods in this direction develop taskspecific models that are trained with type-specific labels, such as moment retrieval (time interval) and highlight detection (worthiness curve), which limits their abilities to generalize to various VTG tasks and labels. In this paper, we propose to Unify the diverse VTG labels and tasks, dubbed UniVTG, along three directions: Firstly, we revisit a wide range of VTG labels and tasks and define a unified formulation. Based on this, we develop data annotation schemes to create scalable pseudo supervision. Secondly, we develop an effective and flexible grounding model capable of addressing each task and making full use of each label. Lastly, thanks to the unified framework, we are able to unlock temporal grounding pretraining from large-scale diverse labels and develop stronger grounding abilities e.g., zero-shot grounding. Extensive experiments on three tasks (moment retrieval, highlight detection and video summarization) across seven datasets (QVHighlights, Charades-STA, TACoS, Ego4D, YouTube Highlights, TVSum, and QFVS) demonstrate the effectiveness and flexibility of our proposed framework. The codes are available at https://github.com/showlab/UniVTG.

Moment Retrieval

A 10 min long video

[Figure 8]

[Figure 9]

[Figure 10]

Highlight of this birthday video?

Highlight Detection

###### UniVTG

[Figure 11]

[Figure 12]

[Figure 13]

Summary this video by ‘food’?

User

User Intent

Video Summarization

Figure 1: Given a video and a specific user query, UniVTG serves as a general video browsing helper that assists users by returning different scale target clips to support various VTG tasks.

ment retrieval tends to localize consecutive temporal windows (interval-level) by giving natural sentences; highlight detection aims to pick out the key segment with highest worthiness (curve-level) that best reflects the video gist; video summarization collects a set of disjoint shots (pointlevel) to summarize the video, with general or user-specific queries. Despite task-specific datasets [10, 5, 47, 46] and models [67, 64, 57] have been developed, these tasks are typically studied separately. In general, these tasks share a common objective of grounding various scale clips based on customized user queries, which we refer to as Video Temporal Grounding (VTG).

Though these tasks are closely related, their relationship has not been explicitly studied until recently. [21] introduces the first unified benchmark QVHighlights for moment retrieval and highlight detection, and presents the first model Moment-DETR for jointly learning. On this basis, UMT [27] expands audio inputs, and QD-DETR [30] develops negative-pairs and saliency tokens. Nevertheless, these studies solely focus on designing models that intersect two subtasks and learn grounding capabilities rely on specific labels. This means that they lack the ability to generalize the VTG across diverse temporal labels, such as unique point-level narrations in Ego4D [13]. Furthermore, we have witnessed promising progress in Vision-Language Pretraining (VLP). One notable work is GLIP [24, 65], which develops a unified model via joint utilizing large-scale diverse image annotations such as image captions and bounding boxes for spatial grounding. However, we do not observe similar progress in video-language pretraining. Most works in this area are designed for video-level tasks such as videotext retrieval [55, 48] rather than temporal grounding. This

### 1. Introduction

With the increasing interest in sharing daily lives, video has emerged as the most informative yet diverse visual form on social media. These videos are collected in a variety of settings, including untrimmed instructional videos [29], and well-edited vlogs [19]. With massive scales and diverse video forms, automatically identifying relevant moments based on user queries has become a critical capability in the industry for efficient video browsing.

This significant demand has given rise to a number of video understanding tasks, including moment retrieval [67, 64, 31], highlight detection [53, 16, 57], and video summarization [14, 46, 43]. As depicted in Fig. 1, mo-

: Corresponding Author.

[Figure 14]

[Figure 15]

[Figure 16]

is largely due to the manual cost of fine-grained temporal annotations is expensive, making it challenging to obtain open-source, scalable yet diverse annotations to support grounding pretraining along the temporal axis in videos.

(a) Point-wise

(b) Interval-wise (c) Curve-wise

| | |
|---|---|
|[Figure 17]<br><br>[Figure 18]<br><br>Ego<br><br>[Figure 19]<br><br>[Figure 20]<br><br>TS|4D<br><br>[Figure 21]<br><br>[Figure 22]<br><br>MR|

| | |
|---|---|
|[Figure 23]<br><br>[Figure 24]<br><br>QV<br><br>[Figure 25]<br><br>[Figure 26]|HL<br><br>[Figure 27]<br><br>[Figure 28]<br><br>HL|
|MR| |

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

QFVS

###### Charades

###### TVSum

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

VS

HL

MR

Therefore, we see a clear motivation to pursue a Unified VTG framework and propose our UniVTG, which aims to unify diversity in VTG along three directions: (i) From the label and task aspect, we first define a formulation for VTG where each video is decomposed as a clip sequence that each clip is assigned three basic query-conditional elements. Such a formulation enables us to unify various VTG labels and tasks under the same framework. Moreover, to address the limitation of temporal labels, we propose a data annotation scheme based on CLIP [37] to produce scalable fine-grained pseudo labels. (ii) From the model aspect, we develop a flexible yet effective grounding model that inherits the principles of our formulation. Our model devises single-stream and dual-stream pathways for modality fusion and modality alignment respectively, and is equipped with three heads to decode three key elements. This favorable design is capable of addressing each task and utilizing each label. (iii) Lastly, thanks to the unified framework and the availability of pseudo labels, we can perform large-scale temporal grounding pretraining across various labels to enhance our grounding abilities. This empowers us to address various VTG downstream tasks across multiple domains, including zero-shot inference.

[Figure 41]

(d) UniVTG

[Figure 42]

[Figure 43]

VS Video Summarization MR Moment Retrieval HL Highlight Detection TS Timestamp Narrations

[Figure 44]

[Figure 45]

Pretraining by scalable labels.

Transfer to downsteam tasks.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Free-form Language Query

Figure 2: Diverse VTG labels can be divided into three types, each mainly associated with specific benchmarks: (a) point-level labels for video summarization [43] and timestamp narrations [13]; (b) interval-level labels for moment retrieval [13, 10, 21]; (c) curvelevel labels for highlight detection [46, 21]. (d) UniVTG unifies diverse labels and tasks within one framework, enabling largescale pretraining with diverse labels (dotted gray line) that can be transferred to various downstream tasks (solid green line).

### 2. Related Work

##### 2.1. Video Temporal Grounding

We review three VTG tasks: moment retrieval, highlight detection, and video summarization, and compare them as different variations of a common problem.

Moment Retrieval aims to localize target moments i.e., one [10] or many [21] continuous intervals within a video by a language query, as shown in Fig. 2 (b). Previous methods fall into two categories: proposal-based and proposalfree. The proposal-based methods [2, 10, 67] employ a two-stage process of scanning the entire video to generate candidate proposals, which are then ranked based on their matching to the text query. In contrast, the proposal-free methods [7, 62, 11, 64, 31] learn to regress the start and end boundaries directly without requiring proposal candidates. Our UniVTG borrows from proposal-free approaches but extends it by incorporating diverse temporal labels and tasks with a concise design.

To validate the effectiveness of our proposed framework, we conduct experiments not only on joint moment retrieval and highlight detection benchmark (QVHighlights [21]), but also on three individual tasks for moment retrieval (Ego4D [13], Charades-STA [10], TACoS [39]), highlight detection (YouTube Highlights [47], TVSum [46]) and video summarization (QFVS [43]). Our UniVTG, one unified model with 4.2M samples for temporal grounding pretraining, has achieved remarkable results, outperforming state-of-the-art methods that are specifically tailored for each task. Overall, our contributions are four folds:

Highlight Detection aims to assign a worthiness score to each video segment e.g., Fig. 2 (c), and then return the top highest scoring segment as the highlight. Previous highlight detection datasets [41, 47, 46] tend to be domainspecific and query-agnostic, in which many efforts [15, 53, 16, 57, 3] treat this task as a visual or visual-audio scoring problem. Nevertheless, video highlights typically have a theme, which is often reflected in the video titles [46] or topics [47] e.g., “surfing”. Recently, [21] proposes a joint moment retrieval and highlight detection benchmark QVHighlights that enables users to produce various highlights for one video conditional on different text queries.

- • To the best of our knowledge, our UniVTG is the first video temporal grounding pretraining that across varied domains and tasks, including moment retrieval, highlight detection and video summarization.
- • We introduce a unified VTG framework that can fully leverage rich supervision from open-source, scalable yet diverse temporal annotations, such as point-level, interval-level, and curve-level labels.
- • To address the limitations of pretraining corpus, we develop an efficient annotation method that uses CLIP as a teacher to produce scalable pseudo temporal labels.
- • We demonstrate the effectiveness and flexibility of the proposed framework across four settings and seven datasets. Detailed ablation studies validate the superiority of the proposed components.

Video Summarization aims to summarize the whole video by a set of shots to provide a quick overview e.g., Fig.2 (a), which contains two forms: Generic video summarization [14, 46, 28, 17] that captures the important scene using visual clues merely, while Query-focused video summariza-

###### Moment Retrieval

#### Interval-wise UniVTG

[Figure 50]

𝑴𝟏 𝑴𝟐

- 0.9M Pesudo

Curve-wise

- 1.5M Pesudo Point-wise

Sentence: Woman are typing.

###### Unified Formulation

Saliency 𝒔𝒊

###### Boundary offsets

###### Highlight Detection

Unified Grounding Model

(i) Converting (ii) Pretraining

𝒅𝒊𝒆

(iii) Transferring

𝒔𝒊 ∈ 𝑻𝒐𝒑𝟏

𝒗𝒊

Foreground 𝒇𝒊

𝒅𝒊𝒔

Video title: Office days

Free-form Language query Q

###### Video Summarization

(a) (b)

𝑴 ≤ 𝜶%|𝑽|

1.8M Manual

Keyword: Phone

Figure 3: Illustration of UniVTG pipeline. (i) Given any kind of labels, such as interval label, we first convert it into our (a) unified formulation (§ 3.1) by deriving other two labels (point and curve labels). (ii) Once we have collect a large-scale diverse labels (§ 3.2), we leverage them to pretrain a unified grounding model (§ 4). (iii) Next, the unified model is transferred to various VTG downsteam tasks e.g., highlight detection.

conduct VLP with grounding as a pretraining task. This way eliminates the need for additional grounding models and enables zero-shot grounding capacity.

tion [43, 33, 50] that allows users to customize the summary by specifying text keywords (e.g., tree and cars). The latter is closer to practical usage hence we focus on it. Recently, IntentVizor [50] proposes an interactive approach allowing users to adjust their intents to obtain a superior summary.

### 3. Towards Unified VTG: Tasks and Labels

In general, each of the three tasks represents a specific form of VTG that grounds different scales of clips from videos (e.g., a consecutive clip set, a single clip or a disjoint clip set) by offering customized text queries (e.g., sentences, titles or keywords). However, previous methods address some subtasks solely. Based on this insight, our goal is to develop a unified framework to handle all of them.

The UniVTG pipeline is displayed in Fig. 3. In this section, we start by introducing the unified formulation.

- 3.1. Unified Formulation Given a video V and a language query Q, we first divide

V into a sequence of Lv fixed-length clips {v1,··· ,vL

v}, where each clip vi is of length l and has a centered timestamp ti. The free-form text query Q has Lq tokens, denoted as Q = {q1,··· ,qL

q}. We then define three elements for each clip vi = (fi,di,si), described as follows:

- • Foreground indicator fi ∈ {0,1}: a binary value indicating whether the i-th clip vi belongs to the foreground or not. If clip vi is the foreground of Q, then fi = 1, otherwise fi = 0.
- • Boundary offsets di = [dsi,dei] ∈ R2: the temporal distance that converts the clip timestamp ti to its inter-

val boundaries. Here, di is valid when fi = 1. The dsi is the distance between the starting of the interval and

ti, whereas dei is the distance between the ending and ti. Thus, the whole temporal interval bi of vi can be represented as bi = [ti − dsi,ti + dei]

- • Saliency score si ∈ [0,1]: a continuous score determining the relevance between the visual content of clip vi and the query Q. If the clip and query are highly correlated, si = 1; If they are totally irrelevant, then si = 0. Notably, it is reasonable to assume that si > 0 if a clip is in the foreground of Q, otherwise si = 0.

In Fig.3 (a), we draw a schematic diagram to represent these three elements of clip vi in our definition.

- 3.2. Revisiting Various VTG Tasks and Labels Treating clips as the atom composition of a video, we

- 2.2. Vision-Language Pretraining The emergence of large-scale vision-language datasets,

such as [44, 42, 29, 4], has paved the way for the development of VLP [37, 23, 20, 35, 22] to enhance video-text representation for various vision-language tasks [61, 55, 54]. The representative CLIP [37] has shown that image-level visual representations can be effectively learned using largescale noisy image-text pairs. Furthermore, GLIP [24, 65] makes an effort along the spatial axis, which leverages various image annotations, such as image labels, captions, and bounding boxes, to develop strong region-level understanding capacity for spatial grounding tasks. However, due to the expensive manual cost of fine-grained temporal-level annotations i.e., temporal bounding box, this grounding pretraining has not been extended to the temporal axis in videos, limiting its progress to match the spatial counterparts. To address this limitation, we explore alternative approaches that leverage accessible timestamp narrations [13] and derive pseudo supervision as the pretraining corpus.

On the other hand, there are several efforts have been made to perform temporal-friendly video pretraining [1, 56, 6, 63] to pursue a better video representation for grounding tasks. But the resulting pretraining model still requires an additional grounding model such as 2D-TAN [67] to perform video grounding. In contrast, powered by our unified framework and scalable pseudo annotations, we can directly

define the VTG problem as collecting a target clip set M =

{vi ∈ V |Q} from V , conditional on language query Q. We next illustrate how to extend this definition to various tasks and labels. Especially, for each label, we answer:

𝒗𝟎 𝒗𝟏 𝒗𝟐 𝒗𝟑 𝒗𝟒

||Ancient rome<br><br>Tour guide<br><br>…<br><br>Gladiator<br><br>Top-5|
|---|
<br><br>Doctor<br><br>Water<br><br>(a) Concept Bank|
|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- 1. How to collect scalable label corpus for pretraining?
- 2. When using the unified formulation, how can we obtain unknown elements based on the available one?
- 3.2.1 Moment Retrieval and Interval-wise Label. Moment retrieval aims to localize one [10] or many [21] intervals in a video corresponding to a sentence Q. As shown in Fig. 3 (Right blue), moment retrieval aims to select m consecutive clip sets M = M1 ∪ ··· ∪ Mm, where m ≥ 1, and Mj is the j-th target moment. M can be simplified as the boundary set of foreground clips {bi|fi = 1}.

CLIP Image Enc.

(b) Derive 𝒔𝒊 based on CLIP similarity

𝒔𝒊 = 𝒄𝒐𝒔(𝒗𝒊,𝑸)

𝑸

CLIP Text Enc.

Thresholding with τ

(c) Derive (𝒇𝒊,𝒅𝒊) based on 𝒔𝒊

𝒅𝟏 𝒅𝒌

𝒇𝟏 = 𝟏 𝒇𝟐 = 𝟎 𝒇𝟑 = 𝟏

𝝉

Figure 4: Process of using CLIP to produce temporal labels. (a) We first use a concept bank to cover diverse open-world concepts. (b) Next, we use CLIP as teacher to calculate the clip-level scores between each concept to get top-5 concepts as video gist, and treat their clip scores as saliency si. (c) Based on si, we further derive the interval and point labels via thresholding.

The temporal interval with specific target boundaries is a common label for moment retrieval. However, annotating intervals requires manually reviewing the full video, which is expensive. A solution is ASR [29, 58] that provide start and end timestamps, but ASR is often too noisy and poorly aligned with the visual content, making it suboptimal. Here, we sought an alternative solution. We found that visual captions [44, 4] tend to be descriptive, making them well-suited as grounding queries, thus if we can know how these videos are cut from the raw source, we can use this information to create pseudo intervals. We find that VideoCC [32] is a viable option for this purpose. It is worth noting that VideoCC is initially developed for video-level pretraining (e.g., power video-text retrieval), and we are the pioneer to investigate its potential in temporal grounding pretraining.

select top-5 concepts as the video gist, and save their CLIP similarities as pseudo curve labels, i.e., Fig. 4 (b).

As shown in Fig. 4 (c), after obtaining curve labels, we assign fi = 1 for clips with si greater than a threshold τ, otherwise fi = 0. The τ is estimated based on the similarity of each video, refer to Supp. for details. The offsets di are defined as the distance between the foreground clip and its nearest neighboring clips where fi = 0.

###### 3.2.3 Video Summarization and Point-wise Label.

Query-focused video summarization [43] aims to summarize the entire video with a set of shots to provide a quick overview, with user-specific concepts (for example, trees and cars). The generated summary should be succinct while representative of the entire video around the given query. We define this task by regarding keywords as Q, and select a set of clips M = {vi|fi = 1}, where the size of M is required to not exceed α% of the original video length |M| ≤ α%|V | e.g., α = 2%.

Once we obtain intervals, we convert interval labels into the proposed formulation by defining fi = 0 and si = 0 for clips that are not in target interval, and we assign fi = 1 and assume si > 0 for clips that belongs to the target interval.

- 3.2.2 Highlight Detection and Curve-wise Label. Highlight detection aims to assign an importance score to each video clip (making its annotations like a curve), then return the few highest-scoring clips as the highlight, where queries may [21] or may not [47, 46] be provided as input. For video highlighting datasets without language queries, we can use video titles [46] or video domain name [47] as Q because they are highly related to the topic of the video. Then, this task is equivalent to picking clips with the top

The annotations in QFVS [43] are point labels that indicate whether each shot belongs to the concept or not. The cost of point labels is much cheaper than that of interval and curve labels since people only need to glance at a specific time. The recently Ego4D [13] dataset uses this point labeling to annotate massive-scale data by assigning a narration to an exact timestamp, such as “I am opening the washingmachine” at ti = 2.30 sec. Due to the favorable scale, it is natural to adapt them for large-scale pretraining. Recently, there have been attempts to improve video-text representation using point-wise annotations to improve the video-text representation [25, 68, 36] and augment NLQ [13] baselines [38]. Despite this, these methods mainly focus on transferring within the same domain.

highest saliency scores i.e. M = {vi|si ∈ top-K}.

Due to the interestingness contain subjectivity, the same video usually needs to be labeled by several people to eliminate bias. This makes curve labels the most expensive yet informative temporal annotations. Therefore, we are motivated to find an efficient way of producing scalable curve labels. Intuitively, interestingness reflects how each clip is relevant to the video gist. As depicted in Fig. 4 (a), we first define a concept bank using an open-world detection class list [42]. Next, we use CLIP as a teacher to get the cliplevel cosine similarities between each concept. Then, we

For point labels, we derive si > 0 if clip fi = 1, otherwise si = 0. During pretraining, we estimate its temporal label bi based on the average distance between consecutive narrations within the video [25, 38, 36].

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Indicator 𝒇 Oﬀsets 𝒅 Saliency 𝒔

- 4. Towards Unified VTG: Model We here introduce our unified model which seamlessly inherits our proposed unified formulation.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Conv Conv Dot Product

- 4.1. Overview As shown in Fig. 5, our model mainly comprises a frozen

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Feed forward

[Figure 77]

A . Pooler

×k

video encoder, a frozen text encoder, and a multi-modal encoder. The video and text encoders are keep consistent with Moment-DETR [19], which employs the concatenation of CLIP [37] (ViT-B/32) and SlowFast [9] (R-50) features as video representation, and use the CLIP text encoder [37] to extract token level features. Our multi-modal encoder contains k self-attention blocks that followed by three specific heads to decode the prediction.

[Figure 78]

[Figure 79]

Mul -head Self-A en on

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Feed Forward Feed Forward

Given an input video V with Lv clips and a language query Q with Lq tokens, we first apply the video encoder and the text encoder to encode the video and text respectively, then project them to the same dimension D by two Feed-Forward Networks (FFN), and thus obtain video features V = {vi}L

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

❄ ❄

Video Encoder Text Encoder

𝑉 𝑄

A view of a bamboo fountain of water in a tea house and people scoop from and wash oﬀ

[Figure 94]

[Figure 95]

[Figure 96]

v×D and text features Q = {qj}Lj=1q ∈ RL

i=1 ∈ RL

v

Figure 5: Unified grounding model contains a video encoder, a text encoder, and a multi-modal encoder followed by three output heads, corresponding to three key elements f ˜i, d˜i, s˜i . Besides, our model has two pathways: one for cross-modal interaction (solid red line) and the other for cross-modal alignment (broken orange line).

q×D. Next, we design two pathways for cross-modal alignment and cross-modal interaction.

(i) For cross-modal alignment, we first adopt an attentive pooling operator to aggregate the query tokens Q ∈ RL

q×D into a sentence representation S ∈ R1×D. Especially,

S = AQ, (1) where the weight A = Softmax(WQ) ∈ R1×L

Foreground head for Matching. Taking the output V˜ k ∈ RL

q and W1×L

v×D from the multi-modal encoder, this head applies three 1×3 Conv layers, each with D filters and followed by a ReLU activation. Finally, sigmoid activations are attached to output the prediction f˜i per clip. We use the binary crossentropy loss as a training objective.

q is a learnable embedding. Then V and S are sent to perform contrastive learning (described in § 4.2).

(ii) For cross-modal interaction, learnable position embeddings Epos and modality-type embeddings Etype are added to each modality to retain both positional and modality information:

Lf = −λf fi log f˜i + (1 − fi)log 1 − f˜i . (4)

V˜ = V + EposV + EtypeV , Q˜ = Q + EposT + EtypeT .

Boundary head for Localization. The design of this head is similar to the foreground head except for the last layer, which has 2 outputs channel for the left and right offsets. Taking the V˜ k ∈ RL

(2)

v×D, this head outputs offsets {d˜i}L

Next, the text and video tokens are concatenated and get a joint input Z0 = [V˜ ;Q˜ ] ∈ RL×D, where L = Lv + Lq. Further, Z0 is fed into the multi-modal encoder, which contains k transformer layers with each layer consisting of a Multi-headed Self-Attention and FFN blocks.

v

i

per clip. Then, we devise the predicted boundary b˜i and use the combination of smooth L1 loss [12] and generalized IoU loss [40] as our training objectives.

i=1 λL1LSmoothL1 d ˜i,di + λiouLiou ˜bi,bi .

Lb = f

Zd = MLP MSA Zd−1 , d ∈ {1...k}. (3) We take the video tokens V˜ k ∈ RL

(5)

Notably, this regression objective is only devised for foreground clips i.e., fi = 1.

v×D from the multimodal encoder Em as output Zk = [V˜k;Q˜k] ∈ R, and feed Zk into the following heads for prediction.

Saliency head for Contrasting. Since we define saliency as the relevance between visual context and text query, it is natural to interpret this score as a similar measurement between video and text modalities. Taking the video tokens V = {vi}L

- 4.2. Pretraining Objectives To match the previous unified formulation i.e.,

v×D and sentence representation S ∈ R1×D, we define the predicted saliency score s˜i between

(fi,di,si), we devise three different heads to decode each element respectively, each one calling a capability.

i=1 ∈ RL

v

clip vi and text query Q as their cosine similarities:

viTS ∥vi∥2∥S∥2

, (6)

s˜i = cos(vi,S) :=

where ∥ · ∥2 represents the L2-norm of a vector.

For each video V, we randomly sample a foreground clip vp with fp = 1 and sp > 0 as a positive sample; we treat other clips in the same video vj with saliency sj less than sp as negative samples, i.e., Ω = {j|sj < sp,1 ≤ j ≤ Lv}, and perform intra-video contrastive learning:

exp (˜sp/τ) exp (˜sp/τ) + j∈Ω exp (˜sj/τ)

Lintras = − log

, (7)

where τ is a temperature parameter and set as 0.07.

Besides, we regard sentences from other samples within batches k ∈ B as negative samples, and develop the intervideo contrastive learning for cross-sample supervision:

exp(˜sp/τ) k∈B exp s ˜kp/τ

Linters = −log

, (8)

where B is the training batch size and s˜kp = cos(vi,Sk).

Our saliency score head training loss is the combination of inter- and intra-video contrastive learning:

Ls = λinterLinters + λintraLintras . (9)

To this end, our total training objective is the combination of each head loss overall clips in the training set.

N

1 N

(Lf + Lb + Ls), (10)

L =

i=1

where N is the clip number of the training set.

##### 4.3. Inference

During inference, given a video V and a language query Q, we first feed forward the model to obtain {f˜i,˜bi,s˜i}L

v

i=1 for each clip vi from three heads. Next, we describe how we carry out output for individual VTG tasks respectively. Moment Retrieval. We rank clips predicted boundaries {˜bi}L

i=1 based on their {f˜i}L

i=1 probabilities. Since the predicted Lv boundaries are dense, we adopt a 1-d Non-Max Suppression (NMS) with a threshold 0.7 to remove highly overlapping boundary boxes, yielding a final prediction.

v

v

Highlight Detection. For each clip, to fully utilize the foreground and saliency terms, we rank all clips based on their {f˜i + s˜i}L

i=1 scores, and then return the few top clip (e.g., Top-1) as predictions.

v

Video Summarization. Using the same preprocessing settings [43, 52], the videos are first divided as multiple segments via KTS algorithm [34]. Then the clip scores from each segment are computed, and these scores are integrated. We rank all clips based on their foreground {f˜i}L

i=1 and return the Top-2% clips as a video summary.

v

Dataset Task Pseudo? Label # Samples Domain

Ego4D [13] PT ✗ Point 1.8M Egocentric VideoCC [32] PT ✓ Interval 0.9M Web CLIP teacher PT ✓ Curve 1.5M Open

QVHighlights [19] MR + HL ✗ Interval + Curve 10.3K VLog, News NLQ [13] MR ✗ Interval 15.1K Egocentric Charades-STA [10] MR ✗ Interval 16.1K Indoor TACoS [39] MR ✗ Interval 18.2K Kitchens YoutubeHL [47] HL ✗ Curve 600 Web TVSum [46] HL ✗ Curve 50 Web QFVS [43] VS ✗ Point 4 Egocentric

Table 1: Dataset statistics. The upper side datasets are used for pretraining (PT) which cover three label types, two of which are pseudo. The lower side datasets are used for downstream tasks (MR: Moment Retrieval, HL: Highlight Detection, VS: Video Summarization).

### 5. Experiments

In this section, we conduct experiments on various benchmarks to evaluate our approach. Mainly, we design the experiments to study the following questions:

- Q1: How much improvement could be made by UniVTG grounding pretraining?
- Q2: What are the effects of using different pretraining corpus from various labels?
- Q3: Is it necessary to use the proposed unified formulation and unified model?

More ablation studies can be found in Supplementary.

##### 5.1. Datasets and Settings

We have summarized the dataset information in Tab.1. For pretraining, we gather 1.8M point labels from Ego4D and 0.9M interval labels from VideoCC [32]. For curve labels, we apply CLIP teacher method (Fig. 4) to Ego4D and VideoCC datasets to get 1.5M pseudo labels. Therefore, a total of 4.2M temporal annotations are used for grounding pretraining. For downstream tasks, we assess our methods on four VTG tasks across seven datasets, spanning (i) Jointly moment retrieval and highlight detection; (ii) Moment Retrieval; (iii) Highlight Detection; (iv) Video Summarization. Additional details are listed in Supp.

Evaluation Metrics. For QVHighlights, we follow official [21], Recall@1 with IoU thresholds 0.5 and 0.7, mean average precision (mAP) with IoU thresholds 0.5 and 0.75, and the average mAP over a series of IoU thresholds [0.5:0.05:0.95] are used for moment retrieval. For highlight detection, mAP and HIT@1 are used, a clip is treated as a true positive if it has the saliency score of Very Good. For Charades-STA, NLQ, TACoS, Recall@1 with IoU thresholds 0.3, 0.5 and 0.7, and mIoU are used. For YouTube Highlights and TVSum, we follow [27] and use mAP and Top-5 mAP, respectively. For QFVS, we follow [50] that reports F1-score per video as well as an average.

Implementation Details. We set k = 4 multi-modal transformer encoder layers, with d = 1024 hidden size and 8 attention heads. The drop path rates are 0.1 for trans-

Moment Retrieval HD Method R1 mAP ≥ Very Good

@0.5 @0.7 @0.5 @0.75 Avg. mAP HIT@1

BeautyThumb [45] − − − − − 14.36 20.88 DVSE [26] − − − − − 18.75 21.79 MCN [2] 11.41 2.72 24.94 8.22 10.67 − − CAL [8] 25.49 11.54 23.40 7.65 9.89 − − CLIP [37] 16.88 5.19 18.11 7.0 7.67 31.30 61.04 XML [21] 41.83 30.35 44.63 31.73 32.14 34.49 55.25 XML+ [19] 46.69 33.46 47.89 34.67 34.90 35.38 55.06

MDETR [19] 52.89 33.02 54.82 29.40 30.73 35.69 55.60 MDETR w/ PT 59.78 40.33 60.51 35.36 36.14 37.43 60.17 UMT†[27] 56.23 41.18 53.83 37.01 36.12 38.18 59.99 UMT† w/ PT 60.83 43.26 57.33 39.12 38.08 39.12 62.39

UniVTG 58.86 40.86 57.60 35.59 35.47 38.20 60.96 UniVTG w/ PT 65.43 50.06 64.06 45.02 43.63 40.54 66.28 UniVTG ZS 25.16 8.95 27.42 7.64 10.87 35.96 53.50

- Table 2: Jointly Moment Retrieval and Highlight Detection results on QVHighlights test split1. †: introduce audio modality. w/ PT: fine-tuning after pre-training; ZS: zero-shot inference.

former layers and 0.5 for input FFN projectors. During the pretraining stage, our experiments are carried out on 8 A100 GPUs. When it comes to downstream tasks, we use one GPU. For moment retrieval, all baselines and UniVTG use the same video and text features. For highlight detection and video summarization, we report results following [27] and [50]. See Supp. for more details.

5.2. Comparison with State-of-the-arts (Q1)

5.2.1 Joint Moment Retrieval and Highlight Detection As illustrated in Tab. 2, we first evaluate our UniVTG on QVHighlights test split: (i) Without pretraining, UniVTG has shown comparable performance to two joint optimization counterparts Moment-DETR [19] and UMT [27], demonstrating its superior model design for joint task optimization. (ii) With large-scale pretraining, UniVTG exhibits a significant improvement on all metrics, such as +8.16 Avg. mAP and +5.32 HIT@1. As a result, UniVTG surpasses all baselines by a large margin. Notably, UMT introduces audio modality and ASR pretraining [27],

1Codalab QVHighlights Evaluation

but it is still worse than us by Avg. mAP of 5.55 and HIT@1 of 3.89. (iii) Due to the large-scale pretraining, UniVTG can perform zero-shot grounding and outperforms several supervised baselines without any training samples.

- 5.2.2 Moment Retrieval

- In Tab. 3, we compare the results of our method and the mainstream moment retrieval methods on three widely used benchmarks. (i) Similar to the observation made by QVHighlights, without pretraining, we find that UniVTG is still superior to other compared methods. This demonstrates once more the effectiveness of our concise architecture. (ii) Large-scale grounding pretraining has resulted in significant improvements, leading to a considerable increase in the mIoU i.e., +2.97 in NLQ, +2.07 in Charades-STA, and

+5.03 in TACoS. (iii) Notably, in NLQ, our zero-shot result has outperformed all the baselines methods due to the close pretraining domain. However, it is worth mentioning that the zero-shot performance on TACoS is inferior. This could be because the videos have scenes that are very similar to each other, with only small spatial variations, making it difficult to effectively apply zero-shot methods.

5.2.3 Highlight Detection

- In Tab. 4 and Tab. 5, we conduct highlight detection experiments on YouTube Highlights and TVSum respectively, where the baselines with † (rows 6-9) are incorporate with audio features. We observe that (i) grounding pretraining brings improvement on UniVTG and surpasses all baselines in Avg. mAP. (ii) In TVSum, gain discrepancy among domains may stem from its small scale (50 samples) and scoring subjectivity. In contrast, the larger YouTube dataset (600 videos) yields more consistent pretraining gains. (ii) Moreover, in zero-shot setting, UniVTG beats several video-only baselines such as [47, 49].

- 5.2.4 Video Summarization

In Tab. 6, we present the QFVS benchmark results. Our pretrained UniVTG achieves a 0.8% higher Avg. F1-score than IntentVizor [50], where the latter is an interactive method and being tailored for the video summarization task. This result demonstrates the generalization of our method on video summarization task.

Method

NLQ [13] Charades-STA [10] TACoS [39]

R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7 mIoU

2D TAN [67] 4.33 1.83 0.60 3.39 58.76 46.02 27.50 41.25 40.01 27.99 12.92 27.22 VSLNet [64] 4.54 2.40 1.01 3.54 60.30 42.69 24.14 41.58 35.54 23.54 13.15 24.99 MDETR [19] 4.34 1.81 0.65 3.53 65.83 52.07 30.59 45.54 37.97 24.67 11.97 25.49

UniVTG 7.28 3.95 1.32 4.91 70.81 58.01 35.65 50.10 51.44 34.97 17.35 33.60 UniVTG w/ PT 11.74 7.54 3.25 7.88 72.63 60.19 38.55 52.17 56.11 43.44 24.27 38.63 UniVTG ZS 6.48 3.48 1.16 4.63 44.09 25.22 10.03 27.12 5.17 1.27 0.27 4.40

- Table 3: Moment Retrieval results on NLQ, Charades-STA, and TACoS benchmarks. All baselines use the same video features (CLIP ViT-B/32 and SlowFast R-50) and text features (CLIP text enc.). w/ PT means fine-tuning after pre-training; ZS means zero-shot inference.

Method Dog Gym. Par. Ska. Ski. Sur. Avg. RRAE [59] 49.0 35.0 50.0 25.0 22.0 49.0 38.3 GIFs [15] 30.8 33.5 54.0 55.4 32.8 54.1 46.4 LSVM [47] 60.0 41.0 61.0 62.0 36.0 61.0 53.6 LIM-S [53] 57.9 41.7 67.0 57.8 48.6 65.1 56.4 SL-Module [57] 70.8 53.2 77.2 72.5 66.1 76.2 69.3 MINI-Net† [16] 58.2 61.7 70.2 72.2 58.7 65.1 64.4 TCG† [60] 55.4 62.7 70.9 69.1 60.1 59.8 63.0 Joint-VA† [3] 64.5 71.9 80.8 62.0 73.2 78.3 71.8 UMT†[27] 65.9 75.2 81.6 71.8 72.3 82.7 74.9 UniVTG 71.8 76.5 73.9 73.3 73.2 82.2 75.2 UniVTG w/ PT 74.3 79.0 74.4 84.9 75.1 83.9 78.6 UniVTG ZS 36.8 62.8 65.9 39.2 64.5 54.0 53.9

Method VT VU GA MS PK PR FM BK BT DS Avg. sLSTM [66] 41.1 46.2 46.3 47.7 44.8 46.1 45.2 40.6 47.1 45.5 45.1 SG [28] 42.3 47.2 47.5 48.9 45.6 47.3 46.4 41.7 48.3 46.6 46.2 LIM-S [53] 55.9 42.9 61.2 54.0 60.4 47.5 43.2 66.3 69.1 62.6 56.3 Trailer [49] 61.3 54.6 65.7 60.8 59.1 70.1 58.2 64.7 65.6 68.1 62.8 SL-Module [57] 86.5 68.7 74.9 86.2 79.0 63.2 58.9 72.6 78.9 64.0 73.3 MINI-Net† [16] 80.6 68.3 78.2 81.8 78.1 65.8 57.8 75.0 80.2 65.5 73.2 TCG† [60] 85.0 71.4 81.9 78.6 80.2 75.5 71.6 77.3 78.6 68.1 76.8 Joint-VA† [3] 83.7 57.3 78.5 86.1 80.1 69.2 70.0 73.0 97.4 67.5 76.3 UMT†[27] 87.5 81.5 88.2 78.8 81.5 87.0 76.0 86.9 84.4 79.6 83.1 UniVTG 83.9 85.1 89.0 80.1 84.6 81.4 70.9 91.7 73.5 69.3 81.0 UniVTG w/ PT 92.0 77.8 89.8 83.8 82.2 85.8 74.3 91.8 90.5 77.6 84.6 UniVTG ZS 78.5 67.0 75.3 63.6 67.0 66.8 35.4 85.3 83.1 50.0 67.2

- Table 4: Highlight Detection results of mAP on YouTube HL. † denotes using audio modality.

###### Table 5: Highlight Detection results of Top-5 mAP on TVSum. † denotes using audio modality.

###### Method V1 V2 V3 V4 Avg.

56

- 38
- 39
- 40
- 41
- 42

QC-DPP [43] 48.68 41.66 36.51 29.96 44.19 CHAN [52] 49.14 46.53 58.65 33.42 46.94 QSAN [51] 48.52 46.64 56.93 34.25 46.59 WHM [33] 50.96 48.28 58.41 39.18 49.20 IntentVizor [50] 51.27 53.48 61.58 37.25 50.90

52

mAPAvg.

R@10.7

48

44

40

0% 25% 50% 75%100%

0% 25% 50% 75%100%

UniVTG 52.54 54.48 56.73 40.37 51.03 UniVTG w/ PT 49.85 56.97 59.35 40.62 51.70

(a) R1@0.7 of Moment Retrieval.

(b) mAP Avg. of Highlight Detection.

Figure 6: Effect of pretraining scale on QVHighlights dataset.

- Table 6: Video Summarization results of F-score on QFVS.

5.3. Ablation Studies

Effect of different labels for pretraining (Q2). In Tab. 7 top half, we investigate the effect of different labels corpus for pretraining. The results here are before unified formulation i.e., the original label provided by the pretraining set. Our findings (rows 1-4) indicate that (i) incorporating any type of label for pretraining yields considerable performance gains on most benchmarks. (ii) Combining all three types of data (row 5) for pretraining further boost the outcomes, such as +5.2 MR’s mAP and +1.1 HL’s mAP over baseline (row 1) on QVHighlights.

Effect of unified formulation (Q3). In Tab. 7 bottom half, we further study the impacts of unified formulation i.e.,

Pretraining Corpus Unified Labels? QVHighlights TACoS YouTube row Ego4D VideoCC CLIP

Point Interval Curve

MR HL MR HL

Point Interval Curve mAP mAP mIoU mAP

- 1 36.13 38.83 33.60 75.15

- 2 ✓ ✓ 39.89 39.48 35.33 75.32

- 3 ✓ ✓ 39.81 39.75 35.11 74.76

- 4 ✓ ✓ 39.16 39.80 35.68 75.44

- 5 ✓ ✓ ✓ ✓ ✓ ✓ 41.37 39.97 35.87 75.66

- 6 ✓ ✓ ✓ ✓ 41.53 39.66 36.52 75.27

- 7 ✓ ✓ ✓ ✓ 40.96 40.10 36.78 76.10

- 8 ✓ ✓ ✓ ✓ 42.19 40.43 35.85 77.48

- 9 ✓ ✓ ✓ ✓ ✓ ✓ 45.99 41.25 38.63 78.56

- Table 7: Ablation studies on pretraining corpus. ✓ denotes the elements derived by us, which are not provided in vanilla training corpus: Ego4D, VideoCC, and CLIP teacher.

the benefits of deriving unknown elements for pretraining. From rows 2-4 vs rows 6-8, We find that (i) training corpora receive performance gains in most settings, which proves that the label converting methods are crucial for better utilizing temporal labels. (ii) Among all settings, curve labels appear to be the most effective ones, and beat the manual point labels except in a few domains e.g., TACoS. (iii) We get the optimal result (row 9) by using full three converted corpus for pretraining, with 4.62 MR’s mAP and 1.28 HL’s mAP increase over counterparts (row 5) on QVHighlights.

Effect or pretraining scale. In Fig. 6, we explore the effect of utilizing various scales of labels for pretraining. We observe a steady performance improvement on both moment retrieval and highlight detection tasks as the training sample size increases. It also shows that unifying labels to construct a large training corpus can greatly benefit the VTG.

### 6. Conclusion

This paper introduces UniVTG, a framework that unifies diverse VTG tasks and labels by addressing three key challenges: (i) We define a unified formulation for VTG to convert various labels and tasks under a single framework, and propose a label scaling scheme. (ii) We develop an effective yet flexible model that can handle various VTG tasks and training labels. (iii) Due to the unified framework and availability of scalable labels, it becomes feasible to perform large-scale temporal grounding pretraining over diverse labels. We demonstrate the effectiveness and flexibility of our UniVTG on four settings across seven datasets, spanning joint optimization as well as individual tasks.

- (a) QVHighlights: Vlog and News domains, videos are average 2.5 minutes long; Each video might have several intervals

[Figure 97]

[Figure 98]

- (b) Charades-STA: Indoor domains, most videos are less than 1 minutes.

[Figure 99]

- (c) Natural Language Queries: Egocentric domain, videos are 8-20 minutes.

[Figure 100]

[Figure 101]

- (d) TACoS: Kitchen domain, videos are average 4.8 minutes.

[Figure 102]

- Figure 7: Visualization of Joint moment retrieval and highlight detection on (a) QVHighlights, and Moment Retrieval on

9

(b) Charades-STA, (c) Ego4D, (d) TACoS. Textual queries are mostly natural sentences.

- (e) TVSum: Web diverse domain, videos are average 4.2 minutes long.

[Figure 103]

[Figure 104]

- (f) YouTube Highlights: Youtube diverse domain, videos are average 1.5 minutes long.

[Figure 105]

[Figure 106]

- (g) Query-Focused Video Summarization: Egocentric domain, each video is between 3-5 hrs.

[Figure 107]

[Figure 108]

- Figure 8: Visualization of Highlight Detection on (e) TVSum, (f) YouTube Highlights; and Video Summarization on (g) QFVS. Textual queries can be video title (e), video domain (f), and keywords (g).

### 7. Acknowledgements

This project is supported by the National Research Foundation, Singapore under its NRFF Award NRF-NRFF132021-0008, the DSO National Laboratories, Mike Zheng Shou’s Start-Up Grant from NUS. The computational work for this article was partially performed on resources of the National Super computing Centre, Singapore.

### References

- [1] Humam Alwassel, Silvio Giancola, and Bernard Ghanem. Tsp: Temporally-sensitive pretraining of video encoders for localization tasks. In ICCV, pages 3173–3183, 2021.
- [2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, pages 5803–5812, 2017.
- [3] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Joint visual and audio learning for video highlight detection. In ICCV, pages 8127–8137, 2021.
- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, pages 1728–1738, 2021.
- [5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, pages 961–970, 2015.
- [6] Meng Cao, Tianyu Yang, Junwu Weng, Can Zhang, Jue Wang, and Yuexian Zou. Locvtp: Video-text pretraining for temporal localization. In ECCV, pages 38– 56, 2022.
- [7] Jingyuan Chen, Xinpeng Chen, Lin Ma, Zequn Jie, and Tat-Seng Chua. Temporally grounding natural sentence in video. In EMNLP, pages 162–171, 2018.
- [8] Victor Escorcia, Mattia Soldan, Josef Sivic, Bernard Ghanem, and Bryan Russell. Temporal localization of moments in video collections with natural language. arXiv preprint arXiv:1907.12763, 2019.
- [9] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, pages 6202–6211, 2019.
- [10] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, pages 5267–5275, 2017.
- [11] Soham Ghosh, Anuva Agarwal, Zarana Parekh, and Alexander G. Hauptmann. Excl: Extractive clip localization using natural language descriptions. In NAACL-HLT, pages 1984–1990, 2019.

- [12] Ross Girshick. Fast r-cnn. In ICCV, pages 1440–1448, 2015.
- [13] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022.
- [14] Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. Creating summaries from user videos. In ECCV, pages 505–520, 2014.
- [15] Michael Gygli, Yale Song, and Liangliang Cao. Video2gif: Automatic generation of animated gifs from video. In CVPR, pages 1001–1009, 2016.
- [16] Fa-Ting Hong, Xuanteng Huang, Wei-Hong Li, and Wei-Shi Zheng. Mini-net: Multiple instance ranking network for video highlight detection. In ECCV, pages 345–360, 2020.
- [17] Hao Jiang and Yadong Mu. Joint video summarization and moment localization by cross-task sample transfer. In CVPR, pages 16388–16398, 2022.
- [18] Yong Jae Lee, Joydeep Ghosh, and Kristen Grauman. Discovering important people and objects for egocentric video summarization. In CVPR, pages 1346–1353, 2012.
- [19] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. In NeurIPS, pages 11846–11858, 2021.
- [20] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In CVPR, pages 7331–7341, 2021.
- [21] Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. Tvr: A large-scale dataset for video-subtitle moment retrieval. In ECCV, pages 447–463, 2020.
- [22] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.
- [23] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In NeurIPS, 2021.
- [24] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In CVPR, 2022.

- [25] Kevin Qinghong Lin, Alex Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Difei Gao, Rongcheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. In NeurIPS, 2022.
- [26] Wu Liu, Tao Mei, Yongdong Zhang, Cherry Che, and Jiebo Luo. Multi-task deep visual-semantic embedding for video thumbnail selection. In CVPR, pages 3707–3715, 2015.
- [27] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multimodal transformers for joint video moment retrieval and highlight detection. In CVPR, pages 3042–3051, 2022.
- [28] Behrooz Mahasseni, Michael Lam, and Sinisa Todorovic. Unsupervised video summarization with adversarial lstm networks. In CVPR, pages 202–211, 2017.
- [29] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV, pages 2630–2640, 2019.
- [30] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23023–23033, 2023.
- [31] Jonghwan Mun, Minsu Cho, and Bohyung Han. Local-global video-text interactions for temporal grounding. In CVPR, pages 10810–10819, 2020.
- [32] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In ECCV, pages 407–426, 2022.
- [33] Saiteja Nalla, Mohit Agrawal, Vishal Kaushal, Ganesh Ramakrishnan, and Rishabh Iyer. Watch hours in minutes: Summarizing videos with user intent. In ECCV, 2020.
- [34] Danila Potapov, Matthijs Douze, Zaid Harchaoui, and Cordelia Schmid. Category-specific video summarization. In ECCV, pages 540–555, 2014.
- [35] Shraman Pramanick, Li Jing, Sayan Nag, Jiachen Zhu, Hardik Shah, Yann LeCun, and Rama Chellappa. Volta: Vision-language transformer with weaklysupervised local-feature alignment. arXiv preprint arXiv:2210.04135, 2022.
- [36] Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. Egovlpv2: Egocentric video-language pre-training with fusion

- in the backbone. arXiv preprint arXiv:2307.05463, 2023.
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [38] Santhosh Kumar Ramakrishnan, Ziad Al-Halah, and Kristen Grauman. Naq: Leveraging narrations as queries to supervise episodic memory. In CVPR, pages 6694–6703, 2023.
- [39] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Trans. Assoc. Comput. Linguistics, 1:25–36, 2013.
- [40] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In CVPR, pages 658– 666, 2019.
- [41] Yong Rui, Anoop Gupta, and Alex Acero. Automatically extracting highlights for tv baseball programs. In MM, pages 105–115, 2000.
- [42] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, pages 8430–8439, 2019.
- [43] Aidean Sharghi, Jacob S Laurel, and Boqing Gong. Query-focused video summarization: Dataset, evaluation, and a memory network based approach. In CVPR, 2017.
- [44] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, pages 2556–2565, 2018.
- [45] Yale Song, Miriam Redi, Jordi Vallmitjana, and Alejandro Jaimes. To click or not to click: Automatic selection of beautiful thumbnails from videos. In CIKM, 2016.
- [46] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In CVPR, pages 5179–5187, 2015.
- [47] Min Sun, Ali Farhadi, and Steve Seitz. Ranking domain-specific highlights by analyzing edited videos. In ECCV, pages 787–802, 2014.
- [48] Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training.

- In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6598– 6608, 2023.
- [49] Lezi Wang, Dong Liu, Rohit Puri, and Dimitris N Metaxas. Learning trailer moments in full-length movies with co-contrastive attention. In ECCV, pages 300–316, 2020.
- [50] Guande Wu, Jianzhe Lin, and Claudio T Silva. Intentvizor: Towards generic query guided interactive video summarization. In CVPR, pages 10503–10512, 2022.
- [51] Shuwen Xiao, Zhou Zhao, Zijian Zhang, Ziyu Guan, and Deng Cai. Query-biased self-attentive network for query-focused video summarization. IEEE Trans. Image Process., 29:5889–5899, 2020.
- [52] Shuwen Xiao, Zhou Zhao, Zijian Zhang, Xiaohui Yan, and Min Yang. Convolutional hierarchical attention network for query-focused video summarization. In AAAI, volume 34, pages 12426–12433, 2020.
- [53] Bo Xiong, Yannis Kalantidis, Deepti Ghadiyaram, and Kristen Grauman. Less is more: Learning highlight detection from video duration. In CVPR, pages 1258– 1267, 2019.
- [54] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In MM, pages 1645– 1653, 2017.
- [55] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, pages 5288–5296, 2016.
- [56] Mengmeng Xu, Juan-Manuel P´erez-R´ua, Victor Escorcia, Brais Martinez, Xiatian Zhu, Li Zhang, Bernard Ghanem, and Tao Xiang. Boundary-sensitive pre-training for temporal localization in videos. In ICCV, pages 7220–7230, 2021.
- [57] Minghao Xu, Hang Wang, Bingbing Ni, Riheng Zhu, Zhenbang Sun, and Changhu Wang. Cross-category video highlight detection via set-based learning. In ICCV, pages 7970–7979, 2021.
- [58] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning, 2023.
- [59] Huan Yang, Baoyuan Wang, Stephen Lin, David Wipf, Minyi Guo, and Baining Guo. Unsupervised extraction of video highlights via robust recurrent autoencoders. In ICCV, pages 4633–4641, 2015.
- [60] Qinghao Ye, Xiyue Shen, Yuan Gao, Zirui Wang, Qi Bi, Ping Li, and Guang Yang. Temporal cue guided

- video highlight detection with low-rank audio-visual fusion. In ICCV, pages 7950–7959, 2021.
- [61] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Trans. Assoc. Comput. Linguistics, 2:67–78, 2014.
- [62] Runhao Zeng, Haoming Xu, Wenbing Huang, Peihao Chen, Mingkui Tan, and Chuang Gan. Dense regression network for video grounding. In CVPR, pages 10287–10296, 2020.
- [63] Can Zhang, Tianyu Yang, Junwu Weng, Meng Cao, Jue Wang, and Yuexian Zou. Unsupervised pretraining for temporal action localization tasks. In CVPR, pages 14031–14041, 2022.
- [64] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. In ACL, pages 6543–6554, 2020.
- [65] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, YenChun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and visionlanguage understanding. In NeurIPS, 2022.
- [66] Ke Zhang, Wei-Lun Chao, Fei Sha, and Kristen Grauman. Video summarization with long short-term memory. In ECCV, pages 766–782, 2016.
- [67] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In AAAI, volume 34, pages 12870–12877, 2020.
- [68] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In CVPR, 2023.

### Appendix of UniVTG

from the above VideoCC subset videos. Below, we describe the benchmarks used for the four settings separately.

##### A. CLIP teacher strategy

###### (i) Joint Moment Retrieval and Highlight Detection.

QVHighlights [19] is the only dataset with available annotations for both moment retrieval and highlight detection, making it an ideal choice for benchmarking multi-task joint optimization. This dataset contains 10,148 videos with an average length of 150 sec that covers daily vlogs, travel vlogs, and news events scenarios. There are a total of 10,310 queries associated with 18,367 moments (on average, 1.8 disjoint moments per query in the video).

The concept bank is a class list for open-world detection, sourced from here2. This list comprises 19,995 class names, such as ”Sandwich Cookies,” ”Air conditioning,” and ”Advertising.” After conducting a manual check, we determined that the class list can effectively encompass the majority of common concepts.

In our approach, we begin by capturing frame-level clip image features from the video at a rate of 2 fps. Following this, we calculate their respective similarity scores in relation to the given class list. We then determine top-5 classes with the highest average scores, representing the most significant concepts within the video.

###### (ii) Moment Retrieval. We utilize three bench-

marks to further evaluate moment retrieval: CharadesSTA [10], Ego4D Natural Language Queries (NLQ) [13] and TACoS [39]. (a) Charades-STA contains 16,128 indoor videos with an average length of 30.6 sec, which are made up of 12,408 query-interval pairs for training and 3,720 query-interval pairs for testing. (b) NLQ focuses on daily egocentric scenarios, where videos are 8 − 20 minutes long and queries are question, e.g.“What did i pour in the bowl?”, making this benchmark challenging. The training set contains 11.3K annotated queries from 1K videos, whereas the validation set contains 3.9K queries from 0.3K videos. (c) TACoS contains 127 videos with an average duration of 4.78 minutes, where 75 videos are used for training, 27 and 25 videos for validation and testing, respectively.

s 0.15 0.10 0.05

- 0

Find the maximum e.g., 0.17

3

- 2

1

- 0

Discrete the curve by dividing 0.05

Threshold as 3

Get intervals using thresholding

3 2

- 1 0

###### (iii) Highlight Detection. We utilize two benchmarks

to further evaluate highlight detection: YouTube Highlights [47] and TVSum [46]. (a) YouTube Highlights has 6 domains with 433 videos, where video titles are not provided, thus we use the domain name of each video as text queries. (b) While TVSum includes 10 domains, each with 5 videos, we use their video titles as text queries. We follow [27] data splits that the ratio of training:testing is 0.8:0.2.

Figure 9: Demonstration of how to threshold each video’s curve.

To derive intervals from the curve obtained from the diverse distributions, a fixed threshold is hard to determined and lacks the flexiblity. Thus, we discretize the continuous curve by a small value of 0.05 and pick the maximum discrete value as our threshold. Then, adjacent clips that share the maximum discrete value to form an interval. In this way, we may produce multiple temporal windows from one video. This process is shown in Fig. 9.

###### (iv) Video Summarization. We utilize the QFVS [43]

benchmark to evaluate the video summarization. This dataset includes the four videos in UT Egocentric dataset [18]. Each video is recorded in daily life and lasts between 3 − 5 hours. Each query in this dataset is represented by two words from a total of 48 pre-defined concepts.

##### C. Experimental settings

##### B. Datasets

- (i) In Tab. 8, we detail the parameters for each set-

ting. Notably, for highlight detection benchmarks YouTube Highlights and TVSum, which contain multiple domains treated as separate splits, we perform parameters tuning for λintra within each domain. Then we aggregate the results obtained using optimal settings. The optimal settings are listed in Tab. 9-10.

- (ii) During training, to maintain the balance between

positive and negative samples, we allocate a weight of 0.1 to the negatives (fi = 0) in binary cross-entropy loss Eq. 4.

- (iii) When inferring highlights scores, we observe that

Pretraining corpus. To establish our pretraining corpus, we collect data through three ways: For point labels, we extract the timestamped narrations from Ego4D [13] by excluding the NLQ val / test splits. For interval labels, we select a subset of videos (less than 300K) sourced from VideoCC 3, and treat their start and end timestamp as windows and caption as query. For curve labels, we derive them

- 2https://storage.googleapis.com/openimages/v6/

oidv6-class-descriptions.csv

- 3https://github.com/google-research-datasets/

videoCC-data

Type Datasets l BS Epoch Warmup LR Weight dacay Gamma LR drop λSmoothL1 λiou λf λintra λinter Pretraining 4.2M corpus 2 64 10 - 1e−4 1e−4 - - 10 1 10 0.1 0.1 Joint MR & HL QVHighlights 2 32 200 10 1e−4 1e−4 0.1 80 10 1 10 0.05 0.01

NLQ 2 32 200 10 1e−5 1e−5 0.1 100 10 1 50 0.1 1.0 Charades-STA 1 32 100 10 1e−5 1e−5 0.1 100 10 1 10 1.0 0.5 TACoS 2 32 100 10 1e−4 1e−4 0.1 30 10 1 10 0.5 0.1

Moment Retrieval

YouTube Highlights 1† 4 100 10 1e−4 1e−4 - - 0 0 1 Search 0 TVSum 2 4 200 10 1e−4 1e−4 - - 0 0 1 Search 0

Highlight Detection

Video Summarization QFVS 5 20∗ 20 0 5e−5 5e−5 - - 0 0 1 0.9 0

- Table 8: Parameter selections for each settings where l denotes the clip length; BS denotes the batch size; LR denotes the learning rate; LR drop denotes the learning rate drop up epoch; Warmup denotes the warmup epoch. Search denotes to parameter searching individually for each domain. † means YouTube Highlights clips has overlapping frames, which is align with the [27]. ∗ means batchsize in QFVS is based on the segment-level instead of video-level.

Domains Dog Gyn Par. Ska. Ski. Sur. λintra 0.6 0.5 0.4 0.5 0 0.7

- Table 9: Optimal λintra under each domain in the Youtube HL.

Domains BK BT DS FM GA MS PK PR VT VU λintra 0.7 0.9 0.6 0.4 0.1 0.1 0 0.6 0.1 0.5

- Table 10: Optimal λintra under each domain in the TVSum.

for each segment within a video, then aggregate these scores to derive an overall video score which is used to compute the metrics. We calculate the conceptual similarity between each two video clip based on the intersection-over-union (IOU) of their related concepts. This conceptual similarity is then used as edge weights in a bipartite graph between two summaries, which aids in identifying the maximum weight match in the graph. Finally, precision, recall, and F1 scores can be determined based on the matching pairs.

{f˜i + s˜i}L

i=1 can typically achieves better performance in QVHighlights, while for smaller datasets YouTube Highlights and TVSum, using f˜i yield more reliable prediction.

v

(iv) For video summarization, we adhere to the same preprocessing settings in [52], which extracts video frame features at 1 FPS and take a 5 seconds as a clip and compute the average frame feature within a clip to generate its cliplevel feature. By applying the KTS algorithm [34], we split a long video into small segments under the conditions that the number of segments in a video is no more than 20 and each segment contains no more than 200 clips.

During evaluation, we compute the foreground scores f˜i

##### D. Ablation studies of training objective

Since we use identical training objectives during the stages of pretraining and downstream transferring. To gain a more thorough understanding of the impact each component has, we have constructed ablation studies as seen in Tab. 11, where the top half, we study the effect of downstream training objectives (without introduce any pretraining), while in the bottom half, we investigate the effect of pretraining training objectives (the downstream tuning use the same optimal parameter settings).

Pretraining Downstream MR@QVHL HL@QVHL MR@NLQ MR@TaCoS

Lf LSmoothL1 Liou Linters Lintras Lf LSmoothL1 Liou Linters Lintras R1@0.5 mAP mAP HIT@1 R1@0.3 mIoU R1@0.3 mIoU

✓ ✓ 54.71 29.64 33.12 46.13 5.96 3.97 48.46 30.20 ✓ ✓ ✓ 58.71 35.89 33.21 45.03 6.50 4.43 50.09 32.42 ✓ ✓ ✓ ✓ 59.16 36.24 38.59 61.81 6.97 4.88 51.14 33.05 ✓ ✓ ✓ ✓ ✓ 59.74 36.13 38.83 61.81 7.28 4.91 51.44 33.60

✓ ✓ ✓ ✓ ✓ ✓ 62.00 39.45 39.59 64.00 8.83 5.82 52.04 32.72 ✓ ✓ ✓ ✓ ✓ ✓ ✓ 63.29 40.43 39.82 64.19 8.49 5.73 51.71 34.76 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 64.52 41.65 39.93 63.68 8.49 5.74 53.11 34.48 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 64.45 41.84 40.07 64.32 9.86 6.52 53.89 36.76 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 68.39 45.99 41.25 67.42 11.74 7.88 56.11 38.63

###### Table 11: Ablation studies of downstream (top) and pretraining objective (bottom) on QVHighlights val split, NLQ val split and TACoS val split.

##### E. Parameters sensitivity

Transformer layers. In Tab. 12, we abalate the transformer layers L ∈ [1,2,3,4,6,8] of multi-modal encoder in our unified model (without pretraining).

MR HD R1@0.5 mAP mAP HIT@1

# Layers

- 1 47.16 26.62 37.35 60.65

- 2 55.25 30.70 38.33 60.52

- 3 59.03 34.06 38.57 62.13

- 4 59.74 36.13 38.83 61.81 6 61.55 39.88 39.20 63.42 8 60.32 38.24 38.72 60.90

Table 12: Ablation studies of different transformer layers for multi-modal encoder on QVHighlights val split.

Projector dimension. In Fig. 10, we study the effect of projector dimension from 256 to 1024 (without pretraining).

48

46

Avg.mAP

44

42

40

38

256 512 768 1,024

68

66

HIT@1

64

62

60

256 512 768 1,024

(a) Avg. mAP of moment retrieval.

(b) HIT@1 of highlight detection.

Figure 10: Ablation studies of projector dimension on QVHighlights val split.

##### F. Loss weights

In Tab. 13, we study the effect of foreground loss on three moment retrieval benchmarks (with pretraining).

QVHighlights NLQ TACoS

λf

R1@0.5 mAP R1@0.3 mIoU R1@0.3 mIoU

0.1 66.97 46.02 9.24 6.64 46.51 33.16 0.5 66.19 46.08 9.50 6.75 50.21 35.06

1 67.74 46.22 9.53 6.80 51.79 35.94 5 67.35 45.63 9.89 6.88 54.01 37.59

10 67.81 45.46 7.26 7.36 54.44 37.55 25 68.00 45.06 11.41 7.77 54.31 37.27 50 66.71 44.32 11.13 7.49 54.21 35.61

Table 13: Ablation studies of foreground loss weight λf on QVHighlights, NLQ, and TACoS moment retrieval benchmarks.

##### G. Visualizations

In Fig. 7 and 8, we show quantitative visualizations of UniVTG predictions across different settings and domains.

