## EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone

# arXiv:2307.05463v2[cs.CV]19Aug2023

Shraman Pramanick1,2† Yale Song2 Sayan Nag3 Kevin Qinghong Lin4 Hardik Shah2 Mike Zheng Shou4 Rama Chellappa1 Pengchuan Zhang2

1Johns Hopkins University, 2Meta AI, 3University of Toronto, 4National University of Singapore

### Abstract

Video-language pre-training (VLP) has become increasingly important due to its ability to generalize to various vision and language tasks. However, existing egocentric VLP frameworks utilize separate video and language encoders and learn task-specific cross-modal information only during fine-tuning, limiting the development of a unified system. In this work, we introduce the second generation of egocentric video-language pre-training (EgoVLPv2), a significant improvement from the previous generation, by incorporating cross-modal fusion directly into the video and language backbones. EgoVLPv2 learns strong video-text representation during pre-training and reuses the cross-modal attention modules to support different downstream tasks in a flexible and efficient manner, reducing fine-tuning costs. Moreover, our proposed fusion in the backbone strategy is more lightweight and compute-efficient than stacking additional fusion-specific layers. Extensive experiments on a wide range of VL tasks demonstrate the effectiveness of EgoVLPv2 by achieving consistent state-of-the-art performance over strong baselines across all downstream. Our project page can be found at https://shramanpramanick.github.io/EgoVLPv2/.

### 1. Introduction

Video-Language Pre-training (VLP) has proven to be the de-facto solution for a variety of video-text tasks, e.g., videotext retrieval [107, 73, 4], VQA [104, 116, 127], zero-shot recognition, [8, 56, 36] and video-text grounding [68, 58]. This is fueled by recent advances in vision [17, 60, 6, 4, 2, 22, 61] and language [92, 16, 59, 112, 81, 14, 80], coupled with large-scale data [107, 126, 66, 4, 27, 15]. Existing video-language datasets generally fall under two categories: third-person view and first-person view (egocentric). The noticeable domain gap between them restricts VLP frame-

†Part of this work was done during an internship at Meta AI.

QFVS (avg F-score)

EgoMCQ (in

EK-100

|(intra-vid. acc.)<br><br>57.2<br><br>60.9<br><br>23.8 18.8|MIR (mAP)<br><br>45.0<br><br>49.7 47.3<br><br>52.1|
|---|---|
|CharadesEgo (mAP)<br><br>32.1 34.1 65.6<br><br>68.2|EgoTaskQA (acc.)<br><br>59.4<br><br>32.7<br><br>37.9|

EgoNLQ (R@5 IoU@0.3)

EK-100 MIR (nDCG)

61.9

Cha (

A

EgoMQ (R@5 IoU@0.3)

EgoVLPv2

EgoVLP [57]

Figure 1: EgoVLPv2 achieves the state-of-the-art performance across a broad range of egocentric video understanding tasks (see Table 1 for details) among similar-sized baselines by incorporating cross-modal attention in the transformer backbones to learn video-language representation.

works pre-trained on third-person videos from performing well on egocentric benchmarks [57]. However, the recent introduction of a massive-scale egocentric dataset Ego4D [27] helps unlock the full potential of egocentric VLP.

Existing egocentric VLP approaches [57, 125, 67, 3] pre-train separate (dual) video and language encoders and learn task-specific cross-modal information only during fine-tuning, limiting the development of unified egocentric VL frameworks. Moreover, they lack strong zeroshot inference ability on multi-modal downstream tasks. This issue is commonly addressed by stacking dedicated fusion layers on top of the dual video and text encoders [64, 44, 105, 90, 109, 110, 117], or with a shared videolanguage architecture [48, 1, 41, 91, 94]. However, these approaches introduce a large number of fusion-specific pa-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(a) Dual Encoders (b) Stacked Fusion Layers (c) Shared Encoders (d) Fusion in the Backbone (Ours)

- Figure 2: Four categories of VLP frameworks. (a) use separate (dual) video and text backbones, with InfoNCE [71] as the common pretraining objective [57, 125, 3, 67] (b) use cross-modal fusion layers on top of dual encoders, with MLM, VTM, etc. as common pretraining tasks [64, 44, 105, 90] (c) use a single encoder for different modalities, with similar learning objectives as (b) [48, 1, 41] (d) Fusion in the Backbone (Ours).

rameters, and the resulting encoder cannot be directly applied to uni-modal (video-only) tasks.

In this work, we present the second generation of egocentric VLP (EgoVLPv2), a significant improvement over the previous generation [57] by incorporating cross-modal fusion directly into the video and language backbones. Our approach improves over existing VLP frameworks by: (i) fewer fusion parameters compared to stacked fusion-specific transformer layers or shared encoders, requiring less GPU memory, compute resources, and training time; (ii) the flexibility to switch between dual and fusion encoders, by turning on and off cross-attention fusion using a gating mechanism; (iii) being applicable to both uni- and multi-modal tasks.

Inserting cross-modal fusion directly into the backbone helps unify a wide range of dual- and fusion-encoder-based downstream tasks. Specifically, the “switching” ability of EgoVLPv2 enables us to utilize the same pre-trained encoders for fast retrieval and grounding tasks, which require dual and fusion encoders, respectively. Moreover, in contrast to existing egocentric VLP frameworks that learn taskspecific fusion parameters during fine-tuning, EgoVLPv2 reuses the pre-trained cross-attention modules across different tasks, significantly reducing the fine-tuning cost. This enables us to introduce query-focused video summarization as a downstream task, which has recently gained attention in the community [69, 100, 101, 34, 102, 70]. The scarcity of annotated data has been a bottleneck to training decent-sized models end-to-end on this task, with the only available egocentric dataset, QFVS [84], providing merely 135 video-query training samples. EgoVLPv2 achieves new state-of-the-art results on QFVS with a decent margin over the baselines.

In summary, our contributions are: (i) We advance a step forward in egocentric VLP by proposing EgoVLPv2, the second generation of EgoVLP [57] with cross-modal fusion in the backbone. Our proposed framework can switch between dual and fusion encoders and requires 45% lesser compute (GMACs) than learning additional fusion-specific transformer layers. (ii) The switching capability of EgoVLPv2 allows us to unify a wide range of dual- and fusion-encoder-

based downstream tasks under the same VLP framework and reduce the task-specific fine-tuning cost by employing the same pre-trained cross-attention modules across different video-language tasks. (iii) We demonstrate the effectiveness of EgoVLPv2 on eight egocentric benchmarks and achieve state-of-the-art performance among comparable-sized backbones. We summarize these results in Figure 1.

### 2. Related Works

#### 2.1. VLP Frameworks

Video-language pre-training (VLP) has attracted increasing attention in recent years, following the success of imagelanguage pre-training [78, 46, 33, 18, 5, 11, 63, 52, 19, 118, 111, 113, 76, 53, 95, 98, 30, 96, 72, 45] and their applications [10, 24, 29, 50, 77]. There are three broad categories of VLP frameworks (see Figure 2):

Dual Encoders: Many existing egocentric VLP frameworks [57, 125, 3, 67] falls into this category. They use separate video and language backbones and learn task-specific crossmodal fusion during fine-tuning [4, 65, 106, 93]. They are commonly trained using InfoNCE [71] or MIL-NCE [65] objectives, and have been successful in video-text retrieval.

Shared Encoder: Approaches that learn a combined encoder for video and text fall under this category [48, 1, 41, 91, 94]. They are modality independent and can be applied to an image, video, text, audio, time-series, and single-view 3D data. Common learning objectives include masked language modeling [16, 127], masked frame modeling [89, 127], masked token modeling [105], masked modal modeling [64, 105], sentence ordering modeling [43], frame ordering modeling [43, 47], and video-text matching [43].

Encoders with Stacked Fusion Layers: This line of work uses dedicated cross-modal fusion layers on top of dual encoders [64, 44, 105, 90, 109, 110, 117], trained using similar objectives as shared encoders.

The latter two categories introduce a large number parameters for cross-modal fusion. In this work, we propose a fourth category (Figure 2 (d)) by inserting cross-modal fusion in uni-modal backbones using a gating mechanism.

EgoNCE Head

MLM Head

VTM Head

|Space-Time Self-Att<br><br>FFN<br><br>Cross-Att|
|---|

M Unfused Layers

M Unfused Layers

M Fused Layers

M Fused Layers

M Fused Layers

M Fused Layers

FFN

FFN

FFN

FFN

FFN

FFN

|Temp. Self-Att<br><br>Spatial Self-Att|
|---|

Cross-Att

Cross-Att

Cross-Att

Cross-Att

Cross-Att Cross-Att

Space-Time Self-Att

Space-Time Self-Att

Space-Time Self-Att

Self-Att

Self-Att

Self-Att

(NL-M) Unfused Layers

(NV-M) Unfused Layers

(NV-M) Unfused Layers

(NL-M) Unfused Layers

(NV-M) Unfused Layers

(NL-M) Unfused Layers

FFN

FFN

FFN

FFN

FFN

FFN

Space-Time Self-Att

Space-Time Self-Att

Space-Time Self-Att

Magnified view of kth Fused TimeSformer Layer

Space-Time Self-Att

Self-Att

Self-Att

Self-Att

#C C stands near a <MASK> and <MASK> the phone.

[Figure 5]

[Figure 6]

#C C picks up a brick from the ground. Masked Caption

[Figure 7]

[Figure 8]

#C C stands near a wheelbarrow and scrolls the phone.

Hard-negative Caption

#C C stands near a wheelbarrow and scrolls the phone.

- Figure 3: Computation of three objectives, LEgoNCE, LMLM, and LVTM. We insert cross-modal fusion into uni-modal backbones with a gating mechanism. During pre-training, every forward iteration contains three steps: (i) cross-attention modules are switched off, EgoVLPv2 acts as dual encoder, LEgoNCE is computed. (ii) cross-attention is switched on, EgoVLPv2 acts as fusion encoder, and video-masked narration pair is fed into EgoVLPv2 to compute LMLM (iii) crossattention is kept on, hard-negative video-narration pairs are fed into EgoVLPv2 to compute LVTM. This fusion in the backbone strategy results in a lightweight and flexible model compared to using fusion-specific transformer layers.

Our framework is flexible to act as either dual or shared encoders by switching cross-attention modules off and on.

- 2.2. Video-Language Datasets

The success of VLP can be partially attributed to the availability of large-scale open-world video-text datasets such as ActivityNet [39], WebVid-2M [4], and HowTo100M [66]. These datasets comprise videos sourced from the Web, and are paired with the corresponding ASR captions, making them popular for VLP pre-training. Despite their impressive size, these existing video-text pretraining datasets typically feature 3rd-person views. On the other hand, egocentric videos has received increasing interests from the community. Previous egocentric datasets [15, 86, 55, 82, 74] were smallscale and domain-specific. The recently released Ego4D [27] is the first massive-scale egocentric dataset consisting of 3670 hours of videos collected by 931 people from 74 locations across 9 different countries world-wide. Recently, EgoClip [57] offered a filtered version of Ego4D with variable-length clip intervals instead of single timestamps. We train our proposed framework, EgoVLPv2, on the EgoClip version of Ego4D.

- 3. EgoVLPv2

- 3.1. Fusion in the Backbone

We use TimeSformer [6, 4] and RoBERTa [59] as our video and language backbones. However, such separate

(dual) uni-modal encoder design does not capture crossmodality interaction and, thus, fails to produce fine-grained multi-modal representation. Existing VLP frameworks achieve cross-modal fusion by: (i) learning a shared architecture [48, 1, 41, 91, 94] or stack fusion layers on top of dual encoders [64, 44, 105, 90, 109, 110, 117], or (ii) learning cross-modal fusion during fine-tuning [57, 125, 3, 67, 4, 65, 106, 93]. While the former offers superior cross-modal representation and zero-shot inference ability on multi-modal downstream tasks, they introduce a large number of fusion parameters than the latter. In this work, we insert crossmodal fusion into the top few layers of uni-modal backbones to strike a balance between the two ideas.

Figure 3 shows the architecture of EgoVLPv2. Each TimeSformer encoder layer has a divided space-time attention module containing temporal and spatial self-attentions with residual connections. The output of space-time attention at kth encoder layer, z(k), can be expressed as:

xˆ(vidk) = xvid(k−1) + TEMP-SA(x(vidk−1)) z(k) = xvid(k−1) + SPA-SA(ˆx(vidk))

= SPACE-TIME(xvid(k−1)) (1)

where x(vidk−1) is the output of the (k − 1)th encoder layer, TEMP-SA and SPA-SA represent temporal and spatial self-

attention blocks, respectively. We insert multi-modal fusion inside the backbone by introducing gated cross-attention after the space-time attention module. Hence, the output

of kth fused TimeSformer layer, x(vidk) , can be expressed as:

z(k) = SPACE-TIME(x(vidk−1)) x(vidk) = x(vidk−1) + z(k) + α ∗ CA(z(k),x(textk−1)) (2) x(vidk) = x(vidk) + FFN(x(vidk))

where x(textk−1) is the output from the (k−1)th RoBERTa layer, CA, FFN denote cross-attention block and feed-forward network, respectively, and α is a learnable gating parameter initialized from 0. Each RoBERTa layer contains multi-head self-attention [92] followed by feed-forward layers. Similar to the fused TimeSformer module, we insert cross-attention into the RoBERTa backbone:

xˆ(textk) = SA(x(textk−1)) x(textk) = x(textk−1) + xˆ(textk) + α ∗ CA(ˆx(textk) ,x(vidk)) (3) x(textk) = x(textk) + FFN(x(textk) )

where SA is the traditional self-attention module. For simplicity, we insert cross-attention into the same number of layers in both backbones. Notably, such fusion in the backbone strategy is not only limited to TimeSformer and RoBERTa; but can also be applied to any transformer-based video [61, 22, 2] and text [16, 81, 112] encoders.

Fusion in the backbone with gated cross-attention has the following advantages: (i) Cross-attention parameters can easily be switched off by setting the gating scalar α to 0; thus, the model behaves as a dual encoder, which is helpful for scenarios that require “unfused” video and textual features; (ii) Our fusion approach is more lightweight and compute-efficient than adding fusion-specific transformer layers, which is demonstrated in detail in Section 4.5.

#### 3.2. Pre-training Objectives

We use three pre-training objectives: (1) Egocentric noise contrastive estimation (EgoNCE), (2) masked language modeling (MLM), and (3) video-text matching (VTM).

EgoNCE: Lin et al. [57] proposed EgoNCE for dualencoder-based egocentric VLP. It makes two modifications over InfoNCE [71]: (i) Besides the matched video-text samples, all pairs that share at least one noun or one verb are treated as positives. (ii) Every batch of N video-text samples is augmented with another N visually similar videos, which are treated as additional negatives. Overall, video-totext EgoNCE objective, Legov2t, can be expressed as:

T i tk

exp v

τ

1 | B| i∈ B

log k∈Pi

Legov2t =

T i tj′

T i tj

τ + exp v

exp v

τ

j∈B

(4)

[Figure 9]

[Figure 10]

[Figure 11]

(a) Retrieval w/ Dual Encoder.

(b) VQA/retrieval w/ Fusion Encoder.

(c) QFVS w/ Fusion Encoder.

Figure 4: EgoVLPv2 can be adapted to various dual- and fusion-encoder-based video-language tasks, ranging from retrieval, video question-answering, and video grounding to query-focused video summarization.

where the ith video embedding vi and jth text embedding tj are L2 normalized features, and τ is a temperature factor. B is the augmented batch with 2N samples. The term in brown are the modified positive samples, and the term in blue are the modified negative samples. The text-to-video EgoNCE objective, Legot2v, can be defined symmetrically. The total EgoNCE loss is: LEgoNCE = Legov2t + Legot2v.

We compute EgoNCE in a dual-encoder setting. Specifically, we set α = 0, and thus, the cross-attention modules are switched off to calculate the EgoNCE loss.

MLM: Masked language modeling and video-text matching are proven helpful in fusion-encoder-based VLP literature [16, 127]. For MLM, we randomly mask 15% text tokens,1 and the loss, LMLM, aims to reconstruct the masked tokens based on surrounding words and video patches by minimizing the negative log-likelihood.

VTM: For the VTM objective, the model is given a videotext sample, and the output is a binary label y ∈ {0,1} indicating if the input pair is matched. LVTM is constructed as a binary cross-entropy loss over the predicted scores. Following [5, 18], we sample the global hard-negative video-text pairs using the similarities computed by EgoNCE.

We compute LMLM and LVTM in a fusion-encoder setting. In this case, α ̸= 0 and the cross-attention modules are switched on. Overall, our EgoVLPv2 pre-training pipeline can be summarized in the following three steps:

- • EgoNCE requires unfused video and text features, so we

switch off cross-attention (α = 0). Thus, LEgoNCE is computed with EgoVLPv2 acting as a dual encoder.

- • MLM & VTM requires multi-modal representation. We switch on cross-attention modules and compute LMLM

1Following BERT, we decompose this 15% into 10% random words, 10% unchanged, and 80% with a special token [MASK].

and LVTM with EgoVLPv2 acting as a fusion encoder.

• For back-propagation, the three losses are added, resulting in Ltotal = (1−γ −δ)LEgoNCE +γLMLM +δLVTM, and back-propagated into the model end-to-end. γ and δ are hyper-parameters that control the contribution of different terms on Ltotal. An ablation on different pre-training objectives of EgoVLPv2 is provided in Section 4.5. The pseudo-code for pre-training EgoVLPv2 can be found in the supplementary.

- 3.3. Adaptation to Downstream Tasks

We now describe how we adapt EgoVLPv2 to different downstream tasks as shown in Figure 4.

Video-Text Retrieval: We perform retrieval in two settings: (i) dual encoders: we switch off cross-attention and use EgoVLPv2 as a dual encoder, and compute the cosine similarity between video clips and text narrations. (ii) fusion encoders: we switch on cross-attention. The top M layers of the video and language backbones interact and produce multi-modal representations, which are fed into the pre-trained VTM head to compute matching scores. We also compute an ensemble of the two to further boost the performance, discussed in Section 4.5.

Video Grounding and Question Answering: We perform both uni- (video-only) and multi-modal (text-guided) video grounding. We switch off cross-attention for unimodal grounding and use only the video encoder. We use EgoVLPv2 as a fusion encoder for text-guided grounding and video question answering.

Query-focused Video Summarization: The input videos are very long (3-5 hours) for this task. We first use the unfused N − M layers2 of our video and text encoders to extract uni-modal features from 5 second clips and the text query. Next, we apply the KTS shot boundary detector [75] to segment the long video. After this, the query and segment-wise clip features are fed into the top M fused layers of EgoVLPv2 to compute the multi-modal representation. Finally, we learn an additional single-layer transformer to design the interrelation across all 5 second long clips in every segment. We present additional details for the query-focused video summarization framework in the supplementary.

- 4. Experiments

- 4.1. Pre-training & Downstream Datasets

We pre-train EgoVLPv2 on the EgoClip [57] version of Ego4D [27], the largest publicly available egocentric video dataset. EgoClip sources untrimmed egocentric videos from Ego4D and offers filtered video-narration samples with

2For simplicity, we keep the number of unfused and fused layers the same in the video and text encoder.

Multimodal

Fusion Metrics (%) Eval.

Dataset Task

MCQ w/ dual ✓ ✗ Inter- & Intra Acc. ZS MCQ w/ fusion ✓ ✓ Inter- & Intra Acc. ZS

Ego4D [27]

NLQ ✓ ✓ Recall@N HT MQ ✗ − mAP, Recall@N HT

QFVS [84] Video Summ. ✓ ✓ F-1 HT EgoTaskQA [32] Video QA ✓ ✓ Mean Acc. HT, FT CharadesEgo [86] CLS† ✓ ✗ Video-level mAP ZS, FT EK-100 [15] MIR w/ dual ✓ ✗ mAP, nDCG ZS, FT

Table 1: Egocentric downstream datasets, metrics, and evaluation protocols. We evaluate EgoVLPv2 on a wide variety of benchmarks: video-text retrieval (EgoMCQ, CharadesEgo, EK-100), uni-modal and text-guided video grounding (EgoMQ, EgoNLQ), video question answering (EgoTaskQA) and query-focused video summarization (QFVS). The evaluation protocols include zero-shot (ZS), task-specific head-tuning (HT), and end-to-end fine-tuning (FT). †ChardesEgo is a multi-class classification problem, but we convert this to a retrieval task. Please find more details in Section 4.1 and in supplementary.

variable-length clip intervals instead of single timestamps of Ego4D. Moreover, EgoClip excludes the videos appearing in the validation and test sets of the Ego4D benchmark [27], resulting in around 3.8M pre-training samples covering over 2927 hours of video from 129 different scenarios.

We evaluate EgoVLPv2 across multiple benchmarks on five egocentric datasets, summarized in Table 1:

- • On Ego4D [27] benchmarks: Multiple-Choice Questions (EgoMCQ) is a text-to-video (T → V) retrieval task with five video clips for every query text. Natural Language Query (EgoNLQ) is a natural language grounding [28, 25, 87] task that aims to localize a single time interval within a video given a text query. Moment Query (EgoMQ) is a video-only temporal action localization [9] task.
- • Query-focused video summarization (QFVS) [84] aims to generate a concise version of a long (3-5 hours) egocentric video based on a natural language query.
- • Video question-answering on EgoTaskQA [32] provides four question types (descriptive, predictive, explanatory, and counterfactual) with direct and indirect references, and evaluates the prediction over spatial, temporal, and causal domains of goal-oriented task understanding. Notably, to the best of our knowledge, we are the first to unify QFVS and EgoTaskQA as two downstream tasks of a VLP framework.
- • Action Recognition on CharadesEgo [86]: a multi-class classification of daily indoor activities, with class names being short natural language phrases like ‘Putting something on a shelf.’ Hence, leveraging text representations with class names, we treat this task as a retrieval problem.

EgoMCQ EgoNLQ validation set Accuracy (%) mIOU@0.3 mIOU@0.5

# Pre-train Dataset

Method

Inter Intra R@1 R@5 R@1 R@5

SlowFast [23] − − − 5.45 10.74 3.12 6.63 EgoVLP [57] 3.8M 90.6 57.2 10.84 18.84 6.81 13.45 HierVL-Avg [3] 3.8M 90.3 53.1 − − − − HierVL-SA [3] 3.8M 90.5 52.4 − − − −

LAVILA-B [125] 56M 93.8 59.9 10.53 19.13 6.69 13.68 EgoVLPv2 3.8M 91.0 60.9 12.95 23.80 7.91 16.11

∆Ours-EgoVLP − 0.4 ↑ 3.7 ↑ 2.11 ↑ 4.96 ↑ 1.10 ↑ 2.66 ↑

- Table 2: Performance on EgoMCQ and EgoNLQ’s validation set. EgoVLPv2 yields significant gains over existing baselines on both tasks. LAVILA is pre-trained on 15× more narrations generated by GPT-2 [79], and is colored gray. On EgoMCQ, reported results are achieved by directly ensembling dual- and fusion-encoder-based inference.

Method

IoU=0.3 IoU=0.5 IoU=0.7 mAP (%) @ IoU

R@1 R@5 R@1 R@5 R@1 R@5 0.1 0.3 0.5 Avg.

SlowFast [23] 33.45 58.43 25.16 46.18 15.36 25.81 9.10 5.76 3.41 6.03 Frozen [4] 40.06 63.71 29.59 48.32 17.41 26.33 15.90 10.54 6.19 10.69 EgoVLP [57] 40.43 65.67 30.14 51.98 19.06 29.77 16.63 11.45 6.57 11.39 EgoVLPv2 41.97 68.24 31.08 54.15 20.96 31.10 17.58 11.92 6.90 12.23

∆Ours-EgoVLP 1.54 ↑ 2.57 ↑ 0.94 ↑ 2.17 ↑ 1.90 ↑ 1.33 ↑ 0.95 ↑ 0.47 ↑ 0.33 ↑ 0.84 ↑

- Table 3: Performance on EgoMQ’s validation set. EgoVLPv2 sets a new state-of-the-art across all baselines using VSGN [124] as grounding head.

• Multi-instance retrieval on Epic-Kitchens-100 [15] (EK100 MIR): this is a text-to-video (T → V) and video-totext (V → T) retrieval task, with a significant semantic overlap between different narrations. Detailed statistics of pre-training and downstream datasets and evaluation metrics are given in the supplementary.

#### 4.2. Evaluation Protocol

We evaluate EgoVLPv2 using three evaluation protocols:

- • Zero-Shot (ZS). The pre-trained backbones are directly applied for V ↔ T retrieval without fine-tuning on downstream datasets. We perform zero-shot retrieval via: (i) dual encoders, computing the cosine similarity between video clips and textual narrations, and (ii) fusion encoder, incorporating the pre-trained VTM head to compute the video-text matching score.
- • Task-specific Head-tune (HT). We extract features using the frozen encoder and train task-specific heads3 using the training split of downstream datasets.
- • Fine-tune (FT). We fine-tune the entire pre-trained videotext model end-to-end using the training split of downstream datasets.

3VSLNet [119] for EgoNLQ, VSGN [124] for EgoMQ, single-layer transformer encoder [92] for summarization, and linear layers for video QA.

Method Video-1 Video-2 Video-3 Video-4 Average

SeqDPP [26] 36.59 43.67 25.26 18.15 30.92 SH-DPP [83] 35.67 42.72 36.51 18.62 33.38 QC-DPP [84] 48.68 41.66 36.51 29.96 44.19 TPAN [122] 48.74 45.30 56.51 33.64 46.05 CHAN [102] 49.14 46.53 58.65 33.42 46.94 HVN [34] 51.45 47.49 61.08 35.47 48.87 QSAN [101] 48.52 46.64 56.93 34.25 46.59 WHM [69] 50.96 48.28 58.41 39.18 49.20 IntentVizor [100] 51.27 53.48 61.58 37.25 50.90

EgoVLP† [57] 49.64 53.60 59.87 35.76 49.72 EgoVLPv2 53.30 54.13 62.64 38.25 52.08

∆Ours-EgoVLP 3.66 ↑ 0.53 ↑ 2.77 ↑ 2.49 ↑ 2.36 ↑

Table 4: Performance on query-focused video summarization (QFVS). Existing baselines are trained end-toend, whereas EgoVLPv2 only learns a tiny head on top of pre-trained encoders. †EgoVLP denotes the performance achieved by the officially released checkpoint.

#### 4.3. Implementation Details

We use TimeSformer-B [6, 4] and RoBERTa-B [59] as our video and language backbones. The video encoder has 12 layers and 12 heads, and is configured with the patch size of 16 × 16 and the hidden dimension of 768. The spatial attention modules are initialized from a ViT [17]. We resize videos to 224 × 224 and sample 4 frames per video for pretraining and 16 for fine-tuning on downstream tasks. We use RoBERTa-B pre-trained on English Wikipedia and Toronto Book Corpus. For our best model,4 we fuse the top 6 layers of the two encoders. We pre-train our model for 20 epochs with a batch size of 256, using AdamW [62] with a peak learning rate of 3e-5 for the backbones and 12e-5 for the cross-modal parameters. We use linear warmup over the first 2 epochs and use linear decay. Pre-training takes five days on 32 A100 GPUs. Other necessary pre-training and downstream details are given in the supplementary.

#### 4.4. Main Results

We use boldface and underline for the best and secondbest performing methods in every table and indicate the performance improvements over the state-of-the-art with ∆. Ego4D: Table 2 and 3 present the performance of EgoVLPv2 on three different Ego4D benchmarks: EgoMCQ, EgoNLQ and EgoMQ. On EgoMCQ, our model achieves 91.0% intervideo and 60.9% intra-video accuracy, significantly improving over the baselines. Note that EgoVLPv2 achieves 1% absolute gain on the challenging intra-video MCQ task over LAVILA, which is trained using 15× more narrations generated by a pre-trained large language model, GPT-2 [79]. On EgoNLQ, EgoVLPv2 yields an impressive gain of 2.11% R@1 for IoU = 0.3 over EgoVLP. Moreover, using a smaller task-specific head and fewer epochs of head-tuning,

4An ablation on the number of fusion layers is provided in Section 4.5.

Direct Indirect

Method Eval.

Open Binary All Open Binary All

VisualBERT [49] FT 24.62 68.08 37.93 21.05 57.61 37.01 PSAC [51] FT 26.97 65.95 38.90 15.31 57.75 32.72 HME [21] FT 27.66 68.60 40.16 18.27 52.55 33.06 HGA [35] FT 22.75 68.53 36.77 8.66 53.72 28.36 HCRN [40] FT 30.23 69.42 42.40 27.82 59.29 41.56 ClipBERT [44] FT 27.70 67.52 39.87 11.17 40.71 24.08

EgoVLP† [57] FT 31.69 71.26 42.51 27.04 55.28 38.69 EgoVLPv2 FT 35.56 75.60 46.26 29.14 59.68 42.28

∆Ours-EgoVLP FT 3.87 ↑ 4.34 ↑ 3.75 ↑ 2.10 ↑ 4.40 ↑ 3.59 ↑ EgoVLP† [57] HT 20.52 64.63 32.76 16.87 48.40 29.19 EgoVLPv2 HT 26.59 69.10 37.87 22.11 57.19 35.20 ∆Ours-EgoVLP HT 6.07 ↑ 4.47 ↑ 5.11 ↑ 5.24 ↑ 8.79 ↑ 6.01 ↑

- Table 5: Performance on EgoTaskQA direct and indirect splits. EgoVLPv2 outperforms prior work across all settings, metrics, and data splits. †EgoVLP denotes the performance achieved by the officially released checkpoint.

Method Eval.

CharadesEgo

Method Eval.

EK-100 MIR

mAP mAP nDCG

Actor [85] FT 20.0 S3D [103] FT 29.2 44.7 SSDA [12] FT 23.1 MME [99] FT 38.5 48.5 Ego-Exo [54] FT 30.1 JPoSE [99] FT 44.0 53.5 EgoVLP [57] FT 32.1 EgoVLP [57] FT 45.0 59.4 HierVL-Avg [3] FT 32.6 HierVL-Avg [3] FT 44.9 59.8 HierVL-SA [3] FT 33.8 HierVL-SA [3] FT 46.7 61.1 EgoVLPv2 FT 34.1 EgoVLPv2 FT 47.3 61.9

∆Ours-EgoVLP FT 2.0 ↑ ∆Ours-EgoVLP FT 2.3 ↑ 2.5 ↑ ∆Ours-HierVL-SA FT 0.3 ↑ ∆Ours-HierVL-SA FT 0.6 ↑ 0.8 ↑

EgoVLP [57] ZS 25.0 EgoVLP [57] ZS 16.6 23.1 HierVL-Avg [3] ZS 25.2 HierVL-Avg [3] ZS 16.7 23.5 HierVL-SA [3] ZS 26.0 HierVL-SA [3] ZS 18.9 24.7 EgoVLPv2 ZS 26.2 EgoVLPv2 ZS 26.7 29.1

∆Ours-EgoVLP ZS 1.2 ↑ ∆Ours-EgoVLP ZS 10.1 ↑ 6.0 ↑ ∆Ours-HierVL-SA ZS 0.2 ↑ ∆Ours-HierVL-SA ZS 7.8 ↑ 4.4 ↑

- Table 6: Performance on CharadesEgo and EK-100 MIR. EgoVLPv2 achieves significant gains in fine-tuning and zeroshot settings for both tasks. Results are achieved by dualencoder-based inference.

EgoVLPv2 outperforms existing baselines, which indicates the importance of learning cross-modal information during pre-training.5 On the uni-modal grounding task, EgoMQ, our framework also sets a new state-of-the-art, outperforming EgoVLP by 1.54% R@1 for IoU = 0.3, implying the flexibility of fusion in the backbone over dual and shared encoder-based pre-training.

QFVS: We evaluate EgoVLPv2 on query-focused video summarization task. The QFVS dataset contains only 135 video-query training samples with long (3-5 hours) videos, and all existing baselines are trained end-to-end. In contrast, we learn a tiny head (single-layer transformer) on top of the pre-trained encoders. As shown in Table 4, our model persistently attains the state-of-the-art F-1 score across all four

5Additional details are provided in supplementary.

Fusion Strategy

# Fusion Layers

#Trainable Params.

GMACs per instance

EgoMCQ

Inter Intra

3 374.5M 288.62 90.5 60.0 Fusion in the 6 381.6M 300.16 91.0 60.9

Backbone 9 388.7M 311.71 91.0 60.9

12 395.8M 323.26 91.0 60.9 Additional Fusion Layers

3 396.9M 402.88 90.5 60.3 6 414.6M 437.90 90.5 60.8 9 432.4M 472.91 90.6 60.8

12 450.1M 507.92 90.6 60.9

Table 7: Ablation study on fusion strategies. Our proposed fusion in the backbone strategy performs slightly better than using fusion-specific transformer layers, but with less parameters and less compute .

videos in this dataset. The pre-trained video-language representation helps EgoVLPv2 to achieve strong performance, whereas the baselines struggle to learn good cross-modal features due to the small training set.

EgoTaskQA: Table 5 shows the results on the egocentric video question-answering tasks on the EgoTaskQA dataset. Our model achieves significant gains across various baselines in the fine-tuning regime. Notably, EgoVLPv2 performs consistently well in the challenging indirect split, which demonstrates its ability to solve complicated reference tasks. In the head-tuning regime, we only learn a linear layer on top of frozen encoders, where EgoVLPv2 beats EgoVLP by a strong margin, which proves the efficacy of cross-modal pre-trained representation.

CharadesEgo: This is a multi-class action recognition task, with class names as short text phrases. We convert this to a video-to-text (V → T) retrieval problem as in CLIP [78], and perform dual-encoder-based retrieval. As shown in Table 6, EgoVLPv2 obtains a new state-of-the-art in both finetuning and zero-shot regimes. Since CharadesEgo videos are significantly different from Ego4D, being captured by crowd-sourced workers using mobile cameras, these results demonstrate the generalizability of EgoVLPv2.

EK-100: Table 6 shows our results on EK-100 MIR. In the fine-tuning regime, EgoVLPv2 achieves noticeable improvements over the supervised approaches (S3D, MME, JPoSE) and VLP methods (EgoVLP, HierVL). In the zero-shot setup, EgoVLPv2 beats EgoVLP and HierVL by 7.8% mAP and 4.4% nDCG scores. The consistent performance gains again show the quality of pre-trained encoders.

#### 4.5. Ablation Study

Fusion in the Backbone: We compare our fusion module to the conventional practice of using fusion-specific transformer layers, which we implement following ALBEF [46].6 Table 7 shows that the proposed fusion strategy performs

6https://github.com/salesforce/ALBEF/

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

#C C stands near the wheelbarrow and scrolls the phone.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

#C C stirs the eggs from the pan on the cooker with right hand.

Figure 5: Text-to-video cross-attention from multiple heads in the last layer of EgoVLPv2 with 16 × 16 patches. We look at the attention maps of the [CLS] token from the text encoder on input video frames. Different heads, depicted in different colors, focus on different objects or parts. These maps show the strong cross-modal representation learned by EgoVLPv2 during pre-training, which helps to enhance performance on video-language downstream tasks.

EgoMCQ (%) Dual Enc. Fusion Enc. Ensemble

Pre-training Objectives

EgoNCE MLM VTM VTM-Hard Inter Intra Inter Intra Inter Intra

✓ − − − 89.5 52.6 − − − − ✓ ✓ − − 89.6 52.4 − − − −

✓ − − ✓ 89.6 53.4 90.6 59.1 91.0 60.0 ✓ ✓ ✓ − 89.5 53.6 89.1 51.5 90.2 56.8 ✓ ✓ − ✓ 89.8 56.7 90.6 59.6 91.0 60.9

Table 8: Ablation study on different pre-training objectives of EgoVLPv2. We evaluate on EgoMCQ using our model either as a dual encoder, as a fusion encoder, or an ensemble of both. Removing any objective leads to a performance drop. The flexibility of the proposed fusion in the backbone module helps us boost retrieval performance using an ensembling strategy.

slightly better than stacked fusion layers. For both methods, increasing the number of fusion layers to 6 results in a nontrivial performance gain. However, our proposed architecture is significantly more parameter- and compute-efficient. For instance, with 6 fusion layers, the proposed architecture contains 33M fewer parameters and requires 45% lesser computing cost, which shows the efficacy of our method.

Pre-training Objectives: We ablate different pre-training objectives and evaluate the pre-trained models on EgoMCQ using EgoVLPv2 as a dual encoder, as a fusion encoder, and an ensemble of the two by summing their similarity scores for each video-text pair. As shown in Table 8, removing any pre-training objective lead to a performance drop. Specifically, VTM with hard-negative mining is largely beneficial across all three evaluation strategies. Fusion encoderbased evaluation brings significant improvements over dualencoders; moreover, as EgoMCQ contains only 5 sentences for every video, both evaluation methods offer similar la-

tency. Ensembling the two yields further 1−2% performance gain for both inter- and intra-video accuracy metrics.

#### 4.6. Attention Visualization & Error Analysis

In Figure 5, we show that different heads in the crossmodal attention can attend to different semantic regions of the video frames, guided by the narration. We observe that the pre-trained model learns well to recognize a wide variety of objects appearing in egocentric actions, such as indoor furniture, cooking appliances, phones, tablets, car steering, bicycle handles, etc. Such strong cross-modal information learned during pre-training helps EgoVLPv2 in multi-modal downstream tasks. The visualizations in Figure 5 are obtained with 960p video frames, resulting in sequences of 3601 tokens for 16 × 16 patches. However, vastly hindered objects in cluttered environments, especially in low-light conditions, are occasionally not focused. We show such error cases in the supplementary.

### 5. Conclusion

This work introduces EgoVLPv2, the second generation of egocentric video-language pre-training and a significant improvement over the previous generation [57] by incorporating cross-modal fusion directly into the video and language backbones. Our proposed fusion in the backbone strategy is lightweight, compute-efficient, and allows us to unify various VL tasks in a flexible and efficient manner. We conduct extensive experiments to demonstrate the effectiveness of EgoVLPv2 on a wide range of downstream tasks, consistently achieving state-of-the-art performance. Moreover, we visually demonstrate the effectiveness of the learned cross-attention representation.

### Acknowledgement

The codebase for this work is built on the EgoVLP [57], LAVILA[125], FIBER [18], and VSLNet [119] repository. We would like to thank the respective authors for their contribution, and the Meta AI team for discussions and feedback. Shraman Pramanick and Rama Chellappa were partially supported by a MURI program from the Army Research Office under the grant W911NF17-1-0304.

### References

- [1] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. Advances in Neural Information Processing Systems, 34:24206–24221, 2021. 1, 2, 3
- [2] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836–6846,

2021. 1, 4

- [3] Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, and Kristen Grauman. Hiervl: Learning hierarchical videolanguage embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2023. 1, 2, 3, 6, 7, 16, 17, 19

- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 1, 2, 3, 6, 16

- [5] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Songhao Piao, and Furu Wei. Vlmo: Unified visionlanguage pre-training with mixture-of-modality-experts. In Advances in Neural Information Processing Systems, 2022. 2, 4
- [6] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In International Conference on Machine Learning, pages 813–824. PMLR, 2021. 1, 3, 6, 16
- [7] Damian Borth, Tao Chen, Rongrong Ji, and Shih-Fu Chang. Sentibank: large-scale ontology and classifiers for detecting sentiment and emotions in visual content. In Proceedings of the 21st ACM international conference on Multimedia, pages 459–460, 2013. 15
- [8] Biagio Brattoli, Joseph Tighe, Fedor Zhdanov, Pietro Perona, and Krzysztof Chalupka. Rethinking zero-shot video classification: End-to-end training for realistic applications. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4613–4623, 2020. 1
- [9] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 5

- [10] Shizhe Chen, Bei Liu, Jianlong Fu, Ruihua Song, Qin Jin, Pingping Lin, Xiaoyu Qi, Chunting Wang, and Jin Zhou. Neural storyboard artist: Visualizing stories with coherent image sequences. In Proceedings of the 27th ACM International Conference on Multimedia, pages 2236–2244, 2019. 2
- [11] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX, pages 104–120. Springer, 2020. 2
- [12] Jinwoo Choi, Gaurav Sharma, Manmohan Chandraker, and Jia-Bin Huang. Unsupervised and semi-supervised domain adaptation for action recognition from drones. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1717–1726, 2020. 7
- [13] Wen-Sheng Chu, Yale Song, and Alejandro Jaimes. Video co-summarization: Video summarization by visual cooccurrence. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3584–3592,

2015. 15

- [14] Alexis Conneau and Guillaume Lample. Cross-lingual language model pretraining. Advances in neural information processing systems, 32, 2019. 1
- [15] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, pages 1–23, 2022. 1, 3, 5, 6, 16
- [16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. 1, 2, 4
- [17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 6
- [18] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, et al. Coarse-to-fine vision-language pretraining with fusion in the backbone. In Advances in Neural Information Processing Systems, 2022. 2, 4, 9
- [19] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training endto-end vision-and-language transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18166–18176, 2022. 2

- [20] Jiri Fajtl, Hajar Sadeghi Sokeh, Vasileios Argyriou, Dorothy Monekosso, and Paolo Remagnino. Summarizing videos with attention. In Computer Vision–ACCV 2018 Workshops: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers 14, pages 39–54. Springer, 2019. 17
- [21] Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, and Heng Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1999–2007,

2019. 7

- [22] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6824–6835, 2021. 1, 4
- [23] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202–6211, 2019. 6, 19
- [24] Jianlong Fu, Tao Mei, Kuiyuan Yang, Hanqing Lu, and Yong Rui. Tagging personal photos with transfer deep learning. In Proceedings of the 24th International Conference on World Wide Web, pages 344–354, 2015. 2
- [25] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 5
- [26] Boqing Gong, Wei-Lun Chao, Kristen Grauman, and Fei Sha. Diverse sequential subset selection for supervised video summarization. Advances in neural information processing systems, 27, 2014. 6, 15
- [27] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 1, 3, 5, 15
- [28] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with temporal language. In Empirical Methods in Natural Language Processing (EMNLP), 2018. 5
- [29] Yupan Huang, Hongwei Xue, Bei Liu, and Yutong Lu. Unifying multimodal transformer for bi-directional image and text generation. In Proceedings of the 29th ACM International Conference on Multimedia, pages 1138–1147, 2021. 2
- [30] Jiho Jang, Chaerin Kong, Donghyeon Jeon, Seonhoon Kim, and Nojun Kwak. Unifying vision-language representation space with single-tower transformer. In AAAI, 2023. 2
- [31] Baoxiong Jia, Yixin Chen, Siyuan Huang, Yixin Zhu, and Song-chun Zhu. Lemma: A multi-view dataset for le arning m ulti-agent m ulti-task a ctivities. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVI 16, pages 767–

786. Springer, 2020. 15

- [32] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 5, 15
- [33] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR,

2021. 2

- [34] Pin Jiang and Yahong Han. Hierarchical variational network for user-diversified & query-focused video summarization. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pages 202–206, 2019. 2, 6
- [35] Pin Jiang and Yahong Han. Reasoning with heterogeneous graph alignment for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11109–11116, 2020. 7
- [36] Alec Kerrigan, Kevin Duarte, Yogesh Rawat, and Mubarak Shah. Reformulating zero-shot action recognition for multilabel actions. Advances in Neural Information Processing Systems, 34:25566–25577, 2021. 1
- [37] Aditya Khosla, Raffay Hamid, Chih-Jen Lin, and Neel Sundaresan. Large-scale video summarization using web-image priors. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2698–2705, 2013. 15
- [38] Gunhee Kim, Leonid Sigal, and Eric P Xing. Joint summarization of large-scale collections of web images and videos for storyline reconstruction. In Proceedings of the IEEE Conference on computer vision and pattern recognition, pages 4225–4232, 2014. 15
- [39] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 3
- [40] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9972–9981, 2020. 7
- [41] Sangho Lee, Youngjae Yu, Gunhee Kim, Thomas Breuel, Jan Kautz, and Yale Song. Parameter efficient multimodal transformers for video representation learning. In International Conference on Learning Representations, 2021. 1, 2, 3
- [42] Yong Jae Lee, Joydeep Ghosh, and Kristen Grauman. Discovering important people and objects for egocentric video summarization. In 2012 IEEE conference on computer vision and pattern recognition, pages 1346–1353. IEEE, 2012. 15
- [43] Chenyi Lei, Shixian Luo, Yong Liu, Wanggui He, Jiamang Wang, Guoxin Wang, Haihong Tang, Chunyan Miao, and Houqiang Li. Understanding chinese video and language via contrastive multimodal pre-training. In Proceedings of the 29th ACM International Conference on Multimedia, pages 2567–2576, 2021. 2
- [44] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert

- for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7331–7341, 2021. 1, 2, 3, 7
- [45] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 2
- [46] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 2, 7
- [47] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+ language omni-representation pre-training. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2046–2065, 2020. 2
- [48] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: Unifying video-language understanding as masked language modeling. arXiv preprint arXiv:2206.07160, 2022. 1, 2, 3
- [49] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 7
- [50] Nanxing Li, Bei Liu, Zhizhong Han, Yu-Shen Liu, and Jianlong Fu. Emotion reinforced visual storytelling. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pages 297–305, 2019. 2
- [51] Xiangpeng Li, Jingkuan Song, Lianli Gao, Xianglong Liu, Wenbing Huang, Xiangnan He, and Chuang Gan. Beyond rnns: Positional self-attention with co-attention for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8658–8665,

- 2019. 7

[52] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 121–137. Springer,

- 2020. 2

- [53] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390– 23400, 2023. 2
- [54] Yanghao Li, Tushar Nagarajan, Bo Xiong, and Kristen Grauman. Ego-exo: Transferring visual representations from third-person to first-person videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6943–6953, 2021. 7
- [55] Yin Li, Zhefan Ye, and James M Rehg. Delving into egocentric actions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 287–295,

2015. 3

- [56] Chung-Ching Lin, Kevin Lin, Lijuan Wang, Zicheng Liu, and Linjie Li. Cross-modal representation learning for zeroshot action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19978–19988, 2022. 1
- [57] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Denial Gao, RongCheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. In Advances in Neural Information Processing Systems, 2022. 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22
- [58] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding, 2023. 1
- [59] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692,

2019. 1, 3, 6, 16

- [60] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 1
- [61] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 1, 4
- [62] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018. 6, 16, 17
- [63] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019. 2
- [64] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. arXiv preprint arXiv:2002.06353, 2020. 1, 2, 3
- [65] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9879– 9889, 2020. 2, 3
- [66] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640, 2019. 1, 3
- [67] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Alireza Dirafzoon, Aparajita Saraf, Amy Bearman, and Babak Damavandi. Imu2clip: Multimodal contrastive learning for

- imu motion sensors from egocentric videos and text. arXiv preprint arXiv:2210.14395, 2022. 1, 2, 3
- [68] Jonghwan Mun, Minsu Cho, and Bohyung Han. Localglobal video-text interactions for temporal grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10810–10819, 2020. 1
- [69] Saiteja Nalla, Mohit Agrawal, Vishal Kaushal, Ganesh Ramakrishnan, and Rishabh Iyer. Watch hours in minutes: Summarizing videos with user intent. In Computer Vision– ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 714–730. Springer, 2020. 2, 6
- [70] Medhini Narasimhan, Anna Rohrbach, and Trevor Darrell. Clip-it! language-guided video summarization. Advances in Neural Information Processing Systems, 34:13988–14000,

2021. 2

- [71] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2, 4, 18
- [72] Jaeyoo Park and Bohyung Han. Multi-modal representation learning with text-driven soft masks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2798–2807, 2023. 2
- [73] Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander G Hauptmann, Joao F Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. In International Conference on Learning Representations, 2020. 1
- [74] Chiara Plizzari, Gabriele Goletto, Antonino Furnari, Siddhant Bansal, Francesco Ragusa, Giovanni Maria Farinella, Dima Damen, and Tatiana Tommasi. An outlook into the future of egocentric vision. arXiv preprint arXiv:2308.07123,

2023. 3

- [75] Danila Potapov, Matthijs Douze, Zaid Harchaoui, and Cordelia Schmid. Category-specific video summarization. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI 13, pages 540–555. Springer, 2014. 5, 17
- [76] Shraman Pramanick, Li Jing, Sayan Nag, Jiachen Zhu, Hardik Shah, Yann LeCun, and Rama Chellappa. Volta: Vision-language transformer with weakly-supervised localfeature alignment. arXiv preprint arXiv:2210.04135, 2022. 2
- [77] Shraman Pramanick, Aniket Roy, and Vishal M Patel. Multimodal learning using optimal transport for sarcasm and humor detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3930–3940, 2022. 2
- [78] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 7, 16
- [79] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are un-

- supervised multitask learners. OpenAI blog, 1(8):9, 2019. 6
- [80] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020. 1
- [81] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108,

2019. 1, 4

- [82] Anshul Shah, Benjamin Lundell, Harpreet Sawhney, and Rama Chellappa. Steps: Self-supervised key step extraction from unlabeled procedural videos. arXiv preprint arXiv:2301.00794, 2023. 3
- [83] Aidean Sharghi, Boqing Gong, and Mubarak Shah. Queryfocused extractive video summarization. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 3–19. Springer, 2016. 6, 15
- [84] Aidean Sharghi, Jacob S Laurel, and Boqing Gong. Queryfocused video summarization: Dataset, evaluation, and a memory network based approach. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4788–4797, 2017. 2, 5, 6, 15, 17
- [85] Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Actor and observer: Joint modeling of first and third-person videos. In proceedings of the IEEE conference on computer vision and pattern recognition, pages 7396–7404, 2018. 7
- [86] Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charades-ego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626, 2018. 3, 5, 16
- [87] Mattia Soldan, Mengmeng Xu, Sisi Qu, Jesper Tegner, and Bernard Ghanem. Vlg-net: Video-language graph matching network for video grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3224–3234, 2021. 5
- [88] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5179–5187, 2015. 15
- [89] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7464–7473, 2019. 2
- [90] Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, and Jianlong Fu. Long-form video-language pretraining with multimodal temporal contrastive learning. In Advances in Neural Information Processing Systems, 2022. 1, 2, 3
- [91] Zineng Tang, Jaemin Cho, Jie Lei, and Mohit Bansal. Perceiver-vl: Efficient vision-and-language modeling with iterative latent attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4410–4420, 2023. 1, 2, 3

- [92] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1, 4, 6
- [93] Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Objectaware video-language pre-training for retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3313–3322, 2022. 2, 3
- [94] Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6598–6608, 2023. 1, 2, 3
- [95] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. Transactions of Machine Learning Research. 2
- [96] Jinpeng Wang, Pan Zhou, Mike Zheng Shou, and Shuicheng Yan. Position-guided text prompt for vision-language pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23242– 23251, 2023. 2
- [97] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442,

2022. 15

- [98] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and visionlanguage tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19175– 19186, 2023. 2
- [99] Michael Wray, Diane Larlus, Gabriela Csurka, and Dima Damen. Fine-grained action retrieval through multiple partsof-speech embeddings. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 450– 459, 2019. 7, 17
- [100] Guande Wu, Jianzhe Lin, and Claudio T Silva. Intentvizor: Towards generic query guided interactive video summarization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10503–10512,

2022. 2, 6

- [101] Shuwen Xiao, Zhou Zhao, Zijian Zhang, Ziyu Guan, and Deng Cai. Query-biased self-attentive network for queryfocused video summarization. IEEE Transactions on Image Processing, 29:5889–5899, 2020. 2, 6
- [102] Shuwen Xiao, Zhou Zhao, Zijian Zhang, Xiaohui Yan, and Min Yang. Convolutional hierarchical attention network for query-focused video summarization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12426–12433, 2020. 2, 6
- [103] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning:

- Speed-accuracy trade-offs in video classification. In Proceedings of the European conference on computer vision (ECCV), pages 305–321, 2018. 7
- [104] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 1
- [105] Hu Xu, Gargi Ghosh, Po-Yao Huang, Prahal Arora, Masoumeh Aminzadeh, Christoph Feichtenhofer, Florian Metze, and Luke Zettlemoyer. Vlm: Task-agnostic videolanguage model pre-training for video understanding. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4227–4239, 2021. 1, 2, 3
- [106] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6787–6800, 2021. 2, 3
- [107] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 1
- [108] Jia Xu, Lopamudra Mukherjee, Yin Li, Jamieson Warner, James M Rehg, and Vikas Singh. Gaze-enabled egocentric video summarization via constrained submodular maximization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2235–2244, 2015. 15
- [109] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022. 1, 2, 3
- [110] Jianwei Yang, Yonatan Bisk, and Jianfeng Gao. Taco: Tokenaware cascade contrastive learning for video-text alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11562–11572, 2021. 1, 2, 3
- [111] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19163–19173, 2022. 2
- [112] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32, 2019. 1, 4
- [113] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang. Unitab: Unifying text and box outputs for grounded visionlanguage modeling. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, 2022, pages 521–539. Springer, 2022. 2
- [114] Serena Yeung, Alireza Fathi, and Li Fei-Fei. Videoset: Video summary evaluation through text. arXiv preprint arXiv:1406.5824, 2014. 15

- [115] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 15
- [116] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In Proceedings of the European Conference on Computer Vision (ECCV), pages 471–487, 2018. 1
- [117] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing Systems, 34:23634–23651,

2021. 1, 2, 3

- [118] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022. 2
- [119] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6543– 6554, 2020. 6, 9, 17, 19
- [120] Ke Zhang, Wei-Lun Chao, Fei Sha, and Kristen Grauman. Summary transfer: Exemplar-based subset selection for video summarization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1059–1067, 2016. 15
- [121] Ke Zhang, Wei-Lun Chao, Fei Sha, and Kristen Grauman. Video summarization with long short-term memory. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part VII 14, pages 766–782. Springer, 2016. 17
- [122] Yujia Zhang, Michael Kampffmeyer, Xiaodan Liang, Min Tan, and Eric P Xing. Query-conditioned three-player adversarial network for video summarization. British Machine Vision Conference (BMVC), 2018. 6
- [123] Bin Zhao and Eric P Xing. Quasi real-time summarization for consumer videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2513– 2520, 2014. 15
- [124] Chen Zhao, Ali K Thabet, and Bernard Ghanem. Video self-stitching graph network for temporal action localization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13658–13667, 2021. 6, 17
- [125] Yue Zhao, Ishan Misra, Philipp Kr¨ahenb¨uhl, and Rohit Girdhar. Learning video representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6586–6597,

2023. 1, 2, 3, 6, 9, 16, 17, 18, 19

- [126] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018. 1
- [127] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In Proceedings of the IEEE/CVF

conference on computer vision and pattern recognition, pages 8746–8755, 2020. 1, 2, 4

### A. Radar Chart Figure 1 Details

Here, we explain the details of the radar chart in Figure 1, which summarizes the comparative performance of EgoVLPv2 with EgoVLP [57]. First, for illustrative purposes, we normalize each axis by the score achieved by EgoVLPv2, which turns the axes in the range (0,1]. Next, we keep the origin of each axis at 0.7 normalized value, which reasonably separates the inner and outer frames for better readability. Finally, we annotate each vertex with absolute performance metric scores. Notably, in most previous radar chats in the vision-language literature [97, 115], the axes have different scales and shifts, which may cause misinterpretations and fallacies. However, our illustration is uniform and accurate to scale.

### B. Algorithm

The algorithm for pre-training EgoVLPv2 is given in Algorithm 1. Section 3.2 provides details of different pretraining objectives.

### C. Dataset Details

This section provides additional details of our pre-training and downstream datasets.

Ego4D & EgoClip: Ego4D [27] is the first-of-its-kind massive-scale egocentric video-language dataset and benchmark suite. It offers 3670 hours of daily life activity videos captured by 931 unique camera wearers from 74 worldwide locations and 9 different countries. The videos in Ego4D span hundreds of scenarios (kitchen, laboratory, workshop, porch, shopping, driving, leisure, etc.) with various daytime and weather conditions. A portion of the dataset is accompanied by audio, 3D meshes of the environment, eye gaze, stereo, and synchronized videos from multiple egocentric cameras at the same event. Each narration in Ego4D is a free-form sentence and has a single timestamp. For example, the narration “#C C walks towards a laundry machine” is associated with the video content, which occurs at 28.3s of a particular video. However, an activity occurs for a certain duration, and such a single timestamp can not reflect the start and end points where the particular activity takes place. EgoClip [57] offers a filtered version of Ego4D and designs a contextual variable-length clip pairing strategy to assign every narration with start and end timestamps. Moreover, EgoClip excludes videos that belong to the validation and test sets of the Ego4D benchmark challenges and retains textual annotation from multiple narrators, allowing us to have narration diversity during pre-training. Overall, EgoClip contains 2927 hours of videos which form 3.8M clip-text pairs, with an average clip length of 1.0s and a standard deviation of 0.9s. We use this EgoClip version of Ego4D for pre-training. We evaluate EgoVLPv2 on three dif-

Algorithm 1 Pre-training EgoVLPv2

Require: Batch BN : {xvid, xtext} Learnable gating parameter: α EgoVLPv2 Encoder: F : Fdual if α = 0

Ffused if α ̸= 0

for (xvid, xtext) ∈ BN do LEgoNCE ← EgoNCE(Fdual(xvid, xtext)) ▷ EgoNCE xMLMtext ← Mask(xtext) LMLM ← MLM(Ffused(xvid, xMLMtext)) ▷ MLM xVTMtext ← HardNeg(xtext) LVTM ← VTM(Ffused(xvid, xVTMtext)) ▷ VTM Ltotal ← (1 − γ − δ)LEgoNCE + γLMLM + δLVTM

end for Back-prop into F end-to-end with Ltotal.

ferent downstream benchmarks of Ego4D: multiple-choice questions (EgoMCQ), natural language query (EgoNLQ), and moment query (EgoMQ).

QFVS: The query-focused video summarization (QFVS) [84] dataset builds upon previously existing UT egocentric (UTE) [42] dataset, which contains four 3-5 hours long videos captured in uncontrolled everyday scenarios. QFVS curates 46 queries for every video, where each query contains two distinct concepts (nouns) [114, 83, 7]. For example, a query can be {HAT, PHONE}, or {FOOD, DRINK}. These 46 queries cover four distinct scenarios: (i) both the concepts appear in the same video shot (15 such queries),7 (ii) the concepts appear in the video, but not in a single shot (15 such queries), (iii) only one concept appears in the video (15 such queries), and (iv) none of the concepts in the query are present in the video (1 such query). We use prompt engineering to generate natural language using the concepts in the query and feed the sentence in our model. For instance, a given query {HAT, PHONE} is converted as “All scenes containing hats and phones”. We use 10 different prompts during head-tuning. The QFVS dataset also annotates concepts for every video shot. It proposes a robust evaluation strategy: find the similarity between the concepts in the generated and ground truth summary by maximum weight matching of a bipartite graph, and compute precision, recall, and F1 score from the number of matched concepts. This evaluation strategy helps to capture how well a system summary can retain semantic information instead of visual quantities, as used in previously existing evaluation methods, such as a system-generated summary has to consist of the same key units (frame or shot) as in the user summary [13, 88, 108] or comparing pixels and low-level features [26, 37, 38, 120, 123].

EgoTaskQA: The EgoTaskQA [32] benchmark uses the same egocentric videos as the LEMMA dataset [31], which contains goal-oriented and multi-tasked human activities

7QFVS defines every consecutive 5s video clip as a shot.

with rich human-object interactions and action dependencies in both single-agent and two-agent collaboration scenarios. The videos are segmented into clips with an average duration of 25s. The questions in the EgoTaskQA dataset are machine-generated and aim to evaluate models’ capabilities to describe, explain, anticipate, and make counterfactual predictions about goal-oriented events. The answers are of two types - open-answer queries and binary statement verifications. The EgoTaskQA dataset contains 40K balanced question-answer pairs selected from 368K programmatically generated questions from 2K egocentric videos. Moreover, this dataset offers two different benchmark splits (i) normal or direct split where the train, test, and validation sets are randomly sampled in a 3:1:1 ratio and (ii) indirect split where the actions and objects are strongly correlated and test the model’s task understanding capability with challenging questions. We approach the video QA as a classification task and report accuracy for open queries and binary verification in the direct and indirect splits.

CharadesEgo: The CharadesEgo [86] dataset consists of 68.5K annotated samples from 7860 videos from both first and third-person views, covering 157 classes of daily indoor activities. We only use the first-person subset, which contains 3085 videos for training and 846 videos for testing. ChardesEgo is originally a multi-class classification problem, with class labels being short phrases like ‘Putting something on the shelf.’ We treat this problem to a video-to-text (V → T) retrieval task as in CLIP [78] by leveraging the text encoder to extract features from class names. We directly evaluate the model on the validation set in the zero-shot setting. In the fine-tuning setting, we leverage the 33.1K training samples to perform an end-to-end fine-tuning of EgoVLPv2. Following the previous literature [57, 125, 3], we report video-level mAP as the evaluation metric.

EK-100: The Epic-Kitchens-100 [15] dataset contains 100 hours of egocentric cooking videos. The training set consists of 67.2K video samples, whereas the validation and test set has 9.6K and 13.1K samples, respectively. Each sample is associated with text narration. We perform multi-instance retrieval (V ↔ T) on the EK-100 dataset, which is challenging due to the significant semantic overlap between different narrations. The evaluation metrics are mean Average Precision (mAP) and the normalized Discounted Cumulative Gain (nDCG).

### D. Implementation Details

#### D.1. Pre-training on EgoClip

Table D.1 presents the hyper-parameters used during pretraining. We use TimeSformer-B [6, 4] and RoBERTa-B [59] as our video and language backbones. We chose the best learning rate using a grid search. We ablate our other design choices in Section E. We use PyTorch’s native FP16

Hyper-parameters Notation Value

Model

Video encoder − TimeSFormer-B [6, 4] Text encoder − roberta-base [59] Video & text embedding − 768 Video encoder patch size − 16 × 16 Video & text projector − 4096-4096-4096 # Fusion layers − 6

Pre-training Batch size − 256 Epochs − 20 Number of frames − 4 Frame resolution − 224 × 224 Vocab size − 50265 MLM prob. − 0.15 Max. length of text − 30 Temp. in Equation 4 τ 0.05 MLM & VTM loss weights γ, δ 0.25, 0.5 Optimizer − AdamW [62] Peak LR for backbones − 3e−5 Peak LR for cross-att − 12e−5 Peak LR for loss heads − 12e−5 Warmup − Linear (first 2 epochs) LR decay − Linear End LR − 1e−7 Betas in AdamW (β1, β2) (0.9,0.98) Eps in AdamW − 1e−8 Weight decay − 1e−2

##### Table D.1: Pre-training hyper-parameter details of EgoVLPv2.

mixed precision training and gradient checkpoint during pre-training.

After every epoch, we validate the pre-trained checkpoint on EgoMCQ and select the model with the best EgoMCQ intra-video score for other downstream tasks. We extract 4 frames for every video sample during pre-training and reshape those to 224 × 224. We also apply standard RandomResizedCrop, RandomHorizontalFlip, ColorJitter and normalization to every frame. We tokenize the text using RoBERTa tokenizer and pad/truncate every narration to a maximum length of 30. Pre-training takes five days on 32 A100 GPUs.

#### D.2. Downstream Settings

This section presents our fine-tuning and head-tuning strategy for different downstream tasks. For a fair comparison with the baselines [57, 125, 3], we follow the same downstream configuration as the baselines when possible. The downstream is performed with 16 frames per video sample.

EgoNLQ: This task is a video-text localization problem, with each video clip longing up to 1200s. Hence, performing end-to-end fine-tuning can be hard on EgoNLQ. Following

PS Pe

Conditioned Span Predictor

QGH

VSLNet

Context-Query Attention

Conv. Shared Layers

Conv. Layers

M Fused Layers

M Fused Layers

FFN

FFN

Cross-Att

Cross-Att

Space-Time Self-Att

Self-Att

(NV-M) Unfused Layers

(NL-M) Unfused Layers

FFN

FFN

Space-Time Self-Att

Self-Att

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

What cloth did I wash?

Figure D.1: Entire pipeline for EgoNLQ. Following EgoVLP [57] and LAVILA [125], we pre-extract video-text features using pre-trained EgoVLPv2, and train VSLNet [119] on top of frozen encoders.

[57, 125], we pre-extract features from the video-text samples using our pre-trained model and train VSLNet [119] for 100 epochs, with a learning rate of 1e−3 and batch size of 32. We keep all other configurations the same as [57].8 However, we observe that we can beat the baselines using even a smaller task head and fewer epochs of tuning, which we describe in Section F. We show the complete EgoNLQ pipeline in Figure D.1.

EgoMQ: This is a video-only localization problem, and similar to EgoNLQ, the input videos are very long. Hence, end-to-end fine-tuning is also hard to perform on EgoMQ. Following EgoVLP [57], we pre-extract video features using pre-trained EgoVLPv2 and train VSGN [124] for 100 epochs, with a learning rate of 1e−4 and batch size of 32. We keep all other configurations the same as [57]. We perform a grid search for other hyper-parameters of VSGN.

QFVS: Query-focused video summarization aims to generate an abridged version of input video guided by a natural language query. To the best of our knowledge, we are the first to unify QFVS as a downstream of a VLP framework.

8https://github.com/showlab/EgoVLP

The input videos for this task are very long (3-5 hours). We first use the unfused N − M layers9 of our video and text encoders to extract uni-modal features from every 5-second clip and the text query. Next, we apply the KTS shot boundary detector [75] to segment the long video.10 After this, the query and segment-wise clip features are fed into the top M fused layers of EgoVLPv2 to compute the multi-modal representation. Finally, we learn an additional single-layer transformer to design the interrelation across all 5 second long clips in every segment. We train the single-layer transformer for 20 epochs, with a batch size of 20, a peak learning rate of 1e−5 using AdamW [62] optimizer, cosine scheduler, and a linear warmup for the first 2 epochs. We also perform an ablation on the single-layer transformer in Section F.

EgoTaskQA: We treat the video QA as a classification problem, where we train linear layers on top of the fused feature representation generated by the pre-trained EgoVLPv2. In the fine-tuning setting, we fine-tune the pre-trained model for 36 epochs with a batch size of 64, using the AdamW [62] optimizer. We use cosine annealing with 10% linear warmup steps, with the peak learning rate of 2e−4 for the direct split and 1e−4 for the indirect split. In the head-tuning setup, we only train the classifier head on top of frozen backbones with the same configuration.

CharadesEgo: Following [57, 125, 3], we convert CharadesEgo as a retrieval problem. In the zero-shot setup, we perform dual-encoder-based inference. In the fine-tuning setup, we use EgoNCE as our objective. We fine-tune the model for 10 epochs with a batch size of 128 using AdamW [62] optimizer with (β1,β2) = (0.9,0.98), and weight decay of 0.01. We use cosine annealing with warmup, with 10% linear warmup steps, peak learning rate of 1.5e−4 and end learning rate of 1e−7. Since this is a multi-class dataset, where each video can include multiple actions, we report mAP as the evaluation metric. For input, we sample 16 frames from each video clip, and reshape the frames into 224 × 224.

EK-100 MIR: Since a narration can jointly be associated with multiple videos for EK-100 multi-instance retrieval task, we use the adaptive multi-instance max-margin loss [99] for this task with a margin value of 0.2. We keep the zero-shot configuration the same as CharadesEgo. We finetune the model for 100 epochs with a batch size of 128 using AdamW [62] optimizer with (β1,β2) = (0.9,0.98), and weight decay of 0.01. We use cosine annealing with warmup, with 10% linear warmup steps, peak learning rate of 2e−4 and end learning rate of 1e−7.

- 9For simplicity, we keep the number of unfused and fused layers the same in the video and text encoder.
- 10Segmentation helps in two ways: (i) TimeSformer can not process the whole 3-5 hours long video (containing tens of thousands of frames) at once. (ii) Segmentation is also used to convert frame-level prediction scores into key shots. For details, please refer to [84, 20, 121].

EgoNCE Sampling EgoMCQ (%)

Pre-training Objectives

Pos. Neg. Inter Intra

InfoNCE + MLM + VTM − − 90.0 55.2 EgoNCE + MLM + VTM ✓ ✗ 90.4 58.8 EgoNCE + MLM + VTM ✗ ✓ 90.5 59.1 EgoNCE + MLM + VTM ✓ ✓ 91.0 60.9

- Table E.1: Ablation on EgoNCE sampling strategy. EgoNCE [57] helps in improving the performance significantly compared to InfoNCE [71]. We also observe that both the positive and negative sampling of EgoNCE is important, and removing any of those leads to a performance drop.

Cross-Att

EgoMCQ (%)

Inter Intra

α = 0.1 90.1 59.8 α = 0.25 90.4 59.9 α = 0.5 90.1 58.0 α = 1 89.4 56.9

Learnable α 91.0 60.9

- Table E.2: Ablation on the gated cross-attention. Learnable gating scaler α performs better than a fixed value.

### E. Additional Ablations on Pre-training

We conduct additional ablation experiments in this section to validate our design choices. Reported results on EgoMCQ in Table E.1, E.2, E.3 and Figure E.1 are achieved by directly ensembling dual- and fusion-encoder-based inference.

Effect of EgoNCE: We study the effect of the EgoNCE loss [57] compared to the more popular InfoNCE objective [71]. Given a batch of N video-text pairs, InfoNCE treats the matched N pairs as positives and every other pair as negatives. However, egocentric videos pose two unique challenges: (i) Same actions in different scenarios appear to be visually different (talking on the phone indoors and outdoors). (ii) Different actions in same scenarios appear to be similar (writing on a tablet and watching a movie on a tablet are visually indistinguishable). To overcome these challenges, EgoNCE is built upon InfoNCE with two modifications: (i) Besides the matched video-text samples in every batch, all narration pairs which share at least one noun and one verb are treated as positives. (ii) Every batch of N video-text pairs is augmented with another N visually similar videos, often containing different actions in the same scenarios. These added videos with the same texts as in the original batch are treated as additional negatives.

Table E.1 shows the effect of the modified positive and negative sampling of EgoNCE on EgoVLPv2. First, we observe that replacing EgoNCE with InfoNCE leads to a performance drop of 5.7% accuracy on the challenging intravideo metric of EgoMCQ. Further, discarding either positive or negative sampling also drops the results by 2.1-1.8% intra-

[Figure 32]

Figure E.1: Ablation on the projector dimension used in the EgoNCE head. A 3-layer projector works better than a single-layer projector. Moreover, an increase in the width of the projector also helps in performance.

video accuracy. These results align with the findings in [57] and indicate the efficacy of the EgoNCE objective for egocentric video-language pre-training.

Effect of Gated Cross-attention: Next, we study the importance of gated cross-attention modules with learnable gating scalar, α. Table E.2 shows that a fixed value of α leads to a significant performance drop. In our best pre-trained model, we also find that the learned value of α varies in different layers, ranging from 0.05 to 0.4.

Effect of Projector: We compare different choices of projector dimensions used in the EgoNCE head in Figure E.1. We observe that a three-layer projector works better than single and two-layer projectors. For instance, a 4096-4096-4096 dimensional projector improves the EgoMCQ intra-video retrieval performance by 0.85% over a single 4096 dimensional projector. Moreover, an increase in the width of the projector also helps in performance. Hence, we use 40964096-4096 as our default projector. Notably, these results oppose the findings in Zhao et al. [125], where the authors observe that using 256-dimension achieves better performance than a 512 dimensional projector. The reason behind such results is, in contrast to Zhao et al., [125], who only use InfoNCE, a larger projector helps us both in EgoNCE and VTM objectives by offering a stronger hard-negative sampling.

Effect of Batch Size: Next, we study the effect of pretraining batch size in Table E.3a. The performance improves

# Frames (Pre-training)

EgoMCQ (%)

EgoMCQ (%)

Batch Size

Inter Intra 2 90.1 56.7

Inter Intra

128 90.6 59.8 256 91.0 60.9 512 91.0 60.6

- 4 91.0 60.9

- 5 91.2 61.2

- 6 91.4 61.5

1024 90.8 60.5

(b) Ablation on number of frames. Increasing frames improve EgoMCQ performance.

(a) Ablation on batch size. EgoMCQ performance is best with a batch size of 256.

- Table E.3: Ablation on pre-training batch size (a) and the number of frames (b). A batch size of 256 produces the best results. Increasing the number of frames helps in a performance gain. For a fair comparison with the baselines [57, 125, 3], we keep 4 as our default frame number.

EgoNLQ validation set mIOU@0.3 mIOU@0.5 R@1 R@5 R@1 R@5

Model + Task head

SlowFast [23] + VSLNet [119] 5.45 10.74 3.12 6.63 EgoVLP [57] + VSLNet [119] 10.84 18.84 6.81 13.45 LAVILA[125] + VSLNet [119] 10.53 19.13 6.69 13.68

EgoVLPv2 + Span 11.08 21.27 7.05 14.29 EgoVLPv2 + QGH + Span 11.95 22.86 7.64 15.80 EgoVLPv2 + VSLNet [119] 12.95 23.80 7.91 16.11

- Table F.1: Ablation on task-head for EgoNLQ. EgoVLPv2 beats existing models even using a smaller task-head.

Model + Task head Video-1 Video-2 Video-3 Video-4 Average EgoVLPv2 + Linear layers 50.17 50.95 59.38 34.58 48.77

- EgoVLPv2 + 1-layer transformer 54.97 55.74 64.10 40.83 53.91

- EgoVLPv2 + 2-layer transformer 52.78 51.98 66.80 34.10 51.42

- EgoVLPv2 + 3-layer transformer 51.87 52.45 63.75 35.55 50.91

- Table F.2: Ablation on task-head for QFVS. A single-layer transformer produces better performance than linear layers and multi-layer transformers.

using a batch size of 256 over 128. However, the performance drops if we further increase the batch size to 512 or 1024. Therefore, we use 256 as our default batch size in all other experiments.

Effect of Number of Frames: Lastly, we ablate the number of frames per sample during pre-training in Table E.3b. We see a good improvement in the EgoMCQ performance when the number of frames is increased to 4. However, after 4, the performance improvement diminishes. We keep 4 as our default frame number for a fair comparison with the baselines [57, 125, 3], who also use 4 frames per sample during pre-training.

### F. Ablations on Downstream

This section presents an ablation on downstream taskspecific heads for EgoNLQ and QFVS.

EgoNLQ: Following EgoVLP [57] and LAVILA [125], we use VSLNet [119] as the task-head for EgoNLQ. However, since our model learns cross-modal features during pretraining, we observe that we can beat the previous methods by a significant margin even using smaller task heads. As shown in Table F.1, when we only use the conditional span predictor module, which is just a linear layer, we can beat EgoVLP by 2.43% R@5 for IoU=0.3. Adding the QGH module further helps in improving the performance. Using the whole VSLNet can significantly beat EgoVLP and LAVILA across all metrics. Moreover, the previous methods train VSLNet for 200 epochs, whereas we achieve the best performance within 100 epochs. These results prove the efficacy of the cross-modal pre-trained representation of EgoVLPv2.

QFVS: Next, we compare different heads for QFVS in Table

- F.2. Notably, this dataset is very small, with only 135 training samples. We observe that a single-layer transformer head performs better than linear layers and multi-layer transformers. Linear layers can not model temporal relations across different video shots, which a transformer can efficiently do. However, multi-layer transformers overfit this dataset due to the small training set. Hence, we use a single-layer transformer for QFVS.
- G. Error Analysis

Although EgoVLPv2 learns impressive cross-modal representation during pre-training, there are still some cases where the model fails to identify tiny and hindered objects, especially in cluttered environments. We show two such examples in Figure G.1. In the first video, the objects ‘bicycle handle’ and ‘T-wrench’ are barely visible even in human eyes, and thus, EgoVLPv2 can not consistently attend to these objects in all frames. However, it can recognize larger, more familiar things like tables and human hands. In the second video, we show an egocentric activity in a wet lab, where the camera wearer is wearing gloves, holding a test tube, and heating a wire using a bunsen burner. This is a complex scenario with multi-agent collaborative activities and fine-grained actions. Interestingly, EgoVLPv2 can correctly identify the human hands and track the motion of the thumb in different frames, even when wearing gloves. However, the test tube and the wire are hindered and are partially attended by the model. Since we pre-train EgoVLPv2 with 224 × 224 video frames, such tiny objects are often hard to be distinguished. However, higher-resolution frames will be more helpful in addressing such intricate scenarios, which we plan to explore in future works.

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

#C C tightens the bolt on the bicycle handle on the table with the T-wrench in his right hand.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

#C C holds the test tube with left hand and heats the wire on the Bunsen burner with right hand.

- Figure G.1: Limitations of our method: tiny and hindered objects in cluttered environments are not distinctly attended by the pre-trained EgoVLPv2. We show the attention maps of the [CLS] token from the text encoder on input video frames in the text-to-video cross-attention module of the last layer of EgoVLPv2. Different heads, shown in different colors, focus on various semantic regions of the video frames. The visualizations are obtained with 960p video frames, resulting in sequences of 3601 tokens for 16 × 16 patches.

### H. Qualitative Downstream Performance

EgoMCQ: In Figure H.1, we show example predictions made by EgoVLP [57] and EgoVLPv2 on multiple choice questions from EgoMCQ validation set. EgoVLPv2 beats EgoVLP substantially on the challenging intra-video setting, where all 5 choices are visually similar. The VTM head pre-trained with hard-negative sampling helps EgoVLPv2 to distinguish between similar videos and boosts the performance over EgoVLP.

QFVS: Figure H.2 shows some examples of query-focused summaries generated by EgoVLPv2 on the QFVS dataset. Given a long egocentric video and a natural language query, our model can summarize all relevant scenes successfully. Notably, the input videos on this dataset are very long (3-5 hours), and the length of the generated summary is 2% input video, which makes this task challenging.

EgoNLQ: Figure H.3 shows examples of predictions made by EgoVLP [57] and EgoVLPv2 on text-guided video localization from the EgoNLQ dataset. Given an untrimmed video and a natural language query, this task aims to predict a single temporal window to answer the query. The predictions of EgoVLPv2 are significantly more aligned with the ground truth than EgoVLP, which supports the impressive quantitative performance gain by EgoVLPv2 over EgoVLP across all metrics.

[Figure 49]

[Figure 50]

(a) Inter-video MCQ. (b) Intra-video MCQ.

- Figure H.1: Examples of predictions made by EgoVLP [57] and EgoVLPv2 on multiple choice questions from EgoMCQ validation set. Left: The “inter-video” setting, each question contains 5 clips from different videos. Right: The “intra-video” setting, each question contains 5 contiguous clips from the same video, making it more challenging.
- Figure H.2: Examples of query-focused video summary generated by EgoVLPv2 on the QFVS daatset. Given a long egocentric video and a natural language query, the generated summary includes all relevant scenes. For example, the query “All the scenes containing streets and trees” summarizes the scenes containing streets and trees in the long input video.

[Figure 51]

[Figure 52]

###### Figure H.3: Examples of predictions made by EgoVLP [57] and EgoVLPv2 on text-guided video localization from the EgoNLQ dataset. Given an untrimmed video and a language query, the prediction is a single temporal window containing the answer to the query. The predictions of EgoVLPv2 are significantly more aligned with the ground truth than EgoVLP.

