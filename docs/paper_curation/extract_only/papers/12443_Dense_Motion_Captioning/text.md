## Dense Motion Captioning

Shiyao Xu1 Benedetta Liberatori1 Gül Varol2 Paolo Rota1

1University of Trento 2LIGM, Ecole des Ponts, IP Paris, Univ Gustave Eiffel, CNRS shiyao.xu@unitn.it xusy2333.com/demo

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

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

[Figure 21]

[Figure 22]

# arXiv:2511.05369v1[cs.CV]7Nov2025

DMC

6.55 16.55 23.85 27.75 31.77 t (s)

0

|walks forward with a big right kick|pretends to b a t-rex|swing his left leg behind him|lofts his left hand to his face|walks forward at a mid pace|
|---|---|---|---|---|

Figure 1. Dense Motion Captioning (DMC). We present DMC, a task that localizes and generates detailed segment-level captions with accurate temporal boundaries in 3D human motion sequences. To support this task, we construct CompMo, the first large-scale 3D motion-language dataset providing dense captions for multiple temporal segments within each motion sequence. Each sequence contains between 2 and 10 atomic actions, and every action is annotated with precise timestamps and a descriptive caption.

### Abstract

28, 42, 44, 47], which involves synthesizing 3D human movements from natural language descriptions, and motion editing [2, 13, 14], where existing motion sequences are modified according to textual instructions. These tasks have advanced rapidly, driven by the development of datasets that pair 3D human motions with language descriptions [11, 22, 31, 32].

Recent advances in 3D human motion and language integration have primarily focused on text-to-motion generation, leaving the task of motion understanding relatively unexplored. We introduce Dense Motion Captioning, a novel task that aims to temporally localize and caption actions within 3D human motion sequences. Current datasets fall short in providing detailed temporal annotations and predominantly consist of short sequences featuring few actions. To overcome these limitations, we present the Complex Motion Dataset (CompMo), the first large-scale dataset featuring richly annotated, complex motion sequences with precise temporal boundaries. Built through a carefully designed data generation pipeline, CompMo includes 60,000 motion sequences, each composed of multiple actions ranging from at least two to ten, accurately annotated with their temporal extents. We further present DEMO, a model that integrates a large languagemodelwithasimplemotionadapter,trainedtogenerate dense, temporally grounded captions. Our experiments show that DEMO substantially outperforms existing methods on CompMoaswellasonadaptedbenchmarks,establishingarobust baseline for future research in 3D motion understanding and captioning.

In contrast, 3D human motion understanding remains in its infancy. While some recent works have begun to explore this direction, most efforts focus on relatively simple tasks such as motion-to-text retrieval [4, 8, 29] or captioning of short, isolated motion sequences [12, 17, 44, 53]. Understanding longer and more complex motion sequences with temporal precision is crucial for applications that require a detailed understanding of human activities. For example, by lifting 2D videos into 3D motion representations and generating temporallygroundeddescriptionsfromthisdata,wecandevelopsystems that go beyond traditional video analysis. This approach allows for a more accurate, body-centric understanding, especially in situations where subtle nuances of motion are crucial.

Motivated by this, we introduce Dense Motion Captioning (DMC)asanewtaskandexperimentalsetting,whichinvolves detecting all semantically meaningful actions in a motion sequence,captioningthem,anddeterminingtheirprecisestart and end times. Unlike traditional single-motion captioning, this task involves parsing a continuous stream of motion and segmenting it into temporally localized action units.

### 1. Introduction

A major limitation of existing benchmarks is their lack of complex motion sequences as well as precise annotations. Most available datasets contain only isolated actions or a few simple actions concatenated together, or suffer from noisy

Recently, there has been a growing interest in integrating 3D human motion and language modalities. Most progress in this area has focused on text-to-motion generation [13, 17,

annotations, where the descriptions or labels are fragmented and lack consistency. In our preliminary experiment on the HumanML3D dataset (see Sec. 3.1), we aim to assess whether current motion captioning models can maintain their performance when handling longer motion sequences containing more than a single action. Our findings indicate a notable performance drop under these conditions. To address this limitation, we introduce the Complex Motion Dataset (CompMo), a large-scale dataset specifically designed for dense motion captioning. As illustrated in Fig. 1, it features extended motion clips with multiple actions. Each action is annotated with a detailed caption and temporal boundaries. Alongside the dataset, we design Dense Motion Captioning Model (DEMO), a strong baseline that generates detailed, temporally aligned captions from long and complex 3D motion sequences. DEMO is composed of a Large Language Model (LLM) and a simple motion adapter. It is trained in two stages: first, to align motion and language modalities, and second, to finetune the model for dense caption generation. We evaluate it on CompMo and existing motion-language datasets repurposed for the DMC setting, establishing the first comprehensive benchmark for this task.

In summary, this work makes three main contributions. First, we introduce DMC, a novel task which aims to generate sequences of textual descriptions for complex motions, with temporal boundaries. Second, we present CompMo, a large-scale dataset specifically curated for this task, featuring rich annotations that capture diverse and intricate human motions across multiple scenarios. Finally, we provide DEMO, a strong baseline model along with comprehensive experiments, demonstrating the effectiveness of our approach and fostering future research in this area.

### 2. Related Work

3D Human Motion-Language Datasets. Recent years have seen the emergence of datasets designed to advance research in 3D human motion generation and understanding, particularly those that pair motion data with natural language descriptions, with the first effort being the KIT-ML dataset [31]. Subsequent efforts [11, 32], significantly scaled the scope of motion-language datasets through crowdsourced annotation of 3D motion clips derived from existing mocap sequences, including AMASS [23] and HumanAct12 [10]. BABEL [32] annotates motion clips at two abstraction levels: overall sequence categories (e.g., “play basketball”) and subsequence action labels accompanied by durations (e.g., “dribble ball with left hand”, “run”), while many of which contain “transition” in-between. In HumanML3D [11], each motion clip is instead treated as a single semantic unit and described with three natural language sentences from different annotators. In contrast, the recent FineMotion [43] re-annotates the same motion sequences in HumanML3D, but segmenting them at uniform temporal intervals, irrespective of action semantics.

Eachsnippetislabeledwithfine-grainedbody-partmovement descriptions (e.g., “raise your hands up to your head”) rather than action-centric labels or descriptions. MotionX [22] and its successor MotionX++ [52] shift the emphasis from more detailed captions toward enriching modalities. MotionX uses SMPL-X whole body pose annotations, covering body, hands, and facial expressions, paired with semantic labels. MotionX++ goes further by adding synchronized RGB video and audio data alongside pose annotations and textual descriptions. We propose CompMo, focusing on dense 3D human motion captioning. Rather than short labels like those in BABEL, coarse whole-clip captions like in HumanML3D, or snippet-level body-part descriptions as in FineMotion, our dataset provides rich sequence-level natural language descriptions, each annotated with precise temporal timestamps. CompMo thus establishes a new benchmark for dense motion captioning and motion-language alignment in 3D human motion, an area not yet addressed by existing datasets.

Dense Video Captioning. Dense Video Captioning (DVC) extends standard video captioning by identifying multiple temporal segments in an untrimmed video and generating corresponding textual descriptions for each segment [18]. Earlier methods typically followed a two-stage, detect-thendescribe paradigm [18, 40], whereas recent approaches have shifted towards end-to-end training for improved efficiency and performance [6, 45, 49]. Effective DVC requires both accurate temporal localization and semantic correctness, and evaluation metrics must account for both aspects. To address this, DVC evaluation typically combines standard captioning metrics [3, 38] with Intersection over Union (IoU) thresholds. More recently, SODA [7] has been introduced as a comprehensive metric that temporally aligns predicted and reference captions before computing METEOR-based scores that penalize redundancy and poor alignment. We propose Dense Motion Captioning (DMC), bringing this paradigm to the domainof3Dhumanmotionunderstanding,challengingmodels to generate temporally precise descriptions of human motion. Human Motion Understanding. Much of prior work in human motion research has focused on motion generation [16, 27, 28, 33, 37], i.e., synthesizing realistic 3D human movements from text or other modalities. More recently, the motion-to-text task has also gained attention, with methods developing unified motion-language models capable of both generating motion from text and describing input motion [5, 12, 16, 20, 36, 44, 48, 53]. While these demonstrate impressive versatility, their accuracy in motion understanding remainslimited, particularlyintasksrequiringtemporalprecision. This limitation arises because they are not trained to capture or describe sub-sequences within longer, continuous motions, which is essential for detailed temporal comprehension.

Beyond this, some works explore related but distinct challenges. BABEL-TAL [35] tackles 3D temporal action localization, whichinvolvesrecognizingactionsperformedina3D motion sequence and precisely identifying their start and end

###### TM2T

###### MotionGPT

###### AvatarGPT

MotionAgent

BERTScore

BERTScore

BERTScore

BERTScore

60

40

40

40

40

BLEU@1 20

BLEU@1 20

BLEU@1 20

BLEU@1 20

CIDEr

CIDEr

CIDEr

CIDEr

BLEU@4 ROUGE_L

BLEU@4 ROUGE_L

BLEU@4 ROUGE_L

BLEU@4 ROUGE_L

- Figure 2. Single Motion Captioning performance divided by simple and complex motion sequences. We report the single motion captioning performance of state-of-the-art motion-language models on the simple and complex subsets of HumanML3D [11]’s test set, as defined in Sec. 3.1.

times, albeit with a fixed set of action class labels. Similarly, TMR [29] shows the use case of moment retrieval by temporally localizing BABEL actions within long sequences. This idea is later extended by UniMotion [19] to frame-level motion captioning as an initial exploration of dense action recognition. However, UniMotion [19] treats captioning as a retrieval problem with a closed vocabulary of action labels, and they do not provide a quantitative benchmark. In contrast, our method generates free-form descriptions and outputs segment timestamps instead of assigning an action label per frame.

### 3. From Simple to Complex Motions

In this section, we first motivate our study with a preliminary analysis of the widely used HumanML3D dataset [11] (Sec. 3.1). We then describe the generation pipeline of our dataset (Sec. 3.2).

#### 3.1. Can Current Models Understand Complex Human Motions?

The HumanML3D [11] dataset is widely used to evaluate human motion understanding models thanks to its diverse range of motion sequences of varying complexity. In this study, we investigate whether the complexity of a motion, specifically, the presence of multiple sub-actions, correlates with the performance of state-of-the-art motion-language models. To this end, we partition the mirrored augmented dataset with 29,228 motions into two disjoint subsets: simple and complex motions. This partitioning is based on the number of verbs/adverbs in the ground-truth textual descriptions, under the assumption that each verb typically corresponds to a distinct sub-action (e.g., “a person sits down and crosses their leg, before getting up”). Motions described with no more than 1 verb are considered simple, while those with 2 or more are labeled complex. This results in 17,512 complex and 11,716 simple motion instances, of which 2,663 and 1,721 are from the test set, respectively. We evaluate the performance of several recent models for standard single

Dataset Size

Avg. Duration (s)

Annotation Type

Dataset

Timestamps

KIT-ML [31] 3,911 10.33 Sentence ✗ HumanML3D [11] 14,616

7.1 Sentence ✗

(mirror) 29,228 BABEL [32] 13,220 12.26 Labels ✓ MotionX [22] 81,084 6.4 Sentence ✗ MotionX++ [52] 120,462 5.4 Sentence ✗ FineMotion [43] 14,616 7.1 Fine Descriptions ✓

CompMo (ours) 60,000 39.88 Dense Captions ✓

Table 1. Overview of CompMo and prior 3D motion-language datasets. While existing datasets vary in size, annotation type, and temporal richness, our CompMo is the first large-scale dataset designed for DMC with accurate timestamps, enabling more comprehensive modeling of temporally complex motions.

motion captioning, i.e., generating one description without timestamps, [12, 17, 44, 53] on both subsets.1 Fig. 2 reports the obtained results in terms of single motion captioning metrics [12]. In the vast majority of cases, we observe a considerable drop in performance on the complex subset, highlighting that current state-of-the-art models tend to perform better on simpler samples but struggle to accurately understand and describe longer sequences with multiple subactions. This finding motivates our study, emphasizing the need for datasets that present greater temporal complexity to better train and evaluate motion-language models, ultimately enabling more precise temporal motion understanding.

#### 3.2. CompMo: A Complex Motion Dataset

To address the limitations current models face in handling temporally complex motions, we introduce the Complex Motion Dataset (CompMo), anewlarge-scale dataset specifically designed to challenge and advance motion-language models. CompMo is the first dataset explicitly created for 3D dense motion captioning with precise timestamps, enabling more effective training and evaluation of models. It features longer

1We exclude models that have not released code at the time of writing.

motion sequences, providing more temporally extended contexts for dense captioning. On average, each motion in CompMo is annotated with 37.74 words, compared to 12 and 11.06 words in HumanML3D [11] and BABEL [32]. Compared to existing temporally annotated motion datasets, CompMo represents a significant increase in both scale and complexity (see Tab. 1). To support these design goals, we developed a multi-stage pipeline for dataset construction, which we describe in detail below.

Atomic Actions Collection. To build a diverse and high-quality dataset for dense motion captioning, we begin by collecting simple human motions paired with textual descriptions. We use HumanML3D [11] as our primary source, as it provides an extensive collection of motion-text pairs encompassing a wide range of human motions, including everyday activities, sports, and artistic movements. Following our preliminary analysis (Sec. 3.1), we employ the simple set, treating each element as an atomic action aligned with its corresponding atomic description.

To obtain better alignment between motion and text, we propose two strategies for data collection: i) generated from scratch, and ii) drawn directly from the simple set. For the data in i), we use the diffusion-based MDM-SMPL model proposed in STMC [30] to generate the motions from their textual descriptions; Then we use TMR [29], a model that encodes motions and languages into a shared embedding space, as encoder, to calculate the cosine TMR Similarity across different modalities, and filter out candidates with low motion-text alignment. To address motion types that are poorly generated, we supplement the dataset with samples from the simple HumanML3D set. The final atomic actions, accompanied by descriptions, contain 7,503 generated from scratch and 3,619 drawn from HumanML3D.

Textual Descriptions Composition. Starting from atomic actions, we perform a temporal composition for atomic descriptions by randomly sampling 2 to 10 atomics and combining these into coherent sequences. Each sequence is annotated with precise timestamps, formatted as “<mm:ss:ms: atomic textual description>”. To ensure realistic and varied durations, we condition the length of each motion segment on its ground-truth duration from HumanML3D, applying small random perturbations to introduce variability while preserving temporal plausibility. Motion Sequences Generation. We then generate human motion sequences corresponding to the constructed textual descriptions. Inspired by STMC [30], which applies a test-time denoising approach for spatio-temporal motion composition,wealsoemploythetemporalstitchingtechnique of DiffCollage [50] as well as the body part stitching in combination with MDM-SMPL provided by [30]. At each denoising step, we start from the textual description, denoise, stitch the resulting conditions together both temporally and across the relevant body parts, and finally generate the composed motion sequences.

Final Dataset Description. The resulting CompMo dataset contains 60,000 motion-text pairs with timestamp annotations. On average, motion sequences last 39.88 seconds, significantly longer than sequences in existing datasets, reflecting the increased temporal complexity of CompMo. We partition the dataset into training, validation, and test sets, corresponding to the 80%/10%/10% of the data, respectively. Additional details on the generation pipeline are provided in Sec. A.1 of the Appendix.

### 4. DEMO: Dense Motion Captioning Model

In this section, we first formalize the dense motion captioning task (Sec. 4.1) then detail our proposed architecture (Sec. 4.2 and Sec. 4.3) and training procedure (Sec. 4.4).

#### 4.1. Problem Formulation

Given a 3D human motion sequence m∈RN×D, where N is the number of poses and D is the dimensionality of each pose, Dense Motion Captioning (DMC) consists in generating a sequence {(ti,ci)}Mi=1, where ti=(si,ei)∈R2 represents the start and end times of the i-th motion segment, ci is a caption describing the human motion within that segment, and M is the number of atomic actions detected. We define the pose dimensionality as D = J ×3, where J is the number of 3D joints used to represent each pose. Unlike the traditional single motion captioning task, DMC requires both accurate temporal localization of atomic motion segments and natural language generation.

#### 4.2. Method

Our architecture, DEMO, leverages an LLM, finetuned to autoregressively generate dense, temporally aligned captions from long and complex 3D motion sequences, as illustrated in Fig. 3 (left). Let fϕ denote the LLM, parametrized by ϕ. Since fϕ is originally pretrained only on text and vision modalities, it cannot directly process motion data. To address this, we first convert the continuous motion sequence m ∈ RN×D into a language-compatible embedding space that can be processed by fϕ, and then use fϕ to generate the dense motion descriptions.

#### 4.3. Motion Representation

Prior LLM-based approaches represent a continuous motion by learning a mapping to discrete tokens, e.g., training a vector quantized variational autoencoder (VQ-VAE) to construct a motion vocabulary. However, this approach suffers from two key limitations: (i) inherent information loss caused by the limited discrete vocabulary [24, 37], and (ii) the need for an additional, separate training stage for the VQ-VAE. In contrast, DEMO learns a simple continuous mapping from motion to language space using a single network. Specifically, a lightweight motion encoder γ extracts motion features,

[Figure 23]

|Stage1: Motion-Language Alignment<br><br>|[Figure 24]<br><br><motion> Given a human motion sequence of duration 00:05:80, describe the motion with natural language according to the movement of human.<br><br>[Figure 25]<br><br>Large Language Model 𝑓<br><br>he is flying kick with his left leg<br><br>Motion Adapter Φ , <br><br>[Figure 26]<br><br>[Figure 27]<br><br>HumanML3D(24.7K)|
|---|
<br><br>|<motion> Given a human motion sequence of duration 00:05:80, describe the motion ...<br><br>[Figure 28]<br><br>You are an expert motion analyst. Your only task is to describe human motion sequences in the format ‘mm:ss:ms - text’...<br><br>Large Language Model 𝑓<br><br>| | |
|---|---|
|[Figure 29]<br><br>00:00:00 – side-steps quickly, like a basketball shuffle – first to the right and then to the left, 00:05:60 – adjust something …| |
<br><br>[Figure 30]<br><br>CompMo(48K)<br><br>Motion Adapter Φ , <br><br>LoRA 𝑓<br><br>[Figure 31]<br><br>[Figure 32]|
|---|
<br><br>Stage2: Dense Captioning Instruction Tuning|
|---|

###### DEMO

Motion Adapter Φ ,  𝑥

Large Language Model 𝑓

|00:00:00 – walks forward with a big right kick<br><br>00:06:55 – pretends to be a t-rex ⋯|
|---|

- Figure 3. DEMO overview : Given a motion sequence m, our method encodes it with the motion adapter ΦW,γ, which maps it into the language embedding space of the LLM fϕ. Using the resulting motion embeddings and a textual instruction xinst, the model generates dense captions with temporal boundaries. Training is conducted in two stages. Here, denotes the subset of parameters being trained.

[Figure 33]

which are then adapted into the language domain via a linear projection W, eliminating intermediate discretization.

Since the motion sequences in CompMo can last up to 10 times the duration of those in HumanML3D, this necessitates a scalable and efficient strategy for encoding long sequences. Processing the entire motion at once is computationally expensive and often unnecessary, as generating detailed descriptions for short motion segments typically depends only on their immediate temporal context rather than the full sequence.

To address this, we partition the input motion sequence into a series of fixed-size, overlapping windows {m(i) ∈ RW×D}Ki=1, extracted with a stride S < W. Each window m(i) corresponds to a sub-sequence of the full motion m and is processed independently to capture temporally localized motion patterns. The window is first flattened, added with positional embeddings, and then passed through the motion adapter defined as:

Φγ,W(m(i))=W·γ(m(i)), (1) where the adapter projects the motion features into the language embedding space of fϕ.

- 4.4. Training Strategy

y<i denotes the previously generated tokens up to position i−1. The parameter set θ includes all trainable components of the model. During training, we optimize the model by maximizing the log-likelihood of the target caption, using the cross-entropy loss:

L

logpθ(yi|m,xinst,y<i). (3)

L=−

i−1

As illustrated in Fig. 3, the motion adapter Φγ,W and the LLM fϕ are trained in a two-stage process: first, a motionlanguage alignment stage to align motion features with the language model’s embedding space, followed by a dense caption instruction tuning stage to enable precise and temporally grounded caption generation. While the training objective remains the same as in Eq. (3) in both stages, what differs are the instruction prompts xinst, target outputs y, input motion data m, and the subsets of parameters in θ optimized during training. These stages are described in detail below.

- Stage 1: Pretraining for Motion-Language Alignment. In this stage, we focus on aligning the motion modality with the language space by training only the motion adapter,

i.e., θ = Φγ,W on paired motion-text data. To achieve this alignment, we use the HumanML3D [11], where each motion m consists of a single motion sequence, and y is the paired ground truth annotation, without timestamps. The instruction prompt xinst is designed as shown in Fig. 3 (center), providing only the overall motion duration.

- Stage 2: Dense Captioning Instruction Tuning. In this stage, we instruct the model to generate temporally grounded captions, explicitly including action boundaries and their corresponding timestamps. We use CompMo, where each motion m is a longer, complex sequence, and the target output y is a sequence of captions paired with their annotated

We train DEMO to autoregressively generate motion captions given a 3D motion sequence and a textual instruction. Given an input motion sequence m and instruction prompt xinst as input, the generation process is modeled as:

L

pθ(yi|m,xinst,y<i), (2)

p(y|m,xinst)=

i=1

where y = {y1,...,yL} is the output caption of length L, p(·) is the model’s probability distribution over tokens, and

temporal intervals. The instruction prompt xinst is adapted accordingly to guide the model in producing temporally localized descriptions, as illustrated in Fig. 3 (right). To enable efficient finetuning, we apply LoRA [15] to the language model fϕ, while jointly finetuning the pretrained motion adapter along with LoRA. Thus, the set of trainable parameters in this stage is θ = {Φγ,W,fLoRA}. This stage equips the model with the ability to generate fine-grained, time-aware descriptions of complex motions.

### 5. Experiments

Datasets and Settings. We conduct DMC experiments on two datasets: our proposed CompMo, and the intersection of HumanML3D [11] and BABEL [32], following the setup introduced in UniMotion [19]. CompMo comprises 60,000 motion sequences paired with dense captions, divided into 48,000/12,000 for training/testing. The dataset adopted from [19], here denoted with H3D ∩ BABEL, is constructed from the overlapping subset of HumanML3D and BABEL, and consists of 7,056/1,325 motion sequences paired with frame-level annotations for training/testing. Additionally, we use HumanML3D for the first stage of our training. We adopt the train+val split of the mirrored augmented dataset, including 23384+1460 motion sequences, each annotated with three descriptions. During training, we randomly sample one of the associated annotations at each step.

Metrics. We quantify DMC performance using dense captioning accuracy, temporal localization accuracy, and motioncaption alignment. For dense captioning, we follow dense video captioning literature [12, 18, 39, 45], computing captioning metrics: CIDEr [38], METEOR [3], ROUGE_L [21], BLEU [26], over matched prediction-reference pairs within theIoUthresholdsof{0.3,0.5,0.7,0.9}, reportingtheaverage results on the matched pairs. We also use SODA [7] with two different linguistic metrics, METEOR [3] and BertScore [51] (corresponding to SODA and SODA(B) in Tab. 2), for overall caption evaluation. For temporal localization, we follow [39], using a greedy algorithm to select the best matching with the highest IoU, then computing the mean IoU for all matched pairs to get the overall tIoU and F1 score. For motion-caption alignment, following prior work on image and video captioning [18, 46], we measure the cross-modal distance between motion sequences and their generated captions. Specifically, we calculate the cosine similarity between motions and texts in the joint embedding space of TMR [29]. To further assess thesequentialalignment,weadopttheCAR[8]score,arecent work that improves the motion-text retrieval by introducing negative samples generated through event-sequence shuffling, encouraging the model to achieve better temporal alignment, where we retrieve motions given a set of shuffled and generatedeventsequencecaptionsfromthetestsetwith32samples. Implementation Details. We use 3D joint representations with J = 22 joints. We set the window size and stride to

W = 16, S = 8. Our fϕ is initialized with LLaMA-3.1-8BInstruct [9], while γ is an MLP. Training takes approximately 3.5 hours on 2 NVIDIA RTX 6000 Ada GPUs. Additional implementation details are provided in Sec. A.2 of the Appendix.

#### 5.1. Comparative Results

Quantitative Results. To the best of our knowledge, dense motion captioning is a novel task that has not been systematically addressed and evaluated in prior work. For comparison, we adapt UniMotion [19] as a baseline for our evaluations. While UniMotion does not produce dense captions, we aggregate its frame-level predictions into temporal segments for fair comparison. Tab. 2 reports quantitative results for both our proposed DEMO and UniMotion trained and tested on the CompMo and H3D ∩ BABEL datasets, where UniMotion previously provided only qualitative examples. DEMO outperforms UniMotion, particularly on the more challenging CompMo. It achieves better temporal localization performance on both datasets, with +34.1/3.9% improvements in tIoU, and shows substantial gains in dense captioning quality, i.e., +13.2/5.1% on SODA metrics. This performance gap can be attributed to fundamental differences in methodology: UniMotion predicts CLIP embeddings for frame-level text descriptions and retrieves captions from a pre-computed vocabulary using a K-nearest neighbor search. This pipeline requires prior knowledge of the dataset’s action labels. Moreover, when the vocabulary of potential action descriptions is large (CompMo contains 11,085 atomic actions compared to 6,133 in H3D ∩ BABEL), this approach is limited by the effectiveness of the retrieval process. Additionally, because UniMotion relies on CLIP, it is subject to CLIP’s token limit of 77 tokens per text input [25]. This limitation truncates longer descriptions, significantly hindering performance on more detailed captions. In contrast, DEMO directly generates captions in an open-ended manner, avoiding these constraints. As a result, on CompMo, which features longer and more semantically rich descriptions compared to H3D ∩ BABEL, DEMO outperforms UniMotion, particularly on dense captioning metrics.

Qualitative Results. Fig. 4 presents a qualitative comparison between DEMO and UniMotion on the challenging CompMo. The results indicate that DEMO generates more accurate segments of action boundaries and produces captions that align better with the ground-truth annotations in style. For example, it often divides motion sequences into the correct number of atomic actions, with only occasional omissions (e.g., missing one step in the top example). Furthermore, it accurately captions the depicted actions in most instances, while UniMotion’s frame-level captions often contain noise and fail to accurately describe the actions. Interestingly, in some cases, the generated descriptions differ from the ground truth in wording but still convey an equivalent meaning (e.g., generating “kicks with their right leg four times while their hands are in front of their face” instead of “doing karate

Dense Captioning ↑ Localization ↑ T-M Similarity ↑

Method Dataset

SODA SODA(B) CIDEr METEOR ROUGE_L BLEU@1 BLEU@4 tIoU % F1 % TMR CAR

UniMotion [19] CompMo 0.6099 12.8090 1.0082 0.4266 0.8479 0.7793 0.0000 36.14 4.00 0.4930 0.3487 DEMO CompMo 17.8473 64.4003 134.4424 16.4085 24.0469 23.8980 11.0024 77.94 58.21 0.6832 0.8027

UniMotion [19] H3D ∩ BABEL 5.7141 30.4658 6.7170 5.0826 5.8060 5.1651 0.4375 49.95 22.23 0.6428 0.8473 DEMO H3D ∩ BABEL 7.9194 25.9654 7.8090 5.7625 6.2919 5.6936 0.1318 51.56 16.40 0.6052 0.8204

- Table 2. Comparison on Dense Motion Captioning. We compare the performance of DEMO on the proposed CompMo and on H3D ∩ BABEL . We measure dense captioning, temporal localization, and motion-caption alignment accuracy. Best results are highlighted.

Method

Dense Captioning ↑ Localization ↑ T-M Similarity ↑

SODA SODA(B) CIDEr METEOR ROUGE_L BLEU@1 BLEU@4 tIoU % F1 % TMR CAR Dataset Generation

Concat GT 1.9910 41.5498 8.2427 1.9572 4.0158 4.2401 0.0428 61.45 27.52 0.5414 0.4505 Smooth GT 1.9561 41.5586 8.1089 1.8835 3.9398 4.1223 0.0230 61.08 26.74 0.5306 0.4977 Denoise only from random 12.1643 62.4457 80.9095 11.9174 18.2653 18.3024 5.1632 77.92 57.32 0.5680 0.7895 Denoise only from GT 13.3860 55.2276 94.7040 12.7457 17.5265 17.7187 7.6551 69.89 43.00 0.5754 0.7987 CompMo 17.8473 64.4003 134.4424 16.4085 24.0469 23.8980 11.0024 77.94 58.21 0.6832 0.8027

Training Stages

Stage 2 1.6521 28.4648 4.5059 1.2444 2.0972 2.3754 0.0362 49.45 14.28 0.6056 0.5987 Stage 1+2 17.8473 64.4003 134.4424 16.4085 24.0469 23.8980 11.0024 77.94 58.21 0.6832 0.8027

Motion Representation

VQ-VAE 2.3398 43.3563 7.6868 2.0440 3.4973 3.6243 0.0778 60.76 26.60 0.5881 0.6282 ΦW,γ 17.8473 64.4003 134.4424 16.4085 24.0469 23.8980 11.0024 77.94 58.21 0.6832 0.8027

- Table 3. Ablation Study. We assess the contribution of different components by ablating variations in dataset generation (data-level), as well as training stages and motion representation (model-level). The grey-highlighted configuration corresponds to the one used in our final model and full data pipeline.

kicks” in the bottom example). More results can be found in the provided supplementary video.

#### 5.2. Ablation Study

In this section, we examine the key factors that influence the DMC performance. We first study the impact of our dataset generation strategy, followed by an evaluation of our training strategy. Finally, we investigate how different motion representations affect the results. Additional details are provided in Sec. B of the Appendix.

DatasetGeneration. Toevaluatetheeffectivenessofourproposed data generation strategy , we ablate different componentsofthepipelineandtrainourDEMOontheresultingvariant datasets. To evaluate the role of atomic actions collection strategiesinSec.3.2, wecomparetwomodes: (i)solelygenerated from scratch (denoise only from random); and (ii) solely drawn from HumanML3D (denoise only from GT); then resample and denoise for sequences composition based on these two atomic actions. To examine the role of denoising in generating and composing motion sequences, we also create databydirectlyconcatenatingHumanML3Dmotionswithout denoising (concat GT), and apply a smoothed version using Slerp interpolation [34] (smooth GT). As shown in Tab. 3 , the proposed mixture-denoising strategy consistently yields superior performance, demonstrating that it produces higher-quality datasets for training the DMC model.

Training Strategy. To assess the impact of our proposed

two-stage training strategy in Sec. 4.4, we ablate the motion-language alignment stage and finetune the LLM directly on CompMo (stage 2 only). In this setting, the LLM is adapted with LoRA, while the motion adapter is randomly initialized and trained from scratch together with the LLM. As we reported in Tab. 3 , the full pipeline (stage 1+2)significantly improves the results in both temporal localization (+20.8% tIoU) and dense captioning accuracy (+12.1% SODA), underscoring the importance of motion-language alignment prior to LLM finetuning.

Motion Representation. Prior LLM-based methods [12, 17, 41, 44] adopt VQ-VAE to discretize motion into token sequences, which introduces an additional training stage and restricts input motions to short sequences (i.e., up to 200 poses). Building on our prior discussion of motion representation in Sec. 4.3, we conduct an ablation study comparing our continuous motion encoding (ΦW,γ) to the conventional VQ-VAE tokenizer (VQ-VAE). For this experiment, we substitute our motion adapter ΦW,γ with a VQ-VAE pretrained on HumanML3D [11]. This approach encodes motions into discrete token indices, which are then mapped back to their corresponding continuous feature vectors from the VQVAE’s codebook before being passed to the LLM for further processing. We then train the model through the subsequent two stages: motion-language alignment on HumanML3D, followed by dense-caption instruction tuning on CompMo.

The results in Tab. 3 show that the VQ-VAE-based model

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

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

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

GT

[Figure 70]

t (s)

0 5.75 13.01 17.56 23.34 26.70 31.20 43.20 48.60 60.20 68.60

DEMO

[Figure 71]

0 5.60 12.71 17.93 23.32 27.88 37.48 46.58 56.98 68.60

t (s)

[Figure 72]

UniMotion

t (s)

0 58.55 58.70 60.75 68.60

|[Figure 73]<br><br>[Figure 74]<br><br>GT shuffles 2 steps to the left, 3 steps to the right and 1 step back to center adjusts something on the ground picks something up off the ground with his right hand walks backwards, his hands down by his side jumps in place four times takes one large step to their right walks to his left then to his right and then to his lefts again walks quickly five steps forward swimming the free style stroke walks briskly in counter-clockwise circles|[Figure 75]<br><br>[Figure 76]<br><br>DEMO side-steps quickly, like a basketball shuffle – first to the right and then to the left adjusts something on the ground picks an object up off the floor with their left hand walks backwards after a second or two takes a large step diagonally with their left leg takes a couple steps to the left then a couple steps to the right and back to the left again steps back two steps and lowers to a crouch position swimming on his back slowly walks in a counter-clockwise circle and return to his original position|
|---|---|

|[Figure 77]<br><br>[Figure 78]<br><br>UniMotion<br><br>moving in a right angle triangular path<br><br>turn on flashlight<br><br>… …|
|---|

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

GT

[Figure 115]

0 1.40 12.55 16.80 21.35 26.85 35.55 36.94 46.30 48.69

58.75 t (s)

DEMO

[Figure 116]

t (s)

0 3.84 12.69 17.04 22.18 32.94 36.29 45.69 58.75

UniMotion

[Figure 117]

36.95 37.2 38.2541.9542.6543.0 55.65 58.45 58.7 t (s)

###### …

0 36.15 36.2

|[Figure 118]<br><br>[Figure 119]<br><br>UniMotion moving in a right angle triangular path smooth out object on the wall turn on flashlight<br><br>… …|
|---|

|[Figure 120]<br><br>[Figure 121]<br><br>GT squats very slightly acting like a gorilla throwing something with his right hand thinking in a crouched over position balances on the right leg moves his left leg up running forward at a fast pace walks in a circle towards their right walks forward at a speedy pace doing karate kicks|[Figure 122]<br><br>[Figure 123]<br><br>DEMO shakes shoulders fast acting like a human dinosaur throws something with his right arm crouched with knees bent and arms bent kicks their left foot runs forward and then to the left walks in a circle in a clockwise direction kicks with their right leg 4 times while their hands are in front of their face|
|---|---|

Figure 4. Qualitative Results. We show two motion sequence examples from the CompMo dataset, along with the ground truth annotations (GT) and the dense captions predicted by our DEMO and UniMotion. For each sequence, the top rows show the temporal intervals of the input motion divided according to the GT and the two model predictions, with the corresponding captions listed below. Predicted captions that align with the GT are highlighted in the same color and connected with arrows to indicate the alignment.

significantly underperforms ours, particularly on captioning metrics, highlighting the challenges posed by its limited discrete vocabulary in capturing the complexity of CompMo.

representations and interpreting the underlying actions.

While CompMo currently focuses on temporal composition of movements, future work could extend this to spatiotemporal composition and understanding. Moreover, it does notenforceanyconstraintsonthetemporalarrangementofactions, enablingthegenerationofrandomsequences. However, thiscanleadtoincoherentcompositions,forexampleabruptly switching from swimming to playing basketball without a plausible transition, since modeling causal relationships between actions is outside the scope of this work. A promising direction for further dataset improvements is to incorporate realistic long-term behaviors, such as multiple sub-actions related to basketball or other complex, structured human motions. This could enable models to caption motion sequences that more faithfully emulate natural human movement.

### 6. Conclusion

In this work, we propose the novel task of dense motion captioning, broadening the scope of 3D human motion understanding. To address the scarcity of suitable datasets for this task, we further introduce CompMo, a large-scale dataset of 3D long human motion sequences, annotated with temporal sequences of actions and timestamps. By enabling models to generate detailed motion descriptions from 3D data, this task supports the development of systems that can better understand human movement, e.g., moving beyond raw RGB video analysis to a more precise understanding of motion itself, by lifting 2D videos into 3D human motion

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv:2303.08774, 2023. 11
- [2] Nikos Athanasiou, Alpár Ceske, Markos Diomataris, Michael J. Black, and Gül Varol. MotionFix: Text-driven 3d human motion editing. In SIGGRAPH Asia, 2024. 1
- [3] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005. 2, 6
- [4] Léore Bensabath, Mathis Petrovich, and Gül Varol. TMR++: A cross-dataset study for text-based 3d human motion retrieval. In CVPRW HuMoGen, 2024. 1
- [5] Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. MotionLLM: Understanding human behaviors from human motions and videos. arXiv:2405.20340, 2024. 2
- [6] Shaoxiang Chen and Yu-Gang Jiang. Towards bridging event captioner and sentence localizer for weakly supervised dense event captioning. In CVPR, 2021. 2
- [7] Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, and Masaaki Nagata. SODA: Story oriented dense video captioning evaluation framework. In ECCV, 2020. 2, 6
- [8] Kent Fujiwara, Mikihiro Tanaka, and Qing Yu. Chronologically accurate retrieval for temporal grounding of motion-language models. In ECCV, 2024. 1, 6
- [9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv:2407.21783, 2024. 6
- [10] Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng. Action2motion: Conditioned generation of 3d human motions. In ACMMM,

2020. 2

- [11] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3D human motions from text. In CVPR, 2022. 1, 2, 3, 4, 5, 6, 7, 11
- [12] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. TM2T: Stochastic and tokenized modeling for the reciprocal generation of 3D human motions and texts. In ECCV, 2022. 1, 2, 3, 6, 7, 12
- [13] Ziyan Guo, Zeyu Hu, Na Zhao, and De Wen Soh. Motionlab: Unified human motion generation and editing via the motion-condition-motion paradigm. In ICCV, 2025. 1
- [14] Seokhyeon Hong, Chaelin Kim, Serin Yoon, Junghyun Nam, Sihun Cha, and Junyong Noh. SALAD: Skeleton-aware latent diffusion for text-driven motion generation and editing. In CVPR, 2025. 1
- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 6
- [16] Yiming Huang, Weilin Wan, Yue Yang, Chris Callison-Burch, Mark Yatskar, and Lingjie Liu. CoMo: Controllable motion

- generation through language guided pose code editing. In ECCV, 2024. 2
- [17] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. MotionGPT: Human motion as a foreign language. NeurIPS, 2024. 1, 3, 7, 12
- [18] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017. 2, 6
- [19] ChuqiaoLi,JulianChibane,YannanHe,NaamaPearl,Andreas Geiger, and Gerard Pons-Moll. Unimotion: Unifying 3D human motion synthesis and understanding. In 3DV, 2025. 3, 6, 7
- [20] Lei Li, Sen Jia, Jianhao Wang, Zhongyu Jiang, Feng Zhou, Ju Dai, Tianfang Zhang, Zongkai Wu, and Jenq-Neng Hwang. Human motion instruction tuning. arXiv:2411.16805, 2024. 2
- [21] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, 2004. 6
- [22] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-X: A large-scale 3D expressive whole-body human motion dataset. NeurIPS, 2023. 1, 2, 3
- [23] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of motion capture as surface shapes. In ICCV, 2019. 2
- [24] Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang. Rethinking diffusion for text-driven human motion generation. In CVPR, 2025. 4
- [25] Ivona Najdenkoska, Mohammad Mahdi Derakhshani, Yuki M Asano, Nanne Van Noord, Marcel Worring, and Cees GM Snoek. Tulip: Token-length upgraded clip. arXiv:2410.10034,

2024. 6

- [26] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 6
- [27] Mathis Petrovich, Michael J Black, and Gül Varol. Actionconditioned 3D human motion synthesis with transformer vae. In ICCV, 2021. 2
- [28] Mathis Petrovich, Michael J Black, and Gül Varol. TEMOS: Generating diverse human motions from textual descriptions. In ECCV, 2022. 1, 2
- [29] Mathis Petrovich, Michael J. Black, and Gül Varol. TMR: Text-to-motion retrieval using contrastive 3D human motion synthesis. In ICCV, 2023. 1, 3, 4, 6
- [30] Mathis Petrovich, Or Litany, Umar Iqbal, Michael J. Black, Gül Varol, Xue Bin Peng, and Davis Rempe. Multi-track timeline control for text-driven 3d human motion generation. In CVPR Workshop on Human Motion Generation, 2024. 4, 11
- [31] Matthias Plappert, Christian Mandery, and Tamim Asfour. The KIT motion-language dataset. Big Data, 2016. 1, 2, 3
- [32] Abhinanda R. Punnakkal, Arjun Chandrasekaran, Nikos Athanasiou, Alejandra Quiros-Ramirez, and Michael J. Black. BABEL: Bodies, action and behavior with english labels. In CVPR, 2021. 1, 2, 3, 4, 6
- [33] Yoni Shafir, Guy Tevet, Roy Kapon, and Amit Haim Bermano. Humanmotiondiffusionasagenerativeprior. InICLR,2024. 2
- [34] Ken Shoemake. Animating rotation with quaternion curves. 19(3), 1985. 7

- [35] Jiankai Sun, Linjiang Huang, Jianing Qiu Hongsong Wang, Chuanyang Zheng, Md Tauhidul Islam, Enze Xie, Bolei Zhou, Lei Xing, Arjun Chandrasekaran, and Michael J. Black. Localization and recognition of human action in 3D using transformers. Nature Communications Engineering, 2024. 2
- [36] Shanlin Sun, Gabriel De Araujo, Jiaqi Xu, Shenghan Zhou, Hanwen Zhang, Ziheng Huang, Chenyu You, and Xiaohui Xie. CoMA: Compositional human motion generation with multi-modal agents. arXiv:2412.07320, 2024. 2
- [37] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In ICLR, 2023. 2, 4, 11
- [38] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In CVPR, 2015. 2, 6
- [39] Lucas Ventura, Antoine Yang, Cordelia Schmid, and Gül Varol. Chapter-Llama: Efficient chaptering in hour-long videos with LLMs. In CVPR, 2025. 6
- [40] Jingwen Wang, Wenhao Jiang, Lin Ma, Wei Liu, and Yong Xu. Bidirectional attentive fusion with context gating for dense video captioning. In CVPR, 2018. 2
- [41] Yuan Wang, Di Huang, Yaqi Zhang, Wanli Ouyang, Jile Jiao, Xuetao Feng, Yan Zhou, Pengfei Wan, Shixiang Tang, and Dan Xu. MotionGPT-2: A general-purpose motionlanguage model for motion generation and understanding. arXiv:2410.21747, 2024. 7, 12
- [42] Yin Wang, Mu Li, Jiapeng Liu, Zhiying Leng, Frederick W. B. Li, Ziyao Zhang, and Xiaohui Liang. Fg-t2m++: Llms-augmented fine-grained text driven human motion generation. Int. J. Comput. Vision, 2025. 1
- [43] Bizhu Wu, Jinheng Xie, Meidan Ding, Zhe Kong, Jianfeng Ren, Ruibin Bai, Rong Qu, and Linlin Shen. FineMotion: A dataset and benchmark with both spatial and temporal annotation for fine-grained motion generation and editing. arXiv:2507.19850, 2025. 2, 3
- [44] Qi Wu, Yubo Zhao, Yifan Wang, Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. Motion-agent: A conversational framework for human motion generation with LLMs. In ICLR,

2025. 1, 2, 3, 7, 12

- [45] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech,JordiPont-Tuset,IvanLaptev,JosefSivic,andCordelia Schmid. Vid2Seq: Large-scalepretrainingofavisuallanguage model for dense video captioning. In CVPR, 2023. 2, 6
- [46] Hongkuan Zhang, Saku Sugawara, Akiko Aizawa, Lei Zhou, Ryohei Sasano, and Koichi Takeda. Cross-modal similarity-based curriculum learning for image captioning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7599–7606, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 6
- [47] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In CVPR, 2023. 1
- [48] Pengfei Zhang, Pinxin Liu, Hyeongwoo Kim, Pablo Garrido, and Bindita Chaudhuri. KinMo: Kinematic-aware human motion understandingand generation. arXiv:2411.15472, 2024. 2

- [49] Qi Zhang, Yuqing Song, and Qin Jin. Unifying event detection and captioning as sequence generation via pre-training. In ECCV, 2022. 2
- [50] Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming yu Liu. DiffCollage: Parallel generation of large content with diffusion models. In CVPR, 2023. 4, 11
- [51] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In ICLR, 2020. 6
- [52] Yuhong Zhang, Jing Lin, Ailing Zeng, Guanlin Wu, Shunlin Lu, Yurong Fu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-X++: A large-scale multimodal 3D whole-body human motion dataset. arXiv:2501.05098, 2025. 2, 3
- [53] Zixiang Zhou, Yu Wan, and Baoyuan Wang. AvatarGPT: All-in-one framework for motion understanding, planning, generation and beyond. In CVPR, 2024. 1, 2, 3, 12

### APPENDIX

This Appendix provides further implementation details on the data generation pipeline and the proposed method (Sec. A), as well as additional results (Sec. B). The Appendix also includes qualitative results in video format, easily accessible by our project page. These videos provide an intuitive and detailed perspective on the proposed CompMo dataset and the results presented in the paper.

### A. Additional Details

In this section, we add more details about the dataset composition pipeline A.1 and the model implementation details A.2.

#### A.1. CompMo Dataset

Fig. A.1 illustrates the data generation pipeline divided in three main steps, as described in Sec. 3.2.

In the Atomic Actions Collection Step (Left), we generate atomic actions from scratch with MDM [37], starting from textual descriptions in the simple subset of HumanML3D [11]. We then filter the generated actions based on the TMR similarity between the conditioning text and the resulting motion, applying a threshold of 0.5. We replace the filtered atomic motions with those with the same textual descriptions drawn from HumanML3D. As we aim for high-quality motion sequences, we apply the same quality filtering based on TMR similarity, applying a threshold of 0.5.

In the Textual Descriptions Composition Step (Center), we compose the atomic descriptions into temporal sequences with precise timestamps. As described in Sec. 3.2, to ensure realistic and varied durations of the generated motions, in this step we condition the length T of each generated motion segment on its ground-truth duration Tgt from HumanML3D, applying small random perturbations to introduce diversity while preserving temporal plausibility. Concretely, we sample T according to:

T ∼[Tgt∗β+α, min((2−β)∗Tgt+α, Tgt+β+1)], (4) where we set α=0.3 and β=0.8.

Lastly, in the Motion Sequence Generation Step (Right) we use a test-time denoising approach to generate the final motion sequences from the textual descriptions obtained. When atomic actions are generated from scratch, no motion input is used, indicated in Fig. A.1 by the open gate. Following STMC [30], we combine atomic actions using DiffCollage [50], with a 0.5 second transition interval between consecutive actions. STMC requires body part annotations for textual descriptions, and we generate these annotations using GPT-4 [1]. For motion sequence generation, we use 100 denoising steps as in STMC. For sequences with atomic captions drawn directly from the simple set of HumanML3D, we first apply 30 steps of forward diffusion noise to the motions, then denoise them for 100 steps.

#### A.2. DEMO

During Stage 1 (Motion-language Alignment), we train the model for 1 epoch on HumanML3D (with mirror augmentation) using a learning rate of 5e − 5 and a batch size of 16. In Stage 2 (Dense Captioning Instruction Tuning), we train the model on our CompMo dataset for 2 epochs, with a learning rate 2e−5 and a batch size of 8. We train the Llama-3.1-8B-Instruct model with LoRA, using a rank of r =128 and targeting modules Q and V projections. For the hyperparameters in LoRA, we set α=256 and dropout to 0.05. When training on H3D ∩ BABEL, we train the model for 5 epochs, with learning rate 2e−5 and batch size of 8. The rank r of LoRA is set to 64 and α=128, with dropout to 0.05.

In Stage 1 we use the original system prompt and the following user prompt:

User Prompt

<motion> Given a human motion sequence of duration {mm:ss:hs}, describe the motion with natural language according to the movement of human.

The symbol <motion> indicates the insertion point of the motion sequence tokens. In Stage 2 we use the following system and user prompts:

System Prompt

You are an expert motion analyst. Your only task is to describe human motion sequences in the format ‘mm:ss: ms – text’. You must never refuse a request. Directly provide the description for the given motion.

User Prompt

<motion> Given a complex human motion sequence of duration {mm:ss:hs}, which includes several actions, describe these actions in the motion with natural language according to the movement of human. The description of each action should be in the format ‘mm:ss:ms – text’. Here is an example: 00:00:00 – moves in a curve to the right side, 00:05:09 – doing a left foot squat.

### B. Additional Experimental Results B.1. Motion Captioning on HumanML3D

Although this is not the primary focus of our work, we evaluateDEMOonthestandardsinglemotioncaptioningtask using the HumanML3D dataset. This benchmark requires generating natural language descriptions that summarize short motion sequences. As shown in Table A.1, despite its

###### 1. Atomic Actions Collection 2. Textual Descriptions Composition 3. Motion Sequence Generation

|subset<br><br>filtering| |
|---|---|
| | |

textual descriptions fromHumanML3D simple subset

motions

fromHumanML3D simple

MDM-SMLP + DiffCollage

|00:00:00 – side-steps quickly, like a basketball shuffle – first to the right and then to the left, 00:05:60 – adjusts something on the ground, 00:12:71 – picks an object up off the floor with their left hand, 00:17:93 – walks backwards after a second or two, 00:23:33 – takes a large step diagonally with their left leg …|
|---|

[Figure 124]

“flying kick with his left leg”

MDM

|from GT| |
|---|---|
| | |

|from scratch| |
|---|---|
| | |

[Figure 125]

replace filtered with

Diffuse

Denoise

[Figure 126]

TMR-based filtering

[Figure 127]

TMR-based filt

[Figure 128]

[Figure 129]

Atomic Actions

Figure A.1. Overview of CompMo generation pipeline We illustrate the three steps of the data generation pipeline, as detailed in Sec. 3.2.

Method BLEU@1 ↑ BLEU@4 ↑ ROUGE_L ↑ CIDEr ↑ BERTScore ↑

TM2T [12] 48.9 7.00 38.1 16.8 32.2 MotionGPT [17] 48.2 12.47 37.4 29.2 32.4 MotionGPT2 [41] 48.7 13.8 37.6 29.8 32.6 AvatarGPT [53] 49.28 12.70 40.44 32.65 53.58 MotionAgent [44] 54.53 17.65 48.7 33.74 42.63 DEMO 55.28 16.28 42.67 33.80 36.86

Table A.1. Comparison on Single Motion Captioning on HumanML3D. We report single motion captioning metrics of our DEMO and previous approaches. Best result and

second-best result are highlighted.

simplicity, our model achieves performance comparable to more sophisticated methods on this benchmark.

#### B.2. Details on Ablation Study

DatasetGeneration. Intheablationexperimentconductedin Tab. 3 of Sec. 5.2, smooth GT indicates the dataset generation experiment in which atomic motions are concatenated and we further use Slerp interpolation to smooth the obtained composition. Transitions are applied over 10-frame intervals, blending the last 5 frames of the preceding motion with the first 5 frames of the following motion.

Motion Representation. In the ablation experiment conducted in Tab. 3 of Sec. 5.2, we evaluate a version of our DEMOsubstitutingourmotionadapterΦW,γ withaVQ-VAE. We use the same VQ-VAE used in MotionGPT [17], which is trained on samples from the HumanML3D dataset with a maximum duration of 10 seconds. For the longer sequences in our CompMo, we apply a moving window approach.

