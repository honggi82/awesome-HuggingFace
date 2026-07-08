# PEEK: Picking Essential frames via Efficient Knowledge distillation

Killian Steunou1,2

- 1Télécom SudParis — SAMOVAR Institut Polytechnique de Paris Palaiseau, France
- 2Moments Lab 69 Avenue Pierre Grenier Boulogne-Billancourt, France

killian.steunou@ip-paris.fr

Anas Filali Razzouki1,2

arXiv:2605.31029v1[cs.CV]29May2026

anas@momentslab.com

Khalil Guetari2

khalil.guetari@momentslab.com

Mounîm A. El-Yacoubi1

mounim.el_yacoubi@telecom-sudparis.eu

Yannis Tevissen2

yannis.tevissen@momentslab.com

Abstract

Video-language models can process only a limited number of frames, making frame selection a key bottleneck for efficient video captioning. Most captioning pipelines still rely on uniform sampling, which is computationally cheap but agnostic to visual content. Adaptive frame sampling has recently emerged as a promising approach for selecting the most informative frames from a video; however, existing methods remain computationally expensive. We introduce PEEK, an efficient dynamic frame sampling method that distills caption-conditioned frame relevance rankings from a stronger teacher model into a lightweight temporal model that operates only on visual content. We find that, overall, on ActivityNet Captions and MSR-VTT, our method outperforms state-of-theart methods across all evaluated downstream vision language models, especially when only one or two frames are selected for captioning, obtaining the best CIDEr for most frame budgets. On ActivityNet Captions, PEEK is particularly strong, winning 14 out of 16 configurations. Zero-shot evaluation on MSR-VTT shows that our model transfers best at low frame budgets, while results at four and eight frames are more mixed as temporal coverage and visual diversity become increasingly competitive. Compared with recent adaptive baselines, PEEK is both more accurate in the low-budget regime and more efficient: it adds only 5.2% to the captioning time, compared with 65.4% for CSTA and 211.9% for MaxInfo. We release our code and pre-trained checkpoint at https://github.com/momentslab/peek.

## 1 Introduction

Modern vision–language models (VLMs) have made strong progress on image-language tasks, but video understanding remains expensive because videos are long, redundant, and often require sparse relevant cues to be extracted from a frame sequence [26, 30]. To process

© 2026. The copyright of this document resides with its authors. It may be distributed unchanged freely in print or electronic forms.

[Figure 1]

|Temporal Scorer<br><br>🔥| |
|---|---|
| | |

[Figure 2]

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

"A player with a red jersey scores a goal"

❄

[Figure 3]

[Figure 4]

SigLIP2

❄

Mobile CLIP2

[Figure 5]

[Figure 6]

- (b) Stage 2 — query-free temporal scorer

[Figure 7]

- (c) Stratified argmax inference with k = 4

| |
|---|
| |
| |

|Cosine Similarity| |
|---|---|
| | |

[Figure 8]

❄

SigLIP2

(a) Stage 1 — Oracle teacher scoring

Figure 1: Overview of PEEK. (a) A frozen SigLIP2 dual encoder acts as an Oracle teacher, producing per-frame relevance targets from ground-truth captions. (b) A small Transformer distills the teacher’s ranking into a query-free selector operating on MobileCLIP2 visual embeddings alone. (c) At inference, the segment is split into k equal temporal windows and the highest-scoring frame within each (blue dot) is kept.

a video, a VLM usually receives only a limited number of frames: enough to give a glimpse of the video, but not enough to guarantee that the decisive visual cue is present. In practice, even for state-of-the-art models, the default strategy is uniform sampling: partition the video into equal temporal segments and keep one frame from each [1, 2, 17, 36]. Uniform sampling is deterministic, model-free, and often produces good results [10, 11, 33], outperforming adaptive strategies on some benchmarks [4]. This makes uniform sampling a good baseline rather than a naive approach. However, it is still fundamentally content-blind: a short clip where the key event happens in a single instant and a clip where useful evidence is spread across the whole duration are treated identically [10, 11]. Other works use a strong textconditioned retriever, such as CLIP [21] or SigLIP [34] to score every frame by image–text similarity and keep the most relevant frames [23]. This scheme is query-dependent, however, which makes it impossible to consider at inference time for video captioning. Nonetheless, such a selector may be used to select visually relevant frames and diagnose how much they can improve captioning. To address this, we propose a caption-conditioned Oracle as a diagnostic for frame relevance: it scores candidate frames against the ground-truth caption and ranks them according to their semantic alignment with the target description. This Oracle cannot be used at inference time, since the target caption is unknown. However, its rankings provide a useful supervisory signal, indicating which frames are visually salient or semantically aligned with the caption, and may indirectly reveal temporally distinctive moments within the segment. Our hypothesis is that part of this caption-conditioned relevance can be distilled into a lightweight visual model that never sees text at inference time.

Concretely, we propose PEEK, a two-stage distillation framework for efficient frame selection in video captioning. In the first stage, a frozen vision-language teacher scores candidate frames against the ground-truth caption, producing dense caption-conditioned relevance rankings used only for supervision. In the second stage, a lightweight temporal Transformer learns to predict these rankings from visual embeddings alone. At inference time, PEEK requires neither captions nor a text encoder: it scores the video visually and selects frames from the predicted relevance scores. Unlike query-aware or captioner-coupled frame se-

lectors, PEEK uses text-conditioned relevance only as an offline supervision signal. The deployed selector is query-free, caption-agnostic, and independent of the downstream captioning model.

We make the following contributions:

- • We introduce PEEK, a query-free frame selector that distills SigLIP2 [28] captionconditioned rankings into a lightweight temporal scorer operating only on visual features.
- • We propose caption-conditioned frame scoring as an Oracle diagnostic to quantify the value of semantic frame relevance for video captioning.
- • We evaluate PEEK on ActivityNet Captions [12] and MSR-VTT [32] with four downstream VLMs, showing consistent gains in the low-frame regime and a much lower selection cost than recent content-aware baselines.

The paper is organized as follows. Section 2 reviews prior work on frame selection. Section 3 presents PEEK, Section 4 describes the datasets, experimental protocol, and results. Finally, we discuss the limitations of our approach and conclude the paper.

## 2 Related Work

VLMs usually operate under a fixed visual-token or frame budget, which makes temporal sampling a central design choice rather than a neutral preprocessing step. Uniform sampling remains widely used because it is deterministic, model-free, and cheap. It is also a strong baseline: Brkic et al. [4] have recently shown, in a controlled benchmark for small VLMs, that frame-sampling choices can substantially affect video question-answering results, and that uniform sampling is the strongest strategy on VideoMME [9] across the evaluated models. This reinforces the need to compare adaptive selectors against uniform sampling carefully, rather than treating it as a weak baseline. Nevertheless, uniform sampling ignores the large variation in information density across videos.

Training-free frame selection. Recent work replaces uniform sampling with adaptive keyframe selection. Training-free methods often optimize a combination of informativeness, diversity, and temporal coverage. MaxInfo selects representative frames by maximizing the geometric volume spanned by frame embeddings, reducing redundancy while preserving visual diversity [13]. Other adaptive samplers combine text relevance with visual coverage [11, 24, 25, 35], which makes them query-dependent. These methods show that the choice of frames matters, but they are computationally expensive to run, as they require running a large visual encoder over densely sampled frames.

Learned frame selectors. Several works train explicit selectors rather than relying only on hand-designed sampling objectives. Frame-Voyager learns to select informative frame combinations by using a pretrained Video-LLM to rank candidate combinations according to their prediction losses [33]. M-LLM-based frame selection trains a lightweight multimodal selector from pseudo-labels obtained with M-LLM and LLM supervision, including single-frame importance and multi-frame temporal signals [10]. VideoBrain considers sampling as an adaptive acquisition process, where a VLM can invoke complementary agents for

semantic retrieval or local dense sampling depending on information sufficiency [37]. Video summarization methods such as CSTA also learn frame-importance scores, although their objective is summarization rather than captioning [22]. These recent methods highlight that frame selection is increasingly treated as a learnable decision process, but they are often tied to the downstream model or input query.

Frame selection for video captioning. Video captioning is a particularly constrained case for adaptive sampling because only the visual signal is available at selection time. This makes text-aware selectors difficult to apply directly, unless the task is changed or a caption is generated before selection. Earlier work such as PickNet learns to select compact frame subsets for video captioning using reinforcement learning and task-specific rewards [7]. More recently, LFS proposes a learnable frame selector for detailed video captioning that balances event relevance and temporal diversity, and learns from caption feedback produced by frozen video-LLMs [6]. These works are close in motivation because they aim to reduce the number of frames while preserving caption quality, but they differ in how frame importance is supervised. Although it has relatively few learnable parameters, LFS still relies on an expensive vision-language backbone.

Text-conditioned frame scoring. Dual-encoder vision-language models such as CLIP [21] and SigLIP [34] make it natural to score individual frames against a caption or question through image-text similarity. This kind of text-conditioned scoring has been used for video retrieval, keyframe selection, and video data management. KeyVideoLLM, for example, uses text-video frame similarity for large-scale keyframe selection and VideoLLM data compression [14]. More generally, text-conditioned retrieval provides a strong estimate of which frames are semantically aligned with a given sentence, but it requires the text to be known before selection and often involves a dense encoder pass over many candidate frames. Ranking-based learning objectives are a natural fit for this setting, since only the relative order of candidate frames determines which visual evidence is eventually forwarded to the downstream model [20, 31].

When does frame selection matter? Another line of work questions whether current video benchmarks always require precise temporal selection. TempCore introduces Frame Selection Sensitivity and reports that many video question answering examples are largely frame-agnostic, while only a subset is genuinely sensitive to which frames are shown [18]. This perspective is important for evaluating frame selection methods: gains should not be expected uniformly across all datasets, budgets, and downstream models. In practice, learned selection is most likely to help when the visual budget is tight or when the relevant evidence is sparse, while uniform temporal coverage becomes a strong baseline as more frames are allowed.

Positioning of PEEK. PEEK is closest in motivation to learnable frame selectors for video captioning, especially PickNet and LFS, because it also aims to reduce the number of frames while preserving caption quality. However, it differs in its source of supervision and in its deployment cost. Rather than learning frame importance from captioner feedback or using a large vision-language model during selection, PEEK distills caption-frame relevance rankings produced by an Oracle teacher into a small visual temporal scorer. The text-conditioned model is used only to generate supervision. PEEK is also different from

training-free diversity-based methods such as MaxInfo: it does not optimize visual diversity directly, but learns a caption-oriented relevance prior.

## 3 Method

We propose a two-stage framework for learning query-free temporal frame selectors. In Stage 1, a strong text-conditioned vision-language teacher scores every candidate frame of a video segment against its ground-truth caption, producing per-frame relevance signals. In Stage 2, a lightweight temporal scorer is trained to imitate the induced ranking without access to the caption. At inference time it can score each frame from an unseen video to select high-scoring frames before caption generation. Figure 1 summarizes our method.

#### 3.1 Stage 1: Oracle Scoring

Given an annotated temporal segment (v,[ts,te],c), where v is the source video, [ts,te] is the temporal window, and c is the associated caption, we subsample candidate frames from this window, resulting in a set of T frames denoted F = {f1,..., fT}. The reason is that video segments are long and redundant and it is typical in the state-of-the-art to subsample the video before applying a frame selection algorithm. For each training segment we compute a per-frame relevance score using a frozen text-conditioned vision-language teacher. We use SigLIP2 as the teacher: frames are processed by its vision encoder, while the caption is processed by its text encoder. Finally, each frame is assigned a relevance score based on the cosine similarity between its visual embedding and the caption textual embedding. Embeddings are L2-normalized prior to cosine similarity computation.

Let ψv denote the teacher vision encoder and ψt the teacher text encoder. Given teacher frame embeddings zt = ψv(ft) ∈ RdS and the caption embedding u = ψt(c) ∈ RdS with dS being the embedding dimension of SigLIP2, the raw teacher score for frame t is the cosine similarity

st = ⟨zt, u⟩

. (1)

∥zt∥∥u∥

Only the resulting scalar scores are retained as supervision for the student. The caption embedding and teacher visual embeddings are not used as Stage 2 inputs. The vector s = (s1,...,sT) is transformed into a training target. We min–max rescale each st to obtain the final score y = (y1,...,yT) which bounds targets to [0,1] while preserving the teacher’s internal ordering.

#### 3.2 Stage 2: Caption-Agnostic Temporal Scorer

The student model is trained on embeddings obtained from a frozen lightweight MobileCLIP2 [8], while SigLIP2 is used only to produce the supervision targets. Let ϕv be the frozen vision encoder and let

xt = ϕv(ft) ∈ R512, t = 1,...,T, (2) be the visual embedding of frame ft. Given the sequence X = (x1,...,xT), we compute

##### H(0) = [h(10),...,h(T0)]⊤, ht(0) = WinLN(xt)+pt +bin,, (3)

H(ℓ) = TransformerLayer(ℓ) H(ℓ−1) , ℓ = 1,...,L, (4)

yˆ = [yˆ1,...,yˆT], yˆt = w⊤ht(L) +b. (5)

Here, h is the hidden dimension, Win ∈ Rh×512, bin ∈ Rh, LN denotes layer normalization, pt ∈ Rh is a fixed sinusoidal positional encoding, H(ℓ) ∈ RT×h, w ∈ Rh, and b ∈ R. The scalar yˆt is an unconstrained relevance logit for frame t. The model has L encoder layers with multihead self-attention, ReLU-activated feed-forward blocks, and dropout. Our model has about 1.7M trainable parameters, excluding the frozen MobileCLIP2-S0 encoder. In total, it has only 13.1M parameters.

Pointwise regression on (yt,yˆt) pairs ignores the fact that frame selection is fundamentally a ranking problem: only the order among candidate frames affects downstream selection. We therefore use the ListMLE listwise objective of Xia et al. [31]. Let π∗ be the permutation of frame indices sorted by decreasing teacher target, such that π∗(r) is the index of the frame at rank r, and

yπ∗(1) ≥ yπ∗(2) ≥ ··· ≥ yπ∗(T). (6)

Viewing the model outputs yˆ as Plackett–Luce utilities [20], the negative log-likelihood of observing π∗ under the model is

T

### ∑

LListMLE = −

t=1

exp y ˆπ∗(t) ∑Tτ=t exp y ˆπ∗(τ)

. (7)

log

Unlike pointwise MSE or pairwise hinge losses, this objective optimizes the probability of the teacher-induced ranking, aligning the training signal with the selection problem.

#### 3.3 Inference-Time Frame Selection

At test time we score all candidate frames of an unseen segment at once, and select a budget of k frames. Motivated by empirical measurements, we use a simple stratified argmax rule to select frames. We partition the segment into k non-overlapping temporal sub-segments and select the highest-scoring frame inside each sub-segment:

Bj =

(j−1)T k

+1, ...,

jT k

, j = 1,...,k, (8)

yˆt, Sk = (t1∗,...,tk∗). (9)

t∗j = argmax t∈Bj

This policy combines two priors: the scorer chooses content-rich frames locally, while the sub-segments preserve temporal coverage. For k = 1, stratified argmax reduces to selecting the single highest-scoring frame in the video. The selected frames are sorted in temporal order before being forwarded to the downstream captioning model.

#### 3.4 Implementation Details

PEEK is trained on ActivityNet Captions (ANC) train segments only. SigLIP2 (so400m-patch14-384) is used to precompute teacher targets, while MobileCLIP2-S0 is used to precompute the frozen 512-dimensional student inputs. The temporal scorer uses

hidden size h = 256, L = 2 encoder layers, 4 attention heads, feed-forward dimension 1024, and dropout 0.15. We train with ListMLE, AdamW [16] with learning rate 2 × 10−4 and a cosine annealing schedule, weight decay 0.03, batch size 1024, 25 epochs, 2 warmup epochs, and gradient clipping at ∥g∥2 = 1.0. Training uses light temporal augmentation: random frame drop in [0.05,0.25] and random crop with minimum fraction 0.7. For efficient training, sequences are capped at 32 frames per segment, with at least 6 frames retained after augmentation.

## 4 Experiments

[Figure 9]

###### (a) Uniform

###### (b) PEEK (ours)

###### (c) Teacher (SigLIP2)

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

A woman in a polka dot shirt is sitting on a couch and holding a tissue to her nose.

In the video, a man is playing bagpipes while sitting on a couch with a group of people watching him.

A man is playing the bagpipes on a TV show.

GT: A man plays the bagpipes in front of a group of people who are watching and looking apprehensive in reaction to the playing.

Figure 2: Top frames selected on an ActivityNet Captions test segment in which a man plays the bagpipes in front of an audience. From left to right: (a) the uniform (center) frame, (b) the top-ranked frame from PEEK, and (c) the top-ranked frame from the SigLIP2 teacher. The one sentence caption below each frame is generated by Qwen2.5-VL-3B; the ground-truth caption is shown at the bottom. Both PEEK and the teacher find the instrument, while the central frame misses it.

#### 4.1 Data

We train our model on ActivityNet Captions [12], using the official splits and report all metrics on the test set. We also evaluate on the MSR-VTT [32] test split to assess zeroshot transfer to clip-level captioning. Table 1 summarizes the splits used for training and evaluation.

ActivityNet Captions. ANC consists of untrimmed YouTube videos drawn from the ActivityNet dataset [5], each densely annotated with multiple temporally-localized naturallanguage descriptions. A single video contains, on average, between 3 and 4 overlapping or sequential events, with a typical total duration of about two minutes. Every annotated event is described by a free-form English sentence.

Avg. segment duration (s)

Avg. video duration (s)

Avg. words / caption

Dataset Split #Videos #Segments

train 10,009 37,421 35.5 117.3 13.5 val 4,917 17,031 37.7 118.2 13.6 test 4,885 17,505 40.2 118.2 12.0

ANC

MSR-VTT test 2,990 2,990 15.2 15.2 9.3†

†Averaged over all 20 reference captions per video (59,800 captions in total).

Table 1: Statistics of the splits used for training and evaluation.

MSR-VTT. MSR-VTT contains short web video clips paired with 20 crowd-sourced English captions per clip. Unlike ANC, captions describe the entire clip rather than localized events, so each test video contributes a single “segment” whose temporal extent coincides with the clip itself. We use MSR-VTT exclusively for evaluation, in order to probe whether our model trained on ANC videos generalizes zero-shot to clips with a different caption distribution.

#### 4.2 Training and Evaluation

We train PEEK on the ANC dataset [12], which provides videos with temporally grounded natural-language descriptions. Each annotated segment is treated as an independent training clip, decoded at a fixed frame rate of 2 fps. During training, long sequences are capped and shorter sequences are zero-padded with an attention mask to allow batch training. No caption text, sentence boundaries, or external metadata are ever exposed to the Stage 2 model: the scorer sees only visual features and temporal positions.

We evaluate our trained selector on video captioning, on both ANC and MSR-VTT test sets, selecting k ∈ {1,2,4,8} frames that are fed to a downstream VLM. We compare PEEK against five training-free frame selection methods:

- • Oracle is the teacher model which has access to ground-truth captions. Although it cannot be used at inference time, we evaluate it to estimate an approximate upper bound on the sampler’s achievable performance.
- • Uniform splits the (densely) sampled frames into k equal temporal sub-segments and select the center frame of each.
- • Random uses the same temporal sub-segments as Uniform and samples one frame at random from each sub-segment, using a fixed seed shared across all VLMs.
- • MaxInfo [13] selects a diverse, high-information subset by applying a maximumvolume criterion to CLIP image embeddings; we use its fixed-cardinality mode with exactly k selected frames.
- • CSTA [22] is originally a video summarization method that predicts frame-importance scores and selects a summary under a length budget. Since our evaluation requires a fixed number of frames, we adapt only its scoring stage: frames are scored with CSTA, then one highest-scoring frame is selected from each of the k temporal sub-segments.

- • PEEK is our student model trained on ANC with stratified argmax selection.

We fix the parameters and seed for all evaluated downstream VLMs for fair comparison, and use the same candidate frames for all methods. We do not include PickNet [7] or LFS [6] in the quantitative comparison, as we could not find official public implementations or pretrained checkpoints at the time of submission.

For each selected frame budget, we generate captions conditioned only on the k chosen frames, in temporal order, and a short captioning prompt. Concretely, we pass the k selected frames as a single multi-image input to the downstream VLM, followed by a one-sentence prompt; we do not use frame grids, timestamps, or explicit delimiters. We evaluate four downstream VLMs of various sizes: SmolVLM2-2.2B-Instruct [17], Qwen2.5-VL-3B [2], Qwen3.5-4B [27] and Qwen2.5-VL-7B [2]. Prompt templates and model-specific generation settings are provided in the supplementary material. We report CIDEr [29], BLEU-4 [19], METEOR [3], and ROUGE-L [15], with all metrics shown on the same ×100 scale. CIDEr remains the primary metric for discussion because it is the most commonly reported metric for video captioning.

#### 4.3 Results

- 4.3.1 ActivityNet Captions

- Table 2 reports the results on ActivityNet Captions. The results show that PEEK is the strongest query-free selector on this benchmark, obtaining the best CIDEr in 14 out of 16 model/budget settings. The gains are most pronounced at k=1, where PEEK improves over the strongest query-free baseline by +1.74 CIDEr points for SmolVLM2-2.2B, +2.34 for Qwen2.5-VL-3B, +2.18 for Qwen3.5-4B, and +3.00 for Qwen2.5-VL-7B. The same conclusion holds at k=2, where PEEK is again best for all four VLMs, with gains ranging from

+0.61 to +1.75 CIDEr points.

Compared with the adaptive baselines, PEEK is consistently stronger in the low-budget regime. Random is close to Uniform but rarely improves substantially, suggesting that the gains are not explained by simply perturbing the center frame within each temporal subsegment. CSTA is generally below Uniform and PEEK, indicating that frame-importance scores learned for summarization do not directly transfer to caption-oriented frame selection. MaxInfo is the weakest method at k=1 and remains inconsistent at larger budgets, which suggests that visual diversity alone is not equivalent to caption relevance.

At larger budgets, the advantage of PEEK becomes smaller but remains strong on ANC. PEEK is best in three out of four CIDEr settings at k=4, losing only for Qwen2.5-VL-7B, where MaxInfo is higher. At k=8, PEEK is also best in three out of four settings, losing only for Qwen3.5-4B, where Uniform is higher by 0.11 CIDEr points. These small reversals show that PEEK is not universally better than temporal coverage, but it is the most reliable query-free selector on ANC, especially when the frame budget is tight.

4.3.2 Zero-shot on MSR-VTT

- Table 3 evaluates the same ActivityNet-trained selector on MSR-VTT, without retraining. This setting tests whether PEEK learns a transferable visual relevance prior rather than ANCspecific domain distribution. The strongest transfer result is again obtained at k=1. PEEK is the best query-free method for all four downstream VLMs and all reported metrics in the oneframe setting. In CIDEr, it improves over the strongest query-free baseline by +2.68 points

VLM Selector CIDEr BLEU-4 METEOR ROUGE-L k=1 2 4 8 k=1 2 4 8 k=1 2 4 8 k=1 2 4 8

Oracle 37.36 38.19 39.41 39.05 2.65 2.66 2.87 3.31 9.00 9.14 9.33 9.54 20.32 20.50 20.79 20.63 Uniform 29.79 31.23 32.76 33.85 2.10 2.19 2.43 2.94 8.08 8.31 8.59 8.95 18.55 19.00 19.39 19.52 Random 28.40 31.06 32.35 33.47 2.03 2.17 2.40 2.91 7.85 8.25 8.58 8.90 18.18 18.88 19.38 19.49 CSTA [22] 28.37 30.38 32.62 33.69 1.99 2.13 2.41 2.88 7.85 8.17 8.58 8.92 18.20 18.72 19.39 19.50 MaxInfo [13] 27.07 29.49 31.91 33.06 1.92 2.07 2.32 2.79 7.47 8.05 8.47 8.81 17.55 18.60 19.23 19.31

SmolVLM2-2.2B

- PEEK (ours) 31.53 32.98 33.45 34.33 2.23 2.32 2.48 2.96 8.37 8.54 8.72 9.02 19.12 19.39 19.62 19.70

- Qwen2.5-VL-3B

Oracle 37.31 41.67 45.23 46.58 4.04 4.34 4.51 4.54 10.84 11.25 11.53 11.64 21.25 22.05 22.67 22.95 Uniform 30.05 35.36 39.51 42.33 3.19 3.73 4.00 4.12 9.72 10.47 10.88 11.12 19.58 20.81 21.60 22.11 Random 29.03 34.99 39.54 41.91 3.17 3.67 3.96 4.11 9.54 10.39 10.84 11.07 19.35 20.71 21.59 22.05 CSTA [22] 28.56 34.53 39.16 42.06 3.09 3.60 3.88 4.09 9.48 10.29 10.79 11.07 19.21 20.60 21.46 22.02 MaxInfo [13] 26.47 35.10 39.47 41.91 2.82 3.65 3.93 4.09 9.05 10.41 10.89 11.08 18.46 20.71 21.59 22.08 PEEK (ours) 32.39 37.02 40.55 42.42 3.45 3.80 4.06 4.18 10.13 10.61 10.93 11.13 20.21 21.06 21.77 22.12

- Qwen3.5-4B

Oracle 38.44 40.64 40.19 40.30 2.56 3.08 3.12 3.26 9.07 9.76 9.96 10.07 19.35 20.35 20.30 20.37 Uniform 29.55 33.35 34.66 35.42 1.93 2.55 2.63 2.76 7.85 8.72 9.12 9.37 17.28 18.67 19.08 19.30 Random 28.94 32.93 34.72 34.93 1.94 2.50 2.64 2.69 7.69 8.70 9.14 9.30 17.06 18.67 19.09 19.25 CSTA [22] 28.51 33.47 34.54 35.39 1.90 2.50 2.66 2.74 7.63 8.69 9.10 9.33 16.95 18.59 19.02 19.26 MaxInfo [13] 27.01 32.97 34.44 35.22 1.80 2.50 2.60 2.68 7.36 8.70 9.08 9.26 16.56 18.66 18.97 19.21 PEEK (ours) 31.73 34.51 34.93 35.31 2.04 2.61 2.73 2.80 8.13 8.84 9.20 9.37 17.71 18.94 19.22 19.29

Qwen2.5-VL-7B

Oracle 36.60 40.10 38.05 33.88 2.60 2.64 2.27 1.92 9.24 9.64 9.18 8.58 19.59 21.12 20.84 19.69 Uniform 28.54 34.43 33.54 29.40 2.00 2.33 2.06 1.71 8.18 8.91 8.70 8.01 17.88 19.84 19.96 18.70 Random 28.44 33.68 33.55 29.58 2.06 2.20 2.07 1.68 8.08 8.80 8.69 8.01 17.79 19.67 20.01 18.72 CSTA [22] 28.02 33.63 33.02 29.77 2.02 2.25 2.03 1.69 8.03 8.78 8.62 7.99 17.71 19.63 19.89 18.75 MaxInfo [13] 26.44 33.56 33.92 29.43 1.99 2.23 2.12 1.67 7.76 8.88 8.73 7.97 17.20 19.73 20.11 18.72 PEEK (ours) 31.54 35.04 33.15 30.01 2.26 2.31 2.03 1.71 8.58 9.05 8.67 8.06 18.54 20.06 19.92 18.75

Table 2: ActivityNet Captions test captioning metrics with different downstream VLMs and frame budgets. PEEK uses the same ActivityNet-trained checkpoint for all downstream VLMs. Oracle scores frames against the ground-truth caption. Bold marks the best queryfree method for each VLM, metric, and frame budget. Underline is second-best.

for SmolVLM2-2.2B, +1.46 for Qwen2.5-VL-3B, +2.26 for Qwen3.5-4B, and +1.25 for Qwen2.5-VL-7B. This confirms that PEEK transfers particularly well when the selector must identify a single representative frame.

At k=2, PEEK remains the best query-free method for three out of four VLMs in CIDEr. The only exception is Qwen2.5-VL-3B, where Random is higher by 0.19 CIDEr points. Across the remaining metrics, however, PEEK remains highly competitive and is often the best method. These results show that the learned relevance signal is not limited to singleframe selection, but also improves small multi-frame captioning budgets.

At k=4 and k=8, the comparison becomes more mixed. At k=4, PEEK is close to the best query-free method for all VLMs but does not win CIDEr: Uniform is best for SmolVLM2-2.2B, Qwen2.5-VL-3B, and Qwen2.5-VL-7B, while MaxInfo is best for Qwen3.5-

- 4B. At k=8, PEEK is best for SmolVLM2-2.2B and Qwen2.5-VL-7B, while Random and MaxInfo are slightly better for Qwen2.5-VL-3B and Qwen3.5-4B, respectively. This behavior suggests that, on short out-of-domain clips, temporal coverage and diversity become increasingly competitive once several frames are available.

The adaptive baselines are therefore not uniformly stronger than Uniform. MaxInfo performs poorly at k=1, despite explicitly optimizing diversity in CLIP feature space, and only becomes competitive at larger budgets. This supports the idea that diversity is useful when several frames can be selected, but is not a substitute for semantic relevance when only one frame is available. CSTA is generally weaker than PEEK and often below Uniform, suggesting that generic summarization importance does not align perfectly with captioning relevance. Overall, MSR-VTT supports the main conclusion from ANC while making it more precise: PEEK transfers best in the low-budget regime, whereas larger frame budgets reduce the advantage of learned caption-relevance selection.

VLM Selector CIDEr BLEU-4 METEOR ROUGE-L k=1 2 4 8 k=1 2 4 8 k=1 2 4 8 k=1 2 4 8

Oracle 48.33 49.35 48.99 33.99 37.19 37.44 37.66 27.52 27.65 27.81 27.90 26.04 59.47 59.58 59.74 53.44 Uniform 42.15 43.87 46.76 31.42 32.26 34.06 35.79 27.27 25.53 26.02 27.01 25.59 56.01 57.11 58.68 53.12 Random 41.77 44.53 46.18 32.52 31.93 34.30 35.50 26.65 25.22 26.11 26.92 25.51 55.85 57.17 58.20 52.80 CSTA [22] 39.99 43.47 45.49 32.99 31.24 33.91 35.43 27.21 24.75 26.02 26.78 25.72 54.99 57.06 58.29 52.90 MaxInfo [13] 34.36 43.01 44.96 31.49 27.11 33.78 34.89 26.82 22.67 25.82 26.57 25.42 52.54 56.57 57.81 53.01 PEEK (ours) 44.83 46.28 46.67 33.71 34.64 35.65 36.05 26.91 26.59 27.03 27.22 25.87 57.69 58.28 58.73 53.14

SmolVLM2-2.2B

- Qwen2.5-VL-3B

Oracle 33.67 38.10 42.45 44.75 23.90 27.04 30.33 31.67 26.36 27.77 28.82 29.28 50.77 52.96 54.99 55.92 Uniform 29.18 34.57 40.79 42.68 21.74 24.92 29.23 30.43 24.50 26.55 28.12 28.79 48.30 51.17 54.16 55.29 Random 28.34 35.04 39.42 43.37 20.86 24.64 28.13 31.07 24.13 26.32 27.91 28.94 47.72 51.01 53.87 55.51 CSTA [22] 28.20 34.19 40.34 43.09 20.73 24.71 28.60 30.62 23.97 26.41 27.90 28.90 47.51 50.80 53.89 55.24 MaxInfo [13] 21.82 33.76 39.56 43.18 16.88 24.18 28.37 30.79 21.85 26.28 27.98 28.93 44.45 50.91 53.83 55.43 PEEK (ours) 30.64 34.85 40.36 42.94 22.60 25.28 28.97 31.08 25.56 27.00 28.28 28.99 49.53 51.65 54.08 55.51

- Qwen3.5-4B

Oracle 38.80 36.67 35.62 34.26 23.70 23.66 22.79 21.68 24.57 25.48 25.71 25.73 48.94 50.35 49.80 49.38 Uniform 32.63 33.32 33.62 32.77 20.74 21.58 21.32 20.88 22.40 24.29 25.27 25.36 45.96 48.40 49.00 48.65 Random 32.21 32.75 33.84 32.88 20.15 21.18 21.41 21.11 22.25 23.96 25.05 25.19 45.77 48.28 48.88 48.54 CSTA [22] 31.23 32.65 33.63 32.79 19.92 21.63 21.62 20.64 21.97 24.23 24.92 25.20 45.26 48.41 48.66 48.39 MaxInfo [13] 26.10 33.30 33.90 33.03 16.30 21.73 21.56 20.82 19.83 24.27 24.94 25.26 42.78 48.82 48.89 48.49 PEEK (ours) 34.89 33.92 33.84 32.89 21.62 22.37 21.62 21.04 23.18 24.80 25.09 25.37 46.89 49.24 48.82 48.65

Qwen2.5-VL-7B

Oracle 36.63 52.42 53.46 47.08 25.30 37.98 39.63 36.17 25.88 28.55 28.53 26.81 51.54 59.43 59.84 56.90 Uniform 31.08 48.57 51.42 45.04 22.38 35.48 39.10 34.45 23.84 27.11 28.16 26.13 48.55 57.46 59.34 56.00 Random 31.91 47.13 48.95 42.10 22.90 34.30 37.02 32.62 23.36 27.02 27.42 25.40 49.01 57.00 58.47 54.82 CSTA [22] 31.54 46.85 49.12 41.71 22.38 34.61 37.31 32.53 23.18 26.79 27.41 25.23 48.61 56.96 58.43 54.48 MaxInfo [13] 25.60 46.53 49.01 42.30 19.18 34.26 37.09 33.32 21.19 26.96 27.39 25.44 46.19 56.98 58.47 54.96 PEEK (ours) 33.16 48.69 50.99 45.29 23.55 36.00 38.33 35.08 24.82 27.56 27.94 26.34 50.01 58.02 59.08 56.24

- Table 3: Zero-shot MSR-VTT test captioning metrics with different downstream VLMs and frame budgets. PEEK uses the same query-free ActivityNet-trained selector for all downstream VLMs. Oracle scores frames against the ground-truth caption. Bold marks the best query-free method for each VLM, metric, and frame budget. Underline is second-best.

Several captioners also exhibit non-monotonic behavior as the number of frames increases. For Qwen2.5-VL-7B, CIDEr peaks at k=4 and drops at k=8 for all selectors, including the Oracle, whose CIDEr decreases from 53.46 to 47.08 despite using the reference caption for selection. SmolVLM2-2.2B shows an even sharper degradation at k=8. These drops are therefore not specific to PEEK. They suggest that, for some captioners and benchmarks, additional visual context can interact unfavorably with captioning metrics. This cautions against treating more frames as automatically better.

4.4 Efficiency

- Table 4 reports the selection and end-to-end captioning time on the full ANC evaluation split. Uniform and Random sampling have negligible selection cost, while all content-aware methods require an additional scoring pass over the candidate frames. On the ANC evaluation split, PEEK scores all 17,505 segments in 1h44m of GPU time, corresponding to 0.36s per segment. By contrast, CSTA requires 21h58m of GPU time, or 4.52s per segment, while MaxInfo requires 71h04m of GPU time, or 14.62s per segment. The Oracle is also more expensive than PEEK, requiring 9h52m of GPU time, or 2.03s per segment, and is not deployable because it uses the ground-truth caption.

When frame scores are reused for the full k ∈ {1,2,4,8} captioning pipeline, PEEK increases total GPU time by only 5.2% over Uniform. In comparison, CSTA increases the total time by 65.4%, MaxInfo by 211.9%, and the Oracle by 29.4%. Thus, PEEK is not free, but it is a lot cheaper than the other content-aware selectors evaluated here. This efficiency is central to its practical value: PEEK recovers part of the Oracle’s caption-relevance signal while remaining query-free and lightweight enough to be used as a practical preprocessing stage.

Selector Text at inference Selector time Per segment Full pipeline Uniform/Random No negligible – 33h35m PEEK (ours) No 1h44m (26m) 0.36s 35h20m (+5.2%) CSTA [22] No 21h58m (5h31m) 4.52s 55h33m (+65.4%) MaxInfo [13] No 71h04m (17h49m) 14.62s 104h44m (+211.9%) Oracle Yes 9h52m (2h28m) 2.03s 43h27m (+29.4%)

- Table 4: Selection and end-to-end captioning time on the full ActivityNet Captions evaluation split with 17,505 segments, with SmolVLM2-2.2B-Instruct. Timings are measured on 4×NVIDIA A10G GPUs. We report total GPU time, with 4-GPU wall-clock estimates in parentheses. The full pipeline evaluates k ∈ {1,2,4,8}.

Selection strategy CIDEr BLEU-4 METEOR ROUGE-L k=2 4 8 k=2 4 8 k=2 4 8 k=2 4 8

Raw top-k 32.79 36.52 39.47 24.09 26.88 28.96 26.44 27.38 28.26 50.51 52.51 54.02 Stratified argmax 34.85 40.36 42.94 25.28 28.97 31.08 27.00 28.28 28.99 51.65 54.08 55.51

- Table 5: MSR-VTT metrics with Qwen2.5-VL-3B when converting PEEK scores into selected frames using raw top-k or stratified argmax.

#### 4.5 Qualitative analysis

To complement the quantitative results, Figure 3 compares PEEK and SigLIP2 scores on ANC test segments. The two methods agree on global salient regions but often differ locally, with PEEK producing smoother temporal profiles than the frame-wise Oracle. Figure 2 shows one such case: both PEEK and the Oracle identify the bagpipes, while the uniform center frame misses the instrument. Additional examples are provided in the supplementary material.

#### 4.6 Ablations

We ablate two design choices of PEEK on MSR-VTT with Qwen2.5-VL-3B: the inferencetime conversion of frame scores into a fixed-size frame set, and the training loss used to distill the teacher ranking.

First, we compare raw top-k selection, which takes the k highest-scoring frames globally, with stratified argmax, which selects the highest-scoring frame within each of k equal temporal bins. Table 5 shows that stratified argmax consistently improves over raw top-k across all metrics and budgets. This confirms that learned relevance scores alone are not sufficient: temporal coverage remains important to avoid selecting near-duplicate frames around the same high-score region.

Second, we compare the ListMLE loss with a pointwise MSE loss combined with a pairwise ranking loss. As shown in Table 6, ListMLE improves all metrics at both k=1 and k=2, with the largest CIDEr gain at k=1. This motivates our use of a listwise objective. Implementation details can be found in the supplementary material.

PEEK (student) Teacher (SigLIP2) Uniform (center)

1.0

| | |
|---|---|
| | |
| | |

Normalizedscore

[Figure 13]

0.5

0.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

Time (s)

GT: "A couple of people are shown inside a grooming facility."

1.0

Normalizedscore

[Figure 14]

0.5

0.0

0 20 40 60 80 100 120 140

Time (s)

GT: "The lady plays the saxophone in NYC."

1.0

| | |
|---|---|
| | |
| | |

Normalizedscore

[Figure 15]

0.5

0.0

0 2 4 6 8 10 12 14 16

Time (s)

GT: "A man plays the bagpipes in front of a group of people who are watching and looking apprehensive in reaction to the playing."

Figure 3: Per-frame relevance scores on three ActivityNet Captions test segments. Curves are min–max normalized per video and markers indicate the argmax frame for each method. PEEK (red) and SigLIP2 (blue) agree on the global temporal structure but disagree locally, and their top-frame choices differ.

Loss CIDEr BLEU-4 METEOR ROUGE-L k=1 2 k=1 2 k=1 2 k=1 2

MSE + pairwise 29.46 34.49 21.79 25.06 25.14 26.73 48.93 51.54 ListMLE 30.64 34.85 22.60 25.28 25.56 27.00 49.53 51.65

Table 6: MSR-VTT metrics with Qwen2.5-VL-3B for PEEK trained with ListMLE or with an MSE + pairwise loss.

## 5 Discussion and limitations

The results indicate that learned frame selection is most useful when the visual budget is tight. Across both benchmarks, PEEK is the best query-free selector in all one-frame CIDEr settings and in most two-frame settings. This supports the central hypothesis of the paper: part of the caption-conditioned relevance signal produced by an Oracle teacher can be recovered from visual evidence alone. The comparison with CSTA and MaxInfo shows that our method is different from generic video summarization, or visual diversity alone. Instead, PEEK learns a caption-oriented notion of visual relevance that is particularly useful when only one or two frames can be passed to the captioner.

At the same time, the results should not be interpreted as showing that learned frame selection is universally preferable to uniform sampling. Uniform remains a strong baseline, especially when several frames can be forwarded to the captioner. This is particularly visible on MSR-VTT at k=4, where Uniform often obtains the best CIDEr. The likely reason is

the evaluation setting: ANC segments and MSR-VTT clips are relatively short, so a few uniformly spaced frames often cover the main event. As the frame budget increases, the value of selecting the single most relevant frame decreases, while temporal coverage and diversity become more important.

Another limitation is that the teacher signal is derived from ground-truth captions. This makes it useful as Oracle supervision, but it also ties the learned notion of relevance to reference-caption alignment rather than to all visually meaningful events in the video. A frame that supports a correct but non-reference caption may receive a weak teacher score. This limitation is also related to the use of reference-based captioning metrics, which can penalize correct captions that differ from the reference and can behave non-monotonically as more visual context is added. Extending this analysis to longer videos, adaptive frame budgets, and human or model-based factuality judgments would give a more complete picture of when learned frame selection is preferable.

A final limitation is that our evaluation is restricted to short-caption generation. Both ANC segments and MSR-VTT clips are associated with relatively compact descriptions, while long-form video captioning may require preserving multiple events, fine-grained temporal order, and details that are not all captured by a single caption-conditioned relevance ranking. In such settings, selecting only the most caption-aligned frames could overemphasize the dominant event and discard secondary but still important visual cues. Moreover, although PEEK is query-free by design, other video understanding tasks such as video question answering or retrieval may benefit from task- or query-specific frame selection. Our method could still be useful as a lightweight first-stage selector or as a transferable initialization, but evaluating this requires dedicated experiments. Extending the distillation framework to longer descriptions, adaptive frame budgets, and query-conditioned supervision is therefore an important direction for future work.

## 6 Conclusion

We introduced PEEK, a query-free frame selector for video captioning trained by distilling caption-conditioned relevance rankings from an Oracle teacher into a lightweight temporal model. The Oracle provides a diagnostic estimate of what caption-aware selection can recover, while PEEK makes part of this signal usable for caption generation at inference time, when the target caption is unavailable. Across ActivityNet Captions and MSR-VTT, PEEK is the strongest query-free selector compared to the selection methods we evaluate. It obtains the best CIDEr in all one-frame settings, in seven out of eight two-frame settings, and in 23 out of 32 CIDEr comparisons across both benchmarks and all evaluated VLMs.

The gains are clearest when the frame budget is tight. On ANC, PEEK remains strong even at larger budgets, winning 14 out of 16 CIDEr settings. On MSR-VTT, transfer is strongest at k=1 and k=2, while its impact for k=4 and k=8 is more mixed, with Uniform, Random, or MaxInfo occasionally performing slightly better. These results show that caption-relevance distillation is not a universal replacement for temporal coverage or diversity, but a particularly effective strategy when only a few frames can be used.

PEEK also provides a favorable efficiency trade-off. It is much faster than CSTA [22] and MaxInfo [13], while consistently outperforming them in the low-frame regime. This makes it a practical selector for efficient video captioning and a natural candidate for related applications such as thumbnail or preview-frame selection.

## 7 Acknowledgments

This work was granted access to the HPC resources of IDRIS under the allocation 20XX[AD011017404] made by GENCI.

## References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL Technical Report, November 2025. URL http: //arxiv.org/abs/2511.21631. arXiv:2511.21631 [cs.CV].
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, February 2025. URL http://arxiv. org/abs/2502.13923. arXiv:2502.13923 [cs].
- [3] Satanjeev Banerjee and Alon Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In Jade Goldstein, Alon Lavie, Chin-Yew Lin, and Clare Voss, editors, Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. URL https://aclanthology.org/W05-0909/.
- [4] Marija Brkic, Anas Filali Razzouki, Yannis Tevissen, Khalil Guetari, and Mounim A. El Yacoubi. Frame Sampling Strategies Matter: A Benchmark for small vision language models, September 2025. URL http://arxiv.org/abs/2509.14769. arXiv:2509.14769 [cs].
- [5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding. pages 961–970, 2015. URL https://www.cv-foundation.org/ openaccess/content_cvpr_2015/html/Heilbron_ActivityNet_A_ Large-Scale_2015_CVPR_paper.html.
- [6] Lianying Chao, Linfeng Yin, Peiyu Ren, Yifan Jiang, Qiaoyu Ren, Dingcheng Shan, Jing-cheng Pang, Sijie Wu, Xubin Li, and Kai Zhang. LFS: Learnable Frame Selector for Event-Aware and Temporally Diverse Video Captioning, January 2026. URL http://arxiv.org/abs/2601.14594.

- [7] Yangyu Chen, Shuhui Wang, Weigang Zhang, and Qingming Huang. Less Is More: Picking Informative Frames for Video Captioning, March 2018. URL http:// arxiv.org/abs/1803.01457. arXiv:1803.01457 [cs.CV].
- [8] Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, and Hadi Pouransari. MobileCLIP2: Improving Multi-Modal Reinforced Training, August 2025. URL http://arxiv.org/abs/2508.20691. arXiv:2508.20691 [cs.CV].
- [9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis, May 2025. URL http: //arxiv.org/abs/2405.21075. arXiv:2405.21075 [cs].
- [10] Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, and Trishul Chilimbi. M-LLM Based Video Frame Selection for Efficient Video Understanding, March 2025. URL http: //arxiv.org/abs/2502.19680.
- [11] Yuning Huang and Fengqing Zhu. Adaptive Greedy Frame Selection for Long Video Understanding, March 2026. URL http://arxiv.org/abs/2603.20180.
- [12] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. DenseCaptioning Events in Videos, May 2017. URL http://arxiv.org/abs/1705. 00754.
- [13] Pengyi Li, Irina Abdullaeva, Alexander Gambashidze, Andrey Kuznetsov, and Ivan Oseledets. MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding, December 2025. URL http://arxiv. org/abs/2502.03183.
- [14] Hao Liang, Jiapeng Li, Tianyi Bai, Xijie Huang, Linzhuang Sun, Zhengren Wang, Conghui He, Bin Cui, Chong Chen, and Wentao Zhang. KeyVideoLLM: Towards Large-scale Video Keyframe Selection, August 2024. URL http://arxiv.org/ abs/2407.03104.
- [15] Chin-Yew Lin. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/ W04-1013/.
- [16] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International conference on learning representations, 2019. URL https://openreview. net/forum?id=Bkg6RiCqY7.
- [17] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. SmolVLM: Redefining small and efficient multimodal models, April 2025. URL https://arxiv.org/abs/2504.05299v1.

- [18] Hyunjong Ok and Jaeho Lee. TempCore: Are Video QA Benchmarks Temporally Grounded? A Frame Selection Sensitivity Analysis and Benchmark, March 2026. URL http://arxiv.org/abs/2509.01167.
- [19] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a Method for Automatic Evaluation of Machine Translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040/.
- [20] R. L. Plackett. The Analysis of Permutations. Applied Statistics, 24(2):193, 1975. ISSN

00359254. doi: 10.2307/2346567. URL https://www.jstor.org/stable/ 2346567?origin=crossref.

- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision, February 2021. URL http://arxiv.org/abs/2103. 00020.
- [22] Jaewon Son, Jaehun Park, and Kwangsu Kim. CSTA: CNN-based Spatiotemporal Attention for Video Summarization, May 2024. URL http://arxiv.org/abs/ 2405.11905. arXiv:2405.11905 [cs.CV].
- [23] Guangyu Sun, Archit Singhal, Burak Uzkent, Mubarak Shah, Chen Chen, and Garin Kessler. From Frames to Clips: Training-free Adaptive Key Clip Selection for LongForm Video Understanding, December 2025. URL http://arxiv.org/abs/ 2510.02262. arXiv:2510.02262 [cs].
- [24] Wenhui Tan, Ruihua Song, Jiaze Li, Jianzhong Ju, and Zhenbo Luo. Think-ClipSample: Slow-Fast Frame Selection for Video Understanding, January 2026. URL http://arxiv.org/abs/2601.11359.
- [25] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive Keyframe Sampling for Long Video Understanding, February 2025. URL http:// arxiv.org/abs/2502.21271.
- [26] Yolo Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Pinxin Liu, Mingqian Feng, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, and Chenliang Xu. Video Understanding with Large Language Models: A Survey, December 2023. URL https://arxiv.org/abs/2312.17432v7.
- [27] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.
- [28] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua

- Zhai. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features, February 2025. URL http: //arxiv.org/abs/2502.14786.
- [29] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. CIDEr: Consensusbased Image Description Evaluation, November 2014. URL https://arxiv. org/abs/1411.5726v2.
- [30] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding, July 2024. URL https://arxiv.org/abs/2407.15754v1.
- [31] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. Listwise approach to learning to rank: Theory and algorithm. In Proceedings of the 25th International Conference on Machine Learning - ICML ’08, pages 1192–1199, Helsinki, Finland,

2008. ACM Press. ISBN 978-1-60558-205-4. doi: 10.1145/1390156.1390306. URL http://portal.acm.org/citation.cfm?doid=1390156.1390306.

- [32] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5288–5296, Las Vegas, NV, USA, June

2016. IEEE. ISBN 978-1-4673-8851-1. doi: 10.1109/CVPR.2016.571. URL http: //ieeexplore.ieee.org/document/7780940/.

- [33] Sicheng Yu, Chengkai Jin, Huanyu Wang, Zhenghao Chen, Sheng Jin, Zhongrong Zuo, Xiaolei Xu, Zhenbang Sun, Bingni Zhang, Jiawei Wu, Hao Zhang, and Qianru Sun. Frame-Voyager: Learning to Query Frames for Video Large Language Models, March

2025. URL http://arxiv.org/abs/2410.03226.

- [34] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid Loss for Language Image Pre-Training, September 2023. URL http://arxiv.org/ abs/2303.15343. arXiv:2303.15343 [cs].
- [35] Shaojie Zhang, Jiahui Yang, Jianqin Yin, Zhenbo Luo, and Jian Luan. Q-Frame: Queryaware Frame Selection and Multi-Resolution Adaptation for Video-LLMs, July 2025. URL http://arxiv.org/abs/2506.22139.
- [36] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe HansenEstruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena YeungLevy, and Xide Xia. Apollo: An Exploration of Video Understanding in Large Multimodal Models, December 2024. URL http://arxiv.org/abs/2412.10360. arXiv:2412.10360 [cs].
- [37] Junbo Zou, Ziheng Huang, Shengjie Zhang, Liwen Zhang, and Weining Shen. VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding, February

2026. URL http://arxiv.org/abs/2602.04094.

