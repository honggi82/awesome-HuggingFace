## RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval

Tyler Skow1* Alexander Martin1* Benjamin Van Durme1,2 Rama Chellappa1 Reno Kriz1,2

1Johns Hopkins University 2Human Language Technology Center of Excellence

{tskow1, amart233, rkriz1}@jhu.edu

# arXiv:2602.02444v2[cs.IR]3Feb2026

### Abstract

Reranking is a critical component of modern retrieval systems, which typically pair an efficient first-stage retriever with a more expressive model to refine results. While large reasoning models have driven rapid progress in text-centric reranking, reasoningbased reranking for video retrieval remains underexplored. To address this gap, we introduce RANKVIDEO, a reasoning-based reranker for video retrieval that explicitly reasons over query-video pairs using video content to assess relevance. RANKVIDEO is trained using a two-stage curriculum consisting of perceptiongrounded supervised fine-tuning followed by reranking training that combines pointwise, pairwise, and teacher confidence distillation objectives, and is supported by a data synthesis pipeline for constructing reasoning-intensive query-video pairs. Experiments on the largescale MULTIVENT 2.0 benchmark demonstrate that RANKVIDEO consistently improves retrieval performance within a two-stage framework, yielding an average improvement of 31% on nDCG@10 and outperforming text-only and vision-language reranking alternatives, while more efficient.1

### 1 Introduction

Platforms across education, entertainment, and social media now host billions of videos, creating a growing demand for effective and scalable retrieval methods. Text-to-video retrieval (video retrieval) addresses this need by ranking large video collections in response to natural language queries. However, the task remains challenging due to the need for strong multimodal representations (Samuel et al., 2025), cross-modal alignment between text and audiovisual content (Reddy et al., 2025; Ma et al., 2025), and the ability to scale to

*Equal Contribution 1https://github.com/tskow99/RANKVIDEO-Reasoning-

Reranker

[Figure 1]

Figure 1: RANKVIDEO judges the relevance between a query-video pair, dynamically reasoning or answering depending on the difficulty of the query-video pair.

large real-world collections containing hundreds of thousands of videos (Kriz et al., 2025).

A common paradigm in information retrieval (IR) for scalable systems is to pair an efficient biencoder (Khattab and Zaharia, 2020; Warner et al., 2024) with a more expressive reranker that refines first-stage results (Reimers and Gurevych, 2019; Pradeep et al., 2023). This two-stage pipeline reduces the number of query-document comparisons, making the slower reranking tractable at query time. While recent work in video retrieval has produced strong first-stage models, (Reddy et al., 2025; Samuel et al., 2025; Faysse et al., 2025; Ma et al., 2025; Xu et al., 2025c), reranking remains largely unexplored. Leveraging textual reasoning models from IR is limited by the usefulness of

extracted text (e.g., captions, transcribed speech, embedded text); this often omits critical visual or audio information, is not always available, and can be computationally expensive to generate.

In contrast, video-native reranking, which uses audiovisual inputs directly rather than relying on extracted text, provides a more robust alternative to text-only rerankers. Inspired by the recent success of Large Reasoning Models (LRMs) in multimodal understanding (Li et al., 2025b; Bai et al., 2025; Feng et al., 2025) and reasoning reranking (Weller et al., 2025; Liu et al., 2025; Yang et al., 2025b), we introduce RANKVIDEO, a video-native reasoning reranker for text-to-video retrieval. Given a query–video pair, RANKVIDEO predicts relevance by comparing the log-probabilities of discrete answer tokens, producing a scalar relevance score without requiring reasoning traces (Figure 1).

To train an effective reranker, we first propose a data synthesis method for creating reasoningintensive queries by leveraging the visual, audio, on-screen embedded text, and metadata features of videos. Using this data, we adopt a two-stage training process. In the first stage, perception-grounded supervised fine-tuning (SFT), the model learns to generate captions grounded in video content. In the second stage, the perception-grounded model is then fine-tuned as a reranker using a unified objective that combines pointwise classification, pairwise ranking, and teacher distillation. The pointwise term trains the model to classify queryvideo pairs as relevant or not, while the pairwise term encourages correct ranking of one positive video against two hard negatives per query. Softrelevance probabilities are distilled from a large reasoning teacher, providing calibrated supervision that captures confidence beyond binary labels.

With this training process, RANKVIDEO achieves substantial retrieval gains across diverse first-stage retrievers, averaging 31% improvement on nDCG@10, while remaining significantly faster than existing reasoning-based reranking baselines. We also observe that the model adaptively allocates reasoning effort, engaging in deeper reasoning only when necessary, which further improves efficiency without sacrificing performance.

Our contributions can be summarized as follows:

- 1. We introduce RANKVIDEO, a video-native reasoning reranker for text-to-video retrieval trained with a two-stage curriculum directly on audiovisual inputs.

- 2. We develop a data synthesis pipeline to generate reasoning-intensive query-video pairs.
- 3. We conduct extensive experiments demonstrating the effectiveness, generalizability, and efficiency of RANKVIDEO compared to strong text-only and vision-language reranking baselines.

### 2 Related

Large Reasoning Models Large reasoning models (LRMs) extend pretrained language and multimodal models with the ability to perform multistep reasoning, often by generating intermediate rationales or explicitly allocating additional computation at inference time (Bai et al., 2025; Xu et al., 2025b; Chen et al., 2025; DeepSeek-AI et al., 2026). Prior work has shown that structured reasoning with chain-of-thought or reinforcement learning can substantially improve performance on complex understanding and decision-making tasks (Wang et al., 2023; Feng et al., 2025; Zhang et al., 2025a). However, the amount of compute spent at test time is a common concern in the literature (Aggarwal and Welleck, 2025; Cheng and Durme,

- 2024; Hao et al., 2025; Yang et al., 2025c).

Reranking Neural information retrieval (IR) commonly adopts a two-stage approach to retrieval, seperating a fast, high-recall first-stage retriever from a more expressive reranker that operates on a samll subset of the first-stage results (Hui et al., 2017; MacAvaney et al., 2019a,b; Pang et al., 2020). Canonically, reranking is performed using a crossencoder that jointly encodes the query and each candidate document to produce a relevance score (Nogueira et al., 2019; Reimers and Gurevych, 2019; Nogueira and Cho, 2020; Pradeep et al., 2023). Recently, large reasoning models have been adapted as rerankers, demonstrating substantial performance gains by explicitly scaling test-time compute for improved relevance judgments (Liu et al.,

- 2025; Sun et al., 2025; Weller et al., 2025; Yang et al., 2025b; Zhang et al., 2025b). While effective, existing approaches are largely developed for textbased retrieval. Our method follows this trajectory but extends reasoning-based reranking to a videonative setting, where relevance judgments require jointly reasoning over visual, audio, textual, and temporal signals.

Text-to-Video Retrieval Video retrieval is a core research area in video-language understanding (Yu

et al., 2018; Yang et al., 2021; Wang and Shi, 2023; Cao et al., 2024; Tang et al., 2025). However, it has traditionally been done with captioning datasets converted to retrieval tasks at small scales (Chen and Dolan, 2011; Hendricks et al., 2017; Krishna et al., 2017; Xu et al., 2016; Wang et al., 2019). MULTIVENT 2.0 (Kriz et al., 2025) introduced a large-scale, more reasoning intensive dataset for video retrieval, which better reflected real world retrieval needs. Kriz et al. (2025) found that stateof-the-art methods don’t scale to 100k+ videos or fail with real-world queries. Instead, for optimal performance and mirroring text IR, video retrieval should be split into two stages, where a first-stage retriever produces a ranked list on the entire index (DeGenaro et al., 2025; Ma et al., 2025; Reddy et al., 2025; Samuel et al., 2025; Zhan et al., 2025) and then a subset of that list is reranked by a more expensive cross-encoder. This work looks to introduce the second stage of this process. Concurrent work also introduces a reranker for video content (Li et al., 2026). While this model performs stateof-the-art on contrived retrieval tasks (e.g., MSRVTT), we find it significantly decreases first stage performance in real-world retrieval settings.

### 3 Data Synthesis

Video retrieval lacks training data for extensive finetuning and challenging retrieval needs that better match real world video retrieval (beyond descriptive captions turned into queries, Kriz et al., 2025). To create high-level, reasoning intensive queries, we first generate and extract text representations of the video content. We caption the videos with QWEN3-OMNI-30B-A3B-INSTRUCT (Xu et al., 2025b), transcribe the audio with WHISPERLARGE-V2 (Radford et al., 2023), and extract OCR with a state-of-the-art multilingual OCR system (Etter et al., 2023). Using these texts as a proxy for the video content, we then provide a text reasoning model (QWEN3-32B, Yang et al., 2025a) 5 variations of the data: caption only, audio only, OCR only, metadata only, and all information.

We filter these queries to a high quality reasoning intensive subset and ensure the queries are not relevant to other videos (e.g., broad queries). To do this filtering, we first discard queries whose true positive video did not exist within the first 1000 candidates returned from OMNIEMBED (Ma et al., 2025). Next, we removed queries whose top first hard negative had a first stage score more then 2x

the size of the true positive. Finally, we discarded query video pairs wrongly classified by REASONRANK-32B, supplying video captions as evidence for judgment. Our filtered dataset contains 35684 records, with 9267 unique positive query-video pairs and 26258 negative query-video pairs. On average each query has 3.85 candidates: 7995 queries have 3 negative samples, 1014 queries have 2 negatives samples, 245 queries have 1 negative sample and 13 queries have 0 negative samples.

### 4 RANKVIDEO Two-Stage Training

With our training data, a mixture of the humanwritten and synthetic queries, we are able to reasonably train a video-native reranker. Our training approach is composed of two stages: (1) perception grounded cold-start supervised SFT and (2) teacher guided ranking optimization.

#### 4.1 Stage 1: Perception Cold Start SFT

An effective video native reranker must be able to reliably extract grounded evidence from raw video, including objects, actions and event context, and then align that evidence with a query. Motivated by recent findings that separating perception from downstream reasoning can improve multimodal training stability and performance (Chen et al., 2025), we introduce a perception grounded ‘cold start’ stage before applying any ranking-specific supervision.

In this stage, we supervise the model to generate teacher provided captions for videos. Captioning provides a direct and dense learning signal for video understanding. Rather than only learning a binary relevance decision, the model is trained to produce an explicit textual description of salient entities present in the clip. Our intuition is that this encourages the model to attend to discriminative visual content.

Training Data Construction We use teacher captions generated offline for the training videos. To bound compute and avoid repeatedly captioning near-duplicate candidates, we restrict this stage to one video per query so that the captioning objective covers a broad set of unique events rather than many candidates for the same query.

Objective Let v denote a video and c(T) = (c(1T),...,c(LT)) its teacher caption token sequence. We fine-tune the model parameters θ to maximize the likelihood of the teacher caption conditioned

[Figure 2]

- Figure 2: RANKVIDEO is trained with a two-stage process. Stage 1 uses a perception-grounded supervised finetuning, where the model learns to generate captions grounded in video content. In Stage 2, for each text query, we sample a query grouped batch containing one positive (relevant) video and one or more negatives (not relevant), and score each candidate using the difference between the logits for yes and no. The model is optimized with a

combined objective: (1) teacher-probability distillation toward pyes, (2) a pointwise loss for stable binary calibration, and (3) pairwise ranking loss that pushes the positive to the top within the query batch.

on the video.

L

Lcap = −

t=1

log pθ c(tT) | c(<tT),v (1)

This stage produces a perception-grounded initialization that we then use as the starting point for Stage 2 ranking fine-tuning. Empirically, we find that this cold-start alone yields improvements.

#### 4.2 Stage 2: Ranking Finetuning

Unlike stage 1, where the objective was detached from the video retrieval task, stage 2’s aim is to directly improve the reranking ability of the model. There are two core components to the effectiveness of this training: hard negative mining and a three part training objective combining pointwise, pairwise, and teacher distillation.

Hard Negative Mining We want to find hard negatives to use in the pointwise portion of the training objective. We do this by partitioning our data into three categories: trusted negatives, suspected positives, and hard negatives, and keep trusted negatives and hard negatives to use during training.

For each query q, let v∗(q) denote its labeled positive video within the first stage candidate pool. We treat the other candidate(s) v ̸= v∗(q) as a

potential negative, but filter and stratify negatives using a reasoning teacher, REASONRANK. Concretely, the teacher provides a binary judgment, yˆT(q,v) ∈ {0,1} and a confidence margin:

δt(q,v) = ℓT(yes|q,v) − ℓT(no|q,v)

where ℓT is the teacher logit at the yes/no decision position. We partition non-positive candidates into:

- • Trusted negatives, where yˆT(q,v) = 0 and δT(q,v) ≤ α1
- • Suspected positives, where yˆT(q,v) = 1 with high margin. We use δT(q,v) > α2. All samples that meet the α2 threshold are dropped to reduce false negative contamination. See Appendix A for details on threshold selection.
- • Hard negatives, consisting of the remaining candidates. We retain ambiguous negatives because they closely resemble the positive under the first stage retriever and thus dominate reranking errors, (see Appendix D).

Training In Stage 2 we train the model to judge query video relevance using a three part objective: pointwise accuracy, pairwise ranking ability, and teacher distilled confidence. Given a query q and

a candidate video v, the model is prompted to answer yes or no using a structured output format <answer>yes/no</answer>. Rather than relying on generated text, we compute a scalar relevance score from the model’s logits at the decision point:

sθ(q,v) = ℓθ(yes | q,v) − ℓθ(no | q,v), (2)

where ℓθ(t | q,v) denotes the model logit for token t at the decision position after the <answer> tag. The logit delta score provides a stable, monotonic ranking signal, and enables fast scoring without decoding long rationales.

Training proceeds on query grouped mini batches. For each query q, we sample a candidate set Bq = {(q,vi)}Ki=1+1 containing one positive v+ = v∗(q) and K negatives {v1−,...,vK−} from the same query.2 Now let si = sθ(q,vi) be the score for each candidate in Bq.

Pairwise ranking loss We define a softmax distribution over candidates within the query batch:

pi =

exp(si/τpair) j exp(sj/τpair)

, (3)

and optimize a batch wise objective that pushes the positive to the top of its query group:

Lpair = −log p+

In Equation 3, we use τpair to prevent early saturation and vanishing gradients as score gaps grow.3 Teacher probability distillation Next, we distill a teacher-provided relevance probability using a temperature-scaled binary cross-entropy with logits (BCEL) on the same score:

Lt = BCEL

sθ(q,v) τteacher

, p(yesT)(q,v) , (4)

This transfers calibrated confidence beyond the binary label and helps align scores across queries. Pointwise loss To stabilize training under class imbalance and provide pointwise supervision, we add a calibration loss with softened negative targets to account for noise in our training data. Let y ∈ {0,1} be the binary relevance label and define

1.0 if y = 1, 0.1 if y = 0,

1.0 if y = 1, 0.5 if y = 0.

(5)

y˜ =

w =

sθ(q,v) τpoint

, y˜;weight = w , (6)

Lpt = BCEL

2In our experiments we use K = 2. 3We use τpair = 10 in our experiments.

Combining Pairwise ranking loss, Teacher probability distillation and Pointwise loss, the final Stage 2 loss is a weighted sum 4:

L = Lpair + λteacher Lt + λptLpt. (7)

### 5 Experiments

Evaluation Setup We evaluate on the MULTIVENT 2.0 (Kriz et al., 2025) test set, consisting of 109,800 videos. For evaluation, we retrieve the top 1000 candidates per query from the first stage retriever. We then score all candidates with the reranker report recall at 10 (R@10), 20 (R@20), 50 (R@50), and 100 (R@100), and normalized discounted cumulative gain (nDCG@N) for the same cutoffs as recall. For all metrics, a higher number indicates better performance.

Results Setup Our main results are reranked from OMNIEMBED (Ma et al., 2025), a dense retrieval model built on Qwen2.5 Omni (Xu et al., 2025a), as the first-stage model. We also explore using four other first-stage systems: (1) MMMORRF (Samuel et al., 2025) a multimodal rank fusion approach; (2) CLIP (Radford et al., 2021) with 16 key frames selected by PYSCENEDETECT (Castellano); (3) LANGUAGEBIND (Zhu et al., 2024) using connected language-vision encoders; (4) and VIDEO-COLBERT (Reddy et al., 2025) a multi-vector late interaction model.

For all first-stage retrievers, we rerank top 100 candidates using a first-stage depth of 1000. Along with RANKVIDEO, we evaluate four other baseline rerankers: (1) REASONRANK (Liu et al., 2025) a text-based reasoning reranker, which reranks the captions and audio transcripts of the video produced by QWEN3-OMNI-30B-A3BINSTRUCT (Bai et al., 2025) and WHISPERLARGE-V2 (Radford et al., 2023); (2) QWEN3VL-8B-INSTRUCT (QVL-I, Bai et al., 2025), a frontier video understanding model; (3) QWEN3VL-8B-THINKING (QVL-T, Bai et al., 2025), the reasoning variant of QVL-I; and (4) QWEN3-VLRERANKER-8B (QVL-R, Li et al., 2026), the reranking variant of QWEN3-VL.

Implementation Details Our base model is intialized from QWEN3-VL-8B-INSTRUCT (QVL-I, Bai et al., 2025). For stage one of training, we use one sample per query in order to generate a single unique caption per video, resulting in a training size set size of 9267. We fix the frame rate at

4We set λteacher = 5 and λpt = 0.5

OE 0.523 0.495 0.598 0.524 0.690 0.556 0.749 0.572 RR

0.570 0.543 0.694 0.585 0.764 0.608 0.800 0.619 8.99 9.70 16.05 11.64 10.72 9.35 6.81 8.22 QVL-I

0.508 0.478 0.60 0.513 0.692 0.544 0.748 0.56 -2.87 -3.43 0.33 -2.10 0.29 -2.16 -0.13 -2.10

0.515 0.483 0.610 0.519 0.719 0.555 0.784 0.572

QVL-T

-1.53 -2.42 2.01 -0.95 4.20 -0.18 4.67 N/A QVL-R

0.537 0.465 0.663 0.517 0.733 0.542 0.749 0.546 2.68 -6.06 10.87 -1.34 6.23 -2.52 N/A -4.55

- RV-1

0.582 0.559 0.661 0.589 0.727 0.612 0.752 0.619

- 11.28 12.93 10.54 12.40 5.36 10.07 0.40 8.21

RV-2

0.590 0.566 0.682 0.599 0.769 0.630 0.820 0.644

- 12.81 14.34 14.05 14.31 11.45 13.31 9.48 12.59

- Table 1: Performance changes from OMNIEMBED first-stage retriever on MULTIVENT 2.0. Each method reports raw scores (top) and deltas as a percentage relative to OMNIEMBED (bottom). Green denotes an increase in performance, while Red denotes a decrease in performance. OE: OMNIEMBED, RR: REASONRANK, QVL-I: QWEN3-VL-8B-INSTRUCT, QVL-T: QWEN3-VL-8B-THINKING, QVL-R QWEN3-VL-RERANKER-8B, RV-1/2: RANKVIDEO Stage 1/2.

- 2 frames per second (FPS), with a maximum 32 frames per video, and train with a learning rate of 1e-5 and batch size of 16. In stage two of training, we form mini-batches of one positive and two negative queries. This yields a dataset of 7995 distinct queries with 23985 total videos. The final data mixture used to train our model contains 1361 human written queries and 7906 synthetic queries.

#### 5.1 RANKVIDEO and Baselines

- In Table 1, we show the results of the reranking methods on MULTIVENT 2.0 with OMNIEMBED (OE) as the first-stage retriever. We see that RANKVIDEO achieves state-of-the-art retrieval results across all metrics, increasing the change in metric performance significantly more than any other reranking baseline. Both Stage-1 and Stage-2 of RANKVIDEO are able to significantly increase reranking performance, with the first and second best results across each metric. We find that the only other method to increase performance from the first stage results is REASONRANK, with strong gains over the first-stage results. The two other video-native baselines are not able to increase performance, struggling to judge the relevance of query-video pairs. QVL-I is unable to improve upon the first stage results. QVL-T is able to improve the recall from a cutoff of 20 and above, but unable to improve nDCG. This result means that

QVL-T is able to remove easy negatives (lower in the ranked list), but struggles to rank relevant videos highly, especially against harder negatives in the first-stage results.

These results demonstrate three core findings. (1) Vision language or reasoning models (QVLI/T) are not able to effectively judge the relevance of videos in zero-shot. This finding largely aligns with hallucination calibration literature (Li et al., 2025a; Guan et al., 2024), as we find QVL-I/T struggle with low precision as a result of a high false positive rate (QVL-I precision 0.055, QVL-T precision 0.037). (2) The gain in performance of RANKVIDEO over REASONRANK demonstrates the importance of performing video-native reranking, incorporating heterogeneous multimodal signals into relevance judgments instead of extracting captions and transcripts, which are not information exhaustive. (3) Reranking models trained for contrived video-retrieval tasks5 do not generalize well to real-world video retrieval (mirroring the first stage findings of Kriz et al., 2025). When comparing to QVL-R, we see the most significant decrease in performance from the first stage results, even compared to models not calibrated for reranking.

5Video retrieval tasks that are captioning datasets converted to retrieval datasets

[Figure 3]

- Figure 3: Training Stage-2 increases score separation in the reranking regime. Empirical CDF of the reranker

score sθ(q,v) = ℓθ(yes|q,v) − ℓθ(no|q,v) for relevant and non relevant query video pairs. Stage 2 shifts relevant pairs towards larger positive margins and suppresses non relevant candidates towards more negative margins, reducing overlaps in the score distributions within reranking candidate pools.

#### 5.2 Score Distribution Shift

RANKVIDEO ranks candidates using logit delta score which provides a monotonic scalar ranking signal without decoding long rationales. Because reranking occurs over the first state top-1000 candidate pool, the dominant error mode is high scoring false positives among hard negatives. Figure 3 shows that Stage 2 training reshapes the score distribution in a label consistent way: relevant pairs shift toward substantially larger positive margins, while non relevant candidates are pushed towards more negative margins, increasing separation and reduction overlap. This distribution shift is key for improving rank quality, especially in early ranks, since top-k metrics are most sensitive to the impact of high scoring hard negatives outranking the true item. This results suggest training does not just improve point wise correctness, but also calibrates the scores to match the reranking objective. Additionally, the increased separation suggests that the logit margin score can serve as a more reliable confidence signal.

#### 5.3 Stability Across First-Stage Model

- In Table 2 we explore the stability of RANKVIDEO across the other first-stage retrieval methods on MULTIVENT 2.0. Like OmniEmbed, we observe a significant gain in performance from first-stage to second-stage results for all first-stage methods. On a strong first stage retriever (MMMORRF), we con-

[Figure 4]

Figure 4: Median query latency for Qwen3VL Instruct/Thinking (QVL-I/T) and RANKVIDEO stages 1 and 2. Latency is computed as the mean for 100 queryvideo pairs with a batch size of 1. All evaluations are run with batch size 1 as larger batches exceed GPU memory for VLMs.

tinue to see solid gains between ranked lists. Most promisingly, we see the largest and most significant gains (>0.1) on the weakest first-stage retrievers.

These results demonstrate that RANKVIDEO is beneficial to any first-stage retriever. Additionally, RANKVIDEO is not dependent on the quality of the initial candidate list, allowing for the use of faster, more efficient (although less accurate) firststage models to handle large indices and rely on RANKVIDEO to refine the ranked list.

#### 5.4 Efficiency and Dynamic Reasoning

In Figure 4 we compare the query latency of ReasonRank, QVL-T/I/R and RANKVIDEO Stages 1 and 2 (RV-1/2) on a randomly selected subset of 100 query-video pairs from MULTIVENT 2.0, reporting the median query latency on these 100 instances.6 We find that RANKVIDEO is much more efficient than the alternative baseline video-native reasoning model (QVL-T), with a difference of 2.67s between RANKVIDEO Stage 2 and QVL-T. We attribute the large difference in query latency because QVL-T produces a reasoning trace for every query-video pair.

In Figure 4, we observe that RANKVIDEO

performs within 0.15s of ReasonRank (1.02s vs. 0.87s). We include a preprocessing latency penalty for ReasonRank to contextualize the offline compute costs for captioning. While our model does require captions for training, at inference, RANKVIDEO’s reliance soley on video frames avoids expensive preprocessing latency. This comes at the trade off of only a small gap in model

6We chose 100 instances to match the reranking cutoff.

CLIP 0.333 0.306 0.419 0.339 0.522 0.373 0.603 0.394

0.477 0.478 0.540 0.503 0.584 0.520 0.603 0.525 43.24 56.21 28.88 48.38 11.88 39.41 N/A 33.25

+ RV

MRF 0.611 0.585 0.697 0.620 0.779 0.649 0.828 0.663

0.634 0.639 0.725 0.639 0.800 0.665 0.828 0.673 3.76 8.45 4.02 3.06 2.70 2.47 N/A 1.51

+ RV

LB 0.355 0.326 0.445 0.359 0.550 0.393 0.620 0.412

0.498 0.487 0.560 0.512 0.607 0.530 0.620 0.533 40.28 49.39 25.84 42.62 10.36 34.86 N/A 28.37

+ RV

VCB 0.341 0.422 0.402 0.447 0.471 0.477 0.518 0.494

0.448 0.535 0.490 0.549 0.514 0.560 0.518 0.561 31.38 26.78 21.89 22.82 9.13 17.40 N/A 13.56

+ RV

- Table 2: Impact of RANKVIDEO applied to different first-stage retrievers on MULTIVENT 2.0. Each method reports raw retrieval performance (top) and absolute deltas as a percentage relative to its corresponding baseline retriever (bottom). Green indicates an improvement in ranked list quality. CLIP: CLIP on 16 frames, MRF: MMMORRF, LB: LanguageBind, VCB: Video-ColBERT, RV: RANKVIDEO.

Method α-nDCG nDCG StRecall InfoP

CLIP 0.538 0.498 0.724 81.5 +RV 0.628 0.629 0.826 91.9

LB 0.476 0.457 0.634 84.0 +RV 0.566 0.563 0.754 94.5

VCB 0.431 0.383 0.634 83.8 +RV 0.553 0.522 0.734 87.1

OE 0.530 0.454 0.721 88.0 +RV 0.584 0.587 0.778 91.2

MRF 0.540 0.503 0.724 94.4 +RV 0.605 0.617 0.785 91.8

- Table 3: Retrieval and generation results on WIKIVIDEO. The retrieval results are calculated using claim-based relevance and a cutoff at 10.

used for document relevance; evaluating retrieval with MIRAGE (Martin et al., 2025b), reporting Information Precision (InfoP) for article factuality; and generating articles using CAG with a QWEN3-VL backbone (Bai et al., 2025; Martin et al., 2025a). In Table 3, we find that RANKVIDEO substantially improves upon the claim coverage of the top 10 videos provided to the generation system, which leads to a large increase in article factuality. This demonstrates the effectiveness of RANKVIDEO, not only as an effective reranker for traditional retrieval metrics, but also as a crucial step in a multimodal RAG pipeline, increasing diversity of information for generation.

#### 5.6 Qualitative Results Discussion

We analyzed the failure modes of RANKVIDEO by examining query performance grouped by metadata attributes. Overall, we observe relatively stable performance across, event types, modalities and some performance stratification across languages (see Appendix C).

latency vs. text based rerankers (ReasonRank). ReasonRank’s latency values relative to all other models should be interpreted as an amortized per candidate estimate, since ReasonRank is listwise and evaluates a sliding window of candidates simultaneously.

To quantify whether coarse metadata can explain which queries the reranker performs poorly on, we fit a regression model to predict per query nDCG@10 using video language, query event type, video type, video modality, and query length. We randomly split queries into train (75%) and test (25%) and train a Random Forest Regressor. On held out queries, the model achieves R2 = 0.093, indicating that these coarse metadata features explain only a small fraction of variance in per query

#### 5.5 Downstream Impact in RAG

We explore the effect of RANKVIDEO in a retrieval augmented generation setting. We perform RAG with the WIKIVIDEO dataset (Martin et al., 2025a), evaluating retrieval with three claim-based relevance judgments α-nDCG, nDCG, and StRecall, where the number of supported article claims are

###### Query Before ↓ After ↓ ∆ ↓

100 1 -99 Korea Blockchain Week 2022 98 5 -93

KAMAZ Autonomous Mining Dump Truck

- 2018 Mark Zuckerberg senate hearing

- 92 1 -91

Sixth SpaceX operational mission

- 93 2 -91

- 2019 Cairo International Film Festival

2 1 -1

2019 Moscow International Film Festival

1 1 0

Digital Services Act 1 1 0 2019 Typhoon Lekima 2 5 +3 Typhoon Maysak 2 6 +4 2018 opinion rigging scandal in South Korea

11 44 +33

- Table 4: Most improved and degraded queries after reranking. Before: rank without reranking. After: rank with reranking. ∆: change in rank (After − Before); lower is better. The maximum rank is capped at 100.

nDCG@10. Feature importances suggest query length is the most informative single feature, consistent with the intuition that shorter/underspecified queries are more difficult.

Separately, we test whether RANKVIDEO ’s ranking scores exhibit query or video specific priors by decomposing variance in the reranker score s(q,v). Let Nq denote the number of scored candidates for query q, and define the query mean score µq = N1

q v s(q,v). Let Nv de-

note the number of queries for which video v is scored, and define the per video mean score µv = N1

v q s(q,v). We evaluate three simple predictors of s(q,v): (i) query only sˆ(q,v) = µq, (ii) video only sˆ(q,v) = µv, and (iii) additive sˆ(q,v) = µq + µv − µ where µ is the global mean. The resulting R2 values are 0.139 (query only) and 0.090 (video only), indicating that RANKVIDEO ’s scoring is not dominated by a video level prior and instead depends substantially on query–video interaction. In contrast, baseline rerankers exhibit stronger video priors (e.g., QVL-R: R2 = 0.755). In Table 5 we see the broader trend that the weaker performing re-ranker also appear to have strong modality bias.

Qualitatively, we attribute the performance of RANKVIDEO to its effectiveness with videos and queries that have visually anchorable events. Consider, for example, the queries that see that greatest improvements in Table 4. Mining dump trucks and SpaceX operational mission are object specific and have discrete visuals (trucks and rockets). On the other side, we attribute the negative performance of

Method Query Only Video Only Q+V QVL-I 0.192 0.119 0.262 RR 0.033 0.105 0.130 QVL-T 0.295 0.265 0.431 QVL-R 0.108 0.755 0.739 RV 0.139 0.090 0.200

Table 5: Query/video prior strength via variance decomposition. Stronger biases underlined.

the other vision baselines to non-visual or weakly visual topics, like "2018 opinion rigging scandal in South Korea." One confounding category of events the model appears to occasionally struggle with, despite intuition suggesting their ought to be an abundance of distinct visually grounded signals, is natural disasters. A plausible explanation is, storm footage, for example, may share more generic visuals (rushing water) rather than unique cues needed for quality reranking, leading to negative results.

### 6 Conclusion

In this work, we introduce RANKVIDEO, a novel approach for video-native reasoning reranking. RANKVIDEO is trained with a two stage process. The first stage perception-grounded SFT teaches the model to generate captions grounded in video content, while the second stage finetunes the stage 1 model for effective reranking by combining pairwise, pointwise, and distillation objectives. On MULTIVENT 2.0, a large scale video retrieval task, RANKVIDEO consistently enhances retrieval performance across various first-stage retrievers, achieving an average improvement of 31% on nDCG@10. Not only is RANKVIDEO the most effective reranker, but it is also significantly faster than existing reasoning-based reranking baselines. Future work in video reranking should look towards new training objectives, like list-wise or grouped reranking which exhibit strong performance gains over their pointwise counterparts in text IR (Liu et al., 2025; Sun et al., 2025; Yang et al., 2025b). Additionally, exploring further optimizations for dynamic reasoning, the disconnect between accuracy and retrieval performance, and improving performance on reasoning-intensive or ambiguous video types suggest ample direction for future work. In applications, reranking and modified objectives could be explored for retrievalaugmented generation to provide better optimized results to article generation models.

### Limitations

Computational Costs In this work, we did not explore list-wise reranking, even though it’s benefit over pointwise is well supported in the literature, because of the computational costs of multivideo inference. To train our pairwise objective (at most

- 3 videos per query), required greatly reducing the batch size and max frames to fit on 8 80GB A100s for training. Efforts towards making multi-video inference more computationally feasible will help reduce this burden. Acknowledgments

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE2139757. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

### References

Pranjal Aggarwal and Sean Welleck. 2025. L1: Controlling how long a reasoning model thinks with reinforcement learning. Preprint, arXiv:2503.04697.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Meng Cao, Haoran Tang, Jinfa Huang, Peng Jin, Can Zhang, Ruyang Liu, Long Chen, Xiaodan Liang, Li Yuan, and Ge Li. 2024. RAP: Efficient text-video retrieval with sparse-and-correlated adapter. In Findings of the Association for Computational Linguistics: ACL 2024, pages 7160–7174, Bangkok, Thailand. Association for Computational Linguistics.

Brandon Castellano. Pyscenedetect. Www.scenedetect.com.

David Chen and William Dolan. 2011. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 190–200, Portland, Oregon, USA. Association for Computational Linguistics.

Yan Chen, Long Li, Teng Xi, Long Zeng, and Jingdong Wang. 2025. Perception before reasoning: Two-stage reinforcement learning for visual reasoning in visionlanguage models. Preprint, arXiv:2509.13031.

Jeffrey Cheng and Benjamin Van Durme. 2024. Compressed chain of thought: Efficient reasoning through dense representations. Preprint, arXiv:2412.13171.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2026. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Dan DeGenaro, Eugene Yang, David Etter, Cameron Carpenter, Kate Sanders, Alexander Martin, Kenton Murray, and Reno Kriz. 2025. FORTIFY: Generative model fine-tuning with ORPO for ReTrieval expansion of InFormal NoisY text. In Proceedings of the 1st Workshop on Multimodal Augmented Generation via Multimodal Retrieval (MAGMaR 2025), pages 100–115, Vienna, Austria. Association for Computational Linguistics.

David Etter, Cameron Carpenter, and Nolan King. 2023. A hybrid model for multilingual ocr. In Document Analysis and Recognition - ICDAR 2023, pages 467– 483, Cham. Springer Nature Switzerland.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. Colpali: Efficient document retrieval with vision language models. Preprint, arXiv:2407.01449.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. 2025. Video-r1: Reinforcing video reasoning in mllms. Preprint, arXiv:2503.21776.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. Preprint, arXiv:2310.14566.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2025. Training large language models to reason in a continuous latent space. Preprint, arXiv:2412.06769.

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing moments in video with natural language. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 5804–5813.

Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2017. Pacrr: A position-aware neural ir model for relevance matching. Preprint, arXiv:1704.03940.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. Preprint, arXiv:2004.12832.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Dense-captioning events in videos. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 706–715.

Reno Kriz, Kate Sanders, David Etter, Kenton Murray, Cameron Carpenter, Hannah Recknor, Jimena Guallar-Blasco, Alexander Martin, Eugene Yang, and Benjamin Van Durme. 2025. Multivent 2.0: A massive multilingual benchmark for event-centric video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24149–24158.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2026. Qwen3-vl-embedding and qwen3vl-reranker: A unified framework for state-of-theart multimodal retrieval and ranking. Preprint, arXiv:2601.04720.

Zongxia Li, Xiyang Wu, Guangyao Shi, Yubin Qin, Hongyang Du, Fuxiao Liu, Tianyi Zhou, Dinesh Manocha, and Jordan Lee Boyd-Graber. 2025a. Videohallu: Evaluating and mitigating multi-modal hallucinations on synthetic video understanding. Preprint, arXiv:2505.01481.

Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, and Dong Yu. 2025b. Self-rewarding vision-language model via reasoning decomposition. Preprint, arXiv:2508.19652.

Wenhan Liu, Xinyu Ma, Weiwei Sun, Yutao Zhu, Yuchen Li, Dawei Yin, and Zhicheng Dou. 2025. Reasonrank: Empowering passage ranking with strong reasoning ability. arXiv preprint arXiv:2508.07050.

Xueguang Ma, Luyu Gao, Shengyao Zhuang, Jiaqi Samantha Zhan, Jamie Callan, and Jimmy Lin. 2025. Tevatron 2.0: Unified document retrieval toolkit across scale, language, and modality. Preprint, arXiv:2505.02466.

Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019a. Cedr: Contextualized embeddings for document ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’19, page 1101–1104. ACM.

Sean MacAvaney, Andrew Yates, Kai Hui, and Ophir Frieder. 2019b. Content-based weak supervision for ad-hoc re-ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’19, page 993–996. ACM.

Alexander Martin, Reno Kriz, William Gantt Walden, Kate Sanders, Hannah Recknor, Eugene Yang, Francis Ferraro, and Benjamin Van Durme. 2025a. Wikivideo: Article generation from multiple videos. Preprint, arXiv:2504.00939.

Alexander Martin, William Walden, Reno Kriz, Dengjia Zhang, Kate Sanders, Eugene Yang, Chihsheng Jin, and Benjamin Van Durme. 2025b. Seeing through the mirage: Evaluating multimodal retrieval augmented generation. Preprint, arXiv:2510.24870.

Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage re-ranking with bert. Preprint, arXiv:1901.04085.

Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-stage document ranking with bert. Preprint, arXiv:1910.14424.

Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, and Jirong Wen. 2020. Setrank: Learning a permutation-invariant ranking model for information retrieval. Preprint, arXiv:1912.05891.

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. Rankzephyr: Effective and robust zeroshot listwise reranking is a breeze! Preprint, arXiv:2312.02724.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. Preprint, arXiv:2103.00020.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR.

Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno Kriz, Celso M. de Melo, Benjamin Van Durme, and Rama Chellappa. 2025. Video-colbert: Contextualized late interaction for text-to-video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19691– 19701.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. Preprint, arXiv:1908.10084.

Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eisape, Tanner Spendlove, Arun Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Carpenter, David Etter, Efsun Kayi, Matthew Wiesner, Kenton Murray, and Reno Kriz. 2025. Mmmorrf: Multimodal multilingual modularized reciprocal rank fusion. Preprint, arXiv:2503.20698.

Duolin Sun, Meixiu Long, Dan Yang, Yihan Jiao, Zhehao Tan, Jie Feng, Junjie Wang, Yue Shen, Peng Wei, Jian Wang, and Jinjie Gu. 2025. Grouprank: A groupwise reranking paradigm driven by reinforcement learning. Preprint, arXiv:2511.11653.

Haoran Tang, Meng Cao, Jinfa Huang, Ruyang Liu, Peng Jin, Ge Li, and Xiaodan Liang. 2025. Muse: Mamba is efficient multi-scale learner for text-video retrieval. Preprint, arXiv:2408.10575.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, YuanFang Wang, and William Yang Wang. 2019. Vatex:

- A large-scale, high-quality multilingual dataset for video-and-language research. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4580–4590.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. Preprint, arXiv:2203.11171.

Yimu Wang and Peng Shi. 2023. Video-text retrieval by supervised sparse multi-grained learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 633–649, Singapore. Association for Computational Linguistics.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. 2024. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. Preprint, arXiv:2412.13663.

Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and Benjamin Van Durme. 2025. Rank1: Test-time compute for reranking in information retrieval. Preprint, arXiv:2502.18418.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025a. Qwen2.5-omni technical report. Preprint, arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025b. Qwen3-omni technical report. Preprint, arXiv:2509.17765.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msrvtt: A large video description dataset for bridging video and language. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5288–5296.

Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even Oldridge, and Benedikt Schifferer. 2025c. Omniembed-nemotron: A unified multimodal retrieval model for text, image, audio, and video. Preprint, arXiv:2510.03458.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Eugene Yang, Andrew Yates, Kathryn Ricci, Orion Weller, Vivek Chari, Benjamin Van Durme, and Dawn Lawrie. 2025b. Rank-k: Test-time reasoning for listwise reranking. Preprint, arXiv:2505.14432.

Jianwei Yang, Yonatan Bisk, and Jianfeng Gao. 2021. Taco: Token-aware cascade contrastive learning for video-text alignment. Preprint, arXiv:2108.09980.

Junjie Yang, Ke Lin, and Xing Yu. 2025c. Think when you need: Self-adaptive chain-of-thought learning. Preprint, arXiv:2504.03234.

Youngjae Yu, Jongseok Kim, and Gunhee Kim. 2018. A joint sequence fusion model for video question answering and retrieval. Preprint, arXiv:1808.02559.

Jiaqi Samantha Zhan, Crystina Zhang, Shengyao Zhuang, Xueguang Ma, and Jimmy Lin. 2025. Magmar shared task system description: Video retrieval with omniembed. Preprint, arXiv:2506.09409.

Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, and Dacheng Tao. 2025a. Consistent paths lead to truth: Selfrewarding reinforcement learning for llm reasoning. Preprint, arXiv:2506.08745.

Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and Aishwarya Agrawal. 2025b. REARANK: Reasoning re-ranking agent via reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2458– 2471, Suzhou, China. Association for Computational Linguistics.

Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. 2024. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. Preprint, arXiv:2310.01852.

### A RANKVIDEO Details

We provide additional details about our training configurations in Table 6 and provide the system and user prompts for RANKVIDEO (Figure 11, Figure 12), QVL-I/T (Figure 13, same as RANKVIDEO), and ReasonRank (Figure 14). For hard negative mining, we used thresholds α1 = −6 and α2 = −8 after reviewing the distribution of logit scores of all query, video pairs.

Setting Value Learning Rate Schedule cosine

Learning Rate 1e-5 Warmup Proportion (Linear) 0.03

Optimizer AdamW Frame rate (FPS) 2

Stage 1 Data 9267/9267

- Stage 1 Batch Size 16

- Stage 1 Epochs 1

- Stage 1 Max Frames 32

Stage 2 Data 7995/23985

- Stage 2 batch Size 3 Stage 2 Epochs 2

- Stage 2 Max Frames 24

- Table 6: Training settings for RANKVIDEO. Stage 1/2 Data is written as queries/videos.

### B Training Loss Ablation

We include the ablation results of our three part loss objective in Table 7 to understand which signals are responsible for reranking gains. This ablation includes our pairwsie ranking loss (P), pointiwse calibration loss (PT), and teacher probability distillation (T). We report results on a 50 query subset of the MULTIVENT 2.0 test set. We observe substantial gains from the inclusion of the pointwise objective. The inclusion of the teacher probability distillation term resulted in an improved nDCG.

### C RANKVIDEO Performance Across Query Types

To better understand whether our gains are concentrated in a subset of the evaluation distribution, we break down RV retrieval performance across several metadata slices. Specifically, we report per query nDCG@10 aggregated by video language

- (Figure 6), query event type (Figure 5), video type
- (Figure 7), and video modality (Figure 8).

### D Disconnect Between Binary Classification and Reranking

One failure mode we observed during our training is a disconnect between binary classification7 and reranking quality. During development, multiple models achieved strong accuracy and recall (Table 8) when evaluated as relevance classifiers.

7In this case, yes/no judgments for the relevance of a queryvideo pair.

[Figure 5]

- Figure 5: RANKVIDEO nDCG@10 by query event type. Multi-word categories are abbreviated as acronyms in the plot (e.g., PD=Political Development, LD=Launch/Discovery, SE=Social Events). Only attributes with ≥ 30 test queries are included.

[Figure 6]

- Figure 6: RANKVIDEO nDCG@10 by video language (en=English, es=Spanish, ar=Arabic, zh=Chinese, ru=Russian, ko=Korean). Only attributes with ≥ 30 test queries are included.

However, these metrics are not indicators of strong second-stage results because they largely reflect performance on easy negatives.

Classifiers blindly trained to increase accuracy end up being insensitive to the error regime that dominates second-stage retrieval. When reranking, the model is only given the top-k candidates returned by a first-stage retriever, where negatives are hard—semantically or visually plausible nearmisses—and therefore disproportionately likely to become high-scoring false positives. While accuracy and reranking performance are slightly positively correlated, we observe that higher accuracy does not always mean strong reranking results.

To address this, we designed our training supervision to better match the reranking objective by mining negatives from each query’s candidate pool and filtering them using teacher confidence. Specifically, we removed likely false negatives by dropping suspected positives. We thresholded suspected positives as non-qrels candidates that the teacher labeled as relevant with high margin. Among the remaining candidates, we retained trusted negatives

Method R@10 nDCG@10 R@20 nDCG@20 R@50 nDCG@50 R@100 nDCG@100

OE 0.554 0.541 0.614 0.566 0.711 0.599 0.755 0.614 P, PT, T 0.650 0.640 0.694 0.657 0.731 0.672 0.755 0.678 P, PT 0.650 0.630 0.687 0.644 0.733 0.663 0.755 0.668 P 0.630 0.620 0.668 0.634 0.705 0.649 0.755 0.661

- Table 7: Performance changes from OMNIEMBED first-stage retriever on MULTIVENT 2.0. OE: OMNIEMBED; P, PT, T: Our full loss objective; P, PT: Our loss objective without teacher distillation; P: Pairwise loss.

[Figure 7]

Figure 7: RANKVIDEO nDCG@10 by video type. Professional: e.g., news broadcasts with reports; Edited: e.g., videos with multiple spliced clips and visual effects; Diet Raw (DR): single-stream videos with minimal text/speech overlays; Raw: continuous, unedited streams. Only attributes with ≥ 30 test queries are included.

Method Accuracy R@10 nDCG@10 OE − 0.52 0.50

- Neg-1 0.74 0.48 0.52

- Neg-2 0.76 0.52 0.54

- Neg-3 0.79 0.54 0.54

- Table 8: Evidence of the disconnect between accuracy and reranking performance. Neg: negative result while training RANKVIDEO.

[Figure 8]

Figure 8: RANKVIDEO nDCG@10 by query/video modality. OCR=queries written from text visible in video frames; Text=queries written using only the YouTube description; Base=Wikipedia-title-style queries; Speech=queries written from spoken content; Specific=queries targeting fine-grained aspects of events. Only attributes with ≥ 30 test queries are included.

video pair, while Figure 10 shows an example where RANKVIDEO needed to reason. In the example with a reasoning trace, the large difference in reasoning length between QWEN3-VL-8BTHINKING and RANKVIDEO is evident.

### F Use of AI Assistants

AI assistants were used to improve the fluency of the writing and in some of the code development.

that the teacher rejected with a large negative margin, and ambiguous negatives that remain difficult but are flagged with lower confidence. We required at least one trusted negative per query. This curation shifts learning away from separating irrelevant videos and toward suppression of retrieval hard negatives while avoiding noisy supervision from unlabeled positives.

### E Reasoning Examples

In Figure 9 and Figure 10 we provide examples of reasoning traces from the baseline and RANKVIDEO rerankers. Figure 9 shows an example where RANKVIDEO did not need to reason to produce a correct answer with a query-

Query: Emergency response Notre-Dame fire

REASONRANK ✗ Pred: no | Gold: yes <think> The video shows interior damage to Notre-Dame after the fire. While related to the event, it does not depict firefighters or emergency responders. Since the query is about emergency response actions, the video is judged not relevant. </think> <answer>no</answer>

QWEN3-VL-8B-THINKING ✗ Pred: no | Gold: yes The video depicts post-fire damage inside Notre-Dame Cathedral. No emergency response activities (e.g., firefighters or rescue operations) are visible. Therefore, the video is not relevant to the emergency response query. <answer>no</answer>

###### Captioning model

The footage shows catastrophic damage to Notre-Dame Cathedral following a deadly fire. Debris and structural collapse are visible inside the building. The cause of the fire remains under investigation.

###### RANKVIDEO ✓ Pred: yes | Gold: yes <answer>yes</answer>

- Figure 9: Reasoning traces for the query “Emergency response Notre-Dame fire”. ReasonRank and Qwen3-VL8B-Thinking models are substantially more verbose then our model. Our final model correctly predicts relevance. Video ID: 45391

Query: Super Bowl 2023 Philadelphia Eagles

REASONRANK Pred: no | Gold: no <think> Okay, let me try to figure this out. The query is about the Super Bowl 2023 Philadelphia Eagles. So, the user wants to know if the video is relevant to that. Looking at the video content, it starts with a close-up of Richard Sherman from the 49ers. Then there’s info about his salary from 2020-2022. Next, there’s game footage of him tackling someone in a yellow uniform (maybe another team?). Then he’s celebrating on the field. The video ends with a blue screen and social media stuff. Now, the query mentions Philadelphia Eagles and Super Bowl 2023. The video mostly talks about Richard Sherman’s salary and some game play. There’s no mention of the Eagles or the 2023 Super Bowl. The focus seems to be on Sherman’s career stats and a play. Since the query is specifically about the Eagles in the 2023 Super Bowl, and the video doesn’t cover that, I think it’s not relevant. Unless maybe the Eagles are involved in the footage, but the caption says Sherman tackles someone in yellow, so probably not Eagles. So the answer should be no. </think> <answer> no </answer>

###### QWEN3-VL-8B-THINKING Pred: no | Gold: no

Got it, let’s check the query: Super Bowl 2023 Philadelphia Eagles. Now, look at the video content. The video shows a player in a San Francisco 49ers uniform (jersey number 20, helmets with 49ers logo), social media prompts, salary info for 2020-2022, game action with 49ers, and other teams like Eagles? Wait, let’s see. The video has players from the 49ers, and there’s a part where the team is in red and gold, which is 49ers. The query is about Super Bowl 2023 Philadelphia Eagles, but the video seems to focus on the 49ers, not the Eagles. Also, Super Bowl 2023 isn’t actually a thing yet (2023 would be Super Bowl LVII, but the Eagles’ Super Bowl is like 2023 vs the Chiefs maybe? Wait, but the video doesn’t mention Philadelphia Eagles or Super Bowl 2023. Let’s check all segments. The video starts with a 49ers player, then salary info, then a game action (49ers), then another team (Patriots vs Seahawks?), then a thank you screen. There’s no mention of Philadelphia Eagles or Super Bowl 2023. The query is about Eagles in 2023 Super Bowl, but the video is about 49ers, so it’s not relevant. <answer>no</answer>

###### Captioning model

A San Francisco 49ers football player in his white jersey with the number 20 is shown on the field. The video then displays a financial breakdown of his contract for the years 2020, 2021, and 2022, detailing his base salary, guaranteed amounts, bonuses, and total salary-cap hits. Action shots from games are shown, including a player celebrating with arms outstretched. The video concludes with an outro screen thanking viewers and asking them to subscribe.

RANKVIDEO ✓ Pred: no | Gold: no The video is not relevant to the query. The query asks about Super Bowl 2023 and the Philadelphia Eagles, but the video content shows a San Francisco 49ers player (number 20) in a Super Bowl LVII jersey, along with salary cap information for the years 2020-2022, and other football-related images. There is no mention or depiction of the Philadelphia Eagles or Super Bowl 2023 specifically. <answer>no</answer>

- Figure 10: Reasoning traces for the query “Super Bowl 2023 Philadelphia Eagles”. In this instance, all models correctly classify the video as not relevant but we see ReasonRank and Qwen3-VL-8B-Thinking use substantially more tokens then RANKVIDEO. Video ID: 45391

|[System Prompt] You are a helpful assistant specialized in video and text understanding. Given a video, your task is to produce an accurate caption. Respond within <think></think><br><br>[User Prompt] Caption this video. Respond within <think></think>.|
|---|

###### Figure 11: RANKVIDEO prompt for stage 1

|[System Prompt] You are a helpful assistant specialized in video and text understanding. Given a text query and a video, your task is to determine if the video is relevant to the query. Respond with <answer>yes</answer> if the video is relevant, or <answer>no</answer> if it is not.<br><br>[User Prompt] Query: {query} Is the video relevant to the query? Respond with <answer>yes</answer> or <answer>no</answer>.|
|---|

###### Figure 12: RANKVIDEO prompt for stage 2

|[System Prompt] You are a helpful assistant specialized in video and text understanding. Given a text query and a video, your task is to determine if the video is relevant to the query. Respond with <answer>yes</answer> if the video is relevant, or <answer>no</answer> if it is not.<br><br>[User Prompt] Query: {query} Is the video relevant to the query? Respond with <answer>yes</answer> or <answer>no</answer>.|
|---|

###### Figure 13: Prompt for the QwenVL Models

|[System Prompt] You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query. Given a query and a passage list, you first thinks about the reasoning process in the mind and then provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.|
|---|

###### Figure 14: Prompt for the REASONRANK

