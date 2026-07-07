## VidVec: Unlocking Video MLLM Embeddings for Video–Text Retrieval

# arXiv:2602.08099v1[cs.CV]8Feb2026

Issar Tzachor1 Dvir Samuel1 Rami Ben-Ari1

#### Abstract

Recent studies have adapted generative Multimodal Large Language Models (MLLMs) into embedding extractors for vision tasks, typically through fine-tuning to produce universal representations. However, their performance on video remains inferior to Video Foundation Models (VFMs). In this paper, we focus on leveraging MLLMs for video–text embedding and retrieval. We first conduct a systematic layer-wise analysis, showing that intermediate (pre-trained) MLLM layers already encode substantial task-relevant information. Leveraging this insight, we demonstrate that combining intermediate-layer embeddings with a calibrated MLLM head yields strong zero-shot retrieval performance without any training. Building on these findings, we introduce a lightweight text-based alignment strategy which maps dense video captions to short summaries and enables task-related video–text embedding learning without visual supervision. Remarkably, without any fine-tuning beyond text, our method outperforms current methods, often by a substantial margin, achieving state-of-the-art results across common video retrieval benchmarks.

#### 1. Introduction

Multimodal Large Language Models (MLLMs) have recently emerged as a dominant paradigm in vision–language understanding, demonstrating strong performance across tasks such as captioning (Jia et al., 2024), visual question answering (Awadalla et al., 2023), and even visual mathematical reasoning (Chen et al., 2025). A common design represents visual input as sequences of visual tokens that are fed into a generative Large Language Model (LLM) which, after joint fine-tuning, enables open-world reasoning and instruction following over multimodal content. Extending this paradigm from images to videos has led to rapid progress in video-based MLLMs (Zhang et al., 2024b; Cheng et al.,

1OriginAI, Israel. Correspondence to: Issar Tzachor <iyttor@gmail.com>. Preprint. February 10, 2026.

2024; Zhang et al., 2025a; Tang et al., 2025; Yang et al., 2025). Compared to images, videos introduce richer content and temporal dynamics, substantially increasing the challenge of representation learning, reasoning, and retrieval.

As MLLMs become increasingly widespread, multimodal representation learning has attracted growing research interest. Early vision-language models (VLMs) such as CLIP (Radford et al., 2021), ALIGN (Jia et al., 2021), and BLIP (Li et al., 2022) established a powerful dual-encoder recipe for text-image representations by aligning image and text embeddings with contrastive learning on large-scale imagetext data. Follow-up studies suggest extensions of these models to video embedding and retrieval (Luo et al., 2021; Ma et al., 2022; Xu et al., 2021).

Recent large Video Foundation Models (VFMs) have pushed zero-shot and transfer performance by scaling video–text pretraining. InternVideo2 (Wang et al., 2024) has pretrained ∼100M video–text pairs to reach superior performance, while VideoPrism (Zhao et al., 2024) contrastively trained with ∼600M pairs. Recent studies suggest that simply scaling video–text pretraining does not uniformly improve performance (Feichtenhofer et al., 2022; Tong et al., 2022; Uselis et al., 2025) motivating alternative approaches that focus on representation quality and task-related alignment, rather than relying solely on ever-larger corpora.

Recently, an emerging line of work investigates whether MLLMs can serve as representation learners across visiontext modalities, motivated by the state-of-the-art performance of LLM-based embedding models on MTEB (Muennighoff et al., 2023). E5-V (Jiang et al., 2024) suggested refining MLLM’s language component using textual supervision to learn image-text aligned embeddings. Subsequent methods (Jiang et al., 2025; Lin et al., 2025) convert MLLMs into embedding models by incorporating visiontext paired data into contrastive training. To overcome the limited scale of curated embedding datasets, MegaPairs (Zhou et al., 2024) and UniIR (Wei et al., 2024) propose large-scale training datasets.

Although recent methods report impressive results on image retrieval (Zhang et al., 2025b; Thirukovalluru et al., 2025; Kong et al., 2025; Liu et al., 2025; Gu et al., 2026), video is often treated as an auxiliary modality rather than a primary focus, and performance on video retrieval remains

###### Calibrated Likelihood Scorer

###### Intermediate Layer Embeddings

MLLM

MLLM

Layer#1

Layer#n

Layer#1

Layer#n

Layer#i

... ...

###### ...

###### V

Video

T

### +

Video

Yes

Text

Text

ℙr.

Emb.

Does the video match the sentence? Answer with a single word - Yes or No.

Summarize in one word:

###### (a) Zero-Shot Retrieval (b) Zero-Shot Reranking

###### (c) In-Context Optimization

Dual Emb

Emb

Lightweight alignment Inference

Softmax Loss

Multimodal Large Language Model (MLLM)

Multimodal Large Language Model (MLLM)

The video depicts a scene in a dimly lit room with a focus on two individuals, a man and a woman, who are engaged in a discussion or

A man and a woman are working on a computer in a dark room.

presentation. The man is seated at a desk with a laptop displaying various diagrams and h hil h d b id hi

A man and a woman working on a computer

Dense Video Caption

Video

Short Summary

Video Caption

Figure 1. An overview of VidVec. (a) Zero-shot retrieval: extract video and text embeddings from an intermediate MLLM layer for initial ranking. (b) Zero-shot reranking: leverage the calibrated MLLM head for pairwise scoring to rerank top-K candidates. (c) In-context optimization: lightweight model alignment using only ∼60K text-only pairs for embedding extraction via a text-to-text mapping from dense video captions to short summaries, designed to mirror the video–text inference setup.

behind that of dedicated Video Foundation Models (VFMs) (Wang et al., 2024; Liu et al., 2022; Zhu et al., 2023; Lan et al., 2025b). The few efforts that explicitly target video have yet to achieve strong results on standard video–text retrieval benchmarks. CARE (Xu et al., 2024) emphasizes fine-grained captioning and retrieval but underperforms in conventional retrieval settings, while VLM2Vec-V2 (Meng et al., 2025) incorporates video–text pairs during training yet reports substantially lower performance, even compared to earlier MLLM-based embedders (Kong et al., 2025).

Overview of our approach. In this work, we study MLLMs as embedding extractors specifically for video–text retrieval. We show that off-the-shelf MLLMs encode substantial retrieval-relevant information in their hidden representations. A systematic layer-wise analysis reveals that selecting appropriate intermediate layers already yield strong zero-shot retrieval performance (Fig. 1a). Further gains are obtained by re-ranking with a calibrated MLLM head (Fig. 1b). Finally, we propose an efficient in-context optimization scheme that maps dense video captions to short summaries, enabling task-related embedding learning without visual inputs (Fig. 1c). Using only ∼60K text-only

in-context pairs, we outperform trained MLLM embedders and video foundation models trained on orders of magnitude more video–text data.

The key contributions of this work are:

- 1. We propose a methodology to assess and exploit hidden representations of Video MLLMs for video–text retrieval, and show that intermediate layers can be markedly more effective than the final layer.
- 2. We introduce an effective zero-shot scheme that leverages the MLLM head as a calibrated likelihood scorer, turning off-the-shelf Video MLLMs into competitive video–text retrievers.
- 3. We present an efficient in-context optimization strategy that maps dense video captions to short summaries, enabling task-related embedding learning without visual supervision.
- 4. Without any fine-tuning beyond text, our method achieves state-of-the-art performance on common video retrieval benchmarks.

#### 2. Related Work

Vision–language embeddings for retrieval A long line of research studies dual-encoder vision–language representation learning, where an image encoder and a text encoder are trained to align paired image–text data via contrastive objectives (Radford et al., 2021; Jia et al., 2021; Zhai et al., 2023). These methods showed a straightforward dual-encoder can produce strong cross-modal retrieval performance. While highly effective for image–text embedding and retrieval, dual-encoder paradigms are less natural for (i) interleaved multimodal inputs and instructionconditioned retrieval, and (ii) richer video understanding scenarios where semantics can depend on temporal dynamics. These limitations motivate embedding approaches that leverage instruction-following generative backbones and deeper multimodal fusion, rather than relying solely on independent encoders optimized for caption-level alignment.

Video–text retrieval and video representation learning. Video retrieval has traditionally been addressed by video–language models that explicitly encode spatiotemporal structure and are trained on large-scale video–text (and sometimes audio/speech) corpora. Alongside joint embedding models, several methods study how to extend image–text alignment mechanisms to the temporal setting (Luo et al., 2021; Xu et al., 2021; Ma et al., 2022). Recent Video Foundation Models (VFMs) substantially scale pretraining data and objectives often using huge datasets for training, for the aligned embedding task. For example, InternVideo (Wang et al., 2022) reports pretraining on roughly 12M video clips spanning multiple domains, HowTo100M trains on nearly 136M short clips (Miech et al., 2019). InternVideo2 (Wang et al., 2024) further scales the data regime to 50M video–text pairs and 50M video–audio–speech–text pairs. Others e.g. UMT (Liu et al., 2022) and LanguageBind (Zhu et al., 2023) and VideoPrism (Zhao et al., 2024) present state-of-the-art results while building on millions or hundreds of millions of paired samples (video–text), often in addition to millions of image-text pairs.

These approaches typically rely on video-specialized encoders, explicit temporal modeling, and large-scale video–text supervision. In contrast, our work studies multimodal LLMs as embedders for video understanding (demonstrated on video retrieval), focusing on (i) where video semantics reside inside the MLLM (layer-wise readout analysis), and (ii) how far we can push video retrieval using text-only supervision.

Multimodal LLMs as embedders and token-based embedding learning. UniIR (Wei et al., 2024) introduced instruction-guided multimodal retrieval with image-text paired post-training while MagicLens (Zhang et al., 2024a) explored self-supervised instruction-conditioned image retrieval using web-mined image pairs and synthesized instruc-

VideoLLaMA3 Intermediate Layer (Jan '25)

42.1

VLM2Vec-V2.0 (April '25)

41.0

UniME-V2 (Oct '25)

40.6

Figure 2. MSR-VTT Text-to-Video Retrieval Performance (Recall@1): MLLM Embedders vs. Off-the-Shelf Video MLLM

tions. Both of these methods are based on VLM backbones e.g. BLIP (Li et al., 2022) or CLIP.

In parallel, VLM2Vec (Jiang et al., 2025) proposed an instruction-guided contrastive framework that converts MLLMs into embedding models by LoRA finetuning on large scale MMEB dataset (Jiang et al., 2025). VLM2VecV2 (Meng et al., 2025) extends this direction by explicitly incorporating videos and visual documents into training.

A rapidly growing line of work re-purposes multimodal LLMs (MLLMs/LMMs) as embedding models by extracting fixed-dimensional representations from hidden states under carefully designed prompts, often centered around a dedicated embedding token. E5-V (Jiang et al., 2024) showed that prompt-based “semantic compression” (e.g., asking the model to produce a one-word summary) can create aligned image-text (multimodal) embeddings, even without fine-tuning; critically, E5-V further demonstrated that single-modality training only on text pairs (NLI (Bowman et al., 2015)) can generate reasonably aligned multimodal embeddings with low training cost avoiding the need for image–text training data.

LamRA (Liu et al., 2025) follows this paradigm built on large multimodal models and showed significantly improved results with additional image-text paired post training. Methodologically, LamRA uses an “Explicit One-word Limitation” prompt of the form “Summarize ... in one word: <emb>” and takes the last hidden state immediately preceding the <emb> token as the embedding. For retrieval, LamRA adopts a two-stage strategy: (1) language-only pretraining to teach the model to output retrieval-friendly embeddings under summarization prompts, and (2) multimodal instruction tuning over diverse retrieval tasks; for reranking, LamRA trains an additional module with pointwise and listwise objectives.

CARE (Xu et al., 2024) pretrains a video MLLM embedder by generating detailed video captions, followed by training on general text pairs. In contrast, UNITE (Kong et al., 2025) trains on modality-specific pairs (text–text, text–image, and text–video), suggesting that generation-based pretraining does not always improve performance.

Our work is aligned with the broader trend of MLLM-asembedder and token-based embedding learning, as E5-V and LamRA. However, we depart from prior universal embedding approaches in several important and largely underexplored aspects. (i) we focus on video–text retrieval as the central task, rather than treating it as one of many downstream applications; (ii) we perform a systematic layer-wise analysis of video representations, demonstrating that the choice of readout layer has a substantial impact on retrieval performance and that a video-centric MLLM can already produce well-aligned embeddings for retrieval without any post-training. (iii) we investigate text-only supervision for video embedding learning, showing that it can be highly effective for multimodal alignment and can achieve stateof-the-art performance in video retrieval without video–text contrastive training. (iv) we introduce in-context text summarization as an efficient training signal, using a mapping from dense video captions to concise summaries as an auxiliary objective for learning text–video aligned embeddings.

Concretely, while many token-optimized embedding methods rely on explicit text summarization with additional supervision, we demonstrate that in-context summarization offers an efficient alternative. By training on densecaption–to–summary mappings, we obtain strong video–text embeddings without visual supervision.

Training efficiency, negatives, and advanced embedding objectives. Beyond token prompting and readout design, some recent work highlights that training efficiency and negative sampling that can impact embedding performance. Such works include B3 (“Breaking the Batch Barrier”) (Thirukovalluru et al., 2025). UniME-V2 (Gu et al., 2026) uses an “MLLM-as-a-judge” to assign soft semantic matching scores for query–candidate pairs, to mine higherquality hard negatives. UME-R1 (Lan et al., 2025a) explores reasoning-driven generative embeddings, using a two-stage training recipe (SFT + RL).

Our work is complementary to these advances. Rather than proposing a new negative mining mechanism or a generative embedding objective, we study how to utilize MLLMs for video–text retrieval and how far carefully designed text-only supervision can push video–text retrieval performance.

#### 3. Method 3.1. Problem Formulation

We formulate video-text retrieval using a generative Multimodal Large Language Model (MLLM). Given a query q (either text or video) and a candidate pool Ω = c1,...,cN of size N, where ci belongs to the other modality of q, we extract a d-dimensional embedding for the query and for each candidate, eq = fθ(q) ∈ Rd and ei = fθ(ci) ∈ Rd, where fθ(·) denotes the embedding extractor induced by the

MLLM parameters θ. We score each candidate by cosine similarity and rank Ω accordingly. For second stage reranking, we select the top-K candidates and apply a secondstage re-ranking over this pool to obtain the final ranked list, R2 = Φrerank(q,RK), where R2 is the reordered set of candidates produced by the re-ranking stage. We next present the embedding extraction approach for eq,ei.

###### 3.2. Embedding Extraction

Multimodal large language models (MLLMs) typically comprise three components: a vision encoder, a vision projector, and a language model. The vision encoder and projector convert the input video into a sequence of visual tokens compatible with the language model, which are then processed jointly with the text tokens.

To extract embeddings from MLLMs, we follow prior work (Jiang et al., 2024; Liu et al., 2025) and adopt an Explicit One-word Limitation (EOL) prompting scheme, in which the prompt ends with the instruction to respond ‘‘in one word’’ followed by a dedicated <emb> token. We take as the embedding the hidden state immediately preceding <emb> and call it <emb-1>.

###### 3.3. MLLM Head as a Calibrated Likelihood Scorer

For the second-stage of re-ranking, we reuse the language model head as a calibrated likelihood scorer. Given the reduced candidate set RK, each query–candidate pair (q,ci) is evaluated independently by prompting the model with a binary relevance question. Concretely, we extend the EOL prompting scheme with the instruction ‘‘Respond in a single word - Yes or No.’’. We then define the re-ranking score for each candidate ci as the likelihood of the affirmative tokens Yes/yes, Srank(q,ci) = Pθ(Yes | q,ci), where Srank is the relevance similarity score used for final ranking. This procedure requires K forward passes and induces the re-ranked list, where candidates in RK are ordered by decreasing Srank. Fig. 1b describes this setting.

###### 3.4. Token Optimization Strategy

Unlike prior work that focuses on NLI-style text-to-text mappings for token optimization, we explore multiple mapping strategies and show that task-oriented mappings closely tied to the visual nature of video has a significant impact for performance.

Architecturally, we fine-tune a lightweight LoRA (Mangrulkar et al., 2022) using text-to-text mappings grounded in video descriptions. To this end, we leverage the publicly available VideoUFO dataset (Wang & Yang, 2025), which contains over 1.09M video clips, each paired with both brief and detailed textual descriptions. From this corpus, we sample 60K data entries and fine-tune the model

to map long, detailed descriptions to short summaries using an alignment objective, without accessing visual data. Our in-context optimization strategy implicitly captures visual content and temporal dynamics, yielding embeddings well suited for video encoding and retrieval. As illustrated in Fig. 1c, our approach goes beyond simply using videorelated text pairs: the detailed descriptions are aligned with the full video content (yellow), while the short summaries act as compact textual anchors aligned with the query text (blue). This formulation encourages the model to learn a visually grounded “summarization” process that effectively mirrors video encoding.

###### 3.5. Training Objective

Let etn = MLLM(tn) and evn = MLLM(vn) denote the text and video embeddings, respectively, extracted from the MLLM, as described in Sec. 3.2. For in-context optimization, we train these embeddings using the Dual-Softmax Loss (DSL) (Cheng et al., 2021), a standard objective for learning video-text representations in prior VFMs (Bain et al., 2021b; Luo et al., 2021; Wang et al., 2022). Specifically, DSL applies softmax normalization along both axes of the similarity matrix, yielding conditional match distributions for text-to-video and video-to-text retrieval. These two distributions are then multiplicatively combined to produce match weights that emphasize pairs with high mutual confidence under both retrieval directions. The training objective is applied symmetrically, encouraging consistent alignment between modalities.

Notably, our in-context optimization is performed solely in the text space, mapping dense video captions, that act as textual proxies for the underlying videos, to summaries. Nevertheless, using the DSL objective encourages the optimized text to act as an effective proxy for the underlying video content.

In summary, our method operates in three complementary settings that can be flexibly combined. (1) In the zeroshot configuration, we extract the <emb-1> token from an intermediate layer as described in Sec. 3.2, producing the initial VidVec-ZS representation. (2) For reranking, we apply the calibrated model head to the top-K retrieved results, yielding a noticeable performance boost. (3) To further enhance performance, we leverage in-context text-totext mapping for token optimization, producing the VidVecO model. This optimized representation can optionally be combined with the calibrated head reranker for additional gains. Fig. 1 illustrates these configurations. Notably, none of these settings require training on visual data: in-context optimization using video descriptions at different levels of granularity, effectively compensates for the absence of direct visual supervision.

For retrieval we follow the standard procedure of query text

Figure 3. Layer-wise Recall@1 on MSR-VTT for zero-shot video embedding extraction. We evaluate embeddings obtained from different layers across several MLLM backbones. While deeper layers generally yield stronger retrieval performance, the optimal results are not achieved at the final layer. Among all evaluated models, VideoLLaMA3-7B attains the best overall performance.

encoding and gallery video encoding followed by ranking by embedding similarities.

#### 4. Zero-shot Layer-wise Analysis

Applying our embedding extraction strategy to an off-theshelf MLLM, by taking the <emb-1> hidden state (token) as the representation, uncovers a retrieval signal, but results in comparably low overall performance on video–text retrieval (i.e. MSR-VTT R@1 is 14.3%, see Tab. 9). Inspired by (Skean et al., 2025) that reveals the strength of mid-depth embedding in LLMs we examine the strength of the text-video alignment signal within MLLMs measured by video–text retrieval performance. To this end, we adopt the layer-wise evaluation protocol of (Tzachor et al., 2025). Concretely, we extract representations from selected intermediate layers using the embedding extraction procedure described in Sec. 3.2. We emphasize that no training is performed in this setting.

We conduct the analysis across multiple Qwen-VL generations and select strong video MLLMs of comparable size based on their performance on the Video-MME shortclip benchmark (Fu et al., 2025). Specifically, we evaluate Qwen2-VL, Qwen2.5, Qwen2.5-VL-7B, Qwen3-VL-8B and the video MLLMs VideoLLaMA3-7B and Keye-VL-8B. Fig. 3 presents layer-wise Recall@1 results on MSR-VTT in the zero-shot setting. While early layers contain little to no retrieval-relevant signal, several mid- to late-stage intermediate layers achieve markedly stronger performance, even without explicit alignment or retrieval-specific fine-tuning. Notably, the largest gains are achieved for VideoLLaMA37B, which exhibits the strongest zero-shot performance from its intermediate layers.

Zero-Shot Prompting For the layer-wise analysis we used general textual prompts: <caption> ”Summarize

above sentence in one word: ”, and video prompt: <video> ”Summarize above video in one word: ”, both end by system <emb> token. We further find that for our off-the-shelf MLLMs video processing, task-aware prompt engineering plays an important role: adding a prefix prompt ‘‘Recover the main subject or subjects, appearance and setting, and main activity in the video’’ leads to a noticeable performance improvement. We therefore adopt this prefix prompt in all subsequent zero-shot experiments.

#### 5. Evaluation

In this section, we compare our approach against several state-of-the-art MLLM embedders and VFMs. For MLLMs, we evaluate seven methods, including UNITE (Kong et al., 2025), MMRet-v1.5 (BGE-VL) (Zhou et al., 2024), VLM2Vec-V2 (Meng et al., 2025), and the recent UniME-V2 (Gu et al., 2026). For VFMs, we compare to seven methods, including InternVideo2-6B (Wang et al., 2024), VideoPrism-g (Zhao et al., 2024), and PE-Core-G (Bolya et al., 2025).

We conduct an evaluation with four common video–text retrieval benchmarks - MSR-VTT (Xu et al., 2016), MSVD (Chen & Dolan, 2011), VATEX (Wang et al., 2019), and DiDeMo (Anne Hendricks et al., 2017). A summary of the datasets is provided in Tab. 6. Additional information on the benchmarks is provided in the Appendix. Across all evaluations, models are not trained on any of the benchmark training sets.

VLM2Vec-V2 (Meng et al., 2025) introduced the MMEBV2 benchmark, which spans multiple video tasks. Here we focus on video–text retrieval and use the standard datasets and splits adopted by recent VFMs (Zhao et al., 2024; Wang

- et al., 2024; Bolya et al., 2025) to enable direct comparison with current state-of-the-art methods. We still compare against VLM2Vec-V2, but report results on the established retrieval splits. While MMEB-V2 includes some existing

retrieval datasets, it does not evaluate the V2T setting and modifies several test splits, making direct comparisons to current state-of-the-art methods difficult. For completeness, in the Appendix we compare against the MMEB-V2 reported results on the overlapping video-retrieval datasets, where VidVec achieves superior performance.

We present three variants of our method: (1) a zero-shot twostage baseline (VidVec-ZS), which involves no training, (2) a one-stage in-context optimization approach (VidVec-O), and (3) two-stage (VidVec) with optional reranking.

###### 5.1. Implementation Details

For the zero-shot variant VidVec-ZS, we use embeddings extracted from layer 24, whereas for the optimized variant VidVec-O we use the final-layer embeddings. In-context optimization uses a batch size of 288 pairs distributed across 4×B200 GPUs and completes in under 30 minutes. For reranking, in zero-shot we use K = 100, and for the optimized model we use K = 10. We adopt VideoLLaMA3 (Zhang et al., 2025a) as our video MLLM since: (i) it achieves the best performance in our layer-wise analysis (Fig. 3), suggesting that its intermediate representations are particularly effective for video–text retrieval; and (ii) it reports strong results on the Video-MME benchmark (Fu et al., 2025) for short-video understanding, among 7B-scale models. Videos are sampled at 2 FPS (matching the Qwen2VL default) and capped at 180 frames (the VideoLLaMA3 default limit). Following prior VFM work, we also apply dual-softmax at inference to boost performance.(Cheng et al., 2021; Wang et al., 2022; 2024; Bolya et al., 2025). To ensure a fair comparison, we re-run competing MLLM embedders, where applicable, using the same FPS and a dual-softmax calibrated scoring. This often improves performance relative to the numbers reported in the original papers. For VFMs, we report the published results. Additional implementation details are provided in the Appendix.

Table 1. Text-to-Video (T2V) Retrieval - our two-stage VidVec-ZS (zero-shot) vs. state-of-the-art MLLM embedders (7B size)

MSR-VTT VATEX DiDeMo Model R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

LamRA 48.9 71.6 79.5 61.4 90.6 95.2 46.3 71.8 80.5 VLM2Vec 41.6 64.5 73.8 53.6 84.4 91.0 37.7 60.9 69.9 MMRet-v1.5 45.2 67.5 75.9 57.7 87.5 93.5 45.1 70.7 78.8 B3 46.7 67.6 75.5 57.3 86.5 92.3 38.7 62.6 70.7 VLM2Vec-V2 41.0 63.6 71.9 52.9 83.2 90.4 41.7 66.5 75.1 UNITE 46.5 69.4 78.0 45.7 74.0 81.0 43.5 69.6 77.3 UniME-V2 40.6 62.5 70.7 52.0 80.8 86.9 36.5 59.2 66.7

###### VidVec-ZS 52.1 74.3 82.9 69.1 93.3 96.7 55.7 80.7 85.4

- Table 2. Text-to-Video (T2V) retrieval - VidVec-O (Optimized Embedding) vs. SoTA MLLM embedders (7B size) on four benchmarks. VidVec-O is using only 60K text-only in-context pairs, whereas prior methods are trained on ×10-×100 larger scale of vision-text data.

MSR-VTT MSVD VATEX DiDeMo Model R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

LamRA 48.9 71.6 79.5 55.7 81.7 88.0 61.4 90.6 95.2 46.3 71.8 80.5 VLM2Vec 41.6 64.5 73.8 52.1 78.4 85.8 53.6 84.4 91.0 37.7 60.9 69.9 MMRet-v1.5 45.2 67.5 75.9 49.5 76.6 83.5 57.7 87.5 93.5 45.1 70.7 78.8 B3 46.7 67.6 75.5 53.8 80.0 86.7 57.3 86.5 92.3 38.7 62.6 70.7 VLM2Vec-V2 41.0 63.6 71.9 48.2 75.4 83.1 52.9 83.2 90.4 41.7 66.5 75.1 UNITE 46.5 69.4 78.0 50.4 78.2 86.4 45.7 74.0 81.0 43.5 69.6 77.3 UniME-V2 40.6 62.5 70.7 52.1 77.5 84.4 52.0 80.8 86.9 36.5 59.2 66.7

VidVec-O 52.5 76.3 83.8 60.8 84.9 90.1 68.2 93.6 96.8 53.7 79.4 85.0

- Table 3. Video-to-Text (V2T) retrieval - VidVec-O (Optimized Embedding) vs. SoTA MLLM embedders (7B size) on four benchmarks.

MSR-VTT MSVD VATEX DiDeMo Model R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

LamRA 50.9 72.8 80.0 81.2 92.5 95.5 82.5 95.4 97.7 47.1 72.8 81.5 VLM2Vec 44.9 68.6 76.1 81.3 93.1 95.7 80.2 95.5 98.7 41.3 65.4 72.5 MMRet-v1.5 47.9 70.9 77.7 77.2 90.9 93.4 81.1 95.7 97.7 46.1 73.1 80.7 B3 48.7 70.5 77.9 83.1 95.2 97.2 83.1 96.4 98.5 43.3 65.3 72.5 VLM2Vec-V2 42.9 64.8 72.2 82.1 95.1 97.5 79.9 95.9 98.2 44.2 68.0 76.1 UNITE 45.2 70.3 79.3 76.1 91.3 94.6 78.9 95.3 98.1 40.3 68.7 78.1 UniME-V2 46.3 65.7 72.4 81.8 93.4 95.5 81.5 96.1 98.4 38.7 61.0 69.0

VidVec-O 54.9 77.5 84.1 85.7 95.5 97.8 89.6 98.5 99.3 56.5 79.7 86.0

###### 5.2. Zero-Shot Performance

In Tab. 1, we evaluate our zero-shot two-stage method, VidVec-ZS, against state-of-the-art MLLM embedders trained to produce vision–language embeddings. We report results on the text-to-video (T2V) retrieval task for MSR-VTT, VATEX and DiDeMo. While prior methods rely on contrastive training on vision–text pairs, VidVec-ZS leverages an off-the-shelf MLLM, using intermediate-layer representations for embedding followed by a re-ranking. Despite conducting no further training, VidVec-ZS outperforms prior methods on all three benchmarks, achieving notable gains in Recall@1 of +3.1%, +7.7% and +9.4%, on MSR-VTT, VATEX and DiDeMo, respectively.

This indicates that off-the-shelf generative MLLMs such as VideoLLaMA3 already contain well-aligned video–text embeddings internally, a property newly demonstrated here and lacking from previous work (Jiang et al., 2024; Liu et al., 2025; Gu et al., 2026).

###### 5.3. Comparison with State-of-the-Art

In this section, we evaluate our lightweight in-context optimization approach for extracting output embeddings. We assess the resulting embeddings on four standard video–text retrieval datasets, comparing with state-of-the-art MLLM embedders and VFMs under the same evaluation protocol.

We compare the embeddings extracted by our optimized

model, VidVec-O, against recent MLLM embedders. In Tab. 2 and Tab. 3, we report Text-to-Video (T2V) and Videoto-Text (V2T) results respectively. Across both retrieval directions, VidVec-O consistently surpasses existing MLLM embedders, often by a substantial margin. The results highlight the effectiveness of our approach, yielding notable Recall@1 improvements, including V2T +6.5% on VATEX and +9.4% on DiDeMo. This demonstrates that incontext optimization can effectively elicit the model’s internal knowledge using relatively small amounts of textual data, yielding aligned video–text embeddings than prior methods trained on millions of vision–text pairs.

While text-based optimization leads to performance gains, the calibrated head reranker in VidVec-ZS provides even larger improvements. Moreover, combining the reranker with VidVec-O for Text-to-Video retrieval results in stateof-the-art performance.

Tab. 4 reports results obtained by adding a zero-training re-ranker as a second stage on top of VidVec-O. We compare Recall@1 performance against state-of-the-art Video Foundation Models (VFMs).

Regarding training data, Perception Encoder (PE-Core) (Bolya et al., 2025) and VideoPrism (Zhao et al., 2024) had used approximately 5.4B image-text (and 22M video–text pairs) and 1B image–text and 600M video–text pairs, including 36.1M manually labeled samples for training. InternVideo2-6B (Wang et al., 2024) which previously

- Table 4. Video-Text retrieval performance (Recall@1): VidVec vs. state-of-the-art Video Foundation Models (VFMs). Dashed entries indicate not reported results in the corresponding paper.

MSR-VTT MSVD VATEX DiDeMo T2V V2T T2V V2T T2V V2T T2V V2T

V-T Pairs

Method

Clip4Clip n/a 32.0 - 38.5 - - - - ViCLIP n/a 42.4 41.3 49.1 75.1 - - 18.4 27.9 InternVideo-L 12M 40.7 39.6 43.4 67.6 49.5 69.5 31.5 33.5 UMT-L 10M 42.6 38.6 49.9 75.4 - - 48.6 49.9 LanguageBind 10M 44.8 40.9 53.9 72.0 - - 39.9 VideoPrism-g 600M 52.7 51.7 - - 62.5 77.1 - PE-Core-G 22M 51.2 49.9 59.7 85.4 - - - InternVideo2-6B 100M 55.9 53.7 59.3 83.1 71.5 85.3 57.9 57.1 VidVec Text-Only 56.2 54.9 60.9 85.7 70.0 89.6 61.8 56.5

achieved state-of-the-art performance on most benchmarks, is trained on about 300M image–text pairs and 100M video–text pairs, with additional gains achieved from a second-stage re-ranking. VidVec delivers state-of-the-art performance on the majority of evaluated benchmarks in both T2V and V2T retrieval. +1.2% improvement on MSRVTT V2T, and +1.2% on MSVD T2V. On VATEX, VidVec underperforms by -1.5% on T2V but improves by +4.2% on V2T, and on DiDeMo, while underperforming by -0.6% on V2T, it improves by +3.9% on T2V.

Although VidVec uses a video-pretrained MLLM backbone, it is not explicitly trained for modality alignment or embedding extraction. This highlights the effectiveness of our minimal post-training in-context optimization and shows that strong video–text representations can be efficiently unlocked from a powerful MLLM.

#### 6. Ablation Study

We conduct an ablation study analyzing different components and design choices of our approach, including alternative text-to-text mapping strategies and experiments with a Qwen-2-VL backbone. Table 5 evaluates the effect of different textual optimization strategies on MSR-VTT Recall@1. Using short video caption pairs improves over prior NLI-based text, while our in-context optimization achieves the best performance. Unlike previous approaches, it relies solely on video-related text and leverages detailed video captions and aligned summaries, resulting in improved performance 1. For further results and details we refer the reader to the Appendix. Overall, we show that in-context learning consistently outperforms NLI-based mappings, and that Qwen-2-VL achieves performance only slightly below VideoLLaMA3 (by 0.2%).

1The ablation experiments were conducted without dualsoftmax to isolate the effect of the examined component.

Table 5. Effect of Textual Data Choice for Optimization on MSRVTT (T2V Recall@1).

Context R@1 Zero-shot 14.3 NLI 45.0 Video Captions 46.7 In-Context 47.7

#### 7. Summary

In this work, we study off-the-shelf video MLLMs and show that they are remarkably effective for video–text retrieval. We introduce a zero-shot MLLM-based embedder that exploits intermediate representations for embedding and the model head for calibrated scoring, yielding strong retrieval performance without training. We then propose an efficient in-context optimization scheme that relies solely on text-to-text mapping, requiring no visual supervision, to further improve multimodal alignment. The resulting embeddings tested on video retrieval, significantly outperform recent trained MLLM embedders, and our complete approach achieves state-of-the-art results across multiple benchmarks. Limitations are discussed in the Appendix.

More broadly, our results highlight the largely untapped potential of large multimodal models for training-free and data-efficient adaptation to embedding-based tasks.

#### Societal Impact

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

#### References

Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., and Russell, B. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pp. 5803–5812, 2017.

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international

- conference on computer vision, pp. 1728–1738, 2021a.

Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international

- conference on computer vision, pp. 1728–1738, 2021b.

Bolya, D., Huang, P.-Y., Sun, P., Cho, J. H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025.

Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. A large annotated corpus for learning natural language inference. In M`arquez, L., Callison-Burch, C., and Su, J. (eds.), Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 632–642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https://aclanthology.org/D15-1075.

Caba Heilbron, F., Escorcia, V., Ghanem, B., and Carlos Niebles, J. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970, 2015.

Chen, D. and Dolan, W. B. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.

Chen, F., Yuan, H., Xu, Y., Feng, T., Cen, J., Liu, P., Huang, Z., and Yang, Y. Mathflow: Enhancing the perceptual flow of mllms for visual mathematical problems. arXiv preprint arXiv:2503.16549, 2025.

Chen, S., Zhao, Y., Jin, Q., and Wu, Q. Fine-grained videotext retrieval with hierarchical graph reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10638–10647, 2020.

Cheng, X., Lin, H., Wu, X., Yang, F., and Shen, D. Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss. arXiv preprint arXiv:2109.04290, 2021.

Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., and Bing, L. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. URL https:// arxiv.org/abs/2406.07476.

Feichtenhofer, C., Li, Y., He, K., et al. Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems, 35:35946–35958, 2022.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025.

Gabeur, V., Sun, C., Alahari, K., and Schmid, C. Multimodal Transformer for Video Retrieval. In European Conference on Computer Vision (ECCV), 2020.

Gu, T., Yang, K., Zhang, K., An, X., Feng, Z., Zhang, Y., Cai, W., Deng, J., and Bing, L. UniME-V2: MLLMas-a-judge for universal multimodal embedding learning. AAAI, 2026.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.

Jia, H., Xu, Y., Zhu, L., Chen, G., Wang, Y., and Yang, Y. Mos2: Mixture of scale and shift experts for textonly video captioning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 8498–8507, 2024.

Jiang, T., Song, M., Zhang, Z., Huang, H., Deng, W., Sun, F., Zhang, Q., Wang, D., and Zhuang, F. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580, 2024.

Jiang, Z., Meng, R., Yang, X., Yavuz, S., Zhou, Y., and Chen, W. VLM2Vec: Training vision-language models for massive multimodal embedding tasks. ICLR, 2025.

Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi`ere,

M., et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Kong, F., Zhang, J., Liu, Y., Zhang, H., Feng, S., Yang, X., Wang, D., Tian, Y., Zhang, F., Zhou, G., et al. Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650, 2025.

Krishna, R., Hata, K., Ren, F., Fei-Fei, L., and Niebles, J. C. Dense-captioning events in videos. In International Conference on Computer Vision (ICCV), 2017.

Lan, Z., Niu, L., Meng, F., Zhou, J., and Su, J. Umer1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405, 2025a.

Lan, Z., Niu, L., Meng, F., Zhou, J., and Su, J. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812, 2025b.

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Lin, S.-C., Lee, C., Shoeybi, M., Lin, J., Catanzaro, B., and Ping, W. MM-Embed: Universal multimodal retrieval with multimodal llms. ICLR, 2025.

Liu, Y., Li, S., Wu, Y., Chen, C.-W., Shan, Y., and Qie, X. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3042–3051, 2022.

Liu, Y., Zhang, Y., Cai, J., Jiang, X., Hu, Y., Yao, J., Wang, Y., and Xie, W. LAMRA: Large multimodal model as your advanced retrieval assistant. In CVPR, pp. 4015– 4025, 2025.

Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., and Li, T. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860,

- 2021.

Ma, Y., Xu, G., Sun, X., Yan, M., Zhang, J., and Ji, R. Xclip: End-to-end multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM international conference on multimedia, pp. 638–647,

- 2022.

Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., and Bossan, B. Peft: State-of-the-art parameter-efficient fine-tuning methods. 2022.

Meng, R., Jiang, Z., Liu, Y., Su, M., Yang, X., Fu, Y., Qin, C., Chen, Z., Xu, R., Xiong, C., et al. VLM2Vec-V2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025.

Miech, A., Zhukov, D., Alayrac, J.-B., Tapaswi, M., Laptev, I., and Sivic, J. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2630–2640, 2019.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2014– 2037, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., and Shwartz-Ziv, R. Layer by layer: Uncovering hidden representations in language models. ICML, 2025.

Tang, C., Li, Y., Yang, Y., Zhuang, J., Sun, G., Li, W., Ma, Z., and Zhang, C. video-salmonn 2: Captioningenhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025.

Thirukovalluru, R., Meng, R., Liu, Y., Su, M., Nie, P., Yavuz, S., Zhou, Y., Chen, W., Dhingra, B., et al. Breaking the batch barrier (b3) of contrastive learning via smart batch mining. arXiv preprint arXiv:2505.11293, 2025.

Tong, Z., Song, Y., Wang, J., and Wang, L. Videomae: Masked autoencoders are data-efficient learners for selfsupervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022.

Tzachor, I., Lerner, B., Levy, M., Green, M., Shalev, T. B., Habib, G., Samuel, D., Zailer, N. K., Shimshi, O., Darshan, N., et al. EffoVPR: Effective foundation model utilization for visual place recognition. ICLR, 2025.

Uselis, A., Dittadi, A., and Oh, S. J. Does data scaling lead to visual compositional generalization? ICML, 2025.

- Wang, W. and Yang, Y. VideoUFO: A million-scale user-focused dataset for text-to-video generation. arXiv preprint arXiv:2503.01739, 2025.
- Wang, X., Wu, J., Chen, J., Li, L., Wang, Y.-F., and Wang, W. Y. Vatex: A large-scale, high-quality multilingual

dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4581–4591, 2019.

Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu, Y., Wang, Z., et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.

Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pp. 396– 416. Springer, 2024.

Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Ritter, A., and Chen, W. Uniir: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pp. 387–404. Springer, 2024.

Xu, H., Ghosh, G., Huang, P.-Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., and Feichtenhofer, C. Videoclip: Contrastive pre-training for zero-shot videotext understanding. EMNLP, 2021.

Xu, J., Mei, T., Yao, T., and Rui, Y. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Xu, Y., Li, X., Yang, Y., Meng, D., Huang, R., and Wang, L. Carebench: A fine-grained benchmark for video captioning and retrieval. arXiv preprint arXiv:2501.00513, 2024.

Yang, B., Wen, B., Ding, B., Liu, C., Chu, C., Song, C., Rao, C., Yi, C., Li, D., Zang, D., et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025a.

Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., and Chang, M.-W. Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv preprint arXiv:2403.19651, 2024a.

Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., and Zhang, M. GME: improving

universal multimodal retrieval by multimodal llms. CVPR, 2025b.

Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024b.

Zhao, L., Gundavarapu, N. B., Yuan, L., Zhou, H., Yan, S., Sun, J. J., Friedman, L., Qian, R., Weyand, T., Zhao, Y., et al. Videoprism: A foundational visual encoder for video understanding. ICML, 2024.

Zhou, J. et al. Megapairs: Massive data synthesis for universal multimodal retrieval, 2024. URL https: //arxiv.org/abs/2412.14475.

Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., Wang, H., Pang, Y., Jiang, W., Zhang, J., Li, Z., et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.

#### Appendix

This Appendix includes details on the following topics 2:

- A. Details on our benchmarks
- B. Evaluations on more datasets
- C. More Ablation Studies
- D. More implementation details
- E. Limitations

#### A. Benchmark Datasets

In this section we provide detailed information on the evaluated benchmarks - MSR-VTT, MSVD, VATEX, DiDeMo and ActivityNet. A summary of the datasets is provided in Tab. 6.

Table 6. Benchmark datasets statistics, with standard test subsets.

Dataset #Videos #Captions MSRVTT 1,000 1,000 MSVD 670 27,763 VATEX 1,500 15,000 DiDeMo 1,004 1,004 ActivityNet 4,917 4,917

###### B.1. MMEB-v2 Comparison

Recently, (Meng et al., 2025) suggested a new benchmark, MMEB-v2, for video tasks, which includes few text-tovideo retrieval datasets. In Tab.7 we compare our method with existing MLLM embedders, using the benchmark’s publicly available results. We report results on MSRVTT and DiDeMo, whose text and video splits follow the standard evaluation and were not modified in MMEB-v2. VidVec raw scores outperforms other methods also here. However, note that the LamRA paper reports substantially higher results (including for VLM2Vec), suggesting that the MMEB-v2 evaluation protocol or implementation may be non-optimal.

###### B.2. ActivityNet Evaluation

We conduct an additional evaluation on ActivityNet (Caba Heilbron et al., 2015). Table 8 summarizes the results on this dataset. Since VideoLLaMA3 includes ActivityNet data during its generative training, we exclude these results from the main paper; nevertheless, we report them here for completeness and to support future studies. Notably, VidVec outperforms even methods that were directly finetuned on ActivityNet’s training set (+5.1% gain relatively to a fine-tuned InternVideo2-6B)

MSR-VTT (Xu et al., 2016) contains 10,000 videos (10–32s each) with 200,000 associated captions. We follow the standard 1k-A test split (Bain et al., 2021a; Luo et al., 2021), which consists of 1,000 video–text pairs.

MSVD (Chen & Dolan, 2011) contains 1,970 videos (1–62s). The train/validation/test splits include 1,200/100/670 videos, respectively. Each video with approximately 40 English captions. In the V2T setting, we treat any caption for a given video as a positive.

VATEX (Wang et al., 2019) contains 25,991 training videos, 3,000 validation videos, and 6,000 test videos, with 10 paired captions per video in both English and Chinese. Following (Chen et al., 2020), we evaluate on 1,500 videos from the validation set using English captions only.

ActivityNet (Caba Heilbron et al., 2015; Krishna et al., 2017) contains 20,000 videos. Following (Gabeur et al., 2020; Luo et al., 2021), we concatenate all descriptions for each video into a single paragraph and evaluate video–paragraph retrieval on the ‘val1’ split.

##### B. More Evaluations In this section we report additional results for our VidVec.

2The ablation experiments through the Appendix were conducted without DSL to isolate the effect of the examined component

#### C. More Ablation Studies

###### C.1. Details on Impact of In-Context Optimization

We hereby provide more details on our ablation study presented in Tab. 5 in the main paper. We evaluated the impact of different textual optimization approaches on Recall@1 performance on MSR-VTT. Optimizing by brief video description pairs (Video Captions) improves over the NLIbased textual data used in prior work, while our in-context optimization achieves the best performance. An example of different textual data is shown in Fig. 4, while our in-context approach is described in Fig. 1. Our in-context approach not only leverages video-related textual pairs (not the videos), but also uses detailed video captions aligned with video content at inference time, together with short summaries that relate directly to the input text.

Table 7. Comparison to MMEB-v2 results on Text-to-Video retrieval (Recall@K).

MSR-VTT DiDeMo Model R@1 R@5 R@10 R@1 R@5 R@10

LamRA 25.0 47.4 57.6 22.8 44.7 56.9 VLM2Vec 34.5 58.5 67.9 29.3 53.5 62.2 VLM2Vec-v2 28.3 50.4 59.4 30.4 53.9 63.1 UniME-v2 27.6 44.9 52.9 31.5 53.0 61.9

###### VidVec 48.4 71.4 81.1 43.1 71.0 79.2

- Table 8. ActivityNet retrieval (R@K): Text-to-Video (left) and Video-to-Text (right). Finetuned on ActivityNet: CLIP4Clip (ft), ViCLIP (ft), UMT-L (ft), InternVideo2-6B (ft).

Text-to-Video (T2V) Video-to-Text (V2T) Model R@1 R@5 R@10 R@1 R@5 R@10 Multimodal LLM Embedders

LamRA 58.5 85.4 91.6 58.7 84.8 92.5 VLM2Vec 32.8 52.9 61.3 40.4 59.2 66.2 MMRet-v1.5 50.6 76.6 85.8 52.7 77.8 86.8 B3 33.8 51.9 59.6 39.2 57.4 65.3 VLM2Vec-V2 35.5 54.4 61.4 38.8 58.0 65.6 UNITE 25.8 42.9 51.6 31.4 47.6 55.0 UniME-V2 31.4 50.9 59.9 38.5 57.2 64.5

Video Foundation Models ViCLIP 15.1 - - 24.0 - InternVideo-L 30.7 - - 31.4 - UMT-L 42.8 69.6 79.8 40.7 67.6 78.6 LanguageBind 41.0 68.4 80.0 - - VideoPrism-g 52.7 79.4 - 50.3 77.1 PE-Core-G 54.7 - - 51.2 - InternVideo2-6B 63.2 85.6 92.5 56.5 82.8 90.3

Fine-tuned on ActivityNet

CLIP4Clip (ft) 40.3 – – 41.6 – – ViCLIP (ft) 49.8 – – 48.1 – – UMT-L (ft) 66.8 – – 64.4 – – InternVideo2-6B (ft) 74.1 – – 69.7 – –

VidVec 79.2 93.4 95.2 71.9 90.8 95.8

- Table 9. Effect of Optimization Approaches on MSR-VTT (T2V Recall@1).

Optimization Strategy R@1 Zero-shot 14.3 NLI 45.0 In-Context 47.7 +DSL 48.4

For video-caption textual data optimization, we process the same data split used for in-context optimization. We prompt an LLM with the dense caption together with the existing short caption, and ask it to generate a revised short caption conditioned on the dense caption, while avoiding the original short caption. Specifically for LLM we use Gemma3 (Kamath et al., 2025).

In Table 9 we further report the impact of Dual Softmax Loss (only in train) on the results.

###### C.2. In-Context Generalization

In Tab. 10, we further evaluate the generalization of our in-context data training on Qwen-2-VL backbone (Bai et al., 2025). Although Qwen2-VL exhibits slightly weaker intermediate-layer performance than VideoLLaMA3 in our layer-wise analysis, it still benefits from our lightweight in-context fine-tuning, improving over prior NLI-context training. Applying our in-context optimization to Qwen2-

- Table 10. Generalization of Token Optimization Strategy on Qwen2-VL tested on MSR-VTT-T2V-R@1

Qwen2-VL R@1 NLI (LamRA) 44.7 In-Context Optimization 47.2

- Table 11. MLLM embedder baselines used in our evaluation.

Method Model Name LamRA LamRA-Ret VLM2Vec VLM2Vec-Qwen2VL-7B MMRet-v1.5 BGE-VL-v1.5-mmeb B3 B3 Qwen2 7B VLM2Vec-V2 VLM2Vec-V2.0 UNITE Unite-Base-Qwen2-VL-7B UniME-V2 UniME-V2-Qwen2VL-7B

VL improves performance from 44.7 to 47.2, only 0.2 points below VideoLLaMA3.

#### D. More Implementation Details

In Tab. 11 we report full model names of MLLM embedders used in our evaluation.

In-Context Optimization. For our token optimization we use LoRA by PEFT on the LLM backbone, optimizing its output token at the <emb-1> position. LoRA rank is 64, and alpha is 128. We run our single epoch optimization using deepspeed zero3, with 72 pairs per B200 GPU X 4 resulting in 288 in a batch.

Evaluation. Recent state-of-the-art VFMs (e.g., InternVideo2 and PE-Core) use dual-softmax score calibration, which accounts for the query distribution at inference time. To ensure a fair comparison, we apply the same protocol to our method and to all MLLM embedder baselines. Specifically, we tune one fixed temperature per retrieval direction of T2V and V2T on the MSR-VTT validation set, scale

###### (i) NLI baseline (ii) Video Captions

Emb

Emb

Contrastive Learning

Contrastive Learning

Multimodal Large Language Model (MLLM)

Multimodal Large Language Model (MLLM)

A man and a woman are working on a computer in a dark room.

A man and a woman working on computer

Caption A

A scientist and a presenter discuss computer diagrams in a dimly lit workspace

Text

A man and a woman working together.

Video Caption B

Entailment

Figure 4. Different Optimization Approaches.

the similarity matrix, and apply dual-softmax by taking the softmax over both columns and rows.

#### E. Limitations

Our in-context optimization relies on textual video descriptions, and its effectiveness is therefore bounded by caption quality and coverage, particularly for fine-grained visual details or long-range temporal dependencies that may not be explicitly described in text.

Our full model includes an inference-time reranking stage, bears further computational cost through additional forward passes over the top-K candidates. This may limit applicability for large K values. Furthermore, our reranking relies on a simple pairwise scoring using the MLLM head, and exploring more advanced reranking strategies remains future work.

