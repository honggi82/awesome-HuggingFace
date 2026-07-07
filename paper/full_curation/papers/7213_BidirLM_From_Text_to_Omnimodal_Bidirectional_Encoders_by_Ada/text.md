# arXiv:2604.02045v1[cs.CL]2Apr2026

## BIDIRLM: FROM TEXT TO OMNIMODAL BIDIRECTIONAL ENCODERS BY ADAPTING AND COMPOSING CAUSAL LLMS

Nicolas Boizard1,3 The´o Deschamps-Berger1 Hippolyte Gisserot-Boukhlef2,3 C´eline Hudelot3 Pierre Colombo4

1Diabolocom 2Artefact Research Center 3MICS, CentraleSup´elec, Universit´e Paris-Saclay 4Cohere

Transforming causal generative language models into bidirectional encoders offers a powerful alternative to BERT-style architectures. However, current approaches remain limited: they lack consensus on optimal training objectives, suffer from catastrophic forgetting at scale, and fail to flexibly integrate the vast ecosystem of specialized generative models. In this work, through systematic ablations on the Gemma3 and Qwen3 families, we identify the key factors driving successful adaptation, highlighting the critical role of an often-omitted prior masking phase. To scale this process without original pre-training data, we introduce a dual strategy combining linear weight merging with a lightweight multi-domain data mixture that mitigates catastrophic forgetting. Finally, we augment our encoders by merging them with specialized causal models, seamlessly transferring modality- and domain-specific capabilities. This open-source recipe, designed for any causal decoder LLM, yields BidirLM, a family of five encoders that outperform alternatives on text, vision, and audio representation benchmarks.

Correspondence: nicolas.boizard@centralesupelec.fr Website: https://huggingface.co/BidirLM Date: April 2, 2026

### 1 Introduction

Causal large language models (LLMs) are not only dominant as generators but serve as the foundation of a vast ecosystem of specialized variants: code (Lozhkov et al., 2024), mathematics (Shao et al., 2024), safety (Zhao et al., 2025a), vision (Bai et al., 2025), and audio (Shi et al., 2026), collectively representing millions of GPU hours of open-source knowledge. Yet, representation tasks remain bound to bidirectional encoders (Devlin et al., 2019; He et al., 2023; Boizard et al., 2025a), leaving this knowledge untapped. Repurposing causal models into encoders is therefore a compelling goal, and recent work has begun to explore this direction (Ma et al., 2023; BehnamGhader et al., 2024; Wang et al., 2024a; Babakhin et al., 2025; Gisserot-Boukhlef et al., 2026). However, this adaptation landscape remains fragmented around three core questions. We address each through a fully opensource framework, validated under strictly identical conditions on Gemma3 and Qwen3.

What drives adaptation quality? Existing methods conflate critical design choices like training objectives and attention mechanisms, leaving no consensus on what drives quality. Through controlled ablations (§2, §3), we disentangle these factors and show that enabling bidirectional attention via a masking objective (a step often omitted) is critical to unlock performance on task-specific benchmarks, while contrastive objectives primarily drive generic embedding quality.

Can adaptation scale without the original pre-training data? Many adapted models are developed by the same organizations that trained the underlying base models (Vera et al., 2025; Zhang et al., 2025). This raises concerns about reproducibility, as these adaptations may implicitly benefit from alignment with undisclosed pre-training corpora, potentially masking the catastrophic forgetting that occurs under distribution shifts. To scale adaptation under strict independent data constraints (§ 4), we propose a dual strategy combining training-free linear weight merging with a lightweight multi-domain data mixture. This approach yields adapted models that outperform current open-source alternatives (§5).

Can adapted encoders compose with the causal ecosystem? Current adapted encoders rely on rigid pipelines that fail to compose with other specialized causal models derived

from the same backbone. By ignoring the vast ecosystem that motivates starting from causal architectures, these methods leave thousands of GPU hours of open-source specialization unused. We address this by pushing the boundaries of weight merging (§6). We seamlessly integrate knowledge from specialized causal variants, extending our encoders to new domains and modalities (safety, audio, vision) without requiring full pipeline re-training.

Accompanying this work, we release the multilingual BidirLM series, which outperforms open-source alternatives on text, vision, and audio representation benchmarks: BidirLM270M/1B (Gemma3-based), BidirLM-0.6B/1.7B (Qwen3-based), and BidirLM-Omni-2.5B (text, vision, audio), alongside our training corpus, checkpoints, and experimental variants.

### 2 Experimental Setup

Models and adaptation objectives. We adapt two causal language model families initialized from pretrained weights: Gemma3 (270M, 1B) (Gemma et al., 2025) and Qwen3 (600M, 1.7B) (Yang et al., 2025a).1 We use the smaller models for ablation studies and the larger models for scaling analysis, covering typical embedding model sizes. From these base models, we derive five distinct variants (detailed in Figure 1) by switching from causal to bidirectional attention and applying two core adaptation objectives either individually or sequentially: Masked Next-Token Prediction (MNTP) (BehnamGhader et al., 2024) and InfoNCE contrastive training (van den Oord et al., 2019).2

###### Bi+MNTP (3)

| | |
|---|---|
| |Contrastive<br><br>followed by|

###### MNTP

Adapted Weights Bidirectional Attention

| | |
|---|---|
| | |

Adaptation Phase

Base (1)

###### Bi+Base (2)

###### Bi+MNTP+Contrastive (5)

Sequential Adaptation Phase

Pretrained Base Weights Causal Attention

Pretrained Base Weights Bidirectional Attention

Sequential Adapted Weights Bidirectional Attention

###### Bi+Contrastive (4)

###### Contrastive

Adapted Weights Bidirectional Attention

Adaptation Phase

- Figure 1: Base (1): The original causal model. Bi+Base (2): The Base model with bidirectional attention enabled. Bi+MNTP (3): The Bi+Base model with an MNTP adaptation phase. Bi+Contrastive (4): The Bi+Base model with a contrastive adaptation phase. Bi+MNTP+Contrastive (5): The Bi+Base model adapted sequentially using MNTP followed by contrastive training. Intermediate dashed blocks denote adaptation phases.

Adaptation corpus. All adaptation experiments rely exclusively on open-source datasets. For clarity, we structure our corpora along three distinct domains:

- 1. English: Masking uses FineWeb-Edu (Penedo et al., 2024); contrastive training uses the English subset of KaLM-embedding (Zhao et al., 2025b) (7 hard negatives per query).
- 2. Multi-domain: Masking relies on FineWeb-Edu (English), FineWeb2-HQ (multilingual, 20 languages) (Messmer et al., 2026), FineMath (Liu et al., 2024a) (mathematics), and Stack V2 (code, 34 languages) (Lozhkov et al., 2024). Contrastive training employs a merged corpus of 89 datasets (1 to 7 hard negatives per query) detailed in Appendix C.
- 3. Multimodal: We introduce Omni-Contrastive,3 a 1.8M-pair contrastive corpus mixing 65% text-text (from the multi-domain corpus), 17.5% audio-text from Laion-Audio-300M (200K, audio-description) and LibriSpeech ASR (100K, speech-transcription), and 17.5% image-text from Colpali (Faysse et al., 2024) (100K, document-query), NatCap (Teiletche et al., 2025), and MSCOCO (Lin et al., 2015) (100K each, image-description).

1Models utilized: Gemma3-270M, Gemma3-1B, Qwen3-0.6B, and Qwen3-1.7B; Appendix A. 2Loss definitions and adaptation hyperparameters are provided in Appendix B. 3Dataset available at: https://huggingface.co/datasets/BidirLM/BidirLM-Omni-Contrastive

Evaluation protocol. To reflect the current usage landscape, we assess encoder performance across diverse representation tasks under two distinct paradigms:

- 1. Fine-tuning evaluation: We apply full-parameter adaptation for downstream tasks spanning the XTREME benchmark (Hu et al., 2020) and four specific task categories: Information Retrieval (IR) via MIRACL (Zhang et al., 2023) and CodeSearchNet (Husain et al., 2020); Sequence Classification (SC) via MNLI (Williams et al., 2018), XNLI (Conneau et al., 2018-10), PAWS-X (Yang et al., 2019), MathShepherd (Wang et al., 2024c), and CodeComplexity (Jeon et al., 2023); Token Classification (TC) via PAN-X and POS (Hu et al., 2020); and Sequence Regression (SR) via Seahorse (Clark et al., 2023-12).4
- 2. Embedding evaluation: We assess off-the-shelf embedding performance via zero-shot and linear probing on MTEB-style benchmarks. Text evaluation uses English and Multilingual MTEB v2 (Muennighoff et al., 2023; Enevoldsen et al., 2025), while cross-modal capabilities rely on MIEB lite (Xiao et al., 2025) (image-only, image-to-image, text-toimage) and MAEB beta (Assadi et al., 2026) (audio-only, audio-to-audio, audio-text).

Causal ecosystem. To verify the capacity to compose our encoder with specialized variants from the causal ecosystem, we perform post-adaptation specialization across three domains:

- 1. Safety moderation: We use Qwen3Guard-Gen-0.6B (Zhao et al., 2025a) to transfer safety moderation knowledge to our encoder, assessed via safe/unsafe classification on the Beaver (Ji et al., 2023), Safe (Ji et al., 2025), and Aegis (Ghosh et al., 2025) datasets.
- 2. Vision: Qwen3-VL-2B-Instruct (Bai et al., 2025) transfers visual-textual knowledge, evaluated via visual-textual entailment on the e-SNLI-VE (Do et al., 2021) benchmark.
- 3. Audio: Qwen3-ASR-0.6B (Shi et al., 2026) transfers audio understanding, which we evaluate via textual comprehension with vocal questions on the BoolQ dataset.

- 3 Adaptation Strategies

Recent adaptation methods often rely exclusively on contrastive training, omitting masking objectives or bidirectional attention. To disentangle these choices,5 we evaluate our five adaptation variants (Figure 1) on Gemma3-270M and Qwen3-0.6B. We adapt these models using 10B tokens for masking and 3M samples for contrastive training on the English corpus (§2), and report their downstream performance against the causal baseline (Base) in Figure 2.

Miracl (IR)

###### XNLI (SC)

Seahorse (SR)

Xtreme Panx (TC)

MTEB (eng, v2)

nDCG@10

Accuracy

Spearman Correlation

F1 Score

Average Score

| | |
|---|---|
| |+0.0|
| |-0.4<br><br>-0.0<br><br>|
| |-0.8|

| |+36.4+36.4+36.3+36.5| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

+27.7

+27.5

97 +84.8+85.8+85.4+85.9

94

61

60

Gemma3

- 84
- 85
- 86

+1.2 +0.9

+14.6

48

92

+8.5

-7.8

35

57

32

10

-15.5

| |+34.3 +35.7|
|---|---|
| |+0.4<br><br>+15.8<br><br>|
| | |

94 +4.8 +5.2 +5.4 +5.3

65

58

+1.8

+5.4 +5.7

+1.1 +1.1

90

97

+1.5

Qwen3

+1.1

+0.1

+0.5

49

88

-3.0

89

40

28

94

-1.6

86

-9.4

Base Bi+Base

Bi+Contrastive

Bi+MNTP

Bi+MNTP+Contrastive

| |
|---|

| |
|---|

| |
|---|

- Figure 2: Performance comparison of model variants across downstream tasks. Bars illustrate the absolute performance change relative to the unmodified Base model. Exact point differences are annotated above or below each bar.

4Hyperparameters for fine-tuning and extensive dataset descriptions are provided in Appendix D. 5Additional comparisons of MNTP and traditional masking objectives are provided in Appendix F.

Bidirectional attention drives performance. As shown in Figure 2, enabling bidirectional attention at the fine-tuning stage only (Bi+Base) produces mixed results: it improves token classification and retrieval across both architectures but degrades performance on XNLI and Seahorse. However, introducing an MNTP adaptation phase unlocks the full benefit of bidirectional attention, boosting performance across all tasks with notable gains on XNLI and Seahorse (Gemma: +0.8 and +9.0; Qwen: +2.7 and +8.4, respectively).

Masking and contrastive objectives are complementary. Consistent with prior work (Gao et al., 2021; Li et al., 2023; BehnamGhader et al., 2024), contrastive objectives drive generic embedding performance under zero-shot and linear probing evaluation (Figure 2), outperforming Bi+MNTP on MTEB by over 13 points across both architectures. However, our controlled comparison reveals that contrastive training alone sacrifices performance on tasks requiring full-parameter fine-tuning (e.g., XNLI, Seahorse) for embedding gains. To leverage the strengths of both paradigms, we employ a sequential adaptation strategy: MNTP followed by contrastive training (Bi+MNTP+Contrastive). This approach matches or surpasses the individual objectives across all tasks.

Key finding: Recent contrastive-only adaptations sacrifice fine-tuning quality for embedding gains; restoring a prior MNTP phase enables achieving peak performance across both regimes.

### 4 Scaling Adaptation Phases

Following §3, we scale the adaptation process while aiming to preserve the foundational knowledge of the base models. However, training on corpora diverging from the original pre-training distribution inherently risks alignment drift and catastrophic forgetting.

###### 4.1 Catastrophic Forgetting

Miracl

#### MNLI

Other Domains

nDCG@10

Accuracy

nDCG@10 / Accuracy

7

3

11

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Gemma3

Bi+MNTP

0

0

0

-7

-3

-11

4

2

3

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Bi+MNTP

Qwen3

0

0

0

-4

-2

-3

10B 20B 30BMergeMultilingual+Merge

10B 20B 30BMergeMultilingual+Merge

10B 20B 30BMergeMultilingual+Merge

Fra Ara Eng Avg CodeSearchNet (IR) Math Shepherd (SC)

- Figure 3: Evolution of model performances during long run adaptation. Solid lines depict the absolute score change relative to the initial 10B adaptation, while dotted lines highlight the impact of complementary solutions to retain general knowledge.

To assess forgetting under realistic constraints (Figure 3), we extend MNTP adaptation for Gemma3-270M and Qwen3-0.6B from 10B to 30B tokens on the English corpus (§2). Simultaneously, we monitor multi-domain performance at 10B-token intervals across multilingual (MIRACL, XNLI), code (CodeSearchNet), and math (Math Shepherd) benchmarks.

Scaled Adaptation Induces Forgetting. As expected, scaling adaptation on a distribution unaligned with the original pre-training data leads to a clear forgetting phenomenon as

training progresses: Gemma declines on Arabic (-7.0 points on MIRACL, -2.0 on XNLI), while Qwen demonstrates forgetting on Math Shepherd (-1.5) and CodeSearchNet (-2.0). To counteract this degradation, we propose two complementary approaches: a data-free model merging strategy and a lightweight multi-domain data mixture.

Model Merging Mitigates Forgetting and Preserves Bidirectional Capabilities. Motivated by the observation that the adapted and base models remain close in weight space, with an average cosine similarity of 0.78 for Gemma and 0.97 for Qwen,6 we explore linear model merging (Wortsman et al., 2022b), a technique shown to mitigate forgetting by averaging weights between different checkpoints. Specifically, we merge the 30B-token English-only MNTP models with their original base checkpoints using interpolation ratios ranging from 10% to 90% (30% means a 0.3 weighting factor is applied to the base model).

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 95

nDCG@10

Miracl (IR)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 25 50 75 100

Merging Ratio (%)

77

81

84

Accuracy

MNLI (SC)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 25 50 75 100

Merging Ratio (%)

54

65

77

nDCG@10

CodeSearch (IR)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 25 50 75 100

Merging Ratio (%)

80

86

92

Accuracy

MathShepherd (SC)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 25 50 75 100

Merging Ratio (%)

0 2 4

Average Normalized

Rank

Gemma3 (Bi+MNTP) Qwen3 (Bi+MNTP)

Figure 4: Model performance across merging ratios. The first four columns report task scores, while the rightmost column reports the model ranking based on average normalized performance across all tasks. Merging ratio index dictates the interpolation weight.

As shown in Figure 4, performance peaks near a 50% merging ratio, intuitively balancing the adapted bidirectional attention patterns with base model’s distributional coverage. Consequently, we report the 30B adapted checkpoints at this 50% ratio (denoted Merge) in Figure 3. Compared to unmerged baselines, these models yield substantial cross-domain gains: +6 points on Arabic MNLI and code for Gemma, and +4 points in math for Qwen. Overall, merging emerges as a highly effective, data-free strategy to recover original knowledge.

Multi-domain data mixtures and weight merging yield optimal retention. Complementing model merging, we investigate how multi-domain training mixtures mitigate adaptation forgetting without prior knowledge of the original pre-training distribution.

0 10 20 30

Mix Ratio (%)

89

93

- 96

93

90

0 25 50 75 100

Merging Ratio (%)

Miracl (IR)

##### MNLI (SC)

CodeSearch (IR)

MathShepherd (SC)

Rank

nDCG@10

Accuracy

nDCG@10

Accuracy

Average Normalized

85

77

92

0 2 4

81

65

87

76

52

83

0 10 20 30

0 10 20 30

0 10 20 30

0 10 20 30

Mix Ratio (%)

Mix Ratio (%)

Mix Ratio (%)

Mix Ratio (%)

Gemma3 (Bi+MNTP) Qwen3 (Bi+MNTP) Merged

- Figure 5: Model performance across data mix ratios. The first four columns report task scores, while the rightmost column reports the model ranking based on average normalized performance across all tasks. The mix ratio specifies the proportion of multi-domain data.

As illustrated in Figure 5, we replace part of our initial English mix with multi-domain data (§2) distributed equally across multilingual, math, and code domains. We observe that

6We further analyze this finding in §F.2, detailing the layer-wise similarity evolution.

performance plateaus when allocating just 20% to 30% of the mixture to this multi-domain subset, indicating that a small fraction is sufficient to preserve original knowledge. To control for this factor, we fix this ratio at 20% for subsequent experiments. Building on our merging strategy, interpolating this checkpoint with the original base weights yields further gains. This final Multilingual+Merge configuration (Figure 3) achieves our best overall results, with an average improvement of +2 points on XNLI and MIRACL for both architectures, and up to +11 points on code benchmarks for Gemma.

Key finding: Combining weight merging with a lightweight multi-domain data mixture preserves the base model’s foundational knowledge and newly acquired bidirectional capabilities.

### 5 Frontier Performance Through Scaled Adaptation

Building upon our best-performing adaptation strategies (§3) and empirical findings (§4), we scale our approach to larger architectures, yielding four Bi+MNTP variants: Gemma3 (270M and 1B) and Qwen3 (0.6B and 1.7B). To establish strong general-purpose embedding capabilities, we execute the second step of our biphasic pipeline via contrastive training on 10M samples from our multi-domain corpus (§2). We evaluate these final models, denoted the BidirLM series, on MTEB and an augmented XTREME benchmark (incorporating math and code domains, detailed in Appendix D), plotting the Pareto frontier against the latest fully open-source models (i.e., those releasing complete contrastive training data).

XTREME-Benchmark Augmented

MTEB (Multilingual V2)

Average Score

Average Score

BidirLM-1.7B BidirLM-0.6B BidirLM-1B

Qwen3-Emb-0.6B

89

65

BidirLM-1.7B

EmbGemma-300m

GTE-Qwen2-7B

EuroBERT-610m

EuroBERT-2.1B

BidirLM-1B BidirLM-0.6B

87

60

BGE-M3

BidirLM-270M Qwen3-Emb-0.6B

KaLM-mini

mmBERT-base EuroBERT-210m

85

55

BidirLM-270m

0.3 0.5 1 2

0.3 0.5 1 2 5 10

Model Size (Billions)

Model Size (Billions)

Gemma-based Qwen-based Open data Closed data Open data Pareto frontier

- Figure 6: Multilingual model performance by size. We report the average scores of the latest multilingual models across individual tasks on the XTREME and MTEB benchmarks. The dashed line indicates the open-source performance Pareto frontier.

Adapted models redefine the Pareto frontier on task-specific benchmarks. Under fullparameter fine-tuning, all BidirLM variants establish a new performance frontier on the augmented XTREME benchmark. Notably, BidirLM-270M matches the performance of mmBERT-base (Marone et al., 2025) while utilizing 10% fewer parameters, and BidirLM0.6B outperforms its closest counterpart (EuroBERT-610m) by more than 1 point.7

Adapted models redefine the open-source Pareto frontier on generic embedding tasks. Traditionally, generic embeddings and task-specific fine-tuning rely on separate model variants. Our adaptation eliminates this trade-off: beyond achieving the strongest performance on full-parameter fine-tuning, the exact same BidirLM variants advance the opensource Pareto frontier across three of our four size configurations on generic embedding

7Models such as BGE-M3, KaLM, and EmbedGemma couldn’t be evaluated due to their lack of architectural support for sentence or token classification, a key limitation of embedding-only models.

benchmarks (MTEB). Notably, we accomplish this using only classical contrastive training, completely avoiding knowledge distillation from proprietary models or costly multi-run averaging. Consequently, our models constitute robust open-source baselines for future work challenging closed-source systems such as Qwen3-Embedding and EmbeddingGemma.

### 6 Domain and Modality Specialization

Motivated by the observation that weight merging efficiently preserves the base model’s foundational knowledge and bidirectional capabilities (§4), we push the boundaries of this technique to tailor our generic encoders to new domains and modalities, harnessing the vast ecosystem of specialized generative models.

###### 6.1 Domain Alignment

We explore domain knowledge transfer by exploiting the shared backbone between our Bi+MNTP Qwen3-0.6B and the Qwen3Guard-Gen-0.6B safety model (§2). We merge them at a 50% ratio8 (cos sim: 0.97) and perform 500 fine-tuning steps on the Beaver training set (two minutes on one MI250X GPU). We evaluate the resulting encoder on the Beaver test set and two out-of-distribution benchmarks (Safe and Aegis) against the specialist causal model fine-tuned with bidirectional attention (Bi+Specialist) and the Bi+MNTP models.

Beaver

Safe

Aegis

F1 Score

F1 Score

F1 Score

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

82

89

78

Qwen3

75

82

69

68

75

60

0 125 250 375 500

0 125 250 375 500

0 125 250 375 500

Training Step

Training Step

Training Step

Bi+Specialist Bi+MNTP Bi+MNTP+Merge

Figure 7: Evolution of performance during domain specialization. We report test split performance on Beaver, SAFE and Aegis. Solid lines correspond to the exponential moving averaged (EMA) curves (α = 0.4), with shaded areas showing raw value deviation.

Merged model outperforms all baseline configurations. As shown in Figure 7, the merged encoder (Bi+MNTP+Merge) outperforms all other configurations by over 1 point on average. It also shows better out-of-distribution generalization and greater training stability, with minimal variance between raw measurements and EMA-smoothed curves.

Merging enables rapid sample-efficient adaptation. The Bi+MNTP+Merge model reaches over 93% of its peak performance across all benchmarks in just 20 steps (80 samples). At this early training stage, it outperforms all other variants by a margin of more than 5 points.

###### 6.2 Modality Alignment

We extend this approach to new modalities by merging the bimodal vision-text Qwen3-VL2B-Instruct and the unimodal audio Qwen3-ASR-0.6B models with our adapted Bi+MNTP encoders (Qwen3-1.7B and Qwen3-0.6B, respectively) in equal proportions (cosine similarities: 0.97 for vision, 0.93 for audio). Finally, we conduct a 500-step fine-tuning phase on e-SNLI-VE (visual-textual entailment) and BoolQ-Audio (vocal comprehension) (§2).

Modality Adaptation Reveals a Warm-Up Phase. Merged variants yield the highest overall performance (Figure 8), exceeding Bi+Specialist by over 1 and 15 points on vision and audio

8We provide a detailed analysis for merge ratios ∈ {0,0.25,0.5,0.75,1} in §F.3

E-SNLI-VE (Visual)

BoolQ-Audio (Audio)

F1 Score

F1 Score

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

80

77

Qwen3

51

58

22

39

0 125 250 375 500

0 125 250 375 500

Training Step

Training Step

Bi+Specialist Bi+MNTP Bi+MNTP+Merge

- Figure 8: Evolution of performance during modality specialization. We report test split F1 score on e-SNLI-VE (vision) and Boolq-Audio (Audio). Solid lines correspond to exponential moving average curves (α = 0.4), with shaded areas showing raw data deviation.

tasks, and surpassing unmerged baselines by over 30 and 19 points respectively. Unlike in domain adaptation, the merged variant exhibits an initial warm-up period of 100 vision steps and 175 audio steps. Consistent with prior literature, this warm-up phase stems from the requirement to align internal representations with the newly introduced modality heads.

Merging succeeds without shared modalities. We observe a clear gap between baseline performances. While the Bi+Specialist remains competitive in vision, trailing the merged model by only 1 point, it degrades significantly in audio. We attribute this to the input modalities of the specialist models: the vision model already possessed multimodal capabilities for text and vision, whereas the audio model was trained exclusively for unimodal speech recognition. Crucially, we observe that merging technique still succeeds, demonstrating that models can be effectively combined even when they share no prior overlapping modalities.

- 6.3 Omnimodal Alignment

Building on the observation that our encoder can easily adapt to new modalities, we introduce BidirLM-Omni-2.5B, a compact omnimodal model. We construct this by merging the textual backbones of three Qwen3-1.7B variants (ASR, VL, and Bi+MNTP) in equal proportions, appending their respective audio and visual heads (Appendix E). Following contrastive training on our multimodal corpus (§2), we evaluate the model against numerous baselines (Figure 9) across MTEB (Text), MIEB (Image), and MAEB (Audio).

0.3 0.5 1 2 5 10

Model Size (Billions)

55

60

65

Average Score

MTEB (Multilingual V2)

Qwen3-Emb-0.6B

EmbGemma...

BidirLM-1B

BidirLM-0.6B

BidirLM-270m

BidirLM-Omni-2.5B

0.3 0.5 1 2 5 10

Model Size (Billions)

36

46

56

Average Score

MIEB (Image)

SigLIP-400m

E5-V

Nemotron-Omni-3B

CLIP-L-DC

CLIP-B16-DC

Jina-CLIP-v1

BidirLM-Omni-2.5B

0.3 0.5 1 2 5 10

Model Size (Billions)

25

38

52

Average Score

MAEB (Audio)

Qwen2-Audio-7B

LCO-Omni-7B (8.9B)

CLAP-general

LCO-Omni-3B (4.7B)

MS-CLAP-23

MuQ-MuLan

Nemotron-Omni-3B

BidirLM-Omni-2.5B

BidirLM Omnimodal Open data Closed data Open data Pareto frontier

- Figure 9: Embedding model performance by size. Average score across individual tasks on MTEB Multilingual V2, MIEB, and MAEB, as a function of model size. The dashed line shows the Pareto frontier over open training data models.

BidirLM-Omni sets a new omnimodal state of the art. BidirLM-Omni-2.5B outperforms the latest best-performing omnimodal model, Nemotron-Omni-3B (Xu et al., 2025), across all modalities, achieving notable gains on text (+17) and image (+5) benchmarks while being nearly half the size (2.5B vs. 4.8B).

BidirLM-Omni surpasses unimodal specialists several times larger. Beyond outperforming its omnimodal counterparts, BidirLM-Omni-2.5B establishes new Pareto frontiers regardless of data transparency. Notably, the merging process efficiently leverages the strengths of each model variant, enabling it to rank first among all baselines on the MIEB benchmark and third on MAEB, surpassing bimodal architectures many times its size.

Composing specialized models yields efficient and flexible encoders. By reusing existing specialized models rather than training from scratch, BidirLM-Omni required only 250 additional GPU hours (MI250X) of compute for merging and contrastive training, demonstrating that new omnimodal architectures can be assembled incrementally as specialized models become available, bypassing the need to retrain the entire pipeline.

Key finding: Weight merging enables the efficient composition of domain and modality causal model specialists into an adapted encoder, succeeding even in the absence of shared prior modalities.

### 7 Related Work

Adapting Causal Models for Generic and Multimodal Representations. Causal LLMs have emerged as strong backbones for text embeddings (Ma et al., 2023; Liu et al., 2024b; Springer et al., 2025), with adaptation strategies generally falling into two paradigms. The first injects bidirectionality through masking-based objectives, using either classical masked language modeling (MLM) (Devlin et al., 2019) or the next-token variant MNTP (BehnamGhader et al., 2024), prior to fine-tuning. While BehnamGhader et al. (2024) first proposed the MNTP-then-contrastive pipeline, their evaluation did not isolate the contributions of bidirectional attention, the masking objective, and contrastive training itself. Consequently, the second paradigm, which is now dominant in practice, skips the masking phase entirely and applies contrastive learning directly (Le-Khac et al., 2020; Wang et al., 2024a; Lee et al., 2025). Within this approach, attention design varies: some methods enable full bidirectionality (e.g., Embedding-Gemma (Vera et al., 2025)), while others preserve causal masking (e.g., Qwen3 Embedding (Zhang et al., 2025)). This contrastive paradigm has recently been extended to multimodal representations, yielding models like VLM2Vec (Jiang et al., 2025) and Nemotron-Omni (Xu et al., 2025).

Weight Merging for Knowledge Transfer and Specialization. Adapting models to new objectives and distributions inevitably risks catastrophic forgetting (French, 1999). While traditional continual learning relies on compute-intensive replay buffers or regularization (Rolnick et al., 2019; Wang et al., 2024b), post-hoc weight merging (Model Soups (Wortsman et al., 2022b) or Task Arithmetic (Ilharco et al., 2023)) offers a highly efficient alternative, enabling models to seamlessly adapt to new distributions. However, these techniques have historically been applied to models sharing similar objectives and attention mechanisms.

### 8 Conclusion

In this work, we introduce a unified, fully open-source framework for transforming causal decoder LLMs into bidirectional encoders spanning text to multiple modality domains. Through systematic comparisons, we show that the masking phase omitted by recent contrastive-only methods is in fact critical for fine-tunig performance. To scale this adaptation without proprietary pre-training data, we employ a dual strategy of linear weight merging and a lightweight multi-domain data mixture, yielding the BidirLM model family. Rather than building inflexible systems, our framework seamlessly composes specialized generative models with our adapted encoders, enabling efficient cross-modal and domainspecific adaptation without retraining entire pipelines, culminating in BidirLM-Omni.

### Future Work

Contrastive training. Our ablations focused on the masking phase, a step frequently omitted in concurrent work. While contrastive training already benefits from an extensive body of prior work and ablations (Xu et al., 2025; Zhang et al., 2025; Vera et al., 2025; Hu et al., 2025), systematically studying data composition, hard-negative mining strategies, and scaling behavior in the omnimodal setting remains a natural next step.

Additional mitigation techniques and model architectures. In this study, we rely on linear merging and data mixing, both lightweight by design. We plan to explore richer regularization strategies, notably knowledge distillation (Hinton et al., 2015) from the base model. Utilizing recent techniques that enable cross-architectural distillation (Boizard et al., 2025b; Minixhofer et al., 2025) may offer stronger knowledge retention at the cost of additional compute. Finally, validating our framework on non-transformer causal architectures, such as state-space models (Gu & Dao, 2024; Yang et al., 2025b), remains an open question.

### Acknowledgments

We sincerely thank the ADASTRA supercomputer (CINES) for its high-performance computing (HPC) resources, provided through grant A0181016236. This work was also supported by the Jean Zay supercomputer (GENCI-IDRIS-CNRS) through grant AD010617149, and the ROMEO HPC center at the University of Reims. Furthermore, we gratefully acknowledge the support of the French government through the France 2030 program as part of the ArGiMi project.

### References

Adnan El Assadi, Isaac Chung, Chenghao Xiao, Roman Solomatin, Animesh Jha, Rahul Chand, Silky Singh, Kaitlyn Wang, Ali Sartaz Khan, Marc Moussa Nasser, Sufen Fong, Pengfei He, Alan Xiao, Ayush Sunil Munot, Aditya Shrivastava, Artem Gazizov, Niklas Muennighoff, and Kenneth Enevoldsen. Maeb: Massive audio embedding benchmark, 2026. URL https://arxiv.org/abs/2602.16008.

Yauhen Babakhin, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo Liu, and Even Oldridge. Llama-embed-nemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks, 2025. URL https://arxiv. org/abs/2511.07025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, and others. Ms marco: A human generated machine reading comprehension dataset. 2016. URL https://arxiv. org/abs/1611.09268.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. Llm2vec: Large language models are secretly powerful text encoders, 2024. URL https://arxiv.org/abs/2404.05961.

Nicolas Boizard, Hippolyte Gisserot-Boukhlef, Duarte M. Alves, Andr´e Martins, Ayoub Hammal, Caio Corro, C´eline Hudelot, Emmanuel Malherbe, Etienne Malaboeuf, Fanny Jourdan, Gabriel Hautreux, Jo˜ao Alves, Kevin El-Haddad, Manuel Faysse, Maxime Peyrard, Nuno M. Guerreiro, Patrick Fernandes, Ricardo Rei, and Pierre Colombo. Eurobert: Scaling multilingual encoders for european languages, 2025a. URL https: //arxiv.org/abs/2503.05500.

Nicolas Boizard, Kevin El Haddad, C´eline Hudelot, and Pierre Colombo. Towards crosstokenizer distillation: the universal logit distillation loss for llms, 2025b. URL https: //arxiv.org/abs/2402.12030.

Elizabeth Clark, Shruti Rijhwani, Sebastian Gehrmann, Joshua Maynez, Roee Aharoni, Vitaly Nikolaev, Thibault Sellam, Aditya Siddhant, Dipanjan Das, and Ankur Parikh. SEAHORSE: A multilingual, multifaceted dataset for summarization evaluation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2023-12. URL https://aclanthology.org/ 2023.emnlp-main.584.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. XNLI: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2018-10. URL https://aclanthology.org/ D18-1269/.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. URL https://arxiv. org/abs/1810.04805.

Virginie Do, Oana-Maria Camburu, Zeynep Akata, and Thomas Lukasiewicz. e-snli-ve: Corrected visual-textual entailment with natural language explanations, 2021. URL https://arxiv.org/abs/2004.03744.

Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, M´arton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzeminski,´ Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Omer¨ ¸Ca˘gatan, Akash Kundu, Martin Bernstorff, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Po´swiata, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Bj¨orn Pluster,¨ Jan Philipp Harries, Lo¨ıc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Suppa,ˇ Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Gunther,¨ Mengzhou Xia, Weijia Shi, Xing Han Lu,` Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorff, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, and Niklas Muennighoff. Mmteb: Massive multilingual text embedding benchmark, 2025. URL https://arxiv.org/abs/2502.13595.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models, 2024. URL https://arxiv.org/abs/2407.01449.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis, 2020. URL https://arxiv.org/abs/ 1912.05671.

Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 6894–6910, 2021.

Team Gemma, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Ga¨el Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, Andr´as Gy¨orgy, Andr´e Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucinska,´ Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew

Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim P˜oder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and L´eonard Hussenot. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786.

Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese, and Christopher Parisien. Aegis2.0: A diverse ai safety dataset and risks taxonomy for alignment of llm guardrails, 2025. URL https: //arxiv.org/abs/2501.09004.

Hippolyte Gisserot-Boukhlef, Nicolas Boizard, Manuel Faysse, Duarte M. Alves, Emmanuel Malherbe, Andr´e F. T. Martins, C´eline Hudelot, and Pierre Colombo. Should we still pretrain encoders with masked language modeling?, 2026. URL https://arxiv.org/abs/ 2507.00994.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces,

2024. URL https://arxiv.org/abs/2312.00752.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing, 2023. URL https://arxiv.org/abs/2111.09543.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network,

2015. URL https://arxiv.org/abs/1503.02531.

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. Xtreme: A massively multilingual multi-task benchmark for evaluating crosslingual generalization, 2020. URL https://arxiv.org/abs/2003.11080.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024. URL https://arxiv.org/abs/2404.06395.

Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Haofen Wang, Jun Yu, and Min Zhang. Kalmembedding: Superior training data brings a stronger embedding model, 2025. URL https://arxiv.org/abs/2501.01028.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. 2019. URL https://arxiv.org/abs/1909.09436.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search, 2020. URL https://arxiv.org/abs/1909.09436.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic,

2023. URL https://arxiv.org/abs/2212.04089.

Mingi Jeon, Seung-yeop Baik, Joonghyuk Hahn, Yo-Sub Han, and Sang-Ki Ko. Deep learning-based source code complexity prediction. openreview, 2023. URL https:// openreview.net/forum?id=9irBKvxsw9.

Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Chi Zhang, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset, 2023. URL https://arxiv.org/abs/2307.04657.

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Juntao Dai, Boren Zheng, Tianyi Qiu, Jiayi Zhou, Kaile Wang, Boxuan Li, Sirui Han, Yike Guo, and Yaodong Yang. Pkusaferlhf: Towards multi-level safety alignment for llms with human preference, 2025. URL https://arxiv.org/abs/2406.15513.

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks, 2025. URL https://arxiv.org/abs/2410.05160.

Phuc H. Le-Khac, Graham Healy, and Alan F. Smeaton. Contrastive representation learning: A framework and review. IEEE Access, 8:193907–193934, 2020. ISSN 2169-3536. doi: 10.1109/access.2020.3031549. URL http://dx.doi.org/10.1109/ACCESS.2020.3031549.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models, 2025. URL https://arxiv.org/abs/2405.17428.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning, 2023. URL https://arxiv.org/abs/2308.03281.

Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context, 2015. URL https://arxiv.org/abs/1405.0312.

Yan Liu, Renren Jin, Ling Shi, Zheng Yao, and Deyi Xiong. Finemath: A fine-grained mathematical evaluation benchmark for chinese large language models, 2024a. URL https://arxiv.org/abs/2403.07747.

Zheng Liu, Chaofan Li, Shitao Xiao, Yingxia Shao, and Defu Lian. Llama2Vec: Unsupervised adaptation of large language models for dense retrieval. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3490–3500, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 191. URL https://aclanthology.org/2024.acl-long.191/.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Munoz˜ Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder 2 and the stack v2: The next generation, 2024. URL https://arxiv.org/abs/2402.19173.

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. Fine-tuning llama for multi-stage text retrieval, 2023. URL https://arxiv.org/abs/2310.08319.

Marc Marone, Orion Weller, William Fleshman, Eugene Yang, Dawn Lawrie, and Benjamin Van Durme. mmbert: A modern multilingual encoder with annealed language learning, 2025. URL https://arxiv.org/abs/2509.06888.

Bettina Messmer, Vinko Sabolˇcec, and Martin Jaggi. Enhancing multilingual llm pretraining with model-based data selection, 2026. URL https://arxiv.org/abs/2502.10361.

Benjamin Minixhofer, Ivan Vuli´c, and Edoardo Maria Ponti. Universal cross-tokenizer distillation via approximate likelihood matching, 2025. URL https://arxiv.org/abs/ 2503.20083.

Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. MTEB: Massive text embedding benchmark, 2023. URL http://arxiv.org/abs/2210.07316.

Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajiˇc, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman. Universal dependencies v2: An evergrowing multilingual treebank collection, 2020. URL https://arxiv.org/abs/2004.10643.

Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models, 2023. URL https://arxiv.org/ abs/2305.12827.

Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. Cross-lingual name tagging and linking for 282 languages. In Regina Barzilay and Min-Yen Kan (eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1946–1958, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1178. URL https://aclanthology.org/P17-1178/.

Guilherme Penedo, Hynek Kydl´ıˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL https://arxiv.org/abs/2406.17557.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. Experience replay for continual learning, 2019. URL https://arxiv.org/abs/1811.11682.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. Qwen3-asr technical report, 2026. URL https://arxiv.org/abs/2601.21337.

Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings, 2025. URL https://arxiv.org/abs/ 2402.15449.

Paul Teiletche, Quentin Mac´e, Max Conti, Antonio Loison, Gautier Viaud, Pierre Colombo, and Manuel Faysse. Modernvbert: Towards smaller visual document retrievers, 2025. URL https://arxiv.org/abs/2510.01149.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.

Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divyashree Sreepathihalli, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hern´andez Abrego,´ Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Ga¨el Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Qin Yin, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, and Mojtaba Seyedhosseini. Embeddinggemma: Powerful and lightweight text representations, 2025. URL https://arxiv.org/abs/2509.20354.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models, 2024a. URL https://arxiv. org/abs/2401.00368.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application, 2024b. URL https://arxiv.org/abs/2302. 00487.

Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024c. URL https://arxiv.org/abs/2312.08935.

Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, and Luca Soldaini. Followir: Evaluating and teaching information retrieval models to follow instructions, 2024. URL https://arxiv.org/abs/2403.15246.

Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference, 2018. URL https://arxiv.org/ abs/1704.05426.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pp. 23965–23998. PMLR, 2022a.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael GontijoLopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time, 2022b. URL https://arxiv.org/ abs/2203.05482.

Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, M´arton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, and Niklas Muennighoff. Mieb: Massive image embedding benchmark, 2025. URL https://arxiv.org/abs/2504. 10471.

Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even Oldridge, and Benedikt Schifferer. Omni-embed-nemotron: A unified multimodal retrieval model for text, image, audio, and video, 2025. URL https://arxiv. org/abs/2510.03458.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. URL https://arxiv.org/abs/2505.09388.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2025b. URL https://arxiv.org/abs/2412.06464.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3687–3692, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1382. URL https://aclanthology.org/D19-1382/.

Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David AlfonsoHermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. MIRACL: A multilingual retrieval dataset covering 18 diverse languages. 11, 2023. URL https: //aclanthology.org/2023.tacl-1.63/. Place: Cambridge, MA Publisher: MIT Press.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025. URL https://arxiv.org/abs/2506.05176.

Haiquan Zhao, Chenhan Yuan, Fei Huang, Xiaomeng Hu, Yichang Zhang, An Yang, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin, Baosong Yang, Chen Cheng, Jialong Tang, Jiandong Jiang, Jianwei Zhang, Jijie Xu, Ming Yan, Minmin Sun, Pei Zhang, Pengjun Xie, Qiaoyu Tang, Qin Zhu, Rong Zhang, Shibin Wu, Shuo Zhang, Tao He, Tianyi Tang, Tingyu Xia, Wei Liao, Weizhou Shen, Wenbiao Yin, Wenmeng Zhou, Wenyuan Yu, Xiaobin Wang, Xiaodong Deng, Xiaodong Xu, Xinyu Zhang, Yang Liu, Yeqiu Li, Yi Zhang, Yong Jiang, Yu Wan, and Yuxin Zhou. Qwen3guard technical report, 2025a. URL https:

//arxiv.org/abs/2510.14276.

Xinping Zhao, Xinshuo Hu, Zifei Shan, Shouzheng Huang, Yao Zhou, Xin Zhang, Zetian Sun, Zhenyu Liu, Dongfang Li, Xinyuan Wei, Youcheng Pan, Yang Xiang, Meishan Zhang, Haofen Wang, Jun Yu, Baotian Hu, and Min Zhang. Kalm-embedding-v2: Superior training techniques and data inspire a versatile embedding model, 2025b. URL https:

//arxiv.org/abs/2506.20923.

Yuchen Zhuang, Aaron Trinh, Rushi Qiang, Haotian Sun, Chao Zhang, Hanjun Dai, and Bo Dai. Towards better instruction following retrieval models, 2025. URL https://arxiv. org/abs/2505.21439.

### A Base Model Architecture Details

This appendix summarizes the two causal language model families used throughout this work in Table 1: Gemma3 (Gemma et al., 2025) and Qwen3 (Yang et al., 2025a). Both families follow a decoder-only transformer design but differ in architectural choices such as attention patterns, normalization layers, vocabulary sizes, and pre-training configurations, providing evidence that our framework generalizes across diverse causal decoder architectures.

Table 1: Architectural comparison of base models used in this work.

Gemma3-270M Gemma3-1B Qwen3-0.6B Qwen3-1.7B

Architecture Parameters 268M 1001M 596M 1721M Layers 18 26 28 28 Hidden dimension 640 1152 1024 2048 Normalization RMSNorm (pre & post attention) RMSNorm (pre attention) Embedding tying No Yes Attention pattern 5:1 sliding / global Uniform global Head dimension 256 128 Attention heads 4 4 16 16 KV heads 1 1 8 8 Sliding-window span 512 tokens RoPE base θ 10k (local) / 1M (global) 1M

Pre-training Tokenizer SentencePiece Byte-level BPE Vocab size 262,144 151,936 Pre-training tokens 6 T 2 T 36 T Distillation Yes No

### B Adaptation Training Details

###### B.1 Loss definitions

###### 1. Masked Language Modeling (MLM). A subset of tokens is randomly masked, and the model is trained to reconstruct them using full bidirectional context:

LMLM(x) = − ∑

log pθ(xi | xM) , (1)

i∈M

where M ⊂ {1, . . . , T} denotes the masked positions and xM is the input sequence with masked tokens replaced by a special [MASK] placeholder. Masking is applied

independently with probability pmask ∈ {10%,20%,30%,40%}, which we evaluate in Appendix F.

###### 2. Masked Next-Token Prediction (MNTP). MNTP combines masked reconstruction with

the causal next-token prediction mechanism by predicting each masked token xi from the logits at position i − 1:

LMNTP(x) = − ∑

log pθ,i−1(xi | xM) . (2) All masking-related notation and hyperparameters follow the MLM setup.

i∈M

###### 3. Contrastive learning (InfoNCE). We employ a contrastive objective with both in-batch and hard negatives to align the representations of semantically equivalent sequences. For each anchor x and positive x+, the negatives N consist of the remaining in-batch samples, augmented with explicitly mined hard negatives:

esim(hx,hx+)/τ

esim(hx,hx+)/τ + ∑x−∈N esim(hx,hx−)/τ , (3)

LInfoNCE = − log

where hx = fθ(x) denotes the sequence representation, obtained either via last-token selection or mean pooling over the final layer hidden states, sim(·, ·) is the cosine similarity and τ is a temperature hyperparameter.

###### B.2 Hyperparameters

To ensure a strictly controlled and fair comparison, all training runs process identical data in the exact same order for a single epoch of unique tokens during masking, and unique sentence pairs during contrastive adaptation. Learning rates (LR) are chosen via grid search over 10 log-spaced values from 1 × 10−5 to 1 × 10−3, selecting the value that minimizes training loss on 1B tokens for masking and 1M samples for contrastive training. All experiments use a fixed seed (42) for reproducibility.

Table 2: Masked adaptation hyperparameters.

Gemma3-270M Qwen3-0.6B Gemma3-1B Qwen3-1.7B

- Learning rate 5 × 10−4 1 × 10−4 7 × 10−5 5 × 10−5 Warmup steps 0.01 epoch LR scheduler Warmup Stable Decay (WSD) (Hu et al., 2024) Optimizer AdamW (fused) Max grad norm 1.0 Seed 42

Batch size (per GPU) 1 Gradient accumulation 3 Num GPUs 96 (12 nodes × 8 GPUs) Sequence length 8192 Effective batch size 2,359,296

Adaptation objective Masked Next Token Prediction (MNTP) Loss function Linear Cross Entropy (fused)

Table 3: Contrastive training hyperparameters.

Gemma3-270M Qwen3-0.6B Gemma3-1B Qwen3-1.7B

- Learning rate 6 × 10−5 3 × 10−5 1 × 10−4 1 × 10−4 Warmup steps 500 LR scheduler Linear Optimizer AdamW (fused) Max grad norm 1.0 Seed 42

Batch size (per GPU) 128 Num GPUs 4 Effective batch size 512 Mini-batch size 64 32 32 16

Loss function Cached Multiple Negatives Ranking Loss Temperature τ 0.05 In-batch negatives Yes Hard negatives 1–7 (0 for multimodal samples)

Max sequence length 512

### C Adaptation Data Details Masking:

- • FineWeb-Edu (Penedo et al., 2024) consists of 1.3T English tokens from educational web pages filtered from the FineWeb dataset.9
- • FineWeb2-HQ (Messmer et al., 2026) is a high-quality, model-filtered pretraining dataset derived as a subset of FineWeb2, spanning 20 languages. It was created by selecting the

9https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

top 10% of documents in each language based on scores from a deep learning classifier trained to identify structured, knowledge-rich samples.10

- • FineMath (Liu et al., 2024a) comprises 54B tokens of mathematical content filtered from CommonCrawl to retain only the most educational material, focusing on clear explanations and step-by-step problem-solving.11
- • The Stack v2 (Lozhkov et al., 2024) contains over 3B files across 600+ programming and markup languages.12

###### Contrastive:

- • Embed-Nemotron-Dataset-V1 (Babakhin et al., 2025) is a curated subset of 11 datasets used for training general-purpose text embedding models.13
- • KaLM-embedding-finetuning-data (Hu et al., 2025) is a diverse collection of 62 datasets spanning retrieval, semantic textual similarity, and classification tasks.14
- • Parallel Data spans 51 English-centric language pairs (3054406 pairs after subsampling; see Table 4), drawn from 4 datasets; OPUS-10015, JW30016, TED Talks17, and WikiMatrix18.
- • FollowIR (Weller et al., 2024) is an instruction-aware retrieval training set, used here excluding Core17/News21/Robust04 evaluation topics.19
- • InF-IR (Zhuang et al., 2025) provides instruction-following information retrieval training data, used here excluding Robust04 evaluation topics.20

Instruction-aware training with single-domain batching. For asymmetric tasks (retrieval, reranking), instruction prefixes are prepended to queries only; for symmetric tasks (STS, pair classification), the same instruction is applied to both anchors and positives. Each training batch is drawn from a single dataset so that all in-batch negatives share the query’s exact task structure and domain. Furthermore, each query is paired with mined hard negatives ranging from 1 to 7 ensuring every sample retains at least one hard negative.

MTEB decontamination. To ensure fair zero-shot evaluation on the MTEB benchmark (Muennighoff et al., 2023), we exclude training domains that overlap with MTEB evaluation tasks. This removes 13 domain families (including both KaLM and Nemotron versions): ArXiv QA, MASSIVE (classification and clustering), CQADupstack, TREC-COVID, DBPedia, FEVER, FiQA, HotpotQA, PAWS-X, Quora, SciFact, and SNLI. The fully decontaminated corpus totals 10110219 training pairs (Table 4).

Dataset deduplication. When merging the NeMo and KaLM sources, we adopt a NeMofirst deduplication policy. Specifically, KaLM datasets that overlap with NeMo Retriever families (e.g., MIRACL, MS MARCO, TriviaQA, SQuAD, NFCorpus, GooAQ, and PAQ) are dropped in favor of their Nemotron counterparts, which provide higher-quality hard negatives.

- 10https://huggingface.co/datasets/epfml/FineWeb2-HQ
- 11https://huggingface.co/datasets/HuggingFaceTB/finemath
- 12https://huggingface.co/datasets/bigcode/the-stack-v2
- 13https://huggingface.co/datasets/nvidia/Embed-Nemotron-Dataset-V1
- 14https://huggingface.co/datasets/HIT-TMG/KaLM-embedding-finetuning-data
- 15https://huggingface.co/datasets/Helsinki-NLP/opus-100
- 16https://huggingface.co/datasets/sentence-transformers/parallel-sentences-jw300
- 17https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks
- 18https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix
- 19https://huggingface.co/datasets/jhu-clsp/FollowIR-train
- 20https://huggingface.co/datasets/InF-IR/InF-IR

- Table 4: Training dataset composition after domain decontamination (10,110,219 training pairs).

Dataset Pairs Dataset Pairs KALM

mMARCO (zh) 379,870 NLLB 26,504 SimCLUE 290,699 ESCI 26,043 Multi-CPR 234,587 Aya Dataset 22,449 SimCSE NLI 217,099 Yahoo Answers 21,724 T2Ranking 188,606 CSL 19,945 nli zh 185,787 LCSTS 19,535 llm retr. short long 149,511 THUCNews 19,288 llm sts monolingual 132,561 WebGPT Comparisons 18,924 CMNLI 119,029 ChatMed-Dataset 18,608 llm retr. long long 114,979 AdvertiseGen 17,526 llm retr. long short 114,584 OCNLI 11,937 DuReader checklist 97,764 ATEC 11,387 cMedQA-V2.0 88,109 BQ 10,000 PubMedQA 79,954 SearchQA 9,988 DuReader 79,229 CMRC 2018 9,753 ELI5 76,408 rag-dataset-12000 9,272 llm retr. short short 76,315 lawzhidao 6,784 llm sts bitext retr. 75,271 webqa 4,988 XNLI (zh) 74,252 CHEF 4,824 MEDI2BGE 71,790 cCOVID-News 4,727 MultiNLI 63,701 DRCD 4,714 Natural Questions 56,377 AFQMC 3,876 RefGPT 49,896 CINLID 2,883 CodeFeedback 49,090 UMETRIP-QA 2,537 WikiAnswers 47,686 ChineseSTS 2,497 QBQTC 47,223 LIMA 1,991 Mr. TyDi 46,997 WebCPM 1,602 OpenOrca 38,623 ExpertQA 1,252 retrieval data llm 32,551 CAIL2019-SCM 648 MLDR 31,097 ContractNLI 628 CC-News 28,246 law-gpt 500

KaLM subtotal 3655225 NEMOTRON OTHER

SyntheticClassif. 1,044,212 Parallel Data (51 lang. pairs) 3,054,406 PAQ 1,000,000 OPUS-100 946,599 MS MARCO 532,751 JW300 701,201 MAmmoTH2 317,180 TED Talks 733,318 NaturalQuestions 100,231 WikiMatrix 673,288 GooAQ 100,000 InF-IR 48,403 SQuAD 87,599 MS MARCO 38,759 MIRACL 79,648 metamath 7,104 TriviaQA 73,346 leetcode 2,540 EmotionClassif. 13,039 FollowIR 494 NFCorpus 3,685

Nemotron subtotal 3351691 Other subtotal 3103303 Total 10110219

### D Details of Evaluation

This appendix offers additional details on the datasets used for evaluation, they are organized into two sections: downstream task evaluation, where the model is fine-tuned on task-specific data, and zero-shot evaluation, where the model’s frozen embeddings are evaluated directly (with at most a lightweight linear probe).

- D.1 Downstream Task Evaluation Sequence classification datasets (F1 score):

- • XNLI (Conneau et al., 2018-10) – General: This natural language inference task extends MNLI (Williams et al., 2018) to non-English languages, involving the classification of sentence pairs into entailment, contradiction, or neutral.21
- • PAWS-X (Yang et al., 2019) – General: This dataset contains 23,659 human-translated Paraphrase Adversaries from Word Scrambling (PAWS) evaluation pairs across six distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. The task aims to determine whether two sentences convey the exact same meaning.
- • MathShepherd (Wang et al., 2024c) – Math: This is a binary classification task aimed at determining whether a step-by-step math rationale is correct given a problem prompt.
- • CodeComplexity (Jeon et al., 2023) – Code: This computational analysis task involves estimating the order of complexity for a code-formulated computer science problem.

###### Retrieval datasets (NDCG@10):

- • MS MARCO (Bajaj et al., 2016) – General: This English-only retrieval dataset is used for fine-tuning. Each anchor–positive pair is augmented with a mined hard negative to form a triplet structure. We use the hard-triplet version of MS MARCO.22
- • MIRACL (Zhang et al., 2023) – General: For this multilingual retrieval dataset, we use the semi-supervised SentenceTransformers version as the primary data source.23 Anchors serve as queries, and the corpus consists of all positive documents in the dataset. Since only a single data split is available, we create validation and test sets by partitioning 50% of the original split for each, using queries as the split key to ensure no data leakage.
- • CodeSearchNet (Husain et al., 2019) – Code: This code retrieval dataset features comment–code query–positive pairs (SentenceTransformers version) and is processed similarly to the previous datasets.24

###### Sequence regression datasets (Spearman correlation):

• SeaHorse (Clark et al., 2023-12) – Summary: This multilingual summarization evaluation task annotates each text–summary pair across six binary dimensions. The final score is obtained by averaging these labels, yielding a continuous value between 0 and 1. To avoid penalizing models with limited context lengths, the summary is placed first in the input, followed by the main text, ensuring the model can attend to the full summary.25

###### Token classification datasets (F1 score):

- • XTREME PAN-X (Hu et al., 2020) – NER: Named entity recognition task which is a balanced subset of the WikiAnn dataset (Pan et al., 2017). Named entities in Wikipedia were automatically annotated with LOC, PER, and ORG tags in IOB2 format using a combination of knowledge base properties, cross-lingual and anchor links, self-training, and data selection.26
- • XTREME POS (Nivre et al., 2020) – POS: Cross-lingual structured prediction task requires assigning a grammatical category (noun, verb, adjective, etc.) to each token in a sentence. It uses data from Universal Dependencies v2.5, and models are evaluated under a zero-shot transfer setting: fine-tuned on English labeled data and directly applied to other languages without retraining.

- 21https://huggingface.co/datasets/mteb/xnli
- 22https://huggingface.co/datasets/bclavie/msmarco-10m-triplets
- 23https://huggingface.co/datasets/sentence-transformers/miracl
- 24https://huggingface.co/datasets/sentence-transformers/codesearchnet
- 25https://huggingface.co/datasets/hgissbkh/seahorse
- 26https://huggingface.co/datasets/google/xtreme

XTREME Augmented benchmark (Average score): We create an augmented XTREME benchmark (Hu et al., 2020) by retaining its original tasks (excluding question answering) and incorporating our CodeComplexity and MathShepherd datasets to cover a broader range of domains.

###### Model specialisation via causal ecosystem (F1 score):

- • Beaver (Ji et al., 2023) – Safety: This classification task evaluates LLM outputs across 14 harm categories to measure content moderation capabilities.27
- • Safe (Ji et al., 2025) – Safety: This binary classification task is derived from RLHF preference data, where each model response is annotated as safe or unsafe.28
- • Aegis (Ghosh et al., 2025) – Safety: This AI content classification task spans 13 risk categories and is designed to evaluate safety filtering in generative AI systems.29
- • E-SNLI-VE (Do et al., 2021) – Image-Text English: This visual entailment task extends E-SNLI to image–text pairs, involving the classification of whether an image entails, contradicts, or is neutral with respect to a textual hypothesis.30
- • BoolQ-Audio – Audio-Text English: This audio-based Boolean question-answering task classifies spoken questions paired with a text passage as yes or no.31

###### D.2 General Embeddings Evaluation

The following benchmarks evaluate general-purpose embeddings without fine-tuning the model on task-specific data. Depending on the task type, evaluation is either fully zero-shot (e.g., cosine similarity for retrieval) or uses a lightweight linear probe (e.g., logistic regression for classification). All three benchmarks are part of the MTEB ecosystem, enabling us to efficiently compare our models against thousands of baselines across a large set of tasks.32

- • MTEB (English, v2) (Muennighoff et al., 2023): A comprehensive English text embedding benchmark derived from MTEB (English, v1). It spans seven task categories: classification, clustering, pair classification, reranking, retrieval, semantic textual similarity, and summarization, comprising a total of 41 tasks.
- • MTEB (Multilingual, v2) (Enevoldsen et al., 2025) A large-scale multilingual text embedding benchmark covering 250+ languages, curated from the full MMTEB collection (Muennighoff et al., 2023) via inter-task correlation-based downsampling to reduce computational cost while preserving model rankings. Tasks span eight categories: classification, clustering, pair classification, reranking, retrieval, semantic textual similarity, bitext mining, and summarization.
- • MIEB (lite) (Xiao et al., 2025) A lightweight image embedding benchmark covering 51 tasks across 10 task types, designed as a cost-efficient version of MIEB(Multilingual) while maintaining relative model rankings. Task types include clustering, few-shot linear probing, zero-shot classification, retrieval (image-to-image, text-to-image, and cross-modal), document understanding, visual STS, compositionality, and interleaved embedding evaluation.
- • MAEB (beta) (Assadi et al., 2026) An audio embedding benchmark with 30 tasks spanning audio-only and audio-text cross-modal evaluation in 100+ languages, derived from a larger 98-task collection (MAEB+). Tasks span seven types: classification (10), retrieval (9), clustering (3), pair classification (3), multilabel classification (2), zero-shot classification (2), and reranking (1).

- 27https://huggingface.co/datasets/PKU-Alignment/BeaverTails
- 28https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
- 29https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- 30https://huggingface.co/datasets/sedrickkeh/e-snli-ve
- 31https://huggingface.co/datasets/fixie-ai/boolq-audio
- 32https://github.com/embeddings-benchmark/mteb

###### D.3 Aggregating Performance Across Tasks

To enable a fair comparison across tasks with heterogeneous metrics and scales, we report an average normalized rank. For each task t ∈ T and model m ∈ M, let vt,m denote the aggregate performance score. We rescale every model to a [0, |M| − 1] interval via

maxm′ vt,m′ − vt,m maxm′ vt,m′ − minm′ vt,m′

rt,m = (|M| − 1) ·

, (4)

so that rt,m = 0 for the best-performing model on task t and rt,m = |M| − 1 for the worst. The overall rank of a model is then the arithmetic mean across all tasks:

1

|T | ∑

r¯m =

rt,m . (5)

t∈T

A lower r¯m therefore indicates a model that performs consistently well across all evaluation tasks, regardless of the individual metric used in each one.

###### D.4 Evaluation Fine-Tuning Protocol

Text fine-tuning protocol. All models are fine-tuned under a consistent protocol with a batch size of 32. For each model–dataset pair, we select the learning rate from 10 log-spaced values between 5 × 10−6 and 5 × 10−3, using a 10% warmup schedule followed by linear decay. To avoid data contamination during model selection and evaluation, we rely on existing training, validation, and test splits, or manually create them when unavailable. To accommodate architectural differences, we follow standard practice by using the final-token representation for causal models and mean pooling for bidirectional models on retrieval, sequence classification, and regression tasks.

- • Retrieval: Fine-tuning runs for 1k steps on MS MARCO, followed by zero-shot crossdomain retrieval on the remaining benchmarks.
- • Sequence regression: Fine-tuning runs for 5k steps.
- • Sequence and token classification: Fine-tuning runs for 10k steps.

For smaller datasets, which undergo multiple training epochs, we apply early stopping with a patience of one epoch based on validation performance during each fine-tuning run.

Post-merging specialisation fine-tuning. To evaluate the effectiveness of merging our adapted model with causal specialists, we assessed performance across several domainspecific tasks. We followed the previously evaluation setup with one exception: given the limited number of samples in these specialized benchmarks and to ensure perfectly balanced label distributions within each training set, we utilized a smaller batch size to guarantee a minimum of 500 training steps per task.

- • Beaver: Batch size: 4.
- • e-SNLI-VE: Batch size: 32.
- • BoolQ-Audio: Batch size: 14.

To accommodate the limited number of samples available in these benchmarks, and to highlight the accelerated convergence of the merged model compared to baseline, we reduce the batch size relative to the previous text benchmarks. This adjustment ensures a minimum of 500 training steps per benchmark.

### E BidirLM-Omni Model Composition

Qwen3-VL-2B-Instruct (Vision)

Qwen-1.7B Bi+MNTP (Text)

Qwen3-ASR-1.7B (Audio)

VL Head

ASR Head

Backbone

Backbone

Backbone

Linear Merge (1/3 ea.)

VL Head ASR Head

Merged Textual Backbone

BidirLM-Omni-2.5B

Legend:

Merging Process Component Transfer

Identical Architectures (Diff. Weights) Frozen Head (Retained)

- Figure 10: The construction of BidirLM-Omni-2.5B relies on a modular composition strategy. We begin with three specialized variants sharing an identical underlying architecture: a vision model (Qwen3-VL-2B), our bidirectional text encoder (Qwen-1.7B Bi+MNTP), and an audio model (Qwen3-ASR-1.7B). First, we isolate their trainable textual backbones and perform a linear weight merge in equal proportions (1/3 each) to forge a unified omnimodal representation space. Second, we extract the frozen, modality-specific projection heads (visual and audio) from the specialist models and seamlessly append them to the newly merged backbone. This composition enables cross-modal routing.

### F Additional Results

###### F.1 Masked Language Objectives and Hyperparameters

We evaluate the MLM and MNTP objectives for bidirectional adaptation across four masking ratios (10%, 20%, 30%, and 40%) using a 10B-token subset of FineWeb-Edu. Figure 11 reports the downstream performance of the resulting Bi+MLM and Bi+MNTP models, along with their ranking based on average normalized scores.33

MNTP outperforms MLM for model adaptation. As shown in Figure 11, Bi+MNTP consistently outperforms Bi+MLM across tasks and architectures, except on the Seahorse dataset for Gemma at 20% and 30%. More generally, Bi+MNTP achieves higher mean performance than Bi+MLM at every corresponding masking ratio. Furthermore, all Bi+MNTP models with masking ratios above 20% surpass the highest average performance of any Bi+MLM variant, establishing Bi+MNTP as the stronger of the two masking objectives.

Optimal masking ratios are objective- and model-dependent. Bi+MLM performance typically peaks at intermediate ratios (20% and 30%, Figure 11), whereas Bi+MNTP benefits

33To manage the extensive search space over masking ratios and learning rates, we limit XNLI and PAN-X fine-tuning to 5k steps for this comparison.

Miracl (IR)

###### XNLI (SC)

Seahorse (SR)

Xtreme Panx (TC)

Rank

nDCG@10

Accuracy

Spearman Correlation

F1 Score

Average Normalized

- 94

- 95

- 96

79

55

93

0 2 4

Gemma3

77

50

91

75

45

89

- 95

- 96

- 97

87

57

94

0 2 4

Qwen3

85

50

92

83

43

90

10 20 30 40

10 20 30 40

10 20 30 40

10 20 30 40

10 20 30 40

Mask Ratio (%)

Mask Ratio (%)

Mask Ratio (%)

Mask Ratio (%)

Mask Ratio (%)

Bi+MLM Bi+MNTP

- Figure 11: Performance comparison of MLM vs. MNTP adaptation. The first four columns report dataset-specific scores, while the rightmost column reports the model ranking based on average normalized performance across all tasks.

from higher masking, achieving optimal average performance at 30% for Qwen3-0.6B and 40% for Gemma3-270M.

###### F.2 Details on Model Similarities

As discussed in §4, the success of our weight-merging strategy relies on the observation that the adapted and causal models remain close in weight space. Here, we ground this observation in prior theoretical literature and extend our analysis to a layer-by-layer level across the various merging configurations explored in this study, providing finer-grained evidence that weight displacement remains bounded and consistent.

Empirical context for merging. Prior work has shown that models fine-tuned from a shared pretrained checkpoint often remain in the same basin of the loss landscape, a property known as linear mode connectivity (Frankle et al., 2020), enabling their convex combinations to perform well (Wortsman et al., 2022a). A complementary observation by Ortiz-Jimenez et al. (2023) suggests that pretraining induces weight disentanglement, whereby distinct capabilities are encoded along approximately orthogonal directions in weight space, reducing interference when models are combined. While these results were established for models sharing the same objective and attention mechanism, we note that our setting shares a key favorable condition: all merged models derive from the identical pretrained backbone. Furthermore, our adaptation processes a small fraction of tokens relative to the original pre-training scale while maintaining next-token prediction objectives, resulting in remarkably limited weight displacement (mean cosine similarity of 0.78 for Gemma and 0.97 for Qwen).

Methodology. For each model pair, we compute the layer-wise cosine similarity between corresponding weight tensors. To achieve this, we flatten and concatenate all Self-Attention and MLP parameters within a given layer into a single vector. Additionally, we disentangle these results into two specific weight groups: Self-Attention (Q, K, V, and O projections) and MLP (gate, up, and down projections).

MNTP adaptation. Figure 12 reports the per-layer cosine similarity between the original causal models and resulting Bi+MNTP encoder counterparts for Gemma3-270M and Qwen30.6B, quantifying the weight displacement introduced by our bidirectional adaptation. The two architectures exhibit different similarity profiles: Gemma3-270M shows substantially

Gemma3-270M vs. Bi+MNTP Gemma3-270M

Qwen3-0.6B vs. Bi+MNTP Qwen3-0.6B

Cosine Similarity

Cosine Similarity

1.00

1.00

0.85

0.97

0.70

0.95

E 2 4 6 8 10 12 14 16 18

E 2 4 6 8 10 12 14 16 18 20 22 24 26 28

1.00

1.00

0.85

0.95

0.70

0.90

E 2 4 6 8 10 12 14 16 18

E 2 4 6 8 10 12 14 16 18 20 22 24 26 28

Layer

Layer

Self-Attention MLP Embedding

- Figure 12: Per-layer cosine similarity between causal models and their Bi+MNTP-adapted encoders. Top: Aggregate cosine similarity per layer. Bottom: Comparison broken down by by Self-Attention and MLP weight group.

E 4 8 12 16 20 24 28

0.90

0.95

1.00

Cosine Similarity

Bi+MNTP Qwen3-0.6B vs. Qwen3-ASR-0.6B

E 4 8 12 16 20 24 28

0.95

0.97

1.00

Cosine Similarity

Bi+MNTP Qwen3-1.7B vs. Qwen3-ASR-1.7B

E 4 8 12 16 20 24 28

0.95

0.97

1.00

Cosine Similarity

Bi+MNTP Qwen3-1.7B vs. Qwen3-VL-2B-Instruct

E 4 8 12 16 20 24 28

Layer

0.85

- 0.93

- 1.00

E 4 8 12 16 20 24 28

Layer

0.85

0.93

1.00

E 4 8 12 16 20 24 28

Layer

0.85

0.93

1.00

Self-Attention MLP Embedding

- Figure 13: Per-layer cosine similarity between Bi+MNTP adapted Qwen3 models and their causal multimodal variants. Top: Aggregate cosine similarity per layer. Bottom: Comparison down by Self-Attention and MLP weight group.

lower overall similarity (mean cosine 0.78) compared to Qwen3-0.6B (mean cosine 0.97). When broken down by weight group, Self-Attention and MLP projections follow a similar trend, with MLP weights consistently exhibiting slightly larger deviations.

Multimodal variants. Figure 13 extends this analysis to the causal multimodal specialists that we used during the multimodal alignment ablation and to construct BidirLM-Omni-

- 2.5B (§6). This entails a comparison of our Bi+MNTP Qwen3-0.6B against Qwen3-ASR-0.6B (left plot), and our Bi+MNTP Qwen3-1.7B against both Qwen3-VL-2B-Instruct (which utilizes the same 1.7B text backbone) and Qwen3-ASR-1.7B. Examining these 1.7B variants (right two subplots), we observe that the similarity with the VL specialist remains slightly higher than with the ASR specialist (mean cosine 0.97 vs. 0.96). Furthermore, the aggregate similarity over Self-Attention and MLP projections generally decreases with depth for each model, indicating that later layers undergo the most modification. All causal models used to construct BidirLM-Omni-2.5B maintain a high overall similarity with the Bi+MNTP Qwen3-1.7B encoder (> 0.96).

###### F.3 Performance by Merging Ratio with Causal Specialists

Following the merging ratio analysis conducted in Figure 4 for catastrophic forgetting mitigation, we investigate the merging behavior of encoder specialization when leveraging causal specialists for domain (Figure 14) and multimodal (Figure 15) adaptation.

Beaver

Safe

Aegis

F1 Score

F1 Score

F1 Score

82

89

79

Qwen3

80

87

75

78

85

71

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

Merging Ratio (%)

Merging Ratio (%)

Merging Ratio (%)

- Figure 14: Model performance on safety classification benchmarks across merging ratios. We report the resulting scores as a function of the weight allocated to the causal Qwen3Guard-Gen-0.6B model when merged with our Bi+MNTP Qwen3-0.6B encoder.

0 25 50 75 100

Merging Ratio (%)

41

61

81

Qwen3

F1 Score

E-SNLI-VE (Visual)

0 25 50 75 100

Merging Ratio (%)

57

67

77

F1 Score

BoolQ-Audio (Audio)

- Figure 15: Model performance on multimodal classification benchmarks across merging ratios. We report the resulting scores as a function of the weight allocated to the causal Qwen3-ASR-0.6B and Qwen3-VL-2B-Instruct models when merged with our Bi+MNTP Qwen3-0.6B and Qwen3-1.7B encoders.

An equal merging ratio emerges as a robust baseline. Consistent with our strategy for mitigating catastrophic forgetting, an equal 50% split consistently yields the highest performance by efficiently weighing the encoder’s bidirectional capabilities against the specialized knowledge of the causal models. Therefore, we advise practitioners to adopt a 0.5 interpolation weight as a strong default, exploring nearby values to extract peak performance given that the merging process is computationally training-free.

Merging with non-shared modalities is ratio-sensitive. While merging models within a common modality such as text yields robust performance even at intermediate ratios like

25% or 75%, merging across distinct modalities results in significant performance drops at these same unbalanced values. This indicates that as the discrepancy between the base models’ domains or modalities increases, overall performance becomes highly sensitive to the interpolation weight, reinforcing a balanced 50% split as the most reliable default choice.

###### F.4 Effect of Merging on Omnimodal Performance

MTEB (Multilingual V2)

MIEB

MAEB

Average Score

Average Score

Average Score

64

54

46

62

52

44

60

50

42

BidirLM-Omni-2.5B (merged) BidirLM-Omni-2.5B (non-merged)

Figure 16: Average score per benchmark: BidirLM-Omni-2.5B (merged) vs. BidirLMOmni-2.5B (non-merged). Average score across individual tasks. Comparison following the contrastive training phase between the merged BidirLM-Omni variant and the non-merged baseline on: MTEB (Multilingual V2), MIEB (lite), and MAEB (beta).

1

To evaluate the impact of our merging strategy when creating BidirLM-Omni, we compare two variants: BidirLM-Omni (where each of the three specialized causal backbones contributes equally), and a non-merged baseline relying exclusively on the Bi+MNTP weights without integrating the common weights shared with the ASR and VL specialists (only concatenating their frozen multimodal heads on top). As shown in Figure 16, merging appears as a key performance factor, with higher scores achieved by the merged variant across all three benchmarks.

###### F.5 Detailed Results Across Models and Benchmarks

- Table 5 reports per-task-type scores on MTEB (Multilingual V2) for our four text-only BidirLM encoders alongside the omnimodal BidirLM-Omni-2.5B. Table 6 and Table 7 further detail BidirLM-Omni-2.5B performance across MIEB (lite) and MAEB (beta) benchmarks.

Mean (TaskType) BidirLM-270M 59.5 58.8 45.9 -2.1 19.7 76.7 56.8 48.7 67.7 55.5 48.0

Instr. Rerank.

ML Class.

Pair Class. Rerank. Retr. STS

Mean (Task)

Bitext Mining Class. Clust.

MTEB (Multilingual V2)

- BidirLM-0.6B 66.7 62.2 49.6 0.8 23.8 78.4 58.1 56.5 70.8 59.6 51.9 BidirLM-1B 71.8 65.9 50.3 -0.4 25.1 79.8 58.6 56.5 74.6 62.1 53.6
- BidirLM-1.7B 72.2 65.7 51.5 0.5 26.6 80.2 62.1 59.9 74.2 62.9 54.8 BidirLM-Omni-2.5B 72.2 65.5 51.6 0.9 26.6 80.7 63.4 59.4 75.7 63.1 55.1

- Table 5: Performance per task type on MTEB (Multilingual V2). Best score per column is bolded. Class.: classification, Clust.: clustering, Instr. Rerank.: instruction reranking, ML Class.: multilabel classification, Pair Class.: pair classification, Rerank.: reranking, Retr.: retrieval.

MIEB (lite) Retr. Comp.

Doc. Underst.

Img. Class.

Img. Clust.

Vision QA

ZS Class.

Mean (Task)

Mean (TaskType) BidirLM-Omni-2.5B 28.8 46.0 76.9 73.6 61.1 48.6 48.1 58.1 54.7

- Table 6: Performance per task type on MIEB (lite) for BidirLM-Omni-2.5B. Comp.: compositionality, Vision QA: vision-centric QA, ZS Class.: zero-shot classification.

Any2Any Retr.

Mean (TaskType) BidirLM-Omni-2.5B 32.8 58.2 5.5 32.3 66.4 74.8 55.3 45.2 46.5

Audio Class.

Audio Clust.

Audio ML Class.

Audio Pair Class.

Audio Rerank.

Audio ZS Class.

Mean (Task)

MAEB (beta)

- Table 7: Performance per task type on MAEB (beta) for BidirLM-Omni-2.5B. Any2Any Retr.: any-to-any retrieval, Audio ML Class.: audio multilabel classification, Audio ZS Class.: audio zero-shot classification.

F.6 MTEB, MIEB, and MAEB (2026-03-30 Snapshot).

MTEB (Multilingual V2) Params

Mean Rank

Zero Shot

Bitext Mining Class. Clust.

Instr. Rerank.

ML Class.

Pair Class. Rerank. Retr. STS

Mean (TaskType)

Mean (Task)

KaLM-Gemma3-12B 11.8B 1 73% 83.8 77.9 55.8 5.5 33.0 84.7 67.3 75.7 79.0 62.5 72.3

· · ·

BidirLM-Omni-2.5B 2.5B 17 100% 72.2 65.5 51.6 0.9 26.6 80.7 63.4 59.4 75.7 55.1 63.1 BidirLM-1.7B 1.7B 18 100% 72.2 65.7 51.5 0.5 26.6 80.2 62.1 59.9 74.2 54.8 62.9

· · ·

GTE-Qwen2-7B 7.1B 22 – 73.9 61.5 52.8 4.9 25.5 85.1 65.5 60.1 74.0 55.9 62.5

· · ·

BidirLM-1B 1.0B 24 100% 71.8 65.9 50.3 -0.4 25.1 79.8 58.6 56.5 74.6 53.6 62.1

· · ·

BidirLM-0.6B 596M 36 100% 66.7 62.2 49.6 0.8 23.8 78.4 58.1 56.5 70.8 51.9 59.6

· · ·

BGE-M3 568M 38 98% 79.1 60.4 40.9 -3.1 20.1 80.8 62.8 54.6 74.1 52.2 59.6

· · ·

GTE-Qwen2-1.5B 1.5B 40 – 62.5 58.3 52.0 0.7 24.0 81.6 62.6 60.8 71.6 52.7 59.5

· · ·

KaLM-mini 494M 49 92% 64.8 57.6 45.6 -1.5 20.7 77.7 60.6 54.2 70.8 50.0 57.0

· · ·

Stella-1.5B 1.5B 54 90% 58.6 56.7 49.7 0.2 21.8 78.5 61.4 52.8 69.9 50.0 56.5

· · ·

BidirLM-270M 268M 58 100% 59.5 58.8 45.9 -2.1 19.7 76.7 56.8 48.7 67.7 48.0 55.5

· · ·

Nomic-v1 137M 87 96% 28.4 49.4 40.8 -3.2 17.8 71.7 46.0 37.0 64.2 39.1 45.7

- Table 8: Per-task-type performance on MTEB (Multilingual V2) for our models (bold) and open-data baselines, ranked by Mean (Task). · · · denotes a jump in leaderboard entries. Zero-shot ratios from the MTEB leaderboard. Best per column in bold.

MIEB (lite) Params

Mean Rank

Zero Shot Retr. Comp.

Doc. Underst.

Img. Class.

Img. Clust.

Vision QA

ZS Class.

Mean (TaskType)

Mean (Task)

BidirLM-Omni-2.5B 2.5B 2 100% 28.8 46.0 76.9 73.6 61.1 48.6 48.1 54.7 58.1

· · ·

E5-V 8.4B 6 100% 26.9 39.4 56.0 70.6 51.7 52.6 36.2 47.6 51.9

· · ·

Nemotron-Omni-3B 4.7B 8 – 31.8 41.0 46.6 68.3 75.6 47.9 43.7 50.7 51.4 CLIP-bigG 2.5B 9 100% 34.2 35.0 35.5 77.8 80.8 43.0 72.4 54.1 51.3

· · ·

EVA02-bigE+ 5.0B 11 100% 35.2 38.9 26.2 80.0 87.3 38.8 74.0 54.4 50.8

· · ·

CLIP-L-DC 428M 13 100% 31.0 31.6 30.8 75.3 80.4 54.9 69.4 53.4 50.4 CLIP-H 986M 14 100% 32.8 34.8 33.7 76.8 79.3 46.8 69.4 53.4 50.0

· · ·

CLIP-B16-DC 150M 22 100% 28.3 31.4 22.7 73.2 73.6 56.9 61.9 49.7 46.6

· · ·

VLM2Vec 4.1B 24 100% 20.9 30.2 42.8 64.8 65.4 65.3 32.1 45.9 46.0

- Table 9: Per-task-type performance on MIEB (lite) for our models (bold) and open-data baselines, ranked by Mean (Task). · · · denotes a jump in leaderboard entries.

MAEB (beta) Params

Mean Rank

Zero Shot

Any2Any Retr.

Audio Class.

Audio Clust.

Audio ML Class.

Audio Pair Class.

Audio Rerank.

Audio ZS Class.

Mean (TaskType)

Mean (Task)

LCO-Omni-7B 8.9B 1 – 55.2 57.1 1.7 38.4 67.3 78.7 64.5 51.9 52.0 LCO-Omni-3B 4.7B 2 – 54.5 55.4 1.3 36.8 66.7 75.4 62.2 50.3 50.7 BidirLM-Omni-2.5B 2.5B 3 100% 32.8 58.2 5.5 32.3 66.4 74.8 55.3 46.5 45.2

· · ·

Nemotron-Omni-3B 4.7B 5 – 38.8 47.6 1.2 21.1 66.1 81.5 69.8 46.6 43.1

· · ·

MS-CLAP-23 160M 9 100% 18.7 45.0 15.2 7.0 53.6 75.4 12.6 32.5 31.3 CLAP-fused 154M 10 100% 17.6 44.4 22.7 4.5 52.0 61.3 13.2 30.8 30.8 CLAP-unfused 153M 11 100% 18.1 45.2 12.6 4.3 52.6 66.5 11.3 30.1 30.3 MS-CLAP-22 196M 12 100% 21.7 38.2 19.9 7.2 51.7 62.9 12.1 30.5 29.8

· · ·

SpeechT5 298M 15 100% 7.9 40.7 1.1 8.5 57.9 56.5 15.9 26.9 25.3

- Table 10: Per-task-type performance on MAEB (beta) for our models (bold) and open-data baselines, ranked by Mean (Task). · · · denotes a jump in leaderboard entries.

