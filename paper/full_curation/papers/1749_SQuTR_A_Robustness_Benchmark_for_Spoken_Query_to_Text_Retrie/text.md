# arXiv:2602.12783v2[cs.IR]6May2026

## SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise

Yuejie Li∗

Ke Yang∗

Yueying Hua∗

Huazhong University of Science and Technology Wuhan, China li_yuejie@alumni.hust.edu.cn

The University of Hong Kong Hong Kong, China yangkeai@connect.hku.hk

Soochow University Suzhou, China yyhua1224@outlook.com

Berlin Chen∗

University of Science and Technology of China Hefei, China berlin@mail.ustc.edu.cn

Jianhao Nie

Wuhan University Wuhan, China 2017282110385@whu.edu.cn

Yueping He

Tsinghua University Beijing, China heyp21@mails.tsinghua.edu.cn

Caixin Kang†

The University of Tokyo Tokyo, Japan cxkang@iis.u-tokyo.ac.jp

### Abstract

Spoken query retrieval is an important interaction mode in modern information retrieval. However, existing evaluation datasets are often limited to simple queries under constrained noise conditions, making them inadequate for assessing the robustness of spoken query retrieval systems under complex acoustic perturbations. To address this limitation, we present SQuTR, a robustness benchmark for spoken query retrieval that includes a large-scale dataset and a unified evaluation protocol. SQuTR aggregates 37,317 unique queries from six commonly used English and Chinese text retrieval datasets, spanning multiple domains and diverse query types. We synthesize speech using voice profiles from 200 real speakers and mix 17 categories of real-world environmental noise under controlled SNR levels, enabling reproducible robustness evaluation from quiet to highly noisy conditions. Under the unified protocol, we conduct large-scale evaluations on representative cascaded and end-to-end retrieval systems. Experimental results show that retrieval performance decreases as noise increases, with substantially different drops across systems. Even large-scale retrieval models struggle under extreme noise, indicating that robustness remains a critical bottleneck. Overall, SQuTR provides a reproducible testbed for benchmarking and diagnostic analysis, and facilitates future research on robustness in spoken query to text retrieval.

### CCS Concepts

• Information systems → Information retrieval.

### Keywords

Spoken Query Retrieval, Noise Robustness, Information Retrieval Benchmark

∗Both authors contributed equally to this research. †Corresponding Author

### 1 Introduction

Speech has become an increasingly important interaction modality for information retrieval systems [15, 31]. From voice assistants in smart home devices to in-vehicle infotainment systems [34], users are turning to spoken queries as a primary means of accessing information. Unlike clear text input, spoken queries in real-world scenarios are often affected by background noise, environmental interference, and speaker variability. These factors degrade Automatic Speech Recognition (ASR) performance, and the resulting transcription errors propagate to the retriever, often causing substantial drops in retrieval effectiveness.

Although prior studies [18, 27, 28] have noted that ASR errors can negatively affect downstream tasks, existing evaluation practices remain highly fragmented. Speech-related work [21, 25, 37] primarily measures ASR robustness under noise using transcription-level metrics such as word error rate (WER), treating ASR as an isolated component. In contrast, information retrieval (IR) research [3, 29] evaluates models under the assumption that queries are clean and unambiguous text. As a result, most IR benchmarks [19, 29] are constructed exclusively with text queries, while speech benchmarks [5, 20, 23] focus primarily on ASR performance rather than downstream retrieval performance. This mismatch makes it hard to compare spoken query retrieval systems under a unified, controlled protocol, and it limits our understanding of system robustness under diverse acoustic conditions.

Recenteffortshavestartedto incorporate speech into benchmarkdriven evaluation. MSEB [11] has introduced a retrieval-related subset with simple voice questions (SVQ) recorded under multiple environments. However, SVQ exhibits several limitations for robustness evaluation. First, its queries are primarily single-hop and fact-oriented, with relatively standardized question forms. Second, the associated corpora largely consist of short passages in general domains (e.g., Wikipedia), limiting contextual and task complexity.

Third, although recordings are collected under several environmental conditions, noise intensity is not explicitly controlled across graded signal-to-noise ratio (SNR) levels.

To address these limitations, we introduce SQuTR, a benchmark for spoken query to text retrieval under controlled acoustic noise. SQuTR provides (i) a large-scale spoken-query dataset derived from text-based IR benchmarks and (ii) a unified evaluation protocol that measures retrieval performance under multiple acoustic conditions. SQuTR constructs spoken queries from real queries in six widely used English and Chinese retrieval benchmarks: FiQA-2018 (finance) [7], HotpotQA (multi-hop QA) [41], Natural Questions (open-domain QA) [13], MedicalRetrieval (medical) [17], DuRetrieval (general-domain retrieval) [22], and T2Retrieval (passage retrieval) [39]. This design preserves realistic query complexity and broad task diversity beyond simple question templates.

We synthesize speech using CosyVoice-3 [8] with 200 diverse speakers. To simulate diverse acoustic environments, we inject real-world environmental noise sampled from over a dozen realworld categories, and construct four graded acoustic conditions (Clean, Low Noise, Medium Noise, and High Noise) by controlling the signal-to-noise ratio (SNR). This design enables controlled and reproducible analysis of retrieval robustness under graded acoustic perturbations.

Our experiments systematically evaluate retrieval performance across different acoustic conditions, multiple ASR front-ends, and both lexical and dense retrievers. We observe a consistent decline in retrieval effectiveness as acoustic noise increases, and this degradation pattern remains stable across ASR models of varying sizes and architectures. These findings underscore robustness as a key challenge for practical spoken query to text retrieval. The code and dataset are publicly available at: https://github.com/ttoyekk1a/ SQuTR-Spoken-Query-to-Text-Retrieval

2 Related Work

- 2.1 Spoken Query Retrieval

Spoken Query Retrieval investigates retrieving relevant information in response to spoken user queries. Prior work largely follows two paradigms: cascaded systems [28] and end-to-end speech retrieval [9, 40]. Cascaded systems first transcribe speech with ASR and then apply mature text retrievers, such as lexical methods (e.g., BM25 [24]) or neural retrievers, benefiting from modularity and reuse of IR infrastructure. However, retrieval quality is inherently affected by transcription errors. End-to-end approaches aim to bypass explicit transcription by directly mapping speech to retrieval representations [9, 14]. Despite progress, existing evaluations rarely provide controlled and systematic analysis of retrieval robustness under varying acoustic noise, making cross-system comparison under realistic conditions difficult.

- 2.2 ASR Robustness and Evaluation

Noise robustness has long been a central topic in ASR, driven by benchmarks and challenge datasets such as CHiME-style evaluations [4, 5]. ASR robustness is typically measured by transcriptionlevel metrics, most notably word error rate (WER) or character error rate (CER), and recent robustness benchmarks [12, 25] further standardize such evaluation under diverse corruptions. However,

#### Table 1: Source datasets used in SQuTR. Query: #queries; Corp.: corpus size; 𝑄len/𝐷len: avg. query/document length.

Dataset Domain Query Corp. 𝑄len 𝐷len NQ [13] Wikipedia 3,452 2.68M 9.16 78.88 HotpotQA [41] Wikipedia 7,405 5.23M 17.61 46.30 FiQA [7] Finance 648 57.6K 10.77 132.32 DuRetrieval [22] Encyclopedia 2,000 0.1M 9.29 398.59 MedicalRetrieval [17] Medical 1,000 0.1M 17.94 122.04 T2Retrieval [39] Encyclopedia 22,812 0.12M 10.94 874.12

#### Table 2: Summary of acoustic conditions used in SQuTR.

Condition Description SNR

Clean Clean speech (no noise) – Low Noise Environmental noise recordings, high SNR 20 dB Medium Noise Environmental noise recordings, moderate SNR 10 dB High Noise Environmental noise recordings, low SNR 0 dB

these benchmarks largely stop at transcription and do not quantify how recognition errors propagate to downstream tasks such as retrieval, limiting end-to-end robustness assessment for spokenquery retrieval systems.

- 2.3 Information Retrieval Benchmarks

Progress in IR has been closely tied to standardized benchmarks. Large-scale datasets such as DuRetrieval [22], T2Ranking [39], and Natural Questions [13], as well as evaluation suites such as BEIR [29] and MTEB [19], have accelerated the development and comparison of retrieval models. However, these benchmarks are fundamentally text-only: they assume clean textual queries and therefore do not capture uncertainty introduced by spoken input and acoustic variability.

Recenteffortshavestartedto incorporate speech into benchmarkdriven evaluation. Massive Sound Embedding Benchmark (MSEB) introduces Simple Voice Questions (SVQ), a large collection of short spoken queries recorded under multiple environments [11]. While MSEB provides valuable coverage for evaluating general auditory representations, SVQ is built around simplified queries and acoustic conditions defined by recording environments rather than explicitly controlled and graded perturbations. Consequently, it does not provide an IR-aligned, end-to-end robustness benchmark for spoken query to text retrieval under increasing acoustic stress. SQuTR fills this gap by extending established IR benchmarks to spoken queries with preserved corpora and relevance judgments, together with controlled, graded acoustic conditions.

3 Dataset Construction

SQuTR is designed to enable controlled and reproducible evaluation of spoken query to text retrieval under varying acoustic conditions. An overview of the benchmark pipeline is provided in Figure 1.

- 3.1 Source Queries

SQuTR reuses queries from six widely used IR benchmarks in English and Chinese, as summarized in Table 1.

[Figure 1]

Figure 1: The SQuTR benchmark pipeline overview. (a-c) Construction of high-fidelity spoken queries from six IR benchmarks using rigorous quality control. (d) Simulation of four acoustic conditions under diverse environmental noise. (e-f) Dataset statistics and the unified evaluation framework for both cascaded and end-to-end retrieval systems.

For the English benchmarks (FiQA [7], HotpotQA [41], and Natural Questions [13]), we use the test splits as queries, with document corpora and relevance judgments following MTEB [19]. For the Chinese benchmarks (MedicalRetrieval [17], DuRetrieval [22], and T2Retrieval [39]), we also use the test splits, following the configurations in C-MTEB [38]. In all cases, document collections and relevance annotations are inherited unchanged from the original benchmarks.

We keep queries in their original form to preserve natural language characteristics and semantic complexity. The resulting set spans factoid QA, multi-hop QA, and general IR, and includes both short keyword-style queries and longer natural-language questions. Overall, SQuTR contains 37,317 unique queries.

### 3.2 Speech Generation

As illustrated in Figure 1(b), we synthesize speech from text using CosyVoice-3 [8] to construct spoken queries under controlled conditions. We fix key generation parameters (speaking rate, pitch, and random seeds) for reproducibility, and apply standard text normalization, including number verbalization, punctuation normalization, and consistent handling of English abbreviations.

We use voice profiles from 200 speakers with varying gender, age, and accents. Queries are grouped by language (English/Chinese), and speakers are uniformly sampled within each group. To mitigate synthesis artifacts, we generate three candidate renditions per query and retain the one with the lowest WER/CER under a reference ASR system.

Audio is generated at 24 kHz with 16-bit depth as clean base signals for subsequent noise injection. We perform automatic checks for synthesis failures and acoustic anomalies and regenerate affected samples when needed.

### 3.3 Acoustic Conditions

To systematically assess retrieval robustness under acoustic perturbations, we generate spoken queries under a set of controlled noise conditions starting from clean speech. SQuTR defines four acoustic

conditions (Clean, Low Noise, Medium Noise, and High Noise), which differ only in noise intensity, as summarized in Table 2.

Additive noise signals follow the DEMAND noise database [30] and cover a diverse set of environmental recordings, including public transportation, office spaces, household environments, and public venues. Babble noise from NOISEX-92 [32] is additionally included. These noise sources are applied at predefined signal-tonoise ratio (SNR) levels to construct the Low, Medium, and High Noise conditions. We further summarize the noise-type distribution in Section 3.5 (Figure 2).

Additive noise is applied using a global RMS-based scaling procedure to achieve the target signal-to-noise ratio (SNR). Given clean speech 𝑥[𝑛] and a noise signal 𝑑[𝑛], the noisy signal is constructed as

𝑦[𝑛] = 𝑥[𝑛] + 𝛼 · 𝑑[𝑛] (1)

where the scaling factor 𝛼 is computed based on the root mean square (RMS) energy of the speech and noise signals:

RMS(𝑥) RMS(𝑑)

SNRdB

· 10−

20 (2)

𝛼 =

RMS energy is computed after trimming leading and trailing silence to ensure that the specified SNR reflects the active speech content. The resulting signal is normalized to prevent clipping.

### 3.4 Quality Control

Figure 1(c) outlines the quality control pipeline, which combines automated filtering with human verification. Automated audio quality checks are used to detect synthesis artifacts, abnormal volume levels, and audio truncation. Clean speech samples are further transcribed using a high-performance ASR system based on WhisperLarge-v3 [23], and WER is computed against the original text to identify samples that may require further inspection.

In the human verification stage, ten bilingual annotators perform auditory checks following standardized guidelines. Verification criteria include speech naturalness and clarity, correctness of the

#### Table 3: Core statistics of SQuTR. Total speech duration sums over all four acoustic conditions.

Metric English Chinese Total #Unique Queries 11,505 25,812 37,317 #Speakers 100 100 200 Average Query Length 13.48 11.08 11.82 Total Speech Duration 76.4 h 114.0 h 190.4 h Average Speech Duration 5.98 s 3.98 s 4.59 s #Evaluation instances 46,020 103,248 149,268

[Figure 2]

#### Figure 2: Distribution of noise categories in the SQuTR.

assigned noise condition, and semantic consistency with the original query. Samples that fail manual inspection are discarded and regenerated. All audio included in SQuTR undergoes at least one round of human verification.

### 3.5 Dataset Statistics

The final SQuTR dataset contains 37,317 unique queries spanning English and Chinese. In our benchmark, an evaluation instance is defined as a (query, acoustic condition) pair. For each query, we generate spoken realizations under four acoustic conditions, resulting in 149,268 evaluation instances across all settings. Table 3 summarizes the dataset, including the number of queries and speakers per language, total speech duration, and the overall scale of the benchmark. By combining realistic query distributions with systematic and controlled acoustic variation, SQuTR provides a solid foundation for robustness evaluation of spoken query to retrieval.

Beyond scale, we further analyze the diversity of acoustic environments represented in SQuTR. Figure 2 shows the distribution of noise categories, covering 16 distinct environmental classes. By maintaining a balanced representation across these noise types, SQuTR enables evaluation across a wide spectrum of acoustic interference, without bias toward specific noise scenarios.

### 3.6 Evaluation Protocol

As shown in Figure 1(f), we define a unified evaluation protocol for spoken query retrieval, treating each system as a single pipeline

Table 4: ASR Performance. Character Error Rate (CER) for Chinese (Paraformer-Large) and Word Error Rate (WER) for English (Whisper-Large-v3) under different noise levels.

Noise Condition Chinese (CER) English (WER)

Clean 2.71% 3.33% Low Noise (20dB) 2.97% 4.10% Medium Noise (10dB) 3.39% 4.48% High Noise (0dB) 7.14% 7.75%

that maps an input audio signal to a ranked list of text documents. In contrast to conventional practices that evaluate ASR and retrieval components separately, SQuTR evaluates retrieval performance directly from spoken input under different acoustic conditions.

To assess performance, we utilize standard information retrieval metricsbasedonexistingrelevancejudgments. We adopt nDCG@10 as the primary metric to evaluate both the presence and ranking position of relevant documents. Complementary metrics include Recall@k, which measures the coverage of relevant documents, and MRR@k, which assesses the capability to retrieve the first relevant result early. All metrics are computed on the final ranked list to reflect user-facing robustness against acoustic perturbations.

### 4 Experiments 4.1 Experimental Setup

- 4.1.1 Datasets. We construct SQuTR by reusing queries from six widely used text retrieval benchmarks [7, 13, 17, 22, 39, 41], including three English and three Chinese datasets, which are defined in Section 3.1.
- 4.1.2 Acoustic Conditions. We evaluate all systems on SQuTR under the four acoustic conditions defined in Section 3.3 and follow the evaluation protocol in Section 3.6. Specifically, we use noise samples corresponding to metro stations (TMETRO), living rooms (DLIVING), parks (PARK), hallways (OHALLWAY), riversides (NRIVER), public squares (SPSQUARE), train stations (PSTATION), meeting rooms (OMEETING), open fields (NFIELD), washing environments (DWASHING), cafeterias (PCAFETER), restaurants (PRESTO), street traffic (STRAFFIC), cafés (SCAFE), kitchens (DKITCHEN), and buses (TBUS). In addition, we include babble noise from the NOISEX-92 corpus (BABBLE) to model background speech interference.
- 4.1.3 Evaluated Systems. Cascaded Systems (ASR + Retrieval): Cascaded systems decompose spoken query retrieval into ASR followed by text-based retrieval. We pair high-performance ASR frontends with a diverse set of text retrieval backends. Specifically, we use Whisper-Largev3 [23] for English and Paraformer-Large [10] for Chinese.

On the retrieval side, we evaluate 12 backends covering both lexical and dense paradigms. (1) Lexical baseline: BM25 [24]. We implement BM25 using Anserini [16], with the default Lucene parameters (𝑘1 = 0.9, 𝑏 = 0.4). (2) Dense retrievers: We include models of varying scales and design choices, including the BGE series [6, 38] (Small, Base, Large, and M3), Qwen3-Embedding [44] (0.6B, 4B, and 8B), and Stella-v5-400M [43]. For BGE-M3, we report results using its dense-only variant for consistency with other dense

#### Table 5: Retrieval performance on Chinese and English sub-datasets under varying acoustic conditions. (nDCG and MRR refer to nDCG@10 and MRR@10. Results for All-MiniLM-L6-v2 and Stella-EN-400M-v5 on Chinese datasets are marked with "–" because these models were not trained on Chinese text.)

Chinese Sub-dataset English Sub-dataset Text Clean Low (20dB) Medium (10dB) High (0dB) Text Clean Low (20dB) Medium (10dB) High (0dB)

Model Configuration

nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR nDCG MRR Cascaded Systems (ASR: Paraformer-Large [Zh] / Whisper-Large-v3 [En]) Lexical Baseline BM25 0.4843 0.5756 0.4380 0.5246 0.4366 0.5229 0.4313 0.5177 0.4061 0.4895 0.3912 0.4547 0.3586 0.4197 0.3570 0.4177 0.3555 0.4157 0.3374 0.3956 Dense (BGE Series)

BGE-Small-(zh/en)-v1.5 0.6871 0.7446 0.6491 0.7064 0.6454 0.7025 0.6402 0.6978 0.6025 0.6593 0.5345 0.5936 0.5070 0.5665 0.5035 0.5632 0.4992 0.5590 0.4756 0.5328 BGE-Base-(zh/en)-v1.5 0.7509 0.7890 0.7157 0.7557 0.7126 0.7523 0.7059 0.7457 0.6670 0.7075 0.5578 0.6130 0.5260 0.5845 0.5220 0.5801 0.5194 0.5766 0.4962 0.5513 BGE-Large-(zh/en)-v1.5 0.7662 0.8032 0.7306 0.7682 0.7274 0.7644 0.7212 0.7585 0.6801 0.7177 0.5801 0.6299 0.5521 0.6058 0.5493 0.6040 0.5466 0.6015 0.5194 0.5721 BGE-M3-dense 0.7320 0.7756 0.6937 0.7381 0.6914 0.7359 0.6864 0.7315 0.6460 0.6912 0.5711 0.6368 0.5397 0.6035 0.5389 0.6034 0.5360 0.5996 0.5097 0.5686

###### Dense (Other)

EmbeddingGemma-300M 0.6952 0.7446 0.6626 0.7122 0.6603 0.7105 0.6554 0.7047 0.6188 0.6681 0.6029 0.6617 0.5797 0.6402 0.5775 0.6373 0.5747 0.6350 0.5497 0.6069 Stella-EN-400M-v5 – – – – – – – – – – 0.6198 0.6786 0.6017 0.6599 0.5986 0.6573 0.5962 0.6546 0.5706 0.6255 All-MiniLM-L6-v2 – – – – – – – – – – 0.4241 0.4862 0.3893 0.4433 0.3854 0.4387 0.3841 0.4368 0.3637 0.4136 Multilingual-E5-Large 0.7479 0.7900 0.7099 0.7528 0.7070 0.7503 0.7008 0.7447 0.6592 0.7019 0.5719 0.6323 0.5398 0.6006 0.5369 0.5967 0.5356 0.5960 0.5115 0.5698

###### Dense (Qwen3 Series)

Qwen3-Embedding-0.6B 0.7405 0.7840 0.7072 0.7512 0.7050 0.7492 0.6992 0.7438 0.6613 0.7063 0.5504 0.6234 0.5288 0.5988 0.5274 0.5978 0.5246 0.5961 0.5026 0.5697 Qwen3-Embedding-4B 0.7936 0.8237 0.7660 0.7978 0.7632 0.7958 0.7573 0.7899 0.7193 0.7528 0.6488 0.7110 0.6252 0.6886 0.6227 0.6860 0.6206 0.6839 0.5947 0.6565 Qwen3-Embedding-8B 0.8033 0.8315 0.7760 0.8057 0.7741 0.8041 0.7686 0.7988 0.7302 0.7608 0.6686 0.7253 0.6450 0.7041 0.6424 0.7021 0.6405 0.7000 0.6120 0.6690

End-to-End Systems (Direct Audio Input - No ASR) Omni-Embed-Nemotron-3B – – 0.6648 0.7201 0.6614 0.7179 0.6507 0.7067 0.5742 0.6314 – – 0.5712 0.5394 0.5680 0.5369 0.5605 0.5289 0.5236 0.4959

[Figure 3]

- Figure 3: Accuracy vs. Stability Trade-off. The scatter plot positions systems based on Mean NDCG@10 (Y-axis) and Standard Deviation (X-axis). Models enclosed in dashed lines are evaluated on English datasets only. retrievers. We further evaluate several widely used strong baselines, including EmbeddingGemma-300M [33], All-MiniLM-L6v2 [36], and Multilingual-E5-Large-Instruct [35].

End-to-End Systems: In contrast to cascaded pipelines, the end-to-end system directly maps spoken queries to retrieval representations without explicit transcription. We evaluate OmniEmbed-Nemotron-3B [40], which projects raw audio signals into a shared embedding space for retrieval.

### 4.2 Main Results

- Table 5 validates SQuTR’s effectiveness in quantifying robustness and discriminating between architectural capabilities under graded acoustic noise. As noise increases (Clean → Low → High), retrieval performance degrades across systems, but the degradation patterns

differ by model scale and design: traditional dense retrievers (e.g., BGE-Large) show a more pronounced monotonic drop, whereas large generative models remain comparatively stable (e.g., Qwen3Embedding-8B sustains stronger performance at 0dB), indicating that SQuTR provides sufficient discriminative power for robustness evaluation.

We also observe that spoken-query retrieval underperforms the text upper bound even in clean conditions, and the gap widens with noise. For example, Qwen3-Embedding-8B (Chinese) drops from 0.8033 on Text to 0.7760 on Clean Speech. This gap is consistent with error propagation in cascaded systems (ASR errors affecting downstream retrieval) and the inherent difficulty of aligning speech and text representations in end-to-end models. Finally, SQuTR remains a challenging testbed. Even the strongest model, Qwen3-Embedding-8B, does not match text performance under noise (0.7302 at High Noise vs. 0.8033 on Text), and end-to-end models still lag behind strong cascaded baselines, leaving substantial headroom for improving noise-robust spoken query retrieval.

### 4.3 Robustness and Stability Analysis

We examine the trade-off between accuracy (mean nDCG@10) and stability (standard deviation, 𝜎), where 𝜎 is computed over all language–condition combinations (two languages × four acoustic conditions). Figure 3 visualizes this distribution. All-MiniLM-L6-v2 and Stella-EN-400M-v5 (dashed circles) are evaluated on English only and excluded from the main comparison.

First, there is a clear stability gap between sparse and dense architectures. BM25 remains comparatively stable (𝜎 = 0.031), while BERT-based dense retrievers (e.g., BGE-Base, 𝜎 = 0.100) exhibit substantially higher variance, indicating greater sensitivity to acoustic

#### Table 6: Retrieval Performance across different ASR and Acoustic Conditions. nDCG and MRR refer to nDCG@10 and MRR@10.

Chinese Sub-dataset English Sub-dataset ASR Model Noise Condition

BM25 Qwen3-Embedding-8B nDCG MRR nDCG MRR nDCG MRR nDCG MRR

BM25 Qwen3-Embedding-8B

CER(%)

WER(%)

Clean 5.32 0.4090 0.4937 0.7562 0.7875 8.82 0.3759 0.3375 0.5991 0.6571 Low Noise (20dB) 6.22 0.4061 0.4915 0.7524 0.7839 9.46 0.3757 0.3363 0.5985 0.6561 Medium Noise (10dB) 6.44 0.4013 0.4861 0.7476 0.7792 10.00 0.3699 0.3309 0.5941 0.6516 High Noise (0dB) 11.99 0.3734 0.4528 0.7109 0.7423 13.32 0.3436 0.3113 0.5599 0.6141

SenseVoice-Small [1]

Clean 3.08 0.4430 0.5298 0.7670 0.7963 6.47 0.3266 0.3837 0.6253 0.6837 Low Noise (20dB) 3.23 0.4419 0.5286 0.7675 0.7965 6.74 0.3253 0.3822 0.6215 0.6792 Medium Noise (10dB) 3.47 0.4389 0.5259 0.7655 0.7948 6.93 0.3238 0.3809 0.6207 0.6784 High Noise (0dB) 6.72 0.4179 0.5017 0.7351 0.7644 10.03 0.3074 0.3626 0.5897 0.6464

Fun-ASR-Nano-2512 [2]

Clean 4.77 0.4305 0.5158 0.7526 0.7831 6.58 0.3256 0.3811 0.6322 0.6921 Low Noise (20dB) 4.73 0.4297 0.5152 0.7532 0.7839 6.37 0.3264 0.3818 0.6306 0.6897 Medium Noise (10dB) 5.08 0.4256 0.5105 0.7483 0.7787 6.25 0.3260 0.3817 0.6284 0.6878 High Noise (0dB) 11.44 0.3905 0.4696 0.6941 0.7245 9.13 0.3094 0.3620 0.5970 0.6522

GLM-ASR-Nano-2512 [42]

Clean 8.77 0.3718 0.4525 0.7019 0.7359 3.33 0.3586 0.4197 0.6450 0.7041 Low Noise (20dB) 8.96 0.3661 0.4471 0.6987 0.7334 4.10 0.3570 0.4177 0.6424 0.7021 Medium Noise (10dB) 9.62 0.3589 0.4387 0.6929 0.7268 4.48 0.3555 0.4157 0.6405 0.7000 High Noise (0dB) 17.31 0.3020 0.3705 0.6128 0.6460 7.75 0.3374 0.3956 0.6120 0.6690

Whisper-Large-V3 [23]

Clean 3.07 0.4416 0.5271 0.7655 0.7957 4.49 0.3503 0.4098 0.6407 0.6991 Low Noise (20dB) 3.19 0.4405 0.5255 0.7648 0.7955 4.90 0.3427 0.4015 0.6384 0.6977 Medium Noise (10dB) 3.40 0.4368 0.5219 0.7611 0.7917 5.15 0.3403 0.3980 0.6357 0.6951 High Noise (0dB) 5.43 0.4198 0.5027 0.7351 0.7654 6.98 0.3290 0.3859 0.6150 0.6713

Qwen3-ASR-1.7B [26]

#### Table 7: Impact of Whisper ASR Model Size on Retrieval Performance: BM25 vs. Qwen3-Embedding-8B. nDCG and MRR refer to nDCG@10 and MRR@10.

Low Noise (20dB) to stochastic variance rather than noise benefits. Since modern ASR models saturate at 20dB (treating it effectively as clean), random variations in error patterns (e.g., hallucinations vs. deletions) can occasionally favor exact keyword matching. In contrast, dense retrieval filters these lexical instabilities, resulting in a smoother, monotonic decline.

BM25 Qwen3-Embedding-8B ASR Model Params Noise Condition WER(%)

nDCG MRR nDCG MRR

Clean 10.26 0.2966 0.3511 0.5862 0.6455 Low Noise (20dB) 10.57 0.2963 0.3506 0.5828 0.6403 Medium Noise (10dB) 12.46 0.2852 0.3387 0.5665 0.6245 High Noise (0dB) 26.48 0.2344 0.2808 0.4666 0.5191

Whisper-Tiny 39M

### 5 Conclusion

Clean 8.12 0.3147 0.3719 0.6022 0.6612 Low Noise (20dB) 8.30 0.3140 0.3706 0.5996 0.6579 Medium Noise (10dB) 9.34 0.3088 0.3644 0.5915 0.6498 High Noise (0dB) 18.19 0.2691 0.3206 0.5148 0.5689

Whisper-Base 74M

We introduce SQuTR, a controllable and reproducible benchmark for evaluating spoken query to text retrieval under graded acoustic noise. The benchmark comprises a large-scale spoken-query dataset derived from widely used text retrieval benchmarks and augmented with systematically controlled acoustic conditions. SQuTR provides a unified framework to assess robustness across lexical, dense, and end-to-end retrieval paradigms. Our experiments reveal consistent performance degradation as noise increases, and highlight architectural and scaling differences in robustness. These findings suggest that noise-robust spoken retrieval remains an open challenge. We hope SQuTR will facilitate more systematic and comparable research on robustness in spoken query to text retrieval.

Clean 6.17 0.3358 0.3944 0.6246 0.6848 Low Noise (20dB) 6.18 0.3365 0.3955 0.6229 0.6820 Medium Noise (10dB) 6.78 0.3339 0.3927 0.6190 0.6775 High Noise (0dB) 12.01 0.3050 0.3599 0.5681 0.6239

Whisper-Small 244M

Clean 5.21 0.3465 0.4061 0.6322 0.6920 Low Noise (20dB) 5.11 0.3476 0.4078 0.6329 0.6923 Medium Noise (10dB) 5.45 0.3451 0.4046 0.6294 0.6878 High Noise (0dB) 9.72 0.3231 0.3807 0.5931 0.6496

Whisper-Medium 769M

Clean 3.33 0.3586 0.4197 0.6450 0.7041 Low Noise (20dB) 4.10 0.3570 0.4177 0.6424 0.7021 Medium Noise (10dB) 4.48 0.3555 0.4157 0.6405 0.7000 High Noise (0dB) 7.75 0.3374 0.3956 0.6120 0.6690

Whisper-Large-v3 1550M

perturbations despite strong clean-condition performance. Second, model scaling and direct modality modeling improve robustness. Within the Qwen3 series, increasing model size from 0.6B to 8B reduces variance (𝜎 : 0.094 → 0.070) while improving mean effectiveness. The end-to-end model Omni-Embed-Nemotron-3B (𝜎 = 0.054) further narrows this gap, suggesting that larger capacity and direct speech-to-embedding mapping mitigate instability.

### References

- [1] Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui Hu, Kai Hu, et al. 2024. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms. arXiv preprint arXiv:2407.04051 (2024).
- [2] Keyu An, Yanni Chen, Zhigao Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Bo Gong, Xiangang Li, Yabin Li, et al. 2025. Fun-ASR Technical Report. arXiv preprint arXiv:2509.12508 (2025).
- [3] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs.CL] https://arxiv.org/abs/1611.09268
- [4] Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe. 2015. The third ‘CHiME’speech separation and recognition challenge: Dataset, task and baselines. In 2015 IEEE workshop on automatic speech recognition and understanding (ASRU). IEEE, 504–511.
- [5] Jon Barker, Shinji Watanabe, Emmanuel Vincent, and Jan Trmal. 2018. The fifth ’CHiME’ Speech Separation and Recognition Challenge: Dataset, task and baselines. arXiv:1803.10609 [cs.SD] https://arxiv.org/abs/1803.10609

### 4.4 Ablation Study

Table 7 confirms that scaling ASR models (e.g., Whisper-Tiny to Whisper-Large-v3) improves retrieval metrics. However, the choice of retrieval backend proves more critical than ASR size: notably, the smallest Whisper-Tiny using dense retrieval (Qwen3-Embedding8B) significantly outperforms the largest Whisper-Large-v3 using BM25. This highlights that semantic robustness effectively compensates for ASR transcription errors where lexical matching fails. Despite the performance gap, lexical and dense retrieval exhibit highly consistent degradation trends, maintaining stability from Clean to Medium Noise (10dB) before dropping sharply at High Noise (0dB). We attribute the minor BM25 performance fluctuations under

- [6] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 4, 5 (2024).
- [7] Dayan de França Costa and Nadia Felix Felipe da Silva. 2018. INF-UFG at FiQA 2018 task 1: Predicting sentiments and aspects on financial tweets and news headlines. In Companion Proceedings of the The Web Conference 2018. 1967–1971.
- [8] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, and Jieping Ye. 2025. CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training. arXiv:2505.17589 [cs.SD] https://arxiv.org/abs/2505.17589
- [9] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang.

2022. CLAP: Learning Audio Concepts From Natural Language Supervision. arXiv:2206.04769 [cs.SD] https://arxiv.org/abs/2206.04769

- [10] Zhifu Gao, Shiliang Zhang, Ian McLoughlin, and Zhijie Yan. 2022. Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition. arXiv preprint arXiv:2206.08317 (2022).
- [11] Georg Heigold, Ehsan Variani, Tom Bagby, Cyril Allauzen, Ji Ma, Shankar Kumar, and Michael Riley. [n.d.]. Massive Sound Embedding Benchmark (MSEB). In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.
- [12] Heeseung Kim, Che Hyun Lee, Sangkwon Park, Jiheum Yeom, Nohil Park, Sangwon Yu, and Sungroh Yoon. 2025. Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models. arXiv:2502.19759 [cs.SD] https://arxiv.org/abs/2502.19759
- [13] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics 7 (2019), 453–466.
- [14] Lin-shan Lee, James Glass, Hung-yi Lee, and Chun-an Chan. 2015. Spoken Content Retrieval—Beyond Cascading Speech Recognition with Text Retrieval. IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, 9 (2015), 1389–1420. doi:10.1109/TASLP.2015.2438543
- [15] Chyi-Jiunn Lin, Guan-Ting Lin, Yung-Sung Chuang, Wei-Lun Wu, Shang-Wen Li, Abdelrahman Mohamed, Hung-yi Lee, and Lin-Shan Lee. 2024. Speechdpr: Endto-end spoken passage retrieval for open-domain spoken question answering. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 12476–12480.
- [16] Jimmy Lin, Matt Crane, Andrew Trotman, Jamie Callan, Ishan Chattopadhyaya, John Foley, Grant Ingersoll, Craig Macdonald, and Sebastiano Vigna. 2016. Toward reproducible baselines: The open-source IR reproducibility challenge. In European Conference on Information Retrieval. Springer, 408–420.
- [17] Dingkun Long, Qiong Gao, Kuan Zou, Guangwei Xu, Pengjun Xie, Ruijie Guo, Jian Xu, Guanjun Jiang, Luxi Xing, and Ping Yang. 2022. Multi-cpr: A multi domain chinese dataset for passage retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 3046–3056.
- [18] Do June Min, Karel Mundnich, Andy Lapastora, Erfan Soltanmohammadi, Srikanth Ronanki, and Kyu Han. 2025. Speech Retrieval-Augmented Generation without Automatic Speech Recognition. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 1–5. doi:10.1109/ICASSP49660.2025.10888900
- [19] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. Mteb: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics. 2014–2037.
- [20] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 5206–5210. doi:10.1109/ICASSP.2015.7178964
- [21] Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. 2019. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779 (2019).
- [22] Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu, and Haifeng Wang. 2022. DuReader_retrieval: A Large-Scale Chinese Benchmark for Passage Retrieval from Web Search Engine. arXiv preprint arXiv:2203.10232

(2022).

- [23] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust Speech Recognition via Large-Scale Weak Supervision. arXiv:2212.04356 [eess.AS] https://arxiv.org/abs/2212.04356
- [24] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and trends® in information retrieval 3, 4 (2009), 333–389.
- [25] Muhammad A. Shah, David Solans Noguero, Mikko A. Heikkila, Bhiksha Raj, and Nicolas Kourtellis. 2024. Speech Robust Bench: A Robustness Benchmark For Speech Recognition. arXiv:2403.07937 [eess.AS] https://arxiv.org/abs/2403.07937

- [26] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al. 2026. Qwen3-ASR Technical Report. arXiv preprint arXiv:2601.21337 (2026).
- [27] Georgios Sidiropoulos and Evangelos Kanoulas. 2024. A Multimodal Dense Retrieval Approach for Speech-Based Open-Domain Question Answering. arXiv preprint arXiv:2409.13483 (2024).
- [28] Georgios Sidiropoulos, Svitlana Vakulenko, and Evangelos Kanoulas. 2022. On the impact of speech recognition errors in passage retrieval for spoken question answering. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 4485–4489.
- [29] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
- [30] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent. 2013. The diverse environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings. In Proceedings of Meetings on Acoustics, Vol. 19. Acoustical Society of America, 035081.
- [31] Christophe Van Gysel. 2023. Modeling spoken information queries for virtual assistants: Open problems, challenges and opportunities. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 3335–3338.
- [32] Andrew Varga and Herman JM Steeneken. 1993. Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems. Speech communication 12, 3 (1993), 247–251.
- [33] Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, et al. 2025. Embeddinggemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354 (2025).
- [34] Ke Wang, Houxing Ren, Zimu Lu, Mingjie Zhan, and Hongsheng Li. 2025. VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing. arXiv preprint arXiv:2509.22651 (2025).
- [35] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 (2024).
- [36] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.

2020. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems 33 (2020), 5776–5788.

- [37] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai. 2018. ESPnet: Endto-End Speech Processing Toolkit. In Interspeech 2018. 2207–2211. doi:10.21437/ Interspeech.2018-1456
- [38] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval. 641–649.
- [39] Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. 2023. T2ranking: A large-scale chinese benchmark for passage ranking. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2681–2690.
- [40] Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even Oldridge, and Benedikt Schifferer. 2025. Omni-EmbedNemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video. arXiv:2510.03458 [cs.CL] https://arxiv.org/abs/2510.03458
- [41] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing. 2369–2380.
- [42] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471

(2025).

- [43] Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong Wang. 2024. Jasper and stella: distillation of sota embedding models. arXiv preprint arXiv:2412.19048 (2024).
- [44] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176 (2025).

### A Mathematical Formulation of Acoustic Simulation

To ensure the reproducibility of the SQuTR acoustic environment, we formally define the signal degradation process based on global energy statistics. The distorted signal 𝑦[𝑛] is modeled as a linear superposition of the reverberant speech and additive noise:

𝑦[𝑛] = (𝑥 ∗ ℎ)[𝑛] + 𝛼 · 𝑑[𝑛] (3)

where 𝑛 is the discrete time index. 𝑥[𝑛] represents the clean speech synthesized by CosyVoice-3. Prior to processing, all audio components are strictly resampled to a unified rate of 24kHz to prevent spectral aliasing.

The convolution (𝑥 ∗ ℎ)[𝑛] incorporates room acoustics via the Room Impulse Response (RIR) ℎ[𝑛]. We define the target signal 𝑠[𝑛] = (𝑥 ∗ ℎ)[𝑛] as the holistic reverberant speech. To ensure the Signal-to-Noise Ratio (SNR) accurately reflects the active speech level, standard leading and trailing silence trimming is applied to 𝑥[𝑛] before energy calculation.

The additive noise sequence 𝑑[𝑛] is randomly sampled from the DEMAND/NOISEX-92 datasets. The scaling factor 𝛼 is derived from the global Root Mean Square (RMS) amplitude:

𝑁∑︁−1

𝑁∑︁−1

1 𝑁

1 𝑁

RMS(𝑠) =

|𝑠[𝑛]|2, RMS(𝑑) =

|𝑑[𝑛]|2 (4)

𝑛=0

𝑛=0

where 𝑁 denotes the signal length. Given a target SNRdB, the mixing coefficient 𝛼 is solved as:

RMS(𝑠) RMS(𝑑)

SNRdB

· 10−

20 (5)

𝛼 =

Finally, the mixed signal𝑦[𝑛] is globally normalized to fit within the standard PCM dynamic range, utilizing a headroom factor 𝛽 = 0.9 to prevent digital clipping.

### B Automated Quality Control Protocol

To ensure the semantic integrity of the dataset, we define a rigorous filtering function Φ : X → {0, 1} that maps a generated audio sample 𝑥 to a binary acceptance decision. The process involves a cascade of objective metrics and subjective proxy evaluations.

The filtering pipeline, formally described in Algorithm 1 (Figure 4), integrates two distinct verification stages:

- (1) Objective Consistency Check: We utilize a large-scale ASR model (M𝐴𝑆𝑅) to verify if the synthesized speech 𝑎𝑖 aligns with the ground-truth text 𝑡𝑖. To mitigate minor formatting discrepancies, a normalization function N(·) (lowercasing, punctuation removal) is applied before computing the Word Error Rate (WER).
- (2) Subjective Proxy Evaluation: We employ a "Judge-LLM" paradigmutilizingtwodistinctLarge Language Models,M𝐺𝑒𝑚𝑖𝑛𝑖 and M𝐺𝑃𝑇, as independent evaluators. They score the audio quality 𝑄 ∈ [1, 5] based on a Chain-of-Thought reasoning process.

|Algorithm 1: Robust Multi-Stage Quality Filtering<br><br>Input: ASR Model M𝐴𝑆𝑅 (Whisper-v3 for En, Paraformer for Zh) Hyperparameters:<br><br>• WER Threshold 𝜏𝑤𝑒𝑟 = 0.30<br>• Quality Score Threshold 𝜏𝑞𝑢𝑎𝑙 = 4<br><br><br>Output: Clean Dataset D𝑐𝑙𝑒𝑎𝑛<br><br>(1) D𝑐𝑙𝑒𝑎𝑛 ← ∅<br>(2) for each pair (𝑎𝑖,𝑡𝑖) ∈ D do<br>(3) // Stage 1: Objective ASR Verification<br>(4) 𝑡ˆ𝑖 ← M𝐴𝑆𝑅 (𝑎𝑖) ⊲ Transcribe Audio<br>(5) 𝑡ˆ𝑖′,𝑡𝑖′ ← N(𝑡ˆ𝑖), N(𝑡𝑖) ⊲ Text Normalization<br>(6) 𝛿𝑤𝑒𝑟 ← CalcWER(𝑡ˆ𝑖′,𝑡𝑖′)<br>(7) if 𝛿𝑤𝑒𝑟 > 𝜏𝑤𝑒𝑟 then<br>(8) continue ⊲ Reject: High mismatch<br>(9) // Stage 2: Dual-LLM Consensus<br>(10) 𝑆𝑠𝑐𝑜𝑟𝑒𝑠 ← ∅<br>(11) for M𝑘 ∈ E do ⊲ Parallel Evaluation<br>(12) 𝑠𝑘, reason𝑘 ← M𝑘.Eval(𝑎𝑖,𝑡𝑖 |Prompt𝐶𝑜𝑇 )<br>(13) 𝑆𝑠𝑐𝑜𝑟𝑒𝑠 ← 𝑆𝑠𝑐𝑜𝑟𝑒𝑠 ∪ {𝑠𝑘 }<br>(14) if min(𝑆𝑠𝑐𝑜𝑟𝑒𝑠 ) ≥ 𝜏𝑞𝑢𝑎𝑙 then<br>(15) D𝑐𝑙𝑒𝑎𝑛 ← D𝑐𝑙𝑒𝑎𝑛 ∪ {(𝑎𝑖,𝑡𝑖)} ⊲ Accept<br>(16) else<br>(17) Discard 𝑎𝑖 (Optionally flag for manual review)<br>(18) end for<br>(19) return D𝑐𝑙𝑒𝑎𝑛<br>|
|---|

Figure 4: Pseudocode for the automated quality control protocol. The pipeline enforces both semantic accuracy (via ASR) and perceptual fidelity (via LLM consensus).

### C Detailed Experimental Setup

To ensure full reproducibility, we strictly controlled the computational environment, inference hyperparameters, and model-specific configurations.

### C.1 Computational Environment

All experiments were conducted on a high-performance computing cluster. The detailed hardware and software specifications are listed in Table 8.

Table 8: Global hardware and software configuration.

Type Component Specification H/W GPU NVIDIA A100-SXM4 (80GB) S/W OS Ubuntu 24.04 LTS

Framework PyTorch 2.5.1 Transformers v4.57.1

ASR Decoding Greedy

Precision FP16

### C.2 ASR Inference Protocol

To comprehensively evaluate the impact of ASR robustness on retrieval, we employed a multi-model strategy covering different parameter scales and languages.

Input Preprocessing. Although the SQuTR dataset is mastered at a sampling rate of 24kHz to preserve high-frequency acoustic details (as detailed in Appendix A), most pre-trained ASR systems are optimized for 16kHz input. To prevent spectral aliasing and timescale modification artifacts during inference, we applied on-the-fly downsampling to 16kHz for all ASR models using torchaudio (Lanczos resampling kernel).

Model Configurations. We evaluated the following systems:

- • English (Whisper Family): We utilized OpenAI’s whisper series:

- – Tiny (39M): openai/whisper-tiny
- – Base (74M): openai/whisper-base
- – Small (244M): openai/whisper-small
- – Medium (769M): openai/whisper-medium

- • Chinese: We adopted Paraformer.

– Paraformer-Large: funasr/paraformer-large

- • Multilingual Baselines: To assess cross-lingual robustness, we also included:

- – SenseVoice-Small: FunAudioLLM/SenseVoiceSmall
- – Whisper-Large-v3: openai/whisper-large-v3
- – GLM-ASR-Nano-2512: zai-org/GLM-ASR-Nano-2512
- – Fun-ASR-Nano-2512: FunAudioLLM/Fun-ASR-Nano-2512
- – Qwen3-ASR-1.7B: Qwen/Qwen3-ASR-1.7B

For all ASR inference, we used deterministic greedy decoding with a forced language token to prevent identification errors.

### C.3 Retrieval Model Configuration

We evaluated 15 distinct cascaded retrieval architectures. Modern embedding models often require specific **task instructions** (prefixes) to activate their asymmetric retrieval capabilities. We strictly adhered to the official prompt templates as detailed in Table 9.

- Table 9: Model-specific query instructions (Prefixes) for all embedding models.

Model Family Query Instruction / Prompt Lexical BM25 N/A (𝑘1 = 0.9,𝑏 = 0.4) BGE Series v1.5 "Represent this sentence for searching relevant passages:" BGE-M3 N/A (Dense vector-only mode) Multilingual-E5-Large "query: " EmbeddingGemma-300M "task: search result | query: " All-MiniLM-L6-v2 N/A Stella-EN-400M-v5 "Instruct: Given a web search query, retrieve relevant pas-

sages that answer the query. Query:"

Qwen3 Embedding Series "Given a web search query, retrieve relevant passages that

answer the query"

N/A

Omni-Embed-Nemotron3B

### C.4 Evaluation Metrics Standard

We report the normalized Discounted Cumulative Gain at rank 10 (nDCG@10). The implementationstrictly follows the NISTtrec_eval standard.

• Relevance Mapping: For datasets with graded relevance (e.g., FiQA), we preserve original grades. For binary datasets (e.g., NQ), relevance is mapped to {0, 1}.

• Global Ranking: For multi-GPU inference, embeddings were gathered across devices to ensure accurate global ranking statistics.

|[SYSTEM INSTRUCTION]<br><br>Role: You are an expert Audio Forensic Analyst and Linguistic Quality Specialist. Task: You will be provided with a Reference Text and an Audio Clip. Your goal is to evaluate the audio’s fidelity and quality with extreme rigor. Evaluation Protocol (Step-by-Step):<br><br>(1) Listen & Transcribe: Mentally transcribe the audio. Compare it strictly against the Reference Text.<br>(2) Check for Fatal Errors:<br><br>• Truncation: Is the start or end of the sentence cut off?<br>• Hallucination: Does the audio contain words not in the text?<br>• Omission: Are any words skipped?<br><br><br>(3) Assess Signal Quality: Detect background static, metallic robotic artifacts, or unnatural pitch shifts.<br><br><br>Scoring Rubric (Strict):<br><br>• 5.0 (Perfect): Indistinguishable from human recording. 100% text match. Zero noise.<br>• 4.0 (Good): Clear and correct. Slight digital "flavor" but fully intelligible.<br>• 3.0 (Acceptable): Noticeable noise or slight mispronunciation, but meaning is preserved.<br>• 2.0 (Bad): Hard to understand. Significant artifacts. Missing 1-2 noncritical words.<br>• 1.0 (Critical Failure): Wrong text, severe truncation, or pure noise.<br><br><br>Constraint Checklist:<br><br>• Do NOT be lenient. Be critical of minor glitches.<br>• If the audio is truncated (incomplete sentence), the max score is 2.0.<br>• Output MUST be valid JSON.<br><br><br>[USER INPUT]<br><br>Reference Text: "{TRANSCRIPT}" Audio Data: <Audio_Attachment><br><br>[MODEL OUTPUT SCHEMA] {<br><br>"analysis_trace": { "text_alignment": "Exact Match / Mismatch", "detected_artifacts": ["None", "Static", "Truncation"], "prosody_assessment": "Natural / Robotic / Monotone" }, "fatal_error_flag": false, "reasoning_summary": "The audio is clear...", "final_quality_score": <Float 1.0-5.0><br><br>}|
|---|

Figure 5: The advanced Chain-of-Thought (CoT) prompt template used for automated quality control. The prompt includes explicit failure mode definitions and specific handling for TTS edge cases (e.g., truncation).

### D Robust Evaluation Prompts

To ensure the validity and consistency of our "LLM-as-a-Judge" pipeline, we developed a sophisticated prompt template incorporating Role-Playing, Chain-of-Thought (CoT) reasoning, and Negative Constraints.

The prompt explicitly enforces an "Audio Forensic Analyst" persona to minimize hallucination. The JSON output schema is strictly defined to ensure parsing reliability. The complete configuration is presented in Figure 5.

### E Extended Experimental Results

We present the full retrieval diagnostics across all six sub-datasets in Tables 10 through 21.

#### Table 10: Main experimental results on the FiQA dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: FiQA Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.2388 0.3002 0.2995 0.2335 0.2946 0.2869 0.2329 0.2943 0.2887 0.2333 0.2934 0.2903 0.2214 0.2787 0.2736 BGE-Small-(zh/en)-v1.5 0.4030 0.4879 0.4640 0.3962 0.4767 0.4635 0.3926 0.4745 0.4568 0.3895 0.4714 0.4525 0.3750 0.4519 0.4390 BGE-Base-(zh/en)-v1.5 0.4062 0.4859 0.4804 0.3933 0.4763 0.4643 0.3883 0.4707 0.4569 0.3911 0.4720 0.4620 0.3766 0.4521 0.4485 BGE-Large-(zh/en)-v1.5 0.4487 0.5327 0.5130 0.4333 0.5158 0.4987 0.4331 0.5195 0.4976 0.4332 0.5203 0.4983 0.4130 0.4951 0.4775 BGE-M3-dense 0.4126 0.5046 0.4720 0.3948 0.4824 0.4574 0.3972 0.4860 0.4592 0.3940 0.4805 0.4540 0.3760 0.4531 0.4438 EmbeddingGemma-300M 0.4739 0.5467 0.5533 0.4666 0.5462 0.5421 0.4634 0.5398 0.5405 0.4631 0.5413 0.5399 0.4485 0.5210 0.5239 Stella-EN-400M-v5 0.5494 0.6290 0.6225 0.5427 0.6225 0.6162 0.5392 0.6195 0.6121 0.5396 0.6186 0.6120 0.5231 0.5942 0.5977 All-MiniLM-L6-v2 0.3687 0.4446 0.4414 0.3475 0.4195 0.4200 0.3419 0.4133 0.4122 0.3468 0.4176 0.4193 0.3306 0.3990 0.4036 Multilingual-E5-Large 0.4617 0.5439 0.5324 0.4391 0.5181 0.5150 0.4346 0.5120 0.5113 0.4366 0.5138 0.5147 0.4229 0.4983 0.4981 Qwen3-Embedding-0.6B 0.4701 0.5546 0.5472 0.4567 0.5376 0.5324 0.4561 0.5388 0.5332 0.4543 0.5397 0.5279 0.4406 0.5199 0.5195 Qwen3-Embedding-4B 0.5889 0.6692 0.6694 0.5763 0.6586 0.6542 0.5730 0.6547 0.6559 0.5738 0.6554 0.6537 0.5523 0.6293 0.6289 Qwen3-Embedding-8B 0.6162 0.6845 0.6990 0.5968 0.6653 0.6793 0.5949 0.6647 0.6786 0.5974 0.6678 0.6788 0.5723 0.6357 0.6571 Omni-Embed-Nemotron-3B – – – 0.4568 0.2922 0.5047 0.4548 0.2935 0.5003 0.4515 0.2894 0.4988 0.4299 0.2794 0.4738

#### Table 11: Main experimental results on the HotpotQA dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: HotpotQA Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.6291 0.8004 0.6531 0.5464 0.7098 0.5727 0.5426 0.7040 0.5697 0.5399 0.7008 0.5667 0.5128 0.6678 0.5394 BGE-Small-(zh/en)-v1.5 0.6993 0.8413 0.7279 0.6197 0.7649 0.6519 0.6158 0.7608 0.6487 0.6119 0.7563 0.6451 0.5801 0.7211 0.6128 BGE-Base-(zh/en)-v1.5 0.7259 0.8610 0.7573 0.6543 0.7953 0.6911 0.6492 0.7904 0.6868 0.6452 0.7846 0.6825 0.6119 0.7498 0.6466 BGE-Large-(zh/en)-v1.5 0.7411 0.8613 0.7800 0.6708 0.8029 0.7106 0.6649 0.7955 0.7071 0.6621 0.7922 0.7041 0.6273 0.7541 0.6695 BGE-M3-dense 0.6944 0.8466 0.7182 0.6294 0.7822 0.6597 0.6287 0.7824 0.6580 0.6264 0.7797 0.6565 0.5941 0.7412 0.6245 EmbeddingGemma-300M 0.7008 0.8581 0.7290 0.6574 0.8166 0.6891 0.6556 0.8147 0.6862 0.6525 0.8110 0.6835 0.6216 0.7760 0.6518 Stella-EN-400M-v5 0.6874 0.8387 0.7204 0.6479 0.7972 0.6848 0.6448 0.7952 0.6808 0.6417 0.7917 0.6771 0.6100 0.7553 0.6464 All-MiniLM-L6-v2 0.4650 0.6282 0.4867 0.4122 0.5560 0.4393 0.4077 0.5501 0.4344 0.4031 0.5437 0.4303 0.3797 0.5114 0.4069 Multilingual-E5-Large 0.6828 0.8298 0.7167 0.6165 0.7656 0.6530 0.6140 0.7622 0.6513 0.6118 0.7607 0.6479 0.5803 0.7232 0.6165 Qwen3-Embedding-0.6B 0.6506 0.8361 0.6600 0.5986 0.7786 0.6151 0.5962 0.7762 0.6132 0.5943 0.7737 0.6109 0.5652 0.7372 0.5835 Qwen3-Embedding-4B 0.7249 0.8831 0.7442 0.6791 0.8404 0.7047 0.6768 0.8383 0.7029 0.6728 0.8338 0.6991 0.6427 0.8013 0.6689 Qwen3-Embedding-8B 0.7424 0.8954 0.7637 0.6968 0.8550 0.7235 0.6931 0.8514 0.7200 0.6891 0.8458 0.7167 0.6574 0.8120 0.6850 Omni-Embed-Nemotron-3B – – – 0.6188 0.7351 0.6657 0.6139 0.7284 0.6629 0.6035 0.7175 0.6531 0.5590 0.6723 0.6057

#### Table 12: Main experimental results on the NQ dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: NQ Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.3056 0.2636 0.4731 0.2959 0.2547 0.4602 0.2954 0.2549 0.4579 0.2934 0.2529 0.4551 0.2780 0.2403 0.4291 BGE-Small-(zh/en)-v1.5 0.5012 0.4516 0.7080 0.5052 0.4580 0.7029 0.5021 0.4542 0.6996 0.4962 0.4494 0.6895 0.4718 0.4254 0.6619 BGE-Base-(zh/en)-v1.5 0.5412 0.4920 0.7464 0.5305 0.4819 0.7326 0.5285 0.4792 0.7294 0.5219 0.4731 0.7220 0.5002 0.4521 0.6964 BGE-Large-(zh/en)-v1.5 0.5504 0.4957 0.7650 0.5522 0.4985 0.7629 0.5499 0.4969 0.7602 0.5446 0.4920 0.7536 0.5178 0.4671 0.7199 BGE-M3-dense 0.6063 0.5593 0.7942 0.5950 0.5459 0.7895 0.5909 0.5417 0.7862 0.5876 0.5387 0.7819 0.5590 0.5115 0.7469 EmbeddingGemma-300M 0.6342 0.5804 0.8447 0.6150 0.5580 0.8322 0.6136 0.5574 0.8287 0.6085 0.5526 0.8227 0.5789 0.5237 0.7908 Stella-EN-400M-v5 0.6227 0.5683 0.8408 0.6143 0.5598 0.8322 0.6117 0.5572 0.8294 0.6074 0.5536 0.8233 0.5787 0.5269 0.7882 All-MiniLM-L6-v2 0.4386 0.3859 0.6474 0.4081 0.3544 0.6181 0.4067 0.3528 0.6153 0.4023 0.3492 0.6097 0.3809 0.3304 0.5797 Multilingual-E5-Large 0.5712 0.5231 0.7675 0.5639 0.5181 0.7565 0.5622 0.5159 0.7574 0.5585 0.5135 0.7506 0.5314 0.4878 0.7183 Qwen3-Embedding-0.6B 0.5305 0.4795 0.7371 0.5310 0.4801 0.7385 0.5300 0.4784 0.7387 0.5253 0.4748 0.7305 0.5020 0.4520 0.7022 Qwen3-Embedding-4B 0.6327 0.5806 0.8388 0.6203 0.5669 0.8284 0.6183 0.5649 0.8262 0.6152 0.5625 0.8220 0.5891 0.5389 0.7880 Qwen3-Embedding-8B 0.6472 0.5961 0.8517 0.6415 0.5920 0.8430 0.6391 0.5901 0.8395 0.6350 0.5865 0.8342 0.6064 0.5592 0.8011 Omni-Embed-Nemotron-3B – – – 0.6380 0.5910 0.8302 0.6353 0.5888 0.8238 0.6264 0.5797 0.8160 0.5818 0.5360 0.7678

#### Table 13: Main experimental results on the DuRetrieval dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: DuRetrieval Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.5613 0.7000 0.5714 0.5109 0.6404 0.5257 0.5093 0.6386 0.5241 0.5012 0.6309 0.5153 0.4706 0.5952 0.4842 BGE-Small-(zh/en)-v1.5 0.7900 0.8776 0.8051 0.7551 0.8409 0.7753 0.7497 0.8369 0.7697 0.7444 0.8325 0.7653 0.6980 0.7831 0.7206 BGE-Base-(zh/en)-v1.5 0.8507 0.9102 0.8728 0.8198 0.8818 0.8429 0.8135 0.8755 0.8366 0.8062 0.8674 0.8313 0.7570 0.8180 0.7818 BGE-Large-(zh/en)-v1.5 0.8634 0.9201 0.8838 0.8348 0.8929 0.8581 0.8286 0.8867 0.8521 0.8227 0.8809 0.8466 0.7706 0.8272 0.7969 BGE-M3-dense 0.8397 0.9081 0.8626 0.8143 0.8844 0.8385 0.8106 0.8811 0.8352 0.8031 0.8746 0.8279 0.7534 0.8251 0.7793 EmbeddingGemma-300M 0.7879 0.8664 0.8131 0.7624 0.8423 0.7876 0.7574 0.8387 0.7827 0.7505 0.8292 0.7775 0.7086 0.7865 0.7358 Stella-EN-400M-v5 – – – – – – – – – – – – – – – All-MiniLM-L6-v2 – – – – – – – – – – – – – – – Multilingual-E5-Large 0.8487 0.9128 0.8663 0.8190 0.8852 0.8408 0.8137 0.8813 0.8349 0.8071 0.8755 0.8292 0.7554 0.8223 0.7790 Qwen3-Embedding-0.6B 0.8265 0.8965 0.8478 0.8021 0.8747 0.8245 0.7991 0.8725 0.8215 0.7911 0.8652 0.8137 0.7460 0.8188 0.7683 Qwen3-Embedding-4B 0.8884 0.9412 0.9020 0.8658 0.9223 0.8800 0.8615 0.9196 0.8755 0.8545 0.9129 0.8686 0.8121 0.8710 0.8270 Qwen3-Embedding-8B 0.8979 0.9477 0.9076 0.8768 0.9307 0.8868 0.8733 0.9273 0.8849 0.8670 0.9219 0.8785 0.8219 0.8781 0.8353 Omni-Embed-Nemotron-3B – – – 0.7612 0.8464 0.7889 0.7546 0.8420 0.7825 0.7470 0.8344 0.7764 0.6624 0.7514 0.6906

#### Table 14: Main experimental results on the MedicalRetrieval dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: MedicalRetrieval Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.3110 0.2937 0.3670 0.2796 0.2625 0.3350 0.2796 0.2624 0.3350 0.2744 0.2575 0.3290 0.2570 0.2406 0.2240 BGE-Small-(zh/en)-v1.5 0.4993 0.4738 0.5850 0.4707 0.4453 0.5560 0.4674 0.4401 0.5560 0.4612 0.4344 0.5480 0.4294 0.4037 0.5130 BGE-Base-(zh/en)-v1.5 0.5649 0.5355 0.6600 0.5358 0.5063 0.6330 0.5353 0.5052 0.6330 0.5271 0.4980 0.6210 0.4948 0.4678 0.5830 BGE-Large-(zh/en)-v1.5 0.5953 0.5692 0.6810 0.5629 0.5345 0.6570 0.5619 0.5320 0.6580 0.5546 0.5249 0.6500 0.5186 0.4912 0.6060 BGE-M3-dense 0.5424 0.5158 0.6280 0.4985 0.4703 0.5910 0.4980 0.4701 0.5900 0.4948 0.4669 0.5860 0.4604 0.4326 0.5530 EmbeddingGemma-300M 0.4990 0.4723 0.5840 0.4666 0.4389 0.5550 0.4669 0.4397 0.5540 0.4634 0.4359 0.5510 0.4324 0.4073 0.5140 Stella-EN-400M-v5 – – – – – – – – – – – – – – – All-MiniLM-L6-v2 – – – – – – – – – – – – – – – Multilingual-E5-Large 0.5609 0.5382 0.6430 0.5198 0.4947 0.6140 0.5186 0.4932 0.6130 0.5107 0.4862 0.6030 0.4753 0.4494 0.5700 Qwen3-Embedding-0.6B 0.5618 0.5356 0.6460 0.5251 0.4951 0.6210 0.5239 0.4939 0.6200 0.5182 0.4885 0.6130 0.4857 0.4584 0.5730 Qwen3-Embedding-4B 0.6207 0.5904 0.7180 0.5959 0.5643 0.6980 0.5943 0.5633 0.6940 0.5878 0.5564 0.6890 0.5514 0.5208 0.6500 Qwen3-Embedding-8B 0.6340 0.6043 0.7310 0.6090 0.5765 0.7160 0.6092 0.5774 0.7140 0.6034 0.5707 0.7110 0.5688 0.5349 0.6790 Omni-Embed-Nemotron-3B – – – 0.4930 0.4676 0.5740 0.4946 0.4695 0.5740 0.4808 0.4529 0.5700 0.4127 0.3876 0.4930

#### Table 15: Main experimental results on the T2Retrieval dataset under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: T2Retrieval Model Config

Text Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

BM25 0.5806 0.7330 0.5690 0.5236 0.6708 0.5157 0.5208 0.6677 0.5130 0.5183 0.6648 0.5105 0.4908 0.6327 0.4842 BGE-Small-(zh/en)-v1.5 0.7721 0.8824 0.7591 0.7216 0.8331 0.7128 0.7192 0.8305 0.7106 0.7149 0.8264 0.7060 0.6802 0.7911 0.6730 BGE-Base-(zh/en)-v1.5 0.8371 0.9214 0.8237 0.7914 0.8790 0.7817 0.7891 0.8762 0.7796 0.7843 0.8716 0.7753 0.7493 0.8368 0.7421 BGE-Large-(zh/en)-v1.5 0.8399 0.9204 0.8295 0.7942 0.8771 0.7881 0.7917 0.8745 0.7858 0.7864 0.8696 0.7807 0.7511 0.8347 0.7467 BGE-M3-dense 0.8138 0.9029 0.8050 0.7684 0.8595 0.7633 0.7655 0.8564 0.7603 0.7614 0.8529 0.7562 0.7243 0.8160 0.7203 EmbeddingGemma-300M 0.7987 0.8951 0.7879 0.7588 0.8555 0.7517 0.7565 0.8531 0.7501 0.7523 0.8490 0.7457 0.7155 0.8105 0.7101 Stella-EN-400M-v5 – – – – – – – – – – – – – – – All-MiniLM-L6-v2 – – – – – – – – – – – – – – – Multilingual-E5-Large 0.8341 0.9189 0.8212 0.7909 0.8786 0.7813 0.7887 0.8763 0.7791 0.7845 0.8725 0.7752 0.7468 0.8340 0.7398 Qwen3-Embedding-0.6B 0.8333 0.9199 0.8185 0.7943 0.8838 0.7829 0.7920 0.8812 0.7813 0.7882 0.8776 0.7773 0.7522 0.8416 0.7435 Qwen3-Embedding-4B 0.8718 0.9394 0.8595 0.8362 0.9067 0.8278 0.8338 0.9045 0.8256 0.8296 0.9003 0.8218 0.7943 0.8665 0.7874 Qwen3-Embedding-8B 0.8780 0.9424 0.8663 0.8423 0.9098 0.8345 0.8398 0.9075 0.8319 0.8355 0.9037 0.8278 0.8000 0.8694 0.7944 Omni-Embed-Nemotron-3B – – – 0.7402 0.8462 0.7347 0.7350 0.8423 0.7303 0.7243 0.8327 0.7203 0.6475 0.7551 0.6459

#### Table 16: Comparison of ASR robustness on the FiQA dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: FiQA ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

##### Retrieval Model: BM25

Fun-ASR-Nano-2512 0.2180 0.2780 0.2656 0.2122 0.2712 0.2598 0.2129 0.2723 0.2604 0.2031 0.2584 0.2510 GLM-ASR-Nano-2512 0.2141 0.2735 0.2657 0.2179 0.2771 0.2702 0.2171 0.2767 0.2693 0.2104 0.2655 0.2621 Qwen3-ASR-1.7B 0.2274 0.2870 0.2809 0.2204 0.2806 0.2705 0.2173 0.2757 0.2666 0.2150 0.2729 0.2653 SenseVoice-Small 0.2047 0.2616 0.2542 0.2011 0.2593 0.2486 0.1979 0.2556 0.2436 0.1891 0.2419 0.2366 Whisper-Large-v3 0.2335 0.2946 0.2869 0.2329 0.2943 0.2887 0.2333 0.2934 0.2903 0.2214 0.2787 0.2736 Whisper-Base 0.2163 0.2755 0.2722 0.2179 0.2765 0.2745 0.2133 0.2703 0.2677 0.1902 0.2454 0.2373 Whisper-Medium 0.2278 0.2882 0.2819 0.2312 0.2918 0.2871 0.2309 0.2911 0.2866 0.2139 0.2723 0.2640 Whisper-Small 0.2268 0.2864 0.2818 0.2306 0.2913 0.2837 0.2277 0.2885 0.2827 0.2026 0.2555 0.2523 Whisper-Tiny 0.2129 0.2714 0.2642 0.2138 0.2711 0.2682 0.2031 0.2612 0.2499 0.1701 0.2192 0.2092

##### Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.5923 0.6574 0.6824 0.5873 0.6529 0.6741 0.5907 0.6562 0.6801 0.5689 0.6339 0.6536 GLM-ASR-Nano-2512 0.5963 0.6689 0.6768 0.5930 0.6632 0.6762 0.5908 0.6617 0.6718 0.5700 0.6339 0.6548 Qwen3-ASR-1.7B 0.6030 0.6712 0.6899 0.5990 0.6689 0.6828 0.5959 0.6648 0.6808 0.5805 0.6426 0.6663 SenseVoice-Small 0.5732 0.6388 0.6592 0.5744 0.6390 0.6609 0.5716 0.6368 0.6548 0.5434 0.6038 0.6332 Whisper-Large-v3 0.5968 0.6653 0.6793 0.5949 0.6647 0.6786 0.5974 0.6678 0.6788 0.5723 0.6357 0.6571 Whisper-Base 0.5727 0.6409 0.6560 0.5713 0.6369 0.6589 0.5686 0.6357 0.6538 0.4975 0.5628 0.5740 Whisper-Medium 0.5917 0.6617 0.6739 0.5920 0.6602 0.6778 0.5890 0.6551 0.6734 0.5589 0.6226 0.6423 Whisper-Small 0.5871 0.6581 0.6681 0.5854 0.6541 0.6669 0.5817 0.6479 0.6634 0.5343 0.5972 0.6124 Whisper-Tiny 0.5669 0.6360 0.6432 0.5670 0.6321 0.6503 0.5514 0.6191 0.6316 0.4636 0.5271 0.5264

#### Table 17: Comparison of ASR robustness on the HotpotQA dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: HotpotQA ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

##### Retrieval Model: BM25

Fun-ASR-Nano-2512 0.4846 0.6342 0.5118 0.4864 0.6362 0.5136 0.4840 0.6334 0.5113 0.4582 0.6036 0.4845 GLM-ASR-Nano-2512 0.4821 0.6284 0.5123 0.4807 0.6269 0.5105 0.4814 0.6274 0.5117 0.4535 0.5930 0.4833 Qwen3-ASR-1.7B 0.5303 0.6904 0.5562 0.5187 0.6749 0.5457 0.5156 0.6706 0.5433 0.4958 0.6478 0.5228 SenseVoice-Small 0.4096 0.5412 0.4370 0.4065 0.5363 0.4350 0.4013 0.5292 0.4300 0.3768 0.4995 0.4030 Whisper-Large-v3 0.5464 0.7098 0.5727 0.5426 0.7040 0.5697 0.5399 0.7008 0.5667 0.5128 0.6678 0.5394 Whisper-Base 0.4646 0.6147 0.4872 0.4628 0.6118 0.4866 0.4554 0.6019 0.4792 0.3978 0.5276 0.4207 Whisper-Medium 0.5237 0.6820 0.5500 0.5241 0.6833 0.5500 0.5194 0.6772 0.5459 0.4873 0.6380 0.5138 Whisper-Small 0.5013 0.6566 0.5255 0.5000 0.6553 0.5240 0.4961 0.6506 0.5203 0.4580 0.6040 0.4820 Whisper-Tiny 0.4314 0.5715 0.4554 0.4299 0.5699 0.4540 0.4168 0.5523 0.4415 0.3450 0.4617 0.3661

##### Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.6551 0.8147 0.6830 0.6520 0.8088 0.6818 0.6476 0.8040 0.6782 0.6113 0.7630 0.6426 GLM-ASR-Nano-2512 0.6666 0.8223 0.6953 0.6657 0.8211 0.6945 0.6645 0.8196 0.6928 0.6253 0.7745 0.6552 Qwen3-ASR-1.7B 0.6802 0.8363 0.7082 0.6787 0.8364 0.7061 0.6771 0.8357 0.7034 0.6500 0.8046 0.6793 SenseVoice-Small 0.6115 0.7706 0.6421 0.6109 0.7700 0.6413 0.6049 0.7633 0.6356 0.5693 0.7210 0.6001 Whisper-Large-v3 0.6968 0.8550 0.7235 0.6931 0.8514 0.7200 0.6891 0.8458 0.7167 0.6574 0.8120 0.6850 Whisper-Base 0.6292 0.7869 0.6579 0.6280 0.7862 0.6562 0.6154 0.7718 0.6433 0.5359 0.6762 0.5658 Whisper-Medium 0.6775 0.8361 0.7045 0.6770 0.8362 0.7038 0.6735 0.8321 0.7008 0.6308 0.7842 0.6578 Whisper-Small 0.6627 0.8220 0.6895 0.6593 0.8173 0.6869 0.6559 0.8144 0.6834 0.6037 0.7549 0.6324 Whisper-Tiny 0.5996 0.7573 0.6262 0.5947 0.7515 0.6219 0.5791 0.7340 0.6066 0.4738 0.6071 0.5024

#### Table 18: Comparison of ASR robustness on the NQ dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: NQ ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

Retrieval Model: BM25

Fun-ASR-Nano-2512 0.2772 0.2390 0.4315 0.2772 0.2391 0.4318 0.2746 0.2369 0.4274 0.2609 0.2259 0.4043 GLM-ASR-Nano-2512 0.2805 0.2415 0.4368 0.2805 0.2414 0.4366 0.2795 0.2409 0.4342 0.2642 0.2274 0.4120 Qwen3-ASR-1.7B 0.2931 0.2521 0.4559 0.2891 0.2489 0.4492 0.2880 0.2478 0.4481 0.2763 0.2371 0.4324 SenseVoice-Small 0.2458 0.2098 0.3903 0.2493 0.2132 0.3937 0.2434 0.2078 0.3855 0.2259 0.1924 0.3605 Whisper-Large-v3 0.2959 0.2547 0.4602 0.2954 0.2549 0.4579 0.2934 0.2529 0.4551 0.2780 0.2403 0.4291 Whisper-Base 0.2632 0.2256 0.4129 0.2612 0.2236 0.4107 0.2576 0.2211 0.4031 0.2192 0.1889 0.3424 Whisper-Medium 0.2880 0.2482 0.4475 0.2876 0.2483 0.4459 0.2851 0.2455 0.4434 0.2682 0.2317 0.4162 Whisper-Small 0.2793 0.2403 0.4342 0.2789 0.2399 0.4335 0.2780 0.2391 0.4324 0.2543 0.2201 0.3922 Whisper-Tiny 0.2456 0.2103 0.3857 0.2453 0.2107 0.3840 0.2358 0.2027 0.3702 0.1880 0.1616 0.2960

Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.6286 0.5791 0.8306 0.6252 0.5760 0.8261 0.6238 0.5749 0.8234 0.5890 0.5422 0.7806 GLM-ASR-Nano-2512 0.6338 0.5850 0.8309 0.6332 0.5849 0.8299 0.6300 0.5820 0.8250 0.5956 0.5483 0.7871 Qwen3-ASR-1.7B 0.6388 0.5899 0.8385 0.6375 0.5877 0.8395 0.6342 0.5847 0.8352 0.6146 0.5668 0.8097 SenseVoice-Small 0.6126 0.5618 0.8162 0.6102 0.5593 0.8134 0.6058 0.5546 0.8093 0.5669 0.5174 0.7635 Whisper-Large-v3 0.6415 0.5920 0.8430 0.6391 0.5901 0.8395 0.6350 0.5865 0.8342 0.6064 0.5592 0.8011 Whisper-Base 0.6046 0.5559 0.8023 0.5994 0.5506 0.7979 0.5905 0.5418 0.7893 0.5109 0.4677 0.6880 Whisper-Medium 0.6274 0.5783 0.8283 0.6297 0.5805 0.8297 0.6256 0.5761 0.8261 0.5897 0.5419 0.7837 Whisper-Small 0.6239 0.5742 0.8253 0.6240 0.5745 0.8248 0.6194 0.5702 0.8187 0.5662 0.5196 0.7554 Whisper-Tiny 0.5921 0.5431 0.7896 0.5868 0.5372 0.7857 0.5689 0.5204 0.7638 0.4623 0.4232 0.6239

#### Table 19: Comparison of ASR robustness on the DuRetrieval dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: DuRetrieval ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

##### Retrieval Model: BM25

Fun-ASR-Nano-2512 0.5144 0.6453 0.5284 0.5115 0.6422 0.5255 0.5062 0.6374 0.5195 0.4842 0.6092 0.4992 GLM-ASR-Nano-2512 0.5034 0.6313 0.5174 0.5029 0.6314 0.5158 0.4959 0.6233 0.5088 0.4534 0.5711 0.4695 Qwen3-ASR-1.7B 0.5108 0.6389 0.5255 0.5108 0.6377 0.5257 0.5077 0.6347 0.5222 0.4859 0.6086 0.5002 SenseVoice-Small 0.4724 0.5974 0.4868 0.4713 0.6000 0.4845 0.4649 0.5917 0.4788 0.4282 0.5457 0.4442 Whisper-Large-v3 0.4401 0.5606 0.4533 0.4340 0.5542 0.4478 0.4245 0.5432 0.4373 0.3514 0.4521 0.3654

##### Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.8694 0.9217 0.8818 0.8684 0.9210 0.8805 0.8639 0.9164 0.8770 0.8297 0.8821 0.8438 GLM-ASR-Nano-2512 0.8590 0.9140 0.8723 0.8568 0.9120 0.8704 0.8515 0.9065 0.8630 0.7912 0.8463 0.8060 Qwen3-ASR-1.7B 0.8645 0.9196 0.8763 0.8649 0.9203 0.8768 0.8615 0.9169 0.8727 0.8321 0.8877 0.8453 SenseVoice-Small 0.8565 0.9146 0.8692 0.8530 0.9110 0.8650 0.8462 0.9030 0.8575 0.7993 0.8550 0.8142 Whisper-Large-v3 0.8036 0.8632 0.8165 0.7983 0.8597 0.8114 0.7880 0.8480 0.8002 0.6937 0.7535 0.7088

#### Table 20: Comparison of ASR robustness on the MedicalRetrieval dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: MedicalRetrieval ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

Retrieval Model: BM25

Fun-ASR-Nano-2512 0.2848 0.2676 0.3400 0.2852 0.2679 0.3410 0.2835 0.2660 0.3400 0.2656 0.2488 0.3200 GLM-ASR-Nano-2512 0.2679 0.2508 0.3230 0.2666 0.2492 0.3230 0.2650 0.2470 0.3230 0.2409 0.2235 0.2970 Qwen3-ASR-1.7B 0.2888 0.2713 0.3450 0.2866 0.2690 0.3430 0.2808 0.2633 0.3370 0.2697 0.2538 0.3210 SenseVoice-Small 0.2596 0.2443 0.3090 0.2557 0.2399 0.3070 0.2522 0.2368 0.3020 0.2351 0.2194 0.2860 Whisper-Large-v3 0.2236 0.2080 0.2740 0.2147 0.2000 0.2620 0.2132 0.1975 0.2640 0.1807 0.1672 0.2240

Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.5910 0.5598 0.6910 0.5945 0.5624 0.6970 0.5962 0.5642 0.6980 0.5724 0.5400 0.6760 GLM-ASR-Nano-2512 0.5741 0.5428 0.6740 0.5788 0.5476 0.6790 0.5749 0.5430 0.6770 0.5262 0.4944 0.6290 Qwen3-ASR-1.7B 0.5949 0.5634 0.6960 0.5935 0.5632 0.6910 0.5882 0.5577 0.6860 0.5666 0.5348 0.6690 SenseVoice-Small 0.5898 0.5562 0.7010 0.5854 0.5520 0.6960 0.5817 0.5495 0.6880 0.5583 0.5266 0.6630 Whisper-Large-v3 0.5301 0.4985 0.6300 0.5285 0.4968 0.6290 0.5299 0.4974 0.6330 0.4673 0.4336 0.5750

#### Table 21: Comparison of ASR robustness on the T2Retrieval dataset using BM25 and Qwen3-Embedding-8B retrievers under different acoustic conditions. The metrics nDCG, MRR, and Recall denote nDCG@10, MRR@10, and Recall@10, respectively. Best results are highlighted in bold.

Dataset: T2Retrieval ASR Model

Clean Low (20dB) Medium (10dB) High (0dB)

nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall nDCG MRR Recall

##### Retrieval Model: BM25

Fun-ASR-Nano-2512 0.5297 0.6765 0.5212 0.5289 0.6758 0.5203 0.5271 0.6743 0.5185 0.5040 0.6470 0.4963 GLM-ASR-Nano-2512 0.5201 0.6654 0.5123 0.5196 0.6651 0.5116 0.5159 0.6611 0.5084 0.4772 0.6143 0.4711 Qwen3-ASR-1.7B 0.5253 0.6710 0.5171 0.5242 0.6699 0.5160 0.5220 0.6677 0.5140 0.5037 0.6457 0.4966 SenseVoice-Small 0.4950 0.6395 0.4884 0.4912 0.6346 0.4846 0.4869 0.6298 0.4802 0.4568 0.5933 0.4518 Whisper-Large-v3 0.4517 0.5888 0.4480 0.4497 0.5870 0.4456 0.4391 0.5754 0.4348 0.3739 0.4922 0.3722

##### Retrieval Model: Qwen3-Embedding-8B

Fun-ASR-Nano-2512 0.8405 0.9074 0.8324 0.8395 0.9061 0.8317 0.8365 0.9039 0.8283 0.8032 0.8710 0.7961 GLM-ASR-Nano-2512 0.8246 0.8924 0.8185 0.8240 0.8920 0.8177 0.8185 0.8867 0.8123 0.7648 0.8328 0.7607 Qwen3-ASR-1.7B 0.8370 0.9040 0.8287 0.8360 0.9031 0.8281 0.8336 0.9006 0.8258 0.8067 0.8738 0.8003 SenseVoice-Small 0.8222 0.8918 0.8155 0.8189 0.8886 0.8125 0.8149 0.8852 0.8088 0.7752 0.8453 0.7711 Whisper-Large-v3 0.7719 0.8460 0.7685 0.7694 0.8437 0.7666 0.7607 0.8350 0.7587 0.6775 0.7509 0.6780

