# arXiv:2601.01554v5[cs.SD]19Jan2026

OpenMOSS

[Figure 1]

## MOSS Transcribe Diarize Technical Report

#### MOSI.AI*

#### Abstract

Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks.

Homepage: https://mosi.cn/models/moss-transcribe-diarize Online Demo: https://moss-transcribe-diarize-demo.mosi.cn

[Figure 2]

###### Figure 1 Overview of key capabilities of MOSS Transcribe Diarize

∗Full contributors can be found in the Contributors section.

#### 1 Introduction

Accurate transcripts of multi-speaker conversations are foundational to a wide range of applications, from meeting assistants and call-center analytics to assistive technologies and legal discovery [3, 4, 9]. In these settings, who said what, and when is as important as what was said: users need speaker-attributed, timestamped transcripts that preserve turn structure, overlaps, and long-range references across a discussion that may span tens of minutes. We refer to this task as Speaker-Attributed, Time-Stamped Transcription (SATS). Despite its practical importance, SATS is typically solved today by stitching together multiple components —automatic speech recognition (ASR) (e.g., Whisper [16]) and speaker diarization (e.g., Pyannote [2] or x-vector clustering [18]), each trained with different objectives and latencies, and often tuned on different datasets. Such modular pipelines are brittle: errors cascade across stages, global context is hard to leverage consistently, and end users must accept trade-offs between attribution accuracy, temporal precision, and throughput [20].

Recent advances in large language models (LLMs) and multimodal large language models (MLLMs) [6, 19, 22]suggestapathtowardunifiedsolutionsthatjointlymodelaudioandtext. However, mostexistingMLLMs are developed and evaluated mainly on single-speaker speech, achieving strong ASR performance but falling shortofourSATSsetting, which requiresjointlyrecognizing content, attributing it tospeakers, and providing time-stamped speaker turns in multi-speaker conversations.

A practical compromise between fully modular pipelines and fully end-to-end SATS is a semi-cascaded (or hybrid) scheme: a strong ASR and an acoustic diarization front-end are still used to produce candidate words and speaker traces, while an LLM-style model is introduced as a global reconciliation layer to resolve speaker permutations, repair boundary inconsistencies around turns/overlaps, and improve the readability and consistency of the final speaker-attributed transcript. For example, DiarizationLM post-processes the independent outputs from ASR and diarization systems with an LLM to refine speaker-attributed transcriptions, but it remains non end-to-end and thus inherits the classic error-propagation and mismatch issues of cascaded designs [20].

Motivated by the brittleness of modular ASR–diarization pipelines, recent work has begun to unify recognition and speaker attribution within a single multimodal framework, so that lexical modeling and speaker attribution can be learned jointly under shared context. This line of research moves SATS closer to an endto-end formulation and reduces cross-module mismatch. Sortformer [14] explores joint modeling with a permutation-invariant objective (Sort Loss) to better align speaker identity with lexical tokens, but its training remains two-stage: it first trains a diarization-only model and then freezes it to provide speaker traces as inputs for training an ASR model. As a result, it is not a truly end-to-end SATS formulation and can still suffer from cross-stage mismatch. SpeakerLM [21] is closer to a unified architecture by integrating speaker-aware modeling into a single MLLM, demonstrating improved speaker attribution without an explicit modular diarization stage; however, in their reported settings these models are still limited to relatively short audio contexts (on the order of 50–90s) and small speaker sets (e.g., up to 4 speakers), which restricts scalability to meeting-style, long-form conversations. Moreover, these approaches do not natively output explicit timestamped speaker segments (i.e., “who spoke when”) at the segment level, which is important for meetingstyle SATS.

To tackle long-form scalability within an MLLM-style framework, Shi et al. [17] propose JEDIS-LLM, an endto-end Speech-LLM for joint ASR and diarization that is trained only on short clips (≤20s) yet supports chunk-wise, streamable inference on long-form audio via a Speaker Prompt Cache (SPC) with on-the-fly updates; SPC also enables integrating pre-enrolled speaker profiles commonly used in meeting transcription. While these hybrid/streaming designs substantially improve long-audio scalability under latency constraints, they still rely on chunk-wise processing and additional mechanisms (e.g., cache management and segmentation/alignment [1, 13]) to maintain global speaker consistency. This further motivates our longcontext, single-passSATSformulationthat avoidschunkboundaries and nativelyemits timestampedspeaker turns.

[0.11] [S01] Good morning! [1.03] [1.11] [S02] Morning, guys! [1.34] [1.20] [S03] Hi, everyone! [1.44]

[Figure 3]

### Autoregressive SpeechLLMs

[Figure 4]

[Figure 5]

[Figure 6]

Audio Encoder

Trainable Parameter

Speech

Figure 2 Overall architecture of the MOSS Transcribe Diarize model.

These trends underscore that most existing SATS systems still fall short of a truly end-to-end formulation. Despite promising progress, current approaches are often constrained in three key aspects. First, limited context windows force long recordings to be processed in short chunks, which disrupts discourse continuity, weakens coreference resolution and speaker consistency, and introduces boundary artifacts for timestamping [1, 17]. Second, long-range speaker memory remains fragile: speaker identities may drift over dozens of turns, especially under similar voices or varying acoustic conditions [8, 18]. Third, many architectures cannot natively produce segment-level timestamps with the granularity needed for retrieval and skimmability, and thus rely on external alignment components that re-introduce modular error propagation [1, 13].

In this paper we address these limitations with MOSS Transcribe Diarize, a unified multimodal large language model designed to perform SATS in a single, end-to-end pass. MOSS Transcribe Diarize jointly recognizes words, attributes them to speakers, and assigns speaker timestamps, eliminating hand-offs between separate subsystems. The model is trained on extensive in-the-wild conversational audio—rich in accents, acoustic environments, overlaps, and domain shifts—to learn robust attribution and timing under realistic conditions. To preserve discourse and speaker coherence over long meetings, MOSS Transcribe Diarize is equipped with a 128k-token context window, allowing it to process inputs up to 90 minutes without chunking. This long-context capability enables the model to build and maintain global representations of participants and topics, improving both diarization consistency and the handling of far-apart references. Our contributions can be summarized as follows:

- • End-to-end SATS system. To our knowledge, MOSS Transcribe Diarize is the first unified multimodal model that jointly performs word recognition, speaker attribution, and timestamp prediction in a single forward pass.
- • 128k-token long-context modeling at meeting scale. MOSS Transcribe Diarize processes inputs up to 90 minutes within a 128k-token context window, preserving discourse continuity and long-range speaker memory without chunking; this reduces identity drift and boundary artifacts and improves SATS metrics across in-domain and out-of-domain evaluations.

#### 2 Model Architecture

As illustrated in Figure 2, MOSS Transcribe Diarize couples an audio encoder with a projection module that maps multi-speaker acoustic embeddings into the feature space of a pretrained text LLM, enabling the backbone to jointly align speaker identities with lexical content and perform unified, long-context modeling in a single end-to-end model [6, 19, 22].

Following recent work on textual token–based time encoding for long-context multimodal models [5, 16], we represent temporal information explicitly as formatted timestamp text inserted between audio encoder

###### Dataset Duration Range (s) Avg. Duration (s) Number of Speakers

AISHELL-4 Test 2195.4 – 2393.9 2290.6 5 – 7 Podcast 1528.7 – 3636.5 2658.9 2 – 11 Movies 0.418 – 29.888 11.526 1 – 6

Table 1 Statistical overview of the evaluation datasets.

chunks. This avoids binding temporal encoding to absolute positional indices, which become sparse and ineffective over long durations, and enables accurate timestamp generation over hour-scale audio with stable speaker attribution [1].

#### 3 Data Composition

- 3.1 Real Data

In this study, we conduct experiments on multilingual audio collected from the Internet. We sample a large number of speaker-containing clips from public corpora for training, covering a wide range of real-world multi-speaker scenarios. In particular, the AISHELL-4 dataset [7] comprises multi-speaker conversational recordings captured in meeting rooms, including both far-field overlapping audio and near-field recordings for each speaker. We use the averaged channel of the far-field signals for both training and evaluation. In addition, we further curated two datasets from podcasts and films to serve as test sets.

- 3.2 Simulated Data

To strengthen speaker attribution and timestamp prediction, and to cope with the scarcity of high-quality real-world recordings, we use simulated data during training. From our in-house corpus, we randomly sample a pool of single-speaker utterances to construct synthetic mixtures. Following previous work [15], we employ a controllable probabilistic simulator to construct synthetic multi-speaker conversational data. Specifically, for each synthetic dialogue, we first draw 2–12 distinct speakers and randomly select one utterance per speaker. Each selected utterance is then partitioned into contiguous word runs by sampling a segment count and log-normal weights; the resulting segments are placed on a single timeline with Gaussiandistributed inter-segment gaps, enforcing speaker alternation while permitting overlaps capped at 80 percent of the shorter segment. To improve perceptual continuity, segment boundaries are snapped to nearby lowenergy points and 50 ms cross-fades are applied. Following prior work [11], we augment the mixtures with real-world noise and reverberation, sampling SNRs uniformly from 0–15 dB.

#### 4 Evaluation

##### 4.1 Evaluation Setups

###### 4.1.1 Evaluation Datasets

To comprehensively evaluate our model’s performance, we use three diverse benchmarks. The AISHELL-4 Test set provides challenging, long-form audio from real-world conference scenarios. The Podcast set is composed of high-quality, multi-guest interviews from YouTube, using the platform’s available subtitles which provide both reference transcripts. The Movies dataset consists of short audio segments derived from online films and TV series, which are rich in multi-speaker overlapping scenarios. It primarily features Chinese and English, but also covers other languages and dialects, including Korean, Japanese, and Cantonese. All samples in this dataset were manually annotated by professionals to ensure high-quality ground truth. The two internally curated datasets, Podcast and Movies, will be open-sourced and publicly released on Hugging Face to facilitate further research. The statistical overview of the datasets is provided in Table 1.

Dataset Metric Doubao ElevenLabs GPT-4o Gemini 2.5 Pro Gemini 3 Pro MOSS Transcribe Diarize

CER (↓) 18.18 19.58 \ 42.70 \ 15.43 cpCER (↓) 27.86 37.95 \ 53.42 \ 20.04 Δcp (↓) 9.68 18.36 \ 10.72 \ 4.61

AISHELL-4

CER (↓) 7.93 8.50 \ 7.38 \ 4.46 cpCER (↓) 10.54 11.34 \ 10.23 \ 6.97 Δcp (↓) 2.61 2.85 \ 2.85 \ 2.50

Podcast

CER (↓) 9.94 11.49 14.37 15.46 8.62 7.50 cpCER (↓) 30.88 17.85 23.67 24.15 14.73 13.36 Δcp (↓) 20.94 6.37 9.31 8.69 6.11 5.86

Movies

Note: GPT-4o is omitted from the first two benchmarks due to its audio input constraint. Gemini 3 Pro is excluded due to its instability in adhering to the required output format for long audio inputs.

Table 2 Performance of MOSS Transcribe Diarize and other models. Best results are in bold.

###### 4.1.2 Metrics

We adopt a comprehensive set of metrics to evaluate our system on both Automatic Speech Recognition (ASR) and Speaker Diarization (SD). Our primary metrics include Character Error Rate (CER), concatenated minimum-permutation CER (cpCER), and their difference, Δcp.

CER measures the performance of the ASR component by comparing the predicted transcript against the ground-truth text, without regard to speaker identities, using the standard minimum edit distance [12].

cpCER jointly evaluates both ASR and SD. It compares the predicted speaker-attributed transcripts against the ground-truth speaker transcripts. To handle label permutation ambiguity, the cpCER is calculated by finding the optimal assignment of predicted speaker labels that yields the minimum edit distance [10, 12, 14], thus reflecting the overall system performance for the speaker-attributed recognition task.

Δcp is the difference between cpCER and CER (Δcp = cpCER − CER). This value isolates the performance degradation caused by speaker attribution errors, thereby serving as a reliable, transcript-based measure of SD performance.

All metrics are reported in percentage (%), where lower values indicate better performance.

##### 4.2 Performance

To benchmark our model against current industry standards, we evaluate its performance relative to the best closed source models including Doubao Speech Recognition Model2, ElevenLabs Scribe v13, GPT-4o Transcribe Diarize4, Gemini 2.5 Pro5, and Gemini 3 Pro6. The results are shown in Table 2.

Table 2 reports the performance of MOSS Transcribe Diarize in comparison with strong closed-source commercial systems across three representative benchmarks, covering long-form meeting recordings (AISHELL4), extended multi-speaker conversations (Podcast), and short, overlap-rich segments (Movies). Together, these benchmarks evaluate robustness with respect to audio duration, speaker cardinality, and conversational structure.

- 2https://www.volcengine.com/docs/6561/1354871
- 3https://elevenlabs.io/docs/models#scribe-v1
- 4https://platform.openai.com/docs/models/gpt-4o-transcribe-diarize
- 5https://ai.google.dev/gemini-api/docs/models?hl=zh-cn#gemini-2.5-pro
- 6https://ai.google.dev/gemini-api/docs/models?hl=zh-cn#gemini-3-pro

Across all datasets where evaluation is feasible, MOSS Transcribe Diarize consistently achieves the best overall performance in terms of cpCER and Δcp, indicating superior joint modeling of transcription and speaker attribution. On AISHELL-4, which consists of nearly 40-minute real-world meeting recordings, our model substantially outperforms all baselines in both CER and cpCER. More importantly, it exhibits a markedly smaller Δcp, demonstrating that speaker attribution errors introduce significantly less additional degradation compared to pure ASR errors. This highlights the effectiveness of long-context, end-to-end modeling in maintaining speaker consistency over extended conversations.

It is worth noting that GPT-4o and Gemini 3 Pro are unable to reliably process long-form audio inputs such as AISHELL-4 and Podcast under our evaluation protocol. GPT-4o is constrained by its audio input length, preventing complete transcription of these recordings, while Gemini 3 Pro frequently fails to generate valid outputs that adhere to the required speaker-attributed format for long audio inputs. As a result, these systems are omitted from the corresponding benchmarks. This limitation underscores a practical gap between nominal multimodal capability and deployable long-form SATS performance.

On the Podcast benchmark, which features high-quality but long-duration, multi-speaker discussions, MOSS Transcribe Diarize again achieves the lowest CER and cpCER among all evaluated systems. While several baselines demonstrate strong ASR accuracy, our model consistently yields the smallest Δcp, indicating more reliable speaker attribution under frequent turn-taking and long-range speaker re-entrance. This advantage is particularly important for real-world conversational analytics, where speaker identity coherence across long temporal spans is critical.

The Movies dataset presents a complementary challenge characterized by short utterances, rapid speaker alternation, and frequent overlaps. Even in this short-form setting, MOSS Transcribe Diarize outperforms all baselines in cpCER and Δcp. Notably, some commercial systems achieve competitive CER but suffer from substantially larger Δcp values, reflecting difficulties in resolving speaker attribution under dense overlap. In contrast, our model maintains a relatively small gap between CER and cpCER, indicating robust handling of speaker boundaries across diverse conversational regimes.

Overall, these results demonstrate that the advantages of MOSS Transcribe Diarize extend beyond improved word recognition accuracy. The consistently low Δcp across both long-form and short-form benchmarks confirms that the proposed end-to-end SATS formulation, together with long-context modeling, yields more reliable speaker-attributed, time-stamped transcripts than modular or semi-cascaded alternatives. Crucially, unlike several general-purpose multimodal models, MOSS Transcribe Diarize remains fully operational on hour-scale audio, making it particularly well suited for real-world meeting transcription and long-form conversational analysis.

#### 5 Conclusions

WeintroducedMOSSTranscribeDiarize, aunifiedaudio–textMLLMforSpeaker-Attributed, Time-Stamped Transcription (SATS) that jointly performs transcription, speaker attribution, and timestamp prediction in a single pass with a 128k-token context window. The model combines a speech encoder with a learned projection into a pretrained text LLM, and is trained on diverse in-the-wild conversations together with propertyaware simulated mixtures that model overlap, turn-taking, and acoustic variability. Across AISHELL-4, Podcast, and Movies, MOSS Transcribe Diarize outperforms strong closed-source systems in CER, cpCER, and Δcp, highlighting the effectiveness of long-context joint modeling and distribution-controlled simulation for end-to-end SATS at meeting scale. Future work includes streaming SATS, finer-grained timestamp evaluation, and broader multilingual robustness.

#### Contributors

Core Contributors Donghua Yu∗, Zhengyuan Lin∗, Chen Yang, Yiyang Zhang, Zhaoye Fei

Contributors Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, Yitian Gong, Yi Jiang, Muchen Li, Shimin Li, Songlin Wang, Wenxuan Wang, Yang Wang, Zhiyu Wu, Zhe Xu, Wenbo Zhang, Yuqian Zhang, Jie Zhu

Advisors Xipeng Qiu†

∗ Equal contribution. † Corresponding author.

#### References

- [1] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of long-form audio. arXiv preprint arXiv:2303.00747, 2023.

- [2] Hervé Bredin, Ruiqing Yin, Juan Manuel Coria, Gregory Gelly, Pavel Korshunov, Marvin Lavechin, Diego Fustes, Hadrien Titeux, Wassim Bouaziz, and Marie-Philippe Gill. Pyannote.audio: neural building blocks for speaker diarization. In ICASSP, pages 7124–7128, 2020.

- [3] Jean Carletta, Simone Ashby, Sophie Bourban, Mike Flynn, Maël Guillemot, Thomas Hain, Jaroslav Kadlec, Vasilis Karaiskos, Wessel Kraaĳ, Melissa Kronenthal, et al. The ami meeting corpus: A pre-announcement. In Machine Learning for Multimodal Interaction, 2005.

- [4] J. Chen, Y. Wang, S. Watanabe, J. Le Roux, and J. R. Hershey. Continuous speech separation: Dataset and analysis. In Proceedings of ICASSP, 2020.

- [5] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024.

- [6] Yunfei Chu, Zhifang Jin, Weihan Xu, Yuchen Wei, et al. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

- [7] Ying Fu, Li Cheng, Shuo Lv, Yuting Jv, Yu Kong, Yan Hu, Lei Xie, Chenglin Zhu, Jian Wu, Hao Bu, Xiong Xu, Jun Du, and Jian Chen. AISHELL-4: An open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario. In Proceedings of Interspeech, pages 3660–3664, 2021.

- [8] Yusuke Fujita, Takuya Yoshioka, Xuankai Chang, and Shinji Watanabe. End-to-end neural speaker diarization with self-attention. In Proceedings of ASRU, 2019.

- [9] Adam Janin, Don Baron, Jane Edwards, Dan Ellis, David Gelbart, Nelson Morgan, Barbara Peskin, Thilo Pfau, Elizabeth Shriberg, and Andreas Stolcke. The icsi meeting corpus. In Proceedings of ICASSP, 2003.

- [10] Naoyuki Kanda, Yashesh Gaur, Xiaofei Wang, Zhong Meng, and Takuya Yoshioka. Serialized output training for end-to-end overlapping speech recognition. In Interspeech, pages 2797–2801, 2020.

- [11] Federico Landini, Jan Profant, Mireia Díez, and Lukáš Burget. From simulated mixtures to simulated conversations as training data for neural diarization. In Proceedings of Interspeech, pages 143–147, 2022.

- [12] Vladimir I. Levenshtein. Binary codes capable of correcting deletions, insertions and reversals. Soviet Physics Doklady, 1966.

- [13] Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger. Montreal forced aligner: A trainable text-speech alignment system. In Proceedings of Interspeech, 2017.

- [14] T. Park, I. Medennikov, K. Dhawan, W. Wang, H. Huang, N. R. Koluguri, K. C. Puvvada, J. Balam, and B. Ginsburg. Sortformer: Seamless integration of speaker diarization and asr by bridging timestamps and tokens. arXiv preprint, 2024.
- [15] T. J. Park, H. Huang, C. Hooper, N. Koluguri, K. Dhawan, I. Medennikov, A. Jukic, J. Balam, and B. Ginsburg. Property-aware multi-speaker data simulation: A probabilistic modelling technique for synthetic data generation. In Proceedings of the CHiME-2023 Workshop, 2023.

- [16] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. ICML, 2023.

- [17] Mohan Shi, Xiong Xiao, Ruchao Fan, Shaoshi Ling, and Jinyu Li. Train short, infer long: Speech-llm enables zeroshot streamable joint asr and diarization on long audio. arXiv preprint arXiv:2511.16046, 2025.

- [18] David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, and Sanjeev Khudanpur. X-vectors: Robust dnn embeddings for speaker recognition. In Proceedings of ICASSP, 2018.

- [19] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhe Chen, et al. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289, 2023.

- [20] Q. Wang et al. Diarizationlm: Speaker diarization post-processing with large language models. In Proc. Interspeech, 2024.

- [21] H. Yin, Y. Chen, C. Deng, L. Cheng, H. Wang, C.-H. Tan, Q. Chen, W. Wang, and X. Li. Speakerlm: End-to-end versatile speaker diarization and recognition with multimodal large language models. arXiv preprint, 2025.
- [22] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yunfei Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In EMNLP, 2023.

## Appendix

#### Appendix Contents

A Additional Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- A.1 Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- A.2 Output Normalization for Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

#### A Additional Details

##### A.1 Evaluation Prompts Gemini 2.5 Pro & Gemini 3 Pro (AISHELL-4, Podcast, Movies)

<audio> 请将音频中的对话内容转换为文本，语言使用音频的语言，使用 [S1],[S2]... 等标签标注出所有说话 人。 比如：[S1] 你好 [S2] 你好 [S1] 你叫什么名字？[S2] 我叫小明。 只需要输出最终的结果，不需要输出其他任何内容，不需要输出换行符

MOSS Transcribe Diarize Prompts AISHELL-4 & Podcast <audio> 请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对 应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。 Movies <audio>

请将以下对话转录为文本，使用 [S1] [S2] 等说话人标签，对于音频中的事件，使用 [event] 标签表 示。富有情感的文本用 <emotion> 对应文本 </emotion> 表示，使用 <ovl> 标签表示音频有部分重 叠，<ins></ins> 标签表示音频有插入。自动检测音频的语言，说话人标签和 <ovl> <ins> 始终用英 文，event 和 emotion 跟随音频语言。

##### A.2 Output Normalization for Evaluation

To ensure fair comparison across systems, we apply the same text normalization to both predictions and references before computing CER/cpCER/Δcp. Given a raw string x, we perform the following steps:

- • Remove parenthetical content. We delete any text in parentheses (and the preceding whitespaces) using \s*\(.*?\).
- • Remove angle-bracket tags. We delete any substrings matching <.*?> (e.g., <emotion>...</emotion>, <ovl>, <ins>...</ins>).
- • Remove non-speaker square-bracket annotations. We delete any square-bracketed spans [...] that are not speaker IDs, using the regex \[(?!S\d+\]).*?\]. This keeps only speaker tags of the form [S1], [S01], etc., and removes other bracketed markers such as events (e.g., [event]).

After normalization, each hypothesis/reference contains only speaker identifiers and plain transcript text for scoring.

