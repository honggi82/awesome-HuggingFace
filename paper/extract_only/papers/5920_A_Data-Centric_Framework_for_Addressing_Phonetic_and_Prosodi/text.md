# arXiv:2507.13563v3[cs.CL]23Jun2026

## Balalaika: Data-Centric, Prosody-Aware Annotation Pipeline for Russian Speech

Kirill Borodin

ID 1,2,∗,∗∗, Nikita Vasiliev

ID 1, Mikhail Gorodnichev

ID 1,∗, Vasiliy Kudryavtsev

ID 1,2, Maxim Maslov

ID 1, Grach Mkrtchian

ID 1

1 lab260, Moscow Technical University of Communications and Informatics, Russia 2 BitmanagerAI, UAE

kborodin.research@gmail.com

### Abstract

We introduce Balalaika, an open-source, data-centric pipeline for processing audio and producing prosody-aware annotations. It combines semantic VAD for context-preserving segmentation, multi-ASR ensembling with ROVER consensus decoding, while retaining word-level timestamps, followed by automatic quality and speaker-purity filtering. The text is further enriched with punctuation restoration, lexical stress and e/yo normalization, and IPA phonemes. Using Balalaika, we build a 5.1khour multi-source Russian corpus with rich annotations, and show consistent gains under equalized training budgets for both speech denoising and TTS; ablations confirm complementary benefits of stress and punctuation and improved synthesis with stricter MOS filtering.

The dataset is publicly available at https://hf.co/collections/lab260/balalaika-dataset

The full version paper is available at https://arxiv.org/abs/2507.13563 Index Terms: audio mining, annotation, TTS, speech quality, dataset, Balalaika

### 1. Introduction

The rapid growth of web audio [1] and voice interfaces [2] makes scalable mining of speech content one of the core challenges. As these technologies become integral to daily interaction [3, 4], they necessitate richly annotated, high-quality audio to ensure intelligibility and naturalness [5, 6].

However, significant gaps hinder effective use of web audio for advanced applications: dataset creation still leans on manual curation [7, 8, 9] and audiobooks [10, 11, 12], which is especially limiting for under-resourced languages with scarce highquality data and tooling [7, 13]. Russian presents unique complexities [14, 15] (vowel reduction [16], palatalization [16], and mobile stress [17] affecting meaning and prosody) that prevailing methods often overlook. As a result, pipelines optimized for high-resource languages tend to ignore phonetic and prosodic nuance, critical for morphologically rich languages [18, 13].

Gaps in scalable annotation and dataset frameworks critically undermine speech technologies, particularly in multilingual contexts. Simplistic annotation limits TTS quality [19, 20], producing unnatural outputs due to poor handling of stress [17] and phonetic nuances [16]. Furthermore, reliance on audiobooks introduces rigid, dictated patterns that lack spontaneous prosody [21, 22], ultimately degrading the performance and accuracy of generative speech tasks [23].

- *These authors contributed equally.
- **indicates the corresponding author.

Beyond technical limitations, these gaps create barriers to multilingual accessibility [24], excluding vast audio resources from integration into web indexes [18], which perpetuates biases toward high-resource languages [25]. They also contribute to stagnation in AI research, as models trained on inadequately annotated data underperform in speech processing tasks [26], ultimately hindering equitable access to speech applications.

To address these challenges, we introduce Balalaika, a scalable framework for processing Russian audio, enabling highquality datasets that capture prosody and linguistic complexities for generative models. This open-source methodology is novel in bridging gaps for Russian, as demonstrated by our creation of the dataset comprising over 5000 hours of high-quality speech. It supports audio mining through a scalable annotation pipeline, mitigating language biases in dataset creation.

To guide our investigation, we address the following research questions (RQs), which explore scalable annotation pipeline and its impact on generative speech models:

#### RQ1 (Framework Comparison): How does our annota-

tion framework compare to existing methods in producing highquality datasets for Russian speech?

#### RQ2 (Denoising Performance): How does training speech

denoising models from scratch on our annotated data, under equal conditions, compare to training on other datasets?

#### RQ3 (TTS performance): How does training text-to-

speech (TTS) models from scratch on our annotated data, under equal conditions, compare to training on other datasets in terms of naturalness and intelligibility?

To investigate these RQs, we built an open-source, modular pipeline that combines semantic VAD [27] for segmentation, multi-ASR ROVER fusion [28, 29, 30] with word-level timestamps, NISQA-based quality [31] and pyannote speaker filtering [32, 33], and prosody-aware text enrichment (punctuation [34], lexical stress [17], and G2P phonemes); Section 2 details each stage.

Our experiments reveal that models trained on data obtained using our proposed Balalaika framework achieve higher scores across different metrics in voice generative tasks compared to baselines, while handling Russian prosody competitively. These results demonstrate that the derived data notably improves the naturalness of the trained models.

### 2. Algorithmic Framework

#### 2.1. Audio segmentation

The initial corpus consists of raw audio signals that are unsuitable for direct modeling due to their excessive length. Since arbitrary splitting can disrupt semantic and phonetic continuity, we employ SmartTurnV3.1 [27], a semantic Voice Activity De-

- Table 1: Datasets comparison. The Hours column indicates the total dataset duration. The Speech column denotes the speech type: S (spontaneous), D (dictated), B (books), or Mixed. The Annotation column describes the annotation methods: sub (subtitles), man (manual), scr (script), pun (punctuation), mixed text (diverse sources), ASR (automatic speech recognition), str (stresses), ph (IPA phonemes), and ts (word-level timestamps). The highest and second-highest scores are shown in bold and underlined text, respectively.

Dataset Hours Speech Annotation NOI COL DIS LOU NMOS UTMOS MOS ± 95% CI TMR ± 95% CI

DeepSpeech [35] 6000 S sub 3.273 3.949 3.4 3.848 3.397 2.483 2.953± 0.078 0.701 ±0.084 GOLOS-C [7] 1095 D man 2.889 3.86 3.237 3.313 3.043 2.393 2.713 ± 0.077 0.76 ± 0.069 GOLOS-F [7] 132 D man 3.293 3.589 2.48 1.597 1.447 1.852 1.477 ±0.142 0.733 ± 0.072

M-AILABS [10] 46.8 B scr+ pun 4.149 3.818 4.012 3.989 3.966 2.765 3.967 ±0.114 0.767± 0.077 OpenSTT [36] 20108 mixed mixed text 2.588 3.661 2.928 3.191 2.723 2.033 2.633 ± 0.127 0.759 ± 0.061

RuLS [11] 98 B scr + pun 3.93 4.066 3.757 4.15 3.892 2.8162 3.8 ± 0.108 0.71 ± 0.0737 RUSLAN [37] 31 B scr+pun 3.827 4.266 3.698 4.168 3.788 2.934 4.049±0.041 0.766 ±0.063

MCV [38] 286 D scr+pun 3.511 3.859 3.391 3.755 3.37 2.4456 3.875 ± 0.125 0.717 ± 0.072 SOVA AB [12] 298 B scr 3.193 3.694 3.212 3.511 3.096 2.215 2.8 ±0.073 0.74 ± 0.071 SOVA YT [12] 17451 S sub 2.888 3.453 2.854 3.167 2.632 2.567 2.874±0.095 0.705 ± 0.082 SOVA D [12] 191 D scr 2.859 3.846 3.15 3.442 3.054 2.1883 2.803 ± 0.086 0.76 ± 0.069

Balalaika (ours) 5078 mixed ASR+pun+str+ph+ts 4.235 4.54 4.132 4.391 4.455 3.019 4.601 ± 0.09 0.767 ± 0.074

tection (VAD) model, to identify complete utterances based on context. We filter the resulting segments to ensure data quality, removing any utterances where the speech share is less than 70% or where internal silence exceeds 1 s. Finally, to optimize input constraints for downstream tasks, we group consecutive segments sk into chunks capped at 15 seconds, targeting a balanced duration between 5 and 15 seconds per chunk.

#### 2.2. Audio filtering

Following segmentation, we apply automatic quality and speaker-purity screening to discard noisy, clipped, and overlapping speech, and to retain only clean single-speaker material for training and evaluation. We first remove utterances shorter than 3 s, then reject segments with excessive impulsive energy by filtering out those whose CREST-factor exceeds a threshold 10 [39, 40]. Next, we predict perceptual quality with the NISQA-S model [41] configured for MOS estimation and discard all segments with MOS (denoted as qk in tab. 3) < 4.2. Finally, we use the pyannote speaker diarization pipeline1 [33, 32] to remove any segment containing overlapped speech or two or more speakers, ensuring that each retained utterance contains exactly one active speaker.

#### 2.3. Transcription

This stage maps each segment to text by ensembling heterogeneous ASR models and decoding strategies to reduce modelspecific errors. For each segment, we generate five hypotheses: (i) GigaAM-CTC-v3 [28] with CTC decoding, (ii) the same GigaAM-CTC-v3 decoded with an external n-gram language model2 [29] to strengthen lexical priors, also producing wordlevel timestamps, (iii) GigaAM-RNNT-v3 [28], (iv) the Russian Vosk model [29], and (v) T-one [30].

The final transcript is obtained using recognizer output voting (ROVER) [42], which aligns all hypotheses into a confusion network and selects a consensus word sequence to minimize word errors. This design combines complementary strengths of CTC vs. RNN-T search/alignment, LM-augmented decoding, and independent model diversity (Vosk and T-one), yielding a more stable transcript for downstream processing.

- 1https://huggingface.co/pyannote/

speaker-diarization-3.1

- 2https://alphacephei.com/vosk/models/

vosk-model-ru-0.22-compile.zip

#### 2.4. Timestamps

We provide word-level timestamps using the CTC+LM decoding output, which yields word-aligned time boundaries. For each segment, we store the ROVER consensus transcript as the primary text annotation, and additionally keep the CTC+LM hypothesis together with its word-level timestamps. This allows downstream applications to use timestamps when needed, while retaining a single, stable transcript for text-based training and evaluation.

#### 2.5. Punctuation

We enrich each ASR-consensus transcript with punctuation to encode prosodic structure, which is important for natural Russian TTS intonation. Using RuPunctBig [34], we restore sentence-final and intra-sentential marks directly from the unpunctuated word sequence produced upstream. In our pipeline, explicit punctuation serves as a lightweight proxy for phrase breaks and discourse structure, and is consistently associated with improved intonational naturalness in synthesized speech (Sec. 4.3).

#### 2.6. Stress placement and e˝-normalization

We enrich each punctuated transcript with lexical stress marks and resolve the Russian e/e˝ ambiguity to support prosody modeling and homograph disambiguation (Sec. 4.3). We apply RuAccent [17] for context-aware stress assignment and to restore e˝ when required by pronunciation. This yields a pronunciationconsistent text representation that improves downstream synthesis stability and naturalness.

#### 2.7. Phonemization

This stage converts stress-normalized text into IPA phoneme sequences while preserving punctuation and stress marks. We apply a word-level G2P model that transliterates graphemes to phonemes and copies non-graphemic symbols unchanged.

Our G2P is a lightweight transformer encoder–decoder [43] trained with a BPE tokenizer on graphemes and an IPA target inventory derived from Wiktextract pronunciations [44]. The model uses d model = 128, d ff = 512, 3 encoder and 3 decoder layers, 4 attention heads, and a 64-token limit; we train with AdamW (lr 3 × 10−4), batch size 256, label smoothing 0.1 for 10 epochs, and decode greedily at inference. Using IPA targets improves coverage of Russian phonology (e.g., vowel reduction and consonant devoicing), yielding stable phoneme

sequences for downstream modeling.

#### 2.8. Dataset: Balalaika

Balalaika is assembled from multiple publicly available Russian speech and text resources; we provide the full provenance and per-source breakdown (including Yandex Music, OpenSTT [36], ESpeech [45], GOLOS [7], DeepSpeech [35], and Biggest Russian Books [46]). The data was processed by our annotation framework, resulting in complete multi-layer annotations for all retained segments. The total size of the dataset is approximately 5078 hours of audio.

We provide all necessary scripts to reproduce the dataset’s annotations from the original sources, (for Yandex Music split)3, and we release the complete end-to-end annotation pipeline4. Original audio remains distributed under the respective source licenses.

3. Experimental Setup

#### 3.1. MOS Evaluation

We collected human ratings using LabelSpeech5 [47]. Raters assigned MOS on a 0–5 scale (0: non-speech; 1: low-quality telephony; 2: very low quality; 3: unprofessional recording; 4: studio quality with artifacts; 5: perfect studio quality).

Since we posit that additional annotations affect the quality of synthetic speech, we additionally collected an intonationfocused MOS (IntMOS) to capture prosodic naturalness beyond overall quality. Raters assigned MOS on a 0–5 scale (0: unintelligible; 1: clearly unnatural; 2: mostly robotic; 3: indeterminate; 4: dictation-like; 5: natural conversational speech).

Each audio item in our human-feedback evaluations received at least seven independent ratings from native speakers of Russian. For each clip, we aggregated ratings by taking the median. System-level scores were computed as the mean across clips, and we report 95% confidence intervals (CI). IntMOS is reported alongside MOS to isolate prosodic and intonational aspects that are specifically targeted by our annotation layers, providing a direct test of their contribution to perceived naturalness.

#### 3.2. RQ1: Framework Comparison

We evaluate RQ1 by comparing datasets. For objective assessment, we apply the original NISQA model [31], reporting its five metrics on each dataset: noisiness (NOI), coloration (COL), discontinuity (DIS), loudness (LOU), and overall MOS (NMOS). We additionally report the UTokyo-SaruLab MOS Prediction System(UTMOS) [48]. For human evaluation, we use classical MOS. We also quantify transcript fidelity via a manual Text Match Rate (TMR), defined as the percentage of clips whose displayed text matches what is spoken according to rater judgment. To keep comparisons balanced, we draw 200 items per dataset for MOS and TMR, while NISQA and UTMOSv2 are computed over the full available audio. We compare against 11 public Russian corpora [35, 7, 10, 36, 11, 37, 12, 38].

#### 3.3. RQ2: Speech Denoising Performance

To isolate the effect of data quality, we train SEMamba [49] from scratch on each dataset under an identical data/training budget. All runs use the same recipe (Adam, lr=5×10−4,

- 3https://hf.co/datasets/lab260/Balalaika2000H
- 4https://github.com/lab260ru/balalaika
- 5https://github.com/lab260ru/LabelSpeech

batch=8, 50k steps) and are evaluated at the final checkpoint (no early stopping). Training uses 25h randomly subsampled per dataset (disjoint from test) with MUSAN noise [50] and RIR augmentation [51] applied uniformly.

The test benchmark contains 3,000 samples: 500 clips each from M-AILABS, RUSLAN, RuLS, and from our dataset, drawn from their held-out test splits to ensure no leakage. In accordance with the methodology of the original paper [49], the following objective metrics were selected for evaluation: prediction of the signal distortion(CSIG) [52], prediction of the background intrusiveness(CBAK) [52], prediction of the overall speech quality(COVL) [52], perceptual evaluation of speech quality(PESQ) [53], short-time objective intelligibility measure(STOI) [54]. Along with this, we used virtual speech quality objective listener(VISQOL) [55] to evaluate audio quality. To evaluate the signal distortion ratio, we used scale invariant signald distortion ratio(SI-SDR).

#### 3.4. RQ3: Text-to-Speech performance

To isolate the effect of training data, we train a single VITS model [56] on each dataset under an equalized data and training budget: 25h of audio and a fixed recipe (Adam, lr=10−4, batch=32, 100k steps), evaluated at the final checkpoint (no early stopping).

Evaluation uses a heterogeneous held-out set of 2,000 texts from Balalaika’s multi-source test subsets, disjoint from the sampled training subsets and shared across systems. Objective quality is assessed with NISQA and NISQA-TTS (TTS MOS) [57]; we also report UTMOS. Intelligibility is measured by Character Error Rate (CER), where predicted text computed by transcribing synthetic audio with GigaAMv3-RNNT [28]. Subjective quality is measured with classical MOS and IntMOS on a randomly sampled 200-text subset.

We analyze how individual annotation layers affect TTS quality by training models on the same audio while varying only the text annotations. Using the pipeline and evaluation protocol from Sec. 3.4, we compare six settings - plain transcripts, +stress, +punctuation, +stress+punctuation, and two different MOS filtering thresholds - to isolate the impact of each annotation choice on naturalness and intelligibility.

4. Results and Discussion

#### 4.1. RQ1: Framework Comparison

Our annotation framework produces higher-quality Russian speech datasets than existing methods in perceptual quality and text–audio consistency, Table 8. The unified Balalaika dataset attains the best scores across all objective predictors and human MOS and IntMOS, outperforming public corpora such as M-AILABS [10], RuLS [11], and RUSLAN [37] on every reported metric; manual TMR likewise ranks it at the top, supporting the reliability of the multi-ASR fusion and confirming that the pipeline yields superior audio quality and transcript fidelity through effective annotation.

#### 4.2. RQ2: Speech Denoising Performance

Training denoisers from scratch on our data yields superior performance to models trained on alternative Russian datasets, with the SEMamba [49] trained on our dataset achieving the top score across most objective metrics in Table 4, indicating simultaneous gains in perceived signal quality, background suppression, overall quality, and intelligibility compared to sys-

- Table 2: Comparison of speech synthesis models trained on different datasets. The highest and second-highest scores for the metrics are shown in bold and underlined text.

Dataset NOI COL DIS LOU NMOS TTS MOS UTMOS MOS ± 95% CI IntMOS ± 95% CI CER

DeepSpeech [35] 2.6946 3.6406 2.6819 3.5315 2.7796 2.4468 1.4871 1.597 ± 0.078 0.62 ± 0.098 0.7693 GOLOS-C [7] 2.2051 3.6959 2.1849 1.9105 1.7415 1.9672 0.9203 0.615 ± 0.064 0.034 ± 0.04 0.9999 GOLOS-F [7] 3.4367 3.6482 2.4353 1.4922 1.1814 1.6324 0.8445 0.036 ± 0.024 0.013 ± 0.026 1

M AILABS [10] 3.7924 3.3723 3.6128 3.8226 3.5301 3.0321 2.30653 2.962 ± 0.052 2.208 ± 0.071 0.0908 RUSLAN [37] 4.0193 4.3278 3.7862 4.2483 3.7438 1.9134 2.12968 3.253 ± 0.068 3.182 ± 0.094 0.0496 OpenSTT [36] 2.1172 3.7438 2.5426 3.1770 2.4066 1.6217 1.6762 1.259 ± 0.058 0.135 ± 0.053 0.96

RuLS [11] 3.9336 4.2351 3.5956 3.9382 3.7876 2.9269 2.1068 2.75 ± 0.094 2.142 ± 0.086 0.1003 MCV [38] 4.0167 4.2227 3.6504 3.9415 3.7557 3.2095 2.12294 2.749 ± 0.062 2.462 ± 0.094 0.238

SOVA AB [12] 2.9993 4.1601 2.9049 3.5548 2.9691 2.5572 1.49 1.354 ± 0.063 0.156 ± 0.049 0.9112 SOVA YT [12] 2.0664 3.0234 1.5248 2.5048 1.4161 1.5147 0.8036 0.979 ± 0.016 0.018 ± 0.017 0.9998

SOVA D [12] 2.3487 4.0085 2.8124 3.4032 2.7414 2.2699 1.43 1.336 ± 0.039 0.196 ± 0.062 0.8714 Balalaika (ours) 4.2002 4.5799 4.3504 4.4632 4.4843 3.5703 2.73848 3.618 ± 0.083 2.532 ± 0.09 0.1062

- Table 3: Ablation study of different configurations of our dataset. The highest and second-highest scores for the metrics are shown in bold and underlined text.

Dataset NOI COL DIS LOU NMOS TTS MOS UTMOS MOS ± 95% CI IntMOS ± 95% CI CER ours (MOS> 4.2) 4.0555 4.5158 4.2646 4.3337 4.3456 3.3155 2.64739 3.41 ± 0.081 2.305 ± 0.088 0.1347

ours + stresses (qk > 4.2) 4.1358 4.5275 4.2979 4.4175 4.4446 3.5472 2.73128 3.522 ± 0.082 2.48 ± 0.094 0.1291 ours + punctuation (qk > 4.2) 4.1533 4.547 4.3249 4.4148 4.4326 3.5671 2.63233 3.44 ± 0.079 2.448 ± 0.089 0.1123

ours + stresses + punctuation (qk > 3) 2.4276 4.2440 3.8387 3.5028 2.6816 2.708 2.17835 2.446 ± 0.079 1.877 ± 0.094 0.2704

- ours + stresses + punctuation (qk > 3.5) 3.3053 4.3606 4.0656 3.9652 3.6308 3.5002 2.48488 3.256 ±0.095 2.377 ± 0.095 0.171
- ours + stresses + punctuation (qk > 4.2) 4.2002 4.5799 4.3504 4.4632 4.4843 3.5703 2.73848 3.618 ± 0.083 2.532 ± 0.09 0.1062

- Table 4: Comparison of speech denoising models trained on different datasets. The highest and second-highest scores for the metrics are shown in bold and underlined text.

bined MOS, objective metrics, and CER yield the strongest naturalness - intelligibility trade-off. Prosody-aware annotations provide additional gains: Table 3 shows that punctuation and stress are most effective jointly, improving MOS/IntMOS and reducing CER compared to either layer alone.

Dataset CSIG CBAK COVL PESQ VISQ UTM STOI SISDR test subset 2.926 2.506 2.127 1.456 3.523 2.250 0.877 7.582 DeepSpeech [35] 3.678 2.800 3.154 2.541 3.957 2.406 0.923 8.577 GOLOS-C [7] 3.715 2.684 3.190 2.504 3.974 2.409 0.922 8.374 GOLOS-F [7] 3.760 2.663 3.170 2.350 3.946 2.380 0.917 7.786 M AILABS [10] 3.778 3.084 3.287 2.721 4.012 2.594 0.930 8.257 RUSLAN [37] 3.530 2.893 3.024 2.476 3.804 2.621 0.878 5.833 OpenSTT [36] 3.688 2.953 3.195 2.634 3.931 2.425 0.924 8.051 RuLS [11] 3.740 2.912 3.204 2.584 3.981 2.537 0.925 8.717 MCV [38] 3.785 3.116 3.228 2.651 4.005 2.520 0.926 8.113 SOVA AB [12] 3.780 3.088 3.258 2.644 4.032 2.420 0.928 8.744 SOVA YT [12] 3.678 2.970 3.188 2.610 3.974 2.414 0.926 8.170 SOVA D [12] 3.743 2.854 3.227 2.582 3.987 2.471 0.924 8.317 Balalaika (ours) 3.856 3.165 3.340 2.723 4.036 2.614 0.931 8.809

Finally, stricter quality filtering improves synthesis: raising the MOS threshold from 3.5 to 4.2 increases predicted and human-rated naturalness, improves IntMOS, and lowers CER, whereas MOS>3 degrades all metrics, suggesting low-quality audio artifacts propagate to both prosody and intelligibility.

#### 4.4. Limitations

All systems were trained under the same fixed data and compute budget and were not run to full convergence. This ensures a fair cross-dataset comparison, but some models may remain undertrained and could achieve higher absolute scores with longer training.

The current pipeline is also Russian-specific, relying on language-dependent components (ASR, punctuation restoration, stress placement, and G2P), which limits direct transfer to other languages. However, the pipeline is modular, so extending it to new languages reduces to swapping in comparable tools as they become available.

tems trained on M-AILABS [10], RuLS [11], MCV [38], OpenSTT [36], and SOVA variants [12].

These gains are mirrored in SI-SDR, where the model trained on our unified dataset achieves the strongest distortion reduction; models trained on our data remain competitive across metrics and generally surpass counterparts trained on other corpora, underscoring that denoising benefits from the higher-quality data produced by the framework.

The TTS test set is a heterogeneous held-out split of Balalaika shared by all systems. Because several source datasets are also represented in Balalaika, models trained on those datasets may benefit from partial domain alignment. The experiment therefore provides a controlled comparison on a common test set, but not a fully source-independent evaluation.

#### 4.3. RQ3: Text-to-Speech performance

Training TTS models on Balalaika yields the highest objective quality scores and human MOS in Table 2, while maintaining competitive CER. RUSLAN achieves the highest IntMOS, whereas Balalaika ranks second; several other differences in subjective scores are smaller than the corresponding confidence intervals. These results indicate improved overall naturalness, but do not establish a strict ordering among systems with overlapping intervals.

### 5. Conclusion

Balalaika combines semantic segmentation, quality and speaker filtering, multi-ASR ROVER transcription, and prosody-aware text enrichment to build a 5.1k-hour Russian corpus. Under equalized budgets, its data improves denoising and TTS, with stress plus punctuation giving the strongest synthesis quality; we release the pipeline and reproducibility scripts.

For intonational naturalness, our system ranks second only to the single-speaker RUSLAN model, consistent with the expressivity advantage of single-speaker data; overall, the com-

### 6. Generative AI Use Disclosure

Generative AI tools were used only for language editing and manuscript polishing (e.g., improving clarity, grammar, and style). They were not used to generate substantial technical content, including the proposed methods, experiments, results, figures/tables, or conclusions. All authors reviewed and approved the final manuscript and take full responsibility for the content.

### 7. References

- [1] Edison Research, “The infinite dial 2025,” Edison Research, Tech. Rep., March 2025, presented by Megan Lazovick and James Cridland; Supported by Audacy, Cumulus Media, and SiriusXM Media. [Online]. Available: https://www.edisonresearch.com/ the-infinite-dial-2025/
- [2] SNS Insider, “Voice user interface market size and growth report 2032,” SNS Insider, Tech. Rep., 2025, market analysis with forecasts from 2025 to 2032. [Online]. Available: https: //www.snsinsider.com/reports/voice-user-interface-market-7025
- [3] J. Hombeck, H. Voigt, and K. Lawonn, “Voice user interfaces for effortless navigation in medical virtual reality environments,” Computers and Graphics, vol. 124, p. 104069, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0097849324002048
- [4] S. M. Wasti, K. Q. Pu, and A. Neshati, “Large language user interfaces: Voice interactive user interfaces powered by llms,” in Intelligent Systems and Applications, K. Arai, Ed. Cham: Springer Nature Switzerland, 2024, pp. 639–655.
- [5] P. S. Ravindra, “Advancing naturalness and intelligibility in text-to-speech systems using artificial intelligence,” Gap Bodhitaru Journal, 2024. [Online]. Available: https://www. gapbodhitaru.org/res/articles/(164-171)%20ADVANCING% 20NATURALNESS%20AND%20INTELLIGIBILITY% 20IN%20TEXT-TO-SPEECH%20SYSTEMS%20USING% 20ARTIFICIAL%20INTELLIGENCE.pdf
- [6] A. T. Team, “Benefits of audio annotation for multilingual speech recognition,” ATL Translate, Tech. Rep.,

2023. [Online]. Available: https://www.atltranslate.com/ai/blog/ audio-annotation-multilingual-speech-recognition-benefits

- [7] N. Karpov, A. Denisenko, and F. Minkin, “Golos: Russian Dataset for Speech Research,” in Proc. Interspeech 2021, 2021, pp. 1419– 1423.
- [8] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, 2020, pp. 4218–4222. [Online]. Available: https: //aclanthology.org/2020.lrec-1.520/
- [9] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “Libritts: A corpus derived from librispeech for textto-speech,” in Interspeech 2019, 2019, pp. 1526–1530.
- [10] I. Celeste, “The m-ailabs speech dataset,” 2019, a large free dataset containing nearly 1000 hours of audio across 8 languages for speech recognition and synthesis. [Online]. Available: https://github.com/imdatceleste/m-ailabs-dataset
- [11] “Russian librispeech (ruls) dataset,” https://openslr.org/96/, 2021.
- [12] SOVA AI, “Sova dataset: Multilingual stt/asr corpus,” https: //github.com/sovaai/sova-dataset, 2022.
- [13] A. Marren, “Phonetic perception and pronunciation difficulties of russian language (from a canadian perspective),” Arbutus Review, vol. 3, no. 1, pp. 75–84, 2012. [Online]. Available: https://journals.uvic.ca/index.php/arbutus/article/view/9064/2666
- [14] E. V. Rodionova, “Word order and information structure in russian syntax,” Master’s thesis, University of North Dakota, Grand Forks, ND, USA, 2001. [Online]. Available: https://commons.und.edu/theses/4482

- [15] A. Rozovskaya and D. Roth, “Grammar error correction in morphologically rich languages: The case of russian,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 1– 17, 2019.
- [16] O. K. Trubach, D. I. Gorshkova, and L. N. Sklyar, “Comparative analysis of phonetic systems of the russian, french and chinese languages,” RUDN Journal of Language Studies, Semiotics and Semantics, vol. 14, no. 1, pp. 171–188, 2023. [Online]. Available: https://journals.rudn.ru/semiotics-semantics/article/view/34176
- [17] D. A. Petrov, “RUAccent: Advanced system for stress placement in Russian with homograph resolution,” in Proceedings of the 31st International Conference on Computational Linguistics. Abu Dhabi, UAE: Association for Computational Linguistics, Jan. 2025, pp. 6642–6648. [Online]. Available: https://aclanthology. org/2025.coling-main.444/
- [18] A. F. M. Saif et al., “M2asr: Multilingual multi-task automatic speech recognition,” in Proceedings of INTERSPEECH, 2024. [Online]. Available: https://www.isca-archive.org/interspeech 2024/saif24 interspeech.pdf

- [19] J. Giraldo, M. Llopart, A. Peir´o-Lilja, C. Armentano-Oller, G. Sant, and B. K¨ulebi, “Enhancing crowdsourced audio for textto-speech models,” in IberSPEECH 2024, 2024, pp. 196–200.
- [20] Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han, and A. Bapna, “Libritts-r: A restored multi-speaker text-to-speech corpus,” in Interspeech 2023, 2023, pp. 5496–5500.
- [21] W. Zhang, C.-C. Yeh, W. Beckman, T. Raitio, R. Rasipuram, L. Golipour, and D. Winarsky, “Audiobook synthesis with longform neural text-to-speech,” in 12th ISCA Speech Synthesis Workshop (SSW2023), 2023, pp. 139–143.
- [22] C. Pethe, B. Pham, F. D. Childress, Y. Yin, and S. Skiena, “Prosody analysis of audiobooks,” in Proceedings of the 19th International Conference on Semantic Computing (ICSC), 2025. [Online]. Available: https://arxiv.org/abs/2310.06930
- [23] J. Peng, Y. Wang, B. Li, Y. Guo, H. Wang, Y. Fang, Y. Xi, H. Li, X. Li, K. Zhang, S. Wang, and K. Yu, “A survey on speech large language models for understanding,” 2025. [Online]. Available: https://arxiv.org/abs/2410.18908
- [24] A. Zee, M. Zee, and A. Søgaard, “Group fairness in multilingual speech recognition models,” in Findings of the Association for Computational Linguistics: NAACL 2024, K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 2213–2226. [Online]. Available: https://aclanthology.org/2024.findings-naacl.143/
- [25] S. Feng, B. M. Halpern, O. Kudina, and O. Scharenborg, “Towards inclusive automatic speech recognition,” Computer Speech and Language, vol. 84, p. 101567, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0885230823000864
- [26] M. Lau, Q. Chen, Y. Fang, T. Xu, T. Chen, and P. Golik, “Data quality issues in multilingual speech datasets: The need for sociolinguistic awareness and proactive language planning,”

2025. [Online]. Available: https://arxiv.org/abs/2506.17525

- [27] Pipecat AI developers, “Improved accuracy in smart turn v3.1,” https://www.daily.co/blog/ improved-accuracy-in-smart-turn-v3-1/, 2025.
- [28] Salute Developers, “Gigaam: the family of open-source acoustic models for speech processing,” https://github.com/ salute-developers/GigaAM, 2024, released under the MIT License. Accessed: April 10, 2025.
- [29] Alpha Cephei developers, “Vosk offline speech recognition api,” https://alphacephei.com/vosk/, 2025.
- [30] t tech, “T-one: Streaming speech recognition model,” https:// huggingface.co/t-tech/T-one, July 2025, accessed: 2026-01-21.
- [31] G. Mittag, B. Naderi, A. Chehadi, and S. M¨oller, “Nisqa: A deep cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets,” Interspeech 2021, 2021.

- [32] A. Plaquet and H. Bredin, “Powerset multi-class cross entropy loss for neural speaker diarization,” in Proc. INTERSPEECH 2023, 2023.
- [33] H. Bredin, “pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,” in Proc. INTERSPEECH 2023,

- 2023.

[34] D. Petrov, “Rupunct models,” https://huggingface.co/RUPunct,

- 2024.

- [35] G. Fedoseev, “Russian speech recognition system based on mozilla’s deepspeech tensorflow implementation,” https: //github.com/GeorgeFedoseev/DeepSpeech, 2017, forked from mozilla/DeepSpeech. [Online]. Available: https://github.com/ GeorgeFedoseev/DeepSpeech
- [36] A. Slizhikova, A. Veysov, D. Nurtdinova, and D. Voronin, “Russian open speech to text (stt/asr) dataset,” 2019. [Online]. Available: https://github.com/snakers4/open stt

- [37] L. Gabdrakhmanov, R. Garaev, and E. Razinkov, “Ruslan: Russian spoken language corpus for speech synthesis,” in Speech and Computer. Cham: Springer International Publishing, 2019, pp. 113–121.
- [38] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, N. Calzolari, F. B´echet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds. Marseille, France: European Language Resources Association, May 2020, pp. 4218–4222. [Online]. Available: https://aclanthology.org/2020.lrec-1.520/
- [39] ITU-T, “Recommendation P.501: Test signals for use in telephony and other speech-based applications,” International Telecommunication Union, Geneva, Switzerland, Recommendation, April 2025.
- [40] M. Chasin, “Hearing aids for musicians,” The Hearing Review, vol. 21, no. 3, pp. 12–16, March 2014. [Online]. Available: https://hearingreview.com/practice-building/ practice-management/hearing-aids-for-musicians
- [41] B. Ivan, “nisqa-s,” https://github.com/deepvk/nisqa-s, 2024.
- [42] J. Fiscus, “A post-processing system to yield reduced word error rates: Recognizer output voting error reduction (rover),” in 1997 IEEE Workshop on Automatic Speech Recognition and Understanding Proceedings, 1997, pp. 347–354.
- [43] S. Yolchuyeva, G. N´emeth, and B. Gyires-T´oth, “Transformer based grapheme-to-phoneme conversion,” in Interspeech 2019. ISCA, Sep. 2019, p. 2095–2099. [Online]. Available: http: //dx.doi.org/10.21437/Interspeech.2019-1954
- [44] T. Ylonen, “Wiktextract: Wiktionary as machine-readable structured data,” in Proceedings of the Thirteenth Language Resources and Evaluation Conference, 2022, pp. 1317–1325. [Online]. Available: https://aclanthology.org/2022.lrec-1.140/
- [45] D. Petrov, “Espeech: Technical report,” Den4ikAI, Technical Report, 2025. [Online]. Available: https://github.com/Den4ikAI/ ESpeech/blob/main/ESpeech techreport.pdf

- [46] its5Q, “biggest-ru-book: Russian multi-speaker audiobook tts and asr corpus,” https://huggingface.co/datasets/its5Q/ biggest-ru-book, 2024, contains almost 1000 hours of highquality Russian audio, 548k rows.
- [47] S. Andrey, V. Petrosyan, A. Gataulova, G. Mkrtchian, E. Zhmakin, R. Zalikov, and M. Alexander, “Labelspeech,” 2025. [Online]. Available: https://github.com/mtuciru/LabelSpeech
- [48] K. Baba, W. Nakata, Y. Saito, and H. Saruwatari, “The t05 system for the VoiceMOS Challenge 2024: Transfer learning from deep image classifier to naturalness MOS prediction of high-quality synthetic speech,” in IEEE Spoken Language Technology Workshop (SLT), 2024.

- [49] R. Chao, W.-H. Cheng, M. L. Quatra, S. M. Siniscalchi, C.-H. H. Yang, S.-W. Fu, and Y. Tsao, “An investigation of incorporating mamba for speech enhancement,” in 2024 IEEE Spoken Language Technology Workshop (SLT), 2024, pp. 302–308.
- [50] D. Snyder, G. Chen, and D. Povey, “MUSAN: A Music, Speech, and Noise Corpus,” 2015, arXiv:1510.08484v1.
- [51] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, and S. Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in ICASSP 2017, 2017, pp. 5220–5224.
- [52] Y. Hu and P. C. Loizou, “Evaluation of objective measures for speech enhancement,” in Interspeech 2006, 2006, pp. paper 2007– Tue3FoP.10.
- [53] A. Rix, J. Beerends, M. Hollier, and A. Hekstra, “Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs,” in ICASSP 2001, 2001, pp. 749–752.
- [54] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An algorithm for intelligibility prediction of time–frequency weighted noisy speech,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125–2136, 2011.
- [55] M. Chinen, F. S. C. Lim, J. Skoglund, N. Gureev, F. O’Gorman, and A. Hines, “Visqol v3: An open source production ready objective speech and audio metric,” in 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), 2020, pp. 1–6.
- [56] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-tospeech,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 5530–5540. [Online]. Available: https: //proceedings.mlr.press/v139/kim21f.html
- [57] G. Mittag and S. M¨oller, “Deep learning based assessment of synthetic speech naturalness,” in Interspeech 2020, 2020, pp. 1748– 1752.
- [58] L. Ma, D. Guo, K. Song, Y. Jiang, S. Wang, L. Xue, W. Xu, H. Zhao, B. Zhang, and L. Xie, “Wenetspeech4tts: A 12,800-hour mandarin tts corpus for large speech generation model benchmark,” in Interspeech 2024, 2024, pp. 1840–1844.
- [59] K. Seki, S. Takamichi, T. Saeki, and H. Saruwatari, “Ttsops: A closed-loop corpus optimization framework for training multispeaker tts models from dark data,” 2025. [Online]. Available: https://arxiv.org/abs/2506.15614
- [60] Y. Cao, S. Li, Y. Liu, Z. Yan, Y. Dai, P. Yu, and L. Sun, “A survey of ai-generated content (aigc),” ACM Comput. Surv., vol. 57, no. 5, Jan. 2025. [Online]. Available: https://doi.org/10.1145/3704262
- [61] Y. Geng, J. Xu, Z. Liang, J. Yang, X. Shi, and X. Shen, “Scaling under-resourced TTS: A data-optimized framework with advanced acoustic modeling for Thai,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), G. Rehm and Y. Li, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 593–604. [Online]. Available: https://aclanthology.org/2025.acl-industry.42/
- [62] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, “Emilia: A large-scale, extensive, multilingual, and diverse dataset for speech generation,” 2025. [Online]. Available: https://arxiv.org/abs/2501.15907
- [63] X. Song, M. Xing, C. Ma, S. Li, D. Wu, B. Zhang, F. Pan, D. Zhou, Y. Zhang, S. Lei, Z. Peng, and Z. Wu, “Touchtts: An embarrassingly simple tts framework that everyone can touch,”

2024. [Online]. Available: https://arxiv.org/abs/2412.08237

- [64] H. Schr¨oter, T. Rosenkranz, A. N. Escalante-B., and A. Maier, “DeepFilterNet: Perceptually motivated real-time speech enhancement,” in INTERSPEECH, 2023.

- [65] S. Kirdey, “Voicerestore: Flow-matching transformers for speech recording quality restoration,” 2025. [Online]. Available: https://arxiv.org/abs/2501.00794
- [66] Y.-X. Lu, Y. Ai, and Z.-H. Ling, “MP-SENet: A speech enhancement model with parallel denoising of magnitude and phase spectra,” in Proc. Interspeech, 2023, pp. 3834–3838.
- [67] S. Zhao, Y. Ma, C. Ni, C. Zhang, H. Wang, T. H. Nguyen, K. Zhou, J. Q. Yip, D. Ng, and B. Ma, “Mossformer2: Combining transformer and rnn-free recurrent network for enhanced timedomain monaural speech separation,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 10356–10360.
- [68] Vikhrmodels, “Tone: Family of russian asr/tts datasets,” https: //huggingface.co/collections/Vikhrmodels/tone, 2024, huggingFace Collection: Tone (ASR/TTS datasets for Russian).
- [69] Arun Babu and Changhan Wang and Andros Tjandra and Kushal Lakhotia and Qiantong Xu and Naman Goyal and Kritika Singh and Patrick von Platen and Yatharth Saraf and Juan Pino and Alexei Baevski and Alexis Conneau and Michael Auli, “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in Interspeech 2022, 2022, pp. 2278–2282.
- [70] S. Liang and G.-A. Levow, “Breaking the transcription bottleneck: Fine-tuning asr models for extremely low-resource fieldwork languages,” in Proceedings of the Fourth Workshop on NLP Applications to Field Linguistics (FieldMatters). Bangkok, Thailand: Association for Computational Linguistics, 2025, pp. 26–37. [Online]. Available: https://aclanthology.org/2025. fieldmatters-1.3
- [71] Y. Khare, T. Peyash, A. Vanzo, and T. Yoshioka, “Universal2-tf: Robust all-neural text formatting for asr,” 2025. [Online]. Available: https://arxiv.org/abs/2501.05948
- [72] A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates, and A. Y. Ng, “Deep speech: Scaling up end-to-end speech recognition,” 2014. [Online]. Available: https://arxiv.org/abs/ 1412.5567

### A. Related Work

#### A.1. Speech Datasets and Annotation

The rapid growth of web audio [1, 2] has made speech data central to multilingual voice interfaces and generative applications [60]. Yet many speech datasets lack scalability and rich annotations, especially for under-resourced languages [61], where simplistic curation misses prosody and phonetic detail critical for TTS [26]. For Russian, these issues are compounded by vowel reduction [16], mobile stress [17], and intonation patterns absent in dictated or synthesized speech [21], leading to biases that hinder web-scale use and inclusivity [24, 25]. As Table 5 shows, audiobook and dictated corpora (e.g., MAILABS [10], RUSLAN [37]) provide scripted text but omit stress and phonemes and often sound unnatural for spontaneous speech, while RuLS [11] and GOLOS [7] emphasize manual or scripted transcription with far-field noise, limiting prosodyaware mining despite openness.

Spontaneous and crowd-sourced corpora such as OpenSTT [36] and SOVA subsets [12] rely on ASR or subtitles but omit prosodic cues, and often suffer from background noise. Most Russian datasets lack stress, phonemes, and timestamps, and many also lack punctuation, with scalability frequently “partial” or “no” due to manual pipelines. Subtitledependent sets (e.g., DeepSpeech [35], SOVA YT [12]) require pre-existing text, MCV’s [8] manual workflows limit flexibility, and OpenSTT’s [36] parsing focus constrains spontaneous sources - together overlooking prosody and spontaneous speech and compounding web-scale mining challenges.

High-resource corpora illustrate stronger annotation pipelines: WenetSpeech4TTS [58] offers 12800 hours of spontaneous Mandarin with ASR, denoising, and word-level timestamps but no explicit stress/phonemes; LibriTTS-R [20] provides 585 hours of English audiobook speech with phoneme alignments, timestamps, and G2P-inferred stress, yet its approach does not transfer to Russian due to homographs and movable stress; and TTSOps [59] yields 66h of Japanese spontaneous speech from noisy YouTube via adaptive cleaning, subtitle-based ASR, and evaluation-in-the-loop, achieving high naturalness while highlighting gaps in annotation depth and scalability for Russian. These gaps in Russian speech datasets highlight the need for data-centric frameworks.

#### A.2. Annotation Pipelines and Frameworks in Speech Processing

Speech annotation pipelines play a pivotal role in web audio processing, enabling scalable mining of vast online resources like podcasts. General frameworks [58, 62, 63, 59] often incorporate modular components such as ASR for transcription, voice activity detection (VAD) for segmentation, and quality filtering metrics to ensure data usability.

In contrast, Russian-specific annotation pipelines typically rely on manual curation [7, 8] or simplistic methods like subtitle alignment [35, 36] and ASR for transcription [36], often lacking prosody-aware features and scalability. To our knowledge, our modular framework is the first of its kind for Russian, integrating prosody-focused annotation with scalable components for data mining.

Unlike previous approaches for Russian that stop at bare text-audio pairs, our framework adds punctuation, lexical stress, and word-level durations, supplying TTS models with rich prosodic cues. A multi-step quality filter removes unusable clips automatically. Each stage – is exposed as a self-contained

component, allowing language-specific modules to be swapped. Together these innovations deliver the first prosody-aware, scalable Russian annotation framework.

### B. Problem Definition and Notation

Algorithm 1 Utterance segmentation with speech/silence filtering

Require: A = {ai}Mi=1; dmax Ensure: D = {(sk)}Nk=1

D ← ∅ for i ← 1 to M do

({tsj}, {tej}) ← SemanticVAD(ai) ▷ Speech regions; j ∈ {1, . . . , Pi}

pieces ← [ ] ▷ Candidate index ranges over VAD regions n ← Pi ▷ Number of speech regions in ai start idx ← 0

while start idx < n do L ← tsstart idx end idx ← start idx while end idx < n do ▷ Extend until exceeding

dmax

R ← teend idx if (R − L) > dmax then ▷ Stop if too long

break end if end idx ← end idx + 1 ▷ Try adding the next

region

end while if end idx > start idx then

∆ ← teend idx−1 − L ▷ Candidate duration if ∆ ≥ dmax/3 and ∆ ≤ dmax then ▷ Keep

only reasonable durations pieces.append(start idx, end idx − 1)

end if end if start idx ← max(end idx, start idx + 1) ▷ Advance

pointer; avoid stalling end while for all (u, v) in pieces do ▷ Validate each candidate

ℓ ← tev − tsu ▷ Total segment length ρ ←

v j=u(tej−tsj)

ℓ ▷ Speech share in the segment

σ ← maxj∈{u,...,v−1}(tsj+1 − tej) ▷ Max internal silence gap

if ρ ≥ 0.7 and σ ≤ 1s then ▷ Accept only mostly-speech segments without long pauses

D ← D ∪ {ai[tsu : tev]} ▷ Extract waveform slice and add to output

#### end if

end for end for return D ▷ Return accepted segments

Our primary objective is to create a scalable, fully automated pipeline that processes Russian speech from diverse sources, filters it for quality, and enriches it with essential annotations, transforming raw audio into structured datasets. Focusing on Russian-specific phonetics, the workflow delivers high-fidelity data at scale with no manual intervention.

Table 5: Datasets comparison

Dataset Hours Speech Type Licensing PUN STR Phonemes Timestamps Multispeaker Text annotation Scalability LibriTTS-R(EN)[20] 585 B CC BY 4.0 ✓ ✓ ✓ ✓ ✓ scr partial

WeNetSpeech4TTS (ZH)[58]

12800 S CC BY 4.0 × × × ✓ ✓ ASR yes TTSops[59](JP) 66 S not open ✓ × × ✓ ✓ ASR, sub yes DeepSpeech [35] 6000 S MPL 2.0 × × × × ✓ sub partial GOLOS-C [7] 1095 D Golosa × × × × ✓ man no GOLOS-F [7] 132 D Golosa × × × × ✓ man no

M-AILABS [10] 46.8 B BSD-3-Clause ✓ × × × ✓ scr no OpenSTT [36] 20108 S, B, SY, D CC-BY-NC × × × × ✓

sub, ASR, scr, TTS

partial RuLS [11] 98 B PDb ✓ × × × ✓ scr no

CC BY-NC-SA 4.0 ✓ × × × × scr no

RUSLAN [37] 31 D

MCV [38] 286 D CC0 ✓ × × × ✓ scr partial SOVA AB [12] 298 B CC BY 4.0 × × × × ✓ scr no SOVA YT [12] 17451 S CC BY 4.0 × × × × ✓ sub partial

SOVA D [12] 191 D CC BY 4.0 × × × × ✓ scr no Ours 5078 S, B, D mixedc ✓ ✓ ✓ ✓ ✓ ASR yes

Note: This table compares speech datasets for TTS training, contrasting high-resource languages (EN: English, ZH: Chinese, JP: Japanese) with Russian ones on scale, annotations, and scalability for web audio mining. Attributes include Hours (dataset size), Speech Type (B = audiobook; S = spoken; D = dictated; SY = synthesized), Licensing, and features (✓ = present, × = absent), including PUN (punctuation annotations) and STR (stress annotations). Text Annotation: scr = scripted (dictated from text); ASR = via Automatic Speech Recognition; sub = subtitles from video hosting; TTS = synthesized from text; man = manual. Scalability: yes = fully automated; partial = limited (e.g., manual intervention); no = non-scalable.

- a Custom SberDevices license (similar to CC BY-SA). Full details available at https://github.com/sberdevices/golos/blob/master/license/en_us. pdf.
- b The dataset is Public Domain in the USA.
- c The code is released under the CC BY-NC-SA 4.0 license, the annotations under CC BY-NC-ND 4.0, and the original audio distributed under its respective licenses.

Algorithm 2 Whole pipeline Require: A = {ai}Mi=1; dmax; θq; vd Ensure: D = {(sk, Tk, Ak}Nk=1

D ← ∅ D ← Segmentation(A, dmax) ▷ alg. 1 for each sk ∈ D do

qk = QualityEstimator(sk) dk = SingleSpeakerDetector(sk) if qk ≥ θq and dk = vd then

rkCTC = ASRCTC(sk) rkCTC+LM, τk = ASRCTC+LM(sk) rkRNNT = ASRRNNT(sk) rkV OSK = ASRV OSK(sk) rkTONE = ASRTONE(sk) Tkr = ROVER(rkCTC, rkCTC+LM, rkRNNT, rkV osk, rkTONE) Ik = 1k(Tkr, rkCTC) ck = CREST(sk) if Ik = 1 and ck < 10 and lk > 3 then

Tkp = Punctuator(Tkr) Tks = StressPlacer(Tkp) Φk = G2P(Tks)

Tk ← {Tkr, Tkp, Tks, Φk}; Ak ← {τk, qk, dk} D ← D ∪ {(sk, Tk, Ak)}

end if

end if end for return D

Let A = {ai}Mi=1 denote the raw web audio corpus, where each ai is an unprocessed audio signal from diverse sources with variable lengths and no initial annotations. The pipeline transforms each ai into a structured dataset D = {sk, Tk, Ak}Nk=1, where sk is a segmented audio clip; Tk = {Tkr, Tkt, Tkp, Tks, Φk} includes textual annotations, with Tkr as the raw transcript, Tkt as the trancript from ASRCTC+LM, Tkp

as the punctuated text, Tks as the stress-annotated text, and Φk as the phoneme sequence; and Ak = {τk, qk, dk, ck, lk} captures auxiliary annotations, where τk is word-level timestamps, qk is the quality score, dk indicates single-speaker status, ck is the CREST factor and lk is the total utterance length.

Our evaluation goals assess the framework’s outputs in data quality, model performance, and annotation impacts for generative tasks.

We compare our datasets to Russian ones on audio quality and understandability (RQ1). We assess denoising models trained on our data against others (RQ2). For TTS, we measure quality, understandability and intonation (RQ3). We evaluate speech restoration models trained on our data versus baseline approaches (RQ4). Finally, we measure stage-wise processing throughput (RQ5).

### C. MOS Evaluation in details

We collected human ratings using Anonymised Platform6. Raters received the following rubric for manual MOS:

”5” – Perfect studio quality: clear sound without noise, reverb, distortion, robotic voice. Examples: podcasts, studio voice recordings, professional voiceovers.

”4” – Studio quality with artifacts: minor noise, slight reverberation, but speech is clear and sounds very good. Example: recordings from a microphone in a quiet room, but with background hum or light music.

”3” – Unprofessional recording: noticeable noise, distortion, poor speech clarity or the presence of any background music. Example: recording on a cheap microphone in a room with echo, social media audio.

”2” – Very low quality: severe distortion, noise, typical telephony. Example: telephone conversation with interference, recording with loud background noise.

”1” – Low quality telephony: speech is barely understandable,

6To ensure authors’ anonimity, the platform link well be available at camera-ready version

intermittent sound, humming. Example: old cell phone recording, audio from a bad VoIP call.

- ”0” – Non-speech: white noise, silence, unrecognizable sounds. Example: fan noise file with no speech, broken data.

Since we posit that punctuation, stress, and phonemic annotations affect the quality of synthetic speech, we additionally collected an intonation-focused MOS (IntMOS) to capture prosodic naturalness beyond overall quality. Raters used the following rubric:

”5” – Speech sounds like a real person in a normal conversation situation: correct stresses, logical pauses, no signs of diction or clich´es.

”4” – The intonation is dictation- or audiobook-like. The speech is recognized as human speech, but the narration is similar to dictation or audiobook, there may be excessive expressiveness and some unnatural pauses, there are no errors in stresses.

”3” – The intonation is indeterminate. It is difficult to tell whether it is a human or a machine: there may be individual mistakes in stresses and pauses, the structure of the intonation is broken.

”2” – Mostly robotic intonation. Speech sounds synthetic, there are often errors in stresses and pauses, intonation is unfamiliar, but there are attempts to imitate a human.

- ”1” – Clearly unnatural intonation. Intonation is clearly artificial, most of the stresses and pauses are wrong, there is no likeness to human speech. ”0” – Speech is unintelligible. It is impossible to understand what is being said, intonation cannot be evaluated due to low comprehensibility or loss of meaning.

Each audio item in our human-feedback evaluations received at least seven independent ratings from native speakers of Russian. For each clip, we aggregated ratings by taking the median. System-level scores were computed as the mean across clips, and we report 95% confidence intervals (CI) using:

S √n

CI = z∗

(1)

where z∗ = 1.96 is the Z-score for 95% confidence interval, S is the standard deviation and n is the number of clips. IntMOS is reported alongside MOS to isolate prosodic and intonational aspects that are specifically targeted by our annotation layers, providing a direct test of their contribution to perceived naturalness.

### D. Additional Research Questions

#### D.1. RQs in Introduction

- RQ4 (Restoration Performance): To what extent do speech restoration models trained on data annotated with our framework outperform existing baseline models?
- RQ5 (Processing Throughput): To what degree does our modular framework achieve web-scale throughput for large audio corpora?

#### D.2. RQs in Experimental Setup

- D.2.1. RQ4: Speech Restoration Performance

We test whether a restoration model trained on our data outperforms strong baselines on a fixed out-of-sample benchmark, isolating dataset quality from model novelty. Baselines are evaluated with official weights and settings, unmodified, to ask if training on our data can beat off-the-shelf models on the same

Table 6: Real-Time Factor comparison

Stage Section RTF hours/hour Segmentation(·) GPU/CPU 0.0107 93.11591

ASRCTC(·) GPU 0.0005 1741.362 ASRCTC+LM(·) GPU 0.0007 1478.0087

ASRRNNT (·) GPU 0.0062 160.9663 ASRV OSK(·) CPU 0.0065 153.2891

ASRtone(·) CPU 0.0128 77.86585 ROV ER(·) CPU 0.0002 5040 Punctuator(·) GPU 0.0003 3308.5851

StressPlacer(·) GPU 0.0005 1879.9074

G2P(·) GPU 0.0012 810.7867 PyAnnote GPU 0.00631 158.06722

NISQA GPU 0.0003 2857.14286

Note: Real-Time Factor (RTF) is defined as processing time/audio duration; values < 1 indicate fasterthan-real-time processing. “hours/hour” denotes the amount of audio (in hours) processed per wall-clock hour, computed as 1/RTF. The “Section” column refers to corresponding subsections in this paper where each stage is described, and “Device” indicates whether the stage runs on GPU, CPU, or both.

test audio. We train a single restoration model (SEMamba [49]) with Adam (lr=5 × 10−4), batch size 20, for 105 steps, using MUSAN [50] and Room Impulse Responses(RIRs) [51] to construct degradations; we use the last checkpoint for evaluation. Training uses the entire corpus to reflect the end-to-end benefit of our approach.

The evaluation set is a 20-hour subset of SOVA RuYouTube [12] selected at random and held fixed for all systems; subjective metrics are computed on a uniformly sampled 200-clip subset from these 20 hours, identical across systems to enable paired comparisons. Objective quality is reported with NISQA metrics [31] and UTMOS [48] on the full test set; subjective quality uses MOS; We also report Accent Rate (AR), defined as the proportion of clips that raters judge to carry a non-native accent, capturing artifacts that may arise when systems are trained on non-Russian or otherwise mismatched data. All baselines (SEMamba [49] (original and trained on our dataset), DeepFilterNet3 [64], VoiceRestore [65], MP-SENet [66], MossFormer2 [67]) are run from official releases with default inference parameters on the same input audio.

D.2.2. RQ5: Processing Throughput

Assessing stage-wise throughput is key to understanding the scalability of our framework. Variability in source material and quality thresholds can significantly change downstream data volume, so we report per-stage metrics rather than an aggregate end-to-end figure, which would be dataset dependent and potentially misleading. Throughput is measured as Real-Time Factor defined as the total processing time divided by the total audio duration. We also report hours processed per wallclock hour. All measurements use a single machine with 3x RTX5060Ti 16GB GPU, 2x RTX4060Ti 16GB and dual Intel Xeon E5-2699a v4 CPU, running four parallel processes and single-item batches.

Table 7: Speech Restoration models comparison

Model NOI DIS COL LOU NMOS UTMOS MOS ± 95% CI AR ± 95% CI Source 2.8884 3.4525 2.8541 3.167 2.6316 2.57 2.87±0.1 N/A SEMamba [49] 3.9307 4.1226 3.6928 3.8473 3.7824 2.6667 2.929 ± 0.103 0.881 ± 0.038 DeepFilterNet3 [64] 4.0474 3.9886 3.5067 3.7764 3.6049 2.4665 2.137 ± 0.057 0.964 ±0.017 VoiceRestore [65] 3.1257 3.8354 3.1551 3.4223 3.0314 2.2688 2.944 ± 0.084 0.933 ±0.03

MP-SENet [66] 3.9409 4.1193 3.7012 3.9491 3.8098 2.4914 3±0.087 0.9234 ± 0.031 MossFormer2 [67] 3.7777 4.0171 3.4800 3.6947 3.5138 2.3973 2.953 ±0.079 0.949± 0.027

##### Balalaika (ours) 4.1723 4.1842 3.922 4.1408 3.8723 2.4152 3.2625 ± 0.116 0.955 ± 0.039

Note: The table shows a comparison of different speech restoration models whose weights and inference code are taken from the original implementations. The table also shows the metrics of the original degraded sample (source) and the SEMamba model trained on the first part of our dataset (ours). The following metrics were used: NISQA[31] (NOI, COL, DIS, LOU, NMOS), UTMOSv2[48] (UTMOS), the manual MOS with 95% confidence intervals (MOS ± 95% CI), and the accent rate with 95% confidence intervals (AR ± 95% CI), which represents the percentage of audio that has an accent. The highest and second-highest scores for the metrics are shown in bold and underlined text.

Table 8: Curated Datasets

Dataset original dataset hours Balalaika version hours original link Balalaika version link

Yandex Music Podcasts 3071 2165 N/A The link will be available at camera-ready version OpenSTT [36] 20108 431.43 link The link will be available at camera-ready version

ESpeech [45] 1726 475.892 link The link will be available at camera-ready version ESpeech Podcasts [45] 3200 900.63 link The link will be available at camera-ready version

GOLOS [7] 1227 49.086 link The link will be available at camera-ready version DeepSpeech [35] 6000 278.634 link The link will be available at camera-ready version

TONE Webinars [68] 2208 248.91 link The link will be available at camera-ready version Biggest Russian Books [46] 1000 528.247 link The link will be available at camera-ready version

Note: “Original dataset hours” are taken from the upstream projects or papers; “Balalaika version hours” reflect the portion processed and released within the Balalaika collection for that source, and may be updated. “Original link” points to the upstream dataset landing page; “Balalaika version link” points to the corresponding Balalaika-packaged subset with enriched annotations. “N/A” indicates the upstream source has no single canonical landing page or redistribution is not permitted.

#### D.3. RQs from Results and Discussion

- D.3.1. RQ4: Speech Restoration Performance

Models trained on our data outperform baseline systems, with our SEMamba [49] achieving the highest human MOS and the strongest overall NISQA profile among all methods in Table 7. The objective predictors and subjective ratings are aligned for our model, and Accent Rate remains on par with leading alternatives, indicating that the gain in perceived restoration quality is consistent across metrics and does not introduce accentrelated artifacts.

D.3.2. RQ5: Processing Throughput

Our framework processes all stages faster than real time, confirming scalability for continuous annotation. As shown in Table 6, the most demanding modules remain below the realtime threshold. This stage-wise balance prevents single-module bottlenecks and demonstrates that the modular design supports web-scale dataset creation without compromising speed or annotation depth.

### E. Ethics

Privacy & Consent. This work processes publicly available audio to create annotations for generative speech research. We neither collect, infer, nor release speaker identities or demographics; single-speaker filtering excludes only overlapping speech. Released artifacts contain solely auto-generated annotations aligned to source IDs and timestamps, which risk deanonymization if combined with external data.

Licensing. Source material is included only when platform terms or licenses allow derivative research use, or where fair use may apply. We do not host third-party audio if prohibited, and require users to comply with local IP law.

Misuse & Safety. TTS/enhancement models trained with our pipeline could be misused for impersonation, fraud, or disinformation. We recommend mitigating this risk by deploying and regularly updating audio anti-spoofing [This citations will be avalable at camera-ready version] and monitoring performance under out-of-domain conditions.

Raters Identities & Consent. Manual evaluations were conducted by an anonymized pool of raters whose roles/affiliations are not disclosed; no identifiers were collected. Raters provided informed consent and participated voluntarily without coercion; no demographics were collected beyond selfattested native Russian proficiency, and no conflicts of interest were reported. The evaluated audio was not expected to contain sensitive content.

### F. Additional Discussions

#### F.1. Annotator pool

This study engaged 68 native Russian annotators; every audio item received at least seven independent ratings, with per-item scores aggregated by the median, and each manual metric was evaluated on a 200-item sample; confidence intervals are reported for all metrics, only native-language status was collected about raters, and the annotation platform is [To ensure authors’ anonymity, the platform name will be available at camera-ready version].

#### F.2. Multilinguality

The pipeline is modular and can be ported beyond Russian by swapping language-dependent components, but several parts require careful adaptation to maintain accuracy and prosody fidelity across languages. Semantic VAD is the first bottleneck: SmartTurn v2 [27] currently supports a fixed set of high-resource languages (English, Italian, French, Spanish, Dutch, Russian, German, Chinese, Korean, Portuguese, Turkish, Japanese, Polish, Hindi), so low-resource languages would need new semantic VAD models or rule-based fallbacks, which is outside this paper’s scope.

Quality and speaker modules are likely portable but should be validated: NISQA [31] and PyAnnotate diarization [33] may be language-agnostic in principle, yet cross-lingual robustness must be tested before production use, especially for accents, phonotactics, and music/ad overlap patterns that differ by language and domain.

Transcription is replaceable where strong ASRs exist [69]; for low-resource languages, choosing resilient architectures and multilingual or self-supervised models is crucial [70] since robust ASR remains an open challenge and directly impacts downstream punctuation and timestamp reliability.

Text enrichment layers are the most language-specific: Russian punctuation restoration models cannot be reused as-is; building or selecting per-language punctuators, or training a multilingual punctuator, remains an open research problem with variable transferability across scripts and orthographies [71].

Stress placement is particularly challenging in Slavic and other movable-stress languages; beyond Russian, reliable public models are scarce, and language-specific stress predictors would need to be developed where applicable to preserve prosody cues [17].

Finally, grapheme-to-phoneme to IPA is comparatively straightforward to extend; encoder–decoder G2P models can be trained per language to supply IPA sequences for TTS and generative tasks with modest engineering effort [44], fitting cleanly into the pipeline’s plug-replace design.

#### F.3. Future work

First, a speech restoration module will be developed that conditions on audio, text, and auxiliary features to reconstruct high-quality speech suitable for generative training [20]; this will unlock currently unused recordings that were excluded by strict filtering, repurposing them as supervised or self-supervised restoration targets to expand training diversity.

Second, a music detector will be integrated to automatically flag and exclude segments with background music, improving data purity for both ASR alignment and downstream TTS prosody modeling in mixed-media sources such as podcasts and YouTube excerpts.

Third, the robustness of NISQA as a no-reference quality estimator will be stress-tested across domains, codecs, and accents represented, with contingency plans to calibrate or design a more resilient variant if systematic biases are observed under cross-dataset or cross-condition evaluation.

Fourth, an audio super-resolution component is planned to upsample 16 kHz material to 48 kHz, targeting clearer fricatives and extended bandwidth that may improve intelligibility and timbral naturalness for modern generative vocoders and diffusion TTS models, with careful human and objective evaluation to validate perceptual benefits before large-scale adoption.

Fifth, a speech prompt generation system will be introduced to synthesize task-specific prompts and reference snippets, en-

abling controllable conditioning for style, pace, and emphasis in generative models and facilitating reproducible benchmarking of prompt-conditioned synthesis and restoration.

### G. Reproducibility checklist

To support reproducibility while preserving double-blind anonymity, we provide an anonymized code repository and release-ready dataset packaging scripts; public links to the nonanonymized repositories and hosted artifacts will be added in the camera-ready version. We release all generated annotations, but we do not attach them to the submission to avoid deanonymization via hosting metadata and repository linkage.

- • Code repository: end-to-end pipeline implementation (URL omitted; public link will be provided at camera-ready).
- • Dataset repositories: one repository per curated dataset split, containing Parquet-formatte annotation artifacts (public links will be provided at camera-ready).
- • Non-redistributable audio sources: for Yandex Music podcasts and ESpeech podcasts, we do not redistribute audio; instead we provide a single entry-point script to (i) re-download the original audio from the upstream source and (ii) deterministically match it to our released annotations and timestamps.
- • External model dependencies: each pipeline stage except G2P relies on external models; we provide download scripts to obtain all required weights and resources.
- • Hosting: the packaged datasets are hosted on Hugging Face and planned to be mirrored on ModelScope; direct links will be added in the camera-ready version.

### H. Curated Data

#### H.1. Yandex Music podcasts annotated by Balalaika

Yandex Music Podcasts serve as a large, publicly accessible catalog of Russian podcast episodes with predominantly conversational, studio-quality speech well-suited for prosody-focused TTS research; in this split, audio is not redistributed due to Russian legal restrictions, but reproducibility is ensured by providing the exact episode links, identifiers, and a yandex musicbased7 Python workflow to download source audio and deterministically match it to released annotations and timestamps; no category constraints were applied during collection; for compliance, source audio may be used for research purposes only, with no redistribution and no commercial use allowed, while the released artifacts are annotations.

#### H.2. OpenSTT annotated by Balalaika

OpenSTT [36] is a large multi-domain Russian speech corpus spanning radio, public speeches, audiobooks, YouTube, and phone calls; in the Balalaika pipeline, this source was consumed as pre-segmented clips without additional segmentation, with a simple duration filter removing all clips shorter than 3 seconds before applying Balalaika pipeline.

#### H.3. ESpeech annotated by Balalaika

ESpeech [45] is a Russian speech dataset introduced in a 2025 technical report by Denis Petrov that documents the corpus composition, collection and processing pipeline, and provides an accessible PDF for reference and citation, serving as a con-

7https://pypi.org/project/yandex-music/

temporary resource accompanying related Den4ikAI speech assets and tooling.

#### H.4. GOLOS annotated by Balalaika

Golos [7] is a freely available Russian speech corpus, built from audio recorded and manually annotated via crowdsourcing, and released alongside an acoustic model and 3-gram KenLM language models trained to facilitate benchmarking and transfer learning studies.

#### H.5. DeepSpeech annotated by Balalaika

DeepSpeech [35] repository is a Russian speech fork of Mozilla’s DeepSpeech [72] that adapts the end-to-end architecture for Russian, provides training/config scripts and Dockerized workflows, and references data acquisition from YouTube captions alongside Russian text corpora for language modeling.

#### H.6. Biggest Russian Books annotated by Balalaika

Biggest Russian Books [46] is a large multi-speaker Russian audiobook corpus on Hugging Face with almost 1000 hours of high-quality recordings and roughly 548k utterances, packaged as WebDataset shards with per-clip text and speaker metadata for speech research.

