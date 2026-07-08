## EMILIA: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation

Haorui He⋆, Zengqiang Shang⋆, Chaoren Wang⋆, Xuyuan Li⋆, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang,

Kai Chen, Pengyuan Zhang†, Zhizheng Wu†, Senior Member, IEEE

### arXiv:2501.15907v2[cs.SD]8Oct2025

Abstract—Recent advancements in speech generation have been driven by large-scale training datasets. However, current models struggle to capture the spontaneity and variability inherent in real-world human speech, as they are primarily trained on audio-book datasets limited to formal, read-aloud speaking styles. To address this limitation, we introduce Emilia-Pipe, an opensource preprocessing pipeline designed to extract high-quality training data from valuable yet under-explored in-the-wild sources that capture spontaneous human speech in real-world contexts. Using Emilia-Pipe, we construct Emilia, which comprises over 101k hours of speech across six languages: English, Chinese, German, French, Japanese, and Korean. Furthermore, we expand Emilia to Emilia-Large, a dataset exceeding 216k hours, making it one of the largest open-source speech generation resources available. Extensive experiments show that Emilia-trained models produce markedly more spontaneous, human-like speech than those trained on traditional audio-book datasets, while matching their intelligibility. These models better capture diverse speaker timbres and the full spectrum of real-world conversational styles. Our work also highlights the importance of scaling dataset size for advancing speech generation performance and validates the effectiveness of Emilia for both multilingual and crosslingual speech generation tasks.

Index Terms—In-the-wild Speech Generation Dataset

I. INTRODUCTION

# I

N recent years, (zero-shot) speech generation research has witnessed significant advancements [1], [2], [3], [4], [5], [6],

with multiple models driven by large-scale training datasets. These advancements lead to improved voice quality, timbre similarity, and naturalness [7]. Nevertheless, the generated speech still falls short of replicating the spontaneity and variability characteristic of real-world human speech [2], [8].

Haorui He, Chaoren Wang, Yicheng Gu, Liwei Liu, Jiaqi Li, Yuancheng Wang, and Zhizheng Wu are with the Chinese University of Hong Kong, Shenzhen, China.

Zengqiang Shang, Xuyuan Li, Hua Hua, Chen Yang, Peiyang Shi, and Pengyuan Zhang are with the Laboratory of Speech and Intelligent Information Processing, Institute of Acoustics, CAS, Beijing, China.

Xuyuan Li, Hua Hua, Chen Yang, and Pengyuan Zhang are also with the

University of Chinese Academy of Sciences, Beijing, China. Kai Chen is with Shanghai AI Laboratory, Shanghai, China. ⋆: Equal contribution, and the names are listed in random order. †: Corresponding authors. Email: wuzhizheng@cuhk.edu.cn

This work is led by CUHK-Shenzhen and supported by the National Natural Science Foundation of China (No. 62376237), the 2023 Shenzhen Stability Science Program, the Shenzhen Science and Technology Program (No. ZDSYS20230626091302006), the Program for Guangdong Introducing Innovative and Entrepreneurial Teams (No. 2023ZT10X044), and the Postdoctoral Fellowship Program of China Postdoctoral Science Foundation (No. GZB20230811).

A primary factor for this limitation is the reliance of current speech generation models on datasets derived from audiobooks [9], [10]. Such datasets predominantly feature formal, read-aloud speaking styles, which contrast with the diverse and spontaneous nature of human speech in casual or conversational settings. Such real-world speech is characterized by a wide range of phenomena, including breathing sounds, pauses, stuttering, repetitions, variations in speaking rate and emotions. Consequently, there remains a significant research gap for a dataset that encompasses a wider spectrum of speaking styles for speech generation model training.

However, directly utilizing in-the-wild speech data presents significant challenges due to variations in quality, such as frequent background noise or music, reverberation, overlapping speakers within a single sample, inconsistent speech lengths, and the absence of essential annotations like text transcriptions [11], [12]. Training speech generation models on such unprocessed raw data can result in degraded performance [11], [13], [14]. While previous studies [11], [12] have proposed automatic preprocessing pipelines to mitigate these issues, they heavily depend on proprietary models, which significantly limits their accessibility for the broader research community. Moreover, these pipelines are typically restricted to monolingual (i.e., Chinese-only) speech data, making them unsuitable for processing the multilingual speech data available in the wild. Finally, the computational efficiency of these approaches also remains undocumented, raising concerns about their practicality for building large-scale speech generation datasets. Therefore, there is an urgent need for an effective and open-source preprocessing pipeline that can efficiently handle multilingual in-thewild speech data and enable large-scale dataset construction.

In response, we present Emilia-Pipe, the first open-source preprocessing pipeline specifically designed to leverage valuable yet underexplored in-the-wild multilingual speech data for constructing high-quality training datasets for spontaneous and human-like speech generation models. Emilia-Pipe comprises six core preprocessing steps: standardization, source separation, speaker diarization, fine-grained segmentation by voice activity detection (VAD), automated speech recognition (ASR), and filtering. In addition, Emilia-Pipe incorporates extensive engineering optimizations to enhance both robustness and efficiency. These features position Emilia-Pipe as an effective and efficient tool for building large-scale, multilingual speech datasets.

Utilizing Emilia-Pipe, we introduce Emilia, the first multilingual speech generation dataset constructed from in-thewild speech data. Emilia comprises over 101k hours of 24

TABLE I: A comparison of Emilia and Emilia-Large datasets with existing datasets for speech generation.

Dataset Data Source Total Duration (hours) Language Samp. Rate (Hz) Dynamic LJSpeech [15] Audio-book 24 En 22.05k

AutoPrepWild [11] In-the-wild 39 Zh 24k/44.1k ✓(Proprietary) VCTK [16] Studio Recording 44 En 48k

AISHELL-3 [17] Studio Recording 85 Zh 44.1k LibriTTS [18] Audio-book 585 En 24k GigaSpeech [19] In-the-wild 10k En 16k

WenetSpeech [20] In-the-wild 10k Zh 16k WenetSpeech4TTS [12] In-the-wild 12k Zh 16k ✓(Proprietary) Libri-Heavy [21] Audio-book 50k En 16k

MLS [10] Audio-book 51k En/Fr/De/Nl/Es/It/Pt/Pl 16k Libri-Light [9] Audio-book 60k En 16k

SeamlessAlign [22] In-the-wild 270k 76 Languages 16k

Emilia In-the-wild 101k En/Zh/De/Fr/Ja/Ko 24k ✓ Emilia-Large In-the-wild 216k En/Zh/De/Fr/Ja/Ko 24k ✓

kHz speech across six languages: English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko). To further scale up our dataset, we present Emilia-Large, which expands the total duration to 216k hours. Table I compares Emilia and Emilia-Large with several existing datasets.

The main advantages of our proposed Emilia and EmiliaLarge datasets are summarized below.

- • Extensive: Emilia contains over 101k hours of speech data, and the Emilia-Large variant expands the size to 216k hours, which is one of the largest open-source speech generation datasets.
- • Multilingual: The Emilia and Emilia-Large datasets cover six languages, supporting the training of multilingual and crosslingual speech generation models.
- • Diverse: Emilia and Emilia-Large distinguish themselves from prior datasets by centering on spontaneous, in-thewild speech. This ensures coverage of a broad spectrum of speaking styles, which is essential for training nextgeneration speech generation models capable of producing natural, spontaneous, and human-like speech.
- • Dynamic: The Emilia and Emilia-Large datasets uniquely feature an automatic and efficient processing pipeline, i.e., Emilia-Pipe, which enables seamless expansion in both dataset size and language coverage, significantly accelerating dataset construction.

This work extends upon our previous research presented at IEEE Spoken Language Technology Workshop 2024 [23], introducing the following four key enhancements:

- • Larger-scale Dataset: We expanded the initial Emilia dataset to create Emilia-Large, a dataset more than twice the size of its predecessor.
- • Comparative Analysis of Audio-book and In-the-Wild Datasets: We train identical models on traditional audiobook datasets and on the in-the-wild Emilia dataset, then compare their speech generation performance. Results demonstrate that models trained with in-the-wild data achieve significantly higher speaker similarity and naturalness in the synthesized speech.
- • Exploration of Data Scaling Laws in Speech Generation: We conducted experiments to investigate the effect

of dataset size on speech generation performance. The results highlight the importance of scaling dataset sizes.

• Multilingual and Crosslingual Effectiveness Analysis: We conduct experiments to validate the multilingual and crosslingual capabilities of Emilia, further demonstrating its applicability across six different languages.

Our code for Emilia-Pipe1 as well as the Emilia and EmiliaLarge datasets2 have been made publicly available to facilitate future research and ensure reproducibility. The remainder of this paper is organized as follows: Section II reviews related work. Section III describes the proposed Emilia-Pipe. Section IV details and analyzes our datasets. Experimental results are presented in Section V. Section VI concludes the paper.

II. RELATED WORK A. Speech Generation Datasets

The size of datasets for speech generation has increased substantially over the years. Early datasets typically comprised tens of hours of speech. For example, LJSpeech [15] contains 24 hours of speech data from a single speaker. The VCTK dataset [16] includes 44 hours of speech data from 109 speakers, while AISHELL-3 [17] comprises approximately 85 hours of recordings from 218 speakers. Subsequently, larger-scale datasets have emerged to enable research in zeroshot speech generation. For instance, LibriTTS [18] contains 585.8 hours of speech data from audio-books, Later, the research community scaled up speech generation datasets to over 10k hours. Emerging datasets such as MLS [10] (51k hours), Libri-Light [9] (60k hours), and Libri-Heavy [21] (50k hours) significantly enhance the performance for zero-shot speech generation models. However, these datasets are derived from audio-books and thus primarily capture formal, readaloud speaking style, limiting their effectiveness for training natural and spontaneous speech generation models [8], [24]. While large-scale in-the-wild corpora such as GigaSpeech [19], WenetSpeech [20], and SeamlessAlign [22] are available alternatives for broader speaking styles, their uncurated content

- 1https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia
- 2https://huggingface.co/datasets/amphion/Emilia-Dataset

[Figure 1]

[Figure 2]

[Figure 3]

Standardization Source Separation

In-the-wild Speech Data Standardized In-the-wild Speech Data Clean Human Vocal

[Figure 4]

[Figure 5]

[Figure 6]

Speaker Diarization

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

Fine-grained Filtering ASR Segmentation by VAD

Speaker A

Speaker A

Speaker B

Speaker C

Speaker A

Speaker B

Speaker C

Resulting Dataset

Speech Segments with Transcripts

Speaker Annotations

3-30s Speech Segments

- Fig. 1: An overview of the Emilia-Pipe pipeline. It consists of six steps, namely, standardization, source separation, speaker diarization, fine-grained segmentation by voice activity detection (VAD), automated speech recognition (ASR), and filtering.

and inconsistent speech quality can severely degrade generation performance if used without careful preprocessing [11], [13].

Fig. 1, Emilia-Pipe includes six steps: standardization, source separation, speaker diarization, fine-grained segmentation by voice activity detection (VAD), automated speech recognition (ASR), and filtering. This section details the six steps of EmiliaPipe and evaluates its performance.

To address this, two previous works [12], [11] propose similar automated preprocessing pipelines for building speech generation datasets from in-the-wild data. However, as discussed earlier, these pipelines heavily rely on proprietary models and are restricted to Chinese-only speech data, with unknown computational efficiency. To bridge this gap, we design and open-source an efficient pipeline, Emilia-Pipe, which can rapidly process large-scale raw multilingual speech data to facilitate large-scale dataset construction.

- a) Standardization: The source speech data in-the-wild vary in encoding formats, sampling rates, and other characteristics. To standardize the collected data, we convert all samples to WAV files, set them to a mono channel, and resample to 24 kHz. We set the sample width to 16 bits and adjust the target decibels relative to full scale to -20 dBFS. The actual gain is constrained within -3 to 3 dB to ensure appropriate volume without distortion. Finally, we normalize the waveform by dividing each sample by the maximum amplitude, ensuring values range between -1 and 1. This step ensures a consistent data format for further processing.
- b) Source Separation: The source speech data in-the-wild often contain background music and noise, which negatively impact the performance of speech generation models [3], [13]. To address this issue, we employ source separation techniques to extract clean human vocals. Specifically, we utilize the open-source pre-trained Ultimate Vocal Remover model [27], UVR-MDX-Net Inst 33. This model achieves a high signalto-distortion ratio (SDR) of 11.15 for vocal separation on the Synth MVSep dataset [28]. Using this model, we effectively separate human vocals for further processing.
- c) Speaker Diarization: After extracting clean human vocals from the source speech data, we apply speaker diarization techniques to partition long-form speech data into multiple utterances based on the speaker. This process generates a series of speech segments, with each segment ideally containing only one speaker, ensuring compatibility with existing datasets for speech generation. To achieve this, we leverage the PyAnnote speaker diarization 3.1 pipeline.4, which includes three core components: speaker segmentation, speaker embedding, and

B. Speech Generation Models

Traditional speech generation models, such as Tacotron [25], FastSpeech [26], are limited by smaller datasets with only tens of hours of speech data. Recent advancements in speech generation, driven by large-scale speech datasets, have significantly improved voice quality, timbre similarity, and naturalness in zero-shot speech generation using only a short reference speech sample of a few seconds. For example, VALLE [1], VoiceBox [3], and SoundStorm [4] use more than 50k hours of speech data for training. Subsequent models, such as NaturalSpeech3 [2], BaseTTS [5], and Seed-TTS [6], further expanded dataset sizes to over 100k hours. Notably, BaseTTS reported the “emergent abilities” of TTS models: as dataset sizes scaled, the models can render complex prosody patterns such as emotions based on textual cues without explicit labels. In this work, we introduce Emilia-Large, one of the largest open-source datasets for speech generation of 216k hours, to further advance the development of speech generation.

III. THE EMILIA-PIPE PROCESSING PIPELINE

As discussed above, source speech data in-the-wild need to be processed to be leveraged for training speech generation models. Therefore, we design an automatic preprocessing pipeline, Emilia-Pipe, for transforming in-the-wild multilingual speech data into high-quality training datasets. As illustrated in

- 3https://github.com/TRvlvr/model repo/releases/tag/all public uvr models

- 4https://github.com/pyannote/pyannote-audio

TABLE II: Processing results of Emilia-Pipe.

Duration (s) DNSMOS ↑ Clips Total Duration (hours) min max avg ± std min max avg ± std

Processing Steps

Source Speech 20.09 3596.27 547.80 ± 653.92 1.04 3.51 2.52 ± 0.68 4383 666.94 (100.00%) + Source Separation 20.09 3596.27 547.80 ± 653.92 0.96 3.54 2.88 ± 0.50 4383 666.94 (100.00%)

+ Speaker Diarization 0.02 1955.93 4.30 ± 10.73 0.64 3.69 2.50 ± 0.69 431876 515.50 (77.29%) + Fine-grained Segmentation by VAD 3.00 30.00 9.05 ± 5.25 0.80 3.69 2.92 ± 0.50 207709 522.03 (78.27%) + ASR 3.00 30.00 9.16 ± 5.25 0.97 3.68 2.98 ± 0.44 172923 439.80 (65.94%) + Filtering 3.00 30.00 9.68 ± 5.17 3.00 3.68 3.26 ± 0.14 96117 258.44 (38.75%)

Total Processing Time: 240.5 mins; Real-Time Factor (RTF): 0.006

clustering, achieving state-of-the-art (SOTA) speaker diarization performance [29], [30]. The output is a list of temporal annotations indicating the start and end times of the singlespeaker segments.

- d) Fine-grained Segmentation by VAD: Although the speaker diarization pipeline provides a coarse segmentation for the source speech data, the resulting utterances may still be too long for training speech generation models. To address this, we use a VAD model to further segment the utterances into smaller segments ranging from 3 to 30 seconds. Specifically, we concatenate consecutive chunks containing voice activity from the same speaker. We leverage the open-source library Silero-VAD5, which achieves a ROC-AUC score of 0.99 on the LibriParty dataset.6
- e) ASR: The absence of text transcriptions impedes the direct use of in-the-wild dataset for TTS. Therefore, we use ASR techniques to transcribe the segmented speech data. Considering the trade-off among speed, robustness, and accuracy, we employ the medium version of the Whisper model [31], a SOTA multilingual ASR model capable of robust speech translation and language identification. To further enhance efficiency, we leverage WhisperX [32], which builds on the faster-whisper backend7 and the CTranslate2 engine.8 This setup is up to four times faster than the official Whisper implementation while maintaining comparable accuracy and using less memory. Additionally, we omit the original model’s inherent VAD component by using the outputs in the last step to avoid redundant processing and develop batched inference to transcribe the speech data in parallel. These improvements allow our ASR step to achieve accurate text transcriptions for the speech data with high efficiency.
- f) Filtering: Real-world noises may not be completely handled by source separation, the ASR step may introduce errors, and some source speech data may be of low quality [11]. Therefore, to ensure the quality of the resulting dataset, we apply the following filtering criteria.9 Firstly, we utilize the language identification results from the Whisper model in the ASR step. We discard speech data that are not predicted to belong to our target languages (En, Zh, De, Fr, Ja, Ko) or have

- 5https://github.com/snakers4/silero-vad
- 6https://github.com/speechbrain/speechbrain/tree/develop/recipes/

LibriParty/generate dataset

- 7https://github.com/SYSTRAN/faster-whisper
- 8https://github.com/OpenNMT/CTranslate2 9Please note that the filtering criteria can be adjusted to fit the specific

needs of different use cases.

a language identification confidence lower than 80%. Secondly, we use the DNSMOS P.835 OVRL score [33] (hereafter referred to as DNSMOS score for brevity) to estimate the overall speech quality. This non-intrusive metric reflects the overall quality of the speech data and is highly correlated with human ratings [33]. Following ITU-T P.835 [34], we retain only utterances whose DNSMOS ≥ 3.0, the lowest “fair” quality anchor, thereby guaranteeing that every training sample meets a minimum bar accepted for high-fidelity speech synthesis. Finally, for each source speech data, we compute the average character duration over its corresponding segments. We consider segments with an average phone duration outside 1.5 times the interquartile range (IQR) above the third quartile or below the first quartile as outliers and discard the speech data for these segments. After filtering, we obtain the resulting dataset.

g) Performance Evaluation: To evaluate the effectiveness of Emilia-Pipe, we processed 666.94 hours of randomly selected source speech data through the pipeline. Table II illustrates the impact of each processing step within Emilia-Pipe.

Before processing, the raw source speech data is highly variable, with clip durations ranging from 20.09 to 3596.27 seconds (averaging 547.80 seconds ± 653.92 seconds), and DNSMOS scores spanning 1.04 to 3.51 (averaging 2.52 ± 0.68), reflecting varied speech quality in-the-wild. Applying source separation preserves the total duration but improves the average DNSMOS score to 2.88 ± 0.50. Speaker diarization segments multi-speaker audio, increasing the number of clips to 431,876 while reducing the total duration to 515.50 hours (77.29% of the original). Subsequent VAD steps further refine the dataset, constraining all segment durations to between 3 and 30 seconds. The final filtering step yields 96,117 highquality clips totaling 258.44 hours (38.75% of the original data), with a notably improved DNSMOS score of 3.26 ± 0.14, indicating low variability and superior speech quality. These results demonstrate Emilia-Pipe’s capability to convert raw speech data in the wild into a high-quality dataset for speech generation.

This experiment was conducted on a server equipped with eight NVIDIA RTX 4090 GPUs, running eight independent processes.10 On this server, processing the 666.94 hours of source speech data took 240.5 minutes, resulting in a Real-Time Factor (RTF) of approximately 0.006 (i.e., 1 second of source speech data processed in 0.006 seconds). This high efficiency

10Processing speed may vary depending on hardware and data characteristics; reported figures are provided for reference.

46.77%

1.72%0.22%

1.38%

1.59%

49.87%

En: 46.8k Zh: 49.9k De: 1.6k

Fr: 1.4k Ja: 1.7k Ko: 0.2k

61.95%

3.44%

1.17%

3.80%

3.15%

26.49%

En: 134.1k (2.9x)

Zh: 57.3k (1.1x)

De: 6.8k (4.3x)

Fr: 8.2k (6.0x) Ja: 2.5k (1.5x) Ko: 7.4k (34.3x)

(a) Emilia

(b) Emilia-Large

- Fig. 2: Duration statistics (in hours) of the speech data in Emilia and Emilia-Large by language. The numbers in parentheses indicate the scaling factor (multiples) of the speech data in Emilia-Large compared to the original Emilia dataset.

underscores Emilia-Pipe’s suitability for large-scale speech dataset preprocessing, enabling fast training data preparation for speech generation models.

IV. THE EMILIA AND EMILIA-LARGE DATASET

Leveraging Emilia-Pipe, we are able to construct speech generation datasets derived from in-the-wild speech data. In this section, we describe our constructed Emilia dataset and the extended Emilia-Large dataset. These datasets contain inthe-wild speech data in six languages (En, Zh, De, Fr, Ja, Ko), processed by Emilia-Pipe. Duration statistics for each language in the datasets are provided in Fig. 2.

- A. The Emilia Dataset

- 1) Overview: Using Emilia-Pipe, we construct Emilia from

in-the-wild speech data sourced from a vast collection of video and podcast platforms on the Internet. This data covers various speaking styles such as audio-books, drama, interviews, talk shows, and commentary, thereby capturing a wide array of real human speaking styles. After processing, Emilia includes a total of 101,654 hours of multilingual speech data across six different languages.

- 2) Dataset Analysis: To validate the quality and diversity

of Emilia, we conduct respective analyses.

- a) Quality: To evaluate the quality, we compared Emilia with several existing datasets using DNSMOS scores. Table III presents the speech quality comparison between Emilia and several existing datasets. Emilia achieves a DNSMOS score of 3.26, ranking third among all datasets. The results indicate that, despite being sourced from source speech data in-the-wild, after preprocessing, the speech quality of Emilia is comparable to existing datasets sourced from studio recordings or audio-books and outperforms the existing datasets sourced from unprocessed in-the-wild speech data.
- b) Diversity: To quantify the diversity of speaking style domains within Emilia, we follow WenetSpeech [20] by classifying the source speech data into ten domains: Audiobook, Commentary, Documentary, Drama, Interview, News, Reading, Talk, Variety, and Others.

- TABLE III: DNSMOS Scores of Emilia and ten existing datasets. The scores for LJSpeech, AutoPrepWild, AISHELL-3, LibriTTS, and WenetSpeech, are derived from [11]. The score for Libri-Light is computed from its official “small” subset, and the score for WenetSpeech4TTS is computed from its official “basic” subset. The scores for MLS and Emilia are computed from a randomly sampled 600-hour subset.

Dataset DNSMOS ↑ LJSpeech [15] 3.30 ± 0.17

AutoPrepWild [11] 3.24 ± 0.21 VCTK [16] 3.20 ± 0.18 AISHELL-3 [17] 3.15 ± 0.17 LibriTTS [18] 3.25 ± 0.19

GigaSpeech [19] 2.52 ± 0.54 WenetSpeech [20] 2.43 ± 0.55

WenetSpeech4TTS [12] 3.18 ± 0.22

MLS [10] 3.33 ± 0.19 Libri-Light [9] 3.25 ± 0.26

Emilia 3.26 ± 0.14

- TABLE IV: Distribution of speaking style domains in Emilia.

Domain Percentage

Audio-book 9.40% Commentary 17.75% Documentary 11.46%

Drama 12.09% Interview 0.93%

News 2.77% Reading 15.60%

Talk 13.02% Variety 6.19% Others 10.79%

This classification is performed by analyzing textual information extracted from source URLs and metadata (such as hashtags) using Gemma-3-12B-it.11 As shown in Table IV, Emilia demonstrates a well-balanced distribution of speaking styles across most domains

To illustrate the diversity advantage of Emilia over traditional audio-book datasets, we perform a comparative analysis

11https://huggingface.co/google/gemma-3-12b-it

[Figure 20]

- (a) Acoustic diversity

[Figure 21]

- (b) Semantic diversity

- Fig. 3: A comparison of acoustic and semantic diversities between Emilia and MLS datasets.

between Emilia and the MLS dataset in both acoustic and semantic feature spaces. Specifically, we randomly select 5,000 samples each from the English subsets of Emilia and MLS, denoted as Emilia English and MLS English.

For acoustic features, we leverage a pre-trained WavLM model [35] to extract 768-dimensional embeddings for each speech sample, which captures various acoustic attributes such as speaker identity, emotion, and prosody. These highdimensional features are then reduced to two dimensions using Principal Component Analysis (PCA) [36]. As shown in

- Fig. 3(a), Emilia displays a wider distribution compared to the more compact clustering of MLS, highlighting Emilia’s richer coverage in acoustic characteristics.

For semantic diversity, we employ a pre-trained SentenceBERT model [37] to generate 768-dimensional text embeddings from the transcripts of each sample. These embeddings capture the semantic content of these samples. Similar to above, we reduce the dimensions of these embeddings to two using PCA.

- Fig. 3(b) demonstrates that Emilia spans a broader semantic coverage than MLS.

- B. The Emilia-Large Dataset

Given the significant advancements in large-scale speech generation [23], we are motivated to further scale up the volume of the training dataset for speech generation models to investigate the data scaling laws in speech generation and potentially enhance model performance. In this work, we expand the initial Emilia dataset to an even larger scale, introducing Emilia-Large, which contains a total of 216,313 hours of spontaneous and human-like speech data. Emilia-Large

Emilia-Large (216k hours)

Emilia (CC-BY-NC-4.0, 101k hours)

Emilia-YODAS (CC-BY-4.0, 114k hours)

Other sources

Fig. 4: The relationship between Emilia and Emilia-Large. Emilia-Large is an extended version of Emilia, incorporating additional data primarily processed from YODAS2.

builds upon the same construction methodology as Emilia using our proposed Emilia-Pipe but doubles the total dataset size.

The expansion from Emilia to Emilia-Large primarily uses YODAS2 [38] as the data source, termed Emilia-YODAS. The relationship between Emilia and Emilia-Large is illustrated in Fig. 4. YODAS2 is a large-scale (500k hours) real-world speech collection from YouTube videos with CC-BY-3.0 licenses in more than 100 languages. We selectively downloaded and processed data from YODAS2 and other smaller sources in the six languages of the original Emilia dataset.12 We process the source speech data using our proposed Emilia-Pipe, with only one alteration: changing the DNSMOS filtering threshold for the incremental data to 2.4 to align with [11] and preserve more data. We make the Emilia-YODAS publicly available under the CC-BY-4.0 license.

Fig. 2(b) demonstrates the duration statistics for each language in Emilia-Large. It is observed that the key distinction of Emilia-Large compared to Emilia is its significantly improved inclusion of low-resource languages, especially for German (De), French (Fr), and Korean (Ko). Specifically, compared to the original Emilia dataset, the data for these three languages has been expanded several-fold (4.3 times for De, 6.0 times for Fr, and 34.3 times for Ko, respectively). This enhancement addresses the relatively limited data volume of these languages in Emilia, improving support for multilingual and crosslingual speech generation tasks.

V. EXPERIMENTS

In this section, we conduct experiments to address the following evaluation questions (EQs) to validate the strengths of Emilia in terms of diversity (EQ1), extensiveness (EQ2), and multilingual utility (EQ3):

- • EQ1: Is Emilia more effective than existing audio-book datasets for training spontaneous and human-like speech generation model?
- • EQ2: What are the data scaling laws in speech generation, i.e., the impact of dataset size on speech generation performance with a fixed number of model parameters?
- • EQ3: How effective is Emilia for training multilingual and crosslingual speech generation models?

12Unused data of other languages in YODAS2 can also be seamlessly processed by Emilia-Pipe for speech generation training. Their effectiveness may depend on the ASR model’s performance in each language.

A. Comparison of Audio-book and In-the-Wild Datasets (EQ1)

To address EQ1, we compare the performance of TTS models trained on existing audio-book datasets to the performance of models trained on Emilia.

- a) Baselines: We implement two SOTA TTS models as baselines: (1) AR+SoundStorm [39]: A two-stage model where a LLaMA-style autoregressive (AR) generative transformer first predicts semantic tokens using text and prompt semantic tokens as input. Then a SoundStorm-based semantic-to-acoustic model [4] that generates acoustic tokens conditioned on the predicted semantic tokens. (2) VoiceBox [3]: A non-autoregressive (NAR) speech generation model that leverages flow matching to predict mel-spectrograms. For comprehensive details on these models, we refer readers to their respective publications.
- b) Training Sets: We evaluate the performance of TTS models trained on two English datasets: the English subset of Emilia (hereafter referred to as Emilia-En for brevity) and the MLS dataset, a high-quality corpus derived from audio-books. The Emilia-En dataset consists of approximately 46k hours of English speech, while the MLS dataset contains 45k hours, rendering their sizes comparable.
- c) Evaluation Sets: To ensure a comprehensive evaluation, we employ two evaluation sets in two domains: (1) LibriSpeechTest: This evaluation set includes 1,200 speech samples in formal reading styles similar to those in the MLS dataset. (2) Emilia-Test: This evaluation set consists of 600 speech samples in spontaneous, human-like speaking styles akin to those in Emilia. Both evaluation sets are unseen by the baseline models during training.
- d) Metrics: We conduct both objective and subjective evaluations to assess the performance of the baseline models.

For objective evaluation, we focus on the following aspects. (1) Intelligibility: This is measured by the Word Error Rate (WER) between the generated speech transcription and the target text. For LibriSpeech-Test, we use a fine-tuned HuBERTLarge ASR model13 to transcribe the generated speech. For Emilia-Test, we use the Whisper-Medium model [31], which offers greater robustness for in-the-wild speech. (2) Similarity: This is measured by Speaker Similarity Score (S-SIM) between the generated speech and the reference speech using the WavLM-TDCNN speaker embedding model.14 (3) Naturalness: This is measured using the Fr´echet Speech Distance (FSD), which quantifies the distance between the distributions of generated and real speech samples in a feature space [3]. We employ the SOTA emotion representation model, emotion2vec [40], to compute FSD and evaluate the emotional naturalness of the generated speech.

For the subjective evaluation, we randomly select sixteen samples, eight from the LibriSpeech-Test and eight from the Emilia-Test evaluation set. Twelve proficient English speakers serve as evaluators. The subjective evaluation includes: (1) Speaker Similarity: We employ the Similarity Mean Opinion Score (SMOS) to assess the speaker similarity of the generated speech to the reference speech. The SMOS scale ranges from

- 13https://huggingface.co/facebook/hubert-large-ls960-ft
- 14https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker

verification

1 to 5, with steps of 0.5. (2) Comparative Naturalness: We use the Comparative Mean Opinion Score (CMOS) to evaluate the comparative naturalness of the generated speech against the reference speech. The CMOS scale ranges from -3 (indicating the generated speech is much worse than the reference speech) to 3 (indicating the generated speech is much better than the reference speech), with steps of 1.

e) Results and Discussions: Table V summarizes the objective and subjective evaluation results of TTS models trained on the Emilia-En and MLS datasets on the LibriSpeechTest and Emilia-Test evaluation sets.

On LibriSpeech-Test, the AR+SoundStorm model trained on Emilia-En achieved a lower WER (8.4%) and FSD (24.73) compared to its MLS-trained counterpart, while the VoiceBox model trained on MLS achieved the best WER (6.1%), SSIM (0.625), and FSD (16.83). On the Emilia-Test, the AR+SoundStorm model trained on Emilia-En outperformed the MLS-trained model across all metrics, including WER (6.6%), S-SIM (0.618), FSD (12.73), CMOS (0.19), and SMOS (3.73). Similarly, the Emilia-En trained VoiceBox achieved superior results in WER (7.4%), S-SIM (0.601), and SMOS (3.76) compared to the MLS-trained version.

The results validate that, on the formal reading style LibriSpeech-Test evaluation set, models trained on both the Emilia-En and MLS datasets achieve comparable levels of intelligibility, speaker similarity, and naturalness. This suggests that Emilia, despite being derived from source speech data inthe-wild, is as effective as high-quality datasets like MLS after processing with our proposed Emilia-Pipe. However, on the Emilia-Test evaluation set, which includes more spontaneous and human-like speech, training on in-the-wild datasets like Emilia significantly enhances the performance of speech generation models.

f) Summary (Answer to EQ1): The comparison between the in-the-wild Emilia dataset and audio-book MLS dataset for speech generation tasks reveals that while both types of datasets yield comparable performance in formal, audiobook-style speech, Emilia significantly outperforms audiobook datasets in generating more spontaneous and humanlike speech, showcasing significantly superior performance in cloning diverse speaker timbre and speaking styles.

B. Data Scaling Law in Speech Generation (EQ2)

To address EQ2, we conduct experiments to investigate the impact of dataset size on speech generation performance with a fixed number of model parameters, i.e., the data scaling law in speech generation.

- a) Experimental Setups: We leverage the baseline models, evaluation sets, and objective metrics described in Sec. V-A. To assess performance scaling, we progressively increase the training set size to the following amounts: 5k, 10k, 46k (the total duration of English speech in Emilia-En), 100k, and 134k hours (the total duration in Emilia-Large). At each stage, we record the corresponding changes in model performance.
- b) Results and Discussions: Fig. 5a shows the WER trends for AR+SoundStorm and VoiceBox as training set size increases. On LibriSpeech-Test (L), both models exhibit a

- TABLE V: Objective and subjective evaluation results of TTS models trained on Emilia-En and MLS on LibriSpeech-Test and Emilia-Test evaluation sets. The best results for each model are highlighted in bold.

LibriSpeech-Test Emilia-Test

Model Training Set

WER ↓ S-SIM ↑ FSD ↓ CMOS ↑ SMOS ↑ WER ↓ S-SIM ↑ FSD ↓ CMOS ↑ SMOS ↑ AR+SoundStorm

MLS 8.9% 0.612 49.11 -0.36 3.13 7.7% 0.587 20.76 0.09 3.71 Emilia-En 8.4% 0.577 24.73 -0.19 3.28 6.6% 0.618 12.73 0.19 3.73

MLS 6.1% 0.625 16.83 0.36 3.62 8.2% 0.528 15.94 0.28 3.61 Emilia-En 7.2% 0.585 23.24 0.42 3.77 7.4% 0.601 14.07 0.28 3.76

VoiceBox

0.8

AR+SoundStorm (L) VoiceBox (L) AR+SoundStorm (E) VoiceBox (E)

AR+SoundStorm (L) VoiceBox (L) AR+SoundStorm (E) VoiceBox (E)

AR+SoundStorm (L) VoiceBox (L) AR+SoundStorm (E) VoiceBox (E)

0.7

7.1

- 20

22

24

22 21.97

20.71 20.58 20.4

- 21.33 21.23 19.94 19.93 19.85

6.7

0.64 0.64

6.5

6.4

0.63 0.62

6.3

0.61

0.61

0.61

0.6 0.6 0.61

0. 0.6

5.9 5.7 5.7 5.7 5.6

0.59 0.59

0.59 0.58

0.6

5.2

18

4.9 4.9 4.9

16.39

0.5

15.62 15.43 15.31

0.46

16

- 4

- 6

8

5.2

4.7

4.4 4.5

4.2

- 7.2

15.27

0.44

15.13

14.82

14.65 14.78 14.48

0.42

0.

14

0.4

5k 10k 46k 100k 134k

5k 10k 46k 100k 134k

5k 10k 46k 100k 134k

(a) WER (%)

(b) S-SIM

(c) FSD

Fig. 5: Model performance vs. training set size on LibriSpeech-Test (L) and Emilia-Test (E).

decrease in WER; for instance, WER of AR+SoundStorm drops from 5.2% at 5k hours to 4.2% at 134k hours. Similarly, on the Emilia test set (E), the WER of AR+SoundStorm decreases from 5.7% to 4.9%, while the WER of VoiceBox declines from 7.1% to 6.4%.

- Fig. 5b illustrates the S-SIM trends. On LibriSpeechTest, AR+SoundStorm improves from 0.587 to 0.620, while VoiceBox rises from 0.418 to 0.606 as the dataset scales to 134k hours. On Emilia-Test, AR+SoundStorm increases from 0.587 to 0.636, and VoiceBox advances from 0.422 to 0.612.
- Fig. 5c shows the FSD trends. On LibriSpeech-Test, AR+SoundStorm decreases from 22.00 to 20.40, and VoiceBox decreases from 21.33 to 19.85. On Emilia-Test, AR+SoundStorm changes from 14.82 to 15.31, while VoiceBox decreases from 16.39 to 14.48.

The results reveal a consistent scalability pattern across all metrics: both models demonstrate steady improvements as the dataset size increases, with only one exception of FSD for AR+SoundStorm (E). For smaller datasets (5k to 10k hours), performance gains are more pronounced, indicating that even modest increases in training data yield significant improvements. As the dataset size exceeds 46k hours, the rate of improvement slows but maintains a positive trend, eventually converging at approximately 100k hours.

c) Summary (Answer to EQ2): Our experimental results demonstrate a data scaling law in speech generation: as dataset size increases, performance improves, but with diminishing returns. Significant gains are observed when scaling up to 46k hours; beyond this point, improvements continue but become less substantial and tend to plateau around 100k hours. This insight can help guide future research in balancing dataset size and computational resources for optimal speech generation.

For TTS models containing around 0.5–1 billion parameters,15 a dataset of approximately 100k hours per language appears to be the most cost-effective choice.

C. Multilingual and Crosslingual Speech Generation (EQ3)

To address EQ3, we conduct experiments to evaluate the effectiveness of Emilia for training both multilingual (where the reference and target speech are in the same language) and crosslingual (where the reference and target speech are in different languages) speech generation models.

- a) Experimental Setup: We utilize the complete EmiliaLarge dataset, which encompasses six languages: English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko), to train speech generation models. For English evaluation, we leverage Emilia-Test. The Chinese evaluation set is randomly sampled from the AISHELL-3 dataset [17]. The evaluation sets for German, French, Japanese, and Korean are sourced from Common Voice [41]. Each evaluation set contains at least 500 reference speech samples. In multilingual experiments, these reference speech samples are used to synthesize target texts in the same language. For crosslingual experiments, the reference speech samples and target texts are selected from evaluation sets of different languages.
- b) Results and Discussions: The experimental results in Table VI demonstrate the effectiveness of the Emilia-Large dataset for multilingual and crosslingual speech generation.

In multilingual generation, where the reference and target languages are the same, both AR+SoundStorm and VoiceBox achieve strong performance across all six languages. For

15The size of our baselines also falls within this range.

- TABLE VI: Experimental results of AR+SoundStorm and VoiceBox for multilingual and crosslingual speech generation. The models were trained on the Emilia-Large dataset. Results for multilingual speech generation are highlighted in gray.

Target

AR+SoundStorm VoiceBox En Zh Fr De Ja Ko En Zh Fr De Ja Ko

Metric

Reference

WER ↓ 5.9% 5.8% 6.4% 5.9% 6.3% 8.3% 6.5% 7.9% 8.8% 8.6% 8.3% 10.2% En S-SIM ↑ 0.568 0.431 0.452 0.529 0.446 0.443 0.588 0.386 0.458 0.490 0.425 0.442

FSD ↓ 24.99 99.40 82.84 26.62 89.40 98.36 24.34 91.29 78.53 68.62 92.54 89.49 WER ↓ 5.3% 3.6% 5.2% 5.4% 4.9% 5.7% 8.6% 5.6% 6.7% 5.9% 6.4% 7.0%

Zh S-SIM ↑ 0.507 0.511 0.509 0.504 0.516 0.523 0.524 0.557 0.524 0.522 0.543 0.591

FSD ↓ 56.15 40.09 56.75 57.10 56.71 52.60 109.67 40.04 58.47 72.47 64.73 57.90 WER ↓ 5.3% 5.3% 5.3% 5.2% 5.8% 8.1% 7.0% 6.3% 5.6% 6.9% 7.5% 9.3%

Fr S-SIM ↑ 0.596 0.527 0.596 0.596 0.572 0.557 0.565 0.485 0.589 0.582 0.547 0.556 FSD ↓ 39.89 66.21 39.88 38.48 51.13 54.41 91.08 80.76 42.38 58.16 63.36 61.51

- WER ↓ 4.5% 4.5% 4.7% 4.2% 4.8% 6.8% 5.2% 7.4% 6.8% 5.2% 6.9% 8.9%

De S-SIM ↑ 0.619 0.545 0.603 0.639 0.596 0.591 0.639 0.519 0.577 0.683 0.538 0.586 FSD ↓ 39.96 57.82 44.86 33.16 53.38 55.12 83.37 72.18 54.77 34.41 67.89 67.46

- WER ↓ 4.6% 4.4% 4.7% 4.5% 4.8% 6.6% 7.4% 5.5% 6.9% 6.7% 6.2% 6.6%

Ja S-SIM ↑ 0.622 0.557 0.626 0.618 0.641 0.633 0.556 0.525 0.521 0.557 0.584 0.596

FSD ↓ 49.42 68.70 44.67 50.47 44.28 52.19 103.68 76.65 63.55 72.41 44.71 56.34 WER ↓ 6.2% 4.1% 6.1% 6.2% 6.2% 6.3% 8.0% 5.6% 7.8% 8.3% 5.6% 6.0%

Ko S-SIM ↑ 0.657 0.593 0.665 0.656 0.673 0.673 0.589 0.567 0.545 0.597 0.595 0.648 FSD ↓ 36.71 58.85 32.27 37.20 31.95 30.27 86.57 63.49 53.75 57.19 52.85 38.82

example, AR+SoundStorm attains WERs ranging from 3.6% (Zh-Zh) to 6.3% (Ko-Ko), with S-SIM scores between 0.511 (Zh-Zh) and 0.673 (Ko-Ko), and FSD as low as 24.99 (En-En).

In crosslingual generation, where the reference and target languages differ, model performance shows moderate degradation. For instance, AR+SoundStorm’s WER increases from 6.3% (Ko-Ko) to 8.3% (En-Ko). Similarly, VoiceBox’s S-SIM decreases from 0.588 (En-En) to 0.386 (En-Zh). Nonetheless, the overall performance of both models in crosslingual generation remains competitive. These results validate the effectiveness of the Emilia-Large dataset in training strong multilingual and crosslingual speech generation models.

Furthermore, when comparing model performance to those trained exclusively on English data, the multilingual models exhibit a slight trade-off in English generation. For example, AR+SoundStorm trained on the 216k-hour multilingual EmiliaLarge dataset achieves a WER of 4.9%, an S-SIM of 0.636, and an FSD of 15.31 on English evaluation samples. These metrics are marginally worse than those of its monolingual counterpart trained on a 134k-hour English-only dataset (WER=4.5%, SSIM=0.65, FSD=14.8, as reported in Fig. 5a–5c). This suggests that while multilingual data enables crosslingual capabilities, it can compromise language-specific performance. Such tradeoffs are important considerations for practical applications, and future work could explore strategies to narrow these performance gaps.

c) Summary (Answer to EQ3): The Emilia-Large dataset can be effectively used to train robust multilingual and crosslingual speech generation models. However, crosslingual generation introduces moderate performance degradation, highlighting challenges in cross-language acoustic transfer. These findings underscore the value of the Emilia-Large dataset as a critical resource for advancing multilingual and crosslingual speech generation. Future work could focus on enhancing model adaptability to address crosslingual challenges.

VI. CONCLUSION AND DISCUSSION

In conclusion, this work first introduces Emilia-Pipe, an effective and efficient preprocessing pipeline designed to transform source speech data in-the-wild into high-quality training datasets for spontaneous and human-like speech generation. Leveraging Emilia-Pipe, we construct Emilia, one of the largest open-source multilingual speech generation datasets, spanning over 101k hours across six languages, as well as its extended version, Emilia-Large, which contains 216k hours of data. Comparative analyses demonstrate that Emilia significantly outperforms traditional audio-book datasets in generating spontaneous and human-like speech. Our experiments also investigate the relationship between dataset size and speech generation performance, revealing consistent improvements with data scaling, though the trend becomes less pronounced as the dataset size exceeds 100k hours. Finally, we validate that the proposed Emilia dataset effectively supports multilingual and crosslingual speech generation, paving the way for future advancements in this field.

Future work may focus on training effective spoof detection models to address potential safety concerns associated with highly spontaneous and human-like speech generation models trained on Emilia, such as the risk of synthetic spoken misinformation [42], [43]. Additionally, expanding Emilia to include the singing/music domain could benefit singing/music voice generation [44].

Despite the advancements, we point out a few limitations. First, the speaker diarization model we use is not perfect and can result in a small proportion of speech segments containing more than one speaker or overlaps. This issue can subsequently affect speech generation performance. Integrating stronger models in the future may alleviate this issue. Second, Emilia-Pipe segments speech samples into intervals of 3 to 30 seconds. It is observed that generating speech outside this range may lead to unexpected outcomes. Adjustments in the hyper-

parameters of Emilia-Pipe may be needed for specific use cases. Third, due to our resource constraints, the current Emilia-Large dataset covers only six languages. We welcome community contributions to extend the dataset to more languages with our open-source Emilia-Pipe to enhance its global applicability.

REFERENCES

- [1] C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li et al., “Neural codec language models are zero-shot text to speech synthesizers,” arXiv preprint arXiv:2301.02111, 2023.
- [2] Z. Ju, Y. Wang, K. Shen, X. Tan, D. Xin, D. Yang, Y. Liu, Y. Leng, K. Song, S. Tang et al., “Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models,” in ICML, 2024.
- [3] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz, M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar et al., “Voicebox: Text-guided multilingual universal speech generation at scale,” in NeurIPS, 2024.
- [4] Z. Borsos, M. Sharifi, D. Vincent, E. Kharitonov, N. Zeghidour, and M. Tagliasacchi, “Soundstorm: Efficient parallel audio generation,” arXiv preprint arXiv:2305.09636, 2023.
- [5] M. Łajszczak, G. C´ambara, Y. Li, F. Beyhan, A. van Korlaar, F. Yang, A. Joly, A.´ Mart´ın-Cortinas, A. Abbas, A. Michalski et al., “Base tts: Lessons from building a billion-parameter text-to-speech model on 100k hours of data,” arXiv preprint arXiv:2402.08093, 2024.
- [6] ByteDance, “Seed-tts: A family of high-quality versatile speech generation models,” arXiv preprint arXiv:2406.02430, 2024.
- [7] X. Zhang, L. Xue, Y. Gu, Y. Wang, J. Li, H. He, C. Wang, T. Song, X. Chen, Z. Fang, H. Chen, J. Zhang, T. Y. Tang, L. Zou, M. Wang, J. Han, K. Chen, H. Li, and Z. Wu, “Amphion: An open-source audio, music and speech generation toolkit,” in SLT, 2024.
- [8] X. Tan, T. Qin, F. Soong, and T.-Y. Liu, “A survey on neural speech synthesis,” arXiv preprint arXiv:2106.15561, 2021.
- [9] J. Kahn, M. Rivi`ere, W. Zheng, E. Kharitonov, Q. Xu, P.-E. Mazar´e, J. Karadayi, V. Liptchinsky, R. Collobert, C. Fuegen, T. Likhomanenko, G. Synnaeve, A. Joulin, A. Mohamed, and E. Dupoux, “Libri-light: A benchmark for asr with limited or no supervision,” in ICASSP, 2020.
- [10] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “Mls: A large-scale multilingual dataset for speech research,” in INTERSPEECH, 2020.
- [11] J. Yu, H. Chen, Y. Bian, X. Li, Y. Luo, J. Tian, M. Liu, J. Jiang, and S. Wang, “Autoprep: An automatic preprocessing framework for in-the-wild speech data,” in ICASSP, 2024.
- [12] L. Ma, D. Guo, K. Song, Y. Jiang, S. Wang, L. Xue, W. Xu, H. Zhao, B. Zhang, and L. Xie, “Wenetspeech4tts: A 12,800-hour mandarin tts corpus for large speech generation model benchmark,” in INTERSPEECH, 2024.
- [13] T.-h. Huang, J.-h. Lin, and H.-y. Lee, “How far are we from robust voice conversion: A survey,” in SLT, 2021.
- [14] X. Li, Z. Shang, H. Hua, P. Shi, C. Yang, L. Wang, and P. Zhang, “Sfspeech: Straightened flow for zero-shot voice clone,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 1706–1718, 2025.
- [15] K. Ito and L. Johnson, “The lj speech dataset,” keithito.com/ LJ-Speech-Dataset/, 2017.
- [16] J. Yamagishi, C. Veaux, and K. MacDonald, “Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92),” datashare.ed.ac.uk/handle/10283/3443, 2019.
- [17] S. Yao, H. Bu, X. Xu, S. Zhang, and M. Li, “Aishell-3: A multi-speaker mandarin tts corpus,” in INTERSPEECH, 2021.
- [18] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “Libritts: A corpus derived from librispeech for text-to-speech,” in INTERSPEECH, 2019.
- [19] G. Chen, S. Chai, G. Wang, J. Du, W.-Q. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang, M. Jin, S. Khudanpur, S. Watanabe, S. Zhao, W. Zou, X. Li, X. Yao, Y. Wang, Y. Wang, Z. You, and Z. Yan, “Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio,” in INTERSPEECH, 2021.
- [20] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng, D. Wu, and Z. Peng, “Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition,” in ICASSP, 2022.
- [21] W. Kang, X. Yang, Z. Yao, F. Kuang, Y. Yang, L. Guo, L. Lin, and D. Povey, “Libriheavy: a 50,000 hours asr corpus with punctuation casing and context,” in ICASSP, 2024.

- [22] L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, P.-A. Duquenne, H. Elsahar, H. Gong, K. Heffernan, J. Hoffman et al., “Seamlessm4t: massively multilingual & multimodal machine translation,” arXiv preprint arXiv:2308.11596, 2023.
- [23] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, “Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation,” in SLT, 2024.
- [24] X. Tan, J. Chen, H. Liu, J. Cong, C. Zhang, Y. Liu, X. Wang, Y. Leng, Y. Yi, L. He, S. Zhao, T. Qin, F. Soong, and T.-Y. Liu, “Naturalspeech: End-to-end text-to-speech synthesis with human-level quality,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 6, pp. 4234–4245, 2024.
- [25] Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio et al., “Tacotron: Towards end-toend speech synthesis,” in INTERSPEECH, 2017.
- [26] Y. Ren, Y. Ruan, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fastspeech: Fast, robust and controllable text to speech,” in NeurIPS, 2019.
- [27] M. Kim, W. Choi, J. Chung, D. Lee, and S. Jung, “Kuielab-mdx-net: A two-stream neural network for music demixing,” in ISMIR MDX Workshop, 2021.
- [28] R. Solovyev, A. Stempkovskiy, and T. Habruseva, “Benchmarks and leaderboards for sound demixing tasks,” arXiv preprint arXiv:2305.07489, 2023.
- [29] H. Bredin, “Pyannote.audio 2.1 speaker diarization pipeline: Principle, benchmark, and recipe,” in INTERSPEECH, 2023.
- [30] A. Plaquet and H. Bredin, “Powerset multi-class cross entropy loss for neural speaker diarization,” in INTERSPEECH, 2023.
- [31] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and

I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in ICML, 2023.

- [32] M. Bain, J. Huh, T. Han, and A. Zisserman, “Whisperx: Time-accurate speech transcription of long-form audio,” in INTERSPEECH, 2023.
- [33] C. K. Reddy, V. Gopal, and R. Cutler, “Dnsmos p.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,” in ICASSP, 2022.
- [34] International Telecommunication Union, “P.835: Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm,” https://www.itu.int/rec/T-REC-P.835, 2003, accessed: 2025-09-05.
- [35] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “Wavlm: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.
- [36] K. P. F.R.S., “Liii. on lines and planes of closest fit to systems of points in space,” The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, vol. 2, no. 11, pp. 559–572, 1901.
- [37] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in EMNLP, 2019.
- [38] X. Li, S. Takamichi, T. Saeki, W. Chen, S. Shiota, and S. Watanabe, “Yodas: Youtube-oriented dataset for audio and speech,” in ASRU, 2023.
- [39] Y. Wang, H. Zhan, L. Liu, R. Zeng, H. Guo, J. Zheng, Q. Zhang, X. Zhang, S. Zhang, and Z. Wu, “Maskgct: Zero-shot text-to-speech with masked generative codec transformer,” in ICLR, 2025.
- [40] Z. Ma, Z. Zheng, J. Ye, J. Li, Z. Gao, S. Zhang, and X. Chen, “Emotion2vec: Self-supervised pre-training for speech emotion representation,” Findings of ACL, 2024.
- [41] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in LREC, 2020.
- [42] P. Liu, L. Wang, R. He, H. He, L. Wang, H. Zheng, J. Shi, T. Xiao, and Z. Wu, “Spmis: An investigation of synthetic spoken misinformation detection,” in SLT, 2024.
- [43] L. Wang, X. Lei, H. He, L. Wang, J. Shi, and Z. Wu, “Over-the-air adversarial attack detection: from datasets to defenses,” arXiv preprint arXiv:2509.09296, 2025.
- [44] Y. Gu, C. Wang, J. Zhang, X. Zhang, Z. Fang, H. He, and Z. Wu, “Singnet: Towards a large-scale, diverse, and in-the-wild singing voice dataset,” arXiv preprint arXiv:2505.09325, 2025.

