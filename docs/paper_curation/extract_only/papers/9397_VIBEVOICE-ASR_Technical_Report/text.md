# arXiv:2601.18184v2[cs.SD]14Mar2026

## VIBEVOICE-ASR Technical Report

Zhiliang Peng∗, Jianwei Yu∗, Yaoyao Chang∗, Zilong Wang∗, Li Dong∗

Yingbo Hao, Yujie Tu, Chenyu Yang, Wenhui Wang, Songchen Xu, Yutao Sun Hangbo Bao, Weijiang Xu, Yi Zhu, Zehua Wang, Ting Song, Yan Xia, Zewen Chi Shaohan Huang, Liang Wang, Chuang Ding, Shuai Wang, Xie Chen, Furu Wei⋄ Microsoft Research https://aka.ms/GeneralAI

This report presents VIBEVOICE-ASR, a general-purpose speech understanding framework built upon VIBEVOICE [PYW+25], designed to address the persistent challenges of context fragmentation and multi-speaker complexity in long-form audio (e.g., meetings, podcasts) that remain despite recent advancements in short-form speech recognition. Unlike traditional pipelined approaches that rely on audio chunking, VIBEVOICE-ASR supports single-pass processing for up to 60 minutes of audio. It unifies Automatic Speech Recognition, Speaker Diarization, and Timestamping into a single end-to-end generation task. In addition, VIBEVOICE-ASR supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Furthermore, we introduce a prompt-based context injection mechanism that allows users to supply customized conetxt, significantly improving accuracy on domain-specific terminology and polyphonic character disambiguation.

##### Code: github.com/microsoft/VibeVoice Demo: aka.ms/VibeVoice-ASR HuggingFace Models Transformers Release Microsoft Foundry

|25.35<br><br>20.82<br><br>29.8 29.51<br><br>15.66<br><br>35.96<br><br>38.35<br><br>41.39<br><br>53.49<br><br>28.9<br><br>54.17<br><br>63.65 64.86 65.61<br><br>58.81<br><br>0<br><br>10<br><br>20<br><br>30<br><br>40<br><br>50<br><br>60<br><br>70<br><br>AISHELL4 AMI_IHM AMI_SDM AliMeeting MLC<br><br>tcpWER (Time-Constrained Permutation WER)↓<br><br>VibeVoice-ASR Gemini-2.5-Pro Gemini-3-Pro<br><br>|
|---|

|6.77<br><br>11.92 13.43<br><br>10.92<br><br>3.42<br><br>15.32<br><br>23.54 23.79<br><br>31.6<br><br>16.29<br><br>22.03<br><br>46.23<br><br>43.04<br><br>38.75<br><br>32.96<br><br>0<br><br>5<br><br>10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>35<br><br>40<br><br>45<br><br>50<br><br>AISHELL4 AMI_IHM AMI_SDM AliMeeting MLC<br><br>DER (Diarization Error Rate) ↓<br><br>VibeVoice-ASR Gemini-2.5-Pro Gemini-3-Pro<br><br>|
|---|

Figure 1: VIBEVOICE-ASR sets a new state-of-the-art for long-form speech understanding, consistently outperforming strong closed-source multimodal models (Gemini-2.5/3-Pro) across five public benchmarks. The results demonstrate superior accuracy in both speaker attribution (DER) and time-aligned transcription (tcpWER), particularly in complex multi-speaker environments.

#### 1 Introduction

Recent years have witnessed a paradigm shift in speech processing, driven by the integration of Large Language Models (LLMs) with acoustic encoders [CXZ+23]. While these large audio models

∗ Core contributors. ⋄ Contact person: fuwei@microsoft.com.

###### Rich Transcription

A S

Who When What

Welcome to Vibe…

- Speaker 1, 0 ~ 10.25,
- Speaker 2, 10.3 ~ 33.33,

Nice to meet…

…

| |
|---|

Speaker N, 3575.5 ~ 3600,

Let’s …

### VibeVoice - ASR

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

<begin>

|+<br><br>|
|---|

Optional Context

A S

[Figure 1]

[Figure 2]

[Figure 3]

60-minute Long-form Audio

Figure 2: The architectural overview of VIBEVOICE-ASR. VIBEVOICE-ASR processes 60-minute long-form audio in a single pass by ingesting continuous latents from dual-tokenizers alongside optional user-provided context. The output is a generated stream of Rich Transcription, explicitly interleaving Speaker ID (Who), Timestamps (When), and Content (What)

have achieved remarkable success in short-form speech recognition, transcribing and analyzing longform audio—such as hour-long meetings, podcasts, and academic lectures—remains a formidable challenge.

The prevailing approach to long-form audio involves cascaded pipelines that segment continuous speech into short clips (typically < 30 seconds) for independent processing [HSW+24, BHHZ23, BYC+20]. While practical, this "divide-and-conquer" strategy suffers from two fundamental limitations: Context Fragmentation and Pipeline Complexity. First, independently processing segments severs global semantic dependencies, causing the model to lose track of cross-sentence context, which is fatal for disambiguating homophones or resolving coreferences in extended dialogue. Second, traditional systems treat Automatic Speech Recognition (ASR), Speaker Diarization, and Timestamping as separate tasks managed by disjoint models. Reconciling their outputs often requires complex heuristics, leading to error propagation where a failure in segmentation or diarization corrupts the final transcript.

To bridge this gap, we introduce VIBEVOICE-ASR, a unified, general-purpose framework designed for high-fidelity long-form speech understanding. Built upon the VibeVoice architecture [PYW+25], our system fundamentally abandons the sliding-window paradigm in favor of a single-pass approach. By leveraging an ultra-low frame rate tokenizer (7.5Hz), VIBEVOICE-ASR compresses an hour of audio into a sequence length that fits comfortably within the context window of modern LLMs. This allows the model to attend to the entire global context of a 60-minute session simultaneously, ensuring semantic coherence and consistent speaker tracking without the need for external clustering algorithms. Concurrent with the development of VIBEVOICE-ASR, a number of related research efforts have emerged [HSZ26, YCD+25, SXF+25, YLY+26]. Nevertheless, the majority of these works have not made their models publicly available.

VIBEVOICE-ASR reformulates long-form transcription as an end-to-end generation task, as shown in Figure 2. Instead of outputting plain text, it generates a structured Rich Transcription stream that explicitly interleaves speaker identities (“Who"), precise timestamps (“When"), and speech content (“What"). Furthermore, acknowledging the diverse needs of real-world applications, we introduce a prompt-based context injection mechanism. This allows users to supply customized context—ranging from hotword lists to background descriptions—significantly enhancing the model’s ability to recognize domain-specific terminology and handle complex code-switching scenarios.

#### 2 Method

- 2.1 Overview

Figure 2 presents the architectural overview of VIBEVOICE-ASR. We formulate long-form speech understanding as a language modeling task. The model takes a sequence of continuous audio embeddings, encoded from from the pre-trained Acoustic and Semantic encoders, as its primary input. To enable context-aware capabilities, optional text prompts (e.g., hotwords or background information) can be prepended to the audio sequence.

These inputs are processed by a decoder-only Large Language Model backbone (e.g., Qwen 2.5 [YYZ+24]) to autoregressively generate the target sequence. Distinct from conventional ASR models that output plain text, VIBEVOICE-ASR is designed to produce a Rich Transcription. As illustrated in the output stream of Figure 2, the model generates a structured sequence that explicitly interleaves speaker identity (“Who”), temporal boundaries (“When”), and speech content (“What”), enabling simultaneous recognition, diarization, and timestamping in a single pass.

- 2.2 Speech Tokenizer

In this work, we directly employ the pre-trained dual-tokenizers from VIBEVOICE [PYW+25], which integrates an Acoustic Tokenizer for spectral fidelity and a Semantic Tokenizer for linguistic alignment. The Acoustic tokenizer, inspired by σ-VAE [SBW+24], applies a hierarchical design with a cumulative 3200× downsampling rate to the 24kHz input, yielding an extremely compact representation of approximately 7.5 tokens per second. Meanwhile, the Semantic module extracts deterministic content features aligned with textual semantics. Note we only use tokenizer encoders here. This ultra-low frame rate is pivotal, as a one-hour continuous audio session translates to:

3600 seconds × 7.5 tokens/sec = 27,000 tokens, (1) which fits comfortably within the single-pass context window of modern LLMs.

- 2.3 VIBEVOICE-ASR

- 2.3.1 Pre-training

We use the data processing pipeline proposed in VIBEVOICE [PYW+25, YCB+24] to obtain the initial data corpus. The pre-training data distribution can be found in Figure 3. The pipeline consists of three stages: segmentation and transcription, diarization, and quality filtering. Long recordings are first segmented using Silero voice activity detection (VAD) into clips of up to 30 seconds, followed by transcription with Whisper-large-v3-turbo [RKX+23] to obtain punctuated text and word-level timestamps; segment boundaries are further refined by splitting at punctuation end timestamps (e.g., [.?!]) to better align with speaker turns. Speech diarization is then performed using the vblinkp model from the WeSpeaker toolkit [WLW+23], where speaker embeddings are extracted from overlapping frames (1.5 s window, 0.75 s hop), clustered with HDBSCAN [CMS13], and refined by merging clusters whose centroids have a cosine similarity greater than 0.67, yielding final speaker turn annotations. Finally, to ensure annotation reliability, segments are re-transcribed using a secondary ASR model [XJM+23], and recordings are discarded if more than 30% of segments have a WER exceeding 20%, if speech accounts for less than 60% of the total duration.

To ensure the effectiveness of the data processing pipeline, we conducted a comparative study between our pipeline and two widely adopted audio processing pipelines, WhisperX [BHHZ23] and Emilia [HSW+24]. The evaluation is performed on three commonly used public multi-speaker meeting datasets—AMI [CAB+05], AliMeeting [YZF+22], and AISHELL-4 [FCL+21]—and reports both diarization error rate (DER) and diarization invariant word error rate (WER). For a fair comparison, we disable the data-filtering module in Emilia, as its default configuration removes a substantial portion of the audio samples.

- As shown in Table 1, the proposed data pipeline consistently achieves lower DER and WER than both baseline systems across the majority of evaluated datasets. These results indicate that our pipeline provides more robust segmentation, diarization, and transcription performance under diverse acoustic conditions.

Table 1: DER and WER comparison across different data pipelines. Model AISHELL4 AMI-IHM AMI-SDM AliMeeting

DER WER DER WER DER WER DER WER

WhisperX 14.55 29.69 18.27 24.12 23.05 39.65 35.53 36.62 Emilia 16.58 49.40 35.44 47.85 46.55 61.70 25.57 54.27 Ours pipeline 16.93 18.99 15.46 23.22 17.78 28.40 25.34 30.82

We employed a curriculum learning strategy for the LLM input sequence length, progressively increasing from 8,192 to 65,536 tokens.

##### 2.3.2 Supervised Fine-Tuning (SFT)

Since the pre-training stage predominantly relies on pseudo-labeled data, the SFT phase is critical for aligning the model with precise instruction-following behaviors. We carefully curate a high-quality dataset composition strategy, categorized into three distinct sources:

High-Quality Speech and Music Benchmarks. To establish a robust baseline for conversational speech recognition and speaker diarization, we utilize established datasets including the training splits of MLC-SLM [MGS+25] and Fisher [CMW04]. These provide high quality labels for multi-speaker interactions. Additionally, we incorporate the open-source synthesized music dataset Muse [JCX+26] as an independent subset. The inclusion of this music data allows the model to learn music-specific acoustic features, explicitly optimizing its performance and robustness when handling musical segments.

Context-Aware Synthetic Data Pipeline. A key capability of VIBEVOICE-ASR is utilizing user-provided Contextual Information—ranging from specific entities to complete sentences and background descriptions—to guide recognition. To bridge the lack of such paired data in real-world scenarios, we constructed a synthetic pipeline:

- • Context-Driven Script Generation: We employ GPT-5 [SFP+25] to generate complex dialogue scripts containing specific entities, technical terms, and cross-lingual content (English, Chinese, and intra-sentential code-switching). Crucially, GPT-5 simultaneously generates the corresponding contextual reference text (e.g., keyword lists, related sentences, or background paragraphs) used to prompt the ASR model.
- • Audio Synthesis: We leverage the VIBEVOICE engine to synthesize high-fidelity multispeaker audio. The synthesis predominantly targets Chinese, English, and complex EnglishChinese code-switching scenarios, fully exploiting VIBEVOICE’s superior capabilities in modeling these specific linguistic distributions and transitions.
- • Quality Filtering: We perform a closed-loop verification where the synthesized speech is transcribed back; samples exceeding a WER threshold are discarded to prevent noise injection. After that, we obtains about 6,000 hours synthesized audio.

Long-Form Transcription Restoration. Existing high-quality datasets are predominantly short (<30 minutes), creating a distribution shift for long-form applications. While we recall long-duration samples (>50 minutes) from our pre-training corpus, their original transcriptions—derived from our chunk-wise pipelines—also suffer from context fragmentation. To address this, we employ GPT-5 as a text refiner to rewrite and merge disjointed transcriptions into coherent, globally consistent long texts ("Global Semantic Rectification").

Furthermore, to handle the non-speech intervals inherent in long-duration recordings, we utilize GPT-Audio2 to automatically annotate these segments with general acoustic tags. Specifically, we label events such as [Unintelligible Speech], [Music], [Human Sounds], [Environmental Sounds], [Noise], and [Silence]. This explicit tagging strategy provides direct supervision for non-speech intervals, designed to prevent the model from hallucinating text during silence or background noise.

2https://platform.openai.com/docs/models/gpt-4o-audio-preview

To balance the VIBEVOICE-ASR’s capabilities across standard recognition, music robustness, context awareness, and long-form coherence, we apply a strategic data mixing ratio. Specifically, the sampling weights for Standard Benchmarks, Music Data, Synthetic Data, and Refined Long-Form Data are set to 0.5 : 0.1 : 0.1 : 0.3, respectively.

#### 3 Results

We follow the MeetEval3 evaluation protocol and report four complementary metrics that capture different aspects of multi-speaker transcription quality.

Diarization Error Rate (DER) measures the accuracy of speaker attribution by accounting for speaker confusion, missed speech, and false alarm speech, and thus directly evaluates the model’s ability to answer who speaks when.

Word Error Rate (WER) ignores speaker labels and timing information and computes the standard word-level error rate over the entire transcription, serving as a measure of pure speech recognition accuracy (what) independent of diarization performance.

Concatenated minimum-Permutation WER (cpWER) evaluates transcription accuracy under speaker permutation invariance by concatenating all utterances belonging to the same speaker and computing the minimum WER over all possible speaker permutations; this metric jointly reflects content recognition accuracy and speaker consistency, while being insensitive to local time alignment errors.

Time-Constrained minimum-Permutation WER (tcpWER) further extends cpWER by enforcing temporal alignment constraints, such that words are only matched if they occur within a predefined temporal collar, making tcpWER sensitive to both speaker attribution and word-level timing accuracy and thus jointly evaluating who, what, and when.

We select Gemini-2.5-Pro and Gemini-3-Pro as comparison baselines, as they represent state-of-theart large-scale multimodal foundation models capable of jointly predicting timestamps, speaker labels, and transcription content. During our experiments, we observe that Gemini models exhibit substantial timestamp inaccuracies and occasional content hallucinations when processing long-form audio inputs. To ensure a fair and stable comparison, we therefore segment the test audio into 240-second chunks before feeding them to the Gemini models. In contrast, VIBEVOICE-ASR processes the entire audio recording in a single pass, without requiring chunk-wise inference.

Table 2: Overall diarization and ASR results across datasets and languages.

Gemini-2.5-Pro Gemini-3-Pro VIBEVOICE-ASR

Dataset Language DER cpWER tcpWER WER DER cpWER tcpWER WER DER cpWER tcpWER WER AISHELL-4 Chinese 15.32 31.59 35.96 22.42 22.03 27.43 54.17 22.75 6.77 24.99 25.35 21.40 AMI-IHM English 23.54 29.57 38.35 18.48 46.23 22.34 63.65 17.61 11.92 20.41 20.82 18.81 AMI-SDM English 23.79 34.78 41.39 22.35 43.04 26.91 64.86 22.09 13.43 28.82 29.80 24.65 AliMeeting Chinese 31.60 41.64 53.49 27.43 38.75 32.84 65.61 26.75 10.92 29.33 29.51 27.40

|English French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese|20.67 16.23 26.72 9.76<br><br>7.66 23.06 24.60 17.17 18.19 30.36 39.43 17.76 12.55 16.88 25.20 12.87 20.40 30.41 37.36 16.58 17.57 19.23 29.81 10.18 20.86 30.03 40.20 20.15<br><br>5.35 14.26 16.59 10.74<br><br>9.10 13.82 17.49 9.09 15.54 20.84 30.28 14.84 14.65 16.71 27.28 12.33<br><br>|30.88 12.85 57.64 10.19 40.82 22.02 71.11 18.71 42.14 23.56 73.86 19.39 23.45 15.59 49.89 13.32 59.68 21.96 81.41 18.47 39.28 19.39 57.33 11.21 39.17 23.29 85.44 20.10 22.76 13.05 51.89 10.31 25.54 12.11 43.72 9.36<br><br>22.09 14.59 39.54 12.03 32.24 13.15 60.43 11.53<br><br>|4.28 11.48 13.02 7.99<br><br>3.80 18.80 19.64 15.21<br><br>1.04 17.10 17.26 16.30<br><br>2.08 15.76 15.91 13.91<br><br><br>0.82 15.33 15.41 14.69<br><br>4.52 15.35 16.07 9.65<br><br><br>7.98 29.91 31.65 21.54 0.90 12.94 12.98 12.40 2.67 10.51 11.71 8.04<br><br>4.09 14.91 15.57 13.61 0.16 14.57 14.57 14.43<br><br>|
|---|---|---|---|
|AVERAGE<br><br>|16.29 20.37 28.90 13.05|32.96 16.38 58.81 13.11<br><br>|3.42 14.81 15.66 12.07|

MLC-Challenge

- As shown in Table 2, VIBEVOICE-ASR consistently outperforms Gemini-2.5-Pro and Gemini3-Pro in terms of DER and tcpWER across all evaluated datasets, demonstrating substantially stronger speaker modeling and more accurate alignment of speaker turns over time. On the cpWER metric, which more directly reflects the model’s ability to maintain speaker consistency, our model

3https://github.com/fgnt/meeteval

achieves the best performance on 11 out of 16 evaluation settings, significantly outperforming both Gemini variants and indicating more reliable speaker differentiation in multi-speaker conditions. Regarding WER, our model attains the lowest error rate on 8 out of 16 settings, while exhibiting only marginal degradation on the remaining datasets. Overall, these results indicate that VIBEVOICEASR achieves a better balance between content recognition accuracy and robust speaker-aware transcription, with particularly strong advantages in speaker attribution, temporal consistency, and multilingual generalization.

#### 4 Conclusion and Limitations

In this report, we presented VIBEVOICE-ASR, a unified single-pass framework that effectively solves context fragmentation in long-form speech understanding. Beyond technical contributions, we commit to comprehensive open-sourcing, releasing the model weights, fine-tuning pipelines, and high-performance inference code (e.g., vLLM [KLZ+23] support). By democratizing access to these tools, we aim to empower the research community to address the SFT gaps in low-resource languages and adapt the framework to diverse downstream applications, ultimately fostering a more inclusive and advanced speech ecosystem.

Despite these advancements, VIBEVOICE-ASR has several limitations that guide future research:

- • Multilingual Forgetting in SFT: While our pre-training covered over 50 languages, the SFT phase predominantly focused on English, Chinese, and code-switching data. Consequently, the model may experience performance degradation on low-resource languages absent from the instruction tuning stage. We hope our open-source fine-tuning code will encourage the community to bridge this gap.
- • Overlapping Speech: The current architecture generates a serialized output stream and does not explicitly handle overlapping speech (the "cocktail party problem"). In scenarios where multiple speakers talk simultaneously, the model tends to transcribe the dominant speaker, potentially missing secondary information. Future iterations will explore separation-aware modeling to address this challenge.

#### Acknowledge

We thank Ruibin Yuan, Tao Zhang and Zhengwei Huang for their in-depth discussions during the research and development of VIBEVOICE-ASR.

#### References

[BHHZ23] Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman. Whisperx: Time-accurate speech transcription of long-form audio. arXiv preprint arXiv:2303.00747, 2023.

[BYC+20] Hervé Bredin, Ruiqing Yin, Juan Manuel Coria, Gregory Gelly, Pavel Korshunov, Marvin Lavechin, Diego Fustes, Hadrien Titeux, Wassim Bouaziz, and Marie-Philippe Gill. Pyannote. audio: neural building blocks for speaker diarization. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7124–7128. IEEE, 2020.

[CAB+05] Jean Carletta, Simone Ashby, Sebastien Bourban, Mike Flynn, Mael Guillemot, Thomas Hain, Jaroslav Kadlec, Vasilis Karaiskos, Wessel Kraaij, Melissa Kronenthal, et al. The ami meeting corpus: A pre-announcement. In International workshop on machine learning for multimodal interaction, pages 28–39. Springer, 2005.

[CMS13] Ricardo JGB Campello, Davoud Moulavi, and Jörg Sander. Density-based clustering based on hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and data mining, pages 160–172. Springer, 2013.

[CMW04] Christopher Cieri, David Miller, and Kevin Walker. The fisher corpus: A resource for the next generations of speech-to-text. In LREC, volume 4, pages 69–71, 2004.

[CXZ+23] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

[FCL+21] Yihui Fu, Luyao Cheng, Shubo Lv, Yukai Jv, Yuxiang Kong, Zhuo Chen, Yanxin Hu, Lei Xie, Jian Wu, Hui Bu, et al. Aishell-4: An open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario. arXiv preprint arXiv:2104.03603, 2021.

[HSW+24] Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, et al. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 885–890. IEEE, 2024.

[HSZ26] Mingyue Huo, Yiwen Shao, and Yuheng Zhang. Tagspeech: End-to-end multispeaker asr and diarization with fine-grained temporal grounding. arXiv preprint arXiv:2601.06896, 2026.

[JCX+26] Changhao Jiang, Jiahao Chen, Zhenghao Xiang, Zhixiong Yang, Hanchen Wang, Jiabao Zhuang, Xinmeng Che, Jiajun Sun, Hui Li, Yifei Cao, et al. Muse: Towards reproducible long-form song generation with fine-grained style control. arXiv preprint arXiv:2601.03973, 2026.

[KLZ+23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[MGS+25] Bingshen Mu, Pengcheng Guo, Zhaokai Sun, Shuai Wang, Hexin Liu, Mingchen Shao, Lei Xie, Eng Siong Chng, Longshuai Xiao, Qiangze Feng, et al. Summary on the multilingual conversational speech language model challenge: Datasets, tasks, baselines, and methods. arXiv preprint arXiv:2509.13785, 2025.

[PYW+25] Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, et al. Vibevoice technical report. arXiv preprint arXiv:2508.19205, 2025.

[RKX+23] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

[SBW+24] Yutao Sun, Hangbo Bao, Wenhui Wang, Zhiliang Peng, Li Dong, Shaohan Huang, Jianyong Wang, and Furu Wei. Multimodal latent language modeling with next-token diffusion. arXiv preprint arXiv:2412.08635, 2024.

[SFP+25] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[SXF+25] Mohan Shi, Xiong Xiao, Ruchao Fan, Shaoshi Ling, and Jinyu Li. Train short, infer long: Speech-llm enables zero-shot streamable joint asr and diarization on long audio. arXiv preprint arXiv:2511.16046, 2025.

[WLW+23] Hongji Wang, Chengdong Liang, Shuai Wang, Zhengyang Chen, Binbin Zhang, Xu Xiang, Yanlei Deng, and Yanmin Qian. Wespeaker: A research and production oriented speaker embedding learning toolkit. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[XJM+23] Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, and Boris Ginsburg. Efficient sequence transduction by jointly predicting tokens and durations. In International Conference on Machine Learning, pages 38462–38484. PMLR, 2023.

[YCB+24] Jianwei Yu, Hangting Chen, Yanyao Bian, Xiang Li, Yi Luo, Jinchuan Tian, Mengyang Liu, Jiayi Jiang, and Shuai Wang. Autoprep: An automatic preprocessing framework for in-the-wild speech data. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1136–1140. IEEE, 2024.

[YCD+25] Han Yin, Yafeng Chen, Chong Deng, Luyao Cheng, Hui Wang, Chao-Hong Tan, Qian Chen, Wen Wang, and Xiangang Li. Speakerlm: End-to-end versatile speaker diarization and recognition with multimodal large language models. arXiv preprint arXiv:2508.06372, 2025.

[YLY+26] Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, et al. Moss transcribe diarize: Accurate transcription with speaker diarization. arXiv preprint arXiv:2601.01554, 2026.

[YYZ+24] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[YZF+22] Fan Yu, Shiliang Zhang, Yihui Fu, Lei Xie, Siqi Zheng, Zhihao Du, Weilong Huang, Pengcheng Guo, Zhijie Yan, Bin Ma, et al. M2met: The icassp 2022 multi-channel multi-party meeting transcription challenge. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6167–6171. IEEE, 2022.

#### A Language Distribution of Training Data

66.651067%

English

14.388077%

Chinese

6.281633%

Spanish

2.377301%

Portuguese

1.852176%

German

0.953292%

Japanese

0.912063%

Korean

0.898955%

French

0.691947%

Russian

0.575294%

Indonesian

0.508351%

Swedish

0.447609%

Italian

0.391830%

Hebrew

0.275293%

Dutch

0.254292%

Polish

0.210149%

Norwegian

0.206286%

Turkish

0.197261%

Thai

0.191945%

Arabic

0.181721%

Hungarian

0.175694%

Catalan

0.144024%

Czech

0.135779%

Danish

0.115226%

Persian

0.100209%

Afrikaans

0.098980%

Hindi

0.093302%

Finnish

0.065677%

Estonian

0.065011%

Afar

0.063801%

Greek

0.063136%

Romanian

0.056151%

Vietnamese

0.056144%

Bulgarian

0.044601%

Icelandic

0.042502%

Slovenian

0.040306%

Slovak

0.039944%

Lithuanian

0.025635%

Swahili

0.023807%

Ukrainian

0.015401%

Kalaallisut

0.012667%

Latvian

0.012208%

Croatian

0.011705%

Nepali

0.010376%

Serbian

0.009273%

Filipino

0.008435%

Yiddish

0.007840%

Malay

0.005445%

Urdu

0.004618%

Mongolian

0.003093%

Armenian

0.002467%

Javanese

10 2 10 1 100 101 102 Percentage (%)

Figure 3: Language distribution in the training data.

