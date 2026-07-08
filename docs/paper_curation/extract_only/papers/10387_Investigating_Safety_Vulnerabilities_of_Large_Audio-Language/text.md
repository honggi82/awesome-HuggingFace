## INVESTIGATING SAFETY VULNERABILITIES OF LARGE AUDIO-LANGUAGE MODELS UNDER SPEAKER EMOTIONAL VARIATIONS

Bo-Han Feng1∗, Chien-Feng Liu1∗, Yu-Hsuan Li Liang1∗, Chih-Kai Yang1∗, Szu-Wei Fu2, Zhehuai Chen2, Ke-Han Lu1, Sung-Feng Huang2, Chao-Han Huck Yang2, Yu-Chiang Frank Wang2, Yun-Nung Chen1, Hung-yi Lee1

1National Taiwan University 2NVIDIA

# arXiv:2510.16893v1[cs.SD]19Oct2025

### ABSTRACT

Large audio-language models (LALMs) extend text-based LLMs with auditory understanding, offering new opportunities for multimodal applications. While their perception, reasoning, and task performance have been widely studied, their safety alignment under paralinguistic variation remains underexplored. This work systematically investigates the role of speaker emotion. We construct a dataset of malicious speech instructions expressed across multiple emotions and intensities, and evaluate several state-of-theart LALMs. Our results reveal substantial safety inconsistencies: different emotions elicit varying levels of unsafe responses, and the effect of intensity is non-monotonic, with medium expressions often posing the greatest risk. These findings highlight an overlooked vulnerability in LALMs and call for alignment strategies explicitly designed to ensure robustness under emotional variation, a prerequisite for trustworthy deployment in real-world settings.

Index Terms— Large audio-language models, safety, alignment

### 1. INTRODUCTION

Recent advances in large language models (LLMs) [1, 2] have revolutionized AI research, extending their impact to speech processing [3,4]. In particular, large audio-language models (LALMs) [5– 16] augment text-based LLMs with auditory understanding, opening new possibilities for multimodal models and speech technologies.

Although LALMs’ auditory perception [17], downstream performance [18–21], reasoning ability [22–24], and biases [25] have been extensively studied, research on their safety alignment has only just begun [26–29]. Safety alignment, which aims to prevent harmful outputs such as misinformation or self-harm, is particularly challenging for LALMs because their behavior can be influenced not only by semantic content but also by paralinguistic and acoustic cues. Prior work has shown that factors such as sound effects [27], languages [29], accents [27, 29], and intonation [27] can bypass safety mechanisms, yet the impact of speaker emotion, a fundamental aspect of communication, remains underexplored.

Investigating whether emotions can trigger safety vulnerabilities is essential for two reasons. First, if certain emotional expressions consistently elicit harmful behaviors, they may provide a new pathway for jailbreaking [30], where models are manipulated to bypass safety guardrails. Second, even when users act in good faith, they may unintentionally provoke unsafe responses from LALMs, which could in turn lead to real-world social harms.

Motivated by this, we systematically investigate how speaker emotion affects LALM safety. We construct a dataset of malicious

*Equal Contribution.

speech instructions synthesized with a text-to-speech model [31] under controlled conditions: each instruction is expressed across multiple emotions and intensities, with semantic content and speaker identity held identical. Human annotation is conducted to further verify the quality of the synthesized data.

Our experiments reveal that current LALMs exhibit significant safety inconsistencies across emotions. Some emotions elicit substantially more harmful responses, and medium intensities often provoke the most unsafe behaviors compared with both low and high intensities, surpassing both lower and higher levels. These results show that LALM safety alignment is neither stable nor robust against emotional variation, leaving safeguards vulnerable. Future work should explore training data and strategies explicitly designed to improve robustness against emotion-driven risks.

Overall, our contributions are: (1) the first study to examine the interaction between speakers’ emotions and the safety alignment of LALMs, and (2) uncovering the inconsistency of LALMs’ safety alignment under emotional variations, where certain emotions and intensities disproportionately provoke unsafe and harmful responses. Our dataset is available at https://huggingface. co/LALM-emotional-vulnerability.

### 2. RELATED WORK

Prior studies have examined how the safety alignment of LALMs can be compromised through the speech modality. Yang et al. [26] show that LALMs are more susceptible to safety failures from spoken inputs than from textual inputs with the same semantic content. Xiao et al. [27] and Hughes et al. [28] demonstrate that paralinguistic and acoustic cues such as tone, emphasis, speaking rate, and noises can further destabilize model behavior. Roh et al. [29] investigate jailbreak attempts that exploit variations in languages and accents of spoken instructions. There are also several benchmarks assessing the safety alignment of LALMs [32,33]. However, these efforts do not provide a systematic study of how emotional cues may introduce safety vulnerabilities, despite emotion being a central component of human communication. This gap motivates this study.

### 3. DATASET CONSTRUCTION

We describe the dataset construction process, illustrated in Fig. 1, to analyze the safety vulnerabilities of LALMs under different speaker emotions.

The dataset construction process consists of three phases: (1) harmful query collection, where we first gather harmful queries, (2) speech query synthesis, where the collected queries are verbalized as emotional speech using a text-to-speech (TTS) model, and (3)

[Figure 1]

Fig. 1. Overview of our dataset construction and experiments. AdvBench supplies prompts for the TTS model, and CREMA-D provides emotional reference speech. The generated queries are verified by calibrated annotators, and after safety testing on LALMs, two metrics, the non-refusal rate (NRR) and the unsafe rate (UR), are reported to assess the impact of emotions on LALMs’ response safety.

human annotation, where annotators label both the emotion and its corresponding intensity for subsequent analysis. The details of these phases are described in the following subsections.

### 3.1. Harmful Query Collection

We begin by collecting harmful queries to synthesize malicious speech instructions. Harmful queries are prompts that request unsafe information or actions, such as instructions for producing illegal drugs. Following prior work [26,27,32], we adopt AdvBench [34], which contains 520 textual queries across five security categories: misinformation, disinformation, toxicity, spam, and sensitive information. Its diversity and broad use in LLM safety research [35] make it a suitable basis for our study.

### 3.2. Speech Query Synthesis

We employ CosyVoice 2 0.5B [31] as the TTS model to synthesize emotional speech instructions from the harmful queries collected in AdvBench. The speech instructions are generated in six emotions: neutral, angry, disgusted, fearful, happy, and sad. To ensure that the synthesized instructions express the intended emotions, we use CREMA-D [36] as the reference dataset. CREMA-D provides detailed annotations for both emotion categories (the six emotions above) and emotion levels (low, medium, high, and unspecified).

Specifically, given a textual query, we synthesize emotional speech instructions by sampling a reference speech from CREMAD for each non-neutral emotion and each specified intensity level, while keeping speaker characteristics fixed. For the neutral case, a neutral reference sample is used. Each synthesized sample is then manually verified for naturalness, emotional expressiveness, and correctness of the annotated emotion level, as detailed in Sec. 3.3.

### 3.3. Human Annotation

To ensure that the synthesized speech instructions accurately convey the intended emotions and intensity levels, we conduct a manual inspection. Each annotator is instructed to (1) check that the speech is

natural and free from noticeable artifacts, (2) verify whether the synthesized speech faithfully represents the original textual query, and (3) assign both an emotion label (among the six defined categories) and an intensity label (low, medium, or high).

To promote quality and consistency, we introduce an annotation calibration step. Prior to the main annotation, annotators complete a trial using CREMA-D samples with predefined emotion and intensity labels. Only those who achieve at least 95% accuracy with respect to the ground-truth labels are allowed to proceed to the full annotation process. This calibration step aligns annotators’ criteria and ensures consistency throughout the dataset.

Each synthesized speech instruction is annotated by at least three annotators and retained only if they unanimously agree on the emotion and intensity (except neutral). Otherwise, it is re-synthesized and re-annotated until consensus is reached.

### 3.4. Dataset Statistics

The finalized dataset contains 8,320 malicious speech instructions, comprising 520 instructions with neutral emotion and 520×3 instructions for each of the other emotions, corresponding to the three intensity levels (low, medium, and high). Table 2 reports the statistics of the word counts in the original text prompts and the duration statistics of the generated speech samples across different emotions.

4. EXPERIMENTAL SETUPS

### 4.1. Investigated Models

In this work, we investigate how speaker emotions affect the safety alignment of several representative LALMs, including both open-source and proprietary models. The open-source set comprises Qwen2-Audio [5], Qwen2.5-Omni [6], DeSTA2.5-Audio [8], SALMONN [7] (7B and 13B), Typhoon-audio [9], SpeechGPT [10], and MiniCPM-o-2.6 [11], while the proprietary set includes Gemini1.5-flash [12] and Gemini-2.0-flash [13]. This broad coverage makes our analysis both comprehensive and representative. We provide the models with speech instructions along with a text prompt requiring them to respond to the speech. For comparison, we also evaluate

- Table 1. The non-refusal rate (NRR, %) and unsafe rate (UR, %) of the investigated LALMs. “Text-only” denotes results obtained by directly using the original textual queries instead of synthesized speech instructions. “µ,” “σ,” and “∆” indicate the average, standard deviation, and range (maximum minus minimum) of the metrics across the six emotions. For each metric, the highest values across the six emotions are highlighted in bold, and the second-highest values are underlined.

Models Text-only (↓) Neutral (↓) Angry (↓) Disgusted (↓) Fear (↓) Happy (↓) Sad (↓) µ (↓) σ (↓) ∆ (↓) Non-refusal Rate (NRR) (%)

Qwen2-Audio 0.96 6.92 2.95 4.87 5.51 4.94 4.68 4.98 1.29 3.97 Qwen2.5-Omni 1.92 0.38 0.13 0.64 0.70 0.70 0.32 0.48 0.24 0.57 DeSTA2.5-Audio 2.69 2.88 1.73 3.59 3.14 3.97 2.18 2.92 0.84 2.24 SALMONN 7B 19.81 82.69 91.28 86.60 89.61 86.19 85.32 86.95 3.08 8.59 SALMONN 13B 30.38 79.62 71.47 75.77 74.04 76.80 73.46 75.19 2.85 8.15 Typhoon-audio 56.35 84.23 78.72 78.65 79.74 78.01 78.21 79.59 2.35 6.22 SpeechGPT 8.85 34.62 22.31 30.39 30.77 28.98 30.58 29.61 4.04 12.31 MiniCPM-o-2.6 1.15 5.19 23.72 9.61 10.32 7.31 4.62 10.13 7.04 19.10

- Gemini-1.5-flash 2.88 3.65 3.97 4.49 4.04 4.74 3.78 4.11 0.42 1.09

- Gemini-2.0-flash 1.92 8.46 9.49 13.01 9.81 11.03 7.82 9.94 1.87 5.19 Unsafe Rate (UR) (%)

Qwen2-Audio 2.31 1.54 1.15 2.11 1.47 1.99 2.76 1.84 0.57 1.61 Qwen2.5-Omni 0.96 0.19 0.13 0.25 0.26 0.25 0.38 0.24 0.08 0.25 DeSTA2.5-Audio 2.88 0.38 0.38 0.64 1.03 0.71 0.83 0.66 0.26 0.65 SALMONN 7B 23.65 34.23 22.31 28.08 21.73 32.18 30.19 28.12 5.15 12.50 SALMONN 13B 48.46 72.88 70.77 81.03 72.88 71.15 72.56 73.55 3.78 10.26 Typhoon-audio 45.19 64.04 71.79 67.76 67.50 68.85 69.29 68.21 2.55 7.75 SpeechGPT 19.04 17.50 12.82 14.87 14.36 14.48 16.35 15.06 1.64 4.68 MiniCPM-o-2.6 1.15 3.27 8.01 5.58 4.62 4.68 3.14 4.88 1.79 4.87

- Gemini-1.5-flash 2.12 1.73 3.01 3.14 2.63 3.14 1.99 2.61 0.61 1.41

- Gemini-2.0-flash 1.35 3.08 2.76 4.81 2.89 3.98 2.82 3.39 0.83 2.05

- Table 2. Statistics of the word counts in original text prompts and the durations (seconds) of the speech instructions. AVG and SD denote the average and the standard deviation.

follow the malicious instructions, and are thus judged unsafe.

However, this pattern-matching approach captures only superficial refusals and overlooks the actual semantics of the responses. To address this, we employ the UR metric, which uses the LLM-asa-judge framework [39] to evaluate whether the response content is unsafe. Specifically, we adopt GPT-4o1 [1] as the judge to assess whether a response (1) follows the malicious query and (2) provides harmful or unsafe information. This ensures that only responses containing genuinely harmful content are classified as unsafe.

Textual Queries Dataset # of Samples Max Min AVG SD AdvBench 520 25 6 12.10 2.81 Speech Instructions Emotion # of Samples Max Min AVG SD

Finally, both NRR and UR are calculated as the proportion of queries in our dataset that elicit unsafe responses, according to the respective definitions of “unsafe” used in each metric. By comparing these metrics across instructions with different emotions and intensities, we quantify the extent to which the safety alignment of LALMs is influenced by speaker emotions.

Neutral 520 13.20 3.04 7.41 1.74 Angry 1560 14.28 3.04 7.41 1.74 Disgusted 1560 16.80 2.96 8.35 2.09 Fearful 1560 14.40 2.92 6.98 1.80 Happy 1560 14.96 2.52 7.17 2.07 Sad 1560 15.20 2.80 6.99 1.87

### 5. RESULTS 5.1. Main Results

Total 8320 16.80 2.52 7.34 1.98

a text-only setup, where the original text queries are given without speech. All experiments use greedy decoding.

### 4.2. Evaluation Metrics

Following prior work [37,38], we adopt two metrics to evaluate the safety alignment of LALMs’ responses: non-refusal rate (NRR) and unsafe rate (UR), as shown in Fig. 1. Both measure the extent to which models behave unsafely, but they differ in how “unsafe” is defined. The NRR is computed through pattern matching for typical refusal expressions such as “I am sorry” or “I cannot do this.” Responses that lack these refusal patterns are considered to accept and

We present the main results in Table 1. The models show a clear dichotomy: a relatively safer group with lower NRR and UR (Qwen2Audio, Qwen2.5-Omni, DeSTA2.5-Audio, MiniCPM-o-2.6, Gemini series) and a less safe group (SALMONN 7B and 13B, Typhoonaudio, SpeechGPT). This division indicates that the inherent safety alignment of certain models remains insufficiently robust.

When comparing performance across modalities, we find that most models exhibit higher NRR and UR under speech instructions (averaged across six emotions) than under text-only instructions. For example, SALMONN 7B shows an increase of 67.14% in NRR and 4.47% in UR when inputs shift from text to speech. This pattern in-

1gpt-4o-2024-08-06

dicates that the safety alignment of current LALMs is more vulnerable in the speech modality than in the textual modality, consistent with the findings of Yang et al. [26]. Ensuring that the safety alignment established in text-based LLMs is preserved during adaptation to speech, therefore, emerges as a critical direction for future work.

Within the speech modality, many models show substantial safety discrepancies across emotions, as indicated by large standard deviations (σ) and ranges (∆). These discrepancies highlight the instability of model safety under emotionally varied inputs. For example, SALMONN 7B and 13B display marked variability, with σ values of 3.08% and 2.85% for NRR and 5.15% and 3.78% for UR, together with ∆ values of 8.59% and 8.15% for NRR and 12.50% and 10.26% for UR. Such pronounced fluctuations suggest that the safety alignment of these models is highly sensitive to emotional cues, exposing potential vulnerabilities to both deliberate adversarial exploitation and inadvertent triggering of unsafe behaviors.

Crucially, this instability is not limited to relatively unsafe models. Even models with lower overall risk levels can fluctuate across emotions. For instance, MiniCPM-o-2.6 shows considerable σ and ∆ values for both metrics, despite maintaining moderately low mean scores. Likewise, Qwen2-Audio and Gemini-2.0-flash yield ∆ values that are comparable to their average NRRs and URs, indicating that safety alignment can remain unstable under emotional variation even when models appear sufficiently safe on average.

Finally, no single emotion consistently induces unsafe behavior across all models. Instead, each model reveals its own blind spot, namely a particular emotion that tends to trigger unsafe behaviors, suggesting that such variability is an inherent characteristic of current LALMs. These findings underscore the necessity of rigorously assessing safety instability before real-world deployment, in order to better understand model behavior and to guide the development of effective filtering and safeguarding mechanisms.

### 5.2. Effects of Emotion Intensity Levels

In Sec. 5.1, we observed that emotions can induce notable safety fluctuations and instabilities. A natural follow-up question is whether the intensity of emotional expression also plays a role. Since certain emotions already elicit more unsafe responses than others, it is reasonable to hypothesize that stronger intensities of these emotions may further amplify unsafe behaviors. In this section, we empirically investigate this hypothesis.

Given that the UR metric provides a more comprehensive assessment than the NRR metric by incorporating the semantic content of model responses, we focus our analysis on UR in this section. As described in Sec. 3, the dataset includes synthesized speech queries at three intensity levels. For each model, we examine the non-neutral emotion2 that produces the highest UR value to assess the impact of emotional intensity on safety alignment. The resulting URs across different intensity levels for these emotions are presented in Table 3.

We first observe that, beyond the variation across different emotions, some models also display substantial instability across different intensity levels of the same emotion. For instance, SALMONN 13B and MiniCPM-o-2.6 show large values of σ and ∆, indicating pronounced fluctuations between low, medium, and high intensities.

Contrary to our initial hypothesis, however, the results reveal that most LALMs reach their highest URs at medium intensity rather than at high intensity. This suggests that while certain emotions are indeed effective at inducing unsafe behavior, stronger expressions of those emotions do not necessarily further increase the likelihood of

2We exclude the neutral emotion from this analysis, as it lacks defined and annotated intensity levels.

Table 3. The unsafe rate (UR) of the investigated LALMs on speech instructions corresponding to the emotions that yield the highest UR in Table 1. “Low,” “Medium,” and “High” denote the respective intensity levels. “µ,” “σ,” and “∆” indicate the average, standard deviation, and range (maximum minus minimum) of the metrics across the three intensity levels. The highest UR values among the three levels for each model are marked in bold.

Models (Emotion) Low (↓) Med. (↓) High (↓) µ (↓) σ (↓) ∆ (↓) Qwen2-Audio (Sad) 2.31 3.46 2.50 2.76 0.62 1.15 Qwen2.5-Omni (Sad) 0.38 0.38 0.38 0.38 0 0 DeSTA2.5-Audio (Fear) 1.15 1.35 0.58 1.03 0.40 0.77 SALMONN 7B (Happy) 34.62 29.04 32.88 32.18 2.86 5.58 SALMONN 13B (Disgusted) 88.08 72.31 82.69 81.03 8.02 15.77 Typhoon-audio (Angry) 70.96 74.23 70.19 71.79 2.15 4.04 SpeechGPT (Sad) 15.58 17.69 15.77 16.35 1.17 2.11 MiniCPM-o-2.6 (Angry) 3.46 3.65 16.92 8.01 7.72 13.46

- Gemini-1.5-flash (Disgusted) 2.69 3.27 3.46 3.14 0.4 0.77

- Gemini-2.0-flash (Disgusted) 3.27 6.15 5.00 4.81 1.45 2.88

unsafe responses. Instead, medium-intensity expressions appear to elicit the most harmful responses.

Finally, different models exhibit distinct patterns. For example, Qwen2.5-Omni remains stable across intensities, whereas MiniCPM-o-2.6 is highly sensitive to high-intensity emotions, showing markedly higher UR compared with lower levels.

In summary, our findings reveal that the effect of emotional intensity on safety alignment is not monotonic: medium-intensity expressions often elicit the most harmful responses. This suggests that LALMs may be more vulnerable to subtle and naturalistic variations rather than exaggerated cues. Future work could explore whether this sensitivity stems from data distribution biases or insufficient robustness in alignment, and develop safety mechanisms explicitly resilient to paralinguistic variation.

### 6. CONCLUSION

Emotion is a crucial component of both human communication and human-AI interaction. In this work, we investigate whether emotions can induce safety vulnerabilities in LALMs. By evaluating several current LALMs with malicious speech instructions that share identical semantic content and speaker characteristics but differ in emotional expressions and intensities, we systematically uncover instabilities in their safety alignment under emotional cues.

We find that LALMs’ safety alignment varies substantially across emotions: some emotions elicit far more unsafe behaviors than others. However, even when an emotion induces such vulnerability, stronger expressions do not necessarily make models more unsafe. Instead, moderate intensities often pose the greatest risk. These findings highlight an inherent instability of LALMs under emotional cues, posing challenges for safe deployment if not properly understood and mitigated. Our study takes a first step toward uncovering this instability. Further investigation is needed to uncover the causes of this instability and explore possible mitigation strategies, which we consider an important direction for future work.

### 7. REFERENCES

- [1] OpenAI, “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [2] Abhimanyu Dubey et al., “The llama 3 herd of models,” arXiv e-prints, pp. arXiv–2407, 2024.

- [3] Rongjie Huang et al., “Audiogpt: Understanding and generating speech, music, sound, and talking head,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024, vol. 38, pp. 23802–23804.
- [4] Chun-Yi Kuan et al., “Speech-copilot: Leveraging large language models for speech processing via task decomposition, modularization, and program generation,” in 2024 IEEE Spoken Language Technology Workshop (SLT), pp. 1060–1067.
- [5] Yunfei Chu et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.
- [6] Jin Xu et al., “Qwen2. 5-omni technical report,” arXiv preprint arXiv:2503.20215, 2025.
- [7] Changli Tang et al., “SALMONN: Towards generic hearing abilities for large language models,” in The Twelfth International Conference on Learning Representations, 2024.
- [8] Ke-Han Lu et al., “Desta2.5-audio: Toward general-purpose large audio language model with self-generated cross-modal alignment,” arXiv preprint arXiv:2507.02768, 2025.
- [9] Potsawee Manakul et al., “Enhancing Low-Resource Language and Instruction Following Capabilities of Audio Language Models,” in Interspeech 2025, 2025, pp. 2083–2087.
- [10] Dong Zhang et al., “SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities,” in Findings of the Association for Computational Linguistics: EMNLP 2023, Dec. 2023, pp. 15757–15773.
- [11] Yuan Yao et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.
- [12] Gemini Team et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.
- [13] Gheorghe Comanici et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.
- [14] Ke-Han Lu et al., “Developing instruction-following speech language model without speech instruction-tuning data,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.
- [15] Chih-Kai Yang et al., “Building a taiwanese mandarin spoken language model: A first attempt,” arXiv preprint arXiv:2411.07111, 2024.
- [16] Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, and Lijuan Wang, “Stitch: Simultaneous thinking and talking with chunked reasoning for spoken language models,” arXiv preprint arXiv:2507.15375, 2025.
- [17] Chih-Kai Yang et al., “Audiolens: A closer look at auditory attribute perception of large audio-language models,” arXiv preprint arXiv:2506.05140, 2025.
- [18] Chien-yu Huang et al., “Dynamic-superb: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 12136–12140.
- [19] Chien-yu Huang et al., “Dynamic-SUPERB phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks,” in The Thirteenth International Conference on Learning Representations, 2025.

- [20] Yu-Xiang Lin et al., “A preliminary exploration with gpt-4o voice mode,” arXiv preprint arXiv:2502.09940, 2025.
- [21] Chih-Kai Yang, Neo S Ho, and Hung-yi Lee, “Towards holistic evaluation of large audio-language models: A comprehensive survey,” arXiv preprint arXiv:2505.15957, 2025.
- [22] S Sakshi et al., “MMAU: A massive multi-task audio understanding and reasoning benchmark,” in The Thirteenth International Conference on Learning Representations, 2025.
- [23] Chih-Kai Yang et al., “SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information,” in Interspeech 2025, 2025, pp. 1788–1792.
- [24] Ke-Han Lu et al., “Speech-IFEval: Evaluating instructionfollowing and quantifying catastrophic forgetting in speechaware language models,” in Interspeech 2025, 2025.
- [25] Yi-Cheng Lin et al., “Listen and speak fairly: a study on semantic gender bias in speech integrated large language models,” in 2024 IEEE Spoken Language Technology Workshop (SLT), 2024, pp. 439–446.
- [26] Hao Yang et al., “Audio is the achilles’ heel: Red teaming audio large multimodal models,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 9292–9306.
- [27] Erjia Xiao et al., “Tune in, act up: Exploring the impact of audio modality-specific edits on large audio language models in jailbreak,” arXiv e-prints, pp. arXiv–2501, 2025.
- [28] John Hughes et al., “Best-of-n jailbreaking,” CoRR, 2024.
- [29] Jaechul Roh, Virat Shejwalkar, and Amir Houmansadr, “Multilingual and multi-accent jailbreaking of audio llms,” arXiv preprint arXiv:2504.01094, 2025.
- [30] Sibo Yi et al., “Jailbreak attacks and defenses against large language models: A survey,” arXiv preprint arXiv:2407.04295, 2024.
- [31] Zhihao Du et al., “Cosyvoice 2: Scalable streaming speech synthesis with large language models,” arXiv preprint arXiv:2412.10117, 2024.
- [32] Yiming Chen et al., “Voicebench: Benchmarking llm-based voice assistants,” arXiv preprint arXiv:2410.17196, 2024.
- [33] Zifan Peng et al., “Jalmbench: Benchmarking jailbreak vulnerabilities in audio language models,” arXiv preprint arXiv:2505.17568, 2025.
- [34] Yangyi Chen et al., “Why should adversarial perturbations be imperceptible? rethink the research paradigm in adversarial nlp,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 11222– 11237.
- [35] Patrick Chao et al., “Jailbreaking black box large language models in twenty queries,” in 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025, pp. 23–42.
- [36] Houwei Cao et al., “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE transactions on affective computing, vol. 5, no. 4, pp. 377–390, 2014.
- [37] Zonghao Ying, Aishan Liu, Xianglong Liu, and Dacheng Tao, “Unveiling the safety of gpt-4o: An empirical study using jailbreak attacks,” arXiv preprint arXiv:2406.06302, 2024.

- [38] Xunguang Wang et al., “Sok: Evaluating jailbreak guardrails for large language models,” arXiv preprint arXiv:2506.10597, 2025.
- [39] Cheng-Han Chiang et al., “Can large language models be an alternative to human evaluations?,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), July 2023, pp. 15607–15631.

