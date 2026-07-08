[Figure 1]

August, 2024

[Figure 2]

arXiv:2408.03900v1[cs.CL]7Aug2024

# Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond

#### Beomseok Lee   , Ioan Calapodescu , Marco Gaido , Matteo Negri , Laurent Besacier 

 University of Trento, Italy  NAVER LABS Europe, France  Fondazione Bruno Kessler, Italy

## Abstract

We present Speech-MASSIVE, a multilingual Spoken Language Understanding (SLU) dataset comprising the speech counterpart for a portion of the MASSIVE textual corpus. Speech-MASSIVE covers 12 languages from diﬀerent families and inherits from MASSIVE the annotations for the intent prediction and slot-ﬁlling tasks. Our extension is prompted by the scarcity of massively multilingual SLU datasets and the growing need for versatile speech datasets to assess foundation models (LLMs, speech encoders) across languages and tasks. We provide a multimodal, multitask, multilingual dataset and report SLU baselines using both cascaded and end-to-end architectures in various training scenarios (zero-shot, few-shot, and full ﬁne-tune). Furthermore, we demonstrate the suitability of Speech-MASSIVE for benchmarking other tasks such as speech transcription, language identiﬁcation, and speech translation. The dataset, models, and code are publicly available at:

https://github.com/hlt-mt/Speech-MASSIVE

- 1. Introduction Multilingual speech corpora have limited coverage of speech-related tasks, primarily focusing on automatic speech recognition (ASR) [1, 5, 16, 23] and speech translation (ST) [7,10,18,19], while neglecting spoken language understanding (SLU – the task of extracting semantic information from spoken utterances, which typically involves subtasks like intent detectionand slot ﬁlling). Unlike text processing, where extensive eﬀorts in natural language understanding (NLU) have led to resources covering a wide range of languages [8,13,14, 21], SLU datasets are mainly English-centric [3], with few exceptions [6,11,12].

Our goal is to bridge the gap in multilingual SLU drawing inspiration from [11] and collecting speech recordings in multiple languages. We start with the MASSIVE NLU (i.e. textual) dataset [8], an ideal foundation due to its size, domain diversity, and broad coverage of languages, intent, and slot types. Developed by commissioning professional translators to localize the English SLURP dataset [3] into 51 languages, MASSIVE comprises 1M labeled utterances spanning 18 domains, with 60 intents and 55 slots. Our contribution, SpeechMASSIVE, spans 12 languages from diverse families: Arabic, German, Spanish, French, Hungarian, Korean, Dutch, Polish, European Portuguese, Russian, Turkish, and Vietnamese. It also facilitates evaluation across various speech tasks beyond SLU, including ASR, ST, and language identiﬁcation (LID). We release Speech-

MASSIVE publicly under CC-BY-NC-SA 4.0 license.1

Besides detailing the creation process involving a crowdsourcing-based protocol for data collection and quality control, this paper presents baseline SLU results on Speech-MASSIVE. Our results with both cascade and end-to-end architectures trained in diﬀerent conditions (zero-shot, few-shot, full ﬁne-tune) will enable future comparisons and tracking SLU advancements compared to the more mature ﬁeld of NLU. Lastly, we showcase Speech-MASSIVE’s versatility through additional experiments on ASR, LID, and ST.

## 2. Speech-MASSIVE

### 2.1. Data collection and validation process

We created the speech counterpart of textual MASSIVE data by recruiting native speakers through the Proliﬁc crowdsourcing platform.2 A ﬁrst group of workers was instructed to record the spoken version of MASSIVE sentences with guidelines emphasizing the importance of accurate and natural reading, as well as proper recording conditions and strict adherence to the corresponding text. To ensure high ﬁnal data quality, a second group of native speakers validated the recorded utterances. During validation, participants were directed to read the original text, listen to the recording, and

[Figure 3]

- 1https://hf.co/datasets/FBK-MT/Speech-MASSIVE
- 2https://www.prolific.com, Compensated £9 per hour.

[Figure 4]

Corresponding author: beomseok.lee@unitn.it

label it as valid or invalid. Those marked as invalid underwent a second iteration of this two-step (recording and validation) process. After the second iteration, the process concluded, irrespective of the outcome of the second validation phase, to avoid potentially endless cycles. This decision was also informed by the observation that, upon inspecting the invalid recordings, we found some were marked as such not due to a lack of adherence of the speech to the text but because of grammatical errors in the original MASSIVE dataset text. Correcting these errors was beyond the scope of our work.

To further enhance the reliability of the collected dataset, we implemented two additional precautions. During the recording phase, we instructed participants to review their own recordings before proceeding to the next sample, allowing them to re-record if the audio was not properly acquired. Additionally, in the validation step, four speech utterances were chosen from Common Voice [1] and inserted among the samples for validation. Out of these four quality control samples, two intentionally featured audio-transcript mismatches to be marked as invalid. The other two cases had perfect audio-transcript alignment to be marked

- as valid. Care was taken to select quality control samples with clear and intelligible audio. Validation results from a Proliﬁc user were retained only if they accurately assessed all four quality control samples. Any mistakes led to the disregarding of their validations, requiring the entire set of samples from that user to be re-validated by other participants.

2.2. Overall statistics

We chose 12 languages based on various criteria. Initially, we considered the number of registered users on Proliﬁc, sorting the 51 languages covered in MASSIVE. Languages with fewer than 200 users were excluded to ensure suﬃcient worker participation to complete the entire acquisition and validation process in reasonable time. Italian was also excluded due to the availability of the full dataset elsewhere [11]. Finally, with an eye

- at the balance between budget considerations and linguistic diversity, from the remaining 18 languages we selected Arabic, German, Spanish, French, Hungarian, Korean, Dutch, Polish, European Portuguese, Russian, Turkish, and Vietnamese.

We collected speech recordings for MASSIVE’s development and test splits. Acquiring the full training dataset (11,514 utterances for each of the 12 languages) exceeded our budget. In a concession, our emphasis was placed on acquiring comprehensive training data for French and German, while we obtained limited few-

Table 1: Speech-MASSIVE’s overall statistics. ‘# hrs’ displays the recording duration for all samples (including invalid), while ‘# spk (Male/Female/Unknown)’ indicates the number of speakers for all the samples (including invalid). The last 2 columns (‘WER’, and ‘CER’) measures Whisper ASR performance.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

total # spk (M/F/U)

lang split # sample # valid # hrs

WER CER

[Figure 14]

train-115 115 115 0.14 8 (4/4/0) - -

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

ar

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

dev 2033 2027 2.12 36 (22/14/0) 31.75 14.43 test 2974 2962 3.23 37 (15/17/5) 34.19 15.85

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

train-115 115 115 0.15 7 (3/4/0) - train-full 11514 11201 12.61 117 (50/63/4) - -

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

de

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

dev 2033 2032 2.33 68 (35/32/1) 11.24 3.96 test 2974 2969 3.41 82 (36/36/10) 11.84 4.16

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

train-115 115 115 0.13 7 (3/4/0) - -

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

es

dev 2033 2024 2.53 109 (51/53/5) 7.61 3.00 test 2974 2948 3.61 85 (37/33/15) 8.95 3.76

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

train-115 115 115 0.12 103 (50/52/1) - train-full 11514 11481 12.42 103 (50/52/1) - -

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

fr

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

dev 2033 2031 2.20 55 (26/26/3) 10.20 4.42 test 2974 2972 2.65 75 (31/35/9) 11.09 4.71

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

train-115 115 115 0.12 8 (3/4/1) - dev 2033 2019 2.27 69 (33/33/3) 25.96 10.93 test 2974 2932 3.30 55 (25/24/6) 20.98 6.01

hu

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

train-115 115 115 0.14 8 (4/4/0) - -

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

ko

dev 2033 2032 2.12 21 (8/13/0) 25.29 7.13 test 2974 2970 2.66 31 (10/18/3) 26.42 8.04

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

train-115 115 115 0.12 7 (3/4/0) - -

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

nl

dev 2033 2032 2.14 37 (17/19/1) 11.03 3.98 test 2974 2959 3.30 100 (48/49/3) 10.52 3.82

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

train-115 115 115 0.10 7 (3/4/0) - -

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

pl

dev 2033 2024 2.24 105 (50/52/3) 9.94 4.88 test 2974 2933 3.21 151 (73/71/7) 12.58 6.22

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

train-115 115 115 0.12 8 (4/4/0) - -

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

pt

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

dev 2033 2031 2.20 107 (51/53/3) 11.73 5.10 test 2974 2967 3.25 102 (48/50/4) 12.11 5.13

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

train-115 115 115 0.12 7 (3/4/0) - -

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

ru

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

dev 2033 2032 2.25 40 (7/31/2) 8.55 4.06 test 2974 2969 3.44 51 (25/23/3) 8.99 4.57

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

train-115 115 115 0.11 6 (3/3/0) - -

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

tr

dev 2033 2030 2.17 71 (36/34/1) 16.65 4.56 test 2974 2950 3.00 42 (17/18/7) 18.06 5.05

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

train-115 115 115 0.11 7 (2/4/1) - -

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

vi

dev 2033 1978 2.10 28 (13/14/1) 16.65 10.5 test 2974 2954 3.23 30 (11/14/5) 14.94 9.77

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

shot training data consisting of 115 utterances from the training set for the remaining 10 languages (train115 split).

Columns 1-6 of Table 1 provide statistics for the collected dataset, including, for each language, the available data splits, the number of recordings, hours of speech, and speakers (total, male, female and unknown). The “# valid” column indicates the count of human-validated utterances for each data split after the two iterations. As a few speech recordings remained invalidated after our two recording-validation cycles, we retained for each utterance the candidate with the lowest Word Error Rate (WER) as transcribed using Whisper [17]. This ensures speech availability for all MASSIVE utterances, even if some may not perfectly align with the reference transcript. Additional information regarding this is included in the corpus metadata.

- 2.3. ASR assessment To assess Speech-MASSIVE in multilingual ASR, we used Whisper, since it is one of the recent state-ofthe-art multilingual speech recognition models. We selected Whisper-large-v3,3 utilizing it without additional ﬁne-tuning for our ASR evaluation. Table 1 shows WER and character error rate (CER) across languages and data splits. We compared ASR error rates to those obtained on the FLEURS dataset [5].4 FLEURS generally yields lower WERs/CERs compared to Speech-MASSIVE. The same observation was made for Italian in [11], which followed a recording methodology similar to ours. This suggests that the higher WERs are likely due to the inherent diﬃculty of MASSIVE utterances comparedto those in FLEURS.Furthermore, there are still discrepancies between our Whisper model’s hypotheses and the references in the MASSIVE dataset (e.g., numbers reported in letters in MASSIVE references), which we did not address as optimizing ASR WER was not our main goal. Finally, we calculated the correlation coeﬃcient between WERs (CER for Korean) on Speech-MASSIVE and FLEURS, resulting in a value of 0.96. This shows that Whisper consistently performs across both datasets, despite SpeechMASSIVE being more challenging than FLEURS for ASR.

## 3. SLU Baselines and Beyond

In this section, we establish several SLU baselines, evaluating them with diﬀerent training conditions and metrics described in §3.1. Firstly (§3.2), we build

[Figure 295]

3https://hf.co/openai/whisper-large-v3 4Accessible for our 12 languages except Arabic at

https://github.com/openai/whisper/discussions/1762

a NLU model, serving as an upper bound free from ASR errors. Secondly, we build a cascaded SLU system(§3.3), in which an ASR component transcribes input audio and the NLU model utilizes ASR output for inference. Thirdly, to complete the inventory of SLU baselines, we introduce an end-to-end (E2E) model (§3.4). We conclude by showcasing the versatility of Speech-MASSIVE beyond SLU, computing additional baselines for tasks such as speech translation and language identiﬁcation (§3.5).

- 3.1. NLU/SLU training conditions and metrics To simulate diﬀerent training resource scenarios, we report performance in three diﬀerent settings: (a) Zeroshot: we train the model only with one language data from the train split (11,514 utterances) and evaluate in all other diﬀerent languages; (b) Few-shot: we employ subsets (115 examples) for each of the 12 nonEnglish languages, aligning with our train-115 split. 5 Additionally, we integrate the full zero-shot training split to enrich the multilingual training dataset, totaling 12.8k samples for training; (c) Full ﬁne-tune (NLU only): 11,514 training examples of all 12 languages are pooled (138k samples for training). We assess intent prediction in a given text or speech with intent accuracy6.
- 3.2. NLU model Our NLU system uses the mT5 encoder-decoder architecture [22], selected for its superior performance as demonstrated in [8], where the mT5 text-to-text model outperformed both the mT5 encoder-only model and the XLM-R model [4]. We use a pre-trained mT5-base model,7 and ﬁne-tune both the encoder and decoder in a sequence-to-sequence manner. We supply source and target texts as described in [8] and shown in Figure 2. For instance, the French sentence (Fr) “où puis-je aller ce soir” is annotated in slots (Fr-Slots) as ‘Other Other Other timeofday timeofday” and intent (Intent) as “recommendation_events” in MASSIVE. We adapt those annotations to create source and target texts to be used in training: for the source text (Fr-Src in [NLU]), we prepend “Annotate:” to the French sentence (Fr); for the target text (Fr-Tgt in [NLU]), we concatenate slots (Fr-Slots) and intent (Intent).

[Figure 296]

- 5train-115 covers all 18 domains, 60 intents, and 55 slots (including empty slots).
- 6Due to space limitations, we report only intent accuracy scores. However, additional SLU metrics (e.g., micro-averaged slot F1, exact match accuracy, slot-type F1, slot-value CER) exhibit a similar trend and are available in the GitHub repository. We report the average result (and standard deviation) of three runs with diﬀerent seeds. All experiments were executed on 1 A100 80GB GPU.
- 7https://huggingface.co/google/mt5-base

90

87.81

87.8

87.7

87.53

87.43

87.15

86.97

86.77

86.59

86.42

86.13

86.57

85

85.87

85.82

85.46

85.24

85.17

82.47

84.73

83.4

83.13

80.4

80.06

79.91

79.79

79.49

81.82

78.87

80

78.55

77.93

77.92

77.82

77.15

76.61 76.6 3

78.93

78.82

75.81

78.17

78.08

78.05

77.56

77.21

76.96

73.99

75

76.11

76.29

72.99

75.96

75.7

75.61

74.57

71.21

70.5

73.12

70

68.65

71.13

68.14

70.32

68.7

68.11

65.23

64.55

65

62.34

65.32

64.77

IntentAccuracy(%)

63.43

60

58.65

60.93

60.19

55

54.56

5 0

49.27

nl-NL fr-FR de-DE pt-PT ru-RU es-ES pl-PL tr-TR hu-HU vi-VN ko-KR ar-SA

Languages

NLU zero -shot Cascaded SLU zero -shot NLU few-shot Cascaded SLU few-shot NLU fine-tune Cascaded SLU fine-tune

Figure 1: NLU vs Cascaded SLU (Intent Accuracy) on our Speech-MASSIVE Dataset.

ASR [<|startofstranscript|>, <|language_id|>, <|transcribe|>, <|notimestamps|>] E2E SLU [<|startofstranscript|>, <|language_id|>, <|transcribe|>, <|startoflm|>, <|notimestamps|>] LID [<|startofstranscript|>] ST [<|startofstranscript|>, <|language_id|>, <|translate|>, <|notimestamps|>]

[Original text in MASSIVE] En) where can i go tonight En-Annot) where can i go [timeofday : tonight] En-Slots) Other Other Other Other timeofday Fr) où puis-je aller ce soir Fr-Annot) où puis-je aller [timeofday:ce soir] Fr-Slots) Other Other Other timeofday timeofday Intent) recommendation_events [NLU] Fr-Src) Annotate: où puis-je aller ce soir Fr-Tgt) Other Other Other timeofday timeofday recommendation_events [Cascaded SLU] Fr-ASR) où puis je aller ce soir Fr-Src) Annotate: où puis je aller ce soir Fr-Tgt) Other Other Other timeofday timeofday recommendation_events [E2E SLU] Fr-Tgt) où puis-je aller ce soir | Other Other Other timeofday timeofday | recommendation_events

Figure 3: Various task control tokens fed to Whisper’s decoder.

The SLU intent accuracy scores in Figure 1 reveal that processing automatically transcribed utterances introduces performance drops of varying magnitude across the diﬀerent languages and training modes. This is especially notable for languages with lower ASR quality (i.e., higher WER), such as Ar, Hu, Ko, Tr, and Vn. This supports our expectations about the diﬃculty for the downstream textual NLU component of the SLU cascade to handle unrecoverable transcription errors. As a matter of fact, in zero-shot mode, the distance with the text-only upper-bound NLU system is considerably smaller for languages featuring higher ASR quality. Similar to what we observed for NLU (§3.2), cascaded SLU performance in few-shot mode improves thanks to the additional multilingual data. The gains are particularly signiﬁcant for languages with lesser representation in mT5 model, such as Tr, Vn, Ko, and Ar. Lastly in full ﬁne-tune mode, leveraging a larger multilingual training dataset leads to substantial performance enhancements. While the gains are variable, we observe that: i) for some languages (i.e. De, Ru, and Es), the gap with the highest results of the textual NLU upper bound shrinks to less than two points, while ii) for all languages, the scores are signiﬁcantly higher than those achieved by the textual NLU models dealing with clean input not only in zero-shot, but also in few-shot mode.

Figure 2: Input/Output formatting across NLU/SLU tasks. En: original English text. Fr: French translation in MASSIVE. Annot, Slots and Intent: slot and intent annotation of MASSIVE.

Figure 1 displays the intent accuracy results of our NLU system across all languages and modes (zero-shot, fewshot, full ﬁne-tune), along with those of the cascaded SLU models discussed in §3.3. Unsurprisingly, NLU performance increases when moving from zero-shot to full ﬁne-tune regimes. Also, as expected, higher scores are observed for languages (Nl, Fr, De, Pt, Ru, Es and Pl) that are better represented in the mC4 multilingual dataset used to train mT5 model [22]. Finally, the highest results align with those reported in the MASSIVE paper [8], serving as a suitable reference upper bound for comparisons with the SLU models discussed in the following section.

- 3.3. Cascaded SLU model We develop a cascaded SLU system in which an ASR model based on Whisper-large-v3 transcribes the speech, and the same NLU models of §3.2 (zero-shot, few-shot, full ﬁne-tune) predict slots and intent from the transcribed texts.

Table 2: Intent accuracy of cascaded and E2E SLU. Both E2E SLU zero-shot and few-shot models are trained either with initial English train set of [3] (En) or with French train set of Speech-MASSIVE (Fr). We exclude French (*) from the average as fr-FR scores are no longer zero/few-shot when French is used as the training language.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Casc. (En) zero-shot

E2E (En) zero-shot

E2E (Fr) zero-shot

Casc. (En) few-shot

E2E (En) few-shot

E2E (Fr) few-shot

lang

[Figure 309]

ar 49.27 ± 0.90 33.04 ± 4.74 40.00 ± 2.44 54.56 ± 0.73 57.71 ± 1.46 61.22 ± 1.74 de 76.29 ± 0.14 70.68 ± 1.37 73.91 ± 0.73 78.08 ± 0.50 78.64 ± 0.65 78.45 ± 0.64 es 75.70 ± 0.19 73.12 ± 0.75 78.62 ± 0.41 78.05 ± 0.33 79.79 ± 0.66 80.59 ± 0.31 fr 75.61 ± 0.48 68.43 ± 2.30 85.87 ± 0.26* 77.56 ± 0.13 77.11 ± 0.77 85.93 ± 0.35*

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

hu 63.43 ± 0.92 36.62 ± 1.49 42.28 ± 2.20 68.70 ± 0.80 60.75 ± 2.40 63.93 ± 0.19 ko 60.93 ± 0.84 57.96 ± 2.26 66.09 ± 1.86 68.11 ± 0.04 72.82 ± 0.23 74.09 ± 0.73 nl 78.82 ± 0.45 65.17 ± 0.57 67.24 ± 1.44 78.93 ± 0.34 77.49 ± 0.77 77.37 ± 0.47 pl 74.57 ± 0.37 64.82 ± 1.51 64.38 ± 1.29 76.11 ± 0.39 74.85 ± 0.58 76.88 ± 1.37 pt 73.12 ± 0.49 62.91 ± 1.97 72.60 ± 1.01 77.21 ± 0.65 78.15 ± 1.16 80.02 ± 0.29 ru 75.96 ± 0.19 69.06 ± 1.71 74.75 ± 0.28 76.96 ± 0.08 79.22 ± 0.67 79.51 ± 0.26 tr 65.32 ± 0.61 47.60 ± 3.08 55.08 ± 1.09 70.32 ± 0.48 69.44 ± 1.62 71.14 ± 1.15 vi 60.19 ± 0.39 35.44 ± 1.48 49.67 ± 2.30 64.77 ± 0.98 63.36 ± 1.69 68.71 ± 0.33

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

###### avg. 69.10 ± 0.19 57.07 ± 1.82 62.24 ± 0.92 72.45 ± 0.32 72.45 ± 0.53 73.81 ± 0.58

[Figure 402]

- 3.4. E2E SLU model To complete the inventory of SLU baselines for comparison, we introduce an end-to-end (E2E) SLU model: a direct solution that bypasses intermediate text representations (ASR transcripts). We utilize Whisper, following the approach proposed in [20], which showed superior performance compared to cascaded systems and other speech encoders like wav2vec2.0 [2] and HuBERT [9]. Model training follows a sequence-tosequence approach, with predictions extended to include transcript, slots, and intent. This allows us to leverage both speech and text information in the model’s predictions. We introduce an additional separator “|” between the tasks, allowing Whisper’s tokenizer to tokenize the target text as is, without the need to add slots or intents to the original vocabulary. Two speciﬁc tokens, “|” and “_”, are removed from Whisper’s suppressed token list, as they are required for predicting SLU outputs as task separators and in certain intent values. In zero-shot mode, we ﬁne-tune Whisper-large-v3 with either a) the English train set of [3], or b) the French train set of SpeechMASSIVE. These two conditions (En vs Fr) allow us to investigate the impact of the training language on zeroshot E2E SLU across all other languages. Additionally, in few-shot mode, we ﬁne-tune Whisper-large-v3 with the English or French train sets, along with train-115 splits from other languages. We do not provide a full ﬁne-tune E2E SLU mode since only two languages in Speech-MASSIVE are supported by full train splits. Table 2 compares cascaded and E2E SLU performance

in both zero-shot and few-shot modes. It is worth noting that the comparison between the two approaches is fair only when using the English train set (En), since they utilize the same training utterances albeit in different modalities (written form for cascade and spoken form for E2E). In this condition (En), for zero-shot mode, cascaded SLU outperforms E2E SLU for all languages. In few-shot mode, we note a diﬀerent trend, with cascaded and E2E models exhibiting similar average performance. Employing the French training set from Speech-MASSIVE (Fr), E2E SLU surpasses models trained on the English dataset from [3] (En) in both zero-shot and few-shot modes. In zero-shot mode, we observe improvements of more than 5 points for 9 out of 11 languages. In few-shot mode, although the inﬂuence of the training language (En vs Fr) diminishes due to multilingual training, using French as the majority language still yields better performance than using English. These results highlight the signiﬁcant inﬂuence of the ‘training language’ on the performance of E2E SLU models in zero/few-shot settings. Speech-MASSIVE provides a unique opportunity to explore this intriguing observation further. Finally, examining French (Fr) results representing the full ﬁne-tune mode for this language, E2E SLU achieves intent accuracy of 85.87%, compared to 84.73% for cascaded SLU and 87.43% for NLU given in Fig.1.

### 3.5. Other baselines

We conclude our experiments using Whisper-large-v3 without any ﬁnetuning to compute other baselines

##### Table 3: LID accuracy and ST BLEU results with Whisper-large-v3 on Speech-MASSIVE.

[Figure 405]

lang ar de es fr hu ko nl pl pt ru tr vi split dev test dev test dev test dev test dev test dev test dev test dev test dev test dev test dev test dev test

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

LID accuracy 90.9 89.5 98.9 98.4 99.0 98.6 98.7 98.9 94.6 95.8 99.1 98.7 94.8 94.9 95.3 94.6 95.9 96.0 99.1 98.8 96.1 96.0 90.7 93.2 ST BLEU 17.2 16.6 36.7 38.2 38.5 38.2 38.7 40.1 19.4 20.6 19.7 19.5 40.0 38.9 29.9 28.8 32.4 32.3 28.4 28.2 26.7 26.0 18.9 20.2

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

and demonstrate the versatility of Speech-MASSIVE. We perform Language Identiﬁcation (LID) and Speech Translation (ST) across x→en language directions. Different types of tokens are fed to Whisper’s decoder depending on the tasks as shown in Figure 3. Table

- 3 reports Whisper-large-v3 model’s LID accuracy and ST BLEU [15] on Speech-MASSIVE. LID is calculated over all the samples in dev and test splits. For ST, instead, BLEU is computed on subsets of dev and test splits identiﬁed using meta information from MASSIVE to exclude samples with localized translation. This ﬁltering is necessary to ensure an accurate assessment of translation quality, as localized references may introduce discrepancies in word choice (see §1). Besides indicating the versatility of Speech-MASSIVE for evaluation purposes, our additional baselines on speechrelated tasks oﬀer valuable reference scores for crosstask comparisons and for exploring collaborative solutions to leverage potential mutual beneﬁts.
- 4. Conclusion We introduced Speech-MASSIVE, a multilingual SLU dataset spanning 12 languages for intent prediction and slot-ﬁlling tasks. Alongside dataset creation, we established baselines for SLU across various resource and architecture conﬁgurations. Additionally, we showcased Speech-MASSIVE’s versatility beyond SLU, extending to tasks such as ASR, LID, and ST. With its diverse array of native speakers and recording environments, Speech-MASSIVE holds promise as a benchmark for multilingual, multimodal, and multi-task speech research. Future research opportunities include exploring further the inﬂuence of training languages on zero/few-shotSLU performance, thoroughly comparing cascade and E2E SLU solutions, assess the eﬀect of including multi-task and multilingual corpora in the training of speech foundation models, and pushing the boundaries of E2E multi-task speech systems beyond our baselines.

## Acknowledgements

The speech collection was funded by EU Horizon Europe (HE) Research and Innovation programme grant No 101070631. We also acknowledge the support of the PNRR project FAIR - Future AI Research (PE00000013), under the NRRP MUR program funded by the NextGenerationEU.

## References

- [1] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4218–4222, Marseille, France, 2020. 1, 2
- [2] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for selfsupervised learning of speech representations. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 5
- [3] Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser. SLURP: A spoken language understanding resource package. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7252–7262, Online,

2020. Association for Computational Linguistics. 1, 5

- [4] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online, 2020. Association for Computational Linguistics. 3
- [5] Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805, 2023. 1, 3
- [6] Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, Maël Primet, and Joseph Dureau. Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces. CoRR, abs/1805.10190, 2018. 1
- [7] Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. MuST-C: a Multilingual Speech Translation Corpus. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2012–2017, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. 1
- [8] Jack FitzGerald, Christopher Hench, Charith Peris,

- Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4277–4302, Toronto, Canada, 2023. Association for Computational Linguistics. 1, 3, 4, 8
- [9] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021. 5
- [10] Javier Iranzo-Sánchez, Joan Albert Silvestre-Cerdà, Javier Jorge, Nahuel Roselló, Adrià Giménez, Albert Sanchis, Jorge Civera, and Alfons Juan. Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8229–8233, 2020. 1
- [11] Alkis Koudounas, Moreno La Quatra, Lorenzo Vaiani, Luca Colomba, Giuseppe Attanasio, Eliana Pastor, Luca Cagliero, and Elena Baralis. ITALIC: An Italian Intent Classiﬁcation Dataset. In Proc. INTERSPEECH 2023, pages 2153–2157, 2023. 1, 2, 3
- [12] Fabrice Lefèvre, Djamel Mostefa, Laurent Besacier, Yannick Estève, Matthieu Quignard, Nathalie Camelin, Benoît Favre, Bassam Jabaian, and Lina Maria RojasBarahona. Leveraging study of robustness and portability of spoken language understanding systems across languages and domains: the PORTMEDIA corpora. In Proceedings of the Eighth International Conference on Language Resources and Evaluation, LREC 2012, Istanbul, Turkey, May 23-25, 2012, pages 1436–1442. European Language Resources Association (ELRA), 2012. 1
- [13] Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. MLQA: Evaluating crosslingual extractive question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7315–7330, Online, 2020. Association for Computational Linguistics. 1
- [14] Nikita Moghe, Evgeniia Razumovskaia, Liane Guillou, Ivan Vulić, Anna Korhonen, and Alexandra Birch. Multi3NLU++: A multilingual, multi-intent, multidomain dataset for natural language understanding in task-oriented dialogue. In Findings of the Association for Computational Linguistics: ACL 2023, pages 3732– 3755, Toronto, Canada, 2023. Association for Computational Linguistics. 1
- [15] Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual

- meeting of the Association for Computational Linguistics, pages 311–318, 2002. 6
- [16] Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. In Interspeech

2020. ISCA, 2020. 1

- [17] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning. JMLR.org, 2023. 3
- [18] Elizabeth Salesky, Matthew Wiesner, Jacob Bremerman, Roldano Cattoni, Matteo Negri, Marco Turchi, Douglas W. Oard, and Matt Post. The Multilingual TEDx Corpus for Speech Recognition and Translation. In Proc. Interspeech 2021, pages 3655–3659, 2021. 1
- [19] Changhan Wang, Anne Wu, and Juan Miguel Pino. Covost 2: A massively multilingual speech-to-text translation corpus. CoRR, abs/2007.10310, 2020. 1
- [20] Minghan Wang, Yinglu Li, Jiaxin Guo, Xiaosong Qiao, Zongyao Li, Hengchao Shang, Daimeng Wei, Shimin Tao, Min Zhang, and Hao Yang. Whislu: End-to-end spoken language understanding with whisper. In Proc. Interspeech, pages 770–774, 2023. 5
- [21] Weijia Xu, Batool Haider, and Saab Mansour. End-toend slot alignment and recognition for cross-lingual NLU. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5052–5063, Online, 2020. Association for Computational Linguistics. 1
- [22] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raﬀel. mT5: A massively multilingual pretrained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online, 2021. Association for Computational Linguistics. 3, 4
- [23] Marcely Zanon Boito, William Havard, Mahault Garnerin, Éric Le Ferrand, and Laurent Besacier. MaSS: A large and clean multilingual corpus of sentence-aligned spoken utterances extracted from the Bible. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 6486–6493, Marseille, France, 2020. European Language Resources Association. 1

## A. Appendix

The hyper parameter used to train the end-to-end Spoken Language Understanding model is presented in Table 4.

We report all the evaluation results Exact match accuracy in Table 5, Intent accuracy in Table 6, Slot microF1 in Table 7 and both Slot type F1 and Slot value CER in Table 8.

Evaluation code used to calculate the metrics of Exact match accuracy, Intent accuracy and Slot micro-F1 is from MASSIVE[8] implementation. 8 For Slot type F1 and slot value CER evaluation, we use S3PRL toolkit.9

[Figure 461]

- 8https://github.com/alexa/massive/blob/main/src/massive/utils/training_utils.py
- 9https://github.com/s3prl/s3prl/blob/aa3ba844bfe2b5402b7f345cbebd72b33ef6aeff/s3prl/metric/slot_filling.py

Table 4: Hyper parameter settings for end-to-end SLU zero-shot and few-shot models.

- adam beta1 0.8

[Figure 464]

[Figure 465]

- adam beta2 0.999

[Figure 466]

[Figure 467]

adam epsilon 0 add separator ‘|’

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

freeze feature encoder FALSE

[Figure 472]

[Figure 473]

gradient accumulation steps 2 gradient checkpointing FALSE label smoothing factor 0

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

learning rate 0.00001 lr scheduler type linear

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

20000 (zero-shot) 25000 (few-shot)

[Figure 484]

max steps

[Figure 485]

per device eval batch size 8 per device train batch size 8

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

target format content transcript slots intent

[Figure 490]

[Figure 491]

[Figure 492]

task transcribe tokens to remove from suppress ‘G|’,˙ ‘_’

[Figure 493]

[Figure 494]

[Figure 495]

2000 (zero-shot) 2500 (few-shot)

[Figure 496]

warmup steps

Table 5: Exact Match Accuracy for all the settings.

[Figure 497]

[Figure 498]

[Figure 499]

zero-shot few-shot ﬁne-tune

[Figure 500]

lang NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU ar-SA 28.39 ± 1.16 20.93 ± 1.04 17.81 ± 2.47 22.22 ± 1.72 39.48 ± 1.32 27.55 ± 0.78 37.47 ± 0.30 39.76 ± 1.32 64.64 ± 0.58 45.06 ± 0.29 de-DE 51.18 ± 1.02 44.71 ± 1.01 46.71 ± 1.43 49.61 ± 0.87 56.06 ± 1.13 48.28 ± 0.54 56.44 ± 0.62 57.64 ± 0.19 69.70 ± 0.53 60.35 ± 0.34 es-ES 47.71 ± 0.44 44.40 ± 0.60 50.00 ± 0.83 54.55 ± 0.47 52.64 ± 0.39 49.23 ± 0.42 57.42 ± 0.45 59.09 ± 0.36 67.09 ± 0.12 61.84 ± 0.09 fr-FR 45.09 ± 1.11 38.3 ± 0.79 43.67 ± 2.09 65.38 ± 0.43 52.59 ± 1.34 38.88 ± 0.14 53.04 ± 0.55 65.61 ± 0.25 67.44 ± 0.17 46.08 ± 0.29

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

hu-HU 38.11 ± 1.25 31.46 ± 1.01 19.1 ± 0.83 22.10 ± 1.46 47.00 ± 0.74 37.40 ± 0.48 38.70 ± 2.60 42.57 ± 0.50 68.35 ± 0.46 53.12 ± 0.12 ko-KR 31.97 ± 0.81 30.08 ± 0.81 33.55 ± 2.06 39.70 ± 1.41 43.97 ± 0.20 37.73 ± 0.12 47.81 ± 0.76 49.18 ± 0.95 69.45 ± 0.45 55.34 ± 0.15 nl-NL 52.05 ± 1.10 46.21 ± 1.24 40.51 ± 0.34 42.17 ± 0.79 56.2 ± 0.19 47.89 ± 0.05 53.78 ± 1.16 54.95 ± 0.27 69.72 ± 0.21 59.18 ± 0.17 pl-PL 45.30 ± 0.61 41.32 ± 0.64 39.59 ± 1.68 40.22 ± 0.91 49.22 ± 0.69 44.27 ± 0.63 51.57 ± 0.69 54.75 ± 0.74 65.98 ± 0.40 58.61 ± 0.44 pt-PT 46.35 ± 0.58 39.84 ± 0.74 38.49 ± 1.00 46.55 ± 1.04 52.72 ± 0.87 44.48 ± 0.59 53.64 ± 0.52 56.01 ± 0.26 68.84 ± 0.05 56.92 ± 0.05 ru-RU 48.9 ± 1.06 46.32 ± 1.35 45 ± 1.68 49.86 ± 0.24 53.25 ± 0.34 49.78 ± 0.40 57.79 ± 0.57 58.09 ± 0.07 70.17 ± 0.14 64.52 ± 0.15 tr-TR 37.88 ± 0.25 32.33 ± 0.2 24.56 ± 2.25 30.14 ± 0.74 45.79 ± 0.88 37.56 ± 0.50 44.98 ± 1.71 46.62 ± 0.88 68.91 ± 0.32 55.51 ± 0.47 vi-VN 30.35 ± 0.93 25.96 ± 0.69 13.85 ± 0.39 21.15 ± 1.77 38.25 ± 1.47 31.01 ± 1.26 35.35 ± 1.12 41.35 ± 0.31 65.78 ± 0.24 51.46 ± 0.45

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

###### avg 41.94 ± 0.67 36.82 ± 0.65 34.4 ± 1.28 38.02 ± 0.84 48.93 ± 0.71 41.17 ± 0.41 49 ± 0.47 50.91 ± 0.4 68.01 ± 0.18 55.66 ± 0.13

Table 6: Intent Accuracy for all the settings.

[Figure 547]

[Figure 548]

[Figure 549]

zero-shot few-shot ﬁne-tune

lang NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU ar-SA 58.65 ± 0.40 49.27 ± 0.90 33.04 ± 4.74 40.00 ± 2.44 65.23 ± 1.23 54.56 ± 0.73 57.71 ± 1.46 61.22 ± 1.74 82.47 ± 0.20 71.13 ± 0.28 de-DE 77.82 ± 0.14 76.29 ± 0.14 70.68 ± 1.37 73.91 ± 0.73 79.49 ± 0.64 78.08 ± 0.50 78.64 ± 0.65 78.45 ± 0.64 86.77 ± 0.21 85.46 ± 0.19 es-ES 76.61 ± 0.27 75.70 ± 0.19 73.12 ± 0.75 78.62 ± 0.41 78.87 ± 0.07 78.05 ± 0.33 79.79 ± 0.66 80.59 ± 0.31 87.15 ± 0.38 85.87 ± 0.27 fr-FR 78.55 ± 0.44 75.61 ± 0.48 68.43 ± 2.30 85.87 ± 0.26 79.91 ± 0.51 77.56 ± 0.13 77.11 ± 0.77 85.93 ± 0.35 87.43 ± 0.41 84.73 ± 0.35

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

hu-HU 68.14 ± 0.95 63.43 ± 0.92 36.62 ± 1.49 42.28 ± 2.20 72.99 ± 0.61 68.70 ± 0.80 60.75 ± 2.40 63.93 ± 0.19 86.42 ± 0.27 81.82 ± 0.21 ko-KR 62.34 ± 0.93 60.93 ± 0.84 57.96 ± 2.26 66.09 ± 1.86 70.50 ± 0.10 68.11 ± 0.04 72.82 ± 0.23 74.09 ± 0.73 86.59 ± 0.29 83.40 ± 0.30 nl-NL 80.06 ± 0.42 78.82 ± 0.45 65.17 ± 0.57 67.24 ± 1.44 80.4 ± 0.22 78.93 ± 0.34 77.49 ± 0.77 77.37 ± 0.47 87.81 ± 0.30 85.82 ± 0.40 pl-PL 76.63 ± 0.42 74.57 ± 0.37 64.82 ± 1.51 64.38 ± 1.29 77.93 ± 0.28 76.11 ± 0.39 74.85 ± 0.58 76.88 ± 1.37 87.53 ± 0.04 85.17 ± 0.16 pt-PT 75.81 ± 0.05 73.12 ± 0.49 62.91 ± 1.97 72.60 ± 1.01 79.79 ± 0.76 77.21 ± 0.65 78.15 ± 1.16 80.02 ± 0.29 87.70 ± 0.26 85.24 ± 0.24 ru-RU 77.15 ± 0.33 75.96 ± 0.19 69.06 ± 1.71 74.75 ± 0.28 77.92 ± 0.30 76.96 ± 0.08 79.22 ± 0.67 79.51 ± 0.26 87.80 ± 0.22 86.57 ± 0.27 tr-TR 68.65 ± 0.30 65.32 ± 0.61 47.60 ± 3.08 55.08 ± 1.09 73.99 ± 0.37 70.32 ± 0.48 69.44 ± 1.62 71.14 ± 1.15 86.97 ± 0.26 83.13 ± 0.24 vi-VN 64.55 ± 0.80 60.19 ± 0.39 35.44 ± 1.48 49.67 ± 2.30 71.21 ± 1.06 64.77 ± 0.98 63.36 ± 1.69 68.71 ± 0.33 86.13 ± 0.16 78.17 ± 0.59

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

avg 72.08 ± 0.21 69.10 ± 0.19 57.07 ± 1.82 62.24 ± 0.92 75.69 ± 0.42 72.45 ± 0.32 72.45 ± 0.53 73.81 ± 0.58 86.73 ± 0.13 83.04 ± 0.15

Table 7: Micro-avg slot F1 for all the settings.

[Figure 595]

[Figure 596]

[Figure 597]

zero-shot few-shot ﬁne-tune

lang NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU E2E SLU (En) E2E SLU (Fr) NLU Cascaded SLU ar-SA 36.49 ± 1.89 26.89 ± 1.77 10.23 ± 2.64 16.41 ± 1.33 54.21 ± 1.14 40.20 ± 1.07 36.27 ± 0.69 38.58 ± 0.67 76.50 ± 0.52 54.66 ± 0.27 de-DE 62.49 ± 0.56 55.04 ± 0.21 50.56 ± 1.95 54.12 ± 1.09 69.32 ± 0.76 58.92 ± 0.37 59.56 ± 0.85 62.12 ± 0.54 80.00 ± 0.54 67.59 ± 0.22 es-ES 56.35 ± 1.06 52.09 ± 1.21 47.51 ± 0.55 50.38 ± 0.80 63.35 ± 0.86 58.79 ± 0.82 55.04 ± 2.03 58.39 ± 0.31 76.00 ± 0.10 69.12 ± 0.20 fr-FR 49.57 ± 0.82 37.91 ± 0.45 37.17 ± 2.2 62.30 ± 0.47 62.30 ± 1.81 38.71 ± 0.68 49.93 ± 0.29 62.31 ± 0.22 76.38 ± 0.33 43.19 ± 0.53

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

hu-HU 44.47 ± 1.03 36.25 ± 1.09 18.54 ± 1.51 22.61 ± 0.67 60.68 ± 0.83 47.16 ± 0.68 42.01 ± 0.24 46.12 ± 0.17 78.70 ± 0.19 59.01 ± 0.07 ko-KR 42.83 ± 1.52 39.29 ± 1.01 23.38 ± 3.55 27.37 ± 1.78 57.20 ± 0.64 47.12 ± 0.09 41.43 ± 0.89 43.30 ± 1.11 80.06 ± 0.60 60.97 ± 0.29 nl-NL 60.92 ± 0.87 52.50 ± 1.39 44.57 ± 0.58 44.88 ± 0.45 68.68 ± 0.34 56.29 ± 0.09 55.90 ± 0.14 57.68 ± 0.30 78.33 ± 0.29 63.33 ± 0.30 pl-PL 52.69 ± 0.96 47.53 ± 1.04 35.93 ± 2.55 40.74 ± 1.07 61.35 ± 0.61 54.69 ± 0.36 51.53 ± 1.29 55.20 ± 0.24 74.14 ± 0.49 65.29 ± 0.35 pt-PT 54.45 ± 0.87 44.50 ± 0.98 32.72 ± 1.31 40.12 ± 2.22 62.49 ± 0.94 50.34 ± 0.65 48.48 ± 1.46 51.13 ± 0.67 77.63 ± 0.19 60.63 ± 0.34 ru-RU 59.73 ± 1.15 55.98 ± 1.00 41.10 ± 3.42 48.70 ± 1.05 64.91 ± 0.71 59.62 ± 0.59 57.40 ± 0.73 59.45 ± 0.47 79.08 ± 0.32 71.60 ± 0.28 tr-TR 46.74 ± 0.96 39.60 ± 0.70 24.44 ± 2.76 29.51 ± 1.09 58.71 ± 0.86 47.94 ± 0.18 45.95 ± 0.88 48.59 ± 0.99 78.58 ± 0.55 60.85 ± 0.51 vi-VN 36.97 ± 1.13 30.24 ± 0.94 12.01 ± 2.04 21.98 ± 3.06 46.02 ± 2.87 37.58 ± 2.45 44.40 ± 1.50 50.26 ± 0.56 75.00 ± 0.52 58.26 ± 0.45

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

avg 50.31 ± 0.86 43.15 ± 0.81 31.51 ± 1.32 36.07 ± 1.04 60.77 ± 0.93 49.78 ± 0.43 48.99 ± 1.50 51.89 ± 1.50 77.53 ± 0.28 61.21 ± 0.15

Table 8: Slot type F1 score and slot value CER for all the settings.

[Figure 643]

[Figure 644]

Slot type F1 score Slot value CER

zero-shot few-shot zero-shot few-shot

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

lang E2E SLU (En) E2E SLU (Fr) E2E SLU (En) E2E SLU (Fr) E2E SLU (En) E2E SLU (Fr) E2E SLU (En) E2E SLU (Fr) ar-SA 70.61 ± 1.14 70.70 ± 1.18 78.70 ± 0.50 80.14 ± 0.78 59.39 ± 2.90 48.75 ± 2.52 33.71 ± 0.21 30.81 ± 0.23 de-DE 86.48 ± 0.38 86.65 ± 0.49 89.45 ± 0.67 89.71 ± 0.19 23.23 ± 1.10 20.74 ± 0.84 16.72 ± 0.34 14.59 ± 0.13 es-ES 87.73 ± 0.34 88.81 ± 0.33 89.70 ± 0.24 90.25 ± 0.07 20.05 ± 0.47 15.61 ± 0.39 14.88 ± 0.63 13.42 ± 0.10 fr-FR 86.73 ± 0.82 92.88 ± 0.13 89.62 ± 0.21 92.97 ± 0.04 25.07 ± 1.25 10.99 ± 0.21 17.40 ± 0.24 10.84 ± 0.03

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

hu-HU 76.20 ± 0.82 77.05 ± 0.45 82.59 ± 0.67 83.75 ± 0.32 43.71 ± 1.25 42.65 ± 0.66 29.12 ± 1.01 25.56 ± 0.05 ko-KR 77.36 ± 2.24 78.62 ± 3.30 84.80 ± 0.37 86.24 ± 0.36 58.56 ± 3.44 50.55 ± 2.92 36.70 ± 1.86 30.89 ± 0.21 nl-NL 86.41 ± 0.28 86.06 ± 0.27 89.11 ± 0.38 89.43 ± 0.20 27.78 ± 0.69 24.65 ± 0.71 17.76 ± 0.29 15.77 ± 0.06 pl-PL 82.83 ± 0.83 83.48 ± 0.34 87.65 ± 0.15 88.30 ± 0.25 28.77 ± 0.93 28.70 ± 0.47 20.58 ± 0.58 17.90 ± 0.25 pt-PT 83.27 ± 0.57 86.23 ± 0.52 89.18 ± 0.30 89.94 ± 0.14 31.32 ± 1.11 25.04 ± 1.63 17.62 ± 0.36 15.78 ± 0.13 ru-RU 84.49 ± 1.00 85.47 ± 0.58 88.76 ± 0.31 89.62 ± 0.20 36.53 ± 2.05 25.28 ± 1.01 17.91 ± 0.66 15.72 ± 0.06 tr-TR 77.86 ± 1.75 78.55 ± 1.93 84.82 ± 0.89 85.78 ± 0.47 37.96 ± 1.59 35.56 ± 1.56 24.43 ± 0.41 21.57 ± 0.44 vi-VN 75.41 ± 0.30 78.67 ± 1.31 83.60 ± 0.91 85.56 ± 0.40 52.96 ± 0.92 41.99 ± 1.96 30.10 ± 0.57 26.57 ± 0.26

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

###### avg 81.28 ± 0.78 81.84 ± 0.90 86.50 ± 0.18 87.16 ± 1.50 37.11 ± 0.98 32.68 ± 1.09 23.08 ± 0.21 20.78 ± 1.50

