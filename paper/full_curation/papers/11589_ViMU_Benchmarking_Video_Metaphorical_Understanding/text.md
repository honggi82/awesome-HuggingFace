# arXiv:2605.14607v1[cs.CV]14May2026

## ViMU: Benchmarking Video Metaphorical Understanding

#### Qi Li Xinchao Wang∗

##### National University of Singapore liqi@u.nus.edu xinchao@nus.edu.sg

Project Page GitHub Dataset

### Abstract

Any new medium, once it emerges, is used for more than the transmission of overt content alone. The information it carries typically operates on two levels: one is the content directly presented, while the other is the subtext beneath it—the implicit ideas and intentions the creator seeks to convey through the medium. Likewise, since video technologies became widely adopted, video has served not only as a powerful tool for recording and communicating visual information, but also as a vehicle for emotions, attitudes, and social meanings that are often difficult to articulate explicitly. Thus, the true meaning of many videos does not reside solely in what is shown on screen; it is often embedded in context, style of expression, and the viewer’s social experience. Some forms of such video subtext are humorous, while others carry irony, mockery, or criticism. These implicit meanings can also be interpreted very differently across cultural backgrounds and social groups. However, most existing video understanding models still focus primarily on literal visual comprehension, such as recognizing objects, actions, or temporal relations, and lack a systematic ability to understand the metaphorical, ironic, and social meanings embedded in videos. To bridge this gap, we introduce ViMU (Video Metaphorical Understanding), the first benchmark designed to systematically evaluate the subtext understanding capabilities of frontier models in videos. ViMU assesses whether video understanding models can go beyond literal perception to infer implicit meaning, rhetorical devices, social signals, target subjects, and culturally grounded subtext, while grounding their interpretations in multimodal evidence and answering both open-ended and multiple-choice questions. Importantly, all questions are designed to be hint-free, ensuring that no key evidence is disclosed to models before answering. Extensive experiments show that most frontier models, including closed-source ones, achieve below 50% overall performance. We further conduct fine-grained analyses to uncover distinctive model behaviors. Disclaimer: This paper contains potentially offensive and harmful content.

### 1 Introduction

“The most important thing in communication is hearing what isn’t said.”

- Peter Drucker

Recent advances in large language models have enabled the integration of rich real-world information, including videos, into model representations [3, 10, 4, 34, 28, 20, 19, 30, 35]. Consequently, video understanding models have become effective for tasks such as visual grounding and causal reasoning [9, 37, 31]. Yet these forms of understanding remain largely confined to the surface-visible

∗Corresponding Author

Preprint.

[Figure 1]

...

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

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

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

background music: ♬Auf der Heide blüht ein kleines Blümelein... (in English: ♬On the heath, there blooms a little flower) "Erika" - German Army Song

Surface-Level Content: A captioned clip from a stage show shows a young woman in a school-uniform-style outfit kneeling and

moving her arms into an awkward pose while text on screen reads, \"What is she trying to do? \ud83d\udc80\" She then smiles and puts her hands together as others on stage react. A Doge image wearing a hat is edited into the lower-left corner, and a German song lyric plays in the audio.

Underlying Meaning: The meme reframes the performer’s movement as an allusion to Nazi-style or fascist imagery, using the soundtrack and the edited dog sticker to push that interpretation. Rather than treating the clip as a simple awkward or suggestive dance moment, it presents the pose as a provocative political visual joke built on shock and recontextualization.

[Figure 26]

...

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

background music: ♬Interstellar Official Soundtrack | Cornfield Chase – Hans Zimmer

Surface-Level Content: The video begins with a cartoon scene of Isaac Newton reading under a tree while another character peeks out holding up an apple. It then cuts to dark outdoor footage of a person in bulky clothing moving through a field with arms spread wide like wings, resembling a bird in flight.

Underlying Meaning: It uses the Newton-and-apple setup to prime an expectation about gravity or scientific discovery, then abruptly switches to an absurd visual of a person moving like a bird, making the point through an unexpected, goofy take on flight rather than a real explanation.

- Figure 1: Examples illustrating the large gap between observable content and underlying subtext in videos. In the top example, the video appears to show a girl dancing on a reality show, while its implied meaning alludes to Nazi symbolism. In the bottom example, the video appears to show a child catching an apple above Newton and a strange flying scene, while its underlying joke is that the apple missing Newton led to a setback in the development of physics.

content. Put simply, directly observable content explains how an event unfolds, but not what it ultimately means, as such meaning often lies in the underlying social subtext2: the deeper layer that maps an event onto broader social meanings, values, and collective attitudes. Together, the visible content and its subtext constitute the full depth of video understanding [16, 11, 15].

As illustrated in Figure 1, the gap between observable content and underlying subtext can be substantial. In such a case, understanding the video requires more than recognizing objects, actions, or temporal structure, which are typically emphasized in prior works [9, 37, 31, 32, 5, 18]. It demands integrating multimodal evidence, recovering culturally situated references, and inferring the creator’s communicative intent beyond what is explicitly shown. Existing evaluations left far behind for such subtext interpretation in videos. Most existing benchmarks fall short in three ways: (i) targeting implicit reasoning over hidden spatial, physical, or interactional relations rather than socially grounded meanings [27, 7]; (ii) focusing only on narrower phenomena such as non-verbal humor [26]; or (iii) relying on multiple-choice formats whose options may expose plausible subtext hypotheses [13]. These settings do not fully capture genuine hint-free inference over socially grounded video meaning.

To fill this gap, we introduce ViMU, a benchmark specifically designed to evaluate whether models can move beyond observable content to recover the underlying subtext of videos. In particular, ViMU requires models to infer implicit meaning in a hint-free manner, without being told in advance which socio-cultural cues are relevant. To achieve this, we build ViMU through a meticulous curation process involving multiple rounds of annotation and filtering by advanced closed-source models and human experts. This procedure is designed not only to ensure task difficulty and a genuinely hint-free evaluation setting, but also to maintain broad coverage of diverse rhetorical mechanisms and social

2As Roland Barthes notes in his book Mythologies, "myth is a second-order semiological system" [16, 1], in which literal content serves as the basis for a secondary layer of cultural or ideological meaning.

value signals. Finally, we obtain a high-quality dataset of 588 videos with 2,352 questions across four tasks, covering both open-ended and multiple-choice questions.

We extensively investigate 16 popular MLLMs with ViMU, which brings in several critical insights. Firstly, video metaphorical understanding remains a technically challenging problem for the existing MLLMs. Even the most advanced closed-source models achieve below 50% average performance across the four tasks. Secondly, many models systematically over-predict generic or safer categories while under-predicting more implicit or socially coded ones, suggesting a shared tendency to favor more accessible interpretations over deeper subtextual inference. Thirdly, we observe a clear mismatch between general video understanding and metaphorical video understanding: models that excel on conventional video understanding task do not necessarily perform best on our tasks. In addition to the overall conclusion, individual tasks enable fine-grained analysis in each specialized aspects. Therefore, we anticipate the benchmark to assist in improving MLLMs’ video metaphorical understanding capabilities by providing insights into their current strengths and weaknesses.

### 2 Related Work

Reasoning beyond explicit visual evidence. Some recent work has moved beyond explicitevidence-centric VideoQA by requiring models to infer answers from indirect or partially unavailable cues. I-VQA [7] studies settings where explicit visual evidence is missing and answers must be inferred from context, building on related work in visual commonsense and context-based reasoning such as VisualCOMET [23], Video2Commonsense [8], and causal video reasoning methods like MECD [5] and MECD+ [6]. VRR-QA [27] further focuses on implicit relational reasoning across frames when key relations are not directly co-visible. While these benchmarks go beyond literal perception, they still focus on inferential VideoQA or inter-frame relation reasoning rather than broader subtext understanding in open online videos.

Humor understanding, meme interpretation, and social meaning. A closely related line of work studies higher-level interpretation in humorous or socially contextualized media. v-HUB [26] focuses on multimodal video humor understanding, especially in non-verbal short videos, while AVMeme Exam [13] extends evaluation to contextual and cultural understanding of Internet audio-visual memes. Related audio benchmarks, including Dynamic-SUPERB [12], AudioBench [29], MMAU [24], and MMAR [22], mainly evaluate recognition, captioning, dialogue, and semantic or reasoning abilities over audio content. Closely related humor benchmarks such as FunQA [33] study surprising or humorous video comprehension, yet are still narrower than the broader space of socially and culturally grounded subtext. In parallel, meme-oriented benchmarks in static image-text settings, including Hateful Memes [14], What Do You Meme? [25], GOAT-Bench [21], MemeSafetyBench [17], and MemeReaCon [36], probe implicit social meaning, safety, and contextual meme understanding, but cannot capture the temporal, auditory, and evolving multimodal cues that are central to video subtext. In contrast, our focus is on structured, hint-free understanding of video subtext, where models must infer latent meaning from jointly evolving visual, auditory, temporal, and social signals.

Position of ViMU. Our work is most closely related to these recent efforts, but differs in both scope and evaluation philosophy. Compared with general video benchmarks, ViMU targets meaning that is not exhausted by visible objects, actions, or temporal relations. Compared with previous works [7, 27], ViMU is not limited to implicit question answering or hidden inter-frame relations, but instead evaluates whether models can move from observable content to latent subtext, including social signals or culturally grounded interpretations. Compared with humor- or meme-centric benchmarks [26, 13], ViMU focuses broadly on subtext understanding in videos through a structured taxonomy and hint-free questioning, so that models must recover the intended reading without being given the relevant latent evidence or interpretive hypothesis in advance.

### 3 ViMU: Video Metaphorical Understanding Benchmark

ViMU is a multi-task benchmark consisting of 2,352 questions from 588 videos across more than ten rhetoric mechanisms and social value signals, specifically designed for video metaphorical understanding, i.e., u derstanding the subtext meaning beyond the surface-level video content. The benchmark is distinguished by the following features.

Other

Role Reversal

Aggression or Hostility

Literal Only

Sarcasm

Fatalism or Cynicism

Stereotype Invocation

Contrast

Political or Identity Signal

Irony

Other

Parody

2.7%

Anti Mainstream Value

3.7%

2.2%

19.9%

Discrimination or Prejudice

2.9%

5.9%

Norm Violation

None

3.2%

Mockery

3.6%

31.7%

6.2%

Contempt

4.4%

Rhetoric Mechanisms

8.5%

Bait and Switch

Social Value Signals

16.2%

11.3%

Deadpan

Negative Affect

10.5%

Absurdism

12.8%

14.8%

10.7%

20.6%

Innuendo

Humiliation

Sexual Implication

Exaggeration

- Figure 2: Distribution of rhetorical mechanisms (left) and social value signals (right) in the dataset. The benchmark covers a wide range of rhetorical devices used to construct implicit meaning (left) and the social attitudes or value stances conveyed by videos (right), reflecting diverse forms of non-literal and socially contextualized video communication.

0 100 200 300 400 500 600

Frames

Visible Text

Editing Pattern

Transcript

Audio Tone

27.8%

26.3%

23.5%

16.6%

5.8%

Evidence Sources (n=2113)

| |31.0%<br><br>28.6%<br><br>16.0%<br><br>8.6%<br><br>4.0%<br><br>2.3% 2.0% 2.0% 1.5% 1.2% 0.9% 0.7% 0.5% 0.4% 0.1%| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

IndividualOtherNoClearTargetRelationshipRoleSelfRacialorEthnicGroupInstitutionAppearanceorBodyTypeNationalityGroupGenderGroupClassorOccupationSexualityRelatedGroupAgeGroupIdeologyReligiousGroupOther

0

50

100

150

200

250

Target Subjects (n=744)

- Figure 3: Distribution of evidence sources (left) and target subjects (right) in the dataset. The dataset includes multiple types of interpretive evidence, such as visual frames, visible text, editing patterns, transcripts, and audio tone (left), as well as diverse target subjects ranging from individuals and social roles to institutions and identity-related groups (right).

Diversified Semantic Categories. As illustrated in Figure 2, our benchmark spans a diverse set of video categories along two complementary semantic dimensions: rhetoric mechanisms and social value signals. Rhetoric mechanisms refer to the communicative devices through which a video conveys its implicit meaning, such as irony, exaggeration, contrast, deadpan delivery, parody, or bait-and-switch. These mechanisms capture how humor, critique, or commentary is constructed at the level of expression. Social value signals, in contrast, describe the underlying social stance, attitude, or normative implication conveyed by the video. These signals capture what the video expresses about social values, emotions, or group relations, including contempt, norm violation, aggression, anti-mainstream sentiment, and others. In shorts, rhetorical mechanisms define how a video should be interpreted, while social value signals capture the stance it conveys. Together, these two dimensions separate how meaning is conveyed from what social meaning is being expressed. Modeling both enables a more comprehensive evaluation of video metaphor understanding beyond literal perception.

Variety of Evidence Sources and Target Subjects. Evidence sources refer to the observable cues (e.g., video frames, audios, on-screen text) within a video that support the interpretation of its implicit meaning. The distribution of different evidence sources reflects the multimodal nature of video communication. Target subjects describe the entities or groups toward which the video’s rhetorical stance or social commentary is directed (e.g., individuals, social groups, institutions, or broader identity categories). Together, these dimensions reveal the wide range of interpretive cues and social referents present in the dataset, supporting comprehensive evaluation of video understanding models.

Comprehensive Evaluation Tasks. ViMU provides diversified evaluation tasks to probe complementary aspects. Specifically, the benchmark includes an open-ended interpretation task for evaluating overall understanding of the video’s intended meaning, multi-choice tasks for identifying rhetorical mechanisms and social value signals, and an evidence grounding task for selecting the elements that support the interpretation. Together, these tasks enable a comprehensive evaluation of whether models can understand what a video means, how that meaning is constructed, what social stance it conveys, and whether their interpretations are grounded in observable evidence.

[Figure 51]

...

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

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

background music: ♬Real background white noise

Question: Which high-level rhetoric categories are present in this video?

Rhetoric Mechanisms

Select all options that are clearly supported by the video.

Options: (A) Literal / Direct (B) Opposition / Incongruity (C) Attitude / Tone-based Rhetoric

(D) Amplification / Stylization (E) Implicit / Coded Social Framing

Intended Meaning: The joke mocks a European environmental rule that keeps bottle caps attached to plastic bottles to reduce litter. By showing the attached cap getting in the way while drinking, the video satirizes green-policy design as inconvenient and slightly absurd in everyday use.

(a) Rhetoric Mechanisms.

[Figure 80]

...

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

background music: ♬Real background sound with racist metaphors

Question: Which high-level social value categories are present in this video?

Social Value Signals

Select all options that are clearly supported by the video.

Options: (A) Neutral / No Social Signal (B) Emotional Attitude (C) Social Evaluation / Devaluation

(D) Norm and Value Framing (E) Identity / Ideological Signaling

Intended Meaning: The humor is built around background sounds in the inserted clips that can be heard as resembling a racial slur. The reaction shot frames those sounds as the punchline, turning the montage into an edgy meme based on phonetic resemblance, racial innuendo, and provocative mockery rather than on the literal visual content.

(b) Social Value Signals.

[Figure 110]

...

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

[Figure 139]

background music: ♬Truck, duck, plulk, fun to rhyme. Quick, stick, lick, dinner time...

Question: Which elements in the video support the intended meaning?

Evidence Grounding

Select all that apply. Options: (A) Audio Tone (B) Editing Pattern (C) Frames (D) Transcript (E) Visible Text Intended Meaning: The joke is a fake children's rhyme game that turns into forming taboo or explicit words from partial letter clusters. It uses innocent educational framing to set up a reveal of words with sexual or vulgar implications, creating shock humor through the contrast between wholesome presentation and dirty-minded completion.

(c) Evidence Grounding.

- Figure 4: Examples of three-types of multiple-choice tasks in ViMU. From top to bottom: evidence grounding, rhetoric mechanisms, and social value signals. Each question has five candidate choices, and the ground-truth answers are marked in purple. 3.1 Construction of ViMU

We categorize the tasks into three types according to the level of semantic reasoning required: 1) interpretation-level understanding, which requires inferring the overall intended meaning of the video; 2) semantic-structure understanding, which focuses on identifying the rhetorical mechanisms and social value signals underlying the video; and 3) evidence-grounded understanding, which

[Figure 140]

...

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

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

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

background music: ♬The original sound of the man in the lower left corner of the video

Question: What interpretation best explains the video as a whole? Standard Answer: The video’s joke is that recognizable Dragon Ball characters are gradually revealed as deliberately sexualized, compromising fan art, and the reaction highlights the amused shock of seeing familiar characters portrayed that way.

Open-ended Interpretation

- Figure 5: An example of the open-ended interpretation task in ViMU. MLLMs are asked to interpret the video based on the video input, textual prompt, and audio transcript when applicable.

- 0.5

- 0.6

- 0.7

- 0.8

- 0.9

C3H

GM3FP

G41N

G52

MIMO

O4M

SEED

GM327B

GM34B

MN14B MN8B

Q3527B

3.5 3.0 2.5

0.3

0.4

GLM45V

-1.5 -1 -0.5

Closed

Open-weight

Selection conservatism (avg #pred - avg #gold)

Micro-F1

(a) Conservatism vs. performance

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

gemini-3-flash-previewclaude-3-haikuglm-4.5vgpt-4.1-nanogpt-5.4-minigpt-5.2grok-4.1-fastmimo-v2-omnio4-miniseed-2.0-litegemma-3-27b-itgemma-3-4b-itministral-14bqwen3-vl-32b-instructministral-8bqwen3.5-27b

0.0

0.2

0.4

0.6

0.8

1.0

Fractionofquestions

Exact

| |
|---|

Miss-Only

| |
|---|

Extra-Only

| |
|---|

Mixed

(b) Error composition

[Figure 169]

audio editing frames speech text

audio

editing

frames

speech

text

- -0.03 -0.08 -0.06 -0.07 -0.09
- -0.08 -0.73 -0.74 -0.43 -0.73
- -0.06 -0.74 -0.14 -0.23 -0.28
- -0.07 -0.43 -0.23 -0.17 -0.27
- -0.09 -0.73 -0.28 -0.27 -0.24 0.6

[Figure 170]

0.4

0.2

0.0

0.2

0.4

0.6

Deltarate

(c) Relation distortion

Figure 6: Evidence grounding analysis. From left to right, we show the trade-off between evidenceselection conservatism and grounding quality, the composition of different error types across models, and the overall distortion in pairwise evidence relations relative to the gold co-occurrence structure.

examines whether models can identify the multimodal evidence supporting their interpretation. The construction process of ViMU is discussed with respect to these three categories.

To ensure the task is meaningful and fairly reflects model utility, the dataset construction follows several key principles: (i). Ensuring broad coverage of diverse rhetorical mechanisms and social value signals. (ii). Given the nature of the task, careful consideration is given to both the sources of implicit meaning and the targets of reference. Implicit cues may arise from visual frames, on-screen text, editing pattern, audio content, or vocal tone. Targets may refer to individuals, other people in the video, or external groups or events not explicitly shown. (iii). For open-ended questions, no explicit answer cues are allowed, as such hints would significantly reduce task difficulty (e.g., directly asking which symbol is being mimicked by the girl through her body movements in Figure 1 would undermine the task). Following these principles, we curate over 500 videos from platforms like YouTube, Bilibili, and TikTok, covering more than 10 types of rhetorical mechanisms and social value signals (Figure 2, detailed explanations of each type are provided in Appendix E and F). In addition, as shown in Figure 3, the dataset exhibits strong diversity in evidence sources and target subjects, spanning three modalities (text, vision, audio), five types of evidence sources, and over

- 10 target categories. This multi-level diversity enables comprehensive evaluation and analysis of model performance. Annotation of these categories and enforcement of hint-free open-ended tasks are achieved through iterative validation by frontier models and human experts. Details are given in Appendix A. Questions regarding different aspects are discussed below.

#### 3.1.1 Interpretation-Level Understanding

Open-ended Interpretation (OI). This task evaluates whether models can infer the overall meaning conveyed by a video. Given a video clip, the model is asked to explain what the video intends to express as a whole (An example is provided in Figure 5). This task requires models to identify the implicit message conveyed through multimodal evidence. The annotation process results in 588

Table 1: Main results on ViMU across open-ended interpretation (OE), evidence grounding (EG), rhetoric mechanisms identification (RM), and social value signal identification (SV). All values are percentage scores (%). SSU-Avg denotes the average of the two structured subtext understanding tasks, RM and SV. All-Avg denotes the average across all four tasks. Green shades mark the top-3 models in each metric column, and purple shades mark the bottom-3 models.

Model Date OE EG RM SV SSU-Avg All-Avg Open-weight Models

Ministral-8B 2024-10 48.25 48.60 31.87 10.45 21.16 34.79 Ministral-14B 2025-12 52.19 55.73 27.29 6.57 16.93 35.45 Gemma-3-4B-it 2025-03 39.43 25.41 21.10 7.17 14.13 23.28 Gemma-3-27B-it 2025-03 55.90 49.38 32.47 7.95 20.21 36.43 Qwen3-VL-32B-Instruct 2025-10 64.09 59.64 27.65 15.17 21.41 41.64 Qwen3.5-27B 2026-02 62.80 60.28 38.18 22.40 30.29 45.91

Closed-source / API Models

Claude-3-Haiku 2024-03 50.41 34.55 2.99 3.64 3.32 22.90 GLM-4.5v 2025-08 62.52 23.11 8.87 9.26 9.06 25.94 Grok-4.1-Fast 2025-09 57.62 63.84 34.91 28.73 31.82 46.28 Gemini-3-Flash-Preview 2025-12 62.54 52.80 33.63 28.26 30.94 44.31 Mimo-V2-Omni 2026-03 64.07 48.94 21.04 18.52 19.78 38.14 Seed-2.0-Lite 2026-03 60.84 66.16 18.75 16.73 17.74 40.62 o4-mini 2025-04 65.27 59.63 33.21 29.51 31.36 46.91

- GPT-4.1-nano 2025-04 50.12 22.31 2.32 9.02 5.67 20.94

- GPT-5.2 2025-12 73.15 67.83 16.55 21.15 18.85 44.67 GPT-5.4-mini 2026-03 66.19 64.45 4.17 11.77 7.97 36.64

questions. The model responses are evaluated by comparing them with the reference interpretation using a structured grading rubric via LLM-as-a-Judge (details are provided in Appendix B).

#### 3.1.2 Semantic-Structure Understanding

Rhetoric Mechanism Identification (RMI). This task requires models to recognize the rhetorical devices used to construct the video’s message (An example is provided in Figure 4a). Given a video, the model must select all applicable choices from a predefined list. Here, to improve evaluation

stability and interpretability, we further group all rhetorical mechanisms in Figure 2 into five categories (see Appendix C for details). The task is finally formulated as a multiple-choice problem.

25

Claude

GLM

20

Gemini

G54M

Gemma

15

Social Value Signal Identification (SVI). This task evaluates whether models can identify the social stance or normative implication conveyed by

Grok

PC2(18.5%var)

10

Mistral

C3H

OpenAI

G41N G52

5

- the video (An example is provided in Figure 4b). Similar to the rhetoric mechanism task, this problem is also formulated as a multiple-choice problem. All the social value signals in Figure 2 are grouped into five categories (Details are provided in Appendix D).

3.1.3 Evidence-Grounded Understanding

Evidence Grounding (EG). This task examines whether models can correctly identify the multimodal evidence supporting their interpretation of

- the video (An example is provided in Figure 4c). The candidate evidence sources are the five types

Qwen

Seed

SEED MIMO

0

GM34B GR41F GM327B

O4M QV32B

Xiaomi

GM3FP

Q3527B

GLM45V

5

MN14B MN8B

10

15 10 5 0 5 10 15 20 PC1 (32.9% var)

- Figure 7: PCA visualization of model similarity based on error signatures in the macro-5 taxonomy tasks. Each point denotes one model; distances reflect similarity in structured error profiles rather than overall score.

illustrated in Figure 2. The task is structured as a multiple-choice problem. This task allows us to analyze whether model reasoning is grounded in observable video cues rather than unsupported speculation.

### 4 Experiments and Analysis

Settings. We conduct a comprehensive investigation of 16 MLLMs using our ViMU benchmark, encompassing both open-source and proprietary models. For all the considered MLLMs, we employ either a uniform sampling strategy for video processing. All models are evaluated based on their official implementations or available APIs [2], with evaluations conducted in a zero-shot manner. More details about the evaluation are provided in Appendix I.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

1.0

1.0

1.0

1.0

0.30 0.22 0.13 0.10 0.02

0.30 0.11 0.01 0.01 0.00

0.09 0.04 0.05 0.01 0.01

0.96 0.03 0.00 0.01 0.00

- A

- B

- C

- D

- E

- A

- B

- C

- D

- E

- A

- B

- C

- D

- E

- A

- B

- C

- D

- E

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

+0.01 +0.01 +0.00 -0.00 -0.00

+0.00 +0.00 -0.00 +0.00 -0.00

- A

- B

- C

- D

- E

- A

- B

- C

- D

- E

[Figure 181]

[Figure 182]

0.02

0.02

0.8

0.8

0.8

0.8

0.22 1.00 0.56 0.29 0.14

0.11 1.00 0.31 0.13 0.19

- 0.04 1.00 0.51 0.30 0.40
- 0.05 0.51 0.70 0.16 0.27

0.03 0.50 0.29 0.07 0.12

+0.01 +0.00 -0.00 -0.02 -0.01

+0.00 +0.00 -0.02 -0.00 -0.01

0.01

0.01

0.6

0.6

0.6

0.6

0.13 0.56 0.64 0.20 0.09

0.01 0.31 0.38 0.07 0.08

- 0.00 0.29 0.82 0.09 0.40
- 0.01 0.07 0.09 0.27 0.11

+0.00 -0.00 -0.00 -0.01 -0.01

-0.00 -0.02 -0.02 -0.01 -0.01

0.00

0.00

0.4

0.4

0.4

0.4

0.10 0.29 0.20 0.35 0.03

0.01 0.13 0.07 0.16 0.03

0.01 0.30 0.16 0.42 0.09

0.01

- -0.00 -0.02 -0.01 -0.02 -0.00
- -0.00 -0.01 -0.01 -0.00 -0.01 A B C D E

+0.00 -0.00 -0.01 -0.00 -0.00

0.01

0.2

0.2

0.2

0.2

0.02

0.02 0.14 0.09 0.03 0.15

0.0 0.00 0.19 0.08 0.03 0.23

0.01 0.40 0.27 0.09 0.52

0.0 0.00 0.12 0.40 0.11 1.00

0.02 -0.00 -0.01 -0.01 -0.00 -0.02

0.0

0.0

A B C D E

A B C D E

A B C D E

A B C D E

A B C D E

(a) GT co-occurrence structure.

(b) Pr. co-occurrence w.o. guidance.

(c) Co-occurrence change.

| |without guidance| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |with guidance| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

Avg.geom.dist.

1.5

1.0

0.5

0.0

claude-3-haikugemini-3-flash-previewgemma-3-27b-itgemma-3-4b-itglm-4.5vgpt-4.1-nanogpt-5.2gpt-5.4-minigrok-4.1-fastmimo-v2-omniministral-14bministral-8bqwen3-vl-32b-instructo4-mini qwen3.5-27bseed-2.0-lite

(d) Average taxonomy-geometry distortion across models.

- Figure 8: Taxonomy geometry analysis of EG and RM predictions. The top row compares the pairwise co-occurrence structure of the ground-truth choices and model predictions. The bottom panel summarizes the average geometry distortion of each model relative to the ground-truth structure.

Overall Performance Analysis. Table 1 reveals a clear pattern: Current models exhibit substantially weaker performance on metaphorical understanding than on general video understanding, which is precisely the gap that ViMU aims to expose. For open-ended interpretation (OE), the strongest performance is achieved by GPT-5.2, which also attains the best evidence grounding (EG) results, both around 70%. However, when tasked with identifying specific rhetoric mechanisms (RM) and social value signals (SV), its performance drops sharply to around 20%. In contrast, models such as Grok-4.1-Fast and Gemini-3-Flash-Preview, while less competitive on OE and EG, achieve significantly better results on RM and SV, reaching around 30%. From these results, we draw three key conclusions: (i) frontier capability in general video interpretation does not automatically translate into precise understanding of implicit stance, rhetorical framing, or socially coded meaning; (ii) different model families, and even models within the same family, exhibit distinct strengths in metaphorical understanding; (iii) closed-source models are not uniformly superior to open-weight models (e.g., Qwen3.5-27B achieves a higher All-Avg than GPT-4.1-nano and Claude-3-Haiku). From a benchmark perspective, these results show that ViMU isolates hidden communicative reasoning and exposes its gap with standard video understanding.

Analysis on Evidence Grounding (EG). Figure 6a visualizes how each model trades off evidence-selection conservatism against overall grounding quality: the x-axis measures whether a model tends to under-select or over-select evidence relative to the gold answer, while the y-axis reports its Micro-F1. For readability, we abbreviate model names as follows: C3H = claude-3-haiku, GM3FP = gemini-3-flash-preview, GLM45V = glm-4.5v, G41N = gpt-4.1-nano, G52 = gpt-5.2, MIMO = mimo-v2-omni, O4M = o4-mini, SEED = seed-2.0-lite, GM327B = gemma-3-27b-it, GM34B = gemma-3-4b-it, MN14B = ministral-14b, MN8B = ministral-8b, and Q3527B = qwen3.5-27b. Figure 6a therefore characterizes the selection style of different models rather than only their final score. As shown, most models lie on the conservative side, indicating that they tend to predict fewer evidence sources than the annotations require (x-axis < 0). Mild conservatism does not necessarily reduce performance, but excessive under-selection is clearly harmful: the most conservative outlier is also among the weakest performers. At the same time, the top closed models occupy the upper region of the figure, whereas the strongest open-weight models are competitive but still generally fall slightly below the best closed

models. Overall, Fig. 6a suggests that the main risk in current evidence grounding models is not aggressive over-selection, but incomplete retrieval of supporting evidence.

- Figure 6b further decompose each model’s predictions into four error types—Exact, Miss-Only,

A B C D E

C3H

GM3FP

GM327B

GM34B

GLM45V

G41N

G52

G54M

GR41F

MIMO

MN14B

MN8B

O4M

QV32B

Q3527B

SEED

+0.56 -0.13 -0.13 -0.21 -0.42

- -0.06 +0.17 -0.25 +0.02 -0.21

+0.12 +0.16 -0.29 -0.26 -0.41

- +0.19 +0.10 +0.00 -0.24 -0.29

+0.05 -0.51 -0.42 -0.35 -0.43

+0.82 +0.06 -0.17 -0.28 -0.38

+0.21 -0.10 +0.35 +0.25 -0.26

+0.75 -0.15 +0.00 +0.56 -0.37

-0.03 +0.14 -0.22 -0.02 -0.36

- +0.20 -0.08 -0.13 -0.02 -0.36

- -0.07 +0.14 +0.39 -0.16 -0.07
- -0.07 +0.13 +0.34 -0.27 -0.27

- +0.05 +0.09 -0.11 +0.00 -0.41

+0.00 +0.00 +0.37 -0.16 -0.28

-0.04 -0.01 -0.07 -0.24 -0.36

- +0.06 +0.08 +0.24 +0.11 -0.28

(a) Rhetoric Mechanisms Identification.

A B C D E

C3H

GM3FP

GM327B

GM34B

GLM45V

G41N

G52

G54M

GR41F

MIMO

MN14B

MN8B

O4M

QV32B

Q3527B

SEED

- -0.02 +0.44 -0.08 +0.09 -0.35
- -0.08 +0.38 -0.07 +0.01 -0.25

+0.09 +0.77 -0.22 -0.10 -0.38

- -0.36 +0.79 +0.08 +0.13 -0.05
- -0.27 +0.20 -0.28 -0.10 -0.40

+0.00 +0.57 +0.01 -0.05 -0.34

- -0.20 +0.58 -0.15 +0.02 -0.34
- -0.04 +0.69 -0.02 +0.19 -0.34
- -0.07 +0.34 +0.07 +0.03 -0.21
- -0.07 +0.27 -0.19 +0.00 -0.37
- -0.36 +0.76 +0.03 +0.04 +0.25
- -0.36 +0.79 -0.12 -0.08 +0.01

+0.02 +0.31 -0.16 -0.07 -0.37

- -0.28 +0.50 +0.09 +0.05 -0.14
- -0.21 +0.39 -0.13 -0.05 -0.36
- -0.22 +0.61 -0.11 -0.02 -0.36

(b) Social Value Signal Identification.

Figure 9: Model–option affinity bias without guidance. Positive values indicate over-prediction relative to ground-truth prevalence, while negative values indicate under-prediction.

Extra-Only, and Mixed—so as to show how models fail rather than merely how often they fail. The figure reveals that a substantial portion of non-exact predictions is driven by omissionrelated errors, especially Miss-Only and Mixed, while purely over-selective behavior (ExtraOnly) is generally less dominant. This confirms the trend already suggested by Fig. 6a: evidence grounding errors are driven more by failing to retrieve all required cues than by indiscriminately hallucinating additional evidence.

- Figure 6c analyzes evidence grounding at the level of pairwise evidence relations. Specifically, it compares the average co-selection pattern produced by models against the gold cooccurrence pattern, thereby revealing whether models over-link or under-link different evidence types. The matrix is uniformly negative, which means that, on average, models under-predict evidence co-occurrence rather than over-connecting evidence sources. The largest negative deviations involve editingrelated pairs, especially editing–frames and editing–text, whereas audio-related relations remain much closer to zero. This suggests that current models are relatively better at handling isolated perceptual cues, but substantially weaker at recovering structured multi-source evidence patterns, particularly when editing signals must be integrated with visual or textual evidence.

Analysis on Rhetoric Mechanisms (RM) and Social Value Signals (SV). Figure 7 visualizes model similarity by applying PCA to each model’s error-signature vector on the RM and SV tasks. PC1

and PC2 denote the first two principal components, explaining 32.9% and 18.5% of the variance, respectively. Note that models lie closer have more similar structured error profiles. We observe family-level clustering: OpenAI models tend to be grouped together, and Qwen and Mistral models also remain close to their family peers, indicating shared inductive biases in how they organize taxonomy labels. Overall,Figure 7 shows that models with similar overall performance may still differ substantially in their decision patterns, and that behavior is strongly shaped by model family.

Rhetoric Social Value

Mean > 0 Mean < 0

| |
|---|

| |
|---|

| |
|---|

0.02

FalsePositiveRate

0.00

0.02

0.04

0.06

A B C D E

In Figure 8, we study taxonomy geometry preservation by comparing the pairwise co-occurrence structure of the five ground-truth choices and model predictions. For each task (rhetoric and social value), we construct a normalized (5×5) co-occurrence matrix, where diagonal entries reflect label prevalence and off-diagonal entries capture label interactions. Figure 8a shows the ground-truth geometry, Figure 8b

Figure 10: Category-wise distribution of guidance-induced shifts in false positive rate (∆FPR). Each violin summarizes the distribution over models for a given category, with rhetoric (green) and social value (red) shown side by side. Points denote model-level values, while markers indicate mean shifts.

shows the average prediction geometry without extra guidance, Figure 8c shows the change induced by guidance, and Figure 8d summarizes model-wise distortion using the Frobenius distance to the ground-truth matrix. The guidance information can be found in Appendix G. As can be observed, models recover part of the taxonomy structure, as several dominant co-occurrence patterns in the annotations also appear in predictions, but the predicted matrices are generally flatter and less con-

trasted, suggesting that fine-grained relations are only partially preserved. Furthermore, the guidance introduces mostly small but systematic local shifts in pairwise relations, yet these changes do not consistently improve global structural fidelity: for many models, the distance to the ground-truth geometry remains similar or becomes slightly larger. Overall, Figure 8 indicates that models capture limited taxonomy structure, and guidance mainly reweights local decisions rather than restoring global structure.

The affinity-bias maps in Figure 9 reveal clear option-level biases rather than uniform error. In rhetoric, many models over-predict A (Literal / Direct) and under-predict E (Implicit / Coded Social Framing), suggesting a tendency to map difficult cases to safer or more generic categories. As for social signal, most models strongly over-predict B (Emotional Attitude) while under-predicting E (Identity / Ideological Signaling), indicating that broad affective readings often act as a default interpretation. We also find that the with-guidance results are qualitatively very similar to the withoutguidance ones (See Appendix H), which suggests that guidance does not substantially change the underlying option-allocation structure but only makes small local adjustments.

In Figure 10, we explore how guidance affects model false positive behavior across different categories. The average results of all considered models are report. For rhetoric, categories such as B (Opposition Incongruity) exhibit larger variance, suggesting increased instability when models handle contrastive or unexpected structures, while D (Amplification Stylization) tends to shift negatively, reflecting more conservative predictions. For social value signals, the overall shifts are more compact but show stronger polarization in categories like E (Identity Ideological Signaling), where models become more conservative yet less consistent across instances.

### 5 Conclusion

In this work, we introduce ViMU, a benchmark designed to evaluate video models beyond literal perception by focusing on subtext understanding, including rhetorical, social, and culturally grounded meanings. Our results show that, despite strong performance on surface-level tasks, current frontier models struggle substantially with interpreting implicit meaning, achieving below 50% overall performance. Through fine-grained analyses, we further reveal systematic gaps and distinct behavioral patterns across models. These findings highlight a fundamental limitation of existing video understanding systems and suggest that advancing toward robust, human-like interpretation requires modeling not only what is shown, but also what is meant.

### References

- [1] Mythologies (book). https://en.wikipedia.org/wiki/Mythologies_(book).
- [2] Openrouter: Unified api for large language models. https://openrouter.ai.
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [5] Tieyuan Chen, Huabin Liu, Tianyao He, Yihang Chen, Chaofan Gan, Xiao Ma, Cheng Zhong, Yang Zhang, Yingxue Wang, Hui Lin, et al. Mecd: Unlocking multi-event causal discovery in video reasoning. Advances in neural information processing systems, 37:92554–92580, 2024.
- [6] Tieyuan Chen, Huabin Liu, Yi Wang, Yihang Chen, Tianyao He, Chaofan Gan, Huanyu He, and Weiyao Lin. Mecd+: Unlocking event-level causal graph discovery for video reasoning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [7] Tieyuan Chen, Huabin Liu, Yi Wang, Chaofan Gan, Mingxi Lyu, Ziran Qin, Shijie Li, Liquan Shen, Junhui Hou, Zheng Wang, et al. Looking beyond visible cues: Implicit video question answering via dual-clue reasoning. arXiv preprint arXiv:2506.07811, 2025.

- [8] Zhiyuan Fang, Tejas Gokhale, Pratyay Banerjee, Chitta Baral, and Yezhou Yang. Video2commonsense: Generating commonsense descriptions to enrich video captioning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 840–860, 2020.
- [9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.
- [10] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [11] Stuart Hall. Encoding—decoding (1980). In Crime and media, pages 44–55. Routledge, 2019.
- [12] Chien-yu Huang, Ke-Han Lu, Shih-Heng Wang, Chi-Yuan Hsiao, Chun-Yi Kuan, Haibin Wu, Siddhant Arora, Kai-Wei Chang, Jiatong Shi, Yifan Peng, et al. Dynamic-superb: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12136–12140. IEEE, 2024.
- [13] Xilin Jiang, Qiaolin Wang, Junkai Wu, Xiaomin He, Zhongweiyang Xu, Yinghao Ma, Minshuo Piao, Kaiyi Yang, Xiuwen Zheng, Riki Shimizu, et al. Avmeme exam: A multimodal multilingual multicultural benchmark for llms’ contextual and cultural knowledge and thinking. arXiv preprint arXiv:2601.17645, 2026.
- [14] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33:2611–2624, 2020.
- [15] Gunther Kress and Theo Van Leeuwen. Reading images: The grammar of visual design. Routledge, 2020.
- [16] Andrew N Leak. Barthes: mythologies. Grant and Cutler, 1994.
- [17] DongGeon Lee, Joonwon Jang, Jihae Jeong, and Hwanjo Yu. Are vision-language models safe in the wild? a meme-based benchmark study. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 30533–30576, 2025.
- [18] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [19] Qi Li, Cheng-Long Wang, Yinzhi Cao, and Di Wang. Cola: A choice leakage attack framework to expose privacy risks in subset training. arXiv preprint arXiv:2604.12342, 2026.
- [20] Qi Li, Runpeng Yu, and Xinchao Wang. Vid-sme: Membership inference attacks against large video understanding models. Advances in Neural Information Processing Systems, 38:111572– 111596, 2026.
- [21] Hongzhan Lin, Ziyang Luo, Bo Wang, Ruichao Yang, and Jing Ma. Goat-bench: Safety insights to large multimodal models through meme-based social abuse. ACM Transactions on Intelligent Systems and Technology, 2024.
- [22] Ziyang Ma, Yinghao Ma, Yanqiao Zhu, Chen Yang, Yi-Wen Chao, Ruiyang Xu, Wenxi Chen, Yuanzhe Chen, Zhuo Chen, Jian Cong, et al. Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix. arXiv preprint arXiv:2505.13032, 2025.
- [23] Jae Sung Park, Chandra Bhagavatula, Roozbeh Mottaghi, Ali Farhadi, and Yejin Choi. Visualcomet: Reasoning about the dynamic context of a still image. In European Conference on Computer Vision, pages 508–524. Springer, 2020.

- [24] Sakshi Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024.
- [25] Shivam Sharma, Siddhant Agarwal, Tharun Suresh, Preslav Nakov, Md Shad Akhtar, and Tanmoy Chakraborty. What do you meme? generating explanations for visual semantic role labelling in memes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9763–9771, 2023.
- [26] Zhengpeng Shi, Hengli Li, Yanpeng Zhao, Jianqun Zhou, Yuxuan Wang, Qinrong Cui, Wei Bi, Songchun Zhu, Bo Zhao, and Zilong Zheng. V-hub: A visual-centric humor understanding benchmark for video llms. arXiv preprint arXiv:2509.25773, 2025.
- [27] Sirnam Swetha, Rohit Gupta, Parth Parag Kulkarni, David G Shatwell, Jeffrey A Chan Santiago, Nyle Siddiqui, Joseph Fioresi, and Mubarak Shah. Vrr-qa: Visual relational reasoning in videos beyond explicit cues, 2026.
- [28] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
- [29] Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy Chen. Audiobench: A universal benchmark for audio large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4297–4316, 2025.
- [30] Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, and Di Wang. Towards lifecycle unlearning commitment management: Measuring sample-level unlearning completeness. In 34th USENIX Security Symposium (USENIX Security 25), pages 6481–6500, 2025.
- [31] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958– 22967, 2025.
- [32] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13204–13214, 2024.
- [33] Binzhu Xie, Sicheng Zhang, Zitang Zhou, Bo Li, Yuanhan Zhang, Jack Hessel, Jingkang Yang, and Ziwei Liu. Funqa: Towards surprising video comprehension. In European Conference on Computer Vision, pages 39–57. Springer, 2024.
- [34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [35] Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models: A survey. arXiv preprint arXiv:2506.13759, 2025.
- [36] Zhengyi Zhao, Shubo Zhang, Yuxi Zhang, Yanxi Zhao, Yifan Zhang, Zezhong Wang, Huimin Wang, Yutian Zhao, Bin Liang, Yefeng Zheng, et al. Memereacon: Probing contextual meme understanding in large vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3559–3582, 2025.
- [37] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13691–13701, 2025.

### A Details of the Dataset Curation Process

The construction of ViMU follows a multi-stage pipeline that integrates multimodal evidence extraction with LLM-driven semantic annotation and question refinement, as well as human expert review. The overall goal is to produce high-quality, hint-free benchmark instances that require genuine subtext understanding. An illustration of the curation pipeline can be found in Figure 11.

- Stage 1: Multimodal Evidence Extraction. Given a set of videos V = {vi}, we construct for each video a multimodal evidence representation by extracting uniformly sampled frames Fi and an audio transcript ti, yielding:

Ei = {Fi,ti}. This ensures that all downstream reasoning is grounded in observable signals rather than external metadata.

- Stage 2: LLM-based Taxonomy Annotation. We employ a frontier modelc (GPT-5.4) to produce structured semantic annotations for each video. The model is prompted to separate literal content from intended meaning and to decompose subtext into multiple axes, including rhetorical mechanisms and social value signals:

Ti = fLLM(Ei).

Annotations rely only on evidence in Ei, ensuring grounding and interpretability.

|Raw Videos|
|---|

|Evidence Extraction<br><br>Frames + ASR|
|---|

|LLM Taxonomy Annotation<br><br>Literal + Subtext|
|---|

|Filtering<br><br>Self-contained|
|---|

|Hint-free QA Generation|
|---|

|LLM Validation<br><br>Iterative Refinement|
|---|

|Human Expert<br><br>Final Review|
|---|

feedback loop (K < 3)

|Task Construction<br><br>OE + EG + RM + SV|
|---|

|Final ViMU<br><br>Validated Benchmark|
|---|

|Open-ended QA|
|---|

|Evidence Grounding|
|---|

|Rhetoric Mechanism|
|---|

|Social Value Signal|
|---|

Figure 11: An illustration of the dataset curation process.

- Stage 3: LLM-based Hint-Free Question Generation. Conditioned on the taxonomy Ti, we use the same LLM to generate a question–answer pair (qi,ai):

(qi,ai) = gLLM(Ti).

- A key constraint is that qi must be hint-free, i.e., it must not explicitly reveal the semantic dimension (e.g., rhetoric or social meaning) being tested. This prevents shortcut learning and forces genuine inference of the intended meaning. This is achieved through the explicit enforcement in the prompt and the iterative refinement in Stage 4.

- Stage 4: Iterative LLM-based Validation and Refinement. To ensure quality, each generated QA instance undergoes an iterative refinement loop. At iteration k, the LLM evaluates the current QA instance Q(ik) and produces structured feedback:

feedback(k) = hLLM(Ti,Q(ik)). The QA is then updated:

Q(ik+1) = gLLM(Ti,feedback(k)),

until it satisfies criteria such as implicitness, difficulty, and alignment with intended meaning. This loop explicitly enforces that the question cannot be solved using surface-level cues alone. We allow at most K = 3 refinement rounds. At round k = 0, the initially generated QA pair is validated. If it is marked as pass or minor revision, it is accepted; if it is marked as reject, it is discarded. If it receives a major revision label, the feedback is used to regenerate the QA pair for the next round. Samples that still fail after K refinement rounds are rejected.

- Stage 5: Evidence and Task Construction. Using the annotated evidence fields, we derive a unified set of evidence sources and construct structured evaluation tasks, including evidence grounding and taxonomy classification. Fine-grained labels are further aggregated into macro-level categories (5 categories) to support analysis at different abstraction levels. Details are given in Appendix C and D

Filtering and Quality Control. We retain only samples that are self-contained and suitable for fair evaluation, filtering out videos that require strong external context or exhibit ambiguous semantics. The dataset is finally validated by 5 human experts. This ensures that performance reflects intrinsic video understanding rather than external knowledge retrieval. The curation process results in a high quality dataset containing 588 videos.

Summary. The final dataset is constructed through a pipeline:

vi → Ei → Ti → (qi,ai) → Q∗i ,

where Q∗i denotes the validated QA instance. Notably, LLMs are used not only as annotators but also as generators and iterative refiners, enabling scalable yet high-quality benchmark construction.

#### A.1 Prompt Design Details

In this section, we summarize the key prompts used for taxonomy annotation, question generation, and iterative validation as follows.

#### Taxonomy Annotation Prompt

Video ID: <video_id> Duration (sec): <duration> Task: Analyze this video meme in a hierarchical, multi-axis way. Available evidence:

- 1. Video frames
- 2. Transcript from audio ASR (may be empty, noisy, or partial): <transcript> Important constraints:

- - You must infer visible on-screen text directly from the frames when relevant.
- - Do not assume any external posting context beyond the video itself.
- - Separate literal content from intended meaning.
- - Use only evidence supported by frames, transcript, visible text, audio tone, or editing/timing cues.
- - If the video is not self-contained, reflect that in the interpretability bucket.

This prompt corresponds exactly to the user-level input used for taxonomy labeling.

#### Hint-Free Question Generation Prompt

Video ID: <vid_id> You are given the annotation for one video meme.

Taxonomy JSON: <taxonomy json> Task:

Create exactly one difficult, hint-free question for evaluating a video understanding model. Hard constraints:

- - The question must NOT contain words that leak the target semantic field, such as: hate, bias, stereotype, discrimination, offensive, toxic, negative, sarcasm, irony, prejudice, controversial
- - The question should not presuppose that the video has hidden meaning.

- - The question should require understanding the intended meaning, not just literal description.
- - The gold answer should be concise but semantically complete.
- - The rubric should make later LLM judging reliable.

This prompt is used for initial QA generation.

#### Validation Prompt

Taxonomy JSON: <taxonomy json> Question package JSON: <qa json> Validation task: Check whether this benchmark question is genuinely hint-free, difficult, and aligned with the intended meaning. Evaluate the following:

- 1. Does the question leak the semantic field or sensitive framing?
- 2. Can the question be answered correctly using only surface-level description?
- 3. Does it force understanding of the intended meaning?
- 4. Is the difficulty appropriate given the video’s taxonomy?
- 5. Is the gold answer aligned with the intended meaning?
- 6. Is the rubric strong enough for later LLM judging? If the question is flawed, provide a better revised_question that is harder and less revealing while still evaluable.

This prompt defines the structured validation step. Iterative Refinement Prompt (Augmented Generation) You are given the annotation for a video. Taxonomy JSON: <taxonomy json> Task: Create exactly one difficult, hint-free question for evaluating a video understanding model. Hard constraints:

- - The question must NOT contain words that leak the target semantic field, such as: hate, bias, stereotype, discrimination, offensive, toxic, negative, sarcasm, irony, prejudice, controversial
- - The question should not presuppose that the video has hidden meaning.
- - The question should require understanding the intended meaning, not just literal description.
- - The gold answer should be concise but semantically complete.
- - The rubric should make later LLM judging reliable. The previous QA attempt was judged as: <validation json> You must improve the QA package based on this feedback. Important improvement instructions:
- - Fix the issues listed in "issues" and "fix_suggestions".
- - If "surface_answerable" was true, make the new question require a more specific implicit inference.
- - If "requires_implicit_understanding" was false, make the question force recovery of the intended meaning.
- - If "difficulty_fit" was "too_easy" or "too_hard", recalibrate the question difficulty.
- - If a "revised_question" is provided by the validator, you may adopt or improve it.
- - Keep the question neutral and non-leading.

This prompt is used during iterative refinement rounds. Multi-Choice Question Prompt You are answering a multi-choice question about a video. Question:

<question> Transcript from ASR (may be noisy, partial, or empty): <transcript> <extra_context> Instruction: <instruction> Available options: <option_text> Return only valid JSON with one field:

{{ "selected_options": ["A", "B"] }} Rules:

- - Select only options that are clearly supported by the video.
- - Return only option letters, not option texts.
- - Do not include any option not in the provided list.
- - Be conservative: do not select an option unless it is clearly justified by the video.
- - If no option is clearly supported, return an empty list.

Open-Ended Question Prompt You are answering a question about a video. Question:

{question} Transcript from ASR (may be noisy, partial, or empty): {transcript} Instructions:

- - Answer the question directly.
- - Use the video frames as the primary source of truth.
- - Infer visible on-screen text from the frames when relevant.
- - Keep the answer concise but semantically complete.
- - Do not add safety disclaimers unless absolutely necessary.

### B LLM-as-a-Judge for Open-Ended Questions

To evaluate model performance on open-ended, hint-free questions, we adopt an LLM-as-a-judge framework that assesses semantic understanding rather than surface-level similarity. Instead of relying on exact match or n-gram overlap, the judge model evaluates whether a prediction captures the intended meaning of the video.

Judging Framework. Given a question q, a gold answer a∗, and a model prediction aˆ, the judge receives the following structured inputs: (i) the question, (ii) the gold answer, (iii) a set of reference points summarizing key aspects of the intended meaning, and (iv) a grading rubric specifying evaluation criteria. The judge then produces a structured judgment consisting of a scalar score, a detailed breakdown, and a qualitative verdict.

Scoring Dimensions. The evaluation decomposes semantic understanding into five components:

- • Core Intent (0–5): whether the prediction captures the video’s primary intended meaning.
- • Implicit Signal (0–3): whether it correctly identifies the key rhetorical or social signal.
- • Target or Social Meaning (0–1): whether it recognizes relevant targets, groups, or social implications when applicable.
- • Hallucination Penalty (0–3): penalizes unsupported or fabricated claims.
- • Literal-Only Penalty (0–3): penalizes answers that remain at surface-level description without capturing subtext.

The final score is computed as:

scoretotal = core_intent + implicit_signal + target_or_social_meaning

− hallucination_penalty − literal_only_penalty.

(1)

This formulation explicitly rewards semantic understanding while penalizing both hallucination and shallow interpretation.

Judgment Output. In addition to the scalar score, the judge produces: (i) a structured score breakdown, (ii) a categorical verdict from {excellent, good, partial, poor, wrong}, and (iii) a short natural language justification.

This structured output enables both quantitative evaluation and qualitative analysis of model behavior.

Design Principles. The LLM judge is designed to follow three principles:

- • Semantic over lexical matching: evaluations are based on meaning rather than wording.
- • Strict hallucination control: unsupported claims are explicitly penalized.
- • Subtext sensitivity: answers that fail to capture implicit meaning receive lower scores even if they are factually correct at the surface level.

#### Judge Prompt

You are grading answers in a benchmark for hint-free implicit video understanding. Your job is to judge semantic understanding, not style. Scoring dimensions:

- - core_intent: Did the model capture the main intended meaning?
- - implicit_signal: Did it recognize the crucial hidden rhetorical or social signal?
- - target_or_social_meaning: Did it identify a relevant target, group, institution, or social implication when supported?
- - hallucination_penalty: Penalize invented claims not grounded by the gold answer, evidence, or rubric.
- - literal_only_penalty: Penalize answers that remain at surface description and miss the point. Score bounds:
- - core_intent must be one of 0, 1, 2, 3, 4, 5
- - implicit_signal must be one of 0, 1, 2, 3
- - target_or_social_meaning must be one of 0, 1
- - hallucination_penalty must be one of 0, 1, 2, 3
- - literal_only_penalty must be one of 0, 1, 2, 3 Scoring rule: score_total = core_intent + implicit_signal + target_or_social_meaning
- - hallucination_penalty - literal_only_penalty The maximum possible score is 9. Interpretation guide:
- - A partially correct answer that captures the meme’s point should score much higher than a polished but purely literal answer.
- - Do not require exact wording match.
- - Be strict with hallucinations.
- - Only assign target_or_social_meaning = 1 when that dimension is genuinely relevant and correctly captured.
- - Use the evidence as grounding support, not as extra hidden labels to overfit.
- - Keep reasoning_short concise. Return only valid JSON.

### C Rhetoric Mechanism Grouping

- Table 2 defines the mapping from fine-grained rhetoric mechanism labels to five macro categories, which serve as the basis for structured evaluation and analysis.

D Social Value Signals Grouping

- Table 3 defines the mapping from fine-grained social value signals to macro-level categories, enabling consistent evaluation of subtext across models.

### E Details of the Taxonomy of Rhetoric Mechanisms

Literal Only. The video’s meaning is largely exhausted by its surface content, with little or no reliance on non-literal interpretation, rhetorical reframing, or implicit contrast.

Sarcasm. The video conveys meaning by expressing a surface attitude that is intentionally opposite to the speaker’s or creator’s actual attitude, typically to signal ridicule, dismissal, or criticism.

Irony. The video derives meaning from a discrepancy between appearance and reality, expectation and outcome, or explicit expression and underlying implication, without necessarily requiring direct mocking intent.

Mockery. The video is structured to ridicule, belittle, or make fun of a target, often by highlighting flaws, incompetence, absurdity, or hypocrisy.

Table 2: Mapping for rhetoric mechanisms.

Macro Category Subcategory

Literal / Direct Literal Only Opposition / Incongruity Contrast

Bait and Switch Role Reversal Absurdism

Attitude / Tone-based Rhetoric Sarcasm Irony Deadpan Mockery

Amplification / Stylization Exaggeration Parody

Implicit / Coded Social Framing Innuendo Stereotype Invocation Dog Whistle or Code Other

Stereotype Invocation. The video relies on a recognizable stereotype, trope, or socially shared caricature to construct its meaning, whether for humor, critique, reinforcement, or inversion.

Exaggeration. The intended meaning is amplified through deliberate overstatement, extreme depiction, or disproportionate framing beyond what would be literally plausible.

Contrast. The meaning is produced through juxtaposition between two incompatible or sharply different elements, such as tone, image, text, expectation, or social role.

Innuendo. The video implies a sensitive, suggestive, or socially loaded meaning indirectly, without stating it explicitly, often relying on implication rather than overt expression.

Absurdism. The video constructs meaning through deliberate irrationality, impossibility, or surreal mismatch, where the humor or point depends on embracing the nonsensical.

Role Reversal. The video derives meaning by inverting expected roles, positions, hierarchies, or behavioral norms, so that one party acts in a way conventionally associated with another.

Dog Whistle or Code. The video contains coded references, euphemisms, or indirect signals that are intended to be legible primarily to audiences with relevant cultural, political, or subcultural knowledge.

Bait and Switch. The video sets up one expectation and then abruptly replaces it with a different, often incompatible, payoff, producing humor or commentary through misdirection.

Table 3: Mapping for social value signals.

Macro Category Subcategory

Neutral / No Social Signal None Emotional Attitude Negative Affect

Fatalism or Cynicism

Deadpan. The video presents absurd, ironic, or exaggerated content in a deliberately flat, matter-of-fact, or emotionally neutral manner, making the restrained delivery central to the effect.

Social Evaluation / Devaluation Contempt Humiliation Aggression or Hostility Exclusion Discrimination or Prejudice

Norm and Value Framing Norm Violation Anti-mainstream Value

Identity / Ideological Signaling Political or Identity Signal Sexual Implication Other

Parody. The video imitates the style, structure, or conventions of another genre, person, discourse, or media form in order to create humor, critique, or commentary.

Other. The video relies on a rhetorical mechanism not adequately captured by the categories above, or on a hybrid mechanism that cannot be cleanly reduced to a single listed type.

### F Details of the Taxonomy of Social Value Signals

None. The video does not clearly communicate a salient social attitude, value judgment, or normative stance beyond its immediate surface content.

Negative Affect. The video conveys or evokes a broadly negative emotional tone, such as frustration, displeasure, discomfort, annoyance, or aversion, without necessarily specifying a stronger social stance.

Contempt. The video signals scorn, disdain, or a sense of superiority toward a target, often implying that the target is foolish, inferior, pathetic, or unworthy of respect.

Exclusion. The video communicates boundary-making, rejection, or denial of belonging, whether socially, culturally, morally, or group-wise.

Discrimination or Prejudice. The video conveys bias, derogation, or unequal judgment toward a group or identity category, whether explicitly or through implication, stereotype, or coded framing.

Norm Violation. The video foregrounds behavior, values, or situations as improper, transgressive, taboo, or outside expected social rules or conventions.

Anti-Mainstream Value. The video endorses, celebrates, or signals attitudes positioned against widely accepted norms, tastes, or mainstream moral or social expectations.

Fatalism or Cynicism. The video expresses resignation, hopelessness, distrust, or a dismissive belief that outcomes, people, or institutions are fundamentally flawed or unchangeable.

Sexual Implication. The video conveys sexualized meaning, innuendo, erotic framing, or sexually suggestive interpretation, whether humorous, implicit, or socially coded.

Political or Identity Signal. The video communicates a political stance, ideological alignment, or identity-linked signal, including cues tied to collective affiliation, social positioning, or worldview.

Aggression or Hostility. The video conveys antagonism, threat, attack, intimidation, or overtly adversarial attitude toward a target.

Humiliation. The video frames a person or target as embarrassed, degraded, exposed, or socially diminished, often making loss of dignity central to the effect.

Other. The video communicates a social attitude or value signal not adequately captured by the categories above, or one that combines multiple signals without a clear dominant type.

### G Guided and Unguided Prompts for Structured Subtext Understanding Tasks

For the two structured subtext understanding tasks, namely rhetoric mechanism identification and social value signal identification, we evaluate models under two prompt settings: without guidance and with guidance. The unguided setting provides only the task question, transcript, options, and output rules, while the guided setting additionally provides taxonomy definitions for the five macro categories.

Without-Guidance Prompt You are answering a multi-choice question about a video. Task name: <task_name> Question: <question> Transcript from ASR (may be noisy, partial, or empty): <transcript> Instruction: <instruction> Available options: <option_text> Return only valid JSON with one field: { "selected_options": ["A", "B"] } Rules:

- - Select only options that are clearly supported by the video.
- - Return only option letters, not option texts.
- - Do not include any option not in the provided list.
- - Be conservative: do not select an option unless it is clearly justified by the video.
- - If no option is clearly supported, return an empty list.

With-Guidance Prompt

You are analyzing a video and answering a multi-choice question. Your goal is to identify the most appropriate categories based on the video’s meaning. Task name: <task_name> Question: <question> Transcript from ASR (may be noisy, partial, or empty): <transcript> <taxonomy_guidance> Instruction: <instruction> Available options: <option_text> Output format: Return ONLY valid JSON with the following structure: { "selected_options": ["A", "B"] } Rules:

- - Only return option letters (A–E).
- - Do NOT return option texts.
- - Select all options that are clearly supported by the video.
- - Do NOT guess if evidence is weak.
- - If none apply, return an empty list.

#### Rhetoric Guidance

Additional guidance on taxonomy categories: The options correspond to high-level rhetorical categories. Each category summarizes several finer-grained rhetorical patterns commonly observed in video memes. Rhetoric Macro Categories:

- A. Literal / Direct Meaning is conveyed directly without rhetorical transformation. Typical patterns include:

- literal_only: the video communicates its message directly without irony, exaggeration, or figurative framing.

- B. Opposition / Incongruity Meaning arises from contradiction, reversal, or unexpected juxtaposition. Typical patterns include:

- - contrast: juxtaposing two opposing situations, ideas, or outcomes.
- - bait_and_switch: setting up one expectation and then suddenly replacing it with a different or contradictory outcome.
- - role_reversal: reversing expected roles, identities, or positions to produce humor or commentary.
- - absurdism: presenting illogical or exaggerated situations that highlight incongruity.

- C. Attitude / Tone-based Rhetoric Meaning is conveyed primarily through tone or speaker attitude. Typical patterns include:

- - sarcasm: expressing a meaning by stating the opposite of what is intended.
- - irony: the intended meaning contrasts with the literal situation or appearance.
- - deadpan: presenting absurd or humorous content in a serious, emotionless manner.
- - mockery: ridiculing or making fun of a person, behavior, or situation.

- D. Amplification / Stylization Meaning is emphasized through exaggeration or stylized imitation. Typical patterns include:

- - exaggeration: overstating a situation or characteristic to emphasize its significance.
- - parody: imitating the style or conventions of a person, genre, or cultural artifact for humorous or critical effect.

- E. Implicit / Coded Social Framing Meaning is conveyed through indirect or socially coded signals. Typical patterns include:

- - innuendo: suggesting a meaning indirectly rather than stating it explicitly.
- - stereotype_invocation: referencing widely known stereotypes to imply a social meaning.
- - dog_whistle_or_code: using coded expressions that convey specific meanings to certain audiences while remaining subtle to others. When answering:
- - Focus on how the video constructs its underlying meaning.
- - Identify the rhetorical strategies used to convey humor, critique, or commentary.
- - A video may contain multiple rhetorical mechanisms.

Social Value Guidance

Additional guidance on taxonomy categories: The options correspond to high-level social value signals. Each category summarizes several finer-grained social attitudes or stances that may be expressed in video memes. Social Value Signal Categories:

- A. Neutral / No Social Signal The video conveys humor or content without expressing a clear social stance. Typical patterns include:

- none: the video does not convey a noticeable social judgment, stance, or value signal.

- B. Emotional Attitude The video expresses a general emotional tone or affect toward a situation. Typical patterns include:

- negative_affect: expressing frustration, disappointment, annoyance, or dissatisfaction.

+0.07 +0.08 +0.19 +0.10 -0.27

- -0.04 +0.43 -0.08 +0.09 -0.34
- -0.11 +0.40 -0.06 +0.01 -0.25

+0.08 +0.75 -0.20 -0.09 -0.36

- -0.38 +0.77 +0.10 +0.12 -0.02
- -0.29 +0.18 -0.27 -0.10 -0.38
- -0.01 +0.55 +0.01 -0.05 -0.32
- -0.22 +0.57 -0.15 +0.01 -0.33
- -0.05 +0.67 +0.00 +0.19 -0.33
- -0.07 +0.33 +0.08 +0.01 -0.21
- -0.09 +0.26 -0.18 -0.01 -0.36
- -0.38 +0.75 +0.03 +0.05 +0.22
- -0.38 +0.78 -0.11 -0.08 -0.02

+0.01 +0.29 -0.15 -0.07 -0.35

- -0.30 +0.50 +0.11 +0.06 -0.15
- -0.23 +0.38 -0.11 -0.05 -0.34
- -0.22 +0.58 -0.11 -0.02 -0.34

SEED

SEED

- -0.06 +0.17 -0.29 +0.01 -0.20

+0.14 +0.16 -0.34 -0.26 -0.38

+0.21 +0.11 -0.04 -0.23 -0.28

+0.04 -0.51 -0.46 -0.33 -0.41

+0.82 +0.06 -0.22 -0.26 -0.36

+0.24 -0.11 +0.33 +0.26 -0.27

+0.75 -0.17 -0.02 +0.57 -0.36

- -0.02 +0.14 -0.26 -0.03 -0.34

+0.21 -0.07 -0.17 -0.03 -0.34

- -0.07 +0.15 +0.36 -0.17 -0.08
- -0.07 +0.14 +0.31 -0.26 -0.26

+0.07 +0.08 -0.16 +0.01 -0.39

+0.01 -0.00 +0.33 -0.15 -0.28

- -0.05 +0.00 -0.12 -0.22 -0.35

Q3527B

Q3527B

QV32B

QV32B

O4M

O4M

MN8B

MN8B

MN14B

MN14B

MIMO

MIMO

GR41F

GR41F

G54M

G54M

G52

G52

G41N

G41N

GLM45V

GLM45V

GM34B

GM34B

GM327B

GM327B

GM3FP

GM3FP

+0.57 -0.11 -0.16 -0.20 -0.40

C3H

C3H

A B C D E

A B C D E

(a) Rhetoric Mechanisms Identification.

(b) Social Value Signal Identification.

Figure 12: Model–option affinity bias with guidance. Positive values indicate over-prediction relative to ground-truth prevalence, while negative values indicate under-prediction.

- - fatalism_or_cynicism: expressing pessimism, resignation, or cynical attitudes about situations or outcomes. C. Social Evaluation / Devaluation The video evaluates, criticizes, or demeans people or groups. Typical patterns include:
- - contempt: expressing disdain or disrespect toward someone or something.
- - humiliation: portraying someone as foolish, incompetent, or inferior.
- - aggression_or_hostility: showing hostility, threats, or aggressive attitudes.
- - exclusion: implying that certain people or groups should be excluded or marginalized.
- - discrimination_or_prejudice: expressing biased or discriminatory attitudes toward social groups.

- D. Norm and Value Framing The video comments on social rules, expectations, or cultural norms. Typical patterns include:

- - norm_violation: highlighting or mocking behavior that breaks accepted social rules.
- - anti_mainstream_value: expressing opposition to widely accepted social norms or values.

- E. Identity / Ideological Signaling The video references identity, ideology, or social group affiliation. Typical patterns include:

- - political_or_identity_signal: expressing political stances or identity-based perspectives.
- - sexual_implication: implying sexual themes or identity-related meanings.
- - other: conveying social signals that do not clearly fall into the above categories. When answering:
- - Focus on what social stance, value judgment, or attitude the video conveys.
- - Identify whether the video expresses opinions about people, groups, norms, or identities.
- - A video may express multiple social value signals.

### H The With-Guidance Counterpart of Affinity Bias

The with-guidance results related to Figure 9 is given in Figure 12. As can be observed, the results of these two figures show simiular pattern.

### I Details of Baselines and the Evaluation Process

For all tasks, model outputs are normalized into percentage scores for consistent comparison across different evaluation settings.

For the open-ended (OE) task, we adopt an LLM-as-a-judge evaluation protocol (as described in Appendix B). Each prediction is compared against the reference answer using a structured rubric that evaluates whether the model captures the core intended meaning, key implicit signals, and relevant social interpretation, while penalizing hallucination and purely literal responses. The final score is aggregated into a continuous value in [0,1] and then scaled to a percentage.

For the multiple-choice tasks (including evidence grounding, rhetoric mechanism identification, and social value signal identification), we adopt a strict yet interpretable set-based scoring rule. Let P denote the set of predicted options and G denote the ground-truth set. If the prediction contains any

incorrect option (i.e., P \ G ̸= ∅), the score is assigned as 0. Otherwise, the score is computed as the proportion of correctly selected options:

score = |P ∩ G|

.

|G|

This design ensures that models are penalized for hallucinated selections, while still receiving partial credit when they correctly identify a subset of the required evidence or categories. A full score is only obtained when all and only the correct options are selected.

Finally, task-level scores are averaged across all samples, and overall performance is reported as the mean across tasks.

### J Limitations

Despite its broad coverage of subtext understanding, ViMU has several limitations. First, the interpretation of metaphorical and socially grounded meaning is inherently subjective, and although we employ structured annotation and validation procedures, residual ambiguity and annotator bias may remain. While ViMU is designed for evaluation rather than training, models may still exploit superficial patterns or dataset-specific regularities, and strong performance on this benchmark does not necessarily imply robust real-world understanding of nuanced social or cultural meaning. Overall, these limitations reflect broader challenges in constructing benchmarks for subjective and socially situated understanding, rather than weaknesses unique to ViMU.

### K Societal Impact

ViMU aims to advance the evaluation of multimodal models by focusing on their ability to interpret implicit, socially grounded meanings in videos. A positive impact of this work is that it helps expose systematic limitations of current models in understanding rhetoric, social signals, and culturally situated subtext, which are critical for safe and reliable deployment in real-world applications such as content moderation, assistive technologies, and human–AI interaction.

However, the dataset also involves potential risks. Because it includes socially sensitive and potentially offensive content, there is a possibility that models evaluated on ViMU may reproduce or amplify harmful stereotypes, biases, or misinterpretations. In addition, improved capability in interpreting implicit meaning could be misused for profiling, surveillance, or manipulation of user intent, especially in contexts involving political or identity-related signals. There is also a risk that benchmark performance may be overinterpreted as a proxy for real-world social understanding, despite the inherent subjectivity and cultural dependency of such tasks.

We want to emphasize that ViMU is intended solely as an evaluation benchmark rather than a training resource, and we encourage users to carefully consider the dataset’s limitations, report model behaviors transparently, and avoid deploying systems based solely on benchmark performance. Future work should further investigate fairness, cultural coverage, and robustness to ensure that advances in subtext understanding benefit diverse user groups without reinforcing existing harms.

