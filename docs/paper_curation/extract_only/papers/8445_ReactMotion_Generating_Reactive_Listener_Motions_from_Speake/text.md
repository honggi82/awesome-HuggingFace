## ReactMotion: Generating Reactive Listener Motions from Speaker Utterance

Cheng Luo1†, Bizhu Wu2,4,5†, Bing Li1∗, Jianfeng Ren4, Ruibin Bai4, Rong Qu5, Linlin Shen2,3∗, and Bernard Ghanem1

1 King Abdullah University of Science and Technology

# arXiv:2603.15083v1[cs.CV]16Mar2026

- 2 Computer Vision Institute, School of Artificial Intelligence, Shenzhen University
- 3 Guangdong Provincial Key Laboratory of Intelligent Information Processing, Shenzhen University
- 4 School of Computer Science, University of Nottingham Ningbo China
- 5 School of Computer Science, University of Nottingham, United Kingdom Project page: https://reactmotion.github.io

Abstract. In this paper, we introduce a new task, Reactive Listener Motion Generation from Speaker Utterance, which aims to generate naturalistic listener body motions that appropriately respond to a speaker’s utterance. However, modeling such nonverbal listener behaviors remains underexplored and challenging due to the inherently non-deterministic nature of human reactions. To facilitate this task, we present ReactMotionNet, a large-scale dataset that pairs speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness. This dataset design explicitly captures the one-to-many nature of listener behavior and provides supervision beyond a single groundtruth motion. Building on this dataset design, we develop preferenceoriented evaluation protocols tailored to evaluate reactive appropriateness, where conventional motion metrics focusing on input–motion alignment ignore. We further propose ReactMotion, a unified generative framework that jointly models text, audio, emotion, and motion, and is trained with preference-based objectives to encourage both appropriate and diverse listener responses. Extensive experiments show that ReactMotion outperforms retrieval baselines and cascaded LLM-based pipelines, generating more natural, diverse, and appropriate listener motions.

Keywords: Dyadic interaction · Interactional AI systems

### 1 Introduction

Modeling dyadic human communication is crucial for virtual agents [33], digital humans [50, 104], and social robots [71]. While prior work has advanced speech-to-speech dialogue [15], language-based interfaces [1,28], and listener facial reactions [57,70], reactive listener body motions remain largely overlooked

† Equal contribution. ∗ Corresponding authors.

[Figure 1]

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

###### 2 C. Luo et al

I'm so excited you're here! I was hoping you'd show up.

[Figure 14]

React Motion

[Figure 15]

Emotion:

Ecstatic

[Figure 16]

Moving and greeting Waving hands

Input: speaker utterance Output: reactive listener motion

- Fig. 1: Illustration of the proposed new task: Reactive Listener Motion Generation from Speech Utterance. Given a speaker’s utterance, i.e., transcript and/or audio (optionally supplemented with emotion), a generative model such as our ReactMotion generates a corresponding responsive body-motion sequence for the listener.

despite being central to face-to-face interaction. Listeners often convey engagement and understanding through posture and subtle gestures, and generating such feedback is important for natural dyadic communication.

We introduce a new task, Reactive Listener Motion Generation from Speech Utterance, which aims to generate naturalistic listener body motions that appropriately respond to a speaker’s utterance given its audio and/or transcript. Unlike text-to-motion [21,62,75,76,96] or audio-driven motion generation [88] that primarily realize the input content, our setting models conversational reactions where speaker cues are indirect and the output is inherently one-to-many.

This task poses three challenges. (i) The same utterance can elicit multiple valid listener reactions [57,70]. Such non-deterministic listener behaviour poses a significant challenge for modeling the listener’s motion responses. (ii) There is no publicly available large-scale dataset with multiple listener-reactive body motions per utterance, to the best of our knowledge. (iii) Reactive appropriateness is difficult to evaluate. Metrics based on a single ground truth or motion diversity are insufficient to measure the appropriateness of a listener’s reaction.

To address these challenges, we introduce ReactMotionNet, a curated dataset with 151,328 (speaker utterance, listener motion) pairs. Unlike prior motion datasets that typically provide a single target per condition, we associate each utterance with multiple candidate reactions and annotate them into three preference tiers, Gold, Silver, and Negative. This tiered design captures one-to-many ambiguity and enables preference-style supervision and evaluation [11,13,102]. Moreover, we propose a scalable pipeline that re-purposes existing motion data into dyadic speaker-listener pairs for dataset construction, which avoids relying on expensive speaker–listener motion capture.

To evaluate reactive appropriateness, we introduce a tier-aware ranking protocol. We train a multimodal judge network to score and rank candidate reactions under the same speaker input and report win rates against the Gold, Silver, or Negative tiers. This relative evaluation goes beyond singlereference similarity and better reflects that multiple reactions can be appropriate for the same utterance. Finally, we propose ReactMotion, a unified generative framework that jointly models speaker transcript, emotion, and audio

to generate listener motions. We leverage the tiered annotations with preferencebased objectives that learn from relative comparisons within each utterance group for the training.

Contributions. (i) To the best of our knowledge, we introduce the first task of reactive listener body motion generation from speaker speech in dyadic interaction. (ii) We present ReactMotionNet, a new dataset with multi-tier (Gold/Silver/Negative) reactive listener motions and a tier-aware evaluation protocol for reactive appropriateness, enabling research on nonverbal listener response behavior. (iii) We propose ReactMotion, a unified multimodal generative model that processes multiple speaker cues and generates high-quality listener body motions in response to the speaker.

### 2 Related Work

Human Motion Generation. Human motion generation can be conditioned on diverse modalities, including text [8, 30, 42, 48, 52, 63, 78, 84, 95, 98], action classes [60,64,74], and audio signals such as music [37,39,40,90] or speech [38, 45,85]). Among these, text- and audio-driven motion generation are most related to our setting. Text-based approaches generate motions from explicit action descriptions [4,10,18,26,34,61,79,80,97,101,105], while audio-driven methods synthesize gestures aligned with temporally synchronized acoustic signals [7,53,99]. Representative modeling paradigms include transformer-based latent models (e.g., [43, 60, 100]), discrete motion tokenization with autoregressive modeling (e.g., [3,9,91,96]), and diffusion-based frameworks (e.g., [2,22,44,76]).

Beyond single-person generation, recent works [24, 41, 53, 55, 73, 81] extend motion synthesis to multi-person scenarios. These approaches typically generate multi-person motions by conditioning on explicit textual descriptions of joint actions or on the audio streams of both individuals. In contrast, our problem setting differs in that the target motion is not directly specified by explicit action instructions or synchronized signals. Instead, the model must infer the implicit interaction intention from the speaker’s utterance, including transcript, audio, and emotion cues, and produce a socially appropriate reactive motion for the listener. This requires reasoning over cross-speaker dynamics rather than direct condition-to-motion mapping.

Human Reaction Generation. Human reaction generation is crucial for AI interaction systems. Spoken language modeling has progressed from cascaded ASR → LLM → TTS pipelines to end-to-end and full-duplex speech-to-speech models [15,66,77,94], while facial reaction generation has advanced from conditional GANs [27] to uncertainty-aware and diffusion-based methods [49,51,57,70,103]. Audio-visual face-to-face dialogue modeling has been explored [14,57,59,103].

In 3D human body modeling, most methods synthesize reactor motion conditioned on actor motion [12,17,46,47,87]. For instance, InterFormer [12] uses temporal-spatial attention in Transformers, and ReGenNet [87] and ReMoS [17] employ diffusion models for full-body motion. Recently, HERO [93] generates

- 3D reactive motion directly from RGB videos, incorporating the actor’s facial expressions to capture emotional cues. Differently, our method generates 3D reactor motion from the speaker’s utterance, which includes transcript, audio, and optional emotion annotations. Transcript provides a lightweight, user-friendly modality, audio offers rich vocal cues, and emotion labels explicitly indicate mood, facilitating more effective interaction modeling.

- 3D Human Body Interaction Datasets. Recent datasets have facilitated research on multi-person dynamics and interaction-aware 3D motion. Several works [20,25,41,86,92] provide paired human motions, modeling interaction as symmetric kinematic coupling, where one participant’s motion is predicted from the other’s. While effective for spatial coordination, this ignores linguistic and affective signals that drive conversation.

Other datasets [31, 32, 35, 56, 67, 68, 93] supply silent RGB videos with 3D reactive motions, offering richer context but still lacking speech semantics and emotional cues, which are central to communicative intent. Some datasets [24,36, 55,73] include both audio and motion for human interactions, but the movements of their motions primarily focus on the upper body, such as arms, and are limited to one-to-one speaker-listener pairs.

In contrast, our dataset provides a one-to-many mapping between speaker utterances and listener reactive motions. Each utterance has multiple responses labeled gold, silver, and neg for appropriate, partially appropriate, and irrelevant reactions, making it better suited for practical applications. Plus, motions are more dynamic, such as jumping, enabling more diverse body reactions.

### 3 Task Definition

In this paper, we study Reactive Listener Motion Generation in dyadic interaction, which consists of a speaker and a listener. Given a speaker utterance Cs, the goal is to generate appropriate reactive body motion of the listener, denoted as Rl. Formally, the objective is to learn the conditional distribution:

pθ Rl | Cs , Cs ∈ As, Ts, (As,Ts), (As,Es), (Ts,Es), (As,Ts,Es) .

(1) Here, As denotes the speaker audio, Ts is the corresponding textual transcript, Es represents the speaker emotion, and θ denotes the model parameters. As shown in Eqn. 1, Cs may consist of different modalities of the speaker utterance or their combinations. At inference time, diverse listener reactions can be sampled from pθ(Rl | Cs).

In contrast to conventional text-to-motion generation, the speaker utterance do not explicitly specify the target listener motion. The mapping from Cs to Rl is therefore inherently one-to-many, which requires the model to generate motions that are contextually appropriate while maintaining diversity.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Step4: Speaker–Listener Candidate Ranking and Preference Tiering

Step1: Dyadic Listener Reactive Motion Curation Step2: -Condition Inference

Step2: Inverse Speaker-Condition Synthesis

Sample caption-motion pairs from HumanML3D dataset

Speaker utterance LLM Inference

Speaker utterance

Motion caption

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

A bunch of your old schoolmates just arrived, and they're all looking this way

A bee seemed to zip past you just now. That gave me a tiny scare

[Figure 30]

✖ ✔

[Figure 31]

TTS

A person is waving hi with his right hand.

[Figure 32]

Synthetic speaker audio

Gold

[Figure 33]

Someone abruptly steps backward, seemingly surprised or startled by something

[Figure 34]

[Figure 35]

Speaker emotion

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Content

[Figure 40]

[Figure 41]

[Figure 42]

LLMs

A person punches with their right hand before they do a counterclockwise spin.

[Figure 43]

[Figure 44]

[Figure 45]

A person is waving hi with his right hand.

Select dyadic conversation related motion

[Figure 46]

LLMs

[Figure 47]

Silver

[Figure 48]

Step3: Data Filtering

[Figure 49]

✖

[Figure 50]

✔

+

[Figure 51]

[Figure 52]

[Figure 53]

###### Step3: Data Filtration

Natural language inference models

A person standing still then suddenly stepping back out of the way

Emotion consistency check (keep/discard)

[Figure 54]

[Figure 55]

[Figure 56]

Speech emotion

Audio emotion

Target speaker emotion

recognition Hume AI

[Figure 57]

Synthetic speaker audio

[Figure 58]

[Figure 59]

Speaker emotion

Speaker transcript

Negative

Rank listener candidates by appropriateness for each speaker utterance, and retain the top-ranked ones.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Listener motion candidates

the person is moving his arms like he is arguing with someone.

All listener motion captions

Person is doing a hand stand.

The man plays the violin

###### Fig. 2: ReactMotionNet dataset construction. We curate dyadic listener motions

- (Step 1), synthesize speaker conditions via inverse inference and Text-to-Speech (TTS)
- (Step 2), filter unreliable samples (Step 3), and rank/re-tier speaker–listener pairs into gold/silver/negative preferences (Step 4).

### 4 ReactMotionNet Dataset

To bridge the gap between existing 3D human motion interaction datasets and real-world conversational dynamics, we construct a dataset, ReactMotionNet, featuring one-to-many speaker utterance–listener reaction mappings with graded appropriateness annotations. To construct this dataset, we present a novel data construction pipeline (Fig. 2) that repurposes existing human motion data into speaker–listener motion–response pairs using powerful LLMs [58, 89], thereby avoiding costly data collection.

#### 4.1 Dataset Construction Pipeline

- Step 1: Dyadic Listener Reactive Motion Curation. Unlike existing audio-driven 3D human interaction datasets, which mainly focus on upper-body movements while standing still, we curate motions from the more dynamic and commonly used HumanML3D dataset [19]. Leveraging the textual captions of motions, we filter out conversation-irrelevant ones (e.g., doing a handstand) using multiple LLM-based verifiers (e.g., ChatGPT-o1 [29], ChatGPT-o3 mini [58]). This step results in a set of motions with reaction-like semantics, which serve as the listener’s reactive motions.
- Step 2: Inverse Speaker-Condition Synthesis. For each listener motion Rl from the last step, we infer multiple plausible speaker utterances that could elicit the observed reaction. Concretely, we input the listener motion’s caption into OpenAI o3-mini [1,58,69] to generate potential speaker transcripts Ts and associated emotion labels Es. We incorporate emotion into utterance generation, as the speaker’s emotional state influences the listener’s reaction. For example, the same transcript, “Do whatever you want,” can lead to different responses: a

- Table 1: Dataset statistics. #Pairs is the total number of labeled speaker–listener pairs (i.e. candidate reactions). #Trans., #Audio, and #Emo. denote the numbers of unique transcripts, audio files, and emotion categories, respectively. #Motion is the number of unique motion sequences. #Motion/Utter. reports the average number of candidate motions per speaker utterance. Label counts report the numbers of gold/silver/negative candidates (#G/#S/#N).

Speaker Utterance Listener Reaction #Motion/Utter. Labels (y) #Trans. #Audio #Emo. #Motion (avg.) (#G/#S/#N)

Split #Pairs

Train 137,879 6,631 6,631 46 1,822 20.79 7,527 / 30,862 / 99,490

Val 6,790 841 841 40 195 8.07 903 / 1,682 / 4,205 Test 6,659 826 826 39 197 8.06 877 / 1,652 / 4,130

All 151,328 8,298 8,298 47 2,029 18.24 9,307 / 34,196 / 107,825

supportive tone may cause the listener to jump happily in place, whereas a frustrated tone may cause the listener to walk away feeling hurt. Given Ts and Es, we synthesize the corresponding speaker audio As using GPT-4o mini TTS [28]. These steps produce a pool of possible speaker utterances (As, Ts, Es).

- Step 3: Data Filtering. We perform a series of procedures to ensure the dataset quality. First, for each speaker utterance, we verify whether the synthesized audio As faithfully reflects the intended emotion Es. Specifically, we apply an automatic speech emotion recognizer (i.e., Hume AI 6) to the generated audio and discard any utterance whose predicted emotion is inconsistent with its assigned emotion label. Next, we pair each remaining speaker utterance with the caption of every listener reactive motion Rl obtained in Step 1. We then employ Qwen (Qwen3-235B-A22B-Instruct) [89] to assign a dyadic conversation appropriateness score to each speaker-utterance and listener motion caption pair. For each speaker utterance, we retain only the top several higher-scoring listener reactive motions, thereby removing inappropriate pairs.
- Step 4: Speaker–Listener Candidate Ranking and Preference Tiering. Given a pair consisting of a speaker utterance and one of its corresponding listener reactive motions from Step 3, we use multiple agents (i.e., ChatGPT-o1 [29], ChatGPT-o3 mini [58], and Qwen3-235B-A22B-Instruct [89]) to evaluate the pair. They score it according to (1) semantic appropriateness (whether the reaction fits the utterance), and (2) conversational plausibility (whether it sounds like a natural dyadic response). We further use a natural language inference (NLI) model7 to verify whether the listener motion caption is a logically plausible inference from the speaker utterance. We then weighted sum the agents’ scores to obtain a final score, which is used to label the pair as gold, silver, or negative according to predefined thresholds.

- 6 https://www.hume.ai/expression-measurement
- 7 https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v1.1all-33

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

###### Abbreviated paper title 7

Listener motion sequence

[Figure 76]

T5-Encoder

[Figure 77]

Unified vocabulary ! = !! ∪ !" ∪ !#

Motion VQ-VAE Decoder Shared T5-Token Embedding

<Audio Token i>… <Motion Token i>…

[Figure 78]

[Figure 79]

MiMi Neural Audio Codec

T5-Tokenizer

[Figure 80]

T5-Decoder

I'm so pumped to try that massive ride. It's much bigger than I imagined!

[Figure 81]

Crossattention

Excited

Speaker audio

Speaker emotion

Autoregressive generation over unified vocabulary ! (motion-only output !#)

Speaker transcript

- Fig. 3: Overview of the ReactMotion framework. We use modality-specific tokenizers to convert raw data, i.e., the speaker’s utterances (including transcript, audio, and emotion) and the listener’s reactive motions, into discrete special tokens. With these tokenizers, a Seq2Seq model is employed to integrate information across modalities and learns to generate the listener’s reactive motions from the speaker’s utterances.

#### 4.2 Dataset Statistics

In total, our dataset contains 151,328 labeled (speaker utterance, listener reactive motion) pairs, covering 8,298 unique speaker utterances and 2,029 unique listener reactive motions. On average, each speaker’s utterance is paired with 18.24 candidate reactive motions, highlighting the one-to-many nature of listener reactions. Overall, 9,307, 34,196, and 107,825 pairs are labeled as Gold, Silver, and Negative, respectively, reflecting graded appropriateness of candidate reactions. We split the dataset by speaker utterance with an 8:1:1 ratio for train/val/test, such that speaker utterances are disjoint across splits (i.e., no utterance appears in more than one split). Tab. 1 lists detailed statistics. Our automated construction pipeline further enables straightforward scaling to larger datasets.

### 5 Methodology

We present ReactMotion, a unified framework for Reactive Listener Motion Generation from Speaker Utterance. As illustrated in Fig. 3, we first introduce modality-specific tokenizers that convert raw inputs, i.e., the speaker utterance (including transcript, audio, and emotion) and the listener’s reactive motions, into discrete special tokens. With these tokenizers, we employ a Seq2Seq model to unify information across modalities and learn the conditional distribution of the task (Eqn. 1). To capture the one-to-many nature of dyadic interactions, we further train the model with a group-wise preference-based learning objective,

which explicitly allows the generation of multiple appropriate reactions for the same speaker utterance.

#### 5.1 Modality-Specific Tokenization

We employ modality-specific tokenizers to convert raw data from different modalities into discrete tokens.

Audio Tokenization. We use Moshi [15] (its Neural Audio Codec MiMi) to convert the audio waveform in the speaker utterance As into discrete codes. Specifically, its audio encoder Eaud(·) is employed to extract audio features from As, which are then quantized using the base codebook Caud.

hsa = Eaud(As), xsa = Qaud(hsa), (2)

where quantizer Qaud(·) maps the features to their nearest entries in the codebook Caud, and outputs the corresponding codebook indices xsa. The resulting indices are treated as discrete audio tokens, allowing the unified model to incorporate audio information while retaining prosody and paralinguistic cues that are informative for reactive behaviors.

Motion Tokenization. We represent the listener’s reactive motion Rl as discrete tokens with [96], similar to the audio tokenization process:

hlm = Emot(Rl), xlm = Qmot(hlm). (3)

where Emot and Qmot are the motion encoder and quantizer, respectively, and xlm are discrete indices of motion codebook Cmot.

Also, the predicted listener reactive motion in the form of discrete tokens from the unified model can be mapped back to the raw motion data through:

hlm = Q−mot1 (xlm), Rl = Dmot(hlm), (4)

where Q−mot1 (·) maps the discrete token indices to the vectors in the codebook, and a VQ-VAE motion decoder [82,96] Dmot(·) decodes the vectors back to the raw motion data.

#### 5.2 Unified Seq2Seq Modeling

With above modality-specific tokenizers, we can now represent information across modalities into a unified space, and thus enable a Seq2Seq model to generate a listener reactive motion conditioned on the speaker utterance.

Specifically, we adopt T5-base [65] as the Seq2Seq backbone and extend its original textual vocabulary Vt to include audio and motion vocabulary:

V = Vt ∪ Vm ∪ Va ∪ Vs, (5)

where Vm are the code indices of the motion codebook Cmot, represented as {<Motion Token i>}|VC

mot|−1

i=0 , and Va are the code indices of audio codebook Caud, represented as {<Audio Token i>}|VC

aud|−1

i=0 , respectively. Vs contains special tokens such as <Motion Tokens>, </Motion Tokens>, <Audio Tokens>, </Audio Tokens>, <Emotion> and </Emotion>, which wrap the motion, audio, and emotion token sequences.

This unified vocabulary allows us to formulate reactive listener motion generation, conditioned on different modalities or their combinations Cs, in a general format and achieve them within a single model. Specifically, we first fit discrete codes of the speaker utterance Cs and the listen reactive motion Rl into fixed prompt templates. Due to page limit, a coarse example task template of using only speaker audio as the condition is shown; detailed one and templates for other conditions are provided in the Appendix A.2.

Input: You are modeling a speaker-listener dyadic interaction. Given SPEAKER_AUDIO: [Audio Tokens Placeholder], return ONLY a sequence of listener reactive motion tokens. Output: [Motion Tokens Placeholder]

Now, the modeling process of generating listener reactive motion can be represented as an auto-regressive one, where each motion token is generated with probability pθ xoutt | xin(Cs),xout<t . Here, xin(Cs) are the input token sequences of the task template embedding with input speaker utterance Cs, and xout are the output token sequences, i.e., listener reactive motion xlm.

#### 5.3 Group-wise Preference Learning

A single speaker utterance Cs can correspond to multiple plausible listener reactive motions Rl. Directly fine-tuning on such one-to-many pairs may lead the model to collapse to averaged and safe behaviors, e.g., standing still. To mitigate this issue, we train the model using group-wise preference learning.

For each speaker utterance Cs, we randomly sample its corresponding listener motions from each label to construct a group {G,S,N}, where G, S, and N denote the sets of motions labeled as Gold, Silver, and Negative, respectively. Each motion Rl in the set is represented as a motion token sequence xlm. We compute the predicted score for each motion using the length-normalized conditional log-likelihood [5,54,83]:

1 |xlm|

ℓ(xlm | Cs) =

|xlm|

log pθ xlm,t | xin(Cs),xlm,<t . (6)

t=1

We then aggregate the predicted scores of motions with the same label using a smooth log-mean-exp operator:

 , A ∈ {G,S,N}. (7)

  1

exp ℓ(xlm | Cs)

ℓA(Cs) = log

|A(Cs)|

xl2m∈A(Cs)

This yields three predicted scores for Cs, namely ℓG, ℓS, and ℓN corresponding to the Gold, Silver, and Negative sets.

Since Gold motions are preferred over Silver, and Silver over Negative, the model is encouraged to produce ℓG > ℓS > ℓN. We enforce this ordering with a soft-margin ranking loss:

Lrank = log 1 + exp m − (ℓG − ℓS) + log 1 + exp m − (ℓS − ℓN)

(8)

+ λgn log 1 + exp m − (ℓG − ℓN) ,

where m specifies the margin between different labels, and λgn controls the strength of the Gold≻Negative constraint.

Training objective with frequency reweighting. To mitigate the dominance of frequently occurring motion sequences, we apply inverse-frequency weighting based on motion sequence IDs. Let i index a group (corresponding to one speaker utterance) and let rij denote the motion sequence ID of the j-th candidate in group i. We compute freq(r) as the number of times motion ID r appears in the training set and assign an item weight w˜ij = √ 1

. We then define the group weight as the mean item weight within the group, wi = |C1

freq(rij)

i| j w˜ij, where Ci denotes the candidate set of group i. Finally, we maximize the aggregated Gold score while applying the ranking loss:

wi − ℓ(Gi) + λrankL(ranki) i wi

L = i

. (9)

### 6 Experiments

#### 6.1 Implementation Details

We train ReactMotion for 100,000 iterations using the default AdamW optimizer and a cosine learning-rate schedule. The learning rate is set to 2×10−5 with 1,000 warmup steps. We use a per-device batch size of 8 with gradient accumulation of 2 steps on a single NVIDIA A100 GPU. We train with six conditioning variants (T, A, T+A, T+E, A+E, T+A+E) and apply modality dropout (p=0.3) to improve robustness (see the Appendix A for more details of the implementation).

#### 6.2 Evaluation Protocol

Evaluation metrics. (i) Reactive appropriateness, i.e., how well the generated reactive human motions respond to the speaker’s input, is a core objective of our task. Inspired by preference-based evaluation paradigms [6,11,13,16,72,102], we evaluate reactive appropriateness using group-level win rates Win(g>G), Win(g>S), and Win(g>N). Specifically, we compare the best generated sample g with annotated listener motions labeled as Gold (G), Silver (S), and Negative (N), and compute the win rate against each reference tier. A win against a

- Table 2: Multi-modal judge network reliability under strict modality missingness (Strict-L2). We evaluate six input modes (text T, audio A, emotion E, and their fusions) on the test set, reporting pairwise win rates (Win(G>N), Win(G>S), Win(S>N)) and ranking metrics (MRR(G), nDCG@K) with graded relevance G>S>N.

Mode Win(G>N) ↑ Win(G>S) ↑ Win(S>N) ↑ MRR(G) ↑ nDCG@3 ↑ nDCG@5 ↑ nDCG@10 ↑

T 0.992 0.873 0.983 0.829 0.864 0.878 0.932 A 0.992 0.872 0.983 0.832 0.866 0.878 0.933 T+E 0.993 0.876 0.982 0.826 0.857 0.876 0.929 A+E 0.992 0.874 0.983 0.831 0.865 0.878 0.933 T+A 0.993 0.879 0.982 0.820 0.855 0.875 0.928 T+A+E 0.993 0.878 0.982 0.828 0.859 0.878 0.930

higher reference tier (e.g., Silver) indicates that the generated motion is ranked above a higher-quality annotated response, reflecting stronger reactive appropriateness. To realize this evaluation, we train a multimodal judge network to rank generated reactive body motions conditioned on the same speaker input. Details of the judge network are provided in the appendix. We also report Gen@3, the fraction of groups where a generated candidate is ranked within the top-3 among {G,S,N} plus generated candidates under the same group. (ii) Motion quality is measured by Fréchet Inception Distance (FID) [23] computed in a motion feature space, and (iii) Diversity is measured as the average pairwise embedding distance across generated samples, following human motion generation [82,96]. (see the Appendix B.4 for more details of the evaluation metrics).

Validation of the multimodal judge network. Since the judge network is central to measuring reactive appropriateness, we validate it on samples with tiered appropriateness annotations (G/S/N). Specifically, we compute the tierconsistency win rates Win(G>S), Win(G>N), and Win(S>N) to test whether the judge assigns higher scores to more appropriate reactions. Higher values indicate a more reliable judge. We further report MRR(G), which measures how highly the Gold reaction is ranked, and nDCG@3/nDCG@5/nDCG@10 to assess graded ranking quality among the top-K candidates.

Table 2 shows the judge consistently preserves the expected preference ordering with near-perfect separation, across all six modes and both Test set. Gold almost always beats negatives (Win(G>N) ≈ 0.99) and silver also strongly beats negatives (Win(S>N) ≈ 0.98), indicating that the judge reliably distinguishes poor motions from plausible ones. Meanwhile, gold beats silver with a clear margin (Win(G>S) ≈ 0.87–0.88), reflecting sensitivity to fine-grained quality differences beyond simply rejecting negatives. The judge further achieves strong ranking quality (MRR(G) ≈ 0.82–0.84; nDCG@5 ≈ 0.87–0.88; nDCG@10 ≈ 0.93), demonstrating stable and meaningful top-K ordering.

Although our multimodal judge network is trained on multiple input modalities, i.e., text (T), audio (A), and emotion (E), it supports missing modalities using Strict-L2. Disabled modalities are replaced with information-free inputs (all-padding text, all-padding audio codes, or an unknown emotion token). This enables the judge network to operate with any subset of modalities; even with a

single modality, it performs well in evaluation. (see the Appendix B.1 and B.2 for more details of the judge network).

#### 6.3 Quantitative Results

Since reactive listener motion generation remains underexplored, we evaluate a set of representative baselines. (a) Random Selection uniformly samples a motion sequence from HumanML3D [19]. (b) Retrieval applies the text–motion matching network from prior HumanML3D T2M work [82,96] to compute text– motion similarity and retrieves the nearest-neighbor listener motion sequence from the training set given the speaker transcript. We also consider stronger cascaded LLM→T2M baselines: given a speaker utterance (and emotion), an LLM [89] first generates a listener-motion caption, which is then passed to a T2M generator to synthesize the final motion. We instantiate the LLM with Qwen3-30B-A3B (30.5B parameters) and a fine-tuned Qwen3-4B-Thinking (4B parameters) trained on our training-set (speaker utterance, listener-motion caption) pairs. The resulting captions are fed into two representative T2M generators, T2M-GPT [96] and MG-MotionLLM [82]. More details of baselines are in the Appendix B.3.

Tab. 3 shows that ReactMotion outperforms all baselines in reactive appropriateness. Among the cascaded LLM→T2M pipelines, LLM→MG-MotionLLM * is the strongest, improving over Random Selection and Retrieval. However, despite using a powerful motion generator, it still performs poorly under strict comparisons to Silver references (Win(g>S)), indicating that the two-stage captionthen-generate pipeline struggles to produce highly appropriate listener reactions.

In contrast, ReactMotion achieves near-perfect Win(g>N) across input modes and substantially improves Win(g>S) and Gen@3. Our full model (T+A+E) yields the best overall Win rates, while maintaining low FID and competitive diversity. Although Retrieval attains the highest diversity by construction, it yields much lower appropriateness and worse realism than our approach. More experimental results are provided in the Appendix D.

#### 6.4 Qualitative Results

We visualize representative examples in Fig. 4, comparing our ReactMotion (Ours), a cross-entropy trained variant (CE), and LLM→MG-MotionLLM * with a finetuned Qwen [89] (Qwen3-4B-Thinking) on training set, together with gold and silver reference reactions under the same speaker condition. Overall, ReactMotion produces reactive motions that are both semantically consistent with the speaker content and expressive in intensity. For instance, for the utterance “The energy in here feels electric right now” with excited emotion, our model generates larger, more dynamic upper-body and arm movements, which better reflect the high-energy “electric” cue and match the communicative style seen in the gold reaction.

In contrast, the silver reaction exhibits a rapid hand-wave but remains relatively low-energy, making it less aligned with the excited condition. The

- Table 3: Quantitative results on the test set. Main evaluation metrics are Win(g>N), Win(g>S), Win(g>G), Gen@3 measuring Reactive Appropriateness. We additionally evaluate motion quality (FID) and diversity. ∗ indicates that the LLM is fine-tuned using training-set speaker utterance and listener motion caption pairs.

Method Input Mod. Win(g>N)↑ Win(g>S)↑ Win(g>G)↑ Gen@3↑ FID↓ Diversity↑

GT - - - - - 0.278 6.187 Random Selection - 0.265 0.122 0.006 0.099 42.363 9.880

Retrieval T 0.392 0.252 0.130 0.206 7.429 8.207 LLM→T2M-GPT T+E 0.138 0.038 0.016 0.199 49.920 4.946 LLM→T2M-GPT * T+E 0.171 0.027 0.017 0.350 42.589 6.102 LLM→MG-MotionLLM T+E 0.775 0.245 0.044 0.345 23.629 5.082 LLM→MG-MotionLLM * T+E 0.883 0.274 0.047 0.380 25.723 4.546

ReactMotion (Ours) T 0.993 0.774 0.258 0.916 4.706 4.789 ReactMotion (Ours) A 0.992 0.614 0.164 0.864 6.221 4.009 ReactMotion (Ours) T+E 0.990 0.696 0.206 0.930 5.422 4.475 ReactMotion (Ours) A+E 0.993 0.736 0.323 0.981 6.485 4.162 ReactMotion (Ours) T+A 0.993 0.651 0.215 0.931 6.560 4.145 ReactMotion (Ours) T+A+E 1.000 0.797 0.266 0.960 4.760 4.804

CE variant tends to regress to generic, weakly-conditioned responses (e.g, a static pose such as crossing arms), indicating limited ability to exploit preference structure and model the one-to-many nature of reactive behaviors. Finally, the LLM→T2M baseline often generates repetitive motions (e.g, near-constant waving) with limited temporal variation, which appears less suitable for dyadic communication, where reactions typically evolve over time (e.g, hands rising and lowering, pose changes and subtle turns). Moreover, because dyadic reactions can be difficult to describe in natural language, the out-of-domain captions produced by the LLM may be noisy, which can lead MG-MotionLLM to produce degraded outputs, including overly short motion sequences.

#### 6.5 User Study

We recruit 59 volunteers and conduct a user study to evaluate the reactive appropriateness of listener motions generated by ReactMotion (Ours) against two baselines (the CE variant and LLM→MG-MotionLLM *) and the bestin-group Silver reference. In each case, participants watch two motion videos (A/B) conditioned on the same speaker utterance (audio with transcript shown) and select the more appropriate listener reaction. Each participant completes 36 cases covering six speaker conditions (six pairwise comparisons per condition).

As shown in Fig. 5, Ours is preferred over the generative baselines, achieving win rates of 67.8% against CE and 72.0% against LLM→MG-MotionLLM *. Ours is also competitive with the Silver reference, receiving 44.1% of votes in Silver vs. Ours, substantially higher than CE (31.9%) and LLM→MGMotionLLM (31.4%).

[Figure 82]

The energy in here feels electric right now. Emotion: Excited

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

Gold

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Silver

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

Ours

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

CE

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

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

LLM→MGMotionLLM

[Figure 152]

- Fig. 4: Qualitative results. We compare gold and silver listener reactions, motions generated by our ReactMotion (Ours), a cross-entropy trained variant (CE), and a cascaded LLM→T2M baseline, all conditioned on the same speaker utterance. We visualize the resulting 3D motion sequences.

#### 6.6 Ablation Studies

Modality study. We study the effect of input modalities in Tab. 3. Across settings, multimodal fusion performs best overall. Text is the strongest single cue, giving high alignment and the lowest single-modality FID (e.g., T: Win(g>N)=0.993, Win(g>S)=0.774, FID=4.706). Audio alone is weaker for fine-grained appropriateness, but adding emotion substantially improves it (best Win(g>G)=0.323 and Gen@3=0.981). Full fusion (T+A+E) is the most balanced, achieving the best Win(g>N)=1.000, strong Win(g>S)=0.797, and a low FID=4.760.

Ablations on group-wise preference learning. Tab. 4 ablates key components of our group-wise preference learning objective. Compared to training with crossentropy only, our full model substantially improves both reactive appropriateness and motion quality (e.g., Win(g>S): 0.741→0.797; Gen@3: 0.938→0.960; FID: 6.555→4.760). Removing inverse-frequency reweighting leads to the largest appropriateness drop, especially against the strongest tier (Win(g>G): 0.266→0.220), highlighting the importance of mitigating the dominance of frequent and generic motions. Removing the ranking loss degrades fidelity (FID: 4.760→5.950) while increasing diversity (4.804→5.453), suggesting that the ranking constraints help enforce correct relative ordering among tiers. Finally, re-

Silver

Ours

CE

LLM→MG-MotionLLM

| |
|---|

| |
|---|

| |
|---|

| |
|---|

100

| | |44.1%| | |31.9%| | |31.4%| | |32.2%| | |28.0%| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |72.0%| | |
| | | | | |68.1%| | |68.6%| | |67.8%| | | | | |
| | |55.9%| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

UserPreferenceRate(%)

75

50

25

0

Silver vs. Ours Silver vs. CE Silver vs. LLM→MGMotionLLM

Ours vs. CE Ours vs. LLM→MGMotionLLM

Fig. 5: User study on reactive appropriateness.

- Table 4: Ablation studies on the test split (all use A+T+E unless noted). w/o denotes training without the corresponding component. The CE baseline trains the same model using only a cross-entropy loss by pairing each speaker input with a single Gold reaction as supervision.

Method Win(g>N)↑ Win(g>S)↑ Win(g>G)↑ Gen@3↑ FID↓ Diversity↑

CE baseline 0.990 0.741 0.262 0.938 6.555 5.448 Ours (full) 1.000 0.797 0.266 0.960 4.760 4.804 w/o Inverse-frequency reweighting 0.979 0.704 0.220 0.946 5.177 4.929 w/o Lrank 0.996 0.781 0.260 0.960 5.950 5.453 w/o ℓG 0.996 0.712 0.215 0.943 6.376 4.493

moving ℓG consistently harms both appropriateness and quality, indicating that likelihood supervision on Gold reactions remains necessary.

### 7 Conclusion

We introduce Reactive Listener Motion Generation from Speaker Utterance, a new task for modeling listener motion responses in dyadic interactions. To support this task, we present ReactMotionNet, a multi-modal dataset that explicitly captures the inherent non-determinism of human behavior: for each speaker utterance, we provide multiple candidate listener motions with preference annotations, enabling supervision beyond a single “ground-truth” response. Building on this dataset design, we develop preference-oriented evaluation protocols tailored to reactive motion generation. Finally, we propose ReactMotion, a unified framework that processes multi-modal speaker cues, substantially outperforms strong baselines in motion quality and reactive appropriateness. We believe this work provides a foundation for future research on modeling dyadic interactions.

### Outline of the Supplementary Material

The supplementary material is organized as follows:

- • Section A presents the implementation details, including the model configuration, vocabulary construction, optimization settings, and training hyperparameters.
- • Section A.1 presents the model size of ReactMotion.
- • Section A.2: prompt templates for different speaker-condition settings;
- • Section B further provides the additional evaluation details, including:

- • Section B.1: the formulation of the multimodal judge network;
- • Section B.3: details of the baseline methods.

- • Section B.4 introduces the evaluation metrics, covering reactive appropriateness, motion quality, and diversity.
- • Section C provides additional statistics and analysis of the ReactMotionNet dataset.
- • Section D.1 presents the hyperparameter sensitivity analysis, including the full sweep results, representative configurations, and heatmap visualizations.
- • Section D.2 evaluates the inference efficiency of the proposed method.
- • Section D.3 reports the protocol and results of the user study.
- • Section D.4 shows representative failure cases.
- • Section E discusses the limitations of the current framework.

### A Implementation Details

Tab. 5 summarizes the key implementation details and training hyperparameters used in our experiments. Specifically, ReactMotion is instantiated with a T5-base Seq2Seq backbone, comprising 222.9M backbone parameters and 235.9M trainable parameters after extending the vocabulary. In accordance with the methodology section, the original textual vocabulary (|Vt| = 32,100) is augmented with motion tokens (|Vm| = 512), MiMi audio tokens (|Va| = 2,048 per codebook; 8 codebooks), and modality-specific special tokens that mark the boundaries of different modalities, resulting in a unified vocabulary of size 63,338. Notably, the vocabulary includes tokens from all 8 MiMi codebooks for completeness, while in practice we only use tokens from the base codebook during training to accelerate the process. The model takes tokenized speaker utterances as input and autoregressively predicts listener reactive motion tokens, with maximum source and target lengths set to 512 and 256, respectively. We train the model using AdamW with learning rate 2.0 × 10−5, β1 = 0.9, β2 = 0.999, weight decay 0.0, 1,000 warmup steps, per-device batch size 8, gradient accumulation over 2 steps, and 100,000 total optimization steps. To capture the one-to-many mapping from a speaker utterance to plausible listener reactions, training adopts the proposed group-wise preference objective with λrank = 0.25, λgn = 0.25, and margin m = 0.5. We further apply modality dropout with rate 0.3 to improve robustness to missing modalities, while length-normalized LogSumExp aggregation is used to obtain stable set-level scores during preference optimization.

Table 5: Implementation details and hyperparameters used in training.

Setup Value

Seq2Seq backbone model T5-base [65] Text tokenizer T5-base tokenizer [65] Audio tokenizer MiMi neural audio codec [15] Motion tokenizer VQ-VAE from T2M-GPT [96]

Per-device batch size 8 Gradient accumulation steps 2 Training steps 100,000 Warmup steps 1,000 Optimizer AdamW

- Adam β1 0.9
- Adam β2 0.999 Weight decay 0.0 Learning rate 2.0 × 10−5

Maximum source length 512 Maximum target length 256 Text vocabulary size |Vt| 32,100 Audio codebook size |Va| 2,048 Number of MiMi audio codebooks 8 Motion VQ-VAE codebook size |Vm| 512 Total vocabulary size |V | 49,002

Backbone parameters 222.9M Total trainable parameters after vocabulary expansion 235.9M

Ranking loss weight λrank 0.25 Gold-negative loss weight λgn 0.25 Ranking Margin m 0.5 Modality dropout rate 0.3 LogSumExp normalization Enabled

#### A.1 Model Size

Table 6: Model Configuration and Parameters of ReactMotion.

Metric Value Backbone parameters 222.9M Total trainable parameters 235.9M Unified vocabulary size 49,002

Table 6 summarizes the model size of ReactMotion. The model is built upon a T5-base backbone with 222.9M parameters and 235.9M trainable parameters after extending the vocabulary to incorporate multimodal tokens.

#### A.2 Prompt Templates

To support unified generation under different speaker-condition settings, we convert the available speaker cues into a fixed natural-language prompt template.

Given a speaker utterance consisting of transcription, audio, and optional emotion annotation, we construct the input prompt by selectively enabling the corresponding fields. The model is instructed to output only the listener motion-token sequence in a strict format, without any additional natural language.

Formally, for a speaker utterance Cs, the prompt is constructed as

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION: [Speaker Transcription]
- - SPEAKER_AUDIO: [Speaker Audio]
- - SPEAKER_EMOTION: <Emotion> [Speaker Emotion] </Emotion> Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

In practice, the fields in the prompt are enabled or disabled depending on the chosen condition mode. For example, when transcription is used but audio is not, the SPEAKER_AUDIO field is left empty; when emotion is disabled, the emotion line is omitted entirely. This design allows us to handle text-only, audio-only, text+audio, text+emotion, audio+emotion, and text+audio+emotion settings within a single unified framework.

Below we show several concrete examples. Text-only condition (T).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION: [Speaker Transcription]
- - SPEAKER_AUDIO: Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Text+Emotion condition (T+E).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION: [Speaker Transcription]
- - SPEAKER_AUDIO:
- - SPEAKER_EMOTION: <Emotion> [Speaker Emotion] </Emotion> Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Audio-only condition (A).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION:
- - SPEAKER_AUDIO: [Speaker Audio] Output: Return ONLY a sequence of listener motion tokens in the exact format:

<Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Audio+Emotion condition (A+E).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION:
- - SPEAKER_AUDIO: [Speaker Audio]
- - SPEAKER_EMOTION: <Emotion> [Speaker Emotion] </Emotion> Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Text+Audio condition (T+A).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION: [Speaker Transcription]
- - SPEAKER_AUDIO: [Speaker Audio] Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Text+Audio+Emotion condition (T+A+E).

Input: You are modeling a speaker-listener dyadic interaction. Input:

- - SPEAKER_TRANSCRIPTION: [Speaker Transcription]
- - SPEAKER_AUDIO: [Speaker Audio]
- - SPEAKER_EMOTION: <Emotion> [Speaker Emotion] </Emotion> Output: Return ONLY a sequence of listener motion tokens in the exact format: <Motion Tokens> <Motion Token i> ... </Motion Tokens> Do NOT output any other words.

Given the constructed prompt xin(Cs), the model auto-regressively predicts the listener motion-token sequence xout as

pθ xoutt | xin(Cs),xout<t .

Here, xin(Cs) denotes the prompt sequence instantiated from the speaker utterance Cs, and xout denotes the output listener motion-token sequence.

### B Additional Evaluation Details

#### B.1 Multimodal Judge Network

To evaluate the reactive appropriateness of generated listener motions and support best-of-K selection, we train a multimodal judge network, illustrated in

Fig. 6. Given a speaker utterance Cs and a candidate listener motion token sequence xlm, the judge network sψ outputs a scalar compatibility score

sψ(Cs,xlm) ∈ R, (10)

where a larger value indicates that the candidate listener motion is more appropriate for the given speaker utterance.

z#

|InfoNCE Loss<br><br>z# z$ z'<br><br>z!<br><br>z$ z$|
|---|

[Figure 153]

Attention Pooling

[Figure 154]

Fusion Transformer

[Figure 155]

###### Mode Encoding

z$

u!

u'

u"

[Figure 156]

Attention Pooling

h"!

E!%&"! Modality type

+

[Figure 157]

embedding

Motion VQ-VAE Decoder

Avg Pooling

Transformer Encoder

h"'

+ E!%&"' Modality type

| | | | | | | |
|---|---|---|---|---|---|---|
|ttention<br><br>| | |Pooling| | | |

embedding

[Figure 158]

z!

###### Avg Pooling

[Figure 159]

Positional Encoding

| | | | | | |
|---|---|---|---|---|---|
|ttention| | |Pooling| | |

A

z'

[Figure 160]

[Figure 161]

h""

+ E!%&"" Modality type

Learnable Motion Embedding

H!

A

embedding

[Figure 162]

z"

Linear Layer

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

H'

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Transformer Layer Linear Layer

T5-Encoder

Motion VQ-VAE Encoder

[Figure 178]

[Figure 179]

Learnable Audio Embedding

[Figure 180]

Learnable Emotion Embedding

T5-Tokenizer

[Figure 181]

MiMi Neural Audio Codec

I'm so pumped to try that massive ride. It's much bigger than I imagined!

Excited

Speaker emotion

[Figure 182]

Speaker transcript

Speaker audio

- Fig. 6: Architecture of the multimodal judge network. Given a speaker utterance and a candidate listener motion, the judge encodes transcript, MiMi audio tokens, and the discrete emotion label with three modality-specific branches, producing modality embeddings zt, za, and ze, as well as hidden summaries used to form fusion tokens ut, ua, and ue. Modality-type embeddings and a mode embedding are added to these tokens, which are then processed by a fusion transformer and attention

pooling to obtain the unified condition embedding zf. In parallel, the candidate listener motion, represented by VQ-VAE motion tokens, is encoded by a motion transformer and pooled into a motion embedding zm. The judge computes compatibility between the condition and motion embeddings in a shared normalized scoring space. During training, a group-wise InfoNCE objective is applied to the fused embedding and auxiliary modality-specific embeddings, enabling reliable scoring under both full and partial speaker utterances. Snowflake and flame icons denote frozen and trainable modules, respectively.

##### Architecture. It contains three branches to encode different modalities: transcript, audio, and emotion in the speaker utterance Cs, a fusion branch that in-

tegrates the available information in Cs while allowing missing modalities, and a motion branch to encode the reactive motion. All branches project their features from dimension d into a shared scoring space of dimension do. By default, all score-space embeddings are ℓ2-normalized.

Text branch. Let xst denote the tokenized speaker transcript, and let Mt ∈ {0,1}T

t denote its padding mask, where Mt(j) = 1 indicates that the j-th token is valid and Mt(j) = 0 indicates padding. We encode the transcript with a T5 encoder ET5(·) and project the resulting hidden states into the shared hidden space:

Ht = Wt ET5(xst) + bt, (11) where Ht ∈ RT

T5, and bt ∈ Rd are learnable parameters. We then aggregate the token-level features into a text embedding in the scoring space:

t×d, Wt ∈ Rd×d

z˜t = AttnPoolt(Ht;Mt), zt = L2Norm(˜zt), (12) where z˜t,zt ∈ Rd

o, AttnPoolt(·) denotes a masked attention-pooling operator that ignores padded positions according to Mt, and L2Norm denotes ℓ2 normalization.

Audio branch. Let xsa denote the speaker audio token sequence obtained from the MiMi neural codec tokenizer [15]. Because MiMi audio is represented by multiple codebooks, we first map the discrete tokens into embeddings, add learnable codebook-level embeddings and positional embeddings, and then process the resulting sequence with a transformer encoder:

Ha = Ea Emba(xsa) + Elvla + Eposa , (13) where Ha ∈ RT

a×d, Emba(·) denotes the learnable audio-token embedding layer, Elvla is the learnable codebook-level embedding, Eposa is the learnable positional embedding, and Ea is the audio transformer encoder. Let Ma ∈ {0,1}T

a denote the audio padding mask, where Ma(j) = 1 indicates a valid audio token and Ma(j) = 0 indicates padding. The token-level audio features are pooled into an audio embedding:

z˜a = AttnPoola(Ha;Ma), za = L2Norm(˜za), (14) where z˜a,za ∈ Rd

o.

Emotion branch. Let es denote the discrete speaker emotion label. We map it to a learnable embedding and project it into the shared scoring space:

he = LayerNorm Embe(es) , z˜e = Wehe + be, ze = L2Norm(˜ze), (15) where Embe(·) is the learnable emotion embedding table, he ∈ Rd, We ∈ Rd

o×d, be ∈ Rd

o, and z˜e,ze ∈ Rd

o.

Fusion branch. To unify all available information, we construct one fusion token for each modality in the d-dimensional hidden space. Let o ⊆ {t,a,e}

denote the active modality set, and let δk(o) ∈ {0,1} indicate whether modality k ∈ {t,a,e} is available under mode o.

For text and audio, we summarize the hidden states by masked mean pooling over valid positions:

 

 

Tt j=1 Mt(j)Ht(j)

Ta j=1 Ma(j)Ha(j)

, δt(o) = 1, 0, δt(o) = 0,

, δa(o) = 1, 0, δa(o) = 0,

h¯t =

h¯a =

Tt j=1 Mt(j)

Ta j=1 Ma(j)





(16) where Ht(j),Ha(j) ∈ Rd denote the j-th hidden states. For emotion, which is already represented by a single hidden vector, we define

h¯e =

he, δe(o) = 1, 0, δe(o) = 0.

(17)

We then form the modality-specific fusion tokens

ut = h¯t + Etypet , ua = h¯a + Etypea , ue = h¯e + Etypee , (18) where ut,ua,ue ∈ Rd, and Etypet , Etypea , and Etypee are learnable type embeddings.

To explicitly encode which modalities are active, we further introduce a learnable mode embedding Emode(o) ∈ Rd. The initial fusion-token sequence is

 

  ∈ R3×d. (19)

ut + Emode(o) ua + Emode(o) ue + Emode(o)

Xf(0) =

Since some modalities may be absent, we define a modality-presence mask

Mf = δt(o), δa(o), δe(o) ∈ {0,1}3. (20) The fusion sequence is processed by a transformer encoder with masking:

Hf = Ef Xf(0);Mf , z˜f = AttnPoolf(Hf;Mf), zf = L2Norm(˜zf),

(21) where Hf ∈ R3×d, Ef denotes the multimodal fusion transformer, and z˜f,zf ∈ Rd

o.

Motion branch. Each candidate listener motion is represented as a motion

token sequence xlm, obtained using a motion VQ-VAE tokenizer. We map the motion tokens to embeddings, add positional embeddings, and encode them with a motion transformer:

Hm = Em Embm(xlm)+Eposm , z˜m = AttnPoolm(Hm;Mm), zm = L2Norm(˜zm),

(22) where Hm ∈ RT

m is the motion padding mask, Embm(·) is the motion-token embedding layer, Eposm is the motion positional embedding, Em is the motion transformer encoder, and z˜m,zm ∈ Rd

m×d, Mm ∈ {0,1}T

o.

Compatibility scoring. Let ϕ(·,·) denote the embedding-space compatibility function. Given a condition embedding z ∈ Rd

o and a motion embedding zm ∈ Rd

o, we define

ϕ(z,zm) = α z⊤zm, α = exp(τ), (23)

where τ is a learnable temperature parameter and α > 0 is the corresponding scaling factor. Because all score-space embeddings are ℓ2-normalized, Eq. (23) is a scaled cosine similarity.

The fused compatibility score is defined as

sψ(Cs,xlm) = ϕ(zf,zm). (24) In addition, we compute auxiliary modality-specific compatibility scores

s(ψk)(Cs,xlm) = ϕ(zk,zm), k ∈ {t,a,e}, (25)

which allow the judge to score candidate motions under partial speaker utterances.

Group-wise contrastive training. For each speaker utterance Cis, we construct a candidate set

Ui = G(Cis) ∪ S(Cis) ∪ N(Cis), (26)

where G(Cis), S(Cis), and N(Cis) denote the Gold, Silver, and Negative listener motion sets, respectively. During training, we randomly sample a small number of candidates from each tier and encode them jointly.

To improve robustness to incomplete conditions, we randomly vary the active modality set o during training. This encourages the judge to remain reliable under different condition modes, including single-modality settings such as textonly and audio-only.

Let Pi ⊆ Ui denote the positive set associated with Cis; in our default setting, Pi = G(Cis). Given a condition embedding zi (which can be the fused embedding zf or an active modality-specific embedding zt,za,ze), we optimize the following group-wise InfoNCE objective:

##### exp ϕ(zi,zm(x))

1 |B| i∈B

log x∈Pi

Lcon(z) = −

,

exp ϕ(zi,zm(x)) +

exp β ϕ(zi,zm(b))

x∈Ui

b∈Bbank

(27) where B is the mini-batch, Bbank is an auxiliary motion bank providing additional generic negatives, zm(x) denotes the motion embedding of candidate x, zm(b) denotes the embedding of a motion sampled from the bank, and β controls the contribution of bank negatives. The motion bank discourages the judge from assigning overly high compatibility scores to generic or template-like motions.

We always apply Eq. (27) to the fused embedding zf. For the modalityspecific auxiliary losses, we apply it only to the modalities active under the

current mode o: Ljudge = λf Lcon(zf) +

λk Lcon(zk), k ∈ {t,a,e}, (28)

k∈o

where λf,λt,λa,λe are loss weights to balance different loss terms.

Validation of the multimodal judge network. Because the judge network is central to our evaluation protocol, we further verify whether its rankings respect the annotated tier ordering G ≻ S ≻ N. For any tier A ∈ {G,S,N}, we define its mean judge score under condition Cs as

1 |A(Cs)|

s¯A(Cs) =

sψ(Cs,x). (29)

x∈A(Cs)

We then report Win(G>S), Win(G>N), and Win(S>N), defined as Win(A > B) =

1 |D| Cs∈D

κ s ¯A(Cs),s¯B(Cs) , (30)

where (A,B) ∈ {(G,S),(G,N),(S,N)}, D denotes the evaluation set of speaker utterances, and

 

1, u > v, 0.5, u = v, 0, u < v.

(31)

κ(u,v) =



We further report MRR(G), defined as MRR(G) =

1 |D| Cs∈D

1 minx∈G(Cs) rankCs(x)

, (32)

where all candidates in U(Cs) are sorted in descending order of sψ(Cs,x), and rankCs(x) denotes the resulting 1-based rank of candidate x.

Finally, we report nDCG@3, nDCG@5, and nDCG@10, using graded relevance labels 2, 1, and 0 for Gold, Silver, and Negative candidates, respectively. These metrics verify whether the learned judge produces rankings aligned with the annotated appropriateness structure.

Strict-L2 missing-modality injection. For partial-condition evaluation, we adopt a Strict-L2 missing-modality injection protocol. Given an active modality set o ⊆ {t,a,e}, every unavailable modality is replaced by a null input before it is processed by its encoder branch. This differs from a weak masking strategy that removes a modality only during fusion while still allowing its encoder to observe the original input.

Formally, let δt(o), δa(o), and δe(o) indicate whether text, audio, and emotion are active under mode o, respectively. For text, if δt(o) = 0, we replace the transcript with an all-padding sequence and set its padding mask to zero:

xst ← PAD, Mt(j) = 0, ∀j. (33)

For audio, if δa(o) = 0, we replace all codec tokens with the audio padding index and mark all time steps as padded:

xsa ← PADa, Ma(j) = 0, ∀j. (34)

For emotion, if δe(o) = 0, we replace the original label with a dedicated unknown symbol:

es ← <unk>. (35) At the fusion stage, the corresponding modality token is additionally masked out through Mf.

As a result, unavailable modalities contribute no semantic information to the final condition representation. This protocol provides a strict test of whether the judge can reliably score listener motions using only the actually available speaker signals. Unless otherwise specified, all partial-condition reliability experiments are conducted under this Strict-L2 protocol.

#### B.2 Implementation Details of Judge Network

Table 7: Hyperparameters for the multimodal judge network.

###### Parameter Value

Backbone encoder T5-base Hidden dimension d 768 Embedding dimension 512 Transformer heads 12 Transformer layers 6 Feedforward dimension 3072 Dropout 0.1 Temperature 0.07 Memory bank size 4096 Optimizer AdamW Learning rate 5 × 10−5 Weight decay 0.01 Batch size 16 Epoch 50 λf 1.0 λt 0.5 λa 0.5 λe 0.2

The multimodal judge network is implemented using a transformer-based architecture that evaluates the compatibility between speaker utterances and candidate listener motions. The textual modality is encoded using a pre-trained T5-base encoder, while audio tokens, emotion labels, and motion tokens are

- Table 8: We evaluate the multi-modal matching judge on validation and test set across six input modes (text T, audio A, emotion E, and their fusions). We report pairwise win rates based on mean score comparisons (Win(G>N), Win(G>S), Win(S>N)) and ranking metrics (MRR(G), nDCG@K with graded relevance G>S>N), where G=G (Gold), S=S (Silver), and N=N (Negative).

Mode Split Win(G>N) ↑ Win(G>S) ↑ Win(S>N) ↑ MRR(G) ↑ nDCG@3 ↑ nDCG@5 ↑ nDCG@10 ↑ Val

T Val 0.990 0.873 0.985 0.839 0.878 0.891 0.939 A Val 0.990 0.873 0.985 0.842 0.881 0.893 0.940 T+A Val 0.993 0.883 0.988 0.840 0.875 0.890 0.937 T+E Val 0.994 0.881 0.988 0.841 0.875 0.891 0.938 A+E Val 0.990 0.875 0.985 0.840 0.878 0.892 0.939 T+A+E Val 0.993 0.882 0.988 0.840 0.876 0.890 0.937

###### Test

T Test 0.992 0.873 0.983 0.829 0.864 0.878 0.932 A Test 0.992 0.872 0.983 0.832 0.866 0.878 0.933 T+A Test 0.993 0.879 0.982 0.820 0.855 0.875 0.928 T+E Test 0.993 0.876 0.982 0.826 0.857 0.876 0.929 A+E Test 0.992 0.874 0.983 0.831 0.865 0.878 0.933 T+A+E Test 0.993 0.878 0.982 0.828 0.859 0.878 0.930

embedded and processed through transformer encoders to obtain modality representations. These representations are projected into a shared embedding space where the final compatibility score is computed.

Table 7 summarizes the key hyperparameters used for training the judge network. The model adopts a hidden dimension of 768 and projects the representations into a 512-dimensional embedding space. The transformer encoder uses 12 attention heads and 6 layers with a feedforward dimension of 3072. Training is performed using the AdamW optimizer with a learning rate of 5 × 10−5, weight decay of 0.01, and batch size of 16. A memory bank of size 4096 is used to provide additional negative samples for contrastive training.

#### B.3 Baseline Methods

GT. We use the ground-truth listener motion sequences from the test set as an upper-bound reference.

Random Selection. We randomly sample a motion sequence from HumanML3D [19]

- as a naive baseline.

Retrieval. Following standard text–motion matching protocols [19,82], we retrieve a listener motion by matching the speaker transcription against candidate motions and returning the top-1 nearest neighbor from the training set. Specifically, we use the pretrained text and motion encoders from [19], which are trained with a contrastive objective so that matched text–motion pairs are close in the shared embedding space, while mismatched pairs are separated by a margin. The text encoder maps the input transcription to a semantic feature vector, while the motion encoder first converts a pose sequence into motion snippet codes and then maps them to a motion feature vector. In practice, the text encoder follows the architecture in [19], and the motion encoder is implemented as a bidirectional GRU with hidden size 1,024.

Cascaded LLM→T2M. We construct cascaded baselines by first prompting an LLM to generate the caption of listener reactive motion conditioned on the speaker transcription and emotion. Then, we feed the generated caption into a text-to-motion (T2M) model to synthesize the final motion. Here, we consider two LLMs, Qwen3-30B-A3B and a fine-tuned Qwen3-4B-Thinking, together with two representative T2M generators, T2M-GPT and MG-MotionLLM.

Accordingly, LLM→T2M-GPT denotes the cascade using Qwen3-30B-

- A3B and T2M-GPT, while LLM→T2M-GPT∗ uses the fine-tuned Qwen3-
- 4B-Thinking together with T2M-GPT. Similarly, LLM→MG-MotionLLM denotes the cascade using Qwen3-30B-A3B and MG-MotionLLM, while LLM→MG-MotionLLM∗ uses the fine-tuned Qwen3-4B-Thinking together with MG-MotionLLM.

To keep the main table concise, we report the cascaded baselines under the T+E setting.

#### B.4 Evaluation Metrics

We evaluate model performance from three complementary perspectives: (i) reactive appropriateness, (ii) motion quality, and (iii) diversity.

Reactive appropriateness. Reactive appropriateness measures how well the generated listener motions respond to the speaker utterance. For each speaker utterance Cs, the annotated listener motions are partitioned into three relevance tiers: Gold G(Cs), Silver S(Cs), and Negative N(Cs). Let

Rl(Cs) = {xˆlm,1,...,xˆlm,M} (36)

denote the set of M generated listener motion sequences for the same condition. To assess relative appropriateness, we use the multimodal judge network introduced in Sec. B.1, which assigns a compatibility score

sψ(Cs,xlm) (37) to a candidate listener motion xlm conditioned on the speaker input Cs.

For any candidate set A(Cs), we define its mean judge score as

1 |A(Cs)|

s¯A(Cs) =

sψ(Cs,xlm). (38)

xlm∈A(Cs)

For brevity, we denote the mean scores of the generated set and the three annotated tiers by

g(Cs) = s¯ Rl(Cs), G(Cs) = s¯G(Cs), S(Cs) = s¯S(Cs), N(Cs) = s¯N(Cs). (39) We then report Win(g>G), Win(g>S), and Win(g>N), defined as

1 |D| Cs∈D

κ g(Cs),s¯A(Cs) , A ∈ {G,S,N}, (40)

Win(g > A) =

where D denotes the evaluation set, and

 

1, u > v, 0.5, u = v, 0, u < v.

(41)

κ(u,v) =



Intuitively, Win(g>N) measures whether the generated motions are preferred over clearly inappropriate responses, Win(g>S) is a stricter criterion against moderately appropriate responses, and Win(g>G) is the most challenging criterion against highly appropriate annotated reactions. Higher values indicate stronger reactive appropriateness.

We further report Gen@3, which measures whether at least one generated motion is ranked within the top 3 among all candidates under the same speaker utterance. For each Cs, we form the candidate pool

C(Cs) = G(Cs) ∪ S(Cs) ∪ N(Cs) ∪ Rl(Cs), (42)

rank all candidates in C(Cs) by sψ(Cs,·) in descending order, and denote the resulting rank of a candidate xlm by rankCs(xlm). We then compute

1 |D| Cs∈D

rankCs(ˆxlm) ≤ 3 . (43)

I min

Gen@3 =

xˆlm∈ Rl(Cs)

This metric is particularly suitable for our task because reactive listener behavior is inherently one-to-many: the same speaker utterance may admit multiple plausible listener reactions, and Gen@3 evaluates whether the model can produce

- at least one highly competitive response within a limited candidate budget.

Motion quality. We evaluate motion quality using Fréchet Inception Distance (FID) [23] in a motion feature space. Let feval(xlm) denote the feature representation of a motion sequence extracted by a pretrained motion evaluation network. We compute the feature statistics of generated motions and real motions in the test set, and then measure the Fréchet distance between the two Gaussian distributions:

FID = ∥µr − µg∥22 + Tr Σr + Σg − 2(ΣrΣg)1/2 , (44)

where (µr,Σr) and (µg,Σg) are the mean and covariance of the real and generated motion features, respectively. Lower FID indicates that the generated motions are closer to the distribution of real listener motions, and therefore reflects better overall motion quality.

Diversity. Since a single speaker utterance may admit multiple plausible listener reactions, it is also important to evaluate the diversity of generated motions. Following prior work in human motion generation [82,96], we measure diversity in the same motion feature space. Given the set of all generated motions, we

##### randomly sample two subsets of equal size Sd, denoted by {xˆlm,1,...,xˆlm,S

##### } and {xˆlm,′ 1,...,xˆlm,S′

d

}, and define diversity as

d

Sd

1 Sd

Diversity =

i=1

feval(ˆxlm,i) − feval(ˆxlm,i′ ) 2 . (45)

Higher diversity indicates that the generated motions exhibit greater variation and are less likely to collapse to a small set of repetitive motion patterns.

- Table 9: Full hyperparameter sweep results for group-wise preference training. We vary

the ranking margin m, ranking-loss weight λrank, and Gold-vs-Negative weight λgn. We report pairwise preference metrics (Win(g>N), Win(g>S), Win(g>G)), together with Gen@3, FID, and Diversity.

m λrank λgn Win(g>N) ↑ Win(g>S) ↑ Win(g>G) ↑ Gen@3 ↑ FID ↓ Diversity ↑

- 0.00 0.00 0.00 0.9976 0.7809 0.2585 0.9600 5.2638 5.3005

- 0.00 0.00 0.25 0.9976 0.7809 0.2633 0.9600 5.2638 5.3005

- 0.00 0.00 0.50 0.9976 0.7809 0.2615 0.9600 5.2638 5.3005
- 0.00 0.00 1.00 0.9964 0.7809 0.2615 0.9613 5.2638 5.3005

- 0.00 0.25 0.00 0.9988 0.7809 0.2331 0.9467 5.9644 4.6993

- 0.00 0.25 0.25 0.9952 0.7482 0.2240 0.9467 5.2102 4.8197 0.00 0.25 0.50 0.9988 0.7288 0.2137 0.9455 5.3426 4.9865 0.00 0.25 1.00 0.9939 0.7815 0.2458 0.9528 5.3948 4.7384

- 0.00 0.50 0.00 0.9927 0.7760 0.2548 0.9443 4.6552 4.7315

- 0.00 0.50 0.25 0.9952 0.7730 0.2482 0.9600 5.4479 4.4127

- 0.00 0.50 0.50 0.9952 0.7476 0.2379 0.9600 5.9814 4.3124
- 0.00 0.50 1.00 0.9939 0.7694 0.2512 0.9576 5.3426 4.5137

- 0.00 1.00 0.00 0.9964 0.7548 0.2312 0.9443 6.5379 3.9613

- 0.00 1.00 0.25 0.9891 0.7306 0.2391 0.9479 7.0065 3.9543

- 0.00 1.00 0.50 0.9964 0.7391 0.2125 0.9540 5.5322 4.4312

- 0.00 1.00 1.00 0.9855 0.6731 0.1925 0.9407 6.8036 3.9632

- 0.50 0.00 0.00 0.9964 0.7809 0.2597 0.9600 5.2638 5.3005

- 0.50 0.00 0.25 0.9976 0.7809 0.2639 0.9613 5.2638 5.3005

- 0.50 0.00 0.50 0.9976 0.7809 0.2615 0.9600 5.2638 5.3005
- 0.50 0.00 1.00 0.9952 0.7809 0.2615 0.9588 5.2638 5.3005

- 0.50 0.25 0.00 0.9939 0.7494 0.2349 0.9407 5.0807 4.8318

- 0.50 0.25 0.25 1.0000 0.7966 0.2663 0.9600 4.7596 4.8039

- 0.50 0.25 0.50 0.9903 0.7337 0.2343 0.9407 4.8888 4.6845
- 0.50 0.25 1.00 0.9952 0.8184 0.2778 0.9552 5.1955 4.8183

- 0.50 0.50 0.00 0.9964 0.8287 0.3057 0.9625 5.8396 4.1884

- 0.50 0.50 0.25 0.9952 0.7579 0.2318 0.9310 5.3855 4.3443

- 0.50 0.50 0.50 0.9952 0.7736 0.2385 0.9625 6.2371 4.3488
- 0.50 0.50 1.00 0.9952 0.6762 0.1913 0.9467 6.1306 4.3766

- 0.50 1.00 0.00 0.9915 0.7337 0.2403 0.9492 6.7096 3.9289

- 0.50 1.00 0.25 0.9915 0.7082 0.2149 0.9443 5.4811 4.1878

- 0.50 1.00 0.50 0.9673 0.6132 0.1901 0.9334 6.9334 3.9102

- 0.50 1.00 1.00 0.9891 0.6168 0.1834 0.9237 6.5986 3.9541

- 1.00 0.00 0.00 0.9976 0.7809 0.2597 0.9600 5.2638 5.3005

m λrank λgn Win(g>N) ↑ Win(g>S) ↑ Win(g>G) ↑ Gen@3 ↑ FID ↓ Diversity ↑

- 1.00 0.00 0.25 0.9976 0.7809 0.2609 0.9588 5.2638 5.3005

- 1.00 0.00 0.50 0.9964 0.7809 0.2609 0.9600 5.2638 5.3005 1.00 0.00 1.00 0.9976 0.7809 0.2627 0.9588 5.2638 5.3005

- 1.00 0.25 0.00 0.9964 0.8008 0.2851 0.9516 6.0285 4.2946

- 1.00 0.25 0.25 0.9939 0.7676 0.2464 0.9552 5.1537 4.6242

- 1.00 0.25 0.50 0.9939 0.7821 0.2682 0.9516 5.3639 4.5391
- 1.00 0.25 1.00 0.9988 0.8117 0.2706 0.9625 5.1943 4.6935

- 1.00 0.50 0.00 0.9927 0.7524 0.2288 0.9528 5.3754 4.3702

- 1.00 0.50 0.25 0.9952 0.7361 0.2288 0.9455 5.6698 4.2394

- 1.00 0.50 0.50 0.9903 0.7113 0.2010 0.9516 5.8942 4.3384
- 1.00 0.50 1.00 0.9915 0.6501 0.1816 0.9310 5.6888 4.2328

1.00 1.00 0.00 0.9952 0.6562 0.1973 0.9310 7.0648 3.9867 1.00 1.00 0.25 0.9849 0.5938 0.1774 0.9262 7.4283 3.8852 1.00 1.00 0.50 0.9921 0.5914 0.1798 0.9104 8.6083 3.6349 1.00 1.00 1.00 0.9831 0.5847 0.1731 0.9237 6.2941 3.9609 2.00 0.00 0.00 0.9976 0.7809 0.2567 0.9600 5.2638 5.3005 2.00 0.00 0.25 0.9976 0.7809 0.2585 0.9600 5.2638 5.3005 2.00 0.00 0.50 0.9964 0.7809 0.2579 0.9600 5.2638 5.3005 2.00 0.00 1.00 0.9976 0.7809 0.2627 0.9613 5.2638 5.3005

- 2.00 0.25 0.00 0.9952 0.7639 0.2512 0.9540 5.6781 4.4907

- 2.00 0.25 0.25 0.9891 0.7433 0.2452 0.9588 5.1178 4.7459

- 2.00 0.25 0.50 0.9964 0.7815 0.2603 0.9588 5.6664 4.3494 2.00 0.25 1.00 0.9939 0.7748 0.2785 0.9697 5.7083 4.1561

- 2.00 0.50 0.00 0.9939 0.7264 0.2228 0.9516 6.1482 4.1211

- 2.00 0.50 0.25 0.9964 0.6477 0.1828 0.9249 6.7075 3.8914

- 2.00 0.50 0.50 0.9964 0.6326 0.1901 0.9249 5.4215 4.1601

- 2.00 0.50 1.00 0.9909 0.6610 0.1907 0.9370 6.8355 3.7096

- 2.00 1.00 0.00 0.9927 0.6423 0.1998 0.9298 7.1093 3.8085

- 2.00 1.00 0.25 0.9715 0.6483 0.2046 0.9407 6.8560 3.7436

- 2.00 1.00 0.50 0.9752 0.6362 0.1907 0.9298 6.1279 3.8659

- 2.00 1.00 1.00 0.9655 0.5648 0.1544 0.9140 6.1125 4.0394

### C More Details of ReactMotionNet Dataset

ReactMotionNet exhibits three desirable properties for studying reactive listener motion generation. First, it provides large-scale supervision, containing over 151K labeled speaker–listener pairs. Second, it explicitly captures the one-tomany nature of listener behavior by associating each speaker utterance with multiple candidate reactive motions. Third, it provides graded supervision through Gold, Silver, and Negative labels, supporting both generative modeling and preference-aware evaluation. Moreover, the dataset is split by disjoint speaker utterances, enabling a cleaner evaluation of generalization to unseen conversational conditions.

In total, ReactMotionNet contains 151,328 labeled speaker–listener pairs, covering 8,298 unique speaker utterances and 2,029 unique listener reactive motions. On average, each speaker utterance is paired with 18.24 candidate reactive

[Figure 183]

[Figure 184]

(a) All (b) Train

[Figure 185]

[Figure 186]

(c) Val (d) Test

- Fig. 7: Emotion distributions over the full dataset and across the train/validation/test splits.

motions, further highlighting the inherently one-to-many nature of reactive listener behavior. Among all pairs, 9,307, 34,196, and 107,825 are annotated as Gold, Silver, and Negative, respectively, reflecting the graded appropriateness of candidate reactions. We partition the dataset by speaker utterance using an 8:1:1 train/validation/test split, ensuring that utterances are disjoint across splits, i.e., no utterance appears in more than one partition.

The dataset covers 47 emotion categories, including admiring, adoring, aesthetically appreciative, amused, angry, anxious, ashamed, aware, awed, awkward, bored, calm, confused, contemplative, contemptuous, content, craving, desirous, determined, disappointed, disgusted, distressed, doubtful, ecstatic, embarrassed, empathetic (in pain), entranced, envious, excited, fearful, focused, guilty, horrified, interested, joyful, loving, nostalgic, pained, proud, relieved, romantic, sad, satisfied, surprised, sympathetic, tired, and triumphant. As shown in Fig. 7, these emotion labels exhibit a broad yet imbalanced distribution across the full dataset and each split, making ReactMotionNet a realistic benchmark for modeling diverse affective conversational responses.

### D Additional Experimental Results

#### D.1 Hyperparameter Sensitivity Analysis

We study the sensitivity of group-wise preference training to the ranking margin m, the ranking-loss weight λrank, and the Gold-vs-Negative weight λgn. We

###### Table 10: Representative hyperparameter configurations selected from the full sweep. We emphasize Gen@3, Win(g>S), and Win(g>G), together with FID and Diversity.

Config m λrank λgn Win(g>N)↑ Win(g>S)↑ Win(g>G)↑ Gen@3↑ FID↓ Diversity↑

C1 2.00 0.25 1.00 0.9939 0.7748 0.2785 0.9697 5.7083 4.1561 C2 0.50 0.50 0.00 0.9964 0.8287 0.3057 0.9625 5.8396 4.1884 C3 0.00 0.50 0.00 0.9927 0.7760 0.2548 0.9443 4.6552 4.7315 C4 1.00 0.00 0.25 0.9976 0.7809 0.2609 0.9588 5.2638 5.3005

primarily consider Gen@3, which measures whether generated motions can be ranked among the top plausible candidates under the same candidate budget. We additionally report Win(g>S) and Win(g>G) to assess relative preference quality against medium-quality and high-quality reference candidates, respectively. FID and Diversity are further included to characterize motion realism and output diversity.

The hyperparameter sweep reveals several consistent patterns. First, introducing a small positive ranking margin is beneficial and more reliable than using no margin. Under λrank = 0.25 and λgn = 0.25, increasing m from 0 to 0.5 improves Win(g>S) from 0.7482 to 0.7966, Win(g>G) from 0.2240 to 0.2663, and Gen@3 from 0.9467 to 0.9600, while simultaneously reducing FID from 5.2102 to 4.7596. Although larger margins can further increase Gen@3 in certain cases, such gains are not consistently accompanied by improvements in preference alignment or motion quality, suggesting that excessively large margins may over-specialize the objective.

Second, λrank is the most sensitive hyperparameter in the sweep. Moderate ranking supervision is beneficial, whereas overly large values tend to degrade both alignment and generation quality. For instance, at m = 0.5 and λgn = 0.25, increasing λrank from 0.25 to 0.5 and 0.1 decreases Win(g>S) from 0.7966 to

- 0.7579 and 0.7082, decreases Win(g>G) from 0.2663 to 0.2318 and 0.2149, and worsens FID from 4.7596 to 5.3855 and 5.4811. This indicates that excessive ranking pressure can bias optimization toward relative ordering at the expense of generative fidelity.

Third, λgn has a secondary but non-negligible effect, with a moderate value yielding the most favorable trade-off. At m = 0.5 and λrank = 0.25, setting λgn = 0.25 improves Win(g>S), Win(g>G), and Gen@3 over λgn = 0, while also reducing FID. By contrast, further increasing λgn to 1.0 slightly improves pairwise preference scores, but lowers Gen@3 and degrades FID, indicating that stronger Gold-vs-Negative separation does not necessarily translate into better overall generation quality.

Accordingly, we use m = 0.5, λrank = 0.25, and λgn = 0.25 in all main experiments, as this setting resides in a stable regime of the sweep and yields the most balanced overall performance across preference-oriented and generationoriented criteria.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- Fig. 8: Hyperparameter sensitivity heatmaps under different ranking margins. We show Gen@3, Win(g>S), and FID as functions of λrank and λgn.

#### D.2 Inference Efficiency

- Table 11 lists the inference efficiency of the proposed ReactMotion. During inference, ReactMotion runs on a single NVIDIA A100 80GB GPU and autoregressively generates listener motion tokens conditioned on the speaker’s multimodal inputs. In our evaluation, the model generates 50 listener reactive motions corresponding to 50 speaker utterances. In total, it produces 1,830 motion tokens in 28.8 seconds, achieving a generation throughput of 63.6 tokens per second and

- 1.74 motion sequences per second, which corresponds to an average latency of approximately 0.60 seconds per listener motion sequence.

The generated motion tokens are then decoded into joint sequences using the VQ-VAE decoder. The decoder processes 39.1 motion sequences per second, introducing minimal computational overhead. As a result, the complete pipeline

Table 11: Inference Efficiency on a single NVIDIA A100 (80GB).

Metric Value

Token generation speed 63.6 tokens/s Motion generation speed 1.74 turns/s End-to-end generation speed 1.66 turns/s Average latency per sample ∼0.60 s VQ-VAE decoding speed 39.12 turns/s

achieves an end-to-end throughput of 1.66 motion sequences per second. These results indicate that ReactMotion maintains a favorable balance between model capacity and inference efficiency, enabling near real-time reactive motion generation in conversational scenarios.

#### D.3 More Details of User Study

We conducted a user study on the Tencent Questionnaire platform to evaluate the listener motions generated by ReactMotion (Ours) against two generative baselines, namely the CE variant and LLM→MG-MotionLLM *, as well as the best-in-group Silver reference. A total of 59 volunteers (16 female and 43 male), all with relevant backgrounds in machine learning or deep learning, participated in the study through an online survey. In each trial, participants were presented with a pair of listener-motion videos (A/B) conditioned on the same speaker utterance, with the speaker’s transcript displayed and the corresponding audio played. They were asked to choose which video exhibited the more appropriate reactive listener motion. To avoid positional bias, the two compared motions were randomly assigned to the A/B positions. Each participant completed 36 trials, covering six speaker utterances with six pairwise comparisons per condition. For the Silver condition, we selected the best candidate within each speaker-condition group based on its motion caption and rendered motion clip.

The results in Fig. 5 reveal three notable findings. First, Ours is consistently preferred over both generative baselines, achieving 67.8% preference against CE and 72.0% against LLM→MG-MotionLLM, which demonstrates the advantage of our unified multimodal Seq2Seq formulation over both standard CE training and cascaded generation pipelines. Second, although the Silver reference remains stronger overall, Ours is substantially closer to Silver than either baseline: Ours receives 44.1% of the votes against Silver, whereas CE and LLM→MG-MotionLLM receive only 31.9% and 31.4%, respectively. This indicates that the motions generated by Ours are perceptually much closer to high-quality in-group references. Third, these results highlight the effectiveness of the proposed group-wise preference learning objective, which explicitly models the ordering among Gold, Silver, and Negative reactions and leads to more appropriate listener behaviors under human evaluation. At the same time, the remaining gap between Ours and Silver suggests that reactive listener motion

generation remains challenging, leaving room for further improvement in motion naturalness, contextual precision, and diversity.

#### D.4 Failure Cases

While the model effectively generates contextually appropriate listener motions in many scenarios, capturing deeper conversational intent in complex dialogues remains challenging. In ambiguous or long-tail situations where appropriate listener behavior requires deeper intent understanding, the current model may still exhibit limited robustness. This highlights a promising research direction for future work to further enhance intent-aware interaction modeling in dyadic interaction.

### E Limitations

Since we are the first to explore this task, we design a relatively simple yet effective model architecture to maintain training stability and computational efficiency. This design allows us to validate the core idea of our approach without introducing excessive architectural complexity. The proposed approach already achieves promising results, demonstrating its feasibility and effectiveness. Nevertheless, there remains a large potential for further improvement. Future work could explore more advanced network architectures and more sophisticated training techniques to further enhance performance.

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report (2023)
- 2. Alexanderson, S., Nagy, R., Beskow, J., Henter, G.E.: Listen, denoise, action! audio-driven motion synthesis with diffusion models. ACM Transactions on Graphics (TOG) 42(4), 1–20 (2023)
- 3. Ao, T., Gao, Q., Lou, Y., Chen, B., Liu, L.: Rhythmic gesticulator: Rhythm-aware co-speech gesture synthesis with hierarchical neural embeddings. ACM Transactions on Graphics (TOG) 41(6), 1–19 (2022)
- 4. Barquero, G., Escalera, S., Palmero, C.: Seamless human motion composition with blended positional encodings. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)
- 5. Bishop, C.M., Nasrabadi, N.M.: Pattern recognition and machine learning, vol. 4. Springer (2006)
- 6. Bradley, R.A., Terry, M.E.: Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika 39(3/4), 324–345 (1952)
- 7. Chen, B., Li, Y., Ding, Y.X., Shao, T., Zhou, K.: Enabling synergistic full-body control in prompt-based co-speech motion generation. In: Proceedings of the ACM International Conference on Multimedia (ACM MM). pp. 6774–6783 (2024)

- 8. Chen, C., Zhang, J., Lakshmikanth, S.K., Fang, Y., Shao, R., Wetzstein, G., Fei-Fei, L., Adeli, E.: The language of motion: Unifying verbal and non-verbal language of 3d human motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6200–6211 (June 2025)
- 9. Chen, C., Zhang, J., Lakshmikanth, S.K., Fang, Y., Shao, R., Wetzstein, G., FeiFei, L., Adeli, E.: The language of motion: Unifying verbal and non-verbal language of 3d human motion. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6200–6211 (2025)
- 10. Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., Yu, G.: Executing your commands via motion diffusion in latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18000– 18010 (2023)
- 11. Chiang, W.L., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhu, B., Zhang, H., Jordan, M., Gonzalez, J.E., et al.: Chatbot arena: An open platform for evaluating llms by human preference. In: The International Conference on Machine Learning (ICML) (2024)
- 12. Chopin, B., Tang, H., Otberdout, N., Daoudi, M., Sebe, N.: Interaction transformer for human reaction generation. IEEE Transactions on Multimedia (TMM) 25, 8842–8854 (2023)
- 13. Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems (NeurIPS) 30 (2017)
- 14. Chu, X., Liu, R., Huang, Y., Liu, Y., Peng, Y., Zheng, B.: Unils: End-to-end audiodriven avatars for unified listening and speaking. arXiv preprint arXiv:2512.09327

(2025)

- 15. Défossez, A., et al.: Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037 (2024)
- 16. Dubois, Y., Galambosi, B., Liang, P., Hashimoto, T.B.: Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475 (2024)
- 17. Ghosh, A., Dabral, R., Golyanik, V., Theobalt, C., Slusallek, P.: Remos: 3d motion-conditioned reaction synthesis for two-person interactions. In: European Conference on Computer Vision (ECCV). pp. 418–437 (2024)
- 18. Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L.: Momask: Generative masked modeling of 3d human motions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1900–1910 (2024)
- 19. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5152– 5161 (2022)
- 20. Guo, W., Bie, X., Alameda-Pineda, X., Moreno-Noguer, F.: Multi-person extreme motion prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13053–13064 (2022)
- 21. Han, B., Peng, H., Dong, M., Ren, Y., Shen, Y., Xu, C.: AMD: autoregressive motion diffusion. In: Wooldridge, M.J., Dy, J.G., Natarajan, S. (eds.) The Association for the Advancement of Artificial Intelligence (AAAI). pp. 2022–2030

(2024)

- 22. He, X., Huang, Q., Zhang, Z., Lin, Z., Wu, Z., Yang, S., Li, M., Chen, Z., Xu, S., Wu, X.: Co-speech gesture video generation via motion-decoupled diffusion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2263–2273 (2024)

- 23. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS) 30 (2017)
- 24. Ho, L., Huang, Y., Qin, D., Shi, M., Tse, W., Liu, W., Yamagishi, J., Komura, T.: Interact: A large-scale dataset of dynamic, expressive and interactive activities between two people in daily scenarios. Proceedings of the ACM on Computer Graphics and Interactive Techniques (PACMCGIT) 8(4), 1–27 (2025)
- 25. Hu, T., Zhu, X., Guo, W., Su, K.: Efficient interaction recognition through positive action representation. Mathematical Problems in Engineering 2013(1), 795360

(2013)

- 26. Huang, Y., Wan, W., Yang, Y., Callison-Burch, C., Yatskar, M., Liu, L.: Como: Controllable motion generation through language guided pose code editing. In: European Conference on Computer Vision (ECCV). pp. 180–196 (2024)
- 27. Huang, Y., Khan, S.M.: Dyadgan: Generating facial expressions in dyadic interactions. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2017)
- 28. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card (2024)
- 29. Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al.: Openai o1 system card (2024)
- 30. Jeong, M., Hwang, Y., Lee, J., Jung, S., Kim, W.H.: Hgm3: Hierarchical generative masked motion modeling with hard token mining. In: International Conference on Learning Representations (ICLR) (2025)
- 31. Khirodkar, R., Bansal, A., Ma, L., Newcombe, R., Vo, M., Kitani, K.: Egohumans: An ego-centric 3d multi-human benchmark. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 19807– 19819 (2023)
- 32. Khirodkar, R., Song, J.T., Cao, J., Luo, Z., Kitani, K.: Harmony4d: A video dataset for in-the-wild close human interactions. Advances in Neural Information Processing Systems (NeurIPS) 37, 107270–107285 (2024)
- 33. Kim, D.Y., Lee, H.K., Chung, K.: Avatar-mediated experience in the metaverse: The impact of avatar realism on user-avatar relationship. Journal of Retailing and Consumer Services 73, 103382 (2023)
- 34. Kim, J., Kim, J., Choi, S.: Flame: Free-form language-based motion synthesis & editing. In: The Association for the Advancement of Artificial Intelligence (AAAI). vol. 37, pp. 8255–8263 (2023)
- 35. Ko, W.R., Jang, M., Lee, J., Kim, J.: Air-act2act: Human–human interaction dataset for teaching non-verbal social behaviors to robots. The International Journal of Robotics Research 40(4-5), 691–697 (2021)
- 36. Lee, G., Deng, Z., Ma, S., Shiratori, T., Srinivasa, S.S., Sheikh, Y.: Talking with hands 16.2 m: A large-scale dataset of synchronized body-finger motion and audio for conversational motion analysis and synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 763–772

(2019)

- 37. Li, B., Zhao, Y., Zhelun, S., Sheng, L.: Danceformer: Music conditioned 3d dance generation with parametric motion transformer. In: The Association for the Advancement of Artificial Intelligence (AAAI). vol. 36, pp. 1272–1279 (2022)
- 38. Li, J., Kang, D., Pei, W., Zhe, X., Zhang, Y., Bao, L., He, Z.: Audio2gestures: Generating diverse gestures from audio. IEEE Transactions on Visualization and Computer Graphics (TVCG) 30(8), 4752–4766 (2023)

- 39. Li, R., Dai, Y., Zhang, Y., Li, J., Yang, J., Guo, J., Li, X.: Exploring multi-modal control in music-driven dance generation. In: IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). pp. 8281–8285 (2024)
- 40. Li, R., Zhang, H., Zhang, Y., Zhang, Y., Zhang, Y., Guo, J., Zhang, Y., Li, X., Liu, Y.: Lodge++: High-quality and long dance generation with robust choreography patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) pp. 1–15 (2025)
- 41. Liang, H., Zhang, W., Li, W., Yu, J., Xu, L.: Intergen: Diffusion-based multi-human motion generation under complex interactions. arXiv preprint arXiv:2304.05684 (2023)
- 42. Liao, T.H., Zhou, Y., Shen, Y., Huang, C.H.P., Mitra, S., Huang, J.B., Bhattacharya, U.: Shape my moves: Text-driven shape-aware synthesis of human motions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1917–1928 (2025)
- 43. Liu, H., Zhu, Z., Becherini, G., Peng, Y., Su, M., Zhou, Y., Zhe, X., Iwamoto, N., Zheng, B., Black, M.J.: Emage: Towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1144–1154 (2024)
- 44. Liu, P., Song, L., Huang, J., Liu, H., Xu, C.: Gesturelsm: Latent shortcut based co-speech gesture generation with spatial-temporal modeling. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10929– 10939 (2025)
- 45. Liu, Y., Cao, Q., Wen, Y., Jiang, H., Ding, C.: Towards variable and coordinated holistic co-speech motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1566–1576

(2024)

- 46. Liu, Y., Chen, C., Ding, C., Yi, L.: Physreaction: Physically plausible real-time humanoid reaction synthesis via forward dynamics guided 4d imitation. In: Proceedings of the ACM International Conference on Multimedia (ACM MM). pp. 3771–3780 (2024)
- 47. Liu, Y., Chen, C., Yi, L.: Interactive humanoid: Online full-body motion reaction synthesis with social affordance canonicalization and forecasting (2023)
- 48. Lu, S., Wang, J., Lu, Z., Chen, L.H., Dai, W., Dong, J., Dou, Z., Dai, B., Zhang, R.: Scamo: Exploring the scaling law in autoregressive motion generation model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 27872–27882 (2025)
- 49. Luo, C., Song, S., Yan, S., Yu, Z., Ge, Z.: Reactdiff: Fundamental multiple appropriate facial reaction diffusion model. In: Proceedings of the ACM International Conference on Multimedia (ACM MM). pp. 5607–5616 (2025)
- 50. Luo, C., Wang, J., Li, B., Song, S., Ghanem, B.: Omniresponse: Online multimodal conversational response generation in dyadic interactions. In: Advances in Neural Information Processing Systems (NeurIPS) (2025)
- 51. Luo, C., et al.: Reactface: Online multiple appropriate facial reaction generation in dyadic interactions. arXiv preprint (2024), arXiv:2305.15748
- 52. Meng, Z., Xie, Y., Peng, X., Han, Z., Jiang, H.: Rethinking diffusion for textdriven human motion generation: Redundant representations, evaluation, and masked autoregression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 27859–27871 (2025)

- 53. Mughal, M.H., Dabral, R., Habibie, I., Donatelli, L., Habermann, M., Theobalt, C.: Convofusion: Multi-modal conversational diffusion for co-speech gesture synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1388–1398 (2024)
- 54. Murray, K., Chiang, D.: Correcting length bias in neural machine translation. In: Proceedings of the Conference on Machine Translation (WMT). pp. 212–223

(2018)

- 55. Ng, E., Romero, J., Bagautdinov, T., Bai, S., Darrell, T., Kanazawa, A., Richard, A.: From audio to photoreal embodiment: Synthesizing humans in conversations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1001–1010 (2024)
- 56. Ng, E., Xiang, D., Joo, H., Grauman, K.: You2me: Inferring body pose in egocentric video via first and second person interactions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 9890–9900 (2020)
- 57. Ng, E., et al.: Learning to listen: Modeling non-deterministic dyadic facial motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 58. OpenAI: Introducing openai o3 and o4-mini. https://openai.com/index/ openai-o3-mini/ (2025)
- 59. Park, S., Kim, C., Rha, H., Kim, M., Hong, J., Yeo, J., Ro, Y.: Let’s go real talk: Spoken dialogue model for face-to-face conversation. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL). pp. 16334–16348 (2024)
- 60. Petrovich, M., Black, M.J., Varol, G.: Action-conditioned 3d human motion synthesis with transformer vae. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021)
- 61. Petrovich, M., Black, M.J., Varol, G.: Temos: Generating diverse human motions from textual descriptions. In: European Conference on Computer Vision (ECCV). pp. 480–497 (2022)
- 62. Petrovich, M., Litany, O., Iqbal, U., Black, M.J., Varol, G., Bin Peng, X., Rempe, D.: Multi-track timeline control for text-driven 3d human motion generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1911–1921 (2024)
- 63. Pinyoanuntapong, E., Saleem, M.U., Karunratanakul, K., Wang, P., Xue, H., Chen, C., Guo, C., Cao, J., Ren, J., Tulyakov, S.: Controlmm: Controllable masked motion generation (2024)
- 64. Raab, S., Leibovitch, I., Li, P., Aberman, K., Sorkine-Hornung, O., Cohen-Or, D.: Modi: Unconditional motion synthesis from diverse data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13873–13883 (2023)
- 65. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-totext transformer. Journal of Machine Learning Research (JMLR) 21(140), 1–67

(2020)

- 66. Rubenstein, P.K., et al.: Audiopalm: A large language model that can speak and listen. arXiv preprint arXiv:2306.12925 (2023)
- 67. Ryoo, M.S., Fuchs, T.J., Xia, L., Aggarwal, J.K., Matthies, L.: Robot-centric activity prediction from first-person videos: What will they do to me? In: Proceedings of the Tenth Annual ACM/IEEE International Conference on Human-Robot Interaction (HRI). pp. 295–302 (2015)

- 68. Ryoo, M.S., Matthies, L.: First-person activity recognition: What are they doing to me? In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2730–2737 (2013)
- 69. Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al.: Openai gpt-5 system card (2025)
- 70. Song, S., et al.: React 2024: the second multiple appropriate facial reaction generation challenge. arXiv preprint (2024), arXiv:2401.05166
- 71. Spaccatini, F., Corlito, G., Sacchi, S.: New dyads? the effect of social robots’ anthropomorphization on empathy towards human beings. Computers in Human Behavior 146, 107821 (2023)
- 72. Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., Christiano, P.F.: Learning to summarize with human feedback. Advances in Neural Information Processing Systems (NeurIPS) 33, 3008–3021 (2020)
- 73. Sun, M., Xu, C., Jiang, X., Liu, Y., Sun, B., Huang, R.: Beyond talking–generating holistic 3d human dyadic motion for communication. International Journal of Computer Vision 133(5), 2910–2926 (2025)
- 74. Tevet, G., Gordon, B., Hertz, A., Bermano, A.H., Cohen-Or, D.: Motionclip: Exposing human motion generation to clip space. In: European Conference on Computer Vision (ECCV). pp. 358–374 (2022)
- 75. Tevet, G., Raab, S., Cohan, S., Reda, D., Luo, Z., Peng, X.B., Bermano, A.H., van de Panne, M.: Closd: Closing the loop between simulation and diffusion for multi-task character control. In: International Conference on Learning Representations (ICLR) (2025)
- 76. Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., Bermano, A.H.: Human motion diffusion model. arXiv preprint arXiv:2209.14916 (2022), also known as MDM; widely used as a diffusion baseline.
- 77. Veluri, B., Peloquin, B.N., Yu, B., Gong, H., Gollakota, S.: Beyond turn-based interfaces: Synchronous llms as full-duplex dialogue agents. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 21390–21402 (2024)
- 78. Wang, T., Wu, Z., He, Q., Chu, J., Qian, L., Cheng, Y., Xing, J., Zhao, J., Jin, L.: Stickmotion: Generating 3d human motions by drawing a stickman. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12370–12379 (2025)
- 79. Wang, Y., Leng, Z., Li, F.W., Wu, S.C., Liang, X.: Fg-t2m: Fine-grained textdriven human motion generation via diffusion model. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22035– 22044 (2023)
- 80. Wang, Y., Li, M., Liu, J., Leng, Z., Li, F.W., Zhang, Z., Liang, X.: Fg-t2m++: Llms-augmented fine-grained text driven human motion generation. International Journal of Computer Vision (IJCV) 133(7), 4277–4293 (2025)
- 81. Wang, Z., Wang, J., Li, Y., Lin, D., Dai, B.: Intercontrol: Zero-shot human interaction generation by controlling every joint. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)
- 82. Wu, B., Xie, J., Shen, K., Kong, Z., Ren, J., Bai, R., Qu, R., Shen, L.: Mgmotionllm: A unified framework for motion comprehension and generation across multiple granularities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 27849–27858 (2025)
- 83. Wu, Y., Schuster, M., Chen, Z., Le, Q.V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., et al.: Google’s neural machine translation system: Bridging the gap between human and machine translation (2016)

- 84. Xiao, L., Lu, S., Pi, H., Fan, K., Pan, L., Zhou, Y., Feng, Z., Zhou, X., Peng, S., Wang, J.: Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10086– 10096 (2025)
- 85. Xu, C., Sun, M., Cheng, Z.Q., Wang, F., Liu, Y., Sun, B., Huang, R., Hauptmann, A.: Combo: Co-speech holistic 3d human motion generation and efficient customizable adaptation in harmony. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) pp. 1–18 (2025)
- 86. Xu, L., Lv, X., Yan, Y., Jin, X., Wu, S., Xu, C., Liu, Y., Zhou, Y., Rao, F., Sheng, X., et al.: Inter-x: Towards versatile human-human interaction analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22260–22271 (2024)
- 87. Xu, L., Zhou, Y., Yan, Y., Jin, X., Zhu, W., Rao, F., Yang, X., Zeng, W.: Regennet: Towards human action-reaction synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1759–1769

(2024)

- 88. Xu, S., Dou, Z., Shi, M., Pan, L., Ho, L., Wang, J., Liu, Y., Lin, C., Ma, Y., Wang, W., et al.: Mospa: Human motion generation driven by spatial audio (2025)
- 89. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report (2025)
- 90. Yang, Y., Huang, Z., Xu, C., He, S.: Lagrangian motion fields for long-term motion generation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 48(2), 1171–1184 (2026)
- 91. Yi, H., Liang, H., Liu, Y., Cao, Q., Wen, Y., Bolkart, T., Tao, D., Black, M.J.: Generating holistic 3d human motion from speech. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 469–480 (2023)
- 92. Yin, Y., Guo, C., Kaufmann, M., Zarate, J.J., Song, J., Hilliges, O.: Hi4d: 4d instance segmentation of close human interaction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17016– 17027 (2023)
- 93. Yu, C., Zhai, W., Yang, Y., Cao, Y., Zha, Z.J.: Hero: Human reaction generation from videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10262–10274 (2025)
- 94. Zhang, D., Li, S., Zhang, X., Zhan, J., Wang, P., Zhou, Y., Qiu, X.: Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In: Findings of the Association for Computational Linguistics. pp. 15757– 15773 (2023)
- 95. Zhang, J., Fan, H., Yang, Y.: Energymogen: Compositional human motion generation with energy-based diffusion model in latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17592–17602 (2025)

- 96. Zhang, J., Zhang, Y., Cun, X., Zhang, Y., Zhao, H., Lu, H., Shen, X., Shan, Y.: Generating human motion from textual descriptions with discrete representations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14730–14740 (2023)
- 97. Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., Liu, Z.: Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 46(6), 4115–4128 (2024)
- 98. Zhang, P., Liu, P., Garrido, P., Kim, H., Chaudhuri, B.: Kinmo: Kinematic-aware human motion understanding and generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 11187–11197 (2025)
- 99. Zhang, X., Li, J., Zhang, J., Dang, Z., Ren, J., Bo, L., Tu, Z.: Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 13761–13771 (2025)
- 100. Zhang, X., Li, J., Zhang, J., Ren, J., Bo, L., Tu, Z.: Echomask: Speech-queried attention-based mask modeling for holistic co-speech motion generation. In: Proceedings of the ACM International Conference on Multimedia (ACM MM). pp. 10827–10836 (2025)
- 101. Zhang, Y., Huang, D., Liu, B., Tang, S., Lu, Y., Chen, L., Bai, L., Chu, Q., Yu, N., Ouyang, W.: Motiongpt: Finetuned llms are general-purpose motion generators. In: The Association for the Advancement of Artificial Intelligence (AAAI). vol. 38, pp. 7368–7376 (2024)
- 102. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al.: Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems (NeurIPS) 36, 46595– 46623 (2023)
- 103. Zhou, M., Bai, Y., Zhang, W., Yao, T., Zhao, T., Mei, T.: Responsive listening head generation: A benchmark dataset and baseline. In: European Conference on Computer Vision (ECCV) (2022)
- 104. Zhu, Y., Zhang, L., Rong, Z., Hu, T., Liang, S., Ge, Z.: Infp: Audio-driven interactive head generation in dyadic conversations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025)
- 105. Zou, Q., Yuan, S., Du, S., Wang, Y., Liu, C., Xu, Y., Chen, J., Ji, X.: Parco: Part-coordinating text-to-motion synthesis. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) European Conference on Computer Vision (ECCV). vol. 15114, pp. 126–143 (2024)

