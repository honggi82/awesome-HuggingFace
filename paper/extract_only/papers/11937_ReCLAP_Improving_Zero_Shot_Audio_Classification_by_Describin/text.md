[Figure 1]

## ReCLAP: Improving Zero Shot Audio Classification by Describing Sounds

Sreyan Ghosh♠ Sonal Kumar♠♦ Chandra Kiran Reddy Evuru ♦ Oriol Nieto♦ Ramani Duraiswami♦ Dinesh Manocha♦

♠University of Maryland, College Park, MD, USA ♦Adobe Research, San Francisco, CA, USA {sreyang, sonalkum, ckevuru, ramanid, dmanocha}@umd.edu

# arXiv:2409.09213v1[eess.AS]13Sep2024

Abstract—Open-vocabulary audio-language models, like CLAP [1], offer a promising approach for zero-shot audio classification (ZSAC) by enabling classification with any arbitrary set of categories specified with natural language prompts. In this paper, we propose a simple but effective method to improve ZSAC with CLAP. Specifically, we shift from the conventional method of using prompts with abstract category labels (e.g., Sound of an organ) to prompts that describe sounds using their inherent descriptive features in a diverse context (e.g., The organ’s deep and resonant tones filled the cathedral.). To achieve this, we first propose ReCLAP, a CLAP model trained with rewritten audio captions for improved understanding of sounds in the wild. These rewritten captions describe each sound event in the original caption using their unique discriminative characteristics. ReCLAP outperforms all baselines on both multi-modal audio-text retrieval and ZSAC. Next, to improve zero-shot audio classification with ReCLAP, we propose prompt augmentation. In contrast to the traditional method of employing hand-written template prompts, we generate custom prompts for each unique label in the dataset. These custom prompts first describe the sound event in the label and then employ them in diverse scenes. Our proposed method improves ReCLAP’s performance on ZSAC by 1%-18% and outperforms all baselines by 1% - 55%1.

I. INTRODUCTION

Audio classification, the foundational task of assigning a category label to an audio sample, remains one of the most important tasks in audio processing and has numerous real-world applications. Zeroshot audio classification (ZSAC) presents a promising approach that provides greater flexibility at the inference stage than supervised methods. Unlike supervised methods that map input audio to a fixed set of categories, models classify by computing a similarity score between an input audio example and a caption. To perform inference, one can generate a caption or “prompt” associated with each desired category and match each audio sample to the best prompt. This means that categories can be selected ad hoc and adjusted without additional training.

Open-vocabulary audio-language models like Contrastive Language-Audio Pre-training (CLAP) [1] have outperformed most other models on ZSAC. CLAP is trained on contrastive objectives between audio-caption pairs, where each audio sample corresponds to non-speech sounds and non-verbal speech, and the captions describe the acoustic events and the scene, not the spoken content. Beyond ZSAC, CLAP has also achieved superior performance on cross-modal audio-text retrieval [2] and has been used as a backbone audio encoder for a variety of audio-language tasks, including generalist audio agents [3], open-domain chat assistants [4], audio captioning [5] and text-to-audio generation [6]. However, ZSAC with CLAP currently remains subpar compared to standard supervised methods [7]. We attribute this to 3 main reasons:

1) Limited access to large-scale audio-caption datasets: Unlike CLIP [8], CLAP has not been trained on large-scale, open-

1Code and Checkpoints: https://github.com/Sreyan88/ReCLAP

source audio-caption datasets [2]. This constrains its ability to fully understand and perceive the diverse range of audio and language interactions [2].

- 2) Lack of generalization beyond training category labels: CLAP struggles to generalize beyond the specific category labels used in its training prompts. For instance, research by Tiago et al. [9] indicates that a model’s ZSAC accuracy is closely related to the clusters in its audio embedding space. Consequently, if CLAP was trained on a dataset where the prompt is “Sound of a toothbrush” from AudioSet [10], it might not accurately generalize to a similar label like “brushing teeth” in the ESC50 dataset [11], even though the sounds are similar (both sound like a soft scrubbing or swishing noise, often with a light, scratchy texture).
- 3) Limitations of hand-written prompts for ZSAC: The current ZSAC setup relies on hand-written prompts that correspond directly to dataset category labels. These prompts fail to provide additional context beyond the label itself. For example, CLAP may struggle to classify a label like “Residential Area” in the CochlScene dataset [12] if it has not encountered that label during training. The label alone offers very little information about what sounds characterize a residential area, leading to potential misclassification.

Main Contributions. In this paper, we propose a simple, scalable, and effective approach to improve ZSAC with CLAP. Our contributions are twofold and are summarized as follows:

- 1) We present ReCLAP, a CLAP model trained using caption augmentation. Specifically, we prompt a Large Language Model (LLM) to generate multiple diverse rewrites of the caption associated with each audio. Each rewrite describes the sounds in a unique way. Additionally, they exhibit diversity in sentence structure and vocabulary while preserving the original key concepts and meanings (example in Fig. 1 and Section III-A). This simple data augmentation technique has several advantages, including (1) It enables the model to learn about the distinct acoustic features of sound events beyond what abstract labels alone can provide. This leads to more accurate clustering of sounds based on their actual acoustic properties rather than relying solely on predefined labels. (2) Text-based augmentation via LLM-generated captions provides an effective and scalable method for training-time data augmentation. Unlike traditional data augmentation techniques, which typically involve random audio perturbations, our method is more interpretable and avoids the complexities and limitations of generating synthetic audio. ReCLAP achieves state-of-the-art performance across various retrieval, and ZSAC benchmarks with standard setups.
- 2) To further improve ZSAC performance with ReCLAP, we go

beyond simple hand-written template prompts (e.g., “The sound of a {category}”) by generating multiple custom prompts for each category. This process involves two steps: (1) We prompt an LLM to describe the sound of each category label in t distinct ways, focusing on its unique acoustic characteristics (e.g., Gasp: “a sharp intake of breath”). (2) We then create prompts that place the sound event in diverse scenes, incorporating the descriptions generated in the previous step (e.g., “A sharp intake of breath sliced through the silence as the verdict was announced”). Our proposed method improves the performance of ReCLAP across various ZSAC benchmarks by 1%-55%.

Caption Augmentation

A chorus of mournful, long howls echoes through the crisp winter air, their haunting pitches

Sled dogs are howling in the distance.

[Figure 2]

KRewrittenCaptions

LLM

Original Caption

fadingintothedistance..

.

A ululating howls echoes in the distance, their wailing pitches and reverberations carried by the cold air.

[Figure 3]

[Figure 4]

Caption Selection

|Audio Encoder| |
|---|---|
| | |

|Text Encoder| |
|---|---|
| | |

ReCLAP

|Contrastive Loss|
|---|

II. RELATED WORK

|Caption Augmentation for ReCLAP|
|---|

Following the initial work on CLAP [1], several works have worked to improve its performance. For example, Wu et al. [13] scaled CLAP to 630k audio-caption pairs (including proprietary datasets) and showed a considerable boost in performance. Following this, Elizade et al. scaled their data to 4.6M audio-caption pairs and included speech samples in their training. Ghosh et al. [2] employed 660k pairs using only public domain data to build CompA-CLAP. They also proposed novel techniques to improve the compositional reasoning abilities of CLAP. CLAP has also been employed as an audio or text backbone for a variety of foundational audio processing tasks including text-to-audio generation [6], [14]–[16], audio captioning [5] and audio chat models [3], [4], [17]. Despite its gaining popularity, research efforts to enhance CLAP’s audio and language comprehension skills have been limited, with prior work focusing mainly on scaling.

{"1": "hissing", "2": "sibilant whisper", "3": "rattling", .....} LLM

[Figure 5]

[Figure 6]

Snake

Prompt Augmentation

Category

LLM

t Acoustic Properties

Amidst the rustling leaves, the rhythmic hissing betrayed the snake's hidden dance.

nCustomPrompts

.

[Figure 7]

[Figure 8]

The cave's eerie silence was broken by a sibilant whisper as the snake announced its unseen presence.

|Audio Encoder|
|---|

|Text Encoder|
|---|

ReCLAP

|Prompt Augmentation for ZSAC|
|---|

Fig. 1. Illustration of our proposed method for improving Zero Shot Audio Classification (ZSAC) with language augmentation. Top: We enhance CLAP training through caption augmentation, where each audio’s caption is expanded and rewritten by prompting LLMs to provide detailed descriptions of the sound events. During training, we choose either the original caption or one of the rewritten captions. Bottom: We perform prompt augmentation and generate custom prompts for each label category in the dataset. These prompts describe the sound in the category in diverse scenes.

III. METHODOLOGY A. Caption Augmentation for ReCLAP

CLAP is trained on a contrastive objective between audio-caption pairs to learn a shared representation between the audio and language modalities. Specifically, let Xa and Xt be the audio and its corresponding caption. Additionally, let fa(.) and fb(.) be the audio and text encoders respectively. We first obtain audio and text representations Xˆa and Xˆb as follows:

where augt(.) denotes the rewriting and choosing operation. The primary objective of the rewriting operation is to rewrite the caption so that each sound in the caption is described using its unique acoustic characteristics. An example is as follows:

Xˆa = fa (Xa);Xˆt = ft (Xt) (1)

where Xˆa ∈ RN×D and Xˆb ∈ RN×D. N here is the batch size and D is the embedding dimension. Next, we measure similarity as follows:

- (1) Original Caption: A traction engine is idling.

- (1) Rewritten Caption: A low, rumbling diesel engine hums steadily, its vibrations resonating through the air.
- (2) Original Caption: Cars are starting in pairs.

- (2) Rewritten Caption: Rapid, low-pitched revving of engines, followed by the synchronized, high-pitched roar of multiple cars starting in unison.

C = τ ∗ (Xˆa ⋅ Xˆt⊤) (2)

where C is any similarity function that measures distance using the dot product (cosine similarity in our case). Finally, the contrastive loss is calculated as:

L = 0.5 ∗ (ℓtext (C) + ℓaudio (C)) (3)

log diag(softmax(C)) (4)

We instruct an LLM to complete this task, which ensures that the rewritten captions or augmentations exhibit high levels of diversity in sentence structure and vocabulary while preserving the original key concepts and meanings. The instruction used to prompt the LLM is provided in our GitHub. We employ LLaMa-3.1-8B [18] with in-context examples written by humans. We randomly sample 5 incontext examples for every prompt from a collection of 50.

N

1 N

∑

ℓk =

i=0

We train ReCLAP with a training objective similar to that in CLAP but with caption augmentation. Specifically, we augment each training sample with K additional text captions by rewriting the original caption associated with each audio sample in the dataset in K diverse ways. During training, for each audio sample, ReCLAP chooses the original caption with a probability p = 0.4 or one of the rewritten versions (with a probability 1 − p) where each rewritten caption has an equal probability of selection. Thus, Eqn. 2 can be re-written as:

B. Prompt Augmentation for ZSAC

After training ReCLAP with rewritten captions, we can now employ ReCLAP for ZSAC. However, the standard approach for ZSAC is to handwrite a prompt template and use it for every category in the classification dataset (e.g.,“The sound of a {category}”), like [1],

C = τ ∗ (Xˆa ⋅ ft(augt(Xt))⊤) (5)

TABLE I

PERFORMANCE COMPARISON OF RECLAP WITH BASELINES ON TEXT-TO-AUDIO AND AUDIO-TO-TEXT RETRIEVAL ON AUDIOCAPS AND CLOTHO. RECLAP OUTPERFORMS BASELINES BY 0.4%-38.9%.

### AudioCaps Clotho

|Model Text-to-Audio Audio-to-Text R@1 R@5 R@10 R@1 R@5 R@10<br><br>| |Text-to-Audio Audio-to-Text R@1 R@5 R@10 R@1 R@5 R@10| |
|---|---|---|---|
|MMT 36.1 72.0 84.5 ML-ACT 33.9 69.7 82.6 CLAP 34.6 70.2 82.0 CompA-CLAP 36.1 72.6 81.6 LAION-CLAP (repro.) 34.5 70.7 80.2<br><br>|39.6 76.8 86.7 39.4 72.0 83.9<br><br>41.9 41.9 84.6 45.2 80.1 86.7<br><br>42.5 77.9 87.4<br>|6.7 21.6 33.2<br><br>14.4 36.6 49.9<br><br>16.7 41.1 54.1<br>16.8 43.5 56.1<br><br><br>15.8 39.7 52.9<br>|7.0 22.7 34.6 16.2 37.6 50.2 20.0 44.9 58.7 19.7 45.2 55.6 19.1 44.1 54.9<br><br>|
|CLAP-2.3M 36.2 72.7 82.8 ReCLAP-660k 35.9 72.3 82.5 ReCLAP 37.1 73.2 85.0<br><br>|46.1 79.1 87.5 45.2 79.6 87.9 48.0 80.4 90.8<br><br>|17.1 43.9 56.8 16.8 44.1 55.8<br><br>18.9 44.7 59.0<br><br><br>|19.2 41.2 56.4 18.9 42.8 57.3<br><br>20.5 45.7 58.9<br><br><br>|

[13]. However, this method has a major drawback: The prompt merely specifies the category without providing details about the unique acoustic characteristics of the audio concept corresponding to the category. This limits CLAP’s understanding of arbitrary categories in the wild, which are just abstract definitions of audio concepts. Therefore, incorporating descriptions of a category’s acoustic features into the prompt provides CLAP with an intermediate level of understanding regarding the expected sound of that category. Our proposed method also complements ReCLAP, which possesses additional knowledge about the acoustic features of many audio concepts.

Thus, moving from one standard prompt for every category, we propose employing N custom prompts for every category in the dataset. Since manually hand-writing such custom prompts is infeasible, we instruct an LLM for this task. We instruct an LLM in two stages, with two different instructions. In the first stage, we instruct an LLM to describe the sound of each category label in t distinct ways, focusing on its unique acoustic characteristics:

- (1) Category: Bicycle bell (FSD50k)

- (1) Acoustic Properties: (i) metallic ring, (ii) high-pitched, tinkling chime ⋯
- (2) Category: mallet (NSynth)

- (2) Acoustic Properties: (i) dull thud, (ii) resonant knock (iii) deep thump, ⋯

Next, using these descriptions, we instruct an LLM to generate n different captions for each property, with the sound described in the category occurring in diverse scenes:

- (1) Prompt Caption: A bicycle bell’s clear, metallic ring slices the silence as a rider announces their presence in the peaceful park.

- (2) Prompt Caption: The mallet’s dull thud reverberated through the silent courtroom as the judge announced the verdict.

Finally, for every category, we randomly sample N unique prompts from the pool of n×t total prompts. For ZSC, we mean pool N text embeddings corresponding to the prompts for every label (RN×d → Rd, where d is the shape of embedding output by CLAP). Finally, we calculate the cosine similarity between each audio embedding and all text embedding for all the labels for ZSAC.

IV. EXPERIMENTAL SETUP ReCLAP Training Datasets. We train ReCLAP from scratch on

a collection of multiple datasets including Sound-VECaps [19] and CompA-660k [2]. Detailed statistics about each dataset are provided on our GitHub. Our dataset has ≈2.3M audio-caption pairs.

Evaluation Datasets. For ZSAC, we adopt an evaluation setup similar to prior works [1], [2], [13] and employ AudioSet [10], ESC-50 [11], FSD50k [20], NSynth [21], TUT-Urban [22], UrbanSound8K [23] and VGGSound [24]. We evaluate for accuracy on multi-class and mAP for multi-label.

Model Architecture and Hyper-parameters. We follow the same model architecture as CompA-CLAP [2] with a T5 large text encoder [25] and HTSAT base audio encoder [26]. We train ReCLAP with a learning rate of 5e-4 and an effective batch size (BS) of 256. This BS is smaller than that in the literature, but we do so due to computational constraints. We employed k=4 for caption and N = 2 for prompt augmentation.

Baselines. We use the following baselines for comparison: MMT [27], ML-ACT [28], CLAP [1], CompA-CLAP [2], LAIONCLAP [13] and LAION-CLAP (ours) (reproduced with BS=256 and excluding non-open-source datasets). For ZSAC, we compare with all the baselines mentioned earlier as well as Wav2CLIP [29], AudioClip [30], CoLLAT [7] and ReCLAP w/ only desc. where we only employ t acoustic properties as prompts and don’t generate captions.

Ablations. We perform several ablations to prove the effectiveness of our approach. For multi-modal retrieval: (i) ReCLAP-660k: ReCLAP trained with caption augmentations on 660k pairs from [2]; (ii) CLAP-2.3M.: CLAP trained on our 2.3M audio-caption dataset (without caption augmentations). For ZSAC: For ZSAC we add another ablation which is: ReCLAP w/ only desc: ZSAC with ReCLAP with only the t descriptions as prompts from the prompt augmentation stage and do not generate the N diverse scenario prompts.

V. RESULTS

Table I compares ReCLAP to prior works on AudioCaps and Clotho for text-to-audio and audio-to-text retrieval, showing SOTA performance in most cases. ReCLAP-660k, trained on the same data as CompA-CLAP, surpasses it on all metrics, and similar results are seen for CLAP-2.3M vs. ReCLAP. This demonstrates that ReCLAP’s improvements are not due to dataset size alone. Inspired by [2], we argue that current benchmarks do not fully capture ReCLAP’s capabilities in free-form T-A and A-T retrieval.

TABLE II

PERFORMANCE COMPARISON OF RECLAP WITH BASELINES ON ZERO-SHOT AUDIO CLASSIFICATION BENCHMARKS. RECLAP OUTPERFORMS BASELINES BY 0.6%-54.8%.

ESC-50 US8K VGGSound FSD50K TUT AudioSet NSynth

Wav2CLIP 41.4 40.4 10.0 3.0 28.2 5.0 5.9 AudioClip 69.4 65.3 9.9 6.6 29.5 3.7 6.8 CLAP 82.6 73.2 16.4 14.0 29.6 5.1 9.9 LAION-CLAP (repro.) 88.2 74.1 21.2 22.4 58.4 20.8 11.8 CoLLAT 84.0 77.0 - 19.0 29.0 9.0 CompA-CLAP 86.5 88.1 21.9 19.6 56.7 21.6 11.8 CLAP-2.3M 88.6 90.3 24.5 30.6 61.5 21.9 11.1 ReCLAP 90.0 94.3 24.7 27.2 63.3 23.5 11.4

w/o Prompt Aug.

CLAP 83.4 74.5 16.4 14.9 33.7 6.2 10.2 LAION-CLAP (repro.) 89.5 76.3 23.1 24.5 61.5 21.4 12.4 CLAP-2.3M 89.9 91.2 25.2 37.8 63.7 23.2 13.1 ReCLAP-660k 89.5 79.0 25.9 28.9 60.3 22.9 13.6 ReCLAP w/ only desc. 89.6 92.9 26.8 37.1 65.9 25.4 14.1 ReCLAP 92.8 95.2 29.2 40.2 67.4 26.1 14.7

w/ Prompt Aug.

Table II compares our prompt augmentation method (“w/ Prompt Aug”) with standard template-based prompting (“w/o Prompt Aug”). Prompt augmentation consistently outperforms baselines, with gains of 0.6%-54.8%. ReCLAP shows a 0.9%-17.5% improvement with prompt augmentation. In contrast, CLAP and LAION-CLAP show limited gains, indicating they don’t interpret sound descriptions as effectively as ReCLAP. This highlights the importance of caption augmentation training as a prior step.

VI. RESULT ANALYSIS

Fig. 2 illustrates how ReCLAP and prompt augmentation increase the number of correct predictions for 4 labels from different datasets on ZSAC. As we can see, training CLAP on caption augmentation on the same dataset (CLAP-2.3M vs ReCLAP) improves retrieval of the correct label, which is further boosted by prompt augmentation.

TUT - Airport TUT - Metro StationUS8K - Car HornNSynth- Organ

CLAP-2.3M 187 179 374 316 ReCLAP w/o Prompt Aug.223 210 415 378 ReCLAP w/ Prompt Aug.249 242 423 415

450

423 415

##### TUT - Airport

##### TUT - Metro Station

415

1.2

1.2

400

378

374

1 1

- 0.8
- 1

- 0.8
- 1

350

316

300

0.6

0.6

249 242

0.4

0.4

250

223

210

0.2

0.2

187 179

200

0

0

Series1 Series2

150

100

50

0

TUT - Airport TUT - Metro Station US8K - Car Horn NSynth- Organ

CLAP-2.3M ReCLAP w/o Prompt Aug. ReCLAP w/ Prompt Aug.

Fig. 2. Comparison of accurately classified instances for 4 labels.

TABLE III

EXAMPLES OF AMBIGUOUS LABELS WHERE PROMPT AUGMENTATIONS PROVIDE ADDITIONAL CONTEXT.

Label Prompt Augmentation metro station (TUT) A metallic rhapsody performed by the tireless locomotives,

with a recurring refrain from the station’s vocal spirit. airport (TUT) A chorus of engines hums persistently, interspersed with the murmur of voices and authoritative announcements. organ (NSynth) The organ unfurled a tapestry of majestic harmonies, filling the cathedral with its thunderous hymn. writing (FSD50K) In the hush of the library, the rhythmic scratch of pen on paper becomes a soft dance of intellect and ink.

For example, the sound of an "organ" could refer to either a human organ or a musical instrument. By adding useful contextual information, prompt augmentations assist in clarifying such ambiguities, leading to more accurate retrievals.

VII. HYPER-PARAMETER TUNING

Table IV compares performance across N={1,2,3,4,5} to show the effect of the number of custom prompts N on the final ZSAC performance. As we see, the optimal performance is achieved at N=2, and model performance decreases with an increase in N. This decline is hypothesized to be due to the introduction of more noise into the process with each additional caption.

TABLE IV IMPACT OF N ON ZSAC WITH RECLAP.

|N<br><br>|1 2 3 4 5|
|---|---|
|Score|48.56 52.22 49.37 47.24 44.35<br><br>|

Additionally, Table V shows the effect of probability p on the final ZSAC performance.

TABLE V IMPACT OF p ON ZSAC WITH RECLAP.

|p|0.2 0.4 0.6 0.8<br><br>|
|---|---|
|Score|47.16 52.22 48.29 45.64<br><br>|

VIII. CONCLUSION

In this paper, we propose to improve ZSAC by interpreting sounds using their descriptive features. To achieve this, we first propose ReCLAP, a CLAP model trained using additional caption augmentations that improve CLAP’s understanding of sounds in the wild. Next, we propose to improve ZSAC with prompt augmentation, where we move beyond standard hand-written prompts and generate custom prompts for each category in the dataset. Our proposed method improves ZSAC over our baselines by significant margins.

IX. LIMITATIONS AND FUTURE WORK

- 1) LLM-generated augmentations may result in errors or repetitive captions, which require substantial human oversight. Future work will explore methods to improve quality control.
- 2) Synthetic augmentations from LLMs may introduce biases into models. Future efforts will focus on mitigating these biases.
- 3) Representations from ReCLAP can be employed to improve a range of tasks, including audio generation and understanding. Future work includes exploring these tasks.

REFERENCES

- [1] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang, “Clap learning audio concepts from natural language supervision,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.
- [2] Sreyan Ghosh, Ashish Seth, Sonal Kumar, Utkarsh Tyagi, Chandra Kiran Reddy Evuru, Ramaneswaran S, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha, “Compa: Addressing the gap in compositional reasoning in audio-language models,” in The Twelfth International Conference on Learning Representations, 2024.
- [3] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang, “Pengi: An audio language model for audio tasks,” 2023.
- [4] Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro, “Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities,” 2024.
- [5] Sreyan Ghosh, Sonal Kumar, Chandra Kiran Reddy Evuru, Ramani Duraiswami, and Dinesh Manocha, “Recap: Retrieval-augmented audio captioning,” 2023.
- [6] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley, “Audioldm: Textto-audio generation with latent diffusion models,” arXiv preprint arXiv:2301.12503, 2023.
- [7] Amila Silva, Spencer Whitehead, Chris Lengerich, and Hugh James Leather, “CoLLAT: On adding fine-grained audio understanding to language models using token-level locked-language tuning,” in Thirtyseventh Conference on Neural Information Processing Systems, 2023.
- [8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [9] Tiago Tavares, Fabio Ayres, Zhepei Wang, and Paris Smaragdis, “On class separability pitfalls in audio-text contrastive zero-shot learning,” arXiv preprint arXiv:2408.13068, 2024.
- [10] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2017, pp. 776–780.
- [11] Karol J. Piczak, “ESC: Dataset for Environmental Sound Classification,” in Proceedings of the 23rd Annual ACM Conference on Multimedia. pp. 1015–1018, ACM Press.
- [12] Il-Young Jeong and Jeongsoo Park, “Cochlscene: Acquisition of acoustic scene data using crowdsourcing,” in 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2022, pp. 17–21.
- [13] Yusong Wu*, Ke Chen*, Tianyu Zhang*, Yuchen Hui*, Taylor BergKirkpatrick, and Shlomo Dubnov, “Large-scale contrastive languageaudio pretraining with feature fusion and keyword-to-caption augmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2023.
- [14] Deepanway Ghosal, Navonil Majumder, Ambuj Mehrish, and Soujanya Poria, “Text-to-audio generation using instruction tuned llm and latent diffusion model,” arXiv preprint arXiv:2304.13731, 2023.
- [15] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao, “Makean-audio: Text-to-audio generation with prompt-enhanced diffusion models,” arXiv preprint arXiv:2301.12661, 2023.
- [16] Andrea Agostinelli, Timo I. Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, Matt Sharifi, Neil Zeghidour, and Christian Frank, “Musiclm: Generating music from text,” 2023.
- [17] Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, and James R. Glass, “Listen, think, and understand,” in The Twelfth International Conference on Learning Representations, 2024.
- [18] Hugo Touvron et al., “Llama 2: Open foundation and fine-tuned chat models,” 2023.
- [19] Yi Yuan, Dongya Jia, Xiaobin Zhuang, Yuanzhe Chen, Zhengxi Liu, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xubo Liu, Mark D Plumbley, et al., “Improving audio generation with visual enhanced caption,” arXiv preprint arXiv:2407.04416, 2024.
- [20] Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra, “Fsd50k: An open dataset of human-labeled sound events,” 2022.

- [21] Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Mohammad Norouzi, Douglas Eck, and Karen Simonyan, “Neural audio synthesis of musical notes with wavenet autoencoders,” in International Conference on Machine Learning. PMLR, 2017, pp. 1068–1077.
- [22] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, “A multidevice dataset for urban acoustic scene classification,” arXiv preprint arXiv:1807.09840, 2018.
- [23] Justin Salamon and Juan Pablo Bello, “Deep convolutional neural networks and data augmentation for environmental sound classification,” IEEE Signal processing letters, vol. 24, no. 3, pp. 279–283, 2017.
- [24] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman, “Vggsound: A large-scale audio-visual dataset,” in International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2020.
- [25] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of Machine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.
- [26] Ke Chen, Xingjian Du, Bilei Zhu, Zejun Ma, Taylor Berg-Kirkpatrick, and Shlomo Dubnov, “Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection,” in ICASSP 2022.
- [27] Andreea-Maria Oncescu, A Koepke, Joao F Henriques, Zeynep Akata, and Samuel Albanie, “Audio retrieval with natural language queries,” arXiv preprint arXiv:2105.02192, 2021.
- [28] Xinhao Mei, Xubo Liu, Jianyuan Sun, Mark D Plumbley, and Wenwu Wang, “On metric learning for audio-text cross-modal retrieval,” arXiv preprint arXiv:2203.15537, 2022.
- [29] Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello, “Wav2clip: Learning robust audio representations from clip,” in ICASSP 2022, 2022.
- [30] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel, “Audioclip: Extending clip to image, text and audio,” in ICASSP 2022, 2022.

