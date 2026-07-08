# arXiv:2407.06189v1[cs.CV]8Jul2024

## VIDEO-STAR: SELF-TRAINING ENABLES VIDEO INSTRUCTION TUNING WITH ANY SUPERVISION

Orr Zohar1†, Xiaohan Wang1, Yonatan Bitton2, Idan Szpektor2 & Serena Yeung-Levy1† 1Stanford University, 2Google Research †{orrzohar,syyeung}@stanford.edu Project page: https://orrzohar.github.io/projects/video-star/

ABSTRACT

The performance of Large Vision Language Models (LVLMs) is dependent on the size and quality of their training datasets. Existing video instruction tuning datasets lack diversity as they are derived by prompting large language models with video captions to generate question-answer pairs, and are therefore mostly descriptive. Meanwhile, many labeled video datasets with diverse labels and supervision exist - however, we find that their integration into LVLMs is non-trivial. Herein, we present Video Self-Training with augmented Reasoning (Video-STaR), the first video self-training approach. Video-STaR allows the utilization of any labeled video dataset for video instruction tuning. In Video-STaR, an LVLM cycles between instruction generation and finetuning, which we show (I) improves general video understanding and (II) adapts LVLMs to novel downstream tasks with existing supervision. During generation, an LVLM is prompted to propose an answer. The answers are then filtered only to those that contain the original video labels, and the LVLM is then re-trained on the generated dataset. By only training on generated answers that contain the correct video labels, Video-STaR utilizes these existing video labels as weak supervision for video instruction tuning. Our results demonstrate that Video-STaR-enhanced LVLMs exhibit improved performance in (I) general video QA, where TempCompass performance improved by 10%, and (II) on downstream tasks, where Video-STaR improved Kinetics700QA accuracy by 20% and action quality assessment on FineDiving by 15%.

1 INTRODUCTION

The advent of large Vision-Language Models (LVLMs) marked a significant milestone in artificial intelligence. These models aim to create versatile systems capable of understanding and executing vision-and-language tasks aligned with human intentions. Early advancements in LVLMs, as exemplified by works such as BLIP (Li et al., 2022; 2023a) and LLaVA (Liu et al., 2023b;a), have been driven by the dissemination of pre-trained large language models (LLMs) (e.g., LLaMA (Touvron et al., 2023a;b)) and pre-trained vision/vision-language models (e.g., CLIP (Radford et al., 2021)). LVLMs connect the two model types via visual-language alignment and instruction tuning.

Wei et al. (2022); Liu et al. (2023a); Karamcheti et al. (2024) demonstrated the importance of visual instruction tuning on the resulting LVLM’s performance. However, while much progress has been made in image-LVLMs, video-LVLMs still face challenges due to the relative difficulty in generating quality video instruction tuning datasets. Even though videos contain more complex scene dynamics and temporal information, indicating a need for larger and more diverse training datasets compared to images, the largest video instruction dataset, VideoInstruct-100K (VI-100K) (Maaz et al., 2023), comprises 100K video-text pairs but only 13K unique videos. This is small compared to image instruction datasets like LLaVA-1.5-Instruct (Liu et al., 2023a), which has ∼ 665K image question pairs and ∼ 350K unique images. Such limitation in video instruction tuning leads to performance saturation (Li et al., 2023b; Maaz et al., 2023; Jin et al., 2023; Wang et al., 2023; Lin et al., 2023).

Furthermore, due to video instruction tuning dataset construction - mainly prompting large language models to produce question-answer pairs - these video datasets often degrade to simplistic questions, prompting for video captions — 75% of VI-100K’s questions are of this type (see App. Fig. 10). Combining different sources of supervision has the potential to generate more diverse video in-

[Figure 1]

Input: Video with ANY label/labels Instruction tuning

Output: Instruction Tuning

[Figure 2]

Question

Answer CoT

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Question: What is the action …?

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

BBOX TAL AR/AQA

[Figure 11]

Answer: We first see the man bending…

Text

[Figure 12]

[Figure 13]

###### Answer Generation

Label Verification

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

###### Label

First, we see the man bending down and lifting a book…

First, we see the man bending down and lifting a book. Then, … Finally, the man can be seen reading a book.

Bend Down Read Book

What is the action sequence in the video?

- Figure 1: Video-STaR Overview. Video-STaR can utilize any labeled video dataset, including AR (Action Recognition), AQA (Action Quality Assessment), and TAL (Temporal Action Localization)

– from which it generates video instruction tuning data (video, question, answer triplets). Internally, Video-STaR cycles between: (I) Answer Generation, where an LVLM is prompted to generate candidate answers for the questions. (II) Label Verification where generated answers are filtered to only those that contain the video labels. And (III) Instruction Tuning, where a model is retrained on answers that pass verification. These cycles continue until performance plateaus.

struction tuning datasets, enhancing video understanding. Such supervision exists, as the broader computer vision community has developed an extensive collection of video benchmarks tailored for diverse tasks such as action recognition (Smaira et al., 2020; Soomro et al., 2012), action quality assessment (Xu et al., 2022b; Zhang et al., 2023c), among others (Heilbron et al., 2015; Wu et al., 2021; Grunde-McLaughlin et al., 2021; Grauman et al., 2023; Luo et al., 2022).

Beyond improving overall LVLM performance, adapting LVLMs to novel or out-of-domain tasks is also crucial. While LVLMs have many novel and impactful applications, many remain out of reach — such as analyzing radiology images (Senkaiahliyan et al., 2023), meteorological data (Lawson et al., 2024), traffic analysis (Zhou & Knoll, 2024), judging sporting events (e.g., gymnastics, Olympic diving), and assisting in surgical procedures, among others (Zheng et al., 2024; Jiang et al., 2024; Deng et al., 2024). These tasks require expert, in-domain knowledge that LVLMs lack, necessitating adaptation through instruction tuning. However, collecting video instruction tuning datasets is complex and requires extensive manual effort. For instance, training an ‘AI judge’ to judge Olympic diving would traditionally involve collecting detailed expert critiques of each dive. On the other hand, these tasks often include auxiliary annotations that could be leveraged, such as surgical outcomes in medical procedures or judging scores in Olympic events.

To address these challenges, we take inspiration from LLM’s capability for self-improvement (a.k.a self-training) (Huang et al., 2022; Zelikman et al., 2022), which involves training a model on its generated data and filtering to exclude low-quality outputs. A model’s performance is improved by cycling between generation, filtering, and training. For example, Zelikman et al. (2022) introduced Self-Taught Reasoners which generate chain-of-thought by prompting an LLM to generate answers and rationalize responses, retaining only correctly answered questions for further training. Herein, we explore self-training in LVLMs and introduce Video Self-Training with augmented Reasoning (Video-STaR, see Fig. 1). Video-STaR enables the incorporation of any labeled video dataset of any format or task by prompting an LVLM with the video and a question to generate answers (Fig. 2, 2.1) containing the video content’s label. If the model cannot correctly answer the question, we provide the video label and ask it to rationalize it (Fig. 2, 2.2). We then utilize the labels as weak supervision, rejecting answers that do not contain the correct label (Fig. 2, 2.3). By facilitating the use of any supervision for video instruction tuning, Video-STaR enables the creation of diverse datasets.

Our experimental setup initializes Video-STaR with Video-LLaVA (Lin et al., 2023), focusing on assessing its impact on video question-answering (VQA) performance. After a few Video-STaR training cycles, we compare the performance of Video-STaR to other state-of-the-art LVLMs and strong baselines to gauge the effectiveness of the Video-STaR framework. Our findings demonstrate notable enhancements in accuracy and reasoning capabilities, highlighting Video-STaR’s role in overcoming the constraints posed by conventional video instruction tuning datasets. We show that the integration of Video-STaR not only boosts Video-LLaVA’s performance on standard zeroshot VQA benchmarks but also significantly improves its adaptability to various downstream video understanding tasks. This underscores Video-STaR’s capacity to advance LVLM training while improving overall performance and versatility.

### Our contributions can be summarized as follows:

- 1. We introduce Video Self-Training with augmented Reasoning (Video-STaR), the first video self-training Large Video-Language Model method. Using self-training, Video-STaR enables the use of any labeled video dataset for video instruction tuning.

- 2. Video-STaR improves zero-shot video question answering performance on various benchmark datasets, as evidenced by increased accuracy on TempCompass (Liu et al., 2024) from 45.7 to 50.3 (+10%) and on MSVD-QA (Maaz et al., 2023) from 69.7 to 71.3 (+2.3%).
- 3. We demonstrate that Video-STaR can adapt LVLMs to diverse video tasks, notably enhancing action quality assessment on FineDiving (Xu et al., 2022b), where accuracy rose from 17.6 to 20.2 (+15%) and in action recognition on Kinetics700 (Smaira et al., 2020), where accuracy increased from 50.0 to 59.9 (+20%).
- 4. Utilizing Video-STaR, we create a large, 1M video instruction tuning dataset - VSTaR-1M, sourced from diverse datasets and tasks, and show that it benefits LVLM performance.

2 VIDEO SELF-TRAINING WITH AUGMENTED REASONING (VIDEO-STAR)

Given a dataset of videos v and their corresponding labels l : D = {(vi,li)}di=1, Video-STaR’s objective is to create question q answer a pairs to instruction-tune the pre-trained model M on the

dataset Dˆ = {(vi,qi,ai)}di=1f , producing the instruction-tuned model Mˆ . Note that videos need not be from the same task, and may contain multiple labels. We start by prompting a large language

model with a task description T and video labels L to generate candidate questions q:

YT,L = A video is labeled {L} for the task of {T}. What questions could you ask someone about the video that should contain the video labels in the response?

Video-STaR performs generation-training cycles, where in cycle i the instruction-tuned model Mˆ i⋆ is produced, while the instruction-tuned model from the previous cycle Mˆ (i−1)⋆ is utilized for training data generation. We initialize the process with Mˆ 0⋆, an existing instruction-tuned model.

To prepare the training data in cycle i, answers are generated either directly via Answer Generation or through backward rationalization via Label Rationalization. In Answer Generation, Mˆ (i−1)⋆ is prompted with questions (Sec. 2.1). Candidate answers are then filtered using the original video labels (Sec. 2.3). Videos rejected during direct Answer Generation are rationalized, where Mˆ (i−1)⋆ is provided both a video vi and labels li, and then prompted with the question again (Sec. 2.2). Candidate answers are filtered again, creating the instruction tuning dataset in cycle i, Dˆi of size di. A pre-trained model M is then finetuned on Di, producing Mˆ i⋆. The next cycle generates data using Mˆ i⋆, until the performance plateaus (see Fig. 2).

- 2.1 ANSWER GENERATION

Each Video-STaR cycle begins in direct Answer Generation. In this phase, Mˆ (i−1)⋆ is prompted with the video-question pair to provide an answer along with a detailed rationale:

YQ = Question: {Q}. Rationalize your answer step-by-step; how can one arrive at

this conclusion?

When prompted with the question qi on a particular video, Mˆ (i−1)⋆ is expected to generate an answer ai that contains the label ˆli and the rationale ri (ai = ri ∪ ˆli, see Fig. 2). We observe that answers containing the correct labels are of higher quality and suffer less from hallucination. Therefore, we filter the generated answers to include only those that contain the correct label (ˆli = li) utilizing a verifier (Sec. 2.3). For an example of Answer Generation, see Fig. 3.

- 2.2 LABEL RATIONALIZATION

Answer generation has two main drawbacks: (i) in some applications, especially on challenging/outof-domain tasks, initial answer generation yield is low, resulting in almost no training samples after filtering (e.g., FineDiving, see Fig. 3); (ii) improvement plateaus as the model fails to solve new problems in the training set, and it is only trained on examples it answers correctly.

|(2.1) Answer Generation<br><br>[Figure 19]<br><br>Video Question<br><br>Rationale Label|[Figure 20]<br><br>(2.3) Label Filtering|
|---|---|
| |[Figure 21]<br><br>Label incorrect|

|(2.2) Label Rationalization Video Question Label<br><br>[Figure 22]<br><br>Rationalization Label|
|---|

Label correct Label correct

[Figure 23]

[Figure 24]

|(2) Instruction Tuning Video Question<br><br>[Figure 25]<br><br>Answer<br><br>[Figure 26]|
|---|

[Figure 27]

Label incorrect

Repeat

(2.3) Label Filtering

Video-STaR

- Figure 2: Video Self-Training with augmented Reasoning. (2.1) We initialize by prompting an LVLM to generate an answer for a particular video. (2.3) We then filter the generated answers to those only containing the original video labels. (2.2) The videos whose generated answer did not contain the ground-truth labels are then sent to label rationalization, where given the video, question, and label - the model is expected to rationalize the label. (2.3) The generated answers are filtered again to those only containing the ground-truth labels, and (2) the LVLM is instruction-tuned from the pre-trained checkpoint on the resulting dataset. The cycle is then repeated.

Inspired by Zelikman et al. (2022), for videos whose Mˆ (i−1)⋆ generated answers did not contain the ground-truth labels, we introduce label rationalization as part of Video-STaR. Concretely, we provide Mˆ (i−1)⋆ the video, question, and video label and instruct the model to rationalize the label:

YQ,L = Question: {Q}. Answer: {L}. Can you rationalize the answer step-by-step? How can one arrive at this conclusion?

Given the correct label, the model can reason backward and more easily generate a rationale leading to the correct answer. However, label rationalization is more prone to hallucinations, so we prefer direct answer generation and use rationalizations only if answer generation fails. For an example of Label Rationalization, see Fig. 3, top. The generated answers are then filtered again, keeping only those that contain the correct label (ˆli = li) utilizing a verifier (Sec. 2.3). Label Rationalization is only utilized in training cycles; only direct answers are used to produce the final model Mˆ ⋆.

- 2.3 LABEL VERIFICATION

Video-STaR aims to utilize the labels as weak supervision in instruction tuning data generation. Gold labels are a grounding aspect of our datasets and represent some ground-truth knowledge. Therefore, we assume that answers that contain the ground-truth labels in their responses are higher quality than those that don’t. While we would like to validate the existence of the different labels in the generated text, this can be non-trivial.

To this end, we introduce the Parser-Verifier. The Parser, P extracts the predicted labels from the generated text (lˆi = P(ai)), using a mixture of named entity recognition and Regex. Regex is used to identify easily identifiable string patterns, such as bounding boxes and time ranges, while named entity recognition is used for more nuanced entities, such as timestamps. The Verfier, V compares the extracted labels with the gold ones using the appropriate metrics (V (li,lˆi) → R). For example, IoU for bounding boxes/temporal action localization, and BERT (Devlin et al., 2018) embedding similarity for sentence ordering. Each video has between 1-3 associated labels. To be classified as correct, the predicted labels must be within a 5% margin of error from the gold.

- 3 VIDEO-STAR GENERATED DATASET - VSTAR-1M

In this section, we detail the different source datasets utilized in our study (Sec. 3.1) and analyze the generated Video-STaR Dataset, VSTaR-1M (Sec. 3.2).

Source Videos Labels Avg. Dur. Source Task Kinetics700 (2020) 650K 700 5.1 Action Recognition STAR benchmark

- (2021)

22K 207 28.6

Video Reasoning, Temporal

Action Localization,

Bounding Box FineDiving

- (2022b)

Action Quality Assessment,

3K 1065 3.2

Action Sequence

- Table 1: Source Dataset Summary. Video datasets used as source datasets for instruction tuning data generation, their source tasks, and the number of unique labels.

Label Rationalization

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Input Instruction] Please assess the performer's skill level and the action's toughness in the video on a scale of 1-100. Answer: Action Sequence: Inward, 3.5 Soms.Tuck, Entry. Action difficulty: 3.2. Overall Score: 65.6. Can you explain step-by-step how one can arrive at this conclusion?

The performer in the video executes a series of flips and twists in the air, showcasing their athleticism and skill. The first flip is an inward somersault... The second flip is a 3.5 somersault,… The third flip is a tuck… The fourth flip is an entry, which is the final maneuver before the performer enters the water. Overall, the performer demonstrates a high level of skill and athleticism throughout the routine, making it a challenging and impressive display of acrobatics.

[Figure 36]

[Figure 37]

[Figure 38]

Generation Finetuning

- Cycle 0
- Cycle 1
- Cycle 2

Answer

First, we need to identify the action sequence performed by the diver… inward somersault, followed by a 3.5 somersault tuck, and then an entry into the water. Next, we need to assess the difficulty of each action in the sequence. … In this case, the difficulty of each action is as follows: Inward: 3.5, 3.5 Soms.Tuck: 3.2, Entry: 3.0… In this case, the average difficulty score is 3.2… we can conclude that the diver's action sequence is of a high skill level and is quite tough, with an overall difficulty score of 65.6.

[Figure 39]

[Figure 40]

[Figure 41]

Answer Generation

[Input Instruction] Please assess the performer's skill level and the action's toughness in the video on a scale of 1-100.. Can you explain step-by-step how one can arrive at this conclusion? First, we need to identify the action sequence performed by the diver. In this case, the diver performs an inward somersault, followed by a 3.5 somersault tuck, and then an entry into the water... The action difficulty score for a inward to a 3.5 somersault tuck is 3.2… we can conclude that the diver's action sequence is of a high skill level and is quite tough, with an overall score of 64.68

[Figure 42]

[Figure 43]

[Label] 65.6

- Figure 3: Qualitative Improvement of Data Generation over Cycles on FineDiving. We initialize the model with Video-LLaVA (Cycle 0), where the model cannot generate an answer (→ |×) or rationalize the label correctly (| → ×). In the second cycle (Cycle 1), the model still cannot generate an answer (→ |×) but can rationalize the video label (✓| →), which is selected for instruction tuning. Finally, in the third cycle (Cycle 2), the model directly generates a correct answer (✓| →), which is selected for visual instruction tuning. We highlight in green correct answers, in red wrong answers, and in yellow - hallucinations.

- 3.1 SOURCE DATASETS

In selecting source datasets, we selected datasets that contain diverse video content and label types, please see Tab. 2.3. These include Kinetics700 (Smaira et al., 2020), which has action recognition annotations and is particularly large and diverse. FineDiving (Xu et al., 2022b) is an action quality assessment dataset of Olympic diving events and has both an overall score and action sequence annotations. Finally, STAR-benchmark (Wu et al., 2021), a video reasoning dataset, also contains bounding box and temporal action localization annotations. Tab. 2.3 contains the relevant dataset statistics, e.g., the number of videos and labels per dataset.

- 3.2 GENERATED DATASET ANALYSIS

Quantitative Analysis. Through the application of Video-STaR, significant dataset augmentation was achieved over two cycles of, illustrated in Fig. 4. This figure displays the Answer Generation and Label Rationalization yield across the datasets source. Notably, the initial application of Video-

###### Kinetics700 STAR-Benchmark FineDiving

[Figure 44]

[Figure 45]

[Figure 46]

Video

On a scale from 1-100, how would you evaluate the execution of the actions in the video?

What type of activity is being performed in this video?

Which object was put down by the person?

Question

In the video, the person is seen picking up a bag and putting it down on the floor. After that, he picks up a book and puts it down on the floor. Based on this information, we can conclude that the person put down the bag and picked up the book.

First, we identify the diver’s action sequence: inward somersault ··· entry into the water. Evaluating each movement’s difficulty, we assign ··· inward somersault: 2.5, ···. By averaging these scores ··· the overall score is 63.0 out of 100.

In the video, a man is seen holding a large axe and swinging it around while a crowd watches. He then proceeds to hit a block of wood with the axe. Based on these observations, it can be concluded that the activity is chopping wood.

Generated Answer

Label Smashing The bag. Overall Score 63.0

- Table 2: Examples of Generated Data. Examples of the video, question, Video-STaR generated answer, and ground-truth label from each source dataset.

Kinetics700

STAR-Benchmark

FineDiving

100

Answer Generation

Label Rationalization

75

Answer Generation

50

Label Rationalization

25

Answer Generation

Label Rationalization

0

0 1 2

0 1 2

0 1 2

Cycle

Cycle

Cycle

- Figure 4: Dataset Yield vs. Cycles. Percentage of the videos converted to instruction tuning by the Answer Generation and Label Rationalization per dataset. As can be seen, on difficult datasets, such as FineDiving, no videos are converted by Answer Generation in the first cycle. By utilizing Label Rationalization, the model is able to improve to eventually generate answers correctly.

LLaVA on datasets like Kinetics700 and STAR-Benchmark showed significant Answer Generation success rates. However, the FineDiving dataset presented a notable challenge, with Answer Generation having no answers generated directly, underscoring the complexity of the dataset and the critical role of Label Rationalization. By the end of the second cycle, a substantial number of highquality instances had been produced, showcasing both the effectiveness of Video-STaR in converting labeled video datasets into video instruction tuning dataset, as evidenced in Fig. 4.

Qualitative Analysis. See Tab. 3.2 for examples of generated question-answer pairs. From Kinetics700, we extracted an instance showcasing a video labeled ‘smashing’. Video-STaR correctly identified a more fine-grained label, ‘chopping wood’. In the FineDiving dataset, a clip depicting a complex dive was accompanied by the question ‘On a scale from 1-100···’ The model’s output text provided a breakdown of the dive’s components, leading to a score (label), as would be desired from an LVLM visual assistant. Finally, in the STAR benchmark, questions are already provided; therefore, we utilized them directly.

In 3, we show the qualitative improvement of the generated data over Video-STaR cycles. In the first cycle (Cycle 0), Video-LLaVA failed at Answer Generation and Label Rationalization. After one Video-STaR cycle (Cycle 1), Video-STaR still failed at Answer Generation but succeeded in Label Rationalization. After the final Video-STaR cycle (Cycle 2), Video-STaR managed to generate the answer without requiring the label via Answer Generation.

- 4 EXPERIMENTS

We experimented with Video-STaR and evaluated its enhanced video understanding capabilities. In Sec. 4.3, we evaluate how Video-STaR adapts Large Vision-Language Models (LVLMs) to the source datasets and how these capabilities are transferred zero-shot to similar benchmarks. In Sec. 4.2, we evaluate the video question-answering capabilities on video benchmark datasets.

###### Action Direction Speed Event Attribute Change Avg.

Fine Coarse Obj. Cam. Abs. Rel. Order Color Size Both Other

|Random mPLUG-Owl (2023) Video-LLaVA (2023) Video-LLaVA+ Vid-LLaVAGemini Video-STaR<br><br>|39.7 40.1 48.8 66.1 63.4 93.5 62.1 93.0 30.7 30.1 68.6 94.1<br><br>|39.8 39.0 38.7 36.8<br><br>36.1 34.8<br><br>35.0 32.6<br><br>37.8 40.0 35.8 38.0<br><br><br>|40.8 42.0 42.2 38.4 42.7 26.5<br>41.1 38.7 41.8 42.4 38.7 37.6<br><br><br>|41.5 42.0 39.1 36.4 21.5 37.1<br><br>|40.4 39.9 38.9 39.4<br>41.7 44.7 41.9 39.9<br><br><br>52.6 37.1 43.3 33.3 59.0 40.2 36.7 44.4 50.4 49.9 38.0 37.4<br><br>53.8 48.5 45.0 55.6<br><br><br>|40.5 44.4 45.7 47.2 38.2 50.3(+10%)<br><br>|
|---|---|---|---|---|---|---|
|Gemini-1.5 (2024)|94.8 98.4<br><br>|43.6 42.4|65.3 48.7<br><br>|55.6|79.5 59.8 70.0 66.7|66.0|

### Table 3: Comparison with state-of-the-art methods on TempCompass. TempCompass (Liu et al.,

2024) assesses the temporal understanding capabilities of video language models across five dimensions Video-STaR improves Video-LLaVA performance on TempCompass by 10%.

Dataset MSVD-QA MSRVTT-QA TGIF-QA ActivityNet-QA size Accuracy Score Accuracy Score Accuracy Score Accuracy Score

Methods

VideoChat (2023b) 4K 56.3 2.8 45.0 2.5 34.4 2.3 - 2.2 Video-LLaMA (2023a) 4K 51.6 2.5 29.6 1.8 - - 12.4 1.1 Video-ChatGPT (2023) 100K 64.9 3.3 49.3 2.8 51.4 3.0 35.2 2.7 Video-LLaVA∗ (2023) 100K 69.7 3.9 57.4 3.5 46.5 3.3 43.2 3.4 Video-LLaVA+ 650K 67.8 3.8 56.0 3.4 46.5 3.3 42.2 3.3 Vid-LLaVAGemini 2k 67.2 3.9 55.9 3.5 44.5 3.2 39.6 3,3 Video-STaR 550K 71.3 4.0 58.2 3.5 47.3 3.3 43.2 3.3

Gemini-1.5-pro (2024) - 71.6 3.9 52.6 3.2 45.0 3.1 56.7 -

#### Table 4: Zero-shot Video QA benchmarks. As can be seen, many models are approaching Gemini performance - indicating that LVLMs may be operating near the noise level on these benchmarks.

- 4.1 EXPERIMENTAL SETTING

Implementation Details. We initialize from the Video-LLaVA (Lin et al., 2023) model, which utilizes the Vicuna-7B v1.5 (Chiang et al., 2023). We ran three Video-STaR cycles, and each cycle was initialized with the pre-trained Video-LLaVA weights. We train for one epoch using a 128 batch size, AdamW optimizer, and a cosine learning rate schedule. The learning rate is 2e − 5 with a 0.03 warmup ratio. In combination with the generated Video-STaR instruction tuning dataset, we additionally utilized the VideoInstruct-100K (Maaz et al., 2023) and the LLaVA v1.5 instruction tuning datasets (Liu et al., 2023a). Additional details are available in the appendix.

Baselines. Besides comparing to Video-LLaVA, we also wanted to evaluate the effect of utilizing additional data and naively adapting the source datasets. Therefore, we utilized simple templates to generate question-answer pairs from the video labels and trained Video-LLaVA on the resulting dataset. We will reference this baseline as Video-LLaVA+. Another baseline for adapting Large Vision-Language Models to novel tasks is model distillation, where a stronger video model - in this work, Gemini 1.5 pro-vision - is utilized to label/annotate a small set of videos (500 from each dataset) and used to finetune the models. We will reference this baseline as Vid-LLaVAGemini.

Evaluation Details. We evaluate on the following benchmarks; the Zero-shot question-answer (QA) benchmarks: MSVD-QA, MSRVTT-QA, TGIF-QA, and ActivityNet-QA (Xu et al., 2017; 2016; Jang et al., 2017; Heilbron et al., 2015). TempCompass (Liu et al., 2024), a multiple-choice fine-grained QA benchmark. Adapted task performance is evaluated by converting source datasets using simple templates and applying the same evaluation protocol as Maaz et al. (Maaz et al., 2023), producing Kinetics700-QA, STAR-benchmark-QA, and FineDiving-QA. This protocol reports two metrics: accuracy (the percentage of correctly answered questions) and the average score (where ChatGPT rates each response on a scale of 1-5 and calculates the mean of these scores). All evaluations utilize the same GPT model (Wu, 2024) (“gpt-3.5-turbo”) to ensure consistent comparisons. Due to cost considerations, 1000 videos were randomly selected from each dataset for Gemini evaluation. The reported values are used on ActivitlyNet-QA.

Kinetics700-QA STAR-bench-QA FineDiving-QA Accuracy Score Accuracy Score Accuracy Score

Methods

Video-LLaVA 50.0 3.2 24.9 2.6 17.6 2.2 Video-LLaVA+ 49.5 3.2 28.8 2.8 19.1 2.2 Vid-LLaVAGemini 41.9 2.9 22.3 2.6 16.3 2.1 Video-STaR 59.9 (+20%) 3.5 (+10%) 33.0 (+33%) 2.9 (+12%) 20.2 (+15%) 2.3 (+5%)

- Table 5: Adapted Dataset Performance. Performance metrics on test sets of Kinetics700, FineDiving, and STAR-benchmark datasets via converting them to QA following Maaz et al. (2023). Video-STaR shows significant improvement over Video-LLaVA and Video-LLaVA+, showing the potential of Video-STaR for LVLM adaptation to new tasks.

- 4.2 QUANTITATIVE EVALUATION ON ZERO-SHOT BENCHMARKS

To evaluate Video-STaR’s effect on general video question answering, we evaluated its effect on Video-LLaVA’s performance on TempCompass, see Tab. 3. On TempCompass, Video-STaR outperformed Video-LLaVA across the board– by ∼ 10%. To see if this performance boost is simply a factor of training on a larger dataset, we also evaluated Video-LLaVA+. Video-LLaVA+was trained on even a larger video dataset by naively utilizing video labels, and yields a more modest improvement of 3%, showing the utility of Video-STaR. TempCompass is also a fine-grained dataset that would be sensitive to hallucinations, indicating that Video-STaR is not more prone to hallucinations compared to existing methods. Gemini 1.5 pro scored an impressive 66.0 on TempCompass, showing there is still much room for improvement on this benchmark.

We then continued and evaluated Video-STaR’s effect on zero-shot video QA performance on the MSVD-QA, MSRVTT-QA, TGIF-QA and ActivityNet-QA benchmarks. As can be seen in Tab. 4, Video-STaR achieves performance improvements where, for instance, on the MSVD-QA dataset, Video-STaR attains the highest accuracy of 71.3% vs Video-LLaVA’s 69.7. On MSRVTT-QA, Video-STaR leads with an accuracy of 58.2% and maintains a competitive edge in other datasets like TGIF-QA and ActivityNet-QA. Seeing the relatively small performance gains compared to TempCompass, we additionally evaluated Gemini 1.5 pro-vision on 1000 video subsets of each dataset and found that its performance is on par with existing open-source models. We believe this shows that we are near the ‘noise’ limit of these benchmarks. Our qualitative analysis indicated that many of the questions selected as ‘wrong’ are actually due to the benchmark design—overly general questions with multiple correct answers. Concurrent work (Wu, 2024) has similarly concluded that the ChatGPT-3.5 version utilized in evaluation can lead to variations of ±10 in accuracy.

- 4.3 QUANTITATIVE EVALUATION ON ADAPTED DATASETS

Besides improving general visual question-answering performance, Video-STaR can also adapt Large Vision-Language models to novel takes. To demonstrate this, we converted the test sets (not included in training) of the source datasets – Kinetics700, STAR-benchmark, and FineDiving. The results of these evaluations are reported in Tab. 5. Adapting LVLMs with easier-to-collect labels can be helpful in various applications, leading to a more versatile, multi-domain capable assistant. When evaluating Video-STaR’s impact on LVLM performance on the diverse source datasets, we found that it significantly improves model performance, particularly on complex tasks. For instance, on Kinetics700, known for its extensive action categories, Video-STaR enhanced Video-LLaVA’s performance accuracy by an average of 20% (as can be seen in Tab. 5), showcasing its ability to develop generalized models adept across multiple domains. Interestingly, Video-LLaVA+’s performance did not improve compared to Video-LLaVA, and in some cases, even worsened, showing that one cannot directly utilize labeled datasets for LVLM adaptation.

Action Quality Assessment (AQA) is a complex video task requiring detailed action understanding, where Video-STaR significantly enhanced LVLM performance on the FineDiving dataset. Our results show a notable improvement from 17.6 to 20.2 in score prediction accuracy, highlighting Video-STaR’s effectiveness in refining LVLM’s temporal reasoning. However, Video-STaR allows LVLMs to not only rate a particular dive but also explain the rationale behind each assessment. This rationale is invaluable for many applications, effectively providing potential user feedback for

Kinetics700-QA STAR-bench-QA FineDiving-QA Accuracy Score Accuracy Score Accuracy Score Video-STaR 59.9 3.5 33.0 2.9 20.2 2.3

Ablations

- - Rationalization 59.8 3.5 26.6 2.7 12.8 2.0

- - Generation 50.0 3.2 24.9 2.6 17.6 2.2

- Table 6: Ablations on Adapted Datasets. Performance metrics on test sets of Kinetics700, STARbenchmark, and FineDiving datasets. Label Rationalization impacts mostly the difficult datasets, such as FineDiving, whose initial Answer Generation yields are low.

Ablations

MSVD-QA MSRVTT-QA TGIF-QA ActivityNet-QA

Accuracy Score Accuracy Score Accuracy Score Accuracy Score Video-STaR 71.3 4.0 58.2 3.5 46.8 3.3 42.2 3.3

- - Rationalization 70.6 3.9 57.5 3.5 47.7 3.4 42.2 3.3

- - Generation 69.7 3.9 57.4 3.5 46.5 3.3 43.2 3.4

- Table 7: Ablations on Zero-Shot Benchmarks. In simpler benchmarks, Answer Generation proved more critical for zero-shot generalization than Label Rationalization.

improvement. This advancement enables novel applications, from sports coaching to automated feedback systems, by offering evaluations and constructive feedback. The ability to interpret and improve action quality underscores the potential of Video-STaR, underscoring the potential of utilizing LVLMs as intelligent and informative visual assistants. For more, please see App. Sec. A.2.

- 4.4 ABLATIONS

In our ablation studies, we evaluated the impact of removing Label Rationalization and Answer Generation from Video-STaR, focusing on adapted datasets (Kinetics700, FineDiving, STARbenchmark) and zero-shot benchmarks (MSVD-QA, MSRVTT-QA, TGIF-QA, ActivityNet-QA).

Adapted Datasets For adapted datasets (Tab. 4.3), excluding Label Rationalization led to a significant performance drop in FineDiving, from 20.2 to 12.8 in accuracy, highlighting its critical role in complex reasoning tasks. This is likely due to the lack of conversion of any examples from the data. However, the removal of Answer Generation resulted in a more pronounced and uniform decline across all datasets. For example, Kinetics700’s accuracy was reduced from 59.9 to 50.0, underscoring its foundational role in generating context-relevant responses.

Zero-shot benchmarks In zero-shot benchmarks (Tab. 4.3), the removal of Label Rationalization had a mixed impact, slightly affecting MSVD-QA where accuracy decreased from 71.3 to 70.6. The elimination of Answer Generation consistently lowered performance, such as a decrease in MSRVTT-QA accuracy from 58.2 to 57.4. ActivityNet-QA performance improved, probably because 100K-Instruct utilizes ActivityNet for instruction tuning. Therefore, the introduction of additional videos decreases performance.

- 5 RELATED WORKS

- 5.1 LARGE VISION-LANGUAGE MODELS

Initial LVLMs, such as LLaVA (Liu et al., 2023b;a) and BLIP-2 (Li et al., 2023a), demonstrated the potential of merging image inputs with large language models. Methods like mPLUG-Owl (Ye et al., 2023) and Flamingo (Alayrac et al., 2022) further allowed for multiple image inputs without architectural changes. Li et al. (2023b) and Zhang et al. (2024) led the transition to video understanding, integrating video/image encoders and LLMs while training on small video instruction tuning datasets. Jin et al. (2023) introduced Chat-UniVi, a unified model employing dynamic visual tokens for both images and videos, optimizing visual token usage and higher frame count sampling. LLaMA-VID (Li et al., 2023c) showed that the token count can be further reduced by pooling the tokens selectively via the text prompt using Q-Former. Recently, Video-LLaVA (Lin et al., 2023)

used modality-specific encoders for video and image inputs to leverage LanguageBind encoders as they are constructively aligned during pretraining and utilized a shared projection.

Maaz et al. (2023) expanded the field with the first large video instruction tuning dataset, VideoInstruct-100K. VideoInstruct-100K was generated from ActivityNet (Heilbron et al., 2015) by prompting chatGPT with the video captions, generating question-answer pairs. While driving much of the performance improvement in the field (Jin et al., 2023; Wang et al., 2023; Lin et al., 2023; Li et al., 2023c), upon examination of VideoInstruct-100K, it is evident that it suffers from quality issues. The questions often degrade into de facto prompts for a video caption (see Fig. 10) and rarely require many spatiotemporal capabilities, which may limit LVLM performance.

- 5.2 LARGE LANGUAGE MODELS AND SELF-TRAINING

The advent of GPT (Radford et al., 2018; Brown et al., 2020) marked significant milestones in natural language processing, showcasing LLMs’ power in understanding and generating human-like text. Open-source LLMs like LLaMA (Touvron et al., 2023a;b) and their instruction-tuned variants like Alpaca and Vicuna (Taori et al., 2023; Chiang et al., 2023) further tailored these models for nuanced human-AI interactions. However, even LLMs have found it challenging to scale annotated datasets for training, prompting work on self-training and self-improvement (Singh et al., 2023; Huang et al., 2022; Ho et al., 2023; Marasovi´c et al., 2022; Hosseini et al., 2024). In this line of work, LLMs cycle between instruction-tuning data generation and instruction tuning, iteratively improving LLM performance over cycles. For instance, Zelikman et al. (2022) introduced the the Self-Taught Reasoners method, used rationalization to generate chain-of-thought (CoT) reasoning, filtering poor rationalizations to retain correctly answered questions. Other self-training approaches include expectation-maximization-based approaches (Singh et al., 2023), which alternate between data generation and improvement between training cycles. Alternatively, majority voting has also been utilized to generate answers and rationale for unlabeled questions (Huang et al., 2022). These methods show the effectiveness of iterative self-training. In our work, we aim to introduce a weakly supervised self-training approach for video instruction tuning, leveraging video supervision that is often easier to collect and exists in many large and diverse datasets.

- 6 CONCLUSIONS In conclusion, Video Self-Taught Reasoners (Video-STaR) presents a novel approach to enhance Large Vision-Language Models (LVLMs) by enabling the use of diverse labeled video datasets for visual instruction tuning. This method addresses critical data diversity and quality challenges, leading to significant performance improvements across various video understanding tasks. Our experiments demonstrate Video-STaR’s effectiveness in both source dataset adaptation and zero-shot generalization, showcasing its potential in advancing LVLM capabilities for complex video content.

The promising results of Video-STaR open new research avenues, particularly in expanding LVLM knowledge bases using readily available image and video datasets. Future work could explore advanced self-training techniques and integration with emerging LVLM architectures, focusing on long-form video understanding to further boost LVLM understanding. Additional work is also needed to reduce hallucinations, perhaps by using grounded VLMs as auxiliary input.

Acknowledgements. We thank Google Research for providing financial support through a Stanford HAI–Google collaboration. This work was partially supported by the National Science Foundation under Grant No. 2026498. We thank the Knight-Hennessy Scholars Foundation for funding OZ.

- 7 LIMITATIONS

While Video-STaR introduces a novel approach to visual instruction tuning, it is not without its limitations. Firstly, the methodology can be computationally intensive due to the cycling of both generating and rationalizing question-answer and instruction tuning. Secondly, the assumption that all labels necessitate a rationale may not always hold true. Certain labels might be straightforward enough not to require elaborate rationalization, potentially leading to unnecessary computational overhead. Lastly, hallucinations, especially in label rationalization can be further reduced by perhaps implementing additional verifiers.

REFERENCES

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning, 2022.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https: //lmsys.org/blog/2023-03-30-vicuna/.

Adam Cohen. seatgeek/fuzzywuzzy : Release 0.18.0, February 2020. URL https://pypi. org/project/fuzzywuzzy/.

Jiawen Deng, Kiyan Heybati, and Matthew Shammas-Toma. When vision meets reality: Exploring the clinical applicability of gpt-4 with vision, 2024.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, Eugene Byrne, Zach Chavis, Joya Chen, Feng Cheng, Fu-Jen Chu, Sean Crane, Avijit Dasgupta, Jing Dong, Maria Escobar, Cristhian Forigua, Abrham Gebreselasie, Sanjay Haresh, Jing Huang, Md Mohaiminul Islam, Suyog Jain, Rawal Khirodkar, Devansh Kukreja, Kevin J Liang, Jia-Wei Liu, Sagnik Majumder, Yongsen Mao, Miguel Martin, Effrosyni Mavroudi, Tushar Nagarajan, Francesco Ragusa, Santhosh Kumar Ramakrishnan, Luigi Seminara, Arjun Somayazulu, Yale Song, Shan Su, Zihui Xue, Edward Zhang, Jinxu Zhang, Angela Castillo, Changan Chen, Xinzhu Fu, Ryosuke Furuta, Cristina Gonzalez, Prince Gupta, Jiabo Hu, Yifei Huang, Yiming Huang, Weslie Khoo, Anush Kumar, Robert Kuo, Sach Lakhavani, Miao Liu, Mi Luo, Zhengyi Luo, Brighid Meredith, Austin Miller, Oluwatumininu Oguntola, Xiaqing Pan, Penny Peng, Shraman Pramanick, Merey Ramazanova, Fiona Ryan, Wei Shan, Kiran Somasundaram, Chenan Song, Audrey Southerland, Masatoshi Tateno, Huiyu Wang, Yuchen Wang, Takuma Yagi, Mingfei Yan, Xitong Yang, Zecheng Yu, Shengxin Cindy Zha, Chen Zhao, Ziwei Zhao, Zhifan Zhu, Jeff Zhuo, Pablo Arbelaez, Gedas Bertasius, David Crandall, Dima Damen, Jakob Engel, Giovanni Maria Farinella, Antonino Furnari, Bernard Ghanem, Judy Hoffman, C. V. Jawahar, Richard Newcombe, Hyun Soo Park, James M. Rehg, Yoichi Sato, Manolis Savva, Jianbo Shi, Mike Zheng Shou, and Michael Wray. Ego-exo4d: Understanding skilled human activity from first- and third-person perspectives. arXiv preprint arXiv:2311.18259, 2023.

Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 961–970, 2015. doi: 10.1109/CVPR.2015. 7298698.

Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. arXiv preprint arXiv:2212.10071, 2023.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-star: Training verifiers for self-taught reasoners, 2024.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. ArXiv, abs/2210.11610, 2022. URL https: //api.semanticscholar.org/CorpusID:253080328.

Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatiotemporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2758–2766, 2017.

Yixing Jiang, Jesutofunmi A Omiye, Cyril Zakka, Michael Moor, Haiwen Gui, Shayan Alipour, Seyed Shahabeddin Mousavi, Jonathan H Chen, Pranav Rajpurkar, and Roxana Daneshjou. Evaluating general vision-language models for clinical medicine. medRxiv, pp. 2024–04, 2024.

Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046, 2023.

Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865, 2024.

John R Lawson, Montgomery L Flora, Kevin H Goebbert, Seth N Lyman, Corey K Potvin, David M Schultz, Adam J Stepanek, and Joseph E Trujillo-Falc´on. Pixels and predictions: Potential of gpt-4v in meteorological imagery analysis and forecast communication. arXiv preprint arXiv:2404.15166, 2024.

Vladimir I. Levenshtein. Binary codes capable of correcting deletions, insertions, and reversals. Soviet physics. Doklady, 10:707–710, 1965. URL https://api.semanticscholar.org/ CorpusID:60827152.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 19730–19742. PMLR, 23–29 Jul 2023a. URL https://proceedings.mlr.press/v202/li23q.html.

Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023b.

Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models, 2023c.

Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv: 2403.00476, 2024.

Zelun Luo, Zane Durante, Linden Li, Wanze Xie, Ruochen Liu, Emily Jin, Zhuoyi Huang, Lun Yu Li, Jiajun Wu, Juan Carlos Niebles, et al. Moma-lrg: language-refined graphs for multi-object multi-actor activity parsing. Advances in Neural Information Processing Systems, 35:5282–5298, 2022.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.

Ana Marasovi´c, Iz Beltagy, Doug Downey, and Matthew E. Peters. Few-shot self-rationalization with natural language prompts. arXiv preprint arXiv:2111.08284, 2022.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ radford21a.html.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Senthujan Senkaiahliyan, Augustin Toma, Jun Ma, An-Wen Chan, Andrew Ha, Kevin R An, Hrishikesh Suresh, Barry Rubin, and Bo Wang. Gpt-4v (ision) unsuitable for clinical care and education: a clinician-evaluated assessment. medrxiv. Preprint posted online on November, 16, 2023.

Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond human data: Scaling self-training for problem-solving with language models, 2023.

Lucas Smaira, Jo˜ao Carreira, Eric Noland, Ellen Clancy, Amy Wu, and Andrew Zisserman. A short note on the kinetics-700-2020 human action dataset, 2020. URL https://deepmind.com/ research/open-source/kinetics.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild, 2012.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn,

Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Yizhou Wang, Ruiyi Zhang, Haoliang Wang, Uttaran Bhattacharya, Yun Fu, and Gang Wu. Vaquita: Enhancing alignment in llm-assisted video understanding. arXiv preprint arXiv:2312.02310, 2023.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=gEZrGCozdqR.

Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B. Tenenbaum, and Chuang Gan. STAR: A benchmark for situated reasoning in real-world videos. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https: //openreview.net/forum?id=EfgNF5-ZAjM.

Wenhao Wu. Freeva: Offline mllm as training-free video assistant. arXiv preprint arXiv:2405.07798, 2024.

Angchi Xu, Ling-An Zeng, and Wei-Shi Zheng. Likert scoring with grade decoupling for long-term action assessment. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3222–3231, 2022a. doi: 10.1109/CVPR52688.2022.00323.

Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In ACM Multimedia, 2017.

Jinglin Xu, Yongming Rao, Xumin Yu, Guangyi Chen, Jie Zhou, and Jiwen Lu. Finediving: A finegrained dataset for procedure-aware action quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2949–2958, June 2022b.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023.

Xumin Yu, Yongming Rao, Wenliang Zhao, Jiwen Lu, and Jie Zhou. Group-aware contrastive regression for action quality assessment. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7919–7928, October 2021.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. STar: Bootstrapping reasoning with reasoning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_3ELRdg2sgI.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023a.

Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. LLaMA-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=d4UiXAHN2W.

Shiyi Zhang, Wenxun Dai, Sujia Wang, Xiangwei Shen, Jiwen Lu, Jie Zhou, and Yansong Tang. Logo: A long-form video dataset for group action quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2405–2414,

- June 2023b.

Shiyi Zhang, Wenxun Dai, Sujia Wang, Xiangwei Shen, Jiwen Lu, Jie Zhou, and Yansong Tang. Logo: A long-form video dataset for group action quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2405–2414,

- June 2023c.

Ziqiang Zheng, Yiwei Chen, Jipeng Zhang, Tuan-Anh Vu, Huimin Zeng, Yue Him Wong Tim, and Sai-Kit Yeung. Exploring boundary of gpt-4v on marine analysis: A preliminary case study. arXiv preprint arXiv:2401.02147, 2024.

Xingcheng Zhou and Alois C Knoll. Gpt-4v as traffic assistant: An in-depth look at vision language model on complex traffic events. arXiv preprint arXiv:2402.02205, 2024.

A APPENDIX

In Sec. A.1, we provide additional implementation details and compute used in developing VideoSTaR. In Sec. A.2, we introduce explainable action quality assessment and provide good and bad examples of Video-STaR on the FineDiving test dataset. Finally, we provide additional qualitative Answer Generation and Label Rationalization examples in Sec. A.3 and A.4.

- A.1 IMPLEMENTATION DETAILS

Video-STaR utilized the Video-LLaVA model, which integrated the Vicuna-7B v1.5 for language processing and ViT-L/14 video and image encoders from LanguageBind for visual encoding. The system’s tokenizer, adapted from LLaMA, has a vocabulary size of around 32,000 classes and a dimensionality of 4096. Two cycles of Video-STaR were executed, each initialed with the pre-trained Video-LLaVA model (before instruction tuning). The training data was augmented by incorporating VideoInstruct-100K and LLaVA v1.5’s visual instruction datasets.

Four clusters of 10 NVIDIA Titan RTX GPUs were employed for 64 hours. The structured prompts for these tasks were as follows:

- • Answer Generation

Question: {Q}. Can you explain step-by-step how one can arrive at this conclusion?

- • Label Rationalization

Question: {Q} Answer: {L}. Can you explain step-by-step how one can arrive at this conclusion?

These prompts guided the model in producing detailed answers and rationalizations, enhancing the depth and utility of the generated instruction-tuning dataset. Answer correctness was evaluated using template matching with Levenshtein Distance-based Levenshtein (1965) fuzzy logic Cohen (2020), considering an answer correct if all keywords from the label were present in the generated response with a minimum similarity score of 80%. For example, if in Kinetics the action label is ‘eating apple pie’, we would only consider a generated answer correct of ‘eating’, ‘apple’, ‘pie’ all appeared with a similarity score of 80.

- A.2 EXPLAINABLE ACTION QUALITY ASSESSMENT

Explainable Action Quality Assessment (AQA) is critical for detailed analysis of performances in precision sports, such as competitive diving, where execution and complexity significantly impact scores. Unlike previous AQA works, which only provide a score Xu et al. (2022b;a); Yu et al. (2021); Zhang et al. (2023b) Video-STaR not only generates scores but also offers detailed justifications akin to expert analysis.

Fig. 5 illustrates instances where Video-STaR’s predictions align closely with established scoring criteria in diving. For example, the model breaks down a dive sequence, Reverse→3.5 Soms.Tuck→Entry, with an assigned difficulty of 3.5, into its components. It then logically assigns scores to each element, such as 1 for more straightforward maneuvers and 4 for more complex twists, culminating in an overall difficulty score of 3.3, close to the GT 3.5. The final predicted score was 79.2, close to the GT score of 85.78.

Video-STaR’s proficiency extends to dives with varying levels of performance. It can discern relatively complex dives with ground-truth scores of 74.8 and 54.6, which Video-STaR scored as 76.5 and 47.0, respectively. In both cases, the model breaks down the actions into sub-actions and rates them in terms of difficulty and execution. While it manages to rate the dives themselves well, in one instance, the model erroneously calculated the average for 2.5,3.4,3.4,3.4 as 3.4.

[Figure 47]

[Question] Could you appraise the structure and performance of the sequence in this video on a scale of 1 to 100?

Sure, I'd be happy to explain the reasoning behind the final label. The first step is to evaluate the action sequence. In this case, the sequence is Back->3.5 Soms.Tuck->Entry. The first action, Back, is a basic backward somersault, which is a 1 on the difficulty scale. The second action, 3.5 Soms.Tuck, is a more advanced move that involves a series of flips and twists, which is a 4 on the difficulty scale. Finally, the third action, Entry, is the most difficult part of the sequence, as it involves diving into the water from a high platform. This is a 5 on the difficulty scale. Next, we need to evaluate the action difficulty. The difficulty scale ranges from 1 to 10, with 1 being the easiest and 10 being the most difficult. In this case, the difficulty scale is 3.3. Finally, we need to combine the action sequence and action difficulty to arrive at an overall score. In this case, the overall score is 79.2. Overall, the conclusion is that the sequence in this video is quite impressive, with a high level of difficulty and execution.

[Figure 48]

###### [Label] : Overall Score 85.78. [Additional Labels]: Action Sequence: Reverse->3.5 Soms.Tuck->Entry. Action Difficulty: 3.5.

[Figure 49]

[Question] How would you rate the complexity and accuracy of the performance in this video on a scale from 1 to 100?

Sure, I'd be happy to explain the reasoning behind the score. First, we need to identify the action sequence performed by the diver. In this case, the diver starts by diving forward, then performs a 2.5 somersaults pike, followed by two twists, and finally, an entry. Next, we need to determine the action difficulty of each part of the sequence. The forward dive is generally considered a basic dive, so we'll give it a score of 2.5. The 2.5 somersaults pike is a more advanced move, so we'll give it a score of 3.4. The two twists are also a more advanced move, so we'll give them a score of 3.4. Finally, the entry is the most advanced part of the sequence, as it requires the diver to execute a smooth entry into the water. We'll give it a score of 3.4. Finally, we need to calculate the overall score for the action sequence. We'll take the average of the scores for each part of the sequence, which is (2.5 + 3.4 + 3.4 + 3.4) / 4 = 3.4. So, the final label for this action sequence is Overall Score 76.5. Answer: Reasoning: Action Sequence: Forward->2.5 Soms.Pike->2 Twists->Entry, action difficulty: 3.4 Final Label: Overall Score 76.5.

[Figure 50]

###### [Label] : Overall Score 74.8. [Additional Labels]: Action Sequence: Forward->2.5 Soms.Pike->2 Twists->Entry. Action Difficulty: 3.4.

[Figure 51]

[Question] How would you rate the complexity and accuracy of the performance in this video on a scale from 1 to 100?

Sure, I'd be happy to explain the reasoning behind the score. First, we need to identify the specific actions performed by the diver in the video. We can see that the diver performs a backflip, followed by a 2.5 somersaults, and then a 1.5 twists before entering the water. Next, we need to evaluate the precision and complexity of each action. The backflip is a relatively simple action, but the entry into the water adds an element of difficulty. The 2.5 somersaults are more complex, requiring precise timing and body control. The 1.5 twists are even more complex, requiring a high level of skill and precision. Finally, we need to assign a score to each action based on the level of difficulty. For the backflip, we can assign a score of 2.5. For the 2.5 somersaults, we can assign a score of 3.0. For the 1.5 twists, we can assign a score of 4.0. Adding up the scores for each action, we get a total score of 9.0. Finally, we need to assign an overall score to the video based on the total score. We can use the overall score to evaluate the overall quality and execution of the video. In this case, we can assign an overall score of 47.0 to the video based on the precision and complexity of the actions performed. Answer: Reasoning: Action Sequence: Back->2.5 Soms.Pike->1.5 Twists->Entry, action difficulty: 3.0 Final Label: Overall Score 47.0

[Figure 52]

###### [Label] : Overall Score 54.6. [Additional Labels]: Action Sequence: Back->2.5 Soms.Tuck->Entry. Action Difficulty: 2.8.

- Figure 5: Action Quality Assessments by Video-STaR on the FineDiving Test Set. Different diving sequences with corresponding Video-STaR evaluations, from a high score of 85.78 for complex sequences (top) to 74.8 for intermediate (middle), and a lower score of 54.6 for basic sequences (bottom), showcasing Video-STaR’s proficiency in scoring dives with varying degrees of difficulty and execution quality.

Challenges in maintaining consistent AQA accuracy are depicted in Fig. 6, showcasing instances of either the model not following instructions or estimating the score incorrectly. In Fig. 6, top, the model did not produce a score and resorted to explaining how a score might be derived. The model occasionally produces an inaccurate score, and either grossly over (Fig. 6, middle) or under (Fig. 6,

[Question] How would you evaluate the magnitude of complexity and performance in the displayed action sequence?

[Figure 53]

The difficulty of the action sequence is evaluated based on the number of movements, the level of difficulty of each movement, and the level of precision required to execute the sequence. The sequence is then scored based on this evaluation.

###### [Label] : Overall Score 52.5. [Additional Labels]: Action Sequence: Back->2.5 Soms.Pike->Entry. Action Difficulty: 3.0.

[Question] How would you rate the complexity and accuracy of the performance in this video on a scale from 1 to 100?

Sure, I'd be happy to explain the reasoning behind this conclusion. First, we need to identify the action sequence performed by the diver. In this case, the diver performs an inward somersault, followed by a 3.5 somersault tuck, and then an entry into the water. Next, we need to assign a difficulty score to each action in the sequence. In this case, we'll give the inward somersault a score of 3.5, the 3.5 somersault tuck a score of 3.2, and the entry a score of 3.5. Finally, we need to calculate the overall score for the action sequence. To do this, we'll take the average of the difficulty scores for each action in the sequence. In this case, the average score is 3.2. Based on this reasoning, we can conclude that the overall score for the action sequence is 72.0. Answer:Reasoning: Action Sequence: Inward->3.5 Soms.Tuck->Entry, action difficulty: 3.2 Final Label: Overall Score 72.0.

[Figure 54]

[Label] : Overall Score 35.0.

- [Additional Labels]: Action Sequence: Back->2.5 Soms.Tuck->Entry. Action Difficulty: 2.8.

[Figure 55]

[Figure 56]

Sure, I'd be happy to explain the reasoning behind the grading. First, we need to identify the action sequence performed by the diver. In this case, the diver performs a reverse dive, followed by a 3.5 somersaults, and then a tuck before entering the water. Next, we need to determine the action difficulty of the sequence. The action difficulty is a measure of how difficult it is to execute the sequence. In this case, the action difficulty is 3.5. Finally, we need to determine the overall score for the action sequence. The overall score is based on the difficulty of the action sequence and the quality of its execution. In this case, the overall score is 34.5. Overall, the conclusion is that the diver's action sequence is quite impressive, with a high level of difficulty and execution quality.

[Question] How well would you grade the action sequence based on its difficulty and the execution quality?

[Label] Overall Score 67.65.

- [Additional Labels]: Action Sequence: Back->3.5 Soms.Tuck->Entry. Action Difficulty: 3.3.

[Figure 57]

[Figure 58]

- Figure 6: Challenges in Action Quality Assessment by Video-STaR on the FineDiving Test Set. Illustrating scenarios where Video-STaR diverges from optimal scoring: a complete absence of prediction (top), an overestimation of a dive’s complexity and execution (middle), and underestimation (bottom).

bottom) estimates the score. For example, it attributed a score of 72.0 to a sequence deemed to have a lower difficulty level of 35.0. These discrepancies underscore the necessity for ongoing improvements in the model’s grasp of nuanced scoring criteria to enhance reliability in AQA predictions.

In summary, Video-STaR enhances Action Quality Assessment (AQA) by supplementing scores with detailed rationales, an advancement over traditional AQA approaches that only provide numerical scores. Although it effectively dissects the components of diving performances, indicating both complexity and execution as seen in Fig. 5, it faces challenges in maintaining consistent accuracy, particularly in aligning scores with established benchmarks, as evidenced in Fig. 6. These instances highlight the need for a deeper understanding of each movement’s difficulty and execution quality to ensure the model’s scoring aligns with professional judging standards. Additionally, they emphasize the ongoing requirement to improve the model’s accuracy and ground its rationalizations in the verifiable aspects of the performances it evaluates.

###### Kinetics700

[Figure 59]

[Input Instruction] What specific action is being performed in this video? Can you explain step-by-step how one can arrive at this conclusion?

Yes, in the video, a man is pretending to be a statue on a small table. He is holding a sign that says "I'm not a statue, I'm a performer." The man's posture and facial expressions indicate that he is fully engaged in the act of pretending to be a statue. Additionally, the fact that he is holding a sign suggests that he is trying to convey a message or entertain passersby. Overall, the video showcases a unique and creative performance that is sure to captivate and amuse viewers.

[Figure 60]

[Label] pretending to be a statue.

[Figure 61]

[Input Instruction] What actions are being performed in this video? Can you explain step-by-step how one can arrive at this conclusion?

[Figure 62]

Yes, in the video, a man is seen walking towards the camera while holding a phone. He then proceeds to text while walking. Later, he is shown driving a car while texting. The video also shows a man in a red shirt walking towards the camera, and another man in a red shirt walking towards a car. Additionally, a man in a red shirt is seen walking towards a car with a camera.

###### [Label] texting.

STAR-benchmark

[Figure 63]

[Input Instruction] Which object was eaten by the person? Can you explain step-by-step how one can arrive at this conclusion?

[Figure 64]

Yes, in the video, the person is seen holding a medicine in their hand and then proceeds to eat it. The video does not provide any information about the type of medicine or its appearance. However, we can assume that the person is taking the medicine to alleviate some sort of discomfort or illness.

###### [Label] : The medicine.

[Figure 65]

[Input Instruction] Which object was sat at by the person? Can you explain step-by-step how one can arrive at this conclusion?

[Figure 66]

Yes, in the video, the person is seen sitting at a table with a plate of food. The camera zooms in on the plate of food, and we can see that it is a white plate with a slice of cake on it. The person then proceeds to eat the cake using his hands. From this, we can conclude that the person is sitting at a table with a plate of food, specifically a slice of cake.

[Label] The table.

FineDiving

[Figure 67]

[Input Instruction] Examine this video, and allocate a score between 1 and 100 for the complexity and efficiency of the performance. Can you explain step-by-step how one can arrive at this conclusion?

Sure, I'd be happy to explain the reasoning behind the score. First, we need to identify the action sequence performed by the diver. In this case, the diver performs an inward somersault, followed by a 2.5 somersault, and then a pike. This sequence is then followed by an entry into the water. Next, we need to evaluate the action difficulty of each movement. The inward somersault is a basic movement, so we'll give it a score of 2.5. The 2.5 somersault is a more complex movement, so we'll give it a score of 3.0. The pike is a more advanced movement, so we'll give it a score of 4.0. Finally, the entry into the water is a basic movement, so we'll give it a score of 2.5. Finally, we need to combine these scores to arrive at an overall score for the action sequence. We'll add up the scores for each movement and divide by the total number of movements to get an average score. In this case, the average score is 3.0. Based on this reasoning, we can conclude that the diver's performance is of medium complexity and efficiency, with a score of 63.0 out of 100. Action Sequence: Inward->2.5 Soms.Pike->Entry, action difficulty: 3.0 Final Label: Overall Score 63.0

[Figure 68]

[Label] : Overall Score 63.0. [Additional Labels]: Action Sequence: Inward->2.5 Soms.Pike->Entry. Action Difficulty: 3.0.

- Figure 7: Answer Generation Across Datasets by Video-STaR. Interpreting actions on Kinetics700 (top), detailing action sequences in STAR-benchmark (middle), and evaluating diving action quality in FineDiving (bottom).

STAR-benchmark

[Figure 69]

[Input Instruction] Question: What did the person do while they were lying on the sofa/couch? Answer: Lied on the sofa/couch. Can you explain step-by-step how one can arrive at this conclusion?

[Figure 70]

Sure, in the video, the person is seen lying on the sofa/couch. They then sit up and grab a blanket. After that, they stand up and grab a shirt. They then proceed to put the shirt on the sofa/couch. Finally, they grab the blanket again and put it on the sofa/couch. Based on this information, it can be concluded that the person was likely getting ready to change their clothes or clean the sofa/couch.

###### [Label] : Lied on the sofa/couch.

[Figure 71]

[Input Instruction] Question: What did the person do with the box? Answer: Closed. Can you explain step-by-step how one can arrive at this conclusion?

[Figure 72]

Yes, in the video, the person is seen sitting in a chair and opening a box. After opening the box, they proceed to close it.

###### [Label] Closed.

FineDiving

[Figure 73]

[Input Instruction] Question: Can you rank the level of complexity and execution in this video from 1 to 100? Answer: Action Sequence: Reverse->3.5 Soms.Tuck->Entry, action difficulty: 3.5. Final Label: Overall Score 71.4. Can you explain step-by-step how one can arrive at this conclusion?

Sure, I'd be happy to explain the reasoning behind the final label. First, we need to identify the action sequence performed by the divers in the video. The action sequence in this video is Reverse->3.5 Soms.Tuck->Entry. Next, we need to determine the action difficulty of each movement. The action difficulty is a measure of how difficult it is to execute each movement. In this case, the action difficulty is 3.5. Finally, we need to calculate the overall score based on the action sequence and action difficulty. The overall score is calculated by multiplying the action sequence score by the action difficulty score. In this case, the overall score is 71.4. So, the conclusion is that the divers in the video performed a complex action sequence with a high level of execution, earning an overall score of 71.4 out of 100.

[Figure 74]

[Label] : Overall Score 71.4. [Additional Labels]: Action Sequence: Reverse->3.5 Soms.Tuck->Entry. Action Difficulty: 3.5.

- Figure 8: Successful Label Rationalizations by Video-STaR. The model accurately infers a preparatory behavior from a person’s interaction with household items (top), correctly interprets box-handling actions (middle), and provides a nuanced breakdown of a complex diving sequence, assigning an overall score of 71.4 (bottom), exemplifying precise action understanding on FineDiving and STAR-benchmark.

- A.3 QUALITATIVE ANALYSIS OF ANSWER GENERATION

Fig. 7 presents our model’s capabilities in generating answers and rationalizing actions across varied video contexts, demonstrating its adeptness in interpreting complex scenes.

Kinetics700. In the first example from Kinetics700 (Fig. 7, Kinetics700, top), the model effectively identifies a man’s act of pretending to be a statue and further discerns the performance’s subtle aspects, such as the engagement level and the humor conveyed through a sign. In another Kinetics700 example (Fig. 7, Kinetics700, bottom), the model processes a scene with multiple concurrent actions. Video-STaR first correctly identifies the man in the red shirt talking towards the camera while holding a phone and texting. It correctly identifies the next scene, where another man is texting while driving. This precision in temporally locating different actions in the videos is invaluable for visual instruction tuning and could potentially enhance models’ The capability to analyze scenes with multiple focal points is an essential feature for comprehensive video understanding.

STAR-benchmark

[Figure 75]

STAR-benchmark

[Input Instruction] Question: Which object was put down by the person? Answer: The bag. Can you explain step-by-step how one can arrive at this conclusion?

[Figure 76]

Sure, in the video, the person is seen sitting on a red sofa and talking on the phone. He then stands up and walks towards a table where he takes a dish and puts it on the table. He then proceeds to wash the dish and places it on the rack to dry.

###### [Label] : Took.

[Figure 77]

[Input Instruction] Question: Which object was opened by the person? Answer: The closet/cabinet. Can you explain step-by-step how one can arrive at this conclusion?

Yes, in the video, the person is seen opening a closet/cabinet. The camera then pans to the right, revealing the person's reflection in the mirror. The person is seen bending down and picking up a pair of shoes. The camera then pans to the left, showing the person's reflection in the mirror again. The person then puts on the shoes and walks away. From this information, we can conclude that the person opened the closet/cabinet to get their shoes.

[Figure 78]

###### [Label] The closet/cabinet.

FineDiving

[Figure 79]

[Input Instruction] Question: Could you appraise the structure and performance of the sequence in this video on a scale of 1 to 10? Answer: Reasoning: Action Sequence: Inward->3.5 Soms.Tuck->Entry, action difficulty: 3.2 Final Label: Overall Score 60.8. Can you explain step-by-step how one can arrive at this conclusion?

Sure, I'd be happy to explain the reasoning behind the final label. The first step is to evaluate the action sequence. In this case, the sequence is Inward->3.5 Soms.Tuck->Entry. The difficulty of this sequence is 3.2. The second step is to evaluate the overall score. The overall score is based on a combination of factors, including the difficulty of the action sequence, the execution of the moves, and the overall style and artistry of the performance. Based on the information provided, the overall score is 60.8. Overall, the conclusion is that the video showcases a skilled diver performing a series of flips and twists with precision and grace, earning an overall score of 60.8 out of 100.

[Figure 80]

[Label] : Overall Score 60.8. [Additional Labels]: Action Sequence: Inward->3.5 Soms.Tuck->Entry. Action Difficulty: 3.2.

- Figure 9: Challenges in Label Rationalization by Video-STaR. Instances of rationalization errors are shown: an incorrect inference about dishwashing (top) and fabricated details about closet interactions (middle). A rationalization for a diving sequence (bottom) is accurate but demonstrates the model’s vulnerability to hallucinations in complex action sequences within FineDiving and STARbenchmark datasets.

STAR-benchmark. In the first STAR-benchmark example (Fig. 7, STAR- benchmark, top), the model uses inferential reasoning to deduce the purpose behind a person consuming medicine despite the absence of explicit details about the medicine. This instance showcases the model’s ability to apply logical assumptions to fill in informational gaps, a valuable trait for interpreting actions without fully detailed context. In the next example (Fig. 7, STAR-benchmark, bottom), Video-STaR identified additional details, such as the person sitting at the table eating cake.

FineDiving. in Fig. 7, FineDiving, Video-STaR’s approach to evaluating a diving sequence in FineDiving illustrates its proficiency in detailed performance analysis. By deconstructing the dive into individual elements and assessing each for difficulty, the model mirrors the evaluative processes of human judges, providing a comprehensive performance score. This depth of analysis, which

[Figure 81]

- Figure 10: VideoInstruct-100K Qualitative Evaluation. Evaluation of VideoInstruct-100K’s question distribution. A wide gap can be seen between ‘grounding’ questions, which contains ‘where/when’ (bottom) and the top-50 most common questions (top). when analyzing the top-50 most common questions (top), it is clear they are all prompting for video captions.

includes a critique of the dive’s complexity and execution, underscores the model’s utility in contexts requiring nuanced assessment, such as athletic performance evaluation.

These examples from Fig. 7 show how the proposed Answer Generation is capable of creating valuable and informative video question-answer pairs that can be utilized in instruction tuning, highlighting its potential applicability in various domains that demand a deep understanding of video scenes.

- A.4 QUALITATIVE EXAMPLES OF LABEL RATIONALIZATIONS

Label rationalization in Video-STaR is initiated when direct answer generation by the Large Multimodal Model (LMM) fails. Although this method proves beneficial in certain situations, it occasionally leads to the generation of hallucinated content—incorrect details or inferences not supported by the video.

STAR-benchmark. Illustrated in Fig. 8, effective label rationalizations provide added context, enriching the model’s responses. A notable example from the STAR benchmark demonstrates the model’s capacity to build upon a basic action, like lying on a couch, by inferring additional activities, such as donning a shirt and tidying up, hinting at the individual’s subsequent actions. This example illustrates Video-STaR’s ability to navigate ambiguous labels and furnish more nuanced, informative responses, crucial for Visual Instruction Tuning.

However, Label Rationalization is also more prone to hallucinations. Fig. 9 exposes the model’s tendency for hallucinations, mainly where it introduces actions and details not evidenced in the video. In Fig. 9, STAR-benchmark, top; the LMM hallucinated that after taking the dish, the man washed it and placed it on a dry rack, which did not occur in the video. In Fig. 9, STAR-benchmark, bottom; the model hallucinated that the camera panned to the right, that it could see the reflection of the man in the mirror, and that he took out shoes from the closet — none of which occur in the video.

FineDiving. Label rationalization proved especially useful in challenging, domain-expert datasets such as the Olympic diving scoring dataset - FineDiving. Answer Generation initially had zero yield, and Label Rationalization allowed the model to start learning about this challenging task. For

example, in Olympic diving events, to get an overall score for the dive, one removes the top and bottom 2 scores (out of a total of 7), then multiplies this score with the dive difficulty. In Fig. 8, FineDiving, the LMM correctly deduced that the execution score is multiplied by the difficulty. With no supervision, the model correctly deduced the rules, deducing that the final execution score is obtained by multiplying the execution score and the action difficulty.

Fig. 9, FineDiving, Label Rationalization failed to generate an answer with sufficient depth. Rather than providing additional insights, the model resorted to re-iterating the labels.

In summary, Label Rationalization produces shorter, less informative answers than Answer Generation and is more prone to hallucinations. It is primarily utilized so complex datasets, such as FineDiving, can be learned, especially in cases with initial zero yield.

- A.5 EVALUATION OF VIDEO INSTRUCTION TUNING DATASETS

We performed a qualitative evaluation of VideoInstruct-100K and found that the broad majority of the questions essentially prompt the Large Vision-Language model for video captions - see Fig. 10. As can be seen, ∼ 75% of VideoInstruct-100K’s questions are of this type.

