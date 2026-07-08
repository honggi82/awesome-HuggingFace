# arXiv:2501.09484v2[cs.CL]11Mar2025

## Exploring the Inquiry-Diagnosis Relationship with Advanced Patient Simulators

Zhaocheng Liu, Quan Tu†, Wen Ye, Yu Xiao, Zhishou Zhang, Hengfu Cui, Yalun Zhu, Qiang Ju, Shizheng Li, Jian Xie Baichuan Inc. † Gaoling School of Artificial Intelligence, Renmin University of China lio.h.zen@gmail.com, quantu@ruc.edu.cn, {yewen, xiaoyu, zhangzhishou, cuihengfu}@baichuan-inc.com {zhuyalun, liulifeng, lishizheng, richard}@baichuan-inc.com

Correspondence: liulifeng@baichuan-inc.com

### Abstract

depend entirely on inquiries to collect necessary information. This reliance limits a comprehensive evaluation of the patient’s health condition and substantially complicates the diagnostic process.

Recently, large language models have shown great potential to transform online medical consultation. Despite this, most research targets improving diagnostic accuracy with ample information, often overlooking the inquiry phase. Some studies try to evaluate or refine doctor models by using prompt-engineered patient agents. However, prompt engineering alone falls short in accurately simulating real patients. We need to explore new paradigms for patient simulation. Furthermore, the relationship between inquiry and diagnosis remains unexplored. This paper extracts dialogue strategies from real doctor-patient conversations to guide the training of a patient simulator. Our simulator shows higher anthropomorphism and lower hallucination rates, using dynamic dialogue strategies. This innovation offers a more accurate evaluation of diagnostic models and generates realistic synthetic data. We conduct extensive experiments on the relationship between inquiry and diagnosis, showing they adhere to Liebig’s law: poor inquiry limits diagnosis effectiveness, regardless of diagnostic skill, and vice versa. The experiments also reveal substantial differences in inquiry performance among models. To delve into this phenomenon, the inquiry process is categorized into four distinct types. Analyzing the distribution of inquiries across these types helps explain the performance differences. The weights of our patient simulator are available here.

In recent years, large language models (LLMs) have demonstrated remarkable capabilities across various domains and tasks. Notably, models such as OpenAI’s o1 (OpenAI, 2024b) have introduced groundbreaking reasoning abilities by employing techniques akin to an internalized chain-ofthought (Wei et al., 2022) process. Building on the core strengths of general-purpose LLMs, domainspecific models (Tian et al., 2023; Saab et al., 2024; Chen et al., 2024; Zhang et al., 2024; Singhal et al., 2025) tailored for healthcare have also emerged. In the field of clinical medicine, numerous studies (Jin et al., 2021, 2019; Xie et al., 2024; Tu et al., 2024; Schmidgall et al., 2024; Liu et al., 2024b) have validated the performance of these models, suggesting their potential for transformative applications in medical practice. For instance, on the MedQA (USMLE) benchmark (Jin et al., 2021), models like GPT-4 (Achiam et al., 2023) with MedPrompt (Nori et al., 2023), Med-GeminiL 1.0 (Saab et al., 2024), and o1-preview (Xie et al., 2024) have achieved performance levels surpassing those of human experts.

However, most doctor models focus on improving diagnostic accuracy under relatively sufficient information conditions, diverging from the challenges of online consultations. OMC can be divided into two key stages: inquiry and diagnosis. Existing research has largely overlooked the inquiry stage, limiting understanding of its relationship with diagnosis. Some studies (Li et al., 2024b; Tu et al., 2024; Schmidgall et al., 2024; Qiu and Lan, 2024; Li et al., 2024a) simulate clinical environments via prompt-engineered patient agents, yet these agents fail to replicate real patient behaviors. In real-life scenarios, patients often show concern and anxiety about their condition in their

### 1 Introduction

Online medical consultation (OMC) (Al-Mahdi et al., 2015; Kessler, 2023), emerges as a revolutionary medical service that significantly improves the accessibility of healthcare, particularly in areas lacking adequate medical resources. However, compared to traditional face-to-face consultations, online consultations present notable limitations. The absence of direct physical examinations and auxiliary diagnostic tools requires physicians to

[Figure 1]

[Figure 2]

[问主诉]你好，请问你最近有什么不适或者症状吗？ [Main Complaint Inquiry]Hello, have you been experiencing any discomfort or symptoms recently?

D

##### Prompt Engineering with GPT-4o Our Patient Simulator

我左眼很疼，尤其是眼球运动或强光下...放射到额头和下颌。眼睛发红、畏 光...已经持续好几天了。 My left eye is very painful, especially during eye movement or exposure to bright light...The pain radiates to the forehead and lower jaw...The eye is red and sensitive to light...it has been going on for several days now.

[提供信息,表达担忧]我左眼好几天前开始越来越疼，眼球发蓝发红，我非常害 怕，还能治好吗？ [Provide Information,Express Concerns]My left eye's been hurting for days, and it's turning bluish-red. I'm really scared. Can it be treated?

[Figure 3]

[Figure 4]

Inquiry Stage

[问伴随症状]你的症状确实需引起重视...请问，除了疼痛和颜色变化，是否还有 其他症状，比如流泪、眼部分泌物、畏光等？ [Inquiry about Accompanying Symptoms]Your symptoms are worrying... Besides pain and color change, do you have any other symptoms like blurry vision, eye discharge, or light sensitivity?

[Figure 5]

[Figure 6]

[Figure 7]

从您描述的这些症状（...），可能是较严重的眼部问题...请问您的视力是否明显 下降？另外，之前左眼是否受过外伤？ From those symptoms you mentioned (...), it could be a serious eye issue. Is your vision noticeably worse? Also, has your left eye been injured before?

D

[Figure 8]

D

...... ......

[诊断,医疗建议]您目前的症状可能是虹膜睫状体炎、葡萄膜炎，鉴于症状严重 且持续加重，建议尽快到眼科就诊，以下是具体建议：... [Diagnosis,Medical Advice]Your current symptoms might be iridocyclitis or uveitis. Since they're severe and getting worse, you should see an eye specialist ASAP. Here's some advice:...

[Figure 9]

您目前的症状可能是巩膜炎、葡萄膜炎，鉴于症状严重且持续加重，建议尽快 到眼科就诊，以下是具体建议：... Your current symptoms might be scleritis or uveitis. Since they're severe and getting worse, you should see an eye specialist ASAP. Here's some advice:...

[Figure 10]

[Figure 11]

[Figure 12]

D

Diagnosis Stage

[Figure 13]

[Figure 14]

再见，医生！谢谢您！ Okay, thank you, doctor. I'll go as soon as possible! Goodbye!

[停止]谢谢医生，我会注意的，再见！ [Stop]Thanks doc, I'll be careful. Bye!

......

- • The dialogue strategies are highlighted in purple.
- • Emotional responses or active inquiries in our patient simulator replies are highlighted in green.
- • Baselines tend to provide more comprehensive symptom descriptions, reducing the need for inquiries, which does not align with actual scenarios, as highlighted in red.

Figure 1: Our patient simulator (right) is compared to the baseline simulator (prompt engineering with GPT-4o, left) using identical patient records and doctor model.

responses. When describing initial symptoms, they urgently highlight the issues they are most worried about, rather than giving a comprehensive list of all symptoms. Real patients also tend to ask questions actively to ease their emotions. They may disengage if doctors repeat questions. These challenges are difficult to address solely with prompt engineering, making it necessary to explore a new paradigm for simulating patients. Furthermore, current studies offering dynamic simulation environments have yet to thoroughly explore the relationship between inquiry and diagnosis.

In this paper, we extract patient dialogue strategies from real doctor-patient conversations to guide the development of a patient simulator. Initially, we annotate and standardize open-source real doctorpatient conversations using LLMs, and then summarize a set of patient dialogue strategies. We manually select strategies that meet specific criteria, such as ensuring dialogue rounds are complete and excluding follow-up visits in favor of initial consultations. Due to limited usable training data and the absence of medical records, we synthesize doctor-patient dialogues through in-context learning. This synthesis relies on two types of inputs: (1) patient records (similar to context in MedQA), and (2) our curated dialogue strategy set. We train our model entirely on the synthesized dialogues with

corresponding patient records. After a thorough evaluation, our patient simulator shows a notable reduction in hallucination rates, achieving 0.31% compared to the state-of-the-art (SOTA) rate of 3.71%. And there is a significant improvement in anthropomorphism, with our simulator scoring 0.87, outperforming the strongest baseline of 0.31. Although our simulator exhibits a higher rate of unrelated responses at 4.79%, compared to the best baseline of 0.93%, this does not necessarily indicate inferior model performance. This is attributed to real patients often exhibiting some refusal to answer, particularly in the latter part of inquiries. Our patient simulator also reflects this behavior.

Based on our patient simulator, we conduct extensive experiments to explore the relationship between inquiry and diagnosis. Utilizing our patient simulator to fix patient simulations, while interacting with different doctor models for a fixed number of rounds to generate inquiry records. Each inquiry record is diagnosed using various doctor models. Upon analyzing the diagnostic accuracy of the inquiries produced by different doctor models, we find that some models consistently yield inquiries with significantly high or low accuracy, regardless of which doctor model performed the diagnosis. This indicates that there are significant differences in the inquiries generated by

different doctor models. Comparing high-quality and low-quality inquiries and observing the accuracy differences after diagnosis by doctor models with varying diagnostic capabilities, we propose that the inquiry-diagnosis relationship follows Liebig’s law. When inquiry quality is inadequate, even strong diagnostic capabilities are insufficient for achieving good outcomes, and vice versa.

System Prompt

I hope you can help me generate an authentic and complete doctor-patient dialogue process. This should be based on patient information and incorporate strategy tags into the dialogue according to the given strategy flow.

Examples of User Input

### Patient Records {examples_of_patient_records}

### Dialogue Strategy Flow {examples_of_dialogue_strategy_flow}

Examples of Assistant Ouput

To further analyze the differences in inquiry processes among different doctor models, we categorize the inquiries into four types: (1) chief complaint inquiry; (2) specification of known symptoms; (3) inquiry about accompanying symptoms; (4) gathering family or medical history. We calculate the distribution of inquiry records across these four types for different inquiry models. By comparing the distribution differences and diagnostic accuracy, we uncover a certain correlation. For instance, when a model asks more questions to specify known symptoms, resulting in relatively fewer inquiries of other types, the final diagnostic accuracy tends to be lower. Our findings suggest that optimizing the allocation of inquiries within typically 3 to 5 rounds, which patients can comfortably accept, presents a valuable research problem. In summary, our contributions are as follows:

{examples_of_patient_records} User Input

### Patient Records {patient_records}

### Dialogue Strategy Flow {a_randomly_sampled_dialogue_strategy_flow}

Figure 2: Prompts for synthesizing patient simulator training dialogues.

is often used to construct patient agents. Unfortunately, real patient behaviors are difficult to replicate through prompt engineering alone.

In order to simulate real patients as accurately as possible, it is necessary to rely on authentic doctorpatient dialogue datasets. In this paper, we utilize the MedDialog (Zeng et al., 2020) dataset which is distributed under CC BY-NC 4.01. We sample the data to ensure it is thoroughly anonymized and free from any personal identifiers or offensive content. Initially, we conduct essential data screening to remove non-consultative records (e.g., patient scheduling and registration) and to select complete initial consultation dialogues. Then, we manually provide a seed set of commonly used dialogue strategy tags found in doctor-patient interactions. GPT4o (OpenAI, 2024a) is employed to expand this seed set, resulting in a candidate set of dialogue strategy tags (see Appendix A). Based on the candidate set of dialogue strategy tags, GPT-4o is further used to annotate the selected complete initial consultation dialogues. Each dialogue’s tags are concatenated in sequence to form a dialogue strategy flow. Finally, high-quality dialogue strategies are manually selected from the deduplicated set. Examples of selected dialogue strategy flows can be found in Appendix B.

- • We propose a novel patient simulator guided by real dialogue strategies that effectively addresses the limitations of prompt-engineered patient agents by demonstrating higher anthropomorphism, reduced hallucination rates, and dynamic dialogue behavior.
- • Extensive experiments explore the relationship between inquiry and diagnosis, showing that they align with Liebig’s law. Poor inquiry quality constrains diagnostic accuracy, regardless of diagnostic capability, and vice versa.
- • The inquiry process is systematically categorized into four distinct types to analyze discrepancies in model performance, thereby providing insights into the significant differences in inquiry quality across models.

### 2 Patient Simulator

Due to the limited availability of usable patientdoctor dialogue data for training after selection, and the absence of corresponding medical records, we synthesize patient-doctor dialogue data to facilitate the training process. We utilize the Chinese

#### 2.1 Methods

Some studies (Li et al., 2024b; Tu et al., 2024; Schmidgall et al., 2024; Qiu and Lan, 2024; Li et al., 2024a) have attempted to assess or enhance doctor models by creating simulated clinical environments. In these studies, prompt engineering

1https://creativecommons.org/licenses/by-nc/4. 0/

medical record dataset released by CCKS 2019 (Han et al., 2020) as a candidate set of medical records. We sample the data to ensure it is thoroughly anonymized and free from any personal identifiers or offensive content. In each data synthesis iteration, a medical record is randomly selected, and a dialogue strategy flow is randomly chosen from the curated set of dialogue strategy flows. Through in-context learning, we synthesize patient-doctor dialogues that align with the selected dialogue strategy flow. For detailed prompts, please refer to Figure 2. Since the medical records are in Chinese, we synthesize only Chinese dialogues, limiting the simulator’s performance in English scenarios.

The format of this synthetic doctor-patient dialogue is shown on the right side of Figure 1. Each round of conversation between the doctor and the patient is preceded by several dialogue strategy tags. We construct a supervised finetuning (SFT) dataset entirely based on this synthetic doctor-patient dialogue dataset. Specifically, in the training and prediction phases, our patient simulator requires only the input of patient medical records into a simple system prompt (see Appendix C). Given a doctor-patient dialogue {d1,p1,d2,p2,...,dn,pn}, where di represents the i-th round of doctor dialogue and pi represents the i-th round of patient dialogue. We divide it into n SFT data instances, that is,

{d1,p1},{d1,p1,d2,p2},...,{d1,p1,...,dn,pn}. It is important to note that for each SFT data instance, we only retain the dialogue strategy tags for the label (the last turn of the patient dialogue). The strategy tags in the preceding dialogues are removed. This is to align with the estimated scenarios of the patient simulator, as the doctor model is not expected to output our dialogue strategy tags. The model needs to learn to predict the appropriate dialogue strategy and the content to be conveyed in the absence of dialogue strategy tags in the context. The SFT dataset comprises 1000 multi-turn dialogues, with the training set and validation set being divided in an 8:2 ratio. We train the LoRA (Hu et al., 2021) weights of the patient simulator on the Qwen2.5-72B-Instruct (Yang et al., 2024) model. Our learning rate is set to 1e-4, and we utilize DeepSpeed2 for training over 3 epochs, consuming a

2https://github.com/deepspeedai/DeepSpeed

total of 64 GPU hours. Finally, the weights of our patient simulator are available here.

#### 2.2 Evaluation Results

Table 1: Evaluation results of different patient simulators based on our defined Hallucination Rate (HR), Irrelevant Response Rate (IRR) and Anthropomorphism Score (AS). All values are presented as percentages. The final row presents the consistency results, derived from sample checks, between the performance of GPT4o and human evaluations across these three indicators.

Model HR ↓ IRR ↓ AS ↑ Qwen2.5-72B-Instruct 4.97 7.48 28.00 AgentClinic 3.71 0.93 31.00 ours 0.31 4.79 87.00 Alignment with human 99.00 100.00 90.60

We conduct extensive experiments to evaluate the performance of our patient simulator. We design a set of concise and practical patient simulator metrics, primarily including the following three indicators:

- • Hallucination Rate (HR): The proportion of dialogue turns where the patient produces responses contradicting the medical record. By inputting the medical record and each round of dialogue content, GPT-4o assigns a score (0 or 1), and the calculated proportion is evidently better when it is lower.
- • Irrelevant Response Rate (IRR): The proportion of dialogue turns where the patient does not address the questions posed by the doctor model. It involves inputting the doctor’s inquiries and the patient’s responses, with GPT4o assigning a score of 0 or 1. Since a certain level of irrelevant answers is also present in real patients, this metric does not necessarily need to be as low as possible and serves as a reference value during application.
- • Anthropomorphism Score (AS): Analyzing the anthropomorphic behaviors exhibited by the patient agent throughout the dialogue, such as expressions of emotion, proactive questioning, and the degree of colloquialism in responses. It is scored by GPT-4o on a scale from 0 to 1, with values closer to 1 indicating a higher level of anthropomorphism.

Our patient simulator is compared with Qwen72B-Instruct and AgentClinic (Schmidgall et al., 2024). AgentClinic implements patient agents through prompt engineering on GPT-4 and is used to benchmark the simulation effectiveness of our patient simulator. Qwen-72B-Instruct benchmarks our training process. AgentClinic utilizes multiple biased prompts that potentially interfere with HR and IRR outcomes. These biased prompts are excluded, retaining only its core system prompt. To ensure consistency, Qwen2.5-72B-Instruct also adopts the same system prompt.

The experimental results presented in Table 1 demonstrate that our patient simulator significantly outperforms all baselines regarding Hallucination Rate. This improvement largely stems from incorporating patient medical records into the system prompt during training. In contrast, baseline approaches depend solely on prompt engineering. From the perspective of IRR, our method achieves a significantly lower value compared to Qwen2.572B-Instruct. However, the IRR of our method is higher than that of the GPT-4-based AgentClinic. This discrepancy may arise from differences in the underlying foundation models, as well as the selected dialogue strategy flow, where patients may ask questions proactively rather than responding to the doctor’s inquiries. It is important to note that a lower IRR is not necessarily better and should only be considered as a reference metric. Lastly, with respect to AS, our model outperforms all baselines by a significant margin, confirming that our training paradigm is capable of successfully guiding the model to emulate a realistic dialogue strategy flow, resembling real patients. To verify the reliability of the metrics implemented in the prompt engineering of GPT-4o, we conduct a manual random sampling inspection and calculate their consistency with human evaluations. As indicated in the last row of Table 1, our implementation of the three metrics demonstrates sufficient reliability.

### 3 Relationship Between Inquiry and Diagnosis: Impact on Diagnostic Accuracy

#### 3.1 Experimental Setup

The inquiry process typically spans n rounds with various inquiry models to generate inquiry records, where n is generally up to 5 rounds to ensure patient tolerance. We set n to discrete values ranging from 1 to 5, with diagnosis made by different doc-

Table 2: The distribution of the models used for inquiry and diagnosis. The o1-mini and o1-preview are used only for diagnosis due to their stronger reasoning capabilities.

Model Inquiry Diagnosis GPT-4o ✓ ✓

GPT-4o-mini ✓ ✓ claude-3-5-sonnet ✓ ✓

- o1-mini ✗ ✓
- o1-preview ✗ ✓

tor models in the (n + 1)th round. The patient side uses our patient simulator and medical records from AgentClinic’s MedQA-Extend. The specific distribution of the models used for inquiry and diagnosis is shown in Table 2. To facilitate the accurate computation of diagnostic accuracy, we design a workflow (detailed in Appendix F). Tasks in the workflow are among the most common for LLMs and can yield preliminary results even without complex prompts.

#### 3.2 Experimental Results

Our experimental results are presented in Figure 3. Patients consistently utilize our patient simulator, while doctors interact with the simulator using various models for a fixed number of rounds (x-axis, where n values are 1, 2, 3, 4, 5) to generate inquiry records. Subsequently, these records are diagnosed by five different doctor models, as shown in Table 2, and the diagnostic accuracy (y-axis) is computed.

Firstly, we analyze Subfigures 2 to 6, excluding the first Subfigure in the upper left corner of Figure 3. These five Subfigures present the accuracy rates of the same three sets of inquiries processed through five different diagnostic models. By examining each Subfigure individually, it becomes apparent that under the same inquiry rounds and diagnostic models, there are significant differences in the accuracy rates of inquiries generated by different models. For example, in Subfigure 6, after 5 inquiry rounds and under the o1-preview diagnostic model, the accuracies for Claude, GPT-4o and GPT-4o-mini (OpenAI, 2024a) are 0.439, 0.481, and 0.5, respectively. Furthermore, across all five Subfigures, the inquiries produced by the model claude-3-5-sonnet consistently exhibit relatively lower accuracy levels, regardless of the diagnostic model used. These indicate that there are significant differences in inquiry capabilities among

(1) Each model conducts its own inquiry and diagnosis

(3) Diagnosis is uniformly conducted using Claude

(5) Diagnosis is uniformly conducted using GPT 4o

0.50

0.50

0.50

Inquiry Model

Inquiry Model

Inquiry Model

GPT-4o

GPT-4o

GPT-4o

0.45

0.45

0.45

GPT-4o-mini

GPT-4o-mini

GPT-4o-mini

claude-3-5-sonnet

claude-3-5-sonnet

claude-3-5-sonnet

0.40

0.40

0.40

0.35

0.35

0.35

Accuracy

Accuracy

Accuracy

0.30

0.30

0.30

0.25

0.25

0.25

0.20

0.20

0.20

0.15

0.15

0.15

0.10

0.10

0.10

0.05

0.05

0.05

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Inquiry Round

Inquiry Round

Inquiry Round

###### (6) Diagnosis is uniformly conducted using o1 preview

(2) Diagnosis is uniformly conducted using GPT 4o mini

(4) Diagnosis is uniformly conducted using o1 mini

0.50

0.50

0.50

Inquiry Model

Inquiry Model

GPT-4o

GPT-4o

0.45

0.45

0.45

GPT-4o-mini

GPT-4o-mini

claude-3-5-sonnet

claude-3-5-sonnet

0.40

0.40

0.40

0.35

0.35

0.35

Accuracy

Accuracy

Accuracy

0.30

0.30

0.30

0.25

0.25

0.25

0.20

0.20

0.20

0.15

0.15

0.15

Inquiry Model

GPT-4o

0.10

0.10

0.10

GPT-4o-mini

claude-3-5-sonnet

0.05

0.05

0.05

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Inquiry Round

Inquiry Round

Inquiry Round

- Figure 3: Patients consistently use our patient simulator, and doctors initially employ different models to interact with the simulator for fixed n rounds (x-axis, n values are 1, 2, 3, 4, 5) to generate inquiry records. These records are then diagnosed using different doctor models, and the diagnostic accuracy (y-axis) is calculated. Each experiment is conducted three times, and the average accuracy is reported.

#### the different models.

Secondly, by comparing the accuracy rates of the same inquiry rounds and models in Subfigures 2 to 6, we observe that different models exhibit varying diagnostic capabilities. Among them, o1-preview demonstrates the strongest diagnostic ability, while GPT-4o-mini shows the weakest. This result correlates with the inherent reasoning capabilities of the models, aligning with intuitive expectations. By further integrating the performance of diagnostic and inquiry abilities, it is observed that there is no significant correlation between the two. For instance, while GPT-4o-mini exhibits weaker diagnostic capabilities, it performs relatively well in inquiry tasks, whereas GPT-4o demonstrates strong performance in both areas. This observation suggests that when developing medical AI models, if a single model struggles to excel in both inquiry and diagnostic abilities, dividing the tasks into two specialized models could serve as a viable solution.

Thirdly, comparing Subfigure 2 with Subfigures

- 3 to 6 for the same inquiry rounds and models reveals that the accuracy rates in Subfigure 2 are significantly lower than those in Subfigures 3 to

6. This is due to the weaker diagnostic capability

of GPT-4o-mini, leading to a lower ceiling for the final accuracy. Conversely, comparing Subfigure 6 with the others for the same rounds and inquiry models shows that Subfigure 6 surpasses the others in accuracy rates. This is attributed to the superior diagnostic ability of o1-preview, resulting in a higher ceiling. Observations from Subfigures 1 and 3 to 6 indicate that diagnostic accuracy increases significantly with more inquiry rounds. Furthermore, regardless of the diagnostic model used, records based on Claude inquiries consistently perform poorly. Hence, we conclude that inquiry and diagnosis adhere to the Liebig’s law: if the quality of the inquiry is insufficient, achieving good results is challenging even with strong diagnostic capabilities, and vice versa.

### 4 Inquiry Differences Among Models

#### 4.1 Four Types of Inquiry

Based on the examples in our inquiry records and systematic descriptions in relevant medical materials (Trousseau, 1873; Adler, 1997; Bickley, 2012; Swartz, 2014), we categorize the doctors’ inquiries into four types: (1) chief complaint in-

[Figure 15]

你好！最近有什么不适或者症状吗？ Hello! Can you please tell me what symptoms you are experiencing?

Chief Complaint Inquiry

[Figure 16]

我最近一直看东西重影，爬楼梯感觉无力 I've been seeing double lately and feel weak when climbing stairs.

[Figure 17]

Specification of Known Symptoms

这种无力感是否在特定部位，还是在整个身体上都有？ Do you feel weak in a specific area, or all over your body?

[Figure 18]

Inquiry about Accompanying Symptoms

请问你是否有头痛、晕眩或任何其他神经系统的问题？ Do you have headaches, dizziness, or any other nerve issues? 你之前有没有被诊断过其他的肌肉或神经相关的疾病？ Have you ever been diagnosed with any other muscle or nerve related issues?

[Figure 19]

Gathering Family or Medical History

- Figure 4: Examples of four types of inquiry with D representing the doctor and P representing the patient in the figure.

quiry; (2) specification of known symptoms; (3) inquiry about accompanying symptoms; (4) gathering family or medical history, as shown in Figure

- 4. A detailed discussion is in Appendix D.

#### 4.2 Experimental Results

We employ GPT-4o to annotate the inquiry records into above four types, with the prompt used detailed in the Appendix E. Our experimental results are shown in Figure 5.

Firstly, as shown in Subfigure 1 of Figure 5, in the vast majority of cases, all inquiry models choose to ask about the chief complaint during the first round. This aligns with expectations, as the doctor models do not possess any information about the patient during the initial round and thus typically begin with a question such as "What symptoms have you been experiencing that brought you here today?". However, there is a small subset involving inquiries about accompanying symptoms, particularly with the use of GPT-4o-mini and Claude. Such initial questions often include: "Hello, could you tell me if you’ve had any discomfort in recent days, like fever, cough, or any other uneasy feelings?" or "Hello, you seem a bit pale; have you been experiencing any symptoms like dizziness, fatigue, or loss of appetite?". Although whether these instances should be tagged as inquiries about accompanying symptoms remains debatable, the comparison shows that these inquiries indeed interfere with the collection of the patient’s chief complaint. This might be the main reason why GPT-4o consistently performs the best in the first round in subfigures 2 to 6 of Figure 3.

Secondly, as shown in Subfigures 2–5 of Figure 5, Claude demonstrates a significantly higher proportion of specification of known symptoms during multi-turn inquiry compared to other models. This leads to a noticeable reduction in the pro-

portions of other inquiry types. Considering that each type of inquiry is crucial for the diagnostic process, we hypothesize that this might indicate a relative weakness in Claude’s overall inquiry capability compared to other models. Correspondingly, in Subfigures 2–6 of Figure 3, the inquiry records generated by Claude are generally associated with the lowest final diagnostic accuracy. Furthermore, when comparing GPT-4o and GPT-4o-mini, the latter consistently exhibits a higher proportion of gathering family or medical history across turns (except for the fourth turn). Based on Subfigure 6 of Figure 3 (where o1-preview is used as the diagnostic model), the contribution of family history to diagnostic accuracy becomes evident starting from the third turn. The focus on subfigure 6 is motivated by the fact that o1-preview demonstrates the strongest diagnostic capability among all models, allowing us to minimize the confounding effects of different levels of diagnostic performance.

### 5 Related Works

#### 5.1 Large Language Models in Medicine

Large language models (LLMs) in medicine are categorized into two types: general-purpose LLMs and medical-specific models. General-purpose LLMs are further divided into open-source and closed-source categories. Examples of open-source models include LLaMA (Dubey et al., 2024), Qwen (Yang et al., 2024), Mixtral (Jiang et al., 2024), and DeepSeek (Liu et al., 2024a), while closedsource models include GPT-4o (OpenAI, 2024a),

- o1-preview (OpenAI, 2024b), Claude (Anthropic,

2024) and Gemini (Team et al., 2023). The primary goal of optimizing a general-purpose LLMs is to enhance its broad applicability, ensuring strong performance across a variety of tasks, including, naturally, medical tasks. Building on the core strengths

- of general-purpose LLMs, numerous researchers focus on developing specialized models for the medical domain. These models enhance their performance in the medical field through prompt engineering, continual pre-training, supervised finetuning (SFT), and reinforcement techniques. (Tian et al., 2023; Saab et al., 2024; Chen et al., 2024; Zhang et al., 2024; Singhal et al., 2025)

#### 5.2 The Evaluation of Language Models in Medicine

The benchmarks to evaluate LLMs in medicine can be categorized into static and dynamic types based

###### (1) Inquiry Round = 1

###### (2) Inquiry Round = 2

###### (3) Inquiry Round = 3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100

40

35

40

80

30

30

Percentage

Percentage

Percentage

25

60

20

20

40

15

10

10

20

5

0

0

0

GPT-4o GPT-4o-mini claude-3-5-sonnet

GPT-4o GPT-4o-mini claude-3-5-sonnet

GPT-4o GPT-4o-mini claude-3-5-sonnet

Models

Models

Models

###### (4) Inquiry Round = 4

###### (5) Inquiry Round = 5

50

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

40

40

Inquiry Type

Chief Complaint Inquiry

30

30

Percentage

Percentage

Specification of Known Symptoms

20

Inquiry about Accompanying Symptoms

20

Gathering Family or Medical History

10

10

0

0

GPT-4o GPT-4o-mini claude-3-5-sonnet

GPT-4o GPT-4o-mini claude-3-5-sonnet

Models

Models

- Figure 5: The comparison focuses on the distribution of four inquiry types across GPT-4o, GPT-4o-mini, and Claude-3-5-sonnet as inquiry models, segmented by inquiry rounds. The x-axis represents the inquiry models, while the y-axis indicates the proportion of the four inquiry types.

on whether they provide a simulated environment.

Static benchmarks primarily assess medical knowledge and typically use a multiple-choice format. The MedQA (Jin et al., 2021) dataset contains question-answer pairs derived from the US, Mainland China, and Taiwan Medical Licensing Exams. It features 4-5 multiple-choice questions with correct answers. The LLMs receive comprehensive context, including patient history, demographics, and symptoms, to generate responses. And similar multiple-choice formats are employed by PubMedQA (Jin et al., 2019), MedMCQA (Pal et al., 2022), MMLU clinical topics (Hendrycks et al., 2020), and MultiMedQA (Singhal et al., 2023).

Dynamic benchmarks assess the performance of doctor models through role-playing scenarios involving doctors and patients, utilizing LLMs. AMIE (Tu et al., 2024) diagnoses simulated patients through history-taking. AgentClinic (Schmidgall et al., 2024) is an open-source multimodal benchmark designed to assess the capability of LLMs to function as agents in simulated clinical settings. Additionally, many other studies (Li et al., 2024b; Qiu and Lan, 2024; Li et al., 2024a; Johri et al., 2023; Tang et al., 2023) provide

simulated clinical environments to evaluate or enhance physician models. However, in these studies, patient simulations predominantly rely on prompt engineering, which does not accurately replicate real patient behavior. Furthermore, the relationship between inquiry and diagnosis remains unexplored.

### 6 Conclusion

In this paper, we use real doctor-patient dialogue strategies to guide the training of our patient simulator, resulting in a simulation that has significantly fewer hallucinations and more accurately resembles a real patient. Utilizing this simulator for comprehensive experiments uncovers significant differences in inquiry strategies across various models and demonstrates that inquiry and diagnosis adhere to Liebig’s law. We classify inquiries into four categories based on data cases and diagnostic definitions.We label and analyze the distribution of inquiries across four types, identifying specific differences in inquiry strategies based on variations in distribution and diagnostic accuracy. Our results suggest that optimizing the allocation of inquiries within typically 3 to 5 rounds, which are acceptable to patients, presents a valuable research problem.

### Limitations

Due to the lack of multimodal information in opensource doctor-patient dialogue data, our patient simulator does not support sending images or videos. During the selection of dialogues, we retained only initial consultations for simplification, which limits the ability of our patient simulator to effectively simulate follow-up consultations. Additionally, since the available open-source medical records data are in Chinese, we chose to synthesize only Chinese dialogues, which constrains our simulator’s performance in English dialogue scenarios. Finally, this paper does not propose a specific approach to allocate questions within the limited rounds of the inquiry stage, leaving this aspect for future work.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Herhert M Adler. 1997. The history of the present illness as treatment: who’s listening, and why does it matter? The Journal of the American Board of Family Practice, 10(1):28–35.

Ibrahim Al-Mahdi, Kathleen Gray, and Reeva Lederman. 2015. Online medical consultation: A review of literature and practice. In Proceedings of the 8th Australasian workshop on health informatics and knowledge management, volume 164, pages 97–100. Australian Computer Society Sydney.

Anthropic. 2024. Claude 3.5 Sonnet. https://www. anthropic.com/news/claude-3-5-sonnet.

L Bickley. 2012. Bates’ Guide to Physical Examination and History Taking. Lippincott Williams & Wilkins.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. 2024. Huatuogpt-o1, towards medical complex reasoning with llms. arXiv preprint arXiv:2412.18925.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Xianpei Han, Zhichun Wang, Jiangtao Zhang, Qinghua Wen, Wenqi Li, Buzhou Tang, Qi Wang, Zhifan Feng, Yang Zhang, Yajuan Lu, et al. 2020. Overview of the ccks 2019 knowledge graph evaluation track: entity, relation, event and qa. arXiv preprint arXiv:2003.03875.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146.

Shreya Johri, Jaehwan Jeong, Benjamin A Tran, Daniel I Schlessinger, Shannon Wongvibulsin, Zhuo Ran Cai, Roxana Daneshjou, and Pranav Rajpurkar. 2023. Guidelines for rigorous evaluation of clinical llms for conversational reasoning. medRxiv, pages 2023– 09.

Sabrina H Kessler. 2023. Online medical consultation services. The International Encyclopedia of Health Communication, pages 1–4.

Junkai Li, Siyu Wang, Meng Zhang, Weitao Li, Yunghwei Lai, Xinhui Kang, Weizhi Ma, and Yang Liu. 2024a. Agent hospital: A simulacrum of hospital with evolvable medical agents. arXiv preprint arXiv:2405.02957.

Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S Ilgen, Emma Pierson, Pang Wei Koh, and Yulia Tsvetkov. 2024b. Mediq: Questionasking llms and a benchmark for reliable interactive clinical reasoning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024a. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Jie Liu, Wenxuan Wang, Zizhan Ma, Guolin Huang, Yihang SU, Kao-Jung Chang, Wenting Chen, Haoliang Li, Linlin Shen, and Michael Lyu. 2024b. Medchain: Bridging the gap between llm agents and clinical practice through interactive sequential benchmarking. arXiv preprint arXiv:2412.01605.

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, et al. 2023. Can generalist foundation models outcompete special-purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452.

- OpenAI. 2024a. GPT-4o system card. https:// openai.com/index/gpt-4o-system-card/.
- OpenAI. 2024b. Openai o1 system card. https:// openai.com/index/openai-o1-system-card/.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pages 248–260. PMLR.

Huachuan Qiu and Zhenzhong Lan. 2024. Interactive agents: Simulating counselor-client psychological counseling via role-playing llm-to-llm interactions. arXiv preprint arXiv:2408.15787.

Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, et al. 2024. Capabilities of gemini models in medicine. arXiv

- preprint arXiv:2404.18416.

Samuel Schmidgall, Rojin Ziaei, Carl Harris, Eduardo Reis, Jeffrey Jopling, and Michael Moor. 2024. Agentclinic: a multimodal agent benchmark to evaluate ai in simulated clinical environments. arXiv

- preprint arXiv:2405.07960.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature, 620(7972):172–180.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-level medical question answering with large language models. Nature Medicine, pages 1–8.

Mark H Swartz. 2014. Textbook of physical diagnosis E-book: history and examination. Elsevier Health Sciences.

Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. 2023. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint arXiv:2311.10537.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Yuanhe Tian, Ruyi Gan, Yan Song, Jiaxing Zhang, and Yongdong Zhang. 2023. Chimed-gpt: A chinese medical large language model with full training regime and better alignment to human preferences. arXiv preprint arXiv:2311.06025.

Armand Trousseau. 1873. Lectures on clinical medicine, volume 2. Lindsay & Blakiston.

Tao Tu, Anil Palepu, Mike Schaekermann, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Nenad Tomasev, et al. 2024. Towards conversational diagnostic ai. arXiv preprint arXiv:2401.05654.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Yunfei Xie, Juncheng Wu, Haoqin Tu, Siwei Yang, Bingchen Zhao, Yongshuo Zong, Qiao Jin, Cihang Xie, and Yuyin Zhou. 2024. A preliminary study of o1 in medicine: Are we closer to an ai doctor? arXiv preprint arXiv:2409.15277.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Guangtao Zeng, Wenmian Yang, Zeqian Ju, Yue Yang, Sicheng Wang, Ruisi Zhang, Meng Zhou, Jiaqi Zeng, Xiangyu Dong, Ruoyu Zhang, et al. 2020. Meddialog: Large-scale medical dialogue datasets. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 9241–9250.

Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li, Ganqu Cui, Biqing Qi, Xuekai Zhu, et al. 2024. Ultramedical: Building specialized generalists in biomedicine. arXiv preprint arXiv:2406.03949.

### A Candidate Set of Dialogue Strategy Tags

|Doctor Dialogue Strategy Labels|Description|
|---|---|
|[Greeting] [Explanation] [Answering] [Clarification] [Medical Advice] [Confirmation] [Concern] [Comfort] [Diagnosis] [Education] [Chief Complaint Inquiry] [Recommendation] [Inquiring about Symptoms] [Inquiry about Accompanying Symptoms] [Gathering Family or Medical History] [Evaluation] [Arrangement] [Prescription] [Farewell]|The doctor initiates the conversation by greeting the patient.<br><br>The doctor explains the patient’s condition, treatment plan, or medication use.<br><br>The doctor responds to the patient’s questions or concerns.<br><br>The doctor or patient clarifies certain issues.<br><br>The doctor offers health advice or lifestyle guidance.<br><br>The doctor or patient confirms certain information or understanding.<br><br>The doctor expresses concern and attention for the patient.<br><br>The doctor shows care and comfort to the patient.<br><br>The doctor identifies the patient’s condition based on symptoms and examination.<br><br>The doctor identifies the illness or other problem of the patient.<br><br>The doctor asks the patient to describe their primary health concern.<br><br>The doctor gives health advice or suggests lifestyle changes.<br><br>The doctor asks about the patient’s symptoms, medical history, and related information.<br><br>The doctor asks about other symptoms alongside the main issue.<br><br>The doctor asks about the patient’s past medical history or family medical history.<br><br>The doctor assesses the patient’s symptoms.<br><br>The doctor arranges for follow-up tests or appointments.<br><br>The doctor prescribes medication or treatment plans for the patient.<br><br>The doctor concludes the conversation.|

|Patient Dialogue Strategy Labels<br><br>|Description|
|---|---|
|[Greeting] [Describe Condition] [Detail Symptoms] [Ask Questions] [Confirm] [Express Concerns] [Seek Help] [Provide Information] [Discuss Treatment Options] [Disagree] [Explanation Request] [Seek Advice] [Complaint or Feedback] [Request Prescription] [Inquire about Treatment Options] [Share Feelings] [Request Recommendation] [Thanks] [Disagree] [Emotional Expression] [Express Concerns] [Ask about Side Effects] [Seek Understanding] [Ask about Follow-up Arrangements] [Stop]<br><br>|The patient initiates the conversation by greeting the doctor. The patient describes their symptoms or medical history. The patient elaborates on specific physical discomforts or symptoms. The patient asks further questions regarding the doctor’s advice or the condition. The patient confirms or shows understanding of what the doctor has said. The patient expresses worries about the condition or the treatment outcome. The patient requests support or assistance from the doctor. The patient proactively offers relevant health information or past medical history. The patient discusses possible treatment options with the doctor. The patient expresses a different opinion on the doctor’s advice or diagnosis. The patient asks the doctor to further explain the test results or treatment plan. The patient requests professional advice or suggestions from the doctor. The patient offers opinions or suggestions regarding the medical service or treatment process. The patient asks the doctor to prescribe medication. The patient asks about feasible treatment options and expected outcomes. The patient shares their feelings about the condition or treatment, such as pain, anxiety, etc. The patient asks the doctor to recommend other specialists or tests. The patient expresses gratitude for the doctor’s help or advice. The patient expresses a different opinion on the doctor’s advice or treatment plan. The patient expresses emotional reactions to the condition, such as depression, anger, gratitude, etc. The patient expresses anxiety or concerns about their health condition or treatment plan. The patient inquires about possible side effects of the medication or treatment. The patient hopes the doctor will provide more explanation and understanding of their condition. The patient inquires about subsequent tests, follow-up visits, or treatment plans. The patient ends the conversation.|

Onset and duration of illness: Each disease has unique characteristics regarding its onset and progression. Thus, detailed inquiry into the onset of illness is essential for differential diagnosis. Some diseases have an acute onset, such as cerebral embolism, while others progress more slowly, like pulmonary tuberculosis. The duration of illness refers to the time from disease onset to the point of clinical consultation or hospitalization. If multiple symptoms appear, it is essential to trace back to the time of the initial symptom and document the entire medical history in chronological order. For instance, the patient may experience palpitations for 3 months and recurrent nocturnal dyspnea for 2 weeks.

### B Dialogue Strategy Flows

This appendix presents examples of high-quality dialogue strategy flows manually selected from the deduplicated set. Each dialogue’s tags are concatenated sequentially to form a structured dialogue strategy flow (see Figure 6).

The characteristics of the main symptoms: The location, nature, duration, and intensity of symptoms, along with factors that alleviate or worsen them, are essential for diagnosing the affected system or organ and determining the pathological changes’ site, extent, and nature. For instance, upper abdominal pain often points to issues with the stomach, duodenum, or pancreas, while acute pain in the right lower abdomen typically suggests appendicitis. The type of pain—whether burning, colicky, distention, or dull—and whether symptoms are continuous or intermittent, as well as their onset and relief patterns, are diagnostically significant.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Doctor: Greeting Patient: Greeting Doctor: Chief Complaint Inquiry Patient: Provide Information

[Figure 24]

[Figure 25]

[Figure 26]

Patient: Express Concerns

Doctor: Gathering Family or Medical History Patient: Provide Information

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Doctor: Evaluation Doctor: Explanation Patient: Explanation Request Doctor: Answering

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Patient: Seek Advice Doctor: Medical Advice Patient: Discuss Treatment Options Doctor: Arrangement

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Patient: Seek Help

Doctor: Medical Advice Patient: Thanks Doctor: Goodbye Patient: Stop

Figure 6: Example for a dialogue strategy flow.

### C System Prompt of Patient Simulator

The detailed system prompt of our patient simulator is shown in Figure 7.

Inquiry about Accompanying Symptoms: On the basis of the primary symptoms, a series of accompanying symptoms often emerge. These accompanying symptoms are crucial for differential diagnosis or indicating possible complications. For instance, diarrhea may be a common symptom of various underlying causes, making it difficult to diagnose a specific disease based solely on this symptom. However, by inquiring about the accompanying symptoms, the diagnostic direction becomes clearer. For example, diarrhea accompanied by vomiting may suggest acute gastroenteritis caused by consumption of contaminated food or toxic substances, whereas diarrhea with a sensation of incomplete evacuation, when considered along with seasonality and dietary habits, is more likely associated with dysentery.

System Prompt

你是一个正在向医生咨询的患者，你的个人信息如下:

You are a patient consulting a doctor, and my personal information is as follows:

{patient_records}

Figure 7: The system prompt of our patient simulator.

### D The Discussion of Inquiry Type

The detailed discussion of four inquiry types is presented below.

Chief Complaint Inquiry: This refers to asking patients about their most significant discomfort, the most prominent symptoms, or signs they experience, which often represent the primary reason for the visit. A precise chief complaint provides an initial indication of the severity and urgency of the condition and offers diagnostic clues for identifying potential systemic diseases.

#### Gathering Family or Medical History

Family history: It is important to inquire about the health and disease conditions of the patient’s parents, siblings, and children. Particular attention should be paid to whether there are diseases similar to that of the patient, or hereditary diseases such

Specification of Known Symptoms

as hemophilia, albinism, familial hypothyroidism, diabetes, and mental illnesses.

lies in constructing a robust test set and conducting multiple iterations (e.g., iterations of examples and instructions). Through sampling inspections, the inconsistency between our workflow and human evaluations remains below 1%.

Diagnosis and treatment history: If the patient has already received medical treatment at other healthcare facilities prior to this visit, it is essential to inquire about the previous diagnoses, treatments, and their outcomes. If treatment has been administered, a thorough understanding of the medications used, including their names, dosages, durations, and effects, is necessary to inform the current diagnosis and treatment plan.

D-P Dialogue

Patient Record

17-year-old male. The patient has been experiencing increasing pain and swelling in his right knee for the past 12 days. He reports pain with urination for the past 3 weeks and ......

Hello, please tell me about your D symptoms.

我最近感觉右膝疼痛和肿胀，已经持续了12 天。Recently, I've been experiencing pain and swelling in my right knee for 12 days.

P

......

Based on this...... I can now make a D diagnosis......Reiter'ssyndrome.....

......

[Figure 40]

Ground Truth

Past medical history (PMH): PMH encompasses the patient’s prior health status and previously diagnosed conditions, including infectious diseases, injuries, surgical procedures, immunization records, and allergy history, with particular emphasis on factors closely related to the current illness.

[Figure 41]

Extract Disease

[Reactive Arthritis]

[Reiter's syndrome]

[Figure 42]

[Figure 43]

Score

[Figure 44]

[Figure 45]

Ground Truth

Rewrite Disease

V.S.

1

[Figure 46]

[Reiter's syndrome] [Reactive Arthritis]

[Reiter's syndrome] [Reactive Arthritis]

Figure 9: Workflow for assessing diagnostic accuracy in conversations using LLMs.

### E Inquiry Type Annotation Prompt

The detailed prompt of inquiry type annotation is shown in Figure 8.

System Prompt Whether the doctor's response includes:

- 1. ask_main_symptoms: Inquire about the main symptoms.
- 2. ask_details_of_know_symptoms: Inquiring about more specific details about the known symptoms (those

already mentioned in the conversation history or the patient's latest response).

- 3. ask_accompanied_symptoms: Inquiring about accompanying symptoms beyond the known symptoms.
- 4. ask_family_or_medical_history: Inquiring about the patient's family history or medical history.
- 5. medical_diagnose_or_advice: Providing medical diagnoses or advice.

Mark the corresponding dimension with a 1 if it is included; otherwise, mark it as 0. Note:

- - For the first three dimensions, only instances where the doctor is actively asking the patient questions count as valid. If the doctor is merely summarizing symptoms or conditions or mentioning relevant information as part of diagnosis/advice, it does not count as valid.
- - A single question cannot simultaneously belong to both 2 (ask_details_of_know_symptoms) and 3 (ask_accompanied_symptoms). Output format: [{"reason": xxx, "ask_main_symptoms": 0/1}, {"reason": xxx, "ask_details_of_know_symptoms": 0/1}, {"reason": xxx, "ask_accompanied_symptoms": 0/1}, {"reason": xxx, "ask_family_or_medical_history": 0/1}, {"reason": xxx, "medical_diagnose_or_advice": 0/1}] Note:

- 1. Do not output any additional information.
- 2. The "reason" field should be output in English.

Figure 8: Inquiry type annotation prompt.

### F Workflow for assessing diagnostic accuracy

To address the variations in the output formats of different diagnostic models and calculate accurate diagnostic accuracy computation using LLMs, we designed a standardized workflow. This workflow is illustrated in Figure 9.

The workflow processes the complete dialogue contents to extract the diagnostic results, which are then rewritten to avoid false negatives that may arise from discrepancies. After that, these results are compared with the ground truth (GT). Tasks mentioned above, like result extraction, are among the most common for LLMs and can yield preliminary results even without complex prompts. In practice, the key to achieving satisfactory outcomes

