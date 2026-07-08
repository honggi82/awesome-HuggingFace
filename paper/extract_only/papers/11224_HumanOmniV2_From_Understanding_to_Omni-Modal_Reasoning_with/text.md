arXiv:2506.21277v1[cs.CV]26Jun2025

# HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context

Qize Yang†∗ Shimin Yao∗ Weixuan Chen∗ Shenghao Fu Detao Bai Jiaxing Zhao Boyuan Sun Bowen Yin Xihan Wei Jingren Zhou Tongyi Lab, Alibaba Group qize.yqz@alibaba-inc.com https://github.com/HumanMLLM/HumanOmniV2

## Abstract

With the rapid evolution of multimodal large language models, the capacity to deeply understand and interpret human intentions has emerged as a critical capability, which demands detailed and thoughtful reasoning. In recent studies, Reinforcement Learning (RL) has demonstrated potential in enhancing the reasoning capabilities of Large Language Models (LLMs). Nonetheless, the challenges associated with adapting RL to multimodal data and formats remain largely unaddressed. In this paper, we identify two issues in existing multimodal reasoning models: insufficient global context understanding and shortcut problems. Insufficient context understanding can happen when a model misinterprets multimodal context, resulting in incorrect answers. The shortcut problem occurs when the model overlooks crucial clues in multimodal inputs, directly addressing the query without considering the multimodal information. To tackle these issues, we emphasize the necessity for the model to reason with a clear understanding of the global context within multimodal inputs. This global context understanding can effectively prevent the model from overlooking key multimodal cues and ensure a thorough reasoning process. To ensure the accurate interpretation of multimodal context information, we implement a context reward judged by a large language model, alongside format and accuracy rewards. Additionally, to improve complex reasoning capability, we employ the LLM to assess the logical reward, determining whether the reasoning process successfully integrates multimodal information with logical methods. Moreover, we develop a reasoning training dataset that incorporates context information across tasks involving images, videos, and audio. We also introduce a reasoning omni-modal benchmark, IntentBench, aimed at evaluating models in understanding complex human intentions and emotions. Our proposed method demonstrates advanced performance across multiple omni-modal benchmarks compared to other open-source omni-modal models.

## 1 Introduction

As the applications of Multimodal Large Language Models (MLLMs) in human interaction rapidly increase, understanding and reasoning human intentions and thoughts in complex scenarios becomes increasingly important. Through recent improvements in pretraining and instruction fine-tuning, the capabilities of omni-modal models [1, 2, 3, 4] have significantly advanced. While these models are able to handle multiple types of inputs like text, video, and audio simultaneously in complex realworld environments, they often lack strong reasoning abilities. Inspired by DeepSeek-R1 [5], many

*Equal contribution †Project lead

Preprint. Under review.

[Figure 1]

00:00 00:05 00:15 00:30

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

When I first saw you, I decided that I would not date you and not because you're good looking or anything like that. I just knew with the head wrap that you weren't, like it wasn't like Christian thing. So just first impulse, probably not. um, but as I got to talking to you, you seemed really nice and I would definitely be friends with you for sure…

###### Answer: D

Q: Does the woman seem open minded?

Q: Describe the man's reaction to what the woman said at 0:05

###### Answer: C

- A. Yes, because she is friendly with the man.
- B. Yes, she seems open to interacting with people of other religions and cultures.
- C. No, because she judges people based on their appearance
- D. No, because she wouldn't date someone on behalf of their religion.

- A.He seems happy and excited.
- B.The man seems hesitant and unsure.
- C.He seems disappointed and offended
- D.He seems indifferent to what the woman said at 0:05

[Figure 9]

[Figure 10]

<think>Considering the woman's response, she seems to be open to interacting with people of different religions and cultures, as she is willing to date someone despite their religious beliefs. This suggests that she is openminded.</think> <answer>B</answer>

<think> Given the options, the most fitting description of his reaction is that he seems happy and excited. This is because the man's reaction is likely to be positive, as he is considering dating the woman despite his initial hesitation.</think> <answer>A</answer>

Ignoringmultimodalinputs

Lackingcontextunderstanding

- Figure 1: Visualizations of the vanilla GRPO method applied in multimodal tasks. When the model is overconfident on questions, it tends to answer questions directly without considering the global context (left) or may overlook key multimodal inputs (right).

reasoning methods [6, 7, 8, 9] for MLLMs adapt Group Relative Policy Optimization(GRPO) [10] to train models. Specifically, given a multimodal input and a question, these methods prompt the MLLM to generate a reasoning chain that leads to an answer. The model is optimized using both accuracy reward and format reward. The accuracy reward assesses the correctness of the answer, while the format reward encourages following the reason-answer output format. By considering the question step-by-step, models achieve enhanced performance on various tasks, especially for multi-modal mathematical problems. However, models directly adapting vanilla GRPO rely heavily on text reasoning, ignoring the abundant multimodal cues and their comprehensive understanding.

In this paper, we highlight two issues prevalent in current multimodal reasoning models: insufficient global context understanding and shortcut problems during reasoning. Without understanding the global context or ignoring some crucial details, models will misinterpret the multimodal inputs, resulting in incorrect answers. For example, in the left part of Figure 1, the model trained with vanilla GRPO only captures part of the evidence in the video and provides a sub-optimal answer, failing to consider the entire video thoroughly. While in the right part, the model only observes the visual reaction of the face without considering the speech from the woman, leading to an incorrect answer.

To address these two problems, we require the model to reason based on a precise understanding of the global context within multimodal inputs. To achieve this, the model first outputs the context information within the <context> tag. The context understanding prevents the model from bypassing crucial multimodal inputs and offers a comprehensive global context during the subsequent reasoning process. For example, when someone says “no”, only within a full context can the model determine whether it is a rejection, a joke, or a reverse request. To ensure the model accurately comprehends the multimodal context information, we introduce a context reward in addition to the format reward and accuracy reward. This context reward guides the model to improve its understanding of context, assessed by a LLM that compares the consistency between the reference context and the model’s output. Additionally, to encourage the model to develop complex reasoning abilities, we introduce logical reward by using the LLM to assess whether the reasoning process integrates multimodal information and incorporates logical analysis techniques such as reflection, deduction, and induction. The context reward is calculated based on the context part of the completions, and the logical reward is related to both the context and reasoning parts in the completions.

Furthermore, training MLLMs to reason is extremely challenging, primarily due to the scarcity of large-scale human-annotated reasoning data [11, 12, 13]. For this purpose, we develop a omni-modal reasoning training dataset, which incorporates context information and consists of understanding tasks

involving images, videos, and audios. Another challenge in developing omni-modal reasoning models is the lack of related benchmarks to evaluate their performance effectively. We present IntentBench, a novel omni-modal benchmark designed to comprehend human activities and intentions in complex scenes. It includes 633 videos and 2,689 questions that are related to auditory and visual cues within the videos. This benchmark requires a strong understanding and reasoning of global context, careful observation, and complex social relationships. Daily-Omni [14] and WorldSense [15] primarily focus on general perception scenarios. In these datasets, some questions are only related to either video or audio clues. In contrast, IntentBench is designed to evaluate the understanding and reasoning abilities of omni-modal models regarding the complex intentions and emotions of humans. Finally, we develop HumanOmniV2, which achieves the best performance among open-source omni-modal models, with scores of 58.47% on Daily-Omni, 47.1% on WorldSense, and 69.33% on IntentBench. Our contributions can be summarized as:

- • We propose that models should summarize the context of multimodal inputs before engaging in the reasoning process. This approach aims to mitigate issues such as skipping crucial multimodal information and context understanding on multimodal inputs. Additionally, we employ context rewards and logical rewards to incentivize models to accurately summarize the context and facilitate complex reasoning.
- • We provide an omni-modal reasoning training dataset, which includes summaries of multimodal input and reasoning paths for both cold start training and the reinforcement learning stage. Furthermore, we have curated a human-centric benchmark, IntentBench, for omnimodal evaluation, which requires simultaneously understanding video and audio, the global context, complex social relationships, and careful observation.
- • Our proposed HumanOmniV2 achieves the best performance across multiple omni-modal benchmarks compared to existing open-source omni-modal methods, including the newly introduced IntentBench, Daily-Omni, and WorldSense.

## 2 Related Works

Omni-Modal Large Language Model and Benchmarks. Omni-modal large language models [1, 16, 17] usually include video and audio modalities, moving towards a more comprehensive multimodal understanding. MiniCPM-o 2.6 [17] and Ocean-Omni-1.5 [2] enhance their visionlanguage foundation by adding audio processing capabilities, enabling operation across more modalities. Ola [3] excels with its progressive modality alignment strategy, which incrementally enhances the language model’s supporting modalities. VITA1.5 [4] focuses primarily on real-time, end-to-end speech interaction, while IXC2.5-OL [18] introduces memory mechanisms to manage long contexts in streaming videos. ViSpeak [19] proposes a novel task named visual instruction feedback in which models should be aware of visual contents and learn to extract instructions from them.

While most existing multimodal benchmarks [20, 21, 22, 23] focus solely on images or videos, only a few validate both audio and video simultaneously. Although OmniBench[24] combines image and audio for evaluation, it primarily focuses on straightforward cognitive tasks and only provides images instead of original videos. Daily-Omni [14] and WorldSense [15] primarily focus on general scenarios where some challenges relate solely to video or audio clues. In contrast, our benchmark requires both audio and video understanding simultaneously to answer each question. Besides, unlike existing reasoning benchmarks that focus on the STEM (science, technology, engineering, and math) domain, our benchmark is designed to help models understand the complex intentions and emotions of humans.

Reinforcement Learning for MultiModal Reasoning. Reinforcement learning has shown greater effectiveness than supervised fine-tuning (SFT) in building general reasoning abilities. It allows models to explore a wider range of language and develop their own thinking processes. Recent studies [25, 26] combine RL with vision and language, enhancing these capacities. Insight-V [27] employs a multi-agent system to refine models by selecting and learning from self-generated reasoning paths. R1-VL [26] introduces two novel rule-based reasoning rewards to improve reasoning accuracy and validity. Vision-R1 [25] proposes a progressive thinking suppression training strategy, employing GRPO with a hard formatting result reward function to gradually enhance the model’s reasoning skills on a 10k multimodal math dataset. Video-R1 [7] introduces the T-GRPO algorithm, aimed at

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Q: The person in the video is answersing the question "What is your personality like?". Whether the person in the video is lying when he/she answers the question.

Q: Why does the woman nod through the video?

- A. The woman's nods suggest that she is agreeing with the man.
- B. The woman is nodding in agreement with the others in the video.
- C. She is agreeing with the things the man says
- D. She is listening to the man talk and indicating her understanding and acceeptance

Answer: D Answer: Yes

(a) (b)

when 5%

[Figure 15]

who/which 1%

[Figure 16]

what 19%

other 25%

Q: What are the dominant emotional states of the man in the video?

why 27%

how 23%

- A. Questioning
- B. Helpless
- C. Curiosity

- D. Sentimentality
- E. Angry
- F. Friendly

Answer: B,E

(c) (d)

- Figure 2: (a)(b)(c) are examples from Social-IQ 2.0, MDPE, and EMER, respectively. (d) is the statistic of the curated testing set from Social-IQ 2.0

enhancing models’ ability to leverage temporal information within videos for improved reasoning. Two concurrent works, Visionary-R1 [28] and Observe-R1 [29], also involve the model observing the image or video first before reasoning. However, they focus exclusively on vision-related tasks and do not evaluate the context comprehensively, largely overlooking the potential of more comprehensive omni-modal integration. R1-Omni [30] primarily focuses on audio-visual referring segmentation, while EchoInk-R1 [31] explores applying vanilla GRPO on OmniBench, which limits their general applicability. In contrast, with accurate context understanding, our omni-modal reasoning model not only understands complex human intentions but also performs exceptionally well on general omni-modal benchmarks.

- 3 IntentBench

The nodding action is happened in 35s to 38s.

Currently, multimodal reasoning evaluation datasets like MathVista [32], MMVU [33], and VideoMMMU [20] primarily focus on the STEM domain, where audio cues are typically unnecessary. We introduce a new benchmark for evaluating omni-modal reasoning, namely IntentBench, which requires analysis on audio and vision clues. Our dataset focuses on understanding human social interactions in videos, including intention, emotion, and deception. In multi-turn-taking conversations and real-world interactions, interpreting a glance, a change in tone, or the varied meanings of the same words in different contexts is a highly challenging task for multimodal large language models. Our benchmark is curating from Social-IQ 2.0 [34, 35], EMER [36], and MDPE [37].

- • Social-IQ 2.0 paves the way for explainable social intelligence. The dataset is meticulously curated, featuring validated videos, questions, and answers, alongside complexity annotations for each question and answer. Social-IQ 2.0 includes more than 1,000 videos, 6,000 questions, and 24,000 answers. While humans can understand social contexts with high accuracy, current advanced computational models still face difficulties with this task.
- • EMER offers detailed explanations for its emotion, unlike traditional emotion recognition. It extracts more reliable open vocabulary emotion labels, as each label is grounded in a specific basis. It includes 332 video samples from MER2023 for annotation. Human emotions are often subtle or mixed, requiring careful observation of facial expressions, body movements, and speech. This complexity requires a deep analysis to interpret emotional states.

- • MDPE includes recordings of 193 individuals, with each person answering 24 questions. Out of these, 9 questions are randomly selected for lying, and the interviewer assesses whether the candidate is lying. After the interview, participants also complete the "Subject Lie Confidence Scale," where they rate their confidence in successfully deceiving, ranging from 1 to 5, with 1 meaning successful deception and 5 meaning not deceived successfully. To effectively judge deception, the model must thoroughly analyze body language, microexpressions, tone, and speech content, which makes it a complex task.

For Social-IQ 2.0, we select 300 videos and 2,356 questions. We use GPT-4o [38] with only text modal for testing to identify challenging questions. We also replace easy options to increase the difficulty of the testing set. Finally, we conduct manual verification to ensure each question is relevant to multimodal information and cannot be answered directly through text alone. GPT-4o achieves 75.71% on the original testing set. After our modifications, the performance is 60.02%. The detailed distribution of the questions of this part of the benchmark is shown in Figure 2 (d).

For EMER, we refine the emotion vocabulary within these videos and organize all descriptive vocabulary into hierarchical categories. For the open vocabulary emotion options of the person in the videos, in addition to the original emotion ground-truth labels, we randomly select emotion description terms from other groups to create a multiple-choice question with multiple answers. We randomly select 133 videos and their corresponding questions as the testing set. We assess the performance of this part of data using the F1-score.

For MDPE, we reformat it to QA format and select samples where the interviewees feel uncertain about successfully deceiving—a total of 60 clips (i.e., the confidence rating is above 3). We also include 20 clips where they feel confident in their deception (i.e., the confidence rating is lower than

- 3, more challenging) and 120 no-deception clips, totaling 200 videos to form the deception data.

Finally, our IntentBench comprises 2,689 questions and 633 videos. Figure 2 shows some examples of IntentBench. Each question requires an analysis of both visual and audio cues from the videos. In the appendix, we also provide a more detailed comparison with the original dataset.

- 4 From Understanding to Reasoning

### 4.1 Preliminary

Group Relative Policy Optimization (GRPO) [10] streamlines the reinforcement learning approach by eliminating the critic model. This is achieved by generating multiple responses for each sample and then calculating a normalized reward within the group to determine the advantage value. We follow GRPO with standard practice, but introduce two key modifications based on recent work [39, 40]. First, we use the token-level loss to overcome the imbalance problem for long sample training. Second, we remove the question-level normalization term, which may result in varying weights in the objective for different questions, leading to a difficulty bias in optimization. Besides, we apply the dynamic KL to avoid restricting exploration in the initial stage and diverging in the late stage, encouraging better exploration and improving training stability. With these changes, the GRPO objective is updated as follows:

J (θ) = Eq∼P(q),{o  i}Gi=1∼πθold(·|q)  1

 

|oi|

G

min ri,t(θ)Aˆi,t,clip(ri,t(θ),1 − ε,1 + ε)Aˆi,t − βˆDKL (πθ∥πref)

G i=1 |oi|

t=1

i=1

(1) where

πθ(oi,t | q,oi,<t) πθ

, Aˆi,t = Ri − mean({Ri}Gi=1), (2)

ri,t(θ) =

(oi,t | q,oi,<t)

old

πref (oi | q) πθ (oi | q) − 1. (3)

πref (oi | q) πθ (oi | q) − log

DKL [πθ||πref] =

where both ε and βˆ are hyper-parameters. ε controls the clipping bound and limits the range of policy updates to avoid large changes that could destabilize training. βˆ is the KL penalty coefficient

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Q: Why was the timeline the man in black looked at earlier funny?

- A.The man thought the timeline was funny because it coincided with his own company's growth.
- B.The man in plaid finds the timeline amusing because it reminds him of a joke.
- C.The man in the suit is laughing because he spilled coffee on his pants.
- D.The timeline was funny to the man in black because it had jokes about his company.

[Figure 21]

Context Reward

<context> In the video, two men are seated on stools on a stage, with a blue background and the word "Snapchat" partially visible. The man on the left is wearing a dark long-sleeve shirt and dark pants, while the man on the right is wearing a gray short-sleeve shirt and dark pants. . . The conversation involves the man on the left discussing the growth of the platform and the timing of the launch of the "Stories" feature... </context >

<think>Okay, I'm trying to figure out why the man on the left found the timeline "funny." Let me take a closer look at the video. I see him gesturing with his hands...

Reasoning Logical Reward

###### Now, let me listen to the audio. The man on the left says, “It was actually funny ...." That's the key part. He's saying it was "funny" because the timeline shows how quickly they adopted the features.

So, if I think about it, "funny" here doesn't mean he's laughing at a joke. It's more about the timing and how it coincides with their own growth ... And it's not about him spilling coffee, because I don't see any coffee or hear any sound effects. So, it really seems to be about the timing and the coincidence with their own needs. That's why I think the best answer is that the

timeline was funny to him because it coincided with his own company's growth.</think> <answer>A</answer>

[Figure 22]

- Figure 3: The reasoning path of our model on an example from Social-IQ 2.0. The model first clearly understands the context information of the video clip in the multi-person talking scenario; then it starts reasoning with the multimodal clues to precisely answer the question.

that regularizes deviation from a reference policy πref. In this work, we dynamically reduce the KL penalty in the first S iterations.

- β1 + Sk · (β2 − β1) if 0 ≤ k ≤ S

- β2 if k > S.

βˆ =

(4)

Using a large constraint β1 in the initial stage keeps the model within a close range of the baseline model, ensuring stable training. Conversely, using a small constraint β2 encourages more in-depth thinking and the generation of long, detailed reasoning, but may also lead to reward hacking.

### 4.2 Shortcuts and Context in Multimodal Reasoning

When training with vanilla GRPO, one key issue is the lack of global context understanding, leading to inaccurate identification of key evidence and context during reasoning, as shown in Figure 1. Another problem is that the model tends to overlook multimodal inputs, relying more on textual patterns from questions to generate answers. However, vision and audio cues are vital in multimodal interactions, and the correct answer often depends on subtle multimodal information.

To address this challenge, we propose two modifications. First, we require the model to explicitly interpret or summarize the context information from multimodal inputs before it starts reasoning, avoiding the shortcut problem. Second, to ensure more reliable reasoning based on multimodal context, we require the model to integrate multimodal information throughout the reasoning process. We encourage reasoning capabilities such as reflection, reviewing multimodal inputs, and logical thinking.

Response format. We train the model to first generate its understanding of the multimodal context before proceeding with reasoning. Specifically, we prompt the model to generate a detailed description, which is wrapped using a <context>...</context> tag, capturing the context information from multimodal inputs. In the <think>...</think> tag, the model starts logically reasoning and is encouraged to reflect during reasoning, reviewing the multimodal content, as shown in Figure 3. The model puts the final answer to the question in <answer>...</answer> tag. The final format we request the model to follow is:

#### <context>...</context> <think>...</think> <answer>...</answer>

[Figure 23]

<context> Two men are sitting on stools …<think> Okay, I'm trying to figure …<answer>D</answer>

r_l r_c r_a r_f r_l r_c r_a r_f

[Figure 24]

< context > The man on the left is wearing …<think> Okay, I'm trying to figure …<answer>D</answer>

Rule-Based Verification

Qwen2.5-Omni Thinker

[Figure 25]

Video

< context > In the video, two men are …<think> I see two men talking. The man …<answer>A</answer>

r_l r_c r_a r_f

Policy model

Audio

Q: Why was the timeline the man in black looked at earlier funny?

Reference Model

LLM as a judge

Text

- Figure 4: Illustration of our method. We use Qwen2.5-Omni-Thinker[1] as our base model. For each training sample, we generate 8 completions and compute format and accuracy rewards with verifiable labels. Additionally, we assess reasoning-logical and context rewards by using a LLM as the judge, applying these rewards only to corresponding seen tokens for different rewards.

- A.The man thought the timeline was funny because it coincided with his own company's growth.
- B.The man in plaid finds the timeline amusing because it reminds him of a joke.
- C.The man in the suit is laughing because he spilled coffee on his pants.
- D.The timeline was funny to the man in black because it had jokes about his company.

The output is evaluated using a binary format reward rf ∈ {0,1}, which verifies whether the generated response follows the context-think-answer format. The system prompt is specified in the appendix. To improve the context understanding and the reasoning process, we design specific rewards for both the context understanding and reasoning phases.

### 4.3 Context Reward and Reasoning Logical Reward with Casual Mask

While the format enforces structure, it does not guarantee that the context is sufficiently detailed and precise to support reasoning. To address this issue, we introduce a specialized context reward rc ∈ {0,1} based on utilizing LLMs to evaluate the summary of context by offering clear scoring guidelines and prompts. Specifically, we feed the generated context into an LLM and ask it to answer the question based on the generated context and reference context for each labeled sample. Furthermore, to assess the logical reward rl ∈ {0,1} of the reasoning process, we prompt the LLM to evaluate whether the reasoning incorporates multimodal information, reflection, confirmation, and logical deduction. Since these two rewards only evaluate certain intermediate tokens within the reasoning path, the context reward and logical reward are masked and applied solely to the corresponding seen tokens, as shown in Figure 4. To ensure balanced training, the context reward and logical reward are individually normalized.

For accuracy rewards, we utilize LLM to obtain similarity scores for open-ended question answers, rather than relying on metrics such as BLEU [41] or ROUGE [42] scores. For multiple-choice questions with a single answer, we use accuracy as the reward. For multiple-choice questions with multiple answers, we utilize the F1-score as the reward. For OCR and ASR tasks, the reward is calculated as 1 − WER (Word Error Rate). In evaluating numerical answers, we check for equality between two numbers.

### 4.4 Training Recipe

Cold start training. We select Qwen2.5-Omni-7B-thinker as our base model and modify the model’s system prompt to ensure it follows our specified output format. During the initial stage, we use long CoT (Chain-of-Thought) data, incorporating video and image reasoning data for cold start training, to stabilize the model in the reasoning phase.

- RL Training Stage 1. After completing the cold start training, we sample 8 times for each sample and retain those with an accuracy within (0,0.75). At this stage, all RL training samples are annotated with context to facilitate rewarding relevant content during RL training. Thus, the reward would be more dense. For reasoning rewards, we use the context generated in each completion instead of the

Table 1: Comparison with other methods on Daily-Omni [14].

LLM Size

AV Event Alignment

Context Understanding

Event Sequence

30s Subset

60s Subset

Methods

Comparative

###### Avg Proprietary MLLMs

Inference Reasoning

Gemini 2.0 Flash - 62.18 73.28 63.73 63.72 76.62 75.43 62.29 56.57 67.84 Gemini 2.0 Flash Lite - 55.04 64.89 58.03 54.25 74.03 72.00 60.57 53.01 61.32

###### Open-Source Video-Audio MLLMs

Unified-IO-2 [45] 8B 25.63 31.30 26.42 25.82 35.06 29.71 26.74 30.00 28.24 VideoLLaMA2 [46] 7B 35.71 35.88 35.75 31.70 40.91 34.29 38.02 31.82 35.17 Qwen2.5-Omni [1] 3B 38.66 48.09 33.68 33.99 54.55 44.00 46.68 48.36 40.52 Qwen2.5-Omni [1] 7B 44.12 51.15 38.86 40.52 57.79 61.71 42.35 38.36 47.45 Ola [3] 7B 40.33 60.30 39.89 44.11 61.03 66.28 50.85 48.72 49.87 MiniCPM-o [3] 7B 40.33 61.06 49.22 48.36 68.83 61.14 54.09 52.00 53.13 Ours 7B 46.63 67.93 51.81 51.63 72.72 74.28 63.06 53.09 58.47

ground-truth context. During this phase, we employ LLM (e.g., Qwen2.5-72B) as the reward model. For detailed prompt information, please refer to the supplementary material.

- RL Training Stage 2. After completing the first stage of RL training, we aim to enhance the model’s general capabilities by utilizing general RL training data, including the data sourced from Video-R1[43] and OmniInstruct [24]. For these datasets, we also sample 8 times for each sample and retain those with an accuracy within the range of (0,0.75). Since the previous phase already guides the model to learn more precise context and complex reasoning processes, in this phase, we only use the format reward and accuracy reward.

## 5 Experiments

### 5.1 Experimental Settings

Training data, details, and baselines. To obtain high-quality CoT data, we sample part of the data from Video-R1, the training set of Social-IQ2.0, and the remaining 200 entries from EMER. Then, we use Gemini-2.5-pro [44] to rewrite the reasoning processes, and for Social-IQ2.0, we carry out manual reviews to ensure the reasoning accuracy. For the sampled data from Video-R1, we also use Gemini-2.5-pro to rewrite the reasoning path. Finally, we collect 24K high-quality video-audio training data for cold-start training and RL training. In the cold start training stage, the learning rate is set to 1e−5 and the batch size is 128. During the RL stages, the clip range for training is 0.2. The step S for KL constraints is half of the total steps, with β1 and β2 set at 0.04 and 0.01, respectively. The learning rate is 1e−6, and the maximum output length is 2048. To validate the effectiveness of our proposed method, we compare our full method against the model trained only with SFT using CoT data, our method without extra rewards, and the vanilla GRPO [10].

Evaluation and benchmarks. In testing, the video frame sampling rate is 2 FPS, with a maximum of 32 frames. We evaluate our method on IntentBench, Daily-Omni [14], and WorldSense [15]. Most questions in these benchmarks require simultaneous understanding of both video and audio, making them well-suited for assessing omni-modal models.

Daily-Omni [14] is an audio-visual questioning and answering benchmark that includes 684 videos capturing everyday life scenarios from various sources. These videos are rich in both audio and visual information, and the benchmark features 1,197 multiple-choice QA pairs spread across 6 major tasks.

WorldSense [15] features a strong coupling of audio and video, requiring models to effectively leverage the synergistic perception capabilities of omni-modal data. It includes 1,662 audio-visual synchronized videos, sorted into 8 main domains. Additionally, there are 3,172 multiple-choice QA pairs across 26 different tasks to enable thorough evaluation.

### 5.2 Main Results

We compare our method with open-source omni-modal methods and proprietary MLLMs on DailyOmni and WorldSense, as shown in Table 1 and Table 2, and IntentBench in Table 3. The experimental results indicate that our model outperforms most open-source omni-modal models. Specifically, in out-of-domain testing, our method achieves the best results among open-source models. Particularly,

Table 2: Comparison with other methods on WorldSense [15].

LLM Size

Tech & Science

Culture & Politics

Daily Life

Film & TV

Performance

Methods

Games Sports Music Avg Proprietary MLLMs

Claude 3.5 Sonnet [47] - 43.7 31.7 30.6 36.5 30.7 31.9 36.6 33.9 34.8 GPT 4o [48] - 48.0 44.0 38.3 43.5 41.9 41.2 42.6 42.7 42.6 Gemini 1.5 Pro [49] - 53.7 47.2 50.3 50.4 52.4 46.8 40.2 42.0 48.0

###### Open-Source Video-Audio MLLMs

Unified-IO-2 XXL [45] 7B 27.1 31.7 23.9 23.7 25.5 23.7 25.7 27.3 25.9 VideoLLaMA2 [46] 7B 29.4 25.4 21.8 24.5 26.2 24.6 25.5 27.1 25.4 VITA-1.5 [4] 7B 38.2 35.9 34.3 39.8 41.2 32.6 34.7 39.9 36.9 Qwen2.5-Omni [1] 7B 47.8 49.8 43.6 43.8 48.3 39.1 43.5 47.3 45.4 Ours 7B 50.2 51.7 47.6 44.8 47.3 44.3 45.2 44.2 47.1

Table 3: Comparison with other methods on our IntentBench. ∗Since these models do not support audio input, we use the transcript of the audio instead. “ER” represents the context reward and logical reward.

Social

Methods LMM

Emotion Deception Avg

Why How What When Who/Which Other

###### Proprietary MLLMs

GPT-4o∗ [38] - 61.46 55.69 60.00 35.71 76.00 63.31 60.99 59.00 59.98 GPT-o1∗(think) [50] - 68.19 65.82 66.04 57.14 76.00 68.83 67.26 59.50 66.69 Gemini-2.5-Pro (think) [44] - 68.57 67.41 65.12 57.14 64.00 70.03 68.23 60.00 67.15

###### Open-Source Video-Audio MLLMs

MiniCPM-o [17] 8B 57.87 53.48 57.14 57.14 68.00 61.14 23.85 49.50 54.51 VITA-1.5 [4] 7B 53.15 49.36 51.66 71.42 64.00 61.14 53.20 59.50 54.17 Ola [3] 7B 60.60 55.37 56.87 64.28 76.00 62.91 46.66 44.50 57.41 Qwen2.5-Omni [1] 7B 62.60 63.44 63.53 57.14 76.00 69.03 59.74 63.50 64.20

Ours Ours (full method) 7B 66.76 67.08 71.25 50.00 84.00 72.39 82.41 64.00 69.33

- - SFT (CoT) 7B 58.30 60.28 62.91 64.28 76.00 65.28 79.00 57.00 62.03

- - Ours w/o ER 7B 67.47 65.98 68.54 50.00 76.00 71.20 84.00 62.00 68.44

- - Ours w/o context and ER 7B 64.47 64.08 66.45 50.00 80.00 71.40 81.37 61.50 66.72

it boosts performance from 61.71 to 74.28 in the "reasoning" task within Daily-Omni, demonstrating comparable performance to Gemini-2.0-Flash-Lite. We also observe that our method performs worse than Qwen2.5-Omni in the “Performance” and “Music” tasks of WorldSense. These two tasks are primarily related to visual perception or auditory perception. In in-domain testing, our method achieves the best results compared to other methods and baselines, as shown in Table 3. Our method effectively combines global context information with fine-grained audio-video clues during reasoning, leading to higher accuracy in complex social situations and understanding human intent.

As shown in Figure 5, we present an example from our IntentBench. Within the <context> tag, our model summarizes the global context information, including who, where, and what is happening in the video. This global context information helps mitigate the shortcut problem in multimodal reasoning and provides a basis for subsequent reasoning. Consequently, the model avoids focusing solely on specific details, ensuring that it does not overlook the broader context during reasoning.

### 5.3 Ablation Study and Analyses

The effect of cold start. As shown in Table 3, the performance of the cold start long CoT training (i.e., “SFT(CoT)” row in Table 3) slightly decreases compared to Qwen2.5-Omni. We think the primary reason is that the CoT training data is largely smaller than Qwen2.5-Omni (1,200 billion tokens in total). Additionally, Qwen2.5-Omni already performs comparably well on original Social-IQ 2.0 and MDPE, possibly because the training sets of these datasets are already included in the training of the original Qwen2.5-Omni.

The effect of context reward and logical reward. The context reward incentivizes the model to enhance context quality, while the logical reward promotes complex reasoning and integrates

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

…He had the opportunity to really i think destroy her in this debate. And he did not take that opportunity. And he didn’t. And he blew it. He didn't take the opportunity and she has calculated. I think that I'm just not going to mention it and …

[Figure 31]

Answer: A

Question: Why does the man stop talking and rest his hands on the table?

- A.He is paying attention to what the woman has to say.
- B.Yes the man and woman are agreeing on the topic that they are talking about.
- C.They are disappointed in him
- D.He is very worked about about how a candidate had missed opportunities

[Figure 32]

<context>In the video, a man and a woman are sitting on opposite sides of a table, with the background being a news studio. The man is wearing a suit and gesturing actively, while the woman is listening intently. The scene then switches to a woman in a pink suit speaking at a podium in a formal event setting. In the audio, the man's initial tone is rapid and high-pitched, with a sense of tension, while the woman's response is in a low and gentle tone, speaking slowly to soothe the man. As the conversation progresses, the man's tone becomes more focused and firm. The content revolves around a political debate, specifically analyzing the Clinton campaign's strategy of avoiding the "WikiLeaks and taxes" issue, the Republican candidate's missed opportunity, and the impact of the issue being "buried by other events." The man emphasizes that the Republican candidate did not mention the issue and "did not take the opportunity," and the woman confirms that "she got away with it," concluding with a summary of how the issue dominated the news cycle.</context> <think>Okay, I'm trying to figure out why the man suddenly stops talking and just rests his hands on the table. I need to look at what's happening visually and listen to what they're saying.

Let me take a close look at the video. When he stops talking, the woman seems to be really focused on him, listening intently. Her body language suggests she's waiting for him to continue. This makes me think she's actively engaged and ready to respond or discuss whatever he was about to say next. So, my first thought is that he stopped because she's now taking the lead or responding to what he was just discussing.

Now, I'll listen to the audio again. Ah, there it is! Right before he stops, he says something like, "And he did not take that." He seems to be talking about the Republican candidate and an "opportunity." Then, after he stops, I hear the woman immediately say, "And he didn't... and he blew it." She's clearly picking up on his point and continuing the conversation. This really shows that she's following his train of thought and is now the active respondent.

So, let me consider the options. One idea is that the man stopped because the woman was about to speak. This aligns perfectly with what I just saw and heard. His speech being interrupted by her active response confirms this. Another thought might be that they're reaching an agreement. But listening to the audio again, it sounds more like he's making a point, and then she pushes back with "and he didn't... and he blew it." That doesn't sound like a consensus; it sounds more like a continuation of a debate. What about "they are disappointed"? I don't see or hear any signs of disappointment from either of them. The man seems focused, and the woman is listening intently, ready to respond. And the idea that he stopped because of the "Republican candidate's missed opportunity"? Well, that's definitely part of what he was saying. But the *reason he stopped* was because the woman was about to respond to that very point he was making. The core reason he paused was so she could jump in and agree or challenge him on that specific issue he was raising. So, while his topic was about the missed opportunity, his immediate reason for stopping was to allow her to respond.</think><answer>A</answer>

Figure 5: Visualization result of our method on IntentBench.

multimodal inputs into the reasoning process. Consequently, as shown in Table 3, the performance without these additional rewards is inferior to our full method.

The effect of context. In Table 3, while our model without context and extra rewards performs better than Qwen2.5-Omni, its performance drops to 66.72 compared to our full method. This result highlights the importance of the global understanding of multimodal inputs. The context can effectively offer a comprehensive understanding for the subsequent reasoning process.

## 6 Conclusion

In this paper, we highlight two issues in multimodal reasoning: insufficient global context understanding and the tendency to overlook multimodal inputs. To address these challenges, we introduce HumanOmniV2, an omni-modal reasoning model that requires summarizing the global context of inputs. By basing reasoning on this comprehensive understanding, it reduces problems like overlooking essential multimodal information and enhances context comprehension of multimodal inputs. We also introduce an omni-modal reasoning benchmark that focuses on understanding human intentions and emotions in complex social interactions. Our proposed method surpasses other open-source omni-modal methods and vanilla GRPO across various benchmarks, demonstrating the effectiveness of summarizing the context of multimodal inputs before reasoning.

## 7 Limitations

Our method reduces the probability of shortcuts and context errors during reasoning, but it does not completely solve the issues. This response format might also restrict the ability of the model to correct context information during the following thinking process. Moreover, our experiments are primarily based on a 7B model, so consistent conclusions may not be guaranteed at larger scales. Additionally, shorter response length also restricts the model’s ability to reflect thoroughly on multimodal information. In future research, along with expanding context and pretraining scale, we aim to explore methods for multiple verification of multimodal information during the reasoning process to enhance accuracy.

## 8 Appendix

### 8.1 More visualizations of our method

From Figure 6 to 9, we present visualizations of the evaluation results for our full method and the compared method on IntentBench. Our model can effectively comprehend global context information and capture fine-grained video-audio clues from multimodal inputs during the reasoning process.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

In the past, after the epidemic, the epidemic situation has improved. Now everything is fine, tourism and other things are open, we have made it lively, and the people are happy and can do whatever they want. This is good.

[Figure 38]

Question: The person in the video is answering the question "What was your most unforgettable experience in the past year? Whether thepersoninthevideoislyingwhenhe/sheanswersthequestion? Answer: Yes

[Figure 39]

<think> The speaker is discussing the positive changes and improvements that have occurred since the end of the pandemic, mentioning that the situation has improved, travel has resumed, and people are feeling happier and more relaxed. The speaker's tone is upbeat and positive, suggesting that they are genuinely pleased with the current state of affairs. There are no obvious signs of deception in the speaker's body language or facial expressions. </think> <answer>No</answer Response from vanilla GRPO

[Figure 40]

<context>The video shows a man sitting in an office-like environment. He is wearing a striped shirt and a jacket, with a cap on his head. His facial expression appears somewhat serious, and his body language is relatively static. He occasionally nods and makes slight hand movements, but overall, his movements are limited. The background includes a desk, computer, and some soundproofing panels, suggesting a professional setting. The audio features a man speaking in a low, gentle tone, with a calm and slow pace. The content of the speech is about positive changes after the pandemic, such as the resumption of travel and business, and people feeling happy. The overall tone is optimistic and positive, without any obvious emotional fluctuations.</context><think>Okay, I'm trying to figure out if the person in this video is lying about their most memorable experience from the past year.

First, I'm looking at the video. The man seems pretty serious, and his movements are quite limited – just some slight nods and hand gestures. He doesn't look particularly animated or excited, which makes me wonder if he's really recalling something vivid or if he's just going through the motions. The setting, an office with soundproofing panels, also feels a bit formal and perhaps a little detached for sharing

something deeply personal. Now, I'm listening to the audio. He's speaking in a low, gentle tone, and his words are about positive things like travel resuming and people being happy after the pandemic. He sounds optimistic, but there's no real emotional intensity, like strong joy or sadness, that I can hear. It's all very calm and positive.

So, putting the visual and audio together: his serious expression and limited, almost static body language don't really match up with the kind of vivid, emotionally charged memory I'd expect someone to be recounting. And the content of his speech, focusing on general positive

###### changes rather than a specific, impactful event from his own life, feels a bit superficial. It's not very specific or personal. This makes me think he might not be sharing a truly personal or significant memory, or perhaps he's not entirely truthful about what he's recalling.

Therefore, based on his demeanor and the somewhat superficial nature of his words, I'm leaning towards the idea that he might not be telling the full truth about his most memorable experience.</think><answer>Yes</answer>

Response from Ours

Figure 6: Visualization result of our method and the compared method on IntentBench.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

…there were a lot of mystery schools some people call them cults…you really eased me into that…i did…yeah okay i was thinking like kids primary school …yeah no talking about giving people poison…

[Figure 46]

Question: Why does the woman sitting closest to the man roll her eyes?

Answer: C

- A. She is offended because she feels accused of wrongdoing.
- B. The woman is checking her phone.
- C. She is admitting that she agrees with what the man is saying.
- D. The woman is frustrated with the man because she misunderstands what he's saying.

[Figure 47]

<think> The audio suggests a conversation about a mysterious school in Egypt, with mentions of rituals, magic, and spiritual power. The woman closest to the man might be rolling her eyes in response to a comment or joke made by the man, possibly related to the topic of the conversation. The context of the conversation and the woman's reaction suggest that she finds the man's comment amusing or inappropriate. </think>

- <answer>A</answer> Response from vanilla GRPO

[Figure 48]

<context>The woman on the left is wearing a black hat, a denim jacket, and a light blue top, with a noticeable necklace. She is smiling and looking at the paper, occasionally nodding and using gestures to assist in her expression. The woman in the middle is wearing a black hat, a denim jacket, and a light blue top, with a noticeable necklace. She is smiling and looking at the paper, occasionally nodding and using

gestures to assist in her expression. The man on the right is wearing a red and black striped hat, glasses, and a black T-shirt, with his arms crossed. He is smiling and looking at the woman in the middle. The three are sitting on a white sofa, with a relaxed and friendly atmosphere, and the background includes a lamp and a pillow. The conversation involves a lot of fantasy elements, such as "mystery school," "spells," and "giving people poison." The tone of the woman in the middle is rapid and high-pitched, while the man on the right responds in a low and gentle tone. The conversation includes many exaggerated expressions (such as "Oh my God" and "You really eased me into that"), with a clear sense of humor and playfulness. The man on the right does not directly refute the content of the conversation but uses laughter and other responses to ease the atmosphere.</context ><think>Okay, I'm trying to understand why the woman on the left rolled her eyes. I need to look at what she's doing, what the others are saying, and how they're saying it.

First, let me focus on her. I see her roll her eyes, and she's also smiling and looking at a piece of paper. She seems to be nodding and using her hands to help explain something. This makes me think she's not really upset or angry. Her body language, the smile, and the way she's interacting with the paper and her gestures all point to her being engaged, not defensive or frustrated.

Now, let me listen to what they're saying. The man on the right is talking about "giving people poison." That sounds pretty serious. But then, I hear the woman in the middle laugh. She also says things like "You really eased me into that," which sounds like she's making light of the situation. The woman on the left is also smiling and seems to be taking it all in. So, the conversation, despite the "poison" comment, doesn't seem to be a serious argument. It feels more like a playful or exaggerated discussion. So, putting it all together:

- * The eye-rolling happens right when the man mentions "giving people poison." But her smile and the overall friendly vibe of the scene make me think it's not a serious accusation. It feels more like a playful or exaggerated reaction.
- * I don't see her looking at a phone; she's clearly focused on the paper and the conversation.
- * She's not denying anything or agreeing too strongly; her eye-rolling and smile seem to be a way to handle the potentially sensitive topic.
- * And it doesn't seem like she's confused; the conversation flows, and everyone seems to be getting along.

Therefore, I think her eye-rolling is more of an exaggerated, playful reaction to a potentially sensitive topic, rather than a sign of actual anger or confusion.</think><answer>C</answer>

Response from Ours

Figure 7: Visualization result of our method and the compared method on IntentBench.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

… we are actually going to create a Facebook feature. and it‘s more meant to make people have a little easier when they are posting stuff … thank you so much. thank you for having me, Julie….

Answer: D

[Figure 54]

Question: What is the relationship between these two people?

- A. They want to call attention to the product.
- B. The relationship between the two individuals are business partners, who are trying to pitch their product.
- C. They are promoting a different product.
- D. It seems like they just met so that he can show his product on her show.

[Figure 55]

<think> The video features two individuals, a man and a woman, who appear to be discussing a product or service. The man is demonstrating a feature on a tablet, and the woman is engaged in the conversation, asking questions and expressing interest. The setting suggests a professional environment, possibly a studio or a set designed for a talk show or interview. The presence of a logo and the mention of a website in the video indicate that the discussion is likely related to promoting a product or service. The man's demonstration and the woman's questions suggest a collaborative effort to showcase the product's features and benefits. </think>

- <answer>B</answer> Response from vanilla GRPO

[Figure 56]

<context>The video shows two people sitting on opposite sides of a table, with a laptop and a flower vase in the middle. The background features a stone wall and greenery, creating a relaxed and formal atmosphere. The man is wearing a light blue shirt, and the woman is wearing a black dress, with jewelry. They are frequently smiling and nodding, displaying a friendly and cooperative relationship. The logo "The Balancing ACT" is clearly displayed in the background. The man in the light blue shirt speaks in a rapid and excited tone, emphasizing the importance of the topic; the woman in the black dress responds with a calm and gentle tone, deliberately slowing her speech to ease the tension. The conversation revolves around the introduction of a new feature (Facebook function) of the "Family iBoard" product, with the woman in the black dress expressing high interest and gratitude multiple times.</context><think>Okay, I'm trying to understand the relationship between these two people. I'm looking at the video and listening to their conversation.

First, I'm noticing the setting. They're in what looks like a studio, with a "The Balancing ACT" logo in the background. This makes me think it's some kind of show or segment. The woman is dressed formally, and the man is in a light blue shirt. They seem comfortable with each other, smiling and nodding, which suggests a professional but friendly dynamic. Now, listening to what they're saying, the man is explaining a new feature of a product called "Family iBoard." He mentions something

about a "Facebook function" and how it can "post to Facebook automatically." The woman, on the other hand, sounds really interested. She says things like, "That is so great," and thanks him, even calling it one of her "favorite segments." This tells me she's not just a customer; she seems to be the host or a key figure here, and she's clearly promoting the product.

So, if I consider the options: One idea is that they're trying to "attract attention to the product." That's definitely happening, but it's a bit too general. I need something more specific. Another thought is that they're "business partners pitching a product." I don't really see any clear evidence of that. There's no talk about profits or business strategies between them; it's all about the product itself. What if they're "promoting a different product"? No, that doesn't fit either. The conversation is entirely focused on the "Family iBoard." I don't hear any mention of other products. The idea that they're "collaborating to showcase the product on a show" seems to fit best. The woman is the host, and the man is presenting the product. Their interaction, like when he says, "Thank you for having me," and she responds, "I can't thank you enough," really points to this. It looks like he's demonstrating the product for her show, and she's the one promoting it.

Let me just re-check the visual cues. Yes, the "The Balancing ACT" logo, the professional attire, and their friendly yet professional interaction all support the idea that this is a collaborative effort on a show. The conversation is clearly about introducing this new feature of the "Family iBoard" to her audience.</think><answer>D</answer>

Response from Ours

Figure 8: Visualization result of our method and the compared method on IntentBench.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

I can't stand the way he sits opposite me and tries to please me pitifully. This is not a …

[Figure 62]

Question: What are the dominant emotional states of the man in the video?

Answer: B,E

- A. Questioning
- B. Helpless
- C. Curiosity

- D. Sentimentality
- E. Angry
- F. Friendly

[Figure 63]

<think> The video shows a man sitting in a car, speaking with a serious expression. The subtitles indicate that he is expressing frustration and annoyance about someone sitting opposite him and trying to please him. The tone of his speech suggests anger and dissatisfaction. The options provided do not directly match the emotion expressed in the video, but the closest match would be: </think> <answer>E</answer>

Response from vanilla GRPO

[Figure 64]

<context>The video shows a male character inside a car. Initially, his facial expression is serious, with slightly furrowed brows and a determined look. As the video progresses, his expression becomes more intense, with his brows furrowing deeper and his eyes squinting slightly. His mouth is slightly open, and his head is slightly tilted forward. Towards the end, his expression softens, and he appears to be

sighing. The audio features the character speaking in a low, gentle tone, with a slow pace. The subtitles display the text: "I really can't stand her sitting opposite me, looking pitiful and trying to please me. This is not a normal mother-son relationship at all."</context><think>Okay, I'm watching this video. At the start, the man in the car looks quite serious. His brows are a bit furrowed, and he has this determined expression. He seems focused on something, maybe a conversation.

As I continue to watch, his face gets more intense. His brows are furrowing deeper, and his eyes are squinting a little. He looks like he's

really trying to get his point across, or perhaps he's feeling some pressure. His mouth is slightly open, and he's leaning forward a bit, which makes me think he's quite engaged and possibly a bit agitated. Now, towards the end of this sequence, his expression changes. He seems to be sighing, and his face has softened. This makes me think he's reached a conclusion or is feeling something different.

Now, I'm listening to the audio. He's speaking, and his voice is low and gentle. He's not shouting or raising his voice at all; it's quite calm. The subtitles show what he's saying: 'I really can't stand her sitting opposite me, looking pitiful and trying to please me. This is not a normal mother-son relationship at all.' So, he's expressing strong dissatisfaction with how someone (presumably a woman, his mother, given the context of the subtitle) is behaving towards him. He finds her pleading look unpleasant and feels this isn't a healthy way for them to interact.

This statement, combined with his earlier intense expression and now softer, sighing demeanor, suggests he's feeling a strong sense of frustration and perhaps a bit of helplessness about this situation. He's not just angry about the behavior itself, but he feels stuck or unable to change it, which makes me think he's feeling helpless.

The initial intensity and the sighing at the end, along with the content of his speech, point to a deep-seated feeling of being unable to accept or deal with the current state of affairs. This leads me to believe he's feeling **Helpless** and also quite **Angry** about the situation.</think><answer>B,E</answer>

Response from Ours

Figure 9: Visualization result of our method and the compared method on IntentBench.

### 8.2 The prompts used in our method

As shown in Figure 10, we carefully design the system prompt to guide our model in thoroughly understanding the global context while encouraging self-reflection and verification, ensuring that the model focuses on and adheres to our specified output format. Figure 11 and Figure 12 are the prompts for assessing logical reward and context reward, respectively. We carefully design the user prompt to guide the LLM in precisely assessing the quality of the context summary and determining whether the reasoning incorporates multimodal information, reflection, confirmation, and logical deduction.

You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from various available modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio, or other available context information, and then proceed with an in-depth analysis related to the question.

In reasoning, it is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.

Figure 10: The system prompt of our model.

Please analyze whether the reasoning text is derived from the evidence and context text based on the following criteria and give a score of 0-5: Grading criteria description (relevance and rationality): Integration of Clues (1 point): During the reasoning process, there is incorporation of clues from the video, image, or audio.

Reflection and Confirmation (1 point): The reasoning involves reflection or second confirmation of choices or answers, including revisiting video, image, or audio evidence.

Logical Reasoning (1 point): The thought process is clear, deriving conclusions through rigorous logical reasoning, analysis, or extension without additional assumptions or contradictions.

Problem Analysis (1 point): The reasoning process includes thorough analysis in conjunction with the problem at hand.

Overall Consistency (1 point): The reasoning text is based on visual or audio evidence and context information, presenting no extra assumptions or contradictions.

Assign one point for each criterion that is met, for a total possible score of five points. Verify that each criterion is addressed and reflect this in your scoring.

context: reference reasoning path: hypothesis

only return the score number:

Figure 11: The prompt for assessing the logical reward.

You are assessing how well the ‘hypothesis’ text covers the key information from the ‘reference’ text. Differences in wording or extra details in the ‘hypothesis’ are fine if the reference’s main points are included:

Score based on this coverage:

5 points : Hypothesis clearly and accurately reflects significant core themes or key aspects of the reference. It demonstrates a good understanding of a substantial part of the reference material. 4 points : Hypothesis reflects some important themes or aspects of the reference. The connection is evident, though perhaps not as comprehensive or central as a 5. 2 points : Hypothesis shows a recognizable connection to themes or aspects of the reference, but it might be more superficial, focus on less central points, or only partially grasp a key aspect. 1 points : Hypothesis has a tenuous or very limited connection to the reference. It might touch on a peripheral detail or a heavily reinterpreted aspect, but largely misses the main substance. 0 points : Hypothesis does not reflect any significant themes or key aspects of the reference, or is on a completely different topic.

Example analysis process: Identify main themes and key aspects in ‘reference’. Determine if ‘hypothesis’ connects to or discusses any of these themes/aspects from ‘reference’. Judge the strength and relevance of this connection. Is a core part of the ‘reference’ reflected? Differences are expected; evaluate if the ‘hypothesis’ still meaningfully reflects some key part of the ‘reference’.

Assign score based on how well a significant aspect is reflected. reference: reference hypothesis: hypothesis

only return the score number:

Figure 12: The prompt for assessing the context reward.

### 8.3 The Differences between IntentBench and Its Original Datasets.

In Social-IQ 2.0 [35], which is developed based on Social IQ 1.0[34], some questions in the dataset are straightforward and can be answered directly using only the text. Additionally, we also notice that some models may have been trained on this dataset. Thus, we use GPT-4o [38] with only text modal for testing to identify challenging questions. We also replace easy options to increase the difficulty of the testing set. Finally, we conduct manual verification to ensure each question is relevant to multimodal information and cannot be answered directly through text alone. GPT-4o achieves 75.71% on the original testing set. After our modifications, the performance is 60.02%. Through these enhancements, the dataset becomes more challenging compared to its original version. Figure 2 shows some comparisons between the original options and our refined version.

The original EMER [36] is not designed for open vocabulary evaluation. In the AffectGPT [51], the authors propose grouping the predicted results and ground truth emotions for each sample using a large language model (LLM) and then calculating the F1 score. However, the need to frequently utilize LLMs for evaluation can lead to instability, causing significant challenges in both training and testing. Thus, we refine the emotion vocabulary within these videos and organize all descriptive vocabulary into hierarchical categories. For the open vocabulary emotion options of the person in the videos, in addition to the original emotion ground-truth labels, we randomly select emotion description terms from other groups to create multiple-choice questions with multiple answers. Figure 14 shows some comparisons between the original answer and our refined version.

The MDPE dataset [37] is initially crafted for classification tasks. To streamline the evaluation of multimodal models, we reformat them into a question-answer format and incorporate the original

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

don‘t forget to leave a comment down\nbelow okay in the past couple of videos. you not haven't been leaving that many. comments so leave some comments down. there I like to read them yeah but\nanyway thank you very much to all my\n\nsupporters I don't tell you enough how\nmuch I appreciate it and I'll do my best\n\nto keep these videos coming out there so\nyou can learn about model railroading\n\nand everything else that you need to\nlearn about model railroading\n\nso until Wednesday we'll see at 2:53\np.m. today is when the maximum coverage\n\nof the Sun happened in Southwest Florida\nand here is a short video of it nothing\n\n

Don t forget to leave a comment down below okay in the past couple of videos. You not haven't been leaving that many comments. So leave some comments down there. I like to read them. Yeah, anyway, thank you very much…

Q: Why does the man in the video remind people to leave more comments?

Answer: A Original options: Refined options:

- A. Because he wants to see what his subscribers think.
- B. The person in the video asks people to leave comments because he wants to improve his public speaking skills.
- C. He thinks that they leave too many comments.
- D. The man reminds people to leave comments because he wants to increase his electricity bill.

- A. Because he wants to see what his subscribers think.
- B. He is an enthusiast who wishes to share his passion
- C. The man enjoys the support he gets
- D. He wants to educate his viewers.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Properly in states where there is alcohol ban …wait wait wait…Did I allow you to speak…I was giving that question to you…I have been called here to speak…It is my right to speak now respond to her…

Q: Are the people in the video angry?

Answer: A Original options: Refined options:

- A. Yes, given how many people kept on interrupting each other.
- B. They are practicing their dance moves.
- C. The group was discussing the weather.
- D. Did it seem like there was tension in the air at the three tables, perhaps due to the lack of communication?

- A. Yes, given how many people kept on interrupting each other.,
- B. Yes they're yelling amongst each other,
- C. Yes, they care a lot about the issue,
- D. She was passionate about her position regarding alcohol prohibition in India.

##### Figure 13: Comparison of the original options in Social-IQ 2.0 and our refined options.

|[Figure 75]<br><br>[Figure 76]| | | |[Figure 77]|[Figure 78]|[Figure 79]|
|---|---|---|---|---|---|---|

I can't stand the way he sits opposite me and tries to please me pitifully. This is not a …

Q: What are the dominant emotional states of the man in the video? Options:

Answer: B,E Answer: ['Angry', 'helpless', 'Discontent']

- A. Questioning
- B. Helpless
- C. Curiosity

- D. Sentimentality
- E. Angry
- F. Friendly

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

It's bad, it's bad. If I really give birth to an imperfect child, his life will be over…

Q: What is the woman feeling in the video?

Options:

- Answer: D,E,F
- D. Nervous
- E. Worry
- F. Surprised

Answer: ['Worry', 'Nervous', 'Surprised']

- A. Approval
- B. Optimism
- C. Admiration

Figure 14: Comparison of the original open-vocabulary answer and our designed multi-choice question with multiple answers in EMER.

questions in each video as part of the query. Additionally, we refine the sample selection process by randomly selecting samples according to specific ratios based on difficulty levels. We select samples where the interviewees feel uncertain about successfully deceiving—a total of 60 clips (i.e., the confidence rating is above 3). We also include 20 clips where they feel confident in their deception (i.e., the confidence rating is lower than 3, more challenging) and 120 no-deception clips, totaling 200 videos to form the deception data.

## References

- [1] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang et al., “Qwen2. 5-omni technical report,” arXiv preprint arXiv:2503.20215, 2025.
- [2] Y. Li, H. Sun, M. Lin, T. Li, G. Dong, T. Zhang, B. Ding, W. Song, Z. Cheng, Y. Huo et al., “Ocean-omni: To understand the world with omni-modality,” arXiv preprint arXiv:2410.08565, 2024.
- [3] Z. Liu, Y. Dong, J. Wang, Z. Liu, W. Hu, J. Lu, and Y. Rao, “Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment,” arXiv preprint arXiv:2502.04328, 2025.
- [4] C. Fu, H. Lin, X. Wang, Y.-F. Zhang, Y. Shen, X. Liu, H. Cao, Z. Long, H. Gao, K. Li et al., “Vita-1.5: Towards gpt-4o level real-time vision and speech interaction,” arXiv preprint arXiv:2501.01957, 2025.
- [5] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [6] F. Meng, L. Du, Z. Liu, Z. Zhou, Q. Lu, D. Fu, T. Han, B. Shi, W. Wang, J. He et al., “Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning,” arXiv preprint arXiv:2503.07365, 2025.
- [7] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue, “Video-r1: Reinforcing video reasoning in mllms,” arXiv preprint arXiv:2503.21776, 2025.
- [8] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang, “Visual-rft: Visual reinforcement fine-tuning,” arXiv preprint arXiv:2503.01785, 2025.
- [9] H. Shen, P. Liu, J. Li, C. Fang, Y. Ma, J. Liao, Q. Shen, Z. Zhang, K. Zhao, Q. Zhang et al., “Vlm-r1: A stable and generalizable r1-style large vision-language model,” arXiv preprint arXiv:2504.07615, 2025.
- [10] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.
- [11] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe, “Let’s verify step by step,” in The Twelfth International Conference on Learning Representations, 2023.
- [12] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017.
- [13] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.
- [14] Z. Zhou, R. Wang, and Z. Wu, “Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities,” arXiv preprint arXiv:2505.17862, 2025.
- [15] J. Hong, S. Yan, J. Cai, X. Jiang, Y. Hu, and W. Xie, “Worldsense: Evaluating real-world omnimodal understanding for multimodal llms,” arXiv preprint arXiv:2502.04326, 2025.
- [16] J. Zhao, Q. Yang, Y. Peng, D. Bai, S. Yao, B. Sun, X. Chen, S. Fu, X. Wei, L. Bo et al., “Humanomni: A large vision-speech language model for human-centric video understanding,” arXiv preprint arXiv:2501.15111, 2025.
- [17] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.
- [18] P. Zhang, X. Dong, Y. Cao, Y. Zang, R. Qian, X. Wei, L. Chen, Y. Li, J. Niu, S. Ding et al., “Internlmxcomposer2. 5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions,” arXiv preprint arXiv:2412.09596, 2024.
- [19] S. Fu, Q. Yang, Y.-M. Li, Y.-X. Peng, K.-Y. Lin, X. Wei, J.-F. Hu, X. Xie, and W.-S. Zheng, “Vispeak: Visual instruction feedback in streaming videos,” arXiv preprint arXiv:2503.12769, 2025.
- [20] K. Hu, P. Wu, F. Pu, W. Xiao, Y. Zhang, X. Yue, B. Li, and Z. Liu, “Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos,” arXiv preprint arXiv:2501.13826, 2025.
- [21] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 24108–24118.
- [22] Y.-X. Peng, Q. Yang, Y.-M. Tang, S. Fu, K.-Y. Lin, X. Wei, and W.-S. Zheng, “Actionart: Advancing multimodal large models for fine-grained human-centric video understanding,” arXiv preprint arXiv:2504.18152, 2025.

- [23] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun et al., “Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9556–9567.
- [24] Y. Li, G. Zhang, Y. Ma, R. Yuan, K. Zhu, H. Guo, Y. Liang, J. Liu, Z. Wang, J. Yang et al., “Omnibench: Towards the future of universal omni-language models,” arXiv preprint arXiv:2409.15272, 2024.
- [25] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, Y. Hu, and S. Lin, “Vision-r1: Incentivizing reasoning capability in multimodal large language models,” arXiv preprint arXiv:2503.06749, 2025.
- [26] J. Zhang, J. Huang, H. Yao, S. Liu, X. Zhang, S. Lu, and D. Tao, “R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization,” arXiv preprint arXiv:2503.12937, 2025.
- [27] Y. Dong, Z. Liu, H.-L. Sun, J. Yang, W. Hu, Y. Rao, and Z. Liu, “Insight-v: Exploring long-chain visual reasoning with multimodal large language models,” arXiv preprint arXiv:2411.14432, 2024.
- [28] J. Xia, Y. Zang, P. Gao, Y. Li, and K. Zhou, “Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning,” arXiv preprint arXiv:2505.14677, 2025.
- [29] Z. Guo, M. Hong, and T. Jin, “Observe-r1: Unlocking reasoning abilities of mllms with dynamic progressive reinforcement learning,” arXiv preprint arXiv:2505.12432, 2025.
- [30] J. Zhao, X. Wei, and L. Bo, “R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning,” arXiv preprint arXiv:2503.05379, 2025.
- [31] Z. Xing, X. Hu, C.-W. Fu, W. Wang, J. Dai, and P.-A. Heng, “Echoink-r1: Exploring audio-visual reasoning in multimodal llms via reinforcement learning,” arXiv preprint arXiv:2505.04623, 2025.
- [32] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” arXiv preprint arXiv:2310.02255, 2023.
- [33] Y. Zhao, L. Xie, H. Zhang, G. Gan, Y. Long, Z. Hu, T. Hu, W. Chen, C. Li, J. Song et al., “Mmvu: Measuring expert-level multi-discipline video understanding,” arXiv preprint arXiv:2501.12380, 2025.
- [34] A. Zadeh, M. Chan, P. P. Liang, E. Tong, and L.-P. Morency, “Social-iq: A question answering benchmark for artificial social intelligence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 8807–8817.
- [35] A. Wilf, L. Mathur, S. Mathew, C. Ko, Y. Kebe, P. P. Liang, and L.-P. Morency, “Social-iq 2.0 challenge: Benchmarking multimodal social understanding,” https://github.com/abwilf/Social-IQ-2.0-Challenge, 2023.
- [36] Z. Lian, H. Sun, L. Sun, H. Gu, Z. Wen, S. Zhang, S. Chen, M. Xu, K. Xu, K. Chen et al., “Explainable multimodal emotion recognition,” arXiv preprint arXiv:2306.15401, 2023.
- [37] C. Cai, S. Liang, X. Liu, K. Zhu, Z. Wen, J. Tao, H. Xie, J. Cui, Y. Ma, Z. Cheng et al., “Mdpe: A multimodal deception dataset with personality and emotional characteristics,” arXiv preprint arXiv:2407.12274, 2024.
- [38] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [39] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, T. Fan, G. Liu, L. Liu, X. Liu et al., “Dapo: An open-source llm reinforcement learning system at scale,” arXiv preprint arXiv:2503.14476, 2025.
- [40] Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, “Understanding r1-zero-like training: A critical perspective,” arXiv preprint arXiv:2503.20783, 2025.
- [41] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.
- [42] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.
- [43] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue, “Video-r1: Reinforcing video reasoning in mllms,” arXiv preprint arXiv:2503.21776, 2025.
- [44] Google, “Gemini 2.5 pro,” https://deepmind.google/technologies/gemini/pro/, 2025.
- [45] J. Lu, C. Clark, S. Lee, Z. Zhang, S. Khosla, R. Marten, D. Hoiem, and A. Kembhavi, “Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26439–26455.
- [46] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms,” arXiv preprint arXiv:2406.07476, 2024.

- [47] Anthropic, “Introducing the next generation of Claude,” https://www.anthropic.com/news/claude-3-family, 2024, accessed: 2024-10-22.
- [48] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [49] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.
- [50] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney et al., “Openai o1 system card,” arXiv preprint arXiv:2412.16720, 2024.
- [51] Z. Lian, H. Sun, L. Sun, J. Yi, B. Liu, and J. Tao, “Affectgpt: Dataset and framework for explainable multimodal emotion recognition,” arXiv preprint arXiv:2407.07653, 2024.

