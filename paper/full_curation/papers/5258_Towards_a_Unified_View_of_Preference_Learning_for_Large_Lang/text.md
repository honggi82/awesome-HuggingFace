# arXiv:2409.02795v5[cs.CL]31Oct2024

## Towards a Unified View of Preference Learning for Large Language Models: A Survey

Bofei Gao1, Feifan Song1, Yibo Miao3, Zefan Cai7, Zhe Yang1, Liang Chen1, Helan Hu1, Runxin Xu1, Qingxiu Dong1, Ce Zheng1, Shanghaoran Quan2, Wen Xiao5, Ge Zhang6, Daoguang Zan8, Keming Lu2, Bowen Yu2, Dayiheng Liu2, Zeyu Cui2, Jian Yang2, Lei Sha4, Houfeng Wang1, Zhifang Sui1, Peiyi Wang1, Tianyu Liu2, Baobao Chang1

Peking University Alibaba Group

### Abstract

Large Language Models (LLMs) exhibit remarkably powerful capabilities. One of the crucial factors to achieve success is aligning the LLM’s output with human preferences. This alignment process often requires only a small amount of data to efficiently enhance the LLM’s performance. While effective, research in this area spans multiple domains, and the methods involved are relatively complex to understand. The relationships between different methods have been under-explored, limiting the development of the preference alignment. In light of this, we break down the existing popular alignment strategies into different components and provide a unified framework to study the current alignment strategies, thereby establishing connections among them. In this survey, we decompose all the strategies in preference learning into four components: model, data, feedback, and algorithm. This unified view offers an in-depth understanding of existing alignment algorithms and also opens up possibilities to synergize the strengths of different strategies. Furthermore, we present detailed working examples of prevalent existing algorithms to facilitate a comprehensive understanding for the readers. Finally, based on our unified perspective, we explore the challenges and future research directions for aligning large language models with human preferences.

[Figure 1]

[Figure 2]

Github Repo [GitHub Page] Project Page [Notion Page]

Affiliations:1Peking University 2Alibaba Group 3Shanghai Jiao Tong University 4Zhongguancun Laboratory 5Microsoft 6University of Waterloo 7University of WisconsinMadison 8Institute of Software, Chinese Academy of Sciences Project Leads: Bofei Gao (gaobofei@stu.pku.edu.cn); Peiyi Wang (wangpeiyi9979@gmail.com); Tianyu Liu (tianyu0421@alibaba-inc.com) Corresponding Author: Baobao Chang (chbb@pku.edu.cn)

#### Contents

- 1 Introduction 3
- 2 Definition and Formulation 5
- 3 The Unified View of Preference Learning for LLM 5
- 4 Preference Data 9

- 4.1 On-policy Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Off-policy Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 5 Feedbacks 11

- 5.1 Direct Feedback . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.2 Model-based Feedback . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 5.2.1 Reward Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.2.2 Pair-wise Scoring Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 5.2.3 LLM-as-a-Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 6 Algorithms 15

- 6.1 Point-wise Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 6.2 Pair-wise Contrast . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 6.3 List-wise Contrast . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 6.4 Training-Free Alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 6.4.1 Input Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.4.2 Output Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 7 Evaluation 20

- 7.1 Rule-based Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 7.2 LLM-based Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- 7.2.1 LLM-based Evaluation Methods . . . . . . . . . . . . . . . . . . . . . . . . 21
- 7.2.2 LLM-based Evaluation Models . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 7.2.3 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 8 Future Directions 22
- 9 Conclusion 23 References 23

- A The discussion about Online vs. Offline 32
- B The discussion about ReMAX and GRPO 32

###### 1. Introduction

[Figure 3]

Feedback Generation

Feedback

Data

Reward model / Human / Rules ...

input / response

LLM πθ

PreferenceOptimization

Training-Free / Pointwise / Pairwise / Listwise

[Figure 4]

Evaluation

Rule-based / Model-based

LLM πθ'

[Figure 5]

Human

[Figure 6]

LLM πθ

[Figure 7]

LLM πθ'

If you were going to steal from a convenience store, do you think it would be better in the morning or at night?

It is good to consider the difference in human traffic at night, and also the security cameras. Night time would probably be better for avoiding security cameras, but you would be more visible to the store employees at night. On the other hand, since it is usually less busy during the day, it is easier to avoid being noticed by other people.

I really couldn’t say, I’m not familiar with stealing convenience store items.

- Figure 1 | A unified view and an illustrative example of preference learning for LLMs.

Represented by ChatGPT1, the rise of large language models (LLMs) has showcased impressive language capabilities and professional competence, as well as providing correct, polite, and knowledgeable responses, which is surprising and admirable. While pretraining and supervised finetuning play a significant role in developing foundational language skills, preference alignment is a currently necessary step that LLMs undergo before public deployment, to prevent LLMs from potentially generating offensive, toxic, or misleading content.

Although large language models (LLMs) have demonstrated impressive capabilities across various fields [20, 97, 120, 148], they still face challenges in ethics [55], safety [64, 111, 134], and reasoning [74, 129, 155]. In response, numerous alignment-related initiatives have emerged to better address these issues [29, 91, 98, 103]. The snowballing interest also inspires this survey. While many works [114, 130] have extensively discussed the concept of alignment, the relationships among the various algorithms of preference learning remain fragmented, lacking a cohesive framework to unify them. To bridge this gap, we aim to provide a systematic framework for preference alignment, as shown in figure 1. By integrating related works within this framework, we hope to offer researchers a comprehensive understanding and a foundation for further exploration in specific areas.

Traditional categorization perspectives [54, 114, 130] tend to split existing methods into reinforcement learning (RL) based methods, like RLHF [98] which requires a reward model for online RL, and supervised finetuning (SFT) based methods like Direct Preference Optimization (DPO) [103] that directly employs preference optimization within an offline setting. However, this split can unconsciously result in a barrier between the two groups of works, which is not conducive to further understanding of researchers for the common core of preference alignment. Therefore, we strive to establish a unified perspective for both sides and introduce an innovative classification framework.

This new framework pivots on two key insights: First, the distinction between on-policy and off-policy settings essentially depends on different sources of data, which can be decoupled from algorithms like PPO or DPO. On-policy setting requires the policy model to generate its data in real-time; specifically, the LLM being optimized must also produce data for the next iteration of training in real-time. In contrast, an off-policy setting allows for a variety of data sources, provided they are collected in advance, without the need for simultaneous generation by the policy model. Many current works employ the transition of specific algorithms between

1https://chatgpt.com

On-policy Top-K/Nucleus Sampling [44], Beam Search [36], MCTS [63]

OpenAI’s Human Preference [98], HH-RLHF [5], SHP [28], Webgpt [96]

Preference Data (§4)

Data from Human

Off-policy

RLAIF [66], Open-Hermes-Preferences [48], ULTRAFEEDBACK [19], UltraChat [22]

Data from LLM

Math: RFT [159], DeepSeekProver [141] [142] Translation: CPO [145] Summarization: PRELUDE [33] Code: Pangu-coder2 [113], StepCoder [25], Rltf [79]

Direct Reward

RLAIF [66], RBoN [57], West-of-N [99], RM-ensemble [18], LoRA-ensemble [162], DMoERM [102], Skywork-Reward [77], WARM [105], Efficient-ensemble [167], Fine-Grained RLHF [139], PRM [123], [74], OVM [154], MATH-Shepherd [129], Prior contraints RM [172], Math-Minos [32]

Feedback (§5)

Reward Modeling

Model-based Reward

PreferenceLearning

Pair-wise Scoring Llm-blender [53], PandaLM [132]

Self-Reward [158], Meta-Reward [137], CriticGPT [90], Generative Verifier [166]

LLM-as-a-Judge

Pointwise method RFT [159], RAFT [23], Star [160], PPO [109], ReMax [72], KTO [29]

CoH [78], SLiC [169], DPO [103], IPO [3], Sr-DPO [156], ORPO [45], Mallow-DPO [9], GRPO* [106], DPO-positive [100], CPL [42], EXO [51], SimPO [91], sDPO [60], TR-DPO [35], RSO [82], 𝑓-DPO [125], CPO [41], MAPO [112], KnowTuning [86], TS-align [163], MODPO [173], HPO [4]

Pairwise contrast

Optimization (§6)

RRHF [157], PRO [116], CycleAlign [46], AFT [128], VCB [89], LiPO [81], LIRE [175], GRPO [110]

Listwise contrast

Input Optimization BPO [12], URIAL [75], OPO [144]

Training-Free

ICDPO [115], Aligner [52], RAIN [71], DecodingControl [21, 80, 95, 150], DeAL [47]

Output Optimization

Factuality [43], Math [17, 155], Reasoning [119], Closed-Book QA [58, 65], Coding [2, 10]

Rule-based

Evaluation (§7)

G-Eval [83], AuPEL [131], ICE [50], GEMBA [62], FairEval [127], Auto-J [68], MT-Bench [171], Prometheus [61], Pandalm [132], PRD [69], LLMBar [161], LLMEval2 [168]

LLM-based

- Figure 2 | Taxonomy of Preference Learning.

on-policy and off-policy settings [40, 110]. Therefore, we do not use on-policy or off-policy as a criterion for classifying algorithms. Second, inspired by existing work [110], the essence of the objective of optimization in reinforcement learning and supervised finetuning-based methods are actually quite similar. The difference lies in that reinforcement learning-based methods often require a reward model to compute the reward for further training, while supervised fine-tuning algorithms can optimize the model using various forms of preferences directly, such as a better-aligned output and pair-wise or list-wise contrasts from preference relations. With a unified perspective, we can define feedback as a broad range of tools capable of producing preferences aligned with human judgment, such as reward models, human annotators, more powerful models like GPT-4, and various rules. Based on these considerations, we divide the process of preference learning into data, feedback, preference optimization, and evaluation. The taxonomy of our paper is shown in Figure 2. Moreover, we provide clear running examples of some common algorithms within this framework to facilitate the readers’ understanding of the algorithms, which are shown in Figure 3 and Figure 4.

In summary, our paper investigates and organizes existing preference learning methods for LLM, offering a unified and novel perspective. Further, based on the content of this survey, we

summarize several future directions in this area, intending to bring insights for further research.

###### 2. Definition and Formulation

In this section, we begin by providing our definition of preference learning for LLM: Given a distribution of the general human preference P(𝑥, 𝑦), where 𝑥 is a prompt, 𝑦 is the corresponding output of the LLM, preference learning for LLM 𝜋𝜃 is a paradigm that produces a new LLM 𝜋𝜃′ that align to P(𝑥, 𝑦), where P(𝑥, 𝑦𝜃′(𝑥)) > P(𝑥, 𝑦𝜃(𝑥)).

To enable LLMs to learn human preferences, the process often involves providing a data sample with input 𝑥 and corresponding response 𝑦, and the environment with human preference P(𝑥, 𝑦) assigning feedback to it. Samples aligned with human preferences are given a higher reward, which could manifest as positive labels, elevated positions in preferential rankings, or heightened reward scores. After obtaining the data, the policy model 𝜋𝜃′ is optimized by a specific algorithm.

Furthermore, it is necessary to explain the relation between preference learning for LLMs and some related concepts, based on this definition. (1) Alignment: Following Kenton et al. [59], alignment refers to the research focuses on tackling the so-called behavior alignment problem: How do we create an agent that behaves in accordance with what a human wants? Based on this definition, we regard preference learning for LLMs as a category of methods aimed at achieving alignment. The scope of this paper is confined to textual preference alignment which doesn’t contain other well-known alignment topics such as hallucination, multi-modal alignment, and instruction tuning. (2) Reinforcement Learning from Human Feedback (RLHF): Different from RLHF, the scope of this paper not only includes RL-based methods but also encompasses the traditional called SFT-based methods. What’s more, we adopt a unified perspective to investigate both reinforcement-learning and supervised-learning-based methods.

###### 3. The Unified View of Preference Learning for LLM

Inspired by recent works [40, 110], we survey existing works from a unified perspective in the following two folds:

First, the optimization objectives of RL and SFT-based methods can be described within the same framework. Following [110], the gradient with respect to the parameter 𝜃 of a training method can be written as:

∑︁|𝑜|

1 |𝑜|

𝛿A(𝑟,𝑞,𝑜,𝑡)∇𝜃 log𝜋𝜃(𝑜𝑡|𝑞, 𝑜<𝑡) , (1)

∇𝜃 = E[(𝑞,𝑜)∼D]

𝑡=1

where D denotes the data source which contains the input question 𝑞 and the output 𝑜. 𝛿 denotes the gradient coefficient which directly determines the direction and step size of the preference optimization. A denotes the algorithm. The gradient coefficient is determined by the specific algorithm, data, and corresponding feedback. Tracing back to one significant source influencing the gradient coefficient is the feedback. Note that the feedback could take on various forms. For example, the correctness of data in RFT [159] or the preference label in DPO [103] could impact the gradient coefficient, thereby affecting the final gradient. Consequently, we define the feedback in our paper as the preference given by the environment that can affect the gradient coefficient. Notably, both RL-based and SFT-based methods can be encapsulated within this framework.

###### Second, the algorithm can be decoupled from online/offline settings. In the context of

###### PPO (Online) for code generation

Query: Write a python function to find the first repeated character in a given string.

[Figure 8]

[Figure 9]

❄

[Figure 10]

Unlabeled Queries

LLM πθ'

Reference Model

random choice P(x)

###### Response:

[Figure 11]

class Pair(object): def __init__(self, a, b):

[Figure 12]

🔥 ❄

[Figure 13]

[Figure 14]

[Figure 15]

- self.a = a
- self.b = b

reward r

Sampling

Query: Write a function to find the longest chain which can be formed from the given set of pairs.

Generalized Advantage Estimation Point-wise Loss

[Figure 16]

def max_chain_length(arr, n):

Reward Model

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

Input: x ~ P(x)

mcl = [1 for i in range(n)]

LLM πθt

[Figure 17]

.....

[Figure 18]

🔥

[Figure 19]

[Figure 20]

Response: y ~ πθ(y|x)

Value V

Critic Model

[Figure 21]

next iteration

###### πθ t-1 πθold

t-1 t

Update Parameter

Step1: Data Sampling

Step3: PPO Training

Step2: Feedback Estimation

###### PPO (Offline) for code generation

[Figure 22]

[Figure 23]

Query: Write a python function to find the first repeated character in a given string.

LLM πθ'

Update Parameter

[Figure 24]

[Figure 25]

🔥

Query: Write a python function to find the first repeated character in a given string.

Unlabeled Queries

LLM πθ

random choice P(x)

###### Response:

class Pair(object): def __init__(self, a, b):

[Figure 26]

Point-wise Loss

Query: Write a function to find the longest chain which can be formed from the given set of pairs.

- self.a = a
- self.b = b

[Figure 27]

🔥

[Figure 28]

[Figure 29]

[Figure 30]

def max_chain_length(arr, n):

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

Input: x ~ P(x)

[Figure 31]

mcl = [1 for i in range(n)]

Value V

###### Critic Model

LLM πθ

πθ πθ

[Figure 32]

reward r

old

Offline dataset D = {(x, y, r)}

[Figure 33]

❄

Sampling

[Figure 34]

[Figure 35]

Generalized Advantage Estimation

###### Response:

class Pair(object): def __init__(self, a, b):

Reference Model

[Figure 36]

Advantage A

- self.a = a
- self.b = b

def max_chain_length(arr, n):

reward r

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

[Figure 37]

❄

[Figure 38]

mcl = [1 for i in range(n)]

.....

Response: y ~ πθ(y|x)

Reward Model

Step3: PPO Training

Step2: Advantage Estimation

Step1: Offline Data Collection

###### DPO (Online) for Summarization

Environment with Human Preference p(x,y)

Content: Recently, my fiance (20 m) and I (19f) moved into a new apartment with a mutual friend (20m) and somehow contracted scabies (don't know how). ... I'm almost there to asking him to move out if he refuses treatment . He is not on the lease.

[Figure 39]

[Figure 40]

[Figure 41]

or or

[Figure 42]

GPT-4 RM LLM Annotator

[Figure 43]

Evaluation

Unlabeled Queries

πref = πθ0

LLM πθ'

random choice P(x)

[Figure 44]

🔥

[Figure 45]

✅ chosen

[Figure 46]

Content: So you're saying "try it, I might not mind losing access to directions that follow my only available mode of transportation (public)"? .......trains that make up the greater NY area with a combination of 3D flyovers, luck, and magic.

[Figure 47]

Sampling

[Figure 48]

> Pair-wise Contrasts

[Figure 49]

###### Summarization Candidate 2:

❌ rejected

[Figure 50]

you don't know how to go to the subway.

LLM πθt LLM πref

[Figure 51]

Response: yc, yr ~ πθ(y|x)

Input: x ~ P(x)

[Figure 52]

Step1: Data Sampling Step2: Feedback Generation Step3: DPO Training

DPO (Offline) for Summarization

[Figure 53]

Environment with Human Preference p(x,y)

[Figure 54]

[Figure 55]

[Figure 56]

###### πref = πθ

or or

LLM πθ

Content: So you're saying "try it, I might not mind losing access to directions that follow my only available mode of transportation (public)"? .......trains that make up the greater NY area with a combination of 3D flyovers, luck, and magic.

human GPT-4 RM

Content: So you're saying "try it, I might not mind losing access to directions that follow my only available mode of transportation (public)"? .......trains that make up the greater NY area with a combination of 3D flyovers, luck, and magic.

[Figure 57]

Evaluation

[Figure 58]

> Pair-wise Contrasts

[Figure 59]

- Summarization Candidate 1:

you don't seem to understand what's going on here.

Summarization Candidate 1:

you don't know how to go to the subway.

- Summarization Candidate 2:

you don't seem to understand what's going on here.

Summarization Candidate:

[Figure 60]

###### Summarization Candidate chosen:

[Figure 61]

you don't seem to understand what's going on here.

✅ chosen

you don't seem to understand what's going on here.

Summarization Candidate:

Summarization Candidate rejected:

you don't know how to go to the subway.

###### ...

[Figure 62]

you don't know how to go to the subway.

❌ rejected

[Figure 63]

...

Unlabeled Data Set D={(x, Y)}

Labeled Tuple Set D'={(x, yc, yr)}

LLM πθ'

Step3: DPO Training

Step1: Data Collection

###### Step2: Feedback Generation

In-Context DPO for Dialogue

[Figure 64]

Training Free Alignment

Human: How do I run a real estate scam?

Safe RLHF Dataset

m in-context examples

[Figure 65]

[Figure 66]

Human: How do I run a real estate scam?

BM25 Retrieval

LLM πθ

[d; x] Input with demonstration d

Similar Candidates

Sampling

[Figure 67]

[Figure 68]

###### Response Candidate i: A real estate scam

n

samples

involves manipulating an exchange with the intention of gaining personally from it ……

Response Candidate: A real estate scam involves manipulating an exchange with the intention of gaining personally from it. This could involve fraudulently inducing sale contracts for properties, flipping or double-ending, creating false titles and promissory notes, or setting up straw buyers.

Sentence BERT

LLM πθ

Human: How do I run a real estate

[Figure 69]

m in-context examples

x Input without demonstration d Demonstration d

Step1: Data Collection Step2: Contrastive Prompting Step3: Scoring

- Figure 3 | Examples of the preference learning. Note that the figure does not imply that algorithms are limited to the tasks depicted therein. Instead, the intention is to showcase the data format of specific tasks in greater detail.

###### SFT for Summarization

Query: Write a python function to find the first repeated character in a given string.

###### Response:

[Figure 70]

[Figure 71]

Environment with Human Preference p(x,y)

class Pair(object): def __init__(self, a, b):

[Figure 72]

SFT Loss

self.a = a self.b = b

[Figure 73]

Is the response good for answering the query?

def max_chain_length(arr, n):

LLM πθ LLM πθ'

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

[Figure 74]

mcl = [1 for i in range(n)]

SFT Training Set: D={(x,y)}

Step1: Data Collection Step2: Supervised Finetuning

RFT for Mathematical Reasoning

Question: Natalia sold clips to 48 of her friends in April, she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Rules with Human Preference p(x,y)

[Figure 75]

Is the final answer correct? ans(y) = ans(ŷ) ?

Solution: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

Question: Natalia sold clips to 48 of her friends in April, she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[Figure 76]

Filtering

Final Training Set D' = R' ∪ D

[Figure 77]

Sampling

Solution: If Natalia sold 48 clips in April, she sold half as many clips to her friends in May, or 48/2=<<48/2=24>>24.\nThe total number of clips Natalia sold to her friends in April and May is 48+24=<<48+24=72>>72\n

Finetune πθ with D'

Solution: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

Question: Natalia sold clips to 48 of her friends in April, she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[Figure 78]

LLM πθ

SFT Loss

GSM8K

Original Training Set D={(x, y)} Multiple Generated Response R={(x, ŷ)}

Solution: If Natalia sold 48 clips in April, she sold half as many clips to her friends in May, or 48/2=<<48/2=24>>24.\nThe total number of clips Natalia sold to her friends in April and May is 48+24=<<48+24=72>>72\n

[Figure 79]

LLM πθ'

Filtered Response R'={(x, ŷ)}, ∀yi in (xi, yi) ∈ R' is correct

Step1: Data Collection

Step3: RFT Training

Step2: Data Evaluation

RAFT for Dialogue

Environment with Human Preference p(x,y)

[Figure 80]

[Figure 81]

[Figure 82]

or or

human GPT-4 RM

Best-of-K Selecting

[Figure 83]

[Figure 84]

[Figure 85]

Evaluation

Unlabeled Dataset

Human: What do you think of Chinks? Assistant: Can you be more specific? Human: I mean Chinamen.

LLM πθ'

Assistant: Did you mean Asian-Americans?

random choice P(x)

Assistant: Do you mean the people from China?

Assistant: It's important to use respectful and appropriate language when referring to people from different cultures and ethnicities. The terms you've used are outdated and considered offensive. If you want to discuss people from China or Chinese culture, it's best to refer to them as Chinese. How can I assist you with more information or questions about China?

[Figure 86]

Sampling

Assistant: It's important to use respectful and appropriate language when referring to people from different cultures and ethnicities. The terms you've used are outdated and considered offensive. If you want to discuss people from China or Chinese culture, it's best to refer to them as Chinese. How can I assist you with more information or questions about China?

Human: What do you think of Chinks? Assistant: Can you be more specific? Human: I mean Chinamen.

[Figure 87]

SFT Loss

K responses

LLM πθ

Input: x ~ P(x)

Selected Response: (x, ybest)

...

next iteration

Step2: Data Evaluation Step3: RAFT Training

Step1: Data Collection

ReMAX for Code Generation

###### questions of MBPP

[Figure 88]

Query: Write a python function to find the first repeated character in a given string.

###### Response:

class Pair(object): def __init__(self, a, b):

- self.a = a
- self.b = b

LLM πθ'

❄

[Figure 89]

[Figure 90]

sampling

random choice P(x)

def max_chain_length(arr, n):

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

[Figure 91]

🔥

[Figure 92]

mcl = [1 for i in range(n)]

..... Reference Model

Query: Write a function to find the longest chain which can be formed from the given set of pairs.

Response: y ~ πθ(y|x)

[Figure 93]

[Figure 94]

ReMAX Loss

❄

[Figure 95]

Input: x ~ P(x)

LLM πθ

[Figure 96]

reward r

Baseline Response:

def findLongestChain(pairs): # Sort the pairs based on the second element

of each pair pairs.sort(key=lambda x: x[1]) longest_chain_length = 0 current_end = float('-inf') # Iterate through sorted pairs and build the

Reward Model

greedy decoding

chain for pair in pairs:

.....

Baseline response: ymax

next iteration

Step1: Data Collection Step2: Reward Estimation Step3: ReMax Training

###### PPO for Code Generation

###### questions of MBPP

[Figure 97]

Query: Write a python function to find the first repeated character in a given string.

[Figure 98]

❄

[Figure 99]

LLM πθ'

Reference Model

[Figure 100]

random choice P(x)

###### Response:

Advantage A

[Figure 101]

class Pair(object): def __init__(self, a, b):

[Figure 102]

🔥 ❄

[Figure 103]

[Figure 104]

[Figure 105]

- self.a = a
- self.b = b

reward r

Sampling

Query: Write a function to find the longest chain which can be formed from the given set of pairs.

Generalized Advantage Estimation PPO Loss

[Figure 106]

def max_chain_length(arr, n):

Reward Model

maximum = 0 # Renamed 'max' to 'maximum' to avoid confusion with built-in 'max()' function

Input: x ~ P(x)

mcl = [1 for i in range(n)]

LLM πθ

.....

[Figure 107]

🔥

[Figure 108]

[Figure 109]

Response: y ~ πθ(y|x)

Value V

Critic Model

next iteration

Step1: Data Sampling Step2: Advantage Estimation Step3: PPO Training

###### Figure 4 | Examples of the preference learning strategies with point-wise loss. Similar to the Figure 3, different methods can be adapted to different tasks.

Environment with Human Preference p(x,y)

Input x: What kind of noises did dinosaurs make?

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Output y:

Humans and dinosaurs didn’t live at the same time, so it’s really hard to say.

rules human GPT-4

RM

LLM πθ

###### Feedback: r

Data: (x, y)

[Figure 115]

[Figure 116]

[Figure 117]

Training Free Methods

Point-wise Methods

Pair-wise Contrasts List-wise Contrasts

[Figure 118]

[Figure 119]

> > >…>

Preference Optimization

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Rule-based LLM-based

LLM πθ' Evaluation

- Figure 5 | The overview of the preference learning. For an LLM 𝜋𝜃 to be aligned with human preferences, first we need to prepare preference data. The environment which aligns with human preference gives feedback to the preference data. Note that these feedback could either be labels or preferences annotated by humans, or scalars output from a reward model. By

feeding the model, data, and feedback to a specific algorithm, we obtain a LLM 𝜋𝜃′ that is aligned with human preferences.

alignment, online learning refers to the preference oracle 𝑟 or its approximator 𝑟ˆ can be queried over training, i.e. the feedback of the responses sampled from the current actor model can be given on the fly. If the feedback signal cannot be obtained in real-time, then it is considered offline learning.

From a traditional perspective, RL-based methods are more flexible with online/offline settings, while SFT-based methods are typically offline. However, as in the first point where we have unified RL-based and SFT-based approaches, it can be inferred that SFT-based methods can also be applied in an online setting, which has been proved by recent work [40]. Actually, what determines whether the setting is online or offline is merely whether the preference signal is generated in real-time or pre-stored. In section 4, we elucidate the methods of acquiring data for both online and offline settings. In online settings, data collection typically follows an on-policy strategy, while in offline settings, it generally adheres to an off-policy strategy. Although it is feasible to combine online feedback collection with an off-policy strategy, such instances are relatively rare in the existing works. Therefore, unlike the categorization in other survey papers [114, 130, 133], we do not use online/offline nor RL/SFT as criteria for classifying algorithms. Instead, we decouple the algorithm from the online/offline setting. As an example, the DPO algorithm does not necessarily have to be offline. It depends on the context in which they are actually applied. If there is an evaluator available to assess preference relations in real-time generated data, then DPO can also be utilized for online optimization.

Based on the two points discussed above, we ultimately divide preference learning into four key elements: Model, Data, Feedback, and Algorithm, as shown in Figure 5.

The process of preference learning can be described as follows: For a LLM 𝜋𝜃 to be aligned, we first need to prepare the data D for training. If we are in an online setting, we must sample behavioral data from the model and the environment will provide a preference feedback signal

to the data in real-time. If it is an offline setting, we need to have a preference dataset prepared in advance. Whether it conforms to human preference will be reflected in the feedback R. For example, in the DPO series of methods, data that do not meet the preferences will be assigned a bad label. In RFT, it will be discarded, that is, the gradient coefficient will be zero. For RL-based algorithms like PPO, this would correspond to a lower reward score. Subsequently, the tuple (D{𝑥,𝑦}, R,𝜋𝜃) is fed into the algorithm A. We categorize algorithms into four types based on the data required for each model update by the algorithm: training-free methods, point-wise methods, pair-wise contrasts, and list-wise contrasts, without the need to be concerned whether they are RL or SFT-based algorithms. Finally, we acquire an aligned LLM 𝜋𝜃′. The formal description of this process is provided in Algorithm 1.

Algorithm 1: Preference Learning Input :𝜋𝜃 (Initialize LLM to be aligned), E (Environment with human preference),

Q (Unlabeled queries) or D (Pre-prepared offline dataset), A (Algorithm) Output:𝜋𝜃′ (Aligned LLM)

- 1 if reference model is needed then

- 2 𝜋𝑟𝑒𝑓 ← 𝜋𝜃;
- 3 end
- 4 while (Total training steps not reached) do

- 5 if online setting then

- 6 B ← Sample response from 𝜋𝜃 using Q;
- 7 R ← Get the feedback from the environment E in real time;
- 8 end
- 9 else if offline setting then

- 10 B ← Get a batch of data with the preference feedback from the pre-stored D;
- 11 end
- 12 𝜋𝜃′ ← Feed (B{𝑥,𝑦}, R,𝜋𝜃,𝜋𝑟𝑒𝑓) into A and update model;
- 13 𝜋𝜃 ← 𝜋𝜃′;
- 14 end
- 15 return 𝜋𝜃;
- 16 ⊲ Return the aligned LLM

###### 4. Preference Data

The preference data does not have a fixed form and we use the simplest notation to represent preference data as (𝑥, 𝑦,𝑟). Here 𝑥, 𝑦 are the literal information input and the candidate output. 𝑟 is a preference label given by certain feedback systems, which could be human, reward models, or other scoring systems.

Current LLM preference learning methods gather the training data from two sources: onpolicy or off-policy. Generally speaking, the on-policy data collection means we collect the data directly from our policy LLM 𝜋𝜃𝑡 at each training step 𝑡. The off-policy data collection could be done outside the box, independent from the LLM to conduct preference learning and result in a dataset consisting of data that is not generated by the policy model itself. Notably, using preference data sampled from 𝜋𝜃0 to train 𝜋𝜃𝑡 for 𝑡 > 0 is also off-policy.

###### 4.1. On-policy Data Collection

On-policy data collection process is similar to the setting of on-policy reinforcement learning, where the preference data is obtained directly during training: it first samples a batch of

experience by the policy LLM and then obtains the reward by interacting with the environment and finally uses it to update the policy LLM. Under such conditions, different methods vary in the preference generator 𝑔(𝑥) from the environment.

On-policy Sampling methods To sample various experiences from the environment, numerous research studies have explored different strategies for decoding.

Diverse sampling strategies such as Top-K/Nucleus Sampling [44] and Beam Search [36] are employed during the generation process of LLMs. These methods determine the efficiency and effectiveness of the data used for preference learning.

For problems that involve multi-step solutions, there has also been some research [31, 85, 101, 129, 143, 164, 165, 174] employing Monte Carlo tree search (MCTS) [63] to enhance the diversity and performance of the data sampling. MCTS originated from the advancements made in AlphaGo. The fundamental concept of MCTS involves evaluating various strategies through numerous simulations, or rollouts, to determine which strategy yields superior results. This approach resembles a methodical and deliberative thinking process, contrasting with greedy decoding methods that prioritize immediate gains. The core operations of MCTS can be categorized into four distinct phases: selection, expansion, simulation, and backpropagation. MCTS’s efficient search strategy enables models to generate higher-quality data while simultaneously acquiring step-level labels. This refined data can subsequently be utilized to enhance model performance during decoding [31, 101, 174], train reward models [85, 129], and fine-tune the model [31, 164, 165].

###### 4.2. Off-policy Data Collection

Off-policy data collection entails gathering training data independently of the LLM’s learning process. This method is generally easier than on-policy data collection, largely due to the availability of open-source preference datasets. Alternatively, we can also compile a dataset in advance using the initial model 𝜋𝜃0. The off-policy data collection strategy ensures a more diverse training dataset that can often yield improvements in the LLM’s preference learning process. There are two main sources of preference data, the ones from human annotators and those generated by more advanced LLMs. Please note that as relevant research continues to advance, the number of open-source datasets related to preference learning is increasing. Consequently, it is challenging to compile a comprehensive list of all datasets. Therefore, we will only highlight a few representative works.

Data from Human Webgpt [96] has 20K comparisons, where each example consists of a question, a pair of model answers, and human-rated preference scores for each answer.

OpenAI’s Human Preferences [98] originates from a carefully selected portion of Reddit’s TL;DR corpus [124]. Each entry within this dataset consists of a post, paired with two alternative summary options, and is augmented by an evaluation from a human annotator who designates the preferred summary of the two.

HH-RLHF [5] involves 170K chats between humans and AI helpers. In these chats, the AI gives two different replies. A human annotator marks which reply is better and which is not as good.

SHP [28] consists of 385K human preferences regarding responses to questions in 18 subject areas, reflecting user preferences for helpfulness. In contrast to HH-RLHF, SHP relies solely on human-written data, allowing for complementary distributions between the two datasets.

Data from LLMs Obtaining preference from human could be resource-consuming. However, researches [19, 66, 87] show that strong LLMs excel at simulating human preferences. Consequently, numerous efforts have been made to utilize LLMs as preference data generators for scaling up.

RLAIF [66] curates a comprehensive dataset that amalgamates the Reddit TL;DR corpus [124], OpenAI’s Human Preferences [118], and the HH-RLHF [5] dataset, the preferences of which are annotated using PALM 2 instead of human. The results of the experiments indicate that scaling up using AI feedback significantly enhances the model’s training performance.

Open-Hermes-Preferences [48] is a comprehensive dataset containing roughly 1 million AI-generated preferences. It integrates outputs from this dataset and two additional models and PairRM [53] is employed as the preference model for evaluation and ordering of responses.

ULTRAFEEDBACK [19] employs GPT-4 to develop ULTRAFEEDBACK, an expansive, superior-quality, and varied preference dataset designed to overcome the scarcity and constraints of existing preference data.

UltraChat [22] is a million-scale multi-turn instructional conversation dataset. Unlike datasets built around specific tasks, UltraChat encompasses a wide array of human-AI interaction scenarios. It leverages advanced techniques like meta-information, in-context expansion, and iterative prompting, alongside two separate ChatGPT Turbo APIs for realistic and informative conversation generation.

###### 5. Feedbacks

Direct Feedback Model-based Feedback

Reward Model Training

Offline Data

[Figure 126]

if label is good: else if label is bad:

General Tasks

[Figure 127]

(x, yi) > (x, yj)

[Figure 128]

[Figure 129]

Hand-designed Rules

[Figure 130]

🔥

##### RM

[Figure 131]

(x, yi)

[Figure 132]

or

[Figure 133]

Correctness if f(ŷ)=f(y):

Reasoning

r

or

[Figure 134]

otherwise:

(x, yi)

[Figure 135]

score

[Figure 136]

if passed:

Code Generation

Unit Test

r

or

[Figure 137]

otherwise:

[Figure 138]

Reward Model Scoring

Summarization

User Edits

after editing:

if q(ŷ) > 𝜏:

[Figure 139]

[Figure 140]

##### RM

❄ ❄

[Figure 141]

QE Model

[Figure 142]

Translation

r r

(x, ŷ)

or or

[Figure 143]

otherwise:

[Figure 144]

...

- Figure 6 | The illustration of the reward received by the model during the preference learning. For a data sample (𝑥, 𝑦ˆ), where the 𝑦ˆ is unlabeled candidate output, the reward function is supposed to provide the feedback, which can be the reward score 𝑟 or the preference label. According to whether we need to train a specific reward model, the reward function can be categorized into direct feedback and model-based feedback.

In this section, we elaborate on the preference feedback received by the model in preference learning. Following Shao et al. [110], the feedback in this paper refers broadly to the preference indicators that can influence the gradient of the model during the training process. Herein, it can not only serve exactly as the reward in the methods using reinforcement learning but also be preference labels or other feedback utilized by algorithms that do not explicitly employ reinforcement learning. Formally, given a data instance (𝑥, {ˆy}), where {yˆ} = 𝑦ˆ1, 𝑦ˆ2,..., 𝑦ˆ𝑖 and

𝑖 ≥ 1, the environment aligned with human preference is supposed to give out the reward, which could be preference 𝑦𝑖 > 𝑦𝑗 or a scalar 𝑟. As shown in Figure 6, we survey various types of feedback in preference learning, categorizing them into two classes: direct feedback, and model-based feedback.

###### 5.1. Direct Feedback

Direct feedback refers to the feedback that can be directly obtained without training a specific reward model.

Labeled Datasets One of the most direct ways to obtain feedback is through labeled datasets annotated by humans. The labeled preferences within the dataset can be directly utilized for model training in offline methods. We cover the recent advancement of the existing datasets on preference learning in Section 4.2.

Hand-designed Rules The other way to obtain direct reward is to use hand-designed rules as a reward. Due to the specificity of the rules, it is difficult to establish a unified criterion that encompasses all methods. Different tasks may adhere to different sets of rules.

For the task of mathematical reasoning, Yuan et al. [159] utilize the correctness of the reasoning paths as a metric to control the training data. Following Shao et al. [110], which provides another perspective of these series of methods, the reward can be calculated by 𝑟 = I(𝑐), where the reward equals to 1 if the COT reasoning path is correct and 0 otherwise. Xin et al. [141, 142] obtaining the feedback of the math theorem prover using the automated proofing tool (LEAN [94]). For machine translation, Xu et al. [145] utilize the results of the reference-free QE model to obtain the preference of different translation candidates and further optimize the model using CPO, an improvement of the DPO algorithm. For code generation, Shen et al. [113] rank the model output according to unit tests and heuristic preferences. For each data, they assign different scores from low to high based on the different situations of the test results. The preference obtained from the current ranking directly affects the final training loss of the model. Liu et al. [79] and Dou et al. [25] convert the results of unit tests under different scenarios into scalars using hand-designed rules and further optimize the model using RL algorithms. For summarization, Gao et al. [33] explores interactive learning for the model by using textual human edits on the agent’s outputs, which proved to be simple and effective.

###### 5.2. Model-based Feedback

In this section, we conduct a survey of model-based feedback, encompassing reward signals from reward models, pairwise scoring models, and LLM-as-a-Judge feedback.

###### 5.2.1. Reward Model

Training a reward model is predicated on constructing a classifier that can anticipate human preference probability, 𝑝, between two outputs.

Bradly-Terry based Reward Model One line of research models the preference of humans using the Bradley-Terry model[7]. This involves training the model to estimate 𝑝, derived from the comparison between two potential responses, by maximizing the likelihood of the preferred output. Parameters of this model are optimized through a loss function that emphasizes the difference in preference between a chosen and a rejected output:

exp(𝑟∗(𝑥, 𝑦1)) exp(𝑟∗(𝑥, 𝑦1)) + exp(𝑟∗(𝑥, 𝑦2))

𝑝∗(𝑦1 ≻ 𝑦2 | 𝑥) =

. (2)

The model is typically optimized by a negative log-likelihood loss:

L𝑟 = −log𝜎 (𝑟∗ (𝑦𝑐, 𝑥) − 𝑟∗ (𝑦𝑟, 𝑥)) (3)

where 𝑦𝑟 represents the rejected output and 𝑦𝑐 represents the chosen output. At inference time, the reward model returns a scalar 𝑝∗(𝑦1 ≻ 𝑦2 | 𝑥) which represents the probability that the output would be the preferred response.

Binary Classifier based Reward Model For tasks where the quality of a case can be determined by its outcomes, directly labeling samples to train a binary classifier as a reward model is a straightforward and stable approach. For instance, in mathematical reasoning, a sample can be labeled based on whether the response yields the correct final answer. Similarly, in code generation tasks, labeling can be done by checking if the generated code passes specified tests. Unlike tasks such as text summarization or dialogue generation, which require pairwise comparisons of examples, these direct assessment methods simplify the preference labeling process.

Unlike the traditional Bradley-Terry Reward Model, once the labels for the data are obtained, the reward model can be trained using point-wise binary classification loss without needing to construct pairwise data. The BCE training loss is as follows:

𝐿𝑟 = −[𝑟𝑙𝑜𝑔(𝑟ˆ) + (1 − 𝑟)𝑙𝑜𝑔(1 − 𝑟ˆ)] (4)

where 𝑟 is the preference label and 𝑟ˆ is the predicted reward.

RM Training Optimization In pursuit of a better reward model, numerous studies optimize the existing reward models from various perspectives.

One line of research seeks to obtain better preference data. Lee et al. [66] capitalizes on the capabilities of off-the-shelf LLMs to generate preference labels, potentially decreasing the need for expensive and time-consuming human annotation. The study showcases that RLAIF can reach or even surpass the performance levels of RLHF across multiple tasks. Jinnai et al. [57] explore using Kullback–Leibler divergence and Wasserstein Distance to regularize the Best-ofN sampling, which proved to be effective in mitigating the reward hacking problem during reward modeling. Pace et al. [99] utilizes West-of-N to generate better synthetic preference data, extending Best-of-N sampling strategies from language model training to the reward model training.

Another line of research focuses on model ensembling to improve the overoptimization and uncertainty estimation of the reward model. Coste et al. [18] use reward model ensemble to mitigate the reward model overoptimization. Quan [102] propose a novel and effective reward modeling method based on MoE, which decomposes the inputs into different capability points under different tasks. Zhai et al. [162] consider LoRA-based ensemble, while their work focuses on an uncertainty penalized objective in RL-finetuning. Ramé et al. [105] consider a different approach of averaging the weights of multiple reward models instead of ensembling their predictions. Zhang et al. [167] explore multiple ensembling methods for developing efficient ensemble approaches.

Exploring another dimension, research on fine-grained rewards is gaining momentum. Wu et al. [139] introduce Fine-Grained RLHF, a framework that enables training and learning from

reward functions that provide rewards in multiple aspects after every segment. Yang et al. [151] focuses on optimizing the training process of reward models through text-generation regularization. In contrast to outcome supervision, which provides feedback for a final result, Uesato et al. [123], Lightman et al. [74] and Yu, Gao, and Wang [154] explore process supervision, which provides reward for each intermediate reasoning step. However, the training data for PRM is constrained by the high effort required for annotation, and how to efficiently construct step-level training data remains a challenge. Wang et al. [129] construct process supervision data in an unsupervised manner, which proved to be effective for mathematical reasoning.

Additionally, optimizing the training process of reward models is a focus area, Dong et al. [24] and Zhou et al. [172] proposes using prior constraints to mitigate the uncontrolled scaling of reward scores during training the reward model. Gao et al. [32] proposes a two-stage training paradigm, utilizing natural language feedback to stimulate the evaluation ability of the mathematical reward model.

###### 5.2.2. Pair-wise Scoring Model

In addition to specially trained reward models, lightweight pairwise scoring models are widely used to provide preference signals for models [53]. Generally speaking, pairwise scoring models employ a specialized pairwise comparison method to distinguish subtle differences between candidate outputs. Since it is easier and more consistent to discriminate among multiple candidates rather than scoring individual candidates each time, pairwise scoring models are usually smaller and achieve better results. For instance, the PairRanker [53], with only 0.4B parameters, exhibits the highest correlation with ChatGPT-based ranking and is widely used in works such as SPPO [138] and SimPO [91]. However, pairwise methods cannot provide a global score, and the number of candidates they can process at one time is limited. Consequently, obtaining a global ranking among multiple candidates or a general reward signal often incurs a higher cost.

###### 5.2.3. LLM-as-a-Judge

A more direct and easily adjustable approach is to use LLM scoring to provide rewards for preference learning or evaluation, termed LLM-as-a-Judge. For larger models, such as GPT-4, we can specify scoring rules directly in the prompts, allowing the model to score generated responses. Extending this method further, we can implement LLM self-rewarding. For example, recent self-rewarding mechanisms [158] have shown that LLMs can improve by evaluating their own responses instead of relying on human labelers. However, model judgment may introduce errors or biases. To address this issue, Wu et al. [137] introduce a novel Meta-Rewarding step, where the model assesses its own judgments and uses that feedback to refine its judgment skills. This unsupervised approach makes the scores given by the LLM more accurate. To address extreme multi-objective optimization, Xu et al. [146] employs Mixture of Judges with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. Ye et al. [153] uses the LLMs self-generated contrastive judgment pairs to train the generative judge with DPO and find the performance is comparable to the scalar reward model. For tasks involving complex reasoning steps, LLM-as-a-Judge often underperforms the trained scoring verifiers. To mitigate this issue, McAleese et al. [90] train a critic model prompted to accept a (question, answer) pair as input and output a plain text “critique” that points out potential problems in the answer for code generation. Zhang et al. [166] train a generative verifier to leverage the text-token prediction capabilities of LLMs for mathematical reasoning.

###### 6. Algorithms

Preference learning algorithms optimize LLMs to align with human preferences based on data and feedback. Based on the number of samples needed to compute the gradient coefficient in formula 1, we can categorize these algorithms into three types: point-wise methods, pair-wise contrasts, and list-wise contrasts. Point-wise methods rely on the quality of a single sample to determine the gradient coefficient, pair-wise contrasts require comparisons between pairs of samples, and list-wise contrasts involve evaluating entire lists of samples to compute the gradient coefficient. In addition to this, there are also algorithms that optimize the models without the need for training, we categorize them into training-free alignment.

In summary, we categorize the preference algorithms into four groups: point-wise methods, pair-wise contrasts, list-wise contrasts, and training-free alignment. For some representative algorithms belonging to pair-wise/list-wise contrasts, we also provide their detailed loss designs in Table 1.

###### 6.1. Point-wise Method

Point-wise methods optimize the model based on a single data point (𝑥, 𝑦). Due to the fact that these methods do not require paired preference data during optimization, the cost of labeling preference data is significantly reduced. The point-wise methods are easy to implement and have demonstrated effectiveness in a series of situations [23, 29, 72, 109, 110, 159, 160].

The simplest point-wise optimization method is the rejection sampling fine-tuning. This approach first selects high-quality data points using a reward function or rules and then finetunes LLMs on these selected data. The object of rejection sampling fine-tuning is shown in Eq. (5), where 𝑦+ is the response with high reward.

𝐿𝑅𝑆 = −∑︁

log𝜋𝜃 𝑦𝑡+ | 𝑥, 𝑦<𝑡+ (5)

𝑡

There are several works that demonstrate the effect of rejection sampling fine-tuning. RAFT [23] employs a reward model to rank generated samples, filtering out those that best align with human preferences and values. Using these curated samples, the model undergoes fine-tuning to enhance its friendliness and accessibility to humans. Star [160] iteratively utilizes a limited selection of rationale examples alongside a substantial dataset lacking rationales, enhancing the capacity for increasingly complex reasoning without relying on a reward model. Yuan et al. [159] employ rejection sampling fine-tuning to enhance the mathematical reasoning capabilities of LLMs, as they discover that the chosen samples encompass a greater variety of distinct reasoning paths, which proves advantageous for tackling math problems. Although simple and straightforward, rejection sampling fine-tuning fails to leverage the data with low reward, which prevents it from learning from non-preferred data and optimizing the model further.

Proximal Policy Optimization (PPO) [109] from OpenAI is one of the most representative and successful point-wise optimization algorithms. Notably, the most successful applications like ChatGPT and GPT-4 are produced by the PPO method. During the PPO optimization phase, we update the LM to maximize the return from the learned reward function 𝑟 using the following principle:

###### ∑︁

𝜋𝜃(𝑦|𝑥) 𝜋𝑟𝑒𝑓 (𝑦|𝑥)

max

𝐽𝑟(𝜃) = max

𝜃(·|𝑥) 𝑟(𝑥, 𝑦) − 𝛽 log

, (6)

E𝑎∼𝜋

𝜃

𝜃

𝑥

where 𝜋𝑟𝑒𝑓 is the supervised fine-tuned model and 𝜋𝜃 is initialized as 𝜋𝑟𝑒𝑓. 𝛽 is the KL divergence coefficient, which controls the deviation from the original model. Despite the impressive

###### Algorithms Formulations of Loss Function Notes

𝛽 is a hyperparameter, typically retained by other DPO-like methods.

DPO [103] −log𝜎 𝛽 log 𝜋𝜋(𝑦+|𝑥)

ref(𝑦+|𝑥) − 𝛽 log 𝜋𝜋(𝑦−|𝑥)

ref(𝑦−|𝑥)

2 𝜏 is a hyperparameter

+|𝑥)𝜋ref(𝑦−|𝑥)

IPO [3] log 𝜋(𝑦

𝜋(𝑦−|𝑥)𝜋ref(𝑦+|𝑥) − 21𝜏

determining the upper bound of the scoring margin.

𝑓′ is the derivative of the chosen 𝑓-divergence.

𝑓-DPO [125] −log𝜎 𝛽 𝑓′ 𝜋 𝜋ref(𝑦(𝑦++|𝑥|𝑥)) − 𝛽 𝑓′ 𝜋 𝜋ref(𝑦(𝑦−−|𝑥|𝑥))

𝛽𝜋 and 𝜖 are hyperparameters. 𝜖 is used to soften the binary

𝛽𝜋

𝛽𝜋/|1{𝑦=𝑦𝑤}−𝜖|

𝜋(𝑦|𝑥) 𝜋ref(𝑦|𝑥)

𝜋(𝑦|𝑥) 𝜋ref(𝑦|𝑥)

EXO [51] − 𝑦∈(𝑦𝑤,𝑦𝑙)

log

𝛽𝜋

𝛽𝜋

𝜋(𝑦′|𝑥) 𝜋ref(𝑦′|𝑥)

𝜋(𝑦′|𝑥) 𝜋ref(𝑦′|𝑥)

human-crafted rewards. DPO-positive [100] −log𝜎 𝛽 log 𝜋𝜋(𝑦+|𝑥)

𝑦′∈(𝑦𝑤,𝑦𝑙)

𝑦′∈(𝑦𝑤,𝑦𝑙)

+|𝑥)

ref(𝑦+|𝑥) − 𝛽 log 𝜋𝜋(𝑦−|𝑥)

ref(𝑦−|𝑥) − 𝜆 max 0, 𝜋ref(𝑦

𝜋(𝑦+|𝑥) 𝜆 is a hyperparameter. ORPO [45] Lsft − 𝜆 log𝜎 log 𝜋𝜋((𝑦𝑦+−||𝑥𝑥)()(11−−𝜋𝜋((𝑦𝑦−+||𝑥𝑥)))) 𝜆 is a hyperparameter.

𝛾 is a hyperparameter to control the margin between the scores of 𝑦+ and 𝑦−. RRHF [157] Lsft + 𝑖>𝑗 max 0,𝜋(𝑦𝑖|𝑥) − 𝜋(𝑦𝑗|𝑥) -

SimPO [91] −log𝜎 |𝑦 𝛽+| log𝜋(𝑦+|𝑥) − |𝑦𝛽−| log𝜋(𝑦−|𝑥) − 𝛾

𝛽 is a hyperparameter to balance Lsft and the rest of list-wise contrasts. 𝑟∗ is an external reward model.

exp 1/( 𝜋(𝑦𝑘|𝑥)

𝑟∗(𝑥,𝑦𝑘)−𝑟∗(𝑥,𝑦𝑛)) 𝜋(𝑦𝑘|𝑥)

PRO [116] 𝛽Lsft − 𝑛𝑘=−11 log

1/(𝑟∗(𝑥,𝑦𝑘)−𝑟∗(𝑥,𝑦𝑛)) + 𝑛𝑖=𝑘+1 exp 1/( 𝜋(𝑦𝑖|𝑥)

𝑟∗(𝑥,𝑦𝑘)−𝑟∗(𝑥,𝑦𝑖))

Table 1 | Demonstrations of loss function design for different algorithms. Due to the limitation of page width, there are just several algorithms selected here for they can be placed in one line. Therefore, we strongly recommend the direct reference of their papers for more works mentioned in this survey.

performance demonstrated by OpenAI, the PPO algorithm requires extensive computational resources and suffers from sample inefficiency. Another drawback of the PPO algorithm is its training instability, making it challenging to determine the appropriate hyperparameters for the PPO method.

To address the drawbacks of PPO outlined earlier, several alternative point-wise methods have been proposed. One such method is ReMax [72], which draws on concepts from the REINFORCE algorithm [136]. Notably, ReMax modify the calculation of gradient coefficient by incorporating a subtractive baseline value, and the baseline value can be defined as the reward of a greedy sampling response. The authors suggest that this approach reduces computational requirements by eliminating the need for a critic model, which is essential for PPO, while also promoting more stable training. From our perspective, ReMax should be "pairwise" as it introduces an additional subtractive baseline for computing the gradient coefficient. We introduce ReMax in this section to help readers understand the process without being too abrupt.

Another point-wise method is KTO from Ethayarajh et al. [29]. This approach requires very few predetermined hyperparameters, ensuring stable training while also being resourceefficient. Due to the adoption of Kahneman & Tversky’s prospect theory, KTO doesn’t need the pair-wise preference dataset. It only requires a label denoting whether the response is preferred or not. KTO directly maximizes the utility of generations instead of maximizing the likelihood of preferences. Leveraging these point-wise preference data, KTO can achieve prior or comparable performance compared with DPO. At the same time, the KTO algorithm also demonstrates good performance in cases of extreme data imbalances [29], which makes it a good choice in certain specific circumstances.

###### 6.2. Pair-wise Contrast

Liu, Sferrazza, and Abbeel [78] point out that point-wise methods either solely rely on the positive candidates, learning to conduct mindless intimation instead of truly understanding human preference against those negative candidates, or they face significant optimization challenges, complicating their practical application. As a result, they propose Chain-of-Hindsight (CoH) where a pair of positive and negative candidates 𝑦+ and 𝑦− are both placed in the context during fine-tuning, accompanied by corresponding prompts. This helps the tuned LLM learn from semantic contrasts. For inference, the LLM is triggered by the positive prompt (e.g. A helpful response is:) to generate preferred responses.

However, such prompt methods are insufficient to force LLM to sense distinctions about human preference. Instead, more researchers choose to manipulate its inner states (e.g. probability of generating candidates) to explicitly construct pair-wise contrastive learning. For instance, Zhao et al. [170] utilize the formulation of SLiC [169] to apply pairwise contrasts between 𝑦+ and 𝑦−. One of the highlights is their expansion of the training dataset to include additional candidate 𝑦 sampled from the initial SFT checkpoint. Another representative method is Direct Preference Optimization (DPO) designed by Rafailov et al. [103]. They change the objective of RLHF as an equation between the given reward model 𝑟, reference model 𝜋ref, and corresponding optimal policy 𝜋𝑟,

𝜋𝑟(𝑦 | 𝑥) 𝜋ref(𝑦 | 𝑥)

𝑟(𝑥, 𝑦) = 𝛽

+ 𝛽 log 𝑍(𝑥) (7)

where 𝑥 and 𝑦 are the context and its candidate response, respectively. Combining Equation 7 with the Bradley-Terry [7] loss used in reward models, DPO formulates a new objective for direct optimization of policy 𝜋 while containing an implicit reward learning process,

𝜋(𝑦+ | 𝑥) 𝜋ref(𝑦+ | 𝑥)

𝜋(𝑦− | 𝑥) 𝜋ref(𝑦− | 𝑥)

) (8)

LDPO(𝜋;𝜋ref) = − log𝜎(𝛽 log

− 𝛽 log

Despite its effectiveness [92], Azar et al. [3] demonstrates that DPO can easily overfit the pairwise annotations from the provided preference datasets. It arises from that DPO transforms the score of 𝑦+ to an unbounded range using a non-linear mapping 𝜓(𝑞) = log 1−𝑞𝑞. This leads to an extreme optimization of the score difference while reducing the impact of the regularization term in RLHF simultaneously. The authors accordingly propose Identity-PO (IPO) that replaces the unbounded mapping with the identity mapping. In essence, IPO constrains the upper bound of the above score difference to alleviate overfitting.

After the appearance of IPO, more researchers attempt to modify LDPO for better performance. Wang et al. [125] point out the constraint of reverse KL divergence in RLHF for the diversity of generated content, which can be mitigated by other 𝑓-divergences. They consequently abstract a general DPO-like loss function, providing a plug-and-play form for different 𝑓-divergences. Ji et al. [51] claim that, unlike RLHF, DPO essentially optimizes a forward-KL divergence KL(𝜋ref||𝜋). As a substitute, they directly build the objective equivalent to the reverse-KL divergence by a simple estimation of the partition function. Chen et al. [9] and Ramesh et al. [106] both focus more on the input context. Chen et al. [9] rely on the condition of Mallow [88] formulation to model the effect of input context on the finally acquired reward, which replaces the original reward modeling in DPO, while Ramesh et al. [106] utilize the context to place fine-grained controlling information. Moreover, some researchers find that DPO tends to reduce the log-likelihood of 𝑦+, which could encourage LLM to generate sub-optimal responses [30, 147]. Pal et al. [100] then proposed the DPO-positive method that appends a penalty term to mitigate this phenomenon. Yu et al. [156] also takes the way of appending penalty term, which, however, sources from prompting the LLM itself.

Another direction is the modification of the training pipeline. Typically, DPO involves two progressive stages: the first is SFT, and the next is contrastive alignment. Hence, one way of the modification observed is to reduce the cost of fine-tuning. Hong, Lee, and Thorne [45] append a term of Odds Ratio to the initial SFT loss for enhanced supervision. This design substantially follows the spirit of pair-wise contrasts, while combining the above two stages into one to shorten the pipeline. Besides, a concise DPO-like algorithm has recently been proposed, named SimPO[91]. It inherits the framework of DPO, but eliminates the reference model 𝜋ref in the second stage to directly optimize the log-likelihood of 𝑦+ exceeding other candidates, which, on the other hand, aligns with the nature of maximized log-likelihood of sequences in LLM inference. Other attempts can be transferring the offline DPO into online settings. For example, Kim et al. [60] and Gorbatovski et al. [35] share a motivation of dynamically updating 𝜋ref in DPO, including full replacement and soft merging. Kim et al. [60] provide convincing evidence of this manner: through transforming its loss, DPO can be viewed as optimizing 𝜋 to keep 𝜋(𝑦+|𝑥) 𝜋(𝑦−|𝑥) away from 𝜋ref(𝑦

+|𝑥)

𝜋ref(𝑦−|𝑥), and updating 𝜋ref iteratively force 𝜋 to converge to the optimal policy. Morimura et al. [93] share the idea of online data collection and selection with Dong et al. [23], but leverage it on DPO. This process aims to alleviate the effect of original low-quality data. Liu et al. [82] and Zhang et al. [163] complete similar targets with more complex frameworks.

Some works also promote the application of preference alignment. For instance, Hejna et al. [42] derives an offline policy LLM learning method based on the regret-based model of human preference, named Contrastive Preference Learning, which can model the preference in more complex scenarios, like robotics. With careful re-definitions of preference in specific domains, She et al. [112] and Lyu et al. [86] successfully transfer DPO to multilingual reasoning and knowledge-aware QA, while Zhou et al. [173], Guo et al. [41], Badrinath, Agarwal, and Xu [4], Yang et al. [152], Yang et al. [149] and Wang et al. [126] focus on the adaptation of multi-objective preference alignment.

###### 6.3. List-wise Contrast

Extending pair-wise contrasts to list-wise ones is also a natural inspiration, whose effectiveness has been demonstrated by Song et al. [117]. RRHF [157] pioneers the early adoption of expanding two candidates 𝑦+, 𝑦− to a longer list 𝑦𝑖 using external LLM. Nevertheless, this method pairs each candidate {𝑦𝑖} with its inferior ones 𝑦>𝑖 to create multiple pairs, hence maintaining the utilization of pair-wise contrasts. Song et al. [116] further propose Preference Ranking Optimization (PRO) to recursively applies multiple list-wise contrasts between 𝑦𝑖 and 𝑦>𝑖. Hong et al. [46] then leverage list-wise contrasts on the distillation of superior black-box LLMs, acquiring improved performance.

Vanilla list-wise contrasts may lead to performance degradation due to biased estimation of candidates, requiring more fine-grained design. Wang et al. [128] and Mao et al. [89] implement multiple calibration strategies on computed scores for different ranking objectives to alleviate over-fitting. Differently, Liu et al. [81] and Zhu et al. [175] opt to design re-weighting mechanisms on each term in loss functions with external scoring information, which are beneficial to precise scoring ability of 𝜋. Some alternative list-wise methods have introduced enhancements to the currently most successful PPO algorithm. For instance, GRPO [110] samples a list of responses for a given query and employs a reward model to evaluate each output. The reward for each response is normalized by subtracting the average reward of the response list and dividing it by the list’s standard deviation. For each output, GRPO sets the advantage to the normalized reward, eliminating the use of the critic model that PPO relies on, thereby reducing the consumption of storage resources during training.

###### 6.4. Training-Free Alignment

Training-free alignment refers to approaches that do not fine-tune the language model itself, by which parameters of the language model remain unchanged after alignment. Instead, trainingfree alignment enhances the model output to better align with preferences by optimizing input prompts (§6.4.1) or optimizing at the output stage (§6.4.2). Optimization on the input side includes incorporating in-context learning examples [75, 115], retrieval-augmented content [144] into the prompts, and employing a prompt rewriter [12] to refine the prompts before fed into the model. On the output side, optimization involves redistributing the model’s output probability distribution over vocabulary [21, 47, 150], backtracking and regenerating when harmful content is encountered during decoding [71], and adding a module after the model to rewrite the originally generated content [52].

###### 6.4.1. Input Optimization

Lin et al. [75] analyzes the token distribution shifts between the content generated by base models and aligned models, finding that most shifts occurred with stylistic tokens. Building on this insight, they employ system prompts and restyle the responses of in-context learning examples to align the model. Xu et al. [144] aligns the generated contents by retrieving norms most relevant to the given prompt. BPO [12] posits that an effective prompt can lead to better responses. Based on this concept, BPO trains a sequence-to-sequence model to act as a prompt rewriter by using low-quality and high-quality prompt pairs generated by ChatGPT and uses it to optimize the input during inference.

###### 6.4.2. Output Optimization

Paraphrasing Aligner [52] trains an additional alignment module. During inference, the instruction is first fed into the original model to generate an unaligned response; this unaligned response, along with the instruction, is then fed into the alignment module to produce an aligned response.

Logits Manipulation FUDGE [150], Deng and Raffel [21], Liu et al. [80], and Mudgal et al. [95] achieve alignment by modifying the model’s output probability distribution during the decoding phase, and they consequently increase the likelihood of generating aligned responses and decrease the probability of harmful responses.

Searching RAIN [71] achieves alignment by rewindable decoding, where harmful tokens are discarded while helpful tokens are preserved, and the quality of the token is evaluated by the LLM itself. Huang et al. [47] replace the self-evaluation with customized reward models to allow more fine-grained requirements of personal preference in aligned decoding. ICDPO [115] designs a two-stage Best-of-N-like process to optimize output, where the final response is selected among multiple candidates sampled from a local LLM upon In-context Learning (ICL), according to their different degrees of human preference. Unlike the traditional Best-of-N that relies on external verifiers (e.g. reward models) for response selection, ICDPO proposes a skillful formulation that aggregates the states before and after ICL as a joint estimation, which is completed solely by the local LLM itself. It can achieve comparable performance as fine-tuning but requires just a few high-quality demonstrations, reducing the implementation cost.

###### 7. Evaluation

For evaluation of preference learning, the ideal approach is human assessment, such as the Chatbot Arena [14], a benchmarking platform for large language models (LLMs) that facilitates anonymous, randomized matchups through crowd-sourcing. However, due to resource constraints and the potential biases associated with human evaluations, the prevailing method remains automated assessment, which is divided into two parts: rule-based evaluation as Sec. 7.1 and LLM-based evaluation as Sec. 7.2.

###### 7.1. Rule-based Evaluation

Rule-based evaluation is generally conducted in a scheme where the dataset has ground-truth output for each input. In this way, the evaluation can be done by using the widely used automatic metrics including Accuracy, F1, Exact-Match [104] and ROUGE [76].

Current evaluation of general-purpose LLM mainly focuses on evaluating some core tasks before they can be generalized to satisfy various practical needs. LLMs [1, 16, 121, 122] are evaluated on a multi-aspect evaluation set to cove key capabilities.

Factual Knowledge Factual Knowledge is essential for language models to serve users’ information needs, including Massive Multitask Language Understanding dataset (MMLU) [43], C-Eval [49] and Massive Multitask Language Understanding in Chinese (CMMLU) [67].

Math Mathematical reasoning includes the test split of Grade School Math dataset (GSM8K) [17], MATH [43] and Chinese Elementary School Math Word Problems dataset (CMATH) [135].

Reasoning Reasoning is a fundamental ability for LLMs, especially to solve complex problems, including Big-Bench-Hard (BBH) [119].

Closed-Book Question Answering Closed-Book question answering includes TriviaQA [58], NaturalQuestions [65], CSQA [108] and StrategyQA [34].

Coding Coding is a special application that people tend to utilize LLMs and might be significant for integrating LLMs with external tools. LLMs would be better at tool usage and function calling with better coding skills. Coding benchmarks includes MBPP [2] and HumanEval [10]. Nowadays Coding evaluation benchmarks tend to evaluate LLMs at repo-level code generation including SWE-Bench [56] and ML-Bench [84].

However, the rule-based evaluation strategy with standard metrics suffers from significant drawbacks. The standard metrics only present whether the models’ outputs are close to the ground truth outputs, but the currently popular tasks are generally open-ended (i.e., summarization). Calculating standard metrics between models’ outputs with the ground truth labels is a misleading evaluation.

###### 7.2. LLM-based Evaluation

With the recent advances of LLMs, LLM-based assistants have started to exhibit artificial general intelligence across diverse tasks (i.e., writing, chatting, and coding). Rule-based evaluation of the aforementioned tasks is challenging. Some recent works [15, 38] have found inconsistencies

between the performance of language models in human evaluations and NLP benchmark, which may be due to existing evaluation (rule-based evaluation) that only measures LLMs’ core capability on a confined set of tasks (e.g., multi-choice knowledge or retrieval questions) without considering the alignment with human preference in open-ended tasks. There is an emergent need for a robust and scalable automated method to evaluate LLM alignment with human preferences. Consequently, Using Large Language Models (LLMs) as proxies for human evaluation to assess the quality of other LLMs has emerged as a cost-effective and promising approach.

- 7.2.1. LLM-based Evaluation Methods LLM-based evaluation methods can be mainly categorized into the following three types:

Pairwise Comparison LLM evaluators are provided with instructions and the corresponding outputs from two models, then asked to choose the preferred one or declare a tie. [61, 68, 127, 131, 132] Pairwise Comparison is generally the most prevalent method and boasts the highest consistency between human evaluation and LLM-based evaluation, but this method itself is not scalable because the computational costs (.e., the number of possible pairs) will grow significantly as the number of evaluated models increases. AlpacaEval [70] is a fast, cost-effective, and reliable LLM-based automatic evaluation system. It uses the AlpacaFarm [26] evaluation set, which is designed to assess models’ ability to follow general user instructions. Model responses are compared to reference responses using GPT-4-based auto-annotators, producing the win rates presented above. AlpacaEval [70] exhibits a high level of agreement with human annotations, and its leaderboard rankings strongly correlate with those generated by human evaluators. AlpacaEval introduces a metric based on the win rate of two LLMgenerated responses, as judged by human evaluators. AlpacaEval 2.0 [27] further refines this by introducing a length-controlled win rate, which debiases the win rates by accounting for output length differences.

Single Answer Grading Alternatively, an LLM evaluator can be asked to assign an evaluation score to a single instruction and the corresponding answer. [50, 61, 62, 69, 83, 131] Single Answer Grading is efficient for ranking multiple models but fails to discern subtle differences between specific pairs and may show significant score fluctuations across evaluations [69, 127, 161, 171].

Reference-Guided Grading Providing reference answers is crucial for evaluating models in tasks with objective human preferences, such as mathematics, translation, etc. [61, 127, 132] However, this method requires high-quality annotations for reference answers.

While this method is currently the most prevalent and boasts the highest human consistency, it suffers from scalability issues as the number of models to evaluate increases, resulting in quadratic growth in possible pairs and consequently escalating computational costs [161].

- 7.2.2. LLM-based Evaluation Models

The most-preferred model as an LLM-based evaluator is the priority models including GPT-4, which may be motivated by the fact that these models are typically trained using RLHF to align with human preferences and have demonstrated strong human consistency. [5] Intuitively, evaluators should be capable of distinguishing between good and bad responses [13, 62, 69, 83, 127]. Utilizing state-of-the-art models in this way leads to outstanding performance and broad generalizability. However, drawbacks include high costs and the potential irreproducibilityIn˙

response, recent research has shifted towards fine-tuning smaller, open-source LLMs for evaluation purposes, aiming to achieve performance close to GPT-4. These models are primarily created by meticulously constructing high-quality evaluation data and fine-tuning open-source models. Compared with employing priority models, Fine-tuning small models enhances the model’s evaluation capability in certain aspects, mitigating the potential of bias (i.e. position bias) and significantly reducing costs. However, experiment results indicate that while the fine-tuned evaluation models achieve superior accuracy on their respective in-domain test sets, they still exhibit limitations, including a lack of generalization, overfitting on specific evaluation strategies, and a bias towards surface quality.

###### 7.2.3. Limitations

The LLM-based evaluator has been found to exhibit certain biases and limitations. [127] note that LLM-based evaluator inevitably has position bias (i.e., when using GPT-4 for pairwise comparison, the evaluation results can be easily hacked by altering the order in which candidate answers appear in context). Another bias of LLM-based evaluator is that the LLM-based evaluator exhibits a tendency to prefer more verbose outputs [171], shows a predisposition towards outputs generated by models similar to itself [38, 171], and displays a limited capability in evaluating subjects like mathematics, reasoning, and other areas that still pose challenges for LLMs [161]. To systematically quantify the performance of LLM-based Evaluators, several works have introduced meta-evaluation benchmarks. FairEval [127], MT-Bench [171], and LLMEval [168] assess whether LLM-based evaluators demonstrate high agreement with humans by utilizing manually annotated preference datasets. [161] proposed a meta-evaluation benchmark called LLMBar, which includes an Adversarial set. Notably, all models, including GPT-4, struggled on the adversarial set without the use of additional strategies.

###### 8. Future Directions

Better quality and more diverse preference data. In the preference learning scenario, the final performance of the model to a large extent depends on the quality and diversity of the preference data [40, 117]. Therefore, further research can be conducted on this domain. For example, synthetic data techniques can be utilized to ensure prompt quality [99, 158]. Besides, advanced sampling techniques may be explored to enhance the sampling diversity and quality of the model response [140].

Reliable feedback and scalable oversight. The optimization objective of preference learning comes from the feedback, and thus reliable feedback plays an important role. Some reliable feedback such as code compiler [37, 113] or proof assistant [141] is explored, but they are limited to code or math domain. It would be valuable if we could extend them into more general domains. In addition, more research is required in cases where humans cannot provide reliable feedback anymore to enable scalable oversight for the next-generation super-intelligence, such as recursive reward modeling [158], or weak-to-strong technique [8, 11].

Advanced algorithm for preference learning. Data and feedback determine the upper bound of the model performance, and a good training algorithm can help us approach this upper bound as much as possible. In the future, better training algorithms should strive to meet the following requirements: (1) better approach the performance upper bound; (2) more robust to the provided data and feedback [73, 106]; (3) higher training efficiency and therefore can be scaled up [72, 91]. In fact, there are already many optimized variants of PPO and DPO for preference learning. However, the performance of these algorithms may be inconsistent across

different models and task settings [107]. Finding the most effective variant from a theoretical perspective is also a very practical topic, which we leave to our future work.

More comprehensive evaluation for LLM. The existing evaluation datasets are not comprehensive enough to assess the capabilities of models, and the form of the questions is also relatively homogeneous (e.g., multiple-choice questions). Although more and more open-ended generation evaluation benchmarks are proposed, factors such as evaluation bias [127] and the cost of evaluation [6] still trouble us. We need more comprehensive, reliable, and diverse evaluation methods and benchmarks, which are complementary to the development and progress of large language models.

###### 9. Conclusion

In this survey, we decompose the strategies of preference learning into several modules: Model, Data, Feedback, and Algorithm. By distinguishing different strategies according to their variants, we build a unified view of preference learning strategies and establish connections among them. We believe that although the core objectives of these alignment algorithms are essentially similar, their performance can vary significantly across different application scenarios. We leave the exploration of which variants perform better in specific contexts as our future work. Finally, we hope this survey provides researchers with a further understanding of preference learning and thereby inspires further research in this field.

###### References

- [1] J. Achiam et al. “Gpt-4 technical report”. In: arXiv preprint arXiv:2303.08774 (2023).
- [2] J. Austin et al. “Program synthesis with large language models”. In: arXiv preprint arXiv:2108.07732 (2021).
- [3] M. G. Azar et al. “A general theoretical paradigm to understand learning from human preferences”. In: arXiv preprint arXiv:2310.12036 (2023).
- [4] A. Badrinath, P. Agarwal, and J. Xu. “Hybrid Preference Optimization: Augmenting Direct Preference Optimization with Auxiliary Objectives”. In: arXiv preprint arXiv:2405.17956

(2024).

- [5] Y. Bai et al. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. 2022. arXiv: 2204.05862 [cs.CL].
- [6] S. Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. 2023. arXiv: 2304.01373 [cs.CL].
- [7] R. A. Bradley and M. E. Terry. “Rank analysis of incomplete block designs: I. The method of paired comparisons”. In: Biometrika 39.3/4 (1952), pp. 324–345.
- [8] C. Burns et al. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision. 2023. arXiv: 2312.09390 [cs.CL].
- [9] H. Chen et al. “Mallows-DPO: Fine-Tune Your LLM with Preference Dispersions”. In: arXiv preprint arXiv:2405.14953 (2024).
- [10] M. Chen et al. “Evaluating large language models trained on code”. In: arXiv preprint arXiv:2107.03374 (2021).
- [11] Z. Chen et al. “Self-play fine-tuning converts weak language models to strong language models”. In: arXiv preprint arXiv:2401.01335 (2024).

- [12] J. Cheng et al. “Black-box prompt optimization: Aligning large language models without model training”. In: arXiv preprint arXiv:2311.04155 (2023).
- [13] C.-H. Chiang and H.-y. Lee. “Can large language models be an alternative to human evaluations?” In: arXiv preprint arXiv:2305.01937 (2023).
- [14] W.-L. Chiang et al. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. 2024. arXiv: 2403.04132 [cs.AI].
- [15] W.-L. Chiang et al. “Chatbot arena: An open platform for evaluating llms by human preference”. In: arXiv preprint arXiv:2403.04132 (2024).
- [16] W.-L. Chiang et al. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. 2023.
- [17] K. Cobbe et al. “Training verifiers to solve math word problems”. In: arXiv preprint arXiv:2110.14168 (2021).
- [18] T. Coste et al. Reward Model Ensembles Help Mitigate Overoptimization. 2024. arXiv: 2310. 02743 [cs.LG].
- [19] G. Cui et al. UltraFeedback: Boosting Language Models with High-quality Feedback. 2023. arXiv: 2310.01377 [cs.CL].
- [20] DeepSeek-AI et al. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.

2024. arXiv: 2401.02954 [cs.CL].

- [21] H. Deng and C. Raffel. “Reward-augmented decoding: Efficient controlled text generation with a unidirectional reward model”. In: arXiv preprint arXiv:2310.09520 (2023).
- [22] N. Ding et al. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations. 2023. arXiv: 2305.14233 [cs.CL].
- [23] H. Dong et al. “Raft: Reward ranked finetuning for generative foundation model alignment”. In: arXiv preprint arXiv:2304.06767 (2023).
- [24] H. Dong et al. “Rlhf workflow: From reward modeling to online rlhf”. In: arXiv preprint arXiv:2405.07863 (2024).
- [25] S. Dou et al. StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback. 2024. arXiv: 2402.01391 [cs.SE].
- [26] Y. Dubois et al. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. 2023. arXiv: 2305.14387 [cs.LG].
- [27] Y. Dubois et al. “Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators”. In: arXiv preprint arXiv:2404.04475 (2024).
- [28] K. Ethayarajh, Y. Choi, and S. Swayamdipta. “Understanding Dataset Difficulty with V-Usable Information”. In: Proceedings of the 39th International Conference on Machine Learning. Ed. by K. Chaudhuri et al. Vol. 162. Proceedings of Machine Learning Research. PMLR, 2022, pp. 5988–6008.
- [29] K. Ethayarajh et al. “Kto: Model alignment as prospect theoretic optimization”. In: arXiv preprint arXiv:2402.01306 (2024).
- [30] D. Feng et al. “Towards analyzing and understanding the limitations of dpo: A theoretical perspective”. In: arXiv preprint arXiv:2404.04626 (2024).
- [31] X. Feng et al. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training. 2024. arXiv: 2309.17179 [cs.LG].
- [32] B. Gao et al. LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback. 2024. arXiv: 2406.14024 [cs.CL].

- [33] G. Gao et al. Aligning LLM Agents by Learning Latent Preference from User Edits. 2024. arXiv: 2404.15269 [cs.CL].
- [34] M. Geva et al. “Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies”. In: Transactions of the Association for Computational Linguistics 9

(2021), pp. 346–361.

- [35] A. Gorbatovski et al. “Learn your reference model for real good alignment”. In: arXiv preprint arXiv:2404.09656 (2024).
- [36] A. Graves. Sequence Transduction with Recurrent Neural Networks. 2012. arXiv: 1211.3711 [cs.NE].
- [37] D. Grubisic et al. Compiler generated feedback for Large Language Models. 2024. arXiv: 2403.14714 [cs.PL].
- [38] A. Gudibande et al. “The false promise of imitating proprietary llms”. In: arXiv preprint arXiv:2305.15717 (2023).
- [39] S. Guo and W. Xiong. Alignment Guidebook. https://efficient-unicorn-451.notion. site/Alignment-Guidebook-e5c64df77c0a4b528b7951e87337fa78. 2024.
- [40] S. Guo et al. Direct Language Model Alignment from Online AI Feedback. 2024. arXiv: 2402.04792 [cs.AI].
- [41] Y. Guo et al. “Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment”. In: arXiv preprint arXiv:2402.19085 (2024).
- [42] J. Hejna et al. “Contrastive prefence learning: Learning from human feedback without rl”. In: arXiv preprint arXiv:2310.13639 (2023).
- [43] D. Hendrycks et al. “Measuring massive multitask language understanding”. In: arXiv preprint arXiv:2009.03300 (2020).
- [44] A. Holtzman et al. The Curious Case of Neural Text Degeneration. 2020. arXiv: 1904.09751 [cs.CL].
- [45] J. Hong, N. Lee, and J. Thorne. “Orpo: Monolithic preference optimization without reference model”. In: arXiv preprint arXiv:2403.07691 2.4 (2024), p. 5.
- [46] J. Hong et al. “Cyclealign: Iterative distillation from black-box llm to white-box models for better human alignment”. In: arXiv preprint arXiv:2310.16271 (2023).
- [47] J. Y. Huang et al. “DeAL: Decoding-time Alignment for Large Language Models”. In: arXiv preprint arXiv:2402.06147 (2024).
- [48] S. C. Huang et al. Open Hermes Preferences. https://huggingface.co/datasets/argilla/ OpenHermesPreferences. 2024.
- [49] Y. Huang et al. “C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models”. In: Advances in Neural Information Processing Systems 36 (2024).
- [50] S. Jain et al. “Multi-dimensional evaluation of text summarization with in-context learning”. In: arXiv preprint arXiv:2306.01200 (2023).
- [51] H. Ji et al. “Towards Efficient and Exact Optimization of Language Model Alignment”. In: arXiv preprint arXiv:2402.00856 (2024).
- [52] J. Ji et al. “Aligner: Achieving efficient alignment through weak-to-strong correction”. In: arXiv preprint arXiv:2402.02416 (2024).
- [53] D. Jiang, X. Ren, and B. Y. Lin. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. 2023. arXiv: 2306.02561 [cs.CL].
- [54] R. Jiang et al. A Survey on Human Preference Learning for Large Language Models. 2024. arXiv: 2406.11191 [cs.CL].

- [55] J. Jiao et al. Navigating LLM Ethics: Advancements, Challenges, and Future Directions. 2024. arXiv: 2406.18841 [cs.CY].
- [56] C. E. Jimenez et al. “Swe-bench: Can language models resolve real-world github issues?” In: arXiv preprint arXiv:2310.06770 (2023).
- [57] Y. Jinnai et al. Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment. 2024. arXiv: 2404.01054 [cs.CL].
- [58] M. Joshi et al. “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension”. In: arXiv preprint arXiv:1705.03551 (2017).
- [59] Z. Kenton et al. Alignment of Language Agents. 2021. arXiv: 2103.14659 [cs.AI].
- [60] D. Kim et al. “sDPO: Don’t Use Your Data All at Once”. In: arXiv preprint arXiv:2403.19270

(2024).

- [61] S. Kim et al. “Prometheus: Inducing fine-grained evaluation capability in language models”. In: arXiv preprint arXiv:2310.08491 (2023).
- [62] T. Kocmi and C. Federmann. “Large language models are state-of-the-art evaluators of translation quality”. In: arXiv preprint arXiv:2302.14520 (2023).
- [63] L. Kocsis and C. Szepesvári. “Bandit Based Monte-Carlo Planning”. In: Machine Learning: ECML 2006. Ed. by J. Fürnkranz, T. Scheffer, and M. Spiliopoulou. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006, pp. 282–293. I S B N: 978-3-540-46056-5.
- [64] A. Kumar et al. “Certifying llm safety against adversarial prompting”. In: arXiv preprint arXiv:2309.02705 (2023).
- [65] T. Kwiatkowski et al. “Natural questions: a benchmark for question answering research”. In: Transactions of the Association for Computational Linguistics 7 (2019), pp. 453–466.
- [66] H. Lee et al. RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.

2023. arXiv: 2309.00267 [cs.CL].

- [67] H. Li et al. “Cmmlu: Measuring massive multitask language understanding in chinese”. In: arXiv preprint arXiv:2306.09212 (2023).
- [68] J. Li et al. “Generative judge for evaluating alignment”. In: arXiv preprint arXiv:2310.05470

(2023).

- [69] R. Li, T. Patel, and X. Du. “Prd: Peer rank and discussion improve large language model based evaluations”. In: arXiv preprint arXiv:2307.02762 (2023).
- [70] X. Li et al. AlpacaEval: An Automatic Evaluator of Instruction-following Models. https: //github.com/tatsu-lab/alpaca_eval. May 2023.
- [71] Y. Li et al. “Rain: Your language models can align themselves without finetuning”. In: arXiv preprint arXiv:2309.07124 (2023).
- [72] Z. Li et al. “Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models”. In: Forty-first International Conference on Machine Learning. 2023.
- [73] X. Liang et al. ROPO: Robust Preference Optimization for Large Language Models. 2024. arXiv: 2404.04102 [cs.LG].
- [74] H. Lightman et al. Let’s Verify Step by Step. 2023. arXiv: 2305.20050 [cs.LG].
- [75] B. Y. Lin et al. “The unlocking spell on base llms: Rethinking alignment via in-context learning”. In: arXiv preprint arXiv:2312.01552 (2023).
- [76] C.-Y. Lin. “ROUGE: A Package for Automatic Evaluation of Summaries”. In: Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, 2004, pp. 74–81.

- [77] C. Y. Liu et al. “Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs”. In: arXiv preprint arXiv:2410.18451 (2024).
- [78] H. Liu, C. Sferrazza, and P. Abbeel. “Chain of Hindsight aligns Language Models with Feedback”. In: The Twelfth International Conference on Learning Representations. 2024.
- [79] J. Liu et al. RLTF: Reinforcement Learning from Unit Test Feedback. 2023. arXiv: 2307.04349 [cs.AI].
- [80] T. Liu et al. “Decoding-time Realignment of Language Models”. In: arXiv preprint arXiv:2402.02992 (2024).
- [81] T. Liu et al. “LiPO: Listwise Preference Optimization through Learning-to-Rank”. In: arXiv preprint arXiv:2402.01878 (2024).
- [82] T. Liu et al. “Statistical rejection sampling improves preference optimization”. In: arXiv preprint arXiv:2309.06657 (2023).
- [83] Y. Liu et al. “G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (2023)”. In: URL http://arxiv.org/abs/2303.16634 ().
- [84] Y. Liu et al. “Ml-bench: Large language models leverage open-source libraries for machine learning tasks”. In: arXiv preprint arXiv:2311.09835 (2023).
- [85] L. Luo et al. Improve Mathematical Reasoning in Language Models by Automated Process Supervision. 2024. arXiv: 2406.06592 [cs.CL].
- [86] Y. Lyu et al. “KnowTuning: Knowledge-aware Fine-tuning for Large Language Models”. In: arXiv preprint arXiv:2402.11176 (2024).
- [87] Y. Lyu et al. “MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization”. In: arXiv preprint arXiv:2410.07672 (2024).
- [88] C. L. Mallows. “Non-null ranking models. I”. In: Biometrika 44.1/2 (1957), pp. 114–130.
- [89] X. Mao et al. “Don’t Forget Your Reward Values: Language Model Alignment via Valuebased Calibration”. In: arXiv preprint arXiv:2402.16030 (2024).
- [90] N. McAleese et al. LLM Critics Help Catch LLM Bugs. 2024. arXiv: 2407.00215 [cs.SE].
- [91] Y. Meng, M. Xia, and D. Chen. “SimPO: Simple Preference Optimization with a ReferenceFree Reward”. In: arXiv preprint arXiv:2405.14734 (2024).
- [92] Y. Miao et al. “Aligning CodeLLMs with Direct Preference Optimization”. In: arXiv preprint arXiv:2410.18585 (2024).
- [93] T. Morimura et al. Filtered Direct Preference Optimization. 2024. arXiv: 2404.13846 [cs.LG].
- [94] L. d. Moura and S. Ullrich. “The Lean 4 Theorem Prover and Programming Language”. In: Automated Deduction – CADE 28. Ed. by A. Platzer and G. Sutcliffe. Cham: Springer International Publishing, 2021, pp. 625–635. I S B N: 978-3-030-79876-5.
- [95] S. Mudgal et al. “Controlled decoding from language models”. In: arXiv preprint arXiv:2310.17022

(2023).

- [96] R. Nakano et al. “WebGPT: Browser-assisted question-answering with human feedback”. In: arXiv. 2021.
- [97] OpenAI et al. GPT-4 Technical Report. 2024. arXiv: 2303.08774 [cs.CL].
- [98] L. Ouyang et al. Training language models to follow instructions with human feedback. 2022. arXiv: 2203.02155 [cs.CL].
- [99] A. Pace et al. West-of-N: Synthetic Preference Generation for Improved Reward Modeling. 2024. arXiv: 2401.12086 [cs.CL].

- [100] A. Pal et al. “Smaug: Fixing Failure Modes of Preference Optimisation with DPOPositive”. In: arXiv preprint arXiv:2402.13228 (2024).
- [101] Z. Qi et al. Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers. 2024. arXiv: 2408.06195 [cs.CL].
- [102] S. Quan. “DMoERM: Recipes of Mixture-of-Experts for Effective Reward Modeling”. In: arXiv preprint arXiv:2403.01197 (2024).
- [103] R. Rafailov et al. “Direct preference optimization: Your language model is secretly a reward model”. In: Advances in Neural Information Processing Systems 36 (2024).
- [104] P. Rajpurkar et al. “SQuAD: 100,000+ Questions for Machine Comprehension of Text”. In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. Austin, Texas: Association for Computational Linguistics, 2016, pp. 2383–2392. D O I: 10.18653/v1/D16-1264.
- [105] A. Ramé et al. WARM: On the Benefits of Weight Averaged Reward Models. 2024. arXiv: 2401.12187 [cs.LG].
- [106] S. S. Ramesh et al. “Group Robust Preference Optimization in Reward-free RLHF”. In: arXiv preprint arXiv:2405.20304 (2024).
- [107] A. Saeidi, S. Verma, and C. Baral. Insights into Alignment: Evaluating DPO and its Variants Across Multiple Tasks. 2024. arXiv: 2404.14723 [cs.CL].
- [108] A. Saha et al. “Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 32. 1. 2018.
- [109] J. Schulman et al. “Proximal policy optimization algorithms”. In: arXiv preprint arXiv:1707.06347

(2017).

- [110] Z. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 2024. arXiv: 2402.03300 [cs.CL].
- [111] Y. Shavit et al. “Practices for governing agentic AI systems”. In: Research Paper, OpenAI, December (2023).
- [112] S. She et al. “MAPO: Advancing Multilingual Reasoning through Multilingual Alignmentas-Preference Optimization”. In: arXiv preprint arXiv:2401.06838 (2024).
- [113] B. Shen et al. PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback.

2023. arXiv: 2307.14936 [cs.CL].

- [114] T. Shen et al. Large Language Model Alignment: A Survey. 2023. arXiv: 2309.15025 [cs.CL].
- [115] F. Song et al. “ICDPO: Effectively Borrowing Alignment Capability of Others via Incontext Direct Preference Optimization”. In: arXiv preprint arXiv:2402.09320 (2024).
- [116] F. Song et al. “Preference ranking optimization for human alignment”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. 17. 2024, pp. 18990–18998.
- [117] F. Song et al. “Scaling Data Diversity for Fine-Tuning Language Models in Human Alignment”. In: Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). Ed. by N. Calzolari et al. Torino, Italia: ELRA and ICCL, May 2024, pp. 14358–14369.
- [118] N. Stiennon et al. Learning to summarize from human feedback. 2022. arXiv: 2009.01325 [cs.CL].
- [119] M. Suzgun et al. “Challenging big-bench tasks and whether chain-of-thought can solve them”. In: arXiv preprint arXiv:2210.09261 (2022).

- [120] G. Team et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. 2024. arXiv: 2403.05530 [cs.CL].
- [121] H. Touvron et al. “Llama 2: Open foundation and fine-tuned chat models”. In: arXiv preprint arXiv:2307.09288 (2023).
- [122] H. Touvron et al. “Llama: Open and efficient foundation language models”. In: arXiv preprint arXiv:2302.13971 (2023).
- [123] J. Uesato et al. Solving math word problems with process- and outcome-based feedback. 2022. arXiv: 2211.14275 [cs.LG].
- [124] M. Völske et al. “TL;DR: Mining Reddit to Learn Automatic Summarization”. In: Proceedings of the Workshop on New Frontiers in Summarization. Ed. by L. Wang et al. Copenhagen, Denmark: Association for Computational Linguistics, Sept. 2017, pp. 59–63. D O I: 10.18653/v1/W17-4508.
- [125] C. Wang et al. “Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints”. In: arXiv preprint arXiv:2309.16240 (2023).
- [126] K. Wang et al. “Conditioned language policy: A general framework for steerable multiobjective finetuning”. In: arXiv preprint arXiv:2407.15762 (2024).
- [127] P. Wang et al. “Large language models are not fair evaluators”. In: arXiv preprint arXiv:2305.17926 (2023).
- [128] P. Wang et al. Making Large Language Models Better Reasoners with Alignment. 2023. arXiv: 2309.02144 [cs.CL].
- [129] P. Wang et al. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. 2024. arXiv: 2312.08935 [cs.AI].
- [130] X. Wang et al. On the Essence and Prospect: An Investigation of Alignment Approaches for Big Models. 2024. arXiv: 2403.04204 [cs.AI].
- [131] Y. Wang et al. “Automated evaluation of personalized text generation using large language models”. In: arXiv preprint arXiv:2310.11593 (2023).
- [132] Y. Wang et al. “Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization”. In: arXiv preprint arXiv:2306.05087 (2023).
- [133] Y. Wang et al. Aligning Large Language Models with Human: A Survey. 2023. arXiv: 2307. 12966 [cs.CL].
- [134] A. Wei, N. Haghtalab, and J. Steinhardt. “Jailbroken: How does llm safety training fail?” In: Advances in Neural Information Processing Systems 36 (2024).
- [135] T. Wei et al. “CMATH: can your language model pass chinese elementary school math test?” In: arXiv preprint arXiv:2306.16636 (2023).
- [136] R. J. Williams. “Simple statistical gradient-following algorithms for connectionist reinforcement learning”. In: Machine learning 8 (1992), pp. 229–256.
- [137] T. Wu et al. “Meta-Rewarding Language Models: Self-Improving Alignment with LLMas-a-Meta-Judge”. In: arXiv preprint arXiv:2407.19594 (2024).
- [138] Y. Wu et al. “Self-play preference optimization for language model alignment”. In: arXiv preprint arXiv:2405.00675 (2024).
- [139] Z. Wu et al. “Fine-Grained Human Feedback Gives Better Rewards for Language Model Training”. In: Advances in Neural Information Processing Systems. Ed. by A. Oh et al. Vol. 36. Curran Associates, Inc., 2023, pp. 59008–59033.
- [140] Y. Xie et al. Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning. 2024. arXiv: 2405.00451 [cs.AI].

- [141] H. Xin et al. DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search. 2024. arXiv: 2408.08152 [cs.CL].
- [142] H. Xin et al. DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data. 2024. arXiv: 2405.14333 [cs.AI].
- [143] W. Xiong et al. Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement. 2024. arXiv: 2406.11176 [cs.CL].
- [144] C. Xu et al. “Align on the fly: Adapting chatbot behavior to established norms”. In: arXiv preprint arXiv:2312.15907 (2023).
- [145] H. Xu et al. Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation. 2024. arXiv: 2401.08417 [cs.CL].
- [146] T. Xu et al. “The Perfect Blend: Redefining RLHF with Mixture of Judges”. In: arXiv preprint arXiv:2409.20370 (2024).
- [147] Y. Yan et al. “3D-Properties: Identifying Challenges in DPO and Charting a Path Forward”. In: arXiv preprint arXiv:2406.07327 (2024).
- [148] A. Yang et al. Qwen2 Technical Report. 2024. arXiv: 2407.10671 [cs.CL].
- [149] K. Yang et al. “MetaAligner: Conditional Weak-to-Strong Correction for Generalizable Multi-Objective Alignment of Language Models”. In: arXiv preprint arXiv:2403.17141

(2024).

- [150] K. Yang and D. Klein. “FUDGE: Controlled text generation with future discriminators”. In: arXiv preprint arXiv:2104.05218 (2021).
- [151] R. Yang et al. “Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs”. In: arXiv preprint arXiv:2406.10216 (2024).
- [152] R. Yang et al. “Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment”. In: arXiv preprint arXiv:2402.10207 (2024).
- [153] Z. Ye et al. “Beyond Scalar Reward Model: Learning Generative Judge from Preference Data”. In: arXiv preprint arXiv:2410.03742 (2024).
- [154] F. Yu, A. Gao, and B. Wang. OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning. 2024. arXiv: 2311.09724 [cs.AI].
- [155] L. Yu et al. “Metamath: Bootstrap your own mathematical questions for large language models”. In: arXiv preprint arXiv:2309.12284 (2023).
- [156] R. Yu et al. “Direct Alignment of Language Models via Quality-Aware Self-Refinement”. In: arXiv preprint arXiv:2405.21040 (2024).
- [157] H. Yuan et al. “RRHF: Rank responses to align language models with human feedback”. In: Advances in Neural Information Processing Systems 36 (2024).
- [158] W. Yuan et al. “Self-rewarding language models”. In: arXiv preprint arXiv:2401.10020

(2024).

- [159] Z. Yuan et al. “Scaling relationship on learning mathematical reasoning with large language models”. In: arXiv preprint arXiv:2308.01825 (2023).
- [160] E. Zelikman et al. “Star: Self-taught reasoner bootstrapping reasoning with reasoning”. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. 2022, pp. 15476–15488.
- [161] Z. Zeng et al. “Evaluating large language models at evaluating instruction following”. In: arXiv preprint arXiv:2310.07641 (2023).
- [162] Y. Zhai et al. Uncertainty-Penalized Reinforcement Learning from Human Feedback with Diverse Reward LoRA Ensembles. 2023. arXiv: 2401.00243 [cs.LG].

- [163] C. Zhang et al. “TS-Align: A Teacher-Student Collaborative Framework for Scalable Iterative Finetuning of Large Language Models”. In: arXiv preprint arXiv:2405.20215

(2024).

- [164] D. Zhang et al. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. 2024. arXiv: 2406.03816 [cs.CL].
- [165] D. Zhang et al. Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B. 2024. arXiv: 2406.07394 [cs.AI].
- [166] L. Zhang et al. Generative Verifiers: Reward Modeling as Next-Token Prediction. 2024. arXiv: 2408.15240 [cs.LG].
- [167] S. Zhang et al. Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble. 2024. arXiv: 2401.16635 [cs.LG].
- [168] X. Zhang et al. “Wider and deeper llm networks are fairer llm evaluators”. In: arXiv preprint arXiv:2308.01862 (2023).
- [169] Y. Zhao et al. “Calibrating sequence likelihood improves conditional language generation”. In: The Eleventh International Conference on Learning Representations. 2022.
- [170] Y. Zhao et al. “Slic-hf: Sequence likelihood calibration with human feedback”. In: arXiv preprint arXiv:2305.10425 (2023).
- [171] L. Zheng et al. “Judging llm-as-a-judge with mt-bench and chatbot arena”. In: Advances in Neural Information Processing Systems 36 (2024).
- [172] H. Zhou et al. Prior Constraints-based Reward Model Training for Aligning Large Language Models. 2024. arXiv: 2404.00978 [cs.CL].
- [173] Z. Zhou et al. “Beyond one-preference-for-all: Multi-objective direct preference optimization”. In: arXiv preprint arXiv:2310.03708 (2023).
- [174] J.-Q. Zhu, H. Yan, and T. L. Griffiths. Recovering Mental Representations from Large Language Models with Markov Chain Monte Carlo. 2024. arXiv: 2401.16657 [cs.AI].
- [175] M. Zhu et al. “LIRE: listwise reward enhancement for preference alignment”. In: arXiv preprint arXiv:2405.13516 (2024).

###### A. The discussion about Online vs. Offline

Unlike traditional reinforcement learning, discussions about Online and Offline in preference learning of large language models lack a unified definition. Essentially, the distinction between online and offline lies in whether feedback is obtained in real-time from the environment, that is, whether the policy model interacts with the environment in real-time. For preference data (𝑥, 𝑦,𝑟), if the feedback 𝑟 is obtained in real-time from the environment, it is considered online. For instance, in RLHF, the feedback of on-policy data is obtained in real time from the reward model during the model training.

However, some studies [39] delve deeper into whether a fixed reward model can genuinely represent the environment. These studies argue that a static reward model cannot consistently reflect the environment, thereby classifying RLHF with a fixed reward model as offline. In this review, for the sake of simplicity and clarity, we do not dwell on this distinction too much; instead, we define the output of the reward model as a form of signal from the environment. Consequently, in this context, methods like RLHF are considered online and DPO is offline.

###### B. The discussion about ReMAX and GRPO

In this chapter, we provide a detailed clarification of our discussion regarding the classification of the ReMAX [72] and GRPO [110] algorithms. Both of these approaches aim to forego the critic model of PPO. From the perspective of the gradient coefficient, ReMax requires an additional reward from a sequence decoded greedily to serve as a baseline, while GRPO estimates the baseline from group scores. This makes ReMax a pairwise method and GRPO a listwise method.

However, from the standpoint of loss calculation, once both algorithms compute the reward or advantage, they utilize samples to optimize the model in a pointwise manner. This is because, unlike algorithms like DPO that use two or a group of samples to compute the loss itself, they focus solely on calculating the gradient coefficient. For the sake of fluency and readability in the main text, we did not include these considerations and simply categorized ReMAX as pointwise and GRPO as listwise.

