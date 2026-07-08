# arXiv:2602.20913v2[cs.CV]15Apr2026

## LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding

Jihao Qiu1, Lingxi Xie2, Xinyue Huo2, Qi Tian2*, Qixiang Ye1* 1University of Chinese Academy of Sciences 2Huawei Consumer Business Group

qiujihao19@mails.ucas.ac.cn 198808xc@gmail.com xinyueh@mail.ustc.edu.cn tian.qi1@huawei.com qxye@ucas.ac.cn

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

50

###### How would you answer the question?

###### How do the two ladybugs escape the attack from a group of mantises?

45

From the initial sparse frames, the scene appears to take place outdoors, possibly a forest. I will examine the segment where the forest first appears.

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

- 1

VCA VideoTree

VideoAgent Ours

Qwen2VL-72B

- 2
- 3
- 4
- 5

1 2

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

This segment shows no mantises, suggesting limited relevance. I will proceed to the segment where the forest appears again.

40

Accuracy(%)

irrelevant

3

In this segment, mantises and ladybugs are present and appear to interact. I will focus on the part where this interaction occurs.

35

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Mantises attack ladybugs, but only the ladybugs are seen escaping. I will check the segment preceding the mantises’ safe boarding of the ship.

4

30

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

This segment reveals that the two ladybugs escape by boarding a small boat under the spider’s cover.

25

60 120 180 240 …

1000

Time Consuming (s)

Figure 1. Motivation and performance comparison. Left: For efficient understanding of long video, the algorithm shall learn to fetch and perceive information effectively, where the core abilities are: (1) judging whether collected information is sufficient for answering, and (2) if not, navigating to the next clip most likely to contain useful information. Drawing style was inspired by Ego-R1 [45]. Right: LongVideo-R1 achieves a better tradeoff compared to recent methods on the LVBench dataset [49]. The marker size indicates model scale.

### Abstract

This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoningequipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideoR1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through

* Corresponding authors.

a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. Code and data are available at https: //github.com/qiujihao19/LongVideo-R1.

### 1. Introduction

The rapid advancement of multimodal large language models (MLLMs) has opened an unprecedented avenue for the semantic understanding of video data [24, 30]. However, the MLLMs’ success in the domain of long-form videos (those spanning 1–2 hours) is obstructed by their finite size of context, making them unable to ingest the rich visual content for comprehensive understanding. This intrinsic limitation forces current methodologies to rely on a costly, brute-force pipeline—partitioning video to short clips, pro-

cessing each clip exhaustively (e.g., generating captions or summarizing events), and finally integrating the results into the final answer. Recent studies such as Ego-R1 [45] and Videotree [53] reported competitive long video QA accuracy, but their complexity grows linearly with the video’s length, leading to prohibitively high computational cost and latency. This severely restricts the deployment of MLLMs in real-world applications, such as embodied agents requiring low-latency world reactions and high-throughput videochat services constrained by per-sample processing budgets.

In this study, we introduce a new, practically motivated research setting: long video understanding under fewer computational constraints. Instead of solely optimizing for question answering (QA) accuracy, we propose that a better measure of model efficacy lies in its ability to achieve a better accuracy-efficiency tradeoff. We formally quantify the computational burden by accumulating the estimated cost of every operation that an MLLM requires to derive an answer. In other words, the objective is to find the Paretooptimal solution where competitive accuracy is maintained with minimal computational expenditure. The key to unlocking this efficiency is replacing exhaustive search with goal-oriented reasoning. We hypothesize that an MLLM must possess the ability to perform dynamic and iterative reasoning: based on partial, high-level context, it must decide which clip to sample next to locate the critical event pertaining to the question.

Motivated by the idea, we propose LongVideo-R1, a novel framework that integrates an MLLM with a large reasoning model (LRM) for smart video navigation. The long video is organized into a hierarchical structure, enabling the LRM to rapidly shift its focus across temporal granularity levels. Given a question, LongVideo-R1 begins its exploration at the top layer and, at each step, calls a video captioning tool to gather local context, and then calls a thinking module to determine whether or not the answer can be derived. If yes, a video QA tool is called to generate the final answer; otherwise, the thinking module dictates the next sampling location—it may drill down to a child clip, traverse laterally to a sibling, or backtrack to an upper layer for renewed context. The process terminates upon reaching a maximum iteration limit.

To train LongVideo-R1, we construct a high-quality dataset of 33K reasoning episodes leveraging the grounding annotations of the CGBench dataset [3] and synthesize explicit reasoning trajectories using the GPT-5 API. We train the Qwen3-8B [55] model using supervised finetuning (SFT) followed by reinforcement learning (RL) with a novel reward mechanism designed specifically to prioritize efficient navigation and accurate grounding results. The training procedure is efficient upon pre-extracted captions and stable throughout a few training epochs.

We test LongVideo-R1 on three challenging long video

QA benchmarks, i.e., LVBench [49], VideoMME [12], and MLVU [66]. The results show that LongVideo-R1 achieves competitive QA accuracy with an average of 10.5 rounds of reasoning and navigation/answering, resulting in a significantly lower computational cost than the linear-scan methods. Furthermore, we showcase its capability for ultra-long video understanding on complex TV dramas, a domain previously inaccessible under strict budget constraints.

### 2. Related Work

Multimodal large language models (MLLMs) [7, 17, 48] represent a paradigm shift in computer vision research. Inheriting the robust reasoning capabilities of large language models (LLMs) [1, 25, 43, 47], MLLMs extend this competency to the visual domain by encoding visual inputs into discrete tokens and integrating them into the model’s textual context [26, 27, 33, 46] and have transcended conventional, bounded visual recognition tasks (e.g., classification, detection) to enable complex, open-world question answering (QA) over video data [24, 30].

As visual understanding performance approaches saturation on static images and short video clips, the community’s focus has substantially shifted toward long-form video understanding. The introduction of large-scale benchmarks featuring hour-long videos and complex QA tasks (e.g., EgoSchema [31], LongVideoBench [54], Video-MME [12], LVBench [49], CG-Bench [3], etc.) poses significant challenges to MLLMs. Two lines of research were conducted to overcome the inherent context length limitations of MLLMs. One direction focuses on devising efficient video representations [18, 21, 37] to maximize the information density [8, 38, 62]. Another direction, which is highly scalable, involves segmenting the video, processing components separately, and integrating the resulting information for final inference [40, 42, 61]. This latter approach has been further refined by the advent of large reasoning models (LRMs) [14, 17, 55], leading to the development of agent-based video understanding systems [9, 51, 53]. In these systems, an LLM agent employs explicit thinking and reasoning to strategically invoke various specialized tools, a methodology that currently dominates performance across many leading benchmarks [11, 41].

Notwithstanding the rapid progress in achieving high accuracy, relatively minimal effort has been dedicated to reducing the computational budget of long-form video understanding. For instance, recent agentic architectures, such as video-SALMONN 2 [41], and Ego-R1 [45], necessitate the exhaustive processing of all or a substantial proportion of video segments, demanding an inordinate number of MLLM calls and consequently imposing severe computational overhead. In this paper, we formally address this deficiency by defining the pursuit of the accuracy-efficiency Pareto-optimum and subsequently introducing a competi-

tive, agent-based solution for the pursuit.

Training a smart agent necessitates with advanced reinforcement learning techniques. Classical algorithms, notably Proximal Policy Optimization (PPO) [35], have been extended into Group Relative Policy Optimization (GRPO) [36] to obviate the explicit reliance on a critic model, improving the efficiency of policy optimization. Numerous subsequent iterations have been proposed to refine policy learning for both LLMs [57, 64, 65] and MLLMs [11, 16, 23, 59]. A predominant theme across these advancements is the engineering of specialized reward functions tailored to guide agent behavior toward desired outcomes.

### 3. On Efficient Long Video Understanding

#### 3.1. What Makes Efficient Video Understanding?

Given that agentic algorithms for long-form video understanding necessitate a multi-stage process (including data preparation, clip navigation, hierarchical reasoning, and final inference), we formally define the total computational cost required for a single QA task. This cost is computed by aggregating the estimated computational overhead incurred at every step within the operational pipeline. Our primary research objective is to devise an algorithmic solution that attains a Pareto-optimal tradeoff between QA accuracy and computational efficiency1.

To achieve this goal, we introduce LongVideo-R1, a dynamic, active exploration framework. As depicted in Figure 1, LongVideo-R1 operates via a self-regulating, closedloop mechanism instantiated by two core functionalities: (1) contextual exploration, which governs the active navigation and information collection within the hierarchical video structure, and (2) reasoning and termination control, which judges the sufficiency of the gathered context for QA and, if necessary, determines the subsequent step for exploration. This iterative paradigm, where the process continues until a definitive answer is produced (or maximum iterations are reached), provides a dramatic reduction in computational expenditure compared to exhaustive search algorithms, while preserving a competitive QA accuracy.

#### 3.2. LongVideo-R1 Framework

The input of LongVideo-R1 consists of a long-form video V and a question q. Let us denote the duration of V as T (in seconds); given 0 ⩽ t1 < t2 ⩽ T, V can be sliced into shorter clips, denoted by V[t1,t2].

To support exploring video clips of different lengths, we organize the video into a multi-level tree structure. The root

1We specifically assume a setting where each QA task is executed individually and on-demand. This explicitly excludes algorithms that rely upon extensive video preprocessing, as such approaches do not satisfy the low-latency requirements of reactive or budget-constrained systems.

node of the tree is the entire video, i.e., V ≡ V[0,T]. The tree has D levels (the root is the 0-th level and the leaf node is the D-th level); each non-leaf node has K children, corresponding to its video clip partitioned into K equal-length, non-overlapping sub-clips. We denote a d-th-level clip as Vk

1,...,kd, where kd′ ∈ {0,...,K − 1} indicates the child index at the d′-th level. Unless otherwise specified, we assume that D = 3 and K = round(

T/16s) so that the video clip at the leaf level is approximately 16-second long. This hierarchical structure allows the agent to check long video clips first and, when necessary, ‘zoom in’ to find an answer in finer-scale visual content. While the uniform partition is easy to implement, we understand that it is not the optimal choice, e.g., it would cause semantically similar content to fall into neighboring sub-clips, increasing the ambiguity of localization.

D

LongVideo-R1 is a large reasoning model (LRM) and follows a chain-of-thought-with-tool (CoTwT) framework, where two multimodal tools are incorporated:

- • The video captioning tool, video cap(). It receives a

video clip Vk

1,...,kd with the number of sampled frames F, and outputs the text description t of the clip.

- • The video QA tool, video qa(). It receives a video clip

1,...,kd with the number of sampled frames F, the question q, and outputs the answer a (it is possible to answer ‘I don’t know’). This tool is allowed only on the lowestlevel clips.

Vk

There is a major difference between these two tools: video cap() aims to offer generic video descriptions that assist the subsequent steps for key content localization, while video qa(), often called at the last step, focuses on answering the specific question. For simplicity, we assume that both tools sample frames time-uniformly from video data, and vanilla visual encoding (i.e., no compression) is performed on the frames.

#### 3.3. Chain-of-Thought-with-Tool Procedure

Based on the above preparation, we formulate the inference of LongVideo-R1 into a chain-of-thought-with-tool (CoTwT) procedure, which is widely used in multi-round tool-use tasks, such as Ego-R1 [45]. A complete inference episode is written as a chain:

E = [S1,S2,...,SL], (1) where each Sl indicates a step:

Sl =

(rl,tl), if l < L, (rl,a) if l = L,

(2)

where rl is the reasoning statement at the l-th step, at the end of which contains information indicating which tool is to be called, and tl and a denote the text description and answer, corresponding to the outputs of video cap()

Algorithm 1 Hierarchical Video Reasoning Require: Video V, question q, reasoning model rea(),

multimodal tools video cap() and video qa() Ensure: Answer a

- 1: Tree depth and width: D = 3, K = round(

D

T/16s)

- 2: Get top-level caption t0 = video cap(V)

- 3: Initialize chat history: E = [t0]
- 4: Get first reasoning output: r1 = rea(E,q)
- 5: Initialize episode length: L = 1
- 6: while rL does not contain the answer do
- 7: Parse: tool ∈ {video cap,video qa},Vk

1,...,kd

- 8: Call the tool: tL = tool(Vk

1,...,kd)

- 9: Update chat history: E ← E + [rL,tL]
- 10: Update reasoning output rL+1 = rea(E,q)
- 11: Update episode length: L ← L + 1
- 12: end while
- 13: return Answer extracted from rL

and video qa(), respectively. Note that the entire episode contains purely natural language (the multimodal tools are called as external functions), making it easier to (1) adapt to the recent advances of LRMs, and (2) explicitly connect thinking with tool-using towards a transparent inference procedure. The procedure is illustrated in Algorithm 1.

### 4. Data Curation

#### 4.1. Data Preparation

We curate a dataset for training LongVideo-R1. We choose the CG-Bench dataset [3] because it contains clue-grounded QA pairs, i.e., we can supervise the model to localize the key sub-clip(s) before answering the question.

CG-Bench contains 1.2K long-form videos, each of which is paired with a diverse set of QA pairs. We chose 800 videos and the corresponding 5.6K QA pairs for generating CoTwT trajectories.

For each video of CG-Bench, V, we first use the Qwen2.5-VL-72B model [2] as the function video cap() to extract its text description. The sampled frames F is set to be 256,128,64,32, and the suggest length of description (in English words) is 400,400,400,200 for level index d = 0,1,2,3, respectively. To guide the LRM to locate sub-clips, we modify the prompt (see Appendix B.2) to insert absolute timestamps into the caption.

#### 4.2. Generating CoTwT Trajectories

The pipeline of generating CoTwT trajectories is illustrated in Figure 2. We guide GPT-5 [32] to perform the CoTwT procedure, which starts with the top-level video clip, and continues when the model is confident to produce the final answer. In the prompt to GPT-5 (see Appendix B.2), we indicate the functionalities of video cap() and video qa()

|[Figure 40]|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

In the video, what color are the headphones that the worker in the blue hat is wearing while producing sausages?

<think>…segment 4 (1755.9–2341.0s) directly mentions workers operating machinery and manually filling sausage

###### thinking tool call

casings — a clear instance of “producing …<think>

<tool> video_cap(4,1) </tool>

<think> …shows workers in protective clothing during sausage production. It doesn’t provide details …<think> <tool> video_qa((4,1,3), What color are the headphones

thinking

tool call

that the worker in the blue hat is wearing?) </tool>

SFT-Data Iterative Correction

|×𝑛|
|---|

Hint: segment that contains information relevant to this question is segment 1.

thinking

answer

Verification

Figure 2. An illustration of generating CoTwT trajectories from clue-grounded video QA data. Drawing style was inspired by EgoR1 [45]

.

and specify the rules of using them (e.g., video cap() can be called at a node if its parent has been traversed, and video qa() can only be called in the lowest level nodes).

GPT-5 performs the above task as zero-shot inference; in many (about 30% of) scenarios, it can produce incorrect answers or fail to pass the above verification. We make two fixes to improve the data quality and guarantee success.

- • Instead of starting with the root node, V, we ask the model to traverse all K sub-clips at the first level. This alleviates the risk that GPT delves deep into local parts without obtaining sufficient global information, and improves stability in particular when exploring hour-long videos.
- • When GPT fails, we use the clue-grounded hints of CGBench to guide it towards the correct answer. Meanwhile, we try to keep the hints to a minimal amount: when GPT fails for the first time, we add the highest-level segment containing the relevant event to the prompt; if it still fails, a deeper-level hint with a more precise segment and event description is added. This process continues until the model produces a correct answer. A comparative example of the original and clue-guided prompts is provided in Appendix B.2. This strategy guarantees the correctness of each CoTWT trajectory while leaking as few hints as possible. Trained on such data, the LRM learns to generalize toward efficient exploration rather than simply memorizing video content and answers.

As a result, we obtain 5.6K CoTwT trajectories with an average of 5.8 steps, yielding approximately 33K highquality samples for supervised fine-tuning. Upon releasing

the SFT data to the community, we show that such data is helpful to train a powerful agent, and the agent’s performance is positively related to the amount of SFT data (i.e., it is important to hint GPT when it goes wrong). We have also revealed a promising path to enhance the agent, for which one only needs to establish more clue-grounded video QA pairs and generate more CoTwT trajectories.

### 5. Training LongVideo-R1 Agent

We follow a well-established, two-stage pipeline to train the LongVideo-R1 agent, i.e., a supervised fine-tuning (SFT) stage as a cold start and a reinforcement learning (RL) stage for further optimization.

#### 5.1. Supervised Fine-tuning

In the first stage, we fine-tune a pretrained large language model on the curated CoTwT data. This cold-start phase equips the model with the ability to generate structured reasoning trajectories under the desired format.

Each training sample simulates a realistic multi-round tool-using process that ultimately leads to the correct answer. Specifically, the reasoning process is enclosed within special tokens ⟨think⟩...⟨/think⟩, followed by either a tool invocation or an answer. Tool calls are enclosed within ⟨tool⟩...⟨/tool⟩ and answers within ⟨answer⟩...⟨/answer⟩. During training, the tool invocation content is parsed and executed to obtain corresponding observations, which are then fed back to the model as new contextual information.

This structured annotation enables the model to learn (1) when to continue reasoning, (2) which tool to invoke, and (3) when to terminate reasoning and produce the final answer. After SFT, the model (denoted as LongVideo-R1SFT) is capable of generating correctly formatted reasoning sequences and performing coherent tool interactions, which serve as a solid foundation for reinforcement learning.

#### 5.2. Reinforcement Learning with GRPO

After the SFT stage, we regard the video reasoning process as an interactive exploration environment: the model acts as an agent, video tools (video cap() and video qa()) form the action space, and the hierarchical video serves as the environment state. This formulation naturally lends itself to optimization via reinforcement learning.

We employ the GRPO algorithm [36] with the version introduced in Ego-R1 [45] to further optimize the policy model πθ, aiming to improve reasoning efficiency and ac-

curacy. The objective is defined as:

|Siy|

G

T

1 G

1 |Siy|

JGRPO(θ) = Eq∼P(Q),{oG

i }Gi=1∼πθold(O|q)

y=1

t=1

i=1

πθ(Si,t|q, Si,<t) πθold(Si,t|q, Si,<t)

Aˆyi,t,

min

πθ(Si,t|q, Si,<t) πθold(Si,t|q, Si,<t)

, 1 − ε, 1 + ε A ˆyi,t − β DKL[πθ ∥ π0] ,

clip

(3)

Q denotes a question sampled from the data distribution D, oi represents the model output, G is the number of rollouts, and T denotes the number of reasoning rounds. The model is parameterized as πθ, where πθ and πθ

denote the current and reference policies, respectively, and π0 represents the policy inherited from the LongVideo-R1-SFT model, used for KL regularization. The advantage term is computed by:

old

riGRPO − mean {rjGRPO} std {rjGRPO}

. (4)

Ai =

#### 5.3. Reward Design

To facilitate the model to explore video content efficiently (while finding the correct answer), we design a composite reward function:

R = wans · rans + wloc · rloc + wrepeat · rrepeat, (5)

where w· are reward weights and the three components are defined as follows:

- • The answer reward, rans ∈ {0,1}, gives a reward of 1 if the final answer matches the ground-truth, otherwise 0.
- • The location reward, rloc, encourages the model to identify the correct segment efficiently:

rloc = 2 ·

cov × pre cov + pre

,

where the coverage and precision are defined as

cov = |Imodel ∩ Igt| |Igt|

, pre = |Imodel ∩ Igt| |Imodel|

,

where Igt and Imodel indicates the ground-truth and predicted sets of time intervals. Imodel is the union of all non-overlapping time segments corresponding to the nodes requested by the model. This F1-like metric encourages high coverage of relevant content while penalizing unnecessary exploration.

- • The repeat penalty, rrepeat, discourages repeatedly visiting the same segments, reducing wasted computation.

#### 5.4. Rollout and Optimization

During RL training, the agent interacts with executable video tools to generate rollout trajectories. Each rollout

continues until the model outputs a final answer or reaches a predefined maximum number of reasoning steps. The collected trajectories are then used to compute policy gradients and update πθ using GRPO.

After RL training, the resulting model, LongVideo-R1, is capable of performing multi-tool reasoning efficiently on long video tasks. It learns to minimize redundant exploration while maintaining high answer accuracy, achieving a superior tradeoff between performance and computational efficiency compared to conventional MLLMs.

### 6. Experiments

#### 6.1. Implementation Details

We train LongVideo-R1 upon a Qwen3-8B model [55]. The multimodal tools, video cap() and video qa(), are chosen to be Qwen2.5VL-72B and Qwen2.5VL-32B [2], respectively. Compared to other agentic approaches [45, 51, 53, 56], that relied on proprietary LLMs such as GPT or Gemini, our setting eases local deployment and fair comparison. We perform SFT for 3 epochs followed by RL for 2 epochs.

LongVideo-R1 is tested on three popular long-form video understanding benchmarks. LVBench [49] contains 103 videos (average duration: 4038 seconds) and 1,549 QA pairs. Video-MME-long [12] contains 300 videos (average duration: 41 minutes) with 3 QA pairs for each video. MLVU [66] contains 1,337 videos with their durations ranging between 3 minutes to 2 hours. All these benchmarks have provided multiple choices for each question; we prompt these choices with questions to LongVideoR1 and ask it to produce the choice index in the answer box. An answer is considered correct if the choice(s) perfectly match the ground-truth.

#### 6.2. Results and Analysis

Results on LVBench. We compare LongVideo-R1 with state-of-the-art models on LVBench in Table 1. The compared methods are categorized into three groups: proprietary models, leading open-sourced MLLMs, and agentbased systems. As illustrated in Table 1, LongVideoR1 achieves a 50.0% accuracy, outperforming the other agent-based methods by at least 5.6%. Besides, with an 8B-LLM, LongVideo-R1 surpasses most proprietary and open-sourced MLLMs; for example, it exceeds GPT4o by 1.1% and GLM-4V-plus by 1.3%. Notably, LongVideo-R1 demonstrates outstanding results on two sub-categories, KIR (Key Information Retrieval) and TG (Temporal Grounding) tasks. In particular, its performance on TG reaches 56.4%, surpassing all other models by a significant margin of 10.9%. These results highlight the strong ability of LongVideo-R1 in accurately locating key temporal segments within long videos. Moreover, the abil-

ity of LongVideo-R1 grows with the multimodal tools: as shown in Table 1, when we use Qwen3-VL-32B-Instruct, a stronger MLLM, for video captioning, the overall accuracy improves significantly, meanwhile the advantages in the KIR and TG sub-categories persist.

Results on MLVU and Video-MME. The comparisons are shown in Tables 2 and 3, respectively. While LongVideoR1 also performs well, it does not excel among the opensourced MLLMs. The reason lies in the property of the benches: MLVU contains many short videos, and VideoMME contains many global questions like ‘What is the main idea of the video?’, which is beneficial for the uniform or adaptive (e.g., [42]) frame sampling methods. LongVideo-R1’s advantage also reflects in inference time. We compare LongVideo-R1 with Ego-R1 [45], which reports a similar accuracy on Video-MME. Differently, EgoR1 requires video captioning every 30 seconds, resulting in an average of 86 caption segments on VideoMME, while LongVideo-R1 only undergoes an average of 10.5 rounds, claiming a much lower computational cost. The model’s performance on MLVU and Video-MME also benefits from improved multimodal tools, e.g., Qwen3-VL-32B-Instruct for video captioning.

Ablative studies. We ablate the performance with respect to the SFT and RL strategies on the LVBench dataset. As shown in Table 4, the model fine-tuned with the full 33K SFT samples outperforms the one trained with a subset of 10K samples, both after the SFT and subsequent RL stages. This demonstrates the importance of increasing the size of SFT data. Another important factor is the location reward, rloc. As shown in Table 5, adding rloc leads to significant performance gains in the overall set and the KIR (Key Information Retrieval) and TG (Temporal Grounding) subsets. These results indicate that rloc effectively enhances the model’s ability of video navigation and such ability contributes to long video QA.

Accuracy-efficiency tradeoff. Figure 1 shows that LongVideo-R1 achieves a favorable tradeoff; it achieves a 50.0% accuracy on LVBench, requiring 3 minutes per QA. The cost can be reduced to 2 minutes per QA at a mere 0.2% accuracy drop. More results are shown in Tables 6 and 7, where we change the MLLM scale for video cap() and alter the maximum number of tool uses, respectively. These results suggest an interesting solution to further improve the tradeoff, i.e., switching the setting of tool use (in both size and number) to answer questions of different difficulties.

Case studies. We further conduct a case study to illustrate the reasoning and planning capability of LongVideoR1. The video in Figure 4 is 102 minutes long and features around 20 performers. The question requires the model to identify a specific performer and then accurately count the number of dogs associated with that performer. Without prior information about the performer, LongVideo-R1 first

Method ER EU KIR TG Rea Sum Overall Proprietary Models

Gemini 1.5 Pro [44] 32.1 30.9 39.3 31.8 27.0 32.8 33.1 GLM-4V-plus-0111 [15] 46.2 47.8 54.1 42.7 46.5 37.9 48.7 GPT-4o-20241120 [17] 48.9 49.5 48.1 40.9 50.3 50.0 48.9

Leading Open-sourced MLLMs

TimeMarker-8B [5] 42.8 39.1 34.9 38.7 38.2 48.8 41.3 VideoLLaMA3-7B [60] 45.8 42.4 47.8 35.9 45.8 36.2 45.3 InternVL2.5-78B [6] 43.8 42.0 42.1 36.8 51.0 37.9 43.6

- Qwen2-VL-72B [2] 38.0 41.1 38.3 41.4 46.5 46.6 41.3 ReTake-7B [50] 49.8 46.2 52.9 45.0 45.8 27.6 47.8 VideoChat-Flash-7B [23] 51.1 46.0 49.0 38.9 48.5 34.5 48.2 AdaReTake-72B [52] 53.0 50.7 62.2 45.5 54.7 37.9 53.3

Agent-based Systems

VideoAgent [51] 28.0 30.3 28.0 29.3 28.0 36.4 29.3 VideoTree [53] 30.3 25.1 26.5 27.7 31.9 25.5 28.8 MemVid [58] 53.4 40.6 37.8 43.9 43.2 28.1 44.4 VCA [56] 43.7 40.8 37.8 38.0 46.2 27.3 41.3

LongVideo-R1 49.2 48.4 56.4 56.4 44.3 43.1 50.0 LongVideo-R1 (new†) 60.9 57.8 70.1 62.7 50.2 55.2 60.7

Table 1. QA accuracy (%) on all sub-tasks of LVBench [49]. † We trained an updated version using video captions generated by

- Qwen3-VL-32B-Instruct and renewed SFT data.

Method Acc. Proprietary Models

GPT-4V [17] 49.2 GPT-4o [17] 64.6

Open-Sourced MLLMs

Video-CCAM-14B [10] 63.1 Video-XL-7B [39] 64.9 LLaVA-OV-72B [19] 66.4 LinVT-7B [13] 68.9 Aria-25B [20] 70.6 Oryx-1.5-32B [29] 72.3 VideoLLaMA3-7B [60] 73.0 VideoChat-Flash-7B [22] 74.7

Agent-based Systems

VideoTree [53] 60.4 VideoMind-7B [28] 64.4

LongVideo-R1 68.1 LongVideo-R1 (new†) 71.3

Method w/o w/ Proprietary Models

Gemini 1.5 Flash [44] 61.1 68.8 GPT-4o-20240513 [17] 65.3 72.1 Gemini 1.5 Pro [44] 67.4 77.4

Leading Open-sourced MLLMs

Qwen2-VL-72B [48] 62.2 74.3 InternVL2.5-72B [6] 62.6 64.8 LLaVA-OV-72B [19] 60.0 62.4 LLaVA-Video-72B [63] 61.5 72.5 VideoChat-Flash-7B [22] 55.6 63.3 VideoLLaMA3-7B [60] 54.9 61.0 LiveCC-7B [4] 53.7 64.1

Agent-based Systems VideoAgent [51] 46.4 – VideoTree [53] 54.2 – MemVid [58] 55.0 – Video-RAG-7B [34] – 59.8 Ego-R1 [45] – 64.9 VCA [56] 56.3 –

LongVideo-R1 55.8 64.4 LongVideo-R1 (new†) 58.0 68.6

Table 2. Model-level QA accuracy (%) on the MLVU dataset [66]. † We trained an updated version using the same setting as in Table 1.

Table 3. QA accuracy (%) on the ‘long’ subset of Video-MME [12] without (w/o) or with (w/) subtitles. † We trained an updated version using the same setting as in Table 1.

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Are there any wooden carts visible when Dong Weihong calls out 'Dongmei sister’ ? A 15-episode TV series with a duration of over 10 hours <think>…The high-level captions mention Dong Weihong in High-level segment 6, where she appears near a rustic wooden structure. However, it doesn’t specify the scene where she calls out Dongmei sister or the presence of wooden carts. To know the detail …</think>

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

<tool>video_cap(6,2)</tool>

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

<think>…caption mentions an outdoor scene with Dong Weihong and Dongmei in a snowy

rural area, which likely includes the moment when Dong Weihong calls out Dongmei sister.

However, it doesn’t mention wooden carts specifically. To confirm …</think> <tool>video_cap(6,2,4)</tool>

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- <think>…caption (6,2,11) mentions the outdoor snowy scene, but doesn’t describe the moment when Dong Weihong calls out Dongmei sister or the wooden carts. Segment (6,2,12) might contain the specific dialogue and details of the setting. I’ll check…</think> <tool>video_cap(6,2,11)</tool>

- <think>…caption (6,2,12) explicitly describes the moment when Dong Weihong calls out Dongmei sister and mentions a wooden cart visible in the foreground. To confirm whether there are indeed wooden carts around, I can query the video segment to verify it.</think> <tool>video_qa((6,2,12), Are there any wooden carts visible around ?)</tool> <answer>There is a wooden cart when Dong Weihong calls out 'Dongmei sister.’<answer>

[Figure 79]

[Figure 80]

Figure 3. LongVideo-R1 can navigate in ultra-long videos efficiently. We show an example in a long-form TV drama, A Lifelong Journey.

locates the segment where the target performer appears, after which it explores fine-grained sub-segments to pinpoint the exact moment of the performance. Finally, it invokes the video QA tool to obtain the precise answer. This example demonstrates the strong reasoning, planning, and temporal localization abilities of LongVideo-R1 for efficient long-

video understanding.

However, LongVideo-R1 can sometimes be distracted by other segments that are semantically related to the question. As shown in the Appendix D, the model may get stuck in an irrelevant segment instead of shifting its focus to the correct one. In contrast, humans can easily recognize such errors

|[Figure 81]|
|---|

|[Figure 82]|
|---|

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

ADRIAN STOICA OLGA AMBER Diana Vedyashinka AARON TRIP HAZARD

Question：How many dogs does Diana Vedyashinka perform with?

###### A 102 minutes video with about 20 performers

<think>The high-level captions provided do not mention Diana Vedyashinka, so I need to find the segment where Diana Vedyashinka appears … However, High-level 2 doesn\'t specify her name. I should get more detailed captions for the relevant part… <think> <tool> VideoCap(2,1) </tool>

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

<think>The medium-level caption confirms Diana Vedyashinka's appearance but does not state the number of dogs she performs with. To answer the question, I need the exact

number, which is likely mentioned in a low-level caption describing …<think>

<tool> VideoCap(2,1,2) </tool>

<think>The low-level caption (2,1,5) does not mention the number of dogs. Since captions aren't providing the number and video_qa also couldn't determine it precisely, I should try another low-level caption that might visually or narratively describe …<think> <tool> VideoQA((2,1,3), How many dogs is Diana Vedyashinka performing with? )</tool> <answer>Diana Vedyashinka is performing with five dogs. <answer>

[Figure 107]

[Figure 108]

Figure 4. An example of how LongVideo-R1 smartly navigates to the critical segment and answers the question.

##### Method KIR TG LVBench V-MME/L

SFT10K 46.4 39.1 39.1 57.7 SFTfull 48.5 50.9 41.6 59.2 Ours10K w/ RL 54.0 55.0 47.4 60.2 Oursfull w/ RL 56.4 56.4 50.0 64.4

Table 4. QA accuracy (%) with respect to SFT data size.

##### Method KIR TG LVBench V-MME/L

SFT 48.5 50.9 41.6 59.2 w/o rloc 49.1 53.2 45.8 61.4 Ours 56.4 56.4 50.0 64.4

- Table 5. QA accuracy (%) with or without the location reward. Model KIR TG LVBench V-MME/L Time

Qwen2.5-VL-3B 48.8 53.2 44.5 56.0 50.5 Qwen2.5-VL-7B 51.5 52.7 46.0 57.8 57.4 Qwen2.5-VL-32B 58.4 54.5 49.8 63.9 135.0 Qwen2.5-VL-72B 56.4 56.4 50.0 64.4 175.7

- Table 6. QA accuracy (%) with respect to MLLM scales. The last column shows average inference time (on LVBench, in seconds).

Max Rounds KIR TG LVBench V-MME/L Time (s)

10 49.1 52.3 43.0 57.1 103.6 20 49.5 58.2 47.3 63.2 170.8 25 54.6 54.1 48.5 63.6 172.8 30 56.4 56.4 50.0 64.4 175.7

- Table 7. QA accuracy (%) with respect to maximum rounds of tool use (mainly the video cap() function).

and redirect attention to the appropriate segment. Interestingly, we find that providing simple textual hints can effectively guide LongVideo-R1 back to the correct segment, enabling it to produce the right answer.

#### 6.3. Extension to Ultra-long Videos

Beyond existing benchmarks, LongVideo-R1 also excels in ultra-long video QA. As illustrated in Figure 3, LongVideoR1 smartly navigates to the accurate location (and gets the correct answer) with 10–20 rounds, even when the input video is tens of hours long. In comparison, open-sourced MLLMs (even sampling 256 frames) can barely find efficient information for QA, and other agent-based systems like Ego-R1 [45] and VideoTree [53] require the number of samples to grow linearly with video duration, leading to prohibitively high computational costs.

#### 6.4. Future Directions

Our work, a preliminary study towards low-cost long video understanding, reveals a few research directions for the future.

- • Extended tools. LongVideo-R1 only considered two tools (besides reasoning), video cap() and video qa(). In the future, one may introduce more tools (e.g., video instance recognition, video clip segmentation) to further improve the model’s ability. In such scenarios, an extra reward term shall be added to penalize the aggregated computational cost of tool use.

- • Advanced settings. We assumed that each video QA is processed individually. In practice, if one video corresponds to multiple QA pairs, the best model choice may

vary, e.g., the model can spend more time on key information indexing because the overhead can be amortized among all pairs. There can also emerge related settings, like incremental QA, that require the model to reuse the information efficiently.

• Enhanced video descriptions. LongVideo-R1 was built upon an LRM whose performance heavily relies on quality video captions. It hence emerges a new topic – enhancing the video description tools for more accurate and efficient reasoning and navigation. We look forward to the agent and tools being optimized simultaneously in a unified framework.

### 7. Conclusion

This paper presents LongVideo-R1, an agentic framework designed for efficient long video understanding. LongVideo-R1 explores long videos like humans: starting with top-level video sections, it maintains video descriptions and performs reasoning to judge whether the question can be answered and which part of video is to be navigated next. LongVideo-R1 was fine-tuned upon a pre-trained LRM via SFT (on a curated dataset) and RL, and we show that richer SFT data can help. LongVideo-R1 achieves competitive QA accuracy on several long video benchmarks and is skilled at information retrieval and grounding tasks; more importantly, it shows a favorable accuracy-efficiency tradeoff over other agentic algorithms. We hope LongVideo-R1 enlightens new directions for long video understanding.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 6, 7
- [3] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cgbench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075,

2024. 2, 4

- [4] Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. Livecc: Learning video llm with streaming speech transcription at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29083–29095, 2025. 7
- [5] Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211, 2024. 7
- [6] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,

- Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 7
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 2
- [8] Rohan Choudhury, Guanglei Zhu, Sihan Liu, Koichiro Niinuma, Kris Kitani, and L´aszl´o Jeni. Don’t look twice: Faster video transformers with run-length tokenization. Advances in Neural Information Processing Systems, 37:28127–28149,

2024. 2

- [9] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92, 2024. 2
- [10] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023, 2024. 7
- [11] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 2, 3

- [12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Computer Vision and Pattern Recognition, pages 24108–24118, 2025. 2, 6, 7
- [13] Lishuai Gao, Yujie Zhong, Yingsen Zeng, Haoxian Tan, Dengjie Li, and Zheng Zhao. Linvt: Empower your imagelevel large language model to understand videos. arXiv preprint arXiv:2412.05185, 2024. 7
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2
- [15] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 7
- [16] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 3

- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 7
- [18] Seon-Ho Lee, Jue Wang, Zhikang Zhang, David Fan, and

- Xinyu Li. Video token merging for long-form video understanding. arXiv preprint arXiv:2410.23782, 2024. 2
- [19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 7
- [20] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al. Aria: An open multimodal native mixtureof-experts model. arXiv preprint arXiv:2410.05993, 2024. 7
- [21] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7486–7495, 2024. 2
- [22] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint

- arXiv:2501.00574, 2024. 7

[23] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint

- arXiv:2501.00574, 2025. 3, 7

- [24] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984, 2024. 1, 2
- [25] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 2
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 2
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2
- [28] Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. Videomind: A chain-of-lora agent for long video reasoning. arXiv preprint arXiv:2503.13444,

2025. 7

- [29] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 7
- [30] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Annual Meeting of the Association for Computational Linguistics, pages 12585–12602, 2024. 1, 2
- [31] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 2
- [32] OpenAI. Gpt-5 system card. Technical report, OpenAI,

2025. Accessed: 2025-11-13. 4

- [33] Jihao Qiu, Yuan Zhang, Xi Tang, Lingxi Xie, Tianren Ma, Pengyu Yan, David Doermann, Qixiang Ye, and Yunjie Tian. Artemis: Towards referential understanding in complex videos. Advances in Neural Information Processing Systems, 37:114321–114347, 2024. 2
- [34] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv preprint arXiv:2502.01549, 2025. 7
- [35] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [36] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3, 5
- [37] Leqi Shen, Tianxiang Hao, Tao He, Sicheng Zhao, Yifeng Zhang, Pengzhang Liu, Yongjun Bao, and Guiguang Ding. Tempme: Video temporal token merging for efficient textvideo retrieval. arXiv preprint arXiv:2409.01156, 2024. 2
- [38] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 2
- [39] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26160–26169,

2025. 7

- [40] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2
- [41] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, and Chao Zhang. videosalmonn 2: Captioning-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025. 2
- [42] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29118–29128, 2025. 2, 6
- [43] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2
- [44] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 7
- [45] Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang,

- Hongyuan Zhu, and Ziwei Liu. Ego-r1: Chain-of-toolthought for ultra-long egocentric video reasoning. arXiv preprint arXiv:2506.13654, 2025. 1, 2, 3, 4, 5, 6, 7, 8
- [46] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv:2401.13307, 2024. 2
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 7
- [49] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In International Conference on Computer Vision, pages 22958–22967, 2025. 1, 2, 6, 7
- [50] Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, and Liqiang Nie. Retake: Reducing temporal and knowledge redundancy for long video understanding. arXiv preprint arXiv:2412.20504, 2024. 7
- [51] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena YeungLevy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76, 2024. 2, 6, 7
- [52] Xiao Wang, Qingyi Si, Shiyu Zhu, Jianlong Wu, Li Cao, and Liqiang Nie. Adaretake: Adaptive redundancy reduction to perceive longer for video-language understanding. In Findings of the Association for Computational Linguistics: ACL 2025, pages 5417–5432, 2025. 7
- [53] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In Computer Vision and Pattern Recognition, pages 3272–3283, 2025. 2, 6, 7, 8
- [54] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 2
- [55] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2, 6
- [56] Zeyuan Yang, Delin Chen, Xueyang Yu, Maohao Shen, and Chuang Gan. Vca: Video curious agent for long video understanding. In International Conference on Computer Vision, pages 20168–20179, 2025. 6, 7
- [57] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 3

- [58] Huaying Yuan, Zheng Liu, Minghao Qin, Hongjin Qian, Yan Shu, Zhicheng Dou, Ji-Rong Wen, and Nicu Sebe. Memoryenhanced retrieval augmentation for long video understanding. arXiv preprint arXiv:2503.09149, 2025. 7
- [59] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025. 3
- [60] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 7
- [61] Ce Zhang, Yan-Bo Lin, Ziyang Wang, Mohit Bansal, and Gedas Bertasius. Silvr: A simple language-based video reasoning framework. arXiv preprint arXiv:2505.24869, 2025. 2
- [62] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 2
- [63] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 7
- [64] Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, et al. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025. 3
- [65] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 3
- [66] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In Computer Vision and Pattern Recognition, pages 13691–13701, 2025. 2, 6, 7

The supplementary document provides (1) details of our hierarchical video definition in A;(2)comprehensive implementation details including the prompts we used for data generation and the experiment setup in B;(3)more qualitative examples and ultra long video examples in C;(4)failure examples and analysis in D.

### A. Hierarchical Video Definition

To enable efficient localization of relevant segments within long videos without requiring any preprocessing, we represent each video using a hierarchical tree structure. Given a tree depth D and width W, the root node corresponds to the entire video. The root is evenly divided into W child segments, each of which is recursively divided into another W sub-segments. Repeating this process yields a hierarchical video tree with D levels.

In practice, we set D = 3, and choose W adaptively according to video length. The leaf-level segment length is fixed at 16 seconds. Let Duration denote the total video length, then the number of leaf segments is Duration/16. We determine the width as:

W =

Duration 16

1/D

,

which typically lies within 4 to 8 across all datasets.

To ensure that deeper nodes receive fine-grained visual signals, we adjust both frame sampling rate and spatial resolution according to the hierarchy. For the caption model, we use {256, 128, 64, 32} frames from level 0 to level 3, respectively. The corresponding image resolutions are set to

512 2

512 √2

512 2√2

,

,

, 512,

ensuring that the overall number of visual tokens remains approximately constant across levels. This design guarantees consistent computation cost per caption call while enabling finer temporal and spatial detail as the model traverses downward in the hierarchy.

### B. Implementation Details B.1. Environment Setup

During both data generation and model evaluation, we use Qwen-2.5-VL-72B as our video cap model and Qwen-2.5VL-32B as our video qa model. These two components can be replaced by other models; for instance, Qwen-2.5-VL32B is also capable of serving as an effective captioning model.

For CoTWT data generation, we employ GPT-5 as the central reasoning model. All SFT and RL training is conducted on a cluster with 8 NVIDIA H800 GPUs (80GB). Mixed-precision training and FSDP sharding are used to maximize training throughput.

##### Configuration SFT RL

Model init Qwen3-8B LongVideo-R1-SFT LLM sequence length 32768 32768 Learning rate 1 × 10−5 1 × 10−6 Learning rate schedule cosine decay constant Training epochs 3 2 Global batch size 32 12 Training steps 384 696 Rollout numbers - 16

Table 8. Training hyper-parameters.

#### B.2. Data Generation

We use a proprietary GPT-5 model to generate the CoTWT supervision signals. Since CGBench provides timestamp annotations for each question, we use CGBench as the primary source for CoTWT construction. CGBench contains approximately 1200 videos and 12,000 QA pairs. We use 800 videos and around 8000 QA pairs to construct the SFT data; after filtering, we obtain 5600 high-quality CoTWT trajectories.

The remaining 400 videos and approximately 4200 QA pairs are reserved for RL training.

The prompts used for data generation and caption extraction are listed in Table 9 and Table 10 of the appendix. Initially, we provided only the root-level caption as GPT5’s starting context, but this resulted in unstable behavior and low accuracy (around 30%). We found that providing the W child captions from the highest-level nodes as initial information substantially improves stability and accuracy.

For all datasets (CGBench, LVBench, VideoMMELong), the tree width W is set between 4 and 8. For EgoSchema, due to its large number of short 2-minute videos, we set W between 3 and 8.

#### B.3. Training hyper-parameters

We adopt Qwen3-8B as the central reasoning model for both SFT and RL training. The model receives the W highest-level captions as its initial observation and interacts with the video tree through a sequence of tool calls.

Training consists of two phases:

SFT. We train the model to imitate CoTWT trajectories by predicting reasoning process, search actions and answers. This stage helps the model acquire hierarchical search behavior and structured video reasoning skills.

RL. During RL, we pre-extract all hierarchical captions to accelerate training, while the video qa tool is invoked in real time. Qwen-2.5-VL-32B is deployed on two GPUs to

Table 9. System prompt for data generation.

[BEGIN OF GOAL] You are a reasoning assistant designed to answer questions about a long video through hierarchical captions. The video is organized into three levels of temporal granularity:

- 1. High-level: The video is divided into width major segments.
- 2. Medium-level: Each High-level segment is further divided into width sub-segments.
- 3. Low-level: Each Medium-level segment is further divided into width finer sub-segments. You will be asked a question about the video. At the beginning, you are given **only the High-level captions**. Your goal is to answer the question as accurately as possible. [END OF GOAL] [BEGIN OF REASONING AND TOOL USAGE INSTRUCTIONS]

- 1. Reason first: Before taking any action, carefully analyze whether the current information (captions you already have) is sufficient to answer the question.
- 2. If sufficient: Directly provide your final answer inside ⟨answer⟩⟨/answer⟩ tags.
- 3. If insufficient: Identify which part(s) of the video might contain the needed information. Then use one of the following tools:

- - To obtain finer captions: ⟨tool⟩get caption((high segment id, medium segment id, low segment id))⟨/tool⟩

- - Each of the three IDs is an integer from 1 to width.
- - To request a Medium-level caption, provide (high segment id, medium segment id) only.

- - To request a Low-level caption, provide the full triplet (high segment id, medium segment id, low segment id).

- - To query visual information from the actual video segment: ⟨tool⟩video qa((high segment id, medium segment id, low segment id), query)⟨/tool⟩

- - This tool sends the corresponding Low-level video segment to a specialized video QA module.
- - The query should specify what exact information you need (e.g., “what color is the person’s shirt?”, “what object is on the table?”).
- - You may only use video qa after you have already retrieved the corresponding Low-level caption for that segment.

- 4. Restriction: In each reasoning round, you may only call one tool (either ‘get caption‘ or ‘video qa‘) once to obtain new information. [END OF REASONING AND TOOL USAGE INSTRUCTIONS]

##### [BEGIN OF FORMAT INSTRUCTIONS]

Your reasoning and actions must follow this structure exactly: ⟨think⟩Your internal reasoning process here. Analyze what information you have, what is missing, and which part might be relevant.⟨/think⟩ ⟨tool⟩(get caption or video qa call here, if needed)⟨/tool⟩ or ⟨think⟩...⟨/think⟩ ⟨answer⟩Your final answer here (only when you are confident the information is sufficient).⟨/answer⟩

##### [END OF FORMAT INSTRUCTIONS]

serve as the video qa module, while the remaining six GPUs are dedicated to RL training.

The detailed hyper-parameters are listed in Table 8.

#### B.4. Time consumption calculation

The timing was tested on an A800. The total inference time of LongVideo-R1 consists of three components: (1) the forward-pass time of the reasoning model T1, (2) the

captioning time required for processing a video segment T2, and (3) the time required for the video qa model to answer a query T3. Let the average number of calls to LongVideoR1, the video cap model, and the video qa model be C1, C2, and C3, respectively. Then the expected time cost for answering one question is:

###### T = C1T1 + C2T2 + C3T3.

Using VideoMME-Long as an example, the model re-

Table 10. Video Caption Model Prompt.

You are a video understanding expert. Please create a detailed description with timestamp information for the current video clip (which contains multiple frames arranged in chronological order). You are given num frame uniformly sampled frames from the video, along with the timestamp (in seconds) of each frame in the entire video. Description Guidelines:

- -Dialogue Description Guidelines:

- 1)In addition to video frames, subtitle information for this video segment is also provided.
- 2)The output description must faithfully include the given subtitle content. Do not invent or add dialogues that are not provided. Avoid redundant repetition, maintain the original order of the lines, and ensure the sentences flow smoothly.
- 3)Your output should be around num words words.

- -Whenever reasonable, include the provided timestamps in your description.

- 1)For multiple frames with short intervals that depict the same continuous action, you may merge them into a single description.
- 2)For example:This video begins at 0.0s with a scene featuring two individuals seated outdoors, engaging in a conversation. The subtitles indicate they are discussing the impact of the pandemic on their ability to shoot videos at a bar. By 14.0s, the dialogue shifts to their newfound regular appearances on a show called Scam Nation. At 28.0s, the conversation turns to the promotion of a product named Kraken, encouraging viewers to visit a website for... Output Format: Your response should be in the following format, wrapped with ⟨caption⟩⟨/caption⟩ tags: ”⟨caption⟩This clip (video) XXX⟨/caption⟩”.

quires on average 10.5 reasoning rounds per question. Therefore, C1 = 10.5. The video qa model is invoked infrequently, with an average of C3 = 0.36 calls per question.

Since the average tree width for VideoMME-Long is W = 5, the initial step requires obtaining the W highestlevel captions. During the subsequent reasoning process, every reasoning step except the one that triggers a video qa call requires an additional caption. Thus the expected number of caption calls is:

###### C2 = W + C1 − 1 − C3 = 5 + 10.5 − 1 − 0.36 = 14.14.

Assuming Qwen-2.5-VL-32B is used for both video cap and video qa, the empirical average runtimes are:

T1 ≈ 2.5s, T2 ≈ 7.0s, T3 ≈ 2.7s.

Therefore, the expected end-to-end time required to an-

swer a single question on VideoMME-Long is: T = 10.5 × 2.5 + 14.14 × 7.0 + 0.36 × 2.7 ≈ 135s. This result reflects the full hierarchical search procedure,

including both caption retrieval and occasional fine-grained Video QA queries.

### C. More Qualitative Examples

We provide additional qualitative results (Figure6, Figure 7) and ultra-long video examples (Figure 5) in this section. These examples illustrate LongVideo-R1’s ability to

perform hierarchical search, disambiguate similar scenes across hours-long content, and jointly use both high-level and fine-grained information.

The examples include cases from TV series such as Downton Abbey, where the model successfully navigates multi-hour narratives, repeatedly locating the correct characters, objects, or events despite substantial visual similarity across episodes.

### D. Failure Examples

Although LongVideo-R1 performs well across various long-video benchmarks, failure cases still occur (Figure 8, Figure 9). When a visually similar but irrelevant object appears in the video, the model sometimes commits to the wrong branch and fails to return to the correct segment.

We also find that simple textual hints can often guide the model back to the correct segment and enable it to produce the correct answer.

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

After Miss Mary introduced Mr. Pamuk to her father, Earl Robert, what questions did Robert ask Pamuk enthusiastically?

A 15-episode TV series with a duration of over 10 hours

<think>…This is a specific dialogue moment, likely in a scene involving Lady Mary Crawley and Kemal Pamuk, which is mentioned in High-level Caption 3. To answer accurately, I need to look at the medium or low level captions around the…</think>

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

<tool>video_cap(3,6)</tool>

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

<think>…The medium-level caption describes scenes of a fox hunt and a church visit, but

it does not mention the introduction of Kemal Pamuk to Robert Crawley. The introduction

scene is likely in another sub-segment of this high-level segment…</think>

- <tool>video_cap(3,6,1)</tool>

<think>…caption now contains the specific lines of dialogue: Robert Crawley says "Heavens, you have been in the wars," then asks Pamuk "How do you do, My Lord? Did you have a good day?" This matches the question asked after the introduction.…</think>

- <tool>video_cap(3,7,2)</tool>

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

<think>The introduction scene is fully described: Robert greets Kemal Pamuk and asks, "How do you do, My Lord? Did you have a good day?" This corresponds to the question, asking about his day’s experience. To ensure accuracy, I can confirm if any other question was asked. The low-level captions show no other question, so the answer is clear.</think> <answer>Robert asked Pamuk’s experiences of the day.<answer>

[Figure 142]

[Figure 143]

Figure 5. More example on ultra-long videos.

|[Figure 144]|
|---|

|[Figure 145]|
|---|

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

###### Question：How many sticks does the protagonist put in the incense burner?

<think>The question asks about the number of sticks the protagonist puts in the incense

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

burner. This detail is likely in a scene involving lighting incense, which is mentioned in High-level Caption 2 between 635.0s and 654.2s … <think> <tool> VideoCap(2,1) </tool>

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

<think>The medium-level caption (2,1) confirms that the incense lighting happens

between 624.8s and 650.8s, but it still does not specify the number of incense sticks used.

To find that detail, I need the low-level caption for the sub-segment… <think> <tool> VideoCap(2,1,2) </tool>

<think> …does not mention the number of sticks either. It seems none of the retrieved

captions explicitly state the number of incense sticks.The detail is likely in the initial

[Figure 170]

[Figure 171]

lighting or placement scene, but captions have omitted it…<think> <tool> VideoQA((2,1,3), How many incense sticks the man place in the burner?)</tool> <answer>The protagonist puts one incense stick into the incense burner. <answer>

Figure 6. More qualitative examples.

|[Figure 172]|
|---|

|[Figure 173]|
|---|

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

###### Question：During Putin's first speech on stage in the video, how many women appeared when the camera panned offstage?

<think>The question asks about Putin's first speech on stage and how many women appeared when the camera panned offstage. This is a very specific visual detail likely to

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

occur in a segment showing Putin speaking. From the high-level captions… <think>

<tool> VideoCap(1,1) </tool> <think>he medium-level caption for (1,1) describes Putin speaking at the podium but does not mention the number of women appearing when the camera panned offstage. To find this detail, I need to look at the low-level captions for the part of his …<think>

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

<tool> VideoCap(1,1,6) </tool>

<think>The low-level caption (1,1,5) mentions a transition to the audience between 63.5s and 65.9s, which could be when the camera pans offstage, but it does not specify the number of women. Since the exact detail is missing, and we already have …<think>

[Figure 198]

[Figure 199]

<tool> VideoQA((2,1,3), how many women appear when the camera pans offstage? )</tool>

<answer> one woman appears among the seated audience. <answer>

Figure 7. More qualitative examples.

|[Figure 200]|
|---|

|[Figure 201]|
|---|

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

###### Question: What kind of transportation do the people carry on the boat?

<think>The question asks about what kind of transportation people carry on the boat. The high-level captions mention boats in High-level Caption 6, but they don't specify what transportation is carried on the boat. To find this detail… <think> <tool> VideoCap(6,6) </tool>

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

<think>The medium-level caption for (6,6) mentions a boat scene but does not specify what transportation is carried on the boat. To find that detail, I need to look at the lowlevel captions for the boat scene, which occurs between 3074.6s and possibly…<think> <tool> VideoCap(6,6,1) </tool>

<think>The low-level caption (6,6,2) does not mention the boat or transportation carried on it. The boat scene seems to be in (6,6,1), but that caption didn't specify what's on the boat. Since captions are not giving the needed detail, I should use video_qa on (6,6,1) …<think> <tool> VideoQA((6,6,1), What transportation do the people carry on the boat?)</tool> < observation >no other form of transportation being carried on the boat. < /observation >

[Figure 226]

[Figure 227]

<think>The video_qa result indicates that no vehicle or transportation is carried on the boat, Given the information, the correct answer is not present.<think> <answer> No transportation was carried on the boat </answer>

Your answer is incorrect, I will give you some hints: The segment that contains information relevant to this question is 4.

<think> The user indicates that the relevant segment is 4, but I have been focusing on segment 6. This suggests that the boat scene with carried transportation is in High-level segment 4, not 6. I need to reorient. High-level 4 mentions a. … <think> <tool> VideoCap(4,1) </tool>

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

<think> The low-level caption (4,4,6) clearly mentions that the group is unloading a

motorcycle from the boat. This directly answers the question about what kind of

transportation the people carry on the boat. The correct answer is "Motocycle." <think> <answer> People carried a motocycle on the boat </answer>

- Figure 8. LongVideo-R1 may sometimes be disturbed by similar information, but people can guide the model back on track with just a few hints.

|[Figure 236]|
|---|

|[Figure 237]|
|---|

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

HURRICANE BONNIE NYMERIA Five Dogs BUDDY LUCY

###### Question: What is the name of the dog that plays the piano?

###### A 102 minutes video with about 20 performers

<think>The question asks for the name of the dog that plays the piano. The high-level captions provided do not mention any dog playing piano; they describe various dog acts, tricks, and performances, but not the specific detail of a dog playing piano … <think> <tool> VideoCap(6,3) </tool>

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

<think>…within High-level 6, possibly (6,2) or (6,4). Since High-level 6 mentions a woman performing with a dog accompanied by a pianist, I should check the mediumlevel caption for (6,2) to see if it contains the piano act and possibly the dog’s …<think> <tool> VideoCap(6,2) </tool>

<think>The videoqa result for (6,2,5) indicates the dog's name is "Mouse" but clarifies that the dog isn't shown playing the piano in these frames. However, the question specifically asks for the dog that plays the piano, so "Mouse" might not be the correct answer…<think> <tool> VideoQA((6,2,2), "What is the name of the dog that plays the piano?")</tool> < observation >The name of the dog that plays the piano is not provided . < /observation >

[Figure 262]

[Figure 263]

<think>After multiple attempts, none of the captions or videoqa results have revealed the dog's name. I cannot deduce it from the given information …<think> <answer> I cannot deduce the dog’s name. </answer>

Your answer is incorrect, I will give you some hints: The segment that contains information relevant to this question is 1.

<think> The user indicates that the relevant segment is 1, but the high-level caption for segment 1 describes a dog performing tricks, not playing piano. I should get the mediumlevel caption for (1,3) or (1,4) to see if the dog's name is given… <think> <tool> VideoCap(1,3) </tool>

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

<think> The caption (1,3,4) explicitly mentions the dog's name as Bonnie, confirming that

Bonnie is the dog in the act with the piano. Since the question asks for the name of the

dog that plays the piano, the correct answer is Bonnie.<think> <answer> The name of the dog that plays the piano is Bonnie. </answer>

- Figure 9. LongVideo-R1 may sometimes be disturbed by similar information, but people can guide the model back on track with just a few hints.

