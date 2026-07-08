# arXiv:2512.08511v2[cs.CV]11Dec2025

### Thinking with Images via Self-Calling Agent

Wenxi Yang1* Yuzhong Zhao1* Fang Wan1 Qixiang Ye1† 1University of Chinese Academy of Sciences

yangwenxi24@mails.ucas.ac.cn zhaoyuzhong20@mails.ucas.ac.cn wanfang@ucas.ac.cn qxye@ucas.ac.cn

###### vstar

###### hrbench-4k

95

91.6

90.1

Interleaved Multimodal Chain of Thoughts

90

[Figure 1]

SubagentVL-7B

...

DeepEyes-7B

85

Hard to incentivize

80

77.0

75.1

75

SubagentVL-7B

71.2

Agentic RL 🚀

DeepEyes-7B

68.8

Self-Calling Chain of Thoughts (Ours)

70

[Figure 2]

Qwen2.5-VL-7B

66.0

...

Qwen2.5-VL-7B

65

GPT-4o

Easy to incentivize

59.0

60

[Figure 3]

[Figure 4]

GPT-4o

...

55

50

Figure 1. Comparison of interleaved multimodal and Self-Calling Chain-of-Thought (sCoT). Left: The interleaved multimodal chain of thought (iMCoT) in DeepEyes [43] requires agents to perform tightly interleaved vision–language reasoning, which is difficult to incentivize. In contrast, the proposed sCoT reformulates the process as a language-only reasoning trajectory (the context may contains images, but the reasoning itself is language-only), making multimodal reasoning much easier to incentivize. Right: The model trained with sCoT outperforms interleaved multimodal chain-of-thoughts upon the challenging benchmarks with ∼ 75% fewer GPU hours.

#### Abstract

up to 1.9% with ∼ 75% fewer GPU hours compared to strong baseline approaches. Code is available at this url.

Thinking-with-images paradigms have showcased remarkable visual reasoning capability by integrating visual information as dynamic elements into the Chain-of-Thought (CoT). However, optimizing interleaved multimodal CoT (iMCoT) through reinforcement learning remains challenging, as it relies on scarce high-quality reasoning data. In this study, we propose Self-Calling Chain-of-Thought (sCoT), a novel visual reasoning paradigm that reformulates iMCoT as a language-only CoT with self-calling. Specifically, a main agent decomposes the complex visual reasoning task to atomic subtasks and invokes its virtual replicas, i.e. parameter-sharing subagents, to solve them in isolated context. sCoT enjoys substantial training effectiveness and efficiency, as it requires no explicit interleaving between modalities. sCoT employs group-relative policy optimization to reinforce effective reasoning behavior to enhance optimization. Experiments on HR-Bench 4K show that sCoT improves the overall reasoning performance by

#### 1. Introduction

The thinking-with-images paradigms like OpenAI o3 [21, 25, 26, 29, 31, 33, 43], which are built on powerful interleaved multimodal large language models (MLLMs), have demonstrated impressive visual reasoning capabilities by incorporating images as dynamic elements in the reasoning process. A key factor behind their success is reinforcement learning, which incentivizes valuable interleaved multimodal chains-of-thought (iMCoT) from MLLMs and further stimulates the emergence of “thinking with images”, Fig. 1(top left).

However, incentivizing iMCoT in MLLMs through reinforcement learning is substantially more challenging than training non-interleaved or language-only Chain-ofThought (CoT), as existing interleaved multimodal reasoning corpora are relatively scarce compared to language-only reasoning data. This limitation not only makes iMCoT

* Equal contribution. † Corresponding Author.

[Figure 5]

[Figure 6]

👩

👨

###### Where was this photo taken?

Is the bell above or below the clock?

🤖

🤖

<think> The image shows a tall clock tower with a bell at its base. There is a clock face on the front of the tower. Below the clock face, there is a space that appears to be an opening where a bell would be located.

<think> The image shows a memorial with inscriptions in Russian and an inscription "9 МАЯ" which translates to "9 May," which is often associated with Victory Day ... Then, I will use the VLM subagent to get more detailed information about the location by grounding the objects of interest in the image and asking speciﬁc questions. <think> <tool_call> {"prompt": "Ground the monument in the image.",

To determine the relative position of the bell and the clock, we need to focus on the area directly below the clock face. Using the grounding prompt, I will zoom in on the region just below the clock face to identify the bell. <think>

"img_idx": 0, "task_type": "subregion_grounding", "bbox_2d": [486, 175, 1073, 1254]} </tool_call>

<tool_call> {"prompt": "Please outline the position of the bell and the clock and output all the coordinates in JSON format.", "img_idx": 0, "task_type": "grounding", "bbox_2d": [785, 469, 1150, 1023]} </tool_call>

[Figure 7]

[subtask: caption] the call subagent monument in the image.

🤖

[Figure 8]

[subtask: grounding] Please outline the position of the bell and the clock and output all the coordinates in JSON format.

<tool_response> The monument is located on a paved area, which appears to be part of a park or memorial site. </tool_response>

subagentcall 🤖

<think> I need to read the text on another monument.</think> <tool_call> {"prompt": "text on the monument", "img_idx": 0, "task_type": "subregion_grounding", "bbox_2d": [1101, 1092, 1331, 1214]} </tool_call>

<tool_response> ```json [ {"bbox_2d": [49, 60, 71, 85], "label": "bell"}, {"bbox_2d": [39, 120, 61, 133], "label": "clock"}] ``` </tool_response>

[Figure 9]

<think>The bell is positioned below the clock on the tower in the image. The VLM model has conﬁrmed that the bell and the clock are correctly grounded, and their relative positions match the observation made by looking closely at the image.</think> <answer> The bell is located below the clock. </answer>

[subtask: ocr] text on call subagent the monument

🤖

<tool_response> The text on the monument translates to: "TO THOSE WHO FELL IN BATTLE FOR FEODOSIA DURING THE YEARS OF THE CIVIL AND GREAT PARENTAL WAR." </tool_response>

🤖🤖🤖

All are virtual replicas that execute on the same served

<answer> Based on the inscription, the photo appears to have been taken at a memorial dedicated to those who fell in battles for Feodosia during the Civil War and the Great Patriotic War. </answer>

backend of the visual language model.

- Figure 2. Thinking-with-images-through-self-calling. A visual language model termed main agent decomposes a complex visual query into simple, atomic subtasks handled by the model’s virtual replicas (“subagents”). Each subagent solves a localized atomic subtask (grounding, captioning, OCR, etc.) and returns textual outputs which are aggregated by the main agent to produce the final answer.

agents hard to optimize, but also undermines their reasoning and tool-calling capabilities.

In this study, we propose the Self-Calling Chain-ofThought (sCoT), a novel visual reasoning paradigm that reformulates iMCoT as language-only CoT augmented with self-calling, Fig. 1(bottom left). A vision-language model, termed the main agent, decomposes a complex visual reasoning task into a sequence of simple atomic subtasks, termed subagents, identical to the language model itself. These atomic subtasks are sufficiently simple to be solved through language-only CoT. In particular, subagents are designed as virtual replicas, which shares exactly the same weights with the main agent. Once all the subtasks are

accomplished by the subagents, the main agent aggregates their textual responses and derives the final answer.

The proposed sCoT can be effectively incentivized through end-to-end reinforcement learning. We optimize the main agent’s reasoning trajectory using group-relative policy optimization [22] (GRPO), while masking out the textual outputs returned by subagents to avoid reward leakage. This setup allows the model to learn effective selfcalling behaviors—when to call subagents and how to structure subtasks—without directly overfitting to subagent responses. With above design, the proposed sCoT paradigm does not require interleaved multimodal reasoning capabilities from the agents, Fig. 1(bottom left), which makes it

much easier to optimize than iMCoT. This design also facilitates saving memory and enhancing generalization of the reasoning capabilities learned by the agents, Fig. 2.

Experiments conducted on V* benchmark and HRBench 4K demonstrate that sCoT delivers superior visual reasoning capabilities. As shown in Fig. 1(right), it outperforms the state-of-the-art iMCoT approach DeepEyes [43] by 1.2% and 1.9% on V* benchmark and HR-Bench 4K respectively. In addition, since sCoT is easier to incentivize than iMCoT, it incurs much lower training costs compared to DeepEyes, thich is trained with iMCoT. Specifically, sCoT achieves better results than Deepeyes with only ∼25% of the GPU hours.

#### 2. Related Work

Thinking with Images extends Chain-of-Thought (CoT) reasoning from the language area to the multimodal domain by enabling MLLMs to actively inspect visual content [19]. Prior studies enhance perception through exploration tools such as zooming, object detection, and segmentation [9, 16, 20, 28, 31, 34], allowing models to examine localized regions of an image. Other approaches strengthen visual reasoning by designing multimodal CoTs —e.g., Visualization-of-Thought [31] and Visual Abstract Thinking [13]—or by constructing manually defined workflows and tool chains that convert visual information into language-friendly intermediate representations [4, 32]. Unlike these pipelines, this study bridges the modality gap without external experts or handcrafted workflows. We instead leverage the MLLM’s intrinsic visual abilities through self-invoked atomic subtasks and use language—only reasoning to coordinate them, yielding a lightweight and flexible alternative to existing multimodal CoT systems.

Reinforcement Learning. Recent advances in reinforcement learning, such as GRPO [22] and its variants [2, 15, 35, 36, 41, 42] offer powerful paradigms for optimizing policies in sequential decision-making. These approaches naturally extend to multimodal large language models (MLLMs) [1, 3, 4, 10, 23, 38] and agentic frameworks [5–8, 10], where RL is used to learn when and how to perceive, reason, and act across visual and textual modalities. In thinking-with-images settings, prior work (e.g., Chain-of-Focus [37], ACTIVE-o3 [44], DeepEyes [43]) typically relies on multimodal CoTs with interleaved visual observations and tool invocations to guide region selection and tool usage. Nevertheless, such heterogeneous reasoning trajectories demand consistent reasoning across modalities, making them difficult to incentivize directly through reinforcement learning. In contrast, we adopt a pure languagebased CoT, applying reinforcement learning only to textual reasoning while accessing tools via self-calling.

#### 3. Preliminary: Thinking with Images

The reasoning process of thinking-with-images can be formulated as an interleaved Multimodal Chain-of-Thought (iMCoT), in which the reasoning modality alternates between textual and visual domains (Fig. 1, top left).

While iMCoT encourages fine-grained visual reasoning, it produces long interleaved chains in which images and text alternate, placing heavy reasoning burden on current MLLMs. Empirically, state-of-the-art models match specialized detectors on single-image tasks suffer pronounced performance degradation as the number of images increases [30]. This degradation becomes more severe when the model must integrate cues across multiple key images or under noisy conditions. For instance, open-source models such as Qwen2-VL-7B-Instruct suffer more than 15% accuracy drops in two-image visual needle-in-a-stack benchmark [30], highlighting the difficulty of scaling visual reasoning beyond single-image contexts.

#### 4. The Proposed Approach 4.1. Self-Calling Chain-of-Thought

State-of-the-art MLLMs already excel at simple, atomic vision–language tasks such as grounding, OCR, and captioning. Motivated by this, we propose the Self-Calling Chainof-Thought (sCoT), an alternative reasoning formulation that avoids jointly processing interleaved modalities. As shown in Fig. 2, a complex visual language task are decomposed to a sequence of atomic subtasks, each aligned with fundamental visual capabilities that the MLLM excels. To reduce the reasoning burden of each step, each subtask is processed in an isolated context, independent of the main reasoning chain. The main reasoning chain then aggregates the textual responses returned by these subtasks to derive the final answer.

Pipeline. The overall pipeline of sCoT is illustrated in Fig. 3. Three elements play important roles.

- • Main Agent: We call the MLLM which receives the user question as the main agent, which understands the user question and decompose it into atomic subtasks. After all subtasks are accomplished, the main agent collects all the textual responses and aggregates them to a final answer, Figs. 2 and 3.
- • Subtask: A subtask consists of

- (a) a simple yet clear question,
- (b) a cropped subregion of the image in the original task

- • Subagent: Each subagent is an identical virtual replica of the main agent.

Here, a virtual replica means that each subagent is a unmaterialised instance of the same model–sharing the exact weights and parameters–without deploying any additional

[Figure 10]

[Figure 11]

###### Self-calling Chain-of-Thoughts (sCoT)

| | |
|---|---|
| | |

[Figure 12]

## ...

CoT

Tool Call

Resp. CoT

Tool Call

Resp.

CoT

[Figure 13]

"the scene depicts a monument, ..."

"The text translates to ... "

Caption

OCR

[Figure 14]

[Figure 15]

CoT

CoT

sCoT answer

[Figure 16]

[Figure 17]

sCoT answer

Group Computation

sCoT answer question

rollout

update 🔥

- Figure 3. Diagram of sCoT. During training, the main agent receives a task given by users and decomposes it into subtasks and invokes parameter-sharing subagents iteratively through pure-language CoT steps. This self-calling mechanism allows the model to simulate toolusing behaviors purely in the language domain. Then, the resulting CoT responses are used to calculate rewards, which includes three aspects: accuracy (Racc), format (Rformat), and tool-usage (Rtool). Finally, GRPO is used to incentivize sCoT.

physical models. In practice, invoking/calling a subagent simply issues another forward call to the LLM server that runs the main agent. Each subagent focuses on a single isolated subtask, processes one cropped image at a time, and returns a concise textual response to the main agent.

Tool Calling Protocol. The self-calling interaction is realized through structured tool calls, enabling the main agent to invoke subagents dynamically during reasoning. Each invocation is parameterized by three arguments, as

- (1) a task type (str), which briefly specifies the subtask category (e.g., OCR, visual question answering, caption, etc.);
- (2) a prompt (str), representing the textual query or instruction for the subtask; and
- (3) a bounding box (list[int]), defining the spatial region of interest within the image.

Subagents operate independently across tool calls, without access to global information or communication with other agents, ensuring isolated and modular reasoning. To provide limited local context while maintaining focus, and inspired by region-level understanding methods such as ControlCap [39] and DynRefer [40], each bounding box is slightly enlarged through the following Equation

Bbox = Bboxground · (1 − α) + Bboxcanvas · α, (1) where α is the interpolation factor (0.05 by default). Bboxground and Bboxcanvas are the bounding boxes of the

grounding result and the size of the image canvas respectively. The enlargement allows the subagents to capture additional contextual information for more comprehensive region-level understanding.

###### Subtask Prompt Template

System: You are a helpful assistant serving as a subagent. Given a cropped image and a question, answer the question and return the result. If further information is needed, encourage the user to make another call to the tool.

User: {cropped image} [subtask: {task type}] {prompt}

##### 4.2. Agentic Reinforcement Learning

The next challenge is enabling the model to learn orchestration of self-calling interactions. Rather than manually defining workflows in the system prompt or relying on incontext examples, we employ end-to-end agentic reinforcement learning to explicitly reward coherent self-reflection and efficient subtask coordination.

Rollout. Although our reasoning traces resemble language-only CoT, the underlying decision process is still naturally expressed as a Markov decision process composed of interleaved states and responses. At step t, the

state st of CoT is defined as

st = {(X1,R1),(X2,R2),...,(Xt,Rt)}, (2)

where R≤t = {R1,R2,...,Rt} denotes the accumulated textual responses from subagents (subtask results), and X≤t = {X1,...,Xt} represents the reasoning tokens generated by the main agent up to step t. These textual tokens may include intermediate thoughts, subtask plans, and summary statements. Therefore, we name st as the Self-Calling Chain-of-Thought (sCoT). The agent now learns a policy that maximizes long-term reward over this language-only reasoning trajectory, effectively aligning self-reasoning with the goal of accurate and efficient visual understanding.

Reward Design. We adopt the reward design from DeepEyes [43] with a minor modification to prevent potential reward hacking. For a rollout trajectory τ, the total reward is defined as

acc>0 · Itool≺ans · Rtool (3)

R(τ) = Racc + Rformat + IR

where Racc measures the correctness of the final answer, and Rformat penalizes ill-structured or malformed outputs. The key modification lies in the tool usage bonus Rtool, which originally is granted only when the model produces a correct answer and invokes at least one external perception tool during reasoning. Take a further step, to ensure causal consistency, the tool invocation must occur before the final answer is generated, and therefore an indicator Itool≺ans is incorporated into Eq. (3). This constraint encourages meaningful visual grounding while discouraging reward exploitation through post-hoc or redundant tool calls.

Optimization. We employ Group Relative Policy Optimization (GRPO) [22] as our reinforcement learning algorithm, owing to its proven efficiency and stability across diverse reasoning and decision-making tasks. During training, all tool calls within a rollout are executed sequentially to preserve causal consistency between reasoning and perception steps. For multi-turn agent trajectories, we apply a token-wise loss mask to exclude gradients on observation tokens (i.e. subagents’ responses) that are not generated by the model itself, ensuring that optimization focuses solely on the model’s own reasoning and action outputs, although the main agent and subagents are running on the same weights.

#### 5. Experiment

##### 5.1. Setting

Baselines. To evaluate the effectiveness of DeepEyes, we assess the trained model, denoted as SubagentVL, on two

Table 1. RL Training Configuration.

DeepEyes think-with-images

SubagentVL think-through-self-calling

GPU 8/32 H100 8 A100 base model Qwen2.5-VL-7B Qwen2.5-VL-7B batch size 256 56 rollout 16 8 LLM-judge Qwen2.5-VL-72B Qwen2.5-VL-7B iterations 80 80

comprehensive benchmarks: V* [29] and HR-Bench. Both benchmarks systematically measure the perception capability of MLLMs across diverse tasks—including OCR, visual prompting, and attribute recognition—using highresolution images up to 8K. We compare SubagentVL against DeepEyes, which is trained via the same end-toend reinforcement learning process. In addition, we include comparisons with state-of-the-art models (e.g., GPT4o, o3) and specialized visual reasoning workflows such as SEAL, DyFo, and ZoomEye, to comprehensively benchmark the performance of our approach under varied reasoning paradigms.

Data. We train Qwen2.5-VL-7B-Instruct on the dataset collected by DeepEyes, which integrates diverse visual reasoning sources to enhance both perception and analytical capability. The corpus consists of three major subsets:

- (1) Fine-grained data (47%), derived from the V ∗ training set [29], containing high-resolution images and detailed perception questions designed to maximize the effectiveness of tool-based reasoning;
- (2) Chart data (30%) [17], composed of synthetic charts and graph images that enrich the diversity of visual elements and quantitative patterns; and
- (3) Reasoning data (23%), incorporated from the ThinkLite-VL dataset [27], which broadens task diversity and strengthens the model’s higher-level analytical reasoning skills.

Training Details. Comparing with DeepEyes, we adopt a similar yet simplified training setup for our Agentic RL process, as detailed in Tab. 1. The training is conducted over the veRL framework. During training, the main agent invokes subagents in a sequential manner, i.e., subtasks are executed one after another rather than in parallel. For reward computation, the accuracy reward Racc ∈ {0,0.8} is assigned by an LLM-as-a-judge, while the format reward Rformat ∈ {−0.2,0} and the tool-usage reward Rtool ∈ {0,1.2} are determined based on the correctness of output format and the existence and ordering of tool calls, respectively.

##### 5.2. Result

Training with sCoT is more efficient. The V* Benchmark and HR-Bench assess reasoning over ultra-highresolution images (2K–8K), where regions of interest are typically confined within 200 pixels. This extreme scale disparity poses challenges for base VLMs, which often fail to localize and reason correctly without auxiliary visual cues. Without relying on SFT initialization or in-context exemplars, both DeepEyes and our SubagentVL achieve strong end-to-end reinforcement learning performance, outperforming open-source models and GPT-4o across all three benchmarks. Remarkably, despite operating under more constrained training resources—smaller batch size (256 vs. 56), fewer rollouts, and less data access—SubagentVL surpasses baselines on V* and HR-Bench 4K, while remaining competitive on 8K. These results highlight that training with the purely textual sCoT framework is more resourceefficient and scalable than image-intensive methods such as iMCoT.

Impact on General Visual Ability. To investigate how reinforcement learning influences the model’s general visual capability, we evaluate whether basic skills such as OCR and grounding exhibit any improvement. The model is examined off the reasoning framework using standard benchmarks including RefCOCO[11], OCR [14], and POPE [? ]. Compared with DeepEyes, the results show that SubagentVL demonstrates only marginal gains in grounding and OCR (Tabs. 3 and 4), while achieving a notable improvement in hallucination benchmarks (Tab. 5). This outcome is reasonable, since only the reasoning trajectories within the main agent are optimized during RL, while the interleaved subagent responses are masked out from logprobability computation. In contrast to the substantial performance boost observed in complex visual reasoning tasks with sCoT, this suggests that the enhanced ability primarily emerges from learning an appropriate subagent-calling strategy rather than from improvements in low-level visual perception.

Training Dynamics. We investigate how the model’s tool-calling behavior evolves throughout the training process. By jointly analyzing the number of tool calls and the corresponding reward trends in Fig. 4, we identify three distinct stages. In the first stage, the frequency of tool calls drops sharply while the reward gradually increases, accompanied by large fluctuations in entropy, contrary to intuition. Later ablation results indicate that strict tool-calling constraints on the three parameters (task type, prompt, bounding) limit the main agent’s capability to formulate subtasks. This suggests that the main agent initially prefers to solve problems independently rather than invoking sub-

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

1.5

reward

1.0

0.5

- 1
- 2
- 3

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | |lea su|rn to c bagent|all s| |divers|e callin|g strat|egies|
|solve ta directl|sk y| | | | | | | | |
| | | | | | | | | | |

toolcalls

s

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
|stage|1|s|tage 2| |s|tage 3| | | |
| | | | | | | | | | |

0.8

0.6

entropy

0.4

0.2

0 10 20 30 40 50 60 70 80

training step

Figure 4. Training Dynamics. Training process could also be divided into three stages.

agents. During the second stage, the number of tool calls begins to rise, leading to a rapid improvement in reward as the agent learns to effectively delegate complex subtasks to subagents. Notably, the transition between the first two stages does not occur at a fixed training step. Finally, in the third stage, the agent consistently issues more subagent calls, indicating a matured coordination strategy between the main agent and subagents for handling challenging reasoning tasks.

##### 5.3. Ablation Study

Constraints on Tool Calling Protocol. When designing the tool-calling format in Sec. 4.1, an important question arises regarding whether arguments should be strictly constrained. Specifically, we examine whether stringtype arguments (task type and prompt) can be left empty, and whether the bounding-box argument should be optional—where an empty bounding box assigns the entire image to the subagent.

As shown in Fig. 5, relaxing these constraints leads to undesirable learning dynamics. Although number of tool calls no longer decreases in the early stage of training, it quickly converges to one, indicating the absence of complex calling strategies. In such cases, the final calling pattern typically degenerates into simply repeating the user query

- Table 2. Comparison on High-Rosolution Benchmarks. The mark * indicates reproduced results and † means results are evaluated using a Qwen2.5-7B-Instruct serving as a judge, while DeepEyes uses Qwen2.5-72B-Instruct.

V ∗ Benchmark HR-Bench 4K HR-Bench 8K Model Size direct relative overall FSP FCP Overall FSP FCP Overall GPT-4o [19] - - - 66.0 70.0 48.0 59.0 62.0 49.0 55.5 o3 [18] - - - 95.7 - - - - - manually-defined-workflow

SEAL [29] 7B 74.8 76.3 75.4 - - - - - DyFo [12] 7B 80.0 82.9 81.2 - - - - - ZoomEye [24] 7B 93.9 85.5 90.6 84.3 55.0 69.6 85.5 50.0 69.3

baseline

Qwen2.5-VL [1] 7B 73.9 67.1 71.2 85.2 52.2 68.8 78.8 51.8 65.3 Qwen2.5-VL [1] 32B 87.8 88.1 87.9 89.8 58.0 73.9 84.5 56.3 70.4

think-with-images

DeepEyes∗† 7B 91.3 82.9 88.0 92.0 58.3 75.1 85.0 57.0 71.0 DeepEyes [43] 7B 91.3 88.2 90.1 91.3 59.0 75.1 86.8 58.5 72.6

think-through-self-calling

SubagentVL† (Ours) 7B 93.0 89.5 91.6 93.3 60.8 77.0 87.0 58.3 72.6 ∆ (vs Qwen2.5-VL-7B) +19.1 +22.4 +20.4 +8.1 +8.6 +8.2 +8.2 +6.5 +7.3 ∆ (vs DeepEyes) +1.7 +1.3 +1.2 +1.8 +1.9 +1.9 +0.2 -0.3 0.0

- Table 3. Visual Grounding Benchmark Results. * indicates reproduced results.

Table 6. Evaluation on V ∗ w.r.t. tool calling protocol constraints.

constraints avg. calls direct relative overall

Model refCOCO refCOCO+ refCOCOg Qwen2.5-VL-7B* 88.19 81.62 82.86 DeepEyes* 88.95 82.56 83.5 SubagentVL (Ours) 88.56 81.97 82.96

w/o 0.8 89.6 80.3 85.9 w/ 1.5 93.0 89.5 91.6

- 1
- 2
- 3

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |w/ co w/o c|nstrain onstra|t int| | | | |
| | | | | | | | |
| | | | | | | | |

Table 4. OCR Benchmark Results.

toolcalls

Qwen2.5-VL DeepEyes SubagentVL OCRBench 0.844 0.846 0.845

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Table 5. POPE Benchmark Results for Hallucination Evaluation. Higher is better. * indicates reproduced results.

0.75

entropy

0.50

Model Adv. Pop. Rand. Overall

0.25

QwenVL-2.5-7B 85.9 86.5 87.2 85.9 DeepEyes 84.0 87.5 91.8 87.7

0 10 20 30 40 50 60 70 80

training step

QwenVL-2.5-7B* 86.6 87.7 88.7 87.7 SubagentVL (Ours) 87.2 88.5 89.7 88.4

Figure 5. Training dynamics w.r.t. existence of constraints on tool calling protocol.

while cropping a single object region, resulting in poor reasoning diversity. Quantitatively, as shown in Tab. 6, the unconstrained variant achieves significantly lower than the constrained version, suggesting that enforcing stricter argu-

ment requirements encourages the agent to explore more structured and effective tool-calling behaviors.

|maxim|um too|l calls| | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

toolcalls

2.5

0.0

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.5

reward

1.0

0.5

gradnorm

10

0

0 5 10 15 20 25 30 35 40

training step

Figure 6. Reward hacking occurs at step 20 when the ordering constraint is missing.

Reward Design. In the RL training setup of DeepEyes, the reward function does not include the ordering indicator Itool≺ans. In contrast, this term proves essential during the training of SubagentVL. As tool calls decrease sharply in the early training stage, SubagentVL tends to solve problems independently rather than invoking subagents. Without Itool≺ans, the model often abuses the reward function by repeatedly appending tool calls after the </answer> tag, leading to reward hacking and unstable policy learning. Incorporating this ordering constraint effectively prevents such behavior and promotes proper sequencing of tool use prior to final answer generation.

Data. We further investigate how different data types affect the learning dynamics of SubagentVL. Recall that our training corpus consists of three categories: Fine data (highresolution or detail-focused images), Chart data (complex structured plots), and Reason data (smaller images paired with arithmetic reasoning, commonsense inference, or problem solving). As shown in Fig. 7, incorporating more diverse data (Fine + Chart) stabilizes the training process: the entropy decreases more gradually, and the model maintains scores on HR-Benchmarks. However, adding Reason data leads to degraded performance. Both the training reward curves and final evaluation metrics show clear drops when Reason data is included. We attribute this to a mismatch between the nature of Reason dataset and the reasoning trajectories of sCoT. While Fine and Chart data encourage sCoT to attend to small visual regions, textual details, and spatially localized structures, the Reason data emphasizes abstract symbolic reasoning rather than visual grounding ability. This shifts the model’s focus away from precise region-based perception and tool-calling strategies, ultimately hindering its ability to generalize on

Table 7. Ablation study on impact of training data.

Fine Chart Reason HR4K HR8K Qwen2.5-VL 68.8 65.3

✓ 77.0 72.6 ✓ ✓ 76.3 72.6 ✓ ✓ ✓ 74.4 70.3

SubagentVL

1.5

reward

1.0

fine

fine+chart

0.5

fine+chart+reason

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.8

0.6

entropy

0.4

0.2

0 10 20 30 40 50 60 70 80

training step

Figure 7. Ablation study on impact of training data.

high-resolution visual tasks. These findings highlight that, for sCoT, data that strengthens fine-grained visual discrimination contributes more effectively than data emphasizing non-visual reasoning.

#### 6. Conclusion

We proposed the Self-Calling Chain-of-Thought (sCoT), a new paradigm that reformulates interleaved multimodal CoT reasoning trajectories into a sequence of atomic subtasks coordinated by the main self-calling agent. This design removed the reasoning burden of managing multimodal CoTs and enables efficient end-to-end RL. Despite limited computational resources, SubagentVL achieves striking performance on V* and HR Bench, outperforming open-source models and surpassing prior multimodal Chain-of-Thought paradigms and manually defined workflows. Ablation studies further show that strict toolcalling protocols, proper reward ordering constraints, and carefully curated training data are essential for stable learning and effective delegation. There findings highlight that structured textual decomposition, combined with lightweight visual subtasks, offers a scalable and resourceefficient solution for high-resolution multimodal reasoning.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 3, 7
- [2] Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346, 2025. 3
- [3] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, and Xihui Liu. Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning. arXiv preprint arXiv:2506.16141, 2025. 3
- [4] Zihui Cheng, Qiguang Chen, Xiao Xu, Jiaqi Wang, Weiyun Wang, Hao Fei, Yidong Wang, Alex Jinpeng Wang, Zhi Chen, Wanxiang Che, et al. Visual thoughts: A unified perspective of understanding multimodal chain-of-thought. arXiv preprint arXiv:2505.15510, 2025. 3
- [5] Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic entropy-balanced policy optimization, 2025. 3
- [6] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning. CoRR, abs/2505.16410, 2025.
- [7] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization. CoRR, abs/2507.19849, 2025.
- [8] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025. 3
- [9] Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. Promptcap: Prompt-guided taskaware image captioning. arXiv preprint arXiv:2211.09699,

2022. 3

- [10] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 3

- [11] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar, 2014. Association for Computational Linguistics. 6
- [12] Geng Li, Jinglin Xu, Yunzhen Zhao, and Yuxin Peng. Dyfo: A training-free dynamic focus visual search for enhancing

- lmms in fine-grained visual understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9098–9108, 2025. 7
- [13] Dairu Liu, Ziyue Wang, Minyuan Ruan, Fuwen Luo, Chi Chen, Peng Li, and Yang Liu. Visual abstract thinking empowers multimodal reasoning. arXiv preprint arXiv:2505.20164, 2025. 3
- [14] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 6
- [15] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025. 3
- [16] Zixian Ma, Jianguo Zhang, Zhiwei Liu, Jieyu Zhang, Juntao Tan, Manli Shu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Caiming Xiong, et al. Taco: Learning multimodal action models with synthetic chains-of-thought-andaction. arXiv preprint arXiv:2412.05479, 2024. 3
- [17] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 5
- [18] OpenAI. GPT-4o system card, 2024. 7
- [19] OpenAI. Thinking with images, 2025. 3, 7
- [20] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, and Jie Tang. Cogcom: A visual language model with chainof-manipulations reasoning. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [21] Runqi Qiao, Qiuna Tan, Minghan Yang, Guanting Dong, Peiqing Yang, Shiqiang Lang, Enhui Wan, Xiaowan Wang, Yida Xu, Lan Yang, et al. V-thinker: Interactive thinking with images. arXiv preprint arXiv:2511.04460, 2025. 1
- [22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3, 5
- [23] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 3
- [24] Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6613–6629, 2025. 7
- [25] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think

- with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025. 1
- [26] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 1
- [27] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025. 5
- [28] Yikun Wang, Siyin Wang, Qinyuan Cheng, Zhaoye Fei, Liang Ding, Qipeng Guo, Dacheng Tao, and Xipeng Qiu. Visuothink: Empowering lvlm reasoning with multimodal tree search. arXiv preprint arXiv:2504.09130, 2025. 3
- [29] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 1, 5, 7
- [30] Tsung-Han Wu, Giscard Biamby, Jerome Quenum, Ritwik Gupta, Joseph E. Gonzalez, Trevor Darrell, and David Chan. Visual haystacks: A vision-centric needle-in-ahaystack benchmark. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [31] Wenshan Wu, Shaoguang Mao, Yadong Zhang, Yan Xia, Li Dong, Lei Cui, and Furu Wei. Mind’s eye of llms: visualization-of-thought elicits spatial reasoning in large language models. Advances in Neural Information Processing Systems, 37:90277–90317, 2024. 1, 3
- [32] Yixuan Wu, Yizhou Wang, Shixiang Tang, Wenhao Wu, Tong He, Wanli Ouyang, Jian Wu, and Philip Torr. Dettoolchain: A new prompting paradigm to unleash detection ability of mllm. arXiv preprint arXiv:2403.12488, 2024. 3
- [33] Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vuli´c. Visual planning: Let’s think only with images. arXiv preprint arXiv:2505.11409, 2025. 1
- [34] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 3
- [35] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 3
- [36] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv

- preprint arXiv:2504.05118, 2025. 3

[37] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv

- preprint arXiv:2505.15436, 2025. 3

- [38] Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning. arXiv preprint arXiv:2503.05379, 2025. 3
- [39] Yuzhong Zhao, Yue Liu, Zonghao Guo, Weijia Wu, Chen Gong, Qixiang Ye, and Fang Wan. Controlcap: Controllable region-level captioning. In European Conference on Computer Vision, pages 21–38. Springer, 2024. 4
- [40] Yuzhong Zhao, Feng Liu, Yue Liu, Mingxiang Liao, Chen Gong, Qixiang Ye, and Fang Wan. Dynrefer: Delving into region-level multimodal tasks via dynamic resolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24742–24752, 2025. 4
- [41] Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, et al. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025. 3
- [42] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 3
- [43] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 1, 3, 5, 7
- [44] Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025. 3

