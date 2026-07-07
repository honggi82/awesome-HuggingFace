## LongVideoAgent: Multi-Agent Reasoning with Long Videos

### Runtao Liu* Ziyi Liu* Jiaqi Tang Yue Ma Renjie Pi Jipeng Zhang Qifeng Chen Hong Kong University of Science and Technology

rliuay@connect.ust.hk, ziyiliu0811@outlook.com

https://longvideoagent.github.io/

[Figure 1]

### Abstract

[Figure 2]

question

wrong answer

Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening temporal grounding and missing fine-grained cues. We propose a multi-agent framework in which a master LLM coordinates a grounding agent to localize question-relevant segments and a vision agent to extract targeted textual observations. The master agent plans with a step limit, and is trained with reinforcement learning to encourage concise, correct, and efficient multiagent cooperation. This design helps the master agent focus on relevant clips via grounding, complements subtitles with visual detail, and yields interpretable trajectories. On our proposed LongTVQA and LongTVQA+ which are episode-level datasets aggregated from TVQA/TVQA+, our multi-agent system significantly outperforms strong non-agent baselines. Experiments also show reinforcement learning further strengthens reasoning and planning for the trained agent.

[Figure 3]

# arXiv:2512.20618v1[cs.AI]23Dec2025

[Figure 4]

compress / down-sampling

single MLLM

[Figure 5]

long video

Multi-round multi-agent reasoning

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

question

...

correct answer

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

...

[Figure 15]

long video

Figure 1: Traditional single-pass MLLMs that ingest entire long videos in one context—typically (may through heavy downsampling and compression) often miss crucial evidence and produce wrong answers, whereas LONGVIDEOAGENT conducts multi-agent, multi-round, and multimodal reasoning to extract sparse, task-relevant cues and answer correctly.

temporally extended queries. Crucially, most prior systems are non-agentic models: they process a static, pre-encoded or down-sampled video. Converting the full visual stream into compressed representations in the LLM’s textual space shifts the burden of temporal reasoning to this early stage—often lossy and irreversible, making it difficult to recover fine-grained evidence. These limitations motivate an agentic, tool-augmented paradigm that can actively decide what to observe next, when to query external visual or other tools, and when enough grounded evidence has been gathered to respond. Despite recent advances, the field still lacks a solution that jointly achieves efficiency, multimodal completeness, and fine-grained temporal reasoning in long videos.

### 1 Introduction

Multimodal large language models (MLLMs) extend LLMs beyond text to perceive and reason over multimodal signals, such as visual frames, audio, and subtitles. A key emerging challenge is robust long video understanding, where information is sparsely distributed across hours of content and multiple modalities (e.g., frames, and dialogue cues). Early instruction-tuned systems such as Video-LLaMA (Zhang et al., 2023; Lin et al., 2024) demonstrated that LLMs can be adapted to jointly process sampled video frames, marking an initial step toward multimodal video reasoning. However, current models remain limited to short clips or coarse summaries and struggle with fine-grained,

Recent works have begun to frame long video understanding as an agent-driven process, rather than a passive encoding task. Notably, VideoAgent (Fan et al., 2024; Wang et al., 2024b) introduced an agent-based framework where a central LLM actively conducts video analysis. In this

* Equal Contribution.

paradigm, the LLM agent iteratively queries external vision models (tools) to retrieve and interpret video frames, progressively compiling the information needed to answer a given query. This interactive strategy mirrors human cognitive behavior and has demonstrated promising effectiveness. These findings highlight the potential of tool-augmented LLM agents in achieving both efficiency and accuracy. However, the initial incarnation of VideoAgent relies on a less powerful toolset, primarily generic vision-language foundation models for captioning and image retrieval. Such tools are often insufficient for capturing fine-grained semantics, precise object references, or subtle temporal cues. This restricts the agent’s ability to understand complex scenes and reason over long temporal spans. Moreover, current frameworks underutilize the LLM’s inherent reasoning abilities and lack mechanisms for multi-step decision making or reinforcementbased planning.

In this paper, shown as Figure 1 we address these challenges by proposing a new multi-agent-based framework for long video understanding that strategically incorporates agents. Our system adopts a multi-agent architecture, where a central MASTERAGENT is responsible for reasoning and answering, while coordinating with other specialized agents. Specifically, a GROUNDINGAGENT locates video segments relevant to the question, and a VISIONAGENT extracts detailed visual information from the selected clips (e.g., objects, faces, actions). The master agent gathers these outputs to iteratively reason over the accumulated evidence. To guide the reasoning process, we design a reward-driven training strategy that encourages the master agent to conduct structured, multi-step reasoning. In each iteration, the master agent generates sub-queries, invokes either the grounding or vision agent as needed, and integrates the returned information before deciding on the next step. When it determines that enough evidence has been collected, it produces a final answer. By designing a reward function that penalizes irrelevant tool use and incoherent reasoning, we guide the agent to “think” in a proper format, effectively learning when to explore the video with tools and when it has gathered sufficient evidence to answer the question. Furthermore, to evaluate long-form video reasoning in a realistic setting, we construct a new benchmark dataset LongTVQA and LongTVQA+. This dataset extends the well-known TVQA video question answering task to much longer video durations, providing a

rigorous testbed for our agent.

Our Agent-with-Tools approach demonstrates superior performance on the LongTVQA benchmark, outperforming all existing baselines by a significant margin. Through ablation studies, we show that both the multi-agent architecture and the reward-guided training contribute critically to the agent’s gains. Our system not only achieves higher accuracy, but also exhibits interpretable decisionmaking, coordinating sub-agents to select relevant video segments and extract fine-grained visual information essential for reasoning. These results underscore the benefit of an agentic framework for long video understanding.

Our contributions are threefold: (i) a modular multi-agent architecture in which a master LLM coordinates grounding and vision specialists; (ii) a reward-driven agentic reinforcement learning training scheme that promotes concise, step-wise reasoning; and (iii) episode-level long video datasets LongTVQA and LongTVQA+ are proposed under which our system achieves state-of-the-art results.

### 2 Related Work

#### 2.1 Video Question Answering

Early work focused on memory and attention mechanisms over appearance–motion features (Gao et al., 2018). This evolved into multimodal transformers designed for efficient frame sampling (Lei et al., 2021). Recent trends emphasize retrievalaware reasoning and efficient tokenization for long videos, as well as integrating LLM-based reasoning with video encoders (Zhang et al., 2023) and employing agentic planners that iteratively gather evidence (Wang et al., 2024b). Long-form systems further explore sparse memory and temporal grounding techniques to handle hour-scale inputs (Song et al., 2024). These developments motivate long-form VideoQA systems that selectively retrieve segments under a limited context budget.

#### 2.2 LLM Agents

LLM agents couple chain-of-thought with actions: planning, tool calls, and iterative evidence gathering. Foundational agent ideas include ReAct, SelfAsk, and WebGPT (Yao et al., 2022; Press et al., 2022; Nakano et al., 2021). Toolformer shows self-supervised API-calling, while orchestration frameworks (HuggingGPT/Gorilla-style) route subtasks to expert models (Schick et al., 2023; Shen et al., 2023). In multimodal settings, MM-ReAct

wires LLMs to vision experts via prompting, and program-of-thought systems like ViperGPT compose perception modules through executable code for transparent, verifiable reasoning (Yang et al., 2023; Surís et al., 2023). For long videos, agentic designs such as VideoAgent/VideoAgent-style frameworks use memory, targeted retrieval, and temporal grounding to operate under strict context budgets while improving faithfulness (Wang et al.,

- 2024b). Beyond planning, video-RAG pipelines extract ASR/OCR/objects and retrieve evidence to augment LVLMs for factual responses (Luo et al., 2024). In addition, long-horizon multimodal agents with persistent memory and structural planning further enhance reliability for extended videos, e.g., Long-Seeing, VideoTree, and Koala (Long et al., 2025; Wang et al., 2025b; Tan et al., 2024); and general reasoning paradigms such as Chainof-Thought, Least-to-Most, Tree-of-Thoughts, and Generative Agents provide foundations for decomposition and memory (Wei et al., 2022; Zhou et al., 2022; Yao et al., 2023; Park et al., 2023). Retrievalfirst paradigms like Retrieving-to-Answer complement agent pipelines with a retrieve-then-reason template (Pan et al., 2023). (We also include the alternative ReAct entry for key consistency (Yao et al., 2022).)

#### 2.3 Multi-Modal LLMs

Modern MLLMs combine strong vision encoders with instruction-tuned LLMs. CLIP pretraining provides broad visual–text transfer (Radford et al., 2021). Flamingo introduces a perceiverstyle resampler for few-shot multimodal learning (Alayrac et al., 2022); BLIP-2/InstructBLIP bridge frozen encoders and LLMs (Li et al., 2023; Dai et al., 2023). Recent visually instruction-tuned MLLMs (Tang et al., 2025; Pi et al., 2024, 2025), such as LLaVA (Liu et al., 2023), scale visual instruction tuning using open components, while LLaVA-OneVision (Li et al., 2024a) unifies highresolution perception with token-efficient processing for both images and videos. Recent videotuned variants (e.g., Video-LLaVA) and trainingfree token schedulers (e.g., SlowFast-LLaVA) further improve temporal coverage and efficiency (Lin et al., 2024; Xu et al., 2024b). Proprietary MLLMs (GPT-4/4o; Gemini 1.5) show long-context multimodal reasoning (Achiam et al., 2023; Gemini Team, 2024), while open models (Qwen2-VL, InternVL) narrow the gap via dynamic resolution, OCR, and video pipelines (Wang et al., 2024a;

Chen et al., 2024). Complementary advances focus on unifying image–video tokens with few, informative representations (e.g., MiniGPT4-Video, VideoChatGPT, Video-LaVIT, LLaMA-VID, LongVU, PLLaVA, LLaVA-Video, Chat-UniVi) (Ataallah et al., 2024; Maaz et al., 2024; Jin et al., 2024b; Li et al., 2024b; Shen et al., 2024; Xu et al., 2024a; Zhang et al., 2024c; Jin et al., 2024a), and on long-context optimization or adaptive input selection (e.g., InternVideo2.5, LongVLM, Long Context training, self-adaptive sampling, simple-buteffective alignment, and question-instructed tuning) (Wang et al., 2025a; Weng et al., 2024; Zhang et al., 2024b; Han et al., 2023; Zhang et al., 2024a; Romero and Solorio, 2024). Comprehensive analyses of video understanding in large multimodal models (e.g., Apollo) situate these models within broader capabilities and evaluation protocols (Zohar et al., 2025). For key harmonization with the bibliography, we also include the alternate VideoLLaMA entry (Zhang et al., 2023). However, most models still face long-video constraints (context length, retrieval). This motivates combining videonative encoders, instruction tuning, retrieval, and tool use for scalable long-form VideoQA.

### 3 Method

As shown in Figure 2, we cast long-video QA as multi-agent reasoning, where a master agent LLM coordinates a grounding agent to temporally localize question-relevant segments and a vision agent to extract targeted observations from those segments. The system proceeds iteratively, maintaining a running context that accumulates subtitles, relevant segment tags, and vision observations, and it produces an answer once the master agent judges that sufficient evidence has been gathered. For opensource LLMs serving as the master agent, we apply reinforcement learning to encourage accurate, concise, and cooperation-efficient behavior while keeping the other agents frozen. At inference, the process yields clear, step-by-step traces aimed at solving the question at hand.

#### 3.1 Multi-agent System Framework

Master agent behavior and training. Specifically, the master agent follows the instruction schema in the System Prompt (Table 1) and the multi-turn policy in Algorithm 1 that coordinates two other specialist agents: a grounding agent and a vision agent. Given an episode with its full subti-

subtitles video frames

[Figure 16]

Penny: that’s ... Leonard: you ... Penny: I can’t ...

[Figure 17]

round , (max round )

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

return:

[Figure 22]

[Figure 23]

grounding result

[Figure 24]

💡Thinking

ground

[Figure 25]

which clip? conﬁdent to reply? need visual info? my next ac on ...

Penny: totally... Leonard: it’s ... Sheldon: ah ... Penny: please... Sheldon: no, ...

grounding agent

ask

call

[Figure 26]

master agent

[Figure 27]

[Figure 28]

[Figure 29]

return: vision info

ground

observe

[Figure 30]

Howard: let’s ... Leonard: okay... Sheldon: ah ...

user

reply

call

answer vision agent

watch

[Figure 31]

Figure 2: Architecture of LONGVIDEOAGENT. A MASTERAGENT runs for up to K rounds, collaborating with a GROUNDINGAGENT to localize relevant clips from videos and a VISIONAGENT to read fine-grained cues from the localized frames. Evidence accumulates until the MASTERAGENT feels confident to answer the user.

Table 1: System prompt for LONGVIDEOAGENT.

System Prompt — LONGVIDEOAGENT You are an agent that answers questions about a long video episode. You may use two tools: a grounding agent to localize relevant segments and a vision agent to extract visual facts from the localized segment. Produce concise, direct answers. Context you may receive. All subtitles and the user question q. When a segment has been localized, you will also have a tag <clipX> (e.g., <clip2>). When the vision agent has been called, you will see its textual response. Available actions (choose exactly one per turn).

- A — Visual query: If current visual information is insufficient, or you need visual details conditioned on the subtitles for the current <clipX>, call the vision engine with <visual_query>query </visual_query>.
- B — (Re)Grounding: If the current text/visual evidence conflicts with the question, or the current location cannot support a confident answer, call the grounding agent with <request_grounding>.
- C — Answer: If evidence is sufficient, return the final answer with <answer>... </answer>. The answer must be concise and direct.

Guidelines. (1) Be conservative with tool calls; answer when sufficient. (2) Do not hallucinate visual details; only use the vision agent for facts not inferable from subtitles. (3) Each turn targets the current <clipX> (if any); if none exists, prefer (re)grounding before visual query.

tles and a question, the master runs a bounded loop (at most K steps). At each turn it emits exactly one structured action token, <watch> for a visual read, <request_grounding> for (re)localization, or <answer> to terminate. After the corresponding agent is invoked, its textual output is appended to the context of the master agent. For open-source masters, we optimize the policy with GRPO while keeping the grounding and vision agents fixed. The rollouts terminated by action tokens in Algorithm 1 provide the trajectories for training and evaluation.

need, the vision agent extracts textual observations from frames within the localized segment (e.g., objects/entities, attributes, actions, OCR/on-screen text, scene cues). These observations are appended to the context and guide the next decision; the loop terminates when the master judges the accumulated visual evidence sufficient to answer. 3.2 Reinforcement Learning for

#### LONGVIDEOAGENT

For open-source LLMs serving as the master agent, we fine-tune the master with GRPO while keeping the grounding and vision agents frozen. Long-video QA is cast as a finite-horizon decision process: at each action step after reasoning the policy emits exactly one structured action token (<visual\_query>, <request_grounding>, or <answer>).

Grounding agent. Given the question and subtitles, the grounding agent proposes a temporal segment and returns a symbolic tag <clip_X> marking the relevant portion of the episode. By default the window context is 1; when larger, the agent outputs a short run of consecutive tags. The master may requery grounding to refine or validate the segment as reasoning progresses.

Trajectory. A full response terminates upon emitting <answer>...</answer> or reaching K steps. We index decision steps by t ∈ {0,1,...,T} with T ≤ K. At step t,

Vision agent. Conditioned on <clip_X> and an on-demand prompt that specifies the current visual

Algorithm 1 LONGVIDEOAGENT with Multi-Turn Reasoning

Require: Subtitles S; question q; video V ; MASTERAGENT parameters πθ; maximum steps K; GROUNDINGAGENT; VISION-

AGENT.

Ensure: Final answer yˆ.

- 1: Initialize rollout sequence y ← ∅
- 2: Initialize step count t ← 0
- 3: while t < K do
- 4: Initialize current action LLM rollout sequence yt ← ∅
- 5: while True do
- 6: Generate thinking token yi ∼ πθ(· | S, q, V, y + yt)
- 7: Append yi to rollout sequence yt ← yt + yi
- 8: if yi in [</visual_query>, </request_grounding>, </answer>, <eos>] then
- 9: break
- 10: end if
- 11: end while
- 12: y ← y + yt
- 13: if <visual_query> detected in yt then
- 14: Extract visual query qvis ← PARSE(yt, <visual_query>, </visual_query>)
- 15: Retrieve vision results d = VISIONAGENT(qvis, V )
- 16: Insert visual results into rollout y ← y + d
- 17: else if <request_grounding> detected in yt then
- 18: Retrieve grounding results clipTag = GROUNDINGAGENT(q, S)
- 19: Insert clip tag into rollout y ← y + clipTag + S(clipTag)
- 20: else if <answer> detected in yt then
- 21: Extract predicted answer yˆ ← PARSE(yt, <answer>, </answer>); Normalize yˆ (trim spaces/punctuation)
- 22: Insert final answer into rollout y ← y + "The answer is: " + yˆ
- 23: return final answer yˆ
- 24: else
- 25: Ask for rethink y ← y + "The action is not correct. Only <visual_query>, <request_grounding>, or <answer>."
- 26: end if
- 27: Increment step count t ← t + 1
- 28: end while
- 29: return final generated response y for q

the policy πθ first plans and then emits a contiguous action string at ending with exactly one closing tag from {</visual_query>, </request_grounding>, </answer>}. If not terminating, the system appends feedback from the invoked agent ot (e.g., a vision observation or a clip tag) to the context for the next step.

Rewards. We use two simple, rule-based rewards as supervision for reinforcement learning: (i) Structural validity rtfmt ∈ {0,1} grants 1 if the action string contains exactly one top-level tag with proper closure and no extraneous text; otherwise 0. (ii) Answer correctness rans ∈ [0,1] is awarded at termination via exact match on the multiple-choice answer; if no valid <answer> appears, rans = 0.

Objective and optimization. We seek a policy that produces well-formed actions at every step and a correct final answer. To balance these goals, the trajectory reward return is R(τ) = α Tt=0 rtfmt + rans where α > 0 weights the per-step structural shaping and rans supplies the terminal task reward. rtfmt encourages the master to emit exactly one correct action tag at each decision, while rans evaluates only the final <answer>. If no valid and correct answer is produced, rans = 0.

We optimize the master agent with GRPO on sampled rollouts: for each episode, the policy generates an action sequence, receives structural rewards at action boundaries and a terminal answer reward, and we compute sequence-level advantages with a learned value baseline. Policy updates follow the GRPO objective with standard clipping and entropy regularization, while the grounding and vision agents remain frozen. This minimal, two-signal objective provides sufficient guidance to learn structured, multi-turn coordination without additional dense rewards.

### 4 Experiments

#### 4.1 Datasets

We build LongTVQA and LongTVQA+ on top of TVQA and TVQA+. TVQA spans six TV shows with 152.5K multiple-choice QAs over 21.8K clips (60–90s) with subtitles and moment annotations; questions require joint dialogue–visual reasoning (Lei et al., 2018). TVQA+ refines a subset with spatio-temporal grounding—adding precise timestamps and 310.8K frame-level boxes for referenced entities (29.4K QAs from 4,198 clips, mainly TBBT)—supporting joint QA and temporal/spatial localization (Lei et al., 2020).

- Table 2: Performance on LongTVQA and LongTVQA+. The left block lists model attributes (Agentic, Input, RL fine-tune); the right block reports validation accuracy (%). GPT-4o and Gemini-2.5 Pro are multimodal baselines that process and accept the full long video directly. Methods labeled Agentic indicate the model operates as the MASTERAGENT; methods labeled AgenticRL additionally denote RL fine-tuning. Parenthesized green numbers denote absolute gains over the immediately preceding (non-agentic or non-RL) setting. We observe that: (i) our multi-agent framework, LONGVIDEOAGENT, consistently outperforms the non-agentic counterparts; (ii) agentic RL yields additional gains, especially for smaller open-source models; (iii) using frames provides visual evidence beyond subtitles, and generally outperforms subtitle-only inputs; (iv) closed-source models remain strong, but the gap narrows much when open-source models adopt agentic designs and agentic RL.

Accuracy (%) LongTVQA LongTVQA+

Method Multi-agent Input RL Finetune

Closed-source (M)LLMs GPT-4o ✗ Subtitle+Frame ✗ 70.78 78.32 Gemini-2.5 Pro ✗ Subtitle+Frame ✗ 78.90 81.28 GPT5-mini ✗ Subtitle ✗ 62.40 66.70 Agentic-GPT5-mini ✓ Subtitle+Frame ✗ 71.11(+8.71) 78.90(+12.20) Grok ✗ Subtitle ✗ 76.90 81.80 Agentic-Grok ✓ Subtitle+Frame ✗ 82.65(+5.75) 85.60(+3.80)

Open-source LLMs DeepSeek-R1(671B) ✗ Subtitle ✗ 68.99 75.04 Agentic-DeepSeek-R1(671B) ✓ Subtitle+Frame ✗ 70.30(+1.31) 79.70(+4.66) Agentic-Qwen2.5(3B) ✓ Subtitle+Frame ✗ 23.50 27.70 AgenticRL-Qwen2.5(3B) ✓ Subtitle+Frame ✓ 47.40(+23.90) 50.10(+22.40) Agentic-Qwen2.5(7B) ✓ Subtitle+Frame ✗ 46.10 60.30 AgenticRL-Qwen2.5(7B) ✓ Subtitle+Frame ✓ 60.20(+14.10) 70.80(+10.50)

To obtain LongTVQA and LongTVQA+, we aggregate all clips from the same TV episode into a single episode-level (hour-scale) sequence. For each episode, we merge the visual stream, subtitles, and all associated questions; clip timestamps are re-indexed into the episode timeline, and TVQA+ bounding boxes are preserved at their corresponding frames. Unless otherwise noted, we report results on the original validation splits after this episode-level aggregation.

#### 4.2 Baselines

We include both open-source and closed-source models (see Table 2), including representative opensource LLMs such as DeepSeek-R1 (Guo et al.,

- 2025) and Qwen2.5-3B/7B (Qwen et al., 2025) , and closed-source models such as Grok, GPT5mini (OpenAI, 2025), GPT-4o (OpenAI) and Gemini 2.5 Pro (Comanici et al., 2025). We adopt unified experimental settings to ensure comparability across backbones. We evaluate base LLMs in a nonagent mode and our agent system, with the base LLM serving as the master agent in both cases. In the non-agent mode, the model consumes the full subtitles and does not invoke grounding or vision agents. We then compare our agent system against the corresponding non-agent runs on the same backbones, so that observed gains can be attributed to agentic behavior rather than backbone

differences. For open-source backbones, we also report results with vs. without reinforcement learning under the identical evaluation protocol described in §4.3. Closed-source models are evaluated as released, without additional training.

#### 4.3 Experimental Setup

By default we use Grok-4-fast-reasoning for temporal localization and GPT-4o as the vision agent. The window context is set to 1, meaning the agent conditions on a single localized clip (no adjacent clips), and the maximum execution steps are K=5. All methods read the full episode subtitles. In the non-agent setting, no external modules are invoked. In the agent setting, the master agent receives symbolic temporal tag(s) <clip_X>, which is produced by the grounding agent, that marks the grounded clip(s) on the episode timeline, and it may request on-demand textual observations from the vision agent via prompting for selected frames within that segment (e.g., objects and attributes, OCR/text, brief scene cues). The master agent only consumes text (subtitles, the <clip_X> tag, and optional visual observations); no raw images are passed to the master agent. We report results on the validation splits of LongTVQA and LongTVQA+, using Answer Accuracy (Acc) as the primary metric (the questions are multiple choice) and additionally Grounding Accuracy for experiments that involve

- Table 3: A case of LONGVIDEOAGENT showing the collaboration among multiple agents.

Question & Choices

Question: Where is Sheldon sitting when he is accompanied by a man? Choices: a0: The Mall a1: A Movie Theatre a2: A Park a3: A Bus Stop ✓ a4: The Store

LONGVIDEOAGENT Execution MASTER AGENT:

<think> Subtitles alone do not specify the location; I will first localize with grounding, then request a visual read. </think>

<request_grounding> ⇒ GROUNDING AGENT GROUNDING AGENT:

Grounding information. Subtitles: “Sheldon: ... apparently we’re just two peas in a pod. ... Mrs Cooper: Here. Thank you.” Clip tag: <s05e06_seg02_clip_15>

###### MASTER AGENT:

<think> The grounded subtitles still don’t reveal the setting; I’ll request a visual description of the localized clip. </think>

<visual_query> visual description of scene in The Big Bang Theory S05E06 segment 02 clip 15 where Sheldon is with a man </visual_query> ⇒ VISION AGENT

###### VISION AGENT:

Visual description. Sheldon is sitting on a bench at night in an urban sidewalk setting; a waste container and building windows are visible nearby.

###### MASTER AGENT:

<think> Bench + sidewalk + trash can + windows strongly indicate a bus stop rather than a mall, theatre, store, or park. No further tools needed. </think>

<answer> a3: A Bus Stop ✓ </answer> ⇒ USER

clip grounding. For reinforcement learning, we use GRPO with a learning rate of 5 × 10−6, up to 2,000 optimization steps, a KL coefficient of 10−3, batch size 4, rollout count N=4, and temperature 1.0. Training Qwen2.5-7B took 12 hours on 4× NVIDIA H800 GPUs, while the 3B variant took 6 hours under the same setup.

#### 4.4 Performance

Table 2 presents overall validation accuracy. Moving from the non-agent setting to our multi-agent framework yields significante gains. This provides direct evidence for the effectiveness of a multi-agentic pipeline that can localize the relevant clips and performs targeted visual inspection. In addition, for several open-source LLMs(as master agent), reinforcement learning consistently improves over their inference-only counterparts under identical prompts and evaluation; notably, the Qwen2.5-7B model with RL attains accuracy comparable to GPT-5-mini (closed-source) on our protocol. Illustrative examples in Table 3 and Table 5 effectively demonstrate the efficacy of our approach, with additional cases provided in the supplementary materials.

Table 4: Ablations and analysis of LONGVIDEOAGENT.

- (a) Comparison of non-agent vs. multi-agent performance. Agentic components progressively improve performance: adding grounding outperforms the non-agent baseline, and adding vision agent yields the best results.

Setting Accuracy (%)

Non-agent (Text-only) 64.3 Multi-Agent (Grounding) 69.0 Multi-Agent (Grounding + Vision) 74.8

- (b) Effect of max steps K. Increasing the MASTERAGENT step budget generally raises both grounding and overall accuracy until reaching a saturation position.

K Grounding Accuracy (%) Accuracy (%)

2 67.00 68.30 5 71.00 73.67 10 72.00 73.67

- (c) Effect of evidence window size. Larger temporal windows supply richer context for grounding and vision.

Window Grounding Accuracy (%) Accuracy (%)

- 1 71.67 70.33
- 2 78.67 75.00
- 3 81.94 77.26

- (d) Vision model ablation. Stronger VISIONAGENT shows higher overall accuracy, reflecting higher quality extraction of visual information from frames.

###### Vision model Grounding Acc. (%) Acc. (%)

Qwen3-VL-235B 71.00 73.67 GPT-4o 73.30 78.00

#### 4.5 Ablation Studies and Analysis

Execution step limit K. Table 4b varies the upper bound on agent actions per question. Increasing K from 2 to 5 raises temporal localization accuracy from 67.0 to 71.0 ( +4.0) and answer accuracy from 68.30 to 73.67 (+5.37). Raising the limit to K=10 yields no additional gain in answer accuracy (73.67) while localization nudges to 72.0, indicating diminishing returns. We therefore adopt K=5 as the default.

Window context. With GPT-4o as the vision agent, expanding the evidence window from 1 to 2 adjacent clips improves localization from 71.67 to 78.67 (+7.00) and answer accuracy from 70.33 to 75.00 ( +4.67); using 3 clips yields further gains to 81.94 and 77.26 (additional +3.27 and +2.26), as shown in Table 4c. The adjacent context helps disambiguate references across shots, but larger windows require more visual queries and latency while delivering smaller accuracy gains; hence we keep Window= 1 for the default and study larger windows in ablations.

Table 5: Example showing LONGVIDEOAGENT reasoning and repeatedly invoking the GROUNDING and VISION agents to gather evidence and answer correctly.

Question & Choices

Question: What side of the bed is Sheldon when he is closer to the window? Choices: a0: the left side ✓ a1: the right side a2: the middle a3: not on the bed a4: the chair

LONGVIDEOAGENT Execution MASTER AGENT:

<think> Subtitles alone cannot disambiguate left vs. right; I will first ground the segment, then request visual evidence. </think>

<request_grounding> localize the referenced segment </request_grounding> ⇒ GROUNDING AGENT GROUNDING AGENT:

Grounding information. Subtitles (localized): <s10e04_seg02_clip_09> “Sheldon: Okay. Now, on this side, ... Bernadette: Raj knows,”

MASTER AGENT: <think> Grounded subtitles still don’t reveal which side is nearer the window; I’ll request a visual read. </think>

<visual_query> Big Bang Theory season 10 episode 4 — bedroom scene layout (Sheldon & Amy); identify bed/window configuration </visual_query> ⇒ VISION AGENT

###### VISION AGENT:

- Visual description (1). Bedroom scene; bed centered against the wall; Sheldon sits on the bed; general layout and decor described, but the side nearest the window is not specified. MASTER AGENT: <think> The first visual read lacks the window–bed side relation; I need a more precise visual query focused on the window’s position relative to Sheldon. </think> <visual_query> The Big Bang Theory S10E04 seg02 clip 09 — which side of the bed is next to the window, and where is Sheldon relative to it? </visual_query> ⇒ VISION AGENT VISION AGENT:

- Visual description (2). A window with patterned curtains is on the left side of the bed; Sheldon is seated against the headboard near that window. MASTER AGENT: <think> Window is on the left side of the bed and Sheldon is closer to it; therefore he is on the left side. No further tools needed. </think> <answer> a0: the left side ✓ </answer> ⇒ USER

Vision model. Table 4d compares perception backbones. GPT-4o attains 73.30 localization and 78.00 answer accuracy, outperforming Qwen3-VL-235B-a22b at 71.00 and 73.67 by +2.30 and +4.33, respectively. The gap indicates that stronger visual recognition (small objects, OCR, fine attributes) translates into better end-task accuracy in long-form QA, so we adopt GPT-4o as the default vision agent.

Contribution of agentic components. Table 4a decomposes the gains when moving from a single LLM to a multi-agent, multimodal system. Adding temporal grounding to the same backbone increases answer accuracy from 64.3 to 69.0 (+4.7), show-

ing that identifying the relevant clip filters distractors and focuses reasoning. Enabling vision after grounding further lifts accuracy to 74.8 (+5.8 over grounding; +10.5 overall): targeted visual inspection complements subtitles with concrete object/text cues and can validate or refine grounding through repeated calls when uncertain. Because backbones and prompts are held fixed, these improvements are attributable to the agentic procedure. We suggest grounding narrows the context length for reasoning and guides the master agent’s attention, while vision supplies the missing finegrained evidence.

### 5 Conclusion

We presented a multi-agent framework, LONGVIDEOAGENT, for long-form video question answering in which a MASTER agent coordinates a GROUNDINGAGENT for temporal localization and a VISIONAGENT for targeted perception. The framework is model-agnostic: we evaluate it with both closed- and open-source LLMs; for open-source masters, we fine-tune with GRPO to encourage accurate, concise, and cooperation-efficient behavior while keeping the other agents frozen. Equipped with a unified context and GRPO training that combines structural and answer rewards, the system where open-source LLMs act as the master agent yields transparent, step-by-step traces and achieves strong gains on LongTVQA / LongTVQA+ over non-agent baselines. Ablations show that grounding+vision is essential, modest step limits suffice, adjacent-window context helps, and stronger perception yields higher accuracy, validating the effectiveness of the framework. Future work includes richer modalities(like audio track and knowledge background), finer grounding and larger-scale RL training.

### Limitations

Our work has several practical limitations. First, based on TVQA and TVQA+, we rely on provided subtitles as the primary textual channel and do not process raw audio; in future work we plan to integrate an audio-to-subtitles (ASR) module to capture raw speech. Second, the vision and grounding modules are kept fixed during RL. Jointly optimizing them could further improve robustness and accuracy. Lastly, the reward is intentionally simple (format + answer correctness), which may still have room for improvements.

### References

Josh Achiam and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Jean-Baptiste Alayrac and 1 others. 2022. Flamingo: A visual language model for few-shot learning. arXiv preprint arXiv:2204.14198.

Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. 2024. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, and 16 others. 2024. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Preprint, arXiv:2404.16821.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Wenliang Dai, Junnan Li, Dongxu Li, and 1 others. 2023. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. arXiv preprint arXiv:2305.06500.

Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2024. VideoAgent: A memory-augmented multimodal agent for video understanding. arXiv preprint arXiv:2403.11481.

Jingjing Gao, Runzhou Ge, Kai Chen, and Ram Nevatia. 2018. Motion-appearance co-memory networks for video question answering. In CVPR. Dual attention over appearance and motion.

Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Wei Han, Hui Chen, Min-Yen Kan, and Soujanya Poria. 2023. Self-adaptive sampling for efficient video question-answering on image–text models. arXiv preprint arXiv:2307.04192.

Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. 2024a. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700–13710.

Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, and 1 others. 2024b. Videolavit: Unified video-language pre-training with decoupled visual-motional tokenization. arXiv preprint arXiv:2402.03161.

Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. 2021. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7331–7341.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L. Berg.

2018. Tvqa: Localized, compositional video question answering. In EMNLP.

Jie Lei, Licheng Yu, Tamara Berg, and Mohit Bansal. 2020. Tvqa+: Spatio-temporal grounding for video question answering. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 8211–8225.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In ICML.

Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024b. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-llava: Learning

united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. arXiv preprint arXiv:2304.08485. LLaVA.

Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. 2025. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. arXiv preprint arXiv:2508.09736.

Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and 1 others. 2024. Video-rag: Visually-aligned retrieval-augmented long video comprehension. arXiv preprint arXiv:2411.13093.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12585– 12602.

Reiichiro Nakano and 1 others. 2021. Webgpt: Browserassisted question-answering with human feedback. In NeurIPS.

OpenAI. Hello GPT-4o. OpenAI. 2025. Introducing GPT-5.

Junting Pan, Ziyi Lin, Yuying Ge, Xiatian Zhu, Renrui Zhang, Yi Wang, Yu Qiao, and Hongsheng Li. 2023. Retrieving-to-answer: Zero-shot video question answering with frozen large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 272–283.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. 2024. Strengthening multimodal large language model with bootstrapped preference optimization. In European Conference on Computer Vision, pages 382–398. Springer.

Renjie Pi, Kehao Miao, Li Peihang, Runtao Liu, Jiahui Gao, Jipeng Zhang, and Xiaofang Zhou. 2025. Pointing to a llama and call it a camel: On the sycophancy of multimodal large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20177–20191.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Jacob Andreas. 2022. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350. Introduces Self-Ask and search integration.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Alec Radford, Jong Wook Kim, Chris Hallacy, and 1 others. 2021. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020.

David Romero and Thamar Solorio. 2024. Questioninstructed visual descriptions for zero-shot video question answering. arXiv preprint arXiv:2402.10698.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761.

Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. 2024. LongVU: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, and 1 others. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In NeurIPS.

Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, and 1 others. 2024. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232.

Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. In ICCV.

Reuben Tan, Ximeng Sun, Ping Hu, Jui-hsien Wang, Hanieh Deilamsalehy, Bryan A Plummer, Bryan Russell, and Kate Saenko. 2024. Koala: Key frameconditioned long video-llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13581–13591.

Jiaqi Tang, Jianmin Chen, Wei Wei, Xiaogang Xu, Runtao Liu, Xiangyu Wu, Qipeng Xie, Jiafei Wu,

Lei Zhang, and Qifeng Chen. 2025. Robust-r1: Degradation-aware reasoning for robust visual understanding. Preprint, arXiv:2512.17532.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Ke-Yang Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024a. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. ArXiv, abs/2409.12191.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024b. VideoAgent: Long-form video understanding with large language model as agent. arXiv preprint arXiv:2403.10517.

Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, and 1 others. 2025a. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386.

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2025b. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3272–3283.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. 2024. Longvlm: Efficient long video understanding via large language models. In European Conference on Computer Vision, pages 453–470. Springer.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. 2024a. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994.

Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. 2024b. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, and 1 others. 2023. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations.

Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. 2024a. A simple llm framework for long-range video question-answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21715–21737.

Hang Zhang, Xin Li, and Lidong Bing. 2023. Videollama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858.

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024b. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2024c. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and 1 others. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625.

Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, and 1 others. 2025. Apollo: An exploration of video understanding in large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18891–18901.

