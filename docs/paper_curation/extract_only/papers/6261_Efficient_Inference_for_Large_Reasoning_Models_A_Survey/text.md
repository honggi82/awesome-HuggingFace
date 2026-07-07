## Efficient Inference for Large Reasoning Models: A Survey

Yue Liu, Jiaying Wu, Yufei He, Ruihan Gong, Jun Xia, Liang Li, Hongcheng Gao, Hongyu Chen, Baolong Bi, Jiaheng Zhang, Zhiqi Huang, Bryan Hooi, Stan Z. Li, Fellow, IEEE and Keqin Li, Fellow, IEEE

### arXiv:2503.23077v3[cs.CL]13Aug2025

Abstract—Large Reasoning Models (LRMs) significantly improve the reasoning ability of Large Language Models (LLMs) by learning to reason, exhibiting promising performance in solving complex tasks. However, their deliberative reasoning process leads to inefficiencies in token usage, memory consumption, and inference time. Thus, this survey provides a review of efficient inference methods designed specifically for LRMs, focusing on mitigating token inefficiency while preserving the reasoning quality. The overview structure of this paper is shown in Figure 1. First, we introduce a taxonomy to group the recent methods into two main categories: (a) explicit compact Chain-ofThought (CoT), which reduces tokens while keeping the explicit reasoning structure, and (b) implicit latent CoT, which encodes reasoning steps within hidden representations instead of explicit tokens. Meanwhile, we discuss their strengths and weaknesses. Then, we conduct empirical analyses on existing methods from reasoning scenarios, object functions, and performance & efficiency aspects. Besides, we present open challenges in this field, including human-centric controllable reasoning, trade-off between interpretability and efficiency of reasoning, ensuring the safety of efficient reasoning, and broader applications of efficient reasoning. In addition, we highlight key insights for enhancing LRMs’ inference efficiency via techniques such as model merging, new architectures, and agent routers. We hope this work serves as a valuable guide, helping researchers overcome challenges in this vibrant field. A collection of efficient reasoning methods for LRMs (papers and codes) is provided at this link: https://github.com/yueliu1999/Awesome-Efficient-Inference-for-LRMs.

Index Terms—Large Language Models, Large Reasoning Models, Efficient Inference, Model Compression, Token Efficiency

✦

1 INTRODUCTION

# L

ARGE Language Models (LLMs), which are trained to provide quick and intuitive responses, have exhibited

great success in various complex fast-thinking applications like ChatBot [1]. Slow-thinking scenarios like math problemsolving [2] or research [3] increasingly require the models to conduct advanced analytical and deliberative reasoning before providing final responses. To tackle these challenges, Large Reasoning Models (LRMs) such as OpenAI o1/o3 [4], [5], DeepSeek R1 [6], and Kimi k1.5 [7] are developed by guiding the model to learn to effectively reason.

Although effective, the intermediate reasoning process of LRMs is highly resource-intensive, learning to three challenges: (1) significant token consumption, (2) high memory overhead, and (3) increased inference time. Bottlenecks in the safety fine-tuning of vision-language models, as discussed in [8], can severely impact their deployment in critical applications, where model reliability and trustworthiness are paramount. These problems not only increase the inference cost of the service companies but also degrade the experience of the users. Therefore, efficient inference for LRMs has become an urgent and crucial direction.

- • Yue Liu, Jiaying Wu, Yufei He, and R. Gong have equal contributions.
- • Yue Liu, Jiaying Wu, Yufei He, Ruihan Gong, Liang Li, Jiaheng Zhang, and Bryan Hooi are with National University of Singapore.
- • Jun Xia is with HKUST (Guangzhou).
- • Hongcheng Gao, Hongyu Chen, and Baolong Bi are with UCAS.
- • Zhiqi Huang is with Moonshot AI.
- • Stan Z. Li is with Westlake University.
- • Keqin Li is with State University of New York. Manuscript received 13th August, 2025.

Since thinking tokens are treated like regular output tokens without cost differentiation, previous efforts in inference efficiency of regular LLMs, e.g., model compression [9], efficient model design [10], and system-level optimization [11], can alleviate problems (2) and (3). These methods are comprehensively studied [12] and not specially designed for LRMs. Therefore, this survey focuses on the challenge (1): token inefficiency, as shown in Figure 1.

To this end, we conduct a comprehensive survey of recent efficient inference methods designed specifically for LRMs, aiming at improving thinking token efficiency while preserving reasoning quality. Concretely, we first illustrate the research landscape over time as shown in Figure 2, which presents a chronological overview of selected highlycited papers on efficient inference for LRMs from July 2024 to July 2025. This timeline highlights representative works that have had a notable impact in the community, rather than providing an exhaustive or complete list. It serves to contextualize the subsequent discussion by showing when key contributions appeared over the past year.

Subsequently, we present a hierarchical taxonomy that categorizes recent approaches into two classes. As shown in Figure 3, it contains (a) the explicit compact CoT, which reduces the number of thinking tokens while maintaining explicit reasoning structure, and (b) the implicit latent CoT, which encodes reasoning steps within hidden representations instead of explicit tokens. In addition, for the explicit compact CoT, we further summarize three sub-categories: (a.1) CoT compression, (a.2) CoT preference optimization, and (a.3) reward-based CoT conciseness. We analyze the characteristics and discuss their strengths and weaknesses

[Figure 1]

- Figure 1. Overview of this Survey. It mainly consists of four parts: taxonomy, empirical analyses, limitations & challenges, & further improvement.

from the aspects of reasoning quality and efficiency.

Moreover, we conduct a comprehensive empirical study on the existing methods from the perspectives of reasoning scenarios, object functions, and performance & efficiency aspects. Besides, we identify four open challenges regarding the inference efficiency of LRMs, including human-centric controllable reasoning, the trade-off between efficiency and interoperability of reasoning, ensuring the safety of efficient reasoning, and broader applications of efficient LRMs beyond math and code. Lastly, we highlight potential techniques for further improving current methods, including model merging, new architectures, and agent routers.

We hope that this survey helps researchers and engineers further improve efficient inference for LRMs. The main contributions of this paper are summarized as follows.

- • We conduct a comprehensive paper review of current methods of efficient inference for LRMs with a hierarchical taxonomy and strength & weakness discussion.
- • We empirically study recent methods from reasoning scenarios, object functions, and performance & efficiency.
- • We summarize four challenges in this domain from user control, interpretability, safety, and application aspects.
- • We highlight technical insights in further improvement of existing methods from the perspectives of model merging, non-autoregressive architectures, and agent routers

#### 2 BACKGROUND

This section first introduces the background of large reasoning models and then highlights the efficiency challenges in the inference phase of large reasoning models.

##### 2.1 Large Reasoning Model

Large Reasoning Models (LRMs) extend the capabilities of Large Language Models (LLMs) by incorporating explicit intermediate tokens that represent reasoning processes, enabling more structured logical reasoning and effective complex problem-solving. LRMs mimic the way humans approach complex problems by first thinking before providing an answer. When faced with a difficult question, they do not immediately respond with an answer; instead, they analyze the problem, break it down into smaller steps, explore different solution paths, and verify their reasoning before arriving

at a conclusion. This human-like reasoning process of LRMs can also be examined through a cognitive framework, as discussed in [13], which provides insights into the underlying mechanisms shaping model reasoning behaviors. The

- o1 series [4] from OpenAI, released in late 2024, marked a significant breakthrough in AI reasoning capabilities, which integrates reinforcement learning and ”Chain of-Thought” prompting [14] techniques. Following this, OpenAI released
- o3 [5], an upgraded version of o1, allowing it to achieve PhD-level performance in mathematics, science, and programming. Notable DeepSeek’s R1 [6] stands out for being fully open-sourced, with transparent and detailed thinking process tokens, which sets it apart from other proprietary LRMs like o1/o3, where the internal reasoning steps are less accessible. However, since LRMs need to generate numerous intermediate thinking tokens over before arriving at final answers, they are significantly less efficient and more expensive compared to regular LLMs. This added complexity in processing demands significantly more computational resources and time.

2.2 Efficiency Challenge in LRM Inference

A key driver of LRMs’ remarkable reasoning capabilities is the scaling of inference-time compute, which enables complex reasoning through long CoTs [4], [6], [15], [16], [17]. Compared to standard short CoTs [14], which are often shallow, heuristic-driven, and less generalizable [18], long CoTs empower LRMs to tackle complex tasks such as advanced mathematics [19] and medical question answering [20]. However, this shift has also introduced the phenomenon

- of overthinking, where LRMs consume excessive inference tokens and reasoning steps even for simple problems, yielding only marginal performance improvements [21], [22], [23]. In real-world applications such as software engineering agents, overthinking has been found to negatively correlate with issue resolution rates [24]. Moreover, LRMs’ reliance on inference-time scaling exposes them to overthinking attacks, where adversarial actors inject benign yet computationally intensive decoy problems (e.g., Sudoku puzzles) into the context for retrieval-augmented question answering, triggering substantial computational overhead [25].

Toward practical and scalable real-world deployment, optimizing the token efficiency of LRMs without compro-

[Figure 2]

- Figure 2. Chronological Milestones of Efficient Inference for Large Reasoning Models. The time range is mainly from July 2024 to July 2025.

mising overall effectiveness remains an underexplored challenge. This paper presents a comprehensive and systematic investigation into recent advances in token-efficient LRMs, examining their underlying approaches, empirical effectiveness, and implications for future research.

- 3 LANDSCAPE OF EFFICIENT REASONING

and (3) reward-based incentivization. To address the limitations of current reasoning models, Sherlock, as proposed by Ding and Zhang [59], introduces a self-correcting mechanism that enhances the accuracy of vision-language models during inference.

3.1.1 CoT Compression

This section surveys the current landscape of research on token-efficient LRM inference, which can be broadly categorized into two approaches: (1) explicit compact CoT, where explicit instructions, rewards, or budget constraints are introduced to encourage shorter reasoning chains over long CoTs (Section 3.1); and (2) implicit latent CoT, which compresses explicit long CoTs into compact, continuous reasoning states (Section 3.2). The taxonomy of recent efficient inference methods is shown in Table 1, 2 and 3, providing a detailed breakdown of both explicit compact CoT and implicit latent CoT methods in terms of their strategies, training regimes, models, and application domains.

Succinct CoT representations effectively streamline inference while preserving solution quality. The diagram in Figure 4 highlights the core steps of each approach, facilitating clear comparison and comprehensive understanding of the different techniques employed for CoT compression.

Several methods directly constrain the reasoning process to essential steps: Constrained-CoT [27] and CoD [28] confine intermediate reasoning to essential steps, ensuring consistent brevity without losing critical information. Sketchof-Thought (SoT) [26] uses a smaller “router” model to prompt the main LLM to generate sketches of reasoning, offering a concise yet cognitively inspired and structured overview. Fractured Sampling [32] interpolates between full CoT and direct-answer generation by recombining partial reasoning traces, enhancing accuracy-cost efficiency significantly without requiring model retraining. InftyThink [40] decomposes complex tasks into bounded-length segments, creating context-rich intermediate summaries at each step.

##### 3.1 Explicit Compact CoT

Recent research has focused on developing effective methods to create more compact reasoning paths while preserving high accuracy through various techniques, including (1) CoT compression, (2) fine-tuning for compact reasoning,

[Figure 3]

- Figure 3. Taxonomy of Efficient Inference for Large Reasoning Models. The large reasoning model typically outputs long CoT (left sub-figure). The recent efficient inference methods for large reasoning models are mainly classify into (a) explicit compact CoT and (b) implicit latent CoT.

[Figure 4]

- Figure 4. Flowchart of CoT Compression Methods. Each column represents one distinct kind of approach for compressing the CoT reasoning process, highlighting the key steps of each method.

puts without affecting accuracy. SReF [62] suppresses selfaffirming reflections, shortening outputs without degrading accuracy across benchmarks. Adaptive GoGI-Skip [58] combines goal-gradient importance with adaptive skipping, reducing tokens by 45FlashThink [63] uses a verifier-based early-exit strategy, cutting token usage by up to 94.7CTS [64] adjusts reasoning speed in real time by editing internal representations, improving the efficiency-accuracy tradeoff.

Verifier-based and answer-aware methods further improve compression: VeriThinker [65] trains models on auxiliary verification tasks to guide reasoning compression, significantly reducing token usage while preserving or improving accuracy. TrimR [36] uses a verifier-based pruning mechanism to detect and remove redundant reasoning steps during inference, significantly improving test-time efficiency. Answer Convergence [66] applies inference-time early stopping based on convergence of predicted answers, enabling significant token reduction without compromising solution correctness. CTS [67] enhances reasoning efficiency by retaining only essential tokens in chain-of-thought traces, reducing inference cost while maintaining accuracy.

Several methods employ step-level or token-level importance scoring: LIMOPro [68] applies perplexity-based reasoning refinement to prune low-importance steps, enabling more efficient and accurate generation across complex benchmarks. LightThinker [41] introduces special tokens that trigger the model to dynamically compress its ongoing thought process, reducing redundancy. ActivationSteered Compression (ASC) [69] injects a learned activation vector during inference to modulate internal states, enabling concise and math-focused rationales without additional training or accuracy loss. TALE-EP [29] dynamically adjusts the allotted reasoning tokens depending on task

Some approaches dynamically adapt compression at inference time: CoThink [60] uses an instruct model to guide concise solution outlines, improving token efficiency without accuracy loss. ConCISE [55] applies confidenceguided early stopping to compress reasoning chains, reducing output length while maintaining accuracy. ThinkLess [34] introduces early terminators and lightweight output regulation, reducing token overhead without extra training. NoWait [61] eliminates filler tokens using a training-free suppression method, producing concise out-

Table 1 Taxonomy of Explicit Compact CoT Methods (Part I). The criteria mainly contain training, strategy, model, and application.

Types Methods Training Strategy Model Application

SoT [26] ✗ Prompt Qwen-2.5-7B/14B/32B Math, Commonsense, Logic, Scientific, Medical Constrained-CoT [27] ✗ Prompt LLaMA-2-70B, Falcon-40B Math

CoD [28] ✗ Prompt GPT-4o, Claude 3.5 Sonnet Math, Commonsense, Symbolic Reasoning TALE-EP [29] ✗ Prompt LLaMA-3.1-8B-Instruct Math

Meta-Reasoner [30] ✗ Prompt GPT-4o, GPT-4o-mini, Gemini-Exp-1206 Math, Scientific

TS [31] ✗ Intervention Qwen-2.5-7B/14B/32B Math Fractured Sampling [32] ✗ Inference-time Scaling DeepSeek-R1/Qwen-1.5B/7B/14B Math, Scientific, Logic

RPC [33] ✗ KV Cache Compression QwQ-32B/DeepSeek-R1-Distill-Qwen-7B Math, Code, Instruction ThinkLess [34] ✗ Prompt Qwen-2.5-7B/14B, LLaMA3.1-8B Math, Commonsense, Logic, Scientific

DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, DeepSeek-R1-Distill-LLaMA-70B, OpenAI o4-mini

PLAN-AND-BUDGET [35] ✗ Prompt

Math, Instruction, Planning

Pangu Pro MoE, Pangu-R-38B, QwQ-32B, DeepSeek-R1-Distill-Qwen-32B

TrimR [36] ✗ Prompt

Math, Scientific

SOLAR [37] ✓ SFT Qwen2VL-7B-Instruct Math

C3oT [38] ✓ SFT LLaMA-2-Chat -7B & -13B Math, Commonsense TokenSkip [39] ✓ SFT LLaMA-3.1-8B-Instruct, Qwen2.5- 14B-Instruct Math InftyThink [40] ✓ SFT

Math, Scientific LightThinker [41] ✓ SFT

Qwen2.5-14B/32B, Qwen2.5-Math-1.5B/7B, LLaMA-3.1-8B

Language Understanding, Math, Scientific, Commonsense, Logic

DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-LLaMA-8B

Explicit Compact CoT

QwQ-32B-Preview, DeepSeek-R1-Distill-LLaMA-8B, LLaMA-3.1-8B, LLaMA-3.2-1B, Qwen32B-Instruct

CoT-Valve [42] ✓ SFT

Math

Distill System 2 [43] ✓ SFT LLaMA-2-70B-chat Math, Commonsense, Coin Flip SF [44] ✓ SFT

LLaMA-3.2-3B, Gemma2-2B , Qwen2.5-3B , Qwen2.5-Math-1.5B, DeepSeekMath-7B

Math Skip Steps [45] ✓ SFT LLaMA2-7b, Phi-3-mini Math, Logic DAST [46] ✓ SimPO

DS-R1-Distill-Qwen-7B, DS-R1-Distill-Qwen-32B

Math

TALE-PT [29] ✓ SFT, DPO LLaMA-3.1-8B-Instruct Math Kimi k1.5 [7] ✓ RL Kimi k1.5 Multimodal Understanding, Math, Code

O1-Pruner [47] ✓ RL Marco-o1-tB, QwQ-32B Math MRT [48] ✓ RL DeepSeek-R1-Distill-Qwen-32B Math ERL [49] ✓ RL DS-R1-Distill-Qwen-1.5B, DS-R1-Distill-Qwen-7B Math

Claude 3.7 [50] ✓ RL Unknown Math, Code, Agent

L1 [51] ✓ RL Qwen-Distilled-R1-1.5B Language Understanding, Logic, Math SPIRIT [52] ✓ RL LLaMA3-8B-Instruct, Qwen2.5- 7B-Instruct Math

IBPO [53] ✓ RL LLaMA-3.1-8B Math LS-Mixture SFT [54] ✓ SFT Qwen2.5-32B-Instruct Math

ConCISE [55] ✓ SFT, SimPO DeepSeek-R1-Distill-Qwen-7B/1.5B Math, Reasoning Elastic Reasoning [56] ✓ RL E1-Math-1.5B/E1-Code-14B Math, Code

S-GRPO [57] ✓ RL Qwen3-8B/14B, DeepSeek-R1-Distill-Qwen-7B/14B Math, Scientific

TLDR [31] ✓ RL Qwen-2.5-7B/14B/32B Math Adaptive GoGI-Skip [58] ✓ SFT Gemma3-1B/4B/12B, Qwen2.5-3B/7B Math

complexity. Meta-Reasoner [30] applies a contextual multiarmed bandit to optimize efficiency. SelfBudgeter [70] adaptively estimates token budgets based on problem complexity and enforces budget adherence during reasoning, reducing output length without sacrificing accuracy.

Memory and representation-level pruning also offer notable benefits: RPC [33] compresses reasoning paths by periodically pruning the KV cache based on inherent semantic sparsity, achieving up to 4× memory reduction and 1.6× speedup. Prune-on-Logic [71] constructs logic graphs from Long Chain-of-Thought (Long-CoT) traces and selectively prunes low-utility reasoning steps under well-defined semantic constraints, enabling more efficient and accurate inference in resource-limited small language models.

Other methods optimize reasoning strategies: Dynamic Thinking [56] reduces overthinking and improves efficiency by segment-level pruning and preference-based learning. Causal [72] prunes redundant steps in CoT reasoning using probabilistic causal processes, enhancing efficiency without losing accuracy. DRP [73] achieves token efficiency gains by combining pruning with skill-aware decomposition and distillation, without accuracy loss. ReCUT [74] balances reasoning depth and brevity using long-short switched sampling

and parameter interpolation, with minimal performance degradation. R1-Compress [75] reduces token usage via a two-stage chunk-level compression strategy, preserving coherence. A*-Thought [76] compresses reasoning chains using bidirectional A* search guided by token-level importance, improving the accuracy-efficiency tradeoff.

3.1.2 Fine-Tuning on Compact Reasoning Chains

As shown in Figure 5, fine-tuning on compact reasoning data enables LRMs to internalize efficient inference behaviors while keeping performance across diverse tasks.

Several methods generate or use condensed versions of chain-of-thought (CoT) reasoning data: C3oT [38] leverages an LLM to generate condensed versions of long CoTs, preserving essential structure before jointly training models on both full and compressed chains. Skip Steps [45] curates expert-validated answers with condensed steps and finetunes LLMs to mimic these concise reasoning paths. SOLAR [37] fine-tunes LLMs using datasets annotated for both correctness and the effectiveness of the underlying taskspecific reasoning topology, encouraging minimal yet truly complete logic flows with consistent performance.

[Figure 5]

- Figure 5. Flowchart of Fine-Tuning on Compact Reasoning Chains. Each column represents one kind of strategy of SFT for token efficiency.

To prune redundancy in reasoning, some works focus on rationale reduction: VARR [77] proposes a sentencelevel rationale reduction framework guided by verbosity likelihood to prune redundant reasoning steps, significantly improving efficiency while preserving accuracy on arithmetic and commonsense tasks. TokenSkip [39] prunes reasoning chains token-by-token based on importance, followed by fine-tuning across various compression ratios to balance brevity and precision. SmartThinker [78] employs a two-stage framework that combines supervised fine-tuning and reinforcement learning with step-level importanceaware compression, selectively preserving essential reasoning steps while removing redundant ones.

From the perspective of controlling token usage during inference or fine-tuning: TALE-EP [29] enhances tokenbudget awareness via SFT and direct preference optimization (DPO). Elastic Reasoning [56] separates the reasoning process into thinking and solution phases with explicit token budgets, enabling efficient CoT generation under strict inference-time constraints. CoT-Valve [42] discovers a latent direction that controls reasoning length, enabling models to flexibly adjust their level of detail based on task demands.

Some works avoid fine-tuning and use lightweight or prompt-based approaches: PREMISE [79] introduces a prompt-only framework for multi-objective optimization, balancing brevity and correctness to reduce token usage without fine-tuning. L2 [80] combines high-quality English samples with multilingual CoTs and a lightweight decoding intervention, achieving long reasoning with reduced token cost. EfficientXLang [81] shows that reasoning in nonEnglish languages can reduce token consumption without performance loss, offering a promising multilingual strategy. ConciseHint [82] injects concise, task-adaptive hints during generation, reducing token usage while maintaining accuracy on multiple benchmarks. Budget Guidance [83] uses a lightweight controller to adjust reasoning length during inference, achieving controlled token usage with maintained or improved accuracy.

Finally, several methods explore mixing long and short CoT data during training: LS-Mixture SFT [54] fine-tunes models on a mixture of long and short chain-of-thought data, promoting efficient reasoning while reducing unnecessary overthinking. TLDR [84] proposes a dynamic reweighting strategy for mixing short and long chain-ofthought data during training, enabling models to generalize across diverse reasoning lengths and achieving substantial compression ( 40%) without compromising overall performance on math reasoning benchmarks.

3.1.3 Reward-Based Incentivization

A growing body of work introduces explicit reward signals to effectively reduce unnecessary CoT complexity while preserving high accuracy across diverse tasks. However, recent studies on LLM-based preference evaluation [85] have highlighted inherent biases in automatic preference scoring, which may also affect the reliability of CoT-length optimization objectives.

[Figure 6]

Figure 6. Flowchart of Reward-Based Incentivization. Each column represents one distinct kind of approach for incentivizing the token efficiency, highlighting the key steps of each method.

The methods for Reward-Based Incentivization are illustrated in Figure 6. The flowchart highlights how different strategies, such as length-based rewards, harmonizing penalties, and reinforcement learning (RL) techniques, contribute to improving token efficiency in reasoning.

Several works introduce length-based or harmonizing reward mechanisms: Kimi k1.5 [7] integrates lengthbased rewards to discourage verbose reasoning. O1-Pruner [47] detects ”length disharmony” and applies harmonizing penalties that promote brevity without sacrificing solution quality. TLDR [31] combines temperature scaling with length-regularized reinforcement learning to improve token efficiency in small language models without compromising reasoning accuracy on math benchmarks. Arora et al. [49] use reinforcement learning to train models that dynamically allocate computational resources based on task difficulty, balancing cost and precision. DAST [46] proposes a Token

Taxonomy of Explicit Compact CoT Methods (Part II). The criteria mainly contain training, strategy, model, and application.

Types Methods Training Strategy Model Application

SelfBudgeter [70] ✓ RL DeepSeek-R1-Distill-Qwen-1.5B Math

Long Short [86] ✓ SFT, RL Qwen2.5-7B, Llama3.1-8B Math, Logical Length-Aware Optimization [87] ✓ RL Qwen-2.5-7B Math, Logic

DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen7B

Prune-on-Logic [71] ✓ SFT

Math, Logic DRP [73] ✓ SFT DeepSeek-R1-Distill-Qwen-7B/1.5B Math

Qwen2.5, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3

FlashThink [63] ✓ Prompt, SFT

Math, Reasoning AnytimeReasoner [88] ✓ RL DeepSeek-R1-Distill-Qwen-1.5B Math

DeepSeek-R1-Distill-Qwen-7B/14B, DeepSeek-R1-Distill-Llama-8B

VeriThinker [65] ✓ SVFT

Math, Reasoning

LASER [89] ✓ RL DeepSeek-R1-Distill-Qwen-1.5B/7B/32B Math, Reasoning, Code R1-Compress [75] ✓ SFT Qwen2.5-14B/32B-Instruct Math, Logic, Scientific

DeepSeek-R1-Distill-Qwen-1.5B/7B, DeepSeek-R1-Distill-Llama-8B

ACPO [90] ✓ SFT, RL

Math ConciseRL [91] ✓ RL

DeepSeek-R1-Distill-Qwen-1.5B, STILL-3-1.5B-preview

Math, Commonsense

CTS [67] ✓ STF Qwen2.5-7B/14B Math PIR [68] ✓ SFT Qwen-2.5-32B Math, Science

ConciseR [92] ✓ RL Qwen2.5-Math-7B Math CoThink [60] ✓ SFT, RL, Distillation

Qwen2.5-Instruct-32B, DAPO, DeepSeek-R1-Distill, QwQ

Math

DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, Llama - 3.3 - 70B - Instruct

DTO [93] ✓ SimPO

Math, Reasoning

QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, s1.1-32B

A*-Thought [76] ✓ SFT

Math

TLDR [84] ✓ SFT, RL DeepSeek-R1-Distill-7B/14B Math Answer Convergence [66] ✓ Inference-time Qwen-32B, Qwen-7B, Llama-8B/70B, QwQ-32B Math

Explicit Compact CoT

REO-RL [94] ✓ RL DeepSeek-R1-Distill-Qwen, Qwen3 Math

Overclocking LLM Reasoning [95] ✓ Intervention DeepSeek-R1-LLaMA-8B/Qwen-32B Math BINGO [96] ✓ SFT, RL Qwen-1.5B / Qwen-7B / Qwen2.5-Math-7B Math Brevity [97] ✓ Prompt GPT-3.5, Llama-2/3, Gemma, Mistral, Phi-3, Falcon, Vicuna

Commonsense, Logic, Scientific, Language Understanding, Instruction NoWait [61] ✓ Inference-Time Filtering Qwen-3-32B, Phi4, QwQ, Kimi-VL, QvQ

Math, Logic, Scientific, Commonsense, Code, Multimodal Causal [72] ✓ SFT, RL Llama-3.2-1B-Instruct/Qwen-1.5B Math, Commonsense

PREMISE [79] ✓ Prompt Claude-3.7-Sonnet / GPT o1 / Gemini-2.5 Math Budget Guidance [83] ✓ Inference-Time Guidance DeepSeek-R1-Distill-Qwen-7B/32B, Qwen3-8B Math, Logic, Scientific, Code

ReCUT [74] ✓ SFT, RL Llama-3.1-8B-Instruct/Qwen2.5-7B-Instruct Math PLP [98] ✓ SFT, RL DeepSeek-R1-Distill-Qwen-1.5B/7B, Qwen2.5-7B-Instruct Math

SReF [62] ✓ SFT, RL R1-Distill-Qwen-1.5B/7B/32B, QwQ-32B, Qwen3-32B Math LC-R1 [99] ✓ SFT, RL DeepSeek-R1-Distill-Qwen-1.5B/7B Math, Code CoLE [100] ✓ SFT, RL Llama-3.2-1B-Instruct/Qwen-1.5B Math

ConciseHint [82] ✓ SFT, RL DeepSeek-R1/Qwen3-1.7B/4B/8B Math, Science AdapThink [101] ✓ SFT, RL DeepSeek-R1-Distill-Qwen-1.5B Math

L2 [80] ✓ SFT, Decoding Intervention Qwen2.5-32B Math, Science DuP-PO [102] ✓ RL DeepSeek-R1-Distill-Qwen-1.5B Math

AALC [103] ✓ RL Qwen2.5-Math-7B, DeepSeek-R1-Distill-Qwen-7B Math EfficientXLang [81] ✗ Prompt DEEPSEEK R1, QWEN 2.5, QWEN 3 Math

ASC [69] ✓ Inference-time DeepSeek-R1-Distill-LLaMA-8B, Qwen-7B, QwQ-32B Math SmartThinker [78] ✓ SFT, RL DeepSeek-R1-Distill-Qwen-1.5B / 7B Math, Reasoning

DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, Qwen3-8B

CTS [64] ✓ None (Plug-and-play)

Math, Science, Code VARR [77] ✓ SFT Mistral-7B/Llama-3.2-1B-3B Math, Commonsense

Length Budget metric that aligns task complexity with output length, encouraging efficiency through targeted penalties and rewards. PLP [98] introduces a reward-modulated reinforcement learning framework that adaptively penalizes output length based on task difficulty, enabling more concise responses for simple tasks while preserving depth on challenging high-complexity reasoning tasks.

While length penalties are widely used to encourage brevity, [104] reveals that LLM-based preference evaluations can exhibit a systematic length bias, favoring unnecessarily long responses in pairwise comparisons. This bias implies that naive length penalties or rewards must be carefully designed to avoid counteracting model alignment goals.

Other methods refine reward structures using tokenlevel semantics or inefficiency suppression: DuP-PO [102] introduces Dual-Policy Preference Optimization, a reinforcement learning strategy that suppresses inefficient ”thinking tokens” (e.g., wait, however), improving both accuracy and token efficiency in math-focused LLMs. S-GRPO [57] applies a decaying-reward reinforcement learning strategy to encourage early exits in reasoning chains, reducing to-

ken usage by up to 61% while improving accuracy across math and science tasks. BINGO [96] introduces dynamic significance-aware reward signals for CoT length optimization under an RL framework, enhancing token efficiency without compromising performance. IBPO [53] adopts a constrained RL framework to control the distribution of reasoning across response groups based on inference cost. AdapThink [101] applies confidence-aware and diversitysensitive reinforcement learning to dynamically regulate reflection and reasoning depth, improving both efficiency and accuracy in complex reasoning tasks.

Some works propose novel training metrics or frameworks: REO-RL [94] defines a Reasoning Efficiency Gap (REG) metric and trains models via reinforcement learning to close this gap under token constraints, achieving improved efficiency-accuracy tradeoffs. CoLE [100] integrates Efficiency Steering and Self-Rewarded Efficiency RL to guide large reasoning models toward shorter solution paths by leveraging their intrinsic reasoning structure. MRT [48] applies meta-reinforcement learning to balance exploration of novel reasoning paths with the exploitation of

###### A Taxonomy of Efficient Inference Methods for Large Reasoning Models. The criteria mainly contain training, strategy, model, and application.

###### Types Methods Training Strategy Model Application

Soft Thinking [105] ✗ Decoding Qwen-32B/70B, LLaMA-70B Math, Code ICoT-KD [106] ✓ SFT GPT-2 Small/Medium Math

CODI [107] ✓ SFT GPT-2 Small, LLaMA-3.2-1B Math

ICoT-SI [108] ✓ SFT GPT-2 Small/Medium, Phi-3 3.8B, Mistral 7B Math COCONUT [109] ✓ SFT GPT-2 Math

Implicit Latent CoT

CCoT [110] ✓ SFT LLaMA2-7B-Chat Math, Logic

Heima [111] ✓ SFT LLaVA-CoT, LLaMA-3.1-8B-Instruct Multimodal Reasoning Token assorted [112] ✓ SFT LLaMA-3.2-1B, LLaMA-3.2-3B, LLaMA-3.1-8B Agentic Planning, Logic, Math.

SoftCoT [113] ✓ SFT LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct Math, Commonsense,Reasoning CoLaR [114] ✓ SFT, RL Llama-3.2-1B-Instruct/Qwen-1.5B Math Efficient Latent Refinement [115] ✓ Post-training (training-free) LLaMA-3.2-3B / Qwen-2.5-1.5B / GPT-2 Math, Commonsense, Multi-hop DART [116] ✓ SFT Llama-3.2-1B-Instruct/Qwen2.5-1.5B/GPT2 Math

concise, proven ones. Short-RL [87] applies length-aware reinforcement learning to reduce reasoning length by up to 40% without extra training stages, maintaining strong performance on logic and math tasks. LASER and its adaptive variants LASER-D/DE [89] use reinforcement learning with difficulty-aware reward shaping to balance reasoning accuracy and token efficiency through adaptive length control.

Interactive and user-directed length control mechanisms are also emerging: Claude 3.7 [50], the first hybrid reasoning model, introduces an extended thinking mode where users can prescribe token budgets. ACPO [90] integrates dualprocess reasoning and difficulty-aware length budgeting into an RL framework, enabling dynamic cognitive control and efficient token use in complex tasks. L1 [51] generalizes this idea with Length Controlled Policy Optimization (LCPO), enabling fully configurable CoT lengths at inference time. AnytimeReasoner [88] uses budget-relative policy optimization to guide reasoning under variable token limits, enabling adaptive token usage without accuracy degradation. Overclocking LLM Reasoning [95] leverages learned internal progress vectors to monitor and accelerate reasoning phases in real time, improving efficiency and interpretability. Long Short [86] uses a collaborative multi-turn reinforcement learning setup, where specialized LLMs for long and short thoughts jointly compress reasoning chains, reducing token usage while maintaining high accuracy.

Other innovative strategies further improve rewardguided compression: ConciseRL [91] leverages an LLMjudged conciseness reward in a hyperparameter-free RL setting to train models for succinct and accurate reasoning. Brevity [97] analyzes verbosity in LLM responses and proposes prompt engineering techniques to reduce reasoning length, enhancing energy efficiency without sacrificing accuracy. ConciseR [92] adopts a two-stage reinforcement learning approach that first ensures correctness, then compresses reasoning to optimize length without sacrificing performance. LC-R1 [99] combines length- and compressionbased rewards within a GRPO framework to eliminate invalid reasoning patterns, achieving approximately 50% output compression with minimal accuracy loss across diverse reasoning benchmarks. AALC [103] proposes an accuracyaware length reward to guide LLMs toward balancing brevity and correctness, reducing response length by over 50% while maintaining high reasoning accuracy.

3.1.4 Takeaways of Explicit Compact CoT

We distill several important insights from our analysis of Explicit Compact CoT strategies. These takeaways reflect critical aspects of reasoning transparency, dataset constraints, reward optimization, and practical deployment challenges.

Takeaways of Explicit Compact CoT

- • CoT compression enhances scalability but may sacrifice transparency. These techniques lower token usage by abstracting reasoning steps, but risk omitting essential intermediate logic, which can undermine interpretability.
- • Supervised fine-tuning improves efficiency, but at high cost. While effective, these methods depend on curated, condensed datasets and heavy preprocessing, limiting their adaptability to open-ended domains.
- • Reward-based brevity can lead to shallow reasoning. Incentivizing shorter outputs may cause models to favor simplistic answers, at the expense of the deeper reasoning needed for complex tasks.
- • Efficiency alone is insufficient for real-world deployment. Real-world applications require a balance between compactness and reasoning robustness, interpretability, and domain generalization.

##### 3.2 Implicit Latent CoT

Implicit latent CoT methods boost token efficiency by shifting reasoning from explicit tokens to latent tokens, encoding reasoning in hidden layers rather than natural language.

A line of knowledge distillation methods [106], [107], [108] trains student models to infer the teacher’s internal CoT representations rather than mimic explicit token sequences, enabling “vertical” reasoning across transformer layers. Chain of Continuous Thought (COCONUT) [109] replaces token-level reasoning chains with autoregressively generated latent embeddings, which are then fed back into the model to emulate breadth-first search during complext problem-solving. Compressed CoT (CCoT) [110] introduces contemplation tokens—dense, compressed representations of full reasoning chains—significantly reducing inference latency while maintaining high accuracy.

###### Table 4 Benchmarks Used by Explicit Compact CoT Methods.

Table 5 Benchmarks Used by Implicit Latent CoT Methods.

###### Types Methods Application (Benchmarks)

SoT [26] MATH, CommonsenseQA, StrategyQA, ECQA Constrained-CoT [27] GSM8K, AQuA, SVAMP, MathQA

CoD [28] GSM8K, SVAMP, MultiArith, GSM-HARD TALE-EP [29] GSM8K, MATH

Meta-Reasoner [30] Game of 24, TheoremQA, SciBench TS [31] MATH500, AMC, AIME24, OlympiadBench Fractured Sampling [32] MATH500 L5, AIME24, AIME25, AIMO2, GPQA Diamond

RPC [33] DROP, GSM8K, PRM800k, PRM12K ThinkLess [34] GSM8K, MMLU, GPQA, BBH

PLAN-AND-BUDGET [35] GSM8K, DROP, ARC

TrimR [36] MATH500, AIME24, AIME25, GPQA Diamond SOLAR [37] GSM8K, MATH

C3oT [38] GSM8K, MathQA, ECQA, StrategyQA TokenSkip [39] GSM8K, MATH500 InftyThink [40] MATH500, AIME24, GPQA Diamond

LightThinker [41] GSM8K, MMLU, GPQA, BBH

CoT-Valve [42] GSM8K, AIME24, PRM800k, PRM12K Distill System 2 [43]

Last Letter Concatenation, Coin Flip, SycophancyEval, OASST2, MT-Bench, GSM8k SF [44] GSM8K, MATH

Skip Steps [45] Analog of Algebra, Multi-digit Addition, Directional Reasoning

DAST [46] AIME24, AIME25, AMC2023, MinervaMATH, MATH500 TALE-PT [29] GSM8K, MATH Kimi k1.5 [7]

MMStar, MMBench V1.1, MMVet, MathVista, AI2D, HallusionBench

O1-Pruner [47] AIME, AMC, GPQA Diamond MRT [48] AIME2024, AIME2025, AMC2023, MinervaMATH, MATH500 ERL [49] GSM8K, MATH500, AIME2024, CommonsenseQA, Logical Deduction

Claude 3.7 [50] GSM8K, BIG-bench, Coin Flip, MathBench

L1 [51] AIME2025, AMC, MATH, OlympiadBench, GPQA, LSAT, MMLU SPIRIT [52]

Algebra-Linear-1d, Number-Base-Conversion, Diff-Calc, Time-Diff, GSM8K, MetaMathQA IBPO [53] MATH500, AMC, Qsdpo, Asdpo golden

LS-Mixture SFT [54] MATH500, AIME24, GPQA Diamond ConCISE [55] GSM8K, Math-500, AIME24, GPQA Diamond Elastic Reasoning [56] AIME2024, AMC, MATH500, OlympiadBench, Minerva Math S-GRPO [57] GSM8K, AIME2024, AMC2023, MATH-500, GPQA Diamond TLDR [31] MATH500, AMC, AIME24, OlympiadBench Adaptive GoGI-Skip [58] AIME2025, AIME2024, GPQA Diamond, GSM8K SelfBudgeter [70] GSM8K, MATH, AIME2024

Explicit Compact CoT

Long Short [86] MATH500, AIME2024, AIME2025, AMC2023, GPQA Diamond Length-Aware Optimization [87]

Logic-RL dataset, AMC23, AIME2024, MATH500, Minerva Math, Olympiad Bench Prune-on-Logic [71] AMC23, AIME, MATH500, GSM8K, BBH

DRP [73] GSM8K, PRM12K, MATH500, AIME24, AMC23 FlashThink [63] GSM8K, MATH, GPQA Diamond, DROP

AnytimeReasoner [88] AIME2024, AMC2022, MATH500, Minerva Math, OlympiadBench VeriThinker [65] MATH500, AIME2024, AIME2025, GSM8K

LASER [89] MATH500, AIME2024, AMC2023, OlympiadBench, GPQA, MMLU, LSAT R1-Compress [75] MATH500, AIME24, GPQA Diamond ACPO [90] MATH500, AIME2024, GSM8K

ConciseRL [91] GSM8K, MATH500, TheoremQA, GPQA-main, MMLU-Pro-1k CTS [67] MATH500, AIME24, GPQA Diamond PIR [68] AIME, AMC, GPQA Diamond

ConciseR [92] AIME2024, MATH-500, AMC2023, Minerva, OlympiadBench CoThink [60] GSM8K, MATH500, AIME24

DTO [93] GSM8K, MATH500, Gaokao, AMC2023, AIME2024, AIME2025 A*-Thought [76] MATH500, AMC23, OlympiadBench, GSM8K

TLDR [84] GSM8K, MATH, AIME, AMC, ASDiv, Minerva Answer Convergence [66] NQ, GSM8K, MATH-500, GPQA, AIME’24

REO-RL [94] AMC 2023, AIME 2024, AIME 2025, Minerva Math

Overclocking LLM Reasoning [95] GSM8K, Math500 BINGO [96] GSM8K, MATH500, TheoremQA, AIME2024 Brevity [97] DOLLY, GOOAQ, MS-MARCO, NARRATIVEQA, TWEETQA

AMC 2023, AIME 2024, AIME 2025, GPQA-D, MMMU, MMMU-Pro, MathVista, EMMA-mini, MMVU, VSI-Bench Causal [72] GSM8K, MATH-500, AIME, CommonsenseQA

NoWait [61]

PREMISE [79] GSM8K, SVAMP, MATH-500 Budget Guidance [83]

MATH-500, AIME-2024, AMC, OlympiadBench, GPQA, FOLIO, TableBench, LiveCodeBench ReCUT [74] GSM8K, AMC23, AIME24, AIME25, MATH500

PLP [98] GSM8K, MATH500, AIME2024

SReF [62] MATH500, AIME24, AMC23, GSM8K LC-R1 [99] AIME25, MATH500, GSM8K, AMC, Olympiad, GPQA-D, LCB CoLE [100] GSM8K-Aug, GSM-Hard, SVAMP, MultiArith, MATH

ConciseHint [82] GSM8K, AIME24, GPQA-Diamond AdapThink [101] AIME2025, AIME2024, MATH500, AMC

L2 [80] AIME24, AIME25, GPQA-Diamond, MATH500, Graduate Entrance Exam DuP-PO [102] MATH500, OlympiadBench, Minerva, AIME24, AIME25, AMC

AALC [103] GSM8K, MATH, AIME24, AMC24, CNMO24, GPQA EfficientXLang [81] AMC23, MATH500, AIME2024, AIME2025

ASC [69] GSM8K, MATH500 SmartThinker [78]

AIME24, AIME25, AMC23, MinervaMATH, MATH, Olympiad-Bench, TruthfulQA, RACE, Live-Code-Bench

CTS [64] MATH-500, AIME24, AIME25, GPQA Diamond, LiveCodeBench VARR [77] GSM8K, MathQA, TriviaQA, CommonsenseQA, StrategyQA

Heima [111] condenses CoT stages into latent thinking tokens and incorporates an explanatory prompt at the decoder stage to interpret the compressed reasoning. SoftCoT [113] utilizes a small instruction-tuned 1B model to obtain instance-specific latent thought tokens and trains a projection layer to incorporate thought tokens into LLM input. Soft Thinking [105] replaces discrete reasoning tokens with probabilistically weighted concept tokens, enabling reasoning in a continuous concept space without training, and improving both accuracy and token efficiency on math and code tasks. Token-Assorted CoT [112] mixes latent and text tokens, encoding the initial part of the CoT into VAEbased discrete latent tokens while preserving the remainder as natural language, resulting in a hybrid representation that enhances reasoning efficiency. CoLaR [114] dynamically compresses reasoning into latent representations using probabilistic latent prediction and reinforcement learning, enabling variable-speed inference with strong accuracy on

Types Methods Application (Benchmarks)

MATH500, AIME2024, GSM8K, GPQA Diamond, HumanEval, MBPP, LiveCodeBench ICoT-KD [106] BIG-Bench Arithmetic, GSM8K

Soft Thinking [105]

CODI [107] GSM8k, SVAMP, GSM-HARD, MultiArith ICoT-SI [108] Multi-digit Multiplication, GSM8K

COCONUT [109] GSM8k, ProntoQA, ProsQA

CCoT [110] GSM8K Heima [111]

MMStar, MMBench V1.1, MMVet, MathVista, AI2D, HallusionBench

Implicit Latent CoT

Keys-Finding Maze, ProntoQA, ProsQA, MATH, GSM8K, Fresh-Gaokao-Math-2023, DeepMind-Math, College-Math, OlympiaBench-Math, TheoremQA SoftCoT [113]

Token assorted [112]

CommonsenseQA, OpenBookQA, GSM8K, Last Letter Concatenation

CoLaR [114] GSM8K-Aug, GSM-Hard, SVAMP, MultiArith, MATH DART [116] GSM8K-Aug, GSM-Hard, SVAMP, MultiArith

Efficient Latent Refinement [115] GSM8K, MathQA, AQUA-RAT, StrategyQA, ProsQA

mathematical benchmarks. Efficient Latent Refinement [115] proposes a training-free, lightweight post-training method that updates residual embeddings using contrastive feedback, boosting latent-space reasoning accuracy by up to 5% on benchmarks like MathQA without modifying model weights or generating intermediate tokens. DART [116] enables efficient non-autoregressive reasoning by distilling CoT into evolving latent “silent thought” representations via a dual-pathway self-distillation framework.

While their implementations vary, these approaches share a common goal: optimizing inference by internalizing the reasoning process. Empirical results suggest that implicit latent CoT models can match or even surpass explicit CoT methods in reasoning accuracy while significantly reducing generation costs, proving their scalability and efficiency.

Takeaways of Implicit Latent CoT

- • Implicit latent CoT improves efficiency by internalizing reasoning steps but sacrifices interpretability, making verification difficult.
- • Different methods (e.g., knowledge distillation, latent embeddings, contemplation tokens) optimize reasoning at various levels, reducing latency while maintaining accuracy.
- • Future work should focus on extracting humaninterpretable reasoning traces from latent representations to balance efficiency and transparency.

4 EMPIRICAL ANALYSES

4.1 Analyses on Reasoning Scenarios

This section conducts empirical analyses of existing reasoning-efficient methods from the perspectives of both performance and token efficiency. In this subsection, we examine the benchmarks adopted in prior work, focusing on their coverage across diverse reasoning scenarios and their implications for performance evaluation. To provide a structured view, the surveyed benchmarks are categorized into ten representative reasoning scenarios, each reflecting distinct task characteristics and cognitive demands.

• Mathematical Reasoning: This category encompasses datasets from grade-school arithmetic (GSM8K [117],

###### Table 6 Acc. and Token Costs of Explicit Compact CoT Methods.

Table 7 Acc. and Token Costs of Implicit Latent CoT Methods.

###### Types Methods Setting Accuracy Models Token Costs

zero-shot 84.40% GPT-4o 76.40 zero-shot 65.50% Claude 3.5 Sonnet 73.70 few-shot 91.10% GPT-4o 43.90 few-shot 91.40% Claude 3.5 Sonnet 39.80

CoD [28]

zero-shot, prompt 84.46% GPT-4o-mini 77.26

TALE [29]

zero-shot, SFT 74.11% LLaMA-3.1-8B-Instruct 149.93 zero-shot, DPO 78.41% LLaMA-3.1-8B-Instruct 113.41

zero-shot 36.92% LLaMA-2-Chat-7B zero-shot 47.10% LLaMA-2-Chat-13B -

C3oT [38]

- zero-shot, ratio=0.5 86.70% LLaMA-3.1-8B-Instruct 113.05
- zero-shot, ratio=0.6 86.10% LLaMA-3.1-8B-Instruct 198.01
- zero-shot, ratio=0.7 84.30% LLaMA-3.1-8B-Instruct 169.89
- zero-shot, ratio=0.8 82.50% LLaMA-3.1-8B-Instruct 150.12
- zero-shot, ratio=0.9 81.10% LLaMA-3.1-8B-Instruct 129.38 zero-shot, ratio=1.0 78.20% LLaMA-3.1-8B-Instruct 113.05

TokenSkip [39]

zero-shot,tho. 90.14% DeepSeek-R1-Distill-Qwen-7B -

zero-shot,token 87.11% DeepSeek-R1-Distill-Qwen-7B zero-shot,tho. 88.25% DeepSeek-R1-Distill-LLaMA-8B zero-shot,tho. 85.52% DeepSeek-R1-Distill-LLaMA-8B -

LightThinker [41]

SF [44] zero-shot 76.72% DeepSeekMath-7B 184.13 O1-Pruner [47] few-shot 96.50% QwQ-32B 343.00

zero-shot 93.99% DeepSeek-R1 90.91% zero-shot 92.65% QwQ-32B 89.60%

FlashThink [63]

- zero-shot 87.26% R1-Distill-Llama-70B 75.73%
- zero-shot 88.32% R1-Distill-Qwen-32B 76.35%

R1-Distill-Qwen-7B Qwen-2.5-Math-7B

407 zero-shot 96.6%

zero-shot 96.1%

VeriThinker [65]

R1-Distill-Qwen-14B Qwen-2.5-Math-7B

387

zero-shot 89.0% QwQ-32B 342.6 zero-shot 96.6% QwQ-32B 745.2 zero-shot 96.6% QwQ-32B 418.9

FlashThink [63]

zero-shot 86.00% Qwen2.5-7B-Instruct 704 zero-shot 73.90% LLaMA-3.1-8B-Instruct 823

ReCUT [74]

DRP [63] zero-shot, SFT 94.10% DeepSeek-R1-Distill-Qwen-7B 328.00 A*-Thought [76] few-shot 91.20% QwQ-32B 843.69 FlashThink [63]

zero-shot 72.5% DeepSeek-R1-Distill-Qwen-1.5B 16.4 zero-shot 80.9% DeepSeek-R1-Distill-Qwen-1.5B 35.8

zero-shot, GSM-init 76.27% DeepSeek-R1-Distill-Qwen-1.5B 523.77 zero-shot, s1k-init 81.50% DeepSeek-R1-Distill-Qwen-1.5B 662.08

Explicit Compact CoT

SelfBudgeter [70]

zero-shot, CCoT-15 31.5% Falcon-40b 12.1 zero-shot, CCoT-30 27.1% Falcon-40b 13.2 zero-shot, CCoT-45 27.6% Falcon-40b 14.5 zero-shot, CCoT-60 28.2% Falcon-40b 14.9

Constrained-CoT [27]

zero-shot, CCoT-100 27.4% Falcon-40b 15.4 ThinkLess [34]

zero-shot 88.40% Qwen2.5-7B 235.41 zero-shot 92.49% Qwen2.5-14B 235.32 zero-shot 78.92% LLaMA3.1-8B 260.74

zero-shot, topo-tuning 84.00% Qwen2-VL-7B-Instruct zero-shot, topo-rewarding 88.00% Qwen2-VL-7B-Instruct zero-shot, hybrid-scaling 89.02% Qwen2-VL-7B-Instruct -

SOLAR [37]

LLaMA-3.2-3B Gemma-2-2B Qwen2.5-3B Qwen2.5-Math-1.5B DeepSeekMath-7B

SF [44] zero-shot, FS-Self 77.27%

190.03

zero-shot 93.8% DeepSeek-R1-Distill-Qwen-7B 906

- zero-shot 96.2% DeepSeek-R1-Distill-Qwen-14B 724 zero-shot 96.1% Qwen3-8B 1,292
- zero-shot 96.3% Qwen3-14B 952

S-GRPO [57]

zero-shot 80.9% DeepSeek-R1-Distill-Qwen-1.5B 543 zero-shot (Separated) 72.5% DeepSeek-R1-Distill-Qwen-1.5B 248

ConciseRL [91]

DTO [93] zero-shot 83.91% DeepSeek-R1-Distill-Qwen-1.5B 844.18 TLDR [84] zero-shot 87.70% DeepSeek-R1-Distill-Qwen-7B 253

zero-shot, α=100 85.96% DeepSeek-R1-Distill-Qwen-32B ∼240 zero-shot, α=100 39.87% DeepSeek-R1-Distill-LLaMA-8B ∼340

Overclocking [95]

zero-shot 87.32% GPT-4o 71.40 few-shot 92.15% GPT-4o 41.80 zero-shot 79.44% LLaMA-3.1-8B-Instruct 120.50

BINGO [96]

Causal [72] zero-shot, PNS-optimized 99.9% DeepSeek-V3 52.2 PREMISE [79]

zero-shot 95.00% Claude 3.7 Sonnet zero-shot 97.00% OpenAI o1 zero-shot 95.00% Gemini 2.5 Flash -

PLP [98] zero-shot 90.10% DeepSeek-R1-Distill-Qwen-7B 218 ConciseHint [82] AALC [103]

- zero-shot, AdaP+ConciseHint 94.75% Qwen3-4B 839
- zero-shot, AdaP+ConciseHint 95.51% Qwen3-8B 935 zero-shot, AdaP+ConciseHint 93.31% DeepSeek-R1-14B 573

zero-shot 97.59% Qwen2.5-Math-7B 97.01 zero-shot 97.72% DeepSeek-R1-Distill-Qwen-7B 100.58

- zero-shot 88.60% DeepSeek-R1-Distill-Qwen-7B 536
- zero-shot 89.30% DeepSeek-R1-Distill-LLaMA-8B 850

ASC [69] VARR [77] zero-shot 54.98% Mistral 7B 100.38

GSM8K-Zero [118], SVAMP [119], AQuA [120], ASDiv [121]) to advanced high-school and competitionlevel mathematics (MathBench [122], TheoremQA [123], MATH [124], MathQA [125], AIME24 [2], OlympiadBench [126]) and graduate-level STEM reasoning (GPQA [127]). Collectively, they evaluate multi-step quantitative reasoning, linguistic robustness, and problem-solving depth across diverse mathematical domains.

- • Causal Reasoning: Encompasses datasets such as QASC [128] and WorldTree [129], which test the ability to identify and link underlying cause–effect relationships, often through multi-hop scientific reasoning.
- • Code Reasoning: Includes LiveCodeBench [130], Codeforces, and SWE-bench [131], evaluating program synthesis, code understanding, and bug fixing in coding environments with strong practical relevance.
- • Logical Reasoning: Covers ProntoQA [132], LogiQA [133], and ReClor [134], focusing on formal logic, deduc-

Types Methods Setting Accuracy Models Token Costs

ICoT-KD [106] zero-shot 45.00% GPT-2 Medium -

CODI [107] zero-shot 55.60% LLaMA-3.2-1B ICoT-SI [108] zero-shot 51.00% Mistral 7B COCONUT [109] zero-shot 34.10% GPT-2 8.20

Implicit Latent CoT

CCoT [110] zero-shot 31.50% LLaMA2-7B-Chat Token assorted [112] zero-shot 37.20% LLaMA-3.1-8B -

SoftCoT [113] zero-shot 85.81% Qwen2.5-7B-Instruct Efficient Latent Refinement [115] zero-shot 40.20% GPT-2 -

zero-shot 96.81% QwQ-32B 1391 zero-shot 95.83% DeepSeek-R1-Distill-Qwen-32B 785 zero-shot 94.90% DeepSeek-R1-Distill-LLaMA-70B 597

Soft Thinking [105]

tive inference, and reasoning over structured premises.

- • Symbolic Reasoning: CoinFlip [14] measures symbolic manipulation and stepwise logical computation.
- • Commonsense Reasoning: Includes CommonsenseQA [135], OpenBookQA [136], ECQA [137], and StrategyQA [138], assessing real-world plausibility, everyday knowledge, and context-aware implicit fact reasoning.
- • General Reasoning: BIG-Bench [139], BIG-Bench Hard [140], HotPotQA [141], MuSiQue [142], MMLU [143], MMMLU [144], ScienceQA [145], and SciBench [146] jointly measure broad multi-domain reasoning, including complext multi-hop retrieval, factual synthesis, and robust interdisciplinary problem solving.
- • Visual Reasoning: MMMU [144], MATH-Vision [147], and MathVista [148] assess integration of visual perception with textual and mathematical reasoning.
- • Agent Reasoning: TAU-bench [149] and Keys-Finding Maze [112] evaluate autonomous decision-making, planning, and environment interaction capabilities.
- • Task-specific Reasoning: PubMedQA [150] measures biomedical question answering using domain-specific scientific literature, particularly focusing on reasoning.

In addition to categorizing benchmark tasks by reasoning scenario, we further provide a taxonomy of specific benchmark datasets used by the surveyed methods. Tables 4 and 5 summarize which datasets are used by different efficient inference methods, grouped into explicit compact CoT and implicit latent CoT, respectively. This benchmark-level mapping enables a clearer view of method applicability across diverse reasoning settings.

##### 4.2 Analyses on Performance & Efficiency

In this subsection, we present a comprehensive examination of the performance and token consumption associated with a set of existing methods when applied to the widely used GSM8K dataset [117]. The evaluation incorporates multiple methods, multiple models, and multiple experimental settings, ensuring that the comparison reflects a broad spectrum of approaches under diverse configurations.

• GPT-4o is widely regarded for strong performance in complex reasoning and multi-turn problem solving, consistently ranking among the top models on publicly reported benchmarks. Its large parameter scale and extensive training enable high accuracy, but this also leads to increased computational requirements, longer inference times, and higher token usage per query.

Table 8 Analyses on Mathematical Objective Functions in Efficient Reasoning Methods (Part I)

Name Method Objective Function

p(Yt | X, Y<t, Lt ≤ ¯l − t) ∝ p(Yt | X, Y<t) · Pr(Lt ≤ ¯l − t | X, Y<t, Yt) ct = normalize(ut ◦ at) p(Lt | X, Y<t, Yt = vi) = Gamma(log(Lt); λt(vi), αt(vi))

Budget Guidance [83] Inference-Time Guidance

θ∗ = arg minθ (h,p)∈D(fθ(h) − p)2 pˆ = θT h hα = h + αθ θT hα = pˆ+ α∥θ∥2

Overclocking LLM Reasoning [95] Intervention

Meta-Reasoner [30] Prompt st = arg maxs∈S x⊤t θˆs + c x⊤t A−s 1xt AnytimeReasoner [88] RL Janytime(θ, ϕ) = Ex,z mj=1 Pj rϕ(x, z≤bj)

|τi| t=1 min r ˆtiAˆit, clip(ˆrti, 1 − ϵ, 1 + ϵ)Aˆit

N+M i=1

J (θ) = ED,(πn,πr) N+ 1M

i=1 |τi|

−β · DKL[πθ ∥ πref] Aˆit = mit · Ait

 

DuP-PO [102] RL

α, if Ait > 0 and τi ∼ πr β, if Ait < 0 and τi ∼ πn and τi,t ∈ Sthink

mit =

- 0, if Ait > 0 and τi ∼ πn and τi,t ∈ Sthink
- 1, otherwise



πˆX,Y∗

∈ arg max

θ

π

Jˆ∆(π; X, Yθ) := nm1 ni=1 mj=1 [π(yij|xi) · r∆(xi, yij)] s.t. π ∈ Π ∩ Φˆ+(X, Yθ),

IBPO [53] RL

y π(y|x) · 1{y ∈ Ξx} ≥ 1, ∀x ∈ X

LEfficiency(θ, D) = L

max

###### L=1 J(D, θ, L) LREO-RL(θ, D) = Ex∼D Ey∼πθ(·|x) Ni=1+1 ci · r(x, y:Li; θ) dREG(θ, D, Θ)ˆ = L

REO-RL [94] RL

J ˆoptimal(D, Θˆ, L) − J(D, θ, L)

max

L=1

LDART = LCoT + LST + λLdistill LCoT = −N1 Ni=1 log p(zi | Q, z1:i−1; θ) − M1 Mi=1 log p(yi | Q, Z, y1:i−1; θ) LST = −M1 Mi=1 log p(yi | Q, X, y1:i−1; θ, ϕ)

DART [116] SFT

Ldistill = L1 Ll=1 σ(1h˜l) h ˜l − hˆl

 

1

Scorei = PPLprune − PPLretain PPLretain = exp L 1

tj k=1 − log P tokkj | s<j, {toklj}l<k; SLM

pe j=ps

Prune-on-Logic [71] SFT

i



tj k=1 − log P tokkj | s<j \ ni, {toklj}l<k; SLM

pe j=ps

PPLprune = exp L 1

i

Ltotal = LGRPR + Laccuracy r(x, G, θ) = clip (|ω(φ)| · (λo − λl) + I[ω(φ) < 0] · ω(φ) · λb, rmin, rmax)

 

+1 if φ ≤ φlow cos φ−φ

AdapThink [101] SFT, RL

φhigh−φlow · π if φlow < φ < φhigh −1 if φ ≥ φhigh

low

ω(φ) =



|o′i| t=1 min Rt(θ) · Aˆi,t, clip(Rt(θ), 1 − ϵ, 1 + ϵ) · Aˆi,t − β · KL[πθ∥πref]

G i=1

Ltotal = Eq,{oi} G 1

i=1 |o′i|

A ˆi,t = ri,combine + γ · I(o′i,t = </think>) · ri,compress ri,length = 1 − |o

′ i|

LC-R1 [99] SFT, RL

maxj∈C |o′j|

 

′ i)|

1 − |t(o

|t(oi)|, if correct and answer in t(o′i) −1, if correct and answer not in t(o′i) 0, if wrong

ri,compress =



minimize Ex∼D [C(∆x)] subject to Ex∼D [P(∆x)] ≥ α

DTO [93] SimPO

- • LLaMA-3.1 delivers strong competitive reasoning performance across diverse benchmarks while generally offering lower inference cost than larger proprietary systems. Its open-weight availability facilitates reproducibility and experimentation, though performance may be slightly lower in highly specialized reasoning tasks.
- • Claude 3.5 Sonnet provides balanced performance, demonstrating robust results in reasoning, summarization, and long-context processing. It is recognized for maintaining efficiency when handling extended inputs, keeping inference latency and token usage moderate relative to output quality, especially in practice.
- • DeepSeek-R1 targets reasoning-intensive applications and maintains stable performance on tasks requiring structured step-by-step outputs. While its overall computational cost is moderate, token efficiency depends on the complexity of the reasoning chain generated.
- • QwQ-32B achieves strong results for its parameter size,

- with broad coverage of general knowledge and reasoning capabilities. However, it incurs higher per-query computational cost compared to smaller models, making it less optimal for resource-limited deployments.
- • Distilled models such as R1-Distill-LLaMA, DeepSeekR1-Distill-Qwen, and DeepSeek-R1-Distill-LLaMA consistently retain a substantial portion of their teacher models’ reasoning accuracy while significantly lowering latency and reducing token consumption, making them well-suited for environments with constrained compute or strict response-time requirements.
- • Qwen-2.5-Math is oriented toward mathematical and symbolic reasoning tasks, showing consistent accuracy in domain-specific benchmarks. Its outputs tend to be concise, which can improve token efficiency in scenarios where step-by-step elaboration is not essential.
- • Falcon-40B demonstrates strong general-purpose capabilities relative to its scale, effectively addressing a wide

range of reasoning and comprehension tasks; however, it lags behind models extensively fine-tuned for multi-step, complex reasoning in terms of peak accuracy.

The analysis focuses on quantitatively comparing the accuracy achieved by each method alongside the corresponding token costs incurred during inference. All experimental results are systematically organized and reported in Table 6 and Table 7, thereby providing a clear and structured basis for subsequent comparative analyses.

##### 4.3 Analyses on Objective Functions

To complement the proposed categorization of efficient reasoning paradigms, we conduct a systematic analysis of objectives adopted in representative methods. The corresponding mathematical formulations are summarized in Table 8, and Table 9 and Table 10 of the Appendix, spanning from prompting-based strategies to supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), selftraining frameworks (STF), and inference-time intervention.

- • Prompt-based methods influence model outputs without parameter updates by modifying decoding scores through token-level likelihoods, structural constraints, or logical validity checks, especially during inference.
- • SFT-based approaches minimize the cross-entropy loss over curated reasoning trajectories, aligning the model distribution Pθ(y|x) with human-verified solutions.
- • Preference optimization methods, such as Direct Preference Optimization (DPO) and SimPO, extend SFT by introducing pairwise ranking losses. DPO directly optimizes for the likelihood ratio between preferred and dispreferred outputs, while SimPO incorporates similarity constraints to stabilize optimization, encouraging outputs that are both preferred and semantically consistent.
- • RL-oriented approaches define explicit reward functions R(y,x) encoding correctness, stepwise validity, or efficiency constraints. Policy gradient algorithms maximize expected rewards, enabling alignment with nondifferentiable evaluation metrics.
- • Self-Training Frameworks (STF) iteratively generate pseudo-labels for unlabeled data and optimize likelihood-based objectives over these augmented datasets. This reduces reliance on costly annotations while propagating reasoning patterns discovered during inference, effectively enhancing generalization.
- • Inference-time intervention methods preserve model parameters and instead manipulate the decoding process through differentiable scoring terms, constraintsatisfaction formulations, or dynamic search strategies. This allows task-specific adaptation without retraining.

These objectives reflect the optimization principles underpinning the reasoning capabilities of modern LLMs.

#### 5 LIMITATIONS & CHALLENGES

Although the recent efficient reasoning methods have achieved promising performance, there are still several important limitations that hinder their widespread adoption and full effectiveness. To this end, we discuss the limitations and challenges of the existing efficient reasoning methods

[Figure 7]

Figure 7. Limitations and Challenges in Reasoning Efficiency. The image highlights key challenges such as Human-centric Controllable CoT, Reasoning Interpretability, Model Safety, and Broader Application.

from the perspectives of user experience, interpretability, safety, and application, as shown in Figure 7.

- 5.1 User-centric Controllable Reasoning

Recent advancements in LRMs, such as OpenAI’s o3 [5] and Anthropic’s Claude 3.7 [50], have introduced userconfigurable reasoning modes, allowing users to choose whether the model engages in explicit reasoning or provides direct answers. Additionally, these models enable users to control the complexity and length of the reasoning process, adapting to different needs and preferences.

This level of control is especially useful in diverse applications, e.g., in educational settings, users may prefer detailed step-by-step explanations for questions, whereas in real-time decision-making tasks, concise responses are typically more desirable. The ability to allow users to adjust reasoning depth enables LRMs to effectively balance efficiency and transparency, thereby enhancing user experience.

Future research should explore more refined control mechanisms, such as interactive reasoning settings that dynamically adjust based on user feedback. Besides, building personalized reasoning profiles could allow LRMs to learn and adapt to user preferences over time, providing a balance between reasoning depth, speed, and interpretability.

- 5.2 Trade-off Between Interpretability and Efficiency

Compared to LLMs, LRMs offer better interpretability due to their structured reasoning process. By explicitly generating intermediate reasoning steps, LRMs allow users to trace how a conclusion is reached, making them particularly valuable for applications where transparency and verifiability are critical, such as scientific research [151], medical diagnosis [152], and legal decision-making [153]. However, current efficiency-focused LRMs may compromise this interpretability. Many recent methods designed to accelerate LRM inference reduce the number of explicit reasoning

steps or shift reasoning to latent representations, making it harder to understand how a model arrives at its conclusions.

Also, the importance of interpretability varies depending on the application. In domains such as healthcare and legal reasoning, where explanations are essential for accountability and human oversight, explicit reasoning steps are preferred despite their computational cost. Conversely, in real-time decision-making tasks, such as automated trading or robotics, efficiency often takes precedence over transparency, making implicit reasoning more desirable. Hybrid approaches, which dynamically adjust the level of explicit reasoning based on task complexity, offer a potential solution but require further refinement to prevent critical reasoning steps from being lost in the pursuit of efficiency.

To address this trade-off more effectively, future research should focus on developing adaptive inference strategies that optimize the balance between reasoning efficiency and interpretability. One promising direction is the integration of external verification mechanisms, such as symbolic reasoning [154], [155], [156], [157] or retrieval-based justifications [158], which can provide post-hoc explanations for implicit reasoning models. Besides, new empirical studies are needed to systematically quantify how different efficiency techniques impact both model accuracy and human trust, guiding the development of LRMs that are both efficient and interpretable in real-world scenarios.

##### 5.3 Ensuring Safety of Efficient Reasoning

Although the existing methods improve the token efficiency of the LRMs, they may destroy the alignment of LRMs, increasing the potential safety risks, e.g., jailbreaking attacks [159], [160] and privacy leakage [161], [162].

Firstly, the current training-based token-efficient methods either train the LRMs to prefer shorter generations [29], [38] or adopt RL and incentivize concise responses via rulebased reward [7], [47], [48]. Given that the safety alignment is conducted on the original long reasoning generations and the safety of the shorter reasoning generations can not be guaranteed, these training processes might break the safety alignment of the original LRMs.

Secondly, as one piece of evidence, researchers [163] found that the frontier LRMs tend to exploit the loopholes once they get a chance. In addition, although they tried to use another LLM to monitor the intermediate CoT, penalizing their misbehavior can not effectively alleviate this problem but further guide them to deliberately hide their misconduct intent. From this phenomenon, we suspect that the existing token-efficient methods unintentionally guide the LRMs to hide their harmful intent during the process of making their response more concise, significantly increasing the difficulty of safeguarding LRMs.

To address this problem, one promising direction is to strictly enforce safety constraints during the training process, like data filtering for the SFT/DPO data and designing the safety-related reward in RL training. Besides, the failure of current monitors may be due to LRMs’ ability being stronger than LLM-based guard models. Thus, it is worth designing stronger reasoning-based safeguard models [164], [165] to monitor the training data or LRMs.

##### 5.4 Broader Application of Efficient Reasoning

As shown in Table 1, 2, 3, existing LRMs are primarily applied in specialized domains including math [37], [39], [166], code [7], and AI research [3] scenarios.

The first reason is that these tasks have relatively fixed answers, making it easier to construct objectives, e.g., preparing reasoning data, formulating preference loss functions, or rule-based rewards. In contrast, other domains, like social sciences [167], emotional intelligence [168], creative writing [169], typically involve open-ended questions, making it difficult to formulate clear objectives.

The second reason is that these scenarios, like math or research, are not highly time-sensitive, allowing for more computational resources to be allocated for reasoning and optimization. The high computational demand and latency of LRMs constrain their applicability in broader time-sensitive domains, such as robotic manipulations [170], [171], [172], financial trading [173], autonomous driving [174].

However, recently developed efficient reasoning methods [7], [50], [110] help LRMs reduce thinking tokens, optimize timing and memory usage, and thus enhance feasibility in real-time applications. For the open-ended questions, efficient reasoning methods enable LRMs to generate more structured and consistent responses while balancing interpretability and computational cost.

##### 5.5 Takeaways of Limitations Challenges

Despite the advantages of efficient reasoning methods, they also pose several challenges. The following summarizes key limitations and potential directions for addressing them.

Takeaways of Limitations Challenges

- • User-controllable reasoning enables users to adjust reasoning depth, striking a balance between transparency and efficiency while optimizing user experience. Future research should focus on interactive and personalized reasoning for users.
- • Efficient reasoning methods may obscure crucial reasoning processes, compromising the interpretability of LRMs. Future research should develop adaptive inference strategies to balance efficiency and interpretability.
- • Efficient reasoning methods may compromise safety alignment, increasing the risk of jailbreaking and privacy leakage. Future work should integrate safety constraints in training and develop stronger reasoning-based safeguards.
- • Efficient reasoning methods may improve the feasibility of LRMs for broader applications like realtime applications and open-ended tasks.

#### 6 FURTHER IMPROVEMENT

Although current approaches achieve strong performance, we present alternative strategies that could further improve

[Figure 8]

Figure 8. Further Improvement Strategies for Reasoning Efficiency. Directions include new architectures, model merge, and agent router.

inference efficiency while maintaining high reasoning quality, as shown in Figure 8, including new architectures, model merge, and agent router.

##### 6.1 New Architecture

Hybrid Autoregressive and Diffusion Models. The fundamental limitation of autoregressive models is their sequential nature, which makes inference slow, particularly for reasoning tasks that require long chains of intermediate steps. A potential solution is integrating diffusion models into LRMs [10]. Diffusion models generate entire sequences in parallel, allowing for global reasoning structure optimization rather than token-by-token generation. However, the challenge lies in controlling the generated reasoning steps to ensure logical consistency. A promising direction is hybrid architectures that use autoregression for fine-grained control over reasoning while leveraging diffusion-based sampling for efficiency, enabling LRMs to reason in a structured yet accelerated manner. While diffusion offers potential speedups, the overhead of managing this synchronicity and the potential need for multiple iterative refinement to correct logical inconsistencies might offset some gains, especially when compared to optimized autoregressive approaches like speculative decoding, which also aim to accelerate generation without sacrificing as much direct control. The actual trade-off between generation speed, resource consumption (as diffusion models can be computationally intensive to train and sample from), and the quality of reasoning remains an open research question.

Memory-Efficient Transformer Variants. One of the primary inefficiencies in LRMs stems from the quadratic complexity of self-attention. Applying linear attention mechanisms(e.g., RWKV [175]) or state-space models (e.g., Mamba [176]) could drastically reduce memory consumption and improve inference speed. The challenge is that such architectures often struggle with long-range dependencies, which are crucial for reasoning. A key question is whether hybrid models can selectively apply full attention for critical reasoning steps while using approximate attention elsewhere to optimize efficiency. The practicality of such hybrids depends on the effective identification of ”critical” steps and the seamless integration of different attention mechanisms without introducing excessive architectural complexity or training instability. Compared to

methods like quantization or sparse attention, which aim to reduce the cost of full attention, linear attention and statespace models represent a more fundamental architectural shift. The actual benefit will depend on whether the reduced memory and computation translate to tangible improvements in reasoning quality per unit of resource, especially for tasks demanding high fidelity.

##### 6.2 Model Merge

The underlying principle of the existing token-efficient methods can be summarized as integrating the strength of the conventional LLMs, i.e., fast responses and low costs, and the strength of the LRMs, i.e., deliberative reasoning and accurate responses.

The existing training-based methods [29], [47], [48] typically involve reasoning data curation and post-training techniques such as SFT, DPO, or RL, making the process complex and expensive. In contrast, the existing trainingfree methods [30] typically just use promoting engineering to guide the LRMs to save the tokens, limiting the adaptability and effectiveness across diverse reasoning tasks.

To solve this problem, another training-free method model merge [177], [178] becomes a promising technique. Concretely, we can simply merge the model weights of one conventional LLM and the corresponding LRM to take their advantages together [7]. During this process, we provide several key points that need to be solved in the future. First, we need to determine which modules or neurons in models should be merged. Should we merge the neurons in shallow networks or deep networks? Then, we should assign merging weights for the merging units. Should we assign static or dynamic weights for each unit? Third, we should consider how to merge models with different architectures and model sizes, e.g., LLaMA-3.1 Instruct 8B [179] and DeepSeekR1-Distill-Qwen-7B [6].

Merging models with disparate architectures or significantly different layer counts presents a particularly substantial technical hurdle. Simple averaging is often not viable. This may require more sophisticated techniques like parameter subspace alignment, or focusing the merge on specific, architecturally compatible layers (e.g., only attention or feed-forward layers if they share dimensions), which might limit the extent of synergistic merging. Compared to other ensemble methods (like distillation or mixtureof-experts where models are used more distinctly), model merging aims for a deeper integration at the parameter level. However, its practical advantage over simpler ensembling or parameter-efficient fine-tuning (PEFT) methods applied to a single base model needs to be clearly demonstrated, especially concerning the development effort and the consistency of performance gains.

##### 6.3 Agent Router

Agent routing could further improve efficiency by directing different parts of a query to specialized agents. By routing the query to the most appropriate agent based on task complexity, this strategy would optimize resource usage and enable faster inference, particularly for tasks that require domain-specific knowledge or specialized reasoning.

Two routing strategies for LLM inference are router models and confidence-based metrics. Router models (e.g., Routellm [180]) select between stronger or weaker LLMs to balance cost and quality. Confidence-based routing (e.g., Self-REF [181]) directs queries based on LLMs’ confidence in their answers, while uncertainty-based routing (e.g., SLM routing [182]) offloads high-stakes queries to more robust models when confidence is low.

These approaches improve inference efficiency by reducing computation and resource usage while maintaining performance. However, agent routing, though promising, introduces challenges like system complexity, model version management, and operational costs. It only delivers a clear advantage when specialization and accurate routing lead to significant improvements over simpler strategies like multipass inference with a single scalable model.

6.4 Takeaways of Further Improvement We summarize key future directions for further improvement of efficient reasoning methods as follows.

Takeaways of Further Improvement

- • New architectures like autoregressive-diffusion models and memory-efficient transformers hold promise, but managing logical consistency and long-range dependencies remains challenging.
- • Model merging shows promise in combining LLM and LRM strengths, but challenges in module selection, merging weights, and handling architectural differences need further exploration.
- • Agent routing offers potential for efficiency by directing queries to specialized agents, but its practical advantages over simpler strategies and the complexity of maintaining multiple models and routers need to be carefully evaluated.

#### 7 FINAL REMARKS

This survey provides an overview of efficient inference techniques for large reasoning models, highlighting the challenges and recent advancements in this area. As reasoning models continue to grow in scale, the computational cost of inference becomes a major bottleneck, necessitating methods that improve efficiency while maintaining performance. We categorized existing approaches, discussing their trade-offs and practical implications. We hope this survey provides a foundation for further research in this area, encouraging the development of more effective and computationally feasible reasoning models.

#### REFERENCES

- [1] OpenAI, “Introducing chatgpt,” https://openai.com/index/chatgpt/, 2022.
- [2] I. M. Olympiad, “American invitational mathematics examination,” https://artofproblemsolving.com/wiki/index. php/American Invitational Mathematics Examination?srsltid= AfmBOoqo573PtuNmYWTobFVQWyhhDjV2VXowjsIZ0kvmHQ UP Jn2wrG/, 2025.

- [3] OpenAI, “Introducing deep research,” https://openai.com/index/introducing-deep-research/, 2025.
- [4] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney et al., “Openai o1 system card,” arXiv preprint arXiv:2412.16720, 2024.
- [5] OpenAI, “Openai o3-mini system card,” 2025.
- [6] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [7] Kimi Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao et al., “Kimi k1. 5: Scaling reinforcement learning with llms,” arXiv preprint arXiv:2501.12599, 2025.
- [8] Y. Ding, L. Li, B. Cao, and J. Shao, “Rethinking bottlenecks in safety fine-tuning of vision language models,” arXiv preprint arXiv:2501.18533, 2025.
- [9] W. Wang, W. Chen, Y. Luo, Y. Long, Z. Lin, L. Zhang, B. Lin, D. Cai, and X. He, “Model compression and efficient inference for large language models: A survey,” arXiv preprint arXiv:2402.09748, 2024.
- [10] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li, “Large language diffusion models,” arXiv preprint arXiv:2502.09992, 2025.
- [11] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024.
- [12] Z. Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S. Li, Y. Lou, L. Wang, Z. Yuan, X. Li et al., “A survey on efficient inference for large language models,” arXiv preprint arXiv:2404.14294, 2024.
- [13] Z. Hu, J. Lian, Z. Xiao, S. Zhang, T. Wang, N. J. Yuan, X. Xie, and H. Xiong, “Unveiling the learning mind of language models: A cognitive framework and empirical study,” arXiv preprint arXiv:2506.13464, 2025.
- [14] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Proc. of NeurIPS, pp. 24824–24837, 2022.
- [15] Q. Chen, L. Qin, J. Liu, D. Peng, J. Guan, P. Wang, M. Hu, Y. Zhou, T. Gao, and W. Che, “Towards reasoning era: A survey of long chain-of-thought for reasoning large language models,” arXiv preprint arXiv:2503.09567, 2025.
- [16] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Cand`es, and T. Hashimoto, “s1: Simple test-time scaling,” arXiv preprint arXiv:2501.19393, 2025.
- [17] F. Liu, W. Chao, N. Tan, and H. Liu, “Bag of tricks for inference-time computation of llm reasoning,” arXiv preprint arXiv:2502.07191, 2025.
- [18] Z. R. Sprague, F. Yin, J. D. Rodriguez, D. Jiang, M. Wadhwa, P. Singhal, X. Zhao, X. Ye, K. Mahowald, and G. Durrett, “To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning,” in Proc. of ICLR, 2025.
- [19] H. Xu, X. Wu, W. Wang, Z. Li, D. Zheng, B. Chen, Y. Hu, S. Kang, J. Ji, Y. Zhang et al., “Redstar: Does scaling long-cot data unlock better slow-reasoning systems?” arXiv preprint arXiv:2501.11284, 2025.
- [20] Z. Huang, G. Geng, S. Hua, Z. Huang, H. Zou, S. Zhang, P. Liu, and X. Zhang, “O1 replication journey–part 3: Inference-time scaling for medical reasoning,” arXiv preprint arXiv:2501.06458, 2025.
- [21] Y. Ma, Z. Chen, T. Liu, M. Tian, Z. Liu, Z. Liu, and W. Luo, “What are step-level reward models rewarding? counterintuitive findings from mcts-boosted mathematical reasoning,” arXiv preprint arXiv:2412.15904, 2024.
- [22] X. Chen, J. Xu, T. Liang, Z. He, J. Pang, D. Yu, L. Song, Q. Liu, M. Zhou, Z. Zhang et al., “Do not think that much for 2+ 3=? on the overthinking of o1-like llms,” arXiv preprint arXiv:2412.21187, 2024.
- [23] Y. Wu, Y. Wang, T. Du, S. Jegelka, and Y. Wang, “When more is less: Understanding chain-of-thought length in llms,” arXiv preprint arXiv:2502.07266, 2025.
- [24] A. Cuadron, D. Li, W. Ma, X. Wang, Y. Wang, S. Zhuang, S. Liu, L. G. Schroeder, T. Xia, H. Mao et al., “The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks,” arXiv preprint arXiv:2502.08235, 2025.
- [25] A. Kumar, J. Roh, A. Naseh, M. Karpinska, M. Iyyer, A. Houmansadr, and E. Bagdasarian, “Overthink: Slowdown attacks on reasoning llms,” arXiv e-prints, pp. arXiv–2502, 2025.

- [26] S. A. Aytes, J. Baek, and S. J. Hwang, “Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching,” arXiv preprint arXiv:2503.05179, 2025.
- [27] S. Nayab, G. Rossolini, M. Simoni, A. Saracino, G. Buttazzo, N. Manes, and F. Giacomelli, “Concise thoughts: Impact of output length on llm reasoning and cost,” arXiv preprint arXiv:2407.19825, 2024.
- [28] S. Xu, W. Xie, L. Zhao, and P. He, “Chain of draft: Thinking faster by writing less,” arXiv preprint arXiv:2502.18600, 2025.
- [29] T. Han, C. Fang, S. Zhao, S. Ma, Z. Chen, and Z. Wang, “Tokenbudget-aware llm reasoning,” arXiv preprint arXiv:2412.18547, 2024.
- [30] Y. Sui, Y. He, T. Cao, S. Han, and B. Hooi, “Meta-reasoner: Dynamic guidance for optimized inference-time reasoning in large language models,” arXiv preprint arXiv:2502.19918, 2025.
- [31] X. Zhang, Z. Huang, C. Ni, Z. Xiong, J. Chen, and S. Oymak, “Making small language models efficient reasoners: Intervention, supervision, reinforcement,” arXiv preprint arXiv:2505.07961, 2025.
- [32] B. Liao, H. Dong, Y. Xu, D. Sahoo, C. Monz, J. Li, and C. Xiong, “Fractured chain-of-thought reasoning,” arXiv preprint arXiv:2505.12992v2, 2025.
- [33] J. Song, D. Jo, Y. Kim, and J.-J. Kim, “Reasoning path compression: Compressing generation trajectories for efficient llm reasoning,” arXiv preprint arXiv:2505.13866, 2025.
- [34] G. Li, Y. Gao, Y. Li, and Y. Wu, “Thinkless: A training-free inference-efficient method for reducing reasoning redundancy,” arXiv preprint arXiv:2505.15684, 2025.
- [35] J. Lin, X. Zeng, J. Zhu, S. Wang, J. Shun, J. Wu, and D. Zhou, “Plan and budget: Effective and efficient test-time scaling on large language model reasoning,” arXiv preprint arXiv:2505.16122, 2025.
- [36] W. Lin, X. Li, Z. Yang, X. Fu, H.-L. Zhen, Y. Wang, X. Yu, W. Liu, X. Li, and M. Yuan, “Trimr: Verifier-based trainingfree thinking compression for efficient test-time scaling,” arXiv preprint arXiv:2505.17155v2, 2025.
- [37] C. Li, Y. Luo, A. Bolimera, and M. Savvides, “Solar: Scalable optimization of large-scale architecture for reasoning,” arXiv preprint arXiv:2503.04530, 2025.
- [38] Y. Kang, X. Sun, L. Chen, and W. Zou, “C3ot: Generating shorter chain-of-thought without compromising effectiveness,” arXiv preprint arXiv:2412.11664, 2024.
- [39] H. Xia, Y. Li, C. T. Leong, W. Wang, and W. Li, “Tokenskip: Controllable chain-of-thought compression in llms,” arXiv preprint arXiv:2502.12067, 2025.
- [40] Y. Yan, Y. Shen, Y. Liu, J. Jiang, M. Zhang, J. Shao, and Y. Zhuang, “Inftythink: Breaking the length limits of long-context reasoning in large language models,” arXiv preprint arXiv:2503.06692, 2025.
- [41] J. Zhang, Y. Zhu, M. Sun, Y. Luo, S. Qiao, L. Du, D. Zheng, H. Chen, and N. Zhang, “Lightthinker: Thinking step-by-step compression,” arXiv preprint arXiv:2502.15589, 2025.
- [42] X. Ma, G. Wan, R. Yu, G. Fang, and X. Wang, “Cot-valve: Length-compressible chain-of-thought tuning,” arXiv preprint arXiv:2502.09601, 2025.
- [43] P. Yu, J. Xu, J. Weston, and I. Kulikov, “Distilling system 2 into system 1,” arXiv preprint arXiv:2407.06023, 2024.
- [44] T. Munkhbat, N. Ho, S. Kim, Y. Yang, Y. Kim, and S.-Y. Yun, “Self-training elicits concise reasoning in large language models,” arXiv preprint arXiv:2502.20122, 2025.
- [45] T. Liu, Q. Guo, X. Hu, C. Jiayang, Y. Zhang, X. Qiu, and Z. Zhang, “Can language models learn to skip steps?” in Proc. of NeurIPS,

- 2024.

[46] Y. Shen, J. Zhang, J. Huang, S. Shi, W. Zhang, J. Yan, N. Wang, K. Wang, and S. Lian, “Dast: Difficulty-adaptive slow-thinking for large reasoning models,” arXiv preprint arXiv:2503.04472,

- 2025.

- [47] H. Luo, L. Shen, H. He, Y. Wang, S. Liu, W. Li, N. Tan, X. Cao, and D. Tao, “O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning,” arXiv preprint arXiv:2501.12570, 2025.
- [48] Y. Qu, M. Y. Yang, A. Setlur, L. Tunstall, E. E. Beeching, R. Salakhutdinov, and A. Kumar, “Optimizing test-time compute via meta reinforcement fine-tuning,” arXiv preprint arXiv:2503.07572, 2025.
- [49] D. Arora and A. Zanette, “Training language models to reason efficiently,” arXiv preprint arXiv:2502.04463, 2025.
- [50] Anthropic, “Claude 3.7 sonnet and claude code,” https://www. anthropic.com/news/claude-3-7-sonnet, 2025.

- [51] P. Aggarwal and S. Welleck, “L1: Controlling how long a reasoning model thinks with reinforcement learning,” arXiv preprint arXiv:2503.04697, 2025.
- [52] Y. Cui, P. He, J. Zeng, H. Liu, X. Tang, Z. Dai, Y. Han, C. Luo, J. Huang, Z. Li et al., “Stepwise perplexity-guided refinement for efficient chain-of-thought reasoning in large language models,” arXiv preprint arXiv:2502.13260, 2025.
- [53] Z. Yu, T. Xu, D. Jin, K. A. Sankararaman, Y. He, W. Zhou, Z. Zeng, E. Helenowski, C. Zhu, S. Wang et al., “Think smarter not harder: Adaptive reasoning with inference aware optimization,” arXiv preprint arXiv:2501.17974, 2025.
- [54] B. Yu, H. Yuan, H. Li, X. Xu, Y. Wei, B. Wang, W. Qi, and K. Chen, “Long-short chain-of-thought mixture supervised finetuning eliciting efficient reasoning in large language models,” arXiv preprint arXiv:2505.03469, 2025.
- [55] Z. Qiao, Y. Deng, J. Zeng, D. Wang, L. Wei, F. Meng, J. Zhou, J. Ren, and Y. Zhang, “Concise: Confidence-guided compression in step-by-step efficient reasoning,” arXiv preprint arXiv:2505.04881, 2025.
- [56] Y. Xu, H. Dong, L. Wang, D. Sahoo, J. Li, and C. Xiong, “Scalable chain of thoughts via elastic reasoning,” arXiv preprint arXiv:2505.05315, 2025.
- [57] M. Dai, C. Yang, and Q. Si, “S-grpo: Early exit via reinforcement learning in reasoning models,” arXiv preprint arXiv:2505.07686, 2025.
- [58] R. Zhuang, B. Wang, and S. Sun, “Accelerating chain-of-thought reasoning: When goal-gradient importance meets dynamic skipping,” arXiv preprint arXiv:2505.08392v2, 2025.
- [59] Y. Ding and R. Zhang, “Sherlock: Self-correcting reasoning in vision-language models,” arXiv preprint arXiv:2505.22651, 2025.
- [60] S. Fan, P. Han, S. Shang, Y. Wang, and A. Sun, “Cothink: Tokenefficient reasoning via instruct models guiding reasoning models,” arXiv preprint arXiv:2505.22017, 2025.
- [61] C. Wang, Y. Feng, D. Chen, Z. Chu, R. Krishna, and T. Zhou, “Wait, we don’t need to “wait”! removing thinking tokens improves reasoning efficiency,” arXiv preprint arXiv:2506.08343, 2025.
- [62] K. Liu, C. Shen, Z. Zhang, J. Liu, X. Yuan, and J. Ye, “Efficient reasoning through suppression of self-affirmation reflections in large reasoning models,” arXiv preprint arXiv:2507.09879, 2025.
- [63] G. Jiang, G. Quan, Z. Ding, Z. Luo, D. Wang, and Z. Hu, “Flashthink: An early exit method for efficient reasoning,” arXiv preprint arXiv:2505.13949v1, 2025.
- [64] Z. Lin, Z. Fu, Z. Chen, C. Chen, L. Xie, W. Wang, D. Cai, Z. Wang, and J. Ye, “Controlling thinking speed in reasoning models,” arXiv preprint arXiv:2507.03704, 2025.
- [65] Z. Chen, X. Ma, G. Fang, R. Yu, and X. Wang, “Verithinker: Learning to verify makes reasoning model efficient,” arXiv preprint arXiv:2505.17941, 2025.
- [66] X. Liu and L. Wang, “Answer convergence as a signal for early stopping in reasoning,” arXiv preprint arXiv:2506.02536, 2025.
- [67] H. Yuan, B. Yu, H. Li, S. Yang, C. D. Wang, Z. Yu, X. Xu, W. Qi, and K. Chen, “Not all tokens are what you need in thinking,” arXiv preprint arXiv:2505.17827, 2025.
- [68] Y. Xiao, J. Wang, R. Yuan, C. Xu, K. Xu, W. Li, and P. Liu, “Limopro: Reasoning refinement for efficient and effective testtime scaling,” arXiv preprint arXiv:2505.19187, 2025.
- [69] S. Azizi, E. B. Potraghloo, and M. Pedram, “Activation steering for chain-of-thought compression,” arXiv preprint arXiv:2507.04742, 2025.
- [70] Z. Li, Q. Dong, J. Ma, D. Zhang, and Z. Sui, “Selfbudgeter: Adaptive token allocation for efficient llm reasoning,” arXiv preprint arXiv:2505.11274, 2025.
- [71] S. Zhao, J. Yuan, G. Yang, and U. Naseem, “Can pruning improve reasoning? revisiting long-cot compression with capability in mind for better reasoning,” arXiv preprint arXiv:2505.14582, 2025.
- [72] X. Yu, Z. Wang, L. Yang, H. Li, A. Liu, X. Xue, J. Wang, and M. Yang, “Causal sufficiency and necessity improves chain-ofthought reasoning,” arXiv preprint arXiv:2506.09853, 2025.
- [73] Y. Jiang, D. Li, and F. Ferraro, “Drp: Distilled reasoning pruning with skill-aware step decomposition for efficient large reasoning models,” arXiv preprint arXiv:2505.13975, 2025.
- [74] Z. Jin, X. Li, Y. Ji, C. Peng, Z. Liu, Q. Shi, Y. Yan, S. Wang, F. Peng, and G. Yu, “Recut: Balancing reasoning length and accuracy in llms via stepwise trails and preference optimization,” arXiv preprint arXiv:2506.10822, 2025.

- [75] Y. Wang, L. Shen, H. Yao, T. Huang, R. Liu, N. Tan, J. Huang, K. Zhang, and D. Tao, “R1-compress: Long chain-of-thought compression via chunk compression and search,” arXiv preprint arXiv:2505.16838v1, 2025.
- [76] X. Xu, S. Wang, X. Han, Z. Liu, H. Wu, P. Li, Z. Liu, M. Sun, and Z. He, “A*-thought: Efficient reasoning via bidirectional compression for low-resource settings,” arXiv preprint arXiv:2505.24550, 2025.
- [77] J. Jang, J. Kim, W. Kweon, S. Lee, and H. Yu, “Verbosity-aware rationale reduction: Sentence-level rationale reduction for efficient and effective reasoning,” in Proc. of ACL Findings, 2025, pp. 20769–20784.
- [78] X. He, X. Ling, and J. Liu, “Smartthinker: Learning to compress and preserve reasoning by step-level length control,” arXiv preprint arXiv:2507.04348, 2025.
- [79] Y. Yu, Y. Yu, and H. Wang, “Premise: Scalable and strategic prompt optimization for efficient mathematical reasoning in large models,” arXiv preprint arXiv:2506.10716, 2025.
- [80] K. Chen, M. Zhang, and Y. Cao, “Less data less tokens: Multilingual unification learning for efficient test-time reasoning in llms,” arXiv preprint arXiv:2506.18341, 2025.
- [81] S. Ahuja, P. Vaddamanu, and B. Patra, “Efficientxlang: Towards improving token efficiency through cross-lingual reasoning,” 2025.
- [82] S. Tang, X. Ma, G. Fang, and X. Wang, “Concisehint: Boosting efficient reasoning via continuous concise hints during generation,” arXiv preprint arXiv:2506.18810, 2025.
- [83] J. Li, W. Zhao, Y. Zhang, and C. Gan, “Steering llm thinking with budget guidance,” arXiv preprint arXiv:2506.13752, 2025.
- [84] Z.-Z. Li, X. Liang, Z. Tang, L. Ji, P. Wang, H. Xu, X. Wu, H. Huang, W. Deng, Y. Gong, Z. Guo, X. Liu, F. Yin, and C.-L. Liu, “Too long, do re-weighting for efficient llm reasoning compression,” arXiv preprint arXiv:2506.02678, 2025.
- [85] Z. Hu, L. Song, J. Zhang, Z. Xiao, J. Wang, Z. Chen, J. Zhao, and H. Xiong, “Rethinking llm-based preference evaluation,” arXiv e-prints, pp. arXiv–2407, 2024.
- [86] Y. Ning, W. Li, J. Fang, N. Tan, and H. Liu, “Not all thoughts are generated equal: Efficient llm reasoning via multi-turn reinforcement learning,” arXiv preprint arXiv:2505.11827v2, 2025.
- [87] D. Yuan, T. Xie, S. Huang, Z. Gong, H. Zhang, C. Luo, F. Wei, and D. Zhao, “Efficient rl training for reasoning models via lengthaware optimization,” arXiv preprint arXiv:2505.12284, 2025.
- [88] P. Qi, Z. Liu, T. Pang, C. Du, W. S. Lee, and M. Lin, “Optimizing anytime reasoning via budget relative policy optimization,” arXiv preprint arXiv:2505.13438v2, 2025.
- [89] W. Liu, R. Zhou, Y. Deng, Y. Huang, J. Liu, Y. Deng, Y. Zhang, and J. He, “Learn to reason efficiently with adaptive length-based reward shaping,” arXiv preprint arXiv:2505.15612v1, 2025.
- [90] X. Cheng, J. Li, Z. Zhang, X. Tang, W. X. Zhao, X. Kong, and Z. Zhang, “Incentivizing dual process thinking for efficient large language model reasoning,” arXiv preprint arXiv:2505.16315, 2025.
- [91] R.-G. Dumitru, D. Peteleaza, V. Yadav, and L. Pan, “Conciserl: Conciseness-guided reinforcement learning for efficient reasoning models,” arXiv:2505.17250v1 [cs.CL], 2025.
- [92] M. Song and M. Zheng, “Walk before you run! concise llm reasoning via reinforcement learning,” arXiv preprint arXiv:2505.21178, 2025.
- [93] S. An, R. Wang, T. Zhou, and C.-J. Hsieh, “Don’t think longer, think wisely: Optimizing thinking dynamics for large reasoning models,” arXiv preprint arXiv:2505.21765, 2025.
- [94] J. Gao, S. Yan, Q. Tan, L. Yang, S. Xu, W. Fu, Z. Mei, K. Lyu, and Y. Wu, “How far are we from optimal reasoning efficiency?” arXiv preprint arXiv:2506.07104, 2025.
- [95] R. Eisenstadt, I. Zimerman, and L. Wolf, “Overclocking llm reasoning: Monitoring and controlling thinking path lengths in llms,” arXiv preprint arXiv:2506.07240, 2025.
- [96] H. Liu, L. Cao, Y. Ren, M. Zhou, H. Dong, X. Ma, S. Han, and D. Zhang, “Bingo: Boosting efficient reasoning of llms via dynamic and significance-based reinforcement learning,” 2025.
- [97] S. Poddar, P. Koley, J. Misra, S. Podder, N. Balani, N. Ganguly, and S. Ghosh, “Brevity is the soul of sustainability: Characterizing llm response lengths,” arXiv preprint arXiv:2506.08686, 2025.
- [98] Z. Ling, D. Chen, H. Zhang, Y. Jiao, X. Guo, and Y. Cheng, “Fast on the easy, deep on the hard: Efficient reasoning via powered length penalty,” arXiv preprint arXiv:2506.10446, 2025.

- [99] Z. Cheng, D. Chen, M. Fu, and T. Zhou, “Optimizing length compression in large reasoning models,” arXiv preprint arXiv:2506.14755, 2025.
- [100] W. Zhao, J. Guo, Y. Deng, X. Sui, Y. Hu, Y. Zhao, W. Che, B. Qin, T.-S. Chua, and T. Liu, “Exploring and exploiting the inherent efficiency within large reasoning models for self-guided efficiency enhancement,” arXiv preprint arXiv:2506.15647, 2025.
- [101] X. Wan, W. Wang, W. Xu, W. Yin, J. Song, and M. Sun, “Adapthink: Adaptive thinking preferences for reasoning language model,” arXiv preprint arXiv:2506.18237, 2025.
- [102] B. Ding, Y. Chen, F. Wang, L. Ming, and T. Lin, “Do thinking tokens help or trap? towards more efficient large reasoning model,” 2025.
- [103] R. Li, Z. Luo, Q. Zhang, R. Li, B. Zhou, A. Payani, and X. Du, “Aalc: Large language model efficient reasoning via adaptive accuracy-length control,” arXiv preprint arXiv:2506.20160, 2025.
- [104] Z. Hu, L. Song, J. Zhang, Z. Xiao, T. Wang, Z. Chen, N. J. Yuan, J. Lian, K. Ding, and H. Xiong, “Explaining length bias in llmbased preference evaluations,” arXiv preprint arXiv:2407.01085, 2024.
- [105] Z. Zhang, X. He, W. Yan, A. Shen, C. Zhao, S. Wang, Y. Shen, and X. E. Wang, “Soft thinking: Unlocking the reasoning potential of llms in continuous concept space,” arXiv preprint arXiv:2505.15778, 2025.
- [106] Y. Deng, K. Prasad, R. Fernandez, P. Smolensky, V. Chaudhary, and S. Shieber, “Implicit chain of thought reasoning via knowledge distillation,” arXiv preprint arXiv:2311.01460, 2023.
- [107] Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, and Y. He, “Codi: Compressing chain-of-thought into continuous space via selfdistillation,” arXiv preprint arXiv:2502.21074, 2025.
- [108] Y. Deng, Y. Choi, and S. Shieber, “From explicit cot to implicit cot: Learning to internalize cot step by step,” arXiv preprint arXiv:2405.14838, 2024.
- [109] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, and Y. Tian, “Training large language models to reason in a continuous latent space,” arXiv preprint arXiv:2412.06769, 2024.
- [110] J. Cheng and B. Van Durme, “Compressed chain of thought: Efficient reasoning through dense representations,” arXiv preprint arXiv:2412.13171, 2024.
- [111] X. Shen, Y. Wang, X. Shi, Y. Wang, P. Zhao, and J. Gu, “Efficient reasoning with hidden thinking,” arXiv preprint arXiv:2501.19201, 2025.
- [112] D. Su, H. Zhu, Y. Xu, J. Jiao, Y. Tian, and Q. Zheng, “Token assorted: Mixing latent and text tokens for improved language model reasoning,” arXiv preprint arXiv:2502.03275, 2025.
- [113] Y. Xu, X. Guo, Z. Zeng, and C. Miao, “Softcot: Soft chainof-thought for efficient reasoning with llms,” arXiv preprint arXiv:2502.12134, 2025.
- [114] W. Tan, J. Li, J. Ju, Z. Luo, J. Luan, and R. Song, “Think silently, think fast: Dynamic latent compression of llm reasoning chains,” arXiv preprint arXiv:2505.16552v3, 2025.
- [115] X. Wang, D. Wang, W. Ying, H. Bai, N. Gong, S. Dong, K. Liu, and Y. Fu, “Efficient post-training refinement of latent reasoning in large language models,” arXiv preprint arXiv:2506.08552, 2025.
- [116] N. Jiang, Z. Wu, D.-C. Zhan, F. Lai, and S. Lian, “Dart: Distilling autoregressive reasoning to silent thought,” arXiv preprint arXiv:2506.11752, 2025.
- [117] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.
- [118] C.-H. Chiang and H.-y. Lee, “Over-reasoning and redundant calculation of large language models,” in Proc. of EACL, 2024, pp. 161–169.
- [119] A. Patel, S. Bhattamishra, and N. Goyal, “Are NLP models really able to solve simple math word problems?” in Proc. of NAACL, 2021, pp. 2080–2094.
- [120] W. Ling, D. Yogatama, C. Dyer, and P. Blunsom, “Program induction by rationale generation: Learning to solve and explain algebraic word problems,” in Proc. of ACL, 2017, pp. 158–167.
- [121] S.-y. Miao, C.-C. Liang, and K.-Y. Su, “A diverse corpus for evaluating and developing English math word problem solvers,” in Proc. of ACL, 2020, pp. 975–984.
- [122] H. Liu, Z. Zheng, Y. Qiao, H. Duan, Z. Fei, F. Zhou, W. Zhang, S. Zhang, D. Lin, and K. Chen, “MathBench: Evaluating the theory and application proficiency of LLMs with a hierarchical mathematics benchmark,” in Proc. of ACL Findings, 2024.

- [123] W. Chen, M. Yin, M. Ku, P. Lu, Y. Wan, X. Ma, J. Xu, X. Wang, and T. Xia, “TheoremQA: A theorem-driven question answering dataset,” in Proc. of EMNLP, 2023, pp. 7889–7901.
- [124] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, “Measuring mathematical problem solving with the math dataset,” NeurIPS, 2021.
- [125] L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu, “Metamath: Bootstrap your own mathematical questions for large language models,” arXiv preprint arXiv:2309.12284, 2023.
- [126] C. He, R. Luo, Y. Bai, S. Hu, Z. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun, “OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems,” in Proc. of ACL, 2024, pp. 3828–3850.
- [127] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman, “GPQA: A graduate-level googleproof q&a benchmark,” in First Conference on Language Modeling, 2024.
- [128] T. Khot, P. Clark, M. Guerquin, P. A. Jansen, and A. Sabharwal, “QASC: A dataset for question answering via sentence composition,” in AAAI, 2019.
- [129] P. Jansen, E. Wainwright, S. Marmorstein, and C. Morrison, “WorldTree: A corpus of explanation graphs for elementary science questions supporting multi-hop inference,” in Proc. of LREC, 2018.
- [130] N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica, “Livecodebench: Holistic and contamination free evaluation of large language models for code,” arXiv preprint arXiv:2403.07974, 2024.
- [131] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan, “Swe-bench: Can language models resolve realworld github issues?” arXiv preprint arXiv:2310.06770, 2023.
- [132] A. Saparov and H. He, “Language models are greedy reasoners: A systematic formal analysis of chain-of-thought,” in Proc. of ICLR, 2023.
- [133] J. Liu, L. Cui, H. Liu, D. Huang, Y. Wang, and Y. Zhang, “Logiqa: A challenge dataset for machine reading comprehension with logical reasoning,” arXiv preprint arXiv:2007.08124, 2020.
- [134] W. Yu, Z. Jiang, Y. Dong, and J. Feng, “Reclor: A reading comprehension dataset requiring logical reasoning,” in Proc. of ICLR, 2020.
- [135] A. Talmor, J. Herzig, N. Lourie, and J. Berant, “Commonsenseqa: A question answering challenge targeting commonsense knowledge,” in Proc. of NAACL, 2019.
- [136] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor conduct electricity? a new dataset for open book question answering,” in EMNLP, 2018.
- [137] S. Aggarwal, D. Mandowara, V. Agrawal, D. Khandelwal, P. Singla, and D. Garg, “Explanations for CommonsenseQA: New Dataset and Models,” in Proc. of ACL, 2021.
- [138] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant, “Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies,” Transactions of the Association for Computational Linguistics (TACL), 2021.
- [139] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. GarrigaAlonso et al., “Beyond the imitation game: Quantifying and extrapolating the capabilities of language models,” arXiv preprint arXiv:2206.04615, 2022.
- [140] M. Suzgun, N. Scales, N. Sch¨arli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, , and J. Wei, “Challenging big-bench tasks and whether chain-of-thought can solve them,” arXiv preprint arXiv:2210.09261, 2022.
- [141] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop question answering,” 2018. [Online]. Available: https://arxiv.org/abs/1809.09600
- [142] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique: Multihop questions via single-hop question composition,” TACL, pp. 539–554, 2022.
- [143] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” arXiv preprint arXiv:2009.03300, 2020.
- [144] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun et al., “Mmmu: A massive multi-

- discipline multimodal understanding and reasoning benchmark for expert agi,” in Proc. of CVPR, 2024, pp. 9556–9567.
- [145] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, “Learn to explain: Multimodal reasoning via thought chains for science question answering,” in Proc. of NeurIPS, 2022.
- [146] X. Wang, Z. Hu, P. Lu, Y. Zhu, J. Zhang, S. Subramaniam, A. R. Loomba, S. Zhang, Y. Sun, and W. Wang, “Scibench: Evaluating college-level scientific problem-solving abilities of large language models,” in Proc. of ICML, 2024.
- [147] K. Wang, J. Pan, W. Shi, Z. Lu, H. Ren, A. Zhou, M. Zhan, and H. Li, “Measuring multimodal mathematical reasoning with math-vision dataset,” in Proc. of NeurIPS, 2024.
- [148] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” in Proc. of ICLR, 2024.
- [149] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan, “tau-bench: A benchmark for tool-agent-user interaction in real-world domains,” arXiv preprint arXiv:2406.12045, 2024.
- [150] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu, “PubMedQA: A dataset for biomedical research question answering,” in Proc. of EMNLP, 2019, pp. 2567–2577.
- [151] N. L. Rane, A. Tawde, S. P. Choudhary, and J. Rane, “Contribution and performance of chatgpt and other large language models (llm) for scientific and research advancements: a double-edged sword,” International Research Journal of Modernization in Engineering Technology and Science, pp. 875–899, 2023.
- [152] E. Ullah, A. Parwani, M. M. Baig, and R. Singh, “Challenges and barriers of using large language models (llm) such as chatgpt for diagnostic medicine with a focus on digital pathology–a recent scoping review,” Diagnostic pathology, p. 43, 2024.
- [153] I. Cheong, K. Xia, K. K. Feng, Q. Z. Chen, and A. X. Zhang, “(a) i am not a lawyer, but...: engaging legal experts towards responsible llm policies for legal advice,” in Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 2024, pp. 2454–2469.
- [154] T. R. Besold, A. d’Avila Garcez, S. Bader, H. Bowman, P. Domingos, P. Hitzler, K.-U. Kuhnberger,¨ L. C. Lamb, P. M. V. Lima, L. de Penning et al., “Neural-symbolic learning and reasoning: A survey and interpretation 1,” in Neuro-Symbolic Artificial Intelligence: The State of the Art, 2021, pp. 1–51.
- [155] V. Gaur and N. Saunshi, “Reasoning in large language models through symbolic math word problems,” arXiv preprint arXiv:2308.01906, 2023.
- [156] Y. Sui, Y. He, N. Liu, X. He, K. Wang, and B. Hooi, “Fidelis: Faithful reasoning in large language model for knowledge graph question answering,” arXiv preprint arXiv:2405.13873, 2024.
- [157] Y. Sui, Y. He, Z. Ding, and B. Hooi, “Can knowledge graphs make large language models more trustworthy? an empirical study over open-ended question answering,” arXiv preprint arXiv:2410.08085, 2024.
- [158] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang, and H. Wang, “Retrieval-augmented generation for large language models: A survey,” arXiv preprint arXiv:2312.10997, 2023.
- [159] Y. Liu, X. He, M. Xiong, J. Fu, S. Deng, and B. Hooi, “Flipattack: Jailbreak llms via flipping,” arXiv preprint arXiv:2410.02832, 2024.
- [160] Y. He, Y. Li, J. Wu, Y. Sui, Y. Chen, and B. Hooi, “Evaluating the paperclip maximizer: Are rl-based language models more likely to pursue instrumental goals?” arXiv preprint arXiv:2502.12206, 2025.
- [161] H. Li, Y. Chen, J. Luo, J. Wang, H. Peng, Y. Kang, X. Zhang, Q. Hu, C. Chan, Z. Xu et al., “Privacy in large language models: Attacks, defenses and future directions,” arXiv preprint arXiv:2310.10383, 2023.
- [162] C. Wang, Y. Liu, B. Li, D. Zhang, Z. Li, and J. Fang, “Safety in large reasoning models: A survey,” arXiv preprint arXiv:2504.17704, 2025.
- [163] OpenAI, “Detecting misbehavior in frontier reasoning models,” https://openai.com/index/chain-of-thought-monitoring/, 2025.
- [164] Y. Liu, H. Gao, S. Zhai, X. Jun, T. Wu, Z. Xue, Y. Chen, K. Kawaguchi, J. Zhang, and B. Hooi, “Guardreasoner: Towards reasoning-based llm safeguards,” arXiv preprint arXiv:2501.18492, 2025.
- [165] Y. Liu, S. Zhai, M. Du, Y. Chen, T. Cao, H. Gao, C. Wang, X. Li, K. Wang, J. Fang, J. Zhang, and B. Hooi, “Guardreasoner-

- vl: Safeguarding vlms via reinforced reasoning,” arXiv preprint arXiv:2505.11049, 2025.
- [166] R. Gong, Y. Liu, W. Qu, M. Du, Y. He, Y. Ma, Y. Chen, X. Liu, Y. Wen, X. Li et al., “Efficient reasoning via chain of unconscious thought,” arXiv preprint arXiv:2505.19756, 2025.
- [167] S. Thapa, S. Shiwakoti, S. B. Shah, S. Adhikari, H. Veeramani, M. Nasim, and U. Naseem, “Large language models (llm) in computational social science: prospects, current state, and challenges,” Social Network Analysis and Mining, pp. 1–30, 2025.
- [168] S. Wu, Y. Deng, Y. Zhu, W. Hsu, and M. L. Lee, “From personas to talks: Revisiting the impact of personas on llm-synthesized emotional support conversations,” arXiv preprint arXiv:2502.11451, 2025.
- [169] OpenAI, “Introducing gpt-4.5,” https://openai.com/index/introducinggpt-4-5/, 2025.
- [170] Y. Ji, H. Tan, J. Shi, X. Hao, Y. Zhang, H. Zhang, P. Wang, M. Zhao, Y. Mu, P. An et al., “Robobrain: A unified brain model for robotic manipulation from abstract to concrete,” arXiv preprint arXiv:2502.21257, 2025.
- [171] Google, “Gemini robotics brings ai into the physical world,” https://deepmind.google/discover/blog/ gemini-robotics-brings-ai-into-the-physical-world/, 2025.
- [172] Nvidia, “Nvidia isaac gr00t n1: An open foundation model for humanoid robots,” https://research.nvidia.com/publication/2025-03 nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots, 2025.

- [173] H. Ding, Y. Li, J. Wang, and H. Chen, “Large language model agent in financial trading: A survey,” arXiv preprint arXiv:2408.06361, 2024.
- [174] Z. Yang, X. Jia, H. Li, and J. Yan, “Llm4drive: A survey of large language models for autonomous driving,” arXiv preprint arXiv:2311.01043, 2023.
- [175] B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, S. Biderman, H. Cao, X. Cheng, M. Chung, M. Grella et al., “Rwkv: Reinventing rnns for the transformer era,” arXiv preprint arXiv:2305.13048, 2023.
- [176] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.
- [177] E. Yang, L. Shen, G. Guo, X. Wang, X. Cao, J. Zhang, and D. Tao, “Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities,” arXiv preprint arXiv:2408.07666,

- 2024.

[178] H. Wu, Y. Yao, S. Liu, Z. Liu, X. Fu, X. Han, X. Li, H.-L. Zhen, T. Zhong, and M. Yuan, “Unlocking efficient long-to-short llm reasoning with model merging,” arXiv preprint arXiv:2503.20641,

- 2025.

- [179] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. AlDahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.
- [180] I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous, and I. Stoica, “RouteLLM: Learning to route LLMs from preference data,” in Proc. of ICLR, 2025.
- [181] Y.-N. Chuang, H. Zhou, P. K. Sarma, P. Gopalan, J. Boccio, S. Bolouki, and X. Hu, “Learning to route llms with confidence tokens,” arXiv preprint arXiv:2410.13284, 2025.
- [182] Y.-N. Chuang, L. Yu, G. Wang, L. Zhang, Z. Liu, X. Cai, Y. Sui, V. Braverman, and X. Hu, “Confident or seek stronger: Exploring uncertainty-based on-device llm routing from benchmarking to generalization,” arXiv preprint arXiv:2502.04428, 2025.

#### APPENDIX

We analyze mathematical objective functions in efficient reasoning methods in Table 9 and Table 10.

Table 9 Analyses on Mathematical Objective Functions in Efficient Reasoning Methods (Part II)

Name Method Objective Function

Soft Thinking [105] Decoding p(y | x) = t1,...,tm p(ti | ·) · p(y | x, t1:m) NoWait [61] Inference-Time Filtering

CoT = {(chunki, ai)}ni=1 Response = (CoT, summary)

- Kα = {v ∈ Vα | ∃ks ∈ K, s.t. is substr(ks, v)}

Answer Convergence [66] Inference-time

- L = −T1 Tt=1 [pt log pˆt + (1 − pt) log(1 − pˆt)]

pˆt = σ(Wzt + b) yt∗ ← yt∗ + α · max(y) − |y1| i yi

vℓ = N1 Ni=1 hℓ(qi ⊕ si)[−1] − hℓ(qi ⊕ li)[−1]

hℓ(xi) ← hℓ(xi) + γvℓ ∀i ∈ [1, decoding steps] KL(softmax(z)∥softmax(˜z)) ≤ ε γmax = max 0, 1 − Lγ

ASC [69] Inference-time

raw

4a γraw Fractured Sampling [32] Inference-time Scaling pseg = 1 − Ht=1(1 − pt)m

TS [31] Intervention r = rˆ− ζ(L) RPC [33] KV Cache Compression Importance(t) = 2w1+1 · RH1 wi=−w r,h Attnℓh(qr, kt+i) CTS [64] None (Plug-and-play)

hl ← hl + α · vl d(xt) = |L1| l∈L JSD(pN(·|x<t), pl(·|x<t)) threshold = µW + λ · σW Efficient Latent Refinement [115] Post-training (training-free)

Residual Refinement: ht = α · ht−1 + (1 − α) · f(ht−1)

Contrastive Update: hupdatedt = ht + η · ∇ht MSE(ht, hgoodt ) − MSE(ht, hbadt ) Constrained-CoT [27] Prompt N1 Ni=1 1(Γ(ˆyi), yi) × p(ˆyi)

n−c(xi,yi) k n k

Pass@k(l, n) = m1 mi=1 1 −

EfficientXLang [81] Prompt

i) ⊮ [LLM(xi, r) = yi ∧ LID(r) = l]

c(xi, yi) = r∈R(n)(x

Ltotal = λ · ∇textLacc + (1 − λ) · ∇textLlen, λ ∈ [0, 1]

∗(q) L(r)

IO(r, q) = L(r)−L

PREMISE [79] Prompt

∗(r,q) L(r)

IU(r, q) = 1 − k

SoT [26] Prompt T(li, lo, B) = t˜PB(li) + lki=+llo−1

tDB(k)

i

ThinkLess [34] Prompt p(x1:(M+N) | q) = Mi=1 p(xreasoni ) Nj=1 p(xanswerj ) TrimR [36] Prompt minc(·) Infer Cost(y<t′) s.t. Perf(X, y<t′) ≥ Perf(X, y<t)

CTS [67] STF miny˜ dist(A, A˜) + λ∥y˜∥0 DAST [46] SimPO

 

max −0.5 · L(yL)−Lbudget

###### + 0.5, 0.1 , if S(y) = 1 min 0.9 · L(yL)−Lbudget

budget



− 0.1, −0.1 , if S(y) = 0

budget

λc · ris(y), if correct λwis · [ris(y) − 1] + min(0, rs(y) − λws ), if incorrect

BINGO [96] SFT, RL RBINGO(y) =

JBINGO(θ) = Et min rt(θ)Aˆt, clip(rt(θ), 1 − ϵ, 1 + ϵ)Aˆt

PNS(S, st, q) := P(AS = y, AS′ ̸= y) PS(S, q) = P(Ado(S) = y | A ̸= y, S, q¯ ) PN(S, st, q) = P(Ado(s<t,s¯t,s′

Causal [72] SFT, RL

>t) ̸= y | A = y, S, q)

Ti

Ti

N

N

1 N

1 N

LCE(θ) = −

log P(yi,t | yi,<t, xi)

LCE(θ) = −

log P(yi,t | yi,<t, xi)

t=1

t=1

i=1

i=1

TALE-PT [29] SFT, DPO

N

N

1 N

1 N

log Pθ(yi ≻ yi′)

log Pθ(yi ≻ yi′)

LDPO(θ) = −

LDPO(θ) = −

i=1

i=1

yt∗ = arg max r(Y[lt]), r(Y[st]) LDPO(D) = −E(q,Y +,Y −)∼D log σ β log M(Y

ReCUT [74] SFT, RL

−|q)

+|q)

Mref(Y +|q) − β log M(Y

Mref(Y −|q) θmerge = θacc + α · Topx(θlen)

0, if Ptoken < θ and token ∈ {”wait”, ”Wait”} Ptoken, otherwise

SReF [62] SFT, RL Ptokenadjusted =

 

 

(1 − k1σ(˜li,j))(1 − k2σ(˜ni)), if ai = a −e−

ρ·d˜′i,j

ri,j =

SmartThinker [78] SFT, RL



k0 , if ai ̸= a Ai,j = kni=0−j γn · r˜i,j+n



L(θ, α) = 2i=1 αi · δi δi = ϕsys-i,bound(x) − ϕsys-i,θ(x)

- λsys-1 = max

ϕsys-1, bound − ϕsys-1,θ

proxy

ϕsys-1,θs − ϕsys-1,θl

, 0

- λsys-2 = max

TLDR [84] SFT, RL

ϕsys-2, bound − ϕsys-2,θ

proxy

, 0

ϕsys-2,θl − ϕsys-2,θs

ReflectionStep, ci < ti NextStep, otherwise

ConCISE [55] SFT, SimPO si+1 = πθ(Si) =

Ltotal = Lcomp + Llatent r(yj) = λ1 · I(yj = yi∗) − λ2 · max (0, ℓ(yj) − ℓMin Correct) vl = µefficientl − µverbosel , h′l = hl + λvl

CoLE [100] SFT, RL

Table 10 Analyses on Mathematical Objective Functions in Efficient Reasoning Methods (Part III)

Name Method Objective Function

FlashThink [63] Prompt, SFT y = LLMθ(x | r) = LLMθ(x | c1, c2, . . . , c|r|)

LAALC = Attacc · Rraw + α · Rlen racc = A

###### Atarget , rlen = min 1, LLpred

val

AALC [103] RL

max

Rlen = 1 − min raccβ , rlen

Attacc = γ + (1 − γ)(1 − racc)

ConciseR [92] RL JGRPO++(θ) = E min(τi(θ)Aˆi, clip(τi)) + αH(πθ) ConciseRL [91] RL J(θ) = Ex∼ρEy∼pθ(·|x)[R(y, x)]

Elastic Reasoning [56] RL J(θ) = Ex∼D,y∼πθ(·|x;t∗,s∗)[r(y)]

ERL [49] RL Ex∼ρEy∼pθ(x) [1{y = y⋆(x)} · (1 − αf(LEN(y)))] Kimi k1.5 [7] RL S(y) +

0.5 · L(y)−L

###### Lmax−Lmin , if S(y) = 1 min(0, 0.5 − L(y)−L

min

Lmax−Lmin ), if S(y) = 0 L1 [51] RL r(y, ygold, ngold) =

min

1 − α · ngold − ny , if exact length constraint is used (L1-Exact) I(y = ygold) · clip α · (ngold − ny) + δ, 0, 1 , if max length constraint is used (L1-Max)

LASER [89] RL Rˆ(x, y) = C(y) + λ(y) · S(y) Length-Aware Optimization [87] RL rewardlen(i) =

β, r(x, yi, y∗) > 0 ∧ acc ≥ accmax − τacc 0, otherwise

MRT [48] RL ∆µk(x; π) := Ez∼π(·|x) kj=0−1 Jr(x; πj∗) − Jr(x; µ(·|x, z0:j)) O1-Pruner [47] RL L

ref

L(y) − 1 + λ(S(y) − S(yref)) S-GRPO [57] RL JS-GRPO(θ) = Eq,{oi} G 1 Gi=1 |o1

|oi| t=1 min π π(oi,t)

Aˆi,t, clip(·)Aˆi,t SelfBudgeter [70] RL R(C, F, ℓ, b) =

i|

old(oi,t)

rf F = 0 PB + PreB(·) F = 1

SPIRIT [52] RL PPL(x, {wk}Nk=1) = exp −N1 Ni=1 log p(wi | x, w1, . . . , wi−1)

 

0 (length within bounds) β length exceeded) η(L) otherwise

TLDR [31] RL r = rˆ− ζ(L), ζ(L) =



A*-Thought [76] SFT f(t′k + rw) = g(t′k + rw) + h(t′k + rw) Adaptive GoGI-Skip [58] SFT G(l

∗)

t = ∂L

ans

∂hlt∗

1

C3oT [38] SFT {(xi, rilong, yi)}Ni=1 CCoT [110] SFT LOSSφ(ziℓ, zˆiℓ) = k1 ki=1 σ2(1zℓ

i) MSE(ziℓ, zˆiℓ)

COCONUT [109] SFT Ht = Transformer(Et); M(xt+1 | x≤t) = softmax(Wht)

CODI [107] SFT L = αLteacher + βLstudent + γLKD CoT-Valve [42] SFT p(a | t1, . . . , tn, q; θ)

n

m

E(q,a)∼D p(a | t1, . . . , tm, q; θ + ∆θ)

p(ti | t<i, q; θ + ∆θ) Distill System 2 [43] SFT SII(x; pθ) → z, y

p(ti | t<i, q; θ); max

∆θ

i=1

i=1

DRP [73] SFT LSFT = − ni=1 log Pθ(yi | x, y<i)

Heima [111] SFT Pθ < CoT >(k) Kk=1i , Ya(i) | Xv(i), Xq(i) ICoT-KD [106] SFT P(y|x) ≈ z ˆ Pθ(ˆz|x)Pθ(y|x, zˆ)

ICoT-SI [108] SFT minθ − log Pθ(y, z1:m | x) InftyThink [40] SFT For i = 1 to n :

Si = summarize(M, RPi, {RPj}ij−=11) if i < n Conclusion = generate(M, RPn, Sn−1) if i = n

L2O 2

Vanilla:Dependency =

+ LP · LO

2LP LC + 2LOLC − L2P − L2C 2 LightThinker:Dependency =

H2O:Dependency =

LightThinker [41] SFT

LO

ContextLengtht LS-Mixture SFT [54] SFT L(Dlong) + L(Dshort) = (x

t=1

i,riS,yi) − log P(riS ⊕ yi | xi) PIR [68] SFT PIRθ(xi|x1:n) = log PPLPPL(R\{(Rx)i})

i,riL,yi) − log P(riL ⊕ yi | xi) + (x

R1-Compress [75] SFT cˆ∗i = arg maxcˆ∈C˜

πθ(ˆc | x, cˆ<i)

i

SF [44] SFT Relative Length = Avg.Avg.outputoutputlengthlength(baseline)(method) ; Relative Accuracy = AccuracyAccuracy(baseline)(method) Skip Steps [45] SFT Mkstandard = (q,a(n))∈D

P(a(n) | q) SOLAR [37] SFT J(θ) = Ex∼D,y∼πθ(·|x;t∗,s∗) [r(y)]

k

SoftCoT [113] SFT L = Ez∼qϕ(z|x,y) [log pθ(y|z, x)] − DKL qϕ(z|x, y) ∥ pθ(z|x)

 

zt(j) + β, if j ∈ TopK(zt, k) and uj < α, zt(j) − β, if j ∈ TopK(zt, k) and uj ≥ α, zt(j), otherwise.

L2 [80] SFT, Decoding Intervention zˆt(j) =



L = − log pθ(yg, R | x)

′,x) pθ(yg|R,x)

verbosity(yg) = log pθ(yg|R

VARR [77] SFT

verbosity(yw) − verbosity(yg) ≤ 0 Token assorted [112] SFT L(X) = log p X | fdec(q(X¯)) | g(P) + Li=1 sg[X¯i] − q(X¯i) 22 + β X ¯i − sg[q(X¯i)] 22 ACPO [90] SFT, RL Ri =

max(waccRacc + wlenRTLB + wthinkRthink, 0.1), yi correct min(· · · , −0.1), yi incorrect

TokenSkip [39] SFT L = li=1 log P(yi | x, γ, y<i; θM)

VeriThinker [65] SVFT minθ Eq [D (Mθ(· | q), Ci)] CoLaR [114] SFT, RL Ltotal = Lcomp + Llatent CoThink [60] SFT, RL, Distillation τ(M, D) = QCM(D)

M(D) , η(MR, MI) = QQRCI

ICR Long Short [86] SFT, RL M(yi) = log2 1 + dy−d{yd1,...,yi}

Niright Nisum − δ(yi)

dy−dyi dy

y

