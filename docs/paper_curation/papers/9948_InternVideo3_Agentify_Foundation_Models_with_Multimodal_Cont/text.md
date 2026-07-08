## arXiv:2606.12195v1[cs.CV]10Jun2026

[Figure 1]

2026-6-11

# InternVideo3: Agentify Foundation Models with Multimodal Contextual Reasoning

###### Ziang Yan1*, Sheng Xia2,3*, Jiashuo Yu1*, Yue Wu1*, Tianxiang Jiang1, Songze Li1, Kanghui Tian1, Yicheng Xu1, Yinan He1, Kai Chen1, Limin Wang3,1, Yu Qiao1 and Yi Wang1†

1Shanghai AI Laboratory, 2Shanghai Innovation Institute, 3Nanjing University

Project Page: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3

Recent progress in foundation models has increasingly shifted from one-shot prediction toward agentic behavior, where models solve tasks through multi-step reasoning, tool use, memory, and self-correction. However, much of the open-source momentum has centered on text-dominant settings such as coding, search, and long-context tool use, while long-horizon multimodal tasks remain comparatively underexplored. This gap is especially visible in video, where real-world tasks often require sustained temporal understanding, visually grounded evidence gathering, and iterative interaction with external tools or memory rather than a single-pass questionanswering step. We present InternVideo3, a framework for improving such capabilities through Multimodal Contextual Reasoning (MCR), a formulation that treats multimodal understanding as a closed-loop process over a shared evolving context. In MCR, multimodal observations, instructions, intermediate reasoning, tool actions, feedback, and memory are all represented within a unified context that is updated over time. This makes long-video understanding a process of evidence accumulation, belief revision, and verification, and provides a practical abstraction for multimodal agentic behavior. To make such long-horizon rollouts efficient, we introduce Multimodal Multi-head Latent Attention (M2LA), a token-preserving attention reparameterization that compresses KV-cache states while retaining the full multimodal token stream. We further develop a staged training recipe consisting of continued pretraining after M2LA conversion, short-to-long supervised fine-tuning for video, rule-based reinforcement learning on verifiable tasks, and on-policy distillation from stronger teacher models. Experiments on short-video and long-video benchmarks show that InternVideo3 achieves strong performance among open video models, with especially notable gains on long-horizon tasks such as Video-MME, MLVU, and EgoSchema. We also instantiate the model as a video agent with retrieval and verification tools, illustrating how recursive multimodal reasoning can support more robust evidence-grounded behavior. Overall, our results suggest that efficient context handling and closed-loop multimodal reasoning are important ingredients for adapting open multimodal models toward long-horizon visually grounded agency.

##### 1. Introduction

Foundation models are increasingly evaluated not only by how well they answer questions, but by how well they act over extended horizons. Across recent research, two themes have become especially prominent: agents, which emphasize multi-step reasoning (Yao et al., 2022b), tool use (Schick et al., 2023), memory (Park et al., 2023), and self-correction (Shinn et al., 2023); and world modeling, which emphasizes persistent internal state, temporal abstraction, and reasoning over evolving environments (Ha and Schmidhuber, 2018). Although these directions are often studied separately, they share a common shift in perspective: moving from one-shot prediction toward systems that maintain, update, and act on contextual knowledge over time.

*Equal contribution. †Corresponding author.

So far, however, much of the open-source progress in agentic AI has concentrated on text-dominant settings, including coding agents (Yang et al., 2024; Wang et al., 2025b), browser (Yao et al., 2022a; Zhou et al., 2024b) and search agents (Yao et al., 2022b; Jin et al., 2025; Zheng et al., 2025), function calling (Schick et al., 2023; Qin et al., 2024), and ultra-long-context language tasks (Zhang

- et al., 2024a; Bai et al., 2025c). By comparison, visually grounded long-horizon reasoning remains significantly less developed. Existing multimodal large language models (MLLMs) (Bai et al., 2025b,a; Wang et al., 2025a; Li et al., 2024b) have made impressive progress on image understanding and shortvideo QA, but they still struggle on tasks that require sustained temporal understanding, evidence localization, spatial reasoning, iterative evidence gathering, and interaction with external tools or memory. Yet these are precisely the settings in which multimodal agents must operate: the model must determine what it has seen, what remains uncertain, what additional evidence is needed, and whether its current conclusion is sufficiently grounded to answer or act.

This gap matters for both practical and scientific reasons. Practically, many real workloads, including long-form video analysis (Wu et al., 2024b; Wang et al., 2024c), instructional understanding (Miech et al., 2019), surveillance review (Sultani et al., 2018), egocentric perception (Grauman

- et al., 2022; Mangalam et al., 2023), and evidence-grounded temporal reasoning (Lei et al., 2018; Xiao et al., 2024; Liu et al., 2024c), cannot be reliably solved by single-pass multimodal QA. They require revisiting observations, preserving relevant memory, localizing supporting evidence, invoking specialized tools, and revising conclusions when new information arrives. Scientifically, these tasks offer a concrete testbed for capabilities often associated with multimodal agency and world-model-like reasoning: latent state tracking (Hansen et al., 2022, 2024), temporal abstraction (Machado et al., 2023; Kong et al., 2024), uncertainty-aware evidence selection (Asai et al., 2024), and closed-loop interaction between perception and decision making (Ahn et al., 2022; Brohan et al., 2022). In this sense, long-horizon video reasoning is not merely another benchmark category, but an important substrate for studying visually grounded agency in foundation models.

In this work, we study how to move open multimodal models toward this regime by agentifying their reasoning process rather than assuming a one-shot mapping from video to answer. We propose Multimodal Contextual Reasoning (MCR), a unified formulation in which multimodal observations, task instructions, intermediate reasoning traces, tool actions, feedback, and memory are all represented within a shared evolving context. Under MCR, long-video understanding becomes a recursive process of observe, reason, act, receive feedback, and update context. This formulation does not attempt to build a full action-conditioned world simulator. Instead, it provides a practical way for a multimodal foundation model to maintain and refine a task-relevant contextual belief state while reasoning over long visual streams and interacting with tools.

A major obstacle to this paradigm is efficiency. In long multimodal rollouts, the bottleneck is not only the number of visual tokens, but the growing KV-cache required to preserve observations, reasoning traces, tool outputs, and memory across many decoding steps. Existing approaches often reduce cost by aggressive frame subsampling, retrieval, or summarization. While useful, such methods may discard information that later becomes important for reasoning or verification. We therefore pursue a complementary direction: preserving the full multimodal token stream while reducing memory usage inside attention. To this end, we introduce Multimodal Multi-head Latent Attention (M2LA), a long-context-efficient attention design that compresses cached KV states while reconstructing head-specific representations on the fly. This makes longer multimodal rollouts feasible under practical hardware constraints.

Building on this efficient attention foundation, we develop a staged training recipe tailored to long-horizon multimodal reasoning. Starting from an open multimodal backbone, we first perform continued pretraining after M2LA conversion to recover language capability and multimodal alignment

under the new attention parameterization. We then carry out large-scale long-video supervised finetuning with a short-to-long curriculum, followed by rule-based reinforcement learning on verifiable tasks and on-policy distillation (Agarwal et al., 2024; Lu and Lab, 2025; Xu et al., 2025) from stronger teacher models. This pipeline strengthens the model’s ability to reason over dense visual evidence and extended temporal dependencies without requiring frontier-scale base models.

Empirically, InternVideo3 achieves strong results across short-video, long-video, and spatiotemporal reasoning benchmarks, with especially notable gains on long-horizon tasks such as Video-MME (Fu et al., 2024), MLVU (Zhou et al., 2024a), and EgoSchema (Mangalam et al., 2023). We further instantiate the model as a video agent with retrieval and verification tools, illustrating how recursive multimodal reasoning can support more robust evidence-grounded behavior. Taken together, our results suggest that context efficiency and closed-loop multimodal reasoning are useful ingredients for adapting open multimodal models toward long-horizon visually grounded agency.

Our contributions are summarized as follows:

- • We propose Multimodal Contextual Reasoning (MCR), a unified formulation for long-horizon multimodal reasoning in which observations, intermediate reasoning, tool use, feedback, and memory are represented within a shared evolving context.
- • We introduce M2LA, a long-context-efficient attention reparameterization that compresses KV-cache states while preserving the full multimodal token stream, enabling longer multimodal rollouts under constrained hardware budgets.
- • We develop a practical training recipe for long-video reasoning, combining continued pretraining after attention conversion, short-to-long supervised fine-tuning, rule-based reinforcement learning on verifiable tasks, and on-policy distillation from stronger teacher models.
- • We demonstrate strong results on short-video, long-video, and spatiotemporal reasoning benchmarks, and further present a video-agent instantiation that illustrates the practical value of recursive multimodal reasoning for evidence-grounded behavior.

##### 2. Related Work

Multimodal Large Language Models. The rapid development of multimodal large language models has expanded foundation-model capabilities from image-text alignment and image understanding (Radford et al., 2021; Alayrac et al., 2022; Liu et al., 2024b; Li et al., 2023; Chen et al., 2024b,a) to more general video understanding and multimodal reasoning (Wang et al., 2022, 2024e, 2025c; Cheng et al., 2024b; Li et al., 2024a; Bai et al., 2025a). Recent open-source video MLLMs achieve strong performance on standard short-to-medium video benchmarks such as VideoMME (Fu et al., 2024), MVBench (Li et al., 2024c), LVBench (Wang et al., 2024c), and related instruction-following settings. However, performance still degrades on genuinely long videos or tasks requiring sustained temporal reasoning, evidence localization, and multi-step context accumulation. A common limitation is that multimodal inputs are often treated as passive context to be summarized once, rather than as evolving evidence to be revisited and refined during reasoning. Our work focuses on this longerhorizon regime, where the challenge is not only to perceive visual content, but to maintain and update a coherent multimodal context over time.

Agents. Turning foundation models into agents has become a major research direction since ReAct (Yao et al., 2022b), which showed the effectiveness of interleaving reasoning and action. Subsequent work extend this paradigm to program synthesis, tool use, GUI interaction, mobile interfaces, and multimodal environments, including ViperGPT (Surís et al., 2023), VisProg (Gupta and Kembhavi,

- 2023), CogAgent (Hong et al., 2024), AppAgent (Zhang et al., 2025a), Mobile-Agent (Wang et al.,
- 2024a), and a growing family of coding and search agents (Yang et al., 2024; Wang et al., 2025b; Jin

- et al., 2025; Zheng et al., 2025). In multimodal settings, the core challenge is often not only selecting the next tool, but deciding what perceptual evidence is missing, where to look next, and whether current evidence is sufficient to answer or act. Prior video-agent systems (Wang et al., 2024d; Fan et al., 2024) demonstrate the value of tool-augmented video reasoning, such as frame sampling, temporal grounding, or subtitle retrieval, but they often rely on planner-controller decompositions, short interaction horizons, or task-specific pipelines. Our work instead studies how to support longerhorizon visually grounded reasoning within a unified MLLM context, where observations, intermediate reasoning, tool outputs, and memory are all represented within a common rollout.

World Models and Predictive Video Learning. World models aim to learn compact internal state representations that support prediction, planning, and decision making in partially observed environments. This line of work spans latent dynamics models for reinforcement learning (Ha and Schmidhuber, 2018; Hafner et al., 2019; Assran et al., 2025), action-conditioned predictive models, generative simulators, and recent self-supervised predictive representation learning approaches. An influential direction is the Joint-Embedding Predictive Architecture (JEPA), learning by predicting missing or future content in a learned embedding space rather than reconstructing pixels (Assran

- et al., 2023; Bardes et al., 2024; Assran et al., 2025; Mur-Labadia et al., 2026). JEPA-style methods emphasize learning abstract, predictable, and task-relevant structure while discarding low-level detail unnecessary for downstream reasoning or control. This paradigm has evolved from image representation learning in I-JEPA (Assran et al., 2023) to video predictive learning in V-JEPA (Bardes et al., 2024), and more recently to V-JEPA2 (Assran et al., 2025), connecting large-scale video pretraining with understanding, prediction, and robot planning. These works suggest predictive latent representations are a promising substrate for world understanding and, with additional action supervision, for planning. Our work is complementary to this literature. Rather than training a full action-conditioned latent world model from interaction data, we study how a multimodal foundation model can maintain and update a task-relevant contextual belief state over long visual streams during reasoning and tool interaction.

Context Modeling and Engineering. Handling long multimodal sequences remains a central challenge in both LLMs and MLLMs. Existing strategies include sparse frame sampling, hierarchical temporal modeling, retrieval-augmented frame selection, summarization, memory tokens, and context compression. For long videos and long documents, systems such as MovieChat (Song et al., 2024), LongVU (Shen et al., 2024), and Gemini 1.5 (Reid et al., 2024) employ retrieval, summarization, or compression to fit large evidence sets into finite contexts. Meanwhile, recent LLMs have increasingly focused on ultra-long contexts for agentic workloads (DeepSeek-AI, 2026). However, multimodal agency requires more than simply fitting long context into memory: it requires dynamic, reasoningguided evidence selection and the preservation of causal dependencies across observations, actions, and feedback. Our M2LA architecture addresses this challenge from an attention-efficiency perspective by compressing KV-cache states without dropping tokens, while MCR provides a formulation for recursively updating multimodal context during long-horizon rollouts.

Reasoning, Reflection, and Test-Time Interaction. Enhancing reasoning in foundation models has progressed through prompting techniques such as Chain-of-Thought (Wei et al., 2022), Treeof-Thought (Yao et al., 2023), self-reflection, and through learning-based methods such as process supervision, outcome-supervised RL, and test-time scaling. In multimodal settings, these mechanisms become tightly coupled with perception: the model must decide whether current observations suffice,

MCR Rollout cleancontextualbelief-updateworkflow

InternVideo3 : Agentify Foundation Models with Multimodal Contextual Reasoning

###### User Queries

#### Multimodal Context Reasoning

###### Tools & Env

Instruction : find out which athlete won the championship..

Temporal Grounding

###### Context : + Visual Evidence + Reasoning Trace + Memory

ASR

 +1 =  ⨁(  ,  +1,  +1)

Receive feedback  +1

Segment

Update belief

###### Visual Inputs

Verification

∼ 

∼  (⋯)

 +1 ∼ 

###### ...

execute  +1

[Figure 2]

[Figure 3]

Observe

Reason

Select action

 +1 ∼  Terminate when satisfies

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

...

Clip 1 Clip n

- Figure 1: Framework of MCR.

whether more evidence is needed, and which perception or tool operation should be performed next. Visual search and grounding methods such as V* (Wu and Xie, 2024) suggest that reasoning can guide what evidence to inspect within an image or video. Our work follows this direction, but places reasoning, perception updates, tool feedback, and memory into a single evolving multimodal context. This enables a closed-loop formulation in which belief updates and evidence gathering are part of the same sequence model, rather than auxiliary stages outside it.

##### 3. Method

We present InternVideo3, a framework for improving long-horizon multimodal reasoning through three complementary components: (1) Multimodal Contextual Reasoning (MCR), a formulation that represents observations, intermediate reasoning, tool actions, feedback, and memory within a shared evolving context; (2) Multimodal Multi-head Latent Attention (M2LA), an efficient attention reparameterization that reduces KV-cache cost for long multimodal rollouts; and (3) a staged training recipe that restores pretrained capability after attention conversion and then specializes the model for long-video reasoning.

The central idea is to treat long-horizon multimodal understanding not as a one-pass mapping from video to answer, but as a closed-loop process in which the model repeatedly observes, reasons, acts, receives feedback, and updates its contextual state. In this section, we first introduce the MCR formulation, then describe the long-context-efficient attention design that makes such rollouts practical, followed by our training recipe and video-agent instantiation.

###### 3.1. Multimodal Contextual Reasoning

Motivation. Conventional multimodal QA typically assumes that a model receives a fixed image or video and produces an answer in a single forward pass. This abstraction is effective for short and self-contained examples, but becomes increasingly inadequate for long videos and visually grounded agentic tasks. In these settings, the model must often decide what evidence is already available, what information remains missing, whether to inspect another temporal segment or spatial region, whether to invoke an external tool, and whether its current conclusion should be revised in light of new evidence. These requirements are more naturally captured by an evolving contextual state than by a single static prompt.

We therefore formulate multimodal reasoning as Multimodal Contextual Reasoning (MCR), in which task-relevant information, including multimodal observations, user instructions, intermediate

reasoning traces, tool actions, external feedback, and memory, is represented within a shared context that grows over time. This context acts as a task-grounded belief state: it records what the model has observed, what it has inferred, what actions it has taken, and what uncertainty remains.

Closed-Loop Rollout. Given a user query Q and an initial multimodal observation Vinit, MCR proceeds through a multi-step rollout. At step 𝑖, the model conditions on the current visual evidence V𝑖, the current action trace A𝑖, the tool or environment feedback F𝑖, and the accumulated context C𝑖, and produces an intermediate reasoning state or response R𝑖:

R𝑖 ∼ 𝑓𝜃(Q | V𝑖, A𝑖, F𝑖, C𝑖), 𝑖 = 1,2,...,𝑁. (1) The resulting reasoning state triggers an updated action decision, tool call, or perception request:

A𝑖+1 ∼ 𝜌(A𝑖 | R𝑖,F𝑖,Q), (2) F𝑖+1 ∼ 𝜋(A𝑖+1 | R𝑖,Q), (3) V𝑖+1 ∼ 𝛿(V𝑖 |C𝑖+1), (4)

where 𝜌(·) denotes the policy that selects the next action, 𝜋(·) denotes environment or tool execution, and 𝛿(·) denotes the perception update operator that determines what visual evidence should be observed next. Note the newly produced reasoning, action, and feedback are appended to the context:

C𝑖+1 = C𝑖 ⊕ (R𝑖,A𝑖+1,F𝑖+1), C1 = Tsys, (V1,A1,F1) = (Vinit,∅,∅), (5)

where ⊕ denotes context aggregation, optionally combined with summarization or compression to control growth over long rollouts. The rollout terminates after 𝑁 steps once the model emits a final answer or a termination action: R𝑁 |= Q.

In practice, we implement this process as an autoregressive rollout in which the model alternates between generating intermediate reasoning tokens, tool calls, and final answers, while appending tool outputs and memory summaries back into the same multimodal context.

Interpretation. MCR can be understood as a contextual belief-update process implemented inside an MLLM. Rather than learning a standalone action-conditioned world simulator, the model incrementally constructs and revises a task-relevant contextual state through multimodal evidence accumulation, reasoning, and tool interaction. Our goal is therefore not full predictive world modeling, but a practical mechanism for maintaining and updating grounded beliefs over long visual streams. In this sense, MCR is best viewed as a useful abstraction for long-horizon multimodal reasoning and visually grounded agentic behavior.

Actions and Tools in MCR. In our instantiation, actions may include:

- • Perceptive actions, such as requesting a temporal zoom-in, revisiting a segment, or selecting a more informative clip.
- • Tool actions, such as invoking ASR, segmentation, temporal grounding, or web search.
- • Memory actions, such as reading, writing, or summarizing intermediate context.
- • Verification actions, such as checking whether sufficient evidence supports the current answer.
- • Termination actions, such as returning the final response.

This unified view allows diverse agentic behaviors to be represented within the same sequence modeling framework.

·

[Figure 8]

InternVideo3 : Agentify Foundation Models with Multimodal Contextual Reasoning

Response: The video opens with a man standing in a mission control room, speaking to the camera. He is surrounded by multiple computer screens displaying various data and maps, including one that shows a global map with orbital paths. The room is filled with technical equipment, including monitors and control panels, indicative of a high-tech operational environment.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### Multimodal Multi-head Latent Attention

### LLM· (Qwen3 )

[Figure 13]

[Figure 14]

[Figure 15]

Hidden State

###### ......

[Figure 16]

[Figure 17]

[Figure 18]

Query Compressed KV RoPE Key

[Figure 19]

LLM Block i+1

⋮

[Figure 20]

[Figure 21]

[Figure 22]

###### LLM Block i · M2LA

Key Value

[Figure 23]

[Figure 24]

[Figure 25]

LLM Block i-1

Softmax · Attention

[Figure 26]

Output

cached latent queries

values

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

· · · · · · · ·

· · · ·

keys (recovered) keys (RoPE, small)

long doc picture short video long video + text

Patch & Project → vision tokens

[Figure 52]

Vision Encoder

Native Resolution Input

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

long doc high-resolution picture short video long video

- Figure 2: Architecture of InternVideo3.

###### 3.2. Long-Context Efficient Attention with M2LA

MCR places observations, reasoning traces, tool outputs, and memory updates into a shared evolving context. As the rollout grows, the main bottleneck is not only the number of input tokens, but the KV-cache footprint required to preserve this context during decoding. This issue is especially severe in multimodal settings, where visual evidence already occupies a large fraction of the context and additional agent-like traces accumulate over multiple reasoning steps.

A straightforward way to reduce this cost is to drop tokens through subsampling, retrieval, or summarization. While useful, such methods may remove information that later becomes important for verification or evidence integration. Since our goal is to preserve as much multimodal evidence as possible while still enabling long-horizon rollouts, we pursue a complementary approach: compressing cached attention states rather than dropping tokens. To this end, we introduce Multimodal Multi-head Latent Attention (M2LA), shown in Figure 2.

M2LA replaces standard GQA-style (Ainslie et al., 2023) attention blocks with a latent KV representation. Instead of storing full per-head keys and values for every token in the cache, the model stores a compact latent vector for each token and reconstructs head-specific keys and values on the fly during attention computation. This shifts long-context efficiency from a token-reduction problem outside the model to a state-compression problem inside attention itself. The full multimodal token stream is preserved, but the memory footprint of cached states is substantially reduced.

###### 3.2.1. Pretrained Attention Reparameterization

Our model is built by converting a pretrained GQA-based backbone into the M2LA form. The goal of this conversion is to preserve the original short-context behavior as much as possible while enabling a lower-memory long-context regime.

RoPE-Aware Positional Aggregation. For channels carrying positions, compressing keys across heads can distort the geometric structure induced by RoPE (Su et al., 2024). We therefore aggregate multi-head positional keys into a shared representation in an MQA-like manner, using a learned position-compatible linear aggregation obtained from a lightweight calibration pass. This preserves positional structure while avoiding redundant caching of similar positional channels across heads.

Low-Rank Latent Factorization for Content Channels. For non-positional content channels, we factorize keys and values through a low-rank bottleneck. Specifically, keys and values are first projected into a compact latent space, which is cached, and then head- or group-specific keys and values are reconstructed via learned up-projections. This decouples the KV-cache size from the original number of heads and head dimensions, allowing us to shrink the cache without discarding multimodal tokens.

Modality-Aware Latent Adaptation. A shared latent space is used for both text and vision tokens, but their distributions differ substantially. To accommodate this mismatch, M2LA includes lightweight modality-aware adapters, such as separate affine or linear mappings for text and vision tokens, applied on the cached latent vectors prior to reconstruction. This preserves a unified attention interface while allowing the latent representation to adapt to modality-specific statistics.

###### 3.2.2. Dynamic Latent Budgeting

The importance of high-rank visual representations is not uniform across layers or modalities. Early layers often require richer visual detail for alignment and retrieval, while deeper layers rely more heavily on distilled semantic representations. M2LA therefore supports layer-wise and modality-wise latent dimensions. In practice, we reduce the latent rank more aggressively in layers or modalities where reconstruction fidelity is less critical. This yields a controllable memory–accuracy trade-off and is particularly useful for multimodal agentic rollouts whose context grows over time.

###### 3.2.3. Compatibility with Head-wise QK-Norm

Recent backbones such as Qwen3 (Yang et al., 2025a) employ head-wise QK-Norm (Henry et al., 2020), applying RMSNorm (Zhang and Sennrich, 2019) independently to the query and key vectors of each attention head before RoPE. While beneficial for training stability, this normalization complicates post-training attention reparameterization because it introduces head-specific and input-dependent scaling factors, making a shared linear aggregation across heads appear invalid.

Conflict with Shared Aggregation. For head ℎ, the normalized key can be written as

xℎ 𝜎(xℎ)

, (6)

RMSNorm(xℎ) =

where 𝜎(xℎ) is the root-mean-square magnitude of the head input. Since 𝜎(xℎ) varies with both the head and the input token, one might expect this to obstruct any static aggregation across heads.

[Figure 64]

[Figure 65]

(a) The distribution of RMSNorm coefficients in Qwen3 is highly concentrated.

(b) Visualization of 2D RMSNorm 𝑍 = 𝑋/√︀

(𝑋2 + 𝑌 2 + 𝜖)/2, demonstrating good “flatness” and linear approximability.

- Figure 3: Empirical validation of RMSNorm concentration and 2D geometric “flatness”.

Training recipe

Continued PreTrain after M2LA conversion

Short-to-long long-video SFT

Rule-based group sequence policy opt.

On-policy distillation from stronger teachers

- Figure 4: Training recipe of InternVideo3.

Norm Concentration and Linear Approximation. We address this by observing that in highdimensional pretrained models, the RMS values are often strongly concentrated. If 𝜎(xℎ) has small variance around a mean 𝜇, then a first-order approximation yields

###### xℎ

1 𝜇

xℎ. (7)

RMSNorm(xℎ) =

𝜎(xℎ) ≈

Under this approximation, head-wise RMS normalization behaves approximately as a constant linear scaling, which restores compatibility with a shared latent reparameterization.

Empirically, we validate this approximation on a calibration set and find that a learned linear substitute can reproduce the original normalized outputs with high fidelity. This allows us to replace the original head-wise normalization path with a Global-Norm-Linear approximation during the M2LA conversion, making Qwen3-style architectures compatible with our attention reparameterization. Overall, M2LA enables substantially longer MCR rollouts by reducing KV-cache memory while keeping the original multimodal token stream intact.

###### 3.3. Training Recipe for Long-Horizon Multimodal Reasoning

To realize long-horizon multimodal reasoning after M2LA conversion, we adopt a staged training recipe that progressively restores pretrained capability, extends temporal understanding, and strengthens multi-step reasoning through verifiable post-training.

###### 3.3.1. Continued Pretraining after M2LA Conversion

Converting a pretrained GQA backbone into M2LA changes the internal attention parameterization and introduces an initial mismatch relative to the original pretrained model. This mismatch affects both language ability and multimodal alignment, especially before the vision encoder and decoder adapt to the new latent attention pathway. We therefore perform a lightweight continued pretraining (CPT) stage to recover these capabilities.

###### 3.3.2. Short-to-Long Supervised Fine-Tuning

After CPT, we conduct a short-to-long supervised fine-tuning stage to progressively build long-range temporal competence. Directly training on maximal-length videos often leads to unstable optimization, since the model must simultaneously learn temporal reasoning and extremely long-context attention. We instead use a curriculum that increases both temporal resolution and context length over stages.

- Stage 1: Short Context. We begin with videos sampled at 2 fps and capped at 512 frames, corresponding to approximately 32k tokens. This stage establishes basic temporal understanding with manageable computational cost.
- Stage 2: Long Context. We increase to 4 fps and up to 2048 frames, supporting contexts up to 256k tokens. This stage makes the model reason over extended durations and finer temporal detail.

###### 3.3.3. Rule-Based Reinforcement Learning

While SFT builds a stable initialization, it is limited by the scale and diversity of curated annotations. To further improve verifiable long-horizon reasoning, we apply rule-based group sequence policy optimization (GSPO) on automatically rewardable tasks.

Training Tasks. We use two families of tasks: 1) video QA, where correctness can be checked against reference answers; and 2) temporal grounding, where predictions can be scored by interval IoU against ground-truth moments. To improve training efficiency, we prioritize examples with meaningful policy uncertainty, filtering out samples that are already solved or too noisy.

Optimization. For each video-query pair, we sample a group of candidate responses, compute their rule-based rewards, normalize these rewards within the group, and optimize a clipped policy objective with KL regularization to the SFT reference model. This encourages the policy to improve relative ranking among candidate trajectories while remaining close to the stable supervised initialization. The resulting RL stage strengthens temporal localization, answer calibration, and other aspects of verifiable multimodal reasoning.

###### 3.3.4. On-Policy Distillation

Finally, we perform video on-policy distillation from a stronger teacher model. Unlike standard distillation that trains only on teacher-generated outputs, on-policy distillation supervises the student on its own sampled trajectories. For each selected prompt, the student generates a response trajectory and the teacher evaluates the same token sequence under the same prefix. The student is then

optimized with a reverse-KL objective toward the teacher distribution:

ℒOPD = EY∼𝜋𝜃(·|V,Q) [︃

(log 𝜋𝜃(𝑦𝑡|V,Q,Y<𝑡) − log 𝜋teacher(𝑦𝑡|V,Q,Y<𝑡))]︃, (8)

1 𝑇

∑︁𝑇

𝑡=1

where 𝜋teacher denotes a teacher model with strong capability (here we use Qwen3-235B). This provides dense supervision on the states actually visited by the student and reduces exposure bias over long reasoning trajectories.

We construct the distillation set from reasoning-heavy video QA and long-form captioning data, keeping only examples where the teacher is correct or significantly more complete while the student remains incorrect, incomplete, or weakly grounded. This stage is useful for transferring stronger behaviors in multi-step reasoning, fine-grained temporal comparison, and long-form visual description.

###### 3.4. Video-Agent Instantiation

To illustrate the practical implications of MCR beyond static QA, we instantiate InternVideo3 as a video agent that alternates between coarse perception, memory retrieval, targeted re-perception, tool use, and answer verification.

Hierarchical Memory. Given a video and a query, we first build a hierarchical memory consisting of sampled frames, scene boundaries, clip-level captions, timestamps, and optional subtitle or OCR signals. Each memory entry stores both semantic content and temporal metadata, enabling retrieval by either meaning or time constraints. During rollout, the agent maintains the current evidence window, a retrieved memory subset, and a compact belief summary recording the current hypothesis, unresolved uncertainty, and candidate timestamps.

Question Routing. Before acting, the model predicts a coarse question type, such as global, speech, knowledge, temporal, or fine-grained. This serves as a soft routing prior for deciding whether the current memory suffices or whether an external tool should be invoked.

Tool Set. The controller may call: 1) video_segmentation, to update scene-level structure; 2) asr, to retrieve spoken content; 3) web_search, to obtain external factual knowledge; 4) temporal_grounding, to localize relevant intervals; 5) and internal actions such as summarize, verify, and answer. Tool outputs are treated as feedback F𝑖+1 in the MCR framework and appended to the evolving context.

Recursive Verification. Before termination, the agent performs a lightweight self-check that asks whether the answer is adequately supported, what evidence backs it, and whether unresolved conflicts remain. If support is insufficient, the agent issues another retrieval or re-perception step focused on the uncertain segment. This verification mechanism reduces hallucination and encourages evidence-grounded responses.

Integration with M2LA. The video-agent system is tightly coupled with our long-context architecture. Because M2LA reduces the KV-cache footprint of accumulated evidence, feedback, and belief updates, the agent can retain longer histories without repeatedly rebuilding context from scratch. As a result, simple questions can be answered after a coarse pass, while harder temporal or causal questions can trigger additional recursive evidence gathering within the same long-horizon rollout.

###### Category Perception & Recognition Spatial-Temporal Understanding Event & Action Reasoning Holistic Semantics

Action Recognition 12.5 Temporal Reasoning 11.3 Action Reasoning 8.4 Plot Understanding 7.6 Object Recognition 9.7 Spatial Perception 10.8 Event Chain Reasoning 6.9 Information Synopsis 5.5 Attribute Perception 6.2 Multi-hop Temporal Reasoning 4.3 Object Reasoning 2.7 - OCR Problems 5.1 Scene State Tracking 3.8 Anomaly Recognition 0.1 - Counting Problem 1.5 Egocentric Understanding 3.2 - - - Needle Detail 0.4 - - - - - -

Sub-category

Total (100%) 35.4 33.4 18.1 13.1

- Table 1: Video understanding capability distribution across perception, spatiotemporal understanding, event reasoning, and holistic semantics.

Summary. Taken together, MCR provides the reasoning formulation, M2LA provides the efficiency mechanism, and our staged post-training recipe provides the capability acquisition path. This combination yields a more effective long-horizon visual reasoner and supports a practical video-agent instantiation built on top of an open multimodal foundation model.

- 4. Data Curation We describe below the data used in each stage of the training recipe.

###### 4.1. Continued Pre-training Data

The continued pre-training (CPT) corpus comprises 16M multimodal samples, corresponding to around 13.5B tokens. This stage is not to teach the model new task-specific behaviors, but to restabilize the converted backbone after the M2LA attention reparameterization and recover its general multimodal capability before downstream long-video training.

We adopt a balanced three-way mixture: 1) text-only data (29.9% in samples, 3.6B) (Wang et al., 2025d; Kuckreja et al., 2024; Yu et al., 2024; Hendrycks et al., 2020; Wei et al., 2024), to recover general language modeling, reasoning, and code ability; 2) image-text pairs (54.9%, 4.1B) (Li et al., 2024a; Marti and Bunke, 2002; Yu et al., 2016; Hudson and Manning, 2019), to preserve vision-language alignment and visual grounding; 3) video-caption data (15.2%, 5.8B) (Zhang et al., 2024b; Maaz et al., 2024; Wiedmann et al., 2025; Yang et al., 2025e; Wang et al., 2023; Clark et al., 2026), to adapt the model to temporal and multimodal inputs under the new attention structure.

###### 4.2. Supervised Fine-Tuning Data

Our supervised fine-tuning (SFT) corpus comprises ∼7.2M multimodal samples. The corpus spans long-video understanding (Zhang et al., 2024b), short-video QA (Caba Heilbron et al., 2015), image understanding (Lin et al., 2014; Lei et al., 2018), STEM reasoning (Kuckreja et al., 2024), code generation (Gui et al., 2025), document comprehension (Mathew et al., 2021; Masry et al., 2023), UI grounding (Cheng et al., 2024a), general conversation (Zhang et al., 2024b), translation (Penedo et al., 2026), and spatial reasoning (Yang et al., 2025e). Specifically, the largest components include STEM/science/math (22.4%), long-video QA (17.2%), general/conversational data (9.9%), other video understanding (9.1%), and image understanding (9.1%), with additional categories covering translation and language, code generation, chart/document/OCR tasks, UI grounding and longdocument understanding, general vision instruction tuning, video reasoning, temporal understanding, and 3D/spatial multimodal tasks. Unless otherwise noted, these datasets are standard public resources or previously used instruction-tuning corpora. Our main data contribution in this stage is the longvideo supervision component, which is specifically designed to support long-horizon multimodal

[Figure 66]

- Figure 5: Overview of the long-video supervision corpus. Top: video length distribution (𝑁 = 380𝐾, mean 15.8 minutes). Bottom: word-count distribution of context, questions, answers, and multiplechoice options.

reasoning while remaining compatible with broader multimodal and agentic workloads.

###### 4.2.1. Long-Video Supervised Fine-Tuning Data

To support long-horizon multimodal reasoning, we curate a large-scale long-video supervised finetuning (SFT) corpus through a hierarchical annotation pipeline designed to preserve both local visual detail and global temporal coherence. The resulting dataset contains 379K videos with a mean duration of 15.8 minutes (totally ∼100K hours), making it better suited to long-context multimodal reasoning than conventional short-video instruction data.

Data Sources. The corpus is constructed from complementary long-form video sources spanning both general and reasoning-intensive domains. The largest portion comes from InternVid (188K videos) (Wang et al., 2023), covering broad real-world content with video durations typically ranging from 5 minutes to over 30 minutes. We further include 115K YouTube-based reasoning videos, which contain rich instructional, explanatory, and cognitively demanding content, and 77K specialized videos from V-MME-style (Fu et al., 2024) tasks, which provide additional supervision for video-centric reasoning and evaluation-aligned capabilities. This mixture gives the training set both broad coverage and strong emphasis on long-form reasoning.

Hierarchical Narrative Annotation. Because raw long videos often exceed the reliable context window of current MLLMs, we decompose them into scene-consistent clips using scene-aware temporal splitting. This avoids the semantic fragmentation caused by naive uniform chunking and yields shorter segments that remain temporally coherent. Each clip is captioned by a strong teacher model to produce fine-grained, localized descriptions of actions, objects, scene changes, and salient events.

Clip-level captions provide local evidence but do not yield a coherent global understanding of the video. We therefore perform hierarchical caption merging: neighboring clip captions are first aggregated into scene-level summaries, and these scene summaries are then merged into a long-form narrative that preserves temporal order, cross-scene entity consistency, and high-level event structure. The resulting long-form narratives are compact but globally coherent, with an average narrative context length of 96 words. This hierarchical design provides both locally grounded supervision through clip-level captioning and globally coherent supervision through cross-temporal narrative integration.

QA Synthesis. On top of these hierarchical narratives, we synthesize over 1M question-answer pairs spanning four capability dimensions:

- • Perception & Recognition (35.4%): including action recognition, object recognition, attribute perception, OCR, and counting;
- • Spatial-Temporal Understanding (33.4%): including temporal reasoning, spatial reasoning, multi-hop temporal inference, scene-state tracking, and egocentric understanding;
- • Event & Action Reasoning (18.1%): including action reasoning, event-chain reasoning, objectcentric reasoning, and anomaly detection;
- • Holistic Semantics (13.1%): including plot understanding and long-form information synopsis.

Analysis. This SFT corpus is tailored to the needs of Multimodal Contextual Reasoning. In particular, the annotation pipeline encourages the model to learn from both fine-grained local evidence and cross-temporal global structure, which is essential for long-horizon multimodal reasoning. The diversity of QA types further encourages the model not only to recognize content, but also to track state over time, connect distant events, and form coherent high-level interpretations of long videos.

###### 4.3. Post-Training Data

The post-training stage uses two types of data, corresponding to the two objectives in our recipe: rule-based reinforcement learning and on-policy distillation (OPD). In both cases, the data is selected to emphasize informative long-horizon video examples rather than easy or low-signal samples.

Data for Rule-Based Reinforcement Learning. For video RL, we construct training data from two complementary sources: temporal grounding and multiple-choice video QA. These tasks provide automatically verifiable rewards, making them suitable for stable large-scale post-training.

For temporal grounding, we run the SFT model on the TimeLens (Zhang et al., 2026) training split and compute the intersection-over-union (IoU) between each predicted temporal span and the ground-truth interval. We retain examples with IoU in the range [0.1,0.7]. This removes both near-trivial cases that the model already solves reliably and severely mismatched cases that are too noisy to provide useful reward signals. From the filtered pool, we sample 5K examples for RL training.

For multiple-choice video QA, we sample responses from our SFT multiple-choice data using temperature-1 decoding. We retain questions for which the sampled responses contain both correct

and incorrect answers, yielding examples with meaningful policy uncertainty and non-degenerate reward structure. This produces a training set of 10K questions with verifiable correctness signals.

Data for On-Policy Distillation. We focus on examples that reveal a clear teacher–student capability gap. Concretely, we retain samples where the teacher model produces a correct, complete, or wellgrounded response, while the student produces an incorrect, incomplete, or weakly grounded one. This filtering ensures distillation is spent on examples that are genuinely informative for the student, rather than on cases the student already handles well or cases where the teacher is unreliable.

This OPD set is drawn from reasoning-intensive video QA and long-form video description data, with an emphasis on examples requiring temporal localization, cross-event comparison, multi-step inference, or dense long-video comprehension. Thus, OPD complements rule-based RL as RL improves performance on verifiable temporal and QA tasks, while OPD transfers stronger teacher behavior on more complex trajectories that are difficult to optimize using hand-designed rewards alone.

##### 5. Experiments

We evaluate InternVideo3 from three complementary perspectives. First, we measure general multimodal capability on standard short-video, long-video, and spatiotemporal reasoning benchmarks. Second, we study agentic multimodal reasoning through a video-agent setup with retrieval, grounding, and verification tools. Third, we analyze the contribution of each component in our recipe, including M2LA, short-to-long training, rule-based RL, and on-policy distillation. Across these experiments, our goal is not only to assess static video QA accuracy, but also to test whether the proposed Multimodal Contextual Reasoning framework improves the model’s ability to reason over evolving visual evidence under long-horizon constraints.

Model. Our model is built on a Qwen3-based (Yang et al., 2025a) multimodal backbone in the 7/8B parameter regime and converted to the proposed M2LA attention form. We then apply the full staged recipe described in Section 3: continued pretraining after M2LA conversion, short-to-long SFT on large-scale video data, rule-based RL, and on-policy distillation from stronger teacher models.

Evaluation Protocol. Unless otherwise stated, we follow the official evaluation protocol of each benchmark. For standard benchmark comparisons, we report the scores in their official metric, such

- as accuracy or average score, and compare against representative open-source video MLLMs. For agentic evaluation, we report task success or answer accuracy under a fixed interaction budget and tool set. Additional implementation details are provided in the appendix.

Benchmarks. We evaluate three groups of video-centric tasks, covering long-context understanding, short-video reasoning, and spatiotemporal intelligence:

• Long-video understanding: We evaluate on Video-MME (Fu et al., 2024) and VideoMME-v2 (Fu et al., 2026), which provide comprehensive video QA tasks over diverse video durations and domains, emphasizing robust multimodal reasoning. LongVideoBench (Wu et al., 2024b), MLVU (Zhou et al.,

- 2024a), and LVBench (Wang et al., 2024c) further stress long-context modeling, requiring models to retrieve sparse evidence and reason over extended videos. VRBench (Yu et al., 2025a) focuses on multi-step reasoning over long narrative videos, while EgoSchema (Mangalam et al., 2023) evaluates long-horizon understanding of egocentric human activities.

Model Video-MME VideoMME-v2 LongVideoBench MLVU LVBench VRBench EgoSchema Proprietary Models

GPT-5 (Singh et al., 2025) 83.3 44.7 72.6 77.7 65.2 - 75.6 GPT-5 mini (Singh et al., 2025) 77.3 - 69.7 69.1 54.7 - 70.9 Gemini 3 Pro (Google DeepMind, 2025) 88.6 56.8 75.9 75.7 77.0 76.1 68.9 Gemini 2.5 Pro (Comanici et al., 2025) 87.8 - 76.8 81.5 75.7 - 72.2 Gemini 2.5 Flash (Comanici et al., 2025) 84.2 - 73.1 75.1 64.9 - 70.2 Claude Sonnet 4.5 (Anthropic, 2024) 74.2 - 65.1 64.0 50.5 - 73.1

###### Open-Weight Models

VideoChat-Flash-7B (Li et al., 2024d) 65.3 27.5 64.7 56.0 48.2 50.8 51.3 InternVL3.5-8B (Wang et al., 2025a) 66.0 26.0 62.1 53.2 43.4 64.1 58.6 Qwen3-VL-8B (Bai et al., 2025a) 71.4 27.9 62.4 57.6 58.0 59.4 69.8 Keye-VL-1.5-8B (Yang et al., 2025b) 73.0 23.8 66.0 53.8 42.8 53.2 56.3 GLM-4.1V-9B (Hong et al., 2025b) 68.2 26.9 65.7 56.6 44.0 59.8 62.6 LLaVA-Video-7B (Zhang et al., 2024b) 63.3 19.9 58.2 52.8 44.2 - 57.3 Molmo2-8B (Clark et al., 2026) 69.9 27.4 67.5 60.2 52.8 - 62.0 MiniCPM-V-4.5-8B (Yu et al., 2025b) 67.9 26.5 63.9 60.6 50.4 55.9 49.6 Eagle2.5-8B (Chen et al., 2026) 72.4 24.9 66.4 60.4 50.9 55.7 72.2 InternVideo2.5-7B (Wang et al., 2025c) 65.1 26.4 60.6 72.8 46.4 51.9 63.9 InternVideo3 73.8 27.6 66.8 77.3 55.7 69.4 76.6

- Table 2: Long-video benchmark results across long video understanding, captioning, and counting benchmarks. Some entries are unavailable because the corresponding models are not applicable or not evaluated on those benchmarks. The best-performing open-weight model is in bold, and the second best is underlined.

- • Short-video understanding: We evaluate on NExT-QA (Xiao et al., 2021) and Perception Test (Patraucean et al., 2024), which test causal, semantic, and multimodal reasoning in short videos. MVBench (Li et al., 2024c), TOMATO (Shangguan et al., 2025), MotionBench (Hong et al., 2025a), and TempCompass (Liu et al., 2024c) focus more explicitly on temporal perception, including event ordering, frame-order sensitivity, motion dynamics, and fine-grained temporal changes.
- • Spatiotemporal intelligence: For temporal grounding, we evaluate on QVHighlights (Lei et al., 2021), Charades-STA (Gao et al., 2017), and ActivityNet Captions (Caba Heilbron et al., 2015), which require localizing query-relevant moments or densely grounding events in untrimmed videos. For spatial intelligence, we use VSIBench (Yang et al., 2025c), MMSIBench (Yang et al., 2025f), MMSIBench-Video (Lin et al., 2025), and DSIBench (Zhang et al., 2025c), which evaluate spatial configuration understanding, multi-view reasoning, dynamic 3D relations, and spatial memory.

We follow the official evaluation protocol of each benchmark and report the official metrics.

- 5.1. Main Results

- 5.1.1. Long-Video Understanding

- Table 2 summarizes results on long-video benchmarks. Overall, InternVideo3 performs strongly across this suite and is particularly competitive on benchmarks that require sustained temporal reasoning, long-range evidence integration, and coherent understanding over extended visual context. Among open-weight models, it achieves the best results on Video-MME, MLVU, VRBench, and EgoSchema, while remaining close to the top tier on VideoMME-v2 and LongVideoBench. These evaluations are still standard QA-style benchmarks rather than explicit agentic rollouts, thus they test the underlying long-horizon reasoning abilities that a multimodal agent would need.

Specifically, InternVideo3 achieves the best open-weight score on Video-MME with 73.8, slightly

surpassing Keye-VL-1.5-8B (73.0), Eagle2.5-8B (72.4), and its base family Qwen3-VL-8B (71.4). On VideoMME-v2, InternVideo3 obtains 27.6, which is slightly below the strongest open baseline Qwen3-VL-8B (27.9) but still among the top open results. This suggests that our post-training and long-horizon adaptation improve performance substantially on most long-video tasks, while some benchmark-specific strengths of the base model remain difficult to fully preserve or exceed.

On LongVideoBench, InternVideo3 reaches 66.8, ranking among the strongest open 8B-scale models and trailing Molmo2-8B (67.5) by only 0.7 points. On MLVU, InternVideo3 achieves 77.3, outperforming all listed open-weight baselines, including InternVideo2.5-7B (72.8), MiniCPM-V-4.58B (60.6), Molmo2-8B (60.2), and Eagle2.5-8B (60.4). This is one of the most notable gains and indicates strong improvement on long-form video reasoning beyond short-range event recognition.

On LVBench, InternVideo3 scores 55.7. This is competitive and above many open baselines, including InternVideo2.5-7B (46.4), but remains below Qwen3-VL-8B (58.0). Since InternVideo3 is built on Qwen3-VL-8B, this result also serves as a useful reminder that our long-horizon adaptation is not uniformly dominant on every benchmark: in some cases, the base model retains dataset-specific advantages that our recipe does not fully improve upon. Even so, the model still achieves a strong overall long-video profile across the broader suite.

Two particularly notable results are VRBench and EgoSchema. On VRBench, InternVideo3 achieves 69.4, the best result among open-weight models and substantially above Qwen3-VL-8B (59.4), InternVL3.5-8B (64.1), and Eagle2.5-8B (55.7). On EgoSchema, InternVideo3 reaches 76.6, the best open-weight result in the table, exceeding Eagle2.5-8B (72.2), Qwen3-VL-8B (69.8), and even the reported human score of 76 on this benchmark. We note that surpassing human performance on a single benchmark should be interpreted cautiously, but it nonetheless indicates that an 8B-scale open model can be highly competitive even against much larger systems in some long-video settings.

Relative to the previous InternVideo2.5-7B baseline, the gains are substantial on several representative long-horizon tasks: +8.7 on Video-MME (73.8 vs. 65.1), +6.2 on LongVideoBench (66.8 vs. 60.6), +4.5 on MLVU (77.3 vs. 72.8), +9.3 on LVBench (55.7 vs. 46.4), +17.5 on VRBench (69.4 vs. 51.9), and +12.7 on EgoSchema (76.6 vs. 63.9). Taken together, these results show that our method is effective in the long-video regime that most directly motivates the paper. While it does not outperform every model on every benchmark, it consistently strengthens long-horizon video understanding and brings an 8B-scale open model close to, and in a few cases beyond, the performance of much larger proprietary systems on selected tasks.

###### 5.1.2. Short-Video Understanding

- Table 3 reports results on short-video benchmarks. Although InternVideo3 is designed primarily for long-horizon multimodal reasoning, it also performs strongly on standard short-video QA, indicating that the proposed attention conversion and long-video-oriented post-training do not come at the expense of short-form capability. More importantly, the model achieves the best Short QA Avg. among the listed open-weight models, suggesting that improvements aimed at long-horizon reasoning also transfer to general short-video understanding.

At the aggregate level, InternVideo3 obtains a Short QA Avg. of 69.0, outperforming Eagle2.58B (67.0), InternVideo2.5-7B (66.5), VideoChat-Flash-7B (66.4), and Qwen3-VL-8B (65.3). This shows although our method is motivated by long-context and long-video reasoning, the resulting model remains broadly competitive rather than narrowly specialized.

Concretely, InternVideo3 matches the best reported open-weight score on NextQA with 85.5, tying VideoChat-Flash-7B and slightly exceeding Eagle2.5-8B (85.0), InternVideo2.5-7B (84.9), and

Model NextQA PerceptionTest MVBench Tomato MotionBench TempCompass Short QA Avg. Proprietary Models

GPT-5 (Singh et al., 2025) 86.3 79.4 74.1 53.0 65.4 80.4 73.1 GPT-5 mini (Singh et al., 2025) 83.2 72.0 66.5 44.1 59.9 74.9 66.8 Gemini 3 Pro (Google DeepMind, 2025) 84.3 77.6 70.4 48.3 62.6 82.8 71.0 Gemini 2.5 Pro (Comanici et al., 2025) 85.3 78.4 70.6 48.6 62.0 81.9 71.1 Gemini 2.5 Flash (Comanici et al., 2025) 81.8 74.7 67.0 39.1 59.3 80.2 67.0 Claude Sonnet 4.5 (Anthropic, 2024) 79.2 64.3 62.1 39.6 58.5 72.8 62.8

###### Open-Weight Models

VideoChat-Flash-7B (Li et al., 2024d) 85.5 76.5 74.0 32.5 60.6 69.4 66.4 InternVL3.5-8B (Wang et al., 2025a) 81.7 72.7 72.1 24.6 56.6 70.3 63.0 Qwen3-VL-8B (Bai et al., 2025a) 83.4 72.7 68.7 35.7 56.9 74.3 65.3 Keye-VL-1.5-8B (Yang et al., 2025b) 75.8 64.2 56.9 33.0 55.1 75.5 60.1 GLM-4.1V-9B (Hong et al., 2025b) 81.3 74.2 68.4 30.0 59.0 72.3 64.2 LLaVA-Video-7B (Zhang et al., 2024b) 83.2 68.8 58.6 24.9 54.2 66.6 59.4 MiniCPM-V-4.5-8B (Yu et al., 2025b) 78.8 70.9 60.5 29.8 59.7 72.7 62.1 Eagle2.5-8B (Chen et al., 2026) 85.0 81.0 74.8 31.0 55.7 74.4 67.0 InternVideo2.5-7B (Wang et al., 2025c) 84.9 74.9 75.7 32.9 60.8 70.1 66.5 InternVideo3 85.5 81.4 75.0 37.4 60.6 74.0 69.0

- Table 3: Short-video benchmark results across short video question-answering benchmarks. The best-performing open-weight model is in bold, and the second best is underlined.

Qwen3-VL-8B (83.4). On PerceptionTest, InternVideo3 achieves 81.4, the best open-weight result in the table, slightly above Eagle2.5-8B (81.0) and well above Qwen3-VL-8B (72.7) and InternVideo2.57B (74.9). On MVBench, InternVideo3 obtains 75.0, which is highly competitive and close to the best open results, though still below InternVideo2.5-7B (75.7). On Tomato, InternVideo3 reaches 37.4, the strongest open-weight result listed, improving over Qwen3-VL-8B (35.7), Keye-VL-1.5-8B (33.0), Eagle2.5-8B (31.0), and InternVideo2.5-7B (32.9).

The remaining two benchmarks show a similar pattern of broad competitiveness without universal dominance. On MotionBench, InternVideo3 scores 60.6, tying VideoChat-Flash-7B and narrowly trailing InternVideo2.5-7B (60.8). On TempCompass, InternVideo3 achieves 74.0, which is competitive but below the best open results from Keye-VL-1.5-8B (75.5) and Eagle2.5-8B (74.4). These cases are worth noting because they show that the gains from our long-horizon training recipe are not uniform across all short-video tasks. In particular, some short-form benchmarks may still reflect strengths of the base model or of alternative data mixtures that are not the main focus of our method.

Compared with InternVideo2.5-7B, InternVideo3 improves on nearly all reported short-video tasks: +0.6 on NextQA, +6.5 on PerceptionTest, +4.5 on Tomato, and +3.9 on TempCompass, while remaining essentially comparable on MVBench and MotionBench. Relative to its base family Qwen3-VL-8B, the gains are also consistent: +2.1 on NextQA, +8.7 on PerceptionTest, +6.3 on MVBench, +1.7 on Tomato, +3.7 on MotionBench, and a small -0.3 on TempCompass. Overall, these results indicate that the model’s improvements are not confined to long-video evaluation alone. Instead, better context handling and post-training for long-horizon reasoning appear to also strengthen short-video perception, temporal reasoning, and answer calibration.

5.1.3. Spatiotemporal Intelligence

- Table 4 reports results on temporal grounding and spatial intelligence benchmarks. Overall, InternVideo3 shows strong performance on temporal grounding, where it achieves the best results among the listed open-weight models on all three benchmarks. These evaluations are particularly relevant to our motivation as they test whether the model can identify when and where relevant evidence appears, rather than only producing the correct final answer. Although the experiments here are still direct

Model Temporal Grounding Spatial Intelligence QVHighlights Charades-STA ANet VSIBench MMSIBench MMSIBench-Video

Proprietary Models GPT-5 (Singh et al., 2025) 56.8 40.5 42.9 — 41.9 36.8 GPT-5 mini (Singh et al., 2025) — — — — 34.9 Gemini 3 Pro (Google DeepMind, 2025) — — — — — 38.0 Gemini 2.5 Pro (Comanici et al., 2025) 70.4 52.8 58.1 — 37.6 Gemini 2.5 Flash (Comanici et al., 2025) 64.3 48.6 52.5 — 33.1 35.4

###### Open-Weight Models

VideoChat-Flash-7B (Li et al., 2024d) 32.7 39.7 24.8 32.8 27.6 28.2 InternVideo2.5-7B (Wang et al., 2025c) 32.7 39.7 24.8 33.4 26.9 27.3 InternVL3.5-8B (Wang et al., 2025a) 31.3 27.8 31.3 56.0 30.5 29.2 Qwen3-VL-8B (Bai et al., 2025a) 59.4 48.3 46.8 59.1 27.0 27.6 Keye-VL-1.5-8B (Yang et al., 2025b) 55.5 45.4 41.3 36.4 26.6 26.5 MiniCPM-V-4.5-8B (Yu et al., 2025b) 28.5 30.0 22.7 32.0 28.1 23.4 InternVideo3 59.9 50.4 47.9 68.1 27.6 30.7

- Table 4: Spatiotemporal intelligence results across temporal grounding and spatial understanding benchmarks.

benchmark evaluations rather than full agentic interactions, these localization and grounding abilities are important ingredients for downstream multimodal agents.

On the three temporal grounding benchmarks, InternVideo3 achieves 59.9 on QVHighlights, 50.4 on Charades-STA, and 47.9 on ANet, establishing the best open-weight results in the table on all three tasks. Compared with its base model Qwen3-VL-8B, the gains are consistent though moderate: +0.5 on QVHighlights, +2.1 on Charades-STA, and +1.1 on ANet. Compared with earlier open baselines such as VideoChat-Flash-7B and InternVideo2.5-7B, the improvements are much larger. At the same time, proprietary Gemini 2.5 Pro remains clearly ahead on this suite, reaching 70.4/52.8/58.1. We therefore do not claim to close the gap to the strongest proprietary systems, but the results do show that our approach yields a strong and consistent grounding capability within the open 8B regime.

On the spatial intelligence side, InternVideo3 achieves 68.1 on VSIBench, 27.6 on MMSIBench, and 30.7 on MMSIBench-Video. The strongest result here is on VSIBench, where InternVideo3 outperforms all listed baselines by a clear margin, including Qwen3-VL-8B (59.1) and InternVL3.5-8B (56.0). On MMSIBench-Video, InternVideo3 also obtains the best result among the listed models. On MMSIBench, however, InternVideo3 is competitive but not best, trailing InternVL3.5-8B (30.5) and slightly below MiniCPM-V-4.5-8B (28.1). This again reflects the method improves the model substantially on many spatiotemporal tasks, but does not dominate every individual benchmark.

These results are important beyond standalone benchmark scores. A model can answer video QA questions reasonably well by relying on priors or coarse summaries, yet still fail to localize the supporting evidence in time or space. In contrast, temporal grounding and spatial intelligence more directly measure whether the model can connect answers to the relevant evidence. This distinction matters for multimodal agents: before an agent can decide what tool to call, what segment to revisit, or whether a conclusion is sufficiently supported, it must first know where and when the evidence lies. From this perspective, the strong grounding and spatial results of InternVideo3 provide useful support for the broader claim that better multimodal contextual reasoning can strengthen capabilities relevant to future agentic behavior.

###### 5.2. Inference Efficiency

[Figure 67]

- (a) Decode throughput.

[Figure 68]

- (b) End-to-end latency, decomposed into prefill and decode time.

- Figure 6: Inference efficiency of the Qwen3-VL-8B and its M2LA-converted variant on a single H200 GPU. Prefill length varies from 32K to 768K tokens, with decode length fixed to 16K. For 256K+ inputs, both models use 64K chunked prefill and no external inference acceleration. M2LA consistently improves long-context decoding efficiency and remains executable up to 768K prefill tokens, whereas the Qwen3-VL runs out of memory from 512K onward.

We evaluate the inference efficiency of M2LA during long-context generation. Specifically, we compare the original Qwen3-8B backbone with its M2LA-converted counterpart under identical inference settings. All measurements are conducted on a single H200 GPU, using one warm-up run followed by ten measured runs. We vary the prefill length from 32K to 512K tokens while fixing the decode length to 16K. No external inference acceleration is applied to either model. For inputs of 256K tokens and above, both models use the same chunked prefill strategy with a chunk size of 64K.

Figure 6 shows that M2LA consistently improves decoding efficiency in the long-context regime. At 32K prefill tokens, the converted model reaches 39.96 tok/s during decoding, compared with 21.74 tok/s for the original model, corresponding to a 1.84× speedup. As the context grows, the

###### Question Why is Angelo, the main character in the video, matching his customized suit?

[Figure 69]

###### Video input

Location 1

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

1

###### 2 3 4

The video also includes scenes where Angelo is depicted in formal attire, interacting with other characters in settings that suggest a blend of personal and professional contexts. This further supports the idea that his relationships and decisions are influenced by both his personal and professional life.

The presence of themes such as money, loyalty, and marriage, as well as the interactions between characters, suggest that Angelo's decisions and actions are deeply tied to his relationships and responsibilities. The scenes involving money and family responsibilities indicate that his choices are shaped by both personal and professional obligations

The video depicts a character named Angelo who is navigating complex personal and professional relationships. He is seen in various settings, including social gatherings, domestic environments, and formal settings, suggesting a rich tapestry of interactions.

Angelo is shown interacting with several characters, including a woman in period clothing, a man in a priest's attire, and another man in a suit, indicating a mix of familial and professional ties. His interactions often involve themes of loyalty, family responsibilities, and financial matters, hinting at a narrative deeply rooted in these themes.

Key visual evidence

Output Because he took the glasses guy's advice about wanting to be a banker, he had two professional designers match custom suits

to him in order to look more like a banker.

- Figure 7: Event Attribution. A qualitative demo on long-video QA: the model is asked why Angelo is fitting a custom suit, and infers that he wants to look more like a banker after following advice from the man with glasses.

benefit becomes more pronounced: decode throughput improves by 4.12× at 128K, 4.77× at 256K, and 5.01× at 384K prefill tokens. In terms of end-to-end latency, M2LA reduces total runtime by 1.83× at 32K and by more than 4× in the 256K–384K regime.

The largest advantage appears at the longest contexts. The original model runs out of memory

- at 512K prefill tokens, whereas the M2LA remains executable and completes the 16K-token decode phase at 8.52 tok/s. This behavior is consistent with the KV-cache analysis under batch size 1 and bf16 precision, where M2LA reduces the KV-cache footprint by roughly 50% across context lengths.

These results validate the practical efficiency benefit of compressing long-context KV states. In particular, the gains are most significant in the decode-dominated regime that arises in long-horizon multimodal reasoning, where the model must preserve large multimodal contexts while generating long intermediate traces, tool calls, and final responses.

###### 5.3. Agentic Video Exploration

In addition to benchmark evaluation, we instantiate InternVideo3 as a video agent using MCR with access to segmentation, ASR, temporal grounding, search, summarization, and verification tools. Although we leave large-scale quantitative agent evaluation to future work, this setup provides a qualitative illustration of the recursive evidence-gathering behavior enabled by our formulation.

Figures 7–10 show four representative agentic video exploration examples. These demos cover complementary forms of long-horizon multimodal reasoning: attributing a character’s action to earlier contextual cues, linking temporally distant scenes through a shared theme, relating visual equipment to downstream battle performance, and inferring a protagonist’s emotional state from implicit narrative evidence. In each case, the agent grounds its answer in selected visual evidence rather than relying on a single global video summary, illustrating how iterative evidence collection helps resolve questions that require cross-scene context and implicit reasoning.

For long-form video questions, the agent first consults coarse hierarchical memory, identifies uncertain or relevant segments, and then issues targeted tool calls. Speech-heavy questions trigger ASR, event-centric questions trigger temporal grounding, and low-confidence answers trigger a

###### Question How does the video showing soldiers on a mission in the forest relate to the peace scenes shown?

[Figure 74]

###### Video input

Location 1

[Figure 75]

[Figure 76]

[Figure 77]

1

###### 2 3

Key visual evidence

The video begins with a scene showing soldiers in a forest, indicating a wartime scenario. This is followed by scenes of people in various settings, including formal interviews and discussions, which suggest a focus on historical or cultural narratives.\

The juxtaposition of the forest scene with the formal interviews and discussions creates a contrast between the harsh realities of war and the structured, reflective moments of peace and reflection. This contrast highlights the tension between the past struggles and the present quest for peace.\

The video's overall theme seems to be exploring the impact of war and the journey towards peace, with the forest scene symbolizing the conflict and the interviews and discussions representing the efforts to achieve peace.

Output The contrast may be showing the tension between the war Vietnam experienced and the quest for peace.

- Figure 8: Logical Linkage. A qualitative demo on thematic reasoning: the model is asked how soldiers on a forest mission relate to later peace scenes, and explains that the contrast reflects wartime tension and the pursuit of peace.

Variant V-MME LVB MLVU LVBench Avg. Full 73.1 64.6 75.4 54.2 66.8 w/o CPT 68.9 62.6 72.4 54.7 64.7 w/o Long Ctx. 70.4 61.6 75.1 54.5 65.4 w/o LV Data 70.9 64.6 72.1 51.4 64.8

- Table 5: Ablation results on representative long-video benchmarks. V-MME and LVB denote VideoMME and LongVideoBench, respectively. We report the official score for each benchmark and the arithmetic average over the four tasks.

verification step before termination. Compared with a single-pass baseline, this iterative procedure more reliably recovers from incomplete first-pass evidence and reduces errors caused by relying on overly coarse summaries.

These observations support the motivation of MCR: in realistic multimodal settings, the problem is often not only to generate an answer from a fixed prompt, but to determine whether the current evidence is sufficient, what additional evidence is needed, and how that new evidence should update the current belief state.

###### 5.4. Ablation Studies

To better understand which parts of our recipe matter most for long-horizon video reasoning, we conduct ablations on three representative ingredients: continued pretraining after M2LA conversion (CPT), long-context training (Long Ctx.), and curated long-video supervision (LV Data). Results on four representative long-video benchmarks are shown in Table 5. The full model achieves the best overall average (66.8), indicating that the different stages of our recipe provide complementary gains rather than redundant improvements. Removing CPT causes the largest drop on Video-MME (-4.2) and a clear degradation on MLVU (-3.0), reducing the overall average from 66.8 to 64.7. This suggests that continued pretraining is an important recovery stage after M2LA conversion: without it, the model does not fully restore its language-side ability and multimodal alignment before downstream

###### Question How does the character's outfit in the video relate to their performance in battle?

[Figure 78]

###### Video input

Location 1

[Figure 79]

[Figure 80]

[Figure 81]

1

###### 2 3

The attire also influences the character's agility and maneuverability, allowing them to navigate challenging environments and execute precise movements during combat.

Key visual evidence

The equipment can also provide bonuses or special abilities, directly impacting their effectiveness in battle and survival chances.

The character's outfit, including armor and weapons, is designed for combat and protection, providing a physical advantage in battle.

Output A character's equipment directly affects their fighting ability and survival chances.

- Figure 9: Relational Reasoning. A qualitative demo on battle understanding: the model is asked how a character’s outfit affects combat performance, and answers that armor, weapons, and gear directly improve fighting ability and survival chances.

Question In the video when the hero learns that his wife chose to commit suicide because of her guilt may I ask how he felt at that moment?

Video input

Location 1

Key visual evidence

1

The video shows a man and his wife in a bedroom setting, indicating a private and intimate moment.

2 3 4

Output At this point he should be feeling very desperate that he might have been able to keep his wife alive by not purchasing the fund,

and remorseful about his inability to save her life

The man is shown speaking on the phone, suggesting a serious conversation, possibly related to their situation.

The woman is depicted lying down, possibly indicating a state of rest or emotional exhaustion, which aligns with the context of her choosing to commit suicide due to guilt.

The narrative in the video implies a sense of helplessness and regret, as the man reflects on the situation and possibly feels responsible for his wife's decision.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Figure 10: Implicit Inference. A qualitative demo on emotion understanding: the model is asked how the hero feels after learning that his wife chose suicide out of guilt, and answers that he feels desperation and remorse.

long-video specialization. Interestingly, the effect on LVBench is small, which indicates that some benchmark gains can still be obtained from later stages, but the broader long-video capability is noticeably weaker. Removing long-context training lowers the average to 65.4, with the largest drop on LongVideoBench (-3.0). This is consistent with the purpose of the short-to-long curriculum: the model needs explicit exposure to extended contexts in order to make effective use of the efficient attention structure. In other words, M2LA provides the capacity for longer reasoning windows, but long-context training is needed for the model to actually learn how to use them. Removing curated long-video data reduces the average to 64.8. The largest degradations appear on MLVU (-3.3) and LVBench (-2.8), while LongVideoBench remains unchanged in this setting. This suggests that long-video supervision provides complementary benefits beyond architecture-level context efficiency and context-length scaling: it improves the model’s ability to connect distant events, maintain temporally coherent state, and answer questions grounded in long-form narrative structure.

Overall, the ablation results support the design logic of the full recipe. Architectural conversion alone is not sufficient; it must be paired with recovery training, explicit long-context exposure, and

###### Inference Setting Video-MME

Direct QA 73.1 Agentic inference (+ MCR) 75.8

- Table 6: Preliminary comparison between direct QA and agentic inference on Video-MME. The agentic setting augments the base model with MCR at inference time.

broad long-video supervision. These components contribute in different ways, and their combination yields the strongest long-horizon performance.

Agentic Inference. In addition to the standalone model ablations above, we conduct a small preliminary study of agentic inference by using InternVideo3 with MCR at test time. On Video-MME, this increases performance from 73.1 to 75.8 (+2.7). This result is encouraging because it suggests that the model’s underlying long-horizon reasoning can be further improved when inference is allowed to revisit evidence and verify intermediate conclusions, consistent with the motivation of MCR.

At the same time, we do not present this as evidence of uniformly better performance across long-video benchmarks. In our current experiments, the agentic setting improves Video-MME but does not consistently help other benchmarks. We therefore view this result as a proof of concept rather than a general claim about agentic superiority. One plausible reason is that today’s public video benchmarks are still designed primarily for direct question answering and do not always reward multi-step retrieval, verification, or iterative evidence gathering. More systematic evaluation of multimodal agents will likely require benchmarks that explicitly measure such behavior.

Even so, the Video-MME result is useful in two ways. First, it shows that the MCR formulation is not only a conceptual framing for training, but can also be instantiated at inference time in a way that yields measurable gains. Second, it suggests that the direct QA improvements reported in the main benchmark tables are likely conservative with respect to the model’s full agentic potential, since current benchmark formats only partially expose the benefits of recursive evidence gathering.

###### 5.5. Discussion

Our experiments suggest that longer context alone is not enough. What matters is whether the model can use long context as an evolving reasoning substrate. The strongest gains of InternVideo3 appear not simply when more frames are available, but when success depends on maintaining a coherent account of what has happened, what evidence matters, and how observations from distant parts of the video relate to one another.

This is also why our improvements are most substantial on long-video benchmarks and selective temporal reasoning tasks rather than uniformly across all short-form tasks. We view this as evidence for the paper’s central claim: long-horizon multimodal agency is a distinct capability axis, and improving it requires more than scaling static multimodal QA.

##### 6. Conclusive Remarks

We presented InternVideo3, a framework for improving long-horizon multimodal reasoning through Multimodal Contextual Reasoning (MCR), efficient long-context attention, and staged post-training. Our central premise is that many video-centric and visually grounded tasks are better modeled not as one-shot multimodal question answering, but as a closed-loop process of evidence accumulation,

context updating, tool interaction, and self-correction. Under this view, multimodal understanding becomes a form of recursive contextual reasoning over evolving observations, intermediate conclusions, and external feedback.

To make such rollouts practical, we introduced Multimodal Multi-head Latent Attention (M2LA), which compresses KV-cache states while preserving the full multimodal token stream. We further developed a training recipe combining continued pretraining after attention conversion, short-to-long supervised fine-tuning, rule-based reinforcement learning, and on-policy distillation. Together, these components enable a moderate-scale open multimodal backbone to better handle long videos, dense temporal evidence, and extended reasoning trajectories.

Empirically, InternVideo3 achieves strong performance across both short-video and long-video benchmarks, with notable gains on long-horizon tasks such as Video-MME, MLVU, and EgoSchema. Relative to InternVideo2.5-7B, it substantially improves performance on several representative longvideo benchmarks, indicating that long-horizon capability can be strengthened through better context handling and long-video-oriented post-training even without moving to a much larger or newer backbone family. We also instantiate the model as a video agent with retrieval and verification tools, illustrating how recursive multimodal reasoning can support more robust evidence-grounded behavior.

More broadly, we view this work as a step toward multimodal agents rather than a claim of solving multimodal agency in full generality. Our setting is video-centric, but it highlights a capability that is likely to matter well beyond video: the ability to maintain, update, and verify a multimodal contextual state over long horizons. In this sense, we believe long-video and spatiotemporal reasoning provide a valuable testbed for studying visually grounded agency and world-model-relevant capabilities in foundation models.

At the same time, the open-weight ecosystem has evolved rapidly, and newer model families increasingly adopt stronger native long-context architectures and broader post-training recipes. We therefore view our contribution not as a claim of frontier-wide architectural finality, but as evidence for two broader lessons: first, that context efficiency remains crucial for long-horizon multimodal rollouts; and second, that closed-loop multimodal reasoning matters when models must perceive, remember, verify, and act over extended visual streams. We hope this work helps motivate further research on adapting open multimodal models into more capable and practically useful systems for long-horizon visually grounded tasks.

##### 7. Limitations, Discussion, and Future Work

Rapidly Evolving Open-Weight Frontier. A primary limitation of this work is that the open-weight multimodal ecosystem evolved rapidly during the course of the project. Newer model families such as Qwen3.5/3.6/3.7 (Team, 2026b), GLM (Zeng et al., 2026), Kimi (Team et al., 2026), StepFun (Huang et al., 2026), and others (DeepSeek-AI, 2026; mim, 2026; MiniMax Research, 2026; Team, 2026a) now surpass our system on a range of benchmarks and often incorporate stronger native long-context designs. Accordingly, this paper should not be read as claiming state-of-the-art multimodal agency or overall frontier open-model performance. Rather, it studies how far a moderate-scale open multimodal model can be pushed toward long-horizon visually grounded reasoning through better contextual reasoning, long-context efficiency, and post-training.

Specific Efficiency Path Versus Broader Efficiency Principle. Our long-context route centers on converting a pretrained GQA backbone into a latent-attention form through M2LA. This was a practical choice when the project began, but it is less directly applicable to newer models that already

incorporate native MLA (Liu et al., 2024a; Guo et al., 2025), linear attention (Team et al., 2025), or more hierarchical compressed-attention designs (DeepSeek-AI, 2026). Even so, we believe the broader principle remains important: long-horizon multimodal reasoning requires efficient context handling, and there is continued value in studying how pretrained attention can be adapted into more deployment-friendly forms.

Conceptual Novelty of MCR. The ideas underlying MCR, including closed-loop reasoning, tool use, memory, self-correction, and iterative evidence gathering, are now widely discussed in agent papers and system overviews (Masterman et al., 2024; Wang et al., 2024b; Zhang et al., 2025b; Ning et al., 2025, 2026). Our contribution is therefore less about inventing these ingredients from scratch and more about offering a clear and extensible formulation for long-horizon multimodal reasoning, especially in video-centric settings. We believe this formulation is useful for training, system design, and evaluation, even if many of its high-level ingredients are increasingly shared across the literature.

Implicit Rather Than Full Predictive World Modeling. We position MCR as a form of contextual belief-state modeling relevant to world-model-like capabilities (Bardes et al., 2024; Assran et al.,

- 2025), not as a full action-conditioned simulator of the environment (Hafner et al., 2019, 2023). This makes the approach practical for multimodal reasoning tasks, but it also limits its scope relative to predictive world models that explicitly model future environment dynamics. A promising direction is to connect contextual reasoning more tightly with learned predictive models so that multimodal agents can combine evidence gathering with explicit future-state evaluation.

Tool Dependence and System Fragility. The agent behavior depends on external tools such as ASR, temporal grounding, segmentation, retrieval, and search. Errors from these tools can propagate into the shared context and distort later reasoning. In the current work, tool interfaces are lightweight and not jointly optimized with the base model. Future work may explore tighter model–tool integration, uncertainty-aware tool use, and better calibration of tool-derived evidence.

Scope of the Current Evaluation. Our experiments focus primarily on long-video understanding, short-video reasoning, and a video-agent setup with perception tools. This covers an important but still partial slice of multimodal agency. We do not yet evaluate broadly on GUI agents (Xie et al., 2024; Bonatti et al., 2024), browser tasks (Zhou et al., 2024b; Koh et al., 2024), mobile agents (Deng et al., 2024; Rawles et al., 2025), embodied robotics (Shridhar et al., 2020; Yang et al., 2025d), or persistent multi-session assistants (Wu et al., 2024a; Maharana et al., 2024). Extending the same contextual reasoning framework to these settings is an important future direction.

Current public evaluation still under-measures many aspects of multimodal agency that motivate this work, such as recursive evidence gathering, memory management, verification quality, and closed-loop decision making over long visual streams. Meanwhile, the broader benchmark emphasis of the field has shifted strongly toward coding, browser, and general tool-use agents. We hope future benchmarks will better reflect visually grounded long-horizon reasoning as a distinct capability axis.

Future Directions. We see several promising directions for follow-up work. First, the idea of adapting pretrained attention into new efficient forms could be explored beyond GQA-to-MLA conversion, including hybrid, hierarchical, linear, or hardware-specialized attention designs. Second, MCR could be tested in broader multimodal agent settings such as GUI interaction, mobile agents, and embodied systems. Third, contextual reasoning should be studied together with stronger predictive world

models, enabling agents that not only gather and organize evidence, but also simulate and compare future outcomes. Finally, we believe there is continuing value in resource-efficient adaptation: a large part of the community does not train frontier-scale models from scratch, but instead works on making existing open weights more useful for new tasks, domains, and deployment constraints. We hope this work contributes to that broader effort.

##### References

Mimo-v2.5. https://huggingface.co/collections/XiaomiMiMo/mimo-v25, 2026. 25

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246– 21263, 2024. 3

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 2

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023. 7

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 3

Anthropic. System card. https://www-cdn.anthropic.com/ 963373e433e489a87a10c823c52a0a013e9172dd.pdf, 2024. Anthropic PDF document. Accessed: 2026-05-28. 16, 18

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141, 2024. 2

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023. 4

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 4, 26

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631. 2, 3, 16, 18, 19

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b. 2

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, 2025c. 2

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024. 4, 26

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024. 26

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, pages 961–970, 2015. 12, 16

Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Max Ehrlich, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. Advances in Neural Information Processing Systems, 38:91077–91100, 2026. 16, 18

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. CoRR, abs/2404.16821, 2024a. 3

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, pages 24185–24198, 2024b. 3

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, 2024a. 12

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. CoRR, abs/2406.07476, 2024b. 3

Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026. 12, 16

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with

advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 16, 18, 19

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. 4, 25, 26

Shihan Deng, Weikai Xu, Hongda Sun, Wei Liu, Tao Tan, Liujianfeng Liujianfeng, Ang Li, Jian Luan, Bin Wang, Rui Yan, et al. Mobile-bench: An evaluation benchmark for llm-based mobile agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8813–8831, 2024. 26

Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memoryaugmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92. Springer, 2024. 4

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 3, 13, 15

Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, et al. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026. 15

Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, pages 5267–5275, 2017. 16

Google DeepMind. Gemini 3 pro model card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-Pro-Model-Card.pdf, 2025. Model card. Accessed: 2026-05-28. 16, 18, 19

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 2

Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Bohua Chen, Yi Su, Dongping Chen, Siyuan Wu, Xing Zhou, et al. Webcode2m: A real-world dataset for code generation from webpage designs. In Proceedings of the ACM on Web Conference 2025, pages 1834–1845, 2025. 12

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 26

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14953–14962, 2023. 3

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3):440, 2018. 1, 4

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019. 4, 26

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 26

Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pages 47376–47405,

2024. 2 Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022. 2

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300,

2020. 12

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4246–4253, 2020. 8

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281– 14290, 2024. 4

Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, and Jie Tang. Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8450–8460, 2025a. 16

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025b. 16, 18

Ailin Huang, Ang Li, Aobo Kong, Bin Wang, Binxing Jiao, Bo Dong, Bojun Wang, Boyu Chen, Brian Li, Buyun Ma, et al. Step 3.5 flash: Open frontier-level intelligence with 11b active parameters. arXiv preprint arXiv:2602.10604, 2026. 25

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 12

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025. 2, 4

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 881–905, 2024. 26

Deqian Kong, Dehong Xu, Minglu Zhao, Bo Pang, Jianwen Xie, Andrew Lizarraga, Yuhao Huang, Sirui Xie, and Ying Nian Wu. Latent plan transformer for trajectory abstraction: Planning as latent space inference. Advances in Neural Information Processing Systems, 37:123379–123401, 2024. 2

Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 27831– 27840, 2024. 12

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018. 2, 12

Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. NeurIPS, 34:11846–11858, 2021. 16

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a. 3, 12

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b. 2

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 3

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024c. 3, 16

Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024d. 16, 18, 19

Jingli Lin, Runsen Xu, Shaohao Zhu, Sihan Yang, Peizhou Cao, Yunlong Ran, Miao Hu, Chenming Zhu, Yiman Xie, Yilin Long, et al. Mmsi-video-bench: A holistic benchmark for video-based spatial intelligence. arXiv preprint arXiv:2512.10863, 2025. 16

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755. Springer, 2014. 12

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024a. 26

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, volume 36, 2024b. 3

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024c. 2, 16

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation. 3

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arxiv, 2024. URL https://arxiv.org/abs/2406.09418. 12

Marlos C Machado, Andre Barreto, Doina Precup, and Michael Bowling. Temporal abstraction in reinforcement learning with the successor representation. Journal of machine learning research, 24

(80):1–69, 2023. 2

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, 2024. 26

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 2, 3, 15

U-V Marti and Horst Bunke. The iam-database: an english sentence database for offline handwriting recognition. International journal on document analysis and recognition, 5(1):39–46, 2002. 12

Ahmed Masry, Parsa Kavehzadeh, Enamul Hoque, Shafiq Joty, et al. Unichart: A universal visionlanguage pretrained model for chart comprehension and reasoning. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 14662–14684, 2023. 12

Tula Masterman, Sandi Besen, Mason Sawtell, and Alex Chao. The landscape of emerging ai agent architectures for reasoning, planning, and tool calling: A survey. arXiv preprint arXiv:2404.11584,

2024. 26

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 12

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 2

MiniMax Research. Minimax m3: Frontier coding capabilities, 1m context, and native multimodality in one model. https://www.minimaxi.com/blog/minimax-m3, June 2026. MiniMax blog, accessed 2026-06-05. 25

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-jepa 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026. 4

Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S Yu, et al. A survey of webagents: Towards next-generation ai agents for web automation with large foundation models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 6140–6150, 2025. 26

Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, Yuanchen Bei, Jiaru Zou, Mengting Ai, Zhining Liu, Ting-Wei Li, et al. Code as agent harness. arXiv preprint arXiv:2605.18747, 2026. 26

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023. 1

Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. volume 36, 2024. 16

Guilherme Penedo, Hynek Kydlíček, Amir Hossein Kargaran, and Leandro von Werra. Finetranslations. https://huggingface.co/datasets/HuggingFaceFW/finetranslations, 2026. 12

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, volume 2024, pages 9695–9717, 2024. 2

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3

Chris Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations, volume 2025, pages 406–441, 2025. 26

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard Ives, James Keeling, Karel Lenc, Salem Haykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. CoRR, abs/2403.05530, 2024. 4

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023. 1, 2

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. In International Conference on Learning Representations, volume 2025, pages 7593–7734, 2025. 16

Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 4

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023. 1

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020. 26

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 16, 18, 19

Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 4

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 8

Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488,

2018. 2

Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11888–11898, 2023. 3

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025. 26

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276,

2026. 25 Meituan LongCat Team. Longcat-next: Lexicalizing modalities as discrete tokens, 2026a. URL https://arxiv.org/abs/2603.27538. 25 Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026b. URL https://qwen.ai/blog?id=qwen3.5. 25

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024a. 4

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024b. 26

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024c. 2, 3, 15

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025a. 2, 16, 18, 19

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024d. 4

Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In International Conference on Learning Representations, volume 2025, pages 65882–65919, 2025b. 2, 4

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 3

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 12, 13

Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. In ECCV, 2024e. 3

Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025c. 3, 16, 18, 19

Yizhou Wang, Chen Tang, Han Deng, Jiabei Xiao, Jiaqi Liu, Jianyu Wu, Jun Yao, Pengze Li, Encheng Su, Lintao Wang, et al. Scireasoner: Laying the scientific reasoning ground across disciplines. arXiv preprint arXiv:2509.21320, 2025d. 12

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 4

Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024. 12

Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andrés Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025. 12

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024a. 26

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024b. 2, 15

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 5

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 16

Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In CVPR, 2024. 2

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024. 26

Wenda Xu, Rujun Han, Zifeng Wang, Long Le, Dhruv Madeka, Lei Li, William Wang, Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative knowledge distillation: Bridging the teacher-student gap through interleaved sampling. In International Conference on Learning Representations, volume 2025, pages 64616–64646, 2025. 3

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. 8, 15

Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025b. 16, 18, 19

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025c. 16

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024. 2, 4

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025d. 26

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis L Brown II, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. In The Fourteenth International Conference on Learning Representations, 2025e. 12

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025f. 16

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022a. 2

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022b. 1, 2, 3

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023. 4

Jiashuo Yu, Yue Wu, Meng Chu, Zhifei Ren, Zizheng Huang, Pei Chu, Ruijie Zhang, Yinan He, Qirui Li, Songze Li, et al. Vrbench: A benchmark for multi-step reasoning in long narrative videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21655–21666, 2025a. 15

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, pages 69–85. Springer, 2016. 12

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations, volume 2024, pages 45040–45061, 2024. 12

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025b. 16, 18, 19

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026. 25

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019. 8

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–20, 2025a. 4

Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, and Limin Wang. Timelens: Rethinking video temporal grounding with multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10419–10429, 2026. 14

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. ∞ Bench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262–15277, 2024a. 2

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024b. 12, 16, 18

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47, 2025b. 26

Ziang Zhang, Zehan Wang, Guanghao Zhang, Weilong Dai, Yan Xia, Ziang Yan, Minjie Hong, and Zhou Zhao. Dsi-bench: A benchmark for dynamic spatial intelligence. arXiv preprint arXiv:2510.18873, 2025c. 16

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 414–431, 2025. 2, 4

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024a. 3, 15

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606, 2024b. 2, 26

