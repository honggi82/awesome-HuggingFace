# arXiv:2606.21337v1[cs.LG]19Jun2026

## DataClaw0: Agentic Tailoring Multimodal Data from Raw Streams

Cong Wan1∗, Zeyu Guo1∗, Zijian Cai2∗, Jiangyang Li1∗, SongLin Dong3, Lin Peng1 Xiangyang Luo4, Zhiheng Ma3, Yihong Gong1,

1Xi’an Jiaotong University 2University of Chinese Academy of Sciences 3Shenzhen University of Advanced Technology 4Tsinghua University

### Abstract

Massive unstructured multimodal streams suffer from high "data entropy," impeding both efficient human knowledge acquisition and high-quality AI post-training. Existing passive annotation paradigms, heavily reliant on heuristic rules or general VLMs, are costly, monotonous, and fail to unlock the deep procedural logic embedded in raw data. We elevate data processing to a learnable capability, proposing a paradigm shift towards Agentic Data Tailoring, which actively refining and structuring data to align with diverse user and downstream intents. To overcome the data scarcity bottleneck in training such high-order capabilities, we design a two-stage pipeline grounding generative semantic synthesis in deterministic Factual Anchors, yielding a large-scale dataset spanning five core physical and digital domains. Building upon this, DataClaw0-9B model synergizes Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), achieving robust alignment with complex refinement and tailoring intents. To systematically quantify this capability, we construct DataClaw0-val, the first benchmark dedicated to data refinement. Crucially, we adopt downstream post-training as the ultimate validation touchstone. Evaluations on video generation, real-world VQA, and GUI navigation confirm that DataClaw0 delivers high-information-density tailored data, facilitating efficient model adaptation to new tasks under limited training data regimes. Project page: https://czjdsg.github.io/MakeAnyData

### 1 Introduction

The rapid evolution of multimodal foundation models is increasingly bottlenecked by a critical scarcity of high-quality training data [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]. While the physical and digital worlds generate an inexhaustible supply of raw multimodal streams—such as hours-long tutorial videos, embodied agent trajectories, and complex Web and GUI operation logs [13, 14, 15, 16, 17, 18]—harvesting this resource presents a formidable challenge. The core obstacle is extreme “data entropy”. Unlike curated image-text pairs, raw multimodal streams are inherently noisy, highly redundant, and weakly structured. They contain dense physical dynamics, procedural knowledge, and implicit decision logic [13, 19, 20, 14, 21, 18], but lack the explicit supervision signals required for efficient knowledge acquisition or high-quality model post-training [22, 23, 24, 25]. Consequently, efficiently distilling continuous, high-entropy streams into structured, high-density knowledge has become the most pressing imperative in multimodal data engineering.

Existing data processing pipelines remain largely passive, relying on heuristic sampling, coarse captioning, or directly prompting general Vision-Language Models (VLMs) to generate captions and question-answer pairs [24, 26, 27]. While effective for short and curated inputs, these methods

Preprint.

[Figure 1]

- Figure 1: The figure shows representative tailoring cases. In each panel, the top clips denote the

raw inputs, followed by a user Intent that specifies a construction goal. DataClaw0 then selects task-relevant visual evidence under Task, formulates a corresponding Question, and decomposes the solution into intermediate Steps with aligned images and textual reasoning/action descriptions. The bottom part gives the final structured outcome, including an Answer when applicable and a Video Output that preserves or reconstructs the tailored visual sequence.

struggle with long, noisy streams that require temporal reasoning, spatio-temporal consistency, and implicit physical understanding [28, 29, 30, 31, 32, 33]. Direct annotation therefore often yields hallucinated, fragmented, or low-density outputs, leaving much of the latent value in raw multimodal data untapped [34, 35]. This motivates us to recast high-quality data production as a learnable, high-order capability, termed Agentic Data Tailoring: given a high-level user intent or downstream training objective, a tailoring agent actively filters redundant information, identifies task-critical evidence, and reorganizes it into dense, verifiable, and application-specific supervision. Unlike general data curation or synthetic instruction generation [36, 37, 38, 39, 40, 23, 41], agentic data tailoring focuses on intent-conditioned entropy reduction over continuous multimodal streams. This paper asks whether such a capability can be formally defined, rigorously evaluated, and effectively learned by compact open-source VLMs.

To answer this, we must first overcome the data paradox: training a model to refine data requires highquality refinement data. We propose a scalable, automated construction pipeline based on a two-stage

agentic architecture. Our first stage extracts deterministic Factual Anchors using lightweight domain experts, metadata parsers, and heuristic rules. These anchors provide reliable low-level grounding, such as object states, temporal boundaries, OCR text, and GUI interaction events [14, 15, 42, 43, 44]. In the second stage, strong VLMs perform long-range logical chaining over these discrete anchors, injecting multi-dimensional reasoning traces inspired by multimodal chain-of-thought [45, 46, 47]. This bottom-up extraction and top-down synthesis strategy yields a massive, cross-domain refinement dataset spanning five representative arenas: daily life, education, embodied intelligence, world models, and GUI agents [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59].

Leveraging this dataset, we introduce DataClaw0, a universal framework for agentic multimodal data tailoring as shown in Fig.1. At its core, DataClaw0 optimizes Qwen3.5-9b to transform high-entropy streams into intent-aligned, customized structured outputs. Moving beyond standard Supervised Fine-Tuning (SFT), DataClaw0 employs Group Relative Policy Optimization (GRPO)[60] to directly optimize for tailoring quality and intent adherence. The reinforcement learning phase utilizes multi-dimensional reward signals that measure intent satisfaction, information density, factual consistency, and structural correctness, drawing inspiration from recent reasoning-oriented alignment techniques[61, 62, 63, 64]. To accommodate diverse deployment needs, we design two complementary architectures: DataClaw0-O (an Omni model jointly trained across all domains) and DataClaw0-E (an Expert-style system comprising decoupled, domain-specific tailoring agents).

To systematically evaluate this capability, we construct DataClaw0-val (including an Intent subset for vague-intent concretization), a benchmark dedicated to agentic data refinement. While our 9B model demonstrates superior refinement quality over existing VLMs on this benchmark, we also introduce a Targeted Refinement evaluation across video generation, real-world VQA, and GUI navigation [13, 65, 66]. Empirical results reveal that models trained on DataClaw0’s highly compressed, task-tailored subsets consistently outperform those trained on conventional full-scale datasets while drastically reducing computational costs.

In summary, our core contributions are: (1) We elevate multimodal data processing to a learnable capability of Agentic Data Tailoring, proposing a two-stage pipeline of deterministic anchor extraction and generative semantic synthesis to build a large-scale cross-domain dataset, and introducing a novel benchmark to reveal profound deficiencies in existing general VLMs. (2) We propose DataClaw0, a universal tailoring framework synergizing SFT with GRPO, and explore both Omni and Expert deployment paradigms to demonstrate this high-order capability’s learnability and scalability within a 9B model. (3) We introduce a Targeted Refinement setting, employing downstream post-training across video generation, real-world VQA, and GUI navigation as an objective touchstone to definitively prove DataClaw0 dynamically tailors vastly superior training subsets from raw streams at drastically reduced data and computational costs.

### 2 Related Work

Multimodal Large Language Models. Large language models (LLMs) have demonstrated strong reasoning, instruction-following, and generalization capabilities in text-only scenarios [5, 67, 68]. By integrating visual encoders, multimodal projectors, and language backbones, multimodal large language models (MLLMs) extend these capabilities to visual perception, grounding, and multimodal reasoning [3, 27, 1, 69, 4, 7]. Recent MLLMs have rapidly evolved from image-level assistants to general-purpose multimodal systems, covering high-resolution perception, OCR, document understanding, multi-image reasoning, video understanding, spatial reasoning, and long-context multimodal interaction [43, 70, 42, 28, 29, 30]. Beyond generic perception and reasoning, MLLMs are increasingly adapted to downstream scenarios in both digital and physical environments, including GUI interaction, web/mobile automation, world model, and vision-language-action modeling [21, 18, 14, 15, 56, 57, 58, 59, 52, 71, 72, 73, 74]. These applications require models to interpret complex observations, localize task-critical entities, understand temporal or procedural dependencies, and sometimes transform multimodal states into executable actions. However, the continued scaling of MLLMs is increasingly constrained by data. High-quality multimodal supervision is costly to obtain, while raw multimodal streams, such as tutorial videos, embodied trajectories, 3D scans, and GUI operation logs, are usually noisy, redundant, weakly structured, and poorly aligned with downstream training objectives [13, 14, 15, 21, 18]. Existing passive annotation pipelines, including heuristic sampling, coarse captioning, and direct VLM-based question-answer generation, have enabled early multimodal instruction tuning [24, 26, 27], but they struggle to extract dense and

[Figure 2]

- Figure 2: Overview. The pipeline consists of two parts. First, DataClaw0 constructs training data by extracting bottom-up factual anchors from raw multimodal data, including key frames, steps, actions, trajectories, and scenes, and then combining them with domain-specific intents for top-down semantic synthesis by a strong VLM. This produces large-scale structured data across daily life, education,

embodied, GUI-agent, and AIGC domains. Second, the constructed data is used to train DataClaw0 models under two paradigms: an omni model trained on mixed-domain data and expert models trained on per-domain subsets. During inference, user intents are handled either directly by the omni model or routed to domain experts, yielding refined structured data for downstream multimodal tasks.

reliable supervision from long, high-entropy streams, especially under temporal reasoning, spatial consistency, and hallucination-sensitive settings [34, 35, 31, 32, 33]. Therefore, scalable transformation of raw multimodal data into structured, task-aligned supervision has become a central bottleneck for pursuing the ceiling of general intelligence.

Agentic AI. Recent LLM agents extend language models from passive response generators to goal-driven systems that can plan, use tools, interact with environments, execute multi-step workflows, and refine intermediate results through feedback [75, 76, 77, 78, 79, 80]. This agentic paradigm has attracted broad attention in both research and open-source communities, with systems and frameworks such as OpenAgents, LangChain, AutoGPT, and OpenClaw enabling language-model agents to orchestrate tools, access external resources, automate user workflows, and operate across heterogeneous digital environments [81, 82, 83, 84]. Beyond task execution, agentic workflows have also been explored for data-centric applications, including synthetic instruction generation, selfimprovement, and high-quality reasoning data construction [38, 39, 40, 85]. These studies suggest that agents can serve not only as executors, but also as scalable organizers of complex data-generation pipelines: they can decompose ambiguous objectives, call specialized tools, inspect intermediate outputs, correct errors, and verify results. Inspired by this observation, DataClaw0 formulates MLLM data synthesis as an agentic workflow. Instead of manually annotating or directly captioning raw streams, DataClaw0 organizes multimodal models and tools, into a data-synthesis agent that converts high-entropy multimodal sources into structured, intent-aligned supervision for training models.

- 3 Method

#### 3.1 Problem Formulation: Agentic Data Tailoring

Traditional multimodal tasks (e.g., Video Captioning or Visual Question Answering) are typically modeled as passive descriptive or question-answering processes. In contrast, DataClaw0 aims to tackle the problem of Agentic Multimodal Data Tailoring. This requires the model to act as an

intelligent agent that actively filters, reasons over, and reorganizes high-value structured assets from lengthy, noisy raw multimodal streams, strictly guided by specific high-level intents.

We define the input as a raw multimodal data stream:

Xraw = {x1,x2,...,xT} (1)

where xt represents the visual frame or multimodal segment at time step t (e.g., a sequence of frames from a long video or GUI operation screenshots). Simultaneously, an intent instruction I is provided to represent the user’s high-level objective or the downstream task requirement.

The goal of the tailoring agent is to generate customized, structured knowledge assets:

Ystruct = {y1,y2,...,yL} (2)

Unlike free-form text, Ystruct must strictly conform to a predefined structural schema Φ (e.g., a specific JSON format, Markdown syntax, or action code logic) tailored to the intent I.

Therefore, the optimization objective of the core DataClaw0 tailoring agent Fθ (parameterized by θ) is to maximize the conditional generation probability given the raw data stream and intent, constrained by the structural schema:

θ∗ = arg max

log P(Ystruct | Xraw,I;θ) · I(Ystruct ∈ Φ) (3)

θ

(Xraw,I,Ystruct)∈D

where D is the training dataset, and I(·) is an indicator function that equals 1 if the generated sequence conforms to the structural schema Φ, and 0 otherwise.

This paradigm necessitates two core agentic capabilities: (1) Information Filtering and Focusing, which involves eliminating redundant background noise from Xraw based on I (where T ≫ L); and (2) Structural Reorganization, ensuring that the generated Ystruct is not only semantically accurate but also strictly adheres to the required formatting.

#### 3.2 Data Construction Pipeline

To train the tailoring agent Fθ, we construct large-scale triplets (Xraw,I,Ystruct) through a twostage automated pipeline. First, a lightweight expert ensemble H extracts factual anchors from raw multimodal streams:

A = H(Xraw) = {ak = (tk,pk,ck)}Kk=1, (4) where each anchor records timestamp, spatial position, and local semantic content. These anchors provide reliable grounding signals and reduce hallucinations in long-sequence annotation.

Second, a strong VLM synthesis engine S generates structured supervision conditioned on the raw input, extracted anchors, and domain intent:

Ystruct = S(Xraw,A,Idomain). (5) The resulting corpus covers multiple multimodal domains and serves as the foundation for subsequent SFT and RL training of DataClaw0. Detailed construction procedures, expert modules, and prompting strategies are provided in Appendix A.3.

#### 3.3 Rule-Driven Reinforcement Learning via GRPO

After SFT, we further optimize DataClaw0 with rule-driven GRPO to improve formatting reliability, reduce hallucinations, and strengthen spatio-temporal grounding. Instead of training an additional neural reward model, we use deterministic rewards tailored to structured multimodal data:

R(Y ) = λ1Rformat(Y,Φ) + λ2Ranchor(Y,A) + λ3Reff(Y ), (6)

where Rformat checks schema compliance, Ranchor measures alignment with extracted factual anchors and trajectories, and Reff discourages overly verbose reasoning.

Given a group of sampled outputs, GRPO normalizes their rewards within the group to estimate relative advantages:

R(g) − µR σR

Aˆ(g) =

. (7)

The policy is then updated with a clipped objective and a KL regularizer against the reference model. This SFT-initialized, rule-reward optimization enables DataClaw0 to produce more valid, grounded, and concise structured outputs. Detailed reward definitions and the full GRPO objective are provided in Appendix B.1.

#### 3.4 Inference and Deployment Paradigms

DataClaw0 is deployed as a structured multimodal tailoring agent that maps raw streams Xraw and user intents I to schema-aligned outputs Ystruct. Its inference pipeline follows three stages: multimodal ingestion and intent parsing, schema-constrained policy inference, and post-hoc grounding verification with factual anchors A. This design improves output validity and keeps the generated structured data grounded in the input stream.

DataClaw0 supports two deployment paradigms. DataClaw0-O uses a unified omni model for flexible cross-domain processing, while DataClaw0-E uses a domain-decoupled expert architecture, where each request is handled by the corresponding expert according to the target scenario or deployment configuration. The omni setting favors simplicity and generality, whereas the expert setting provides stronger domain specialization and modular scalability.

Detailed inference architecture and deployment mechanisms are provided in Appendix B.2.

#### 3.5 Benchmark Construction and Evaluation

We construct DataClaw0-val by diversity-aware sampling 200 high-quality examples, ensuring coverage of diverse multimodal inputs, target schemas, and long-tail cases. We also introduce

DataClaw0-Intent, a fuzzy-intent stress test that evaluates whether the agent can infer underspecified user intents from colloquial, ambiguous, or incomplete requests. We evaluate outputs with a hierarchical metric tailored to structured multimodal data. The metric first enforces JSON validity and then measures schema-field correctness, textual semantic alignment, and trajectory-shape similarity. Full construction details and metric definitions are provided in Appendix A.4.

### 4 Experiments

#### 4.1 Experimental Setup

Implementation Details: We initialize DataClaw0 based on an open-source multimodal large language model with 9B parameters (using the latest Qwen3.5-9B as the base). During the SFT phase, we utilize 34K strictly cleaned instruction refinement data for 1 epoch. In the GRPO reinforcement learning phase, we set the format reward weight λfmt = 0.7, the physical anchor reward weight λanc = 1.0, the reasoning efficiency penalty weight λeff = 0.3, and the learning rate to 4 × 10−6. All training is conducted on 8× A100 GPUs.

#### 4.2 Main Results: Specialist vs. Generalist

Table 1 compares DataClaw0 with leading proprietary and open-source MLLMs across five representative domains and the Fuzzy instruction subset. We evaluate three aspects of structured data synthesis: Field for schema completeness, Semantic for content correctness, and Sequence for ordering and structural consistency.

Overall, DataClaw0-E achieves strong performance among all evaluated models. In terms of schema following, it obtains an overall Field score of 97.53, comparable to Gemini-3.1-pro-preview (98.12)

and GPT-4o (97.27). More importantly, DataClaw0-E is particularly competitive on sequencesensitive tasks. It achieves the best Sequence score on the Embodied domain (71.60) and the Fuzzy subset (50.31), and remains strong on GUI (96.33). These results show that domain-specific refinement is effective for producing well-structured and temporally consistent annotations.

The comparison between DataClaw0-O and DataClaw0-E further validates the benefit of specialization. DataClaw0-E improves over the omni variant on overall Field (97.53 vs. 87.65), Semantic (74.94 vs. 62.46), and Sequence (48.86 vs. 44.82), with consistent gains on the Fuzzy subset. This suggests that routing heterogeneous multimodal streams to domain-specific experts is more effective than relying on a single general-purpose annotator.

- Table 1: Comparison of DataClaw0 against state-of-the-art MLLM models. We evaluate models across five distinct domains and a newly introduced Fuzzy instruction subset. The best results in each metric per column are highlighted in bold.

Domains GUI Embodied AIGC Daily Education Fuzzy Overall

Model Metric

Field 57.50 100.00 100.00 100.00 100.00 76.35 88.98 Semantic 50.05 84.83 74.26 54.46 54.46 65.72 63.96 Sequence 54.06 50.11 33.38 41.07 41.07 36.48 42.70

Claude-Sonnet-4-6

Field 100.00 100.00 100.00 100.00 100.00 83.61 97.27 Semantic 84.81 87.55 69.38 54.58 80.21 74.39 75.15 Sequence 80.69 46.33 46.95 29.33 50.71P 42.57 49.43

GPT-4o (1120-global)

Field 100.00 100.00 100.00 100.00 100.00 88.74 98.12 Semantic 90.01 89.17 75.26 54.51 54.51 79.63 73.85 Sequence 99.67 67.97 33.14 51.48 51.48 47.25 58.50

Gemini-3.1-Pro-Preview

Field 92.50 100.00 100.00 97.50 95.00 73.29 93.05 Semantic 78.93 79.28 69.07 44.38 43.52 61.85 62.84 Sequence 77.86 51.84 6.83 32.89 17.54 32.16 36.52

MiniMax-M2.7

Field 70.00 100.00 95.00 95.00 82.50 77.58 86.68 Semantic 62.64 87.40 67.82 51.20 62.14 64.37 65.93 Sequence 66.96 60.33 39.71 42.44 30.85 35.92 46.04

Qwen3.6-plus

Field 94.87 100.00 95.00 87.50 90.00 70.46 89.64 Semantic 72.72 77.48 65.27 45.66 43.41 58.24 60.46 Sequence 72.71 59.35 3.29 27.70 24.75 29.63 36.24

Qwen3.5-9B

Field 100.00 100.00 85.00 92.50 70.00 78.42 87.65 Semantic 80.01 63.37 55.70 62.61 45.71 67.35 62.46 Sequence 85.70 67.01 23.90 35.05 17.41 39.84 44.82

DataClaw0-O [Ours]

Field 100.00 100.00 100.00 100.00 100.00 85.17 97.53 Semantic 89.18 82.93 75.36 49.72 76.43 76.28 74.94 Sequence 96.33 71.60 15.26 42.59 19.75 50.31 48.86

DataClaw0-E [Ours]

Meanwhile, proprietary MLLMs still show advantages in some semantic-heavy settings, benefiting from broader pretraining and stronger open-ended reasoning. For example, GPT-4o obtains the best overall Semantic score (75.15), slightly higher than DataClaw0-E (74.94). Nevertheless, DataClaw0E achieves comparable semantic performance while offering stronger controllability and competitive sequence consistency, making it an effective specialized framework for multimodal data synthesis.

#### 4.3 Downstream Application: Targeted Refinement & Efficiency

We evaluate whether DataClaw0-generated data can effectively improve downstream multimodal models. To this end, we conduct SFT experiments on three representative tasks: long-horizon GUI navigation on AgentNet [86], action video generation on Ego4D [13], and spatio-temporal VQA on ReMoT [65]. For each task, we select strong open-weights base models that match the target modality and task format: Qwen3.5-4B for the understanding-oriented GUI and VQA tasks, and Wan2.2-I2V-5B [11] for image-to-video generation. The evaluation metrics are selected according to the task focus and benchmark protocols. For GUI navigation, we report Step Success Rate (SSR) and Task Success Rate (TSR), which measure local action correctness and end-to-end task completion. For action video generation, we use the standard Fréchet Video Distance (FVD) for overall video quality, and additionally report temporal consistency and Contact mAP, since our setting emphasizes physically plausible affordance and action-object interaction. For spatio-temporal VQA, we follow the official ReMoT protocol and report Partial Accuracy and Overall Accuracy. To isolate data quality, we construct SFT data from the same raw streams for each task. Specifically, the identical inputs are processed by three sources: Qwen3.5-9B, Gemini-3.1-Pro-Preview, and DataClaw0. We then apply the same filtering procedure and sample an equal number of valid instances from each source, ensuring that the comparison focuses on annotation quality rather than data quantity.

As shown in Table 2, self-processed data brings only limited gains over the zero-shot base models, while data generated by stronger annotators substantially improves downstream performance. DataClaw0 achieves performance comparable to Gemini overall, and obtains better results on several

- Table 2: Downstream Application Performance across Three Distinct Tasks. We compare the effectiveness of SFT data generated by different sources under strict volume alignment. Best results are in bold.

GUI Navigation Action Video Generation Spatio-temporal VQA

Data Source for SFT

(Base: Qwen3.5-4B) (Base: Wan2.2-I2V-5B) (Base: Qwen3.5-4B)

SSR (%) ↑ TSR (%) ↑ FVD ↓ Consis. (%) ↑ Contact mAP ↑ Partial Acc. (%) ↑ Overall Acc. (%) ↑ Zero-shot Base Model 12.4 1.2 385.2 68.4 18.5 28.3 9.8

Processed by Base Model 16.8 3.5 362.1 69.1 24.2 33.5 14.2 Processed by Gemini-3.1-Pro 39.5 14.2 295.4 76.2 48.5 53.4 31.5 Processed by DataClaw0 38.2 15.6 288.6 75.8 51.2 52.1 33.2

Lifestyle, Food & Ent. 100

[Figure 3]

Product Display 90

Interaction 800

DvToo&SymMgm 270

Resea ch &

Scheduling &

Anticipation 450

Pred.

Infor

Travel

Observation 440

me

mation &

333

bject

Keyfra

Planning &

Shopping &

Ve

460

500

Scene-Action

482

Collaboration

O

ca on

707

Future

Hand-

rounding

Mobile App &

Notes

758

Booking

Co

mparison

Su

Device Mg mt

970

AIGC 2,240 (6.5%)

Content Creation &

1,612

GUI 5,732

Docs

Home Repair & DIY 7,955

Near-Stal Recovery

(16.5%)

1,475

Next Subtask Pred. 1,476

DataClaw 34,717

Embodied 7,381 (21.3%)

[Figure 4]

Illustrated Tutorial 1,476

Daily Life 16,121 (46.4%) Education 3,243 (9.3%)

Hesitation Recovery 1,477

Curent Task Description

First-Person Indoor Nav.

1,477

Extraction 593

4,933

→ Tutorial 674

Dificulty

L

M

T

wn

Lecture

H

Markdo

→

988

Lecture

uaeB y & S y

Spo&Fne 590

Cooking

→

988

2,335

803

Lecture

gn

- Figure 3: Overview of the DataClaw0 data mixture and scaling behavior. Left: domain and subtask distribution of the constructed samples. Right: Comparison of scaling curves for DataClaw0-E and DataClaw0-O, and t-SNE visualization of DataClaw0-E, the raw data, and Qwen3.5-9B.

end-to-end metrics. These results indicate that DataClaw0 can produce compact and task-relevant supervision that transfers effectively to downstream models under the same data budget.

Overall, the downstream evaluation demonstrates that DataClaw0 is not merely producing valid structured annotations, but generating high-utility training data for targeted model refinement. More detailed experimental settings and analysis are provided in Appendix C.1.

#### 4.4 Scaling Laws and Emergent Diversity

We study the scaling behavior and emergent diversity of DataClaw0 to evaluate the necessity of domain-decoupled expert routing. As summarized in Figure 3, the unified DataClaw0-O model shows unstable scaling under mixed-domain training, suggesting severe task interference among heterogeneous multimodal extraction tasks. By contrast, DataClaw0-E mitigates this issue through expert routing, enabling each domain expert to specialize on its local data distribution.

We also observe that DataClaw0 improves semantic diversity rather than merely replicating training patterns. Feature-space analysis and fuzzy-intent evaluation show that DataClaw0 covers a broader intent space and remains robust to ambiguous, colloquial, or incomplete requests. Full analysis is reported in Appendix D.1.

#### 4.5 Ablation Studies

We conduct ablations on reward design and expert routing in Table 3. SFT establishes basic instruction following and structured generation, raising the Field score to 100.00 and largely improving Semantic

- Table 3: Ablation studies on reward design and expert routing. Reward ablation uses the 10% SFT initialization. Routing ablation is conducted on Embodied and GUI tasks by comparing correct routing with forced wrong routing.

(a) Reward Design Variant Field Sem. Seq.

Minimal Init. 82.50 36.79 45.40 SFT Only 100.00 82.54 70.83 GRPO w/o Ranchor 100.00 83.32 70.11 GRPO w/ Ranchor 100.00 82.36 71.96

###### (b) Expert Routing Domain (Expert) Field Sem. Seq.

Embodied (Gui) 0.00 0.00 50.00 Embodied (Emb) 96.50 74.21 63.48 GUI (Gui) 100.00 84.93 76.41 GUI (Emb) 0.00 52.55 0.00

and Sequence scores. GRPO without Ranchor slightly increases Semantic performance but lowers Sequence accuracy, suggesting that format- or text-oriented rewards alone may not preserve spatio-

temporal fidelity. With Ranchor, the model obtains the best Sequence score of 71.96, confirming the necessity of explicit spatial-temporal grounding. Ablation results of expert routing demonstrate that accurate expert selection is indispensable. The GUI and embodied experts possess strong domainspecific capabilities, with each excelling only in its own task domain and suffering severe performance degradation when applied to the other.

#### 4.6 Qualitative Analysis

As illustrated in Figure 4, across both robot manipulation and GUI reconstruction scenarios, DataClaw0-E outperforms baseline methods by accurately identifying key behavioral patterns, selecting temporally consistent evidence, eliminating irrelevant trajectory fragments, and establishing complete, structured task supervision; by contrast, baseline approaches suffer from clip mismatches, missing critical fields, retained distractors, and unstructured sequence outputs.

###### You are given some short-horizon GUI trajectory fragments. Your task is to identify a plausible long-horizon GUI task that can be composed from them, infer the correct fragment order, and output the detailed reconstructed task.

###### Please analyze this robot manipulation video and create one example of *Hesitation Loop Fault Diagnosis and Recovery* data.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

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

[Figure 22]

[Figure 23]

Fragment A：Open Google Chrome, use Google Lens to upload the saved image, find a similar cartoon character image, and save it to Downloads folder as \"Avatar\". Step 1: ......

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

16

0 9 10 11 12 24 28 30

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- Fragment D：Export document as PDF to Desktop. Step 1 : ......
- Fragment E：Open System Settings, navigate to Users section, select the saved Avatar.jpeg file from Downloads, and set it as the system avatar. Step 1: ......
- Fragment F：Open Thunderbird, navigate to Trash folder, open the deleted email, and save the cartoon character image to the desktop. Step 1: ......

{

[Figure 36]

[Figure 37]

[Figure 38]

"question": "Why are the robotic arms repeatedly moving slightly up and down over the box without gripping the mouse

and cable?", "input_video": { "start_frame": 0, "end_frame": 12, "loop_start_frame": 11, "loop_end_frame": 12,"loop_repeat_count": 2}, ……

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

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

}

In the constructed input video, the robotic arm has already grasped the mouse.️

[Figure 56]

{ 0 11 12 11’ 12’ 11‘’ 12’’ "start_frame": 4, "end_frame": 14, "loop_start_frame": 13, "loop_end_frame": 14, "loop_repeat_count": 2,

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

} { The constructed data does not include questions or answer️

{

"question": "Why is the robot's right arm repeatedly moving back and forth over the box without dropping the mouse?", "answer": "The robot is stuck in a hesitation loop. Its right arm keeps swinging, preventing it from completing the grasping operation. To restore normal operation, return to a stable approach state, firmly grasp the mouse, smoothly place it into the box, and then release it. <video_gen></video_gen>",

"ordered_fragment_labels": ["F","A","E","D"] "original_task": "Open Thunderbird,......Then, open Google Chrome, ...... Next, open System Settings, ......Finally,

[Figure 65]

terminate the task after confirming the avatar has been successfully updated." },

Fragment D is unrelated to { the constructed task.️

"cot": "The robot's right arm repeatedly picks up and puts down the mouse. This swinging indicates that the robot is stuck in a hesitation loop malfunction, and the weak grasp is preventing it from successfully picking up the object. The system should revert to the state before the hesitation began, as shown in <image_recall></image_recall>. From this state, the robot should smoothly grasp the mouse, place it in the box, and then retract the robotic arm.",

[Figure 66]

"raw_response": ["F","A","E","D","B","C"],

}, No standard JSON output, no task description.️ {

"ordered_fragment_labels": ["F","A","E"], "original_task": "Open the email in Thunderbird, find the cartoon character image in the email, save it to the

"input_video": {"start_frame": 1,"end_frame": 11,"loop_start_frame": 9, "loop_end_frame": 11,"loop_repeat_count": 2}, "output_video": [9,24], "output_image": [9],

[Figure 67]

desktop, then open Google Chrome, use Google Lens to find a similar cartoon character image, and save it to the desktop, name it \"Avatar\". Finally, open the system settings, find the avatar, and set the avatar to the image \"Avatar\" in the download folder."

[Figure 68]

},

}

- Figure 4: Qualitative visualization. The left panel shows robot manipulation data construction, and the right panel shows GUI task reconstruction. We compare the outputs of general-purpose MLLMs

with DataClaw0-E, where red crosses indicate invalid or incomplete tailoring results and green checks indicate correct structured outputs.

- 5 Conclusion

We present DataClaw0, a structured multimodal data tailoring framework for transforming raw heterogeneous streams into high-quality, schema-aligned training data. Extensive experiments

show that DataClaw0 effectively improves structured data generation across diverse multimodal domains. Compared with self-refinement baselines, DataClaw0 produces substantially higher-quality annotations, and achieves performance competitive with strong proprietary-model annotators under the same data budget. Downstream adaptation results further demonstrate that DataClaw0-generated data provides useful supervision for targeted model refinement, especially on end-to-end task success metrics.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv:2502.13923, 2025. 1, 3
- [2] QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. https: //qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from= research.latest-advancements-list, 2025. 1
- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 3
- [4] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv:2410.21276, 2024. 1, 3
- [5] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023. 1, 3
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 3
- [8] OpenAI Team. Gpt 5. https://openai.com/zh-Hans-CN/index/introducing-gpt-5/, 2025. 1
- [9] Introducing the next generation of claude. https://www.anthropic.com/news/ claude-3-family, March 2024. 1
- [10] MiniMax-AI. Minimax-m2.7. https://github.com/MiniMax-AI/MiniMax-M2.7, 2025. Accessed: 2026-05-05. 1
- [11] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 7
- [12] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. 1
- [13] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 1, 3, 7, 31
- [14] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9339–9347, 2019. 1, 3

- [15] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 1, 3
- [16] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708–59728, 2023. 1
- [17] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024. 1
- [18] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web, 2023. 1, 3
- [19] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400,

2024. 1

- [20] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 1
- [21] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. 1, 3
- [22] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023. 1
- [23] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023. 1, 2
- [24] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 1, 3
- [25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. TMLR, 2025. 1
- [26] Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024. 1, 3
- [27] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning, 2023. 1, 3
- [28] Jianshu Zhang, Dongyu Yao, Renjie Pi, Paul Pu Liang, and Yi R Fung. Vlm2-bench: A closer look at how well vlms implicitly link explicit matching visual cues. arXiv:2502.12084, 2025. 2, 3
- [29] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024. 2, 3

- [30] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025. 2, 3
- [31] Disheng Liu, Jingyu Wang, Zihan Chen, and Yuhao Zhang. Thinking in space: How multimodal large language models see, remember and recall spaces. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 4
- [32] Xi Chen, Mingkang Zhu, Shaoteng Liu, Xiaoyang Wu, Xiaogang Xu, Yu Liu, Xiang Bai, and Hengshuang Zhao. Mico: Multi-image contrast for reinforcement visual reasoning. arXiv preprint arXiv:2506.22434, 2025. 2, 4
- [33] Jiawei Li, Yang Zhang, Ming Chen, and Hao Wang. Viewspatial-bench: Evaluating multiperspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500,

2025. 2, 4

- [34] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In CVPR, 2024. 2, 4
- [35] Zichao Li, Ines Arous, Siva Reddy, and Jackie Chi Kit Cheung. Evaluating dependencies in fact editing for language models: Specificity and implication awareness. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7623–7636, 2023. 2, 4
- [36] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. Data-centric artificial intelligence: A survey. arXiv preprint arXiv:2303.10158,

2023. 2

- [37] Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Alicia Parrish, Hannah Rose Kirk, et al. Dataperf: Benchmarks for data-centric ai development. Advances in Neural Information Processing Systems, 36:5320– 5347, 2023. 2
- [38] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions, 2022. 2, 4
- [39] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations,

2024. 2, 4

- [40] Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023. 2, 4
- [41] Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, and Hongxia Jin. Alpagasus: Training a better alpaca with fewer data. In The Twelfth International Conference on Learning Representations,

2024. 2

- [42] Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P Meyer, Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M Wolff, and Xin Huang. Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024. 3
- [43] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455–14465, June 2024. 3

- [44] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. RoboSpatial: Teaching spatial understanding to 2D and 3D vision-language models for robotics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. Oral Presentation. 3
- [45] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. 3
- [46] Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, and Ajay Divakaran. Measuring and improving chain-of-thought reasoning in vision-language models. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, pages 192–210, 2024. 3
- [47] Shengju Qian, Hao Shao, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual CoT: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37, 2024. 3
- [48] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207–1216, 2019. 3
- [49] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Kaipeng Zhang, and Tong He. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling, 2025. 3
- [50] Zeqi Xiao, LAN Yushi, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. 3
- [51] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The epic-kitchens dataset: Collection, challenges and baselines. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):4125–4141, 2020. 3
- [52] Cong Wan, Xiangyang Luo, Zijian Cai, Yiren Song, Yunlong Zhao, Yifan Bai, Yuhang He, and Yihong Gong. Grid: Visual layout generation. arXiv preprint arXiv:2412.10718, 2024. 3
- [53] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, and Hengshuang. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774,

2024. 3

- [54] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 3
- [55] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 3
- [56] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246,

2024. 3

- [57] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 3
- [58] Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493,

2025. 3

- [59] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 3
- [60] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [61] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2024. 3
- [62] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yujia Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. LLaVA-RLHF: Aligning large multimodal models with factually augmented RLHF. https://llava-rlhf.github.io/,

2024. Accessed: 2024-10-30. 3

- [63] Yiwei Wang, Kai-Wei Chang, and Ming-Hsuan Yang. Towards comprehensive reasoning in vision-language models. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2024. 3
- [64] Zhiyuan Li, Dongnan Liu, Chaoyi Zhang, Heng Wang, Tengfei Xue, and Weidong Cai. Enhancing advanced visual reasoning ability of large language models. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1915–1929, 2024. 3
- [65] Cong Wan, Zeyu Guo, Jiangyang Li, SongLin Dong, Yifan Bai, Lin Peng, Zhiheng Ma, and Yihong Gong. Remot: Reinforcement learning with motion contrast triplets. arXiv preprint arXiv:2603.00461, 2026. 3, 7, 31
- [66] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 3
- [67] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [68] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 3
- [69] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 3
- [70] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Eric Xin Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. In Proceedings of the IEEE/CVF international conference on computer vision, 2025. 3
- [71] Jiangyang Li, Cong Wan, SongLin Dong, Chenhao Ding, Qiang Wang, Zhiheng Ma, and Yihong Gong. Trajectory-diversity-driven robust vision-and-language navigation. arXiv preprint arXiv:2603.15370, 2026. 3

- [72] Xiangyang Luo, Ye Zhu, Yunfei Liu, Lijian Lin, Cong Wan, Zijian Cai, Yu Li, and Shao-Lun Huang. Canonswap: High-fidelity and consistent video face swapping via canonical space modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10064–10074, 2025. 3
- [73] Jianchao Zhao, Huoren Yang, Hu Yusong, Yuyang Gao, Qiguan Ou, Cong Wan, SongLin Dong, Zhiheng Ma, and Yihong Gong. Retrieve-then-steer: Online success memory for test-time adaptation of generative vlas. arXiv preprint arXiv:2605.10094, 2026. 3
- [74] Jiangyang Li, Cong Wan, Changjie Wu, Songlin Dong, Lingjun Zhang, Linzhe Shi, Xu Wang, Zhiheng Ma, Hang Zhang, Mu Xu, et al. Prosr: Process-shaped spatial reasoning for reliable chain-of-thought in vlms. arXiv preprint arXiv:2605.25524, 2026. 3
- [75] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. 4
- [76] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2023. 4
- [77] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In Advances in Neural Information Processing Systems, 2023. 4
- [78] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, 2023. 4
- [79] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. 4
- [80] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. 4
- [81] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Lu Weng, Yitao Liu, Toh Jing Hua, Junda Zhao, Qian Liu, Che Liu, Qian Zeng, Yiheng Liu, Hongjin Su, Qizhi Su, Jiazhan Liu, Xuanliang Yao, Chenglin Liu, and Yanchao Liu. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023. 4
- [82] LangChain AI. Langchain. https://github.com/langchain-ai/langchain, 2023. Opensource framework for building applications with language models. 4
- [83] Significant Gravitas. Autogpt: An experimental open-source attempt to make gpt-4 fully autonomous. https://github.com/Significant-Gravitas/AutoGPT, 2023. Open-source autonomous AI agent framework. 4
- [84] OpenClaw. Openclaw: Open-source ai agent platform. https://openclawdoc.com/, 2025. Accessed: 2026-05-05. 4
- [85] Arindam Mitra, Luciano Del Corro, Guoqing Zheng, Shweti Mahajan, Dany Rouhana, Andres Codas, Yadong Lu, Wei-ge Chen, Olga Vrousgos, Corby Rosset, Fillipe Silva, Hamed Khanpour, Yash Lara, and Ahmed Hassan Awadallah. Agentinstruct: Toward generative teaching with agentic flows. arXiv preprint arXiv:2407.03502, 2024. 4
- [86] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025. 7, 31
- [87] Modi Shi, Yuxiang Lu, Huijie Wang, Chengen Xie, and Qingwen Bu. Introducing agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. https://opendrivelab.com/AgiBot-World/, March 2025. Blog post. 21

##### [88] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025. 21

### A DataClaw0 Dataset and Benchmark

This appendix provides additional details about the DataClaw0 data corpus, the data construction pipeline, the DataClaw0-val benchmark, the fuzzy-intent evaluation subset, and the structured evaluation metrics used throughout the paper. The goal of this section is to make the data construction and evaluation protocols transparent and reproducible.

#### A.1 Dataset Overview

DataClaw0 is constructed to cover a broad spectrum of high-entropy multimodal streams, ranging from long GUI operation logs to physical manipulation trajectories and instructional videos. We organize the corpus into five representative domains: Daily Life, Education, Embodied AI, World Models/AIGC, and GUI Agents. Each domain contains raw multimodal streams, factual anchors extracted by domain-specific tools, and intent-conditioned structured outputs synthesized and verified by our construction pipeline.

Domain diversity. The five domains are selected to cover different forms of multimodal entropy. GUI streams contain dense text, UI elements, and temporally ordered actions. Embodied trajectories contain object states, spatial relations, contact events, and continuous action paths. Daily Life videos emphasize procedural understanding and event segmentation. World Models/AIGC samples require extracting scene layouts, motion patterns, and generation-oriented visual structures.

Intent diversity. For each domain, we design a set of domain-specific tailoring intents. These intents specify the target downstream use case and output schema, such as GUI navigation training data, spatio-temporal VQA data, embodied action trajectory data, or video generation supervision. Table 4 lists representative intent categories.

Table 4: Representative intent categories in DataClaw0.

Domain Representative Intent Category Typical Output Structure GUI Agents UI action extraction ordered action JSON with coordinates Embodied AI robot recovery reasoning fault diagnosis, corrective trajectories World Models / AIGC video generation planning prompt, motion plan, key evidence frames Daily Life procedural knowledge extraction step-by-step reasoning and QA Education lecture summarization key concept, interlaced text and image

Data splits and leakage control. We ensure that validation samples are source-disjoint from training data whenever the raw source provides identifiable stream or task IDs. Specifically, raw videos, GUI sessions, and embodied trajectories used in DataClaw0-val are excluded from the SFT and GRPO pools. For GUI and embodied trajectories, we also remove repeated task templates and nearly identical action sequences.

#### A.2 A Complete Data Construction Case

This subsection provides concrete examples of how DataClaw0 converts raw multimodal streams into intent-conditioned structured supervision. We show three representative cases from different domains: embodied manipulation, long-horizon GUI composition, and daily-life environmental understanding. These cases illustrate that DataClaw0 is not designed to merely caption raw videos or screenshots; instead, it transforms high-entropy multimodal observations into structured, task-specific data instances that can be directly used for downstream post-training.

- Case 1: embodied manipulation. The first example comes from a robot manipulation trajectory. The raw input is a short robot video consisting of sampled frames from an episode in which the robot manipulates objects on a table. The user intent is to construct a Predict Next Primary Subtask training example. This intent requires the model to infer the next high-level manipulation step from the partially observed trajectory, rather than simply describe the current frame.

Given the raw video, the bottom-up extraction stage identifies manipulation-relevant anchors, such as visible objects, object states, robot arm motion, and temporal progress. In this example, the robot has already placed a bag of yellow beads into a blue bowl, and the green block remains on the table as the next salient object to be manipulated. The synthesis stage then converts these anchors into a structured question-answer instance with an explicit reasoning trace.

Sample ID: 120_embodied_30 Raw input: a sampled robot manipulation video from an episode involving a beauty blender and building blocks. User intent: Construct one Predict Next Primary Subtask example from the robot manipulation video. Extracted anchors:

[

{

"type": "object_state", "content": "a bag of yellow beads has been placed into the blue bowl", "evidence_frames": [0, 16]

}, {

"type": "object_presence", "content": "a green block remains on the table", "evidence_frames": [0, 16]

}, {

"type": "robot_motion", "content": "the robot arm is retreating after completing the previous placement", "evidence_frames": [0, 16]

}, {

"type": "task_progress", "content": "the previous manipulation subtask appears completed; the next object should be selected", "evidence_frames": [0, 16]

}

] Structured output: {

"question": "What is the robot’s next primary subtask?", "answer": "Pick up the green block.", "cot": "The robot has just finished placing the bag of yellow beads into the blue bowl. The left arm is retreating, leaving the green block as the remaining object to be manipulated on the table. Therefore, the next logical step is for the robot to approach and grasp the green block.", "input_video": [0, 16], "input_image": null, "output_video": null, "output_image": null

}

This example demonstrates how DataClaw0 turns a robot trajectory into a compact high-level planning instance. The key supervision signal is not the surface-level description of the scene, but the inferred next subtask grounded in temporal evidence and object-state transitions.

- Case 2: long-horizon GUI task composition. The second example comes from the GUI domain. The raw input contains multiple short-horizon GUI trajectory fragments. Each fragment consists of screenshots, low-level actions, and textual descriptions. The user intent is to compose a plausible long-horizon task from these fragments, infer the correct fragment order, and reconstruct the original task. This is a challenging data construction setting because the model must distinguish semantically compatible fragments from unrelated distractors.

In this case, the candidate fragments include spreadsheet editing, Twitter/X Spaces sharing, Trello due-date editing, and screen-time setting configuration. Only three fragments belong to the same Excel spreadsheet task. DataClaw0 must identify that Fragments B, A, and C are mutually consistent, while Fragments D, E, and F are distractors from unrelated applications.

Sample ID: 145_GUI_15 Raw input: six short-horizon GUI trajectory fragments, each containing screenshots, lowlevel GUI actions, and natural-language step descriptions. User intent: Help compose a complex long-horizon GUI task from these short-horizon GUI trajectory fragments. Extracted fragment-level anchors:

[

{

"fragment": "A", "domain": "spreadsheet", "application": "Excel", "content": "create column headers Ratings, Cost, Location,

Number of playerz, and enter Boggle rating", "key_cells": ["B1", "B2", "C1", "D1", "E1"], "key_actions": ["click", "write", "press_enter", "press_tab"]

}, {

"fragment": "B", "domain": "spreadsheet", "application": "Excel", "content": "adjust column A width, undo, auto-adjust column A,

adjust row 5 height, and undo", "key_objects": ["column A", "row 5", "Undo button"], "key_actions": ["drag", "click_undo", "double_click"]

}, {

"fragment": "C", "domain": "spreadsheet", "application": "Excel", "content": "correct typo in E1 by changing playerz to players", "key_cells": ["E1"], "key_actions": ["double_click", "press_right", "backspace", "write"]

}, {

"fragment": "D", "domain": "social_media", "application": "Twitter/X", "content": "share a scheduled Space by creating a post"

}, {

"fragment": "E", "domain": "project_management", "application": "Trello", "content": "set a card due date to 10/30/2025"

}, {

"fragment": "F", "domain": "settings", "application": "TikTok or mobile settings", "content": "enable screen time breaks and set interval to 30 minutes"

}

] Composition rationale:

Fragment B, Fragment A, and Fragment C all operate on the same Excel spreadsheet for a Family Game Inventory. Fragment B prepares the sheet layout by adjusting column and row sizes. Fragment A then enters the spreadsheet headers and rating data, including the typo "Number of playerz"

in cell E1. Fragment C continues from that state and corrects the typo in E1 from "playerz" to "players". Fragments D, E, and F are discarded because they belong to unrelated applications and task contexts.

Structured output: {

"ordered_fragment_labels": ["B", "A", "C"], "original_task": "In Excel, widen the width of column A, undo it, use automatic adjustment to change the width of column A, widen the height of row 5, undo it, type Ratings in B1, enter 4 as the Boggle rating and try to move in different directions using the keyboard, type Cost in C1, Location in D1, Number of playerz in E1, replace the information in E1 by Number of playerz and change the last letter to s."

}

This case highlights three important capabilities of DataClaw0. First, it performs cross-fragment semantic alignment by recognizing that the spreadsheet fragments share the same application, table content, and cell-level state. Second, it performs temporal ordering: formatting operations in Fragment

- B should happen before data entry in Fragment A, while typo correction in Fragment C must happen after the erroneous header has been created. Third, it rejects distractor fragments from unrelated domains, preventing the constructed long-horizon task from mixing incompatible applications.

- Case 3: daily-life environmental understanding. The third example comes from a daily-life video. The intent is to construct a Contextual Reasoning and Environmental Understanding instance. The raw video contains an egocentric or first-person view of an indoor scene. Unlike the GUI case, the goal is not to reconstruct executable actions, but to generate a grounded reasoning question that requires understanding the surrounding environment.

Sample ID: 17_daily_7 Raw input: a sampled daily-life video showing an indoor scene with a wooden table and household objects. User intent: Construct one contextual reasoning and environmental understanding questionanswer pair from the video. Extracted anchors:

[

{

"type": "scene_layout", "content": "an indoor room with a wooden table surface", "evidence_frames": [0, 24]

}, {

"type": "object_presence", "content": "books and household objects are visible on the table", "evidence_frames": [0, 24]

}, {

"type": "object_presence", "content": "a glass container or cup-like object appears on the table", "evidence_frames": [0, 24]

}, {

"type": "environment_cue", "content": "the objects and furniture are consistent with a living area or study space", "evidence_frames": [0, 24]

}

] Structured output: {

"question": "What type of room am I in at the start of the video?", "answer": "I’m in a living area or study space, identified by the wooden

table with various items on it including books, a glass, and other household objects.", "cot": "The visual scene shows a wooden table surface with multiple objects like a red book, glass container, and other items typical of a living or study area. The flooring appears to be wooden tiles, consistent with a main living space rather than a bedroom or kitchen.", "input_image": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,

16, 17, 18, 19, 20, 21, 22, 23, 24], "output_image": null

}

Across these examples, DataClaw0 follows the same general construction principle. First, it extracts or identifies factual anchors from the raw multimodal stream, such as object states, GUI actions, OCR text, application context, scene layout, or temporal transitions. Second, it conditions generation on a user-specified data construction intent. Third, it produces a structured output that follows the required schema and is suitable for downstream training. The embodied case emphasizes temporal prediction and object-state reasoning; the GUI case emphasizes long-horizon task composition and distractor rejection; and the daily-life case emphasizes contextual scene understanding.

#### A.3 Data Construction Pipeline

DataClaw0 builds training data through an intent-conditioned construction pipeline. Instead of directly prompting a VLM on arbitrary raw videos or GUI trajectories, we first transform raw streams into compact, verifiable construction units, and then ask a strong multimodal model to synthesize structured samples under explicit schema constraints. This design separates deterministic preprocessing from semantic synthesis: lightweight scripts identify candidate temporal windows, event fragments, and media resources, while the VLM performs higher-level reasoning such as subtask prediction, failure diagnosis, task decomposition, and long-horizon reconstruction.

At a high level, the pipeline contains five stages:

- 1. Candidate discovery: scan raw videos or GUI trajectories to find segments that are likely to support a target data construction intent.
- 2. Anchor and context packaging: convert selected segments into compact requests containing sampled frames, action traces, fragment descriptions, timestamps, or frame ranges.
- 3. Intent-conditioned VLM synthesis: call a strong VLM with a task-specific prompt and a strict output schema.
- 4. Validation and materialization: parse the generated JSON, validate temporal ranges and required fields, and materialize referenced videos or images.
- 5. Corpus export: write normalized training samples and separate artifact mapping files for downstream training and inspection.

This section describes the pipeline using two representative sources: RoboCOIN robot manipulation videos and GUI multi-event trajectories.

#### A.3.1 Robot Data Pipeline

For robot data [87, 88], DataClaw0 processes raw robot operation videos into two types of embodied training data: robot fault diagnosis and recovery data, and robot operation understanding data. Both branches share the same general principle: first select informative video windows from long episodes, then ask the VLM to construct structured supervision from a small number of representative frames, and finally materialize the referenced video or image assets.

Fault synthesis branch. The fault synthesis branch constructs training samples for robot failure diagnosis and recovery. The goal is to synthesize examples where the input depicts a robot getting stuck or hesitating, and the output asks the model to identify the failure and infer the correct recovery behavior.

The pipeline is implemented in three scripts:

robocoin_fault_candidate_mine.py → robocoin_fault_synthesis_prepare.py → robocoin_fault_synthesis_run.py.

The first stage mines candidate windows from the original RoboCOIN videos. It scans the headcamera stream, e.g., observation.images.cam_high_rgb, reads video metadata such as frame rate, duration, and total frame count, and filters invalid or too-short videos. For each valid episode, it samples anchor positions at fixed fractions of the trajectory, such as 35%, 50%, and 65%. Around each anchor, the script constructs two temporal ranges: an observation range and a future range. The observation range is used to synthesize the faulty input, while the future range provides evidence of the correct continuation.

The candidate mining stage produces a JSONL file of the following form: {

"video_path": ".../episode_xxx.mp4", "fps": 30.0, "num_frames": 1800, "obs_start": 540, "obs_end": 600, "future_start": 600, "future_end": 660

}

The second stage prepares VLM requests. For each candidate, it samples a small number of frames from the observation and future ranges, typically eight observation frames and four future frames. These frames are encoded into the API request together with a task-specific system prompt. The prompt defines the intended fault type, such as approach_stall, and requires the VLM to return a structured JSON object containing the freeze frame, the number of repeated frames, the recall frame, the recovery video range, and the natural-language question, answer, and reasoning.

A simplified target schema is: {

"freeze_frame": int, "freeze_repeat_count": int, "output_image": [int], "output_video": [int, int], "question": str, "answer": str, "cot": [str, ...]

}

The third stage calls the VLM API, parses the returned JSON, and materializes the actual media assets. The faulty input video is generated by repeating the selected freeze_frame, thereby simulating a robot stall. The recovery output video is clipped from the original future trajectory according to output_video. The recall image is extracted from the specified output_image frame. The final output directory contains normalized training samples and a separate artifact file that maps sample IDs to physical video and image paths:

output_root/

+-- training_samples.jsonl

+-- training_artifacts.jsonl

+-- videos/ | +-- input/ | \-- output/ \-- images/

\-- output/

This branch therefore turns ordinary successful robot trajectories into counterfactual-style failure supervision. The synthetic input contains a plausible failure, while the output remains grounded in the original successful continuation.

Understanding branch. The understanding branch constructs robot operation understanding data. It targets tasks such as describing the current subtask, predicting the next primary subtask, and generating interleaved tutorial-style supervision. The corresponding scripts are:

robocoin_understanding_prepare.py → robocoin_understanding_run.py.

Unlike the fault branch, this branch does not create synthetic stalled videos. Instead, it samples global keyframes from each episode and asks the VLM to construct an understanding-oriented question-answer pair. For each episode, the preparation script uniformly samples a fixed number of frames, e.g., 20 frames, from the full trajectory. The task family determines the prompt. For example, in predict_next_primary_subtask, the model must infer the next high-level manipulation step from the visible progress of the episode.

The VLM is required to select an input video range and produce the corresponding structured sample: {

"input_video": [start_frame, end_frame], "input_image": null, "output_video": null, "output_image": null, "question": "What is the robot’s next primary subtask?", "answer": "...", "cot": ["...", "..."]

}

The run script validates the output, clips the selected input video range, optionally extracts referenced output images, and writes the normalized training sample. In this way, long raw robot episodes are converted into compact embodied reasoning examples with explicit temporal grounding.

#### A.3.2 GUI Multi-event Construction Pipeline

For GUI data, DataClaw0 focuses on long-horizon task structure. The goal is not only to record low-level GUI actions, but also to identify trajectories that contain multiple meaningful subgoals and transform them into data for task decomposition, fragment ordering, and long-horizon task reconstruction.

The GUI pipeline is centered on a multi-event filtering and decomposition module. It processes raw GUI trajectories containing user instructions, screenshots, application or URL context, and low-level events such as clicks, typing, scrolling, keyboard shortcuts, and navigation changes. The pipeline first determines whether a trajectory is a genuine multi-event task, then decomposes it into subtask fragments, and finally constructs reconstruction-style training data by shuffling the fragments and asking a model to infer the original task order.

- Stage 1: heuristic multi-event candidate filtering. The first stage uses lightweight heuristics to identify trajectories that are likely to contain multiple separable objectives. The filtering logic considers several task-level signals:

- • whether the user instruction contains multiple action goals;
- • whether it includes sequential markers such as “first”, “then”, or “after that”;
- • whether the trajectory crosses multiple applications, pages, or URLs;
- • whether similar operations are repeated over multiple objects;
- • whether the visual or event stream shows clear phase changes.

Based on these signals, each trajectory is assigned a coarse event-structure label:

- • none: the trajectory is a single-goal task and should not be decomposed;
- • sequential: the task contains multiple stages executed in order;
- • parallel: the task applies similar operations to multiple objects;
- • hybrid: the task combines sequential stages and parallel sub-blocks.

This stage is intentionally conservative. Its purpose is to reduce the search space for later VLM processing while avoiding the over-segmentation of ordinary single-goal workflows.

- Stage 2: initial subtask proposal. For trajectories predicted as multi-event, DataClaw0 produces an initial subtask structure. Sequential tasks are split by stage boundaries, parallel tasks are split by object or repeated operation, and hybrid tasks are first divided into coarse sequential stages before expanding parallel blocks inside each stage. This heuristic decomposition is not treated as final supervision. Instead, it provides useful scaffolding for the subsequent LLM-based refinement.
- Stage 3: LLM-based decomposition and reconstruction. The next stage sends candidate trajectories and their preliminary structure to an LLM for finer-grained judgment. The LLM is asked to decide whether the task truly merits decomposition and, if so, to output continuous, non-overlapping, and collectively complete subtask fragments. The prompt explicitly distinguishes single-task trajectories from sequential, parallel, and hybrid multi-event trajectories.

After obtaining reliable subtask fragments, DataClaw0 constructs long-horizon reconstruction data. The fragments are shuffled, and the model is asked to infer their correct order and reconstruct the original user task. This creates training examples that directly target long-context GUI reasoning, including fragment ordering, cross-fragment state tracking, and task-level composition.

A representative output schema is: {

"ordered_fragment_labels": ["B", "A", "C"], "original_task": "In Excel, adjust the spreadsheet layout, enter inventory headers and ratings, and correct the typo in the final header."

}

In the Excel example, fragments involving spreadsheet formatting, header entry, and typo correction are selected and ordered into one coherent long-horizon task, while distractor fragments from Twitter/X, Trello, or settings pages are discarded. This demonstrates that the GUI pipeline is not merely segmenting actions; it is reasoning about application context, task continuity, and whether fragments can compose a plausible original instruction.

#### A.3.3 Unified Output Format and Artifact Management

Although the RoboCOIN and GUI pipelines operate on different raw modalities, they share a unified output philosophy. The training sample stores semantic supervision, while large binary assets are materialized separately and referenced through artifact metadata. This keeps the training JSONL lightweight and makes it possible to relocate or regenerate media files without changing the logical sample content.

- A typical normalized sample has the following structure:

{

"id": "unique_sample_id", "data_type": "embodied_robot_fault_decision_world_model", "task_family": "approach_stall", "input_video": [start_frame, end_frame], "input_image": null, "output_video": [start_frame, end_frame], "output_image": [frame_index], "question": "What went wrong and how should the robot recover?", "answer": "The robot stalls while approaching the object. It should resume

the approach and continue with the successful recovery motion. <video_gen>",

"cot": [ "The observation frames show the robot approaching the target.", "The repeated freeze frame indicates a stall rather than normal motion.", "The future segment shows the correct continuation and recovery."

]

} For samples that require media outputs, the corresponding artifact file stores the concrete paths: {

"id": "unique_sample_id", "input_video_path": "videos/input/unique_sample_id.mp4", "output_video_path": "videos/output/unique_sample_id.mp4", "output_image_path": "images/output/unique_sample_id.jpg"

}

We use placeholders such as <video_gen> and <image_recall> in natural-language answers to indicate where generated or recalled media should be inserted during model training or evaluation.

#### A.3.4 Validation and Quality Control

DataClaw0 applies deterministic validation after VLM synthesis. The validation procedure depends on the modality, but follows the same principles across domains.

Schema validation. Each generated response is parsed as JSON and checked against the required task schema. Samples are rejected if required fields are missing, frame ranges have invalid types, coordinates are malformed, or the output contains unparseable nested structures.

Temporal and artifact validation. For video-based samples, frame indices must fall within the original episode length. Input and output ranges must have positive duration, and extracted clips or images must be successfully materialized by ffmpeg. In the RoboCOIN fault branch, the selected freeze frame must lie inside the observation range, while the recovery video should lie inside the future range.

Task-structure validation. For GUI multi-event data, decomposed fragments are checked for structural consistency. The final fragments should be continuous, non-overlapping, and collectively cover the intended trajectory region. Reconstruction samples must contain a valid permutation of fragment labels, and irrelevant distractor fragments should not be included in the ordered solution.

Resumability and auditability. All run scripts support resumable processing, so interrupted API calls or media extraction failures do not require rebuilding the entire corpus. In addition, DataClaw0 writes intermediate JSONL files at each major stage, such as candidate windows, API requests, training samples, and artifact mappings. These files make the construction process auditable: one can inspect how a raw episode or trajectory was selected, what evidence was sent to the VLM, what structured supervision was generated, and which media files were finally materialized.

Overall, DataClaw0’s data construction pipeline is designed to combine deterministic preprocessing with model-based semantic synthesis. The deterministic stages provide grounding, compactness, and reproducibility, while the VLM stage supplies the high-level reasoning needed to create diverse supervision formats from raw multimodal interaction data.

#### A.4 DataClaw0-val

DataClaw0-val is designed to evaluate agentic multimodal data tailoring rather than conventional captioning, VQA, or action recognition. Each example provides a high-entropy multimodal input together with an explicit data-construction intent, and asks the model to produce a structured trainingdata instance. The benchmark therefore measures whether a model can decide what should be extracted, selected, rewritten, or recomposed from raw multimodal evidence for a downstream data-construction purpose.

Concretely, a DataClaw0-val instance consists of four components:

- 1. Multimodal input: a video, an image sequence, GUI trajectory fragments, or interleaved visual-text educational material.

- 2. Tailoring intent: a natural-language instruction specifying the target data type, such as video-generation training, VLN-style navigation, robot fault recovery, multimodal tutorial construction, or long-horizon GUI task reconstruction.
- 3. Target schema: a required JSON format that defines the fields to be returned.
- 4. Reference output: a schema-valid structured output that selects the relevant frames or fragments and writes the corresponding question, answer, reasoning, or reconstructed task.

All examples require structured JSON outputs. Depending on the domain, the output may include frame indices, selected input or output images, generated natural-language supervision, chain-ofthought reasoning, video-generation placeholders, or ordered GUI fragment labels. This design makes the benchmark closer to real data-engineering workflows: the model must not only understand the media, but also transform it into a usable data sample.

In addition to DataClaw0-val, we construct DataClaw0-Intent, a fuzzy-intent stress test. DataClaw0Intent removes or weakens the explicit task specification and asks the model to infer a suitable data-construction objective from an underspecified user request such as “Help me design an example for data construction.” Unlike DataClaw0-val, DataClaw0-Intent does not rely on a single canonical ground-truth answer. Its final quality is evaluated through user study, because multiple valid tailoring choices may exist for the same raw input.

#### A.4.1 Benchmark Composition

DataClaw0-val contains 200 examples covering five representative multimodal data-tailoring scenarios: AIGC/World, Daily Life, Education, Embodied AI, and GUI Agents. DataClaw0-Intent contains fuzzy-intent examples derived from the same general pool of multimodal sources, but with underspecified or colloquial user instructions.

The five categories are designed to stress different aspects of multimodal data tailoring.

AIGC/World. AIGC/World examples evaluate whether a model can identify visually dynamic segments useful for generative-model training. The input is typically a product-promotion or openworld video represented as sampled frames. The model must locate a segment with rich motion, object manipulation, or character-object interaction, and return a concise description together with the selected frame indices. For example, in a product-promotion video, the desired output may identify the segment where a host receives a speaker with both hands and places it on a table, or where a host adjusts a handbag strap and demonstrates the crossbody pose.

Daily Life. Daily Life examples evaluate embodied navigation and procedural understanding in egocentric or indoor videos. For example, the model is asked to create a VLN-style navigation sample from a video. The output must include a navigation question, an executable natural-language answer, a concise reasoning trace, and selected input/output frames. For example, one sample asks the model to construct navigation supervision from a room video where the camera moves from a desk area toward a bed; another asks for directions from a bed toward a workspace by a window. Beyond navigation, this category also tests a model’s ability to construct world-knowledge understanding and to generate interleaved visual-text tutorial data, such as step-by-step household instruction samples or multimodal daily-task explanations.

Education. Education examples evaluate whether a model can reconstruct multimodal teaching material from instructional images or noisy original annotations. The input may be a sequence of lecture screenshots, visual derivations, or partially aligned transcript-image pairs. The model must produce an interleaved visual-text learning sample, usually in the form of a student-facing question and a step-by-step answer with image placeholders. For instance, one sample transforms calculus screenshots into a partial-fraction and substitution explanation for integrating 1/(x2 − x). Another reconstructs a Law of Cosines lesson and explains how to isolate c, apply the square root, use parentheses in the calculator, and ensure degree mode.

Embodied AI. Embodied examples evaluate robot-operation data construction. We include both fault-diagnosis examples and tutorial-style manipulation examples. In the fault-diagnosis setting, the model receives robot manipulation frames and must synthesize an approach-stall recovery sample. It

selects an input video range, a freeze frame, a freeze-repeat count, a recovery video range, and a recall image. The output also includes a question, answer, and reasoning trace, often with placeholders such as <video_gen> and <image_recall>.

GUI Agents. GUI examples evaluate long-horizon task reconstruction from short GUI trajectory fragments. Each input contains several fragments, where each fragment includes an instruction-level description, low-level GUI actions, screenshots, and step descriptions. The model must determine which fragments can compose a plausible long task, infer their correct order, and write the reconstructed user goal.

#### A.4.2 Reference Output Construction

The reference outputs in DataClaw0-val are constructed to be directly comparable with model predictions. Each reference follows the task-specific JSON schema and contains only indices or fragment labels that exist in the provided input. For video and image-sequence tasks, frame indices are numbered from zero according to the provided sampled frames. For GUI tasks, fragment labels are drawn from the candidate fragments in the prompt.

The construction process follows three principles.

Schema validity. All references are valid JSON objects and conform to the required output schema in the prompt. We remove examples with missing required fields, invalid frame-index types, malformed lists, or inconsistent null values.

Evidence grounding. Frame selections and fragment orders must be supported by the provided multimodal evidence. For example, an AIGC/World reference should select frames that contain the described hand-object or character-object interaction. A Daily Life navigation reference should use input frames showing the starting location and output frames showing the target location or arrival segment. For Daily Life world-knowledge understanding and interleaved visual-text tutorial tasks, the reference should select key frames that capture object state changes or critical action moments, supporting causal inference or conceptual explanation. An Embodied fault reference should ensure that the freeze frame lies within the observed approach segment and that the recovery video corresponds to a plausible continuation. A GUI reconstruction reference should only include fragments that can be semantically connected into a coherent long-horizon task.

Intent alignment. The reference must satisfy the stated data-construction intent rather than merely describe the media. For instance, in AIGC/World, the output is not a generic product caption; it must identify a segment useful for video-generation training. In Education, the output is not a screenshot summary; it must become interleaved teaching material. In GUI, the output is not a list of all fragment contents; it must reconstruct a plausible original long-horizon task.

We additionally ensure that the raw streams used in DataClaw0-val are separated from the training corpus to avoid direct leakage.

#### A.4.3 DataClaw0-Intent: Fuzzy-intent Stress Test

DataClaw0-Intent evaluates whether a model can act as an agentic data-construction assistant when the user intent is vague. Unlike DataClaw0-val, where the target task and output schema are explicitly specified, DataClaw0-Intent provides underspecified instructions such as:

“Help me design an example for data construction.”

The model must infer a useful data-construction objective from the raw input and produce a structured output accordingly.

These fuzzy-intent examples stress a different capability from standard schema following. The model needs to decide:

- • what kind of training data the media is most suitable for;
- • which temporal segment or visual evidence is most informative;
- • what fields should be included in the output;

- • how to phrase the constructed question, answer, and reasoning;
- • whether the selected supervision is useful for downstream model training.

For example, given a game-like video of a player approaching a spear-wielding enemy, a valid model output may transform the clip into a predictive world-modeling sample. The model can select the frames where the player closes distance as input evidence and the frames where the enemy initiates a spear attack as output evidence:

{

}

"task_type": "action anticipation / predictive world modeling", "question": "If the player keeps moving toward the spear-wielding enemy,

what action is likely to happen next?", "answer": null, "cot": "The media is best suited for action-anticipation data rather than

generic captioning because it contains a clear before-and-after temporal structure. Frames 8--12 show the informative input segment: the player is running down a rocky path, closing the distance to an enemy who is holding a spear and preparing to react. Frames 13--18 show the target future segment: the enemy initiates a spear attack once the player enters striking distance. The constructed question turns this transition into a future-prediction task. The answer is null because the supervision is provided by the selected output frames. This example is useful for downstream training because it teaches a model to infer future combat behavior from approach direction, distance, enemy pose, and weapon state.",

"field_design": { "task_type": "records the inferred data-construction objective", "question": "phrases the supervision as a conditional future-prediction

query", "answer": "kept null because the expected target is visual", "cot": "explains the task choice, evidence selection, and training value", "input_image": "contains the pre-event visual context", "output_image": "contains the future event used as supervision"

}, "input_image": [8, 9, 10, 11, 12], "output_image": [13, 14, 15, 16, 17, 18]

Importantly, DataClaw0-Intent does not have a unique ground-truth output. The same raw video could plausibly be converted into different useful training instances, such as future-frame prediction, action

anticipation, event segmentation, or visual reasoning. Therefore, we do not score DataClaw0-Intent with exact-match metrics against a single reference. Instead, the final evaluation is conducted through user study. 100 human users judge whether the model’s constructed sample is useful, grounded in the media, appropriately structured, and aligned with a reasonable inferred data-construction goal.

#### A.4.4 Metrics

Structured multimodal data cannot be faithfully evaluated by conventional text-generation metrics such as BLEU or ROUGE. We therefore define a hierarchical score:

Stotal = Gjson · (λ1Sfield + λ2Ssemantic + λ3Straj), (8) where Gjson ∈ {0,1} is a hard JSON-validity gate, assigning zero score to malformed outputs. We use λ1 = 0.20, λ2 = 0.35, and λ3 = 0.45. The field score measures schema integrity:

Sfield = 0.7Cfield + 0.3(1 − Rfield), (9)

where Cfield is ground-truth field coverage and Rfield is the truncated extraneous-field ratio, with robust key matching to tolerate minor formatting differences.

The semantic score measures embedding-based cosine similarity over textual fields:

αsimquestion + βsimanswer + γsimcot α + β + γ

, (10)

Ssemantic =

where α = 0.40, β = 0.40, and γ = 0.20 by default, dynamically renormalized based on field availability.

For spatio-temporal samples, the visual score is defined based on trajectory-shape similarity:

Straj = exp(−dshape/τs), (11)

where trajectories are normalized and resampled to K = 50; dshape denotes the trajectory Mean Absolute Error (MAE), and the temperature hyperparameter is set to τs = 0.10.

### B Training and Deployment Details

This section provides details about GRPO training, reward design, and the Omni/Expert deployment paradigms.

#### B.1 Rule-Driven Reinforcement Learning via GRPO

To overcome hallucinations and formatting collapses in long-sequence multimodal tasks, we introduce reinforcement learning atop the initial policy πSFT. Traditional multimodal RLHF requires training a reward model of comparable size to the policy model, which incurs prohibitive GPU memory overhead when processing long videos or high-resolution image sequences. Therefore, DataClaw0 employs Group Relative Policy Optimization (GRPO), eliminating the reliance on a Critic model through intra-group relative advantage estimation.

Crucially, because DataClaw0’s objective is to generate strictly structured data tailored to specific intents, we can entirely discard subjective and computationally expensive neural network reward models in favor of model-free, deterministic Rule-based Rewards. For any trajectory Y generated by the model, we define the joint reward function R(Y ):

R(Y ) = λ1Rformat(Y,Φ) + λ2Ranchor(Y,A) + λ3Reff(Y ) (12)

- • Format Compliance Reward (Rformat): We utilize an Abstract Syntax Tree (AST) or regex parser to strictly verify whether the generated content Y is 100% compliant with the target schema Φ. Successful parsing yields a high positive reward, whereas structural or syntactic errors result in an immediate truncation penalty.
- • Spatio-Temporal Anchor Reward (Ranchor): This reward is designed for embodied and videobased tasks that exhibit complex spatio-temporal dynamics. It measures the temporal shape similarity between the predicted and ground-truth trajectories—defined as the similarity of their frame-wise distributions in the constructed data sequence. To ensure temporal alignment across trajectories of varying lengths, we first normalize discrete frame indices to a standardized interval and apply interpolation to resample each trajectory into K = 50 uniformly distributed alignment points. We then compute the Mean Absolute Error (MAE) between the predicted and ground-

truth distributions of these alignment points to obtain the trajectory deviation dshape. Finally, this deviation is converted into a similarity-based reward through an exponential mapping:

Ranchor = exp −

dshape τs

(13)

where the temperature hyperparameter τs = 0.10 provides tolerance to minor misalignments. This design allows DataClaw0 to receive fine-grained quantitative feedback reflecting both the temporal shape coherence and the overall spatio-temporal alignment quality.

- • Reasoning Efficiency Penalty (Reff): To prevent the model from producing overly verbose or trivially short Chain-of-Thought (CoT) reasoning, we introduce a dynamic length regularization mechanism. Since the appropriate CoT length varies with the underlying sample complexity, an ideal reasoning trace should be sufficiently elaborate to cover key analytical steps, yet concise enough to avoid redundant self-expansion. To operationalize this notion, we assess CoT length

adaptively within each GRPO rollout group. Specifically, we identify the rollouts whose overall reward ranks within the top 50% of the group and treat their CoT lengths as reference standards. For the remaining rollouts, CoT sequences that are significantly longer or shorter than this reference are penalized, while those falling close to the reference range receive a slight positive adjustment. This relative, group-wise normalization encourages the policy to learn reasoning behaviors that are both efficient and well-calibrated to task complexity.

GRPO Optimization Objective: During training, for a given input (Xraw,I), the current policy model πθ samples a candidate group G containing G diverse outputs: G = {Y (1),Y (2),...,Y (G)}. We compute the joint reward R(g) = R(Y (g)) for each candidate and normalize them within the group to obtain the relative advantage estimation:

R(g) − µR σR

Aˆ(g) =

(14)

where µR and σR are the mean and standard deviation of the G rewards in the group, respectively. Subsequently, we update the model parameters θ by maximizing the following GRPO objective function:

G

1 G

min ρ(g)(θ)Aˆ(g),clip(ρ(g)(θ),1 − ϵ,1 + ϵ)Aˆ(g) − βDKL(πθ ∥ πref)

J(θ) = E

g=1

(15) where the importance sampling ratio is ρ(g)(θ) = π

θ(Y (g)|Xraw,I)

πold(Y (g)|Xraw,I), and ϵ is the clipping hyperparameter.

Through this “SFT Initialization + Rule-Reward Driven GRPO” paradigm, DataClaw0 significantly stimulates the model’s agentic capability to proactively tailor high-value structured knowledge from complex multimodal streams, particularly excelling in spatio-temporal grounding due to the injection of Ranchor.

#### B.2 Inference Architecture & Deployment Paradigms

To transform the GRPO-aligned policy model πθ into a practical structured tailoring agent, we design an inference architecture that emphasizes output validity, grounding reliability, and deployment

flexibility. Given raw multimodal streams Xraw and user-defined intents I, DataClaw0 generates schema-aligned structured outputs Ystruct.

- B.2.1 Core System Architecture DataClaw0’s inference process consists of three decoupled modules:

- • Multimodal Ingestion & Intent Parsing: This module receives raw multimodal data streams

Xraw and user-defined intents I, and converts them into structured context sequences readable by the policy model. In the current implementation, we employ rule-based automation scripts to segment long video trajectories into broad yet semantically coherent clips that contain key events based on raw dataset annotations. This step serves as a scalable preprocessing pipeline for multimodal grounding. In future iterations, this stage will be extended into a more intelligent pipeline using a long-video understanding model capable of adaptively identifying event boundaries and intent-relevant sub-sequences.

- • Schema-Constrained Inference Engine: This module deploys the policy model πθ to generate structured outputs. To reduce formatting failures, a lightweight schema-aware constraint can be applied during decoding, suppressing invalid continuations that violate the target schema Φ.
- • Structural Verification & Grounding: This module conducts post-hoc structural and semantic verification by leveraging factual anchors A together with quantitative evaluation metrics.

When the generated output Ystruct contains concrete elements such as UI components, spatial coordinates, timestamps, or action trajectories, the system cross-checks these elements against extracted anchors to ensure factual consistency and spatial grounding. Furthermore, the verification process is quantitatively guided by three complementary metrics: (i) schema–field correctness for structural validity, (ii) textual semantic alignment for content faithfulness, and (iii) trajectory–shape similarity for spatio-temporal coherence. Together, these mechanisms ensure that the tailored outputs are both semantically grounded and structurally reliable.

#### B.2.2 Omni vs. Expert Deployment Paradigms

Considering the heterogeneity of multimodal data and varying deployment requirements, DataClaw0 supports two complementary architectures:

- • DataClaw0-O (Omni Tailoring Agent): A unified model jointly trained across all domains. It serves as a generalist agent for diverse cross-domain intents I within a single model.
- • DataClaw0-E (Expert Tailoring System): A decoupled architecture composed of domainspecific tailoring agents, such as GUI, embodied AI experts. In deployment, the corresponding expert is selected according to the target scenario, domain setting, or user-specified configuration. This paradigm improves domain-specific robustness and allows modular updates.

### C Downstream Evaluation

#### C.1 Detailed Analysis of Downstream Application

The downstream evaluation aims to verify whether the structured data generated by DataClaw0 can provide effective supervision for targeted model refinement. We consider three representative multimodal tasks: long-horizon GUI navigation [86], action video generation [13], and spatiotemporal VQA [65]. These tasks cover different output spaces and reasoning requirements, including action planning, visual dynamics generation, and fine-grained temporal understanding.

For a fair comparison, all training sets are constructed from the same raw input streams. Specifically, each raw stream is processed by three different annotators: the base model itself, Gemini-3.1-ProPreview, and DataClaw0. The base-model processing setting serves as a self-refinement baseline, while Gemini represents a strong proprietary-model annotator. To ensure that the comparison focuses on data quality rather than data quantity, we apply the same coarse rule-based filtering procedure to remove malformed samples and then randomly sample an equal number of valid instances from each processed data pool.

As shown in Table 2, fine-tuning on self-processed data only provides limited improvements over the zero-shot base models, suggesting that simple self-refinement is insufficient for these challenging multimodal tasks. In contrast, both Gemini-processed and DataClaw0-processed data lead to substantial gains, demonstrating that high-quality structured supervision is crucial for downstream adaptation.

A closer comparison between Gemini and DataClaw0 shows a consistent trade-off. Gemini achieves slightly better results on some partial or step-level metrics, such as GUI SSR and VQA Partial Accuracy. This suggests that strong proprietary models can provide rich intermediate supervision that benefits local or partially correct predictions. However, DataClaw0 achieves stronger results on several end-to-end metrics that better reflect final task completion. For example, DataClaw0 obtains higher GUI TSR and VQA Overall Accuracy, indicating better transfer to complete task-solving ability. In action video generation, DataClaw0 also achieves lower FVD and higher Contact mAP, showing improved visual quality and stronger modeling of physical interactions.

These results support the motivation of DataClaw0. Instead of producing verbose general-purpose annotations, DataClaw0 is optimized to extract compact, schema-aligned, and task-relevant structured supervision from multimodal streams. Under the same data budget, such targeted supervision can be more effective for improving end-to-end performance. Therefore, DataClaw0 provides a practical and controllable alternative to proprietary annotators for downstream multimodal data construction.

#### C.2 Unified Refinement Protocol

For each downstream task, we use the same raw data streams and process them with different annotators: the base model itself, a strong closed-source VLM, and DataClaw0. All annotators receive the same task intent and target schema. Their outputs are passed through the same coarse filtering pipeline. We then sample an equal number of valid instances for downstream fine-tuning.

The protocol is:

- 1. Select raw multimodal streams for a downstream task.

- 2. Process the same raw streams using Self-Refinement, Gemini-3.1-Pro / Qwen3.5, and DataClaw0.
- 3. Apply the same schema and quality filters.
- 4. Randomly sample the same number of valid instances from each method.
- 5. Fine-tune the same downstream base model using identical hyperparameters.
- 6. Evaluate on held-out task-specific benchmarks.

#### C.3 GUI Navigation

Downstream model. We fine-tune Qwen3.5-4B. The model predicts actions from screenshots, task instructions, and interaction history.

Metrics. We report Step Success Rate (SSR) and Task Success Rate (TSR). SSR measures the fraction of predicted steps that match the reference action or lead to the correct UI state. TSR measures whether the full task is completed successfully within a maximum number of steps. Formally,

#successful steps #total steps

#successful tasks #total tasks

. (16)

SSR =

, TSR =

Figure 5 visualizes a representative downstream example.

[Figure 69]

Figure 5: Qualitative example for GUI navigation.

#### C.4 Action Video Generation

Downstream model. We fine-tune Wan2.2-I2V-5B. The model receives an input image and text prompt and generates a short video.

Fine-tuning hyperparameters. Table 5 lists the video generation fine-tuning setup.

Table 5: Video generation downstream fine-tuning hyperparameters.

Hyperparameter Value Base model Wan2.2-I2V-5B Training clips 200 Resolution 480 × 832 Frames per clip 81 Epochs 3 Learning rate 1 × 10−5 Batch size 1 Hardware 8 × NVIDIA A100 80GB

- Visualization. Figure 6 compares videos generated by models trained with different refined datasets.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Figure 6: Qualitative comparison for action video generation.

C.5 Spatio-temporal VQA Downstream model. We fine-tune Qwen3.5-4B. The model receives video frames and a question and predicts the answer.

Metrics. We report Partial Accuracy and Overall Accuracy. Partial Accuracy gives credit for partially correct structured answers, such as correctly identifying the object but missing the temporal order. Overall Accuracy requires the full answer to match the reference.

- Visualization. Figure 7 shows a representative VQA example.

### D Additional Experiments and Ablations

#### D.1 Scaling Laws and Emergent Diversity

In this section, we investigate the training dynamics of our framework, specifically addressing why a domain-decoupled expert architecture is superior to a unified model, and verifying the emergent generalization capabilities of DataClaw0.

Scaling Laws and Task Interference. We train a unified model, denoted as DataClaw0-O. As illustrated in Figure 3, instead of exhibiting a smooth, log-linear scaling curve, the Omni model

[Figure 77]

[Figure 78]

[Figure 79]

Question: Ignoresimplecameratranslationandonlyconsiderthecamera'shorizontalorverticalrotation. Please answer the following questions based on the three images above.Please select the horizontal rotation

- direction of the camera's viewpoint from Image 1 to Image 2 among the following options, A: Right, B: Left, C: No rotation.\nPlease select the horizontal rotation angle of the camera's viewpoint from Image 1 to Image 2 among the following options, A: 15 degrees, B: No rotation, C: 10 degrees, D: 5 degrees.\nPlease select the vertical

- rotation direction of the camera's viewpoint from Image 1 to Image 2 among the following options, A: Down, B:

- Up, C: No rotation.\nPlease select the vertical rotation angle of the camera's viewpoint from Image 1 to Image 2 among the following options, A: 143 degrees, B: 5 degrees, C: No rotation.\nPlease select the horizontal rotation

direction of the camera's viewpoint from Image 1 to Image 3 among the following options, A: Left, B: No rotation, C: Right.\nPlease select the horizontal rotation angle of the camera's viewpoint from Image 1 to Image 3 among the following options, A: 67 degrees, B: 135 degrees, C: 175 degrees, D: No rotation.\nPlease select the vertical rotation direction of the camera's viewpoint from Image 1 to Image 3 among the following options, A: Down, B:

- Up, C: No rotation.\nPlease select the vertical rotation angle of the camera's viewpoint from Image 1 to Image 3 among the following options, A: No rotation, B: 90 degrees, C: 125 degrees, D: 48 degrees.\nPlease select the horizontal rotation direction of the camera's viewpoint from Image 2 to Image 3 among the following options, A: No rotation, B: Left, C: Right.\nPlease select the horizontal rotation angle of the camera's viewpoint from Image 2 to Image 3 among the following options, A: 5 degrees, B: No rotation, C: 100 degrees.\nPlease select the vertical rotation direction of the camera's viewpoint from Image 2 to Image 3 among the following options, A: No rotation, B: Up, C: Down.\nPlease select the vertical rotation angle of the camera's viewpoint from Image 2 to Image 3 among the following options, A: No rotation, B: 5 degrees, C: 15 degrees, D: 9 degrees.\nPlease answer the multiple-choice questions above in sequence.Only return A, B, C, or D in <answer></answer>, for example <answer>ADCAABADBACA</answer>. Answer: <answer>BDBBBDCABABB</answer>

[Figure 80]

Figure 7: Qualitative example for spatio-temporal VQA.

demonstrates severe performance oscillation. For instance, the overall score starts at 53.60 with 1/15 data, drops sharply to 47.23 at 7/15, rebounds to 57.84 at 12/15, and then drops again.

We attribute this instability to severe Task Interference (or negative transfer) within the shared weights of the 9B-parameter base model. The multimodal extraction tasks in our benchmark span vastly different distributions. Forcing a single compact model to optimize for these divergent objectives simultaneously leads to gradient conflicts and catastrophic forgetting. In contrast, our final DataClaw0-E (Expert) ensemble completely bypasses this interference. By routing tasks to domain-specific experts, each model perfectly fits its local data distribution, achieving a stable and superior combined score of 68.86. This fundamentally justifies the absolute necessity and scientific soundness of our “Domain-Decoupled + Expert Agent Routing” strategy.

Emergent Diversity and Intent Comprehension. To examine whether DataClaw0 merely memorizes surface patterns or demonstrates genuine generalization ability, we extract the semantic embeddings of three data sources: the original Raw Data, the data refined by the base model (Qwen3.5-9B), and the data refined by DataClaw0. We visualize their feature distributions using t-SNE dimensionality reduction (Figure 3).

The visualization shows that the feature space of the base model’s outputs expands slightly beyond that of the Raw Data, indicating limited diversification in its refinements. In contrast, the data refined by DataClaw0 exhibits a substantially broader and more evenly distributed coverage across the semantic space. This suggests that DataClaw0 injects stronger emergent diversity into the generated trajectories, revealing new clusters and long-tail patterns that the base model fails to capture. Rather than marginally shifting the training distribution, DataClaw0 reconstructs it into a richer and more heterogeneous semantic landscape.

To further quantify this capability, we build a DataClaw0-Intent evaluation subset composed entirely of vague, high-level instructions. As shown in Table 1, DataClaw0 significantly surpasses its base model (Qwen3.5-9B) and approaches the performance of Gemini 3.1-pro-preview on this challenging fuzzy-instruction benchmark.

### E Qualitative Analysis

#### E.1 Successful Case Studies

We showcase qualitative case studies covering representative domains to exemplify DataClaw0’s end-to-end process. Each case presents the raw input, user intent, and DataClaw0 output.

Please analyze this gameplay video and create one example of *Observe Surroundings Future Generation* data. The constructed question-answer pair should include a Chain-of-Thought (CoT) reasoning process.

user input

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

0

2 5 9

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

10 13 15 18

{

model response

"question": "What would the scene look like if the character ran to the right?", "cot": "The input frame shows the character running forward along a path with a low boulder slope on

the right. If the character then runs to the right, future frames show the character crossing the boulder slope and continuing along a dirt path with a large tree on the right.",

"input_video": [2,9], "output_image": [10,18]

},

Figure 8: Successful world-model or video-generation tailoring case.

Please analyze this video and create one example of *Tutorials or Keyframe Step-by-Step Breakdowns with Interwoven Text and Images* data. <video> The constructed question–answer pair should include a Chain-of-Thought (CoT) reasoning process.

user input

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

0

5 10 15

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

17 23 27 30

{

model response

"question": "Please provide a step-by-step diagram of the long jump movement.", "answer": "Step 1: The athlete sprints down the runway to gain speed. <img> Step 2: He takes off and flies

through the air. <img> Step 3: He lands safely in the sand pit. <img>",

"cot": "To create a tutorial breakdown, I identified the distinct phases of the jump: preparation/identity, the run-up, the jump itself, and the landing. I selected keyframes that best represent each of these stages to guide the viewer through the technique.",

"input_image": null, "output_image": [10,17,23]

}

Figure 9: Successful daily-life tailoring case.

#### E.2 Failure Cases

Although DataClaw0 improves structured multimodal data tailoring, it still fails in several scenarios. As shown in Figure 10, the constructed sample is semantically correct—the question asks how to navigate from the workspace to the bed, and the generated answer properly describes that transition. However, the CoT text describes the video as starting from the workspace and then moving toward the bed, while the actual input frames (0 → 20) are temporally ordered in the opposite direction (from the bed to the workspace).

Such temporal inconsistencies are occasionally observed when large multimodal language models generate narrative-style reasoning. They tend to infer a likely or contextually coherent event flow based on spatial cues rather than the exact chronological order of frames, a behavior commonly referred to as temporal hallucination. This phenomenon is largely inherent to the base model’s reasoning prior. Although minor, these cases highlight the intrinsic difficulty of enforcing strict temporal grounding in long-horizon multimodal reasoning, which remains an open challenge for current LLM-based architectures.

Please analyze this video and create one example of *VLN-style Navigation* data. <video>The constructed question–answer pair should include a Chain-of-Thought (CoT) reasoning process.

user input

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

0

6 10 11

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

15 19 21 25

{

model response

"question": "How do I navigate from the workspace to the bed?", "answer": "Turn away from the desk and walk towards the bed visible in the background.", "cot": "The video begins with a view of the desk and laptop. The camera then pans and moves forward,

revealing the bed structure in the distance, indicating the path from the workspace to the bed.", "input_image": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], "output_image": null

}

Figure 10: Failed daily-life tailoring case.

### F Limitations and Future Work

Despite its effectiveness, DataClaw0 still has several limitations. First, the current data scale remains moderate compared with large-scale pretraining corpora. Although our pipeline demonstrates promising data-refinement capability, future work should extend the construction process to million-scale multimodal supervision data to further improve coverage and robustness.

Second, DataClaw0 currently relies on user-provided raw streams as the input source. In other words, the system mainly performs refinement, filtering, and annotation over existing multimodal trajectories or interaction records. It does not yet support fully autonomous data creation from user intent alone. A more ambitious direction is to allow the agent to enter embodied, game, or GUI simulation environments, actively explore, collect interactions, and construct supervision data based only on high-level user intent. This would move DataClaw0 from raw-stream refinement toward autonomous data synthesis and self-improving data generation.

