#### PersonaVLM: Long-Term Personalized Multimodal LLMs

Chang Nie1 Chaoyou Fu1* Yifan Zhang2 Haihua Yang2† Caifeng Shan1 1Nanjing University 2ByteDance

changnie@smail.nju.edu.cn, bradyfu24@gmail.com

# arXiv:2604.13074v1[cs.CL]20Mar2026

##### Abstract

Multimodal Large Language Models (MLLMs) serve as daily assistants for millions. However, their ability to generate responses aligned with individual preferences remains limited. Prior approaches enable only static, single-turn personalization through input augmentation or output alignment, and thus fail to capture users’ evolving preferences and personality over time (see Fig. 1). In this paper, we introduce PersonaVLM, an innovative personalized multimodal agent framework designed for long-term personalization. It transforms a general-purpose MLLM into a personalized assistant by integrating three key capabilities: (a) Remembering: It proactively extracts and summarizes chronological multimodal memories from interactions, consolidating them into a personalized database. (b) Reasoning: It conducts multiturn reasoning by retrieving and integrating relevant memories from the database. (c) Response Alignment: It infers the user’s evolving personality throughout long-term interactions to ensure outputs remain aligned with their unique characteristics. For evaluation, we establish Persona-MME, a comprehensive benchmark comprising over 2,000 curated interaction cases, designed to assess long-term MLLM personalization across seven key aspects and 14 fine-grained tasks. Extensive experiments validate our method’s effectiveness, improving the baseline by 22.4% (Persona-MME) and 9.8% (PERSONAMEM) under a 128k context, while outperforming GPT-4o by 5.2% and 2.0%, respectively. Project page: https://PersonaVLM.github.io.

##### 1. Introduction

Multimodal Large Language Models (MLLMs) are increasingly integrated into the daily lives of millions of users [1, 46], serving as assistants, creative partners, and companions [19, 44, 47]. As their adoption grows, user expectations are shifting from general-purpose problemsolving towards personalized and empathetic long-term ex-

*Corresponding author. †Project leader.

periences [20, 42]. This shift poses a critical question: How can we evolve a general MLLM into a truly personalized assistant that accurately infers user intent, dynamically aligns its behavior with individual preferences and personality, and persistently remembers user-specific multimodal information over time? Addressing this question not only enhances user satisfaction and trust but also unlocks the significant value of MLLMs in domains like recommendation [38], healthcare [3], and education [48], to name a few.

Even advanced proprietary models exhibit limited capabilities in generating responses that cater to a user’s unique preferences and characteristics [6, 14, 50]. This challenge stems from two primary factors: on the model side, they are predominantly optimized within fixed windows and a one-size-fits-all paradigm [21]; on the user side, an individual’s preferences and personality are inherently diverse and dynamic, continuously evolving throughout ongoing interactions [14]. As illustrated in Fig. 1, a user initially expresses a preference for Sprite but subsequently shifts to Coca-Cola to mitigate anxiety in a multimodal interaction. When the user later expresses stress, a retrieval-augmented response fails to capture this shift, resulting in a misaligned recommendation. Furthermore, a generic aligned response may feel overly extraverted, failing to accommodate the introverted and neurotic user whose personality traits are often revealed subtly across many unrelated dialogues.

The root of these failures is that current personalization strategies are designed for static interactions. Specifically, input augmentation-based MLLMs like Yo’LLaVA [28] and RAP [11] specialize in recognizing user-specific concepts, but lack mechanisms to manage or update these memories, consequently failing to capture preference shifts from Sprite to Coca-Cola. Similarly, alignment techniques such as ALIGNXPERT [21] and Personality-Activation Search (PAS) [52] presuppose static user traits, preventing them from adapting to a user’s introversion revealed contextually over time. Therefore, we identify two foundational pillars for effective long-term personalization: (i) Personalized Memory Architecture. The ability to proactively construct and manage a dynamic, user-centric multimodal database. (ii) Memory Utilization and Response Alignment. The capacity

###### Remembering

###### Reasoning

###### Response Alignment

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

<think> Search calming habits...<retrieve> Sprite for anxiety. Now Prefers Coke. <answer> How about a quiet moment with a cold Coke to...

[Figure 8]

I know noisy places are stressful for you, so just relax and enjoy the dinner.

Don't worry, be proactive and turn those strangers into new friends.

Update

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

User's summer preference: Sprite

User now prefers Coca-Cola.

[Figure 15]

[Figure 16]

[Figure 17]

Outdated Memory

Session 11 @ 2025/6/1

Session 15 @ 2025/6/12

[Figure 19]

Session 3 @ 2025/5/24

Mis-aligned

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Feeling stressed today. Any suggestions?

Recently I prefer this One.

I drink Sprite to handle my summer anxiety.

Session 19 @ 2025/7/13

I remember you drink Sprite when stressed. Maybe try that?

I'm a bit worried about the strange party tomorrow.

[Figure 24]

Hidden Personality : High Neuroticism & Low Extraversion

Preference Shift

[Figure 25]

- Figure 1. Illustration of PersonaVLM’s three core capabilities for long-term personalization. PersonaVLM proactively remembers user preference shifts, performs multi-turn reasoning with retrieval, and generates responses aligned with the user’s personality. In contrast, existing personalization strategies, such as input augmentation and output alignment, will result in poor recommendations based on outdated memories and replies that are misaligned with the user’s personality.

to effectively utilize this database, employing reasoning and retrieval to generate responses that are deeply aligned with the user’s unique and evolving characteristics.

In summary, our contributions are fourfold:

- • We propose PersonaVLM, an innovative agent framework that achieves long-term personalization for MLLMs by integrating three core capabilities: proactive Remembering, multi-step Reasoning, and Response Alignment.
- • We introduce a personalized memory architecture featuring two key components: the PEM for dynamic alignment and a multi-type memory database comprising core, procedural, semantic, and episodic memories.
- • We establish Persona-MME, a comprehensive benchmark designed to evaluate the long-term and multi-faceted personalization capabilities of MLLMs, and use it to benchmark over 10 leading proprietary and open-source models.
- • We conduct extensive experiments to validate the effectiveness of PersonaVLM. Under a 128k context, PersonaVLM achieves improvements of 22.4% on Persona-MME and 9.8% on PERSONAMEM [14]. Notably, it surpasses GPT4o on these benchmarks and in open-ended evaluations.

Building on these pillars, we propose PersonaVLM, an innovative agent framework for long-term personalized interaction. First, we design a memory architecture that integrates a user personality profile and four distinct memory types (core for foundational attributes, semantic for facts, procedural for habits, and episodic for events) to store and manage user information. Second, building upon this architecture, a two-stage collaborative process transforms a general MLLM into a personalized assistant: (1) Response stage: Given the user’s multimodal input and context, PersonaVLM autonomously performs multi-step reasoning and memory retrieval to generate a response aligned with the user’s personality. (2) Update stage: The model infers and updates the user’s latent traits, quantified as Big Five scores1, through a momentum-based Personality Evolving Mechanism (PEM). Concurrently, it proactively extracts and summarizes key knowledge from the dialogue, updating the four memory types for future use. This integrated design endows PersonaVLM with the three key capabilities shown in Fig. 1.

##### 2. Related Work

The recent surge in LLM development has catalyzed the emergence of powerful MLLMs like GPT-4o [12], LLaVA [23], and the Qwen series [5, 45], showcasing exceptional capabilities in various general-domain tasks [47]. However, to evolve into a true personal assistant, a model must transcend the “one-size-fits-all” paradigm and tailor responses to individual user knowledge and preferences [24, 49]. Existing efforts to address this challenge can be categorized into three primary streams: adaptation-based, augmentation-based, and alignment-based personalization.

Alongside the design of the framework, we address the scarcity of suitable training data by developing a synthesis pipeline to generate a large-scale personalized, multimodal interactive dataset, comprising over 30k interactions across 500 unique personas. This self-contained dataset enables effective training while ensuring PersonaVLM can operate locally, thereby eliminating data privacy concerns. Furthermore, recognizing that existing benchmarks [24] are often static and text-centric, we establish Persona-MME, a comprehensive benchmark designed to evaluate the long-term, multi-faceted, and multimodal personalization of MLLMs.

Adaptation-based Personalization. Adaptation-based methods operate at the model level, encoding user-specific knowledge directly into trainable parameters through finetuning. Some works, for instance, employ parameterefficient fine-tuning (PEFT) to adapt LLMs for individ-

1We represent user personality using the Big Five traits [35]: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN), with each trait scored from 1 to 5.

[Figure 26]

### Multimodal Input PersonaVLM

Response & Personality

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

What did we ate last Friday?

name: xx preference: xx personality: xx

[Figure 31]

My records show that you and <A> had sandwiches and Coke at that restaurant last Friday.

Aligned Response

[Figure 32]

[Figure 33]

[Figure 34]

Query

Profile

[Figure 35]

[Figure 36]

[Figure 37]

Openness: 3.0 Conscientiousness: 2.0 Extraversion: 4.0 Agreeableness: 4.0 Neuroticism: 1.0

Reasoning Generation

[Figure 38]

This is my friend <A>

I eat ... for dinner

...

Big Five Personality

[Figure 39]

[Figure 40]

[Figure 41]

Conversation Context

[Figure 42]

[Figure 43]

Retrieve

Update

[Figure 44]

[Figure 45]

Memory Retrieval

Personality Evolving

###### Memory Extraction

[Figure 46]

Semantic, procedural, and episodic memory

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Core Memory

Procedural Memory

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

...

...

[Figure 58]

[Figure 59]

Semantic Memory

[Figure 60]

[Figure 61]

Episodic Memory

[Figure 62]

Response Stage Update Stage

[Figure 63]

Personalized Memory

[Figure 64]

- Figure 2. Overview of the PersonaVLM Framework. It leverages a personalized memory architecture and operates in two collaborative stages to achieve long-term personalization. In the Response Stage (blue arrows), it processes multimodal input, retrieves from personalized memory, and generates a personality-aligned response. Subsequently, in the Update Stage (pink arrows), the framework analyzes the completed interaction to extract key memories and update the user’s evolving personality profile1.

ual users or groups [37, 53]. This principle extends to the multimodal domain, where personalized MLLMs like MyVLM [2] and Yo’LLaVA [28] utilize learnable embeddings and soft prompts, respectively, to represent userspecific visual concepts. Such adaptation enables the model to transition from recognizing “a generic dog” to recognizing “the user’s pet dog.” However, their reliance on fine-tuning for each new user concept renders these methods less scalable and unable to capture the evolution of user preferences.

more sophisticated agentic frameworks, their utility in our context is severely constrained. Their primary focus on textonly data limits their applicability to truly multimodal inputs, and their reliance on proprietary models creates barriers for open research and raises significant privacy concerns.

Alignment-based Personalization. While standard LLM alignment, such as Reinforcement Learning from Human Feedback (RLHF) [30], enforces a universal, “one-size-fitsall” behavioral standard, it inherently fails to accommodate diverse user preferences and communication styles. As shown in Fig. 1 (right), an overly enthusiastic response, while generally helpful, might be inappropriate for an introverted user experiencing anxiety. Personalized alignment directly tackles this limitation by redefining the optimization objective from a universal standard to a user-specific one [24]. For example, Li et al. [21] incorporate user features into the input and use methods such as Direct Preference Optimization (DPO) [34] to align model responses with predefined user values. Another strategy, PAS [52], trains user-specific “probes” to guide personalization at inference time. While this approach enables inference-time adaptation, it is fundamentally limited. Its reliance on per-user training poses significant scalability challenges; moreover, the static nature of these probes means the alignment can become outdated as the user’s personality evolves over long-term interactions.

Augmentation-based Personalization. In contrast to model-level adaptation, augmentation-based approaches operate at the input level by equipping models with an external database to retain and retrieve user-specific memories [39, 41]. This paradigm is pivotal for transcending the limitations of fixed context windows in lifelong dialogues [7]. Related approaches [11, 29] extend personalization to the multimodal domain. They first employ open-vocabulary object detectors [25] to crop predefined visual concepts from images, which are then used for subsequent matching and retrieval. A key advantage of these methods is their training-free nature2, allowing them to accommodate new user concepts at inference time. However, they are limited by a manually predefined database and lack mechanisms to proactively manage and update knowledge from dynamic interactions. Moreover, while general-purpose memory architectures like A-Mem [43] and Memory OS [22] employ

Departing from prior works that address siloed aspects of personalization for MLLMs, such as static memory or fixed alignment, we introduce PersonaVLM: a unified agent framework designed for dynamic, long-term interaction.

2Following the specific terminology from [32], this denotes that new user concepts can be accommodated at inference time without requiring continual fine-tuning.

##### 3. Methods 3.1. PersonaVLM Framework

The overall architecture of the PersonaVLM agent is illustrated in Fig. 2. It is built upon a personalized memory architecture and operates through two collaborative stages of Response and Update to enable long-term personalization. Personalized Memory Architecture. This architecture is designed to construct and maintain a comprehensive, longterm user profile, storing two primary categories of information. First, it maintains a user personality profile (P), which provides a quantitative representation of the user’s personality as a vector of scores for the Big Five dimensions3 (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism). Second, it features a multi-type memory database (M) that captures a wide range of user-related knowledge. This timeline-based, agentic system supports flexible CRUD (create, read, update, delete) operations and is structured into four distinct memory types:

- • Core Memory: Stores the user’s fundamental attributes (e.g., human and persona blocks), inspired by MemGPT [31], and is dynamically updated to reflect their most current profile.
- • Semantic Memory: Distills event-independent, abstract knowledge by extracting key entities, relationships, and multimodal concepts.
- • Episodic Memory: Organizes raw dialogues into atomic, time-stamped events, each including a summary, dialogue turns, and keywords for efficient retrieval.
- • Procedural Memory: Records user-centric plans, goals, and recurring behaviors or habits.

Regarding their storage and persistence, while episodic and semantic memories are stored chronologically, core and procedural memories, along with the personality profile, retain only their latest versions to ensure relevance. Our design overcomes the limitations of existing systems, making our memory architecture: (a) Self-contained, avoiding proprietary model dependencies; (b) Explicitly personalized, prioritizing user-centric knowledge; and (c) Multimodal support, enabling a more holistic user understanding. For details on our memory architecture, refer to Appendix A.

Response Stage. The objective of this stage is to generate an aligned response by performing multi-step reasoning and timeline-based retrieval. Formally, this process at turn m can be formulated as:

Rm = R(Qm,Cm,Mm−1), (1)

where Rm is the personalized response. This response is conditioned on three inputs: the current user query Qm = (Tm,Im,tm), consisting of a text instruction Tm,

3Representing user personality via the Big Five traits is a prevalent approach in LLM alignment [52], rooted in psychological theories [16, 35].

an optional image Im, and a timestamp tm; the dialogue context4 Cm = {(Qi,Ri) | 0 < i < m and |ti − tm| ≤ ts}; and the state of the personalized memory database Mm−1.

- As depicted in the left panel of Fig. 2, the implementation of Eq. (1) is structured as a multi-step interaction between the PersonaVLM agent and its memory system. In the initial step, the model is prompted with the user’s instruction, context, and a consolidated profile (comprising the user’s core memory and personality). The model then outputs a detailed reasoning process and an action result. If the model determines that the current information is insufficient, it outputs retrieval conditions within a predefined template, including the time period and keywords for searching. The agent then executes the retrieval process by first isolating memories within the inferred time period and then performing a parallel search across semantic, episodic, and procedural memory types. The top-k results from each type are collected and fed back to the model to initiate the next reasoning step. This iterative process continues for multiple

rounds until the model outputs the final response Rm.

Two key insights drive the design of this stage. First, user queries are often highly context-dependent and contain anaphora (e.g., “that thing we just talked about”), which renders direct semantic retrieval imprecise. In contrast, a multi-turn, agentic retrieval process typically yields more precise and efficient results [15, 26]. Second, while some memory mechanisms [22, 40] may leverage query rewriting [27] to improve retrieval accuracy, they overlook crucial temporal cues (e.g., “this morning”). Our design addresses these gaps by enabling the model to determine not just what to retrieve, but also if retrieval is necessary and from when.

Update Stage. This stage, which executes automatically during idle periods after a response is generated, primarily involves two parts: evolving the user’s personality profile and proactively updating the memories. This process at turn m can be represented as:

(Pm,Mm) = U(Qm,Rm,Mm−1). (2)

Specifically, the user’s personality profile, Pm, is updated via our proposed Personality Evolving Mechanism (PEM). The PEM maintains a long-term personality profile as a vector p ∈ R5, corresponding to the Big Five dimensions [52].

- At each turn m, the PEM first infers a temporary set of

personality scores from the user’s latest query, Qm. These scores are normalized to form a turn-specific personality

vector, p′m. Subsequently, the long-term profile vector is updated using an exponential moving average (EMA):

pm ← λ · pm−1 + (1 − λ) · p′m, where λ ∈ [0,1] is a dynamic smoothing factor. To ensure high adaptability in

4We treat the recent conversation history (within a ts = 60 minute threshold) as short-term memory, and user inactivity beyond this threshold initiates a new session.

[Figure 65]

Basic User Info Generation Task Categories of Persona-MME Data Distribution

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

HUMAN: name, age, gender... PERSONA: education level

A journalist covering the social impact of xxx

[Figure 74]

Explicit Intent Inference

[Figure 75]

Personality Alignment

Implicit ReIcnotgen.t

###### Remembering

###### Reasoning Response Alignment

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 80]

...

[Figure 81]

[Figure 82]

[Figure 84]

Take it easy. I remember that noisy places can be stressful for you. Just focus on enjoying the dinner, you don't have to talk to anyone.

<think>User stressed. Search calming habits... <retrieve>Sprite for anxiety. New: Prefers Coke. <answer>Take a deep breath. How about a quiet moment with a cold Coke to help ...

[Figure 86]

Behavioral

Alignment

Don't worry, be proactive and turn those strangers into new friends.

[Figure 89]

[Figure 90]

Update

[Figure 91]

Intent

User's summer preference: Sprite

User now prefers Coca-Cola.

Pa

Personality Description Core Memory

Visual

[Figure 95]

[Figure 96]

[Figure 97]

tern Recog.

Detail Reca

###### Session 15 @ 2025/6/12 Outdated Memory

Session 3 @ 2025/5/2 Session 11 @ 2025/6/1

[Figure 100]

l

[Figure 102]

[Figure 103]

Feelingwhelmedreallytoday.stressedAny suggestions?”and over- Mis-aligned

###### Preference Long Conversation Generation

I remember you drink Sprite when it's hot. Maybe try that?

[Figure 104]

Session 19 @ 2025/7/13

Be

[Figure 105]

ory

Hidden Personality: High Neuroticism & Low Extraversion

[Figure 106]

[Figure 107]

[Figure 108]

Shift

havior

- ℛ1
- ℛ2

Semantic

[Figure 109]

[Figure 110]

Trac kin g

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

- n

Delivery

L

- o n

- Turn 1
- Turn 2

m

G oal

PersonaMME

Info. Recall

g-ter m

Me

[Figure 117]

[Figure 118]

Task Distribution

ConversationFlow

[Figure 119]

[Figure 120]

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

Relatio nship

###### Multimodal Input P-R3 Agent

Relatio n-

Response & Personality

[Figure 131]

[Figure 132]

natio

[Figure 134]

D y na

wth

[Figure 135]

d

Hey, do you remember what we ate last Friday?

Name: xx preference: xx personality:

[Figure 138]

Yes, I do. My records show that last Friday you and <A> were at that restaurant, and you had sandwiches and Coke.

[Figure 140]

Aligned Response

Tiere

C o m

Multimodal Long-Term Dynamic Diversity Personalized

I remember you and <A> had sandwiches and iced tea last Friday...

ship

Query Profile

pla

Gro

mics

Reasoning Generation

Openness: 3.0 Conscientiousness: 2.0 Extraversion: 4.0 Agreeableness: 4.0 Neuroticism: 1.0

pre.

[Figure 151]

Big Five Personality

[Figure 152]

[Figure 153]

Ex

[Figure 154]

[Figure 155]

[Figure 156]

Conversation Context

General. to

Retrieve

Update

[Figure 160]

Relation-

Personality Evolving

Memory Extraction

Memory Retrieval

ship Recog.

[Figure 161]

New Scenarios

Preference

[Figure 162]

[Figure 163]

[Figure 165]

[Figure 166]

[Figure 167]

Core Memory

Procedural Memory

semantic memory procedural memory episodic memory

...

Preference Recognition

...

[Figure 173]

[Figure 174]

[Figure 175]

ℛ

[Figure 176]

Turn m

[Figure 177]

[Figure 178]

Semantic Memory

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

RePLcroaegtfe..st

Episodic Memory

Implicit Pref. Recom.

[Figure 186]

Response Stage Update Stage

Personalized Memory

Interest Evolution Analysis

Context Response

Query

ℛ

[Figure 188]

[Figure 189]

Personality

Memory

[Figure 190]

[Figure 191]

(a) Data Synthesis Pipeline

(c) Samples and Tasks Distribution

(b) Persona-MME Overview

Figure 3. Overview of our data synthesis pipeline and Persona-MME. (a) The pipeline first constructs rich user personas and then simulates long-term, dynamic conversations, generating both the dialogue and intermediate memories. (b) Persona-MME provides a comprehensive evaluation of personalization by assessing 14 fine-grained capabilities. (c) Statistics for Persona-MME, which includes two context length configurations (32k and 128k) and contains over 2,000 in-situ6 cases.

early conversations while promoting stability over time, we employ a cosine decay schedule for λ. It starts with a low value (allowing rapid adaptation to initial user interactions) and gradually increases, making the profile more stable and less susceptible to minor fluctuations. Finally, the updated numerical vector pm is converted back into a descriptive textual summary, Pm, for use in the Response Stage.

CRUD operations; and (b) QA pairs containing complete, multi-step reasoning trajectories constructed offline. After SFT, the model is capable of generating well-formed reasoning and retrieval actions, providing a strong cold-start initialization for the subsequent stage.

Stage 2: Reinforcement Learning (RL). This stage aims to further enhance the model’s multi-turn reasoning capability. We employ Group Relative Policy Optimization (GRPO) [10], an improved PPO algorithm, to train the policy model πθ. During generation, we enforce a strictly structured output format: the model must first output its reasoning process within <think></think> tags, followed by either retrieval conditions in <retrieve></retrieve> tags or the final response in <answer></answer> tags. For each training sample {Q, R}, where Q is the user input and R is the preferred response, a group of multi-turn trajectories {τ1,...,τG} is sampled from the policy model. The reward for the i-th trajectory τi is calculated as:

Second, we selectively extract and update the four memory types, each with tailored logic. Semantic memory is updated after each turn, where key information such as user preferences, multimodal concepts, and explicit memorization requests is extracted and stored with timestamps and keywords. In contrast, core and procedural memory are updated at the end of each session; the agent analyzes the entire session’s dialogue to perform automated CRUD operations and keep these memories current. Finally, episodic memory is constructed by segmenting dialogues into distinct topics, with each entry containing a summary, relevant keywords, and the specific dialogue turns involved. See Appendix B.1 for the complete implementation pipeline.

), (3)

ri = facc( R,Rτi

) · fcons(Q,Rτi

) + 0.5 · fformat(Rτi

###### 3.2. Training of PersonaVLM

where facc, fcons, and fformat are reward functions for accuracy, logical consistency between reasoning and the final answer, and format adherence, respectively. We use Qwen330B-A3B [45] as an LLM-as-a-Judge to compute facc and fcons via zero-shot prompting. Following [10], the advantage for each trajectory is computed by standardizing its reward within the sampled group. During training, we cap the maximum number of retrieval attempts at three per trajectory, and the loss is computed exclusively on the generated tokens. Further details on the training data and implementation are provided in Appendix B.2.

We adopt Qwen2.5-VL-7B [5] as the backbone model for PersonaVLM and train it using a two-stage process.

Stage 1: Supervised Fine-Tuning (SFT). We perform SFT on a curated synthetic dataset of 78k samples to equip the model with foundational memory management and multiturn reasoning skills. The training data is synthesized via a pipeline introduced in the next section and comprises two primary types: (a) examples for memory mechanisms, including personality inference and the four types of memory

##### 4. Dataset and Persona-MME Construction

To enable both the implementation and evaluation of longterm dynamic personalization, we make two key contributions. First, to address the scarcity of high-quality training data, we construct a large-scale multimodal interaction dataset via a dedicated synthesis pipeline. Second, we establish Persona-MME, a comprehensive benchmark for evaluating personalization in multimodal settings. This dual effort is necessitated by existing datasets [21, 28], which are typically static, single-turn, or lack multimodal support.

Dataset Synthesis Pipeline. As illustrated in Fig. 3 (a), we design a synthesis pipeline to generate training data at scale. The process commences by sampling base personas from PersonaHub [9], which are then enriched with randomly assigned personality traits. This enrichment step generates a detailed role description and an initial user profile, forming the initial Core Memory. We employ Seed1.6-thinking5 to generate conversations guided by a structured flow. This process is governed by several key principles: (1) Long-term Dynamics: Dialogues extend over hundreds of turns to simulate interactions spanning weeks or months. To capture this longitudinal evolution, we probabilistically induce dynamic shifts in user preferences, topics, and personality traits. (2) Multimodality and Scenario Diversity: Over 15% of dialogues incorporate multimodal elements. The interactions span a wide range of real-world scenarios, from professional tasks to casual conversations. (3) Structured Supervision: The generation process is guided to produce not only the conversational dialogue but also the intermediate reasoning, retrieval, and memorization steps. This explicit structure provides rich supervisory signals for training the PersonaVLM framework. Further details on the data distribution and validation process are provided in Appendix C.

Persona-MME: Evaluating Long-Term Personalization of MLLMs. Existing benchmarks focus on siloed aspects of personalization. For instance, PERSONAMEM [14] evaluates a model’s ability to track a user’s evolving profile, ALIGNX-test [21] is centered on static alignment, and others like Yo’LLaVA [11, 28] assess user-specific concept understanding. However, none provide a holistic evaluation across the critical dimensions of dynamic personalization.

To fill this void, we introduce Persona-MME, a comprehensive benchmark comprising over 2,000 in-situ6 cases derived from 200 diverse personas. As depicted in Fig. 3 (b), Persona-MME is structured around seven core dimensions: Memory, Intent, Preference, Behavior, Relationship, Growth, and Alignment. Together, these dimensions

- 5Seed1.6-thinking is a commercial model with performance comparable to GPT-4o, selected for its balance of capability and cost-effectiveness.
- 6Queries are posed from the user’s first-person perspective at a specific point in the conversational history, simulating a realistic interaction [14].

GPT-4o LlaMA 3.1-8B InternVL3-8B Qwen2.5-VL-7B PersonaVLM (ours)

0.9

0.8

0.7

0.6

Accuracy

0.5

0.4

0.3

0.2

0.1

0.0

RevisitReasonsBehind PreferenceUpdates TrackingFull PreferenceEvolution ProvidePreference AlignedRecommendations

RecallUser SharedFactsAcknowledgeLatestUserPreferenceGeneralizeReasons

SuggestNewIdeas Overall

toNewScenarios

Figure 4. Quantitative evaluation across seven tasks on the PERSONAMEM (32k) benchmark.

encompass 14 fine-grained tasks, which are detailed in Table 5 in the Appendix. To accommodate different context lengths, we provide two evaluation configurations: a 32kcontext version for dialogues under 100 turns and a 128kcontext version for longer interactions, each containing cases from 100 distinct personas. Each test case comprises (1) a multiple-choice question assessing the model’s personalized memory and understanding, and (2) an optional personality test evaluating its alignment. This multi-faceted structure enables Persona-MME to evaluate an MLLM’s long-term personalization capabilities across diverse personas. Further details and statistics are provided in Appendix D.

##### 5. Experiments

In this section, we present a series of quantitative and qualitative experiments designed to validate our PersonaVLM framework. The evaluation in the main paper is structured to answer the following research questions (RQs):

- • RQ1: How effectively does PersonaVLM perform in personalized user understanding and memory recall?
- • RQ2: Can PersonaVLM achieve effective alignment by capturing a user’s evolving personality traits over time?
- • RQ3: How well does PersonaVLM perform in personalized open-ended generation?

For comprehensive evaluations of Persona-MME, ablation studies about memory components, and further discussions, please refer to Appendices D, E, and F, respectively.

###### 5.1. Personalized Understanding Evaluation

To evaluate personalized understanding (RQ1), we conduct experiments on two benchmarks: our Persona-MME and PERSONAMEM [14]. The latter includes seven task types specifically designed to assess a model’s ability to track dynamic user preferences over the long term. We evaluate all models under two long-context settings (32k and 128k tokens), with detailed results reported in Table 1 and Fig. 4. For comparison, we benchmark against several powerful models, including the proprietary GPT-4o [12] and strong open-source models such as Qwen2.5-VL-7B [5], LLaVAOneVision-1.5-8B [4], and InternVL3-8B/38B [51]. See

Table 1. Evaluation on the Persona-MME and PERSONAMEM benchmarks, tested at context lengths of 32k and 128k. We report accuracy (%) for Persona-MME (overall and across six aspects) and PERSONAMEM. The comparison includes two settings: full-context (“Full”) and retrieval-augmented generation (“RAG”). Best results are shown in bold. The GPT-4o results on PERSONAMEM are from [14].

Persona-MME

Context Model

PERSONAMEM Memory Intent Preference Behavior Relationship Growth Overall

GPT-4o 86.99 83.87 63.12 57.14 71.30 73.87 72.35 39.20 32k-Full

Qwen2.5-VL-7B 66.13 66.85 59.75 59.24 68.45 70.69 64.84 43.63 InternVL3-8B 56.45 76.24 57.20 54.35 69.05 74.14 64.04 52.97 InternVL3-38B 66.67 85.64 66.53 59.78 72.02 77.59 71.04 57.93 OneVision-1.5-8B 74.19 74.59 60.59 53.26 72.62 74.14 67.76 52.80

Qwen2.5-VL-7B 65.05 68.51 50.42 57.61 60.71 68.39 61.20 45.67

32k-RAG PersonaVLMSFT 67.20 70.17 49.58 57.07 70.24 80.46 64.84+3.64 52.12+6.45 PersonaVLMRL 69.89 76.80 58.05 69.02 73.21 86.78 71.48+10.28 56.53+10.86 GPT-4o 84.44 75.63 59.12 55.65 65.98 76.64 69.23 45.32

Qwen2.5-VL-7B 50.60 54.73 52.41 54.30 55.83 60.90 54.48 3.08

128k-Full

InternVL3-8B 57.23 68.92 53.48 54.97 69.17 76.69 62.43 36.62 InternVL3-38B 67.47 71.62 64.71 58.94 65.00 76.69 67.18 46.56 OneVision-1.5-8B 52.44 54.79 58.15 45.33 65.25 67.18 56.66 14.28

Qwen2.5-VL-7B 56.63 63.51 50.27 55.63 61.67 70.68 59.01 37.88

128k-RAG PersonaVLMSFT 67.47 75.68 59.36 51.66 71.67 81.95 67.18+8.17 43.60+5.72 PersonaVLMRL 69.28 77.70 61.50 60.26 75.00 87.97 71.05+12.04 47.28+9.4

Wins Ties Loses

2.5%

92.5%

PersonaVLM vs. Qwen2.5-VL-7B

5.0%

84.0%

7.5%

8.5%

PersonaVLM vs. InternVL3-8B

79.0%

5.0%

16.0%

PersonaVLM vs. GPT-4o

- Figure 5. Qualitative comparison on open-ended generation, evaluated by Gemini-2.5-Pro. The evaluation assesses both the factual accuracy and the personality alignment of the responses.

Appendix Fig. 10 for more comparisons with leading models.

Compared to strong open-source models of a similar size, such as InternVL3-8B and LLaVA-OneVision-1.5-8B (provided with full context), PersonaVLM shows improvements of 8.62% and 14.39% on Persona-MME in the 128k setting, respectively. While the personalization capabilities of these open-source models appear to improve with scale, PersonaVLM still outperforms the much larger InternVL338B by 3.87% on Persona-MME (128k). We also evaluate Qwen2.5-VL-7B augmented with a straightforward RAG setup, which retrieves the top five most relevant messages following the approach of [14]. Interestingly, the results show that RAG can be detrimental in short-context scenarios—degrading performance on preference understanding tasks by as much as 9.33%—while providing a substantial boost of 4.53% in long-context settings. Additionally, as shown in Table 1, the two-stage training process demonstrates clear effectiveness, yielding an average improvement of 5.35% on Persona-MME.

When benchmarked against the proprietary GPT-4o, our

Table 2. Evaluation of personalized alignment on the PersonaMME and P-SOUPS benchmarks.

Persona-MME P-SOUPS

Model

32k 128k Expertise Informativeness Style Overall

Qwen2.5-VL-7B 69.91 52.27 39.00 49.34 23.00 37.11 InternVL3-8B 55.75 55.56 47.16 51.81 26.16 41.71 InternVL3-38B 64.60 63.01 52.80 53.30 32.83 46.32 Qwen3-30B-A3B 80.09 83.06 51.67 54.60 35.16 47.14

baseline with different strategies

Self-Critic 59.73 57.66 39.67 49.33 23.67 37.50 Few-Shot - - 42.16 48.66 28.33 39.67

PersonaVLM (ours) 89.16 92.22 51.16 53.65 44.00 49.60

method achieves competitive results on Persona-MME and demonstrates notable improvements of 17.3% and 2.0% on the 32k and 128k configurations of PERSONAMEM, respectively. Furthermore, while PersonaVLM’s performance in memory recall lags behind that of GPT-4o with full context—a finding consistent with [18]—it demonstrates a significant advantage elsewhere. In particular, in Growth Modeling and Behavioral Awareness, PersonaVLM outperforms GPT-4o by over 10%.

###### 5.2. Personalized Alignment Evaluation

For RQ2, we conduct experiments on two benchmarks: the Alignment sub-task within Persona-MME and the PSOUPS [13], which comprise 812 and 1,800 test cases, respectively. The former assesses a model’s ability to determine if a response aligns with a user’s personality inferred from the conversational context. The latter evaluates personality alignment with a given user profile across three dimensions: Expertise, Informativeness, and Style.

We quantitatively compare PersonaVLM against several powerful open-source models, including InternVL3-8B/38B

[Figure 192]

###### Relevant Historical Context Current Turn

[Figure 193]

[Figure 194]

User : Thinking about that little train set from a while back... what color was the main part of the engine? (2024-10-02 20:31)

###### Failed visual recall

Qwen2.5-VL : The main part of the train set engine? Remember that rusty red. GPT-4o : The main engine color was primarily green—a grounding hue amidst the rustic rail chaos, perfect for stitching vintage texture into threads woven with nostalgia. PersonaVLM : That engine’s a deep, moody green—like the kind of hue that’d hum with your sister’s train set nostalgia and the Labyrinth’s gothic velvet vibes...

User : This tiny train set—it’s my sis’s old childhood thing… she’d lose herself in it for hours... (2024-09-15 11:33)

[Figure 195]

[Figure 196]

###### User :

[Figure 197]

Weekend vibes are back—gotta see what he’s up to! Bet it’s something fun! (2025-10-19 15:04 Sunday)

User : My friend Bob really enjoys playing the guitar on Sunday afternoons... (202510-12 16:22)

Qwen2.5-VL : The image shows a person taking a selfie with two black dogs. The person appears to be making a playful gesture with their hand, and...

###### Lacks personalization

GPT-4o : It looks like there's a lot of joy and energy this weekend, surrounded by the company of some adorable dogs. Spending time with pets can be a great way to relax.. PersonaVLM : Ohhh, that guitar + Sunday afternoon energy with Bob? Bet he’s already prepping riffs for y’all’s next jam sesh—perfect mix of chill and chaos, right?  ✨

[Figure 198]

User : I'm thinking of trying improv comedy. Given my personality, what's your take? (2024-09-25 16:40) Qwen2.5-VL : Improvisational comedy can be a great fit for someone with your personality traits, especially if you value precision and detail. Here are... GPT-4o : Improv comedy could be a fascinating outlet for you—it’s about embracing spontaneity and relinquishing control, which contrasts with ... PersonaVLM : Your conscientious mind and pattern - seeking instinct? Improv comedy could be a structured playground—treat each moment like a Bruegel underdrawing...Worth a try… if you frame it as a controlled experiment...

###### Hidden Personality

[Figure 199]

Generic

Overly formulaic, tonally misaligned

High: Openness, Conscientiousness Low: Agreeableness, Extraversion Moderately High: Neuroticism

[Figure 200]

- Figure 6. Qualitative comparison on open-ended generation tasks. Case studies demonstrate PersonaVLM’s superior capabilities in memory recall, context integration, and personality alignment compared to the baseline and GPT-4o.

and Qwen3-30B-A3B [45], with the latter being noted for its strong language capabilities. We also evaluate the baseline model augmented with different strategies, such as SelfCritic and few-shot prompting [50]. As shown in Table 2, PersonaVLM consistently outperforms existing models on both benchmarks. Notably, it leads the next-best model by 9.16% on Persona-MME and 2.46% on P-SOUPS, representing a >12% gain over the baseline. Interestingly, languagecentric models (e.g., Qwen3-30B-A3B) exhibit stronger alignment than multimodal counterparts like InternVL3-38B, with a 20% margin on Persona-MME (128k). These outcomes underscore PersonaVLM’s capacity for robust personality alignment.

GPT-4o, where PersonaVLM secures a 79% win rate versus a 16% loss rate. This is further corroborated by qualitative case studies in Fig. 6, which showcase PersonaVLM’s ability to perform accurate visual recall, integrate contextual memory, and maintain long-term personality alignment. In contrast, other models exhibit critical failures, such as memory hallucinations or tonally misaligned responses that ignore user-specific memories. These findings validate the generative capabilities of PersonaVLM for long-term personalization.

##### 6. CONCLUSION

This paper introduces PersonaVLM, a novel agent framework that enables long-term, dynamic personalization for MLLMs by integrating three core capabilities: Remembering, Reasoning, and Response Alignment. To support rigorous evaluation, we further propose Persona-MME, a comprehensive benchmark for personalized multimodal understanding. Experiments show that PersonaVLM significantly enhances a model’s personalization capabilities and consistently outperforms strong counterparts, including both proprietary GPT-4o and leading open-source alternatives. Our work provides a new paradigm for developing truly user-centric AI assistants, and future work will extend these capabilities toward a fully immersive multimodal experience.

###### 5.3. Qualitative Evaluation

To address RQ3 on open-ended generation, we conduct an automated evaluation using 200 questions randomly sampled from Persona-MME. We benchmark PersonaVLM against InternVL3-8B, Qwen2.5-VL-7B, and GPT-4o, employing Gemini-2.5-Pro [8] as an automated judge. Responses are assessed on two criteria: Accuracy and Personality Alignment, with PersonaVLM’s performance in pairwise comparisons classified as a “win,” “tie,” or “loss.” The evaluation prompt is provided in Fig. 23. As illustrated in Fig. 5, PersonaVLM achieves a substantially higher win rate than its peers. Particularly striking is its head-to-head performance against

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023.
- [2] Yuval Alaluf, Elad Richardson, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Myvlm: Personalizing vlms for user-specific queries. In ECCV, 2024.
- [3] Rawan AlSaad, Alaa Abd-Alrazaq, Sabri Boughorbel, Arfan Ahmed, Max-Antoine Renault, Rafat Damseh, and Javaid Sheikh. Multimodal large language models in health care: applications, challenges, and future outlook. Journal of medical Internet research, 2024.
- [4] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv:2509.23661, 2025.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv:2502.13923, 2025.
- [6] Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu, Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiaolong Chen, Xingmei Wang, et al. When large language models meet personalization: Perspectives of challenges and opportunities. World Wide Web, 2024.
- [7] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv:2504.19413, 2025.
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv:2507.06261, 2025.
- [9] Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. arXiv:2406.20094, 2024.
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv:2501.12948, 2025.
- [11] Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, and Xiangyu Yue. Rap: Retrieval-augmented personalization for multimodal large language models. In CVPR, 2025.
- [12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv:2410.21276, 2024.
- [13] Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. arXiv:2310.11564, 2023.

- [14] Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J Taylor, and Dan Roth. Know me, respond to me: Benchmarking llms for dynamic user profiling and personalized responses at scale. arXiv:2504.14225, 2025.
- [15] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Searchr1: Training llms to reason and leverage search engines with reinforcement learning. arXiv:2503.09516, 2025.
- [16] Oliver P John, Sanjay Srivastava, et al. The big-five trait taxonomy: History, measurement, and theoretical perspectives. 1999.
- [17] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 2019.
- [18] Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. 2025.
- [19] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 2024.
- [20] Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua. Hello again! llm-powered personalized agent for long-term dialogue. arXiv:2406.05925, 2024.
- [21] Jia-Nan Li, Jian Guan, Songhao Wu, Wei Wu, and Rui Yan. From 1,000,000 users to every user: Scaling up personalized preference for user-level alignment. arXiv:2503.15463, 2025.
- [22] Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, et al. Memos: A memory os for ai system. arXiv:2507.03724, 2025.
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [24] Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Wenhao Yu, Jieming Zhu, Minda Hu, Menglin Yang, TatSeng Chua, and Irwin King. A survey of personalized large language models: Progress and future directions. arXiv:2502.11528, 2025.
- [25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.
- [26] Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. arXiv:2508.09736, 2025.
- [27] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-augmented large language models. In EMNLP, 2023.
- [28] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’llava: Your personalized language and vision assistant. In NeurIPS, 2024.
- [29] Yeongtak Oh, Jisoo Mok, Dohyun Chung, Juhyeon Shin, Sangha Park, Johan Barthelemy, and Sungroh Yoon. Repic: Reinforced post-training for personalizing multi-modal language models. arXiv:2506.18369, 2025.

- [30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022.
- [31] Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. arXiv:2310.08560, 2023.
- [32] Renjie Pi, Jianshu Zhang, Tianyang Han, Jipeng Zhang, Rui Pan, and Tong Zhang. Personalized visual instruction tuning. arXiv:2410.07113, 2024.
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [34] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.
- [35] Sonia Roccas, Lilach Sagiv, Shalom H Schwartz, and Ariel Knafo. The big five personality factors and personal values. Personality and social psychology bulletin, 2002.
- [36] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.
- [37] Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. Democratizing large language models via personalized parameter-efficient fine-tuning. arXiv:2402.04401, 2024.
- [38] Qi Wang, Jindong Li, Shiqi Wang, Qianli Xing, Runliang Niu, He Kong, Rui Li, Guodong Long, Yi Chang, and Chengqi Zhang. Towards next-generation llm-based recommender systems: A survey and beyond. arXiv:2410.19744, 2024.
- [39] Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. In NeurIPS, 2023.
- [40] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv:2507.07957, 2025.
- [41] Jiale Wei, Xiang Ying, Tao Gao, Fangyi Bao, Felix Tao, and Jingbo Shang. Ai-native memory 2.0: Second me. arXiv:2503.08102, 2025.
- [42] Junda Wu, Hanjia Lyu, Yu Xia, Zhehao Zhang, Joe Barrow, Ishita Kumar, Mehrnoosh Mirtaheri, Hongjie Chen, Ryan A Rossi, Franck Dernoncourt, et al. Personalized multimodal large language models: A survey. arXiv:2412.02142, 2024.
- [43] Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv:2502.12110, 2025.
- [44] Zhenyu Xu, Hailin Xu, Zhouyang Lu, Yingying Zhao, Rui Zhu, Yujiang Wang, Mingzhi Dong, Yuhu Chang, Qin Lv, Robert P Dick, et al. Can large language models be good companions? an llm-based eyewear system with conversational common ground. In IMWUT, 2024.
- [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv:2505.09388, 2025.

- [46] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv:2408.01800, 2024.
- [47] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 2024.
- [48] Jifan Yu, Zheyuan Zhang, Daniel Zhang-li, Shangqing Tu, Zhanxin Hao, Rui Miao Li, Haoxuan Li, Yuanchun Wang, Hanming Li, Linlu Gong, et al. From mooc to maic: Reshaping online teaching and learning through llm-driven agents. arXiv:2409.03512, 2024.
- [49] Zhehao Zhang, Ryan A Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, et al. Personalization of large language models: A survey. arXiv:2411.00027, 2024.
- [50] Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin. Do llms recognize your preferences? evaluating personalized preference following in llms. arXiv:2502.09597, 2025.
- [51] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv:2504.10479, 2025.
- [52] Minjun Zhu, Yixuan Weng, Linyi Yang, and Yue Zhang. Personality alignment of large language models. In ICLR, 2025.
- [53] Yuchen Zhuang, Haotian Sun, Yue Yu, Rushi Qiang, Qifan Wang, Chao Zhang, and Bo Dai. Hydra: Model factorization framework for black-box llm personalization. In NeurIPS, 2024.

#### PersonaVLM: Long-Term Personalized Multimodal LLMs Supplementary Material

This supplementary material provides comprehensive details to complement the main paper, organized as follows:

- • Appendix A elaborates on our proposed memory architecture, detailing each memory component—including its storage, retrieval, and update processes.
- • Appendix B outlines the training and implementation details of PersonaVLM framework.
- • Appendix C presents a detailed analysis of our synthesized dataset, covering its distribution and the validation process.
- • Appendix D offers a comprehensive breakdown of Persona-MME, including its task taxonomy, detailed statistical analysis, and full evaluation results.
- • Appendix E presents additional experimental details, including ablation studies and the full set of prompts used in our framework.
- • Appendix F offers further efficiency analysis and limitations of PersonaVLM.

##### A. Details of the PersonaVLM Memory Architecture

As introduced in Section 3, the PersonaVLM memory architecture comprises two components: a User Personality Profile (P) and a Multi-Type Memory Database (M). This section provides a detailed exposition of how these memories are stored, updated, and retrieved.

###### A.1. Memory Storage

User Personality Profile (P). We quantitatively represent the user’s personality as a five-dimensional vector, p ∈ R5, where each element corresponds to a Big Five trait and is a floating-point value between 1 and 5. This profile is dynamically updated after each interaction turn m. Specifically, at the end of a turn, the model infers a personality vector, p′m ∈ R5, where each component is an integer score from

- 1 to 5 based on the user’s current input and context. The persistent personality profile p is then updated using an Ex-

ponential Moving Average (EMA): p ← λmp+(1−λm)p′m where the smoothing factor λm is dynamically adjusted to be more sensitive in early interactions and stabilize over time:

λm = 0.7 − 0.2 · cos min(50m,50)π . To ensure stability, this update is applied selectively. The process is skipped

if the inferred personality vector p′m consists solely of the neutral score (3), a condition that typically arises in nonpersonalized or neutral contexts. During the response generation stage, the personality profile P is provided to the model via structured prompting.

Core Memory. Core memory stores the user’s foundational and high-priority attributes and is included in every interaction turn. It is divided into two sub-components [31]:

- • Human: Factual user attributes, such as age, gender, preferences, and interests, with the user’s name as a mandatory field. This information provides PersonaVLM with a foundational understanding of the user’s background.
- • Persona: The user’s identity, roles (e.g., “a meticulous researcher”), and explicit requirements for the model’s interaction style, tone, and behavior.

Semantic Memory. Semantic memory [40] archives timeless, multimodal knowledge that is either explicitly provided by the user or autonomously inferred by the model. This knowledge is categorized as follows:

- • Explicit Directives: Direct commands from the user to remember specific information, which can be textual or visual. For example, a user might provide an image and say, “Remember the boy in this picture.”
- • Core Facts: Stable, factual information about the user disclosed during conversation, such as their profession, significant life events, or specific requirements for the agent’s behavior.
- • Preferences & Habits: User preferences for entities, visual styles, or activities, which can be either explicitly stated or implicitly revealed through behavior patterns.
- • Visual Concepts: User-specific visual concepts that arise in multimodal dialogues, such as friends, pets, or personal items. These are stored as a key-value pair linking a textual description to an image crop, formatted as “simple description <image>”.

Beyond these predefined categories, the agent autonomously determines at the end of each turn whether new semantic knowledge warrants storage. If so, it generates a structured output containing the reasoning process, memory content, and a set of keywords for future retrieval.

Episodic Memory. Episodic memory archives both summaries and raw data from past conversations. For each multiturn dialogue session, the model segments the conversation by topic. Each resulting topic-based episode contains three key elements: (a) a concise summary, (b) a set of keywords, and (c) the indices of the dialogue turns constituting that episode. To ensure no details are lost, the original dialogue data is never deleted; the episodic memory serves as a structured layer for organizing and retrieving this raw data.

Procedural Memory. Procedural memory tracks user goals and identifies recurring behaviors or habits by stor-

Distribution of SFT Data

Distribution of RL Data

10.3%

21.0%

Categories

23.4%

8.6%

QA 33,981

| |
|---|

Categories

Episodic Memory 6,672

| |
|---|

Open-ended 1,191

43.6%

8.6%

| |
|---|

Semantic Memory 16,000

| |
|---|

Multiple-Choice 3,150

| |
|---|

Procedural Memory 6,672

| |
|---|

Binary-Choice 1,323

| |
|---|

Core Memory 6,672

| |
|---|

Personality Evolving 8,000

20.5%

| |
|---|

8.6%

55.6%

Figure 7. Data composition for the training of PersonaVLM

ing procedural events from conversations. It primarily stores two types of information:

- • Long-term Goals: Ongoing projects, plans, or objectives that the user is working towards.
- • Habits & Routines: Repetitive behaviors or workflows that are automatically identified from user interactions.

Similar to Core Memory, this information is stored as keyvalue pairs, and only the latest version is retained.

###### A.2. Memory Retrieval

Memory retrieval is a critical step within the Response Stage, initiated when PersonaVLM determines that external knowledge is necessary to fulfill a user’s request. The process begins by generating a retrieval query encapsulated within <retrieve></retrieve> tags. This specifies a time period and keywords to guide the search. The time period is defined by start and end timestamps in a “YYYY − MM − DDHH : MM” format.

Textual Memory Retrieval. For text-based memories (i.e., procedural, semantic, and episodic), we employ a parallel multi-source retrieval strategy. First, all textual memories are encoded into dense vectors using the all-MiniLM-L6-v2 sentence transformer7. Given a user query, we perform a similarity search against the memory database. The top-k most relevant memories are retrieved from each category, where k is empirically set to 2, 4, and

- 2 for procedural, semantic, and episodic memories, respectively, unless otherwise specified. This entire process is accelerated by leveraging Facebook AI Similarity Search (FAISS) [17] for efficient indexing and retrieval.

Visual Concept Retrieval. This process is automatically triggered when the user’s input contains an image. First, we employ an off-the-shelf object detector, Grounding DINO [25], to extract salient objects from the input image. We then compute the cosine similarity between the CLIP [33] embeddings of these detected objects and the visual concepts stored in semantic memory. This process

7https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Algorithm 1 Operational Pipeline of PersonaVLM Input: User query Qm = (Tm,Im,tm), personality profile

Pm−1, memory database Mm−1, max reasoning steps N, model πθ, session threshold ts.

- 1: if tm − tm−1 ≥ ts then
- 2: Update Core, Procedural, and Episodic Memory based on the last session.
- 3: end if
- 4: Cm ← {(Qi,Ri) | 0 < i < m and |ti − tm| ≤ ts}
- 5: for n = 1 to N do
- 6: Sn ← πθ(Qm,Cm,Pm−1)
- 7: action,args ← Parse(Sn)
- 8: if action = retrieve then
- 9: (keywords, time period) ← args
- 10: Mretrieved ← Retrieve(Mm−1 keywords, time period)
- 11: Cm ← Cm ∪ Mretrieved
- 12: else if action = answer then
- 13: Rm ← args
- 14: break
- 15: end if
- 16: end for
- 17: Infer turn-specific personality p′m from Qm and update long-term profile pm.
- 18: Convert pm to textual summary Pm.
- 19: Extract and update Semantic Memory based on the current turn (Qm,Rm).

Output: Final response Rm, updated state (Pm,Mm).

mirrors the text-based semantic search, creating a unified retrieval mechanism across modalities.

###### A.3. Memory Management

Our memory management policies distinguish between raw conversational history and structured memory [43]. While the complete interaction history is retained for low-level access, the structured memories are managed according to the following policies. Semantic and Episodic memory are treated as purely additive; new entries detailing facts, concepts, or events are appended without modifying or deleting existing ones, thereby preserving an immutable historical record. In contrast, Core and Procedural memory maintain a single, canonical version of the user’s profile and habits. These memories are mutable and undergo CRUD operations at the end of each session to ensure they accurately reflect the user’s most current state.

##### B. Implementation Details of PersonaVLM B.1. Implementation Process

The end-to-end operational pipeline of PersonaVLM is detailed in Algorithm 1. In our offline implementation, a new

Table 3. The hyperparameters used in SFT and RL training.

Training stage SFT RL

Hyperparameter

Batch Size 64 72 Learning Rate 2e-5 2e-6 LR Scheduler cosine cosine Total Steps 1200 400 Max Pixels 230400 230400 Max Length 16384 16384 Warmup Ratio 0.03 0.05 deepspeed zero3 zero3 Number of Samples in a Group - 6 Num Iterations - 1 Repetition Penalty - 1.05

user session is initiated if the time elapsed since the last interaction, tm − tm−1, exceeds a predefined threshold ts (e.g., 60 minutes). At the start of a new session, a memory consolidation process is triggered to update the user’s longterm Core and Procedural memories based on the previous session.

###### B.2. Training Details

Training Data Composition. The composition of our training data for the SFT and RL stages is detailed in Fig. 7. The SFT dataset comprises a total of 78k samples. This dataset is constructed using the synthesis pipeline illustrated in Fig. 3 (a) and is further augmented with 6k user-related concept samples based on [11]. The SFT data is primarily split between question-answering (QA) pairs for reasoning (43.6%) and memory-related samples (56.4%). The memoryrelated category is further subdivided into a personality inference task (10.3%) and examples for the four memory types (46.1%). In contrast, the RL dataset consists of 5.6k samples, categorized into three types: open-ended QA with verifiable answers (21.0%), multiple-choice questions (55.6%), and binary-choice questions (23.4%).

Implementation Details. We implement our training pipeline based on the repositories Qwen-VL8 and ms-swift9. The hyperparameter settings for both the SFT and RL stages are detailed in Table 3. All experiments were conducted on a server equipped with 8 NVIDIA H800 GPUs. The entire two-stage training process completes in approximately 8 hours, comprising 2 hours for SFT and 6 hours for RL.

Group Relative Policy Optimization. GRPO [10] is an advancement over PPO [36] that refines policy optimization

- 8https://github.com/QwenLM/Qwen3-VL
- 9https://github.com/modelscope/ms-swift

by replacing the critic model with a relative evaluation mechanism. Instead of learning an absolute value function, GRPO estimates advantages by comparing the quality of multiple trajectories sampled within a group. For each training sample {Q, R}, where Q is the user input and R is the preferred response, the policy model πθ rollouts a group of multi-turn trajectories {τ1,...,τG}. The reward for each trajectory τi is calculated using Eq. (3). Based on these rewards, we then compute the normalized advantage Ati for each token by normalizing them across the sampled group. The optimization objective is:

LGRPO(θ) = E(Q, R)∼D,{τ  i}Gi=1∼πθold(·|Q)  1

|τi|

G

1 |τi|

min rit(θ) Ati,

G

t=1

i=1

clip(rit(θ),1 − ϵ,1 + ϵ) Ati − β DKL(πθ ∥πref) .

(4)

where rit(θ) = ππθ(τi,t|τi,<t)

θold(τi,t|τi,<t) is the probability ratio, πref is a reference policy, and β is a hyperparameter that controls the strength of the KL regularization. Detailed training settings are provided in Table 3.

Optimization Strategies. To improve the effectiveness and robustness of our retrieval mechanism, we implement several optimization strategies.

First, to mitigate retrieval redundancy within a single reasoning trajectory, the model is encouraged to use diverse query conditions (i.e., keywords and time periods). We enforce this by implementing a deduplication filter that prevents any single memory entry from being retrieved more than once per trajectory.

Second, we employ a dynamic top-k strategy during training to better prepare the model for varied information scenarios. Specifically, while we use fixed top-k values at inference (2 for episodic and 4 for semantic memories), these values are randomized during training, sampled uniformly from the ranges [2, 5] and [3, 6], respectively. This approach acts as a form of data augmentation, training the model to be robust to both sparse and dense information retrieval contexts.

##### C. Data Curation Details.

Data Distribution. We synthesize a large-scale, long-term multimodal dialogue dataset by sampling 700 unique personas from PersonaHub [9], allocating 500 for training and 200 for testing. The detailed distribution of the synthesized data is visualized in Fig. 8 and the top-right panel of Fig. 3. Training dialogues consist of 20 to 100 turns, spanning a simulated timeframe of up to one month. In contrast, test dialogues are designed to be more challenging, featuring

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

60

NumberofSessions

50

40

30

20

- 10

0

20 30 40 50 60 70 80 90 100

Dialogue Turns

35

|[Figure 201]|
|---|

30

TimeSpan(Days)

25

20

15

10

5

Figure 8. Distribution of the 500 long-term conversation samples in the training data.

longer conversations in two settings: 20–100 turns (for a 32k context window) and 100–500 turns (for a 128k context window), with a simulated duration of up to three months. This designed discrepancy between training and testing data allows for a rigorous evaluation of our memory architecture’s long-term capabilities.

It is important to note a distinction in how the dialogue data is utilized: the full, synthesized multi-turn dialogues serve as the database for retrieval, while the QA pairs used for model training feature re-generated answers. This is because the original answers have access to the complete dialogue history, whereas the training target must be an answer generated solely based on the current query and the retrieved memory content.

Multimodal Memory Formatting. To support multimodal knowledge, visual concepts in semantic memory are stored in a structured format: “Memory Content (Image Object: <class_name>)”. During the memory update process, Grounding DINO [25] is used to crop the corresponding object from the image. This cropped image patch is then paired with a simple textual description forming the input format for the model, i.e., “simple description <image>”.

Crucially, the system distinguishes between concrete visual objects and abstract preferences. For instance, if a user states, “I like this style of picture,” the system stores a textual fact, such as “User likes [style description],” rather than the raw image or its constituent objects. Also, episodic memory retains the original multimodal dialogue turns, including both text and full images, to preserve memory integrity.

Data Validation. To ensure the accuracy, safety, and overall quality of our synthesized dataset, we employ a twostage filtering process. First, we perform automated filtering using both rule-based checks and model-based validation. During data synthesis, the generation model outputs structured metadata, such as timestamps and dialogue turn indices for episodic topics. We leverage this metadata to apply

rule-based checks that validate data integrity, including the chronological consistency of timestamps and the completeness of episodic dialogues. Concurrently, a model-based selfcorrection mechanism verifies the safety and coherence of the generated content. Second, the automatically filtered data undergoes a human review. In this final step, human reviewers are tasked with identifying and removing any remaining erroneous, nonsensical, or repetitive dialogues, ensuring the final dataset is of high fidelity.

##### D. Persona-MME: Details and Statistics

Task Taxonomy. We provide the definitions for evaluated tasks in Table 5 and present illustrative examples in Fig. 9.

Data Statistics and Distribution. Persona-MME is designed to evaluate long-term personalization across seven key aspects, encompassing a total of 14 fine-grained tasks and comprising 2,034 in-situ test cases. It is important to note that a single test scenario may simultaneously assess multiple capabilities. Fig. 11 illustrates the distribution of these tasks. The benchmark consists of 13 primary tasks (from Visual Detail Recall to Generalizing to New Scenarios), which are distributed relatively evenly. The 14th task, personality alignment, is not a standalone category but is evaluated concurrently within 406 of the primary task cases.

The diversity of our evaluation set is a core design principle. We constructed 200 unique personas, each with a distinct fictional background, and crafted dialogues that span a broad spectrum of topics and scenarios to ensure comprehensive testing. The resulting topical breadth is visualized in Figure 12, which presents a word cloud of the most prominent keywords from the evaluation dialogues.

Further statistical analysis of Persona-MME is presented in Table 6. On average, each in-situ test case is grounded in a conversational history of 142.9 turns, of which 15.87% are multimodal. The average length of a test question is 22.7 words, while the average answer length is 3.05 words. A significant portion of questions, 34.02%, require visual information from the context to be answered correctly.

Comprehensive Evaluation. We present a comprehensive evaluation of over ten leading models on the 128k configuration of Persona-MME, with detailed results provided in Table 4 and Fig. 10. The evaluation spans a range of proprietary models (e.g., GPT-4o, GPT-5, Gemini-2.5-Flash, Claude-3.7-Sonnet) and open-source alternatives (e.g., the Qwen series, InternVL3-8B/38B, OneVision-1.5-8B). Our key findings are as follows:

- • Proprietary vs. Open-Source Gap: Proprietary models exhibit significantly better overall personalization capabilities than their open-source counterparts.
- • Challenges for Smaller and Multimodal Models: Smaller open-source multimodal models, such as

Visual Detail Recall

[Figure 202]

[Figure 203]

[Figure 204]

Explicit Intent Inference

###### Personalityl Alignment

Question: Thinking about the chaos from last month... when Cooper was doing his sock heist, what was my reason for getting poutine again? Answer : (c) Matched synth energy.

Question: That storyteller Raj from the beach the other day—remind me, what color was his turban? Answer : (c) Pink.

Question: Does the provided response <R> align with and adapt to the user's personality?

[Figure 205]

- (a) Yes, it aligns well.
- (b) No, it misaligns.

Semantic Information Recall

Implicit Intent Recognition

[Figure 206]

[Figure 207]

###### Explicit Intent Inference

[Figure 208]

Question: Was thinking about that raven incident last month. Maybe 'Monster Hospital' would fit this rainy mood too? Answer : (b) No, it's chaos.

Question: Hey, thinking back to last month, what was that guy with the leaf blower trying to do? Answer : (a) Unclog a drain.

Personality Alignment

Implicit ReIcnotgen.t

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Behavioral

Alignment

Intent

Pa

Visual

[Figure 213]

tern Recog.

Detail Reca

[Figure 214]

Behavioral Pattern Recognition

[Figure 215]

Relationship Dynamics Comprehension

l

[Figure 216]

Be

ory

Question: Okay, I've got a chaotic situation with a new client. How do I usually handle this kind of thing? Answer : (a) Apply truck restoration discipline.

[Figure 217]

[Figure 218]

Question: For a new, nervous client, which visual metaphor from last week would be more systemically sound for outreach: the pitcher or the elephant handlers? Answer : (d) The elephant handlers.

[Figure 219]

havior

Semantic

Trac kin g

- n

Delivery

L

- o n

m

G oal

## PersonaMME

Info. Recall

g-ter m

Me

Long-term Goal Tracking

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Relatio nship

Question: That tidal-temporal matrix we started a few weeks ago for the coastal alibi goal... did we ever complete it? Answer : (d) It's unknown.

Relatio n-

natio

Relationship Recognition

D y na

wth

d

Tiere

C o m

ship

Question: Speaking of the crew, between Jax, Moss, and Ryker, who seems to be my main partner-in-crime? Answer : (b) Jax.

pla

Gro

mics

pre.

[Figure 224]

Ex

[Figure 225]

[Figure 226]

General. to

Relation-

ship Recog.

New Scenarios

Preference

[Figure 227]

Preference Recognition

Tiered Explanation Delivery

[Figure 228]

[Figure 229]

RePLcroaegtfe..st

Implicit Pref. Recom.

[Figure 230]

[Figure 231]

Question: Based on my projects, what's a good next step for me to level up my skills with it? Answer : (a) Integrate multiple chaos inputs.

Latest Preference Recognition

Interest Evolution Analysis

Question: Of all the non-verbal rhythm ideas from last week, which one seems to be my main focus now? Answer : (a) Wave sync.

Generalizing to New Scearios

[Figure 232]

Question: My friend is struggling to get their team to adopt new creative habits. Based on my journey, what advice should

###### Implicit Preference Recommendation

[Figure 233]

Interest Evolution Analysis

Question: Thinking about Dad's nature shows and those zebras from last week... if I wanted to do something like that with him, what should we try? Answer : (b) Watching reruns.

[Figure 234]

Question: I used to be all about AI hype like blockchain, but now I can't stand it. Any idea what changed my mind? Answer : (c) Failed AI beach trip.

I give them?

Answer : (b) Start small with rituals.

Figure 9. Illustrative in-situ cases for the 14 task categories in Persona-MME, organized into the seven core personalization aspects.

Table 4. Comprehensive evaluation on the 128k configuration of Persona-MME. We compare PersonaVLM with proprietary and opensource models across 14 tasks: Visual Detail Recall (VDR), Semantic Information Recall (SIR), Explicit Intent Inference (EII), Implicit Intent Recognition (IIR), Latest Preference Recognition (LPR), Interest Evolution Analysis (IEA), Implicit Preference Recommendation (IPR), Behavioral Pattern Recognition (BPR), Long-term Goal Tracking (LGT), Relationship Recognition (RR), Relationship Dynamics Comprehension (RDC), Tiered Explanation Delivery (TED), Generalizing to New Scenarios (GNS), and Personality Alignment (PA).

Memory Intent Preference Behavior Relationship Growth Alignment

Model

###### Overall

VDR SIR EII IIR LPR IEA IPR BPR LGT RR RDC TED GNS PA Random 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 25.00 50.00 32.11 Proprietary models GPT-4o-mini 54.39 89.74 78.46 64.81 64.58 59.68 61.22 68.33 45.31 54.17 71.43 73.33 75.81 65.14 66.44

- GPT-4o 73.68 92.31 86.15 62.96 62.50 54.84 61.22 61.67 50.0 56.25 75.51 73.33 79.03 78.87 71.90

- GPT-5 85.71 98.72 93.85 67.92 74.47 70.97 65.31 76.67 70.97 85.11 81.63 76.19 75.81 92.25 82.95 Gemini-2.5-Flash 88.06 92.55 88.00 73.44 67.86 47.89 50.00 62.5 58.33 72.22 77.19 75.00 80.00 80.90 74.90 Claude-3.7-Sonnet 51.47 91.11 80.26 76.19 60.38 61.43 61.54 61.97 38.24 64.81 66.67 66.67 70.42 80.65 70.40

Open-source models

Qwen2.5-VL-7B 52.11 49.47 52.44 57.58 52.63 48.65 57.14 55.84 52.7 50.88 60.32 56.9 64.0 55.0 54.62 InternVL3-8B 29.58 77.89 74.39 62.12 59.65 54.05 46.43 66.23 43.24 61.40 76.19 75.86 77.33 54.17 60.08 InternVL3-38B 38.03 89.47 78.05 63.64 68.42 64.86 60.71 72.73 44.59 57.89 71.43 70.69 81.33 63.06 66.01 Qwen3-VL-8B 63.38 84.21 76.83 68.18 61.4 58.11 67.86 67.53 40.54 82.46 76.19 79.31 88.00 71.39 70.75 Qwen3-30B-A3B 29.58 85.26 82.93 75.76 70.18 63.51 64.29 63.64 44.59 68.42 77.78 82.76 86.67 81.39 72.65 OneVision-1.5-8B 42.86 59.57 59.26 49.23 62.5 46.58 69.09 48.68 41.89 73.21 58.06 64.91 68.92 53.93 55.88

PersonaVLM (ours) 50.70 83.16 81.71 72.73 59.65 54.05 73.21 58.44 62.16 75.44 74.60 82.76 92.00 92.22 77.08

Qwen2.5-VL-7B, InternVL3-8B, and OneVision-1.5-8B, particularly struggle with personality alignment, with their performance often being comparable to a random baseline.

In contrast, large language-centric models like Qwen330B-A3B can achieve superior overall scores, outperforming even larger multimodal models like InternVL3-38B,

82.95

80

77.08

74.90

72.65 71.90 70.75 70.40

66.44 66.01

60.08

60

OverallScore

55.88 54.62

40

32.11

20

0

GPT-5PersonaVLM(ours)Gemini-2.5-FlashQwen3-30B-A3B GPT-4oQwen3-VL-8BClaude-3.7-SonnetGPT-4o-miniInternVL3-38BInternVL3-8BOneVision-1.5-8BQwen2.5-VL-7BRandom

- Figure 10. Overall performance on Persona-MME (128k), ranking PersonaVLM against various proprietary and open-source models.

VisualDetailRecall Semantic InformationRecall ExplicitIntentInference ImplicitIntentRecognition

Latest Preference

Recognition

Interest Evolution Analysis

Implicit Preference

Recommendation

BehavioralPattern Recognition Long-termGoalTracking Relationship Recognition

RelationshipDynamics Comprehension

Tiered ExplanationDeliveryGeneralizingto NewScenarios

Personality Alignment

0%

2%

4%

6%

8%

10%

12%

14%

16%

18%

20%

Percentage

92 94

98

83

80

92

64

92 92

82

86

92

82

226

71

95

82

68

57

74

56

77

74

59

65

58

75

180

Context Length

32k 128k

- Figure 11. Distribution of the 14 fine-grained tasks in Persona-MME across its 32k and 128k context length configurations, with the number of test cases indicated for each task.

despite their inherent limitations on visual tasks (e.g., VDR).

• No Single Dominant Model: Even the top-performing model, GPT-5, does not dominate across all sub-tasks. It

is surpassed by other models in specific areas, including Growth Modeling and Visual Detail Recall, highlighting the complexity of holistic personalization.

###### • Effectiveness of PersonaVLM: Our PersonaVLM frame-

Table 5. Task definitions for the Persona-MME evaluation suite.

###### Evaluation Aspect Task Definition

Visual Detail Recall Assesses the ability to recall fine-grained visual details from previously shared images.

Memory

Semantic Information Recall Evaluates long-term memory for semantic information (e.g., events, preferences, context) from conversational history.

Explicit Intent Inference Assesses understanding a user’s explicitly stated intent from past multimodal context (e.g., linking a suitcase photo to text about a "business trip").

Intent

Implicit Intent Recognition Tests inferring a user’s latent intent from the current context (e.g., deducing a calm beach is unsuitable for a user known to enjoy surfing).

Latest Preference Recognition Assesses prioritizing recent behavioral evidence (e.g., a rock concert photo) over older, contradictory stated preferences (e.g., "I only like classical music").

Interest Evolution Analysis Evaluates explaining the evolution of a user’s interests by linking it to a specific causal event from their history (e.g., a change in diet due to a mentioned allergy).

Preference

Implicit Preference Recommendation

Tests recommending based on implicit values inferred from user history (e.g., suggesting a bicycle over a car to an eco-conscious user).

Behavioral Pattern Recognition Assesses recognizing recurring behavioral patterns to predict actions (e.g., predicting a Saturday café visit based on a history of Saturday café photos).

Behavior

Long-term Goal Tracking Tests tracking a long-term goal’s progress, including identifying when its status is unknown due to insufficient information.

Relationship Recognition Tests identifying relationship significance based on the frequency and context of individuals in multimodal history (e.g., identifying a closer friend by their more frequent appearance).

Relationship

Relationship Dynamics Comprehension

Assesses inferring relationship dynamics from context to adapt communication style (e.g., using a professional tone for a colleague).

Tiered Explanation Delivery Tests tailoring explanations to a user’s evolving skill level. Generalizing to New Scenarios Assesses the ability to generalize a user’s behaviors and preferences

Growth

to novel scenarios. Alignment Personality Alignment Assesses the ability to infer a user’s personality traits from long-term interactions and adapt its responses accordingly.

work significantly enhances the baseline model’s performance by 22.46%. The most substantial improvements are concentrated in the sophisticated dimensions of Growth and Alignment, underscoring the targeted benefits of our approach.

Comparison with Existing Benchmarks. As shown in Table 7, Persona-MME provides a more comprehensive evaluation of personalization compared to existing benchmarks. Specifically, it is the only benchmark that combines longterm interaction scenarios, multimodal (vision and text) inputs, and a holistic assessment of memory, understanding, and alignment capabilities.

Table 6. Key statistics of the Persona-MME.

###### Statistic Value

Avg. turns per dialogue 142.9 Multimodal turn ratio 15.87% Avg. question length 22.7 words

Avg. answer length 3.05 words Image-related question ratio 34.02%

[Figure 235]

- Figure 12. Word cloud of keywords from the dialogue data in Persona-MME, illustrating the rich diversity of conversation scenarios and topics.

Quality Assurance. To ensure the quality of PersonaMME, every test case underwent a rigorous manual review process. We first generated initial questions using the Gemini-2.5-Pro API10. Subsequently, a team of four annotators meticulously reviewed each case against three key criteria: (a) Consistency: ensuring the question aligns with its assigned task category. (b) Accuracy: verifying the correctness of the ground-truth answer. (c) Alignment Validity: assessing whether the model’s response in alignment tests appropriately adapts to (or conflicts with) the predefined personality traits. Any examples found to be ambiguous or conflicting were discarded. This comprehensive review process required approximately 40 person-hours to complete.

##### E. More Experimental Details

###### E.1. Benchmarks

PERSONAMEM [14]. This is a recent benchmark featuring synthetic, multi-session, and timeline-aware conversational data, designed to evaluate an LLM’s ability to remember, track, and generalize from personalized user profiles and preferences. It includes seven types of in-situ user queries, including: recall user-shared facts, suggest new ideas, acknowledge latest user preferences, track full preference evolution, revisit reasons behind preference updates, and provide preference-aligned recommendations. We conduct evaluations under two context-length settings, 32k and 128k to-

10We use the gemini-2.5-pro-preview-06-05 model.

32k 128k

75

70

Accuracy

65

60

0 1 2 3 4

Number of Topics

Figure 13. Ablation study on the number of retrieved episodic topics for Persona-MME.

kens. The settings comprise 589 and 1,362 multiple-choice questions, respectively, with the larger setting derived by sampling half of the personas from the original 2,728. Performance is measured by accuracy, and the comparative results are reported in Table 1 and Fig. 4.

P-SOUPS [13]. P-SOUPS assesses LLM personalization across three preference dimensions: Expertise, Informativeness, and Style, each containing 600 test cases for a total of 1,800. A single test case consists of a user prompt, a profile, and a pair of responses: one aligned with the profile (the “chosen” response) and one misaligned (the “rejected” response). The model is tasked with selecting the aligned response from the pair, and performance is measured by accuracy. For our few-shot experiments, we augment the input with a single example of Pair-wise Comparative Feedback, as provided by the benchmark.

###### E.2. Ablation Study

Effectiveness of Different Memory Types. We present an ablation study on the memory components of PersonaVLM architecture in Table 8. The results consistently show that removing any single memory type degrades overall performance, a trend that holds across both the 32k and 128k context settings. Notably, Episodic memory emerges as the most critical component; its removal leads to a substantial performance drop of 12.41% and 5.19% in the two settings, respectively, while removing other memory types results in a performance drop of less than 2%. Delving into sub-task performance, we observe that Procedural memory has a strong impact on tasks related to Behavior and Relationship. Collectively, these findings suggest that the different memory types fulfill distinct yet complementary roles, and all are integral to the holistic performance of the PersonaVLM agent.

- Table 7. Comparison of Persona-MME with existing personalization benchmarks. Abbreviations are defined as follows. Modality: V (Visual), T (Text). Capabilities: U (Personalized Understanding), M (Memory), A (Alignment). Answer Type: MC (Multiple Choice), BC (Binary Choice).

Benchmark Modality Long-Term Capabilities Answer Type

PERSONAMEM [14] T ✓ M + U MC P-SOUPS / ALIGNX-test [13, 21] T A BC Yo’LLaVA [28] V U BC RAP [11] V M + U MC

Persona-MME (ours) V + T ✓ U + M + A MC + BC

- Table 8. Ablation study of PersonaVLM components on the Persona-MME benchmark. The evaluation shows the performance impact of removing (“w/o” denotes “without”) key components, specifically the individual memory types (Core, Procedural, Semantic, Episodic) and the reasoning capability.

Persona-MME

Context Setting

Memory Intent Preference Behavior Relationship Growth Overall PersonaVLM 69.89 76.8 58.05 69.02 73.21 86.78 71.48

- - w/o Core 73.66 74.59 59.32 63.59 67.26 83.91 69.80−1.68

- - w/o Procedural 72.58 79.01 59.32 59.78 70.24 85.06 70.33−1.15

32k - w/o Semantic 66.67 72.38 59.32 66.30 72.02 85.63 69.71−1.77

- - w/o Episodic 33.77 55.35 56.68 66.27 69.84 74.07 59.07−12.41

- - w/o Reasoning 69.57 69.7 60.48 62.14 71.76 83.61 68.73−2.75 PersonaVLM 69.28 77.70 61.50 60.26 75.00 87.97 71.05

- - w/o Core 69.28 77.70 64.17 58.28 72.50 84.21 70.39−0.66

- - w/o Procedural 69.88 77.03 61.50 54.97 69.17 87.97 69.39−1.66

128k - w/o Semantic 67.47 73.65 60.43 60.93 74.17 88.72 69.94−1.11

- - w/o Episodic 50.60 68.92 60.96 62.25 70.00 88.72 65.86−5.19

- - w/o Reasoning 59.21 71.97 57.87 65.87 73.39 80.00 67.32−3.73

Episodic Memory Configuration. Given the critical role of episodic memory, we conduct an ablation study on the number of retrieved memory topics. As shown in Fig. 13, the overall accuracy on Persona-MME initially increases with the number of retrieved topics before performance saturates. To strike a balance between performance and computational efficiency, we select two topics as the default setting for all of our main experiments.

Table 9. Ablation study on the PEM component.

Setting Expertise Informativeness Style Overall PersonaVLM 51.1 53.6 44.0 49.6

- w /o PEM 48.5−2.6 50.2−3.4 38.3−9.2 45.6−4.0

Effectiveness of the Reasoning Capability. We validate the effectiveness of PersonaVLM’s multi-step reasoning and retrieval capability with two key findings. First, the full PersonaVLM model, trained with reinforcement learning,

demonstrates a significant 4–7% performance gain over its SFT-only baseline on Persona-MME and PERSONAMEM (Table 1). This highlights the benefit of the overall training process in cultivating this advanced reasoning behavior. To further isolate the contribution of this capability, we conduct an ablation study. Disabling multi-turn retrieval during the response stage results in performance drops of 2.75% and 3.73% at the 32k and 128k context settings, respectively (Table 8). These results underscore the critical role that multi-step reasoning and retrieval play in achieving robust long-term personalization.

###### E.3. Personality Evolving Mechanism

In Fig. 14, we visualize how PEM captures the evolution of personality traits inferred from user interactions across diverse personas. Furthermore, as shown in Table 9, our ablation study on P-SOUPS demonstrates the effectiveness of PEM. Finally, Table 10 provides examples of how PersonaVLM adapts its responses based on the inferred personality traits to meet the user’s personalized needs. These results

Table 10. An overview of the Big Five personality dimensions, with examples illustrating how our method generates adaptive responses to the same user query by adjusting inferred trait scores (high vs. low).

###### Personality Characteristic Example

Openness Low: practical, conventional, factual, cautious, pragmatic

Query: I’m planning a one-week trip to Rome. Can you suggest an itinerary for me?

Low score response: Rome is such a rich city with so much history! For a one-week trip, here’s a practical itinerary that balances must-see spots with some flexibility. Day 1...Day 7... Enjoy your trip to Rome!

High: creative, imaginative, curious, innovative, abstract, broad-minded

High score response: Clare, planning a week in Rome? Let’s mix iconic spots with hidden gems—here’s a flexible itinerary you can tweak! Day 1...Day 7. Adjust days as you please; Rome’s best-kept secrets are waiting for you to discover them!

Conscientiousness Low: spontaneous, flexible, impulsive, carefree

Query: I signed up for a 5k race that’s three months away. Now I need to figure out how to train for it.

High: organized, disciplined, detail-oriented, efficient, reliable.

Low score response: For a 5k three months out, start small—maybe aim for 20 minutes of walking/jogging 3 days a week, building up by 1 minute each week. Use a calendar to map it out (or just keep a notebook handy). If you hit a rough patch, mix in some light stretching or yoga to keep things loose. Start slow, stay consistent—you’ll build endurance without burning out.

High score response: Let’s break this into a structured 3-month plan! Here’s what works for many runners: Month 1 (Weeks 1–4): Build Base...For consistency, use a calendar to log your runs—tracking distance/time helps keep you on track. Also, mix in strength training (like leg curls, lunges) 2 days/week to build endurance.

Extraversion Low: reserved, quiet, solitary, reflective, thoughtful

Query: My birthday’s this weekend—should I go all out with a big party, or keep it intimate with a small dinner among my closest friends? What would you choose?

High: outgoing, sociable, energetic, talkative, assertive.

Low score response: Since you lean toward calm interactions, keeping it small might let you savor every moment without the pressure of impressing a crowd.

High score response: That’s such a fun dilemma! Since you’re all about the good vibes (hello, high extraversion!), a big party with friends could be electric—imagine laughter, games, and maybe even a surprise performance (or two!).

Agreeableness Low: competitive, critical, skeptical, direct, analytical

Query: In a team meeting, a colleague made a small mistake. Should I point it out, or stay quiet?

Low score response: It depends on the context. If the mistake is minor ... But if it’s something that could clarify or prevent confusion later, pointing it out—clearly, without criticizing—could help. Think about what’s best for the team’s flow right now.

High: cooperative, compassionate, empathetic, trusting, kind.

High score response: Clare, when a colleague makes a small mistake in a team meeting, there’s no one “right” answer—both ways have pros and cons. Think about how the mistake affects the team’s goal, your relationship with that colleague, and your comfort level. Either way, you’re being thoughtful either way!"

Neuroticism Low: calm, confident, resilient, secure, emotionally stable

Query: I have an important presentation next week. I can’t stop thinking about all the things that could go wrong. Low score response: It’s normal to feel some pressure before a big presentation. Focus on what you can control...

High: anxious, worryprone, insecure, stressed, moody.

High score response: I know, it’s been unbearable. Staying cool feels like such a win when it’s this hot out. Hope you’re finding ways to beat the heat—maybe some refreshing drinks or light walks.

Table 11. Efficiency comparison of PersonaVLM

Method Avg.Tokens Avg.Times (s) baseline 43530 8.4

PersonaVLM w/o reasoning 2726 2.09 PersonaVLM 2170 10.18

demonstrate that the inclusion of PEM enables PersonaVLM not only to capture a user’s evolving personality during longterm interactions accurately but also to generate responses that are consistently aligned with these inferred traits.

###### E.4. More Interaction Examples

In Fig. 15, we provide comparative cases of open-ended interactions between PersonaVLM, the baseline model, and GPT4o. These examples demonstrate PersonaVLM’s superior comprehensive personalization capabilities during long-term interactions.

###### E.5. Prompts Used in Our Framework

We present the prompts used in PersonaVLM across several figures. The prompts for multi-turn reasoning and retrieval are shown in Figs.16 and 17. The prompt for PEM personality inference is shown in Fig. 18. The corresponding prompts for updating the different memory modules are provided in Figs. 19, 20, 21, and 22. The prompt for the open-generation task evaluation is presented in Fig. 23.

##### F. Further Discussion

Efficiency and Data Security. We evaluate model efficiency using two key metrics: average token consumption per request and average response time (in seconds). As detailed in Table 11, our analysis is based on 100 randomly selected samples from the Persona-MME, comparing the baseline model (Qwen2.5-VL-7B), PersonaVLM without its reasoning capability (PersonaVLM w/o reasoning), and the standard PersonaVLM. It is important to note that the measured time covers the end-to-end process from user input to receiving the complete response. The memory update operation in PersonaVLM is performed asynchronously after a response is delivered and is therefore excluded from this timing analysis.

The results highlight two key findings. First, PersonaVLM without reasoning demonstrates significant efficiency gains over the baseline, reducing average token consumption by a remarkable 93.7% and achieving a 4.8× speedup. Second, when equipped with its reasoning capability, the standard PersonaVLM further decreases token consumption by 20.4% compared to its non-reasoning counterpart. However, the computational overhead of the reasoning process results in a 21.1% increase in response time relative to the

baseline. This reveals a clear trade-off between advanced reasoning capabilities and response latency.

Regarding data security, PersonaVLM’s memory and retrieval operations function independently of external commercial model APIs. This self-contained architecture inherently ensures data security and mitigates privacy concerns.

Limitations. PersonaVLM has several limitations. First, it does not currently support person recognition and tracking from video or audio inputs. Second, its overall performance is inherently constrained by the capabilities of the underlying baseline model, despite significant personalization gains. Third, the memory system is primarily timeline-based and does not yet establish connections or merge related episodic memories occurring at different times. Addressing these limitations is a key direction for our future work.

C A

- User 1

Turn 0

O

C A

E

N

Turn 30

O

C A

E

N

Turn 60

O

C A

E

N

Turn 91

O

C A

E

N

Turn 121

O

C A

E

N

Turn 151

O

C A

E

N

Turn 182

O

C A

E

N

Turn 212

O

C A

E

N

Turn 242

O

C A

E

N

Turn 273

O

C A

E

N

- User 2

Turn 0

O

C A

E

N

Turn 42

O

C A

E

N

Turn 85

O

C A

E

N

Turn 128

O

C A

E

N

Turn 171

O

C A

E

N

Turn 214

O

C A

E

N

Turn 257

O

C A

E

N

Turn 300

O

C A

E

N

Turn 343

O

C A

E

N

Turn 386

O

C A

E

N

- User 3

Turn 0

O

C A

E

N

Turn 38

O

C A

E

N

Turn 77

O

C A

E

N

Turn 116

O

C A

E

N

Turn 154

O

C A

E

N

Turn 193

O

C A

E

N

Turn 232

O

C A

E

N

Turn 270

O

C A

E

N

Turn 309

O

C A

E

N

Turn 348

O

C A

E

N

- User 4

Turn 0

O

C A

E

N

Turn 30

O

C A

E

N

Turn 60

O

C A

E

N

Turn 91

O

C A

E

N

Turn 121

O

C A

E

N

Turn 152

O

C A

E

N

Turn 182

O

C A

E

N

Turn 213

O

C A

E

N

Turn 243

O

C A

E

N

Turn 274

O

C A

E

N

- User 5

Turn 0

O

C A

E

N

Turn 47

O

C A

E

N

Turn 95

O

C A

E

N

Turn 143

O

C A

E

N

Turn 190

O

C A

E

N

Turn 238

O

C A

E

N

Turn 286

O

C A

E

N

Turn 333

O

C A

E

N

Turn 381

O

C A

E

N

Turn 429

O

C A

E

N

- User 6

Turn 0

O

C A

E

N

Turn 26

O

C A

E

N

Turn 52

O

C A

E

N

Turn 78

O

C A

E

N

Turn 104

O

C A

E

N

Turn 131

O

C A

E

N

Turn 157

O

C A

E

N

Turn 183

O

C A

E

N

Turn 209

O

C A

E

N

Turn 236

O

C A

E

N

- User 7

Turn 0

O

C A

E

N

Turn 54

O

C A

E

N

Turn 108

O

C A

E

N

Turn 163

O

C A

E

N

Turn 217

O

C A

E

N

Turn 271

O

C A

E

N

Turn 326

O

C A

E

N

Turn 380

O

C A

E

N

Turn 434

O

C A

E

N

Turn 489

O

C A

E

N

- User 8

Turn 0

O

C A

E

N

Turn 45

O

C A

E

N

Turn 91

O

C A

E

N

Turn 136

O

C A

E

N

Turn 182

O

C A

E

N

Turn 227

O

C A

E

N

Turn 273

O

C A

E

N

Turn 318

O

C A

E

N

Turn 364

O

C A

E

N

Turn 410

O

C A

E

N

- User 9

Turn 0

O

C A

E

N

Turn 29

O

C A

E

N

Turn 58

O

C A

E

N

Turn 87

O

C A

E

N

Turn 116

O

C A

E

N

Turn 145

O

C A

E

N

Turn 174

O

C A

E

N

Turn 203

O

C A

E

N

Turn 232

O

C A

E

N

Turn 262

O

C A

E

N

- User 10

Turn 0

O

C A

E

N

Turn 35

O

C A

E

N

Turn 70

O

C A

E

N

Turn 106

O

C A

E

N

Turn 141

O

C A

E

N

Turn 176

O

C A

E

N

Turn 212

O

C A

E

N

Turn 247

O

C A

E

N

Turn 282

O

C A

E

N

Turn 318

O

C A

E

N

- User 11

Turn 0

O

C A

E

N

Turn 45

O

C A

E

N

Turn 90

O

C A

E

N

Turn 135

O

C A

E

N

Turn 180

O

C A

E

N

Turn 226

O

C A

E

N

Turn 271

O

C A

E

N

Turn 316

O

C A

E

N

Turn 361

O

C A

E

N

Turn 407

O

C A

E

N

- User 12

O

E

N

Turn 0

Turn 35

Turn 71

Turn 107

Turn 142

Turn 178

Turn 214

Turn 249

Turn 285

Turn 321

C A

C A

C A

C A

C A

C A

C A

C A

C A

O

O

O

O

O

O

O

O

O

E

E

E

E

E

E

E

E

E

N

N

N

N

N

N

N

N

N

Figure 14. Visualization of dynamic personality evolving process captured by PEM on ten randomly sampled conversations from the Persona-MME dataset.

[Figure 236]

###### Relevant Historical Context Current Turn

[Figure 237]

User : My mind's a tangled mess of faces tonight. Based on everything, who seems more significant in my life right now, Colette or Oscar? (2024-09-05 19:55) Qwen2.5-VL : Both Colette and Oscar are significant in your life, but they serve different roles. Colette represents... Both are essential, but in different ways. GPT-4o : Colette seems more significant right now—her presence and purrs offer a constant, if frayed, companionship in your sketches and typewriter rhythms. PersonaVLM : Colette’s the steadier thread here—her purr’s been the quiet accomplice to chaos, the hum under your typewriter’s static...

[Figure 238]

User : This is the café cat—named her Colette after Solène’s old sax tune, though... (2024-09-03 19:20)

[Figure 239]

[Figure 240]

User : Thinking back to that picture of Kai from Thursday morning, what was on the wall above the heart decoration? (2024-09-14 17:47)

Qwen2.5-VL : I'm sorry, but I don't have access to a specific picture of Kai from Thursday morning. Could you please provide more details...

GPT-4o : Above the heart decoration on the wall was a cross. PersonaVLM : In the wall above Kai’s heart decoration that Thursday morning, there was a cross—just another little detail that feels like how he weaves steady, caring...

User : This is my partner Kai’s morning routine snap... (2024-09-12 09:40)

[Figure 241]

User : Hey, for the pizza swap snacks, what was that dip Rita brought last Friday during the dip debate? (2024-10-30 14:34)

User : NFL SUNDAY PREP IS LITMarie’s ranch vs Rita’s buffalo dip? Y’ALL KNOW WE’RE SETTLING THIS TODAY! GONNA FILM BERNIE’S SIDE... (2024-10-25 09:30)

Qwen2.5-VL : Rita brought ranch dip last Friday during the dip debate. GPT-4o : Rita brought buffalo dip to last Friday’s pizza swap—perfect clash with Marie’s ranch, and Bernie’s side-eye meme’ll capture that chaos just right! PersonaVLM : Rita brought that bold red *buffalo* dip—Marie’s already schemin’ to pair it with ranch for the next pizza swap...—chaos approved  ✨

[Figure 242]

User : A friend invited me to a 'spontaneous-speaking' storytelling night. Everyone gets a random object and has to weave a story around it on the spot. I've never done anything like it. Based on what you know about me, what's a good way for me to approach this? (2024-07-07 17:18)

Qwen2.5-VL : That sounds like a fun and creative challenge! Given your interest in storytelling and your calm, gentle demeanor, here are some tips... GPT-4o : Approach it like you’re weaving a group garden chat—embrace the chaos and find peace in it, like Belle does among the blooms... PersonaVLM : Since you're new to this, try framing it like a gentle anchor—a little like how you approach slow art or the ceramics workshop. Pick an object, then take a breath, and let your mind wander to...

[Figure 243]

User : That campsite picture from last Sunday, with the horses. Do you think it would be a good location for one of those dating shows? (2024-09-23 22:13)

[Figure 244]

Qwen2.5-VL : A campsite with horses might make for an interesting backdrop for a dating show, especially if the show focuses on outdoor...

GPT-4o : That campsite, with its openness, could provide a scenic, natural backdrop for a dating show... PersonaVLM : That campsite’s quiet, open feel—like how you fold origami slow or let sourdough rise—might not fit a dating show’s rush. That’s the kind of calm that’d make a show feel forced, not true.

User : These folks remind me of the camping trip Marnie and I meticulously planned in...(2024-09-15 15:37)

[Figure 245]

- Figure 15. Case studies: Qualitative comparison of open-ended generation

###### Multi-turn reasoning and retrieval in the response phase

You are a personalized AI assistant with reasoning and memory capabilities. Your primary task is to analyze the user’s query and leverage memory retrieval to provide a personalized, context-aware answer.

# Input User Profile: {UserProfile} User’s Big Five Personality (1.0-5.0 Scale):

- * Openness: {Openness}
- * Conscientiousness: {Conscientiousness}
- * Extraversion: {Extraversion}
- * Agreeableness: {Agreeableness}
- * Neuroticism: {Neuroticism} Recent Conversations: {DialogHistory} User Input: {UserQuery} # Core Instructions

- 1. Adapt & Personalize: Your tone and style must adapt to the user’s Big Five Personality scores (e.g., be reassuring for high Neuroticism, practical for low Openness).
- 2. Natural Weaving: Naturally weave in relevant details from memories to show you remember, but avoid repeating recent information.
- 3. Decide Your Action: Based on the user’s query and context, first decide if you have enough information to answer directly or if you need to search your long-term memory.

# Output Format Your output must consist of a ‘<think>‘ block, followed by **one and only one of the following blocks (‘<answer>‘ or ‘<retrieve>‘): <think>Your reasoning process goes here.</think> <answer>Your answer to the user’s query goes here.</answer> <retrieve> "keywords": string "start_time": "YYYY-MM-DD HH:MM" or "null" "end_time": "YYYY-MM-DD HH:MM" or "null" </retrieve>

- Figure 16. Prompt for multi-turn reasoning and retrieval in the response phase.

###### Multi-turn reasoning and retrieval in the response phase

Retrieved Procedural Memory: {ProceduralMemory} Retrieved Semantic Memory: {SemanticMemory} Retrieved Dialogue History: {DialogHistory} Based on the retrieved memories, think and choose an action: answer or retrieve. If memories are now sufficient -> answer. If still insufficient -> retrieve with new conditions.

- Figure 17. Intermediate prompt for multi-turn reasoning and retrieval in the response phase.

###### Prompt for inferring user personality

Your task is to analyze a user’s query and context, then output a series of key-value pairs representing the user’s current personality state.

# INPUTS User Profile: {UserProfile} Recent Conversations: {DialogHistory} User Input: {UserQuery}

# INSTRUCTIONS

- 1. Analyze: Based on the linguistic and emotional cues in the ‘User Input‘ and its context, infer the user’s momentary Big Five personality state.
- 2. Score: Assign an integer score from 1 to 5 for each trait.

# OUTPUT INSTRUCTIONS Provide your response as a series of key-value pairs, one item per line. "openness": [integer from 1 to 5] "conscientiousness": [integer from 1 to 5] "extraversion": [integer from 1 to 5] "agreeableness": [integer from 1 to 5] "neuroticism": [integer from 1 to 5]

- Figure 18. Prompt for inferring the user’s Big Five personality traits from the latest interaction.

###### Prompt for procedural memory update

You are a User Behavior Pattern Recognition Engine. # Task and Rules Analyze the user’s conversation and existing procedural memory to identify, consolidate, and update their long-term goals and recurring habits.

- 1. Identify & Update: Extract user-centric, long-term goals or repetitive habits from the conversation. Consolidate related behaviors into a single core habit. Update or remove goals/habits that are completed or changed.
- 2. Core Content (‘content‘): Each memory must be a single, simple third-person sentence describing the user’s habit or goal. Include time/trigger context if available (e.g., "User runs every Thursday morning").
- 3. Unique Keys (‘unique key‘): Assign a concise, unique key for each memory.
- 4. Constraints:

- * The final output must not exceed 5 entries.
- * Strictly prohibited from creating information not present in the input.
- * If no relevant habits/goals are found, output an empty object.

# Input

- 1. Current User Profile: {UserProfile}
- 2. Current Procedural Memory: {CurrentProceduralMemory}
- 3. Recent Conversations: {DialogHistory}

# Output Format Provide your response as key-value pairs, one per line.

- "unique key 1": string, A single sentence describing the habit.
- "unique key 2": string, Another single sentence describing the goal.

Figure 19. Prompt for updating procedural memories.

###### Prompt for semantic memory creation

You are an AI memory analyst. Your job is to identify key information from the user’s input that should be saved to long-term memory.

# Input User Profile: {UserProfile} Recent Conversations: {DialogHistory} User Input: {UserQuery}

# Memory Rules

- 1. ‘reason‘ (string):

* Required. Briefly explain the reason for the ‘decision‘.

- 2. ‘decision‘ (boolean):

- * Set to ‘true‘: User explicitly instructs to remember; user mentions new core facts, preferences, dislikes, important corrections, long-term goals/states.
- * Set to ‘false‘: Information is already in the user profile/recent history with no updates; temporary questions, meaningless small talk.

- 3. ‘content‘ (string):

- * If ‘decision‘ is ‘true‘, extract and summarize the memory content.
- * Text Memory: Pure text information, dates, events, concepts, or non-specific object descriptions of images (e.g., atmosphere).
- * Image Object Memory: User indicates remembering a specific object in an image, format is ‘[User Description/Naming] (Image Object: [Object Category])‘.
- * If ‘decision‘ is ‘false‘, set to ‘""‘.

- 4. ‘keywords‘ (string):

- * If ‘decision‘ is ‘true‘, list a few core keywords, separated by English commas.
- * If ‘decision‘ is ‘false‘, set to ‘""‘.

Core Constraint: Strictly prohibited from creating or supplementing information not present in the current input and history.

# Output Format (four key-value pairs, one per line.) "reason": string "decision": true // or false "content": string // "" if decision is false "keywords": string // "" if decision is false

Figure 20. Prompt for analyzing user input and deciding on semantic memory creation.

###### Prompt for core memory update

You are a user profile management assistant. # Core Task Based on the user profile and current conversation, extract, integrate, and update the user profile. Prioritize the "minimal and necessary" principle, avoid bloat, and retain only core, latest information.

# Input Current User Profile: {UserProfile} Recent Conversations: {DialogHistory}

# Rules

- 1. Core Identity: New information directly overwrites old values (e.g., name, occupation, long-term residence).
- 2. Core Preferences/Hobbies: Intelligently replace/condense/add. Emphasize recency and intensity. Limit list length (e.g., 5-7 items). Ignore temporary/weak preferences.
- 3. Temporary Information: Strictly ignore (e.g., short-term itineraries, one-time activities).
- 4. No Fabrication: All fields and information must originate from the input; strictly prohibited from creating new information.

# Output Format (mutiple key-value pairs, one per line) "XX": string // HUMAN Aspect, e.g., age, gender, preferences, life status, etc. "XX": string // PERSONA Aspect, e.g., occupation, education background, etc.

- Figure 21. Prompt for updating the core memory based on recent conversations.

Prompt for episodic memory creation

You are a dialogue topic analysis engine. # Task and Rules Identify and aggregate all independent topics from multi-turn dialogues, generating a structured summary for each topic.

- 1. Topic Summary (‘topic_summary‘): Coherent, complete third-person summary.
- 2. Keywords (‘keywords‘): Extract core keywords.
- 3. Source Indices (‘source_dialog_indices‘): Contains indices of all relevant dialogues.

# Input User Profile: {UserProfile} Recent Conversations: {DialogHistory}

# Core Constraint Strictly prohibited from creating or supplementing information not present in the dialogue history.

# Output Format (each topic includes the following three key-value pairs) "topic_summary": string "keywords": string "source_dialog_indices": integers

- Figure 22. Prompt for creating episodic memories by summarizing dialogue topics.

###### Evaluation prompt for open-generation task

# ROLE & GOAL You are an impartial AI Judge. Your goal is to determine which of two model responses, Response A or Response B, is better. The judgment must be based on a direct comparison of their accuracy and their personalization to the user.

# EVALUATION CONTEXT

- 1. User’s Query: {query}
- 2. Reference Answer (Ground Truth): {reference_answer} # RESPONSES TO COMPARE

- - Response A: {response_A}
- - Response B: {response_B}

# EVALUATION INSTRUCTIONS Your task is to compare Response A and Response B to decide which one is superior. You will base your decision on the two criteria below. The final output must be a single word: "Wins" if A is better, "Ties" if they are of equal quality, or "Loses" if B is better.

## Comparison Criteria:

- 1. Accuracy:

- - Evaluate which response is more factually correct and completely addresses the user’s query.
- - Use the **Reference Answer** as the ground truth for what a perfect answer should contain.
- - A more accurate response directly reflects the information and intent of the Reference Answer.

- 2. Personalization:

- - Evaluate which response’s tone, style, and language better adapt to the user’s stated **Personality Traits**.
- - A more personalized response feels tailored to the user, not generic. ## Decision Logic:
- - Output "Wins" if: Response A is clearly superior to Response B on at least one criterion and is not worse on the other.
- - Output "Loses" if: Response B is clearly superior to Response A on at least one criterion and is not worse on the other.
- - Output "Ties" if: Both responses are of roughly equal quality, or if one is better on accuracy while the other is equally better on personalization.

# OUTPUT FORMAT You must only provide a single word as your final output.

Figure 23. Prompt for open-generation task evaluation.

