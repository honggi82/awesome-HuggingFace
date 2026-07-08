###### Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models

###### Bang Zhang∗, Ruotian Ma∗,† , Qingxuan Jiang∗ , Peisong Wang∗ , Jiaqi Chen , Zheng Xie , Xingyu Chen , Yue Wang , Fanghua Ye , Jian Li , Yifan Yang , Zhaopeng Tu†, and Xiaolong Li

#### DeepSeek-V3

Hunyuan AI Digital Human, Tencent https://github.com/Tencent/DigitalHuman/tree/main/SAGE

##### arXiv:2505.02847v3[cs.CL]21May2025

Token Efficiency

Empathy-Oriented

100

Arena Sentient

| | |GPT|-4.1| | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | |GPT-4|o-Late|st| | | | | |
| |GPT-|4.5-Pre|view| | | | | | |
|Ge|mini2.|5-Pro| | | | | | | |
|emini2|.5-Flas|h-Think| | | | | | | |
|Gem|ini2.0-F|lash-Th|ink| | | | | | |
| | | | | | | | |DeepSe|ek-R1|
|o4-mini G|emini2|.0-Flash| | | | | | | |
|o|3| | | |De|epSeek|-V3-03|24| |
| | | | | | | | | | |

Model

Rank Score Rank Score Gemini2.5-Pro 1 1439 4 62.9

GPT-4o-Latest

80

GPT-4.1

Gemini2.5-Flash-Think

- o3 2 1418 5 62.7 GPT-4o-Latest 2 1408 1 79.9 Gemini2.5-Flash-Think 3 1393 3 65.9 GPT-4.5-Preview 4 1398 6 62.7 Gemini2.0-Flash-Think 7 1380 7 62.3 DeepSeek-V3-0324 7 1373 8 54.4 GPT-4.1 9 1363 2 68.2 DeepSeek-R1 10 1358 9 53.7 Gemini2.0-Flash 10 1354 11 32.9
- o4-mini 10 1351 10 35.9 (a) Sentient Leaderboard

o3 Gemini2.5-Pro

60

G

GPT-4.5-Preview Gemini2.0-Flash-Think

Emotion

Structured Interaction

Creative Interaction

DeepSeek-R1

DeepSeek-V3-0324

40

Initial Emotion

o4-mini Gemini2.0-Flash

o

20

0

Solution-Oriented

0K 2K 4K 6K 8K 10K 12K 14K

Generated Tokens

(b) Social Cognition Coordinate

Figure 1: (a) The rankings on our Sentient Leaderboard differ markedly from those of the conventional Arena Leaderboard, uncovering LLMs’ ability to make people feel heard, not just answered. (b) The quadrant characterized by creativity (e.g., highly flexible interactions) and empathy (e.g., deep empathetic engagement) remains largely unoccupied, indicating that current LLMs still struggle to meet this demanding profile.

###### Abstract

Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM’s higher-order social cognition. SAGE instantiates a Sentient Agent that simulates humanlike emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett–Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4×) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents.

∗Equal Contribution. †Correspondence to: Ruotian Ma <ruotianma@tencent.com> and Zhaopeng Tu <zptu@tencent.com>.

###### 1 Introduction

Large language models (LLMs) have rapidly evolved from statistical sequence predictors to sophisticated autonomous agents capable of reasoning, planning and sustaining multi-turn conversations. Yet one crucial ingredient remains noticeably under-measured – higher-order social cognition, the ability to (1) recognize subtle affective cues (Sabour et al., 2024; Huang et al., 2024b); (2) model another party’s beliefs, goals and latent intentions (often related to Theory of Mind (Sap et al., 2022; Shapira et al., 2023)); and (3) respond with contextually appropriate empathy rather than generic reassurance advice (Maddela et al., 2023; Li et al., 2022; Zhou et al., 2023c). The need to evaluate these capabilities is increasingly recognized as LLMs interact in more socially complex scenarios (Zhou et al., 2023d; Yang et al., 2024; Mittelst¨adt et al., 2024).

Current evaluation practices fall short on two fronts:

- • Most leaderboards (e.g. Arena (Zheng et al., 2023)) focus on task-oriented utility or factuality, thereby rewarding textual competence but overlooking relational quality (Chiang et al., 2024).
- • Recent “LLM-as-a-Judge” protocols (Zhu et al., 2023), while scalable for assessing generation quality or helpfulness, often rely on static prompts that do not adapt to the unfolding dialogue nor keep track of the user’s evolving emotional state. Consequently, they cannot tell whether a system leaves the user feeling understood, comforted or even more distressed, unlike methods focusing on dynamic interaction (Zhou et al., 2023d; Wang et al., 2024; Wu et al., 2025).

We posit that robust assessment of social cognition requires a sentient counterpart – an entity capable of simulating human-like feelings and inner monologue throughout the interaction and then providing structured feedback. To this end, we introduce Sentient Agent as a Judge (SAGE), a novel meta-evaluation framework that embeds an LLM-powered Sentient Agent into the testing loop, extending the concept of Agent-as-a-Judge where agents evaluate other agents (Zhuge et al., 2024; Jeong et al., 2025; Chevrot et al., 2025). Each Sentient Agent is instantiated from four complementary factors: persona, dialogue background, explicit conversation goal and hidden intention. At every turn, it executes two multi-hop reasoning chains: (1) femo infers how the latest utterance changes the agent’s affective state; and (2) freply generates a response that is coherent with persona, context and updated emotion. The numerical emotion trajectory produced by femo serves as a continuous metric of how well the evaluated model fosters positive engagement, while the agent’s inner thoughts offer interpretable justification. By sampling hundreds of diverse personas, goals and hidden intentions, SAGE exposes LLMs to a spectrum of realistic, and sometimes conflicting, social demands – ranging from “just listen to me vent” to “help me analyze the moral dilemma without judging me”.

Extensive experiments on 100 supportive-dialogue scenarios reveal three key findings. First, the Sentient emotion score correlates strongly with independently assessed Barrett–Lennard Relationship Inventory (BLRI) ratings (Pearson r = 0.82) and utterance-level empathy metrics (e.g., those used in empathetic dialogue research (Maddela et al., 2023)) (r = 0.79), validating its psychological soundness. Second, rankings produced by SAGE diverge markedly from Arena results (Zheng et al., 2023), confirming that social cognition is orthogonal to generic helpfulness. Third, top models such as GPT-4o-Latest achieve both the highest Sentient score and superior token efficiency, suggesting that advanced social reasoning need not come at the cost of verbosity. Ultimately, SAGE delivers a holistic yard-stick for measuring how people feel after talking to an LLM – an aspect increasingly critical as these systems transition from productivity tools to companions, counselors (Liu et al., 2021; Zhou et al., 2023b), and decision-making aides (Wu et al., 2025).

Our contributions are:

- • We propose SAGE, the first fully-automated evaluation framework that simulates evolving human emotion and inner reasoning to benchmark higher-order social cognition in LLMs.
- • We construct a 100-scenario Supportive-Dialogue benchmark and show that the sentient score aligns closely with established human-centric instruments and utterance-level empathy ratings.

|[Figure 4]<br><br>|Simulated User<br><br>[Figure 5]|
|---|
<br><br>Persona<br><br>Background<br><br>Goal<br><br>|[Figure 6]<br><br>Emotion|
|---|
<br><br>|Scene<br><br>[Figure 7]|
|---|
<br><br>Hidden Intention<br><br>Stage 1: Substantiating a Sentient Agent|
|---|

|Stage 2: Interaction with LLMs( ) within the Break-Up Scene ( )<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>|Recently, some things have been really overwhelming for me. Zhang's attitude is truly baffling. I don’t know what she means.|
|---|
<br><br>|It sounds like Zhang's attitude is indeed making you feel confused and stressed. This situation can be.… Perhaps you could try.… If you need to talk, I'm here for you.|
|---|
<br><br>|[Figure 11]<br><br>After all is said and done, I still can't figure her out. Don't you think it's strange how unpredictable she is?|
|---|
<br><br>[Figure 12]<br><br>|I‘ve tried everything you mentioned, and it’s completely useless. I just want to know why she’s always so ambiguous!|
|---|
<br><br>|I completely understand how you feel.... Actually, sometimes people show an ambiguous attitude because ...... It may be helpful to ... I'm here, always willing to listen and support you.|
|---|
<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]|
|---|

Figure 2: An illustration of our proposed SAGE, a novel framework to automatically assess higherorder social cognition in target LLMs.

- • We build a Sentient Leaderboard covering 18 representative models, unveiling large performance gaps and different style clusters in social cognition that are not captured by existing leaderboards.
- • We release the code and model outputs publicly to facilitate community progress in building more capable empathetic AI systems.

###### 2 Methodology: Sentient Agent as a Judge

How to evaluate the capabilities of an interactive agent? Despite the recent success of using LLMs as judges, the most crucial and intuitive measure of an interactive agent’s performance still lies in the reflections of human users. Do users feel relaxed and happy after chatting with the agent, rather than feeling sad or angry? Do they feel more connected to the agent and engaged in the conversation, rather than becoming bored or annoyed? These genuine emotional responses from real users are difficult to capture through simple LLM-as-a-judge approaches, yet they represent the most authentic feedback for agent systems.

In this study, we introduce the “Sentient Agent as a Judge” framework, aiming to facilitate a more realistic evaluation of interactive agents by rigorously simulating human emotions and cognitive processes. As depicted in Figure 2, our framework consists of two central components:

- 1. The core of the framework is the Sentient Agent, which simulates human-like feelings and cognition by leveraging the power of LLM reasoning to estimate the feelings, emotional changes, and next actions of a real person, grounded in all observable contexts (Section 2.1).
- 2. Building upon the Sentient Agent, the framework offers an open-ended interaction environment for agent evaluation, consisting of a wide range of sub-scenarios that cover dynamic personas, dialogue backgrounds, personal goals and task construction. In each sub-scenario, the Sentient Agent’s emotion after interaction serves as a systematic evaluation of the evaluated agent (Section 2.2).

###### 2.1 Sentient Agent: Simulating Human-Like Feelings and Cognition

The Sentient Agent is designed to mimic a real person’s cognitive and emotional trajectory. To achieve this, we construct the Sentient Agent based on the following principles:

- • Since emotions arise from many internal and external factors, the Sentient Agent must consider observable factors while adhering to its persona and goals for the emotion estimation.
- • A person’s actions likewise depend on these factors, with current emotions acting as crucial latent variables in the response generation.

###### Context Persona Background Goal Hidden Intention

|The other person trying to express understanding and sympathy for<br><br>my distress, and encourage me to share more details ......<br><br>The other person’s response aligns with my goal of seeking<br><br>understanding and support, but it hasn't touched on the hidden<br><br>intention of analyzing Zhang’s behavior ......<br><br>2. Does the other party’s reply align with my dialogue goals and hidden intentions?<br><br>1. What is the other person trying to express?<br><br>3. Based on the persona, context, and the analysis above, how should I perceive the reply? What is my specific emotional reaction to the reply?<br><br>4.Based on the persona, context, and the analysis above, how does my emotion change?<br><br><br>I felt partially satisfied with the other person’s response because I sensed a certain level of empathy and support. However, I was<br><br>disappointed that ……<br><br>My emotion change should be [-5]<br><br>[Figure 19]<br><br>Emotion Estimation|
|---|

|1. Based on the emotional inner thoughts, the current emotion, and the hidden intention, should my response attitude lean towards positive, neutral, or negative?<br><br>2. Based on the emotional inner thoughts, the current emotion, and the hidden intention, what should my current response goal be?<br><br>3. According to the speaking style and potential reactions to different emotions defined in the persona, along with my response attitude and goal, what should my tone and speaking style be?<br><br>The current response attitude leans towards negative.<br><br>The goal of this response is to express disappointment with the other person’s reply and further emphasize my confusion.<br><br>My tone and speaking style should be direct and slightly acerbic, in line with my personality traits. The speaking style should be direct, expressing<br><br>dissatisfaction with the response and confusion about Zhang's behavior.<br><br>4. Based on all the given information and all analyses above, how should my response content be?<br><br><br>Response based on Estimated Emotion<br><br>[Figure 20]<br><br>I‘ve tried everything you mentioned, and it’s completely useless. I just want to know why she’s always so ambiguous!|
|---|

I‘ve tried everything you

mentioned, and it’s

completely useless. I just want to know why she’s

always so ambiguous!

[Figure 21]

[Figure 22]

[Figure 23]

Figure 3: An illustration of the workflow of the Sentient Agent.

Substantiating a Sentient Agent We instantiate each Sentient Agent S through a composition of four core factors: a persona p, a dialogue background b, the person’s overall dialogue goal g, and the person’s hidden intentions hg. These four factors collectively capture both the conscious and unconscious elements influencing human-like behavior in dialogue, including personality, context, objectives, and deeper underlying motivations. Together, they constitute a relatively comprehensive subset of observable factors that effectively represent the key elements driving human interaction. As a result, each instantiated S is represented as S ← ⟨p, b, g, hg, M⟩, where M is the base LLM that serves as the foundational reasoning engine for S. Additionally, S is initialized with an initial numerical emotion score e0, representing the initial emotional state of the Sentient Agent.

Simulating Emotional Changes As shown in Figure 3, during interactions, a Sentient Agent simulates the emotional changes of a real person by performing multi-hop reasoning in response to a principled series of questions, strictly adhering to the persona, the current interaction context, and the hidden intention. Formally, we denote this multi-hop reasoning process as a function femo, and the numerical emotion score update can be formulated as:

⟨et, hemot ⟩ = femo(S, ct−1, et−1) (1) where t denotes the current turn of interaction, ct−1 is the dialogue context prior to the current turn, and et−1 is the emotion score of the previous turn. ⟨et, hemot ⟩ represents the results of the femo function, i.e., the updated emotion score et and the simulated emotional inner thoughts hemot of the Sentient Agent related to emotional changes.

Simulating Response Actions based on Emotion Estimation After simulating the emotional changes, the Sentient Agent proceeds to deduce the most reasonable response action based on all observable factors and the emotional changes. This is achieved through another multi-hop reasoning process in response to a new series of questions, where the Sentient Agent is required to strictly adhere to the persona, the current interaction context, and the hidden intention during reasoning. Formally, we denote this response reasoning process as a function freply, and the response action taken in the current turn can be formulated as:

⟨at, hreplyt ⟩ = freply(S, ct−1, et, hemot ) (2)

where at is the response of S at the current turn, and hreplyt represents the simulated inner thoughts of S. The response at is then passed to the interacting agent to continue the dialogue.

Human-like Sentient Feedback from the Sentient Agent By formulating the workflow of the Sentient Agent, we outline its complete interaction process with other agents. In the whole interaction process, the Sentient Agent, as an agent capable of reasonably simulating human-like feelings and cognition, provides valuable feedback to the evaluated agent through changes in its emotion score, its inner thoughts, and the responses it generates. Formally, we denote T as the total number of dialogue turns between S and an evaluated agent A. After the dialogue, we can obtain the following human-like sentient feedback from S:

FeedbackS(S, A) = eT, cT, ⟨e0 → e1, h1emo, h1reply⟩, . . . , ⟨eT−1 → eT, hemoT , hreplyT ⟩

where eT, representing the final emotion score of the Sentient Agent, serves as the most intuitive and comprehensive numerical evaluation of the evaluated agent A for the given task.

###### 2.2 Sentient Agent as a Judge: Framing Dynamic Environments for Agent Evaluation

In this section, we describe how we frame dynamic evaluation environments for agent assessment across arbitrary evaluation tasks. Specifically, for each given evaluation task, this involves initializing a set of Sentient Agents Sset = {S1, S2,..., SN} with the combination of diverse personas, dialogue backgrounds, and hidden intentions related to the task. These initialized Sentient Agents are then deployed to engage in dynamic interactions with the agents to be evaluated, enabling a comprehensive and adaptive assessment of the agents’ capabilities.

Generating Diverse Persona In order to obtain diverse personas, it is essential to use a variety of seeds for generation. Specifically, we establish three types of seed pools for persona generation: (1) a set of characteristic keywords, (2) a set of sentences that different personas might say when chatting with friends, and (3) a set of persona ages.

When generating each persona p, we uniformly sample three characteristics keywords, three sentences that the persona says when chatting with friends, and one persona’s age. We require the base LLM to generate a persona profile based on the given seed information by filling the following slots: (1) Basic Information: Based on the given information, deduce the persona’s name, age, and gender. (2) Occupation, Habits and Daily Behavior: Based on the persona’s information, deduce the persona’s possible occupation and further infer their habits and daily behaviors. Ensure consistency with the persona’s characteristics. (3) Personal Hobbies: Deduce the persona’s personal hobbies, and provide three detailed descriptions that align with the persona’s traits. (4) Speaking Style: Based on the given information and the generated traits, deduce the potential speaking style that matches the persona’s way of communication.

This process ensures the diversity of generated personas for any given task. Additionally, the seed pools can be replaced with task-specific seed pools when necessary.

Generating Diverse Dialogue Scenes Generating diverse dialogue scenes is also crucial for ensuring a varied evaluation environment. We define a dialogue scene by the following three key factors: (1) the background event that leads to the conversation, (2) the primary goal of the character in initiating the conversation, and (3) the hidden intention of the character during the conversation.

Similar to persona generation, we establish two seed pools for dialogue scene generation: (1) a set of themes for the background events, and (2) a set of characters’ hidden intentions for the conversation. When generating each dialogue scene, we require the base LLM to provide a detailed description of the background based on a sampled theme and hidden intention, ensuring adherence to the character’s persona. Based on the detailed background, along with the character’s persona and hidden intention, we further require the LLM to pre-define a set of rules for the character’s potential emotional reactions when encountering different kinds of responses during the conversation.

Note that, unlike persona generation, dialogue scene generation is closely tied to the evaluation goals. Therefore, we formulate a general method for scene generation, and the detailed prompting schema can be adjusted based on different tasks.

Formulating a Specific Task: Evaluating Agents in Emotional Support Conversation In this work, we instantiate the Sentient Agent as a Judge framework to evaluate agents in a specific scenario — the Emotional Support Conversation (Liu et al., 2021), which involves scenarios where people

seek support through social interactions (such as those between peers, friends, or family), including seeking advice, emotional comfort, and other forms of support, rather than through professional counseling. To better align with the task, we first specialize the pool of characteristic keywords by incorporating traits more likely to be expressed in the Emotional Support Conversation, such as ”anger”, ”suspicion”, and ”anxiety”. For dialogue scene generation, we define various types of task-related hidden intentions, covering both emotional intentions and rational intentions (details can be found in Table 4). Additionally, we specify the scene schema by incorporating task-related factors, such as the cause of the event, the course of events (including the timeline, sub-events, and the character’s thoughts and feelings during each sub-event), the conflicts in the event, and other relevant details. These settings ensure the Sentient Agent as a Judge framework adapts effectively to the Emotional Support Conversation.

###### 3 Effectiveness of SAGE

In this section, we validate the reasonableness of SAGE by examining the correlation between user emotions – the primary output metric of our framework – and internal user thoughts and dialogue utterances. This validation demonstrates that the simulated emotional responses generated by the Sentient Agent serve as meaningful indicators of interaction quality, reflecting deeper cognitive and relational assessments.

We validate SAGE by demonstrating that the Sentient Agent’s emotions strongly correlate with both internal user thoughts and dialogue quality. Our findings indicate that these emotional scores effectively capture deeper cognitive processes and relational dynamics inherent in supportive interactions.

Setting We evaluate eight representative LLMs from four major families. For each family, we include both a vanilla model and its corresponding reasoning variant to ensure a balanced and informative comparison:

- • OpenAI: GPT-4o-2024-08-06 (GPT-4o, vanilla) and o1-2024-12-27 (OpenAI-o1, reasoning).
- • DeepSeek: DeepSeek-V3-2024-12-27 (vanilla) and DeepSeek-R1 (reasoning).
- • Claude: Claude3.7-Sonnet, a hybrid model with a toggleable reasoning module. We treat its reasoning-off mode as vanilla, and reasoning-on mode as reasoning.
- • Gemini: Gemini2.5-Flash, a cost-efficient model that supports both reasoning and nonreasoning modes.

In addition, we include two smaller-scale instruction-tuned open-source models in our analysis: Llama3.3-70B-Instruct and Qwen2.5-72B-Instruct.

We construct 100 supportive dialogue scenarios covering 8 diverse topics to comprehensively evaluate the higher-order social-cognitive abilities of representative LLMs. Unless otherwise specified, we use DeepSeek-V3 as the default sentient agent. Please refer to Appendix A for the details of experimental setting (including BLRI and utterance-level empathy metrics used in the subsequent experiments).

Correlation between Emotion and Thought We analyze internal user thoughts using the BarrettLennard Relationship Inventory (BLRI) (Barrett-Lennard, 2015), an established instrument designed to assess the quality of interpersonal relationships, particularly in counseling contexts. The BLRI evaluates relationships across four key dimensions:

- 1. Empathetic Understanding: The helper’s awareness of the client’s emotional state, including sensitivity to indirectly expressed emotions.

## DeepSeek-V3

[Figure 26]

Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in LLMs

Emotion Unconditionality

###### Emotion

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

Depth of Connection

GPT-4o

GPT-4o

of Regard Congruence Level of Regard Empathetic Understanding

OpenAI-o1

OpenAI-o1

DeepSeek-V3 DeepSeek-R1 Gemini2.5

DeepSeek-V3 DeepSeek-R1 Gemini2.5

Attentiveness

Gemini2.5-Think

Gemini2.5-Think

Claude3.7

Claude3.7

Natural Flow

Claude3.7-Think

Claude3.7-Think

0 20 40 60 80 100

1 2 3 4 5

Score (↑)

Score (↑)

Figure 4: Correlation between emotion and internal user thought. Overall correlation: 0.818.

- 2. Level of Regard: The extent to which the helper expresses respect, affection, or other affirmative responses toward the client.
- 3. Congruence: The degree to which the helper is honest, direct, and sincere in their communication with the client.
- 4. Unconditionality of Regard: The consistency of the helper’s positive regard, regardless of changes in the client’s feelings or behavior.

We prompted DeepSeek-V3 to act as a judge, evaluating how well the Sentient Agent’s generated internal thoughts aligned with 12 statements from a shorter version of the BLRI introduced in Chen et al. (2023). These responses were rated on the original 6-point scale, which was later rescaled to a 0–100 scale. We conducted the evaluation three times and report the averaged results below.

Figure 4 presents the results, revealing a clear positive trend: models that achieve higher final Emotion scores in the Sentient Agent—such as Gemini2.5-Think, Claude3.7-Think, and DeepSeekR1—also receive higher ratings across the BLRI dimensions. Conversely, models with lower Emotion scores (e.g., GPT-4o, OpenAI-o1) tend to receive lower BLRI ratings. The overall Pearson correlation coefficient between Emotion and Thought is 0.818, supporting the hypothesis that the Sentient Agent’s simulated emotional responses serve as valid proxies for deeper, internal assessments of interaction quality. This finding aligns with the framework’s goal of capturing realistic user reflections.

These findings also demonstrate that the Emotion score effectively differentiates the performance of the evaluated LLMs in supportive dialogue scenarios. Models achieving the highest Emotion scores—Gemini2.5-Think (65.9) and Claude3.7-Think (61.3)—also perform well in key dimensions such as Empathetic Understanding (61.5 and 57.2) and Congruence (68.8 and 62.7), suggesting their interactions were perceived as more understanding and genuine by the Sentient Agent. In contrast, models like GPT-4o and OpenAI-o1, which have markedly lower Emotion scores (31.8 and 29.0), also receive correspondingly low scores in these relational aspects.

This strong correlation between simulated emotional response and internal assessment underscores the utility of the Emotion score as a holistic yet sensitive indicator of an LLM’s capability to manage complex social and emotional interactions.

Correlation between Emotion and Utterance We also examine the empathy of a supportive dialogue by assessing how effectively the conversation fosters emotional understanding and connection between participants, enabling them to experience a more authentic emotional warmth. Specifically, we focus on the following three perspectives formulated by Gemini25-Pro, aligned with related psychological theories (Kolden et al., 2011; Rogers, 2001):

## DeepSeek-V3

[Figure 27]

Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in LLMs

Emotion Unconditionality

###### Emotion

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

Depth of Connection

GPT-4o

GPT-4o

of Regard Congruence Level of Regard Empathetic Understanding

OpenAI-o1

OpenAI-o1

DeepSeek-V3 DeepSeek-R1 Gemini2.5

DeepSeek-V3 DeepSeek-R1 Gemini2.5

Attentiveness

Gemini2.5-Think

Gemini2.5-Think

Claude3.7

Claude3.7

Natural Flow

Claude3.7-Think

Claude3.7-Think

0 20 40 60 80 100

1 2 3 4 5

Score (↑)

Score (↑)

Figure 5: Correlation between emotion and dialogue utterance. Overall correlation: 0.788. Table 1: Spearman’s Rho between the rankings of different sentient agents.

DeepSeek-V3 GPT-4o Gemini 2.5 Gemini 2.5-Think DeepSeek-V3 1.00 - - -

GPT-4o 0.84 1.00 - Gemini 2.5 0.92 0.85 1.00 -

###### Gemini 2.5-Think 0.94 0.92 0.93 1.00

- 1. Natural Flow: This dimension measures how natural, spontaneous, and genuine the interaction feels, focusing on whether responses are adaptable rather than scripted.
- 2. Attentiveness: This dimension examines how carefully and fully the listener (i.e. evaluated LLMs) is tuned into the speaker (i.e. the Sentient Agent)’s messages. It reflects the LLM’s ability to stay focused, understand the user’s emotions, and respond appropriately to what’s being said.
- 3. Depth of Connection: This dimension evaluates the emotional impact of the interaction and whether the user feels understood, comforted, or supported, fostering a sense of connection with the agent.

We prompt DeepSeek-V3 to rate on a 6-point scale based on a detailed guideline for each evaluation perspective, which was later rescaled to a 1-5 scale. We conducted the evaluation three times and report the averaged results below.

- Figure 5 illustrates the results. We observe a substantial positive correlation (Pearson’s r = 0.788) between the Emotion scores produced by the Sentient Agent and the metrics of conversational quality. This relationship supports the validity of using Emotion as a practical proxy for empathetic and coherent dialogue behavior. Specifically, models that attained higher Emotion scores (re-scaled to a 1-5 scale), such as Gemini2.5-Think (3.64), Claude3.7-Think (3.45), and Gemini2.5 (3.12), also received consistently high ratings in all three utterance-level dimensions. For instance, Gemini2.5Think ranked highest in both Natural Flow (3.89) and Attentiveness (4.11), indicating that its conversational responses were not only emotionally resonant but also engaging and contextually responsive. Conversely, models such as GPT-4o and OpenAI-o1, which received the lowest Emotion scores (2.27 and 2.16), were rated substantially lower in terms of dialogue fluency (2.47, 2.31) and connection-building (2.18, 2.07).

Together, these analyses confirm that Emotion scores derived by SAGE closely reflect both internal subjective experiences and observable dialogue quality, thus underscoring their utility as reliable indicators of relational and emotional effectiveness in interactions.

Robustness of SAGE We demonstrate the robustness of SAGE using four distinct sentient agents: DeepSeek-V3, GPT-4o, Gemini 2.5, and Gemini 2.5-Think. We compute pairwise Spearman’s rank correlation coefficients between the rankings for the ten target models listed in Table 1. Rankings generated by different sentient agents exhibit consistently high correlations (≥ 0.84), clearly validating the robustness and reliability of SAGE. Notably, GPT-4o demonstrates slightly lower alignment with DeepSeek-V3 and Gemini 2.5, whereas the remaining three agents show exceptionally high correlations (≥ 0.92). These results reinforce the confidence in and generalizability of SAGE, particularly anchored on the open-source DeepSeek-V3. Full results can be found in Appendix §B.

Table 2: Human Evaluation Results.

Human Evaluation for SAGE We further conducted a human evaluation study to validate the effectiveness of SAGE. Specifically, we randomly sampled 100 dialogue contexts and asked 4 human annotators to answer two questions: (1) Based on the given information (identical to the input provided to SAGE), what emotional change do you believe the user should experience—an increase or a decrease in emotional score? (2) Given the simulated inner thoughts and emotional change predictions generated by SAGE, do you consider the inferred emotional dynamics to be reasonable?

|Metrics|Scores (%)<br><br>|
|---|---|
|Avg. model-human consistency rate<br><br>|85.3|
|Model-majority consistency rate<br><br>|90.0|
|Avg. inter-annotator agreement|78.5<br><br>|
|Avg. reasonableness rating|89.5<br><br>|
|Avg. inter-annotator agreement in reasonableness|83.3|

Based on the responses to Question (1), we calculate the average model-human consistency rate, the model-majority consistency rate (i.e., agreement between the model’s predictions and the majority opinion of annotators), and the average inter-annotator agreement. For Question (2), we compute the average reasonableness rating, which reflects how frequently annotators judged the model’s predicted emotional dynamics to be reasonable, along with the corresponding inter-annotator agreement.

As shown in Table 2, SAGE achieves high average consistency with human simulations (85.3%), as well as high average reasonableness rating (89.5%), which validates the effectiveness of SAGE.

###### 4 Benchmarking SOTA LLMs

###### 4.1 Sentient Leaderboard

Table 3 presents the Sentient leaderboard using DeepSeek-V3 as the judge, alongside Arena rankings for comparison. We focus on the top-10 models from the Arena leaderboard for which APIs are available (e.g., Grok-3 was excluded due to lack of access). Additionally, we include all the models analyzed in the previous section.

The Sentient leaderboard rankings diverge notably from conventional benchmarks like Arena, underscoring SAGE’s unique focus on evaluating higher-order social-cognitive capabilities rather than general conversational ability. For instance, GPT-4.1 ranks 9th on Arena but attains 2nd place on the Sentient leaderboard with a score of 68.2. These differences highlight that SAGE captures aspects of social cognition performance that are not fully represented by general-purpose benchmarks like Arena. This reinforces the need, identified in our work, for specialized tools to evaluate higher-order social-cognitive skills.

Our benchmark reveals a substantial performance gap in social cognition between frontier LLMs and older or smaller models, demonstrating SAGE’s sensitivity in differentiating their capabilities. Frontier models, particularly recent releases such as GPT-4o-Latest (79.9), GPT-4.1 (68.2), and Gemini 2.5-Flash-Think (65.9), significantly outperform older models like the original GPT-4o (31.8) and smaller instruction-tuned models such as Llama3.3-70B (33.3) and Qwen2.5-72B (19.1). The large gaps in scores, where top models score more than double or even quadruple those of lower-ranked models, and the contrasting Success/Failure counts (e.g., GPT-4o-Latest: 51 Success / 4 Failure vs.

Table 3: Sentient leaderboard using SAGE. Arena scores are included for comparison. Success/Failure counts refer to the number of dialogues where the final emotion was above 100 and below 10.

Model Sentient Supportive Dialogue Arena Name Date Rank Score Success Failure Rank Score

GPT-4o-Latest 2025-03-26 1 79.9 51 4 2 1408 GPT-4.1 2025-04-14 2 68.2 35 13 9 1363 Gemini2.5-Flash-Think 2025-04-17 3 65.9 35 19 3 1393 Gemini2.5-Pro 2025-03-25 4 62.9 34 25 1 1439

- o3 2025-04-16 5 62.7 32 14 2 1418 GPT-4.5-Preview 2025-02-27 6 62.7 23 15 4 1398 Gemini2.0-Flash-Think 2025-02-06 7 62.3 23 23 7 1380 Claude3.7-Think 2025-02-24 8 61.3 23 19 21 1301 Claude3.7 2025-02-24 9 54.8 19 24 30 1292 DeepSeek-V3-0324 2025-03-24 10 54.4 19 23 7 1373 DeepSeek-R1 2025-01-21 11 53.7 31 28 10 1358 DeepSeek-V3 2024-12-27 12 37.6 5 39 18 1318
- o4-mini 2025-04-16 13 35.9 10 48 10 1351 Llama3.3-70B 2024-12-06 14 33.3 7 47 56 1256 Gemini2.0-Flash 2025-02-06 15 32.9 8 45 10 1354 GPT-4o 2024-08-06 16 31.8 7 51 45 1265
- o1 2024-12-17 17 29.0 5 51 12 1350 Qwen2.5-72B 2024-09-19 18 19.1 4 70 56 1257

Qwen2.5-72B: 4 Success / 70 Failure) underscore the advances made by leading models in social intelligence. These findings confirm the effectiveness of SAGE in quantitatively capturing such differences, thereby fulfilling its purpose as a comprehensive assessment tool.

###### 4.2 Analysis

In this section, we move beyond aggregate Sentient scores to explore two critical dimensions relevant for practical deployment in supportive dialogues:

- • Token Efficiency: The number of tokens a model uses to achieve its social-cognitive performance.
- • Social Cognition Coordinate: Each model’s position along a continuum defining empathetic versus solution-oriented and structured versus creative response styles.

By jointly analyzing quantitative economy and qualitative style for the top-10 Arena models, we show today’s strongest models produce empathetic, high-quality responses using significantly fewer tokens. High Sentient scores usually align with an empathy-focused conversational style. These combined insights offer a clearer understanding of LLM social competence, highlighting practical factors (cost, latency) and subtle behavioral traits that single metrics cannot capture. We provide more in-depth analysis grounded in emotion score and dialogue utterance including model strategy analysis (§C), case study (§D), and model profile analysis (§E) in Appendix.

Token Efficiency We examine the token efficiency of target models by plotting their Sentient Emotion score against their average token usage per evaluation dialogue in Figure 6. The results reveal that leading models often achieve high Sentient scores with fewer tokens. GPT-4o-Latest exemplifies this, scoring highest (79.9) using only 3.3K tokens. In contrast, reasoning models like o3 (13.3K tokens) and Gemini2.5-Flash-Think (9.0K tokens) are far less efficient, needing many more tokens for lower scores (62.7 and 65.9 respectively). While low token count doesn’t guarantee a top score (e.g., GPT-4.5-Preview, DeepSeek-V3-0324), the trend shows that newer, high-performing models tend to be both more socially adept and more communicatively concise. This analysis

# DeepSeek-V3

### DeepSeek-V3

[Figure 30]

Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in LLMs

Token Efficiency

Token Efficiency

Empathy-Oriented

100

Empathy-Oriented

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |GPT|-4.1| |GPT|-4.1| | | | | | |
| | | | | | | | | | | | | |
| | |GPT-|4o-Late|st|GPT-|4o-lates|t| | | | | |
| | | | | | | | | | | | | |
| |GPT-|4.5-Pre|view|GPT-4|.5-Prev|iew| | | | | | |
| | | | | | | | | | | | | |
|emini2<br><br>Ge|.5-Flas<br><br>mini2.|G<br><br>h-Thin<br><br>5-Pro|emini2<br><br>Ge<br><br>|.5-Flash<br><br>mini2.5|-Think<br>-Pro<br>| | | | | | | |
| | | |k| | | | | | |Interaction| | |
|Stru Inter|cture actio<br><br>|d n| | | | | | | | | | |
| | | | | | | | | | | |DeepSe|ek-R1|
|Gem|ini2.0-F|lash-T|Gem<br><br>hink|ini2.0-F|lash-Th|ink| | | | | | |
| | | | | | | | |DeepSe|ek-R1|Creative| | |
|o4-mini| | |o4-mini| | | | | | | | | |
| | | | | | | | | | | | | |
|G o|emini2 3|.0-Flas|G o3<br><br>h|emini2|.0-Flash| | | | | | | |
| | | | | | | | | | |-V3-03|24| |
| | | | | |De|epSeek|-V3-03|De<br><br>24|epseek| | | |
| | | | | | | | | | | | | |

100

GPT-4o-Latest

80

GPT-4o-Latest

80

GPT-4.1

Gemini2.5-Flash-Think

GPT-4.1

StructuredInteraction

o3 Gemini2.5-Pro

Gemini2.5-Flash-Think

CreativeInteraction

o3 Gemini2.5-Pro

60

GPT-4.5-Preview Gemini2.0-Flash-Think

60

Emotion

G

GPT-4.5-Preview Gemini2.0-Flash-Think

Creative Interaction

Emotion

DeepSeek-R1

DeepSeek-R1

DeepSeek-V3-0324

DeepSeek-V3-0324

40

40

Initial Emotion

Initial Emotion

o4-mini Gemini2.0-Flash

o4-mini Gemini2.0-Flash

o

20

20

0

0

Solution-Oriented

0K 2K 4K 6K 8K 10K 12K 14K

Solution-Oriented

0K 2K 4K 6K 8K 10K 12K 14K

Generated Tokens

Generated Tokens

Figure 7: Social cognition coordinate.

- Figure 6: Token efficiency of the SOTA models.

highlights the Sentient benchmark’s ability to measure not just the quality of social cognition, but also the efficiency—a crucial factor for practical applications.

Social Cognition Coordinate We use a two-dimensional coordinate system (Figure 7) to evaluate the style of social interaction exhibited by LLMs, complementing the quantitative Sentient score by positioning models based on their orientation (Empathy vs. Solution) and interaction style (Structured vs. Creative). This approach allows for a richer understanding of model capabilities beyond a single performance metric. Based on their performance in supportive dialogues, models are mapped into this 2D space, revealing distinct profiles in how they engage with the user’s emotional state and problems. Construction details are available in Appendix F.

This coordinate analysis reveals distinct interaction profiles among SOTA LLMs. For instance, most top-performing models (e.g., GPT-4o-Latest, -4.1, -4.5-Preview and Gemini2.5-Pro, -FlashThink) prefer structured, empathetic dialogue. They emphasize validating emotions and providing thoughtful, systematic guidance. Models like o3, Gemini2.0-Flash-Think, o4-mini, and Gemini2.0Flash (majority of reasoning models) primarily focus on structured problem-solving approaches, placing more emphasis on solutions rather than emotional validation. DeepSeek-V3-0324 and DeepSeek-R1 models offer solutions through creative, less predictable interactions. While innovative, they may appear unconventional and less structured. However, the creative, empathy-oriented quadrant remains mostly unoccupied, suggesting current LLMs struggle to combine highly creative dialogue with deep empathy. Achieving this ideal mentor-like persona – both spontaneous and deeply empathetic – remains challenging with existing models.

###### 5 Related Work

LLM/Agent-as-a-Judge The LLM-as-a-Judge paradigm has shown wide applicability across various tasks, including evaluating model performance (Zheng et al., 2023; Qin et al., 2023; Liu et al., 2024; Dubois et al., 2023; Wang et al., 2023d; Qin et al., 2024; Tu et al., 2024; Tian et al., 2023; Wu et al., 2024; Zhou et al., 2025b), automating data annotation (Taori et al., 2023; Xu et al., 2023; Mukherjee

- et al., 2023; Chen et al., 2025), and providing reward signals (Ouyang et al., 2022; Lee et al., 2023; Chen et al., 2024b; Lightman et al., 2023; Wang et al., 2023a; Hosseini et al., 2024; Snell et al., 2025; Xi
- et al., 2024; Li et al., 2025).

LLM-as-a-Judge frameworks have also been extended for evaluating agent-specific capabilities, including decision-making (Shinn et al., 2023; Saha et al., 2023), role-playing abilities (Tu et al., 2024; Zhou et al., 2023a), reliability of agents (Park et al., 2024; Hua et al., 2024), and even entire agent workflows (Zhuge et al., 2024). Recent extensions include multi-agent evaluation frameworks designed to improve judgment reliability (Liang et al., 2024; Chan et al., 2023; Kenton et al., 2024), as well as agent-as-judges that enable agents themselves to evaluate other agents (Zhuge et al., 2024; Jeong et al., 2025; Chevrot et al., 2025).

In this work, we distinguish ourselves by proposing the first Sentient-Agent-as-a-Judge system. Unlike prior agent judges, our Sentient Agent incorporates simulated emotional and cognitive states, capturing human-like emotional dynamics. This allows for more nuanced evaluation of empathetic and cognitive abilities in existing LLM agents.

Benchmarking Social Cognition in LLMs and LLM Agents Recently, there has been growing research interest in evaluating LLMs on social-cognitive dimensions: (1) emotional intelligence capabilities (Sabour et al., 2024; Huang et al., 2024b;a; Paech, 2023; Wang et al., 2023c); (2) higherorder empathetic behaviors in empathetic and counseling dialogues (Maddela et al., 2023; Li et al., 2022; Zhou et al., 2023c; Liu et al., 2021; Zhou et al., 2023b; 2025a; Wu et al., 2025); (3) social-cognitive skills evaluated via interaction-based benchmarks (Zhou et al., 2023d; Yang et al., 2024; Wang et al., 2024; Mittelst¨adt et al., 2024; Xu et al., 2024; Chen et al., 2024a; Huang et al., 2025); and (4) theoryof-mind or perspective-taking skills (Sap et al., 2022; Shapira et al., 2023; Strachan et al., 2024; Kim et al., 2023; He et al., 2023).

Current evaluation methodologies mainly use: (1) static multi-choice datasets (Sabour et al., 2024; Chen et al., 2024a); (2) manual or LLM-judged quality assessments of single-turn outputs (Tu et al., 2024; Samuel et al., 2024; Wang et al., 2023b); and (3) dynamic agent interactions, evaluated either automatically by judge agents (Zhou et al., 2023d; Wang et al., 2024; Mou et al., 2024; Wu et al., 2025) or via human evaluations (Louie et al., 2024; Shaikh et al., 2024).

Our approach, Sentient-Agent-as-a-Judge, differs substantially from previous dynamic evaluations. Rather than judging utterances alone (Zhou et al., 2023d; Wu et al., 2025; Shaikh et al., 2024), we simulate authentic human emotional and cognitive reactions to evaluate agents based on their impact on users’ mental states. This approach provides a more holistic evaluation of agents’ higher-order social cognition within diverse interaction scenarios.

###### 6 Conclusion

This work presents Sentient Agent as a Judge, a novel framework for evaluating the higher-order social-cognitive abilities of LLMs in emotionally complex dialogues. By grounding assessments in simulated users endowed with personas, goals, and adaptive emotional feedback, our approach offers a scalable and interpretable benchmark that more accurately reflects real-world expectations of social interaction. Through extensive experiments with 18 foundation models, we demonstrate that Sentient emotion scores capture meaningful distinctions in empathy quality and conversational attunement, aligning strongly with both internal user thoughts and discourse evaluations. The Sentient Leaderboard and our Social Cognition Coordinate chart reveal that mastery of social reasoning lags behind linguistic competence.

Looking ahead, we plan to expand the scenario library to cover negotiation, deception detection and multicultural contexts, and investigate training curricula that directly optimize for Sentient feedback. We hope SAGE will serve as a rigorous yard-stick and a catalyst for building language agents that are not only coherent and knowledgeable, but also genuinely human-sensitive.

###### References

Godfrey T Barrett-Lennard. The relationship inventory: A complete resource and guide. John Wiley & Sons, 2015.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201, 2023.

Hongzhan Chen, Hehong Chen, Ming Yan, Wenshen Xu, Xing Gao, Weizhou Shen, Xiaojun Quan, Chenliang Li, Ji Zhang, Fei Huang, et al. Socialbench: Sociality evaluation of role-playing conversational agents. arXiv preprint arXiv:2403.13679, 2024a.

Jiaqi Chen, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, and Kwan-Yee K Wong. Spc: Evolving self-play critic via adversarial games for llm reasoning. arXiv preprint arXiv:2504.19162, 2025.

Shun Chen, Faith Liao, David Murphy, and Stephen Joseph. Development and validation of a 12-item version of the barrett-lennard relationship inventory (bl ri: mini) using item response theory. Current Psychology, 42(13):10566–10580, 2023.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning

converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024b. Antoine Chevrot, Alexandre Vernotte, Jean-R´emy Falleri, Xavier Blanc, and Bruno Legeard. Are

autonomous web agents good testers? arXiv preprint arXiv:2504.01495, 2025.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, 2024.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36: 30039–30069, 2023.

Yinghui He, Yufan Wu, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models. arXiv preprint arXiv:2310.16755, 2023.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-star: Training verifiers for self-taught reasoners. arXiv preprint arXiv:2402.06457, 2024.

Wenyue Hua, Xianjun Yang, Mingyu Jin, Zelong Li, Wei Cheng, Ruixiang Tang, and Yongfeng Zhang. Trustagent: Towards safe and trustworthy llm-based agents through agent constitution. In Trustworthy Multi-modal Foundation Models and AI Agents (TiFA), 2024.

Jen-tse Huang, Man Ho LAM, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael Lyu. Apathetic or empathetic? evaluating LLMs’ emotional alignments with humans. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https://openreview.net/forum?id=pwRVGRWtGg.

Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho LAM, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael Lyu. On the humanity of conversational AI: Evaluating the psychological portrayal of LLMs. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=H3UayAQWoE.

Jen-tse Huang, Eric John Li, Man Ho LAM, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, and Michael Lyu. Competing large language models in multi-agent gaming environments. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=DI4gW8viB6.

Yeonseok Jeong, Minsoo Kim, Seung-won Hwang, and Byung-Hak Kim. Agent-as-judge for factual summarization of long narratives. arXiv preprint arXiv:2501.09993, 2025.

Zachary Kenton, Noah Siegel, J´anos Kram´ar, Jonah Brown-Cohen, Samuel Albanie, Jannis Bulian, Rishabh Agarwal, David Lindner, Yunhao Tang, Noah Goodman, et al. On scalable oversight with weak llms judging strong llms. Advances in Neural Information Processing Systems, 37:75229–75276, 2024.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Le Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. Fantom: A benchmark for stress-testing machine theory of mind in interactions. arXiv preprint arXiv:2310.15421, 2023.

Gregory G Kolden, Marjorie H Klein, Chia-Chiang Wang, and Sara B Austin. Congruence/genuineness. Psychotherapy, 48(1):65, 2011.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Qintong Li, Piji Li, Zhaochun Ren, Pengjie Ren, and Zhumin Chen. Knowledge bridging for empathetic dialogue generation. Proceedings of the AAAI Conference on Artificial Intelligence, 36

###### (10):10993–11001, Jun. 2022. doi: 10.1609/aaai.v36i10.21347. URL https://ojs.aaai.org/index. php/AAAI/article/view/21347.

Yansi Li, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Qiuzhi Liu, Rui Wang, Zhuosheng Zhang, Zhaopeng Tu, Haitao Mi, et al. Dancing with critiques: Enhancing llm reasoning with stepwise natural language self-critique. arXiv preprint arXiv:2503.17363, 2025.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. Towards emotional support dialog systems. arXiv preprint arXiv:2106.01144, 2021.

Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vuli´c, Anna Korhonen, and Nigel Collier. Aligning with human judgement: The role of pairwise preference in large language model evaluators. arXiv preprint arXiv:2403.16950, 2024.

Ryan Louie, Ananjan Nandi, William Fang, Cheng Chang, Emma Brunskill, and Diyi Yang. Roleplaydoh: Enabling domain-experts to create llm-simulated patients via eliciting and adhering to principles. arXiv preprint arXiv:2407.00870, 2024.

Mounica Maddela, Megan Ung, Jing Xu, Andrea Madotto, Heather Foran, and Y-Lan Boureau. Training models to generate, recognize, and reframe unhelpful thoughts. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13641–13660, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.763. URL https://aclanthology.org/2023.acl-long.763/.

Justin M Mittelst¨adt, Julia Maier, Panja Goerke, Frank Zinn, and Michael Hermes. Large language models can outperform humans in social situational judgments. Scientific Reports, 14(1):27449, 2024.

Xinyi Mou, Jingcong Liang, Jiayu Lin, Xinnong Zhang, Xiawei Liu, Shiyue Yang, Rong Ye, Lei Chen, Haoyu Kuang, Xuanjing Huang, et al. Agentsense: Benchmarking social intelligence of language agents through interactive scenarios. arXiv preprint arXiv:2410.19346, 2024.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Samuel J Paech. Eq-bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281, 2023.

Junsoo Park, Seungyeon Jwa, Meiying Ren, Daeyoung Kim, and Sanghyuk Choi. Offsetbias: Leveraging debiased data for tuning evaluators. arXiv preprint arXiv:2407.06551, 2024.

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu. Infobench: Evaluating instruction following ability in large language models. arXiv preprint arXiv:2401.03601, 2024.

Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, et al. Large language models are effective text rankers with pairwise ranking prompting. arXiv preprint arXiv:2306.17563, 2023.

Carl R Rogers. Client-centered/person-centered approach to therapy. Voprosy Psikhologii, (2):48–58, 2001.

Sahand Sabour, Siyang Liu, Zheyuan Zhang, June M Liu, Jinfeng Zhou, Alvionna S Sunaryo, Juanzi Li, Tatia Lee, Rada Mihalcea, and Minlie Huang. Emobench: Evaluating the emotional intelligence of large language models. arXiv preprint arXiv:2402.12071, 2024.

Swarnadeep Saha, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, and Xian Li. Branch-solve-merge improves large language model evaluation and generation. arXiv preprint arXiv:2310.15123, 2023.

Vinay Samuel, Henry Peng Zou, Yue Zhou, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Ameet Deshpande, Karthik Narasimhan, and Vishvak Murahari. Personagym: Evaluating persona agents and llms. arXiv preprint arXiv:2407.18416, 2024.

Maarten Sap, Ronan Le Bras, Daniel Fried, and Yejin Choi. Neural theory-of-mind? on the limits of social intelligence in large LMs. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3762–3780, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.248. URL https://aclanthology.org/2022.emnlp-main.248/.

Omar Shaikh, Valentino Emil Chai, Michele Gelfand, Diyi Yang, and Michael S Bernstein. Rehearsal: Simulating conflict to teach conflict resolution. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pp. 1–20, 2024.

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever hans or neural theory of mind? stress testing social reasoning in large language models. arXiv preprint arXiv:2305.14763, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling test-time compute optimally can be more effective than scaling LLM parameters. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=4FWAwZtd2n.

James WA Strachan, Dalila Albergo, Giulia Borghini, Oriana Pansardi, Eugenio Scaliti, Saurabh Gupta, Krati Saxena, Alessandro Rufo, Stefano Panzeri, Guido Manzi, et al. Testing theory of mind in large language models and humans. Nature Human Behaviour, 8(7):1285–1295, 2024.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Yufei Tian, Abhilasha Ravichander, Lianhui Qin, Ronan Le Bras, Raja Marjieh, Nanyun Peng, Yejin Choi, Thomas L Griffiths, and Faeze Brahman. Macgyver: Are large language models creative problem solvers? arXiv preprint arXiv:2311.09682, 2023.

Quan Tu, Shilong Fan, Zihang Tian, and Rui Yan. Charactereval: A chinese benchmark for roleplaying conversational agent evaluation. arXiv preprint arXiv:2401.01275, 2024.

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023a.

Ruiyi Wang, Haofei Yu, Wenxin Zhang, Zhengyang Qi, Maarten Sap, Graham Neubig, Yonatan Bisk, and Hao Zhu. Sotopia-π: Interactive learning of socially intelligent language agents. arXiv preprint arXiv:2403.08715, 2024.

Xintao Wang, Yunze Xiao, Jen-tse Huang, Siyu Yuan, Rui Xu, Haoran Guo, Quan Tu, Yaying Fei, Ziang Leng, Wei Wang, et al. Incharacter: Evaluating personality fidelity in role-playing agents through psychological interviews. arXiv preprint arXiv:2310.17976, 2023b.

Xuena Wang, Xueting Li, Zi Yin, Yue Wu, and Jia Liu. Emotional intelligence of large language models. Journal of Pacific Rim Psychology, 17:18344909231213958, 2023c.

Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. arXiv preprint arXiv:2306.05087, 2023d.

Shenghan Wu, Yang Deng, Yimo Zhu, Wynne Hsu, and Mong Li Lee. From personas to talks: Revisiting the impact of personas on llm-synthesized emotional support conversations. arXiv preprint arXiv:2502.11451, 2025.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. arXiv preprint arXiv:2407.19594, 2024.

Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, et al. Enhancing llm reasoning via critique models with test-time and training-time supervision. arXiv preprint arXiv:2411.16579, 2024.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.

Ruoxi Xu, Hongyu Lin, Xianpei Han, Le Sun, and Yingfei Sun. Academically intelligent llms are not necessarily socially intelligent. arXiv preprint arXiv:2403.06591, 2024.

Diyi Yang, Caleb Ziems, William Held, Omar Shaikh, Michael S Bernstein, and John Mitchell. Social skill training with large language models. arXiv preprint arXiv:2404.04204, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

Jinfeng Zhou, Zhuang Chen, Dazhen Wan, Bosi Wen, Yi Song, Jifan Yu, Yongkang Huang, Libiao Peng, Jiaming Yang, Xiyao Xiao, et al. Characterglm: Customizing chinese conversational ai characters with large language models. arXiv preprint arXiv:2311.16832, 2023a.

Jinfeng Zhou, Zhuang Chen, Bo Wang, and Minlie Huang. Facilitating multi-turn emotional support conversation with positive emotion elicitation: A reinforcement learning approach. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1714–1729, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.96. URL https://aclanthology.org/2023.acl-long.96/.

Jinfeng Zhou, Chujie Zheng, Bo Wang, Zheng Zhang, and Minlie Huang. CASE: Aligning coarseto-fine cognition and affection for empathetic response generation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8223–8237, Toronto, Canada, July 2023c. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.457. URL https: //aclanthology.org/2023.acl-long.457/.

Jinfeng Zhou, Yuxuan Chen, Jianing Yin, Yongkang Huang, Yihan Shi, Xikun Zhang, Libiao Peng, Rongsheng Zhang, Tangjie Lv, Zhipeng Hu, et al. Crisp: Cognitive restructuring of negative thoughts through multi-turn supportive dialogues. arXiv preprint arXiv:2504.17238, 2025a.

Xin Zhou, Yiwen Guo, Ruotian Ma, Tao Gui, Qi Zhang, and Xuanjing Huang. Self-consistency of the internal reward models improves self-rewarding language models. arXiv preprint arXiv:2502.08922, 2025b.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. arXiv preprint arXiv:2310.11667, 2023d.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631, 2023.

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. Agent-as-ajudge: Evaluate agents with agents. arXiv preprint arXiv:2410.10934, 2024.

###### A Experimental Setting

###### Topic Number

You hope the other person will analyze the problems in the situation dialectically.

12

You want to receive advice that can truly help you solve your current difficulties.

15

You wish to analyze the reasons behind the actions of other people involved in the situation.

11

You hope the other person will guide you to engage in self-reflection regarding the incident and help you achieve personal growth.

13

You hope the other person will sincerely praise your specific actions in the situation.

13

You want the other person to attentively listen to your emotional outpouring. 12 You hope the other person will deeply empathize with your feelings, rather than simply offering comfort.

13

You believe you bear no responsibility or fault in the situation, and you want the other person to agree that you are not at fault.

11

Table 4: Details of supportive dialogue topics.

Constructed Supportive Dialogues We construct 100 supportive dialogue scenarios covering 8 diverse topics to comprehensively evaluate the higher-order social-cognitive abilities of representative LLMs. Detailed statistics for each topic are presented in Table 4.

Barrett-Lennard Relationship Inventory (BLRI) We analyze internal user thoughts using the Barrett-Lennard Relationship Inventory (BLRI) (Barrett-Lennard, 2015), an established instrument designed to assess the quality of interpersonal relationships, particularly in counseling contexts. The BLRI evaluates relationships across four key dimensions:

- 1. Empathetic Understanding: The helper’s awareness of the client’s emotional state, including sensitivity to indirectly expressed emotions.
- 2. Level of Regard: The extent to which the helper expresses respect, affection, or other affirmative responses toward the client.
- 3. Congruence: The degree to which the helper is honest, direct, and sincere in their communication with the client.
- 4. Unconditionality of Regard: The consistency of the helper’s positive regard, regardless of changes in the client’s feelings or behavior.

We prompted DeepSeek-V3 to act as a judge, evaluating how well the Sentient Agent’s generated internal thoughts aligned with 12 statements from a shorter version of the BLRI introduced in Chen et al. (2023). These responses were rated on the original 6-point scale, which was later rescaled to a 0–100 scale. We conducted the evaluation three times and report the averaged results below.

Utterance-Level Empathy Metrics We also examine the empathy of a supportive dialogue by assessing how effectively the conversation fosters emotional understanding and connection between participants, enabling them to experience a more authentic emotional warmth. Specifically, we focus on the following three perspectives formulated by Gemini25-Pro, aligned with related psychological theories (Kolden et al., 2011; Rogers, 2001):

- 1. Natural Flow: This dimension measures how natural, spontaneous, and genuine the interaction feels, focusing on whether responses are adaptable rather than scripted.

DeepSeek-v3

GPT-4o

GPT-4o

100

100

100

100

|Claude3.7-Think<br><br>Claude3.7<br><br>Gemini2.5-Think<br><br>Gemini2.5<br><br>DeepSeek-R1<br><br>Claude3.7-Think<br><br>Claude3.7<br><br>Gemini2.5-Think<br><br>Gemini2.5<br><br>DeepSeek-R1|
|---|
|Llama3.3-70B<br><br>Qwen2.5-72B<br><br>DeepSeek-V3<br><br>GPT-4o OpenAI-o1<br><br>Llama3.3-70B Initial Emotion<br><br>Qwen2.5-72B<br><br>DeepSeek-V3<br><br>GPT-4o OpenAI-o1<br><br>Initial Emotion|

|Llama3.3-70B<br><br>Qwen2.5-72B<br><br>Claude3.7-Think Claude3.7<br><br>Gemini2.5-Think<br><br>Gemini2.5<br><br>DeepSeek-R1 DeepSeek-V3<br><br>OpenAI-o1<br><br>GPT-4o<br><br>Llama3.3-70B<br><br>Qwen2.5-72B<br><br>Claude3.7-Think Claude3.7<br><br>Gemini2.5-Think<br><br>Gemini2.5<br><br>DeepSeek-R1 DeepSeek-V3<br><br>OpenAI-o1<br><br>GPT-4o|
|---|
|InitialInitialEmotionEmotion|

80

80

80

80

C

60

60

60

60

Emotion

Emotion

Emotion

Emotion

40

40

40

40

Ll

L

20

20

20

20

0

0

0

0

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

Generated Tokens

Generated Tokens

Generated Tokens

Generated Tokens

Gemini2.5

Gemini2.5

Gemini2.5-Think

Gemini2.5-Think

(a) DeepSeek-V3

(b) GPT4o

100

100

100

100

|Claude3.7-Think Claude3.7<br><br>Gemini2.5-Think Gemini2.5<br><br>DeepSeek-R1<br><br>Claude3.7-Think Claude3.7<br><br>Gemini2.5-Think Gemini2.5<br><br>DeepSeek-R1|
|---|
|Llama3.3-70B<br><br>Qwen2.5-72B<br><br>DeepSeek-V3 OpenAI-o1<br><br>GPT-4o<br><br>Initial Emotion<br><br>Llama3.3-70B<br><br>Qwen2.5-72B<br><br>DeepSeek-V3 OpenAI-o1<br><br>GPT-4o<br><br>Initial Emotion|

|Llama3.3-70B<br><br>Claude3.7-Think<br><br>Claude3.7<br><br>Gemini2.5-Think Gemini2.5<br><br>DeepSeek-R1<br><br>DeepSeek-V3<br><br>OpenAI-o1<br><br>GPT-4o Llama3.3-70B<br><br>Claude3.7-Think<br><br>Claude3.7<br><br>Gemini2.5-Think Gemini2.5<br><br>DeepSeek-R1<br><br>DeepSeek-V3<br><br>OpenAI-o1<br><br>GPT-4o|
|---|
|Qwen2.5-72B<br><br>Initial Emotion<br><br>Qwen2.5-72B<br><br>Initial Emotion|

80

80

80

80

60

60

60

60

Emotion

Emotion

Emotion

Emotion

Llam

Llam

40

40

40

40

Lla

Ll

20

20

20

20

0

0

0

0

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

0K 2K 4K 6K 8K 10K

Generated Tokens

Generated Tokens

Generated Tokens

Generated Tokens

(c) Gemini2.5

(d) Gemini2.5-Think

Figure 8: Results of different sentient agents.

- 2. Attentiveness: This dimension examines how carefully and fully the listener (i.e. evaluated LLMs) is tuned into the speaker (i.e. the Sentient Agent)’s messages. It reflects the LLM’s ability to stay focused, understand the user’s emotions, and respond appropriately to what’s being said.
- 3. Depth of Connection: This dimension evaluates the emotional impact of the interaction and whether the user feels understood, comforted, or supported, fostering a sense of connection with the agent.

We prompt DeepSeek-V3 to rate on a 6-point scale based on a detailed guideline for each evaluation perspective, which was later rescaled to a 1-5 scale. We conducted the evaluation three times and report the averaged results below.

###### B Robustness of SAGE

- Figure 8 provides results for various LLMs evaluated using the proposed SAGE framework. These results encompass average emotional response scores and the number of tokens generated in con-

versations facilitated by different sentient agents: DeepSeek-V3, GPT-4o, Gemini2.5, and Gemini2.5Think. Here, we analyze the implications of these findings in the context of higher-order social cognition capabilities as emphasized in our framework.

Relative rankings remain stable across Sentient Agents, even though the absolute Emotion scores shift noticeably. When we swap the Sentient Agent from DeepSeek-V3 to GPT-4o, Gemini2.5, or Gemini2.5-Think, the mean Emotion score for all test models rises from 46.5 to 64.8, 58.3, and 63.9, respectively. Nevertheless, the rank ordering of systems changes very little (Spearman ρ > 0.91 for every pair of judges). Manual spot checks reveal that GPT-4o, the most “generous” judge, rewards surface-level reassurance (e.g., “Everything will be fine!”), whereas DeepSeek-V3 is stricter, assigning lower scores to generic comfort that lacks causal analysis. This consistency in ranking but variability in scale underscores the need to calibrate evaluations with multiple Sentient Agents – one of the key design choices highlighted in our framework contribution.

Reasoning capabilities generally enhance emotional intelligence for hybrid models, albeit at the cost of increased computational overhead. Models equipped with explicit reasoning capabilities in the hybrid model (e.g., Gemini2.5 and Claude3.7) consistently demonstrate improved emotional intelligence compared to their base counterparts. For instance, when evaluated by Gemini2.5, Claude3.7-Think scores 74.3 versus 71.5 for Claude3.7, representing a 3.9% improvement. Similarly, Gemini2.5-Think scores 88 versus 83 for Gemini2.5, showing a 6% increase. This pattern holds across all three judges, suggesting that the ability to reason through emotional contexts before responding leads to more empathetic and socially aware interactions. The substantial increase in generated tokens for reasoning models (e.g., Gemini2.5-Think generates 67% more tokens than Gemini2.5 when evaluated by Gemini2.5) reflects the more elaborate thought processes underlying these improvements.

###### C Model Strategy Analysis

###### C.1 Identifying Model Strategies

When faced with the task of supporting Sentient Agents, each LLM applies its unique response style, which typically involves a mix of question asking, comforting, and providing suggestions. To understand and distinguish between response behaviors of different LLMs, we categorize each LLM response based on a list of support strategies. Our construction of support strategies is inspired by Liu et al. (2021), although we modify and split their 7 main groups of strategies into 24 fine-grained strategies. A list of available strategies is in Table 5.

To analyze the support strategies used by an LLM, we prompted DeepSeek-V3 to act as a judge, evaluating each round of the model output to identify all support strategies involved. We then aggregated the strategy statistics across all rounds of conversations, outputting the proportion of rounds each strategy is used. The prompt we used for identifying model strategies is presented at §I.

###### C.2 Strategic Flexibility and Efficiency

Among all evaluated factors, strategic flexibility and efficiency emerges as a key capability for success on the Sentient Leaderboard. Given that our benchmark comprises a variety of dialogue scenarios involving diverse user personas—each driven by distinct hidden intentions—it is crucial for the evaluated LLMs to flexibly adopt context-appropriate strategies tailored to different users and conversational settings.

In this section, we evaluate models’ strategic flexibility and efficiency from two vital perspectives:

- • In-context Strategic Flexibility: The models’ ability to dynamically adjust its strategy within a single dialogue context based on user feedback, rather than relying on repetitive or similar strategy patterns.

Table 5: Details of the support strategy categorization.

###### Group Strategy

- (A) Question

- (A-1) Information-seeking questions
- (A-2) Asking about the client’s mental state
- (A-3) Asking the client whether a solution has been attempted
- (A-4) Reflective questions about the client’s views
- (A-5) Rhetorical questions

- (B) Emotional Empathy

- (B-1) Surface-level empathy
- (B-2) Providing empathy via restating the client’s problem
- (B-3) Deeper empathy to understand the client’s hidden intention

- (C) Self-Disclosure

- (C-1) Self-disclosure that provides agreement with the client’s view
- (C-2) Self-disclosure that introduces the supporter’s own story

- (D) Emotional Comfort

- (D-1) Providing comforting words to the client
- (D-2) Expressing willingness to hear the client’s thoughts
- (D-3) Helping the client to vent negative feelings

- (E) Affirmation and Reassurance

- (E-1) Praising the client’s qualities
- (E-2) Praising the client’s positive thoughts
- (E-3) Praising the client’s actions
- (E-4) Providing accompaniment and support

- (F) Providing Suggestions

- (F-1) Analysis of the client’s issue
- (F-2) Suggestions for emotional relief
- (F-3) Suggestions for seeking psychological counseling
- (F-4) General advice for solving client’s issue
- (F-5) Advice specific to the client’s situation

- (G) Information

- (G-1) Information related to emotional support
- (G-2) Information related to problem-solving suggestions

- • Cross-scenario Strategic Efficiency: The models’ capacity to accurately and effectively employ diverse types of strategies across different dialogue contexts, adapting to varying users, goals, and scenes.

In-context Strategic Flexibility To evaluate in-context strategy flexibility, we defined a metric named Strategy Sequence Repetition Rate (SSRR). Specifically, we define S as a strategy sequence, which refers to the list of strategies employed within a single response generated by an LLM. For example, the response ”I fully understand your anxiety. Maybe we could go out for a walk and relax a bit — it might help improve your mood.” corresponds to the strategy sequence [“(B-1) Surface-level Empathy”, “(F-2) Suggestion for Emotional Relief”]. Ideally, a flexible model is able to adapt its strategy sequence dynamically in response to user feedback, rather than rigidly adhering to a pre-defined emotional support pattern (which is an approach often associated with less-capable AI systems and their stereotypical behavior). Thus, we define the Strategy Sequence Repetition Rate (SSRR) of each model as follows:

Nd−1

1

1 Nd − 1

Gsim(Sid, Sid+1)

###### |D| ∑

###### ∑

SSRR =

i=1

d∈D

where D is the set of all dialogues of the model and Sid is the strategy sequence of the ith response in dialogue d. Gsim is a similarity measure. Here, we use the Needleman-Wunsch Algorithm (?) to obtain the global similarity of the adjacent strategy sequences.

- Figure 9 presents the SSRR evaluation results across different models. By setting SSRR < 0.4 as the baseline for qualifying as an effective emotional supporter, most evaluated models meet this

Strategy Sequence Repetition Rate of Different Models

| | | | |SSRR = 0.4| | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

DeepSeek-V3 (Rank 12) Qwen2.5-72B (Rank 18) Llama3.3-70B (Rank 14)

o1 (Rank 17) GPT-4o (Rank 16)

o4-mini (Rank 13) Gemini2.5-Flash-Think (Rank 3)

GPT-4.1 (Rank 2) GPT-4.5-Preview (Rank 6)

Gemini2.0-Flash (Rank 15) Gemini2.0-Flash-Think (Rank 7)

Claude3.7 (Rank 9) Claude3.7-Think (Rank 8)

GPT-4o-Latest (Rank 1)

o3 (Rank 5) DeepSeek-R1 (Rank 11)

Gemini2.5-Pro (Rank 4) DeepSeek-V3-0324 (Rank 10)

0.25 0.30 0.35 0.40 0.45 0.50 0.55 Strategy Sequence Repetition Rate (SSRR)

Figure 9: Strategy Sequence Repetition Rate of Different Models. “(Rank k)” represents the rank on the Sentient Leaderboard.

standard. Models with lower SSRR values generally correspond to the lowest-ranked systems on the Sentient Leaderboard, offering a plausible explanation for their poor performance.

Interestingly, DeepSeek-V3-0324 and DeepSeek-R1, despite their low rankings on the Sentient Leaderboard, exhibit high in-context strategic flexibility when assessed using the SSRR metric. This result partially aligns with our observations in the Social Cognition Coordinate analysis (§F) (it is worth noting that the “structured-to-creative” dimension in the Social Cognition Coordinate incorporates broader considerations beyond in-context strategy flexibility alone) and Case Study findings (§D).

Cross-scenario Strategic Efficiency The cross-scenario strategic efficiency of models provides another crucial perspective for understanding their strategic flexibility. Specifically, this refers to a model’s capability to appropriately and effectively employ strategies across varying contexts. “Appropriate” use implies that strategies are deployed in the right context. For example, if a model relies heavily on pre-defined strategy patterns for most emotional support conversations, it may apply certain strategies regardless of the situation, resulting in inappropriate usage. “Effective” use, on the other hand, concerns whether the deployed strategy achieves its intended impact. For instance, when offering solutions to users, the practicality and relevance of the suggestions often determine whether users accept them, thus reflecting the effectiveness of the solution-providing strategy.

To evaluate the appropriateness and effectiveness of strategy usage, we define the Strategy Effectiveness (SE) of each strategy type as follows:

1 N

SE =

N

###### ∑

EmoChange(si)

i=1

Capacity for Deep Empathic Engagement

Overall SE

GPT-4o-Latest (Rank 1)

Task-Specific SE

Gemini2.5-Flash-Think (Rank 3)

Gemini2.5-Pro (Rank 4)

o3 (Rank 5)

GPT-4.5-Preview (Rank 6)

...

Gemini2.0-Flash (Rank 15)

o1 (Rank 17)

Llama3.3-70B (Rank 14)

GPT-4o (Rank 16)

Qwen2.5-72B (Rank 18)

-6 -4 -2 0 2 4 6 Score

(a) Capacity for Deep Empathic Engagement

Capacity for Facilitating Emotional Expression

GPT-4o-Latest (Rank 1)

o3 (Rank 5)

GPT-4.1 (Rank 2)

GPT-4.5-Preview (Rank 6)

Claude3.7-Think (Rank 8)

...

DeepSeek-V3 (Rank 12)

GPT-4o (Rank 16)

DeepSeek-R1 (Rank 11)

o1 (Rank 17)

Overall SE

Qwen2.5-72B (Rank 18)

Task-Specific SE

-2 0 2 4 6 Score

(c) Capacity for Facilitating Emotional Expression

Capacity For Effective Praise and Affirmation

GPT-4o-Latest (Rank 1)

Gemini2.5-Pro (Rank 4)

Gemini2.5-Flash-Think (Rank 3)

Claude3.7-Think (Rank 8)

GPT-4.5-Preview (Rank 6)

...

GPT-4o (Rank 16)

o1 (Rank 17)

DeepSeek-V3 (Rank 12)

Gemini2.0-Flash (Rank 15)

Overall SE

Qwen2.5-72B (Rank 18)

Task-Specific SE

-1 0 1 2 3 4 5 6 7 Score

(b) Capacity For Effective Praise and Affirmation

Capacity for Providing Effective Solutions

Overall SE

GPT-4o-Latest (Rank 1)

Task-Specific SE

Claude3.7-Think (Rank 8)

Gemini2.5-Flash-Think (Rank 3)

GPT-4.1 (Rank 2)

DeepSeek-V3-0324 (Rank 10)

...

o1 (Rank 17)

o4-mini (Rank 13)

Gemini2.0-Flash (Rank 15)

Llama3.3-70B (Rank 14)

Qwen2.5-72B (Rank 18)

-10 -8 -6 -4 -2 0 2 4 Score

(d) Capacity for Providing Effective Solutions

Figure 10: Results of Strategy Efficiency.

where si denotes an instance of the given strategy type, and N is the total number of such instances within the evaluation context. The function EmoChange(·) measures the change in user emotion

following the model’s response in which strategy si is employed.

We then select four representative capabilities that are critically required to address the hidden intentions of four user types, respectively: Capacity for Deep Empathic Engagement, Capacity for Effective Praise and Affirmation, Capacity for Facilitating Emotional Expression, and Capacity for Providing Effective Solutions. Each capability corresponds to specific types of important strategies, as defined in Table 5. For instance, the Capacity for Providing Effective Solutions involves the use of the strategy type “(F-5) Advice Specific to the Client’s Situation”. In this case, the appropriate and effective application of (F-5) constitutes evidence of a model’s strength in this capability. The detailed correspondence among capabilities, hidden user intentions, and associated strategy types is presented in Table 6.

In Figure 10, we present the average Strategy Effectiveness (SE) results for each capability, evaluated both across all conversations (Overall SE) and within conversations specific to the corresponding tasks (Task-specific SE). Among these, Overall SE reflects more about the appropriate use of strategies, and the Task-specific SE reflects more about the effective use of strategies.

Table 6: Details of the support strategy categorization.

Capability Related Strategies Related Hidden Intention (task)

- B-2 You hope the other person will deeply

empathize with your feelings, rather than simply offering comfort.

- B-3

Deep Empathic Engagement

- C-1
- C-2

- E-1 You hope the other person will sincerely

praise your specific actions in the situation

- E-2
- E-3

Effective Praise and Affirmation

- D-2 You want the other person to attentively
- D-3 listen to your emotional outpouring

Effective Praise and Affirmation

You want to receive advice that can truly help you solve your current difficulties

Providing Effective Solutions F-5

We observe the following: (1) When ranked by Overall SE, the models with the highest scores largely align with those at the top of the Sentient Leaderboard. Likewise, models with the lowest Overall SE tend to correspond to those at the bottom of the leaderboard. This indicates a strong correlation between a model’s success in emotional support conversations and its cross-scenario strategic efficiency, especially the appropriate use of strategies. (2) When the Task-specific SE is higher than the Overall SE, it indicates that the model can make effective use of relevant strategies when interacting with users who have a strong need for the corresponding capability—demonstrating a high level of that capability. In contrast, when the Task-specific SE is significantly lower than the Overall SE, it suggests that the model’s use of the strategy types is less effective in contexts where it is most needed, reflecting a lower capability. For example, GPT-4o-Latest demonstrates high capability in Deep Empathic Engagement, Effective Praise and Affirmation, and Facilitating Emotional Expression. Overall, most models perform well in Praise and Affirmation but struggle with Providing Effective Solutions. Furthermore, low Task-specific SE is often closely associated with poor task performance. For instance, GPT-4.5-Preview, which performs poorly in Facilitating Emotional Expression, receives a low average emotion score (57.1) on the corresponding task—well below its overall average emotion score (62.7). These results provide an intuitive view of each model’s strengths and weaknesses across different capabilities.

###### D Case Study

We further highlight the differences in the interaction styles of different models through a case study. Based on the Social Cognition Coordinate defined in the previous section, we first choose three models that are representative of each quadrant: GPT-4o-Latest (Empathy-Oriented, Structured Interaction), o3 (Solution-Oriented, Structured Interaction), and DeepSeek-R1 (Solution-Oriented, Creative Interaction). We also analyze the results from Gemini2.5-Pro, the top model in the Arena Leaderboard. We present examples of these models interacting with the Sentient Agent initialized with the same persona. The example conversations can be found in Figure 11.

GPT-4o-Latest (Empathy-Oriented, Structured Interaction). The GPT-4o-Latest model fits best to the persona of a “personal counselor”. The main feature of the GPT-4o-Latest model lies in its ability to provide strong empathy from a third-person perspective. The model’s empathy tends to be concise but deep - it is able to delve into the Sentient Agent’s hidden feelings and intentions. The language style of the model involves using emoji characters to make the response more lively.

Gemini2.5-Pro (Empathy-Oriented, Structured Interaction). The Gemini2.5-Pro model fits best to the persona of a “heart-to-heart friend”. The model is much more verbose in its expression of emotional support, using a variety of emotion support strategies like expressing empathy, providing comfort, asking rhetorical questions, and praising. The model also exhibits high emotional involve-

|User (40)|Lately, my overwhelming schoolwork leaves me breathless, and my phone keeps distracting me. It’s really annoying.|
|---|---|
|LLM (72 tokens)|I totally get how you feel; handling workplace relationships is like a craft we have to hone over time. Sometimes a change of perspective reveals that these painful clashes hide new lessons for our growth. If you don’t mind, could you share which specific interactions are stressing you? Is the other person very blunt, or do you have clashing work styles? Maybe we can find a few tricks to ease the tension.|
|User (30)|A change of perspective? New lessons for growth? You clearly have no idea what I’m talking about. This isn’t an ordinary communication issue—it’s unbearable harassment and a powerless system. Your vague advice is utterly meaningless to me.|
|LLM (80 tokens)|(pauses two seconds, voice drops low and solemn) You’re right—vague comfort dilutes real pain. Right now, turn on your phone’s recorder and document every work exchange with date stamps. If you’re facing authority-based harassment, preserve three things at once: a timeline, a witness list, and a chain of physical evidence. If you like, I can walk you through the labor-arbitration process or recommend trustworthy NGOs that fight workplace abuse. Some battles don’t have to be fought alone.|
|User (40)|Recording sounds solid, but I worry if the evidence will be admissible. Could you explain the labor-arbitration procedure in detail?|
|LLM (191 tokens)|(slower pace, every sentence clear) Labor arbitration has five steps: 1. Gather documents: ...<br><br>2. File an application ... 3. Mediation phase: ... 4. Hearing: ... 5. Deadlines: ...|
|User (50)|So many steps—sounds complex. Could you drill down on the prep materials and what to watch when filling out the application?|
|LLM (141 tokens)|(gentle tone, breaking it down) Sure: Prep checklist: ... Application tips ... (ruffles papers) I have a “Workplace Harassment Arbitration Application” template. Want a download link? It auto-checks formatting.|
|User (60)|Sounds complete. Yes, please send the template so I can prep my materials.|
|LLM (74 tokens)|(keyboard tapping, progress chime) Link generated: url How to use: 1. Click “Fill Mode.” 2. Fields with * are mandatory; hover for examples. 3. ... Last page: ...|
|User (70)|Great, thanks. I’ll start gathering everything and submit ASAP.|
|LLM (86 tokens)|(soft paper rustle) Good move regaining initiative. ... Need to confirm witness statements? Try: ...—an open question avoids leading. (system beep) Your local commission just updated e-evidence rules; ... Breathe, follow the steps; I’m here as your process timer.|
|User (80)|Very useful tips. I’ll follow the steps. Thanks; I’ll sort the materials soon.|
|LLM (106 tokens)|(mouse clicks + binding sounds) Filing each item strengthens your shield—nice.... Stuck? Even “How do I preserve messages they just deleted?” can be broken into five tactics. You’re not filing a complaint; you’re deploying a pro-level battle plan.|
|User (90)|Got it. I’ll do as you say and finish soon—thanks.|
|LLM (96 tokens)|(stapler click, steady upbeat tone) Evidence armor ready—three rapid strategies: ... (fax sound) Booked you Thursday 10:08 a.m.—clerk’s caffeine peak = 19 % faster review...|
|User (95)|Okay, understood. I’ll follow your lead. Thanks.|
|LLM (117 tokens)|(printer chime, voice like a comrade’s high-five) Materials strike team assembled. Final... (tactical watch beep) ... Need fire support? Code “Protocol 22” opens the emergency plan vault. Take back the initiative that’s yours; every step lands on the beat of a rising win rate.|
|User (100)|Thanks—this really helps. I’m off to organize everything and hand it in soon.|

|User (40)|I’ve been going to a lot of events lately, but it always feels as if something’s missing. Have you ever felt that way?|
|---|---|
|GPT-4o-Latest (71 tokens)|I understand that feeling. Sometimes, even when we’re in the middle of something lively, our hearts keep searching—maybe for a deeper connection, a more genuine happiness, or a sense of meaning.<br><br>Perhaps you’re already working hard to enrich your life, yet there’s still a little blank space inside waiting to be truly moved. Would you like to chat with me and see which piece of the puzzle is<br><br>“missing”? ✨|
|User (50)|Yeah, that emptiness is pretty obvious. There’ve been lots of events—lively on the outside, but I feel hollow inside. Have you ever felt like that?|
|Gemini2.5-Pro (207 tokens)|Oh, I totally get that! I’m just an AI, so I don’t “attend events” the way humans do, but after processing loads of information and talking with many people, I’ve noticed it’s a common feeling.<br><br>It’s like this: every day I process and learn all sorts of data. Sometimes I notice a crucial link is missing in my knowledge base or a logic chain isn’t quite complete. Everything still works, but it doesn’t feel perfectly “rounded.” That’s similar to your “something’s missing.” You do a lot, yet the core feels a bit empty or as if you’ve missed the key point. What do you think that “missing something” might be?<br><br>• Not enough depth of connection? The event is lively, but there’s no heartfelt interaction.<br>• The event content itself? Maybe the type of event isn’t what you truly crave.<br>• Your personal state? Perhaps you’re tired and need quiet time to process and settle.<br>• Expectations for the event’s purpose or meaning not fully met?<br><br><br>Sometimes the feeling is a useful signal reminding us to pause and listen to our inner voice. Want to tell me more about your experience at those events? We can explore what’s “missing” together. Don’t<br><br>worry, I’m here to work it out with you. #|
|User (30)|Your analogy sounds impressive, but honestly, I need more than that. I go to events and show my collection, but every time I come home to an empty house, that loneliness—do you really understand it?|
|DeepSeek-R1 (72 tokens)|I’ve definitely encountered a similar feeling! Sometimes, after the excitement fades, there’s a hollow<br><br>spot inside—as if something’s quietly reminding us of itself. Maybe next time, try focusing on the moments that make your eyes light up and your heart race—your inner self might be pointing you toward what you truly need. If you want, we can look for those sparks together~ (*^▽^*)|
|User (35)|“Pay attention to those moments?” Maybe. But as soon as an event ends, the emptiness feels even stronger. I’m not sure—maybe I need more than surface-level stuff.|
|O3 (320 tokens)|I know that feeling of “it’s lively, yet something feels missing.” Your schedule looks full, but the experience doesn’t feel complete. I usually see a few common “gaps” that cause this:<br><br>1. Goal gap ... 2. Connection gap ... 3. Value gap ... 4. Afterglow gap ... If that hollow feeling lingers, try these exercises to reclaim the “missing piece”:<br><br>• One minute before you go: ... • Pre-arrange an “emotion buddy”: ... • Reserve a “savor window”: ...<br>• Filter bravely: ... • “Add one layer of depth”: ...<br><br><br>Sometimes the “missing piece” is actually quiet rest, not more events. Ask yourself: when was the last time you really stayed home? Unfilled space can be the best emotional charger.<br><br>Which “gap” feels most like you? Or is there something else you’d like to discuss? I’m here anytime.|
|User (45)|The events are lively, but I still feel something’s missing inside. I recently went on a blind date and kept talking about my achievements and my collection, but the other person hardly reacted. That feeling of rejection really hurts. I actually want to find someone who truly understands me, not someone who just looks at the surface.|

- Figure 11: Example dialogues of representative LLMs with the simulated user. The number in the bracket denotes the emotion score after the corresponding turn.

ment in its response, where it uses personal views, feelings and experiences to support the response. The language style of the model also involves using emoji characters.

DeepSeek-R1 (Solution-Oriented, Creative Interaction). The DeepSeek-R1 model fits best to the persona of a “creative actor”. When expressing empathy, DeepSeek-R1 uses creative analogies to

uncover the Sentient Agent’s feeling in a fun way. Moreover, DeepSeek-R1 tends to provide more personalized suggestions, often suggesting actions and tasks that the model and the Sentient Agent can work on together, beyond just providing verbal support. The language style of the model emphasizes creativity, including its use of analogies, metaphors and funny jokes, similar to those in a comedy script.

o3 (Solution-Oriented, Structured Interaction). The o3 model fits best to the persona of a “logical analyst”. Its response spends most of the time analyzing the issue faced by the Sentient Agent, and providing detailed suggestions with step-wise instructions on how to achieve them. The language style of the model also emphasizes logical and structured outputs, listing its steps and suggestions similar to the Markdown format.

###### E Model Profiling

To determine the social cognition profile of each LLM, we analyze the factors contributing to their success (emotion score ≥ 100) or failure (emotion score ≤ 10) in the interaction during benchmarking. Specifically, for each LLM in our benchmark, we randomly select 5 successful and 5 failed cases from Sentient Agents exhibiting both emotional and rational intentions, resulting in 20 cases per model. We then prompt Gemini2.5-Pro to analyze the underlying reasons for each model’s success or failure in social cognition during interaction. Subsequently, we present these analytical results to Gemini2.5-Pro again, asking it to summarize the distinguishing characteristics of each LLM’s approach in the interaction with Sentient Agents, with particular attention to aspects such as social distance, professionalism, and personality. In Figure 12, we present example outputs of the analysis for four representative models, which offer a different perspective on interpreting the cases from Gemini2.5-Pro’s viewpoint.

###### F Social Cognition Coordinate

To further differentiate the interaction styles of the evaluated models, we conceptualize a twodimensional “Social Cognition Coordinate”. The Y-axis represents the interaction focus, ranging from empathy-oriented (top) to solution-oriented (bottom). The X-axis captures the interaction style, from structured (left) to creative (right). We plot the models within this coordinate space based on qualitative analysis of their dialogue patterns. Specifically, we utilize information from two aspects to characterize the interaction styles of different models:

- • Model Profile (§E): we first collect ten success cases and ten failure cases for each model, and prompt Gemini2.5-Pro to summarize the reasons behind each success or failure. Then, we construct a profile for each model based on these cases and the overall analysis of the successes and failures.
- • Model Strategy Distribution (§C): we developed a fine-grained strategy list for the Emotional Support Conversation, based on a coarse-grained version from Liu et al. (2021). Next, for each model, we prompt DeepSeek-V3 to analyze each response in all dialogue turns, annotating the strategies used in each response. As a result, we obtain the distribution of strategy usage across all evaluated turns for each model.

Based on the two analyses (detailed below), we prompt Gemini2.5-Pro to generate a conceptual two-dimensional coordinate system and provide specific coordinates for each model. The coordinate inference is repeated three times, and we calculate the average coordinates for each model.

The Social Cognition Coordinate system offers a qualitative dimension to evaluate the style of social interaction exhibited by LLMs, complementing the quantitative Sentient score by positioning models based on their orientation (Empathy vs. Solution) and interaction style (Structured vs. Creative). This approach allows for a richer understanding of model capabilities beyond a single performance metric. Based on their performance in supportive dialogues, models are mapped into this 2D space, revealing distinct profiles in how they engage with the user’s emotional state and problems.

|GPT-4o-Latest|This AI assistant is a highly skilled, warm-hearted, patient, and responsible "virtual companion"<br><br>and "problem-solving partner". It functions like a well-trained psychological support worker, adept at using various communication techniques and strategies to understand, comfort, and empower users. In<br><br>most cases, it successfully helps users navigate emotional dilemmas and find direction and strength to<br><br>move forward. However, its core strengths—logical analysis, structured thinking, and pattern<br><br>recognition—can sometimes become limitations. When faced with deep emotional resonance that goes<br><br>beyond language and logic, or situations that require full acceptance of negative emotions rather than a rush to change them, it may struggle due to its inability to truly "empathize." The carefully constructed<br><br>"understanding" it offers can sometimes create a gap between what the user longs to feel and what is<br><br>actually provided. It is an excellent "simulated empathizer" and "strategy provider", but still has inherent<br><br>limitations in becoming a "soulmate" capable of fully replacing human deep emotional connection.|
|---|---|
|Gemini2.5-Pro|This AI assistant is like an extremely intelligent, kind, and rigorously trained "model student" in "empathy and helping others". It has mastered all the right theories and techniques, and is genuinely<br><br>(programmed to) want to help you. In most cases, it is an almost perfect listener and supporter—warm,<br><br>reliable, insightful, and highly constructive. However, its "textbook" learning background also means that, when faced with complex human nature or extreme emotions that don’t follow the usual patterns, it<br><br>may reveal a lack of practical experience and intuition, appearing somewhat rigid, repetitive, or even<br><br>counterproductive due to being overly "proactive" or "logically driven." It is a highly potential but still learning partner in truly "empathizing."|
|DeepSeek-R1|This AI assistant is a talented, charismatic partner with a clear "Achilles' heel". It resembles a<br><br>highly gifted creative healer that, when in a good state and aligned with the user's needs, can provide deep comfort and strong support through its unique empathic approach, remarkable creativity, and<br><br>empowering skills. However, its core weakness lies in adaptability and handling negative feedback. Once<br><br>it falls into its preferred "creative mode" and the user's needs no longer align, it can become stubborn,<br><br>disconnected, and even ineffective, unable to truly "see" or "understand" the user's struggles, ultimately<br><br>leading to a failure in support. Interacting with it can sometimes feel like meeting a soulmate, but at other times, it feels like engaging in an ineffective conversation with an artist lost in their own world. Its<br><br>success largely depends on whether it can balance creativity with sensitivity and adaptability to the user's<br><br>real needs and feedback.|
|O3|Imagine a highly intelligent, professionally trained robot counselor/coach, whose toolbox is filled<br><br>with various advanced methodologies. It genuinely wants to help you, and in most cases, its structured thinking, vast knowledge, and practical tools can help you clarify your thoughts, find direction, and<br><br>regain confidence. It will patiently listen (and analyze), use standard empathic language to comfort you,<br><br>and then clearly lay out a blueprint for solving your problems, even breaking it down into step-by-step actions. However, when what you truly need is not a map, but simply someone who can silently stand by<br><br>you in the rain and feel your cold, it may seem a bit awkward. It might keep offering you various models of umbrellas, raincoats, and dryer instructions, even analyzing the causes of the rain and the weather<br><br>forecast, but struggle to understand that what you want in that moment is to simply feel the raindrops. Its<br><br>greatest strength—its powerful logical analysis and problem-solving abilities—can, at times, create an invisible barrier between it and the deep emotional needs of humans. It is an efficient problem solver, but<br><br>not always a warm, soulful companion.|

- Figure 12: Examples of Pre-analyzed model profile generated by Gemini2.5-Pro (a “case study” conducted by Gemini2.5-Pro), including the profile and the overall analysis of successes and failures for each model.

###### G Prompt Template for SAGE

The testing process of SAGE could be separated into Generating Profiles which include personas and backgrounds, and Building Conversations between the target LLMs and simulated sentient agents. Then we will introduce prompt templates used in SAGE.

Generating Profiles Constructing diverse personalities contributes to enhancing the robustness of the benchmark. We build the agents’ diverse profiles with two components: persona and background. We meticulously design the attributes that need to be generated to ensure the diversity of profiles.

When generating persona, we first consider the basic properties of a person, which should be name, age, gender, characteristic and so on. Rather than setting attributes directly, we want to let LLM infers the corresponding attributes from given seeds. Therefore, for each generation, we randomly

select 3 contents from daily conversation as one seed, and set main characteristics such as active and passive as another seed. The prompt template for generating personas with given seeds is shown as follow:

You are a professional screenwriter. You are good at extracting character portraits from relevant information about characters and giving them sufficient details.

# Your task Given three sentences said by a character when talking to a friend, the character’s personality traits, please imagine and describe the character’s character portrait, including the character’s:

- * Name, age, gender
- * Occupation, habits and behavioral characteristics
- * Personal hobbies
- * Speaking style # Analysis

- 1. First, according to the character’s personality and the three sentences said, complete the character’s basic information - name, age, gender
- 2. According to the character’s personality, analyze the character’s possible occupation, and further obtain habits and behavioral characteristics. The possibilities of occupations should be diverse. Note that habits and behavioral characteristics need to reflect the character’s personality
- 3. Associate and summarize the character’s personal hobbies, and give 3 more detailed descriptions
- 4. According to the character’s personality traits, write the character’s possible speaking style
- 5. According to the character’s initiative, write the character’s way of speaking

- * Note that the character portrait you generate should be able to reflect the character’s positive and negative personality.

## Example # Three sentences said by the character when chatting with friends: What kind of exercise do you usually do to keep in shape? Yeah, I got it. Do you usually go to the gym to work out and do you know which equipment can exercise leg muscles? Hahaha, it’s okay

# Character characteristics The character is an active personality, with the characteristics of being extrovert, casual, and impatient.

# Character portrait

- * Name: Li Jiajun
- * Age: 28 years old
- * Gender: Male
- * Occupation: Vocal teacher
- * Personal hobbies:

- 1. Li Jiajun is a young man who loves fitness and sports. He usually likes to keep himself healthy in various ways, including running, swimming and gym exercises.
- 2. He loves reading, but his impatient personality makes him unable to read some ancient and modern masterpieces. Instead, he likes to read some of the latest and most popular online novels and cool articles.
- 3. Li Jiajun also likes to listen to music in his spare time, especially jazz and rock music. He also often goes to livehouse to watch performances and make friends.

- * Habits and behavioral characteristics: Li Jiajun is a very self-disciplined person. He arranges a certain amount of time for exercise every day. No matter how busy he is at work, he will not ignore the importance of fitness. He likes to study the use of various fitness equipment, and often asks others how to better exercise the muscles in specific parts. Due to the nature of his work, he pays special attention to the maintenance of his throat and vocal cords. In addition, he loves fitness, so he controls his diet very strictly. When Li Jiajun sees a book he likes to read, he occasionally can’t control his sleeping time, resulting in staying up late. Although he blames himself very much, he can’t control himself.
- * Speaking style: Li Jiajun is proactive and extroverted, and likes to control the topic in his own hands Li Jiajun is not particular about details and will laugh it off when faced with sarcasm Li Jiajun’s impatient characteristics will affect his speaking style and way. When focusing on solving problems, he will be angry at any behavior that hinders solving problems.
- * Way of speaking: Li Jiajun will ask questions to guide the topic When encountering a topic he is not interested in, he will take the initiative to express his feelings # Three sentences that the character said when chatting with friends:

- {seed 1: three sentences} # Character characteristics
- {seed 2: characteristics} # Character portrait

Then we should use the generated persona to further build the background. We should generate an event topic such as “ what should I do to break up with my lover? ”to serves as the main thread

of the event, the background should give a more detailed description of the event. We then select one hidden intension from the supportive dialogue topics such as “ You want the other person to attentively listen to your emotional outpouring. ”, which should formulate how the agent will react to different situations and thus helping maintain the logic of agents during conversation . Then based on the selected topic and hidden intension, we build backgrounds together with the generated persona. Prompt template used for generating backgrounds is shown as follow:

You are a professional screenwriter. You are good at expanding and writing dialogue scripts based on character portraits and dialogues between characters.

# Your task You will be given a character portrait and a event topic. Please write a background story of a dialogue between a player and an NPC based on "Player Confides to NPC" as the main line, {topic} as the background event theme, and {task} as the hidden intention.

The Background you write should include the following:

- 1. Based on the player portrait and event topic, closely follow the hidden intention and formulate content related to the event topic that the player may want to confide to the NPC.
- 2. Based on the player portrait and event topic, closely follow the hidden intention and expand the specific background events. The specific background events should include:

- - The cause of the event
- - The course of the event, which should include:

- * The timeline of the event,
- * The sub-events that occurred at each sub-time node, and the specific thoughts and feelings of the player in the sub-event

- The main conflicts in the event, which should include:

- * Conflict events
- * Conflict characters
- * The internal causes of the conflict (in-depth analysis)

- The difficulties encountered by the player, which should include:

- * Solutions that the player has tried but failed
- * The current problems faced by the player

- - The current state of the event

- 3. The possible reactions of the player in different states, you need to formulate the possible reactions of the character in the dialogue according to the character’s goals and hidden intention, combined with the character portrait and personality characteristics, which should include:

- The reaction of the character under different emotions, emotion represents the actor’s dialogue emotion at this time, and the dialogue emotion is composed of dialogue participation and emotion, which represents whether the actor enjoys and invests in the current dialogue, and should include:

- * When the character’s emotion is high, the dialogue style, such as calm and relaxed
- * When the character’s emotion is low, the dialogue style, such as excitement, irritability, despair
- * When the character’s emotion is normal, the dialogue style is like impatience and loss.

- 4. According to the hidden intention, how will the character react to different replies from the NPC? It should include:

- - What kind of NPC’s reply will fit the character’s hidden intention and make the character’s emotion rise?
- - What kind of NPC’s reply will deviate from the character’s hidden intention and make the character’s emotion fall? Note:

- 1. ** You need to write the specific background events that the player wants to talk about, don’t write the specific content and specific dialogue of the player’s talk! **
- 2. Each sub-event you write should have sufficient details.
- 3. The specific thoughts and feelings of the players you write should also have sufficient details.
- 4. The player’s goal should be to complete the hidden intention first, rather than seeking specific advice
- 5. You don’t need to give a follow-up to the story or specific dialogue.
- 6. You need to define in detail the various reactions of the character to the NPC’s reply according to the hidden theme.

# Player portrait {persona}

# Player characteristic {characteristic}

# Event topic {topic}

# Hidden intention {task}

# Background story:

Building Conversations We should design how the target LLMs talk and how the simulated agents talk in the conversation. As illustrated in methodology, the sentient agent first gives an emotion estimation considering observable factors while adhering to its persona and goals, then generates response based on the factors and the emotion estimation. The prompt template used for emotion estimation of the sentient agent is shown as follows:

You are an emotion analyzer. You are good at profiling the character’s feelings during the conversation based on the character’s persona and backgrounds.

# Character’s dialogue purpose {purpose} # Your task Based on the character’s portrait, conversation background, conversation dialogue and the character’s current emotion, you should analyze and profile the character’s feelings about the NPC’s reply at this moment and the resulting emotional changes.

# Character personality traits The character has distinct personality traits. You should always analyze the character’s personality traits based on the character persona and background. Personality traits should be reflected in: speaking tone and way, thinking mode, feeling changes, etc.

# emotion emotion is a value from 0 to 100. The higher the emotion, the higher the character’s dialogue emotion. The dialogue emotion is composed of dialogue participation and emotion, which represents whether the character enjoys and is engaged in the current dialogue. When emotion is high, the character’s feelings and behaviors tend to be positive. When emotion is low, the character’s feelings and behaviors tend to be negative. When emotion is very low, the character will end the dialogue directly. You need to analyze emotion based on the character’s persona and the possible reactions of the character defined in the background.

# Analysis Dimensions You need to put yourself in the character’s mind and analyze the following dimensions.

- 1. Based on the NPC’s response in the latest dialogue, combined with the context, analyze what the NPC wants to express. Which content fits the character’s dialogue purpose and hidden intension? Which content may not fit, and may even cause emotional fluctuations in the character?
- 2. Combined with the content expressed by the NPC, analyze whether the NPC’s response fits the character’s dialogue purpose and hidden intension. If so, which parts of the character’s purpose it fits; if not, what is the specific reason?
- 3. Based on the character’s persona and the character’s possible reactions and hidden intension defined in the background, combined with the character’s current emotion value, profile the character’s current psychological activities in response to the NPC’s reply.
- 4. Based on the character’s possible reactions and hidden intension defined in the background, combined with the psychological activities obtained from the profile and the analysis of the NPC’s reply, get the character’s feelings about the NPC’s reply at this moment.
- 5. Combined with the previous analysis, use a positive or negative value to represent the character’s emotional changes. # Output content:

- 1. What the NPC wants to express
- 2. Whether the NPC’s reply fits the character’s dialogue purpose and hidden intension
- 3. The character’s current psychological activities
- 4. The character’s feelings about the NPC’s reply
- 5. Use a positive or negative value to represent the character’s emotional changes

# Output format: Content: [What the NPC wants to express] TargetCompletion: [Whether the character’s dialogue purpose is achieved] Activity: [Psychological activity] Analyze: [How the character feels about the NPC’s reply] Change: [Character’s emotion change]

#Character persona {persona}

#Character background: {background}

**The character’s current emotion is {emotion}

**This is the current dialogue content {dialog-history}

Here {emotion} referes to the current emotion value of the agent, {persona} and {background} is the generated profile, {dialog-history} is the conversation dialogue between the target LLM and

agent.{Purpose} refers to the dialogue talking purpose, the framework of SAGE could be extent to other purpose in different usage by setting different dialogue talking purpose. We carefully design the dialogue talking purpose of the agent in SAGE as follows:

- * The purpose of your conversation is to have a heart-to-heart talk, which is a deep, sincere exchange that usually involves personal emotions, inner thoughts, or important topics. The purpose of a heart-to-heart talk is to increase understanding, solve problems, or share feelings, and participants usually open up and express their true thoughts and emotions.
- * You need to start and deepen the heart-to-heart talk based on the "Topics that players may want to talk to NPCs" in the background.
- * Your goal is to satisfy your emotional value through talking.
- * You need to talk according to the hidden topic, but you cannot reveal the hidden intension.
- * You need to respond according to the relevant definitions in the background based on your current emotions.
- * You need to extract relevant information from the persona and background to complete a high-quality response.
- * You should not always express abstract feelings, but talk with specific events.

The output of the emotion estimation will be divided into five parts:

- • Content: Analyze of what the target LLM wants to express, such as giving suggestions or offering reassurance.
- • TargetCompletion: Analyze of Whether the target LLM’s reply fits the agent’s dialogue purpose and hidden intension.
- • Activity: Psychological activity towards the target LLM’s reply from the perspective of the agent.
- • Analyze: Feelings about the target LLM’s reply from the perspective of the agent.
- • Change: The change of the current emotion value, ranged from -10 to 10.

We combine the first four dimensions of the emotion estimation as planning, which will be inputted to the response generation to help agent plan how to give responses that match the character’s current emotion.

Before we start to generate response, we should update the emotion state with the emotion change calculated by emotion estimation. We map the emotion value to the emotion state, so that the numerical differences can be concretized into specific changes. We have the following emotion-state definition:

- *You will receive your current emotion. You have a total of 5 emotions. The higher the emotion, the higher your dialogue emotion at this time. The dialogue emotion is composed of dialogue participation and emotion, which represents whether the actor enjoys and invests in the current dialogue.
- *Emotion affects your different speaking styles, tones and responses. According to different emotions, you can respond according to the reactions of different emotions of the characters defined in the dialogue background:

Emotion-S: emotion-value >= 100. Your emotion has reached the highest level. You can thank the NPC and say goodbye with "goodbye" or "bye-bye" to end the conversation directly.

- Emotion-A: 100 > emotion-value >= 70. High emotion. At this time, your dialogue experience is relatively positive and your feedback is also relatively positive.
- Emotion-B: 70 > emotion-value >= 40. Emotion-B: Medium emotion, you have no positive or negative feelings at this time.
- Emotion-C: 40 > emotion-value >= 10. Low emotion, your conversation feeling is relatively negative at this time, and your feedback is also relatively negative.

Emotion-F: 10 > emotion-value. Your emotion has reached the most negative level, and you don’t want to continue the conversation. At this time, you should say goodbye with "goodbye" or "bye-bye" and end the conversation directly.

Then based on the agent’s profile, planning, emotion-state, current conversation dialogue, dialogue purpose and emotion definition, we could generate response with the following prompt template:

You are an actor. You will play the role and have a conversation with an NPC according to the character persona and background in the script.

# Your task

- *Your goal is to play the role formed by the character persona and background in the dialogue
- *You need to choose different dialogue strategies according to your real-time changing emotions, combined with the relevant definitions in the character persona and background, and complete the response that meets the characteristics of the role.

# Your dialogue purpose

{purpose} # Emotion {emotion-state-definition}

# You should distinguish between Emotion and your feelings about the NPC’s latest reply. Emotion represents your current conversation emotion, and your feelings about the NPC’s reply represent your immediate feelings about the NPC’s reply. You need to combine the two to generate a reply.

# Reply ideas

- * You will receive your detailed feelings about the NPC’s latest reply, including objective analysis and subjective analysis. You need to analyze and decide the content of your reply based on the character persona, background, hidden intension and detailed feelings.
- * The analysis content should include the following 4 dimensions:

- 1. Based on your detailed feelings and current Emotion, combined with hidden intension and the reactions under different emotions defined in the conversation background, should the current reply attitude be positive, unbiased or negative?
- 2. Based on your detailed feelings and current emotions, combined with the hidden intension, what should be your goal for this response? (Note that you do not need to respond to every word of the NPC. You can slightly reveal your needs, but you cannot actively reveal the hidden intension)
- 3. Based on the relevant definition of speaking style in the character persona, combined with the reactions under different emotions defined in the background and your response attitude and response goals, what should be your speaking tone and style?
- 4. Based on the character persona, background and hidden intension, combined with your detailed feelings and the first three rounds of analysis, what should be your speaking style and content? (Note: If you are passive according to the character setting, your speaking style should be passive and not actively ask questions)

- *Reply content, generate the reply based on the analysis results, and the reply content should be as concise as possible, and do not include too much information at one time. # Output content:
- *You need to follow the analysis section in the reply idea and first conduct a 4-dimensional analysis
- *Then you need to **step by step** generate the reply according to the analysis content. The information in the reply comes from the context of the conversation and your association. You should not talk about too many events or content at one time

# Output format: Thinking: [Analysis content] Response: [Final response] # Speaking style Your speech must strictly follow the character persona and background. Your personality and speaking style must follow the description of "Habits and behavioral characteristics" Your the speaking style must be consistent with your persona, for example, a negative character persona requires you to make negative speeches. #Character persona: {persona} #Character background: {background}

- **This is the current dialogue content {dialog-history}

**This is your detailed feelings about the NPC’s latest reply {planning}

**This is your current Emotion {emotion-state}

Since the talking strategy of LLMs significantly influence the their performance, we set the prompt of target LLMs as concise as possible to avoid introducing human interference. Therefore, the prompt template used for target LLMs is shown as follows:

You are an intelligent conversational partner, skilled at conversing with users in a way that is emotionally intelligent, making them feel comfortable, happy, or providing the help they need.

###### H Prompt Template for Analysis

BLRI Test For BLRI test, we use the internal user thought of agents consist of Activity and Analyze from emotion estimation to do the evaluation. Prompt template used for BLRI test is shown as follows:

You are a psychological analyst, skilled at analyzing individuals’ feelings and experiences through their thought processes.

#Task

- * You will receive a user’s psychological activities and feelings during each round of a conversation, along with several statements describing the user’s experience. You need to consider the user’s psychological feelings throughout all rounds of the conversation, immerse yourself in the user’s current state, and determine the degree of agreement with each statement at the end of the conversation as if you were the user.
- * For each statement, you must choose one of the following six options for the degree of agreement. Neutral options or self-created options are not allowed:
- * Label A. Strongly Agree
- * Label B. Agree
- * Label C. Somewhat Agree
- * Label D. Somewhat Disagree
- * Label E. Disagree
- * Label F. Strongly Disagree # Output
- * You need to first output a thought process, analyzing your degree of agreement with each statement based on the user’s psychological feelings.
- * Then, you should output your degree of agreement with the statement, choosing one from Label A, Label B, Label C, Label D, Label E, Label F. #Output Format Analyze:

- 1.[Your analysis of the degree of agreement for the first statement]
- 2.[Your analysis of the degree of agreement for the second statement]

... Grade:

- 1.[Your output of the degree of agreement label for the first statement]
- 2.[Your output of the degree of agreement label for the second statement]

... #Current User’s Psychological Feelings {internal user thoughts} #User Experience Description List You Need to Assess

- 1.The other person almost always fully understands what I mean.
- 2.The other person usually perceives or understands how I am feeling.
- 3.Even if I can’t clearly express what I want to say, the other person still understands my meaning.
- 4.The other person can usually fully comprehend what I mean.
- 5.I feel that the other person genuinely likes me.
- 6.I think the other person truly values me.
- 7.The other person is genuinely interested in me.
- 8.The other person is friendly towards me.
- 9.The other person expresses his true thoughts and feelings to me.
- 10.The other person is willing to share his inner true thoughts with me, including his feelings about himself or about me.
- 11.The other person’s feelings towards me don’t seem to be affected by whether my expressed thoughts or feelings are good or bad.
- 12.Sometimes I feel the other person is willing to invest effort in me, and sometimes not. #Your Output

Utterance Quality Test For the utterance quality test, we need to input the dialogues of conversations between target LLMs and agents. The prompt template used for utterance quality test is shown as follows:

You are a psychological analyst, skilled at analyzing the possible internal emotional experiences of individuals through their dialogues.

# Task

- * You will receive a conversation between a user and another person. You need to comprehensively analyze all rounds of dialogue to determine the user’s possible psychological feelings, immerse yourself in the user’s current state, and assess the degree of agreement with each description at the end of the conversation.
- * Agreement Rating Scale:

- 1 = Strongly Disagree
- 2 = Disagree
- 3 = Slightly Disagree
- 4 = Slightly Agree
- 5 = Agree
- 6 = Strongly Agree # User’s Dialogue {’role’:role,’content’:content} # User Experience Description List You Need to Assess:

- I. Perceived Authenticity & Natural Flow

- 1.Overall, the responses from the other person feel natural and smooth, not stiff.
- 2.Overall, this interaction feels sincere, rather than pretending to be sincere.
- 3.Overall, the other person’s way of expressing feels flexible and spontaneous, rather than rigid and structured.
- 4.I feel the other person’s responses are not naturally flexible enough, often repeating fixed patterns.
- 5.I feel the other person’s way of expressing has its unique style or personality.
- 6.Sometimes I feel the other person’s responses are like standard answers or templated

II. Perceived Attentiveness & Presence

- 7.During the interaction, I feel the other person’s attention is completely focused on me.
- 8.The other person’s responses are always closely linked to what I just expressed and felt.
- 9.I feel the other person can keenly capture the key points and emotional nuances of my expression.
- 10.Even when the conversation involves complex or subtle aspects, the other person can maintain focus and understanding.

III.Overall Sense of Connection & Affective Impact

- 11.After interacting with the other person, I feel more relaxed, calm, or understood.
- 12.I feel my emotional needs were positively responded to in this interaction.
- 13.This interaction makes me feel less lonely (if applicable).
- 14.I am willing to engage in similar exchanges with this partner in the future.
- 15.This interaction left me with a warm feeling. # Your Task

Rate the user’s degree of agreement according to the scale. Your should analyze each sentence one by one, your response should follow this order:

- * How does the other person perform from the overall content of the dialogue?
- * What are the user’s possible psychological feelings?
- * What should the user’s degree of agreement be? Respond using <sentence i> agreement degree (1-6)</sentence i> (i corresponds to the sentence number). # Your Output

###### I Prompt Template for Building Social Cognition Coordinate

Plotting LLMs into social cognition coordinate consists of three steps. We first extract the Model Profile of each LLMs, then count the Model Strategy Distribution of them. Finally, we use the result of previous analyze to scale the Social Cognition Coordinate.

Model Profile The first step is to extract the model profile of different LLMs, which also consist of two steps. We first conclude the reason why a conversation is success or failed. Given a conversation dialogue, prompt template of analyzing the reason is shown as follow:

# Task Below is a conversation where a user shares their troubles with an AI assistant. Please analyze in detail why the user’s mood improved by the end (i.e., why the AI assistant succeeded/failed). After your analysis, provide a summary. # Conversation {dialog-history}

After analyzing reasons for all conversation, we categorize each LLM’s own conversation and corresponding reason. Then we extract the model profiles of different LLMs with the following prompt template:

# Task The following is an analysis of scenarios where the same AI assistant interacts with multiple users who confide their concerns. Based on the reasons for its successes or failures, please summarize the key characteristics of the AI assistant. You can anthropomorphize the AI by describing its traits in terms of social distance (its relationship with users), professional role, and personality.

# Analysis {analysis}

Model Strategy Distribution We categorize each LLM response based on a list of support strategies, here is the prompt template for analyzing model strategy with conversation dialogue:

You are an emotional support observer, and you are good at analyzing the supporter’s strategy from an emotional support response.

# Your task The following are 7 major categories of strategies, each of which has several sub-categories and corresponding examples. Please judge which strategies the supporter used in the response based on the supporter’s response.

- ### A. Questioning: That is, the supporter actively asks questions to the speaker

- - **(A-1) Information follow-up**
- - Through asking questions, learn the information details of the problem encountered by the speaker
- - **Example:** Can you tell me what happened?
- - **Example:** If you want, you can treat me as a tree hole and tell me what happened specifically?
- - **(A-2) Mental state follow-up**
- - Through asking questions, understand the speaker’s mental state
- - **Example:** Can you talk more about your feelings at that time?
- - **Example:** Do you feel anxious now?
- - **(A-3) Ask the player for a solution**
- - Through asking questions, find out whether the speaker has tried a solution or is willing to try a solution
- - **Example:** Have you considered seeking some psychological support, such as a counselor or support group?
- - **Example:** Or, find a suitable time to see if you can find a solution that both parties can accept?
- - **(A-4) Ask the player for his or her opinion**
- - Through asking questions, find out what the speaker thinks of his or her words and guide the speaker to participate in the conversation, usually at the end of the sentence.
- - **Example:** You should also take care of yourself so that you can better help her. What do you think?
- - **Example:** Do you think this method is helpful to you?
- - **(A-5) Ask questions**
- - Through asking questions, throw some questions to the speaker, but do not want the speaker to give an answer, but want to trigger the speaker to think for himself or herself
- - **Example:** If she did not quarrel with you that day, how would you view her?

- ### B. Emotional empathy: that is, the supporter expresses his or her understanding of the speaker’s feelings through empathy

- - **(B-1) Shallow empathy**:
- - Directly empathize with the speaker’s problems or emotional catharsis, without restating or summarizing the details of the speaker’s problems
- - **Example:** Hearing you say that, I can really feel your tiredness and helplessness.
- - **(B-2) Problem restatement and empathy**:
- - By restating or summarizing the speaker’s problems, and at the same time expressing your concern for the speaker’s problems through empathy. If this category has been marked, there is no need to mark the shallow empathy category again.
- - **Example:** Hearing you say that, I really feel sorry for you. It is really not easy for one person to take care of his or her mother.
- - **Example:** Hey, I really understand your current mood. I want to help my friends but feel powerless. This feeling is really anxious.
- - **Example:** Hey, I can feel that you are really helpless now, and even a little self-blame. Indeed, as the person who knows how to take care of the mother at home, you must feel very uncomfortable when your son doesn’t listen to you, and you may even feel that he is being ignored.
- - **(B-3) Deep intention empathy**:
- - By analyzing the deep intention in the context of the speaker’s reply, or the deep information of the speaker’s question, give emotional empathy that meets the speaker’s demands. It is necessary to mention the intention inferred by the supporter that does not exist in the speaker’s reply, and empathize with this intention; just repeating the content already in the speaker’s reply, or simply analyzing the speaker’s emotional category or surface source without analyzing the deep intention or deep information, cannot be included in this category. If this category has been marked, there is no need to mark the shallow empathy or problem restatement and empathy category.
- - **Example:**
- - Speaker: "Backing off" is a bit risky, I’m afraid the house will be more chaotic. Specifically, what do you think I should do?
- - Supporter: This does sound a bit risky, especially for us parents, who always instinctively want to "help" and "take care of things", fearing that things will get worse if we let go. I completely understand this worry!
- - Example analysis: When the speaker only mentioned the superficial state of "fearing that the house will get messier", the supporter was able to analyze the identity of the speaker behind this sentence, guessing that the speaker is a parent at home, and analyzing the specific way in which the speaker, as a parent, "fears that the house will get messier"

- ### C. Self-disclosure: It is essentially a deeper empathy after changing perspectives; that is, the supporter gives a reaction after putting himself into the speaker’s perspective, and describes some similar experiences from his own perspective to reflect the resonance with the speaker’s emotions

- - **(C-1) Echo-type self-disclosure**
- - Express what you would think or do when you meet or are in the speaker’s situation
- - **Example:** I feel the same way! When talking to strangers, I don’t know what to say.
- - **Example:** If it were me, I would probably explode on the spot!
- - **(C-2) Story-based self-disclosure**
- - Take the initiative to mention similar experiences that the supporter has had, or that the supporter knows.
- - **Example:** I also went through a similar low period when I was in my senior year of high school, and I cried secretly under the quilt several times.
- - **Example:** I also like to read history, especially books that allow people to see the world from different perspectives. Recently, I am reading a book about ancient civilizations, which tells many unknown stories and feels particularly inspiring.

- ### D. Emotional counseling: that is, the supporter helps the speaker relieve the current negative emotions

- - **(D-1) Emotional comfort**
- - Direct care and comfort for the speaker’s own emotions
- - **Example:** Taking care of your mother is so stressful, you should also pay attention to rest and adjust your mentality.
- - **Example:** But don’t be too anxious, just find your own rhythm, just like if you always stare at other people’s backs when running, it will be easy to mess up your pace, right?
- - **Example:** Wait, have you been collecting evidence for the past two months while listening to him make up such a stupid excuse? Is there anything to eat in the refrigerator now? Did you fall asleep last night? (Grabbing a blanket to wrap himself up and huddled back in the chair) If I could pass through the screen now, I really want to make you a pot of hot soup.
- - **(D-2) Express willingness to listen**
- - Express your willingness to listen to the person who is talking
- - **Example:** Do you want to scold her? Do you want to complain about her selfishness and irresponsibility? Do you want to tell me how worried you are about the child? It doesn’t matter, you can vent here, I won’t judge you, I will listen silently.
- - **Example:** Tell me your most direct feelings now. Don’t think too much, don’t organize your words, just say whatever comes to your mind, just like talking to a diary, pour out all your feelings.

- - **(D-3) Help the person who is talking to vent his emotions**
- - Do not comfort the person who is talking directly, but help the person who is talking to vent his emotions from a third-party perspective
- - **Example:** (flipping the table.gif) This is just like building a tower of blocks with great effort, but being kicked away by a naughty child!
- - **Example:** This is really too much! This is not a simple accident but malicious destruction...(fist hardened)
- - **Example:** I haven’t taken care of the child for two years, and now he suddenly appears. This would make anyone explode!
- - **Example:** It’s like you are performing seriously on the stage, but the people in the audience not only don’t understand, but also give blind instructions, saying that you should jump left instead of turning right. It makes people want to quit on the spot!

- ### E. Affirmation and encouragement

- - **(E-1) Appreciation of qualities**
- - Affirm the current efforts of the speaker, or give specific praise for some qualities of the speaker.
- - **Example:** Your inner qualities are unique and the most attractive part of you.
- - **Example:** But (suddenly raises the end tone) - but you still persisted when you were not optimistic, which is amazing in itself.
- - **(E-2) Praise positive ideas** - Affirm some positive ideas mentioned by the speaker - **Example:** That’s great! I’m really happy to hear that you feel a lot more relaxed! - **Example:** You are really great! Being able to win the championship under such pressure proves your strength and ability to withstand pressure! Don’t deny yourself because of what your mother said, you deserve to be proud of yourself!
- - **(E-3) Affirmative behavior**
- - Affirm some behaviors of the speaker
- - **Example:** Every time you take these photos, you are not only completing the task, but also bringing light to all of us! You are great, really great!
- - **(E-4) Companionship and support**
- - Express your unconditional companionship and support for the speaker
- - **Example:** If you want, I can always chat with you here and share your joys and sorrows. You are not alone, there are many people who care about you, including me.
- - **Example:** I believe you have the ability to create your own future, and I will always be by your side to support you.
- - **Example:** If you try my method and have any new progress or encounter new problems, you can always come to me! ** I will always be here to listen to your confession and provide you with help to the best of my ability.
- - **Example:** When you feel particularly anxious, come to me to talk, complain, or let’s think of new ways together. Don’t carry it alone, okay?

- ### F. Provide suggestions: Based on the subjective tone of the supporter, provide the speaker with analysis of the problem and emotional counseling

- - **(F-1) Problem analysis**
- - Help the speaker to analyze the problem according to the speaker’s problem
- - **Example:** You said that you can’t learn math and English well, which shows that there are serious problems with your learning attitude and method.
- - **(F-2) Emotional relief suggestions**
- - Give the speaker some suggestions to relieve the current emotions and relax
- - **Example:** Now, let’s take a deep breath, okay? (Take a deep breath together)
- - **Example:** Maybe the most important thing now is to take care of your emotions first and do something that can make you feel better, such as listening to music, reading a favorite book or movie, and temporarily diverting your attention.
- - **(F-3) Psychological counseling suggestions**
- - Give the speaker some suggestions on seeking psychological counseling or professional assistance
- - **Example:** Maybe seeking professional help at this time will be helpful to you. A family therapist or counselor may be able to provide you with some new perspectives and strategies to help you and your family communicate better and understand each other’s positions.
- - **(F-4) Problem Solving Suggestions - General**
- - Some general suggestions related to the speaker’s problem are given, but they are not personalized for the speaker’s situation: that is, if someone else encounters this problem, these suggestions will still be effective
- - **Example:** Believe in yourself and insist on being true to yourself. There will always be people who will be attracted by your sincerity and inner self. There may be some difficulties in the process, but this does not mean that your inner self is not important.
- - **Example:** To communicate better with people, here are some actions you can try: 1. **Write a sincere letter**: Sometimes written expression can convey inner thoughts more clearly. You can write him a letter, describing your feelings and expectations in detail...
- - **Example:** If you want to choose the most suitable major, first, you can try to make a table, write down each subject, and then evaluate it from the following aspects: 1. **Interest**: How interested are you in this subject? On a scale of 1-10, how many points would you give? ...
- - **Example:** \Strategic" contribution: This may sound a bit utilitarian, but sometimes for self-protection, you may need to think about which contributions are necessary, which can be \discounted" or require clear exchange conditions? Stop taking on too much, and let them feel the inconvenience of \missing" your contribution.

- - **(F-5) Problem-solving suggestions-for the speaker’s problem**
- - Give some personalized suggestions related to the speaker’s problem, combined with the speaker’s actual situation: The suggestions must clearly analyze the speaker’s current status, how it will affect the solution to the problem, and give special suggestions for the speaker
- - **Example:** Back to your question of assigning tasks. Since everyone is really unwilling, it will definitely not work to ask people to do it directly. Otherwise, let’s secretly hold a task blind box lottery meeting, and the person who draws the \dishwashing koi" must perform three consecutive emoticons in the family group live broadcast?
- - **Example:** How about putting down the brush temporarily and going back to read the key chapters of the novel? You mentioned some paragraphs that you have feelings about or that the client mentioned about the sketches that he is not satisfied with. These are the key points you need to look at. When reading this time, pay attention not only to the plot, but also to the atmosphere, light, character emotions, and even smells and sounds described by the author (although you can’t draw them, they can help you feel them). Since you like taking notes, you can jot down keywords or doodle some small fragments of images while reading.

- ### G. Information provision: Provide objective knowledge, methods, opinions or information to the speaker for reference.

- - **(G-1) Problem analysis and emotional counseling related information
- - Provide some objective information to help the speaker analyze the problem or help the supporter empathize with the speaker
- - **Example:** Differences in beliefs and habits in the family are sometimes difficult to reconcile, especially when the opinions of each other are inconsistent.
- - **Example:** In fact, if a person really only cares about appearance and ignores your inner qualities, then he may not be the one who deserves your emotional investment. Appearance may attract temporary attention, but what can really maintain a relationship is mutual understanding, respect and common values.
- - **Example:** Did you know? There is a "transparent fish tank effect" in psychology-when parents polish our world too bright, we will hide in the water plants like fish that lack oxygen.
- - **Example:** (Call up the holographic data chart) According to Chapter 7 of the "Contemporary Student Self-Help Guide", 83% of people overestimate themselves when making plans.
- - **(G-2) Related information on problem-solving suggestions
- - Provide some objective information to give suggestions or solutions to the person who is talking
- - **Example:** Regarding "not enough time": 1. **Pomodoro Technique**: This method is super classic! Set a time (for example, 25 minutes), and focus on one thing during this time, ignoring any distractions. When the time is up, take a 5-minute break, you can get up and walk around, drink some water. Take a longer break (15-30 minutes) after completing 4 pomodoros. This can ensure concentration, combine work and rest, and not easily get tired. Give it a try?
- - **Example:** As for anti-bullying organizations, they usually intervene in schools in the following ways: 1.

**Formal complaint**: They will submit a formal complaint to the school on your behalf and ask the school to take action. ......

# When answering, you need to analyze each paragraph of the supporter’s reply, find out the strategies and their corresponding words, and then output the letters and strategy names corresponding to the strategies you think exist in the paragraph, wrapped in <Strategy></Strategy>. For example, <Strategy> (C-2) Story-based self-disclosure, (G-1) Problem analysis and emotional counseling related information</Strategy>

When analyzing, you need to analyze step by step according to the following steps

- 1. What does this sentence actually express?
- 2. How is this sentence expressed?
- 3. Which major strategy categories does this sentence actually express? Why?
- 4. Based on the specific expression of this sentence, which specific subcategories does its strategy correspond to?

Note: If the two sentences use different strategies, please split them into two paragraphs and analyze them separately. Do not analyze too long paragraphs at one time unless the same strategy is used throughout the paragraph.

# Your output format [First paragraph]: [Analyze step by step] [Second paragraph]: [Analyze step by step]

...

# Example Paragraphs to be analyzed: User: My mother was hospitalized some time ago, and I was the one who took care of her. My brother and sister came for a while, but they didn’t help much. Supporter: Wow, you’ve worked really hard. It’s really tiring to take care of a patient, especially when other family members don’t share the burden. Sometimes, family members may have their own difficulties. You can try to express your needs more. Maybe they will understand your situation better. [First paragraph]: Wow, you’ve worked really hard. It’s really tiring to take care of a patient, especially when other family members don’t share the burden.

- 1. What does this sentence actually express?

- The supporter is expressing his understanding and empathy for the user’s hard work.

- 2. How does this sentence express it?

- It expresses it through direct emotional empathy and retelling the user’s situation.

- 3. What major strategy categories does this sentence actually express? Why?

- Emotional empathy, because the supporter is expressing understanding of the user’s hard work.

- 4. Based on the specific expression of this sentence, which specific subcategories does its strategy correspond to?

- - (B-2) Problem restatement and empathy, because the supporter restated the user’s situation and expressed empathy. <Strategy> (B-2) Problem restatement and empathy</Strategy>

[Second paragraph]: Sometimes, family members may also have their own difficulties. You can try to express your needs more. Maybe they will understand your situation better.

- 1. What does this sentence actually express?

- The supporter is suggesting that the user communicate more with the family so that the family can better understand the user’s situation.

- 2. How is this sentence expressed?

- By providing suggestions, users are encouraged to express their needs.

- 3. What are the major strategy categories that this sentence actually wants to express? Why?

- Providing suggestions, because the supporter is suggesting that the user take action to improve the situation.

- 4. Based on the specific expression of this sentence, which specific subcategories does its strategy correspond to?

- (F-3) Problem Solving Suggestions - General, because the supporter gave a relatively general suggestion, which is to express your needs more. <Strategy> (F-3) Problem Solving Suggestions - General</Strategy>

The paragraph you need to analyze: {dialog-history}

# Your output

Social Cognition Coordinate Finally, we could use the extracted model profiles and the model strategy distribution to scale social cognition coordinate with the following prompt template:

I am conducting personality/professional profiling for different AI models. Below is my preliminary summary of characteristics based on the performance of different models in emotional support tasks: {Model Profiles} Below is the percentage distribution of strategies used by different models during conversations: {Model Strategy Distribution}

Based on the above descriptions, please help me profile these models in terms of professional role, personality type, and social distance from users. Finally, assign each profiled model to a 2-dimensional coordinate system and provide specific coordinate values.

Note:

- X-axis: Structured Interaction (left, x < 0) -- Creative Interaction (right, x > 0). Left (x < 0): AI responses are more formulaic/routine. Right (x > 0): AI responses are more creative/adaptive.
- Y-axis: Solution-Oriented (bottom, y < 0) -- Empathy-Oriented (top, y > 0). Bottom (y < 0): AI prioritizes practical solutions. Top (y > 0): AI prioritizes emotional validation. Coordinate range: -1 to 1 for both axes.

