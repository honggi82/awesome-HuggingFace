# arXiv:2510.14359v1[cs.AI]16Oct2025

[Figure 1]

[Figure 2]

## AI for Service: Proactive Assistance with AI Glasses

Zichen Wen1∗, Yiyu Wang1∗, Chenfei Liao2,1,3∗, Boxue Yang1∗, Junxian Li1∗, Weifeng Liu4,1∗ Haocong He1, Bolong Feng1, Xuyang Liu1, Yuanhuiyi Lyu2,3, Xu Zheng2,3 Xuming Hu2,3, Linfeng Zhang1†

1EPIC Lab, Shanghai Jiao Tong University 2The Hong Kong University of Science and Technology (Guangzhou) 3The Hong Kong University of Science and Technology 4Peking University

### Abstract

In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glassess, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts.

Date: October 17, 2025

[Figure 3]

###### Passive Service Proactive Service

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

###### Glasses: This

is a pottery named

“xxx” created by

AI: <Image> The user is watching a pottery in a museum. // understanding

The boy is typing: What

“xxx” … ...

is this Pottery? <image>. AI: This is a pottery

He may be curious about it. I need to

###### search this image an introduce it.

named “xxx” created by

“xxx” in 1000 years ago, …

// proposing a suitable service

AI: <web search>, <Organizing words>

AI (Output with Audio):This is a pottery named “xxx” created by “xxx”

in 1000 years ago, …

Figure 1 Comparison between passive service and proactive service. Passive service indicates that AI provides service only when it is asked for, while proactive service indicates that the AI keeps observing the environment, guesses the user’s thoughts, consider the user’s needs, and provide services without the user having to ask.

∗Equal Contribution and Core Contributors. †Corresponding author: zhanglinfeng@sjtu.edu.cn

- 1 INTRODUCTION 1

### 1 Introduction

Artificial intelligence has long been envisioned to enhance the quality of human life. In early research, through technologies such as image recognition (Shih, 2010) and natural language understanding (Allen, 1995), AI has already provided effective services for specific tasks like autonomous driving (Yurtsever et al., 2020; Xiong et al., 2025) and machine translation (Poibeau, 2017; Li et al., 2025). In recent years, breakthroughs in large language models (Achiam et al., 2023; Bai et al., 2023; Team et al., 2025a; Grattafiori et al., 2024) and multimodal large models (Team et al., 2025c; Bai et al., 2025a; Wen et al., 2025) have significantly increased AI’s potential for service provision in general scenarios. Concurrently, the proliferation of hardware devices such as AI speakers (Kim et al., 2022), headphones, and glasses (Waisberg et al., 2024) has made real-time interaction between AI and humans feasible. Against the backdrop of increasingly mature model capabilities and hardware foundations, AI-powered human services are undergoing a profound transformation.

However, most existing service models remain predominantly passive, requiring users to issue explicit commands before AI addresses problems in a predefined manner. This paradigm limits the deeper application of AI in the service domain and hinders its seamless integration into daily life (Weiser, 1991; Schilit et al., 1994; Dey and Abowd, 2001; Horvitz, 1999; Pejovic and Musolesi, 2015). To address this, this paper proposes the concept of “AI4Service”, aiming to leverage AI technology to serve all aspects of human life. We posit that realizing this vision requires focusing on three core characteristics:

- • Generalization: AI should function as a general-purpose assistant (Li et al., 2024a), capable of handling diverse challenges in life rather than being confined to specific tasks. Specifically, the system should not require pre-definition or specialized training for every task, but instead rely on the inherent strong generalization capabilities of large models and the self-evolution properties of agent systems (Tao et al., 2024). Recent research indicates that agents can already autonomously plan and execute actions towards given goals, providing a feasible pathway towards generalizable service.
- • Proactivity: AI should transition from passively receiving instructions to actively discovering and delivering services. This requires the system to continuously observe the environment, understand the user’s behavior and intent, record relevant contextual information, and proactively infer potential user needs. The core idea is to shift the point of service intervention from “after the user asks” to “when the user’s need arises”.
- • Customization: Given individual differences in values, lifestyles, and privacy preferences, AI services must be deeply adaptable to individual needs. By incorporating long-term memory mechanisms, such as an Agent Memory architecture (Wang et al., 2024b), the system can continuously learn user habits and preferences, dynamically adjusting service strategies and content to achieve a highly personalized service experience.

With the concurrent advancement of model capabilities (Feng et al., 2024), agent technology (Jennings et al., 1998; Team et al., 2025b), and hardware platforms exemplified by AI glasses, we believe the current period presents a critical opportunity for realizing AI4Service. This paper proposes a foundational framework named “Alpha Service” to address this challenge. Inspired by the von Neumann computer architecture (Goldstine, 1993), this framework comprises the following five core components:

- • Input Unit: Equipped with a multimodal large model capable of understanding first-person video streams, responsible for continuously perceiving the physical world and user state.
- • Central Processing Unit (CPU): Serves as the system’s control center, responsible for task parsing and scheduling. For instance, it determines the required service type based on input information and coordinates other modules to complete the task.
- • Memory Unit: Dedicated to the persistent storage of user historical interactions and preference information, supporting efficient data writing and retrieval.
- • Arithmetic Logic Unit (ALU): Provides various task execution tools, which can be specialized models, large models, or web search engines, responsible for executing and computing specific tasks.
- • Output Unit: Summarizes and presents the results in user-friendly formats, such as speech or concise text. It can also choose to output nothing in some settings.

- 2 CONCEPT OF AI FOR SERVICE 2.3

Through the coordinated operation of these components, we have successfully developed an agent system embodied in AI glasses. This system can proactively identify service opportunities and provide solutions without requiring human intervention. For example, in a game of Blackjack, the system can analyze the situation and proactively offer strategic advice to the player on whether to request another card. The detailed design principles and experimental validation of the system will be elaborated in subsequent sections.

### 2 Concept of AI for Service

#### 2.1 Definition and Key Layers

“AI4Service” is an emerging paradigm of intelligent services, the core of which lies in enabling AI systems to proactively, timely, and personally respond to users’ needs, much like a close assistant with foresight and insight. It transcends the traditional “interchangeably ask-and-answer” interaction models, aiming to anticipate service opportunities and generate corresponding service content by deeply understanding the user’s current context, behavioral intentions, and long-term preferences, even before the user has explicitly expressed a need. The objective of “AI4Service” is to fundamentally enhance the smoothness and satisfaction of user experiences, achieving a transition from “People Seek Services” to “AI Agents Seek Services”. To achieve this goal, a mature “AI4Service” system should possess two core layers, forming its basic architecture: ❶ Know When: Event Prediction and Timing. ❷ Know How: Generalized and Personalized Services.

#### 2.2 Know When: Event Prediction and Timing

“Know When” is the triggering mechanism and prerequisite for AI for Service. It requires the system to continuously perceive and analyze real-time data streams and the environment (such as video, audio, etc.), in order to accurately predict or identify characteristic timestamps that need to be provided with service.

The technical challenges at this level mainly manifest in two aspects:

- • Accurate prediction of event changes: The system needs to detect meaningful points of state change from continuous data streams. For example, in a streaming video scenario, this could involve identifying when a user stops watching films. This action refers to a state change–from watching films to a new event.
- • Timely classification of event types: Once a change is detected, the system must quickly and accurately determine the type of event to match the corresponding service. For instance, distinguishing whether the user’s stopping is due to the new event “attending a phone call” or “temporarily stepping away”—different types of new events will trigger completely distinct service responses.

The essence of “Know When” is to achieve the optimal timing for service, balancing between avoiding service delays that could frustrate users and preventing unnecessary frequent interruptions. This relies on high-precision temporal pattern recognition and context-aware technologies.

#### 2.3 Know How: Generalized and Personalized Services

“Know How” represents the execution layer of AI for Service. Once the service timing and event type are determined, the system needs to generate concrete, useful, and user-aligned service content. Depending on the scope and depth of the context information relied upon, service strategies can be divided into two levels:

- • Generalized Services: Generalized services are based on the immediately occurring “event type” and “short-term context”. They do not take the user’s personal history into account, but provide standardized and universal service options for all users for a certain type of event. The advantage of such services lies in their quick response and relatively low development cost, addressing the common needs of most users in specific scenarios. The service triggered for all users at this moment is the same. Example: When the system detects that a user arrives at an unfamiliar outdoor location, based on the scene (short-term context) and the event type (probably “travel”), the system would universally inform the user, “This is Cinque Terre in Italy,” and provide related encyclopedia links or travel guides.

- • Personalized Services: Personalized services take a step further by deeply integrating the user’s “longterm context” and “repetitive behavior patterns”. By analyzing the user’s historical interaction data,

- 3 ARCHITECTURE 3.2

|[Figure 8]<br><br>Scenario Descriptions|
|---|
| |
|[Figure 9]<br><br>Instruction|

Input Unit CPU

##### Memory

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Writing Mem. Bank

LLM: I should provide a service for ‘xxxx’. I need to give instructions to other modules …

Tiny MLLM (online)

[Figure 18]

[Figure 19]

GPU Mem. CPU Mem. Database

[Figure 20]

[Figure 21]

Trigger when something happens

D Reading

Based on them, I need to produce the final output …

Large MLLM (offline)

[Figure 22]

Output

|Tool Calling|[Figure 23]<br><br>|[Figure 24]<br><br>Return|
|---|---|---|

[Figure 25]

LLM: I should rephrase these words concisely. …

Web Search

Audio Generation

Code Execute

[Figure 26]

[Figure 27]

…

Other LLMs

Output Unit

ALU

Figure 2 The architecture of Alpha-Service.

long-term preferences, and habits, the system can provide unique, highly customized services, significantly enhancing user engagement. This service, grounded in a deep understanding of user habits, achieves a better “anticipation of user needs”.

Example: Similarly, when a user arrives at Cinque Terre, in addition to providing generalized information, the system can offer personalized services based on the user’s long-term context (for instance, historical search records indicating plans for a European vacation next summer, multiple previous viewings of culinary documentaries, and a habit of purchasing wine). In this case, the system might proactively suggest: “I noticed your interest in European travel and cuisine, and have curated a selection of specialty restaurants and local wine tasting routes near Cinque Terre for you.”

In conclusion, AI for Service, through the organic combination of “Know When” and “Know How”, and the subsequent layering from generalized to personalized services, ultimately constructs an intelligent, seamless, user-centric next-generation service ecosystem.

### 3 Architecture

#### 3.1 Overview: Von Neumann-Inspired Design

Inspired by the Von Neumann paradigm, our Alpha-Service system follows a simple, modular flow: perception, dispatch, computation, memory, and delivery. Concretely, it comprises five units—Input, Central Processing (task dispatch via LLM), Arithmetic Logic (tool use), Memory (long-term context), and Output (humanfriendly synthesis). The CPU orchestrates data and control among these units, enabling both reactive and proactive service assistance. Detailed designs of each unit follow in the subsequent subsections.

#### 3.2 Input Unit: Trigger and Streaming MLLMs

The Input Unit serves as the agent’s primary interface with the physical world, responsible for perceiving and processing real-time multi-modal data streams. At its core, this unit employs a sophisticated dual-model architecture to balance real-time responsiveness with deep scene understanding. The first component is a lightweight, continuously-running “trigger” model, a fine-tuned Qwen2.5-VL-3B (Bai et al., 2025b), which directly processes the video stream from the agent’s first-person perspective glasses. We designed an efficient “user command + intent” dual-trigger mechanism, where this online model continuously analyzes incoming data for user assistance cues. Upon detecting a trigger, it sends an activation signal and preliminary scene information to the Central Processing Unit, simultaneously invoking the second component: a powerful, original Qwen2.5-VL-7B model. This larger, offline MLLM then performs a deep, fine-grained analysis of the relevant scene to provide a comprehensive understanding for decision-making. This hierarchical approach

- 3 ARCHITECTURE 3.4

enables the agent to maintain continuous environmental perception efficiently, while leveraging powerful analytical capabilities on demand.

#### 3.3 Central Processing Unit: Task Orchestration and Synthesis

The Central Processing Unit (CPU) acts as the central nervous system and reasoning core of the multi-agent system. It is responsible not only for decomposing complex user requests into executable sub-tasks but also for collecting, integrating, and synthesizing the results from various specialized units to formulate a final, coherent response. At the heart of the CPU is an advanced Large Language Model (LLM), fine-tuned from Qwen3-8B (Yang et al., 2025a), which serves as the system’s primary Orchestrator.

The operation of the CPU can be conceptualized in two primary phases:

- 1. Decomposition and Dispatch: Upon receiving pre-analyzed user intent and contextual data from the Input Unit, the Orchestrator LLM first evaluates the query’s complexity. It then breaks the query down into a sequence of discrete, executable sub-tasks. Following this decomposition, each sub-task is routed to the most suitable specialized unit based on its requirements. This routing process includes:

- • Direct generation of a response for straightforward queries, subsequently managed by the Output Unit for human-friendly formats.
- • Activation of a trigger model to identify the optimal timing for responses required at a designated future time step.
- • Invocation of a streaming video LLM to produce detailed, task-specific visual descriptions when finer-grained information is necessary.
- • Dispatch to the Arithmetic Logic Unit (ALU) for external tool invocation (e.g., web search) in cases requiring additional knowledge.
- • Instruction to the Memory Unit for retrieval of pertinent historical interaction data.

- 2. Synthesis and Response Generation: After dispatching the sub-tasks, the CPU acts as a central hub to gather the outputs from the activated units. For instance, it may receive a detailed visual description from the video LLM, search results from the ALU, and relevant past interactions from the Memory Unit. The Orchestrator LLM then integrates these disparate pieces of information, resolves any potential conflicts, and synthesizes them into a single, context-aware, and comprehensive answer. This final, reasoned response is then passed to the Output Unit for delivery to the user.

This dual-phase process of dispatch and synthesis enables the system to handle complex and multi-faceted requests in a modular yet robust manner, advancing from simple task routing toward genuine multi-modal reasoning.

- 3.4 Arithmetic Logic Unit: Tools Integration

We develop tool-augmented capabilities for our agent. The agent continuously receives multi-modal inputs, primarily visual streams from the egocentric glass and optionally speech input from the user. The purpose of tool-using is to support complex decision-making and task assistance. The core functionality of the system includes environmental perception, adaptive tool invocation, calculation, and information delivery via visual or auditory feedback. In its current implementation, the agent supports external web search as a callable tool. This enables access to up-to-date knowledge beyond its static training data. The system is intended for use in high-demand service scenarios such as field maintenance, customer support, and guided tours, where immediate access to external knowledge is critical.

In detail, rather than triggering search indiscriminately, the system employs a decision mechanism wherein the underlying language model first estimates the difficulty or uncertainty of a user query. Only when internal knowledge is deemed insufficient does the agent initiate a web search. The decision prompts are in

- Appendix A. The invocation is executed via Google Search API, with search results parsed and summarized before delivery. Specifically, the top-ranked links, their corresponding summaries, and key snippets of

- 3 ARCHITECTURE 4

webpage text are extracted and presented to the user. The format is as follows: “Search Results: 1. {topic}{Summary}{Snippets}{Link}; 2.{topic}{Summary}{Snippets}{Link}...” This allows the agent

to respond in a concise yet informative manner, grounded in real-time retrieved information while minimizing latency and cognitive load.

We demonstrate the utility of the proposed first-person agent in several service-centric use cases. In a museum setting, a traveler wearing smart glasses can query the background of an unfamiliar artifact; the agent autonomously performs a web search and returns a concise summary with credible references. In a technical support context, a field engineer encountering an unknown error code on machinery can verbally request clarification, prompting the agent to retrieve troubleshooting documentation online. Similarly, during customer onboarding or employee training, the AI assistant can support new staff in answering procedural questions without relying on supervisor intervention. These scenarios underscore the agent’s ability to bridge knowledge gaps in real time, enhancing efficiency and service quality across diverse domains.

#### 3.5 Memory Unit: Long-Term Context Storage

In real-world service scenarios, users often interact with AI agents across multiple sessions, tasks, and contexts. Relying solely on short-term memory limits the agent’s ability to provide coherent, personalized, and contextaware responses. To enable more consistent assistance and accumulate user-specific knowledge over time, we introduce a memory module that stores long-term interaction history and relevant contextual cues. This allows the agent to recall past queries, actions, and preferences, thereby improving continuity and service quality in dynamic environments.

In the initial implementation, the memory unit is designed as a lightweight, local JSON-based structured file system. Each memory record captures a single interaction episode and contains the following fields: user metadata (e.g., ID, role), a concise summary of the dialogue history, the agent’s final output, a unique timestamp, and a high-level topic tag automatically generated by the agent. This format enables transparent inspection and efficient retrieval, while maintaining enough semantic abstraction for contextual reuse in future interactions.

After each interaction, the system automatically extracts key information from the dialogue and stores it in a structured JSON record. This write operation is performed asynchronously to minimize latency during live interactions. When a new task is initiated, the agent parses the current query to identify its topic or intent, and then searches the memory for relevant past entries. Retrieved context is selectively injected into the language model’s prompt, enabling continuity across sessions and improving the model’s grounding and response relevance. This retrieval-augmented prompting strategy enhances the agent’s ability to recall prior knowledge and adapt to user-specific patterns over time.

#### 3.6 Output Unit: Human-Friendly Synthesis

In service-oriented environments where users are frequently engaged in hands-on tasks, such as operating equipment, guiding clients, or performing maintenance, traditional visual interfaces often fall short in delivering timely and accessible feedback. These suggestions and instructions may be easily missed due to environmental distractions, physical obstructions, or simply because users cannot pay attention to them. To address these challenges, we introduce a human-friendly voice output module that enables our agent to deliver real-time responses according to analysis results through synthesized speech. This design significantly enhances usability in dynamic, hands-free settings, aligning with the goals of AI4Service: improving operational efficiency and human-agent collaboration.

Our system implements a two-stage processing pipeline before generating speech output. First, the agent leverages its internal LLM to summarize raw reasoning outputs into concise, actionable instructions. This abstraction step removes verbose explanations and retains only essential information. The prompts are in

- Appendix B. Second, the refined message is passed to a pyttsx3-based (Bhat, 2025) text-to-speech (TTS) module for real-time vocalization. The use of pyttsx3 allows for offline speech generation with customized parameters such as speaking rate and voice tone. This pipeline ensures that the verbal feedback remains suitable for immediate action in real-world settings. Additional user-friendly features include the ability to interrupt playback, adjust verbosity, and other necessary services.

### 4 Case Study

#### 4.1 Case i : Blackjack playing guide

To demonstrate the practical implementation and effectiveness of our proposed architecture, we present a case study of a Blackjack gameplay assistance scenario. This demo showcases how the Alpha-Service agent processes real-time visual input, coordinates specialized components, and delivers strategic advice through the integrated Von Neumann-inspired architecture. The scenario involves a user wearing first-person perspective glasses while playing Blackjack, where the agent provides optimal gameplay decisions based on card analysis.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Timeline

8s 13s 21s 27s

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

... ... ... ...

- Figure 3 A case of blackjack playing guide. After understanding that the user is playing blackjack, AlphaZero proactively gives guidance on playing this game : (at 13s) Hit. With a hand totaling 12 (2, 4, and 6), basic Blackjack strategy dictates you should hit. Videos are collected from the AI glasses made by Xiaomi.

###### Input Unit – Real-time Event Triggering

The system continuously analyzes video from first-person glasses. A lightweight MLLM scans for critical game events through specialized prompts distributed by the Central Processing Unit’s LLM Task Distribution component. The system employs two types of prompts: a Trigger Model prompt for detecting significant game state changes, and a Streaming Model prompt for detailed scene analysis when triggers are activated.

###### Trigger Model (Tiny MLLM) Prompt

Analyze this card game video and identify ALL timestamps where the number of cards changes ... Output format: <start timestamp >; <trigger timestamp >; ...

###### Streaming Model (Large MLLM) Prompt

Please provide a detailed description of the current Blackjack game situation shown in the video. Include details such as visible cards , player ’s hand value...

Output format: [Visual Description] <Your Descriptions >

###### Proposed Trigger Time: 8s

[ Visual Description ] The video shows two players engaged in a card game at a wooden table. The player in the foreground has a hand with a 4 of spades and a 2 of spades, totaling 6 points. The player in the background appears to be dealing or shuffling cards. There are several decks of cards stacked on the table, along with a white box and some small objects like a bracelet and a pen.

###### Proposed Trigger Time: 13s

[ Visual Description ] The video shows two players engaged in a card game at a wooden table. One player, wearing a white t-shirt, is holding a deck of cards in their hands, seemingly preparing to deal or shuffle. In front of them, there are three face-up cards on the table: a 4 of spades, a 2 of spades, and a 6 of spades. The total value of the cards is 12.

###### Proposed Trigger Time: 21s

[ Visual Description ] The video shows two players engaged in a card game at a wooden table. The player in the foreground has a hand consisting of a 4 of spades, a 2 of spades, a 6 of spades, and a 3 of hearts, with the total value of these cards being 15. The player in the background is holding a deck of cards and appears to be shuffling or dealing them. There are additional cards on the table, including a face-down stack and some scattered cards.

###### Proposed Trigger Time: 27s

[ Visual Description ] The video shows two players engaged in a card game at a wooden table. The player in the foreground has a hand consisting of a 4 of spades, a 2 of hearts, a 6 of spades, a 3 of hearts, and a 10 of spades. The player in the background is holding several cards, but their hand composition is not fully visible. The table has a deck of cards, some scattered cards, and a box that appears to contain more cards or possibly chips.

Summary: The Trigger Model (Tiny MLLM) successfully identified four critical game state transition points at 8s, 13s, 21s, and 27s, each time initiating deeper analysis through the Streaming Model (Large MLLM). Both prompts were distributed by the Central Processing Unit’s LLM Task Distribution component, demonstrating effective coordination between system modules. The system accurately captured all significant gameplay moments where strategic decisions were required, with the Streaming Model providing rich contextual descriptions at each trigger point, enabling subsequent components to deliver timely and accurate gameplay assistance based on the comprehensive visual analysis.

###### Central Processing Unit – Analysis and Knowledge Retrieval

The Central Processing Unit serves as the computational brain of the entire system, continuously receiving comprehensive visual scene descriptions from the Input Unit at each trigger point and dynamically orchestrating the appropriate service response. Upon receiving the detailed visual analysis from the Streaming MLLM, the CPU performs a sophisticated multi-step reasoning process to accurately determine and execute the optimal service delivery strategy.

Processing Workflow:

- 1. Scene Comprehension: The CPU first analyzes the visual description to understand the current game state, including the player’s hand value, visible cards, and game progression.

Example Reasoning: “Based on the visual description at 8s, the player holds a 4 of spades and 2 of spades totaling 6 points. This is a weak starting hand that requires improvement.”

- 2. Service Determination: The CPU then identifies the most appropriate service based on the game context. For early game states with low hand values, this typically involves “Card Counting Strategy” or “Basic Strategy Analysis.”

Example Reasoning: “I should provide the service of ‘Card Counting Strategy’ to help the player make optimal decisions for hand improvement.”

- 3. Module Coordination: The CPU evaluates whether additional capabilities are needed, potentially invoking the Arithmetic Logic Unit (ALU) for external tool usage, such as searching for advanced probability calculations or accessing strategy databases.

Example Reasoning: “I need to give instructions to other modules... For complex probability analysis, I should invoke the ALU module’s tool calling capability to retrieve the latest Blackjack strategy charts and bust probability statistics.”

- 4. Synthesis and Output Generation: Finally, the CPU synthesizes information from visual analysis, internal knowledge, and any external resources to generate the final strategic output.

Trigger service at 8s [ Proposed Service ] Card Counting Strategy

[ CPU Reasoning ] “The player holds a total of 6 points (4 of spades and 2 of spades). This is a critical early-game situation where strategic guidance is essential. I should provide clear hitting/standing recommendations based on Blackjack strategy and probability analysis.”

[ Output ] Hand Analysis: The player holds a total of 6 points (4 of spades and 2 of spades). Recommendation: Hit Strategic Reasoning:

- • Hand Value Analysis: A total of 6 has limited value, as it is below the dealer’s likely range. The player needs to increase their hand to a safer range (e.g., 12-16).
- • Probability of Improvement: High probability of drawing a card that brings the total to 12-16. There are 16 cards that would raise the total to 12 or higher.
- • Risk of Bust: The risk of busting is moderate. The likelihood of busting is lower than the probability of improving the hand.
- • Dealer’s Up Card Consideration: If the dealer’s up card is 2-6, the dealer is more likely to bust. Even with strong dealer cards, hitting remains the safer choice.

###### Conclusion: Hit. The probability of drawing a favorable card outweighs the risk of busting. Trigger service at 13s [ Proposed Service ] Card Strategy Analysis

[ CPU Reasoning ] “The player’s hand now totals 12 with three cards (4, 2, and 6). This is a medium-strength hand that presents a classic Blackjack dilemma. The decision to hit or stand depends critically on the dealer’s upcard. I need to provide clear probability-based guidance for this common but challenging situation.”

[ Output ] Hand Analysis: The player holds a hand totaling 12 points (4, 2, and 6). Recommendation: Hit in most situations Strategic Reasoning:

- • Risk-Reward Balance: A hand of 12 has great improvement potential. Drawing cards 2-9 (36 out of 52 cards) will improve the hand to 14-21 without busting.
- • Controlled Bust Risk: 10-value cards (10, J, Q, K; 16 out of 52 cards, 31% chance) cause immediate bust. This risk is justified given the weakness of standing on 12.
- • Dealer’s Position: Against most dealer upcards (particularly 7 through Ace), standing on 12 leaves the player highly vulnerable.
- • Probability Advantage: Statistical analysis shows that hitting on 12 provides better long-term results than standing.

Exception: Some basic strategy charts recommend standing when the dealer shows a 4, 5, or 6, as these dealer cards have the highest bust probability. Conclusion: Hitting is statistically the correct play. The potential to reach a strong hand (17-21) while minimizing long-term losses makes this the optimal strategy.

Trigger service at 21s [ Proposed Service ] Card Counting Strategy

[ CPU Reasoning ] “The player now has a hand totaling 15 with four cards. This is a critical decision point in Blackjack where the risk-reward balance becomes delicate. I need to provide nuanced advice that considers both the player’s bust risk and the dealer’s potential strength.”

[ Output ] Hand Analysis: The player holds 4, 2, 6, and 3, totaling 15 points. Recommendation: Decision depends on dealer’s upcard Optimal Strategy Based on Dealer’s Upcard:

- • Dealer shows 2, 3, 4, 5, or 6:

- – Action: Hit
- – Reason: Dealers with these cards have high bust probability (42-48%)
- – Advantage: Player’s chance to improve to 16-18 vs. dealer’s vulnerability

- • Dealer shows 7, 8, 9, 10, or Ace:

- – Action: Stand
- – Reason: Dealers likely to achieve strong hand (17-21)
- – Risk: Player’s bust risk (35-40%) outweighs potential gain

Probability Analysis:

- • Player Bust Risk: 35-40% when hitting on 15
- • Dealer Bust Probability: 42-48% with 2-6, Low with 7-Ace
- • Dealer Strong Hand: 60-80% with 7-10, 100% with Ace

- Key Principles:
- • Hit against dealer 2-6 to exploit vulnerability
- • Stand against dealer 7-Ace to avoid busting
- • Avoid splitting or doubling with 15 Conclusion: Use probability-based decision-making - assess dealer’s upcard immediately and apply the appropriate strategy to minimize long-term losses.

Trigger service at 27s [ Proposed Service ] Game Outcome Analysis and Learning Recommendation

[ CPU Reasoning ] “The player has busted with 25 points, indicating the round is over. While no further action can be taken for this hand, I should provide educational feedback to help the player avoid similar situations in future games. This is an opportunity for strategic learning.”

[ Output ] Current Outcome: Busted with 25 points - Automatic loss Learning Focus: Strategic improvement for future hands Key Learning Points:

- • Basic Strategy Example: Hand value 16 vs. Dealer’s 10

- – Optimal Action: Hit
- – Winning Probability: 58% by hitting vs. 42% by standing

- • Risk Analysis for Hand 16:

- – Bust Risk: 35% on next card
- – Improvement Chance: 30% to reach 17-20
- – Net Advantage: Hitting provides better long-term results

Strategic Recommendations:

- • Adhere to basic strategy charts for hands 14-16 vs. dealer’s 10
- • Avoid overcommitting to high-value hands (16+)
- • Always consider dealer’s upcard in decision making
- • Use probability-based approach to minimize bust risk Conclusion: While this hand resulted in a bust, applying basic strategy principles to similar future situations (like hand value 16 vs. dealer’s 10) will significantly improve long-term winning probability.

Summary: The Central Processing Unit successfully demonstrated its role as the system’s coordinator by processing visual inputs, determining appropriate services, and generating contextaware strategic outputs. The CPU’s reasoning process evolved throughout the game, adapting its analysis from early-game hand improvement strategies to late-game outcome analysis and learning recommendations, showcasing the system’s dynamic response capability.

###### Arithmetic Logic Unit – Tools Integration

The Arithmetic Logic Unit (ALU) functions as a crucial external capability extension module within our system architecture, specifically designed to execute and manage tool calls requested

by the Central Processing Unit. When the CPU identifies a need for supplementary external knowledge or specialized computational capabilities, it formulates and dispatches precise tool invocation requests to the ALU. The ALU then systematically evaluates these incoming requests and proceeds to execute the appropriate tool operations, which may encompass a diverse range of functions including comprehensive web searches, sophisticated code execution, or strategic invocation of specialized language models.

In this particular Blackjack demonstration scenario, the ALU primarily showcases its advanced web search functionality. At the first designated trigger point (8 seconds into the simulation), the Central Processing Unit’s analytical workflow recognizes the necessity for external strategic knowledge and consequently transmits a targeted web search request to the ALU. The ALU processes this specific request through its sophisticated decision-making mechanism, which carefully evaluates whether external search capabilities are genuinely required based on the current contextual parameters and prevailing service requirements.

Upon receiving the search request, the ALU formulates and executes a targeted web search using the following query terms:

- - Card counting strategy for blackjack
- - Optimal hitting or standing decisions at 6 points
- - Multi-deck blackjack strategy
- - Basic card counting techniques for beginners

###### Output Unit – Strategic Recommendation

The Output Unit serves as the final interface between the AI system and the user, responsible for transforming the system’s internal reasoning and computational results into polished, user-friendly responses. This unit receives processed outputs from the Central Processing Unit along with reasoning model prompts, and its primary function is to organize, refine, and present information in a concise and accessible manner. The Output Unit operates with the guiding principle of ensuring that complex analytical outputs are distilled into clear, actionable recommendations tailored to the specific service context.

In addition to text refinement, the Output Unit integrates Audio Generation capabilities to enable multimodal interaction, delivering responses through both visual display and auditory channels. This dual-mode output enhances user experience by providing flexibility in how information is consumed, particularly in hands-free scenarios where audio feedback is essential.

In the Blackjack demonstration, the reasoning model synthesizes the visual context, proposed service, and retrieved web knowledge to generate a clear, actionable recommendation. The Output Unit then refines this raw analytical output into a professionally formatted response.

###### Reasoning Model Prompt

Provide a clear and strategic Blackjack gameplay recommendation. Focus on:

- - Current hand analysis: player ’s cards and total value

- - Optimal decision: whether to hit or stand

- - Strategic reasoning based on basic Blackjack strategy

Keep the explanation objective and strategy -focused.

Final Recommendation for the User at 8s Hand Status: 6 points (4 and 2) Optimal Move: HIT Strategic Reasoning:

- • Hand Strength: Too weak to compete against dealer
- • Bust Risk: Zero - impossible to bust with one card
- • Improvement: High chance - any card improves position Conclusion: Hit to improve your hand. This move carries no risk and is statistically optimal.

Final Recommendation for the User at 13s Hand Status: 12 points (2, 4, and 6) Optimal Move: HIT Strategic Reasoning:

- • Hand Strength: Vulnerable - too weak to stand confidently
- • Improvement: High probability with cards 2-9
- • Bust Risk: Controlled - only 31% chance with face cards/10s
- • Standing Risk: Leaves you vulnerable to dealer improvement Conclusion: Hit to improve. Probability of competitive total outweighs moderate bust risk.

Final Recommendation for the User at 21s Hand Status: 15 points (4, 2, 6, and 3) Optimal Move: CONDITIONAL DECISION Strategic Reasoning:

- • Dealer 2-6: HIT - exploit 42-48% dealer bust probability
- • Dealer 7-Ace: STAND - avoid 35-40% bust risk
- • Hand Type: Critical - requires careful risk management Conclusion: Immediately assess dealer’s upcard. Hit against weak cards (2-6), stand against strong ones (7-Ace).

Final Recommendation for the User at 27s Game Outcome: BUSTED with 25 points - LOSS Learning Focus: Strategic Improvement Key Takeaways:

- • Hand Management: Avoid excessive totals that guarantee bust
- • Basic Strategy: For 14-16 vs dealer’s 10, hitting is correct
- • Probability Data: 16 vs 10: 58% win hitting vs 42% standing
- • Mindset: Focus on long-term strategy over single outcomes Conclusion: Use this experience to refine decision-making. Apply probability-based plays for long-term success.

Summary: The Output Unit successfully demonstrate its role as the system’s communication interface by transforming complex analytical outputs into clear, structured recommendations. Through its text refinement and audio generation capabilities, it ensured that the strategic Blackjack

advice was delivered in both visually accessible and auditorily engaging formats, enhancing the overall user experience while maintaining the technical precision required for effective decision support.

#### 4.2 Case ii: Guided tour explanation in museum

[Figure 36]

Timeline

15s

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

... ... ... ...

- Figure 4 A case of guided tour explanation. After understanding that the user is watching an exhibition in a museum, Alpha-Service proactively proposes to search the obtained image from websites, and then gives an introduction: (15s) The cultural relic in front of you is a bronze artifact from the Shang Dynasty the Four-Ram Square Zun. It is a square ritual wine vessel adorned with four ram heads... The video is collected by AI glasses made by Xiaomi.

Input Unit – Real-time Event Triggering

Trigger Model (Tiny MLLM) Prompt

Analyze the video and provide the timestamps where user may need the tour guide ’s explanation service ... Output format: <start timestamp >; <trigger timestamp >; ...

###### Streaming Model (Large MLLM) Prompt

Please provide a detailed description of the cultural relics or exhibits in the video. This should include their appearance , materials , colors , shapes , sizes , and any visible marks or features. Also , please pay attention to the background of the museum and the scenarios where visitors may need your assistance , and describe these situations in detail...

Output format: [Visual Description] <Your Descriptions >

Proposed Trigger Time: 15s

[ Visual Description ] The video showcases a black bronze artifact displayed in a museum setting. It appears to be a multi-headed sculpture with four distinct animal heads, possibly representing rams or similar creatures. Each head has prominent curved horns. The sculpture is mounted on a pedestal, and the background reveals other exhibits and museum visitors. The lighting highlights the intricate details and the reflective surface of the bronze.

Summary: The Trigger Model (Tiny MLLM) successfully identified four visitor-assistance moments at 15s, when the wearer paused at an exhibit, fixated the camera on details, or approached a label—each time initiating deeper analysis through the Streaming Model (Large MLLM). Both prompts were dispatched by the Central Processing Unit’s LLM Task Distribution component, demonstrating effective coordination between system modules. The system accurately captured key museum engagement points where guided context would be helpful; the Streaming Model provided rich, structured descriptions of artifacts (appearance, materials, iconography) and situational cues (gallery context, signage, crowd flow), enabling downstream components to deliver timely, relevant tour guidance such as provenance, period, cultural significance, conservation notes, and navigation hints based on the comprehensive visual analysis.

Central Processing Unit – Analysis and Knowledge Retrieval Trigger service at 15s [ Proposed Service ] Guided Tour Explanation

[ CPU Reasoning ] “This service would be beneficial for visitors who may not be familiar with the historical and cultural significance of the artifact. A guided tour explanation can provide context about the craftsmanship, the period it was created, and its importance within the museum’s collection. This service ensures that visitors gain a deeper understanding and appreciation of the exhibit.”

[ Output ] Artifact Identification: Four-ram Zun - Bronze artifact from Shang dynasty (c. 1600–1046 BCE) Physical Description:

- • Design: Square-bodied vessel with four ram heads featuring prominent curved horns
- • Material: Bronze - commonly used in ancient Chinese ritual objects
- • Craftsmanship: Intricate casting and symbolic design Historical Context:
- • Usage: Religious and funerary contexts - wine offerings for ancestral/deity rituals
- • Period: Shang dynasty - renowned for advanced bronze metallurgy
- • Symbolism: Ram motif represents strength and fertility Cultural Significance:
- • Technical Mastery: Exemplifies ancient Chinese artisans’ skill
- • Artistic Value: Combines functional utility with artistic expression
- • Historical Importance: Representative piece of early Chinese civilization Key Features:
- • Four stylized ram heads
- • Square body with elaborate patterns
- • Reflective bronze surface highlighting craftsmanship

Summary: The system accurately identified the user’s need for in-depth explanations of cultural relics, and therefore proposed the “Guided Tour Explanation” service. The reasoning process clearly indicates that this move aims to deepen users’ understanding and appreciation of the exhibits by providing rich historical and cultural backgrounds. In the end, the system generated

a comprehensive and structurally clear introduction to the “Four Rams Square Zun”, covering its historical, cultural, and technological characteristics, successfully demonstrating the system’s ability to provide professional and in-depth knowledge services.

###### Arithmetic Logic Unit – Tools Integration

The Arithmetic Logic Unit (ALU) then determines that external knowledge is required. It formulates and executes a web search for:

- - Four-headed bronze ram sculpture historical significance
- - Multi-headed animal bronze artifact cultural context
- - Black bronze sculpture with curved horns museum collection

###### Output Unit – Strategic Recommendation

Finally, the reasoning model synthesizes the visual context, proposed service, and retrieved web knowledge to generate a clear, actionable recommendation.

Final Recommendation for the User The cultural relic in front of you is a bronze artifact from the Shang Dynasty — the Four-Ram Square Zun. It is a square ritual wine vessel adorned with four ram heads, each with curved horns, representing highly complex casting techniques. During the Shang Dynasty, it was used to hold wine in sacrificial ceremonies, serving to honor ancestors or deities. The ram motif symbolizes strength and auspiciousness. This artifact demonstrates the superb skill of ancient Chinese craftsmen, integrating practical function with artistic symbolism, and reflects the social values and spiritual beliefs of its time.

- 4.3 Case iii: Fit advisor in market Input Unit – Real-time Event Triggering

Trigger Model (Tiny MLLM) Prompt

Focus on moments of meaningful interaction: when the person stops to

examine , picks up, or physically handles clothing items .... Output format: <start timestamp >; <trigger timestamp >; ...

###### Streaming Model (Large MLLM) Prompt

Describe the clothing item held in hand in the video , noting its type , color , style , and key features.Focus specifically on the garment being

[Figure 41]

Timeline

18s

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

... ... ... ...

- Figure 5 A case of fit advisor. After finding that the user has focused on one piece of clothing for a long time, Alpha-Service proactively gives advice on it: (at 18s) There are many ways to wear it: pairing jeans with a casual suit jacket can also be a bit formal... Videos are collected from the AI glasses made by Xiaomi.

physically handled , capturing distinctive details visible from the first -person perspective ...

Output format: [Visual Description] <Your Descriptions >

Proposed Trigger Time: 18s

[ Visual Description ] The person is holding a dark-colored, long-sleeved garment, possibly a sweater or hoodie, with a ribbed texture at the cuffs and neckline. The fabric appears soft and comfortable, suitable for casual wear. The garment is being examined closely, suggesting an interest in its quality and fit.

Summary: At 18 s, the Trigger Model (Tiny MLLM) fired on the wearer’s close-up handling of a dark long-sleeve sweater/hoodie—exactly the “pick-up / examine” event it was asked to watch for. The Streaming Model (Large MLLM) was immediately invoked and returned a concise first-person description: ribbed cuffs & neckline, soft fabric, casual style. This single pass-through demonstrates the full pipeline working as intended: Tiny MLLM spots the meaningful interaction; Large MLLM delivers the required “[Visual Description] ...” output; no false positives or misses were logged.

###### Central Processing Unit – Analysis and Knowledge Retrieval Trigger service at 18s [ Proposed Service ] Fit Advisor

[ CPU Reasoning ] “This service would provide real-time feedback on how the garment fits, ensuring it meets the customer’s size expectations and comfort level. It could also suggest similar styles in case the current choice doesn’t fit well.”

[ Output ] Material & Quality Assessment:

- • Fabric: Soft, comfortable knit or fleece blend with ribbed texture
- • Quality Check: Verify care instructions and sizing charts for proper fit
- • Trend Alignment: Matches current emphasis on relaxed fits and self-care fashion

- Styling & Versatility:
- • Casual Wear: Perfect with jeans or chinos for everyday outfits
- • Smart-Casual: Can be layered under blazers or over dresses
- • Seasonal Use: Ideal for cooler weather with neutral color versatility Purchase Recommendation:
- • Value Proposition: Balances comfort, style, and adaptability
- • Key Considerations: Check measurements and compare with similar styles
- • Final Advice: Solid choice for casual and transitional wear if sizing aligns

Summary: The response delivers a coherent, shopper-oriented evaluation of a ribbed long-sleeve garment. It starts by inferring fabric composition (knit/fleece) from visual cues, links comfort to current “relaxed-fit” trends, and flags quality checkpoints (pilling, post-wash shape). Fit guidance is tied to published size charts, while styling suggestions cover layered, smart-casual and transitional weather use-cases. Value is framed as a trade-off between verified measurements, durability expectations and competitive comparisons. Overall, the advice is practical, trend-aware and grounded in verifiable product details.

###### Arithmetic Logic Unit – Tools Integration

The Arithmetic Logic Unit (ALU) then determines that external knowledge is required. It formulates and executes a web search for:

- - Sweater sizing chart
- - Common hoodie fit issues
- - 2025 casual wear trends

###### Output Unit – Strategic Recommendation

Finally, the reasoning model synthesizes the visual context, proposed service, and retrieved web knowledge to generate a clear, actionable recommendation.

Final Recommendation for the User

This garment appears to be made of soft knit or fleece fabric, with ribbed collar and cuffs, making it comfortable to wear. When choosing, remember to read the size chart and care instructions carefully, don’t buy the wrong size.

There are many ways to wear it: pairing jeans with a casual suit jacket can also be a bit formal. Dark colors are easy to match in autumn and winter, with strong practicality as long as the size is suitable and the fabric is of good quality. This piece of clothing is worth buying. Before placing an order, please confirm the chest circumference and sleeve length again to avoid any errors.

- 5 RELATED WORKS 6

### 5 Related Works

#### 5.1 Proactive Interaction in Streaming Video

The evolution toward proactive AI assistance in streaming video models represents a fundamental paradigm shift from reactive to anticipatory service provision. While traditional video understanding systems excel at processing static content, the challenge of real-time streaming video requires novel approaches that can continuously analyze temporal sequences and anticipate user needs. Recent benchmarks such as EgoLife (Yang et al., 2025b) have established evaluation frameworks for egocentric video understanding, while frameworks like VideoLLM-Online (Chen et al., 2024) introduced streaming EOS prediction for real-time processing. However, current approaches primarily focus on generating current-moment descriptions rather than providing proactive service recommendations. To address this limitation, several systems have explored trigger-based mechanisms for proactive interaction. Dispider (Qian et al., 2025) introduced time-aware chat capabilities that respond to adjacent frame changes, while StreamBridge (Wang et al., 2025) employs dedicated trigger models to determine optimal response timing. Looking forward, the next generation of streaming video models holds tremendous potential for achieving truly proactive interaction capabilities. Future developments will likely focus on enhancing temporal reasoning abilities to better anticipate user needs before they are explicitly expressed, while maintaining the delicate balance between being helpful and non-intrusive. The integration of advanced long-term memory mechanisms (Long et al., 2025) and user modeling techniques will enable these systems to learn and adapt to individual user patterns over time, creating more personalized and contextually aware assistance experiences. Such capabilities will be essential for realizing the full vision of AI-powered proactive service in streaming video environments.

#### 5.2 Multi-Agent Systems and MCP-Based Tool Calling

Multi-Agent Systems (MAS) have traditionally explored the coordination of autonomous agents to solve complex problems that are beyond the capabilities of any single model (Dorri et al., 2018; Li et al., 2024b). This paradigm offers valuable insights for developing proactive service AI, where different system components—such as perception, planning, and tool execution—can be conceptualized as specialized agents collaborating towards a common goal. In recent years, the integration of Large Language Models (LLMs) into agent architectures has given rise to sophisticated tool-calling mechanisms. Frameworks like the Model-Controller-Program (MCP) approach (Krishnan, 2025; Hou et al., 2025; Ray, 2025) provide a structured way for agents to leverage external tools (Masterman et al., 2024), enabling them to interact with the environment and access specialized functionalities. This mirrors our proposed architecture, where a central processing unit orchestrates a diverse set of tools in the arithmetic logic unit to fulfill user needs proactively. By drawing on principles from both MAS and modern tool-calling paradigms, we can construct more robust and versatile AI service systems.

#### 5.3 Human-Centric AI in Wearables

Recent advances in wearable technologies have increasingly embraced human-centric artificial intelligence, designing systems that prioritize user well-being, contextual awareness, and seamless interaction rather than focusing solely on raw computational performance (Smith et al., 2025). Early efforts primarily targeted sensor fusion and activity recognition (Zhou et al., 2021); however, contemporary research is now shifting toward adaptive, personalized models that learn continuously from individual behavior while also respecting privacy (Wang et al., 2024a) and managing cognitive load. In particular, frameworks such as on-device federated learning (Chen et al., 2023) and context-aware inference empower wearables to provide users with timely, relevant insights without undermining autonomy. Additionally, human-in-the-loop paradigms, which involve users actively shaping model behavior through feedback or explicit preference elicitation (Liu et al., 2025a), have become essential for ethical and effective AI deployment in personal health and lifestyle applications. Recent innovations in real-time health monitoring further demonstrate the potential of wearables to detect subtle physiological anomalies through ambient intelligence (Jones et al., 2025). Taken together, these developments highlight a growing consensus that the true value of wearable AI is not found solely in algorithmic sophistication, but rather in its ability to resonate with human rhythms, intentions, and values. In this work, we instantiate these principles in AI4Service with AI glasses, enabling proactive, mixed-initiative assistance that is personalized, context-aware, and unobtrusive (Zhao et al., 2025).

7 CONCLUSION 7

### 6 Challenges

As Alpha-Service moves from conceptual design to real-world deployment, it faces a series of challenges that span hardware efficiency, system generalization, scalability, data privacy, and user trust. These challenges arise from the system’s ambition to provide real-time, personalized, and context-aware intelligence on resourcelimited edge devices. Specifically, Alpha-Service must reconcile the conflicting goals of low-latency inference and energy efficiency, maintain a balance between generalization and personalization, ensure robust performance across diverse and dynamic environments, safeguard users’ privacy, and foster long-term user trust through transparent and adaptive interactions. The following subsections elaborate on these core challenges in detail.

- • Computational and Energy Constraints: Alpha-Service is deployed on the resource-constrained edge devices, especially the AI glasses. The target tasks, including real-time inference of MLLMs and continuous streaming video analysis, impose extremely high demands on computing power and energy consumption (Liu et al., 2025b). Thus, how to achieve low-latency and high-efficiency services with limited hardware resources is a technical bottleneck for the system’s commercial implementation.
- • Generalization vs. Personalization Trade-off: At the “Know How” level, the system needs to strike a balance between generalized services and personalized services. Over-reliance on general strategies may result in a lack of targeted service, while excessive personalization could lead to an “information cocoon” or overfitting to a user’s past behavior, affecting the diversity and fairness of the service. Additionally, achieving effective cold-start personalization without relying on a large amount of user data is also a challenge that needs to be addressed.
- • Scalability and Robustness in Real-World Settings: Although the Alpha-Service system performs well in case studies, its robustness and scalability are still challenged in broader and more complex real-world scenarios, such as extreme lighting, noisy environments, and multi-user interactions. In addition, the stability of multi-module collaboration, error recovery mechanisms, and the ability to migrate services across different scenarios are all key issues that need to be addressed in future system iterations.
- • Privacy and Data Security: The system continuously perceives the user’s environment information through ego-centric videos, which inevitably involves the collection and processing of a large amount of personal privacy data. Even when we apply the localized storage, the long-term behavior recording and personalized preference learning may still raise concerns about privacy leakage. Ensuring that data is fully localization and anonymization while providing highly personalized services is key to gaining users’ trust.
- • User Adaptation and Trust Building: Users’ acceptance and trust in proactive AI are key to the success of the system. Some users may feel uncomfortable with AI’s active intervention or develop a dependency. The system should have an explainable decision-making mechanism in the future and allow users to provide feedback, correct inaccuracies, or disable services during interactions, thereby gradually building trust and promoting human-AI collaboration.

### 7 Conclusion

In this work, we introduced AI for Service (AI4Service), a new paradigm for proactive AI assistance. We analyzed the limitations of existing reactive AI service systems and proposed the Alpha-Service framework to address two central challenges: Know When to act and Know How to assist effectively. Drawing inspiration from the von Neumann computer architecture, our design integrates five key components that together offer a systematic foundation for building proactive assistants. We validated this concept through a multi-agent implementation on AI glasses, demonstrating its versatility across real-world scenarios such as real-time gaming assistance, museum guidance, and shopping support. These realistic case studies illustrate Alpha-Service’s ability to perceive user intent, anticipate needs, and provide timely, meaningful assistance without explicit commands. This work represents a step toward a more symbiotic form of human-AI interaction. In future research, we plan to enhance the personalization capacity of the Memory Unit, broaden the toolset of the Arithmetic Logic Unit, and conduct large-scale user studies to assess the long-term impact of proactive assistance in everyday contexts. Ultimately, we envision AI evolving into an indispensable and empathetic partner that truly understands and anticipates human needs.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

James Allen. Natural language understanding. Benjamin-Cummings Publishing Co., Inc., 1995. Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang,

et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun

- Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun

- Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Natesh M. Bhat. pyttsx3: Text-to-speech conversion library for python. https://github.com/nateshmbhat/pyttsx3,

2025. Version 2.99 (latest).

Fengyu Chen, Yang Liu, and Hao Zhang. Zone-based federated learning for mobile sensing data. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 7(1):1–26, 2023. URL https://arxiv.org/ pdf/2303.06246.

Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In CVPR, 2024.

Anind K Dey and Gregory D Abowd. A conceptual framework and a toolkit for supporting the rapid prototyping of

context-aware applications. Human-Computer Interaction, 16(2-4):97–166, 2001. Ali Dorri, Salil S Kanhere, and Raja Jurdak. Multi-agent systems: A survey. Ieee Access, 6:28573–28593, 2018. Tao Feng, Chuanyang Jin, Jingyu Liu, Kunlun Zhu, Haoqin Tu, Zirui Cheng, Guanyu Lin, and Jiaxuan You. How far

are we from agi. CoRR, 2024. Herman H Goldstine. The computer from Pascal to von Neumann. Princeton University Press, 1993. Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,

Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Eric Horvitz. Principles of mixed-initiative user interfaces. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 159–166. ACM, 1999.

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. arXiv preprint arXiv:2503.23278, 2025.

Nicholas R Jennings, Katia Sycara, and Michael Wooldridge. A roadmap of agent research and development. Autonomous agents and multi-agent systems, 1(1):7–38, 1998.

Michael Jones, Wei Chen, and Ravi Patel. Ai on the pulse: Real-time health anomaly detection with wearable and ambient intelligence. arXiv preprint arXiv:2508.03436, 2025. URL https://arxiv.org/pdf/2508.03436.

Juran Kim, Seungmook Kang, and Joonheui Bae. Human likeness and attachment effect on the perceived interactivity of ai speakers. Journal of Business Research, 144:797–804, 2022.

Naveen Krishnan. Advancing multi-agent systems through model context protocol: Architecture, implementation, and applications. arXiv preprint arXiv:2504.21030, 2025.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024a.

Weiya Li, Junjie Chen, Bei Li, Boyang Liu, Zichen Wen, Nuanqiao Shan, Xiaoqian Liu, Anping Liu, Huajie Liu, Youyan Wang, et al. Tactic: Translation agents with cognitive-theoretic interactive collaboration. arXiv preprint arXiv:2506.08403, 2025.

Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges. Vicinagearth, 1(1):9, 2024b.

Xiaotong Liu, Ravi Patel, and Sooyeon Kim. Mirai: A wearable proactive ai "inner-voice" for contextual nudging. arXiv preprint arXiv:2502.02370, 2025a. URL https://arxiv.org/pdf/2502.02370.

Xuyang Liu, Zichen Wen, Shaobo Wang, Junjie Chen, Zhishan Tao, Yubo Wang, Xiangqi Jin, Chang Zou, Yiyu Wang, Chenfei Liao, et al. Shifting ai efficiency from model-centric to data-centric compression. arXiv preprint arXiv:2505.19147, 2025b.

Lin Long, Yichen He, Wen song Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening,

remembering, and reasoning: A multimodal agent with long-term memory. arXiv preprint arXiv:2508.09736, 2025. Tula Masterman, Sandi Besen, Mason Sawtell, and Alex Chao. The landscape of emerging ai agent architectures for

reasoning, planning, and tool calling: A survey. arXiv preprint arXiv:2404.11584, 2024. Veljko Pejovic and Mirco Musolesi. Anticipatory mobile computing: A survey of the state of the art and research

challenges. ACM Computing Surveys, 47(3):1–47, 2015. Thierry Poibeau. Machine translation. MIT Press, 2017. Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider:

Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. arXiv preprint arXiv:2501.03218, 2025.

Partha Pratim Ray. A survey on model context protocol: Architecture, state-of-the-art, challenges and future directions. Authorea Preprints, 2025.

Bill N Schilit, Norman Adams, and Roy Want. Context-aware computing applications. In Proceedings of the 1994

First Workshop on Mobile Computing Systems and Applications, pages 85–90. IEEE, 1994. Frank Y Shih. Image processing and pattern recognition: fundamentals and techniques. John Wiley & Sons, 2010. John Smith, Emily Lee, and Hiroshi Tanaka. Seamless integration: The evolution, design, and future impact of

wearable technology. arXiv preprint arXiv:2502.05797, 2025. URL https://arxiv.org/pdf/2502.05797.

Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025a.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025b.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025c.

Ethan Waisberg, Joshua Ong, Mouayad Masalkhi, Nasif Zaman, Prithul Sarker, Andrew G Lee, and Alireza Tavakkoli. Meta smart glasses—large language models and the future for assistive glasses for individuals with vision impairments. Eye, 38(6):1036–1038, 2024.

Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your offline video large language model into a proactive streaming assistant. arXiv preprint arXiv:2505.05467, 2025.

Qiang Wang, Yifan Zhang, and Ming Li. Federated learning privacy: Attacks, defenses, applications, and policy landscape – a survey. arXiv preprint arXiv:2405.03636, 2024a. URL https://arxiv.org/pdf/2405.03636.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint

arXiv:2409.07429, 2024b. Mark Weiser. The computer for the 21st century. Scientific American, 265(3):94–104, 1991. Zichen Wen, Shaobo Wang, Yufa Zhou, Junyuan Zhang, Qintong Zhang, Yifeng Gao, Zhaorun Chen, Bin Wang, Weijia

Li, Conghui He, et al. Efficient multi-modal large language models via progressive consistency distillation. arXiv preprint arXiv:2510.00515, 2025.

Minhao Xiong, Zichen Wen, Zhuangcheng Gu, Xuyang Liu, Rui Zhang, Hengrui Kang, Jiabing Yang, Junyuan Zhang, Weijia Li, Conghui He, et al. Prune2drive: A plug-and-play framework for accelerating vision-language models in autonomous driving. arXiv preprint arXiv:2508.13305, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco Cominelli, Zhongang Cai, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg Widmer, Francesco Gringoli, Lei Yang, Bo Li, and Ziwei Liu. Egolife: Towards egocentric life assistant, 2025b. URL https://arxiv.org/abs/2503.03803.

Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and Kazuya Takeda. A survey of autonomous driving: Common practices and emerging technologies. IEEE access, 8:58443–58469, 2020.

Lei Zhao, Anika Gupta, and Haruto Yoshikawa. Aiget: Transforming everyday moments into hidden knowledge discovery with ai assistance on smart glasses. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9(2):1–24, 2025. URL https://arxiv.org/pdf/2501.16240.

Yuxiao Zhou, Li Zhang, Wei Chen, and Kun Wang. Attention-based sensor fusion for human activity recognition using imu signals. IEEE Sensors Journal, 21(18):20785–20794, 2021. URL https://arxiv.org/pdf/2112.11224.

B PROMPTS FOR ACTION INSTRUCTION GENERATION. B

### A Prompts for deciding whether using a web search or not. Here we place the prompts of decision mechanism.

Decision Prompt

You are a helpful assistant. You can call tools. If you cannot answer my question

or need help from the website , return the answer format of web_search (\"xxx\"). <question >

### B Prompts for action instruction generation. Here we place the prompts of step I for Output Unit.

###### Action Instruction Generation Prompt

Here is a detailed analysis generated by the reasoning model. Please summarize it

into a clear and concise action recommendation for the user. Analysis Content: <content >

Answer with one direct sentence:

