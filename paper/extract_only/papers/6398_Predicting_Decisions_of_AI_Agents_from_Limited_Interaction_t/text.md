# arXiv:2605.12411v1[cs.LG]12May2026

## Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling

Eilam Shapira Moshe Tennenholtz Roi Reichart Faculty of Data and Decision Sciences Technion – Israel Institute of Technology Haifa, Israel {eilam.shapira, moshe.tennenholtz, roireichart}@gmail.com

### Abstract

AI agents increasingly negotiate and transact in natural language with unfamiliar counterparts: a buyer bot facing an unknown seller, or a procurement assistant negotiating with a supplier. In such interactions, the counterpart’s underlying LLM, prompts, control logic, and rule-based fallbacks are hidden, while each decision can have monetary consequences. We ask whether an agent can predict an unfamiliar counterpart’s next decision from only a few prior interactions. To avoid real-world logging confounds, we study this problem in controlled bargaining and negotiation games, formulating it as target-adaptive text-tabular prediction: each decision point is a table row combining structured game state, offer history, and dialogue, while K previous games of the same target agent, i.e., the counterpart being modeled, are provided in the prompt as labeled adaptation examples. Our model is built on a tabular foundation model that represents rows using game-state features and LLM-based text representations, and adds LLM-as-Observer as an additional representation: a small frozen LLM reads the public decision-time state and dialogue; its answer is discarded, and its hidden state becomes a decisionoriented feature, making the LLM an encoder rather than a direct few-shot predictor. Training on 13 frontier-LLM agents and testing on 91 held-out scaffolded agents, the full model outperforms direct LLM-as-Predictor prompting and game+text features baselines. Within this tabular model, Observer features contribute beyond the other feature schemes: at K = 16, they improve response-prediction AUC by about 4 points across both tasks and reduce bargaining offer-prediction error by 14%. These results show that formulating counterpart prediction as a target-adaptive text-tabular task enables effective adaptation, and that hidden LLM representations expose decision-relevant signals that direct prompting does not reliably surface.1

### 1 Introduction

AI agents increasingly negotiate and transact in natural language with unfamiliar counterparts: a buyer bot facing an unknown seller, or a procurement assistant negotiating with a supplier. In such interactions, the counterpart’s underlying LLM, prompts, control logic, and rule-based fallbacks are hidden, while each decision can have monetary consequences. We ask whether an agent can predict an unfamiliar counterpart’s next decision from only a few prior interactions.

Real marketplace logs would be the most direct testbed, but they are rarely public and typically do not support systematic comparison across many agents under matched strategic conditions with known payoffs and ground-truth decisions. We therefore study the problem in controlled bargaining and negotiation games. These games preserve key elements of language-mediated commerce: multi-turn

1Code and the 91-agent dataset of 4,921 bargaining and negotiation games will be released upon acceptance.

Preprint.

[Figure 1]

[Figure 2]

Round 1

8000$

This is a real gem, you will have to pay a lot!

Reject

Round 2

4000$

But today is boxing day!

Reject

Round 3

6000$ Let's split the difference

Reject

Round 4

5000$ Not worth a penny more than 5k$

Alice ? Bob

- Figure 1: Alice (seller) and Bob (buyer) negotiate via free-text offers. Following Bob’s $5,000 round-4 offer, Alice’s next move is the prediction target. (a) Response prediction (classification): will she accept? (b) Proposal prediction (regression): if she rejects, what will she propose?

offers, accept/reject decisions, private valuations, monetary payoffs, and free-text dialogue. They also let us vary horizons, valuations, and information regimes while observing the decisions agents actually make.

We call the unfamiliar counterpart being modeled the target agent. For each target, the predictor is given K complete prior games played by that same agent, which serve as labeled examples of the target’s behavior. At test time, the predictor receives a new decision point: the public game state, the offer history, and the dialogue so far. It must predict the target’s next move. We study two complementary tasks, illustrated in Figure 1: response prediction, a binary classification task asking whether the target accepts the current offer, and proposal prediction, a regression task asking what offer the target will make next.

We formulate this as target-adaptive text-tabular prediction. Each decision point is represented as a table row combining structured game variables, offer history, and dialogue-derived text features. A tabular foundation model conditions on labeled rows from a source population of previously observed agents together with the K labeled games of the current target agent. This allows the predictor to combine population-level regularities with target-specific evidence, adapting to a new counterpart without observing its prompt, code, or control logic.

Our model uses three complementary feature blocks. The first contains game-state features, such as public configuration variables, round number, current offer, and previous offers. The second contains generic text representations of the dialogue. The third is our new decision-oriented representation, LLM-as-Observer: a small frozen LLM reads the public decision-time state and dialogue, its direct answer is discarded, and its hidden state is used as an additional feature for the tabular predictor. Thus, the LLM is used as an encoder rather than as the final few-shot predictor.

This design contrasts with a natural alternative, LLM-as-Predictor: prompting a large frontier LLM with the current game and the target’s K prior games, and asking it to predict the next decision directly. Direct prompting can read the dialogue and reason over examples in context, but it must commit to an answer and cannot easily combine the target’s few games with a large labeled source population. In our formulation, the LLM contributes a reusable representation, while adaptation is performed by the tabular learner over source and target rows.

For the source population, we use the 13-agent round-robin tournament released as part of GLEE [59], where frontier LLMs2 from six providers play under identical prompts, varying only in the underlying LLM. For the held-out target population, we introduce a 91-agent university-hackathon dataset: student-built agents that share one underlying LLM but differ in prompting, control logic, and rulebased fallbacks. This split tests whether predictors learned from one axis of agent variation transfer to newly encountered engineered agents whose heterogeneity comes from scaffolding.

The full target-adaptive text-tabular model, trained on 13 frontier-LLM agents and tested on 91 held-out scaffolded agents, outperforms direct LLM-as-Predictor prompting and game+text features baselines. Within the tabular model, Observer features add complementary signal beyond structured game features and generic dialogue representations. At K = 16, they improve response-prediction AUC by about four percentage points across both game families and reduce bargaining offer-prediction

2frontier LLMs: state-of-the-art API Large Language Models

error by 14%. The gain is not mainly in the Observer’s committed answer: hidden states provide substantially more value than its direct output, suggesting that frozen LLM representations expose decision-relevant information that direct prompting does not reliably surface.

Contributions. First, we formulate few-shot prediction of unfamiliar language-based agents as a target-adaptive text-tabular task, where K prior games of the target agent provide labeled adaptation examples. Second, we build a prediction model that combines game-state features, dialogue representations, and a new decision-oriented feature block, LLM-as-Observer. Third, we introduce a 91-agent hackathon dataset and a cross-population transfer evaluation from frontier-LLM agents to scaffolded agents, showing that the full model outperforms direct LLM-as-Predictor prompting and game+text features baselines, and that Observer hidden states add complementary decision-relevant signal.

### 2 Related work

Multi-agent applications and the role of language. The applications motivating this paper sit in language-mediated commerce: consumer-to-consumer marketplaces [29, 75], residential real-estate transactions [30], tourism and travel-package negotiations [52], multi-stakeholder contract deliberations [1], and the broader emerging “agentic economy” of LLM-based shopping and procurement assistants [56], with early controlled deployments of LLM-vs-LLM marketplaces already reported [5]. They differ from non-language multi-agent AI such as multi-agent autonomous driving [21], multirobot coordination [23], algorithmic trading [69], and distributed power-grid control [17], where agents observe each other through sensors, actions, and shared infrastructure, rather than through a dialogue. A second line of multi-agent learning research trains agents to coordinate through continuous vectors optimised end-to-end with their policies [65] or through emergent discrete codes invented for the task [40]: in those settings the communication channel is task-tuned, opaque to outside observers, and trained jointly with the policy. The setting we study sits on the opposite end of this axis: target agents emit fluent natural-language messages produced by pretrained LLMs [28], the channel itself is human-readable and not co-trained with the predictor, and any external observer must read the same public stream of strategic state and free-form dialogue that a human auditor would.

LLMs as strategic agents. A growing literature studies LLMs and other AI systems as strategic agents in language-mediated settings: bargaining and negotiation [59, 71, 38, 12], persuasion and social influence [10, 15, 58, 60, 66], auctions and market-like environments [18, 24, 77], social dilemmas and cooperation [42, 9, 43], and broader social-agent benchmarks [78, 73, 35, 70]. Whereas this prior work characterises how LLMs behave as a population of strategic agents, we ask a per-agent predictive question: given K observed games of a specific unseen agent, what will it decide next? Methods of population characterisation do not directly transfer to this task: they aggregate across agents, while we need to make a prediction at the individual-agent level.

Predicting agent behavior from limited histories. Predicting another actor’s behaviour from limited interaction histories is a long-standing problem in multi-agent AI. Classical opponent-modelling maintains beliefs over a library of hypothesised agent types and updates them from observed actions [3, 47, 26, 2, 4]; automated negotiation learns preferences from partial dialogue [8, 19, 16]; ad-hoc teamwork predicts the behaviour of unfamiliar teammates [64, 44, 55, 68]; and Theory-ofMind networks [54, 48, 49, 41, 46, 72, 76] and predictors for human decisions in negotiation and persuasion [14, 58, 60, 61, 39] learn end-to-end from behavioural traces. These methods show that short histories can support prediction, but assume an agent type drawn from a known prior or a population matched to training, not an open-ended LLM-based agent whose implementation style is previously unseen. A modern alternative is to prompt a large API-based LLM in-context as a few-shot predictor [13, 22]. Throughout this paper we use “LLM-as-Predictor” to mean exactly this: a large API-based LLM prompted at inference time as a predictor. Our small-Observer pipeline is both cheaper at inference and more accurate (Section 6).

Multi-modal text–tabular learning. Each decision point in our setting combines structured game fields, such as offers, round number, and configuration parameters, with free-form dialogue. We therefore treat the task as text–tabular prediction. Tabular foundation models support in-context prediction from labeled examples without gradient-based retraining [32, 33, 53], matching our fewshot target-agent setting. Prior work studies text–tabular learning through multi-modal AutoML,

dedicated benchmarks, cross-table transfer, and foundation models for tables with text fields [62, 45, 7, 36, 37, 6]. Our setting differs in requiring rapid adaptation to a newly observed strategic agent from only K games, using source-population rows and target-specific examples without gradient-based retraining.

Frozen LM representations as transferable features. Frozen LMs expose information through intermediate hidden states that is not always captured by their final outputs. Probing work shows that syntactic, semantic, and task-relevant variables can be decoded from these states [11, 20, 31, 67]. Related work further shows that intermediate or layer-combined representations often transfer better than final-layer outputs on downstream tasks [51, 34, 63]. Recent studies also find that hidden states can encode knowledge or signals that are not reflected in the model’s generated answer [25, 50]. We use this line of work as motivation for a feature block in a text-tabular predictor: the Observer reads the public game state and dialogue, but the downstream model predicts the target agent’s decision from its hidden state together with game and dialogue features. This differs from standard probing in the target being predicted: the representation is extracted from one model observing the interaction, while the label is the next decision of another, black-box strategic agent.

### 3 Data

We instantiate our prediction task in GLEE [59], a benchmark and simulation framework for twoplayer, sequential, language-based economic games. In GLEE, agents repeatedly make strategic decisions–such as proposing an offer or accepting/rejecting one–while observing the public interaction history and, in the language condition, exchanging free-text messages. The benchmark fixes the game rules while systematically varying payoff parameters, horizons, information regimes, and communication channels. This makes GLEE a natural source for our task: it preserves key ingredients of language-mediated commerce–private values, monetary incentives, multi-turn offers, and strategic dialogue–while providing controlled conditions and ground-truth agent decisions.

We focus on GLEE’s two mixed-motive families most aligned with our prediction setting: bargaining and negotiation. In both, two agents alternate offers accompanied by free-text messages, and each decision point can be represented as a text-tabular row containing the public configuration, offer history, dialogue so far, and the target agent’s next move. This yields our two prediction tasks: response prediction, asking whether the target accepts the current offer, and proposal prediction, asking what offer the target makes next.

Bargaining. Two agents divide a fixed sum M over multiple rounds in an alternating-offers game [57]. At each round, the proposer suggests a split (p,1−p) and sends a message; the responder accepts, ending the game, or rejects, allowing the interaction to continue with reversed roles. Delay is costly through per-round discount factors δ1,δ2 ∈ (0,1]. Configurations vary in the horizon, the discount factors, and whether each agent observes the other’s discount factor. Thus, agents must interpret both offers and language when deciding whether to concede, reject, or counter-offer.

Negotiation. A seller with private reserve value VS and a buyer with private valuation VB negotiate over the price of a single indivisible good. They alternate price offers, each accompanied by a free-text message. The responder can accept, ending the game; reject and continue when the horizon allows it; or exercise an outside option that guarantees zero surplus. For response prediction, we group outside-option decisions with rejection, since both are decisions not to accept the current offer. Configurations vary in the horizon, valuations, and whether each side observes the other’s valuation.

Table 1: Agent populations used for cross-population transfer, split by game family.

Population Family # Agents # Games # Decisions Avg. # rounds Variation axis Role GLEE tournament [59]

Barg. 13 30,888 97,031 3.8

LLM source Nego. 13 30,940 99,889 4.0

Barg. 90 2,936 8,762 6.0

University hackathon (ours)

scaffolding target Nego. 60 1,985 2,579 2.6

LLM as Predictor

Textual Prompt

ŷ

Textual-Tabular Task

A B

|0|
|---|

TabPFN

Predict the decision:

LLM

|1|2| |3|4| |5| |6| |7|8| |9|10| |11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Accept or Reject

|1|
|---|

|2|
|---|

ŷ

|3|
|---|

|4|
|---|

|5|
|---|

Textual-Tabular + Observer

C

###### ×K

TabPFN

|0|
|---|

|6| |7|
|---|---|---|

|8|
|---|

|1|2| |3|4| |5| |6| |7|8| |9|10| |11| |12| |13| |14| |15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

|9| |10|
|---|---|---|

|11|
|---|

Decision:

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Game Features

Text Representation

Observer Representation

Label

- Figure 2: Three approaches for predicting decisions of a target agent. (A) LLM-as-Predictor receives the decision-time state, dialogue, and K observed target games, and directly outputs the decision. (B) Textual-tabular prediction represents each decision point as a row of game features and dialogue. (C) Our method augments this row with Observer hidden-state representations from a frozen LLM.

Because valuations are private, agents must infer value from offers, signal credibly through language, and decide when agreement remains worthwhile.

We use two complementary agent populations (Table 1): the GLEE frontier-LLM tournament as the training source, where agents vary in the underlying LLM, and a new university-hackathon dataset as the held-out target population, where agents vary in scaffolding around a shared underlying LLM. This split tests whether predictors trained on one axis of agent variation transfer to newly encountered agents whose heterogeneity comes from a different source.

Frontier-LLM tournament (training source). The source population is the GLEE round-robin tournament: 13 frontier LLMs from six providers (full model list in Appendix B) play bargaining and negotiation games under identical system prompts, so agents vary only in the underlying model. The tournament covers 960 configurations over horizons, discount factors, valuations, information regimes, and communication regimes, yielding ≈64K games and 197K accept/reject decisions.

University hackathon (held-out target). The target population is a new dataset from a competitive university hackathon held in December 2025, where 34 teams competed for a $2,000 prize. In contrast to the GLEE tournament, agents were restricted to the Gemini 2.5 Flash/Flash-Lite API surface but differed in scaffolding: engineered control logic, prompting pipelines, rule-based fallbacks, or combinations of these. We include logs from all competition stages, treating each submitted team-stage version as a distinct agent, yielding 91 agents, 4,921 games, and 11,341 decisions.

This source–target design tests whether predictors trained on agents that differ mainly in their underlying LLM transfer to agents that differ mainly in scaffolding.

### 4 Method

Our goal is to predict the next decision of a previously unseen language-based agent from only a few observed games. The central design choice is to treat this as target-adaptive tabular prediction. Instead of asking an LLM to directly imitate the target agent, we represent each decision point through complementary feature modalities and let a tabular foundation model adapt to the target from its K labeled games.

- Figure 2 summarizes the model. At a decision point, the predictor observes only the public game state and the dialogue so far. We convert this information into three feature modalities: structured gamestate features, a generic dialogue representation, and a decision-oriented hidden-state representation from a small frozen LLM, which we call the Observer. These features are combined by the same tabular predictor, which conditions on a large source population together with the target’s K observed games. We first define the prediction setting, then describe the three feature modalities, the tabular predictor, and the baselines.

#### 4.1 Prediction setting

At each round, the target agent makes one of two types of decisions. In response prediction, the target receives an offer and must decide whether to accept it. This is a binary classification task. In proposal prediction, the target makes the next offer. This is a regression task over a normalized offer

value. Together, these two tasks cover the main observable moves made by agents in bargaining and negotiation games.

For a new target agent, we are given K previously observed games and must predict its decisions in held-out games. The target itself is never queried at inference time, and we never observe its prompt, code, or control logic. All predictors receive only the information that would be public at the decision point. In private-information configurations, values that are private to either player are masked and are not supplied to the game+text features, the LLM-as-Predictor prompt, or the Observer input.

#### 4.2 Feature modalities

Each decision point is represented by three complementary modalities: structured game-state features, a generic dialogue representation, and the Observer hidden-state representation (Figure 4). Together they form a single multimodal tabular row that the predictor of Section 4 consumes.

- • Game-state features. These features encode the structured strategic state of the game: the public configuration, the current offer, the round index, previous offers and decisions, and negotiationspecific information such as outside options when they are public. This modality gives the predictor direct access to the incentives and history that shape rational play.
- • Dialogue representation. Because agents communicate in natural language, the same offer can have different implications depending on the accompanying message. We therefore encode the dialogue so far with a sentence encoder and reduce the representation before passing it to the tabular predictor. This modality captures semantic information from the conversation, but it is not explicitly trained to represent the target’s strategic decision.
- • Observer representation. The Observer is a small frozen LLM that reads the public decision-time state and dialogue. It is prompted toward the same decision the target is about to make, but its direct answer is discarded. Instead, we extract an internal hidden state and use it as a decision-oriented representation of the situation. The Observer is never fine-tuned, never sees the target’s prompt or code, and does not receive the target’s K past games in its prompt. Adaptation to the target happens only in the downstream tabular predictor. This separation is a key methodological observation. A frontier LLM-as-Predictor must both understand the situation and directly commit to a prediction. LLM-as-Observer uses the LLM only for the first role: constructing a representation. The final decision is made by a tabular model that can combine this representation with source-population data and the target’s few labeled games.

Because the conditioning set mixes rows from many source agents with rows from the current target agent, identical game states may correspond to different behavioral policies. The agent-identity indicator tells the tabular predictor which rows come from the same decision-maker, allowing it to distinguish population-level regularities from target-specific deviations. Without this marker, source and target rows are treated as exchangeable observations even when they are generated by different agents.

#### 4.3 Tabular predictor

The prediction module performs target-adaptive tabular inference over the multimodal row representation. For each target, the predictor conditions on labeled examples from the source population together with the target’s K observed games. The same predictor is used for both tasks: classification mode for response prediction and regression mode for proposal prediction.

#### 4.4 Baselines and controls

We evaluate the full target-adaptive text-tabular model against direct prompting and reduced tabular baselines, and then isolate the marginal contribution of the Observer feature block.

- • Game+text features baseline. This baseline uses the same tabular predictor and the same targetadaptation protocol, but removes the Observer representation. It receives only the structured game-state features, the dialogue representation, and the agent-identity indicator. This tests whether the Observer adds information beyond a strong tabular model.
- • LLM-as-Predictor. This baseline prompts a frontier LLM with the current game, the dialogue, and the target’s K observed games, and asks it to predict the target’s next decision directly. This tests the natural alternative of using a large LLM as the predictor itself. The approach can read

- Table 2: Cross-population transfer. Response (left): mean AUC, 5 seeds; For Observer methods, we average results over layers at relative depth 0.6–0.9. Proposal (right): median R2 on scale-normalised offers, 5 seeds. ±SE. Bold: best per family per K; shading: Observer beats strongest baseline.

Response (AUC) Proposal (R2) Method K=0 K=2 K=4 K=8 K=16 K=0 K=2 K=4 K=8 K=16

Game+text features .621±.01 .683±.01 .712±.01 .748±.01 .791±.01 .366±.08 .407±.10 .503±.08 .502±.07 .622±.05 LLM-as-Predictor .577±.01 .686±.01 .726±.01 .750±.01 .770±.01 −2.8±.80 −1.6±.24 −0.5±.19 −0.5±.19 −0.3±.27 Gemma-as-Observer .666±.01 .719±.01 .754±.01 .793±.01 .831±.01 .470±.07 .513±.07 .598±.06 .635±.05 .676±.03 Qwen3-as-Observer .655±.01 .714±.01 .749±.01 .786±.01 .826±.01 .506±.06 .530±.07 .590±.05 .641±.04 .673±.02 Llama-as-Observer .642±.01 .695±.01 .730±.01 .770±.01 .812±.01 .486±.07 .519±.07 .596±.06 .638±.04 .674±.03

Barg.

Game+text features .719±.01 .768±.01 .765±.01 .767±.01 .803±.01 .552±.11 .579±.12 .567±.11 .723±.12 .857±.08 LLM-as-Predictor .486±.01 .684±.01 .684±.01 .713±.01 .785±.02 −8.0±4.8 −3.2±2.9 −0.4±.81 .164±.21 .506±.29 Gemma-as-Observer .738±.01 .782±.01 .794±.01 .792±.01 .826±.01 .551±.12 .578±.13 .583±.12 .731±.12 .845±.09 Qwen3-as-Observer .808±.01 .825±.01 .833±.01 .833±.01 .852±.01 .540±.12 .573±.13 .558±.12 .711±.11 .852±.09 Llama-as-Observer .743±.01 .774±.01 .782±.01 .779±.01 .810±.01 .555±.12 .582±.13 .544±.11 .725±.12 .862±.09

Nego.

the dialogue and reason over examples in context, but it must commit to a prediction from the prompt alone and does not produce a reusable representation that can be combined with labeled source-population rows.

### 5 Experimental setup

Evaluation protocol. Our main evaluation is cross-population transfer. We train on the frontierLLM tournament population and test on held-out hackathon agents, one target at a time. The source population varies the underlying LLM while holding scaffolding fixed; the target population holds the underlying LLM fixed while varying prompts, control logic, and rule-based fallbacks. This protocol tests whether features learned from one axis of agent variation transfer to the other.

For each target, we sample K ∈ {0,2,4,8,16} observed games as adaptation examples and evaluate on the remaining games. Response prediction is evaluated with AUC, and proposal prediction is evaluated with R2 over normalized offers. The tabular classifier is trained on up to 3,000 individual decisions drawn from GLEE source agents (balanced across agents) together with the target’s Kgame decisions. We instantiate the tabular prediction module with TabPFN v2.6. Observer metrics average over each model’s upper-stack layer band (relative depth 0.6–0.9), removing the per-layer optimisation knob. Observer models are small frozen LLMs (Gemma-2-2B, Qwen3-1.7B, and Llama3.2-1B; 1–2B parameters). The LLM-as-Predictor baseline uses Gemini 2.5 Flash–the model made available to hackathon participants–giving direct prompting a substantial capacity advantage over the Observer.

### 6 Results

We evaluate cross-population transfer from the 13-agent frontier-LLM tournament to the held-out hackathon agents. The results are organized around two questions. First, does a target-adaptive tabular predictor outperform direct LLM prompting? Second, does the Observer hidden state add information beyond game+text features? All tabular methods use the same TabPFN predictor, with the Observer referring to the configuration using game features, text embeddings, and LLM hidden states. Specific models are named by their LLM (e.g., Gemma-as-Observer), while Game+text features is the baseline without hidden states. The only non-tabular method is LLM-as-Predictor, which prompts a frontier API directly. Throughout, K is the number of adaptation games, and performance is measured by AUC (response) and R2 (proposals).

Response prediction: Observer hidden states are the strongest predictor. Table 2 shows that LLM-as-Observer improves response prediction across both game families and all values of K. In bargaining, the best Observer yields a substantial gain of +4.0pp over the game+text features baseline and +6.1pp over the LLM-as-Predictor at K=16. In negotiation, the best Observer provides a +4.9pp improvement over game+text features and +6.7pp over LLM-as-Predictor. The same pattern appears already at K=0, where the Observer improves over the tabular baseline without any target-specific examples.

- Table 3: Feature ablation at K=16 (Gemma-2-2B Observer). Left: leave-one-out from the full model; Right: reduced feature stacks. G = Game, T = Text, O = Observer, I = Identity. Results show mean AUC (response) and median R2 (proposal), averaged over mid-to-late Observer layers.

Full −O −T −G −I G+I T+I O+I I Response (AUC)

Barg. .831 .791 .833 .778 .779 .796 .583 .783 .500 Nego. .826 .803 .815 .615 .817 .779 .621 .604 .500

Barg. .689 .616 .695 .618 .607 .621 .013 .625 −.038 Nego. .841 .844 .843 .450 .722 .843 .063 .438 −.057

Proposal (R2)

LLM-as-Predictor is weaker. The LLM-as-Predictor baseline is a demanding comparison: it uses a large frontier API model, receives the current game and the target’s K observed games directly in context, and is from the same model family used by most hackathon agents. Nevertheless, Table 2 shows that it trails LLM-as-Observer at every K in both families. The failure is not due to the Predictor lacking language understanding or being mismatched to the target population. Direct few-shot prompting is a weaker interface for this prediction problem than extracting a reusable representation and letting the tabular model adapt over labeled source and target rows.

Proposal prediction: the Observer helps when structured history is not enough. Table 2 shows a more nuanced effect for proposal prediction. In bargaining, all three Observer variants improve over game+text features across K, yielding a median R2 increase of approximately +0.05 over the game+text features baseline at K=16. Using the Gemma-2-2B Observer as a representative point, this reduces the typical one-offer prediction error on a nominally $10,000 split from $552 to $473 at K=16, a 14% reduction. Importantly, this gain is specific to the bargaining setting, where the strategic dynamic relies heavily on interpreting text alongside numerical offers. In negotiation, by contrast, Table 2 shows that the game+text features baseline is already very strong at K=16, and the Observer variants do not provide a clear additional improvement. The pattern in Table 2 therefore sharpens the claim: Observer hidden states help when the next offer is not already captured by the structured game history.

LLM-as-Predictor is especially weak for numerical offers. Table 2 also shows that direct LLM prompting is poorly calibrated for proposal prediction. In bargaining, LLM-as-Predictor has a negative median R2 even at K=16. In negotiation, it improves with more examples but still does not match the game+text features baseline. The large LLM can read the game and produce plausible numbers, but it is not a reliable regression model: autoregressive token decoding is poorly suited to calibrated numerical regression, and the in-context K-shot mechanism that helps the Predictor on binary classification has weak traction on continuous values. The tabular formulation is therefore not merely cheaper or more convenient; it is the right prediction interface for turning few observed games into calibrated numerical estimates.

### 7 Robustness and ablation

We isolate the load-bearing components of our framework to validate its two primary pillars: the text-tabular formulation and the integration of frozen Observer representations.

The feature hierarchy. Table 3 reveals a clear performance hierarchy across four feature blocks. Structured game features provide the essential backbone; removing them leads to the most substantial performance collapse, particularly in negotiation. The Observer hidden states supply the critical situational layer that generic dialogue embeddings fail to capture. Notably, once the Observer is integrated, generic sentence embeddings become largely redundant, providing no meaningful marginal gains.

Latent representation vs. direct prediction. A key architectural choice is using the LLM as an Observer rather than a Predictor. Analysis shows that feeding the hidden states into the tabular model consistently outperforms the Observer’s direct accept/reject probabilities (logits) across all conditions (Appendix E). This justifies the text-tabular approach: the tabular learner (TabPFN) decodes strategic signals from the LLM’s latent space more effectively than the LLM’s own output head.

###### Response prediction Proposal prediction

K = 4

K = 4

K = 0

K = 2

K = 8

K = 16

K = 0

K = 2

K = 8

K = 16

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

Barg.AUC

0.050

2Barg.R

0.10

0.025

0.05

0.000

0.00

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

Nego.AUC

2Nego.R

0.10

0.04

0.05

0.02

0.00

0.00

0 1

0 1 0 1

0 1

0 1

0 1

0 1

0 1

0 1

0 1

Relative depth Relative depth

Gemma-2-2B Qwen3-1.7B Llama-3.2-1B

- Figure 3: Observer gain over the game+text features baseline by relative depth. Observer gains are stable across mid-to-late layers (relative depth 0.6–0.9) (Left: Response, ∆AUC; Right: Proposal, ∆R2). Rows: bargaining (top), negotiation (bottom); columns: K-shot examples.

Stability across providers, tasks, and layers. The Observer effect is robust across encoder LLMs: hidden states from Gemma-2-2B, Qwen3-1.7B, and Llama-3.2-1B all yield a consistent improvement over the baseline when fed into the tabular model. Figure 3 explains this stability and shows why the hidden state is the relevant LLM signal: rather than peaking at a single tuned layer, the gains remain stable across mid-to-late layers (relative depth 0.6–0.9) across both providers and the response and bargaining-proposal tasks. This confirms that the predictive signal is a stable, intrinsic property of mid-to-late Observer representations rather than an artifact of specific layer selection.

### 8 Discussion and conclusion

The main takeaway of this paper is that predicting the decisions of an unfamiliar language-based agent is better framed as target-adaptive text-tabular learning than as direct few-shot LLM prediction. In this formulation, structured game-state and offer-history features provide the strategic backbone, dialogue features expose the language channel, and the target’s K observed games provide adaptation evidence. The LLM-as-Observer adds a decision-oriented feature block that substantially improves the model.

This framing explains why the text-tabular model outperforms direct prompting. A frontier LLMas-Predictor can read the current interaction and the target’s past games, but it must compress all evidence into a single generated answer and cannot naturally combine those examples with a large labeled source population. The tabular learner instead conditions jointly on source-population rows and target-specific rows, which better matches the statistical structure of the problem.

Within our model, the Observer adds a complementary signal. It consistently improves response prediction across both game families and outperforms LLM prompting despite using much smaller frozen LLMs. The gains do not come mainly from the Observer’s final answer, but from its hidden representation, suggesting that frozen LLMs encode decision-relevant information that is not reliably exposed in their generated output. For proposal prediction, the picture is more nuanced: the Observer helps in bargaining, where language and strategic positioning matter beyond offer history, but adds little in negotiation, where the next offer is already well predicted from the structured game state, offer history, and target examples.

Our cross-population evaluation tests transfer to deployment-like agents. The source population consists of controlled frontier-LLM agents, which provide a broad labeled prior over strategic behavior. The held-out hackathon population consists of black-box scaffolded agents that differ in prompting, control logic, and rule-based fallbacks. Training on the former and testing on the latter evaluates whether a predictor learned from a reusable controlled source population can adapt to newly encountered engineered agents.

While our results are encouraging, several limitations remain. The games are controlled abstractions of language-mediated commerce, not real markets; the method assumes access to a relevant source population; and the Observer’s contribution varies across tasks. Overall, our results suggest a general recipe for modeling the decision making of unfamiliar AI agents: separate representation from adaptation. Use language models to construct decision-relevant representations of strategic dialogue,

but let the final prediction be made by a supervised model that can combine structured incentives, source-population evidence, and the target’s few observed decisions.

### Acknowledgments and Disclosure of Funding

Eilam Shapira is supported by a Google PhD Fellowship. Roi Reichart has been partially supported by a VATAT grant on data science. We thank Alan Arazi, Maya Zadok, and Shoham Grunblat for helpful comments on earlier versions of this work. We are grateful to Itamar Reichman, Gur Keinan, Idan Hahn, Gila Molcho, Omer Ben Porat, Avigdor Gal, and Rann Smorodinsky for their support in organizing the hackathon.

### References

- [1] Sahar Abdelnabi, Amr Gomaa, Sarath Sivaprasad, Lea Schönherr, and Mario Fritz. Cooperation, competition, and maliciousness: LLM-stakeholders interactive negotiation. In Advances in Neural Information Processing Systems: Datasets and Benchmarks Track, volume 37, 2024.
- [2] Stefano V Albrecht and Subramanian Ramamoorthy. A game-theoretic model and best-response learning method for ad hoc coordination in multiagent systems. In Proceedings of the 12th International Conference on Autonomous Agents and Multiagent Systems, pages 1155–1156, 2013.
- [3] Stefano V Albrecht and Peter Stone. Autonomous agents modelling other agents: A comprehensive survey and open problems. Artificial Intelligence, 258:66–95, 2018.
- [4] Stefano V Albrecht, Jacob W Crandall, and Subramanian Ramamoorthy. Belief and truth in hypothesised behaviours. Artificial Intelligence, 235:63–94, 2016.
- [5] Anthropic. Project Deal: A Claude-run marketplace experiment. Anthropic research blog, 2026. https://www.anthropic.com/features/project-deal.
- [6] Alan Arazi, Eilam Shapira, and Roi Reichart. TabSTAR: A tabular foundation model for tabular data with text fields. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=FrXHdcTEzE.
- [7] Alan Arazi, Eilam Shapira, Shoham Grunblat, Mor Ventura, Elad Hoffer, Gioia Blayer, David Holzmüller, Lennart Purucker, Gaël Varoquaux, Frank Hutter, and Roi Reichart. MulTaBench: Benchmarking multimodal tabular learning with text and image. arXiv preprint arXiv:2605.10616, 2026.
- [8] Tim Baarslag, Mark J. C. Hendrikx, Koen V. Hindriks, and Catholijn M. Jonker. Learning about the opponent in automated bilateral negotiation: A comprehensive survey of opponent modeling techniques. Autonomous Agents and Multi-Agent Systems, 30(5):849–898, 2016. doi: 10.1007/s10458-015-9309-1.
- [9] Steffen Backmann, David Guzman Piedrahita, Emanuel Tewolde, Rada Mihalcea, Bernhard Schölkopf, and Zhijing Jin. When ethics and payoffs diverge: LLM agents in morally charged social dilemmas. arXiv preprint arXiv:2505.19212, 2025.
- [10] Michael Bao. ElecTwit: A framework for studying persuasion in multi-agent social systems. arXiv preprint arXiv:2601.00994, 2026.
- [11] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022. doi: 10.1162/coli_a_00422.
- [12] Federico Bianchi, Patrick John Chia, Mert Yuksekgonul, Jacopo Tagliabue, Dan Jurafsky, and James Zou. How well can LLMs negotiate? NegotiationArena platform and analysis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 3935–3951. PMLR, 2024.

- [13] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), volume 33, pages 1877–1901, 2020.
- [14] Anaïs Cadilhac, Nicholas Asher, Farah Benamara, and Alex Lascarides. Grounding strategic conversation: Using negotiation dialogues to predict trades in a win-lose game. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 357–368, 2013.
- [15] Gian Maria Campedelli, Nicolò Penzo, Massimo Stefan, Roberto Dessì, Marco Guerini, Bruno Lepri, and Jacopo Staiano. I want to break free! persuasion and anti-social behavior of LLMs in multi-agent settings with social hierarchy. arXiv preprint arXiv:2410.07109, 2024.
- [16] Kushal Chawla, Gale Lucas, Jonathan May, and Jonathan Gratch. Opponent modeling in negotiation dialogues by related data adaptation. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 661–674, Seattle, United States, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-naacl.50.
- [17] Dong Chen, Kaian Chen, Zhaojian Li, Tianshu Chu, Rui Yao, Feng Qiu, and Kaixiang Lin. PowerNet: Multi-agent deep reinforcement learning for scalable powergrid control. IEEE Transactions on Power Systems, 37(2):1587–1599, 2022.
- [18] Jiangjie Chen, Siyu Yuan, Rong Ye, Bodhisattwa Prasad Majumder, and Kyle Richardson. Put your money where your mouth is: Evaluating strategic planning and execution of LLM agents in an auction arena. arXiv preprint arXiv:2310.05746, 2023.
- [19] Robert M. Coehoorn and Nicholas R. Jennings. Learning on opponent’s preferences to make effective multi-issue negotiation trade-offs. In Proceedings of the 6th International Conference on Electronic Commerce, pages 59–68, 2004.
- [20] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia, 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1198.
- [21] Jiaxun Cui, William Macke, Harel Yedidsion, Aastha Goyal, Daniel Urieli, and Peter Stone. Scalable multiagent driving policies for reducing traffic congestion. In Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS), 2021.
- [22] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.64.
- [23] Erwan Escudie, Laetitia Matignon, and Jacques Saraydaryan. Attention graph for multi-robot social navigation with deep reinforcement learning. In Proceedings of the 23rd International Conference on Autonomous Agents and MultiAgent Systems (AAMAS), 2024. Extended Abstract.
- [24] Sara Fish, Yannai A. Gonczarowski, and Ran I. Shorrer. Algorithmic collusion by large language models. arXiv preprint arXiv:2404.00806, 2024.
- [25] Zorik Gekhman, Eyal Ben David, Hadas Orgad, Eran Ofek, Yonatan Belinkov, Idan Szpektor, Jonathan Herzig, and Roi Reichart. Inside-Out: Hidden factual knowledge in LLMs. In Conference on Language Modeling, 2025.
- [26] Piotr J Gmytrasiewicz and Prashant Doshi. A framework for sequential planning in multi-agent settings. Journal of Artificial Intelligence Research, 24:49–79, 2005.
- [27] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [28] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence, pages 8048–8057, 2024. doi: 10.24963/ijcai.2024/890.
- [29] He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. Decoupling strategy and generation in negotiation dialogues. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2333–2343, Brussels, Belgium, 2018. Association for Computational Linguistics.
- [30] Mourad Heddaya, Solomon Dworkin, Chenhao Tan, Rob Voigt, and Alexander Zentefis. Language of bargaining. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13161–13185. Association for Computational Linguistics, 2023.
- [31] John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4129–4138, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1419.
- [32] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations (ICLR), 2023.
- [33] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025.
- [34] MohammadSaleh Hosseini, Munawara Munia, and Latifur Khan. BERT has more to offer: BERT layers combination yields better sentence embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15419–15431, 2023.
- [35] Seth Karten, Wenzhe Li, Zihan Ding, Samuel Kleiner, Yu Bai, and Chi Jin. LLM economist: Large population models and mechanism design in multi-agent generative simulacra. arXiv preprint arXiv:2507.15815, 2025.
- [36] Myung Jun Kim, Leo Grinsztajn, and Gael Varoquaux. CARTE: Pretraining and transfer for tabular learning. In Proceedings of the 41st International Conference on Machine Learning, pages 23843–23866, 2024.
- [37] Boshko Koloski, Andrei Margeloiu, Xiangjian Jiang, Blaž Škrlj, Nikola Simidjievski, and Mateja Jamnik. LLM embeddings for deep learning on tabular data. arXiv preprint arXiv:2502.11596, 2025.
- [38] Deuksin Kwon, Emily Weiss, Tara Kulshrestha, Kushal Chawla, Gale Lucas, and Jonathan Gratch. Are LLMs effective negotiators? systematic evaluation of the multifaceted capabilities of LLMs in negotiation dialogues. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5391–5413, 2024.
- [39] Tiziano Labruna, Arkadiusz Modzelewski, Giorgio Satta, and Giovanni Da San Martino. Detecting winning arguments with large language models and persuasion strategies. In Findings of the Association for Computational Linguistics: EACL 2026, pages 1888–1915, Rabat, Morocco,

2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.findings-eacl.97.

- [40] Angeliki Lazaridou, Alexander Peysakhovich, and Marco Baroni. Emergence of linguistic communication from referential games with symbolic and pixel input. In Proceedings of the 6th International Conference on Learning Representations (ICLR), 2018.
- [41] Huao Li, Yu Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Charles Lewis, and Katia Sycara. Theory of mind for multi-agent collaboration via large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 180–192, 2023.

- [42] Yuxuan Li and Hirokazu Shirado. Spontaneous giving and calculated greed in language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5271–5286, 2025.
- [43] Hachem Madmoun and Salem Lahlou. Communication enables cooperation in LLM agents: A comparison with curriculum-based approaches. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pages 307–321, Rabat, Morocco, 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.eacl-short.23.
- [44] Reuth Mirsky, Ignacio Carlucho, Arrasy Rahman, Elliot Fosong, William Macke, Mohan Sridharan, Peter Stone, and Stefano V. Albrecht. A survey of ad hoc teamwork research. In Multi-Agent Systems. EUMAS 2022, volume 13442 of Lecture Notes in Computer Science, pages 275–293. Springer, 2022. doi: 10.1007/978-3-031-20614-6_16.
- [45] Martin Mráz, Breenda Das, Anshul Gupta, Lennart Purucker, and Frank Hutter. Towards benchmarking foundation models for tabular data with text. In Foundation Models for Structured Data Workshop at ICML, 2025.
- [46] Chunjiang Mu, Ya Zeng, Qiaosheng Zhang, Kun Shao, Chen Chu, Hao Guo, Danyang Jia, Zhen Wang, and Shuyue Hu. Adaptive theory of mind for LLM-based multi-agent coordination. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 29608–29616,

2026. doi: 10.1609/aaai.v40i35.40204.

- [47] Samer B Nashed and Shlomo Zilberstein. A survey of opponent modeling in adversarial domains. Journal of Artificial Intelligence Research, 73:277–327, 2022.
- [48] Dung Nguyen, Phuoc Nguyen, Hung Le, Kien Do, Svetha Venkatesh, and Truyen Tran. Learning theory of mind via dynamic traits attribution. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems, pages 954–962, 2022.
- [49] Dung Nguyen, Phuoc Nguyen, Hung Le, Kien Do, Svetha Venkatesh, and Truyen Tran. Memoryaugmented theory of mind network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11630–11637, 2023. doi: 10.1609/aaai.v37i10.26374.
- [50] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In The Thirteenth International Conference on Learning Representations, 2025.
- [51] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2227–2237, New Orleans, Louisiana, 2018. Association for Computational Linguistics.
- [52] Priyanshu Priya, Rishikant Chigrupaatii, Mauajama Firdaus, and Asif Ekbal. GENTEELNEGOTIATOR: LLM-enhanced mixture-of-expert-based reinforcement learning approach for polite negotiation dialogue. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25010–25018, 2025.
- [53] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 50817–50847. PMLR, 2025.
- [54] Neil Rabinowitz, Frank Perbet, Francis Song, Chiyuan Zhang, S. M. Ali Eslami, and Matthew Botvinick. Machine theory of mind. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 4218–4227. PMLR, 2018.
- [55] João G. Ribeiro, Gonçalo Rodrigues, Alberto Sardinha, and Francisco S. Melo. TEAMSTER: Model-based reinforcement learning for ad hoc teamwork. Artificial Intelligence, 324:104013,

2023. doi: 10.1016/j.artint.2023.104013.

- [56] David M. Rothschild, Markus Mobius, Jake M. Hofman, Eleanor W. Dillon, Daniel G. Goldstein, Nicole Immorlica, Sonia Jaffe, Brendan Lucier, Aleksandrs Slivkins, and Matthew Vogel. The agentic economy. arXiv preprint arXiv:2505.15799, 2025.
- [57] Ariel Rubinstein. Perfect equilibrium in a bargaining model. Econometrica, 50(1):97–109, 1982.
- [58] Eilam Shapira, Omer Madmon, Roi Reichart, and Moshe Tennenholtz. Can LLMs replace economic choice prediction labs? The case of language-based persuasion games. arXiv preprint arXiv:2401.17435, 2024.
- [59] Eilam Shapira, Omer Madmon, Itamar Reinman, Samuel Joseph Amouyal, Roi Reichart, and Moshe Tennenholtz. GLEE: A unified framework and benchmark for language-based economic environments. arXiv preprint arXiv:2410.05254, 2024.
- [60] Eilam Shapira, Omer Madmon, Reut Apel, Moshe Tennenholtz, and Roi Reichart. Human choice prediction in language-based persuasion games: Simulation-based off-policy evaluation. Transactions of the Association for Computational Linguistics, 13:980–1006, 2025.
- [61] Eilam Shapira, Moshe Tennenholtz, and Roi Reichart. Alignment makes language models normative, not descriptive. arXiv preprint arXiv:2603.17218, 2026.
- [62] Xingjian Shi, Jonas Mueller, Nick Erickson, Mu Li, and Alexander J. Smola. Benchmarking multimodal AutoML for tabular data with text fields. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.
- [63] Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. In Proceedings of the 42nd International Conference on Machine Learning, pages 55854–55875, 2025.
- [64] Peter Stone, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein. Ad hoc autonomous agent teams: Collaboration without pre-coordination. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 1504–1509, 2010. doi: 10.1609/aaai.v24i1.7529.
- [65] Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. Learning multiagent communication with backpropagation. In Advances in Neural Information Processing Systems 29 (NIPS), 2016.
- [66] Amir Taubenfeld, Yaniv Dover, Roi Reichart, and Ariel Goldstein. Systematic biases in LLM simulations of debates. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 251–267, 2024.
- [67] Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4593–4601, 2019. doi: 10.18653/v1/P19-1452.
- [68] Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone. N-agent ad hoc teamwork. In Advances in Neural Information Processing Systems, 2024.
- [69] Ziyi Wang, Carmine Ventre, and Maria Polukarov. Multi-agent reinforcement learning for market making: Competition without collusion. In Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF), 2025.
- [70] Yusen Wu, Yiran Liu, and Xiaotie Deng. MALLES: A multi-agent LLMs-based economic sandbox with consumer preference alignment. arXiv preprint arXiv:2603.17694, 2026.
- [71] Tian Xia, Zhiwei He, Tong Ren, Yibo Miao, Zhuosheng Zhang, Yang Yang, and Rui Wang. Measuring bargaining abilities of LLMs: A benchmark and a buyer-enhancement method. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3579–3602, 2024.
- [72] Yang Xiao, Jiashuo Wang, Qiancheng Xu, Changhe Song, Chunpu Xu, Yi Cheng, Wenjie Li, and Pengfei Liu. Towards dynamic theory of mind: Evaluating LLM adaptation to temporal evolution of human states. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24036–24057, 2025.

- [73] Sixiong Xie, Zhuofan Shi, Haiyang Shen, Yun Ma, Gang Huang, and Xiang Jing. M3-BENCH: Process-aware evaluation of LLM agents’ social behaviors in mixed-motive games. arXiv preprint arXiv:2601.08462, 2026.
- [74] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [75] Runzhe Yang, Jingxiao Chen, and Karthik Narasimhan. Improving dialog systems for negotiation with personality modeling. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 681–693. Association for Computational Linguistics, 2021.
- [76] Zhining Zhang, Chuanyang Jin, Mung Yao Jia, and Tianmin Shu. AutoToM: Automated bayesian inverse planning and model discovery for open-ended theory of mind. arXiv preprint arXiv:2502.15676, 2025.
- [77] Yushuo Zheng, Huiyu Duan, Zicheng Zhang, Yucheng Zhu, Xiongkuo Min, and Guangtao Zhai. Market-Bench: Benchmarking large language models on economic and trade competition. arXiv preprint arXiv:2604.05523, 2026.
- [78] Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, LouisPhilippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. SOTOPIA: Interactive evaluation for social intelligence in language agents. In International Conference on Learning Representations, 2024.

### A Game configurations

This section lists the exact game configurations played by each population, in support of Section 3. The hackathon target population is restricted to configurations with free-text messages enabled (messages_allowed=true); the GLEE source population includes both communication regimes.

GLEE source population. The frontier-LLM tournament sweeps 384 distinct bargaining configurations and 576 distinct negotiation configurations. Bargaining varies the money-to-divide M ∈ {100,10000,1,000,000}, the round horizon max_rounds ∈ {12,∞}, the information regime (complete vs. incomplete), the communication regime (free-text messages enabled vs. disabled), and the per-player discount factors δ1,δ2 ∈ {0.8,0.9,0.95,1.0}. Negotiation varies the seller’s reserve value VS and the buyer’s valuation VB (each on a relative scale, with both values ∈ {0.8,1.0,1.2,1.5} scaled by a price order ∈ {100,10000,1,000,000}), the round horizon max_rounds ∈ {1,10,∞}, the information regime, and the communication regime.

University-hackathon target population. A hackathon configuration uses the same GLEE parameterisation as the source population: in addition to the headline parameters released to participants in advance (horizon, information regime, parameter ranges), each game also fixes per-player payoff parameters that the engine uses to evaluate outcomes — the discount factors δ1,δ2 in bargaining and the valuations VS,VB in negotiation. The specific values used at evaluation time were not disclosed to teams in advance, so they could not be hard-coded into agent design. At runtime, each agent always observes its own parameter (its own δ in bargaining, its own valuation in negotiation); the opponent’s parameter is observed only under the complete-information regime, and is private otherwise — exactly as in GLEE. All such parameters are recorded in each game’s config.json and vary across configurations within a stage. We therefore enumerate configurations on the full GLEE parameter set, which yields 10 distinct bargaining and 8 distinct negotiation configurations across the four stages. Table 4 gives the full list.

### B Frontier-LLM tournament model list

The 13 frontier LLMs in the round-robin tournament of Shapira et al. [59], grouped by provider. All 13 receive the same game-facing system prompt; the only variable across source agents is the underlying LLM.

- Table 4: Configurations played at each hackathon stage, enumerated on the full GLEE parameter

set (so per-player discount factors δ1,δ2 that vary within a stage produce distinct configurations). Bargaining columns: M = money to divide; max R = max rounds; info = complete (C) or incom-

plete (I) information; δ1,δ2 = per-player discount factors. Negotiation columns: VS,VB = seller / buyer relative values; price order = scale; max R, info as above (negotiation configurations do not parameterise per-player discounting). All configurations have messages_allowed = true.

Bargaining Negotiation

Stage

M max R info δ1 δ2 VS, VB price order max R info

- 0.8 0.95 — — — —
- 0.8 1.0 — — — 0.95 0.95 — — — —

###### 1 100 12 C

1.0, 1.2 10,000 1 C 0.8, 1.5 10,000 1 I

###### 2 10,000 12 I 0.8 0.8

###### 3 1,000,000 12 I 0.9 0.9 1.0, 1.5 1,000,000 10 I

100 12 I 1.0 1.0 1.2, 1.0 100 10 I 10,000 ∞ I 0.9 0.8 1.0, 1.2 10,000 ∞ I 10,000 12 C 0.8 1.0 1.2, 1.5 10,000 1 C

Final

1,000,000 12 I 1.0 0.8 0.8, 1.5 1,000,000 10 I 1,000,000 ∞ C 0.9 0.9 0.8, 1.5 1,000,000 ∞ C

Provider Model Anthropic Claude 3.5 Sonnet (2024-10-22)

Claude 3.7 Sonnet (2025-02-19) Google Gemini 1.5 Flash

- Gemini 1.5 Pro
- Gemini 2.0 Flash Gemini 2.0 Flash-Lite

Meta Llama-3.3-70B-Instruct

Llama-3.1-405B-Instruct-FP8 Mistral Mistral-Large-Latest OpenAI GPT-4o

GPT-4o-Mini o3-Mini xAI Grok-2 (2024-12-12)

### C Hackathon competition details

This appendix expands on the held-out target population introduced in Section 3. The hackathon ran in December 2025 over four stages: three preliminary stages with progressively richer game configurations, and a final round in which the top six teams (selected by stage-3 payoff) competed for a $2,000 prize pool. Game configurations released at each stage are listed in Appendix A. To control API cost across 34 teams playing thousands of games, we restricted participants to the Gemini 2.5 Flash and Gemini 2.5 Flash-Lite API surface; this choice is what makes the target population orthogonal to the frontier-LLM source on the variation axis (scaffolding instead of underlying LLM).

Released dataset vs. predictive cohort. We distinguish two cuts of the data. The publicly released dataset contains the 23 final-round submissions for which complete code is available for audit. For the predictive evaluation reported in the main text, we additionally retain the decision logs of every (team,stage) agent that appeared at any stage of the competition: teams submitted revised versions between stages, so a single team contributes multiple distinct agents. This yields 91 team-stage agents and 11,341 accept/reject decisions over 4,921 bargaining and negotiation games.

### D Proposal prediction: model details

This appendix documents the model choices specific to the regression task. Aggregate numbers appear in Table 2.

Task and inverse normalisation. For each two-player game in which the target agent plays as proposer, and for each round r≥2, we predict the offer the proposer will submit–equivalently, the offer its next opponent will face. Training is on a scale-normalised target so that configurations with different monetary scales live on the same axis; the dollar amount a stakeholder would see is recovered from the regressor output by a closed-form inverse. For bargaining, the normalised target is the proposer’s own share of the divided sum (self_gain/(self_gain + other_gain) ∈ [0,1]); inverting the normalisation returns the dollar offer to the opponent as (1 − yˆ) · M, where M is the configuration’s total to split. For negotiation, the normalised target is the proposer’s price divided by a configuration-specific scale constant S (the reference price defined per configuration); inverting returns yˆ · S, the dollar price the opponent is being offered. Normalised values can exceed 1 when the proposer’s price sits above the nominal scale. Round 1 is excluded because it has no prior-round state to condition on.

Task-oriented prompt suffix. The reconstructed decision-time prompt used by the responseprediction model ends with {"decision": ", which primes the Observer for an accept/reject token. That suffix is uncorrelated with the proposal regression target. We therefore swap the suffix for a task-matched one that orients the Observer toward the proposer’s own next offer:

- • Bargaining proposal: Offer: {proposer_name}_gain: $ — primes for the proposer’s own dollar gain, which is our target variable for that family.
- • Negotiation proposal: Offer: $ — primes for the proposer’s price amount.

This is the only change at Observer-extraction time. The response-prediction suffix is unchanged from the main model.

Feature stacks. The proposal-prediction comparison in Table 2 contrasts the same two stacks used for response prediction:

- • Game+text features baseline: the proposer-side game-feature schema (round index, horizon, discount factors, per-side valuations, prior-round offers and decisions), the dialogue representation, and the agent-identity indicator.
- • Observer-augmented model: the same stack plus the Observer hidden-state representation. This is the row reported under each Observer model in Table 2.

The primary comparison is the Observer’s marginal contribution on top of the game+text features baseline.

Cohort filters. We include hackathon agents with at least 30 round-≥2 proposer decisions and target standard deviation ≥0.02; bargaining passes 78 agents, negotiation passes 20.

Evaluation. Same protocol as response prediction: cross-population transfer from the frontier-LLM tournament, training rows capped at 3,000 (balanced across source agents), test rows capped at 500 per cell, game-level splits, K∈{0,2,4,8,16}, 5 seeds, TabPFN v2.6 in regressor mode.

Why median. Per-agent R2 in this regression task is heavy-tailed: for some agents, predictions extrapolate to extreme values on a small subset of configurations, driving per-agent R2 to large negative values. Means over agents are dominated by these tail cases, while medians are stable. We therefore report medians in Table 2 and in the primary discussion of proposal-prediction results.

### E Provider replication of the logits-vs-hidden-states gap

This appendix supports the robustness paragraph of Section 7. We test whether the gap between an Observer’s direct logits and its hidden-state representation depends on the choice of provider.

The comparison is run in the cross-population transfer protocol of Section 6 (frontier-LLM source population, hackathon target population, K=16, TabPFN classifier, 5 seeds), so the contrast is purely between two read-outs of the same Observer under the same evaluation setup used for the main results. The cohort matches the main response-prediction evaluation: 72 bargaining and 39 negotiation hackathon agents.

Logits vs. hidden states. Table 5 isolates the effect for the Gemma-2-2B Observer. Logits alone, whether passed through a TabPFN classifier or read directly as p(accept), are far weaker than the Observer hidden-state representation. The game+text features baseline is already strong; adding the logit scalar yields only a marginal improvement, while adding hidden states yields a clearly larger one. Adding both logits and hidden states is a wash relative to hidden states alone. The pattern matches the Section 6 takeaway: the predictive signal lives in the Observer’s representation, not in its direct readout.

- Table 5: Matched comparison of logits vs. hidden states under cross-population transfer at K=16 with TabPFN, mean AUC over 5 seeds (SE over agents × seeds in parentheses; 72 bargaining, 39 negotiation hackathon targets). Logits = p(accept) obtained by renormalising the Observer’s next-token probabilities over the accept/reject verbalisers; hidden state = Observer hidden-state representation averaged over Gemma-2-2B’s upper-stack layer band (relative depth 0.6–0.9).

Input to the TabPFN classifier Bargaining Negotiation

Logits alone .585(.010) .464(.015) Observer hidden states alone .701(.008) .536(.013) Game-state features alone .756(.010) .827(.014) Game-state features + logits .767(.009) .831(.014) Game-state features + Observer hidden states .793(.008) .838(.013) Game-state features + logits + Observer hidden states .791(.008) .835(.013)

Replication across three providers. If the logit-vs-hidden-state gap were an artifact of the hackathon agents’ underlying LLM matching the Observer’s training pipeline, Observers from unrelated providers should fail to reproduce it. They do not. We replicate with Qwen3-1.7B [74] (Alibaba) and Llama-3.2-1B [27] (Meta), neither of which shares a training pipeline or parent company with the hackathon agents’ underlying LLM, and recover the same pattern (Table 6). Game-state features plus Observer hidden states sit in a tight band across providers (bargaining ∈ [0.76,0.79], negotiation ∈ [0.84,0.86]). Direct p(accept) AUC is more variable across providers, especially in negotiation: Gemma and Llama remain near chance (≤ 0.45) while Qwen3 reaches 0.717, an outlier we report transparently rather than aggregate over. The hidden-state-augmented predictions are consistent across providers; the direct readout is not.

- Table 6: Logits vs. hidden states across Observer providers, cross-population transfer at K=16. Logits columns report direct p(accept) AUC of the Observer (no classifier fit; values can fall below chance because the Observer’s preferred direction is not enforced to align with acceptance). Hiddenstate columns use game-state features plus the Observer hidden-state representation (averaged over the upper-stack layer band, relative depth 0.6–0.9) under the same TabPFN classifier as Table 5.

Logits alone Game + Hidden states

Observer Provider Params Barg Nego Barg Nego Gemma-2-2B Google 2.6B .548 .403 .793 .838 Qwen3-1.7B Alibaba 1.7B .607 .717 .785 .863 Llama-3.2-1B Meta 1.2B .577 .446 .763 .837

### F Additional experimental details

Hardware. All GPU experiments ran on an internal cluster with up to 2× NVIDIA RTX A6000 (48GB each); individual jobs used a single GPU at a time. CPU-only steps (game-feature extraction and dialogue encoding with all-MiniLM-L6-v2) ran on the same hosts. The direct LLM-asPredictor baseline (Section 6 and Appendix H) is API-only and consumes no local GPU time.

Observer feature extraction. Decision-time hidden states for the three Observers (Gemma-2-

- 2B, Qwen3-1.7B, Llama-3.2-1B) were extracted with TransformerLens run_with_cache over the reconstructed prompt for ≈ 67K games (bargaining and negotiation, both populations; ≈ 200K decisions). All upper-stack layers used in the layer sweep of Figure 3 are cached in a single forward pass per game; the main Observer hidden-state representation is the average over each model’s upperstack layer band (relative depth 0.6–0.9). Three suffix variants are extracted per Observer—response prediction ({"decision": "), bargaining proposal, and negotiation proposal—because the suffix re-orients the Observer toward the task (Appendix D). The same forward pass also caches the perdecision next-token logits used in the logits-vs-hidden-states comparison of Appendix E, so that comparison adds no extraction cost. Cumulative Observer extraction: ≈80A6000 GPU-hours.

Tabular evaluation. TabPFN v2.6 was run with default settings; each evaluation cell uses up to

- 3,000 source-balanced training rows together with the target’s K-game decisions and at most 500 test rows, taking ≈10–30s on a single A6000. The reported tabular budget covers all such cells across: the cross-population response and proposal tables (Tables 2–2); the layer sweep over each Observer’s full stack (Figure 3), which dominates this budget because every layer is evaluated separately rather than averaged over the upper-stack band; and the logits-vs-hidden-states matched comparison and its three-provider replication (Appendix E). Cumulative tabular evaluation: ≈60A6000 GPU-hours.

LLM-as-Predictor (API). The LLM-as-Predictor numbers in Tables 2–2 use Gemini 2.5 Flash with thinking_budget=0: 65 bargaining and 33 negotiation hackathon targets × K ∈ {0,2,4,8,16} × 5 seeds × up to 30 test decisions per cell. Because no thinking tokens are generated, the percall cost is much lower than the pilot. The thinking-on pilot in Appendix H used 4,010 calls with thinking_budget=2000 at a cost of ≈ $31; we did not scale this configuration to the full table, which would have cost an estimated ≈$235.

Total. The experiments reported in the paper consumed ≈ 140A6000 GPU-hours cumulatively (extraction + tabular evaluation), plus the Gemini API budget above. Including preliminary and discarded runs—alternative Observer layers and pooling strategies, predictor sweeps that did not converge under the cross-population protocol, abandoned feature stacks, and exploratory hackathononly protocols not reported in the final paper—the broader project consumed roughly ≈400GPUhours.

Model hyperparameters. TabPFN v2.6 with default settings (no tuning). Observer hidden-state representations are averaged over each model’s upper-stack layer band (relative depth 0.6–0.9); layer counts per Observer are Gemma-2-2B (26), Qwen3-1.7B (28), and Llama-3.2-1B (16). Seeds {0,1,2,3,4}. Full feature-block specifications (game-feature schemas per family, dialogue construction, PCA dimensions, identity-column construction) are in Appendix G.

Observer prompt suffixes. The suffix appended after the dialogue history orients the Observer toward the decision to predict: {"decision": " for response prediction (the original GLEE accept/reject format), Offer: $ for proposal prediction in negotiation, and Offer: {proposer_name}_gain: $ for proposal prediction in bargaining (proposer’s own dollar gain).

Game-level split. For each held-out target agent, at each K, we randomly select K of the agent’s games as the K-shot pool (all round-by-round decisions within those games enter the adaptation rows); the remaining games form the test set. Splitting at the game level ensures that rounds from the same game never appear on both sides of the split, and makes explicit that K counts whole observed games–each with multiple decisions–rather than individual decisions.

Error bars. Throughout the paper, ± values report standard errors of the per-(agent,seed) AUC or R2 values for each cell (σ/

##### √

N with N = Ntargets × 5). The Observer-by-layer figure (Figure 3) reports paired 95% confidence intervals as 1.96 · SEM on the per-(agent,seed) deltas of full − baseline, anchored at the baseline mean.

### G Game+text features baseline: feature specifications

This appendix specifies the three feature blocks of the game+text features baseline (Section 4): gamestate features, the dialogue representation, and the agent-identity indicator. The Observer hidden-state representation, appended in the Observer-augmented model, is documented in Appendix F.

Current Offer & Message

Situation Features

Round t-4

Round t-2 Round t-3

Round t-1

|1|2| | |3| |4| |5|6| |7|8| | |9|10| |11| |12| |13|14| |15|16| |17|18| |19|20| |21| |22|23| |24| |25| |26| |27|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

GameHorizon Observer

Product Valuation Textual Message Offer

| |
|---|

| |
|---|

| |
|---|

Game Features Text Representation Observer Representation

Figure 4: Schematic of the multimodal tabular row at a single decision point. The row concatenates the three feature modalities of Section 4: game-state features (red), the dialogue representation produced by the sentence encoder (blue), and the Observer hidden-state representation of the current decision-time state (purple). Game-state features are divided into configuration-level situation features (e.g., game horizon, product valuation) and per-round entries summarizing the last few rounds and the current offer; the dialogue representation contributes per-round textual entries. Cell counts are illustrative; actual modality dimensions and game-feature columns differ by game family (bargaining vs. negotiation).

Game-state features – bargaining (24 columns). At each responder decision at round r we extract a fixed-schema vector from the publicly observable game state, with N=5 prior rounds of history. Missing entries (history slots before round 1, or fields that do not apply in a particular configuration) are encoded as NaN; TabPFN handles NaN natively. The columns are:

- • Configuration (8): round, max_rounds, round_frac (= r/max_rounds), money (the amount to divide M), delta_1 and delta_2 (the two players’ per-round discount factors), messages (binary: free-text exchange allowed in this game), complete_info (binary: parameters shared with both players or private).
- • Current offer (5): offer_frac (=responder_gain/(proposer_gain+responder_gain)), responder_gain and proposer_gain (the split in absolute units), inflation_loss_1 (=1−delta_1r−1) and inflation_loss_2 (cumulative discounting incurred up to round r).
- • History (10): for h=1,...,5, prevh_offer_frac and prevh_decision: the proposer’s split and responder’s accept/reject at round r−h.
- • Family indicator (1): family_idx (constant within bargaining; included for cross-family compatibility of the schema).

#### Game-state features – negotiation (25 columns). At each responder decision we extract:

- • Configuration (8): round, max_rounds, round_frac, sv (seller’s reservation valuation), bv (buyer’s reservation valuation), product_price_order (typical-price scale used to order configurations), messages, complete_info.
- • Outside-option references (2): seller_outside (= sv · product_price_order) and buyer_outside (= bv · product_price_order): the absolute payoffs of the outside option for each player.
- • Current offer (4): price (the proposed sale price), offer_frac (= price/product_price_order), offer_vs_buyer_outside (= price/buyer_outside, the relative cost compared to the buyer’s outside option), rounds_remaining.

- • History (10): prevh_offer_frac and prevh_decision for h=1,...,5.
- • Family indicator (1): family_idx.

Dialogue representation. For each responder decision at round r, we collect every message field exchanged within round r in the game log (the proposer’s message accompanying the offer, plus any responder-side message in the same round), concatenate them into a single string with singlespace separators, and encode the string with the sentence-transformers/all-MiniLM-L6-v2 sentence encoder, yielding a 384-dimensional vector. When the round contains no messages we use the placeholder "Round r" so the representation is always defined. The 384-dimensional vectors are PCA-reduced to 5 dimensions; the projection is fit on the training pool of each evaluation cell and applied unchanged to the test pool. The choice of 5 is intentionally low: a value such as 30 would be a more natural default for MiniLM, but 5 performed slightly better for the game+text features baseline at K=0 in pilot runs, and we keep the same value for the Observer-augmented model so that any difference between the two is not attributable to dialogue-representation capacity.

Agent-identity indicator. Let S denote the source agents in the training pool and t the held-out target agent. The agent-identity indicator is a one-hot vector of dimension |S| + 1 over S ∪ {t}, with a single 1 marking the agent whose decision is being predicted. At K=0, every training row lies in S (so the t-column is always zero in training) and every test row activates the t-column; the column therefore separates train from test deterministically and adds no within-test signal. At K>0, the K adaptation games of t enter the training pool and activate the t-column there, which gives the tabular predictor a within-target anchor when it predicts on the held-out test games of t.

PCA fitting. All PCA projections in the model (dialogue representation, Observer hidden states) are fit on the training pool of each evaluation cell and then applied unchanged to the test pool. No test-row information enters PCA fitting at any K.

### H Thinking-budget pilot for the LLM-as-Predictor baseline

The main LLM-as-Predictor table uses a full run with thinking_budget=0: 65 bargaining and 33 negotiation hackathon target agents, K ∈ {0,2,4,8,16}, and up to 30 test decisions per cell, matching the seed protocol of Section 5. To check whether this underpowers the direct LLM-as-Predictor baseline, we ran a stratified pilot on 5 target agents per family with thinking_budget=2000. The pilot cost was approximately $31 for 4,010 API calls, with thinking tokens accounting for most of the cost; scaling the same configuration to the full table was estimated at roughly $235, so we did not run it as a main experiment.

- Table 7: Thinking-budget pilot on the same 5 target agents per family and matched cells. The rightmost column reports our cross-population response-prediction model (game+text features + Observer hidden-state representation) restricted to the pilot cells. Family K LLM-as-Predictor (budget 0) LLM-as-Predictor (budget 2000) Observer-augmented model

Barg 0 .628 .656 .672 Barg 8 .792 .860 .857 Barg 16 .733 .806 .881

Nego 0 .492 .700 .623 Nego 8 .645 .636 .771 Nego 16 .720 .642 .750

The result is mixed rather than a monotone strengthening of the LLM-as-Predictor. Thinking improves some low-K cells, including negotiation at K=0, but hurts negotiation at high K on this subset. The pilot does not overturn the main comparison: a stronger direct LLM-as-Predictor can improve isolated cells, especially at low K, but the Observer-augmented model remains stronger at K=16 in both families.

### I Broader impacts

Predicting unfamiliar agents’ decisions has dual-use implications. Constructive uses include mechanism design, planning under counterpart uncertainty, and prediction-aware agents in languagemediated commerce that adapt to a partner’s behaviour without sharing private state. The same capability could be misused for adversarial counterpart modelling–e.g., extracting concession patterns to exploit them–deployed without disclosure to the counterparty. As predictive models of language-based agents become more accurate, transparency norms about whether counterparts are being modelled, analogous to consumer-facing disclosures, become an important complement to the technical work.

